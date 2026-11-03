"""Host entry for the gfx942 FP8 block-scale MoE (prefill path, aiter fused_moe calling convention)."""

import os

import flydsl.compiler as flyc
import torch

from . import kernels as _k
from . import kernels_m64 as _k64

FP8 = torch.float8_e4m3fnuz
BLOCK_M = _k.BM
# Short-K path: experts padded to 64 rows only; GEMM tiles stay 256 rows with one 64..192-row tail per expert.
COMPACT_BLOCK = 64
# Waves per CTA of the fused 1-stage kernel: 8 (one CTA per CU) measured faster than 4 (two per CU).
FUSED_WAVES = int(os.environ.get("PYHIP_MOE_FUSED_WAVES", "8"))
_compiled = {}


def _run(key, launcher, *args):
    compiled = _compiled.get(key)
    if compiled is None:
        # flyc.compile also launches this first call.
        _compiled[key] = flyc.compile(launcher, *args)
    else:
        compiled(*args)


def quantize_input(hidden_states):
    """BF16 [T, H] -> (FP8 [T, H], scale [H / 128, T]) with aiter's per-1x128 dynamic quantization."""
    from aiter import QuantType
    from aiter.fused_moe import get_quant

    x8, scale = get_quant(QuantType.per_1x128)(hidden_states, quant_dtype=FP8, transpose_scale=True)
    t, h = hidden_states.shape
    return x8, torch.as_strided(scale, (h // 128, t), (t, 1))


def moe_sorting(topk_ids, topk_weights, num_experts, hidden, expert_mask=None, block_size=BLOCK_M):
    """aiter moe_sorting (experts padded to block_size rows): (sorted_ids, sorted_weights, sorted_expert_ids,
    num_valid_ids).

    With expert_mask (EP), only routes to masked experts are kept and block expert ids are local.
    """
    from aiter.fused_moe import moe_sorting as _sort

    sid, sw, seid, nv, _ = _sort(topk_ids, topk_weights, num_experts, hidden, torch.bfloat16,
                                 block_size=block_size, expert_mask=expert_mask, accumulate=False)
    return sid, sw, seid, nv


def _tiles(seid, nv, experts, block, stream):
    """Compact GEMM tiles over the sorted blocks (device-side, no sync): ([cap, 2] int32, count [1], cap)."""
    nb = seid.numel()
    per = BLOCK_M // block
    cap = (nb + per - 1) // per + (experts if per > 1 else 0)
    tiles = torch.empty(2 * cap, dtype=torch.int32, device=seid.device)
    count = torch.empty(1, dtype=torch.int32, device=seid.device)
    _run(("tiles", seid.device, experts, block), _k.launch_tiles, seid, nv, tiles, count, nb, cap, experts, block,
         stream)
    return tiles, count, cap


def fused_moe_fp8_blockscale_942(hidden_states, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale,
                                 a1_scale=None, out=None, expert_mask=None):
    """out = sum_k w[t, k] * W2[e] @ (silu(W13g[e] @ x) * W13u[e] @ x) with 128x128 FP8 block scales.

    hidden_states: [T, H] BF16, or FP8 E4M3FNUZ with a1_scale whose bytes are [H / 128, T] (aiter's
    transposed per-1x128 activation scale). w13: [E, 2 I, H] FP8 (gate rows first), w2: [E, H, I] FP8, both
    aiter shuffle_weight(16, 16). w13_scale: [E, 2 I / 128, H / 128], w2_scale: [E, H / 128, I / 128] FP32.
    topk_ids / topk_weights: [T, topk] int32 / FP32. I % 128 == 0, H % 256 == 0.
    expert_mask: optional int32 [E_global] marking this rank's experts (EP without all-to-all): weights are
    the local experts in mask order, out sums only the local routes.
    """
    tokens, hidden = hidden_states.shape
    experts, inter2, k = w13.shape
    inter = inter2 // 2
    if k != hidden or w2.shape != (experts, hidden, inter) or inter % 128 or hidden % 256:
        raise ValueError("needs w13 [E, 2I, H], w2 [E, H, I], I % 128 == 0, H % 256 == 0")
    if w13_scale.numel() != experts * (inter2 // 128) * (hidden // 128) or \
            w2_scale.numel() != experts * (hidden // 128) * (inter // 128):
        raise ValueError("block scale shapes do not match")
    if out is None:
        out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=hidden_states.device)
    if tokens == 0:
        return out
    if hidden_states.dtype == FP8:
        x8 = hidden_states
        xs = torch.as_strided(a1_scale, (hidden // 128, tokens), (tokens, 1))
    else:
        x8, xs = quantize_input(hidden_states)
    num_experts = experts if expert_mask is None else expert_mask.numel()
    return _moe(x8, xs, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale, out, expert_mask, num_experts)


def _moe(x8, xs, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale, out, expert_mask, num_experts):
    tokens, hidden = x8.shape
    inter = w2.shape[2]
    topk = topk_ids.shape[1]
    mode = os.environ.get("PYHIP_MOE_DOWN", "direct") if inter == 256 else "dense"
    # fused: 1-stage M64 kernel (gemm1 + requant + down per 64-row block; opt-in, slower on MI300X, see readme).
    fused = mode == "fused" and hidden % 1024 == 0
    direct = mode == "direct" or (mode == "fused" and not fused)
    block = COMPACT_BLOCK if direct or fused else BLOCK_M
    if (tokens * topk + num_experts * block) * hidden * 2 >= 1 << 32:
        # Sorted rows x hidden must stay inside one 4 GiB buffer: run token halves.
        half = tokens // 2
        for lo, hi in ((0, half), (half, tokens)):
            _moe(x8[lo:hi], xs[:, lo:hi].contiguous(), w13, w2, topk_weights[lo:hi], topk_ids[lo:hi], w13_scale,
                 w2_scale, out[lo:hi], expert_mask, num_experts)
        return out
    sid, sw, seid, nv = moe_sorting(topk_ids, topk_weights, num_experts, hidden, expert_mask, block)
    rows, blocks = sid.numel(), seid.numel()
    device = x8.device
    stream = torch.cuda.current_stream(device)
    a2, a2s, y = _workspace(device, rows, inter, hidden)
    # Routes without a sorted row (other ranks' experts) stay -1 and read as zero rows in the sum.
    inv = torch.full((tokens, topk), -1, dtype=torch.int32, device=device)
    u8 = torch.uint8
    if fused:
        _run(("moe64", device, inter, hidden, FUSED_WAVES), _k64.launch_moe64,
             x8.view(u8).view(-1), w13.view(u8).view(-1), xs.reshape(-1), w13_scale.reshape(-1),
             w2.view(u8).view(-1), w2_scale.reshape(-1), sid, seid, sw, nv, y, tokens, rows, blocks, inter, hidden,
             hidden, FUSED_WAVES, stream)
        _run(("inverse", device, topk), _k.launch_inverse,
             sid, nv, inv.view(-1), tokens, rows, (rows + _k.SUM_THREADS - 1) // _k.SUM_THREADS, topk, stream)
        _top_k_sum(y, inv, out, tokens, rows, hidden, topk, stream)
        return out
    tiles, count, cap = _tiles(seid, nv, w13.shape[0], block, stream)
    _run(("gate_up", device, inter, hidden), _k.launch_gate_up,
         x8.view(u8).view(-1), w13.view(u8).view(-1), xs.reshape(-1), w13_scale.reshape(-1), sid, tiles, count, sw,
         a2, a2s, tokens, rows, cap * (inter // 128), inter, hidden, stream)
    _run(("inverse", device, topk), _k.launch_inverse,
         sid, nv, inv.view(-1), tokens, rows, (rows + _k.SUM_THREADS - 1) // _k.SUM_THREADS, topk, stream)
    if direct:
        _down_and_sum(a2, w2.view(u8).view(-1), a2s, w2_scale.reshape(-1), tiles, count, cap, sid, y, inv, out,
                      tokens, rows, hidden, inter, topk, stream)
        return out
    _run(("down", device, hidden, inter), _k.launch_down,
         a2, w2.view(u8).view(-1), a2s, w2_scale.reshape(-1), seid, nv, y, rows, blocks * (hidden // _k.BN),
         hidden, inter, stream)
    _run(("sum", device, hidden, hidden, topk), _k.launch_sum,
         y, inv.view(-1), out.view(-1), tokens, rows, 0, tokens, hidden, hidden, topk, stream)
    return out


_side = {}


def _top_k_sum(y, inv, out, tokens, rows, hidden, topk, stream):
    """out[t] = sum of t's route rows of Y (the fused path's tail; probes swap in sum + all-reduce)."""
    _run(("sum", y.device, hidden, hidden, topk), _k.launch_sum,
         y, inv.view(-1), out.view(-1), tokens, rows, 0, tokens, hidden, hidden, topk, stream)


def _down_and_sum(a2, w2, a2s, w2s, tiles, count, cap, sid, y, inv, out, tokens, rows, hidden, inter, topk,
                  stream):
    """Short-K down and top-k sum, optionally in column slices with slice s's sum on a side stream while slice
    s + 1's down runs (PYHIP_MOE_SPLIT; measured slower on MI300X, default 1)."""
    device = y.device
    groups = hidden // _k.DOWN_COLS
    split = int(os.environ.get("PYHIP_MOE_SPLIT", "1"))
    split = split if split > 0 and groups % split == 0 else 1
    per, cols = groups // split, hidden // split
    side = stream
    if split > 1:
        side = _side.get(device)
        if side is None:
            side = _side[device] = torch.cuda.Stream(device)
    for part in range(split):
        _run(("down_direct", device, hidden, inter, per), _k.launch_down_direct,
             a2, w2, a2s, w2s, tiles, count, sid, y, rows, tokens, part * per, cap * per, hidden, inter, per, stream)
        if side is not stream:
            ready = torch.cuda.Event()
            ready.record(stream)
            side.wait_event(ready)
        _run(("sum", device, hidden, cols, topk), _k.launch_sum,
             y, inv.view(-1), out.view(-1), tokens, rows, part * cols, tokens, hidden, cols, topk, side)
    if side is not stream:
        done = torch.cuda.Event()
        done.record(side)
        stream.wait_event(done)
        out.record_stream(side)
        inv.record_stream(side)


_buffers = {}


def _workspace(device, rows, inter, hidden):
    """Flat A2 (FP8 bytes), A2S (FP32) and Y (BF16) views for `rows` sorted rows; grown, never shrunk.

    Kernels on one stream finish with a buffer before the next call reuses it. Grow before CUDA graph
    capture (warm up with the largest token count) so captured graphs keep valid addresses.
    """
    need = (rows * inter, (inter // 128) * rows, rows * hidden)
    have = _buffers.get(device)
    if have is None or any(h.numel() < n for h, n in zip(have, need)):
        have = _buffers[device] = (torch.empty(need[0], dtype=torch.uint8, device=device),
                                   torch.empty(need[1], dtype=torch.float32, device=device),
                                   torch.empty(need[2], dtype=torch.bfloat16, device=device))
    return tuple(h[:n] for h, n in zip(have, need))
