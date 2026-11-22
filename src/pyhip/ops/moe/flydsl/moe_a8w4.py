# SPDX-License-Identifier: MIT

"""A8W4 MoE for gfx942: FP8 per-token activations x MXFP4 weights on the FlyDSL two-stage kernels.

``prepare_a8w4_weights`` converts standard MXFP4 weights once (at load time) into the private
layout of ``moe_gemm_2stage/a8w4_common.py``; ``fused_moe_a8w4`` runs the whole MoE:
per-row FP8 quantization of the input, Opus ``moe_sorting``, the A8W4 Gate/Up kernel
(SiLU(gate) * up -> f32), per-row FP8 quantization of the intermediate, the A8W4 Down
kernel (routing weight applied) and ``sorted_sum``. Every token count uses this one path.
"""

import functools
from dataclasses import dataclass

import torch

MAX_DELTA = 11
_PAD_I = 64
# Token counts up to this use the decode kernels (no sorting, one workgroup per route / token).
DECODE_MAX_M = 32
# The decode path quantizes activations in separate row kernels instead of inside every workgroup:
# Down's intermediate rows from DECODE_PREQUANT_MIN_M tokens on, Gate/Up's token rows from
# DECODE_GATEUP_PREQUANT_MIN_WGS Gate/Up workgroups (M * topk * I / 16) on.
DECODE_PREQUANT_MIN_M = 4
DECODE_GATEUP_PREQUANT_MIN_WGS = 2560
DECODE_DOWN_PQ_WAVES = 3

# Kernels launched by the most recent ``fused_moe_a8w4`` call, in order. Kernels of the A8W4 op are
# named ``a8w4_*`` (FP8 activations) or ``a16w4_*`` (BF16 activations); tests pick the reference by it.
last_path = []


@dataclass(frozen=True)
class A8W4Weights:
    """Prepared weights; tensors are flat uint8 / float32 per expert (see ``a8w4_common``)."""

    w1: torch.Tensor
    w1_delta: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_delta: torch.Tensor
    w2_scale: torch.Tensor
    experts: int
    hidden: int
    inter: int
    inter_padded: int


def _codes(scale, shape):
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise TypeError(f"MXFP4 scales must be float8_e8m0fnu or uint8, got {scale.dtype}")
    codes = scale.view(torch.uint8).reshape(shape)
    if bool((codes == 255).any()):
        raise ValueError("E8M0 scale 0xFF (NaN) is not supported")
    return codes


def _packed(weight, shape):
    if weight.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise TypeError(f"MXFP4 weights must be float4_e2m1fn_x2 or uint8, got {weight.dtype}")
    if tuple(weight.shape) != shape:
        raise ValueError(f"expected MXFP4 weight shape {shape}, got {tuple(weight.shape)}")
    return weight.view(torch.uint8)


def _to_layout(q, codes):
    """[E, N, K/2] FP4 bytes + [E, N, K/32] E8M0 codes -> (q, delta, channel scale) layouts."""
    E, N, half_k = q.shape
    K = half_k * 2
    KB = K // 64
    KQ = (KB + 3) // 4
    codes = codes.to(torch.int32)
    cmax = codes.amax(dim=-1)
    diff = cmax[..., None] - codes
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).view(E, N, K // 32, 32)
    nib = nib.masked_fill(nib == 8, 0).masked_fill((diff > MAX_DELTA)[..., None], 0)
    nib = nib.view(E, N, K // 8, 8)
    out_q = (nib[..., :4] | (nib[..., 4:] << 4)).view(E, N // 16, 16, KB, 4, 8)
    out_q = out_q.permute(0, 1, 3, 4, 2, 5).reshape(E, -1)
    delta = (diff.clamp(max=MAX_DELTA) * 8).to(torch.uint8).view(E, N, KB, 2)
    delta = torch.nn.functional.pad(delta, (0, 0, 0, KQ * 4 - KB))
    out_d = delta.view(E, N // 16, 16, KQ, 4, 2).permute(0, 1, 3, 5, 2, 4).reshape(E, -1)
    scale = torch.ldexp(torch.ones_like(cmax, dtype=torch.float64), cmax - 132).float()
    return out_q, out_d, scale


def _convert(q, codes, chunk=32):
    parts = [_to_layout(q[e:e + chunk], codes[e:e + chunk]) for e in range(0, q.shape[0], chunk)]
    return tuple(torch.cat(p).contiguous() for p in zip(*parts))


def prepare_a8w4_weights(w1, w1_scale, w2, w2_scale):
    """Convert standard MXFP4 MoE weights to the A8W4 private layout (run once at load time).

    Inputs: ``w1`` [E, 2I, H/2] FP4x2 (``torch.float4_e2m1fn_x2`` or uint8; rows ``[0, I)`` gate,
    ``[I, 2I)`` up; low nibble = even K), ``w1_scale`` [E, 2I, H/32] E8M0 (``float8_e8m0fnu`` or
    uint8), ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32]. Requires ``H % 512 == 0`` and
    ``I % 32 == 0``. I is zero-padded to a multiple of 64 (160 -> 192): padded Gate/Up rows and
    Down columns are zero, so results equal the unpadded MoE.

    Precision: each 32-K group is decoded to FP8 relative to its row's largest E8M0 code; groups
    more than 11 codes (2**11) below that maximum are treated as zero (``MAX_DELTA``).
    One result serves every token count of ``fused_moe_a8w4``.
    """
    E, n1, half_h = w1.shape
    I, H = n1 // 2, half_h * 2
    if n1 % 2 or H % 512 or I % 32:
        raise ValueError(f"unsupported shape: H={H} (needs % 512), I={I} (needs % 32)")
    q1 = _packed(w1, (E, 2 * I, H // 2))
    q2 = _packed(w2, (E, H, I // 2))
    c1 = _codes(w1_scale, (E, 2 * I, H // 32))
    c2 = _codes(w2_scale, (E, H, I // 32))
    ip = -(-I // _PAD_I) * _PAD_I
    if ip != I:
        pad = ip - I
        q1 = torch.cat([q1[:, :I], q1.new_zeros(E, pad, H // 2), q1[:, I:], q1.new_zeros(E, pad, H // 2)], 1)
        c1 = torch.cat([c1[:, :I], c1.new_full((E, pad, H // 32), 127), c1[:, I:],
                        c1.new_full((E, pad, H // 32), 127)], 1)
        q2 = torch.nn.functional.pad(q2, (0, pad // 2))
        c2 = torch.nn.functional.pad(c2, (0, pad // 32))
    w1q, w1d, w1s = _convert(q1.contiguous(), c1.contiguous())
    w2q, w2d, w2s = _convert(q2.contiguous(), c2.contiguous())
    return A8W4Weights(w1q, w1d, w1s, w2q, w2d, w2s, E, H, I, ip)


@functools.cache
def _gateup_kernel(N, K, topk, block_m, block_n, tile_k, experts):
    from .moe_gemm_2stage.a8w4_gemm1 import compile_a8w4_gemm1

    return compile_a8w4_gemm1(N=N, K=K, TOPK=topk, BLOCK_TILE_SIZE_M=block_m,
                              BLOCK_TILE_SIZE_N=block_n, TILE_K=tile_k, E=experts)


@functools.cache
def _down_kernel(N, K, topk, block_m, experts):
    from .moe_gemm_2stage.a8w4_gemm2 import compile_a8w4_gemm2

    return compile_a8w4_gemm2(N=N, K=K, TOPK=topk, BLOCK_TILE_SIZE_M=block_m, E=experts)


@functools.cache
def _decode_kernels(N1, H, I, topk, gateup_pq, down_pq):
    from .moe_gemm_2stage.a8w4_decode import (compile_a8w4_down_decode, compile_a8w4_down_decode_pq,
                                              compile_a8w4_gateup_decode)

    gateup = compile_a8w4_gateup_decode(N=N1, K=H, TOPK=topk, PREQUANT=gateup_pq)
    if down_pq:
        # More tokens: 16-byte loads of slot pairs, DECODE_DOWN_PQ_WAVES waves with pairs in rounds.
        return gateup, compile_a8w4_down_decode_pq(N=H, K=I, TOPK=topk,
                                                   WAVES=min((topk + 1) // 2, DECODE_DOWN_PQ_WAVES))
    # Few tokens (latency bound): one wave per routing slot.
    return gateup, compile_a8w4_down_decode(N=H, K=I, TOPK=topk, WAVES=topk if topk <= 16 else 4)


def _fused_moe_decode(hidden_states, weights, topk_weight, topk_ids, output):
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = weights.inter_padded
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
    # Few tokens: each workgroup quantizes its own activation rows; more tokens: quantize once.
    gateup_pq = M * topk * (I // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS
    # The pre-quantized Down addresses all experts' weights with 32-bit byte offsets.
    down_pq = M >= DECODE_PREQUANT_MIN_M and weights.w2.numel() * weights.w2.element_size() < 2**32
    gateup, down = _decode_kernels(2 * I, H, I, topk, gateup_pq, down_pq)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    if gateup_pq:
        xq, xs = _quantize_decode(hidden_states, M, H, True)
    else:
        xq, xs = hidden_states, topk_weight
    _launch("a8w4_gateup_decode" + ("_pq" if gateup_pq else ""), gateup, xq, xs, topk_ids, weights.w1,
            weights.w1_delta, weights.w1_scale, mid, M)
    if down_pq:
        mq, ms = _quantize_decode(mid, M * topk, I, False)
    else:
        mq, ms = mid, topk_weight
    _launch("a8w4_down_decode" + ("_pq" if down_pq else ""), down, mq, ms, topk_ids, topk_weight, weights.w2,
            weights.w2_delta, weights.w2_scale, output, M)
    return output


@functools.cache
def _row_quant_decode(cols, bf16):
    from .moe_gemm_2stage.a8w4_decode import compile_a8w4_row_quant

    return compile_a8w4_row_quant(COLS=cols, BF16=bf16)


def _quantize_decode(x, rows, cols, bf16):
    """``_quantize`` with the decode path's vectorized kernel (same arithmetic, bit-identical)."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _launch("a8w4_row_quant_decode", _row_quant_decode(cols, bf16), x, q, scale, rows)
    return q, scale


def _quantize(x, rows, cols, dtype):
    from .moe_gemm_2stage.a8w4_quant import compile_row_quant_fp8

    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _launch("a8w4_row_quant_fp8", compile_row_quant_fp8(cols, dtype), x, q, scale, rows)
    return q, scale


def _arg(t):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .moe_gemm_2stage.common import _ptr

    if t.dtype == torch.uint8:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())
    return _ptr(t)


def _launch(name, kernel, *args):
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    _run_compiled(kernel, *(_arg(a) if isinstance(a, torch.Tensor) else a for a in args),
                  torch.cuda.current_stream())
    last_path.append(name)


def fused_moe_a8w4(hidden_states, weights, topk_weight, topk_ids, *, output=None,
                   block_m=64, block_n=128, tile_k=256):
    """A8W4 MoE forward: BF16 ``hidden_states`` [M, H] -> BF16 [M, H].

    ``weights`` comes from ``prepare_a8w4_weights``; ``topk_weight`` float32 [M, topk] is applied
    in Down; ``topk_ids`` int32 [M, topk] in ``[0, E)`` (a fused shared expert is an ordinary
    expert id). The input and the f32 intermediate SiLU(gate) * up are quantized per row to FP8
    E4M3FNUZ with ``scale = amax / 240`` and ``q = RNE(x / scale)``. SiLU only, no expert mask /
    bias; ``M * topk * I_padded * 4 < 2**31``. Issues no host synchronization. ``block_m`` is the
    sorting block and M tile of both kernels (``block_m * I_padded % 4096 == 0``, e.g. 64 or 128);
    ``block_n`` / ``tile_k`` are the Gate/Up N tile (128 or 256, dividing 2 * padded I into whole
    gate/up halves) and K tile (128 or 256).
    """
    from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

    from .moe_gemm_2stage import sorted_sum

    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise TypeError("hidden_states must be a 2-D bfloat16 tensor")
    M, H = hidden_states.shape
    if H != weights.hidden:
        raise ValueError(f"hidden size {H} does not match the weights ({weights.hidden})")
    topk = topk_ids.shape[1]
    E, I = weights.experts, weights.inter_padded
    if output is None:
        output = torch.empty_like(hidden_states)
    last_path.clear()
    if M <= DECODE_MAX_M:
        return _fused_moe_decode(hidden_states, weights, topk_weight, topk_ids, output)

    sorted_ids, sorted_weights, sorted_experts, num_valid, _, _, loc_ids = moe_sorting(
        topk_ids, topk_weight, E, H, hidden_states.dtype, block_m, accumulate=False,
        output_aux=AUX_SORT_OPUS)
    last_path.append("moe_sorting_opus")
    xq, xs = _quantize(hidden_states, M, H, "bf16")
    grid = sorted_experts.numel()
    routing = (sorted_ids, sorted_weights, sorted_experts, num_valid)

    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    _launch(f"a8w4_gateup_prefill_bm{block_m}_bn{block_n}_tk{tile_k}",
            _gateup_kernel(2 * I, H, topk, block_m, block_n, tile_k, E), xq, weights.w1,
            weights.w1_delta, mid, *routing, weights.w1_scale, xs, M, grid)
    mq, ms = _quantize(mid, M * topk, I, "f32")
    routes = torch.empty((grid * block_m, H), dtype=torch.bfloat16, device=hidden_states.device)
    _launch(f"a8w4_down_prefill_bm{block_m}", _down_kernel(H, I, topk, block_m, E), mq,
            weights.w2, weights.w2_delta, routes, *routing, weights.w2_scale, ms, M, grid)
    sorted_sum(topk, H)(loc_ids, routes, output, M)
    last_path.append("sorted_sum")
    return output
