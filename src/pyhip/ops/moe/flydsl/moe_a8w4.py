# SPDX-License-Identifier: MIT

"""A8W4 MoE for gfx942: FP8 per-token activations x MXFP4 weights on the FlyDSL two-stage kernels.

``prepare_a8w4_weights`` converts standard MXFP4 weights once (at load time) into the private
layout of ``moe_gemm_2stage/a8w4_common.py``; ``fused_moe_a8w4`` runs the whole MoE:
per-row FP8 quantization of the input, Opus ``moe_sorting``, the A8W4 Gate/Up kernel
(SiLU(gate) * up -> f32), per-row FP8 quantization of the intermediate, the A8W4 Down
kernel (routing weight applied) and ``sorted_sum``. Token counts up to ``DECODE_MAX_M`` skip the
sorting and use the decode kernels of ``moe_gemm_2stage/a8w4_decode.py`` (Gate/Up per route, Down
per token summing its routes in the workgroup and writing BF16; activations quantized inside the
kernels or by a row kernel). Larger token counts with at most ``STREAM_MAX_ROWS`` rows per expert
(M * topk / E) use the weight-streaming kernels of ``moe_gemm_2stage/a8w4_stream.py`` (small sorting
blocks; with larger blocks the Gate/Up kernel also quantizes the intermediate; ``a8w4_route_sum``
does the ``sorted_sum`` step), more rows per expert the prefill kernels.
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
# From DECODE_DEDUP_MIN_M tokens on, for hidden sizes from DECODE_DEDUP_MIN_H on and padded I a multiple of 128,
# the decode Down reads each expert's weights once per DECODE_DEDUP_GROUP of its routes and adds into the
# zeroed output with BF16x2 atomics (``a8w4_decode_dedup``; DECODE_DEDUP_DOWN_BLOCKS 16-channel blocks per
# wave). While tokens times padded I stay below DECODE_DEDUP_GATEUP_MIN_MI the per-route Gate/Up writes the plan
# and zeroes the output and Down quantizes its intermediate rows itself (two kernels); from it on a kernel that
# quantizes x writes the plan, Gate/Up is deduplicated too and a row kernel quantizes the intermediate (the
# repeated quantization in Down and the per-route Gate/Up cost more than that kernel with more work; MiMo TP8
# I=256 from M12, TP4 I=512 from M6).
DECODE_DEDUP_MIN_M = 2
DECODE_DEDUP_MIN_H = 4096
# Routes per group: 2 up to DECODE_DEDUP_PAIR_MAX_M tokens (fewer staged rows; an expert rarely has three
# routes), DECODE_DEDUP_GROUP beyond.
DECODE_DEDUP_GROUP = 3
DECODE_DEDUP_PAIR_MAX_M = 16
DECODE_DEDUP_DOWN_BLOCKS = 2
DECODE_DEDUP_GATEUP_MIN_MI = 3072
# The deduplicated Gate/Up splits K over 4 waves per workgroup, over 8 up to DECODE_DEDUP_GATEUP_WIDE_MAX_M
# tokens (MiMo TP8 M12 / M16 3% faster; with more tokens 3-5% slower).
DECODE_DEDUP_GATEUP_WIDE_MAX_M = 16
# Above DECODE_MAX_M tokens the dedup decode path still runs, up to DECODE_DEDUP_MAX_M tokens, while the mean
# routes per expert (M * topk / E) stay at or below 1.5 (MiMo E256 topk 8: M <= 48; streaming beyond).
DECODE_DEDUP_MAX_M = 64
# Above DECODE_MAX_M tokens the weight-streaming kernels run while the mean rows per expert
# (M * topk / E) stay at or below this, the prefill kernels (TILE_K 128) beyond: measured crossover
# between 22.0 and 24.7 rows per expert for TP2 and TP4 alike (E513, topk 11: M = 1024 / 1152).
STREAM_MAX_ROWS = 22
STREAM_ATOMIC = 1
# 16-row weight blocks the streaming Down keeps in flight.
STREAM_DOWN_PREFETCH = 4

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


def _decode_waves(topk, paired):
    """Waves of the Down decode kernel: few tokens (latency bound) one per routing slot; pre-quantized
    rows (``paired``) DECODE_DOWN_PQ_WAVES waves with slot pairs in rounds."""
    return min((topk + 1) // 2, DECODE_DOWN_PQ_WAVES) if paired else (topk if topk <= 16 else 4)


def _decode_fits(H, I, topk, paired):
    """Whether the decode kernels support this shape (Gate/Up and the single-slot or slot-pair Down)."""
    from .moe_gemm_2stage import a8w4_decode as d

    return d.gateup_decode_fits(H) and d.down_decode_fits(I, topk, _decode_waves(topk, paired), paired)


@functools.cache
def _decode_kernels(N1, H, I, topk, gateup_pq, down_pq):
    from .moe_gemm_2stage.a8w4_decode import (compile_a8w4_down_decode, compile_a8w4_down_decode_pq,
                                              compile_a8w4_gateup_decode)

    gateup = compile_a8w4_gateup_decode(N=N1, K=H, TOPK=topk, PREQUANT=gateup_pq)
    if down_pq:
        # More tokens: 16-byte loads of slot pairs.
        return gateup, compile_a8w4_down_decode_pq(N=H, K=I, TOPK=topk, WAVES=_decode_waves(topk, True))
    return gateup, compile_a8w4_down_decode(N=H, K=I, TOPK=topk, WAVES=_decode_waves(topk, False))


@functools.cache
def _gateup_plan_kernel(N1, H, topk, gateup_pq, group):
    from .moe_gemm_2stage.a8w4_decode import compile_a8w4_gateup_decode

    return compile_a8w4_gateup_decode(N=N1, K=H, TOPK=topk, PREQUANT=gateup_pq, PLAN=(group, DECODE_MAX_M, H))


@functools.cache
def _gateup_dd_kernel(N1, H, topk, group, waves):
    from .moe_gemm_2stage.a8w4_decode_dedup import compile_a8w4_gateup_decode_dd

    return compile_a8w4_gateup_decode_dd(N=N1, K=H, TOPK=topk, GROUP=group, WAVES=waves)


@functools.cache
def _row_quant_plan(cols, topk, group, max_m):
    from .moe_gemm_2stage.a8w4_decode_dedup import compile_a8w4_row_quant_plan

    return compile_a8w4_row_quant_plan(COLS=cols, TOPK=topk, MAX_M=max_m, GROUP=group, ZERO_COLS=cols)


@functools.cache
def _down_dd_kernel(H, I, topk, group, cb, quant):
    from .moe_gemm_2stage.a8w4_decode_dedup import compile_a8w4_down_decode_dd

    return compile_a8w4_down_decode_dd(N=H, K=I, TOPK=topk, GROUP=group, CB=cb, QUANT=quant)


def _down_dd_blocks(H, I):
    """16-channel blocks per wave of the dedup Down, or 0 if it does not support the shape."""
    cb = DECODE_DEDUP_DOWN_BLOCKS
    return cb if I % 128 == 0 and H % (64 * cb) == 0 else 0


def _dedup_beyond_decode(M, H, I, topk, experts):
    """Whether more than DECODE_MAX_M tokens still take the dedup decode path (see DECODE_DEDUP_MAX_M)."""
    return (M <= DECODE_DEDUP_MAX_M and 2 * M * topk <= 3 * experts and H >= DECODE_DEDUP_MIN_H
            and _down_dd_blocks(H, I) > 0)


def _overlaps(a, b):
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


def _fused_moe_decode_dd(hidden_states, weights, topk_weight, topk_ids, output, down_cb):
    """Decode with expert deduplication (``a8w4_decode_dedup``): Down reads each expert's weights once per group
    of its routes and adds into the zeroed output. While M * I stays below DECODE_DEDUP_GATEUP_MIN_MI the
    per-route Gate/Up also writes the plan and zeroes the output (two kernels; not when the output overlaps the
    input, which that Gate/Up still reads); from it on one kernel quantizes x, writes the plan and zeroes the
    output, and Gate/Up is deduplicated as well."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = weights.inter_padded
    group = 2 if M <= DECODE_DEDUP_PAIR_MAX_M else DECODE_DEDUP_GROUP
    device = hidden_states.device
    plan = torch.empty((M * topk, 4), dtype=torch.int32, device=device)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    if M * I < DECODE_DEDUP_GATEUP_MIN_MI and I % 256 == 0 and not _overlaps(output, hidden_states):
        gateup_pq = M * topk * (I // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS
        if gateup_pq:
            xq, xs = _quantize_decode(hidden_states, M, H, True)
        else:
            xq, xs = hidden_states, topk_weight
        _launch("a8w4_gateup_decode_plan" + ("_pq" if gateup_pq else ""),
                _gateup_plan_kernel(2 * I, H, topk, gateup_pq, group), xq, xs, topk_ids, weights.w1,
                weights.w1_delta, weights.w1_scale, mid, plan, output, M)
        quant = True
    else:
        xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
        xs = torch.empty((M,), dtype=torch.float32, device=device)
        _launch("a8w4_row_quant_plan_decode",
                _row_quant_plan(H, topk, group, DECODE_MAX_M if M <= DECODE_MAX_M else DECODE_DEDUP_MAX_M),
                hidden_states, xq, xs, topk_ids, plan, output, M)
        waves = 8 if M <= DECODE_DEDUP_GATEUP_WIDE_MAX_M and H % 1024 == 0 else 4
        _launch("a8w4_gateup_decode_dd", _gateup_dd_kernel(2 * I, H, topk, group, waves), xq, xs, topk_ids, plan,
                weights.w1, weights.w1_delta, weights.w1_scale, mid, M)
        quant = False
    mq, ms = (mid, topk_weight) if quant else _quantize_decode(mid, M * topk, I, False)
    _launch("a8w4_down_decode_dd", _down_dd_kernel(H, I, topk, group, down_cb, quant), mq, ms, topk_ids, topk_weight,
            plan, weights.w2, weights.w2_delta, weights.w2_scale, output, M)
    return output


def _fused_moe_decode(hidden_states, weights, topk_weight, topk_ids, output):
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = weights.inter_padded
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
    down_cb = _down_dd_blocks(H, I)
    if M >= DECODE_DEDUP_MIN_M and H >= DECODE_DEDUP_MIN_H and down_cb:
        return _fused_moe_decode_dd(hidden_states, weights, topk_weight, topk_ids, output, down_cb)
    # Few tokens: each workgroup quantizes its own activation rows; more tokens: quantize once.
    gateup_pq = M * topk * (I // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS
    # The pre-quantized Down addresses all experts' weights with 32-bit byte offsets; larger weights or
    # shapes the slot-pair kernel does not support take the single-slot Down.
    down_pq = (M >= DECODE_PREQUANT_MIN_M and weights.w2.numel() * weights.w2.element_size() < 2**32
               and _decode_fits(H, I, topk, True))
    gateup, down = _decode_kernels(2 * I, H, I, topk, gateup_pq, down_pq)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    if gateup_pq:
        xq, xs = _quantize_decode(hidden_states, M, H, True)
    else:
        xq, xs = hidden_states, topk_weight
    _launch("a8w4_gateup_decode" + ("_pq" if gateup_pq else ""), gateup, xq, xs, topk_ids, weights.w1,
            weights.w1_delta, weights.w1_scale, mid, mid, mid, M)
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


def _stream_fused_quant(inter, block_m, hidden=0):
    """Whether the streaming Gate/Up kernel quantizes the intermediate itself: four waves (G = I / 64 gate
    blocks each) cover all of I, when their accumulators fit (8 * G * sub-blocks <= 96 VGPRs). From hidden
    4096 on, waves of one gate block plus the row kernel are faster (measured: MiMo TP8 M512 191 -> 153 us)."""
    return block_m > 16 and hidden < 4096 and 8 * (inter // 64) * (block_m // 16) <= 96


@functools.cache
def _stream_gateup_kernel(N, K, topk, block_m, experts):
    from .moe_gemm_2stage.a8w4_stream import compile_a8w4_gateup_stream

    # Fused quantization: one workgroup of four waves per block covers all of I and quantizes the intermediate
    # in the epilogue; with up to 4 + 4 row blocks per wave the activations are a large share of the traffic, so
    # the waves share them in LDS. Otherwise many light waves (one gate and one up block each, four per
    # workgroup, eight with larger blocks) sharing the block's activations in LDS.
    if _stream_fused_quant(N // 2, block_m, K):
        G = N // 2 // 64
        tile = dict(G=G, PREFETCH=(4, 3), QUANT=True, SHARE=G <= 4)
    else:
        tile = dict(G=1, W=8 if block_m > 16 and (N // 2) % 128 == 0 else 4, PREFETCH=(6, 6))
    return compile_a8w4_gateup_stream(N=N, K=K, TOPK=topk, BLOCK_M=block_m, E=experts, **tile)


@functools.cache
def _stream_quant_kernel(cols):
    from .moe_gemm_2stage.a8w4_stream import compile_row_quant_bf16

    return compile_row_quant_bf16(cols)


@functools.cache
def _stream_prep_kernel(hidden, topk, experts, block_m, zero, qw=4):
    from .moe_gemm_2stage.a8w4_stream import compile_stream_prep

    return compile_stream_prep(H=hidden, TOPK=topk, E=experts, BM=block_m, ZERO=zero, QW=qw)


def _stream_prep(hidden_states, topk_weight, topk_ids, experts, block_m, output, zero=True):
    """Sorting and input quantization of the streaming path in one launch (``compile_stream_prep``); with
    ``zero`` the same launch zeroes ``output`` for the atomic Down."""
    from .moe_gemm_2stage.a8w4_stream import stream_prep_sizes

    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
    n_sort, rows, blocks = stream_prep_sizes(M, topk, experts, block_m)
    sorted_ids = torch.empty((rows,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((rows,), dtype=torch.float32, device=device)
    sorted_experts = torch.empty((blocks,), dtype=torch.int32, device=device)
    num_valid = torch.empty((2,), dtype=torch.int32, device=device)
    loc_ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    # Four waves per input row while the row's 512-column chunks split evenly or rows are few (measured: H 4096
    # M512 13.2 vs 14.5 us; H 2560 M1024 one wave per row 15.4 vs 17.7 us).
    qw = 4 if H % 2048 == 0 or M <= 256 else 1
    _launch("a8w4_stream_prep", _stream_prep_kernel(H, topk, experts, block_m, zero, qw), topk_ids, topk_weight,
            hidden_states, sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs, output, M,
            n_sort, rows)
    return sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs


@functools.cache
def _stream_sum_kernel(topk, cols):
    from .moe_gemm_2stage.a8w4_stream import compile_route_sum

    return compile_route_sum(topk, cols)


def _stream_down_shared(N, K, block_m):
    """Whether the streaming Down runs four waves of 8 row blocks that share the block's activation rows
    through LDS (``down_stream_lds``)."""
    from .moe_gemm_2stage import a8w4_common
    from .moe_gemm_2stage.a8w4_stream import down_stream_lds

    # Blocks of 32 and 64 rows: four waves of 8 row blocks each share the block's activation rows
    # through LDS when they fit (with a single sub-block the sharing costs more than it saves, except for
    # a wide hidden dimension over a short K: MiMo TP8 M256 103 -> 91 us).
    wide = N >= 4096 and K <= 256
    return ((block_m >= 32 or wide) and N % 512 == 0
            and down_stream_lds(K, block_m, W=4) <= a8w4_common.LDS_BYTES)


@functools.cache
def _stream_down_kernel(N, K, topk, block_m, experts, atomic=False):
    from .moe_gemm_2stage.a8w4_stream import compile_a8w4_down_stream

    tile = dict(R=8, W=4) if _stream_down_shared(N, K, block_m) else {}
    return compile_a8w4_down_stream(N=N, K=K, TOPK=topk, BLOCK_M=block_m, E=experts, ATOMIC=atomic,
                                    PREFETCH=STREAM_DOWN_PREFETCH, **tile)


def _stream_atomic(hidden, inter, block_m):
    """Whether the streaming Down adds its routes into the output with BF16 atomics (no routes buffer and
    route_sum): measured faster up to I = 320 (Qwen TP2 M512 355 -> 336 us, MiMo TP8 M512 272 -> 263 us), slower
    at I = 512 (MiMo TP4 M128 / M512 +4 us: the atomic epilogue's extra registers on top of K = 512 activations).
    ``STREAM_ATOMIC``: 0 never, 2 always (diagnostics)."""
    return STREAM_ATOMIC == 2 or (STREAM_ATOMIC == 1 and inter < 512)


def _stream_block_m(tokens, topk, experts):
    """Sorting block of the streaming kernels from the mean rows per expert (sub-blocks of 16).

    Block 16 only once Aiter sorts it with Opus (3 * routes >= 4 * experts); below that Aiter switches
    block 16 to its fused MXFP4 sort, a separate JIT module.
    """
    rows = tokens * topk / experts
    if rows <= 12:
        return 16 if 3 * tokens * topk >= 4 * experts else 32
    return 32 if rows <= 28 else 64


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


def _fused_moe_stream(hidden_states, weights, topk_weight, topk_ids, output):
    from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

    from .moe_gemm_2stage import sorted_sum

    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    E, I = weights.experts, weights.inter_padded
    block_m = _stream_block_m(M, topk, E)
    device = hidden_states.device
    atomic = H % 512 == 0 and _stream_atomic(H, I, block_m)
    if H % 512 == 0:
        sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs = _stream_prep(
            hidden_states, topk_weight, topk_ids, E, block_m, output, atomic)
    else:
        sorted_ids, sorted_weights, sorted_experts, num_valid, _, _, loc_ids = moe_sorting(
            topk_ids, topk_weight, E, H, hidden_states.dtype, block_m, accumulate=False,
            output_aux=AUX_SORT_OPUS)
        last_path.append("moe_sorting_opus")
        xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
        xs = torch.empty((M,), dtype=torch.float32, device=device)
        _launch("a8w4_row_quant_bf16", _stream_quant_kernel(H), hidden_states, xq, xs, M)
    task_num = sorted_experts.numel()
    gateup = _stream_gateup_kernel(2 * I, H, topk, block_m, E)
    routing = (sorted_ids, sorted_experts, num_valid)
    if not _stream_fused_quant(I, block_m, H):
        mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
        _launch(f"a8w4_gateup_stream_bm{block_m}", gateup, xq, weights.w1, weights.w1_delta, mid, *routing,
                weights.w1_scale, xs, xs, M, task_num)
        mq, ms = _quantize(mid, M * topk, I, "f32")
    else:
        mq = torch.empty((M * topk, I), dtype=torch.float8_e4m3fnuz, device=device)
        ms = torch.empty((M * topk,), dtype=torch.float32, device=device)
        _launch(f"a8w4_gateup_quant_stream_bm{block_m}", gateup, xq, weights.w1, weights.w1_delta, mq, *routing,
                weights.w1_scale, xs, ms, M, task_num)
    down_args = (sorted_ids, sorted_weights, sorted_experts, num_valid, weights.w2_scale, ms, M, task_num)
    if atomic:
        # The prep zeroed output: Down adds every route into its token's row with BF16 atomics.
        _launch(f"a8w4_down_stream_bm{block_m}", _stream_down_kernel(H, I, topk, block_m, E, True), mq,
                weights.w2, weights.w2_delta, output, *down_args)
        return output
    routes = torch.empty((task_num * block_m, H), dtype=torch.bfloat16, device=hidden_states.device)
    _launch(f"a8w4_down_stream_bm{block_m}", _stream_down_kernel(H, I, topk, block_m, E), mq, weights.w2,
            weights.w2_delta, routes, *down_args)
    if H % 512 == 0:
        _launch("a8w4_route_sum", _stream_sum_kernel(topk, H), loc_ids, routes, output, M, task_num * block_m)
        return output
    sorted_sum(topk, H)(loc_ids, routes, output, M)
    last_path.append("sorted_sum")
    return output


def fused_moe_a8w4(hidden_states, weights, topk_weight, topk_ids, *, output=None,
                   block_m=None, block_n=None, tile_k=None):
    """A8W4 MoE forward: BF16 ``hidden_states`` [M, H] -> BF16 [M, H].

    ``weights`` comes from ``prepare_a8w4_weights``; ``topk_weight`` float32 [M, topk] is applied
    in Down; ``topk_ids`` int32 [M, topk] in ``[0, E)`` (a fused shared expert is an ordinary
    expert id), distinct within a token (the sorted paths keep one route of a repeated id). The input
    and the f32 intermediate SiLU(gate) * up are quantized per row to FP8
    E4M3FNUZ with ``scale = amax / 240`` and ``q = RNE(x / scale)``. SiLU only, no expert mask /
    bias; ``M * topk * I_padded * 4 < 2**31``. Issues no host synchronization. Unless a tile option is
    given, M <= ``DECODE_MAX_M`` runs the decode kernels (no sorting) when they support the shape
    (H <= 4096, the intermediate rows fit in LDS; with expert deduplication up to ``DECODE_DEDUP_MAX_M``
    tokens while at most 1.5 routes fall on an expert), larger M with at most ``STREAM_MAX_ROWS`` rows per
    expert the streaming kernels and the rest the prefill kernels (TILE_K 128 when chosen for having
    more rows per expert). ``block_m`` / ``block_n`` / ``tile_k`` select the prefill kernels'
    tiles: ``block_m`` (default 64) is the sorting block and M tile of both kernels
    (``block_m * I_padded % 4096 == 0``, e.g. 64 or 128); ``block_n`` (default 128) / ``tile_k`` are
    the Gate/Up N tile (128 or 256, dividing 2 * padded I into whole gate/up halves) and K tile (128
    or 256; default 128 for M > 2048, where its lower LDS and register use raise occupancy, else 256).
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
    if block_m is None and block_n is None and tile_k is None:
        if ((M <= DECODE_MAX_M or _dedup_beyond_decode(M, H, I, topk, E))
                and _decode_fits(H, I, topk, False)):
            return _fused_moe_decode(hidden_states, weights, topk_weight, topk_ids, output)
        many_rows = M * topk > STREAM_MAX_ROWS * E
        if M > DECODE_MAX_M and not many_rows:
            return _fused_moe_stream(hidden_states, weights, topk_weight, topk_ids, output)
        tile_k = 128 if many_rows else 256
    block_m = 64 if block_m is None else block_m
    block_n = 128 if block_n is None else block_n
    tile_k = (128 if M > 2048 else 256) if tile_k is None else tile_k

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
