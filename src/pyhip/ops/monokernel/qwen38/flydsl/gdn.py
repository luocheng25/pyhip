"""Gated DeltaNet decode stages.

``qkvz_conv``: the W8A8 ``in_proj_qkvz`` (rows [q | k | v | z] of this rank) writes BF16 QKVZ; for the
conv channels (q, k, v) the epilogue also runs the depthwise causal conv over [state, x] like
SGLang's Triton kernel (BF16-rounded tap products summed in FP32, SiLU, BF16), writes CONV and
shifts x into the conv state of the row's slot. Rows with a negative slot write zero CONV and keep
their state.

``gdn_recurrent``: one 512-thread CTA per (row, v-head) running ``phases.recurrent`` (shared with the
fused launch); thread ``t`` owns state row ``t // 4`` and 32 columns. L2-normalized q (scaled by
128^-0.5) and k of the head's k-head, ``beta = bf16(sigmoid(b))``, decay
``exp(-exp(A_log) * softplus(a + dt_bias))``; then ``S *= decay``, ``delta = (v - S k) beta``,
``S += delta k^T``, ``core = bf16(S q)`` and the gated RMSNorm
``y = bf16(core * rsqrt(mean(core^2) + eps) * w * sigmoid(z))``.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from ..config import QWEN38_FLASH_NEXT
from ._ir import (DROP, barrier, bf16_pair, bf16_round, bf16_word, buffer, exp_accurate, lds_base,
                  load, scalar, store, words)
from .linear import WAVES, Gemm, _call, tiles_for

HIDDEN, D, KERNEL = QWEN38_FLASH_NEXT.hidden, QWEN38_FLASH_NEXT.gdn_head_dim, QWEN38_FLASH_NEXT.conv_kernel
L2NORM_EPS = 1.0e-6


def _bf16s(word_list):
    out = []
    for word in word_list:
        out += list(bf16_pair(word))
    return out


@cache
def make_qkvz_conv(n, conv_dim, tiles):
    gemm = Gemm(HIDDEN, tiles, "fp8")
    assert conv_dim % 16 == 0 and KERNEL == 4

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def qkvz_conv(XQ: fx.Tensor, XS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, CW: fx.Tensor, SLOTS: fx.Tensor,
                  STATE: fx.Tensor, QKVZ: fx.Tensor, CONV: fx.Tensor, rows: fx.Int32, state_bytes: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, gemm.lds_words, 16]).peek()
        shared = lds_base(storage, gemm.lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        acc = gemm.accumulate(buffer(XQ, rows * gemm.row_bytes), buffer(W, n * gemm.row_bytes), block,
                              fx.Int32(0), lane, wave)
        gemm.stash(shared, acc, lane, wave)
        barrier()
        n0 = block * 16 + (lane >> 4) * 4
        ws = load(buffer(WS, n * 4), n0 * 4).bitcast(fx.Float32)
        for t in range_constexpr(tiles):
            if wave == t % WAVES:
                total = gemm.gather(shared, t, lane)
                token = 16 * t + (lane & 15)
                xs = load(buffer(XS, rows * 4), token * 4, words=1).bitcast(fx.Float32)
                x = [bf16_round(total[i] * xs * ws[i]) for i in range_constexpr(4)]
                store(buffer(QKVZ, rows * n * 2), (token * n + n0) * 2,
                      words((bf16_word(x[0], x[1]), bf16_word(x[2], x[3]))))
                if block * 16 < conv_dim:
                    slot = load(buffer(SLOTS, rows * 4), token * 4, words=1)
                    active = (token < rows) & (slot >= 0)
                    offset = active.select((slot * conv_dim + n0) * 6, fx.Int32(DROP))
                    state = buffer(STATE, state_bytes)
                    first = load(state, offset, words=4)
                    second = load(state, offset + 16, words=2)
                    taps = _bf16s([first[i] for i in range_constexpr(4)] + [second[i] for i in range_constexpr(2)])
                    weight_words = load(buffer(CW, conv_dim * KERNEL * 2), n0 * KERNEL * 2)
                    weight_words2 = load(buffer(CW, conv_dim * KERNEL * 2), n0 * KERNEL * 2 + 16)
                    weights = _bf16s([weight_words[i] for i in range_constexpr(4)]
                                     + [weight_words2[i] for i in range_constexpr(4)])
                    outs, shifted = [], []
                    for c in range_constexpr(4):
                        window = [taps[3 * c], taps[3 * c + 1], taps[3 * c + 2], x[c]]
                        total_c = fx.Float32(0.0)
                        for j in range_constexpr(KERNEL):
                            product = bf16_round(window[j] * weights[KERNEL * c + j])
                            total_c = product if j == 0 else total_c + product
                        activated = total_c / (1.0 + exp_accurate(fx.Float32(0.0) - total_c))
                        outs.append(active.select(activated, fx.Float32(0.0)))
                        shifted += window[1:]
                    store(buffer(CONV, rows * conv_dim * 2), (token * conv_dim + n0) * 2,
                          words((bf16_word(outs[0], outs[1]), bf16_word(outs[2], outs[3]))))
                    packed = [bf16_word(shifted[2 * i], shifted[2 * i + 1]) for i in range_constexpr(6)]
                    store(state, offset, words(packed[:4]))
                    store(state, active.select(offset + 16, fx.Int32(DROP)), words(packed[4:]))

    @flyc.jit
    def launch(XQ: fx.Tensor, XS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, CW: fx.Tensor, SLOTS: fx.Tensor,
               STATE: fx.Tensor, QKVZ: fx.Tensor, CONV: fx.Tensor, rows: fx.Int32, state_bytes: fx.Int32,
               stream: fx.Stream):
        if rows > 0:
            qkvz_conv(XQ, XS, W, WS, CW, SLOTS, STATE, QKVZ, CONV, rows, state_bytes).launch(
                grid=(n // 16, 1, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch


@cache
def make_recurrent(v_heads, k_heads, conv_dim, eps):
    from . import phases
    from ._phase import THREADS, Ctx
    from .sync import address

    width = conv_dim + v_heads * D  # QKVZ row: conv channels then z

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def gdn_recurrent(CONV: fx.Tensor, QKVZ: fx.Tensor, BA: fx.Tensor, ALOG: fx.Tensor, DTB: fx.Tensor,
                      NW: fx.Tensor, SSM: fx.Tensor, SLOTS: fx.Tensor, CORE: fx.Tensor, Y: fx.Tensor, rows: fx.Int32):
        ctx = Ctx(rows, False, lds_words=THREADS // 64)
        phases.recurrent(ctx, buffer(CONV, rows * conv_dim * 2), buffer(QKVZ, rows * width * 2),
                         buffer(BA, rows * 2 * v_heads * 2), buffer(ALOG, v_heads * 4), buffer(DTB, v_heads * 2),
                         buffer(NW, D * 2), address(SSM), buffer(SLOTS, rows * 4), buffer(CORE, rows * v_heads * D * 2),
                         buffer(Y, rows * v_heads * D * 2), v_heads, k_heads, conv_dim, eps)

    @flyc.jit
    def launch(CONV: fx.Tensor, QKVZ: fx.Tensor, BA: fx.Tensor, ALOG: fx.Tensor, DTB: fx.Tensor, NW: fx.Tensor,
               SSM: fx.Tensor, SLOTS: fx.Tensor, CORE: fx.Tensor, Y: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            gdn_recurrent(CONV, QKVZ, BA, ALOG, DTB, NW, SSM, SLOTS, CORE, Y, rows).launch(
                grid=(rows * v_heads, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------- host


def qkvz_conv(xq, xs, weight, weight_scale, conv_weight, slots, conv_state, qkvz, conv_out):
    """xq/xs: per-token FP8 activations; weight/weight_scale: SGLang ``in_proj_qkvz``; conv_weight:
    [conv_dim, 1, 4] BF16; slots: int32 [rows]; conv_state: SGLang pool [slots, conv_dim, 3] BF16
    (updated); qkvz: BF16 [rows, N]; conv_out: BF16 [rows, conv_dim]."""
    rows = xq.shape[0]
    n, conv_dim = weight.shape[0], conv_weight.shape[0]
    tiles = tiles_for(rows)
    stream = torch.cuda.current_stream(xq.device)
    args = (xq.view(torch.uint8).view(-1), xs, weight.view(torch.uint8).view(-1), weight_scale.reshape(-1),
            conv_weight.view(-1), slots, conv_state.view(-1), qkvz.view(-1), conv_out.view(-1), rows,
            conv_state.numel() * 2, stream)
    with torch.cuda.device(xq.device):
        _call(("qkvz_conv", xq.device, n, conv_dim, tiles), make_qkvz_conv(n, conv_dim, tiles), args)


def gdn_recurrent(conv_out, qkvz, ba, a_log, dt_bias, norm_weight, ssm_state, slots, core, y, v_heads, k_heads,
                  eps=QWEN38_FLASH_NEXT.eps):
    """ssm_state: SGLang pool [slots, v_heads, 128, 128] FP32 (updated); core: BF16 [rows, v_heads, 128];
    y: BF16 [rows, v_heads * 128] (gated norm output)."""
    rows, conv_dim = conv_out.shape
    stream = torch.cuda.current_stream(conv_out.device)
    args = (conv_out.view(-1), qkvz.view(-1), ba.view(-1), a_log.view(-1), dt_bias.view(-1), norm_weight.view(-1),
            ssm_state.view(-1), slots, core.view(-1), y.view(-1), rows, stream)
    with torch.cuda.device(conv_out.device):
        _call(("gdn_recurrent", conv_out.device, v_heads, k_heads, conv_dim, eps),
              make_recurrent(v_heads, k_heads, conv_dim, float(eps)), args)
