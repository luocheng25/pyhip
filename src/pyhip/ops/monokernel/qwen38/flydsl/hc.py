"""Hyper-connection (HC) stages: low-rank mix of the 4 branches and the combine with the next norm.

Mix (``GatedResidual.mix`` on the normalized branches ``hn`` [rows, 4 * 2560]), as SGLang's
fused_hc_mix on ROCm: ``t = hn @ [down; inject]^T`` in FP32 (``linear_bf16`` split-K partials),
``lowrank = bf16(silu(t / 4))``, ``gate = sigmoid(lowrank @ up^T)`` and ``x[:, j] = sum_g gate[g, j]
* hn[g, j] / 4``. The inject rows give the combine gains ``2 * sigmoid(t / 4)``.

Combine (``GatedResidual.combine`` plus the next GroupedGemmaRMSNorm): ``h' = bf16(h + y * gain_g)``
per branch ``g`` and ``hn' = bf16(h' * rsqrt(mean(h'^2) + eps) * (1 + w))`` over each 2560 group.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl

from ..config import QWEN38_FLASH_NEXT
from ._ir import (DROP, barrier, bf16_pair, bf16_round, bf16_word, bf16x4_half, buffer, lds_base, lds_load,
                  lds_store, load, mfma_bf16, rne_bits, scalar, sigmoid, silu, store, store_short, wave_sum, words,
                  zeros4)
from .linear import WAVES, _call, linear_bf16, tiles_for

C, H = QWEN38_FLASH_NEXT.hc_count, QWEN38_FLASH_NEXT.hidden
WIDTH, LOWRANK = C * H, QWEN38_FLASH_NEXT.hc_lowrank
DOWN_ROWS = 336  # low-rank rows, 4 inject rows, zero padding
KSPLIT = 16


@cache
def make_lowrank():
    threads = 192

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def hc_lowrank(P: fx.Tensor, LOW: fx.Tensor, GAIN: fx.Tensor, rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        row = fx.Int32(fx.block_idx.x)
        partials = buffer(P, KSPLIT * rows * DOWN_ROWS * 4)
        # Thread i < 160 owns low-rank columns 2i, 2i + 1; threads 160..161 own the 4 inject columns.
        column = (tid < LOWRANK // 2).select(2 * tid, LOWRANK + 2 * (tid - LOWRANK // 2))
        valid = tid < (LOWRANK + C) // 2
        a, b = fx.Float32(0.0), fx.Float32(0.0)
        for s in range_constexpr(KSPLIT):
            pair = load(partials, valid.select(((s * rows + row) * DOWN_ROWS + column) * 4, fx.Int32(DROP)), words=2)
            pair = pair.bitcast(fx.Float32)
            a, b = a + pair[0], b + pair[1]
        a, b = a * 0.25, b * 0.25
        if tid < LOWRANK // 2:
            store(buffer(LOW, rows * LOWRANK * 2), (row * LOWRANK + column) * 2, bf16_word(silu(a), silu(b)))
        if tid >= LOWRANK // 2:
            gains = words((sigmoid(a) * 2.0, sigmoid(b) * 2.0), fx.Float32).bitcast(fx.Int32)
            store(buffer(GAIN, rows * C * 4), valid.select((row * C + column - LOWRANK) * 4, fx.Int32(DROP)), gains)

    @flyc.jit
    def launch(P: fx.Tensor, LOW: fx.Tensor, GAIN: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            hc_lowrank(P, LOW, GAIN, rows).launch(grid=(rows, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@cache
def make_mix_up(tiles):
    row_bytes = LOWRANK * 2
    loads = row_bytes // 64  # 10 per 16-row block
    per_wave = -(-loads // WAVES)
    lds_words = WAVES * tiles * 64 * 4

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def hc_mix_up(LOW: fx.Tensor, W: fx.Tensor, HN: fx.Tensor, X: fx.Tensor, rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_words, 16]).peek()
        shared = lds_base(storage, lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        a = buffer(LOW, rows * row_bytes)
        w = buffer(W, WIDTH * row_bytes)
        acc = [zeros4() for _ in range_constexpr(tiles)]
        for j in range_constexpr(per_wave):
            q = wave + WAVES * j
            valid = q < loads
            weight_word = load(w, valid.select(block * (16 * row_bytes) + q * 1024 + lane * 16, fx.Int32(DROP)))
            for t in range_constexpr(tiles):
                token = 16 * t + (lane & 15)
                act = load(a, valid.select(token * row_bytes + q * 64 + (lane >> 4) * 16, fx.Int32(DROP)))
                for h in range_constexpr(2):
                    acc[t] = mfma_bf16(bf16x4_half(weight_word, h), bf16x4_half(act, h), acc[t])
        for t in range_constexpr(tiles):
            lds_store(shared + ((wave * tiles + t) * 64 + lane) * 16, acc[t].bitcast(fx.Int32))
        barrier()
        # Lane l holds the 4 branches of column 4 * block + l // 16 for token l % 16.
        column = block * 4 + (lane >> 4)
        hn = buffer(HN, rows * WIDTH * 2)
        out = buffer(X, rows * H * 2)
        for t in range_constexpr(tiles):
            if wave == t % WAVES:
                total = lds_load(shared + (t * 64 + lane) * 16).bitcast(fx.Float32)
                for other in range_constexpr(1, WAVES):
                    total = total + lds_load(shared + ((other * tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
                token = 16 * t + (lane & 15)
                mixed = fx.Float32(0.0)
                for g in range_constexpr(C):
                    index = token * WIDTH + g * H + column
                    word = load(hn, (index >> 1) * 4, words=1)
                    lo, hi = bf16_pair(word)
                    value = ((index & 1) == 1).select(hi, lo)
                    mixed = mixed + sigmoid(total[g]) * value
                store_short(out, (token * H + column) * 2, rne_bits(mixed * 0.25).bitcast(fx.Int32) >> 16)

    @flyc.jit
    def launch(LOW: fx.Tensor, W: fx.Tensor, HN: fx.Tensor, X: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            hc_mix_up(LOW, W, HN, X, rows).launch(grid=(WIDTH // 16, 1, 1), block=(64 * WAVES, 1, 1),
                                                 stream=stream)

    return launch


@cache
def make_combine_norm(combine, eps):
    threads = H // 8  # 320: 8 columns of the branch per thread
    waves = threads // 64

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def hc_combine_norm(Hin: fx.Tensor, Y: fx.Tensor, GAIN: fx.Tensor, NW: fx.Tensor, OUT: fx.Tensor,
                        NORMED: fx.Tensor, rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, 8, 16]).peek()
        shared = lds_base(storage, 8)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        row, branch = block // C, block % C
        offset = (row * WIDTH + branch * H + tid * 8) * 2
        residual = load(buffer(Hin, rows * WIDTH * 2), offset)
        weight = load(buffer(NW, WIDTH * 2), (branch * H + tid * 8) * 2)
        values = []
        for i in range_constexpr(4):
            values += list(bf16_pair(residual[i]))
        if fx.const_expr(combine):
            y = load(buffer(Y, rows * H * 2), (row * H + tid * 8) * 2)
            gain = load(buffer(GAIN, rows * C * 4), (row * C + branch) * 4, words=1).bitcast(fx.Float32)
            ys = []
            for i in range_constexpr(4):
                ys += list(bf16_pair(y[i]))
            values = [bf16_round(values[i] + ys[i] * gain) for i in range_constexpr(8)]
            store(buffer(OUT, rows * WIDTH * 2), offset,
                  words([bf16_word(values[2 * i], values[2 * i + 1]) for i in range_constexpr(4)]))
        squares = fx.Float32(0.0)
        for v in values:
            squares = squares + v * v
        squares = wave_sum(squares, lane)
        if lane == 0:
            lds_store(shared + wave * 4, squares.bitcast(fx.Int32), words=1)
        barrier()
        total = lds_load(shared, words=1).bitcast(fx.Float32)
        for w in range_constexpr(1, waves):
            total = total + lds_load(shared + w * 4, words=1).bitcast(fx.Float32)
        mean = total * fx.Float32(1.0 / H)
        rstd = fx.Float32(rocdl.rsq(fx.Float32.ir_type, (mean + eps).ir_value()))
        gains = []
        for i in range_constexpr(4):
            gains += list(bf16_pair(weight[i]))
        normed = [(values[i] * rstd) * (1.0 + gains[i]) for i in range_constexpr(8)]
        store(buffer(NORMED, rows * WIDTH * 2), offset,
              words([bf16_word(normed[2 * i], normed[2 * i + 1]) for i in range_constexpr(4)]))

    @flyc.jit
    def launch(Hin: fx.Tensor, Y: fx.Tensor, GAIN: fx.Tensor, NW: fx.Tensor, OUT: fx.Tensor, NORMED: fx.Tensor,
               rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            hc_combine_norm(Hin, Y, GAIN, NW, OUT, NORMED, rows).launch(grid=(rows * C, 1, 1),
                                                                        block=(threads, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------- host


class HCWorkspace:
    """Per-call FP32 split-K partials, low-rank activations and combine gains for ``rows``."""

    def __init__(self, rows, device):
        self.partial = torch.empty(KSPLIT, rows, DOWN_ROWS, dtype=torch.float32, device=device)
        self.lowrank = torch.empty(rows, LOWRANK, dtype=torch.bfloat16, device=device)
        self.gain = torch.empty(rows, C, dtype=torch.float32, device=device)


def hc_mix(hn, packed_down, packed_up, x, work):
    """hn: BF16 [rows, 10240]; packed_* from ``pack_layer``; x: BF16 [rows, 2560] (output).
    Leaves the low-rank activations and the combine gains in ``work``."""
    rows = hn.shape[0]
    linear_bf16(hn, packed_down, work.partial, ksplit=KSPLIT)
    stream = torch.cuda.current_stream(hn.device)
    with torch.cuda.device(hn.device):
        _call(("hc_lowrank", hn.device), make_lowrank(), (work.partial.view(-1), work.lowrank.view(-1),
                                                          work.gain.view(-1), rows, stream))
        tiles = tiles_for(rows)
        _call(("hc_mix_up", hn.device, tiles), make_mix_up(tiles),
              (work.lowrank.view(-1), packed_up.view(-1), hn.view(-1), x.view(-1), rows, stream))


def hc_combine_norm(h, y, gain, norm_weight, out, normed, eps=QWEN38_FLASH_NEXT.eps):
    """out = bf16(h + y * gain per branch); normed = GroupedGemmaRMSNorm(out). With ``y`` None, only
    ``normed = norm(h)`` (``out`` and ``gain`` unused)."""
    rows = h.shape[0]
    combine = y is not None
    stream = torch.cuda.current_stream(h.device)
    args = (h.view(-1), (y if combine else h).view(-1), (gain if combine else h).view(-1), norm_weight.view(-1),
            (out if combine else h).view(-1), normed.view(-1), rows, stream)
    with torch.cuda.device(h.device):
        _call(("hc_combine_norm", h.device, combine, eps), make_combine_norm(combine, float(eps)), args)
