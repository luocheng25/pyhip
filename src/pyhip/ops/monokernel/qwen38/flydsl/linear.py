"""Per-token dynamic FP8 quantization (AITER's formula) and the W8A8 linear that reads SGLang's
shuffled FP8 weights in place.

``quant_rows``: one wave per row; ``scale = amax * f32(1 / 240)`` (1 for an all-zero row) and
``q = clamp(x * (1 / scale), +-240)`` rounded to nearest even, like ``reference.quantize_rows``.

``linear_fp8``: ``out[s, n] = bf16(acc[s, n] * x_scale[s] * w_scale[n])`` with ``acc`` the FP32
sum of FP8 products. One 4-wave CTA owns one 16-row block of the [N, K] weight: wave ``w``
streams the block's 1 KiB loads ``w, w + 4, ...`` (16 rows x 64 K each, contiguous in the 16x16
shuffle) and multiplies them with the matching activation bytes of up to ``16 * tiles`` rows,
which lanes read straight from global memory in the same K order; the four K quarters meet in
LDS. MFMA rows are weight rows and columns are tokens, so lane ``l`` ends with outputs
``4 (l // 16) + [0, 4)`` of token ``l % 16`` (one 8-byte BF16 store).
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr

from ._ir import (DROP, FP8_MAX, barrier, bf16_pair, bf16_word, bf16x4_half, buffer, fabs, fmax, fmin, fp8_pack,
                  i64_half, lds_base, lds_load, lds_store, load, mfma_bf16, mfma_fp8, scalar, store, wave_max, words,
                  zeros4)

WAVES = 4


@cache
def make_quant_rows(k):
    chunks = k // 8 // 64  # 16-byte BF16 chunks per lane
    assert k == chunks * 8 * 64

    @flyc.kernel(known_block_size=[64, 1, 1])
    def quant_rows(X: fx.Tensor, XQ: fx.Tensor, XS: fx.Tensor):
        lane = fx.Int32(fx.thread_idx.x)
        row = fx.Int32(fx.block_idx.x)
        x = buffer(X, k * 2, row * k)
        values = []
        for c in range_constexpr(chunks):
            word = load(x, (c * 64 + lane) * 16)
            for i in range_constexpr(4):
                values += list(bf16_pair(word[i]))
        amax = fx.Float32(0.0)
        for value in values:
            amax = fmax(amax, fabs(value))
        amax = wave_max(amax, lane)
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        scale = (amax > 0.0).select(scale, fx.Float32(1.0))
        inverse = fx.Float32(1.0) / scale
        q = buffer(XQ, k, row * k)
        for c in range_constexpr(chunks):
            packed = []
            for j in range_constexpr(2):
                quads = [fmin(fmax(values[c * 8 + 4 * j + e] * inverse, fx.Float32(-FP8_MAX)), fx.Float32(FP8_MAX))
                         for e in range_constexpr(4)]
                packed.append(fp8_pack(*quads))
            store(q, (c * 64 + lane) * 8, words(packed))
        first = (lane == 0).select(fx.Int32(0), fx.Int32(DROP))
        store(buffer(XS, 4, row), first, scale.bitcast(fx.Int32))

    @flyc.jit
    def launch(X: fx.Tensor, XQ: fx.Tensor, XS: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            quant_rows(X, XQ, XS).launch(grid=(rows, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


class Gemm:
    """K loop of one 16-row weight block against ``16 * tiles`` activation rows, split over the 4
    waves of a CTA (wave ``w`` takes loads ``w, w + 4, ...`` of its K range), plus the LDS reduction.
    ``weight``: "fp8" (FP8 x FP8) or "bf16" (BF16 x BF16), both in the 16x16 shuffle."""

    def __init__(self, k, tiles, weight="fp8", ksplit=1):
        esize = 1 if weight == "fp8" else 2
        self.row_bytes = k * esize
        self.loads = self.row_bytes // 64
        assert self.loads % ksplit == 0 and tiles in (1, 2, 4)
        self.split_loads = self.loads // ksplit
        self.per_wave = -(-self.split_loads // WAVES)
        self.tiles = tiles
        self.lds_words = WAVES * tiles * 64 * 4
        self.mfma = mfma_fp8 if weight == "fp8" else mfma_bf16
        self.operand = i64_half if weight == "fp8" else bf16x4_half

    def accumulate(self, a, w, block, split, lane, wave):
        acc = [zeros4() for _ in range_constexpr(self.tiles)]
        for j in range_constexpr(self.per_wave):
            local = wave + WAVES * j
            q = split * self.split_loads + local
            valid = local < self.split_loads
            weight_word = load(w, valid.select(block * (16 * self.row_bytes) + q * 1024 + lane * 16, fx.Int32(DROP)))
            for t in range_constexpr(self.tiles):
                token = 16 * t + (lane & 15)
                act = load(a, valid.select(token * self.row_bytes + q * 64 + (lane >> 4) * 16, fx.Int32(DROP)))
                for h in range_constexpr(2):
                    acc[t] = self.mfma(self.operand(weight_word, h), self.operand(act, h), acc[t])
        return acc

    def stash(self, shared, acc, lane, wave):
        for t in range_constexpr(self.tiles):
            lds_store(shared + ((wave * self.tiles + t) * 64 + lane) * 16, acc[t].bitcast(fx.Int32))

    def gather(self, shared, t, lane):
        """Sum of the 4 waves' accumulators of token tile ``t`` (call after a barrier)."""
        total = lds_load(shared + (t * 64 + lane) * 16).bitcast(fx.Float32)
        for other in range_constexpr(1, WAVES):
            total = total + lds_load(shared + ((other * self.tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
        return total


@cache
def make_linear(n, k, tiles, weight="fp8", epilogue="scaled", ksplit=1, n_out=None):
    """``weight``: "fp8" (W8A8, FP8 activations) or "bf16" (BF16 activations and 16x16-shuffled BF16
    weights). ``epilogue``: "scaled" BF16 ``acc * x_scale * w_scale``, "bf16", or "f32" (with
    ``ksplit`` > 1 the FP32 partials of K split ``s`` go to ``out[s]``). ``n`` weight rows (a multiple
    of 16) produce ``n_out`` (a multiple of 4) output columns."""
    gemm = Gemm(k, tiles, weight, ksplit)
    n_out = n if n_out is None else n_out
    assert n % 16 == 0 and n_out % 4 == 0 and n_out <= n

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def linear(A: fx.Tensor, XS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, gemm.lds_words, 16]).peek()
        shared = lds_base(storage, gemm.lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        split = fx.Int32(fx.block_idx.y)
        acc = gemm.accumulate(buffer(A, rows * gemm.row_bytes), buffer(W, n * gemm.row_bytes), block, split,
                              lane, wave)
        gemm.stash(shared, acc, lane, wave)
        barrier()
        n0 = block * 16 + (lane >> 4) * 4
        for t in range_constexpr(tiles):
            if wave == t % WAVES:
                total = gemm.gather(shared, t, lane)
                token = 16 * t + (lane & 15)
                live = (token < rows) & (n0 < n_out)
                if fx.const_expr(epilogue == "f32"):
                    # Padding tokens of split s would land in split s + 1: drop them explicitly.
                    offset = live.select(((split * rows + token) * n_out + n0) * 4, fx.Int32(DROP))
                    store(buffer(OUT, ksplit * rows * n_out * 4), offset, total.bitcast(fx.Int32))
                else:
                    vals = [total[i] for i in range_constexpr(4)]
                    if fx.const_expr(epilogue == "scaled"):
                        xs = load(buffer(XS, rows * 4), token * 4, words=1).bitcast(fx.Float32)
                        ws = load(buffer(WS, n * 4), n0 * 4).bitcast(fx.Float32)
                        vals = [vals[i] * xs * ws[i] for i in range_constexpr(4)]
                    store(buffer(OUT, rows * n_out * 2), live.select((token * n_out + n0) * 2, fx.Int32(DROP)),
                          words((bf16_word(vals[0], vals[1]), bf16_word(vals[2], vals[3]))))

    @flyc.jit
    def launch(A: fx.Tensor, XS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, rows: fx.Int32,
               stream: fx.Stream):
        if rows > 0:
            linear(A, XS, W, WS, OUT, rows).launch(grid=(n // 16, ksplit, 1), block=(64 * WAVES, 1, 1),
                                                   stream=stream)

    return launch


_COMPILED = {}


def _call(key, builder, args):
    function = _COMPILED.get(key)
    if function is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm MonoKernel stages before graph capture")
        # FlyDSL keys its disk cache on scalar closure values only; parameters held in objects (e.g.
        # Gemm's K) would let another configuration's kernel load, so the full key goes in a hint.
        # flyc.compile also runs the call once; launching again would repeat in-place state updates.
        with CompilationContext.compile_hints({"monokernel_config": repr(key)}):
            _COMPILED[key] = flyc.compile(builder, *args)
        return
    function(*args)


def tiles_for(rows):
    return 1 if rows <= 16 else 2 if rows <= 32 else 4


def quant_rows(x, xq, xs):
    """x: BF16 [rows, K]; xq: FP8 (any 1-byte dtype) [rows, K]; xs: FP32 [rows]."""
    rows, k = x.shape
    stream = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device):
        _call(("quant", x.device, k), make_quant_rows(k), (x.view(-1), xq.view(torch.uint8).view(-1), xs, rows, stream))


def linear_fp8(xq, xs, weight, weight_scale, out):
    """xq: FP8 [rows, K]; xs: FP32 [rows]; weight: SGLang FP8 [N, K] (16x16 shuffled); weight_scale:
    FP32 [N] or [N, 1]; out: BF16 [rows, N]."""
    rows, k = xq.shape
    n = weight.shape[0]
    tiles = tiles_for(rows)
    stream = torch.cuda.current_stream(xq.device)
    args = (xq.view(torch.uint8).view(-1), xs, weight.view(torch.uint8).view(-1), weight_scale.reshape(-1),
            out.view(-1), rows, stream)
    with torch.cuda.device(xq.device):
        _call(("linear", xq.device, n, k, tiles, "fp8", "scaled", 1), make_linear(n, k, tiles), args)


def linear_bf16(x, weight, out, ksplit=1):
    """x: BF16 [rows, K]; weight: 16x16-shuffled BF16 [N, K] (N padded to 16); out: BF16 [rows, n_out], or
    FP32 [rows, n_out] / [ksplit, rows, n_out] partial sums."""
    rows, k = x.shape
    n, n_out = weight.shape[0], out.shape[-1]
    tiles = tiles_for(rows)
    epilogue = "f32" if out.dtype == torch.float32 else "bf16"
    stream = torch.cuda.current_stream(x.device)
    args = (x.view(-1), x.view(-1), weight.view(-1), weight.view(-1), out.view(-1), rows, stream)
    with torch.cuda.device(x.device):
        _call(("linear", x.device, n, k, tiles, "bf16", epilogue, ksplit, n_out),
              make_linear(n, k, tiles, "bf16", epilogue, ksplit, n_out), args)
