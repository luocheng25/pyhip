# SPDX-License-Identifier: MIT

"""Per-row FP8 E4M3FNUZ quantization for the A8W4 path: ``scale = amax / 240``, ``q = RNE(x / scale)``.

Matches ``torch`` exactly (IEEE division, round-to-nearest-even conversion, ``-0 -> +0``), so the
activation quantization points agree bit for bit with ``pyhip.testing.moe_a8w4``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import common as fxh

ROWS_PER_BLOCK = 4


@functools.cache
def compile_row_quant_fp8(cols, in_dtype):
    """Launcher ``(x, q, scale, rows, stream)``: x ``in_dtype`` [rows, cols] -> q FP8 [rows, cols], scale f32 [rows].

    One wave per row; ``cols % 64 == 0``; ``in_dtype`` is ``"bf16"`` or ``"f32"``. All-zero rows
    get scale 1.
    """
    assert cols % 64 == 0, f"cols={cols} must be a multiple of 64"
    per_lane = cols // 64
    src_type = {"bf16": fx.BFloat16, "f32": fx.Float32}[in_dtype]

    @flyc.kernel
    def row_quant_fp8(p_in: fx.Pointer, p_out: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32):
        tid = gpu.thread_idx.x
        lane = tid % 64
        row = gpu.block_idx.x * ROWS_PER_BLOCK + tid // 64
        if row < rows:
            src = fx.make_view(fxh._as_ptr(p_in, src_type) + fx.Int64(row) * cols, fx.make_layout(cols, 1))
            dst = fx.make_view(fxh._as_ptr(p_out, fx.Uint8) + fx.Int64(row) * cols, fx.make_layout(cols, 1))
            values = [fx.Float32(src[j * 64 + lane].to(fx.Float32)) for j in range_constexpr(per_lane)]
            amax = abs(values[0])
            for j in range_constexpr(1, per_lane):
                amax = amax.maximumf(abs(values[j]))
            for s in range_constexpr(6):
                amax = amax.maximumf(amax.shuffle_xor(32 >> s, 64))
            scale = amax / 240.0
            scale = (scale == 0.0).select(fx.Float32(1.0), scale)
            for j in range_constexpr(per_lane):
                q = values[j] / scale
                q = q.maximumf(fx.Float32(-240.0)).minimumf(fx.Float32(240.0))
                packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, q, fx.Float32(0.0), fx.Int32(0), False))
                byte = packed & fx.Uint32(0xFF)
                byte = (byte == fx.Uint32(0x80)).select(fx.Uint32(0), byte)
                dst[j * 64 + lane] = byte.to(fx.Uint8)
            if lane == 0:
                scale_out = fx.make_view(fxh._as_ptr(p_scale, fx.Float32) + fx.Int64(row), fx.make_layout(1, 1))
                scale_out[0] = scale

    @flyc.jit
    def launch(p_in: fx.Pointer, p_out: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32, stream: fx.Stream):
        CompilationContext.get_current()
        row_quant_fp8(p_in, p_out, p_scale, rows).launch(
            grid=((rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 1, 1),
            block=(64 * ROWS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    return launch
