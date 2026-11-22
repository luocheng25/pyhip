# SPDX-License-Identifier: MIT

"""Per-row FP8 E4M3FNUZ quantization for the A8W4 path: ``scale = amax / 240``, ``q = RNE(x / scale)``.

Matches ``torch`` exactly (IEEE quotient, round-to-nearest-even conversion, ``-0 -> +0``), so the
activation quantization points agree bit for bit with ``pyhip.testing.moe_a8w4``.

A row is split over ``L`` lanes, each owning a contiguous run (16-byte loads). The quotient
``x / scale`` uses ``y = 1 / scale`` (IEEE, once per row) and two FMA corrections
``q += (x - q * scale) * y``: the first makes ``q`` faithful, the second then yields the correctly
rounded quotient (Markstein). Rows with ``scale < 2**-64``, where the residual could underflow,
divide per element instead.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh

_ESIZE = {"bf16": 2, "f32": 4}
_SMALL_SCALE = 2.0 ** -64


def _lanes_per_row(cols, esize):
    for lanes in (64, 32, 16, 8, 4):
        if cols % lanes == 0 and (cols // lanes * esize) % 16 == 0:
            return lanes
    return None


def _buffer_rsrc(ptr, nbytes):
    view = fx.make_view(fx.recast_iter(fx.Uint8, fxh._as_ptr(ptr)), fx.make_layout(1, 1))
    tensor = fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=nbytes)
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


def _store(rsrc, value, offset):
    """Raw buffer store; bytes past the buffer end are dropped."""
    rocdl_dialect.RawPtrBufferStoreOp(
        _raw(value), rsrc, offset.ir_value(), fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    )


def _pack_fp8x4(q):
    """4 clamped f32 -> one dword of FP8 E4M3FNUZ (RNE); 0x80 (-0, NaN in FNUZ) becomes 0."""
    packed = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], packed, True))
    t = packed ^ fx.Uint32(0x80808080)
    nonzero = ((t & fx.Uint32(0x7F7F7F7F)) + fx.Uint32(0x7F7F7F7F)) | t
    return packed & (nonzero | fx.Uint32(0x7F7F7F7F))


@functools.cache
def compile_row_quant_fp8(cols, in_dtype):
    """Launcher ``(x, q, scale, rows, stream)``: x ``in_dtype`` [rows, cols] -> q FP8 [rows, cols], scale f32 [rows].

    ``in_dtype`` is ``"bf16"`` or ``"f32"``; ``cols % 64 == 0``. All-zero rows get scale 1.
    """
    assert cols % 64 == 0, f"cols={cols} must be a multiple of 64"
    esize = _ESIZE[in_dtype]
    lanes = _lanes_per_row(cols, esize)
    per_lane = cols // lanes
    loads = per_lane * esize // 16
    words = per_lane // 4
    pair_store = words % 2 == 0
    rows_per_block = 256 // lanes
    shuffles = [lanes >> (i + 1) for i in range(lanes.bit_length() - 1)]

    @flyc.kernel
    def row_quant_fp8(p_in: fx.Pointer, p_out: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32):
        tid = fx.Int32(gpu.thread_idx.x)
        sub = tid % lanes
        row = fx.Int32(gpu.block_idx.x) * rows_per_block + tid // lanes
        in_rsrc = _buffer_rsrc(p_in, fx.Int64(rows) * (cols * esize))
        out_rsrc = _buffer_rsrc(p_out, fx.Int64(rows) * cols)
        scale_rsrc = _buffer_rsrc(p_scale, fx.Int64(rows) * 4)
        in_base = row * (cols * esize) + sub * (per_lane * esize)
        values = []
        for v in range_constexpr(loads):
            raw = Vec(rocdl_dialect.RawPtrBufferLoadOp(
                ir.VectorType.get([4], fx.Int32.ir_type), in_rsrc,
                (in_base + v * 16).ir_value(), fx.Int32(0).ir_value(),
                aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0)).result).bitcast(fx.Uint32)
            for j in range_constexpr(4):
                word = raw[j]
                if const_expr(in_dtype == "bf16"):
                    values.append((word << 16).bitcast(fx.Float32))
                    values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
                else:
                    values.append(word.bitcast(fx.Float32))
        amax = abs(values[0])
        for j in range_constexpr(1, per_lane):
            amax = amax.maximumf(abs(values[j]))
        for i in range_constexpr(len(shuffles)):
            amax = amax.maximumf(amax.shuffle_xor(shuffles[i], 64))
        scale = amax / 240.0
        scale = (scale == 0.0).select(fx.Float32(1.0), scale)
        out_base = row * cols + sub * per_lane
        _store(scale_rsrc, scale, (sub == 0).select(row * 4, fx.Int32(0x7FFFFFFF)))

        def emit(quotient):
            dwords = []
            for w in range_constexpr(words):
                q = [quotient(values[4 * w + i]).maximumf(fx.Float32(-240.0)).minimumf(fx.Float32(240.0))
                     for i in range_constexpr(4)]
                dwords.append(_pack_fp8x4(q))
            if const_expr(pair_store):
                for w in range_constexpr(words // 2):
                    _store(out_rsrc, Vec.from_elements([dwords[2 * w], dwords[2 * w + 1]], fx.Uint32),
                           out_base + w * 8)
            else:
                for w in range_constexpr(words):
                    _store(out_rsrc, dwords[w], out_base + w * 4)

        if scale >= fx.Float32(_SMALL_SCALE):
            y = 1.0 / scale

            def markstein(x):
                q = x * y
                q = fx.Float32(fx.fma(fx.fma(-q, scale, x), y, q))
                return fx.Float32(fx.fma(fx.fma(-q, scale, x), y, q))

            emit(markstein)
        else:
            emit(lambda x: x / scale)

    @flyc.jit
    def launch(p_in: fx.Pointer, p_out: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32, stream: fx.Stream):
        CompilationContext.get_current()
        row_quant_fp8(p_in, p_out, p_scale, rows).launch(
            grid=((rows + rows_per_block - 1) // rows_per_block, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
