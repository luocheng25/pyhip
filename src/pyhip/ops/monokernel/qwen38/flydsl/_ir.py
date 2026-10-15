"""Low-level building blocks of the MonoKernel stages (gfx942, wave64).

Buffers are raw descriptors addressed in bytes: loads past ``num_bytes`` return zeros and
stores past it are dropped, which bounds the live rows of fixed-size buffers. MFMA helpers
take one lane's 16-byte words in the layout of AITER's 16x16 weight shuffle: lane ``l`` of a
1 KiB wave load holds row ``l % 16`` and 16-byte chunk ``l // 16`` of a 16-row tile.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import rocdl

from pyhip.codegen.flydsl.helpers import rocdl_aux

LOG2E = 1.4426950408889634
FP8_MAX = 240.0
# Byte offset outside every descriptor.
DROP = 0x40000000
HIGH_HALF = 0xFFFF0000
# DPP controls: quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_ror:4, row_ror:8.
DPP_XOR1, DPP_XOR2, DPP_ROR4, DPP_ROR8 = 0xB1, 0x4E, 0x124, 0x128
# ds_swizzle bit mode: lane ^ 16 inside 32-lane halves.
SWIZZLE_XOR16 = 0x401F


def scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def scalar_f32(value):
    return scalar(value.bitcast(fx.Int32)).bitcast(fx.Float32)


def buffer(tensor, num_bytes, element_offset=None):
    """Raw buffer descriptor over ``tensor`` (optionally from an element offset) of ``num_bytes``."""
    base = fx.get_iter(tensor) if element_offset is None else fx.get_iter(tensor) + element_offset
    address = fx.Int64(fx.ptrtoint(base))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer, fx.Int16(0).ir_value(),
                                  fx.Int64(num_bytes).ir_value(), fx.Int32(0x27000).ir_value())


def load(rsrc, offset, words=4, aux=0):
    result_type = fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type)
    value = rocdl.raw_ptr_buffer_load(result_type, rsrc, fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(),
                                      aux=rocdl_aux(aux))
    return fx.Int32(value) if words == 1 else fx.Vector(value)


def raw(value):
    """MLIR value of an fx numeric/vector or of a bare arith result."""
    return value.ir_value() if hasattr(value, "ir_value") else fx.arith.unwrap(value)


def store(rsrc, offset, value, aux=0):
    rocdl.raw_ptr_buffer_store(raw(value), rsrc, fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(),
                               aux=rocdl_aux(aux))


def store_short(rsrc, offset, bits, aux=0):
    """Store the low 16 bits of an i32."""
    value = arith.trunci(ir.IntegerType.get_signless(16), fx.Int32(bits).ir_value())
    rocdl.raw_ptr_buffer_store(value, rsrc, fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), aux=rocdl_aux(aux))


def words(values, dtype=fx.Int32):
    return fx.Vector.from_elements(list(values), dtype)


def lds_base(storage, num_words):
    """Byte address of an LDS ``fx.Array[fx.Int32, num_words]`` allocation."""
    return fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(num_words, 1)))))


def _lds(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def lds_load(address, words=4):
    if words == 1:
        return fx.Int32(llvm.load(fx.Int32.ir_type, _lds(address), alignment=4))
    return fx.Vector(llvm.load(ir.VectorType.get([words], fx.Int32.ir_type), _lds(address), alignment=4 * words))


def lds_store(address, value, words=4):
    llvm.store(raw(value), _lds(address), alignment=4 * words)


def barrier():
    fx.gpu.barrier()


# ------------------------------------------------------------------ MFMA


def mfma_fp8(a, b, acc):
    """16x16x32 E4M3FNUZ MFMA; ``a``/``b`` are i64 (8 FP8 values of one lane)."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_fp8_fp8(ir.VectorType.get([4], fx.Float32.ir_type),
                                                     [a.ir_value(), b.ir_value(), acc.ir_value(), 0, 0, 0]))


def mfma_bf16(a, b, acc):
    """16x16x16 BF16 MFMA; ``a``/``b`` are 4 BF16 values (v4i16) of one lane."""
    return fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(ir.VectorType.get([4], fx.Float32.ir_type),
                                                    [a.ir_value(), b.ir_value(), acc.ir_value(), 0, 0, 0]))


def i64_half(value, half):
    """Bytes [8 * half, 8 * half + 8) of a 16-byte word vector as one i64."""
    return words((value[2 * half], value[2 * half + 1])).bitcast(fx.Int64)[0]


def bf16x4_half(value, half):
    """BF16 elements [4 * half, 4 * half + 4) of a 16-byte word vector as v4i16."""
    return words((value[2 * half], value[2 * half + 1])).bitcast(fx.Int16)


def zeros4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


# ------------------------------------------------------------- conversion


def fp8_pair(word, high):
    """Two E4M3FNUZ bytes (0-1, or 2-3 when ``high``) of a DWORD as an FP32 pair."""
    asm = "v_cvt_pk_f32_fp8_sdwa $0, $1 src0_sel:WORD_1" if high else "v_cvt_pk_f32_fp8 $0, $1"
    return fx.Vector(llvm.inline_asm(ir.VectorType.get([2], fx.Float32.ir_type), [fx.Int32(word).ir_value()],
                                     asm, "=v,v", has_side_effects=False))


def fp8_to_bf16_words(word):
    """Four FP8 values of a DWORD as two DWORDs of BF16 pairs (exact: E4M3FNUZ fits BF16)."""
    out = []
    for high in (False, True):
        pair = fp8_pair(word, high).bitcast(fx.Uint32)
        out.append(pack_high(pair[1], pair[0]).bitcast(fx.Int32))
    return out


def fp8_pack(a, b, c, d):
    """Four FP32 values (already within +-240) as one DWORD of E4M3FNUZ, round to nearest even."""
    low = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, a.ir_value(), b.ir_value(), fx.Int32(0).ir_value(), False)
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, c.ir_value(), d.ir_value(), low, True))


def bf16_pair(word):
    """One DWORD of two BF16 values as an FP32 pair (low element first)."""
    bits = fx.Int32(word).bitcast(fx.Uint32)
    return ((bits << 16).bitcast(fx.Float32), (bits & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32))


def rne_bits(value):
    """FP32 bits plus the BF16 round-to-nearest-even bias; the high half is the BF16 result."""
    bits = value.bitcast(fx.Uint32)
    return bits + 0x7FFF + ((bits >> 16) & 1)


def bf16_round(value):
    """FP32 rounded to BF16 (round to nearest even), kept as FP32."""
    return (rne_bits(value) & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32)


def pack_high(hi_bits, lo_bits):
    """(hi_bits[31:16] << 16) | lo_bits[31:16]."""
    return fx.Uint32(llvm.call_intrinsic(fx.Uint32.ir_type, "llvm.amdgcn.perm",
                                         [hi_bits.ir_value(), lo_bits.ir_value(), fx.Uint32(0x07060302).ir_value()],
                                         [], []))


def bf16_word(lo, hi):
    """Two FP32 values rounded to BF16 (nearest even) and packed into one DWORD."""
    return pack_high(rne_bits(hi), rne_bits(lo)).bitcast(fx.Int32)


# ------------------------------------------------------------------- math


def exp2(value):
    return fx.Float32(rocdl.exp2(fx.Float32.ir_type, value.ir_value()))


def rcp(value):
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, value.ir_value()))


def sigmoid(value):
    return rcp(1.0 + exp2(value * (-LOG2E)))


def silu(value):
    return value * sigmoid(value)


# Torch-like FP32 math (LLVM's accurate expansions; IEEE division) for per-head scalars.


def exp_accurate(value):
    return fx.Float32(fx.math.exp(value))


def sqrt_accurate(value):
    return fx.Float32(fx.math.sqrt(value))


def sigmoid_accurate(value):
    return fx.Float32(1.0) / (1.0 + exp_accurate(fx.Float32(0.0) - value))


def softplus(value):
    """``F.softplus`` (beta 1, threshold 20); log1p by Kahan's ``log(u) * y / (u - 1)``."""
    y = exp_accurate(value)
    u = 1.0 + y
    log1p = (u == 1.0).select(y, fx.Float32(fx.math.log(u)) * (y / (u - 1.0)))
    return (value > 20.0).select(value, log1p)


def fmax(a, b):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.maxnum.f32", [a.ir_value(), b.ir_value()],
                                          [], []))


def fmin(a, b):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.minnum.f32", [a.ir_value(), b.ir_value()],
                                          [], []))


def fabs(value):
    return (value.bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF)).bitcast(fx.Float32)


# ---------------------------------------------------------- packed FP32 (v_pk_*_f32)


def f32x2(a, b):
    return fx.Vector.from_elements([a, b], fx.Float32)


def bf16x2_f32(word):
    """One DWORD of two BF16 values as a packed FP32 pair (low element first)."""
    bits = fx.Int32(word)
    return fx.Vector.from_elements([bits << 16, bits & fx.Int32(-65536)], fx.Int32).bitcast(fx.Float32)


def fma2(a, b, c):
    """Packed ``a * b + c`` on FP32 pairs (single rounding)."""
    v2 = ir.VectorType.get([2], fx.Float32.ir_type)
    return fx.Vector(llvm.call_intrinsic(v2, "llvm.fma.v2f32", [raw(a), raw(b), raw(c)], [], []))


def splat2(value):
    return f32x2(value, value)


# ------------------------------------------------------------- lane exchange


def dpp(value, ctrl):
    """Full-mask DPP move of an FP32 value."""
    i32 = fx.Int32.ir_type
    bound_ctrl = fx.arith.unwrap(fx.arith.constant(True, type=ir.IntegerType.get_signless(1)))
    moved = llvm.call_intrinsic(i32, "llvm.amdgcn.update.dpp.i32", [
        llvm.mlir_poison(i32), value.bitcast(fx.Int32).ir_value(), fx.Int32(ctrl).ir_value(),
        fx.Int32(0xF).ir_value(), fx.Int32(0xF).ir_value(), bound_ctrl], [], [])
    return fx.Int32(moved).bitcast(fx.Float32)


def _swap16(value):
    swapped = rocdl.ds_swizzle(fx.Int32.ir_type, value.bitcast(fx.Int32).ir_value(),
                               fx.Int32(SWIZZLE_XOR16).ir_value())
    return fx.Int32(swapped).bitcast(fx.Float32)


def _swap32(value, lane):
    permuted = rocdl.ds_bpermute(fx.Int32.ir_type, ((lane ^ 32) * 4).ir_value(), value.bitcast(fx.Int32).ir_value())
    return fx.Int32(permuted).bitcast(fx.Float32)


def wave_reduce(value, lane, op):
    """``op``-reduction of an FP32 value over all 64 lanes; every lane gets the result."""
    for ctrl in (DPP_XOR1, DPP_XOR2, DPP_ROR4, DPP_ROR8):
        value = op(value, dpp(value, ctrl))
    value = op(value, _swap16(value))
    return op(value, _swap32(value, lane))


def wave_sum(value, lane):
    return wave_reduce(value, lane, lambda a, b: a + b)


def wave_max(value, lane):
    return wave_reduce(value, lane, fmax)
