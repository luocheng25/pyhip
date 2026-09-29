"""Query-pair direct attention kernel `pair_qsa_bf16_d256` (reference prototype, not used by qsa()).

A CTA has two waves and two adjacent queries; wave h owns query h. Phase 1 walks
the whole 32-token chunks both queries selected: wave h loads D half h of the
packed K/V and computes that half of QK and PV for both queries, plus the softmax
of its own query. One LDS record per chunk carries the partner's partial scores
of chunk t+1 and P/alpha of chunk t, so QK(t+1) overlaps softmax(t). The waves
then swap O halves, and phase 2 streams each query's private blocks with the
one-wave packed direct loop. Loop loads keep the order K(t+2), block list(t+3),
V(t+1) (prologue: K(1) before V(0)), so the loop head waits only for K(t+1).
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ...mha._common import (
    _buffer, _buffer_words, _exp, _maximum, _min, _pack_bf16, _pin, _read_address, _stage_end, _uniform, _wait,
)
from .._direct_packed import _key_load, _mfma, _pack, _pack_gated, _pv, _qk, _reduce, _rescale, _scale_scores
from ..direct import _load, _output

SLOT = 4096  # one wave's exchange record: 64 lanes x 16 dwords


# ---- Phase 1: one D half per wave, both queries ----

def _load_half(resource, offset, lane, half):
    parts = [_buffer_words(resource, offset + (lane >> 4) * 16 + (4 * half + k) * 64) for k in range(4)]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _key_load_half(resource, offset, lane, hk, half):
    base = offset + (lane >> 4) * 64
    words = []
    for p in range(4):
        part = 4 * half + p
        piece = _buffer_words(resource, base + (part // 2) * hk * 512 + (part % 2) * 256)
        words.extend(piece[i] for i in range(4))
    return fx.Vector.from_elements(words, fx.Int32)


def _value_half(resource, source_base, half, lane, hk):
    base = source_base + half * 2 * hk * 512 + (lane & 15) * 16
    parts = [_buffer_words(resource, base + (part // 2) * hk * 512 + (part % 2) * 256) for part in range(4)]
    return fx.Vector.from_elements([part[i] for part in parts for i in range(4)], fx.Int32)


def _vrow(offset, lane):
    return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, fx.Int32((lane >> 4) * 16).ir_value(), fx.Int32(offset).ir_value()))


def _qk_half(q, k0, k1):
    q = fx.Vector(q).bitcast(fx.Int16)
    keys = (fx.Vector(k0).bitcast(fx.Int16), fx.Vector(k1).bitcast(fx.Int16))
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for step in range(8):
        b = fx.Vector.from_elements([q[step * 4 + i] for i in range(4)], fx.Int16)
        for n in range(2):
            a = fx.Vector.from_elements([keys[n][step * 4 + i] for i in range(4)], fx.Int16)
            acc[n] = _mfma(a, b, acc[n])
    return acc[0], acc[1]


def _pv_half(p, values, output):
    rocdl.s_setprio(2)
    values, output = fx.Vector(values), fx.Vector(output)
    acc = []
    for n in range(8):
        c = fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
        a = fx.Vector.from_elements([values[n * 2 + i] for i in range(2)], fx.Int32).bitcast(fx.Int16)
        acc.append(_mfma(a, p, c))
    rocdl.s_setprio(0)
    return fx.Vector.from_elements([acc[n][i] for n in range(8) for i in range(4)], fx.Float32)


def _scaled(x0, x1, scale):
    return fx.Vector.from_elements([x0[i] for i in range(4)] + [x1[i] for i in range(4)], fx.Float32) * scale


@flyc.jit
def _rescale1(output, alpha):
    # Unconditional rescale was 3% slower; the ballot skips it for the whole wave.
    output = fx.Vector(output)
    changed = rocdl.ballot(fx.Int64.ir_type, (alpha != fx.Float32(1.0)).ir_value())
    if fx.Int64(changed) != 0:
        output = output * alpha
    return output


def _reduce_free(value, maximum):
    # Same expression order as _direct_packed._reduce, without its sched_barrier.
    a = value.shuffle_xor(16, 64)
    b = value.shuffle_xor(32, 64)
    c = value.shuffle_xor(48, 64)
    if maximum:
        return _maximum(_maximum(value, a), _maximum(b, c))
    return (value + a) + (b + c)


def _softmax_free(scores, maximum, total):
    candidate = fx.Float32(-1e30)
    for score in scores:
        candidate = _maximum(candidate, score)
    candidate = _reduce_free(candidate, True)
    updated = _maximum(maximum, candidate)
    alpha = _exp(maximum - updated)
    probabilities = [_exp(value - updated) for value in scores]
    current = fx.Float32(0.0)
    for value in probabilities:
        current = current + value
    current = _reduce_free(current, False)
    total = total * alpha + current
    p0 = _pack_bf16(fx.Vector.from_elements(probabilities[:4], fx.Float32))
    p1 = _pack_bf16(fx.Vector.from_elements(probabilities[4:], fx.Float32))
    return updated, total, alpha, p0, p1


# ---- LDS exchange ----

def _lds_ptr(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_store(address, words):
    llvm.StoreOp(fx.Vector(words).ir_value(), _lds_ptr(address), alignment=16)


def _lds_store1(address, word):
    llvm.StoreOp(fx.Int32(word).ir_value(), _lds_ptr(address), alignment=4)


def _lds_load(address):
    return fx.Vector(llvm.LoadOp(ir.VectorType.get([4], fx.Int32.ir_type), _lds_ptr(address), alignment=16).result)


def _lds_load1(address):
    return fx.Int32(llvm.LoadOp(fx.Int32.ir_type, _lds_ptr(address), alignment=4).result)


def _barrier():
    # Hardware rendezvous plus an IR/MI memory barrier; DS stores are waited
    # explicitly before it (the asm is opaque to SIInsertWaitcnts).
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_barrier", "~{memory}", has_side_effects=True)


def _write(address, immediate, words):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"), [fx.Int32(address).ir_value(), fx.Vector(words).ir_value()],
        f"ds_write_b128 $0, $1 offset:{immediate}", "v,v,~{memory}", has_side_effects=True)


def _write_words(address, words, count):
    words = fx.Vector(words)
    for i in range(count // 4):
        _write(address, i * 1024, fx.Vector.from_elements([words[4 * i + j] for j in range(4)], fx.Int32))


def _read_words(address, count):
    # Inline-asm LDS reads land asynchronously; the compiler believes their
    # registers are defined at the asm. Wait and fence immediately so no dead
    # destination is reused (and no consumer is scheduled) before the data lands.
    parts = [_read_address(address, i * 1024) for i in range(count // 4)]
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


# ---- Selected-token addressing and phase 2 (one wave per query) ----

def _key_offset(cached, n, complete, width, tail_base, extent, hk):
    selector = fx.Int32((_min(n >> 2, fx.Int32(511)) & 63) * 4)
    block = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, selector.ir_value(), fx.Int32(cached).ir_value()))
    base = (n < complete * 4).select(block * (hk * 2048), tail_base)
    return (n < width).select(base + (n & 3) * 16, fx.Int32(extent))


@flyc.jit
def _mask_value_tail(value, tile, segment, lane, width):
    value = fx.Vector(value)
    if tile * 32 + 32 > width:
        first = tile * 32 + segment * 16 + (lane >> 4) * 4
        mask0 = (first < width).select(fx.Int32(65535), fx.Int32(0)) | (first + 1 < width).select(fx.Int32(-65536), fx.Int32(0))
        mask1 = (first + 2 < width).select(fx.Int32(65535), fx.Int32(0)) | (first + 3 < width).select(fx.Int32(-65536), fx.Int32(0))
        value = fx.Vector.from_elements([value[i] & (mask1 if i % 2 else mask0) for i in range(16)], fx.Int32)
    return value


def _value_load(resource, source_base, tile, segment, half, lane, width, hk):
    return _mask_value_tail(_value_half(resource, source_base, half, lane, hk), tile, segment, lane, width)


def _softmax(s0, s1, maximum, total, o0, o1, scale, count, t, lane):
    scaled0, scaled1 = _scale_scores(s0, s1, count, t, lane, scale)
    scores = [scaled0[i] for i in range(4)] + [scaled1[i] for i in range(4)]
    candidate = fx.Float32(-1e30)
    for score in scores:
        candidate = _maximum(candidate, score)
    candidate = _reduce(candidate, True)
    updated = _maximum(maximum, candidate)
    alpha = _exp(maximum - updated)
    probabilities = fx.Vector.from_elements([_exp(value - updated) for value in scores], fx.Float32)
    current = fx.Float32(0.0)
    for i in fx.range_constexpr(8):
        current = current + probabilities[i]
    current = _reduce(current, False)
    total = total * alpha + current
    o0, o1 = _rescale(o0, o1, alpha)
    p0 = _pack_bf16(fx.Vector.from_elements([probabilities[i] for i in range(4)], fx.Float32)).bitcast(fx.Int16)
    p1 = _pack_bf16(fx.Vector.from_elements([probabilities[4 + i] for i in range(4)], fx.Float32)).bitcast(fx.Int16)
    return updated, total, o0, o1, p0, p1


@flyc.jit
def _body(
    Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
    LISTS: fx.Tensor, META: fx.Tensor, COMMON: fx.Tensor, ACTIVE: fx.Tensor, QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NQ: fx.Constexpr[int], NK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, 16384, 16]).peek().view(fx.make_layout(16384, 1))
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    tid = fx.Int32(gpu.thread_id("x"))
    half, lane = _uniform(tid >> 6), tid & 63
    task = fx.Int32(gpu.block_id("x"))
    tile, hkv = task // HK, task % HK
    first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    k0, kv_len = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 3])
    position0 = _uniform(META[tile * 5 + 4])
    chunks = _uniform(COMMON[tile]) >> 3
    qe = (NQ - first) * H * 512 - hkv * (H // HK) * 512
    qptr = fx.get_iter(Q) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    qr = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * 256, 1)), qe)
    head_ok = (lane & 15) < H // HK
    own_query_ok = head_ok & (half < rows)
    oth_query_ok = head_ok & ((1 - half) < rows)
    own_off = own_query_ok.select((half * H + (lane & 15)) * 512, fx.Int32(qe))
    oth_off = oth_query_ok.select(((1 - half) * H + (lane & 15)) * 512, fx.Int32(qe))
    extent = (kv_len * HK - hkv) * 512
    kr = _buffer(fx.make_view(fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * 256, fx.make_layout(NK * HK * 256, 1)), extent)
    vr = _buffer(fx.make_view(fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * 256, fx.make_layout(NK * HK * 256, 1)), extent)
    lists = rocdl.make_buffer_tensor(LISTS)
    scale = fx.Float32(SCALE * math.log2(math.e))
    maximum, total = fx.Float32(-1e30), fx.Float32(0.0)
    o_own = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
    o_oth = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
    own_slot = shared + half * SLOT + lane * 16
    partner_slot = shared + (1 - half) * SLOT + lane * 16

    # ---- Phase 1 prologue: QK(0) + partial-score exchange, K(1) then V(0) ----
    q_own = _load_half(qr, own_off, lane, half)
    q_oth = _load_half(qr, oth_off, lane, half)
    common_width = chunks * 32
    last16 = (((common_width + 15) >> 4) - 1) * 16
    list_row = first * 512
    cached = fx.Int32(lists[list_row + lane])
    _wait(vmcnt=0)
    offset0 = _key_offset(cached, lane & 15, chunks * 8, common_width, fx.Int32(0), extent, HK)
    offset1 = _key_offset(cached, _min(fx.Int32(16), last16) + (lane & 15), chunks * 8, common_width, fx.Int32(0), extent, HK)
    kval0 = _key_load_half(kr, offset0, lane, HK, half)
    kval1 = _key_load_half(kr, offset1, lane, HK, half)
    vrow0, vrow1 = _vrow(offset0, lane), _vrow(offset1, lane)
    offset0 = _key_offset(cached, _min(fx.Int32(32), last16) + (lane & 15), chunks * 8, common_width, fx.Int32(0), extent, HK)
    offset1 = _key_offset(cached, _min(fx.Int32(48), last16) + (lane & 15), chunks * 8, common_width, fx.Int32(0), extent, HK)
    a0, a1 = _qk_half(q_own, kval0, kval1)
    b0, b1 = _qk_half(q_oth, kval0, kval1)
    a, b = _scaled(a0, a1, scale), _scaled(b0, b1, scale)
    _write_words(own_slot + 2 * SLOT, b.bitcast(fx.Int32), 8)
    _wait(lgkmcnt=0)
    _stage_end()
    partial = _read_words(partner_slot + 2 * SLOT, 8)
    s_own = a + partial.bitcast(fx.Float32)
    kval0 = _key_load_half(kr, offset0, lane, HK, half)
    kval1 = _key_load_half(kr, offset1, lane, HK, half)
    rocdl.sched_barrier(0)
    values0 = _value_half(vr, vrow0, half, lane, HK)
    values1 = _value_half(vr, vrow1, half, lane, HK)
    rocdl.sched_barrier(0)
    for t in range(fx.Int32(0), chunks, fx.Int32(1)):
        parity = (t & 1) * (2 * SLOT)
        # Chunk t+2 K offsets (cache window loaded one iteration earlier); the
        # bpermute latency overlaps the QK MFMAs below.
        following0 = _min((t + 2) * 32, last16)
        following1 = _min((t + 2) * 32 + 16, last16)
        offset0n = _key_offset(cached, following0 + (lane & 15), chunks * 8, common_width, fx.Int32(0), extent, HK)
        offset1n = _key_offset(cached, following1 + (lane & 15), chunks * 8, common_width, fx.Int32(0), extent, HK)
        # Segment 1: QK(t+1) (clamped past the end, unused) || softmax(t), o_own rescale.
        a0, a1 = _qk_half(q_own, kval0, kval1)
        b0, b1 = _qk_half(q_oth, kval0, kval1)
        updated, total, alpha_own, p_own0, p_own1 = _softmax_free([s_own[i] for i in range(8)], maximum, total)
        o_own = _rescale1(o_own, alpha_own)
        rocdl.sched_barrier(0)
        # K(t+2) as soon as QK(t+1) has consumed K(t+1); pending order stays
        # K(t+2), cache(t+3), V(t+1) (the prologue has K(1) before V(0)).
        kval0 = _key_load_half(kr, offset0n, lane, HK, half)
        kval1 = _key_load_half(kr, offset1n, lane, HK, half)
        cached = fx.Int32(lists[list_row + _min((t + 3) >> 3, fx.Int32(7)) * 64 + lane])
        rocdl.sched_barrier(0)
        a, b = _scaled(a0, a1, scale), _scaled(b0, b1, scale)
        words = b.bitcast(fx.Int32)
        pw0, pw1 = fx.Vector(p_own0).bitcast(fx.Int32), fx.Vector(p_own1).bitcast(fx.Int32)
        _lds_store(own_slot + parity, fx.Vector.from_elements([words[i] for i in range(4)], fx.Int32))
        _lds_store(own_slot + parity + 1024, fx.Vector.from_elements([words[4 + i] for i in range(4)], fx.Int32))
        _lds_store(own_slot + parity + 2048, fx.Vector.from_elements([pw0[0], pw0[1], pw1[0], pw1[1]], fx.Int32))
        _lds_store1(own_slot + parity + 3072, fx.Vector.from_elements([alpha_own], fx.Float32).bitcast(fx.Int32)[0])
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        _barrier()
        rocdl.sched_barrier(0)
        r0 = _lds_load(partner_slot + parity)
        r1 = _lds_load(partner_slot + parity + 1024)
        r2 = _lds_load(partner_slot + parity + 2048)
        r3 = _lds_load1(partner_slot + parity + 3072)
        # Segment 2a: PV(t) own query (local P) while the record is in flight.
        o_own = _pv_half(p_own0.bitcast(fx.Int16), values0, o_own)
        o_own = _pv_half(p_own1.bitcast(fx.Int16), values1, o_own)
        s_own = a + fx.Vector.from_elements([r0[i] for i in range(4)] + [r1[i] for i in range(4)], fx.Int32).bitcast(fx.Float32)
        alpha_oth = fx.Vector.from_elements([r3], fx.Int32).bitcast(fx.Float32)[0]
        o_oth = _rescale1(o_oth, alpha_oth)
        rocdl.sched_barrier(0)
        # Segment 2b: PV(t) of the partner's query with the received P.
        p_oth0 = fx.Vector.from_elements([r2[0], r2[1]], fx.Int32).bitcast(fx.Int16)
        p_oth1 = fx.Vector.from_elements([r2[2], r2[3]], fx.Int32).bitcast(fx.Int16)
        o_oth = _pv_half(p_oth0, values0, o_oth)
        o_oth = _pv_half(p_oth1, values1, o_oth)
        vrow0, vrow1 = _vrow(offset0, lane), _vrow(offset1, lane)
        rocdl.sched_barrier(0)
        values0 = _value_half(vr, vrow0, half, lane, HK)
        values1 = _value_half(vr, vrow1, half, lane, HK)
        offset0, offset1 = offset0n, offset1n
        maximum = updated
        rocdl.sched_barrier(0)
    _wait(vmcnt=0)
    _stage_end()
    # Swap O halves: wave h sends the partner's query half h, receives its own other half.
    # The opaque x1.0 puts a VALU between the MFMA results and the inline-asm ds_write (no hazard wait for asm).
    one = fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [fx.Float32(1.0).ir_value()], "", "=v,0", has_side_effects=True))
    _write_words(shared + half * 8192 + lane * 16, (fx.Vector(o_oth) * one).bitcast(fx.Int32), 32)
    _wait(lgkmcnt=0)
    _stage_end()
    received_o = _read_words(shared + (1 - half) * 8192 + lane * 16, 32).bitcast(fx.Float32)
    _stage_end()
    is_first = half == 0
    o0 = is_first.select(o_own, received_o)
    o1 = is_first.select(received_o, o_own)

    # ---- Phase 2: this wave's query streams its remaining blocks ----
    wave = half
    valid_wave = wave < rows
    visible = position0 + wave + 1
    row = _min(first + wave, fx.Int32(NQ - 1))
    qoffset = (valid_wave & head_ok).select((wave * H + (lane & 15)) * 512, fx.Int32(qe))
    q = _load(qr, qoffset, lane)
    complete = _min(visible >> 2, fx.Int32(512)) - chunks * 8
    width = complete * 4 + (visible & 3)
    tail_base = (visible & -4) * (HK * 512)
    tiles = valid_wave.select((width + 31) >> 5, fx.Int32(0))
    list_row = row * 512 + chunks * 8
    cached = fx.Int32(lists[list_row + lane])
    _wait(vmcnt=0)
    offset0 = valid_wave.select(_key_offset(cached, lane & 15, complete, width, tail_base, extent, HK), fx.Int32(extent))
    kv0 = _key_load(kr, offset0, lane, HK)
    offset1 = _key_offset(cached, _min(fx.Int32(16), (((width + 15) >> 4) - 1) * 16) + (lane & 15), complete, width, tail_base, extent, HK)
    offset1 = valid_wave.select(offset1, fx.Int32(extent))
    kv1 = _key_load(kr, offset1, lane, HK)
    _wait(vmcnt=0)
    rocdl.sched_barrier(0)
    next_cached = cached
    for t in range(fx.Int32(0), tiles, fx.Int32(1)):
        vrow0, vrow1 = _vrow(offset0, lane), _vrow(offset1, lane)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        values0 = _value_load(vr, vrow0, t, 0, 0, lane, width, HK)
        values1 = _value_load(vr, vrow0, t, 0, 1, lane, width, HK)
        if (((t + 1) & 7) == 0) & (t < 2147483647):
            chunk = _min((t + 1) >> 3, fx.Int32(7)) * 64
            next_cached = fx.Int32(lists[list_row + chunk + lane])
        rocdl.sched_barrier(0)
        _wait(vmcnt=8)
        rocdl.sched_barrier(0)
        s0, s1 = _qk(q, kv0, kv1)
        updated, total, o0, o1, p0, p1 = _softmax(s0, s1, maximum, total, o0, o1, scale, width, t, lane)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        _wait(vmcnt=0)
        rocdl.sched_barrier(0)
        if (((t + 1) & 7) == 0) & (t < 2147483647):
            cached = next_cached
        following0 = _min((t + 1) * 32, (((width + 15) >> 4) - 1) * 16)
        following1 = _min((t + 1) * 32 + 16, (((width + 15) >> 4) - 1) * 16)
        offset0 = _key_offset(cached, following0 + (lane & 15), complete, width, tail_base, extent, HK)
        offset1 = _key_offset(cached, following1 + (lane & 15), complete, width, tail_base, extent, HK)
        next_values0 = _value_load(vr, vrow1, t, 1, 0, lane, width, HK)
        next_values1 = _value_load(vr, vrow1, t, 1, 1, lane, width, HK)
        rocdl.sched_barrier(0)
        o0 = _pv(p0, values0, o0)
        rocdl.sched_barrier(0)
        kv0 = _key_load(kr, offset0, lane, HK)
        rocdl.sched_barrier(0)
        o1 = _pv(p0, values1, o1)
        rocdl.sched_barrier(0)
        kv1 = _key_load(kr, offset1, lane, HK)
        rocdl.sched_barrier(0)
        _wait(vmcnt=16)
        rocdl.sched_barrier(0)
        o0 = _pv(p1, next_values0, o0)
        o1 = _pv(p1, next_values1, o1)
        maximum = updated
    _wait(vmcnt=0)
    _stage_end()
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(O) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = rocdl.make_buffer_tensor(fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), num_records_bytes=qe)
    _output(o0, o1, inv, output, shared, tid, H, H // HK, rows, qe, 128, first, QUERY_TILES, ACTIVE, GATED, NQ)


@flyc.kernel(name="pair_qsa_bf16_d256")
def _kernel(
    Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
    LISTS: fx.Tensor, META: fx.Tensor, GROUPS: fx.Tensor, COMMON: fx.Tensor,
    ACTIVE: fx.Tensor, QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NQ: fx.Constexpr[int], NK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
):
    if fx.const_expr(GATED):
        tile = fx.Int32(gpu.block_id("x")) // HK
        if _uniform(ACTIVE[_uniform(GROUPS[tile * 2])]) == 0:
            _body(Q, K, V, O, LISTS, META, COMMON, ACTIVE, QUERY_TILES, H, HK, NQ, NK, GATED, SCALE)
    else:
        _body(Q, K, V, O, LISTS, META, COMMON, ACTIVE, QUERY_TILES, H, HK, NQ, NK, GATED, SCALE)


@flyc.jit
def _launch(
    Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, PK: fx.Tensor, PV: fx.Tensor,
    LISTS: fx.Tensor, META: fx.Tensor, GROUPS: fx.Tensor, COMMON: fx.Tensor,
    ACTIVE: fx.Tensor, QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], NQ: fx.Constexpr[int], NK: fx.Constexpr[int],
    TASKS: fx.Constexpr[int], GATED: fx.Constexpr[bool], SCALE: fx.Constexpr[float],
    ACTIVE_COUNT: fx.Constexpr[int], stream: fx.Stream,
):
    if fx.const_expr(GATED):
        _pack_gated(K, V, PK, PV, ACTIVE, NK, HK, ACTIVE_COUNT).launch(
            grid=((NK // 4 * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)
    else:
        _pack(K, V, PK, PV, NK, HK).launch(grid=((NK // 4 * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)
    _kernel(
        Q, PK, PV, O, LISTS, META, GROUPS, COMMON, ACTIVE, QUERY_TILES, H, HK, NQ, NK, GATED, SCALE,
        # Packed FP32 ops measured about 2% slower here.
        value_attrs={"llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')},
    ).launch(grid=(TASKS, 1, 1), block=(128, 1, 1), stream=stream)


_COMPILED = {}


def run(inputs, plan, out):
    """Pack K/V and run the pair kernel on the current stream (the pack is skipped when gated and unused)."""
    stream = torch.cuda.current_stream(inputs.q.device)
    args = (
        inputs.q.view(-1), inputs.k.view(-1), inputs.v.view(-1), out.view(-1),
        plan.packed_key.view(-1), plan.packed_value.view(-1),
        plan.lists.view(-1), plan.metadata.view(-1), plan.groups.view(-1), plan.common,
        plan.active_source, plan.query_tiles,
        inputs.q.shape[1], inputs.k.shape[1], inputs.q.shape[0], inputs.k.shape[0],
        plan.num_tiles * inputs.k.shape[1], plan.gated, inputs.scale, plan.active_source.numel(), stream,
    )
    key = (inputs.q.device, tuple(
        (a.dtype, tuple(a.shape)) if isinstance(a, torch.Tensor) else ("stream",) if isinstance(a, torch.cuda.Stream) else a
        for a in args))
    with torch.cuda.device(inputs.q.device):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm the pair kernel before graph capture")
            _COMPILED[key] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
