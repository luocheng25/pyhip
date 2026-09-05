"""OPUS-style gfx950 BF16 D192/V128 attention with the paged-prefill ABI.

SHUFFLE-5D page64 KV is adapted to linear KV by a separate FlyDSL gather. The
attention kernel uses OPUS's Q/V LDS alias, padded K/V layouts, two compile-time
wave-group bodies and eight-stage ping/pong loop. Gather runs on every call and
must be included when timing the public interface; no attention data is cached.
"""

import functools
import math

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm


BM, BN, DQ, DV, THREADS = 256, 64, 192, 128, 512
K_TILE = 8 * 3 * 520
V_TILE = 8 * 2 * 544
Q_TILE = 32 * 3 * 520
LDS_ELEMENTS = 2 * K_TILE + max(Q_TILE, 2 * V_TILE)
LOG2E = math.log2(math.e)


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _exp(value):
    value = fx.Float32(value)
    return fx.Float32(llvm.call_intrinsic(
        fx.Float32.ir_type, "llvm.amdgcn.exp2.f32", [value.ir_value()], [], []
    ))


def _scale_sub(value, scale, maximum):
    value, scale, maximum = fx.Float32(value), fx.Float32(scale), fx.Float32(maximum)
    # Keep scalar FMA: packed FP32 VALU cannot co-issue with this MFMA schedule.
    return fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [value.ir_value(), scale.ir_value(), maximum.ir_value()],
        "v_fma_f32 $0, $1, $2, -$3", "=v,v,v,v", has_side_effects=False
    ))


def _pin_i32(value):
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [fx.Int32(value).ir_value()], "", "=v,0", has_side_effects=True
    ))


def _pin_f32(value):
    return fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [fx.Float32(value).ir_value()], "", "=v,0", has_side_effects=True
    ))


def _slice(values, begin, count, dtype=fx.Float32):
    return fx.Vector.from_elements([values[begin + i] for i in range(count)], dtype)


def _join(left, right, count=16, dtype=fx.Float32):
    return fx.Vector.from_elements(
        [left[i] for i in range(count)] + [right[i] for i in range(count)], dtype
    )


def _pin(values, chunk=8):
    # Chunked tied operands avoid demanding one contiguous 32/64-VGPR block.
    values = fx.Vector(values)
    dtype = values.dtype
    if dtype.width < 32:
        values = values.bitcast(fx.Int32)
    chunks = []
    for start in range(0, values.numel, chunk):
        part = _slice(values, start, min(chunk, values.numel - start), values.dtype)
        part = fx.Vector(llvm.inline_asm(
            part.ir_value().type, [part.ir_value()], "", "=v,0", has_side_effects=True
        ))
        chunks.extend(part[i] for i in range(part.numel))
    return fx.Vector.from_elements(chunks, values.dtype).bitcast(dtype)


def _cross32(value, maximum=False):
    pair = rocdl.permlane32_swap(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        value.bitcast(fx.Int32).ir_value(), value.bitcast(fx.Int32).ir_value(), False, True
    )
    a = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [0])).bitcast(fx.Float32)
    b = fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pair, [1])).bitcast(fx.Float32)
    return a.maximumf(b) if maximum else a + b


def _row_max(scores):
    scores = fx.Vector(scores)
    result = fx.Float32(-1.0e30)
    for i in range(32):
        result = result.maximumf(scores[i])
    return _cross32(result, True)


def _row_sum(scores):
    scores = fx.Vector(scores)
    result = fx.Float32(0.0)
    for i in range(32):
        result = result + scores[i]
    return _cross32(result)


def _exp_slice(scores, start, end):
    scores = fx.Vector(scores)
    return fx.Vector.from_elements(
        [_exp(scores[i]) if start <= i < end else scores[i] for i in range(32)], fx.Float32
    )


def _center_slice(scores, scale, maximum, start, end):
    scores = fx.Vector(scores)
    return _pin(fx.Vector.from_elements(
        [_pin_f32(_scale_sub(scores[i], scale, maximum)) if start <= i < end else scores[i]
         for i in range(32)], fx.Float32
    ), 32)


def _stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _schedule(pairs, valu, group, exp=False):
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, valu, group)


def _dma(resource, storage, destination, voffset, soffset):
    rocdl.raw_ptr_buffer_load_lds(
        resource, fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(16), voffset, soffset, fx.Int32(0), fx.Int32(0)
    )


def _read_operand(storage, base, immediate=0, transpose=False):
    # The copy atoms currently model all direct-LDS DMA as one aliasing store:
    # they insert vmcnt(0) even when the phase reads a different ring buffer.
    # Match OPUS's explicit LDS-read boundary; the caller owns lgkmcnt(0) and
    # the cross-wave stage barrier. No loaded value may be consumed before it.
    # ds_read has a 16-bit byte immediate; Q's third D chunk exceeds it.
    high = (immediate * 2) & ~0xFFFF
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base + high // 2))
    width = 2 if transpose else 4
    instruction = "ds_read_b64_tr_b16" if transpose else "ds_read_b128"
    return fx.Vector(llvm.inline_asm(
        ir.VectorType.get([width], fx.Int32.ir_type), [address.ir_value()],
        f"{instruction} $0, $1 offset:{(immediate * 2) & 0xFFFF}", "=v,v,~{memory}",
        has_side_effects=True,
    )).bitcast(fx.BFloat16)


def _read_qk(storage, base, q=False):
    result = fx.make_rmem_tensor(fx.make_layout((8, 12), (1, 8)), fx.BFloat16)
    chunk_stride = 16448 if q else 4160
    for k in range(12):
        result[None, k].store(_read_operand(storage, base, (k // 4) * chunk_stride + (k % 4) * 16))
    return result


def _read_v(storage, base):
    result = fx.make_rmem_tensor(fx.make_layout((4, 16), (1, 4)), fx.BFloat16)
    for i in range(16):
        result[None, i].store(_read_operand(storage, base, (i % 8) * 64 + (i // 8) * 32, True))
    return fx.make_view(fx.get_iter(result), fx.make_layout((8, 4, 2), (1, 8, 32)))


def _qk(q, k):
    atom = fx.make_mma_atom(rocdl.MFMA(32, 32, 16, fx.BFloat16))
    acc = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float32)
    acc.fill(0.0)
    for i in range(12):
        fx.mma_atom_call(atom, acc, k[None, i], q[None, i], acc)
    return acc.load()


def _pv(p, v, o0, o1):
    atom = fx.make_mma_atom(rocdl.MFMA(32, 32, 16, fx.BFloat16))
    probs = fx.make_rmem_tensor(fx.make_layout((8, 4), (1, 8)), fx.BFloat16)
    probs.store(p)
    acc = fx.make_rmem_tensor(fx.make_layout((16, 2), (1, 16)), fx.Float32)
    acc.store(_join(o0, o1))
    # OPUS's tiled MMA is K-major: alternate independent output accumulators.
    for k in range(4):
        for n in range(2):
            c = acc[None, n]
            fx.mma_atom_call(atom, c, v[None, k, n], probs[None, k], c)
    return acc[None, 0].load(), acc[None, 1].load()


@flyc.jit
def _mask(scores, tile, wave_row, kv_len, q_len, causal: fx.Constexpr[bool]):
    scores, tile = fx.Vector(scores), fx.Int32(tile)
    lane_half = (fx.Int32(gpu.thread_id("x")) & 32) // 8
    if fx.const_expr(causal):
        if wave_row + kv_len - q_len < (tile + 1) * BN:
            mask_lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
            boundary = _pin_i32(kv_len - q_len + wave_row + (mask_lane & 31)
                                - tile * 64 - (mask_lane & 32) // 8)
            scores = fx.Vector.from_elements([
                (boundary >= (i // 16) * 32 + ((i % 16) // 4) * 8 + i % 4).select(
                    scores[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    else:
        if ((kv_len & 63) != 0) & (tile == (kv_len - 1) // 64):
            # Materialize inside the tail branch. Otherwise LICM keeps 32 wave
            # predicates (64 SGPRs) live throughout the entire main loop.
            border_limit = _pin_i32(kv_len - 1 - tile * 64 - lane_half)
            scores = fx.Vector.from_elements([
                (border_limit >= (i // 16) * 32 + ((i % 16) // 4) * 8 + i % 4).select(
                    scores[i], fx.Float32(float("-inf")))
                for i in range(32)
            ], fx.Float32)
    return scores


@flyc.jit
def _rescale(o0, o1, o2, o3, old_max, new_max, row_sum, mask):
    o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
    old_max, new_max, row_sum = fx.Float32(old_max), fx.Float32(new_max), fx.Float32(row_sum)
    if mask != fx.Int64(0):
        correction = _exp(old_max - new_max)
        o0 = o0 * correction
        o1 = o1 * correction
        o2 = o2 * correction
        o3 = o3 * correction
        row_sum = row_sum * correction
    return o0, o1, o2, o3, row_sum


@flyc.jit
def _attention_body(
    Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, qb,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], CAP: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int],
    OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool],
    REVERSE: fx.Constexpr[bool] = False, MERGED: fx.Constexpr[bool] = False,
):
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    if fx.const_expr(MERGED):
        lane = _pin_i32(lane)
    q_start = qb * BM
    wave_row = q_start + wave * 32
    row = wave_row + (lane & 31)
    q_valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    q_ptr = fx.get_iter(Q) + (fx.Int64(q0) + q_start) * QROW + head * QHEAD
    q_view = fx.make_view(q_ptr, fx.make_layout((BM, DQ), (QROW, 1)))
    q_buf = rocdl.make_buffer_tensor(q_view, num_records_bytes=q_valid * QROW * 2)
    k_ptr = fx.get_iter(K) + (fx.Int64(batch) * CAP * HK + hkv) * DQ
    v_ptr = fx.get_iter(V) + (fx.Int64(batch) * CAP * HK + hkv) * DV
    k_buf = rocdl.make_buffer_tensor(
        fx.make_view(k_ptr, fx.make_layout((CAP, DQ), (HK * DQ, 1))),
        num_records_bytes=kv_len * HK * DQ * 2,
    )
    v_buf = rocdl.make_buffer_tensor(
        fx.make_view(v_ptr, fx.make_layout((CAP, DV), (HK * DV, 1))),
        num_records_bytes=kv_len * HK * DV * 2,
    )
    gq = rocdl.get_buffer_rsrc(fx.get_iter(q_buf))
    gk = rocdl.get_buffer_rsrc(fx.get_iter(k_buf))
    gv = rocdl.get_buffer_rsrc(fx.get_iter(v_buf))
    scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
    if fx.const_expr(MERGED):
        scale = _uniform(scale.bitcast(fx.Int32)).bitcast(fx.Float32)
    if fx.const_expr(PER_TOKEN):
        qs_buf = rocdl.make_buffer_tensor(QS, max_size=False)
        scale = scale * qs_buf[(q0 + row) * H + head]
    else:
        scale = scale * QS[0]
    scale = fx.Float32(llvm.inline_asm(
        fx.Float32.ir_type, [fx.Float32(scale).ir_value()], "", "=v,0", has_side_effects=True
    ))
    rocdl.sched_barrier(0)

    total_tiles = (kv_len + 63) // 64
    tiles = total_tiles
    if fx.const_expr(CAUSAL):
        tiles = _max(_min((q_start + q_valid + kv_len - q_len + 63) // 64, total_tiles), fx.Int32(1))
    last = tiles - 1
    local_g = wave + (lane >> 3) * 8
    gq_off = (local_g * QROW + (lane & 7) * 8) * 2
    gk_off = (local_g * HK * DQ + (lane & 7) * 8) * 2
    gv_off = (local_g * HK * DV + (lane & 7) * 8) * 2
    rk = (lane & 7) * 520 + ((lane >> 3) & 3) * 64 + (lane >> 5) * 8
    rq = (lane & 7) * 2056 + wave * 256 + ((lane >> 3) & 3) * 64 + (lane >> 5) * 8
    rv = (lane >> 5) * 2176 + ((lane >> 2) & 3) * 544 + ((lane >> 4) & 1) * 16 + (lane & 3) * 4

    def data_tile(position):
        return last - position if REVERSE else position

    def load_k(tile, slot, begin=0, end=3):
        offset = data_tile(_min(tile, last)) * (64 * HK * DQ * 2)
        for d in fx.range_constexpr(begin, end):
            _dma(gk, storage, slot * K_TILE + (d * 8 + wave) * 520,
                 gk_off + d * 128, offset)

    def load_v(tile, slot):
        offset = data_tile(_min(tile, last)) * (64 * HK * DV * 2)
        for d in fx.range_constexpr(2):
            _dma(gv, storage, 2 * K_TILE + slot * V_TILE + (d * 8 + wave) * 544,
                 gv_off + d * 128, offset)

    # Q aliases the entire V region until Q has been read into registers.
    for d in fx.range_constexpr(3):
        for n in fx.range_constexpr(4):
            _dma(gq, storage, 2 * K_TILE + (d * 8 + wave) * 2056 + n * 512,
                 gq_off + d * 128 + n * 64 * QROW * 2, fx.Int32(0))
    load_k(fx.Int32(0), 0)
    rocdl.s_waitcnt(vmcnt=3)
    _stage_end()
    q_frag = _read_qk(storage, 2 * K_TILE + rq, True)
    load_k(fx.Int32(1), 1)
    rocdl.s_waitcnt(lgkmcnt=0, vmcnt=3)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()

    k_frag = _read_qk(storage, rk)
    load_v(fx.Int32(0), 0)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    s0 = _qk(q_frag, k_frag)
    o0 = fx.Vector.filled(16, 0.0, fx.Float32)
    o1 = fx.Vector.filled(16, 0.0, fx.Float32)
    o2 = fx.Vector.filled(16, 0.0, fx.Float32)
    o3 = fx.Vector.filled(16, 0.0, fx.Float32)
    _schedule(12, 3, 5)
    o0, o1, o2, o3 = _pin(o0), _pin(o1), _pin(o2), _pin(o3)
    _stage_end()
    k_frag = _read_qk(storage, rk + 256)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    s1 = _qk(q_frag, k_frag)
    scores = _mask(_join(s0, s1), data_tile(fx.Int32(0)), wave_row, kv_len, q_len, CAUSAL)
    maximum = (_row_max(scores) * scale).maximumf(fx.Float32(-1.0e30))
    scores = _center_slice(scores, scale, maximum, 0, 32)
    row_sum = fx.Float32(0.0)
    rocdl.s_waitcnt(vmcnt=2)
    _stage_end()
    load_k(fx.Int32(2), 0)
    _stage_end()

    @flyc.jit
    def phase(previous, maximum, row_sum, o0, o1, o2, o3, t, CUR: fx.Constexpr[int], PREV: fx.Constexpr[int]):
        previous = fx.Vector(previous)
        o0, o1, o2, o3 = fx.Vector(o0), fx.Vector(o1), fx.Vector(o2), fx.Vector(o3)
        maximum, row_sum, t = fx.Float32(maximum), fx.Float32(row_sum), fx.Int32(t)
        # S0/S1: memory of t, then QK(t) + exp(t-1).
        k = _read_qk(storage, CUR * K_TILE + rk)
        if fx.const_expr(STAGGER):
            load_v(t, CUR)
        rocdl.s_waitcnt(lgkmcnt=0)
        _stage_end()
        lo = _qk(q_frag, k)
        previous = _pin(_exp_slice(previous, 0, 24), 32)
        if fx.const_expr(STAGGER):
            _schedule(1, 3, 1, True)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 3, 1)
            _schedule(8, 3, 1, True)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 1)
        else:
            rocdl.sched_group_barrier(rocdl.mask_mfma, 3, 1)
            _schedule(1, 2, 1, True)
            _schedule(8, 3, 1, True)
        _stage_end()
        # S2/S3: second QK half + remaining exp, sum, BF16 P.
        k = _read_qk(storage, CUR * K_TILE + rk + 256)
        if fx.const_expr(not STAGGER):
            load_v(t, CUR)
        rocdl.s_waitcnt(lgkmcnt=0, vmcnt=5)
        _stage_end()
        hi = _qk(q_frag, k)
        previous = _exp_slice(previous, 24, 32)
        row_sum = _pin_f32(row_sum + _row_sum(previous))
        p = _pin(previous.to(fx.BFloat16), 16)
        _schedule(2, 3, 2, True)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 2)
        rocdl.sched_group_barrier(0x400, 2, 2)
        rocdl.sched_group_barrier(0x002, 2, 2)
        _schedule(5, 6, 2)
        _schedule(1, 5, 2)
        _schedule(2, 6, 2)
        _schedule(1, 2, 2)
        _stage_end()
        current = _join(lo, hi)
        # S4/S5: V(t-1), K(t+2), then PV + current softmax head.
        v = _read_v(storage, 2 * K_TILE + PREV * V_TILE + rv)
        load_k(t + 2, CUR, 0, 1 if STAGGER else 3)
        current = _mask(current, data_tile(t), wave_row, kv_len, q_len, CAUSAL)
        rocdl.s_waitcnt(lgkmcnt=0)
        _stage_end()
        o0, o1 = _pv(p, v, o0, o1)
        candidate = _row_max(current) * scale
        mask = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (candidate - maximum > 8.0).ir_value()))
        new_max = _pin_f32((mask != fx.Int64(0)).select(maximum.maximumf(candidate), maximum))
        split = 12 if STAGGER else 6
        current = _center_slice(current, scale, new_max, 0, split)
        for count in (5, 5, 6, 4, 2, 4, 4, 4):
            _schedule(1, count, 3)
        _stage_end()
        # S6/S7: other V half, then finish PV and conditional rescale.
        v = _read_v(storage, 2 * K_TILE + PREV * V_TILE + V_TILE // 2 + rv)
        if fx.const_expr(STAGGER):
            load_k(t + 2, CUR, 1, 3)
        rocdl.s_waitcnt(lgkmcnt=0, vmcnt=5)
        _stage_end()
        o2, o3 = _pv(p, v, o2, o3)
        current = _center_slice(current, scale, new_max, split, 32)
        if fx.const_expr(STAGGER):
            for count in (3, 2, 2, 2, 3, 3, 4, 4):
                _schedule(1, count, 4)
        else:
            _schedule(2, 3, 4)
            _schedule(5, 4, 4)
        o0, o1, o2, o3, row_sum = _rescale(o0, o1, o2, o3, maximum, new_max, row_sum, mask)
        _stage_end()
        return current, new_max, row_sum, o0, o1, o2, o3

    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        rocdl.sched_barrier(0)
        scores, maximum, row_sum, o0, o1, o2, o3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, t, 1, 0)
        rocdl.sched_barrier(0)
        scores, maximum, row_sum, o0, o1, o2, o3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, t + 1, 0, 1)
        rocdl.sched_barrier(0)
    rocdl.sched_barrier(0)
    if (tiles & 1) == 0:
        scores, maximum, row_sum, o0, o1, o2, o3 = phase(scores, maximum, row_sum, o0, o1, o2, o3, tiles - 1, 1, 0)

    maximum, row_sum = fx.Float32(maximum), fx.Float32(row_sum)
    rocdl.s_waitcnt(vmcnt=3)
    _stage_end()
    scores = _exp_slice(scores, 0, 32)
    row_sum = row_sum + _row_sum(scores)
    p = _pin(scores.to(fx.BFloat16))
    _stage_end()
    last_slot = (tiles - 1) & 1
    v = _read_v(storage, 2 * K_TILE + last_slot * V_TILE + rv)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o0, o1 = _pv(p, v, o0, o1)
    _stage_end()
    v = _read_v(storage, 2 * K_TILE + last_slot * V_TILE + V_TILE // 2 + rv)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o2, o3 = _pv(p, v, o2, o3)
    rocdl.sched_barrier(0)
    if fx.const_expr(not STAGGER):
        rocdl.s_barrier()

    out_tid = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [tid.ir_value()], "", "=v,0", has_side_effects=True))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    o_ptr = fx.get_iter(O) + (fx.Int64(q0) + q_start) * OROW + head * OHEAD
    o_buf = rocdl.make_buffer_tensor(fx.make_view(o_ptr, fx.make_layout((BM, DV), (OROW, 1))), num_records_bytes=q_valid * OROW * 2)
    out_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    outputs = (o0, o1, o2, o3)
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    for g in fx.range_constexpr(8):
        vals = outputs[g // 2]
        begin = (g & 1) * 8
        packed = [rocdl.cvt_pk_bf16_f32(vals[begin + i] * inv, vals[begin + i + 1] * inv) for i in range(0, 8, 2)]
        pairs = [rocdl.permlane32_swap(pair_type, packed[i], packed[i + 2], False, True) for i in range(2)]
        words = fx.Vector.from_elements([
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, pairs[i & 1], [i // 2])) for i in range(4)
        ], fx.Int32)
        src = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        src.store(words.bitcast(fx.BFloat16))
        offset = out_row * OROW + ((out_tid >> 5) & 1) * 8 + g * 16
        dst = fx.make_view(fx.get_iter(o_buf) + offset, fx.make_layout(8, 1))
        fx.copy(out_atom, src, dst)
    if fx.const_expr(WITH_LSE):
        if (out_tid & 63) < 32:
            if out_row < q_valid:
                log_l = fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []))
                lse = (row_sum > 0.0).select((maximum + log_l) * fx.Float32(math.log(2.0)), fx.Float32(float("-inf")))
                LSE[(q0 + q_start + out_row) * H + head] = lse


@flyc.jit
def _attention_sequence(
    Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, qb,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], CAP: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], STAGGER: fx.Constexpr[bool], MERGE: fx.Constexpr[bool],
):
    _attention_body(Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, qb,
                   H, HK, CAP, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, STAGGER,
                   False, MERGE)
    if fx.const_expr(MERGE):
        mirror = (q_len + BM - 1) // BM - 1 - qb
        if mirror > qb:
            # OPUS pairs the shortest and longest causal query blocks. The
            # mirror scans backward so both passes balance work and reuse L2.
            _attention_body(Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, mirror,
                           H, HK, CAP, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, STAGGER,
                           True, MERGE)


@flyc.jit
def _zero_tile(O, LSE, q0, q_len, head, qb,
               H: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int], WITH_LSE: fx.Constexpr[bool]):
    tid = fx.Int32(gpu.thread_id("x"))
    valid = _min(q_len - qb * BM, fx.Int32(BM))
    ptr = fx.get_iter(O) + (fx.Int64(q0) + qb * BM) * OROW + head * OHEAD
    buf = rocdl.make_buffer_tensor(
        fx.make_view(ptr, fx.make_layout((BM, DV), (OROW, 1))),
        num_records_bytes=valid * OROW * 2,
    )
    zeros = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
    zeros.fill(0.0)
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    for i in fx.range_constexpr(8):
        index = tid * 8 + i * THREADS * 8
        offset = (index // DV) * OROW + index % DV
        dst = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(8, 1))
        fx.copy(atom, zeros, dst)
    if fx.const_expr(WITH_LSE):
        if (tid < BM) & (qb * BM + tid < q_len):
            LSE[(q0 + qb * BM + tid) * H + head] = fx.Float32(float("-inf"))


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, KI: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], CAP: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float], MERGE: fx.Constexpr[bool]):
    head = fx.Int32(gpu.block_id("x"))
    batch = fx.Int32(gpu.block_id("y"))
    qb = fx.Int32(gpu.block_id("z"))
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    pages = _uniform(KI[batch + 1]) - _uniform(KI[batch])
    kv_len = (pages > 0).select((pages - 1) * 64 + _uniform(LAST[batch]), fx.Int32(0))
    storage = fx.SharedAllocator().allocate(fx.Array[fx.BFloat16, LDS_ELEMENTS, 16]).peek().view(fx.make_layout(LDS_ELEMENTS, 1))
    q_blocks = (q_len + BM - 1) // BM
    work_blocks = (q_blocks + 1) // 2 if MERGE else q_blocks
    if qb < work_blocks:
        if kv_len > 0:
            group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
            if group != 0:
                _attention_sequence(Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, qb, H, HK, CAP, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, True, MERGE)
            else:
                _attention_sequence(Q, K, V, O, LSE, QS, KS, VS, storage, q0, q_len, kv_len, batch, head, qb, H, HK, CAP, QROW, QHEAD, OROW, OHEAD, PER_TOKEN, CAUSAL, WITH_LSE, SCALE, False, MERGE)
        else:
            _zero_tile(O, LSE, q0, q_len, head, qb, H, OROW, OHEAD, WITH_LSE)
            if fx.const_expr(MERGE):
                mirror = q_blocks - 1 - qb
                if mirror > qb:
                    _zero_tile(O, LSE, q0, q_len, head, mirror, H, OROW, OHEAD, WITH_LSE)


@flyc.kernel(known_block_size=[128, 1, 1])
def _gather_kernel(K: fx.Tensor, V: fx.Tensor, KL: fx.Tensor, VL: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor,
                   HK: fx.Constexpr[int], CAP: fx.Constexpr[int]):
    tid = fx.Int32(gpu.thread_id("x"))
    block = fx.Int32(gpu.block_id("x"))
    ph, part = block // 4, block % 4
    batch = fx.Int32(gpu.block_id("y"))
    logical, head = ph // HK, ph % HK
    start = KI[batch]
    if logical < KI[batch + 1] - start:
        page = _uniform(PAGES[start + logical])
        kb = (fx.Int64(page) * HK + head) * (64 * DQ)
        vb = (fx.Int64(page) * HK + head) * (64 * DV)
        ko = (fx.Int64(batch) * CAP + logical * 64) * HK * DQ + head * DQ
        vo = (fx.Int64(batch) * CAP + logical * 64) * HK * DV + head * DV
        copy = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
        # Four 16-token blocks per page expose enough parallelism at short KV.
        for i in fx.range_constexpr(3):
            item = tid + i * 128
            token, d8 = part * 16 + item // 24, item % 24
            src = fx.make_view(fx.get_iter(K) + kb + d8 * 512 + token * 8, fx.make_layout(8, 1))
            dst = fx.make_view(fx.get_iter(KL) + ko + token * HK * DQ + d8 * 8, fx.make_layout(8, 1))
            reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            fx.copy(copy, src, reg)
            fx.copy(copy, reg, dst)
        # Each lane reads eight contiguous tokens for one V dimension, then
        # scatters to coalesced output rows; two vector reads replace 16 scalars.
        for group in fx.range_constexpr(2):
            token = part * 16 + group * 8
            src = fx.make_view(fx.get_iter(V) + vb + (token // 8) * 1024 + tid * 8, fx.make_layout(8, 1))
            reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            fx.copy(copy, src, reg)
            vals = reg.load()
            for i in fx.range_constexpr(8):
                VL[vo + (token + i) * HK * DV + tid] = vals[i]


@flyc.jit
def _launch_gather(K: fx.Tensor, V: fx.Tensor, KL: fx.Tensor, VL: fx.Tensor, KI: fx.Tensor, PAGES: fx.Tensor,
                   HK: fx.Constexpr[int], CAP: fx.Constexpr[int], B: fx.Constexpr[int], stream: fx.Stream):
    _gather_kernel(K, V, KL, VL, KI, PAGES, HK, CAP).launch(grid=(CAP // 16 * HK, B, 1), block=(128, 1, 1), stream=stream)


@flyc.jit
def _launch_attention(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, LSE: fx.Tensor,
    CQ: fx.Tensor, CK: fx.Tensor, KI: fx.Tensor, LAST: fx.Tensor, QS: fx.Tensor, KS: fx.Tensor, VS: fx.Tensor,
    H: fx.Constexpr[int], HK: fx.Constexpr[int], CAP: fx.Constexpr[int], B: fx.Constexpr[int], MAX_Q: fx.Constexpr[int],
    QROW: fx.Constexpr[int], QHEAD: fx.Constexpr[int], OROW: fx.Constexpr[int], OHEAD: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool], CAUSAL: fx.Constexpr[bool], WITH_LSE: fx.Constexpr[bool], SCALE: fx.Constexpr[float], stream: fx.Stream):
    query_blocks = (MAX_Q + BM - 1) // BM
    merge = CAUSAL and query_blocks * H * B >= 512
    _attention_kernel(Q, K, V, O, LSE, CQ, CK, KI, LAST, QS, KS, VS, H, HK, CAP, QROW, QHEAD, OROW, OHEAD,
        PER_TOKEN, CAUSAL, WITH_LSE, SCALE, merge,
        value_attrs={"rocdl.waves_per_eu": 2},
    ).launch(grid=(H, B, (query_blocks + 1) // 2 if merge else query_blocks), block=(THREADS, 1, 1), stream=stream)


def _flat(tensor):
    """View storage, preserving the real Q/O row and head strides separately."""
    if tensor.numel() == 0:
        return tensor.reshape(-1)
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((extent,), (1,))


class _PagedAttention:
    bf16_backend = "native-8wave"

    def __init__(self, heads, kv_heads, causal, quant_query_mode):
        self.heads, self.kv_heads, self.causal = heads, kv_heads, causal
        self.quant_query_mode = quant_query_mode
        self._compiled, self._workspace = {}, {}

    def _run(self, fn, args):
        signature = tuple((x.dtype, tuple(x.shape), tuple(x.stride())) if isinstance(x, torch.Tensor)
                          else ("stream",) if hasattr(x, "cuda_stream") else x for x in args)
        key = (fn, args[0].device, signature)
        compiled = self._compiled.get(key)
        with torch.cuda.device(args[0].device):
            if compiled is None:
                # flyc.compile traces, compiles AND performs the first launch.
                self._compiled[key] = flyc.compile(fn, *args)
            else:
                compiled(*args)

    def prepare_kv(self, K, V, kv_indptr, kv_page_indices, max_seqlen_k, stream=None):
        stream = torch.cuda.current_stream(K.device) if stream is None else stream
        batch = kv_indptr.numel() - 1
        cap = max(64, math.ceil(max_seqlen_k / 64) * 64)
        key = (K.device, stream.cuda_stream, batch, cap)
        if key not in self._workspace:
            with torch.cuda.stream(stream):
                self._workspace[key] = (
                    torch.empty((batch, cap, self.kv_heads, DQ), device=K.device, dtype=torch.bfloat16),
                    torch.empty((batch, cap, self.kv_heads, DV), device=K.device, dtype=torch.bfloat16),
                )
        kl, vl = self._workspace[key]
        self._run(_launch_gather, (_flat(K), _flat(V), _flat(kl), _flat(vl), kv_indptr, kv_page_indices,
                                  self.kv_heads, cap, batch, stream))
        return kl, vl

    def attend_linear(self, Q, KL, VL, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_last_page_lens,
                      q_descale, k_descale, v_descale, max_seqlen_q, out, lse=None, stream=None, softmax_scale=None):
        stream = torch.cuda.current_stream(Q.device) if stream is None else stream
        per_token = q_descale.numel() != 1
        self._run(_launch_attention, (
            _flat(Q), _flat(KL), _flat(VL), _flat(out), _flat(lse) if lse is not None else k_descale.reshape(-1),
            cu_seqlens_q, cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
            kv_indptr, kv_last_page_lens, _flat(q_descale), k_descale.reshape(-1), v_descale.reshape(-1),
            self.heads, self.kv_heads, KL.shape[1], KL.shape[0], max_seqlen_q,
            Q.stride(0), Q.stride(1), out.stride(0), out.stride(1),
            per_token, self.causal, lse is not None, float(1 / math.sqrt(DQ) if softmax_scale is None else softmax_scale), stream,
        ))
        return out

    def __call__(self, Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                 max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
                 kv_last_page_lens, out=None, sink_ptr=None, stream=None, *, return_lse=False, lse=None, softmax_scale=None):
        """Run paged attention; descales must be finite and strictly positive.

        GPU metadata values (page IDs, prefix sums, last-page lengths and the
        supplied maximum lengths) must be consistent. They are not copied to
        the CPU, so the hot path remains asynchronous and graph-capturable.
        """
        if not Q.is_cuda or K.device != Q.device or V.device != Q.device:
            raise ValueError("Q/K/V must be on the same GPU")
        if "gfx950" not in torch.cuda.get_device_properties(Q.device).gcnArchName:
            raise NotImplementedError("this kernel requires gfx950")
        if Q.dtype != torch.bfloat16 or K.dtype != Q.dtype or V.dtype != Q.dtype:
            raise NotImplementedError("only BF16 Q/K/V are supported")
        if causal != self.causal or sink_ptr is not None:
            raise ValueError("causal must match the factory; attention sinks are unsupported")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, DQ) or Q.stride(-1) != 1:
            raise ValueError("Q must be [tokens, heads, 192] with contiguous head dimension")
        if K.ndim != 5 or V.ndim != 5 or K.shape != (V.shape[0], self.kv_heads, 24, 64, 8) or V.shape[1:] != (self.kv_heads, 8, 128, 8):
            raise ValueError("K/V must use page64 SHUFFLE-5D layouts")
        if not K.is_contiguous() or not V.is_contiguous():
            raise ValueError("paged K/V storage must be contiguous")
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("sequence-length bounds must be nonnegative")
        if max_seqlen_k * self.kv_heads * DQ * 2 >= 2**31:
            raise NotImplementedError("linear KV spans must fit the signed 32-bit buffer offset")
        if softmax_scale is not None and (not math.isfinite(softmax_scale) or softmax_scale <= 0):
            raise ValueError("softmax_scale must be finite and positive")
        batch = cu_seqlens_q.numel() - 1
        if batch < 1 or kv_indptr.numel() != batch + 1 or kv_last_page_lens.numel() != batch:
            raise ValueError("inconsistent batch metadata")
        metadata = (cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens)
        if cu_seqlens_k is not None:
            if cu_seqlens_k.numel() != batch + 1:
                raise ValueError("inconsistent KV prefix length")
            metadata += (cu_seqlens_k,)
        if any(t.ndim != 1 or t.dtype != torch.int32 or t.device != Q.device or not t.is_contiguous() for t in metadata):
            raise ValueError("metadata must be contiguous device int32 tensors")
        if k_descale.numel() != 1 or v_descale.numel() != 1 or q_descale.numel() not in (1, Q.shape[0] * self.heads):
            raise ValueError("expected scalar K/V descales and scalar or per-token/head Q descales")
        if any(t.dtype != torch.float32 or t.device != Q.device or not t.is_contiguous() for t in (q_descale, k_descale, v_descale)):
            raise ValueError("descales must be contiguous device FP32 tensors")
        stream = torch.cuda.current_stream(Q.device) if stream is None else stream
        if stream.device != Q.device:
            raise ValueError("stream must belong to the input GPU")
        with torch.cuda.stream(stream):
            if out is None:
                out = torch.empty((Q.shape[0], self.heads, DV), dtype=torch.bfloat16, device=Q.device)
            if return_lse and lse is None:
                lse = torch.empty((Q.shape[0], self.heads), dtype=torch.float32, device=Q.device)
        if out.shape != (Q.shape[0], self.heads, DV) or out.dtype != torch.bfloat16 or out.device != Q.device or out.stride(-1) != 1:
            raise ValueError("invalid output buffer")
        if lse is not None and (lse.shape != (Q.shape[0], self.heads) or lse.dtype != torch.float32 or lse.device != Q.device or not lse.is_contiguous()):
            raise ValueError("LSE must be contiguous FP32 [tokens, heads]")
        if Q.shape[0] > 0:
            kl, vl = self.prepare_kv(K, V, kv_indptr, kv_page_indices, max_seqlen_k, stream)
            self.attend_linear(Q, kl, vl, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_last_page_lens,
                               q_descale, k_descale, v_descale, max_seqlen_q, out, lse, stream, softmax_scale)
        return (out, lse) if return_lse else out


@functools.cache
def PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
                   is_causal, quant_query_mode="per-token", key_layout="vectorized",
                   window_left=-1, has_sink=False):
    if (head_dim_qk, head_dim_v, page_size, key_layout, window_left, has_sink) != (DQ, DV, BN, "vectorized", -1, False):
        raise NotImplementedError("only gfx950 BF16 D192/V128 page64 full attention is supported")
    if num_kv_heads <= 0 or num_qo_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("unsupported query scale mode")
    return _PagedAttention(num_qo_heads, num_kv_heads, is_causal, quant_query_mode)