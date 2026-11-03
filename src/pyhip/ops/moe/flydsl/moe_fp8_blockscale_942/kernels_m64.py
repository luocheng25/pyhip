"""M64 kernels for the TP8 short-K MoE (one 64-row sorted block of one expert per CTA of NW = 8 or 4 waves).

gemm1 (_gemm1_64): the block's 64 rows x 128 intermediate columns per 4 waves; column wave cw owns gate and up
columns 32 cw .. + 31 (N-blocks g0 g1 u0 u1). NW = 8 covers I = 256 in one pass (cw = wave), NW = 4 in two
passes (cw = 4 pass + wave, X streamed twice). X rows (gathered by token) stream through a 2-slot LDS ring of
8 KiB K-blocks (row-XOR-swizzled 16-byte chunks); the wave's W13 fragments come straight from the preshuffled
weights into VGPRs, one K-block ahead. Every 16x16 block's K-block partial is retired two blocks late with its
1x128 x 128x128 scale (FMA) in K order, so the result is bit-identical to kernels._gate_up_body's.

gate_up64: gemm1 + SiLU(gate) * up + 1x128 FP8 requant -> A2 / A2S in memory (as kernels._gate_up_kernel).
moe64 (1-stage): gemm1 + the same requant into LDS (A2 tile [64, 256], row-XOR-swizzled, and its scales),
then the down GEMM in the same CTA: wave w computes output columns N / NW * w .. of the 64 rows with the A2
rows in VGPRs and W2 fragments loaded straight from global (one 32-column unit ahead), retiring and storing
exactly like kernels._down_direct_body (Y rows bit-identical to the 2-stage path).
NW = 8 needs ~220 VGPRs, so one CTA fills a CU; NW = 4 runs two independent CTAs per CU at the same wave count.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl

from .kernels import (BK, FP8_MAX, _DROP, _abs, _as_f32, _attrs, _buffer, _cvt_fp8x4, _dma, _fma, _load_f32,
                      _load_i32, _load_v4, _max, _maybe, _mfma, _min, _operand, _pin_i32, _read_address, _read_frag,
                      _read_lds_f32, _read_scalar_g2r, _retire, _scalar_ready, _schedule, _silu_mul, _stage_end,
                      _store, _store_pair, _uniform, _wait, _write_lds_f32, _xcd_order)

M64 = 64
THREADS64 = 512
XSLOT = M64 * BK          # one K-block of the 64 X rows; slots 0 / 1 (slot 0 later holds A2 half 1)
AMAX = 2 * XSLOT          # per-wave row maxima [8][64] f32
A2SC = AMAX + 8 * M64 * 4  # A2 scales [64][2] f32 (moe64)
A2H0 = A2SC + M64 * 2 * 4  # A2 half 0 [64, 128] (moe64), written while the X ring is still in use
LDS64 = A2H0 + M64 * BK


def _compute_n(w, x, acc, queue, scales, nbase, nblocks):
    """kernels._compute for `nblocks` N-blocks: 4 MFMAs per 16x16 block, each block's K-block partial retired
    (C += p * scale, FMA) during the MFMAs of the block two after it. Returns the last two partials."""
    queue = list(queue)
    for nbl in range(nblocks):
        for mb in range(4):
            part, (pnb, pmb), scale = queue.pop(0)
            p = fx.Vector.filled(4, 0.0, fx.Float32)
            total = [acc[pnb][4 * pmb + e] for e in range(4)]
            for step in range(4):
                p = _mfma(_operand(w[nbl], step), _operand(x[mb], step), p)
                total[step] = _fma(part[step], scale, total[step])
            acc[pnb] = fx.Vector.from_elements([total[i - 4 * pmb] if 4 * pmb <= i < 4 * pmb + 4 else acc[pnb][i]
                                                for i in range(16)], fx.Float32)
            queue.append((p, (nbase + nbl, mb), scales[mb]))
    return queue


def _retire_queue(acc, queue):
    for part, (pnb, pmb), scale in queue:
        acc[pnb] = fx.Vector.from_elements([_fma(part[i - 4 * pmb], scale, acc[pnb][i]) if 4 * pmb <= i < 4 * pmb + 4
                                            else acc[pnb][i] for i in range(16)], fx.Float32)
    return acc


def _dma_x64(ctx, slot, kb):
    """X K-block kb of the 64 rows -> LDS slot: 32 / NW row-pair DMAs per wave."""
    count = len(ctx["dma_v"])
    for i in range(count):
        _dma(ctx["x_res"], ctx["storage"], slot * XSLOT + (count * ctx["wave"] + i) * 256, ctx["dma_v"][i], kb * BK)


def _load_w64(ctx, kb):
    """The wave's W13 fragments of K-block kb: N-blocks g0 g1 u0 u1, two dwordx4 each, straight to VGPRs."""
    return [_load_v4(ctx["w_res"], ctx["w_lane"] + j * 256, ctx["groups"][nb] * (16 * ctx["K"]) + kb * 2048)
            for nb in range(4) for j in range(2)]


def _kstep64(ctx, state, kb, slot, last):
    """K-block kb from LDS slot `slot`; unless last, K-block kb + 1 is prefetched (X DMA into the other slot,
    W13 fragments and scales into registers) and the closing barrier publishes it."""
    acc, queue, wf, xs, wsg, wsu = state
    if not last:
        _dma_x64(ctx, 1 - slot, kb + 1)
        w_next = _load_w64(ctx, kb + 1)
        xs_next = [_load_f32(ctx["xs_res"], ctx["xs_off"][mb], (kb + 1) * (ctx["T"] * 4)) for mb in range(4)]
        wsg_request = _read_scalar_g2r(ctx["WS"], ctx["wsg_row"] + kb + 1)
        wsu_request = _read_scalar_g2r(ctx["WS"], ctx["wsu_row"] + kb + 1)
    x = [_read_frag(ctx["x_read"], slot * XSLOT + mb * 2048) for mb in range(4)]
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    ws_g, ws_u = _as_f32(wsg), _as_f32(wsu)
    queue = _compute_n([[wf[0], wf[1]], [wf[2], wf[3]]], x, acc, queue, [xs[mb] * ws_g for mb in range(4)], 0, 2)
    queue = _compute_n([[wf[4], wf[5]], [wf[6], wf[7]]], x, acc, queue, [xs[mb] * ws_u for mb in range(4)], 2, 2)
    _schedule(16, 0)
    rocdl.sched_barrier(0)
    if last:
        return acc, queue, wf, xs, wsg, wsu
    wsg_next, wsu_next = _scalar_ready(wsg_request), _scalar_ready(wsu_request)
    _wait(vmcnt=0)
    _stage_end()
    return acc, queue, w_next, xs_next, wsg_next, wsu_next


@flyc.jit
def _gemm1_64(X, W, XS, WS, SID, storage, T, row0, expert, cw, I: fx.Constexpr[int], K: fx.Constexpr[int],
              NW: fx.Constexpr[int]):
    """Gate / up accumulators (FP32, block scales applied) of rows row0 .. + 63: acc[nb][4 mb + e] = row
    16 mb + r16, column 32 cw + 16 (nb & 1) + 4 g + e of gate (nb < 2) or up (nb >= 2)."""
    _helpers = (_dma, _read_frag, _load_v4, _load_f32, _load_i32, _compute_n, _retire_queue, _dma_x64, _load_w64,
                _kstep64, _buffer, _pin_i32, _read_scalar_g2r, _scalar_ready, _schedule, _stage_end, _uniform,
                _wait, _as_f32)
    assert I == 256 and NW in (4, 8), "8 column waves x 32 columns"
    KB = K // BK
    PAIRS = 32 // NW
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g, h = lane & 15, lane >> 4, lane >> 5
    x_res = _buffer(X, T * K)
    w_res = _buffer(fx.make_view(fx.get_iter(W) + fx.Int64(expert) * (2 * I * K), fx.make_layout(2 * I * K, 1)),
                    2 * I * K)
    xs_res = _buffer(XS, T * (KB * 4))
    sid_res = _buffer(fx.make_view(fx.get_iter(SID) + fx.Int64(row0), fx.make_layout(M64, 1)), M64 * 4)

    def token_offset(row, unit):
        token = _load_i32(sid_res, row * 4) & 0xFFFFFF
        return (token < T).select(token * unit, fx.Int32(_DROP))

    # DMA i of the wave: rows 2 (PAIRS wave + i) + h; LDS chunk c of row r holds K chunk c ^ (r & 7).
    dma_v = [_pin_i32(token_offset(2 * (PAIRS * wave + i) + h, K)
                      + (((((lane & 31) >> 2) ^ ((2 * i + h) & 7)) << 4) + ((lane & 3) << 2))) for i in range(PAIRS)]
    xs_off = [_pin_i32(token_offset(16 * mb + r16, 4)) for mb in range(4)]
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    x_read = [_pin_i32(shared + r16 * BK + (((2 * g + j) ^ (r16 & 7)) << 4)) for j in range(2)]
    # N-block nb: preshuffled 16-row group (gate 2 cw + nb, up I / 16 + 2 cw + nb - 2), K chunks 2 g + j.
    groups = [2 * cw, 2 * cw + 1, I // 16 + 2 * cw, I // 16 + 2 * cw + 1]
    w_lane = _pin_i32(r16 * 16 + g * 512)
    wsg_row = (expert * (2 * I // 128) + (cw >> 2)) * KB
    ctx = dict(x_res=x_res, storage=storage, wave=wave, dma_v=dma_v, w_res=w_res, w_lane=w_lane, groups=groups,
               K=K, xs_res=xs_res, xs_off=xs_off, T=T, WS=WS, wsg_row=wsg_row, wsu_row=wsg_row + (I // 128) * KB,
               x_read=x_read)

    @flyc.jit
    def pair(a0, a1, a2, a3, p0, s0, p1, s1, f0, f1, f2, f3, f4, f5, f6, f7, x0, x1, x2, x3, wsg, wsu, kb,
             LAST: fx.Constexpr[bool]):
        """K-blocks kb (slot 0) and kb + 1 (slot 1)."""
        kb = fx.Int32(kb)
        state = ([fx.Vector(a) for a in (a0, a1, a2, a3)],
                 [(fx.Vector(p0), (3, 2), fx.Float32(s0)), (fx.Vector(p1), (3, 3), fx.Float32(s1))],
                 [fx.Vector(f) for f in (f0, f1, f2, f3, f4, f5, f6, f7)],
                 [fx.Float32(v) for v in (x0, x1, x2, x3)], fx.Int32(wsg), fx.Int32(wsu))
        state = _kstep64(ctx, state, kb, 0, False)
        state = _kstep64(ctx, state, kb + 1, 1, LAST)
        acc, queue, wf, xs, wsg_, wsu_ = state
        return (*acc, queue[0][0], queue[0][2], queue[1][0], queue[1][2], *wf, *xs, wsg_, wsu_)

    _dma_x64(ctx, 0, fx.Int32(0))
    wf0 = _load_w64(ctx, fx.Int32(0))
    xs0 = [_load_f32(xs_res, xs_off[mb], 0) for mb in range(4)]
    wsg0 = _scalar_ready(_read_scalar_g2r(WS, ctx["wsg_row"]))
    wsu0 = _scalar_ready(_read_scalar_g2r(WS, ctx["wsu_row"]))
    _wait(vmcnt=0)
    _stage_end()
    zero16 = fx.Vector.filled(16, 0.0, fx.Float32)
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (zero16, zero16, zero16, zero16, zero4, fx.Float32(0.0), zero4, fx.Float32(0.0), *wf0, *xs0, wsg0, wsu0)
    for kb in range(fx.Int32(0), fx.Int32(KB - 2), fx.Int32(2)):
        state = pair(*state, kb, False)
    state = pair(*state, fx.Int32(KB - 2), True)
    acc = _retire_queue([fx.Vector(a) for a in state[:4]],
                        [(fx.Vector(state[4]), (3, 2), fx.Float32(state[5])),
                         (fx.Vector(state[6]), (3, 3), fx.Float32(state[7]))])
    return acc[0], acc[1], acc[2], acc[3]


def _act64(acc, shared, wave, lane, r16, g, SW, row0):
    """SiLU(gate) * up of the wave's 32 columns and its 1x128 requant: the 128-column group is physical waves
    4 (wave >> 2) .. + 3, whose row maxima meet in LDS. Returns (codes[mb][nb]: 4 FP8 of row 16 mb + r16,
    columns 32 cw + 16 nb + 4 g .., scale of row `lane` = amax / 240 * route weight)."""
    act = [[_silu_mul(acc[nb][i], acc[2 + nb][i]) for i in range(16)] for nb in range(2)]
    amax = []
    for mb in range(4):
        m = _abs(act[0][4 * mb])
        for i in range(1, 8):
            m = _max(m, _abs(act[i // 4][4 * mb + i % 4]))
        m = _max(m, fx.Float32(m.shuffle_xor(16, 64)))
        amax.append(_max(m, fx.Float32(m.shuffle_xor(32, 64))))
    own = (g == 0).select(amax[0], (g == 1).select(amax[1], (g == 2).select(amax[2], amax[3])))
    _write_lds_f32(shared + AMAX + (wave * M64 + lane) * 4, own)
    _wait(lgkmcnt=0)
    _stage_end()
    group = shared + AMAX + ((wave >> 2) * 4) * (M64 * 4)
    # Inline-asm ds_reads: every value is read before the lgkmcnt wait, used after it.
    parts = [[_read_lds_f32(group + k * (M64 * 4) + (16 * mb + r16) * 4) for k in range(4)] for mb in range(4)]
    parts_own = [_read_lds_f32(group + k * (M64 * 4) + lane * 4) for k in range(4)]
    sw_res = _buffer(fx.make_view(fx.get_iter(SW) + fx.Int64(row0), fx.make_layout(M64, 1)), M64 * 4)
    route = _load_f32(sw_res, lane * 4)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    top = [_max(_max(p[0], p[1]), _max(p[2], p[3])) for p in parts]
    top_own = _max(_max(parts_own[0], parts_own[1]), _max(parts_own[2], parts_own[3]))
    codes = []
    for mb in range(4):
        inv = (top[mb] > fx.Float32(0.0)).select(fx.Float32(FP8_MAX) / top[mb], fx.Float32(0.0))
        codes.append([_cvt_fp8x4([act[nb][4 * mb + e] * inv for e in range(4)]) for nb in range(2)])
    scale = (top_own > fx.Float32(0.0)).select(top_own * fx.Float32(1.0 / FP8_MAX) * route, fx.Float32(0.0))
    return codes, scale


@flyc.jit
def _gate_up64_body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, row0, expert, I: fx.Constexpr[int],
                    K: fx.Constexpr[int], NW: fx.Constexpr[int]):
    _helpers = (_gemm1_64, _act64, _buffer, _store)
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g = lane & 15, lane >> 4
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    a2_res = _buffer(fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * I, fx.make_layout(M64 * I, 1)), M64 * I)
    a2s_res = _buffer(A2S, (I // 128) * R * 4)
    for p in range_constexpr(8 // NW):
        rocdl.sched_barrier(0)
        cw = wave + NW * p
        acc = list(_gemm1_64(X, W, XS, WS, SID, storage, T, row0, expert, cw, I, K, NW))
        codes, scale = _act64(acc, shared, wave, lane, r16, g, SW, row0)
        for mb in range_constexpr(4):
            for nb in range_constexpr(2):
                _store(codes[mb][nb], a2_res, (16 * mb + r16) * I + cw * 32 + nb * 16 + g * 4)
        _store(scale, a2s_res, ((cw & 3) == 0).select(((cw >> 2) * R + row0 + lane) * 4, fx.Int32(_DROP)))


def _write_lds_i32(address, value):
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [fx.Int32(address).ir_value(), fx.Int32(value).ir_value()],
                    "ds_write_b32 $0, $1", "v,v,~{memory}", has_side_effects=True)


def _unit64(wf, x, scales, item, held, y_res, vo, soffset, odd, prefetch):
    """kernels._unit with the next unit's W2 fragments loaded from global (vmem) instead of LDS."""
    rocdl.sched_barrier(0)
    following = prefetch() if prefetch is not None else None
    held = list(held)
    for mb in range_constexpr(4):
        acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)] for _ in range(2)]
        for step in range(4):
            for v in range(2):
                for kb in range(2):
                    acc[v][kb] = _mfma(_operand(wf[v][kb], step), _operand(x[mb][kb], step), acc[v][kb])
        words = _retire(item)
        slot = 3 if mb == 0 else mb - 1
        if (not odd) if mb == 0 else odd:
            _store_pair(held[slot], words, y_res, item[3], item[4])
        else:
            held[slot] = words
        item = (acc, scales[mb][0], scales[mb][1], vo[mb], soffset)
    if prefetch is not None:
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 8, 0)
    for _ in range(64):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
        rocdl.sched_group_barrier(0x002, 3, 0)
    rocdl.sched_barrier(0)
    return item, held, following


@flyc.jit
def _moe64_body(X, W, XS, WS, W2, WS2, SID, SW, Y, storage, T, R, row0, expert, I: fx.Constexpr[int],
                K: fx.Constexpr[int], N: fx.Constexpr[int], NW: fx.Constexpr[int]):
    """Rows row0 .. + 63: gemm1 -> requant into LDS -> down (wave w: columns N / NW * w ..) -> Y rows."""
    _helpers = (_gemm1_64, _act64, _write_lds_i32, _unit64, _retire, _store_pair, _read_address, _read_lds_f32,
                _load_v4, _load_i32, _buffer, _as_f32, _min, _maybe, _pin_i32, _read_scalar_g2r, _scalar_ready,
                _stage_end, _uniform, _wait)
    COLS = N // NW
    assert COLS % 128 == 0
    K2B = I // BK
    ITERS = COLS // 128
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g = lane & 15, lane >> 4
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    # A2 half h (K-block h of the down) [64, 128]: half 0 at A2H0, half 1 in X slot 0 (free once the last pass's
    # X reads finished before its _act64 barrier); row r at 128 r, 16-byte chunk c at c ^ ((r >> 1) & 7).
    for p in range_constexpr(8 // NW):
        rocdl.sched_barrier(0)
        cw = wave + NW * p
        acc = list(_gemm1_64(X, W, XS, WS, SID, storage, T, row0, expert, cw, I, K, NW))
        codes, scale = _act64(acc, shared, wave, lane, r16, g, SW, row0)
        if (cw & 3) == 0:
            _write_lds_f32(shared + A2SC + (lane * 2 + (cw >> 2)) * 4, scale)
        half = shared + ((cw >> 2) == 0).select(fx.Int32(A2H0), fx.Int32(0))
        for mb in range_constexpr(4):
            row = 16 * mb + r16
            for nb in range_constexpr(2):
                _write_lds_i32(half + row * BK + (((2 * (cw & 3) + nb) ^ ((r16 >> 1) & 7)) << 4) + g * 4,
                               codes[mb][nb])
    _wait(lgkmcnt=0)
    _stage_end()

    # ---- down: A2 rows 16 mb + r16 (K chunks 8 kb + 2 g + j) and their scales to VGPRs --------------------
    x = [[[_read_address(_pin_i32(shared + (A2H0 if kb == 0 else 0) + (16 * mb + r16) * BK
                                  + (((2 * g + j) ^ ((r16 >> 1) & 7)) << 4)))
           for j in range(2)] for kb in range(K2B)] for mb in range(4)]
    xs = [[_read_lds_f32(shared + A2SC + ((16 * mb + r16) * 2 + kb) * 4) for kb in range(K2B)] for mb in range(4)]
    col0 = wave * COLS
    w2_res = _buffer(fx.make_view(fx.get_iter(W2) + fx.Int64(expert) * (N * I), fx.make_layout(N * I, 1)), N * I)
    # MFMA v of a 32-column unit takes unit row 8 (r16 >> 2) + 4 v + (r16 & 3) as its A row r16 (lane (r16, g)
    # then holds 8 consecutive output columns): preshuffled 16-row group u >> 4, row u & 15, K chunk 2 g + j.
    unit_row = [8 * (r16 >> 2) + 4 * v + (r16 & 3) for v in range(2)]
    w2_lane = [[_pin_i32((unit_row[v] >> 4) * (16 * I) + (unit_row[v] & 15) * 16 + (2 * g + j) * 256) for j in range(2)]
               for v in range(2)]

    def load_unit(column):
        return [[[_load_v4(w2_res, w2_lane[v][j], (column >> 4) * (16 * I) + kb * 2048) for j in range(2)]
                 for kb in range(K2B)] for v in range(2)]

    sid_res = _buffer(fx.make_view(fx.get_iter(SID) + fx.Int64(row0), fx.make_layout(M64, 1)), M64 * 4)
    tokens = [_load_i32(sid_res, (16 * mb + r16) * 4) & 0xFFFFFF for mb in range(4)]
    y_res = _buffer(fx.make_view(fx.get_iter(Y) + fx.Int64(row0) * N, fx.make_layout(M64 * N, 1)), M64 * N * 2)
    ws_row = (expert * (N // 128) + col0 // 128) * K2B
    requests = [_read_scalar_g2r(WS2, ws_row + kb) for kb in range(K2B)]
    wf_first = load_unit(col0)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    ws_first = [_scalar_ready(r) for r in requests]
    vo = [_pin_i32((tokens[mb] < T).select(((16 * mb + r16) * N + 8 * g) * 2, fx.Int32(_DROP))) for mb in range(4)]

    @flyc.jit
    def pair2(f0, f1, f2, f3, f4, f5, f6, f7, p0, p1, p2, p3, s0, s1, item_vo, h0, h1, h2, h3, ws0, ws1, it,
              LAST: fx.Constexpr[bool]):
        """Columns col0 + 128 it .. + 127 (one W2 block scale) as units U0-U3 (see _down_direct_body.pair)."""
        it = fx.Int32(it)
        wf = [[[fx.Vector(f0), fx.Vector(f1)], [fx.Vector(f2), fx.Vector(f3)]],
              [[fx.Vector(f4), fx.Vector(f5)], [fx.Vector(f6), fx.Vector(f7)]]]
        column = col0 + it * 128
        cbase = column * 2
        item = ([[fx.Vector(p0), fx.Vector(p1)], [fx.Vector(p2), fx.Vector(p3)]], fx.Float32(s0),
                fx.Float32(s1), fx.Int32(item_vo), cbase - 64)
        held = [None, None, None, [fx.Int32(h) for h in (h0, h1, h2, h3)]]
        ws = (_as_f32(ws0), _as_f32(ws1))
        scales = [[fx.Float32(xs[mb][kb]) * ws[kb] for kb in range(K2B)] for mb in range(4)]
        following = _min(it + 1, fx.Int32(ITERS - 1))
        requests = [_read_scalar_g2r(WS2, ws_row + following * K2B + kb) for kb in range(K2B)]
        item, held, wf = _unit64(wf, x, scales, item, held, y_res, vo, cbase, False, lambda: load_unit(column + 32))
        item, held, wf = _unit64(wf, x, scales, item, held, y_res, vo, cbase + 64, True,
                                 lambda: load_unit(column + 64))
        item, held, wf = _unit64(wf, x, scales, item, held, y_res, vo, cbase + 128, False,
                                 lambda: load_unit(column + 96))
        item, held, following_wf = _unit64(wf, x, scales, item, held, y_res, vo, cbase + 192, True,
                                           _maybe(not LAST, lambda: load_unit(column + 128)))
        ws_next = [_scalar_ready(r) for r in requests]
        wf = wf if following_wf is None else following_wf
        parts = item[0]
        return (*[wf[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)],
                parts[0][0], parts[0][1], parts[1][0], parts[1][1], item[1], item[2], vo[3], *held[3], *ws_next)

    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (*[wf_first[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)], zero, zero, zero, zero,
             fx.Float32(0.0), fx.Float32(0.0), fx.Int32(_DROP), *[fx.Int32(0)] * 4, *ws_first)
    for it in range(fx.Int32(0), fx.Int32(ITERS - 1), fx.Int32(1)):
        state = pair2(*state, it, False)
    state = pair2(*state, fx.Int32(ITERS - 1), True)
    last = ([[state[8], state[9]], [state[10], state[11]]], state[12], state[13])
    _store_pair([fx.Int32(h) for h in state[15:19]], _retire(last), y_res, vo[3], (col0 + COLS - 32) * 2)


@flyc.kernel(known_block_size=[THREADS64, 1, 1])
def _gate_up64_kernel(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, SID: fx.Tensor, SEID: fx.Tensor,
                      SW: fx.Tensor, NV: fx.Tensor, A2: fx.Tensor, A2S: fx.Tensor, T: fx.Int32, R: fx.Int32,
                      I: fx.Constexpr[int], K: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS64, 16]).peek().view(fx.make_layout(LDS64, 1))
    block = fx.Int32(gpu.block_id("x"))
    blocks = _scalar_ready(_read_scalar_g2r(NV, 0)) // M64
    if block < blocks:
        b = _xcd_order(block, blocks)
        expert = _scalar_ready(_read_scalar_g2r(SEID, b))
        _gate_up64_body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, b * M64, expert, I, K, 8)


@flyc.kernel(known_block_size=[THREADS64 // 2, 1, 1])
def _gate_up64x4_kernel(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, SID: fx.Tensor, SEID: fx.Tensor,
                        SW: fx.Tensor, NV: fx.Tensor, A2: fx.Tensor, A2S: fx.Tensor, T: fx.Int32, R: fx.Int32,
                        I: fx.Constexpr[int], K: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS64, 16]).peek().view(fx.make_layout(LDS64, 1))
    block = fx.Int32(gpu.block_id("x"))
    blocks = _scalar_ready(_read_scalar_g2r(NV, 0)) // M64
    if block < blocks:
        b = _xcd_order(block, blocks)
        expert = _scalar_ready(_read_scalar_g2r(SEID, b))
        _gate_up64_body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, b * M64, expert, I, K, 4)


@flyc.kernel(known_block_size=[THREADS64, 1, 1])
def _moe64_kernel(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, W2: fx.Tensor, WS2: fx.Tensor,
                  SID: fx.Tensor, SEID: fx.Tensor, SW: fx.Tensor, NV: fx.Tensor, Y: fx.Tensor, T: fx.Int32,
                  R: fx.Int32, I: fx.Constexpr[int], K: fx.Constexpr[int], N: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS64, 16]).peek().view(fx.make_layout(LDS64, 1))
    block = fx.Int32(gpu.block_id("x"))
    blocks = _scalar_ready(_read_scalar_g2r(NV, 0)) // M64
    if block < blocks:
        b = _xcd_order(block, blocks)
        expert = _scalar_ready(_read_scalar_g2r(SEID, b))
        _moe64_body(X, W, XS, WS, W2, WS2, SID, SW, Y, storage, T, R, b * M64, expert, I, K, N, 8)


@flyc.kernel(known_block_size=[THREADS64 // 2, 1, 1])
def _moe64x4_kernel(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, W2: fx.Tensor, WS2: fx.Tensor,
                    SID: fx.Tensor, SEID: fx.Tensor, SW: fx.Tensor, NV: fx.Tensor, Y: fx.Tensor, T: fx.Int32,
                    R: fx.Int32, I: fx.Constexpr[int], K: fx.Constexpr[int], N: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS64, 16]).peek().view(fx.make_layout(LDS64, 1))
    block = fx.Int32(gpu.block_id("x"))
    blocks = _scalar_ready(_read_scalar_g2r(NV, 0)) // M64
    if block < blocks:
        b = _xcd_order(block, blocks)
        expert = _scalar_ready(_read_scalar_g2r(SEID, b))
        _moe64_body(X, W, XS, WS, W2, WS2, SID, SW, Y, storage, T, R, b * M64, expert, I, K, N, 4)


@flyc.jit
def launch_gate_up64(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, SID: fx.Tensor, SEID: fx.Tensor,
                     SW: fx.Tensor, NV: fx.Tensor, A2: fx.Tensor, A2S: fx.Tensor, T: fx.Int32, R: fx.Int32,
                     GRID: fx.Int32, I: fx.Constexpr[int], K: fx.Constexpr[int], NW: fx.Constexpr[int],
                     stream: fx.Stream):
    kernel = _gate_up64_kernel if NW == 8 else _gate_up64x4_kernel
    kernel(X, W, XS, WS, SID, SEID, SW, NV, A2, A2S, T, R, I, K, value_attrs=_attrs()).launch(
        grid=(GRID, 1, 1), block=(64 * NW, 1, 1), stream=stream)


@flyc.jit
def launch_moe64(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, W2: fx.Tensor, WS2: fx.Tensor,
                 SID: fx.Tensor, SEID: fx.Tensor, SW: fx.Tensor, NV: fx.Tensor, Y: fx.Tensor, T: fx.Int32,
                 R: fx.Int32, GRID: fx.Int32, I: fx.Constexpr[int], K: fx.Constexpr[int], N: fx.Constexpr[int],
                 NW: fx.Constexpr[int], stream: fx.Stream):
    kernel = _moe64_kernel if NW == 8 else _moe64x4_kernel
    kernel(X, W, XS, WS, W2, WS2, SID, SEID, SW, NV, Y, T, R, I, K, N, value_attrs=_attrs()).launch(
        grid=(GRID, 1, 1), block=(64 * NW, 1, 1), stream=stream)
