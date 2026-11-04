"""gfx942 FP8 block-scale MoE kernels (prefill): gate-up GEMM with SiLU/FP8 requant, down GEMM, top-k sum.

Both GEMMs run gemm_fp8_blockscale_8w_942's 256x256 8-wave staged Memory/Compute pipeline (16x16x32 MFMA,
four 16 KiB LDS slots, dword LDS-DMA, 1x128 / 128x128 block scales retired two blocks late). Rows are
aiter's moe_sorting order (experts padded to 64 rows on the short-K path, else 256; padding rows carry token
id T), cut by _tiles_kernel into compact tiles of one expert: 256 rows, plus one 64-192-row tail per expert.
Rows of a tile past its row count are masked; their 64-row bands (gate_up) / waves (short-K down) issue no
MFMAs.

gate_up (_gate_up_body), per (M-block, 128 intermediate columns jt):
    X rows are gathered: lane l of `table` holds the byte offset of DMA row 128 (l >> 5) + 2 q + 8 ((l & 31) >> 1)
    + (l & 1); every buffer_load_dword ... lds v_readlane's its two rows. The W tile is [gate 128 jt .. + 127 |
    up I + 128 jt .. + 127] (P1 = gate, P2 = up), so wave (wm, wn) holds gate (N-blocks 0-3) and up (4-7) of
    the same 64 columns and C1 / C2 use the gate / up weight scale. Epilogue: SiLU(gate) * up, row maxima
    over the 128-column group (two waves, exchanged in LDS), FP8 E4M3FNUZ requant ->
    A2 [R, I] (sorted rows; staged in LDS so every store writes whole 128-byte rows) and
    A2S [I / 128, R] = amax / 240 * route weight.
down: I = 256 (_down_direct_body): 4-wave CTAs, each wave keeps 64 A2 rows (whole K) in registers while W2
    streams through a 2-slot LDS ring in 64-row chunks; MFMA A rows are permuted so each lane's outputs are 8
    consecutive columns, partials retire during the next M-block's MFMAs, and even/odd 32-column units are
    stored back to back (full 128-byte lines). I >= 512: gemm_fp8_blockscale_8w_942._body unchanged on
    (A2 rows, W2[expert]). The route weight rides in A2S, so Y rows are already weighted. Y [R, H] in
    sorted order; padding rows are not written by the short-K path.
inverse / sum: inv[t, k] = sorted row of route (t, k); out[t] = BF16(sum_k Y[inv[t, k]]) in FP32, k order,
    one CTA per token, streaming (nt) loads of Y.

Grids cover the worst-case block count; blocks past the device-side valid count exit, the rest run in
XCD-chunked order (block b runs on XCD b % 8, which gets a contiguous run of M-blocks, i.e. of experts).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl

from pyhip.codegen.flydsl.helpers import rocdl_aux
from pyhip.ops.gemm.flydsl.gemm_fp8_blockscale_8w_942 import (
    BK, BM, BN, LDS_BYTES, P1_SLOT, P2_SLOT, SLOT, THREADS, _DROP, _as_f32, _body, _buffer, _compute, _dma,
    _fma, _mfma, _min, _operand, _pack_bf16, _pin_i32, _read_address, _read_frag, _read_scalar_g2r,
    _scalar_ready, _schedule, _stage_end, _uniform, _wait)

FP8_MAX = 240.0
SUM_THREADS = 256
TILE_THREADS = 256
# Short-K down (K <= 256): 4-wave CTAs, two per CU; W2 streams through a ring of two 64-row LDS chunks.
DOWN_THREADS = 256
DOWN_CHUNK = 64
DOWN_COLS = 1024


def _load_i32(resource, voffset, soffset=0):
    return fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, resource, fx.Int32(voffset).ir_value(),
                                              fx.Int32(soffset).ir_value(), aux=0))


def _load_f32(resource, voffset, soffset=0):
    return fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, resource, fx.Int32(voffset).ir_value(),
                                                fx.Int32(soffset).ir_value(), aux=0))


def _store(value, resource, voffset):
    rocdl.raw_ptr_buffer_store(value.ir_value(), resource, fx.Int32(voffset).ir_value(), fx.Int32(0).ir_value())


def _readlane(value, lane):
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, fx.Int32(value).ir_value(), lane))


def _write_lds_f32(address, value):
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [fx.Int32(address).ir_value(), fx.Float32(value).ir_value()],
                    "ds_write_b32 $0, $1", "v,v,~{memory}", has_side_effects=True)


def _read_lds_f32(address):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [fx.Int32(address).ir_value()], "ds_read_b32 $0, $1",
                                      "=v,v,~{memory}", has_side_effects=True))


def _dma_x_gather(resource, storage, q, table, chunk, h, half, kb):
    """X slot <half>, K-block kb: DMA t covers sorted rows 128 half + 2 q + 8 t + h (table lanes 32 half + 2 t + h)."""
    # Pinned per call so the loop-invariant readlanes stay next to their DMAs instead of 64 SGPRs.
    table = _pin_i32(table)
    for t in range(16):
        lo = _readlane(table, 32 * half + 2 * t)
        hi = _readlane(table, 32 * half + 2 * t + 1)
        _dma(resource, storage, half * SLOT + q * 256 + t * 1024, (h == 0).select(lo, hi) + chunk, kb * BK)


def _dma_w13(resource, storage, q, lane4, packet, kb, base16, K):
    """Preshuffled W13 row groups base16 .. + 7 (P1 gate / P2 up) of K-block kb: group t / 2 at 2 KiB, 16-byte
    chunk q + 4 (t & 1) at 256 bytes (the dense kernel's preshuffled slot layout)."""
    slot = P1_SLOT if packet == 0 else P2_SLOT
    for t in range(16):
        _dma(resource, storage, slot + q * 256 + t * 1024, lane4,
             (base16 + (t >> 1)) * (16 * K) + (kb * 8 + q + 4 * (t & 1)) * 256)


def _load_xs_gather(resource, offsets, soffset):
    return [_load_f32(resource, offsets[mb], soffset) for mb in range(4)]


def _silu_mul(gate, up):
    e = fx.Float32(rocdl.exp2(fx.Float32.ir_type, (gate * fx.Float32(-1.4426950408889634)).ir_value()))
    return gate * fx.Float32(rocdl.rcp(fx.Float32.ir_type, (e + fx.Float32(1.0)).ir_value())) * up


def _max(a, b):
    return (a > b).select(a, b)


def _abs(a):
    return (a < fx.Float32(0.0)).select(fx.Float32(0.0) - a, a)


def _cvt_fp8x4(values):
    low = rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[0], values[1], fx.Int32(0), False)
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[2], values[3], low, True))


def _xcd_order(block, tiles):
    """Logical tile of block b (< tiles): XCD b % 8 takes a contiguous chunk of the logical order."""
    x8 = block & 7
    return x8 * (tiles >> 3) + _min(x8, tiles & 7) + (block >> 3)



def _retire_last(state):
    acc = [fx.Vector(c) for c in state[:8]]
    # The last two blocks (up N-block 7, M-blocks 2 and 3) are still unretired.
    p2, s2, p3, s3 = fx.Vector(state[8]), fx.Float32(state[9]), fx.Vector(state[10]), fx.Float32(state[11])
    acc[7] = fx.Vector.from_elements(
        [_fma(p2[i - 8], s2, acc[7][i]) if 8 <= i < 12 else _fma(p3[i - 12], s3, acc[7][i]) if i >= 12
         else acc[7][i] for i in range(16)], fx.Float32)
    return acc


def _act_epilogue(acc, storage, shared, wave, lane, wm, wn, r16, g, SW, A2, A2S, R, row0, rows, jt, I):
    """SiLU(gate) * up, 1x128 FP8 requant of the tile's first `rows` rows into A2 / A2S (see module docstring)."""
    # act[nbl][4 mb + e]: token 16 mb + r16, column 64 wn + 16 nbl + 4 g + e of this 128-column group.
    act = [[_silu_mul(acc[nbl][i], acc[4 + nbl][i]) for i in range(16)] for nbl in range(4)]
    amax = []
    for mb in range(4):
        m = _abs(act[0][4 * mb])
        for i in range(1, 16):
            m = _max(m, _abs(act[i // 4][4 * mb + i % 4]))
        m = _max(m, fx.Float32(m.shuffle_xor(16, 64)))
        amax.append(_max(m, fx.Float32(m.shuffle_xor(32, 64))))
    # The group spans waves wn = 0, 1: exchange row maxima through the wave-private LDS regions (G0 reuses
    # X0 / X1, read for the last time two stages ago; G1 reuses P1 / P2).
    own = (g == 0).select(amax[0], (g == 1).select(amax[1], (g == 2).select(amax[2], amax[3])))
    _write_lds_f32(shared + wave * 8192 + lane * 4, own)
    _wait(lgkmcnt=0)
    _stage_end()
    other = shared + (wave ^ 1) * 8192
    partner = [_read_lds_f32(other + (16 * mb + r16) * 4) for mb in range(4)]
    partner_own = _read_lds_f32(other + lane * 4)
    sw_res = _buffer(fx.make_view(fx.get_iter(SW) + fx.Int64(row0), fx.make_layout(BM, 1)), BM * 4)
    route = _load_f32(sw_res, (wm * 64 + lane) * 4)
    _wait(lgkmcnt=0)

    # The FP8 tile goes through LDS so every store writes whole 128-byte rows: tile row r at 4 KiB of region
    # r >> 5 (clear of the row maxima), 16-byte chunks swizzled by r & 7.
    for mb in range(4):
        top = _max(amax[mb], partner[mb])
        inv = (top > fx.Float32(0.0)).select(fx.Float32(FP8_MAX) / top, fx.Float32(0.0))
        stage = shared + (2 * wm + (mb >> 1)) * 8192 + 4096 + (16 * (mb & 1) + r16) * 128 + g * 4
        for nbl in range(4):
            word = _cvt_fp8x4([act[nbl][4 * mb + e] * inv for e in range(4)])
            _write_lds_f32(stage + (((4 * wn + nbl) ^ (r16 & 7)) << 4), _as_f32(word))
    _wait(lgkmcnt=0)
    _stage_end()
    # Wave w stores tile rows 32 w .. + 31; lane l: row 8 i + (l >> 3), chunk l & 7.
    rows8 = [_read_address(shared + wave * 8192 + 4096 + (8 * i + (lane >> 3)) * 128
                           + (((lane & 7) ^ (lane >> 3)) << 4)) for i in range(4)]
    a2_res = _buffer(fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * I, fx.make_layout(BM * I, 1)), rows * I)
    _wait(lgkmcnt=0)
    for i in range(4):
        _store(rows8[i], a2_res, (wave * 32 + 8 * i + (lane >> 3)) * I + jt * 128 + (lane & 7) * 16)
    # Scale (times the route weight) of token `lane` of the wave, written by the wn = 0 wave.
    top = _max(own, partner_own)
    scale = (top > fx.Float32(0.0)).select(top * fx.Float32(1.0 / FP8_MAX) * route, fx.Float32(0.0))
    a2s_res = _buffer(A2S, (I // 128) * R * 4)
    _store(scale, a2s_res, ((wn == 0) & (wm * 64 + lane < rows)).select((jt * R + row0 + wm * 64 + lane) * 4,
                                                                        fx.Int32(_DROP)))


@flyc.jit
def _gate_up_body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, row0, rows, jt, expert,
                  I: fx.Constexpr[int], K: fx.Constexpr[int], GROUP: fx.Constexpr[int]):
    """Tile rows row0 .. row0 + rows - 1 (rows = 64, 128, 192 or 256, one expert), intermediate columns
    128 jt .. + 127. Waves whose 64-row band lies past `rows` keep the DMA / barrier schedule but issue no
    MFMAs."""
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (_dma, _dma_x_gather, _dma_w13, _read_frag, _compute, _fma, _as_f32, _load_xs_gather, _buffer,
                _min, _pin_i32, _read_scalar_g2r, _scalar_ready, _schedule, _stage_end, _uniform, _wait,
                _load_i32, _load_f32, _store, _readlane, _write_lds_f32, _read_lds_f32, _silu_mul, _max, _abs,
                _cvt_fp8x4, _retire_last, _act_epilogue, _read_address)
    KB = K // BK
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    q = wave & 3
    wm, wn = (q >> 1) + 2 * GROUP, q & 1
    h = lane >> 5
    r16, g = lane & 15, lane >> 4
    active = wm * 64 < rows

    x_res = _buffer(X, T * K)
    w_res = _buffer(fx.make_view(fx.get_iter(W) + fx.Int64(expert) * (2 * I * K), fx.make_layout(2 * I * K, 1)),
                    2 * I * K)
    xs_res = _buffer(XS, T * (KB * 4))
    sid_res = _buffer(fx.make_view(fx.get_iter(SID) + fx.Int64(row0), fx.make_layout(BM, 1)), BM * 4)

    def token_offset(row, unit):
        token = _load_i32(sid_res, row * 4) & 0xFFFFFF
        return ((row < rows) & (token < T)).select(token * unit, fx.Int32(_DROP))

    # Padding rows (token T) and rows past the tile read past the X / scale bounds: zero rows, zero scales.
    table = _pin_i32(token_offset(128 * h + 2 * q + 8 * ((lane & 31) >> 1) + (lane & 1), K))
    xs_off = [_pin_i32(token_offset(wm * 64 + 16 * mb + r16, 4)) for mb in range(4)]
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    x_read = [_pin_i32(shared + GROUP * SLOT + ((q >> 1) * 64 + r16) * 128 + (((2 * g + j) ^ (r16 & 7)) << 4))
              for j in range(2)]
    w_read = [_pin_i32(shared + wn * 8192 + (2 * g + j) * 256 + r16 * 16) for j in range(2)]
    chunk = _pin_i32(((((lane & 31) >> 2) ^ ((q << 1) | h)) << 4) + ((lane & 3) << 2))
    lane4 = _pin_i32(lane * 4)
    gate16 = jt * 8
    up16 = jt * 8 + I // 16
    wsg_row = (expert * (2 * I // 128) + jt) * KB
    wsu_row = wsg_row + (I // 128) * KB

    @flyc.jit
    def kblock(c0, c1, c2, c3, c4, c5, c6, c7, part0, scale0, part1, scale1, xs0, xs1, xs2, xs3, wsg_bits,
               wsu_bits, kb, LAST: fx.Constexpr[bool]):
        acc = [fx.Vector(c) for c in (c0, c1, c2, c3, c4, c5, c6, c7)]
        kb = fx.Int32(kb)
        following = _min(kb + 1, fx.Int32(KB - 1))
        # M1: X and gate rows; G0: DMA up rows (kb), G1: DMA X0(kb+1).
        rocdl.s_setprio(0)
        x = [_read_frag(x_read, mb * 2048) for mb in range(4)]
        w = [_read_frag(w_read, P1_SLOT + nbl * 2048) for nbl in range(4)]
        wsg = _as_f32(wsg_bits)
        scales = tuple(fx.Float32(s) * wsg for s in (xs0, xs1, xs2, xs3))
        if fx.const_expr(GROUP == 0):
            _dma_w13(w_res, storage, q, lane4, 1, kb, up16, K)
        else:
            if fx.const_expr(not LAST):
                _dma_x_gather(x_res, storage, q, table, chunk, h, 0, kb + 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # C1: gate N-blocks 0-3 (idle bands keep their accumulators).
        rocdl.s_setprio(3)
        queue = [(fx.Vector(part0), (7, 2), fx.Float32(scale0)), (fx.Vector(part1), (7, 3), fx.Float32(scale1))]
        a0, a1, a2, a3, a4, a5, a6, a7 = acc
        p0, s0, p1, s1 = queue[0][0], queue[0][2], queue[1][0], queue[1][2]
        if active:
            acc_c = [a0, a1, a2, a3, a4, a5, a6, a7]
            queue_c = _compute(w, x, acc_c, queue, scales, 0)
            _schedule(16, 0)
            rocdl.sched_barrier(0)
            a0, a1, a2, a3, a4, a5, a6, a7 = acc_c
            p0, s0, p1, s1 = queue_c[0][0], queue_c[0][2], queue_c[1][0], queue_c[1][2]
        acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        queue = [(p0, (3, 2), s0), (p1, (3, 3), s1)]
        rocdl.sched_barrier(0)
        if fx.const_expr(GROUP == 0):
            # Up rows (kb) for M2 of both groups.
            _wait(vmcnt=0)
        _stage_end()
        # M2: up rows; G0: DMA gate rows (kb+1), G1: DMA X1(kb+1); scales of K-block kb+1.
        rocdl.s_setprio(0)
        w = [_read_frag(w_read, P2_SLOT + nbl * 2048) for nbl in range(4)]
        wsu = _as_f32(wsu_bits)
        scales = tuple(fx.Float32(s) * wsu for s in (xs0, xs1, xs2, xs3))
        wsg_request = _read_scalar_g2r(WS, wsg_row + following)
        wsu_request = _read_scalar_g2r(WS, wsu_row + following)
        if fx.const_expr(not LAST):
            if fx.const_expr(GROUP == 0):
                _dma_w13(w_res, storage, q, lane4, 0, kb + 1, gate16, K)
            else:
                _dma_x_gather(x_res, storage, q, table, chunk, h, 1, kb + 1)
                # X0(kb+1) for G0's next M1; X1(kb+1) may stay in flight.
                _wait(vmcnt=16)
        xs_next = _load_xs_gather(xs_res, xs_off, following * (T * 4))
        _wait(lgkmcnt=0)
        wsg_next = _scalar_ready(wsg_request)
        wsu_next = _scalar_ready(wsu_request)
        _stage_end()
        # C2: up N-blocks 4-7.
        rocdl.s_setprio(3)
        a0, a1, a2, a3, a4, a5, a6, a7 = acc
        p0, s0, p1, s1 = queue[0][0], queue[0][2], queue[1][0], queue[1][2]
        if active:
            acc_c = [a0, a1, a2, a3, a4, a5, a6, a7]
            queue_c = _compute(w, x, acc_c, queue, scales, 4)
            _schedule(16, 1)
            rocdl.sched_barrier(0)
            a0, a1, a2, a3, a4, a5, a6, a7 = acc_c
            p0, s0, p1, s1 = queue_c[0][0], queue_c[0][2], queue_c[1][0], queue_c[1][2]
        acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        queue = [(p0, (7, 2), s0), (p1, (7, 3), s1)]
        rocdl.sched_barrier(0)
        _wait(vmcnt=0)
        if fx.const_expr(not (LAST and GROUP == 1)):
            # G1 skips its last barrier: G0 has no partner stage left.
            _stage_end()
        return (*acc, queue[0][0], queue[0][2], queue[1][0], queue[1][2], *xs_next, wsg_next, wsu_next)

    # Prologue: G0 loads the gate rows of K-block 0, G1 X0(0) and X1(0); scales of K-block 0.
    if fx.const_expr(GROUP == 0):
        _dma_w13(w_res, storage, q, lane4, 0, fx.Int32(0), gate16, K)
    else:
        _dma_x_gather(x_res, storage, q, table, chunk, h, 0, fx.Int32(0))
        _dma_x_gather(x_res, storage, q, table, chunk, h, 1, fx.Int32(0))
    wsg_request = _read_scalar_g2r(WS, wsg_row)
    wsu_request = _read_scalar_g2r(WS, wsu_row)
    xs_first = _load_xs_gather(xs_res, xs_off, 0)
    _wait(vmcnt=0, lgkmcnt=0)
    wsg_first = _scalar_ready(wsg_request)
    wsu_first = _scalar_ready(wsu_request)
    _stage_end()
    if fx.const_expr(GROUP == 1):
        # G1 falls one stage behind.
        _stage_end()
    zero = fx.Vector.filled(16, 0.0, fx.Float32)
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (zero, zero, zero, zero, zero, zero, zero, zero, zero4, fx.Float32(0.0), zero4, fx.Float32(0.0),
             *xs_first, wsg_first, wsu_first)
    for kb in range(fx.Int32(0), fx.Int32(KB - 1), fx.Int32(1)):
        state = kblock(*state, kb, False)
    state = kblock(*state, fx.Int32(KB - 1), True)
    _act_epilogue(_retire_last(state), storage, shared, wave, lane, wm, wn, r16, g, SW, A2, A2S, R, row0, rows, jt,
                  I)


def _tile(TILES, tile):
    """(first row, rows, expert) of compact tile `tile` (see _tiles_kernel)."""
    row0 = _scalar_ready(_read_scalar_g2r(TILES, 2 * tile))
    info = _scalar_ready(_read_scalar_g2r(TILES, 2 * tile + 1))
    return row0, (info >> 16) * 64, info & 0xFFFF


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _gate_up_kernel(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, SID: fx.Tensor, TILES: fx.Tensor,
                    COUNT: fx.Tensor, SW: fx.Tensor, A2: fx.Tensor, A2S: fx.Tensor, T: fx.Int32, R: fx.Int32,
                    I: fx.Constexpr[int], K: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(
        fx.make_layout(LDS_BYTES, 1))
    body = _gate_up_body
    order = _xcd_order
    block = fx.Int32(gpu.block_id("x"))
    TN = I // 128
    tiles = _scalar_ready(_read_scalar_g2r(COUNT, 0)) * TN
    group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
    if block < tiles:
        v = order(block, tiles)
        tile = v // TN
        jt = v - tile * TN
        row0, rows, expert = _tile(TILES, tile)
        if group != 0:
            body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, row0, rows, jt, expert, I, K, 1)
        else:
            body(X, W, XS, WS, SID, SW, A2, A2S, storage, T, R, row0, rows, jt, expert, I, K, 0)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _down_kernel(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, SEID: fx.Tensor, NV: fx.Tensor,
                 Y: fx.Tensor, R: fx.Int32, N: fx.Constexpr[int], K: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, LDS_BYTES, 16]).peek().view(
        fx.make_layout(LDS_BYTES, 1))
    body = _body
    order = _xcd_order
    block = fx.Int32(gpu.block_id("x"))
    TN = N // BN
    tiles = (_scalar_ready(_read_scalar_g2r(NV, 0)) // BM) * TN
    group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
    if block < tiles:
        v = order(block, tiles)
        mblk = v // TN
        nt = v - mblk * TN
        expert = _scalar_ready(_read_scalar_g2r(SEID, mblk))
        w = fx.make_view(fx.get_iter(W) + fx.Int64(expert) * (N * K), fx.make_layout(N * K, 1))
        ws = fx.make_view(fx.get_iter(WS) + fx.Int64(expert) * ((N // 128) * (K // BK)),
                          fx.make_layout((N // 128) * (K // BK), 1))
        if group != 0:
            body(A2, w, A2S, ws, Y, Y, storage, R, mblk, nt, fx.Int32(0), fx.Int32(K // BK), fx.Int32(0), N, K, K,
                 True, False, 1)
        else:
            body(A2, w, A2S, ws, Y, Y, storage, R, mblk, nt, fx.Int32(0), fx.Int32(K // BK), fx.Int32(0), N, K, K,
                 True, False, 0)


def _load_v4(resource, voffset, soffset=0):
    vt = ir.VectorType.get([4], fx.Int32.ir_type)
    return fx.Vector(rocdl.raw_ptr_buffer_load(vt, resource, fx.Int32(voffset).ir_value(),
                                               fx.Int32(soffset).ir_value(), aux=0))


def _down_lane(lane):
    """DMA lane offset inside one 256-byte piece (16 rows x 16 K bytes): LDS row position rho holds row
    rho ^ (4 (rho >> 3)), which keeps the permuted fragment reads of _read_unit bank-conflict free."""
    return (((lane >> 2) ^ ((lane >> 5) << 2)) << 4) + ((lane & 3) << 2)


def _dma_chunk(resource, storage, wave, lane_off, slot, chunk, K):
    """W2 rows 64 chunk .. + 63 of the task (contiguous 64 K bytes in the 16x16 shuffle) -> LDS slot."""
    pieces = DOWN_CHUNK * K // 1024
    for t in range(pieces):
        p = wave * pieces + t
        _dma(resource, storage, slot * (DOWN_CHUNK * K) + p * 256, lane_off, chunk * (DOWN_CHUNK * K) + p * 256)


def _read_unit(w_read, slot, unit, K):
    """Fragments wf[v][kb] (two ds_read_b128 each) of a 32-row unit. MFMA v takes unit row
    8 (r16 >> 2) + 4 v + (r16 & 3) as its A row r16, so output lane (r16, g) holds columns 8 g + 4 v .. + 3."""
    base = slot * (DOWN_CHUNK * K) + unit * (32 * K)
    return [[[_read_address(w_read[v][j], base + kb * 2048) for j in range(2)] for kb in range(K // BK)]
            for v in range(2)]


def _fma4(a, b, c):
    vt = ir.VectorType.get([4], fx.Float32.ir_type)
    splat = fx.Vector.from_elements([fx.Float32(b)] * 4, fx.Float32)
    return fx.Vector(llvm.call_intrinsic(vt, "llvm.fma.v4f32", [fx.Vector(a).ir_value(), splat.ir_value(),
                                                                 fx.Vector(c).ir_value()], [], []))


def _retire(item):
    """out = p0 * s0 + p1 * s1 (FP32, K-block order) -> BF16 RNE: 4 words = the lane's 8 consecutive columns."""
    parts, s0, s1 = item[:3]
    words = []
    for v in range(2):
        packed = _pack_bf16(_fma4(parts[v][1], s1, fx.Vector(parts[v][0]) * fx.Float32(s0))).bitcast(fx.Int32)
        words += [packed[0], packed[1]]
    return words


def _store_pair(first, second, y_res, voffset, soffset):
    """Columns 8 g .. + 7 of an even unit and of the odd unit after it, back to back: each 128-byte line
    of the row is complete within two adjacent stores (64-byte halves written apart cost ~40% bandwidth)."""
    for words, offset in ((first, soffset - 64), (second, soffset)):
        rocdl.raw_ptr_buffer_store(fx.Vector.from_elements(words, fx.Int32).ir_value(), y_res,
                                   fx.Int32(voffset).ir_value(), fx.Int32(offset).ir_value())


def _unit(wf, x, scales, item, held, y_res, vo, soffset, odd, prefetch):
    """64 MFMAs of a 32-column unit (4 M-blocks x 2 v x 2 K-blocks x 4 steps). Each M-block's partials are
    retired during the next M-block's MFMAs (3 VALU slots per MFMA); the first retires `item` (the previous
    unit's last M-block), the last is returned as the next item. Even units' words wait in `held` and are
    stored with the odd unit's. prefetch() issues the next unit's fragment reads up front."""
    rocdl.sched_barrier(0)
    following = prefetch() if prefetch is not None else None
    held = list(held)
    for mb in range(4):
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
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 8, 0)
    for _ in range(64):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
        rocdl.sched_group_barrier(0x002, 3, 0)
    rocdl.sched_barrier(0)
    return item, held, following


def _maybe(flag, fn):
    return fn if flag else None


def _down_idle(w_res, storage, wave, lane_off, ITERS, K):
    """After the common prologue, a wave without rows in the tile: its quarter of every W2 chunk DMA and the
    barriers of the computing waves (after U0 of every pair, after U2 of every pair but the last)."""
    for it in range(ITERS - 1):
        _wait(vmcnt=0)
        _stage_end()
        _dma_chunk(w_res, storage, wave, lane_off, 0, fx.Int32(2 * it + 2), K)
        _wait(vmcnt=0)
        _stage_end()
        _dma_chunk(w_res, storage, wave, lane_off, 1, fx.Int32(2 * it + 3), K)
    _wait(vmcnt=0)
    _stage_end()


@flyc.jit
def _down_direct_body(A2, W, A2S, WS, SID, Y, storage, R, T, tile_row, nblk, cg, expert, N: fx.Constexpr[int],
                      K: fx.Constexpr[int]):
    """Rows tile_row .. + 64 nblk - 1 (one expert) x columns cg * DOWN_COLS ..: wave w < nblk owns rows 64 w ..
    + 63, the other waves only take their part of the W2 DMA ring (no MFMAs)."""
    _helpers = (_load_v4, _load_i32, _load_f32, _dma, _down_lane, _dma_chunk, _read_unit, _read_address, _fma4,
                _retire, _store_pair, _unit, _maybe, _operand, _mfma, _pack_bf16, _as_f32, _buffer, _min, _pin_i32,
                _read_scalar_g2r, _scalar_ready, _stage_end, _uniform, _wait, _down_idle)
    assert K == 256, "direct down: K = 256 (two K-blocks per unit)"
    KB = K // BK
    ITERS = DOWN_COLS // 128
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g = lane & 15, lane >> 4
    row0 = tile_row + wave * 64
    col0 = cg * DOWN_COLS
    active = wave < nblk
    a2_res = _buffer(fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * K, fx.make_layout(64 * K, 1)),
                     active.select(fx.Int32(64 * K), fx.Int32(0)))
    a2s_res = _buffer(A2S, KB * R * 4)
    sid_res = _buffer(SID, R * 4)
    w_res = _buffer(fx.make_view(fx.get_iter(W) + (fx.Int64(expert) * N + fx.Int64(col0)) * K,
                                 fx.make_layout(DOWN_COLS * K, 1)), DOWN_COLS * K)
    y_res = _buffer(fx.make_view(fx.get_iter(Y) + fx.Int64(tile_row) * N, fx.make_layout(BM * N, 1)),
                    nblk * (64 * N * 2))
    lane_off = _pin_i32(_down_lane(lane))
    ws_row = (expert * (N // 128) + col0 // 128) * KB

    # Prologue: W2 chunks 0 / 1 -> slots 0 / 1; the wave's 64 A2 rows (whole K) and scales stay in registers.
    _dma_chunk(w_res, storage, wave, lane_off, 0, fx.Int32(0), K)
    _dma_chunk(w_res, storage, wave, lane_off, 1, fx.Int32(1), K)
    x = [[[_load_v4(a2_res, (16 * mb + r16) * K + kb * 128 + 32 * g + 16 * j) for j in range(2)]
          for kb in range(KB)] for mb in range(4)]
    xs = [[_load_f32(a2s_res, (kb * R + row0 + 16 * mb + r16) * 4) for kb in range(KB)] for mb in range(4)]
    tokens = [_load_i32(sid_res, (row0 + 16 * mb + r16) * 4) & 0xFFFFFF for mb in range(4)]
    requests = [_read_scalar_g2r(WS, ws_row + kb) for kb in range(KB)]
    _wait(vmcnt=0)
    ws_first = [_scalar_ready(r) for r in requests]
    # Lane (r16, g) stores columns 8 g .. + 7 of row 16 mb + r16; padding rows (token T) store nothing.
    vo = [_pin_i32((tokens[mb] < T).select(((wave * 64 + 16 * mb + r16) * N + 8 * g) * 2, fx.Int32(_DROP)))
          for mb in range(4)]
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    b = (r16 >> 2) & 1
    w_read = [[_pin_i32(shared + (r16 >> 3) * (16 * K) + (2 * g + j) * 256 + (8 * b + 4 * (b ^ v) + (r16 & 3)) * 16)
               for j in range(2)] for v in range(2)]
    _stage_end()
    wf_first = _read_unit(w_read, 0, 0, K)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)

    @flyc.jit
    def pair(f0, f1, f2, f3, f4, f5, f6, f7, p0, p1, p2, p3, s0, s1, item_vo, h0, h1, h2, h3, ws0, ws1, it,
             LAST: fx.Constexpr[bool]):
        """Columns 128 it .. + 127 (one weight block scale): chunk 2 it in slot 0, 2 it + 1 in slot 1, as
        units U0-U3 of 32 columns. Entry: U0's fragments in f*, chunk 2 it + 1 in flight, the previous
        pair's last M-block unretired in p* / s* / item_vo, and U2's M-block 3 words in h*."""
        it = fx.Int32(it)
        wf = [[[fx.Vector(f0), fx.Vector(f1)], [fx.Vector(f2), fx.Vector(f3)]],
              [[fx.Vector(f4), fx.Vector(f5)], [fx.Vector(f6), fx.Vector(f7)]]]
        cbase = (col0 + it * 128) * 2
        item = ([[fx.Vector(p0), fx.Vector(p1)], [fx.Vector(p2), fx.Vector(p3)]], fx.Float32(s0),
                fx.Float32(s1), fx.Int32(item_vo), cbase - 64)
        held = [None, None, None, [fx.Int32(h) for h in (h0, h1, h2, h3)]]
        ws = (_as_f32(ws0), _as_f32(ws1))
        scales = [[fx.Float32(xs[mb][kb]) * ws[kb] for kb in range(KB)] for mb in range(4)]
        following = _min(it + 1, fx.Int32(ITERS - 1))
        requests = [_read_scalar_g2r(WS, ws_row + following * KB + kb) for kb in range(KB)]
        item, held, wf = _unit(wf, x, scales, item, held, y_res, vo, cbase, False,
                               lambda: _read_unit(w_read, 0, 1, K))
        _wait(lgkmcnt=0)
        # Chunk 2 it + 1: its DMA preceded the 8 stores of the previous U3 and this U0 (in-order vmcnt).
        # After the barrier every wave has slot 0 in registers: refill it with chunk 2 it + 2.
        _wait(vmcnt=8)
        _stage_end()
        if fx.const_expr(not LAST):
            _dma_chunk(w_res, storage, wave, lane_off, 0, it * 2 + 2, K)
        item, held, wf = _unit(wf, x, scales, item, held, y_res, vo, cbase + 64, True,
                               lambda: _read_unit(w_read, 1, 0, K))
        _wait(lgkmcnt=0)
        item, held, wf = _unit(wf, x, scales, item, held, y_res, vo, cbase + 128, False,
                               lambda: _read_unit(w_read, 1, 1, K))
        _wait(lgkmcnt=0)
        if fx.const_expr(not LAST):
            # Chunk 2 it + 2 (8 stores ago) is visible after this barrier; slot 1 is in registers.
            _wait(vmcnt=8)
            _stage_end()
            _dma_chunk(w_res, storage, wave, lane_off, 1, it * 2 + 3, K)
        item, held, following_wf = _unit(wf, x, scales, item, held, y_res, vo, cbase + 192, True,
                                         _maybe(not LAST, lambda: _read_unit(w_read, 0, 0, K)))
        _wait(lgkmcnt=0)
        ws_next = [_scalar_ready(r) for r in requests]
        wf = wf if following_wf is None else following_wf
        parts = item[0]
        return (*[wf[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)],
                parts[0][0], parts[0][1], parts[1][0], parts[1][1], item[1], item[2], vo[3], *held[3], *ws_next)

    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    state0 = (*[wf_first[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)], zero, zero, zero, zero,
              fx.Float32(0.0), fx.Float32(0.0), fx.Int32(_DROP), *[fx.Int32(0)] * 4, *ws_first)
    if active:
        state = state0
        for it in range(fx.Int32(0), fx.Int32(ITERS - 1), fx.Int32(1)):
            state = pair(*state, it, False)
        state = pair(*state, fx.Int32(ITERS - 1), True)
        last = ([[state[8], state[9]], [state[10], state[11]]], state[12], state[13])
        _store_pair([fx.Int32(h) for h in state[15:19]], _retire(last), y_res, vo[3], (col0 + DOWN_COLS - 32) * 2)
    else:
        _down_idle(w_res, storage, wave, lane_off, ITERS, K)


@flyc.kernel(known_block_size=[DOWN_THREADS, 1, 1])
def _down_direct_kernel(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, TILES: fx.Tensor,
                        COUNT: fx.Tensor, SID: fx.Tensor, Y: fx.Tensor, R: fx.Int32, T: fx.Int32, CG0: fx.Int32,
                        N: fx.Constexpr[int], K: fx.Constexpr[int], GROUPS: fx.Constexpr[int]):
    """K = 256 down GEMM: task = (compact tile, 1024 columns of groups CG0 .. CG0 + GROUPS - 1)."""
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, 2 * DOWN_CHUNK * K, 16]).peek().view(
        fx.make_layout(2 * DOWN_CHUNK * K, 1))
    body = _down_direct_body
    order = _xcd_order
    block = fx.Int32(gpu.block_id("x"))
    tiles = _scalar_ready(_read_scalar_g2r(COUNT, 0)) * GROUPS
    if block < tiles:
        v = order(block, tiles)
        tile = v // GROUPS
        cg = CG0 + v - tile * GROUPS
        row0, rows, expert = _tile(TILES, tile)
        body(A2, W, A2S, WS, SID, Y, storage, R, T, row0, rows >> 6, cg, expert, N, K)


@flyc.kernel(known_block_size=[TILE_THREADS, 1, 1])
def _tiles_kernel(SEID: fx.Tensor, NV: fx.Tensor, TILES: fx.Tensor, COUNT: fx.Tensor, NB: fx.Int32, CAP: fx.Int32,
                  E: fx.Constexpr[int], BLOCK: fx.Constexpr[int]):
    """Compact tiles from aiter's sorted blocks (BLOCK rows each; every expert one contiguous run): each run
    is cut into 256-row tiles and one tail of its remaining blocks. TILES[2 t] = first row, TILES[2 t + 1] =
    expert | rows / 64 << 16, in run order; COUNT[0] = tiles."""
    _helpers = (_buffer, _load_i32, _load_v4, _store, _read_scalar_g2r, _scalar_ready, _min)
    width = max(TILE_THREADS, 1 << (E - 1).bit_length())
    rounds = width // TILE_THREADS
    per = BM // BLOCK
    shared = fx.SharedAllocator().allocate(fx.Array[fx.Int32, 3 * width, 16]).peek().view(
        fx.make_layout(3 * width, 1))
    tid = fx.Int32(gpu.thread_id("x"))
    for part in range_constexpr(rounds):
        shared[tid + part * TILE_THREADS] = fx.Int32(0)
        shared[width + tid + part * TILE_THREADS] = fx.Int32(0)
    gpu.barrier()
    seid_res = _buffer(SEID, NB * 4)
    blocks = _scalar_ready(_read_scalar_g2r(NV, 0)) // BLOCK
    # Thread t scans blocks 16 t .. + 15 of each round (four dwordx4 loads plus both neighbours in flight).
    for base in range(fx.Int32(0), blocks, fx.Int32(16 * TILE_THREADS)):
        b0 = base + tid * 16
        if b0 < blocks:
            quads = [_load_v4(seid_res, (b0 + 4 * j) * 4) for j in range(4)]
            ids = [_load_i32(seid_res, (b0 - 1) * 4)] + [fx.Int32(quads[i >> 2][i & 3]) for i in range(16)] + \
                [_load_i32(seid_res, (b0 + 16) * 4)]
            for i in range_constexpr(16):
                b = b0 + i
                e = ids[i + 1]
                if (b < blocks) & (e >= 0) & (e < E):
                    if (b == 0) | (ids[i] != e):
                        shared[e] = b
                    if (b + 1 == blocks) | (ids[i + 2] != e):
                        shared[width + e] = b + 1
    gpu.barrier()
    for part in range_constexpr(rounds):
        e = tid + part * TILE_THREADS
        shared[2 * width + e] = (shared[width + e] - shared[e] + per - 1) // per
    gpu.barrier()
    # Inclusive scan of the per-expert tile counts.
    for shift in range_constexpr(width.bit_length() - 1):
        d = 1 << shift
        sums = []
        for part in range_constexpr(rounds):
            e = tid + part * TILE_THREADS
            peer = (e >= d).select(e - d, e)
            sums.append(shared[2 * width + e] + (e >= d).select(shared[2 * width + peer], fx.Int32(0)))
        gpu.barrier()
        for part in range_constexpr(rounds):
            shared[2 * width + tid + part * TILE_THREADS] = sums[part]
        gpu.barrier()
    tiles_res = _buffer(TILES, CAP * 8)
    for part in range_constexpr(rounds):
        e = tid + part * TILE_THREADS
        first = shared[e]
        size = shared[width + e] - first
        n = (size + per - 1) // per
        start = shared[2 * width + e] - n
        for i in range(fx.Int32(0), n, fx.Int32(1)):
            rows = _min(fx.Int32(BM), (size - i * per) * BLOCK)
            _store(fx.Int32((first + i * per) * BLOCK), tiles_res, (start + i) * 8)
            _store(fx.Int32(e | ((rows >> 6) << 16)), tiles_res, (start + i) * 8 + 4)
    if tid == 0:
        _store(fx.Int32(shared[3 * width - 1]), _buffer(COUNT, 4), fx.Int32(0))


def _inverse_row(SID, NV, INV, T, R, row, TOPK):
    valid = _scalar_ready(_read_scalar_g2r(NV, 0))
    sid = _load_i32(_buffer(SID, R * 4), row * 4)
    token, slot = sid & 0xFFFFFF, (sid >> 24) & 0xFF
    target = (slot < TOPK).select((token * TOPK + slot) * 4, fx.Int32(_DROP))
    target = (token < T).select(target, fx.Int32(_DROP))
    _store(row, _buffer(INV, T * (TOPK * 4)), (row < valid).select(target, fx.Int32(_DROP)))


@flyc.kernel(known_block_size=[SUM_THREADS, 1, 1])
def _inverse_kernel(SID: fx.Tensor, NV: fx.Tensor, INV: fx.Tensor, T: fx.Int32, R: fx.Int32,
                    TOPK: fx.Constexpr[int]):
    """inv[t * TOPK + k] = sorted row of route (t, k), from the valid prefix of the sorted ids."""
    _helpers = (_buffer, _load_i32, _store, _read_scalar_g2r, _scalar_ready, _inverse_row)
    row = fx.Int32(gpu.block_id("x")) * SUM_THREADS + fx.Int32(gpu.thread_id("x"))
    _inverse_row(SID, NV, INV, T, R, row, TOPK)


def _sum_chunk(Y, INV, OUT, T, R, token, C0, N, NS, TOPK):
    """CTA = token, columns C0 .. C0 + NS - 1: its routes' sorted rows come in by scalar loads; wave w sums
    columns C0 + NS / 4 w .. + NS / 4 - 1, lane l the 8 columns 8 (l + 64 h) of that range (1 KB per load
    instruction, streamed past the caches). A missing route (-1) reads past the buffer: zero."""
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    requests = [_read_scalar_g2r(INV, token * TOPK + k) for k in range(TOPK)]
    rows = [_scalar_ready(r) for r in requests]
    y_res = _buffer(Y, R * (N * 2))
    out_res = _buffer(OUT, T * (N * 2))
    vt = ir.VectorType.get([4], fx.Int32.ir_type)
    span = NS // 4
    pieces = (span // 8 + 63) // 64
    columns = [C0 + wave * span + 8 * (lane + 64 * h) for h in range(pieces)]
    inside = [lane + 64 * h < span // 8 for h in range(pieces)]
    parts = [[fx.Vector(rocdl.raw_ptr_buffer_load(vt, y_res, fx.Int32(rows[k] * (N * 2) + columns[h] * 2).ir_value(),
                                                  fx.Int32(0).ir_value(), aux=rocdl_aux("nt")))
              for k in range(TOPK)] for h in range(pieces)]
    for h in range(pieces):
        total = [fx.Float32(0.0)] * 8
        for k in range(TOPK):
            bits = parts[h][k].bitcast(fx.Uint32)
            for i in range(4):
                lo = fx.Vector.from_elements([bits[i] << 16], fx.Uint32).bitcast(fx.Float32)[0]
                hi = fx.Vector.from_elements([bits[i] & fx.Uint32(0xFFFF0000)], fx.Uint32).bitcast(fx.Float32)[0]
                total[2 * i] = total[2 * i] + lo
                total[2 * i + 1] = total[2 * i + 1] + hi
        words = _pack_bf16(fx.Vector.from_elements(total, fx.Float32))
        voffset = inside[h].select((token * N + columns[h]) * 2, fx.Int32(_DROP))
        rocdl.raw_ptr_buffer_store(words.bitcast(fx.Int32).ir_value(), out_res, fx.Int32(voffset).ir_value(),
                                   fx.Int32(0).ir_value())


@flyc.kernel(known_block_size=[SUM_THREADS, 1, 1])
def _sum_kernel(Y: fx.Tensor, INV: fx.Tensor, OUT: fx.Tensor, T: fx.Int32, R: fx.Int32, C0: fx.Int32,
                N: fx.Constexpr[int], NS: fx.Constexpr[int], TOPK: fx.Constexpr[int]):
    """out[t, C0 .. C0 + NS - 1] = BF16(sum_k Y[inv[t, k]]) in FP32, k order; one CTA per token."""
    _helpers = (_buffer, _load_i32, _pack_bf16, _sum_chunk, _read_scalar_g2r, _scalar_ready, _uniform)
    _sum_chunk(Y, INV, OUT, T, R, fx.Int32(gpu.block_id("x")), C0, N, NS, TOPK)


def _attrs():
    return {"rocdl.waves_per_eu": 2,
            "llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')}


@flyc.jit
def launch_gate_up(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, SID: fx.Tensor, TILES: fx.Tensor,
                   COUNT: fx.Tensor, SW: fx.Tensor, A2: fx.Tensor, A2S: fx.Tensor, T: fx.Int32, R: fx.Int32,
                   GRID: fx.Int32, I: fx.Constexpr[int], K: fx.Constexpr[int], stream: fx.Stream):
    _gate_up_kernel(X, W, XS, WS, SID, TILES, COUNT, SW, A2, A2S, T, R, I, K, value_attrs=_attrs()).launch(
        grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_down(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, SEID: fx.Tensor, NV: fx.Tensor,
                Y: fx.Tensor, R: fx.Int32, GRID: fx.Int32, N: fx.Constexpr[int], K: fx.Constexpr[int],
                stream: fx.Stream):
    _down_kernel(A2, W, A2S, WS, SEID, NV, Y, R, N, K, value_attrs=_attrs()).launch(
        grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_down_direct(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, TILES: fx.Tensor,
                       COUNT: fx.Tensor, SID: fx.Tensor, Y: fx.Tensor, R: fx.Int32, T: fx.Int32, CG0: fx.Int32,
                       GRID: fx.Int32, N: fx.Constexpr[int], K: fx.Constexpr[int], GROUPS: fx.Constexpr[int],
                       stream: fx.Stream):
    _down_direct_kernel(A2, W, A2S, WS, TILES, COUNT, SID, Y, R, T, CG0, N, K, GROUPS,
                        value_attrs=_attrs()).launch(grid=(GRID, 1, 1), block=(DOWN_THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_tiles(SEID: fx.Tensor, NV: fx.Tensor, TILES: fx.Tensor, COUNT: fx.Tensor, NB: fx.Int32, CAP: fx.Int32,
                 E: fx.Constexpr[int], BLOCK: fx.Constexpr[int], stream: fx.Stream):
    _tiles_kernel(SEID, NV, TILES, COUNT, NB, CAP, E, BLOCK).launch(grid=(1, 1, 1), block=(TILE_THREADS, 1, 1),
                                                                   stream=stream)


@flyc.jit
def launch_inverse(SID: fx.Tensor, NV: fx.Tensor, INV: fx.Tensor, T: fx.Int32, R: fx.Int32, GRID: fx.Int32,
                   TOPK: fx.Constexpr[int], stream: fx.Stream):
    _inverse_kernel(SID, NV, INV, T, R, TOPK).launch(grid=(GRID, 1, 1), block=(SUM_THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_sum(Y: fx.Tensor, INV: fx.Tensor, OUT: fx.Tensor, T: fx.Int32, R: fx.Int32, C0: fx.Int32,
               GRID: fx.Int32, N: fx.Constexpr[int], NS: fx.Constexpr[int], TOPK: fx.Constexpr[int],
               stream: fx.Stream):
    _sum_kernel(Y, INV, OUT, T, R, C0, N, NS, TOPK).launch(grid=(GRID, 1, 1), block=(SUM_THREADS, 1, 1),
                                                           stream=stream)
