"""Short-K down GEMM with the top-k reduction folded into the epilogue: BF16 packed atomics into OUT [T, H]
(readme 4.12). Same MFMA / W2 ring / scheduling as kernels._down_direct_body; only the epilogue differs.

Retired M-block words: lane (r16, g) holds dwords 4 g .. + 3 of row r16 in a 32-column unit (the MFMA output
order of _down_direct_body). Two DPP quad exchanges (register bit 0 <-> lane bit 0, register bit 1 <-> lane bit 1)
regroup them so that register i of lane L holds row 4 ((L >> 2) & 3) + i, dword 4 (L >> 4) + (L & 3): each of
the four global_atomic_pk_add_bf16 then covers four complete 64-byte row pieces (1.40 TB/s in atomic_probe tile64
vs 0.36 TB/s for the unshuffled order). Padding rows are masked with EXEC (inline asm s_and_saveexec); an EXEC = 0
VMEM instruction is still counted by vmcnt on gfx942, so the explicit vmcnt waits of the W2 ring stay exact.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942.kernels import (  # noqa: F401
    BK, BM, _as_f32, _buffer, _dma, _dma_chunk, _down_lane, _fma4, _load_f32, _load_i32, _load_v4, _maybe, _mfma,
    _min, _operand, _pack_bf16, _pin_i32, _read_address, _read_scalar_g2r, _read_unit, _retire, _scalar_ready,
    _stage_end, _uniform, _wait)

DOWN_COLS = 512    # column group width (the driver sets CW)
VALU_SLOTS = 3     # VALU instructions scheduled after each MFMA
UNIT_ATOMICS = 16  # VMEM instructions per unit (4 retires x 4 atomics)
DMA_WAIT = 2 * UNIT_ATOMICS  # vmcnt before a W2 chunk is read (timing probes may raise it: wrong results)


def _i64():
    return ir.IntegerType.get_signless(64)


def _xchg(a, b, hi, ctrl):
    """Exchange a register bit with a lane bit (partner = DPP quad_perm ctrl): lanes with hi clear keep a and get
    the partner's a as b, lanes with hi set keep b and get the partner's b as a."""
    t = hi.select(a, b)
    u = fx.Int32(rocdl.update_dpp(fx.Int32.ir_type, t.ir_value(), t.ir_value(), ctrl, 0xF, 0xF, False))
    return hi.select(u, a), hi.select(b, u)


def _atomic(mask, voffset, word, base, imm):
    llvm.inline_asm(_i64(), [fx.Int64(mask).ir_value(), fx.Int32(voffset).ir_value(), fx.Int32(word).ir_value(),
                             fx.Int64(base).ir_value()],
                    f"s_and_saveexec_b64 $0, $1\n\tglobal_atomic_pk_add_bf16 $2, $3, $4 offset:{imm}\n\t"
                    "s_mov_b64 exec, $0", "=&s,s,v,v,s,~{scc}", has_side_effects=True)


def _retire_atomic(item, ctx, masks=None):
    """BF16 words of an M-block (item = partials, scales, M-block, byte offset from ctx base) -> OUT atomics."""
    parts, s0, s1, mb, rel = item
    base, ao, all_masks, hi0, hi1 = ctx
    w = _retire((parts, s0, s1))
    a0, a1 = _xchg(w[0], w[1], hi0, 0xB1)
    a2, a3 = _xchg(w[2], w[3], hi0, 0xB1)
    b0, b2 = _xchg(a0, a2, hi1, 0x4E)
    b1, b3 = _xchg(a1, a3, hi1, 0x4E)
    masks = all_masks[mb] if masks is None else masks
    for i, word in enumerate((b0, b1, b2, b3)):
        _atomic(masks[i], ao[mb][i], word, base, rel)


def _unit_atomic(wf, x, scales, item, ctx, rel, first, prefetch):
    """64 MFMAs of a 32-column unit; each M-block retires the previous one (the first: `item`, masked by
    `first` if given) as 4 atomics. Returns the unit's last M-block as the next item."""
    rocdl.sched_barrier(0)
    following = prefetch() if prefetch is not None else None
    for mb in range(4):
        acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)] for _ in range(2)]
        for step in range(4):
            for v in range(2):
                for kb in range(2):
                    acc[v][kb] = _mfma(_operand(wf[v][kb], step), _operand(x[mb][kb], step), acc[v][kb])
        _retire_atomic(item, ctx, first if mb == 0 else None)
        item = (acc, scales[mb][0], scales[mb][1], mb, rel)
    if prefetch is not None:
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 8, 0)
    for _ in range(64):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
        rocdl.sched_group_barrier(0x002, VALU_SLOTS, 0)
    rocdl.sched_barrier(0)
    return item, following


@flyc.jit
def _down_atomic_body(A2, W, A2S, WS, SID, OUT, storage, R, T, mblk, cg, expert, N: fx.Constexpr[int],
                      K: fx.Constexpr[int]):
    """OUT[token, cg * DOWN_COLS ..] += route-weighted down rows of M-block mblk (weights folded into A2S)."""
    _helpers = (_load_v4, _load_i32, _load_f32, _dma, _down_lane, _dma_chunk, _read_unit, _read_address, _fma4,
                _retire, _xchg, _atomic, _retire_atomic, _unit_atomic, _maybe, _operand, _mfma, _pack_bf16, _as_f32,
                _buffer, _min, _pin_i32, _read_scalar_g2r, _scalar_ready, _stage_end, _uniform, _wait, _i64)
    assert K == 256, "direct down: K = 256 (two K-blocks per unit)"
    KB = K // BK
    ITERS = DOWN_COLS // 128
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g = lane & 15, lane >> 4
    row0 = mblk * BM + wave * 64
    col0 = cg * DOWN_COLS
    a2_res = _buffer(fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * K, fx.make_layout(64 * K, 1)), 64 * K)
    a2s_res = _buffer(A2S, KB * R * 4)
    sid_res = _buffer(SID, R * 4)
    w_res = _buffer(fx.make_view(fx.get_iter(W) + (fx.Int64(expert) * N + fx.Int64(col0)) * K,
                                 fx.make_layout(DOWN_COLS * K, 1)), DOWN_COLS * K)
    out_ptr = fx.Int64(fx.ptrtoint(fx.get_iter(OUT)))
    lane_off = _pin_i32(_down_lane(lane))
    ws_row = (expert * (N // 128) + col0 // 128) * KB

    _dma_chunk(w_res, storage, wave, lane_off, 0, fx.Int32(0), K)
    _dma_chunk(w_res, storage, wave, lane_off, 1, fx.Int32(1), K)
    x = [[[_load_v4(a2_res, (16 * mb + r16) * K + kb * 128 + 32 * g + 16 * j) for j in range(2)]
          for kb in range(KB)] for mb in range(4)]
    xs = [[_load_f32(a2s_res, (kb * R + row0 + 16 * mb + r16) * 4) for kb in range(KB)] for mb in range(4)]
    # After the exchange, atomic i of M-block mb: row 16 mb + 4 quad + i, dword dw of the unit.
    quad, dw = (lane >> 2) & 3, 4 * g + (lane & 3)
    tok = [[_load_i32(sid_res, (row0 + 16 * mb + 4 * quad + i) * 4) & 0xFFFFFF for i in range(4)]
           for mb in range(4)]
    requests = [_read_scalar_g2r(WS, ws_row + kb) for kb in range(KB)]
    _wait(vmcnt=0)
    ws_first = [_scalar_ready(r) for r in requests]
    ao = [[_pin_i32((tok[mb][i] < T).select(tok[mb][i] * (N * 2) + dw * 4, fx.Int32(-1))) for i in range(4)]
          for mb in range(4)]
    masks = [[fx.Int64(rocdl.ballot(_i64(), tok[mb][i] < T)) for i in range(4)] for mb in range(4)]
    hi0, hi1 = (lane & 1) != 0, (lane & 2) != 0
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    b = (r16 >> 2) & 1
    w_read = [[_pin_i32(shared + (r16 >> 3) * (16 * K) + (2 * g + j) * 256 + (8 * b + 4 * (b ^ v) + (r16 & 3)) * 16)
               for j in range(2)] for v in range(2)]
    _stage_end()
    wf_first = _read_unit(w_read, 0, 0, K)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)

    @flyc.jit
    def pair(f0, f1, f2, f3, f4, f5, f6, f7, p0, p1, p2, p3, s0, s1, ws0, ws1, it, LAST: fx.Constexpr[bool]):
        """Columns 128 it .. + 127 as units U0-U3 (see _down_direct_body.pair); atomics instead of stores, so
        16 VMEM instructions per unit follow each W2 chunk DMA before its vmcnt wait."""
        it = fx.Int32(it)
        wf = [[[fx.Vector(f0), fx.Vector(f1)], [fx.Vector(f2), fx.Vector(f3)]],
              [[fx.Vector(f4), fx.Vector(f5)], [fx.Vector(f6), fx.Vector(f7)]]]
        ctx = (out_ptr + fx.Int64((col0 + it * 128) * 2), ao, masks, hi0, hi1)
        item = ([[fx.Vector(p0), fx.Vector(p1)], [fx.Vector(p2), fx.Vector(p3)]], fx.Float32(s0), fx.Float32(s1),
                3, -64)
        first = [(it > 0).select(masks[3][i], fx.Int64(0)) for i in range(4)]
        ws = (_as_f32(ws0), _as_f32(ws1))
        scales = [[fx.Float32(xs[mb][kb]) * ws[kb] for kb in range(KB)] for mb in range(4)]
        following = _min(it + 1, fx.Int32(ITERS - 1))
        requests = [_read_scalar_g2r(WS, ws_row + following * KB + kb) for kb in range(KB)]
        item, wf = _unit_atomic(wf, x, scales, item, ctx, 0, first, lambda: _read_unit(w_read, 0, 1, K))
        _wait(lgkmcnt=0)
        _wait(vmcnt=DMA_WAIT)
        _stage_end()
        if fx.const_expr(not LAST):
            _dma_chunk(w_res, storage, wave, lane_off, 0, it * 2 + 2, K)
        item, wf = _unit_atomic(wf, x, scales, item, ctx, 64, None, lambda: _read_unit(w_read, 1, 0, K))
        _wait(lgkmcnt=0)
        item, wf = _unit_atomic(wf, x, scales, item, ctx, 128, None, lambda: _read_unit(w_read, 1, 1, K))
        _wait(lgkmcnt=0)
        if fx.const_expr(not LAST):
            _wait(vmcnt=DMA_WAIT)
            _stage_end()
            _dma_chunk(w_res, storage, wave, lane_off, 1, it * 2 + 3, K)
        item, following_wf = _unit_atomic(wf, x, scales, item, ctx, 192, None,
                                          _maybe(not LAST, lambda: _read_unit(w_read, 0, 0, K)))
        _wait(lgkmcnt=0)
        ws_next = [_scalar_ready(r) for r in requests]
        wf = wf if following_wf is None else following_wf
        parts = item[0]
        return (*[wf[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)],
                parts[0][0], parts[0][1], parts[1][0], parts[1][1], item[1], item[2], *ws_next)

    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (*[wf_first[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)], zero, zero, zero, zero,
             fx.Float32(0.0), fx.Float32(0.0), *ws_first)
    for it in range(fx.Int32(0), fx.Int32(ITERS - 1), fx.Int32(1)):
        state = pair(*state, it, False)
    state = pair(*state, fx.Int32(ITERS - 1), True)
    last = ([[state[8], state[9]], [state[10], state[11]]], state[12], state[13], 3, 0)
    _retire_atomic(last, (out_ptr + fx.Int64((col0 + DOWN_COLS - 32) * 2), ao, masks, hi0, hi1))
