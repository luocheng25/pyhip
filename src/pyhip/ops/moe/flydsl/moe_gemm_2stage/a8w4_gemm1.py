# SPDX-License-Identifier: MIT

"""A8W4 Gate/Up prefill kernel: the prefill_1x4 kernel of ``gemm1.py`` with MXFP4 weights.

FP8 activations are gathered by sorted id into double-buffered LDS exactly as in
``gemm1.py``; the weight operand is read in the ``a8w4_common`` layout (8 bytes per lane and
64-K block), decoded to FP8 in registers and fed to MFMA 16x16x32 FP8. The per-channel
factor and the per-token activation scale are applied in the epilogue, then SiLU(gate) * up
is written as float32 by (token, slot): the intermediate is quantized from f32 so its FP8
rounding matches the reference.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import a8w4_common as a8w4
from . import common as fxh
from .common import make_gemm_helpers


def compile_a8w4_gemm1(
    *, N, K, TOPK, BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128, TILE_K=256, E=None
):
    """Build the A8W4 Gate/Up launcher.

    ``N`` = 2 * I (weight rows ``[0, I)`` gate, ``[I, 2I)`` up; ``I % (BLOCK_TILE_SIZE_N/2) == 0``),
    ``K`` = hidden (``K % (2 * TILE_K) == 0``). ``BLOCK_TILE_SIZE_M`` must equal the sorting block.
    Launch arguments: x FP8 [M, K], w and w_delta (``a8w4_common`` layout, uint8), out f32
    [M, TOPK, I] (``M * TOPK * I * 4 < 2**31``), sorted_ids, sorted_weights (unused),
    sorted_expert_ids, num_valid_ids, w_scale f32 [E, N], x_scale f32 [M], then M, task_num, stream.
    """
    BM, BN = BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N
    assert BN in (128, 256), "BLOCK_TILE_SIZE_N must be 128 or 256"
    assert (N // 2) % (BN // 2) == 0, f"I={N // 2} must be a multiple of {BN // 2}"
    assert TILE_K in (128, 256), f"TILE_K must be 128 or 256, got {TILE_K}"
    assert K % (2 * TILE_K) == 0, f"K={K} needs an even number of {TILE_K}-wide tiles"
    assert 32 <= BM <= 256 and BM % 32 == 0, f"BLOCK_TILE_SIZE_M={BM} must be 32..256, step 32"
    a_lds_size = BM * TILE_K
    assert 2 * a_lds_size <= 64 * 1024, f"A ping-pong needs {2 * a_lds_size} B of LDS"

    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    KB_PER_TILE = TILE_K // 64
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    fp8 = fx.Float8E4M3FNUZ

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[fp8, a_lds_size, 16]
        a_pong: fx.Array[fp8, a_lds_size, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers

    TensorWithIndex, _read_sorted_index, _ = make_gemm_helpers(K, fp8, BM, TOPK)

    def _store_silu_f32(gate_frag, up_frag, tid, blk_n, contiguous_n, asc_idx, M, p_output):
        # Lane holds channels 4 * (lane // 16) + v of token lane % 16 for each (m, n) MMA tile.
        log2_exp1 = -1.4426950408889634
        I = N // 2
        lane = tid % 64
        wave = tid // 64
        oob = M * (TOPK * I * 4)
        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float32, fxh._as_ptr(p_output)), fx.make_layout(1, 1)),
            max_size=False,
            num_records_bytes=fx.Int64(oob),
        )
        channel = blk_n * contiguous_n + wave * 16 + (lane // 16) * 4
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for n in range_constexpr(n_reps):
            sorted_id = asc_idx[0, n]
            token = sorted_id & 0xFFFFFF
            slot = (sorted_id >> 24) & 0xFF
            row_bytes = (token * TOPK + slot) * (I * 4)
            for m in range_constexpr(m_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                acc = []
                for j in range_constexpr(gate.numel):
                    tmp = rocdl.exp2(T.f32, _raw(gate[j] * log2_exp1))
                    acc.append((gate[j] * rocdl.rcp(T.f32, 1.0 + tmp)) * up[j])
                offset = row_bytes + (channel + m * 64) * 4
                # Padding rows carry token id >= M: redirect them past the buffer end.
                offset = (token < M).select(offset, oob)
                a8w4.store_vector(out, Vec.from_elements(acc, fx.Float32), offset, fx.Int32(0))

    def _make_1x4_tiled_mma():
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_1x4(TILE_M, TILE_N, blk_n: int, arg_p_input: fx.Tensor, w_buf, d_buf, lds):
        tid = gpu.thread_idx.x
        contiguous_n = TILE_N // 2

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        mma_atom, tiled_mma = _make_1x4_tiled_mma()

        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M, TILE_K))[None, None, 0, None]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        _val_per_thr = 16
        _thrs_k = TILE_K // _val_per_thr
        _thrs_m = 256 // _thrs_k
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        _m_per_wave = _thrs_m // 4
        tiled_copy_sortid_a = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = _read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, index_size=TILE_M
        )
        a_idx = TensorWithIndex(a_tensor, TILE_M, TILE_K, a_index_frag, a_mem_cp_g2r, tid)
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])

        # All indices must be captured before a_ping reuses sorted_lds.
        gpu.barrier()

        swz = fx.SwizzleType.get(3, 4, 3)
        a_ping = fx.make_view(
            lds.gemm.a_ping.ptr,
            fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
            ),
        )
        a_pong = fx.make_view(
            lds.gemm.a_pong.ptr,
            fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
            ),
        )
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
        a_r2s = fx.make_tiled_copy(uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K))
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)
        a_lds_r = [
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_ping),
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_pong),
        ]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)

        w_fake = fx.make_view(
            fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, TILE_K), (TILE_K, 1))
        )
        # Decoded FP8 weight operands of the current tile: [gate, up], each (8, m_reps, (2, k_iters)).
        w_frag = [tiled_mma.make_fragment_A(w_fake), tiled_mma.make_fragment_A(w_fake)]

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[None, None, 0, 0]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        num_tiles = K // TILE_K
        _m_reps = fx.size(fx.get_shape(c_gate)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_gate)[2]).to_py_value()
        k_iters = KB_PER_TILE

        w_voff, d_voff = a8w4.lane_offsets(tid, K)
        # Raw FP4 dwords and delta dwords per [slot][gate/up][m]; slot = tile parity.
        w_regs = [
            [
                [
                    [fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Uint32) for _ in range(k_iters)]
                    for _ in range(_m_reps)
                ]
                for _ in range(2)
            ]
            for _ in range(2)
        ]
        d_regs = [
            [[fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Uint32) for _ in range(_m_reps)] for _ in range(2)]
            for _ in range(2)
        ]

        def row_block16(gu, m):
            return blk_n * (contiguous_n // 16) + (gu * (N // 32) + m * 4)

        def load_weights(slot, k_tile):
            kb0 = k_tile * KB_PER_TILE
            for gu in range_constexpr(2):
                for m in range_constexpr(_m_reps):
                    rb = row_block16(gu, m)
                    for ki in range_constexpr(k_iters):
                        w_regs[slot][gu][m][ki].store(a8w4.load_dwordx2(
                            w_buf, w_voff,
                            fx.Int32((rb * KB + kb0 + ki) * a8w4.W_BLOCK_BYTES)))
                    d_regs[slot][gu][m].store(a8w4.load_dword(
                        d_buf, d_voff, fx.Int32((rb * KQ + kb0 // 4) * a8w4.D_BLOCK_BYTES)))

        def decode_weights(slot, k_tile, ki):
            for gu in range_constexpr(2):
                for m in range_constexpr(_m_reps):
                    if const_expr(KB_PER_TILE % 4 == 0):
                        selector = fx.Uint32(ki * 0x01010101)
                    else:
                        selector = fx.Uint32(((k_tile * KB_PER_TILE + ki) % 4) * 0x01010101)
                    lut = a8w4.fp8_lut(d_regs[slot][gu][m].load()[0], selector)
                    raw = w_regs[slot][gu][m][ki].load()
                    for c in range_constexpr(2):
                        w_frag[gu][None, m, (c, ki)].store(a8w4.fp4x8_to_fp8(raw[c], lut))

        def pipeline_stage(read_i, k_cur, k_next, do_prefetch):
            write_i = read_i ^ 1
            if const_expr(do_prefetch):
                a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)
                load_weights(write_i, k_next)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                decode_weights(read_i, k_cur, ki)
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
                            fx.mma_atom_call(
                                mma_atom,
                                c_gate[None, m, n],
                                w_frag[0][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_gate[None, m, n],
                            )
                            fx.mma_atom_call(
                                mma_atom,
                                c_up[None, m, n],
                                w_frag[1][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_up[None, m, n],
                            )
            if const_expr(do_prefetch):
                fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
            rocdl.sched_barrier(0)
            gpu.barrier()

        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag)
        load_weights(0, fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        gpu.barrier()

        acc_init = [c_gate.load(), c_up.load()]
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            pipeline_stage(0, kb, kb + 1, True)
            pipeline_stage(1, kb + 1, kb + 2, True)
            results = yield [c_gate.load(), c_up.load()]
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        pipeline_stage(0, kb, kb + 1, True)
        pipeline_stage(1, kb + 1, fx.Int32(0), False)
        return c_gate, c_up

    def _apply_dequant(c_gate_frag, c_up_frag, tid, expert_id, blk_n, contiguous_n, asc_idx, M,
                       p_w_scale, p_a_scale):
        m_reps = fx.size(fx.get_shape(c_gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_gate_frag)[2]).to_py_value()
        scale_gate = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N + blk_n * contiguous_n,
            fx.make_layout(contiguous_n, 1),
        )
        scale_up = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N + N // 2 + blk_n * contiguous_n,
            fx.make_layout(contiguous_n, 1),
        )
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        scale_copy = fx.make_tiled_copy(
            cp_atom_scale,
            fx.make_layout(((16, 4, 4), 4), ((0, 4, 16), 1)),
            fx.make_tile(64),
        )
        sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
        su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
        gate_scale = fx.make_fragment_like(sg_thr)
        up_scale = fx.make_fragment_like(su_thr)
        fx.copy(cp_atom_scale, sg_thr, gate_scale)
        fx.copy(cp_atom_scale, su_thr, up_scale)

        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)), fx.make_layout(M, 1)),
            max_size=False,
        )
        a_sc_n = [a_scale_tensor[asc_idx[0, n] & 0xFFFFFF] for n in range_constexpr(n_reps)]
        for m in range_constexpr(m_reps):
            sg_v = gate_scale[None, m].load()
            su_v = up_scale[None, m].load()
            for n in range_constexpr(n_reps):
                a_sc = a_sc_n[n]
                cg = c_gate_frag[None, m, n].load()
                cu = c_up_frag[None, m, n].load()
                cg_items = []
                cu_items = []
                for v in range_constexpr(4):
                    cg_items.append(cg[v] * sg_v[v] * a_sc)
                    cu_items.append(cu[v] * su_v[v] * a_sc)
                c_gate_frag[None, m, n].store(Vec.from_elements(cg_items, fx.Float32))
                c_up_frag[None, m, n].store(Vec.from_elements(cu_items, fx.Float32))

    @flyc.kernel
    def a8w4_gateup_prefill(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        in_ptr = fx.recast_iter(fp8, fxh._as_ptr(p_input))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if e_idx * BM < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BM),
                fx.make_layout(BM, 1),
            )
            expert_id = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))[e_idx]
            contiguous_n = BN // 2
            w_buf = a8w4.expert_bytes(p_weight, expert_id, W_BYTES)
            d_buf = a8w4.expert_bytes(p_w_delta, expert_id, D_BYTES)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            if tid < BM:
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Capture scatter indices before a_ping reuses sorted_lds.
            asc_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                fx.make_tile(16),
            )
            asc_lds = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
            asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
            asc_idx = fx.make_fragment_like(asc_thr)
            fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32), asc_thr, asc_idx)

            c_gate_frag, c_up_frag = gemm_1x4(BM, BN, blk_n, arg_p_input, w_buf, d_buf, lds)
            _apply_dequant(c_gate_frag, c_up_frag, tid, expert_id, blk_n, contiguous_n, asc_idx,
                           M, p_w_scale, p_a_scale)
            _store_silu_f32(c_gate_frag, c_up_frag, tid, blk_n, contiguous_n, asc_idx, M, p_output)

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BN)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        a8w4_gateup_prefill(
            p_input,
            p_weight,
            p_w_delta,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
        ).launch(grid=(num_n_blocks, task_num, 1), block=(256, 1, 1), stream=stream)

    return launch
