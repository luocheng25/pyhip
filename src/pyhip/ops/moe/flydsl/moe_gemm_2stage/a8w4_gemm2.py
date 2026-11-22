# SPDX-License-Identifier: MIT

"""A8W4 Down prefill kernel: the default prefill Down kernel of ``gemm2.py`` with MXFP4 weights.

Each workgroup keeps its BM x I FP8 activation tile (gathered by sorted id) in registers and
walks the hidden dimension in 64-row blocks; each wave reads its 16 weight rows in the
``a8w4_common`` layout, decodes them to FP8, runs MFMA 16x16x32 FP8 and scales by the
channel factor, the activation scale and the routing weight. Rows are written by sorted
position as BF16 for ``sorted_sum``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr

from . import a8w4_common as a8w4
from . import common as fxh
from .common import _f32_to_bf16


def compile_a8w4_gemm2(*, N, K, TOPK, BLOCK_TILE_SIZE_M=64, E=None):
    """Build the A8W4 Down launcher.

    ``N`` = hidden (``N % 256 == 0``), ``K`` = I (``K % 64 == 0``; pad I in the weights);
    ``BLOCK_TILE_SIZE_M * K`` must be a multiple of 4096 and at most 64 KiB.
    Launch arguments: x FP8 [M, TOPK, K], w and w_delta (``a8w4_common`` layout, uint8),
    out BF16 [task_num * BLOCK_TILE_SIZE_M, N] by sorted row, sorted_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, w_scale f32 [E, N], x_scale f32 [M, TOPK], then M,
    task_num, stream.
    """
    BM = BLOCK_TILE_SIZE_M
    assert K % 64 == 0, f"Down K={K} must be a multiple of 64"
    assert N % 256 == 0, f"Down N={N} must be a multiple of 256"
    assert 16 <= BM <= 256 and BM % 16 == 0
    assert BM * K <= 64 * 1024, f"Down A tile needs {BM * K} B of LDS"
    assert BM * K % (256 * 16) == 0, f"Down A gather needs BLOCK_TILE_SIZE_M * K={BM * K} % 4096 == 0"
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    fp8 = fx.Float8E4M3FNUZ
    flyobj = fxh.FlyObjCache()

    def _cvt_f32_to_bf16(c_frag, c_frag_bf16):
        c_frag_bf16.store(_f32_to_bf16(c_frag.load()))
        return c_frag_bf16

    @flyc.kernel
    def a8w4_down_prefill(
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
        tid = fx.gpu.thread_idx.x
        e_idx = fx.gpu.block_idx.y
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BM < max_valid_id:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fp8)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(e_idx) * (BM * N), (BM, N)
            )
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_idx * BM, (BM,), fx.Int32
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_idx * BM, (BM,), fx.Float32
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]
            w_buf = a8w4.expert_bytes(p_weight, expert_id, W_BYTES)
            d_buf = a8w4.expert_bytes(p_w_delta, expert_id, D_BYTES)
            arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)

            BLOCK_N = 64
            BLOCK_K = 64
            swz = fx.SwizzleType.get(3, 4, 3)

            @fx.union
            class SharedStorage:
                A: fx.Array[fp8, BM * K]
                C: fx.Array[fx.BFloat16, 2 * BM * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BM, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BM, BLOCK_N, 2), (1, 0, 2))
            )
            layoutCt = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BLOCK_N, BM, 2), (0, 1, 2))
            )
            ldsC = lds.C.peek().view(layoutC)
            ldsCt = lds.C.peek().view(layoutCt)

            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M) * (TOPK * K),
            )
            cp_atom = flyobj.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            cp_ldsA0 = flatten_A(ldsA0)
            cp_rows = flatten_A(fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids)))
            cp_cols = flatten_A(fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0)))
            for dst, row, col in fxh.all_copy_atoms(
                cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_p_input, (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]), 128
                )
                fx.copy(cp_atom, atom_A, dst)
            fx.gpu.barrier()

            ldsA = fx.flat_divide(ldsA0, (BM, BLOCK_K))
            nBN = N // BLOCK_N
            mm = flyobj.create_thr_mma(fp8, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input), fx.make_ordered_layout((BLOCK_N, BM), (0, 1))
            )
            fragC = [mm.make_fragment_C(c_fake_tensor), mm.make_fragment_C(c_fake_tensor)]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            fx.gpu.barrier()  # Finish reading ldsA before ldsC reuses its storage.

            arg_w_scale = fx.flat_divide(
                fx.make_view(fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout((N, 1), (1, 0))),
                (BLOCK_N, 1),
            )
            arg_a_scale = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * 4,
            )

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights), fx.make_layout((BLOCK_N, BM), (0, 1))
            )
            frag_sorted_weight = flyobj.load_tiled_mma_fragC(mm, sorted_weights, copy_atom_bits=32)
            sc_atom = flyobj.get_buffer_copy_atom(p_a_scale.dtype, 32)
            coord_tensor = fx.make_view(
                fx.get_iter(arg_p_sorted_ids), fx.make_layout((BLOCK_N, BM), (0, 1))
            )
            frag_coord = flyobj.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
            frag_row_scale = mm.make_fragment_C(coord_tensor)
            frag_row_scale_r = flyobj.get_tiled_mma_retile(mm, frag_row_scale, "C", copy_atom=sc_atom)
            for dst, coord in fxh.all_elements(frag_row_scale_r, frag_coord):
                sorted_id = coord[0].bitcast(fx.Uint32)
                atom = fxh.atom_tensor(arg_a_scale, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32)
                fx.copy(sc_atom, atom, dst)
            for frag_pt, frag_sw in fxh.all_elements(frag_row_scale, frag_sorted_weight):
                frag_pt.store(frag_pt.load() * frag_sw.load())

            # Decoded FP8 weight operand of one 64-row block: (8, 1, 2, KB).
            w_fake = fx.make_view(
                fx.get_iter(arg_p_input), fx.make_layout((BLOCK_N, BLOCK_K, KB), (K, 1, BLOCK_K))
            )
            frag_w = mm.make_fragment_A(w_fake)
            w_voff, d_voff = a8w4.lane_offsets(tid, K)
            w_regs = [
                [fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Uint32) for _ in range(KB)]
                for _ in range(2)
            ]
            d_regs = [
                [fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Uint32) for _ in range(KQ)]
                for _ in range(2)
            ]
            frag_pc_scales = [None, None]

            def load_block(n, slot):
                rb = n * (BLOCK_N // 16)
                for kb in range_constexpr(KB):
                    w_regs[slot][kb].store(a8w4.load_dwordx2(
                        w_buf, w_voff, fx.Int32((rb * KB + kb) * a8w4.W_BLOCK_BYTES)))
                for q in range_constexpr(KQ):
                    d_regs[slot][q].store(a8w4.load_dword(
                        d_buf, d_voff, fx.Int32((rb * KQ + q) * a8w4.D_BLOCK_BYTES)))
                if const_expr(frag_pc_scales[slot] is None):
                    frag_pc_scales[slot] = flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n, 0], copy_atom_bits=128
                    )
                else:
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n, 0], frag_pc_scales[slot]
                    )

            def gemm_compute(slot, fragC):
                for kb in range_constexpr(KB):
                    lut = a8w4.fp8_lut(
                        d_regs[slot][kb // 4].load()[0], fx.Uint32((kb % 4) * 0x01010101)
                    )
                    raw = w_regs[slot][kb].load()
                    for c in range_constexpr(2):
                        frag_w[None, 0, c, kb].store(a8w4.fp4x8_to_fp8(raw[c], lut))
                fragC.fill(0)
                for k in range_constexpr(KB):
                    fx.gemm(mm, fragC, frag_w[None, None, None, k], frag_act[None, None, None, 0, k], fragC)
                for fc, fpc in fxh.all_elements(fragC, frag_pc_scales[slot]):
                    fc.store(fc.load() * fpc.load())

            tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )
            thrv_ldsC = tcopyLDS.partition_S(ldsC)
            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)
            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_row_scale):
                    fc.store(fc.load() * fsw.load())
                _cvt_f32_to_bf16(fragC, fragC_bf16)
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            arg_p_output = fx.flat_divide(arg_p_output, (BM, BLOCK_N))
            cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)
            thrv_out = tcopyLDS.partition_D(arg_p_output)
            fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

            def postprocess_store2vmem(n, ldsc_idx):
                fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])

            load_block(0, 0)
            gemm_compute(0, fragC[0])
            load_block(1, 1)
            postprocess_store2lds(fragC[0], 0)
            fx.gpu.barrier()
            for n, state in range(0, nBN - 2, 2, init=[]):
                postprocess_store2vmem(n, 0)
                load_block(n + 2, 0)
                gemm_compute(1, fragC[1])
                postprocess_store2lds(fragC[1], 1)
                fx.gpu.barrier()

                postprocess_store2vmem(n + 1, 1)
                load_block(n + 3, 1)
                gemm_compute(0, fragC[0])
                postprocess_store2lds(fragC[0], 0)
                fx.gpu.barrier()

            postprocess_store2vmem(nBN - 2, 0)
            gemm_compute(1, fragC[1])
            postprocess_store2lds(fragC[1], 1)
            fx.gpu.barrier()
            postprocess_store2vmem(nBN - 1, 1)

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
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        a8w4_down_prefill(
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
        ).launch(grid=(1, task_num, 1), block=(256, 1, 1), stream=stream)

    return launch
