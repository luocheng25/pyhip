# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# pyhip: derived from ROCm/aiter@df3e0b2f aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/ (the M0
# decode kernels, unchanged in pyhip's moe_gemm_2stage/):
#   common.py  make_gemm_helpers: gemm_splitk, _setup_b_operand   -> _make_fp4_gemm_splitk
#   gemm1.py   moe_2stage_gateup_splitk / _batch1, launch_splitk / launch_batch1,
#              _apply_scale_gateup_bf16, _make_gateup_weight_view -> _build_gateup
#   gemm2.py   moe_2stage_down_splitk / _batch1, launch_splitk / launch_batch1,
#              _apply_down_scale, _make_down_weight_view           -> _build_down
# Changes: the weights are mxfp4 in the MXFP4Weights format (weights.py) instead of FP8. B reads 8 bytes
# (16 K) per lane and N repeat instead of 16, plus one delta dword per 64 weight rows; the fd1 nibbles
# are decoded with a v_perm table (PR #64 fd2) into the FP8 dwords the FP8 path loads, which the
# unchanged _cvt_fp8_bf16 widens to BF16. The kernels take p_w_delta after p_w_scale, and p_w_scale is
# the per-channel factor (the ptpc epilogue is unchanged). Only SiLU and the ptpc scale path are kept;
# batch1 drops has_num_local_tokens and its launchers take the batch size instead of batch * TOPK.
# Added (not upstream): gemv_splitk and the gemv kernels, batch1's grid and weight reads with the
# products on the VALU (v_cvt_pk_f32_fp8 + v_pk_fma_f32) instead of BF16 MFMA.

"""W4A16 decode kernels for mxfp4 MoE weights and BF16 activations, FP32 accumulation.

batch1 / split-K multiply with BF16 MFMA; gemv (one CTA group per route, like batch1) uses packed f32
FMAs on the VALU, because a batch-1 MFMA uses one of its 16 columns and sustained BF16 MFMA issue
throttles the MI308X clock.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..moe_gemm_2stage import common as fxh
from ..moe_gemm_2stage.common import (
    _cvt_fp8_bf16,
    _f32_to_bf16,
    _select,
    get_device_cache_key,
    make_gemm_helpers,
)

TILE_K = 64
# FP8 E4M3FNUZ bytes of |E2M1| * 2^5 for magnitudes 0..3 / 4..7; byte 0 is 0x80 so that subtracting
# the delta byte (<= 88) never borrows, and is masked to 0 afterwards.
_LUT_LO, _LUT_HI = 0x6C686080, 0x7C787470


def _perm(hi, lo, sel):
    return fx.Uint32(rocdl.perm_b32(fx.Uint32(hi).ir_value(), fx.Uint32(lo).ir_value(),
                                    fx.Uint32(sel).ir_value()))


def _decode_fp4x8(raw, lo, hi):
    """One dword of fd1 nibbles -> FP8 dwords for k0..k3 and k4..k7 (PR #64 _fp4x8_to_fp8x8_fast)."""
    even = ((raw << 4) & fx.Uint32(0x80808080)) | _perm(hi, lo, raw & fx.Uint32(0x07070707))
    odd = (raw & fx.Uint32(0x80808080)) | _perm(hi, lo, (raw >> 4) & fx.Uint32(0x07070707))
    return even, odd


def _fp8_words(raw, dwords, rep_sel):
    """Per N repeat: broadcast the row's delta byte, shift the |E2M1| -> FP8 table by it, and decode the
    16 nibbles (two dwords of ``raw``) into four FP8 dwords (k0..3, 4..7, 8..11, 12..15)."""
    words = []
    for n in range_constexpr(len(rep_sel)):
        slot, sel = rep_sel[n]
        d = _perm(dwords[slot], dwords[slot], sel)
        lo = (fx.Uint32(_LUT_LO) - d) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(_LUT_HI) - d
        for v in range_constexpr(2):
            even, odd = _decode_fp4x8(fx.Uint32(raw[2 * n + v]), lo, hi)
            words.append(even)
            words.append(odd)
    return words


def _u32_ptr(p, align):
    it = fxh._as_ptr(p)
    return fx.recast_iter(fx.PointerType.get(fx.Uint32.ir_type, it.memspace, align), it)


def _delta_source(p_w_delta, expert_id, N, K, slot_rows, rep_sel):
    """Per-lane delta reads: ``slot_rows`` are the 64-row block indices the CTA's N repeats use,
    ``rep_sel[n] = (slot, v_perm selector of the row's byte)``."""
    groups = K // 32
    tid = gpu.thread_idx.x
    view = fx.make_view(_u32_ptr(p_w_delta, 4) + fx.Int64(expert_id) * (N * groups // 4),
                        fx.make_layout(N * groups // 4, 1))
    # The lane reads 16 K of 32-K group tid // 32 of each K tile, for row tid % 16 of a 16-row block.
    lane = (tid // 32) * 16 + tid % 16
    bases = [nb * (groups * 16) + lane for nb in slot_rows]
    return fx.rocdl.make_buffer_tensor(view, max_size=False), bases, rep_sel


def _make_fp4_gemm_splitk(K, BLOCK_TILE_SIZE_M, TOPK):
    TensorWithIndex, _read_sorted_index, _ = make_gemm_helpers(
        K, fx.Float8E4M3FNUZ, BLOCK_TILE_SIZE_M, TOPK
    )

    def _setup_b_operand(arg_p_weight, arg_p_input, tiled_mma, blk_n, TILE_N, tile_k_per_wg, tid):
        b_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_layout((TILE_N, tile_k_per_wg), (tile_k_per_wg, 1)),
        )
        b_frag = [
            tiled_mma.make_fragment_B(b_fake_tensor),
            tiled_mma.make_fragment_B(b_fake_tensor),
        ]
        # arg_p_weight is a dword view (N, K / 8): 8 fd1 nibbles per dword.
        b_tensor_u32 = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        b_tile = fx.flat_divide(b_tensor_u32, fx.make_tile(TILE_N, tile_k_per_wg // 8))[
            None, None, blk_n, None
        ]
        b_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        # The MMA's B mapping with 8 values per dword: a lane reads two dwords (16 K) per N repeat.
        n_mma = fx.get_scalar(fx.size(fx.select(tiled_mma.tile_size_mnk, [1])))
        _tvB = tiled_mma.tv_layout_B_tiled
        _n0 = fx.get_scalar(_tvB.shape[0][0])
        _n1 = fx.get_scalar(_tvB.shape[0][1])
        _s0 = fx.get_scalar(_tvB.stride[0][0])
        _s1 = fx.get_scalar(_tvB.stride[0][1])
        # Lane l reads 16 consecutive K at 16 * (l // 16) of the K tile (_delta_source relies on it).
        assert (_n0, _s0, _s1) == (16, 1, 16 * n_mma), (_n0, _n1, _s0, _s1)
        _s0 = _s0 if _s0 < 16 else _s0 // 8
        _s1 = _s1 if _s1 < 16 else _s1 // 8
        tv_u32 = fx.make_layout(((_n0, _n1), 2), ((_s0, _s1), n_mma))
        tile_mn = fx.make_tile(
            fx.make_layout(n_mma, 1),
            fx.make_layout(tile_k_per_wg // 8, 1),
        )
        b_tiled_thr = fx.make_tiled_copy(b_cp_atom_r, tv_u32, tile_mn).get_slice(tid)
        b_tensor_thr = b_tiled_thr.partition_S(b_tile)
        b_raw = [
            fx.make_fragment_like(b_tensor_thr[None, None, None, 0], fx.Uint32),
            fx.make_fragment_like(b_tensor_thr[None, None, None, 0], fx.Uint32),
        ]
        return b_cp_atom_r, b_tensor_thr, b_frag, b_raw

    @flyc.jit
    def gemm_splitk(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,
        arg_p_input: fx.Tensor,
        arg_p_weight: fx.Tensor,
        delta,
        lds,
        splitk_waves=4,
        a_with_index=True,
    ):
        tid = gpu.thread_idx.x

        tile_k_per_wg = TILE_K * splitk_waves
        assert K % tile_k_per_wg == 0, (
            f"split-K requires K to be divisible by TILE_K * splitk_waves, "
            f"got K={K}, TILE_K={TILE_K}, splitk_waves={splitk_waves}"
        )

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        a_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), arg_p_input.dtype)

        rep_k_per_lane = 4
        k_perm = fx.make_tile(
            None,
            None,
            fx.make_layout(
                (4, 4 * splitk_waves, rep_k_per_lane), (1, 4 * rep_k_per_lane, 4)
            ),
        )
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            fx.make_layout((1, 1, splitk_waves), (0, 0, 1)),
            k_perm,
        )
        if const_expr(a_with_index):
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds,
                fx.make_layout(((16, 4 * splitk_waves), 1), ((1, 0), 0)),
                fx.make_tile(16),
            )
            a_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            a_tensor_thr = TensorWithIndex(
                a_tensor,
                TILE_M,
                tile_k_per_wg,
                a_index_frag,
                fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma),
                tid,
            )
            a_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((TILE_M, tile_k_per_wg), (tile_k_per_wg, 1)),
            )
            a_frag = [
                tiled_mma.make_fragment_A(a_fake_tensor),
                tiled_mma.make_fragment_A(a_fake_tensor),
            ]
        else:
            a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, tile_k_per_wg))[
                None, None, 0, None
            ]
            a_tiled_thr = fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid)
            a_tensor_thr = a_tiled_thr.partition_S(a_tile)
            a_frag = [
                tiled_mma.make_fragment_A(a_tile[None, None, 0]),
                tiled_mma.make_fragment_A(a_tile[None, None, 0]),
            ]

        a_frag_retile = [
            fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma)
            .get_slice(tid)
            .retile(a_frag[0]),
            fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma)
            .get_slice(tid)
            .retile(a_frag[1]),
        ]

        b_cp_atom_r, b_tensor_thr, b_frag, b_raw = _setup_b_operand(
            arg_p_weight, arg_p_input, tiled_mma, blk_n, TILE_N, tile_k_per_wg, tid
        )
        n_rep = TILE_N // 16
        delta_buf, delta_bases, rep_sel = delta
        n_slots = len(delta_bases)
        delta_step = tile_k_per_wg // 32 * 16
        d_frag = [fx.make_rmem_tensor(n_slots, fx.Uint32), fx.make_rmem_tensor(n_slots, fx.Uint32)]
        fp8_frag = fx.make_rmem_tensor(4 * n_rep, fx.Uint32)

        c_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input), fx.make_layout((TILE_N, TILE_M), (TILE_M, 1))
        )
        c_frag = tiled_mma.make_fragment_C(c_fake_tensor)
        c_frag.fill(0)

        num_k_iters = K // TILE_K // splitk_waves

        def _prefetch_a(k_idx, buf):
            if const_expr(a_with_index):
                a_tensor_thr.copy(a_cp_atom_r, k_idx, a_frag_retile[buf])
            else:
                fx.copy(
                    a_cp_atom_r,
                    a_tensor_thr[None, None, None, k_idx],
                    a_frag_retile[buf],
                )

        def _prefetch_b(k_idx, buf):
            fx.copy(b_cp_atom_r, b_tensor_thr[None, None, None, k_idx], b_raw[buf])
            for s in range_constexpr(n_slots):
                d_frag[buf][s] = delta_buf[delta_bases[s] + k_idx * delta_step]

        def _decode_b(buf):
            # The four FP8 dwords of the FP8 path per N repeat, widened to BF16.
            words = _fp8_words(b_raw[buf].load(), d_frag[buf].load(), rep_sel)
            fp8_frag.store(Vec.from_elements(words, fx.Uint32))
            _cvt_fp8_bf16(fp8_frag, b_frag[buf])

        _prefetch_a(fx.Int32(0), 0)
        _prefetch_b(fx.Int32(0), 0)

        acc_init = c_frag.load()

        a_load_bytes = arg_p_input.dtype.width // 8
        a_vmem_cnt = a_frag_retile[0].load().numel * a_load_bytes // 16
        b_vmem_cnt = b_raw[0].load().numel * 4 // 8 + n_slots
        vmcnt_per_prefetch = a_vmem_cnt + b_vmem_cnt

        rocdl.sched_barrier(0)

        results = acc_init
        for k2, state in range(0, num_k_iters // 2, 1, init=[acc_init]):
            c_frag.store(state[0])
            k_base = fx.Int32(k2 * 2)
            _prefetch_a(k_base + 1, 1)
            _prefetch_b(k_base + 1, 1)
            rocdl.s_waitcnt(vmcnt=vmcnt_per_prefetch)
            rocdl.sched_barrier(0)
            _decode_b(0)
            fx.gemm(tiled_mma, c_frag, b_frag[0], a_frag[0], c_frag)
            rocdl.sched_barrier(0)
            _prefetch_a(k_base + 2, 0)
            _prefetch_b(k_base + 2, 0)
            rocdl.s_waitcnt(vmcnt=vmcnt_per_prefetch)
            rocdl.sched_barrier(0)
            _decode_b(1)
            fx.gemm(tiled_mma, c_frag, b_frag[1], a_frag[1], c_frag)
            rocdl.sched_barrier(0)

            results = yield [c_frag.load()]
        c_frag.store(results)

        if const_expr(num_k_iters % 2 == 1):
            _decode_b(0)
            fx.gemm(tiled_mma, c_frag, b_frag[0], a_frag[0], c_frag)

        c_frag = _select(c_frag, [0, 2, 1])

        if const_expr(splitk_waves == 1):
            return c_frag

        if const_expr(TILE_N == 32):
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr, fx.make_ordered_layout((16 * 4, 32), order=(1, 0))
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                fx.make_layout(((16, 4, 4), (4, 2)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)
        else:
            swz = fx.SwizzleType.get(3, 3, 3)
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr,
                fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((16 * 4, 64), order=(1, 0))
                ),
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                fx.make_layout(((16, 4, 4), (4, 4)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)

        if const_expr(TILE_N == 32):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                fx.make_layout(((16, 4, 4), (1, 4)), ((32 * 2, 1, 4), (32, 16))),
                fx.make_tile(16 * 4, 16 * 1),
            )
            tile_sub_n = 16
        elif const_expr(TILE_N == 64):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                fx.make_layout(((16, 4, 4), (2, 4)), ((64 * 2, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            tile_sub_n = 32
        else:
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                fx.make_layout(((16, 4, 4), (4, 4)), ((256, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            tile_sub_n = 64
        c_tensor_thr_lds_r = c_tiled_lds_r.get_slice(tid).partition_S(c_lds)

        c_frag_vec = c_frag.load()
        shape_v = fx.size(fx.get_shape(c_tensor_thr_lds_r)[0][0]).to_py_value()
        read_rep_n = fx.size(fx.get_shape(c_tensor_thr_lds_r)[2]).to_py_value()
        if const_expr(shape_v == 1):
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout((1, TILE_M // 16, read_rep_n), (0, read_rep_n, 1)),
                fx.Float32,
            )
        else:
            stride_v = 1
            stride_sub_rn = shape_v * stride_v
            stride_rn = stride_sub_rn * (64 // tile_sub_n)
            stride_rm = stride_rn * TILE_N // tile_sub_n
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout(
                    (shape_v, TILE_M // (4 * 4), (64 // tile_sub_n, TILE_N // 64)),
                    (stride_v, stride_rm, (stride_sub_rn, stride_rn)),
                ),
                fx.Float32,
            )
        n_blocks = max(1, TILE_N // 64)
        w_size = fx.size(fx.get_shape(c_tensor_thr_lds_w)).to_py_value()
        for m in range_constexpr(TILE_M // 16):
            for n in range_constexpr(n_blocks):
                items = []
                for i in range_constexpr(w_size):
                    n_idx = n * (w_size // 4) + i // 4
                    idx = fx.get_scalar(fx.crd2idx((i % 4, m, n_idx), c_frag.layout))
                    items.append(c_frag_vec[idx])
                sub_c_frag = fx.make_fragment_like(c_tensor_thr_lds_w)
                sub_c_frag.store(Vec.from_elements(items, fx.Float32))
                fx.copy(cp_atom_lds_w, sub_c_frag, c_tensor_thr_lds_w)
                gpu.barrier()

                sub_c_frag_reduce = fx.make_fragment_like(c_tensor_thr_lds_r)
                fx.copy(cp_atom_lds_r, c_tensor_thr_lds_r, sub_c_frag_reduce)
                acc = sub_c_frag_reduce[(None, 0), None, None].load()
                for i in range_constexpr(1, 4):
                    acc += sub_c_frag_reduce[(None, i), None, None].load()

                if const_expr(shape_v == 1):
                    c_frag_reduce[0, m, None].store(acc)
                else:
                    c_frag_reduce[None, m, (None, n)].store(acc)
                gpu.barrier()

        return c_frag_reduce

    @flyc.jit
    def gemv_splitk(
        TILE_N,
        TILE_K,
        blk_n: int,
        arg_p_input: fx.Tensor,
        x_row: fx.Tensor,
        arg_p_weight: fx.Tensor,
        delta,
        x_lds: fx.Tensor,
        splitk_waves=4,
    ):
        """Batch-1 product of the CTA's TILE_N weight rows with one BF16 row on the VALU (no MFMA).

        The lanes read the same weight bytes and delta dwords as gemm_splitk; ``x_row`` is a dword view of
        the K BF16 activations, widened once per CTA into the f32 LDS view ``x_lds`` (K elements). Each
        FP8 dword is widened to two f32 pairs (v_cvt_pk_f32_fp8) and accumulated with packed f32 FMAs.
        Returns the lane's sums for rows 16 * n + lane % 16 over the wave's K slices, reduced over the
        wave's four 16-lane groups (all four hold the same sums).
        """
        tid = gpu.thread_idx.x
        tile_k_per_wg = TILE_K * splitk_waves
        assert K % tile_k_per_wg == 0, (K, tile_k_per_wg)
        # Only the B partition of gemm_splitk's MMA is used: it fixes the rows and K each lane reads.
        rep_k_per_lane = 4
        k_perm = fx.make_tile(
            None,
            None,
            fx.make_layout(
                (4, 4 * splitk_waves, rep_k_per_lane), (1, 4 * rep_k_per_lane, 4)
            ),
        )
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            fx.make_layout((1, 1, splitk_waves), (0, 0, 1)),
            k_perm,
        )
        b_cp_atom_r, b_tensor_thr, _, b_raw = _setup_b_operand(
            arg_p_weight, arg_p_input, tiled_mma, blk_n, TILE_N, tile_k_per_wg, tid
        )
        n_rep = TILE_N // 16
        delta_buf, delta_bases, rep_sel = delta
        n_slots = len(delta_bases)
        delta_step = tile_k_per_wg // 32 * 16
        d_frag = [fx.make_rmem_tensor(n_slots, fx.Uint32), fx.make_rmem_tensor(n_slots, fx.Uint32)]
        num_k_iters = K // tile_k_per_wg
        # Widen the row to f32 once per CTA (x_lds[k] = x[k]) rather than per lane and K tile.
        x_buf = fx.rocdl.make_buffer_tensor(x_row, max_size=False)
        n_threads = 64 * splitk_waves
        for r in range_constexpr((K // 2 + n_threads - 1) // n_threads):
            i = tid + r * n_threads
            if i < K // 2:
                u = Vec.from_elements([x_buf[i]], fx.Uint32)
                x_lds[2 * i] = (u << 16).bitcast(fx.Float32)[0]
                x_lds[2 * i + 1] = (u & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32)[0]
        gpu.barrier()
        # Lane (wave w, group q = lane // 16) holds K [tile_k_per_wg * k + 64 w + 16 q, +16) of its rows.
        x_lane = (tid // 64) * 64 + (tid % 64) // 16 * 16
        x_cp = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

        def _prefetch(k_idx, buf):
            fx.copy(b_cp_atom_r, b_tensor_thr[None, None, None, k_idx], b_raw[buf])
            for s in range_constexpr(n_slots):
                d_frag[buf][s] = delta_buf[delta_bases[s] + k_idx * delta_step]

        def _x_pairs(k_idx):
            # f32 pair i of the lane's 16 K is k 2i, 2i+1.
            vals = []
            for j in range_constexpr(4):
                frag = fx.make_rmem_tensor(4, fx.Float32)
                src = fx.make_view(
                    fx.get_iter(x_lds) + (k_idx * tile_k_per_wg + x_lane + 4 * j),
                    fx.make_layout(4, 1),
                )
                fx.copy(x_cp, src, frag)
                v = frag.load()
                vals += [v[e] for e in range_constexpr(4)]
            return [Vec.from_elements([vals[2 * i], vals[2 * i + 1]], fx.Float32) for i in range_constexpr(8)]

        def _fma(buf, acc, k_idx):
            # FP8 dword v of an N repeat holds k 4v..4v+3.
            xp = _x_pairs(k_idx)
            words = _fp8_words(b_raw[buf].load(), d_frag[buf].load(), rep_sel)
            out = []
            for n in range_constexpr(n_rep):
                a = Vec.from_elements([acc[2 * n], acc[2 * n + 1]], fx.Float32)
                for v in range_constexpr(4):
                    for h in range_constexpr(2):
                        w = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, words[4 * n + v], word_sel=h == 1))
                        a = fx.fma(w, xp[2 * v + h], a)
                out.append(a[0])
                out.append(a[1])
            return Vec.from_elements(out, fx.Float32)

        _prefetch(fx.Int32(0), 0)
        acc_init = Vec.filled(2 * n_rep, 0.0, fx.Float32)
        rocdl.sched_barrier(0)

        results = acc_init
        for k2, state in range(0, num_k_iters // 2, 1, init=[acc_init]):
            k_base = fx.Int32(k2 * 2)
            _prefetch(k_base + 1, 1)
            rocdl.sched_barrier(0)
            acc = _fma(0, state[0], k_base)
            rocdl.sched_barrier(0)
            _prefetch(k_base + 2, 0)
            rocdl.sched_barrier(0)
            acc = _fma(1, acc, k_base + 1)
            rocdl.sched_barrier(0)
            results = yield [acc]
        acc = results
        if const_expr(num_k_iters % 2 == 1):
            acc = _fma(0, acc, fx.Int32(num_k_iters - 1))

        sums = []
        for n in range_constexpr(n_rep):
            s = acc[2 * n] + acc[2 * n + 1]
            s = s + s.shuffle_xor(fx.Int32(16), fx.Int32(64))
            s = s + s.shuffle_xor(fx.Int32(32), fx.Int32(64))
            sums.append(s)
        return sums

    return TensorWithIndex, _read_sorted_index, gemm_splitk, gemv_splitk


def _build_gateup(N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, alg, E):
    """Gate/Up (N = 2I rows: gate then up, K = H): split-K over sorted rows, or one CTA group per route
    (batch1 with MFMA, gemv on the VALU)."""
    assert alg in ("splitk", "batch1", "gemv"), alg
    assert K % 256 == 0 and (N // 2) % 64 == 0, (N, K)
    assert BLOCK_TILE_SIZE_N in (32, 64, 128), BLOCK_TILE_SIZE_N
    c_reduce_lds_size = 16 * 64 * 4
    if alg == "splitk":
        assert BLOCK_TILE_SIZE_N % 64 == 0, "split-K Gate/Up needs BLOCK_TILE_SIZE_N % 64 == 0"

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    elif alg == "gemv":

        @fx.struct
        class SharedStorage:
            red_lds: fx.Array[fx.Float32, 4 * BLOCK_TILE_SIZE_N, 16]
            x_lds: fx.Array[fx.Float32, K, 16]

    else:

        @fx.struct
        class SharedStorage:
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    TensorWithIndex, _read_sorted_index, gemm_splitk, gemv_splitk = _make_fp4_gemm_splitk(
        K, BLOCK_TILE_SIZE_M, TOPK
    )

    def _apply_scale_gateup_bf16(c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale):
        m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()
        v_reps = fx.size(fx.get_shape(c_frag)[0]).to_py_value()
        group_layout_silu = fx.make_layout(
            ((contiguous_n, 2, N // (2 * contiguous_n)), 1),
            ((1, N // 2, contiguous_n), 0),
        )
        arg_p_scale = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N,
            fx.composition(fx.make_layout(N, 1), group_layout_silu),
        )
        scale_tile = fx.flat_divide(arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N, 1))[
            None, None, blk_n, 0
        ]
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        tiled_copy_scale = fx.make_tiled_copy(
            cp_atom_scale,
            fx.make_layout(
                ((16, 4, 4), contiguous_n // 16),
                ((contiguous_n // 16, 0, 0), 1),
            ),
            fx.make_tile(contiguous_n, 1),
        )
        scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(scale_tile)
        scale_frag = fx.make_fragment_like(scale_frag_tensor)
        fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
        for n in range_constexpr(n_reps):
            scale_vec = scale_frag[None, n, 0].load()
            for m in range_constexpr(m_reps):
                c_vec = c_frag[None, m, n].load()
                c_frag[None, m, n].store(c_vec * scale_vec)

        n_half = n_reps // 2
        if const_expr(v_reps == 1):
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout((1, m_reps, n_half), (0, n_half, 1)), fx.BFloat16
            )
        else:
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout(
                    ((v_reps, 1), m_reps, n_half), ((1, 0), n_half * v_reps, v_reps)
                ),
                fx.BFloat16,
            )
        log2_exp1 = -1.4426950408889634
        for i in range_constexpr(n_half):
            gate = c_frag[None, None, 2 * i + 0].load()
            up = c_frag[None, None, 2 * i + 1].load()
            acc = []
            for j in range_constexpr(gate.numel):
                g, u = gate[j], up[j]
                tmp = rocdl.exp2(T.f32, _raw(g * log2_exp1))
                acc.append((g * rocdl.rcp(T.f32, 1.0 + tmp)) * u)
            acc = Vec.from_elements(acc, fx.Float32)
            c_frag_bf16[None, None, i].store(_f32_to_bf16(acc))
        return c_frag_bf16

    def _weight_and_delta(p_weight, p_w_delta, expert_id, blk_n, contiguous_n):
        group_layout_silu = fx.make_layout(
            ((contiguous_n, 2, N // (contiguous_n * 2)), K // 8),
            ((1, N // 2, contiguous_n), N),
        )
        weight = fx.make_view(
            _u32_ptr(p_weight, 8) + fx.Int64(expert_id) * (N * K // 8),
            fx.composition(
                fx.make_layout(((16, N // 16), (2, K // 16)), ((2, 2 * K), (1, 32))),
                group_layout_silu,
            ),
        )
        # N repeat n is row 16 * (n % r) of the gate (n < r) or up half of this CTA's column block;
        # each half lies in one 64-row delta block.
        r = contiguous_n // 16
        row0 = blk_n * contiguous_n
        first = (row0 & 63) >> 4
        slots = [row0 // 64, (row0 + N // 2) // 64]
        rep_sel = [(n // r, (first + n % r) * 0x01010101) for n in range(2 * r)]
        return weight, _delta_source(p_w_delta, expert_id, N, K, slots, rep_sel)

    @flyc.kernel
    def moe_a8w4_gateup_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(fxh._as_ptr(p_input), fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.c_reduce_lds = lds.c_reduce_lds.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, fxh._as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            contiguous_n = 64 if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else 32

            arg_p_weight, delta = _weight_and_delta(
                p_weight, p_w_delta, expert_id, blk_n, contiguous_n
            )

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            cp_atom_w = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy64b()
                    if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                    else fx.rocdl.BufferCopy32b()
                ),
                fx.BFloat16,
            )
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                fx.make_layout(
                    ((16, 4, 4), contiguous_n // 16), ((contiguous_n, 1, 4), 16)
                ),
                fx.make_tile(16, contiguous_n),
            )
            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * N // 2 * fx.BFloat16.width // 8),
            )
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 16), 1), ((0, 1), 0)),
                fx.make_tile(16),
            )
            c_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            c_tensor = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N // 2,
                c_index_frag,
                c_tiled_g,
                tid,
                is_read_from_mem=False,
            )

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                delta,
                lds,
                splitk_waves=4,
            )

            c_frag_bf16 = _apply_scale_gateup_bf16(
                c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale
            )

            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_a8w4_gateup_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_clear_output: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(batch_idx) * K,
            fx.make_layout((1, K), (K, 1)),
        )
        arg_p_expert_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        expert_id = arg_p_expert_ids[route_idx]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        contiguous_n = min(64, BLOCK_TILE_SIZE_N // 2)

        # Routing gives pad rows expert -1, which sorting would drop.
        if expert_id >= 0:
            arg_p_weight, delta = _weight_and_delta(
                p_weight, p_w_delta, expert_id, blk_n, contiguous_n
            )

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                delta,
                lds,
                splitk_waves=4,
                a_with_index=False,
            )

            c_frag_bf16 = _apply_scale_gateup_bf16(
                c_frag, tid, expert_id, blk_n, contiguous_n, p_w_scale
            )

            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output) + fx.Int64(batch_idx) * TOPK * (N // 2),
                fx.make_layout((1, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=1 * TOPK * N // 2 * fx.BFloat16.width // 8,
            )
            cp_atom_w = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy64b()
                    if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                    else (
                        fx.rocdl.BufferCopy32b()
                        if const_expr(BLOCK_TILE_SIZE_N >= 64)
                        else fx.rocdl.BufferCopy16b()
                    )
                ),
                fx.BFloat16,
            )
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                fx.make_layout(
                    ((16, 4, 4), max(1, contiguous_n // 16)),
                    ((max(1, contiguous_n), 1, 4), 16),
                ),
                fx.make_tile(16, max(16, contiguous_n)),
            )
            c_tile = fx.flat_divide(
                out_tensor[None, e_idx, None],
                fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2),
            )[None, None, None, blk_n]
            c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
            c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)

            fx.copy(cp_atom_w, c_src, c_dst[None, None, None, 0])

        _clear_output_row(p_clear_output, blk_n, e_idx, batch_idx, tid)

    @flyc.jit
    def _clear_output_row(p_clear_output, blk_n, e_idx, batch_idx, tid):
        # Zero this token's output row for the Down atomics (same stream, after this kernel).
        clear_blocks = (N + BLOCK_TILE_SIZE_N - 1) // BLOCK_TILE_SIZE_N
        clear_threads = clear_blocks * 256
        clear_rounds = (K // 8 + clear_threads - 1) // clear_threads
        for clear_round in range_constexpr(clear_rounds):
            clear_idx = blk_n * 256 + tid + clear_round * clear_threads
            if e_idx == 0 and clear_idx < K // 8:
                clear_output = fx.make_view(
                    fxh._as_ptr(p_clear_output)
                    + fx.Int64(batch_idx) * K
                    + fx.Int64(clear_idx) * 8,
                    fx.make_layout(8, 1),
                )
                clear_frag = fx.make_fragment_like(clear_output)
                clear_frag.fill(0)
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
                    clear_frag,
                    clear_output,
                )

    @flyc.kernel
    def moe_a8w4_gateup_gemv(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_clear_output: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
    ):
        # batch1's grid and weight reads, but the products run on the VALU (gemv_splitk): BF16 MFMA
        # would use one of its 16 columns, and its issue rate is what throttles the clock on MI308X.
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(batch_idx) * K,
            fx.make_layout((1, K), (K, 1)),
        )
        x_row = fx.make_view(
            _u32_ptr(p_input, 16) + fx.Int64(batch_idx) * (K // 2),
            fx.make_layout(K // 2, 1),
        )
        arg_p_expert_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        expert_id = arg_p_expert_ids[route_idx]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        contiguous_n = min(64, BLOCK_TILE_SIZE_N // 2)

        # Routing gives pad rows expert -1, which sorting would drop.
        if expert_id >= 0:
            arg_p_weight, delta = _weight_and_delta(
                p_weight, p_w_delta, expert_id, blk_n, contiguous_n
            )
            sums = gemv_splitk(
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                x_row,
                arg_p_weight,
                delta,
                fx.make_view(lds.x_lds.ptr, fx.make_layout(K, 1)),
                splitk_waves=4,
            )
            # Tile row r < contiguous_n is gate row blk_n * contiguous_n + r, row contiguous_n + r the
            # matching up row; sum the four waves' K slices through LDS.
            red = fx.make_view(lds.red_lds.ptr, fx.make_layout(4 * BLOCK_TILE_SIZE_N, 1))
            lane = tid % 64
            if lane < 16:
                for n in range_constexpr(BLOCK_TILE_SIZE_N // 16):
                    red[(tid // 64) * BLOCK_TILE_SIZE_N + 16 * n + lane] = sums[n]
            gpu.barrier()
            if tid < contiguous_n:
                gate = red[tid]
                up = red[tid + contiguous_n]
                for w in range_constexpr(1, 4):
                    gate = gate + red[w * BLOCK_TILE_SIZE_N + tid]
                    up = up + red[w * BLOCK_TILE_SIZE_N + contiguous_n + tid]
                scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
                col = blk_n * contiguous_n + tid
                gate = gate * scale[expert_id * N + col]
                up = up * scale[expert_id * N + N // 2 + col]
                tmp = rocdl.exp2(T.f32, _raw(gate * -1.4426950408889634))
                act = (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up
                out = fx.make_view(
                    fxh._as_ptr(p_output) + fx.Int64(route_idx) * (N // 2),
                    fx.make_layout(N // 2, 1),
                )
                out[col] = _f32_to_bf16(Vec.from_elements([act], fx.Float32))[0]

        _clear_output_row(p_clear_output, blk_n, e_idx, batch_idx, tid)

    @flyc.jit
    def launch_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_a8w4_gateup_splitk(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_w_delta,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_clear_output: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_a8w4_gateup_batch1(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_clear_output,
            p_w_scale,
            p_w_delta,
        ).launch(
            grid=(num_n_blocks, TOPK, batch),
            block=(256, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_gemv(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_clear_output: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_a8w4_gateup_gemv(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_clear_output,
            p_w_scale,
            p_w_delta,
        ).launch(
            grid=(num_n_blocks, TOPK, batch),
            block=(256, 1, 1),
            stream=stream,
        )

    return {"batch1": launch_batch1, "gemv": launch_gemv, "splitk": launch_splitk}[alg]


def _build_down(N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, alg, E):
    """Down (N = H rows, K = I): one wave per CTA, atomic BF16 add into the output."""
    assert alg in ("splitk", "batch1", "gemv"), alg
    assert K % TILE_K == 0 and N % BLOCK_TILE_SIZE_N == 0, (N, K)
    assert BLOCK_TILE_SIZE_N in (64, 128), BLOCK_TILE_SIZE_N

    @fx.struct
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]

    @fx.struct
    class GemvStorage:
        x_lds: fx.Array[fx.Float32, K, 16]

    TensorWithIndex, _read_sorted_index, gemm_splitk, gemv_splitk = _make_fp4_gemm_splitk(
        K, BLOCK_TILE_SIZE_M, TOPK
    )

    def _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale):
        arg_p_scale = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
        )
        scale_tile = fx.flat_divide(arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N))[
            None, blk_n
        ]
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        tiled_copy_scale = fx.make_tiled_copy(
            cp_atom_scale,
            fx.make_layout(((16, 4), 4), ((0, 4), 1)),
            fx.make_tile(16),
        )
        scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(scale_tile)
        scale_frag = fx.make_fragment_like(scale_frag_tensor)
        fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
        m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()
        for n in range_constexpr(n_reps):
            scale_vec = scale_frag[None, n].load()
            for m in range_constexpr(m_reps):
                c_vec = c_frag[None, m, n].load()
                c_frag[None, m, n].store(c_vec * scale_vec)

    def _cvt_f32_to_bf16(c_frag):
        c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
        c_frag_bf16.store(_f32_to_bf16(c_frag.load()))
        return c_frag_bf16

    def _weight_and_delta(p_weight, p_w_delta, expert_id, blk_n):
        weight = fx.make_view(
            _u32_ptr(p_weight, 8) + fx.Int64(expert_id) * (N * K // 8),
            fx.make_layout(((16, N // 16), (2, K // 16)), ((2, 2 * K), (1, 32))),
        )
        # Four N repeats per 64-row delta block.
        per_block = BLOCK_TILE_SIZE_N // 64
        slots = [blk_n * per_block + s for s in range(per_block)]
        rep_sel = [(n // 4, (n % 4) * 0x01010101) for n in range(BLOCK_TILE_SIZE_N // 16)]
        return weight, _delta_source(p_w_delta, expert_id, N, K, slots, rep_sel)

    @flyc.kernel
    def moe_a8w4_down_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input), fx.make_layout((M, TOPK, K), (TOPK * K, K, 1))
        )
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, fxh._as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            arg_p_weight, delta = _weight_and_delta(p_weight, p_w_delta, expert_id, blk_n)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            lds_view = fx.make_view(
                lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
            )
            for idx in range(tid, BLOCK_TILE_SIZE_M, 64):
                lds_view[idx] = sorted_ids_buf[idx]
            gpu.barrier()

            cp_atom_weight = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            arg_p_sorted_weights = fx.make_view(
                fx.recast_iter(
                    fx.Float32,
                    fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M,
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            sorted_weights_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_weights, max_size=False
            )
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds, fx.make_layout(((16, 4), 1), ((1, 0), 0)), fx.make_tile(16)
            )
            sorted_weights_tensor = tiled_copy_sortid_lds.get_slice(tid).partition_S(
                sorted_weights_buf
            )
            sorted_weight_frag = fx.make_fragment_like(
                sorted_weights_tensor, fx.Float32
            )
            fx.copy(cp_atom_weight, sorted_weights_tensor, sorted_weight_frag)

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                delta,
                lds,
                splitk_waves=1,
            )

            _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

            sorted_weight_frag_vec = sorted_weight_frag.load()
            for m in range_constexpr(BLOCK_TILE_SIZE_M // 16):
                w = sorted_weight_frag_vec[m]
                v = c_frag[None, m, None].load()
                v *= w
                c_frag[None, m, None].store(v)

            c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output), fx.make_layout((M, N), (N, 1))
            )
            cp_atom_w = fx.make_copy_atom(
                fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
            )
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                fx.make_tile(16, 16),
            )
            c_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            c_tensor = TensorWithIndex(
                arg_p_output,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                c_index_frag,
                c_tiled_g,
                tid,
                is_read_from_mem=False,
                TOPK=TOPK,
                is_atomic_write=True,
            )
            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_a8w4_down_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(route_idx) * K,
            fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)),
        )
        arg_p_topk_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weights))
        expert_id = arg_p_topk_ids[route_idx]
        topk_weight = arg_p_topk_weights[route_idx]
        # Routing gives pad rows expert -1, which sorting would drop.
        if expert_id >= 0:
            arg_p_weight, delta = _weight_and_delta(p_weight, p_w_delta, expert_id, blk_n)

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                delta,
                None,
                splitk_waves=1,
                a_with_index=False,
            )

            _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

            c_frag.store(c_frag.load() * topk_weight)

            c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output) + fx.Int64(batch_idx) * N,
                fx.make_layout((1, N), (N, 1)),
            )
            cp_atom_w = fx.make_copy_atom(
                fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
            )
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                fx.make_tile(16, 16),
            )
            c_tile = fx.flat_divide(
                arg_p_output, fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
            )[None, None, None, blk_n]
            c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
            c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            rep_m = fx.size(fx.get_shape(c_src)[1]).to_py_value()
            rep_n = fx.size(fx.get_shape(c_src)[2]).to_py_value()
            if tid % 16 == 0:
                for m in range_constexpr(rep_m):
                    for n in range_constexpr(rep_n):
                        reg_vec = c_src[None, m, n].load()
                        ptr_base = fx.get_iter(c_dst[None, m, n, 0])
                        fxh.atomic_add_bf16(ptr_base, reg_vec)

    @flyc.kernel
    def moe_a8w4_down_gemv(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
    ):
        # batch1's grid and weight reads with the products on the VALU (see moe_a8w4_gateup_gemv).
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(route_idx) * K,
            fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)),
        )
        x_row = fx.make_view(
            _u32_ptr(p_input, 16) + fx.Int64(route_idx) * (K // 2),
            fx.make_layout(K // 2, 1),
        )
        arg_p_topk_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weights))
        expert_id = arg_p_topk_ids[route_idx]
        topk_weight = arg_p_topk_weights[route_idx]
        # Routing gives pad rows expert -1, which sorting would drop.
        if expert_id >= 0:
            arg_p_weight, delta = _weight_and_delta(p_weight, p_w_delta, expert_id, blk_n)
            lds = fx.SharedAllocator().allocate(GemvStorage).peek()
            sums = gemv_splitk(
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                x_row,
                arg_p_weight,
                delta,
                fx.make_view(lds.x_lds.ptr, fx.make_layout(K, 1)),
                splitk_waves=1,
            )
            # Lane l takes tile row 64 * j + l (row 16 * n + l % 16 is in sums[n] of every lane), and the
            # even lanes add the BF16 pair of rows (l, l + 1).
            group = tid // 16
            scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
            for j in range_constexpr(BLOCK_TILE_SIZE_N // 64):
                value = sums[4 * j]
                for i in range_constexpr(1, 4):
                    value = (group == i).select(sums[4 * j + i], value)
                row = blk_n * BLOCK_TILE_SIZE_N + 64 * j + tid
                value = value * scale[expert_id * N + row] * topk_weight
                pair = _f32_to_bf16(
                    Vec.from_elements(
                        [value, value.shuffle_xor(fx.Int32(1), fx.Int32(64))], fx.Float32
                    )
                )
                if tid % 2 == 0:
                    fxh.atomic_add_bf16(
                        fxh._as_ptr(p_output) + (fx.Int64(batch_idx) * N + fx.Int64(row)), pair
                    )

    @flyc.jit
    def launch_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_a8w4_down_splitk(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_w_delta,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_a8w4_down_batch1(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_topk_weights,
            p_w_scale,
            p_w_delta,
        ).launch(
            grid=(num_n_blocks, TOPK, batch),
            block=(64, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_gemv(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_a8w4_down_gemv(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_topk_weights,
            p_w_scale,
            p_w_delta,
        ).launch(
            grid=(num_n_blocks, TOPK, batch),
            block=(64, 1, 1),
            stream=stream,
        )

    return {"batch1": launch_batch1, "gemv": launch_gemv, "splitk": launch_splitk}[alg]


@functools.cache
def _compile_cached(device_cache_key, stage, N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, alg, E):
    del device_cache_key
    build = _build_gateup if stage == "gateup" else _build_down
    return build(N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, alg, E)


def compile_decode_gemm(*, stage, N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, alg, E=None):
    """Launcher of one W4A16 decode stage, cached per device (like ``moe_gemm_2stage.gemm``).

    split-K: ``(input, weight, output, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    w_factor, w_delta, M, task_num, stream)``; batch1 / gemv Gate/Up: ``(input, weight, mid, topk_ids,
    output_to_clear, w_factor, w_delta, batch, stream)``; batch1 / gemv Down: ``(mid, weight, output,
    topk_ids, topk_weights, w_factor, w_delta, batch, stream)``.
    """
    assert stage in ("gateup", "down"), stage
    assert BLOCK_TILE_SIZE_M == 16, "the decode kernels use one 16-row MFMA tile"
    return _compile_cached(get_device_cache_key(), stage, N, K, TOPK, BLOCK_TILE_SIZE_M,
                           BLOCK_TILE_SIZE_N, alg, E)
