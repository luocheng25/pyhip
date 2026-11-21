# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# pyhip: derived from ROCm/aiter@df3e0b2f aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/ (the M0
# prefill kernels, unchanged in pyhip's moe_gemm_2stage/):
#   gemm1.py      compile_moe_gemm1(alg="prefill_1x4"): _make_gateup_weight_view, _make_1x4_tiled_mma,
#                 gemm_1x4, _apply_1x4_fp8_dequant, _gateup_pair_bf16, moe_2stage_gateup_prefill_1x4,
#                 launch_prefill_1x4                                         -> _build_gateup
#   gemm2_1x4.py  _build_moe_gemm2_1x4_n256 (down_path="1x4_64x256"); its module-level helpers are
#                 imported unchanged                                         -> _build_down
# Changes: the weights are mxfp4 in the MXFP4Weights format (weights.py) instead of FP8. B reads 8 bytes
# (16 K) per lane and (M repeat, 64-K block) instead of 16, plus one w1_pdelta / w2_pdelta dword per
# lane, M repeat and K tile; before the MFMAs the fd1 nibbles are decoded with decode.py's v_perm table
# (PR #64 fd2) into the FP8 bytes the FP8 path loads, so the activation path, the FP8 MFMAs and the
# epilogues are unchanged. p_w_scale is the per-channel factor and the kernels take p_w_delta after it.
# Only per-token FP8 activations (ptpc) and SiLU are kept. The Down always uses BLOCK_K = 64 (upstream
# picks 128 when K % 128 == 0), so upstream's delayed 4-wave store (BLOCK_K = 128, nBK = 2) and the
# task table are dropped; its weights are carried across the N loop as fp4 dwords plus the delta dword,
# PREFETCH K cores ahead (upstream: one), and the next N tile's PTPC scales are loaded at K core 0.
# HALF_M (both stages): when the first row of a sorted block's second half is padding, the block runs
# the same tile code with half the rows (the sorting pads each expert's last block at its end).

"""A8W4 prefill kernels (1x4) for mxfp4 MoE weights: per-token FP8 activations, FP8 MFMA."""

import functools
from functools import partial

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..moe_gemm_2stage import common as fxh
from ..moe_gemm_2stage.common import (
    FlyObjCache,
    _f32_to_bf16,
    get_device_cache_key,
    get_down_device_config,
    make_gemm_helpers,
)
from ..moe_gemm_2stage.gemm2_1x4 import (
    _act_scale_tensor,
    _apply_token_scales,
    _gather_sorted_rows,
    _set_priority,
    _store_cshuffle_row_pair,
    _store_scaled_bf16,
    _write_cshuffle_vector,
)
from .decode import _LUT_HI, _LUT_LO, _decode_fp4x8, _perm, _u32_ptr

GATEUP_TILE_K = (128, 256)
DELTA_K = 256  # K covered by one w1_pdelta dword (weights.py)
DOWN_BLOCK = (64, 256)  # Down M x N tile (upstream 1x4_64x256)


def _b_copy(tid):
    """The MMA's A mapping (lane % 16 + 16 * wave: row of a 64-row block, lane // 16: 16 K at
    16 * (lane // 16) of a 64-K block) over a dword view with 8 fd1 nibbles per dword."""
    return fx.make_tiled_copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32),
        fx.make_layout(((16, 4, 4), 2), ((1, 2 * 64, 16), 64)),
        fx.make_tile(fx.make_layout(64, 1), fx.make_layout(8, 1)),
    ).get_slice(tid)


def _decode_rep(raw, dword, byte, frag, slices):
    """16 fd1 nibbles of one (M repeat, 64-K block) -> the FP8 bytes of the FP8 path: broadcast the
    delta byte, shift the |E2M1| -> FP8 table by it, decode k0..7 / k8..15 into ``frag[slices[k]]``."""
    d = _perm(dword, dword, byte * 0x01010101)
    lo = (fx.Uint32(_LUT_LO) - d) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(_LUT_HI) - d
    words = raw.load()
    for k in range_constexpr(2):
        even, odd = _decode_fp4x8(fx.Uint32(words[k]), lo, hi)
        frag[slices[k]].store(Vec.from_elements([even, odd], fx.Uint32).bitcast(fx.Float8E4M3FNUZ))


def _build_gateup(N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, TILE_K, E, HALF_M=False, PREFETCH=1):
    """Gate/Up (N = 2I rows: gate then up, K = H) over sorted rows: 4 waves split N, FP8 MFMA.

    ``HALF_M``: a sorted block whose second half is padding computes only its first half (a
    ``BLOCK_TILE_SIZE_M // 2`` tile); the sorting pads each expert's last block at its end.
    ``PREFETCH``: K tiles of fp4 weights in flight for 32-row tiles (1 or 2; larger tiles, with more
    MFMAs per weight, keep 1; A always goes one tile ahead via LDS).
    """
    assert PREFETCH in (1, 2), PREFETCH
    assert not HALF_M or BLOCK_TILE_SIZE_M % 64 == 0, (
        f"HALF_M needs a half tile that is a multiple of 32, got BM={BLOCK_TILE_SIZE_M}"
    )
    assert 128 <= BLOCK_TILE_SIZE_N <= 256 and BLOCK_TILE_SIZE_N % 128 == 0, (
        "For prefill_1x4 alg, BLOCK_TILE_SIZE_N must be in [128, 256] and a multiple of 128 "
        "(each wave owns contiguous_n//4 = BN//8 output channels, a multiple of 16)"
    )
    assert N % BLOCK_TILE_SIZE_N == 0, (N, BLOCK_TILE_SIZE_N)
    assert TILE_K in GATEUP_TILE_K, f"prefill_1x4 fp8 TILE_K must be 128 or 256, got {TILE_K}"
    assert K % DELTA_K == 0, f"A8W4 prefill K={K} must be a multiple of {DELTA_K} (w1_pdelta)"
    assert (K // TILE_K) % 2 == 0, (
        f"prefill_1x4 needs an even K-tile count for the 2-stage pipeline, got "
        f"K//TILE_K={K // TILE_K} (K={K}, TILE_K={TILE_K})"
    )
    _val_per_thr = 16
    _thrs_m_aload = 256 // (TILE_K // _val_per_thr)
    assert 32 <= BLOCK_TILE_SIZE_M <= 256 and BLOCK_TILE_SIZE_M % 32 == 0, (
        f"For prefill_1x4 alg, BLOCK_TILE_SIZE_M must be a multiple of 32 in [32, 256] "
        f"(A g2r load needs BM >= {_thrs_m_aload}; the CShuffle 32-token read tile forces a "
        f"32-multiple, so 32 is the effective minimum). got BM={BLOCK_TILE_SIZE_M}"
    )
    a_lds_size = BLOCK_TILE_SIZE_M * TILE_K
    lds_elem = fx.Float8E4M3FNUZ

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[lds_elem, a_lds_size, 16]
        a_pong: fx.Array[lds_elem, a_lds_size, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers

    _gemm_lds_bytes = 2 * a_lds_size
    _cshuffle_bytes = BLOCK_TILE_SIZE_M * (BLOCK_TILE_SIZE_N // 2) * 2
    assert _cshuffle_bytes <= _gemm_lds_bytes, (
        f"CShuffle needs {_cshuffle_bytes} B of LDS but GemmBuffers only allocates "
        f"{_gemm_lds_bytes} B (BM={BLOCK_TILE_SIZE_M}, BN={BLOCK_TILE_SIZE_N})"
    )
    assert _gemm_lds_bytes <= 64 * 1024, (
        f"prefill_1x4 A ping-pong needs {_gemm_lds_bytes} B of LDS but gfx942 allows "
        f"{64 * 1024} B (BM={BLOCK_TILE_SIZE_M}, TILE_K={TILE_K}, dtype=fp8)"
    )

    weight_dtype = fx.Float8E4M3FNUZ

    TensorWithIndex, _read_sorted_index, _ = make_gemm_helpers(
        K, weight_dtype, BLOCK_TILE_SIZE_M, TOPK
    )

    def _gateup_pair_bf16(gate_frag, up_frag):
        log2_exp1 = -1.4426950408889634
        out_bf16 = fx.make_fragment_like(gate_frag, dtype=fx.BFloat16)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            for n in range_constexpr(n_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                acc = []
                for j in range_constexpr(gate.numel):
                    g = gate[j]
                    u = up[j]
                    tmp = rocdl.exp2(T.f32, _raw(g * log2_exp1))
                    acc.append((g * rocdl.rcp(T.f32, 1.0 + tmp)) * u)
                acc = Vec.from_elements(acc, fx.Float32)
                out_bf16[None, m, n].store(_f32_to_bf16(acc))
        return out_bf16

    def _make_gateup_weight_view(p_weight, expert_id, contiguous_n):
        # Dword view (N, K / 8) of the fd2 bytes [N/16, K/16, 16, 8]: 8 nibbles per dword.
        group_layout_silu = fx.make_layout(
            ((contiguous_n, 2, N // (contiguous_n * 2)), K // 8),
            ((1, N // 2, contiguous_n), N),
        )
        return fx.make_view(
            _u32_ptr(p_weight, 8) + fx.Int64(expert_id) * (N * K // 8),
            fx.composition(
                fx.make_layout(((16, N // 16), (2, K // 16)), ((2, 2 * K), (1, 32))),
                group_layout_silu,
            ),
        )

    def _make_gateup_delta(p_w_delta, expert_id, blk_n, contiguous_n):
        """w1_pdelta reads: dword (n16, kb, h, l) holds groups 8 * kb + 2 * i + h of row 16 * n16 + l.
        Returns the buffer and the K-tile-0 dword of each (gate / up half, M repeat) of this lane."""
        dwords = N * K // 128
        view = fx.make_view(
            _u32_ptr(p_w_delta, 4) + fx.Int64(expert_id) * dwords, fx.make_layout(dwords, 1)
        )
        tid = gpu.thread_idx.x
        lane = tid % 64
        # Lane l reads 16 K at 16 * (l // 16) of each 64-K block: groups of parity (l // 32).
        rows = [
            h * (N // 32) + blk_n * (contiguous_n // 16) + 4 * m + tid // 64
            for h in range(2)
            for m in range(contiguous_n // 64)
        ]
        bases = [(n16 * (K // DELTA_K) * 2 + lane // 32) * 16 + lane % 16 for n16 in rows]
        return fx.rocdl.make_buffer_tensor(view, max_size=False), bases

    def _make_1x4_tiled_mma():
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_1x4(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,
        arg_p_input: fx.Tensor,
        arg_p_weight: fx.Tensor,
        delta,
        lds,
        B_AHEAD=1,
    ):
        tid = gpu.thread_idx.x
        contiguous_n = TILE_N // 2

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        mma_atom, tiled_mma = _make_1x4_tiled_mma()

        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
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
        cp_atom_sortid_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_a = fx.make_tiled_copy(
            cp_atom_sortid_a,
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = _read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, index_size=TILE_M
        )
        a_idx = TensorWithIndex(
            a_tensor,
            TILE_M,
            TILE_K,
            a_index_frag,
            a_mem_cp_g2r,
            tid,
        )
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

        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)
        a_lds_r = [
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(a_ping),
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(a_pong),
        ]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = (
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)
        )

        # fp4 B: raw dwords double-buffered like upstream's FP8 fragments, decoded into one FP8
        # fragment per half right before the MFMAs.
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K // 8))[
            None, None, blk_n * 2 + 0, None
        ]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K // 8))[
            None, None, blk_n * 2 + 1, None
        ]
        b_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        b_g2r = _b_copy(tid)
        bl_g2r = b_g2r.partition_S(bl_tile)
        br_g2r = b_g2r.partition_S(br_tile)
        bl_raw_st = [
            fx.make_fragment_like(bl_g2r[None, None, None, 0], fx.Uint32),
            fx.make_fragment_like(bl_g2r[None, None, None, 0], fx.Uint32),
        ]
        br_raw_st = [
            fx.make_fragment_like(br_g2r[None, None, None, 0], fx.Uint32),
            fx.make_fragment_like(br_g2r[None, None, None, 0], fx.Uint32),
        ]
        b_fake = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_layout((contiguous_n, TILE_K), (TILE_K, 1)),
        )
        bl_frag = tiled_mma.make_fragment_A(b_fake)
        br_frag = tiled_mma.make_fragment_A(b_fake)
        delta_buf, delta_bases = delta
        n_delta = len(delta_bases)
        d_raw_st = [
            fx.make_rmem_tensor(n_delta, fx.Uint32),
            fx.make_rmem_tensor(n_delta, fx.Uint32),
        ]

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[
            None, None, 0, 0
        ]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        num_tiles = K // TILE_K

        k_per_mma = 32
        _m_reps = fx.size(fx.get_shape(c_gate)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_gate)[2]).to_py_value()
        mfma_per_gemm = _m_reps * _n_reps * (TILE_K // k_per_mma)
        mem_a_cnt = a_cp_frag.load().numel * weight_dtype.width // 8 // 16
        mem_b_cnt = bl_raw_st[0].load().numel * 4 // 8
        k_iters = TILE_K // (2 * k_per_mma)
        lds_a_cnt = a_frag.load().numel * weight_dtype.width // 8 // 16

        def hot_loop_scheduler():
            mfma_cnt = 2 * mfma_per_gemm
            n_vmem = mem_a_cnt + 2 * mem_b_cnt + n_delta
            n_dswr = mem_a_cnt
            n_dsrd = lds_a_cnt
            used = 0
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(n_vmem):
                rocdl.sched_dsrd(1)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(2)
                used += 4
            for _ in range_constexpr(n_dsrd - 2 * n_vmem - 2):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                used += 1
            if const_expr(mfma_cnt - n_dswr * 2 - used > 0):
                rocdl.sched_mfma(mfma_cnt - n_dswr * 2 - used)
            for _ in range_constexpr(n_dswr):
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(2)
                used += 2
            if const_expr(mfma_cnt - used > 0):
                rocdl.sched_mfma(mfma_cnt - used)

        def prefetch_b(k_next, write_i):
            fx.copy(b_cp_atom_r, bl_g2r[None, None, None, k_next], bl_raw_st[write_i])
            fx.copy(b_cp_atom_r, br_g2r[None, None, None, k_next], br_raw_st[write_i])
            k_block = k_next * TILE_K // DELTA_K
            for s in range_constexpr(n_delta):
                d_raw_st[write_i][s] = delta_buf[delta_bases[s] + k_block * 32]

        def decode_b(read_i):
            # Buffer read_i holds the K tiles of parity read_i; delta byte i is 64-K block i of the
            # 256-K block, i.e. 2 * read_i + ki when TILE_K = 128.
            dwords = d_raw_st[read_i].load()
            halves = ((bl_raw_st[read_i], bl_frag), (br_raw_st[read_i], br_frag))
            for h in range_constexpr(2):
                raw, frag = halves[h]
                for m in range_constexpr(_m_reps):
                    dword = fx.Uint32(dwords[h * _m_reps + m])
                    for ki in range_constexpr(k_iters):
                        byte = ki if const_expr(TILE_K == DELTA_K) else 2 * read_i + ki
                        _decode_rep(raw[None, m, ki], dword, byte, frag,
                                    [(None, m, (k, ki)) for k in range(2)])

        def pipeline_stage(read_i, k_next, do_prefetch, b_next=None):
            # B_AHEAD == 2: B of K tile k_next + 1 (parity read_i) goes into buffer read_i right
            # after it is decoded, two K tiles ahead of its use.
            write_i = read_i ^ 1
            if const_expr(do_prefetch):
                a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)
                if const_expr(B_AHEAD == 1):
                    prefetch_b(k_next, write_i)
            decode_b(read_i)
            if const_expr(B_AHEAD == 2 and b_next is not None):
                prefetch_b(b_next, read_i)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
                            fx.mma_atom_call(
                                mma_atom,
                                c_gate[None, m, n],
                                bl_frag[None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_gate[None, m, n],
                            )
                            fx.mma_atom_call(
                                mma_atom,
                                c_up[None, m, n],
                                br_frag[None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_up[None, m, n],
                            )
            if const_expr(do_prefetch):
                fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
                hot_loop_scheduler()
            rocdl.sched_barrier(0)
            gpu.barrier()

        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag)
        prefetch_b(fx.Int32(0), 0)
        if const_expr(B_AHEAD == 2):
            prefetch_b(fx.Int32(1), 1)
        rocdl.s_waitcnt(vmcnt=0)
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        gpu.barrier()

        acc_init = [c_gate.load(), c_up.load()]
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            pipeline_stage(0, kb + 1, True, kb + 2)
            pipeline_stage(1, kb + 2, True, kb + 3)
            results = yield [c_gate.load(), c_up.load()]
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        pipeline_stage(0, kb + 1, True)
        pipeline_stage(1, fx.Int32(0), False)
        return c_gate, c_up

    def _apply_1x4_fp8_dequant(
        c_gate_frag,
        c_up_frag,
        tid,
        expert_id,
        blk_n,
        contiguous_n,
        asc_idx,
        M,
        p_w_scale,
        p_a_scale,
    ):
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
            fx.make_view(
                fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                fx.make_layout(M, 1),
            ),
            max_size=False,
        )
        a_sc_n = [
            a_scale_tensor[asc_idx[0, n] & 0xFFFFFF] for n in range_constexpr(n_reps)
        ]
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

    def _gateup_tile(tile_m, tid, blk_n, expert_id, M, arg_p_input, arg_p_weight, delta, lds,
                     p_output, p_w_scale, p_a_scale):
        """Gate/Up, dequant, SiLU and the BF16 store of the block's first ``tile_m`` sorted rows."""
        contiguous_n = BLOCK_TILE_SIZE_N // 2
        # Capture scatter indices before a_ping reuses sorted_lds.
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
        buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        c_rw_copy = fx.make_tiled_copy(
            buf_atom_w128,
            fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
            fx.make_tile(32, 64),
        )
        c_index_copy = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
            fx.make_tile(32),
        )
        c_out_index_frag = _read_sorted_index(c_index_copy, tid, lds.sorted_lds, index_size=tile_m)
        c_out = TensorWithIndex(
            out_tensor,
            tile_m,
            contiguous_n,
            c_out_index_frag,
            c_rw_copy,
            tid,
            is_read_from_mem=False,
            TOPK=TOPK,
        )

        asc_index_copy = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
            fx.make_tile(16),
        )
        cp_atom_idx = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        asc_lds = fx.make_view(
            lds.sorted_lds.ptr, fx.make_layout(tile_m, 1)
        )
        asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
        asc_idx = fx.make_fragment_like(asc_thr)
        fx.copy(cp_atom_idx, asc_thr, asc_idx)

        c_gate_frag, c_up_frag = gemm_1x4(
            tile_m,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            delta,
            lds,
            PREFETCH if tile_m <= 32 else 1,
        )

        _apply_1x4_fp8_dequant(
            c_gate_frag,
            c_up_frag,
            tid,
            expert_id,
            blk_n,
            contiguous_n,
            asc_idx,
            M,
            p_w_scale,
            p_a_scale,
        )

        c_out_bf16 = _gateup_pair_bf16(c_gate_frag, c_up_frag)

        _, _tiled_mma = _make_1x4_tiled_mma()
        cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        cshuf_ptr = fx.recast_iter(fx.BFloat16, lds.gemm.a_ping.ptr)
        # Aliased transpose views require the same BF16 swizzle.
        swz_c = fx.SwizzleType.get(3, 3, 3)
        lds_c_store = fx.make_view(
            cshuf_ptr,
            fx.make_composed_layout(
                fx.static(swz_c),
                fx.make_ordered_layout(
                    (contiguous_n, tile_m), order=(0, 1)
                ),
            ),
        )
        lds_c = fx.make_view(
            cshuf_ptr,
            fx.make_composed_layout(
                fx.static(swz_c),
                fx.make_ordered_layout(
                    (tile_m, contiguous_n), order=(1, 0)
                ),
            ),
        )

        gpu.barrier()  # Finish gemm_1x4's LDS reads before reusing GemmBuffers.
        store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
        fx.copy(
            cshuf_atom_w,
            store_c.retile(c_out_bf16),
            store_c.partition_D(lds_c_store),
        )
        gpu.barrier()
        rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
        fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
        c_out.copy(buf_atom_w128, blk_n, rd)

    @flyc.kernel
    def moe_a8w4_gateup_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        in_ptr = fx.recast_iter(weight_dtype, fxh._as_ptr(p_input))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
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

            contiguous_n = BLOCK_TILE_SIZE_N // 2
            arg_p_weight = _make_gateup_weight_view(p_weight, expert_id, contiguous_n)
            delta = _make_gateup_delta(p_w_delta, expert_id, blk_n, contiguous_n)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            tile_args = (tid, blk_n, expert_id, M, arg_p_input, arg_p_weight, delta, lds, p_output,
                         p_w_scale, p_a_scale)
            if const_expr(HALF_M):
                ids_lds = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1))
                second = fx.Int32(rocdl.readfirstlane(
                    ir.IntegerType.get_signless(32), _raw(ids_lds[BLOCK_TILE_SIZE_M // 2])))
                if (second & 0xFFFFFF) < M:
                    _gateup_tile(BLOCK_TILE_SIZE_M, *tile_args)
                else:
                    _gateup_tile(BLOCK_TILE_SIZE_M // 2, *tile_args)
            else:
                _gateup_tile(BLOCK_TILE_SIZE_M, *tile_args)

    @flyc.jit
    def launch_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_a8w4_gateup_prefill_1x4(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_w_delta,
            p_a_scale,
            M,
        ).launch(
            grid=(num_n_blocks, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_prefill_1x4


def _build_down(N, K, TOPK, padding, PREFETCH=1, HALF_M=False):
    """Down (N = H rows, K = I) M64xN256 over sorted rows: each sorted row to its own output row.

    ``PREFETCH``: K cores (BLOCK_K = 64) of weights in flight ahead of the one being computed.
    ``HALF_M``: a task whose second half is padding computes (and stores) only its first 32 rows.
    """
    assert PREFETCH >= 1, PREFETCH
    assert padding in (0, 32, 64, 128)
    assert K % 64 == 0 and N % 256 == 0, (N, K)
    _, generic_xcd_count = get_down_device_config()
    block_m_per_4wave_group = 64
    activation_bytes = block_m_per_4wave_group * K
    scale_bytes = 256 * 4
    cshuffle_bytes = 4 * 16 * 64 * (fx.BFloat16.width // 8)
    assert activation_bytes + scale_bytes + cshuffle_bytes <= 64 * 1024, (
        "1x4_64x256 exceeds gfx942 LDS capacity; "
        f"activation={activation_bytes}B, scale={scale_bytes}B, "
        f"cshuffle={cshuffle_bytes}B"
    )
    output_row_stride = N + padding // (fx.BFloat16.width // 8)

    weight_dtype = fx.Float8E4M3FNUZ
    down_ops = FlyObjCache(cache_ir=False)

    @flyc.jit
    def _map_down_task(valid_rows: fx.Int32, task_rows: fx.Constexpr[int]):
        """Interleave down tasks across XCDs."""
        workgroup_idx = fx.Int32(fx.gpu.block_idx.y)
        valid_rows_u32 = fx.Uint32(valid_rows)
        valid_tasks = valid_rows_u32 // task_rows
        valid_tasks += fx.Uint32(valid_rows_u32 % task_rows != 0)

        swizzle_chunk = valid_tasks // generic_xcd_count
        swizzle_limit = swizzle_chunk * generic_xcd_count
        swizzled_e_idx = (
            workgroup_idx % generic_xcd_count
        ) * swizzle_chunk + workgroup_idx // generic_xcd_count
        return fx.Int32(
            arith.select(workgroup_idx < swizzle_limit, swizzled_e_idx, workgroup_idx)
        )

    BLOCK_N, BLOCK_K = DOWN_BLOCK[1], 64
    WAVE_N = BLOCK_N // 4

    @flyc.jit
    def _down_tile(BLOCK_M, M, expert_id, arg_p_input, arg_p_weight, ldsA0, scale_lds, cshuffle_lds,
                   output_store_rsrc, arg_p_sorted_ids, arg_p_sorted_weights, p_w_scale, p_w_delta,
                   p_a_scale):
        """Down of the task's first ``BLOCK_M`` sorted rows; the activations of all
        ``block_m_per_4wave_group`` rows are already in LDS (``ldsA0``)."""
        # (BLOCK_N, BLOCK_K / 8, num_blocks_N, num_blocks_K)
        weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K // 8))
        ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))
        nBN = fxh.div_up(N, BLOCK_N)
        nBK = fxh.div_up(K, BLOCK_K)
        mm = down_ops.create_thr_mma(weight_dtype, (4, 1, 1))

        c_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
        )
        fragC = mm.make_fragment_C(c_fake_tensor)
        fragC_bf16 = fx.make_fragment_like(fragC, fx.BFloat16)
        frag_act = mm.make_fragment_B(ldsA[None, None, 0, 0])
        frag_weight = mm.make_fragment_A(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((BLOCK_N, BLOCK_K), (BLOCK_K, 1)),
            )
        )
        # fp4 weights: (2 dwords, 4 M repeats) per lane and K core, and the w2_pdelta dword
        # (4 * block_n + wave, group, lane % 16) whose byte m is M repeat m's delta.
        weight_thr = _b_copy(fx.thread_idx.x).partition_S(weight)
        b_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        groups = K // 32
        delta_dwords = N * K // 128
        delta_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                _u32_ptr(p_w_delta, 4) + fx.Int64(expert_id) * delta_dwords,
                fx.make_layout(delta_dwords, 1),
            ),
            max_size=False,
        )
        lane = fx.Int32(fx.thread_idx.x % 64)
        delta_base = (fx.Int32(fx.thread_idx.x // 64) * groups + lane // 32) * 16 + lane % 16

        # 3. Prepare weight, activation, routing scales, and output CShuffle resources.
        scale_global = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
        )
        scale_global_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(
                fx.get_iter(scale_global),
                num_records_bytes=N * (fx.Float32.width // 8),
            )
        )
        scale_lds_logical = fx.make_view(
            fx.get_iter(scale_lds),
            fx.make_layout(((4, 2, 2, 4, 4), 1), ((1, 16, 32, 64, 4), 0)),
        )
        scale_lds_copy_atom = down_ops.get_universal_copy_atom(fx.Float32, 128)

        def issue_scale_block_global(block_n):
            lane_id = fx.Int32(fx.thread_idx.x % 64)
            wave_id = fx.Int32(fx.thread_idx.x // 64)
            scale_local_offset = wave_id * WAVE_N + lane_id * 4
            scale_offset = fx.Int32(block_n) * BLOCK_N + scale_local_offset
            # The original buffer_load used element offsets; raw ROCDL uses bytes.
            # Invalid lanes still use out-of-bounds zero filling.
            scale_byte_offset = (lane_id < WAVE_N // 4).select(
                scale_offset * (fx.Float32.width // 8), fx.Int32(0x7FFFFFFF)
            )
            scale_vec = Vec(
                fx.rocdl.RawPtrBufferLoadOp(
                    ir.VectorType.get([4], fx.Float32.ir_type),
                    scale_global_rsrc,
                    scale_byte_offset.ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
                ).result
            )
            return scale_vec

        def commit_scale_block_lds(scale_vec, dst=None):
            lane_id = fx.Int32(fx.thread_idx.x % 64)
            wave_id = fx.Int32(fx.thread_idx.x // 64)
            if lane_id < WAVE_N // 4:
                scale_local_offset = wave_id * WAVE_N + lane_id * 4
                scale_dst = fx.make_view(
                    fx.get_iter(scale_lds) + scale_local_offset,
                    fx.make_layout(4, 1),
                )
                scale_frag = fx.make_fragment_like(scale_dst)
                scale_frag.store(scale_vec)
                fx.copy(scale_lds_copy_atom, scale_frag, scale_dst)
            return down_ops.load_tiled_mma_fragC(
                mm, scale_lds_logical, dst=dst, copy_atom_bits=128
            )

        arg_a_scale = _act_scale_tensor(p_a_scale, "ptpc", M, TOPK)

        sorted_weights = fx.make_view(
            fx.get_iter(arg_p_sorted_weights),
            fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
        )
        # load rows/token weights using C layout
        frag_sorted_weight = down_ops.load_tiled_mma_fragC(
            mm, sorted_weights, copy_atom_bits=32
        )
        frag_sorted_weight = _apply_token_scales(
            down_ops,
            mm,
            arg_a_scale,
            arg_p_sorted_ids,
            frag_sorted_weight,
            BLOCK_N,
            BLOCK_M,
        )
        cshuffle_copy_atom = down_ops.get_universal_copy_atom(fx.BFloat16, 128)
        write_cshuffle_vector = partial(
            _write_cshuffle_vector, cshuffle_lds, cshuffle_copy_atom
        )
        store_cshuffle_row_pair = partial(
            _store_cshuffle_row_pair,
            cshuffle_lds,
            cshuffle_copy_atom,
            output_store_rsrc,
            block_n_size=BLOCK_N,
            wave_n=WAVE_N,
            row_stride=output_row_stride,
        )

        def store_cshuffle_n256(output, block_n):
            lane_id = fx.Int32(fx.thread_idx.x % 64)
            wave_id = fx.Int32(fx.thread_idx.x // 64)
            lane_group = lane_id // 16
            lane_row = lane_id % 16
            wave_lds_base = wave_id * (16 * 64)

            # All 64 lanes produce one M16 pair, halving the DS writes used
            # by the previous two masked M8 producers.
            for row_pair in range_constexpr(BLOCK_M // 16):
                row_in_8 = lane_row % 8
                row_half = lane_row // 8
                for channel_piece in range_constexpr(2):
                    channels_lo = Vec(
                        output[None, 2 * channel_piece, row_pair].load()
                    )
                    channels_hi = Vec(
                        output[None, 2 * channel_piece + 1, row_pair].load()
                    )
                    write_cshuffle_vector(
                        wave_lds_base,
                        lane_group,
                        row_in_8,
                        row_half,
                        channel_piece,
                        channels_lo.shuffle(channels_hi, list(range(8))),
                    )
                store_cshuffle_row_pair(
                    wave_lds_base, lane_id, wave_id, row_pair, block_n
                )

        pf = min(PREFETCH, nBK)

        def weight_slot():
            return (fx.make_fragment_like(weight_thr[None, None, None, 0, 0], fx.Uint32),
                    fx.make_rmem_tensor(1, fx.Uint32))

        def load_weight(block_n, k_core, slot):
            raw, delta = slot
            fx.copy(b_cp_atom, weight_thr[None, None, None, block_n, k_core], raw)
            delta[0] = delta_buf[
                delta_base + fx.Int32(block_n) * (4 * groups * 16) + k_core * 2 * 16
            ]

        def decode_weight(slot):
            raw, delta = slot
            dword = fx.Uint32(delta.load()[0])
            for m in range_constexpr(4):
                _decode_rep(raw[None, m, 0], dword, m, frag_weight,
                            [(None, m, k) for k in range(2)])

        # Prologue: the first pf K cores of N tile 0 and its PTPC scales.
        _set_priority(0)
        carried = [weight_slot() for _ in range(pf)]
        for k_core in range_constexpr(pf):
            load_weight(fx.Int32(0), k_core, carried[k_core])
        frag_pc_scale = commit_scale_block_lds(issue_scale_block_global(0))
        next_frag_pc_scale = fx.make_fragment_like(frag_pc_scale)

        def run_n_tile(block_n, slots):
            """K cores of one N tile; ``slots[k]`` holds K core k for k < pf. The weights are
            loaded pf K cores ahead (wrapping into the next N tile, whose first pf K cores are
            returned) and the next tile's PTPC scales at K core 0."""
            ahead = [None] * pf
            next_scale_vec = issue_scale_block_global(block_n + 1)
            fx.rocdl.sched_barrier(0)
            for k_core in range_constexpr(nBK):
                down_ops.load_tiled_mma_fragB(
                    mm, ldsA, [None, None, 0, k_core], frag_act
                )
                target, slot = k_core + pf, weight_slot()
                if const_expr(target < nBK):
                    load_weight(block_n, target, slot)
                    slots.append(slot)
                else:
                    load_weight(block_n + 1, target - nBK, slot)
                    ahead[target - nBK] = slot
                decode_weight(slots[k_core])
                _set_priority(3)
                fx.gemm(mm, fragC, frag_weight, frag_act, fragC)
                _set_priority(0)
            commit_scale_block_lds(next_scale_vec, next_frag_pc_scale)
            return ahead

        # The first pf K cores of the next N tile are carried across the loop.
        loop_state = ([slot[0].load() for slot in carried] + [slot[1].load() for slot in carried]
                      + [frag_pc_scale.load()])
        for block_n, state in range(
            fx.Int64(0), fx.Int64(nBN), fx.Int64(1), init=loop_state
        ):
            for k_core in range_constexpr(pf):
                carried[k_core][0].store(state[k_core])
                carried[k_core][1].store(state[pf + k_core])
            frag_pc_scale.store(state[2 * pf])

            fragC.fill(0)
            ahead = run_n_tile(block_n, list(carried))

            # Apply channel scale, routing weight and BF16 conversion.
            for fc, fpc in fxh.all_elements(fragC, frag_pc_scale):
                fc.store(fc.load() * fpc.load())
            _store_scaled_bf16(fragC, frag_sorted_weight, fragC_bf16)
            store_cshuffle_n256(fragC_bf16, block_n)

            next_state = ([slot[0].load() for slot in ahead] + [slot[1].load() for slot in ahead]
                          + [next_frag_pc_scale.load()])
            results = yield next_state

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_a8w4_down_prefill_1x4_64x256(
        p_input: fx.Pointer,  # fp8 [M, TOPK, K]            K = HIDDEN_STATES//TP
        p_weight: fx.Pointer,  # fd2 mxfp4 [E, N/16, K/16, 16, 8]   N = HIDDEN_STATES
        p_output: fx.Pointer,  # bf16 [num_tokens_sorted, N + padding]
        p_sorted_ids: fx.Pointer,  # int32 [num_tokens_sorted]
        p_sorted_weights: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_expert_ids: fx.Pointer,  # int32 [num_blocks] num_tokens_sorted <= num_blocks * BLOCK_TILE_SIZE_M
        p_num_valid_ids: fx.Pointer,  # int32 [2]  value: (sorting valid rows incl. expert padding, M)
        p_w_scale: fx.Pointer,  # f32 [E, N] per-channel factor 2^(c_max - 132)
        p_w_delta: fx.Pointer,  # w2_pdelta [E, N/64, K/32, 16, 4]
        p_a_scale: fx.Pointer,  # f32 [M, TOPK] input fp8 scale (per-token ptpc)
        M: fx.Int32,
    ):
        """M64xN256: each workgroup uses four waves to compute one M64 task."""
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        e_idx = _map_down_task(max_valid_id, block_m_per_4wave_group)
        e_offset = fx.Int64(e_idx)
        if e_idx * block_m_per_4wave_group < max_valid_id:
            # 1. Create input, output, sorted-metadata, and weight views for this expert task.
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16)
                + e_offset * (block_m_per_4wave_group * output_row_stride),
                (block_m_per_4wave_group, output_row_stride),
            )
            output_store_rsrc = fx.rocdl.get_buffer_rsrc(
                fx.rocdl.make_buffer_ptr(
                    fx.get_iter(arg_p_output),
                    num_records_bytes=block_m_per_4wave_group * output_row_stride * 2,
                )
            )

            def sorted_rows(pointer, dtype):
                return fxh.view_as_torch_tensor(
                    fxh._as_ptr(pointer) + e_offset * block_m_per_4wave_group,
                    (block_m_per_4wave_group,),
                    dtype,
                )

            arg_p_sorted_ids = sorted_rows(p_sorted_ids, fx.Int32)
            arg_p_sorted_weights = sorted_rows(p_sorted_weights, fx.Float32)
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]

            # Dword view of the fd2 bytes (8 nibbles per dword) in upstream's N256 row order.
            arg_p_weight = fx.make_view(
                _u32_ptr(p_weight, 8) + fx.Int64(expert_id) * (N * K // 8),
                fx.make_layout(
                    (((4, 2, 2, 4, 4, N // 256)), (2, K // 16)),
                    ((2, 2 * K, 4 * K, 8 * K, 8, 32 * K), (1, 32)),
                ),
            )
            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
            BLOCK_M = block_m_per_4wave_group

            # mask,base,shift, swizzle always in unit of 128b,
            swz_base = ((128 // weight_dtype.width) - 1).bit_length()
            a_swz = fx.SwizzleType.get(3, swz_base, 4)

            shared_allocator = fx.SharedAllocator()
            activation_storage = shared_allocator.allocate(
                fx.Array[weight_dtype, BLOCK_M * K]
            )
            scale_storage = shared_allocator.allocate(fx.Array[fx.Float32, BLOCK_N, 16])
            scale_lds = scale_storage.peek().view(fx.make_layout(BLOCK_N, 1))
            cshuffle_storage = shared_allocator.allocate(
                fx.Array[fx.BFloat16, 4 * 16 * 64, 16]
            )
            cshuffle_lds = cshuffle_storage.peek().view(fx.make_layout(4 * 16 * 64, 1))
            ldsA0 = activation_storage.peek().view(
                fx.make_composed_layout(fx.static(a_swz), fxh.torch_layout(BLOCK_M, K))
            )
            # Sorting pads each expert's last block at its end: the second half is padding when
            # its first row is, and the task then computes a half-height tile.
            if const_expr(HALF_M):
                second_id = arg_p_sorted_ids[BLOCK_M // 2]
            # 2. Gather activations into LDS; the barrier starts the four-wave pipeline.
            arg_p_input = _gather_sorted_rows(
                down_ops, ldsA0, arg_p_input, arg_p_sorted_ids, M, TOPK, K
            )

            tile_args = (M, expert_id, arg_p_input, arg_p_weight, ldsA0, scale_lds, cshuffle_lds,
                         output_store_rsrc, arg_p_sorted_ids, arg_p_sorted_weights, p_w_scale,
                         p_w_delta, p_a_scale)
            if const_expr(HALF_M):
                second = fx.Int32(rocdl.readfirstlane(ir.IntegerType.get_signless(32), _raw(second_id)))
                if (second & 0xFFFFFF) < M:
                    _down_tile(BLOCK_M, *tile_args)
                else:
                    _down_tile(BLOCK_M // 2, *tile_args)
            else:
                _down_tile(BLOCK_M, *tile_args)
            _set_priority(0)

    @flyc.jit
    def launch_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        kernel = moe_a8w4_down_prefill_1x4_64x256(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_w_delta,
            p_a_scale,
            M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(256, 1, 1), stream=stream)

    launch_prefill_1x4.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_1x4


# ---------------------------------------------------------------- per-row FP8 quantization
# The prefill's activations (per token) and Gate/Up results (per (token, k)) are quantized per row with
# the arithmetic of pyhip.testing.moe_a8w4_ptpc.quantize_rows: scale = amax * fp32(1/240),
# q = sat(x / scale) rounded to nearest (the exact reciprocal; Aiter's HIP kernel multiplies by an
# approximate one); an all-zero row has scale 0 and quantizes to 0. ``lanes`` threads share a row (the
# largest power of two <= 64 dividing cols / 8), each loading 16 bytes per step; the row maximum is a
# shuffle_xor reduction.

QUANT_FP8_MAX = 240.0
QUANT_VEC = 8  # BF16 per 16-byte load
QUANT_THREADS = 256


def _quant_lanes_per_row(cols):
    assert cols % QUANT_VEC == 0, cols
    lanes = 64
    while (cols // QUANT_VEC) % lanes:
        lanes //= 2
    return lanes


def _build_quant_rows(cols):
    lanes = _quant_lanes_per_row(cols)
    chunks = cols // (QUANT_VEC * lanes)
    rows_per_block = QUANT_THREADS // lanes

    @flyc.kernel(known_block_size=[QUANT_THREADS, 1, 1])
    def a8w4_quant_rows(p_x: fx.Pointer, p_q: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        row = fx.Int32(fx.block_idx.x) * rows_per_block + tid // lanes
        lane = tid % lanes
        elems = fx.Int64(rows) * cols
        x = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.BFloat16, fxh._as_ptr(p_x)), fx.make_layout(elems, 1)),
            max_size=False,
        )
        q = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_q)), fx.make_layout(elems, 1)),
            max_size=False,
        )
        scale_out = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float32, fxh._as_ptr(p_scale)), fx.make_layout(rows, 1)),
            max_size=False,
        )
        x_vec = fx.logical_divide(x, fx.make_layout(QUANT_VEC, 1))
        q_vec = fx.logical_divide(q, fx.make_layout(QUANT_VEC, 1))
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(64), fx.Float8E4M3FNUZ)

        # Rows past ``rows`` read zeros and their stores are dropped (buffer bounds).
        first = row * (cols // QUANT_VEC) + lane
        values = []
        amax = fx.Float32(0.0)
        for c in range_constexpr(chunks):
            src = x_vec[None, first + c * lanes]
            frag = fx.make_fragment_like(src)
            fx.copy(load_atom, src, frag)
            vec = frag.load().to(fx.Float32)
            values.append(vec)
            amax = fx.math.absf(vec).reduce(fx.ReductionOp.MAX, amax)
        for step in range_constexpr(lanes.bit_length() - 1):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(lanes >> (step + 1)), fx.Int32(64)))
        scale = amax * fx.Float32(1.0 / QUANT_FP8_MAX)
        inv = (scale == 0.0).select(fx.Float32(0.0), fx.Float32(1.0) / scale)

        lo, hi = fx.Float32(-QUANT_FP8_MAX), fx.Float32(QUANT_FP8_MAX)
        for c in range_constexpr(chunks):
            vec = values[c]
            scaled = [fx.math.clampf(vec[k] * inv, lo, hi) for k in range(QUANT_VEC)]
            words = []
            for w in range_constexpr(QUANT_VEC // 4):
                low = rocdl.cvt_pk_fp8_f32(T.i32, _raw(scaled[4 * w]), _raw(scaled[4 * w + 1]),
                                           fx.Int32(0), False)
                words.append(rocdl.cvt_pk_fp8_f32(T.i32, _raw(scaled[4 * w + 2]), _raw(scaled[4 * w + 3]),
                                                  low, True))
            dst = q_vec[None, first + c * lanes]
            frag = fx.make_fragment_like(dst)
            frag.store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32)
                       .bitcast(fx.Float8E4M3FNUZ))
            fx.copy(store_atom, frag, dst)
        if lane == 0:
            scale_out[row] = scale

    @flyc.jit
    def launch(p_x: fx.Pointer, p_q: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32, blocks: fx.Int32,
               stream: fx.Stream):
        a8w4_quant_rows(p_x, p_q, p_scale, rows).launch(grid=(blocks, 1, 1), block=(QUANT_THREADS, 1, 1),
                                                         stream=stream)

    return launch, rows_per_block


@functools.cache
def _compile_cached(device_cache_key, stage, N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, tile_k,
                    padding, E, half_m, prefetch):
    del device_cache_key
    if stage == "gateup":
        return _build_gateup(N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, tile_k, E, half_m, prefetch)
    return _build_down(N, K, TOPK, padding, prefetch, half_m)


def compile_prefill_gemm(*, stage, N, K, TOPK, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, tile_k=None,
                         padding=0, E=None, half_m=False, prefetch=1):
    """Launcher of one A8W4 prefill stage, cached per device (like ``moe_gemm_2stage.gemm``).

    Both stages: ``(input_fp8, weight, output, sorted_ids, sorted_weights, sorted_expert_ids,
    num_valid_ids, w_factor, w_pdelta, a_scale, M, task_num, stream)``. Gate/Up writes BF16
    ``[M, TOPK, N / 2]``; Down (``BLOCK_TILE_SIZE_M`` x ``BLOCK_TILE_SIZE_N`` = 64 x 256, ``tile_k``
    unused) writes one BF16 row of stride ``N + padding / 2`` per sorted row. ``prefetch``: weights in
    flight ahead of their use (Gate/Up: K tiles, 1 or 2; Down: 64-K cores). With ``half_m``, a sorted
    block whose second half is padding computes (and writes) only its first ``BLOCK_TILE_SIZE_M // 2``
    rows in both stages.
    """
    assert stage in ("gateup", "down"), stage
    if stage == "gateup":
        assert tile_k in GATEUP_TILE_K, tile_k
        padding = 0
    else:
        assert (BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N) == DOWN_BLOCK, (BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        tile_k = None
    return _compile_cached(get_device_cache_key(), stage, N, K, TOPK, BLOCK_TILE_SIZE_M,
                           BLOCK_TILE_SIZE_N, tile_k, padding, E, bool(half_m), int(prefetch))


@functools.cache
def _compile_quant_cached(device_cache_key, cols):
    del device_cache_key
    return _build_quant_rows(cols)


def compile_prefill_quant(cols):
    """Per-row FP8 quantization of BF16 ``[rows, cols]``, cached per device: returns ``(launch,
    rows_per_block)``; ``launch(x_bf16, q_fp8, scale_f32[rows], rows, blocks, stream)``."""
    return _compile_quant_cached(get_device_cache_key(), cols)
