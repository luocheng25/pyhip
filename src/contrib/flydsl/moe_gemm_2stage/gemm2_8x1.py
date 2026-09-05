# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental MoE stage2 8x1 down-projection kernel builder."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly_rocdl, llvm, rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import layout_helpers as fxh
from .common import get_down_device_config


def _build_moe_gemm2_8x1(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    del E, activation, swiglu_limit
    assert stage == "down"
    assert alg == "prefill_1x4"
    assert down_path == "8x1"
    assert weight_dtype == "fp8"
    assert weight_quant_type == "ptpc"
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert act_quant_type == "ptpc"
    assert not USE_ATOMIC_WRITE
    assert BLOCK_TILE_SIZE_M == 256
    assert BLOCK_TILE_SIZE_N == 128
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert METADATA_TILE_SIZE_M == BLOCK_TILE_SIZE_M
    assert K in (128, 192, 256, 384, 512)
    expected_tile_k = 64 if K == 192 else 128
    if tile_k is None:
        tile_k = expected_tile_k
    assert tile_k == expected_tile_k
    assert N % BLOCK_TILE_SIZE_N == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)

    BLOCK_M = BLOCK_TILE_SIZE_M
    BLOCK_N = BLOCK_TILE_SIZE_N
    BLOCK_K = tile_k
    NUM_WAVES = 8
    WAVE_M = BLOCK_M // NUM_WAVES
    K_STAGES = K // BLOCK_K
    DEDICATED_K256 = K == 256 and BLOCK_K == 128
    N_TILES = N // BLOCK_N
    TOTAL_CORES = N_TILES * K_STAGES
    WEIGHT_HALF_ATOMS = BLOCK_N * BLOCK_K // (16 * 2)
    WEIGHT_QUARTER_ATOMS = WEIGHT_HALF_ATOMS // 2
    WEIGHT_COPY_ROUNDS_PER_GROUP = WEIGHT_HALF_ATOMS // 256
    OUTPUT_STORES_PER_WAVE = WAVE_M * BLOCK_N * 2 // (64 * 16)
    CSHUFFLE_RECORDS_PER_ROW = BLOCK_N // 8
    CSHUFFLE_N_PAIRS = BLOCK_N // 32
    output_row_stride = N + down_output_padding_bytes // 2
    ROLLING_EPILOGUE = (
        DEDICATED_K256
        and os.environ.get("MOE_8X1_ROLLING_EPILOGUE", "1") != "0"
    )
    down_ops = fxh.FlyObjCache()
    topology_enabled, xcc_count = get_down_device_config()
    se_per_xcc = 4
    cu_per_se = 5
    se_count = xcc_count * se_per_xcc

    assert WEIGHT_COPY_ROUNDS_PER_GROUP in (1, 2)
    assert WEIGHT_QUARTER_ATOMS in (128, 256)
    assert OUTPUT_STORES_PER_WAVE == 8

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    def _pack_scaled_bf16_pairs(values, scales):
        fma_bias = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
        scaled = fxh.eltwise_op("llvm.fma.f32", values, scales, fma_bias)
        selector = fx.Uint32(0x07060302)
        packed = []
        for index in range_constexpr(0, scaled.numel, 2):
            packed.append(
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [
                        _raw(scaled[index + 1]),
                        _raw(scaled[index]),
                        _raw(selector),
                    ],
                    "v_perm_b32 $0, $1, $2, $3",
                    "=v,v,v,s",
                    has_side_effects=True,
                )
            )
        return packed

    def _stage_end():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def _enter_memory_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)

    def _enter_compute_stage():
        rocdl.sched_barrier(0)
        rocdl.s_setprio(3)
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane_id = tid % 64
        wave_id = tid // 64
        wave_group = fx.Int32(
            rocdl.readfirstlane(
                ir.IntegerType.get_signless(32),
                _raw(tid // 256),
            )
        )
        group_tid = tid % 256
        max_valid_id = fxh.view_as_torch_tensor(
            p_num_valid_ids, (1,), fx.Int32
        )[0]
        workgroup_idx = fx.Int32(gpu.block_idx.y)
        e_idx = workgroup_idx
        if const_expr(topology_enabled):
            valid_tasks = fxh.div_up(fx.Uint32(max_valid_id), BLOCK_M)
            tasks_per_se = valid_tasks // se_count
            mapped_tasks = tasks_per_se * se_count
            tasks_per_xcc = tasks_per_se * se_per_xcc
            workgroup_idx_u32 = fx.Uint32(workgroup_idx)
            xcc_id = workgroup_idx_u32 & (xcc_count - 1)
            xcc_local_idx = workgroup_idx_u32 >> 2
            se_slot = xcc_local_idx & (se_per_xcc - 1)
            within_se = xcc_local_idx >> 2
            cu_slot = within_se % cu_per_se
            cu_round = within_se // cu_per_se
            short_cu_tasks = tasks_per_se // cu_per_se
            long_cu_count = tasks_per_se % cu_per_se
            cu_prefix_extra = arith.select(
                cu_slot < long_cu_count,
                cu_slot,
                long_cu_count,
            )
            se_local_rank = (
                cu_slot * short_cu_tasks + cu_prefix_extra + cu_round
            )
            logical_xcc = (xcc_id + 2) & (xcc_count - 1)
            mapped_e_idx = (
                logical_xcc * tasks_per_xcc
                + se_slot * tasks_per_se
                + se_local_rank
            )
            e_idx = fx.Int32(
                arith.select(
                    workgroup_idx_u32 < mapped_tasks,
                    mapped_e_idx,
                    workgroup_idx_u32,
                )
            )

        if e_idx * BLOCK_M < max_valid_id:
            input_tensor = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    p_input, (M, TOPK, K), fx.Float8E4M3FNUZ
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * K,
            )
            sorted_ids = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_sorted_ids) + fx.Int64(e_idx) * BLOCK_M,
                    (BLOCK_M,),
                    fx.Int32,
                ),
                max_size=False,
                num_records_bytes=BLOCK_M * 4,
            )
            sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + fx.Int64(e_idx) * BLOCK_M,
                (BLOCK_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(
                p_sorted_expert_ids, (1,), fx.Int32
            )[e_idx]

            shared_allocator = fx.SharedAllocator()
            weight_ping_storage = shared_allocator.allocate(
                fx.Array[fx.Float8E4M3FNUZ, BLOCK_N * BLOCK_K, 16]
            )
            weight_pong_storage = shared_allocator.allocate(
                fx.Array[fx.Float8E4M3FNUZ, BLOCK_N * BLOCK_K, 16]
            )
            cshuffle_storage = shared_allocator.allocate(
                fx.Array[fx.BFloat16, 4 * 16 * BLOCK_N, 16]
            )
            sorted_lds = fx.make_view(
                fx.recast_iter(fx.Int32, cshuffle_storage.peek().ptr),
                fx.make_layout(BLOCK_M, 1),
            )
            if tid < BLOCK_M:
                sorted_lds[tid] = sorted_ids[tid]
            gpu.barrier()

            mm = down_ops.create_thr_mma(
                fx.Float8E4M3FNUZ, (1, NUM_WAVES, 1)
            )
            mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ)
            )
            c_fake = fx.make_view(
                fx.get_iter(input_tensor),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_c = mm.make_fragment_C(c_fake)
            row_tensor = fx.make_view(
                fx.get_iter(sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_row_scale = down_ops.load_tiled_mma_fragC(
                mm, row_tensor, copy_atom_bits=32
            )
            coord_tensor = fx.make_view(
                fx.get_iter(sorted_lds),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_coord = down_ops.load_tiled_mma_fragC(
                mm, coord_tensor, copy_atom_bits=32
            )

            a_scale_tensor = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    p_a_scale, (M, TOPK), fx.Float32
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * 4,
            )
            a_scale_copy = down_ops.get_buffer_copy_atom(fx.Float32, 32)
            frag_a_scale = mm.make_fragment_C(coord_tensor)
            frag_a_scale_retile = down_ops.get_tiled_mma_retile(
                mm, frag_a_scale, "C", copy_atom=a_scale_copy
            )
            for dst, coord in fxh.all_elements(
                frag_a_scale_retile, frag_coord
            ):
                sorted_id = coord[0].bitcast(fx.Uint32)
                source = fxh.atom_tensor(
                    a_scale_tensor,
                    (sorted_id & 0xFFFFFF, sorted_id >> 24),
                    32,
                )
                fx.copy(a_scale_copy, source, dst)
            frag_row_scale.store(
                frag_row_scale.load() * frag_a_scale.load()
            )

            weight_base = (
                fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
                + fx.Int64(expert_id) * N * K
            )
            weight_view = fx.make_view(
                weight_base, fx.make_layout(N * K, 1)
            )
            weight_flat = fx.rocdl.make_buffer_tensor(
                weight_view,
                max_size=False,
                num_records_bytes=N * K,
            )
            weight_rsrc = fly_rocdl.get_buffer_rsrc(
                _raw(fx.get_iter(weight_flat)),
                results=[ir.Type.parse("!llvm.ptr<8>")],
            )
            weight_staging = [
                fx.make_rmem_tensor(
                    fx.make_layout(16, 1), fx.Float8E4M3FNUZ
                )
                for _ in range_constexpr(
                    max(2, WEIGHT_COPY_ROUNDS_PER_GROUP)
                )
            ]
            weight_store_atom = fx.make_copy_atom(
                fx.UniversalCopy128b(), fx.Float8E4M3FNUZ
            )

            def weight_lds_half_view(storage, n_half):
                return fx.make_view(
                    storage.peek().ptr
                    + n_half * (BLOCK_N // 2) * BLOCK_K,
                    fx.make_layout(
                        ((16, BLOCK_N // 32), (16, BLOCK_K // 16)),
                        ((16, 16 * BLOCK_K), (1, 256)),
                    ),
                )

            def weight_lds_quarter_view(storage, n_quarter):
                return fx.make_view(
                    storage.peek().ptr
                    + n_quarter * (BLOCK_N // 4) * BLOCK_K,
                    fx.make_layout(
                        ((16, BLOCK_N // 64), (16, BLOCK_K // 16)),
                        ((16, 16 * BLOCK_K), (1, 256)),
                    ),
                )

            lds_weight_halves = [
                [
                    weight_lds_half_view(storage, n_half)
                    for n_half in range_constexpr(2)
                ]
                for storage in [weight_ping_storage, weight_pong_storage]
            ]
            lds_weight_quarters = [
                [
                    weight_lds_quarter_view(storage, n_quarter)
                    for n_quarter in range_constexpr(4)
                ]
                for storage in [weight_ping_storage, weight_pong_storage]
            ]
            weight_storage_ptrs = [
                weight_ping_storage.peek().ptr,
                weight_pong_storage.peek().ptr,
            ]

            def weight_atom_index(copy_round):
                return (
                    wave_group * WEIGHT_HALF_ATOMS
                    + group_tid
                    + copy_round * 256
                )

            weight_lane_offsets_bytes = []
            for copy_round in range_constexpr(
                WEIGHT_COPY_ROUNDS_PER_GROUP
            ):
                atom_index = weight_atom_index(copy_round)
                n_group = atom_index // BLOCK_K
                within_group = atom_index % BLOCK_K
                k_group = within_group // 16
                n_inner = within_group % 16
                weight_lane_offsets_bytes.append(
                    n_group * (16 * K)
                    + k_group * 256
                    + n_inner * 16
                )

            quarter_atom_index = (
                wave_group * WEIGHT_QUARTER_ATOMS + group_tid
            )
            quarter_n_group = quarter_atom_index // BLOCK_K
            quarter_within_group = quarter_atom_index % BLOCK_K
            quarter_k_group = quarter_within_group // 16
            quarter_n_inner = quarter_within_group % 16
            weight_quarter_lane_offset_bytes = (
                quarter_n_group * (16 * K)
                + quarter_k_group * 256
                + quarter_n_inner * 16
            )

            def issue_weight_full_load(block_n, k_stage):
                core_base_bytes = (
                    block_n * (BLOCK_N * K)
                    + k_stage * (BLOCK_K // 16) * 256
                )
                for copy_round in range_constexpr(
                    WEIGHT_COPY_ROUNDS_PER_GROUP
                ):
                    loaded = Vec(
                        rocdl_dialect.RawPtrBufferLoadOp(
                            ir.VectorType.get(
                                [4], ir.IntegerType.get_signless(32)
                            ),
                            weight_rsrc,
                            _raw(
                                fx.Int32(
                                    weight_lane_offsets_bytes[copy_round]
                                )
                            ),
                            _raw(fx.Int32(core_base_bytes)),
                            aux=ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(32), 0
                            ),
                        ).result
                    ).bitcast(fx.Float8E4M3FNUZ)
                    weight_staging[copy_round].store(
                        loaded
                    )

            def commit_weight_full(slot):
                for copy_round in range_constexpr(
                    WEIGHT_COPY_ROUNDS_PER_GROUP
                ):
                    atom_index = weight_atom_index(copy_round)
                    n_group = atom_index // BLOCK_K
                    within_group = atom_index % BLOCK_K
                    k_group = within_group // 16
                    n_inner = within_group % 16
                    lds_offset = (
                        n_group * (16 * BLOCK_K)
                        + k_group * 256
                        + n_inner * 16
                    )
                    destination = fx.make_view(
                        weight_storage_ptrs[slot] + lds_offset,
                        fx.make_layout(16, 1),
                    )
                    fx.copy(
                        weight_store_atom,
                        weight_staging[copy_round],
                        destination,
                    )

            def issue_weight_quarter_load(
                block_n, k_stage, n_half, staging_index=0
            ):
                core_base_bytes = (
                    block_n * (BLOCK_N * K)
                    + k_stage * (BLOCK_K // 16) * 256
                    + n_half * (BLOCK_N // 2) * K
                )
                if (
                    const_expr(BLOCK_K == 128)
                    or group_tid < WEIGHT_QUARTER_ATOMS
                ):
                    loaded = Vec(
                        rocdl_dialect.RawPtrBufferLoadOp(
                            ir.VectorType.get(
                                [4], ir.IntegerType.get_signless(32)
                            ),
                            weight_rsrc,
                            _raw(
                                fx.Int32(
                                    weight_quarter_lane_offset_bytes
                                )
                            ),
                            _raw(fx.Int32(core_base_bytes)),
                            aux=ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(32), 0
                            ),
                        ).result
                    ).bitcast(fx.Float8E4M3FNUZ)
                    weight_staging[staging_index].store(loaded)

            def commit_weight_quarter(
                slot, n_half, staging_index=0
            ):
                n_group = quarter_atom_index // BLOCK_K
                within_group = quarter_atom_index % BLOCK_K
                k_group = within_group // 16
                n_inner = within_group % 16
                lds_offset = (
                    n_half * (BLOCK_N // 2) * BLOCK_K
                    + n_group * (16 * BLOCK_K)
                    + k_group * 256
                    + n_inner * 16
                )
                if (
                    const_expr(BLOCK_K == 128)
                    or group_tid < WEIGHT_QUARTER_ATOMS
                ):
                    destination = fx.make_view(
                        weight_storage_ptrs[slot] + lds_offset,
                        fx.make_layout(16, 1),
                    )
                    fx.copy(
                        weight_store_atom,
                        weight_staging[staging_index],
                        destination,
                    )

            # P0 starts before A gather so younger A/scale VMEM can hide B0 latency.
            issue_weight_full_load(0, 0)
            fx.rocdl.sched_barrier(0)

            input_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ
            )
            a_fragments = []
            for k_stage in range_constexpr(K_STAGES):
                a_fake = fx.make_view(
                    fx.get_iter(input_tensor),
                    fx.make_layout((BLOCK_M, BLOCK_K), (1, BLOCK_M)),
                )
                frag_a = mm.make_fragment_B(a_fake)
                for m_rep in range_constexpr(WAVE_M // 16):
                    local_row = (
                        wave_id * 16
                        + m_rep * (NUM_WAVES * 16)
                        + lane_id % 16
                    )
                    sorted_id = sorted_lds[local_row].bitcast(fx.Uint32)
                    for k64 in range_constexpr(BLOCK_K // 64):
                        k_offset = (
                            k_stage * BLOCK_K
                            + k64 * 64
                            + (lane_id // 16) * 16
                        )
                        source = fxh.atom_tensor(
                            input_tensor,
                            (
                                sorted_id & 0xFFFFFF,
                                sorted_id >> 24,
                                k_offset,
                            ),
                            128,
                        )
                        packed_input = fx.make_rmem_tensor(
                            fx.make_layout(16, 1), fx.Float8E4M3FNUZ
                        )
                        fx.copy(input_copy, source, packed_input)
                        packed_values = Vec(packed_input.load())
                        for k8 in range_constexpr(2):
                            if const_expr(BLOCK_K == 64):
                                frag_a[None, m_rep, k8].store(
                                    packed_values.shuffle(
                                        packed_values,
                                        list(range(k8 * 8, k8 * 8 + 8)),
                                    )
                                )
                            else:
                                frag_a[None, m_rep, (k8, k64)].store(
                                    packed_values.shuffle(
                                        packed_values,
                                        list(range(k8 * 8, k8 * 8 + 8)),
                                    )
                                )
                a_fragments.append(frag_a)

            output_base = (
                fxh._as_ptr(p_output, fx.BFloat16)
                + fx.Int64(e_idx) * BLOCK_M * output_row_stride
            )
            output_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    output_base,
                    fx.make_layout(
                        (N, BLOCK_M), (1, output_row_stride)
                    ),
                ),
                max_size=False,
                num_records_bytes=BLOCK_M * output_row_stride * 2,
            )
            cshuffle_lds = cshuffle_storage.peek().view(
                fx.make_layout(4 * 16 * BLOCK_N, 1)
            )
            cshuffle_write_atom = down_ops.get_universal_copy_atom(
                fx.BFloat16, 128
            )
            cshuffle_read_atom = down_ops.get_universal_copy_atom(
                fx.BFloat16, 64
            )
            output_store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(cache_modifier=2), fx.BFloat16
            )
            lane_group = lane_id // 16
            lane_row = lane_id % 16
            row_in_8 = lane_row % 8
            row_half = lane_row // 8
            local_wave = wave_id % 4
            wave_lds_base = local_wave * (16 * BLOCK_N)
            output_destination_offsets = []
            for row_pair in range_constexpr(WAVE_M // 16):
                row_pair_offsets = []
                for n_half in range_constexpr(2):
                    for output_row_half in range_constexpr(2):
                        output_atom = n_half * 8 + lane_id % 8
                        output_row = (
                            wave_id * 16
                            + row_pair * (NUM_WAVES * 16)
                            + output_row_half * 8
                            + lane_id // 8
                        )
                        row_pair_offsets.append(
                            output_tensor.layout(
                                output_atom * 8, output_row
                            )
                        )
                output_destination_offsets.append(row_pair_offsets)

            def pack_cshuffle_record(
                output,
                row_scales,
                weight_scales,
                row_pair,
                n_pair,
            ):
                packed_chunks = []
                for n_group in range_constexpr(
                    2 * n_pair, 2 * n_pair + 2
                ):
                    values = Vec(
                        output[None, n_group, row_pair].load()
                    )
                    weight_scale_values = Vec(
                        weight_scales[None, n_group, row_pair].load()
                    )
                    row_scale_values = Vec(
                        row_scales[None, n_group, row_pair].load()
                    )
                    weighted_values = fxh.eltwise_op(
                        "v_fma_f32",
                        values,
                        weight_scale_values,
                        fx.Float32(0.0),
                    )
                    packed_chunks.extend(
                        _pack_scaled_bf16_pairs(
                            weighted_values, row_scale_values
                        )
                    )
                return Vec.from_elements(
                    packed_chunks, fx.Uint32
                ).bitcast(fx.BFloat16)

            def pack_cshuffle_row_pair(
                output, row_scales, weight_scales, row_pair
            ):
                return [
                    pack_cshuffle_record(
                        output,
                        row_scales,
                        weight_scales,
                        row_pair,
                        n_pair,
                    )
                    for n_pair in range_constexpr(CSHUFFLE_N_PAIRS)
                ]

            def pack_cshuffle_super_record(
                output, row_scales, weight_scales, n_pair
            ):
                weighted_fragments = []
                row_scale_fragments = []
                for row_pair in range_constexpr(WAVE_M // 16):
                    for n_group in range_constexpr(
                        2 * n_pair, 2 * n_pair + 2
                    ):
                        values = Vec(
                            output[None, n_group, row_pair].load()
                        )
                        weighted_fragments.append(
                            fxh.eltwise_op(
                                "v_fma_f32",
                                values,
                                Vec(
                                    weight_scales[
                                        None,
                                        n_group
                                        % (BLOCK_N // 64),
                                        row_pair,
                                    ].load()
                                ),
                                fx.Float32(0.0),
                            )
                        )
                        row_scale_fragments.append(
                            Vec(
                                row_scales[
                                    None, n_group, row_pair
                                ].load()
                            )
                        )

                fma_bias = as_ir_value(
                    fx.Uint32(0x8000)
                ).bitcast(fx.Float32.ir_type)
                scaled_fragments = [
                    fxh.eltwise_op(
                        "llvm.fma.f32",
                        weighted_fragments[fragment_index],
                        row_scale_fragments[fragment_index],
                        fma_bias,
                    )
                    for fragment_index in range_constexpr(
                        len(weighted_fragments)
                    )
                ]

                selector = fx.Uint32(0x07060302)
                packed_records = [
                    []
                    for _ in range_constexpr(WAVE_M // 16)
                ]
                for fragment_index in range_constexpr(
                    len(scaled_fragments)
                ):
                    scaled = scaled_fragments[fragment_index]
                    row_pair = fragment_index // 2
                    for index in range_constexpr(0, scaled.numel, 2):
                        packed_records[row_pair].append(
                            llvm.inline_asm(
                                ir.IntegerType.get_signless(32),
                                [
                                    _raw(scaled[index + 1]),
                                    _raw(scaled[index]),
                                    _raw(selector),
                                ],
                                "v_perm_b32 $0, $1, $2, $3",
                                "=v,v,v,s",
                                has_side_effects=True,
                            )
                        )
                return [
                    Vec.from_elements(chunks, fx.Uint32).bitcast(
                        fx.BFloat16
                    )
                    for chunks in packed_records
                ]

            def write_cshuffle_row_pair(packed_records):
                for n_pair in range_constexpr(CSHUFFLE_N_PAIRS):
                    logical_record = n_pair * 4 + lane_group
                    physical_record = logical_record ^ row_in_8
                    lds_offset = (
                        wave_lds_base
                        + (
                            (row_half * 8 + row_in_8)
                            * CSHUFFLE_RECORDS_PER_ROW
                            + physical_record
                        )
                        * 8
                    )
                    destination = fx.make_view(
                        fx.get_iter(cshuffle_lds) + lds_offset,
                        fx.make_layout(8, 1),
                    )
                    fragment = fx.make_fragment_like(destination)
                    fragment.store(packed_records[n_pair])
                    fx.copy(
                        cshuffle_write_atom, fragment, destination
                    )

            def write_cshuffle_quarter(packed_records, n_half):
                for local_n_pair in range_constexpr(
                    CSHUFFLE_N_PAIRS // 2
                ):
                    n_pair = (
                        n_half * (CSHUFFLE_N_PAIRS // 2)
                        + local_n_pair
                    )
                    logical_record = n_pair * 4 + lane_group
                    physical_record = logical_record ^ row_in_8
                    lds_offset = (
                        wave_lds_base
                        + (
                            (row_half * 8 + row_in_8)
                            * CSHUFFLE_RECORDS_PER_ROW
                            + physical_record
                        )
                        * 8
                    )
                    destination = fx.make_view(
                        fx.get_iter(cshuffle_lds) + lds_offset,
                        fx.make_layout(8, 1),
                    )
                    fragment = fx.make_fragment_like(destination)
                    fragment.store(packed_records[local_n_pair])
                    fx.copy(
                        cshuffle_write_atom, fragment, destination
                    )

            def issue_read_cshuffle_row_pair(block_n, row_pair):
                output_fragments = []
                destinations = []
                for n_half in range_constexpr(2):
                    for output_row_half in range_constexpr(2):
                        output_atom = n_half * 8 + lane_id % 8
                        n_group = output_atom // 2
                        n_pair = n_group // 2
                        chunk_half = n_group % 2
                        lane_group_begin = (output_atom % 2) * 2
                        fragment_pair = []
                        for source_group in range_constexpr(2):
                            logical_record = (
                                n_pair * 4
                                + lane_group_begin
                                + source_group
                            )
                            physical_record = (
                                logical_record ^ (lane_id // 8)
                            )
                            lds_offset = (
                                wave_lds_base
                                + (
                                    (
                                        output_row_half * 8
                                        + lane_id // 8
                                    )
                                    * CSHUFFLE_RECORDS_PER_ROW
                                    + physical_record
                                )
                                * 8
                                + chunk_half * 4
                            )
                            source = fx.make_view(
                                fx.get_iter(cshuffle_lds) + lds_offset,
                                fx.make_layout(4, 1),
                            )
                            fragment = fx.make_fragment_like(source)
                            fx.copy(cshuffle_read_atom, source, fragment)
                            fragment_pair.append(fragment)
                        output_fragments.append(fragment_pair)
                        destination_index = (
                            n_half * 2 + output_row_half
                        )
                        destinations.append(
                            fx.make_view(
                                fx.get_iter(output_tensor)
                                + output_destination_offsets[row_pair][
                                    destination_index
                                ]
                                + block_n * BLOCK_N,
                                fx.make_layout(8, 1),
                            )
                        )
                return output_fragments, destinations

            def issue_read_cshuffle_quarter(
                block_n, row_pair, n_half
            ):
                output_fragments = []
                destinations = []
                for output_row_half in range_constexpr(2):
                    output_atom = n_half * 8 + lane_id % 8
                    n_group = output_atom // 2
                    n_pair = n_group // 2
                    chunk_half = n_group % 2
                    lane_group_begin = (output_atom % 2) * 2
                    fragment_pair = []
                    for source_group in range_constexpr(2):
                        logical_record = (
                            n_pair * 4
                            + lane_group_begin
                            + source_group
                        )
                        physical_record = (
                            logical_record ^ (lane_id // 8)
                        )
                        lds_offset = (
                            wave_lds_base
                            + (
                                (
                                    output_row_half * 8
                                    + lane_id // 8
                                )
                                * CSHUFFLE_RECORDS_PER_ROW
                                + physical_record
                            )
                            * 8
                            + chunk_half * 4
                        )
                        source = fx.make_view(
                            fx.get_iter(cshuffle_lds) + lds_offset,
                            fx.make_layout(4, 1),
                        )
                        fragment = fx.make_fragment_like(source)
                        fx.copy(cshuffle_read_atom, source, fragment)
                        fragment_pair.append(fragment)
                    output_fragments.append(fragment_pair)
                    destination_index = (
                        n_half * 2 + output_row_half
                    )
                    destinations.append(
                        fx.make_view(
                            fx.get_iter(output_tensor)
                            + output_destination_offsets[row_pair][
                                destination_index
                            ]
                            + block_n * BLOCK_N,
                            fx.make_layout(8, 1),
                        )
                    )
                return output_fragments, destinations

            def store_cshuffle_read_results(
                output_fragments, destinations, lgkmcnt=0
            ):
                fx.rocdl.s_waitcnt(
                    _encode_waitcnt(lgkmcnt=lgkmcnt)
                )
                for output_index in range_constexpr(
                    len(output_fragments)
                ):
                    first = Vec(
                        output_fragments[output_index][0].load()
                    )
                    second = Vec(
                        output_fragments[output_index][1].load()
                    )
                    output_fragment = fx.make_rmem_tensor(
                        fx.make_layout(8, 1), fx.BFloat16
                    )
                    output_fragment.store(
                        first.shuffle(second, list(range(8)))
                    )
                    fx.copy(
                        output_store_atom,
                        output_fragment,
                        destinations[output_index],
                    )

            def retire_output(block_n):
                weight_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32)
                    + fx.Int64(expert_id) * N
                    + block_n * BLOCK_N,
                    fx.make_layout((BLOCK_N, BLOCK_M), (1, 0)),
                )
                frag_weight_scale = down_ops.load_tiled_mma_fragC(
                    mm, weight_scale, copy_atom_bits=32
                )
                for row_pair in range_constexpr(WAVE_M // 16):
                    packed_records = pack_cshuffle_row_pair(
                        frag_c,
                        frag_row_scale,
                        frag_weight_scale,
                        row_pair,
                    )
                    write_cshuffle_row_pair(packed_records)
                    output_fragments, destinations = (
                        issue_read_cshuffle_row_pair(
                            block_n, row_pair
                        )
                    )
                    store_cshuffle_read_results(
                        output_fragments, destinations
                    )

            def issue_packed_output_quarter(
                block_n,
                packed_super_records,
                row_pair,
                n_half,
            ):
                packed_records = [
                    packed_super_records[
                        n_half * (CSHUFFLE_N_PAIRS // 2)
                        + local_n_pair
                    ][row_pair]
                    for local_n_pair in range_constexpr(
                        CSHUFFLE_N_PAIRS // 2
                    )
                ]
                write_cshuffle_quarter(packed_records, n_half)
                return issue_read_cshuffle_quarter(
                    block_n, row_pair, n_half
                )

            def store_packed_output_quarter(
                block_n,
                packed_super_records,
                row_pair,
                n_half,
                lgkmcnt=0,
            ):
                output_fragments, destinations = (
                    issue_packed_output_quarter(
                        block_n,
                        packed_super_records,
                        row_pair,
                        n_half,
                    )
                )
                store_cshuffle_read_results(
                    output_fragments,
                    destinations,
                    lgkmcnt=lgkmcnt,
                )

            def run_super_record_mfma(
                frag_weight, k_stage, n_pair, weight_n_group_begin
            ):
                local_n_pair = n_pair % (CSHUFFLE_N_PAIRS // 2)
                for k_iter in range_constexpr(BLOCK_K // 64):
                    for k_atom in range_constexpr(2):
                        for row_pair in range_constexpr(WAVE_M // 16):
                            for quarter_n_group in range_constexpr(2):
                                local_n_group = (
                                    2 * local_n_pair + quarter_n_group
                                )
                                n_group = (
                                    (n_pair // 2)
                                    * (BLOCK_N // 32)
                                    + local_n_group
                                )
                                fx.mma_atom_call(
                                    mma_atom,
                                    frag_c[
                                        None, n_group, row_pair
                                    ],
                                    (
                                        frag_weight[
                                            None,
                                            weight_n_group_begin
                                            + quarter_n_group,
                                            k_atom,
                                        ]
                                        if const_expr(BLOCK_K == 64)
                                        else frag_weight[
                                            None,
                                            weight_n_group_begin
                                            + quarter_n_group,
                                            (k_atom, k_iter),
                                        ]
                                    ),
                                    (
                                        a_fragments[k_stage][
                                            None, row_pair, k_atom
                                        ]
                                        if const_expr(BLOCK_K == 64)
                                        else a_fragments[k_stage][
                                            None,
                                            row_pair,
                                            (k_atom, k_iter),
                                        ]
                                    ),
                                    frag_c[
                                        None, n_group, row_pair
                                    ],
                                )

            def clear_super_record(n_pair):
                for row_pair in range_constexpr(WAVE_M // 16):
                    for n_group in range_constexpr(
                        2 * n_pair, 2 * n_pair + 2
                    ):
                        frag_c[
                            None, n_group, row_pair
                        ].fill(0)

            def load_weight_scale_quarter(block_n, n_pair):
                weight_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32)
                    + fx.Int64(expert_id) * N
                    + block_n * BLOCK_N
                    + n_pair * (BLOCK_N // CSHUFFLE_N_PAIRS),
                    fx.make_layout(
                        (
                            BLOCK_N // CSHUFFLE_N_PAIRS,
                            BLOCK_M,
                        ),
                        (1, 0),
                    ),
                )
                return down_ops.load_tiled_mma_fragC(
                    mm, weight_scale, copy_atom_bits=32
                )

            def constrain_mfma_valu_packet():
                for slot in range_constexpr(16):
                    fx.rocdl.sched_group_barrier(0x8, 1, 0)
                    if const_expr(slot < 13):
                        fx.rocdl.sched_group_barrier(0x2, 3, 0)
                    elif const_expr(slot == 13):
                        fx.rocdl.sched_group_barrier(0x2, 1, 0)
                fx.rocdl.sched_barrier(0)

            def pending_weight_position(half_core):
                source_block_n = half_core // (K_STAGES * 2)
                source_stage = half_core % (K_STAGES * 2)
                source_k_stage = source_stage % K_STAGES
                source_n_half = source_stage // K_STAGES
                pending_core = (
                    source_block_n * K_STAGES
                    + source_k_stage
                    + 1
                )
                return (
                    pending_core // K_STAGES,
                    pending_core % K_STAGES,
                    source_n_half,
                    pending_core < TOTAL_CORES,
                    pending_core & 1,
                )

            # P0: fill B(0, 0), then seed two future half-B requests.
            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=4))
            commit_weight_full(0)
            fx.rocdl.sched_barrier(0)
            prefetch0 = pending_weight_position(0)
            prefetch1 = pending_weight_position(1)
            if const_expr(prefetch0[3]):
                issue_weight_quarter_load(
                    prefetch0[0], prefetch0[1], prefetch0[2], 0
                )
            if const_expr(prefetch1[3]):
                issue_weight_quarter_load(
                    prefetch1[0], prefetch1[1], prefetch1[2], 1
                )
            fx.rocdl.s_waitcnt(
                _encode_waitcnt(vmcnt=1 if prefetch1[3] else 0)
            )
            _stage_end()

            frag_c.fill(0)
            pending_packed_super_records = None
            pending_scale_quarters = None

            # S0/S1 of q=0 are peeled into the prologue. Their future operand is q=2.
            _enter_memory_stage()
            scale_quarters = []
            if const_expr(ROLLING_EPILOGUE):
                scale_quarters.append(
                    load_weight_scale_quarter(0, 0)
                )
            frag_weight = down_ops.load_tiled_mma_fragA(
                mm,
                lds_weight_halves[0][0],
                copy_atom_bits=128,
            )
            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=2))
            if const_expr(prefetch0[3]):
                commit_weight_quarter(
                    prefetch0[4], prefetch0[2], 0
                )
            prefetch2 = pending_weight_position(2)
            if const_expr(prefetch2[3]):
                fx.rocdl.sched_barrier(0)
                issue_weight_quarter_load(
                    prefetch2[0], prefetch2[1], prefetch2[2], 0
                )
            fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            _stage_end()

            _enter_compute_stage()
            for n_pair in range_constexpr(2):
                run_super_record_mfma(
                    frag_weight, 0, n_pair, 2 * n_pair
                )
            _enter_memory_stage()
            _stage_end()

            if wave_group == 1:
                _stage_end()

            packed_super_records = []
            total_half_cores = TOTAL_CORES * 2
            for half_core in range_constexpr(1, total_half_cores):
                block_n = half_core // (K_STAGES * 2)
                stage_in_tile = half_core % (K_STAGES * 2)
                k_stage = stage_in_tile % K_STAGES
                n_half = stage_in_tile // K_STAGES
                current_core = block_n * K_STAGES + k_stage
                slot = current_core & 1
                n_pair_begin = n_half * (CSHUFFLE_N_PAIRS // 2)
                pending = pending_weight_position(half_core)
                pending_n_half = pending[2]
                has_pending = pending[3]

                next_pending = pending_weight_position(half_core + 1)
                has_next_pending = (
                    half_core + 1 < total_half_cores
                    and next_pending[3]
                )

                future_pending = pending_weight_position(half_core + 2)
                has_future_pending = (
                    half_core + 2 < total_half_cores
                    and future_pending[3]
                )
                staging_index = half_core & 1

                if const_expr(stage_in_tile == 0):
                    scale_quarters = []
                    packed_super_records = []

                _enter_memory_stage()
                if const_expr(ROLLING_EPILOGUE):
                    scale_quarters.append(
                        load_weight_scale_quarter(
                            block_n, stage_in_tile
                        )
                    )

                output_fragments = None
                output_destinations = None
                if const_expr(block_n > 0):
                    if const_expr(ROLLING_EPILOGUE):
                        output_n_half = stage_in_tile // 2
                        output_row_pair = stage_in_tile % 2
                        (
                            output_fragments,
                            output_destinations,
                        ) = issue_packed_output_quarter(
                            block_n - 1,
                            pending_packed_super_records,
                            output_row_pair,
                            output_n_half,
                        )
                    elif const_expr(stage_in_tile == 0):
                        retire_output(block_n - 1)

                frag_weight_quarters = []
                frag_weight_quarters.append(
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_quarters[slot][2 * n_half],
                        copy_atom_bits=128,
                    )
                )

                if const_expr(ROLLING_EPILOGUE and block_n > 0):
                    store_cshuffle_read_results(
                        output_fragments,
                        output_destinations,
                        lgkmcnt=4,
                    )
                    fx.rocdl.sched_barrier(0)

                frag_weight_quarters.append(
                    down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_quarters[slot][2 * n_half + 1],
                        copy_atom_bits=128,
                    )
                )

                if const_expr(not ROLLING_EPILOGUE or block_n == 0):
                    frag_weight = down_ops.load_tiled_mma_fragA(
                        mm,
                        lds_weight_halves[slot][n_half],
                        copy_atom_bits=128,
                    )

                if const_expr(has_pending):
                    current_vmem = (
                        2 if ROLLING_EPILOGUE else 0
                    ) + (
                        2
                        if ROLLING_EPILOGUE and block_n > 0
                        else (
                            OUTPUT_STORES_PER_WAVE
                            if not ROLLING_EPILOGUE
                            and block_n > 0
                            and stage_in_tile == 0
                            else 0
                        )
                    ) + (
                        1 if has_next_pending else 0
                    )
                    fx.rocdl.s_waitcnt(
                        _encode_waitcnt(vmcnt=current_vmem)
                    )
                    commit_weight_quarter(
                        pending[4],
                        pending_n_half,
                        staging_index,
                    )

                if const_expr(has_future_pending):
                    fx.rocdl.sched_barrier(0)
                    issue_weight_quarter_load(
                        future_pending[0],
                        future_pending[1],
                        future_pending[2],
                        staging_index,
                    )

                _stage_end()

                _enter_compute_stage()
                for local_n_pair in range_constexpr(
                    CSHUFFLE_N_PAIRS // 2
                ):
                    n_pair = n_pair_begin + local_n_pair
                    if const_expr(k_stage == 0):
                        clear_super_record(n_pair)
                    fx.rocdl.sched_barrier(0)
                    if const_expr(ROLLING_EPILOGUE and block_n > 0):
                        run_super_record_mfma(
                            frag_weight_quarters[local_n_pair],
                            k_stage,
                            n_pair,
                            0,
                        )
                    else:
                        run_super_record_mfma(
                            frag_weight,
                            k_stage,
                            n_pair,
                            2 * local_n_pair,
                        )
                    if const_expr(
                        ROLLING_EPILOGUE
                        and stage_in_tile == 0
                        and local_n_pair == 0
                        and block_n > 0
                    ):
                        pending_packed_super_records.append(
                            pack_cshuffle_super_record(
                                frag_c,
                                frag_row_scale,
                                pending_scale_quarters[-1],
                                CSHUFFLE_N_PAIRS - 1,
                            )
                        )
                        constrain_mfma_valu_packet()
                    if const_expr(
                        ROLLING_EPILOGUE
                        and stage_in_tile == 1
                        and local_n_pair == 1
                    ):
                        packed_super_records.append(
                            pack_cshuffle_super_record(
                                frag_c,
                                frag_row_scale,
                                scale_quarters[0],
                                0,
                            )
                        )
                        constrain_mfma_valu_packet()
                    if const_expr(
                        ROLLING_EPILOGUE
                        and stage_in_tile == 2
                        and local_n_pair == 0
                    ):
                        packed_super_records.append(
                            pack_cshuffle_super_record(
                                frag_c,
                                frag_row_scale,
                                scale_quarters[1],
                                1,
                            )
                        )
                        constrain_mfma_valu_packet()
                    if const_expr(
                        ROLLING_EPILOGUE
                        and stage_in_tile == 3
                        and local_n_pair == 1
                    ):
                        packed_super_records.append(
                            pack_cshuffle_super_record(
                                frag_c,
                                frag_row_scale,
                                scale_quarters[2],
                                2,
                            )
                        )
                        constrain_mfma_valu_packet()

                if const_expr(
                    stage_in_tile + 1 == K_STAGES * 2
                ):
                    pending_packed_super_records = (
                        packed_super_records
                    )
                    pending_scale_quarters = scale_quarters

                fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                _enter_memory_stage()
                _stage_end()

            if const_expr(ROLLING_EPILOGUE):
                fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                pending_packed_super_records.append(
                    pack_cshuffle_super_record(
                        frag_c,
                        frag_row_scale,
                        pending_scale_quarters[-1],
                        CSHUFFLE_N_PAIRS - 1,
                    )
                )
                for output_n_half in range_constexpr(2):
                    for row_pair in range_constexpr(
                        WAVE_M // 16
                    ):
                        store_packed_output_quarter(
                            N_TILES - 1,
                            pending_packed_super_records,
                            row_pair,
                            output_n_half,
                        )
            else:
                retire_output(N_TILES - 1)
            _stage_end()
            if wave_group == 0:
                _stage_end()

    @flyc.jit
    def launch_prefill_8x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
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
        down_ops.clear_all()
        kernel = moe_2stage_down_prefill_8x1(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            value_attrs={
                "passthrough": [["target-features", "-packed-fp32-ops"]]
            },
        )
        kernel.launch(
            grid=(1, task_num, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch_prefill_8x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_8x1
