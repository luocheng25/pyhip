# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage2 8x1 down-projection kernel builder.

BM=256 / BN=128 / BK=128, 8 waves (512 threads) split along M only:
每个 wave 独占 32 行 activation（常驻 VGPR），8 个 wave 共享同一块经 LDS
ping-pong 的 128xBK weight tile。设计与取舍见
``docs/design_moe_gemm2_8x1.md``。
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import layout_helpers as fxh
from .common import get_down_device_config as _get_down_device_config

# gfx942 raw-buffer aux bit 1 selects the non-temporal policy.
_DOWN_STORE_CACHE_MODIFIER = 2

BLOCK_M = 256
BLOCK_N = 128
BLOCK_K = 128
NUM_WAVES = 8
NUM_THREADS = NUM_WAVES * 64
WAVE_M = BLOCK_M // NUM_WAVES  # 32 rows per wave
# preshuffle 的原子单元：16 通道 x 16 K-byte = 256B，通道组步长 16*K。
PRESHUFFLE_UNIT = 256
PRESHUFFLE_GROUP = 16


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
    down_path="8x1",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    assert stage == "down"
    assert down_path == "8x1"
    assert alg == "prefill_1x4"
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert weight_dtype == "fp8", "8x1 currently only supports native fp8"
    assert weight_quant_type == "ptpc" and act_quant_type == "ptpc", (
        "8x1 currently only supports PTPC weight+activation scales; "
        "per_tensor is a TODO"
    )
    assert BLOCK_TILE_SIZE_M == BLOCK_M
    assert BLOCK_TILE_SIZE_N == BLOCK_N
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert BLOCK_TILE_SIZE_M == METADATA_TILE_SIZE_M
    assert N % BLOCK_N == 0, "8x1 requires N to be a multiple of 128"
    assert K % BLOCK_K == 0, "8x1 requires K in {128, 256, 384, ...}; K=192 is a TODO"
    assert K <= 384, "K>384 exceeds the activation register budget"
    assert down_output_padding_bytes in (0, 32, 64, 128)

    topology_enabled, generic_xcd_count = _get_down_device_config()

    gfx942_xcc_count = 4
    gfx942_se_per_xcc = 4
    gfx942_cu_per_se = 5
    gfx942_se_count = gfx942_xcc_count * gfx942_se_per_xcc
    gfx942_cu_count = gfx942_se_count * gfx942_cu_per_se

    nBN = N // BLOCK_N
    nBK = K // BLOCK_K

    weight_dtype = fx.Float8E4M3FNUZ

    output_row_stride = N + (
        down_output_padding_bytes // (fx.BFloat16.width // 8)
        if down_output_padding_bytes is not None
        else 0
    )

    lds_b_elems = BLOCK_N * BLOCK_K
    lds_act_elems = BLOCK_M * BLOCK_K
    cshuffle_elems = NUM_WAVES * 16 * 64
    lds_total = (
        2 * lds_b_elems
        + cshuffle_elems * (fx.BFloat16.width // 8)
        + 2 * BLOCK_N * 4
        + BLOCK_M * 8
    )
    assert lds_total <= 64 * 1024, f"8x1 LDS budget exceeded: {lds_total}B"
    # A 的 prologue staging 复用 ping+pong 这段连续 32KiB。
    assert lds_act_elems == 2 * lds_b_elems

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    def _pack_scaled_bf16_pairs(values, scales):
        # 0x8000在这里按f32位型参与FMA；v_perm只取高16位，因此不是BF16 RNE。
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

    down_ops = fxh.FlyObjCache()

    @flyc.jit
    def _map_down_task(
        valid_rows: fx.Int32,
        topology_map: fx.Constexpr[bool],
        task_rows: fx.Constexpr[int],
        topology_min_tasks: fx.Constexpr[int] = 0,
    ):
        """将down工作组映射到generic XCD或MI308X XCC/SE/CU。"""
        workgroup_idx = fx.Int32(fx.gpu.block_idx.y)
        valid_rows_u32 = fx.Uint32(valid_rows)
        valid_tasks = valid_rows_u32 // task_rows
        valid_tasks += fx.Uint32(valid_rows_u32 % task_rows != 0)

        swizzle_chunk = valid_tasks // generic_xcd_count
        swizzle_limit = swizzle_chunk * generic_xcd_count
        swizzled_e_idx = (
            workgroup_idx % generic_xcd_count
        ) * swizzle_chunk + workgroup_idx // generic_xcd_count
        generic_e_idx = fx.Int32(
            arith.select(workgroup_idx < swizzle_limit, swizzled_e_idx, workgroup_idx)
        )
        e_idx = generic_e_idx
        if const_expr(topology_map):
            workgroup_idx_u32 = fx.Uint32(workgroup_idx)
            tasks_per_se = fx.Uint32(valid_tasks // gfx942_se_count)
            mapped_tasks = tasks_per_se * gfx942_se_count
            tasks_per_xcc = tasks_per_se * gfx942_se_per_xcc

            xcc_id = workgroup_idx_u32 & (gfx942_xcc_count - 1)
            xcc_local_idx = workgroup_idx_u32 >> 2
            se_slot = xcc_local_idx & (gfx942_se_per_xcc - 1)
            within_se = xcc_local_idx >> 2
            cu_slot = within_se % gfx942_cu_per_se
            cu_round = within_se // gfx942_cu_per_se
            short_cu_tasks = tasks_per_se // gfx942_cu_per_se
            long_cu_count = tasks_per_se % gfx942_cu_per_se
            cu_prefix_extra = arith.select(
                cu_slot < long_cu_count, cu_slot, long_cu_count
            )
            se_local_rank = cu_slot * short_cu_tasks + cu_prefix_extra + cu_round
            logical_xcc = (xcc_id + 2) & (gfx942_xcc_count - 1)
            topology_e_idx = (
                logical_xcc * tasks_per_xcc + se_slot * tasks_per_se + se_local_rank
            )
            topology_e_idx = fx.Int32(
                arith.select(
                    workgroup_idx_u32 < mapped_tasks, topology_e_idx, workgroup_idx_u32
                )
            )
            e_idx = fx.Int32(
                arith.select(
                    valid_tasks >= topology_min_tasks, topology_e_idx, generic_e_idx
                )
            )
        return e_idx

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer,  # fp8 [M, TOPK, K]
        p_weight: fx.Pointer,  # preshuffled fp8 [E, N, K]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N]
        p_sorted_ids: fx.Pointer,  # int32 [num_tokens_sorted]
        p_sorted_weights: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_expert_ids: fx.Pointer,  # int32 [num_blocks]
        p_num_valid_ids: fx.Pointer,  # int32 [2]
        p_w_scale: fx.Pointer,  # f32 [E, N]  per-output-channel
        p_a_scale: fx.Pointer,  # f32 [M, TOPK] per-token
        M: fx.Int32,
        topology_map: fx.Constexpr[bool],
    ):
        """M256xN128：8 个 wave 沿 M 分布，共享经 LDS ping-pong 的 weight tile。"""
        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        e_idx = _map_down_task(max_valid_id, topology_map, BLOCK_M, gfx942_cu_count)
        e_offset = fx.Int64(e_idx)
        if e_idx * BLOCK_M < max_valid_id:
            tid = fx.Int32(fx.thread_idx.x)
            wave_id = tid // 64
            lane_id = tid % 64
            lane_row = lane_id % 16  # MMA_N / MMA_M 的行内索引
            lane_grp = lane_id // 16

            # ---- 1. views ----
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M) * (TOPK * K),
            )
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16)
                + e_offset * (BLOCK_M * output_row_stride),
                (BLOCK_M, output_row_stride),
            )
            out_bt = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=BLOCK_M * output_row_stride * 2,
            )
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_offset * BLOCK_M, (BLOCK_M,), fx.Int32
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_offset * BLOCK_M,
                (BLOCK_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]

            # preshuffle 的等价形式：addr(c,k) = (c/16)*16K + (k/16)*256 + (c%16)*16 + k%16
            weight_bt = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id) * (N * K),
                    fx.make_layout(N * K, 1),
                ),
                max_size=False,
                num_records_bytes=N * K,
            )
            w_scale_bt = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fxh._as_ptr(p_w_scale) + fx.Int64(expert_id) * N,
                    fx.make_layout(N, 1),
                ),
                max_size=False,
                num_records_bytes=N * 4,
            )
            arg_a_scale = fx.make_view(
                fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                fx.make_layout((M, TOPK), (TOPK, 1)),
            )
            arg_a_scale = fx.rocdl.make_buffer_tensor(
                arg_a_scale,
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * 4,
            )

            # ---- 2. LDS ----
            shared_allocator = fx.SharedAllocator()
            b_ping_storage = shared_allocator.allocate(
                fx.Array[weight_dtype, lds_b_elems, 16]
            )
            b_pong_storage = shared_allocator.allocate(
                fx.Array[weight_dtype, lds_b_elems, 16]
            )
            scale_storage = shared_allocator.allocate(
                fx.Array[fx.Float32, 2 * BLOCK_N, 16]
            )
            cshuffle_storage = shared_allocator.allocate(
                fx.Array[fx.BFloat16, cshuffle_elems, 16]
            )

            # weight tile 的 native 布局：((16 c, 8 组), (16 k, 8 单元))
            b_tile_layout = fx.make_layout(
                ((PRESHUFFLE_GROUP, BLOCK_N // PRESHUFFLE_GROUP),
                 (PRESHUFFLE_GROUP, BLOCK_K // PRESHUFFLE_GROUP)),
                ((PRESHUFFLE_GROUP, BLOCK_K * PRESHUFFLE_GROUP),
                 (1, PRESHUFFLE_UNIT)),
            )
            lds_b = [
                b_ping_storage.peek().view(b_tile_layout),
                b_pong_storage.peek().view(b_tile_layout),
            ]
            lds_b_raw = [
                b_ping_storage.peek().view(fx.make_layout(lds_b_elems, 1)),
                b_pong_storage.peek().view(fx.make_layout(lds_b_elems, 1)),
            ]
            # ping+pong 连续 32KiB，prologue 借用为 activation staging。
            lds_act = b_ping_storage.peek().view(
                fx.make_layout(
                    ((PRESHUFFLE_GROUP, BLOCK_M // PRESHUFFLE_GROUP),
                     (PRESHUFFLE_GROUP, BLOCK_K // PRESHUFFLE_GROUP)),
                    ((PRESHUFFLE_GROUP, BLOCK_K * PRESHUFFLE_GROUP),
                     (1, PRESHUFFLE_UNIT)),
                )
            )
            lds_act_raw = b_ping_storage.peek().view(
                fx.make_layout(lds_act_elems, 1)
            )
            scale_lds = scale_storage.peek().view(fx.make_layout(2 * BLOCK_N, 1))
            cshuffle_lds = cshuffle_storage.peek().view(
                fx.make_layout(cshuffle_elems, 1)
            )

            u8_copy_atom = down_ops.get_universal_copy_atom(weight_dtype, 128)
            f32_copy_atom = down_ops.get_universal_copy_atom(fx.Float32, 128)
            bf16_copy_atom = down_ops.get_universal_copy_atom(fx.BFloat16, 128)
            u8_buf_atom = down_ops.get_buffer_copy_atom(weight_dtype, 128)
            f32_buf_atom = down_ops.get_buffer_copy_atom(fx.Float32, 128)
            bf16_buf_atom = down_ops.get_buffer_copy_atom(fx.BFloat16, 128)

            def flat_atom(buffer_tensor, elem_off, num_values):
                """rank-1 buffer tensor 的 copy atom（atom_tensor 只接受多维 coord）。"""
                return fx.make_view(
                    fx.get_iter(buffer_tensor) + elem_off,
                    fx.make_layout(num_values, 1),
                )

            # ---- 3. weight tile 的 global<->LDS 静态地址（循环不变） ----
            # thread t 负责 tile 内第 t 和第 t+512 个 16B 单元。
            def weight_unit_addrs(unit):
                # unit 在 tile 内线性编号（按 global 连续序），共 1024 个 16B。
                gidx = unit // 128  # 通道组 0..7
                off = (unit % 128) * 16  # 组内字节 0..2047
                k1 = off // PRESHUFFLE_UNIT
                cr = (off % PRESHUFFLE_UNIT) // 16
                c_local = gidx * PRESHUFFLE_GROUP + cr
                # perm: c = (m%4) + 4*(m/16 >= 4) + 8*(m%16/4) + 32*((m/16)%4)
                # 反解 m
                v = c_local % 4
                rm_hi = (c_local // 4) % 2
                g = (c_local // 8) % 4
                rm_lo = c_local // 32
                rm = rm_hi * 4 + rm_lo
                m_r = v + 4 * g
                lds_byte = rm * (PRESHUFFLE_GROUP * BLOCK_K) + k1 * PRESHUFFLE_UNIT + m_r * 16
                return gidx, off, lds_byte

            weight_units = [
                weight_unit_addrs(tid + fx.Int32(i * NUM_THREADS))
                for i in range_constexpr(2)
            ]

            def load_weight_tile(block_n, block_k, dst_regs):
                for i in range_constexpr(2):
                    gidx, off, _ = weight_units[i]
                    elem_off = (
                        (fx.Int32(block_n) * (BLOCK_N // PRESHUFFLE_GROUP) + gidx)
                        * (PRESHUFFLE_GROUP * K)
                        + fx.Int32(block_k) * (PRESHUFFLE_GROUP * BLOCK_K)
                        + off
                    )
                    fx.copy(
                        u8_buf_atom,
                        flat_atom(weight_bt, elem_off, 16),
                        dst_regs[i],
                    )

            def commit_weight_tile(slot, src_regs):
                for i in range_constexpr(2):
                    _, _, lds_byte = weight_units[i]
                    dst = fx.make_view(
                        fx.get_iter(lds_b_raw[slot]) + lds_byte, fx.make_layout(16, 1)
                    )
                    fx.copy(u8_copy_atom, src_regs[i], dst)

            stage_regs = [
                [
                    fx.make_fragment_like(
                        fx.make_view(fx.get_iter(lds_b_raw[0]), fx.make_layout(16, 1))
                    )
                    for _ in range_constexpr(2)
                ]
                for _ in range_constexpr(2)
            ]

            # ---- 4. prologue: activation gather -> LDS -> 常驻 VGPR ----
            mm = down_ops.create_thr_mma(weight_dtype, (1, NUM_WAVES, 1))
            frag_act_slots = []
            for kc in range_constexpr(nBK):
                # thread t 每轮搬 4 个 16B 单元；unit = (row, k1)
                for i in range_constexpr(4):
                    unit = tid + fx.Int32(i * NUM_THREADS)
                    row = unit // (BLOCK_K // PRESHUFFLE_GROUP)
                    k1 = unit % (BLOCK_K // PRESHUFFLE_GROUP)
                    sorted_id = fx.Uint32(arg_p_sorted_ids[row])
                    atom_A = fxh.atom_tensor(
                        arg_p_input,
                        (
                            sorted_id & 0xFFFFFF,
                            sorted_id >> 24,
                            fx.Int32(kc * BLOCK_K) + k1 * PRESHUFFLE_GROUP,
                        ),
                        128,
                    )
                    lds_byte = (
                        (row // PRESHUFFLE_GROUP) * (PRESHUFFLE_GROUP * BLOCK_K)
                        + k1 * PRESHUFFLE_UNIT
                        + (row % PRESHUFFLE_GROUP) * 16
                    )
                    dst = fx.make_view(
                        fx.get_iter(lds_act_raw) + lds_byte, fx.make_layout(16, 1)
                    )
                    fx.copy(
                        down_ops.get_buffer_copy_atom(weight_dtype, 128), atom_A, dst
                    )
                fx.gpu.barrier()
                frag_act_slots.append(
                    down_ops.load_tiled_mma_fragB(mm, lds_act, copy_atom_bits=128)
                )
                fx.gpu.barrier()

            # ---- 5. C / scale fragments ----
            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = mm.make_fragment_C(c_fake_tensor)

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_sorted_weight = down_ops.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )
            a_scale_atom = down_ops.get_buffer_copy_atom(fx.Float32, 32)
            coord_tensor = fx.make_view(
                fx.get_iter(arg_p_sorted_ids),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_coord = down_ops.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
            frag_pt_scales = mm.make_fragment_C(coord_tensor)
            frag_pt_scalesr = down_ops.get_tiled_mma_retile(
                mm, frag_pt_scales, "C", copy_atom=a_scale_atom
            )
            for dst, coord in fxh.all_elements(frag_pt_scalesr, frag_coord):
                sorted_id = coord[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_a_scale, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32
                )
                fx.copy(a_scale_atom, atom_A, dst)
            for frag_pt, frag_sw in fxh.all_elements(frag_pt_scales, frag_sorted_weight):
                frag_pt.store(frag_pt.load() * frag_sw.load())
            frag_sorted_weight = frag_pt_scales

            # ---- 6. per-channel weight scale -> LDS ----
            def load_w_scale(block_n):
                # 双缓冲：预取 block_n+1 时 block_n 的 epilogue 还没读完。
                if tid < BLOCK_N // 4:
                    off = fx.Int32(block_n) * BLOCK_N + tid * 4
                    dst = fx.make_view(
                        fx.get_iter(scale_lds) + (block_n % 2) * BLOCK_N + tid * 4,
                        fx.make_layout(4, 1),
                    )
                    frag = fx.make_fragment_like(dst)
                    fx.copy(
                        f32_buf_atom, flat_atom(w_scale_bt, off, 4), frag
                    )
                    fx.copy(f32_copy_atom, frag, dst)

            # ---- 7. epilogue: scale + pack + CShuffle + dwordx4 store ----
            wave_lds_base = wave_id * (16 * 64)

            def epilogue(block_n):
                scale_base = (block_n % 2) * BLOCK_N
                for h in range_constexpr(2):  # 通道半区 [64h, 64h+64)
                    scales = []
                    for cp in range_constexpr(2):
                        t = 2 * h + cp
                        src = fx.make_view(
                            fx.get_iter(scale_lds) + scale_base + lane_grp * 8 + t * 32,
                            fx.make_layout(8, 1),
                        )
                        frag = fx.make_fragment_like(src)
                        fx.copy(f32_copy_atom, src, frag)
                        scales.append(frag)
                    for rest_n in range_constexpr(2):
                        for cp in range_constexpr(2):
                            t = 2 * h + cp
                            lo = Vec(fragC[None, t, rest_n].load())
                            hi = Vec(fragC[None, t + 4, rest_n].load())
                            sw_lo = Vec(frag_sorted_weight[None, t, rest_n].load())
                            sw_hi = Vec(frag_sorted_weight[None, t + 4, rest_n].load())
                            sc = Vec(scales[cp].load())
                            packed = Vec.from_elements(
                                _pack_scaled_bf16_pairs(
                                    lo * Vec.from_elements(
                                        [sc[j] for j in range_constexpr(4)], fx.Float32
                                    ),
                                    sw_lo,
                                )
                                + _pack_scaled_bf16_pairs(
                                    hi * Vec.from_elements(
                                        [sc[4 + j] for j in range_constexpr(4)],
                                        fx.Float32,
                                    ),
                                    sw_hi,
                                ),
                                fx.Uint32,
                            ).bitcast(fx.BFloat16)
                            logical_atom = lane_grp + 4 * cp
                            physical_atom = logical_atom ^ (lane_row % 8)
                            lds_off = (
                                wave_lds_base + (lane_row * 8 + physical_atom) * 8
                            )
                            dst = fx.make_view(
                                fx.get_iter(cshuffle_lds) + lds_off,
                                fx.make_layout(8, 1),
                            )
                            frag = fx.make_fragment_like(dst)
                            frag.store(packed)
                            fx.copy(bf16_copy_atom, frag, dst)

                        fx.rocdl.sched_barrier(0)
                        out_frags = []
                        out_atoms = []
                        for row_half in range_constexpr(2):
                            out_row = row_half * 8 + lane_id // 8
                            atom = lane_id % 8
                            physical = atom ^ (lane_id // 8)
                            lds_off = wave_lds_base + (out_row * 8 + physical) * 8
                            src = fx.make_view(
                                fx.get_iter(cshuffle_lds) + lds_off,
                                fx.make_layout(8, 1),
                            )
                            frag = fx.make_fragment_like(src)
                            fx.copy(bf16_copy_atom, src, frag)
                            out_frags.append(frag)
                            global_row = (
                                rest_n * 128 + wave_id * 16 + out_row
                            )
                            global_col = (
                                fx.Int32(block_n) * BLOCK_N + h * 64 + atom * 8
                            )
                            out_atoms.append(
                                fxh.atom_tensor(out_bt, (global_row, global_col), 128)
                            )
                        # 先消费较旧的 ds_read，不强制较新的一条完成。
                        # CShuffle 区是 wave 私有的，只需 lgkmcnt 定序，不需 workgroup barrier。
                        fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=1))
                        fx.copy(bf16_buf_atom, out_frags[0], out_atoms[0])
                        fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                        fx.copy(bf16_buf_atom, out_frags[1], out_atoms[1])
                        fx.rocdl.sched_barrier(0)

            # ---- 8. 主循环 ----
            def enter_read_write_stage():
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_barrier()
                fx.rocdl.s_setprio(0)
                fx.rocdl.sched_barrier(0)

            def enter_compute_stage():
                fx.rocdl.sched_barrier(0x40)
                fx.rocdl.s_barrier()
                fx.rocdl.s_setprio(1)
                fx.rocdl.sched_barrier(0x40)

            load_weight_tile(0, 0, stage_regs[0])
            load_w_scale(0)
            fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            commit_weight_tile(0, stage_regs[0])
            if const_expr(nBK > 1):
                load_weight_tile(0, 1, stage_regs[1])
            else:
                load_weight_tile(1, 0, stage_regs[1])
            fx.gpu.barrier()

            for block_n in range_constexpr(nBN):
                fragC.fill(0)
                for k in range_constexpr(nBK):
                    step = block_n * nBK + k
                    cur = step % 2
                    nxt = 1 - cur
                    # 一个 step 一个 barrier：它同时保证上一步对 lds_b[cur] 的写入已完成，
                    # 以及上一步对 lds_b[nxt] 的读取已完成。
                    fx.gpu.barrier()
                    frag_weight = down_ops.load_tiled_mma_fragA(
                        mm, lds_b[cur], copy_atom_bits=128
                    )
                    # 只等本步要用的 2 条 weight load。gfx9 的 vmcnt 按发射序退休，
                    # 上一块 epilogue 的 8 条 store 发射在它们之后，可以继续在飞。
                    keep_vmcnt = 8 if (k == 0 and block_n > 0) else 0
                    fx.rocdl.s_waitcnt(_encode_waitcnt(vmcnt=keep_vmcnt))
                    commit_weight_tile(nxt, stage_regs[nxt])
                    nk = step + 2
                    nj, nkk = nk // nBK, nk % nBK
                    if const_expr(nj < nBN):
                        load_weight_tile(nj, nkk, stage_regs[cur])
                    if const_expr(k == nBK - 1 and block_n + 1 < nBN):
                        load_w_scale(block_n + 1)
                    fx.rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                    fx.gemm(mm, fragC, frag_weight, frag_act_slots[k], fragC)
                epilogue(block_n)

            fx.rocdl.sched_barrier(0)

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
            False,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch_prefill_8x1
