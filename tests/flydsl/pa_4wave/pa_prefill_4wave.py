import functools
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import as_ir_value
from flydsl.runtime.device import get_rocm_arch

LOG2E = 1.4426950408889634
_SCHED_MASK_DS_WRITE = 0x200
_SCHED_MASK_TRANS = 0x400
_EXP_DSWR_SYNC_ID = 1


def _tensor_signature(tensor):
    return (
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _maxnumf(lhs, rhs):
    return type(lhs)(arith.maxnumf(arith.unwrap(lhs), arith.unwrap(rhs)))


def _exp2_f32(value):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.call_intrinsic(F32Type.get(), "llvm.amdgcn.exp2.f32", [arith.unwrap(value)], [], []))


def _exp2_vec_f32(values):
    from flydsl._mlir.dialects import vector
    from flydsl._mlir.ir import F32Type, VectorType

    raw = arith.unwrap(values)
    f32 = F32Type.get()
    result = [llvm.call_intrinsic(
        f32, "llvm.amdgcn.exp2.f32",
        [vector.extract(raw, static_position=[index], dynamic_position=[])], [], [],
    ) for index in range(raw.type.shape[0])]
    return fx.Vector(vector.from_elements(VectorType.get([len(result)], f32), result))


def _fma_f32(lhs, rhs, acc, negate_acc=False):
    from flydsl._mlir.ir import F32Type

    instruction = "v_fma_f32 $0, $1, $2, -$3" if negate_acc else "v_fma_f32 $0, $1, $2, $3"
    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [arith.unwrap(lhs), arith.unwrap(rhs), arith.unwrap(acc)],
        instruction, "=v,v,v,v", has_side_effects=False,
    ))


def _sub_f32(lhs, rhs):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [arith.unwrap(lhs), arith.unwrap(rhs)],
        "v_sub_f32 $0, $1, $2", "=v,v,v", has_side_effects=False,
    ))


def _add_f32(lhs, rhs):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [arith.unwrap(lhs), arith.unwrap(rhs)],
        "v_add_f32 $0, $1, $2", "=v,v,v", has_side_effects=False,
    ))


def _max_f32(lhs, rhs):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [arith.unwrap(lhs), arith.unwrap(rhs)],
        "v_max_f32 $0, $1, $2", "=v,v,v", has_side_effects=False,
    ))


def _max3_f32(lhs, middle, rhs):
    from flydsl._mlir.ir import F32Type

    return fx.Float32(llvm.inline_asm(
        F32Type.get(), [
            arith.unwrap(lhs), arith.unwrap(middle), arith.unwrap(rhs)
        ],
        "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v", has_side_effects=False,
    ))


def _reduce_max_f32(values):
    assert values.numel == 16
    partials = [
        _max3_f32(
            values[3 * index], values[3 * index + 1], values[3 * index + 2]
        )
        for index in range_constexpr(5)
    ]
    lower = _max3_f32(partials[0], partials[1], partials[2])
    upper = _max3_f32(partials[3], partials[4], values[15])
    return _max_f32(lower, upper)


def _cross_lane_max32(value):
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    value_bits = as_ir_value(value).bitcast(fx.Int32.ir_type)
    swapped = rocdl.permlane32_swap(
        pair_type, value_bits, value_bits, False, True
    )
    lower = llvm.extractvalue(fx.Int32.ir_type, swapped, [0]).bitcast(
        fx.Float32.ir_type
    )
    upper = llvm.extractvalue(fx.Int32.ir_type, swapped, [1]).bitcast(
        fx.Float32.ir_type
    )
    return _max_f32(fx.Float32(lower), fx.Float32(upper))


def _reduce_add_f32(values):
    assert values.numel == 16
    pair_sums = [
        _add_f32(values[2 * index], values[2 * index + 1])
        for index in range_constexpr(8)
    ]
    quad_sums = [
        _add_f32(pair_sums[2 * index], pair_sums[2 * index + 1])
        for index in range_constexpr(4)
    ]
    octet_sums = [
        _add_f32(quad_sums[2 * index], quad_sums[2 * index + 1])
        for index in range_constexpr(2)
    ]
    return _add_f32(octet_sums[0], octet_sums[1])


def _read_hw_wave_slot():
    return fx.Int32(llvm.inline_asm(
        fx.Int32.ir_type, [], "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 0, 4)",
        "=s", has_side_effects=True,
    ))


def _set_hw_slot_priority(wave_slot, slot0_priority, slot1_priority):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"), [arith.unwrap(wave_slot)],
        (
            "s_cmp_eq_u32 $0, 0\n\t"
            "s_cbranch_scc0 1f\n\t"
            f"s_setprio {slot0_priority}\n\t"
            "s_branch 2f\n\t"
            "1:\n\t"
            f"s_setprio {slot1_priority}\n\t"
            "2:"
        ),
        "s", has_side_effects=True,
    )


def _cvt_f32_to_bf16(fragment):
    """Use native packed RNE conversion on gfx950 and legacy packing elsewhere."""
    result = fx.make_fragment_like(fragment, dtype=fx.BFloat16)
    if const_expr(get_rocm_arch().startswith("gfx950")):
        result.store(fragment.load().to(fx.BFloat16))
        return result

    result.store(((fragment.load().bitcast(fx.Uint32) + fx.Uint32(0x8000)) >> 16).to(fx.Uint16).bitcast(fx.BFloat16))
    return result


def _pack_f32x4_to_fp8(values):
    packed = fx.Int32(0)
    packed = fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[0], values[1], packed, False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[2], values[3], packed, True))


def _pack_probability_fp8(probability, start, fp8_dtype):
    values = fx.Vector.from_elements(
        [probability[start + offset] for offset in range_constexpr(4)], fx.Float32
    )
    return fx.Vector.from_elements([_pack_f32x4_to_fp8(values)], fx.Int32).bitcast(
        fp8_dtype
    )


@flyc.jit
def _rescale_accumulator_if_needed(output_accumulator, correction, max_advances):
    if max_advances:
        output_accumulator.store(output_accumulator.load() * correction)


def _store_fp8_probability(score_fragment, probability_operand, fp8_dtype):
    probability = score_fragment.load()
    for k_group in range_constexpr(2):
        start = k_group * 8
        probability_lo = _pack_probability_fp8(probability, start, fp8_dtype)
        probability_hi = _pack_probability_fp8(
            probability, start + 4, fp8_dtype
        )
        probability_operand[None, 0, k_group].store(
            probability_lo.shuffle(probability_hi, list(range(8)))
        )


def _store_bf16_probability(score_fragment, probability_storage):
    probability_storage.store(_cvt_f32_to_bf16(score_fragment).load())


def _make_fp8_epilogue_tid(tid, running_sum):
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [arith.unwrap(tid), arith.unwrap(running_sum)],
            "v_and_or_b32 $0, $2, 0, $1",
            "=v,v,v",
            has_side_effects=False,
        )
    )


def _schedule_qk_bf16_d128(num_v_loads, head_dim_qk, mma_k):
    for _ in range_constexpr(num_v_loads):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(1)
    rocdl.sched_mfma(head_dim_qk // mma_k - num_v_loads)


def _schedule_qk_bf16_d192(num_v_loads, mma_k):
    if mma_k == 8:
        for _ in range_constexpr(num_v_loads):
            rocdl.sched_vmem(1)
            rocdl.sched_mfma(3)
    else:
        for _ in range_constexpr(num_v_loads):
            rocdl.sched_vmem(1)
            rocdl.sched_mfma(1)
        rocdl.sched_mfma(192 // mma_k - num_v_loads)


def _schedule_qk_fp8(num_v_loads):
    for _ in range_constexpr(num_v_loads):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(2)


def _schedule_pv_bf16(
    num_k_prefetches, num_k_reads, head_dim_v, mma_k, direct_k_lds=False
):
    if const_expr(not direct_k_lds):
        for _ in range_constexpr(num_k_prefetches):
            rocdl.sched_vmem(1)
            rocdl.sched_dswr(1)
    leading_mfma = 3 if mma_k == 8 else 2
    rocdl.sched_mfma(leading_mfma)
    for _ in range_constexpr(num_k_reads):
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(1)
    remaining_mfma = head_dim_v // mma_k - num_k_reads - leading_mfma
    if remaining_mfma > 0:
        rocdl.sched_mfma(remaining_mfma)
    if const_expr(direct_k_lds):
        for _ in range_constexpr(num_k_prefetches):
            rocdl.sched_vmem(1)


def _schedule_pv_fp8(num_k_reads):
    rocdl.sched_vmem(1)
    rocdl.sched_dswr(1)
    rocdl.sched_mfma(7)
    rocdl.sched_vmem(1)
    rocdl.sched_mfma(3)
    rocdl.sched_dswr(1)
    rocdl.sched_mfma(4)
    for _ in range_constexpr(num_k_reads):
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(1)


def _s_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    rocdl.s_waitcnt(vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14))


def _schedule_fence():
    rocdl.sched_barrier(0)


def _schedule_ds_write(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_DS_WRITE, count, sync_id)


def _schedule_trans(count, sync_id=_EXP_DSWR_SYNC_ID):
    rocdl.sched_group_barrier(_SCHED_MASK_TRANS, count, sync_id)


def _recast_tensor(tensor, dtype):
    pointer_type = fx.PointerType.get(dtype.ir_type, tensor.memspace, dtype.width // 8)
    iterator = fx.recast_iter(pointer_type, fx.get_iter(tensor))
    layout = fx.recast_layout(tensor.layout, tensor.dtype.width, dtype.width)
    return fx.make_view(iterator, layout)


def _prepare_paged_v_tile(v_tile, permute_bf16_tokens: fx.Constexpr[bool]):
    if const_expr(v_tile.dtype == fx.BFloat16 and permute_bf16_tokens):
        token_permutation = fx.make_layout((8, (2, 2)), (1, (16, 8)))
        v_tile = fx.composition(v_tile, fx.make_tile(None, token_permutation, None))
    return fx.rocdl.make_buffer_tensor(v_tile, max_size=False)


def _compile_hints_for_dtype(dtype):
    return (
        {"fast_fp_math": True}
        if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        else {}
    )


@flyc.jit
def _online_softmax(
    score_fragment,
    output_accumulator,
    qk_scale_log2,
    running_max,
    running_sum,
    query_tile_start,
    kv_block_index,
    kv_len,
    query_sequence_length,
    all_kv_valid: fx.Constexpr[bool],
    is_causal: fx.Constexpr[bool],
    split_score_scaling: fx.Constexpr[bool],
    defer_output_rescale: fx.Constexpr[bool],
    interleaved_score_columns: fx.Constexpr[bool],
    window_left: fx.Constexpr[int] = -1,
):
    if const_expr(not all_kv_valid):
        if const_expr(window_left >= 0):
            # Rematerialize the SWA lane coordinate at its mask consumer. If it
            # is derived from the kernel-entry tid, LLVM keeps it live across
            # every persistent work item and spills the final VGPR.
            mask_tid = fx.Int32(
                llvm.inline_asm(
                    fx.Int32.ir_type,
                    [as_ir_value(fx.thread_idx.x)],
                    "v_mov_b32 $0, $1",
                    "=v,v",
                    has_side_effects=True,
                )
            )
            lane_id = mask_tid & 63
        else:
            lane_id = fx.thread_idx.x & 63
        column_base = (lane_id < 32).select(fx.Int32(0), fx.Int32(16))
        lane_column_group = (lane_id < 32).select(fx.Int32(0), fx.Int32(8))
        block_base = fx.Int32(kv_block_index * 32)
        if const_expr(is_causal):
            if const_expr(window_left >= 0):
                wave_id = mask_tid // 64
                query_row = mask_tid & 31
            else:
                wave_id = fx.thread_idx.x // 64
                query_row = fx.thread_idx.x & 31
            query_position = query_tile_start + wave_id * 32 + query_row
            causal_limit = kv_len - query_sequence_length + query_position
            for index in range_constexpr(16):
                if const_expr(interleaved_score_columns):
                    column = lane_column_group + fx.Int32((index // 8) * 16 + index % 8)
                else:
                    column = column_base + fx.Int32(index)
                kv_position = block_base + column
                outside_window = kv_position > causal_limit
                if const_expr(window_left >= 0):
                    outside_window = outside_window | (
                        kv_position < causal_limit - fx.Int32(window_left)
                    )
                if outside_window:
                    score_fragment[index, 0, 0] = float("-inf")
        else:
            for index in range_constexpr(16):
                if const_expr(interleaved_score_columns):
                    column = lane_column_group + fx.Int32((index // 8) * 16 + index % 8)
                else:
                    column = column_base + fx.Int32(index)
                if block_base + column >= kv_len:
                    score_fragment[index, 0, 0] = float("-inf")

    score = score_fragment.load()
    # Q/K scale is positive, so the cross-lane max can run before scaling.
    row_max = _reduce_max_f32(score)
    if const_expr(split_score_scaling):
        _schedule_fence()
    if const_expr(split_score_scaling):
        scaled_values = [
            score[index] * qk_scale_log2 for index in range_constexpr(11)
        ]
        _schedule_fence()
        row_max = _cross_lane_max32(row_max)
        scaled_values.extend(
            [
                score[index] * qk_scale_log2 for index in range_constexpr(11, score.numel)
            ]
        )
        scaled_score = fx.Vector.from_elements(scaled_values, fx.Float32)
    else:
        row_max = _cross_lane_max32(row_max)
    if const_expr(split_score_scaling):
        row_max = row_max * qk_scale_log2
    else:
        row_max = _fma_f32(row_max, qk_scale_log2, fx.Float32(0.0))

    # AITER-style lazy max avoids rescaling until the row max advances by 8.
    max_advances = row_max > running_max + fx.Float32(8.0)
    updated_max = max_advances.select(row_max, running_max)
    # Deriving correction from the selected max avoids a second cndmask.
    correction = _exp2_f32(running_max - updated_max)

    if const_expr(split_score_scaling):
        shifted_score = fx.Vector.from_elements(
            [
                _sub_f32(scaled_score[index], updated_max)
                for index in range_constexpr(scaled_score.numel)
            ],
            fx.Float32,
        )
    else:
        shifted_score = fx.Vector.from_elements(
            [
                _fma_f32(
                    score[index],
                    qk_scale_log2,
                    updated_max,
                    negate_acc=True,
                )
                for index in range_constexpr(score.numel)
            ],
            fx.Float32,
        )
    probability = _exp2_vec_f32(shifted_score)
    tile_sum = (
        _reduce_add_f32(probability)
        if const_expr(split_score_scaling)
        else probability.reduce("add")
    )
    updated_sum = _fma_f32(running_sum, correction, tile_sum)
    score_fragment.store(probability)

    if const_expr(not defer_output_rescale):
        _rescale_accumulator_if_needed(
            output_accumulator, correction, max_advances
        )
    return updated_max, updated_sum, correction, max_advances


@functools.cache
def MHA(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
    key_layout="vectorized",
    window_left=-1,
    has_sink=False,
    force_dynamic_schedule=False,
):
    assert head_dim_qk in (128, 192)
    assert head_dim_v == 128
    assert page_size in (32, 64, 128)
    assert num_qo_heads % num_kv_heads == 0
    assert key_layout in ("vectorized", "linear")
    assert window_left == -1 or window_left >= 0
    if key_layout == "linear":
        assert page_size == 32
        assert head_dim_qk == head_dim_v == 128
    if window_left >= 0:
        assert is_causal
        assert key_layout == "vectorized"
        assert has_sink
        assert page_size == 64
    else:
        assert not has_sink

    block_m = 128
    block_n = 32
    num_blocks_per_page = page_size // block_n
    num_threads = 256
    causal_tile_step = 251
    causal_tile_offset = 251
    qk_scale_log2_base = float(LOG2E / (head_dim_qk**0.5))
    is_gfx950 = "gfx950" in torch.cuda.get_device_properties().gcnArchName

    @flyc.jit
    def attention_pipeline(
        q_tile,
        k_tile,
        k_dma_resource,
        v_tile,
        o_tile,
        query_pos0,
        query_len,
        kv_len,
        full_qo_len,
        kv_page_table,
        num_kv_pages,
        first_page,
        kv_sequence_start,
        kv_head,
        qk_scale_log2,
        v_scale,
        sink_logit,
        shared_allocator,
    ):
        tid = fx.thread_idx.x
        dtype = q_tile.dtype
        is_fp8 = dtype in (fx.Float8E4M3FN, fx.Float8E4M3FNUZ)
        is_bf16 = dtype == fx.BFloat16
        use_direct_k_lds = is_gfx950 and is_bf16
        pv_mfma_k = 16 if is_gfx950 or not is_bf16 else 8
        # gfx950 can consume 64 FP8 reduction elements per QK instruction.
        # PV still reduces over block_n=32, so it retains the K16 atom.
        use_fp8_qk_k64 = (
            is_gfx950
            and dtype == fx.Float8E4M3FN
            and head_dim_qk % 64 == 0
        )
        qk_mfma_k = 64 if use_fp8_qk_k64 else pv_mfma_k
        if const_expr(window_left >= 0):
            num_kv_blocks = num_kv_pages * num_blocks_per_page
        else:
            num_kv_blocks = (kv_len + block_n - 1) // block_n
        first_kv_block = first_page * num_blocks_per_page
        use_hw_slot_priority = is_bf16 and (head_dim_qk == 128 or is_gfx950)
        defer_output_rescale = (
            is_bf16 and head_dim_qk in (128, 192) and num_qo_heads >= 4
        )
        interleave_exp_ds_write = (
            defer_output_rescale and head_dim_qk == 128 and not use_direct_k_lds
        )
        interleave_d192_k_writes = (
            defer_output_rescale and head_dim_qk == 192 and not use_direct_k_lds
        )
        interleaved_score_columns = is_bf16 and pv_mfma_k == 16
        hw_wave_slot = _read_hw_wave_slot() if const_expr(use_hw_slot_priority) else None

        def enter_softmax_stage():
            _schedule_fence()
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 1, 0)
            else:
                rocdl.s_setprio(0)
            _schedule_fence()

        def enter_mma_stage():
            _schedule_fence()
            if const_expr(use_hw_slot_priority):
                _set_hw_slot_priority(hw_wave_slot, 3, 2)
            else:
                rocdl.s_setprio(2)
            _schedule_fence()

        if const_expr(use_fp8_qk_k64):
            qk_mma_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, dtype)
            )
            qk_mma_atom = fx.atom_set_value(
                qk_mma_atom, "scale_a", fx.Int32(0)
            )
            qk_mma_atom = fx.atom_set_value(
                qk_mma_atom, "scale_b", fx.Int32(0)
            )
        else:
            qk_mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(32, 32, qk_mfma_k, dtype)
            )
        pv_mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(32, 32, pv_mfma_k, dtype)
        )
        atom_values = pv_mfma_k // 2
        vector_values = 128 // dtype.width
        packed_atoms = vector_values // atom_values
        wave_layout = fx.make_layout((1, 4, 1), (1, 1, 0))
        if const_expr(use_fp8_qk_k64):
            qk_tiled_mma = fx.make_tiled_mma(
                qk_mma_atom,
                wave_layout,
                (None, None, fx.make_layout((32, 2), (1, 32))),
            )
        elif const_expr(packed_atoms == 1):
            qk_tiled_mma = fx.make_tiled_mma(qk_mma_atom, wave_layout)
        else:
            qk_permutation = fx.make_layout(
                (atom_values, 2, packed_atoms),
                (1, vector_values, atom_values),
            )
            qk_tiled_mma = fx.make_tiled_mma(
                qk_mma_atom,
                wave_layout,
                (None, None, qk_permutation),
            )
        if const_expr(packed_atoms == 1):
            pv_tiled_mma = fx.make_tiled_mma(pv_mma_atom, wave_layout)
        else:
            pv_permutation = fx.make_layout(
                (atom_values, 2, packed_atoms),
                (1, vector_values, atom_values),
            )
            pv_tiled_mma = fx.make_tiled_mma(
                pv_mma_atom,
                wave_layout,
                (None, None, pv_permutation),
            )
        qk_thread_mma = qk_tiled_mma.thr_slice(tid)
        pv_thread_mma = pv_tiled_mma.thr_slice(tid)

        q_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
        q_thread_copy = fx.make_tiled_copy_B(q_copy_atom, qk_tiled_mma).get_slice(tid)
        q_fragment = qk_thread_mma.make_fragment_B(q_tile)
        fx.copy(q_copy_atom, q_thread_copy.partition_S(q_tile), q_thread_copy.retile(q_fragment))

        k_mma_tile = fx.Tensor(fx.make_view(
            fx.get_iter(k_tile), fx.make_layout((block_n, head_dim_qk), (head_dim_qk, 1))
        ))
        v_mma_tile = fx.Tensor(fx.make_view(
            fx.get_iter(v_tile), fx.make_layout((head_dim_v, block_n), (block_n, 1))
        ))
        k_fragment = qk_thread_mma.make_fragment_A(k_mma_tile)
        v_fragment = pv_thread_mma.make_fragment_A(v_mma_tile)
        k_fragment.fill(0)
        v_fragment.fill(0)
        score_fragment = qk_thread_mma.make_fragment_C(
            fx.make_rmem_tensor(fx.make_layout((block_n, block_m), (block_m, 1)), fx.Float32)
        )
        transposed_output_tile = fx.select(o_tile, [1, 0])
        output_accumulator = pv_thread_mma.make_fragment_C(transposed_output_tile)
        if const_expr(is_bf16):
            probability_storage = fx.make_fragment_like(score_fragment, dtype=fx.BFloat16)
            probability_layout = (
                fx.make_layout((8, 1, 2), (1, 0, 8))
                if pv_mfma_k == 16
                else fx.make_layout((4, 1, (2, 2)), (1, 0, (4, 8)))
            )
            probability_operand = fx.make_view(
                fx.get_iter(probability_storage), probability_layout
            )
        else:
            probability_operand = pv_thread_mma.make_fragment_B(fx.make_rmem_tensor(
                fx.make_layout((block_m, block_n), (block_n, 1)), dtype
            ))
        score_fragment.fill(0)
        probability_operand.fill(0)

        k_lds_stride = head_dim_qk + (8 if is_bf16 else 16)
        use_grouped_direct_k_lds = use_direct_k_lds and key_layout == "vectorized"
        # One 64-lane DMA copies two 32-row vector groups. The 16-byte gap
        # after each pair rotates subsequent LDS read banks without breaking
        # the vectorized cache's contiguous source order.
        k_lds_group_stride = block_n * vector_values
        num_k_group_pairs = head_dim_qk // (2 * vector_values)
        k_lds_group_pair_stride = 2 * k_lds_group_stride + vector_values
        k_lds_stage_elements = (
            num_k_group_pairs * k_lds_group_pair_stride
            if use_grouped_direct_k_lds
            else block_n * k_lds_stride
        )

        @fx.struct
        class KStorage:
            k_lds0: fx.Array[dtype, k_lds_stage_elements, 16]
            k_lds1: fx.Array[dtype, k_lds_stage_elements, 16]

        @fx.union
        class SharedStorage:
            k: KStorage
            o_lds: fx.Array[fx.BFloat16, block_m * (head_dim_v // 2), 16]

        shared = shared_allocator.allocate(SharedStorage)
        k_lds_slot_layout = (
            fx.make_layout(
                (block_n, (vector_values, (2, num_k_group_pairs))),
                (
                    vector_values,
                    (1, (k_lds_group_stride, k_lds_group_pair_stride)),
                ),
            )
            if use_grouped_direct_k_lds
            else fx.make_layout(
                (block_n, head_dim_qk),
                (k_lds_stride, 1),
            )
        )
        k_lds_storage = [
            fx.make_view(shared.k.k_lds0.peek().ptr, k_lds_slot_layout),
            fx.make_view(shared.k.k_lds1.peek().ptr, k_lds_slot_layout),
        ]
        if const_expr(is_bf16):
            k_row_permutation = (
                fx.make_layout((4, 2, 2, 2), (1, 8, 4, 16))
                if pv_mfma_k == 16
                else fx.make_layout((4, 2, 4), (1, 16, 4))
            )
            k_lds = [
                fx.composition(
                    k_lds_storage[slot],
                    fx.make_tile(k_row_permutation, None),
                )
                for slot in range_constexpr(2)
            ]
        else:
            k_lds = k_lds_storage
        output_swizzle = fx.SwizzleType.get(3, 3, 3)
        o_lds_store = shared.o_lds.peek().view(fx.make_composed_layout(
            fx.static(output_swizzle), fx.make_ordered_layout((head_dim_v // 2, block_m), (0, 1))
        ))
        o_lds_read = shared.o_lds.peek().view(fx.make_composed_layout(
            fx.static(output_swizzle), fx.make_ordered_layout((block_m, head_dim_v // 2), (1, 0))
        ))

        if const_expr(is_bf16):
            num_k_copies = head_dim_qk // 64
            if const_expr(use_direct_k_lds):
                lane_id = tid & 63
                wave_id = tid // 64
                linear_vectors_per_row = head_dim_qk // vector_values
                linear_padded_vectors_per_row = linear_vectors_per_row + 1
                linear_rows_per_dma = 64 // linear_padded_vectors_per_row
                linear_dma_calls_per_wave = (
                    8 + linear_rows_per_dma - 1
                ) // linear_rows_per_dma
                def prefetch_k_bf16_direct(
                    logical_block_id, physical_page_id, page_block_index, lds_slot
                ):
                    logical_block_id = fx.Int32(arith.minsi(
                        arith.unwrap(fx.Int32(logical_block_id)),
                        arith.unwrap(num_kv_blocks - 1),
                    ))
                    page_token_offset = fx.Int32(page_block_index) * fx.Int32(block_n)
                    if const_expr(use_grouped_direct_k_lds):
                        group_pairs_per_wave = num_k_group_pairs // (num_threads // 64)
                        source_row = lane_id & (block_n - 1)
                        for pair_index in range_constexpr(group_pairs_per_wave):
                            group_pair = (
                                wave_id
                                + pair_index * (num_threads // 64)
                            )
                            d_group = group_pair * 2 + lane_id // block_n
                            source_offset = (
                                physical_page_id * page_size * num_kv_heads * head_dim_qk
                                + kv_head * page_size * head_dim_qk
                                + d_group * page_size * vector_values
                                + (page_token_offset + source_row) * vector_values
                            )
                            destination_offset = (
                                group_pair * k_lds_group_pair_stride
                            )
                            destination_view = fx.make_view(
                                fx.get_iter(k_lds_storage[lds_slot])
                                + destination_offset,
                                fx.make_layout(1, 1),
                            )
                            destination = fly_dialect.extract_aligned_pointer_as_index(
                                ir.Type.parse("!llvm.ptr<3>"),
                                arith._to_raw(destination_view),
                            )
                            rocdl.raw_ptr_buffer_load_lds(
                                k_dma_resource,
                                destination,
                                fx.Int32(16),
                                fx.Int32(source_offset * (dtype.width // 8)),
                                fx.Int32(0),
                                fx.Int32(0),
                                fx.Int32(0),
                            )
                    else:
                        row_in_dma = lane_id // linear_padded_vectors_per_row
                        vector_in_row = lane_id % linear_padded_vectors_per_row
                        for dma_index in range_constexpr(linear_dma_calls_per_wave):
                            first_wave_row = dma_index * linear_rows_per_dma
                            rows_this_dma = min(linear_rows_per_dma, 8 - first_wave_row)
                            if (row_in_dma < rows_this_dma) & (vector_in_row < linear_vectors_per_row):
                                source_row = wave_id * 8 + first_wave_row + row_in_dma
                                source_offset = (
                                    (kv_sequence_start + logical_block_id * block_n + source_row)
                                    * num_kv_heads * head_dim_qk
                                    + kv_head * head_dim_qk
                                    + vector_in_row * vector_values
                                )
                                destination_offset = (
                                    (wave_id * 8 + first_wave_row) * k_lds_stride
                                )
                                destination_view = fx.make_view(
                                    fx.get_iter(k_lds_storage[lds_slot])
                                    + destination_offset,
                                    fx.make_layout(1, 1),
                                )
                                destination = fly_dialect.extract_aligned_pointer_as_index(
                                    ir.Type.parse("!llvm.ptr<3>"),
                                    arith._to_raw(destination_view),
                                )
                                rocdl.raw_ptr_buffer_load_lds(
                                    k_dma_resource,
                                    destination,
                                    fx.Int32(16),
                                    fx.Int32(source_offset * (dtype.width // 8)),
                                    fx.Int32(0),
                                    fx.Int32(0),
                                    fx.Int32(0),
                                )

                def store_k_to_lds_bf16_direct(register_slot, lds_slot):
                    return

                prefetch_k = prefetch_k_bf16_direct
                store_k_to_lds = store_k_to_lds_bf16_direct
                num_k_prefetches = (
                    num_k_group_pairs // (num_threads // 64)
                    if use_grouped_direct_k_lds
                    else linear_dma_calls_per_wave
                )
            else:
                k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
                k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), dtype)
                prefetched_k = [
                    fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), dtype),
                    fx.make_rmem_tensor(fx.make_layout((8, num_k_copies), (1, 8)), dtype),
                ]

                def prefetch_k_bf16(logical_block_id, physical_page_id, page_block_index, register_slot):
                    logical_block_id = fx.Int32(arith.minsi(
                        arith.unwrap(fx.Int32(logical_block_id)), arith.unwrap(num_kv_blocks - 1)
                    ))
                    page_token_offset = fx.Int32(page_block_index) * fx.Int32(block_n)
                    for atom_index in range_constexpr(num_k_copies):
                        linear_atom = tid + atom_index * num_threads
                        source_row = linear_atom & (block_n - 1)
                        d_group = linear_atom // block_n
                        if const_expr(key_layout == "linear"):
                            source_offset = (
                                (kv_sequence_start + logical_block_id * block_n + source_row)
                                * num_kv_heads * head_dim_qk
                                + kv_head * head_dim_qk
                                + d_group * vector_values
                            )
                        else:
                            source_offset = (
                                physical_page_id * page_size * num_kv_heads * head_dim_qk
                                + kv_head * page_size * head_dim_qk
                                + d_group * page_size * vector_values
                                + (page_token_offset + source_row) * vector_values
                            )
                        source_offset = fx.Int32(source_offset)
                        source = fx.make_view(
                            fx.get_iter(k_tile) + source_offset, fx.make_layout(8, 1)
                        )
                        fx.copy(k_global_copy_atom, source, prefetched_k[register_slot][None, atom_index])

                def store_k_to_lds_bf16(register_slot, lds_slot):
                    for atom_index in range_constexpr(num_k_copies):
                        linear_atom = tid + atom_index * num_threads
                        source_row = linear_atom & (block_n - 1)
                        d_group = linear_atom // block_n
                        destination_offset = (
                            source_row * k_lds_stride
                            + d_group * vector_values
                        )
                        destination = fx.make_view(
                            fx.get_iter(k_lds_storage[lds_slot])
                            + destination_offset,
                            fx.make_layout(8, 1),
                        )
                        fx.copy(k_lds_store_atom, prefetched_k[register_slot][None, atom_index], destination)

                prefetch_k = prefetch_k_bf16
                store_k_to_lds = store_k_to_lds_bf16
                num_k_prefetches = num_k_copies
        else:
            k_global_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
            k_lds_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Uint32)
            k_tile_u32 = _recast_tensor(k_tile, fx.Uint32)
            num_k_copies = head_dim_qk // 64
            prefetched_k = [
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
                fx.make_rmem_tensor(fx.make_layout((2, num_k_copies), (1, 2)), fx.Uint32),
            ]
            k_row = tid // 8
            k_chunk_in_group = tid & 7
            k_source_row = (k_row & 3) + ((k_row // 4) & 1) * 16 + (k_row // 8) * 4

            def prefetch_k_fp8(logical_block_id, physical_page_id, page_block_index, register_slot):
                prefetched_k[register_slot].fill(0)
                page_token_offset = fx.Int32(page_block_index) * fx.Int32(block_n)
                for atom_index in range_constexpr(num_k_copies):
                    chunk = k_chunk_in_group + atom_index * 8
                    d_group = chunk // 2
                    d_half = chunk & 1
                    source_offset = (
                        physical_page_id * page_size * num_kv_heads * head_dim_qk
                        + kv_head * page_size * head_dim_qk
                        + d_group * page_size * 16
                        + (page_token_offset + k_source_row) * 16
                        + d_half * 8
                    )
                    source_offset = fx.Int32(source_offset)
                    source = fx.make_view(
                        fx.get_iter(k_tile_u32) + source_offset // 4, fx.make_layout(2, 1)
                    )
                    fx.copy(k_global_copy_atom, source, prefetched_k[register_slot][None, atom_index])

            def store_k_to_lds_fp8(register_slot, lds_slot):
                for atom_index in range_constexpr(num_k_copies):
                    chunk = k_chunk_in_group + atom_index * 8
                    destination_offset = k_row * k_lds_stride + chunk * 8
                    destination = fx.make_view(
                        fx.get_iter(k_lds[lds_slot]) + destination_offset,
                        fx.make_layout(8, 1),
                    )
                    fx.copy(k_lds_store_atom, prefetched_k[register_slot][None, atom_index],
                            _recast_tensor(destination, fx.Uint32))

            prefetch_k = prefetch_k_fp8
            store_k_to_lds = store_k_to_lds_fp8
            num_k_prefetches = num_k_copies

        k_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), dtype)
        k_lds_copy = fx.make_tiled_copy_A(k_lds_copy_atom, qk_tiled_mma).get_slice(tid)

        def partition_k_lds(lds_slot):
            return k_lds_copy.partition_S(k_lds[lds_slot])

        v_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)
        v_copy = fx.make_tiled_copy_A(v_copy_atom, pv_tiled_mma).get_slice(tid)
        num_v_loads = fx.size(v_fragment.shape).get_static_leaf_int * v_fragment.dtype.width // 128
        num_k_fragment_bits = fx.size(k_fragment.shape).get_static_leaf_int * k_fragment.dtype.width

        output_accumulator.fill(0.0)

        def compute_qk():
            if const_expr(use_fp8_qk_k64 or packed_atoms == 1):
                fx.gemm(
                    qk_thread_mma,
                    score_fragment,
                    k_fragment,
                    q_fragment,
                    score_fragment,
                )
            else:
                for k_group in range_constexpr(head_dim_qk // (2 * qk_mfma_k)):
                    for k_atom in range_constexpr(2):
                        accumulator = score_fragment[None, 0, 0]
                        fx.mma_atom_call(
                            qk_mma_atom,
                            accumulator,
                            k_fragment[None, 0, (k_atom, k_group)],
                            q_fragment[None, 0, (k_atom, k_group)],
                            accumulator,
                        )

        def schedule_qk_and_v_loads():
            if const_expr(is_bf16 and head_dim_qk == 128):
                _schedule_qk_bf16_d128(
                    num_v_loads, head_dim_qk, qk_mfma_k
                )
            elif const_expr(is_fp8):
                _schedule_qk_fp8(num_v_loads)
            else:
                _schedule_qk_bf16_d192(num_v_loads, qk_mfma_k)
            rocdl.sched_vmem(100)
            rocdl.sched_mfma(100)

        def schedule_pv_and_next_k():
            num_k_reads = num_k_fragment_bits // 128
            if const_expr(is_bf16):
                _schedule_pv_bf16(
                    num_k_prefetches,
                    num_k_reads,
                    head_dim_v,
                    pv_mfma_k,
                    use_direct_k_lds,
                )
            else:
                _schedule_pv_fp8(num_k_reads)
            _schedule_fence()

        def process_kv_block(
            kv_block_index,
            k_pipeline_slot,
            running_max,
            running_sum,
            current_v_page_id,
            prefetch_k_page_id,
            is_all_kv_valid: fx.Constexpr[bool] = True,
        ):
            current_page_block = fx.Int32(kv_block_index % num_blocks_per_page)
            prefetch_page_block = fx.Int32(
                (kv_block_index + 2) % num_blocks_per_page
            )
            # For page64, the first block's lookahead page is already carried
            # as prefetch_k_page_id, so avoid reloading the page table.
            if const_expr(num_blocks_per_page == 2 and k_pipeline_slot == 0):
                lookahead_page_id = prefetch_k_page_id
            else:
                lookahead_page_id = kv_page_table[
                    (kv_block_index + 3) // num_blocks_per_page
                ]

            score_fragment.fill(0.0)
            compute_qk()
            fx.copy(
                v_copy_atom,
                v_copy.partition_S(
                    v_tile[None, None, current_v_page_id, current_page_block]
                ),
                v_copy.retile(v_fragment),
            )
            schedule_qk_and_v_loads()

            enter_softmax_stage()
            if const_expr(not use_direct_k_lds):
                prefetch_k(
                    kv_block_index + 2,
                    prefetch_k_page_id,
                    prefetch_page_block,
                    k_pipeline_slot ^ 1,
                )

            running_max, running_sum, correction, max_advances = _online_softmax(
                score_fragment, output_accumulator, qk_scale_log2, running_max, running_sum,
                query_pos0, first_kv_block + kv_block_index, kv_len, full_qo_len,
                is_all_kv_valid, is_causal,
                is_fp8,
                defer_output_rescale,
                interleaved_score_columns,
                window_left,
            )

            store_k_to_lds(k_pipeline_slot, k_pipeline_slot ^ 1)
            if const_expr(interleave_exp_ds_write):
                _schedule_trans(16)
                _schedule_ds_write(1)
                _schedule_trans(1)
                _schedule_ds_write(1)
            elif const_expr(interleave_d192_k_writes):
                for _ in range_constexpr(3):
                    _schedule_trans(4)
                    _schedule_ds_write(1)
                _schedule_trans(5)

            if const_expr(defer_output_rescale):
                _rescale_accumulator_if_needed(
                    output_accumulator, correction, max_advances
                )
            if const_expr(is_fp8):
                _store_fp8_probability(
                    score_fragment, probability_operand, dtype
                )
            else:
                _store_bf16_probability(score_fragment, probability_storage)

            enter_mma_stage()

            fx.gemm(
                pv_mma_atom, output_accumulator, v_fragment,
                probability_operand, output_accumulator,
            )

            if const_expr(use_direct_k_lds):
                # The previous iteration's K DMA precedes this iteration's V
                # loads, so leaving only the newer V requests outstanding is
                # sufficient before reading the opposite K field.
                _s_waitcnt(vmcnt=num_v_loads)
                rocdl.s_barrier()
            else:
                gpu.barrier()
            fx.copy(
                k_lds_copy_atom, partition_k_lds(k_pipeline_slot ^ 1),
                k_lds_copy.retile(k_fragment),
            )
            if const_expr(use_direct_k_lds):
                prefetch_k(
                    kv_block_index + 2,
                    prefetch_k_page_id,
                    prefetch_page_block,
                    k_pipeline_slot,
                )
            schedule_pv_and_next_k()

            return running_max, running_sum, lookahead_page_id

        current_max = fx.Float32(float("-inf"))
        running_sum = fx.Float32(0.0)
        if const_expr(has_sink):
            current_max = sink_logit * fx.Float32(LOG2E)
            running_sum = fx.Float32(0.5)
        current_page_id = kv_page_table[0]
        next_page_id = kv_page_table[1 // num_blocks_per_page]
        prefetch_page_id = kv_page_table[2 // num_blocks_per_page]

        prefetch_k(0, current_page_id, 0, 0)
        store_k_to_lds(0, 0)
        if const_expr(use_direct_k_lds):
            _s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
        else:
            prefetch_k(
                1,
                next_page_id,
                1 % num_blocks_per_page,
                0,
            )
            gpu.barrier()
        fx.copy(k_lds_copy_atom, partition_k_lds(0), k_lds_copy.retile(k_fragment))
        if const_expr(use_direct_k_lds):
            _s_waitcnt(lgkmcnt=0)
            prefetch_k(
                1,
                next_page_id,
                1 % num_blocks_per_page,
                1,
            )
        enter_mma_stage()

        if const_expr(window_left >= 0):
            num_fast_path_blocks = fx.Int32(0)
            num_blocks_to_process = num_kv_blocks
        elif const_expr(is_causal):
            causal_base = kv_len - full_qo_len + query_pos0
            num_fully_valid_blocks = (causal_base + 1) // block_n
            num_fast_path_blocks = (num_fully_valid_blocks // 2) * 2
            num_intersecting_blocks = (causal_base + query_len + block_n - 1) // block_n
            num_blocks_to_process = (num_intersecting_blocks < num_kv_blocks).select(
                num_intersecting_blocks, num_kv_blocks
            )
        else:
            num_fast_path_blocks = num_kv_blocks - 2
            if (num_kv_blocks & 1) == 1:
                num_fast_path_blocks = num_kv_blocks - 1
            num_blocks_to_process = num_kv_blocks

        loop_state = [current_max, running_sum, current_page_id, next_page_id, prefetch_page_id]
        for kv_block_index, state in range(0, num_fast_path_blocks, 2, init=loop_state):
            current_max, running_sum, current_page_id, next_page_id, prefetch_page_id = state
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index, 0, current_max, running_sum, current_page_id, prefetch_page_id
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index + 1, 1, current_max, running_sum,
                current_page_id, prefetch_page_id,
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            loop_state = yield [
                current_max, running_sum, current_page_id, next_page_id, prefetch_page_id
            ]

        for kv_block_index, state in range(
            num_fast_path_blocks, num_blocks_to_process, 2, init=loop_state
        ):
            current_max, running_sum, current_page_id, next_page_id, prefetch_page_id = state
            current_max, running_sum, lookahead_page_id = process_kv_block(
                kv_block_index, 0, current_max, running_sum,
                current_page_id, prefetch_page_id, is_all_kv_valid=False,
            )
            current_page_id, next_page_id, prefetch_page_id = (
                next_page_id, prefetch_page_id, lookahead_page_id
            )
            if fx.Int32(kv_block_index + 1) < num_blocks_to_process:
                current_max, running_sum, lookahead_page_id = process_kv_block(
                    kv_block_index + 1, 1, current_max, running_sum,
                    current_page_id, prefetch_page_id,
                    is_all_kv_valid=False,
                )
                current_page_id, next_page_id, prefetch_page_id = (
                    next_page_id, prefetch_page_id, lookahead_page_id
                )
            loop_state = yield [
                current_max, running_sum, current_page_id, next_page_id, prefetch_page_id
            ]

        running_sum = loop_state[1]
        if const_expr(use_direct_k_lds):
            # The final clamped lookahead DMA must finish before the epilogue
            # reuses the K/output LDS union.
            _s_waitcnt(vmcnt=0, lgkmcnt=0)
        denominator = running_sum + running_sum.shuffle_xor(32, 64)
        output_accumulator.store(output_accumulator.load() * (v_scale / denominator))
        output_fragment_bf16 = _cvt_f32_to_bf16(output_accumulator)
        if const_expr(is_fp8):
            epilogue_tid = _make_fp8_epilogue_tid(tid, running_sum)
            cshuffle_store_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            cshuffle_store = fx.make_tiled_copy_C(
                cshuffle_store_atom, pv_tiled_mma
            ).get_slice(epilogue_tid)
            cshuffle_read_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            cshuffle_read = fx.make_tiled_copy_tv(
                cshuffle_read_atom,
                fx.make_layout((32, 8), (8, 1)),
                fx.make_layout((4, 8), (8, 1)),
            ).get_slice(epilogue_tid)
            store_source_halves = fx.logical_divide(
                cshuffle_store.retile(output_fragment_bf16), (None, 2, None)
            )
            store_destination = cshuffle_store.partition_D(o_lds_store)
            read_source = cshuffle_read.partition_S(o_lds_read)
            output_halves = fx.logical_divide(o_tile, (None, head_dim_v // 2))
            output_fragment = fx.make_fragment_like(read_source)
            output_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )

            gpu.barrier()
            for half in range_constexpr(2):
                fx.copy(
                    cshuffle_store_atom,
                    store_source_halves[None, (None, half), None],
                    store_destination,
                )
                gpu.barrier()
                fx.copy(cshuffle_read_atom, read_source, output_fragment)
                # Half 0 must drain before LDS is overwritten. After half 1,
                # static exits and dynamic reaches the ticket barrier.
                if const_expr(half == 0):
                    gpu.barrier()
                fx.copy(
                    output_copy_atom,
                    output_fragment,
                    cshuffle_read.partition_D(
                        output_halves[None, (None, half)]
                    ),
                )
        else:
            # Rebuild the C-shuffle partitions from an opaque, epilogue-local
            # tid. High-level get_slice() captured the entry tid and kept eight
            # LDS address VGPRs live across the persistent loop.
            epilogue_tid = fx.Int32(
                llvm.inline_asm(
                    fx.Int32.ir_type,
                    [as_ir_value(fx.thread_idx.x)],
                    "v_mov_b32 $0, $1",
                    "=v,v",
                    has_side_effects=True,
                )
            )
            cshuffle_store_atom = fx.make_copy_atom(
                fx.UniversalCopy64b(), fx.BFloat16
            )
            cshuffle_store_tiled = fx.make_tiled_copy_C(
                cshuffle_store_atom, pv_tiled_mma
            )
            cshuffle_read_atom = fx.make_copy_atom(
                fx.UniversalCopy128b(), fx.BFloat16
            )
            cshuffle_read_tiled = fx.make_tiled_copy_tv(
                cshuffle_read_atom,
                fx.make_layout((32, 8), (8, 1)),
                fx.make_layout((4, 8), (8, 1)),
            )
            epilogue_thread_coord = fx.make_int_tuple(epilogue_tid)
            store_source_halves = fx.logical_divide(
                fx.tiled_copy_retile(cshuffle_store_tiled, output_fragment_bf16),
                (None, 2, None),
            )
            store_destination = fx.tiled_copy_partition_dst(
                cshuffle_store_tiled, o_lds_store, epilogue_thread_coord
            )
            read_source = fx.tiled_copy_partition_src(
                cshuffle_read_tiled, o_lds_read, epilogue_thread_coord
            )
            output_halves = fx.logical_divide(o_tile, (None, head_dim_v // 2))
            output_fragment = fx.make_fragment_like(read_source)
            output_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), fx.BFloat16
            )

            gpu.barrier()
            for half in range_constexpr(2):
                fx.copy(
                    cshuffle_store_atom,
                    store_source_halves[None, (None, half), None],
                    store_destination,
                )
                gpu.barrier()
                fx.copy(cshuffle_read_atom, read_source, output_fragment)
                # Half 0 must drain before LDS is overwritten. After half 1,
                # static exits and dynamic reaches the ticket barrier.
                if const_expr(half == 0):
                    gpu.barrier()
                output_destination = fx.tiled_copy_partition_dst(
                    cshuffle_read_tiled,
                    output_halves[None, (None, half)],
                    epilogue_thread_coord,
                )
                fx.copy(output_copy_atom, output_fragment, output_destination)

    @flyc.jit
    def process_work_item(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        q_descale,
        kv_last_page_lens,
        sink_ptr,
        output,
        batch_index,
        head_index,
        query_tile_index,
        tid,
        k_scale,
        v_scale,
        shared_allocator,
    ):
        query_pos0 = query_tile_index * block_m
        query_start = cu_seqlens_q[batch_index] + query_pos0
        query_end = fx.Int32(arith.minsi(
            arith.unwrap(query_start + block_m), arith.unwrap(cu_seqlens_q[batch_index + 1])
        ))
        query_len = query_end - query_start
        full_qo_len = cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
        kv_start = kv_indptr[batch_index]
        num_kv_pages = kv_indptr[batch_index + 1] - kv_start
        if const_expr(key_layout == "linear"):
            kv_len = cu_seqlens_k[batch_index + 1] - cu_seqlens_k[batch_index]
        else:
            kv_len = (num_kv_pages - 1) * page_size + kv_last_page_lens[batch_index]

        first_page = fx.Int32(0)
        pages_to_process = num_kv_pages
        if const_expr(window_left >= 0):
            absolute_first_q = kv_len - full_qo_len + query_pos0
            first_key = absolute_first_q - fx.Int32(window_left)
            first_key = (first_key > fx.Int32(0)).select(
                first_key, fx.Int32(0)
            )
            first_page = first_key // page_size
            absolute_last_q_exclusive = absolute_first_q + query_len
            last_page = (
                absolute_last_q_exclusive + (page_size - 1)
            ) // page_size
            last_page = (last_page < num_kv_pages).select(
                last_page, num_kv_pages
            )
            pages_to_process = last_page - first_page

        qo_head = head_index
        kv_head = (qo_head * num_kv_heads) // num_qo_heads

        q_tile = fx.make_view(
            fx.get_iter(q) + query_start * num_qo_heads * head_dim_qk,
            fx.make_ordered_layout((block_m, num_qo_heads, head_dim_qk), (2, 1, 0)),
        )
        q_tile = fx.rocdl.make_buffer_tensor(
            q_tile, max_size=False,
            num_records_bytes=query_len * num_qo_heads * head_dim_qk * (q_tile.dtype.width // 8),
        )[None, qo_head, None]

        q_scale_tile = fx.make_view(
            fx.get_iter(q_descale) + query_start * num_qo_heads,
            fx.make_ordered_layout((block_m, num_qo_heads), (1, 0)),
        )
        q_scale_tile = fx.rocdl.make_buffer_tensor(
            q_scale_tile, max_size=False, num_records_bytes=query_len * num_qo_heads * 4
        )[None, qo_head]
        query_row = (tid // 64) * 32 + (tid & 31)
        qk_scale_log2 = q_scale_tile[query_row] * k_scale * fx.Float32(qk_scale_log2_base)

        o_tile = fx.make_view(
            fx.get_iter(output) + query_start * num_qo_heads * head_dim_v,
            fx.make_ordered_layout((block_m, num_qo_heads, head_dim_v), (2, 1, 0)),
        )
        o_tile = fx.rocdl.make_buffer_tensor(
            o_tile, max_size=False, num_records_bytes=query_len * num_qo_heads * head_dim_v * 2
        )[None, qo_head, None]

        k_tile = fx.rocdl.make_buffer_tensor(k, max_size=False)
        k_dma_resource = rocdl.get_buffer_rsrc(fx.get_iter(k_tile))
        v_tile = v[None, kv_head, None, None, None, None]
        v_tile = fx.group(fx.select(v_tile, (3, 4, 2, 0, 1)), 1, 3)
        v_tile = _prepare_paged_v_tile(v_tile, not is_gfx950)

        kv_page_table = fx.make_view(
            fx.get_iter(kv_page_indices) + kv_start + first_page,
            fx.make_layout(pages_to_process, 1),
        )
        kv_page_table = fx.rocdl.make_buffer_tensor(
            kv_page_table, max_size=False
        )
        sink_logit = fx.Float32(0.0)
        if const_expr(has_sink):
            sink_logit = sink_ptr[qo_head]
        attention_pipeline(
            q_tile, k_tile, k_dma_resource, v_tile, o_tile,
            query_pos0, query_len, kv_len, full_qo_len,
            kv_page_table, pages_to_process, first_page,
            cu_seqlens_k[batch_index], kv_head, qk_scale_log2, v_scale, sink_logit,
            shared_allocator,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel_static(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        sink_ptr: fx.Tensor,
        output: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        shared_allocator = fx.SharedAllocator()
        work_ticket = fx.Int32(fx.block_idx.x)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        if const_expr(is_causal):
            physical_tile = work_ticket // num_qo_heads
            head_index = work_ticket - physical_tile * num_qo_heads
            half_tile = physical_tile // 2
            balanced_work = ((physical_tile & 1) == 0).select(half_tile, works_per_head - 1 - half_tile)
            affine_work = (physical_tile * causal_tile_step + causal_tile_offset) % works_per_head
            query_tile_index = (works_per_head == 256).select(affine_work, balanced_work)
        else:
            head_index = work_ticket // works_per_head
            query_tile_index = work_ticket - head_index * works_per_head
        process_work_item(
            q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
            q_descale, kv_last_page_lens, sink_ptr, output,
            fx.Int32(0), head_index, query_tile_index, tid, k_descale[0], v_descale[0],
            shared_allocator,
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def attention_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        sink_ptr: fx.Tensor,
        output: fx.Tensor,
        work_counter: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        batch_size = fx.size(cu_seqlens_q.shape).to_py_value() - 1
        shared_allocator = fx.SharedAllocator()
        # The dedicated four-byte LDS mailbox is independently aligned from
        # the 16-byte K/O union allocated later by attention_pipeline.
        ticket_mailbox = shared_allocator.allocate(
            fx.Array[fx.Int32, 1, 4]
        ).peek()

        @flyc.jit
        def fetch_work(work_counter, ticket_mailbox, tid):
            if tid == 0:
                address = fx.ptrtoint(fx.get_iter(work_counter))
                llvm_pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(address))
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add, llvm_pointer, as_ir_value(fx.Int32(1)), llvm.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4,
                )
                ticket = fx.Int32(old.result)
                ticket_mailbox[0] = ticket
                _s_waitcnt(lgkmcnt=0)
            # One barrier broadcasts the ticket and also closes the previous
            # work item's C-shuffle lifetime before K reuses the union.
            gpu.barrier()
            ticket = ticket_mailbox[0]
            _s_waitcnt(lgkmcnt=0)
            return ticket

        @flyc.jit
        def finish_work(work_counter, tid):
            # Every workgroup exits exactly once. The final completion resets
            # the cached ticket/completion header for the next stream launch.
            if tid == 0:
                address = fx.ptrtoint(fx.get_iter(work_counter) + 1)
                llvm_pointer = llvm.inttoptr(
                    ir.Type.parse("!llvm.ptr<1>"), as_ir_value(address)
                )
                old = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    llvm_pointer,
                    as_ir_value(fx.Int32(1)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                if fx.Int32(old.result) == fx.Int32(fx.grid_dim.x - 1):
                    work_counter[0] = fx.Int32(fx.grid_dim.x)
                    work_counter[1] = fx.Int32(0)

        @flyc.jit
        def advance_work_ticket(ticket_delta, query_tile_index, head_index, batch_index, works_per_head):
            query_tile_index += ticket_delta
            while (batch_index < batch_size) & (query_tile_index >= works_per_head):
                query_tile_index -= works_per_head
                head_index += 1
                if head_index >= num_qo_heads:
                    head_index = 0
                    batch_index += 1
                    if batch_index < batch_size:
                        works_per_head = (cu_seqlens_q[batch_index + 1] - cu_seqlens_q[batch_index]
                                          + block_m - 1) // block_m
            return query_tile_index, head_index, batch_index, works_per_head

        work_ticket = fx.Int32(fx.block_idx.x)
        batch_index = fx.Int32(0)
        head_index = fx.Int32(0)
        query_tile_index = fx.Int32(0)
        works_per_head = (cu_seqlens_q[1] - cu_seqlens_q[0] + block_m - 1) // block_m
        k_scale = k_descale[0]
        v_scale = v_descale[0]
        query_tile_index, head_index, batch_index, works_per_head = advance_work_ticket(
            work_ticket, query_tile_index, head_index, batch_index, works_per_head
        )

        while batch_index < batch_size:
            process_work_item(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                q_descale, kv_last_page_lens, sink_ptr, output,
                batch_index, head_index, query_tile_index, tid, k_scale, v_scale,
                shared_allocator,
            )

            next_ticket = fetch_work(work_counter, ticket_mailbox, tid)
            ticket_delta = next_ticket - work_ticket
            work_ticket = next_ticket
            query_tile_index, head_index, batch_index, works_per_head = advance_work_ticket(
                ticket_delta, query_tile_index, head_index, batch_index, works_per_head
            )
        finish_work(work_counter, tid)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_page_indices: fx.Tensor,
        q_descale: fx.Tensor,
        k_descale: fx.Tensor,
        v_descale: fx.Tensor,
        kv_last_page_lens: fx.Tensor,
        sink_ptr: fx.Tensor,
        output: fx.Tensor,
        work_counter: fx.Tensor,
        num_workgroups: fx.Int32,
        static_schedule: fx.Constexpr[bool],
        stream: fx.Stream,
    ):
        num_query_tokens = q.shape[0].to_py_value()
        num_physical_pages = v.shape[0].to_py_value()
        vector_size = 128 // k.dtype.width
        q = fx.make_view(
            fx.get_iter(q),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, head_dim_qk), (2, 1, 0)
            ),
        )
        if fx.const_expr(key_layout == "linear"):
            num_kv_tokens = k.shape[0].to_py_value()
            k = fx.make_view(
                fx.get_iter(k),
                fx.make_ordered_layout(
                    (num_kv_tokens, num_kv_heads, head_dim_qk), (2, 1, 0)
                ),
            )
        else:
            k = fx.make_view(
                fx.get_iter(k),
                fx.make_ordered_layout(
                    (
                        num_physical_pages,
                        num_kv_heads,
                        head_dim_qk // vector_size,
                        page_size,
                        vector_size,
                    ),
                    (4, 3, 2, 1, 0),
                ),
            )
        v = fx.make_view(
            fx.get_iter(v),
            fx.make_ordered_layout(
                (
                    num_physical_pages,
                    num_kv_heads,
                    num_blocks_per_page,
                    block_n // vector_size,
                    head_dim_v,
                    vector_size,
                ),
                (5, 4, 3, 2, 1, 0),
            ),
        )
        q_descale = fx.make_view(
            fx.get_iter(q_descale),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, 1), (2, 1, 0)
            ),
        )
        k_descale = fx.make_view(fx.get_iter(k_descale), fx.make_layout(1, 1))
        v_descale = fx.make_view(fx.get_iter(v_descale), fx.make_layout(1, 1))
        output = fx.make_view(
            fx.get_iter(output),
            fx.make_ordered_layout(
                (num_query_tokens, num_qo_heads, head_dim_v), (2, 1, 0)
            ),
        )
        value_attrs = {"passthrough": [["target-features", "-packed-fp32-ops"]]}
        persistent_value_attrs = dict(value_attrs)
        if fx.const_expr(q.dtype == fx.BFloat16 and head_dim_qk == 192):
            persistent_value_attrs["rocdl.waves_per_eu"] = 2
        if static_schedule:
            attention_kernel_static(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, sink_ptr, output,
                value_attrs=value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)
        else:
            attention_kernel(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, sink_ptr, output, work_counter,
                value_attrs=persistent_value_attrs,
            ).launch(grid=(num_workgroups, 1, 1), block=(num_threads, 1, 1), stream=stream)

    def callable(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        q_descale,
        k_descale,
        v_descale,
        kv_last_page_lens,
        out,
        stream=None,
        sink_ptr=None,
    ):
        stream = torch.cuda.current_stream() if stream is None else stream
        assert causal == is_causal
        assert not causal or max_seqlen_k >= max_seqlen_q
        if cu_seqlens_k is None:
            if key_layout != "vectorized":
                raise ValueError(
                    "cu_seqlens_k=None requires key_layout='vectorized'"
                )
            cu_seqlens_k = cu_seqlens_q
        arch = torch.cuda.get_device_properties().gcnArchName
        native_fp8_dtype = (
            torch.float8_e4m3fn
            if "gfx950" in arch
            else torch.float8_e4m3fnuz
        )
        assert q.dtype in (native_fp8_dtype, torch.bfloat16)
        if key_layout == "linear":
            assert q.dtype == torch.bfloat16
            assert head_dim_qk == head_dim_v == 128
        assert k.dtype == q.dtype
        assert v.dtype == q.dtype
        tensors = (q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                   q_descale, k_descale, v_descale, kv_last_page_lens, out)
        assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_seqlens_k.dtype == torch.int32
        assert kv_indptr.dtype == torch.int32
        assert kv_page_indices.dtype == torch.int32
        assert kv_last_page_lens.dtype == torch.int32
        assert q_descale.dtype == k_descale.dtype == v_descale.dtype == torch.float32
        assert out.dtype == torch.bfloat16
        assert q.shape[1:] == (num_qo_heads, head_dim_qk)
        num_query_tokens = q.shape[0]
        assert q_descale.shape == (num_query_tokens, num_qo_heads, 1)
        assert out.shape == (num_query_tokens, num_qo_heads, head_dim_v)
        vector_size = 16 // q.element_size()
        num_physical_pages = v.shape[0]
        if key_layout == "linear":
            assert k.shape[1:] == (num_kv_heads, head_dim_qk)
            assert k.shape[0] == int(cu_seqlens_k[-1].item())
        else:
            assert k.shape == (
                num_physical_pages,
                num_kv_heads,
                head_dim_qk // vector_size,
                page_size,
                vector_size,
            )
        assert v.shape == (
            num_physical_pages, num_kv_heads,
            page_size // vector_size, head_dim_v, vector_size,
        )
        assert k_descale.numel() == 1
        assert v_descale.numel() == 1
        assert cu_seqlens_q.ndim == cu_seqlens_k.ndim == kv_indptr.ndim == kv_page_indices.ndim == 1
        assert kv_last_page_lens.ndim == 1
        assert cu_seqlens_q.shape == cu_seqlens_k.shape == kv_indptr.shape
        assert kv_last_page_lens.shape[0] == cu_seqlens_q.shape[0] - 1
        if has_sink:
            assert sink_ptr is not None
            assert sink_ptr.shape == (num_qo_heads,)
            assert sink_ptr.dtype == torch.float32
            assert sink_ptr.device == q.device
            assert sink_ptr.is_contiguous()
        elif sink_ptr is None:
            sink_ptr = q_descale
        assert k.numel() * k.element_size() <= 2**31 - 1
        assert any(supported_arch in arch for supported_arch in ("gfx942", "gfx950"))
        batch_size = cu_seqlens_q.shape[0] - 1
        static_schedule = batch_size == 1 and not force_dynamic_schedule
        if static_schedule:
            works_per_head = (num_query_tokens + block_m - 1) // block_m
            num_workgroups = num_qo_heads * works_per_head
        else:
            multiprocessor_count = torch.cuda.get_device_properties().multi_processor_count
            # A 1/2/3/4 WG-per-CU sweep selected two for both B=1 and B=4:
            # one under-fills latency hiding, while three or four add agents
            # without increasing residency at this LDS/VGPR footprint.
            num_workgroups = multiprocessor_count * 2
        if static_schedule:
            work_counter = getattr(launch, "_static_work_counter", None)
            if work_counter is None:
                work_counter = torch.empty(1, device="cuda", dtype=torch.int32)
                launch._static_work_counter = work_counter
        else:
            # Counter state is stream-local because launches on different
            # streams may overlap. The device-side finalizer makes reuse free
            # of per-call allocation, fill, and seed-copy dispatches.
            work_counter_cache = getattr(launch, "_work_counter_cache", {})
            work_counter_key = (
                torch.cuda.current_device(),
                stream.cuda_stream,
                num_workgroups,
            )
            work_counter = work_counter_cache.get(work_counter_key)
            if work_counter is None:
                with torch.cuda.stream(stream):
                    work_counter = torch.zeros(
                        2, device="cuda", dtype=torch.int32
                    )
                    work_counter[0] = num_workgroups
                work_counter_cache[work_counter_key] = work_counter
                launch._work_counter_cache = work_counter_cache

        compiled_cache = getattr(launch, "_compiled", {})
        cache_key = (
            static_schedule,
            num_workgroups,
            torch.cuda.current_device(),
            torch.cuda.get_device_properties().gcnArchName,
            *(_tensor_signature(tensor) for tensor in (
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
                q_descale, k_descale, v_descale, kv_last_page_lens, sink_ptr, out,
                work_counter,
            )),
        )
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            saved_compile_hints = launch.compile_hints
            try:
                compile_hints = _compile_hints_for_dtype(q.dtype)
                launch.compile_hints = {**saved_compile_hints, **compile_hints}
                compiled = flyc.compile(
                    launch, q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                    kv_page_indices, q_descale, k_descale, v_descale,
                    kv_last_page_lens, sink_ptr, out, work_counter, num_workgroups,
                    static_schedule, stream,
                )
            finally:
                launch.compile_hints = saved_compile_hints
            compiled_cache[cache_key] = compiled
            launch._compiled = compiled_cache
        else:
            compiled(
                q, k, v, cu_seqlens_q, cu_seqlens_k, kv_indptr,
                kv_page_indices, q_descale, k_descale, v_descale,
                kv_last_page_lens, sink_ptr, out, work_counter, num_workgroups,
                static_schedule, stream,
            )
        return out

    return callable
