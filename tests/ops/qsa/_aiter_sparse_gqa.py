# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's Triton sparse paged GQA (BF16 or FP8 K/V), vendored for benchmark comparison.

From ROCm/aiter@0de43e66 (PR #7008; the /opt/aiter checkout predates it):
`_qsa_sparse_paged_gqa_kernel` of aiter/ops/triton/_triton_kernels/attention/qsa.py
(unchanged but for the kernel repr) and the single-split path of `qsa_sparse_paged_gqa`
of aiter/ops/triton/attention/qsa.py, with the gfx942 "gqa_fp8"/"prefill" tile of
configs/gfx942/triton/attention/qsa/DEFAULT.json inlined. `out` may be passed.
"""

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fnuz
# configs/gfx942/triton/attention/qsa/DEFAULT.json: gqa_fp8.prefill
GQA_FP8_PREFILL_TILE = dict(BLOCK_N=32, num_warps=2, num_stages=2, waves_per_eu=0)


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_scale_page,
    stride_scale_token,
    stride_scale_head,
    stride_indices_token,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_tokens,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_SPLITS: tl.constexpr = 1,
    SUB_GROUP: tl.constexpr = 0,
    KV_FP8: tl.constexpr = False,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged K/V, unquantized
    or FP8.

    FP8 K/V have one FP32 scale per (token, KV head). The key scale multiplies
    the scores. The value scale multiplies the probabilities before PV.
    """
    token = tl.program_id(0)
    part = tl.program_id(2)
    ACTIVE_SUB: tl.constexpr = SUB_GROUP if SUB_GROUP > 0 else GROUP_SIZE
    NUM_HEAD_GROUPS: tl.constexpr = (GROUP_SIZE + ACTIVE_SUB - 1) // ACTIVE_SUB
    TAIL: tl.constexpr = GROUP_SIZE % ACTIVE_SUB
    kv_head = tl.program_id(1) // NUM_HEAD_GROUPS
    head_group = tl.program_id(1) % NUM_HEAD_GROUPS
    if TAIL != 0:
        valid_sub = tl.where(head_group < NUM_HEAD_GROUPS - 1, ACTIVE_SUB, TAIL)
    else:
        valid_sub = ACTIVE_SUB
    if KV_SPLITS > 1:  # noqa: SIM102
        if tl.program_id(1) == 0 and part == 0:
            tl.store(widths_ptr + token, TOPK)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_q_head = kv_head * GROUP_SIZE + head_group * ACTIVE_SUB
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < valid_sub) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    if part * partition_size >= TOPK:
        return
    for start in tl.range(
        part * partition_size, tl.minimum((part + 1) * partition_size, TOPK), BLOCK_N
    ):
        columns = start + column_offsets
        logical_token = tl.load(
            logical_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        safe_logical_token = tl.maximum(logical_token, 0)
        logical_page = safe_logical_token // PAGE_SIZE
        page_offset = safe_logical_token % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & request_valid
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[None, :] * stride_k_page
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
            cache_modifier=".cg",
        )

        qk = tl.dot(query, keys.to(query.dtype))
        if KV_FP8:
            scale_offset = (
                safe_physical_page * stride_scale_page
                + page_offset * stride_scale_token
                + kv_head * stride_scale_head
            )
            qk *= tl.load(k_scale_ptr + scale_offset, mask=valid, other=0.0)[None, :]
            value_scale = tl.load(v_scale_ptr + scale_offset, mask=valid, other=0.0)
        scores = tl.where(valid[None, :], qk, -1.0e20)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        weights = probabilities * value_scale[None, :] if KV_FP8 else probabilities
        accumulator = tl.dot(
            weights.to(query.dtype),
            values.to(query.dtype),
            acc=accumulator * alpha[:, None],
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max

    if KV_SPLITS > 1:
        partial_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + head_offsets
        ) * KV_SPLITS + part
        tl.store(
            partial_max_ptr + partial_offset, running_max, head_offsets < valid_sub
        )
        tl.store(
            partial_sum_ptr + partial_offset, running_sum, head_offsets < valid_sub
        )
        output_ptr += part * HEAD_DIM
        output = accumulator
    else:
        output = tl.where(
            running_sum[:, None] > 0,
            accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
            0.0,
        )
    tl.store(
        output_ptr
        + token * stride_output_token
        + (first_q_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(token < num_tokens)
        & (head_offsets[:, None] < valid_sub)
        & (dim_offsets[None, :] < HEAD_DIM),
    )


def _prev_pow2(n):
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _table_width_bucket(width):
    """Page-table width as seen by the kernels, rounded up to a power of two."""
    return 1 << (max(int(width), 1) - 1).bit_length()


def _kv_splits_heuristic(T, kv_heads, topk, num_cu, target_wg_per_cu=4.0, max_kv_splits=64):
    if topk < 512:
        return 1
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    base_ctas = max(1, T * kv_heads)
    if base_ctas >= target_wg:
        return 1
    return _prev_pow2(min(target_wg // base_ctas, max_kv_splits))


def _kernel_config(T, kv_heads, kv_splits, group_size, num_cu):
    """Pick (BLOCK_N, num_warps, num_stages, waves_per_eu, sub_group)."""
    sub_group = group_size
    if group_size > 16:
        sub_group = 8
    num_head_groups = (group_size + sub_group - 1) // sub_group
    grid_size = T * kv_heads * num_head_groups * kv_splits
    if grid_size <= num_cu:
        return 64, 4, 1, 0, group_size
    if grid_size >= num_cu * 8:
        return 32, 2, 1, 0, sub_group
    return 32, 4, 1, 3, sub_group


def sparse_paged_gqa(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_request,
    softmax_scale=None,
    k_scale=None,
    v_scale=None,
    out=None,
):
    """qsa_sparse_paged_gqa for prefill (num_decode_requests=None): GQA of q[T,H,D] over
    the request-local tokens logical_indices[T,W] (-1 = padding) of paged K/V
    [pages, page, HK, D]. FP8 K/V take FP32 k_scale / v_scale [pages, page, HK]."""
    kv_fp8 = k_cache.dtype == FP8
    if kv_fp8 != (k_scale is not None) or kv_fp8 != (v_scale is not None):
        raise ValueError("FP8 K/V need k_scale and v_scale; BF16 K/V take none")
    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out
    num_cu = torch.cuda.get_device_properties(q.device).multi_processor_count
    group_size = q.shape[1] // k_cache.shape[2]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    kv_splits = _kv_splits_heuristic(q.shape[0], k_cache.shape[2], logical_indices.shape[1], num_cu)
    if kv_splits != 1:
        raise NotImplementedError("the KV-split reduction is not vendored")
    block_n, num_warps, num_stages, waves_per_eu, sub_group = _kernel_config(
        q.shape[0], k_cache.shape[2], kv_splits, group_size, num_cu
    )
    if kv_fp8:
        tile = GQA_FP8_PREFILL_TILE
        block_n, num_warps = tile["BLOCK_N"], tile["num_warps"]
        num_stages, waves_per_eu = tile["num_stages"], tile["waves_per_eu"]
    num_head_groups = (group_size + sub_group - 1) // sub_group
    block_m = max(16, triton.next_power_of_2(sub_group))
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2] * num_head_groups, kv_splits)](
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        logical_indices,
        block_table,
        token_to_request,
        out,
        out,
        out,
        None,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        *(k_scale.stride() if kv_fp8 else (0, 0, 0)),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(-1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=_table_width_bucket(block_table.shape[1]),
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        KV_SPLITS=kv_splits,
        SUB_GROUP=sub_group if sub_group < group_size else 0,
        KV_FP8=kv_fp8,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    return out
