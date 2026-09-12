# SPDX-License-Identifier: MIT
"""Restore experimental sequential MoE outputs to contiguous [token, topk, N]."""

from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from moe_multistage_down import _scalar


@cache
def make_restore_output(*, n, topk, num_oc_splits, output_layout, by_tile=False):
    assert n > 0 and n % (128 * num_oc_splits) == 0
    assert output_layout in ("sorted", "packed", "linear", "linear_raw")
    n_split = n // num_oc_splits
    n_packets = n_split // 64

    @flyc.kernel
    def restore_moe_output_kernel(
        output: fx.Pointer, source: fx.Pointer, sorted_ids: fx.Pointer,
        valid_ids: fx.Pointer, tokens: fx.Int32, capacity_rows: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        block = fx.Int32(fx.block_idx.x)
        limit = _scalar(valid_ids[0])
        s_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(source, fx.make_layout(capacity_rows * n, 1)), False)
        d_buffer = fx.rocdl.make_buffer_tensor(fx.make_view(output, fx.make_layout(tokens * topk * n, 1)), False)
        srsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(s_buffer))
        drsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(d_buffer))
        zero = fx.Int32(0).ir_value()
        vector_type = ir.VectorType.get([4], fx.Int32.ir_type)
        pair_type = ir.VectorType.get([2], fx.Int32.ir_type)

        def copy_packet(sorted_row, column, source_element):
            encoded = sorted_ids[sorted_row].bitcast(fx.Uint32)
            token, slot = encoded & 0xFFFFFF, encoded >> 24
            valid = (token < fx.Uint32(tokens)) & (slot < topk)
            if valid:
                dest = ((fx.Int32(token) * topk + fx.Int32(slot)) * n + column) * 2
                if const_expr(output_layout == "linear_raw" and not by_tile):
                    left = fx.Vector(rocdl.raw_ptr_buffer_load(pair_type, srsrc, (source_element * 2).ir_value(), zero))
                    right = fx.Vector(rocdl.raw_ptr_buffer_load(pair_type, srsrc, ((source_element + 32 * 8) * 2).ir_value(), zero))
                    value = fx.Vector.from_elements([left[0], left[1], right[0], right[1]], fx.Int32)
                    rocdl.raw_ptr_buffer_store(value.ir_value(), drsrc, dest.ir_value(), zero)
                elif const_expr(output_layout == "linear_raw"):
                    value = fx.Vector(rocdl.raw_ptr_buffer_load(vector_type, srsrc, (source_element * 2).ir_value(), zero))
                    for half in range_constexpr(2):
                        pair = fx.Vector.from_elements([value[half * 2], value[half * 2 + 1]], fx.Int32)
                        rocdl.raw_ptr_buffer_store(pair.ir_value(), drsrc, (dest + half * 16).ir_value(), zero)
                else:
                    value = rocdl.raw_ptr_buffer_load(vector_type, srsrc, (source_element * 2).ir_value(), zero)
                    rocdl.raw_ptr_buffer_store(value, drsrc, dest.ir_value(), zero)

        def restore_tile():
            bm = block // (num_oc_splits * n_packets)
            oc = block // n_packets % num_oc_splits
            q = block % n_packets
            if bm * 256 < limit:
                # Each turn reads one wave's 4 KiB output. The linear format
                # reads 16 bytes per adjacent lane with no address gaps.
                lane, record = tid % 64, tid // 64
                row_lane, k_lane = lane % 32, lane // 32
                local_column = record // 2 * 32 + record % 2 * 16 + k_lane * 4 if const_expr(output_layout == "linear_raw") else k_lane * 8 + row_lane // 16 * 16 + row_lane % 2 * 32
                for wave in range_constexpr(8):
                    row = wave * 32 + row_lane if const_expr(output_layout == "linear_raw") else wave * 32 + (row_lane % 16 // 2) * 2 + record // 2 + record % 2 * 16
                    sorted_row = bm * 256 + row
                    column = oc * n_split + q * 64 + local_column
                    source_element = (sorted_row * n + column if output_layout == "sorted" else
                                      bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + row * 64 + local_column if output_layout == "packed" else
                                      bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + record * 512 + lane * 8)
                    copy_packet(sorted_row, column, source_element)

        def restore_row():
            sorted_row = block
            bm, row = sorted_row // 256, sorted_row % 256
            local_row, wave = row % 32, row // 32
            record = local_row % 2 * 2 + local_row // 16
            if sorted_row < limit:
                for turn in range_constexpr((n + 2047) // 2048):
                    column = tid * 8 + turn * 2048
                    oc = column // n_split
                    q, local_column = column % n_split // 64, column % 64
                    row_lane = local_row % 16 // 2 * 2 + local_column // 32 + local_column % 32 // 16 * 16
                    lane = row_lane + (local_column % 16 // 8) * 32
                    source_element = (sorted_row * n + column if output_layout == "sorted" else
                                      bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + row * 64 + local_column if output_layout == "packed" else
                                      bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + (local_column // 16) * 512 + local_row * 8 + (local_column % 16 // 8) * 4 if output_layout == "linear_raw" else
                                      bm * (256 * n) + oc * (256 * n_split) + q * (256 * 64) + wave * 2048 + record * 512 + lane * 8)
                    if column < n:
                        copy_packet(sorted_row, column, source_element)

        if const_expr(by_tile):
            restore_tile()
        else:
            restore_row()

    @flyc.jit
    def launch(output: fx.Pointer, source: fx.Pointer, ids: fx.Pointer, valid: fx.Pointer,
               tokens: fx.Int32, capacity_rows: fx.Int32, stream: fx.Stream):
        grid = capacity_rows // 256 * (num_oc_splits * n_packets) if const_expr(by_tile) else capacity_rows
        restore_moe_output_kernel(output, source, ids, valid, tokens, capacity_rows).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream,
        )

    def restore(output: torch.Tensor, source: torch.Tensor,
                sorted_ids: torch.Tensor, valid_ids: torch.Tensor):
        assert output.ndim == 3 and output.shape[1:] == (topk, n)
        assert source.ndim == 2 and source.shape[1] == n
        # AITER rounds expert_ids capacity up to complete M256 blocks, while
        # sorted_ids can be a few entries shorter. Only rows below the device
        # valid_ids count are read; unused rounded capacity is never consumed.
        assert source.shape[0] == (sorted_ids.numel() + 255) // 256 * 256
        assert output.dtype == source.dtype == torch.bfloat16
        assert sorted_ids.dtype == valid_ids.dtype == torch.int32
        assert sorted_ids.ndim == 1 and valid_ids.numel() >= 1
        assert all(t.is_cuda and t.is_contiguous() and t.device == output.device
                   for t in (output, source, sorted_ids, valid_ids))
        assert source.numel() * 2 < 1 << 32 and output.numel() * 2 < 1 << 32
        stream = torch.cuda.current_stream(output.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in (output, source, sorted_ids, valid_ids)],
                          fx.Int32(output.shape[0]), fx.Int32(source.shape[0]), fx.Stream(stream.cuda_stream))
        else:
            compiled(output.data_ptr(), source.data_ptr(), sorted_ids.data_ptr(), valid_ids.data_ptr(),
                     output.shape[0], source.shape[0], stream.cuda_stream)
        return output

    return restore