# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit down+TOPK-reduce composition; output is [tokens,N], not [tokens,topk,N]."""

import torch

from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_down_mfma32 import flydsl_moe_gemm_8wave_down_mfma32
from moe_multistage_reduce import make_moe_sum
from pyhip.contrib.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids, sorted_sum


PACKED_DOWN_REDUCE_CONFIG = {
    **ATT_TUNED_BN128_CONFIG,
    "down_kind": "mfma16", "output_layout": "packed", "reduction": "custom",
    "reduce_threads": 256, "reduce_cols": 2048, "read_policy": 2,
}


class DownReduceWorkspace:
    """Reusable caller-owned storage; one instance per stream/in-flight invocation."""

    def __init__(self):
        self.data = None
        self.inverse = None

    def prepare(self, output, input_q, expert_ids):
        assert output.ndim == 2 and input_q.ndim == 3
        tokens, topk, _ = input_q.shape
        rows = max(tokens * topk, expert_ids.numel() * 256)
        shape = (rows, output.shape[1])
        if self.data is None or self.data.shape != shape or self.data.device != output.device:
            self.data = torch.empty(shape, dtype=torch.bfloat16, device=output.device)
        if self.inverse is None or self.inverse.shape != (tokens, topk) or self.inverse.device != output.device:
            self.inverse = torch.empty((tokens, topk), dtype=torch.int32, device=output.device)
        return self.data, self.inverse


def make_down_reduce(*, n, k, topk, num_experts, down_kind="mfma32", output_layout="sorted",
                     reduction="custom", reduce_threads=64, reduce_cols=1024, preload=True,
                     read_policy=0, write_policy=0, tree_sum=False, workspace=None, **down_options):
    assert down_kind in ("mfma16", "mfma32") and reduction in ("torch", "custom", "reference")
    assert k == 256 and n > 0 and 0 < topk <= min(num_experts, 255)
    assert reduction != "torch" or output_layout == "routed"
    assert reduction != "reference" or output_layout == "sorted"
    assert down_kind != "mfma16" or output_layout in ("routed", "sorted", "packed")
    if workspace is None:
        workspace = DownReduceWorkspace()
    down_factory = flydsl_moe_gemm_8wave_down if down_kind == "mfma16" else flydsl_moe_gemm_8wave_down_mfma32
    options = dict(n=n, k=k, topk=topk, num_experts=num_experts, **down_options)
    options["output_layout"] = output_layout
    down = down_factory(**options)
    inverse_kernel = invert_sorted_ids(topk) if output_layout != "routed" else None
    reducer = (sorted_sum(topk, n) if reduction == "reference" else
               make_moe_sum(n=n, topk=topk, num_oc_splits=down_options.get("num_oc_splits", 4),
                            output_layout=output_layout, num_threads=reduce_threads,
                            block_cols=reduce_cols, preload=preload, read_policy=read_policy,
                            write_policy=write_policy, tree_sum=tree_sum) if reduction == "custom" else None)

    def buffers(args):
        output, input_q, expert_ids = args[0], args[1], args[7]
        assert output.shape == (input_q.shape[0], n) and output.dtype == torch.bfloat16
        assert input_q.shape[1:] == (topk, k)
        assert output.is_cuda and output.is_contiguous() and output.device == input_q.device
        storage, inverse = workspace.prepare(output, input_q, expert_ids)
        if output_layout == "routed":
            middle = storage[:input_q.shape[0] * topk].view(input_q.shape[0], topk, n)
        else:
            middle = storage[:expert_ids.numel() * 256]
        return middle, inverse

    def invert(args, inverse):
        # Rebuild every invocation, even with unchanged pointers: routing is
        # dynamic. Clear missing routes to zero-contribution sentinels.
        inverse.fill_(-1)
        inverse_kernel(args[5], inverse, args[8], args[5].numel(), args[1].shape[0])

    def sum_output(output, middle, inverse):
        if reduction == "torch":
            # Current physical down layout is [tokens,topk,N]; TOPK is dim1.
            torch.sum(middle, dim=1, out=output)
        elif reduction == "reference":
            reducer(inverse, middle, output, output.shape[0])
        else:
            reducer(output, middle, inverse)
        return output

    def launch(*args):
        middle, inverse = buffers(args)
        down(middle, *args[1:])
        if inverse_kernel is not None:
            invert(args, inverse)
        return sum_output(args[0], middle, inverse)

    def components(*args):
        middle, inverse = buffers(args)
        result = {"gemm": lambda: down(middle, *args[1:])}
        if inverse_kernel is not None:
            result["inverse"] = lambda: invert(args, inverse)
        result["reduce"] = lambda: sum_output(args[0], middle, inverse)
        return result

    def poison(*args):
        middle, inverse = buffers(args)
        middle.fill_(torch.nan)
        inverse.fill_(0x123456)

    launch.benchmark_components = components
    launch.poison_workspace = poison
    launch.workspace = workspace
    launch.config = {**down.config, "reduction": reduction, "output_layout": output_layout,
                     "reduce_threads": reduce_threads, "reduce_cols": reduce_cols,
                     "read_policy": read_policy, "write_policy": write_policy, "tree_sum": tree_sum,
                     "includes_inverse": inverse_kernel is not None, "includes_reduce": True}
    return launch


def compile_packed_down_reduce(*, n, k, topk, num_experts, workspace=None, **overrides):
    """Build the measured packed down+sum path; weight shuffle remains (16,16).

    The returned ten-tensor callable writes [tokens,N] and includes rebuilding
    the inverse routing map. Warm once before CUDA graph capture, or provide
    a preallocated DownReduceWorkspace. Use separate workspaces concurrently.
    """
    options = {**PACKED_DOWN_REDUCE_CONFIG, **overrides}
    return make_down_reduce(n=n, k=k, topk=topk, num_experts=num_experts,
                            workspace=workspace, **options)