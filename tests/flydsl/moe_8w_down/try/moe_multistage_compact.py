# SPDX-License-Identifier: MIT
"""GPU task tables: full M256 + M64 tails, retaining M256 physical sorted rows.

Adapted from gemm2_8x1_compact: table entries are [row_begin, expert], counts
remain on device, and an underfilled last full-task batch may become M64s.
The input is AITER's M256 sorting (valid prefix within each expert run).
Unlike re-sorting to M64, this keeps the exact routing and packed buffer
addresses of the baseline, including its unused padding. No host count read
or tensor-content cache participates in launch/grid selection.
"""

from fractions import Fraction
from functools import cache

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from pyhip.contrib.flydsl.moe_gemm_2stage.common import torch_tensor_to_pointer as _ptr
from pyhip.contrib.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids

from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_down_m64 import make_m64_down
from moe_multistage_pipeline import DownReduceWorkspace
from moe_multistage_reduce import make_moe_sum


def device_cu_count(device):
    """Use the actual device's available CUs, not the MI308X reference's 80."""
    count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    assert count > 0
    return count


def task_launch_stats(tasks, capacity, num_oc_splits, cu_count):
    """Equal-cost, one-active-WG-per-CU round model; not measured occupancy."""
    assert 0 <= tasks <= capacity and num_oc_splits > 0 and cu_count > 0
    active = tasks * num_oc_splits
    launch = capacity * num_oc_splits
    rounds = (active + cu_count - 1) // cu_count
    last = (active % cu_count or cu_count) if active else 0
    return {"tasks": tasks, "cu_count": cu_count, "num_oc_splits": num_oc_splits,
            "active_workgroups": active, "launch_workgroups": launch,
            "early_exit_workgroups": launch - active,
            "cu_rounds_model": rounds, "last_batch_workgroups": last,
            "last_batch_utilization": last / cu_count,
            "cu_capacity_loss": 1 - active / (rounds * cu_count) if rounds else 0.0}


@cache
def make_task_builder(capacity, topk, cu_count, min_tail_utilization=0.6, num_oc_splits=1):
    assert capacity > 0 and 0 < topk <= 255 and cu_count > 0
    assert num_oc_splits > 0
    assert 0 <= min_tail_utilization <= 1
    threshold = Fraction(str(min_tail_utilization)).limit_denominator(1_000_000)
    numerator, denominator = threshold.numerator, threshold.denominator
    width = max(256, 1 << (capacity - 1).bit_length())
    assert width * 16 <= 160 * 1024, "task scan exceeds gfx950 LDS"

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_multistage_build_tasks(ids: fx.Pointer, experts: fx.Pointer, valid: fx.Pointer,
                                    full: fx.Pointer, tail: fx.Pointer, counts: fx.Pointer,
                                    tokens: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        shared = fx.SharedAllocator().allocate(fx.Array[fx.Int32, width * 4, 16]).peek().view(fx.make_layout(width * 4, 1))
        blocks = valid[0] // 256
        for part in range_constexpr(width // 256):
            idx = tid + part * 256
            is_full, tails = fx.Int32(0), fx.Int32(0)
            if idx < blocks:
                # As in the M64-metadata compact reference, four occupied
                # M64 chunks form one M256 task. Its final chunk may be padded.
                last_chunk = ids[idx * 256 + 192].bitcast(fx.Uint32)
                is_full = fx.Int32(((last_chunk & 0xFFFFFF) < fx.Uint32(tokens)) & ((last_chunk >> 24) < topk))
                for piece in range_constexpr(4):
                    first = ids[idx * 256 + piece * 64].bitcast(fx.Uint32)
                    live = ((first & 0xFFFFFF) < fx.Uint32(tokens)) & ((first >> 24) < topk)
                    tails += fx.Int32(live) * (1 - is_full)
            shared[idx] = is_full
            shared[width + idx] = tails
            shared[2 * width + idx] = is_full
            shared[3 * width + idx] = tails
        fx.barrier()
        for shift in range_constexpr(width.bit_length() - 1):
            distance = 1 << shift
            next_full, next_tail = [], []
            for part in range_constexpr(width // 256):
                idx = tid + part * 256
                peer = (idx >= distance).select(idx - distance, idx)
                next_full.append(shared[2 * width + idx] + (idx >= distance).select(shared[2 * width + peer], fx.Int32(0)))
                next_tail.append(shared[3 * width + idx] + (idx >= distance).select(shared[3 * width + peer], fx.Int32(0)))
            fx.barrier()
            for part in range_constexpr(width // 256):
                idx = tid + part * 256
                shared[2 * width + idx] = next_full[part]
                shared[3 * width + idx] = next_tail[part]
            fx.barrier()
        original_full = shared[3 * width - 1]
        original_tail = shared[4 * width - 1]
        # The reference visits all N in one WG. Here each M task launches
        # OC splits WGs: measure last-batch utilization in WG units, not M
        # tasks, otherwise384 M tasks at split4 would be split needlessly.
        remainder = (original_full * num_oc_splits) % cu_count
        split_count = (remainder + num_oc_splits - 1) // num_oc_splits
        split_full = ((remainder > 0) & (remainder * denominator < cu_count * numerator)).select(split_count, fx.Int32(0))
        keep_full = original_full - split_full
        if tid == 0:
            counts[0] = keep_full * 256
            counts[1] = (original_tail + 4 * split_full) * 64
        for part in range_constexpr(width // 256):
            idx = tid + part * 256
            if idx < blocks:
                is_full, tail_count = shared[idx], shared[width + idx]
                full_begin = shared[2 * width + idx] - is_full
                tail_begin = shared[3 * width + idx] - tail_count
                split_before = (full_begin > keep_full).select(full_begin - keep_full, fx.Int32(0))
                tail_begin += 4 * split_before
                split_here = (is_full != 0) & (full_begin >= keep_full)
                if (is_full != 0) & (full_begin < keep_full):
                    full[2 * full_begin] = idx * 256
                    full[2 * full_begin + 1] = experts[idx]
                for piece in range_constexpr(4):
                    if (piece < tail_count) | split_here:
                        slot = tail_begin + piece
                        tail[2 * slot] = idx * 256 + piece * 64
                        tail[2 * slot + 1] = experts[idx]

    @flyc.jit
    def launch(ids: fx.Pointer, experts: fx.Pointer, valid: fx.Pointer,
               full: fx.Pointer, tail: fx.Pointer, counts: fx.Pointer,
               tokens: fx.Int32, stream: fx.Stream):
        moe_multistage_build_tasks(ids, experts, valid, full, tail, counts, tokens).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream,
        )

    def build(ids, experts, valid, full, tail, counts, tokens):
        tensors = (ids, experts, valid, full, tail, counts)
        assert experts.numel() == capacity and full.ndim == tail.ndim == 2
        assert full.shape[1] == tail.shape[1] == 2 and counts.numel() == 2
        assert all(t.dtype == torch.int32 and t.is_cuda and t.is_contiguous() and t.device == ids.device for t in tensors)
        stream = torch.cuda.current_stream(ids.device)
        compiled = getattr(launch, "_cf", None)
        if compiled is None:
            _run_compiled(launch, *[_ptr(t) for t in tensors], fx.Int32(tokens), fx.Stream(stream.cuda_stream))
        else:
            compiled(*(t.data_ptr() for t in tensors), tokens, stream.cuda_stream)
    return build


class CompactWorkspace(DownReduceWorkspace):
    def __init__(self):
        super().__init__()
        self.full = self.tail = self.counts = None

    def prepare_tables(self, input_q, expert_ids, num_experts, num_oc_splits, utilization):
        capacity = expert_ids.numel()
        # Four occupied M64 chunks require >=193 real routes; a partial
        # expert run contributes <=3 M64 tails. Bounds use host shapes only.
        # num_oc_splits is the M256 scheduler's split count, even when the
        # M64 scheduler differs: only full tasks are converted into tails.
        rows = input_q.shape[0] * input_q.shape[1]
        full_capacity = max(1, min(capacity, rows // 193))
        cu_count = device_cu_count(input_q.device)
        ratio = Fraction(str(utilization)).limit_denominator(1_000_000)
        max_remainder = max(0, (cu_count * ratio.numerator - 1) // ratio.denominator)
        max_split = (max_remainder + num_oc_splits - 1) // num_oc_splits
        tail_capacity = max(1, min(capacity * 4, rows, 3 * num_experts + 4 * max_split))
        if (self.full is None or self.full.shape[0] != full_capacity or self.tail.shape[0] != tail_capacity
                or self.full.device != input_q.device):
            self.full = torch.empty((full_capacity, 2), dtype=torch.int32, device=input_q.device)
            self.tail = torch.empty((tail_capacity, 2), dtype=torch.int32, device=input_q.device)
            self.counts = torch.empty(2, dtype=torch.int32, device=input_q.device)


def make_compact_down_reduce(*, n, k, topk, num_experts, workspace=None,
                             min_tail_utilization=0.6, xcd_swizzle=True,
                             block_m=256, block_n=128, num_oc_splits=4,
                             tail_num_oc_splits=None, **overrides):
    """Use num_oc_splits for M256 and, by default, M64; override tails separately.

    Packed [block256,OC,N64,row256,col64] is OC-independent: when N/OC is
    divisible by64, OC and local N64 flatten to global column//64. Thus both
    kernels share the same storage, inverse and reducer without a restore.
    Full-task balancing and tail-table capacity always use M256's OC count.
    """
    assert k == 256 and block_m == 256 and block_n == 128
    tail_num_oc_splits = num_oc_splits if tail_num_oc_splits is None else tail_num_oc_splits
    assert num_oc_splits > 0 and tail_num_oc_splits > 0
    assert n > 0 and n % (128 * num_oc_splits) == 0 and n % (128 * tail_num_oc_splits) == 0
    workspace = workspace if workspace is not None else CompactWorkspace()
    options = {**ATT_TUNED_BN128_CONFIG, **overrides, "num_oc_splits": num_oc_splits,
               "persistent": False, "xcd_swizzle": xcd_swizzle, "task_table": True, "output_layout": "packed"}
    full = flydsl_moe_gemm_8wave_down(n=n, k=k, topk=topk, num_experts=num_experts, **options)
    tail = make_m64_down(n=n, k=k, topk=topk, num_experts=num_experts, num_oc_splits=tail_num_oc_splits,
                         block_n=32 if n // tail_num_oc_splits < 192 else 64,
                         xcd_swizzle=xcd_swizzle, output_layout="packed")
    inverse_kernel = invert_sorted_ids(topk)
    reduce = make_moe_sum(n=n, topk=topk, num_oc_splits=num_oc_splits, output_layout="packed",
                          num_threads=256, block_cols=2048, read_policy=2)

    def components(*args):
        out, a, b, sa, sb, ids, routes, eids, valid, counter = args
        middle, inverse = workspace.prepare(out, a, eids)
        workspace.prepare_tables(a, eids, num_experts, num_oc_splits, min_tail_utilization)
        middle = middle[:eids.numel() * 256]
        full_count, tail_count = workspace.counts[:1], workspace.counts[1:]
        builder = make_task_builder(eids.numel(), topk, device_cu_count(a.device),
                                    min_tail_utilization, num_oc_splits)

        def tasks():
            builder(ids, eids, valid, workspace.full, workspace.tail, workspace.counts, a.shape[0])

        def full_down():
            full(middle, a, b, sa, sb, ids, routes, workspace.full, full_count, counter)

        def tail_down():
            tail(middle, a, b, sa, sb, ids, routes, workspace.tail, tail_count, counter)

        def gemm():
            tasks()
            full_down()
            tail_down()

        def invert():
            inverse.fill_(-1)
            inverse_kernel(ids, inverse, valid, ids.numel(), a.shape[0])

        return {"gemm": gemm, "task_build": tasks, "full256": full_down, "tail64": tail_down,
                "inverse": invert, "reduce": lambda: reduce(out, middle, inverse)}

    def launch(*args):
        calls = components(*args)
        calls["gemm"]()
        calls["inverse"]()
        calls["reduce"]()
        return args[0]

    def poison(*args):
        middle, inverse = workspace.prepare(args[0], args[1], args[7])
        workspace.prepare_tables(args[1], args[7], num_experts, num_oc_splits, min_tail_utilization)
        middle.fill_(torch.nan)
        inverse.fill_(0x123456)
        workspace.counts.fill_(0x123456)

    launch.benchmark_components = components
    launch.poison_workspace = poison
    launch.workspace = workspace

    def stats(output, input_q, weight, input_scales, weight_scales,
              sorted_ids, sorted_weights, expert_ids, valid_ids, counter):
        # Diagnostic only, called AFTER every timer and final launch. Counting
        # live rows/unique experts here must never affect a grid or task split.
        full_rows, tail_rows = workspace.counts.tolist()
        cu_count = device_cu_count(input_q.device)
        limit = int(valid_ids[0].item())
        assert 0 <= limit <= sorted_ids.numel() and limit % 256 == 0
        encoded = sorted_ids[:limit].detach().cpu().to(torch.int64)
        live = ((encoded & 0xFFFFFF) < input_q.shape[0]) & (((encoded >> 24) & 0xFF) < topk)
        prefix = torch.cat((torch.zeros(1, dtype=torch.int64, device="cpu"), live.cumsum(0)))
        original_full = int(live[192::256].sum().item())

        def kernel_work(table, rows, bm, config):
            tasks = rows // bm
            entries = table[:tasks].detach().cpu().to(torch.int64)
            starts = entries[:, 0]
            assert torch.all((starts >= 0) & (starts + bm <= limit))
            valid_rows = int((prefix[starts + bm] - prefix[starts]).sum().item())
            unique_experts = torch.unique(entries[:, 1]).numel()
            # Same logical A-once/B-per-task/C-valid model as the main table.
            # Metadata, transaction amplification and repeated A per OC are
            # intentionally excluded; this is not an HBM counter measurement.
            non_weight_bytes = valid_rows * (k * input_q.element_size() + n * output.element_size())
            weight_bytes = n * k * weight.element_size()
            return {**task_launch_stats(tasks, table.shape[0], config["num_oc_splits"], cu_count),
                    "block_m": bm, "valid_rows": valid_rows, "compute_rows": rows,
                    "unique_experts": unique_experts,
                    "effective_flops": 2 * valid_rows * n * k, "padded_flops": 2 * rows * n * k,
                    "rw_bytes": non_weight_bytes + tasks * weight_bytes,
                    "ideal_rw_bytes": non_weight_bytes + unique_experts * weight_bytes,
                    "xcd_swizzle": config["xcd_swizzle"], "xcd_count": config["xcd_count"]}

        work = {"full256": kernel_work(workspace.full, full_rows, 256, full.config),
                "tail64": kernel_work(workspace.tail, tail_rows, 64, tail.config)}
        assert sum(row["valid_rows"] for row in work.values()) == int(live.sum().item())
        original = task_launch_stats(original_full, expert_ids.numel(), num_oc_splits, cu_count)
        return {"compute_rows": full_rows + tail_rows,
                "full256_tasks": full_rows // 256, "tail64_tasks": tail_rows // 64,
                "weight_task_count": full_rows // 256 + tail_rows // 64,
                "full_launch_workgroups": workspace.full.shape[0] * num_oc_splits,
                "tail_launch_workgroups": workspace.tail.shape[0] * tail_num_oc_splits,
                "split_cu_count": cu_count, "original_full256_tasks": original_full,
                "converted_full256_tasks": original_full - full_rows // 256,
                "original_full_active_workgroups": original["active_workgroups"],
                "original_full_last_batch_utilization": original["last_batch_utilization"],
                "kernel_work": work}
    launch.benchmark_stats = stats
    launch.config = {**full.config, "reduction": "custom", "reduce_threads": 256, "reduce_cols": 2048,
                     "read_policy": 2, "includes_inverse": True, "includes_reduce": True,
                     "includes_task_build": True, "min_tail_utilization": min_tail_utilization,
                     "full_num_oc_splits": num_oc_splits, "tail_num_oc_splits": tail_num_oc_splits}
    return launch