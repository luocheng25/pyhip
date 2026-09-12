# SPDX-License-Identifier: MIT
"""Compare OC splitting with fixed down/reduce options and shared buffers."""

import argparse
from functools import partial
import statistics

import test_blockscaled as comparison
from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_pipeline import PACKED_DOWN_REDUCE_CONFIG, DownReduceWorkspace, make_down_reduce
from moe_multistage_compact import CompactWorkspace, make_compact_down_reduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--paired", action="store_true", help="two split counts, alternating ABBA/BAAB")
    parser.add_argument("--profile", action="store_true", help="23 launches, one split count, no timing")
    parser.add_argument("--mode", choices=("down", "down-reduce"), default="down-reduce")
    parser.add_argument("--layout", choices=("routed", "packed"), default="packed")
    parser.add_argument("--schedule", choices=("persistent", "oneshot", "xcd", "compact", "compact0"), default="persistent")
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.splits)) != len(args.splits) or any(s <= 0 or 6144 % (128 * s) for s in args.splits):
        parser.error("positive rounds and unique split counts dividing N6144 into N128 tiles required")
    if args.paired and len(args.splits) != 2:
        parser.error("--paired requires exactly two split counts")
    if args.profile and len(args.splits) != 1:
        parser.error("--profile requires exactly one split count")
    if args.mode == "down" and args.layout != "routed":
        parser.error("packed down-only timing is the gemm component of --mode down-reduce")

    if args.schedule.startswith("compact") and (args.mode != "down-reduce" or args.layout != "packed"):
        parser.error("compact requires packed down-reduce")
    workspace = CompactWorkspace()
    order, counts = [], {}
    rounds = 1 if args.profile else args.rounds
    for index in range(rounds):
        splits = args.splits if index % 2 == 0 else args.splits[::-1]
        if args.paired:
            a, b = splits
            splits = [a, b, b, a]
        for repeat, split in enumerate(splits):
            key = f"oc{split}_r{index + 1}_{repeat}"
            factory = flydsl_moe_gemm_8wave_down if args.mode == "down" else partial(make_down_reduce, workspace=workspace)
            options = dict(ATT_TUNED_BN128_CONFIG if args.mode == "down" else PACKED_DOWN_REDUCE_CONFIG)
            options.update(num_oc_splits=split, output_layout=args.layout)
            if args.schedule in ("oneshot", "xcd"):
                options.update(persistent=False, xcd_swizzle=args.schedule == "xcd")
            elif args.schedule.startswith("compact"):
                factory = partial(make_compact_down_reduce, workspace=workspace)
                options = {"num_oc_splits": split, "min_tail_utilization": 0 if args.schedule == "compact0" else 0.6}
            comparison.CANDIDATES[key] = (key, 128, factory)
            comparison.CANDIDATE_OPTIONS[key] = options
            counts[key] = (index + 1, split)
            order.append(key)
    print(f"OC_SWEEP splits={args.splits} rounds={rounds} paired={args.paired} mode={args.mode} "
          f"layout={args.layout} schedule={args.schedule} N=6144 K=256", flush=True)
    # A single data setup keeps the exact same A/B, routing, intermediate and
    # output allocations across ALL rounds, including correctness checks.
    results = comparison.run_test(
        tokens=16384, model_dim=6144, inter_dim=256, experts=384, topk=8,
        block_m=256, num_oc_splits=1, seed=1234, candidates=order,
        profile=args.profile, reduce_output=args.mode == "down-reduce",
    )
    if args.profile:
        return
    grouped = {s: [] for s in args.splits}
    for row in results:
        assert row["status"] == "PASS", row
        index, split = counts[row["name"]]
        grouped[split].append(row)
        print(f"OC_SAMPLE round={index} splits={split} down_us={row['down_elapsed_us']:.6f} "
              f"total_us={row['elapsed_us']:.6f} components={row['components_us']}", flush=True)
    table = []
    for split, rows in grouped.items():
        down = statistics.mean(row["down_elapsed_us"] for row in rows)
        total = statistics.mean(row["elapsed_us"] for row in rows)
        table.append([split, 6144 // split, 768 * split, f"{down:.3f}", f"{total:.3f}"])
        print(f"OC_MEAN splits={split} down_us={down:.6f} total_us={total:.6f} checks={len(rows)}", flush=True)
    comparison.print_markdown_table(["OC splits", "N/task", "Tasks", "Down us", "Total us"], table)


if __name__ == "__main__":
    main()