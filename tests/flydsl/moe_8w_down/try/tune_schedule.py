# SPDX-License-Identifier: MIT
"""Same-buffer scheduling/compact comparisons; no default dispatch changes."""

import argparse
from functools import partial
import importlib.util
from pathlib import Path
import statistics
import sys

import test_blockscaled as comparison
from moe_multistage_pipeline import PACKED_DOWN_REDUCE_CONFIG, make_down_reduce
from moe_multistage_compact import CompactWorkspace, make_compact_down_reduce


def load_snapshot(name, filename):
    path = Path(__file__).parent / "ck_test/oc_compact_20260912/baseline/source" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", default=["baseline", "persistent4", "oneshot4", "xcd4", "compact0_4", "compact4"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or (args.paired and len(args.candidate) != 2) or (args.profile and len(args.candidate) != 1):
        parser.error("positive rounds, two paired candidates or one profile candidate required")
    workspace = CompactWorkspace()
    frozen = load_snapshot("oc_frozen_down", "moe_multistage_down.py")
    baseline = load_snapshot("oc_frozen_pipeline", "moe_multistage_pipeline.py")
    baseline.flydsl_moe_gemm_8wave_down = frozen.flydsl_moe_gemm_8wave_down
    factories = {"baseline": (partial(baseline.make_down_reduce, workspace=workspace), dict(PACKED_DOWN_REDUCE_CONFIG))}
    for split in (1, 2, 4, 8):
        for schedule in ("persistent", "oneshot", "xcd"):
            factories[f"{schedule}{split}"] = (partial(make_down_reduce, workspace=workspace), {
                **PACKED_DOWN_REDUCE_CONFIG, "num_oc_splits": split,
                "persistent": schedule == "persistent", "xcd_swizzle": schedule == "xcd",
            })
        for name, threshold in (("compact", 0.6), ("compact0_", 0)):
            factories[f"{name}{split}"] = (partial(make_compact_down_reduce, workspace=workspace), {
                "num_oc_splits": split, "min_tail_utilization": threshold,
            })
    for name, threshold in (("compact1_4", 0.6), ("compact0_1_4", 0)):
        factories[name] = (partial(make_compact_down_reduce, workspace=workspace), {
            "num_oc_splits": 1, "tail_num_oc_splits": 4, "min_tail_utilization": threshold,
        })
    if not all(key in factories for key in args.candidate):
        parser.error(f"candidates: {tuple(factories)}")
    order, labels = [], {}
    for r in range(1 if args.profile else args.rounds):
        keys = args.candidate if r % 2 == 0 else args.candidate[::-1]
        if args.paired:
            a, b = keys
            keys = [a, b, b, a]
        for index, key in enumerate(keys):
            name = f"{key}_r{r + 1}_{index}"
            factory, options = factories[key]
            comparison.CANDIDATES[name] = (name, 128, factory)
            comparison.CANDIDATE_OPTIONS[name] = options
            order.append(name)
            labels[name] = (key, r + 1)
    print(f"SCHEDULE_CONFIG candidates={args.candidate} rounds={args.rounds} paired={args.paired}", flush=True)
    rows = comparison.run_test(tokens=16384, model_dim=6144, inter_dim=256, experts=384, topk=8,
                               block_m=256, num_oc_splits=1, seed=1234, candidates=order,
                               reduce_output=True, profile=args.profile)
    grouped = {key: [] for key in args.candidate}
    for row in rows:
        assert row["status"] == "PASS", row
        if args.profile:
            continue
        key, r = labels[row["name"]]
        grouped[key].append(row)
        print(f"SCHEDULE_SAMPLE round={r} candidate={key} down_us={row['down_elapsed_us']:.6f} "
              f"total_us={row['elapsed_us']:.6f} rows={row.get('compute_rows', 196608)} "
              f"full={row.get('full256_tasks', 768)} tail={row.get('tail64_tasks', 0)} components={row['components_us']}", flush=True)
    for key, samples in grouped.items():
        if samples:
            print(f"SCHEDULE_MEAN candidate={key} down_us={statistics.mean(r['down_elapsed_us'] for r in samples):.6f} "
                  f"total_us={statistics.mean(r['elapsed_us'] for r in samples):.6f} checks={len(samples)}", flush=True)


if __name__ == "__main__":
    main()