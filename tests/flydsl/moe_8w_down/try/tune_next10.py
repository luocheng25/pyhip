# SPDX-License-Identifier: MIT
"""Compare MFMA32 against the frozen256-CTA tuned baseline; no target claim."""

import argparse
import importlib.util
from pathlib import Path
import statistics
import sys

import test_blockscaled as comparison
from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_down_mfma32 import MFMA32_CONFIG, flydsl_moe_gemm_8wave_down_mfma32


def register_candidates():
    snapshot = Path(__file__).parent / "ck_test/next10_20260912/baseline/source/moe_8wave_down.py"
    baseline_factory, baseline_config = flydsl_moe_gemm_8wave_down, ATT_TUNED_BN128_CONFIG
    if snapshot.exists():
        spec = importlib.util.spec_from_file_location("moe_down_frozen256", snapshot)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        baseline_factory, baseline_config = baseline.flydsl_moe_gemm_8wave_down, baseline.ATT_TUNED_BN128_CONFIG
    factories = {"baseline16": (baseline_factory, dict(baseline_config))}
    variants = {
        "mfma32_pipe128": {},
        "mfma32_w128": {"fold_routing": False, "defer_k1": False},
        "mfma32_fold128": {"defer_k1": False},
        "mfma32_pipe_skip": {"skip_empty_waves": True},
        "mfma32_pipe_rows": {"cache_row_scales": True},
        "mfma32_group2": {"task_m_group": 2},
        "mfma32_pipe_combo": {"skip_empty_waves": True, "cache_row_scales": True, "task_m_group": 2},
    }
    factories.update({key: (flydsl_moe_gemm_8wave_down_mfma32, {**MFMA32_CONFIG, **options})
                      for key, options in variants.items()})
    for key, (factory, options) in factories.items():
        comparison.CANDIDATES[key] = (key, 128, factory)
        comparison.CANDIDATE_OPTIONS[key] = options
        if key.startswith("mfma32"):
            comparison.CANDIDATE_WEIGHT_LAYOUTS[key] = (32, 16)
    return tuple(factories)


def main():
    keys = register_candidates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", choices=keys, default=["baseline16", "mfma32_pipe128"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--paired", action="store_true", help="two candidates, ABBA/BAAB ordering")
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.candidate)) != len(args.candidate):
        parser.error("positive rounds and unique candidates required")
    if args.profile and (args.rounds != 1 or len(args.candidate) != 1 or args.paired):
        parser.error("profile needs one candidate and one round")
    if args.paired and (len(args.candidate) != 2 or args.candidate[0] != "baseline16"):
        parser.error("paired mode requires --candidate baseline16 <trial>")
    measurements = {key: [] for key in args.candidate}
    aliases = {}
    if args.paired:
        for key in args.candidate:
            alias = key + "_repeat"
            label, bn, factory = comparison.CANDIDATES[key]
            comparison.CANDIDATES[alias] = (alias, bn, factory)
            comparison.CANDIDATE_OPTIONS[alias] = comparison.CANDIDATE_OPTIONS[key]
            comparison.CANDIDATE_WEIGHT_LAYOUTS[alias] = comparison.CANDIDATE_WEIGHT_LAYOUTS.get(key, (16, 16))
            aliases[alias] = key
    for index in range(args.rounds):
        order = args.candidate if index % 2 == 0 else args.candidate[::-1]
        if args.paired:
            outer, inner = order
            order = [outer, inner, inner + "_repeat", outer + "_repeat"]
        results = comparison.run_test(
            tokens=16384, model_dim=6144, inter_dim=256, experts=384,
            topk=8, block_m=256, num_oc_splits=1, seed=1234,
            candidates=order, profile=args.profile,
        )
        round_times = {key: [] for key in args.candidate}
        for row in results:
            assert row["status"] == "PASS", row
            if not args.profile:
                round_times[aliases.get(row["name"], row["name"])].append(row["elapsed_us"])
        if not args.profile:
            for key, values in round_times.items():
                elapsed = statistics.mean(values)
                measurements[key].append(elapsed)
                print(f"NEXT10_ROUND round={index + 1} candidate={key} us={elapsed:.6f}", flush=True)
            if args.paired:
                a, b = (statistics.mean(round_times[key]) for key in args.candidate)
                print(f"NEXT10_PAIRED round={index + 1} baseline_us={a:.6f} trial_us={b:.6f} speedup={a / b:.6f}", flush=True)
    if not args.profile:
        baseline = statistics.mean(measurements["baseline16"]) if "baseline16" in measurements else None
        for key, samples in measurements.items():
            elapsed = statistics.mean(samples)
            speedup = baseline / elapsed if baseline is not None else 1.0
            print(f"NEXT10_MEAN candidate={key} us={elapsed:.6f} speedup={speedup:.6f} "
                  f"latency_drop_pct={(1 - 1 / speedup) * 100:.6f}", flush=True)


if __name__ == "__main__":
    main()