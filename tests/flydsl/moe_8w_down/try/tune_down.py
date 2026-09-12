# SPDX-License-Identifier: MIT
"""Paired PyHIP / ATT-tuned comparisons and isolated persistent-CTA ablations."""

import argparse
import statistics

import test_blockscaled as comparison


def compare_persistent_ctas(counts, rounds, seed):
    """Change only the launch grid; balance order with ABBA / BAAB rounds."""
    base_count, trial_count = counts
    _, block_n, factory = comparison.CANDIDATES["4stage_bn128_tuned"]
    fixed_options = dict(comparison.CANDIDATE_OPTIONS["4stage_bn128_tuned"])
    fixed_options.pop("persistent_workgroups")
    keys = {}
    for count in counts:
        keys[count] = []
        for repeat in ("a", "b"):
            key = f"4stage_bn128_tuned_cta{count}_{repeat}"
            keys[count].append(key)
            comparison.CANDIDATES[key] = (f"Tuned CTA{count} {repeat}", block_n, factory)
            comparison.CANDIDATE_OPTIONS[key] = dict(
                fixed_options, persistent_workgroups=count,
            )

    print(f"CTA_ABLATION base_ctas={base_count} trial_ctas={trial_count} "
          f"rounds={rounds} fixed_options={fixed_options}", flush=True)
    measurements = []
    common_config = None
    for index in range(rounds):
        outer, inner = counts if index % 2 == 0 else counts[::-1]
        order = [keys[outer][0], keys[inner][0], keys[inner][1], keys[outer][1]]
        results = comparison.run_test(
            tokens=16384, model_dim=6144, inter_dim=256, experts=384,
            topk=8, block_m=256, num_oc_splits=1, seed=seed, candidates=order,
        )
        times = {count: [] for count in counts}
        for row in results:
            assert row["status"] == "PASS", row
            config = dict(row["config"])
            count = config.pop("persistent_workgroups")
            if common_config is None:
                common_config = config
            assert config == common_config, "CTA ablation changed another kernel option"
            times[count].append(row["elapsed_us"])
        assert all(len(values) == 2 for values in times.values())
        base = statistics.mean(times[base_count])
        trial = statistics.mean(times[trial_count])
        measurements.append((base, trial))
        print(f"CTA_ROUND {index + 1} order={outer},{inner},{inner},{outer} "
              f"base_us={base:.6f} trial_us={trial:.6f} speedup={base / trial:.6f} "
              f"latency_drop_pct={(1 - trial / base) * 100:.6f}", flush=True)

    base = statistics.mean(row[0] for row in measurements)
    trial = statistics.mean(row[1] for row in measurements)
    table = [
        [str(index + 1), f"{a:.3f}", f"{b:.3f}", f"{a / b:.6f}x",
         f"{(1 - b / a) * 100:.3f}%"]
        for index, (a, b) in enumerate(measurements)
    ]
    table.append(["Mean", f"{base:.3f}", f"{trial:.3f}", f"{base / trial:.6f}x",
                  f"{(1 - trial / base) * 100:.3f}%"])
    comparison.print_markdown_table(
        ["Round", f"CTA{base_count} us", f"CTA{trial_count} us", "Speedup", "Latency drop"],
        table,
    )
    print(f"CTA_AGGREGATE base_ctas={base_count} trial_ctas={trial_count} "
          f"base_us={base:.6f} trial_us={trial:.6f} speedup={base / trial:.6f} "
          f"latency_drop_pct={(1 - trial / base) * 100:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile", action="store_true", help="23 launches of only the tuned candidate; no timings")
    parser.add_argument("--compare-ctas", type=int, nargs=2, metavar=("BASE", "TRIAL"),
                        help="isolate two persistent CTA counts with all other tuned options fixed")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.compare_ctas is not None:
        if args.profile:
            parser.error("--compare-ctas cannot be combined with --profile")
        if len(set(args.compare_ctas)) != 2 or any(count <= 0 or count % 8 for count in args.compare_ctas):
            parser.error("--compare-ctas requires two distinct positive multiples of 8")
        compare_persistent_ctas(args.compare_ctas, args.rounds, args.seed)
        return
    tuned_key = "4stage_bn128_tuned"
    if args.profile:
        comparison.run_test(
            tokens=16384, model_dim=6144, inter_dim=256, experts=384,
            topk=8, block_m=256, num_oc_splits=1, seed=args.seed,
            candidates=[tuned_key], profile=True,
        )
        return

    comparison.CANDIDATES["pyhip_after"] = ("PyHIP after", 64, None)
    measurements = []
    for index in range(args.rounds):
        # Rotate the first candidate; also measure PyHIP after the tuned run.
        keys = ["pyhip", tuned_key, "pyhip_after"] if index % 2 == 0 else [tuned_key, "pyhip", "pyhip_after"]
        results = comparison.run_test(
            tokens=16384, model_dim=6144, inter_dim=256, experts=384,
            topk=8, block_m=256, num_oc_splits=1, seed=args.seed, candidates=keys,
        )
        by_name = {row["name"]: row for row in results}
        first = by_name["PyHIP"]["elapsed_us"]
        last = by_name["PyHIP after"]["elapsed_us"]
        tuned = by_name[comparison.CANDIDATES[tuned_key][0]]["elapsed_us"]
        baseline = (first + last) / 2
        measurements.append((baseline, tuned))
        print(f"PAIRED_ROUND {index + 1} pyhip_a={first:.6f} pyhip_b={last:.6f} "
              f"tuned={tuned:.6f} speedup={baseline / tuned:.6f} "
              f"latency_drop_pct={(1 - tuned / baseline) * 100:.6f}", flush=True)

    baseline = statistics.mean(row[0] for row in measurements)
    tuned = statistics.mean(row[1] for row in measurements)
    print(f"PAIRED_AGGREGATE pyhip_us={baseline:.6f} tuned_us={tuned:.6f} "
          f"speedup={baseline / tuned:.6f} latency_drop_pct={(1 - tuned / baseline) * 100:.6f}", flush=True)


if __name__ == "__main__":
    main()