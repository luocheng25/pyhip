"""分析 profile_h3_attention_throttle.py 生成的逐 dispatch JSON。

仅使用原始顺序、峰谷、相关性和高吞吐 burst 间隔，不计算中值。
"""

import argparse
import json
import math
from pathlib import Path


def correlation(lhs, rhs):
    if len(lhs) < 2:
        return None
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    numerator = sum(
        (lhs_value - lhs_mean) * (rhs_value - rhs_mean)
        for lhs_value, rhs_value in zip(lhs, rhs)
    )
    lhs_square_sum = sum((value - lhs_mean) ** 2 for value in lhs)
    rhs_square_sum = sum((value - rhs_mean) ** 2 for value in rhs)
    denominator = math.sqrt(lhs_square_sum * rhs_square_sum)
    return numerator / denominator if denominator else None


def high_runs(values, threshold):
    runs = []
    for index, value in enumerate(values):
        if value < threshold:
            continue
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def wall_start_seconds(rows):
    if all("wall_start_seconds" in row for row in rows):
        return [row["wall_start_seconds"] for row in rows]

    starts = []
    elapsed_seconds = 0.0
    for row in rows:
        starts.append(elapsed_seconds)
        elapsed_seconds += row["elapsed_ms"] / 1e3
    return starts


def arithmetic_mean(values):
    return sum(values) / len(values) if values else None


def complete_cycles(rows, starts, wall_starts):
    cycles = []
    for cycle_index, (begin, end) in enumerate(zip(starts, starts[1:])):
        cycle_rows = rows[begin:end]
        cycles.append(
            {
                "cycle_index": cycle_index,
                "dispatch_start": begin,
                "dispatch_end": end - 1,
                "dispatch_count": len(cycle_rows),
                "period_seconds": wall_starts[end] - wall_starts[begin],
                "mean_elapsed_ms": arithmetic_mean(
                    [row["elapsed_ms"] for row in cycle_rows]
                ),
                "mean_h3_tflops": arithmetic_mean(
                    [row["h3_tflops"] for row in cycle_rows]
                ),
                "mean_sclk_mhz": arithmetic_mean(
                    [row["sclk_mean_mhz"] for row in cycle_rows]
                ),
                "mean_power_w": arithmetic_mean(
                    [row["power_mean_w"] for row in cycle_rows]
                ),
            }
        )
    return cycles


def analyze_result(result, high_fraction):
    rows = result["dispatches"]
    if not rows:
        raise ValueError(f"{result['name']} contains no dispatches")

    tflops = [row["h3_tflops"] for row in rows]
    sclk = [row["sclk_mean_mhz"] for row in rows]
    power = [row["power_mean_w"] for row in rows]
    minimum = min(tflops)
    maximum = max(tflops)
    threshold = minimum + (maximum - minimum) * high_fraction
    runs = high_runs(tflops, threshold)
    starts = [run[0] for run in runs]
    dispatch_intervals = [end - begin for begin, end in zip(starts, starts[1:])]
    wall_starts = wall_start_seconds(rows)
    time_intervals = [
        wall_starts[end] - wall_starts[begin] for begin, end in zip(starts, starts[1:])
    ]

    repeating_intervals = []
    if dispatch_intervals:
        interval_min = min(dispatch_intervals)
        interval_max = max(dispatch_intervals)
        repeating_intervals = [
            interval for interval in dispatch_intervals if interval_min <= interval <= interval_max
        ]
    cycle_detected = (
        len(starts) >= 4
        and bool(dispatch_intervals)
        and max(dispatch_intervals) - min(dispatch_intervals) <= 2
    )
    cycles = complete_cycles(rows, starts, wall_starts) if cycle_detected else []

    peak_index = max(range(len(rows)), key=lambda index: tflops[index])
    floor_index = min(range(len(rows)), key=lambda index: tflops[index])
    return {
        "name": result["name"],
        "sample_count": len(rows),
        "mean_elapsed_ms": arithmetic_mean([row["elapsed_ms"] for row in rows]),
        "mean_h3_tflops": arithmetic_mean(tflops),
        "mean_sclk_mhz": arithmetic_mean(sclk),
        "mean_power_w": arithmetic_mean(power),
        "peak": {
            "index": peak_index,
            "h3_tflops": tflops[peak_index],
            "elapsed_ms": rows[peak_index]["elapsed_ms"],
        },
        "floor": {
            "index": floor_index,
            "h3_tflops": tflops[floor_index],
            "elapsed_ms": rows[floor_index]["elapsed_ms"],
        },
        "peak_to_floor_drop_percent": (1.0 - minimum / maximum) * 100.0,
        "correlation_tflops_sclk": correlation(tflops, sclk),
        "correlation_tflops_power": correlation(tflops, power),
        "high_threshold_tflops": threshold,
        "high_runs": [[run[0], run[-1]] for run in runs],
        "high_run_starts": starts,
        "dispatch_intervals": dispatch_intervals,
        "time_intervals_seconds": time_intervals,
        "complete_cycles": cycles,
        "incomplete_tail": (
            {"dispatch_start": starts[-1], "dispatch_end": len(rows) - 1}
            if cycle_detected
            else None
        ),
        "cycle_detected": cycle_detected,
        "throttle_delta": result.get("throttle_delta", {}),
    }


def print_samples(data):
    print(
        "sample,impl,index,elapsed_ms,h3_tflops,sclk_mean_mhz,power_mean_w,"
        "junction_max_c,mem_max_c"
    )
    for result in data["results"]:
        for row in result["dispatches"]:
            print(
                f"sample,{result['name']},{row['index']},{row['elapsed_ms']:.3f},"
                f"{row['h3_tflops']:.3f},{row['sclk_mean_mhz']:.1f},"
                f"{row['power_mean_w']:.1f},{row['junction_max_c']:.1f},"
                f"{row['mem_max_c']:.1f}"
            )


def print_analysis(analyses):
    for analysis in analyses:
        peak = analysis["peak"]
        floor = analysis["floor"]
        print(f"\n[{analysis['name']}]")
        print(
            f"samples={analysis['sample_count']} "
            f"mean={analysis['mean_elapsed_ms']:.3f}ms/{analysis['mean_h3_tflops']:.3f}T "
            f"peak={peak['h3_tflops']:.3f}T@{peak['index']}/{peak['elapsed_ms']:.3f}ms "
            f"floor={floor['h3_tflops']:.3f}T@{floor['index']}/{floor['elapsed_ms']:.3f}ms "
            f"drop={analysis['peak_to_floor_drop_percent']:.2f}%"
        )
        sclk_correlation = analysis["correlation_tflops_sclk"]
        power_correlation = analysis["correlation_tflops_power"]
        print(
            "corr_tflops_sclk="
            + (f"{sclk_correlation:.4f}" if sclk_correlation is not None else "unavailable")
            + " corr_tflops_power="
            + (f"{power_correlation:.4f}" if power_correlation is not None else "unavailable")
        )
        print(
            f"high_threshold={analysis['high_threshold_tflops']:.3f}T "
            f"high_runs={analysis['high_runs']}"
        )
        print(f"dispatch_intervals={analysis['dispatch_intervals']}")
        print(
            "time_intervals_seconds="
            + str([round(value, 4) for value in analysis["time_intervals_seconds"]])
        )
        print(f"cycle_detected={analysis['cycle_detected']}")
        if not analysis["cycle_detected"]:
            print("complete_cycles=none (no repeatable cycle detected)")
        for cycle in analysis["complete_cycles"]:
            print(
                f"cycle={cycle['cycle_index']} "
                f"dispatches={cycle['dispatch_start']}-{cycle['dispatch_end']} "
                f"count={cycle['dispatch_count']} period={cycle['period_seconds']:.4f}s "
                f"mean={cycle['mean_elapsed_ms']:.3f}ms/{cycle['mean_h3_tflops']:.3f}T "
                f"sclk={cycle['mean_sclk_mhz']:.1f}MHz power={cycle['mean_power_w']:.1f}W"
            )
        if analysis["incomplete_tail"]:
            tail = analysis["incomplete_tail"]
            print(
                f"incomplete_tail={tail['dispatch_start']}-{tail['dispatch_end']} "
                "(excluded from complete-cycle summaries)"
            )
        print(f"throttle_delta={json.dumps(analysis['throttle_delta'], sort_keys=True)}")


def compare_profiles(baseline, candidate):
    baseline_results = {result["name"]: result for result in baseline["results"]}
    candidate_results = {result["name"]: result for result in candidate["results"]}
    common_names = [
        result["name"]
        for result in baseline["results"]
        if result["name"] in candidate_results
    ]
    if not common_names:
        raise ValueError("profiles have no implementations in common")

    comparisons = []
    print(
        "compare,impl,index,baseline_ms,candidate_ms,candidate_over_baseline_ms,"
        "baseline_h3_tflops,candidate_h3_tflops,candidate_over_baseline_tflops"
    )
    for name in common_names:
        baseline_rows = baseline_results[name]["dispatches"]
        candidate_rows = candidate_results[name]["dispatches"]
        if len(baseline_rows) != len(candidate_rows):
            raise ValueError(
                f"{name} sample count differs: {len(baseline_rows)} vs {len(candidate_rows)}"
            )
        for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
            if baseline_row["index"] != candidate_row["index"]:
                raise ValueError(
                    f"{name} indices differ: {baseline_row['index']} vs {candidate_row['index']}"
                )
            row = {
                "impl": name,
                "index": baseline_row["index"],
                "baseline_ms": baseline_row["elapsed_ms"],
                "candidate_ms": candidate_row["elapsed_ms"],
                "candidate_over_baseline_ms": (
                    candidate_row["elapsed_ms"] / baseline_row["elapsed_ms"]
                ),
                "baseline_h3_tflops": baseline_row["h3_tflops"],
                "candidate_h3_tflops": candidate_row["h3_tflops"],
                "candidate_over_baseline_tflops": (
                    candidate_row["h3_tflops"] / baseline_row["h3_tflops"]
                ),
            }
            comparisons.append(row)
            print(
                f"compare,{name},{row['index']},{row['baseline_ms']:.3f},"
                f"{row['candidate_ms']:.3f},{row['candidate_over_baseline_ms']:.6f},"
                f"{row['baseline_h3_tflops']:.3f},{row['candidate_h3_tflops']:.3f},"
                f"{row['candidate_over_baseline_tflops']:.6f}"
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="profiler 生成的 JSON")
    parser.add_argument(
        "--high-fraction",
        type=float,
        default=0.65,
        help="峰谷范围内判定高吞吐区间的位置，默认 0.65",
    )
    parser.add_argument("--analysis-json", type=Path, help="可选：保存分析结果 JSON")
    parser.add_argument(
        "--no-samples", action="store_true", help="不在终端重复打印全部逐次样本"
    )
    parser.add_argument("--compare", type=Path, help="可选：另一台机器生成的 JSON")
    parser.add_argument(
        "--comparison-json", type=Path, help="可选：保存逐 dispatch 对比 JSON"
    )
    args = parser.parse_args()
    if not 0.0 < args.high_fraction < 1.0:
        parser.error("--high-fraction must be between 0 and 1")

    data = json.loads(args.profile.read_text())
    if not data.get("results"):
        raise ValueError("profile contains no results")
    analyses = [
        analyze_result(result, args.high_fraction) for result in data["results"]
    ]

    if not args.no_samples:
        print_samples(data)
    print_analysis(analyses)

    if args.analysis_json:
        args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_json.write_text(json.dumps(analyses, indent=2))
        print(f"\nanalysis_output={args.analysis_json}")

    if args.compare:
        candidate = json.loads(args.compare.read_text())
        comparisons = compare_profiles(data, candidate)
        if args.comparison_json:
            args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
            args.comparison_json.write_text(json.dumps(comparisons, indent=2))
            print(f"comparison_output={args.comparison_json}")
    elif args.comparison_json:
        parser.error("--comparison-json requires --compare")


if __name__ == "__main__":
    main()