"""Analyze the H3 fixed-SCLK sweep and compare it with auto-DPM profiles."""

import argparse
import json
import statistics
from pathlib import Path


FREQUENCIES = tuple(range(1100, 1801, 100))
IMPLEMENTATIONS = (
    "triton",
    "asm_mi308",
    "asm_mi300",
    "8wave_32x32",
    "4wave_varlen",
)
PROFILE_GROUPS = {
    "aiter-mi308": ("triton", "asm_mi308"),
    "aiter-mi300": ("asm_mi300",),
    "flydsl": ("8wave_32x32", "4wave_varlen"),
}


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def summarize_rows(rows):
    tflops = [row["h3_tflops"] for row in rows]
    elapsed = [row["elapsed_ms"] for row in rows]
    sclk = [row["sclk_mean_mhz"] for row in rows]
    return {
        "sample_count": len(rows),
        "mean_elapsed_ms": statistics.mean(elapsed),
        "min_elapsed_ms": min(elapsed),
        "max_elapsed_ms": max(elapsed),
        "mean_h3_tflops": statistics.mean(tflops),
        "min_h3_tflops": min(tflops),
        "max_h3_tflops": max(tflops),
        "tflops_cv_percent": statistics.pstdev(tflops) / statistics.mean(tflops) * 100,
        "mean_sclk_mhz": statistics.mean(sclk),
        "min_dispatch_mean_sclk_mhz": min(sclk),
        "max_dispatch_mean_sclk_mhz": max(sclk),
        "mean_power_w": mean(rows, "power_mean_w"),
        "max_power_w": max(row["power_max_w"] for row in rows),
        "max_junction_c": max(row["junction_max_c"] for row in rows),
        "max_mem_c": max(row["mem_max_c"] for row in rows),
        "sensor_count_min": min(row["sensor_count"] for row in rows),
        "sensor_count_max": max(row["sensor_count"] for row in rows),
    }


def validate_result(result, expected_name):
    if result["name"] != expected_name:
        raise ValueError(f"expected {expected_name}, got {result['name']}")
    rows = result["dispatches"]
    if len(rows) != 70:
        raise ValueError(f"{expected_name} has {len(rows)} dispatches instead of 70")
    if [row["index"] for row in rows] != list(range(70)):
        raise ValueError(f"{expected_name} dispatch indices are not 0..69")
    if not all(
        row["sensor_count"] > 0 and row["elapsed_ms"] > 0 and row["h3_tflops"] > 0
        for row in rows
    ):
        raise ValueError(f"{expected_name} contains an invalid dispatch")


def load_auto_profiles(auto_dir):
    results = {}
    for path in sorted(auto_dir.glob("profile-auto-*.json")):
        data = json.loads(path.read_text())
        for result in data.get("results", []):
            results[result["name"]] = summarize_rows(result["dispatches"])
    missing = set(IMPLEMENTATIONS) - set(results)
    if missing:
        raise ValueError(f"auto profiles are missing {sorted(missing)}")
    return results


def load_frequency_profiles(sweep_dir):
    results = {name: [] for name in IMPLEMENTATIONS}
    source_hashes = set()
    for frequency in FREQUENCIES:
        for suffix, expected_names in PROFILE_GROUPS.items():
            path = sweep_dir / f"profile-{frequency}-{suffix}.json"
            data = json.loads(path.read_text())
            if data.get("schema_version") != 2:
                raise ValueError(f"{path} is not schema v2")
            if data.get("warmup") != 3 or data.get("iters") != 70:
                raise ValueError(f"{path} does not use warmup=3 and iters=70")
            profile = data["environment"]["profile"]
            if profile.get("requested_sclk_mhz") != frequency:
                raise ValueError(f"{path} does not record requested SCLK {frequency}")
            dpm = data["environment"]["gpu"]["dpm_force_performance_level"]
            if dpm != "perf_determinism":
                raise ValueError(f"{path} recorded DPM level {dpm!r}")
            names = tuple(result["name"] for result in data["results"])
            if names != expected_names:
                raise ValueError(f"{path} has implementations {names}, expected {expected_names}")
            source_hashes.add(
                data["environment"]["source_sha256"][
                    "tests/flydsl/profile_h3_attention_throttle.py"
                ]
            )
            for result, expected_name in zip(data["results"], expected_names):
                validate_result(result, expected_name)
                summary = summarize_rows(result["dispatches"])
                summary["requested_sclk_mhz"] = frequency
                results[expected_name].append(summary)
    if len(source_hashes) != 1:
        raise ValueError(f"frequency profiles use different profiler hashes: {source_hashes}")
    return results, source_hashes.pop()


def add_comparisons(auto_results, sweep_results):
    implementations = []
    for name in IMPLEMENTATIONS:
        auto = auto_results[name]
        points = sweep_results[name]
        for point in points:
            point["mean_tflops_vs_auto_percent"] = (
                point["mean_h3_tflops"] / auto["mean_h3_tflops"] - 1.0
            ) * 100.0
            point["mean_latency_vs_auto_percent"] = (
                point["mean_elapsed_ms"] / auto["mean_elapsed_ms"] - 1.0
            ) * 100.0
        best = max(points, key=lambda point: point["mean_h3_tflops"])
        most_stable = min(points, key=lambda point: point["tflops_cv_percent"])
        implementations.append(
            {
                "name": name,
                "auto": auto,
                "points": points,
                "best_mean_tflops_point": best,
                "most_stable_point": most_stable,
            }
        )
    return implementations


def print_summary(implementations):
    print(
        "implementation,auto_tflops,best_requested_sclk_mhz,best_tflops,"
        "best_vs_auto_percent,best_actual_sclk_mhz,auto_cv_percent,best_cv_percent"
    )
    for result in implementations:
        auto = result["auto"]
        best = result["best_mean_tflops_point"]
        print(
            f"{result['name']},{auto['mean_h3_tflops']:.3f},"
            f"{best['requested_sclk_mhz']},{best['mean_h3_tflops']:.3f},"
            f"{best['mean_tflops_vs_auto_percent']:.3f},{best['mean_sclk_mhz']:.1f},"
            f"{auto['tflops_cv_percent']:.3f},{best['tflops_cv_percent']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("artifacts/h3-frequency-sweep"),
        help="directory containing profile-<MHz>-*.json",
    )
    parser.add_argument(
        "--auto-dir",
        type=Path,
        default=Path("artifacts/h3-five-kernel"),
        help="directory containing profile-auto-*.json",
    )
    parser.add_argument("--output", type=Path, help="optional analysis JSON output")
    args = parser.parse_args()

    auto_results = load_auto_profiles(args.auto_dir)
    sweep_results, profiler_sha256 = load_frequency_profiles(args.sweep_dir)
    implementations = add_comparisons(auto_results, sweep_results)
    output = {
        "schema_version": 1,
        "frequencies_mhz": FREQUENCIES,
        "warmup": 3,
        "iters": 70,
        "profile_count": len(FREQUENCIES) * len(PROFILE_GROUPS),
        "implementation_point_count": len(FREQUENCIES) * len(IMPLEMENTATIONS),
        "dispatch_count": len(FREQUENCIES) * len(IMPLEMENTATIONS) * 70,
        "profiler_sha256": profiler_sha256,
        "implementations": implementations,
    }
    print_summary(implementations)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"analysis_output={args.output}")


if __name__ == "__main__":
    main()