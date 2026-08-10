"""Validate H3 FP8 auto-DPM data and compare it with the BF16 auto profile."""

import argparse
import json
import statistics
from pathlib import Path


EXPECTED = ("8wave_varlen_fp8", "4wave_varlen_fp8")
BF16_NAMES = ("8wave_32x32", "4wave_varlen")


def summarize(rows):
    tflops = [row["h3_tflops"] for row in rows]
    elapsed = [row["elapsed_ms"] for row in rows]
    return {
        "sample_count": len(rows),
        "mean_elapsed_ms": statistics.mean(elapsed),
        "min_elapsed_ms": min(elapsed),
        "max_elapsed_ms": max(elapsed),
        "mean_h3_tflops": statistics.mean(tflops),
        "min_h3_tflops": min(tflops),
        "max_h3_tflops": max(tflops),
        "tflops_cv_percent": statistics.pstdev(tflops) / statistics.mean(tflops) * 100,
        "mean_sclk_mhz": statistics.mean(row["sclk_mean_mhz"] for row in rows),
        "mean_power_w": statistics.mean(row["power_mean_w"] for row in rows),
        "max_power_w": max(row["power_max_w"] for row in rows),
        "max_junction_c": max(row["junction_max_c"] for row in rows),
        "max_mem_c": max(row["mem_max_c"] for row in rows),
    }


def validate_result(result, name):
    if result["name"] != name:
        raise ValueError(f"expected {name}, got {result['name']}")
    rows = result["dispatches"]
    if len(rows) != 70:
        raise ValueError(f"{name} has {len(rows)} dispatches instead of 70")
    if [row["index"] for row in rows] != list(range(70)):
        raise ValueError(f"{name} indices are not 0..69")
    if not all(
        row["sensor_count"] > 0 and row["elapsed_ms"] > 0 and row["h3_tflops"] > 0
        for row in rows
    ):
        raise ValueError(f"{name} contains an invalid dispatch")


def load_bf16(path):
    data = json.loads(path.read_text())
    results = {result["name"]: summarize(result["dispatches"]) for result in data["results"]}
    missing = set(BF16_NAMES) - set(results)
    if missing:
        raise ValueError(f"BF16 profile is missing {sorted(missing)}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--bf16-profile",
        type=Path,
        default=Path("artifacts/h3-five-kernel/profile-auto-flydsl.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.profile.read_text())
    if data.get("schema_version") != 2:
        raise ValueError("FP8 profile is not schema v2")
    if data.get("warmup") != 3 or data.get("iters") != 70:
        raise ValueError("FP8 profile does not use warmup=3 and iters=70")
    environment = data["environment"]
    if environment["profile"].get("h3_input_dtype") != "fp8_e4m3fnuz":
        raise ValueError("FP8 profile does not record h3_input_dtype=fp8_e4m3fnuz")
    if environment["gpu"]["dpm_force_performance_level"] != "auto":
        raise ValueError("FP8 profile was not collected under auto DPM")
    names = tuple(result["name"] for result in data["results"])
    if names != EXPECTED:
        raise ValueError(f"FP8 profile has implementations {names}, expected {EXPECTED}")

    bf16 = load_bf16(args.bf16_profile)
    implementations = []
    for result, name, bf16_name in zip(data["results"], EXPECTED, BF16_NAMES):
        validate_result(result, name)
        summary = summarize(result["dispatches"])
        baseline = bf16[bf16_name]
        summary["mean_tflops_vs_bf16_percent"] = (
            summary["mean_h3_tflops"] / baseline["mean_h3_tflops"] - 1
        ) * 100
        summary["mean_latency_vs_bf16_percent"] = (
            summary["mean_elapsed_ms"] / baseline["mean_elapsed_ms"] - 1
        ) * 100
        implementations.append(
            {"name": name, "fp8": summary, "bf16_baseline_name": bf16_name, "bf16": baseline}
        )
        print(
            f"{name}: mean={summary['mean_elapsed_ms']:.3f}ms/"
            f"{summary['mean_h3_tflops']:.3f}T "
            f"range={summary['min_h3_tflops']:.3f}-{summary['max_h3_tflops']:.3f}T "
            f"cv={summary['tflops_cv_percent']:.2f}% "
            f"vs_bf16={summary['mean_tflops_vs_bf16_percent']:+.2f}%"
        )

    output = {
        "schema_version": 1,
        "profile": str(args.profile),
        "bf16_profile": str(args.bf16_profile),
        "implementations": implementations,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"analysis_output={args.output}")


if __name__ == "__main__":
    main()