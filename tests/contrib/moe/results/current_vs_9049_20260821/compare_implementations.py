#!/usr/bin/env python3
import argparse
import builtins
import json
import os
import statistics
import sys
from pathlib import Path

import torch

if not hasattr(builtins, "FusedMoeImplSpec"):
    builtins.FusedMoeImplSpec = object

CURRENT_ROOT = Path("/tmp/pyhip-m64n512-rebased-tune")
sys.path.insert(0, str(CURRENT_ROOT / "tests/contrib/moe"))
sys.path.insert(0, str(CURRENT_ROOT / "src"))

import benchmark_down_variants as bench  # noqa: E402

OLD_PATH = Path("/tmp/pyhip-9049ddb/src/contrib/flydsl/moe_gemm_splitk.py")
CURRENT_PATH = CURRENT_ROOT / "src/contrib/flydsl/moe_gemm_splitk.py"
LABELS = ("old_9049", "current_59dd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(bench.MODELS), required=True)
    parser.add_argument("--physical-device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    physical_device = args.physical_device
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(physical_device):
        raise RuntimeError(f"HIP_VISIBLE_DEVICES must be {physical_device}")

    config = dict(bench.MODELS[args.model])
    old = bench.load_isolated_module(f"{args.model}_9049", OLD_PATH)
    current = bench.load_isolated_module(f"{args.model}_current", CURRENT_PATH)
    original = bench.state_helper.read_gpu_state(
        physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
    )
    if (
        original["performance_level"] != "auto"
        or original["gpu_busy_percent"] > 5
        or original["vram_allocated_percent"] > 20
    ):
        raise RuntimeError(f"GPU is not idle: {original}")

    result = None
    try:
        bench.state_helper.set_experiment_state(
            physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        managed = bench.state_helper.read_gpu_state(
            physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        if (
            managed["performance_level"] != "perf_determinism"
            or managed["ptl_state"] != "Enabled"
            or managed["ptl_format"] != "VECTOR,F8"
        ):
            raise RuntimeError(f"bad managed state: {managed}")

        metadata = bench.build_metadata(current, config, bench.DEFAULT_BATCH)
        metas = {
            "old_9049": metadata["m64"] if args.model == "hy3" else metadata["paired"],
            "current_59dd": metadata["paired"],
        }
        activations, weights, weight_scales, activation_scales = bench.build_data(
            config, 10, bench.DEFAULT_BATCH
        )
        common = dict(
            N=config["n"],
            K=config["k"],
            weight_dtype="fp8",
            weight_quant_type=config["quant_type"],
            act_quant_type=config["quant_type"],
            TOPK=config["topk"],
            BLOCK_TILE_SIZE_M=bench.BLOCK_M,
            BLOCK_TILE_SIZE_N=512,
            stage="down",
            alg="prefill_1x4",
            E=config["experts"],
            USE_ATOMIC_WRITE=False,
            down_physical_n512=True,
            down_paired_row_major=True,
            down_output_padding_bytes=0,
        )
        launches = {
            "old_9049": old.compile_gemm(
                **common,
                down_single_m_n512=args.model == "hy3",
            ),
            "current_59dd": current.compile_gemm(**common),
        }
        sums = {
            "old_9049": old.sorted_sum(config["topk"], config["n"], 0),
            "current_59dd": current.sorted_sum(config["topk"], config["n"], 0),
        }
        launch_gate = bench.build_gate(
            current, config, metadata["m64"], bench.DEFAULT_BATCH
        )
        stream = torch.cuda.current_stream()
        max_rows = max(meta["rows"] for meta in metas.values())
        physical = [
            torch.empty(max_rows * config["n"], dtype=torch.bfloat16, device="cuda")
            for _ in range(10)
        ]
        reduced = [
            torch.empty(
                bench.DEFAULT_BATCH,
                config["n"],
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(10)
        ]
        check_physical = torch.empty_like(physical[0])
        check_reduced = torch.empty_like(reduced[0])

        def launch_down(label, index, output=None):
            meta = metas[label]
            launches[label](
                bench.ptr(activations[index]),
                bench.ptr(weights[index]),
                bench.ptr(physical[index] if output is None else output),
                bench.ptr(meta["ids"]),
                bench.ptr(meta["weights"]),
                bench.ptr(meta["experts"]),
                bench.ptr(meta["valid"]),
                bench.ptr(weight_scales[index]),
                bench.ptr(activation_scales[index]),
                bench.DEFAULT_BATCH,
                meta["task_num"],
                stream,
            )

        def launch_sum(label, index, source=None, output=None):
            sums[label](
                metas[label]["loc"],
                physical[index] if source is None else source,
                reduced[index] if output is None else output,
                bench.DEFAULT_BATCH,
            )

        correctness = []
        for index in range(10):
            launch_down("old_9049", index)
            launch_sum("old_9049", index)
            launch_down("current_59dd", index, check_physical)
            launch_sum("current_59dd", index, check_physical, check_reduced)
            torch.cuda.synchronize()
            rel_l2 = bench.relative_l2(check_reduced, reduced[index])
            if rel_l2 > 5e-3:
                raise AssertionError(f"reduced mismatch: {rel_l2}")
            correctness.append(rel_l2)

        for index in range(10):
            for label in LABELS:
                launch_gate()
                launch_down(label, index)
                launch_sum(label, index)
        torch.cuda.synchronize()

        orders = (
            ("old_9049", "current_59dd", "current_59dd", "old_9049"),
            ("current_59dd", "old_9049", "old_9049", "current_59dd"),
        )

        def measure(combined):
            samples = {label: [] for label in LABELS}
            ratios = []
            call_index = 0
            for round_index in range(24):
                events = []
                for label in orders[round_index % 2]:
                    index = call_index % 10
                    call_index += 1
                    launch_gate()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch_down(label, index)
                    if combined:
                        launch_sum(label, index)
                    end.record()
                    events.append((label, start, end))
                torch.cuda.synchronize()
                current_round = {label: [] for label in LABELS}
                for label, start, end in events:
                    elapsed = start.elapsed_time(end)
                    samples[label].append(elapsed)
                    current_round[label].append(elapsed)
                ratios.append(
                    statistics.mean(current_round["current_59dd"])
                    / statistics.mean(current_round["old_9049"])
                )
            return {
                "versions": {
                    label: bench.summarize(samples[label]) for label in LABELS
                },
                "current_over_old": bench.summarize(ratios),
            }

        result = {
            "model": args.model,
            "shape": {
                "batch": bench.DEFAULT_BATCH,
                **config,
                "block_m": bench.BLOCK_M,
            },
            "sources": {
                "old_9049": bench.sha256(OLD_PATH),
                "current_59dd": bench.sha256(CURRENT_PATH),
            },
            "metadata": {
                label: {
                    "tasks": meta["task_num"],
                    "rows": meta["rows"],
                    "padded_rows": meta["padded_rows"],
                }
                for label, meta in metas.items()
            },
            "max_reduced_rel_l2": max(correctness),
            "initial_state": bench.compact_state(original),
            "managed_state": bench.compact_state(managed),
            "phases": {"down": measure(False), "combined": measure(True)},
        }
    finally:
        bench.state_helper.restore_experiment_state(
            physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT, original
        )
        restored = bench.state_helper.read_gpu_state(
            physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )

    for key in ("performance_level", "ptl_state", "ptl_format", "numa_balancing"):
        if restored[key] != original[key]:
            raise RuntimeError(
                f"restore mismatch {key}: {original[key]} -> {restored[key]}"
            )
    result["restored_state"] = bench.compact_state(restored)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
