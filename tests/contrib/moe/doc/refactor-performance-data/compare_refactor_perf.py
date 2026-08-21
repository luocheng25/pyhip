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

REPO = Path("/root/workspace/luocheng/pyhip")
sys.path.insert(0, str(REPO / "tests/contrib/moe"))

import benchmark_down_variants as bench

OLD_9049 = Path("/tmp/pyhip-9049ddb/src/contrib/flydsl/moe_gemm_splitk.py")
OLD_59DD = Path("/tmp/pyhip-59dd-stash-control/src/contrib/flydsl/moe_gemm_splitk.py")
NEW = REPO / "src/contrib/flydsl/moe_gemm_splitk.py"
BATCH = 32768
BUFFERS = 2
CASES = {
    "n256": dict(model="hy3", old=OLD_9049, metadata="m64", mode="n256"),
    "paired": dict(model="h3", old=OLD_9049, metadata="paired", mode="paired"),
    "true8": dict(model="hy3", old=OLD_9049, metadata="m64", mode="true8"),
    "nsplit192": dict(model="hy3", old=OLD_59DD, metadata="paired", mode="nsplit"),
    "nsplit": dict(model="h3", old=OLD_59DD, metadata="paired", mode="nsplit"),
    "qwen397": dict(model="qwen35_397B", old=OLD_59DD, metadata="paired", mode="nsplit"),
    "qwen35": dict(model="qwen35_35B", old=OLD_59DD, metadata="paired", mode="nsplit"),
    "persistent": dict(model="xiaomi", old=OLD_59DD, metadata="paired", mode="persistent"),
}
LABELS = ("old", "new")


def compile_path(module, config, mode, is_new):
    common = dict(
        N=config["n"],
        K=config["k"],
        weight_dtype="fp8",
        weight_quant_type=config["quant_type"],
        act_quant_type=config["quant_type"],
        TOPK=config["topk"],
        BLOCK_TILE_SIZE_M=bench.BLOCK_M,
        stage="down",
        alg="prefill_1x4",
        E=config["experts"],
        USE_ATOMIC_WRITE=False,
    )
    if mode == "n256":
        return module.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=256,
            down_physical_n256=True,
            down_output_padding_bytes=0,
        )
    kwargs = dict(
        BLOCK_TILE_SIZE_N=512,
        down_physical_n512=True,
        down_paired_row_major=True,
        down_output_padding_bytes=0,
    )
    if mode == "true8":
        kwargs["down_single_m_n512"] = True
    elif is_new and mode in ("nsplit", "persistent"):
        kwargs["down_nsplit_n512"] = True
        kwargs["expanded_m64_tasks"] = True
    elif is_new and mode == "paired":
        kwargs["expanded_m64_tasks"] = True
    return module.compile_gemm(**common, **kwargs)


def summarize(values):
    q1, _, q3 = statistics.quantiles(values, n=4)
    return {
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "min": min(values),
        "max": max(values),
        "samples": len(values),
        "raw": values,
    }


def run(args):
    case = CASES[args.case]
    config = dict(bench.MODELS[case["model"]])
    old = bench.load_isolated_module(f"refactor_{args.case}_old", case["old"])
    new = bench.load_isolated_module(f"refactor_{args.case}_new", NEW)
    modules = {"old": old, "new": new}

    state = bench.state_helper.read_gpu_state(
        args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
    )
    if state["performance_level"] != "auto" or state["gpu_busy_percent"] > 5:
        raise RuntimeError(f"GPU is not idle enough for paired testing: {state}")
    contaminated = state["vram_allocated_percent"] > 20

    managed = None
    restored = None
    result = None
    try:
        bench.state_helper.set_experiment_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        managed = bench.state_helper.read_gpu_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        if (
            managed["performance_level"] != "perf_determinism"
            or managed["ptl_state"] != "Enabled"
            or managed["ptl_format"] != "VECTOR,F8"
        ):
            raise RuntimeError(f"failed to enter managed state: {managed}")

        metadata = bench.build_metadata(new, config, BATCH)
        meta = metadata[case["metadata"]]
        activations, weights, weight_scales, activation_scales = bench.build_data(
            config, BUFFERS, BATCH
        )
        launches = {
            "old": compile_path(old, config, case["mode"], False),
            "new": compile_path(new, config, case["mode"], True),
        }
        sums = {
            label: modules[label].sorted_sum(config["topk"], config["n"], 0)
            for label in LABELS
        }
        launch_gate = bench.build_gate(new, config, metadata["m64"], BATCH)
        stream = torch.cuda.current_stream()

        numel = meta["rows"] * config["n"]
        physical = [
            torch.empty(numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(BUFFERS)
        ]
        reduced = [
            torch.empty(BATCH, config["n"], dtype=torch.bfloat16, device="cuda")
            for _ in range(BUFFERS)
        ]

        def launch_down(label, index):
            launches[label](
                bench.ptr(activations[index]),
                bench.ptr(weights[index]),
                bench.ptr(physical[index]),
                bench.ptr(meta["ids"]),
                bench.ptr(meta["weights"]),
                bench.ptr(meta["experts"]),
                bench.ptr(meta["valid"]),
                bench.ptr(weight_scales[index]),
                bench.ptr(activation_scales[index]),
                BATCH,
                meta["task_num"],
                stream,
            )

        def launch_sum(label, index):
            sums[label](meta["loc"], physical[index], reduced[index], BATCH)

        sentinel = -123.0
        physical[0].fill_(sentinel)
        launch_down("old", 0)
        launch_sum("old", 0)
        torch.cuda.synchronize()
        old_physical = physical[0][: meta["padded_rows"] * config["n"]].clone()
        old_reduced = reduced[0].clone()

        physical[1].fill_(sentinel)
        launch_down("new", 1)
        launch_sum("new", 1)
        torch.cuda.synchronize()
        new_physical = physical[1][: meta["padded_rows"] * config["n"]]
        physical_equal = bool(torch.equal(old_physical, new_physical))
        reduced_equal = bool(torch.equal(old_reduced, reduced[1]))
        rel_l2 = bench.relative_l2(reduced[1], old_reduced)
        tail_old = physical[0][meta["padded_rows"] * config["n"] :]
        tail_new = physical[1][meta["padded_rows"] * config["n"] :]
        tail_clean = bool(
            (tail_old.numel() == 0 or torch.all(tail_old == sentinel).item())
            and (tail_new.numel() == 0 or torch.all(tail_new == sentinel).item())
        )
        if not tail_clean or rel_l2 > 5e-3:
            raise AssertionError(
                f"correctness failed physical={physical_equal} reduced={reduced_equal} "
                f"rel_l2={rel_l2} tail={tail_clean}"
            )
        del old_physical, old_reduced

        for index in range(BUFFERS):
            for label in LABELS:
                launch_gate()
                launch_down(label, index)
                launch_sum(label, index)
        torch.cuda.synchronize()

        orders = (
            ("old", "new", "new", "old"),
            ("new", "old", "old", "new"),
        )

        def measure(combined):
            samples = {label: [] for label in LABELS}
            ratios = []
            call_index = 0
            for round_index in range(args.rounds):
                events = []
                for label in orders[round_index % 2]:
                    index = call_index % BUFFERS
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
                current = {label: [] for label in LABELS}
                for label, start, end in events:
                    elapsed = start.elapsed_time(end)
                    samples[label].append(elapsed)
                    current[label].append(elapsed)
                ratios.append(statistics.mean(current["new"]) / statistics.mean(current["old"]))
            return {
                "versions": {label: summarize(samples[label]) for label in LABELS},
                "new_over_old": summarize(ratios),
            }

        pre_measure = bench.state_helper.read_gpu_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        phases = {"down": measure(False), "combined": measure(True)}
        post_measure = bench.state_helper.read_gpu_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )
        if pre_measure["gpu_busy_percent"] > 5:
            raise RuntimeError(
                f"external GPU activity detected: pre={pre_measure} post={post_measure}"
            )
        result = {
            "case": args.case,
            "model": case["model"],
            "shape": {"batch": BATCH, **config},
            "sources": {
                "old": {"path": str(case["old"]), "sha256": bench.sha256(case["old"])},
                "new": {"path": str(NEW), "sha256": bench.sha256(NEW)},
            },
            "rounds": args.rounds,
            "buffers": BUFFERS,
            "contaminated_vram": contaminated,
            "metadata": {
                "kind": case["metadata"],
                "tasks": meta["task_num"],
                "rows": meta["rows"],
                "padded_rows": meta["padded_rows"],
            },
            "correctness": {
                "physical_equal": physical_equal,
                "reduced_equal": reduced_equal,
                "reduced_rel_l2": rel_l2,
                "tail_clean": tail_clean,
            },
            "initial_state": bench.compact_state(state),
            "managed_state": bench.compact_state(managed),
            "pre_measure_state": bench.compact_state(pre_measure),
            "post_measure_state": bench.compact_state(post_measure),
            "phases": phases,
        }
    finally:
        bench.state_helper.restore_experiment_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT, state
        )
        restored = bench.state_helper.read_gpu_state(
            args.physical_device, bench.state_helper.DEFAULT_AMDSMI_ROOT
        )

    for key in ("performance_level", "ptl_state", "ptl_format", "numa_balancing"):
        if restored[key] != state[key]:
            raise RuntimeError(f"restore mismatch {key}: {state[key]} -> {restored[key]}")
    result["restored_state"] = bench.compact_state(restored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--physical-device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be >= 2")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_device):
        parser.error("HIP_VISIBLE_DEVICES must match --physical-device")
    result = run(args)
    summary = {
        phase: {
            "old_ms": data["versions"]["old"]["median"],
            "new_ms": data["versions"]["new"]["median"],
            "new_over_old": data["new_over_old"]["median"],
            "ratio_q1": data["new_over_old"]["q1"],
            "ratio_q3": data["new_over_old"]["q3"],
        }
        for phase, data in result["phases"].items()
    }
    print(json.dumps({"case": args.case, "correctness": result["correctness"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
