#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import statistics
import sys
import types
from pathlib import Path

import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx

STATE_HELPER_DIR = Path("/root/workspace/luocheng/pyhip/tests/contrib/moe")
sys.path.insert(0, str(STATE_HELPER_DIR))
state_helper = importlib.import_module("probe_control_k128_hardware")

BASE_MODULE = Path("/tmp/pyhip-e6fe8e9-base/src/contrib/flydsl/moe_gemm_splitk.py")
CONTROL_MODULE = Path(
    "/tmp/pyhip-a452743-rowmajor-control/src/contrib/flydsl/moe_gemm_splitk.py"
)
NSPLIT_MODULE = Path(
    "/tmp/pyhip-m64n512-rebased-tune/src/contrib/flydsl/moe_gemm_splitk.py"
)
DEFAULT_BATCH = 32768
BLOCK_M = 64
FP8 = torch.float8_e4m3fnuz
EXPECTED_POWER_CAP_W = 650
LABELS = ("base_e6", "n256_4wave", "m128n256_8wave", "m64n512_8wave")
MODELS = {
    "hy3": dict(n=4096, k=192, experts=193, topk=9, quant_type="per_tensor", padding=0),
    "qwen35_397B": dict(
        n=4096, k=512, experts=512, topk=10, quant_type="ptpc", padding=128
    ),
    "qwen35_35B": dict(
        n=2048, k=512, experts=256, topk=8, quant_type="ptpc", padding=128
    ),
    "xiaomi": dict(n=6144, k=256, experts=384, topk=8, quant_type="ptpc", padding=0),
    "h3": dict(n=6144, k=384, experts=128, topk=4, quant_type="ptpc", padding=128),
}
_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
}


def load_isolated_module(tag, path):
    path = path.resolve()
    package_name = f"_moe_down_bench_{tag}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    helper_name = f"{package_name}.helpers"
    helper_spec = importlib.util.spec_from_file_location(
        helper_name, path.parent / "helpers.py"
    )
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"cannot load helpers for {path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper_module
    helper_spec.loader.exec_module(helper_module)

    module_name = f"{package_name}.moe_gemm_splitk"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ptr(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def relative_l2(actual, expected):
    error = (actual.float() - expected.float()).square().sum().sqrt()
    reference = expected.float().square().sum().sqrt()
    return (error / reference).item()


def compact_state(state):
    keys = (
        "gpu_busy_percent",
        "vram_allocated_percent",
        "performance_level",
        "sclk",
        "ptl_state",
        "ptl_format",
        "power_cap_w",
        "numa_balancing",
    )
    return {key: state[key] for key in keys}


def build_metadata(candidate, config, batch):
    useful_rows = batch * config["topk"]
    topk_ids = (
        torch.arange(useful_rows, dtype=torch.int32, device="cuda")
        .remainder(config["experts"])
        .reshape(batch, config["topk"])
    )
    topk_weights = torch.linspace(
        0.5, 1.0, useful_rows, dtype=torch.float32, device="cuda"
    ).reshape(batch, config["topk"])

    sorted64 = moe_sorting(
        topk_ids,
        topk_weights,
        config["experts"],
        config["n"],
        torch.bfloat16,
        BLOCK_M,
        None,
        None,
        0,
    )
    sorted128 = moe_sorting(
        topk_ids,
        topk_weights,
        config["experts"],
        config["n"],
        torch.bfloat16,
        2 * BLOCK_M,
        None,
        None,
        0,
    )

    ids64, weights64, experts64, valid64, _ = sorted64
    ids128, weights128, experts128, valid128, _ = sorted128
    experts128_m64 = experts128.repeat_interleave(2)
    loc64 = torch.zeros(batch, config["topk"], dtype=torch.int32, device="cuda")
    loc128 = torch.zeros_like(loc64)
    invert = candidate.invert_sorted_ids(config["topk"])
    invert(ids64, loc64, valid64, ids64.shape[0], batch)
    invert(ids128, loc128, valid128, ids128.shape[0], batch)
    torch.cuda.synchronize()

    return {
        "m64": dict(
            ids=ids64,
            weights=weights64,
            experts=experts64,
            valid=valid64,
            loc=loc64,
            task_num=experts64.shape[0],
            rows=experts64.shape[0] * BLOCK_M,
            padded_rows=int(valid64[0].item()),
        ),
        "paired": dict(
            ids=ids128,
            weights=weights128,
            experts=experts128_m64,
            valid=valid128,
            loc=loc128,
            task_num=experts128_m64.shape[0],
            rows=experts128_m64.shape[0] * BLOCK_M,
            padded_rows=int(valid128[0].item()),
        ),
        "useful_rows": useful_rows,
    }


def build_data(config, buffer_copies, batch):
    torch.manual_seed(20260820)
    activation_base = (
        0.125
        * torch.randn(
            batch,
            config["topk"],
            config["k"],
            dtype=torch.float16,
            device="cuda",
        )
    ).to(FP8)
    weight_base = shuffle_weight(
        torch.ones(
            config["experts"],
            config["n"],
            config["k"],
            dtype=FP8,
            device="cuda",
        ),
        layout=(16, 16),
    )
    activations = [activation_base.clone() for _ in range(buffer_copies)]
    weights = [weight_base.clone() for _ in range(buffer_copies)]
    del activation_base, weight_base

    if config["quant_type"] == "ptpc":
        weight_scale_base = torch.linspace(
            0.001,
            0.002,
            config["experts"] * config["n"],
            dtype=torch.float32,
            device="cuda",
        ).reshape(config["experts"], config["n"])
        activation_scale_base = torch.linspace(
            0.01,
            0.02,
            batch * config["topk"],
            dtype=torch.float32,
            device="cuda",
        ).reshape(batch, config["topk"])
    else:
        weight_scale_base = torch.linspace(
            0.001,
            0.002,
            config["experts"],
            dtype=torch.float32,
            device="cuda",
        )
        activation_scale_base = torch.tensor(
            [0.015], dtype=torch.float32, device="cuda"
        )
    weight_scales = [weight_scale_base.clone() for _ in range(buffer_copies)]
    activation_scales = [activation_scale_base.clone() for _ in range(buffer_copies)]
    del weight_scale_base, activation_scale_base
    return activations, weights, weight_scales, activation_scales


def compile_paths(base, control, candidate, config):
    common = dict(
        N=config["n"],
        K=config["k"],
        weight_dtype="fp8",
        weight_quant_type=config["quant_type"],
        act_quant_type=config["quant_type"],
        TOPK=config["topk"],
        BLOCK_TILE_SIZE_M=BLOCK_M,
        stage="down",
        alg="prefill_1x4",
        E=config["experts"],
        USE_ATOMIC_WRITE=False,
    )
    launches = {
        "base_e6": base.compile_gemm(**common, BLOCK_TILE_SIZE_N=64),
        "n256_4wave": control.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=256,
            down_physical_n256=True,
            down_output_padding_bytes=config["padding"],
        ),
        "m128n256_8wave": control.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=512,
            down_physical_n512=True,
            down_paired_row_major=True,
            down_output_padding_bytes=0,
        ),
        "m64n512_8wave": candidate.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=512,
            down_physical_n512=True,
            down_paired_row_major=True,
            down_output_padding_bytes=0,
        ),
    }
    sums = {
        "base_e6": base.sorted_sum(config["topk"], config["n"]),
        "n256_4wave": control.sorted_sum(
            config["topk"], config["n"], config["padding"]
        ),
        "m128n256_8wave": control.sorted_sum(config["topk"], config["n"], 0),
        "m64n512_8wave": candidate.sorted_sum(config["topk"], config["n"], 0),
    }
    return launches, sums


def build_gate(control, config, meta64, batch):
    gate_n = 2 * config["k"]
    gate_input = torch.ones(batch, config["n"], dtype=FP8, device="cuda")
    gate_weight = shuffle_weight(
        torch.ones(config["experts"], gate_n, config["n"], dtype=FP8, device="cuda"),
        layout=(16, 16),
    )
    gate_output = torch.empty(
        batch,
        config["topk"],
        config["k"],
        dtype=torch.bfloat16,
        device="cuda",
    )
    if config["quant_type"] == "ptpc":
        gate_weight_scale = torch.ones(
            config["experts"], gate_n, dtype=torch.float32, device="cuda"
        )
        gate_input_scale = torch.ones(batch, 1, dtype=torch.float32, device="cuda")
    else:
        gate_weight_scale = torch.ones(
            config["experts"], dtype=torch.float32, device="cuda"
        )
        gate_input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    gate = control.compile_gemm(
        N=gate_n,
        K=config["n"],
        weight_dtype="fp8",
        weight_quant_type=config["quant_type"],
        act_quant_type=config["quant_type"],
        TOPK=config["topk"],
        BLOCK_TILE_SIZE_M=BLOCK_M,
        BLOCK_TILE_SIZE_N=128,
        stage="gateup",
        alg="prefill_1x4",
        E=config["experts"],
        USE_ATOMIC_WRITE=False,
    )
    stream = torch.cuda.current_stream()

    def launch_gate():
        gate(
            ptr(gate_input),
            ptr(gate_weight),
            ptr(gate_output),
            ptr(meta64["ids"]),
            ptr(meta64["weights"]),
            ptr(meta64["experts"]),
            ptr(meta64["valid"]),
            ptr(gate_weight_scale),
            ptr(gate_input_scale),
            batch,
            meta64["task_num"],
            stream,
        )

    return launch_gate


def run_benchmark(args):
    config = dict(MODELS[args.model])
    batch = args.batch
    base = load_isolated_module("base", args.base)
    control = load_isolated_module("control", args.control)
    candidate = load_isolated_module("candidate", args.candidate)

    original = state_helper.read_gpu_state(args.physical_device, args.amdsmi_root)
    if not args.correctness_only:
        if original["performance_level"] != "auto":
            raise RuntimeError(f"GPU must start in auto: {original}")
        if original["gpu_busy_percent"] > 5 or original["vram_allocated_percent"] > 20:
            raise RuntimeError(f"GPU is not idle: {original}")
        if original["power_cap_w"] != EXPECTED_POWER_CAP_W:
            raise RuntimeError(f"expected 650W cap: {original}")

    result = None
    restored = None
    state_managed = False
    try:
        if args.correctness_only:
            managed = original
        else:
            state_helper.set_experiment_state(args.physical_device, args.amdsmi_root)
            state_managed = True
            managed = state_helper.read_gpu_state(
                args.physical_device, args.amdsmi_root
            )
            if managed["performance_level"] != "perf_determinism":
                raise RuntimeError(f"failed to set determinism: {managed}")
            if (
                managed["ptl_state"] != "Enabled"
                or managed["ptl_format"] != "VECTOR,F8"
            ):
                raise RuntimeError(f"failed to set PTL: {managed}")

        metadata = build_metadata(candidate, config, batch)
        meta_for_label = {
            "base_e6": metadata["m64"],
            "n256_4wave": metadata["m64"],
            "m128n256_8wave": metadata["paired"],
            "m64n512_8wave": metadata["paired"],
        }
        activations, weights, weight_scales, activation_scales = build_data(
            config, args.buffer_copies, batch
        )
        launches, sums = compile_paths(base, control, candidate, config)
        launch_gate = build_gate(control, config, metadata["m64"], batch)
        stream = torch.cuda.current_stream()

        strides = {
            "base_e6": config["n"],
            "n256_4wave": config["n"] + config["padding"] // 2,
            "m128n256_8wave": config["n"],
            "m64n512_8wave": config["n"],
        }
        path_numel = {
            label: meta_for_label[label]["rows"] * strides[label] for label in LABELS
        }
        max_numel = max(path_numel.values())
        physical = [
            torch.empty(max_numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(args.buffer_copies)
        ]
        reduced = [
            torch.empty(batch, config["n"], dtype=torch.bfloat16, device="cuda")
            for _ in range(args.buffer_copies)
        ]

        def launch_down(label, index):
            meta = meta_for_label[label]
            launches[label](
                ptr(activations[index]),
                ptr(weights[index]),
                ptr(physical[index]),
                ptr(meta["ids"]),
                ptr(meta["weights"]),
                ptr(meta["experts"]),
                ptr(meta["valid"]),
                ptr(weight_scales[index]),
                ptr(activation_scales[index]),
                batch,
                meta["task_num"],
                stream,
            )

        def launch_sum(label, index):
            sums[label](
                meta_for_label[label]["loc"], physical[index], reduced[index], batch
            )

        sentinel = -123.0
        physical[0].fill_(sentinel)
        launch_down("base_e6", 0)
        launch_sum("base_e6", 0)
        torch.cuda.synchronize()
        reference = reduced[0].clone()
        correctness = {}
        for label in LABELS:
            physical[0].fill_(sentinel)
            launch_down(label, 0)
            launch_sum(label, 0)
            torch.cuda.synchronize()
            rel_l2 = relative_l2(reduced[0], reference)
            finite = bool(torch.isfinite(reduced[0]).all().item())
            path_view = physical[0][: path_numel[label]].view(
                meta_for_label[label]["rows"], strides[label]
            )
            inactive_rows = path_view[meta_for_label[label]["padded_rows"] :]
            allocation_tail = physical[0][path_numel[label] :]
            tail_clean = bool(
                (
                    inactive_rows.numel() == 0
                    or torch.all(inactive_rows == sentinel).item()
                )
                and (
                    allocation_tail.numel() == 0
                    or torch.all(allocation_tail == sentinel).item()
                )
            )
            padding_clean = True
            if label == "n256_4wave" and config["padding"]:
                padding_clean = bool(
                    torch.all(path_view[:, config["n"] :] == sentinel).item()
                )
            correctness[label] = {
                "reduced_rel_l2_vs_base": rel_l2,
                "finite": finite,
                "tail_clean": tail_clean,
                "padding_clean": padding_clean,
            }
            if not finite or not tail_clean or not padding_clean or rel_l2 > 5e-3:
                raise AssertionError(
                    f"correctness failed for {label}: {correctness[label]}"
                )

        if args.correctness_only:
            result = {
                "model": args.model,
                "shape": {"batch": batch, **config, "block_m": BLOCK_M},
                "correctness_only": True,
                "correctness": correctness,
                "initial_state": compact_state(original),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return result

        print(
            f"[{args.model}] buffers={args.buffer_copies} max_physical_gib="
            f"{max_numel * 2 * args.buffer_copies / 2**30:.2f}"
        )
        for index in range(args.buffer_copies):
            for label in LABELS:
                launch_gate()
                launch_down(label, index)
                launch_sum(label, index)
        torch.cuda.synchronize()

        latin = (
            ("base_e6", "n256_4wave", "m64n512_8wave", "m128n256_8wave"),
            ("n256_4wave", "m128n256_8wave", "base_e6", "m64n512_8wave"),
            ("m128n256_8wave", "m64n512_8wave", "n256_4wave", "base_e6"),
            ("m64n512_8wave", "base_e6", "m128n256_8wave", "n256_4wave"),
        )

        def measure(phase):
            samples = {label: [] for label in LABELS}
            per_round = []
            call_index = 0
            for round_index in range(args.rounds):
                half = latin[round_index % len(latin)]
                order = half + tuple(reversed(half))
                events = []
                for label in order:
                    index = call_index % args.buffer_copies
                    call_index += 1
                    launch_gate()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch_down(label, index)
                    if phase == "combined":
                        launch_sum(label, index)
                    end.record()
                    events.append((label, start, end))
                torch.cuda.synchronize()
                current = {label: [] for label in LABELS}
                for label, start, end in events:
                    elapsed = start.elapsed_time(end)
                    samples[label].append(elapsed)
                    current[label].append(elapsed)
                means = {label: statistics.mean(current[label]) for label in LABELS}
                per_round.append(
                    {
                        "means": means,
                        "over_base": {
                            label: means[label] / means["base_e6"] for label in LABELS
                        },
                    }
                )
            return {
                "versions": {label: summarize(samples[label]) for label in LABELS},
                "over_base": {
                    label: summarize([entry["over_base"][label] for entry in per_round])
                    for label in LABELS
                },
                "rounds": per_round,
            }

        phases = {"down": measure("down"), "combined": measure("combined")}
        useful_flops = 2 * metadata["useful_rows"] * config["n"] * config["k"]
        for entry in phases["down"]["versions"].values():
            entry["useful_tflops"] = useful_flops / (entry["median"] * 1e9)

        result = {
            "model": args.model,
            "shape": {"batch": batch, **config, "block_m": BLOCK_M},
            "sources": {
                "base_e6": {
                    "path": str(args.base.resolve()),
                    "sha256": sha256(args.base),
                },
                "n256_4wave": {
                    "path": str(args.control.resolve()),
                    "sha256": sha256(args.control),
                },
                "m128n256_8wave": {
                    "path": str(args.control.resolve()),
                    "sha256": sha256(args.control),
                },
                "m64n512_8wave": {
                    "path": str(args.candidate.resolve()),
                    "sha256": sha256(args.candidate),
                },
            },
            "device": {
                "name": torch.cuda.get_device_properties(0).name,
                "compute_units": torch.cuda.get_device_properties(
                    0
                ).multi_processor_count,
            },
            "rounds": args.rounds,
            "buffer_copies": args.buffer_copies,
            "metadata": {
                "m64_tasks": metadata["m64"]["task_num"],
                "paired_m64_tasks": metadata["paired"]["task_num"],
                "m64_rows": metadata["m64"]["rows"],
                "paired_rows": metadata["paired"]["rows"],
            },
            "correctness": correctness,
            "initial_state": compact_state(original),
            "managed_state": compact_state(managed),
            "phases": phases,
        }
    finally:
        if state_managed:
            state_helper.restore_experiment_state(
                args.physical_device, args.amdsmi_root, original
            )
        restored = state_helper.read_gpu_state(args.physical_device, args.amdsmi_root)

    for key in ("performance_level", "ptl_state", "ptl_format", "numa_balancing"):
        if restored[key] != original[key]:
            raise RuntimeError(
                f"state restoration mismatch {key}: {original[key]} -> {restored[key]}"
            )
    result["restored_state"] = compact_state(restored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--base", type=Path, default=BASE_MODULE)
    parser.add_argument("--control", type=Path, default=CONTROL_MODULE)
    parser.add_argument("--candidate", type=Path, default=NSPLIT_MODULE)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--buffer-copies", type=int, default=10)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--physical-device", type=int, default=4)
    parser.add_argument(
        "--amdsmi-root", type=Path, default=state_helper.DEFAULT_AMDSMI_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if args.buffer_copies < 1:
        parser.error("--buffer-copies must be positive")
    if args.batch < 1:
        parser.error("--batch must be positive")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_device):
        parser.error("HIP_VISIBLE_DEVICES must match --physical-device")
    return args


def main():
    args = parse_args()
    result = run_benchmark(args)
    if args.correctness_only:
        print(json.dumps(result, indent=2))
        return
    summary = {}
    for phase_name, phase in result["phases"].items():
        summary[phase_name] = {
            label: {
                "median_ms": phase["versions"][label]["median"],
                "q1_ms": phase["versions"][label]["q1"],
                "q3_ms": phase["versions"][label]["q3"],
                "over_base": phase["over_base"][label]["median"],
            }
            for label in LABELS
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
