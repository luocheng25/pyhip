#!/usr/bin/env python3
"""Benchmark current MoE down kernels with round-robin A/B/D buffers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "src/contrib/flydsl/moe_gemm_splitk.py"
PROBE_PATH = Path(__file__).with_name("probe-batched-gemm-core-ceiling.py")
DEFAULT_AMDSMI_ROOT = Path("/opt/rocm/share/amd_smi")
BATCH = 32768
EXPECTED_POWER_CAP_W = 650.0
FP8 = torch.float8_e4m3fnuz
MODELS = {
    "hy3": dict(
        n=4096,
        k=192,
        experts=193,
        topk=9,
        quant="per_tensor",
        path="true8_hy3",
        m_groups=1,
        metadata_groups=1,
        tile_n=512,
        padding=0,
        waves=8,
    ),
    "h3": dict(
        n=6144,
        k=384,
        experts=128,
        topk=4,
        quant="ptpc",
        path="physical_n256",
        m_groups=2,
        metadata_groups=2,
        tile_n=256,
        padding=0,
        waves=8,
    ),
    "xiaomi": dict(
        n=6144,
        k=256,
        experts=384,
        topk=8,
        quant="ptpc",
        path="physical_n256",
        m_groups=1,
        metadata_groups=1,
        tile_n=256,
        padding=128,
        waves=4,
    ),
    "q35": dict(
        n=2048,
        k=512,
        experts=256,
        topk=8,
        quant="ptpc",
        path="legacy",
        m_groups=1,
        metadata_groups=1,
        tile_n=256,
        padding=None,
        waves=4,
    ),
    "q397": dict(
        n=4096,
        k=512,
        experts=512,
        topk=10,
        quant="ptpc",
        path="legacy",
        m_groups=1,
        metadata_groups=1,
        tile_n=256,
        padding=None,
        waves=4,
    ),
}
TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    FP8: fx.Uint8,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ptr(tensor: torch.Tensor):
    return flyc.from_c_void_p(TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def compact_state(state: dict) -> dict:
    keys = (
        "physical_device",
        "performance_level",
        "sclk",
        "mclk",
        "fclk",
        "power_cap_w",
        "ptl_state",
        "ptl_format",
        "numa_balancing",
        "gpu_busy_percent",
        "vram_allocated_percent",
    )
    return {key: state[key] for key in keys}


def build_workload(config: dict, buffer_copies: int):
    production = load_module(
        "pyhip.contrib.flydsl.moe_gemm_splitk_ten_buffer_profile",
        MODULE_PATH,
    )
    useful_rows = BATCH * config["topk"]
    topk_ids = (
        torch.arange(useful_rows, dtype=torch.int32, device="cuda")
        .remainder(config["experts"])
        .reshape(BATCH, config["topk"])
    )
    topk_weights = torch.ones(
        BATCH, config["topk"], dtype=torch.float32, device="cuda"
    )
    sorting_m = 64 * config["metadata_groups"]
    sorted_ids, sorted_weights, sorted_experts, valid, _ = moe_sorting(
        topk_ids,
        topk_weights,
        config["experts"],
        config["n"],
        torch.bfloat16,
        sorting_m,
        None,
        None,
        0,
    )
    if config["metadata_groups"] == 2:
        sorted_experts = sorted_experts.repeat_interleave(2)
    task_num = sorted_experts.shape[0]
    valid_rows = int(valid[0].item())
    active_m64_tasks = (valid_rows + 63) // 64
    active_workgroups = (
        active_m64_tasks + config["m_groups"] - 1
    ) // config["m_groups"]

    activation_base = torch.ones(
        BATCH, config["topk"], config["k"], dtype=FP8, device="cuda"
    )
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
    activations = [activation_base] + [
        activation_base.clone() for _ in range(buffer_copies - 1)
    ]
    weights = [weight_base] + [
        weight_base.clone() for _ in range(buffer_copies - 1)
    ]
    row_stride = config["n"] + ((config["padding"] or 0) // 2)
    outputs = [
        torch.empty(
            task_num * 64,
            row_stride,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(buffer_copies)
    ]
    if config["quant"] == "ptpc":
        weight_scale = torch.ones(
            config["experts"],
            config["n"],
            dtype=torch.float32,
            device="cuda",
        )
        activation_scale = torch.ones(
            BATCH, config["topk"], dtype=torch.float32, device="cuda"
        )
    else:
        weight_scale = torch.ones(
            config["experts"], dtype=torch.float32, device="cuda"
        )
        activation_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    launch = production.compile_gemm(
        N=config["n"],
        K=config["k"],
        weight_dtype="fp8",
        weight_quant_type=config["quant"],
        act_quant_type=config["quant"],
        TOPK=config["topk"],
        BLOCK_TILE_SIZE_M=64,
        BLOCK_TILE_SIZE_N=config["tile_n"],
        stage="down",
        alg="prefill_1x4",
        E=config["experts"],
        USE_ATOMIC_WRITE=False,
        down_path=config["path"],
        down_m_groups=config["m_groups"],
        metadata_m_groups=config["metadata_groups"],
        down_output_padding_bytes=config["padding"],
    )
    stream = torch.cuda.current_stream()

    def launch_once(index: int) -> None:
        launch(
            ptr(activations[index]),
            ptr(weights[index]),
            ptr(outputs[index]),
            ptr(sorted_ids),
            ptr(sorted_weights),
            ptr(sorted_experts),
            ptr(valid),
            ptr(weight_scale),
            ptr(activation_scale),
            BATCH,
            task_num,
            stream,
        )

    bytes_per_buffer = (
        activation_base.numel() * activation_base.element_size()
        + weight_base.numel() * weight_base.element_size()
        + outputs[0].numel() * outputs[0].element_size()
    )
    metadata = {
        "task_num": task_num,
        "valid_rows": valid_rows,
        "active_m64_tasks": active_m64_tasks,
        "active_workgroups": active_workgroups,
        "launched_workgroups": (
            (task_num + 1) // 2 if config["m_groups"] == 2 else task_num
        ),
        "active_waves": active_workgroups * config["waves"],
        "useful_flops": 2 * useful_rows * config["n"] * config["k"],
        "bytes_per_buffer": bytes_per_buffer,
    }
    return launch_once, metadata


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_device):
        raise RuntimeError(
            "HIP_VISIBLE_DEVICES must equal --physical-device"
        )
    probe = load_module("production_profile_probe_helpers", PROBE_PATH)
    state_helper = probe._wave_probe._load_state_helper()
    original = state_helper.read_gpu_state(
        args.physical_device, args.amdsmi_root
    )
    if original["performance_level"] != "auto":
        raise RuntimeError(f"GPU must start in auto: {original}")
    if original["gpu_busy_percent"] > 5:
        raise RuntimeError(f"GPU is not idle: {original}")
    if abs(original["power_cap_w"] - EXPECTED_POWER_CAP_W) > 0.5:
        raise RuntimeError(f"expected 650 W power cap: {original}")

    changed = False
    payload = None
    try:
        changed = True
        state_helper.set_experiment_state(
            args.physical_device, args.amdsmi_root
        )
        managed = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )
        torch.cuda.set_device(args.device)
        torch.manual_seed(20260827)
        config = MODELS[args.model]
        launch, metadata = build_workload(config, args.buffer_copies)

        for launch_index in range(args.warmups):
            launch(launch_index % args.buffer_copies)
        torch.cuda.synchronize()

        benchmark = None
        if args.command == "bench":
            events = []
            for sample_index in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                launch((args.warmups + sample_index) % args.buffer_copies)
                stop.record()
                events.append((start, stop))
            torch.cuda.synchronize()
            kernel_ms = [start.elapsed_time(stop) for start, stop in events]
            useful_tflops = [
                metadata["useful_flops"] / elapsed_ms / 1.0e9
                for elapsed_ms in kernel_ms
            ]
            benchmark = {
                "kernel_ms": summary(kernel_ms),
                "useful_tflops": summary(useful_tflops),
            }
        else:
            target_index = args.warmups % args.buffer_copies
            launch(target_index)
            torch.cuda.synchronize()

        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            **config,
            **metadata,
            "method": {
                "buffer_copies": args.buffer_copies,
                "buffer_rotation": "round_robin_across_warmups_and_samples",
                "warmup_dispatches": args.warmups,
                "samples": args.samples if args.command == "bench" else 0,
                "sample_sync": "end",
                "rotated_tensors": ["activation", "weight", "output"],
                "fixed_tensors": [
                    "routing_metadata",
                    "weight_scale",
                    "activation_scale",
                ],
                "profiled_buffer_index": (
                    args.warmups % args.buffer_copies
                    if args.command == "profile"
                    else None
                ),
            },
            "benchmark": benchmark,
            "initial_state": compact_state(original),
            "managed_state": compact_state(managed),
        }
    finally:
        if changed:
            state_helper.restore_experiment_state(
                args.physical_device, args.amdsmi_root, original
            )
        restored = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )
    if payload is None:
        raise RuntimeError("production profile did not produce a payload")
    payload["restored_state"] = compact_state(restored)
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for command in ("bench", "profile"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", choices=MODELS, required=True)
        subparser.add_argument("--device", type=int, default=0)
        subparser.add_argument("--physical-device", type=int, default=4)
        subparser.add_argument("--buffer-copies", type=int, default=10)
        subparser.add_argument("--warmups", type=int, default=40)
        subparser.add_argument("--samples", type=int, default=50)
        subparser.add_argument("--json", type=Path)
        subparser.add_argument(
            "--amdsmi-root", type=Path, default=DEFAULT_AMDSMI_ROOT
        )
    return root


def main() -> None:
    args = parser().parse_args()
    if args.buffer_copies < 1 or args.warmups < 1:
        raise RuntimeError("buffer-copies and warmups must be positive")
    if args.command == "bench" and args.samples < 1:
        raise RuntimeError("samples must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
