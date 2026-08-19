#!/usr/bin/env python3
"""统一8-wave候选的受控10-buffer ABBA性能与逐bit正确性比较。"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx

import probe_control_k128_hardware as state_helper

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODULE = REPO_ROOT / "src/contrib/flydsl/moe_gemm_splitk.py"
DEFAULT_AMDSMI_ROOT = state_helper.DEFAULT_AMDSMI_ROOT
BATCH, TOPK, HIDDEN, K, EXPERTS, BLOCK_M = 32768, 9, 4096, 384, 193, 64
BUFFER_COPIES = 10
EXPECTED_POWER_CAP_W = 650
FP8 = torch.float8_e4m3fnuz
_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
}


def load_module(name: str, path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_file = module.__file__
    if loaded_file is None or Path(loaded_file).resolve() != path:
        raise RuntimeError(f"loaded unexpected module: {module.__file__}")
    return module


def ptr(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_MODULE)
    parser.add_argument(
        "--control-path",
        choices=("physical4", "unified8"),
        default="unified8",
    )
    parser.add_argument(
        "--candidate-path",
        choices=("physical4", "unified8"),
        default="unified8",
    )
    parser.add_argument("--candidate-packed-direct", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--down-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--physical-device", type=int, default=4)
    parser.add_argument("--max-initial-vram-percent", type=int, default=20)
    parser.add_argument("--amdsmi-root", type=Path, default=DEFAULT_AMDSMI_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exact-valid-grid", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if (
        not args.skip_correctness
        and args.control_path != args.candidate_path
    ):
        parser.error(
            "cross-path comparisons require --skip-correctness; validate "
            "bitwise output against a control using the same physical layout"
        )
    return args


def run(args):
    visible_device = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_device != str(args.physical_device):
        raise RuntimeError(
            "HIP_VISIBLE_DEVICES must contain exactly the managed physical "
            f"device {args.physical_device}, got {visible_device!r}"
        )
    if state_helper.JIT.gfx != 942:
        raise RuntimeError(
            f"this comparison requires gfx942, got gfx{state_helper.JIT.gfx}"
        )
    device_properties = torch.cuda.get_device_properties(0)
    if device_properties.multi_processor_count != 80:
        raise RuntimeError(
            "this comparison requires an 80-CU MI308X, got "
            f"{device_properties.name} with {device_properties.multi_processor_count} CUs"
        )
    control = load_module("pyhip.contrib.flydsl.unified8_control", args.control)
    candidate = load_module("pyhip.contrib.flydsl.unified8_candidate", args.candidate)

    original_state = state_helper.read_gpu_state(args.physical_device, args.amdsmi_root)
    if original_state["performance_level"] != "auto":
        raise RuntimeError(
            "formal state management requires performance_level=auto before "
            f"the run, got {original_state['performance_level']!r}"
        )
    if original_state["power_cap_w"] != EXPECTED_POWER_CAP_W:
        raise RuntimeError(
            f"formal comparison requires a {EXPECTED_POWER_CAP_W}W power cap, "
            f"got {original_state['power_cap_w']}W"
        )
    if (
        original_state["gpu_busy_percent"] > 5
        or original_state["vram_allocated_percent"]
        > args.max_initial_vram_percent
    ):
        raise RuntimeError(f"GPU{args.physical_device} is not idle: {original_state}")

    restored_state = None
    result = None
    try:
        state_helper.set_experiment_state(args.physical_device, args.amdsmi_root)
        managed_state = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )
        if (
            managed_state["performance_level"] != "perf_determinism"
            or managed_state["ptl_state"] != "Enabled"
            or managed_state["ptl_format"] != "VECTOR,F8"
            or managed_state["numa_balancing"] != original_state["numa_balancing"]
        ):
            raise RuntimeError(f"failed to enter experiment state: {managed_state}")

        torch.manual_seed(20260820)
        useful_rows = BATCH * TOPK
        topk_ids = (
            torch.arange(useful_rows, dtype=torch.int32, device="cuda")
            .remainder(EXPERTS)
            .reshape(BATCH, TOPK)
        )
        topk_weights = torch.linspace(
            0.5, 1.0, useful_rows, dtype=torch.float32, device="cuda"
        ).reshape(BATCH, TOPK)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids,
            topk_weights,
            EXPERTS,
            HIDDEN,
            torch.bfloat16,
            BLOCK_M,
            None,
            None,
            0,
        )
        grid = sorted_expert_ids.shape[0]
        padded_rows = int(num_valid_ids[0].item())
        down_grid = padded_rows // BLOCK_M if args.exact_valid_grid else grid
        allocated_rows = grid * BLOCK_M
        loc_ids = torch.zeros(BATCH, TOPK, dtype=torch.int32, device="cuda")
        control.invert_sorted_ids(TOPK)(
            sorted_ids,
            loc_ids,
            num_valid_ids,
            sorted_ids.shape[0],
            BATCH,
        )

        activation_base = (
            0.1 * torch.randn(BATCH, TOPK, K, dtype=torch.float16, device="cuda")
        ).to(FP8)
        weight_base = shuffle_weight(
            (
                0.1
                * torch.randn(
                    EXPERTS,
                    HIDDEN,
                    K,
                    dtype=torch.float16,
                    device="cuda",
                )
            ).to(FP8),
            layout=(16, 16),
        )
        weight_scale_base = torch.linspace(
            0.75, 1.25, EXPERTS, dtype=torch.float32, device="cuda"
        )
        activation_scale_base = torch.tensor(
            [0.625], dtype=torch.float32, device="cuda"
        )
        activations = [activation_base.clone() for _ in range(BUFFER_COPIES)]
        weights = [weight_base.clone() for _ in range(BUFFER_COPIES)]
        weight_scales = [weight_scale_base.clone() for _ in range(BUFFER_COPIES)]
        activation_scales = [
            activation_scale_base.clone() for _ in range(BUFFER_COPIES)
        ]
        del (
            activation_base,
            weight_base,
            weight_scale_base,
            activation_scale_base,
        )

        labels = ("control", "candidate")
        output_sentinels = {"control": -3.0, "candidate": 5.0}
        physical_outputs = [
            torch.empty(
                (allocated_rows, HIDDEN),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(BUFFER_COPIES)
        ]
        reduced_outputs = [
            torch.empty(
                (BATCH, HIDDEN),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(BUFFER_COPIES)
        ]
        candidate_physical_output = torch.empty_like(physical_outputs[0])
        candidate_reduced_output = torch.empty_like(reduced_outputs[0])

        common = dict(
            N=HIDDEN,
            K=K,
            weight_dtype="fp8",
            weight_quant_type="per_tensor",
            act_quant_type="per_tensor",
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_M,
            stage="down",
            alg="prefill_1x4",
            E=EXPERTS,
            USE_ATOMIC_WRITE=False,
            down_output_padding_bytes=0,
        )
        modules = {"control": control, "candidate": candidate}
        paths = {
            "control": args.control_path,
            "candidate": args.candidate_path,
        }

        def compile_down(label):
            if paths[label] == "unified8":
                return modules[label].compile_gemm(
                    **common,
                    BLOCK_TILE_SIZE_N=512,
                    down_physical_n512=True,
                )
            return modules[label].compile_gemm(
                **common,
                BLOCK_TILE_SIZE_N=256,
                down_physical_n256=True,
            )

        down_launch = {label: compile_down(label) for label in labels}
        sum_launch = {
            "control": control.sorted_sum(TOPK, HIDDEN, 0),
            "candidate": (
                candidate.sorted_sum(TOPK, HIDDEN, 0, packed_direct=True)
                if args.candidate_packed_direct
                else candidate.sorted_sum(TOPK, HIDDEN, 0)
            ),
        }
        decode_candidate = (
            candidate.sorted_sum(1, HIDDEN, 0, packed_direct=True)
            if args.candidate_packed_direct
            else None
        )
        identity_locs = (
            torch.arange(padded_rows, dtype=torch.int32, device="cuda").view(-1, 1)
            if args.candidate_packed_direct
            else None
        )
        decoded_candidate = (
            torch.empty(
                (padded_rows, HIDDEN),
                dtype=torch.bfloat16,
                device="cuda",
            )
            if args.candidate_packed_direct
            else None
        )

        gate_input = torch.ones(BATCH, HIDDEN, dtype=FP8, device="cuda")
        gate_weight = shuffle_weight(
            torch.ones(EXPERTS, 384, HIDDEN, dtype=FP8, device="cuda"),
            layout=(16, 16),
        )
        gate_output = torch.empty(BATCH, TOPK, 192, dtype=torch.bfloat16, device="cuda")
        gate_weight_scale = torch.ones(EXPERTS, dtype=torch.float32, device="cuda")
        gate_input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        gate = control.compile_gemm(
            N=384,
            K=HIDDEN,
            weight_dtype="fp8",
            weight_quant_type="per_tensor",
            act_quant_type="per_tensor",
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_M,
            BLOCK_TILE_SIZE_N=128,
            stage="gateup",
            alg="prefill_1x4",
            E=EXPERTS,
            USE_ATOMIC_WRITE=False,
        )
        stream = torch.cuda.current_stream()

        def launch_gate():
            gate(
                ptr(gate_input),
                ptr(gate_weight),
                ptr(gate_output),
                ptr(sorted_ids),
                ptr(sorted_weights),
                ptr(sorted_expert_ids),
                ptr(num_valid_ids),
                ptr(gate_weight_scale),
                ptr(gate_input_scale),
                BATCH,
                grid,
                stream,
            )

        def launch_down(label, index, output=None):
            down_launch[label](
                ptr(activations[index]),
                ptr(weights[index]),
                ptr(physical_outputs[index] if output is None else output),
                ptr(sorted_ids),
                ptr(sorted_weights),
                ptr(sorted_expert_ids),
                ptr(num_valid_ids),
                ptr(weight_scales[index]),
                ptr(activation_scales[index]),
                BATCH,
                down_grid,
                stream,
            )

        def launch_sum(label, index, physical_output=None, reduced_output=None):
            sum_launch[label](
                loc_ids,
                (
                    physical_outputs[index]
                    if physical_output is None
                    else physical_output
                ),
                reduced_outputs[index] if reduced_output is None else reduced_output,
                BATCH,
            )

        if not args.skip_correctness:
            for index in range(BUFFER_COPIES):
                physical_outputs[index].fill_(output_sentinels["control"])
                reduced_outputs[index].fill_(output_sentinels["control"])
                candidate_physical_output.fill_(output_sentinels["candidate"])
                candidate_reduced_output.fill_(output_sentinels["candidate"])
                launch_down("control", index)
                launch_sum("control", index)
                launch_down("candidate", index, candidate_physical_output)
                launch_sum(
                    "candidate",
                    index,
                    candidate_physical_output,
                    candidate_reduced_output,
                )
                if args.candidate_packed_direct:
                    decode_candidate(
                        identity_locs,
                        candidate_physical_output,
                        decoded_candidate,
                        padded_rows,
                    )
                torch.cuda.synchronize()

                physical_lhs = physical_outputs[index][:padded_rows]
                physical_rhs = (
                    decoded_candidate
                    if args.candidate_packed_direct
                    else candidate_physical_output[:padded_rows]
                )
                if not torch.equal(physical_lhs, physical_rhs):
                    bad_rows = (physical_lhs != physical_rhs).any(dim=1)
                    bad_blocks = (
                        bad_rows.reshape(padded_rows // BLOCK_M, BLOCK_M)
                        .any(dim=1)
                        .nonzero()
                        .flatten()
                    )
                    raise AssertionError(
                        f"physical mismatch buffer={index} "
                        f"blocks={bad_blocks[:32].tolist()}"
                    )
                for label, output in (
                    ("control", physical_outputs[index]),
                    ("candidate", candidate_physical_output),
                ):
                    inactive_tail = output[padded_rows:]
                    if inactive_tail.numel() and not torch.all(
                        inactive_tail == output_sentinels[label]
                    ):
                        raise AssertionError(
                            f"{label} wrote inactive tail rows in buffer={index}"
                        )
                if not torch.equal(
                    reduced_outputs[index],
                    candidate_reduced_output,
                ):
                    raise AssertionError(f"reduced output mismatch buffer={index}")

        del candidate_physical_output, candidate_reduced_output
        torch.cuda.empty_cache()

        for index in range(BUFFER_COPIES):
            for label in labels:
                launch_down(label, index)
                if not args.down_only:
                    launch_sum(label, index)
        torch.cuda.synchronize()

        orders = (
            ("control", "candidate", "candidate", "control"),
            ("candidate", "control", "control", "candidate"),
        )

        def measure(phase):
            samples = {label: [] for label in labels}
            ratios = []
            call_index = 0
            for warmup_index in range(BUFFER_COPIES):
                for label in labels:
                    launch_gate()
                    if phase == "consumer":
                        launch_sum(label, warmup_index)
                    else:
                        launch_down(label, warmup_index)
                        if phase == "combined":
                            launch_sum(label, warmup_index)
            torch.cuda.synchronize()
            for round_index in range(args.rounds):
                events = []
                for label in orders[round_index % 2]:
                    index = call_index % BUFFER_COPIES
                    call_index += 1
                    launch_gate()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    if phase == "consumer":
                        launch_sum(label, index)
                    else:
                        launch_down(label, index)
                        if phase == "combined":
                            launch_sum(label, index)
                    end.record()
                    events.append((label, start, end))
                torch.cuda.synchronize()
                current = {label: [] for label in labels}
                for label, start, end in events:
                    elapsed = start.elapsed_time(end)
                    samples[label].append(elapsed)
                    current[label].append(elapsed)
                ratios.append(
                    statistics.mean(current["candidate"])
                    / statistics.mean(current["control"])
                )
            return {
                "versions": {
                    label: summarize(values) for label, values in samples.items()
                },
                "candidate_over_control": summarize(ratios),
            }

        phases = {"down": measure("down")}
        if not args.down_only:
            phases.update(
                consumer=measure("consumer"),
                combined=measure("combined"),
            )
        useful_flops = 2 * useful_rows * HIDDEN * K
        for entry in phases["down"]["versions"].values():
            entry["useful_tflops"] = useful_flops / (entry["median"] * 1e9)

        result = {
            "control": str(args.control.resolve()),
            "candidate": str(args.candidate.resolve()),
            "paths": paths,
            "candidate_packed_direct": args.candidate_packed_direct,
            "source_sha256": {
                "control": sha256(args.control.resolve()),
                "candidate": sha256(args.candidate.resolve()),
            },
            "device": {
                "name": device_properties.name,
                "gfx": state_helper.JIT.gfx,
                "compute_units": device_properties.multi_processor_count,
            },
            "shape": {
                "batch": BATCH,
                "topk": TOPK,
                "n": HIDDEN,
                "k": K,
                "experts": EXPERTS,
                "block_m": BLOCK_M,
            },
            "rounds": args.rounds,
            "buffer_copies": BUFFER_COPIES,
            "correctness": {
                "skipped": args.skip_correctness,
                "all_valid_physical_rows_bit_equal": not args.skip_correctness,
                "inactive_tail_rows_untouched": not args.skip_correctness,
                "full_reduced_outputs_bit_equal": not args.skip_correctness,
                "buffers_checked": 0 if args.skip_correctness else BUFFER_COPIES,
            },
            "padded_rows": padded_rows,
            "allocated_rows": allocated_rows,
            "initial_state": compact_state(original_state),
            "managed_state": compact_state(managed_state),
            "phases": phases,
        }
    finally:
        state_helper.restore_experiment_state(
            args.physical_device, args.amdsmi_root, original_state
        )
        restored_state = state_helper.read_gpu_state(
            args.physical_device, args.amdsmi_root
        )

    for key in (
        "performance_level",
        "ptl_state",
        "ptl_format",
        "numa_balancing",
    ):
        if restored_state[key] != original_state[key]:
            raise RuntimeError(
                f"GPU state restoration mismatch for {key}: "
                f"{original_state[key]!r} -> {restored_state[key]!r}"
            )

    result["restored_state"] = compact_state(restored_state)
    return result


def main():
    args = parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
