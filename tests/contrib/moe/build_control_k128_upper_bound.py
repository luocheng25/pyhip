#!/usr/bin/env python3
"""Close Control-K128 wall time and bound local scheduling headroom."""

import argparse
import json
from pathlib import Path

USEFUL_FLOPS = 927_712_935_936
ARCHITECTURE_ROOF_TFLOPS = 582.944
MEMORY_BUCKETS = (
    "VMEM stall/wait candidate",
    "LDS stall/wait candidate",
    "mixed vmcnt/lgkmcnt wait candidate",
)
EXPECTED_RESOURCES = {
    "vgpr": 64,
    "agpr": 128,
    "scratch_bytes_per_lane": 0,
    "lds_bytes_per_block": 28_672,
}


def load_result(path, label):
    payload = json.loads(path.read_text())
    for result in payload["results"]:
        if result["label"] == label:
            return result
    available = ", ".join(result["label"] for result in payload["results"])
    raise RuntimeError(f"{label!r} not found in {path}; available: {available}")


def percentage(value):
    return f"{100.0 * value:.2f}%"


def validate_trace(result, label):
    metadata = result["metadata"]
    expected = {
        "vgpr": 64,
        "accum_vgpr": 128,
        "lds_bytes": 28_672,
        "workgroup_size": 256,
    }
    mismatches = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} resource mismatch: {mismatches}")
    if (
        result["mfma_busy_cycles"] + result["mfma_idle_cycles"]
        != result["both_active_cycles"]
    ):
        raise RuntimeError(f"{label} lifecycle ledger does not close")
    if (
        result["steady_mfma_busy_cycles"] + result["steady_mfma_idle_cycles"]
        != result["steady_cycles"]
    ):
        raise RuntimeError(f"{label} steady ledger does not close")
    if sum(result["fixability_cycles"].values()) != result["mfma_idle_cycles"]:
        raise RuntimeError(f"{label} lifecycle fixability ledger does not close")
    if (
        sum(result["steady_fixability_cycles"].values())
        != result["steady_mfma_idle_cycles"]
    ):
        raise RuntimeError(f"{label} steady fixability ledger does not close")


def validate_hardware(hardware, expected_raw_sha256):
    if not hardware["formal_result"] or hardware["invalidity_reasons"]:
        raise RuntimeError("hardware distribution is not a formal result")
    if hardware["config"]["samples_per_metric"] != 40_960:
        raise RuntimeError(
            "hardware distribution does not contain 40,960 samples per metric"
        )
    mapping = hardware["mapping_validation"]
    if mapping["physical_simds"] != 320 or mapping["waves_per_physical_simd"] != 2:
        raise RuntimeError(
            "hardware probe did not establish two waves on all 320 SIMD units"
        )
    if mapping["overlap_fraction_min"] < 0.999:
        raise RuntimeError("resident-wave lifetimes did not overlap for at least 99.9%")

    observed = hardware.get("observed_resources", {})
    mismatches = {
        key: (value, observed.get(key))
        for key, value in EXPECTED_RESOURCES.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"hardware-probe resource mismatch: {mismatches}")
    if (
        expected_raw_sha256
        and hardware["full_raw_artifact"]["sha256"] != expected_raw_sha256
    ):
        raise RuntimeError("hardware raw artifact SHA256 mismatch")
    for key in ("performance_level", "ptl_state", "ptl_format", "numa_balancing"):
        if hardware["state"]["original"][key] != hardware["state"]["restored"][key]:
            raise RuntimeError(f"hardware state restoration mismatch for {key}")

    mfma_fast = hardware["results"]["mfma_four_chain_cycles_per_instruction"]
    if mfma_fast["p50"] > 16.0:
        raise RuntimeError(f"four-chain MFMA p50 exceeds 16 cycles: {mfma_fast['p50']}")


def memory_fraction(result):
    return (
        sum(
            result["steady_fixability_cycles"].get(bucket, 0)
            for bucket in MEMORY_BUCKETS
        )
        / result["steady_cycles"]
    )


def remove_steady_fraction(stable_ms, stable_tflops, fraction):
    return {
        "removed_steady_fraction": fraction,
        "estimated_ms": stable_ms * (1.0 - fraction),
        "estimated_tflops": stable_tflops / (1.0 - fraction),
        "gain": 1.0 / (1.0 - fraction) - 1.0,
    }


def build_model(args):
    control = load_result(args.slots, args.control_label)
    ablation = load_result(args.slots, args.ablation_label)
    hardware = json.loads(args.hardware.read_text())
    validate_trace(control, args.control_label)
    validate_trace(ablation, args.ablation_label)
    validate_hardware(hardware, args.expected_hardware_raw_sha256)

    att_tflops = USEFUL_FLOPS / (args.att_ms * 1e9)
    stable_tflops = USEFUL_FLOPS / (args.stable_ms * 1e9)
    ablation_tflops = USEFUL_FLOPS / (args.ablation_ms * 1e9)
    ablation_att_tflops = USEFUL_FLOPS / (args.ablation_att_ms * 1e9)

    control_slot_prediction = ARCHITECTURE_ROOF_TFLOPS * control["mfma_busy_fraction"]
    ablation_slot_prediction = ARCHITECTURE_ROOF_TFLOPS * ablation["mfma_busy_fraction"]
    trace_to_stable_factor = args.att_ms / args.stable_ms
    control_stable_prediction = control_slot_prediction * trace_to_stable_factor

    control_memory = memory_fraction(control)
    ablation_memory = memory_fraction(ablation)
    control_specific_memory = control_memory - ablation_memory
    if control_specific_memory <= 0:
        raise RuntimeError(
            "Control does not have more memory exposure than the ablation"
        )

    scenarios = {
        "measured_control": {
            "estimated_ms": args.stable_ms,
            "estimated_tflops": stable_tflops,
            "gain": 0.0,
            "status": "measured",
        },
        "measured_same_work_ablation": {
            "estimated_ms": args.ablation_ms,
            "estimated_tflops": ablation_tflops,
            "gain": ablation_tflops / stable_tflops - 1.0,
            "status": "measured reachable",
        },
        "conservative_25pct_memory_recovery": {
            **remove_steady_fraction(
                args.stable_ms, stable_tflops, 0.25 * control_memory
            ),
            "status": "conditional sensitivity; not yet demonstrated",
        },
        "optimistic_retain_ablation_residual": {
            **remove_steady_fraction(
                args.stable_ms, stable_tflops, control_specific_memory
            ),
            "status": "optimistic local bound; retains ablation memory residual",
        },
        "algebraic_remove_all_memory_exposure": {
            **remove_steady_fraction(args.stable_ms, stable_tflops, control_memory),
            "status": "algebraic bound; not a performance promise",
        },
    }

    return {
        "schema_version": 1,
        "constants": {
            "useful_flops": USEFUL_FLOPS,
            "architecture_roof_tflops": ARCHITECTURE_ROOF_TFLOPS,
        },
        "wall_time_closure": {
            "control": {
                "att_ms": args.att_ms,
                "att_tflops": att_tflops,
                "lifecycle_mfma_busy": control["mfma_busy_fraction"],
                "slot_prediction_tflops": control_slot_prediction,
                "att_relative_error": control_slot_prediction / att_tflops - 1.0,
                "trace_to_stable_factor": trace_to_stable_factor,
                "stable_prediction_tflops": control_stable_prediction,
                "stable_ms": args.stable_ms,
                "stable_tflops": stable_tflops,
                "stable_relative_error": control_stable_prediction / stable_tflops
                - 1.0,
            },
            "ablation": {
                "att_ms": args.ablation_att_ms,
                "att_tflops": ablation_att_tflops,
                "lifecycle_mfma_busy": ablation["mfma_busy_fraction"],
                "slot_prediction_tflops": ablation_slot_prediction,
                "att_relative_error": ablation_slot_prediction / ablation_att_tflops
                - 1.0,
            },
        },
        "steady_fractions": {
            "memory_exposure_control": control_memory,
            "memory_exposure_ablation": ablation_memory,
            "control_specific_memory_exposure": control_specific_memory,
            "structural_tail": control["steady_fixability_cycles"]["structural tail"]
            / control["steady_cycles"],
            "edge_prologue_drain_lifecycle": control["fixability_cycles"][
                "edge/prologue/drain"
            ]
            / control["both_active_cycles"],
        },
        "hardware_checks": {
            "samples_per_metric": hardware["config"]["samples_per_metric"],
            "physical_simds": hardware["mapping_validation"]["physical_simds"],
            "waves_per_physical_simd": hardware["mapping_validation"][
                "waves_per_physical_simd"
            ],
            "overlap_fraction_min": hardware["mapping_validation"][
                "overlap_fraction_min"
            ],
            "mfma_four_chain_cycles_per_instruction": {
                key: hardware["results"]["mfma_four_chain_cycles_per_instruction"][key]
                for key in ("p50", "p95", "p99", "max")
            },
            "vmem_load_cold_cycles": {
                key: hardware["results"]["vmem_load_cold"][
                    "fixed_timer_corrected_cycles"
                ][key]
                for key in ("p50", "p95", "p99", "max")
            },
            "lds_read_single_cycles": {
                key: hardware["results"]["lds_read_single"][
                    "fixed_timer_corrected_cycles"
                ][key]
                for key in ("p50", "p95", "p99", "max")
            },
        },
        "scenarios": scenarios,
    }


def print_model(model):
    control = model["wall_time_closure"]["control"]
    ablation = model["wall_time_closure"]["ablation"]
    print("Control K128 wall-time closure")
    print(f"  useful FLOPs:              {USEFUL_FLOPS:,}")
    print(f"  architecture roof:         {ARCHITECTURE_ROOF_TFLOPS:.3f} T")
    print(f"  lifecycle MFMA busy:       {percentage(control['lifecycle_mfma_busy'])}")
    print(f"  ATT slot prediction:       {control['slot_prediction_tflops']:.3f} T")
    print(f"  ATT measured:              {control['att_tflops']:.3f} T")
    print(f"  ATT model error:           {percentage(control['att_relative_error'])}")
    print(f"  ATT/stable duration ratio: {control['trace_to_stable_factor']:.6f}")
    print(f"  stable prediction:         {control['stable_prediction_tflops']:.3f} T")
    print(f"  stable measured:           {control['stable_tflops']:.3f} T")
    print(
        f"  stable model error:        {percentage(control['stable_relative_error'])}"
    )
    print()
    print("9aa595d cross-check")
    print(f"  ATT slot prediction:       {ablation['slot_prediction_tflops']:.3f} T")
    print(f"  ATT measured:              {ablation['att_tflops']:.3f} T")
    print(f"  ATT model error:           {percentage(ablation['att_relative_error'])}")
    print()
    print("Upper-bound scenarios")
    for name, scenario in model["scenarios"].items():
        print(
            f"  {name:38s} {scenario['estimated_ms']:.6f} ms  "
            f"{scenario['estimated_tflops']:.3f} T  {percentage(scenario['gain'])}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--control-label", default="Control K128")
    parser.add_argument("--ablation-label", default="9aa595d")
    parser.add_argument("--stable-ms", type=float, default=2.217927)
    parser.add_argument("--att-ms", type=float, default=2.280010)
    parser.add_argument("--ablation-ms", type=float, default=2.194646)
    parser.add_argument("--ablation-att-ms", type=float, default=2.194649)
    parser.add_argument("--expected-hardware-raw-sha256")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    model = build_model(args)
    print_model(model)
    for name, closure in model["wall_time_closure"].items():
        if abs(closure["att_relative_error"]) > 0.015:
            raise RuntimeError(f"{name} ATT model error exceeds 1.5%")
    if abs(model["wall_time_closure"]["control"]["stable_relative_error"]) > 0.015:
        raise RuntimeError("Control stable model error exceeds 1.5%")
    if args.json:
        args.json.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
