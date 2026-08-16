#!/usr/bin/env python3
"""Measure gfx942 MFMA, VMEM, and LDS distributions at Control-K128 occupancy."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("PYHIP_DEBUG_LOG", "")

import numpy as np
import torch  # pyright: ignore[reportMissingImports]

from pyhip.core.asmjit import JIT, jit  # pyright: ignore[reportMissingImports]

VOID_POINTER = "void*"
THREADS = 256
WAVES_PER_BLOCK = 4
LDS_BYTES = 28 * 1024
LOADS_PER_BURST = 8
BYTES_PER_REQUEST = 1024
META_DWORDS = 10
DEFAULT_AMDSMI_ROOT = Path(
    os.environ.get(
        "AMDSMI_ROOT",
        "/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3",
    )
)
METRIC_NAMES = (
    "timer_overhead",
    "mfma_dependent64",
    "mfma_four_chain64",
    "vmem_load_cold",
    "vmem_load_l2_hit",
    "vmem_load_burst8",
    "vmem_load_wait_after_burst8",
    "vmem_store_single",
    "vmem_store_burst8",
    "vmem_store_wait_after_burst8",
    "lds_read_single",
    "lds_read_burst8",
    "lds_read_wait_after_burst8",
    "lds_write_single",
    "lds_write_burst8",
    "lds_write_wait_after_burst8",
)
METRIC_DEFINITIONS = {
    "timer_overhead": "back-to-back s_memtime calls plus lgkmcnt(0)",
    "mfma_dependent64": "64 FP8 MFMAs on one dependent accumulator chain",
    "mfma_four_chain64": "64 FP8 MFMAs round-robin over four accumulator chains",
    "vmem_load_cold": "one untouched 1024-byte-per-wave buffer_load_dwordx4 request plus vmcnt(0)",
    "vmem_load_l2_hit": "same address immediately after vmem_load_cold plus vmcnt(0)",
    "vmem_load_burst8": "eight untouched 1024-byte-per-wave loads plus vmcnt(0)",
    "vmem_load_wait_after_burst8": "vmcnt(0) timed after issuing eight untouched loads",
    "vmem_store_single": "one 1024-byte-per-wave store plus vmcnt(0)",
    "vmem_store_burst8": "eight 1024-byte-per-wave stores plus vmcnt(0)",
    "vmem_store_wait_after_burst8": "vmcnt(0) timed after issuing eight stores",
    "lds_read_single": "one balanced ds_read_b128 plus lgkmcnt(0)",
    "lds_read_burst8": "eight balanced ds_read_b128 requests plus lgkmcnt(0)",
    "lds_read_wait_after_burst8": "lgkmcnt(0) timed after issuing eight ds_read_b128 requests",
    "lds_write_single": "one balanced ds_write_b128 plus lgkmcnt(0)",
    "lds_write_burst8": "eight balanced ds_write_b128 requests plus lgkmcnt(0)",
    "lds_write_wait_after_burst8": "lgkmcnt(0) timed after issuing eight ds_write_b128 requests",
}


def percentile(values, fraction):
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def distribution(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "samples": int(array.size),
        "min": float(array.min()),
        "p50": percentile(array, 0.50),
        "p90": percentile(array, 0.90),
        "p95": percentile(array, 0.95),
        "p99": percentile(array, 0.99),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def infer_physical_device(explicit_device):
    if explicit_device is not None:
        return explicit_device
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if visible.isdigit():
        return int(visible)
    raise RuntimeError(
        "pass --physical-device or set HIP_VISIBLE_DEVICES to one physical GPU index"
    )


def run_json(command, env=None):
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout), result.stderr


def amdsmi_command(root, *arguments):
    cli = root / "libexec/amdsmi_cli/amdsmi_cli.py"
    if not cli.is_file():
        raise RuntimeError(f"AMDSMI 26.2.2 CLI not found under {root}")
    env = os.environ.copy()
    python_paths = [root / "share/amd_smi", root / "libexec/amdsmi_cli"]
    library_paths = [root / "lib", root / "share/amd_smi/amdsmi"]
    env["PYTHONPATH"] = ":".join(str(path) for path in python_paths) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths) + (
        f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
    )
    return [sys.executable, str(cli), *arguments], env


def read_gpu_state(physical_device, amdsmi_root):
    rocm_payload, rocm_stderr = run_json(
        [
            "rocm-smi",
            "-d",
            str(physical_device),
            "--showuse",
            "--showmemuse",
            "--showclocks",
            "--showperflevel",
            "--showpower",
            "--showpids",
            "--json",
        ]
    )
    card = rocm_payload[f"card{physical_device}"]
    static_command, env = amdsmi_command(
        amdsmi_root,
        "static",
        "-g",
        str(physical_device),
        "--limit",
        "--json",
    )
    static_payload, static_stderr = run_json(static_command, env)
    limit = static_payload["gpu_data"][0]["limit"]
    return {
        "physical_device": physical_device,
        "gpu_busy_percent": int(card["GPU use (%)"]),
        "vram_allocated_percent": int(card["GPU Memory Allocated (VRAM%)"]),
        "memory_activity_percent": int(card["GPU Memory Read/Write Activity (%)"]),
        "performance_level": card["Performance Level"],
        "sclk": card["sclk clock speed:"],
        "mclk": card["mclk clock speed:"],
        "fclk": card["fclk clock speed:"],
        "socket_power_w": float(card["Current Socket Graphics Package Power (W)"]),
        "ptl_state": limit["ptl_state"],
        "ptl_format": limit["ptl_format"],
        "power_cap_w": limit["ppt0"]["socket_power_limit"]["value"],
        "numa_balancing": int(
            Path("/proc/sys/kernel/numa_balancing").read_text().strip()
        ),
        "rocm_smi_stderr": rocm_stderr.strip(),
        "amdsmi_stderr": static_stderr.strip(),
    }


def set_experiment_state(physical_device, amdsmi_root):
    set_status, env = amdsmi_command(
        amdsmi_root,
        "set",
        "-g",
        str(physical_device),
        "-S",
        "1",
    )
    subprocess.run(set_status, check=True, env=env)
    set_format, env = amdsmi_command(
        amdsmi_root,
        "set",
        "-g",
        str(physical_device),
        "-F",
        "VECTOR,F8",
    )
    subprocess.run(set_format, check=True, env=env)
    subprocess.run(
        ["rocm-smi", "-d", str(physical_device), "--setperfdeterminism", "1800"],
        check=True,
    )


def restore_experiment_state(physical_device, amdsmi_root, original_state):
    errors = []
    try:
        subprocess.run(
            ["rocm-smi", "-d", str(physical_device), "--resetperfdeterminism"],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    try:
        subprocess.run(
            ["rocm-smi", "-d", str(physical_device), "--setperflevel", "auto"],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    try:
        if original_state["ptl_format"] not in {"N/A", "UNKNOWN,UNKNOWN"}:
            set_format, env = amdsmi_command(
                amdsmi_root,
                "set",
                "-g",
                str(physical_device),
                "-F",
                original_state["ptl_format"],
            )
            subprocess.run(set_format, check=True, env=env)
        set_status, env = amdsmi_command(
            amdsmi_root,
            "set",
            "-g",
            str(physical_device),
            "-S",
            "1" if original_state["ptl_state"] == "Enabled" else "0",
        )
        subprocess.run(set_status, check=True, env=env)
    except subprocess.CalledProcessError as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("failed to restore GPU state: " + "; ".join(errors))


@jit(no_pass=["pass_dse", "pass_dce"])
def measure_hardware_distribution(
    jit_builder: JIT,
    samples,
    bytes_per_sample,
    bytes_per_block,
    bytes_per_wave,
    data_bytes,
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    stores: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    jit = jit_builder
    lds_base = jit.alloc_lds(LDS_BYTES, align=16)
    lane = jit.lane_id
    wave = jit.warp_id[0]
    block = jit.blockIdx.x[0]

    record_dwords = META_DWORDS + samples * len(METRIC_NAMES)
    record_index = jit.gpr("su32", block * WAVES_PER_BLOCK + wave)
    record_byte_offset = jit.gpr("su32", record_index[0] * record_dwords * 4)

    simd_id = jit.gpr("su32")
    cu_id = jit.gpr("su32")
    se_id = jit.gpr("su32")
    xcc_id = jit.gpr("su32")
    jit.s_getreg_b32(simd_id, mod="hwreg(HW_REG_HW_ID, 4, 2)")
    jit.s_getreg_b32(cu_id, mod="hwreg(HW_REG_HW_ID, 8, 4)")
    jit.s_getreg_b32(se_id, mod="hwreg(HW_REG_HW_ID, 13, 3)")
    jit.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")

    sample_base = jit.gpr(
        "su32",
        block * bytes_per_block + wave * bytes_per_wave,
    )
    lane_offset = jit.gpr("vu32", lane[0] * 16)
    offsets = jit.gpr(LOADS_PER_BURST, "vu32")
    wait_offsets = jit.gpr(LOADS_PER_BURST, "vu32")
    for index in range(LOADS_PER_BURST):
        offsets[index] = lane_offset[0] + (index + 1) * BYTES_PER_REQUEST
        wait_offsets[index] = (
            lane_offset[0] + (index + LOADS_PER_BURST + 1) * BYTES_PER_REQUEST
        )

    data_buffer = jit.Buffer(data, data_bytes)
    store_buffer = jit.Buffer(stores, data_bytes)
    values = jit.gpr(LOADS_PER_BURST, 4, "vu32", align=4)
    store_values = jit.gpr(4, "vu32", 0x12345678, align=4)
    mfma_a = jit.gpr(2, "vu32", 0x40404040, align=2)
    mfma_b = jit.gpr(2, "vu32", 0x40404040, align=2)
    mfma_accumulators = jit.gpr(32, 4, "af32", align=4)
    mfma_accumulators[...] = 0.0
    dependent_accumulator = mfma_accumulators[0]

    lds_addresses = jit.gpr(4, "vu32")
    for index in range(4):
        lds_addresses[index] = lds_base + wave * 4096 + lane[0] * 16 + index * 1024
        jit.ds_write_b128(lds_addresses[index], store_values)
    jit.s_waitcnt(mod="lgkmcnt(0)")
    jit.s_barrier()

    start = jit.gpr(2, "su32", align=2)
    stop = jit.gpr(2, "su32", align=2)
    kernel_start = jit.gpr(2, "su32", align=2)
    kernel_end = jit.gpr(2, "su32", align=2)
    jit.s_memtime(kernel_start)
    jit.s_waitcnt(mod="lgkmcnt(0)")

    def begin():
        jit.s_memtime(start)

    def finish(wait_kind):
        if wait_kind == "vm":
            jit.s_waitcnt(mod="vmcnt(0)")
        elif wait_kind == "lds":
            jit.s_waitcnt(mod="lgkmcnt(0)")
        jit.s_memtime(stop)
        jit.s_waitcnt(mod="lgkmcnt(0)")
        jit.s_sub_u32(stop[0], stop[0], start[0])
        jit.s_subb_u32(stop[1], stop[1], start[1])
        value = jit.gpr("su32")
        value[0] = stop[0]
        return value

    sample_index = jit.gpr("su32", 0)
    sample_byte_offset = jit.gpr("su32", record_byte_offset[0] + META_DWORDS * 4)
    with jit.While(sample_index[0] < samples):
        measured = []

        begin()
        measured.append(finish("none"))

        begin()
        for _ in range(64):
            jit.v_mfma_f32_16x16x32_fp8_fp8(
                dependent_accumulator,
                mfma_a,
                mfma_b,
                dependent_accumulator,
            )
        measured.append(finish("none"))

        begin()
        for index in range(64):
            chain = index % 4
            jit.v_mfma_f32_16x16x32_fp8_fp8(
                mfma_accumulators[chain + 1],
                mfma_a,
                mfma_b,
                mfma_accumulators[chain + 1],
            )
        measured.append(finish("none"))

        begin()
        data_buffer.load_dwordx4(values[0], lane_offset, sample_base)
        measured.append(finish("vm"))

        begin()
        data_buffer.load_dwordx4(values[1], lane_offset, sample_base)
        measured.append(finish("vm"))

        begin()
        for index in range(LOADS_PER_BURST):
            data_buffer.load_dwordx4(values[index], offsets[index], sample_base)
        measured.append(finish("vm"))

        for index in range(LOADS_PER_BURST):
            data_buffer.load_dwordx4(values[index], wait_offsets[index], sample_base)
        begin()
        measured.append(finish("vm"))

        begin()
        store_buffer.store_dwordx4(store_values, lane_offset, sample_base)
        measured.append(finish("vm"))

        begin()
        for index in range(LOADS_PER_BURST):
            store_buffer.store_dwordx4(store_values, offsets[index], sample_base)
        measured.append(finish("vm"))

        for index in range(LOADS_PER_BURST):
            store_buffer.store_dwordx4(store_values, wait_offsets[index], sample_base)
        begin()
        measured.append(finish("vm"))

        begin()
        jit.ds_read_b128(values[0], lds_addresses[0])
        measured.append(finish("lds"))

        begin()
        for index in range(LOADS_PER_BURST):
            jit.ds_read_b128(values[index], lds_addresses[index % 4])
        measured.append(finish("lds"))

        for index in range(LOADS_PER_BURST):
            jit.ds_read_b128(values[index], lds_addresses[index % 4])
        begin()
        measured.append(finish("lds"))

        begin()
        jit.ds_write_b128(lds_addresses[0], store_values)
        measured.append(finish("lds"))

        begin()
        for index in range(LOADS_PER_BURST):
            jit.ds_write_b128(lds_addresses[index % 4], store_values)
        measured.append(finish("lds"))

        for index in range(LOADS_PER_BURST):
            jit.ds_write_b128(lds_addresses[index % 4], store_values)
        begin()
        measured.append(finish("lds"))

        for index, value in enumerate(measured):
            jit.s_store_dword(
                value,
                output,
                sample_byte_offset[0] + index * 4,
                mod="glc",
            )
        jit.s_waitcnt(mod="lgkmcnt(0)")
        sample_index[0] += 1
        sample_base[0] += bytes_per_sample
        sample_byte_offset[0] += len(METRIC_NAMES) * 4

    sink = jit.gpr("vu32", 0)
    for index in range(LOADS_PER_BURST):
        sink[0] ^= values[index, 0]
    mfma_sink = jit.gpr("vu32")
    for group in range(32):
        for element in range(4):
            jit.v_accvgpr_read_b32(mfma_sink, mfma_accumulators[group, element])
            sink[0] ^= mfma_sink[0]
    sink_scalar = jit.gpr("su32")
    jit.v_readfirstlane_b32(sink_scalar, sink)
    jit.s_store_dword(sink_scalar, stores, record_index[0] * 4, mod="glc")

    jit.s_memtime(kernel_end)
    jit.s_waitcnt(mod="lgkmcnt(0)")
    metadata = (
        xcc_id,
        se_id,
        cu_id,
        simd_id,
        jit.gpr("su32", block),
        jit.gpr("su32", wave),
        kernel_start[0],
        kernel_start[1],
        kernel_end[0],
        kernel_end[1],
    )
    for index, value in enumerate(metadata):
        jit.s_store_dword(
            value,
            output,
            record_byte_offset[0] + index * 4,
            mod="glc",
        )
    jit.s_waitcnt(mod="lgkmcnt(0)")


def resident_pair_indices(mapping):
    pairs = []
    for waves in mapping["mapping"].values():
        indices = [wave["block"] * WAVES_PER_BLOCK + wave["wave"] for wave in waves]
        if len(indices) != 2:
            raise RuntimeError(f"invalid resident pair: {waves}")
        pairs.append(indices)
    return np.asarray(pairs, dtype=np.int64)


def summarize(records, mapping):
    result = {}
    raw_overhead = records[:, :, 0].reshape(-1).astype(np.int64)
    timer_baseline = percentile(raw_overhead, 0.50)
    pair_indices = resident_pair_indices(mapping)

    for column, name in enumerate(METRIC_NAMES):
        values = records[:, :, column].reshape(-1).astype(np.int64)
        corrected = (
            values.astype(np.float64) - timer_baseline if column else values.copy()
        )
        paired = records[pair_indices, :, column]
        pair_min = paired.min(axis=1).reshape(-1).astype(np.float64)
        pair_max = paired.max(axis=1).reshape(-1).astype(np.float64)
        pair_skew = (
            (paired.max(axis=1) - paired.min(axis=1)).reshape(-1).astype(np.float64)
        )
        pair_left = paired[:, 0, :].reshape(-1).astype(np.float64)
        pair_right = paired[:, 1, :].reshape(-1).astype(np.float64)
        combined = np.concatenate((pair_left, pair_right))
        lower_quartile = percentile(combined, 0.25)
        upper_quartile = percentile(combined, 0.75)
        both_upper = np.mean(
            (pair_left > upper_quartile) & (pair_right > upper_quartile)
        )
        opposite_quartiles = np.mean(
            ((pair_left > upper_quartile) & (pair_right < lower_quartile))
            | ((pair_right > upper_quartile) & (pair_left < lower_quartile))
        )
        if column:
            pair_min -= timer_baseline
            pair_max -= timer_baseline
        result[name] = {
            "definition": METRIC_DEFINITIONS[name],
            "raw_cycles": distribution(values),
            "fixed_timer_baseline_cycles": timer_baseline,
            "fixed_timer_corrected_cycles": distribution(corrected),
            "resident_pair_same_ordinal": {
                "definition": (
                    "min/max/skew across the two mapped resident waves at the "
                    "same loop ordinal; ordinals are not hardware issue timestamps"
                ),
                "min_fixed_timer_corrected_cycles": distribution(pair_min),
                "max_fixed_timer_corrected_cycles": distribution(pair_max),
                "skew_cycles": distribution(pair_skew),
                "pearson_correlation": float(np.corrcoef(pair_left, pair_right)[0, 1]),
                "both_upper_quartile_fraction": float(both_upper),
                "opposite_quartiles_fraction": float(opposite_quartiles),
            },
            "raw": [int(value) for value in values],
            "fixed_timer_corrected_raw": [float(value) for value in corrected],
        }

    for name, source in (
        ("mfma_dependent_cycles_per_instruction", "mfma_dependent64"),
        ("mfma_four_chain_cycles_per_instruction", "mfma_four_chain64"),
    ):
        values = (
            np.asarray(result[source]["fixed_timer_corrected_raw"], dtype=np.float64)
            / 64.0
        )
        result[name] = {
            "definition": f"fixed-timer-corrected {source} divided by 64",
            **distribution(values),
            "raw": [float(value) for value in values],
        }

    paired_metrics = {
        "vmem_load_service_interval": ("vmem_load_burst8", "vmem_load_cold"),
        "vmem_store_service_interval": (
            "vmem_store_burst8",
            "vmem_store_single",
        ),
        "lds_read_service_interval": ("lds_read_burst8", "lds_read_single"),
        "lds_write_service_interval": ("lds_write_burst8", "lds_write_single"),
    }
    for name, (burst_name, single_name) in paired_metrics.items():
        burst = np.asarray(
            result[burst_name]["fixed_timer_corrected_raw"], dtype=np.float64
        )
        single = np.asarray(
            result[single_name]["fixed_timer_corrected_raw"], dtype=np.float64
        )
        values = (burst - single) / (LOADS_PER_BURST - 1)
        result[name] = {
            "definition": (
                f"paired same-wave same-sample ({burst_name} - {single_name}) / "
                f"{LOADS_PER_BURST - 1}; negative values are retained"
            ),
            **distribution(values),
            "negative_fraction": float(np.mean(values < 0)),
            "raw": [float(value) for value in values],
        }
    return result


def validate_mapping(metadata, num_cu):
    by_simd = defaultdict(list)
    for row in metadata:
        xcc, se, cu, simd, block, wave = (int(value) for value in row[:6])
        begin = int(row[6]) | (int(row[7]) << 32)
        end = int(row[8]) | (int(row[9]) << 32)
        by_simd[(xcc, se, cu, simd)].append(
            {"block": block, "wave": wave, "begin": begin, "end": end}
        )
    expected_simds = num_cu * 4
    if len(by_simd) != expected_simds:
        raise RuntimeError(
            f"expected {expected_simds} physical SIMDs, got {len(by_simd)}"
        )

    overlap_fractions = []
    for key, waves in by_simd.items():
        if len(waves) != 2:
            raise RuntimeError(f"{key} has {len(waves)} resident waves, expected 2")
        if len({wave["block"] for wave in waves}) != 2:
            raise RuntimeError(f"{key} waves do not come from two workgroups: {waves}")
        left = max(wave["begin"] for wave in waves)
        right = min(wave["end"] for wave in waves)
        shortest = min(wave["end"] - wave["begin"] for wave in waves)
        overlap = max(0, right - left) / shortest
        if overlap < 0.95:
            raise RuntimeError(
                f"{key} resident-wave overlap is only {overlap:.2%}: {waves}"
            )
        overlap_fractions.append(overlap)
    return {
        "physical_simds": len(by_simd),
        "waves_per_physical_simd": 2,
        "overlap_fraction_min": min(overlap_fractions),
        "overlap_fraction_p50": percentile(overlap_fractions, 0.50),
        "overlap_fraction_p95": percentile(overlap_fractions, 0.95),
        "mapping": {
            "/".join(str(value) for value in key): waves
            for key, waves in sorted(by_simd.items())
        },
    }


def parse_resource_log(path):
    text = path.read_text()
    fields = {
        "vgpr": r"VGPRs: (\d+)",
        "agpr": r"AGPRs: (\d+)",
        "scratch_bytes_per_lane": r"ScratchSize \[bytes/lane\]: (\d+)",
        "lds_bytes_per_block": r"LDS Size \[bytes/block\]: (\d+)",
    }
    resources = {}
    for name, pattern in fields.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"{name} not found in resource log {path}")
        resources[name] = int(matches[-1])
    resources["path"] = str(path.resolve())
    resources["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return resources


def compact_payload(payload, full_path, full_text, resource_log=None):
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"mapping_validation", "results"}
    }
    summary["mapping_validation"] = {
        key: value
        for key, value in payload["mapping_validation"].items()
        if key != "mapping"
    }
    summary["results"] = {
        name: {
            key: value
            for key, value in result.items()
            if key not in {"raw", "fixed_timer_corrected_raw"}
        }
        for name, result in payload["results"].items()
    }
    summary["full_raw_artifact"] = {
        "path": str(full_path.resolve()),
        "bytes": len(full_text.encode()),
        "sha256": hashlib.sha256(full_text.encode()).hexdigest(),
    }
    if resource_log:
        summary["observed_resources"] = parse_resource_log(resource_log)
    return summary


def records_from_payload(payload):
    samples = payload["config"]["samples_per_wave"]
    wave_records = payload["config"]["blocks"] * payload["config"]["waves_per_block"]
    records = np.empty((wave_records, samples, len(METRIC_NAMES)), dtype=np.uint32)
    for column, name in enumerate(METRIC_NAMES):
        raw = np.asarray(payload["results"][name]["raw"], dtype=np.uint32)
        if raw.size != wave_records * samples:
            raise RuntimeError(
                f"{name} has {raw.size} raw samples, expected {wave_records * samples}"
            )
        records[:, :, column] = raw.reshape(wave_records, samples)
    return records


def resummarize(full_path, summary_path, resource_log):
    original_text = full_path.read_text()
    payload = json.loads(original_text)
    old_results = payload["results"]
    records = records_from_payload(payload)
    new_results = summarize(records, payload["mapping_validation"])
    for name in METRIC_NAMES:
        if new_results[name]["raw"] != old_results[name]["raw"]:
            raise RuntimeError(f"raw samples changed while resummarizing {name}")
    payload["results"] = new_results
    payload["postprocess"] = {
        "note": (
            "Summary recomputed from unchanged raw arrays with fixed-P50 timer "
            "correction and resident-pair statistics"
        )
    }
    summary = compact_payload(payload, full_path, original_text, resource_log)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def check_original_state(original_state):
    required = {
        "performance_level": "auto",
        "ptl_state": "Enabled",
        "ptl_format": "F16,BF16",
        "power_cap_w": 650,
        "numa_balancing": 1,
    }
    mismatches = {
        key: (expected, original_state[key])
        for key, expected in required.items()
        if original_state[key] != expected
    }
    if mismatches:
        raise RuntimeError(
            "formal state management requires the known restorable baseline: "
            f"{mismatches}"
        )


def collect(args):
    if not args.json:
        raise RuntimeError("--json is required for GPU collection")
    physical_device = infer_physical_device(args.physical_device)
    original_state = read_gpu_state(physical_device, args.amdsmi_root)
    busy_reasons = []
    if original_state["gpu_busy_percent"] > args.max_gpu_busy:
        busy_reasons.append(
            f"GPU busy {original_state['gpu_busy_percent']}% > {args.max_gpu_busy}%"
        )
    if original_state["vram_allocated_percent"] > args.max_vram_percent:
        busy_reasons.append(
            "VRAM allocated "
            f"{original_state['vram_allocated_percent']}% > {args.max_vram_percent}%"
        )
    if busy_reasons and not args.allow_busy:
        raise RuntimeError(
            "; ".join(busy_reasons)
            + "; refusing formal collection; use --allow-busy only for smoke"
        )

    state_was_managed = not args.no_manage_state
    if state_was_managed:
        check_original_state(original_state)
    if JIT.gfx != 942:
        raise RuntimeError(f"this probe is only validated on gfx942, got gfx{JIT.gfx}")

    managed_state = None
    restored_state = None
    run_error = None
    try:
        if state_was_managed:
            set_experiment_state(physical_device, args.amdsmi_root)
            managed_state = read_gpu_state(physical_device, args.amdsmi_root)
            if (
                managed_state["ptl_state"] != "Enabled"
                or managed_state["ptl_format"] != "VECTOR,F8"
                or managed_state["performance_level"] != "perf_determinism"
            ):
                raise RuntimeError(
                    f"failed to enter controlled experiment state: {managed_state}"
                )
            if managed_state["numa_balancing"] != original_state["numa_balancing"]:
                raise RuntimeError(
                    "NUMA balancing changed while entering experiment state"
                )

        torch.set_default_device("cuda")
        properties = torch.cuda.get_device_properties(0)
        num_cu = properties.multi_processor_count
        if num_cu != 80:
            raise RuntimeError(f"this probe expects 80 CUs, got {num_cu}")
        blocks = num_cu * 2
        wave_records = blocks * WAVES_PER_BLOCK
        bytes_per_wave = (2 * LOADS_PER_BURST + 1) * BYTES_PER_REQUEST
        bytes_per_block = WAVES_PER_BLOCK * bytes_per_wave
        bytes_per_sample = blocks * bytes_per_block
        required_data_bytes = args.samples * bytes_per_sample
        data_bytes = args.data_mib * 1024 * 1024
        if data_bytes < required_data_bytes:
            raise RuntimeError(
                f"--data-mib must be at least "
                f"{(required_data_bytes + 1024**2 - 1) // 1024**2}"
            )

        data = torch.arange(data_bytes // 4, dtype=torch.int32)
        stores = torch.zeros_like(data)
        record_dwords = META_DWORDS + args.samples * len(METRIC_NAMES)
        output = torch.zeros(wave_records, record_dwords, dtype=torch.uint32)

        measure_hardware_distribution(
            [blocks],
            [THREADS],
            args.samples,
            bytes_per_sample,
            bytes_per_block,
            bytes_per_wave,
            data_bytes,
            data.data_ptr(),
            stores.data_ptr(),
            output.data_ptr(),  # pyright: ignore[reportCallIssue]
        )
        torch.cuda.synchronize()
        host = output.cpu().numpy()
        metadata = host[:, :META_DWORDS]
        records = host[:, META_DWORDS:].reshape(
            wave_records, args.samples, len(METRIC_NAMES)
        )
        if not np.all(records > 0):
            raise RuntimeError("one or more cycle records are zero")
        mapping = validate_mapping(metadata, num_cu)

        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "formal_result": (
                not args.smoke and not args.allow_busy and state_was_managed
            ),
            "invalidity_reasons": (
                (["smoke mode"] if args.smoke else [])
                + (
                    ["busy preflight overridden"]
                    if args.allow_busy and busy_reasons
                    else []
                )
                + (["GPU state was not managed"] if not state_was_managed else [])
            ),
            "device": {
                "physical_device": physical_device,
                "name": properties.name,
                "gcn_arch": properties.gcnArchName,
                "num_cu": num_cu,
                "pyhip_arch": JIT.arch,
                "pyhip_gfx": JIT.gfx,
            },
            "state": {
                "original": original_state,
                "managed": managed_state,
                "restored": None,
            },
            "config": {
                "blocks": blocks,
                "blocks_per_cu": 2,
                "threads": THREADS,
                "waves_per_block": WAVES_PER_BLOCK,
                "lds_bytes_per_block": LDS_BYTES,
                "expected_resources": {
                    "vgpr": 64,
                    "agpr": 128,
                    "scratch_bytes_per_lane": 0,
                },
                "expected_waves_per_simd": 2,
                "samples_per_wave": args.samples,
                "samples_per_metric": args.samples * wave_records,
                "data_bytes": data_bytes,
                "busy_limits": {
                    "gpu_busy_percent": args.max_gpu_busy,
                    "vram_allocated_percent": args.max_vram_percent,
                },
            },
            "mapping_validation": mapping,
            "results": summarize(records, mapping),
        }
    except BaseException as error:
        run_error = error
        raise
    finally:
        if state_was_managed:
            try:
                restore_experiment_state(
                    physical_device, args.amdsmi_root, original_state
                )
                restored_state = read_gpu_state(physical_device, args.amdsmi_root)
            except BaseException as restore_error:
                if run_error is not None:
                    raise RuntimeError(
                        f"experiment failed with {run_error!r}, and GPU state "
                        "restoration also failed"
                    ) from restore_error
                raise

    payload["state"]["restored"] = restored_state
    if restored_state is not None:
        for key in (
            "performance_level",
            "ptl_state",
            "ptl_format",
            "numa_balancing",
        ):
            if restored_state[key] != original_state[key]:
                raise RuntimeError(
                    f"GPU state restoration mismatch for {key}: "
                    f"{original_state[key]} -> {restored_state[key]}"
                )
    full_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.json.write_text(full_text)
    if args.summary_json:
        summary = compact_payload(payload, args.json, full_text, args.resource_log)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--data-mib", type=int, default=768)
    parser.add_argument("--physical-device", type=int)
    parser.add_argument("--max-gpu-busy", type=int, default=5)
    parser.add_argument("--max-vram-percent", type=int, default=20)
    parser.add_argument("--amdsmi-root", type=Path, default=DEFAULT_AMDSMI_ROOT)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--no-manage-state", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--resource-log", type=Path)
    parser.add_argument("--resummarize-full", type=Path)
    args = parser.parse_args()

    if args.resummarize_full:
        if not args.summary_json:
            parser.error("--summary-json is required with --resummarize-full")
        summary = resummarize(
            args.resummarize_full, args.summary_json, args.resource_log
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.smoke and args.samples == 64:
        args.samples = 1
    payload = collect(args)
    printable = compact_payload(payload, args.json, args.json.read_text())
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
