#!/usr/bin/env python3
"""Model physical N256 PTPC prefetch/store credit under strict 8-wave anti-phase.

Each 8-wave workgroup contains two 4-wave groups. Real barriers offset them by
one stage, so exactly four waves execute the memory stage while their same-SIMD
peers execute 64 BF16 MFMAs. Every modeled N block has three K128 cores.

The memory stage models, per wave and K core:

* eight 128-bit weight loads (N64 x K128 FP8), optionally split 4 + 4;
* eight activation LDS reads;
* delayed output stores distributed as 2/2/4 or 3/3/2;
* one extra PTPC scale load in K2;
* optional ``vmcnt(6)`` before the four tail loads.

ATT analysis compares VMEM issue stalls, final completion waits, stage barriers,
and the empirical request-credit knee while holding total work constant.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from pyhip.core.asmjit import JIT, jit

VOID_POINTER = "void*"
WAVES_PER_BLOCK = 8
THREADS_PER_BLOCK = 512
LDS_BYTES = 64 * 1024
WEIGHT_LOADS_PER_CORE = 8
LDS_READS_PER_CORE = 8
MFMA_PER_CORE = 64
MFMA_OPCODE = "v_mfma_f32_16x16x32_fp8_fp8"
K_CORES = 3
SCALE_LOAD_CORE = 2
BYTES_PER_WAVE_LOAD = 1024
OUTPUT_BYTES_PER_WAVE = 8 * BYTES_PER_WAVE_LOAD
METADATA_DWORDS = 8
MEMORY_WAVES_PER_CU = 4
CREDIT_KNEE_PER_CU = 48
MFMA_EXECUTION_CYCLES = 16
SCHEDULES = ("burst", "split", "credit")
MARKERS = {
    2: (0, "memory"),
    3: (0, "compute"),
    4: (1, "memory"),
    5: (1, "compute"),
    6: (2, "memory"),
    7: (2, "compute"),
}
WAVE_PATH_PATTERN = re.compile(
    r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json$"
)
INCOMPLETE_TRACE_PATTERN = re.compile(
    r"stitch incomplete|wave incomplete|trace was cutoff|"
    r"parser could not fully match",
    flags=re.IGNORECASE,
)


def _read_realtime(builder):
    value = builder.gpr(2, "su32", align=2)
    builder.s_memrealtime(value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    return value


def _make_mfma_registers(builder):
    operand_a = builder.gpr(2, "vu32", 0x40404040, align=2)
    operand_b = builder.gpr(2, "vu32", 0x40404040, align=2)
    accumulators = builder.gpr(4, 4, "vf32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b


def _emit_mfma(builder, accumulators, operand_a, operand_b):
    for index in range(MFMA_PER_CORE):
        builder.v_mfma_f32_16x16x32_fp8_fp8(
            accumulators[index & 3],
            operand_a,
            operand_b,
            accumulators[index & 3],
        )


def _emit_memory_core(
    builder,
    core,
    store_count,
    store_begin,
    schedule,
    data_buffer,
    output_buffer,
    weight_values,
    scale_value,
    ds_values,
    store_value,
    load_sink,
    vector_offset,
    data_offset,
    output_base,
    lds_addresses,
    stream_stride,
    data_mask,
):
    for store_index in range(store_count):
        output_buffer.store_dwordx4(
            store_value,
            vector_offset,
            output_base + (store_begin + store_index) * BYTES_PER_WAVE_LOAD,
        )

    if core == SCALE_LOAD_CORE:
        data_buffer.load_dwordx4(
            scale_value,
            vector_offset,
            data_offset,
            non_temporal=True,
        )
        data_offset[0] += stream_stride
        data_offset[0] = data_offset[0] & data_mask

    head_count = 4 if schedule in ("split", "credit") else WEIGHT_LOADS_PER_CORE
    for load_index in range(head_count):
        data_buffer.load_dwordx4(
            weight_values[load_index],
            vector_offset,
            data_offset,
            non_temporal=True,
        )
        data_offset[0] += stream_stride
        data_offset[0] = data_offset[0] & data_mask

    for read_index in range(LDS_READS_PER_CORE):
        builder.ds_read_b128(ds_values[read_index], lds_addresses[read_index])
    builder.s_waitcnt(mod="lgkmcnt(0)")
    for read_index in range(LDS_READS_PER_CORE):
        builder.v_xor_b32(
            load_sink, load_sink, ds_values[read_index, 0]
        )

    if schedule == "credit" and core == SCALE_LOAD_CORE:
        builder.s_waitcnt(mod="vmcnt(6)")

    if schedule in ("split", "credit"):
        for load_index in range(4, WEIGHT_LOADS_PER_CORE):
            data_buffer.load_dwordx4(
                weight_values[load_index],
                vector_offset,
                data_offset,
                non_temporal=True,
            )
            data_offset[0] += stream_stride
            data_offset[0] = data_offset[0] & data_mask

    builder.s_waitcnt(mod="vmcnt(0)")
    for load_index in range(WEIGHT_LOADS_PER_CORE):
        builder.v_xor_b32(
            load_sink, load_sink, weight_values[load_index, 0]
        )
    if core == SCALE_LOAD_CORE:
        builder.v_xor_b32(load_sink, load_sink, scale_value[0])


@jit(no_pass=["pass_dse", "pass_dce"])
def strict_8wave_n256_prefetch(
    builder: JIT,
    rounds,
    store_k0,
    store_k1,
    store_k2,
    schedule,
    data_bytes,
    output_bytes,
    total_waves,
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    metadata: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    assert schedule in SCHEDULES
    assert store_k0 + store_k1 + store_k2 == 8
    lds_base = builder.alloc_lds(LDS_BYTES, align=16)
    data_buffer = builder.Buffer(data, data_bytes)
    output_buffer = builder.Buffer(output, output_bytes)
    weight_values = builder.gpr(WEIGHT_LOADS_PER_CORE, 4, "vu32", align=4)
    scale_value = builder.gpr(4, "vu32", align=4)
    ds_values = builder.gpr(LDS_READS_PER_CORE, 4, "vu32", align=4)
    store_value = builder.gpr(4, "vu32", 0x12345678, align=4)
    load_sink = builder.gpr("vu32", 0)
    vector_offset = builder.gpr("vu32", builder.lane_id[0] * 16)
    accumulators, operand_a, operand_b = _make_mfma_registers(builder)

    hw_id = builder.gpr("su32")
    xcc_id = builder.gpr("su32")
    builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    kernel_start = _read_realtime(builder)

    global_wave = builder.blockIdx.x[0] * WAVES_PER_BLOCK + builder.warp_id[0]
    data_offset = builder.gpr("su32", global_wave * BYTES_PER_WAVE_LOAD)
    output_base = builder.gpr("su32", global_wave * OUTPUT_BYTES_PER_WAVE)
    stream_stride = total_waves * BYTES_PER_WAVE_LOAD
    data_mask = data_bytes - 1
    lds_addresses = builder.gpr(LDS_READS_PER_CORE, "vu32")
    for read_index in range(LDS_READS_PER_CORE):
        lds_addresses[read_index] = (
            lds_base
            + builder.warp_id[0] * 8192
            + builder.lane_id[0] * 16
            + read_index * 1024
        )
        builder.ds_write_b128(lds_addresses[read_index], store_value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    builder.s_barrier()

    with builder.If(builder.warp_id[0] >= 4):
        builder.s_barrier()
    builder.s_barrier()

    store_counts = (store_k0, store_k1, store_k2)
    store_begins = (0, store_k0, store_k0 + store_k1)
    memory_markers = (2, 4, 6)
    compute_markers = (3, 5, 7)
    round_index = builder.gpr("su32", 0)
    with builder.While(round_index[0] < rounds):
        for core in range(K_CORES):
            builder.s_nop(memory_markers[core])
            _emit_memory_core(
                builder,
                core,
                store_counts[core],
                store_begins[core],
                schedule,
                data_buffer,
                output_buffer,
                weight_values,
                scale_value,
                ds_values,
                store_value,
                load_sink,
                vector_offset,
                data_offset,
                output_base,
                lds_addresses,
                stream_stride,
                data_mask,
            )
            builder.s_barrier()

            builder.s_nop(compute_markers[core])
            _emit_mfma(builder, accumulators, operand_a, operand_b)
            builder.s_barrier()
        round_index[0] += 1

    with builder.If(builder.warp_id[0] < 4):
        builder.s_barrier()

    kernel_stop = _read_realtime(builder)
    metadata_offset = builder.gpr("su32", global_wave * METADATA_DWORDS * 4)
    record = builder.gpr(METADATA_DWORDS, "su32", align=4)
    record[0] = builder.blockIdx.x[0]
    record[1] = builder.warp_id[0]
    record[2] = hw_id[0]
    record[3] = xcc_id[0]
    record[4] = kernel_start[0]
    record[5] = kernel_start[1]
    record[6] = kernel_stop[0]
    record[7] = kernel_stop[1]
    builder.s_store_dwordx4(record[0:3], metadata, metadata_offset, mod="glc")
    builder.s_store_dwordx4(
        record[4:7], metadata, metadata_offset + 16, mod="glc"
    )
    builder.s_waitcnt(mod="lgkmcnt(0)")

    sink = builder.gpr("su32")
    sink_component = builder.gpr("su32")
    builder.v_readfirstlane_b32(sink, accumulators[0, 0])
    builder.v_readfirstlane_b32(sink_component, load_sink)
    builder.s_xor_b32(sink, sink, sink_component)
    builder.s_xor_b32(sink, sink, hw_id)


def _u32(value):
    return int(value) & 0xFFFFFFFF


def _decode_metadata(row):
    values = [_u32(value) for value in row]
    hw_id = values[2]
    return {
        "block": values[0],
        "logical_wave": values[1],
        "xcc": values[3],
        "slot": hw_id & 0xF,
        "simd": (hw_id >> 4) & 0x3,
        "cu": (hw_id >> 8) & 0xF,
        "se": (hw_id >> 13) & 0x7,
        "start": values[4] | (values[5] << 32),
        "stop": values[6] | (values[7] << 32),
    }


def _validate_runtime_placement(metadata):
    groups = defaultdict(list)
    for row in metadata.tolist():
        wave = _decode_metadata(row)
        key = (wave["xcc"], wave["se"], wave["cu"], wave["simd"])
        groups[key].append(wave)
    failures = []
    pairs = 0
    for key, waves in sorted(groups.items()):
        slots = sorted(wave["slot"] for wave in waves)
        blocks = {wave["block"] for wave in waves}
        if len(waves) != 2 or slots != [0, 1] or len(blocks) != 1:
            failures.append(
                {"key": key, "count": len(waves), "slots": slots, "blocks": sorted(blocks)}
            )
        else:
            pairs += 1
    return {
        "wave_count": metadata.shape[0],
        "physical_simd_groups": len(groups),
        "valid_pairs": pairs,
        "placement_failures": failures,
        "valid": not failures and pairs == len(groups),
    }


def _summary(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def _parse_marker(instruction):
    match = re.fullmatch(r"s_nop\s+(0x[0-9a-f]+|\d+)", instruction.strip())
    if match is None:
        return None
    return MARKERS.get(int(match.group(1), 0))


def _static_map(ui_directory, store_split, schedule):
    code_path = ui_directory / "code.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))["code"]
    markers = defaultdict(list)
    for instruction_id, row in enumerate(code):
        marker = _parse_marker(row[0])
        if marker is not None:
            markers[marker].append(instruction_id)
    for core in range(K_CORES):
        for phase in ("memory", "compute"):
            if len(markers[(core, phase)]) != 1:
                raise RuntimeError(f"invalid marker {(core, phase)} in {code_path}")

    cores = []
    for core in range(K_CORES):
        memory_begin = markers[(core, "memory")][0]
        compute_begin = markers[(core, "compute")][0]
        memory_ids = range(memory_begin + 1, compute_begin)
        next_memory = (
            markers[(core + 1, "memory")][0]
            if core + 1 < K_CORES
            else len(code)
        )
        compute_ids = range(compute_begin + 1, next_memory)
        loads = [
            instruction_id
            for instruction_id in memory_ids
            if code[instruction_id][0].strip().startswith("buffer_load_dwordx4")
        ]
        stores = [
            instruction_id
            for instruction_id in memory_ids
            if code[instruction_id][0].strip().startswith("buffer_store_dwordx4")
        ]
        ds_reads = [
            instruction_id
            for instruction_id in memory_ids
            if code[instruction_id][0].strip().startswith("ds_read_b128")
        ]
        vm_waits = [
            instruction_id
            for instruction_id in memory_ids
            if code[instruction_id][0].strip().startswith("s_waitcnt")
            and "vmcnt" in code[instruction_id][0]
        ]
        memory_barriers = [
            instruction_id
            for instruction_id in memory_ids
            if code[instruction_id][0].strip().startswith("s_barrier")
        ]
        mfmas = [
            instruction_id
            for instruction_id in compute_ids
            if code[instruction_id][0].strip().split()[0] == MFMA_OPCODE
        ]
        compute_barriers = [
            instruction_id
            for instruction_id in compute_ids
            if code[instruction_id][0].strip().startswith("s_barrier")
        ]
        expected_loads = WEIGHT_LOADS_PER_CORE + (core == SCALE_LOAD_CORE)
        if len(loads) != expected_loads:
            raise RuntimeError(f"core {core} loads={len(loads)}, expected {expected_loads}")
        if len(stores) != store_split[core]:
            raise RuntimeError(f"core {core} stores={len(stores)}, expected {store_split[core]}")
        if len(ds_reads) != LDS_READS_PER_CORE:
            raise RuntimeError(f"core {core} ds_reads={len(ds_reads)}")
        expected_waits = (
            2 if schedule == "credit" and core == SCALE_LOAD_CORE else 1
        )
        if len(vm_waits) != expected_waits:
            raise RuntimeError(f"core {core} vm_waits={len(vm_waits)}, expected {expected_waits}")
        if len(memory_barriers) != 1 or not compute_barriers:
            raise RuntimeError(f"core {core} barrier mismatch")
        if len(mfmas) != MFMA_PER_CORE:
            raise RuntimeError(f"core {core} mfmas={len(mfmas)}")
        cores.append(
            {
                "memory_marker": memory_begin,
                "compute_marker": compute_begin,
                "loads": loads,
                "stores": stores,
                "ds_reads": ds_reads,
                "vm_waits": vm_waits,
                "memory_barrier": memory_barriers[0],
                "mfmas": mfmas,
                "compute_barrier": compute_barriers[0],
            }
        )
    return {"code": code, "cores": cores}


def _record(row):
    return {
        "attempt": int(row[0]),
        "stall": int(row[2]),
        "duration": int(row[3]),
        "instruction_id": int(row[4]),
        "issue": int(row[0]) + int(row[2]),
    }


def _load_wave(path, static, rounds):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["num_insts"] != payload["num_stitched"]:
        raise RuntimeError(f"incomplete wave {path}")
    wave = payload["wave"]
    match = WAVE_PATH_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"cannot parse {path}")
    se, filename_simd, filename_slot, wave_id = map(int, match.groups())
    if filename_simd != int(wave["simd"]) or filename_slot != int(wave["slot"]):
        raise RuntimeError(f"wave filename mismatch {path}")

    target_ids = set()
    for core in static["cores"]:
        target_ids.update(core["loads"])
        target_ids.update(core["stores"])
        target_ids.update(core["ds_reads"])
        target_ids.update(core["vm_waits"])
        target_ids.update(core["mfmas"])
        target_ids.update(
            (
                core["memory_marker"],
                core["compute_marker"],
                core["memory_barrier"],
                core["compute_barrier"],
            )
        )
    records_by_id = defaultdict(list)
    for row in wave["instructions"]:
        instruction_id = int(row[4])
        if instruction_id in target_ids:
            records_by_id[instruction_id].append(_record(row))
    for instruction_id in target_ids:
        if len(records_by_id[instruction_id]) != rounds:
            raise RuntimeError(
                f"{path} id={instruction_id} count={len(records_by_id[instruction_id])}"
            )

    core_rounds = []
    for core in static["cores"]:
        rows = []
        for round_index in range(rounds):
            loads = [records_by_id[value][round_index] for value in core["loads"]]
            stores = [records_by_id[value][round_index] for value in core["stores"]]
            ds_reads = [records_by_id[value][round_index] for value in core["ds_reads"]]
            waits = [records_by_id[value][round_index] for value in core["vm_waits"]]
            mfmas = [records_by_id[value][round_index] for value in core["mfmas"]]
            rows.append(
                {
                    "memory_marker": records_by_id[core["memory_marker"]][round_index],
                    "compute_marker": records_by_id[core["compute_marker"]][round_index],
                    "loads": loads,
                    "stores": stores,
                    "ds_reads": ds_reads,
                    "waits": waits,
                    "memory_barrier": records_by_id[core["memory_barrier"]][round_index],
                    "mfmas": mfmas,
                    "compute_barrier": records_by_id[core["compute_barrier"]][round_index],
                }
            )
        core_rounds.append(rows)
    return {
        "path": path.name,
        "key": (se, int(wave["cu"]), int(wave["simd"])),
        "slot": int(wave["slot"]),
        "wave_id": wave_id,
        "cores": core_rounds,
    }


def _intervals(records):
    return [
        (record["attempt"], record["issue"])
        for record in records
        if record["stall"]
    ]


def _overlap_cycles(left, right):
    left = sorted(left)
    right = sorted(right)
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left) and right_index < len(right):
        left_begin, left_end = left[left_index]
        right_begin, right_end = right[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_begin, right_begin))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _analyze_pair(left, right):
    core_results = []
    same_stage_overlap = 0
    opposite_stage_overlap = 0
    for core in range(K_CORES):
        left_rows = left["cores"][core]
        right_rows = right["cores"][core]
        left_vmem = [record for row in left_rows for record in row["loads"] + row["stores"]]
        right_vmem = [record for row in right_rows for record in row["loads"] + row["stores"]]
        left_wait = [row["waits"][-1] for row in left_rows]
        right_wait = [row["waits"][-1] for row in right_rows]
        left_compute_barrier = [row["compute_barrier"] for row in left_rows]
        right_compute_barrier = [row["compute_barrier"] for row in right_rows]
        vmem_intervals = _intervals(left_vmem) + _intervals(right_vmem)
        wait_intervals = _intervals(left_wait) + _intervals(right_wait)
        peer_barriers = _intervals(left_compute_barrier) + _intervals(right_compute_barrier)
        vmem_overlap = _overlap_cycles(_intervals(left_vmem), _intervals(right_compute_barrier)) + _overlap_cycles(_intervals(right_vmem), _intervals(left_compute_barrier))
        wait_overlap = _overlap_cycles(_intervals(left_wait), _intervals(right_compute_barrier)) + _overlap_cycles(_intervals(right_wait), _intervals(left_compute_barrier))

        left_memory = [(row["memory_marker"]["issue"], row["memory_barrier"]["issue"]) for row in left_rows]
        right_memory = [(row["memory_marker"]["issue"], row["memory_barrier"]["issue"]) for row in right_rows]
        left_compute = [(row["compute_marker"]["issue"], row["compute_barrier"]["issue"]) for row in left_rows]
        right_compute = [(row["compute_marker"]["issue"], row["compute_barrier"]["issue"]) for row in right_rows]
        same_stage_overlap += _overlap_cycles(left_memory, right_memory) + _overlap_cycles(left_compute, right_compute)
        opposite_stage_overlap += _overlap_cycles(left_memory, right_compute) + _overlap_cycles(right_memory, left_compute)
        core_results.append(
            {
                "core": core,
                "vmem_issue_stall_cycles": sum(end - begin for begin, end in vmem_intervals),
                "wait_vmcnt_stall_cycles": sum(end - begin for begin, end in wait_intervals),
                "peer_compute_barrier_stall_cycles": sum(end - begin for begin, end in peer_barriers),
                "vmem_peer_barrier_overlap_cycles": vmem_overlap,
                "wait_peer_barrier_overlap_cycles": wait_overlap,
            }
        )
    active_total = same_stage_overlap + opposite_stage_overlap
    return {
        "key": left["key"],
        "slots": sorted((left["slot"], right["slot"])),
        "wave_files": [left["path"], right["path"]],
        "active_same_stage_overlap_cycles": same_stage_overlap,
        "active_opposite_stage_overlap_cycles": opposite_stage_overlap,
        "active_same_stage_loss_rate": same_stage_overlap / active_total if active_total else 1.0,
        "cores": core_results,
    }


def _analyze_dispatch(
    ui_directory,
    rounds,
    store_split,
    schedule,
    ignore_edge_rounds,
):
    static = _static_map(ui_directory, store_split, schedule)
    waves = [_load_wave(path, static, rounds) for path in sorted(ui_directory.glob("se*.json"))]
    analyzed_rounds = rounds - 2 * ignore_edge_rounds
    for wave in waves:
        wave["cores"] = [
            rows[ignore_edge_rounds : rounds - ignore_edge_rounds]
            for rows in wave["cores"]
        ]
    groups = defaultdict(list)
    for wave in waves:
        groups[wave["key"]].append(wave)
    failures = []
    pairs = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda wave: wave["slot"])
        if len(group) != 2 or [wave["slot"] for wave in group] != [0, 1]:
            failures.append({"key": key, "count": len(group), "slots": [wave["slot"] for wave in group]})
        else:
            pairs.append(_analyze_pair(group[0], group[1]))

    cores = []
    for core in range(K_CORES):
        vmem_rows = []
        waits = []
        memory_barriers = []
        compute_barriers = []
        stage_spans = []
        credit_waits = []
        ordinal_stalls = defaultdict(list)
        for wave in waves:
            for row in wave["cores"][core]:
                vmem = sorted(row["stores"] + row["loads"], key=lambda record: (record["attempt"], record["instruction_id"]))
                vmem_rows.extend(vmem)
                for ordinal, record in enumerate(vmem):
                    ordinal_stalls[ordinal].append(record["stall"])
                waits.append(row["waits"][-1]["stall"])
                if len(row["waits"]) > 1:
                    credit_waits.append(row["waits"][0]["stall"])
                memory_barriers.append(row["memory_barrier"]["stall"])
                compute_barriers.append(row["compute_barrier"]["stall"])
                stage_spans.append(row["waits"][-1]["issue"] - row["memory_marker"]["issue"])
        pair_core = [pair["cores"][core] for pair in pairs]
        same_stage = sum(pair["active_same_stage_overlap_cycles"] for pair in pairs)
        opposite_stage = sum(pair["active_opposite_stage_overlap_cycles"] for pair in pairs)
        cores.append(
            {
                "core": core,
                "store_count_per_wave": store_split[core],
                "weight_loads_per_wave": WEIGHT_LOADS_PER_CORE,
                "scale_loads_per_wave": int(core == SCALE_LOAD_CORE),
                "nominal_requests_per_wave": store_split[core] + WEIGHT_LOADS_PER_CORE + int(core == SCALE_LOAD_CORE),
                "nominal_requests_per_cu": MEMORY_WAVES_PER_CU * (store_split[core] + WEIGHT_LOADS_PER_CORE + int(core == SCALE_LOAD_CORE)),
                "stage_span_cycles": _summary(stage_spans),
                "vmem_issue_stall_cycles": _summary([record["stall"] for record in vmem_rows]),
                "wait_vmcnt_stall_cycles": _summary(waits),
                "credit_wait_stall_cycles": _summary(credit_waits),
                "memory_barrier_stall_cycles": _summary(memory_barriers),
                "compute_barrier_stall_cycles": _summary(compute_barriers),
                "vmem_issue_stall_sum": sum(value["vmem_issue_stall_cycles"] for value in pair_core),
                "wait_vmcnt_stall_sum": sum(value["wait_vmcnt_stall_cycles"] for value in pair_core),
                "peer_compute_barrier_stall_sum": sum(value["peer_compute_barrier_stall_cycles"] for value in pair_core),
                "vmem_peer_barrier_overlap_sum": sum(value["vmem_peer_barrier_overlap_cycles"] for value in pair_core),
                "wait_peer_barrier_overlap_sum": sum(value["wait_peer_barrier_overlap_cycles"] for value in pair_core),
                "request_ordinal_stall": [
                    {
                        "ordinal": ordinal + 1,
                        "stall_cycles": _summary(values),
                    }
                    for ordinal, values in sorted(ordinal_stalls.items())
                ],
                "active_same_stage_loss_rate": same_stage / (same_stage + opposite_stage) if same_stage + opposite_stage else 1.0,
            }
        )
    total_same = sum(pair["active_same_stage_overlap_cycles"] for pair in pairs)
    total_opposite = sum(pair["active_opposite_stage_overlap_cycles"] for pair in pairs)
    return {
        "ui_directory": str(ui_directory),
        "analyzed_rounds": analyzed_rounds,
        "wave_count": len(waves),
        "valid_pairs": len(pairs),
        "placement_failures": failures,
        "active_same_stage_loss_rate": total_same / (total_same + total_opposite) if total_same + total_opposite else 1.0,
        "cores": cores,
        "pairs": pairs,
    }


def _analyze_root(args):
    store_split = tuple(int(value) for value in args.store_split)
    ui_directories = sorted(args.att_root.glob("ui_output_agent_*"))
    if not ui_directories:
        raise RuntimeError(f"no UI directories under {args.att_root}")
    capture_text = args.capture_log.read_text(encoding="utf-8", errors="replace")
    incomplete = sorted(set(INCOMPLETE_TRACE_PATTERN.findall(capture_text)))
    dispatches = [
        _analyze_dispatch(
            directory,
            args.rounds,
            store_split,
            args.schedule,
            args.ignore_edge_rounds,
        )
        for directory in ui_directories
    ]
    failures = []
    if incomplete:
        failures.append(f"incomplete trace markers: {incomplete}")
    if len(dispatches) != args.expected_dispatches:
        failures.append(f"dispatches={len(dispatches)}, expected {args.expected_dispatches}")
    for index, dispatch in enumerate(dispatches):
        if dispatch["valid_pairs"] != args.expected_pairs_per_dispatch or dispatch["placement_failures"]:
            failures.append(f"dispatch {index} placement/pair mismatch")

    core_summary = []
    analyzed_rounds = args.rounds - 2 * args.ignore_edge_rounds
    for core in range(K_CORES):
        values = [dispatch["cores"][core] for dispatch in dispatches]
        def med(field, stat="median"):
            return statistics.median(value[field][stat] for value in values if value[field] is not None)
        valid_pairs_per_dispatch = statistics.median(
            dispatch["valid_pairs"] for dispatch in dispatches
        )
        wave_rounds = valid_pairs_per_dispatch * 2 * analyzed_rounds
        pair_rounds = valid_pairs_per_dispatch * analyzed_rounds
        issue_sum = statistics.median(
            value["vmem_issue_stall_sum"] for value in values
        )
        wait_sum = statistics.median(
            value["wait_vmcnt_stall_sum"] for value in values
        )
        barrier_sum = statistics.median(
            value["peer_compute_barrier_stall_sum"] for value in values
        )
        issue_overlap = statistics.median(
            value["vmem_peer_barrier_overlap_sum"] for value in values
        )
        wait_overlap = statistics.median(
            value["wait_peer_barrier_overlap_sum"] for value in values
        )
        core_summary.append(
            {
                "core": core,
                "nominal_requests_per_cu": values[0]["nominal_requests_per_cu"],
                "stage_span_median_cycles": med("stage_span_cycles"),
                "vmem_issue_stall_mean_cycles": med("vmem_issue_stall_cycles", "mean"),
                "vmem_issue_stall_p95_cycles": med("vmem_issue_stall_cycles", "p95"),
                "wait_vmcnt_stall_median_cycles": med("wait_vmcnt_stall_cycles"),
                "credit_wait_stall_median_cycles": med("credit_wait_stall_cycles") if values[0]["credit_wait_stall_cycles"] else None,
                "memory_barrier_stall_median_cycles": med("memory_barrier_stall_cycles"),
                "compute_barrier_stall_median_cycles": med("compute_barrier_stall_cycles"),
                "vmem_issue_stall_sum": issue_sum,
                "wait_vmcnt_stall_sum": wait_sum,
                "peer_compute_barrier_stall_sum": barrier_sum,
                "vmem_peer_barrier_overlap_sum": issue_overlap,
                "wait_peer_barrier_overlap_sum": wait_overlap,
                "owner_stall_cycles_per_wave_round": {
                    "vmem_issue": issue_sum / wave_rounds,
                    "wait_vmcnt": wait_sum / wave_rounds,
                    "compute_barrier": barrier_sum / wave_rounds,
                },
                "physical_joint_cycles_per_pair_round": {
                    "vmem_issue_and_peer_barrier": issue_overlap / pair_rounds,
                    "wait_vmcnt_and_peer_barrier": wait_overlap / pair_rounds,
                    "memory_blocker_and_peer_barrier": (
                        issue_overlap + wait_overlap
                    )
                    / pair_rounds,
                },
            }
        )
    total_issue = sum(value["vmem_issue_stall_sum"] for value in core_summary)
    total_wait = sum(value["wait_vmcnt_stall_sum"] for value in core_summary)
    total_barrier = sum(value["peer_compute_barrier_stall_sum"] for value in core_summary)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "att_root": str(args.att_root),
        "config": {
            "rounds": args.rounds,
            "ignore_edge_rounds": args.ignore_edge_rounds,
            "analyzed_rounds": analyzed_rounds,
            "analysis_window": (
                "steady" if args.ignore_edge_rounds else "lifecycle"
            ),
            "store_split": args.store_split,
            "schedule": args.schedule,
            "weight_loads_per_wave_per_core": WEIGHT_LOADS_PER_CORE,
            "scale_load_core": SCALE_LOAD_CORE,
            "mfma_per_core": MFMA_PER_CORE,
            "memory_waves_per_cu": MEMORY_WAVES_PER_CU,
            "credit_knee_per_cu": CREDIT_KNEE_PER_CU,
            "successful_issue_formula": "first_attempt + stall",
        },
        "capture_log": {
            "path": str(args.capture_log),
            "sha256": hashlib.sha256(capture_text.encode()).hexdigest(),
            "incomplete_markers": incomplete,
        },
        "formal_valid": not failures,
        "validation_failures": failures,
        "dispatch_count": len(dispatches),
        "valid_pairs": sum(dispatch["valid_pairs"] for dispatch in dispatches),
        "active_same_stage_loss_rate": statistics.median(dispatch["active_same_stage_loss_rate"] for dispatch in dispatches),
        "core_summary": core_summary,
        "totals": {
            "vmem_issue_stall_sum": total_issue,
            "wait_vmcnt_stall_sum": total_wait,
            "peer_compute_barrier_stall_sum": total_barrier,
            "issue_stall_share": total_issue / (total_issue + total_wait) if total_issue + total_wait else 0.0,
        },
        "dispatches": dispatches,
    }
    print(
        f"ATT root={args.att_root} split={args.store_split} schedule={args.schedule} "
        f"pairs={payload['valid_pairs']} issue={total_issue:.0f} wait={total_wait:.0f} "
        f"barrier={total_barrier:.0f} same-stage={payload['active_same_stage_loss_rate']:.4%} "
        f"valid={payload['formal_valid']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {args.json}")
    if failures:
        raise RuntimeError(f"invalid result: {failures}")


def _discover_busy_path(properties):
    bdf = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    return Path("/sys/bus/pci/devices") / bdf / "gpu_busy_percent", bdf


def _run(args):
    store_split = tuple(int(value) for value in args.store_split)
    if sum(store_split) != 8:
        raise RuntimeError("store split must sum to 8")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not properties.gcnArchName.startswith("gfx94"):
        raise RuntimeError(f"gfx94x required, got {properties.gcnArchName}")
    busy_path, bdf = _discover_busy_path(properties)
    busy = int(busy_path.read_text().strip())
    if busy > 5 and not args.allow_busy:
        raise RuntimeError(f"GPU {bdf} busy={busy}%")
    blocks = properties.multi_processor_count
    wave_count = blocks * WAVES_PER_BLOCK
    data_bytes = args.data_mib * 1024 * 1024
    output_bytes = 8 * 1024 * 1024
    data = torch.arange(data_bytes // 4, dtype=torch.int32, device="cuda")
    output = torch.zeros(output_bytes // 4, dtype=torch.int32, device="cuda")
    metadata = torch.zeros((wave_count, METADATA_DWORDS), dtype=torch.int32, device="cuda")
    dispatches = []
    for dispatch in range(args.dispatches):
        metadata.zero_()
        strict_8wave_n256_prefetch(
            [blocks], [THREADS_PER_BLOCK], args.rounds,
            store_split[0], store_split[1], store_split[2], args.schedule,
            data_bytes, output_bytes, wave_count,
            data.data_ptr(), output.data_ptr(), metadata.data_ptr(),
        )
        torch.cuda.synchronize()
        placement = _validate_runtime_placement(metadata.cpu())
        placement["dispatch"] = dispatch
        dispatches.append(placement)
        print(f"dispatch={dispatch} pairs={placement['valid_pairs']} valid={placement['valid']}")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {"name": properties.name, "arch": properties.gcnArchName, "bdf": bdf},
        "config": {"rounds": args.rounds, "store_split": args.store_split, "schedule": args.schedule},
        "all_placements_valid": all(value["valid"] for value in dispatches),
        "dispatches": dispatches,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["all_placements_valid"]:
        raise RuntimeError("placement failure")


def _self_test():
    assert tuple(int(value) for value in "224") == (2, 2, 4)
    assert tuple(int(value) for value in "332") == (3, 3, 2)
    assert [4 * (8 + value + int(core == 2)) for core, value in enumerate((2, 2, 4))] == [40, 40, 52]
    assert [4 * (8 + value + int(core == 2)) for core, value in enumerate((3, 3, 2))] == [44, 44, 44]
    assert _overlap_cycles([(0, 10)], [(10, 20)]) == 0
    print("self-test passed: request ledgers and interval overlap")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--rounds", type=int, default=128)
    run_parser.add_argument("--store-split", choices=("224", "332"), required=True)
    run_parser.add_argument("--schedule", choices=SCHEDULES, required=True)
    run_parser.add_argument("--data-mib", type=int, default=512)
    run_parser.add_argument("--dispatches", type=int, default=2)
    run_parser.add_argument("--allow-busy", action="store_true")
    run_parser.add_argument("--json", type=Path)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--att-root", type=Path, required=True)
    analyze_parser.add_argument("--rounds", type=int, required=True)
    analyze_parser.add_argument("--ignore-edge-rounds", type=int, default=0)
    analyze_parser.add_argument("--store-split", choices=("224", "332"), required=True)
    analyze_parser.add_argument("--schedule", choices=SCHEDULES, required=True)
    analyze_parser.add_argument("--expected-dispatches", type=int, default=2)
    analyze_parser.add_argument("--expected-pairs-per-dispatch", type=int, default=16)
    analyze_parser.add_argument("--capture-log", type=Path, required=True)
    analyze_parser.add_argument("--json", type=Path)
    subparsers.add_parser("self-test")
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "run":
        _run(args)
    elif args.command == "analyze":
        if (
            args.ignore_edge_rounds < 0
            or 2 * args.ignore_edge_rounds >= args.rounds
        ):
            raise RuntimeError(
                "ignore-edge-rounds must leave at least one analyzed round"
            )
        _analyze_root(args)
    else:
        _self_test()


if __name__ == "__main__":
    main()
