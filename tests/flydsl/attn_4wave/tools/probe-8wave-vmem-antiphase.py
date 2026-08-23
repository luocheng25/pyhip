#!/usr/bin/env python3
"""Measure VMEM latency and FIFO pressure with strict 8-wave anti-phase.

One 64 KiB, 8-wave workgroup resides on each CU. Conditional entry and drain
barriers offset the two waves on every physical SIMD by one stage:

* stage 0 issues non-temporal HBM loads, waits at ``vmcnt(0)``, then consumes;
* stage 1 contains only BF16 MFMA instructions.

The hot loop has no clock reads or global observer stores. Run it under
rocprofv3 ATT and use ``analyze`` to separate VMEM-load issue stalls,
``wait-vmcnt`` completion stalls, and the peer compute wave's barrier stalls.
ATT timestamps use ``successful_issue = first_attempt + stall``.
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
MFMA_OPCODE = "v_mfma_f32_16x16x16_bf16"
PHASE_MARKERS = {2: "memory", 3: "compute"}
WAVES_PER_BLOCK = 8
THREADS_PER_BLOCK = 512
LDS_BYTES = 64 * 1024
BYTES_PER_WAVE_LOAD = 1024
METADATA_DWORDS = 8
MFMA_EXECUTION_CYCLES = 16
MEMORY_WAVES_PER_CU = 4
FIFO_ENTRIES_PER_CU = 48
INCOMPLETE_TRACE_PATTERN = re.compile(
    r"stitch incomplete|wave incomplete|trace was cutoff|"
    r"parser could not fully match",
    flags=re.IGNORECASE,
)
WAVE_PATH_PATTERN = re.compile(
    r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json$"
)


def _read_realtime(builder):
    value = builder.gpr(2, "su32", align=2)
    builder.s_memrealtime(value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    return value


def _make_mfma_registers(builder):
    operand_a = builder.gpr(2, "vu32", 0x3F803F80, align=2)
    operand_b = builder.gpr(2, "vu32", 0x3F803F80, align=2)
    accumulators = builder.gpr(4, 4, "vf32", align=4)
    accumulators[...] = 0.0
    return accumulators, operand_a, operand_b


def _emit_mfma_segment(builder, accumulators, operand_a, operand_b, count):
    for index in range(count):
        builder.v_mfma_f32_16x16x16_bf16(
            accumulators[index & 3], operand_a, operand_b, 0
        )


@jit(no_pass=["pass_dse", "pass_dce"])
def strict_8wave_vmem_antiphase(
    builder: JIT,
    rounds,
    loads_per_stage,
    mfmas_per_stage,
    data_bytes,
    total_waves,
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    metadata: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    builder.alloc_lds(LDS_BYTES, align=16)
    data_buffer = builder.Buffer(data, data_bytes)
    load_values = builder.gpr(loads_per_stage, 4, "vu32", align=4)
    load_sink = builder.gpr("vu32", 0)
    vector_offset = builder.gpr("vu32", builder.lane_id[0] * 16)
    accumulators, operand_a, operand_b = _make_mfma_registers(builder)

    hw_id = builder.gpr("su32")
    xcc_id = builder.gpr("su32")
    builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    kernel_start = _read_realtime(builder)

    global_wave = builder.blockIdx.x[0] * WAVES_PER_BLOCK + builder.warp_id[0]
    scalar_offset = builder.gpr(
        "su32", global_wave * BYTES_PER_WAVE_LOAD
    )
    stream_stride = total_waves * BYTES_PER_WAVE_LOAD
    buffer_mask = data_bytes - 1

    # High waves consume generation 0, then wait at generation 1 while low
    # waves execute the first memory stage.
    with builder.If(builder.warp_id[0] >= 4):
        builder.s_barrier()
    builder.s_barrier()

    round_index = builder.gpr("su32", 0)
    with builder.While(round_index[0] < rounds):
        builder.s_nop(2)
        for load_index in range(loads_per_stage):
            data_buffer.load_dwordx4(
                load_values[load_index],
                vector_offset,
                scalar_offset,
                non_temporal=True,
            )
            scalar_offset[0] += stream_stride
            scalar_offset[0] = scalar_offset[0] & buffer_mask
        builder.s_waitcnt(mod="vmcnt(0)")
        for load_index in range(loads_per_stage):
            builder.v_xor_b32(
                load_sink, load_sink, load_values[load_index, 0]
            )
        builder.s_barrier()

        builder.s_nop(3)
        _emit_mfma_segment(
            builder,
            accumulators,
            operand_a,
            operand_b,
            mfmas_per_stage,
        )
        builder.s_barrier()
        round_index[0] += 1

    with builder.If(builder.warp_id[0] < 4):
        builder.s_barrier()

    kernel_stop = _read_realtime(builder)
    metadata_offset = builder.gpr(
        "su32", global_wave * METADATA_DWORDS * 4
    )
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
    slot_pairs = Counter()
    for key, waves in sorted(groups.items()):
        if len(waves) != 2:
            failures.append(
                {"key": key, "reason": "not_two_waves", "count": len(waves)}
            )
            continue
        slots = tuple(sorted(wave["slot"] for wave in waves))
        blocks = {wave["block"] for wave in waves}
        overlap = min(wave["stop"] for wave in waves) - max(
            wave["start"] for wave in waves
        )
        if slots != (0, 1):
            failures.append(
                {"key": key, "reason": "unexpected_slots", "slots": slots}
            )
        elif len(blocks) != 1:
            failures.append(
                {"key": key, "reason": "cross_workgroup_pair"}
            )
        elif overlap <= 0:
            failures.append(
                {"key": key, "reason": "no_lifetime_overlap"}
            )
        else:
            slot_pairs[slots] += 1
    return {
        "wave_count": metadata.shape[0],
        "physical_simd_groups": len(groups),
        "valid_pairs": sum(slot_pairs.values()),
        "slot_pairs": {str(key): value for key, value in slot_pairs.items()},
        "placement_failures": failures,
        "valid": not failures and sum(slot_pairs.values()) == len(groups),
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
    match = re.fullmatch(
        r"s_nop\s+(0x[0-9a-f]+|\d+)", instruction.strip()
    )
    if match is None:
        return None
    return PHASE_MARKERS.get(int(match.group(1), 0))


def _static_map(ui_directory, loads_per_stage, mfmas_per_stage):
    code_path = ui_directory / "code.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))["code"]
    marker_ids = defaultdict(list)
    for instruction_id, row in enumerate(code):
        marker = _parse_marker(row[0])
        if marker is not None:
            marker_ids[marker].append(instruction_id)
    marker_counts = {name: len(ids) for name, ids in marker_ids.items()}
    if marker_counts != {"memory": 1, "compute": 1}:
        raise RuntimeError(f"invalid phase markers in {code_path}: {marker_counts}")
    memory_marker = marker_ids["memory"][0]
    compute_marker = marker_ids["compute"][0]
    if memory_marker >= compute_marker:
        raise RuntimeError(f"reversed phase markers in {code_path}")

    load_ids = [
        instruction_id
        for instruction_id in range(memory_marker + 1, compute_marker)
        if code[instruction_id][0].strip().split()[0]
        == "buffer_load_dwordx4"
    ]
    wait_ids = [
        instruction_id
        for instruction_id in range(memory_marker + 1, compute_marker)
        if code[instruction_id][0].strip().startswith("s_waitcnt")
        and "vmcnt" in code[instruction_id][0]
    ]
    memory_barriers = [
        instruction_id
        for instruction_id in range(memory_marker + 1, compute_marker)
        if code[instruction_id][0].strip().startswith("s_barrier")
    ]
    mfma_ids = [
        instruction_id
        for instruction_id, row in enumerate(code)
        if row[0].strip().split()[0] == MFMA_OPCODE
    ]
    compute_barriers = [
        instruction_id
        for instruction_id in range(max(mfma_ids) + 1, len(code))
        if code[instruction_id][0].strip().startswith("s_barrier")
    ]
    if len(load_ids) != loads_per_stage:
        raise RuntimeError(
            f"{code_path} has {len(load_ids)} loads, expected {loads_per_stage}"
        )
    if len(wait_ids) != 1 or len(memory_barriers) != 1:
        raise RuntimeError(
            f"{code_path} has waits={wait_ids}, memory barriers={memory_barriers}"
        )
    if len(mfma_ids) != mfmas_per_stage:
        raise RuntimeError(
            f"{code_path} has {len(mfma_ids)} MFMAs, expected {mfmas_per_stage}"
        )
    if not compute_barriers:
        raise RuntimeError(f"{code_path} has no compute-stage barrier")
    return {
        "code": code,
        "marker_ids": {
            "memory": memory_marker,
            "compute": compute_marker,
        },
        "load_ids": load_ids,
        "wait_id": wait_ids[0],
        "memory_barrier_id": memory_barriers[0],
        "mfma_ids": mfma_ids,
        "compute_barrier_id": compute_barriers[0],
    }


def _dynamic_record(row):
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
        raise RuntimeError(
            f"incomplete wave {path}: "
            f"{payload['num_stitched']}/{payload['num_insts']}"
        )
    wave = payload["wave"]
    match = WAVE_PATH_PATTERN.search(path.name)
    if match is None:
        raise RuntimeError(f"cannot parse wave filename {path}")
    se, filename_simd, filename_slot, wave_id = map(int, match.groups())
    if filename_simd != int(wave["simd"]) or filename_slot != int(wave["slot"]):
        raise RuntimeError(f"wave filename/metadata mismatch: {path}")

    records_by_id = defaultdict(list)
    marker_sequence = []
    target_ids = {
        *static["load_ids"],
        static["wait_id"],
        static["memory_barrier_id"],
        *static["mfma_ids"],
        static["compute_barrier_id"],
        *static["marker_ids"].values(),
    }
    marker_by_id = {
        instruction_id: phase
        for phase, instruction_id in static["marker_ids"].items()
    }
    for row in wave["instructions"]:
        instruction_id = int(row[4])
        if instruction_id not in target_ids:
            continue
        record = _dynamic_record(row)
        records_by_id[instruction_id].append(record)
        marker = marker_by_id.get(instruction_id)
        if marker is not None:
            marker_sequence.append(marker)

    expected_markers = [
        phase for _round in range(rounds) for phase in ("memory", "compute")
    ]
    if marker_sequence != expected_markers:
        raise RuntimeError(
            f"{path} marker mismatch: {marker_sequence[:16]}"
        )
    for instruction_id in static["load_ids"]:
        if len(records_by_id[instruction_id]) != rounds:
            raise RuntimeError(f"{path} load count mismatch at {instruction_id}")
    for instruction_id in (
        static["wait_id"],
        static["memory_barrier_id"],
        static["compute_barrier_id"],
        *static["mfma_ids"],
    ):
        if len(records_by_id[instruction_id]) != rounds:
            raise RuntimeError(
                f"{path} dynamic count {len(records_by_id[instruction_id])} "
                f"for {instruction_id}, expected {rounds}"
            )

    round_records = []
    for round_index in range(rounds):
        loads = [
            records_by_id[instruction_id][round_index]
            for instruction_id in static["load_ids"]
        ]
        wait = records_by_id[static["wait_id"]][round_index]
        memory_barrier = records_by_id[static["memory_barrier_id"]][round_index]
        mfmas = [
            records_by_id[instruction_id][round_index]
            for instruction_id in static["mfma_ids"]
        ]
        compute_barrier = records_by_id[static["compute_barrier_id"]][round_index]
        round_records.append(
            {
                "loads": loads,
                "wait": wait,
                "memory_barrier": memory_barrier,
                "mfmas": mfmas,
                "compute_barrier": compute_barrier,
                "memory_ready_cycles": wait["issue"] - loads[0]["issue"],
                "load_issue_span_cycles": loads[-1]["issue"] - loads[0]["issue"],
                "mfma_issue_span_cycles": (
                    mfmas[-1]["issue"]
                    - mfmas[0]["issue"]
                    + MFMA_EXECUTION_CYCLES
                ),
            }
        )
    return {
        "path": path.name,
        "key": (se, int(wave["cu"]), int(wave["simd"])),
        "se": se,
        "cu": int(wave["cu"]),
        "simd": int(wave["simd"]),
        "slot": int(wave["slot"]),
        "wave_id": wave_id,
        "rounds": round_records,
    }


def _intervals(wave, record_name):
    intervals = []
    for row in wave["rounds"]:
        record = row[record_name]
        if record["stall"]:
            intervals.append((record["attempt"], record["issue"]))
    return intervals


def _load_stall_intervals(wave):
    intervals = []
    for row in wave["rounds"]:
        for record in row["loads"]:
            if record["stall"]:
                intervals.append((record["attempt"], record["issue"]))
    return intervals


def _active_stage_intervals(wave):
    memory = []
    compute = []
    for row in wave["rounds"]:
        memory.append(
            (row["loads"][0]["attempt"], row["wait"]["issue"])
        )
        compute.append(
            (
                row["mfmas"][0]["issue"],
                row["mfmas"][-1]["issue"] + MFMA_EXECUTION_CYCLES,
            )
        )
    return memory, compute


def _overlap_cycles(left, right):
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left) and right_index < len(right):
        left_begin, left_end = left[left_index]
        right_begin, right_end = right[right_index]
        overlap += max(
            0, min(left_end, right_end) - max(left_begin, right_begin)
        )
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _analyze_pair(left, right):
    left_load = _load_stall_intervals(left)
    right_load = _load_stall_intervals(right)
    left_wait = _intervals(left, "wait")
    right_wait = _intervals(right, "wait")
    left_compute_barrier = _intervals(left, "compute_barrier")
    right_compute_barrier = _intervals(right, "compute_barrier")
    load_cycles = sum(end - begin for begin, end in left_load + right_load)
    wait_cycles = sum(end - begin for begin, end in left_wait + right_wait)
    compute_barrier_cycles = sum(
        end - begin
        for begin, end in left_compute_barrier + right_compute_barrier
    )
    load_overlap = _overlap_cycles(
        left_load, right_compute_barrier
    ) + _overlap_cycles(right_load, left_compute_barrier)
    wait_overlap = _overlap_cycles(
        left_wait, right_compute_barrier
    ) + _overlap_cycles(
        right_wait, left_compute_barrier
    )
    left_memory, left_compute = _active_stage_intervals(left)
    right_memory, right_compute = _active_stage_intervals(right)
    same_stage_overlap = _overlap_cycles(
        left_memory, right_memory
    ) + _overlap_cycles(left_compute, right_compute)
    opposite_stage_overlap = _overlap_cycles(
        left_memory, right_compute
    ) + _overlap_cycles(right_memory, left_compute)
    active_overlap = same_stage_overlap + opposite_stage_overlap
    return {
        "key": left["key"],
        "slots": sorted((left["slot"], right["slot"])),
        "wave_files": [left["path"], right["path"]],
        "load_issue_stall_cycles": load_cycles,
        "wait_stall_cycles": wait_cycles,
        "peer_compute_barrier_stall_cycles": compute_barrier_cycles,
        "load_peer_barrier_overlap_cycles": load_overlap,
        "wait_peer_barrier_overlap_cycles": wait_overlap,
        "combined_peer_barrier_overlap_cycles": load_overlap + wait_overlap,
        "load_issue_exposed_fraction": (
            load_overlap / load_cycles if load_cycles else 0.0
        ),
        "wait_exposed_fraction": (
            wait_overlap / wait_cycles if wait_cycles else 0.0
        ),
        "peer_barrier_explained_fraction": (
            (load_overlap + wait_overlap) / compute_barrier_cycles
            if compute_barrier_cycles
            else 0.0
        ),
        "active_same_stage_overlap_cycles": same_stage_overlap,
        "active_opposite_stage_overlap_cycles": opposite_stage_overlap,
        "active_same_stage_loss_rate": (
            same_stage_overlap / active_overlap if active_overlap else 1.0
        ),
    }


def _analyze_dispatch(
    ui_directory,
    rounds,
    loads_per_stage,
    mfmas_per_stage,
):
    static = _static_map(ui_directory, loads_per_stage, mfmas_per_stage)
    waves = []
    for path in sorted(ui_directory.glob("se*.json")):
        wave = _load_wave(path, static, rounds)
        waves.append(wave)

    groups = defaultdict(list)
    for wave in waves:
        groups[wave["key"]].append(wave)
    placement_failures = []
    pairs = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda wave: wave["slot"])
        if len(group) != 2 or [wave["slot"] for wave in group] != [0, 1]:
            placement_failures.append(
                {
                    "key": key,
                    "wave_count": len(group),
                    "slots": [wave["slot"] for wave in group],
                }
            )
            continue
        pairs.append(_analyze_pair(group[0], group[1]))

    load_stalls_by_ordinal = []
    for ordinal in range(loads_per_stage):
        values = [
            row["loads"][ordinal]["stall"]
            for wave in waves
            for row in wave["rounds"]
        ]
        load_stalls_by_ordinal.append(
            {
                "ordinal": ordinal + 1,
                "nominal_cu_requests": (ordinal + 1) * MEMORY_WAVES_PER_CU,
                "stall_cycles": _summary(values),
            }
        )

    load_stalls = [
        load["stall"]
        for wave in waves
        for row in wave["rounds"]
        for load in row["loads"]
    ]
    wait_stalls = [
        row["wait"]["stall"] for wave in waves for row in wave["rounds"]
    ]
    memory_barrier_stalls = [
        row["memory_barrier"]["stall"]
        for wave in waves
        for row in wave["rounds"]
    ]
    compute_barrier_stalls = [
        row["compute_barrier"]["stall"]
        for wave in waves
        for row in wave["rounds"]
    ]
    ready_cycles = [
        row["memory_ready_cycles"]
        for wave in waves
        for row in wave["rounds"]
    ]
    issue_spans = [
        row["load_issue_span_cycles"]
        for wave in waves
        for row in wave["rounds"]
    ]
    mfma_spans = [
        row["mfma_issue_span_cycles"]
        for wave in waves
        for row in wave["rounds"]
    ]
    pair_load_overlap = sum(
        pair["load_peer_barrier_overlap_cycles"] for pair in pairs
    )
    pair_wait_overlap = sum(
        pair["wait_peer_barrier_overlap_cycles"] for pair in pairs
    )
    pair_load = sum(pair["load_issue_stall_cycles"] for pair in pairs)
    pair_wait = sum(pair["wait_stall_cycles"] for pair in pairs)
    pair_compute_barrier = sum(
        pair["peer_compute_barrier_stall_cycles"] for pair in pairs
    )
    active_same_stage = sum(
        pair["active_same_stage_overlap_cycles"] for pair in pairs
    )
    active_opposite_stage = sum(
        pair["active_opposite_stage_overlap_cycles"] for pair in pairs
    )
    active_overlap = active_same_stage + active_opposite_stage

    fifo_boundary = FIFO_ENTRIES_PER_CU // MEMORY_WAVES_PER_CU
    pre_fifo_stalls = [
        value
        for ordinal in range(min(loads_per_stage, fifo_boundary))
        for value in [
            row["loads"][ordinal]["stall"]
            for wave in waves
            for row in wave["rounds"]
        ]
    ]
    post_fifo_stalls = [
        value
        for ordinal in range(fifo_boundary, loads_per_stage)
        for value in [
            row["loads"][ordinal]["stall"]
            for wave in waves
            for row in wave["rounds"]
        ]
    ]
    load_stall_sum = sum(load_stalls)
    wait_stall_sum = sum(wait_stalls)
    return {
        "ui_directory": str(ui_directory),
        "wave_count": len(waves),
        "physical_simd_groups": len(groups),
        "valid_pairs": len(pairs),
        "placement_failures": placement_failures,
        "static": {
            "load_instruction_ids": static["load_ids"],
            "wait_instruction_id": static["wait_id"],
            "memory_barrier_id": static["memory_barrier_id"],
            "mfma_instruction_ids": static["mfma_ids"],
            "compute_barrier_id": static["compute_barrier_id"],
        },
        "stall_cycles": {
            "vmem_load_issue": _summary(load_stalls),
            "wait_vmcnt": _summary(wait_stalls),
            "memory_stage_barrier": _summary(memory_barrier_stalls),
            "compute_stage_barrier": _summary(compute_barrier_stalls),
        },
        "stage_cycles": {
            "memory_ready": _summary(ready_cycles),
            "load_issue_span": _summary(issue_spans),
            "mfma_issue_span": _summary(mfma_spans),
        },
        "load_stall_by_ordinal": load_stalls_by_ordinal,
        "fifo": {
            "entries_per_cu_hypothesis": FIFO_ENTRIES_PER_CU,
            "active_memory_waves_per_cu": MEMORY_WAVES_PER_CU,
            "nominal_requests_per_cu": loads_per_stage * MEMORY_WAVES_PER_CU,
            "boundary_load_ordinal_per_wave": fifo_boundary,
            "pre_boundary_load_stall": _summary(pre_fifo_stalls),
            "post_boundary_load_stall": _summary(post_fifo_stalls),
            "load_issue_stall_share": (
                load_stall_sum / (load_stall_sum + wait_stall_sum)
                if load_stall_sum + wait_stall_sum
                else 0.0
            ),
        },
        "strict_antiphase_exposure": {
            "load_issue_stall_cycles": pair_load,
            "wait_stall_cycles": pair_wait,
            "compute_barrier_stall_cycles": pair_compute_barrier,
            "load_peer_barrier_overlap_cycles": pair_load_overlap,
            "wait_peer_barrier_overlap_cycles": pair_wait_overlap,
            "combined_peer_barrier_overlap_cycles": (
                pair_load_overlap + pair_wait_overlap
            ),
            "load_issue_exposed_fraction": (
                pair_load_overlap / pair_load if pair_load else 0.0
            ),
            "wait_exposed_fraction": (
                pair_wait_overlap / pair_wait if pair_wait else 0.0
            ),
            "compute_barrier_explained_fraction": (
                (pair_load_overlap + pair_wait_overlap) / pair_compute_barrier
                if pair_compute_barrier
                else 0.0
            ),
            "active_same_stage_overlap_cycles": active_same_stage,
            "active_opposite_stage_overlap_cycles": active_opposite_stage,
            "active_same_stage_loss_rate": (
                active_same_stage / active_overlap if active_overlap else 1.0
            ),
        },
        "pairs": pairs,
    }


def _combine_summaries(dispatches, path):
    values = []
    for dispatch in dispatches:
        value = dispatch
        for key in path:
            value = value[key]
        if value is not None:
            values.append(value)
    return values


def _analyze_root(args):
    ui_directories = sorted(args.att_root.glob("ui_output_agent_*"))
    if not ui_directories:
        raise RuntimeError(f"no decoded ATT directories under {args.att_root}")
    capture_text = args.capture_log.read_text(
        encoding="utf-8", errors="replace"
    )
    incomplete_markers = sorted(
        set(INCOMPLETE_TRACE_PATTERN.findall(capture_text))
    )
    dispatches = [
        _analyze_dispatch(
            directory,
            args.rounds,
            args.loads_per_stage,
            args.mfmas_per_stage,
        )
        for directory in ui_directories
    ]
    validation_failures = []
    if incomplete_markers:
        validation_failures.append(
            f"capture log reports incomplete ATT: {incomplete_markers}"
        )
    if len(dispatches) != args.expected_dispatches:
        validation_failures.append(
            f"captured {len(dispatches)} dispatches, expected "
            f"{args.expected_dispatches}"
        )
    for index, dispatch in enumerate(dispatches):
        if dispatch["valid_pairs"] != args.expected_pairs_per_dispatch:
            validation_failures.append(
                f"dispatch {index} has {dispatch['valid_pairs']} pairs, "
                f"expected {args.expected_pairs_per_dispatch}"
            )
        if dispatch["placement_failures"]:
            validation_failures.append(
                f"dispatch {index} placement failures"
            )

    load_ordinal = []
    for ordinal in range(args.loads_per_stage):
        summaries = [
            dispatch["load_stall_by_ordinal"][ordinal]["stall_cycles"]
            for dispatch in dispatches
        ]
        # Dispatch summaries retain sums/counts but not raw values. Pool by using
        # weighted means and report the range of medians/p95 across dispatches.
        load_ordinal.append(
            {
                "ordinal": ordinal + 1,
                "nominal_cu_requests": (ordinal + 1) * MEMORY_WAVES_PER_CU,
                "median_range": [
                    min(summary["median"] for summary in summaries),
                    max(summary["median"] for summary in summaries),
                ],
                "p95_range": [
                    min(summary["p95"] for summary in summaries),
                    max(summary["p95"] for summary in summaries),
                ],
                "mean": sum(
                    summary["mean"] * summary["count"] for summary in summaries
                )
                / sum(summary["count"] for summary in summaries),
                "sum": sum(summary["sum"] for summary in summaries),
            }
        )

    def dispatch_metric(*path):
        values = _combine_summaries(dispatches, path)
        return {
            "dispatch_values": values,
            "median": statistics.median(values),
        }

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "att_root": str(args.att_root),
        "config": {
            "rounds": args.rounds,
            "loads_per_stage": args.loads_per_stage,
            "mfmas_per_stage": args.mfmas_per_stage,
            "nominal_vmem_requests_per_cu": (
                args.loads_per_stage * MEMORY_WAVES_PER_CU
            ),
            "memory_waves_per_cu": MEMORY_WAVES_PER_CU,
            "fifo_entries_per_cu_hypothesis": FIFO_ENTRIES_PER_CU,
            "successful_issue_formula": "first_attempt + stall",
            "expected_dispatches": args.expected_dispatches,
            "expected_pairs_per_dispatch": args.expected_pairs_per_dispatch,
        },
        "capture_log": {
            "path": str(args.capture_log),
            "sha256": hashlib.sha256(capture_text.encode()).hexdigest(),
            "incomplete_markers": incomplete_markers,
        },
        "formal_valid": not validation_failures,
        "validation_failures": validation_failures,
        "dispatch_count": len(dispatches),
        "valid_pairs": sum(
            dispatch["valid_pairs"] for dispatch in dispatches
        ),
        "load_stall_by_ordinal": load_ordinal,
        "summary": {
            "memory_ready_median_cycles": dispatch_metric(
                "stage_cycles", "memory_ready", "median"
            ),
            "load_issue_span_median_cycles": dispatch_metric(
                "stage_cycles", "load_issue_span", "median"
            ),
            "mfma_issue_span_median_cycles": dispatch_metric(
                "stage_cycles", "mfma_issue_span", "median"
            ),
            "load_issue_stall_mean_cycles": dispatch_metric(
                "stall_cycles", "vmem_load_issue", "mean"
            ),
            "wait_vmcnt_stall_median_cycles": dispatch_metric(
                "stall_cycles", "wait_vmcnt", "median"
            ),
            "memory_barrier_stall_median_cycles": dispatch_metric(
                "stall_cycles", "memory_stage_barrier", "median"
            ),
            "compute_barrier_stall_median_cycles": dispatch_metric(
                "stall_cycles", "compute_stage_barrier", "median"
            ),
            "load_issue_stall_share": dispatch_metric(
                "fifo", "load_issue_stall_share"
            ),
            "wait_exposed_fraction": dispatch_metric(
                "strict_antiphase_exposure", "wait_exposed_fraction"
            ),
            "load_issue_exposed_fraction": dispatch_metric(
                "strict_antiphase_exposure", "load_issue_exposed_fraction"
            ),
            "compute_barrier_explained_fraction": dispatch_metric(
                "strict_antiphase_exposure",
                "compute_barrier_explained_fraction",
            ),
            "active_same_stage_loss_rate": dispatch_metric(
                "strict_antiphase_exposure", "active_same_stage_loss_rate"
            ),
        },
        "dispatches": dispatches,
    }
    print(
        f"ATT root={args.att_root} loads={args.loads_per_stage} "
        f"MFMA={args.mfmas_per_stage} pairs={payload['valid_pairs']} "
        f"ready={payload['summary']['memory_ready_median_cycles']['median']:.1f} "
        f"wait={payload['summary']['wait_vmcnt_stall_median_cycles']['median']:.1f} "
        f"load-stall={payload['summary']['load_issue_stall_mean_cycles']['median']:.2f} "
        f"same-stage={payload['summary']['active_same_stage_loss_rate']['median']:.4%} "
        f"valid={payload['formal_valid']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    if not payload["formal_valid"]:
        raise RuntimeError(f"invalid ATT result: {validation_failures}")


def _discover_busy_path(properties):
    bdf = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    return Path("/sys/bus/pci/devices") / bdf / "gpu_busy_percent", bdf


def _run_workload(args):
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not properties.gcnArchName.startswith("gfx94"):
        raise RuntimeError(f"gfx94x is required, got {properties.gcnArchName}")
    busy_path, bdf = _discover_busy_path(properties)
    busy = int(busy_path.read_text(encoding="utf-8").strip())
    if busy > 5 and not args.allow_busy:
        raise RuntimeError(f"target GPU {bdf} is busy={busy}%")

    blocks = properties.multi_processor_count
    wave_count = blocks * WAVES_PER_BLOCK
    data_bytes = args.data_mib * 1024 * 1024
    if data_bytes < wave_count * BYTES_PER_WAVE_LOAD:
        raise RuntimeError("data buffer is too small for one request per wave")
    if data_bytes & (data_bytes - 1):
        raise RuntimeError("data-mib must produce a power-of-two byte size")
    data = torch.arange(
        data_bytes // 4, dtype=torch.int32, device="cuda"
    )
    metadata = torch.zeros(
        (wave_count, METADATA_DWORDS), dtype=torch.int32, device="cuda"
    )
    dispatches = []
    for dispatch in range(args.dispatches):
        metadata.zero_()
        strict_8wave_vmem_antiphase(
            [blocks],
            [THREADS_PER_BLOCK],
            args.rounds,
            args.loads_per_stage,
            args.mfmas_per_stage,
            data_bytes,
            wave_count,
            data.data_ptr(),
            metadata.data_ptr(),
        )
        torch.cuda.synchronize()
        placement = _validate_runtime_placement(metadata.cpu())
        placement["dispatch"] = dispatch
        dispatches.append(placement)
        print(
            f"dispatch={dispatch} pairs={placement['valid_pairs']} "
            f"valid={placement['valid']} "
            f"failures={len(placement['placement_failures'])}"
        )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "torch_device": args.device,
            "name": properties.name,
            "arch": properties.gcnArchName,
            "bdf": bdf,
            "compute_units": properties.multi_processor_count,
            "initial_busy_percent": busy,
        },
        "config": {
            "rounds": args.rounds,
            "loads_per_stage": args.loads_per_stage,
            "mfmas_per_stage": args.mfmas_per_stage,
            "data_bytes": data_bytes,
            "blocks": blocks,
            "threads": THREADS_PER_BLOCK,
            "waves_per_block": WAVES_PER_BLOCK,
            "nominal_vmem_requests_per_cu": (
                args.loads_per_stage * MEMORY_WAVES_PER_CU
            ),
        },
        "all_placements_valid": all(row["valid"] for row in dispatches),
        "dispatches": dispatches,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    if not payload["all_placements_valid"]:
        raise RuntimeError("one or more dispatches failed placement validation")


def _self_test():
    assert _overlap_cycles([(0, 10), (20, 30)], [(5, 25)]) == 10
    assert _overlap_cycles([], [(0, 10)]) == 0
    assert _overlap_cycles([(0, 10)], [(10, 20)]) == 0
    assert FIFO_ENTRIES_PER_CU // MEMORY_WAVES_PER_CU == 12
    summary = _summary([0, 4, 8, 12])
    assert summary["median"] == 6.0
    assert summary["sum"] == 24
    print("self-test passed: interval overlap and stall summaries")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="launch the strict 8-wave workload"
    )
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--rounds", type=int, default=128)
    run_parser.add_argument("--loads-per-stage", type=int, required=True)
    run_parser.add_argument("--mfmas-per-stage", type=int, required=True)
    run_parser.add_argument("--data-mib", type=int, default=512)
    run_parser.add_argument("--dispatches", type=int, default=2)
    run_parser.add_argument("--allow-busy", action="store_true")
    run_parser.add_argument("--json", type=Path)

    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze decoded rocprofv3 ATT output"
    )
    analyze_parser.add_argument("--att-root", type=Path, required=True)
    analyze_parser.add_argument("--rounds", type=int, required=True)
    analyze_parser.add_argument("--loads-per-stage", type=int, required=True)
    analyze_parser.add_argument("--mfmas-per-stage", type=int, required=True)
    analyze_parser.add_argument("--expected-dispatches", type=int, default=2)
    analyze_parser.add_argument(
        "--expected-pairs-per-dispatch", type=int, default=16
    )
    analyze_parser.add_argument("--capture-log", type=Path, required=True)
    analyze_parser.add_argument("--json", type=Path)

    subparsers.add_parser("self-test", help="test offline helpers")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        if (
            args.rounds <= 0
            or args.loads_per_stage <= 0
            or args.mfmas_per_stage <= 0
            or args.data_mib <= 0
            or args.dispatches <= 0
        ):
            parser.error("all counts and sizes must be positive")
        _run_workload(args)
    elif args.command == "analyze":
        if (
            args.rounds <= 0
            or args.loads_per_stage <= 0
            or args.mfmas_per_stage <= 0
            or args.expected_dispatches <= 0
            or args.expected_pairs_per_dispatch <= 0
        ):
            parser.error("all analysis counts must be positive")
        _analyze_root(args)
    else:
        _self_test()


if __name__ == "__main__":
    main()
