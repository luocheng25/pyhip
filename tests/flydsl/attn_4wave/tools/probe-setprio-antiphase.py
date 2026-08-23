#!/usr/bin/env python3
"""Probe whether s_setprio can replace barriers for exact SIMD anti-phase.

The workload has no clock reads or global observer stores in its hot loop. Use
``run`` directly to validate physical placement, or run it under rocprofv3
advanced thread trace and use ``analyze`` on the decoded output. Stage 0 is a
tunable VMEM + LDS + VALU pipeline; stage 1 is a fixed MFMA segment. ATT
distinguishes them by static PCs and uses
``successful_issue = first_attempt + stall``.

A phase-run event is stable when its nearest peer run is mutually nearest and
has the opposite phase. Loss is therefore
``unstable phase runs / observed phase runs``. Startup and drain runs outside
the common center span are censored. The physical anti-phase metric separately
reports same-stage overlap across the two waves' active stage issue spans.
"""

import argparse
import bisect
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
MODES = ("barrier8", "priority4", "equal4")
MFMA_OPCODE = "v_mfma_f32_16x16x16_bf16"
PHASE_MARKERS = {2: "A", 3: "B"}
STAGE0_OPCODES = {
    "buffer_load_dwordx4": 1,
    "ds_write_b128": 1,
    "ds_read_b128": 1,
    "v_xor_b32": 4,
}
BYTES_PER_WAVE_LOAD = 1024
METADATA_DWORDS = 8
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


def _emit_stage0_tile(
    builder,
    data_buffer,
    load_value,
    ds_value,
    valu_value,
    vector_offset,
    scalar_offset,
    lds_address,
    stream_stride,
    buffer_mask,
):
    data_buffer.load_dwordx4(
        load_value,
        vector_offset,
        scalar_offset,
        non_temporal=True,
    )
    builder.s_waitcnt(mod="vmcnt(0)")
    builder.ds_write_b128(lds_address, load_value)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    builder.ds_read_b128(ds_value, lds_address)
    builder.s_waitcnt(mod="lgkmcnt(0)")
    for index in range(4):
        builder.v_xor_b32(valu_value[index], valu_value[index], ds_value[index])
    scalar_offset[0] += stream_stride
    scalar_offset[0] = scalar_offset[0] & buffer_mask


@jit(no_pass=["pass_dse", "pass_dce"])
def antiphase_att(
    builder: JIT,
    mode,
    rounds,
    stage0_repeats,
    stage1_mfmas,
    data_bytes,
    total_waves,
    data: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    metadata: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    """Run an uninstrumented two-phase loop for hardware ATT capture."""
    assert mode in MODES
    waves_per_block = 8 if mode == "barrier8" else 4
    lds_bytes = 64 * 1024 if mode == "barrier8" else 32 * 1024
    lds_base = builder.alloc_lds(lds_bytes, align=16)

    accumulators, operand_a, operand_b = _make_mfma_registers(builder)
    data_buffer = builder.Buffer(data, data_bytes)
    load_value = builder.gpr(4, "vu32", align=4)
    ds_value = builder.gpr(4, "vu32", align=4)
    valu_value = builder.gpr(4, "vu32", 0, align=4)
    vector_offset = builder.gpr("vu32", builder.lane_id[0] * 16)
    lds_address = builder.gpr(
        "vu32", lds_base + builder.threadIdx.x[0] * 16
    )
    hw_id = builder.gpr("su32")
    xcc_id = builder.gpr("su32")
    builder.s_getreg_b32(hw_id, mod="hwreg(HW_REG_HW_ID, 0, 20)")
    builder.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")
    kernel_start = _read_realtime(builder)
    global_wave = builder.blockIdx.x[0] * waves_per_block + builder.warp_id[0]
    scalar_offset = builder.gpr(
        "su32", global_wave * BYTES_PER_WAVE_LOAD
    )
    stream_stride = total_waves * BYTES_PER_WAVE_LOAD
    buffer_mask = data_bytes - 1

    if mode == "barrier8":
        # The first generation pairs low-wave common arrivals with high-wave
        # extra arrivals. High waves then wait while low waves execute stage 0.
        with builder.If(builder.warp_id[0] >= 4):
            builder.s_barrier()
        builder.s_barrier()
    else:
        # This aligns each 4-wave workgroup internally, but there is deliberately
        # no synchronization between the two resident workgroups.
        builder.s_barrier()

    round_index = builder.gpr("su32", 0)
    with builder.While(round_index[0] < rounds):
        if mode == "priority4":
            builder.s_setprio(0)

        builder.s_nop(2)
        for _ in range(stage0_repeats):
            _emit_stage0_tile(
                builder,
                data_buffer,
                load_value,
                ds_value,
                valu_value,
                vector_offset,
                scalar_offset,
                lds_address,
                stream_stride,
                buffer_mask,
            )
        if mode == "barrier8":
            builder.s_barrier()

        if mode == "priority4":
            builder.s_setprio(1)

        builder.s_nop(3)
        _emit_mfma_segment(
            builder, accumulators, operand_a, operand_b, stage1_mfmas
        )
        if mode == "barrier8":
            builder.s_barrier()
        round_index[0] += 1

    if mode == "barrier8":
        with builder.If(builder.warp_id[0] < 4):
            builder.s_barrier()
    else:
        builder.s_setprio(0)
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
    builder.v_readfirstlane_b32(sink, accumulators[0, 0])
    sink_component = builder.gpr("su32")
    builder.v_readfirstlane_b32(sink_component, valu_value[0])
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
        "hw_id": hw_id,
        "xcc": values[3],
        "slot": hw_id & 0xF,
        "simd": (hw_id >> 4) & 0x3,
        "cu": (hw_id >> 8) & 0xF,
        "se": (hw_id >> 13) & 0x7,
        "start": values[4] | (values[5] << 32),
        "stop": values[6] | (values[7] << 32),
    }


def _validate_runtime_placement(mode, metadata):
    groups = defaultdict(list)
    for row in metadata.tolist():
        wave = _decode_metadata(row)
        key = (wave["xcc"], wave["se"], wave["cu"], wave["simd"])
        groups[key].append(wave)

    failures = []
    slot_pairs = Counter()
    block_relationships = Counter()
    for key, waves in sorted(groups.items()):
        slots = tuple(sorted(wave["slot"] for wave in waves))
        if len(waves) != 2:
            failures.append(
                {
                    "key": key,
                    "reason": "not_two_waves",
                    "wave_count": len(waves),
                }
            )
            continue
        overlap = min(wave["stop"] for wave in waves) - max(
            wave["start"] for wave in waves
        )
        if overlap <= 0:
            failures.append({"key": key, "reason": "no_lifetime_overlap"})
            continue
        slot_pairs[slots] += 1
        blocks = {wave["block"] for wave in waves}
        relationship = (
            "same_workgroup" if len(blocks) == 1 else "cross_workgroup"
        )
        block_relationships[relationship] += 1

    expected_relationship = (
        "same_workgroup" if mode == "barrier8" else "cross_workgroup"
    )
    valid_pairs = sum(slot_pairs.values())
    return {
        "wave_count": metadata.shape[0],
        "physical_simd_groups": len(groups),
        "valid_pairs": valid_pairs,
        "placement_failures": failures,
        "slot_pairs": {str(key): value for key, value in slot_pairs.items()},
        "block_relationships": dict(block_relationships),
        "expected_relationship": expected_relationship,
        "valid": (
            not failures
            and slot_pairs == Counter({(0, 1): len(groups)})
            and block_relationships
            == Counter({expected_relationship: len(groups)})
        ),
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
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _independent_duty_loss_bound(stage0_cycles, stage1_cycles):
    return abs(stage0_cycles - stage1_cycles) / (
        stage0_cycles + stage1_cycles
    )


def _find_first_stable_window(events, window):
    if len(events) < window:
        return None
    losses = [not event[1] for event in events]
    rolling = sum(losses[:window])
    if rolling == 0:
        return events[0][0]
    for index in range(window, len(events)):
        rolling += losses[index] - losses[index - window]
        if rolling == 0:
            return events[index - window + 1][0]
    return None


def _find_permanent_stable_suffix(events, minimum_events):
    if len(events) < minimum_events:
        return None
    suffix_losses = 0
    result = None
    for index in range(len(events) - 1, -1, -1):
        if not events[index][1]:
            suffix_losses += 1
        remaining = len(events) - index
        if remaining >= minimum_events and suffix_losses == 0:
            result = events[index][0]
    return result


def _longest_stable_run(events):
    best_count = 0
    best_start = None
    best_stop = None
    current_count = 0
    current_start = None
    for timestamp, stable, _phase, _side in events:
        if stable:
            if current_count == 0:
                current_start = timestamp
            current_count += 1
            if current_count > best_count:
                best_count = current_count
                best_start = current_start
                best_stop = timestamp
        else:
            current_count = 0
            current_start = None
    return best_count, best_start, best_stop


def _nearest_index(centers, timestamp):
    insertion = bisect.bisect_left(centers, timestamp)
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(centers)
    ]
    distances = [abs(centers[index] - timestamp) for index in candidates]
    minimum = min(distances)
    nearest = [
        index
        for index, distance in zip(candidates, distances)
        if distance == minimum
    ]
    return nearest[0], minimum, len(nearest) > 1


def _active_stage_overlap(left_runs, right_runs):
    left_index = 0
    right_index = 0
    same_stage_cycles = 0
    opposite_stage_cycles = 0
    while left_index < len(left_runs) and right_index < len(right_runs):
        left = left_runs[left_index]
        right = right_runs[right_index]
        overlap = max(
            0,
            min(left["stop"], right["stop"])
            - max(left["start"], right["start"]),
        )
        if overlap:
            if left["phase"] == right["phase"]:
                same_stage_cycles += overlap
            else:
                opposite_stage_cycles += overlap
        if left["stop"] <= right["stop"]:
            left_index += 1
        else:
            right_index += 1
    return same_stage_cycles, opposite_stage_cycles


def _analyze_run_pair(left, right, stable_window_events):
    left_runs = left["runs"]
    right_runs = right["runs"]
    left_centers = [
        (run["start"] + run["stop"]) // 2 for run in left_runs
    ]
    right_centers = [
        (run["start"] + run["stop"]) // 2 for run in right_runs
    ]
    common_start = max(left_centers[0], right_centers[0])
    common_stop = min(left_centers[-1], right_centers[-1])
    if common_start >= common_stop:
        raise ValueError("paired waves have no common phase-run center span")

    classified = []
    center_distances = []
    reasons = Counter()
    censored = 0
    for side, own, own_centers, peer, peer_centers in (
        ("left", left_runs, left_centers, right_runs, right_centers),
        ("right", right_runs, right_centers, left_runs, left_centers),
    ):
        for own_index, (run, center) in enumerate(zip(own, own_centers)):
            if center < common_start or center > common_stop:
                censored += 1
                continue
            peer_index, distance, ambiguous = _nearest_index(
                peer_centers, center
            )
            reverse_index, _reverse_distance, reverse_ambiguous = _nearest_index(
                own_centers, peer_centers[peer_index]
            )
            opposite = peer[peer_index]["phase"] != run["phase"]
            mutual = reverse_index == own_index
            stable = opposite and mutual and not ambiguous and not reverse_ambiguous
            if not stable:
                if not opposite:
                    reasons["same_phase_nearest"] += 1
                elif ambiguous or reverse_ambiguous:
                    reasons["ambiguous_nearest"] += 1
                else:
                    reasons["non_mutual_nearest"] += 1
            center_distances.append(distance)
            classified.append((center, stable, run["phase"], side))
    classified.sort(key=lambda event: (event[0], event[3]))

    unstable = sum(not event[1] for event in classified)
    first_window = _find_first_stable_window(classified, stable_window_events)
    permanent = _find_permanent_stable_suffix(
        classified, stable_window_events
    )
    longest_count, longest_start, longest_stop = _longest_stable_run(classified)
    same_stage_cycles, opposite_stage_cycles = _active_stage_overlap(
        left_runs, right_runs
    )
    active_overlap_cycles = same_stage_cycles + opposite_stage_cycles
    quartile_rates = []
    for quartile in range(4):
        begin = len(classified) * quartile // 4
        end = len(classified) * (quartile + 1) // 4
        section = classified[begin:end]
        quartile_rates.append(
            sum(not event[1] for event in section) / len(section)
            if section
            else None
        )
    return {
        "key": left["key"],
        "slots": sorted((left["slot"], right["slot"])),
        "wave_files": [left["path"], right["path"]],
        "left_stage_events": len(left["events"]),
        "right_stage_events": len(right["events"]),
        "left_phase_runs": len(left_runs),
        "right_phase_runs": len(right_runs),
        "common_start_cycle": common_start,
        "common_stop_cycle": common_stop,
        "common_observation_cycles": common_stop - common_start,
        "total_events": len(classified),
        "unstable_events": unstable,
        "censored_edge_events": censored,
        "loss_rate": unstable / len(classified) if classified else 1.0,
        "loss_reasons": dict(reasons),
        "active_same_stage_cycles": same_stage_cycles,
        "active_opposite_stage_cycles": opposite_stage_cycles,
        "active_overlap_cycles": active_overlap_cycles,
        "active_time_loss_rate": (
            same_stage_cycles / active_overlap_cycles
            if active_overlap_cycles
            else 1.0
        ),
        "phase_center_distance_cycles": _summary(center_distances),
        "quartile_loss_rates": quartile_rates,
        "first_zero_loss_window_cycles": (
            None if first_window is None else first_window - common_start
        ),
        "permanent_stable_suffix_cycles": (
            None if permanent is None else permanent - common_start
        ),
        "longest_stable_run_events": longest_count,
        "longest_stable_run_cycles": (
            0
            if longest_start is None or longest_stop is None
            else longest_stop - longest_start
        ),
    }


def _phase_runs(events):
    runs = []
    for timestamp, phase in events:
        if not runs or runs[-1]["phase"] != phase:
            runs.append(
                {
                    "phase": phase,
                    "count": 1,
                    "start": timestamp,
                    "stop": timestamp,
                }
            )
        else:
            runs[-1]["count"] += 1
            runs[-1]["stop"] = timestamp
    return runs


def _load_att_wave(path, phase_by_instruction, marker_by_instruction):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["num_insts"] != payload["num_stitched"]:
        raise RuntimeError(
            f"incomplete wave {path}: "
            f"{payload['num_stitched']}/{payload['num_insts']}"
        )
    wave = payload["wave"]
    events = []
    markers = []
    for row in wave["instructions"]:
        instruction_id = row[4]
        successful_issue = row[0] + row[2]
        phase = phase_by_instruction.get(instruction_id)
        if phase is not None:
            events.append((successful_issue, phase))
        marker = marker_by_instruction.get(instruction_id)
        if marker is not None:
            markers.append((successful_issue, marker))
    if not events:
        return None
    match = re.search(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json$", path.name)
    if match is None:
        raise RuntimeError(f"cannot parse ATT wave filename: {path}")
    se, filename_simd, filename_slot, wave_id = map(int, match.groups())
    if filename_simd != int(wave["simd"]) or filename_slot != int(wave["slot"]):
        raise RuntimeError(f"ATT filename/metadata mismatch: {path}")
    return {
        "path": path.name,
        "key": (se, int(wave["cu"]), int(wave["simd"])),
        "se": se,
        "cu": int(wave["cu"]),
        "simd": int(wave["simd"]),
        "slot": int(wave["slot"]),
        "wave_id": wave_id,
        "begin": int(wave["begin"]),
        "end": int(wave["end"]),
        "events": events,
        "markers": markers,
        "runs": _phase_runs(events),
    }


def _parse_marker(instruction):
    match = re.fullmatch(
        r"s_nop\s+(0x[0-9a-f]+|\d+)", instruction.strip()
    )
    if match is None:
        return None
    return PHASE_MARKERS.get(int(match.group(1), 0))


def _stage0_kind(opcode):
    for prefix in STAGE0_OPCODES:
        if opcode.startswith(prefix):
            return prefix
    return None


def _static_phase_map(ui_directory, stage0_repeats, stage1_mfmas):
    code_path = ui_directory / "code.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))["code"]
    marker_by_instruction = {
        instruction_id: marker
        for instruction_id, row in enumerate(code)
        if (marker := _parse_marker(row[0])) is not None
    }
    marker_ids = defaultdict(list)
    for instruction_id, marker in marker_by_instruction.items():
        marker_ids[marker].append(instruction_id)
    marker_counts = {name: len(ids) for name, ids in marker_ids.items()}
    if marker_counts != {"A": 1, "B": 1}:
        raise RuntimeError(
            f"{code_path} must contain one A and one B marker, got "
            f"{dict(marker_ids)}"
        )
    stage0_begin = marker_ids["A"][0]
    stage0_end = marker_ids["B"][0]
    if stage0_begin >= stage0_end:
        raise RuntimeError(f"{code_path} has reversed stage markers")

    stage0_ids = []
    stage0_kinds = Counter()
    for instruction_id in range(stage0_begin + 1, stage0_end):
        opcode = code[instruction_id][0].strip().split()[0]
        kind = _stage0_kind(opcode)
        if kind is not None:
            stage0_ids.append(instruction_id)
            stage0_kinds[kind] += 1
    expected_stage0_kinds = Counter(
        {
            opcode: count * stage0_repeats
            for opcode, count in STAGE0_OPCODES.items()
        }
    )
    if stage0_kinds != expected_stage0_kinds:
        raise RuntimeError(
            f"{code_path} stage0 static mix {dict(stage0_kinds)}, expected "
            f"{dict(expected_stage0_kinds)}"
        )

    mfma_ids = [
        instruction_id
        for instruction_id, row in enumerate(code)
        if row[0].strip().split()[0] == MFMA_OPCODE
    ]
    if len(mfma_ids) != stage1_mfmas:
        raise RuntimeError(
            f"{code_path} has {len(mfma_ids)} target MFMAs, expected "
            f"{stage1_mfmas}"
        )
    if any(instruction_id <= stage0_end for instruction_id in mfma_ids):
        raise RuntimeError(f"{code_path} has MFMA outside stage1")
    phase_by_instruction = {
        **{instruction_id: "A" for instruction_id in stage0_ids},
        **{instruction_id: "B" for instruction_id in mfma_ids},
    }
    static = []
    for instruction_id in stage0_ids + mfma_ids:
        row = code[instruction_id]
        static.append(
            {
                "instruction_id": instruction_id,
                "phase": phase_by_instruction[instruction_id],
                "instruction": row[0],
                "pc": row[5],
            }
        )
    return {
        "phase_by_instruction": phase_by_instruction,
        "marker_by_instruction": marker_by_instruction,
        "static": static,
        "stage0_events_per_run": len(stage0_ids),
        "stage1_events_per_run": len(mfma_ids),
        "stage0_static_mix": dict(stage0_kinds),
    }


def _analyze_att_dispatch(
    ui_directory,
    rounds,
    stage0_repeats,
    stage1_mfmas,
    stable_window_events,
):
    static_info = _static_phase_map(
        ui_directory, stage0_repeats, stage1_mfmas
    )
    waves = []
    for path in sorted(ui_directory.glob("se*.json")):
        wave = _load_att_wave(
            path,
            static_info["phase_by_instruction"],
            static_info["marker_by_instruction"],
        )
        if wave is not None:
            waves.append(wave)

    stage0_events_per_run = static_info["stage0_events_per_run"]
    stage1_events_per_run = static_info["stage1_events_per_run"]
    expected_events = rounds * (
        stage0_events_per_run + stage1_events_per_run
    )
    expected_runs = 2 * rounds
    expected_run_phases = [
        phase for _round in range(rounds) for phase in ("A", "B")
    ]
    expected_run_counts = [
        count
        for _round in range(rounds)
        for count in (stage0_events_per_run, stage1_events_per_run)
    ]
    trace_failures = []
    stage0_durations = []
    stage1_durations = []
    for wave in waves:
        run_phases = [run["phase"] for run in wave["runs"]]
        run_counts = [run["count"] for run in wave["runs"]]
        if len(wave["events"]) != expected_events:
            trace_failures.append(
                {
                    "path": wave["path"],
                    "reason": "unexpected_stage_event_count",
                    "actual": len(wave["events"]),
                    "expected": expected_events,
                }
            )
        if (
            len(run_counts) != expected_runs
            or run_phases != expected_run_phases
            or run_counts != expected_run_counts
        ):
            trace_failures.append(
                {
                    "path": wave["path"],
                    "reason": "unexpected_phase_runs",
                    "actual_run_count": len(run_counts),
                    "expected_run_count": expected_runs,
                    "actual_phase_prefix": run_phases[:16],
                    "actual_count_prefix": run_counts[:16],
                }
            )
        marker_phases = [phase for _timestamp, phase in wave["markers"]]
        if marker_phases != expected_run_phases:
            trace_failures.append(
                {
                    "path": wave["path"],
                    "reason": "unexpected_phase_markers",
                    "actual_count": len(marker_phases),
                    "expected_count": expected_runs,
                    "actual_prefix": marker_phases[:16],
                }
            )
        for run in wave["runs"]:
            duration = run["stop"] - run["start"]
            if run["phase"] == "A":
                stage0_durations.append(duration)
            else:
                stage1_durations.append(duration)

    groups = defaultdict(list)
    for wave in waves:
        groups[wave["key"]].append(wave)
    placement_failures = []
    pairs = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda wave: (wave["slot"], wave["begin"]))
        if len(group) != 2:
            placement_failures.append(
                {
                    "key": key,
                    "reason": "not_two_stage_waves",
                    "count": len(group),
                }
            )
            continue
        if [wave["slot"] for wave in group] != [0, 1]:
            placement_failures.append(
                {
                    "key": key,
                    "reason": "unexpected_slots",
                    "slots": [wave["slot"] for wave in group],
                }
            )
            continue
        pairs.append(
            _analyze_run_pair(
                group[0], group[1], stable_window_events
            )
        )

    total_events = sum(pair["total_events"] for pair in pairs)
    unstable_events = sum(pair["unstable_events"] for pair in pairs)
    same_stage_cycles = sum(
        pair["active_same_stage_cycles"] for pair in pairs
    )
    opposite_stage_cycles = sum(
        pair["active_opposite_stage_cycles"] for pair in pairs
    )
    active_overlap_cycles = same_stage_cycles + opposite_stage_cycles
    first_windows = [
        pair["first_zero_loss_window_cycles"]
        for pair in pairs
        if pair["first_zero_loss_window_cycles"] is not None
    ]
    permanent = [
        pair["permanent_stable_suffix_cycles"]
        for pair in pairs
        if pair["permanent_stable_suffix_cycles"] is not None
    ]
    longest_runs = [pair["longest_stable_run_events"] for pair in pairs]
    observations = [pair["common_observation_cycles"] for pair in pairs]
    stage0_median = statistics.median(stage0_durations)
    stage1_median = statistics.median(stage1_durations)
    active_time_loss_rate = (
        same_stage_cycles / active_overlap_cycles
        if active_overlap_cycles
        else 1.0
    )
    duty_loss_bound = _independent_duty_loss_bound(
        stage0_median, stage1_median
    )
    return {
        "ui_directory": str(ui_directory),
        "wave_count": len(waves),
        "physical_simd_groups": len(groups),
        "valid_pairs": len(pairs),
        "placement_failures": placement_failures,
        "trace_failures": trace_failures,
        "static_stages": static_info["static"],
        "stage0_static_mix": static_info["stage0_static_mix"],
        "stage0_duration_cycles": _summary(stage0_durations),
        "stage1_duration_cycles": _summary(stage1_durations),
        "stage0_to_stage1_ratio": stage0_median / stage1_median,
        "independent_duty_loss_bound": duty_loss_bound,
        "total_events": total_events,
        "unstable_events": unstable_events,
        "loss_rate": unstable_events / total_events if total_events else 1.0,
        "active_same_stage_cycles": same_stage_cycles,
        "active_opposite_stage_cycles": opposite_stage_cycles,
        "active_overlap_cycles": active_overlap_cycles,
        "active_time_loss_rate": active_time_loss_rate,
        "active_loss_excess_over_duty_bound": (
            active_time_loss_rate - duty_loss_bound
        ),
        "first_zero_loss_window_cycles": _summary(first_windows),
        "first_zero_loss_window_pair_fraction": (
            len(first_windows) / len(pairs) if pairs else 0.0
        ),
        "permanent_stable_suffix_cycles": _summary(permanent),
        "permanent_stable_pair_fraction": (
            len(permanent) / len(pairs) if pairs else 0.0
        ),
        "longest_stable_run_events": _summary(longest_runs),
        "common_observation_cycles": _summary(observations),
        "_stage0_durations": stage0_durations,
        "_stage1_durations": stage1_durations,
        "pairs": pairs,
    }


def _cycles_to_us(summary, clock_mhz):
    if summary is None or clock_mhz is None:
        return None
    return {
        key: (value / clock_mhz if key != "count" else value)
        for key, value in summary.items()
    }


def _analyze_att_root(args):
    ui_directories = sorted(args.att_root.glob("ui_output_agent_*"))
    if not ui_directories:
        raise RuntimeError(f"no decoded ATT directories under {args.att_root}")
    capture_log = None
    incomplete_markers = []
    if args.capture_log is not None:
        capture_text = args.capture_log.read_text(
            encoding="utf-8", errors="replace"
        )
        incomplete_markers = INCOMPLETE_TRACE_PATTERN.findall(capture_text)
        capture_log = {
            "path": str(args.capture_log),
            "sha256": hashlib.sha256(capture_text.encode()).hexdigest(),
            "incomplete_markers": sorted(set(incomplete_markers)),
        }
    dispatches = [
        _analyze_att_dispatch(
            directory,
            args.rounds,
            args.stage0_repeats,
            args.stage1_mfmas,
            args.stable_window_events,
        )
        for directory in ui_directories
    ]
    stage0_durations = [
        duration
        for dispatch in dispatches
        for duration in dispatch.pop("_stage0_durations")
    ]
    stage1_durations = [
        duration
        for dispatch in dispatches
        for duration in dispatch.pop("_stage1_durations")
    ]
    pairs = [pair for row in dispatches for pair in row["pairs"]]
    total_events = sum(pair["total_events"] for pair in pairs)
    unstable_events = sum(pair["unstable_events"] for pair in pairs)
    same_stage_cycles = sum(
        pair["active_same_stage_cycles"] for pair in pairs
    )
    opposite_stage_cycles = sum(
        pair["active_opposite_stage_cycles"] for pair in pairs
    )
    active_overlap_cycles = same_stage_cycles + opposite_stage_cycles
    first_windows = [
        pair["first_zero_loss_window_cycles"]
        for pair in pairs
        if pair["first_zero_loss_window_cycles"] is not None
    ]
    permanent = [
        pair["permanent_stable_suffix_cycles"]
        for pair in pairs
        if pair["permanent_stable_suffix_cycles"] is not None
    ]
    longest = [pair["longest_stable_run_events"] for pair in pairs]
    observations = [pair["common_observation_cycles"] for pair in pairs]
    pair_loss_rates = [pair["loss_rate"] for pair in pairs]
    placement_failure_count = sum(
        len(row["placement_failures"]) for row in dispatches
    )
    trace_failure_count = sum(len(row["trace_failures"]) for row in dispatches)
    validation_failures = []
    if incomplete_markers:
        validation_failures.append("capture log reports incomplete ATT")
    if (
        args.expected_dispatches is not None
        and len(dispatches) != args.expected_dispatches
    ):
        validation_failures.append(
            f"captured {len(dispatches)} dispatches, expected "
            f"{args.expected_dispatches}"
        )
    if args.expected_pairs_per_dispatch is not None:
        for index, dispatch in enumerate(dispatches):
            if dispatch["valid_pairs"] != args.expected_pairs_per_dispatch:
                validation_failures.append(
                    f"dispatch {index} has {dispatch['valid_pairs']} pairs, "
                    f"expected {args.expected_pairs_per_dispatch}"
                )
    stage0_median = statistics.median(stage0_durations)
    stage1_median = statistics.median(stage1_durations)
    active_time_loss_rate = (
        same_stage_cycles / active_overlap_cycles
        if active_overlap_cycles
        else 1.0
    )
    duty_loss_bound = _independent_duty_loss_bound(
        stage0_median, stage1_median
    )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "att_root": str(args.att_root),
        "config": {
            "rounds": args.rounds,
            "stage0_repeats": args.stage0_repeats,
            "stage1_mfmas": args.stage1_mfmas,
            "stable_window_events": args.stable_window_events,
            "clock_mhz": args.clock_mhz,
            "expected_dispatches": args.expected_dispatches,
            "expected_pairs_per_dispatch": args.expected_pairs_per_dispatch,
            "successful_issue_formula": "first_attempt + stall",
            "loss_definition": (
                "phase run whose nearest peer run is not mutually nearest and opposite"
            ),
            "active_time_loss_definition": (
                "same-stage overlap cycles / all overlap cycles across active "
                "stage issue spans"
            ),
            "independent_duty_loss_bound_definition": (
                "abs(stage0-stage1)/(stage0+stage1) for continuously "
                "alternating independent waves without compensating waits"
            ),
        },
        "dispatch_count": len(dispatches),
        "valid_pairs": len(pairs),
        "placement_failure_count": placement_failure_count,
        "trace_failure_count": trace_failure_count,
        "capture_log": capture_log,
        "validation_failures": validation_failures,
        "total_events": total_events,
        "unstable_events": unstable_events,
        "loss_rate": unstable_events / total_events if total_events else 1.0,
        "pair_loss_rate": _summary(pair_loss_rates),
        "active_same_stage_cycles": same_stage_cycles,
        "active_opposite_stage_cycles": opposite_stage_cycles,
        "active_overlap_cycles": active_overlap_cycles,
        "active_time_loss_rate": active_time_loss_rate,
        "stage0_duration_cycles": _summary(stage0_durations),
        "stage1_duration_cycles": _summary(stage1_durations),
        "stage0_to_stage1_ratio": stage0_median / stage1_median,
        "independent_duty_loss_bound": duty_loss_bound,
        "active_loss_excess_over_duty_bound": (
            active_time_loss_rate - duty_loss_bound
        ),
        "first_zero_loss_window_cycles": _summary(first_windows),
        "first_zero_loss_window_us": _cycles_to_us(
            _summary(first_windows), args.clock_mhz
        ),
        "first_zero_loss_window_pair_fraction": (
            len(first_windows) / len(pairs) if pairs else 0.0
        ),
        "permanent_stable_suffix_cycles": _summary(permanent),
        "permanent_stable_suffix_us": _cycles_to_us(
            _summary(permanent), args.clock_mhz
        ),
        "permanent_stable_pair_fraction": (
            len(permanent) / len(pairs) if pairs else 0.0
        ),
        "longest_stable_run_events": _summary(longest),
        "common_observation_cycles": _summary(observations),
        "common_observation_us": _cycles_to_us(
            _summary(observations), args.clock_mhz
        ),
        "formal_valid": (
            bool(pairs)
            and placement_failure_count == 0
            and trace_failure_count == 0
            and not validation_failures
        ),
        "dispatches": dispatches,
    }
    print(
        f"ATT root={args.att_root} dispatches={payload['dispatch_count']} "
        f"pairs={payload['valid_pairs']} loss={payload['loss_rate']:.6%} "
        f"active_loss={payload['active_time_loss_rate']:.6%} "
        f"stage_ratio={payload['stage0_to_stage1_ratio']:.4f} "
        f"placement_failures={placement_failure_count} "
        f"trace_failures={trace_failure_count}"
    )
    print(
        "  first zero-loss window pair fraction: "
        f"{payload['first_zero_loss_window_pair_fraction']:.3%}"
    )
    print(
        "  permanent stable pair fraction: "
        f"{payload['permanent_stable_pair_fraction']:.3%}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    if not payload["formal_valid"]:
        raise RuntimeError("ATT result failed placement or trace validation")


def _discover_busy_path(properties):
    bdf = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    path = Path("/sys/bus/pci/devices") / bdf / "gpu_busy_percent"
    return path, bdf


def _run_workload(args):
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not properties.gcnArchName.startswith("gfx94"):
        raise RuntimeError(f"gfx94x is required, got {properties.gcnArchName}")
    busy_path, bdf = _discover_busy_path(properties)
    busy = int(busy_path.read_text(encoding="utf-8").strip())
    if busy > 5 and not args.allow_busy:
        raise RuntimeError(f"target GPU {bdf} is busy={busy}%")

    waves_per_block = 8 if args.mode == "barrier8" else 4
    blocks_per_cu = 1 if args.mode == "barrier8" else 2
    blocks = properties.multi_processor_count * blocks_per_cu
    threads = waves_per_block * 64
    wave_count = blocks * waves_per_block
    data_bytes = args.data_mib * 1024 * 1024
    if data_bytes < wave_count * BYTES_PER_WAVE_LOAD:
        raise RuntimeError("data buffer is too small for one load per wave")
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
        antiphase_att(
            [blocks],
            [threads],
            args.mode,
            args.rounds,
            args.stage0_repeats,
            args.stage1_mfmas,
            data_bytes,
            wave_count,
            data.data_ptr(),
            metadata.data_ptr(),
        )
        torch.cuda.synchronize()
        placement = _validate_runtime_placement(args.mode, metadata.cpu())
        placement["dispatch"] = dispatch
        dispatches.append(placement)
        print(
            f"dispatch={dispatch} mode={args.mode} "
            f"pairs={placement['valid_pairs']} valid={placement['valid']} "
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
            "mode": args.mode,
            "rounds": args.rounds,
            "stage0_repeats": args.stage0_repeats,
            "stage1_mfmas": args.stage1_mfmas,
            "data_bytes": data_bytes,
            "dispatches": args.dispatches,
            "blocks": blocks,
            "threads": threads,
            "waves_per_block": waves_per_block,
            "priority_pattern": {
                "all_waves": {"stage0": 0, "stage1": 1},
            },
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
    def wave(name, slot, events):
        return {
            "key": (0, 0, 0),
            "slot": slot,
            "path": name,
            "events": events,
            "runs": _phase_runs(events),
        }

    left = []
    right_opposite = []
    right_same = []
    for round_index in range(8):
        base = round_index * 256
        for offset in range(16):
            left.append((base + offset * 4, "A"))
            right_opposite.append((base + offset * 4, "B"))
            right_same.append((base + offset * 4, "A"))
        for offset in range(16):
            left.append((base + 128 + offset * 4, "B"))
            right_opposite.append((base + 128 + offset * 4, "A"))
            right_same.append((base + 128 + offset * 4, "B"))
    exact = _analyze_run_pair(
        wave("left", 0, left),
        wave("right", 1, right_opposite),
        8,
    )
    failed = _analyze_run_pair(
        wave("left", 0, left),
        wave("right", 1, right_same),
        8,
    )
    assert exact["loss_rate"] == 0.0
    assert exact["active_time_loss_rate"] == 0.0
    assert exact["permanent_stable_suffix_cycles"] is not None
    assert failed["loss_rate"] == 1.0
    assert failed["active_time_loss_rate"] == 1.0
    assert failed["permanent_stable_suffix_cycles"] is None
    print("self-test passed: opposite=0% loss, same-phase=100% loss")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="launch the hot loop and validate physical placement"
    )
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--mode", choices=MODES, required=True)
    run_parser.add_argument("--rounds", type=int, default=512)
    run_parser.add_argument("--stage0-repeats", type=int, required=True)
    run_parser.add_argument("--stage1-mfmas", type=int, default=64)
    run_parser.add_argument("--data-mib", type=int, default=512)
    run_parser.add_argument("--dispatches", type=int, default=1)
    run_parser.add_argument("--allow-busy", action="store_true")
    run_parser.add_argument("--json", type=Path)

    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze decoded rocprofv3 ATT output"
    )
    analyze_parser.add_argument("--att-root", type=Path, required=True)
    analyze_parser.add_argument("--rounds", type=int, required=True)
    analyze_parser.add_argument("--stage0-repeats", type=int, required=True)
    analyze_parser.add_argument("--stage1-mfmas", type=int, default=64)
    analyze_parser.add_argument("--stable-window-events", type=int, default=64)
    analyze_parser.add_argument("--clock-mhz", type=float)
    analyze_parser.add_argument("--expected-dispatches", type=int)
    analyze_parser.add_argument("--expected-pairs-per-dispatch", type=int)
    analyze_parser.add_argument("--capture-log", type=Path)
    analyze_parser.add_argument("--json", type=Path)

    subparsers.add_parser("self-test", help="test the offline loss classifier")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        if (
            args.rounds <= 0
            or args.stage0_repeats <= 0
            or args.stage1_mfmas <= 0
            or args.data_mib <= 0
            or args.dispatches <= 0
        ):
            parser.error("all run counts and sizes must be positive")
        _run_workload(args)
    elif args.command == "analyze":
        if (
            args.rounds <= 0
            or args.stage0_repeats <= 0
            or args.stage1_mfmas <= 0
            or args.stable_window_events <= 0
        ):
            parser.error(
                "rounds, stage sizes, and stable-window-events must be positive"
            )
        _analyze_att_root(args)
    else:
        _self_test()


if __name__ == "__main__":
    main()
