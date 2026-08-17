#!/usr/bin/env python3
"""Project gfx9 ATT into single-wave and physical-SIMD stall exposure ledgers."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Set
from pathlib import Path

import numpy as np

import analyze_down_mfma_slots as slots  # pyright: ignore[reportMissingImports]

PRIMARY_REASONS = (
    "vmem_issue_stall",
    "vmem_wait_stall",
    "ds_issue_stall",
    "ds_wait_stall",
    "structural_tail",
)
ALL_REASONS = PRIMARY_REASONS + (
    "mixed_wait_stall",
    "mfma_issue_unavailable",
    "other_dependency_stall",
    "normal_issue_exposure",
    "scheduler_ready",
    "other_exposure",
)
DEFAULT_SOFTWARE_TARGETS = (
    "vmem_issue_stall",
    "ds_issue_stall",
    "ds_wait_stall",
    "structural_tail",
)
REGISTER_OPERAND_RE = re.compile(r"([va])(?:\[(\d+)(?::(\d+))?\]|(\d+))")
WAIT_COUNTER_RE = {
    "vmcnt": re.compile(r"vmcnt\((\d+)\)"),
    "lgkmcnt": re.compile(r"lgkmcnt\((\d+)\)"),
}


def exclusive_reason(phase: int, blocker: str) -> str:
    """Classify one non-MFMA wave tick into a mutually exclusive reason.

    Explicit memory blockers take priority over the tail phase. This keeps VMEM
    and DS wait visible inside the epilogue; ``structural_tail`` is the tail
    remainder after those explicit blockers are removed from the accounting.
    """

    if blocker.startswith(("stall:VMEM-load", "stall:VMEM-store")):
        return "vmem_issue_stall"
    if blocker.startswith("stall:wait-vmcnt"):
        return "vmem_wait_stall"
    if blocker.startswith(("stall:DS-read", "stall:DS-write", "stall:LDS/")):
        return "ds_issue_stall"
    if blocker.startswith("stall:wait-lgkmcnt"):
        return "ds_wait_stall"
    if blocker.startswith("stall:wait-mixed"):
        return "mixed_wait_stall"
    if phase == slots.PHASE_TAIL:
        return "structural_tail"
    if blocker.startswith("stall:MFMA"):
        return "mfma_issue_unavailable"
    if blocker.startswith("stall:"):
        return "other_dependency_stall"
    if blocker.startswith("issue:"):
        return "normal_issue_exposure"
    if blocker == "scheduler/ready":
        return "scheduler_ready"
    return "other_exposure"


def distribution(values: list[float]) -> dict:
    if not values:
        return {
            "samples": 0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def counter_summary(
    counter: Mapping[str, float], denominator: float, idle: float
) -> dict:
    return {
        reason: {
            "cycles": float(counter.get(reason, 0.0)),
            "total_fraction": (
                float(counter.get(reason, 0.0) / denominator) if denominator else 0.0
            ),
            "idle_fraction": float(counter.get(reason, 0.0) / idle) if idle else 0.0,
        }
        for reason in ALL_REASONS
    }


def build_blocker_names(code: list[slots.CodeInfo]) -> list[str]:
    names = ["inactive", "scheduler/ready"]
    categories = sorted({info.category for info in code})
    names.extend(f"stall:{category}" for category in categories)
    names.extend(f"issue:{category}" for category in categories)
    return names


def instruction_operands(assembly: str) -> list[str]:
    _, separator, operands = assembly.partition(" ")
    if not separator:
        return []
    return [operand.strip() for operand in operands.split(",")]


def operand_registers(operand: str) -> frozenset[tuple[str, int]]:
    match = REGISTER_OPERAND_RE.fullmatch(operand.strip())
    if match is None:
        return frozenset()
    bank = match.group(1)
    low = int(match.group(2) or match.group(4))
    high = int(match.group(3) or low)
    return frozenset((bank, index) for index in range(low, high + 1))


def counter_increment(category: str, counter: str) -> bool:
    if counter == "vmcnt":
        return category == "VMEM-load"
    return category.startswith(("DS-", "LDS/")) or category == "SMEM"


def find_covering_wait(
    records: list[dict], producer_index: int, consumer_index: int, counter: str
) -> dict | None:
    """Find a waitcnt that guarantees completion of one memory producer."""

    for wait_index in range(producer_index + 1, consumer_index):
        wait = records[wait_index]
        match = WAIT_COUNTER_RE[counter].search(wait["asm"])
        if match is None:
            continue
        threshold = int(match.group(1))
        younger_operations = sum(
            counter_increment(record["category"], counter)
            for record in records[producer_index + 1 : wait_index]
        )
        if younger_operations >= threshold:
            return {
                "assembly": wait["asm"],
                "threshold": threshold,
                "younger_operations": younger_operations,
            }
    return None


def analyze_mfma_readiness(
    groups: Mapping[tuple[int, int, int], list[dict]],
    first_n: int,
    last_n_exclusive: int,
) -> dict:
    first_ordinal = first_n * slots.MFMA_PER_N_BLOCK
    last_ordinal = last_n_exclusive * slots.MFMA_PER_N_BLOCK
    selected_mfmas = 0
    consecutive_pairs = 0
    back_to_back_raw_pairs = 0
    accumulator_literal_consumers = 0
    accumulator_register_consumers = 0
    accumulator_reuse_distance = []
    accumulator_reuse_issue_cycles = []
    same_wave_issue_intervals = []
    physical_issue_intervals = []
    source_edges = {
        "A/source0": Counter(),
        "B/source1": Counter(),
    }
    source_examples = {}

    for waves in groups.values():
        physical_events = []
        for wave in waves:
            records = wave["records"]
            mfmas = [record for record in records if record["category"] == "MFMA"]
            selected_wave_mfmas = [
                record
                for record in mfmas
                if first_ordinal <= int(record["mfma_ordinal"]) < last_ordinal
            ]
            physical_events.extend(record["issue"] for record in selected_wave_mfmas)
            same_wave_issue_intervals.extend(
                right["issue"] - left["issue"]
                for left, right in zip(selected_wave_mfmas, selected_wave_mfmas[1:])
            )
            last_writer: dict[tuple[str, int], int] = {}
            last_mfma_writer: dict[tuple[str, int], dict] = {}
            for record_index, record in enumerate(records):
                operands = instruction_operands(record["asm"])
                if record["category"] == "MFMA":
                    if len(operands) != 4:
                        raise RuntimeError(
                            f"cannot parse MFMA operands: {record['asm']}"
                        )
                    destination = operand_registers(operands[0])
                    source_operands = (
                        (
                            "A/source0",
                            operand_registers(operands[1]),
                            "VMEM-load",
                            "vmcnt",
                        ),
                        (
                            "B/source1",
                            operand_registers(operands[2]),
                            "DS-read",
                            "lgkmcnt",
                        ),
                    )
                    accumulator = operand_registers(operands[3])
                    ordinal = int(record["mfma_ordinal"])
                    selected = first_ordinal <= ordinal < last_ordinal
                    if selected:
                        selected_mfmas += 1
                        consecutive_pairs += 1
                        previous_operands = instruction_operands(
                            mfmas[ordinal - 1]["asm"]
                        )
                        previous_destination = operand_registers(previous_operands[0])
                        back_to_back_raw_pairs += int(
                            bool(previous_destination & accumulator)
                        )

                        if accumulator:
                            accumulator_register_consumers += 1
                            producer_records = {
                                int(
                                    last_mfma_writer[register]["mfma_ordinal"]
                                ): last_mfma_writer[register]
                                for register in accumulator
                                if register in last_mfma_writer
                            }
                            if len(producer_records) != 1:
                                raise RuntimeError(
                                    f"ambiguous MFMA accumulator producer: {record['asm']}"
                                )
                            producer = next(iter(producer_records.values()))
                            accumulator_reuse_distance.append(
                                ordinal - int(producer["mfma_ordinal"])
                            )
                            accumulator_reuse_issue_cycles.append(
                                record["issue"] - producer["issue"]
                            )
                        else:
                            accumulator_literal_consumers += 1

                        for (
                            source_name,
                            registers,
                            expected_category,
                            counter,
                        ) in source_operands:
                            producer_indices = {
                                last_writer[register]
                                for register in registers
                                if register in last_writer
                            }
                            if len(producer_indices) != 1:
                                raise RuntimeError(
                                    f"ambiguous {source_name} producer: {record['asm']}"
                                )
                            producer_index = producer_indices.pop()
                            producer = records[producer_index]
                            source_edges[source_name]["edges"] += 1
                            source_edges[source_name][
                                f"producer:{producer['category']}"
                            ] += 1
                            if producer["category"] != expected_category:
                                source_edges[source_name]["unexpected_producer"] += 1
                            wait = find_covering_wait(
                                records, producer_index, record_index, counter
                            )
                            if wait is not None:
                                source_edges[source_name]["wait_covered"] += 1
                                source_examples.setdefault(
                                    source_name,
                                    {
                                        "producer": producer["asm"],
                                        "wait": wait,
                                        "consumer": record["asm"],
                                    },
                                )
                    for register in destination:
                        last_mfma_writer[register] = record

                destination = (
                    operand_registers(operands[0]) if operands else frozenset()
                )
                for register in destination:
                    last_writer[register] = record_index
        physical_events.sort()
        physical_issue_intervals.extend(
            right - left for left, right in zip(physical_events, physical_events[1:])
        )

    source_summary = {}
    for source_name, counts in source_edges.items():
        edges = counts["edges"]
        source_summary[source_name] = {
            "edges": edges,
            "producer_categories": {
                key.removeprefix("producer:"): value
                for key, value in sorted(counts.items())
                if key.startswith("producer:")
            },
            "unexpected_producers": counts["unexpected_producer"],
            "wait_covered_edges": counts["wait_covered"],
            "wait_covered_fraction": counts["wait_covered"] / edges if edges else 0.0,
            "example": source_examples.get(source_name),
        }
    return {
        "selected_mfmas": selected_mfmas,
        "consecutive_pair_checks": consecutive_pairs,
        "back_to_back_accumulator_raw_pairs": back_to_back_raw_pairs,
        "back_to_back_accumulator_raw_fraction": (
            back_to_back_raw_pairs / consecutive_pairs if consecutive_pairs else 0.0
        ),
        "accumulator_literal_consumers": accumulator_literal_consumers,
        "accumulator_register_consumers": accumulator_register_consumers,
        "accumulator_reuse_mfma_distance": distribution(accumulator_reuse_distance),
        "accumulator_reuse_issue_cycles": distribution(accumulator_reuse_issue_cycles),
        "same_wave_successful_issue_interval_cycles": distribution(
            same_wave_issue_intervals
        ),
        "physical_successful_issue_interval_cycles": distribution(
            physical_issue_intervals
        ),
        "physical_issue_intervals_below_execution_cycles": sum(
            interval < slots.MFMA_EXEC_CYCLES for interval in physical_issue_intervals
        ),
        "physical_issue_intervals_equal_execution_cycles": sum(
            interval == slots.MFMA_EXEC_CYCLES for interval in physical_issue_intervals
        ),
        "source_readiness": source_summary,
        "all_memory_source_edges_wait_covered": all(
            values["wait_covered_edges"] == values["edges"]
            for values in source_summary.values()
        ),
    }


def analyze_mfma_issue_unavailability(
    group: dict, first_n: int, last_n_exclusive: int
) -> Counter:
    unavailable_id = ALL_REASONS.index("mfma_issue_unavailable")
    mfma_stall_id = group["blocker_names"].index("stall:MFMA")
    counts = Counter()
    for window in group["single_windows"]:
        if not first_n <= window["n_block"] < last_n_exclusive:
            continue
        slot = window["slot"]
        for tick in range(window["left"], window["right"]):
            peers = [
                peer
                for peer in range(group["mfma"].shape[0])
                if peer != slot and group["active"][peer, tick]
            ]
            own_execution = bool(group["mfma"][slot, tick])
            peer_execution = any(group["mfma"][peer, tick] for peer in peers)
            peer_issue = any(group["mfma_issue"][peer, tick] for peer in peers)

            if int(group["blocker"][slot, tick]) == mfma_stall_id:
                counts["raw_stall_ticks"] += 1
                overlap_name = f"raw_stall_own_{int(own_execution)}_peer_{int(peer_execution)}_ticks"
                counts[overlap_name] += 1

            if own_execution or int(group["reason"][slot, tick]) != unavailable_id:
                continue
            counts["wave_ticks"] += 1
            counts["peer_execution_ticks"] += int(peer_execution)
            counts["peer_same_tick_issue_ticks"] += int(peer_issue)
            counts["without_peer_execution_ticks"] += int(not peer_execution)
    return counts


def paint_group(waves: list[dict], blocker_names: list[str]) -> dict:
    waves = sorted(waves, key=lambda wave: (wave["begin"], wave["slot"]))
    origin = min(wave["begin"] for wave in waves)
    end = max(wave["end"] for wave in waves)
    ticks = slots.tick_ceil(end, origin)
    slot_count = max(wave["slot"] for wave in waves) + 1
    phase = np.zeros((slot_count, ticks), dtype=np.uint8)
    blocker = np.zeros((slot_count, ticks), dtype=np.uint16)
    active = np.zeros((slot_count, ticks), dtype=np.bool_)
    mfma = np.zeros((slot_count, ticks), dtype=np.bool_)
    mfma_issue = np.zeros((slot_count, ticks), dtype=np.bool_)
    reason = np.zeros((slot_count, ticks), dtype=np.uint8)
    n_index = np.full((slot_count, ticks), -1, dtype=np.int16)
    blocker_to_id = {name: index for index, name in enumerate(blocker_names)}
    reason_to_id = {name: index for index, name in enumerate(ALL_REASONS)}
    single_windows = []

    for wave in waves:
        slot = wave["slot"]
        slots.paint(active[slot], wave["begin"], wave["end"], origin, True)
        slots.paint(
            phase[slot], wave["begin"], wave["end"], origin, slots.PHASE_PROLOGUE
        )
        slots.paint(
            blocker[slot],
            wave["begin"],
            wave["end"],
            origin,
            blocker_to_id["scheduler/ready"],
        )
        mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
        for n_block in range(slots.N_BLOCKS):
            n_base = n_block * slots.MFMA_PER_N_BLOCK
            cores = []
            for core in range(slots.CORES_PER_N_BLOCK):
                records = mfmas[
                    n_base
                    + core * slots.MFMA_PER_CORE : n_base
                    + (core + 1) * slots.MFMA_PER_CORE
                ]
                core_begin = records[0]["issue"]
                core_end = records[-1]["issue"] + slots.MFMA_EXEC_CYCLES
                cores.append((core_begin, core_end))
                slots.paint(
                    phase[slot],
                    core_begin,
                    core_end,
                    origin,
                    slots.PHASE_CORE0 + 2 * core,
                )
            slots.paint(
                phase[slot], cores[0][1], cores[1][0], origin, slots.PHASE_BOUNDARY01
            )
            slots.paint(
                phase[slot], cores[1][1], cores[2][0], origin, slots.PHASE_BOUNDARY12
            )
            if n_block + 1 < slots.N_BLOCKS:
                next_begin = mfmas[(n_block + 1) * slots.MFMA_PER_N_BLOCK]["issue"]
                slots.paint(
                    phase[slot], cores[2][1], next_begin, origin, slots.PHASE_TAIL
                )
                slots.paint(
                    n_index[slot],
                    mfmas[n_base]["issue"],
                    next_begin,
                    origin,
                    n_block,
                )
                single_windows.append(
                    {
                        "slot": slot,
                        "n_block": n_block,
                        "left": slots.tick_floor(mfmas[n_base]["issue"], origin),
                        "right": slots.tick_floor(next_begin, origin),
                    }
                )
        slots.paint(
            phase[slot],
            mfmas[-1]["issue"] + slots.MFMA_EXEC_CYCLES,
            wave["end"],
            origin,
            slots.PHASE_DRAIN,
        )

        for record in wave["records"]:
            slots.paint(
                blocker[slot],
                record["attempt"],
                record["issue"],
                origin,
                blocker_to_id[f"stall:{record['category']}"],
            )
            slots.paint(
                blocker[slot],
                record["issue"],
                record["complete"],
                origin,
                blocker_to_id[f"issue:{record['category']}"],
            )
            if record["category"] == "MFMA":
                slots.paint(
                    mfma_issue[slot],
                    record["issue"],
                    record["issue"] + slots.TICK_CYCLES,
                    origin,
                    True,
                )
                slots.paint(
                    mfma[slot],
                    record["issue"],
                    record["issue"] + slots.MFMA_EXEC_CYCLES,
                    origin,
                    True,
                )

    for slot in range(slot_count):
        state = phase[slot].astype(np.int64) * len(blocker_names) + blocker[
            slot
        ].astype(np.int64)
        for value in np.unique(state):
            phase_id = int(value) // len(blocker_names)
            blocker_id = int(value) % len(blocker_names)
            name = exclusive_reason(phase_id, blocker_names[blocker_id])
            reason[slot, state == value] = reason_to_id[name]

    return {
        "origin": origin,
        "phase": phase,
        "blocker": blocker,
        "blocker_names": blocker_names,
        "active": active,
        "mfma": mfma,
        "mfma_issue": mfma_issue,
        "reason": reason,
        "n_index": n_index,
        "single_windows": single_windows,
    }


def analyze_single_wave(group: dict, first_n: int, last_n_exclusive: int) -> dict:
    mfma = group["mfma"]
    reason = group["reason"]
    counts = Counter()
    slot_counts: dict[int, Counter] = defaultdict(Counter)
    per_n_cycles: dict[str, list[float]] = defaultdict(list)
    total = 0
    busy = 0
    n_blocks = 0

    for window in group["single_windows"]:
        n_block = window["n_block"]
        if not first_n <= n_block < last_n_exclusive:
            continue
        slot = window["slot"]
        left = window["left"]
        right = window["right"]
        if right <= left:
            continue
        window_mfma = mfma[slot, left:right]
        idle_reason = reason[slot, left:right][~window_mfma]
        reason_counts = np.bincount(idle_reason, minlength=len(ALL_REASONS))
        window_cycles = (right - left) * slots.TICK_CYCLES
        busy_cycles = int(np.count_nonzero(window_mfma)) * slots.TICK_CYCLES
        total += window_cycles
        busy += busy_cycles
        n_blocks += 1
        slot_counts[slot]["total_cycles"] += window_cycles
        slot_counts[slot]["mfma_busy_cycles"] += busy_cycles
        slot_counts[slot]["n_blocks"] += 1
        for reason_id, reason_name in enumerate(ALL_REASONS):
            cycles = int(reason_counts[reason_id]) * slots.TICK_CYCLES
            counts[reason_name] += cycles
            slot_counts[slot][reason_name] += cycles
            per_n_cycles[reason_name].append(cycles)

    idle = total - busy
    if busy + sum(counts.values()) != total:
        raise RuntimeError("single-wave ledger does not close")
    by_slot = {}
    for slot, values in sorted(slot_counts.items()):
        slot_total = values["total_cycles"]
        slot_busy = values["mfma_busy_cycles"]
        slot_idle = slot_total - slot_busy
        by_slot[str(slot)] = {
            "n_blocks": values["n_blocks"],
            "cycles_per_n": slot_total / values["n_blocks"],
            "mfma_busy_fraction": slot_busy / slot_total,
            "reasons": counter_summary(values, slot_total, slot_idle),
        }
    return {
        "n_blocks": n_blocks,
        "cycles_per_n": total / n_blocks,
        "total_cycles": total,
        "mfma_busy_cycles": busy,
        "mfma_busy_fraction": busy / total,
        "mfma_idle_cycles": idle,
        "mfma_idle_fraction": idle / total,
        "reasons": counter_summary(counts, total, idle),
        "cycles_per_n_distribution": {
            reason_name: distribution(values)
            for reason_name, values in per_n_cycles.items()
        },
        "by_slot": by_slot,
    }


def wave_is_oracle_fillable(phase: int, reason_name: str, targets: Set[str]) -> bool:
    if "structural_tail" in targets and phase == slots.PHASE_TAIL:
        return True
    return reason_name in targets and phase in slots.CORE_PHASES | slots.BOUNDARY_PHASES


def analyze_physical(
    group: dict,
    first_n: int,
    last_n_exclusive: int,
    requested_resident_waves: int | None,
    targets: Set[str],
) -> dict:
    phase = group["phase"]
    active = group["active"]
    mfma = group["mfma"]
    reason = group["reason"]
    n_index = group["n_index"]
    active_count = active.sum(axis=0)
    inferred_resident_waves = int(active_count.max())
    resident_waves = requested_resident_waves or inferred_resident_waves
    in_loop = np.isin(
        phase, list(slots.CORE_PHASES | slots.BOUNDARY_PHASES | slots.TAIL_PHASES)
    )
    in_requested_n = (n_index >= first_n) & (n_index < last_n_exclusive)
    steady = active_count == resident_waves
    steady &= np.all(~active | (in_loop & in_requested_n), axis=0)
    union_busy = steady & np.any(mfma, axis=0)
    union_idle = steady & ~np.any(mfma, axis=0)

    owner = defaultdict(float)
    any_reason = Counter()
    all_same = Counter()
    joint = Counter()
    oracle_recovered = 0
    oracle_recovered_by_reason = defaultdict(float)
    oracle_remaining_owner = defaultdict(float)
    strict_vmem_wait_floor = 0

    for tick in np.flatnonzero(union_idle):
        active_slots = np.flatnonzero(active[:, tick])
        names = [ALL_REASONS[int(reason[slot, tick])] for slot in active_slots]
        phases = [int(phase[slot, tick]) for slot in active_slots]
        if len(active_slots) != resident_waves:
            raise RuntimeError("physical steady tick does not have full occupancy")
        counts = Counter(names)
        joint_name = " + ".join(
            f"{name}*{count}" for name, count in sorted(counts.items())
        )
        joint[joint_name] += slots.TICK_CYCLES
        for name, count in counts.items():
            owner[name] += slots.TICK_CYCLES * count / resident_waves
            any_reason[name] += slots.TICK_CYCLES
        if len(counts) == 1:
            all_same[names[0]] += slots.TICK_CYCLES

        fillable = [
            wave_is_oracle_fillable(wave_phase, name, targets)
            for wave_phase, name in zip(phases, names)
        ]
        if any(fillable):
            oracle_recovered += slots.TICK_CYCLES
            fillable_names = [name for name, value in zip(names, fillable) if value]
            fillable_counts = Counter(fillable_names)
            for name, count in fillable_counts.items():
                oracle_recovered_by_reason[name] += (
                    slots.TICK_CYCLES * count / len(fillable_names)
                )
        else:
            for name, count in counts.items():
                oracle_remaining_owner[name] += (
                    slots.TICK_CYCLES * count / resident_waves
                )
            core_names = [
                name
                for wave_phase, name in zip(phases, names)
                if wave_phase in slots.CORE_PHASES | slots.BOUNDARY_PHASES
            ]
            if core_names and all(name == "vmem_wait_stall" for name in core_names):
                strict_vmem_wait_floor += slots.TICK_CYCLES

    total = int(np.count_nonzero(steady)) * slots.TICK_CYCLES
    busy = int(np.count_nonzero(union_busy)) * slots.TICK_CYCLES
    idle = int(np.count_nonzero(union_idle)) * slots.TICK_CYCLES
    mfma_execution_mass = int(np.count_nonzero(mfma[:, steady])) * slots.TICK_CYCLES
    mfma_overlap = mfma_execution_mass - busy
    if busy + idle != total:
        raise RuntimeError("physical union ledger does not close")
    if not math.isclose(sum(owner.values()), idle, abs_tol=1e-6):
        raise RuntimeError("physical owner ledger does not close")
    if sum(joint.values()) != idle:
        raise RuntimeError("physical joint ledger does not close")
    if oracle_recovered + sum(oracle_remaining_owner.values()) != total - busy:
        raise RuntimeError("oracle recovered/remaining ledger does not close")

    remaining = idle - oracle_recovered
    return {
        "resident_waves": resident_waves,
        "inferred_max_resident_waves": inferred_resident_waves,
        "total_cycles": total,
        "mfma_busy_cycles": busy,
        "mfma_busy_fraction": busy / total,
        "mfma_idle_cycles": idle,
        "mfma_idle_fraction": idle / total,
        "mfma_execution_mass_cycles": mfma_execution_mass,
        "mfma_execution_mass_fraction": mfma_execution_mass / total,
        "mfma_overlap_cycles": mfma_overlap,
        "mfma_overlap_fraction": mfma_overlap / total,
        "owner_reasons": counter_summary(owner, total, idle),
        "any_wave_reason_cycles": dict(any_reason),
        "all_waves_same_reason_cycles": dict(all_same),
        "joint_reason_cycles": dict(joint),
        "oracle": {
            "targets": sorted(targets),
            "recovered_cycles_upper_bound": oracle_recovered,
            "recovered_total_fraction_upper_bound": oracle_recovered / total,
            "recovered_idle_fraction_upper_bound": oracle_recovered / idle,
            "recovered_by_reason_shares": dict(oracle_recovered_by_reason),
            "remaining_cycles": remaining,
            "remaining_total_fraction": remaining / total,
            "remaining_owner_reasons": counter_summary(
                oracle_remaining_owner, total, remaining
            ),
            "strict_vmem_wait_floor_cycles": strict_vmem_wait_floor,
            "strict_vmem_wait_floor_total_fraction": strict_vmem_wait_floor / total,
        },
    }


def merge_counter_summaries(results: list[dict], path: tuple[str, ...]) -> Counter:
    counter = Counter()
    for result in results:
        value = result
        for key in path:
            value = value[key]
        counter.update(value)
    return counter


def analyze_trace(
    label: str,
    dispatch: Path,
    first_n: int,
    last_n_exclusive: int,
    resident_waves: int | None,
    targets: Set[str],
) -> dict:
    code, groups, trace_stats = slots.load_group_payload(dispatch)
    readiness_proof = analyze_mfma_readiness(groups, first_n, last_n_exclusive)
    blocker_names = build_blocker_names(code)
    group_results = []
    for key, waves in sorted(groups.items()):
        painted = paint_group(waves, blocker_names)
        group_results.append(
            {
                "key": list(key),
                "single_wave": analyze_single_wave(painted, first_n, last_n_exclusive),
                "mfma_issue_unavailability": analyze_mfma_issue_unavailability(
                    painted, first_n, last_n_exclusive
                ),
                "physical": analyze_physical(
                    painted,
                    first_n,
                    last_n_exclusive,
                    resident_waves,
                    targets,
                ),
            }
        )

    single_total = sum(
        result["single_wave"]["total_cycles"] for result in group_results
    )
    single_busy = sum(
        result["single_wave"]["mfma_busy_cycles"] for result in group_results
    )
    single_n = sum(result["single_wave"]["n_blocks"] for result in group_results)
    single_reasons = Counter()
    for result in group_results:
        for reason_name, values in result["single_wave"]["reasons"].items():
            single_reasons[reason_name] += values["cycles"]
    single_idle = single_total - single_busy
    if single_busy + sum(single_reasons.values()) != single_total:
        raise RuntimeError(f"{label}: merged single-wave ledger does not close")
    unavailable = merge_counter_summaries(group_results, ("mfma_issue_unavailability",))
    unavailable_cycles = unavailable["wave_ticks"] * slots.TICK_CYCLES
    if unavailable_cycles != single_reasons["mfma_issue_unavailable"]:
        raise RuntimeError(f"{label}: MFMA issue-unavailable proof does not close")

    physical_total = sum(result["physical"]["total_cycles"] for result in group_results)
    physical_busy = sum(
        result["physical"]["mfma_busy_cycles"] for result in group_results
    )
    physical_idle = physical_total - physical_busy
    physical_mfma_mass = sum(
        result["physical"]["mfma_execution_mass_cycles"] for result in group_results
    )
    physical_mfma_overlap = physical_mfma_mass - physical_busy
    owner = Counter()
    joint = Counter()
    all_same = Counter()
    any_reason = Counter()
    oracle_recovered_by_reason = Counter()
    oracle_remaining_owner = Counter()
    oracle_recovered = 0
    strict_floor = 0
    inferred = set()
    residents = set()
    for result in group_results:
        physical = result["physical"]
        inferred.add(physical["inferred_max_resident_waves"])
        residents.add(physical["resident_waves"])
        for reason_name, values in physical["owner_reasons"].items():
            owner[reason_name] += values["cycles"]
        joint.update(physical["joint_reason_cycles"])
        all_same.update(physical["all_waves_same_reason_cycles"])
        any_reason.update(physical["any_wave_reason_cycles"])
        oracle = physical["oracle"]
        oracle_recovered += oracle["recovered_cycles_upper_bound"]
        oracle_recovered_by_reason.update(oracle["recovered_by_reason_shares"])
        strict_floor += oracle["strict_vmem_wait_floor_cycles"]
        for reason_name, values in oracle["remaining_owner_reasons"].items():
            oracle_remaining_owner[reason_name] += values["cycles"]
    if len(inferred) != 1 or len(residents) != 1:
        raise RuntimeError(f"{label}: inconsistent resident-wave counts")
    if physical_busy + sum(owner.values()) != physical_total:
        raise RuntimeError(f"{label}: merged physical owner ledger does not close")
    if sum(joint.values()) != physical_idle:
        raise RuntimeError(f"{label}: merged physical joint ledger does not close")
    oracle_remaining = physical_idle - oracle_recovered
    if oracle_recovered + sum(oracle_remaining_owner.values()) != physical_idle:
        raise RuntimeError(f"{label}: merged oracle ledger does not close")

    return {
        "label": label,
        "dispatch": str(dispatch),
        "trace": trace_stats,
        "mfma_readiness_proof": readiness_proof,
        "mfma_issue_unavailability": {
            "raw_stall_cycles": unavailable["raw_stall_ticks"] * slots.TICK_CYCLES,
            "raw_stall_execution_window_overlap_cycles": {
                "own_only": unavailable["raw_stall_own_1_peer_0_ticks"]
                * slots.TICK_CYCLES,
                "peer_only": unavailable["raw_stall_own_0_peer_1_ticks"]
                * slots.TICK_CYCLES,
                "own_and_peer": unavailable["raw_stall_own_1_peer_1_ticks"]
                * slots.TICK_CYCLES,
                "neither": unavailable["raw_stall_own_0_peer_0_ticks"]
                * slots.TICK_CYCLES,
            },
            "cycles": unavailable_cycles,
            "total_fraction": unavailable_cycles / single_total,
            "idle_fraction": unavailable_cycles / single_idle,
            "peer_execution_window_cycles": unavailable["peer_execution_ticks"]
            * slots.TICK_CYCLES,
            "peer_execution_window_coverage_fraction": (
                unavailable["peer_execution_ticks"] / unavailable["wave_ticks"]
                if unavailable["wave_ticks"]
                else 0.0
            ),
            "peer_same_tick_issue_cycles": unavailable["peer_same_tick_issue_ticks"]
            * slots.TICK_CYCLES,
            "peer_same_tick_issue_fraction": (
                unavailable["peer_same_tick_issue_ticks"] / unavailable["wave_ticks"]
                if unavailable["wave_ticks"]
                else 0.0
            ),
            "without_peer_execution_cycles": unavailable["without_peer_execution_ticks"]
            * slots.TICK_CYCLES,
        },
        "single_wave": {
            "first_n": first_n,
            "last_n_exclusive": last_n_exclusive,
            "n_blocks": single_n,
            "cycles_per_n": single_total / single_n,
            "total_cycles": single_total,
            "mfma_busy_cycles": single_busy,
            "mfma_busy_fraction": single_busy / single_total,
            "mfma_idle_cycles": single_idle,
            "mfma_idle_fraction": single_idle / single_total,
            "reasons": counter_summary(single_reasons, single_total, single_idle),
        },
        "physical_union": {
            "resident_waves": residents.pop(),
            "inferred_max_resident_waves": inferred.pop(),
            "total_cycles": physical_total,
            "mfma_busy_cycles": physical_busy,
            "mfma_busy_fraction": physical_busy / physical_total,
            "mfma_idle_cycles": physical_idle,
            "mfma_idle_fraction": physical_idle / physical_total,
            "mfma_execution_mass_cycles": physical_mfma_mass,
            "mfma_execution_mass_fraction": physical_mfma_mass / physical_total,
            "mfma_overlap_cycles": physical_mfma_overlap,
            "mfma_overlap_fraction": physical_mfma_overlap / physical_total,
            "owner_reasons": counter_summary(owner, physical_total, physical_idle),
            "any_wave_reason_cycles": dict(any_reason),
            "all_waves_same_reason_cycles": dict(all_same),
            "joint_reason_cycles": dict(joint),
            "oracle": {
                "targets": sorted(targets),
                "recovered_cycles_upper_bound": oracle_recovered,
                "recovered_total_fraction_upper_bound": oracle_recovered
                / physical_total,
                "recovered_idle_fraction_upper_bound": oracle_recovered / physical_idle,
                "recovered_by_reason_shares": dict(oracle_recovered_by_reason),
                "remaining_cycles": oracle_remaining,
                "remaining_total_fraction": oracle_remaining / physical_total,
                "remaining_owner_reasons": counter_summary(
                    oracle_remaining_owner, physical_total, oracle_remaining
                ),
                "strict_vmem_wait_floor_cycles": strict_floor,
                "strict_vmem_wait_floor_total_fraction": strict_floor / physical_total,
            },
        },
        "group_results": group_results,
    }


def markdown_report(results: list[dict]) -> str:
    lines = [
        "# Fine-grained MFMA gap exposure",
        "",
        "All reason rows are mutually exclusive owner attributions, not additive speedup budgets. Physical `any` and `all` counters are non-additive witnesses; joint states are exact.",
    ]
    for result in results:
        single = result["single_wave"]
        physical = result["physical_union"]
        readiness = result["mfma_readiness_proof"]
        unavailable = result["mfma_issue_unavailability"]
        lines.extend(
            [
                "",
                f"## {result['label']}",
                "",
                f"Single wave: {single['cycles_per_n']:.2f} cycles/N, MFMA busy {single['mfma_busy_fraction']:.2%}.",
                "",
                "| Single-wave reason | cycles/N | total | MFMA-idle |",
                "|---|---:|---:|---:|",
            ]
        )
        for reason_name in ALL_REASONS:
            values = single["reasons"][reason_name]
            lines.append(
                f"| {reason_name} | {values['cycles'] / single['n_blocks']:.2f} | "
                f"{values['total_fraction']:.2%} | {values['idle_fraction']:.2%} |"
            )
        reuse_distance = readiness["accumulator_reuse_mfma_distance"]
        reuse_cycles = readiness["accumulator_reuse_issue_cycles"]
        physical_interval = readiness["physical_successful_issue_interval_cycles"]
        source_a = readiness["source_readiness"]["A/source0"]
        source_b = readiness["source_readiness"]["B/source1"]
        lines.extend(
            [
                "",
                "### MFMA issue-unavailable proof",
                "",
                f"- Checked {readiness['consecutive_pair_checks']:,} dynamic consecutive MFMA pairs; "
                f"back-to-back accumulator RAW pairs: {readiness['back_to_back_accumulator_raw_pairs']:,}.",
                f"- Accumulator reuse distance min/p50/p99: "
                f"{reuse_distance['min']:.0f}/{reuse_distance['p50']:.0f}/{reuse_distance['p99']:.0f} MFMAs; "
                f"successful-issue distance: "
                f"{reuse_cycles['min']:.0f}/{reuse_cycles['p50']:.0f}/{reuse_cycles['p99']:.0f} cycles.",
                f"- Physical-SIMD successful MFMA issue interval min/p50/p99: "
                f"{physical_interval['min']:.0f}/{physical_interval['p50']:.0f}/{physical_interval['p99']:.0f} cycles; "
                f"intervals below {slots.MFMA_EXEC_CYCLES} cycles: "
                f"{readiness['physical_issue_intervals_below_execution_cycles']:,}.",
                f"- A/source0: {source_a['wait_covered_edges']:,}/{source_a['edges']:,} VMEM producer edges "
                "covered by a counter-valid `s_waitcnt vmcnt`.",
                f"- B/source1: {source_b['wait_covered_edges']:,}/{source_b['edges']:,} DS producer edges "
                "covered by a counter-valid `s_waitcnt lgkmcnt`.",
                f"- `mfma_issue_unavailable`: {unavailable['cycles']:,.0f} wave-cycles; peer MFMA execution-window "
                f"coverage {unavailable['peer_execution_window_coverage_fraction']:.2%}; same-tick peer MFMA issue "
                f"{unavailable['peer_same_tick_issue_fraction']:.2%}; without peer execution "
                f"{unavailable['without_peer_execution_cycles']:,.0f} cycles.",
                f"- Raw `stall:MFMA`: {unavailable['raw_stall_cycles']:,.0f} cycles, split into "
                f"own-window only {unavailable['raw_stall_execution_window_overlap_cycles']['own_only']:,.0f}, "
                f"peer-window only {unavailable['raw_stall_execution_window_overlap_cycles']['peer_only']:,.0f}, "
                f"both {unavailable['raw_stall_execution_window_overlap_cycles']['own_and_peer']:,.0f}, "
                f"neither {unavailable['raw_stall_execution_window_overlap_cycles']['neither']:,.0f}.",
                "",
                "Immediate accumulator RAW and uncovered A/B memory readiness are therefore excluded for this "
                "bucket. It means the wave's MFMA/VALU issue class was temporarily unavailable while another "
                "resident wave owned the same SIMD's 16-cycle MFMA issue slot. ATT's 4-cycle `issue_cost` is the "
                "trace record cost, not the physical MFMA initiation interval; the merged successful-issue "
                "timeline has no interval below 16 cycles. Priority can change wave arbitration, but this is a "
                "scheduling defect only if it creates later physical-union idle time; peer-covered single-wave "
                "stalls are not an additive performance gap.",
            ]
        )
        lines.extend(
            [
                "",
                f"Physical union ({physical['resident_waves']} waves): MFMA busy {physical['mfma_busy_fraction']:.2%}.",
                "",
                "| Physical owner reason | total | MFMA-idle | all waves same / total |",
                "|---|---:|---:|---:|",
            ]
        )
        for reason_name in ALL_REASONS:
            values = physical["owner_reasons"][reason_name]
            all_same = physical["all_waves_same_reason_cycles"].get(reason_name, 0)
            lines.append(
                f"| {reason_name} | {values['total_fraction']:.2%} | "
                f"{values['idle_fraction']:.2%} | {all_same / physical['total_cycles']:.2%} |"
            )
        oracle = physical["oracle"]
        lines.extend(
            [
                "",
                f"Oracle target set: `{', '.join(oracle['targets'])}`.",
                f"Upper-bound recoverable physical cycles: {oracle['recovered_total_fraction_upper_bound']:.2%} of steady time; residual: {oracle['remaining_total_fraction']:.2%}.",
                f"Strict VMEM-wait witness: {oracle['strict_vmem_wait_floor_total_fraction']:.4%} of steady time.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_trace(value: str) -> tuple[str, Path]:
    return slots.parse_trace_argument(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, type=parse_trace)
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=slots.N_BLOCKS,
        help="N blocks executed by each complete wave (default: 16)",
    )
    parser.add_argument("--first-n", type=int, default=2)
    parser.add_argument("--last-n-exclusive", type=int, default=14)
    parser.add_argument("--resident-waves", type=int)
    parser.add_argument(
        "--software-target",
        action="append",
        choices=ALL_REASONS,
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.n_blocks <= 0:
        parser.error("--n-blocks must be positive")
    slots.configure_n_blocks(args.n_blocks)
    if not 0 <= args.first_n < args.last_n_exclusive <= slots.N_BLOCKS - 1:
        parser.error(
            "single-wave N range must satisfy "
            f"0 <= first < last <= {slots.N_BLOCKS - 1}"
        )
    targets = set(args.software_target or DEFAULT_SOFTWARE_TARGETS)
    results = [
        analyze_trace(
            label,
            dispatch,
            args.first_n,
            args.last_n_exclusive,
            args.resident_waves,
            targets,
        )
        for label, dispatch in args.trace
    ]
    payload = {
        "schema_version": 2,
        "model": {
            "tick_cycles": slots.TICK_CYCLES,
            "mfma_execution_cycles": slots.MFMA_EXEC_CYCLES,
            "reason_priority": "explicit memory blocker, then structural tail, then residual",
            "physical_owner_policy": "split each physical idle tick equally across resident waves",
            "mfma_issue_unavailable_policy": (
                "exclude immediate accumulator RAW and waitcnt-covered A/B producers, then report peer MFMA "
                "execution-window coverage; only physical-union exposure is a scheduling gap"
            ),
            "oracle_policy": (
                "a tick is optimistically recovered if any core/boundary wave has a targeted blocker, "
                "or any tail wave is targeted by structural_tail"
            ),
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    report = markdown_report(results)
    if args.markdown:
        args.markdown.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
