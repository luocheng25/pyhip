#!/usr/bin/env python3
"""Build a physical-SIMD MFMA execution-slot ledger from gfx9 ATT traces.

rocprof ATT records gfx9 instructions as::

    [first_attempt, category, stall, duration, pc_index]

The successful issue time is ``first_attempt + stall``.  Treating the first
attempt as the issue time overstates cross-wave overlap.  This analyzer merges
both resident wave slots on each physical SIMD, marks every successful MFMA as
a 16-cycle matrix execution window, and classifies uncovered windows using the
instructions blocking both resident waves.

The fixability labels are deliberately conservative:

* local schedule candidate: a non-MFMA issue/stall occurs while a wave still has
  later MFMAs in the same K core;
* prefetch candidate: VMEM/LDS/wait exposure occurs in a core or core boundary;
* structural tail: both waves are outside their MFMA cores, so filling the gap
  requires cross-N double buffering or a different epilogue, not local motion;
* edge/replacement: fewer than two resident waves are active and is excluded
  from the core utilization denominator.

Usage:

    python tests/contrib/moe/analyze_down_mfma_slots.py \
      --trace 'control=/tmp/.../ui_output_agent_*_dispatch_19' \
      --trace '9aa=/tmp/.../ui_output_agent_*_dispatch_19' \
      --workers 4 --json /tmp/slots.json --markdown /tmp/slots.md \
      --svg /tmp/slots.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

TICK_CYCLES = 4
MFMA_EXEC_CYCLES = 16
MFMA_PER_CORE = 64
CORES_PER_N_BLOCK = 3
MFMA_PER_N_BLOCK = MFMA_PER_CORE * CORES_PER_N_BLOCK
N_BLOCKS = 16
EXPECTED_MFMA_PER_WAVE = MFMA_PER_N_BLOCK * N_BLOCKS
WAVE_FILE_RE = re.compile(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json")

PHASE_INACTIVE = 0
PHASE_PROLOGUE = 1
PHASE_CORE0 = 2
PHASE_BOUNDARY01 = 3
PHASE_CORE1 = 4
PHASE_BOUNDARY12 = 5
PHASE_CORE2 = 6
PHASE_TAIL = 7
PHASE_DRAIN = 8
PHASE_NAMES = {
    PHASE_INACTIVE: "inactive",
    PHASE_PROLOGUE: "prologue",
    PHASE_CORE0: "core0",
    PHASE_BOUNDARY01: "core0->1",
    PHASE_CORE1: "core1",
    PHASE_BOUNDARY12: "core1->2",
    PHASE_CORE2: "core2",
    PHASE_TAIL: "tail",
    PHASE_DRAIN: "drain",
}
CORE_PHASES = {PHASE_CORE0, PHASE_CORE1, PHASE_CORE2}
BOUNDARY_PHASES = {PHASE_BOUNDARY01, PHASE_BOUNDARY12}
TAIL_PHASES = {PHASE_TAIL}
EDGE_PHASES = {PHASE_PROLOGUE, PHASE_DRAIN}


def configure_n_blocks(n_blocks: int) -> None:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    global N_BLOCKS, EXPECTED_MFMA_PER_WAVE
    N_BLOCKS = n_blocks
    EXPECTED_MFMA_PER_WAVE = MFMA_PER_N_BLOCK * N_BLOCKS


@dataclass(frozen=True)
class CodeInfo:
    asm: str
    opcode: str
    category: str
    source: str


def instruction_category(asm: str) -> str:
    opcode = asm.strip().lower()
    if opcode.startswith("v_mfma"):
        return "MFMA"
    if opcode.startswith("s_waitcnt"):
        has_vmem = "vmcnt" in opcode
        has_lds = "lgkmcnt" in opcode
        if has_vmem and has_lds:
            return "wait-mixed"
        if has_vmem:
            return "wait-vmcnt"
        if has_lds:
            return "wait-lgkmcnt"
        return "wait-other"
    if opcode.startswith(("buffer_load", "global_load", "flat_load")):
        return "VMEM-load"
    if opcode.startswith(("buffer_store", "global_store", "flat_store")):
        return "VMEM-store"
    if opcode.startswith("ds_read"):
        return "DS-read"
    if opcode.startswith("ds_write"):
        return "DS-write"
    if opcode.startswith("ds_"):
        return "LDS/crosslane"
    if opcode.startswith("s_barrier"):
        return "barrier"
    if opcode.startswith(("v_exp", "v_rcp")):
        return "TRANS"
    if opcode.startswith("v_"):
        return "VALU"
    if opcode.startswith("s_load"):
        return "SMEM"
    if opcode.startswith("s_"):
        return "SALU/control"
    return "other"


def opcode_of(asm: str) -> str:
    parts = asm.strip().split()
    return parts[0] if parts else "other"


def parse_trace_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--trace must be LABEL=DISPATCH_DIR")
    label, path = value.split("=", 1)
    dispatch = Path(path).resolve()
    if not label or not dispatch.is_dir():
        raise argparse.ArgumentTypeError(f"invalid trace: {value}")
    return label, dispatch


def load_code(dispatch: Path) -> tuple[list[CodeInfo], list[list]]:
    rows = json.loads((dispatch / "code.json").read_text(encoding="utf-8"))["code"]
    max_index = max(int(row[2]) for row in rows)
    code = [CodeInfo("", "other", "other", "") for _ in range(max_index + 1)]
    for row in rows:
        asm = row[0].strip()
        code[int(row[2])] = CodeInfo(
            asm=asm,
            opcode=opcode_of(asm),
            category=instruction_category(asm),
            source=row[3] or "",
        )
    return code, rows


def quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def tick_floor(value: int, origin: int) -> int:
    return max(0, (value - origin) // TICK_CYCLES)


def tick_ceil(value: int, origin: int) -> int:
    return max(0, (value - origin + TICK_CYCLES - 1) // TICK_CYCLES)


def paint(array: np.ndarray, begin: int, end: int, origin: int, value: int) -> None:
    if end <= begin:
        return
    left = tick_floor(begin, origin)
    right = min(array.shape[-1], tick_ceil(end, origin))
    if right > left:
        array[left:right] = value


def classify_fixability(phases: tuple[int, int], blockers: tuple[str, str]) -> str:
    active = [
        (phase, blocker)
        for phase, blocker in zip(phases, blockers)
        if phase != PHASE_INACTIVE
    ]
    if len(active) < 2:
        return "edge/replacement"
    phase_values = {phase for phase, _ in active}
    if phase_values <= TAIL_PHASES:
        return "structural tail"
    if phase_values <= EDGE_PHASES | TAIL_PHASES and not (
        phase_values & (CORE_PHASES | BOUNDARY_PHASES)
    ):
        return "edge/prologue/drain"

    preferred = [(phase, blocker) for phase, blocker in active if phase in CORE_PHASES]
    if not preferred:
        preferred = [
            (phase, blocker) for phase, blocker in active if phase in BOUNDARY_PHASES
        ]
    if not preferred:
        return "structural tail"
    blocker_values = [blocker for _, blocker in preferred]

    if any(blocker.startswith("stall:wait-mixed") for blocker in blocker_values):
        return "mixed vmcnt/lgkmcnt wait candidate"
    if any(
        blocker.startswith("stall:wait-vmcnt") or blocker.startswith("stall:VMEM-")
        for blocker in blocker_values
    ):
        return "VMEM stall/wait candidate"
    if any(blocker.startswith("issue:VMEM-") for blocker in blocker_values):
        return "VMEM issue scheduling"
    if any(
        blocker.startswith("stall:wait-lgkmcnt")
        or blocker.startswith("stall:DS-")
        or blocker.startswith("stall:LDS/")
        for blocker in blocker_values
    ):
        return "LDS stall/wait candidate"
    if any(
        blocker.startswith("issue:DS-") or blocker.startswith("issue:LDS/")
        for blocker in blocker_values
    ):
        return "LDS issue scheduling"
    if any(blocker.startswith("stall:MFMA") for blocker in blocker_values):
        return "MFMA dependency/operand-ready"
    if any(blocker.startswith("stall:VALU") for blocker in blocker_values):
        return "VALU dependency stall"
    if any(blocker.startswith("stall:TRANS") for blocker in blocker_values):
        return "TRANS dependency stall"
    if any(blocker.startswith("stall:barrier") for blocker in blocker_values):
        return "barrier imbalance"
    if any(blocker.startswith("issue:VALU") for blocker in blocker_values):
        return "VALU issue scheduling"
    if any(blocker.startswith("issue:TRANS") for blocker in blocker_values):
        return "TRANS issue scheduling"
    if any(blocker.startswith("issue:") for blocker in blocker_values):
        return "other issue scheduling"
    if any(blocker == "scheduler/ready" for blocker in blocker_values):
        return "scheduler/ready candidate"
    if any("barrier" in blocker for blocker in blocker_values):
        return "barrier imbalance"
    return "other core exposure"


def counter_from_joint(
    joint_values: np.ndarray,
    mask: np.ndarray,
    names: list[str],
    cycle_scale: int,
) -> Counter:
    counts = Counter()
    for joint, count in enumerate(np.bincount(joint_values[mask])):
        if count == 0:
            continue
        left = joint // len(names)
        right = joint % len(names)
        name = "+".join(sorted((names[left], names[right])))
        counts[name] += int(count) * cycle_scale
    return counts


def blocker_counters(
    blocker_joint: np.ndarray,
    mask: np.ndarray,
    blocker_names: list[str],
    cycle_scale: int,
) -> tuple[Counter, dict[str, float]]:
    pair_counts = Counter()
    owner_counts: dict[str, float] = defaultdict(float)
    for joint, count in enumerate(np.bincount(blocker_joint[mask])):
        if count == 0:
            continue
        left = joint // len(blocker_names)
        right = joint % len(blocker_names)
        pair = " + ".join(sorted((blocker_names[left], blocker_names[right])))
        cycles = int(count) * cycle_scale
        pair_counts[pair] += cycles
        owner_counts[blocker_names[left]] += cycles / 2
        owner_counts[blocker_names[right]] += cycles / 2
    return pair_counts, owner_counts


def fixability_counter(
    full_joint: np.ndarray,
    mask: np.ndarray,
    blocker_names: list[str],
    cycle_scale: int,
) -> Counter:
    counts = Counter()
    for joint, count in enumerate(np.bincount(full_joint[mask])):
        if count == 0:
            continue
        phase_id = joint // (len(blocker_names) ** 2)
        blocker_id = joint % (len(blocker_names) ** 2)
        phases = (phase_id // len(PHASE_NAMES), phase_id % len(PHASE_NAMES))
        blockers = (
            blocker_names[blocker_id // len(blocker_names)],
            blocker_names[blocker_id % len(blocker_names)],
        )
        counts[classify_fixability(phases, blockers)] += int(count) * cycle_scale
    return counts


def load_group_payload(
    dispatch: Path,
) -> tuple[list[CodeInfo], dict[tuple[int, int, int], list[dict]], dict]:
    code, code_rows = load_code(dispatch)
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    trace_stats = {
        "waves": 0,
        "complete_waves": 0,
        "early_exit_waves": 0,
        "active_waves": 0,
        "context_records": 0,
        "opcode_stats": defaultdict(lambda: {"count": 0, "stall": 0, "issue": 0}),
    }
    for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
        match = WAVE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        wave = data["wave"]
        trace_stats["waves"] += 1
        trace_stats["complete_waves"] += int(data["num_insts"] == data["num_stitched"])
        records = []
        mfma_ordinal = 0
        for raw in wave["instructions"]:
            pc_index = int(raw[4])
            info = code[pc_index]
            attempt = int(raw[0])
            stall = int(raw[2])
            duration = int(raw[3])
            issue = attempt + stall
            complete = attempt + duration
            record = {
                "attempt": attempt,
                "issue": issue,
                "complete": complete,
                "stall": stall,
                "issue_cost": duration - stall,
                "pc_index": pc_index,
                "opcode": info.opcode,
                "category": info.category,
                "asm": info.asm,
                "source": info.source,
                "mfma_ordinal": mfma_ordinal if info.category == "MFMA" else None,
            }
            if info.category == "MFMA":
                mfma_ordinal += 1
            if int(raw[1]) == 10:
                trace_stats["context_records"] += 1
            op_entry = trace_stats["opcode_stats"][info.opcode]
            op_entry["count"] += 1
            op_entry["stall"] += stall
            op_entry["issue"] += duration - stall
            records.append(record)
        if mfma_ordinal == 0 and data["num_insts"] == data["num_stitched"]:
            trace_stats["early_exit_waves"] += 1
            continue
        if mfma_ordinal != EXPECTED_MFMA_PER_WAVE:
            raise RuntimeError(
                f"{path}: expected {EXPECTED_MFMA_PER_WAVE} MFMA, got {mfma_ordinal}"
            )
        trace_stats["active_waves"] += 1
        key = (int(match.group(1)), int(wave["cu"]), int(wave["simd"]))
        groups[key].append(
            {
                "path": str(path),
                "begin": int(wave["begin"]),
                "end": int(wave["end"]),
                "slot": int(wave["slot"]),
                "records": records,
            }
        )
    if not groups:
        raise RuntimeError(f"no raw wave files in {dispatch}")
    if trace_stats["waves"] != trace_stats["complete_waves"]:
        raise RuntimeError(f"incomplete raw waves in {dispatch}: {trace_stats}")
    trace_stats["opcode_stats"] = dict(trace_stats["opcode_stats"])
    trace_stats["static_instructions"] = len(code_rows)
    trace_stats["static_mfma"] = sum(
        instruction_category(row[0]) == "MFMA" for row in code_rows
    )
    trace_stats["static_setprio"] = sum(
        row[0].strip().startswith("s_setprio") for row in code_rows
    )
    return code, groups, trace_stats


def analyze_group(payload: tuple[tuple[int, int, int], list[dict], list[str]]) -> dict:
    key, waves, blocker_names = payload
    waves = sorted(waves, key=lambda wave: (wave["begin"], wave["slot"]))
    origin = min(wave["begin"] for wave in waves)
    end = max(wave["end"] for wave in waves)
    ticks = tick_ceil(end, origin)
    phase = np.zeros((2, ticks), dtype=np.uint8)
    blocker = np.zeros((2, ticks), dtype=np.uint16)
    active = np.zeros((2, ticks), dtype=np.bool_)
    mfma_exec = np.zeros(ticks, dtype=np.bool_)
    non_mfma_issue = np.zeros((2, ticks), dtype=np.uint16)
    blocker_to_id = {name: index for index, name in enumerate(blocker_names)}

    single_wave_gaps = Counter()
    gap_samples: dict[str, list[int]] = defaultdict(list)
    wave_durations = []
    for wave in waves:
        slot = wave["slot"]
        wave_durations.append(wave["end"] - wave["begin"])
        paint(active[slot], wave["begin"], wave["end"], origin, True)
        paint(phase[slot], wave["begin"], wave["end"], origin, PHASE_PROLOGUE)
        paint(
            blocker[slot],
            wave["begin"],
            wave["end"],
            origin,
            blocker_to_id["scheduler/ready"],
        )

        mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
        for tile in range(N_BLOCKS):
            tile_base = tile * MFMA_PER_N_BLOCK
            cores = []
            for core in range(CORES_PER_N_BLOCK):
                core_records = mfmas[
                    tile_base
                    + core * MFMA_PER_CORE : tile_base
                    + (core + 1) * MFMA_PER_CORE
                ]
                core_begin = core_records[0]["issue"]
                core_end = core_records[-1]["issue"] + MFMA_EXEC_CYCLES
                cores.append((core_begin, core_end))
                paint(phase[slot], core_begin, core_end, origin, PHASE_CORE0 + 2 * core)
            paint(phase[slot], cores[0][1], cores[1][0], origin, PHASE_BOUNDARY01)
            paint(phase[slot], cores[1][1], cores[2][0], origin, PHASE_BOUNDARY12)
            if tile + 1 < N_BLOCKS:
                next_begin = mfmas[(tile + 1) * MFMA_PER_N_BLOCK]["issue"]
                paint(phase[slot], cores[2][1], next_begin, origin, PHASE_TAIL)
        paint(
            phase[slot],
            mfmas[-1]["issue"] + MFMA_EXEC_CYCLES,
            wave["end"],
            origin,
            PHASE_DRAIN,
        )

        for left, right in zip(mfmas, mfmas[1:]):
            exposed = max(0, right["issue"] - left["issue"] - MFMA_EXEC_CYCLES)
            if exposed == 0:
                continue
            ordinal = int(left["mfma_ordinal"])
            position = ordinal % MFMA_PER_N_BLOCK
            if position == MFMA_PER_CORE - 1:
                kind = "core0->1"
            elif position == 2 * MFMA_PER_CORE - 1:
                kind = "core1->2"
            elif position == MFMA_PER_N_BLOCK - 1:
                kind = "tail->next-N"
            else:
                kind = "intra-core"
            single_wave_gaps[kind] += exposed
            gap_samples[kind].append(exposed)

        for record in wave["records"]:
            stall_name = f"stall:{record['category']}"
            issue_name = f"issue:{record['category']}"
            paint(
                blocker[slot],
                record["attempt"],
                record["issue"],
                origin,
                blocker_to_id[stall_name],
            )
            paint(
                blocker[slot],
                record["issue"],
                record["complete"],
                origin,
                blocker_to_id[issue_name],
            )
            if record["category"] != "MFMA":
                paint(
                    non_mfma_issue[slot],
                    record["issue"],
                    record["complete"],
                    origin,
                    blocker_to_id[issue_name],
                )
            else:
                paint(
                    mfma_exec,
                    record["issue"],
                    record["issue"] + MFMA_EXEC_CYCLES,
                    origin,
                    True,
                )

    both_active = active[0] & active[1]
    in_n_loop = np.isin(
        phase,
        list(CORE_PHASES | BOUNDARY_PHASES | TAIL_PHASES),
    )
    steady = both_active & in_n_loop[0] & in_n_loop[1]
    busy = both_active & mfma_exec
    idle = both_active & ~mfma_exec
    steady_busy = steady & mfma_exec
    steady_idle = steady & ~mfma_exec
    cycle_scale = TICK_CYCLES
    phase_joint = phase[0].astype(np.int64) * len(PHASE_NAMES) + phase[1].astype(
        np.int64
    )
    blocker_joint = blocker[0].astype(np.int64) * len(blocker_names) + blocker[
        1
    ].astype(np.int64)
    full_joint = phase_joint * (len(blocker_names) ** 2) + blocker_joint

    phase_counts = counter_from_joint(
        phase_joint, both_active, list(PHASE_NAMES.values()), cycle_scale
    )
    phase_idle_counts = counter_from_joint(
        phase_joint, idle, list(PHASE_NAMES.values()), cycle_scale
    )
    steady_phase_counts = counter_from_joint(
        phase_joint, steady, list(PHASE_NAMES.values()), cycle_scale
    )
    steady_phase_idle_counts = counter_from_joint(
        phase_joint, steady_idle, list(PHASE_NAMES.values()), cycle_scale
    )
    blocker_pair_counts, blocker_owner_counts = blocker_counters(
        blocker_joint, idle, blocker_names, cycle_scale
    )
    steady_blocker_pair_counts, steady_blocker_owner_counts = blocker_counters(
        blocker_joint, steady_idle, blocker_names, cycle_scale
    )
    fixability_counts = fixability_counter(full_joint, idle, blocker_names, cycle_scale)
    steady_fixability_counts = fixability_counter(
        full_joint, steady_idle, blocker_names, cycle_scale
    )

    coissue_counts = Counter()
    for slot in range(2):
        values = non_mfma_issue[slot][busy]
        for blocker_id, count in enumerate(np.bincount(values)):
            if blocker_id and count:
                coissue_counts[blocker_names[blocker_id]] += int(count) * cycle_scale

    return {
        "key": list(key),
        "waves": len(waves),
        "wave_duration_median": statistics.median(wave_durations),
        "both_active_cycles": int(np.count_nonzero(both_active)) * cycle_scale,
        "mfma_busy_cycles": int(np.count_nonzero(busy)) * cycle_scale,
        "mfma_idle_cycles": int(np.count_nonzero(idle)) * cycle_scale,
        "steady_cycles": int(np.count_nonzero(steady)) * cycle_scale,
        "steady_mfma_busy_cycles": int(np.count_nonzero(steady_busy)) * cycle_scale,
        "steady_mfma_idle_cycles": int(np.count_nonzero(steady_idle)) * cycle_scale,
        "phase_cycles": dict(phase_counts),
        "phase_idle_cycles": dict(phase_idle_counts),
        "steady_phase_cycles": dict(steady_phase_counts),
        "steady_phase_idle_cycles": dict(steady_phase_idle_counts),
        "blocker_pair_cycles": dict(blocker_pair_counts),
        "blocker_owner_cycles": dict(blocker_owner_counts),
        "steady_blocker_pair_cycles": dict(steady_blocker_pair_counts),
        "steady_blocker_owner_cycles": dict(steady_blocker_owner_counts),
        "fixability_cycles": dict(fixability_counts),
        "steady_fixability_cycles": dict(steady_fixability_counts),
        "coissue_cycles": dict(coissue_counts),
        "single_wave_gap_cycles": dict(single_wave_gaps),
        "single_wave_gap_stats": {
            name: {
                "count": len(values),
                "median": statistics.median(values),
                "p95": quantile(values, 0.95),
                "max": max(values),
            }
            for name, values in gap_samples.items()
        },
    }


def merge_numeric_dict(results: list[dict], key: str) -> dict[str, float]:
    merged = Counter()
    for result in results:
        merged.update(result[key])
    return dict(merged)


def read_kernel_metadata(dispatch: Path) -> dict:
    candidates = [
        dispatch.parent / "out_kernel_trace.csv",
        dispatch.parent.parent / "out_kernel_trace.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if "moe_2stage_down_prefill_1x4_0" in row.get("Kernel_Name", "")
            ]
        if not rows:
            continue
        dispatch_match = re.search(r"_dispatch_(\d+)$", dispatch.name)
        if dispatch_match:
            matching = [
                row for row in rows if row.get("Dispatch_Id") == dispatch_match.group(1)
            ]
            if matching:
                rows = matching
        row = rows[-1]
        return {
            "dispatch_ms": (int(row["End_Timestamp"]) - int(row["Start_Timestamp"]))
            / 1e6,
            "lds_bytes": int(row["LDS_Block_Size"]),
            "vgpr": int(row["VGPR_Count"]),
            "accum_vgpr": int(row["Accum_VGPR_Count"]),
            "sgpr": int(row["SGPR_Count"]),
            "workgroup_size": int(row["Workgroup_Size_X"]),
            "grid_size": [
                int(row["Grid_Size_X"]),
                int(row["Grid_Size_Y"]),
                int(row["Grid_Size_Z"]),
            ],
        }
    return {}


def summarize_trace(label: str, dispatch: Path, workers: int) -> dict:
    code, groups, trace_stats = load_group_payload(dispatch)
    blocker_names = ["inactive", "scheduler/ready"]
    categories = sorted({info.category for info in code})
    blocker_names.extend(f"stall:{category}" for category in categories)
    blocker_names.extend(f"issue:{category}" for category in categories)
    payloads = [(key, waves, blocker_names) for key, waves in sorted(groups.items())]
    if workers == 1:
        group_results = [analyze_group(payload) for payload in payloads]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(workers, len(payloads))) as pool:
            group_results = pool.map(analyze_group, payloads)

    both_active = sum(result["both_active_cycles"] for result in group_results)
    busy = sum(result["mfma_busy_cycles"] for result in group_results)
    idle = sum(result["mfma_idle_cycles"] for result in group_results)
    steady = sum(result["steady_cycles"] for result in group_results)
    steady_busy = sum(result["steady_mfma_busy_cycles"] for result in group_results)
    steady_idle = sum(result["steady_mfma_idle_cycles"] for result in group_results)
    if busy + idle != both_active:
        raise RuntimeError(f"physical ledger does not close for {label}")
    fixability = merge_numeric_dict(group_results, "fixability_cycles")
    if abs(sum(fixability.values()) - idle) > 1e-6:
        raise RuntimeError(f"fixability ledger does not close for {label}")
    steady_fixability = merge_numeric_dict(group_results, "steady_fixability_cycles")
    if steady_busy + steady_idle != steady:
        raise RuntimeError(f"steady MFMA ledger does not close for {label}")
    if abs(sum(steady_fixability.values()) - steady_idle) > 1e-6:
        raise RuntimeError(f"steady fixability ledger does not close for {label}")

    normal_issue_cost = defaultdict(list)
    for opcode, values in trace_stats["opcode_stats"].items():
        if values["count"]:
            normal_issue_cost[opcode].append(values["issue"] / values["count"])
    result = {
        "label": label,
        "dispatch": str(dispatch),
        "metadata": read_kernel_metadata(dispatch),
        "trace": trace_stats,
        "physical_simds": len(group_results),
        "group_results": group_results,
        "both_active_cycles": both_active,
        "mfma_busy_cycles": busy,
        "mfma_idle_cycles": idle,
        "mfma_busy_fraction": busy / both_active,
        "mfma_idle_fraction": idle / both_active,
        "equivalent_16cycle_slots": both_active / MFMA_EXEC_CYCLES,
        "busy_16cycle_slots": busy / MFMA_EXEC_CYCLES,
        "idle_16cycle_slots": idle / MFMA_EXEC_CYCLES,
        "steady_cycles": steady,
        "steady_mfma_busy_cycles": steady_busy,
        "steady_mfma_idle_cycles": steady_idle,
        "steady_mfma_busy_fraction": steady_busy / steady,
        "steady_mfma_idle_fraction": steady_idle / steady,
        "steady_equivalent_16cycle_slots": steady / MFMA_EXEC_CYCLES,
        "steady_busy_16cycle_slots": steady_busy / MFMA_EXEC_CYCLES,
        "steady_idle_16cycle_slots": steady_idle / MFMA_EXEC_CYCLES,
        "phase_cycles": merge_numeric_dict(group_results, "phase_cycles"),
        "phase_idle_cycles": merge_numeric_dict(group_results, "phase_idle_cycles"),
        "blocker_pair_cycles": merge_numeric_dict(group_results, "blocker_pair_cycles"),
        "blocker_owner_cycles": merge_numeric_dict(
            group_results, "blocker_owner_cycles"
        ),
        "fixability_cycles": fixability,
        "steady_phase_cycles": merge_numeric_dict(group_results, "steady_phase_cycles"),
        "steady_phase_idle_cycles": merge_numeric_dict(
            group_results, "steady_phase_idle_cycles"
        ),
        "steady_blocker_pair_cycles": merge_numeric_dict(
            group_results, "steady_blocker_pair_cycles"
        ),
        "steady_blocker_owner_cycles": merge_numeric_dict(
            group_results, "steady_blocker_owner_cycles"
        ),
        "steady_fixability_cycles": steady_fixability,
        "coissue_cycles": merge_numeric_dict(group_results, "coissue_cycles"),
        "single_wave_gap_cycles": merge_numeric_dict(
            group_results, "single_wave_gap_cycles"
        ),
        "wave_duration_median": statistics.median(
            result["wave_duration_median"] for result in group_results
        ),
        "normal_issue_cost_by_opcode": {
            opcode: statistics.median(values)
            for opcode, values in normal_issue_cost.items()
        },
    }
    return result


def top_rows(
    mapping: dict[str, float], total: float, count: int = 10
) -> list[tuple[str, float, float]]:
    return [
        (name, value, 100.0 * value / total if total else 0.0)
        for name, value in sorted(
            mapping.items(), key=lambda item: item[1], reverse=True
        )[:count]
    ]


def markdown_report(results: list[dict]) -> str:
    lines = [
        "# Control K128 physical MFMA slot model",
        "",
        "ATT gfx9 timestamps use `successful_issue = first_attempt + stall`; each successful FP8 MFMA is marked as a 16-cycle matrix execution window. The main denominator is the steady N-loop where both resident slots are in core/core-boundary/tail phases; prologue, drain and slot replacement are reported separately as lifecycle context.",
        "",
        "| Trace | SIMD | Waves | steady MFMA busy | steady MFMA idle | steady 16-cycle idle slots | lifecycle busy | Dispatch | Resources |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metadata = result["metadata"]
        resources = (
            f"{metadata.get('vgpr', '?')}V+{metadata.get('accum_vgpr', '?')}A, "
            f"{metadata.get('lds_bytes', '?')}B LDS"
        )
        lines.append(
            f"| {result['label']} | {result['physical_simds']} | {result['trace']['waves']} | "
            f"{result['steady_mfma_busy_fraction']:.2%} | {result['steady_mfma_idle_fraction']:.2%} | "
            f"{result['steady_idle_16cycle_slots']:.1f} | {result['mfma_busy_fraction']:.2%} | "
            f"{metadata.get('dispatch_ms', float('nan')):.6f} ms | {resources} |"
        )
    for result in results:
        idle = result["steady_mfma_idle_cycles"]
        lines.extend(
            [
                "",
                f"## {result['label']}",
                "",
                "### Fixability ledger",
                "",
                "| Bucket | Cycles | Share of MFMA-idle | 16-cycle slots |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, cycles, share in top_rows(
            result["steady_fixability_cycles"], idle, 20
        ):
            lines.append(
                f"| {name} | {cycles:,.0f} | {share:.2f}% | {cycles / MFMA_EXEC_CYCLES:,.1f} |"
            )
        lines.extend(
            [
                "",
                "### Phase-pair idle",
                "",
                "| Resident phases | Idle cycles | Share |",
                "|---|---:|---:|",
            ]
        )
        for name, cycles, share in top_rows(
            result["steady_phase_idle_cycles"], idle, 20
        ):
            lines.append(f"| {name} | {cycles:,.0f} | {share:.2f}% |")
        lines.extend(
            [
                "",
                "### Physical blocker pairs",
                "",
                "| Pair | Idle cycles | Share |",
                "|---|---:|---:|",
            ]
        )
        for name, cycles, share in top_rows(
            result["steady_blocker_pair_cycles"], idle, 15
        ):
            lines.append(f"| `{name}` | {cycles:,.0f} | {share:.2f}% |")
    return "\n".join(lines) + "\n"


def svg_report(results: list[dict]) -> str:
    width = 1200
    row_height = 184
    height = 96 + row_height * len(results)
    colors = {
        "busy": "#238636",
        "VMEM stall/wait candidate": "#d29922",
        "VMEM issue scheduling": "#e3b341",
        "mixed vmcnt/lgkmcnt wait candidate": "#f0883e",
        "LDS stall/wait candidate": "#db6d28",
        "LDS issue scheduling": "#f78166",
        "MFMA dependency/operand-ready": "#a371f7",
        "VALU dependency stall": "#8957e5",
        "VALU issue scheduling": "#1f6feb",
        "TRANS dependency stall": "#bf4b8a",
        "TRANS issue scheduling": "#db61a2",
        "other issue scheduling": "#58a6ff",
        "scheduler/ready candidate": "#58a6ff",
        "structural tail": "#da3633",
        "edge/prologue/drain": "#6e7681",
        "edge/replacement": "#8b949e",
        "other core exposure": "#bc8cff",
        "barrier imbalance": "#f85149",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        "<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#e6edf3}.label{font-size:18px;font-weight:600}.small{font-size:14px;fill:#b1bac4}.legend{font-size:13px;fill:#d0d7de}</style>",
        '<text x="40" y="42" class="label">Physical SIMD MFMA execution slots (two resident waves)</text>',
        '<text x="40" y="66" class="small">Steady N-loop only. Green = 16-cycle MFMA execution; other colors classify uncovered physical cycles. ATT issue = first_attempt + stall.</text>',
    ]
    bar_x = 40
    bar_width = 1120
    bar_height = 30
    legend_columns = 3
    legend_column_width = bar_width / legend_columns
    for row_index, result in enumerate(results):
        y = 96 + row_index * row_height
        bar_y = y + 54
        legend_y = bar_y + bar_height + 28
        total = result["steady_cycles"]
        segments = [("busy", result["steady_mfma_busy_cycles"])] + sorted(
            result["steady_fixability_cycles"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        parts.append(
            f'<text x="40" y="{y + 24}" class="label">{result["label"]}</text>'
        )
        parts.append(
            f'<text x="40" y="{y + 49}" class="small">steady busy {result["steady_mfma_busy_fraction"]:.2%}; idle {result["steady_mfma_idle_fraction"]:.2%}; lifecycle busy {result["mfma_busy_fraction"]:.2%}</text>'
        )
        cursor = bar_x
        for name, cycles in segments:
            segment_width = bar_width * cycles / total
            if segment_width <= 0:
                continue
            color = colors.get(name, "#8b949e")
            parts.append(
                f'<rect x="{cursor:.2f}" y="{bar_y}" width="{segment_width:.2f}" height="{bar_height}" fill="{color}"><title>{name}: {cycles / total:.2%}</title></rect>'
            )
            cursor += segment_width
        for legend_index, (name, cycles) in enumerate(segments[:9]):
            legend_column = legend_index % legend_columns
            legend_row = legend_index // legend_columns
            legend_x = bar_x + legend_column * legend_column_width
            item_y = legend_y + legend_row * 24
            color = colors.get(name, "#8b949e")
            parts.append(
                f'<rect x="{legend_x:.2f}" y="{item_y - 12}" width="14" height="14" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{legend_x + 20:.2f}" y="{item_y}" class="legend">{name} {cycles / total:.1%}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def interval_union_cycles(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    right = -1
    for left, stop in sorted(intervals):
        if stop <= right:
            continue
        covered += stop - max(left, right)
        right = stop
    return covered


def select_detail_window(dispatch: Path, target_busy: float) -> dict:
    _, groups, _ = load_group_payload(dispatch)
    candidates = []
    for key, waves in sorted(groups.items()):
        for anchor in waves:
            if anchor["slot"] != 0:
                continue
            anchor_mfmas = [
                record for record in anchor["records"] if record["category"] == "MFMA"
            ]
            for tile in range(2, N_BLOCKS - 2):
                begin = anchor_mfmas[tile * MFMA_PER_N_BLOCK]["issue"]
                end = anchor_mfmas[(tile + 1) * MFMA_PER_N_BLOCK]["issue"]
                peers = []
                for peer in waves:
                    if peer["slot"] == anchor["slot"]:
                        continue
                    peer_mfmas = [
                        record
                        for record in peer["records"]
                        if record["category"] == "MFMA"
                    ]
                    covers_window = peer["begin"] <= begin and peer["end"] >= end
                    covers_mfma_lifecycle = (
                        peer_mfmas[0]["issue"] <= begin
                        and peer_mfmas[-1]["issue"] + MFMA_EXEC_CYCLES >= end
                    )
                    if covers_window and covers_mfma_lifecycle:
                        peers.append(peer)
                if len(peers) != 1:
                    continue
                peer = peers[0]
                intervals = []
                for wave in (anchor, peer):
                    for record in wave["records"]:
                        if record["category"] != "MFMA":
                            continue
                        left = max(begin, record["issue"])
                        stop = min(end, record["issue"] + MFMA_EXEC_CYCLES)
                        if stop > left:
                            intervals.append((left, stop))
                busy = interval_union_cycles(intervals) / (end - begin)
                candidates.append(
                    {
                        "key": key,
                        "tile": tile,
                        "begin": begin,
                        "end": end,
                        "busy": busy,
                        "anchor": anchor,
                        "peer": peer,
                    }
                )
    if not candidates:
        raise RuntimeError(f"no complete two-slot N-loop windows in {dispatch}")

    median_duration = statistics.median(
        candidate["end"] - candidate["begin"] for candidate in candidates
    )
    near_target = [
        candidate
        for candidate in candidates
        if abs(candidate["busy"] - target_busy) <= 0.001
    ]
    pool = near_target or candidates
    chosen = min(
        pool,
        key=lambda candidate: (
            abs((candidate["end"] - candidate["begin"]) - median_duration),
            abs(candidate["busy"] - target_busy),
            candidate["key"],
            candidate["tile"],
            candidate["anchor"]["path"],
        ),
    )
    chosen["candidate_count"] = len(candidates)
    chosen["near_target_count"] = len(near_target)
    chosen["median_duration"] = median_duration
    return chosen


def wave_phase_segments(wave: dict) -> list[tuple[int, int, int, str]]:
    mfmas = [record for record in wave["records"] if record["category"] == "MFMA"]
    segments = []
    for tile in range(N_BLOCKS):
        tile_base = tile * MFMA_PER_N_BLOCK
        cores = []
        for core in range(CORES_PER_N_BLOCK):
            begin = mfmas[tile_base + core * MFMA_PER_CORE]["issue"]
            end = (
                mfmas[tile_base + (core + 1) * MFMA_PER_CORE - 1]["issue"]
                + MFMA_EXEC_CYCLES
            )
            phase = PHASE_CORE0 + 2 * core
            segments.append((begin, end, phase, f"N{tile} K{core}"))
            cores.append((begin, end))
        segments.append((cores[0][1], cores[1][0], PHASE_BOUNDARY01, f"N{tile} K0->K1"))
        segments.append((cores[1][1], cores[2][0], PHASE_BOUNDARY12, f"N{tile} K1->K2"))
        if tile + 1 < N_BLOCKS:
            next_begin = mfmas[(tile + 1) * MFMA_PER_N_BLOCK]["issue"]
            segments.append((cores[2][1], next_begin, PHASE_TAIL, f"N{tile} tail"))
    return segments


DETAIL_EVENT_NAMES = [
    "none",
    "VMEM stall/wait",
    "VMEM issue",
    "LDS stall/wait",
    "LDS issue",
    "mixed wait",
    "MFMA ATT stall",
    "VALU issue",
    "other stall/issue",
]


def detail_event_id(category: str, stalled: bool) -> int:
    if category in {"VMEM-load", "VMEM-store", "wait-vmcnt"}:
        return 1 if stalled else 2
    if category in {"DS-read", "DS-write", "LDS/crosslane", "wait-lgkmcnt"}:
        return 3 if stalled else 4
    if category == "wait-mixed":
        return 5
    if category == "MFMA" and stalled:
        return 6
    if category == "VALU" and not stalled:
        return 7
    return 8


def build_detail_arrays(
    wave: dict,
    begin: int,
    end: int,
    blocker_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ticks = tick_ceil(end, begin)
    phase = np.zeros(ticks, dtype=np.uint8)
    blocker = np.full(ticks, blocker_names.index("scheduler/ready"), dtype=np.uint16)
    mfma_exec = np.zeros(ticks, dtype=np.bool_)
    events = np.zeros(ticks, dtype=np.uint8)
    blocker_to_id = {name: index for index, name in enumerate(blocker_names)}

    for left, stop, phase_id, _ in wave_phase_segments(wave):
        paint(phase, left, stop, begin, phase_id)
    for record in wave["records"]:
        if record["complete"] <= begin or record["attempt"] >= end:
            continue
        stall_name = f"stall:{record['category']}"
        issue_name = f"issue:{record['category']}"
        paint(
            blocker,
            record["attempt"],
            record["issue"],
            begin,
            blocker_to_id[stall_name],
        )
        paint(
            blocker,
            record["issue"],
            record["complete"],
            begin,
            blocker_to_id[issue_name],
        )
        if record["stall"]:
            paint(
                events,
                record["attempt"],
                record["issue"],
                begin,
                detail_event_id(record["category"], True),
            )
        if record["issue_cost"]:
            paint(
                events,
                record["issue"],
                record["complete"],
                begin,
                detail_event_id(record["category"], False),
            )
        if record["category"] == "MFMA":
            paint(
                mfma_exec,
                record["issue"],
                record["issue"] + MFMA_EXEC_CYCLES,
                begin,
                True,
            )
    return phase, blocker, mfma_exec, events


def array_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    runs = []
    left = 0
    value = int(values[0])
    for index in range(1, values.size):
        next_value = int(values[index])
        if next_value == value:
            continue
        runs.append((left, index, value))
        left = index
        value = next_value
    runs.append((left, values.size, value))
    return runs


def append_timeline_runs(
    parts: list[str],
    values: np.ndarray,
    y: int,
    height: int,
    plot_x: int,
    plot_width: int,
    names: list[str],
    colors: dict[str, str],
    skip_zero: bool = True,
) -> None:
    ticks = values.size
    for left, right, value in array_runs(values):
        if skip_zero and value == 0:
            continue
        name = names[value]
        x = plot_x + plot_width * left / ticks
        width = plot_width * (right - left) / ticks
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="{height}" fill="{colors[name]}">'
            f"<title>{escape(name)}: {(right - left) * TICK_CYCLES} cycles</title></rect>"
        )


def detail_svg_report(label: str, dispatch: Path, target_busy: float) -> str:
    code, _, _ = load_group_payload(dispatch)
    detail = select_detail_window(dispatch, target_busy)
    begin = detail["begin"]
    end = detail["end"]
    duration = end - begin
    waves = [detail["anchor"], detail["peer"]]
    blocker_names = ["inactive", "scheduler/ready"]
    categories = sorted({info.category for info in code})
    blocker_names.extend(f"stall:{category}" for category in categories)
    blocker_names.extend(f"issue:{category}" for category in categories)
    slot_arrays = [
        build_detail_arrays(wave, begin, end, blocker_names) for wave in waves
    ]
    ticks = slot_arrays[0][0].size

    physical_names = [
        "MFMA execution",
        "VMEM stall/wait candidate",
        "LDS stall/wait candidate",
        "mixed vmcnt/lgkmcnt wait candidate",
        "structural tail",
        "scheduler/other",
    ]
    physical = np.zeros(ticks, dtype=np.uint8)
    for tick in range(ticks):
        if slot_arrays[0][2][tick] or slot_arrays[1][2][tick]:
            continue
        phases = (int(slot_arrays[0][0][tick]), int(slot_arrays[1][0][tick]))
        blockers = (
            blocker_names[int(slot_arrays[0][1][tick])],
            blocker_names[int(slot_arrays[1][1][tick])],
        )
        bucket = classify_fixability(phases, blockers)
        if bucket == "VMEM stall/wait candidate":
            physical[tick] = 1
        elif bucket == "LDS stall/wait candidate":
            physical[tick] = 2
        elif bucket == "mixed vmcnt/lgkmcnt wait candidate":
            physical[tick] = 3
        elif bucket == "structural tail":
            physical[tick] = 4
        else:
            physical[tick] = 5

    anchor_counts = Counter(
        record["opcode"]
        for record in waves[0]["records"]
        if begin <= record["issue"] < end
    )
    expected_counts = {
        "v_mfma_f32_16x16x32_fp8_fp8": 192,
        "buffer_load_dwordx4": 24,
        "ds_read_b128": 32,
        "v_fmaak_f32": 64,
        "v_perm_b32": 32,
        "ds_write_b128": 16,
        "buffer_store_dwordx4": 8,
    }
    for opcode, expected in expected_counts.items():
        if anchor_counts[opcode] != expected:
            raise RuntimeError(
                f"detail window has {anchor_counts[opcode]} {opcode}, expected {expected}"
            )

    phase_names = [PHASE_NAMES[index] for index in range(len(PHASE_NAMES))]
    phase_colors = {
        "inactive": "#30363d",
        "prologue": "#6e7681",
        "core0": "#1f6feb",
        "core0->1": "#484f58",
        "core1": "#388bfd",
        "core1->2": "#484f58",
        "core2": "#58a6ff",
        "tail": "#da3633",
        "drain": "#8b949e",
    }
    event_colors = {
        "none": "#161b22",
        "VMEM stall/wait": "#d29922",
        "VMEM issue": "#e3b341",
        "LDS stall/wait": "#db6d28",
        "LDS issue": "#f78166",
        "mixed wait": "#f0883e",
        "MFMA ATT stall": "#8957e5",
        "VALU issue": "#1f6feb",
        "other stall/issue": "#6e7681",
    }
    physical_colors = {
        "MFMA execution": "#238636",
        "VMEM stall/wait candidate": "#d29922",
        "LDS stall/wait candidate": "#db6d28",
        "mixed vmcnt/lgkmcnt wait candidate": "#f0883e",
        "structural tail": "#da3633",
        "scheduler/other": "#6e7681",
    }

    width = 1400
    height = 878
    plot_x = 190
    plot_width = 1170
    axis_y = 310
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        "<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#e6edf3}.title{font-size:20px;font-weight:600}.label{font-size:15px;font-weight:600}.small{font-size:13px;fill:#b1bac4}.tiny{font-size:12px;fill:#8b949e}</style>",
        f'<text x="40" y="38" class="title">{escape(label)}: representative physical two-slot N-loop</text>',
        f'<text x="40" y="62" class="small">SE{detail["key"][0]} CU{detail["key"][1]} SIMD{detail["key"][2]}; slot 0 N{detail["tile"]}; absolute cycles [{begin}, {end}); {duration} cycles; MFMA busy {detail["busy"]:.2%}</text>',
        f'<text x="40" y="82" class="tiny">Selected from {detail["candidate_count"]} complete windows: busy within 0.1 percentage point of {target_busy:.2%}, then duration nearest the {detail["median_duration"]:.0f}-cycle median. ATT issue = first_attempt + stall.</text>',
        '<text x="40" y="112" class="label">Anchor slot 0 logical N-block recipe (counts validated against this ATT window)</text>',
    ]

    recipe_y = 126
    recipe_height = 110
    recipe_x = [40, 300, 560, 820]
    recipe_width = [250, 250, 250, 540]
    for core in range(CORES_PER_N_BLOCK):
        parts.extend(
            [
                f'<rect x="{recipe_x[core]}" y="{recipe_y}" width="{recipe_width[core]}" height="{recipe_height}" rx="4" fill="#161b22" stroke="{phase_colors[f"core{core}"]}"/>',
                f'<text x="{recipe_x[core] + 14}" y="{recipe_y + 24}" class="label">K{core}: K128 core</text>',
                f'<text x="{recipe_x[core] + 14}" y="{recipe_y + 47}" class="small">8 x ds_read + 8 x VMEM load</text>',
                f'<text x="{recipe_x[core] + 14}" y="{recipe_y + 70}" class="tiny">DSRD8 -> 8 x (VMEM1 -> MFMA4)</text>',
                f'<text x="{recipe_x[core] + 14}" y="{recipe_y + 91}" class="small">-> MFMA32 (64 MFMA total)</text>',
            ]
        )
    parts.extend(
        [
            f'<rect x="{recipe_x[3]}" y="{recipe_y}" width="{recipe_width[3]}" height="{recipe_height}" rx="4" fill="#161b22" stroke="{phase_colors["tail"]}"/>',
            f'<text x="{recipe_x[3] + 14}" y="{recipe_y + 24}" class="label">Postprocess / CShuffle / store</text>',
            f'<text x="{recipe_x[3] + 14}" y="{recipe_y + 47}" class="small">64 x FMA + 32 x perm</text>',
            f'<text x="{recipe_x[3] + 14}" y="{recipe_y + 70}" class="small">16 x LDS write -> wait -> 8 x LDS read</text>',
            f'<text x="{recipe_x[3] + 14}" y="{recipe_y + 93}" class="small">8 x VMEM store; next N waits for accumulator retirement</text>',
            '<text x="40" y="274" class="label">Actual ATT timeline (4-cycle ticks; colored runs preserve observed timing)</text>',
        ]
    )

    tick_values = list(range(0, duration, 1000)) + [duration]
    for cycle in tick_values:
        x = plot_x + plot_width * cycle / duration
        parts.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="592" stroke="#30363d"/>'
        )
        anchor = "end" if cycle == duration else ("start" if cycle == 0 else "middle")
        parts.append(
            f'<text x="{x:.2f}" y="{axis_y - 8}" text-anchor="{anchor}" class="tiny">+{cycle}</text>'
        )

    row_y = [(334, 364, 390), (448, 478, 504)]
    for slot, (wave, arrays) in enumerate(zip(waves, slot_arrays)):
        phase, _, mfma_exec, events = arrays
        phase_y, mfma_y, event_y = row_y[slot]
        parts.append(
            f'<text x="40" y="{phase_y + 16}" class="label">slot {slot} phase</text>'
        )
        parts.append(
            f'<text x="40" y="{mfma_y + 14}" class="small">MFMA execution</text>'
        )
        parts.append(
            f'<text x="40" y="{event_y + 14}" class="small">ATT stall / issue</text>'
        )
        append_timeline_runs(
            parts,
            phase,
            phase_y,
            22,
            plot_x,
            plot_width,
            phase_names,
            phase_colors,
            False,
        )
        append_timeline_runs(
            parts,
            mfma_exec.astype(np.uint8),
            mfma_y,
            16,
            plot_x,
            plot_width,
            ["none", "MFMA execution"],
            {"none": "#161b22", "MFMA execution": physical_colors["MFMA execution"]},
        )
        append_timeline_runs(
            parts,
            events,
            event_y,
            16,
            plot_x,
            plot_width,
            DETAIL_EVENT_NAMES,
            event_colors,
        )
        parts.append(
            f'<text x="{plot_x}" y="{event_y + 34}" class="tiny">{escape(Path(wave["path"]).name)}</text>'
        )

    physical_y = 582
    parts.append(
        f'<text x="40" y="{physical_y + 18}" class="label">physical union</text>'
    )
    append_timeline_runs(
        parts,
        physical,
        physical_y,
        28,
        plot_x,
        plot_width,
        physical_names,
        physical_colors,
        False,
    )
    busy_cycles = int(np.count_nonzero(physical == 0)) * TICK_CYCLES
    parts.append(
        f'<text x="{plot_x}" y="638" class="small">Any-slot MFMA execution: {busy_cycles:,} / {duration:,} cycles = {busy_cycles / duration:.2%}. Idle colors use the same conservative two-wave classifier as the overview.</text>'
    )

    legend_rows = [
        (
            666,
            [
                ("phase core0", phase_colors["core0"]),
                ("phase core1", phase_colors["core1"]),
                ("phase core2", phase_colors["core2"]),
                ("phase tail", phase_colors["tail"]),
            ],
        ),
        (698, [(name, event_colors[name]) for name in DETAIL_EVENT_NAMES[1:5]]),
        (730, [(name, event_colors[name]) for name in DETAIL_EVENT_NAMES[5:]]),
        (762, [(name, physical_colors[name]) for name in physical_names[1:]]),
    ]
    for y, entries in legend_rows:
        column_width = 1320 / len(entries)
        for index, (name, color) in enumerate(entries):
            x = 40 + index * column_width
            parts.append(
                f'<rect x="{x:.2f}" y="{y - 13}" width="14" height="14" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + 20:.2f}" y="{y}" class="small">{escape(name)}</text>'
            )

    parts.extend(
        [
            '<text x="40" y="806" class="small">Reading rule: a wave can issue VMEM/LDS/VALU while an earlier MFMA remains in its 16-cycle execution window, so the MFMA and ATT-event rows are intentionally separate.</text>',
            '<text x="40" y="828" class="small">An idle physical segment means neither resident slot executes MFMA. Its color describes the observed two-wave blockers; it is not a claim that every colored cycle is locally removable.</text>',
            '<text x="40" y="850" class="tiny">The anchor excludes N0/N1 and the final two N blocks; both wave files cover the full window with no slot replacement, prologue, drain, or incomplete stitching.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", action="append", required=True, type=parse_trace_argument
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 1)))
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=N_BLOCKS,
        help="N blocks executed by each complete wave (default: 16)",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--detail-svg", type=Path)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.n_blocks <= 0:
        parser.error("--n-blocks must be positive")
    configure_n_blocks(args.n_blocks)

    results = [
        summarize_trace(label, dispatch, args.workers) for label, dispatch in args.trace
    ]
    payload = {
        "schema_version": 1,
        "model": {
            "tick_cycles": TICK_CYCLES,
            "mfma_execution_cycles": MFMA_EXEC_CYCLES,
            "successful_issue_formula": "first_attempt + stall",
            "physical_group_key": ["shader_engine", "cu", "simd"],
            "denominator": "steady N-loop cycles with both resident slots in core, core-boundary, or tail phases",
            "lifecycle_denominator": "all cycles with both resident wave slots active",
            "attribution_policy": "split physical idle equally between simultaneous per-wave blockers",
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown_report(results), encoding="utf-8")
    if args.svg:
        args.svg.write_text(svg_report(results), encoding="utf-8")
    if args.detail_svg:
        label, dispatch = args.trace[0]
        args.detail_svg.write_text(
            detail_svg_report(label, dispatch, results[0]["steady_mfma_busy_fraction"]),
            encoding="utf-8",
        )
    print(markdown_report(results))


if __name__ == "__main__":
    main()
