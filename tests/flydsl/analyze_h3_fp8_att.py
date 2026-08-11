"""Analyze decoded rocprofv3 ATT output for H3 FP8 attention kernels."""

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    "MFMA",
    "EXP",
    "VALU",
    "VMEM",
    "LDS read",
    "LDS write",
    "waitcnt",
    "barrier",
    "SALU/branch",
    "other",
)


def instruction_category(instruction):
    instruction = instruction.strip()
    if "mfma" in instruction:
        return "MFMA"
    if instruction.startswith("v_exp"):
        return "EXP"
    if instruction.startswith(
        ("buffer_load", "buffer_store", "global_load", "global_store", "flat_load", "flat_store")
    ):
        return "VMEM"
    if instruction.startswith("ds_read"):
        return "LDS read"
    if instruction.startswith("ds_write"):
        return "LDS write"
    if instruction.startswith("s_waitcnt"):
        return "waitcnt"
    if instruction.startswith("s_barrier"):
        return "barrier"
    if instruction.startswith("v_"):
        return "VALU"
    if instruction.startswith("s_"):
        return "SALU/branch"
    return "other"


def union_intervals(intervals, begin, end):
    merged = []
    for left, right in sorted(
        (max(left, begin), min(right, end))
        for left, right in intervals
        if right > begin and left < end
    ):
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return merged


def intersection_length(lhs, rhs):
    lhs_index = rhs_index = total = 0
    while lhs_index < len(lhs) and rhs_index < len(rhs):
        begin = max(lhs[lhs_index][0], rhs[rhs_index][0])
        end = min(lhs[lhs_index][1], rhs[rhs_index][1])
        total += max(0, end - begin)
        if lhs[lhs_index][1] < rhs[rhs_index][1]:
            lhs_index += 1
        else:
            rhs_index += 1
    return total


def analyze_trace(root):
    root = Path(root)
    ui_dirs = sorted(root.glob("ui_output_agent_*"))
    stats_paths = sorted(root.glob("stats_ui_output_agent_*.csv"))
    att_paths = sorted(root.glob("*.att"))
    if len(ui_dirs) != 1 or len(stats_paths) != 1 or len(att_paths) != 1:
        raise ValueError(
            f"expected one UI directory, stats CSV, and ATT file under {root}; "
            f"got {len(ui_dirs)}, {len(stats_paths)}, {len(att_paths)}"
        )
    ui_dir = ui_dirs[0]

    wave_rows = []
    intervals = defaultdict(list)
    instructions_by_slot = Counter()
    for path in sorted(ui_dir.glob("se*.json")):
        data = json.loads(path.read_text())
        wave = data["wave"]
        if data["num_insts"] != data["num_stitched"]:
            raise ValueError(
                f"{path} is incomplete: {data['num_stitched']} of {data['num_insts']} instructions stitched"
            )
        key = (wave["simd"], wave["slot"])
        intervals[key].append((wave["begin"], wave["end"]))
        instructions_by_slot[key] += data["num_insts"]
        wave_rows.append(
            {
                "simd": wave["simd"],
                "slot": wave["slot"],
                "begin": wave["begin"],
                "end": wave["end"],
                "duration": data["duration"],
                "num_insts": data["num_insts"],
            }
        )
    if not wave_rows:
        raise ValueError(f"{ui_dir} contains no decoded waves")

    categories = defaultdict(Counter)
    top_rows = []
    with stats_paths[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values = {
                    key: int(row[key]) for key in ("Hitcount", "Latency", "Stall", "Idle")
                }
            except ValueError:
                continue
            category = instruction_category(row["Instruction"])
            categories[category].update(
                hit=values["Hitcount"],
                latency=values["Latency"],
                stall=values["Stall"],
                idle=values["Idle"],
            )
            top_rows.append(
                {
                    "instruction": row["Instruction"],
                    "hitcount": values["Hitcount"],
                    "latency": values["Latency"],
                    "stall": values["Stall"],
                    "idle": values["Idle"],
                }
            )

    mfma_count = categories["MFMA"]["hit"]
    if mfma_count == 0:
        raise ValueError(f"{stats_paths[0]} contains no MFMA hits")
    total_stall = sum(values["stall"] for values in categories.values())
    total_idle = sum(values["idle"] for values in categories.values())
    total_latency = sum(values["latency"] for values in categories.values())
    total_hits = sum(values["hit"] for values in categories.values())

    timeline_begin = min(row["begin"] for row in wave_rows)
    timeline_end = max(row["end"] for row in wave_rows)
    trim_begin = timeline_begin + (timeline_end - timeline_begin) // 10
    trim_end = timeline_end - (timeline_end - timeline_begin) // 10
    overlap_percent = []
    issue_density = []
    for simd in sorted({row["simd"] for row in wave_rows}):
        slot0 = union_intervals(intervals[(simd, 0)], trim_begin, trim_end)
        slot1 = union_intervals(intervals[(simd, 1)], trim_begin, trim_end)
        overlap_percent.append(
            intersection_length(slot0, slot1) / (trim_end - trim_begin) * 100.0
        )
        active_cycles = sum(
            row["duration"] for row in wave_rows if row["simd"] == simd
        )
        dynamic_instructions = sum(
            row["num_insts"] for row in wave_rows if row["simd"] == simd
        )
        issue_density.append(dynamic_instructions / active_cycles)

    return {
        "root": str(root),
        "ui_directory": str(ui_dir),
        "att_file": str(att_paths[0]),
        "att_bytes": att_paths[0].stat().st_size,
        "wave_count": len(wave_rows),
        "all_waves_complete": True,
        "simds": sorted({row["simd"] for row in wave_rows}),
        "slots": sorted({row["slot"] for row in wave_rows}),
        "timeline_begin": timeline_begin,
        "timeline_end": timeline_end,
        "wave_duration_median": statistics.median(row["duration"] for row in wave_rows),
        "wave_instruction_median": statistics.median(row["num_insts"] for row in wave_rows),
        "resident_slot_overlap_percent_by_simd": overlap_percent,
        "resident_slot_overlap_percent_median": statistics.median(overlap_percent),
        "instruction_issue_density_by_simd": issue_density,
        "instruction_issue_density_median": statistics.median(issue_density),
        "mfma_count": mfma_count,
        "total_instruction_hits": total_hits,
        "total_latency": total_latency,
        "total_stall": total_stall,
        "total_idle": total_idle,
        "instruction_hits_per_mfma": total_hits / mfma_count,
        "latency_per_mfma": total_latency / mfma_count,
        "stall_per_mfma": total_stall / mfma_count,
        "idle_per_mfma": total_idle / mfma_count,
        "stall_plus_idle_per_mfma": (total_stall + total_idle) / mfma_count,
        "categories": {
            category: {
                "hitcount": categories[category]["hit"],
                "latency": categories[category]["latency"],
                "stall": categories[category]["stall"],
                "idle": categories[category]["idle"],
                "stall_per_mfma": categories[category]["stall"] / mfma_count,
                "stall_share_percent": categories[category]["stall"] / total_stall * 100.0,
            }
            for category in CATEGORIES
        },
        "top_stalls": sorted(top_rows, key=lambda row: row["stall"], reverse=True)[:20],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    labels = args.labels or [path.name for path in args.traces]
    if len(labels) != len(args.traces):
        parser.error("--labels must have the same length as traces")

    results = []
    for label, path in zip(labels, args.traces):
        result = analyze_trace(path)
        result["label"] = label
        results.append(result)
        print(
            f"{label}: waves={result['wave_count']} complete={result['all_waves_complete']} "
            f"MFMA={result['mfma_count']} stall/MFMA={result['stall_per_mfma']:.3f} "
            f"idle/MFMA={result['idle_per_mfma']:.3f} "
            f"issue_density={result['instruction_issue_density_median']:.6f} "
            f"slot_overlap={result['resident_slot_overlap_percent_median']:.3f}%"
        )
    output = {"schema_version": 1, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"analysis_output={args.output}")


if __name__ == "__main__":
    main()