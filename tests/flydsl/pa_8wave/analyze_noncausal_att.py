"""Summarize complete gfx950 D192 traces without adding overlapping wave stalls.

ATT gfx9 duration is stall + issue time, not MFMA execution latency. The
32-cycle MFMA interval below is an explicitly labelled execution-window model.
CTA stage times instead use observed, aligned workgroup-barrier completions.
This analysis expects the 41-KV-tile, 8-wave, nonpersistent workload.
"""

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if not result or begin > result[-1][1]:
            result.append([begin, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def cycles(intervals):
    return sum(end - begin for begin, end in union(intervals))


def analyze(ui):
    code = json.loads((ui / "code.json").read_text())["code"]
    ctas = defaultdict(list)
    opcode_counts = Counter()
    opcode_stalls = Counter()
    opcode_latencies = Counter()
    opcode_idle = Counter()
    for row in csv.DictReader((ui.parent / f"stats_{ui.name}.csv").open()):
        if int(row["Hitcount"]) == 0:
            continue
        op = row["Instruction"].split()[0]
        opcode_counts[op] += int(row["Hitcount"])
        opcode_stalls[op] += int(row["Stall"])
        opcode_latencies[op] += int(row["Latency"])
        opcode_idle[op] += int(row["Idle"])
    for path in sorted(ui.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text())
        assert payload["num_insts"] == payload["num_stitched"], path
        wave = payload["wave"]
        se = int(path.name.split("_")[0][2:])
        instructions = wave["instructions"]
        assert sum("v_mfma_" in code[e[4]][0] for e in instructions) == 1640, path
        barriers = [e[0] + e[3] for e in instructions if code[e[4]][0] == "s_barrier"]
        assert len(barriers) == 333, (path, len(barriers))
        wave["barriers"] = barriers
        wave["path"] = path.name
        ctas[se, wave["cu"], wave["id"]].append(wave)
    assert ctas, f"No decoded waves in {ui}"
    cta_rows = []
    for key, waves in sorted(ctas.items()):
        assert len(waves) == 8, (key, len(waves))
        assert {(w["simd"], w["slot"]) for w in waves} == {(s, t) for s in range(4) for t in (0, 1)}, key
        assert max(w["begin"] for w in waves) - min(w["begin"] for w in waves) < 512, key
        begin, end = min(w["begin"] for w in waves), max(w["end"] for w in waves)
        boundaries = [max(w["barriers"][i] for w in waves) for i in range(333)]
        assert max(max(w["barriers"][i] for w in waves) - min(w["barriers"][i] for w in waves)
                   for i in range(333)) <= 64, key
        durations = [boundaries[0] - begin] + [b - a for a, b in zip(boundaries, boundaries[1:])]
        # Nonstagger phase t=1 starts at barrier interval 7. Stagger is shifted
        # by one interval, so each physical stage overlaps two different roles.
        # Drop t=1 and t=40 to avoid prologue/tail effects: t=2..39, 38 phases.
        stage_samples = [[durations[7 + t * 8 + stage] for t in range(1, 39)] for stage in range(8)]
        mfma_union = 0
        resident_simd_span = 0
        for simd in range(4):
            pair = [w for w in waves if w["simd"] == simd]
            intervals = [(e[0] + e[2], e[0] + e[2] + 32)
                         for w in pair for e in w["instructions"] if "v_mfma_" in code[e[4]][0]]
            mfma_union += cycles(intervals)
            resident_simd_span += max(w["end"] for w in pair) - min(w["begin"] for w in pair)
        cta_rows.append({
            "se": key[0], "cu": key[1], "generation": key[2], "begin": begin, "end": end,
            "span_cycles": end - begin,
            "prologue_through_barrier6_cycles": boundaries[6] - begin,
            "main_physical_intervals7_326_cycles": boundaries[326] - boundaries[6],
            "drain_and_epilogue_cycles": end - boundaries[326],
            "steady_stage_mean_cycles": [statistics.mean(s) for s in stage_samples],
            "steady_stage_median_cycles": [statistics.median(s) for s in stage_samples],
            "steady_phase_mean_cycles": sum(statistics.mean(s) for s in stage_samples),
            "mfma_union_model_cycles": mfma_union,
            "resident_simd_span_cycles": resident_simd_span,
            "mfma_union_model_pct": 100 * mfma_union / resident_simd_span,
        })
    nwaves = len(cta_rows) * 8
    identity_bfi = sum(row[6] for row in code
                       if re.match(r"v_bfi_b32 (v\d+), [^,]+, \1, \1$", row[0]))
    equal_source_bfi = sum(row[6] for row in code
                          if re.match(r"v_bfi_b32 v\d+, [^,]+, (v\d+), \1$", row[0]))
    return {
        "ui_directory": str(ui), "complete_waves": nwaves, "sampled_ctas": len(cta_rows),
        "decoded_se_cus": sorted({(r["se"], r["cu"]) for r in cta_rows}),
        "instructions_per_wave": opcode_counts.total() / nwaves,
        "identity_v_bfi_per_wave": identity_bfi / nwaves,
        "equal_source_v_bfi_per_wave": equal_source_bfi / nwaves,
        "mean_cta_span_cycles": statistics.mean(r["span_cycles"] for r in cta_rows),
        "mean_prologue_cycles": statistics.mean(r["prologue_through_barrier6_cycles"] for r in cta_rows),
        "mean_main_cycles": statistics.mean(r["main_physical_intervals7_326_cycles"] for r in cta_rows),
        "mean_drain_epilogue_cycles": statistics.mean(r["drain_and_epilogue_cycles"] for r in cta_rows),
        "steady_stage_mean_cycles": [statistics.mean(r["steady_stage_mean_cycles"][s] for r in cta_rows) for s in range(8)],
        "steady_phase_mean_cycles": statistics.mean(r["steady_phase_mean_cycles"] for r in cta_rows),
        "mfma_union_model_pct": 100 * sum(r["mfma_union_model_cycles"] for r in cta_rows)
                                  / sum(r["resident_simd_span_cycles"] for r in cta_rows),
        "opcode_per_wave": {
            op: {"count": opcode_counts[op] / nwaves, "stall": opcode_stalls[op] / nwaves,
                 "latency": opcode_latencies[op] / nwaves, "idle": opcode_idle[op] / nwaves}
            for op in sorted(opcode_counts)
        },
        "ctas": cta_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    results = [analyze(path) for path in args.directories]
    print(json.dumps({
        "method": "Observed aligned barrier intervals for whole CTA; stages contain BOTH stagger groups. Per-wave opcode latency/stall are descriptive, not additive kernel-time attribution.",
        "mfma_model": "Union of [issue, issue+32) on each SIMD for gfx950 32x32x16 BF16; duration from ATT is issue time, not execution. Model, not hardware utilization counter.",
        "traces": results,
    }, indent=2))


if __name__ == "__main__":
    main()