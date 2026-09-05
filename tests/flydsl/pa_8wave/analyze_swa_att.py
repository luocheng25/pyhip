"""SWA ATT accounting with complete-wave checks and labelled interval models.

Do not sum stalls from different waves as kernel time. CTA grouping is used
only for 8-wave traces where all eight wave lifetimes and barriers are checked.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def interval_union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - begin for begin, end in merged)


def analyze(directory, waves_per_cta):
    code = json.loads((directory / "code.json").read_text())["code"]
    counts, stalls, spans, groups = Counter(), Counter(), [], defaultdict(list)
    for path in sorted(directory.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text())
        assert payload["num_insts"] == payload["num_stitched"], path
        wave = payload["wave"]
        wave_counts = Counter(code[e[4]][0].split()[0] for e in wave["instructions"])
        counts.update(wave_counts)
        for e in wave["instructions"]:
            stalls[code[e[4]][0].split()[0]] += e[2]
        spans.append(wave["end"] - wave["begin"])
        wave["barriers"] = [e[0] + e[3] for e in wave["instructions"] if code[e[4]][0] == "s_barrier"]
        se = int(path.name.split("_")[0][2:])
        groups[se, wave["cu"]].append(wave)
    assert spans, directory
    ctas = []
    if waves_per_cta == 8:
        for (se, cu), waves in groups.items():
            generations = defaultdict(list)
            for wave in waves:
                generations[wave["id"]].append(wave)
            for generation, members in generations.items():
                assert len(members) == 8, (directory, generation, len(members))
                assert len({(w["simd"], w["slot"]) for w in members}) == 8
                assert max(w["begin"] for w in members) - min(w["begin"] for w in members) < 512
                nbarriers = len(members[0]["barriers"])
                assert all(len(w["barriers"]) == nbarriers for w in members)
                assert max(max(w["barriers"][i] for w in members) - min(w["barriers"][i] for w in members)
                           for i in range(nbarriers)) <= 64
                boundaries = [max(w["barriers"][i] for w in members) for i in range(nbarriers)]
                begin, end = min(w["begin"] for w in members), max(w["end"] for w in members)
                ctas.append({"se": se, "cu": cu, "generation": generation,
                             "begin": begin, "end": end, "span_cycles": end - begin,
                             "barriers": nbarriers, "prologue_through_barrier6_cycles": boundaries[6] - begin,
                             "main_cycles": boundaries[-7] - boundaries[6],
                             "epilogue_cycles": end - boundaries[-7]})
    coverage = []
    for (se, cu), waves in groups.items():
        for simd in sorted({w["simd"] for w in waves}):
            members = [w for w in waves if w["simd"] == simd]
            span = max(w["end"] for w in members) - min(w["begin"] for w in members)
            resident = interval_union((w["begin"], w["end"]) for w in members)
            mfma = interval_union((e[0] + e[2], e[0] + e[2] + 32)
                                  for w in members for e in w["instructions"] if "v_mfma_" in code[e[4]][0])
            coverage.append({"se": se, "cu": cu, "simd": simd, "span_cycles": span,
                             "resident_union_cycles": resident, "mfma_32cycle_union_model": mfma})
    nwaves = len(spans)
    return {"directory": str(directory), "complete_waves": nwaves, "mean_wave_span_cycles": statistics.mean(spans),
            "instructions_per_wave": sum(counts.values()) / nwaves,
            "opcode_per_wave": {op: {"count": n / nwaves, "stall_cycles": stalls[op] / nwaves}
                                for op, n in sorted(counts.items())}, "cta_samples": ctas, "coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--waves-per-cta", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"method": "Complete waves only. Per-wave stalls overlap and are not additive. MFMA 32-cycle union is a model, not a hardware utilization counter.",
              "traces": [analyze(path, args.waves_per_cta) for path in args.directories]}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()