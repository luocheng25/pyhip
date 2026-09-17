# SPDX-License-Identifier: MIT
"""Select whole fixed-repeat pairs and merge matched latency without editing runs.

Bad raw records remain preserved. A failed pair is replaced as an entire
three-run sequential/random cohort, never by choosing faster waves or runs.
"""

import argparse
from collections import defaultdict
import copy
from pathlib import Path
import statistics

from vmem_bandwidth_cache import load_json, sha, write_csv, write_json
from vmem_pointer_chase import POLICIES


def select_cohorts(root):
    inputs, originals, repeats = {}, {}, {}
    invalid = set(); invalid_cases = []
    for prefix, dest in (("main_recovered", originals), ("retest", repeats)):
        for tag in ("first", "second", "third"):
            path = root / f"{prefix}_{tag}"; manifest = load_json(path / "verified.json")
            for p, h in manifest["input_sha256"].items(): assert sha(p) == h; inputs[p] = h
            for name, h in manifest["output_sha256"].items(): assert sha(path / name) == h; inputs[str(path / name)] = h
            data = load_json(path / "summary.json")
            window = load_json(Path(load_json(path / "command.json")["window"]))
            assert window["idle_required"] and not window["live_pids"]
            assert all(window["metric"]["gpu_data"][0]["usage"][k]["value"] == 0 for k in ("gfx_activity", "umc_activity"))
            dest[tag] = data
            if prefix == "main_recovered":
                for c in data["cases"]:
                    if not c["valid"]:
                        invalid.add((c["config"]["policy"], c["config"]["scope"]))
                        invalid_cases.append({"source": str(path), "case": c["config"]["name"], "reason": "CU migration; original invalid record preserved"})
    policy_path = root / "retest_policy.json"; policy = load_json(policy_path)
    assert invalid == {tuple(x) for x in policy["pairs"]} == {(0, 0), (6, 0)}
    assert policy["maximum_retest_rounds"] == 1
    inputs[str(policy_path)] = sha(policy_path)
    selected, ledger = defaultdict(list), []
    for p in range(8):
        for scope in range(3):
            use_repeat = (p, scope) in invalid
            cohort = repeats if use_repeat else originals
            for tag in ("first", "second", "third"):
                data = cohort[tag]; path = root / f"{'retest' if use_repeat else 'main_recovered'}_{tag}"
                cases = [r for r in data["cases"] if (r["config"]["policy"], r["config"]["scope"]) == (p, scope)]
                assert len(cases) == 2 and {r["config"]["random"] for r in cases} == {0, 1}
                for c in cases:
                    selected[p, scope, c["config"]["random"]].append({"source": str(path), **c})
                ledger.append({"policy": p, "scope": scope, "run_tag": tag, "source": str(path), "whole_pair_retest": use_repeat,
                    "valid": all(c["valid"] for c in cases), "selection_reason": "predeclared full-pair repeat after migration" if use_repeat else "original predeclared run"})
    assert len(selected) == 48 and all(len(v) == 3 for v in selected.values())
    return selected, ledger, invalid_cases, inputs


def finalize(root, old, output, pmcs):
    output.mkdir(parents=True, exist_ok=False)
    selected, ledger, invalid, inputs = select_cohorts(root)
    base = load_json(old / "summary.json"); old_manifest = load_json(old / "verified.json")
    for p, h in old_manifest["input_sha256"].items(): assert sha(p) == h; inputs[p] = h
    for name, h in old_manifest["output_sha256"].items(): assert sha(old / name) == h; inputs[str(old / name)] = h
    updated = copy.deepcopy(base); details = []
    for row in updated["rows"]:
        if row["store"]: continue
        values = selected[POLICIES.index(row["cache"]), row["scope_id"], int(row["pattern"] == "random")]
        valid = all(v["valid"] for v in values)
        prior_fields = {k: copy.deepcopy(row[k]) for k in ("latency_ns", "latency_core_cycles", "latency_frequency_MHz", "latency_power_W",
                                                        "equivalent_FIFO_per_CU", "per_run_latency_ns", "latency_kind")}
        if "latency_retest" in row: prior_fields["latency_retest"] = copy.deepcopy(row["latency_retest"])
        for key in ("latency_ns", "latency_core_cycles", "latency_frequency_MHz", "latency_power_W"):
            xs = [r[key] for r in values]; n = sum(x["count"] for x in xs) if valid else 0
            row[key] = {"count": n, "mean": sum(x["mean"] * x["count"] for x in xs) / n,
                        "min": min(x["min"] for x in xs), "max": max(x["max"] for x in xs)} if valid else None
        row["valid"] = row["valid"] and valid
        row["per_run_latency_ns"] = [v["latency_ns"] for v in values]
        row["per_run_valid"] = [v["valid"] for v in values] + row["per_run_valid"][3:]
        row["equivalent_FIFO_per_CU"] = row["payload_GBs"] / row["memory_CUs_bandwidth"] * row["latency_ns"]["mean"] / 1024 if row["valid"] else None
        row["latency_kind"] = "matched128Bline-nodes;fourindependentwaveleaders;" + row["pattern"]
        row["latency_retest"] = {"source": str(output / "matched_comparison.json"), "cohort_sources": [v["source"] for v in values], "prior_fields": prior_fields,
            "node_spacing_bytes": 128, "load_bytes": 16, "nodes_each_sweep": 2097153, "address_span_each_wave": 268435472,
            "probe_waves": 4, "active_lanes_per_wave": 1, "shared_measured_codeobject": True, "same_pair_buffer_addresses": True,
            "background_for_both_patterns": "continuous same-cache read" if row["scope_id"] == 2 else "MFMA" if row["scope_id"] == 1 else "none",
            "old_bandwidth_source": str(old / "summary.json"),
            "FIFO_limit": "retainedBWrowpattern and matchedLfixedbackground are not identical populations; explicit requestbudgetproxy only"}
        details.append({"pattern": row["pattern"], "cache": row["cache"], "scope_id": row["scope_id"], "selected_runs": values})
    comparisons = []
    for p in range(8):
        for scope in range(3):
            seq = selected[p, scope, 0]; rnd = selected[p, scope, 1]
            valid = all(r["valid"] for r in [*seq, *rnd])
            per_run = []
            for a, b in zip(seq, rnd):
                assert a["source"] == b["source"]
                per_run.append({"source": a["source"], "sequential_ns": a["latency_ns"]["mean"] if valid else None,
                    "random_ns": b["latency_ns"]["mean"] if valid else None,
                    "random_vs_sequential_pct": (b["latency_ns"]["mean"] / a["latency_ns"]["mean"] - 1) * 100 if valid else None,
                    "same_physical_probe_CU": a["probe_CU"] == b["probe_CU"], "sequential_CU": a["probe_CU"], "random_CU": b["probe_CU"]})
            s = statistics.mean(v["latency_ns"]["mean"] for v in seq) if valid else None
            q = statistics.mean(v["latency_ns"]["mean"] for v in rnd) if valid else None
            comparisons.append({"cache": POLICIES[p], "scope_id": scope, "valid": valid, "sequential_ns": s, "random_ns": q,
                "random_vs_sequential_pct": (q / s - 1) * 100 if valid else None, "paired_runs": per_run})
    profiles = []
    for path in pmcs:
        m = load_json(path / "verified.json")
        for p, h in m["input_sha256"].items(): assert sha(p) == h; inputs[p] = h
        for name, h in m["output_sha256"].items(): assert sha(path / name) == h; inputs[str(path / name)] = h
        profiles.append({"source": str(path), "rows": load_json(path / "summary.json")["PMC"]})
    updated["scope_definition"] = {"single_CU": "loadLfour1-laneleaderswith4disjoint128Bnodebuffers;storeLoriginalsinglelane;BWoriginal4wave",
        "single_CU_plus_255_MFMA": "sameproberesourcesplus255registerMFMAbackgroundCUs",
        "all_CUs": "BWoriginal256CU;bothloadLpatternsoneCU4leaders+255continuoussamecachebulkreadCUs"}
    if "continuous_load_latency_retest" in updated: updated["historical_contiguous_retest"] = updated.pop("continuous_load_latency_retest")
    updated["matched_load_latency_retest"] = {"source": str(output / "matched_comparison.json"), "cells_replaced": 48, "store_cells_retained": 48,
        "bandwidths_retained": True, "measured_fullsweeps_per_cell": 12, "nodes_per_sweep": 2097153, "node_spacing": 128, "load_bytes": 16,
        "address_span": 268435472, "random_line_repeats_in_one_full_sweep": 0, "selection_ledger": "selection.json"}
    updated["power_definition"] = "loadLwholeGPUsensor20ms conservativeGPUepochhostbracketinsideall4chains;BW/storeLretainoriginalwindows"
    updated["limits"] = ["Onlyprobechainorderchangeswithineachpair;samepreservedmachinecode/resources/addresses/cache/background/schedule.",
        "ActualprobeCUmaychangebetweenlaunches;recordednotmasked/locked;differentstaticaddressesarenotacausalclaim.",
        "One128Bnodeperline,fullFisherYatescycleeachwave;nointrasweeprepeatednode/line,butwarmupandmeasuredsweepsreusewholebuffer.",
        "RandomLbackgroundcontinuousformatching,randomBWoriginalrandomperlane;FIFOisexplicitcrossprotocolproxy,notphysicaloccupancy/depth.",
        "Repeatedpaircohortspredeclaredonmigrationonly;allfailedandquarantinedcaptureartifactspreservedandexcludedexplicitly."]
    write_json(output / "summary.json", updated)
    write_json(output / "matched_comparison.json", {"status": "PASS", "comparisons": comparisons, "details": details, "PMC": profiles})
    write_json(output / "selection.json", {"status": "PASS", "ledger": ledger, "original_invalid_cases": invalid,
        "excluded_attempts": ["main_first (original sharedgateprovenance overwritten; preserved)", "main_second (quarantinedpartial)", "main_window.json (quarantined)"],
        "incident": str(root / "execution_incident.json"), "retest_policy": str(root / "retest_policy.json")})
    flat = [{k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in row.items()
             if k not in ("PMC", "per_run_valid", "per_run_latency_ns", "per_run_payload_GBs", "latency_retest")} for row in updated["rows"]]
    write_csv(output / "hardware_tables.csv", flat)
    write_json(output / "verified.json", {"status": "PASS", "input_sha256": inputs, "source_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, required=True); p.add_argument("--old", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--pmcs", nargs="*", type=Path, default=[])
    a = p.parse_args(); finalize(a.root.resolve(), a.old.resolve(), a.output.resolve(), [p.resolve() for p in a.pmcs])