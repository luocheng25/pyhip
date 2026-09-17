# SPDX-License-Identifier: MIT
"""Pool predeclared XCD/dense-CU cohorts without selecting fast observations."""
import argparse
from collections import defaultdict
import statistics
from pathlib import Path

from vmem_bandwidth_cache import load_json, sha, write_csv, write_json
from vmem_hardware_tables import POLICIES, stats
from vmem_xcd_dense import DENSE, LOCAL, case


def pool(rows, field):
    values = [r[field] for r in rows]
    if any(v is None for v in values):
        return None
    count = sum(v["count"] for v in values)
    return {"count": count, "mean": sum(v["count"]*v["mean"] for v in values)/count,
            "min": min(v["min"] for v in values), "max": max(v["max"] for v in values)}


def ratios(numerator, denominator):
    return 100*(numerator/denominator-1)


def combine(root, output, retests=()):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}; all_rows, replacements = defaultdict(list), defaultdict(list)

    def verified(path):
        m = load_json(path/"verified.json")
        for p, h in m["input_sha256"].items():
            assert sha(p) == h, p; inputs[p] = h
        for p, h in m["output_sha256"].items():
            assert sha(path/p) == h; inputs[str(path/p)] = h
        return load_json(path/"summary.json")

    for tag in ("first", "second", "third"):
        path = root/("main_"+tag)
        for r in verified(path)["cases"]:
            all_rows[r["config"]["name"]].append({"source": str(path), **r})
    invalid = set()
    for rows in all_rows.values():
        for r in rows:
            if not r["valid"]:
                c = r["config"]
                for mode in ((1, 2) if c["mode"] else (0,)):
                    invalid.add((mode, c["cus"], c["policy"]))
    for path in retests:
        for r in verified(path)["cases"]:
            replacements[r["config"]["name"]].append({"source": str(path), **r})
    used = dict(all_rows); selected = []
    for m, n, p in sorted(invalid):
        for bw in (0, 1):
            name = case(m, n, p, bw)["name"]
            if name in replacements:
                assert len(replacements[name]) == 3
                used[name] = replacements[name]
                selected.append({"case": name, "original_sources": [r["source"] for r in all_rows[name]],
                                 "replacement_sources": [r["source"] for r in replacements[name]]})
    assert set(replacements) <= {case(m, n, p, b)["name"] for m, n, p in invalid for b in (0, 1)}
    pmc = {r["case"]: r for r in verified(root/"pmc_first")["PMC"]}
    points = []
    for mode, counts in ((0, DENSE+(1, 8, 32)), (1, LOCAL), (2, LOCAL)):
        for n in counts:
            for policy in (0, 6):
                lat = used[case(mode, n, policy, 0)["name"]]; bw = used[case(mode, n, policy, 1)["name"]]
                assert len(lat) == len(bw) == 3
                assert len({(r["config"]["target"], r["config"]["rotate"]) for r in lat}) == 3
                physical = pmc[case(mode, n, policy, 1)["name"]]
                valid = all(r["valid"] for r in [*lat, *bw, physical])
                l, b = pool(lat, "latency_ns"), statistics.mean(r["work_payload_GBs"] for r in bw) if valid else None
                total = b*l["mean"]/1024 if valid else None
                per_run = []
                for lrow, brow in zip(lat, bw):
                    assert lrow["source"] == brow["source"]
                    cfg = lrow["config"]
                    if mode:
                        assert lrow["CU_keys"] == brow["CU_keys"]
                        assert lrow["probe_CU"] in brow["CU_keys"]
                    per_run.append({"source": lrow["source"], "target_XCD": cfg["target"], "rotate": cfg["rotate"],
                        "valid": lrow["valid"] and brow["valid"], "latency_ns": lrow["latency_ns"]["mean"] if lrow["valid"] else None,
                        "payload_GBs": brow["work_payload_GBs"], "FIFO_equivalent_total": brow["work_payload_GBs"]*lrow["latency_ns"]["mean"]/1024 if lrow["valid"] and brow["valid"] else None,
                        "probe_CU": lrow["probe_CU"], "XCD_CU_counts": brow["XCD_CU_counts"],
                        "B_frequency_MHz": brow["frequency_MHz"], "L_frequency_MHz": lrow["frequency_MHz"],
                        "B_power_W": brow["power_W"], "L_power_W": lrow["power_W"]})
                per_xcd = []
                for r in bw:
                    by_xcd = defaultdict(list)
                    for w in r["per_wave_payload"]: by_xcd[w["XCD"]].append(w["payload_GBs"])
                    per_xcd.append({"source": r["source"], "XCD_CU_counts": r["XCD_CU_counts"],
                        "own_wave_window_payload_GBs": {str(x): sum(v) for x, v in sorted(by_xcd.items())},
                        "note": "Individual wave windows, not global-envelope throughput; no local latency measured for non-probe XCDs."})
                points.append({"mode": mode, "placement": ("natural", "one_XCD", "spread_XCD")[mode], "cus": n,
                    "policy": policy, "cache": POLICIES[policy], "valid": valid, "latency_ns": l,
                    "latency_core_cycles": pool(lat, "latency_core_cycles"), "L_frequency_MHz": pool(lat, "frequency_MHz"),
                    "B_frequency_MHz": pool(bw, "frequency_MHz"), "L_power_W": pool(lat, "power_W"), "B_power_W": pool(bw, "power_W"),
                    "payload_GBs": b, "payload_per_CU_GBs": b/n if valid else None, "DRAM_GBs": physical["DRAM_GBs"] if physical["valid"] else None,
                    "FIFO_equivalent_per_CU": total/n if valid else None, "FIFO_equivalent_total": total,
                    "per_run_total": stats([r["FIFO_equivalent_total"] for r in per_run]) if valid else None,
                    "per_run": per_run, "L_sources": [r["source"] for r in lat], "B_sources": [r["source"] for r in bw], "per_XCD": per_xcd, "PMC": physical})
    assert len(points) == 60
    old_path = root.parent/"cu_scaling_20260916/result_first/summary.json"; inputs[str(old_path)] = sha(old_path)
    old = load_json(old_path); old_full = next(r for r in old["points"] if r["cus"] == 256 and r["policy"] == 0)
    historical = old_full["FIFO_equivalent_per_CU"]*256
    table = {(r["mode"], r["cus"], r["policy"]): r for r in points}
    dense, placements, parking = [], [], []
    for p in (0, 6):
        full = table[0, 256, p]
        for n in DENSE:
            r = table[0, n, p]
            if r["valid"] and full["valid"]:
                dense.append({"cus": n, "policy": p, "total": r["FIFO_equivalent_total"],
                    "total_vs_fresh_256_percent": ratios(r["FIFO_equivalent_total"], full["FIFO_equivalent_total"]),
                    "total_vs_historical_default_percent": ratios(r["FIFO_equivalent_total"], historical),
                    "payload_vs_fresh_256_percent": ratios(r["payload_GBs"], full["payload_GBs"]),
                    "latency_vs_fresh_256_percent": ratios(r["latency_ns"]["mean"], full["latency_ns"]["mean"])})
        for n in LOCAL:
            concentrated, spread = table[1, n, p], table[2, n, p]
            if concentrated["valid"] and spread["valid"]:
                paired = []
                for a, b in zip(concentrated["per_run"], spread["per_run"]):
                    assert a["source"] == b["source"] and a["probe_CU"] == b["probe_CU"]
                    paired.append({"source": a["source"], "probe_CU": a["probe_CU"],
                        "concentrated_vs_spread_payload_percent": ratios(a["payload_GBs"], b["payload_GBs"]),
                        "concentrated_vs_spread_latency_percent": ratios(a["latency_ns"], b["latency_ns"])})
                placements.append({"cus": n, "policy": p, "concentrated_vs_spread_payload_percent": ratios(concentrated["payload_GBs"], spread["payload_GBs"]),
                    "concentrated_vs_spread_latency_percent": ratios(concentrated["latency_ns"]["mean"], spread["latency_ns"]["mean"]), "per_run": paired})
        for n in (1, 8, 32):
            natural, spread = table[0, n, p], table[2, n, p]
            if natural["valid"] and spread["valid"]:
                parking.append({"cus": n, "policy": p, "spread_parked_vs_natural_payload_percent": ratios(spread["payload_GBs"], natural["payload_GBs"]),
                    "spread_parked_vs_natural_latency_percent": ratios(spread["latency_ns"]["mean"], natural["latency_ns"]["mean"]),
                    "limitation": "Natural placement is observed, not same physical-CU binding; measures combined parking/placement change."})
    write_json(output/"summary.json", {"status": "PASS", "points": points, "historical_default_total": historical,
        "selection": selected, "invalid_original_conditions": [list(x) for x in sorted(invalid)],
        "limits": ["FIFOeq is B(full-wave bulk) * L(fourleader dependency chains)/1024, not measured physical occupancy/capacity.",
                   "PMC is one separate whole-dispatch capture per BW configuration, including200mswarmup/guard and launch/parking tails.",
                   "Dense natural counts do not control physical CU identity; XCD concentrated/spread selection is verified from raw hardware IDs.",
                   "Single-XCD domain throughput does not identify a particular FIFO; physical memory channel placement is uncontrolled.",
                   "L is measured only on selected index0 CU; applying it to all CU rates is an explicit proxy, not measured per-domain latencies."]})
    write_json(output/"comparison.json", {"historical_default_total_reference": historical, "dense": dense, "placement": placements, "parking_controls": parking})
    simple = [{k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in r.items()
               if k not in ("per_run", "per_XCD", "L_sources", "B_sources", "PMC")} for r in points]
    write_csv(output/"all_points.csv", simple)
    write_csv(output/"dense_CU.csv", [r for r in simple if r["mode"] == 0 and r["cus"] in DENSE])
    write_csv(output/"XCD_placement.csv", [r for r in simple if r["mode"] != 0])
    write_json(output/"verified.json", {"status": "PASS", "input_sha256": inputs, "source_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print("XCD_DENSE_POOL", len(points), "invalid", [(r["mode"], r["cus"], r["policy"]) for r in points if not r["valid"]], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p.add_argument("--retests", nargs="*", type=Path, default=[]); a = p.parse_args()
    combine(a.root.resolve(), a.output.resolve(), [r.resolve() for r in a.retests])