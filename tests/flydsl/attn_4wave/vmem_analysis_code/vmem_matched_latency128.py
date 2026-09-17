# SPDX-License-Identifier: MIT
"""Matched full-line sequential/random pointerchase; oldtimedbinary reused.

Only thelinkorder changes in each pair. Four separatelyallocatedbuffers,
128Balignednodes, fullcoverage, timers, cache andbackground remain matched.
"""

import argparse
from collections import defaultdict
import copy
from decimal import Decimal
from pathlib import Path
import statistics
import subprocess

import numpy as np

from vmem_bandwidth_cache import COUNTERS, csv_rows, gate, load_json, sha, write_csv, write_json
from vmem_contiguous_latency128 import SAMPLE, WORKER, NODES, SPAN, PROBE_BYTES, host_power_window, probe_ranges_disjoint, isa_check
from vmem_hardware_tables import POWER
from vmem_pointer_chase import POLICIES
from vmem_pointer_chase_frequency import clock_metrics, cu, stats


HERE = Path(__file__).resolve().parent
METHOD = "samepreservedtimedcodeobject,same4buffersandbackground,one128Bnodeperline,fullsweeps;onlylinkorderdiffers"


def plans(output):
    output.mkdir(parents=True, exist_ok=False)
    pairs = [{"name": f"pair_p{p}_s{s}", "policy": p, "scope": s} for p in range(8) for s in range(3)]
    for tag, rows in (("first", pairs), ("second", list(reversed(pairs))), ("third", pairs[12:] + pairs[:12])):
        write_csv(output / f"{tag}.csv", rows)
    write_csv(output / "smoke.csv", [p for p in pairs if p["policy"] == 0])
    write_csv(output / "pmc.csv", [p for p in pairs if p["policy"] in (0, 6)])
    write_json(output / "plan.json", {"status": "PASS", "method": METHOD, "pairs": pairs,
        "runs": {"first": {"seed": 95011, "order": 0}, "second": {"seed": 95023, "order": 1}, "third": {"seed": 95037, "order": 0}},
        "waves": 4, "active_lanes_per_wave": 1, "node_spacing": 128, "request_bytes": 16, "nodes_each_sweep": NODES,
        "address_span_each_wave": SPAN + 16, "fullwarmups_perwave": 1, "measuredsweeps_perwave_run": 1,
        "measured_wave_sweeps_each_cell": 12, "background": "scope2same-cachecontinuousD32readforBOTHpatterns;scope1registerMFMA",
        "only_latency_changes": "48loadcellsL/frequencyL/powerL/FIFO;48storecellsandALLbandwidthsretained"})


def validate_orders(root):
    files, orders = [], []
    for wave in range(4):
        path = root / f"wave_{wave}.order.bin"; a = np.fromfile(path, dtype="<u4")
        assert len(a) == NODES and a[0] == 0 and np.array_equal(np.sort(a), np.arange(NODES, dtype=np.uint32))
        # Fullpermutationoflineindices: nonode/linecanrepeatinonesweep.
        assert int(a.max()) * 128 + 16 == SPAN + 16 and int(a.min()) == 0
        files.append(path); orders.append(a)
    assert len({sha(p) for p in files}) == 4
    return orders, files


def audit(root):
    dev, configs = load_json(root / "device.json"), load_json(root / "cases.json")
    assert dev["bdf"] == "0000:85:00.0" and dev["CUs"] == 256 and dev["wall_khz"] == 100000
    assert (dev["threads"], dev["wave_size"], dev["LDS"], dev["LDS_per_CU"], dev["args_bytes"]) == (256, 64, 98304, 163840, 80)
    assert (dev["nodes"], dev["node_spacing"], dev["load_bytes"], dev["probe_bytes_per_wave"], dev["address_span_per_sweep"]) == (NODES, 128, 16, PROBE_BYTES, SPAN + 16)
    assert dev["sample_bytes"] == SAMPLE.itemsize == 112 and dev["worker_bytes"] == WORKER.itemsize == 128
    assert probe_ranges_disjoint(dev["probe_allocations"], dev["background_allocation"], dev["background_bytes"])
    assert all((p + 65536) % 128 == 0 for p in dev["probe_allocations"])
    orders, files = validate_orders(root); files += [root / "device.json", root / "cases.json"]
    results, sample_rows, power_rows, worker_rows = [], [], [], []
    by_pair = defaultdict(list)
    for c in configs:
        paths = [root / (c["name"] + ext) for ext in (".samples.bin", ".workers.bin", ".sinks.bin", ".power.bin")]; files += paths
        s = np.fromfile(paths[0], dtype=SAMPLE).reshape(4, 3); w = np.fromfile(paths[1], dtype=WORKER)
        sink, power = np.fromfile(paths[2], dtype="<u4"), np.fromfile(paths[3], dtype=POWER)
        assert c["blocks"] == (1 if c["scope"] == 0 else 256) and len(w) == c["blocks"] * 4 and len(sink) == c["blocks"] * 256
        assert c["checks"] and c["private"] == 0 and c["occupancy_max"] == 1
        assert np.array_equal(w["block"], np.arange(len(w)) // 4) and np.array_equal(w["wave"], np.arange(len(w)) % 4)
        assert (w["role"] == (w["block"] != 0) * c["scope"]).all() and (w["active_lanes"] == 64).all()
        k0 = np.array([cu(x["hw0"], x["xcc0"]) for x in w]); k1 = np.array([cu(x["hw1"], x["xcc1"]) for x in w])
        topology = bool(np.array_equal(k0, k1) and len(np.unique(k0)) == c["blocks"] and (k0.reshape(-1, 4) == k0[::4, None]).all())
        assert np.array_equal(np.sort(((w["hw0"] >> 4) & 3).reshape(-1, 4), axis=1), np.broadcast_to(np.arange(4), (c["blocks"], 4)))
        assert (w["xcc0"] < 8).all() and (w["xcc1"] < 8).all()
        assert np.array_equal(s["wave"], np.broadcast_to(np.arange(4)[:, None], (4, 3)))
        assert np.array_equal(s["phase"], np.broadcast_to(np.arange(3), (4, 3)))
        assert (s["chase"]["active_lanes"] == 1).all() and (s["chase"]["start"] == 0).all() and (s["chase"]["end"] == 0).all()
        blank = statistics.median((s[:, 0]["chase"]["core1"] - s[:, 0]["chase"]["core0"]).tolist())
        lo, hi = int(s[:, 2]["chase"]["wall0"].min()), int(s[:, 2]["chase"]["wall1"].max())
        overlap = bool((w["export_wall"] > hi).all() and (s[:, :2]["export_wall"] < lo).all() and (s[:, 2]["export_wall"] > hi).all())
        if c["scope"]: overlap &= bool((w[4:]["work0"] <= lo).all() and (w[4:]["work1"] >= hi).all())
        ns_values, cycles_values, freq = [], [], []
        for wave in range(4):
            tail = int(orders[wave][-1]) if c["random"] else NODES - 1
            assert c["tail_nodes"][wave] == tail
            for phase in range(3):
                sample = s[wave, phase]; z = sample["chase"]; clock = clock_metrics(z)
                assert z["steps"] == (NODES if phase else 0)
                if not clock["frequency_valid"] or clock["CU_begin"] != k0[0]: topology = False
                scheduled = c["epoch"] + dev["warm_ticks"] + dev["guard_ticks"] + (dev["slot_ticks"] if phase == 2 else 0)
                assert sample["scheduled"] == scheduled and sample["deadline"] == scheduled + dev["slot_ticks"]
                timely = bool(z["wall0"] >= scheduled and z["wall1"] < sample["deadline"]); overlap &= timely
                if phase:
                    salt = c["salt"] ^ (wave * 0x13579BD)
                    assert z["last"][0] == 0
                    for p in range(1, 4): assert int(z["last"][p]) == ((tail * (0x9E3779B9 + p * 2)) ^ (salt + p * 0x13579BD)) & 0xFFFFFFFF
                ns = cycles = None
                if phase and clock["frequency_valid"]:
                    cycles = (clock["raw_core_cycles"] - blank) / NODES
                    ns = cycles * clock["wall_ns"] / clock["raw_core_cycles"]
                    if phase == 2: ns_values.append(ns); cycles_values.append(cycles); freq.append(clock["effective_GPU_MHz"])
                sample_rows.append({"case": c["name"], "pattern": "random" if c["random"] else "sequential", "wave": wave, "phase": phase,
                    "measured": phase == 2, "latency_ns": ns, "core_cycles_per_hop": cycles, "full_sweep_loads": NODES if phase else 0,
                    "unique128Bnodes": NODES if phase else 0, "within_slot": timely, "GPU_wall0": int(z["wall0"]), "GPU_wall1": int(z["wall1"]), **clock})
        bg_freq, warm_bytes, work_bytes = [], 0, 0
        for i, x in enumerate(w[4:], 4):
            assert x["work1"] > x["work0"] and x["warm_chunks"] > 0 and x["work_chunks"] > 0
            dc, ticks = int(x["core1"]) - int(x["core0"]), int(x["work1"]) - int(x["work0"])
            if k0[i] == k1[i] and dc > 0: bg_freq.append(dc * 100 / ticks)
            block, wave = int(x["block"]), int(x["wave"])
            want = np.full(64, 32, dtype=np.uint64)
            if c["scope"] == 2:
                index = ((int(x["warm_chunks"]) + int(x["work_chunks"])) * 65536 - 8192) % ((64 << 20) // 16)
                assert x["last_index"] == index and x["warm_last_index"] == (int(x["warm_chunks"]) * 65536 - 8192) % ((64 << 20) // 16)
                assert int(x["work_chunks"]) * 65536 * 16 >= 64 << 20
                warm_bytes += int(x["warm_chunks"]) * 65536 * 4; work_bytes += int(x["work_chunks"]) * 65536 * 4
                want[:] = 0x2468ACE0; tids = np.arange(wave * 64, (wave + 1) * 64, dtype=np.uint64)
                for slot in range(32):
                    for p in range(4):
                        index_word = (block - 1) * ((64 << 20) // 4) + (index + slot * 256 + tids) * 4 + p
                        want += (index_word * np.uint64(0x9E3779B9) + np.uint64(c["salt"])) & np.uint64(0xFFFFFFFF)
                want &= np.uint64(0xFFFFFFFF)
            assert np.array_equal(sink[block * 256 + wave * 64:block * 256 + (wave + 1) * 64], want) and x["sum"] == want[0]
            worker_rows.append({"case": c["name"], "block": block, "wave": wave, "role": int(x["role"]), "CU_begin": int(k0[i]), "CU_end": int(k1[i]),
                "warm_chunks": int(x["warm_chunks"]), "work_chunks": int(x["work_chunks"]), "work0": int(x["work0"]), "work1": int(x["work1"]),
                "raw_core_cycles": dc, "realtime_ticks": ticks, "export_wall": int(x["export_wall"])})
        assert bool(c["topology_valid"]) == topology and bool(c["overlap_valid"]) == overlap
        valid = bool(topology and overlap)
        p0, p1 = host_power_window(c, s[:, 2]); kept = power[(power["begin_ns"] >= p0) & (power["end_ns"] <= p1)]
        assert len(kept) >= 2 and (kept["microwatts"] > 0).all()
        for p in power: power_rows.append({"case": c["name"], "begin_ns": int(p["begin_ns"]), "end_ns": int(p["end_ns"]), "watts": int(p["microwatts"]) / 1e6,
            "inside_all_four_measured_chains": bool(p["begin_ns"] >= p0 and p["end_ns"] <= p1)})
        row = {"config": c, "valid": valid, "latency_ns": stats(ns_values) if valid else None, "latency_core_cycles": stats(cycles_values) if valid else None,
            "latency_frequency_MHz": stats(freq) if valid else None, "latency_power_W": stats(kept["microwatts"].astype(float) / 1e6) if valid else None,
            "power_window_host_ns": [p0, p1], "blank_core_cycles": blank, "probe_CU": int(k0[0]),
            "background_frequency_MHz": stats(bg_freq) if valid and bg_freq else None, "software_warm_background_bytes": warm_bytes,
            "software_work_background_bytes": work_bytes, "measured_full_sweeps": 4, "measured_dependent_loads": 4 * NODES,
            "measured_span_per_wave": SPAN + 16, "unique_nodes_per_measured_wave": NODES, "same_line_repeats_per_sweep": 0}
        results.append(row); by_pair[c["pair"]].append(row)
    pairs = []
    for pair, values in by_pair.items():
        assert len(values) == 2 and {r["config"]["random"] for r in values} == {0, 1}
        a, b = sorted(values, key=lambda x: x["config"]["random"])
        assert all(a["config"][k] == b["config"][k] for k in ("policy", "scope", "salt", "blocks", "VGPR", "private", "occupancy_max"))
        valid = a["valid"] and b["valid"]
        pairs.append({"pair": pair, "policy": a["config"]["policy"], "scope": a["config"]["scope"], "valid": valid,
            "sequential_L_ns": a["latency_ns"]["mean"] if valid else None, "random_L_ns": b["latency_ns"]["mean"] if valid else None,
            "random_vs_sequential_percent": (b["latency_ns"]["mean"] / a["latency_ns"]["mean"] - 1) * 100 if valid else None,
            "same_physical_probe_CU_across_dispatches": a["probe_CU"] == b["probe_CU"],
            "same_GPU_allocation_addresses": True, "same_machine_code": True, "background_pattern": "continuous" if a["config"]["scope"] == 2 else "MFMA" if a["config"]["scope"] else "none"})
    return {"status": "PASS", "device": dev, "cases": results, "pairs": pairs, "method": METHOD}, files, sample_rows, worker_rows, power_rows


def profile(capture, result):
    paths = list(capture.glob("pass_*/matched128_counter_collection.csv")); assert len(paths) == 1
    cp = paths[0]; tp, ap = cp.parent / "matched128_kernel_trace.csv", cp.parent / "matched128_agent_info.csv"
    trace = sorted([t for t in csv_rows(tp) if "contiguous_latency128_kernel" in t["Kernel_Name"]], key=lambda t: int(t["Start_Timestamp"]))
    assert len(trace) == len(result["cases"])
    ident = lambda r: (r["Agent_Id"], int(r["Queue_Id"]), int(r["Dispatch_Id"]))
    counts, spans = defaultdict(dict), {}
    for row in csv_rows(cp):
        key, name, n = ident(row), row["Counter_Name"], Decimal(row["Counter_Value"])
        assert name in COUNTERS.values() and name not in counts[key] and n >= 0 and n == n.to_integral_value()
        counts[key][name] = int(n) * 32; period = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
        assert spans.setdefault(key, period) == period
    assert set(counts) == {ident(t) for t in trace}
    profiles = []
    for t, row in zip(trace, result["cases"]):
        c, key = row["config"], ident(t)
        assert f"contiguous_latency128_kernel<{c['policy']}u>" in t["Kernel_Name"]
        assert int(t["Grid_Size_X"]) == c["blocks"] * 256 and int(t["Workgroup_Size_X"]) == 256
        assert spans[key] == (int(t["Start_Timestamp"]), int(t["End_Timestamp"])) and set(counts[key]) == set(COUNTERS.values())
        ns = spans[key][1] - spans[key][0]; assert ns > 0
        values = {k + "_bytes": counts[key][v] for k, v in COUNTERS.items()}; assert values["atomic_bytes"] == 0
        profiles.append({"case": c["name"], "random": c["random"], "scope": c["scope"], "valid": row["valid"], "agent": key[0], "queue": key[1], "dispatch": key[2], "duration_ns": ns,
            **values, "DRAM_read_GBs": values["read_bytes"] / ns,
            "DRAM_bytes_per_dependent_load": values["read_bytes"] / (NODES * 4 * 2) if c["scope"] != 2 else None,
            "note": "whole dispatchinclwarmupandguards;notbandwidthcolumn,notexacthit-rate;scope2includesbackground"})
    agent = next(a for a in csv_rows(ap) if "Agent " + a["Logical_Node_Id"] == trace[0]["Agent_Id"]); loc = int(agent["Location_Id"])
    assert (int(agent["Domain"]), loc >> 8, (loc >> 3) & 31, loc & 7, int(agent["Cu_Count"]), int(agent["Num_Xcc"])) == (0, 0x85, 0, 0, 256, 8)
    return profiles, [cp, tp, ap]


def run(output, binary, codeobject, isa, plan, seed, order, window=None, pmc=False):
    output.mkdir(parents=True, exist_ok=False); write_json(output / "isa.json", isa_check(isa))
    if window:
        initial = load_json(window); assert initial["idle_required"] and not initial["live_pids"]
        assert all(initial["metric"]["gpu_data"][0]["usage"][k]["value"] == 0 for k in ("gfx_activity", "umc_activity"))
    gate(output / "preflight.json", idle_required=window is None)
    powers = list(Path("/sys/bus/pci/devices/0000:85:00.0/hwmon").glob("hwmon*/power1_input")); assert len(powers) == 1
    cmd = [str(binary), str(output / "raw"), str(plan), str(seed), str(order), str(powers[0].resolve()), str(codeobject)]
    if pmc: cmd = ["/opt/rocm/bin/rocprofv3", "-i", str(HERE / "vmem_matched_latency128_pmc.yaml"), "-d", str(output / "capture"), "--", *cmd]
    sourcefiles = (binary, codeobject, isa, plan, Path(__file__).resolve(), HERE / "vmem_matched_latency128.cpp", HERE / "vmem_matched_latency128_pmc.yaml", HERE / "vmem_contiguous_latency128.py", HERE / "vmem_contiguous_latency128.cpp")
    hashes = {str(p): sha(p) for p in sourcefiles}
    write_json(output / "command.json", {"command": cmd, "source_sha256": hashes, "window": str(window) if window else None})
    with (output / "run.log").open("x") as f: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    data, files, samples, workers, power = audit(output / "raw")
    if pmc: data["PMC"], more = profile(output / "capture", data); files += more
    write_csv(output / "samples.csv", samples)
    if workers: write_csv(output / "workers.csv", workers)
    write_csv(output / "power.csv", power)
    assert hashes == {p: sha(p) for p in hashes}
    write_json(output / "summary.json", data)
    write_json(output / "verified.json", {"status": "PASS", "source_sha256": hashes, "input_sha256": {str(p): sha(p) for p in files},
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.name in ("samples.csv", "workers.csv", "power.csv", "summary.json")}})
    print("MATCHED128_AUDIT", output.name, len(data["cases"]), "invalid", sum(not c["valid"] for c in data["cases"]), flush=True)


def combine(output, old, runs, pmcs=()):
    output.mkdir(parents=True, exist_ok=False)
    original, om = load_json(old / "summary.json"), load_json(old / "verified.json")
    inputs = {}
    for p, h in om["input_sha256"].items(): assert sha(p) == h; inputs[p] = h
    for name, h in om["output_sha256"].items(): assert sha(old / name) == h; inputs[str(old / name)] = h
    grouped, paired = defaultdict(list), defaultdict(list)
    for root in [*runs, *pmcs]:
        m = load_json(root / "verified.json")
        for p, h in m["input_sha256"].items(): assert sha(p) == h; inputs[p] = h
        for name, h in m["output_sha256"].items(): assert sha(root / name) == h; inputs[str(root / name)] = h
        if root in runs:
            data = load_json(root / "summary.json")
            for row in data["cases"]: grouped[row["config"]["policy"], row["config"]["scope"], row["config"]["random"]].append(row)
            for row in data["pairs"]: paired[row["policy"], row["scope"]].append({"run": str(root), **row})
    assert len(grouped) == 48 and len(paired) == 24
    updated = copy.deepcopy(original); replacements = []
    for row in updated["rows"]:
        if row["store"]: continue
        values = grouped[POLICIES.index(row["cache"]), row["scope_id"], int(row["pattern"] == "random")]; assert len(values) == 3
        valid = all(v["valid"] for v in values)
        previous = {k: copy.deepcopy(row[k]) for k in ("latency_ns", "latency_core_cycles", "latency_frequency_MHz", "latency_power_W", "equivalent_FIFO_per_CU", "per_run_latency_ns", "latency_kind")}
        for k in ("latency_ns", "latency_core_cycles", "latency_frequency_MHz", "latency_power_W"):
            xs = [v[k] for v in values]
            n = sum(x["count"] for x in xs) if valid else 0
            row[k] = {"count": n, "mean": sum(x["mean"] * x["count"] for x in xs) / n, "min": min(x["min"] for x in xs), "max": max(x["max"] for x in xs)} if valid else None
        row["valid"] = row["valid"] and valid
        row["equivalent_FIFO_per_CU"] = row["payload_GBs"] / row["memory_CUs_bandwidth"] * row["latency_ns"]["mean"] / 1024 if row["valid"] else None
        row["per_run_latency_ns"] = [v["latency_ns"] for v in values]
        if "latency_retest" in row: previous["prior_retest"] = row["latency_retest"]
        row["latency_kind"] = "matchedfull128Bnodes4independentwaveleaders:" + row["pattern"]
        row["latency_retest"] = {"source": str(output / "matched_comparison.json"), "method": METHOD,
            "per_run_valid": [v["valid"] for v in values], "prior_latency_fields": previous, "background_in_latency": "continuousread" if row["scope_id"] == 2 else "MFMA" if row["scope_id"] else "none",
            "bandwidth_source_unchanged": str(old / "summary.json"), "FIFO_proxy_caveat": "Lbackgroundheldfixedforordercomparison;retainedBWusesroworiginalcontiguous/randompattern;notonepopulationphysicaloccupancy"}
        replacements.append({"pattern": row["pattern"], "cache": row["cache"], "scope_id": row["scope_id"], "runs": values, "prior_fields": previous})
    updated["scope_definition"] = {"single_CU": "loadLfourwaveleaders4disjointlinebuffers;storeLoldsinglelane;BWold4waves",
        "single_CU_plus_255_MFMA": "sameprobetopologyperrowwith255registerMFMAbackground",
        "all_CUs": "BWall256;loadLoneCUfourwaveleaders+255continuousreadbackgroundforbothorders;samecachewithinpair"}
    if "continuous_load_latency_retest" in updated:
        updated["historical_contiguous_retest"] = updated.pop("continuous_load_latency_retest")
    updated["matched_load_latency_retest"] = {"source": str(output / "matched_comparison.json"), "updated_cells": 48, "retained_store_cells": 48,
        "all_bandwidth_fields_unchanged": True, "measured_sweeps_per_cell": 12, "nodes_per_sweep": NODES, "node_spacing_bytes": 128, "load_bytes": 16, "address_span_per_sweep": SPAN + 16,
        "method": METHOD, "background_pattern_is_not_changed_with_probe_order": True}
    updated["limits"] = ["Onlylinkorderchangeswithinpair;nohardwarefrequency/CUbindingforced,CUidentityandlocalfrequencyrecorded.",
        "Eachsweepvisitsall128Bnodesonce;warmupisaseparatefullsweep,nocachehitguaranteeorclaimbasedonaddressalone.",
        "Lperwaveaverageunderfourparallelchains,notindividualrequestlatencydistribution.",
        "Alloriginalbandwidths/powersB/storecellsretained;randomLbackgroundisnowcontinuousforfaircomparison,notrandomBWworkload.",
        "FIFOremainsexplicitserialL/bulkBWproxy,notphysicaldepth/occupancy;no5TBscaledL."]
    comparisons = []
    for (p, scope), values in paired.items():
        assert len(values) == 3
        seq = next(r for r in updated["rows"] if not r["store"] and r["pattern"] == "contiguous" and r["cache"] == POLICIES[p] and r["scope_id"] == scope)
        rand = next(r for r in updated["rows"] if not r["store"] and r["pattern"] == "random" and r["cache"] == POLICIES[p] and r["scope_id"] == scope)
        comparisons.append({"cache": POLICIES[p], "scope_id": scope, "valid": seq["valid"] and rand["valid"], "sequential_ns": seq["latency_ns"]["mean"],
            "random_ns": rand["latency_ns"]["mean"], "random_vs_sequential_pct": (rand["latency_ns"]["mean"] / seq["latency_ns"]["mean"] - 1) * 100, "paired_runs": values})
    write_json(output / "summary.json", updated)
    write_json(output / "matched_comparison.json", {"status": "PASS", "method": METHOD, "comparisons": comparisons, "replacements": replacements,
        "PMC": [{"source": str(p), "rows": load_json(p / "summary.json")["PMC"]} for p in pmcs]})
    flat = [{k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in row.items()
             if k not in ("PMC", "per_run_valid", "per_run_latency_ns", "per_run_payload_GBs", "latency_retest")} for row in updated["rows"]]
    write_csv(output / "hardware_tables.csv", flat)
    write_json(output / "verified.json", {"status": "PASS", "input_sha256": inputs, "source_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("action", choices=("plans", "run", "pmc", "combine")); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path); p.add_argument("--codeobject", type=Path); p.add_argument("--isa", type=Path); p.add_argument("--plan", type=Path)
    p.add_argument("--seed", type=int, default=95011); p.add_argument("--order", type=int, default=0); p.add_argument("--window", type=Path)
    p.add_argument("--old", type=Path); p.add_argument("--runs", nargs="+", type=Path); p.add_argument("--pmcs", nargs="*", type=Path, default=[])
    a = p.parse_args()
    if a.action == "plans": plans(a.output.resolve())
    elif a.action == "combine": combine(a.output.resolve(), a.old.resolve(), [r.resolve() for r in a.runs], [r.resolve() for r in a.pmcs])
    else: run(a.output.resolve(), a.binary.resolve(), a.codeobject.resolve(), a.isa.resolve(), a.plan.resolve(), a.seed, a.order, a.window.resolve() if a.window else None, a.action == "pmc")