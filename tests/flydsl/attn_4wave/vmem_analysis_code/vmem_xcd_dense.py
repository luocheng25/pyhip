#!/usr/bin/env python3
"""Fixed-design physical-XCD comparison and dense upper CU-count measurements."""
import argparse
import re
import statistics
import subprocess
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from vmem_bandwidth_cache import gate, load_json, sha, write_csv, write_json
from vmem_contiguous_latency128 import NODES, POWER, PROBE_BYTES, SAMPLE, WORKER, executed_region, host_power_window
from vmem_hardware_tables import COUNTERS, POLICIES, csv_rows, stats
from vmem_pointer_chase_frequency import clock_metrics, cu
from vmem_read_cu_scaling import power_window

HERE = Path(__file__).resolve().parent
DENSE = (128, 144, 160, 176, 192, 208, 224, 232, 240, 244, 248, 252, 256)
LOCAL = (1, 2, 4, 8, 16, 24, 32)
SLICE = 512 << 20
DW = np.dtype([("clock", WORKER), ("index", "<i4"), ("rank", "<i4"), ("selected", "<u4"), ("pad", "<u4")])


def selection(mode, n, xcd, hw, target, rotate, block):
    se, c = (int(hw) >> 13) & 7, (int(hw) >> 8) & 15
    rank = se * 8 + c - (se & 1) if se < 4 and (se & 1) <= c < 8 + (se & 1) else -1
    if mode == 0:
        return (block if block < n else -1), rank
    if rank < 0 or not 0 <= xcd < 8:
        return -1, rank
    r, x = (rank - rotate) % 32, (xcd - target) % 8
    v = r if mode == 1 and x == 0 else r * 8 + x if mode == 2 else -1
    return (v if 0 <= v < n else -1), rank


def case(mode, n, p, b):
    return {"name": f"m{mode}_cu{n}_p{p}_{'bw' if b else 'lat'}", "mode": mode, "cus": n, "policy": p, "bandwidth": b}


def plans(out):
    out.mkdir(parents=True, exist_ok=False)
    # Natural-grid controls quantify the effect of full-grid software parking.
    primary = [case(m, n, p, b) for m, ns in ((0, DENSE + (1, 8, 32)), (1, LOCAL), (2, LOCAL))
               for n in ns for p in (0, 6) for b in (0, 1)]
    for tag, rows in (("first", primary), ("second", primary[::-1]), ("third", primary[44:] + primary[:44])):
        write_csv(out / f"{tag}.csv", rows)
    smoke = [case(m, n, p, b) for m, n, p in ((1, 8, 0), (2, 8, 0), (0, 144, 6)) for b in (0, 1)]
    write_csv(out / "smoke.csv", smoke)
    write_csv(out / "pmc.csv", [r for r in primary if r["bandwidth"]])
    write_json(out / "plan.json", {"status": "PREDECLARED", "dense_counts": DENSE, "local_counts": LOCAL,
        "natural_grid_parking_controls": [1, 8, 32], "policies": [0, 6], "ordinary_seeds": [95111, 95123, 95137],
        "ordinary_target_xcds": [0, 3, 6], "ordinary_local_rank_rotations": [0, 11, 22], "pmc_seed_target_rotate": [95171, 3, 11],
        "primary_dispatches_per_run": len(primary), "PMC_dispatches": len(primary) // 2,
        "scope": "N full-wave bulk CUs for B; one of these CUs replaced by four independent lane0 full-256MiB chains for L",
        "memory": "512MiB/bulk CU, 4waves D32 width16B/lane. Dense natural-grid, not hardware pinning. Same-XCD/spread launch256CTAs and select work from actual HW_ID/XCC.",
        "parking": "96KiB LDS=>max1CTA/CU. Inactive CTA keeps residency with one wave polling bounded walltime+s_sleep15 and 3waves at localbarrier; no data polling/interCTA sync.",
        "causality_limits": "XCD work placement is controlled, HBM physical page/channel placement is not. Parking may affect clocks/power; natural-grid 1/8/32 controls report this.",
        "retest": "At most one fixed three-run cohort for any invalid condition, replace BOTH L/B and both concentrated/spread paired modes; preserve failed raw. No fastest selection.",
        "plateau_rule": "No fitting target. Report total B*L/1024 and endpoint-relative deviations; report all upper points. Historical5025.311 is a reference only."})


def isa_check(path):
    text = path.read_text()
    bodies = re.findall(r"^(_Z\S*xcd_domain_kernel\S*):[^\n]*\n(.*?)^\.Lfunc_end\d+:", text, re.M | re.S)
    assert len(bodies) == 4, len(bodies)
    result = []
    for name, body in bodies:
        p = int(re.search(r"Lj(\d+)E", name)[1]); bw = "Lb1E" in name
        flags = {"nt", "sc1"} if p == 6 else set()
        for marker in ("DOMAIN_BULK_WARM", "DOMAIN_BULK_WORK"):
            region = executed_region(body, marker + "_BEGIN", marker + "_END")
            ops = [x for x in region.splitlines() if x.startswith("buffer_load")]
            assert len(ops) == 32 and all(x.startswith("buffer_load_dwordx4") for x in ops)
            regs = set()
            for x in ops:
                assert set(x.split()) & {"nt", "sc0", "sc1"} == flags
                m = re.match(r"buffer_load_dwordx4 v\[(\d+):(\d+)\],", x); assert m
                r = set(range(int(m[1]), int(m[2]) + 1)); assert len(r) == 4 and not regs & r; regs |= r
            assert not re.search(r"\b(?:global|flat|scratch)_(?:load|store)|\bbuffer_store|\bds_(?:read|write)|\bs_(?:load|store|buffer_load)", region)
        if not bw:
            for marker in ("DOMAIN_CHASE_WARM", "DOMAIN_CHASE_MEASURE"):
                lines = executed_region(body, marker + "_BEGIN", marker + "_END").splitlines()
                indices = [i for i, x in enumerate(lines) if x.startswith("buffer_load")]; assert len(indices) == 129
                for i in indices:
                    assert lines[i].startswith("buffer_load_dwordx4") and lines[i+1] == "s_waitcnt vmcnt(0)"
                    assert set(lines[i].split()) & {"nt", "sc0", "sc1"} == flags
        meta = re.search(r"\.amdhsa_kernel " + re.escape(name) + r"\s+(.*?)\.end_amdhsa_kernel", text, re.S)[1]
        resources = {k: int(v) for k, v in re.findall(r"\.amdhsa_(next_free_vgpr|next_free_sgpr|private_segment_fixed_size|group_segment_fixed_size)\s+(\d+)", meta)}
        assert resources["private_segment_fixed_size"] == resources["group_segment_fixed_size"] == 0
        result.append({"symbol": name, "policy": p, "bandwidth": bw, "resources": resources})
    return result


def audit(root):
    dev, configs = load_json(root / "device.json"), load_json(root / "cases.json")
    assert (dev["bdf"], dev["CUs"], dev["wall_khz"], dev["LDS"], dev["LDS_per_CU"], dev["slice_bytes"]) == ("0000:85:00.0", 256, 100000, 98304, 163840, SLICE)
    assert dev["sample_bytes"] == SAMPLE.itemsize == 112 and dev["worker_bytes"] == DW.itemsize == 144
    ranges = [(x, x + PROBE_BYTES + 131072) for x in dev["probe_allocations"]]
    ranges.append((dev["data_allocation"], dev["data_allocation"] + dev["data_bytes"] + 131072)); ranges.sort()
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    files = [root / "device.json", root / "cases.json"]; results, samples_out, workers_out, powers_out = [], [], [], []
    for c in configs:
        paths = [root / (c["name"] + s) for s in (".samples.bin", ".workers.bin", ".sinks.bin", ".power.bin")]; files += paths
        ss, dw = np.fromfile(paths[0], SAMPLE), np.fromfile(paths[1], DW)
        w, sinks, power = dw["clock"], np.fromfile(paths[2], "<u4"), np.fromfile(paths[3], POWER)
        assert len(w) == c["grid"] * 4 and len(sinks) == c["grid"] * 256 and len(ss) == (0 if c["bandwidth"] else 12)
        assert c["grid"] == (256 if c["mode"] else c["cus"]) and c["slice"] == SLICE and c["private"] == 0 and c["occupancy_max"] == 1 and c["checks"]
        assert np.array_equal(w["block"], np.arange(len(w)) // 4) and np.array_equal(w["wave"], np.arange(len(w)) % 4) and (w["active_lanes"] == 64).all()
        begin = np.array([cu(x["hw0"], x["xcc0"]) for x in w]); end = np.array([cu(x["hw1"], x["xcc1"]) for x in w])
        topology = bool(np.array_equal(begin, end) and len(np.unique(begin)) == c["grid"] and (begin.reshape(-1, 4) == begin[::4, None]).all())
        for i, x in enumerate(w):
            idx, rank = selection(c["mode"], c["cus"], int(x["xcc0"]), x["hw0"], c["target"], c["rotate"], int(x["block"]))
            assert dw[i]["index"] == idx and dw[i]["rank"] == rank and dw[i]["selected"] == (idx >= 0)
            assert x["role"] == (2 if c["bandwidth"] or idx != 0 else 1) if idx >= 0 else x["role"] == 0
            if rank < 0: topology = False
        active = dw["selected"].astype(bool); memory = np.flatnonzero(w["role"] == 2); selected_cus = np.unique(begin[active])
        topology &= bool(len(selected_cus) == c["cus"] and sorted(dw[::4]["index"][dw[::4]["selected"].astype(bool)].tolist()) == list(range(c["cus"])))
        assert len(memory) == (c["cus"] if c["bandwidth"] else c["cus"] - 1) * 4
        coverage = True; ns, cycles, freq = [], [], []; blank = None; lo = hi = None; probe_key = None
        if len(ss):
            probe_indices = np.flatnonzero(w["role"] == 1); assert len(probe_indices) == 4
            probe_key = int(begin[probe_indices[0]]); s = ss.reshape(4, 3); measured = s[:, 2]
            blank = statistics.median((s[:, 0]["chase"]["core1"] - s[:, 0]["chase"]["core0"]).tolist())
            lo, hi = int(measured["chase"]["wall0"].min()), int(measured["chase"]["wall1"].max())
            coverage = bool((w["export_wall"] > hi).all() and (s[:, :2]["export_wall"] < lo).all() and (measured["export_wall"] > hi).all())
            if len(memory): coverage &= bool((w[memory]["work0"] <= lo).all() and (w[memory]["work1"] >= hi).all())
            for wave in range(4):
                for phase in range(3):
                    q = s[wave, phase]; z = q["chase"]; clock = clock_metrics(z)
                    assert (q["wave"], q["phase"], z["start"], z["end"], z["steps"], z["active_lanes"]) == (wave, phase, 0, 0, NODES if phase else 0, 1)
                    topology &= clock["frequency_valid"] and clock["CU_begin"] == probe_key
                    scheduled = c["epoch"] + dev["warm_ticks"] + dev["guard_ticks"] + (dev["slot_ticks"] if phase == 2 else 0)
                    assert q["scheduled"] == scheduled and q["deadline"] == scheduled + dev["slot_ticks"]
                    coverage &= bool(scheduled <= z["wall0"] < z["wall1"] < q["deadline"])
                    if phase:
                        salt = c["salt"] ^ (wave * 0x13579BD); assert z["last"][0] == 0
                        for p in range(1, 4): assert int(z["last"][p]) == (((NODES-1)*(0x9E3779B9+p*2))^(salt+p*0x13579BD)) & 0xFFFFFFFF
                    cyc = (clock["raw_core_cycles"] - blank) / NODES if phase and clock["frequency_valid"] else None
                    latency = cyc * clock["wall_ns"] / clock["raw_core_cycles"] if cyc is not None else None
                    if phase == 2 and latency is not None: ns.append(latency); cycles.append(cyc); freq.append(clock["effective_GPU_MHz"])
                    samples_out.append({"case": c["name"], "wave": wave, "phase": phase, "latency_ns": latency, "cycles": cyc, **clock})
            p0, p1 = host_power_window(c, measured)
        warm_bytes = work_bytes = 0; mem_freq, passes, individual = [], [], []
        for i in memory:
            x = w[i]; idx = int(dw[i]["index"]); block, wave = int(x["block"]), int(x["wave"])
            dc, dr = int(x["core1"])-int(x["core0"]), int(x["work1"])-int(x["work0"])
            assert dc > 0 and dr > 0 and x["warm_chunks"] > 0 and x["work_chunks"] > 0
            mem_freq.append(dc*100/dr)
            last = ((int(x["warm_chunks"])+int(x["work_chunks"]))*65536-8192) % (SLICE//16)
            assert x["last_index"] == last and x["warm_last_index"] == (int(x["warm_chunks"])*65536-8192) % (SLICE//16)
            passes.append(int(x["work_chunks"])*65536*16//SLICE); coverage &= passes[-1] >= 1
            wb, b = int(x["warm_chunks"])*32*64*16*8, int(x["work_chunks"])*32*64*16*8
            warm_bytes += wb; work_bytes += b
            tid = np.arange(wave*64, (wave+1)*64, dtype=np.uint64); expected = np.full(64, 0x2468ACE0, dtype=np.uint64)
            slice_id = idx if c["bandwidth"] else idx-1
            for slot in range(32):
                for part in range(4):
                    address = slice_id*(SLICE//4)+(last+slot*256+tid)*4+part
                    expected += (address*np.uint64(0x9E3779B9)+np.uint64(c["salt"])) & np.uint64(0xFFFFFFFF)
            expected &= np.uint64(0xFFFFFFFF)
            assert np.array_equal(sinks[block*256+wave*64:block*256+(wave+1)*64], expected) and x["sum"] == expected[0]
            individual.append({"CU": int(begin[i]), "XCD": int(x["xcc0"]), "index": idx, "wave": wave, "payload_GBs": b/(dr*10)})
        for i, x in enumerate(w):
            workers_out.append({"case": c["name"], "block": int(x["block"]), "wave": int(x["wave"]), "index": int(dw[i]["index"]),
                "role": int(x["role"]), "CU_begin": int(begin[i]), "CU_end": int(end[i]), "XCD": int(x["xcc0"]),
                "entry": int(x["entry_wall"]), "work0": int(x["work0"]), "work1": int(x["work1"]), "export": int(x["export_wall"]),
                "warm_chunks": int(x["warm_chunks"]), "work_chunks": int(x["work_chunks"])})
        common = None
        if c["bandwidth"]:
            freq = mem_freq; lo, hi = int(w[memory]["work0"].max()), int(w[memory]["work1"].min())
            common = (hi-lo)*10; coverage &= common >= 290000000
            p0, p1 = power_window(c, lo, hi)
        # All parked/selected CTAs must arrive before work and remain until after it.
        residency = bool((w["entry_wall"] < lo).all() and (w["export_wall"] > hi).all())
        kept = power[(power["begin_ns"] >= p0) & (power["end_ns"] <= p1)]
        assert len(kept) >= 2 and (kept["microwatts"] > 0).all()
        for x in power:
            powers_out.append({"case": c["name"], "begin_ns": int(x["begin_ns"]), "end_ns": int(x["end_ns"]),
                "watts": int(x["microwatts"])/1e6, "inside_work": bool(x["begin_ns"] >= p0 and x["end_ns"] <= p1)})
        assert bool(c["topology_valid"]) == bool(topology)
        # Native coverage also checks chain windows and payload coverage; the
        # stricter Python common-wave/residency check is additional, not hidden.
        valid = bool(topology and c["coverage_valid"] and coverage and residency)
        envelope = (int(w[memory]["work1"].max())-int(w[memory]["work0"].min()))*10 if len(memory) else 0
        xcds = {str(x): len(set(begin[active & (w["xcc0"] == x)].tolist())) for x in range(8)}
        results.append({"config": c, "valid": valid, "topology_valid": bool(topology), "coverage_valid": bool(coverage), "residency_valid": residency,
            "actual_CUs": len(selected_cus), "resident_CUs": len(np.unique(begin)), "CU_keys": selected_cus.tolist(), "XCD_CU_counts": xcds,
            "probe_CU": probe_key, "blank_core_cycles": blank, "latency_ns": stats(ns) if valid and ns else None,
            "latency_core_cycles": stats(cycles) if valid and cycles else None, "frequency_MHz": stats(freq) if valid and freq else None,
            "power_W": stats(kept["microwatts"].astype(float)/1e6), "power_window_host_ns": [p0, p1],
            "work_payload_GBs": work_bytes/envelope if valid and envelope else None, "software_warm_bytes": warm_bytes,
            "software_work_bytes": work_bytes, "work_envelope_ns": envelope, "common_BW_ns": common,
            "minimum_buffer_passes": min(passes) if passes else None, "per_wave_payload": individual})
    return {"status": "PASS", "device": dev, "cases": results}, files, samples_out, workers_out, powers_out


def profile(capture, data):
    paths = list(capture.glob("pass_*/xcd_dense_counter_collection.csv")); assert len(paths) == 1
    cp = paths[0]; tp, ap = cp.parent/"xcd_dense_kernel_trace.csv", cp.parent/"xcd_dense_agent_info.csv"
    trace = sorted([r for r in csv_rows(tp) if "xcd_domain_kernel" in r["Kernel_Name"]], key=lambda r: int(r["Start_Timestamp"]))
    assert len(trace) == len(data["cases"])
    ident = lambda r: (r["Agent_Id"], int(r["Queue_Id"]), int(r["Dispatch_Id"]))
    counts, spans = defaultdict(dict), {}
    for r in csv_rows(cp):
        k, n, v = ident(r), r["Counter_Name"], Decimal(r["Counter_Value"])
        assert n in COUNTERS.values() and n not in counts[k] and v >= 0 and v == v.to_integral_value()
        counts[k][n] = int(v)*32
        span = int(r["Start_Timestamp"]), int(r["End_Timestamp"]); assert spans.setdefault(k, span) == span
    assert set(counts) == {ident(x) for x in trace}; out = []
    for t, r in zip(trace, data["cases"]):
        c, k = r["config"], ident(t); assert c["bandwidth"]
        assert f"xcd_domain_kernel<{c['policy']}u, true>" in t["Kernel_Name"]
        assert int(t["Grid_Size_X"]) == c["grid"]*256 and int(t["Workgroup_Size_X"]) == 256
        assert spans[k] == (int(t["Start_Timestamp"]), int(t["End_Timestamp"])) and set(counts[k]) == set(COUNTERS.values())
        ns = spans[k][1]-spans[k][0]; values = {kind+"_bytes": counts[k][name] for kind, name in COUNTERS.items()}
        assert ns > 0 and values["atomic_bytes"] == 0
        payload = r["software_warm_bytes"]+r["software_work_bytes"]
        out.append({"case": c["name"], "valid": r["valid"], "agent": k[0], "queue": k[1], "dispatch": k[2], "duration_ns": ns,
            **values, "DRAM_GBs": values["read_bytes"]/ns, "same_dispatch_payload_GBs": payload/ns,
            "DRAM_to_payload": values["read_bytes"]/payload, "scope": "whole_dispatch_with_warmup_and_parking_not_300ms_work"})
    agent = next(x for x in csv_rows(ap) if "Agent "+x["Logical_Node_Id"] == trace[0]["Agent_Id"]); loc = int(agent["Location_Id"])
    assert (int(agent["Domain"]), loc>>8, (loc>>3)&31, loc&7, int(agent["Cu_Count"]), int(agent["Num_Xcc"])) == (0, 0x85, 0, 0, 256, 8)
    return out, [cp, tp, ap]


def run(output, binary, isa, plan, seed, target, rotate, window=None, pmc=False):
    output.mkdir(parents=True, exist_ok=False); write_json(output/"isa.json", isa_check(isa))
    if window:
        w = load_json(window); assert w["idle_required"] and not w["live_pids"]
        assert all(w["metric"]["gpu_data"][0]["usage"][k]["value"] == 0 for k in ("gfx_activity", "umc_activity"))
    gate(output/"preflight.json", idle_required=window is None)
    power = list(Path("/sys/bus/pci/devices/0000:85:00.0/hwmon").glob("hwmon*/power1_input")); assert len(power) == 1
    command = [str(binary), str(output/"raw"), str(plan), str(seed), str(target), str(rotate), str(power[0].resolve())]
    if pmc: command = ["/opt/rocm/bin/rocprofv3", "-i", str(HERE/"vmem_xcd_dense_pmc.yaml"), "-d", str(output/"capture"), "--", *command]
    source_paths = [binary, isa, plan, Path(__file__), HERE/"vmem_xcd_dense.cpp", HERE/"vmem_xcd_dense_pmc.yaml", HERE/"vmem_matched_latency128.cpp"]
    sources = {str(p): sha(p) for p in source_paths}; write_json(output/"command.json", {"command": command, "source_sha256": sources, "window": str(window) if window else None})
    with (output/"run.log").open("x") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    data, files, samples, workers, powers = audit(output/"raw")
    if pmc:
        data["PMC"], extra = profile(output/"capture", data); files += extra
    if samples: write_csv(output/"samples.csv", samples)
    write_csv(output/"workers.csv", workers); write_csv(output/"power.csv", powers)
    assert sources == {p: sha(p) for p in sources}
    write_json(output/"summary.json", data)
    write_json(output/"verified.json", {"status": "PASS", "source_sha256": sources, "input_sha256": {str(p): sha(p) for p in files},
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.name in ("summary.json", "samples.csv", "workers.csv", "power.csv")}})
    print("XCD_DENSE_AUDIT", output.name, len(data["cases"]), "invalid", [r["config"]["name"] for r in data["cases"] if not r["valid"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plans", "run", "pmc")); parser.add_argument("--output", type=Path, required=True)
    for name in ("binary", "isa", "plan", "window"): parser.add_argument("--"+name, type=Path)
    parser.add_argument("--seed", type=int, default=95111); parser.add_argument("--target", type=int, default=0); parser.add_argument("--rotate", type=int, default=0)
    a = parser.parse_args()
    if a.action == "plans": plans(a.output.resolve())
    else: run(a.output.resolve(), a.binary.resolve(), a.isa.resolve(), a.plan.resolve(), a.seed, a.target, a.rotate, a.window.resolve() if a.window else None, a.action == "pmc")