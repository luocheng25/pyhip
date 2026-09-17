# SPDX-License-Identifier: MIT
"""Two compact hardware tables from fresh, versioned measurements.

16B/lane payload units; serialload latency and store+wait completion are distinct.
Power is a sampled whole-GPU sensor, not per-CU power. No hardware controls.
"""

import argparse
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import re
import statistics
import subprocess

import numpy as np

from vmem_bandwidth_cache import COUNTERS, csv_rows, gate, load_json, sha, write_csv, write_json
from vmem_pointer_chase import POLICIES
from vmem_pointer_chase_frequency import SAMPLE, clock_metrics, cu, executed_region, stats


HERE = Path(__file__).resolve().parent
WORKER = np.dtype([(k, "<u8") for k in ("entry_wall", "warm0", "warm1", "work0", "work1", "core0", "core1", "warm_chunks", "work_chunks", "store_wall")]
    + [(k, "<u4") for k in ("hw0", "hw1", "xcc0", "xcc1", "block", "wave", "role", "active_lanes", "last_index", "warm_last_index", "sum", "slice")])
POWER = np.dtype([("begin_ns", "<u8"), ("end_ns", "<u8"), ("microwatts", "<u8")])
FIELDS = ("name", "random", "store", "policy", "scope", "bandwidth", "blank")
SCOPES = ("single_CU", "single_CU_plus_255_MFMA", "all_CUs")
CHUNK_VECTORS = 256 * 32 * 8


def permutation(x, mask, salt):
    x = (np.asarray(x, dtype=np.uint64) ^ np.uint64(salt & mask)) & np.uint64(mask)
    x = ((x ^ (x >> np.uint64(13))) * np.uint64(0x7FEB352D)) & np.uint64(mask)
    x = ((x ^ (x >> np.uint64(9))) * np.uint64(0x846CA68B)) & np.uint64(mask)
    return (x ^ (x >> np.uint64(16))) & np.uint64(mask)


def vector_at(x, mask, salt, random):
    return permutation(x, mask, salt) if random else np.asarray(x, dtype=np.uint64) & np.uint64(mask)


def case(random, store, policy, scope, bandwidth, blank=0):
    name = f"{'random' if random else 'contig'}_{'store' if store else 'load'}_p{policy}_s{scope}_{'bw' if bandwidth else 'blank' if blank else 'lat'}"
    return dict(zip(FIELDS, (name, random, store, policy, scope, bandwidth, blank)))


def plans(output):
    output.mkdir(parents=True, exist_ok=False)
    measurements = [case(r, s, p, scope, bw) for r in range(2) for s in range(2) for p in range(8)
                    for scope in range(3) for bw in (0, 1)]
    blanks = [case(r, s, 0, scope, 0, 1) for r in range(2) for s in range(2) for scope in range(3)]
    for tag, sequence in (("first", measurements), ("second", list(reversed(measurements))),
                          ("third", measurements[len(measurements)//2:] + measurements[:len(measurements)//2])):
        write_csv(output / f"{tag}.csv", [*blanks, *sequence], FIELDS)
    # One complete policy acrossbothpatterns/scopes exercisesallruntimebranches.
    smoke = [case(r, s, 6, scope, bw, blank) for r in range(2) for s in range(2) for scope in range(3)
             for bw, blank in ((0, 1), (0, 0), (1, 0))]
    write_csv(output / "smoke.csv", smoke, FIELDS)
    # Allbandwidthcells getactualDRAMPMC;profilelatencyalso separatelycollected.
    for tag in ("first", "second", "third"):
        write_csv(output / f"pmc_{tag}.csv", csv_rows(output / f"{tag}.csv"), FIELDS)
    write_json(output / "plan.json", {"status": "PASS", "width_per_lane": 16, "wave_size": 64,
        "rows_per_table": 16, "scopes": SCOPES, "main_cells": 96, "dispatches_per_run_including_blanks": 204,
        "latency_chain_hops": 4097, "latency_chains_per_run": 29, "warmup_chains": 8,
        "runs": {"first": 95011, "second": 95023, "third": 95037},
        "contiguous": "adjacent16Bvectors;loadserialnextpointerstoredatthenext16Bnode",
        "random": "invertiblexor-shift/oddmultiplypermutationof16Bvectors,independentper-laneuncoalescedaddresses;loadserialnextstoredinpreviousnode",
        "power": "whole-GPU power1_input microwatts sampled20ms withCPUtimerfd;no per-CUattribution orhardwarecontrols",
        "bandwidth": "D32independentloads;store8*32opsperdrain;chunkworkpayloadandseparatewhole-dispatchDRAMPMC",
        "FIFO": "Bpayload/CU * Lserial /1024B;modelonly,notphysicaldepth;storeLusesstore+waitnotloadL"})


def isa_check(path):
    text = path.read_text()
    bodies = re.findall(r"^(_Z\S*hardware_table_kernel\S*):[^\n]*\n(.*?)^\.Lfunc_end\d+:", text, re.M | re.S)
    assert len(bodies) == 16, len(bodies)
    result = []
    for symbol, body in bodies:
        store, policy = map(int, re.search(r"ILb([01])ELj([0-7])EE", symbol).groups())
        flags = set(POLICIES[policy].split()) - {"default"}
        for marker in ("TABLE_STREAM_WARM", "TABLE_STREAM_WORK"):
            region = executed_region(body, marker + "_BEGIN", marker + "_END")
            operations = [x for x in region.splitlines() if x.startswith("buffer_")]
            assert len(operations) == 32 and all(x.startswith("buffer_store_dwordx4" if store else "buffer_load_dwordx4") for x in operations)
            assert all(set(x.split()) & {"sc0", "sc1", "nt"} == flags for x in operations)
            assert not re.search(r"\b(?:global|flat|scratch)_(?:load|store)|\bds_(?:read|write)|\bs_(?:load|store|buffer_load)", region)
            occupied = set()
            if not store:
                for x in operations:
                    m = re.match(r"buffer_load_dwordx4 v\[(\d+):(\d+)\],", x); assert m
                    regs = set(range(int(m[1]), int(m[2]) + 1))
                    assert len(regs) == 4 and not regs & occupied; occupied |= regs
        for marker in ("TABLE_MFMA_WARM", "TABLE_MFMA_WORK"):
            region = executed_region(body, marker + "_BEGIN", marker + "_END")
            assert region.count("v_mfma_f32_16x16x128_f8f6f4") == 128
            assert not re.search(r"\b(?:buffer|flat|global|scratch|ds)_(?:load|store|read|write)|\bs_(?:load|store|buffer_load)", region)
        region = executed_region(body, "TABLE_SERIAL_BEGIN", "TABLE_SERIAL_END")
        assert not re.search(r"\bs_memtime|\bs_memrealtime|\b(?:global|flat|scratch)_(?:load|store)", region)
        operations = [x for x in region.splitlines() if x.startswith("buffer_")]
        assert len(operations) == 129 and all(set(x.split()) & {"sc0", "sc1", "nt"} == flags for x in operations)
        assert all(x.startswith("buffer_store_dwordx4" if store else "buffer_load_dwordx4") for x in operations)
        metadata = re.search(r"\.amdhsa_kernel " + re.escape(symbol) + r"\s+(.*?)\.end_amdhsa_kernel", text, re.S)[1]
        resources = {k: int(v) for k, v in re.findall(r"\.amdhsa_(next_free_vgpr|next_free_sgpr|private_segment_fixed_size|group_segment_fixed_size)\s+(\d+)", metadata)}
        assert resources["private_segment_fixed_size"] == resources["group_segment_fixed_size"] == 0
        result.append({"symbol": symbol, "store": store, "policy": policy, "bulk_static_ops": 32, "serial_static_ops": 129, "resources": resources})
    assert set(re.findall(r"\.(?:sgpr_spill_count|vgpr_spill_count|private_segment_fixed_size):\s*(\d+)", text)) == {"0"}
    return result


def power_stats(raw, cfg):
    # Conservatively trimwarmup/launch andexport. Hostmonotonic sampledwindow
    # is approximate, not a claim of nanosecond-aligned sensorintegration.
    begin, end = cfg["host_begin_ns"] + 215_000_000, cfg["host_end_ns"] - 10_000_000
    keep = raw[(raw["begin_ns"] >= begin) & (raw["end_ns"] <= end)]
    assert len(keep) >= 2 and (keep["microwatts"] > 0).all(), "missing in-work power samples"
    return {**stats(keep["microwatts"].astype(float) / 1e6), "scope": "whole GPU socket",
            "host_window_begin_ns": begin, "host_window_end_ns": end, "sensor_unit": "microwatts",
            "method": "arithmetic mean of in-window power1_input samples; sensorrefresh/averaginglatency not deconvolved"}


def audit(root, warmups):
    dev, configs = load_json(root / "device.json"), load_json(root / "cases.json")
    assert dev["bdf"] == "0000:85:00.0" and dev["CUs"] == 256 and dev["wall_khz"] == 100000
    assert dev["LDS"] == 98304 and dev["LDS_per_CU"] == 163840
    assert dev["sample_bytes"] == SAMPLE.itemsize == 104 and dev["worker_bytes"] == WORKER.itemsize == 128
    assert dev["probe_allocation"] + dev["probe_bytes"] + 131072 <= dev["data_allocation"] or dev["data_allocation"] + dev["data_allocation_bytes"] <= dev["probe_allocation"]
    files = [root / "device.json", root / "cases.json"]
    blanks = defaultdict(list)
    for c in configs:
        if c["blank"] and c["topology_valid"] and c["overlap_valid"]:
            a = np.fromfile(root / (c["name"] + ".samples.bin"), dtype=SAMPLE)[warmups:]
            blanks[c["random"], c["store"], c["scope"]].extend((a["chase"]["core1"] - a["chase"]["core0"]).tolist())
    baselines = {k: statistics.median(v) for k, v in blanks.items()}
    results, sample_rows, worker_rows, power_rows = [], [], [], []
    for c in configs:
        paths = [root / (c["name"] + ext) for ext in (".samples.bin", ".workers.bin", ".sinks.bin", ".power.bin")]
        files += paths
        s, w = np.fromfile(paths[0], dtype=SAMPLE), np.fromfile(paths[1], dtype=WORKER)
        sinks, powers = np.fromfile(paths[2], dtype="<u4"), np.fromfile(paths[3], dtype=POWER)
        assert len(s) == c["chains"] and len(w) == c["blocks"] * 4 and len(sinks) == c["blocks"] * 256
        assert c["private"] == 0 and c["occupancy_max"] == 1 and c["checks"] and c["store_validation_errors"] == 0
        assert np.array_equal(w["block"], np.arange(len(w)) // 4) and np.array_equal(w["wave"], np.arange(len(w)) % 4)
        assert (w["active_lanes"] == 64).all() and (w["xcc0"] < 8).all() and (w["xcc1"] < 8).all()
        start_keys = np.array([cu(x["hw0"], x["xcc0"]) for x in w]); end_keys = np.array([cu(x["hw1"], x["xcc1"]) for x in w])
        topology = bool(np.array_equal(start_keys, end_keys) and len(np.unique(start_keys)) == c["blocks"]
                        and (start_keys.reshape(-1, 4) == start_keys[::4, None]).all())
        assert np.array_equal(np.sort(((w["hw0"] >> 4) & 3).reshape(-1, 4), axis=1), np.broadcast_to(np.arange(4), (c["blocks"], 4)))
        target = start_keys[0]; overlap = True
        if len(s):
            overlap = bool((w["store_wall"] > s["chase"]["wall1"].max()).all())
            if c["scope"]:
                overlap &= bool((w[4:]["work0"] <= s["chase"]["wall0"].min()).all() and (w[4:]["work1"] >= s["chase"]["wall1"].max()).all())
        latency, cycles, frequency = [], [], []
        hops = 0 if c["blank"] else 4097
        baseline = baselines.get((c["random"], c["store"], c["scope"]))
        if len(s):
            assert baseline is not None
            mask = dev["probe_bytes"] // 16 - 1
            ranks = np.arange(len(s) + 1, dtype=np.uint64) * hops
            expected = vector_at(ranks, mask, c["salt"], c["random"]) * 16
            assert np.array_equal(s["chase"]["start"], expected[:-1]) and np.array_equal(s["chase"]["end"], expected[1:])
            for i, sample in enumerate(s):
                z = sample["chase"]; clock = clock_metrics(z)
                assert z["steps"] == hops and z["active_lanes"] == 1 and sample["iteration"] == i
                if not clock["frequency_valid"] or clock["CU_begin"] != target:
                    topology = False
                scheduled = c["epoch"] + dev["warm_ticks"] + dev["guard_ticks"] + i * dev["slot_ticks"]
                assert sample["deadline"] == scheduled + dev["slot_ticks"]
                timely = bool(z["wall0"] >= scheduled and z["wall1"] < sample["deadline"] and not sample["late"])
                overlap &= timely
                if not c["store"] and hops:
                    assert z["last"][0] == z["end"]
                    node = int(vector_at((i + 1) * hops - 1, mask, c["salt"], c["random"]))
                    for part in range(1, 4):
                        assert int(z["last"][part]) == ((node * (0x9E3779B9 + 2 * part)) ^ (c["salt"] + part * 0x13579BD)) & 0xFFFFFFFF
                ns = cyc = None
                if clock["frequency_valid"] and hops:
                    cyc = (clock["raw_core_cycles"] - baseline) / hops
                    ns = cyc * clock["wall_ns"] / clock["raw_core_cycles"]
                    if i >= warmups:
                        latency.append(ns); cycles.append(cyc); frequency.append(clock["effective_GPU_MHz"])
                sample_rows.append({"case": c["name"], "iteration": i + 1, "warmup": i < warmups,
                    "within_slot": timely, "latency_ns": ns, "core_cycles_per_op": cyc, **clock})
        warm_bytes = work_bytes = 0; memory_clocks, background_clocks = [], []
        memory_workers = []
        for i, wr in enumerate(w):
            block, wave = int(wr["block"]), int(wr["wave"])
            probe = not c["bandwidth"] and block == 0
            compute = c["scope"] == 1 and block != 0
            assert wr["role"] == (0 if probe else 2 if compute else 1)
            if probe:
                continue
            assert wr["warm_chunks"] > 0 and wr["work_chunks"] > 0 and wr["work1"] > wr["work0"]
            dc, ticks = int(wr["core1"]) - int(wr["core0"]), int(wr["work1"]) - int(wr["work0"])
            mhz = dc * 100 / ticks if start_keys[i] == end_keys[i] and dc > 0 else None
            if mhz is not None:
                (background_clocks if compute else memory_clocks).append(mhz)
            if compute:
                assert (sinks[block * 256 + wave * 64:block * 256 + (wave + 1) * 64] == 32).all()
            else:
                memory_workers.append(wr)
                mask = c["slice"] // 16 - 1
                expected_last = ((int(wr["warm_chunks"]) + int(wr["work_chunks"])) * CHUNK_VECTORS - 8192) & mask
                assert wr["last_index"] == expected_last
                assert wr["warm_last_index"] == (int(wr["warm_chunks"]) * CHUNK_VECTORS - 8192) & mask
                assert int(wr["work_chunks"]) * CHUNK_VECTORS * 16 >= c["slice"]
                warm_bytes += int(wr["warm_chunks"]) * CHUNK_VECTORS * 4
                work_bytes += int(wr["work_chunks"]) * CHUNK_VECTORS * 4
                want = np.full(64, 0x2468ACE0, dtype=np.uint64)
                if not c["store"]:
                    tids = np.arange(wave * 64, (wave + 1) * 64, dtype=np.uint64)
                    slice_id = block if c["bandwidth"] else block - 1
                    for j in range(32):
                        at = vector_at(expected_last + j * 256 + tids, mask, c["salt"], c["random"])
                        for part in range(4):
                            word = slice_id * (c["slice"] // 4) + at * 4 + part
                            want += (word * np.uint64(0x9E3779B9) + np.uint64(c["salt"])) & np.uint64(0xFFFFFFFF)
                    want &= np.uint64(0xFFFFFFFF)
                actual = sinks[block * 256 + wave * 64:block * 256 + (wave + 1) * 64]
                assert np.array_equal(actual, want) and wr["sum"] == want[0]
            worker_rows.append({"case": c["name"], "block": block, "wave": wave, "role": int(wr["role"]),
                "CU_begin": int(start_keys[i]), "CU_end": int(end_keys[i]), "effective_MHz": mhz,
                "warm_chunks": int(wr["warm_chunks"]), "work_chunks": int(wr["work_chunks"]),
                "work_begin": int(wr["work0"]), "work_end": int(wr["work1"]), "last_index": int(wr["last_index"])})
        assert bool(c["topology_valid"]) == topology and bool(c["overlap_valid"]) == overlap
        valid = bool(topology and overlap)
        if c["bandwidth"]:
            frequency = memory_clocks
        bw = None
        if memory_workers:
            span = (max(int(x["work1"]) for x in memory_workers) - min(int(x["work0"]) for x in memory_workers)) * 10
            bw = work_bytes / span
        for p in powers:
            power_rows.append({"case": c["name"], "begin_ns": int(p["begin_ns"]), "end_ns": int(p["end_ns"]), "watts": float(p["microwatts"]) / 1e6})
        results.append({"config": c, "valid": valid, "blank_core_cycles": baseline,
            "latency_ns": stats(latency) if valid and latency else None, "latency_core_cycles": stats(cycles) if valid and cycles else None,
            "effective_MHz": stats(frequency) if valid and frequency else None,
            "MFMA_effective_MHz": stats(background_clocks) if valid and background_clocks else None,
            "power_W": power_stats(powers, c), "work_payload_GBs": bw if valid else None,
            "software_warm_payload_bytes": warm_bytes, "software_work_payload_bytes": work_bytes,
            "ordinary_dispatch_payload_GBs": (warm_bytes + work_bytes) / (c["event_ms"] * 1e6) if valid and memory_workers else None,
            "latency_kind": "store+wait completion includingaddressgeneration" if c["store"] else "dependentload-to-nextaddress",
            "actual_CUs": len(np.unique(start_keys)), "full_window_covered": overlap,
            "min_work_full_slice_passes": min(int(x["work_chunks"]) * CHUNK_VECTORS * 16 // c["slice"] for x in memory_workers) if memory_workers else None})
    return {"status": "PASS", "device": dev, "cases": results}, files, sample_rows, worker_rows, power_rows


def profile(capture, checked):
    paths = list(capture.glob("pass_*/hardware_tables_counter_collection.csv")); assert len(paths) == 1
    cp = paths[0]; tp, ap = cp.parent / "hardware_tables_kernel_trace.csv", cp.parent / "hardware_tables_agent_info.csv"
    trace = sorted([r for r in csv_rows(tp) if "hardware_table_kernel" in r["Kernel_Name"]], key=lambda t: int(t["Start_Timestamp"]))
    assert len(trace) == len(checked["cases"])
    ident = lambda row: (row["Agent_Id"], int(row["Queue_Id"]), int(row["Dispatch_Id"]))
    counters, periods = defaultdict(dict), {}
    for r in csv_rows(cp):
        k, name, value = ident(r), r["Counter_Name"], Decimal(r["Counter_Value"])
        assert name in COUNTERS.values() and name not in counters[k] and value >= 0 and value == value.to_integral_value()
        counters[k][name] = int(value) * 32
        times = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        assert periods.setdefault(k, times) == times
    assert set(counters) == {ident(t) for t in trace}
    profiles = []
    for t, row in zip(trace, checked["cases"]):
        c, k = row["config"], ident(t)
        expected = f"hardware_table_kernel<{'true' if c['store'] else 'false'}, {c['policy']}u>"
        assert expected in t["Kernel_Name"] and set(counters[k]) == set(COUNTERS.values())
        assert periods[k] == (int(t["Start_Timestamp"]), int(t["End_Timestamp"]))
        assert int(t["Grid_Size_X"]) == c["blocks"] * 256 and int(t["Workgroup_Size_X"]) == 256
        ns = periods[k][1] - periods[k][0]; assert ns > 0
        values = {kind + "_bytes": counters[k][name] for kind, name in COUNTERS.items()}
        assert values["atomic_bytes"] == 0
        profiles.append({"case": c["name"], "valid": row["valid"], "duration_ns": ns, "agent": k[0], "queue": k[1], "dispatch": k[2],
            **values, "DRAM_read_GBs": values["read_bytes"] / ns, "DRAM_write_GBs": values["write_bytes"] / ns,
            "payload_GBs_same_dispatch": (row["software_warm_payload_bytes"] + row["software_work_payload_bytes"]) / ns,
            "scope": "whole dispatch inclwarmup/guards; notperchainwindow; noordinary/PMCdenominator mixing"})
    agent = next(a for a in csv_rows(ap) if "Agent " + a["Logical_Node_Id"] == trace[0]["Agent_Id"])
    loc = int(agent["Location_Id"])
    assert (int(agent["Domain"]), loc >> 8, (loc >> 3) & 31, loc & 7, int(agent["Cu_Count"]), int(agent["Num_Xcc"])) == (0, 0x85, 0, 0, 256, 8)
    return profiles, [cp, tp, ap]


def run(output, binary, isa, plan, seed, chains=29, warmups=8, window=None, pmc=False):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "isa.json", isa_check(isa))
    if window:
        initial = load_json(window)
        assert initial["idle_required"] and not initial["live_pids"]
        assert all(initial["metric"]["gpu_data"][0]["usage"][k]["value"] == 0 for k in ("gfx_activity", "umc_activity"))
    gate(output / "preflight.json", idle_required=window is None)
    hwmon = list(Path("/sys/bus/pci/devices/0000:85:00.0/hwmon").glob("hwmon*/power1_input")); assert len(hwmon) == 1
    power_path = hwmon[0].resolve()
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), HERE / "vmem_hardware_tables.cpp", HERE / "vmem_pointer_chase.cpp",
               HERE / "vmem_hardware_tables_pmc.yaml", binary, isa, plan)}
    command = [str(binary), str(output / "raw"), str(plan), str(seed), str(chains), str(power_path)]
    if pmc:
        command = ["/opt/rocm/bin/rocprofv3", "-i", str(HERE / "vmem_hardware_tables_pmc.yaml"), "-d", str(output / "capture"), "--", *command]
    write_json(output / "command.json", {"command": command, "source_sha256": sources, "warmups": warmups,
        "window": str(window) if window else None, "power_path": str(power_path), "power_unit": "microW;wholeGPU"})
    with (output / "run.log").open("x") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    result, files, samples, workers, power = audit(output / "raw", warmups)
    if pmc:
        result["PMC"], more = profile(output / "capture", result); files += more
    if samples: write_csv(output / "samples.csv", samples)
    if workers: write_csv(output / "workers.csv", workers)
    write_csv(output / "power.csv", power)
    assert sources == {p: sha(p) for p in sources}
    write_json(output / "summary.json", result)
    write_json(output / "verified.json", {"status": "PASS", "source_sha256": sources, "input_sha256": {str(p): sha(p) for p in files},
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.name in ("summary.json", "samples.csv", "workers.csv", "power.csv")}})
    print("HARDWARE_TABLE_AUDIT", output.name, "cases", len(result["cases"]), "invalid", sum(not c["valid"] for c in result["cases"]), flush=True)


def combine(output, runs, pmcs):
    output.mkdir(parents=True, exist_ok=False)
    inputs, groups, profiles = {}, defaultdict(list), defaultdict(list)
    for root in [*runs, *pmcs]:
        m = load_json(root / "verified.json")
        for p, h in m["input_sha256"].items():
            assert sha(p) == h; inputs[p] = h
        for name, h in m["output_sha256"].items():
            assert sha(root / name) == h; inputs[str(root / name)] = h
        data = load_json(root / "summary.json")
        if root in runs:
            for c in data["cases"]:
                if not c["config"]["blank"]: groups[c["config"]["name"]].append(c)
        else:
            by_name = {c["config"]["name"]: c for c in data["cases"]}
            for p in data["PMC"]:
                profiles[p["case"]].append({**p, "measurement": by_name[p["case"]]})
    rows = []
    for random in range(2):
        for store in range(2):
            for policy in range(8):
                for scope in range(3):
                    lat = groups[case(random, store, policy, scope, 0)["name"]]
                    bw = groups[case(random, store, policy, scope, 1)["name"]]
                    assert len(lat) == len(bw) == 3
                    pp = profiles[case(random, store, policy, scope, 1)["name"]]
                    assert len(pp) == 3
                    valid = all(c["valid"] for c in [*lat, *bw, *pp])
                    def pool(cases, name):
                        if not valid or any(c[name] is None for c in cases): return None
                        xs = [c[name] for c in cases]; n = sum(x["count"] for x in xs)
                        return {"count": n, "mean": sum(x["count"] * x["mean"] for x in xs) / n,
                                "min": min(x["min"] for x in xs), "max": max(x["max"] for x in xs)}
                    l, cy = pool(lat, "latency_ns"), pool(lat, "latency_core_cycles")
                    mem_cus = 256 if scope == 2 else 1
                    payload = statistics.mean(c["work_payload_GBs"] for c in bw) if valid else None
                    duration = sum(p["duration_ns"] for p in pp)
                    dram = sum(p["write_bytes" if store else "read_bytes"] for p in pp) / duration if valid else None
                    rows.append({"pattern": "random" if random else "contiguous", "store": bool(store), "cache": POLICIES[policy], "scope": SCOPES[scope], "scope_id": scope,
                        "valid": valid, "latency_ns": l, "latency_core_cycles": cy, "latency_frequency_MHz": pool(lat, "effective_MHz"), "bandwidth_frequency_MHz": pool(bw, "effective_MHz"),
                        "latency_power_W": pool(lat, "power_W"), "bandwidth_power_W": pool(bw, "power_W"),
                        "payload_GBs": payload, "DRAM_GBs": dram, "memory_CUs_bandwidth": mem_cus,
                        "equivalent_FIFO_per_CU": payload / mem_cus * l["mean"] / 1024 if valid else None,
                        "latency_kind": lat[0]["latency_kind"], "per_run_valid": [c["valid"] for c in [*lat, *bw, *pp]],
                        "per_run_latency_ns": [c["latency_ns"] for c in lat], "per_run_payload_GBs": [c["work_payload_GBs"] for c in bw],
                        "PMC": [{k: v for k, v in p.items() if k != "measurement"} for p in pp]})
    write_json(output / "summary.json", {"status": "PASS", "rows": rows, "cells": len(rows),
        "scope_definition": {"single_CU": "1CTA/CU4wavesbandwidth;lane0onlyseriallatency",
            "single_CU_plus_255_MFMA": "same memoryCU +255actualotherCUsregisteronlyMFMA",
            "all_CUs": "bandwidth256CUs;latency1CUserialprobe +255CUsbulk sameop/cache/pattern(disjointbuffer)"},
        "FIFO_formula": "payload_GBs / memory_CUs_bandwidth * latency_ns /1024;singlelaneserial-to-wavebulkproxy explicitlynotphysicalFIFO",
        "power_definition": "wholeGPU microW sensor20mssamples;mean,min,max;latency/BWwindowsseparate;notCUattribution",
        "limits": ["Contiguous16Bserialload includes spatialcachehits;randompermutation is pseudorandom notcryptographic.",
                   "Store+wait is completionsemantics notphysicalDRAMretirement;randomaddressgenerationincluded.",
                   "Payload andDRAMPMC are differentpopulations/runs;FIFOusespayload,not amplifiedphysicalbytes.",
                   "No fixed5TB/s scaling;all cellsmeasured orflaggedinvalid, noimputation."]})
    flat = []
    for r in rows:
        flat.append({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in r.items()
                     if k not in ("PMC", "per_run_latency_ns", "per_run_valid", "per_run_payload_GBs")})
    write_csv(output / "hardware_tables.csv", flat)
    write_json(output / "verified.json", {"status": "PASS", "input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("action", choices=("plans", "run", "pmc", "combine"))
    p.add_argument("--output", type=Path, required=True); p.add_argument("--binary", type=Path); p.add_argument("--isa", type=Path); p.add_argument("--plan", type=Path)
    p.add_argument("--seed", type=int, default=95011); p.add_argument("--chains", type=int, default=29); p.add_argument("--warmups", type=int, default=8)
    p.add_argument("--window", type=Path); p.add_argument("--runs", nargs="+", type=Path); p.add_argument("--pmcs", nargs="+", type=Path)
    a = p.parse_args()
    if a.action == "plans": plans(a.output.resolve())
    elif a.action == "combine": combine(a.output.resolve(), [x.resolve() for x in a.runs], [x.resolve() for x in a.pmcs])
    else: run(a.output.resolve(), a.binary.resolve(), a.isa.resolve(), a.plan.resolve(), a.seed, a.chains, a.warmups, a.window.resolve() if a.window else None, a.action == "pmc")