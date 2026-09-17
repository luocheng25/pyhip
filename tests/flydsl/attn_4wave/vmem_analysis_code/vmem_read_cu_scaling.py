# SPDX-License-Identifier: MIT
"""Fixed512MiB/CU continuousread scaling and a separate single-CU MLP control.

No physicalFIFOdepth inference: keep payload,DRAMPMC,serialL and budget separate.
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
from vmem_contiguous_latency128 import SAMPLE, WORKER, NODES, PROBE_BYTES, SPAN, host_power_window
from vmem_hardware_tables import POWER
from vmem_pointer_chase import POLICIES
from vmem_pointer_chase_frequency import clock_metrics, cu, executed_region, stats


HERE = Path(__file__).resolve().parent
CUS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SLICE = 512 << 20


def case(cus, policy, bandwidth, waves=4, depth=32):
    return {"name": f"cu{cus}_p{policy}_{'bw' if bandwidth else 'lat'}_w{waves}_d{depth}", "cus": cus,
            "policy": policy, "bandwidth": bandwidth, "waves": waves, "depth": depth}


def plans(output):
    output.mkdir(parents=True, exist_ok=False)
    primary = [case(c, p, b) for c in CUS for p in range(8) for b in (0, 1)]
    # IncludeD32baselineagainwiththeMLPcontrol,donotcompareacrossdifferentlaunchcohorts.
    controls = [case(1, p, 1, waves=w, depth=d) for p in (0, 6) for w in (1, 2, 4, 8) for d in (8, 16, 32)]
    for tag, rows in (("first", primary), ("second", list(reversed(primary))), ("third", primary[64:] + primary[:64])):
        write_csv(output / f"{tag}.csv", rows)
    for tag, rows in (("first", controls), ("second", list(reversed(controls))), ("third", controls[12:] + controls[:12])):
        write_csv(output / f"control_{tag}.csv", rows)
    write_csv(output / "smoke.csv", [case(c, p, b) for c, p in ((1, 0), (2, 6), (128, 0), (256, 6)) for b in (0, 1)])
    write_csv(output / "pmc.csv", [c for c in primary if c["bandwidth"]])
    write_csv(output / "control_pmc.csv", controls)
    write_json(output / "plan.json", {"status": "PREDECLARED", "CU_counts": CUS, "cache_policies": POLICIES,
        "slice_bytes_per_memory_CU": SLICE, "primary_wave_depth": [4, 32], "memory_width_per_lane": 16,
        "latency": "oneactualCU4independentlane0chainsfull256MiB+16span,remainingN-1CUscontinuoussamecachebulkD32",
        "bandwidth": "NactualCUsallcontinuous512MiBdisjointslices,200mswarm+300mswork",
        "control": "N=1waves1/2/4/8anddepth8/16/32forcachedefault/ntsc1;separatefromprimaryCUscaling",
        "seeds": [95011, 95023, 95037], "primary_conditions": 72, "primary_dispatches_per_run": 144,
        "PMC": "onefreshcountedrunall72BWcells+24MLPcells;notordinarytime,whole_dispatchbytes/duration",
        "retest_policy": "atmostonefixedthree-runcohortforanyinvalidprimarycondition(includebothL/B)orcontrolconfig;keepfailedraw;nofastestsampling"})


def isa_check(path):
    text = path.read_text()
    bodies = re.findall(r"^(_Z\S*cu_scale_(?:bw|latency)_kernel\S*):[^\n]*\n(.*?)^\.Lfunc_end\d+:", text, re.M | re.S)
    assert len(bodies) == 20, len(bodies)
    result = []
    for name, body in bodies:
        bw = "cu_scale_bw_kernel" in name
        numbers = list(map(int, re.findall(r"Lj(\d+)E", name))); policy = numbers[0]; depth = numbers[1] if bw else 32
        flags = set(POLICIES[policy].split()) - {"default"}
        for marker in ("SCALE_BW_WARM", "SCALE_BW_WORK") if bw else ("SCALE_LAT_BG_WARM", "SCALE_LAT_BG_WORK"):
            region = executed_region(body, marker + "_BEGIN", marker + "_END")
            ops = [l for l in region.splitlines() if l.startswith("buffer_load")]
            assert len(ops) == depth and all(l.startswith("buffer_load_dwordx4") for l in ops)
            assert all(set(l.split()) & {"sc0", "sc1", "nt"} == flags for l in ops)
            assert not re.search(r"\b(?:global|flat|scratch)_(?:load|store)|\bbuffer_store|\bds_(?:read|write)|\bs_(?:load|store|buffer_load)", region)
            registers = set()
            for line in ops:
                m = re.match(r"buffer_load_dwordx4 v\[(\d+):(\d+)\],", line); assert m
                regs = set(range(int(m[1]), int(m[2]) + 1)); assert len(regs) == 4 and not registers & regs; registers |= regs
        if not bw:
            for marker in ("SCALE_CHASE_WARM", "SCALE_CHASE_MEASURE"):
                region = executed_region(body, marker + "_BEGIN", marker + "_END")
                lines = region.splitlines(); indices = [i for i, l in enumerate(lines) if l.startswith("buffer_load")]
                assert len(indices) == 129
                assert not re.search(r"\bs_memtime|\bs_memrealtime|\b(?:global|buffer|flat|scratch)_store", region)
                for i in indices:
                    assert lines[i].startswith("buffer_load_dwordx4") and lines[i+1] == "s_waitcnt vmcnt(0)"
                    assert set(lines[i].split()) & {"sc0", "sc1", "nt"} == flags
        meta = re.search(r"\.amdhsa_kernel " + re.escape(name) + r"\s+(.*?)\.end_amdhsa_kernel", text, re.S)[1]
        resources = {k: int(v) for k, v in re.findall(r"\.amdhsa_(next_free_vgpr|next_free_sgpr|private_segment_fixed_size|group_segment_fixed_size)\s+(\d+)", meta)}
        assert resources["private_segment_fixed_size"] == resources["group_segment_fixed_size"] == 0
        result.append({"symbol": name, "bandwidth": bw, "policy": policy, "depth": depth, "resources": resources})
    assert set(re.findall(r"\.(?:sgpr_spill_count|vgpr_spill_count|private_segment_fixed_size):\s*(\d+)", text)) == {"0"}
    return result


def power_window(c, start, end):
    lo = c["host_epoch1"] + (int(start) - c["raw_epoch"]) * 10 + 2000000
    hi = c["host_epoch0"] + (int(end) - c["raw_epoch"]) * 10 - 2000000
    assert hi > lo
    return lo, hi


def audit(root):
    dev, configs = load_json(root / "device.json"), load_json(root / "cases.json")
    assert dev["bdf"] == "0000:85:00.0" and dev["CUs"] == 256 and dev["wall_khz"] == 100000
    assert dev["slice_bytes"] == SLICE and dev["LDS"] == 98304 and dev["LDS_per_CU"] == 163840
    assert dev["sample_bytes"] == SAMPLE.itemsize == 112 and dev["worker_bytes"] == WORKER.itemsize == 128
    ranges = [(x, x + PROBE_BYTES + 131072) for x in dev["probe_allocations"]]
    ranges.append((dev["data_allocation"], dev["data_allocation"] + dev["data_bytes"] + 131072)); ranges.sort()
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    files = [root / "device.json", root / "cases.json"]; results, samples_out, workers_out, powers_out = [], [], [], []
    for c in configs:
        paths = [root / (c["name"] + x) for x in (".samples.bin", ".workers.bin", ".sinks.bin", ".power.bin")]; files += paths
        ss = np.fromfile(paths[0], dtype=SAMPLE); w = np.fromfile(paths[1], dtype=WORKER)
        sinks, powers = np.fromfile(paths[2], dtype="<u4"), np.fromfile(paths[3], dtype=POWER)
        assert c["cus"] in CUS and c["slice"] == SLICE and c["private"] == 0 and c["occupancy_max"] == 1 and c["checks"]
        assert len(ss) == (0 if c["bandwidth"] else 12) and len(w) == c["cus"] * c["waves"] and len(sinks) == len(w) * 64
        assert np.array_equal(w["block"], np.arange(len(w)) // c["waves"]) and np.array_equal(w["wave"], np.arange(len(w)) % c["waves"])
        assert (w["role"] == (1 if c["bandwidth"] else w["block"] != 0)).all() and (w["active_lanes"] == 64).all()
        begin = np.array([cu(x["hw0"], x["xcc0"]) for x in w]); end = np.array([cu(x["hw1"], x["xcc1"]) for x in w])
        topology = bool(np.array_equal(begin, end) and len(np.unique(begin)) == c["cus"] and (begin.reshape(-1, c["waves"]) == begin[::c["waves"], None]).all())
        overlap = True; ns_values, cycle_values, freq = [], [], []; blank = None
        if len(ss):
            s = ss.reshape(4, 3); measured = s[:, 2]
            assert np.array_equal(s["wave"], np.broadcast_to(np.arange(4)[:, None], (4, 3))) and np.array_equal(s["phase"], np.broadcast_to(np.arange(3), (4, 3)))
            assert (s["chase"]["start"] == 0).all() and (s["chase"]["end"] == 0).all() and (s["chase"]["active_lanes"] == 1).all()
            blank = statistics.median((s[:,0]["chase"]["core1"] - s[:,0]["chase"]["core0"]).tolist())
            lo, hi = int(measured["chase"]["wall0"].min()), int(measured["chase"]["wall1"].max())
            overlap = bool((w["export_wall"] > hi).all() and (s[:,:2]["export_wall"] < lo).all() and (measured["export_wall"] > hi).all())
            if c["cus"] > 1: overlap &= bool((w[4:]["work0"] <= lo).all() and (w[4:]["work1"] >= hi).all())
            for wave in range(4):
                for phase in range(3):
                    q = s[wave,phase]; z = q["chase"]; clock = clock_metrics(z)
                    assert z["steps"] == (NODES if phase else 0)
                    if not clock["frequency_valid"] or clock["CU_begin"] != begin[0]: topology = False
                    scheduled = c["epoch"] + dev["warm_ticks"] + dev["guard_ticks"] + (dev["slot_ticks"] if phase == 2 else 0)
                    assert q["scheduled"] == scheduled and q["deadline"] == scheduled + dev["slot_ticks"]
                    timely = bool(z["wall0"] >= scheduled and z["wall1"] < q["deadline"]); overlap &= timely
                    if phase:
                        salt = c["salt"] ^ (wave * 0x13579BD)
                        assert z["last"][0] == 0
                        for p in range(1,4): assert int(z["last"][p]) == (((NODES-1)*(0x9E3779B9+p*2))^(salt+p*0x13579BD)) & 0xFFFFFFFF
                    ns = cyc = None
                    if phase and clock["frequency_valid"]:
                        cyc = (clock["raw_core_cycles"] - blank) / NODES; ns = cyc * clock["wall_ns"] / clock["raw_core_cycles"]
                        if phase == 2: ns_values.append(ns); cycle_values.append(cyc); freq.append(clock["effective_GPU_MHz"])
                    samples_out.append({"case":c["name"],"wave":wave,"phase":phase,"latency_ns":ns,"cycles":cyc,"within_slot":timely,**clock})
            p0,p1 = host_power_window(c,measured)
        memory = w if c["bandwidth"] else w[4:]; warm_bytes = work_bytes = 0; memory_freq = []
        for i, x in enumerate(memory):
            block,wave = int(x["block"]),int(x["wave"]); index_offset = 0 if c["bandwidth"] else 4
            assert x["work1"] > x["work0"] and x["work_chunks"] > 0 and x["warm_chunks"] > 0
            dc,dr = int(x["core1"])-int(x["core0"]),int(x["work1"])-int(x["work0"])
            if begin[i+index_offset]==end[i+index_offset] and dc>0: memory_freq.append(dc*100/dr)
            batch = c["depth"]*c["waves"]*64; chunk = batch*8
            last = ((int(x["warm_chunks"])+int(x["work_chunks"]))*chunk-batch)%(SLICE//16)
            assert x["last_index"]==last and x["warm_last_index"]==(int(x["warm_chunks"])*chunk-batch)%(SLICE//16)
            assert int(x["work_chunks"])*chunk*16>=SLICE
            warm_bytes += int(x["warm_chunks"])*c["depth"]*64*16*8
            work_bytes += int(x["work_chunks"])*c["depth"]*64*16*8
            tids=np.arange(wave*64,(wave+1)*64,dtype=np.uint64);want=np.full(64,0x2468ACE0,dtype=np.uint64)
            slice_id=block if c["bandwidth"] else block-1
            for slot in range(c["depth"]):
                for part in range(4):
                    word=slice_id*(SLICE//4)+(last+slot*c["waves"]*64+tids)*4+part
                    want+=(word*np.uint64(0x9E3779B9)+np.uint64(c["salt"]))&np.uint64(0xFFFFFFFF)
            want&=np.uint64(0xFFFFFFFF);actual=sinks[(block*c["waves"]+wave)*64:(block*c["waves"]+wave+1)*64]
            assert np.array_equal(actual,want) and x["sum"]==want[0]
            workers_out.append({"case":c["name"],"block":block,"wave":wave,"CU_begin":int(begin[i+index_offset]),"CU_end":int(end[i+index_offset]),
                "XCD":int(x["xcc0"]),"work0":int(x["work0"]),"work1":int(x["work1"]),"raw_core_cycles":dc,"realtime_ticks":dr,
                "warm_chunks":int(x["warm_chunks"]),"work_chunks":int(x["work_chunks"]),"effective_MHz":dc*100/dr if dc>0 else None})
        if c["bandwidth"]:
            freq=memory_freq;p0,p1=power_window(c,int(memory["work0"].max()),int(memory["work1"].min()))
        kept=powers[(powers["begin_ns"]>=p0)&(powers["end_ns"]<=p1)];assert len(kept)>=2 and (kept["microwatts"]>0).all()
        for x in powers: powers_out.append({"case":c["name"],"begin_ns":int(x["begin_ns"]),"end_ns":int(x["end_ns"]),"watts":int(x["microwatts"])/1e6,"inside_work":bool(x["begin_ns"]>=p0 and x["end_ns"]<=p1)})
        assert bool(c["topology_valid"])==topology and bool(c["overlap_valid"])==overlap
        valid=bool(topology and overlap)
        envelope=(int(memory["work1"].max())-int(memory["work0"].min()))*10 if len(memory) else 0
        xcds={str(x):len({int(begin[i]) for i in range(len(w)) if int(w[i]["xcc0"])==x}) for x in range(8)}
        results.append({"config":c,"valid":valid,"blank_core_cycles":blank,"latency_ns":stats(ns_values) if valid and ns_values else None,
            "latency_core_cycles":stats(cycle_values) if valid and cycle_values else None,"frequency_MHz":stats(freq) if valid and freq else None,
            "power_W":stats(kept["microwatts"].astype(float)/1e6) if valid else None,"power_window_host_ns":[p0,p1],
            "work_payload_GBs":work_bytes/envelope if valid and envelope else None,"software_warm_bytes":warm_bytes,"software_work_bytes":work_bytes,
            "work_envelope_ns":envelope,"actual_CUs":len(np.unique(begin)),"CU_keys":list(map(int,np.unique(begin))),"XCD_CU_counts":xcds,
            "min_work_slice_passes":min(int(x["work_chunks"])*c["depth"]*c["waves"]*64*8*16//SLICE for x in memory) if len(memory) else None})
    return {"status":"PASS","device":dev,"cases":results},files,samples_out,workers_out,powers_out


def profile(capture,data):
    paths=list(capture.glob("pass_*/cu_scaling_counter_collection.csv"));assert len(paths)==1
    cp=paths[0];tp,ap=cp.parent/"cu_scaling_kernel_trace.csv",cp.parent/"cu_scaling_agent_info.csv"
    trace=sorted([t for t in csv_rows(tp) if "cu_scale_bw_kernel" in t["Kernel_Name"] or "cu_scale_latency_kernel" in t["Kernel_Name"]],key=lambda x:int(x["Start_Timestamp"]))
    assert len(trace)==len(data["cases"])
    ident=lambda x:(x["Agent_Id"],int(x["Queue_Id"]),int(x["Dispatch_Id"]))
    counts,spans=defaultdict(dict),{}
    for r in csv_rows(cp):
        k,n,v=ident(r),r["Counter_Name"],Decimal(r["Counter_Value"])
        assert n in COUNTERS.values() and n not in counts[k] and v>=0 and v==v.to_integral_value()
        counts[k][n]=int(v)*32;span=int(r["Start_Timestamp"]),int(r["End_Timestamp"]);assert spans.setdefault(k,span)==span
    assert set(counts)=={ident(x) for x in trace};result=[]
    for t,row in zip(trace,data["cases"]):
        c,k=row["config"],ident(t);expected=f"cu_scale_bw_kernel<{c['policy']}u, {c['depth']}u>" if c["bandwidth"] else f"cu_scale_latency_kernel<{c['policy']}u>"
        assert expected in t["Kernel_Name"] and int(t["Grid_Size_X"])==c["cus"]*c["waves"]*64 and int(t["Workgroup_Size_X"])==c["waves"]*64
        assert spans[k]==(int(t["Start_Timestamp"]),int(t["End_Timestamp"])) and set(counts[k])==set(COUNTERS.values())
        ns=spans[k][1]-spans[k][0];v={kind+"_bytes":counts[k][name] for kind,name in COUNTERS.items()};assert ns>0 and v["atomic_bytes"]==0
        payload=row["software_warm_bytes"]+row["software_work_bytes"]
        result.append({"case":c["name"],"valid":row["valid"],"agent":k[0],"queue":k[1],"dispatch":k[2],"duration_ns":ns,**v,
            "DRAM_read_GBs":v["read_bytes"]/ns,"same_dispatch_payload_GBs":payload/ns,"DRAM_to_payload":v["read_bytes"]/payload if payload else None,
            "scope":"whole_dispatchincludingwarmandguards;notordinaryworkenvelope"})
    agent=next(a for a in csv_rows(ap) if "Agent "+a["Logical_Node_Id"]==trace[0]["Agent_Id"]);loc=int(agent["Location_Id"])
    assert (int(agent["Domain"]),loc>>8,(loc>>3)&31,loc&7,int(agent["Cu_Count"]),int(agent["Num_Xcc"]))==(0,0x85,0,0,256,8)
    return result,[cp,tp,ap]


def run(output,binary,isa,plan,seed,window=None,pmc=False):
    output.mkdir(parents=True,exist_ok=False);write_json(output/"isa.json",isa_check(isa))
    if window:
        w=load_json(window);assert w["idle_required"] and not w["live_pids"]
        assert all(w["metric"]["gpu_data"][0]["usage"][k]["value"]==0 for k in ("gfx_activity","umc_activity"))
    gate(output/"preflight.json",idle_required=window is None)
    power=list(Path("/sys/bus/pci/devices/0000:85:00.0/hwmon").glob("hwmon*/power1_input"));assert len(power)==1
    command=[str(binary),str(output/"raw"),str(plan),str(seed),str(power[0].resolve())]
    if pmc:command=["/opt/rocm/bin/rocprofv3","-i",str(HERE/"vmem_read_cu_scaling_pmc.yaml"),"-d",str(output/"capture"),"--",*command]
    files=(binary,isa,plan,Path(__file__),HERE/"vmem_read_cu_scaling.cpp",HERE/"vmem_matched_latency128.cpp",HERE/"vmem_read_cu_scaling_pmc.yaml")
    sources={str(p):sha(p) for p in files};write_json(output/"command.json",{"command":command,"source_sha256":sources,"window":str(window) if window else None})
    with (output/"run.log").open("x") as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    data,files,samples,workers,powers=audit(output/"raw")
    if pmc:data["PMC"],extra=profile(output/"capture",data);files+=extra
    if samples:write_csv(output/"samples.csv",samples)
    if workers:write_csv(output/"workers.csv",workers)
    write_csv(output/"power.csv",powers);assert sources=={p:sha(p) for p in sources}
    write_json(output/"summary.json",data)
    write_json(output/"verified.json",{"status":"PASS","source_sha256":sources,"input_sha256":{str(p):sha(p) for p in files},
        "output_sha256":{p.name:sha(p) for p in output.iterdir() if p.name in ("summary.json","samples.csv","workers.csv","power.csv")}})
    print("CU_SCALING_AUDIT",output.name,len(data["cases"]),"invalid",[x["config"]["name"] for x in data["cases"] if not x["valid"]],flush=True)


def combine(output,runs,controls,pmc,control_pmc,retests=()):
    output.mkdir(parents=True,exist_ok=False);inputs={};grouped,control_group=defaultdict(list),defaultdict(list);all_raw=[]
    def load_verified(path):
        m=load_json(path/"verified.json")
        for p,h in m["input_sha256"].items():assert sha(p)==h;inputs[p]=h
        for p,h in m["output_sha256"].items():assert sha(path/p)==h;inputs[str(path/p)]=h
        return load_json(path/"summary.json")
    for path in runs:
        data=load_verified(path)
        for r in data["cases"]:grouped[r["config"]["name"]].append({"source":str(path),**r});all_raw.append({"source":str(path),**r})
    for path in controls:
        for r in load_verified(path)["cases"]:control_group[r["config"]["name"]].append({"source":str(path),**r})
    replacement=defaultdict(list)
    for path in retests:
        for r in load_verified(path)["cases"]:replacement[r["config"]["name"]].append({"source":str(path),**r})
    invalid_pairs={(r["config"]["cus"],r["config"]["policy"]) for vs in grouped.values() for r in vs if not r["valid"]}
    for cus,p in invalid_pairs:
        for bw in (0,1):
            name=case(cus,p,bw)["name"]
            if name in replacement:assert len(replacement[name])==3;grouped[name]=replacement[name]
    prof={r["case"]:r for r in load_verified(pmc)["PMC"]};control_prof={r["case"]:r for r in load_verified(control_pmc)["PMC"]}
    def pool(rows,key):
        if any(not r["valid"] or r[key] is None for r in rows):return None
        vs=[r[key] for r in rows];n=sum(v["count"] for v in vs)
        return {"count":n,"mean":sum(v["mean"]*v["count"] for v in vs)/n,"min":min(v["min"] for v in vs),"max":max(v["max"] for v in vs)}
    points=[]
    for cus in CUS:
        for p in range(8):
            lat=grouped[case(cus,p,0)["name"]];bw=grouped[case(cus,p,1)["name"]];assert len(lat)==len(bw)==3
            physical=prof[case(cus,p,1)["name"]];valid=all(r["valid"] for r in [*lat,*bw,physical])
            l=pool(lat,"latency_ns");b=statistics.mean(r["work_payload_GBs"] for r in bw) if valid else None
            points.append({"cus":cus,"cache":POLICIES[p],"policy":p,"valid":valid,"latency_ns":l,"latency_core_cycles":pool(lat,"latency_core_cycles"),
                "L_frequency_MHz":pool(lat,"frequency_MHz"),"B_frequency_MHz":pool(bw,"frequency_MHz"),"L_power_W":pool(lat,"power_W"),"B_power_W":pool(bw,"power_W"),
                "payload_GBs":b,"payload_per_CU_GBs":b/cus if valid else None,"DRAM_GBs":physical["DRAM_read_GBs"] if physical["valid"] else None,
                "FIFO_equivalent_per_CU":b/cus*l["mean"]/1024 if valid else None,"per_run_payload_GBs":[r["work_payload_GBs"] for r in bw],
                "per_run_latency_ns":[r["latency_ns"] for r in lat],"L_sources":[r["source"] for r in lat],"B_sources":[r["source"] for r in bw],
                "XCD_CU_counts":[r["XCD_CU_counts"] for r in bw],"PMC":physical})
    ctrl=[]
    for name,vs in control_group.items():
        assert len(vs)==3;c=vs[0]["config"];physical=control_prof[name];valid=all(v["valid"] for v in [*vs,physical])
        ctrl.append({"case":name,"policy":c["policy"],"cache":POLICIES[c["policy"]],"waves":c["waves"],"depth":c["depth"],"software_batch_per_CU":c["waves"]*c["depth"],
            "valid":valid,"payload_GBs":statistics.mean(v["work_payload_GBs"] for v in vs) if valid else None,
            "DRAM_GBs":physical["DRAM_read_GBs"] if physical["valid"] else None,"per_run_payload_GBs":[v["work_payload_GBs"] for v in vs],
            "frequency_MHz":pool(vs,"frequency_MHz"),"power_W":pool(vs,"power_W"),"PMC":physical})
    write_json(output/"summary.json",{"status":"PASS","points":points,"single_CU_MLP_controls":ctrl,"original_invalid_pairs":[list(x) for x in sorted(invalid_pairs)],
        "limits":["NbulkCUsvs1probe+(N-1)bulkCUslatency;FIFOisproxy,notphysicalcapacity.","Fixed512MiBperbulkCUallcounts;fourprobechains256MiB+128Bdisjoint.",
                  "ActualCU/XCDmappingrecorded,notassumeduniformorbound;nohardwareclockpowerCUqueuechanges.","PMCwhole-dispatch includeswarm/guardseparatefromordinaryworkwindow.",
                  "D*waves is softwarebatchnottruehardwareinflight or queue slots."]})
    write_csv(output/"cu_scaling.csv",[{k:(v["mean"] if isinstance(v,dict) and "mean" in v else v) for k,v in r.items() if k not in ("PMC","per_run_payload_GBs","per_run_latency_ns","L_sources","B_sources","XCD_CU_counts")} for r in points])
    write_csv(output/"single_CU_MLP.csv",[{k:(v["mean"] if isinstance(v,dict) and "mean" in v else v) for k,v in r.items() if k not in ("PMC","per_run_payload_GBs")} for r in ctrl])
    write_json(output/"verified.json",{"status":"PASS","input_sha256":inputs,"source_sha256":sha(Path(__file__)),"output_sha256":{p.name:sha(p) for p in output.iterdir() if p.is_file()}})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("plans","run","pmc","combine"));p.add_argument("--output",required=True,type=Path)
    p.add_argument("--binary",type=Path);p.add_argument("--isa",type=Path);p.add_argument("--plan",type=Path);p.add_argument("--seed",type=int,default=95011);p.add_argument("--window",type=Path)
    p.add_argument("--runs",nargs="+",type=Path);p.add_argument("--controls",nargs="+",type=Path);p.add_argument("--pmc",type=Path);p.add_argument("--control-pmc",type=Path);p.add_argument("--retests",nargs="*",type=Path,default=[])
    a=p.parse_args()
    if a.action=="plans":plans(a.output.resolve())
    elif a.action=="combine":combine(a.output.resolve(),[r.resolve() for r in a.runs],[r.resolve() for r in a.controls],a.pmc.resolve(),a.control_pmc.resolve(),[r.resolve() for r in a.retests])
    else:run(a.output.resolve(),a.binary.resolve(),a.isa.resolve(),a.plan.resolve(),a.seed,a.window.resolve() if a.window else None,a.action=="pmc")