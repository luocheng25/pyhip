# SPDX-License-Identifier: MIT
"""gfx950 physical-SIMD ATT ledger for persistent and one-shot MoE tasks.

Uses the classification/PC mapping from attn_4wave/tools/stall_analysis.md.
MFMA16x16x128 executes for32 cycles on gfx950 (not the gfx94216-cycle
example). Successful issue=attempt+stall; complete=attempt+duration.
Task prologue/epilogue includes each persistent task's preparation/drain.
Task boundaries are located at the repeated mbcnt task-entry PC, never by
a guessed gap threshold. Instruction timestamps do not identify expert/OC
IDs, so an inter-task gap is not asserted to be the same expert's next OC.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

METHOD = Path(__file__).resolve().parents[2] / "attn_4wave/tools/analyze_mfma_stall.py"
spec = importlib.util.spec_from_file_location("att_ledger_method", METHOD)
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
TICK = 4
# The reference module's MFMA_EXEC_CYCLES=16 is for its gfx942 example.
# Reuse classification/4-cycle painting only, never its execution-window code.
MFMA_EXEC_CYCLES = 32
MFMA_PER_N = 32  # Instruction count per BN128 tile, not execution latency.


def parse(dispatch, per_task, cycles, persistent):
    if cycles != MFMA_EXEC_CYCLES:
        raise ValueError("gfx950 MFMA16x16x128 requires 32-cycle execution windows")
    code, _ = method.load_code(dispatch)
    groups = defaultdict(list)
    counts = Counter()
    for path in sorted(dispatch.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text())
        assert payload["num_insts"] == payload["num_stitched"], f"incomplete trace: {path}"
        wave = payload["wave"]
        records, mfmas = [], []
        for raw in wave["instructions"]:
            info = code[int(raw[4])]
            r = {"attempt": int(raw[0]), "issue": int(raw[0]) + int(raw[2]),
                 "end": int(raw[0]) + int(raw[3]), "pc": int(raw[4]),
                 "category": info.category, "opcode": info.opcode}
            assert r["attempt"] <= r["issue"] <= r["end"]
            records.append(r)
            if info.category == "MFMA":
                mfmas.append(r)
        counts["wave_files"] += 1
        if not mfmas:
            counts["early_exit_waves"] += 1
            continue
        assert len(mfmas) % per_task == 0, (path, len(mfmas), per_task)
        task_count = len(mfmas) // per_task
        assert persistent or task_count == 1
        # Only the explicit _task_thread_id identity uses literal -1. LLVM
        # may add exec-mask mbcnts in the lane0 atomic block on one wave.
        task_starts = [r for r in records if r["opcode"].startswith("v_mbcnt_lo_u32_b32")
                       and ", -1, 0" in code[r["pc"]].asm]
        if persistent:
            # One mbcnt at each task acquisition, plus a final empty dequeue.
            assert len(task_starts) in (task_count, task_count + 1), (path, len(task_starts), task_count)
        tasks = []
        for i in range(task_count):
            ms = mfmas[i * per_task:(i + 1) * per_task]
            begin = wave["begin"] if i == 0 or not persistent else task_starts[i]["attempt"]
            end = task_starts[i + 1]["attempt"] if persistent and i + 1 < task_count else wave["end"]
            assert begin <= ms[0]["issue"] and ms[-1]["issue"] + cycles <= end
            tasks.append({"begin": begin, "end": end, "mfmas": ms})
        se = int(re.match(r"se(\d+)_", path.name)[1])
        groups[(se, int(wave["cu"]), int(wave["simd"]))].append(
            {"file": path.name, "slot": int(wave["slot"]), "begin": wave["begin"], "end": wave["end"],
             "records": records, "mfmas": mfmas, "tasks": tasks})
        counts["active_waves"] += 1
        counts["wave_tasks"] += task_count
    assert groups, "no complete active waves"
    return code, groups, counts


def ledger(code, groups, n_tiles, mfma_per_n, cycles, first_n, last_n):
    if cycles != MFMA_EXEC_CYCLES:
        raise ValueError("gfx950 MFMA16x16x128 requires 32-cycle execution windows")
    names = ["inactive", "scheduler/ready"]
    categories = sorted({c.category for c in code})
    names += [f"{state}:{category}" for state in ("stall", "issue") for category in categories]
    name_id = {name: i for i, name in enumerate(names)}
    # Exclusive owner priorities exactly follow the reference method.
    phase_names = ["inactive", "prologue", "steady", "tail", "epilogue"]
    owner_lookup = np.zeros((len(phase_names), len(names)), dtype=np.int8)
    detail_lookup = {}
    for p, phase in enumerate(phase_names):
        for b, blocker in enumerate(names):
            category, detail = method.classify(blocker, phase)
            owner_lookup[p, b] = method.CATEGORY_RANK[category]
            detail_lookup[p, b] = detail
    lifecycle = defaultdict(list)
    wave_lifecycle = defaultdict(list)
    totals = Counter()
    main = Counter()
    details, opcodes, phases, pcs, accesses, waits = (defaultdict(Counter) for _ in range(6))
    joint, all_same = Counter(), Counter()
    physical = []
    for key, waves in sorted(groups.items()):
        origin = min(w["begin"] for w in waves) // TICK * TICK
        end = max(w["end"] for w in waves)
        length = math.ceil((end - origin) / TICK)
        slots = sorted({w["slot"] for w in waves})
        slot_ids = {s: i for i, s in enumerate(slots)}
        shape = (len(slots), length)
        active, mfma, internal = (np.zeros(shape, dtype=bool) for _ in range(3))
        phase = np.zeros(shape, dtype=np.int8)
        state = np.zeros(shape, dtype=np.int16)
        pc = np.full(shape, -1, dtype=np.int32)
        by_slot = defaultdict(list)
        for wave in waves:
            s = slot_ids[wave["slot"]]
            method.paint(active[s], wave["begin"], wave["end"], origin, True)
            method.paint(state[s], wave["begin"], wave["end"], origin, name_id["scheduler/ready"])
            for task in wave["tasks"]:
                by_slot[s].append(task)
                ms = task["mfmas"]
                method.paint(phase[s], task["begin"], ms[0]["issue"], origin, 1)
                method.paint(phase[s], ms[0]["issue"], ms[-1]["issue"] + cycles, origin, 2)
                method.paint(phase[s], ms[-1]["issue"] + cycles, task["end"], origin, 4)
                for tile in range(n_tiles - 1):
                    left = ms[(tile + 1) * mfma_per_n - 1]["issue"] + cycles
                    right = ms[(tile + 1) * mfma_per_n]["issue"]
                    method.paint(phase[s], left, right, origin, 3)
                if last_n > first_n:
                    method.paint(internal[s], ms[first_n * mfma_per_n]["issue"],
                                 ms[last_n * mfma_per_n - 1]["issue"] + cycles, origin, True)
            for r in wave["records"]:
                method.paint(state[s], r["attempt"], r["issue"], origin, name_id[f"stall:{r['category']}"])
                method.paint(state[s], r["issue"], r["end"], origin, name_id[f"issue:{r['category']}"])
                method.paint(pc[s], r["attempt"], r["end"], origin, r["pc"])
            for r in wave["mfmas"]:
                method.paint(mfma[s], r["issue"], r["issue"] + cycles, origin, True)
        # Pair concurrent resident tasks by slot order only when their spans
        # overlap. Refuse incomplete/unequal captures rather than silently drop.
        for tasks in by_slot.values():
            tasks.sort(key=lambda t: t["begin"])
        assert len({len(t) for t in by_slot.values()}) == 1, f"unequal resident task counts at {key}"
        task_lifetimes, boundary_gaps = [], []
        previous_last = None
        batches = [[by_slot[s][i] for s in sorted(by_slot)]
                   for i in range(len(next(iter(by_slot.values()))))]
        for i, batch in enumerate(batches):
            assert max(t["begin"] for t in batch) < min(t["end"] for t in batch), f"nonconcurrent task pairing {key}"
            t0 = min(t["begin"] for t in batch)
            t1 = min(t["mfmas"][0]["issue"] for t in batch)
            t2 = max(t["mfmas"][-1]["issue"] + cycles for t in batch)
            t3 = max(t["end"] for t in batch)
            if i + 1 < len(batches):
                # Paired waves cross the persistent-loop boundary a few cycles
                # apart. Use a single cut to avoid double-counting that overlap.
                t3 = min(t3, min(t["begin"] for t in batches[i + 1]))
            assert t0 <= t1 <= t2 <= t3
            for name, val in (("prologue", t1-t0), ("steady", t2-t1), ("epilogue", t3-t2), ("lifetime", t3-t0)):
                lifecycle[name].append(val)
            assert (t1-t0)+(t2-t1)+(t3-t2) == t3-t0
            if previous_last is not None:
                gap = max(0, t1-previous_last)
                boundary_gaps.append(gap)
                lifecycle["inter_task_mfma_gap"].append(gap)
            previous_last = t2
            task_lifetimes.append((t0,t3))
        # The reference method's resident *wave* batch lifetime is distinct
        # from the repeated logical tasks in one persistent resident batch.
        wave_slots = defaultdict(list)
        for wave in waves:
            wave_slots[wave["slot"]].append(wave)
        for batch in wave_slots.values():
            batch.sort(key=lambda w: w["begin"])
        assert len({len(w) for w in wave_slots.values()}) == 1
        wave_ends = []
        for i in range(len(next(iter(wave_slots.values())))):
            batch = [wave_slots[s][i] for s in sorted(wave_slots)]
            begin = min(w["begin"] for w in batch)
            end_batch = max(w["end"] for w in batch)
            first = min(w["mfmas"][0]["issue"] for w in batch)
            last = max(w["mfmas"][-1]["issue"] + cycles for w in batch)
            assert max(w["begin"] for w in batch) < min(w["end"] for w in batch)
            for name, val in (("prologue", first-begin), ("steady", last-first),
                              ("epilogue", end_batch-last), ("lifetime", end_batch-begin)):
                wave_lifecycle[name].append(val)
            if wave_ends:
                wave_lifecycle["inter_batch_gap"].append(max(0, begin-wave_ends[-1]))
            wave_ends.append(end_batch)
        union = np.any(mfma, axis=0)
        active_union = np.any(active, axis=0)
        # Dynamically discover resident slots; no hardcoded two-wave divisor.
        resident = int(active.sum(axis=0).max())
        selected = (active.sum(axis=0) == resident) & np.all(~active | internal, axis=0)
        busy, idle = selected & union, selected & ~union
        totals["selected_cycles"] += int(selected.sum()) * TICK
        totals["busy_cycles"] += int(busy.sum()) * TICK
        totals["idle_cycles"] += int(idle.sum()) * TICK
        totals["active_cycles"] += int(active_union.sum()) * TICK
        totals["active_mfma_cycles"] += int((active_union & union).sum()) * TICK
        physical.append({"key": key, "resident_waves": resident, "wave_count": len(waves),
                         "task_batches": len(task_lifetimes), "inter_task_gap_cycles": sum(boundary_gaps),
                         "min_cycle": origin, "max_cycle": end,
                         "active_mfma_fraction": float((active_union & union).sum()/active_union.sum())})
        ranks = owner_lookup[phase, state]
        ranks[~active] = 127
        chosen = ranks.min(axis=0)
        for rank, category in enumerate(method.CATEGORIES):
            owned = idle & (chosen == rank)
            main[category] += int(owned.sum()) * TICK
            matching = active & (ranks == rank) & owned
            shares = TICK / np.maximum(1, matching.sum(axis=0))
            for s in range(len(slots)):
                indices = np.flatnonzero(matching[s])
                for tick in indices:
                    weight = float(shares[tick])
                    p, b, index = int(phase[s,tick]), int(state[s,tick]), int(pc[s,tick])
                    details[category][detail_lookup[p,b]] += weight
                    phases[category][phase_names[p]] += weight
                    opcodes[category][code[index].opcode if index >= 0 else "<ready>"] += weight
                    pcs[category][index] += weight
                    asm = code[index].asm if index >= 0 else "<ready>"
                    if category in ("VMEM issue", "LDS issue"):
                        access = "load/read" if code[index].category in ("VMEM-load", "DS-read") else "store/write"
                        accesses[category][access] += weight
                    if category in ("VMEM wait", "LDS wait"):
                        waits[category][asm] += weight
        for tick in np.flatnonzero(idle):
            rs = [int(ranks[s, tick]) for s in np.flatnonzero(active[:, tick])]
            joint[" + ".join(sorted(method.CATEGORIES[r] for r in rs))] += TICK
            if len(set(rs)) == 1:
                all_same[method.CATEGORIES[rs[0]]] += TICK
    assert totals["selected_cycles"] == totals["busy_cycles"] + totals["idle_cycles"]
    assert sum(main.values()) == totals["idle_cycles"]
    for category in method.CATEGORIES:
        for children in (details, phases, opcodes, pcs):
            assert math.isclose(sum(children[category].values()), main[category], abs_tol=1e-5)
        if category in ("VMEM issue", "LDS issue"):
            assert math.isclose(sum(accesses[category].values()), main[category], abs_tol=1e-5)
        if category in ("VMEM wait", "LDS wait"):
            assert math.isclose(sum(waits[category].values()), main[category], abs_tol=1e-5)
    lifetime = sum(lifecycle["lifetime"])
    segments = {key: {"cycles": sum(lifecycle[key]), "fraction": sum(lifecycle[key])/lifetime,
                       "min": min(lifecycle[key]),
                       **method.distribution(lifecycle[key])} for key in ("prologue","steady","epilogue")}
    assert sum(v["cycles"] for v in segments.values()) == lifetime
    wave_lifetime = sum(wave_lifecycle["lifetime"])
    wave_segments = {key: {"cycles": sum(wave_lifecycle[key]), "fraction": sum(wave_lifecycle[key])/wave_lifetime,
                           "min": min(wave_lifecycle[key]), **method.distribution(wave_lifecycle[key])}
                     for key in ("prologue", "steady", "epilogue")}
    assert sum(v["cycles"] for v in wave_segments.values()) == wave_lifetime
    gap = sum(wave_lifecycle["inter_batch_gap"])
    return {"mfma_execution_cycles": cycles, "equivalent_slot_cycles": cycles,
            "selected_mfma_slots": totals["selected_cycles"] / cycles,
            "busy_mfma_slots": totals["busy_cycles"] / cycles,
            "idle_mfma_slots": totals["idle_cycles"] / cycles,
            "physical_simds": physical, "task_lifecycle": segments, "task_lifetime_cycles": lifetime,
            "wave_lifecycle": wave_segments, "wave_lifetime_cycles": wave_lifetime,
            "inter_wave_batch_gap": {"cycles": gap, "horizon_fraction": gap/(wave_lifetime+gap),
                                     **method.distribution(wave_lifecycle["inter_batch_gap"])},
            "inter_task_mfma_gap": {"cycles": sum(lifecycle["inter_task_mfma_gap"]),
                                     "active_fraction": sum(lifecycle["inter_task_mfma_gap"])/totals["active_cycles"],
                                     "horizon_fraction": sum(lifecycle["inter_task_mfma_gap"])/(wave_lifetime+gap),
                                     "note": "supplemental last-MFMA to next-first-MFMA gaps include any no-wave inter-batch gap; do not add to P/S/E",
                                     **method.distribution(lifecycle["inter_task_mfma_gap"])},
            **dict(totals), "internal_window_coverage": totals["selected_cycles"]/segments["steady"]["cycles"],
            "steady_mfma_fraction": totals["busy_cycles"]/totals["selected_cycles"],
            "active_mfma_fraction": totals["active_mfma_cycles"]/totals["active_cycles"],
            "categories": {c: {"cycles": main[c], "equivalent_mfma_slots": main[c]/cycles,
                                 "steady_fraction": main[c]/totals["selected_cycles"],
                                 "idle_fraction": main[c]/totals["idle_cycles"] if totals["idle_cycles"] else 0,
                                 "details": dict(details[c]), "phases": dict(phases[c]), "opcodes": dict(opcodes[c]),
                                 "accesses": dict(accesses[c]), "wait_thresholds": dict(waits[c]),
                                 "top_pcs": [{"pc_index": pc, "cycles": val, "asm": code[pc].asm if pc>=0 else "<ready>",
                                              "source": code[pc].source if pc>=0 else ""}
                                             for pc,val in pcs[c].most_common(8)]} for c in method.CATEGORIES},
            "witness_joint_states": dict(joint.most_common(10)), "witness_all_waves_same": dict(all_same)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--schedule", choices=("persistent","oneshot","xcd"), default="persistent")
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--active-mblocks", type=int, default=768)
    parser.add_argument("--capacity-mblocks", type=int, default=896)
    parser.add_argument("--first-n", type=int, default=2)
    parser.add_argument("--last-n-exclusive", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    tiles = args.n // args.splits // 128
    end = args.last_n_exclusive or tiles - 2
    assert 0 <= args.first_n < end <= tiles
    code, groups, counts = parse(args.dispatch, tiles * MFMA_PER_N, MFMA_EXEC_CYCLES, args.schedule=="persistent")
    result = ledger(code, groups, tiles, MFMA_PER_N, MFMA_EXEC_CYCLES, args.first_n, end)
    tasks = args.active_mblocks * args.splits
    launch = 256 if args.schedule=="persistent" else args.capacity_mblocks * args.splits
    static = method.static_distribution(tasks, tasks, 256, 8, 4, 2)
    static = {"launch_workgroups": launch, "logical_tasks": tasks,
              "active_workgroups": None if args.schedule=="persistent" else tasks,
              "uniform_early_exit_workgroups": None if args.schedule=="persistent" else launch-tasks,
              "cu_count": 256, "xcd_count": 8,
              "ideal_task_capacity_model": static,
              "model_note": "ideal logical-task quotient/remainder counts are NOT measured or guaranteed per-CU workgroup distribution",
              "scheduler_note": ("global atomic queue; actual per-CU tasks depend on execution and cannot be statically determined"
                                 if args.schedule=="persistent" else
                                 "active-prefix transpose8 and remainder identity" if args.schedule=="xcd" else "identity task mapping")}
    static["ideal_task_capacity_model"]["model"] = "logical-task-equivalent batches; not resident wave counts for persistent scheduling"
    output = {"trace": str(args.dispatch), "tick_cycles": TICK,
              "code_sha256": hashlib.sha256((args.dispatch/"code.json").read_bytes()).hexdigest(),
              "splits":args.splits, "schedule":args.schedule, "n_tiles_per_task":tiles,
              "window":[args.first_n,end], "trace_counts":dict(counts), "static":static, **result}
    print(json.dumps({k:v for k,v in output.items() if k not in ("categories","physical_simds")}, indent=2))
    print(f"MFMA window: [successful issue, issue + {MFMA_EXEC_CYCLES}); "
          f"equivalent slots = cycles / {MFMA_EXEC_CYCLES}; {TICK}-cycle sampling ticks.")
    print(f"| Category | cycles | {MFMA_EXEC_CYCLES}-cycle equivalent slots | % steady | % idle |")
    print("|---|---:|---:|---:|---:|")
    print(f"| MFMA busy | {result['busy_cycles']} | {result['busy_mfma_slots']:.3f} | "
          f"{100*result['steady_mfma_fraction']:.3f} | - |")
    for name,row in result["categories"].items():
        print(f"| {name} | {row['cycles']} | {row['equivalent_mfma_slots']:.3f} | "
              f"{100*row['steady_fraction']:.3f} | {100*row['idle_fraction']:.3f} |")
    print("LEDGER_ASSERTIONS_PASS; sampled SIMD results, not dispatch wall time; per-task P/S/E denominators; gaps supplemental, not additive")
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()