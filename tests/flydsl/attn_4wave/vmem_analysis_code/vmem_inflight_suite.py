# SPDX-License-Identifier: MIT
"""Common capture-to-model adapter for all maintained MoE down candidates.

Preserves every decoded wave (including empty CTAs) and the complete
first/last-MFMA hull. Source maps are validated against the frozen ISA; the
payload model is not an address trace or a DRAM-transaction measurement.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np

from vmem_att_suite import CASES, ROOT
from vmem_inflight import BIN, CLASSES, HERE, PERCENTILES, fixed_model, sha, write_csv, write_json
from vmem_inflight_m256 import SCENARIOS


def canonical(op):
    name = op.split()[0]
    return re.sub(r"_(?:e32|e64)$", "", name)


def isa_map(path, code):
    """Align real machine instructions; tolerate only assembler nop padding.

    The standalone PyHIP source has assembler aliases (_e32 omitted), which
    are compared canonically; all ISA operands remain available alongside
    the decoder's spelling. No dynamic instructions may be omitted.
    """
    lines = path.read_text().splitlines()
    entries, source_files = [], {}
    dwarf_file, dwarf_line, phase, stage = None, None, "entry", -1
    start = False
    for lineno, line in enumerate(lines, 1):
        text = line.strip()
        declaration = re.match(r'\.file\s+(\d+)\s+"([^\"]+)"(?:\s+"([^\"]+)")?', text)
        if declaration:
            number, first, second = declaration.groups()
            source_files[int(number)] = str(Path(first) / second) if second else first
        location = re.match(r"\.loc\s+(\d+)\s+(\d+)", text)
        if location:
            dwarf_file, dwarf_line = map(int, location.groups())
        marker = re.search(r"MOE(?:8|128)_(MEMORY|COMPUTE)_(BEGIN|END)_\d", text)
        if marker:
            kind, edge = marker.groups()
            if kind == "MEMORY" and edge == "BEGIN":
                phase, start = "memory", True
                stage += 1
            elif kind == "COMPUTE" and edge == "BEGIN":
                phase = "compute"
            elif kind == "COMPUTE" and edge == "END":
                phase = "transition"
        if not re.match(r"(?:[sv]_|buffer_|global_|flat_|ds_|scratch_)", text):
            continue
        asm, _, comment = text.partition(";")
        origin = source_files.get(dwarf_file)
        entries.append({"op": asm.strip(), "isaLine": lineno, "comment": comment,
                        "dwarf": f"{origin}:{dwarf_line}" if origin and dwarf_line else None,
                        "phase": phase, "stage": stage, "memoryStart": start})
        start = False
    rows = [row for row in code if not str(row[0]).lstrip().startswith(";")]
    mapped, at = {}, 0
    for entry in entries:
        if at >= len(rows):
            # PyHIP emits an unreachable compiler tail after its assembly
            # s_endpgm; it is absent from the decoder's live code range.
            break
        if canonical(entry["op"]) != canonical(str(rows[at][0])):
            previous_line = next(reversed(mapped.values()))["isaLine"] if mapped else 0
            alignments = [i + 1 for i in range(previous_line, entry["isaLine"] - 1)
                          if re.match(r"\s*\.p2align", lines[i])]
            assert alignments, (path, entry, rows[at])
            while str(rows[at][0]).strip() == "s_nop 0":
                row = rows[at]
                mapped[int(row[2])] = {**entry, "op": str(row[0]).strip(), "isaLine": alignments[0],
                                      "padding": True, "memoryStart": False, "phase": "alignment_padding"}
                at += 1
        row = rows[at]
        assert canonical(entry["op"]) == canonical(str(row[0])), (path, entry, row)
        mapped[int(row[2])] = {**entry, "op": str(row[0]).strip(), "source": str(row[3]),
                              "address": int(row[5]), "padding": False}
        at += 1
    assert at == len(rows), (path, len(entries), len(rows), at)
    return mapped, lines


def decode(capture):
    manifest = json.loads((capture / "summary.json").read_text())
    assert manifest["status"] == "PASS"
    case = manifest["candidate"]
    info = CASES[case]
    assert manifest["schema"] == "moe-att-suite-v1"
    for name, digest in manifest["raw_sha256"].items():
        assert sha(capture / name) == digest, name
    ui, source = capture / manifest["ui"], capture / manifest["source_snapshot"]
    code = json.loads((ui / "code.json").read_text())["code"]
    static, isa = isa_map(capture / manifest["isa"], code)
    sources, sites = {}, {}
    source_lines = source.read_text().splitlines()
    sources[source.name] = {"lines": source_lines, "sha256": sha(source), "original": info["source"]}

    def locate(text):
        matches = [i + 1 for i, line in enumerate(source_lines) if text in line]
        assert len(matches) == 1, (text, matches)
        return matches[0]

    expert_line = locate({"m128": "expert = _scalar(expert_ids[block_m])",
                          "m256": "expert = _scalar(sorted_expert_ids[blk_m])",
                          "legacy": "expert = fx.Int32(sorted_expert_ids[blk_m])",
                          "pyhip": "J.s_load_dword(expert_id, _sorted_expert_ids"}[info["family"]])
    helpers = {"m128": {"b_dma": "def dma_turn(", "a_load": "awords[None, part, mi, kb].store(apacket.load", "c_store": "def store_one("},
               "m256": {"b_dma": "def dma_b(", "a_load": "a_words[None, part, mi, kb].store(a_packet.load", "c_store": "def store_quarter("},
               "legacy": {"b_dma": "def global_load_b(", "a_load": "a_mma_frag_w[None, step, mi, kb] = input_data[", "c_store": "def global_store("},
               "pyhip": {"b_dma": "def vm_load_B(", "a_load": "J.global_load_dwordx4(mfma_A", "c_store": "def storeC("}}[info["family"]]
    helper_lines = {kind: locate(text) for kind, text in helpers.items()}
    expert_pcs, aliases = [], Counter()
    memory_starts = {pc for pc, entry in static.items() if entry["memoryStart"]}
    legacy_metadata = [pc for pc, entry in static.items() if not entry["padding"]
                       and entry["op"].startswith("buffer_load_dwordx4") and " lds" in entry["op"]
                       and "moe_8wave_down_utils.py:94" in entry["source"]] if info["family"] == "legacy" else []
    if info["family"] == "legacy":
        assert len(legacy_metadata) == 3, legacy_metadata
    for pc, entry in static.items():
        if entry["padding"]:
            sites[str(pc)] = {"pc": pc, "op": entry["op"], "phase": "alignment_padding", "kind": None,
                "file": None, "line": None, "isaLine": entry["isaLine"], "meaning": "Dead assembler padding", "anchors": [], "payload": 0, "address": "", "payloadSemantics": "not executed"}
            continue
        filename, line_number, mapping = None, None, "No DWARF source"
        if info["family"] == "pyhip":
            matches = re.findall(r"moe_gemm_8wave\.py:(\d+)", entry["comment"])
            if matches:
                filename, line_number, mapping = source.name, int(matches[-1]), "PyHIP emitter comment (not DWARF)"
        else:
            raw_source = entry["source"]
            path, sep, number = raw_source.rpartition(":")
            if sep and number.isdigit() and int(number) > 0:
                snapshots = list(ui.glob("source_*_" + Path(path).name))
                if len(snapshots) == 1:
                    frozen = snapshots[0]
                    filename, line_number, mapping = frozen.name, int(number), "DWARF frozen source"
                    if filename not in sources:
                        sources[filename] = {"lines": frozen.read_text().splitlines(), "sha256": sha(frozen), "original": path}
        if filename and line_number:
            assert line_number <= len(sources[filename]["lines"]), (pc, filename, line_number)
        text = sources[filename]["lines"][line_number - 1] if filename and line_number else ""
        primary = (filename == source.name or (filename and filename.endswith(Path(info["source"]).name)))
        op = entry["op"]
        is_expert = primary and line_number == expert_line and (op.startswith("s_load_") or op.startswith("global_load_"))
        if is_expert:
            expert_pcs.append(pc)
        kind, payload, semantics, meaning, latency_class = None, 0, "", entry["phase"], None
        if op.startswith(("global_atomic_", "buffer_atomic_")):
            kind, payload, semantics, meaning = "atomic", 4, "one_lane_operand_not_DRAM_transaction", "Queue atomic"
        elif op.startswith(("buffer_store_dwordx4", "global_store_dwordx4")):
            kind, payload, semantics, meaning = "c_store", 1024, "capacity_upper_unknown_row_masks", "C store"
        elif op.startswith("global_store_dword"):
            kind, payload, semantics, meaning = "metadata", 4, "one_lane_queue_reset", "Queue reset"
            latency_class = "c_store"
        elif op.startswith("buffer_load_dwordx4") and " lds" in op:
            if info["family"] == "legacy" and filename and filename.endswith("moe_8wave_down_utils.py"):
                # Partial helper (tid<tail_atoms) transfers metadata; its
                # full-round site supplies B weights. The frozen line says
                # which branch contains the exact intrinsic.
                partial = line_number > 88
                if partial:
                    ordinal = legacy_metadata.index(pc)
                    kind, semantics = "metadata", "source_verified_cooperative_partial_dma"
                    payload = 384 if ordinal == 2 else 1024
                    meaning = ("Sorted IDs DMA", "Routing weights DMA", "B scales DMA (24 lanes)")[ordinal]
                    latency_class = "b_dma"
                else:
                    kind, payload, semantics, meaning = "b_dma", 1024, "full_wave64", "B DMA → LDS"
            else:
                kind, payload, semantics, meaning = "b_dma", 1024, "full_wave64", "B DMA → LDS"
        elif op.startswith("global_load_lds_dwordx4"):
            assert info["family"] == "pyhip"
            kind, payload, semantics, meaning = "metadata", (384 if "wg_load_lds(lds_scaleB" in text else 1024), "capacity_upper_exec_mask_not_observed", "Metadata DMA → LDS"
            latency_class = "b_dma"
        elif op.startswith(("buffer_load_dwordx4", "global_load_dwordx4")):
            kind, payload, semantics, meaning = "a_load", 1024, "capacity_upper_unknown_row_masks", "A load"
        elif op.startswith("buffer_load_dword"):
            kind, payload, semantics, meaning = "metadata", 256, "capacity_upper_wave_operand_bytes", "A scales"
        elif op.startswith("global_load_dword"):
            kind, semantics = "metadata", "source_verified_uniform_or_masked"
            payload, meaning = (4, "Expert ID") if is_expert else (256, "Metadata")
            if "weight_scales[" in text:
                payload, meaning = 4 * (6144 // info["oc"] // 128 * 2), "B scales"
            elif "sorted_ids[" in text:
                meaning = "Sorted IDs"
            elif "sorted_weights[" in text:
                meaning = "Routing weights"
        else:
            assert not op.startswith(("buffer_", "global_", "flat_", "scratch_")), (case, pc, op)
        if kind:
            latency_class = latency_class or kind
            anchors = [{"file": source.name, "line": helper_lines[kind], "label": meaning + " helper"}] if kind in helper_lines else []
        else:
            anchors = []
        entry.update(kind=kind, payload=payload, semantics=semantics, latencyClass=latency_class, expertStart=is_expert)
        phase = entry["phase"]
        if kind == "atomic":
            phase = "queue"
        elif kind == "metadata":
            phase = "queue_exit" if "reset" in meaning else "metadata"
        elif not memory_starts:
            # Legacy/PyHIP have no MOE phase markers. Do not call their
            # entire statically unrolled pipeline a task prologue.
            phase = {"b_dma": "B_DMA_site", "a_load": "A_payload_site", "c_store": "C_store_site"}.get(kind, "unmarked_code_context")
        elif phase == "entry" and kind in ("a_load", "b_dma"):
            phase = "task_prologue"
        sites[str(pc)] = {"pc": pc, "op": op, "phase": phase, "file": filename, "line": line_number,
            "sourceMapping": mapping, "isaLine": entry["isaLine"], "address": hex(entry["address"]),
            "kind": CLASSES.index(kind) if kind else None, "payload": payload, "payloadSemantics": semantics,
            "latencyClass": CLASSES.index(latency_class) if latency_class else None,
            "meaning": meaning, "anchors": anchors}
        if info["family"] == "legacy" and pc in legacy_metadata:
            anchors.append({"file": source.name, "line": locate(("ROCDLBuffer(sorted_ids_src).load_async(",
                "ROCDLBuffer(sorted_weights_src).load_async(", "ROCDLBuffer(weight_scales).load_async(")[legacy_metadata.index(pc)]), "label": meaning + " caller"})
        if kind:
            aliases[(kind, op.split()[0], payload, latency_class)] += 1
    assert len(expert_pcs) == 1, (case, expert_pcs)
    waves, events, task_stats = [], [], []
    for wave_id, path in enumerate(sorted(ui.glob("se*_sm*_sl*_wv*.json"))):
        raw = json.loads(path.read_text())
        wave = raw["wave"]
        task, q = -1, -1
        records, mfmas, per_task = [], [], defaultdict(Counter)
        count_b, count_c = 0, 0
        for attempt, category, stall, duration, pc in wave["instructions"]:
            pc = int(pc)
            entry = static[pc]
            assert not entry["padding"]
            if entry["expertStart"]:
                task += 1
                q, count_b, count_c = -1, 0, 0
            if pc in memory_starts:
                q += 1
            issue = int(attempt + stall)
            phase = sites[str(pc)]["phase"]
            if phase in ("memory", "compute", "transition") and q >= 0:
                phase += f"_q{q}"
            records.append([issue, pc, task, phase, q])
            if entry["op"].startswith("v_mfma_"):
                mfmas.append(issue)
                per_task[task]["mfma"] += 1
            if entry["kind"]:
                kind = entry["kind"]
                per_task[task][kind] += 1
                supplied = -1
                if kind == "b_dma":
                    turns = (64 if info["family"] in ("m128", "m256") else info["block_n"]) * 256 // (info["waves"] * 1024)
                    supplied = count_b // turns
                    count_b += 1
                elif kind == "c_store":
                    records_per_packet = (64 if info["family"] in ("m128", "m256") else info["block_n"]) * 32 * 2 // 1024
                    supplied = count_c // records_per_packet
                    count_c += 1
                events.append({"issue": issue, "pc": pc, "wave": wave_id, "task_ordinal": task, "kind": kind,
                               "payload_bytes": entry["payload"], "payload_semantics": entry["semantics"],
                               "latency_class": entry["latencyClass"], "packet": q, "supplied_packet": supplied,
                               "phase": phase, "source": entry["source"], "op": entry["op"], "isa_line": entry["isaLine"]})
        for ordinal, counts in per_task.items():
            mfma = counts["mfma"]
            assert mfma in (0, manifest["mfma_per_task"]), (case, wave_id, ordinal, counts)
            if mfma:
                assert counts["b_dma"] == 6144 // info["oc"] * 256 // (info["waves"] * 1024), (case, counts)
                assert counts["c_store"] == 6144 // info["oc"] * 32 * 2 // 1024, (case, counts)
                assert counts["a_load"] <= 8  # PyHIP can skip a fully masked row group.
            task_stats.append({"wave": wave_id, "task": ordinal, **counts})
        assert len(mfmas) == manifest["waves"][wave_id]["mfma"]
        waves.append({"id": wave_id, "file": path.name, "simd": wave["simd"], "slot": wave["slot"],
                      "begin": wave["begin"], "end": wave["end"], "tasks": len(mfmas) // manifest["mfma_per_task"],
                      "attemptedTasks": task + 1, "mfma": mfmas, "records": records})
    return manifest, sites, sources, isa, waves, sorted(events, key=lambda e: e["issue"]), task_stats


def analyze(capture, output):
    manifest, sites, sources, isa, waves, events, tasks = decode(capture)
    latency_path = HERE / "result_final/latency_summary.json"
    latency = json.loads(latency_path.read_text())
    ns_cycle = manifest["ns_per_cycle"]
    begin = min(w["begin"] for w in waves) // BIN * BIN
    end = math.ceil(max(w["end"] for w in waves) / BIN) * BIN
    starts = np.arange(begin, end, BIN)
    centers = starts + BIN / 2
    bins = {"cycle_begin": starts, "cycle_end": starts + BIN, "us_from_first_wave": (centers - begin) * ns_cycle / 1000}
    models, latencies_ns = {}, {}
    # Most classes have one latency. Metadata can use DMA/load/store as
    # indicated per event, without reclassifying it as matrix traffic.
    for model in (*PERCENTILES, "common_L"):
        values = {kind: latency["scenarios"][SCENARIOS[kind] if model != "common_L" else "dma_loaded_stream"]["corrected_ns"][model if model != "common_L" else "mean"] for kind in CLASSES}
        lengths = {kind: value / ns_cycle for kind, value in values.items()}
        counts = {kind: np.zeros(len(starts)) for kind in CLASSES}
        bandwidth = {kind: np.zeros(len(starts)) for kind in CLASSES}
        for latency_kind in CLASSES:
            selected = [e for e in events if e["latency_class"] == latency_kind]
            n, bw = fixed_model(selected, dict.fromkeys(CLASSES, lengths[latency_kind]), begin, end, ns_cycle)
            for kind in CLASSES:
                counts[kind] += n[kind]
                bandwidth[kind] += bw[kind]
        bins["inflight_uniform_L" if model == "common_L" else f"inflight_{model}"] = sum(counts.values())
        bins["uniform256_common_L_tbs" if model == "common_L" else f"uniform256_payload_tbs_{model}"] = sum(bandwidth.values()) * 256 / 1000
        for kind in CLASSES:
            bins[f"inflight_{kind}_{model}"] = counts[kind]
            bins[f"cu_{kind}_gbs_{model}"] = bandwidth[kind]
        models[model] = {"latency_att_cycles": lengths}
        latencies_ns[model] = values
    mfmas = [time for wave in waves for time in wave["mfma"]]
    first, last = min(mfmas), max(mfmas)
    interior = (centers >= first) & (centers <= last)
    selected = np.flatnonzero(interior)
    residency = sum(((centers >= w["begin"]) & (centers < w["end"])).astype(int) for w in waves)
    bins["wave_residency"] = residency
    bins["issued_vmem"] = np.histogram([event["issue"] for event in events], np.arange(begin, end + BIN, BIN))[0]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "inflight_16cycles.csv.gz", bins)
    with gzip.open(output / "vmem_events.jsonl.gz", "wt") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    context = {"sites": sites, "sources": sources, "isa": isa, "isaName": Path(manifest["isa"]).name,
               "waves": [{key: value for key, value in wave.items() if key not in ("records", "mfma")} for wave in waves],
               "waveInstructions": [wave["records"] for wave in waves]}
    with (output / "context.json.gz").open("xb") as stream:
        stream.write(gzip.compress(json.dumps(context, separators=(",", ":")).encode(), mtime=0))
    notes = ["One-CU instruction-payload fixed-L model, not measured HBM bandwidth or actual retire timestamps.",
             "Preserve all complete waves including empty CTAs; interior is the complete first/last-MFMA hull (no residency mask).",
             "Count each wave-level instruction once; its width/lane mask determines payload; scalar SMEM is excluded.",
             "Original measured ns latencies are proxies, not case-specific measurements; custom values are assumptions.",
             "Metadata DMA uses the B-DMA latency; metadata VGPR loads use A-load proxy; queue reset uses store L.",
             "A/C actual row masks unknown; capacity upper bounds retained, no substitution with DRAM transactions.",
             "Successful issue timestamps only; next-supply PC labels are not proof of causality."]
    report = {"status": "PASS", "schema": "moe-inflight-suite-v1", "candidate": manifest["candidate"],
        "family": manifest["family"], "capture": str(capture), "config": manifest["config"], "shape": manifest["shape"],
        "source_snapshot_sha256": manifest["kernel_sha256"], "capture_summary_sha256": sha(capture / "summary.json"),
        "range_cycles": [begin, end], "interior": [int(starts[selected[0]]), int(starts[selected[-1]] + BIN)],
        "interior_cycles": int(interior.sum() * BIN), "mfma_first_last_issue": [first, last], "bin_cycles": BIN,
        "att_clock": {"ns_per_cycle": ns_cycle, "gfx_ghz": 1 / ns_cycle}, "models": models, "latency_ns": latencies_ns,
        "common_L_att_cycles": models["common_L"]["latency_att_cycles"]["b_dma"], "target_TBs": 5,
        "vmem_count": len(events), "vmem_by_class": dict(Counter(e["kind"] for e in events)),
        "complete_waves": len(waves), "wave_tasks": sum(w["tasks"] for w in waves),
        "empty_waves": sum(w["tasks"] == 0 for w in waves), "resources": manifest["resources"],
        "latency_input": str(latency_path), "latency_input_sha256": sha(latency_path), "notes": notes}
    write_json(output / "summary.json", report)
    write_json(output / "task_checks.json", tasks)
    write_json(output / "verified.json", {"status": "PASS", "input_sha256": {str(path): sha(path) for path in (
        Path(__file__), capture / "summary.json", latency_path)},
        "protected_sha256": manifest["protected_sha256"],
        "output_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()}})
    print("SUITE_MODEL_PASS", json.dumps({key: report[key] for key in ("candidate", "complete_waves", "wave_tasks", "vmem_count", "vmem_by_class", "range_cycles", "interior")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.capture.resolve(), args.output.resolve())