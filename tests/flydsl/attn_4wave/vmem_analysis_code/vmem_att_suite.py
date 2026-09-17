# SPDX-License-Identifier: MIT
"""Capture the seven maintained down candidates without changing their tests.

One fresh process per case; 23 exact profile launches, native sorting and the
existing reference/tolerances. Reuse of a previous capture is explicit in the
suite manifest, never a renamed capture. No clock/power/CU-mask changes.
"""

import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

from vmem_att_capture import MAIN, ROOT, TOOLS, protected_paths, sha, table, write_json


CASES = {
    "pyhip": {"slug": "pyhip", "family": "pyhip", "pattern": r".*moe_gemm_8wave_down.*",
              "source": "src/contrib/moe_gemm_8wave.py", "block_m": 256, "block_n": 64, "waves": 8, "oc": 1, "persistent": True},
    "flydsl_bn32": {"slug": "flydsl_bn32", "family": "legacy", "pattern": r"^flydsl_moe_gemm_8wave_down_kernel(_[0-9]+)?$",
                     "source": "tests/flydsl/moe_8w_down/moe_8wave_down.py", "block_m": 256, "block_n": 32, "waves": 8, "oc": 1, "persistent": True},
    "flydsl_bn64": {"slug": "flydsl_bn64", "family": "legacy", "pattern": r"^flydsl_moe_gemm_8wave_down_kernel(_[0-9]+)?$",
                     "source": "tests/flydsl/moe_8w_down/moe_8wave_down.py", "block_m": 256, "block_n": 64, "waves": 8, "oc": 1, "persistent": True},
    "256x128 persist": {"slug": "m256_persist", "family": "m256", "pattern": r"^moe_down_8stage_kernel(_[0-9]+)?$",
                        "source": "tests/flydsl/moe_8w_down/moe_multistage_down.py", "block_m": 256, "block_n": 128, "waves": 8, "oc": 4, "persistent": True},
    "256x128": {"slug": "m256", "family": "m256", "pattern": r"^moe_down_8stage_kernel(_[0-9]+)?$",
                "source": "tests/flydsl/moe_8w_down/moe_multistage_down.py", "block_m": 256, "block_n": 128, "waves": 8, "oc": 4, "persistent": False},
    "128x128": {"slug": "m128", "family": "m128", "pattern": r"^moe_down_m128_kernel(_[0-9]+)?$",
                "source": "tests/flydsl/moe_8w_down/moe_multistage_down_m128.py", "block_m": 128, "block_n": 128, "waves": 4, "oc": 8, "persistent": False},
    "128x128 persist": {"slug": "m128_persist", "family": "m128", "pattern": r"^moe_down_m128_kernel(_[0-9]+)?$",
                        "source": "tests/flydsl/moe_8w_down/moe_multistage_down_m128.py", "block_m": 128, "block_n": 128, "waves": 4, "oc": 8, "persistent": True},
}


def verify_sources(output):
    manifest = json.loads((output / "sources.json").read_text())
    for path, digest in manifest["protected_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert (MAIN / "try").exists() == manifest["try_existed"]
    return manifest


def workload(case, output):
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "5" and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    sys.path.insert(0, str(MAIN))
    import torch
    import test_blockscaled as test

    props = torch.cuda.get_device_properties(0)
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    bus = ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(bus, 32, 0) == 0
    assert bus.value.decode().lower() == "0000:85:00.0"
    assert props.gcnArchName.startswith("gfx950") and props.multi_processor_count == 256
    assert tuple(test.CANDIDATES) == tuple(CASES)
    assert test.SORT_BLOCK_M[case] == CASES[case]["block_m"]
    print("SUITE_WORKLOAD_START", case, flush=True)
    results = test.run_test(candidates=[case], profile=True, reduce_output=False)
    assert len(results) == 1
    result = results[0]
    assert result["candidate"] == case and result["status"] == "PASS" and result["mismatch_count"] == 0
    for key in ("block_m", "block_n", "num_oc_splits", "num_waves", "sort_block_m"):
        expected = CASES[case][{"num_oc_splits": "oc", "num_waves": "waves", "sort_block_m": "block_m"}.get(key, key)]
        assert result["config"][key] == expected, (key, result["config"])
    verify_sources(output)
    write_json(output / "profile.json", {"candidate": case, "result": result, "config": result["config"],
        "shape": {"tokens": 16384, "n": 6144, "k": 256, "experts": 384, "topk": 8}, "seed": 1234,
        "source_unchanged": True, "profile_launches": 23, "baseline_launches_of_target": int(case == "pyhip"),
        "device": {"arch": props.gcnArchName, "cus": props.multi_processor_count, "pci_bdf": bus.value.decode()}})
    print("SUITE_WORKLOAD_PASS", case, flush=True)


def capture(case, output):
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "5" and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    output.mkdir(parents=True, exist_ok=False)
    info = CASES[case]
    paths = protected_paths() + [MAIN / "analysis/inflight_m256_persist_20260914/timeline_release/inflight_timeline.html",
                                 MAIN / "analysis/inflight_m256_persist_20260914/model_second/inflight_16cycles.csv.gz",
                                 ROOT / "src/core/asmjit.py"]
    sources = [path for path in paths if path.suffix == ".py"] + [Path(__file__)]
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    iteration = 22 if case == "pyhip" else 21
    yaml = f"""jobs:
  - advanced_thread_trace: true
    kernel_trace: true
    att_library_path:
      - /host_lc/pyhip/tests/gluon/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib
    att_target_cu: 0
    att_shader_engine_mask: '0x1'
    att_simd_select: '0xf'
    att_buffer_size: '0x10000000'
    kernel_include_regex: '{info['pattern']}'
    kernel_iteration_range: '[{iteration}]'
    output_format: [csv]
    output_file: trace
"""
    with (output / "att.yaml").open("x") as stream:
        stream.write(yaml)
    command = ["/opt/rocm/bin/rocprofv3", "-i", str(output / "att.yaml"), "-d", str(output / "capture"),
               "--", sys.executable, "-B", str(Path(__file__)), "--workload", "--candidate", case, "--output", str(output)]
    environment = {"HIP_VISIBLE_DEVICES": "5", "PYTHONDONTWRITEBYTECODE": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
                   "FLYDSL_DEBUG_ENABLE_DEBUG_INFO": "1", "FLYDSL_DUMP_IR": "1", "FLYDSL_DUMP_DIR": str(output / "ir")}
    write_json(output / "sources.json", {"candidate": case, "command": command, "environment": environment,
        "protected_sha256": {str(path): sha(path) for path in paths},
        "source_sha256": {str(path): sha(path) for path in sources},
        "archive_sha256": sha(output / "sources.tar.gz"), "try_existed": (MAIN / "try").exists()})
    print("SUITE_CAPTURE_START", case, output, flush=True)
    with (output / "capture.log").open("x") as stream:
        subprocess.run(command, env={**os.environ, **environment}, stdout=stream, stderr=subprocess.STDOUT, check=True)
    validate(case, output)


def validate(case, output):
    manifest = verify_sources(output)
    info = CASES[case]
    profile = json.loads((output / "profile.json").read_text())
    assert profile["candidate"] == case and profile["result"]["status"] == "PASS"
    directory = output / "capture"
    targets = [row for row in table(directory / "trace_kernel_trace.csv") if re.fullmatch(info["pattern"], row["Kernel_Name"])]
    iteration = 22 if case == "pyhip" else 21
    assert len(targets) == 23 + int(case == "pyhip"), (case, len(targets))
    target = targets[iteration - 1]
    assert int(target["Workgroup_Size_X"]) == info["waves"] * 64
    dispatch = int(target["Dispatch_Id"])
    agent = next(row for row in table(directory / "trace_agent_info.csv") if "Agent " + row["Logical_Node_Id"] == target["Agent_Id"])
    location = int(agent["Location_Id"])
    bdf = f'{int(agent["Domain"]):04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7}'
    assert bdf == "0000:85:00.0" and agent["Name"] == "gfx950"
    assert int(agent["Cu_Count"]) == 256 and int(agent["Num_Xcc"]) == 8
    indices = list(directory.glob("ui_output_*/filenames.json"))
    assert len(indices) == 1 and indices[0].parent.name.endswith(f"_dispatch_{dispatch}")
    ui = indices[0].parent
    code = json.loads((ui / "code.json").read_text())["code"]
    sites = {int(row[2]): str(row[0]).strip() for row in code}
    raw = list(directory.glob("*.att"))
    assert len(raw) == 1 and raw[0].stat().st_size and raw[0].name.endswith(f"_shader_engine_0_{dispatch}.att")
    objects = [directory / f"trace_gfx950_code_object_id_{number}.out" for number in sorted({int(row[4]) for row in code})]
    assert all(path.is_file() and path.stat().st_size for path in objects)
    mfma_task = 2 * (6144 // info["oc"]) * 256 * (info["block_m"] // info["waves"]) // (2 * 16 * 16 * 128)
    waves = []
    for path in sorted(ui.glob("se*_sm*_sl*_wv*.json")):
        record = json.loads(path.read_text())
        wave = record["wave"]
        assert record["num_insts"] == record["num_stitched"] == len(wave["instructions"]), path
        assert path.name.startswith("se0_") and wave["cu"] == 0 and wave["end"] > wave["begin"]
        counts = Counter()
        for attempt, category, stall, duration, pc in wave["instructions"]:
            assert 0 <= stall <= duration
            counts[sites[int(pc)].split()[0]] += 1
        mfma = sum(count for op, count in counts.items() if op.startswith("v_mfma_"))
        assert mfma % mfma_task == 0, (path.name, mfma, mfma_task)
        waves.append({"file": path.name, "simd": wave["simd"], "slot": wave["slot"], "begin": wave["begin"],
                      "end": wave["end"], "instructions": record["num_insts"], "mfma": mfma,
                      "tasks": mfma // mfma_task, "opcounts": dict(counts)})
    assert waves and sum(w["mfma"] for w in waves) > 0 and {w["simd"] for w in waves} == {0, 1, 2, 3}
    snapshots = list(ui.glob("source_*_" + Path(info["source"]).name))
    source_path = ROOT / info["source"]
    if snapshots:
        assert len(snapshots) == 1 and sha(snapshots[0]) == sha(source_path)
    else:
        snapshot = output / "kernel_source.py"
        with source_path.open("rb") as src, snapshot.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        snapshots = [snapshot]
    isa = list((output / "ir").glob("*/[0-9]*_final_isa.s"))
    if case == "pyhip":
        candidates = [path for path in Path("/root/.pyhip").glob("moe_gemm_8wave_down*.co") if all(value in path.name for value in (
            "is_output_over_4GB=False-", "AB_dtype=fp8-", "wg_M=256-", "wg_N=64-", "NUM_EXPERTS=384-", "OC=6144-", "IC=256-", "num_oc_splits=1-", "TOPK=8-"))]
        assert len(candidates) == 1, candidates
        cached = candidates[0]
        assert len(objects) == 1 and sha(cached) == sha(objects[0]), "captured PyHIP binary must match cache source"
        for extension, destination in ((".s", "pyhip_final_isa.s"), (".cpp", "pyhip_generated.cpp")):
            with cached.with_suffix(extension).open("rb") as src, (output / destination).open("xb") as dst:
                shutil.copyfileobj(src, dst)
        isa = [output / "pyhip_final_isa.s"]
    assert len(isa) == 1, isa
    text = isa[0].read_text()
    resources = {}
    for key in ("vgpr_count", "sgpr_count", "agpr_count", "vgpr_spill_count", "sgpr_spill_count",
                "group_segment_fixed_size", "private_segment_fixed_size"):
        found = re.findall(rf"\.{key}:\s*(\d+)", text)
        resources[key] = int(found[0]) if len(found) == 1 else None
    realtime = json.loads((ui / "realtime.json").read_text())
    (a, ta), (b, tb) = realtime["SE0"]
    ns_cycle = (tb - ta) * (1e9 / realtime["metadata"]["frequency"]) / (b - a)
    assert 0 < ns_cycle < 10
    paths = [*raw, *objects, *[path for path in ui.rglob("*") if path.is_file()], *isa, *snapshots,
             output / "profile.json", output / "sources.json", output / "sources.tar.gz", output / "att.yaml",
             directory / "trace_kernel_trace.csv", directory / "trace_agent_info.csv"]
    if case == "pyhip":
        paths.append(output / "pyhip_generated.cpp")
    report = {"status": "PASS", "schema": "moe-att-suite-v1", "candidate": case, "family": info["family"],
        "slug": info["slug"], "config": {**profile["config"], "persistent": info["persistent"]}, "shape": profile["shape"],
        "profile_iteration": 21, "profiler_iteration": iteration, "target_dispatches": len(targets),
        "kernel_name": target["Kernel_Name"], "dispatch": dispatch, "queue": int(target["Queue_Id"]),
        "grid_workgroups": int(target["Grid_Size_X"]) // int(target["Workgroup_Size_X"]),
        "pci_bdf": bdf, "cus": 256, "xcds": 8, "ns_per_cycle": ns_cycle,
        "instrumented_dispatch_us": (int(target["End_Timestamp"]) - int(target["Start_Timestamp"])) / 1000,
        "complete_waves": len(waves), "empty_waves": sum(w["tasks"] == 0 for w in waves),
        "wave_tasks": sum(w["tasks"] for w in waves), "mfma_per_task": mfma_task,
        "waves": waves, "resources": resources, "ui": str(ui.relative_to(output)),
        "isa": str(isa[0].relative_to(output)), "source_snapshot": str(snapshots[0].relative_to(output)),
        "kernel_sha256": sha(source_path), "source_unchanged": True, "check": profile["result"]["status"],
        "raw_sha256": {str(path.relative_to(output)): sha(path) for path in paths},
        "protected_sha256": manifest["protected_sha256"],
        "notes": ["Same maintained test,23 profile launches; correctness/tolerances unchanged.",
                  "PyHIP has one earlier reference launch, so profiler iteration22 selects profile iteration21.",
                  "Include complete empty/retired/independent waves; do not impose all-waves-resident mask.",
                  "ATT instrumented time is not ordinary timing; no HBM claim without same-dispatch PMC."]}
    write_json(output / "summary.json", report)
    print("SUITE_CAPTURE_PASS", json.dumps({key: report[key] for key in ("candidate", "kernel_name", "dispatch", "complete_waves", "empty_waves", "wave_tasks", "mfma_per_task", "resources", "ns_per_cycle")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=CASES)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--workload", action="store_true")
    group.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (workload if args.workload else validate if args.validate else capture)(args.candidate, args.output.resolve())