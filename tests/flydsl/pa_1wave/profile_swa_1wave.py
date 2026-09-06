"""Capture warmed SWA dispatches and/or native ISA resources; not a timer."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch
from flydsl.utils import env

from test_swa_1wave import HERE, MHA, PA8, PagedAttention, make_call, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("1wave", "4static", "4dynamic", "8static", "8persistent"), default="1wave")
    parser.add_argument("--dq", type=int, choices=(128, 192), default=192)
    parser.add_argument("--q", type=int, default=16384)
    parser.add_argument("--kv", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--query-tile", type=int, choices=(16, 32))
    parser.add_argument("--bn", type=int, choices=(16, 32, 64))
    parser.add_argument("--sink", type=int, choices=(0, 1), default=1)
    parser.add_argument("--lse", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if "gfx950" not in reference.GPU_ARCH:
        raise SystemExit("requires native gfx950")
    if args.candidate != "1wave" and (args.query_tile is not None or args.bn is not None or args.lse):
        parser.error("tile overrides and LSE apply only to 1wave")
    if args.dump_dir:
        if args.dump_dir.exists() and any(args.dump_dir.rglob("*_final_isa.s")):
            parser.error("use a fresh dump directory to avoid reading stale ISA")
        env.debug.dump_ir = True
        env.debug.dump_dir = str(args.dump_dir)
    case = reference.make_case((args.q,), (args.kv,), dq=args.dq, heads=args.heads,
                               window_left=args.window, has_sink=bool(args.sink), poison_tail=False)
    factories = {
        "1wave": (PagedAttention, {"query_tile": args.query_tile, "block_n": args.bn}, HERE / "swa_1wave.py"),
        "4static": (MHA, {}, HERE.parent / "pa_4wave/pa_prefill_4wave.py"),
        "4dynamic": (MHA, {"force_dynamic_schedule": True}, HERE.parent / "pa_4wave/pa_prefill_4wave.py"),
        "8static": (PA8, {}, HERE.parent / "pa_8wave/pa_8wave_950.py"),
        "8persistent": (PA8, {"persistent": True}, HERE.parent / "pa_8wave/pa_8wave_950.py"),
    }
    factory, options, source = factories[args.candidate]
    lse = torch.empty(args.q, args.heads, device="cuda", dtype=torch.float32) if args.lse else None
    call, out, kernel = make_call(case, factory, lse=lse, **options)
    record = {**vars(args), "gfx": reference.GPU_ARCH, "gpu": torch.cuda.get_device_name(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source": str(source.relative_to(HERE.parent)), "launches": 23, "selected_iterations": [21, 22, 23]}
    if args.candidate == "1wave":
        record.update(query_tile=kernel.query_tile, bn=kernel.block_n, threads=64,
                      grid=[args.heads, 1, (args.q + kernel.query_tile - 1) // kernel.query_tile])
    print("SWA_PROFILE", json.dumps(record, default=str), flush=True)
    for _ in range(23):
        call()
        torch.cuda.synchronize()
    ref, ref_lse = reference.run_torch(case, True)
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
    if lse is not None:
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)
    record["correct"] = True
    if args.dump_dir:
        files = list(args.dump_dir.rglob("*_final_isa.s"))
        if len(files) != 1:
            raise RuntimeError(f"expected one fresh ISA file, got {files}")
        isa = files[0].read_text()
        keys = ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size",
                "vgpr_spill_count", "sgpr_spill_count", "wavefront_size", "max_flat_workgroup_size")
        record["resources"] = {key: int(re.search(r"\." + key + r":\s*(\d+)", isa)[1]) for key in keys}
        accum_offset = int(re.search(r"\.amdhsa_accum_offset\s+(\d+)", isa)[1])
        record["resources"]["accum_offset"] = accum_offset
        record["resources"]["agpr_count"] = max(0, record["resources"]["vgpr_count"] - accum_offset)
        counts = Counter(re.findall(r"^\s+([vs]_[a-zA-Z0-9_]+)\b", isa, re.MULTILINE))
        record["mfma_static_sites"] = {name: count for name, count in counts.items() if "mfma" in name}
        record["barrier_static_sites"] = counts["s_barrier"]
        record["isa_sha256"] = hashlib.sha256(isa.encode()).hexdigest()
        record["isa_path"] = str(files[0])
        if args.candidate == "1wave":
            assert record["resources"]["wavefront_size"] == 64
            assert record["resources"]["max_flat_workgroup_size"] == 64
            assert record["resources"]["group_segment_fixed_size"] == 0
            assert all(record["resources"][key] == 0 for key in
                       ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"))
            assert record["barrier_static_sites"] == 0
            assert all("_16x16x" in name for name in record["mfma_static_sites"])
    if args.output:
        args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print("SWA_PROFILE_VALIDATED", json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()