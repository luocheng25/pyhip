"""Per-kernel PMC summary of the TP8 MoE on one GPU: clock, MFMA busy, HBM bytes, L2 atomics.

Usage: python pmc_kernels.py [pyhip|aiter] [tokens]   (runs rocprofv3 one pass per counter group)
"""

import collections
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
PASSES = [
    "GRBM_GUI_ACTIVE SQ_VALU_MFMA_BUSY_CYCLES SQ_BUSY_CYCLES SQ_WAVES",
    "FETCH_SIZE",
    "WRITE_SIZE",
    "TCC_HIT_sum TCC_MISS_sum",
    "TCC_ATOMIC_sum TCC_EA0_ATOMIC_sum",
]
KERNELS = {
    "pyhip": {"gate_up": "_gate_up_kernel", "down": "_down_direct_kernel", "sum": "_sum_kernel",
              "moe64 x8": "_moe64_kernel", "moe64 x4": "_moe64x4_kernel"},
    "aiter": {"aiter_1stage": "fmoe_bf16_blockscaleFp8"},
}


def workload(impl, tokens):
    import torch

    import pyhip

    pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
    sys.path.insert(0, os.path.dirname(here))
    import bench_moe_fp8_blockscale as b

    c = b.make_case("tp8", tokens)
    if "FUSED_LDS" in os.environ:  # LDS bytes per fused CTA, e.g. 40960 -> one CTA per CU (FLYDSL_RUNTIME_ENABLE_CACHE=0)
        from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels_m64
        kernels_m64.LDS64 = int(os.environ["FUSED_LDS"])
    for _ in range(4):
        (b.run_pyhip if impl == "pyhip" else b.run_aiter)(c)
    torch.cuda.synchronize()


def profile(impl, tokens):
    values = collections.defaultdict(lambda: collections.defaultdict(list))
    durations = collections.defaultdict(list)
    for counters in PASSES:
        out = tempfile.mkdtemp()
        subprocess.run(["rocprofv3", "--pmc", *counters.split(), "-d", out, "-o", "run", "--output-format", "csv",
                        "--", sys.executable, os.path.abspath(__file__), "--run", impl, str(tokens)],
                       env=dict(os.environ, HIP_VISIBLE_DEVICES="0"), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        files = glob.glob(os.path.join(out, "**", "*counter_collection.csv"), recursive=True)
        if not files:
            print("no output for", counters)
        for row in csv.DictReader(open(files[0])) if files else []:
            for name, pattern in KERNELS[impl].items():
                if pattern in row["Kernel_Name"]:
                    values[name][row["Counter_Name"]].append(float(row["Counter_Value"]))
                    durations[name].append((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3)
        shutil.rmtree(out, ignore_errors=True)

    for name in KERNELS[impl]:
        if not durations[name]:
            continue
        v = {k: sum(x) / len(x) for k, x in values[name].items()}
        us = sorted(durations[name])[len(durations[name]) // 2] if durations[name] else float("nan")
        line = [f"{name:12s} {us:8.1f} us"]
        if "GRBM_GUI_ACTIVE" in v:
            cycles = v["GRBM_GUI_ACTIVE"] / 8  # summed over 8 XCDs
            line.append(f"clk {cycles / us / 1e3:4.2f} GHz")
            line.append(f"MFMA busy {v['SQ_VALU_MFMA_BUSY_CYCLES'] / (304 * 4) / cycles * 100:4.1f}%")
        for key, label in (("FETCH_SIZE", "HBM read"), ("WRITE_SIZE", "write")):
            if key in v:
                gb = v[key] * 1024 / 1e9
                line.append(f"{label} {gb:5.3f} GB ({gb * 1e3 / us:4.2f} TB/s)")
        if "TCC_HIT_sum" in v:
            line.append(f"L2 hit {v['TCC_HIT_sum'] / max(1.0, v['TCC_HIT_sum'] + v['TCC_MISS_sum']) * 100:4.1f}%")
        if v.get("TCC_ATOMIC_sum"):
            line.append(f"atomics L2 {v['TCC_ATOMIC_sum']:.3g} / EA {v.get('TCC_EA0_ATOMIC_sum', 0):.3g}")
        print("  ".join(line))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run"]:
        workload(sys.argv[2], int(sys.argv[3]))
    else:
        profile(sys.argv[1] if len(sys.argv) > 1 else "pyhip", int(sys.argv[2]) if len(sys.argv) > 2 else 32768)
