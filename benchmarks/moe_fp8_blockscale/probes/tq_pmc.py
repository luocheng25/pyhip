"""PMC of tingqli's fused A8W4 bm64 kernel (TP8 inter 256) on GPU 0, same counters as probes/pmc_kernels.py.

TQ_AITER=/root/workspace/tq_aiter PYTHONPATH=$TQ_AITER:$TQ_AITER/doc-mimo/experiments/tools/pyhip_shim \
    python tq_pmc.py [tokens]          (rocprofv3 on PATH; G2_NO_ATOMIC=1 for the no-atomic kernel)
"""
import collections
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

PASSES = ["GRBM_GUI_ACTIVE SQ_VALU_MFMA_BUSY_CYCLES", "FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum TCC_MISS_sum"]
PATTERN = "fmoe_bm64"


def workload(tokens):
    import torch
    TQ = os.path.join(os.environ.get("TQ_AITER", "/root/workspace/tq_aiter"), "doc-mimo")
    sys.path.insert(0, f"{TQ}/experiments/moe_2k_vs_1k")
    import bench as BB
    from aiter import dtypes
    from aiter.fused_moe import fused_topk
    W = BB.make_weights(256)
    torch.manual_seed(tokens)
    x = torch.randn((tokens, 4096), dtype=dtypes.bf16, device="cuda")
    tw, ti = fused_topk(x, torch.randn((tokens, 256), dtype=dtypes.bf16, device="cuda"), 8, True)
    for _ in range(4):
        BB.F.fused_moe_a8w4_bm64(x, W, tw, ti, b_cache_mod=1, waves_per_eu=2)
    torch.cuda.synchronize()


def profile(tokens):
    values, durations = collections.defaultdict(list), []
    for counters in PASSES:
        out = tempfile.mkdtemp()
        subprocess.run(["rocprofv3", "--pmc", *counters.split(), "-d", out, "-o", "run", "--output-format", "csv",
                        "--", sys.executable, os.path.abspath(__file__), "--run", str(tokens)],
                       env=dict(os.environ, HIP_VISIBLE_DEVICES="0"), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        files = glob.glob(os.path.join(out, "**", "*counter_collection.csv"), recursive=True)
        for row in csv.DictReader(open(files[0])) if files else []:
            if PATTERN in row["Kernel_Name"]:
                values[row["Counter_Name"]].append(float(row["Counter_Value"]))
                durations.append((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3)
        shutil.rmtree(out, ignore_errors=True)
    v = {k: sum(x) / len(x) for k, x in values.items()}
    us = sorted(durations)[len(durations) // 2]
    cycles = v["GRBM_GUI_ACTIVE"] / 8
    print(f"tq fused {us:8.1f} us  clk {cycles / us / 1e3:4.2f} GHz  "
          f"MFMA busy {v['SQ_VALU_MFMA_BUSY_CYCLES'] / (304 * 4) / cycles * 100:4.1f}%  "
          f"L2-miss read {v['FETCH_SIZE'] * 1024 / 1e9:5.3f} GB  write {v['WRITE_SIZE'] * 1024 / 1e9:5.3f} GB  "
          f"L2 hit {v['TCC_HIT_sum'] / (v['TCC_HIT_sum'] + v['TCC_MISS_sum']) * 100:4.1f}%")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run"]:
        workload(int(sys.argv[2]))
    else:
        profile(int(sys.argv[1]) if len(sys.argv) > 1 else 32768)
