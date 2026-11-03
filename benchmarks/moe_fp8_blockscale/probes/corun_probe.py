"""Can the TOPK output path of a 1-stage MoE overlap with MFMA work of a co-resident WG? (MI300X)

Each CU runs exactly 2 WGs: one MFMA role (optionally reading a 1 GB stream or a MALL-resident
128 MB buffer with 1 or 4 reads in flight per wave), one output role over all real top-8 routes:
row-contiguous BF16 atomics into OUT[T, 4096] (default), or with "store" the plain Y row writes of
the 2-stage down kernel (16 B per lane, full rows, sorted order). Two output schedules:
  burst: as fast as possible; overlap = (t_mfma + t_out - t_both) / min(t_mfma, t_out)
  paced: spread evenly over the MFMA role's duration (steady state of a 1-stage kernel);
         exposed = t_both - t_mfma, relative to the stand-alone burst time t_out.

Usage: python corun_probe.py [tokens] [iters] [load_every list, e.g. 0,16,8] [store]
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import torch

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as m  # noqa: E402

so = os.path.join(tempfile.gettempdir(), "corun_probe.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "corun_probe.hip"), "-o", so])
lib = ctypes.CDLL(so)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
STORE = int(sys.argv[4:5] == ["store"])
c = b.make_case("tp8", T)
sid, _, _, nv = m.moe_sorting(c["ids"], c["wts"], 256, 4096)
rows = int(nv[0])
real = int(((sid[:rows] & 0xFFFFFF) < T).sum())
stream = torch.cuda.current_stream().cuda_stream
out = torch.zeros((rows if STORE else T) * 4096, dtype=torch.bfloat16, device="cuda")
ctr = torch.zeros(1025, dtype=torch.int32, device="cuda")
clk = torch.zeros(2, dtype=torch.int64, device="cuda")
buf = torch.ones(1 << 30, dtype=torch.uint8, device="cuda")
p = ctypes.c_void_p


def run(mode, load_every, footprint, depth, period=0):
    lib.launch(p(out.data_ptr()), p(sid.data_ptr()), rows, T, p(ctr.data_ptr()), p(buf.data_ptr()),
               ctypes.c_uint32(footprint // 16 - 1), ITERS, load_every, mode, period, depth, STORE, None,
               p(clk.data_ptr()), p(stream))


def timed(*args):
    for _ in range(3):
        run(*args)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    clk.zero_()
    start.record()
    for _ in range(10):
        run(*args)
    end.record()
    torch.cuda.synchronize()
    timed.ghz = float(clk[0]) / max(1, float(clk[1])) * 0.1  # s_memrealtime runs at 100 MHz
    return start.elapsed_time(end) * 100


out.zero_()
run(3, 0, 1 << 30, 1)
torch.cuda.synchronize()
if os.environ.get("CORUN_ONE"):  # "mode,load_every,footprint_mb,depth,paced": 12 launches for a profiler
    mode, load_every, mb, depth, paced = map(int, os.environ["CORUN_ONE"].split(","))
    period = int(timed(1, load_every, mb << 20, depth) * 100 / (real / 1216)) if paced else 0
    for _ in range(12):
        run(mode, load_every, mb << 20, depth, period)
    torch.cuda.synchronize()
    sys.exit(0)
arrivals = ctr[1:]
per_cu = torch.bincount(arrivals[arrivals > 0].cpu())
expect = real * 4096  # 1.0 per element of every real route, stored once or added once
ok = abs(float(out.float().sum()) - expect) < 1e-3 * expect
print(f"T {T} real routes {real}: {'Y stores' if STORE else 'BF16 atomics'} {real * 8192 / 1e9:.3f} GB  "
      f"CUs by WG count {per_cu.tolist()}  sum {'ok' if ok else 'BAD'}")
flop = 304 * 4 * ITERS * 8 * 16 * 16 * 32 * 2
cases = [(0, 1 << 30, 1), (8, 1 << 30, 1), (8, 128 << 20, 1), (8, 1 << 30, 4), (8, 128 << 20, 4),
         (4, 1 << 30, 4)]
if len(sys.argv) > 3:  # e.g. "0,24,16,12,8": reads every N iterations, depth 4, 1 GB and 128 MB footprints
    cases = [(n, f, 4) for n in map(int, sys.argv[3].split(",")) for f in ((1 << 30, 128 << 20) if n else (1 << 30,))]
for load_every, footprint, depth in cases:
    t_m = timed(1, load_every, footprint, depth)
    g_m = timed.ghz
    t_a, t_b = timed(2, load_every, footprint, depth), timed(3, load_every, footprint, depth)
    period = int(t_m * 100 / (real / 1216))  # 100 MHz ticks per real row per wave (1216 output waves)
    t_p = timed(3, load_every, footprint, depth, period)
    g_p = timed.ghz
    read = 304 * 4 * (ITERS // load_every) * 1024 if load_every else 0
    src = f"{read / 1e9:4.2f} GB from {footprint >> 20:4d} MB, depth {depth}" if read else "no reads" + " " * 25
    print(f"MFMA role {src}: mfma {t_m:7.1f} us ({flop / t_m / 1e6:4.0f} TF, {g_m:4.2f} GHz)  out alone {t_a:6.1f} us "
          f"({real * 8192 / t_a / 1e6:4.2f} TB/s)  burst: both {t_b:7.1f} us, "
          f"overlap {(t_m + t_a - t_b) / min(t_m, t_a) * 100:3.0f}%  "
          f"paced: both {t_p:7.1f} us ({g_p:4.2f} GHz), exposed {t_p - t_m:6.1f} us ({(t_p - t_m) / t_a * 100:3.0f}%)")
