"""Throughput of atomic TOPK accumulation patterns vs the plain Y store (MI300X).

Usage: python atomic_probe.py [tokens]   (TP8 shape, real top-8 routing from bench_moe_fp8_blockscale)
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

so = os.path.join(tempfile.gettempdir(), "atomic_probe.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "atomic_probe.hip"), "-o", so])
lib = ctypes.CDLL(so)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
c = b.make_case("tp8", T)
sid, _, _, nv = m.moe_sorting(c["ids"], c["wts"], 256, 4096)
rows = int(nv[0])
stream = torch.cuda.current_stream().cuda_stream
names = ["row_bf16", "mfma_bf16", "swap_bf16", "row_f32", "row_store", "tile64_bf16", "tile128_bf16", "tile64x_bf16", "tile64_cgmaj", "tile64x_cgmaj", "tile64x_cgmaj8", "tile64x_cgmaj16", "tile64x_cg8"]
real = int(((sid[:rows] & 0xFFFFFF) < T).sum())
print(f"T {T} rows {rows} real routes {real}")
for kind, name in enumerate(names):
    dtype = torch.float32 if kind == 3 else torch.bfloat16
    out = torch.zeros((rows if kind == 4 else T) * 4096, dtype=dtype, device="cuda")
    for grid in (304 * 2, 304 * 8):
        def run():
            lib.launch(kind, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(sid.data_ptr()), rows, T, grid,
                       ctypes.c_void_p(stream))
        out.zero_()
        run()
        torch.cuda.synchronize()
        if kind != 4:
            total = float(out.float().sum())
            ok = abs(total - real * 4096) < 1e-3 * real * 4096
        else:
            ok = True
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            run()
        start.record()
        for _ in range(10):
            run()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 100
        logical = (rows if kind == 4 else real) * 4096 * 2
        print(f"{name:10s} grid {grid:5d}: {us:8.1f} us  {logical / us / 1e6:6.2f} TB/s (bf16-equivalent)"
              f"  {'ok' if ok else 'BAD SUM'}")

# H: atomics + the down kernel's A2 / W2 reads (interference)
seid = m.moe_sorting(c["ids"], c["wts"], 256, 4096)[2]
a2 = torch.randint(0, 255, (rows * 256,), dtype=torch.uint8, device="cuda")
w2 = torch.randint(0, 255, (256 * 4096 * 256,), dtype=torch.uint8, device="cuda")
sink = torch.zeros(16, dtype=torch.int32, device="cuda")
out = torch.zeros(T * 4096, dtype=torch.bfloat16, device="cuda")
for grid in (304 * 2, 304 * 8):
    def run():
        lib.launch_reads(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(sid.data_ptr()),
                         ctypes.c_void_p(seid.data_ptr()), ctypes.c_void_p(a2.data_ptr()),
                         ctypes.c_void_p(w2.data_ptr()), ctypes.c_void_p(sink.data_ptr()), rows, T, grid,
                         ctypes.c_void_p(stream))
    for _ in range(3):
        run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        run()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 100
    print(f"tile64x_cgmaj8+reads grid {grid:5d}: {us:8.1f} us  {real * 4096 * 2 / us / 1e6:6.2f} TB/s")
