"""Write bandwidth of Y store patterns of the TP8 down (rows x 4096 BF16, tasks = 256-row block x 1024 cols).

Usage: python store_probe.py
"""

import ctypes
import os
import subprocess
import tempfile

import torch

here = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(tempfile.gettempdir(), "store_probe.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "store_probe.hip"), "-o", so])
lib = ctypes.CDLL(so)
rows = 293888
tasks = rows // 256 * 4
y = torch.empty(rows * 4096, dtype=torch.bfloat16, device="cuda")
stream = torch.cuda.current_stream().cuda_stream
names = ["mfma 64B/row aligned", "mfma 64B/row rotated", "4 rows x 256B", "2 rows x 512B", "mfma pairs"]
for kind, name in enumerate(names):
    for grid in (608, tasks):
        def run():
            lib.launch(ctypes.c_void_p(y.data_ptr()), tasks, kind, grid, ctypes.c_void_p(stream))
        for _ in range(3):
            run()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            run()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 100
        print(f"{name:22s} grid {grid:5d}: {us:7.1f} us {rows * 8192 / us / 1e6:5.2f} TB/s")
