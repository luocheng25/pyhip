"""MALL (Infinity Cache) residency: write a buffer then read it back, for sizes 32 MB .. 2.4 GB.

Usage: python mall_probe.py
"""

import ctypes
import os
import subprocess
import tempfile

import torch

here = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(tempfile.gettempdir(), "mall_probe.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "mall_probe.hip"), "-o", so])
lib = ctypes.CDLL(so)
lib.launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
stream = torch.cuda.current_stream().cuda_stream
big = torch.empty(2400 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros(4, dtype=torch.int32, device="cuda")
grid = 304 * 16
for mb in (32, 64, 128, 192, 256, 384, 600, 2400):
    nbytes = mb << 20
    reps = max(1, 2400 // mb)

    def w(off):
        lib.launch(0, big.data_ptr() + off, nbytes, sink.data_ptr(), grid, stream)

    def r(off):
        lib.launch(1, big.data_ptr() + off, nbytes, sink.data_ptr(), grid, stream)
    times = {}
    for name, seq in (("write", lambda: [w(0) for _ in range(reps)]), ("read", lambda: [r(0) for _ in range(reps)]),
                      ("w+r same", lambda: [(w(0), r(0)) for _ in range(reps)])):
        seq()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        seq()
        e.record()
        torch.cuda.synchronize()
        times[name] = s.elapsed_time(e) * 1000 / reps
    print(f"{mb:5d} MB: write {times['write']:8.1f} us ({nbytes / times['write'] / 1e6:5.2f} TB/s)  "
          f"read {times['read']:8.1f} us ({nbytes / times['read'] / 1e6:5.2f} TB/s)  "
          f"w+r {times['w+r same']:8.1f} us ({2 * nbytes / times['w+r same'] / 1e6:5.2f} TB/s)")
