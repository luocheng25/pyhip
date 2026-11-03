"""Does a VMEM instruction issued with EXEC = 0 still count in vmcnt on gfx942? (down_atomic.py relies on it.)

python execz_vmcnt.py
A cold global load, then 20 global atomics with EXEC = 0, then s_waitcnt vmcnt(20): 'loaded' for every block means
the masked atomics were counted (the wait covered the load); the control without atomics must show 'sentinel'.
"""

import ctypes
import os
import subprocess
import tempfile

import torch

here = os.path.dirname(os.path.abspath(__file__))
so = os.path.join(tempfile.gettempdir(), "execz_vmcnt.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "execz_vmcnt.hip"), "-o", so])
lib = ctypes.CDLL(so)
n = 256 << 20  # 1 GB of u32: each block loads a cold line
src = torch.full((n,), 0x1234, dtype=torch.int32, device="cuda")
dst = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
blocks = 1000
stream = torch.cuda.current_stream().cuda_stream
for zero_exec in (0, 1):
    res = torch.zeros(blocks, dtype=torch.int32, device="cuda")
    evict = torch.empty(512 << 20, dtype=torch.int8, device="cuda")
    evict.fill_(1)
    del evict
    lib.launch(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(res.data_ptr()),
               blocks, n // blocks, zero_exec, ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    r = res.cpu()
    print(f"{'20 EXEC=0 atomics' if zero_exec else 'control (no atomics)'}: loaded {int((r == 0x1234).sum())}, "
          f"sentinel {int((r == 0xDEAD).sum())} of {blocks}; atomic target sum {int(dst.sum())}")
