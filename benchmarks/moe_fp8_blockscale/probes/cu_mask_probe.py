"""CU-masked streams on MI300X: which CUs does a mask bit select, and do two disjoint masked streams overlap?

python cu_mask_probe.py
"""

import ctypes
import os
import subprocess
import tempfile

import torch

SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>
__global__ void where(int* out, int spin) {
    uint32_t hw, xcc;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
    if (threadIdx.x == 0) out[blockIdx.x] = ((xcc & 7) << 8) | ((hw >> 8) & 0xFF);
    uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (__builtin_amdgcn_s_memrealtime() - t0 < (uint64_t)spin) __builtin_amdgcn_s_sleep(2);
}
extern "C" void launch(int* out, int grid, int spin, hipStream_t s) { where<<<grid, 64, 0, s>>>(out, spin); }
"""
so = os.path.join(tempfile.gettempdir(), "cu_mask_probe.so")
src = so.replace(".so", ".hip")
open(src, "w").write(SRC)
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC", src, "-o", so])
lib = ctypes.CDLL(so)
hip = ctypes.CDLL("libamdhip64.so")


def masked_stream(cus):
    words = (ctypes.c_uint32 * 10)()
    for c in cus:
        words[c // 32] |= 1 << (c % 32)
    s = ctypes.c_void_p()
    assert hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), 10, words) == 0
    return s


def run(stream, grid, spin_ticks):
    out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
    lib.launch(ctypes.c_void_p(out.data_ptr()), grid, spin_ticks, stream)
    return out


torch.cuda.synchronize()
for name, cus in (("bits 0-7", range(8)), ("bits 0-37", range(38)), ("bits 0-47", range(48)),
                  ("bits 48-303", range(48, 304)), ("every 8th bit", range(0, 304, 8))):
    s = masked_stream(cus)
    out = run(s, 2048, 2000)
    hip.hipStreamSynchronize(s)
    keys = out.cpu().tolist()
    per_xcd = [len({k for k in keys if k >> 8 == x}) for x in range(8)]
    print(f"{name:14s}: {len(set(keys)):3d} distinct CUs, per XCD {per_xcd}")

# concurrency: 1 ms spin per WG on two disjoint XCD-balanced masks (bit i -> XCD i % 8). Nothing may run on the
# legacy default stream meanwhile: it synchronizes with every blocking stream (the masked ones are blocking).
a, b = masked_stream(range(48, 304)), masked_stream(range(48))
outs = {id(s): torch.empty(1216, dtype=torch.int32, device="cuda") for s in (a, b)}
side = torch.cuda.Stream()
with torch.cuda.stream(side):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for label, streams in (("A alone (256 CUs)", [a]), ("B alone (48 CUs)", [b]), ("A and B", [a, b])):
        torch.cuda.synchronize()
        start.record(side)
        ev = []
        for s in streams:
            ext = torch.cuda.ExternalStream(s.value)
            ext.wait_event(start)
            lib.launch(ctypes.c_void_p(outs[id(s)].data_ptr()), 256 if s is a else 48, 100000, s)
            ev.append(ext.record_event())
        for e in ev:
            side.wait_event(e)
        end.record(side)
        torch.cuda.synchronize()
        print(f"{label}: {start.elapsed_time(end) * 1e3:.0f} us (one WG per CU, each spins 1000 us)")
