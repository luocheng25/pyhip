"""How many 512-thread / 32 KB-LDS workgroups are co-resident on a CU-masked stream (MI300X)?

Each WG increments an arrival counter, then waits (up to 2 s) until all `grid` WGs have arrived; the kernel
reports how many it saw. python residency_probe.py
"""

import ctypes
import os
import subprocess
import tempfile

import torch

SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>
__global__ void __launch_bounds__(512) arrive(int* ctr, int* seen, int grid) {
    __shared__ int pad[8192];  // 32 KiB, like sum_ar_cols' LDS ring
    pad[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(ctr, 1);
        uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        int v = 0;
        while ((v = __hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < grid &&
               __builtin_amdgcn_s_memrealtime() - t0 < 200000000ull)
            __builtin_amdgcn_s_sleep(1);
        atomicMax(seen, v + pad[blockIdx.x & 511] * 0);
    }
}
extern "C" void launch(int* ctr, int* seen, int grid, hipStream_t s) { arrive<<<grid, 512, 0, s>>>(ctr, seen, grid); }
"""
so = os.path.join(tempfile.gettempdir(), "residency_probe.so")
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


side = torch.cuda.Stream()
with torch.cuda.stream(side):
    for ncu in (48, 304):
        s = masked_stream(range(ncu))
        for grid in sorted({ncu, ncu - 8, ncu * 3 // 4, ncu // 2}, reverse=True):
            ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
            torch.cuda.synchronize()
            lib.launch(ctypes.c_void_p(ctr.data_ptr()), ctypes.c_void_p(ctr.data_ptr() + 4), grid, s)
            hip.hipStreamSynchronize(s)
            print(f"mask {ncu:3d} CUs, grid {grid:3d}: max co-resident WGs seen {int(ctr[1])}", flush=True)
