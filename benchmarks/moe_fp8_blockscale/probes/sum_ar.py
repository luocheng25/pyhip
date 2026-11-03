"""Fused MoE top-k sum + two-shot all-reduce (sum_ar.hip) vs sum kernel + RCCL all-reduce, 8x MI300X.

torchrun --nproc_per_node 8 sum_ar.py [tokens...]   (env GRID=96, ITERS=20, DIAG=1,2,4,... for timing ablations,
                                                     INV_RANGE=n: cache-resident Y rows)

Every rank: Y [T*TOPK, H] bf16 route rows, INV [T, TOPK] (a random permutation), same token count.
Checks the fused output bit-exactly against: per-rank partial = bf16(fp32 sum over k), then fp32 sum over
ranks in rank order, bf16 -- for several epochs with fresh data. Times (slowest rank, median):
  sum      : pyhip _sum_kernel (one CTA per token, nt loads) -> out [T, H]
  rccl     : dist.all_reduce(out)
  sum+rccl : both back to back
  twoshot  : sum_ar mode 1 (all-reduce of a ready partial)
  fused    : sum_ar mode 0 (top-k sum + two-shot all-reduce in one kernel)
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as K  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M  # noqa: E402

H, NPES, TOPK = 4096, 8, 8
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
assert dist.get_world_size() == NPES

so = os.path.join(tempfile.gettempdir(), f"sum_ar_{rank}.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "sum_ar.hip"), "-o", so])
lib = ctypes.CDLL(so)
lib.sym_bytes.restype = lib.gather_offset.restype = ctypes.c_uint64
lib.sym_bytes.argtypes = lib.gather_offset.argtypes = [ctypes.c_int]
hip = ctypes.CDLL("libamdhip64.so")


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def check(err, what):
    assert err == 0, f"{what} failed: {err}"


class Symmetric:
    """One torch allocation per rank, mapped into every rank through HIP IPC."""

    def __init__(self, nbytes):
        self.storage = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        local = self.storage.data_ptr()
        base = ctypes.c_void_p()
        check(hip.hipPointerGetAttribute(ctypes.byref(base), 11, ctypes.c_void_p(local)), "range start")
        handle = Handle()
        check(hip.hipIpcGetMemHandle(ctypes.byref(handle), base), "hipIpcGetMemHandle")
        info = [None] * NPES
        dist.all_gather_object(info, (bytes(handle.reserved), local - base.value))
        self.opened, addrs = [], []
        for r, (raw, off) in enumerate(info):
            if r == rank:
                addrs.append(local)
                continue
            h = Handle()
            ctypes.memmove(ctypes.byref(h), raw, 64)
            mapped = ctypes.c_void_p()
            check(hip.hipIpcOpenMemHandle(ctypes.byref(mapped), h, 1), "hipIpcOpenMemHandle")
            self.opened.append(mapped.value)
            addrs.append(mapped.value + off)
        self.addrs = torch.tensor(addrs, dtype=torch.int64)  # host table, copied into the kernel args

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        for p in self.opened:
            check(hip.hipIpcCloseMemHandle(ctypes.c_void_p(p)), "hipIpcCloseMemHandle")


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


tokens = [int(t) for t in sys.argv[1:]] or [8192, 16384, 32768]
tmax = max(tokens)
sym = Symmetric(lib.sym_bytes(tmax))
gather_off = lib.gather_offset(tmax)
grid = int(os.environ.get("GRID", "96"))  # throttled: full-rate Y reads and xGMI pushes contend in the fabric
iters = int(os.environ.get("ITERS", "20"))
stream = torch.cuda.current_stream()
tim = torch.zeros(grid * 4, dtype=torch.int64, device="cuda")
epoch = 0


def fused(y, inv, T, mode, diag=0, stamps=False):
    global epoch
    epoch += 1
    lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
               ctypes.c_void_p(sym.addrs.data_ptr()), T, tmax, rank, epoch, mode, grid,
               ctypes.c_void_p(stream.cuda_stream), diag, ctypes.c_void_p(tim.data_ptr() if stamps else 0))
    return sym.storage[gather_off: gather_off + T * H * 2].view(torch.bfloat16).view(T, H)


for T in tokens:
    rows = T * TOPK
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    y = (torch.randn(rows, H, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    inv = torch.randperm(rows, device="cuda", generator=g).to(torch.int32).view(T, TOPK)
    if os.environ.get("INV_RANGE"):  # cache-resident Y rows: separates HBM-read contention from the kernel
        inv = (inv % int(os.environ["INV_RANGE"])).contiguous()
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    bad = 0
    for rep in range(3):  # fresh data every epoch
        y.mul_(-1.0) if rep else None
        partial = y[inv[:, 0].long()].float()  # k in order, as the kernel (torch's sum(1) reorders)
        for k in range(1, TOPK):
            partial += y[inv[:, k].long()].float()
        partial = partial.to(torch.bfloat16)
        parts = [torch.empty_like(partial) for _ in range(NPES)]
        dist.all_gather(parts, partial)
        ref = parts[0].float()
        for p in parts[1:]:
            ref = ref + p.float()
        ref = ref.to(torch.bfloat16)
        got = fused(y, inv, T, 0).clone()
        got1 = fused(partial, inv, T, 1).clone()
        torch.cuda.synchronize()
        bad += int((got != ref).sum()) + int((got1 != ref).sum())

    def sum_only():
        M._run(("sum", out.device, H, H, TOPK), K.launch_sum, y.view(-1), inv.view(-1), out.view(-1), T, rows,
               0, T, H, H, TOPK, stream.cuda_stream)

    res = {
        "sum": timed(sum_only, iters),
        "rccl": timed(lambda: dist.all_reduce(out), iters),
        "sum+rccl": timed(lambda: (sum_only(), dist.all_reduce(out)), iters),
        "twoshot": timed(lambda: fused(partial, inv, T, 1), iters),
        "fused": timed(lambda: fused(y, inv, T, 0), iters),
    }
    b = torch.tensor(bad, device="cuda")
    dist.all_reduce(b)
    diags = {d: timed(lambda d=d: fused(y, inv, T, 0, d), iters)
             for d in [int(x) for x in os.environ.get("DIAG", "").split(",") if x]}
    fused(y, inv, T, 0, 0, True)
    torch.cuda.synchronize()
    st = (tim.view(grid, 4) - tim.view(grid, 4)[:, :1].min()).double() / 100.0  # 100 MHz ticks -> us
    phases = "  ".join(f"{n} med {float(st[:, i].median()):.0f} max {float(st[:, i].max()):.0f}"
                       for i, n in ((1, "RS end"), (2, "AG end"), (3, "exit")))
    if rank == 0:
        hidden = (res["sum"] + res["twoshot"] - res["fused"]) / res["sum"]
        print(f"T={T:6d} grid {grid}: mismatches {int(b)}  " + "  ".join(f"{k} {v:7.1f}" for k, v in res.items())
              + f" us | fused vs sum+rccl {res['fused'] / res['sum+rccl']:.2f}x, sum hidden {hidden * 100:.0f}%",
              flush=True)
        print(f"    rank 0 phases (us from first WG start): {phases}", flush=True)
        if diags:
            names = {1: "local push", 2: "no Y loads", 4: "no AG"}
            print("    diag: " + "  ".join(f"[{'+'.join(v for k, v in names.items() if d & k)}] {t:.1f}"
                                       for d, t in diags.items()), flush=True)
sym.close()
dist.destroy_process_group()
