"""Can a 1-stage MoE hide its top-k atomics? tingqli's fused A8W4 kernel (TP8 I=256, 8x MI300X):
  atomic : 1-stage with bf16 atomics into out, then all-reduce (RCCL, and our two-shot = sum_ar.hip mode 1)
  route  : 1-stage storing route rows at their sorted position (FMOE_ROUTE_STORE=1, tq_fmoe_route_store.patch),
           then sum_ar.hip mode 0: top-k sum fused into the two-shot all-reduce (no atomics anywhere)
Both include moe_sorting + x row-quant. Needs tingqli's fork with the patch applied (see tq_store_sum_check.py).

TQ_AITER=... PYTHONPATH=$TQ_AITER:$TQ_AITER/doc-mimo/experiments/tools/pyhip_shim \
  torchrun --nproc_per_node 8 tq_1stage_route_ar.py 8192 16384            # atomic variant
FMOE_ROUTE_STORE=1 FLYDSL_RUNTIME_CACHE_DIR=/tmp/tq_rst ... (same)         # route variant
(route offsets are i32 bytes: up to 16384 tokens)
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist

TQ = os.path.join(os.environ.get("TQ_AITER", "/root/workspace/tq_aiter"), "doc-mimo")
here = os.path.dirname(os.path.abspath(__file__))
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
sys.path.insert(0, f"{TQ}/experiments/moe_2k_vs_1k")
import bench as BB  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows  # noqa: E402

F = BB.F
ROUTE = F.ROUTE_STORE
E, H, TOPK, INTER, NPES = 256, 4096, 8, 256, 8

so = os.path.join(tempfile.gettempdir(), f"sum_ar_tq_{rank}.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "sum_ar.hip"), "-o", so])
lib = ctypes.CDLL(so)
lib.sym_bytes.restype = lib.gather_offset.restype = ctypes.c_uint64
lib.sym_bytes.argtypes = lib.gather_offset.argtypes = [ctypes.c_int]
hip = ctypes.CDLL("libamdhip64.so")


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


tokens = [int(t) for t in sys.argv[1:]] or [8192, 16384]
tmax = max(tokens)
storage = torch.zeros(lib.sym_bytes(tmax), dtype=torch.uint8, device="cuda")
local = storage.data_ptr()
base = ctypes.c_void_p()
assert hip.hipPointerGetAttribute(ctypes.byref(base), 11, ctypes.c_void_p(local)) == 0
handle = Handle()
assert hip.hipIpcGetMemHandle(ctypes.byref(handle), base) == 0
info = [None] * NPES
dist.all_gather_object(info, (bytes(handle.reserved), local - base.value))
addrs = []
for r, (raw, off) in enumerate(info):
    if r == rank:
        addrs.append(local)
        continue
    h = Handle()
    ctypes.memmove(ctypes.byref(h), raw, 64)
    mapped = ctypes.c_void_p()
    assert hip.hipIpcOpenMemHandle(ctypes.byref(mapped), h, 1) == 0
    addrs.append(mapped.value + off)
addrs = torch.tensor(addrs, dtype=torch.int64)
state = {"epoch": 0}
stream = torch.cuda.current_stream()


def sum_ar(y, inv, T, mode):
    state["epoch"] += 1
    lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()), ctypes.c_void_p(addrs.data_ptr()),
               T, tmax, rank, state["epoch"], mode, 96, ctypes.c_void_p(stream.cuda_stream), 0, None)
    o = lib.gather_offset(tmax)
    return storage[o: o + T * H * 2].view(torch.bfloat16).view(T, H)


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        e.record()
        e.synchronize()
        times.append(a.elapsed_time(e) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


W = BB.make_weights(INTER)
for T in tokens:
    torch.manual_seed(T)
    x = torch.randn((T, H), dtype=dtypes.bf16, device="cuda")
    tw, ti = fused_topk(x, torch.randn((T, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)

    def moe(route_buf=None):
        sids, sw, seids, nv, out = moe_sorting(ti, tw, E, H, dtypes.bf16, 64)[:5]
        x8, xs = quantize_rows(x)
        F.a8w4_fmoe_bm64(x8, xs, W, sids, sw, seids, nv, out, topk=TOPK, b_cache_mod=1, waves_per_eu=2,
                         route=route_buf)
        return sids, nv, out

    if ROUTE:
        srt = moe_sorting(ti, tw, E, H, dtypes.bf16, 64)
        rows = max(srt[0].numel(), srt[2].numel() * 64)
        route_buf = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")

        def run():
            sids, nv, _ = moe(route_buf)
            slot_idx = torch.arange(sids.numel(), dtype=torch.int32, device="cuda")
            loc = torch.zeros(T * TOPK, dtype=torch.int32, device="cuda")
            valid = (slot_idx < nv[0]) & ((sids & 0xFFFFFF) < T)
            key = ((sids & 0xFFFFFF) * TOPK + (sids >> 24)).clamp_max(T * TOPK - 1)
            loc.scatter_(0, torch.where(valid, key, torch.zeros_like(key)).long(),
                         torch.where(valid, slot_idx, loc[0].expand_as(slot_idx)))
            return sum_ar(route_buf, loc, T, 0)
        res = {"route+sum_ar": timed(run), "1stage only": timed(lambda: moe(route_buf))}
        got = run().float().clone()
    else:
        def run_rccl():
            _, _, out = moe()
            dist.all_reduce(out)
            return out

        def run_twoshot():
            _, _, out = moe()
            return sum_ar(out, out, T, 1)
        res = {"atomic+rccl": timed(run_rccl), "atomic+twoshot": timed(run_twoshot),
               "1stage only": timed(moe)}
        got = run_twoshot().float().clone()
    torch.cuda.synchronize()
    ref = torch.load(f"/tmp/tq_ref_{T}_{rank}.pt") if ROUTE and os.path.exists(f"/tmp/tq_ref_{T}_{rank}.pt") else None
    if not ROUTE:
        torch.save(got.cpu(), f"/tmp/tq_ref_{T}_{rank}.pt")
    cos = ""
    if ref is not None:
        ref = ref.cuda()
        cos = f" | cos diff vs atomic run {1 - torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), 0).item():.1e}"
    if rank == 0:
        print(f"T={T:6d} " + "  ".join(f"{k} {v:7.1f} us" for k, v in res.items()) + cos, flush=True)
dist.barrier()
dist.destroy_process_group()
