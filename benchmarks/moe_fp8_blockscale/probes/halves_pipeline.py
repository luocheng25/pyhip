"""3b prototype (readme 4.14): the MoE layer in two token halves A / B; A's top-k sum + two-shot all-reduce
(sum_ar.hip) runs on comm CUs while B's gate_up runs on the other CUs (two CU-masked streams), then B's
inverse / down / sum_ar run on the whole GPU.

torchrun --nproc_per_node 8 halves_pipeline.py [tokens...]    env CUS=64,96 (comm CUs), ROUNDS=15

Times (slowest rank, round-robin bursts) the one-shot layer (quant + sort + MoE + sum_ar) against the halves
pipeline sequential (everything on one stream) and overlapped, and checks that every variant's all-reduced
output is bit-identical to the one-shot layer.
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
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as K  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M  # noqa: E402

H, NPES, TOPK, E = 4096, 8, 8, 256
lr = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
hip = ctypes.CDLL("libamdhip64.so")
tmp = tempfile.gettempdir()


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


class Symmetric:
    """One torch allocation per rank, mapped into every rank through HIP IPC."""

    def __init__(self, nbytes):
        self.storage = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        local = self.storage.data_ptr()
        base = ctypes.c_void_p()
        assert hip.hipPointerGetAttribute(ctypes.byref(base), 11, ctypes.c_void_p(local)) == 0
        handle = Handle()
        assert hip.hipIpcGetMemHandle(ctypes.byref(handle), base) == 0
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
            assert hip.hipIpcOpenMemHandle(ctypes.byref(mapped), h, 1) == 0
            self.opened.append(mapped.value)
            addrs.append(mapped.value + off)
        self.addrs = torch.tensor(addrs, dtype=torch.int64)

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        for p in self.opened:
            hip.hipIpcCloseMemHandle(ctypes.c_void_p(p))


def masked_stream(cus):
    words = (ctypes.c_uint32 * 10)()
    for c in cus:  # bit i -> XCD i % 8: a contiguous bit range is balanced over the XCDs
        words[c // 32] |= 1 << (c % 32)
    s = ctypes.c_void_p()
    assert hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), 10, words) == 0
    return torch.cuda.ExternalStream(s.value)


so = os.path.join(tmp, f"sum_ar_{rank}.so")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                       os.path.join(here, "sum_ar.hip"), "-o", so])
lib = ctypes.CDLL(so)
lib.sym_bytes.restype = lib.gather_offset.restype = ctypes.c_uint64
lib.sym_bytes.argtypes = lib.gather_offset.argtypes = [ctypes.c_int]

tokens = [int(t) for t in sys.argv[1:]] or [16384, 32768]
tmax = max(tokens)
COMM_CUS = [int(c) for c in os.environ.get("CUS", "64,96").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "15"))
# sum_ar's closing grid barrier counts epoch * grid: one symmetric buffer per grid size
syms = {k: Symmetric(lib.sym_bytes(tmax if k == "full" else tmax // 2))
        for k in ["full", "B", "A"] + [f"A{c}" for c in COMM_CUS]}
epochs = {k: 0 for k in syms}
masked = {c: (masked_stream(range(c)), masked_stream(range(c, 304))) for c in COMM_CUS}
# never the legacy default stream: it serializes with the (blocking) masked streams
torch.cuda.set_stream(torch.cuda.Stream())


def gather(key, T, tm):
    o = lib.gather_offset(tm)
    return syms[key].storage[o: o + T * H * 2].view(torch.bfloat16).view(T, H)


def sum_ar(key, y, inv, T, tm, grid, stream):
    epochs[key] += 1
    lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
               ctypes.c_void_p(syms[key].addrs.data_ptr()), T, tm, rank, epochs[key], 0, grid,
               ctypes.c_void_p(stream.cuda_stream), 0, None)


class Half:
    """Sorted rows, compact tiles and workspaces of one token range."""

    def __init__(self, c, lo, hi, x8, xs, stream):
        self.T = hi - lo
        self.x8 = x8[lo:hi]
        self.xs = xs[:, lo:hi].contiguous()
        self.sid, self.sw, seid, nv = M.moe_sorting(c["ids"][lo:hi], c["wts"][lo:hi], E, H, None, M.COMPACT_BLOCK)
        self.nv = nv
        self.tiles, self.count, self.cap = M._tiles(seid, nv, E, M.COMPACT_BLOCK, stream)
        self.rows = self.sid.numel()
        inter = c["inter"]
        self.a2 = torch.empty(self.rows * inter, dtype=torch.uint8, device="cuda")
        self.a2s = torch.empty((inter // 128) * self.rows, dtype=torch.float32, device="cuda")
        self.y = torch.empty(self.rows * H, dtype=torch.bfloat16, device="cuda")
        self.inv = torch.full((self.T, TOPK), -1, dtype=torch.int32, device="cuda")


def gate_up(c, h, stream):
    u8 = torch.uint8
    inter = c["inter"]
    M._run(("gate_up", torch.device("cuda", lr), inter, H), K.launch_gate_up,
           h.x8.view(u8).view(-1), c["w13_sh"].view(u8).view(-1), h.xs.view(-1), c["s13"].reshape(-1), h.sid,
           h.tiles, h.count, h.sw, h.a2, h.a2s, h.T, h.rows, h.cap * (inter // 128), inter, H, stream)


def inverse_down(c, h, stream):
    inter = c["inter"]
    dev = torch.device("cuda", lr)
    M._run(("inverse", dev, TOPK), K.launch_inverse, h.sid, h.nv, h.inv.view(-1), h.T, h.rows,
           (h.rows + K.SUM_THREADS - 1) // K.SUM_THREADS, TOPK, stream)
    groups = H // K.DOWN_COLS
    M._run(("down_direct", dev, H, inter, groups), K.launch_down_direct,
           h.a2, c["w2_sh"].view(torch.uint8).view(-1), h.a2s, c["s2"].reshape(-1), h.tiles, h.count, h.sid,
           h.y, h.rows, h.T, 0, h.cap * groups, H, inter, groups, stream)


def timed_rounds(fns, rounds, burst=4):
    for fn in fns.values():
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(burst):
                fn()
            e.record()
            e.synchronize()
            times[k].append(a.elapsed_time(e) * 1e3 / burst)
    t = torch.tensor([sorted(v)[len(v) // 2] for v in times.values()], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return dict(zip(fns, t.tolist()))


for T in tokens:
    c = b.make_case("tp8", T, seed=rank)
    main = torch.cuda.current_stream()
    half = T // 2
    full_out = {}
    _down_and_sum = M._down_and_sum

    def tail_sum_ar(a2_, w2_, a2s_, w2s_, tiles_, count_, cap_, sid_, y_, inv_, out_, tokens_, rows_, hidden_,
                    inter_, topk_, stream_):
        groups = hidden_ // K.DOWN_COLS
        M._run(("down_direct", torch.device("cuda", lr), hidden_, inter_, groups), K.launch_down_direct,
               a2_, w2_, a2s_, w2s_, tiles_, count_, sid_, y_, rows_, tokens_, 0, cap_ * groups, hidden_, inter_,
               groups, stream_)
        sum_ar("full", y_, inv_, tokens_, tmax, 96, stream_)

    def one_shot():
        M._down_and_sum = tail_sum_ar
        try:
            b.run_pyhip(c)
        finally:
            M._down_and_sum = _down_and_sum

    def halves(comm=None):
        """comm = number of comm CUs (overlapped) or None (all on the current stream)."""
        x8, xs = M.quantize_input(c["x"])
        A = Half(c, 0, half, x8, xs, main)
        B = Half(c, half, T, x8, xs, main)
        gate_up(c, A, main)
        inverse_down(c, A, main)
        if comm is None:
            sum_ar("A", A.y, A.inv, A.T, tmax // 2, 96, main)
            gate_up(c, B, main)
        else:
            s_comm, s_cmp = masked[comm]
            ready = torch.cuda.Event()
            ready.record(main)
            s_comm.wait_event(ready)
            s_cmp.wait_event(ready)
            sum_ar(f"A{comm}", A.y, A.inv, A.T, tmax // 2, comm * 2 // 3, s_comm)
            gate_up(c, B, s_cmp)
            for s in (s_comm, s_cmp):
                done = torch.cuda.Event()
                done.record(s)
                main.wait_event(done)
            for t_ in (A.y, A.inv, B.x8, B.xs, B.sid, B.tiles, B.count, B.sw, B.a2, B.a2s):
                t_.record_stream(s_comm)
                t_.record_stream(s_cmp)
        inverse_down(c, B, main)
        sum_ar("B", B.y, B.inv, B.T, tmax // 2, 96, main)

    one_shot()
    torch.cuda.synchronize()
    ref = gather("full", T, tmax).clone()
    checks = {}
    for name, key_a, fn in [("halves seq", "A", lambda: halves())] + [
            (f"halves overlap {k}", f"A{k}", (lambda k=k: halves(k))) for k in COMM_CUS]:
        for key in (key_a, "B"):
            gather(key, half, tmax // 2).zero_()
        fn()
        torch.cuda.synchronize()
        got = torch.cat([gather(key_a, half, tmax // 2), gather("B", T - half, tmax // 2)])
        bad = torch.tensor(int((got != ref).sum()), device="cuda")
        dist.all_reduce(bad)
        checks[name] = int(bad)
    variants = {"one-shot + sum_ar": one_shot, "halves seq": lambda: halves()}
    variants.update({f"halves overlap {k}": (lambda k=k: halves(k)) for k in COMM_CUS})
    res = timed_rounds(variants, ROUNDS)
    if rank == 0:
        print(f"T={T}: mismatches {checks} | " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()) + " us",
              flush=True)
for s in syms.values():
    s.close()
dist.destroy_process_group()
