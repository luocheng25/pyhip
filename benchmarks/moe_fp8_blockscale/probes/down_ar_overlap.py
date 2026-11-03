"""Hide the MoE all-reduce under the down GEMM (TP8 I=256, 8x MI300X): down runs its four 1024-column groups as
four launches on a CU-masked stream (CUs >= K), and sum_ar_cols.hip (top-k sum + two-shot all-reduce of one
column group) follows each group on a second CU-masked stream (K CUs), so the communication of group g overlaps
the down GEMM of groups g+1.. .

torchrun --nproc_per_node 8 down_ar_overlap.py [tokens...]   (env CUS=48: CUs of the comm stream, COMM_GRID, ITERS=10)

Checks the overlapped output bit-exactly against pyhip's sum kernel + fp32 rank-order reduction, then times
(slowest rank, median):
  base      : down + sum + RCCL all_reduce (full GPU, current path)
  fused     : down + sum_ar (sum_ar.hip, full GPU, grid 96)
  down_mask : down's four group launches on the masked stream alone
  comm_mask : the four sum_ar_cols launches on the comm stream alone (Y ready)
  overlap   : the pipeline
  nodep     : both streams without the group dependency (upper bound of the concurrency; output not checked)
  gate_mask / gate_nodep : gate_up alone on the main partition / with the four comm launches concurrently (no
              dependency): would the all-reduce of one token half hide under the next half's gate_up?
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

H, NPES, TOPK = 4096, 8, 8
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
hip = ctypes.CDLL("libamdhip64.so")


def build(name):
    so = os.path.join(tempfile.gettempdir(), f"{name}_{rank}.so")
    subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                           os.path.join(here, f"{name}.hip"), "-o", so])
    lib = ctypes.CDLL(so)
    lib.sym_bytes.restype = lib.gather_offset.restype = ctypes.c_uint64
    lib.sym_bytes.argtypes = lib.gather_offset.argtypes = [ctypes.c_int]
    if hasattr(lib, "err_offset"):
        lib.err_offset.restype, lib.err_offset.argtypes = ctypes.c_uint64, [ctypes.c_int]
    return lib


cols_lib, full_lib = build("sum_ar_cols"), build("sum_ar")


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


tokens = [int(t) for t in sys.argv[1:]] or [8192, 16384, 32768]
tmax = max(tokens)
CUS = int(os.environ.get("CUS", "48"))
# a CU-masked queue keeps only ~2/3 of its CUs co-resident for this 8-wave, 172-VGPR kernel (48 CUs: 32 WGs
# all start together, 40 do not); the WGs wait on each other, so the grid must stay within that
COMM_GRID = int(os.environ.get("COMM_GRID", str(CUS * 2 // 3)))
ITERS = int(os.environ.get("ITERS", "10"))
s_main, s_comm = masked_stream(range(CUS, 304)), masked_stream(range(CUS))
ctrl = torch.cuda.Stream()  # never the legacy default stream: it would serialize with the masked (blocking) ones
sym_cols, sym_full = Symmetric(cols_lib.sym_bytes(tmax)), Symmetric(full_lib.sym_bytes(tmax))
state = {"cols_epoch": 0, "cols_bars": 0, "full_epoch": 0}


def gather(sym, lib, T):
    o = lib.gather_offset(tmax)
    return sym.storage[o: o + T * H * 2].view(torch.bfloat16).view(T, H)


def progress(msg):
    torch.cuda.synchronize()
    o = cols_lib.err_offset(tmax)
    flag = sym_cols.storage[o: o + 4].view(torch.int32)
    err = int(flag[0])
    flag.zero_()
    torch.cuda.synchronize()
    if rank == 0 or err:
        print(f"[rank {rank}] {msg}{f'  ERROR {err}' if err else ''}", flush=True)


def comm_group(y, inv, T, g, stream):
    state["cols_epoch"] += 1
    bar = 0
    if g == 3:
        state["cols_bars"] += 1
        bar = state["cols_bars"] * COMM_GRID
    cols_lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
                    ctypes.c_void_p(sym_cols.addrs.data_ptr()), T, tmax, rank, state["cols_epoch"], 0, g, bar,
                    COMM_GRID, ctypes.c_void_p(stream.cuda_stream))


def fused_full(y, inv, T, stream):
    state["full_epoch"] += 1
    full_lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
                    ctypes.c_void_p(sym_full.addrs.data_ptr()), T, tmax, rank, state["full_epoch"], 0, 96,
                    ctypes.c_void_p(stream.cuda_stream), 0, None)


def timed(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(ITERS):
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(ctrl)
        fn()
        e.record(ctrl)
        e.synchronize()
        times.append(a.elapsed_time(e) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


for T in tokens:
    c = b.make_case("tp8", T, seed=rank)
    captured = {}
    _run = M._run

    def capture(key, launcher, *args):
        captured[key[0]] = (key, launcher, args)
        return _run(key, launcher, *args)

    M._run = capture
    b.run_pyhip(c)
    torch.cuda.synchronize()
    M._run = _run
    dkey, dlaunch, dargs = captured["down_direct"]
    a2, w2, a2s, w2s, seid, nv, sid, y, rows, tok, _, grid, hidden, inter, groups, _ = dargs
    blocks = grid // groups
    skey, slaunch, sargs = captured["sum"]
    gkey, glaunch, gargs = captured["gate_up"]
    inv = sargs[1]
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")

    def gate_up(stream):
        M._run(gkey, glaunch, *gargs[:-1], stream)

    def down_all(stream):
        M._run(dkey, dlaunch, a2, w2, a2s, w2s, seid, nv, sid, y, rows, tok, 0, blocks * groups, hidden, inter,
               groups, stream)

    def down_group(g, stream):
        M._run(("down_direct", dkey[1], hidden, inter, 1), dlaunch, a2, w2, a2s, w2s, seid, nv, sid, y, rows, tok,
               g, blocks, hidden, inter, 1, stream)

    def sum_only(stream):
        M._run(skey, slaunch, y, inv, out.view(-1), T, rows, 0, T, H, H, TOPK, stream)

    def base():
        with torch.cuda.stream(ctrl):
            down_all(ctrl)
            sum_only(ctrl)
            dist.all_reduce(out)

    def fused():
        down_all(ctrl)
        fused_full(y, inv, T, ctrl)

    def pipeline(do_down=True, do_comm=True, dep=True, gate=False):
        start = ctrl.record_event()
        s_main.wait_event(start)
        s_comm.wait_event(start)
        ready = []
        if gate:
            gate_up(s_main)
        for g in range(4):
            if do_down:
                down_group(g, s_main)
            ready.append(s_main.record_event())
        for g in range(4):
            if do_comm:
                if dep:
                    s_comm.wait_event(ready[g])
                comm_group(y, inv, T, g, s_comm)
        ctrl.wait_event(s_main.record_event())
        ctrl.wait_event(s_comm.record_event())

    # compile every launch configuration on every rank before any cross-rank wait can start
    with torch.cuda.stream(ctrl):
        for g in range(4):
            down_group(g, s_main)
        gate_up(s_main)
        down_all(ctrl)
        sum_only(ctrl)
    torch.cuda.synchronize()
    dist.barrier()

    # correctness: the pipeline vs pyhip sum (same fp32 k order) + fp32 rank-order reduction, fresh epochs
    with torch.cuda.stream(ctrl):
        bad = 0
        for rep in range(3):
            pipeline()
            progress(f"T={T} check {rep}: pipeline done")
            got = gather(sym_cols, cols_lib, T).clone()
            sum_only(ctrl)
            parts = [torch.empty_like(out) for _ in range(NPES)]
            dist.all_gather(parts, out)
            ref = parts[0].float()
            for p in parts[1:]:
                ref = ref + p.float()
            torch.cuda.synchronize()
            bad += int((got != ref.to(torch.bfloat16)).sum())
        res = {}
        for name, fn in (("base", base), ("fused", fused), ("down_mask", lambda: pipeline(True, False)),
                         ("comm_mask", lambda: pipeline(False, True)), ("overlap", pipeline),
                         ("nodep", lambda: pipeline(True, True, False)),
                         ("gate_mask", lambda: pipeline(False, False, False, True)),
                         ("gate_nodep", lambda: pipeline(False, True, False, True))):
            res[name] = timed(fn)
            progress(f"T={T} timed {name}: {res[name]:.1f} us")
        badt = torch.tensor(bad, device="cuda")
        dist.all_reduce(badt)
    if rank == 0:
        print(f"T={T:6d} comm CUs {CUS} grid {COMM_GRID}: mismatches {int(badt)}  "
              + "  ".join(f"{k} {v:7.1f}" for k, v in res.items())
              + f" us | overlap vs base {res['overlap'] / res['base']:.2f}x, vs fused {res['overlap'] / res['fused']:.2f}x",
              flush=True)
sym_cols.close()
sym_full.close()
dist.destroy_process_group()
