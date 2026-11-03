"""Fused MoE down GEMM + top-k sum + two-shot all-reduce in ONE kernel (TP8 I=256, 8x MI300X, HIP IPC), in the
style of aiter's mega_moe_tp: compute waves and communication waves share the CUs of one persistent launch.

256-thread WGs, two per CU (the down body needs 206 VGPRs and 32 KB LDS, so a CU holds two 4-wave WGs and no
more; s_barrier is WG-wide, so the comm waves live in their own WG). The first WG on a CU runs 4 down waves
(pyhip's short-K _down_direct_body, unchanged); a second WG becomes a comm WG (2 sum + 2 push waves) while its
XCD's quota of NCOMM / 8 lasts, else another compute WG. Compute WGs take (column group, M-block) tasks
column-group-major from per-XCD queues; comm WGs reduce and all-reduce column group g (CW columns) as soon as every
M-block of g is done, while the compute WGs work on groups > g. Device code of the roles, queues and comm:
down_ar_comm.hip, linked into the FlyDSL kernel as LLVM bitcode (flydsl.compiler.extern_link).

torchrun --nproc_per_node 8 down_ar_fused.py [tokens...]
  env CW=512 (column group width), CONFIGS=160:76 (ncomm:cpx[:probe], see below), WBL2=2, STORE_AUX, ITERS=10,
  ATOMIC=1: the down adds route rows into acc [T, H] with BF16 atomics (down_atomic.py; VALU_SLOTS, DMA_WAIT) and
  the comm all-reduces acc (readme 4.12; numerics reported as rel_l2 instead of bit-exactness),
  LAYER=1: also time whole MoE layers incl. quant + sort + all-reduce (aiter / pyhip with RCCL, two-shot, sum_ar,
  down_ar; the fused 1-stage pyhip with sum_ar; compare with tq_layer_compare.py)

Checks the output bit-exactly against pyhip down + sum + fp32 rank-order reduction (= sum_ar), then times (slowest
rank, median): base (down + sum + RCCL AR), seq (down + sum_ar.hip), fused, the fused kernel with the comm WGs
idle (compute only) and with the GEMM skipped (comm only), and prints the device timeline (per group: compute done,
comm saw it, RS done, AG done; us after the first WG start).
"""

import ctypes
import os
import subprocess
import sys
import tempfile

if os.environ.get("DUMPISA") and os.environ.get("LOCAL_RANK", "0") == "0":
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = os.environ["DUMPISA"]

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.compiler.extern_link import link_extern
from flydsl.expr import gpu
from flydsl.expr.extern import ffi

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as KM  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M  # noqa: E402

import down_atomic as DA  # noqa: E402

H, NPES, TOPK, CUS = 4096, 8, 8, 304
CW = int(os.environ.get("CW", "512"))  # column group width: the unit of down tasks and of the comm pipeline
NCG = H // CW
lr = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
hip = ctypes.CDLL("libamdhip64.so")

tmp = tempfile.gettempdir()
bc = os.path.join(tmp, f"down_ar_comm_{rank}.bc")
subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "--cuda-device-only", "-emit-llvm", "-c",
                       "-fgpu-rdc", "-nogpulib", "-mcode-object-version=6", "-Xclang", "-target-feature", "-Xclang",
                       "-packed-fp32-ops", f"-DCW={CW}", *os.environ.get("HIPFLAGS", "").split(),
                       os.path.join(here, "down_ar_comm.hip"), "-o", bc])
_role = link_extern(ffi("dar_role", ["int64", "int32", "int32", "int32"], "int32"), bitcode_path=bc)
_xcc = link_extern(ffi("dar_xcc", [], "int32"), bitcode_path=bc)
_claim = link_extern(ffi("dar_claim", ["int64", "int32"], "int32"), bitcode_path=bc)
_publish = link_extern(ffi("dar_publish", ["int64"] + ["int32"] * 5, "void"), bitcode_path=bc)
_cend = link_extern(ffi("dar_cend", ["int64"], "void"), bitcode_path=bc)
_comm = link_extern(ffi("dar_comm", ["int64"] * 5 + ["int32"] * 8, "void"), bitcode_path=bc)
C_DBG, C_ERR = 2080, 2073
# Y store cache policy in the fused kernel (16 = sc1: write-through at agent scope, then publish needs no wbl2)
STORE_AUX = int(os.environ.get("STORE_AUX", "0"))
_store_pair_orig = KM._store_pair


def _store_pair_aux(first, second, y_res, voffset, soffset):
    from flydsl._mlir import ir
    from flydsl.expr import rocdl
    aux = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), STORE_AUX)
    for words, offset in ((first, soffset - 64), (second, soffset)):
        rocdl.raw_ptr_buffer_store(fx.Vector.from_elements(words, fx.Int32).ir_value(), y_res,
                                   fx.Int32(voffset).ir_value(), fx.Int32(offset).ir_value(), aux=aux)


def _addr(t):
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


@flyc.kernel(known_block_size=[KM.DOWN_THREADS, 1, 1])
def _down_ar_kernel(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, TILES: fx.Tensor, COUNT: fx.Tensor,
                    SID: fx.Tensor, Y: fx.Tensor, YC: fx.Tensor, INV: fx.Tensor, CTL: fx.Tensor, PEERS: fx.Tensor,
                    R: fx.Int32, T: fx.Int32, TC: fx.Int32, TMAX: fx.Int32, RANK: fx.Int32, EPOCH: fx.Int32,
                    NCOMM: fx.Int32, COMM: fx.Int32, WBL2: fx.Int32, CPX: fx.Int32, CMODE: fx.Int32,
                    N: fx.Constexpr[int], K: fx.Constexpr[int], ATOM: fx.Constexpr[int]):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, 2 * KM.DOWN_CHUNK * K, 16]).peek().view(
        fx.make_layout(2 * KM.DOWN_CHUNK * K, 1))
    lds = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    ctl = _addr(CTL)
    role = fx.Int32(_role(ctl, NCOMM, CPX, lds))
    if role == -1:
        x = fx.Int32(_xcc())
        ntile = KM._scalar_ready(KM._read_scalar_g2r(COUNT, 0))
        lo = x * (ntile >> 3) + KM._min(x, ntile & 7)
        nb = (ntile >> 3) + (x < (ntile & 7)).select(fx.Int32(1), fx.Int32(0))
        nx = KM._min(ntile, fx.Int32(8))
        i = fx.Int32(_claim(ctl, lds))
        while i < nb * NCG:
            cg = i // nb
            row0, rows, expert = KM._tile(TILES, lo + i - cg * nb)
            if COMM != 2:
                if fx.const_expr(ATOM):  # 256-row sorting only (ATOMIC=1 sets it)
                    DA._down_atomic_body(A2, W, A2S, WS, SID, Y, storage, R, TC, row0 // KM.BM, cg, expert, N, K)
                else:
                    KM._down_direct_body(A2, W, A2S, WS, SID, Y, storage, R, TC, row0, rows >> 6, cg, expert, N, K)
            _publish(ctl, cg, WBL2, nb, nx, i)
            i = fx.Int32(_claim(ctl, lds))
        _cend(ctl)
    else:
        if role >= 0:
            if COMM != 0:
                _comm(_addr(YC), _addr(INV), _addr(PEERS), ctl, _addr(COUNT), T, TMAX, RANK, EPOCH, lds, NCOMM, role,
                      CMODE)


@flyc.jit
def launch_down_ar(A2: fx.Tensor, W: fx.Tensor, A2S: fx.Tensor, WS: fx.Tensor, TILES: fx.Tensor, COUNT: fx.Tensor,
                   SID: fx.Tensor, Y: fx.Tensor, YC: fx.Tensor, INV: fx.Tensor, CTL: fx.Tensor, PEERS: fx.Tensor,
                   R: fx.Int32, T: fx.Int32, TC: fx.Int32, TMAX: fx.Int32, RANK: fx.Int32, EPOCH: fx.Int32,
                   NCOMM: fx.Int32, COMM: fx.Int32, WBL2: fx.Int32, CPX: fx.Int32, CMODE: fx.Int32, GRID: fx.Int32,
                   N: fx.Constexpr[int], K: fx.Constexpr[int], ATOM: fx.Constexpr[int], stream: fx.Stream):
    _down_ar_kernel(A2, W, A2S, WS, TILES, COUNT, SID, Y, YC, INV, CTL, PEERS, R, T, TC, TMAX, RANK, EPOCH, NCOMM, COMM,
                    WBL2, CPX, CMODE, N, K, ATOM, value_attrs=KM._attrs()).launch(grid=(GRID, 1, 1),
                                                                            block=(KM.DOWN_THREADS, 1, 1),
                                                                            stream=stream)


def sym_layout(tmax):
    rmax = tmax // NPES
    gather = tmax * H * 2
    rs = gather + tmax * H * 2
    ag = rs + NCG * NPES * rmax * 4
    return gather, ag + NCG * tmax * 4


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


def build_sum_ar():
    so = os.path.join(tmp, f"sum_ar_{rank}.so")
    subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                           os.path.join(here, "sum_ar.hip"), "-o", so])
    lib = ctypes.CDLL(so)
    lib.sym_bytes.restype = lib.gather_offset.restype = ctypes.c_uint64
    lib.sym_bytes.argtypes = lib.gather_offset.argtypes = [ctypes.c_int]
    return lib


tokens = [int(t) for t in sys.argv[1:]] or [8192, 16384, 32768]
tmax = max(tokens)
# ncomm:cpx[:probe] = comm WGs (multiple of 8) : max compute WGs per XCD (76 = no cap) : probe bits (4: all-reduce
# the ready sum output instead of reading Y, +8: without waiting for the down groups, +16: down drops its Y stores;
# timing only); ncomm 0 = compute only
CONFIGS = [tuple(int(v) for v in (c + ":0").split(":")[:3]) for c in os.environ.get("CONFIGS", "160:76").split(",")]
WBL2 = int(os.environ.get("WBL2", "2"))
ITERS = int(os.environ.get("ITERS", "10"))
# ATOMIC=1: the down epilogue adds the weighted route rows into acc [T, H] with BF16 atomics (down_atomic.py)
# and the comm all-reduces acc (no Y round trip, no top-k sum)
ATOMIC = int(os.environ.get("ATOMIC", "0"))
if ATOMIC:
    M.COMPACT_BLOCK = M.BLOCK_M
DA.VALU_SLOTS = int(os.environ.get("VALU_SLOTS", str(DA.VALU_SLOTS)))
DA.DMA_WAIT = int(os.environ.get("DMA_WAIT", str(DA.DMA_WAIT)))


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())

assert all(n % 8 == 0 and n <= CUS for n, _, _ in CONFIGS)
gather_off, sym_bytes = sym_layout(tmax)
sym = Symmetric(sym_bytes)
peers = sym.addrs.cuda()
full_lib = build_sum_ar()
sym_full = Symmetric(full_lib.sym_bytes(tmax))
ctl = torch.zeros(2560 + 8 * 640 * 4, dtype=torch.int32, device="cuda")
state = {"epoch": 0, "full_epoch": 0}
compiled = {}


def check_ctl(msg):
    torch.cuda.synchronize()
    err = int(ctl[C_ERR])
    if err:
        print(f"[rank {rank}] {msg}: ERROR {err}", flush=True)
    return err


def timeline():
    """us after the first WG start: per group compute done / comm saw it / RS done / AG done, comm end, compute end"""
    torch.cuda.synchronize()
    d = ctl[C_DBG: C_DBG + 2 * (3 + 4 * NCG)].view(torch.int64).cpu().tolist()
    m = (1 << 64) - 1
    t0 = (~d[0]) & m
    rel = [((v & m) - t0) / 100.0 if v else float("nan") for v in d]
    seen = [(((~d[1 + NCG + g]) & m) - t0) / 100.0 for g in range(NCG)]
    return ("ready " + "/".join(f"{rel[1 + g]:.0f}" for g in range(NCG)) + " seen " + "/".join(f"{s:.0f}" for s in seen)
            + " rs " + "/".join(f"{rel[1 + 2 * NCG + g]:.0f}" for g in range(NCG)) + " ag "
            + "/".join(f"{rel[1 + 3 * NCG + g]:.0f}" for g in range(NCG)) + f" fin {rel[1 + 4 * NCG]:.0f}"
            + f" cend {rel[2 + 4 * NCG]:.0f}")


def timed(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(ITERS):
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        e.record()
        e.synchronize()
        times.append(a.elapsed_time(e) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def timed_rounds(fns, rounds, burst=4):
    """Round-robin over the variants (clock drift at the power cap hits all alike), each timed as a burst of
    back-to-back calls (host launch latency hidden); median per variant, slowest rank."""
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
    a2, w2, a2s, w2s, tiles, count, sid, y, rows, tok, _, grid, hidden, inter, groups, _ = dargs
    skey, slaunch, sargs = captured["sum"]
    inv = sargs[1]
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()

    def down():
        M._run(dkey, dlaunch, *dargs[:-1], stream)

    def sum_only():
        M._run(skey, slaunch, y, inv, out.view(-1), T, rows, 0, T, H, H, TOPK, stream)

    def base():
        down()
        sum_only()
        dist.all_reduce(out)

    def seq():
        down()
        state["full_epoch"] += 1
        full_lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
                        ctypes.c_void_p(sym_full.addrs.data_ptr()), T, tmax, rank, state["full_epoch"], 0, 96,
                        ctypes.c_void_p(stream.cuda_stream), 0, None)

    def fused(ncomm, comm=1, cpx=76, cmode=0, atom=ATOMIC):
        if comm:
            state["epoch"] += 1
        ctl.zero_()
        if atom:  # the down accumulates into acc (zeroed by the caller), the comm all-reduces acc
            ybuf, ycbuf, mode = acc, acc, (cmode & 15) | 4
        else:
            ybuf, ycbuf, mode = y, out if cmode & 4 else y, cmode & 15
        args = (a2, w2, a2s, w2s, tiles, count, sid, ybuf, ycbuf, inv, ctl, peers, rows, T, 0 if cmode & 16 else T, tmax,
                rank, state["epoch"], ncomm, comm, WBL2, cpx, mode, 2 * CUS, hidden, inter, atom, stream)
        fn = compiled.get(atom)
        if fn is None:
            KM._store_pair = _store_pair_aux if STORE_AUX else _store_pair_orig
            cols, KM.DOWN_COLS, DA.DOWN_COLS = KM.DOWN_COLS, CW, CW
            compiled[atom] = flyc.compile(launch_down_ar, *args)
            KM._store_pair, KM.DOWN_COLS = _store_pair_orig, cols
        else:
            fn(*args)

    def gather():
        return sym.storage[gather_off: gather_off + T * H * 2].view(torch.bfloat16).view(T, H)

    # reference: pyhip down Y and sum, fp32 rank-order reduction
    y.zero_()
    down()
    y_ref = y.clone()
    sum_only()
    local_ref = out.clone()
    parts = [torch.empty_like(out) for _ in range(NPES)]
    dist.all_gather(parts, out)
    ref = parts[0].float()
    for p in parts[1:]:
        ref = ref + p.float()
    ref = ref.to(torch.bfloat16)
    # FP32 truth on the first SUB tokens from the same BF16 route rows (top-k and rank sums in FP32)
    SUB = min(T, 2048)
    idx = inv.view(T, TOPK)[:SUB].long()
    truth = torch.zeros(SUB, H, dtype=torch.float32, device="cuda")
    for k in range(TOPK):
        r = idx[:, k]
        truth += torch.where((r >= 0)[:, None], y_ref.view(-1, H)[r.clamp(min=0)].float(), 0.0)
    dist.all_reduce(truth)

    # compile (comm idle) and check Y, then the full fused kernel, fresh epochs
    acc = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    y.zero_()
    fused(0, 0)
    check_ctl(f"T={T} compute-only")
    ybad = rel_l2(acc, local_ref) if ATOMIC else int((y != y_ref).sum())
    torch.cuda.synchronize()
    dist.barrier()
    res = {}
    for ncomm, cpx, cmode in CONFIGS:
        tag = f"{ncomm}:{cpx}:{cmode}"
        if ncomm == 0:
            res[f"comp{tag}"] = timed(lambda: fused(0, 0, cpx, cmode))
            continue
        if cmode:
            res[f"fused{tag}"] = timed(lambda: fused(ncomm, 1, cpx, cmode))
            tl = timeline()
            res[f"comm{tag}"] = timed(lambda: fused(ncomm, 2, cpx, cmode))
            tl2 = timeline()
            if rank == 0:
                print(f"T={T} {tag}: probe fused {res[f'fused{tag}']:.1f}  comm-only {res[f'comm{tag}']:.1f} us\n"
                      f"   fused     {tl}\n   comm-only {tl2}", flush=True)
            check_ctl(f"T={T} {tag} probe")
            continue
        bad = 0
        outs = []
        for rep in range(2):
            sym.storage[gather_off: gather_off + T * H * 2].zero_()
            acc.zero_()
            fused(ncomm, 1, cpx)
            err = check_ctl(f"T={T} {tag} check {rep}")
            outs.append(gather().clone())
            bad += (0 if ATOMIC else int((outs[-1] != ref).sum())) + (1 << 30) * (err != 0)
        badt = torch.tensor(bad, device="cuda")
        dist.all_reduce(badt)
        if ATOMIC:  # BF16 atomics: order-dependent rounding; error vs the BF16 path and vs FP32 truth
            errs = torch.tensor([rel_l2(outs[0], ref), rel_l2(outs[0][:SUB], truth), rel_l2(ref[:SUB], truth),
                                 float((outs[0] != outs[1]).float().mean())], device="cuda")
            dist.all_reduce(errs, op=dist.ReduceOp.MAX)
            numerics = (f"rel_l2 vs bf16 path {errs[0]:.2e}, vs fp32 truth {errs[1]:.2e} (bf16 path {errs[2]:.2e}),"
                        f" run-to-run differing {errs[3] * 100:.2f}%")
        else:
            numerics = f"mismatches {int(badt)}"
        res[f"fused{tag}"] = timed(lambda: fused(ncomm, 1, cpx))
        tl = timeline()
        if rank == 0 and os.environ.get("DUMP"):
            torch.save(ctl.cpu(), f"{os.environ['DUMP']}_T{T}_{ncomm}_{cpx}_fused.pt")
        res[f"comp{tag}"] = timed(lambda: fused(ncomm, 0, cpx))
        res[f"comm{tag}"] = timed(lambda: fused(ncomm, 2, cpx))
        tl2 = timeline()
        if rank == 0:
            print(f"T={T} {tag}: {numerics}  fused {res[f'fused{tag}']:.1f}  "
                  f"compute-only {res[f'comp{tag}']:.1f}  comm-only {res[f'comm{tag}']:.1f} us\n"
                  f"   fused     {tl}\n   comm-only {tl2}", flush=True)
        check_ctl(f"T={T} {tag} timed")
    res["down"] = timed(down)
    res["base"] = timed(base)
    res["seq"] = timed(seq)
    if ATOMIC:
        res["zero acc"] = timed(acc.zero_)
    ybadt = torch.tensor(ybad, device="cuda")
    dist.all_reduce(ybadt, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"T={T:6d}: " + (f"compute-only acc rel_l2 vs local sum {float(ybadt):.2e}" if ATOMIC else
                               f"Y mismatches {int(ybadt)}") + " | "
              + "  ".join(f"{k} {v:7.1f}" for k, v in res.items()) + " us", flush=True)

    if os.environ.get("LAYER"):  # whole MoE layer incl. quant + sort + all-reduce (readme 4.11)
        ncomm, cpx, _ = [cfg for cfg in CONFIGS if cfg[0]][-1]
        down_and_sum = M._down_and_sum

        def tail_sum_ar(a2_, w2_, a2s_, w2s_, tiles_, count_, cap_, sid_, y_, inv_, out_, tokens_, rows_, hidden_,
                        inter_, topk_, stream_):
            M._run(dkey, dlaunch, a2_, w2_, a2s_, w2s_, tiles_, count_, sid_, y_, rows_, tokens_, 0, cap_ * groups,
                   hidden_, inter_, groups, stream_)
            state["full_epoch"] += 1
            full_lib.launch(ctypes.c_void_p(y_.data_ptr()), ctypes.c_void_p(inv_.data_ptr()),
                            ctypes.c_void_p(sym_full.addrs.data_ptr()), tokens_, tmax, rank, state["full_epoch"], 0,
                            96, ctypes.c_void_p(stream_.cuda_stream), 0, None)

        def tail_fused(a2_, w2_, a2s_, w2s_, tiles_, count_, cap_, sid_, y_, inv_, out_, tokens_, rows_, hidden_,
                       inter_, topk_, stream_):
            state["epoch"] += 1
            ctl.zero_()
            compiled[0](a2_, w2_, a2s_, w2s_, tiles_, count_, sid_, y_, y_, inv_.view(-1), ctl, peers, rows_, tokens_,
                        tokens_, tmax, rank, state["epoch"], ncomm, 1, WBL2, cpx, 0, 2 * CUS, hidden_, inter_, 0,
                        stream_)

        # atomic variant: moe_sorting zeroes the accumulator (aiter accumulate=True, output=layer_acc)
        layer_acc = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
        sorting = M.moe_sorting

        def sort_into_acc(topk_ids, topk_weights, num_experts, hidden_, expert_mask=None, block_size=M.BLOCK_M):
            from aiter.fused_moe import moe_sorting as _sort
            sid_, sw_, seid_, nv_, _ = _sort(topk_ids, topk_weights, num_experts, hidden_, torch.bfloat16,
                                             block_size=block_size, expert_mask=expert_mask, accumulate=True,
                                             output=layer_acc)
            return sid_, sw_, seid_, nv_

        def tail_atomic(a2_, w2_, a2s_, w2s_, tiles_, count_, cap_, sid_, y_, inv_, out_, tokens_, rows_, hidden_,
                        inter_, topk_, stream_):
            state["epoch"] += 1
            ctl.zero_()
            compiled[1](a2_, w2_, a2s_, w2s_, tiles_, count_, sid_, layer_acc, layer_acc, inv_.view(-1), ctl, peers,
                        rows_, tokens_, tokens_, tmax, rank, state["epoch"], ncomm, 1, WBL2, cpx, 4, 2 * CUS, hidden_,
                        inter_, 1, stream_)

        def pyhip_layer(tail, sort=None):
            def run():
                M._down_and_sum = tail
                M.moe_sorting = sort or sorting
                try:
                    b.run_pyhip(c)
                finally:
                    M._down_and_sum, M.moe_sorting = down_and_sum, sorting
            return run

        def with_rccl(fn):
            return lambda: dist.all_reduce(fn())

        top_k_sum = M._top_k_sum

        def sum_ar_only(y_, inv_, out_, tokens_, rows_, hidden_, topk_, stream_):
            state["full_epoch"] += 1
            full_lib.launch(ctypes.c_void_p(y_.data_ptr()), ctypes.c_void_p(inv_.data_ptr()),
                            ctypes.c_void_p(sym_full.addrs.data_ptr()), tokens_, tmax, rank, state["full_epoch"], 0,
                            96, ctypes.c_void_p(stream_.cuda_stream), 0, None)

        def pyhip_fused(tail=None):  # PYHIP_MOE_DOWN=fused: 1-stage moe64 (readme 4.15)
            def run():
                os.environ["PYHIP_MOE_DOWN"] = "fused"
                M._top_k_sum = tail or top_k_sum
                try:
                    return b.run_pyhip(c)
                finally:
                    os.environ.pop("PYHIP_MOE_DOWN")
                    M._top_k_sum = top_k_sum
            return run

        def aiter_twoshot():
            o = b.run_aiter(c)
            state["full_epoch"] += 1
            full_lib.launch(ctypes.c_void_p(o.data_ptr()), ctypes.c_void_p(o.data_ptr()),
                            ctypes.c_void_p(sym_full.addrs.data_ptr()), T, tmax, rank, state["full_epoch"], 1, 96,
                            ctypes.c_void_p(stream.cuda_stream), 0, None)

        go = full_lib.gather_offset(tmax)
        atoms = (0, 1) if ATOMIC else (0,)  # the atomic body needs 256-row tiles (ATOMIC=1)
        for atom in atoms:
            if atom not in compiled:
                acc.zero_()
                fused(0, 0, atom=atom)
        pyhip_layer(tail_sum_ar)()
        torch.cuda.synchronize()
        y_sum_ar = sym_full.storage[go: go + T * H * 2].clone()
        pyhip_layer(tail_fused)()
        check_ctl(f"T={T} layer")
        lbad = torch.tensor(int((sym.storage[gather_off: gather_off + T * H * 2] != y_sum_ar).sum()), device="cuda")
        dist.all_reduce(lbad)
        pyhip_fused(sum_ar_only)()
        torch.cuda.synchronize()
        fbad = torch.tensor(int((sym_full.storage[go: go + T * H * 2] != y_sum_ar).sum()), device="cuda")
        dist.all_reduce(fbad)
        lerr = torch.tensor(0.0, device="cuda")
        if ATOMIC:
            pyhip_layer(tail_atomic, sort_into_acc)()
            check_ctl(f"T={T} layer atomic")
            lerr = torch.tensor(rel_l2(gather(), y_sum_ar.view(torch.bfloat16).view(T, H)), device="cuda")
            dist.all_reduce(lerr, op=dist.ReduceOp.MAX)
        layer = {"aiter only": lambda: b.run_aiter(c), "aiter + rccl": with_rccl(lambda: b.run_aiter(c)),
                 "aiter + twoshot": aiter_twoshot, "pyhip only": lambda: b.run_pyhip(c),
                 "pyhip + rccl": with_rccl(lambda: b.run_pyhip(c)), "pyhip + sum_ar": pyhip_layer(tail_sum_ar),
                 "pyhip + down_ar": pyhip_layer(tail_fused), "pyhip fused only": pyhip_fused(),
                 "pyhip fused + sum_ar": pyhip_fused(sum_ar_only)}
        if ATOMIC:
            layer["pyhip + down_ar atomic"] = pyhip_layer(tail_atomic, sort_into_acc)
        lres = timed_rounds(layer, int(os.environ.get("ROUNDS", "20")))
        if rank == 0:
            print(f"T={T:6d} layer (down_ar {ncomm}:{cpx}, byte mismatches vs sum_ar {int(lbad)}, fused + sum_ar "
                  f"{int(fbad)}, atomic rel_l2 {float(lerr):.2e}): "
                  + "  ".join(f"{k} {v:.1f}" for k, v in lres.items()) + " us", flush=True)
sym.close()
sym_full.close()
dist.destroy_process_group()
