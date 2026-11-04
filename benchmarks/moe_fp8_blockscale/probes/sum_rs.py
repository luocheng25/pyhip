"""Fused MoE top-k sum + reduce-scatter (sum_rs.hip) vs sum kernel + RCCL reduce-scatter, 8x MI300X.

torchrun --nproc_per_node 8 sum_rs.py [tokens...]   (env GRID=96, ITERS=20, SKEW=0: DP-group-1 tokens fraction)

Every rank: Y [T*TOPK, H] bf16 route rows, INV [T, TOPK] (a random permutation), same gathered batch of T
tokens; rank s owns its shard rows (SKEW > 0: ranks 4-7 get SKEW of the DP-group-0 shard size, as with two DP
groups of different lengths). Checks the fused output bit-exactly against: per-rank partial = bf16(fp32 sum
over k), fp32 sum over ranks in rank order, bf16, own rows -- several epochs with fresh data. Times (slowest
rank, median):
  sum      : pyhip _sum_kernel (one CTA per token, nt loads) -> out [T, H]
  rs       : dist.reduce_scatter_tensor(out) (equal shards) / reduce_scatter of padded shards
  sum+rs   : both back to back
  rsonly   : sum_rs mode 1 (reduce-scatter of a ready partial)
  fused    : sum_rs mode 0 (top-k sum + reduce-scatter in one kernel)
"""

import ctypes
import os
import subprocess
import tempfile

import torch
import torch.distributed as dist

H, NPES, TOPK = 4096, 8, 8
HERE = os.path.dirname(os.path.abspath(__file__))


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def _check(err, what):
    assert err == 0, f"{what} failed: {err}"


class FusedSumRS:
    """sum_rs.hip on this rank: one symmetric buffer (rcap rows per source) mapped into every rank."""

    def __init__(self, rank, rcap, grid=96):
        so = os.path.join(tempfile.gettempdir(), f"sum_rs_{rank}.so")
        subprocess.check_call(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx942", "-shared", "-fPIC",
                               os.path.join(HERE, "sum_rs.hip"), "-o", so])
        self.lib = ctypes.CDLL(so)
        self.lib.sym_bytes.restype = ctypes.c_uint64
        self.lib.sym_bytes.argtypes = [ctypes.c_int]
        self.hip = ctypes.CDLL("libamdhip64.so")
        self.rank, self.rcap, self.grid, self.epoch = rank, rcap, grid, 0
        self.storage = torch.zeros(self.lib.sym_bytes(rcap), dtype=torch.uint8, device="cuda")
        local = self.storage.data_ptr()
        base = ctypes.c_void_p()
        _check(self.hip.hipPointerGetAttribute(ctypes.byref(base), 11, ctypes.c_void_p(local)), "range start")
        handle = Handle()
        _check(self.hip.hipIpcGetMemHandle(ctypes.byref(handle), base), "hipIpcGetMemHandle")
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
            _check(self.hip.hipIpcOpenMemHandle(ctypes.byref(mapped), h, 1), "hipIpcOpenMemHandle")
            self.opened.append(mapped.value)
            addrs.append(mapped.value + off)
        self.addrs = (ctypes.c_uint64 * NPES)(*addrs)

    def __call__(self, y, inv, out, sizes, mode=0, tim=None):
        assert max(sizes) <= self.rcap and out.shape[0] == sizes[self.rank]
        self.epoch += 1
        self.lib.launch(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(inv.data_ptr()),
                        ctypes.c_void_p(out.data_ptr()), self.addrs, (ctypes.c_int * NPES)(*sizes), self.rcap,
                        self.rank, self.epoch, mode, self.grid,
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                        ctypes.c_void_p(tim.data_ptr() if tim is not None else 0))
        return out

    def close(self):
        torch.cuda.synchronize()
        dist.barrier()
        for p in self.opened:
            _check(self.hip.hipIpcCloseMemHandle(ctypes.c_void_p(p)), "hipIpcCloseMemHandle")


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


def shard_sizes(tokens, skew):
    """Per-rank shard rows: DP group 0 (ranks 0-3) and DP group 1 (ranks 4-7, skew x as many tokens)."""
    if skew <= 0:
        return [tokens // NPES] * NPES
    r0 = int(tokens / (NPES // 2) / (1 + skew))
    r1 = tokens // (NPES // 2) - r0
    return [r0] * (NPES // 2) + [r1] * (NPES // 2)


def main():
    import sys

    import pyhip

    pyhip.__path__.insert(0, os.path.join(HERE, "..", "..", "..", "src", "pyhip"))
    from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as K
    from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M

    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
    rank = dist.get_rank()
    assert dist.get_world_size() == NPES
    tokens = [int(t) for t in sys.argv[1:]] or [8192, 16384, 32768]
    skew = float(os.environ.get("SKEW", "0"))
    grid, iters = int(os.environ.get("GRID", "96")), int(os.environ.get("ITERS", "20"))
    rcap = max(max(shard_sizes(t, skew)) for t in tokens)
    rs_fused = FusedSumRS(rank, rcap, grid)
    stream = torch.cuda.current_stream()
    tim = torch.zeros(grid * 3, dtype=torch.int64, device="cuda")
    for T in tokens:
        sizes = shard_sizes(T, skew)
        T = sum(sizes)
        off = [sum(sizes[:i]) for i in range(NPES + 1)]
        mine = slice(off[rank], off[rank + 1])
        rows = T * TOPK
        g = torch.Generator(device="cuda").manual_seed(1234 + rank)
        y = (torch.randn(rows, H, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
        inv = torch.randperm(rows, device="cuda", generator=g).to(torch.int32).view(T, TOPK)
        out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
        got = torch.empty(sizes[rank], H, dtype=torch.bfloat16, device="cuda")
        bad = 0
        for rep in range(3):  # fresh data every epoch
            if rep:
                y.mul_(-1.0)
            partial = y[inv[:, 0].long()].float()  # k in order, as the kernel
            for k in range(1, TOPK):
                partial += y[inv[:, k].long()].float()
            partial = partial.to(torch.bfloat16)
            parts = [torch.empty_like(partial) for _ in range(NPES)]
            dist.all_gather(parts, partial)
            ref = parts[0].float()
            for p in parts[1:]:
                ref = ref + p.float()
            ref = ref.to(torch.bfloat16)[mine]
            rs_fused(y, inv, got, sizes, 0)
            torch.cuda.synchronize()
            bad += int((got != ref).sum())
            rs_fused(partial, inv, got, sizes, 1)
            torch.cuda.synchronize()
            bad += int((got != ref).sum())

        def sum_only():
            M._run(("sum", out.device, H, H, TOPK), K.launch_sum, y.view(-1), inv.view(-1), out.view(-1), T, rows,
                   0, T, H, H, TOPK, stream.cuda_stream)

        equal = len(set(sizes)) == 1
        shard = torch.empty(max(sizes), H, dtype=torch.bfloat16, device="cuda")
        padded = torch.zeros(NPES * max(sizes), H, dtype=torch.bfloat16, device="cuda")

        def rccl_rs():
            if equal:
                dist.reduce_scatter_tensor(shard, out)
            else:  # what reduce_scatterv costs at best: pad every shard to the largest
                for s in range(NPES):
                    padded[s * max(sizes):s * max(sizes) + sizes[s]].copy_(out[off[s]:off[s + 1]])
                dist.reduce_scatter_tensor(shard, padded)

        res = {
            "sum": timed(sum_only, iters),
            "rs": timed(rccl_rs, iters),
            "sum+rs": timed(lambda: (sum_only(), rccl_rs()), iters),
            "rsonly": timed(lambda: rs_fused(partial, inv, got, sizes, 1), iters),
            "fused": timed(lambda: rs_fused(y, inv, got, sizes, 0), iters),
        }
        b = torch.tensor(bad, device="cuda")
        dist.all_reduce(b)
        rs_fused(y, inv, got, sizes, 0, tim)
        torch.cuda.synchronize()
        st = (tim.view(grid, 3) - tim.view(grid, 3)[:, :1].min()).double() / 100.0  # 100 MHz ticks -> us
        phases = "  ".join(f"{n} med {float(st[:, i].median()):.0f} max {float(st[:, i].max()):.0f}"
                           for i, n in ((1, "RS end"), (2, "reduce end")))
        if rank == 0:
            print(f"T={T:6d} sizes {sizes[0]}/{sizes[-1]} grid {grid}: mismatches {int(b)}  "
                  + "  ".join(f"{k} {v:7.1f}" for k, v in res.items())
                  + f" us | fused vs sum+rs {res['fused'] / res['sum+rs']:.2f}x", flush=True)
            print(f"    rank 0 phases (us from first WG start): {phases}", flush=True)
    rs_fused.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
