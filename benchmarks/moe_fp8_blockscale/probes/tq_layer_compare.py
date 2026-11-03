"""tingqli's A8W4 TP8 MoE layers including the all-reduce (TP8 I=256, 8x MI300X), for the layer comparison of
readme 4.11. Every variant includes moe_sorting + x row-quant, and its output is the all-reduced [T, H] bf16:
  1k + rccl / 1k + twoshot : fused 1-stage fmoe_bm64 (bf16 atomics) + RCCL all_reduce / his two-shot all-reduce
  2k nt4 + rccl / + twoshot : bm64 gemm1 (q8s) + gemm2 nt4 (his best 2-kernel config) + the same two all-reduces
  2k base + twoshot         : bm64 gemm1 + gemm2 base (the variant the fused all-reduce supports) + two-shot
  2k base + gemm2_ar        : bm64 gemm1 + gemm2 with the chunked two-shot all-reduce fused in (gemm2_ar_fused.py)
Needs his fork (see tq_store_sum_check.py):

TQ_AITER=... PYTHONPATH=$TQ_AITER:$TQ_AITER/doc-mimo/experiments/tools/pyhip_shim \
  torchrun --nproc_per_node 8 tq_layer_compare.py 8192 16384 32768     (env CHUNK=256 CWG=64 ROUNDS=20)
"""

import os
import sys

import torch
import torch.distributed as dist

TQ = os.path.join(os.environ.get("TQ_AITER", "/root/workspace/tq_aiter"), "doc-mimo")
sys.path.insert(0, f"{TQ}/experiments/moe_2k_vs_1k")
sys.path.insert(0, f"{TQ}/kernels")
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
rank = dist.get_rank()
import bench as BB  # noqa: E402
import twoshot_allreduce as tsar  # noqa: E402
from gemm2_ar_fused import FusedGemm2AR  # noqa: E402
from test_a8w4_gemm2 import a8w4_gemm2  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows  # noqa: E402

F, G1 = BB.F, BB.G1
E, H, TOPK, INTER, BM = 256, 4096, 8, 256, 64
CHUNKS = [int(v) for v in os.environ.get("CHUNK", "256").split(",")]
CWGS = [int(v) for v in os.environ.get("CWG", "64").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "20"))
# gemm2 base config of his bm64 2-kernel (bench.two_kernel) on the q8s output of gemm1
G2KW = dict(num_experts=E, tile_m=BM, tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True, a_group=128,
            fast_dec=2)


def gemm1(x, tw, ti):
    groups = INTER // 128
    sids, sw, seids, nv, out = moe_sorting(ti, tw, E, H, dtypes.bf16, BM)[:5]
    x8, xs = quantize_rows(x)
    rows = seids.numel() * BM
    buf = torch.empty(rows * (INTER + groups * 4), dtype=torch.uint8, device="cuda")
    G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf, num_experts=E, tile_m=BM,
                  epilog="q8s", inter_dim=INTER, topk=TOPK, tile_n=128, b_cache_mod=1, xcd_swizzle=1, dyn_m=True,
                  fast_dec=2)
    nd = rows * INTER
    a2 = buf[:nd].view(torch.float8_e4m3fnuz).view(rows, INTER)
    a2s = buf[nd:].view(torch.float32).view(rows, groups)
    return (a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv), out


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


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


W = BB.make_weights(INTER)
for T in [int(t) for t in sys.argv[1:]] or [8192, 16384, 32768]:
    torch.manual_seed(T)
    x = torch.randn((T, H), dtype=dtypes.bf16, device="cuda")
    tw, ti = fused_topk(x, torch.randn((T, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)
    ts = tsar.TwoShotAllreduce((T, H))
    fused = {(ch, cwg): FusedGemm2AR(H, mmax=max(2048, T), cwg=cwg, vpt=1, chunk_n=ch) for ch in CHUNKS for cwg in CWGS}

    def one():
        return F.fused_moe_a8w4_bm64(x, W, tw, ti, b_cache_mod=1, waves_per_eu=2)

    def two_nt4():
        return BB.two_kernel(x, W, tw, ti, bm=BM, g1_bcm=1, g2="nt4")

    def two_base():
        args, out = gemm1(x, tw, ti)
        a8w4_gemm2(*args, out, **G2KW)
        return out

    def two_fused(f):
        def run():
            args, out = gemm1(x, tw, ti)
            return f(a8w4_gemm2, *args, out=out, **G2KW)
        return run

    def rccl(fn):
        def run():
            out = fn()
            dist.all_reduce(out)
            return out
        return run

    def twoshot(fn):
        return lambda: ts(fn())

    variants = {"1k only": one, "1k + rccl": rccl(one), "1k + twoshot": twoshot(one),
                "2k nt4 only": two_nt4, "2k nt4 + rccl": rccl(two_nt4), "2k nt4 + twoshot": twoshot(two_nt4),
                "2k base only": two_base, "2k base + twoshot": twoshot(two_base)}
    variants.update({f"2k base + gemm2_ar ch{ch} c{cwg}": two_fused(f) for (ch, cwg), f in fused.items()})
    # numerics: fused vs unfused 2-kernel (atomic order differs), 1-kernel vs 2-kernel
    y_fused = two_fused(next(iter(fused.values())))().clone()
    y_base = twoshot(two_base)().clone()
    y_one = twoshot(one)().clone()
    torch.cuda.synchronize()
    check = f"rel_l2 gemm2_ar vs 2k base+twoshot {rel(y_fused, y_base):.1e}, 1k vs 2k {rel(y_one, y_base):.1e}"
    res = timed_rounds(variants, ROUNDS)
    if rank == 0:
        print(f"T={T:6d} {check}\n   " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()) + " us", flush=True)
    for f in fused.values():
        f.close()
    ts.close()
dist.barrier()
dist.destroy_process_group()
