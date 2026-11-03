"""MI300X check of tingqli's A8W4 TP8 MoE (inter 256): does store+sum fix gemm2's atomic cost, and where does
the 1-stage gain come from? Needs his fork (git clone --branch tq/mimo-moe https://github.com/tingqli/aiter,
checked at 8f43399) with the prebuilt aiter JIT modules of a matching aiter linked into its aiter/jit/.

TQ_AITER=/path/to/fork PYTHONPATH=$TQ_AITER:$TQ_AITER/doc-mimo/experiments/tools/pyhip_shim \
  python tq_store_sum_check.py 8192 16384 32768
G2_NO_ATOMIC=1 FLYDSL_RUNTIME_CACHE_DIR=/tmp/tq_noat ... (same)   -> gemm2 / fused kernel without atomics
"""

import os
import sys
import time

import torch

TQ = os.path.join(os.environ.get("TQ_AITER", "/root/workspace/tq_aiter"), "doc-mimo")
sys.path.insert(0, f"{TQ}/experiments/moe_2k_vs_1k")
import bench as BB  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows  # noqa: E402

F, B, G1 = BB.F, BB.B, BB.G1
E, MD, TOPK, INTER = 256, 4096, 8, 256
NOAT = os.environ.get("G2_NO_ATOMIC") == "1"


def two_kernel_bm128(x, W, tw, ti, mode):
    sids, sw, seids, nv, out = moe_sorting(ti, tw, E, MD, dtypes.bf16, 128)[:5]
    x8, xs = quantize_rows(x)
    rows = seids.numel() * 128
    buf = torch.empty(rows * (INTER + 2 * 4), dtype=torch.uint8, device=x.device)
    G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf,
                  num_experts=E, tile_m=128, epilog="q8s", inter_dim=INTER, topk=TOPK,
                  tile_n=128, b_cache_mod=1, xcd_swizzle=1, dyn_m=True, fast_dec=2)
    a2 = buf[: rows * INTER].view(torch.float8_e4m3fnuz).view(rows, INTER)
    a2s = buf[rows * INTER:].view(torch.float32).view(rows, 2)
    routes = loc = None
    if mode == "sorted":
        routes = torch.empty((rows, MD), dtype=dtypes.bf16, device=x.device)
        loc = torch.empty(x.shape[0] * TOPK, dtype=torch.int32, device=x.device)
    B.a8w4_gemm2_bm128(a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv, out,
                       num_experts=E, n_tiles=8, topk=TOPK, out_mode=mode, routes=routes, loc=loc)
    return out


def loop_us(fn, seconds=2.0):
    """Same as bench_moe_fp8_blockscale.timeit: 10-call bursts for `seconds`, CUDA events around the whole loop."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0, calls = time.time(), 0
    start.record()
    while time.time() - t0 < seconds:
        for _ in range(10):
            fn()
        calls += 10
        torch.cuda.synchronize()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls * 1e3


W = BB.make_weights(INTER)
cands = [
    ("2k bm64 nt4 atomic", lambda x, tw, ti: BB.two_kernel(x, W, tw, ti, bm=64, g1_bcm=1, g2="nt4")),
    ("2k bm128 atomic", lambda x, tw, ti: two_kernel_bm128(x, W, tw, ti, "atomic")),
    ("2k bm128 store+sum", lambda x, tw, ti: two_kernel_bm128(x, W, tw, ti, "sorted")),
    ("1k bm64 (fused)", lambda x, tw, ti: F.fused_moe_a8w4_bm64(x, W, tw, ti, b_cache_mod=1, waves_per_eu=2)),
]
if NOAT:  # only these two kernels honour G2_NO_ATOMIC
    cands = [cands[1], cands[3]]

for tok in [int(a) for a in sys.argv[1:]] or [8192, 16384, 32768]:
    torch.manual_seed(tok)
    x = torch.randn((tok, MD), dtype=dtypes.bf16, device="cuda")
    tw, ti = fused_topk(x, torch.randn((tok, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)
    ref = None
    print(f"== tokens {tok}{' (G2_NO_ATOMIC: outputs wrong)' if NOAT else ''}", flush=True)
    for name, f in cands:
        fn = lambda f=f: f(x, tw, ti)  # noqa: E731
        o = fn().float().clone()
        cos = "" if NOAT else f"cos {G1._cos_diff(ref if ref is not None else o, o):.1e}"
        ref = o if ref is None else ref
        runs = sorted((F._profile(fn) for _ in range(3)), key=lambda r: r["total"])
        t = runs[1]
        stages = "  ".join(f"{k} {v:.0f}" for k, v in sorted(t.items(), key=lambda kv: -kv[1]) if k != "total")
        print(f"   {name:20s} total {t['total']:7.1f} us  loop {loop_us(fn):7.1f} us  {cos}  [{stages}]", flush=True)
