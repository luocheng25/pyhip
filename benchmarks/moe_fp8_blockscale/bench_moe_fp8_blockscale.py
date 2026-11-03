"""PyHIP gfx942 FP8 block-scale MoE (prefill) vs aiter fused_moe, MiMo-V2.6-Flash shapes.

Configurations (H 4096, 256 experts, top-8, 128x128 FP8 block scales, BF16 input/output):
  tp8  each rank: all 256 experts, I = 256   (current MiMo deployment: global MoE TP8)
  tp4  each rank: all 256 experts, I = 512
  ep8  each rank: 32 experts, I = 2048, global top-8 ids + expert_mask (sglang EP without all-to-all)
Timing: CUDA events over a sustained loop (default 2 s) of the full call, i.e. input quantization, sorting,
GEMMs and top-k reduction; same inputs and weights for both implementations.

usage: python benchmarks/moe_fp8_blockscale/bench_moe_fp8_blockscale.py [--configs tp8 tp4 ep8]
       [--tokens 4096 8192 16384 32768] [--check] [--breakdown] [--seconds 2]
"""

import argparse
import collections
import os
import time

import torch

import pyhip

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "pyhip")
if os.path.isdir(_SRC) and os.path.abspath(_SRC) not in pyhip.__path__:
    pyhip.__path__.insert(0, os.path.abspath(_SRC))

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import fused_moe_fp8_blockscale_942  # noqa: E402

FP8 = torch.float8_e4m3fnuz
HIDDEN, EXPERTS, TOPK = 4096, 256, 8
CONFIGS = {"tp8": (256, 256, None), "tp4": (512, 256, None), "ep8": (2048, 32, 3)}


def make_case(config, tokens, seed=0):
    from aiter.ops.shuffle import shuffle_weight

    inter, local, rank = CONFIGS[config]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    w13 = (torch.randn(local, 2 * inter, HIDDEN, device="cuda", generator=g) * 0.5).to(FP8)
    w2 = (torch.randn(local, HIDDEN, inter, device="cuda", generator=g) * 0.5).to(FP8)
    s13 = torch.rand(local, 2 * inter // 128, HIDDEN // 128, device="cuda", generator=g) * 0.02 + 0.005
    s2 = torch.rand(local, HIDDEN // 128, inter // 128, device="cuda", generator=g) * 0.02 + 0.005
    wts, ids = torch.topk(torch.sigmoid(torch.randn(tokens, EXPERTS, device="cuda", generator=g)), TOPK, -1)
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    mask = None
    if rank is not None:
        mask = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
        mask[rank * local:(rank + 1) * local] = 1
    w13_sh, w2_sh = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    w13_sh.is_shuffled = w2_sh.is_shuffled = True
    return dict(x=x, w13=w13, w2=w2, w13_sh=w13_sh, w2_sh=w2_sh, s13=s13, s2=s2, wts=wts,
                ids=ids.to(torch.int32), mask=mask, inter=inter, local=local, rank=rank)


def routed_flops(c):
    ids = c["ids"]
    routes = ids.numel() if c["mask"] is None else int(c["mask"][ids.long()].sum())
    return 2 * routes * (HIDDEN * 2 * c["inter"] + c["inter"] * HIDDEN)


def run_pyhip(c):
    return fused_moe_fp8_blockscale_942(c["x"], c["w13_sh"], c["w2_sh"], c["wts"], c["ids"], c["s13"], c["s2"],
                                        expert_mask=c["mask"])


def run_aiter(c):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    mask, ids, wts = c["mask"], c["ids"], c["wts"]
    if mask is not None:
        # sglang's routed-only EP convention: one fake slot to a masked-off sink expert.
        mask = torch.cat((mask, mask.new_zeros(1)))
        ids = torch.cat((ids, ids.new_full((ids.shape[0], 1), EXPERTS)), -1)
        wts = torch.cat((wts, wts.new_zeros((wts.shape[0], 1))), -1)
    return fused_moe(c["x"], c["w13_sh"], c["w2_sh"], wts, ids, quant_type=QuantType.per_128x128,
                     activation=ActivationType.Silu, w1_scale=c["s13"], w2_scale=c["s2"], expert_mask=mask,
                     dtype=torch.bfloat16)


def reference(c):
    """FP32 reference with per-1x128 FP8 requantization of both GEMM inputs (local routes only)."""
    def q128(a):
        t = a.float().view(a.shape[0], -1, 128)
        s = t.abs().amax(-1, keepdim=True) / 240.0
        return ((t / s.clamp_min(1e-30)).to(FP8).float() * s).view(a.shape[0], -1)

    inter, base = c["inter"], (c["rank"] or 0) * c["local"]
    w13 = c["w13"].float() * c["s13"].repeat_interleave(128, 1).repeat_interleave(128, 2)
    w2 = c["w2"].float() * c["s2"].repeat_interleave(128, 1).repeat_interleave(128, 2)
    xq, out = q128(c["x"]), torch.zeros(c["x"].shape[0], HIDDEN, device="cuda")
    for e in range(c["local"]):
        tok, slot = (c["ids"].long() == base + e).nonzero(as_tuple=True)
        h = xq[tok] @ w13[e].t()
        a = q128(torch.nn.functional.silu(h[:, :inter]) * h[:, inter:])
        out.index_add_(0, tok, (a @ w2[e].t()) * c["wts"][tok, slot, None])
    return out


def timeit(fn, seconds):
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


def breakdown(fn, reps=10):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    totals = collections.Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            totals[ev.name[:60]] += ev.device_time_total / reps
    return totals


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["tp8", "tp4", "ep8"], choices=list(CONFIGS))
    parser.add_argument("--tokens", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--check", action="store_true", help="also compare with the FP32 reference (T <= 8192)")
    parser.add_argument("--breakdown", action="store_true", help="per-kernel GPU time (torch profiler)")
    args = parser.parse_args()
    print(f"device {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).gcnArchName})")
    print(f"{'config':6s} {'tokens':>7s} {'pyhip us':>10s} {'aiter us':>10s} {'speedup':>8s} "
          f"{'pyhip T':>8s} {'aiter T':>8s} {'rel vs aiter':>12s} {'rel vs ref':>10s}")
    for config in args.configs:
        for tokens in args.tokens:
            c = make_case(config, tokens)
            y, ya = run_pyhip(c), run_aiter(c)
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(y.float()).all())
            r_ref = f"{rel(y, reference(c)):.4f}" if args.check and tokens <= 8192 else "-"
            tp, ta = timeit(lambda: run_pyhip(c), args.seconds), timeit(lambda: run_aiter(c), args.seconds)
            f = routed_flops(c)
            print(f"{config:6s} {tokens:7d} {tp:10.1f} {ta:10.1f} {ta / tp:7.2f}x {f / tp / 1e6:8.0f} "
                  f"{f / ta / 1e6:8.0f} {rel(y, ya):12.4f} {r_ref:>10s}{'' if ok else '  NON-FINITE'}", flush=True)
            if args.breakdown:
                for name, fn in (("pyhip", lambda: run_pyhip(c)), ("aiter", lambda: run_aiter(c))):
                    parts = breakdown(fn)
                    print(f"    {name}: " + ", ".join(f"{k.split('(')[0][:40]} {v:.0f}"
                                                      for k, v in parts.most_common()), flush=True)


if __name__ == "__main__":
    main()
