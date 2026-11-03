"""MiMo-V2.6-Flash DP2/TP8 prefill FP8 linears on gfx942: aiter CK bpreshuffle dispatch (sglang default)
vs PyHIP gemm_a8w8_blockscale_942, same 1x128 / 128x128 block-scale inputs, BF16 out; plus the row-strided
X check (activation view of sglang's packed FP8 + scale all-gather, row stride K + K/128*4 bytes): bitwise
equal to the contiguous X and timed.

python benchmarks/gemm/bench_mimo_fp8_linear_942.py   (GPU from HIP_VISIBLE_DEVICES; aiter on PYTHONPATH)
"""
import os
import statistics

import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

import pyhip

pyhip.__path__.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/pyhip"))
from pyhip.ops.gemm.flydsl.gemm_fp8_blockscale_8w_942 import gemm_a8w8_blockscale_942  # noqa: E402

SHAPES = [(16384, 3712, 4096, "SWA QKV (39 layers)"), (16384, 3392, 4096, "full QKV (9 layers)"),
          (32768, 4096, 4096, "dense gate_up (layer 0)"), (32768, 4096, 2048, "dense down (layer 0)"),
          (8192, 3712, 4096, "SWA QKV, 8K chunk"), (4096, 3712, 4096, "SWA QKV, 4K chunk")]
SETS, WARMUP, ITERS, ROUNDS = 4, 10, 30, 3


def inputs(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(dtypes.fp8)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(dtypes.fp8)
    xs = torch.rand(m, k // 128, device="cuda", generator=g) * 0.02 + 0.01
    ws = torch.rand((n + 127) // 128, k // 128, device="cuda", generator=g) * 0.02 + 0.01
    xs_t = xs.t().contiguous()  # [K/128, M]: the bpreshuffle activation-scale bytes
    lda = k + k // 128 * 4
    packet = torch.full((m, lda), 0x7F, dtype=torch.uint8, device="cuda")  # junk where the scales live
    packet[:, :k] = x.view(torch.uint8)
    return dict(x=x, xp=packet[:, :k].view(dtypes.fp8), w=w, xs=xs, xs_t=xs_t, ws=ws,
                wsh=shuffle_weight(w, layout=(16, 16)))


def reference(d):
    n = d["w"].shape[0]
    x = d["x"].float() * d["xs"].repeat_interleave(128, dim=1)
    ws = d["ws"].repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
    return x @ (d["w"].float() * ws).t()


CANDS = {
    "CK": lambda d: aiter.gemm_a8w8_blockscale_bpreshuffle(
        d["x"], d["wsh"], d["xs_t"].view(d["x"].shape[0], -1), d["ws"], dtypes.bf16),
    "PyHIP": lambda d: gemm_a8w8_blockscale_942(d["x"], d["wsh"], d["xs_t"], d["ws"], w_preshuffled=True),
    "PyHIP strided X": lambda d: gemm_a8w8_blockscale_942(d["xp"], d["wsh"], d["xs_t"], d["ws"], w_preshuffled=True),
}


def time_call(fn, sets):
    for i in range(WARMUP):
        fn(sets[i % len(sets)])
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(ITERS):
        fn(sets[i % len(sets)])
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / ITERS * 1e3


torch.set_grad_enabled(False)
print("| shape | M x N x K | CK µs (TFLOPS) | PyHIP µs (TFLOPS) | PyHIP strided X µs | PyHIP/CK | rel_rms CK / PyHIP "
      "| strided == contiguous |")
print("|---|---|---:|---:|---:|---:|---|---|")
for m, n, k, label in SHAPES:
    sets = [inputs(m, n, k, 100 + s) for s in range(SETS)]
    ref = reference(sets[0])
    out = {name: fn(sets[0]) for name, fn in CANDS.items()}
    err = {name: ((y.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()).item()
           for name, y in out.items()}
    times = {name: [] for name in CANDS}
    names = list(CANDS)
    for r in range(ROUNDS):
        for name in (names if r % 2 == 0 else names[::-1]):
            times[name].append(time_call(CANDS[name], sets))
    t = {name: statistics.median(v) for name, v in times.items()}
    f = 2 * m * n * k
    print(f"| {label} | {m}x{n}x{k} | {t['CK']:.0f} ({f / t['CK'] / 1e6:.0f}) | {t['PyHIP']:.0f} "
          f"({f / t['PyHIP'] / 1e6:.0f}) | {t['PyHIP strided X']:.0f} | {t['PyHIP'] / t['CK']:.2f} | "
          f"{err['CK']:.2e} / {err['PyHIP']:.2e} | {torch.equal(out['PyHIP'], out['PyHIP strided X'])} |", flush=True)
    del sets, ref, out
    torch.cuda.empty_cache()
