"""Numerics of lossy transports (aiter PR 6514 big-M options) on the TP8 bench case.

1. MoE-output reduce-scatter over 8 TP ranks: BF16 ring (BF16 rounding per hop) vs int8 + FP32 scale
   per 32 values vs int4 + BF16 scale per 32 (E4), FP32 sum at the owner.
2. Route rows Y (sorted, route weight folded in) before the top-k sum: BF16 (current) vs FP8 E4M3FNUZ
   + power-of-two scale per 32 (R8) vs int8 + FP32 scale per 32.
rel_l2 against the FP32 reference of the same quantized MoE (bench reference(), per rank's I slice).

Usage: python transport_numerics.py [tokens]
"""

import os
import sys

import torch

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as m  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096


def rel(a, ref):
    return float((a.float() - ref).norm() / ref.norm())


def q_int(p, bits, scale_dtype):
    t = p.float().view(p.shape[0], -1, 32)
    qmax = 2 ** (bits - 1) - 1
    s = (t.abs().amax(-1, keepdim=True) / qmax).to(scale_dtype).float().clamp_min(1e-30)
    return ((t / s).round().clamp(-qmax, qmax) * s).view_as(p)


def q_fp8_e8m0(y):
    t = y.float().view(y.shape[0], -1, 32)
    amax = t.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / 240.0)))
    return ((t / s).to(torch.float8_e4m3fnuz).float() * s).view_as(y)


c0 = b.make_case("tp8", T, seed=0)
partials, refs = [], []
for r in range(8):  # rank r's I slice: own weights, shared input and routing
    c = c0 if r == 0 else b.make_case("tp8", T, seed=r)
    c.update(x=c0["x"], ids=c0["ids"], wts=c0["wts"])
    out = b.run_pyhip(c).clone()
    refs.append(b.reference(c))
    if r == 0:
        sid, _, _, nv = m.moe_sorting(c["ids"], c["wts"], 256, 4096)
        rows = int(nv[0])
        y = m._buffers[out.device][2][: rows * 4096].view(rows, 4096).clone()
        token = (sid[:rows] & 0xFFFFFF).long()
        real = token < T

        def topk_sum(yy):
            o = torch.zeros(T, 4096, device="cuda")
            o.index_add_(0, token[real], yy[real].float())
            return o.bfloat16()

        print(f"rank 0 MoE output rel_l2: BF16 Y {rel(out, refs[0]) * 100:.3f}%  "
              f"FP8 Y (R8) {rel(topk_sum(q_fp8_e8m0(y)), refs[0]) * 100:.3f}%  "
              f"int8 Y + FP32 scale/32 {rel(topk_sum(q_int(y, 8, torch.float32)), refs[0]) * 100:.3f}%")
    partials.append(out)

exact = torch.stack(refs).sum(0)
ring = partials[0].float()
for p in partials[1:]:
    ring = (ring + p.float()).bfloat16().float()
i8 = sum(q_int(p, 8, torch.float32) for p in partials)
i4 = sum(q_int(p, 4, torch.bfloat16) for p in partials)
print(f"8-rank sum rel_l2 (T={T}): BF16 ring {rel(ring.bfloat16(), exact) * 100:.3f}%  "
      f"int8+scale/32 {rel(i8.bfloat16(), exact) * 100:.3f}%  int4+scale/32 (E4) {rel(i4.bfloat16(), exact) * 100:.3f}%")
