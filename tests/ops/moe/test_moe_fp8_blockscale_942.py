"""Correctness of the gfx942 FP8 block-scale MoE (PyHIP FlyDSL two-stage) against an FP32 reference.

Run: python -m pytest -q tests/ops/moe/test_moe_fp8_blockscale_942.py
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl.compiler")
pytest.importorskip("aiter")

import pyhip  # noqa: E402

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "pyhip"))
if os.path.isdir(_SRC) and _SRC not in pyhip.__path__:
    pyhip.__path__.insert(0, _SRC)

if not torch.cuda.is_available() or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
    pytest.skip("gfx942 only", allow_module_level=True)

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import fused_moe_fp8_blockscale_942, quantize_input  # noqa: E402

FP8 = torch.float8_e4m3fnuz
HIDDEN, EXPERTS, TOPK = 4096, 256, 8


def _q128(a):
    t = a.float().view(a.shape[0], -1, 128)
    s = t.abs().amax(-1, keepdim=True) / 240.0
    return ((t / s.clamp_min(1e-30)).to(FP8).float() * s).view(a.shape[0], -1)


def _case(tokens, inter, local, rank=None, seed=0):
    from aiter.ops.shuffle import shuffle_weight

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
    # Reference: per-1x128 FP8 requantization of both GEMM inputs, local routes only.
    w13d = w13.float() * s13.repeat_interleave(128, 1).repeat_interleave(128, 2)
    w2d = w2.float() * s2.repeat_interleave(128, 1).repeat_interleave(128, 2)
    xq, ref = _q128(x), torch.zeros(tokens, HIDDEN, device="cuda")
    for e in range(local):
        tok, slot = (ids == (rank or 0) * local + e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = xq[tok] @ w13d[e].t()
        a = _q128(torch.nn.functional.silu(h[:, :inter]) * h[:, inter:])
        ref.index_add_(0, tok, (a @ w2d[e].t()) * wts[tok, slot, None])
    weights = (shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16)), s13, s2)
    return x, weights, wts, ids.to(torch.int32), mask, ref


def _rel(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("tokens,inter,local,rank", [
    (3000, 256, 256, None),   # MiMo TP8 shard; token count not a block multiple
    (2048, 512, 256, None),   # TP4 shard
    (2048, 2048, 32, 5),      # EP8 rank without all-to-all (expert_mask)
    (77, 256, 256, None),     # mostly padding rows
])
def test_bf16_input(tokens, inter, local, rank):
    x, (w13, w2, s13, s2), wts, ids, mask, ref = _case(tokens, inter, local, rank)
    out = fused_moe_fp8_blockscale_942(x, w13, w2, wts, ids, s13, s2, expert_mask=mask)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert _rel(out, ref) < 0.01


def test_fp8_input_and_reuse():
    x, (w13, w2, s13, s2), wts, ids, _, ref = _case(2048, 512, 256)
    x8, xs = quantize_input(x)
    a1_scale = torch.as_strided(xs, (x.shape[0], HIDDEN // 128), (HIDDEN // 128, 1))  # bytes [H / 128, T]
    out = torch.full_like(x, float("nan"))
    for _ in range(2):  # second call reuses the cached kernels and workspace
        got = fused_moe_fp8_blockscale_942(x8, w13, w2, wts, ids, s13, s2, a1_scale=a1_scale, out=out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    assert _rel(out, ref) < 0.01
