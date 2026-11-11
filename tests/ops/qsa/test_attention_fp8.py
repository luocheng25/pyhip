"""FP8 K/V through attention(): E4M3FNUZ K/V with per-tensor or per-token FP32 scales, run
with FP8 MFMA (compute="fp8") or dequantized to BF16 (compute="bf16"), against the reference on
the dequantized K/V; slot-table pools, graph replay and input errors."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ops.qsa._attention import (
    _call,
    _gpu,
    _make_case,
    _paged,
    _runtime,
    reference,
)

torch.set_default_device(None)

FP8 = torch.float8_e4m3fnuz
FP8_MAX = 240.0
# Per-(row, head) norm-relative error bounds (mean, worst) of compute="fp8" on these random
# cases: FP8 Q dominates (scores err by ~2**-5 of |q||k|), FP8 probabilities add ~2%; measured
# about 0.047 / 0.12, as for the same rounding emulated in PyTorch.
FP8_TOL = (0.06, 0.16)

CASES = [
    pytest.param((9,), (130,), 3, False, 1, id="no-plan"),
    pytest.param((1024,), (3000,), 12, False, 1, id="union-direct"),
    pytest.param((1024,), (3001,), 6, True, 1, id="shared-nonmultiple"),
    pytest.param((2048,), (30000,), 3, True, 1, id="union-bq32"),
    pytest.param((64,), (70001,), 12, False, 1, id="direct"),
    pytest.param((800,), (70000,), 6, False, 1, id="gated"),
    pytest.param((480, 1000, 301), (11392, 3, 4001), 12, False, 1, id="ragged"),
    pytest.param((768, 768), (0, 5), 12, True, 1, id="two-requests"),
    pytest.param((1024,), (3000,), 12, False, 2, id="hk2-g6"),
    pytest.param((512,), (9000,), 6, True, 2, id="hk2-g3"),
]
SCALES = pytest.mark.parametrize("per_token", (False, True), ids=("tensor", "token"))


@pytest.fixture(params=(None, 0), ids=("default-plan", "always-plan"))
def union_min_rows(request, monkeypatch):
    if request.param is not None:
        monkeypatch.setattr(_runtime, "_union_min_rows", lambda group: request.param)
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    return request.param


def _quantize(x, per_token):
    """E4M3FNUZ x and its FP32 scale amax / 240, per (token, head) [N, HK] or per tensor [1]."""
    x = x.float()
    amax = x.abs().amax(-1) if per_token else x.abs().amax().reshape(1)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    x8 = (x / (scale[..., None] if per_token else scale)).clamp(-FP8_MAX, FP8_MAX)
    return x8.to(FP8), scale.contiguous()


def _dequant(x8, scale):
    return x8.float() * (scale[..., None] if scale.ndim == 2 else scale)


def _fp8_case(
    queries, prefixes, heads, shared=False, kv_heads=1, per_token=True, seed=17
):
    """A random case whose K/V tokens span 2**-1 .. 2**1 in magnitude, quantized to FP8; the
    reference inputs hold the dequantized K/V."""
    device = _gpu()
    value = _make_case(queries, prefixes, heads, device, seed, shared, kv_heads)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    fp8 = []
    for x in (value.k, value.v):
        spread = torch.rand(x.shape[:2], generator=generator, device=x.device) * 2 - 1
        fp8 += _quantize(x.float() * torch.exp2(spread)[..., None], per_token)
    k8, ks, v8, vs = fp8
    exact = SimpleNamespace(
        **{**vars(value), "k": _dequant(k8, ks), "v": _dequant(v8, vs)}
    )
    return SimpleNamespace(value=value, k8=k8, v8=v8, ks=ks, vs=vs, exact=exact)


def _call_fp8(
    case, compute="fp8", out=None, k=None, v=None, ks=None, vs=None, slots=None
):
    value = case.value
    return _runtime.attention(
        value.q,
        case.k8 if k is None else k,
        case.v8 if v is None else v,
        value.indices,
        query_lens=value.query_lens,
        prefix_lens=value.prefix_lens,
        softmax_scale=value.scale,
        out=out,
        kv_slots=slots,
        k_scale=case.ks if ks is None else ks,
        v_scale=case.vs if vs is None else vs,
        compute=compute,
    )


def _sample_rows(value, count=64):
    """Request edges plus evenly spaced rows."""
    edges = np.cumsum((0, *value.query_lens))
    rows = {*np.linspace(0, len(value.q) - 1, count).astype(int).tolist()}
    rows |= {int(x) for x in (*edges[:-1], *(edges[1:] - 1)) if 0 <= x < len(value.q)}
    return sorted(rows)


def _within_fp8_error(output, expected):
    """Mean and worst norm-relative error over (row, head) vectors within FP8_TOL."""
    diff = (output.float() - expected).norm(dim=-1)
    error = diff / expected.norm(dim=-1).clamp_min(1e-6)
    return float(error.mean()) < FP8_TOL[0] and float(error.max()) < FP8_TOL[1]


@SCALES
@pytest.mark.parametrize("queries,prefixes,heads,shared,kv_heads", CASES)
def test_fp8_matches_reference(
    queries, prefixes, heads, shared, kv_heads, per_token, union_min_rows
):
    """FP8 MFMA output is within FP8 error of the reference on the dequantized K/V; BF16
    compute equals BF16 attention on the dequantized K/V bit for bit."""
    case = _fp8_case(queries, prefixes, heads, shared, kv_heads, per_token)
    rows = _sample_rows(case.value)
    expected = reference(case.exact, rows)
    output = _call_fp8(case)
    assert bool(torch.isfinite(output.float()).all())
    assert _within_fp8_error(output[rows], expected)
    again = _call_fp8(case)
    assert torch.equal(again.view(torch.int16), output.view(torch.int16))

    bf16 = SimpleNamespace(
        **{
            **vars(case.value),
            "k": case.exact.k.to(torch.bfloat16),
            "v": case.exact.v.to(torch.bfloat16),
        }
    )
    dequantized = _call_fp8(case, compute="bf16")
    assert torch.equal(dequantized.view(torch.int16), _call(bf16).view(torch.int16))


@SCALES
@pytest.mark.parametrize(
    "queries,prefixes,heads",
    [
        pytest.param((1024,), (3000,), 12, id="union-direct"),
        pytest.param((64,), (70001,), 3, id="direct"),
        pytest.param((480, 1000, 301), (11392, 3, 4001), 6, id="ragged"),
    ],
)
def test_fp8_uniform_exact(queries, prefixes, heads, per_token, monkeypatch):
    """K = 0 makes every selected token weigh the same; power-of-two V scales keep the scaled
    probabilities exact, so the output is the selected tokens' mean V up to rounding."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    value = _make_case(queries, prefixes, heads, _gpu())
    generator = torch.Generator(device=value.q.device).manual_seed(3)
    tokens = value.k.shape[:2]
    shape = tokens if per_token else (1,)
    exponent = torch.randint(-3, 2, shape, generator=generator, device=value.q.device)
    v8 = torch.randn(value.v.shape, generator=generator, device=value.q.device) * 60
    v8 = v8.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    case = SimpleNamespace(
        value=value,
        k8=torch.zeros(value.k.shape, dtype=FP8, device=value.q.device),
        v8=v8,
        ks=torch.ones(shape, device=value.q.device),
        vs=torch.exp2(exponent.float()).contiguous(),
    )
    exact = SimpleNamespace(
        **{**vars(value), "k": value.k * 0, "v": _dequant(v8, case.vs)}
    )
    output = _call_fp8(case)
    rows = _sample_rows(value, 96)
    torch.testing.assert_close(
        output[rows].float(), reference(exact, rows), rtol=0.01, atol=2e-3
    )


def _fp8_pools(case, slots_from):
    """K/V pools holding the packed FP8 tokens at the slots of `slots_from`; every other pool
    row and its per-token scales are NaN."""
    *_, slots = slots_from
    rows = slots_from[0].shape[0]
    pools = []
    for x8, scale in ((case.k8, case.ks), (case.v8, case.vs)):
        pool = torch.full(
            (rows, *x8.shape[1:]), 0x80, dtype=torch.uint8, device=x8.device
        )
        pool[slots.long()] = x8.view(torch.uint8)
        if scale.ndim == 2:
            table = torch.full((rows, scale.shape[1]), float("nan"), device=x8.device)
            table[slots.long()] = scale
            scale = table
        pools += [pool.view(FP8), scale]
    return pools, slots


@SCALES
@pytest.mark.parametrize(
    "queries,prefixes,heads,shared,kv_heads",
    [
        pytest.param((9,), (130,), 3, False, 1, id="no-plan"),
        pytest.param((1024,), (3001,), 6, True, 1, id="shared-nonmultiple"),
        pytest.param((64,), (70001,), 12, False, 1, id="direct"),
        pytest.param((480, 1000, 301), (11392, 3, 4001), 12, False, 1, id="ragged"),
        pytest.param((1024,), (3000,), 12, False, 2, id="hk2-g6"),
    ],
)
def test_fp8_kv_slots_match_packed(
    queries, prefixes, heads, shared, kv_heads, per_token, union_min_rows
):
    """FP8 K/V and per-token scales read from a paged pool through kv_slots give the packed
    result bit for bit, in both compute modes."""
    case = _fp8_case(queries, prefixes, heads, shared, kv_heads, per_token)
    (pool_k, k_scale, pool_v, v_scale), slots = _fp8_pools(case, _paged(case.value))
    for compute in ("fp8", "bf16"):
        expected = _call_fp8(case, compute=compute)
        output = _call_fp8(
            case, compute, k=pool_k, v=pool_v, ks=k_scale, vs=v_scale, slots=slots
        )
        assert bool(torch.isfinite(output.float()).all())
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@SCALES
def test_fp8_graph_replay(per_token, monkeypatch):
    """A captured FP8 call replayed with Q/K/V, scales and indices rewritten in place matches
    the eager call on the new inputs."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    first = _fp8_case((1024,), (3000,), 12, per_token=per_token)
    second = _fp8_case((1024,), (3000,), 12, True, per_token=per_token, seed=29)
    second.value.q = second.value.q.flip(0).contiguous()
    names = ("k8", "v8", "ks", "vs")
    live = SimpleNamespace(
        value=SimpleNamespace(**vars(first.value)),
        **{name: getattr(first, name).clone() for name in names},
    )
    live.value.q = first.value.q.clone()
    live.value.indices = first.value.indices.clone()
    out = torch.empty_like(live.value.q)
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call_fp8(live, out=out)
        with torch.cuda.graph(graph, stream=stream):
            _call_fp8(live, out=out)
        for name in names:
            getattr(live, name).copy_(getattr(second, name))
        live.value.q.copy_(second.value.q)
        live.value.indices.copy_(second.value.indices)
        out.fill_(float("nan"))
        graph.replay()
        expected = _call_fp8(second)
    stream.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    rows = _sample_rows(second.value)
    exact = SimpleNamespace(**{**vars(second.exact), "q": second.value.q})
    assert _within_fp8_error(out[rows], reference(exact, rows))


def test_fp8_input_errors():
    case = _fp8_case((9,), (130,), 3, per_token=True)
    value, ks = case.value, case.ks
    one = torch.ones(1, device=value.q.device)
    with pytest.raises(ValueError, match="apply only to FP8"):
        _runtime.attention(
            value.q, value.k, value.v, value.indices, k_scale=one, v_scale=one
        )
    with pytest.raises(ValueError, match="k_scale"):
        _runtime.attention(value.q, case.k8, case.v8, value.indices)
    for bad in (1.0, ks.to(torch.bfloat16), ks[:-1], ks.repeat(1, 2)[:, :1], one):
        with pytest.raises(ValueError, match="k_scale"):
            _call_fp8(case, ks=bad)
    with pytest.raises(ValueError, match="compute"):
        _call_fp8(case, compute="fp16")
    with pytest.raises(ValueError, match="BF16"):
        _call_fp8(case, v=value.v)
    # out overlapping the K scales.
    buffer = torch.zeros(value.q.numel(), dtype=torch.bfloat16, device=value.q.device)
    alias = buffer.view(torch.float32)[: ks.numel()].view(ks.shape)
    with pytest.raises(ValueError, match="overlap"):
        _call_fp8(case, ks=alias, out=buffer.view(value.q.shape))
