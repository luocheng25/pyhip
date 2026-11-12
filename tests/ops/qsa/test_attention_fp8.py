"""FP8 K/V through attention(): E4M3FNUZ K/V with per-tensor or per-token FP32 scales, run
with FP8 MFMA (compute="fp8") or dequantized to BF16 (compute="bf16"), against the reference on
the dequantized K/V; slot-table pools, graph replay and input errors."""

from collections import OrderedDict
import math
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
        monkeypatch.setattr(
            _runtime, "_union_min_rows", lambda group, fp8=None: request.param
        )
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
    probabilities exact, so the output is the selected tokens' mean V up to rounding (whole
    unions; test_fp8_core_residual_uniform covers core/residual tiles)."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    monkeypatch.setattr(_runtime.prepare, "CORE_RATIOS", {})
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


@SCALES
@pytest.mark.parametrize(
    "queries,prefixes,heads",
    [
        pytest.param((1024,), (3000,), 12, id="union-direct"),
        pytest.param((64,), (70001,), 3, id="direct"),
    ],
)
def test_fp8_attention_sink(queries, prefixes, heads, per_token, monkeypatch):
    """Every row selects an attention sink (token 0) over a flat tail whose 2047 keys weigh
    2**-12 of it each (a third of the mass) with similar values: P' of the tail must stay
    above the FP8 underflow. With the row max at P' = 1 it flushed to zero and the outputs
    lost a quarter of their norm."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    case = _fp8_case(queries, prefixes, heads, per_token=per_token)
    value = case.value
    device = value.q.device
    generator = torch.Generator(device=device).manual_seed(5)
    direction = torch.randn(256, generator=generator, device=device)
    direction = direction / direction.norm()
    value.q = (16 * direction).expand(value.q.shape).to(torch.bfloat16).contiguous()
    k = 0.1 * torch.randn(value.k.shape, generator=generator, device=device)
    # Scores in log2 units: q . k * scale * log2(e) = 12 for the sink, ~0 for the tail.
    k[0, :] = direction * (12 / (16 * value.scale * math.log2(math.e)))
    v = torch.randn(256, generator=generator, device=device).expand(value.v.shape)
    v = v + 0.3 * torch.randn(value.v.shape, generator=generator, device=device)
    case.k8, case.ks = _quantize(k, per_token)
    case.v8, case.vs = _quantize(v, per_token)
    blocks = value.indices[:, :2048:4] // 4
    missing = ~(blocks == 0).any(1) & (value.indices[:, 2047] >= 0)
    value.indices[missing, :4] = torch.arange(4, dtype=torch.int32, device=device)
    exact = SimpleNamespace(
        **{
            **vars(case.exact),
            "q": value.q,
            "indices": value.indices,
            "k": _dequant(case.k8, case.ks),
            "v": _dequant(case.v8, case.vs),
        }
    )
    rows = _sample_rows(value)
    expected = reference(exact, rows)
    error = (_call_fp8(case)[rows].float() - expected).norm(dim=-1)
    error = error / expected.norm(dim=-1)
    assert float(error.mean()) < 0.01 and float(error.max()) < 0.03, (
        error.mean(),
        error.max(),
    )


def _partial_indices(rows, prefix, seed=23):
    """Rows choosing 256 of their 512 blocks from a set their group of 32 rows shares and 256 of
    their own: union tiles hold a core every row selects plus each row's residual blocks."""
    rng = np.random.default_rng(seed)
    indices = np.full((rows, 2051), -1, dtype=np.int32)
    for local in range(rows):
        visible = prefix + local + 1
        blocks = visible // 4
        if local % 32 == 0:
            shared = rng.choice(prefix // 4, 256, replace=False)
        own = rng.choice(np.setdiff1d(np.arange(blocks), shared), 256, replace=False)
        chosen = np.sort(np.concatenate((shared, own)))
        tokens = np.concatenate(
            ((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(blocks * 4, visible))
        )
        indices[local, : len(tokens)] = tokens
    return indices


@SCALES
@pytest.mark.parametrize("heads,rows", ((12, 1024), (6, 2048)))
def test_fp8_core_residual(heads, rows, per_token, monkeypatch):
    """Union tiles keep the core their rows share; each row's other blocks run in the direct
    kernel and merge into the union output by log-sum-exp. Against the fp8_query reference the
    output stays within the service check's error and as close as without the split."""
    from pyhip.testing.qsa_reference import fp8_query
    from pyhip.testing.qsa_reference import reference as service_reference

    case = _fp8_case((rows,), (8000,), heads, per_token=per_token)
    value = case.value
    value.indices = torch.from_numpy(_partial_indices(rows, 8000)).to(value.q.device)
    expected = service_reference(
        fp8_query(value.q),
        case.exact.k,
        case.exact.v,
        value.indices,
        query_lens=value.query_lens,
        prefix_lens=value.prefix_lens,
        scale=value.scale,
    )
    norms = expected.norm(dim=-1)
    floor = torch.maximum(norms, norms.median())
    errors = []
    for core_ratios in (None, {}):
        if core_ratios is not None:
            monkeypatch.setattr(_runtime.prepare, "CORE_RATIOS", core_ratios)
        monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
        output = _call_fp8(case)
        plan = next(iter(_runtime._workspaces.values())).union
        split = plan.active.bool() & (plan.residual > 0)
        assert bool(split.any()) == (core_ratios is None)
        if core_ratios is None:
            residual = plan.residual_rows[split[plan.query_tiles.long()]]
            assert bool((residual > 0).any())
            again = _call_fp8(case)
            assert torch.equal(again.view(torch.int16), output.view(torch.int16))
        error = (output.float() - expected).norm(dim=-1) / floor
        errors.append((float(error.mean()), float(error.max())))
    (mean, worst), (whole_mean, _) = errors
    assert mean < 0.05 and worst < 0.2, errors
    assert mean < 1.1 * whole_mean, errors


@SCALES
def test_fp8_core_residual_uniform(per_token, monkeypatch):
    """K = 0 and power-of-two V scales on core/residual tiles: each split row's block_indices
    lists exactly its complete blocks outside the tile's core, and the output is the union
    part's mean V (rounded to BF16, as the union writes it) and the residual part's mean merged
    by their token counts, up to rounding."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    rows, prefix = 1024, 8000
    value = _make_case((rows,), (prefix,), 12, _gpu())
    value.indices = torch.from_numpy(_partial_indices(rows, prefix)).to(value.q.device)
    generator = torch.Generator(device=value.q.device).manual_seed(5)
    shape = value.k.shape[:2] if per_token else (1,)
    exponent = torch.randint(-3, 2, shape, generator=generator, device=value.q.device)
    v8 = torch.randn(value.v.shape, generator=generator, device=value.q.device) * 60
    v8 = v8.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    vs = torch.exp2(exponent.float()).contiguous()
    case = SimpleNamespace(
        value=value,
        k8=torch.zeros(value.k.shape, dtype=FP8, device=value.q.device),
        v8=v8,
        ks=torch.ones(shape, device=value.q.device),
        vs=vs,
    )
    output = _call_fp8(case).float()
    workspace = next(iter(_runtime._workspaces.values()))
    plan = workspace.union
    v = _dequant(v8, vs)[:, 0]
    indices = value.indices.long()
    split_rows = 0
    for row in _sample_rows(value, 96):
        visible = prefix + row + 1
        complete = min(visible // 4, 512)
        blocks = indices[row, : 4 * complete : 4] // 4
        tail = indices[row, 4 * complete : 4 * complete + visible % 4]
        tile = int(plan.query_tiles[row])
        tokens = torch.cat(((blocks[:, None] * 4 + torch.arange(4, device=v.device)).reshape(-1), tail))
        expected = v[tokens].mean(0)
        if bool(plan.active[tile]) and int(plan.residual[tile]) > 0:
            split_rows += 1
            core = plan.blocks[tile, : int(plan.counts[tile])].long()
            outside = ~torch.isin(blocks, core)
            count = int(plan.residual_rows[row])
            listed = workspace.block_indices[row, :count].long()
            assert torch.equal(listed, blocks[outside])
            core_tokens = torch.cat(
                ((blocks[~outside][:, None] * 4 + torch.arange(4, device=v.device)).reshape(-1), tail)
            )
            residual_tokens = (blocks[outside][:, None] * 4 + torch.arange(4, device=v.device)).reshape(-1)
            n_u, n_d = len(core_tokens), len(residual_tokens)
            union_part = v[core_tokens].mean(0).to(torch.bfloat16).float()
            expected = (union_part * n_u + v[residual_tokens].mean(0) * n_d) / (n_u + n_d)
        torch.testing.assert_close(
            output[row], expected.expand(output.shape[1], -1), rtol=0.01, atol=2e-3
        )
    assert split_rows > 0


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


def test_reference_with_fp8_query(monkeypatch):
    """pyhip.testing.qsa_reference with fp8_query's FP32 Q (SGLang's TEST-mode check of
    compute="fp8") equals the FP32 torch reference on the same Q, and the FP8 output stays
    within its FP8-probability error of it: norms floored at the median, mean below 5% and
    worst below 20% also for sharp scores (Q x8), where the exact reference differs far more.
    """
    from pyhip.testing.qsa_reference import fp8_query
    from pyhip.testing.qsa_reference import reference as service_reference

    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    for gain in (1, 8):
        case = _fp8_case((480, 1000, 301), (11392, 3, 4001), 12, per_token=True)
        value = case.value
        value.q = (value.q.float() * gain).to(torch.bfloat16)
        q32 = fp8_query(value.q)
        assert q32.dtype == torch.float32 and q32.shape == value.q.shape
        exact = case.exact
        fp8_reference = service_reference(
            q32,
            exact.k,
            exact.v,
            value.indices,
            query_lens=value.query_lens,
            prefix_lens=value.prefix_lens,
            scale=value.scale,
        )
        rows = _sample_rows(value)
        torch_reference = reference(SimpleNamespace(**{**vars(exact), "q": q32}), rows)
        torch.testing.assert_close(
            fp8_reference[rows], torch_reference, rtol=1e-4, atol=1e-4
        )
        output = _call_fp8(case).float()
        norms = fp8_reference.norm(dim=-1)
        floor = torch.maximum(norms, norms.median())
        error = (output - fp8_reference).norm(dim=-1) / floor
        assert float(error.mean()) < 0.05 and float(error.max()) < 0.2, (
            gain,
            error.mean(),
            error.max(),
        )


@SCALES
def test_fp8_plans_route_with_fp8_terms(per_token, monkeypatch):
    """FP8 workspaces plan with the FP8 kernels' route terms; when every tile goes union the
    FP8 pack is skipped (direct_flag[0] == 0) and the output still matches."""
    monkeypatch.setattr(_runtime, "_workspaces", OrderedDict())
    mode = "token" if per_token else "tensor"
    case = _fp8_case((768, 768), (0, 5), 12, True, per_token=per_token)
    output = _call_fp8(case)
    plan = next(reversed(_runtime._workspaces.values())).union
    terms = _runtime.platform(mode)
    assert plan.union_ratio == terms.union_ratio != _runtime.platform().union_ratio
    assert plan.route_model == (
        terms.union_fixed,
        terms.union_step,
        terms.row_step,
        terms.route_margin,
    )
    assert bool(plan.active.all()) and int(plan.direct_flag[0]) == 0
    rows = _sample_rows(case.value)
    assert _within_fp8_error(output[rows], reference(case.exact, rows))


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
