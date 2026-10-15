"""MonoKernel stage kernels against the goldens: per-token FP8 quantization and the W8A8 linear.

Weights come from ``sglang_format`` (bit-identical to SGLang's served tensors), activations are
random. FP32 sums run in another order than Torch, so linear outputs may differ by one BF16 ulp.
"""

import dataclasses

import pytest
import torch

from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard
from pyhip.ops.monokernel.qwen38.reference import (
    ACTIVATION_FP8,
    bf,
    fp8_linear,
    golden_gdn_core,
    golden_gdn_out,
    golden_gdn_proj,
    golden_hc_lowrank,
    golden_hc_norm,
    quantize_rows,
    resolve_hc_mix_ties,
)
from pyhip.ops.monokernel.qwen38.weights import (
    pack_layer,
    sglang_format,
    shard_layer_weights,
    synthetic_tensors,
    tensor_getter,
)

if not torch.cuda.is_available():
    pytest.skip("MonoKernel stages run on a GPU", allow_module_level=True)

from pyhip.ops.monokernel.qwen38.flydsl.gdn import gdn_recurrent, qkvz_conv  # noqa: E402
from pyhip.ops.monokernel.qwen38.flydsl.hc import HCWorkspace, hc_combine_norm, hc_mix  # noqa: E402
from pyhip.ops.monokernel.qwen38.flydsl.linear import linear_bf16, linear_fp8, quant_rows  # noqa: E402

CONFIG = dataclasses.replace(QWEN38_FLASH_NEXT, n_experts=16, top_k=4)
DEVICE = torch.device("cuda")
ROWS = (1, 5, 16, 33, 64)


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.randn(*shape, generator=gen, device=DEVICE) * scale).to(torch.bfloat16)


def assert_bf16_ulp(got: torch.Tensor, expected: torch.Tensor, exact: float = 0.98, ulp1: float = 0.999) -> None:
    """At least ``ulp1`` of the elements within one BF16 ulp and ``exact`` bit-identical; relative L2
    within 1e-3 (results that cancel to near zero can sit several ulps apart)."""

    got, expected = got.float(), expected.float()
    magnitude = torch.maximum(got.abs(), expected.abs()).clamp_min(torch.finfo(torch.bfloat16).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
    delta = (got - expected).abs()
    assert float((delta <= ulp).float().mean()) >= ulp1
    assert float((delta == 0).float().mean()) >= exact
    assert float(delta.norm() / expected.norm()) <= 1e-3


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("k", (2560, 3072))
def test_quant_rows(rows: int, k: int) -> None:
    x = _randn(rows, k, seed=rows, scale=3.0)
    x[rows // 2] = 0
    xq = torch.empty(rows, k, dtype=ACTIVATION_FP8, device=DEVICE)
    xs = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    quant_rows(x, xq, xs)
    values, scale = quantize_rows(x)
    assert torch.equal(xs, scale[:, 0])
    assert torch.equal(xq.float(), values)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", ("gdn_qkvz", "gdn_out"))
def test_linear_fp8(rows: int, name: str) -> None:
    kind = LayerKind.LINEAR
    rank_shard = shard(2, 1, CONFIG)
    t = shard_layer_weights(tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE)), rank_shard, kind)
    served = sglang_format(t, rank_shard, kind)
    weight, weight_scale = served[name], served[f"{name}_scale"]
    x = _randn(rows, weight.shape[1], seed=7 + rows)
    xq = torch.empty(rows, weight.shape[1], dtype=ACTIVATION_FP8, device=DEVICE)
    xs = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    quant_rows(x, xq, xs)
    out = torch.empty(rows, weight.shape[0], dtype=torch.bfloat16, device=DEVICE)
    linear_fp8(xq, xs, weight, weight_scale, out)
    assert_bf16_ulp(out, fp8_linear(x, t[name], t[f"{name}_scale"]))


@pytest.fixture(scope="module")
def linear_layer():
    kind = LayerKind.LINEAR
    rank_shard = shard(2, 0, CONFIG)
    t = shard_layer_weights(tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE)), rank_shard, kind)
    served = sglang_format(t, rank_shard, kind)
    return t, served, pack_layer(served, kind, CONFIG)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("prefix", ("hc_attn", "hc_mlp"))
def test_hc_mix(rows: int, prefix: str, linear_layer) -> None:
    """Low-rank activations, combine gains and the mixed input; BF16 low-rank values within a few FP32
    ulps of a tie may round the other way (resolved like the capture check)."""

    t, _, packed = linear_layer
    hn = _randn(rows, CONFIG.hc_width, seed=100 + rows)
    x = torch.empty(rows, CONFIG.hidden, dtype=torch.bfloat16, device=DEVICE)
    work = HCWorkspace(rows, DEVICE)
    hc_mix(hn, packed[f"{prefix}_dn"], packed[f"{prefix}_up"], x, work)
    act = golden_hc_lowrank(hn, t[f"{prefix}_down"], CONFIG)
    lowrank_ok = (work.lowrank.float() == bf(act)).float().mean()
    assert float(lowrank_ok) >= 0.999
    gain = 2.0 * torch.sigmoid((hn.float() @ t[f"{prefix}_inject"].float().T) / CONFIG.hc_count)
    torch.testing.assert_close(work.gain, gain, rtol=1e-5, atol=1e-6)
    mixed, flips = resolve_hc_mix_ties(hn, t[f"{prefix}_down"], t[f"{prefix}_up"], CONFIG, x)
    assert len(flips) <= rows
    assert_bf16_ulp(x, mixed, exact=0.99)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("combine", (True, False))
def test_hc_combine_norm(rows: int, combine: bool, linear_layer) -> None:
    t, _, _ = linear_layer
    h = _randn(rows, CONFIG.hc_width, seed=200 + rows, scale=4.0)
    y = _randn(rows, CONFIG.hidden, seed=300 + rows)
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    gain = 2.0 * torch.rand(rows, CONFIG.hc_count, generator=gen, device=DEVICE)
    out = torch.empty_like(h)
    normed = torch.empty_like(h)
    hc_combine_norm(h, y if combine else None, gain, t["hc_mlp_norm"], out, normed)
    expected = h
    if combine:
        view = h.float().view(rows, CONFIG.hc_count, CONFIG.hidden) + y.float()[:, None, :] * gain[:, :, None]
        expected = bf(view.view(rows, -1)).to(torch.bfloat16)
        assert torch.equal(out, expected)
    assert_bf16_ulp(normed, golden_hc_norm(expected, t["hc_mlp_norm"], CONFIG), exact=0.995)


@pytest.mark.parametrize("rows", ROWS)
def test_gdn(rows: int, linear_layer) -> None:
    """qkvz projection + conv (with the conv state), ba, delta rule (with the SSM state) and gated norm,
    each checked from the kernels' own inputs."""

    t, served, packed = linear_layer
    s = shard(2, 0, CONFIG)
    slots_total = rows + 3
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    slots = torch.randperm(slots_total, generator=gen, device=DEVICE)[:rows].to(torch.int32)
    if rows > 1:
        slots[rows // 2] = -1
    conv_state = (0.5 * torch.randn(slots_total, s.conv_dim, 3, generator=gen, device=DEVICE)).to(torch.bfloat16)
    ssm_state = 0.05 * torch.randn(slots_total, s.gdn_v_heads, CONFIG.gdn_head_dim, CONFIG.gdn_head_dim,
                                   generator=gen, device=DEVICE)
    conv0, ssm0 = conv_state.clone(), ssm_state.clone()
    x = _randn(rows, CONFIG.hidden, seed=400 + rows)
    xq = torch.empty(rows, CONFIG.hidden, dtype=ACTIVATION_FP8, device=DEVICE)
    xs = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    quant_rows(x, xq, xs)
    n = served["gdn_qkvz"].shape[0]
    qkvz = torch.empty(rows, n, dtype=torch.bfloat16, device=DEVICE)
    conv_out = torch.empty(rows, s.conv_dim, dtype=torch.bfloat16, device=DEVICE)
    qkvz_conv(xq, xs, served["gdn_qkvz"], served["gdn_qkvz_scale"], served["gdn_conv"], slots, conv_state, qkvz,
              conv_out)
    ba = torch.empty(rows, 2 * s.gdn_v_heads, dtype=torch.bfloat16, device=DEVICE)
    linear_bf16(x, packed["gdn_ba"], ba)
    qkvz_ref, ba_ref = golden_gdn_proj(x, t)
    assert_bf16_ulp(qkvz, qkvz_ref)
    assert_bf16_ulp(ba, ba_ref, exact=0.99 if rows > 1 else 0.97)

    conv_ref, core_ref = golden_gdn_core(qkvz[:, : s.conv_dim], ba, t, s, state_indices=slots.cpu(),
                                         conv_state=conv0, ssm_state=ssm0)
    assert torch.equal(conv_state, conv0)
    assert_bf16_ulp(conv_out, conv_ref, exact=0.995)
    core = torch.empty(rows, s.gdn_v_heads, CONFIG.gdn_head_dim, dtype=torch.bfloat16, device=DEVICE)
    y = torch.empty(rows, s.gdn_value_dim, dtype=torch.bfloat16, device=DEVICE)
    gdn_recurrent(conv_ref.to(torch.bfloat16), qkvz, ba, served["gdn_a_log"], served["gdn_dt_bias"], served["gdn_norm"],
                  ssm_state, slots, core, y, s.gdn_v_heads, s.gdn_k_heads)
    torch.testing.assert_close(ssm_state, ssm0, rtol=1e-5, atol=1e-6)
    assert_bf16_ulp(core, core_ref, exact=0.99)
    normed_ref, _ = golden_gdn_out(core, qkvz[:, s.conv_dim:], t, s)
    assert_bf16_ulp(y, normed_ref.reshape(rows, -1), exact=0.99)
