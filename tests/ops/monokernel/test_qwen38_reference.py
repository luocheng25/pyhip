"""Qwen3.8-Flash-Next MonoKernel goldens: TP sharding invariants and routing ties.

Runs on one GPU with a 16-expert copy of the model geometry. The real-weight
check against SGLang decode captures is benchmarks/monokernel/check_reference.py.
"""

import dataclasses

import pytest
import torch

from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard
from pyhip.ops.monokernel.qwen38.reference import (
    golden_attention_core,
    golden_attention_proj,
    golden_gdn_core,
    golden_gdn_proj,
    golden_hc_lowrank,
    golden_hc_mix,
    golden_layer_tp,
    golden_route,
    hc_mix_from_lowrank,
    resolve_hc_mix_ties,
)
from pyhip.ops.monokernel.qwen38.weights import (
    shard_gdn_state,
    shard_kv_heads,
    shard_layer_weights,
    synthetic_tensors,
    tensor_getter,
)

if not torch.cuda.is_available():
    pytest.skip("Qwen3.8 goldens run on a GPU", allow_module_level=True)

CONFIG = dataclasses.replace(QWEN38_FLASH_NEXT, n_experts=16, top_k=4)
DEVICE = torch.device("cuda")
ROWS = 3
# One BF16 ulp: SGLang-style projections of different widths may sum in another order.
BF16_ULP = dict(rtol=2**-7, atol=1e-6)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _rel_l2(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got.float() - expected.float()).norm() / expected.float().norm())


def _columns(tensor: torch.Tensor, blocks: tuple[int, ...], tp: int, rank: int) -> torch.Tensor:
    """This rank's contiguous share of every block of a TP1 column-parallel output."""

    parts, start = [], 0
    for size in blocks:
        part = size // tp
        parts.append(tensor[..., start + rank * part : start + (rank + 1) * part])
        start += size
    return torch.cat(parts, dim=-1)


def _gdn_states(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    full = shard(1, 0, CONFIG)
    gen = _generator(seed)
    conv = torch.randn(ROWS + 1, full.conv_dim, CONFIG.conv_kernel - 1, generator=gen, device=DEVICE)
    ssm = torch.randn(
        ROWS + 1, CONFIG.gdn_v_heads, CONFIG.gdn_head_dim, CONFIG.gdn_head_dim, generator=gen, device=DEVICE
    )
    return (0.5 * conv).to(torch.bfloat16), 0.05 * ssm


def _kv_inputs(seed: int) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    gen = _generator(seed)
    positions = torch.tensor([37, 64, 5], device=DEVICE)
    shape = (80, CONFIG.kv_heads, CONFIG.head_dim)
    k_cache = [torch.randn(shape, generator=gen, device=DEVICE).to(torch.bfloat16) for _ in range(ROWS)]
    v_cache = [torch.randn(shape, generator=gen, device=DEVICE).to(torch.bfloat16) for _ in range(ROWS)]
    selected = torch.full((ROWS, 40), -1, dtype=torch.int32, device=DEVICE)
    for row in range(ROWS):
        visible = int(positions[row]) + 1
        count = min(visible, selected.shape[1])
        order = torch.randperm(visible, generator=torch.Generator().manual_seed(seed + row))
        selected[row, :count] = order[:count].to(DEVICE, torch.int32)
    return positions, k_cache, v_cache, selected


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_gdn_shards_match_tp1(tp: int) -> None:
    """Per-rank projections, conv, recurrence, and states are TP1 slices; negative slots stay idle."""

    kind = LayerKind.LINEAR
    get = tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE))
    full = shard(1, 0, CONFIG)
    x = torch.randn(ROWS, CONFIG.hidden, generator=_generator(1), device=DEVICE).to(torch.bfloat16)
    qkvz, ba = golden_gdn_proj(x, shard_layer_weights(get, full, kind, experts=()))
    conv, ssm = _gdn_states(2)
    state_indices = torch.tensor([2, -1, 0])
    conv1, ssm1 = conv.clone(), ssm.clone()
    t1 = shard_layer_weights(get, full, kind, experts=())
    _, core1 = golden_gdn_core(
        qkvz[:, : full.conv_dim], ba, t1, full, state_indices=state_indices, conv_state=conv1, ssm_state=ssm1
    )
    assert torch.count_nonzero(core1[1]) == 0
    assert torch.equal(conv1[[1, 3]], conv[[1, 3]]) and torch.equal(ssm1[[1, 3]], ssm[[1, 3]])
    key_dim, value_dim, v_heads = full.gdn_key_dim, full.gdn_value_dim, CONFIG.gdn_v_heads
    cores = []
    for rank in range(tp):
        rank_shard = shard(tp, rank, CONFIG)
        t = shard_layer_weights(get, rank_shard, kind, experts=())
        rank_qkvz, rank_ba = golden_gdn_proj(x, t)
        expected_qkvz = _columns(qkvz, (key_dim, key_dim, value_dim, value_dim), tp, rank)
        expected_ba = _columns(ba, (v_heads, v_heads), tp, rank)
        torch.testing.assert_close(rank_qkvz, expected_qkvz, **BF16_ULP)
        torch.testing.assert_close(rank_ba, expected_ba, **BF16_ULP)
        rank_conv, rank_ssm = shard_gdn_state(conv, ssm, rank_shard)
        _, core = golden_gdn_core(
            expected_qkvz[:, : rank_shard.conv_dim],
            expected_ba,
            t,
            rank_shard,
            state_indices=state_indices,
            conv_state=rank_conv,
            ssm_state=rank_ssm,
        )
        cores.append(core)
        expected_conv, expected_ssm = shard_gdn_state(conv1, ssm1, rank_shard)
        assert torch.equal(rank_conv, expected_conv)
        torch.testing.assert_close(rank_ssm, expected_ssm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.cat(cores, dim=1), core1, **BF16_ULP)


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_attention_shards_match_tp1(tp: int) -> None:
    """Per-rank Q/K/V/gate and sparse attention are TP1 head slices, with KV heads replicated past TP2."""

    kind = LayerKind.FULL
    get = tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE))
    full = shard(1, 0, CONFIG)
    x = torch.randn(ROWS, CONFIG.hidden, generator=_generator(3), device=DEVICE).to(torch.bfloat16)
    positions, k_cache, v_cache, selected = _kv_inputs(4)
    proj1 = golden_attention_proj(x, shard_layer_weights(get, full, kind, experts=()), full, positions)
    core1 = golden_attention_core(
        proj1["q"],
        proj1["k"],
        proj1["v"],
        full,
        positions=positions,
        k_cache=[cache.clone() for cache in k_cache],
        v_cache=[cache.clone() for cache in v_cache],
        selected=selected,
    )
    cores = []
    for rank in range(tp):
        rank_shard = shard(tp, rank, CONFIG)
        heads = slice(rank * rank_shard.heads, (rank + 1) * rank_shard.heads)
        kv = slice(rank_shard.kv_head_start, rank_shard.kv_head_start + rank_shard.kv_heads)
        proj = golden_attention_proj(x, shard_layer_weights(get, rank_shard, kind, experts=()), rank_shard, positions)
        for name, part in (("q", heads), ("gate", heads), ("k", kv), ("v", kv)):
            torch.testing.assert_close(proj[name], proj1[name][:, part], **BF16_ULP)
        cores.append(
            golden_attention_core(
                proj1["q"][:, heads],
                proj1["k"][:, kv],
                proj1["v"][:, kv],
                rank_shard,
                positions=positions,
                k_cache=[shard_kv_heads(cache, rank_shard) for cache in k_cache],
                v_cache=[shard_kv_heads(cache, rank_shard) for cache in v_cache],
                selected=selected,
            )
        )
    torch.testing.assert_close(torch.cat(cores, dim=1), core1, **BF16_ULP)


@pytest.mark.parametrize("kind", (LayerKind.LINEAR, LayerKind.FULL))
@pytest.mark.parametrize("tp", (2, 4, 8))
def test_layer_tp_tracks_tp1(kind: LayerKind, tp: int) -> None:
    """A whole layer at TP2/4/8 stays near TP1; per-rank FP8 requantization keeps them from matching."""

    get = tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE))
    h = torch.randn(ROWS, CONFIG.hc_width, generator=_generator(5), device=DEVICE).to(torch.bfloat16)
    if kind is LayerKind.LINEAR:
        conv, ssm = _gdn_states(6)
        state_indices = torch.tensor([2, -1, 0])

        def mixer_inputs(rank_shard):
            rank_conv, rank_ssm = shard_gdn_state(conv, ssm, rank_shard)
            return dict(state_indices=state_indices, conv_state=rank_conv, ssm_state=rank_ssm)

    else:
        positions, k_cache, v_cache, selected = _kv_inputs(7)

        def mixer_inputs(rank_shard):
            return dict(
                positions=positions,
                k_cache=[shard_kv_heads(cache, rank_shard) for cache in k_cache],
                v_cache=[shard_kv_heads(cache, rank_shard) for cache in v_cache],
                selected=selected,
            )

    def run(tp_size):
        shards = [shard(tp_size, rank, CONFIG) for rank in range(tp_size)]
        weights = [shard_layer_weights(get, rank_shard, kind) for rank_shard in shards]
        return golden_layer_tp(kind, h, weights, shards, [mixer_inputs(s) for s in shards])

    expected, result = run(1), run(tp)
    assert torch.equal(result["ids"].sort(-1).values, expected["ids"].sort(-1).values)
    assert _rel_l2(result["attn"], expected["attn"]) < 6e-2
    assert _rel_l2(result["mlp"], expected["mlp"]) < 1e-1
    assert _rel_l2(result["out"], expected["out"]) < 2e-2


def test_route_ties_prefer_lower_expert() -> None:
    config = dataclasses.replace(CONFIG, n_experts=8, top_k=2)
    logits = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5], device=DEVICE)
    x = torch.zeros(1, config.hidden, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0] = 1.0
    router = torch.zeros(config.n_experts, config.hidden, device=DEVICE, dtype=torch.bfloat16)
    router[:, 0] = logits.to(torch.bfloat16)
    t = {"router": router, "shared_gate": torch.zeros(config.hidden, device=DEVICE, dtype=torch.bfloat16)}
    _, ids, weights = golden_route(x, t, shard(1, 0, config))
    assert ids.tolist() == [[1, 2, config.shared_expert]]
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 0.5]], device=DEVICE))


def test_hc_mix_resolves_near_tie_rounding() -> None:
    """A target rounding the activation nearest a BF16 tie the other way is reproduced by one flip."""

    gen = _generator(8)
    hn = torch.randn(ROWS, CONFIG.hc_width, generator=gen, device=DEVICE).to(torch.bfloat16)
    down = (0.02 * torch.randn(CONFIG.hc_lowrank, CONFIG.hc_width, generator=gen, device=DEVICE)).to(torch.bfloat16)
    up = (0.05 * torch.randn(CONFIG.hc_width, CONFIG.hc_lowrank, generator=gen, device=DEVICE)).to(torch.bfloat16)
    act = golden_hc_lowrank(hn, down, CONFIG)
    lowrank = act.to(torch.bfloat16)
    step = torch.where(act.abs() > lowrank.float().abs(), 1, -1).to(torch.int16)
    other = lowrank.view(torch.int16) + step
    tie = (lowrank.float() + other.view(torch.bfloat16).float()) / 2
    row = 1
    index = int((act[row] - tie[row]).abs().div(act[row].abs()).argmin())
    flipped = lowrank.view(torch.int16).clone()
    flipped[row, index] = other[row, index]
    target = hc_mix_from_lowrank(hn, flipped.view(torch.bfloat16), up, CONFIG)
    golden = golden_hc_mix(hn, down, up, CONFIG)
    assert not torch.equal(target, golden)

    out, flips = resolve_hc_mix_ties(hn, down, up, CONFIG, target, window_ulps=float("inf"))
    assert torch.equal(out, target)
    assert [(r, i) for r, i, _ in flips] == [(row, index)]
    out, flips = resolve_hc_mix_ties(hn, down, up, CONFIG, target, window_ulps=0.0)
    assert torch.equal(out, golden) and flips == []
