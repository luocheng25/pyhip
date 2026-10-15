"""SGLang served weight format: AITER 16x16 shuffle, E4M3FNUZ normalization, and MoE padding.

The real-weight comparison with a running SGLang server is the ``served_format``
section of benchmarks/monokernel/check_reference.py.
"""

import dataclasses

import pytest
import torch

from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard
from pyhip.ops.monokernel.qwen38.weights import (
    KERNEL_FP8,
    SGLANG_NAMES,
    moe_inter_padded,
    sglang_format,
    shard_layer_weights,
    shuffle_16x16,
    synthetic_tensors,
    tensor_getter,
    to_fnuz,
    unshuffle_16x16,
)

if not torch.cuda.is_available():
    pytest.skip("weight format tests run on a GPU", allow_module_level=True)

CONFIG = dataclasses.replace(QWEN38_FLASH_NEXT, n_experts=16, top_k=4)
DEVICE = torch.device("cuda")


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fnuz, torch.bfloat16))
def test_shuffle_matches_aiter(dtype: torch.dtype) -> None:
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    bits = torch.randint(-128, 128, (3, 64, 96 * (2 if dtype.itemsize == 1 else 1)), device=DEVICE)
    value = bits.to(torch.int8).view(dtype) if dtype.itemsize == 1 else bits.to(dtype)
    expected = shuffle.shuffle_weight(value, (16, 16))
    got = shuffle_16x16(value)
    assert torch.equal(got.view(torch.uint8 if dtype.itemsize == 1 else torch.int16),
                       expected.view(torch.uint8 if dtype.itemsize == 1 else torch.int16))
    assert torch.equal(unshuffle_16x16(got).view(torch.uint8 if dtype.itemsize == 1 else torch.int16),
                       value.view(torch.uint8 if dtype.itemsize == 1 else torch.int16))


def test_fnuz_keeps_dequantized_values() -> None:
    bits = torch.arange(-128, 128, device=DEVICE, dtype=torch.int16).to(torch.int8)
    weight = bits.view(torch.float8_e4m3fn)
    scale = torch.full((256,), 0.37, device=DEVICE)
    fnuz, fnuz_scale = to_fnuz(weight, scale)
    finite = ~weight.float().isnan()
    torch.testing.assert_close((fnuz.float() * fnuz_scale)[finite], (weight.float() * scale)[finite], rtol=0, atol=0)
    assert fnuz.view(torch.int8)[0] == 0  # E4M3FN -0 (0x80) would be NaN in E4M3FNUZ


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("kind", (LayerKind.LINEAR, LayerKind.FULL))
def test_served_format(kind: LayerKind, tp: int) -> None:
    """Every SGLang tensor is produced; dense FP8 and experts dequantize to the golden values."""

    get = tensor_getter(synthetic_tensors(CONFIG, kind, device=DEVICE))
    rank_shard = shard(tp, tp - 1, CONFIG)
    t = shard_layer_weights(get, rank_shard, kind)
    served = sglang_format(t, rank_shard, kind)
    assert set(served) == set(SGLANG_NAMES[kind])
    for name in ("gdn_qkvz", "gdn_out") if kind is LayerKind.LINEAR else ("attn_qkv", "attn_o"):
        assert served[name].dtype == KERNEL_FP8
        dequant = unshuffle_16x16(served[name]).float() * served[f"{name}_scale"]
        torch.testing.assert_close(dequant, t[name].float() * t[f"{name}_scale"][:, None], rtol=0, atol=0)
    inter, padded = CONFIG.inter // tp, moe_inter_padded(rank_shard)
    experts = CONFIG.n_experts + 1
    w13 = unshuffle_16x16(served["moe_w13"]).float() * served["moe_w13_scale"]
    w2 = unshuffle_16x16(served["moe_w2"]).float() * served["moe_w2_scale"]
    assert w13.shape == (experts, 2 * padded, CONFIG.hidden) and w2.shape == (experts, CONFIG.hidden, padded)
    up = t["moe_up"].float() * t["moe_up_scale"][..., None]
    for half in range(2):
        rows = slice(half * padded, half * padded + inter)
        torch.testing.assert_close(w13[:, rows], up[:, half * inter : (half + 1) * inter], rtol=0, atol=0)
        assert not w13[:, half * padded + inter : (half + 1) * padded].any()
    torch.testing.assert_close(w2[..., :inter], t["moe_down"].float() * t["moe_down_scale"][..., None], rtol=0, atol=0)
    assert not w2[..., inter:].any()
