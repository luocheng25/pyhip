import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import pytest
import torch
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx
from pyhip.contrib.flydsl.moe_gemm_splitk import (
    compile_gemm,
    flydsl_absmax,
    sorted_sum,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
}


def _ptr(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).square().sum().sqrt()
    reference = expected.float().square().sum().sqrt()
    return (error / reference).item()


def test_absmax_reuse_clears_output():
    amax = torch.empty(1, dtype=torch.float32, device="cuda")
    launch = flydsl_absmax()
    for value in (16.0, 0.125):
        source = torch.full((4096,), value, dtype=torch.bfloat16, device="cuda")
        launch(source, amax)
        torch.cuda.synchronize()
        assert amax.item() == value


@pytest.mark.parametrize("hidden_size", [128, 384])
def test_down_prefill_1x4_fp8_ptpc(hidden_size):
    torch.manual_seed(7)
    batch_size = 64
    block_m = 64
    intermediate_size = 384
    num_experts = 1
    topk = 1
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max

    activation = 0.1 * torch.randn(
        batch_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    activation_scale = activation.float().abs().amax(dim=1, keepdim=True) / fp8_max
    activation_fp8 = (
        (activation.float() / activation_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    )

    weight = 0.1 * torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight_scale = weight.float().abs().amax(dim=2) / fp8_max
    weight_fp8 = (
        (weight.float() / weight_scale[..., None])
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    shuffled_weight = shuffle_weight(weight_fp8, layout=(16, 16))

    routing_weight = torch.linspace(
        0.25, 1.0, batch_size, dtype=torch.float32, device="cuda"
    )
    sorted_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.tensor(
        [batch_size, batch_size], dtype=torch.int32, device="cuda"
    )
    output = torch.full(
        (batch_size, hidden_size),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )

    launch = compile_gemm(
        N=hidden_size,
        K=intermediate_size,
        weight_dtype="fp8",
        weight_quant_type="ptpc",
        act_quant_type="ptpc",
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=128,
        stage="down",
        alg="prefill_1x4",
        E=num_experts,
        USE_ATOMIC_WRITE=False,
    )
    launch(
        _ptr(activation_fp8),
        _ptr(shuffled_weight),
        _ptr(output),
        _ptr(sorted_ids),
        _ptr(routing_weight),
        _ptr(sorted_expert_ids),
        _ptr(num_valid_ids),
        _ptr(weight_scale),
        _ptr(activation_scale),
        batch_size,
        1,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    expected = (activation_fp8.float() * activation_scale) @ (
        weight_fp8[0].float() * weight_scale[0, :, None]
    ).T
    expected = (expected * routing_weight[:, None]).to(torch.bfloat16)

    assert torch.isfinite(output).all()
    block_errors = [
        _relative_l2(output[:, start : start + 64], expected[:, start : start + 64])
        for start in range(0, hidden_size, 64)
    ]
    assert _relative_l2(output, expected) < 3e-2, block_errors


@pytest.mark.parametrize(
    "padding_bytes",
    [0, 32, 64, 128],
)
def test_down_prefill_physical_cshuffle_sorted_sum(padding_bytes):
    torch.manual_seed(11)
    batch_size = 64
    block_m = 64
    intermediate_size = 384
    hidden_size = 512
    topk = 1
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max

    activation = 0.1 * torch.randn(
        batch_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    activation_scale = activation.float().abs().amax(dim=1, keepdim=True) / fp8_max
    activation_fp8 = (
        (activation.float() / activation_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    )

    weight = 0.1 * torch.randn(
        1, hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    weight_scale = weight.float().abs().amax(dim=2) / fp8_max
    weight_fp8 = (
        (weight.float() / weight_scale[..., None])
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    shuffled_weight = shuffle_weight(weight_fp8, layout=(16, 16))

    routing_weight = torch.linspace(
        0.25, 1.0, batch_size, dtype=torch.float32, device="cuda"
    )
    sorted_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.tensor(
        [batch_size, batch_size], dtype=torch.int32, device="cuda"
    )
    output_row_size = hidden_size + (
        padding_bytes // 2 if padding_bytes is not None else 0
    )
    physical_output = torch.full(
        (batch_size, output_row_size),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )

    launch = compile_gemm(
        N=hidden_size,
        K=intermediate_size,
        weight_dtype="fp8",
        weight_quant_type="ptpc",
        act_quant_type="ptpc",
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=256,
        stage="down",
        alg="prefill_1x4",
        E=1,
        USE_ATOMIC_WRITE=False,
        down_physical_n128=True,
        down_output_padding_bytes=padding_bytes,
    )
    launch(
        _ptr(activation_fp8),
        _ptr(shuffled_weight),
        _ptr(physical_output),
        _ptr(sorted_ids),
        _ptr(routing_weight),
        _ptr(sorted_expert_ids),
        _ptr(num_valid_ids),
        _ptr(weight_scale),
        _ptr(activation_scale),
        batch_size,
        1,
        torch.cuda.current_stream(),
    )

    logical_output = torch.empty(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    loc_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, topk
    )
    sorted_sum(topk, hidden_size, padding_bytes)(
        loc_ids, physical_output, logical_output, batch_size
    )
    torch.cuda.synchronize()

    expected = (activation_fp8.float() * activation_scale) @ (
        weight_fp8[0].float() * weight_scale[0, :, None]
    ).T
    expected = (expected * routing_weight[:, None]).to(torch.bfloat16)

    assert torch.isfinite(physical_output[:, :hidden_size]).all()
    assert torch.isfinite(logical_output).all()
    assert _relative_l2(logical_output, expected) < 3e-2


def test_down_prefill_physical_per_tensor_k192_cshuffle():
    torch.manual_seed(17)
    batch_size = 64
    intermediate_size = 192
    hidden_size = 512
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max

    activation = 0.1 * torch.randn(
        batch_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    activation_scale = activation.float().abs().amax() / fp8_max
    activation_fp8 = (
        (activation.float() / activation_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    )
    weight = 0.1 * torch.randn(
        1, hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    weight_scale = weight.float().abs().amax(dim=(1, 2)) / fp8_max
    weight_fp8 = (
        (weight.float() / weight_scale[:, None, None])
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    shuffled_weight = shuffle_weight(weight_fp8, layout=(16, 16))

    routing_weight = torch.linspace(
        0.25, 1.0, batch_size, dtype=torch.float32, device="cuda"
    )
    sorted_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.tensor(
        [batch_size, batch_size], dtype=torch.int32, device="cuda"
    )
    physical_output = torch.full(
        (batch_size, hidden_size),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )

    launch = compile_gemm(
        N=hidden_size,
        K=intermediate_size,
        weight_dtype="fp8",
        weight_quant_type="per_tensor",
        act_quant_type="per_tensor",
        TOPK=1,
        BLOCK_TILE_SIZE_M=64,
        BLOCK_TILE_SIZE_N=256,
        stage="down",
        alg="prefill_1x4",
        E=1,
        USE_ATOMIC_WRITE=False,
        down_physical_n128=True,
        down_output_padding_bytes=0,
    )
    launch(
        _ptr(activation_fp8),
        _ptr(shuffled_weight),
        _ptr(physical_output),
        _ptr(sorted_ids),
        _ptr(routing_weight),
        _ptr(sorted_expert_ids),
        _ptr(num_valid_ids),
        _ptr(weight_scale),
        _ptr(activation_scale.reshape(1)),
        batch_size,
        1,
        torch.cuda.current_stream(),
    )

    logical_output = torch.empty(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    loc_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda").view(
        batch_size, 1
    )
    sorted_sum(1, hidden_size, 0)(loc_ids, physical_output, logical_output, batch_size)
    torch.cuda.synchronize()

    expected = (activation_fp8.float() * activation_scale) @ (
        weight_fp8[0].float() * weight_scale[0]
    ).T
    expected = (expected * routing_weight[:, None]).to(torch.bfloat16)
    assert torch.isfinite(logical_output).all()
    assert _relative_l2(logical_output, expected) < 3e-2
