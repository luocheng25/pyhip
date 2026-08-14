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


def _pack_bn256_physical(logical_rows):
    row_count, hidden_size = logical_rows.shape
    assert row_count % 64 == 0
    assert hidden_size % 256 == 0

    token_loc = torch.arange(row_count, dtype=torch.int64, device=logical_rows.device)[
        :, None
    ]
    column = torch.arange(
        0, hidden_size, 8, dtype=torch.int64, device=logical_rows.device
    )[None, :]
    block_row = token_loc % 64
    column_in_block = column % 256
    wave_id = column_in_block // 64
    lane_group = (column_in_block % 64) // 16
    channel_piece = (column_in_block % 16) // 8
    store_index = (block_row // 16) * 2 + channel_piece
    physical_lane = (block_row % 16) * 4 + lane_group
    physical_offset = (
        (token_loc // 64) * (64 * hidden_size)
        + (column // 256) * (64 * 256)
        + wave_id * (8 * 64 * 8)
        + store_index * (64 * 8)
        + physical_lane * 8
    )

    element_index = physical_offset[..., None] + torch.arange(
        8, dtype=torch.int64, device=logical_rows.device
    )
    physical_rows = torch.empty_like(logical_rows)
    physical_rows.view(-1)[element_index.reshape(-1)] = logical_rows.reshape(-1)
    return physical_rows


def _pack_bn256_tile_major(logical_rows, row_group):
    row_count, hidden_size = logical_rows.shape
    assert row_count % 64 == 0
    assert hidden_size % 256 == 0
    assert row_group in (1, 2, 4, 8, 16)

    token_loc = torch.arange(row_count, dtype=torch.int64, device=logical_rows.device)[:, None]
    column = torch.arange(0, hidden_size, 8, dtype=torch.int64, device=logical_rows.device)[None, :]
    block_row = token_loc % 64
    row_in_tile = block_row % 16
    column_in_block = column % 256
    lane_group = (column_in_block % 64) // 16
    channel_piece = (column_in_block % 16) // 8
    physical_lane = (
        ((row_in_tile // row_group) * 4 + column_in_block // 64)
        * (2 * row_group * 4)
        + channel_piece * (row_group * 4)
        + (row_in_tile % row_group) * 4
        + lane_group
    )
    physical_offset = (
        (token_loc // 64) * (64 * hidden_size)
        + (column // 256) * (64 * 256)
        + (block_row // 16) * (16 * 256)
        + physical_lane * 8
    )
    element_index = physical_offset[..., None] + torch.arange(
        8, dtype=torch.int64, device=logical_rows.device
    )
    physical_rows = torch.empty_like(logical_rows)
    physical_rows.view(-1)[element_index.reshape(-1)] = logical_rows.reshape(-1)
    return physical_rows


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
    "tile_n,hidden_size,padding_bytes,row_group,cshuffle_output",
    [
        (128, 512, None, None, False),
        (128, 1024, None, None, False),
        (192, 1536, None, None, False),
        (256, 512, None, None, False),
        (256, 512, 0, None, False),
        (256, 512, 32, None, False),
        (256, 512, 64, None, False),
        (256, 512, 128, None, False),
        (256, 512, 128, None, True),
        (256, 512, None, 1, False),
        (256, 512, None, 2, False),
        (256, 512, None, 4, False),
        (256, 512, None, 8, False),
        (256, 512, None, 16, False),
    ],
)
def test_down_prefill_physical_sorted_sum(
    tile_n, hidden_size, padding_bytes, row_group, cshuffle_output
):
    torch.manual_seed(11)
    batch_size = 64
    block_m = 64
    intermediate_size = 384
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
        BLOCK_TILE_SIZE_N=tile_n,
        stage="down",
        alg="prefill_1x4",
        E=1,
        USE_ATOMIC_WRITE=False,
        down_physical_n128=True,
        down_output_padding_bytes=padding_bytes,
        down_output_row_group=row_group,
        down_cshuffle_output=cshuffle_output,
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
    sorted_sum(topk, hidden_size, True, tile_n, padding_bytes, row_group)(
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
        (activation.float() / activation_scale)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
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
        (batch_size, hidden_size + 64),
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
        down_output_padding_bytes=128,
        down_cshuffle_output=True,
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
    sorted_sum(1, hidden_size, True, 256, 128)(
        loc_ids, physical_output, logical_output, batch_size
    )
    torch.cuda.synchronize()

    expected = (activation_fp8.float() * activation_scale) @ (
        weight_fp8[0].float() * weight_scale[0]
    ).T
    expected = (expected * routing_weight[:, None]).to(torch.bfloat16)
    assert torch.isfinite(logical_output).all()
    assert _relative_l2(logical_output, expected) < 3e-2


def test_sorted_sum_bn256_topk4_prefetch():
    torch.manual_seed(19)
    batch_size = 64
    topk = 4
    hidden_size = 1024
    logical_rows = torch.randn(
        batch_size * topk,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    physical_rows = _pack_bn256_physical(logical_rows)
    loc_ids = torch.randperm(batch_size * topk, dtype=torch.int32, device="cuda").view(
        batch_size, topk
    )
    output = torch.empty(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")

    sorted_sum(topk, hidden_size, True, 256)(loc_ids, physical_rows, output, batch_size)
    torch.cuda.synchronize()

    expected = logical_rows[loc_ids.long()].float().sum(dim=1).to(torch.bfloat16)
    assert torch.isfinite(output).all()
    assert _relative_l2(output, expected) < 1e-3


@pytest.mark.parametrize("row_group", [1, 2, 4, 8, 16])
def test_sorted_sum_bn256_tile_major_row_group(row_group):
    torch.manual_seed(23)
    batch_size = 64
    topk = 4
    hidden_size = 1024
    logical_rows = torch.randn(
        batch_size * topk,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    physical_rows = _pack_bn256_tile_major(logical_rows, row_group)
    loc_ids = torch.randperm(
        batch_size * topk, dtype=torch.int32, device="cuda"
    ).view(batch_size, topk)
    output = torch.empty(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    sorted_sum(topk, hidden_size, True, 256, None, row_group)(
        loc_ids, physical_rows, output, batch_size
    )
    torch.cuda.synchronize()

    expected = logical_rows[loc_ids.long()].float().sum(dim=1).to(torch.bfloat16)
    assert torch.isfinite(output).all()
    assert _relative_l2(output, expected) < 1e-3
