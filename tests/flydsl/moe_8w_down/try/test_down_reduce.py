# SPDX-License-Identifier: MIT
"""Packed down+sum: independent reducer and complete-pipeline correctness."""

import pytest
import torch

from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_pipeline import compile_packed_down_reduce
from moe_multistage_reduce import make_moe_sum
from pyhip.contrib.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids
from test_8stage import make_case


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")


@pytest.mark.parametrize("block_n", [32, 64])
def test_legacy_down_topk_cancellation(block_n):
    """Separate cross-K mul/add hits a BF16 tie and loses a TOPK residual."""
    require_gpu()
    from aiter.ops.shuffle import shuffle_weight
    from moe_8wave_down import flydsl_moe_gemm_8wave_down as legacy_down
    from moe_multistage_pipeline import DownReduceWorkspace
    from test_blockscaled import with_torch_sum

    tokens, topk, n, k, experts = 3, 2, 512, 256, 2
    a = torch.zeros((tokens, topk, k), dtype=torch.bfloat16, device="cuda")
    a[..., 0] = 1
    a[..., 128] = 1
    w = torch.zeros((experts, n, k), dtype=torch.bfloat16, device="cuda")
    w[0, :, 0] = 16
    w[0, :, 128] = -5
    w[1, :, 0] = -4
    a_scales = torch.ones((2, tokens * topk), dtype=torch.float32, device="cuda")
    b_scales = torch.ones((experts, n // 128, 2), dtype=torch.float32, device="cuda")
    b_scales[0, :, 1] = 2.3968749046325684
    # fma(-5, scale, 16) = 4.015625476837158, while rounded mul+add
    # gives 4.015625. Routing by 1/2 then BF16 RNE gives 2.015625 vs 2.
    # The other route is exactly -2, exposing a 0.015625 vs 0 sum.
    ids = torch.full((experts * 256,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    routes = torch.zeros(ids.shape, dtype=torch.float32, device="cuda")
    for expert in range(experts):
        ids[expert * 256:expert * 256 + tokens] = (
            torch.arange(tokens, dtype=torch.int32, device="cuda") | (expert << 24)
        )
        routes[expert * 256:expert * 256 + tokens] = 0.5
    eids = torch.arange(experts, dtype=torch.int32, device="cuda")
    valid = torch.tensor([experts * 256], dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    guard = torch.full((tokens * n + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = guard[n:-n].view(tokens, n)
    workspace = DownReduceWorkspace()
    down = legacy_down(n=n, k=k, topk=topk, num_experts=experts, block_n=block_n)
    pipeline = with_torch_sum(down, workspace)
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(16, 16)),
            a_scales, b_scales, ids, routes, eids, valid, counter)
    expected = torch.full_like(output, 0.015625)

    def check():
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.isfinite(output).all()
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()

    pipeline.poison_workspace(*args)
    pipeline(*args)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    output.fill_(torch.nan)
    counter.fill_(0x123456)
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    check()


@pytest.mark.parametrize("n,splits,topk,tokens,ctas", [
    (128, 1, 1, 65, 8),
    (384, 1, 2, 129, 8),
    (512, 4, 2, 257, 256),
    (6144, 4, 8, 513, 8),
])
def test_packed_pipeline(n, splits, topk, tokens, ctas):
    require_gpu()
    args, ref, _ = make_case(tokens, n, 256, 8, topk, 2026, "aiter", (16, 16))
    baseline = flydsl_moe_gemm_8wave_down(
        n=n, k=256, topk=topk, num_experts=8,
        **{**ATT_TUNED_BN128_CONFIG, "num_oc_splits": splits, "persistent_workgroups": ctas},
    )
    baseline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(args[0], ref, rtol=0.01, atol=0.01)
    expected = args[0].sum(dim=1)
    guard = torch.full((expected.numel() + n * 2,), torch.nan, device="cuda", dtype=torch.bfloat16)
    output = guard[n:-n].view_as(expected)
    inputs = (output, *args[1:])
    pipeline = compile_packed_down_reduce(n=n, k=256, topk=topk, num_experts=8,
                                         num_oc_splits=splits, persistent_workgroups=ctas)

    def check():
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.isfinite(output).all()
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()

    pipeline.poison_workspace(*inputs)
    pipeline(*inputs)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*inputs)
    for _ in range(3):
        output.fill_(torch.nan)
        inputs[-1].fill_(0x123456)
        pipeline.poison_workspace(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        check()
    # Change routing in-place, preserving addresses, to prove no inverse-data
    # cache and no consumption of unwritten rows from the previous launch.
    sentinel = (topk << 24) | tokens
    inputs[5].fill_(sentinel)
    output.fill_(torch.nan)
    pipeline.poison_workspace(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    inputs[8].zero_()
    pipeline(*inputs)
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    assert inputs[-1].item() == ctas


@pytest.mark.parametrize("layout", ["routed", "sorted", "packed", "linear", "linear_raw"])
def test_reducer_same_bf16(layout):
    require_gpu()
    tokens, topk, n, splits = 19, 8, 512, 4
    rows = 256
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn(tokens, topk, n, generator=generator, device="cuda").to(torch.bfloat16)
    # Permute the valid routes and poison the unwritten capacity tail.
    permutation = torch.randperm(tokens * topk, generator=generator, device="cuda")
    ids = torch.full((rows,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    ids[:tokens * topk] = ((permutation % topk) << 24 | permutation // topk).to(torch.int32)
    valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    inverse = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
    invert_sorted_ids(topk)(ids, inverse, valid, ids.numel(), tokens)
    sorted_values = torch.full((rows, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    sorted_values[:tokens * topk] = values.reshape(-1, n)[permutation]
    if layout == "routed":
        source = values
    else:
        source = torch.full((rows, n), torch.nan, dtype=torch.bfloat16, device="cuda")
        row = torch.arange(rows, device="cuda")[:, None]
        column = torch.arange(n, device="cuda")[None, :]
        oc, q, c = column // (n // splits), column % (n // splits) // 64, column % 64
        wave, r = row // 32, row % 32
        record = r % 2 * 2 + r // 16
        lane = r % 16 // 2 * 2 + c // 32 + c % 32 // 16 * 16 + c % 16 // 8 * 32
        prefix = oc * (256 * (n // splits)) + q * (256 * 64)
        index = (row * n + column if layout == "sorted" else
                 prefix + row * 64 + c if layout == "packed" else
                 prefix + wave * 2048 + record * 512 + lane * 8 + c % 8 if layout == "linear" else
                 prefix + wave * 2048 + c // 16 * 512 + (r + c % 8 // 4 * 32) * 8 + c % 16 // 8 * 4 + c % 4)
        source.view(-1)[index.expand(rows, n).reshape(-1)] = sorted_values.reshape(-1)
    output = torch.full((tokens, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    reduce = make_moe_sum(n=n, topk=topk, num_oc_splits=splits, output_layout=layout,
                          num_threads=256, block_cols=2048, read_policy=2)
    reduce(output, source, inverse)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, values.sum(dim=1), rtol=0.01, atol=0.01)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("argv,reduce_output", [
    ([], True),
    (["--mode", "down"], False),
    (["--profile", "--candidate", "4stage_bn128_tuned"], False),
    (["--profile", "--mode", "down-reduce", "--candidate", "4stage_bn128_tuned"], True),
])
def test_blockscaled_cli_scope(monkeypatch, argv, reduce_output):
    import sys
    import test_blockscaled as comparison

    calls = []
    monkeypatch.setattr(sys, "argv", [comparison.__file__, *argv])
    monkeypatch.setattr(comparison, "run_test", lambda *args, **kwargs: calls.append((args, kwargs)))
    comparison.main()
    assert len(calls) == 1
    assert calls[0][1]["reduce_output"] is reduce_output
    assert calls[0][0][9] is ("--profile" in argv)


@pytest.mark.parametrize("reduce_output", [False, True])
def test_blockscaled_down_and_total_metrics(reduce_output, capsys):
    require_gpu()
    import test_blockscaled as comparison

    tokens, topk, n, k = 513, 2, 512, 256
    results = comparison.run_test(
        tokens=tokens, model_dim=n, inter_dim=k, experts=4, topk=topk,
        block_m=256, num_oc_splits=1, seed=1234,
        candidates=["pyhip", "4stage_bn128_tuned"], reduce_output=reduce_output,
    )
    printed = capsys.readouterr().out
    assert len(results) == 2 and all(row["status"] == "PASS" for row in results)
    for row in results:
        assert row["elapsed_us"] > 0 and row["down_elapsed_us"] > 0
        assert row["unique_experts"] == 4 and row["valid_expert_blocks"] > 4
        assert row["rw_bytes"] == row["down_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["ideal_rw_bytes"] == row["down_ideal_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["rw_bytes"] - row["ideal_rw_bytes"] == (row["valid_expert_blocks"] - 4) * n * k
        assert row["down_effective_tflops"] == f"{2 * tokens * topk * n * k / row['down_elapsed_us'] / 1e6:.3f}"
        assert row["down_tb_per_s"] == f"{row['down_rw_bytes'] / row['down_elapsed_us'] / 1e6:.3f}"
        assert row["tb_per_s_ideal"] == f"{row['ideal_rw_bytes'] / row['elapsed_us'] / 1e6:.3f}"
        if reduce_output:
            assert row["measurement_scope"] == "down+reduce+inverse_if_needed"
            assert row["down_elapsed_us"] == row["components_us"]["gemm"]
            assert row["components_us"]["reduce"] > 0
            assert row["reduce_rw_bytes"] == (tokens * topk * n + tokens * n) * 2
            assert row["reduce_tb_per_s"] == f"{row['reduce_rw_bytes'] / row['components_us']['reduce'] / 1e6:.3f}"
            assert "| Reduce TB/s" in printed
            assert "Down time (us)" in printed and "Total Time (us)" in printed
        else:
            assert row["measurement_scope"] == "down"
            assert row["down_elapsed_us"] == row["elapsed_us"]
            assert row["reduce_rw_bytes"] == row["inverse_rw_bytes"] == 0
            assert row["reduce_tb_per_s"] == "N/A"
            assert "| Reduce TB/s" not in printed
            assert row["config"]["output_layout"] == "routed"
            assert "Total Time (us)" not in printed
    if reduce_output:
        pyhip, tuned = results
        assert pyhip["config"]["reduction"] == "torch" and not pyhip["config"]["includes_inverse"]
        assert "inverse" not in pyhip["components_us"]
        assert tuned["config"]["output_layout"] == "packed" and tuned["config"]["reduction"] == "custom"
        assert tuned["config"]["reduce_threads"] == 256 and tuned["config"]["reduce_cols"] == 2048
        assert tuned["config"]["read_policy"] == 2 and tuned["config"]["includes_inverse"]
        assert tuned["components_us"]["inverse"] > 0
        assert tuned["inverse_rw_bytes"] == (tuned["valid_expert_blocks"] * 256 + 2 * tokens * topk) * 4