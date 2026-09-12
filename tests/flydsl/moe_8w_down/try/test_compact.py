# SPDX-License-Identifier: MIT
"""Local validation for M256 one-shot scheduling and mixed M64 tail tasks."""

import pytest
import torch

from test_8stage import make_case
from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_compact import make_compact_down_reduce, make_task_builder, task_launch_stats


@pytest.mark.parametrize("tasks", [0, 1, 7, 8, 9, 255, 256, 257, 768, 3072])
def test_xcd8_bijection(tasks):
    chunk = tasks // 8
    mapping = [(i % 8) * chunk + i // 8 if i < chunk * 8 else i for i in range(tasks + 17)]
    assert sorted(mapping[:tasks]) == list(range(tasks))
    assert mapping[tasks:] == list(range(tasks, tasks + 17))


@pytest.mark.parametrize("tokens,experts,topk,n,splits,threshold,tail_splits", [
    (65, 4, 2, 512, 4, 0, None),
    (257, 4, 2, 512, 4, 0, None),
    (193, 1, 1, 512, 2, 0, None),
    (1025, 4, 2, 512, 4, 0, None),
    (1025, 4, 2, 512, 1, 0.6, None),
    (769, 1, 1, 512, 2, 0, None),
    (513, 8, 8, 6144, 4, 0, None),
    (65, 4, 2, 512, 1, 0, 4),
    (193, 1, 1, 512, 1, 0, 4),
    (1025, 4, 2, 512, 1, 0, 4),
    (1025, 4, 2, 512, 1, 0.6, 4),
    (513, 8, 8, 6144, 1, 0, 4),
    (1025, 4, 2, 512, 4, 0, 1),
])
def test_compact_pipeline(tokens, experts, topk, n, splits, threshold, tail_splits):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")
    args, reference, guard = make_case(tokens, n, 256, experts, topk, 2026, "aiter")
    options = {**ATT_TUNED_BN128_CONFIG, "num_oc_splits": splits}
    baseline = flydsl_moe_gemm_8wave_down(n=n, k=256, topk=topk, num_experts=experts, **options)
    baseline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(args[0], reference, rtol=0.01, atol=0.01)
    expected = args[0].sum(dim=1)
    oneshot = flydsl_moe_gemm_8wave_down(n=n, k=256, topk=topk, num_experts=experts,
                                         **options, persistent=False, xcd_swizzle=True)
    args[0].fill_(torch.nan)
    oneshot(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(args[0].sum(dim=1), expected, rtol=0.01, atol=0.01)
    assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
    compact = make_compact_down_reduce(n=n, k=256, topk=topk, num_experts=experts,
                                       num_oc_splits=splits, tail_num_oc_splits=tail_splits,
                                       min_tail_utilization=threshold)
    output = torch.full_like(expected, torch.nan)
    inputs = (output, *args[1:])
    compact.poison_workspace(*inputs)
    compact(*inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
    assert torch.isfinite(output).all()
    stats = compact.benchmark_stats(*inputs)
    assert stats["split_cu_count"] == torch.cuda.get_device_properties(output.device).multi_processor_count
    assert sum(part["valid_rows"] for part in stats["kernel_work"].values()) == tokens * topk
    assert sum(part["compute_rows"] for part in stats["kernel_work"].values()) == stats["compute_rows"]
    actual_tail_splits = splits if tail_splits is None else tail_splits
    assert compact.config["full_num_oc_splits"] == splits
    assert compact.config["tail_num_oc_splits"] == actual_tail_splits
    for label, part in stats["kernel_work"].items():
        expected_splits = splits if label == "full256" else actual_tail_splits
        assert part["xcd_swizzle"] and part["xcd_count"] == 8
        assert part["num_oc_splits"] == expected_splits
        assert part["active_workgroups"] == part["tasks"] * expected_splits
        assert part["effective_flops"] == 2 * part["valid_rows"] * n * 256
        assert part["rw_bytes"] == part["valid_rows"] * (256 + n * 2) + part["tasks"] * n * 256
    assert stats["tail_launch_workgroups"] == compact.workspace.tail.shape[0] * actual_tail_splits
    full_rows, tail_rows = compact.workspace.counts.tolist()
    assert full_rows % 256 == 0 and tail_rows % 64 == 0
    ids = args[5].cpu().to(torch.int64)
    limit = args[8][0].item()
    live = ((ids[:limit] & 0xFFFFFF) < tokens) & ((ids[:limit] >> 24) < topk)
    coverage = torch.zeros(limit, dtype=torch.int32, device="cpu")
    for table, rows, bm in ((compact.workspace.full, full_rows, 256), (compact.workspace.tail, tail_rows, 64)):
        for start, expert in table[:rows // bm].cpu().tolist():
            assert 0 <= start < limit and start % 64 == 0
            assert expert == args[7][start // 256].item()
            assert start // 256 == (start + bm - 1) // 256
            coverage[start:start + bm] += 1
    assert torch.all(coverage[live] == 1) and torch.all(coverage <= 1)
    if threshold == 0 and tokens * topk > experts * 256:
        assert full_rows > 0
    if tokens == 193 and experts == 1:
        assert full_rows == 256 and tail_rows == 0, "four occupied M64 chunks must form an M256 task"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compact(*inputs)
    compact.poison_workspace(*inputs)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
    # Same pointers but new empty routing must rebuild task counts and inverse.
    args[8].zero_()
    compact.poison_workspace(*inputs)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    assert compact.workspace.counts.tolist() == [0, 0]
    assert torch.count_nonzero(output).item() == 0
    empty = compact.benchmark_stats(*inputs)
    assert all(p["valid_rows"] == p["rw_bytes"] == p["padded_flops"] == 0 for p in empty["kernel_work"].values())


@pytest.mark.parametrize("splits,cu_count,last_util,loss", [
    (1, 256, 0.5, 0.25),
    (2, 256, 1.0, 0.0),
    (4, 256, 1.0, 0.0),
    (1, 80, 0.8, 0.04),
])
def test_384_tasks_cu_round_model(splits, cu_count, last_util, loss):
    stats = task_launch_stats(384, 512, splits, cu_count)
    assert stats["last_batch_utilization"] == last_util
    assert stats["cu_capacity_loss"] == pytest.approx(loss)


@pytest.mark.parametrize("splits,cu_count,expected_full,expected_tail", [
    (1, 256, 256, 512),
    (2, 256, 384, 0),
    (4, 256, 384, 0),
    (1, 80, 384, 0),
])
def test_task_builder_384_cu_balance(splits, cu_count, expected_full, expected_tail):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")
    # Inject CU counts only in this builder unit test; the production path
    # always queries the actual device. Distinguish256 from the old80 example.
    capacity, tokens = 384, 384 * 256
    ids = torch.arange(tokens, dtype=torch.int32, device="cuda")
    experts = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    full = torch.empty((capacity, 2), dtype=torch.int32, device="cuda")
    tail = torch.empty((capacity * 4, 2), dtype=torch.int32, device="cuda")
    counts = torch.empty(2, dtype=torch.int32, device="cuda")
    make_task_builder(capacity, 1, cu_count, 0.6, splits)(ids, experts, valid, full, tail, counts, tokens)
    torch.cuda.synchronize()
    assert counts.tolist() == [expected_full * 256, expected_tail * 64]
    assert full[:expected_full, 0].cpu().tolist() == list(range(0, expected_full * 256, 256))
    if expected_tail:
        assert tail[:expected_tail, 0].cpu().tolist() == list(range(expected_full * 256, tokens, 64))


@pytest.mark.parametrize("splits,tail_splits,threshold", [(4, None, 0), (1, 4, 0), (1, 4, 0.6)])
def test_compact_component_report(monkeypatch, capsys, splits, tail_splits, threshold):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")
    import test_blockscaled as comparison

    key = "compact_component_test"
    monkeypatch.setitem(comparison.CANDIDATES, key, (key, 128, make_compact_down_reduce))
    monkeypatch.setitem(comparison.CANDIDATE_OPTIONS, key, {
        "num_oc_splits": splits, "tail_num_oc_splits": tail_splits, "min_tail_utilization": threshold,
    })
    result, = comparison.run_test(tokens=1025, model_dim=512, inter_dim=256, experts=4, topk=2,
                                  block_m=256, num_oc_splits=1, seed=1234,
                                  candidates=[key], reduce_output=True)
    text = capsys.readouterr().out
    assert result["status"] == "PASS" and "Compact kernel breakdown:" in text
    assert "M256" in text and "M64" in text and "Overall CU capacity loss (model)" in text
    assert "Full M256 OC splits" in text and "Tail M64 OC splits" in text
    assert sum(p["valid_rows"] for p in result["component_metrics"].values()) == 2050
    for label, part in result["component_metrics"].items():
        assert part["num_oc_splits"] == (splits if label == "full256" or tail_splits is None else tail_splits)
        us = result["components_us"][label]
        assert part["elapsed_us"] == us > 0
        assert part["effective_tflops"] == f"{part['effective_flops'] / us / 1e6:.3f}"
        assert part["padded_tflops"] == f"{part['padded_flops'] / us / 1e6:.3f}"
        assert part["tb_per_s"] == f"{part['rw_bytes'] / us / 1e6:.3f}"
        assert part["tb_per_s_ideal"] == f"{part['ideal_rw_bytes'] / us / 1e6:.3f}"


@pytest.mark.parametrize("n", [128, 512, 768, 6144])
def test_packed_offset_is_oc_independent(n):
    rows = torch.tensor([0, 1, 63, 64, 127, 128, 192, 255, 256, 319, 511], device="cpu")[:, None]
    columns = torch.arange(n, device="cpu")[None, :]
    canonical = rows // 256 * (256 * n) + columns // 64 * (256 * 64) + rows % 256 * 64 + columns % 64
    for splits in (1, 2, 3, 4, 8):
        if n % (128 * splits):
            continue
        n_split = n // splits
        # Same element formula as the M64 store/reducer. M256's byte base
        # plus its N64 soffset is equivalent when its N starts on a128 boundary.
        packed = (rows // 256 * (256 * n) + columns // n_split * (256 * n_split)
                  + columns % n_split // 64 * (256 * 64) + rows % 256 * 64 + columns % 64)
        torch.testing.assert_close(packed, canonical, rtol=0, atol=0)