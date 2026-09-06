"""Strict same-input single-wave SWA correctness and multi-wave comparisons."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pa_8wave"))
spec = importlib.util.spec_from_file_location("swa_reference", HERE.parent / "pa_8wave/test_pa_prefill.py")
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
sys.path.insert(0, str(HERE.parent / "pa_4wave"))
from pa_prefill_4wave import MHA
from pa_8wave_950 import PagedAttention as PA8
from swa_1wave import PagedAttention
from aiter.test_common import benchmark, checkAllclose, run_perftest
import aiter
import pandas as pd


requires_gfx950 = pytest.mark.skipif("gfx950" not in reference.GPU_ARCH, reason="requires gfx950")


def make_call(case, factory=PagedAttention, *, out=None, lse=None, **options):
    if out is None:
        out = torch.full((case.q.shape[0], case.heads, 128), -123.0, device=case.q.device, dtype=torch.bfloat16)
    kernel = factory(case.heads, case.kv_heads, case.dq, 128, 64, True,
                     window_left=case.window_left, has_sink=case.sinks is not None, **options)
    def call(**runtime):
        buffers = {"out": out, "sink_ptr": case.sinks}
        if lse is not None:
            buffers["lse"] = lse
        buffers.update(runtime)
        return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                      max(case.q_lens, default=0), max(case.kv_lens, default=0), True, case.qs, case.ks, case.vs,
                      case.last, **buffers)
    return call, out, kernel


def assert_case(case, *, layout="contiguous", softmax_scale=None, repeats=3, **options):
    shape = (case.q.shape[0], case.heads, 128)
    if layout == "padded":
        backing = torch.full((shape[0], shape[1] + 1, 144), -123, device="cuda", dtype=torch.bfloat16)
        out = backing[:, :case.heads, :128]
    elif layout == "head-major":
        backing = torch.full((shape[1], shape[0], 128), -123, device="cuda", dtype=torch.bfloat16)
        out = backing.transpose(0, 1)
    else:
        out = torch.full(shape, -123, device="cuda", dtype=torch.bfloat16)
    lse = torch.full(shape[:2], -123, device="cuda", dtype=torch.float32)
    call, _, _ = make_call(case, out=out, lse=lse, **options)
    first = None
    for _ in range(repeats):
        actual, actual_lse = call(return_lse=True, softmax_scale=softmax_scale)
        assert actual is out and actual_lse is lse
        if first is None:
            first = out.clone(), lse.clone()
        else:
            torch.testing.assert_close(out, first[0], rtol=0, atol=0)
            torch.testing.assert_close(lse, first[1], rtol=0, atol=0)
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    ref, ref_lse = reference.run_torch(case, True, softmax_scale)
    torch.testing.assert_close(out[begin:end].float(), ref[begin:end], rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse[begin:end], ref_lse[begin:end], rtol=2e-4, atol=5e-4)
    assert (out[:begin] == -123).all() and (out[end:] == -123).all()
    assert (lse[:begin] == -123).all() and (lse[end:] == -123).all()
    if layout == "padded":
        assert (backing[:, case.heads:] == -123).all()
        assert (backing[:, :case.heads, 128:] == -123).all()
    return out, lse


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("bn", [16, 32, 64])
@pytest.mark.parametrize("query_tile", [16, 32])
@pytest.mark.parametrize("q,kv,window,sink", [(33, 129, 128, True), (257, 777, 128, True),
                                             (17, 1, 0, False), (129, 193, 1, True),
                                             (33, 65, 64, False)])
def test_swa(dq, bn, query_tile, q, kv, window, sink):
    case = reference.make_case((q,), (kv,), heads=4, dq=dq, window_left=window, has_sink=sink)
    assert_case(case, block_n=bn, query_tile=query_tile)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window", [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512, 1024])
@pytest.mark.parametrize("sink", [False, True])
def test_window_boundaries(dq, window, sink):
    assert_case(reference.make_case((97,), (393,), heads=4, dq=dq, window_left=window, has_sink=sink))


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("kv", [0, 1, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 192, 193, 255, 256, 257, 321])
def test_poisoned_pages_and_all_masked_rows(dq, kv):
    assert_case(reference.make_case((257,), (kv,), heads=3, dq=dq, window_left=128))


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "head-major"])
@pytest.mark.parametrize("window", [1, 128])
def test_ragged_strides_offsets_and_scales(dq, layout, window):
    case = reference.make_case((0, 7, 129, 259), (63, 0, 193, 901), heads=6, kv_heads=2,
                               dq=dq, window_left=window, has_sink=True, nonunit_scales=True,
                               layout=layout, q_offset=5, table_offset=3)
    assert_case(case, layout=layout, softmax_scale=0.0625)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("mode", ["per-token", "per-tensor"])
@pytest.mark.parametrize("tile", [16, 32])
def test_large_logits_lazy_max_and_descales(dq, mode, tile):
    case = reference.make_case((129,), (513,), heads=4, kv_heads=2, dq=dq, mode=mode,
                               nonunit_scales=True, magnitude=4.0, window_left=128, has_sink=True)
    assert_case(case, query_tile=tile, softmax_scale=0.0625)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("sink", [-80.0, 0.0, 80.0, -float("inf")])
def test_sink_empty_rows(dq, tile, sink):
    case = reference.make_case((33, 259), (0, 65), heads=4, dq=dq, window_left=128,
                               has_sink=True, q_offset=5, table_offset=2, nonunit_scales=True)
    case.sinks.fill_(sink)
    assert_case(case, query_tile=tile)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window", [0, 16, 128])
def test_exact_inclusive_window_and_sink(dq, window):
    case = reference.make_case((129,), (257,), heads=2, dq=dq, window_left=window, has_sink=True)
    case.q.zero_()
    case.k_pages.zero_()
    case.v_pages.fill_(1)
    case.sinks.zero_()
    case.k, case.v = reference.vectorize_kv_cache(case.k_pages, case.v_pages, 1, dq, 128, 64)
    out, lse = assert_case(case)
    torch.testing.assert_close(out, torch.full_like(out, (window + 1) / (window + 2)), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(window + 2)), rtol=1e-6, atol=1e-6)
    case.sinks.fill_(-float("inf"))
    out, _ = assert_case(case)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("tile", [16, 32])
def test_runtime_lengths_pages_and_sinks(dq, tile):
    case = reference.make_case((65,), (2049,), heads=4, dq=dq, mode="per-tensor", window_left=128,
                               has_sink=True, nonunit_scales=True, poison_tail=False)
    lse = torch.empty(65, 4, device="cuda")
    _, out, kernel = make_call(case, lse=lse, query_tile=tile)
    keys, values = case.k_pages.clone(), case.v_pages.clone()
    compiled_count = None
    for length, sink in ((2049, 0.0), (257, -80.0), (128, 80.0), (0, -float("inf")), (193, 1.0), (65, 0.0)):
        case.kv_lens = (length,)
        case.indptr.copy_(reference._i32([0, (length + 63) // 64]))
        case.last.fill_((length - 1) % 64 + 1 if length else 0)
        case.sinks.fill_(sink)
        case.k_pages.copy_(keys)
        case.v_pages.copy_(values)
        if length % 64:
            physical = case.page_order[(length - 1) // 64]
            case.k_pages[physical, length % 64:] = float("nan")
            case.v_pages[physical, length % 64:] = float("nan")
        k, v = reference.vectorize_kv_cache(case.k_pages, case.v_pages, 1, dq, 128, 64)
        case.k.copy_(k)
        case.v.copy_(v)
        kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
               65, 2049, True, case.qs, case.ks, case.vs, case.last, out=out, lse=lse, sink_ptr=case.sinks)
        if compiled_count is None:
            compiled_count = len(kernel._compiled)
        assert len(kernel._compiled) == compiled_count
        ref, ref_lse = reference.run_torch(case, True)
        torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
def test_page_table_and_cache_mutations(dq):
    case = reference.make_case((33, 65), (192, 256), heads=4, kv_heads=2, dq=dq, table_offset=2,
                               window_left=128, has_sink=True)
    call, out, kernel = make_call(case)
    for mutation in range(3):
        if mutation == 1:
            case.page_order[2:] = case.page_order[:1] * (len(case.page_order) - 2)
            case.indices.copy_(reference._i32(case.page_order))
        if mutation == 2:
            case.k_pages.mul_(2)
            case.v_pages.mul_(0.5)
            k, v = reference.vectorize_kv_cache(case.k_pages, case.v_pages, 2, dq, 128, 64)
            case.k.copy_(k)
            case.v.copy_(v)
        before = len(kernel._compiled)
        call()
        if mutation:
            assert len(kernel._compiled) == before
        ref, _ = reference.run_torch(case, True)
        torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("tile", [16, 32])
def test_excluded_prefix_is_not_read(dq, tile):
    case = reference.make_case((257,), (8193,), heads=4, dq=dq, window_left=128, has_sink=True)
    case.indices[:(8193 - 257 - 128) // 64] = 2**30
    assert_case(case, query_tile=tile)


@requires_gfx950
@pytest.mark.parametrize("window", [0, 128])
def test_empty_queries(window):
    assert_case(reference.make_case((0,), (0,), heads=2, window_left=window, has_sink=True))


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window", [0, 128])
def test_streams_graphs_and_output_allocation(dq, window):
    case = reference.make_case((127,), (193,), heads=4, dq=dq, window_left=window, has_sink=True)
    call, _, _ = make_call(case)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs, lses, graphs = [], [], []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out, lse = call(out=None, return_lse=True, stream=stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call(out=out, lse=lse, stream=stream)
            outputs.append(out)
            lses.append(lse)
            graphs.append(graph)
    for _ in range(5):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    ref, ref_lse = reference.run_torch(case, True)
    for out, lse in zip(outputs, lses):
        torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window", [0, 128])
def test_direct_only_one_launch_no_workspace(dq, window, monkeypatch):
    case = reference.make_case((257,), (901,), dq=dq, window_left=window, has_sink=True)
    call, out, kernel = make_call(case)
    call()
    torch.cuda.synchronize()
    assert not hasattr(kernel, "prepare_kv") and not hasattr(kernel, "_workspace")
    def no_allocation(*args, **kwargs):
        pytest.fail("warmed single-wave dispatch must not allocate a workspace")
    before = torch.cuda.memory_allocated()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", no_allocation)
        patch.setattr(torch, "empty_like", no_allocation)
        call()
        torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    gpu_events = [event for event in prof.events() if "CUDA" in str(event.device_type)]
    assert len(gpu_events) == 1 and "_swa" in gpu_events[0].name
    ref, _ = reference.run_torch(case, True)
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)


def test_explicit_scope_and_default_tiles():
    for window, tile in ((0, 16), (16, 16), (17, 32), (128, 32)):
        kernel = PagedAttention(16, 1, 192, 128, 64, window_left=window)
        assert (kernel.query_tile, kernel.block_n) == (tile, tile)
    for config in ((16, 1, 64, 128, 64), (16, 1, 192, 64, 64), (16, 1, 192, 128, 32)):
        with pytest.raises(NotImplementedError):
            PagedAttention(*config)
    with pytest.raises(ValueError):
        PagedAttention(7, 2, 192, 128, 64)
    for options in ({"is_causal": False}, {"window_left": -1}, {"query_tile": 64}, {"block_n": 8}):
        with pytest.raises(ValueError):
            PagedAttention(16, 1, 192, 128, 64, **options)


@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("bn", [16, 32, 64])
def test_fragment_address_coverage(dq, bn):
    # Pure Python address model: each K/V element is loaded by exactly one
    # lane/register, including the BN32/64 interleaved key-row permutation.
    for tile in range(0, 64, bn):
        key_addresses, value_addresses = [], []
        atom_k = 16 if bn == 16 else 32
        for lane in range(64):
            for n in range(bn // 16):
                row = lane & 15 if bn == 16 else (lane & 3) + ((lane & 12) << 1) + (n & 1) * 4 + (n // 2) * 32
                for k in range(dq // 32):
                    key_addresses.extend((((lane >> 4) + k * 4) * 64 + tile + row) * 8 + i for i in range(8))
            for n in range(8):
                for k in range(bn // atom_k):
                    token = tile + (lane >> 4) * (atom_k // 4) + k * atom_k
                    value_addresses.extend((token // 8 * 128 + n * 16 + (lane & 15)) * 8 + (token & 7) + i
                                           for i in range(atom_k // 4))
        expected_k = {(d // 8 * 64 + token) * 8 + d % 8 for token in range(tile, tile + bn) for d in range(dq)}
        expected_v = {(token // 8 * 128 + d) * 8 + token % 8 for token in range(tile, tile + bn) for d in range(128)}
        assert len(key_addresses) == len(set(key_addresses)) == bn * dq
        assert len(value_addresses) == len(set(value_addresses)) == bn * 128
        assert set(key_addresses) == expected_k and set(value_addresses) == expected_v


@pytest.mark.parametrize("qt", [16, 32])
@pytest.mark.parametrize("bn", [16, 32, 64])
def test_query_output_layout_and_window_coverage(qt, bn):
    output = [((lane & 15) + m * 16, (lane >> 4) * 4 + n * 16 + i)
              for lane in range(64) for m in range(qt // 16) for n in range(8) for i in range(4)]
    assert len(output) == len(set(output)) == qt * 128
    assert set(output) == set(itertools.product(range(qt), range(128)))
    for q_len, kv_len, window in itertools.product((1, 17, 33, 129), (0, 1, 63, 65, 256), (0, 1, 16, 31, 64, 128, 512)):
        for qstart in range(0, q_len, qt):
            valid_q = min(qt, q_len - qstart)
            first = max(0, qstart + kv_len - q_len - window) & -bn
            end = max(0, min(kv_len, qstart + valid_q + kv_len - q_len))
            visited = [col for tile in range(first, end, bn) for col in range(tile, tile + bn)]
            for row in range(qstart, qstart + valid_q):
                diagonal = kv_len - q_len + row
                accepted = {col for col in visited if ((diagonal - col) & 0xFFFFFFFF) <= window}
                expected = set(range(max(0, diagonal - window), max(0, min(kv_len, diagonal + 1))))
                assert accepted == expected


@requires_gfx950
@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (4, 4), (8, 2), (6, 3)])
@pytest.mark.parametrize("dq", [128, 192])
def test_mha_gqa_and_storage_offsets(dq, heads, kv_heads):
    case = reference.make_case((65,), (257,), heads=heads, kv_heads=kv_heads, dq=dq,
                               window_left=128, nonunit_scales=True, has_sink=True)
    backing_q = torch.zeros(67, heads + 1, dq + 16, device="cuda", dtype=torch.bfloat16)
    q = backing_q[1:66, :heads, 8:8 + dq]
    q.copy_(case.q)
    case.q = q
    backing_out = torch.full((67, heads + 1, 144), -123, device="cuda", dtype=torch.bfloat16)
    out = backing_out[1:66, :heads, 8:136]
    call, _, _ = make_call(case, out=out)
    call()
    ref, _ = reference.run_torch(case, True)
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
    assert (backing_out[0] == -123).all() and (backing_out[-1] == -123).all()
    assert (backing_out[:, heads:] == -123).all()
    assert (backing_out[:, :heads, :8] == -123).all() and (backing_out[:, :heads, 136:] == -123).all()


@requires_gfx950
def test_invalid_buffers_fail_before_launch():
    case = reference.make_case((9,), (65,), heads=4, window_left=128, has_sink=True)
    call, _, _ = make_call(case)
    for sink in (None, torch.zeros(3, device="cuda"), torch.zeros(4),
                 torch.zeros(8, device="cuda")[::2], torch.zeros(4, device="cuda", dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match="sink"):
            call(sink_ptr=sink)
    with pytest.raises(ValueError, match="output"):
        call(out=torch.empty(9, 4, 128, device="cuda", dtype=torch.float32))
    with pytest.raises(ValueError, match="LSE"):
        call(lse=torch.empty(9, 4, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="softmax_scale"):
        call(softmax_scale=0)
    case.q = case.q.float()
    with pytest.raises(NotImplementedError, match="BF16"):
        call()


@benchmark()
def bench(q_len, kv_len, dq, window, configs, candidates, records):
    case = reference.make_case((q_len,), (kv_len,), dq=dq, window_left=window, has_sink=True, poison_tail=False)
    ref, _ = reference.run_torch(case, True)
    calls, tiles = {}, {}
    factories = {"1w_auto": (PagedAttention, {}), "4w_static": (MHA, {}),
                 "4w_dynamic": (MHA, {"force_dynamic_schedule": True}),
                 "8w_static": (PA8, {}), "8w_persistent": (PA8, {"persistent": True})}
    for name in candidates:
        if name == "1w_tiles":
            for qt, bn in configs:
                key = f"1w_q{qt}_bn{bn}"
                calls[key] = make_call(case, query_tile=qt, block_n=bn)[0]
                tiles[key] = {"query_tile": qt, "block_n": bn, "threads": 64}
        else:
            factory, options = factories[name]
            calls[name], _, kernel = make_call(case, factory, **options)
            if name == "1w_auto":
                tiles[name] = {"query_tile": kernel.query_tile, "block_n": kernel.block_n, "threads": 64}
    errors, samples = {}, {name: [] for name in calls}
    for name, call in calls.items():
        out = call()
        errors[name] = float(checkAllclose(ref, out.float(), rtol=0.02, atol=0.02, tol_err_ratio=0, msg=name))
        assert errors[name] == 0, name
        first = out.clone()
        for _ in range(2):
            torch.testing.assert_close(call(), first, rtol=0, atol=0)
    dispatches = {}
    for name, call in calls.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        events = [event.name for event in prof.events() if "CUDA" in str(event.device_type)]
        assert len(events) == 1, (name, events)
        dispatches[name] = events
    for _ in range(100):
        for call in calls.values():
            call()
    for trial in range(5):
        for name in list(calls) if trial % 2 == 0 else list(reversed(calls)):
            _, us = run_perftest(calls[name], num_warmup=20, num_iters=100, num_rotate_args=1)
            samples[name].append(float(us))
    medians = {name: statistics.median(v) for name, v in samples.items()}
    pairs = sum(max(0, min(kv_len, kv_len - q_len + r + 1) - max(0, kv_len - q_len + r - window)) for r in range(q_len))
    flops = 2 * 16 * pairs * (dq + 128)
    nbytes = 2 * (q_len * 16 * (dq + 128) + (kv_len - max(0, (kv_len - q_len - window) // 64) * 64) * (dq + 128))
    sources = [HERE / "swa_1wave.py", HERE / "test_swa_1wave.py", HERE.parent / "pa_4wave/pa_prefill_4wave.py",
               HERE.parent / "pa_8wave/pa_8wave_950.py", HERE.parent / "pa_8wave/test_pa_prefill.py"]
    records.append({"q": q_len, "kv": kv_len, "dq": dq, "window": window, "samples_us": samples,
                    "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    "gfx": reference.GPU_ARCH, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "hip": torch.version.hip,
                    "batch": 1, "heads": 16, "kv_heads": 1, "dv": 128, "page": 64, "dtype": "bfloat16",
                    "sink": True, "preallocated_output": True, "with_lse": False, "seed": 20260905,
                    "dispatches": dispatches, "single_wave_tiles": tiles,
                    "timing": {"method": "aiter.run_perftest", "common_warmup": 100, "warmup": 20,
                               "iterations": 100, "trials": 5, "alternating_candidate_order": True},
                    "median_us": medians, "errors": errors, "effective_flops": flops, "logical_bytes": nbytes})
    print("SWA1_RESULT", json.dumps(records[-1]), flush=True)
    ret = {"gfx": reference.GPU_ARCH}
    for name, us in medians.items():
        ret.update({f"{name} us": us, f"{name} TFLOPS": flops / us / 1e6,
                    f"{name} TB/s": nbytes / us / 1e6, f"{name} err": errors[name]})
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, nargs="+", default=[16384])
    parser.add_argument("--kv", type=int, nargs="+", default=[131072])
    parser.add_argument("--dq", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--window", type=int, nargs="+", default=[128])
    parser.add_argument("--bn", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--query-tile", type=int, nargs="+", choices=(16, 32), default=[16, 32])
    parser.add_argument("--candidates", nargs="+", choices=("1w_auto", "1w_tiles", "4w_static", "4w_dynamic", "8w_static", "8w_persistent"),
                        default=["1w_auto", "4w_static", "4w_dynamic", "8w_static", "8w_persistent"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if "gfx950" not in reference.GPU_ARCH:
        raise SystemExit("requires gfx950")
    records, rows = [], []
    for dq, q, kv, window in itertools.product(args.dq, args.q, args.kv, args.window):
        row = bench(q, kv, dq, window, list(itertools.product(args.query_tile, args.bn)), args.candidates, records)
        for name in ("records", "configs", "candidates"):
            row.pop(name, None)
        rows.append(row)
        if args.output:
            args.output.write_text(json.dumps(records, indent=2) + "\n")
    aiter.logger.info("Single-wave SWA candidate comparison:\n%s", pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()