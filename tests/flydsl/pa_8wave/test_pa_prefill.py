"""Strict direct-paged gfx950 BF16 D128/D192, V128 attention tests and benchmarks.

Only the supported OPUS-style specialization is exercised here. The torch
reference is independent of the attention kernel. No accuracy
exception is caught; every candidate must pass before its timing is reported.
"""

import argparse
import importlib.util
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass
from importlib import metadata
from itertools import accumulate
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import torch

# This workspace's pytest collection can pre-load a namespace named triton.
# Resolve the installed editable distribution before aiter imports torch.dynamo.
import triton
if getattr(triton, "__version__", None) is None:
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            sys.modules.pop(name)
    origin = json.loads(metadata.distribution("triton").read_text("direct_url.json"))
    package = Path(unquote(urlparse(origin["url"]).path)) / "python" / "triton"
    spec = importlib.util.spec_from_file_location("triton", package / "__init__.py", submodule_search_locations=[str(package)])
    triton = importlib.util.module_from_spec(spec)
    sys.modules["triton"] = triton
    spec.loader.exec_module(triton)

from aiter.test_common import benchmark, checkAllclose, run_perftest
from pa_8wave_950 import PagedAttention


GPU_ARCH = torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else ""
requires_gfx950 = pytest.mark.skipif("gfx950" not in GPU_ARCH, reason="requires gfx950")
DQ, DV, PAGE = 192, 128, 64


def vectorize_kv_cache(k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size):
    """Convert [page, token, head, dim] to the public SHUFFLE-5D ABI."""
    vector = 16 // k_cache.element_size()
    k = k_cache.reshape(-1, page_size, num_kv_heads, head_dim_qk // vector, vector)
    v = v_cache.reshape(-1, page_size // vector, vector, num_kv_heads, head_dim_v)
    return k.permute(0, 2, 3, 1, 4).contiguous(), v.permute(0, 3, 1, 4, 2).contiguous()


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@dataclass
class Case:
    q: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cq: torch.Tensor
    ck: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    vs: torch.Tensor
    q_lens: tuple
    kv_lens: tuple
    page_order: list
    q_offset: int
    table_offset: int
    mode: str
    window_left: int
    sinks: torch.Tensor | None

    @property
    def heads(self):
        return self.q.shape[1]

    @property
    def kv_heads(self):
        return self.k_pages.shape[2]

    @property
    def dq(self):
        return self.q.shape[-1]

    def factory(self, causal, persistent=False):
        return PagedAttention(self.heads, self.kv_heads, self.dq, DV, PAGE, causal, self.mode,
                              window_left=self.window_left, has_sink=self.sinks is not None,
                              persistent=persistent)

    def run(self, causal, out=None, *, persistent=False, **kwargs):
        kwargs.setdefault("sink_ptr", self.sinks)
        return self.factory(causal, persistent)(
            self.q, self.k, self.v, self.cq, self.ck, self.indptr, self.indices,
            max(self.q_lens, default=0), max(self.kv_lens, default=0), causal,
            self.qs, self.ks, self.vs, self.last, out=out, **kwargs,
        )

    def logical_kv(self):
        keys, values = [], []
        pos = self.table_offset
        for length in self.kv_lens:
            count = (length + PAGE - 1) // PAGE
            ids = self.page_order[pos:pos + count]
            keys.append(self.k_pages[ids].reshape(-1, self.kv_heads, self.dq)[:length])
            values.append(self.v_pages[ids].reshape(-1, self.kv_heads, DV)[:length])
            pos += count
        return keys, values


def make_case(q_lens=(256,), kv_lens=(256,), *, heads=16, kv_heads=1,
              mode="per-token", layout="contiguous", q_offset=0, table_offset=0,
              nonunit_scales=False, magnitude=1.0, seed=20260905, dq=DQ,
              window_left=-1, has_sink=False, poison_tail=True):
    assert len(q_lens) == len(kv_lens)
    torch.manual_seed(seed)
    q_lens, kv_lens = tuple(q_lens), tuple(kv_lens)
    tokens = q_offset + sum(q_lens) + (7 if q_offset else 0)
    if layout == "head-major":
        q = torch.randn(heads, tokens, dq, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    elif layout == "padded":
        q = torch.randn(tokens, heads + 1, dq + 16, dtype=torch.bfloat16, device="cuda")[:, :heads, :dq]
    else:
        q = torch.randn(tokens, heads, dq, dtype=torch.bfloat16, device="cuda")
    q.mul_(magnitude)
    counts = [(n + PAGE - 1) // PAGE for n in kv_lens]
    pages = max(1, sum(counts))
    k_pages = torch.randn(pages, PAGE, kv_heads, dq, dtype=torch.bfloat16, device="cuda") * magnitude
    v_pages = torch.randn(pages, PAGE, kv_heads, DV, dtype=torch.bfloat16, device="cuda")
    order = [0] * table_offset + torch.randperm(sum(counts)).tolist()
    start = table_offset
    for length, count in zip(kv_lens, counts):
        if length % PAGE:
            # A bounds bug must not pass just because padded KV happens to be 0.
            k_pages[order[start + count - 1], length % PAGE:] = float("nan") if poison_tail else 0
            v_pages[order[start + count - 1], length % PAGE:] = float("nan") if poison_tail else 0
        start += count
    k, v = vectorize_kv_cache(k_pages, v_pages, kv_heads, dq, DV, PAGE)
    qs = torch.ones((tokens, heads, 1) if mode == "per-token" else (1,), device="cuda")
    ks, vs = torch.ones(1, device="cuda"), torch.ones(1, device="cuda")
    if nonunit_scales:
        qs.uniform_(0.5, 1.25)
        ks.fill_(0.75)
        vs.fill_(1.25)
    return Case(
        q, k_pages, v_pages, k, v,
        _i32(list(accumulate(q_lens, initial=q_offset))),
        _i32(list(accumulate(kv_lens, initial=11 if table_offset else 0))),
        _i32(list(accumulate(counts, initial=table_offset))), _i32(order),
        _i32([(n - 1) % PAGE + 1 if n else 0 for n in kv_lens]),
        qs, ks, vs, q_lens, kv_lens, order, q_offset, table_offset, mode, window_left,
        torch.linspace(-1.0, 1.0, heads, device="cuda", dtype=torch.float32) if has_sink else None,
    )


def run_torch(case, causal, softmax_scale=None):
    """FP32 reference with bottom-right causal alignment and empty-row semantics."""
    keys, values = case.logical_kv()
    output = torch.full((case.q.shape[0], case.heads, DV), float("nan"), device=case.q.device)
    lse = torch.full((case.q.shape[0], case.heads), float("nan"), device=case.q.device)
    scale = 1 / math.sqrt(case.dq) if softmax_scale is None else softmax_scale
    offset = case.q_offset
    for q_len, kv_len, k, v in zip(case.q_lens, case.kv_lens, keys, values):
        k = (k.float() * case.ks).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        v = (v.float() * case.vs).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        for start in range(0, q_len, 256):
            end = min(q_len, start + 256)
            if kv_len == 0:
                output[offset + start:offset + end] = 0
                lse[offset + start:offset + end] = case.sinks if case.sinks is not None else -float("inf")
                continue
            q = case.q[offset + start:offset + end].float()
            qs = case.qs if case.qs.numel() == 1 else case.qs[offset + start:offset + end]
            key_start, key_end = 0, kv_len
            if case.window_left >= 0:
                # Crop only the union of visible keys for this reference chunk.
                key_start = max(0, kv_len - q_len + start - case.window_left)
                key_end = max(0, min(kv_len, kv_len - q_len + end))
            logits = ((q * qs).transpose(0, 1) @ k[:, key_start:key_end].transpose(1, 2)) * scale
            if causal:
                rows = torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(key_start, key_end, device=q.device)[None, :]
                logits.masked_fill_(cols > kv_len - q_len + rows, -float("inf"))
                if case.window_left >= 0:
                    logits.masked_fill_(cols < kv_len - q_len + rows - case.window_left, -float("inf"))
            if case.sinks is not None:
                sink_logits = case.sinks[:, None, None].expand(-1, end - start, 1)
                logits = torch.cat((logits, sink_logits), dim=-1)
            probs = torch.softmax(logits, dim=-1).nan_to_num(0.0)
            if case.sinks is not None:
                probs = probs[..., :-1]
            output[offset + start:offset + end] = (probs @ v[:, key_start:key_end]).transpose(0, 1)
            lse[offset + start:offset + end] = torch.logsumexp(logits, dim=-1).transpose(0, 1)
        offset += q_len
    return output, lse


def assert_case(case, causal, *, repeats=3, layout="contiguous", softmax_scale=None, persistent=False):
    shape = (case.q.shape[0], case.heads, DV)
    if layout == "padded":
        backing = torch.full((shape[0], shape[1] + 1, DV + 16), -123.0, device="cuda", dtype=torch.bfloat16)
        out = backing[:, :case.heads, :DV]
    elif layout == "head-major":
        backing = torch.full((shape[1], shape[0], DV), -123.0, device="cuda", dtype=torch.bfloat16)
        out = backing.transpose(0, 1)
    else:
        out = torch.full(shape, -123.0, device="cuda", dtype=torch.bfloat16)
    lse = torch.full(shape[:2], -123.0, device="cuda", dtype=torch.float32)
    first = None
    for _ in range(repeats):
        result, result_lse = case.run(causal, out, return_lse=True, lse=lse, softmax_scale=softmax_scale,
                        persistent=persistent)
        assert result is out and result_lse is lse
        if first is None:
            first = (out.clone(), lse.clone())
        else:
            torch.testing.assert_close(out, first[0], rtol=0, atol=0)
            torch.testing.assert_close(lse, first[1], rtol=0, atol=0)
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    reference, ref_lse = run_torch(case, causal, softmax_scale)
    assert torch.isfinite(out[begin:end]).all()
    torch.testing.assert_close(out[begin:end].float(), reference[begin:end], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse[begin:end], ref_lse[begin:end], rtol=2e-4, atol=5e-4)
    assert (out[:begin] == -123).all() and (out[end:] == -123).all()
    assert (lse[:begin] == -123).all() and (lse[end:] == -123).all()
    if layout == "padded":
        assert (backing[:, case.heads:] == -123).all()
        assert (backing[:, :case.heads, DV:] == -123).all()
    return out, lse


@requires_gfx950
@pytest.mark.parametrize("kv_len", [1, 63, 64, 65, 128, 129, 192, 193, 255, 256, 257, 320, 321])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dq", [128, 192])
def test_page_parity_and_tails(kv_len, causal, dq):
    assert_case(make_case((257,), (kv_len,), heads=4, dq=dq), causal)


@requires_gfx950
def test_native_bf16_direct_lds_subgroup_rings_are_deterministic():
    assert_case(make_case((256,), (256,)), False, repeats=10)


@requires_gfx950
@pytest.mark.parametrize("kv_len", [256, 321])
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window_left,has_sink", [(-1, False), (-1, True), (128, False), (128, True)])
def test_default_no_lse_path_is_deterministic(kv_len, dq, window_left, has_sink):
    case = make_case((257,), (kv_len,), dq=dq, window_left=window_left, has_sink=has_sink)
    causal = window_left >= 0
    out = torch.full((257, 16, DV), float("nan"), device="cuda", dtype=torch.bfloat16)
    first = case.run(causal, out).clone()
    for _ in range(10):
        assert case.run(causal, out) is out
        torch.testing.assert_close(out, first, rtol=0, atol=0)
    ref, _ = run_torch(case, causal)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "head-major"])
def test_ragged_batch_strides_and_metadata_offsets(causal, layout):
    case = make_case((0, 7, 129, 259), (63, 0, 193, 321), heads=6, kv_heads=2,
                     q_offset=5, table_offset=3, layout=layout, nonunit_scales=True)
    assert_case(case, causal, layout=layout)


@requires_gfx950
@pytest.mark.parametrize("mode", ["per-token", "per-tensor"])
@pytest.mark.parametrize("causal", [False, True])
def test_descales_and_lazy_max_rescale(mode, causal):
    case = make_case((129,), (321,), heads=4, kv_heads=2, mode=mode,
                     nonunit_scales=True, magnitude=4.0)
    assert_case(case, causal, softmax_scale=0.0625)


@requires_gfx950
def test_direct_attention_reads_current_pages():
    case = make_case((17, 23), (129, 256), heads=6, kv_heads=2, table_offset=2)
    for factor in (1.0, 2.0):
        case.k_pages.mul_(factor)
        case.v_pages.mul_(factor)
        k, v = vectorize_kv_cache(case.k_pages, case.v_pages, 2, DQ, DV, PAGE)
        case.k.copy_(k)
        case.v.copy_(v)
        assert_case(case, False)


@requires_gfx950
@pytest.mark.parametrize("persistent", [False, True])
def test_page_table_changes_between_calls(persistent):
    case = make_case((129,), (320,), heads=4, table_offset=2)
    assert_case(case, False, persistent=persistent)
    case.page_order[2:] = list(reversed(case.page_order[2:]))
    case.indices.copy_(_i32(case.page_order))
    assert_case(case, False, persistent=persistent)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_shared_physical_pages_across_sequences(dq, causal, persistent):
    case = make_case((9, 33), (129, 257), heads=6, kv_heads=2, dq=dq,
                     table_offset=2, q_offset=3, nonunit_scales=True, poison_tail=False)
    # Eight logical pages reuse just three physical pages, including aliases
    # across requests. Physical cache capacity must not be treated as KV length.
    case.k_pages = case.k_pages[:3]
    case.v_pages = case.v_pages[:3]
    case.page_order[2:] = [2, 0, 2, 1, 2, 0, 1, 0]
    case.indices.copy_(_i32(case.page_order))
    case.k, case.v = vectorize_kv_cache(case.k_pages, case.v_pages, 2, dq, DV, PAGE)
    assert_case(case, causal, persistent=persistent)


@requires_gfx950
def test_runtime_lengths_change_without_recompilation():
    case = make_case((129,), (320,), heads=4)
    kernel = case.factory(False)
    out = torch.empty(129, 4, DV, dtype=torch.bfloat16, device="cuda")
    for length in (65, 128, 193, 256, 320):
        case.kv_lens = (length,)
        case.indptr.copy_(_i32([0, math.ceil(length / PAGE)]))
        case.last.fill_((length - 1) % PAGE + 1)
        kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
               129, 320, False, case.qs, case.ks, case.vs, case.last, out=out)
        ref, _ = run_torch(case, False)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_lse", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_packed_v_runtime_poisoned_tails_forward_and_reverse(dq, causal, with_lse, persistent):
    # Causal chooses MERGE (32 Q blocks * 16 heads == 512). Its mirror
    # consumes the physical tail in the first phase, not in the epilogue.
    q_len = 7937 if causal else 257
    case = make_case((q_len,), (321,), dq=dq, poison_tail=False)
    kernel = case.factory(causal, persistent)
    source_k, source_v = case.k_pages.clone(), case.v_pages.clone()
    out = torch.empty(q_len, 16, DV, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(q_len, 16, device="cuda") if with_lse else None
    compiled_count = None
    for length in (1, 64, 65, 128, 193, 321, 320):
        case.kv_lens = (length,)
        count = math.ceil(length / PAGE)
        case.indptr.copy_(_i32([0, count]))
        case.last.fill_((length - 1) % PAGE + 1)
        case.k_pages.copy_(source_k)
        case.v_pages.copy_(source_v)
        if length % PAGE:
            physical_tail = case.page_order[count - 1]
            case.k_pages[physical_tail, length % PAGE:] = float("nan")
            case.v_pages[physical_tail, length % PAGE:] = float("nan")
        k, v = vectorize_kv_cache(case.k_pages, case.v_pages, 1, dq, DV, PAGE)
        case.k.copy_(k)
        case.v.copy_(v)
        first, first_lse = None, None
        for _ in range(3):
            kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
                   q_len, 321, causal, case.qs, case.ks, case.vs, case.last, out=out, lse=lse)
            if first is None:
                first = out.clone()
                first_lse = lse.clone() if with_lse else None
            else:
                torch.testing.assert_close(out, first, rtol=0, atol=0)
                if with_lse:
                    torch.testing.assert_close(lse, first_lse, rtol=0, atol=0)
        reference, ref_lse = run_torch(case, causal)
        torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
        if with_lse:
            torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)
        if compiled_count is None:
            compiled_count = len(kernel._compiled)
        assert len(kernel._compiled) == compiled_count


@requires_gfx950
@pytest.mark.parametrize("persistent", [False, True])
def test_empty_queries_and_empty_keys(persistent):
    case = make_case((0,), (0,), heads=2)
    out, lse = case.run(False, return_lse=True, persistent=persistent)
    assert out.shape == (0, 2, DV) and lse.shape == (0, 2)
    assert_case(make_case((65,), (0,), heads=2), False, persistent=persistent)


@requires_gfx950
@pytest.mark.parametrize("dq,window_left,has_sink", [(192, -1, False), (128, 128, True), (192, 0, False)])
def test_custom_stream_graph_and_output_allocation(dq, window_left, has_sink):
    case = make_case((127,), (193,), heads=4, dq=dq, window_left=window_left, has_sink=has_sink)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out, lse = case.run(True, return_lse=True, stream=stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            case.run(True, out, lse=lse, stream=stream)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    ref, ref_lse = run_torch(case, True)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)


@requires_gfx950
def test_target_shape_accuracy():
    assert_case(make_case((10240,), (2583,)), False)


@requires_gfx950
@pytest.mark.parametrize("q_len", [7937, 8193])
@pytest.mark.parametrize("dq,has_sink", [(192, False), (128, False), (128, True), (192, True)])
@pytest.mark.parametrize("persistent", [False, True])
def test_causal_head_tail_merge(q_len, dq, has_sink, persistent):
    assert_case(make_case((q_len,), (q_len + 7,), dq=dq, has_sink=has_sink), True, persistent=persistent)


@requires_gfx950
@pytest.mark.parametrize("persistent", [False, True])
def test_causal_merge_ragged_empty_and_all_masked(persistent):
    # Max grid is large enough to select paired blocks, while each sequence
    # still has its own odd/even query-block count and KV extent.
    case = make_case((33, 4097, 65), (0, 193, 129), q_offset=3, table_offset=2)
    assert_case(case, True, persistent=persistent)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window_left", [0, 1, 63, 64, 65, 127, 128, 129, 512])
@pytest.mark.parametrize("has_sink", [False, True])
def test_sliding_window_boundaries(dq, window_left, has_sink):
    case = make_case((257,), (777,), heads=4, dq=dq, window_left=window_left, has_sink=has_sink)
    assert_case(case, True)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("window_left", [-1, 0, 128])
@pytest.mark.parametrize("sink_value", [-80.0, 0.0, 80.0])
def test_sink_denominator_and_empty_rows(dq, window_left, sink_value):
    case = make_case((33, 259), (0, 65), heads=4, dq=dq, window_left=window_left,
                     has_sink=True, q_offset=5, table_offset=2, nonunit_scales=True)
    case.sinks.fill_(sink_value)
    assert_case(case, True)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal", [False, True])
def test_full_attention_sink(dq, causal):
    case = make_case((129,), (321,), heads=4, kv_heads=2, dq=dq, has_sink=True, magnitude=4.0)
    assert_case(case, causal)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("layout", ["padded", "head-major"])
def test_swa_ragged_strides_and_nonunit_scales(dq, layout):
    case = make_case((0, 7, 129, 259), (63, 0, 193, 901), heads=6, kv_heads=2,
                     dq=dq, window_left=128, has_sink=True, nonunit_scales=True,
                     layout=layout, q_offset=5, table_offset=3)
    assert_case(case, True, layout=layout)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("persistent", [False, True])
def test_direct_swa_does_not_read_excluded_prefix(dq, persistent):
    case = make_case((257,), (8193,), heads=4, dq=dq, window_left=128, has_sink=True)
    first_page = (8193 - 257 - 128) // PAGE
    # Excluded page-table entries are intentionally invalid, not just zeros.
    case.indices[:first_page] = 2**30
    assert_case(case, True, repeats=10, persistent=persistent)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("persistent", [False, True])
def test_swa_runtime_lengths_and_sinks_are_not_cached(dq, persistent):
    case = make_case((65,), (2049,), heads=4, dq=dq, mode="per-tensor", window_left=128, has_sink=True,
                     nonunit_scales=True)
    kernel = case.factory(True, persistent)
    out = torch.empty(65, 4, DV, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(65, 4, device="cuda")
    for length, sink in ((2049, 0.0), (257, -80.0), (128, 80.0), (0, -float("inf")), (193, 1.0)):
        case.kv_lens = (length,)
        case.indptr.copy_(_i32([0, math.ceil(length / PAGE)]))
        case.last.fill_((length - 1) % PAGE + 1 if length else 0)
        case.sinks.fill_(sink)
        kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
               65, 2049, True, case.qs, case.ks, case.vs, case.last, out=out,
               lse=lse, sink_ptr=case.sinks)
        ref, ref_lse = run_torch(case, True)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("persistent", [False, True])
def test_swa_zero_logits_has_exact_inclusive_window_and_sink(dq, persistent):
    case = make_case((129,), (257,), heads=2, dq=dq, window_left=128, has_sink=True)
    case.q.zero_()
    case.k_pages.zero_()
    case.v_pages.fill_(1.0)
    case.sinks.zero_()
    case.k, case.v = vectorize_kv_cache(case.k_pages, case.v_pages, 1, dq, DV, PAGE)
    out, lse = assert_case(case, True, persistent=persistent)
    # 129 visible keys plus one zero-value sink: no off-by-one or double sink.
    torch.testing.assert_close(out, torch.full_like(out, 129 / 130), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(130)), rtol=1e-6, atol=1e-6)
    case.sinks.fill_(-float("inf"))
    out, _ = assert_case(case, True, persistent=persistent)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@requires_gfx950
def test_sink_buffer_contract():
    case = make_case((9,), (65,), heads=4, dq=128, window_left=128, has_sink=True)
    for invalid in (None, torch.zeros(3, device="cuda"), torch.zeros(4, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(8, device="cuda")[::2], torch.zeros(4)):
        with pytest.raises(ValueError, match="sink_ptr"):
            case.run(True, sink_ptr=invalid)
    no_sink = make_case((9,), (65,), heads=4)
    with pytest.raises(ValueError, match="has_sink"):
        no_sink.run(False, sink_ptr=case.sinks)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
def test_direct_only_no_workspace_single_launch(dq, monkeypatch):
    case = make_case((257,), (901,), dq=dq, window_left=128, has_sink=True)
    out = torch.empty(257, 16, DV, device="cuda", dtype=torch.bfloat16)
    kernel = case.factory(True)
    case.run(True, out)
    torch.cuda.synchronize()
    assert not hasattr(kernel, "prepare_kv")
    assert not hasattr(kernel, "attend_linear")
    assert not hasattr(kernel, "_workspace")

    def no_allocation(*args, **kwargs):
        pytest.fail("direct-paged dispatch must not allocate a KV workspace")

    before = torch.cuda.memory_allocated()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", no_allocation)
        patch.setattr(torch, "empty_like", no_allocation)
        case.run(True, out)
        torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        case.run(True, out)
        torch.cuda.synchronize()
    gpu_events = [event for event in prof.events() if "CUDA" in str(event.device_type)]
    assert len(gpu_events) == 1, [event.name for event in gpu_events]
    assert "_attention_kernel" in gpu_events[0].name
    ref, _ = run_torch(case, True)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal,window,has_sink", [(False, -1, False), (True, -1, True),
                                                  (True, 0, False), (True, 128, True)])
@pytest.mark.parametrize("layout", ["padded", "head-major"])
def test_persistent_ragged_gqa_strides_and_empty_requests(dq, causal, window, has_sink, layout):
    case = make_case((0, 7, 1025, 0, 513), (63, 0, 129, 0, 901), heads=6, kv_heads=2,
                     dq=dq, q_offset=5, table_offset=3, layout=layout, nonunit_scales=True,
                     window_left=window, has_sink=has_sink)
    assert_case(case, causal, persistent=True, layout=layout, repeats=5, softmax_scale=0.0625)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal,window", [(False, -1), (True, -1), (True, 128)])
def test_persistent_more_tasks_than_cus_and_counter_reuse(dq, causal, window, monkeypatch):
    # 48 query blocks * 16 heads is three times the persistent grid; causal
    # pairing still leaves 384 tasks. Every CTA must fetch work after task 0.
    case = make_case((12289,), (901,), dq=dq, window_left=window, has_sink=window >= 0)
    kernel = case.factory(causal, True)
    out = torch.empty(12289, 16, DV, device="cuda", dtype=torch.bfloat16)
    expected = case.run(causal).clone()
    first = case.run(causal, out, persistent=True).clone()
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    for _ in range(12):
        case.run(causal, out, persistent=True)
        torch.testing.assert_close(out, first, rtol=0, atol=0)
    counters = list(kernel._scheduler_counters.items())
    for (_, _, grid), counter in counters:
        assert counter.tolist() == [grid, 0]
    before = torch.cuda.memory_allocated()
    def no_allocation(*args, **kwargs):
        pytest.fail("warmed persistent call must not allocate or reset a tensor")
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "tensor"):
            patch.setattr(torch, name, no_allocation)
        case.run(causal, out, persistent=True)
        torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert _gpu_dispatch(lambda: case.run(causal, out, persistent=True)) == ["_attention_persistent_kernel_0"]
    ref, _ = run_torch(case, causal)
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
def test_persistent_runtime_query_mapping_changes_and_empty_grid(dq):
    case = make_case((257, 513, 257), (129, 321, 193), heads=6, kv_heads=2, dq=dq,
                     q_offset=3, table_offset=2, poison_tail=False)
    kernel = case.factory(False, True)
    out = torch.empty(case.q.shape[0], 6, DV, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(case.q.shape[:2], device="cuda")
    compiled_count = None
    for lengths in ((257, 513, 257), (0, 1, 1026), (1027, 0, 0), (0, 0, 0), (513, 257, 257)):
        case.q_lens = lengths
        case.cq.copy_(_i32(list(accumulate(lengths, initial=3))))
        out.fill_(-123)
        lse.fill_(-123)
        # Keep host bounds/storage fixed while device metadata changes.
        kernel(case.q, case.k, case.v, case.cq, None, case.indptr, case.indices,
               1027, 321, False, case.qs, case.ks, case.vs, case.last, out=out, lse=lse)
        begin, end = 3, 3 + sum(lengths)
        ref, ref_lse = run_torch(case, False)
        torch.testing.assert_close(out[begin:end].float(), ref[begin:end], rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse[begin:end], ref_lse[begin:end], rtol=2e-4, atol=5e-4)
        assert (out[:begin] == -123).all() and (out[end:] == -123).all()
        for (_, _, grid), counter in kernel._scheduler_counters.items():
            assert counter.tolist() == [grid, 0]
        if compiled_count is None:
            compiled_count = len(kernel._compiled)
        assert len(kernel._compiled) == compiled_count


@requires_gfx950
@pytest.mark.parametrize("dq,window", [(128, -1), (192, 128)])
def test_persistent_stream_isolation_and_repeated_graphs(dq, window):
    case = make_case((4097,), (777,), dq=dq, window_left=window, has_sink=window >= 0)
    kernel = case.factory(True, True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [torch.empty(4097, 16, DV, device="cuda", dtype=torch.bfloat16) for _ in streams]
    lses = [torch.empty(4097, 16, device="cuda") for _ in streams]
    graphs = []
    for stream, out, lse in zip(streams, outputs, lses):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.run(True, out, persistent=True, lse=lse, stream=stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                case.run(True, out, persistent=True, lse=lse, stream=stream)
                case.run(True, out, persistent=True, lse=lse, stream=stream)
            graphs.append(graph)
    # Distinct stream-local counters permit overlapping launches/replays.
    for _ in range(8):
        for graph, stream in zip(graphs, streams):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    ref, ref_lse = run_torch(case, True)
    for out, lse in zip(outputs, lses):
        torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=5e-4)
    active = [counter for (_, sid, _), counter in kernel._scheduler_counters.items()
              if sid in {stream.cuda_stream for stream in streams}]
    assert len(active) == 2 and active[0].data_ptr() != active[1].data_ptr()
    for (_, _, grid), counter in kernel._scheduler_counters.items():
        assert counter.tolist() == [grid, 0]


@requires_gfx950
@pytest.mark.parametrize("dq,window", [(128, 64), (128, 65), (192, 128), (192, 129)])
@pytest.mark.parametrize("q_len", [16128, 16129, 16385])
@pytest.mark.parametrize("persistent", [False, True])
def test_swa_pruning_dispatch_thresholds(dq, window, q_len, persistent):
    # 63/64/65 Q blocks straddle the 1024-task gate. KV<Q exercises fully
    # masked waves, non-page-aligned diagonals, partial V and actual Q tails.
    case = make_case((q_len,), (901,), dq=dq, window_left=window,
                     has_sink=True, nonunit_scales=True, magnitude=4.0)
    assert_case(case, True, persistent=persistent, softmax_scale=0.0625)


@requires_gfx950
@pytest.mark.parametrize("dq,window", [(128, 1), (192, 128)])
@pytest.mark.parametrize("has_sink", [False, True])
def test_swa_pruned_large_visible_prefix_and_exact_zero_logits(dq, window, has_sink):
    case = make_case((16384,), (32769,), dq=dq, window_left=window, has_sink=has_sink)
    case.q.zero_()
    case.k_pages.zero_()
    case.v_pages.fill_(1.0)
    case.k, case.v = vectorize_kv_cache(case.k_pages, case.v_pages, 1, dq, DV, PAGE)
    first_page = (32769 - 16384 - window) // PAGE
    case.indices[:first_page] = 2**30
    if has_sink:
        case.sinks.zero_()
    for persistent in (False, True):
        out, lse = assert_case(case, True, persistent=persistent)
        expected = (window + 1) / (window + 2) if has_sink else 1.0
        torch.testing.assert_close(out, torch.full_like(out, expected), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.full_like(lse, math.log(window + 1 + int(has_sink))), rtol=1e-6, atol=1e-6)


def test_supported_scope_is_explicit():
    assert PagedAttention(16, 1, DQ, DV, PAGE, False).bf16_backend == "native-8wave"
    assert PagedAttention(16, 1, 128, DV, PAGE, False).bf16_backend == "native-8wave"
    assert not PagedAttention(16, 1, DQ, DV, PAGE, False).persistent
    assert PagedAttention(16, 1, DQ, DV, PAGE, False, persistent=True).persistent
    with pytest.raises(ValueError, match="persistent"):
        PagedAttention(16, 1, DQ, DV, PAGE, False, persistent="yes")
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, 64, DV, PAGE, False)
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, DQ, DV, 32, False)
    with pytest.raises(ValueError):
        PagedAttention(16, 1, DQ, DV, PAGE, False, window_left=128)
    with pytest.raises(ValueError):
        PagedAttention(16, 1, DQ, DV, PAGE, True, window_left=-2)
    with pytest.raises(ValueError):
        PagedAttention(7, 2, DQ, DV, PAGE, False)


@requires_gfx950
def test_invalid_runtime_buffers_fail_before_launch():
    case = make_case((9,), (65,), heads=4)
    with pytest.raises(ValueError, match="output"):
        case.run(False, torch.empty(9, 4, DV, device="cuda", dtype=torch.float32))
    with pytest.raises(ValueError, match="softmax_scale"):
        case.run(False, softmax_scale=0.0)
    with pytest.raises(ValueError, match="LSE"):
        case.run(False, lse=torch.empty(9, 4, device="cuda", dtype=torch.bfloat16))
    case.q = case.q.float()
    with pytest.raises(NotImplementedError, match="BF16"):
        case.run(False)


def _load_4wave():
    name = "pa_prefill_4wave_comparison"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "pa_4wave" / "pa_prefill_4wave.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].MHA


def run_aiter_5d(case, causal, out):
    """Pass the same SHUFFLE-5D cache and page table, without repacking."""
    from aiter import mha_batch_prefill_func

    return mha_batch_prefill_func(
        case.q, case.k, case.v, case.cq, case.indptr, case.indices,
        max(case.q_lens), max(case.kv_lens), causal=causal,
        window_size=(case.window_left, -1), sink_ptr=case.sinks,
        kv_last_page_lens=case.last, out=out,
    )


def _probe_aiter_5d(fn):
    """Only a missing specialization is unavailable; all other failures raise."""
    try:
        result = fn()
        torch.cuda.synchronize()
    except RuntimeError as error:
        if "no matching kernel found" not in str(error):
            raise
        return None, str(error)
    return result, None


def _make_opus_call(case, causal, k_linear, v_linear, out):
    """Select OPUS explicitly, never the public router's ASM/CK fallback."""
    from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, fmha_fwd_bf16_opus_varlen_fwd

    if case.window_left >= 0 or case.sinks is not None:
        raise ValueError("OPUS D128/D192 has no SWA or sink support")
    if case.dq == 128:
        if len(case.q_lens) != 1 or case.q_offset:
            raise ValueError("OPUS D128 comparison requires one dense sequence")
        q, k, v, output = (x.unsqueeze(0) for x in (case.q, k_linear, v_linear, out))

        def call():
            fmha_fwd_bf16_opus_fwd(q, k, v, case.dq**-0.5, causal, out=output)
            return out
    else:
        def call():
            fmha_fwd_bf16_opus_varlen_fwd(
                case.q, k_linear, v_linear, case.dq**-0.5, causal,
                case.cq, case.ck, max(case.q_lens), max(case.kv_lens), out=out,
            )
            return out
    return call


def _gpu_dispatch(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({event.name for event in prof.events() if "CUDA" in str(event.device_type)})


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal,window_left,has_sink", [(False, -1, False), (True, -1, False), (True, 128, True)])
def test_aiter_5d_comparison(dq, causal, window_left, has_sink):
    case = make_case((257,), (777,), dq=dq, window_left=window_left, has_sink=has_sink, poison_tail=False)
    out = torch.empty(257, 16, DV, device="cuda", dtype=torch.bfloat16)
    actual, reason = _probe_aiter_5d(lambda: run_aiter_5d(case, causal, out))
    if reason is not None:
        pytest.skip(reason)
    assert actual.data_ptr() == out.data_ptr()
    reference, _ = run_torch(case, causal)
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)


@requires_gfx950
@pytest.mark.parametrize("dq", [128, 192])
@pytest.mark.parametrize("causal", [False, True])
def test_explicit_opus_comparison(dq, causal):
    case = make_case((257,), (777,), dq=dq, poison_tail=False)
    keys, values = case.logical_kv()
    out = torch.empty(257, 16, DV, device="cuda", dtype=torch.bfloat16)
    fn = _make_opus_call(case, causal, keys[0], values[0], out)
    assert fn() is out
    torch.cuda.synchronize()
    reference, _ = run_torch(case, causal)
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
    dispatch = _gpu_dispatch(fn)
    assert any(f"gqa_d{dq}" in name for name in dispatch), dispatch


def test_5d_probe_does_not_swallow_other_failures():
    def missing():
        raise RuntimeError("invalid argument: no matching kernel found")

    def broken():
        raise RuntimeError("unexpected kernel failure")

    assert _probe_aiter_5d(missing)[1] is not None
    with pytest.raises(RuntimeError, match="unexpected kernel failure"):
        _probe_aiter_5d(broken)


@requires_gfx950
@pytest.mark.parametrize("window_left,has_sink", [(128, False), (-1, True), (128, True)])
def test_opus_comparison_rejects_unsupported_semantics(window_left, has_sink):
    case = make_case((9,), (65,), window_left=window_left, has_sink=has_sink)
    keys, values = case.logical_kv()
    out = torch.empty(9, 16, DV, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="no SWA or sink"):
        _make_opus_call(case, True, keys[0], values[0], out)


@benchmark()
def benchmark_prefill(q_len, kv_len, heads, kv_heads, causal, dq=DQ, window_left=-1, has_sink=False):
    import aiter

    # Legacy 4-wave multiplies masked probabilities by padded V without first
    # zeroing V (0*NaN is NaN). Use equal zero-padded caches for performance;
    # the new kernel's correctness tests still poison tails with NaNs.
    case = make_case((q_len,), (kv_len,), heads=heads, kv_heads=kv_heads,
                     dq=dq, window_left=window_left, has_sink=has_sink, poison_tail=False)
    keys, values = case.logical_kv()
    k_linear, v_linear = keys[0], values[0]
    out = torch.empty(q_len, heads, DV, dtype=torch.bfloat16, device="cuda")
    aiter_out = torch.empty_like(out)
    ref, _ = run_torch(case, causal)

    def aiter_call(k, v):
        return aiter.flash_attn_varlen_func(
            case.q, k, v, case.cq, case.ck, q_len, kv_len,
            causal=causal, window_size=(window_left, -1, 0), sink_ptr=case.sinks, out=aiter_out,
        )

    candidates = {
        "flydsl_5d": lambda: case.run(causal, out),
        "aiter_linear": lambda: aiter_call(k_linear, v_linear),
    }
    unavailable = {}
    if has_sink and not causal:
        unavailable["aiter_5d"] = "AITER batch-prefill ignores a noncausal full-attention sink"
    else:
        five_d_out = torch.empty_like(out)
        candidates["aiter_5d"] = lambda: run_aiter_5d(case, causal, five_d_out)
    if window_left < 0 and not has_sink:
        candidates["opus_linear"] = _make_opus_call(case, causal, k_linear, v_linear, torch.empty_like(out))
    else:
        unavailable["opus_linear"] = "OPUS D128/D192 has no SWA or sink support"
    # The existing 4-wave API supports only SWA+sink together, or neither.
    if (window_left >= 0) == has_sink and (not causal or kv_len >= q_len):
        factory = _load_4wave()
        for dynamic in (False, True):
            four = factory(heads, kv_heads, dq, DV, PAGE, causal, window_left=window_left,
                           has_sink=has_sink, force_dynamic_schedule=dynamic)
            four_out = torch.empty_like(out)

            def run_four(four=four, four_out=four_out):
                return four(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                            q_len, kv_len, causal, case.qs, case.ks, case.vs, case.last,
                            four_out, sink_ptr=case.sinks)

            candidates["4wave_dynamic_5d" if dynamic else "4wave_static_5d"] = run_four
    first_page = max(0, (kv_len - q_len - window_left) // PAGE) if window_left >= 0 else 0
    # checkAllclose returns a ratio, not an assertion: enforce zero mismatches.
    errors = {}
    for name, fn in list(candidates.items()):
        if name == "aiter_5d":
            result, reason = _probe_aiter_5d(fn)
            if reason is not None:
                unavailable[name] = reason
                del candidates[name]
                continue
        else:
            result = fn()
            torch.cuda.synchronize()
        errors[name] = checkAllclose(ref, result.float(), rtol=2e-2, atol=2e-2, tol_err_ratio=0, msg=name)
        assert errors[name] == 0, f"{name} failed numerical validation"

    dispatches = {}
    for name in ("flydsl_5d", "aiter_5d", "opus_linear", "aiter_linear"):
        if name in candidates:
            dispatches[name] = _gpu_dispatch(candidates[name])
            aiter.logger.info("%s dispatch: %s", name, dispatches[name])
    if "opus_linear" in candidates:
        assert any(f"gqa_d{dq}" in name for name in dispatches["opus_linear"]), dispatches["opus_linear"]
    for name, reason in unavailable.items():
        aiter.logger.warning("%s unavailable (no timing): %s", name, reason)

    for _ in range(100):
        for fn in candidates.values():
            fn()
    torch.cuda.synchronize()

    times = {name: [] for name in candidates}
    for trial in range(5):
        names = list(candidates) if trial % 2 == 0 else list(reversed(candidates))
        for name in names:
            _, us = run_perftest(candidates[name], num_iters=100, num_warmup=20, num_rotate_args=1)
            times[name].append(us)
    pairs = q_len * kv_len
    if causal:
        pairs = sum(max(0, min(kv_len, kv_len - q_len + row + 1)
                        - (max(0, kv_len - q_len + row - window_left) if window_left >= 0 else 0))
                    for row in range(q_len))
    flops = 2 * heads * pairs * (dq + DV)
    attn_bytes = 2 * (q_len * heads * (dq + DV) + (kv_len - first_page * PAGE) * kv_heads * (dq + DV))
    ret = {"gfx": GPU_ARCH.split(":")[0]}
    for name, samples in times.items():
        us = float(statistics.median(samples))
        ret.update({
            f"{name} us": us,
            f"{name} TFLOPS": flops / us / 1e6,
            f"{name} TB/s": attn_bytes / us / 1e6,
            f"{name} err": errors[name],
        })
    for name in ("aiter_5d", "opus_linear"):
        if name in unavailable:
            for metric in ("us", "TFLOPS", "TB/s", "err"):
                ret[f"{name} {metric}"] = float("nan")
        ret[f"{name} status"] = "unavailable" if name in unavailable else "measured"
    observation = {
        "q": q_len, "kv": kv_len, "heads": heads, "kv_heads": kv_heads, "dq": dq,
        "causal": causal, "window_left": window_left, "has_sink": has_sink,
        "timing_us": {name: [float(x) for x in samples] for name, samples in times.items()},
        "median_us": {name: float(statistics.median(samples)) for name, samples in times.items()},
        "errors": {name: float(err) for name, err in errors.items()},
        "unavailable": unavailable, "dispatch": dispatches,
        "layout_note": "*_5d candidates share the exact paged cache; *_linear use identical logical KV prepared before timing",
    }
    aiter.logger.info("5D_OPUS_RESULT %s", json.dumps(observation, ensure_ascii=False, allow_nan=False))
    return ret


def main():
    import aiter
    import pandas as pd

    if "gfx950" not in GPU_ARCH:
        aiter.logger.warning("gfx950 BF16 attention benchmark skipped on %s", GPU_ARCH)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, nargs="+", default=[10240])
    parser.add_argument("--kv-len", type=int, nargs="+", default=[2583])
    parser.add_argument("--heads", type=int, nargs="+", default=[16])
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[1])
    parser.add_argument("--causal", type=int, choices=[0, 1], nargs="+", default=[0])
    parser.add_argument("--head-dim", type=int, choices=[128, 192], nargs="+", default=[192])
    parser.add_argument("--window-left", type=int, nargs="+", default=[-1])
    parser.add_argument("--sink", type=int, choices=[0, 1], nargs="+", default=[0])
    args = parser.parse_args()
    rows = [benchmark_prefill(q, k, h, hk, bool(c), d, w, bool(s)) for q, k, h, hk, c, d, w, s in itertools.product(
        args.q_len, args.kv_len, args.heads, args.kv_heads, args.causal, args.head_dim, args.window_left, args.sink
    )]
    aiter.logger.info("Prefill median GPU timing (*_5d = identical page64 cache; OPUS explicit linear, no conversion timed; unavailable = no result):\n%s",
                      pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()