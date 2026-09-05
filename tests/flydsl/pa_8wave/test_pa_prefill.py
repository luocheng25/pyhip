"""Strict tests and performance comparisons for gfx950 BF16 D192/V128 prefill.

Only the supported OPUS-style specialization is exercised here. The torch
reference is independent of the gather and attention kernels. No accuracy
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

    @property
    def heads(self):
        return self.q.shape[1]

    @property
    def kv_heads(self):
        return self.k_pages.shape[2]

    def factory(self, causal):
        return PagedAttention(self.heads, self.kv_heads, DQ, DV, PAGE, causal, self.mode)

    def run(self, causal, out=None, **kwargs):
        return self.factory(causal)(
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
            keys.append(self.k_pages[ids].reshape(-1, self.kv_heads, DQ)[:length])
            values.append(self.v_pages[ids].reshape(-1, self.kv_heads, DV)[:length])
            pos += count
        return keys, values


def make_case(q_lens=(256,), kv_lens=(256,), *, heads=16, kv_heads=1,
              mode="per-token", layout="contiguous", q_offset=0, table_offset=0,
              nonunit_scales=False, magnitude=1.0, seed=20260905):
    assert len(q_lens) == len(kv_lens)
    torch.manual_seed(seed)
    q_lens, kv_lens = tuple(q_lens), tuple(kv_lens)
    tokens = q_offset + sum(q_lens) + (7 if q_offset else 0)
    if layout == "head-major":
        q = torch.randn(heads, tokens, DQ, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    elif layout == "padded":
        q = torch.randn(tokens, heads + 1, DQ + 16, dtype=torch.bfloat16, device="cuda")[:, :heads, :DQ]
    else:
        q = torch.randn(tokens, heads, DQ, dtype=torch.bfloat16, device="cuda")
    q.mul_(magnitude)
    counts = [(n + PAGE - 1) // PAGE for n in kv_lens]
    pages = max(1, sum(counts))
    k_pages = torch.randn(pages, PAGE, kv_heads, DQ, dtype=torch.bfloat16, device="cuda") * magnitude
    v_pages = torch.randn(pages, PAGE, kv_heads, DV, dtype=torch.bfloat16, device="cuda")
    order = [0] * table_offset + torch.randperm(sum(counts)).tolist()
    start = table_offset
    for length, count in zip(kv_lens, counts):
        if length % PAGE:
            # A bounds bug must not pass just because padded KV happens to be 0.
            k_pages[order[start + count - 1], length % PAGE:] = float("nan")
            v_pages[order[start + count - 1], length % PAGE:] = float("nan")
        start += count
    k, v = vectorize_kv_cache(k_pages, v_pages, kv_heads, DQ, DV, PAGE)
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
        qs, ks, vs, q_lens, kv_lens, order, q_offset, table_offset, mode,
    )


def run_torch(case, causal, softmax_scale=None):
    """FP32 reference with bottom-right causal alignment and empty-row semantics."""
    keys, values = case.logical_kv()
    output = torch.full((case.q.shape[0], case.heads, DV), float("nan"), device=case.q.device)
    lse = torch.full((case.q.shape[0], case.heads), float("nan"), device=case.q.device)
    scale = 1 / math.sqrt(DQ) if softmax_scale is None else softmax_scale
    offset = case.q_offset
    for q_len, kv_len, k, v in zip(case.q_lens, case.kv_lens, keys, values):
        k = (k.float() * case.ks).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        v = (v.float() * case.vs).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        for start in range(0, q_len, 256):
            end = min(q_len, start + 256)
            if kv_len == 0:
                output[offset + start:offset + end] = 0
                lse[offset + start:offset + end] = -float("inf")
                continue
            q = case.q[offset + start:offset + end].float()
            qs = case.qs if case.qs.numel() == 1 else case.qs[offset + start:offset + end]
            logits = ((q * qs).transpose(0, 1) @ k.transpose(1, 2)) * scale
            if causal:
                rows = torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(kv_len, device=q.device)[None, :]
                logits.masked_fill_(cols > kv_len - q_len + rows, -float("inf"))
            probs = torch.softmax(logits, dim=-1).nan_to_num(0.0)
            output[offset + start:offset + end] = (probs @ v).transpose(0, 1)
            lse[offset + start:offset + end] = torch.logsumexp(logits, dim=-1).transpose(0, 1)
        offset += q_len
    return output, lse


def assert_case(case, causal, *, repeats=3, layout="contiguous", softmax_scale=None):
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
        result, result_lse = case.run(causal, out, return_lse=True, lse=lse, softmax_scale=softmax_scale)
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
def test_page_parity_and_tails(kv_len, causal):
    assert_case(make_case((257,), (kv_len,), heads=4), causal)


@requires_gfx950
def test_native_bf16_direct_lds_subgroup_rings_are_deterministic():
    assert_case(make_case((256,), (256,)), False, repeats=10)


@requires_gfx950
@pytest.mark.parametrize("kv_len", [256, 321])
def test_default_no_lse_path_is_deterministic(kv_len):
    case = make_case((257,), (kv_len,))
    out = torch.full((257, 16, DV), float("nan"), device="cuda", dtype=torch.bfloat16)
    first = case.run(False, out).clone()
    for _ in range(10):
        assert case.run(False, out) is out
        torch.testing.assert_close(out, first, rtol=0, atol=0)
    ref, _ = run_torch(case, False)
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
def test_gather_is_exact_and_reads_current_pages():
    case = make_case((17, 23), (129, 256), heads=6, kv_heads=2, table_offset=2)
    kernel = case.factory(False)
    for factor in (1.0, 2.0):
        case.k_pages.mul_(factor)
        case.v_pages.mul_(factor)
        k, v = vectorize_kv_cache(case.k_pages, case.v_pages, 2, DQ, DV, PAGE)
        case.k.copy_(k)
        case.v.copy_(v)
        kl, vl = kernel.prepare_kv(case.k, case.v, case.indptr, case.indices, max(case.kv_lens))
        keys, values = case.logical_kv()
        for b, n in enumerate(case.kv_lens):
            torch.testing.assert_close(kl[b, :n], keys[b], rtol=0, atol=0)
            torch.testing.assert_close(vl[b, :n], values[b], rtol=0, atol=0)
        assert_case(case, False)


@requires_gfx950
def test_page_table_changes_between_calls():
    case = make_case((129,), (320,), heads=4, table_offset=2)
    assert_case(case, False)
    case.page_order[2:] = list(reversed(case.page_order[2:]))
    case.indices.copy_(_i32(case.page_order))
    assert_case(case, False)


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
def test_empty_queries_and_empty_keys():
    case = make_case((0,), (0,), heads=2)
    out, lse = case.run(False, return_lse=True)
    assert out.shape == (0, 2, DV) and lse.shape == (0, 2)
    assert_case(make_case((65,), (0,), heads=2), False)


@requires_gfx950
def test_custom_stream_graph_and_output_allocation():
    case = make_case((127,), (193,), heads=4)
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
def test_causal_head_tail_merge(q_len):
    assert_case(make_case((q_len,), (q_len + 7,)), True)


@requires_gfx950
def test_causal_merge_ragged_empty_and_all_masked():
    # Max grid is large enough to select paired blocks, while each sequence
    # still has its own odd/even query-block count and KV extent.
    case = make_case((33, 4097, 65), (0, 193, 129), q_offset=3, table_offset=2)
    assert_case(case, True)


def test_supported_scope_is_explicit():
    assert PagedAttention(16, 1, DQ, DV, PAGE, False).bf16_backend == "native-8wave"
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, 128, DV, PAGE, False)
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, DQ, DV, 32, False)
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, DQ, DV, PAGE, True, window_left=128)
    with pytest.raises(NotImplementedError):
        PagedAttention(16, 1, DQ, DV, PAGE, True, has_sink=True)
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


@benchmark()
def benchmark_prefill(q_len, kv_len, heads, kv_heads, causal):
    import aiter

    case = make_case((q_len,), (kv_len,), heads=heads, kv_heads=kv_heads)
    kernel = case.factory(causal)
    kl, vl = kernel.prepare_kv(case.k, case.v, case.indptr, case.indices, kv_len)
    keys, values = case.logical_kv()
    k_linear, v_linear = keys[0], values[0]
    out = torch.empty(q_len, heads, DV, dtype=torch.bfloat16, device="cuda")
    opus_out = torch.empty_like(out)
    ref, _ = run_torch(case, causal)
    candidates = {
        "flydsl_full": lambda: case.run(causal, out),
        "flydsl_core": lambda: kernel.attend_linear(
            case.q, kl, vl, case.cq, case.ck, case.indptr, case.last,
            case.qs, case.ks, case.vs, q_len, out,
        ),
        "opus_linear": lambda: aiter.flash_attn_varlen_func(
            case.q, k_linear, v_linear, case.cq, case.ck, q_len, kv_len,
            causal=causal, out=opus_out,
        ),
        "gather": lambda: kernel.prepare_kv(case.k, case.v, case.indptr, case.indices, kv_len),
    }
    # checkAllclose returns a ratio, not an assertion: enforce zero mismatches.
    errors = {}
    for name, fn in candidates.items():
        result = fn()
        if name == "gather":
            torch.testing.assert_close(result[0][0, :kv_len], k_linear, rtol=0, atol=0)
            torch.testing.assert_close(result[1][0, :kv_len], v_linear, rtol=0, atol=0)
            errors[name] = 0.0
        else:
            errors[name] = checkAllclose(ref, result.float(), rtol=2e-2, atol=2e-2, tol_err_ratio=0, msg=name)
            assert errors[name] == 0, f"{name} failed numerical validation"

    times = {name: [] for name in candidates}
    for trial in range(5):
        names = list(candidates) if trial % 2 == 0 else list(reversed(candidates))
        for name in names:
            _, us = run_perftest(candidates[name], num_iters=100, num_warmup=20, num_rotate_args=1)
            times[name].append(us)
    pairs = q_len * kv_len
    if causal:
        pairs = sum(max(0, min(kv_len, kv_len - q_len + row + 1)) for row in range(q_len))
    flops = 2 * heads * pairs * (DQ + DV)
    attn_bytes = 2 * (q_len * heads * (DQ + DV) + kv_len * kv_heads * (DQ + DV))
    gather_bytes = 4 * math.ceil(kv_len / PAGE) * PAGE * kv_heads * (DQ + DV)
    ret = {"gfx": GPU_ARCH.split(":")[0]}
    for name, samples in times.items():
        us = statistics.median(samples)
        nbytes = gather_bytes if name == "gather" else attn_bytes + (gather_bytes if name == "flydsl_full" else 0)
        ret.update({
            f"{name} us": us,
            f"{name} TFLOPS": (0 if name == "gather" else flops) / us / 1e6,
            f"{name} TB/s": nbytes / us / 1e6,
            f"{name} err": errors[name],
        })
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
    args = parser.parse_args()
    rows = [benchmark_prefill(q, k, h, hk, bool(c)) for q, k, h, hk, c in itertools.product(
        args.q_len, args.kv_len, args.heads, args.kv_heads, args.causal
    )]
    aiter.logger.info("Prefill median GPU timing (full includes gather; core/OPUS exclude it):\n%s",
                      pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()