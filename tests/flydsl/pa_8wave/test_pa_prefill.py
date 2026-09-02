import math
import importlib
import importlib.util
import json
import os
import sys
from itertools import accumulate
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

triton = sys.modules.get("triton")
if getattr(triton, "__version__", None) is None:
    for module_name in list(sys.modules):
        if module_name == "triton" or module_name.startswith("triton."):
            sys.modules.pop(module_name)
    direct_url = json.loads(metadata.distribution("triton").read_text("direct_url.json"))
    source_root = Path(unquote(urlparse(direct_url["url"]).path))
    package_dir = source_root / "python" / "triton"
    package_spec = importlib.util.spec_from_file_location(
        "triton",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    triton = importlib.util.module_from_spec(package_spec)
    sys.modules["triton"] = triton
    package_spec.loader.exec_module(triton)
tl = importlib.import_module("triton.language")

import pyhip
from aiter import mha_batch_prefill_func
from dataclasses import dataclass

from pa_prefill_8w32x32 import PagedAttention

import pytest


GPU_ARCH = (
    torch.cuda.get_device_properties(0).gcnArchName
    if torch.cuda.is_available()
    else ""
)
FP8_DTYPE = (
    torch.float8_e4m3fn if "gfx950" in GPU_ARCH else torch.float8_e4m3fnuz
)


def pertoken_quant(x, scale=None, quant_dtype=FP8_DTYPE):
    x_f32 = x.float()
    if scale is None:
        scale = x_f32.abs().amax(dim=-1, keepdim=True) / torch.finfo(
            quant_dtype
        ).max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x_f32 / scale).to(quant_dtype), scale.float()


def per_tensor_quant(x, scale=None, quant_dtype=FP8_DTYPE):
    x_f32 = x.float()
    if scale is None:
        scale = x_f32.abs().max() / torch.finfo(quant_dtype).max
    return (x_f32 / scale).to(quant_dtype), scale.reshape(1).float()


requires_gfx942 = pytest.mark.skipif(
    "gfx942" not in GPU_ARCH,
    reason="requires gfx942",
)
requires_gfx950 = pytest.mark.skipif(
    "gfx950" not in GPU_ARCH,
    reason="requires gfx950",
)
requires_gfx942_or_gfx950 = pytest.mark.skipif(
    not any(arch in GPU_ARCH for arch in ("gfx942", "gfx950")),
    reason="requires gfx942 or gfx950",
)

H_Q = 16
H_KV = 1
D_QK = 192
D_V = 128
SWA_PAGE_SIZE = 64
SWA_WINDOW_LEFT = 128
SWA_VECTOR_SIZE = 8


@triton.jit
def _gather_shuffle5d_to_linear_kernel(
    k_cache,
    v_cache,
    slot_ids,
    k_linear,
    v_linear,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_ids + token_idx)
    page = slot // PAGE_SIZE
    page_offset = slot % PAGE_SIZE

    qk_dim = tl.arange(0, BLOCK_QK)
    k_offset = (
        page * HEAD_DIM_QK * PAGE_SIZE
        + (qk_dim // VECTOR_SIZE) * PAGE_SIZE * VECTOR_SIZE
        + page_offset * VECTOR_SIZE
        + qk_dim % VECTOR_SIZE
    )
    k_value = tl.load(k_cache + k_offset, mask=qk_dim < HEAD_DIM_QK)
    tl.store(
        k_linear + token_idx * HEAD_DIM_QK + qk_dim,
        k_value,
        mask=qk_dim < HEAD_DIM_QK,
    )

    value_dim = tl.arange(0, BLOCK_V)
    v_offset = (
        page * HEAD_DIM_V * PAGE_SIZE
        + (page_offset // VECTOR_SIZE) * HEAD_DIM_V * VECTOR_SIZE
        + value_dim * VECTOR_SIZE
        + page_offset % VECTOR_SIZE
    )
    v_value = tl.load(v_cache + v_offset, mask=value_dim < HEAD_DIM_V)
    tl.store(
        v_linear + token_idx * HEAD_DIM_V + value_dim,
        v_value,
        mask=value_dim < HEAD_DIM_V,
    )


def launch_gather_shuffle5d_to_linear(k_cache, v_cache, slot_ids, out=None):
    total_kv = slot_ids.numel()
    if out is None:
        k_linear = torch.empty(
            total_kv, 1, 1, D_QK, dtype=k_cache.dtype, device=k_cache.device
        )
        v_linear = torch.empty(
            total_kv, 1, 1, D_V, dtype=v_cache.dtype, device=v_cache.device
        )
    else:
        k_linear, v_linear = out
    _gather_shuffle5d_to_linear_kernel[(total_kv,)](
        k_cache,
        v_cache,
        slot_ids,
        k_linear,
        v_linear,
        PAGE_SIZE=SWA_PAGE_SIZE,
        HEAD_DIM_QK=D_QK,
        HEAD_DIM_V=D_V,
        VECTOR_SIZE=16 // k_cache.element_size(),
        BLOCK_QK=256,
        BLOCK_V=128,
        num_warps=4,
    )
    return k_linear, v_linear


def _make_indptr(lengths):
    return torch.tensor(
        [0, *accumulate(lengths)], dtype=torch.int32, device="cuda"
    )


def run_shuffle5d_swa_reference(case, linear_kv=None, out=None):
    if case["q"].dtype != torch.bfloat16:
        return mha_batch_prefill_func(
            case["q_reference"],
            case["k_reference"][:, None],
            case["v_reference"][:, None],
            case["qo_indptr"],
            case["kv_indptr"],
            case["identity"],
            case["max_q"],
            case["max_kv"],
            causal=True,
            window_size=(case["window_left"], -1),
            sink_ptr=case["sinks"],
            out=out,
        )
    owns_linear_kv = linear_kv is None
    if linear_kv is None:
        linear_kv = launch_gather_shuffle5d_to_linear(
            case["k_cache"], case["v_cache"], case["slot_ids"]
        )
    k_linear, v_linear = linear_kv
    result = mha_batch_prefill_func(
        case["q"],
        k_linear,
        v_linear,
        case["qo_indptr"],
        case["kv_indptr"],
        case["identity"],
        case["max_q"],
        case["max_kv"],
        causal=True,
        logits_soft_cap=0.0,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=(SWA_WINDOW_LEFT, -1),
        sink_ptr=case["sinks"],
        out=out,
    )
    if owns_linear_kv:
        result._linear_kv_keepalive = linear_kv
    return result


def _cuda_event_benchmark(function, warmup=5, iterations=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def attention_error_metrics(actual, expected):
    absolute_error = (actual.float() - expected.float()).abs()
    relative_error = absolute_error / expected.float().abs().clamp_min(1e-12)
    return {
        "max_abs": absolute_error.max().item(),
        "max_rel": relative_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
    }


def vectorize_kv_cache(
    k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size
):
    k_vector_size = 16 // torch.tensor([], dtype=k_cache.dtype).element_size()

    """
    [num_pages, page_size, num_kv_heads, head_dim]
      ->
    K: [num_pages, num_kv_heads, (head_dim // k_vector_size, page_size, k_vector_size)]
    V: [num_pages, num_kv_heads, (page_size // k_vector_size, head_dim, k_vector_size)]

    对于K， head_dim 是 Q @ K gemm的reduce维度K
    对于V， page_size(token数) 是 P @ V gemm的reduce维度K

    最内层维度是16字节,确保 K 维度可以使用 DWORDx4(b128) 的读取宽度：

        K: (head_dim // 16, page_size, 16)
        V: (page_size // 16, head_dim, 16)

    """
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    k_cache = (
        k_cache.view(
            -1, page_size, num_kv_heads, head_dim_qk // k_vector_size, k_vector_size
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(
            -1, page_size // k_vector_size, k_vector_size, num_kv_heads, head_dim_v
        )
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


def make_paged_shuffle5d_swa_case(
    prefix_lens,
    extend_lens,
    seed=1,
    *,
    num_qo_heads=H_Q,
    num_kv_heads=H_KV,
    head_dim_qk=D_QK,
    head_dim_v=D_V,
    window_left=SWA_WINDOW_LEFT,
    quant_dtype=torch.bfloat16,
):
    assert len(prefix_lens) == len(extend_lens)
    assert quant_dtype in (torch.bfloat16, FP8_DTYPE)
    sequence_lens = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ]
    page_counts = [
        math.ceil(sequence_len / SWA_PAGE_SIZE)
        for sequence_len in sequence_lens
    ]
    total_q = sum(extend_lens)
    total_kv = sum(sequence_lens)
    total_pages = sum(page_counts)
    physical_page_count = total_pages + 8
    generator = torch.Generator(device="cuda").manual_seed(seed)

    q_2d_bf16 = torch.randn(
        total_q,
        num_qo_heads * head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_logical_bf16 = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v_logical_bf16 = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_v,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    physical_pages = torch.randperm(
        physical_page_count - 1,
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )[:total_pages].add(1).to(torch.int32)
    slot_segments = []
    page_offset = 0
    for sequence_len, page_count in zip(sequence_lens, page_counts):
        request_pages = physical_pages[page_offset : page_offset + page_count]
        token_positions = torch.arange(
            sequence_len, dtype=torch.int64, device="cuda"
        )
        slot_segments.append(
            request_pages.long()[token_positions // SWA_PAGE_SIZE]
            * SWA_PAGE_SIZE
            + token_positions % SWA_PAGE_SIZE
        )
        page_offset += page_count
    slot_ids = torch.cat(slot_segments).to(torch.int32)

    q_bf16 = q_2d_bf16.contiguous().view(
        total_q, num_qo_heads, head_dim_qk
    )
    if quant_dtype == torch.bfloat16:
        q = q_bf16
        k_logical = k_logical_bf16
        v_logical = v_logical_bf16
        q_descale = torch.ones(
            total_q, num_qo_heads, 1, dtype=torch.float32, device="cuda"
        )
        k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        v_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    else:
        q, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_logical, k_scale = per_tensor_quant(
            k_logical_bf16, quant_dtype=quant_dtype
        )
        v_logical, v_scale = per_tensor_quant(
            v_logical_bf16, quant_dtype=quant_dtype
        )
    vector_size = 16 // q.element_size()
    q_reference = (q.float() * q_descale).to(torch.bfloat16)
    k_reference = (k_logical.float() * k_scale).to(torch.bfloat16)
    v_reference = (v_logical.float() * v_scale).to(torch.bfloat16)

    k_cache = torch.zeros(
        physical_page_count,
        num_kv_heads,
        head_dim_qk // vector_size,
        SWA_PAGE_SIZE,
        vector_size,
        dtype=quant_dtype,
        device="cuda",
    )
    v_cache = torch.zeros(
        physical_page_count,
        num_kv_heads,
        SWA_PAGE_SIZE // vector_size,
        head_dim_v,
        vector_size,
        dtype=quant_dtype,
        device="cuda",
    )
    k_cache[0].fill_(float("nan"))
    v_cache[0].fill_(float("nan"))
    slot_long = slot_ids.long()
    page = slot_long // SWA_PAGE_SIZE
    page_token = slot_long % SWA_PAGE_SIZE
    for kv_head in range(num_kv_heads):
        k_cache[page, kv_head, :, page_token, :] = k_logical[:, kv_head].view(
            total_kv, head_dim_qk // vector_size, vector_size
        )
        v_cache[
            page,
            kv_head,
            page_token // vector_size,
            :,
            page_token % vector_size,
        ] = v_logical[:, kv_head]

    page_indices = torch.cat(
        [
            physical_pages,
            torch.zeros(256, dtype=torch.int32, device="cuda"),
        ]
    )
    return {
        "q": q,
        "k_logical": k_logical,
        "v_logical": v_logical,
        "q_reference": q_reference,
        "k_reference": k_reference,
        "v_reference": v_reference,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "slot_ids": slot_ids,
        "qo_indptr": _make_indptr(extend_lens),
        "kv_indptr": _make_indptr(sequence_lens),
        "paged_kv_indptr": _make_indptr(page_counts),
        "paged_kv_indices": page_indices,
        "kv_last_page_lens": torch.tensor(
            [
                (sequence_len - 1) % SWA_PAGE_SIZE + 1
                for sequence_len in sequence_lens
            ],
            dtype=torch.int32,
            device="cuda",
        ),
        "identity": torch.cat(
            [
                torch.arange(total_kv, dtype=torch.int32, device="cuda"),
                torch.zeros(256, dtype=torch.int32, device="cuda"),
            ]
        ),
        "q_descale": q_descale,
        "scale": k_scale,
        "v_scale": v_scale,
        "sinks": torch.linspace(
            -1.0, 1.0, num_qo_heads, dtype=torch.float32, device="cuda"
        ),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim_qk,
        "head_dim_v": head_dim_v,
        "window_left": window_left,
        "max_q": max(extend_lens),
        "max_kv": max(sequence_lens),
        "total_q": total_q,
        "total_kv": total_kv,
        "total_pages": total_pages,
    }


def run_paged_shuffle5d_swa_flypa(case, out=None):
    kernel = PagedAttention(
        case["num_qo_heads"],
        case["num_kv_heads"],
        case["head_dim_qk"],
        case["head_dim_v"],
        SWA_PAGE_SIZE,
        True,
        "per-token",
        "vectorized",
        case["window_left"],
        True,
    )
    return kernel(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["qo_indptr"],
        case["kv_indptr"],
        case["paged_kv_indptr"],
        case["paged_kv_indices"],
        case["max_q"],
        case["max_kv"],
        True,
        case["q_descale"],
        case["scale"],
        case["v_scale"],
        case["kv_last_page_lens"],
        out=out,
        sink_ptr=case["sinks"],
    )


def run_paged_shuffle5d_swa_ck(case, out=None):
    return mha_batch_prefill_func(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["qo_indptr"],
        case["paged_kv_indptr"],
        case["paged_kv_indices"],
        case["max_q"],
        case["max_kv"],
        causal=True,
        window_size=(SWA_WINDOW_LEFT, -1),
        sink_ptr=case["sinks"],
        kv_last_page_lens=case["kv_last_page_lens"],
        out=out,
    )


def benchmark_paged_shuffle5d_swa(
    case,
    warmup=5,
    iterations=20,
    benchmark_ck_direct=None,
):
    linear_kv = (
        torch.empty(
            case["total_kv"], 1, 1, D_QK, dtype=torch.bfloat16, device="cuda"
        ),
        torch.empty(
            case["total_kv"], 1, 1, D_V, dtype=torch.bfloat16, device="cuda"
        ),
    )
    reference_out = torch.empty(
        case["total_q"], H_Q, D_V, dtype=torch.bfloat16, device="cuda"
    )
    flypa_out = torch.empty_like(reference_out)
    ck_out = torch.empty_like(reference_out)

    def gather():
        return launch_gather_shuffle5d_to_linear(
            case["k_cache"],
            case["v_cache"],
            case["slot_ids"],
            out=linear_kv,
        )

    def attention():
        return run_shuffle5d_swa_reference(
            case, linear_kv=linear_kv, out=reference_out
        )

    def reference_total():
        gather()
        return attention()

    def flypa():
        return run_paged_shuffle5d_swa_flypa(case, out=flypa_out)

    gather()
    attention()
    flypa()
    timings = {
        "gather_us": _cuda_event_benchmark(gather, warmup, iterations),
        "attention_us": _cuda_event_benchmark(attention, warmup, iterations),
        "reference_us": _cuda_event_benchmark(
            reference_total, warmup, iterations
        ),
        "flypa_us": _cuda_event_benchmark(flypa, warmup, iterations),
    }
    timings["speedup"] = timings["reference_us"] / timings["flypa_us"]
    timings["temporary_mib"] = (
        case["total_kv"] * (D_QK + D_V) * 2 / 1024**2
    )
    timings["gather_rw_mib"] = case["total_kv"] * 1280 / 1024**2

    if benchmark_ck_direct is None:
        benchmark_ck_direct = os.environ.get("PYHIP_BENCH_CK_DIRECT_5D") == "1"
    ck_error = None
    timings["ck_direct_us"] = None
    if benchmark_ck_direct:
        try:
            run_paged_shuffle5d_swa_ck(case, out=ck_out)
            timings["ck_direct_us"] = _cuda_event_benchmark(
                lambda: run_paged_shuffle5d_swa_ck(case, out=ck_out),
                warmup,
                iterations,
            )
        except RuntimeError as error:
            ck_error = str(error).splitlines()[0]
    else:
        ck_error = (
            "not probed; set PYHIP_BENCH_CK_DIRECT_5D=1 on an AITER build "
            "with the D192/V128 page64 patch"
        )

    ck_display = (
        f"{timings['ck_direct_us']:.2f}"
        if timings["ck_direct_us"] is not None
        else "N/A"
    )
    print(
        "\n| workload | gather (us) | AITER linear (us) | gather+AITER (us) "
        "| CK direct-5D (us) | FlyPA direct-5D (us) | speedup "
        "| temporary MiB | gather R+W MiB |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| B={case['qo_indptr'].numel() - 1}, "
        f"Q={case['total_q']}, KV={case['total_kv']} "
        f"| {timings['gather_us']:.2f} | {timings['attention_us']:.2f} "
        f"| {timings['reference_us']:.2f} | {ck_display} "
        f"| {timings['flypa_us']:.2f} | {timings['speedup']:.3f}x "
        f"| {timings['temporary_mib']:.2f} | {timings['gather_rw_mib']:.2f} |"
    )
    if ck_error is not None:
        print(f"CK direct-5D unavailable: {ck_error}")
    return timings


@dataclass
class ModelConfig:
    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    page_size: int = 32
    quant_dtype = FP8_DTYPE
    is_causal: bool = True

def do_test_pa_prefill(
    modelcfg: ModelConfig, batch_size, qo_len, kv_len, is_causal = None, page_size = None, quant_query_mode="per-token", num_iters=10
):

    """Cover MiMo's direct cached-prefill contract and ragged last pages."""
    torch.cuda.empty_cache()
    torch.manual_seed(20260730)
    num_qo_heads, num_kv_heads = modelcfg.num_qo_heads, modelcfg.num_kv_heads
    page_size = modelcfg.page_size if page_size is None else page_size
    head_dim_qk = modelcfg.head_dim_qk
    head_dim_v = modelcfg.head_dim_v
    quant_dtype = modelcfg.quant_dtype
    is_causal = modelcfg.is_causal if is_causal is None else is_causal

    pages_per_seq = math.ceil(kv_len / page_size)
    num_pages = batch_size * pages_per_seq

    """
    q:torch.Size([10240, 16, 128]) 
    k:torch.Size([640, 1, 8, 16, 16])
    
    k.size(-3) * k_vector_size != head_size_q_og
    
    (head_dim // k_vector_size) 

    K: [num_pages, num_kv_heads, (head_dim_qk // k_vector_size, page_size, k_vector_size)]
    V: [num_pages, num_kv_heads, (page_size // k_vector_size, head_dim_v, k_vector_size)]

    """

    q_bf16 = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )                                 # [batch_size * qo_len, num_qo_heads, head_dim_qk]
    k_bf16 = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_bf16 = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
    ) # [num_pages, page_size, num_kv_heads, head_dim_v]
    if quant_dtype != torch.bfloat16:
        q_fp8, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype) if quant_query_mode == "per-token" else per_tensor_quant(q_bf16, quant_dtype=quant_dtype)
        k_fp8, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
        v_fp8, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)
    else:
        q_fp8, q_descale = q_bf16, torch.ones([batch_size * qo_len, num_qo_heads, 1], device="cuda", dtype=torch.float32)
        k_fp8, k_descale = k_bf16, torch.ones(1, device="cuda", dtype=torch.float32)
        v_fp8, v_descale = v_bf16, torch.ones(1, device="cuda", dtype=torch.float32)

    # Reverse each request's physical pages so a linear-addressing accident
    # cannot pass while still keeping page ownership disjoint across requests.
    page_table = torch.arange(num_pages, dtype=torch.int32).view(
        batch_size, pages_per_seq
    )
    page_table = page_table.flip(1).contiguous()
    kv_page_indices = page_table.flatten().to("cuda")
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * qo_len,
        qo_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * kv_len,
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_seq,
        pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    kv_last_page_lens = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    k_vec, v_vec = vectorize_kv_cache(
        k_fp8,
        v_fp8,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        page_size,
    )

    # Use a NaN sentinel so a tile that silently leaves any token/head lane
    # unwritten cannot pass merely because zero is inside the loose FP8
    # absolute-error threshold.
    out = torch.full(
        (batch_size * qo_len, num_qo_heads, head_dim_v),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )

    fly_mha = PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, is_causal, quant_query_mode)

    pyhip.run_perftest(
        # aiter.mha_batch_prefill_func,
        fly_mha,
        q_fp8,
        k_vec,
        v_vec,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=is_causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens,
        out=out,
        num_iters=num_iters,
        num_verbose=1,
        num_flops = (batch_size * num_qo_heads * (qo_len * kv_len * head_dim_qk + qo_len * kv_len * head_dim_v) * 2)//(2 if is_causal else 1)
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), f"{out}"

    # assert 0, f"{q_fp8.shape} {q_fp8.dtype} / {q_descale.shape} {q_descale.dtype} / {k_descale.dtype} / {v_descale.dtype}"
    try:
        q_ref = q_fp8.float() * q_descale
        refs = []
        for batch_idx in range(batch_size):
            pages = page_table[batch_idx].long().to("cuda")
            k_ref = (k_fp8[pages].reshape(-1, num_kv_heads, head_dim_qk)[:kv_len].float())
            v_ref = (v_fp8[pages].reshape(-1, num_kv_heads, head_dim_v)[:kv_len].float())
            k_ref = (k_ref * k_descale).repeat_interleave(
                num_qo_heads // num_kv_heads, dim=1
            )
            v_ref = (v_ref * v_descale).repeat_interleave(
                num_qo_heads // num_kv_heads, dim=1
            )
            q_ref_batch = q_ref[
                batch_idx * qo_len : (batch_idx + 1) * qo_len
            ]
            rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
            cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
            causal_mask = cols <= (kv_len - qo_len + rows) if is_causal else None
            refs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_ref_batch.transpose(0, 1).unsqueeze(0),
                    k_ref.transpose(0, 1).unsqueeze(0),
                    v_ref.transpose(0, 1).unsqueeze(0),
                    # Paged-prefill uses a bottom-right-aligned causal mask when
                    # Q and KV lengths differ.  PyTorch's is_causal=True is
                    # top-left aligned, so pass the explicit mask instead.
                    attn_mask=causal_mask,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        reference = torch.cat(refs, dim=0)
        pyhip.allclose(out.float(), reference.float(), rtol=1e-1, atol=1e-1)
        diff = pyhip.calc_diff(out.float(), reference.float())
        assert diff < 0.001, f"big diff: {diff}"
    except Exception as e:
        print("[accuracy unknown]: ", e)

    #verify_fp8_output()

multi_processor_count = torch.cuda.get_device_properties().multi_processor_count

model_d128 = ModelConfig("Llama3_70B_TP8", num_qo_heads=8, num_kv_heads=1, head_dim_qk=128, head_dim_v=128)
model_mimo = ModelConfig("MiMo_TP8", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128)
model_mimo_padv = ModelConfig("MiMo_TP8", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=192)
model_d128_bf16 = ModelConfig("Llama3_BF16", num_qo_heads=16, num_kv_heads=1, head_dim_qk=128, head_dim_v=128)
model_d128_bf16.quant_dtype = torch.bfloat16
model_mimo_bf16 = ModelConfig("MiMo_BF16", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128)
model_mimo_bf16.quant_dtype = torch.bfloat16

@pytest.mark.parametrize("modelcfg", [model_d128, model_mimo, model_mimo_padv])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("quant_query_mode", ["per-token", "per-tensor"])
@requires_gfx942_or_gfx950
def test_accuracy(modelcfg, is_causal, page_size, quant_query_mode):
    do_test_pa_prefill(modelcfg, 3, 8192+79, 8192+153, is_causal=is_causal, page_size=page_size, quant_query_mode=quant_query_mode, num_iters=1)

@pytest.mark.parametrize(
    ("modelcfg", "qo_len", "kv_len", "is_causal", "page_size"),
    [
        (model_d128_bf16, 9, 9, False, 32),
        (model_mimo_bf16, 9, 9, False, 32),
        (model_mimo_bf16, 1024, 1024, True, 64),
    ],
)
@requires_gfx942_or_gfx950
def test_bf16_accuracy(modelcfg, qo_len, kv_len, is_causal, page_size):
    do_test_pa_prefill(
        modelcfg,
        1,
        qo_len,
        kv_len,
        is_causal=is_causal,
        page_size=page_size,
        num_iters=1,
    )


@requires_gfx950
@pytest.mark.parametrize(
    ("prefix_lens", "extend_lens"),
    [
        ([1], [62]),
        ([1], [63]),
        ([1], [64]),
        ([1], [126]),
        ([1], [127]),
        ([1], [128]),
        ([128], [1]),
        ([128], [128]),
        ([128], [129]),
        ([1025], [129]),
        ([64, 191, 320], [129, 256, 17]),
    ],
)
def test_paged_shuffle5d_swa_matches_gather_aiter(prefix_lens, extend_lens):
    case = make_paged_shuffle5d_swa_case(prefix_lens, extend_lens)
    gathered = launch_gather_shuffle5d_to_linear(
        case["k_cache"], case["v_cache"], case["slot_ids"]
    )
    torch.testing.assert_close(gathered[0][:, 0, 0], case["k_logical"][:, 0])
    torch.testing.assert_close(gathered[1][:, 0, 0], case["v_logical"][:, 0])

    expected = run_shuffle5d_swa_reference(case, linear_kv=gathered)
    actual = run_paged_shuffle5d_swa_flypa(case)
    assert actual.shape == (case["total_q"], H_Q, D_V)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    print(attention_error_metrics(actual, expected))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_paged_shuffle5d_swa_reuses_counter():
    case = make_paged_shuffle5d_swa_case([1025], [129])
    expected = run_shuffle5d_swa_reference(case)
    outputs = [run_paged_shuffle5d_swa_flypa(case) for _ in range(8)]
    torch.cuda.synchronize()
    for output in outputs:
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


@requires_gfx950
@pytest.mark.parametrize(
    ("prefix_lens", "extend_lens"),
    [([1025], [129]), ([64, 191, 320], [129, 256, 17])],
)
def test_paged_shuffle5d_swa_fp8_matches_dequantized_reference(
    prefix_lens, extend_lens
):
    case = make_paged_shuffle5d_swa_case(
        prefix_lens, extend_lens, quant_dtype=FP8_DTYPE
    )
    expected = run_shuffle5d_swa_reference(case)
    actual = run_paged_shuffle5d_swa_flypa(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "head_dim_qk", "head_dim_v", "window_left"),
    [
        (16, 1, 192, 128, 64),
        (8, 1, 192, 128, 128),
        (8, 2, 128, 192, 64),
    ],
)
def test_paged_shuffle5d_swa_relaxed_specializations(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    window_left,
):
    case = make_paged_shuffle5d_swa_case(
        [1025],
        [129],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        window_left=window_left,
    )
    expected = mha_batch_prefill_func(
        case["q"],
        case["k_logical"][:, None],
        case["v_logical"][:, None],
        case["qo_indptr"],
        case["kv_indptr"],
        case["identity"],
        case["max_q"],
        case["max_kv"],
        causal=True,
        window_size=(window_left, -1),
        sink_ptr=case["sinks"],
    )
    actual = run_paged_shuffle5d_swa_flypa(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_paged_shuffle5d_swa_prunes_page_zero_tombstones():
    case = make_paged_shuffle5d_swa_case([4096], [129])
    expected = run_shuffle5d_swa_reference(case)

    first_visible_token = 4096 - SWA_WINDOW_LEFT
    first_visible_page = first_visible_token // SWA_PAGE_SIZE
    case["paged_kv_indices"][:first_visible_page] = 0
    assert torch.isnan(case["k_cache"][0]).all()
    assert torch.isnan(case["v_cache"][0]).all()

    actual = run_paged_shuffle5d_swa_flypa(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_paged_shuffle5d_swa_partial_prefix_continuation():
    prefix_len = 65
    extend_len = 128
    case = make_paged_shuffle5d_swa_case([prefix_len], [extend_len])

    k_cache = torch.zeros_like(case["k_cache"])
    v_cache = torch.zeros_like(case["v_cache"])
    k_cache[0].fill_(float("nan"))
    v_cache[0].fill_(float("nan"))
    slot_ids = case["slot_ids"].long()

    def write_segment(begin, end):
        segment_slots = slot_ids[begin:end]
        page = segment_slots // SWA_PAGE_SIZE
        page_token = segment_slots % SWA_PAGE_SIZE
        k_cache[page, 0, :, page_token, :] = case["k_logical"][begin:end, 0].view(
            end - begin, D_QK // SWA_VECTOR_SIZE, SWA_VECTOR_SIZE
        )
        v_cache[
            page,
            0,
            page_token // SWA_VECTOR_SIZE,
            :,
            page_token % SWA_VECTOR_SIZE,
        ] = case["v_logical"][begin:end, 0]

    write_segment(0, prefix_len)
    write_segment(prefix_len, prefix_len + extend_len)
    case["k_cache"] = k_cache
    case["v_cache"] = v_cache

    expected = run_shuffle5d_swa_reference(case)
    actual = run_paged_shuffle5d_swa_flypa(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_paged_shuffle5d_swa_metadata_is_shared_across_60_layers():
    case = make_paged_shuffle5d_swa_case([128], [1])
    shared_page_indices = case["paged_kv_indices"]
    metadata_ptr = shared_page_indices.data_ptr()
    layer_k_buffers = [case["k_cache"].clone() for _ in range(60)]
    layer_v_buffers = [case["v_cache"].clone() for _ in range(60)]
    k_ptrs = set()
    v_ptrs = set()

    for k_buffer, v_buffer in zip(layer_k_buffers, layer_v_buffers):
        layer_case = dict(case)
        layer_case["k_cache"] = k_buffer
        layer_case["v_cache"] = v_buffer
        layer_case["paged_kv_indices"] = shared_page_indices
        k_ptrs.add(layer_case["k_cache"].data_ptr())
        v_ptrs.add(layer_case["v_cache"].data_ptr())
        assert layer_case["paged_kv_indices"].data_ptr() == metadata_ptr

        output = run_paged_shuffle5d_swa_flypa(layer_case)
        assert torch.isfinite(output).all()
        torch.cuda.synchronize()

    assert len(k_ptrs) == 60
    assert len(v_ptrs) == 60


def test_slot_mapped_lookup_is_not_a_current_kernel_contract():
    with pytest.raises(AssertionError):
        PagedAttention(
            H_Q,
            H_KV,
            D_QK,
            D_V,
            SWA_PAGE_SIZE,
            True,
            "per-token",
            "slot_mapped",
            SWA_WINDOW_LEFT,
            True,
        )


@requires_gfx950
@pytest.mark.parametrize("sink_mode", ["negative", "zero", "positive", "per_head"])
def test_paged_shuffle5d_swa_sink_matches_gather_aiter(sink_mode):
    case = make_paged_shuffle5d_swa_case([1024], [129])
    if sink_mode == "negative":
        case["sinks"].fill_(-20.0)
    elif sink_mode == "zero":
        case["sinks"].zero_()
    elif sink_mode == "positive":
        case["sinks"].fill_(20.0)
    else:
        case["sinks"].copy_(
            torch.linspace(-20.0, 20.0, H_Q, dtype=torch.float32, device="cuda")
        )

    expected = run_shuffle5d_swa_reference(case)
    actual = run_paged_shuffle5d_swa_flypa(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_paged_shuffle5d_full_regression():
    case = make_paged_shuffle5d_swa_case([128], [129])
    linear_kv = launch_gather_shuffle5d_to_linear(
        case["k_cache"], case["v_cache"], case["slot_ids"]
    )
    expected = mha_batch_prefill_func(
        case["q"],
        *linear_kv,
        case["qo_indptr"],
        case["kv_indptr"],
        case["identity"],
        case["max_q"],
        case["max_kv"],
        causal=True,
        window_size=(-1, -1),
    )
    kernel = PagedAttention(
        H_Q, H_KV, D_QK, D_V, SWA_PAGE_SIZE, True, "per-token", "vectorized"
    )
    actual = kernel(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["qo_indptr"],
        case["kv_indptr"],
        case["paged_kv_indptr"],
        case["paged_kv_indices"][: case["total_pages"]],
        case["max_q"],
        case["max_kv"],
        True,
        case["q_descale"],
        case["scale"],
        case["scale"],
        case["kv_last_page_lens"],
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.skipif(
    os.environ.get("PYHIP_RUN_PA_SWA_PERF") != "1",
    reason="set PYHIP_RUN_PA_SWA_PERF=1 to run the production SWA benchmark",
)
def test_paged_shuffle5d_swa_performance():
    case = make_paged_shuffle5d_swa_case([16384], [16384])
    linear_kv = launch_gather_shuffle5d_to_linear(
        case["k_cache"], case["v_cache"], case["slot_ids"]
    )
    expected = run_shuffle5d_swa_reference(case, linear_kv=linear_kv)
    torch.cuda.synchronize()
    actual = run_paged_shuffle5d_swa_flypa(case)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    timings = benchmark_paged_shuffle5d_swa(case)
    assert timings["flypa_us"] < timings["reference_us"]

if __name__ == "__main__":
    if "gfx950" in GPU_ARCH:
        benchmark_case = make_paged_shuffle5d_swa_case([16384], [16384])
        reference = run_shuffle5d_swa_reference(benchmark_case)
        candidate = run_paged_shuffle5d_swa_flypa(benchmark_case)
        torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
        benchmark_paged_shuffle5d_swa(benchmark_case)
    else:
        model = ModelConfig(
            "MiMo_TP8",
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim_qk=192,
            head_dim_v=128,
        )
        do_test_pa_prefill(
            model,
            1,
            32768,
            32768,
            is_causal=True,
            page_size=64,
            quant_query_mode="per-tensor",
        )
