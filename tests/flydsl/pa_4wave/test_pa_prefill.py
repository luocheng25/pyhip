import math
import importlib
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
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
from aiter import flash_attn_varlen_func, mha_batch_prefill_func
from pa_prefill_4wave import MHA

EIGHT_WAVE_DIR = Path(__file__).resolve().parent.parent / "pa_8wave"
eight_wave_spec = importlib.util.spec_from_file_location(
    "pa_prefill_8w32x32", EIGHT_WAVE_DIR / "pa_prefill_8w32x32.py"
)
assert eight_wave_spec is not None and eight_wave_spec.loader is not None
eight_wave_module = importlib.util.module_from_spec(eight_wave_spec)
sys.modules[eight_wave_spec.name] = eight_wave_module
eight_wave_spec.loader.exec_module(eight_wave_module)
PagedAttention = eight_wave_module.PagedAttention


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
    not torch.cuda.is_available()
    or "gfx942" not in GPU_ARCH,
    reason="requires gfx942",
)
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in GPU_ARCH,
    reason="requires gfx950",
)
requires_gfx942_or_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(arch in GPU_ARCH for arch in ("gfx942", "gfx950")),
    reason="requires gfx942 or gfx950",
)


def vectorize_kv_cache(k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size):
    vector_size = 16 // k_cache.element_size()
    k_cache = (
        k_cache.contiguous()
        .view(-1, page_size, num_kv_heads, head_dim_qk // vector_size, vector_size)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.contiguous()
        .view(-1, page_size // vector_size, vector_size, num_kv_heads, head_dim_v)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


@triton.jit
def _gather_swa_kv_kernel(
    k_cache,
    v_cache,
    slot_ids,
    k_linear,
    v_linear,
    PAGE_SIZE: tl.constexpr,  # type: ignore
    HEAD_DIM_QK: tl.constexpr,  # type: ignore
    HEAD_DIM_V: tl.constexpr,  # type: ignore
    VECTOR_SIZE: tl.constexpr,  # type: ignore
    BLOCK_QK: tl.constexpr,  # type: ignore
    BLOCK_V: tl.constexpr,  # type: ignore
):
    token_index = tl.program_id(0)
    slot = tl.load(slot_ids + token_index)
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
        k_linear + token_index * HEAD_DIM_QK + qk_dim,
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
        v_linear + token_index * HEAD_DIM_V + value_dim,
        v_value,
        mask=value_dim < HEAD_DIM_V,
    )


def gather_swa_kv(case, out=None):
    if out is None:
        k_linear = torch.empty(
            case["total_kv"],
            1,
            1,
            MIMO_BF16.head_dim_qk,
            dtype=case["k_cache"].dtype,
            device="cuda",
        )
        v_linear = torch.empty(
            case["total_kv"],
            1,
            1,
            MIMO_BF16.head_dim_v,
            dtype=case["v_cache"].dtype,
            device="cuda",
        )
    else:
        k_linear, v_linear = out
    vector_size = 16 // case["k_cache"].element_size()
    _gather_swa_kv_kernel[(case["total_kv"],)](
        case["k_cache"],
        case["v_cache"],
        case["slot_ids"],
        k_linear,
        v_linear,
        PAGE_SIZE=case["page_size"],
        HEAD_DIM_QK=MIMO_BF16.head_dim_qk,
        HEAD_DIM_V=MIMO_BF16.head_dim_v,
        VECTOR_SIZE=vector_size,
        BLOCK_QK=256,
        BLOCK_V=128,
        num_warps=4,
    )
    return k_linear, v_linear


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    page_size: int = 32
    quant_dtype: torch.dtype = FP8_DTYPE


MIMO_TP8 = ModelConfig("MiMo_TP8", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128)
FP8_REF = ModelConfig(
    "FP8_REF", num_qo_heads=8, num_kv_heads=1, head_dim_qk=128, head_dim_v=128
)
MIMO_BF16 = ModelConfig(
    "MiMo_BF16", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
BF16_REF = ModelConfig(
    "BF16_REF", num_qo_heads=1, num_kv_heads=1, head_dim_qk=128, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
H3_BF16 = ModelConfig(
    "MiniMax_H3", num_qo_heads=14, num_kv_heads=14, head_dim_qk=128, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
H3_SEGMENTS = (63225, 7)


def attention_flops(segments, num_heads, head_dim_qk, head_dim_v):
    return sum(
        2 * length * length * (head_dim_qk + head_dim_v) * num_heads
        for length in segments
    )


def run_formal_benchmark(
    kernel,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_indptr,
    kv_page_indices,
    qo_len,
    kv_len,
    causal,
    q_descale,
    k_descale,
    v_descale,
    kv_last_page_lens,
    output,
    flops,
    name="pa_prefill_4wave",
):
    num_buffers = 10
    num_warmup = 10
    num_samples = 50
    q_buffers = [q.clone() for _ in range(num_buffers)]
    k_buffers = [k.clone() for _ in range(num_buffers)]
    v_buffers = [v.clone() for _ in range(num_buffers)]
    q_descale_buffers = [q_descale.clone() for _ in range(num_buffers)]
    output_buffers = [torch.empty_like(output) for _ in range(num_buffers)]

    def launch(buffer_index):
        kernel(
            q_buffers[buffer_index], k_buffers[buffer_index], v_buffers[buffer_index],
            cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
            max_seqlen_q=qo_len, max_seqlen_k=kv_len, causal=causal,
            q_descale=q_descale_buffers[buffer_index], k_descale=k_descale, v_descale=v_descale,
            kv_last_page_lens=kv_last_page_lens, out=output_buffers[buffer_index],
        )

    for iteration in range(num_warmup):
        launch(iteration % num_buffers)
    torch.cuda.synchronize()

    samples_us = []
    for iteration in range(num_samples):
        with pyhip.cudaPerf(flops=flops, name=name, verbose=0) as perf:
            launch(iteration % num_buffers)
        samples_us.append(perf.dt() * 1e6)

    samples_us.sort()
    median_us = samples_us[num_samples // 2]
    median_tflops = flops * 1e-6 / median_us
    print(
        f"[formal:{name}] median={median_us:.3f} us tflops={median_tflops:.3f} "
        f"min={samples_us[0]:.3f} us max={samples_us[-1]:.3f} us"
    )
    return median_us, median_tflops


def make_h3_inputs(quant_dtype=torch.bfloat16):
    """Build the real H3 varlen pack in the paged-KV ABI used by this kernel."""
    assert quant_dtype in (torch.bfloat16, FP8_DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(1101)
    segments = H3_SEGMENTS
    num_qo_heads = H3_BF16.num_qo_heads
    num_kv_heads = H3_BF16.num_kv_heads
    head_dim = H3_BF16.head_dim_qk
    page_size = H3_BF16.page_size
    total_tokens = sum(segments)

    shape = (total_tokens, num_qo_heads, head_dim)
    q_bf16, k_packed, v_packed = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )

    pages_per_sequence = [(length + page_size - 1) // page_size for length in segments]
    num_pages = sum(pages_per_sequence)
    k_pages = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_pages = torch.zeros_like(k_pages)
    page_base = 0
    token_base = 0
    for length, page_count in zip(segments, pages_per_sequence):
        padded_length = page_count * page_size
        k_pages[page_base : page_base + page_count].view(padded_length, num_kv_heads, head_dim)[:length].copy_(
            k_packed[token_base : token_base + length]
        )
        v_pages[page_base : page_base + page_count].view(padded_length, num_kv_heads, head_dim)[:length].copy_(
            v_packed[token_base : token_base + length]
        )
        page_base += page_count
        token_base += length

    if quant_dtype == torch.bfloat16:
        q_input, k_input, v_input = q_bf16, k_pages, v_pages
        q_descale = torch.ones(total_tokens, num_qo_heads, 1, device="cuda", dtype=torch.float32)
        k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    else:
        q_input, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_input, k_descale = per_tensor_quant(k_pages, quant_dtype=quant_dtype)
        v_input, v_descale = per_tensor_quant(v_pages, quant_dtype=quant_dtype)

    k_input, v_input = vectorize_kv_cache(
        k_input, v_input, num_kv_heads, head_dim, head_dim, page_size
    )
    kv_page_indices = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    )
    kv_indptr = torch.tensor(
        [0, *torch.tensor(pages_per_sequence).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    kv_last_page_lens = torch.tensor(
        [(length - 1) % page_size + 1 for length in segments], device="cuda", dtype=torch.int32
    )
    output = torch.empty_like(q_bf16)
    kernel = MHA(num_qo_heads, num_kv_heads, head_dim, head_dim, page_size, False)

    def launch():
        kernel(
            q_input, k_input, v_input, cu_seqlens, cu_seqlens, kv_indptr, kv_page_indices,
            max_seqlen_q=max(segments), max_seqlen_k=max(segments), causal=False,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            kv_last_page_lens=kv_last_page_lens, out=output,
        )

    return q_bf16, k_packed, v_packed, cu_seqlens, output, launch


def run_h3_benchmark(dtype="bf16"):
    """Run the real MiniMax-H3 varlen pack with the AITER benchmark protocol."""
    quant_dtype = torch.bfloat16 if dtype == "bf16" else FP8_DTYPE
    q, _, _, _, output, launch = make_h3_inputs(quant_dtype)
    segments = H3_SEGMENTS
    num_qo_heads = H3_BF16.num_qo_heads
    head_dim = H3_BF16.head_dim_qk

    for _ in range(3):
        launch()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        samples_ms.append(start.elapsed_time(stop))

    flops = attention_flops(segments, num_qo_heads, head_dim, head_dim)
    assert flops == 28_653_368_031_232
    median_ms = statistics.median(samples_ms)
    tflops = flops / 1e9 / median_ms
    print(
        f"[h3] dtype={dtype} segments={segments} heads={num_qo_heads} dim={head_dim} "
        f"flops={flops / 1e12:.6f} TFLOP"
    )
    print(
        f"[h3:{dtype}:4wave] median={median_ms:.3f} ms min={min(samples_ms):.3f} ms "
        f"max={max(samples_ms):.3f} ms tflops={tflops:.3f}"
    )
    print("[h3:protocol] warmup=3 samples=10 timing=CUDA-event aggregation=statistics.median")
    print("[h3:formula] FLOPs=sum(4 * S_i^2 * head_dim * heads); TFLOPS=FLOPs/(median_ms*1e9)")
    assert torch.isfinite(output).all()
    return median_ms, tflops


def run_pa_prefill(model_config, batch_size, qo_len, kv_len, causal, num_iters=10):
    torch.manual_seed(20260730)
    num_qo_heads = model_config.num_qo_heads
    num_kv_heads = model_config.num_kv_heads
    head_dim_qk = model_config.head_dim_qk
    head_dim_v = model_config.head_dim_v
    page_size = model_config.page_size
    quant_dtype = model_config.quant_dtype

    pages_per_sequence = math.ceil(kv_len / page_size)
    num_pages = batch_size * pages_per_sequence
    q_bf16 = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim_qk, device="cuda", dtype=torch.bfloat16
    )
    k_bf16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim_qk, device="cuda", dtype=torch.bfloat16
    )
    v_bf16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim_v, device="cuda", dtype=torch.bfloat16
    )
    if quant_dtype == torch.bfloat16:
        q_input, k_input, v_input = q_bf16, k_bf16, v_bf16
        q_descale = torch.ones((batch_size * qo_len, num_qo_heads, 1), device="cuda", dtype=torch.float32)
        k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    else:
        q_input, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_input, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
        v_input, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)

    page_table = torch.arange(num_pages, dtype=torch.int32).view(batch_size, pages_per_sequence)
    page_table = page_table.flip(1).contiguous()
    kv_page_indices = page_table.flatten().to("cuda")
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * qo_len, qo_len, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.arange(
        0, (batch_size + 1) * pages_per_sequence, pages_per_sequence, dtype=torch.int32, device="cuda"
    )
    kv_last_page_lens = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )
    k_vectorized, v_vectorized = vectorize_kv_cache(
        k_input, v_input, num_kv_heads, head_dim_qk, head_dim_v, page_size
    )
    output = torch.full(
        (batch_size * qo_len, num_qo_heads, head_dim_v), float("nan"), device="cuda", dtype=torch.bfloat16
    )

    kernel = MHA(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, causal)
    flops = (
        batch_size
        * num_qo_heads
        * (qo_len * kv_len * head_dim_qk + qo_len * kv_len * head_dim_v)
        * 2
    ) // (2 if causal else 1)
    dtype_name = "bf16" if quant_dtype == torch.bfloat16 else "fp8"
    print(f"[case] dtype={dtype_name} batch={batch_size} qo={qo_len} kv={kv_len} causal={causal}")
    pyhip.run_perftest(
        kernel, q_input, k_vectorized, v_vectorized, cu_seqlens_q, cu_seqlens_k,
        kv_indptr, kv_page_indices,
        max_seqlen_q=qo_len, max_seqlen_k=kv_len, causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens, out=output,
        num_iters=num_iters, num_verbose=1, num_flops=flops,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()

    reference = None
    diff = None
    if os.environ.get("PA_SKIP_REFERENCE") != "1":
        q_reference = q_input.float() * q_descale
        references = []
        for batch_index in range(batch_size):
            pages = page_table[batch_index].long().to("cuda")
            k_reference = (
                k_input[pages].reshape(-1, num_kv_heads, head_dim_qk)[:kv_len].float()
                * k_descale
            )
            v_reference = (
                v_input[pages].reshape(-1, num_kv_heads, head_dim_v)[:kv_len].float()
                * v_descale
            )
            repeat = num_qo_heads // num_kv_heads
            k_reference = k_reference.repeat_interleave(repeat, dim=1)
            v_reference = v_reference.repeat_interleave(repeat, dim=1)
            q_batch = q_reference[batch_index * qo_len : (batch_index + 1) * qo_len]
            rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
            columns = torch.arange(kv_len, device="cuda").unsqueeze(0)
            causal_mask = columns <= (kv_len - qo_len + rows) if causal else None
            references.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_batch.transpose(0, 1).unsqueeze(0),
                    k_reference.transpose(0, 1).unsqueeze(0),
                    v_reference.transpose(0, 1).unsqueeze(0),
                    attn_mask=causal_mask,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )

        reference = torch.cat(references, dim=0)
        pyhip.allclose(output.float(), reference.float(), rtol=0.1, atol=0.1)
        diff = pyhip.calc_diff(output.float(), reference.float())
        print(f"[accuracy] diff={diff:.8f}")
        assert diff < 0.001, f"big diff: {diff}"
    else:
        print("[accuracy] reference skipped")

    if os.environ.get("PA_FORMAL_BENCH") == "1":
        run_formal_benchmark(
            kernel, q_input, k_vectorized, v_vectorized, cu_seqlens_q, cu_seqlens_k,
            kv_indptr, kv_page_indices,
            qo_len, kv_len, causal, q_descale, k_descale, v_descale, kv_last_page_lens, output, flops,
            name="4wave",
        )
    return diff


def make_swa_case(
    prefix_lens,
    extend_lens,
    seed=1,
    window_left=128,
    quant_dtype=torch.bfloat16,
):
    assert quant_dtype in (torch.bfloat16, FP8_DTYPE)
    page_size = 64
    num_qo_heads = MIMO_BF16.num_qo_heads
    num_kv_heads = MIMO_BF16.num_kv_heads
    head_dim_qk = MIMO_BF16.head_dim_qk
    head_dim_v = MIMO_BF16.head_dim_v
    sequence_lens = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ]
    page_counts = [math.ceil(length / page_size) for length in sequence_lens]
    total_q = sum(extend_lens)
    total_kv = sum(sequence_lens)
    total_pages = sum(page_counts)
    physical_page_count = total_pages + 8
    generator = torch.Generator(device="cuda").manual_seed(seed)

    q_bf16 = torch.randn(
        total_q,
        num_qo_heads,
        head_dim_qk,
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
        page_size,
        vector_size,
        dtype=quant_dtype,
        device="cuda",
    )
    v_cache = torch.zeros(
        physical_page_count,
        num_kv_heads,
        page_size // vector_size,
        head_dim_v,
        vector_size,
        dtype=quant_dtype,
        device="cuda",
    )
    k_cache[0].fill_(float("nan"))
    v_cache[0].fill_(float("nan"))
    logical_offset = 0
    page_offset = 0
    slot_id_chunks = []
    for sequence_len, page_count in zip(sequence_lens, page_counts):
        pages = physical_pages[page_offset : page_offset + page_count]
        token_positions = torch.arange(sequence_len, device="cuda")
        page = pages.long()[token_positions // page_size]
        page_token = token_positions % page_size
        slot_id_chunks.append(page * page_size + page_token)
        logical_slice = slice(logical_offset, logical_offset + sequence_len)
        k_cache[page, 0, :, page_token, :] = k_logical[logical_slice, 0].view(
            sequence_len, head_dim_qk // vector_size, vector_size
        )
        v_cache[
            page,
            0,
            page_token // vector_size,
            :,
            page_token % vector_size,
        ] = v_logical[logical_slice, 0]
        logical_offset += sequence_len
        page_offset += page_count

    def make_indptr(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
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
        "qo_indptr": make_indptr(extend_lens),
        "kv_indptr": make_indptr(sequence_lens),
        "paged_kv_indptr": make_indptr(page_counts),
        "paged_kv_indices": torch.cat(
            [
                physical_pages,
                torch.zeros(256, dtype=torch.int32, device="cuda"),
            ]
        ),
        "kv_last_page_lens": torch.tensor(
            [(length - 1) % page_size + 1 for length in sequence_lens],
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
        "slot_ids": torch.cat(slot_id_chunks).to(torch.int32),
        "sinks": torch.linspace(
            -1.0, 1.0, num_qo_heads, dtype=torch.float32, device="cuda"
        ),
        "page_size": page_size,
        "max_q": max(extend_lens),
        "max_kv": max(sequence_lens),
        "window_left": window_left,
        "total_q": total_q,
        "total_kv": total_kv,
    }


def run_swa_reference(case):
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
    )


def run_swa_aiter_batch_prefill(case, linear_kv, out=None):
    assert case["q"].dtype == torch.bfloat16
    return mha_batch_prefill_func(
        case["q"],
        linear_kv[0],
        linear_kv[1],
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


def run_swa_aiter_direct_paged(case, out=None):
    assert case["q"].dtype == torch.bfloat16
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
        window_size=(case["window_left"], -1),
        sink_ptr=case["sinks"],
        kv_last_page_lens=case["kv_last_page_lens"],
        out=out,
    )


def run_swa_aiter_varlen(case, linear_kv, out=None):
    assert case["q"].dtype == torch.bfloat16
    return flash_attn_varlen_func(
        case["q"],
        linear_kv[0][:, 0],
        linear_kv[1][:, 0],
        case["qo_indptr"],
        case["kv_indptr"],
        case["max_q"],
        case["max_kv"],
        causal=True,
        window_size=(case["window_left"], -1, 0),
        sink_ptr=case["sinks"],
        out=out,
    )


def run_swa_4wave(case, out=None, force_dynamic_schedule=False):
    if out is None:
        out = torch.empty(
            case["total_q"],
            MIMO_BF16.num_qo_heads,
            MIMO_BF16.head_dim_v,
            dtype=torch.bfloat16,
            device="cuda",
        )
    kernel = MHA(
        MIMO_BF16.num_qo_heads,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        MIMO_BF16.head_dim_v,
        64,
        True,
        "vectorized",
        case["window_left"],
        True,
        force_dynamic_schedule,
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
        out,
        sink_ptr=case["sinks"],
    )


def run_swa_8wave(case, out=None):
    kernel = PagedAttention(
        MIMO_BF16.num_qo_heads,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        MIMO_BF16.head_dim_v,
        case["page_size"],
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


@pytest.mark.parametrize(
    ("model_config", "batch_size", "qo_len", "kv_len", "causal"),
    [
        (BF16_REF, 2, 129, 83, False),
        (BF16_REF, 1, 129, 129, True),
        (FP8_REF, 2, 128, 83, False),
        (FP8_REF, 1, 128, 129, True),
        (MIMO_TP8, 2, 128, 83, False),
        (MIMO_TP8, 2, 128, 259, False),
        (MIMO_TP8, 1, 128, 129, True),
        (MIMO_TP8, 2, 128, 259, True),
        (MIMO_BF16, 2, 128, 129, True),
    ],
)
@pytest.mark.parametrize("page_size", [32, 64, 128])
@requires_gfx942_or_gfx950
def test_accuracy(model_config, batch_size, qo_len, kv_len, causal, page_size):
    diff = run_pa_prefill(
        replace(model_config, page_size=page_size),
        batch_size,
        qo_len,
        kv_len,
        causal,
        num_iters=1,
    )
    assert diff is not None and diff < 0.001


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
def test_swa_matches_aiter(prefix_lens, extend_lens):
    case = make_swa_case(prefix_lens, extend_lens)
    expected = run_swa_reference(case)
    actual = run_swa_4wave(case)
    assert actual is not None
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
def test_swa_prunes_invisible_pages():
    case = make_swa_case([4096], [129])
    expected = run_swa_reference(case)

    first_visible_page = (4096 - case["window_left"]) // 64
    invisible_pages = case["paged_kv_indices"][:first_visible_page].long()
    case["k_cache"].index_fill_(0, invisible_pages, float("nan"))
    case["v_cache"].index_fill_(0, invisible_pages, float("nan"))

    actual = run_swa_4wave(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.parametrize("sink_value", [-20.0, 0.0, 20.0])
def test_swa_sink_values(sink_value):
    case = make_swa_case([1024], [129])
    case["sinks"].fill_(sink_value)
    expected = run_swa_reference(case)
    actual = run_swa_4wave(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.parametrize(
    ("prefix_lens", "extend_lens"),
    [([1025], [129]), ([64, 191, 320], [129, 256, 17])],
)
def test_swa_fp8_matches_dequantized_reference(prefix_lens, extend_lens):
    case = make_swa_case(
        prefix_lens, extend_lens, quant_dtype=FP8_DTYPE
    )
    expected = run_swa_reference(case)
    actual = run_swa_4wave(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    torch.cuda.synchronize()


@requires_gfx950
def test_swa_forced_dynamic_matches_static():
    case = make_swa_case([1025], [129])
    static = run_swa_4wave(case)
    dynamic = run_swa_4wave(case, force_dynamic_schedule=True)
    assert torch.isfinite(dynamic).all()
    torch.testing.assert_close(dynamic, static, rtol=0, atol=0)
    torch.cuda.synchronize()


@requires_gfx950
def test_swa_forced_dynamic_reuses_counter():
    case = make_swa_case([1025], [129])
    static = run_swa_4wave(case)
    dynamic_outputs = [
        run_swa_4wave(case, force_dynamic_schedule=True) for _ in range(8)
    ]
    torch.cuda.synchronize()
    for dynamic in dynamic_outputs:
        assert torch.isfinite(dynamic).all()
        torch.testing.assert_close(dynamic, static, rtol=0, atol=0)


@requires_gfx950
def test_swa_aiter_paths():
    from torch.profiler import ProfilerActivity, profile

    case = make_swa_case([1025], [129])
    linear_kv = gather_swa_kv(case)
    torch.testing.assert_close(
        linear_kv[0][:, 0, 0], case["k_logical"][:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        linear_kv[1][:, 0, 0], case["v_logical"][:, 0], rtol=0, atol=0
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        batch_prefill = run_swa_aiter_batch_prefill(case, linear_kv)
        torch.cuda.synchronize()
    batch_events = {event.key for event in profiler.key_averages()}
    assert any("aiter::mha_batch_prefill" in event for event in batch_events)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        varlen = run_swa_aiter_varlen(case, linear_kv)
        torch.cuda.synchronize()
    varlen_events = {event.key for event in profiler.key_averages()}
    assert any("aiter::mha_varlen_fwd" in event for event in varlen_events)
    torch.testing.assert_close(varlen, batch_prefill, rtol=2e-2, atol=2e-2)

    try:
        direct = run_swa_aiter_direct_paged(case)
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert "no matching kernel found" in str(error)
    else:
        torch.testing.assert_close(direct, batch_prefill, rtol=2e-2, atol=2e-2)


@requires_gfx950
@pytest.mark.skipif(
    os.environ.get("PYHIP_RUN_PA_AITER_SWA_PERF") != "1",
    reason="set PYHIP_RUN_PA_AITER_SWA_PERF=1 to run production SWA AITER comparisons",
)
def test_swa_aiter_production_performance():
    rows = []
    for total_kv in (32768, 65536, 131072):
        case = make_swa_case([total_kv - 16384], [16384])
        linear_kv = (
            torch.empty(
                total_kv,
                1,
                1,
                MIMO_BF16.head_dim_qk,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty(
                total_kv,
                1,
                1,
                MIMO_BF16.head_dim_v,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        )
        batch_out = torch.empty(
            case["total_q"],
            MIMO_BF16.num_qo_heads,
            MIMO_BF16.head_dim_v,
            dtype=torch.bfloat16,
            device="cuda",
        )
        varlen_out = torch.empty_like(batch_out)
        direct_out = torch.empty_like(batch_out)
        four_wave_out = torch.empty_like(batch_out)
        eight_wave_out = torch.empty_like(batch_out)

        def gather():
            return gather_swa_kv(case, out=linear_kv)

        def batch_prefill():
            return run_swa_aiter_batch_prefill(case, linear_kv, out=batch_out)

        def gather_batch_prefill():
            gather()
            return batch_prefill()

        def varlen():
            return run_swa_aiter_varlen(case, linear_kv, out=varlen_out)

        def gather_varlen():
            gather()
            return varlen()

        def four_wave():
            return run_swa_4wave(case, out=four_wave_out)

        def eight_wave():
            return run_swa_8wave(case, out=eight_wave_out)

        gather()
        varlen_result = varlen()
        four_wave_result = four_wave()
        eight_wave_result = eight_wave()
        torch.testing.assert_close(
            four_wave_result, varlen_result, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            eight_wave_result, varlen_result, rtol=2e-2, atol=2e-2
        )
        candidates = {
            "gather": gather,
            "flash_varlen": varlen,
            "gather_flash_varlen": gather_varlen,
            "4-wave": four_wave,
            "8-wave": eight_wave,
        }
        if total_kv < 131072:
            batch_result = batch_prefill()
            torch.testing.assert_close(
                batch_result, varlen_result, rtol=2e-2, atol=2e-2
            )
            candidates["batch_prefill"] = batch_prefill
            candidates["gather_batch_prefill"] = gather_batch_prefill

        direct_status = "unsupported"
        try:
            direct_result = run_swa_aiter_direct_paged(case, out=direct_out)
            torch.cuda.synchronize()
        except RuntimeError as error:
            assert "no matching kernel found" in str(error)
        else:
            torch.testing.assert_close(
                direct_result, varlen_result, rtol=2e-2, atol=2e-2
            )
            candidates["direct_paged"] = lambda: run_swa_aiter_direct_paged(
                case, out=direct_out
            )
            direct_status = "supported"

        timings = benchmark_comparison_candidates(candidates)
        rows.append(
            {
                "kv": total_kv,
                "gather": timings["gather"],
                "batch_prefill": timings.get("batch_prefill"),
                "gather_batch_prefill": timings.get("gather_batch_prefill"),
                "flash_varlen": timings["flash_varlen"],
                "gather_flash_varlen": timings["gather_flash_varlen"],
                "direct_paged": timings.get("direct_paged"),
                "direct_status": direct_status,
                "4-wave": timings["4-wave"],
                "8-wave": timings["8-wave"],
            }
        )

    print(
        "\n| KV | gather | mha_batch_prefill_func only "
        "| gather + mha_batch_prefill_func | flash_attn_varlen_func only "
        "| gather + flash_attn_varlen_func | mha_batch_prefill_func direct-5D "
        "| 4-wave static | 8-wave persistent |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        batch_display = (
            f"{row['batch_prefill']:.2f} us"
            if row["batch_prefill"] is not None
            else "not run: known 128K fault"
        )
        total_display = (
            f"{row['gather_batch_prefill']:.2f} us"
            if row["gather_batch_prefill"] is not None
            else "not run: known 128K fault"
        )
        direct_display = (
            f"{row['direct_paged']:.2f} us"
            if row["direct_paged"] is not None
            else row["direct_status"]
        )
        print(
            f"| {row['kv'] // 1024}K | {row['gather']:.2f} us "
            f"| {batch_display} | {total_display} "
            f"| {row['flash_varlen']:.2f} us "
            f"| {row['gather_flash_varlen']:.2f} us | {direct_display} "
            f"| {row['4-wave']:.2f} us | {row['8-wave']:.2f} us |"
        )


def make_aiter_comparison_case(qo_len, kv_len, causal, seed=20260902):
    assert not causal or kv_len >= qo_len
    page_size = 64
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        qo_len,
        MIMO_BF16.num_qo_heads,
        MIMO_BF16.head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        kv_len,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        kv_len,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_v,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    num_pages = math.ceil(kv_len / page_size)
    padded_tokens = num_pages * page_size - kv_len
    if padded_tokens:
        k_padded = torch.cat(
            [
                k,
                torch.zeros(
                    padded_tokens,
                    MIMO_BF16.num_kv_heads,
                    MIMO_BF16.head_dim_qk,
                    dtype=k.dtype,
                    device=k.device,
                ),
            ]
        )
        v_padded = torch.cat(
            [
                v,
                torch.zeros(
                    padded_tokens,
                    MIMO_BF16.num_kv_heads,
                    MIMO_BF16.head_dim_v,
                    dtype=v.dtype,
                    device=v.device,
                ),
            ]
        )
    else:
        k_padded, v_padded = k, v

    k_pages = k_padded.view(
        num_pages,
        page_size,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
    )
    v_pages = v_padded.view(
        num_pages,
        page_size,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_v,
    )
    k_cache, v_cache = vectorize_kv_cache(
        k_pages.flip(0).contiguous(),
        v_pages.flip(0).contiguous(),
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        MIMO_BF16.head_dim_v,
        page_size,
    )

    def make_indptr(length):
        return torch.tensor([0, length], dtype=torch.int32, device="cuda")

    return {
        "qo_len": qo_len,
        "kv_len": kv_len,
        "causal": causal,
        "page_size": page_size,
        "q": q,
        "k": k,
        "v": v,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "cu_seqlens_q": make_indptr(qo_len),
        "cu_seqlens_k": make_indptr(kv_len),
        "kv_indptr": make_indptr(num_pages),
        "kv_page_indices": torch.arange(
            num_pages - 1, -1, -1, dtype=torch.int32, device="cuda"
        ),
        "kv_last_page_lens": torch.tensor(
            [(kv_len - 1) % page_size + 1], dtype=torch.int32, device="cuda"
        ),
        "identity": torch.arange(kv_len, dtype=torch.int32, device="cuda"),
        "q_descale": torch.ones(
            qo_len,
            MIMO_BF16.num_qo_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        ),
        "k_descale": torch.ones(1, dtype=torch.float32, device="cuda"),
        "v_descale": torch.ones(1, dtype=torch.float32, device="cuda"),
    }


def run_dispatched_aiter(case, out=None):
    if case["qo_len"] == case["kv_len"]:
        result = mha_batch_prefill_func(
            case["q"],
            case["k"][:, None],
            case["v"][:, None],
            case["cu_seqlens_q"],
            case["cu_seqlens_k"],
            case["identity"],
            case["qo_len"],
            case["kv_len"],
            causal=case["causal"],
            out=out,
        )
        return result, "mha_batch_prefill"

    result = flash_attn_varlen_func(
        case["q"],
        case["k"],
        case["v"],
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["qo_len"],
        case["kv_len"],
        causal=case["causal"],
        out=out,
    )
    return result, "flash_attn_varlen_func"


def run_comparison_4wave(case, out=None):
    if out is None:
        out = torch.empty(
            case["qo_len"],
            MIMO_BF16.num_qo_heads,
            MIMO_BF16.head_dim_v,
            dtype=torch.bfloat16,
            device="cuda",
        )
    kernel = MHA(
        MIMO_BF16.num_qo_heads,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        MIMO_BF16.head_dim_v,
        case["page_size"],
        case["causal"],
    )
    return kernel(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["qo_len"],
        case["kv_len"],
        case["causal"],
        case["q_descale"],
        case["k_descale"],
        case["v_descale"],
        case["kv_last_page_lens"],
        out,
    )


def run_comparison_8wave(case, out=None):
    kernel = PagedAttention(
        MIMO_BF16.num_qo_heads,
        MIMO_BF16.num_kv_heads,
        MIMO_BF16.head_dim_qk,
        MIMO_BF16.head_dim_v,
        case["page_size"],
        case["causal"],
    )
    return kernel(
        case["q"],
        case["k_cache"],
        case["v_cache"],
        case["cu_seqlens_q"],
        case["cu_seqlens_k"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["qo_len"],
        case["kv_len"],
        case["causal"],
        case["q_descale"],
        case["k_descale"],
        case["v_descale"],
        case["kv_last_page_lens"],
        out=out,
    )


def profile_aiter_dispatch(case):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        _, route = run_dispatched_aiter(case)
        torch.cuda.synchronize()
    event_names = {event.key for event in profiler.key_averages()}
    expected_event = (
        "aiter::mha_batch_prefill"
        if route == "mha_batch_prefill"
        else "FlashAttnVarlenFunc"
    )
    unexpected_event = (
        "FlashAttnVarlenFunc"
        if route == "mha_batch_prefill"
        else "aiter::mha_batch_prefill"
    )
    assert any(expected_event in event_name for event_name in event_names), (
        f"{route} did not emit {expected_event}; events={sorted(event_names)}"
    )
    assert not any(unexpected_event in event_name for event_name in event_names), (
        f"{route} unexpectedly emitted {unexpected_event}"
    )
    return route


def benchmark_comparison_candidates(candidates, warmup=20, samples=100, rounds=5):
    names = list(candidates)
    round_medians = {name: [] for name in names}
    for round_index in range(rounds):
        for iteration in range(warmup):
            offset = (round_index + iteration) % len(names)
            for name in names[offset:] + names[:offset]:
                candidates[name]()
        torch.cuda.synchronize()

        elapsed = {name: [] for name in names}
        pending = []
        for iteration in range(samples):
            offset = (round_index + iteration) % len(names)
            for name in names[offset:] + names[:offset]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                candidates[name]()
                end.record()
                pending.append((name, start, end))
        torch.cuda.synchronize()
        for name, start, end in pending:
            elapsed[name].append(start.elapsed_time(end) * 1000)
        for name in names:
            round_medians[name].append(statistics.median(elapsed[name]))

    return {
        name: statistics.median(round_medians[name])
        for name in names
    }


def aiter_comparison_flops(case):
    flops = (
        2
        * MIMO_BF16.num_qo_heads
        * case["qo_len"]
        * case["kv_len"]
        * (MIMO_BF16.head_dim_qk + MIMO_BF16.head_dim_v)
    )
    return flops // (2 if case["causal"] else 1)


@requires_gfx950
@pytest.mark.parametrize(
    ("qo_len", "kv_len", "causal", "expected_route"),
    [
        (128, 128, True, "mha_batch_prefill"),
        (128, 96, False, "flash_attn_varlen_func"),
    ],
)
def test_pa_matches_dispatched_aiter(qo_len, kv_len, causal, expected_route):
    case = make_aiter_comparison_case(qo_len, kv_len, causal)
    expected, route = run_dispatched_aiter(case)
    assert route == expected_route
    assert profile_aiter_dispatch(case) == expected_route

    outputs = {
        "4-wave": run_comparison_4wave(case),
        "8-wave": run_comparison_8wave(case),
    }
    for name, output in outputs.items():
        assert torch.isfinite(output).all(), name
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        outputs["4-wave"], outputs["8-wave"], rtol=2e-2, atol=2e-2
    )
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.skipif(
    os.environ.get("PYHIP_RUN_PA_AITER_PERF") != "1",
    reason="set PYHIP_RUN_PA_AITER_PERF=1 to run production AITER comparisons",
)
def test_pa_aiter_production_performance():
    rows = []
    for qo_len, kv_len, causal in (
        (10240, 2583, False),
        (32768, 32768, True),
    ):
        case = make_aiter_comparison_case(qo_len, kv_len, causal)
        outputs = {
            name: torch.empty(
                qo_len,
                MIMO_BF16.num_qo_heads,
                MIMO_BF16.head_dim_v,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for name in ("aiter", "4-wave", "8-wave")
        }
        expected, route = run_dispatched_aiter(case, outputs["aiter"])
        actual_4wave = run_comparison_4wave(case, outputs["4-wave"])
        actual_8wave = run_comparison_8wave(case, outputs["8-wave"])
        torch.testing.assert_close(actual_4wave, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_8wave, expected, rtol=2e-2, atol=2e-2)

        timings = benchmark_comparison_candidates(
            {
                "aiter": lambda: run_dispatched_aiter(case, outputs["aiter"])[0],
                "4-wave": lambda: run_comparison_4wave(case, outputs["4-wave"]),
                "8-wave": lambda: run_comparison_8wave(case, outputs["8-wave"]),
            }
        )
        flops = aiter_comparison_flops(case)
        rows.append(
            {
                "shape": f"Q={qo_len},KV={kv_len},causal={causal}",
                "route": route,
                **{
                    f"{name}_us": timings[name]
                    for name in ("aiter", "4-wave", "8-wave")
                },
                **{
                    f"{name}_tflops": flops / (timings[name] * 1e6)
                    for name in ("aiter", "4-wave", "8-wave")
                },
                "4-wave_speedup": timings["aiter"] / timings["4-wave"],
                "8-wave_speedup": timings["aiter"] / timings["8-wave"],
            }
        )

    print(
        "\n| shape | AITER route | AITER us / TFLOPS | 4-wave us / TFLOPS "
        "| 4-wave vs AITER | 8-wave us / TFLOPS | 8-wave vs AITER |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        print(
            f"| {row['shape']} | {row['route']} "
            f"| {row['aiter_us']:.2f} / {row['aiter_tflops']:.2f}T "
            f"| {row['4-wave_us']:.2f} / {row['4-wave_tflops']:.2f}T "
            f"| {row['4-wave_speedup']:.3f}x "
            f"| {row['8-wave_us']:.2f} / {row['8-wave_tflops']:.2f}T "
            f"| {row['8-wave_speedup']:.3f}x |"
        )


def main():
    cases = {
        "short": [(1, 128, 3, False)],
        "tails": [
            (1, 128, 3, False),
            (1, 128, 13, False),
            (1, 128, 23, False),
            (1, 128, 53, False),
            (1, 128, 83, False),
        ],
        "noncausal": [(1, 256 * 40, 256 * 10 + 23, False)],
        "batch": [(4, 256 * 40, 256 * 10, False)],
        "causal": [(1, 32768, 32768, True)],
        "bf16_ref_short": [(1, 128, 83, False)],
        "bf16_ref": [(1, 40960, 40960, False)],
    }
    selected = os.environ.get("PA_CASE", "all")
    dtype = os.environ.get("PA_DTYPE", "bf16" if selected == "h3" else "fp8")
    if dtype not in ("fp8", "bf16"):
        raise ValueError(f"unknown PA_DTYPE={dtype!r}; expected 'fp8' or 'bf16'")
    if selected == "h3":
        run_h3_benchmark(dtype)
        return
    if selected == "all":
        selected_cases = [
            *cases["tails"],
            *cases["noncausal"],
            *cases["batch"],
            *cases["causal"],
        ]
    else:
        if selected not in cases:
            raise ValueError(
                f"unknown PA_CASE={selected!r}; expected one of {sorted(cases)} or 'all'"
            )
        selected_cases = cases[selected]

    num_iters = int(os.environ.get("PA_NUM_ITERS", "10"))
    if selected in ("bf16_ref_short", "bf16_ref"):
        model_config = BF16_REF
    else:
        model_config = MIMO_BF16 if dtype == "bf16" else MIMO_TP8
    for batch_size, qo_len, kv_len, causal in selected_cases:
        run_pa_prefill(model_config, batch_size, qo_len, kv_len, causal, num_iters=num_iters)


if __name__ == "__main__":
    main()
