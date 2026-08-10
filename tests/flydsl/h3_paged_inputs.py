"""Shared MiniMax-H3 paged-attention inputs for FlyDSL FP8/BF16 tests."""

from dataclasses import dataclass
import math

import torch


H3_SEGMENTS = (63225, 7)
H3_HEADS = 14
H3_HEAD_DIM = 128
H3_PAGE_SIZE = 32
H3_FLOPS = sum(
    4 * length * length * H3_HEAD_DIM * H3_HEADS for length in H3_SEGMENTS
)
FP8_DTYPE = torch.float8_e4m3fnuz


def per_token_quant(input_tensor, quant_dtype=FP8_DTYPE):
    """Match AITER pertoken_quant with no AITER import dependency."""
    values = input_tensor.float()
    scale = values.abs().amax(dim=-1, keepdim=True) / torch.finfo(quant_dtype).max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / scale).to(quant_dtype), scale.float()


def per_tensor_quant(input_tensor, quant_dtype=FP8_DTYPE):
    """Match AITER per_tensor_quant with no AITER import dependency."""
    values = input_tensor.float()
    scale = values.abs().max() / torch.finfo(quant_dtype).max
    return (values / scale).to(quant_dtype), scale.view(1).float()


def vectorize_kv_cache(
    k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size
):
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


@dataclass
class H3PagedInputs:
    dtype: torch.dtype
    q: torch.Tensor
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    cu_seqlens: torch.Tensor
    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor


@dataclass
class H3LinearInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    v_descale: torch.Tensor


def make_h3_linear_fp8_inputs(seed=1101):
    """Build the common per-tensor FP8 contract supported by Triton and ASM."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (sum(H3_SEGMENTS), H3_HEADS, H3_HEAD_DIM)
    q_bf16, k_bf16, v_bf16 = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    q, q_descale = per_tensor_quant(q_bf16)
    k, k_descale = per_tensor_quant(k_bf16)
    v, v_descale = per_tensor_quant(v_bf16)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(H3_SEGMENTS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return H3LinearInputs(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


def make_h3_paged_inputs(dtype=FP8_DTYPE, seed=1101):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_tokens = sum(H3_SEGMENTS)
    shape = (total_tokens, H3_HEADS, H3_HEAD_DIM)
    q_bf16, k_bf16, v_bf16 = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    if dtype == torch.bfloat16:
        q, k_packed, v_packed = q_bf16, k_bf16, v_bf16
        q_descale = torch.ones(
            total_tokens, H3_HEADS, 1, device="cuda", dtype=torch.float32
        )
        k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    elif dtype == FP8_DTYPE:
        q, q_descale = per_token_quant(q_bf16, dtype)
        k_packed, k_descale = per_tensor_quant(k_bf16, dtype)
        v_packed, v_descale = per_tensor_quant(v_bf16, dtype)
    else:
        raise ValueError(f"unsupported H3 dtype: {dtype}")

    pages_per_sequence = [
        math.ceil(length / H3_PAGE_SIZE) for length in H3_SEGMENTS
    ]
    num_pages = sum(pages_per_sequence)
    page_shape = (num_pages, H3_PAGE_SIZE, H3_HEADS, H3_HEAD_DIM)
    k_pages = torch.zeros(page_shape, device="cuda", dtype=dtype)
    v_pages = torch.zeros_like(k_pages)
    page_base = 0
    token_base = 0
    for length, page_count in zip(H3_SEGMENTS, pages_per_sequence):
        padded_length = page_count * H3_PAGE_SIZE
        k_pages[page_base : page_base + page_count].view(
            padded_length, H3_HEADS, H3_HEAD_DIM
        )[:length].copy_(k_packed[token_base : token_base + length])
        v_pages[page_base : page_base + page_count].view(
            padded_length, H3_HEADS, H3_HEAD_DIM
        )[:length].copy_(v_packed[token_base : token_base + length])
        page_base += page_count
        token_base += length

    k_pages, v_pages = vectorize_kv_cache(
        k_pages, v_pages, H3_HEADS, H3_HEAD_DIM, H3_HEAD_DIM, H3_PAGE_SIZE
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(H3_SEGMENTS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    kv_indptr = torch.tensor(
        [0, *torch.tensor(pages_per_sequence).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    kv_page_indices = torch.nn.functional.pad(
        torch.arange(num_pages, device="cuda", dtype=torch.int32), (0, 256)
    )
    kv_last_page_lens = torch.tensor(
        [(length - 1) % H3_PAGE_SIZE + 1 for length in H3_SEGMENTS],
        device="cuda",
        dtype=torch.int32,
    )
    return H3PagedInputs(
        dtype=dtype,
        q=q,
        k_packed=k_packed,
        v_packed=v_packed,
        k_pages=k_pages,
        v_pages=v_pages,
        cu_seqlens=cu_seqlens,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


def bind_h3_kernel(kernel_factory, inputs):
    output = torch.empty(
        inputs.q.shape, device="cuda", dtype=torch.bfloat16
    )
    kernel = kernel_factory(
        H3_HEADS, H3_HEADS, H3_HEAD_DIM, H3_HEAD_DIM, H3_PAGE_SIZE, False
    )

    def launch():
        kernel(
            inputs.q,
            inputs.k_pages,
            inputs.v_pages,
            inputs.cu_seqlens,
            inputs.kv_indptr,
            inputs.kv_page_indices,
            max_seqlen_q=max(H3_SEGMENTS),
            max_seqlen_k=max(H3_SEGMENTS),
            causal=False,
            q_descale=inputs.q_descale,
            k_descale=inputs.k_descale,
            v_descale=inputs.v_descale,
            kv_last_page_lens=inputs.kv_last_page_lens,
            out=output,
        )

    return output, launch


def h3_dequantized_sdpa_reference(inputs):
    k_input = inputs.k_packed if hasattr(inputs, "k_packed") else inputs.k
    v_input = inputs.v_packed if hasattr(inputs, "v_packed") else inputs.v
    q = (inputs.q.float() * inputs.q_descale).to(torch.bfloat16)
    k = (k_input.float() * inputs.k_descale).to(torch.bfloat16)
    v = (v_input.float() * inputs.v_descale).to(torch.bfloat16)
    output = torch.empty_like(q)
    start = 0
    for length in H3_SEGMENTS:
        stop = start + length
        q_segment, k_segment, v_segment = (
            tensor[start:stop].transpose(0, 1).unsqueeze(0)
            for tensor in (q, k, v)
        )
        output[start:stop] = torch.nn.functional.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(0).transpose(0, 1)
        start = stop
    return output


def compare_outputs(reference, candidate):
    ref = reference.float()
    value = candidate.float()
    difference = (ref - value).abs()
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            ref.flatten(), value.flatten(), dim=0
        ).item(),
        "max_abs": difference.max().item(),
        "rel_l2": (
            torch.linalg.vector_norm(ref - value)
            / torch.linalg.vector_norm(ref).clamp_min(1e-12)
        ).item(),
        "finite": bool(torch.isfinite(value).all().item()),
    }