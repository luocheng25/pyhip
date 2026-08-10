"""Low-level H3 FP8 runners for AITER Triton and FMHA v3 ASM."""

import math

import torch

from h3_paged_inputs import H3_HEAD_DIM, H3_HEADS, H3_SEGMENTS


def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def run_triton_fp8(inputs):
    """Run AITER's FP8-capable Triton kernel with explicit descales."""
    from aiter.ops.triton.attention.mha import _flash_attn_forward

    batch = len(H3_SEGMENTS)
    q_descale = inputs.q_descale.expand(batch, H3_HEADS).contiguous()
    k_descale = inputs.k_descale.expand(batch, H3_HEADS).contiguous()
    v_descale = inputs.v_descale.expand(batch, H3_HEADS).contiguous()
    output, *_ = _flash_attn_forward(
        inputs.q,
        inputs.k,
        inputs.v,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(H3_HEAD_DIM),
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        bias=None,
        alibi_slopes=None,
        return_lse=False,
        return_softmax=False,
        max_seqlen_q=max(H3_SEGMENTS),
        max_seqlen_k=max(H3_SEGMENTS),
        cu_seqlens_q=inputs.cu_seqlens,
        cu_seqlens_k=inputs.cu_seqlens,
        descale_q=q_descale,
        descale_k=k_descale,
        descale_v=v_descale,
    )
    return output


def make_triton_fp8_launcher(inputs):
    output = None

    def launch():
        nonlocal output
        output = run_triton_fp8(inputs)
        return output

    def get_output():
        return output

    return launch, get_output


def make_asm_fp8_launcher(inputs):
    """Bind AITER FMHA v3 grouped-varlen FP8 with a reusable BF16 output."""
    from aiter.ops.mha import _flash_attn_varlen_forward

    output = torch.empty(
        (sum(H3_SEGMENTS), H3_HEADS, H3_HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def launch():
        result = _flash_attn_varlen_forward(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.cu_seqlens,
            inputs.cu_seqlens,
            None,
            None,
            max(H3_SEGMENTS),
            max(H3_SEGMENTS),
            0,
            0.0,
            1.0 / math.sqrt(H3_HEAD_DIM),
            False,
            q_descale=inputs.q_descale,
            k_descale=inputs.k_descale,
            v_descale=inputs.v_descale,
            out=output,
        )
        return _first(result)

    return launch, lambda: output