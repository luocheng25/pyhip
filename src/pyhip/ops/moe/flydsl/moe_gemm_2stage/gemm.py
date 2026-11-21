# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-device cached dispatcher of the stage-specific MoE GEMM builders (pyhip only).

Upstream aiter defines ``compile_gemm`` in gfx9_fp8/__init__.py without a device cache, so one
process would reuse a launcher compiled on one GPU on another; here every device gets its own.
Each config goes to upstream gfx9_fp8 (``gemm1.compile_moe_gemm1`` / ``gemm2.compile_moe_gemm2``)
or to the previous pyhip implementation (``gemm1_legacy`` / ``gemm2_legacy``), see
``gemm_implementation``. The upstream batch1 and compact launchers take other arguments than
pyhip's; they are wrapped so that every launcher keeps pyhip's calling convention:

- batch1 Gate/Up and Down: ``(input, weight, output, topk_ids, clear_output | topk_weights,
  w_scale, batch, stream)``;
- compact Down: ``(..., full_capacity, tail_capacity, stream)``, the task buffers sized by
  ``gemm2_8x1_compact.allocate_task_buffers`` for this device's CU count.

``has_num_local_tokens=True`` (upstream only) returns upstream's batch1 launchers unchanged: they
also take the token-count pointer and the route count ``batch * TOPK``.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx

from . import gemm1_legacy, gemm2_legacy
from .common import get_device_cache_key, get_gfx_runtime
from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2
from .gemm2_8x1_compact_legacy import device_cu_count

UPSTREAM, LEGACY = "upstream", "legacy"
_COMPACT_K = (192, 256, 384, 512, 640)
_LDS_BYTES = 64 * 1024


def gemm_implementation(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    force_batch1_path=False,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    has_num_local_tokens=False,
):
    """``"upstream"`` if upstream gfx9_fp8 runs this ``compile_gemm`` config, else ``"legacy"``.

    Upstream covers gfx942 BF16/FP8 weights with SiLU/SwiGLU, and only the configs its host
    (aiter fused_moe_gfx942 ``Config.unsupported_reason``) accepts; its builders skip some of these
    checks. The rest stays on the legacy builders, which keep pyhip's assertions: SiTUv2, FP4/MXFP4,
    gfx950, METADATA_TILE_SIZE_M != BLOCK_TILE_SIZE_M, the standalone 8x1 Down, compact Down with
    K=320 or N % 256 != 0, and tiles the upstream checks reject.
    """
    if has_num_local_tokens:
        return UPSTREAM
    if force_batch1_path:
        alg = "batch1"
    if get_gfx_runtime() != "gfx942":
        return LEGACY
    if weight_dtype not in ("bf16", "fp8") or activation not in ("silu", "swiglu"):
        return LEGACY
    if METADATA_TILE_SIZE_M not in (None, BLOCK_TILE_SIZE_M):
        return LEGACY
    if alg not in ("splitk", "batch1", "prefill_1x4"):
        return LEGACY
    padding = down_output_padding_bytes or 0
    if stage == "gateup":
        if down_path != "default" or padding:
            return LEGACY
        if alg == "prefill_1x4":
            tile_k = tile_k or (128 if weight_dtype == "fp8" else 64)
            if N % BLOCK_TILE_SIZE_N or K % (2 * tile_k):
                return LEGACY
            if weight_dtype == "bf16" and (
                tile_k not in (64, 128) or 2 * BLOCK_TILE_SIZE_M * tile_k * 2 > _LDS_BYTES
            ):
                return LEGACY
        return UPSTREAM
    if stage != "down":
        return LEGACY
    if down_path == "default":
        if padding:
            return LEGACY
        if alg == "prefill_1x4" and (
            K % 64
            or N % 256
            or (weight_dtype == "bf16" and BLOCK_TILE_SIZE_M * K * 2 > _LDS_BYTES)
        ):
            return LEGACY
        return UPSTREAM
    if alg != "prefill_1x4" or weight_dtype != "fp8" or BLOCK_TILE_SIZE_M != 64:
        return LEGACY
    scale_bytes = 0 if weight_quant_type == "per_tensor" else 256 * 4
    if (
        padding not in (0, 32, 64, 128)
        or K % 64
        or N % 256
        or 64 * K + scale_bytes + 4 * 16 * 64 * 2 > _LDS_BYTES
    ):
        return LEGACY
    if down_path == "1x4_64x256":
        return UPSTREAM if BLOCK_TILE_SIZE_N == 256 else LEGACY
    if down_path == "8x1_compact":
        if (
            BLOCK_TILE_SIZE_N == 128
            and K in _COMPACT_K
            and N >= 512
            and E is not None
            and 0 < E <= 2048
        ):
            return UPSTREAM
    return LEGACY


def _batch1_launcher(launch, TOPK):
    """pyhip's batch1 arguments for an upstream batch1 launcher.

    Without has_num_local_tokens the upstream kernels never read the token-count pointer, so the
    topk ids stand in for it; upstream takes the route count instead of the batch size.
    """

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_clear_or_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        launch(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_clear_or_weights,
            p_w_scale,
            p_topk_ids,
            M * TOPK,
            stream,
        )

    launch_batch1.compile_hints.update(launch.compile_hints)
    return launch_batch1


def _compact_launcher(launch, cu_count):
    """pyhip's compact arguments for upstream's compact launcher, with this device's CU count."""

    @flyc.jit
    def launch_compact(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        p_full_tasks: fx.Pointer,
        p_tail_tasks: fx.Pointer,
        p_counts: fx.Pointer,
        full_capacity: fx.Int32,
        tail_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        launch(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            task_num,
            p_full_tasks,
            p_tail_tasks,
            p_counts,
            full_capacity,
            tail_capacity,
            fx.Int32(cu_count),
            stream,
        )

    launch_compact.compile_hints.update(launch.compile_hints)
    return launch_compact


@functools.cache
def _compile_gemm_cached(
    device_cache_key,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage,
    alg,
    E,
    USE_ATOMIC_WRITE,
    act_quant_type,
    tile_k,
    activation,
    swiglu_limit,
    situ_beta,
    situ_linear_beta,
    mxfp4_gate_up_interleaved,
    fused_down_clear,
    down_path,
    down_output_padding_bytes,
    METADATA_TILE_SIZE_M,
    has_num_local_tokens,
):
    del device_cache_key
    implementation = gemm_implementation(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        stage=stage,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
        has_num_local_tokens=has_num_local_tokens,
    )
    common = dict(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        alg=alg,
        E=E,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )
    if implementation == LEGACY:
        legacy = dict(
            common,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
            fused_down_clear=fused_down_clear,
            METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        )
        if stage == "gateup":
            return gemm1_legacy.compile_moe_gemm1(**legacy)
        return gemm2_legacy.compile_moe_gemm2(
            **legacy,
            USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
            down_path=down_path,
            down_output_padding_bytes=down_output_padding_bytes,
        )
    if stage == "gateup":
        launch = compile_moe_gemm1(
            **common,
            fused_down_clear=fused_down_clear,
            has_num_local_tokens=has_num_local_tokens,
        )
    else:
        launch = compile_moe_gemm2(
            **common,
            USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
            down_path=down_path,
            down_output_padding_bytes=down_output_padding_bytes or 0,
            has_num_local_tokens=has_num_local_tokens,
        )
    if alg == "batch1" and not has_num_local_tokens:
        return _batch1_launcher(launch, TOPK)
    if stage == "down" and down_path == "8x1_compact":
        return _compact_launcher(launch, device_cu_count())
    return launch


def compile_gemm(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    force_batch1_path=False,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    has_num_local_tokens=False,
):
    if force_batch1_path:
        alg = "batch1"
    # Gate/Up BK fallback the legacy builder reads itself; resolved here for both implementations.
    if stage == "gateup" and tile_k is None and os.environ.get("MOE_PREFILL_TILE_K"):
        tile_k = int(os.environ["MOE_PREFILL_TILE_K"])
    # Positional and routed inside the cache: this runs on every launch of tuned_moe.
    return _compile_gemm_cached(
        get_device_cache_key(),
        N,
        K,
        weight_dtype,
        weight_quant_type,
        TOPK,
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        stage,
        alg,
        E,
        USE_ATOMIC_WRITE,
        act_quant_type,
        tile_k,
        activation,
        swiglu_limit,
        situ_beta,
        situ_linear_beta,
        mxfp4_gate_up_interleaved,
        fused_down_clear,
        down_path,
        down_output_padding_bytes,
        METADATA_TILE_SIZE_M,
        has_num_local_tokens,
    )


compile_gemm.cache_clear = _compile_gemm_cached.cache_clear
compile_gemm.cache_info = _compile_gemm_cached.cache_info
