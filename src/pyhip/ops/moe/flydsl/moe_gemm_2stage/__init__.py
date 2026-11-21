# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# pyhip: from ROCm/aiter@df3e0b2f,
# aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/__init__.py.
# compile_gemm comes from gemm.py (per-device cache; routes each config to the upstream builders
# or to the *_legacy ones) instead of upstream's uncached definition, and pyhip's
# compile_moe_reduction is exported too. sorted_sum is pyhip's per-device version (moe_reduce.py).

"""Baseline gfx9 FP8 two-stage MoE kernels."""

from .gemm import compile_gemm
from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2
from .moe_reduce import compile_moe_reduction, sorted_sum
from .quant import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
)

__all__ = [
    "compile_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "sorted_sum",
]

