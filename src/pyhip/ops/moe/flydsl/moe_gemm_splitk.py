# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility facade for the refactored two-stage MoE kernels."""

from .moe_gemm_2stage import (
    compile_gemm,
    flydsl_absmax,
    flydsl_quant_per_tensor,
    sorted_sum,
)

__all__ = [
    "compile_gemm",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "sorted_sum",
]
