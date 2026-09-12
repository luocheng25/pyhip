# SPDX-License-Identifier: MIT
"""Targeted correctness for the opt-in MFMA32 path, including persistent reuse."""

import pytest
import torch

from moe_multistage_down_mfma32 import (
    MFMA32_EXPERIMENT_CONFIG, flydsl_moe_gemm_8wave_down_mfma32,
)
from test_8stage import make_case


@pytest.mark.parametrize("n,splits,tokens,ctas,fold", [
    (128, 1, 65, 256, True),
    (256, 1, 257, 256, True),
    (384, 1, 257, 256, True),
    (640, 1, 257, 256, True),
    (6144, 4, 513, 8, True),
    (512, 4, 129, 256, False),
])
def test_mfma32(n, splits, tokens, ctas, fold):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA32 required")
    args, reference, guard = make_case(tokens, n, 256, 8, 2, 2026, "aiter", (32, 16))
    options = {**MFMA32_EXPERIMENT_CONFIG, "num_oc_splits": splits,
               "persistent_workgroups": ctas, "fold_routing": fold, "defer_k1": fold}
    kernel = flydsl_moe_gemm_8wave_down_mfma32(n=n, k=256, topk=2, num_experts=8, **options)

    def check():
        torch.testing.assert_close(args[0], reference, rtol=0.01, atol=0.01)
        assert torch.isfinite(args[0]).all()
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()

    kernel(*args)
    torch.cuda.synchronize()
    check()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        kernel(*args)
    for _ in range(3):
        args[0].fill_(torch.nan)
        args[-1].fill_(0x123456)
        capture.replay()
        torch.cuda.synchronize()
        check()
    # A new launch must reset the queue, including for empty work.
    args[8].zero_()
    args[0].fill_(torch.nan)
    kernel(*args)
    torch.cuda.synchronize()
    assert torch.isnan(args[0]).all()
    assert args[-1].item() == ctas