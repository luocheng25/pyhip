"""Launch exactly one H3 attention dispatch for ATT collection."""

import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "pa_4wave"), str(HERE / "pa_8wave"), str(HERE)]

from h3_aiter_fp8 import make_asm_fp8_launcher, make_triton_fp8_launcher  # noqa: E402
from h3_paged_inputs import (  # noqa: E402
    FP8_DTYPE,
    bind_h3_kernel,
    make_h3_linear_fp8_inputs,
    make_h3_paged_inputs,
)


def make_launcher(name):
    if name in ("asm_bf16", "triton_bf16"):
        from h3_attn_kernel_test import (
            H3_CASE,
            make_inputs,
            run_asm_group,
            run_triton,
        )

        q, k, v, cu = make_inputs(H3_CASE, torch.device("cuda"))
        scale = H3_CASE.head_dim ** -0.5
        runner = run_asm_group if name == "asm_bf16" else run_triton
        return lambda: runner(q, k, v, cu, H3_CASE, scale, 1)
    if name in ("asm_fp8", "triton_fp8"):
        inputs = make_h3_linear_fp8_inputs()
        binder = make_asm_fp8_launcher if name == "asm_fp8" else make_triton_fp8_launcher
        launch, _ = binder(inputs)
        return launch
    if name in ("flydsl_8wave_fp8", "flydsl_4wave_fp8"):
        inputs = make_h3_paged_inputs(dtype=FP8_DTYPE)
        if name == "flydsl_8wave_fp8":
            from pa_prefill_8w32x32 import MHA
        else:
            from pa_prefill_4wave import MHA
        _, launch = bind_h3_kernel(MHA, inputs)
        return launch
    raise ValueError(f"unsupported H3_ATT_IMPL={name!r}")


def main():
    name = os.environ.get("H3_ATT_IMPL", "asm_fp8")
    torch.set_default_device("cuda")
    launch = make_launcher(name)
    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()
    print(f"att_target_complete,{name}", flush=True)


if __name__ == "__main__":
    main()