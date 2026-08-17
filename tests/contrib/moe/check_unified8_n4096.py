#!/usr/bin/env python3
"""逐bit比较两个统一8-wave N512 kernel，并与PyTorch参考比较。"""

import argparse
import importlib.util
from pathlib import Path

import torch
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODULE = REPO_ROOT / "src/contrib/flydsl/moe_gemm_splitk.py"
BATCH, N, K = 64, 4096, 384
FP8 = torch.float8_e4m3fnuz
_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
}


def load_module(name: str, path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(f"loaded unexpected module: {module.__file__}")
    return module


def ptr(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def build(module):
    return module.compile_gemm(
        N=N,
        K=K,
        weight_dtype="fp8",
        weight_quant_type="per_tensor",
        act_quant_type="per_tensor",
        TOPK=1,
        BLOCK_TILE_SIZE_M=64,
        BLOCK_TILE_SIZE_N=512,
        stage="down",
        alg="prefill_1x4",
        E=1,
        USE_ATOMIC_WRITE=False,
        down_physical_n512=True,
        down_output_padding_bytes=0,
    )


def relative_l2(actual, expected):
    error = (actual.float() - expected.float()).square().sum().sqrt()
    reference = expected.float().square().sum().sqrt()
    return (error / reference).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_MODULE)
    args = parser.parse_args()

    torch.manual_seed(20260817)
    reference = load_module("pyhip.contrib.flydsl.unified8_reference", args.reference)
    candidate = load_module("pyhip.contrib.flydsl.unified8_candidate", args.candidate)

    activation_fp16 = 0.1 * torch.randn(BATCH, K, dtype=torch.float16, device="cuda")
    weight_fp16 = 0.1 * torch.randn(1, N, K, dtype=torch.float16, device="cuda")
    fp8_max = torch.finfo(FP8).max
    activation_scale = activation_fp16.float().abs().amax().reshape(1) / fp8_max
    weight_scale = weight_fp16.float().abs().amax().reshape(1) / fp8_max
    activation = (
        (activation_fp16.float() / activation_scale).clamp(-fp8_max, fp8_max).to(FP8)
    )
    weight = (weight_fp16.float() / weight_scale).clamp(-fp8_max, fp8_max).to(FP8)
    shuffled_weight = shuffle_weight(weight, layout=(16, 16))

    sorted_ids = torch.arange(BATCH, dtype=torch.int32, device="cuda")
    sorted_weights = torch.linspace(
        0.25, 1.0, BATCH, dtype=torch.float32, device="cuda"
    )
    sorted_expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.tensor([BATCH, BATCH], dtype=torch.int32, device="cuda")
    outputs = {
        name: torch.full((BATCH, N), torch.nan, dtype=torch.bfloat16, device="cuda")
        for name in ("reference", "candidate")
    }
    launches = {
        "reference": build(reference),
        "candidate": build(candidate),
    }
    stream = torch.cuda.current_stream()

    for name in outputs:
        launches[name](
            ptr(activation),
            ptr(shuffled_weight),
            ptr(outputs[name]),
            ptr(sorted_ids),
            ptr(sorted_weights),
            ptr(sorted_expert_ids),
            ptr(num_valid_ids),
            ptr(weight_scale),
            ptr(activation_scale),
            BATCH,
            1,
            stream,
        )
    torch.cuda.synchronize()

    if not torch.equal(outputs["reference"], outputs["candidate"]):
        difference = outputs["reference"] != outputs["candidate"]
        max_abs = (
            (outputs["reference"].float() - outputs["candidate"].float())
            .abs()
            .max()
            .item()
        )
        raise AssertionError(
            f"candidate mismatch count={int(difference.sum())} max_abs={max_abs}"
        )

    expected = (activation.float() * activation_scale) @ (
        weight[0].float() * weight_scale
    ).T
    expected = (expected * sorted_weights[:, None]).to(torch.bfloat16)
    rel_l2 = relative_l2(outputs["candidate"], expected)
    if not torch.isfinite(outputs["candidate"]).all():
        raise AssertionError("candidate output contains non-finite values")
    if rel_l2 >= 3e-2:
        raise AssertionError(f"rel_l2={rel_l2}")
    print(f"N={N} blocks={N // 512} physical_bit_equal=True " f"rel_l2={rel_l2:.8f}")


if __name__ == "__main__":
    main()
