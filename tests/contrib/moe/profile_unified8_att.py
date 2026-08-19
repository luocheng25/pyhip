#!/usr/bin/env python3
"""为rocprofv3 ATT采集运行6次统一8-wave N512 down dispatch。"""

import argparse
import importlib.util
from pathlib import Path

import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import flydsl.compiler as flyc
import flydsl.expr as fx

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODULE = REPO_ROOT / "src/contrib/flydsl/moe_gemm_splitk.py"
BATCH, TOPK, N, K, EXPERTS, BLOCK_M = 32768, 9, 4096, 384, 193, 64
FP8 = torch.float8_e4m3fnuz
_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
}


def load_module(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(
        "pyhip.contrib.flydsl.moe_gemm_splitk_profile_n512", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ptr(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--dispatches", type=int, default=6)
    parser.add_argument(
        "--path",
        choices=("physical4", "unified8"),
        default="unified8",
    )
    parser.add_argument("--exact-valid-grid", action="store_true")
    args = parser.parse_args()
    if args.dispatches < 5:
        parser.error("--dispatches must be at least 5 for the checked-in ATT YAML")

    module = load_module(args.module)
    torch.manual_seed(20260817)
    useful_rows = BATCH * TOPK
    topk_ids = (
        torch.arange(useful_rows, dtype=torch.int32, device="cuda")
        .remainder(EXPERTS)
        .reshape(BATCH, TOPK)
    )
    topk_weights = torch.ones(BATCH, TOPK, dtype=torch.float32, device="cuda")
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        EXPERTS,
        N,
        torch.bfloat16,
        BLOCK_M,
        None,
        None,
        0,
    )
    grid = sorted_expert_ids.shape[0]
    padded_rows = int(num_valid_ids[0].item())
    task_num = padded_rows // BLOCK_M if args.exact_valid_grid else grid
    activation = torch.ones(BATCH, TOPK, K, dtype=FP8, device="cuda")
    weight = shuffle_weight(
        torch.ones(EXPERTS, N, K, dtype=FP8, device="cuda"), layout=(16, 16)
    )
    output = torch.empty(grid * BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    weight_scale = torch.ones(EXPERTS, dtype=torch.float32, device="cuda")
    activation_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    common = dict(
        N=N,
        K=K,
        weight_dtype="fp8",
        weight_quant_type="per_tensor",
        act_quant_type="per_tensor",
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_M,
        stage="down",
        alg="prefill_1x4",
        E=EXPERTS,
        USE_ATOMIC_WRITE=False,
        down_output_padding_bytes=0,
    )
    launch = (
        module.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=512,
            down_physical_n512=True,
        )
        if args.path == "unified8"
        else module.compile_gemm(
            **common,
            BLOCK_TILE_SIZE_N=256,
            down_physical_n256=True,
        )
    )
    stream = torch.cuda.current_stream()
    for _ in range(args.dispatches):
        launch(
            ptr(activation),
            ptr(weight),
            ptr(output),
            ptr(sorted_ids),
            ptr(sorted_weights),
            ptr(sorted_expert_ids),
            ptr(num_valid_ids),
            ptr(weight_scale),
            ptr(activation_scale),
            BATCH,
            task_num,
            stream,
        )
    torch.cuda.synchronize()
    if not torch.isfinite(output[:padded_rows]).all():
        raise AssertionError("profile output contains non-finite values")
    print(
        f"module={args.module.resolve()} path={args.path} grid={grid} "
        f"task_num={task_num} "
        f"padded_rows={padded_rows} dispatches={args.dispatches}"
    )


if __name__ == "__main__":
    main()
