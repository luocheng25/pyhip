"""Check AITER Triton and ASM FP8 attention on the shared H3 input."""

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "pa_4wave"), str(HERE)]

from h3_aiter_fp8 import make_asm_fp8_launcher, make_triton_fp8_launcher  # noqa: E402
from h3_paged_inputs import (  # noqa: E402
    H3_SEGMENTS,
    compare_outputs,
    h3_dequantized_sdpa_reference,
    make_h3_linear_fp8_inputs,
)


SUPPORTED_IMPLS = {
    "triton_fp8": make_triton_fp8_launcher,
    "asm_mi308_fp8": make_asm_fp8_launcher,
    "asm_mi300_fp8": make_asm_fp8_launcher,
}


def score_slices(reference, candidate):
    split = H3_SEGMENTS[0]
    return {
        "whole": compare_outputs(reference, candidate),
        "main": compare_outputs(reference[:split], candidate[:split]),
        "tail": compare_outputs(reference[split:], candidate[split:]),
    }


def require_correct(name, scores):
    for section in ("whole", "main", "tail"):
        score = scores[section]
        if not score["finite"]:
            raise AssertionError(f"{name}/{section} contains non-finite values")
        if score["cosine"] <= 0.998:
            raise AssertionError(
                f"{name}/{section} cosine={score['cosine']:.9f} <= 0.998"
            )
        if score["rel_l2"] >= 0.06:
            raise AssertionError(
                f"{name}/{section} rel_l2={score['rel_l2']:.6f} >= 0.06"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impls",
        default="triton_fp8,asm_mi308_fp8",
        help="comma-separated implementations",
    )
    args = parser.parse_args()
    selected = args.impls.split(",")
    unknown = sorted(set(selected) - set(SUPPORTED_IMPLS))
    if unknown:
        parser.error(f"unsupported implementations: {','.join(unknown)}")

    torch.set_default_device("cuda")
    inputs = make_h3_linear_fp8_inputs()
    reference = h3_dequantized_sdpa_reference(inputs)
    for name in selected:
        launch, get_output = SUPPORTED_IMPLS[name](inputs)
        launch()
        torch.cuda.synchronize()
        output = get_output()
        scores = score_slices(reference, output)
        require_correct(name, scores)
        print(
            f"{name} dtype={output.dtype} shape={tuple(output.shape)} "
            f"{json.dumps(scores, sort_keys=True)}"
        )


if __name__ == "__main__":
    main()