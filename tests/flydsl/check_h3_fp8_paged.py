"""Check 8-wave and 4-wave FP8 paged attention on one shared H3 input."""

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "pa_4wave"), str(HERE / "pa_8wave"), str(HERE)]

from h3_paged_inputs import (  # noqa: E402
    FP8_DTYPE,
    H3_SEGMENTS,
    bind_h3_kernel,
    compare_outputs,
    h3_dequantized_sdpa_reference,
    make_h3_paged_inputs,
)
from pa_prefill_4wave import MHA as MHA4Wave  # noqa: E402
from pa_prefill_8w32x32 import MHA as MHA8Wave  # noqa: E402


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
    inputs = make_h3_paged_inputs(dtype=FP8_DTYPE)
    reference = h3_dequantized_sdpa_reference(inputs)
    outputs = {}
    for name, factory in (("8wave_varlen_fp8", MHA8Wave), ("4wave_varlen_fp8", MHA4Wave)):
        output, launch = bind_h3_kernel(factory, inputs)
        launch()
        torch.cuda.synchronize()
        scores = score_slices(reference, output)
        require_correct(name, scores)
        outputs[name] = output.clone()
        print(f"[{name}] {json.dumps(scores, sort_keys=True)}")

    mutual = score_slices(outputs["8wave_varlen_fp8"], outputs["4wave_varlen_fp8"])
    require_correct("8wave_vs_4wave", mutual)
    print(f"[8wave_vs_4wave] {json.dumps(mutual, sort_keys=True)}")


if __name__ == "__main__":
    main()