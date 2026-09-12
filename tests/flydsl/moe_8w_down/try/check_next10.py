# SPDX-License-Identifier: MIT
"""Small numerical/graph checks for next-10-percent down experiments."""

import argparse
import torch

from test_8stage import make_case
import test_blockscaled as comparison
from tune_next10 import register_candidates


def main():
    keys = register_candidates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=keys, nargs="+", default=list(keys))
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    for key in args.candidate:
        layout = comparison.CANDIDATE_WEIGHT_LAYOUTS.get(key, (16, 16))
        tensors, reference, guard = make_case(257, args.n, 256, 4, 2, 2026, "aiter", layout)
        kernel = comparison.CANDIDATES[key][2](
            n=args.n, k=256, topk=2, num_experts=4,
            **comparison.CANDIDATE_OPTIONS[key],
        )
        kernel(*tensors)
        torch.cuda.synchronize()
        torch.testing.assert_close(tensors[0], reference, rtol=0.01, atol=0.01)
        assert torch.isfinite(tensors[0]).all()
        assert torch.isnan(guard[:args.n]).all() and torch.isnan(guard[-args.n:]).all()
        if args.graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                kernel(*tensors)
            for _ in range(3):
                tensors[0].fill_(torch.nan)
                tensors[-1].fill_(0x123456)
                capture.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(tensors[0], reference, rtol=0.01, atol=0.01)
                assert torch.isfinite(tensors[0]).all()
                assert torch.isnan(guard[:args.n]).all() and torch.isnan(guard[-args.n:]).all()
        print(f"CHECK_PASS candidate={key} n={args.n} layout={layout} graph={args.graph} config={kernel.config}", flush=True)


if __name__ == "__main__":
    main()