# SPDX-License-Identifier: MIT
"""Sequential temporary output versus routed stores, including restoration."""

import argparse
import statistics

import torch

import test_blockscaled as comparison
from moe_multistage_down_mfma32 import (
    MFMA32_EXPERIMENT_CONFIG, flydsl_moe_gemm_8wave_down_mfma32,
)
from moe_multistage_restore import make_restore_output


def sequential_factory(*, output_layout, restore_by_tile=False, **options):
    gemm = flydsl_moe_gemm_8wave_down_mfma32(output_layout=output_layout, **options)
    restore = make_restore_output(
        n=options["n"], topk=options["topk"], num_oc_splits=options["num_oc_splits"],
        output_layout=output_layout, by_tile=restore_by_tile,
    )
    temporary = None

    def get_temporary(args):
        nonlocal temporary
        output, expert_ids = args[0], args[7]
        shape = (expert_ids.numel() * 256, options["n"])
        if temporary is None or temporary.shape != shape or temporary.device != output.device:
            temporary = torch.empty(shape, dtype=torch.bfloat16, device=output.device)
        return temporary

    def launch(*args):
        scratch = get_temporary(args)
        gemm(scratch, *args[1:])
        restore(args[0], scratch, args[5], args[8])
        return args[0]

    def components(*args):
        scratch = get_temporary(args)
        return {
            "gemm": lambda: gemm(scratch, *args[1:]),
            "restore": lambda: restore(args[0], scratch, args[5], args[8]),
        }

    launch.benchmark_components = components
    launch.get_temporary = get_temporary
    launch.config = {**gemm.config, "restore_by_tile": restore_by_tile, "includes_restore": True}
    return launch


def register_candidates():
    keys = ["4stage_bn128_tuned", "4stage_bn128_mfma32"]
    for layout in ("sorted", "packed", "linear", "linear_raw"):
        for cache in (0, 2):
            key = f"sequential_{layout}_c{cache}"
            keys.append(key)
            comparison.CANDIDATES[key] = (key, 128, sequential_factory)
            comparison.CANDIDATE_OPTIONS[key] = {
                **MFMA32_EXPERIMENT_CONFIG, "output_layout": layout,
                "output_cache_policy": cache,
            }
            comparison.CANDIDATE_WEIGHT_LAYOUTS[key] = (32, 16)
    for layout in ("packed", "linear", "linear_raw"):
        key = f"sequential_{layout}_tile"
        keys.append(key)
        comparison.CANDIDATES[key] = (key, 128, sequential_factory)
        comparison.CANDIDATE_OPTIONS[key] = {
            **MFMA32_EXPERIMENT_CONFIG, "output_layout": layout,
            "output_cache_policy": 2, "restore_by_tile": True,
        }
        comparison.CANDIDATE_WEIGHT_LAYOUTS[key] = (32, 16)
    return keys


def main():
    keys = register_candidates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", choices=keys, default=keys)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="small numerical and graph checks only")
    parser.add_argument("--n", type=int, default=6144)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.check:
        from test_8stage import make_case
        for key in args.candidate:
            options = comparison.CANDIDATE_OPTIONS[key]
            layout = comparison.CANDIDATE_WEIGHT_LAYOUTS.get(key, (16, 16))
            tensors, reference, guard = make_case(257, args.n, 256, 4, 2, 2026, "aiter", layout)
            kernel = comparison.CANDIDATES[key][2](n=args.n, k=256, topk=2, num_experts=4, **options)
            if hasattr(kernel, "get_temporary"):
                kernel.get_temporary(tensors).fill_(torch.nan)
            kernel(*tensors)
            torch.cuda.synchronize()
            torch.testing.assert_close(tensors[0], reference, rtol=0.01, atol=0.01)
            assert torch.isfinite(tensors[0]).all()
            assert torch.isnan(guard[:args.n]).all() and torch.isnan(guard[-args.n:]).all()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(*tensors)
            for _ in range(3):
                tensors[0].fill_(torch.nan)
                tensors[-1].fill_(0x123456)
                if hasattr(kernel, "get_temporary"):
                    kernel.get_temporary(tensors).fill_(torch.nan)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(tensors[0], reference, rtol=0.01, atol=0.01)
                assert torch.isfinite(tensors[0]).all()
                assert torch.isnan(guard[:args.n]).all() and torch.isnan(guard[-args.n:]).all()
            tensors[8].zero_()
            tensors[0].fill_(torch.nan)
            kernel(*tensors)
            torch.cuda.synchronize()
            assert torch.isnan(tensors[0]).all(), "empty routing wrote output"
            assert torch.isnan(guard[:args.n]).all() and torch.isnan(guard[-args.n:]).all()
            print(f"SEQUENTIAL_CHECK_PASS candidate={key} n={args.n} graph=True", flush=True)
        return

    measurements = {key: [] for key in args.candidate}
    components = {key: {} for key in args.candidate}
    for index in range(args.rounds):
        order = args.candidate if index % 2 == 0 else args.candidate[::-1]
        results = comparison.run_test(
            tokens=16384, model_dim=6144, inter_dim=256, experts=384, topk=8,
            block_m=256, num_oc_splits=1, seed=1234, candidates=order,
        )
        for key, row in zip(order, results):
            assert row["status"] == "PASS", row
            measurements[key].append(row["elapsed_us"])
            for name, value in row.get("components_us", {}).items():
                components[key].setdefault(name, []).append(value)
            print(f"SEQUENTIAL_ROUND round={index + 1} candidate={key} total_us={row['elapsed_us']:.6f} "
                  f"components={row.get('components_us', {})}", flush=True)
    for key, values in measurements.items():
        parts = {name: statistics.mean(values) for name, values in components[key].items()}
        print(f"SEQUENTIAL_MEAN candidate={key} total_us={statistics.mean(values):.6f} components={parts}", flush=True)


if __name__ == "__main__":
    main()