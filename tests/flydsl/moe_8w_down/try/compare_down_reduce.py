# SPDX-License-Identifier: MIT
"""Select down+TOPK-reduce as a whole, including inverse-map construction."""

import argparse
from functools import partial
import statistics

import torch

import test_blockscaled as comparison
from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_down_mfma32 import MFMA32_EXPERIMENT_CONFIG
from moe_multistage_pipeline import PACKED_DOWN_REDUCE_CONFIG, DownReduceWorkspace, make_down_reduce


def register_candidates():
    # One intermediate and inverse allocation across all candidates within a
    # round, not just equal-shaped buffers with different physical addresses.
    workspace = DownReduceWorkspace()
    exact32 = {**MFMA32_EXPERIMENT_CONFIG, "fold_routing": False, "defer_k1": False}
    factories = {
        "packed_selected": dict(PACKED_DOWN_REDUCE_CONFIG),
        "tuned16_torch": {**ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": "routed", "reduction": "torch"},
        "mfma32_torch": {**exact32, "down_kind": "mfma32", "output_layout": "routed", "reduction": "torch"},
        "tuned16_custom": {**ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": "routed"},
        "mfma32_custom": {**exact32, "output_layout": "routed"},
        "sorted_reference": {**exact32, "output_layout": "sorted", "reduction": "reference"},
        "tuned16_sorted": {**ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": "sorted"},
        "tuned16_packed": {**ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": "packed"},
    }
    for layout in ("sorted", "packed", "linear", "linear_raw"):
        for policy in (0, 2):
            factories[f"{layout}_c{policy}"] = {**exact32, "output_layout": layout,
                                                "output_cache_policy": policy}
    for layout in ("routed", "sorted", "packed"):
        for threads, columns in ((64, 512), (64, 2048), (64, 6144), (128, 1024), (128, 2048), (256, 2048), (256, 6144)):
            factories[f"r16_{layout}_t{threads}_n{columns}"] = {
                **ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": layout,
                "reduce_threads": threads, "reduce_cols": columns,
            }
        for label, options in {
            "nt": {"read_policy": 2}, "sc1": {"read_policy": 16},
            "sc1nt": {"read_policy": 18}, "wnt": {"write_policy": 2},
            "tree": {"tree_sum": True}, "stream": {"preload": False},
            "nttree": {"read_policy": 2, "tree_sum": True},
        }.items():
            factories[f"r16_{layout}_{label}"] = {
                **ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": layout,
                "reduce_cols": 512, **options,
            }
    for layout in ("routed", "packed"):
        for threads, columns in ((64, 512), (64, 1024), (64, 2048), (64, 6144), (128, 1024), (128, 2048), (256, 2048), (256, 6144)):
            factories[f"nt_{layout}_t{threads}_n{columns}"] = {
                **ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": layout,
                "reduce_threads": threads, "reduce_cols": columns, "read_policy": 2, "write_policy": 2,
            }
        for threads, columns in ((128, 1024), (256, 2048)):
            for policy in (2, 18):
                factories[f"read{policy}_{layout}_t{threads}"] = {
                    **ATT_TUNED_BN128_CONFIG, "down_kind": "mfma16", "output_layout": layout,
                    "reduce_threads": threads, "reduce_cols": columns, "read_policy": policy,
                }
    for key, options in factories.items():
        comparison.CANDIDATES[key] = (key, 128, partial(make_down_reduce, workspace=workspace))
        comparison.CANDIDATE_OPTIONS[key] = options
        comparison.CANDIDATE_WEIGHT_LAYOUTS[key] = (16, 16) if options.get("down_kind") == "mfma16" else (32, 16)
    return tuple(factories)


def check_candidates(candidates, n):
    from test_8stage import make_case
    for key in candidates:
        layout = comparison.CANDIDATE_WEIGHT_LAYOUTS[key]
        args, reference, _ = make_case(257, n, 256, 4, 2, 2026, "aiter", layout)
        baseline_args, baseline_reference, _ = make_case(257, n, 256, 4, 2, 2026, "aiter", (16, 16))
        baseline = flydsl_moe_gemm_8wave_down(n=n, k=256, topk=2, num_experts=4, **ATT_TUNED_BN128_CONFIG)
        baseline(*baseline_args)
        torch.cuda.synchronize()
        torch.testing.assert_close(baseline_args[0], baseline_reference, rtol=0.01, atol=0.01)
        # Validate replacement of the existing BF16 down+sum pipeline, not
        # a different FP32 accumulation tree before intermediate rounding.
        reference = baseline_args[0].sum(dim=1)
        guard = torch.full((reference.numel() + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
        output = guard[n:-n].view_as(reference)
        args = (output, *args[1:])
        pipeline = comparison.CANDIDATES[key][2](n=n, k=256, topk=2, num_experts=4,
                                               **comparison.CANDIDATE_OPTIONS[key])

        def check():
            torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
            assert torch.isfinite(output).all()
            assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()

        pipeline.poison_workspace(*args)
        pipeline(*args)
        torch.cuda.synchronize()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline(*args)
        for _ in range(3):
            output.fill_(torch.nan)
            args[-1].fill_(0x123456)
            pipeline.poison_workspace(*args)
            graph.replay()
            torch.cuda.synchronize()
            check()
        if pipeline.config["includes_inverse"] and pipeline.config["reduction"] == "custom":
            args[8].zero_()
            pipeline.poison_workspace(*args)
            output.fill_(torch.nan)
            pipeline(*args)
            torch.cuda.synchronize()
            assert torch.count_nonzero(output).item() == 0
        print(f"DOWN_REDUCE_CHECK_PASS candidate={key} N={n} graph=True", flush=True)


def main():
    keys = register_candidates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=keys, nargs="+", default=["tuned16_torch", "packed_selected"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--paired", action="store_true", help="two candidates, ABBA/BAAB")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--components", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.candidate)) != len(args.candidate):
        parser.error("positive rounds and unique candidates required")
    if args.paired and len(args.candidate) != 2:
        parser.error("paired mode requires two candidates")
    if args.check:
        check_candidates(args.candidate, args.n)
        return
    aliases = {}
    if args.paired:
        for key in args.candidate:
            alias = key + "_repeat"
            _, bn, factory = comparison.CANDIDATES[key]
            comparison.CANDIDATES[alias] = (alias, bn, factory)
            comparison.CANDIDATE_OPTIONS[alias] = comparison.CANDIDATE_OPTIONS[key]
            comparison.CANDIDATE_WEIGHT_LAYOUTS[alias] = comparison.CANDIDATE_WEIGHT_LAYOUTS[key]
            aliases[alias] = key
    measurements = {key: [] for key in args.candidate}
    components = {key: {} for key in args.candidate}
    for index in range(args.rounds):
        order = args.candidate if index % 2 == 0 else args.candidate[::-1]
        if args.paired:
            outer, inner = order
            order = [outer, inner, inner + "_repeat", outer + "_repeat"]
        results = comparison.run_test(
            tokens=16384, model_dim=args.n, inter_dim=256, experts=384, topk=8,
            block_m=256, num_oc_splits=1, seed=1234, candidates=order,
            reduce_output=True, component_diagnostics=args.components,
        )
        times = {key: [] for key in args.candidate}
        for row in results:
            assert row["status"] == "PASS", row
            key = aliases.get(row["name"], row["name"])
            times[key].append(row["elapsed_us"])
            for name, value in row["components_us"].items():
                components[key].setdefault(name, []).append(value)
        for key, values in times.items():
            elapsed = statistics.mean(values)
            measurements[key].append(elapsed)
            print(f"DOWN_REDUCE_ROUND round={index + 1} candidate={key} total_us={elapsed:.6f}", flush=True)
        if args.paired:
            a, b = (statistics.mean(times[key]) for key in args.candidate)
            print(f"DOWN_REDUCE_PAIRED round={index + 1} baseline_us={a:.6f} trial_us={b:.6f} speedup={a / b:.6f}", flush=True)
    for key, values in measurements.items():
        parts = {name: statistics.mean(values) for name, values in components[key].items()}
        print(f"DOWN_REDUCE_MEAN candidate={key} total_us={statistics.mean(values):.6f} components={parts}", flush=True)


if __name__ == "__main__":
    main()