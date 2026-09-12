# SPDX-License-Identifier: MIT
"""ABBA same-buffer check of the full archived winner versus the lean parent."""

import importlib.util
from pathlib import Path
import statistics
import sys

import test_blockscaled as comparison
import moe_multistage_pipeline as archived_pipeline


def load_parent_pipeline():
    """Resolve the parent's sibling imports without polluting archive imports."""
    names = ("moe_multistage_down", "moe_multistage_reduce", "moe_multistage_pipeline")
    saved = {name: sys.modules.get(name) for name in names}
    root = Path(__file__).resolve().parent.parent
    loaded = {}
    try:
        for name in names:
            alias = "winner_cleanup_" + name
            spec = importlib.util.spec_from_file_location(alias, root / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded["moe_multistage_pipeline"]


def main():
    cleaned_pipeline = load_parent_pipeline()
    workspace = archived_pipeline.DownReduceWorkspace()

    def archived(*, n, k, topk, num_experts, **unused):
        return archived_pipeline.compile_packed_down_reduce(
            n=n, k=k, topk=topk, num_experts=num_experts, workspace=workspace,
        )

    def cleaned(*, n, k, topk, num_experts, **unused):
        return cleaned_pipeline.compile_packed_down_reduce(
            n=n, k=k, topk=topk, num_experts=num_experts, workspace=workspace,
        )

    order = ("archive_before", "cleaned_before", "cleaned_after", "archive_after")
    for name in order:
        comparison.CANDIDATES[name] = (name, 128, archived if name.startswith("archive") else cleaned)
    rows = comparison.run_test(tokens=16384, model_dim=6144, inter_dim=256,
                               experts=384, topk=8, block_m=256, num_oc_splits=1,
                               seed=1234, candidates=order, reduce_output=True)
    print(f"ARCHIVED_SOURCE {archived_pipeline.__file__}")
    print(f"CLEANED_SOURCE {cleaned_pipeline.__file__}")
    for prefix in ("archive", "cleaned"):
        samples = [row for row in rows if row["name"].startswith(prefix)]
        assert len(samples) == 2 and all(row["status"] == "PASS" for row in samples)
        down = statistics.mean(row["down_elapsed_us"] for row in samples)
        total = statistics.mean(row["elapsed_us"] for row in samples)
        print(f"CLEANUP_MEAN candidate={prefix} down_us={down:.6f} total_us={total:.6f} checks={len(samples)}")


if __name__ == "__main__":
    main()