"""Exploratory check and timing of the pair prototype against production QSA.

python -m experiments.attention.flydsl.qsa.pair.bench [--m 12000 --heads 12 --delta 100]
python -m experiments.attention.flydsl.qsa.pair.bench --capture CAPTURE.pt --tp 2
One buffer set with alternating order, not the formal protocol behind ../try.md.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from pyhip.testing.misc import cudaPerf

from .. import direct, test_qsa
from ..qsa import _Workspace, qsa
from . import prepare, qsa_all_pair, rebuild, run


def sliding_indices(m, delta, seed):
    """Each sparse query keeps 512 random blocks; each next query swaps `delta` of them."""
    rng = np.random.default_rng(seed)
    rows = np.full((m, 2051), -1, dtype=np.int32)
    current = None
    for i in range(m):
        visible = i + 1
        blocks = visible // 4
        if blocks <= 512:
            chosen = np.arange(blocks)
            current = None
        else:
            if current is None:
                current = set(rng.choice(blocks, 512, replace=False).tolist())
            else:
                out = rng.choice(np.array(sorted(current)), min(delta, len(current)), replace=False)
                current.difference_update(out.tolist())
                while len(current) < 512:
                    candidate = int(rng.integers(0, blocks))
                    if candidate not in current:
                        current.add(candidate)
            chosen = np.fromiter(current, dtype=np.int64)
            rng.shuffle(chosen)
        tokens = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(blocks * 4, visible)))
        rows[i, :len(tokens)] = tokens
    return rows


def synthetic(m, heads, delta, device, seed=17):
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((m, heads, 256), generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn((m, 1, 256), generator=generator, device=device, dtype=q.dtype)
    v = torch.randn(k.shape, generator=generator, device=device, dtype=q.dtype)
    indices = torch.from_numpy(sliding_indices(m, delta, seed=delta)).to(device)
    return test_qsa._metadata(q, k, v, indices, (m,), (0,))


def timing(calls, samples):
    """Median us per call and median paired ratio to the first call; order alternates per sample."""
    timer = cudaPerf(name="pair", verbose=0)
    names = list(calls)
    raw = {name: [] for name in names}
    for sample in range(samples):
        for name in names if sample % 2 == 0 else names[::-1]:
            with timer:
                calls[name]()
            raw[name].append(timer.latencies[-1] * 1e6)
    base = raw[names[0]]
    return {name: dict(median_us=round(statistics.median(values), 1),
                       ratio=round(statistics.median(b / a for a, b in zip(base, values)), 4))
            for name, values in raw.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, default=12000)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--delta", type=int, default=100, help="blocks swapped between adjacent queries")
    parser.add_argument("--capture", type=Path, help="TP2 capture, e.g. under mytest/mydata/qsa_real_study_20260925")
    parser.add_argument("--tp", type=int, default=2, help="local-head slice of the TP2 capture")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    if args.capture:
        value = test_qsa._tp_case(test_qsa._load(args.capture, device), args.tp)
    else:
        value = synthetic(args.m, args.heads, args.delta, device)
    workspace = _Workspace(value.q, value.k, value.v, value.indices, value.query_lens, value.prefix_lens, value.scale)
    if workspace.union is None:
        raise SystemExit("Every row is dense; nothing for the pair kernel")
    inputs = workspace.bind(value.q, value.k, value.v, value.indices)
    plan = prepare(inputs, workspace.union, gated=False)
    forced = direct.prepare(inputs=inputs, skip_counts=workspace.dense.query_counts, union=None)
    outs = {name: torch.full_like(value.q, float("nan")) for name in ("direct", "pair", "qsa", "qsa_all_pair")}
    kernels = {
        "direct": lambda: direct.run(inputs=inputs, prepared=forced, out=outs["direct"]),
        "pair": lambda: (rebuild(inputs, plan), run(inputs, plan, outs["pair"])),
    }
    calls = {
        "qsa": lambda: qsa(value.q, value.k, value.v, value.indices, query_lens=value.query_lens,
                           prefix_lens=value.prefix_lens, softmax_scale=value.scale, out=outs["qsa"]),
        "qsa_all_pair": lambda: qsa_all_pair(workspace, inputs, plan, outs["qsa_all_pair"]),
    }
    # qsa_all_pair recovers the blocks and membership that the kernel-only calls reuse.
    for call in (*calls.values(), *kernels.values()):
        call()
    torch.cuda.synchronize()

    begin = workspace.dense.query_counts[0]
    rows = [row for row in test_qsa._rows(value) if row >= begin]
    reference = test_qsa.reference(value, rows)
    errors = {}
    for name, out in outs.items():
        errors[name] = (out[rows].float() - reference).abs().max().item()
        torch.testing.assert_close(out[rows].float(), reference, rtol=.02, atol=.02)
    torch.testing.assert_close(outs["pair"][begin:], outs["direct"][begin:], rtol=.02, atol=.02)
    torch.testing.assert_close(outs["qsa_all_pair"], outs["qsa"], rtol=.02, atol=.02)
    if value.captured is not None:
        torch.testing.assert_close(outs["qsa_all_pair"], value.captured, rtol=.02, atol=.02)

    meta, common = plan.metadata.cpu().numpy(), plan.common.cpu().numpy()
    selected = sum(min((position + j + 1) // 4, 512) for _, count, _, _, position in meta for j in range(count))
    print(json.dumps(dict(
        rows=value.q.shape[0], heads=value.q.shape[1], dense_rows=begin,
        shared_fraction=round(float((common * meta[:, 1]).sum()) / selected, 3),
        max_error_vs_reference=errors,
        kernels=timing(kernels, args.samples), calls=timing(calls, args.samples)), indent=1))


if __name__ == "__main__":
    main()
