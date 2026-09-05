"""Capture one warmed direct-paged scheduler for ATT/PMC; not a timing tool."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmark_revisions import current, load_revision, make_call, MHA
from test_pa_prefill import make_case, run_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("static", "persistent", "4static", "4dynamic"), required=True)
    parser.add_argument("--dq", type=int, choices=(128, 192), default=192)
    parser.add_argument("--q", type=int, default=16384)
    parser.add_argument("--kv", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--revision", help="optional Git revision:path for a before-change capture")
    args = parser.parse_args()
    case = make_case((args.q,), (args.kv,), dq=args.dq, heads=args.heads,
                     window_left=args.window, has_sink=args.window >= 0, poison_tail=False)
    module = current
    source_hash = hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()
    if args.revision:
        module, source_hash = load_revision(args.revision)
    factories = {"static": (module.PagedAttention, {}),
                 "persistent": (module.PagedAttention, {"persistent": True}),
                 "4static": (MHA, {"force_dynamic_schedule": False}),
                 "4dynamic": (MHA, {"force_dynamic_schedule": True})}
    factory, options = factories[args.candidate]
    call = make_call(factory, case, args.window >= 0, options)
    if args.candidate.startswith("4"):
        source = Path(__file__).parent.parent / "pa_4wave/pa_prefill_4wave.py"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    print("SCHEDULER_PROFILE", json.dumps({**vars(args), "source_sha256": source_hash,
          "launches": 23, "selected_iterations": [21, 22, 23]}), flush=True)
    for _ in range(23):
        actual = call()
        torch.cuda.synchronize()
    reference, _ = run_torch(case, args.window >= 0)
    torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.02)
    print("SCHEDULER_PROFILE_VALIDATED", args.candidate, flush=True)


if __name__ == "__main__":
    main()