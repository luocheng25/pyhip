"""Single-candidate warmed capture using the current README benchmark inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmark_readme import build_candidates, make_inputs, reference_module, workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(workloads()), default="noncausal_d192")
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--candidate", choices=("8wave_5d", "4wave_static_5d", "4wave_dynamic_5d", "opus_linear"), required=True)
    args = parser.parse_args()
    case = make_inputs(workloads()[args.case], args.dtype)
    candidates, unavailable = build_candidates(case)
    if args.candidate not in candidates:
        raise SystemExit(unavailable[args.candidate])
    fn = candidates[args.candidate]
    source_dir = Path(__file__).resolve().parent
    source = (source_dir.parent / "pa_8wave/pa_8wave_950.py" if args.candidate == "8wave_5d"
              else source_dir / "pa_prefill_4wave.py")
    print("README_PROFILE", json.dumps({
        "case": args.case, "dtype": args.dtype, "candidate": args.candidate,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if args.candidate != "opus_linear" else None,
        "launches": 23, "selected_iterations": [21, 22, 23],
    }), flush=True)
    for _ in range(23):
        actual = fn()
        torch.cuda.synchronize()
    ref, _ = reference_module.run_torch(case, case.workload.causal)
    tolerance = 0.02 if args.dtype == "bf16" else 0.1
    torch.testing.assert_close(actual.float(), ref, rtol=tolerance, atol=tolerance)
    print("README_PROFILE_VALIDATED", args.candidate, flush=True)


if __name__ == "__main__":
    main()