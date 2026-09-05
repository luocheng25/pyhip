"""Reproducible ATT workload for D192 noncausal direct-paged versus OPUS.

Use rocprofv3's kernel filter and iteration range to capture launches after
the first 20 warmups. OPUS receives equivalent linear KV prepared before all
launches; FlyDSL always reads the original page64 cache. This is not a timer.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from test_pa_prefill import _make_opus_call, make_case, run_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flydsl", "opus"), required=True)
    parser.add_argument("--q-len", type=int, default=10240)
    parser.add_argument("--kv-len", type=int, default=2583)
    parser.add_argument("--launches", type=int, default=23)
    args = parser.parse_args()
    case = make_case((args.q_len,), (args.kv_len,), dq=192, poison_tail=False)
    out = torch.empty(args.q_len, 16, 128, device="cuda", dtype=torch.bfloat16)
    if args.backend == "flydsl":
        def launch():
            return case.run(False, out)
    else:
        keys, values = case.logical_kv()
        launch = _make_opus_call(case, False, keys[0], values[0], out)
    props = torch.cuda.get_device_properties(0)
    source = Path(__file__).with_name("pa_8wave_950.py")
    print("ATT_WORKLOAD", json.dumps({
        "backend": args.backend, "q": args.q_len, "kv": args.kv_len,
        "dq": 192, "dv": 128, "heads": 16, "kv_heads": 1, "batch": 1,
        "page": 64, "seed": 20260905, "causal": False, "return_lse": False,
        "gpu": props.name, "arch": props.gcnArchName, "cus": props.multi_processor_count,
        "q_stride": list(case.q.stride()), "k_shape": list(case.k.shape),
        "v_shape": list(case.v.shape), "launches": args.launches,
        "direct_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layout": "original 5D paged" if args.backend == "flydsl" else "prebuilt equivalent linear, packed group mode",
    }), flush=True)
    for _ in range(args.launches):
        launch()
        torch.cuda.synchronize()
    reference, _ = run_torch(case, False)
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
    print("ATT_VALIDATED", args.backend, "max_abs_error", (out.float() - reference).abs().max().item(), flush=True)


if __name__ == "__main__":
    main()