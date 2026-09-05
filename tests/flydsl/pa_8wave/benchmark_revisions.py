"""Same-input revision/scheduler comparisons and fresh ISA resource captures.

The optional baseline is a Git object (blob or revision:path), loaded only for
this benchmark. Production attention never imports a historical implementation.
"""

import argparse
from collections import Counter
import hashlib
import json
import linecache
from pathlib import Path
import re
import statistics
import subprocess
import sys
import types

import torch

from test_pa_prefill import make_case, run_torch, _gpu_dispatch
import aiter
import pandas as pd
from aiter.test_common import benchmark, checkAllclose, run_perftest
from flydsl.utils import env
import pa_8wave_950 as current


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "pa_4wave"))
from pa_prefill_4wave import MHA


def load_revision(revision):
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", revision], text=True)
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = f"/tmp/pa950_baseline_{digest}.py"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(f"pa950_baseline_{digest}")
    module.__file__ = filename
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module, digest


def read_isa(directory):
    files = list(directory.rglob("*_final_isa.s"))
    assert len(files) == 1, files
    text = files[0].read_text()
    instructions = [line.strip().split("//")[0].strip() for line in text.splitlines()
                    if re.match(r"^\s+(?:[sv]_\w+|ds_|buffer_|global_|scratch_|flat_)", line)]
    resources = {}
    for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size",
                "vgpr_spill_count", "sgpr_spill_count"):
        matches = re.findall(r"\." + key + r":\s*(\d+)", text)
        assert len(matches) == 1, (key, matches)
        resources[key] = int(matches[0])
        normalized = "\n".join(re.sub(r"\.LBB\d+_", ".LBB_", line) for line in instructions)
    return {"path": str(files[0]), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "instruction_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "resources": resources, "instruction_count": len(instructions),
            "mnemonics": dict(sorted(Counter(line.split()[0] for line in instructions).items()))}


def make_call(factory, case, causal, options):
    kernel = factory(case.heads, case.kv_heads, case.dq, 128, 64, causal,
                     window_left=case.window_left, has_sink=case.sinks is not None, **options)
    out = torch.empty(case.q.shape[0], case.heads, 128, device="cuda", dtype=torch.bfloat16)

    def call():
        return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                      max(case.q_lens), max(case.kv_lens), causal, case.qs, case.ks, case.vs,
                      case.last, out=out, sink_ptr=case.sinks)
    return call


@benchmark()
def compare(scenario, dq, q_len, kv_len, heads, candidates, baseline, dump_root, records):
    causal, window = scenario in ("causal", "swa"), 128 if scenario == "swa" else -1
    q_lens, kv_lens, kv_heads = (q_len,), (kv_len,), 1
    if scenario == "batch4":
        q_lens, kv_lens = (q_len,) * 4, (kv_len,) * 4
    elif scenario == "h3":
        q_lens, kv_lens, heads, kv_heads = (63225, 7), (63225, 7), 14, 14
    case = make_case(q_lens, kv_lens, dq=dq, heads=heads, kv_heads=kv_heads,
                     window_left=window, has_sink=window >= 0, poison_tail=False)
    reference, _ = run_torch(case, causal)
    factories = {"static": (current.PagedAttention, {}),
                 "persistent": (current.PagedAttention, {"persistent": True}),
                 "4static": (MHA, {"force_dynamic_schedule": False}),
                 "4dynamic": (MHA, {"force_dynamic_schedule": True})}
    baseline_hash = None
    if baseline:
        module, baseline_hash = load_revision(baseline)
        factories["baseline"] = (module.PagedAttention, {})
    calls, isa, errors, dispatch = {}, {}, {}, {}
    for name in candidates:
        if name == "4static" and len(q_lens) != 1:
            raise ValueError("4-wave static only supports batch1")
        factory, options = factories[name]
        call = make_call(factory, case, causal, options)
        if dump_root:
            directory = Path(dump_root) / f"{scenario}_{dq}_{q_len}_{kv_len}_{heads}_{name}"
            env.debug.dump_ir, env.debug.dump_dir = True, str(directory)
        output = call()
        torch.cuda.synchronize()
        if dump_root:
            isa[name] = read_isa(directory)
            env.debug.dump_ir = False
        errors[name] = float(checkAllclose(reference, output.float(), rtol=0.02, atol=0.02,
                                           tol_err_ratio=0, msg=name))
        assert errors[name] == 0, (scenario, dq, name)
        first = output.clone()
        for _ in range(2):
            torch.testing.assert_close(call(), first, rtol=0, atol=0)
        calls[name] = call
        dispatch[name] = _gpu_dispatch(call)
        assert len(dispatch[name]) == 1, dispatch[name]
    for _ in range(100):
        for call in calls.values():
            call()
    torch.cuda.synchronize()
    samples = {name: [] for name in calls}
    for trial in range(5):
        names = list(calls) if trial % 2 == 0 else list(reversed(calls))
        for name in names:
            _, us = run_perftest(calls[name], num_warmup=20, num_iters=100, num_rotate_args=1)
            samples[name].append(float(us))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    pairs, visible_kv = 0, 0
    for q, k in zip(q_lens, kv_lens):
        pairs += q * k if not causal else sum(max(0, min(k, k - q + r + 1)
                    - (max(0, k - q + r - window) if window >= 0 else 0)) for r in range(q))
        visible_kv += k - (max(0, (k - q - window) // 64) * 64 if window >= 0 else 0)
    flops = 2 * heads * pairs * (dq + 128)
    nbytes = 2 * (sum(q_lens) * heads * (dq + 128) + visible_kv * kv_heads * (dq + 128))
    record = {"scenario": scenario, "dq": dq, "q_lens": q_lens, "kv_lens": kv_lens,
              "heads": heads, "kv_heads": kv_heads, "causal": causal, "window_left": window,
              "sink": window >= 0, "timing_us": samples, "median_us": medians, "errors": errors,
              "dispatch": dispatch, "isa": isa, "effective_flops": flops, "logical_bytes": nbytes,
              "baseline": baseline, "baseline_source_sha256": baseline_hash,
              "source_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()}
    records.append(record)
    print("REVISION_RESULT", json.dumps(record), flush=True)
    ret = {}
    for name, us in medians.items():
        ret.update({f"{name} us": us, f"{name} TFLOPS": flops / us / 1e6,
                    f"{name} TB/s": nbytes / us / 1e6, f"{name} err": errors[name]})
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", choices=("nc", "causal", "swa", "batch4", "h3"), default=["nc", "causal", "swa"])
    parser.add_argument("--dq", type=int, nargs="+", default=[128, 192])
    parser.add_argument("--q", type=int, nargs="+")
    parser.add_argument("--kv", type=int, nargs="+")
    parser.add_argument("--heads", type=int, nargs="+", default=[16])
    parser.add_argument("--candidate", nargs="+", choices=("baseline", "static", "persistent", "4static", "4dynamic"), default=["static"])
    parser.add_argument("--baseline", help="Git blob or revision:path for a same-process baseline")
    parser.add_argument("--dump-root", help="fresh compiler-dump directory; disable disk cache for ISA capture")
    parser.add_argument("--output", type=Path, help="write all raw samples and resource metadata as JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise SystemExit("requires gfx950")
    if "baseline" in args.candidate and not args.baseline:
        parser.error("baseline candidate requires --baseline")
    shapes = {"nc": (10240, 2583), "causal": (32768, 32768), "swa": (16384, 131072),
              "batch4": (10240, 2560), "h3": (63225, 63225)}
    rows, records = [], []
    for scenario in args.scenario:
        q, kv = shapes[scenario]
        for dq in args.dq:
            for q_len in args.q or [q]:
                for kv_len in args.kv or [kv]:
                    for heads in args.heads:
                        row = compare(scenario, dq, q_len, kv_len, heads, args.candidate, args.baseline, args.dump_root, records)
                        for key in ("records", "candidates", "baseline", "dump_root"):
                            row.pop(key, None)
                        rows.append(row)
                        if args.output:
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            args.output.write_text(json.dumps(records, indent=2) + "\n")
    aiter.logger.info("Revision/scheduler comparison (GPU microseconds):\n%s", pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()