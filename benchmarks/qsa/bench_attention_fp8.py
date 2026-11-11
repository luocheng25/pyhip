"""FP8 K/V QSA prefill against BF16 PyHIP and the Triton implementations on identical inputs.

Scopes, all full calls on the same synthetic Q/K/V and selection (HK1, one request):
- pyhip_bf16: attention() with BF16 K/V (the production path).
- pyhip_fp8_tensor / pyhip_fp8_token: attention() with E4M3FNUZ K/V and per-tensor / per-token
  scales, compute="fp8" (Q quantization, FP8 union, FP8 context pack and direct).
- pyhip_fp8_bf16: per-token FP8 K/V with compute="bf16" (dequantize, then the BF16 path).
- sglang_triton_bf16: SGLang's native QSA extend Triton kernel (the op test's copy), prebuilt
  arguments.
- aiter_bf16 / aiter_fp8_token: AITER #7008's Triton sparse paged GQA (vendored), K/V in
  64-token pages in order.

Each timed output is compared bit for bit with the scope's validated output; validated outputs
match CPU FP32 reference rows (on the dequantized K/V for FP8): .02/.02, or for compute="fp8" the
norm-relative error bounds of tests/ops/qsa/test_attention_fp8.py. Original cudaPerf, 10 input
buffers, rotated and alternated scope order. The GPU must be idle on entry; hardware snapshots
before sampling and at the end are kept. PyHIP scopes also get captured per-kernel timings.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gr_read.bench_gr_read_compare import read_hardware, validate_hardware
from benchmarks.qsa._synthetic import make_case, reference_rows
from pyhip.testing.misc import cudaPerf
from tests.ops.qsa import sparse_prefill_op_test as op
from tests.ops.qsa._aiter_sparse_gqa import sparse_paged_gqa
from tests.ops.qsa._attention import _make_case, _runtime
from tests.ops.qsa._benchmark import (
    DATA, ROOT, capture_kernels, default_output, format_table, measure_kernels,
    recording_matrix, source_files, tensor_address,
)
from tests.ops.qsa.test_attention_fp8 import FP8_TOL

# The op test makes CUDA the default device on import.
torch.set_default_device(None)

FP8 = torch.float8_e4m3fnuz
PAGE = 64
SCOPES = ("pyhip_bf16", "pyhip_fp8_tensor", "pyhip_fp8_token", "pyhip_fp8_bf16",
          "sglang_triton_bf16", "aiter_bf16", "aiter_fp8_token")
KERNEL_SCOPES = ("pyhip_bf16", "pyhip_fp8_tensor", "pyhip_fp8_token")
# Scope -> K/V the reference uses: BF16, or dequantized per-tensor / per-token FP8.
REFERENCE = dict(pyhip_bf16="bf16", sglang_triton_bf16="bf16", aiter_bf16="bf16",
                 pyhip_fp8_tensor="tensor", pyhip_fp8_token="token", pyhip_fp8_bf16="token",
                 aiter_fp8_token="token")
CASES = {
    # Every row a complete causal prefix: union only.
    "dense2048": lambda heads: make_case(2048, heads)[0],
    # Synthetic 12k prefill (complete prefix, high- and low-sharing windows).
    "m12000": lambda heads: make_case(12000, heads)[0],
    # Independent random selections over a 128k context: direct.
    "random128k": lambda heads: _make_case((4096,), (126976,), heads, torch.device("cpu")),
}


def _quantize(x, per_token):
    x = x.float()
    amax = x.abs().amax(-1) if per_token else x.abs().amax().reshape(1)
    scale = (amax / 240.0).clamp_min(1e-12)
    x8 = (x / (scale[..., None] if per_token else scale)).clamp(-240.0, 240.0).to(FP8)
    return x8, scale.contiguous()


def _dequant(x8, scale):
    return x8.float() * (scale[..., None] if scale.ndim == 2 else scale)


def _paged(x):
    """[pages, PAGE, ...] copy of x with its rows in order (zero padding)."""
    pages = -(-x.shape[0] // PAGE)
    out = torch.zeros((pages * PAGE, *x.shape[1:]), dtype=x.dtype, device=x.device)
    out[: x.shape[0]] = x
    return out.view(pages, PAGE, *x.shape[1:])


def _inputs(host, quantized, device):
    """One buffer: fresh device copies of every scope's inputs and outputs."""
    def copy(tensor):
        return tensor.to(device, copy=True)

    q, k, v, indices = (copy(getattr(host, n)) for n in ("q", "k", "v", "indices"))
    fp8 = {mode: tuple(copy(t) for t in tensors) for mode, tensors in quantized.items()}
    token = fp8["token"]
    paged = dict(
        bf16=(_paged(k), _paged(v), None, None),
        token=(_paged(token[0].view(torch.uint8)).view(FP8), _paged(token[2].view(torch.uint8)).view(FP8),
               _paged(token[1]), _paged(token[3])),
    )
    pages = paged["bf16"][0].shape[0]
    table = torch.arange(pages, dtype=torch.int32, device=device)[None]
    owner = torch.zeros(q.shape[0], dtype=torch.int32, device=device)
    with torch.device(device):
        triton_fn, triton_args = op.sglang_triton(q, k, v, indices, host.query_lens, host.prefix_lens)
    outputs = {scope: torch.empty_like(q) for scope in SCOPES}
    lens = dict(query_lens=host.query_lens, prefix_lens=host.prefix_lens, softmax_scale=host.scale)

    def pyhip(scope, kv, **kwargs):
        return lambda: _runtime.attention(q, *kv, indices, out=outputs[scope], **lens, **kwargs)

    def aiter(scope, mode):
        kc, vc, ks, vs = paged[mode]
        return lambda: sparse_paged_gqa(q, kc, vc, indices, table, owner, host.scale,
                                        k_scale=ks, v_scale=vs, out=outputs[scope])

    tensor = fp8["tensor"]
    calls = dict(
        pyhip_bf16=pyhip("pyhip_bf16", (k, v)),
        pyhip_fp8_tensor=pyhip("pyhip_fp8_tensor", (tensor[0], tensor[2]),
                               k_scale=tensor[1], v_scale=tensor[3]),
        pyhip_fp8_token=pyhip("pyhip_fp8_token", (token[0], token[2]),
                              k_scale=token[1], v_scale=token[3]),
        pyhip_fp8_bf16=pyhip("pyhip_fp8_bf16", (token[0], token[2]),
                             k_scale=token[1], v_scale=token[3], compute="bf16"),
        # SGLang's interface allocates its output.
        sglang_triton_bf16=lambda: triton_fn(*triton_args),
        aiter_bf16=aiter("aiter_bf16", "bf16"),
        aiter_fp8_token=aiter("aiter_fp8_token", "token"),
    )
    addresses = {name: tensor_address(t) for name, t in
                 dict(q=q, k=k, v=v, indices=indices, k8_tensor=tensor[0], k8_token=token[0]).items()}
    return SimpleNamespace(calls=calls, outputs=outputs, addresses=addresses,
                           output_addresses={s: tensor_address(o, output=True) for s, o in outputs.items()})


def _check(scope, actual, expected):
    actual = actual.float()
    assert bool(torch.isfinite(actual).all())
    if scope in ("pyhip_fp8_tensor", "pyhip_fp8_token"):
        error = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
        assert float(error.mean()) < FP8_TOL[0] and float(error.max()) < FP8_TOL[1], scope
        return dict(mean_rel=float(error.mean()), max_rel=float(error.max()))
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02, msg=scope)
    error = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
    return dict(mean_rel=float(error.mean()), max_rel=float(error.max()))


def _host_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def benchmark(case, tp_size, folder, gpu, *, buffers=10, samples=128, check_only=False, scopes=SCOPES):
    folder = Path(folder)
    if tp_size not in (2, 4, 8) or buffers != 10 or samples < buffers:
        raise ValueError("Require TP2/4/8, 10 buffers and samples >= 10")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    sources = source_files()

    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    if "pyhip_bf16" not in scopes:
        raise ValueError("pyhip_bf16 is the speedup baseline")
    report = dict(complete=False, raw=[], summary={}, gpu=gpu, tp_size=tp_size, case=case, scopes=list(scopes),
                  buffers=buffers, samples=samples, check_only=check_only, source_sha256=hashes(),
                  hardware={}, accuracy={},
                  scope="full calls; PyHIP via attention() (host checks, plan lookup, K1-K3, compute); "
                        "Triton with prebuilt arguments; references, allocation of inputs and JIT excluded")
    try:
        # The process entry gate (main) checks idleness; per job only snapshots.
        report["hardware"]["entry"] = read_hardware(gpu)
        device = torch.device("cuda", gpu)
        with torch.cuda.device(gpu):
            host = CASES[case](24 // tp_size)
            heads, rows = host.q.shape[1], host.q.shape[0]
            report["shape"] = dict(rows=rows, heads=heads, kv_heads=host.k.shape[1],
                                   context=host.k.shape[0], query_lens=host.query_lens,
                                   prefix_lens=host.prefix_lens, width=host.indices.shape[1])
            quantized = {}
            for mode in ("tensor", "token"):
                k8, ks = _quantize(host.k.to(device), mode == "token")
                v8, vs = _quantize(host.v.to(device), mode == "token")
                quantized[mode] = tuple(t.cpu() for t in (k8, ks, v8, vs))
            selected = sorted(set(np.linspace(0, rows - 1, min(rows, 48), dtype=int).tolist()
                                  + [n for n in (0, 3, 31, 32, 2047, 2050, 2051, rows - 1) if n < rows]))
            references = {"bf16": reference_rows(host, selected)}
            for mode, (k8, ks, v8, vs) in quantized.items():
                exact = SimpleNamespace(**{**vars(host), "k": _dequant(k8, ks), "v": _dequant(v8, vs)})
                references[mode] = reference_rows(exact, selected)
            bundles, validated = [], {}
            for buffer in range(1 if check_only else buffers):
                bundle = _inputs(host, quantized, device)
                bundles.append(bundle)
                for scope in scopes:
                    for _ in range(3):
                        bundle.outputs[scope].fill_(float("nan"))
                        output = bundle.calls[scope]()
                    actual = output.cpu()
                    if buffer:
                        assert torch.equal(actual.view(torch.int16), validated[scope].view(torch.int16)), scope
                    else:
                        validated[scope] = actual
                        report["accuracy"][scope] = _check(scope, actual[selected], references[REFERENCE[scope]])
            report["addresses"] = [b.output_addresses for b in bundles]
            report["input_addresses"] = [b.addresses for b in bundles]
            report["input_sha256"] = {name: _host_hash(getattr(host, name)) for name in ("q", "k", "v", "indices")}
            report["output_sha256"] = {scope: _host_hash(value) for scope, value in validated.items()}
            work = int((host.indices >= 0).sum()) * 4 * heads * 256
            report["useful_flops"] = work
            torch.cuda.synchronize(gpu)
            report["hardware"]["before_sampling"] = read_hardware(gpu)
            if not check_only:
                timer = cudaPerf(name="qsa_fp8_scopes", verbose=0)
                if not timer.enable:
                    raise RuntimeError("CUDAPERF disabled timing")
                for sample in range(samples):
                    buffer = sample % buffers
                    order = scopes[sample % len(scopes):] + scopes[:sample % len(scopes)]
                    for scope in (order if sample % 2 == 0 else order[::-1]):
                        with timer:
                            output = bundles[buffer].calls[scope]()
                        us = timer.latencies[-1] * 1e6
                        assert math.isfinite(us) and us > 0
                        report["raw"].append(dict(scope=scope, sample=sample, buffer=buffer, us=us))
                        assert torch.equal(output.cpu().view(torch.int16), validated[scope].view(torch.int16)), scope
                base = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "pyhip_bf16")
                for scope in scopes:
                    times = [r["us"] for r in report["raw"] if r["scope"] == scope]
                    median = statistics.median(times)
                    report["summary"][scope] = dict(median_us=median, min_us=min(times), max_us=max(times),
                                                    effective_tflops=work / median / 1e6,
                                                    speedup_vs_pyhip_bf16=base / median)
                report["kernels"] = {}
                for scope in (s for s in KERNEL_SCOPES if s in scopes):
                    kernel_cases = []
                    try:
                        for bundle in bundles:
                            def check_kernel(actual, scope=scope):
                                assert torch.equal(actual.cpu().view(torch.int16), validated[scope].view(torch.int16))

                            kernel_cases.append(capture_kernels(
                                bundle.calls[scope], lambda b=bundle, s=scope: b.outputs[s].fill_(float("nan")),
                                check_kernel))
                        report["kernels"][scope] = measure_kernels(kernel_cases, folder / f"kernels_{scope}",
                                                                   samples=samples)
                    finally:
                        for kernel_case in kernel_cases:
                            kernel_case.kernels.close()
            assert hashes() == report["source_sha256"]
            report["hardware"]["end"] = read_hardware(gpu)
            report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Captured workspaces stay pinned; jobs are independent.
        _runtime._workspaces.clear()
        torch.cuda.empty_cache()
        with (folder / "result.json").open("x") as stream:
            json.dump(report, stream, indent=2, default=str)
        rows = [(scope, *(report["summary"].get(scope, {}).get(key) for key in
                          ("median_us", "min_us", "max_us", "effective_tflops", "speedup_vs_pyhip_bf16")),
                 report["accuracy"].get(scope, {}).get("mean_rel"), report["accuracy"].get(scope, {}).get("max_rel"))
                for scope in scopes]
        print(f"\n{folder.name}: TP{tp_size} ({'complete' if report['complete'] else report.get('error')})\n"
              + format_table(("Scope", "Median_us", "Min_us", "Max_us", "TFLOPS", "vs_bf16", "MeanRel", "MaxRel"),
                             rows), flush=True)
        for scope, kernels in report.get("kernels", {}).items():
            print(f"  {scope}: " + ", ".join(f"{name} {values['median_us']:.1f}"
                                            for name, values in kernels.get("summary", {}).items()), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--tp-sizes", nargs="+", type=int, choices=(2, 4, 8), default=[2, 4, 8])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path, default=default_output("fp8_attention"))
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=list(SCOPES))
    parser.add_argument("--check-only", action="store_true", help="validate without timing")
    args = parser.parse_args(argv)
    if args.samples < 10 or args.gpu < 0:
        parser.error("Require gpu >= 0 and samples >= 10")
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Results must stay under mytest/mydata")
    entry = read_hardware(args.gpu)
    validate_hardware(entry)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "hardware_entry.json").write_text(json.dumps(entry, indent=2, default=str))
    jobs = [(f"{case}_tp{tp}", case, tp) for case in args.cases for tp in args.tp_sizes]
    with recording_matrix(args.output, [name for name, _, _ in jobs], check_only=args.check_only) as reports:
        for name, case, tp in jobs:
            result = benchmark(case, tp, args.output / name, args.gpu, samples=args.samples,
                               check_only=args.check_only, scopes=tuple(args.scopes))
            reports[name] = result
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
