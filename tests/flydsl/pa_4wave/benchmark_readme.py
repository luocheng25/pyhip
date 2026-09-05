"""Retest the README workload matrix with current kernels and explicit layouts.

All attention candidates must pass the independent chunked FP32 reference
before timing. Historical 8-wave, gather and disabled K16 variants are not
substituted for unsupported current specializations.
"""

import argparse
import importlib.util
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PA8 = HERE.parent / "pa_8wave"
sys.path.insert(0, str(PA8))
spec = importlib.util.spec_from_file_location("readme_current_attention_tests", PA8 / "test_pa_prefill.py")
reference_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference_module
spec.loader.exec_module(reference_module)

import aiter
import pandas as pd
from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, fmha_fwd_bf16_opus_varlen_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest
from pa_8wave_950 import PagedAttention
from pa_prefill_4wave import MHA


@dataclass(frozen=True)
class Workload:
    name: str
    q_lens: tuple
    kv_lens: tuple
    dq: int = 192
    heads: int = 16
    kv_heads: int = 1
    page: int = 64
    causal: bool = False
    window: int = -1


def workloads():
    cases = []
    for dq in (128, 192):
        cases += [Workload(f"noncausal_d{dq}", (10240,), (2583,), dq=dq),
                  Workload(f"causal_d{dq}", (32768,), (32768,), dq=dq, causal=True)]
        cases += [Workload(f"swa{kv // 1024}k_d{dq}", (16384,), (kv,), dq=dq, causal=True, window=128)
                  for kv in (32768, 65536, 131072)]
    cases += [Workload("page32_noncausal", (10240,), (2583,), page=32),
              Workload("page32_causal", (32768,), (32768,), page=32, causal=True)]
    for page in (32, 64):
        cases += [Workload(f"batch4_d192_p{page}", (10240,) * 4, (2560,) * 4, page=page),
                  Workload(f"batch4_d128_h1_p{page}", (10240,) * 4, (2560,) * 4, dq=128, heads=1, page=page),
                  Workload(f"singlehead_full_p{page}", (40960,), (40960,), dq=128, heads=1, page=page),
                  Workload(f"singlehead_causal_p{page}", (32768,), (32768,), dq=128, heads=1, page=page, causal=True),
                  Workload(f"h3_p{page}", (63225, 7), (63225, 7), dq=128, heads=14, kv_heads=14, page=page)]
    return {case.name: case for case in cases}


@dataclass
class Inputs:
    workload: Workload
    q: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    vs: torch.Tensor
    cq: torch.Tensor
    ck: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    sinks: torch.Tensor | None
    q_offset: int = 0

    @property
    def q_lens(self):
        return self.workload.q_lens

    @property
    def kv_lens(self):
        return self.workload.kv_lens

    @property
    def heads(self):
        return self.workload.heads

    @property
    def kv_heads(self):
        return self.workload.kv_heads

    @property
    def dq(self):
        return self.workload.dq

    @property
    def window_left(self):
        return self.workload.window

    def logical_kv(self):
        keys, values = [], []
        start = 0
        for length in self.kv_lens:
            count = math.ceil(length / self.workload.page)
            ids = self.indices[start:start + count].long()
            def select(pages):
                if pages.element_size() == 1:
                    return pages.view(torch.uint8).index_select(0, ids).view(pages.dtype)
                return pages.index_select(0, ids)
            keys.append(select(self.k_pages).reshape(-1, self.kv_heads, self.dq)[:length])
            values.append(select(self.v_pages).reshape(-1, self.kv_heads, 128)[:length])
            start += count
        return keys, values


def make_inputs(w, dtype):
    torch.manual_seed(20260905)
    total_q = sum(w.q_lens)
    counts = [math.ceil(k / w.page) for k in w.kv_lens]
    q = torch.randn(total_q, w.heads, w.dq, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(counts), w.page, w.kv_heads, w.dq, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(counts), w.page, w.kv_heads, 128, device="cuda", dtype=torch.bfloat16)
    order = torch.randperm(sum(counts)).tolist()
    start = 0
    for length, count in zip(w.kv_lens, counts):
        if length % w.page:
            k[order[start + count - 1], length % w.page:] = 0
            v[order[start + count - 1], length % w.page:] = 0
        start += count
    if dtype == "bf16":
        qs = torch.ones(total_q, w.heads, 1, device="cuda")
        ks, vs = torch.ones(1, device="cuda"), torch.ones(1, device="cuda")
    else:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        fp8 = torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz
        limit = torch.finfo(fp8).max
        qs = q.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / limit
        ks = (k.float().abs().max() / limit).reshape(1)
        vs = (v.float().abs().max() / limit).reshape(1)
        q, k, v = (q.float() / qs).to(fp8), (k.float() / ks).to(fp8), (v.float() / vs).to(fp8)
    kc, vc = reference_module.vectorize_kv_cache(k, v, w.kv_heads, w.dq, 128, w.page)
    def i32(data):
        return torch.tensor(data, device="cuda", dtype=torch.int32)
    return Inputs(w, q, k, v, kc, vc, qs, ks, vs,
                  i32(list(accumulate(w.q_lens, initial=0))),
                  i32(list(accumulate(w.kv_lens, initial=0))),
                  i32(list(accumulate(counts, initial=0))), i32(order),
                  i32([(k - 1) % w.page + 1 for k in w.kv_lens]),
                  torch.linspace(-1, 1, w.heads, device="cuda") if w.window >= 0 else None)


def build_candidates(case):
    w = case.workload
    candidates, unavailable = {}, {}
    def output():
        return torch.empty(sum(w.q_lens), w.heads, 128, device="cuda", dtype=torch.bfloat16)
    args = (case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
            max(w.q_lens), max(w.kv_lens), w.causal, case.qs, case.ks, case.vs, case.last)
    if case.q.dtype == torch.bfloat16 and w.page == 64:
        kernel = PagedAttention(w.heads, w.kv_heads, w.dq, 128, w.page, w.causal,
                                window_left=w.window, has_sink=case.sinks is not None)
        out = output()
        candidates["8wave_5d"] = lambda kernel=kernel, out=out: kernel(*args, out=out, sink_ptr=case.sinks)
    else:
        unavailable["8wave_5d"] = "current 8-wave supports BF16 page64 only; no historical fallback"
    # B>1 has no static 4-wave path: do not call the default dynamic route static.
    schedules = (False, True) if len(w.q_lens) == 1 else (True,)
    if len(w.q_lens) > 1:
        unavailable["4wave_static_5d"] = "4-wave static scheduling supports batch1 only"
    for dynamic in schedules:
        kernel = MHA(w.heads, w.kv_heads, w.dq, 128, w.page, w.causal,
                     window_left=w.window, has_sink=case.sinks is not None, force_dynamic_schedule=dynamic)
        out = output()
        candidates["4wave_dynamic_5d" if dynamic else "4wave_static_5d"] = (
            lambda kernel=kernel, out=out: kernel(*args, out=out, sink_ptr=case.sinks))
    if case.q.dtype != torch.bfloat16:
        unavailable["opus_linear"] = "OPUS D128/D192 only supports BF16"
        unavailable["aiter_linear"] = "not compared: README FP8 contract uses per-token Q scaling, no equivalent linear baseline configured"
        unavailable["aiter_5d"] = "not compared: per-token FP8 Q scaling is not the batch-prefill per-tensor scale contract"
        return candidates, unavailable
    keys, values = case.logical_kv()
    k, v = torch.cat(keys), torch.cat(values)
    linear_out, paged_out = output(), output()
    candidates["aiter_linear"] = lambda: aiter.flash_attn_varlen_func(
        case.q, k, v, case.cq, case.ck, max(w.q_lens), max(w.kv_lens), causal=w.causal,
        window_size=(w.window, -1, 0), sink_ptr=case.sinks, out=linear_out)
    candidates["aiter_5d"] = lambda: aiter.mha_batch_prefill_func(
        case.q, case.k, case.v, case.cq, case.indptr, case.indices,
        max(w.q_lens), max(w.kv_lens), causal=w.causal, window_size=(w.window, -1),
        sink_ptr=case.sinks, kv_last_page_lens=case.last, out=paged_out)
    if w.window >= 0:
        unavailable["opus_linear"] = "OPUS does not support SWA or sink"
    elif w.dq == 128 and (len(set(w.q_lens)) != 1 or len(set(w.kv_lens)) != 1):
        unavailable["opus_linear"] = "OPUS D128 has no packed ragged entry"
    else:
        opus_out = output()
        if w.dq == 192:
            def opus():
                fmha_fwd_bf16_opus_varlen_fwd(case.q, k, v, w.dq**-0.5, w.causal,
                                             case.cq, case.ck, max(w.q_lens), max(w.kv_lens), out=opus_out)
                return opus_out
        else:
            batch = len(w.q_lens)
            q4 = case.q.view(batch, w.q_lens[0], w.heads, w.dq)
            k4 = k.view(batch, w.kv_lens[0], w.kv_heads, w.dq)
            v4 = v.view(batch, w.kv_lens[0], w.kv_heads, 128)
            out4 = opus_out.view(batch, w.q_lens[0], w.heads, 128)
            def opus():
                fmha_fwd_bf16_opus_fwd(q4, k4, v4, w.dq**-0.5, w.causal, out=out4)
                return opus_out
        candidates["opus_linear"] = opus
    return candidates, unavailable


@benchmark()
def benchmark_case(case_name, dtype):
    w = workloads()[case_name]
    case = make_inputs(w, dtype)
    reference, _ = reference_module.run_torch(case, w.causal)
    candidates, unavailable = build_candidates(case)
    tolerance = 0.02 if dtype == "bf16" else 0.1
    errors, max_errors, dispatch = {}, {}, {}
    for name, fn in list(candidates.items()):
        try:
            actual = fn()
            torch.cuda.synchronize()
        except RuntimeError as error:
            if name != "aiter_5d" or "no matching kernel found" not in str(error):
                raise
            unavailable[name] = str(error)
            del candidates[name]
            continue
        errors[name] = float(checkAllclose(reference, actual.float(), rtol=tolerance, atol=tolerance, tol_err_ratio=0, msg=name))
        assert errors[name] == 0, f"{case_name}/{dtype}/{name}: numerical validation failed"
        max_errors[name] = (actual.float() - reference).abs().max().item()
        first = actual.clone()
        for _ in range(2):
            torch.testing.assert_close(fn(), first, rtol=0, atol=0)
        dispatch[name] = reference_module._gpu_dispatch(fn)
        if name == "opus_linear":
            assert any(f"gqa_d{w.dq}" in n for n in dispatch[name]), dispatch[name]
        if name == "8wave_5d":
            assert len(dispatch[name]) == 1 and "_attention_kernel" in dispatch[name][0], dispatch[name]
    for name, reason in unavailable.items():
        aiter.logger.warning("%s/%s %s N/A: %s", case_name, dtype, name, reason)
    for _ in range(100):
        for fn in candidates.values():
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in candidates}
    for trial in range(5):
        names = list(candidates) if trial % 2 == 0 else list(reversed(candidates))
        for name in names:
            _, us = run_perftest(candidates[name], num_warmup=20, num_iters=100, num_rotate_args=1)
            samples[name].append(float(us))
    pairs, visible_kv = 0, 0
    for q, k in zip(w.q_lens, w.kv_lens):
        pairs += (q * k if not w.causal else sum(max(0, min(k, k - q + r + 1)
                  - (max(0, k - q + r - w.window) if w.window >= 0 else 0)) for r in range(q)))
        first = max(0, (k - q - w.window) // w.page) if w.window >= 0 else 0
        visible_kv += k - first * w.page
    flops = 2 * w.heads * pairs * (w.dq + 128)
    nbytes = (case.q.element_size() * (sum(w.q_lens) * w.heads * w.dq
              + visible_kv * w.kv_heads * (w.dq + 128)) + 2 * sum(w.q_lens) * w.heads * 128)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    record = {"case": case_name, "dtype": dtype, "q_lens": w.q_lens, "kv_lens": w.kv_lens,
              "dq": w.dq, "heads": w.heads, "kv_heads": w.kv_heads, "page": w.page,
              "causal": w.causal, "window_left": w.window, "sink": w.window >= 0,
              "timing_us": samples, "median_us": medians, "errors": errors, "max_abs_errors": max_errors,
              "unavailable": unavailable, "dispatch": dispatch, "effective_flops": flops, "logical_bytes": nbytes}
    aiter.logger.info("README_RETEST_RESULT %s", json.dumps(record, allow_nan=False))
    ret = {"gfx": reference_module.GPU_ARCH.split(":")[0], "page": w.page, "dq": w.dq,
           "batch": len(w.q_lens), "heads": w.heads}
    for name in ("8wave_5d", "4wave_static_5d", "4wave_dynamic_5d", "aiter_linear", "aiter_5d", "opus_linear"):
        us = medians.get(name, float("nan"))
        ret.update({f"{name} us": us, f"{name} TFLOPS": flops / us / 1e6,
                    f"{name} TB/s": nbytes / us / 1e6, f"{name} err": errors.get(name, float("nan"))})
    return ret


def main():
    if "gfx950" not in reference_module.GPU_ARCH:
        raise SystemExit("This current-8-wave README retest requires gfx950; no gfx942 results will be inferred.")
    matrix = workloads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(matrix), nargs="+", default=list(matrix))
    parser.add_argument("--dtype", choices=("bf16", "fp8"), nargs="+", default=["bf16", "fp8"])
    args = parser.parse_args()
    for dtype in args.dtype:
        rows = [benchmark_case(name, dtype) for name in args.case]
        aiter.logger.info("README retest %s (5D inputs shared; linear KV prepared before timing):\n%s",
                          dtype, pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()