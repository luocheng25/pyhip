#!/usr/bin/env python3
"""Profile the gateup GEMM kernel ALONE (prefill 2x2 / 1x4 / 2x2_simple) with cudaPerf
(a GPU-side sleep before the start event hides CPU launch overhead, so the measured
event span is pure GPU kernel time), reporting per-config latency and TFLOPS.

TFLOPS = gateup FLOPs / gateup GPU time, where the gateup FLOPs count ONLY the
gate+up projection:  2 * B * TOPK * HIDDEN * INTER_TP * 2   (this is the test_moe.py
`flops` formula with the down-stage term `HIDDEN * INTER_TP` dropped).

Sweeps  wave{2x2,1x4,2x2_simple} x dtype{bf16,per_tensor,ptpc} x BM{64,128} x
BN{128,256} x B{64,256,8192}.  Filter any axis from the CLI, e.g.:

  FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 profile_gateup.py \
      waves=2x2_simple dtypes=bf16,ptpc batches=8192 bms=128 bns=128

Run with no args for the full sweep.  Cache MUST be disabled for reliable per-config
timing (a shared cache can load the wrong compiled kernel).
"""
import os
import sys

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import flydsl.expr as fx
import flydsl.compiler as flyc
from contrib.flydsl.moe_gemm_splitk import compile_gemm
from misc import cudaPerf

import aiter
from aiter.fused_moe import moe_sorting
from pyhip import div_up

import test_moe  # reuse quant_expert_weights + get_fp8type

torch.cuda.set_device(0)
torch.set_default_device("cuda")
torch.manual_seed(0)

# Model dims: match tests/contrib/moe/_tmp_check.py (HIDDEN=4096, INTER=1024, TP=8, E=64).
# These can be overridden via CLI: e=192 topk=10 inter_tp=192
E, TOPK, HIDDEN, INTER_TP = 64, 8, 4096, 128
# Parse model-dim overrides early so N1/K1 are correct.
for _a in sys.argv[1:]:
    if "=" not in _a:
        continue
    _k, _v = _a.split("=", 1)
    if _k == "e":
        E = int(_v)
    elif _k == "topk":
        TOPK = int(_v)
    elif _k == "hidden":
        HIDDEN = int(_v)
    elif _k == "inter_tp":
        INTER_TP = int(_v)
N1, K1 = INTER_TP * 2, HIDDEN  # gateup weight: [E, INTER_TP*2, HIDDEN]
STREAM = torch.cuda.current_stream()

_T2FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


BUF_COPY = 10  # rotate input buffers to flush L2 cache between iterations
# Allow CLI override: buf_copy=5
for _a in sys.argv[1:]:
    if _a.startswith("buf_copy="):
        BUF_COPY = int(_a.split("=", 1)[1])


def _ptr(t):
    return flyc.from_c_void_p(_T2FX[t.dtype], t.data_ptr())


def cudaperf_us(fn_list, n=10, warmup=5):
    """GPU time (us) via pyhip cudaPerf; runs n timed iterations rotating through
    fn_list (one per buffer set), returns (min_us, max_tflops) from n runs."""
    nb = len(fn_list)
    for i in range(warmup * nb):
        fn_list[i % nb]()
    torch.cuda.synchronize()
    times = []
    for i in range(n):
        p = cudaPerf(name="", verbose=0)
        with p:
            fn_list[i % nb]()
        times.append(p.dt() * 1e6)
    return min(times)


_INPUT_CACHE = {}


def build_inputs(B, weight_type, quant_type, BM):
    """Build (and cache) BUF_COPY sets of gateup launch args for one (B, dtype, quant, BM)."""
    key = (B, weight_type, quant_type, BM)
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]

    all_args = []
    all_hold = []
    for _buf_i in range(BUF_COPY):
        hs = (torch.randn([B, HIDDEN], dtype=torch.bfloat16) + 1) * 0.001
        if weight_type == torch.bfloat16:
            w1 = torch.randn([E, N1, K1], dtype=torch.bfloat16)
            w1_scale = torch.empty(1, dtype=torch.float32)
        else:
            w_ = torch.randn([E, N1, K1], dtype=torch.bfloat16)
            w1, w1_scale, _ = test_moe.quant_expert_weights(w_, quant_type, weight_type)

        topk_weight = torch.randn([B, TOPK], dtype=torch.float32)
        topk_ids = torch.ones([B, TOPK], dtype=torch.int32)
        rep_e = div_up(B * TOPK, E)
        t1d = torch.ones([rep_e, E], dtype=torch.int32)
        t1d[:, ] = torch.randperm(E, dtype=torch.int32)
        topk_ids[:, ] = t1d.reshape(-1)[: B * TOPK].reshape(B, TOPK)

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids, topk_weight, E, K1, torch.bfloat16, BM, None, None, 0
        )
        grid = sorted_expert_ids.shape[0]
        gemm1_out = torch.empty([B, TOPK, N1 // 2], dtype=torch.bfloat16)

        if weight_type == torch.bfloat16:
            gin = hs
            a_scale = torch.empty(1, dtype=torch.float32)
        elif quant_type == "ptpc":
            gin, a_scale = aiter.get_hip_quant(aiter.QuantType.per_Token)(
                hs, quant_dtype=weight_type
            )
            a_scale = a_scale.to(torch.float32).contiguous()
        else:  # per_tensor
            fmax = torch.finfo(weight_type).max
            a_scale = hs.float().abs().amax() / fmax
            gin = (hs.float() / a_scale).clamp(-fmax, fmax).to(weight_type)
            a_scale = a_scale.reshape(1).to(torch.float32)

        args = (
            _ptr(gin), _ptr(w1), _ptr(gemm1_out), _ptr(sorted_ids), _ptr(sorted_weights),
            _ptr(sorted_expert_ids), _ptr(num_valid_ids), _ptr(w1_scale), _ptr(a_scale),
            B, grid, STREAM,
        )
        all_args.append(args)
        all_hold.append((hs, w1, w1_scale, gin, a_scale, gemm1_out, sorted_ids,
                         sorted_weights, sorted_expert_ids, num_valid_ids))

    _INPUT_CACHE[key] = (all_args, all_hold)
    return _INPUT_CACHE[key]


_ALG = {"2x2": "prefill_2x2", "1x4": "prefill_1x4", "2x2_simple": "prefill_2x2_simple"}


def profile_one(wave, weight_type, quant_type, BM, BN, B, tile_k=None):
    all_args, _hold = build_inputs(B, weight_type, quant_type, BM)
    weight_dtype = "bf16" if weight_type == torch.bfloat16 else "fp8"
    cq = "no" if weight_type == torch.bfloat16 else quant_type
    jit = compile_gemm(
        N=N1, K=K1, weight_dtype=weight_dtype, weight_quant_type=cq, TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BM, BLOCK_TILE_SIZE_N=BN, stage="gateup", alg=_ALG[wave], E=E,
        tile_k=tile_k,
    )
    fn = flyc.compile(jit, *all_args[0])  # traces + compiles + runs once
    fn_list = [lambda a=a: fn(*a) for a in all_args]
    us = cudaperf_us(fn_list)
    flops = 2 * B * TOPK * HIDDEN * INTER_TP * 2
    tflops = flops / (us * 1e-6) / 1e12
    return us, tflops


def _parse_filters(argv):
    f = {
        "waves": ["2x2", "1x4", "2x2_simple"],
        "dtypes": ["bf16", "per_tensor", "ptpc"],
        "batches": [64, 256, 1024, 4096, 8192, 16384],
        "bms": [64, 128],
        "bns": [128, 256],
        "bks": [64, 128, 256],  # BK == gateup TILE_K; a knob only for 1x4 (bf16 64/128, fp8 128/256)
    }
    for a in argv:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        if k in ("bk", "tk", "tks"):  # aliases for bks
            k = "bks"
        if k not in f:
            continue
        vals = v.split(",")
        f[k] = [int(x) for x in vals] if k in ("batches", "bms", "bns", "bks") else vals
    return f


def _print_summary(rows, f):
    """Pivot the flat rows into per-wave comparison tables (row = dtype/BM/BN/BK/B,
    column = wave), one for TFLOPS and one for latency, marking the best wave per row."""
    data = {}
    for wave, dt, BM, BN, BK, B, us, tfl in rows:
        data.setdefault((dt, BM, BN, BK, B), {})[wave] = (us, tfl)
    waves = f["waves"]
    bks = sorted({r[4] for r in rows})
    for metric, label, better in (
        ("tfl", "TFLOPS by wave (higher = better)", max),
        ("us", "latency us by wave (lower = better)", min),
    ):
        print(f"\n=== {label}; * = best per row ===")
        hdr = (f"{'dtype':<11} {'BM':>4} {'BN':>4} {'BK':>4} {'B':>6} "
               + " ".join(f"{w:>12}" for w in waves))
        print(hdr)
        print("-" * len(hdr))
        for dt in f["dtypes"]:
            for BM in f["bms"]:
                for BN in f["bns"]:
                    for BK in bks:
                        for B in f["batches"]:
                            cell = data.get((dt, BM, BN, BK, B), {})
                            vals = {
                                w: (cell[w][1] if metric == "tfl" else cell[w][0])
                                for w in waves if w in cell
                            }
                            if not vals:
                                continue
                            best_w = better(vals, key=vals.get)
                            cols = []
                            for w in waves:
                                if w in vals:
                                    mark = "*" if w == best_w else " "
                                    cols.append(f"{vals[w]:>11.1f}{mark}")
                                else:
                                    cols.append(f"{'-':>12}")
                            print(f"{dt:<11} {BM:>4} {BN:>4} {BK:>4} {B:>6} " + " ".join(cols))


def main():
    f = _parse_filters(sys.argv[1:])
    fp8 = test_moe.get_fp8type()
    dtype_map = {"bf16": (torch.bfloat16, "no"), "per_tensor": (fp8, "per_tensor"),
                 "ptpc": (fp8, "ptpc")}

    print(f"gateup profile: E={E} TOPK={TOPK} HIDDEN={HIDDEN} INTER_TP={INTER_TP} "
          f"N1={N1} K1={K1}")
    hdr = (f"{'wave':<11} {'dtype':<11} {'BM':>4} {'BN':>4} {'BK':>4} {'B':>6} "
           f"{'us':>9} {'TFLOPS':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for wave in f["waves"]:
        for dt in f["dtypes"]:
            weight_type, quant_type = dtype_map[dt]
            # BK (gateup TILE_K) only varies for the bf16 1x4 core (per-ki loop); every other
            # (wave, dtype) has a single valid TILE_K, so use the default (tile_k=None).
            # BK (gateup TILE_K) only varies for the 1x4 core: bf16 supports {64,128}, fp8
            # supports {128,256}. Every other wave has a single valid TILE_K (tile_k=None).
            if wave == "1x4":
                valid = (64, 128) if dt == "bf16" else (128, 256)
                bks = [b for b in f["bks"] if b in valid] or [None]
            else:
                bks = [None]
            for BM in f["bms"]:
                for BN in f["bns"]:
                    for bk in bks:
                        bk_disp = bk if bk is not None else (128 if dt != "bf16" else 64)
                        for B in f["batches"]:
                            try:
                                us, tfl = profile_one(wave, weight_type, quant_type, BM, BN, B, bk)
                                print(f"{wave:<11} {dt:<11} {BM:>4} {BN:>4} {bk_disp:>4} {B:>6} "
                                      f"{us:>9.1f} {tfl:>8.1f}", flush=True)
                                rows.append((wave, dt, BM, BN, bk_disp, B, us, tfl))
                            except Exception as ex:  # noqa
                                print(f"{wave:<11} {dt:<11} {BM:>4} {BN:>4} {bk_disp:>4} {B:>6} "
                                      f"{'FAIL':>9} {str(ex)[:40]}", flush=True)
    _print_summary(rows, f)
    return rows


if __name__ == "__main__":
    main()
