#!/usr/bin/env python3
"""Minimal benchmark for preshuffle_gemm_v2 as a pure GEMM.
Matches bench_gemm_core tile sizes (BM=64, BN=256, TK=64, bf16).
Uses SAME measurement method as bench_gemm_core (cudaPerf, single buffer).
Usage: HIP_VISIBLE_DEVICES=4 python3 bench_fly_ref.py [batches=4096]
"""
import os, sys
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
import torch

# Add FlyDSL to path
flydsl_root = os.path.join(os.path.dirname(__file__), "../../../../../FlyDSL")
sys.path.insert(0, flydsl_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../src"))

import flydsl.compiler as flyc
from misc import cudaPerf

torch.cuda.set_device(0)
torch.set_default_device("cuda")

from kernels.preshuffle_gemm_v2 import compile_preshuffle_gemm_v2
from tests.utils import shuffle_weight

N, K = 5120, 8192
TM, TN, TK = 64, 256, 64

batches = int(sys.argv[1].split("=")[1]) if len(sys.argv) > 1 and "=" in sys.argv[1] else int(sys.argv[1]) if len(sys.argv) > 1 else 4096

M = ((batches + TM - 1) // TM) * TM

a = torch.rand(M, K, dtype=torch.bfloat16)
b_raw = torch.rand(N, K, dtype=torch.bfloat16)
b_shuf = shuffle_weight(b_raw, layout=(16, 16))
c = torch.zeros(M, N, dtype=torch.bfloat16)
sa = torch.empty(0, dtype=torch.float32, device="cuda")
sb = torch.empty(0, dtype=torch.float32, device="cuda")

STREAM = torch.cuda.current_stream()

jit = compile_preshuffle_gemm_v2(
    N=N, K=K, tile_m=TM, tile_n=TN, tile_k=TK,
    in_dtype="bf16", out_dtype="bf16",
)

args = (c.view(-1), a.view(-1), b_shuf.view(-1), sa, sb, M, N, STREAM)
fn = flyc.compile(jit, *args)

# warmup
for _ in range(10):
    fn(*args)
torch.cuda.synchronize()

p = cudaPerf(name="", verbose=0)
for _ in range(20):
    with p:
        fn(*args)
us = p.dt() * 1e6
tflops = (2 * M * N * K) / (us * 1e-6) / 1e12
print(f"preshuffle_gemm_v2  TM={TM} TN={TN} TK={TK}  M={M}  {us:.1f} us  {tflops:.1f} TFLOPS")
