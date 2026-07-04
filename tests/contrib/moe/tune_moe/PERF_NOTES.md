# 1x4 GEMM 优化记录

MI308X gfx942, bf16, M=4096, N=5120, K=8192, 10-buffer, min of 5 runs.
bench = 双 GEMM (gate+up) B-first; fly = preshuffle_gemm_v2 单 GEMM A-first.

## 性能数据

| 版本 | 64×256×64 | VGPR | 64×128×128 | VGPR | SGPR |
|---|---|---|---|---|---|
| 基线 | 239T | 172 | 219T | 169 | 96 |
| fly | 275T | 144 | 262T | 138 | 24 |

## 复现

```bash
cd tests/contrib/moe/tune_moe
# bench (自动选空闲 GPU, min of 5 runs)
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4,fly bms=64 bns=256 batches=4096 runs=5
# 指定 tile_k
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4 bms=64 bns=128 batches=4096 tk=128 runs=5
# dump ISA
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4 bms=64 bns=256 batches=4096 runs=1
grep 'next_free_vgpr\|next_free_sgpr' /tmp/isa/bench_kernel_0/21_final_isa.s
```
