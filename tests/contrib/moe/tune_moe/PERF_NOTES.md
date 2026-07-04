# 1x4 GEMM 优化记录

MI308X gfx942, bf16, M=4096, N=5120, K=8192, 10-buffer, min of 5 runs.
bench = 双 GEMM (gate+up) B-first; fly = preshuffle_gemm_v2 单 GEMM A-first.

## 性能数据

| 版本 | 64×256×64 | VGPR | 64×128×128 | VGPR | SGPR |
|---|---|---|---|---|---|
| 基线 | 239T | 172 | 219T | 169 | 96 |
| 基线优化 | 268T | 154 | 196T | 169 | 96 |
| 方案1 ping-pong | 238T | 208 | 206T | 200 | 96 |
| fly | 275T | 144 | 262T | 138 | 24 |

## 优化方案

### 基线优化: preshuffle_v2 scheduler + CShuffle bf16 epilogue
1. hot_loop_scheduler 改为 preshuffle_v2 gfx942 模式: dsrd(2)+mfma(1)+mfma(1) header + 均匀 vmem+mfma_group+dsrd+mfma_group+dswr 循环 + sched_barrier(0) 末尾
2. epilogue 从 f32 global_store 改为 bf16 CShuffle via LDS + buffer_store_dwordx4
- 64×256×64: 239→268 (+29T), VGPR 172→154

### 方案1: a_cp_frag ping-pong 解决数据依赖
将 A staging fragment 拆为 ping/pong 双 buffer，消除 ds_write→buffer_load RAW 依赖。
- **缺点**: VGPR +36 (172→208, 64×256×64), +31 (169→200, 64×128×128)
- 性能无改善 (239→238), 因 VGPR 增加降低 occupancy 抵消了消除依赖的收益

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
