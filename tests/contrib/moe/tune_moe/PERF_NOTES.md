# 1x4 GEMM 优化记录

MI308X gfx942, bf16, M=4096, N=5120, K=8192, 10-buffer, min of 5 runs.
bench = 双 GEMM (gate+up) B-first; fly = preshuffle_gemm_v2 单 GEMM A-first.

## 性能数据

| 版本 | 64×256×64 | VGPR | 64×128×64 | VGPR | 64×128×128 | VGPR | 128×128×64 | VGPR | SGPR |
|---|---|---|---|---|---|---|---|---|---|
| 基线 | 239T | 172 | — | — | 219T | 169 | — | — | 96 |
| 方案2+CShuffle | 273T | 190 | 220T | 124 | 212T | 212 | 225T | 208 | 96 |
| 方案3 直接store | 275T | 188 | 231T | 122 | 262T | 212 | 259T | 208 | 96 |
| **方案4 gate/up交错** | **268T** | **169** | **238T** | **97** | **252T** | **169** | — | — | **96** |
| fly | 272T | 144 | 228T | — | 258T | 138 | 238T | — | 24 |
| ratio (方案4/fly) | **0.99** | | **1.04** | | **0.98** | | — | | |

## 优化方案

### 基线优化: preshuffle_v2 scheduler + CShuffle bf16 epilogue
1. hot_loop_scheduler 改为 preshuffle_v2 gfx942 模式: dsrd(2)+mfma(1)+mfma(1) header + 均匀 vmem+mfma_group+dsrd+mfma_group+dswr 循环 + sched_barrier(0) 末尾
2. epilogue 从 f32 global_store 改为 bf16 CShuffle via LDS + buffer_store_dwordx4
- 64×256×64: 239→268 (+29T), VGPR 172→154

### 方案1: a_cp_frag ping-pong 解决数据依赖
将 A staging fragment 拆为 ping/pong 双 buffer，消除 ds_write→buffer_load RAW 依赖。
- **缺点**: VGPR +36 (172→208, 64×256×64), +31 (169→200, 64×128×128)
- 性能无改善 (239→238), 因 VGPR 增加降低 occupancy 抵消了消除依赖的收益

### 方案2: 计算/读写分步 (sched_barrier + s_setprio)
sched_barrier(0) 隔离 gemm 计算块 + s_setprio(1) 提升首条 mfma 优先级 + 手动展开首条 mfma。
- 64×256×64: 239→254 (+15T), VGPR 186
- 核心: sched_barrier(0) 让 LLVM 将 buffer_load 提前到 s_barrier 前发射, vmcnt 从 (1) 变为 (7)

### 方案2+CShuffle: bf16 epilogue + BufferCopy128b store
基于方案2, 将 epilogue 从 f32 global_store 改为 bf16 CShuffle + buffer_store_dwordx4。
- CShuffle 在 BN=128 (contiguous_n=64) 时有 LDS 写竞争导致非确定性 (已废弃)

### 方案3: 直接 BufferCopy64b store + (M,N) 标准输出
去掉 CShuffle，改用 `make_tiled_copy_C(BufferCopy64b, tiled_mma)` 直接将 bf16 写到 (M,N) 标准 layout。
- B-first value dim 4 contiguous channels/lane → 64b store (buffer_store_short_x2) 自然对齐
- 消除 CShuffle 的 LDS 转置开销和精度问题
- 输出标准 (M, N) 行主序 bf16 矩阵 (gate 在前 N/2 列, up 在后 N/2 列)
- sched_barrier(0) + s_setprio(1) + gemm_with_setprio 函数封装手动展开
- hot_loop_scheduler 按 tile 自适应: TK=128/BM=128 用 per-iteration dsrd+vmem loop

### 方案4 (当前): gate/up MFMA 交错 + preshuffle pipeline
将两次 `fx.gemm` (gate/up) 展开为显式 `mma_atom_call` 循环，gate 和 up 在最内层交错执行。
每个 activation ds_read 结果被 gate 和 up 两条 MFMA 连续消费，形成天然的 ds_read pipeline 重叠。
- 循环次序: ki(外) → k_atom(中) → n_reps(内) → m_reps(最内)，gate/up 在 atom 级交错
- LLVM 自动生成 lgkmcnt(1)~(2) 而非 lgkmcnt(0)，ds_read pipeline 与 fly 模式匹配
- 去掉 s_setprio / gemm_with_setprio 手动展开，改用 scheduler 自然调度
- preshuffle_v2 风格 per-ki ds_read + gemm pipeline + hot_loop_scheduler
- lgkmcnt 分布: lgkmcnt(1)×14, lgkmcnt(0)×8 (vs fly: lgkmcnt(1)×14, lgkmcnt(0)×8 完全一致)
- VGPR: 169 (vs fly 138)，差距来自双 C fragment (gate+up) 和双 weight buffer
- 精度: 对比 torch `A @ shuffle_weight(W)^T`, max_diff=0.031, rel_err=0.0003%

## 精度验证

使用 `check_bench_1x4.py` 对比 torch `A @ W^T` 参考（shuffle_weight 后传入 kernel）:

```bash
cd tests/contrib/moe/tune_moe
# tk=64, 默认 tiles=64x128,64x256,128x128
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 check_bench_1x4.py tk=64
# tk=128
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 check_bench_1x4.py tk=128
# 指定 tiles
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 check_bench_1x4.py tk=64 tiles=64x128,128x128
```

通过标准: rel_err < 1%, max_diff < 1.0 (bf16 ULP 级别)。
典型结果: max_diff=0.002, rel_err=0.0002%。

## 复现

```bash
cd tests/contrib/moe/tune_moe
# bench vs fly 对比 (默认 3 标准 tiles: 64x256x64, 64x128x64, 64x128x128)
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4,fly
# 指定 tiles
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4,fly tiles=64x256x64,64x128x128,128x128x64
# 仅 bench
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4 tiles=64x128x128
# 仅 fly (preshuffle_gemm_v2 参考)
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=fly tiles=64x128x128
# 指定 tile_k
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4,fly bms=64 bns=128 tk=128
# dump ISA
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 bench_gemm_core.py waves=1x4 tiles=64x256x64
grep 'next_free_vgpr\|next_free_sgpr' /tmp/isa/bench_kernel_0/21_final_isa.s
```
