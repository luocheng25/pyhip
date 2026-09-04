# MoE down `8x1` 优化记录（BM=256 / 8 wave / 两段反相）

目标 kernel：`moe_2stage_down_prefill_8x1`，实现文件计划为
`src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py`，`down_path="8x1"`。
只修改 `pyhip` 模块，不改 `aiter` 与 `FlyDSL`。

---

## 0. 性能总结

> **状态说明**：本节分为三类数据，请勿混用。
> `[实测]` 来自已归档报告的 CUDA-event 中位数；
> `[推导]` 由 `[实测]` 数据加设备常数算得，不含新假设；
> `[预测]` 含未验证的效率假设，**在 §0.4 填入实测前不得作为结论引用**。

设备常数（MI308X / gfx942 / 80 CU / 1800 MHz determinism）：

$$
\text{FP8 peak}=80\times4096\times1.8\times10^{9}=589.8\ \text{TFLOPS},\qquad
\text{HBM}=5.3\ \text{TB/s}
$$

### 0.1 基线（Batch=32K，`[实测]`）

来源：[`tests/flydsl/attn_4wave/tools/batched-gemm-core-ceiling.md`](../tests/flydsl/attn_4wave/tools/batched-gemm-core-ceiling.md)
的「预测与实测」表（生产列与 ceiling 列同协议：10 buffer / 40 warmup / 50 sample / `sample-sync=end`）。

| Case | 当前 path | 生产 ms | 生产 useful TFLOPS | ceiling TFLOPS | 达到率 | 占 FP8 peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 397B K=256 | `1x4_64x256` | 1.8616 | 369.14 | 404.45 | 91.27% | 62.6% |
| Qwen3.5 35B K=256 | `1x4_64x256` | 0.7665 | 358.62 | 407.96 | 87.91% | 60.8% |
| Xiaomi K=256 | `1x4_64x256` | 2.2746 | 362.53 | 390.57 | 92.82% | 61.5% |
| H3 K=384 | `2x4` | 1.5777 | 392.01 | 423.34 | 92.60% | 66.5% |

关键观察：**生产已经把当前 tile 的 ceiling 吃掉了 88%–93%，但 ceiling 本身只有 peak 的 66%–69%。**
所以差距不在 kernel 调度，而在 **tile 选择**——这一点由 §0.2 定量确认。

### 0.2 Roofline 诊断（`[推导]`）

按 `逻辑访存量 = weight读 + output写 + activation读` 计算（不扣 MALL 命中，因此是上界）：

```text
weight 读 = WG数 x N x K x 1B          WG数 = E x ceil(rows_per_expert / BM)
output 写 = padded_rows x N x 2B
act   读 = padded_rows x K x 1B
```

| Case | BM | WG 数 | weight GB | output GB | act GB | 合计 GB | 实测 ms | 等效 TB/s | 占 HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 397B K=256 | 64 | 5120 | 5.369 | 2.684 | 0.084 | **8.14** | 1.8616 | 4.37 | **82.5%** |
| Qwen 35B K=256 | 64 | 4096 | 2.147 | 1.074 | 0.067 | **3.29** | 0.7665 | 4.29 | **81.0%** |
| Xiaomi K=256 | 64 | 4224 | 6.643 | 3.322 | 0.069 | **10.03** | 2.2746 | 4.41 | **83.2%** |
| H3 K=384 | 128 | 1024 | 2.416 | 1.611 | 0.050 | **4.08** | 1.5777 | 2.59 | 48.8% |

（Xiaomi 的 `rows/expert=682.7` 在 `BM=64` 下也要补到 11 tile，因此 output/act 按 `4224x64=270336`
行计；其余三个 case 在 `BM=64/128` 下正好整除。）

**结论：三个 K=256 case 都卡在 HBM 上（~82%），不是算力不足。**
`BM=64` 时 weight 的字节/FLOP 是 $1/(2\cdot BM)=1/128$，
折算成 32 B/clk/CU，而 MI308X 每 CU 只有 $5.3\text{TB/s}/80/1.8\text{GHz}=36.8$ B/clk。
`BM=256` 把它降到 $1/512$，即 8 B/clk/CU —— 这就是 8x1 的全部立论。
H3 是唯一不受带宽约束的 case，其收益来源不同（见 §9）。

### 0.3 8x1 目标（`[预测]`）

假设：8x1 在 MFMA 上达到 peak 的 **80% / 85% / 90%** 三档；`useful` 分母仍用未 padding 的原始 FLOPs。

| Case | rows/expert | 8x1 tile 数 | padding 系数 | WG 数 | 访存 GB | @80% | @85% | @90% | vs 基线 @85% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 397B K=256 | 640 | 3 | **1.200** | 1536 | 4.93 | 393.2 | 417.8 | 442.4 | **+13.2%** |
| Qwen 35B K=256 | 1024 | 4 | **1.000** | 1024 | 1.68 | 471.8 | 501.3 | 530.8 | **+39.8%** |
| Xiaomi K=256 | 683 | 3 | **1.125** | 1152 | 5.51 | 419.5 | 445.7 | 471.9 | **+22.9%** |
| H3 K=384 | 1024 | 4 | **1.000** | 512 | 2.87 | 471.8 | 501.3 | 530.8 | +27.9% |

同时校验新配置下访存不再是瓶颈（@85% 档）：

| Case | 8x1 预测 ms | 等效 TB/s | 占 HBM |
| --- | ---: | ---: | ---: |
| Qwen 397B K=256 | 1.645 | 3.00 | 57% |
| Qwen 35B K=256 | 0.548 | 3.06 | 58% |
| Xiaomi K=256 | 1.850 | 2.98 | 56% |
| H3 K=384 | 1.234 | 2.33 | 44% |

**收益强弱完全由 `rows_per_expert mod 256` 决定**：35B 与 H3 正好整除，收益最大；
397B 有 20% 的 M padding 浪费，吃掉了大半带宽收益。这是 8x1 最大的单点风险（§9）。

### 0.4 实测记录（`[实测]`）

**v1（2026-09-04，无两段反相）**。协议：MI308X GPU0、3 套 buffer 轮换、40 warmup、50 CUDA-event 样本、
取中位数；均衡路由（每 expert 恰好 `B*TopK/E` 行）；未设置 performance determinism。
基准列来自 §0.1，**跨环境、非配对**（当前容器跑不了 `1x4_64x256`，见 §9.1）。

| Case | 8x1 ms | useful TFLOPS | executed TFLOPS | 占 FP8 peak | §0.1 基准 ms / TFLOPS | 比值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 35B K=256 | 1.3890 `[1.3875--1.3908]` | 197.89 | 197.89 | 33.6% | 0.7665 / 358.62 | **0.55x** |
| Qwen3.5 397B K=256 | 3.8049 `[3.8014--3.8103]` | 180.61 | 216.73 | 36.7% | 1.8616 / 369.14 | **0.49x** |
| H3 K=384 | 2.4959 `[2.4916--2.4984]` | 247.80 | 247.80 | 42.0% | 1.5777 / 392.01 | **0.63x** |

**v1 还慢于基准约 2x。** 这符合预期：v1 故意不含 §5 的两段反相，epilogue 也完全不与 MFMA 重叠；
每个 K step 是严格串行的 `barrier -> 16x ds_read -> s_waitcnt lgkmcnt(0) -> 64x MFMA`。
测到的 33-42% peak 与“MFMA 完全暴露在 LDS 延迟后面”一致。

**资源（rocprofv3 kernel-trace 实测，验收门槛 2 已过）**：

| 项 | 实测 | 设计预算（§4） | 结论 |
| --- | ---: | ---: | --- |
| LDS | **50176 B (49.0 KiB)** | 50176 B | 完全命中 |
| Scratch | **0** | 0 | 无 spill |
| VGPR + AGPR | **108 + 132 = 240** | ≈222 | ≤256，2 waves/SIMD、1 WG/CU |
| SGPR | 112 | — | — |

数值正确性（验收门槛 1 已过）：`K x N` ∈ {128,256,384} x {128,256,384,512} 共 8 组全部通过，
相对误差 0.37%-0.68%（来自 `fma+0x8000+v_perm` 的截断式 BF16 转换，与现有生产 kernel 同口径）。

验收门槛（三项全过才可进 path 矩阵）：

1. ✅ 与 torch 参考逐 tile 数值一致；
2. ✅ 最终 ISA：`VGPR+AGPR = 240 ≤ 256`、`scratch = 0`；
3. ❌ 相对基准的提升率（v1 为负，待 v2 反相改造）。

### 0.5 优化尝试记录（`[实测]`，Qwen3.5 35B K=256）

> ⚠️ **测量协议缺陷（2026-09-04 发现，已修正）**：下表 #1–#4 是在**未锁频、且每个配置只测一次**
> 的条件下取得的。事后对**同一份未改动的代码**连测 4 次得到
> `1.3896 / 1.5117 / 1.5077 / 1.5134 ms`——**run-to-run 漂移 8.9%**，而单次运行内
> P25–P75 仅 ±0.3%。设 `perf_determinism` 后仍呈双峰 `1.5150 / 1.3899 / 1.3930 / 1.3884`：
> **每个进程的第一次运行偏慢约 8%**，因为一次 bench 只有 ~0.14 s GPU 负载，不足以让时钟从
> idle 爬满。
> **因此 #1–#3 的差异全部落在噪声带内，结论不成立；只有 #4 的两次测量（1.58/1.60）
> 高于观测到的噪声上界（1.5150），可判定为真实回退。**

| # | 改动 | ms（单次，含噪声） | VGPR+AGPR | 判定 |
| --- | --- | ---: | ---: | --- |
| 0 | v1 初版 | 1.5591 | — | 基线 |
| 1 | 删 CShuffle 内多余 workgroup barrier + 每 K step 2 barrier 降为 1 | 1.3890 | 240 | ⚠️ 噪声内，**不确定** |
| 2 | MFMA 提到 prefetch 之前 | 1.4112 | — | ⚠️ 噪声内，不确定 |
| 3 | `frag_sorted_weight` 由 fragC 形状（64）压为 2 个标量 | 1.5056 | **248** | ⚠️ 时间不确定；但**寄存器 240→248 是确定的**，故仍回退 |
| 4 | **两段反相**（条件错位 barrier + `s_setprio`） | 1.6037 / 1.5832 | — | ❌ 高于噪声上界，**真实回退** |

**当前稳态基准**：连测 3 次 `1.3899 / 1.3930 / 1.3884 ms`（弃首次），即 **≈1.389 ms / 197.9 useful TFLOPS**。

需要记录的结论：

- **测量协议**：后续所有对比必须（a）弃掉每轮首次运行，（b）每个配置重复 ≥3 次，
  （c）用 ABBA 配对而非独立测量。这与 `MAIN_MERGE_PERFORMANCE_REPORT.md` 的协议一致——
  之前偷懒省掉这些步骤，导致 3 个结论作废。
- **尝试 3 的寄存器发现仍然有效**（编译期属性，与时钟无关）：把 `frag_sorted_weight`
  从 fragC 形状压成标量后，**AGPR 仍为 132 不变、VGPR 反而 108→116**。说明这 64 个值本来
  就分配在 AGPR（MFMA 专用池，不与 VGPR 争抢），是"免费存储"。
  → **§4.1 的寄存器预算应区分 VGPR/AGPR 两个池，不能只看总数。**
- **尝试 4（反相）是真实回退**，与 §5.1 的周期账预测相反。周期账假设 mem stage ≈ 600–700 clk
  < compute stage 1024 clk；实测反相后变慢，说明 **mem stage 实际远长于 compute stage**。
  最可能的原因是 §3.5 声称"免 swizzle、零 bank conflict"的 `ds_read` 布局并不成立。
  → **下一步必须用 ATT 实测 mem stage 时长与 LDS conflict，而不是继续盲调。**

---

## 1. 为什么是 8x1 而不是继续调 1x4

| 方案 | weight 字节/FLOP | 结果 |
| --- | --- | --- |
| `1x4_64x256`（BM=64） | 1/128 → 32 B/clk/CU | 已达 ceiling 91%，但 ceiling 被带宽压到 peak 的 68% |
| `2x4`（BM=128） | 1/256 → 16 B/clk/CU | 好一半，但 4-wave 子组各自算 N，LDS 里 activation 翻倍 |
| **`8x1`（BM=256）** | **1/512 → 8 B/clk/CU** | 带宽退居次要，瓶颈交回 MFMA |

`8x1` 的代价是 M 方向 8 个 wave 共享同一块 B，LDS 读放大 8×（§5），
以及 sorting tile 变成 256 带来的 padding 浪费（§9）。

先例：仓库内 [`docs/design_moe_gemm_8wave_down.md`](design_moe_gemm_8wave_down.md)
（`wg_M=256` / 8 wave / A 常驻 VGPR / B 走 LDS / 条件错位 barrier）已验证该拓扑可行。
本设计在其基础上新增 **BN=128 外层 + BK=128 内层 + 严格两段反相 + ping-pong（2 级而非 4 级）**，
把 LDS 从 68 KiB 压回 49 KiB。

---

## 2. 几何与工作划分

```text
BM=256   BN=128   BK=128    8 wave / 512 thread    1 WG/CU
wave w :  A[32 x K] 常驻 VGPR ,  C[32 x 128] 常驻 accumulator
8 wave :  共享同一块 B[128(N) x 128(K)]  (LDS)

MFMA   :  v_mfma_f32_16x16x32_fp8_fp8
  MMA_M = weight_N (128) -> 8 tile     MFMA A operand = weight (来自 LDS)
  MMA_N = act_M   (32)  -> 2 tile      MFMA B operand = activation (来自 VGPR)
  每 compute stage : 8(N) x 4(k-step) x 2(M) = 64 条 MFMA x 16 clk = 1024 clk

循环   :  外层 N 步长 128 (nBN = N/128) ; 内层 K 步长 128 (nBK = K/128)
          K=256 -> nBK=2 ; K=384 -> nBK=3 ; K=128 -> nBK=1
```

`create_thr_mma(fp8, wave_mnk=(1, 8, 1))`：8 个 wave 沿 `MMA_N`（= activation M）分布。

---

## 3. 关键设计决策

### 3.1 MFMA 形状：`16x16x32` 而非 `32x32x16`

两者都是 1024 clk/stage，但 C fragment 的 lane 映射不同：

| 形状 | 每 lane 每 atom 覆盖的 `MMA_M` 值 | per-channel `w_scale` 所需寄存器 |
| --- | ---: | ---: |
| `16x16x32` | 4 个连续 | **32**（分块加载后 16） |
| `32x32x16` | 16 个 | 64 |

`16x16x32` 还让 lane 内天然持有 **4 个连续 N**，正好 pack 成 2 个 dword 供 `ds_write_b64`/`b128`。

### 3.2 MMA 方向：`MMA_M = weight_N`

被否的反向（`MMA_M = act_M`）虽然把 `w_scale` 压到 4 个寄存器，
但 lane 内变成 4 个连续 **M**，写 LDS 时退化成 `ds_write_b16` ×4，
指令数是本方案的 4 倍——与本轮「输出必须以 128-bit 写入 LDS」的要求冲突。

### 3.3 输出路径：C → (128-bit) LDS → `dwordx4` 全局写

本轮明确要求，且与 §0.2 一致：`8x1` 下 output 写已经是**第一大**全局流量
（Qwen 397B：3.22 GB output vs 1.61 GB weight）。因此必须保证写出是满 128-bit 事务。

沿用 `gemm2_1x4.py` 的 CShuffle：`ds_write` 8×bf16（128 bit）→ `s_waitcnt lgkmcnt` → `ds_read_b128`
→ `buffer_store_dwordx4`，XOR swizzle（`physical_atom = logical_atom ^ row_in_8`）消 bank conflict，
store 带 `cache_modifier=2`（gfx942 raw-buffer aux bit1 = 非时间性）。

被否方案（记 TODO，仅在 LDS 成为实测瓶颈时启用）：

- 直接 `buffer_store_dwordx2`：省 16 KiB LDS 与全部 cshuffle 流量，但事务降到 8 B；
- `ds_bpermute_b32` 做 lane 内 4×4 转置：LDS 流量降 4×，但仍占 LDS 管线。

### 3.4 地址寄存器：`raw_ptr_buffer_load/store` + `readfirstlane`

已确认 FlyDSL 暴露：

```text
fx.rocdl.raw_ptr_buffer_load (res, rsrc, offset, soffset, aux)
fx.rocdl.raw_ptr_buffer_store(vdata, rsrc, offset, soffset, aux)
fx.rocdl.readfirstlane      (res, src)
```

`soffset` 是独立的 SGPR 操作数。做法：**lane 内偏移固定进一个 VGPR（循环不变），
每个 tile 的推进量走 `readfirstlane` 得到的 SGPR `soffset`，用 `s_add` 递推。**

收益量化：核心循环每 stage 有 16 `ds_read` + 2 `buffer_load` + 2 `ds_write` + 8 `buffer_store` = 28 条访存。
若每条都要一次 `v_add_u32` 算地址，就占掉 28 条 VALU；而按 §5.3 每 stage 的 VALU 预算只有 192 条，
仅地址就吃掉 **15%**，且这些 VGPR 全程 live。改走 `soffset` 后地址推进变成 SALU，
既不占 VALU co-issue 窗口，也不占 VGPR。

> 落地方式：先用现有 `fx.copy` + `buffer copy atom` 打通功能，再在 ISA 复查阶段
> （§8 第 2 项）逐点替换为 `raw_ptr_buffer_*`，用反汇编确认核心循环无 `v_add`/`v_lshl` 地址链。

### 3.5 weight 排布：沿用 `moe_gemm_8wave_down` 的 preshuffle

逻辑视图 `[N/16][Kbytes/64] x (16 行 x 64 B = 1024 B)`。该排布同时满足：

- global 侧 `buffer_load_dwordx4` 线性、全 coalesced；
- LDS 侧 `ds_write_b128` 线性；
- `ds_read_b128` 取 `unit_base + lane*16`，64 lane 扫 1024 B 连续 → **零 bank conflict，不需要 XOR swizzle**；
- 每个 1024 B unit 恰好喂 2(k-step) × 2(M-tile) = 4 条 MFMA，16 unit × 4 = 64 条 ✓。

一块 `128N x 128K` tile = $(128/16)\times(128/64)\times1024 = 16$ KiB。

### 3.6 禁用 packed FP32（`v_pk_*`）

依据 [`tests/flydsl/attn_4wave/tools/mfma-valu-coissue.md`](../tests/flydsl/attn_4wave/tools/mfma-valu-coissue.md)
的 gfx942 正式结果（`I/I` = intra/inter，容量指能被 MFMA 完全遮盖的条数）：

| opcode | cycle/inst | hidden I/I | 说明 |
| --- | ---: | ---: | --- |
| `v_mul_f32` / `v_add_f32` | 4.012 | **3/3** | epilogue 主力 |
| `v_fma_f32` | 4.016 | **3/3** | bf16 pack 的 FMA |
| `v_perm_b32` | 4.012 | **3/3** | bf16 pack 的打包 |
| `v_add_u32` / `v_cndmask_b32` | 4.008 | 3/3 | 地址/掩码 |
| `v_pk_mul_f32` / `v_pk_add_f32` | 4.008 | **0/0** | intra 下首条 +12 cycle，inter +5 cycle/条 |
| `v_exp_f32` / `v_rcp_f32` | 16.000 | 0/0 | 本 kernel 不涉及 |

packed FP32 吞吐本身没问题（4.008 cycle），但**无法被 MFMA 遮盖**，是 pipeline 冲突而非慢吞吐。
本 kernel 的 epilogue 全部由容量 3 的指令构成，因此必须沿用现有的
`value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]}` 抑制 LLVM 自动打包。

---

## 4. 资源预算

### 4.1 寄存器（每 lane，上限 256 → 2 wave/SIMD）

| 项目 | 计算 | regs |
| --- | --- | ---: |
| `fragC` accumulator | 32×128 f32 / 64 lane | **64**（AGPR） |
| `frag_weight`（MFMA A，BK=128） | 128×128 B / 64 = 256 B | **64** |
| `frag_act`（MFMA B，K=256 常驻） | 32×256 B / 64 = 128 B | **32** |
| B global staging ×2 组 | 2 × 32 B/lane | **16** |
| `w_scale` 分块（4 个 N-atom/次） | 4×4 f32 | **16** |
| `a_scale × routing_weight` | 每 `MMA_N` tile 一个 | **2** |
| bf16 pack + 延迟半个 tile 的 store | | **16** |
| 地址 / descriptor / 循环 / sorted_id | 走 `soffset` 后大幅压缩 | **~20** |
| **合计** | | **≈ 222** |

按 8 对齐后 224 ≤ 256 → 2 wave/SIMD，与 8 wave/WG、1 WG/CU 自洽。

**关键取舍**：不保留第二份 accumulator。`gemm2_2x4.py` 用 `previous_fragC` 把 store 摊到下一个 N 块
（+64 regs），在本拓扑下 `64+64+64+32=224` 已经用满，再加 scale 必爆。
改为「C 算完后在紧邻的 mem stage 一次性 scale+pack，只延迟 bf16 结果的一半」（+16 regs）。

`frag_act` 随 K 变化：K=128 → 16、K=192 → 24、K=256 → **32**、K=384 → 48（总计 ≈238，仍可行）。

### 4.2 LDS（上限 64 KiB，ping/pong 用两个独立变量）

```python
lds_b_ping   = alloc(fp8,  128*128)     # 16384 B
lds_b_pong   = alloc(fp8,  128*128)     # 16384 B
lds_cshuffle = alloc(bf16, 8 * 16*64)   # 16384 B   每 wave 私有 2 KiB
lds_wscale   = alloc(f32,  2 * 128)     #  1024 B   double buffer
# --------------------------------------------------------------
# 合计 50176 B = 49 KiB  <= 64 KiB   -> 1 WG/CU（与 2 wave/SIMD 自洽）
```

对比 `moe_gemm_8wave_down` 的 4 级 ring（`4 x 16 KiB = 64 KiB`，加 routing 后 68 KiB）：
本设计靠「BK 分块 + 寄存器中转」把 ring 压到 2 级，腾出 16 KiB 给 CShuffle。

---

## 5. 两段反相流水与遮盖论证

一个 stage ≈ **1024 clk**（由 compute 组的 MFMA 定长）。同一时刻：4 wave 在 compute（每 SIMD 一个），
4 wave 在 mem。gfx942 把 wave $i$ 与 wave $i+4$ 放到同一 SIMD，因此
`grp0 = wave0-3`、`grp1 = wave4-7` 天然构成每 SIMD 一对。

### 5.1 每 stage 周期账（1 CU）

| 管线 | 占用 (clk) | 预算 | 利用率 |
| --- | ---: | ---: | ---: |
| Matrix core（compute wave，per SIMD） | 1024 | 1024 | **100%（目标瓶颈）** |
| LDS：B `ds_read` | 4 wave × 16 KiB ÷ 128 B/clk = 512 | | |
| LDS：B `ds_write`（下一块） | 4 × 2 KiB → 64 | | |
| LDS：CShuffle（半个 block 的 epilogue） | 4 × 8 KiB → 256 | | |
| **LDS 合计** | **832** | 1024 | **81%** |
| VALU（mem wave，per SIMD） | 见 §5.3 | | 83% |
| Global 读 weight | 16 KiB / 2048 clk = 8 B/clk | ~37 | 22% |
| Global 写 output | 32 KiB / 2048 clk = 16 B/clk | ~37 | 43% |

**MFMA 是唯一 100% 项，反相可以遮盖。** 但 LDS 81% 余量很小，因此 epilogue 必须摊到
两个 mem stage：若压进一个 stage，该 stage 的 LDS 变成 `4×(128+16+128)=1088 > 1024`，overrun 6%。

### 5.2 ping-pong 深度 2 成立的前提：**global → VGPR → LDS**

反相意味着 grp0 在 slot $2i$ 做 `mem(i)`，grp1 到 slot $2i+1$ 才做 `mem(i)`。
若用 `buffer_load_to_lds`（直写 LDS），grp1 在 slot $2i+1$ 发出的 load
对 grp0 在 slot $2i+2$ 的 `ds_read` **不可见**（grp1 来不及在 barrier 前 `s_waitcnt vmcnt`），
只能把预取距离拉到 2 → **需要 3 块 LDS buffer**。走寄存器中转就解开了：

```text
slot 2i    (grp0: mem(i))   : ds_write buf[(i+1)%2] <- stg   (数据是 slot 2i-2 load 的)
slot 2i+1  (grp1: mem(i))   : ds_write buf[(i+1)%2] <- stg
           ^ barrier         -> buf[(i+1)%2] 被 8 wave 写全，且各自 lgkmcnt(0) 过
slot 2i+2  (grp0: mem(i+1)) : ds_read  buf[(i+1)%2]                      OK
                              ds_write buf[i%2]   (slot 2i+1 已被读完)    OK
```

global load 从发出（slot $2i$，取 chunk $i+2$）到消费（slot $2i+2$ 的 `ds_write`）
= 2 slot = 2048 clk ≈ **1.14 µs @1.8 GHz**，足够盖住 HBM 延迟。代价只有 16 个 staging VGPR。

### 5.3 VALU 遮盖：按实测 co-issue 容量核算

每 wave 每 N 块的 epilogue：

$$
64\ \texttt{v\_mul\_f32}\ (w\_scale)\;+\;64\ \texttt{v\_fma\_f32}\ (tok\_scale + 0\text{x}8000)\;+\;32\ \texttt{v\_perm\_b32}=160\ \text{条}
$$

全部落在 §3.6 表中「容量 3」那一类。每个 compute stage 有 64 条 MFMA，
按实测 inter co-issue 容量 3：

$$
\text{每 stage 可遮盖 VALU}=64\times3=192\ \text{条}
$$

- 全部压在一个 stage：$160/192=83\%$，可行但无余量；
- 摊到两个 mem stage（本设计，同时也是 §5.1 的 LDS 要求）：$80/192=42\%$，余量充足。

若误用 `v_pk_mul_f32` 把 128 条 mul/fma 折成 64 条，遮盖容量直接归零，
这 64 条会**串行**叠加到 stage 时间上（约 +256 clk / stage，即 +25%）。

---

## 6. 正确性关键点

### 6.1 条件错位 barrier

```text
grp0 (wave0-3):            [barrier]  stage1 [b] stage2 [b] stage1 ... [barrier_Z]
grp1 (wave4-7): [barrier]  [barrier]  stage1 [b] stage2 [b] ...
```

- prologue 里 grp1 多打一次 `s_barrier`，epilogue 里 grp0 补一次 → **两组 barrier 总数配平**；
- 任何 wave 不得中途 `return`：早退判断必须整 WG 一致（`e_idx*256 < max_valid_id`）；
- stage 边界用 `sched_barrier(0) → s_barrier → sched_barrier(0)`；
  compute stage 进入前 `s_setprio(1)`、退出置 0（同 `gemm2_2x4.py` 的
  `enter_compute_stage` / `enter_read_write_stage`）。

### 6.2 waitcnt

- mem stage 内顺序：`ds_write` → `ds_read` → `s_waitcnt lgkmcnt(0)`；
- CShuffle 的自写自读是 wave 私有区，用 `lgkmcnt(1)/(0)` 分级即可，**不需要 barrier**；
- `vmcnt` 用滚动值：发射顺序固定为「prefetch load → output store」，
  下一 stage 顶部 `s_waitcnt vmcnt(pending_stores)` 只等 load，不等 store。

---

## 7. 流水线伪代码

```python
# ============ constexpr ============
BM, BN, BK, NW, NTHR = 256, 128, 128, 8, 512
WM  = BM // NW          # 32   每 wave 的 M 行
MT  = WM // 16          # 2    MMA_N tile (act M)
NT  = BN // 16          # 8    MMA_M tile (weight N)
KS  = BK // 32          # 4    每 K chunk 的 k-step
nBN = N // BN
nBK = K // BK           # K=256 -> 2

def moe_2stage_down_prefill_8x1(...):          # block = (512,1,1)
    wave = tid // 64 ;  grp = wave // 4
    e_idx = map_down_task(block_idx.y, task_rows=256)   # + XCD/SE topology swizzle
    if e_idx * BM >= max_valid_id: return               # 整 WG 一致

    # ---------------- prologue ----------------
    load sorted_ids[256], sorted_weights[256]
    rA   = gather_A(sorted_ids[wave*32 : wave*32+32], K)   # 8 x buffer_load_dwordx4 -> 32 regs
    rTok = a_scale[row] * sorted_weight[row]               # 2 regs (lane 内 act_M 固定)
    acc  = 0

    stg[0] = load_B_global(j=0, k=0)                       # 2 x buffer_load_dwordx4 -> 8 regs
    stg[1] = load_B_global(j=0, k=1)
    s_waitcnt vmcnt(1)
    ds_write(lds_b_ping, stg[0])
    s_waitcnt lgkmcnt(0)
    stage_end()                                            # sched_barrier(0); s_barrier; sched_barrier(0)

    if grp == 1: stage_end()                               # <<< 反相错位

    # ---------------- main loop ----------------
    for j in range(nBN):                     # 第一层: N, 步长 128
      for k in constexpr_range(nBK):         # 第二层: K, 步长 128
        # ============ STAGE 1 : mem / VALU ============
        s_setprio(0)
        cur, nxt   = (j*nBK+k) % 2, 1 - (j*nBK+k) % 2
        B_cur, B_nxt = (ping, pong) if cur == 0 else (pong, ping)

        s_waitcnt vmcnt(pending_stores)                    # 只等 load
        ds_write(B_nxt, stg[nxt])                          # 2 x ds_write_b128

        (jn, kn) = (j, k+2) if k+2 < nBK else (j+1, k+2-nBK)
        stg[cur] = load_B_global(jn, kn)                   # raw_ptr_buffer_load + soffset
        if kn == 0: prefetch_wscale_to_lds(jn)

        rB = ds_read(B_cur)                                # 16 x ds_read_b128 (128x128 fp8)

        if k == 0 and j > 0:                               # ---- 上一个 N 块的 epilogue ----
            for t in range(0, 16, 4):                      # 每次 4 个 C atom -> w_scale 只占 16 regs
                ws = ds_read(lds_wscale[(j-1)%2], atoms=t..t+3)
                v  = acc[t*4 : t*4+16] * ws                # v_mul_f32
                bf = pack_bf16(v, rTok)                    # v_fma_f32 + v_perm_b32
                ds_write(lds_cshuffle[wave], bf)           # 128-bit
            s_waitcnt lgkmcnt(0)
            for r in range(4):
                o = ds_read_b128(lds_cshuffle[wave], r)
                raw_ptr_buffer_store_dwordx4(out_rsrc, o, soffset=..., aux=nt)
            acc = 0
        elif k == 1 and j > 0:
            for r in range(4, 8):
                raw_ptr_buffer_store_dwordx4(out_rsrc, held_bf[r], soffset=..., aux=nt)

        s_waitcnt lgkmcnt(0)
        stage_end()

        # ============ STAGE 2 : MFMA only ============
        s_setprio(1)
        for n in range(NT):            # 8   weight N tile
          for ks in range(KS):         # 4   k-step (32)
            for m in range(MT):        # 2   act M tile
              acc[n, m] = mfma_f32_16x16x32_fp8(rB[n, ks], rA[m, k*KS + ks], acc[n, m])
        stage_end()                    # 64 x 16 clk = 1024 clk

    # ---------------- epilogue ----------------
    drain_and_store(acc, block_n = nBN-1)
    if grp == 0: stage_end()           # 补齐 barrier 代次
    s_setprio(0)
```

稳态时间线（每格 ≈1024 clk）：

```text
slot :     2i          2i+1        2i+2        2i+3        2i+4
grp0 :  mem(i)     |  MFMA(i)  |  mem(i+1) |  MFMA(i+1)|  mem(i+2)
grp1 :  MFMA(i-1)  |  mem(i)   |  MFMA(i)  |  mem(i+1) |  MFMA(i+1)
         ^ 每个 SIMD 恒有 1 wave 占 matrix core, 1 wave 占 LDS/VALU/VMEM
LDS  : buf[i%2] 被读        buf[(i+1)%2] 被写满     buf[i%2] 可回收
```

---

## 8. 实现约束清单（本轮明确要求）

| # | 约束 | 落地位置 |
| --- | --- | --- |
| 1 | 输出经 LDS，`ds_write` 128-bit，`buffer_store_dwordx4` 出去 | §3.3 |
| 2 | `raw_ptr_buffer_load/store` + `readfirstlane` 设 wave 级 `soffset`；核心循环无地址 VALU | §3.4，ISA 复查阶段验证 |
| 3 | weight 沿用 `moe_gemm_8wave_down` 的 preshuffle | §3.5 |
| 4 | 无 `v_pk_*`（保留 `-packed-fp32-ops`） | §3.6，ISA 复查 `grep -c v_pk_` == 0 |
| 5 | 先 fp8 PTPC；bf16 记 TODO | §9 |
| 6 | profiler 用 ATT（`rocprofv3 -i input.yaml`，`advanced_thread_trace: true`，`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`） | 见 `FlyDSL/.claude/skills/capture-kernel-trace` |
| 7 | 代码改动只在 `pyhip` 模块内 | `src/contrib/flydsl/moe_gemm_2stage/` |

ATT 重点观察项（对应 §5 的三条论证）：

1. 两组 wave 是否真正反相 —— 同 SIMD 的 wave $i$ / $i+4$ 的 MFMA 区段应互补无重叠；
2. compute stage 内是否出现非 MFMA 指令（应为 0）；
3. mem stage 的 `ds_read` 是否被 MFMA 完全遮盖 —— 看 `SQ_WAIT_ANY` 是否集中在 barrier 前。

---

## 9. 风险与 TODO

### 9.1 当前环境阻塞（实测未开始的原因）

| # | 现象 | 影响 |
| --- | --- | --- |
| B1 | 容器内 FlyDSL 为 `v0.3.2`（`/usr/local/.../flydsl`，与 `/host_root/FlyDSL@dd837334` 同版本），**不提供 `flydsl.expr.buffer_ops`**；而 `gemm2_1x4.py`(×7)、`gemm2_2x4.py`(×5)、`gemm2_1x8.py`(×3) 都在用 `fx.buffer_ops.create_buffer_resource/buffer_load/buffer_store` | 现有 down kernel 与新 `gemm2_8x1` 在本机都无法 JIT 通过。v0.3.2 的等价 API 是 `fx.rocdl.make_buffer_tensor` + `fx.rocdl.raw_ptr_buffer_load/store` |
| B2 | `/opt/aiter` 源码在 `10b192f5b`，但 `aiter/jit/*.so` 停留在 8-20；`import aiter` 抛 `module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'` | `tests/contrib/moe/test_moe.py` 全链路不可运行，只能用自建 metadata 的独立 harness |

已完成的绕行：`/tmp/pyhip_ws.py` 把 editable install（指向陈旧的 `/opt/pyhip/src`）重定向到
`/host_root/pyhip/src`，并对 `aiter.ops.flydsl.kernels.tensor_shim` 打桩；
`compile_moe_gemm2(down_path="8x1")` 的 builder 侧已 `BUILD OK`；kernel 侧在 launch 触发 trace 时停在 B1。

### 9.2 设计风险

| 项 | 说明 | 状态 |
| --- | --- | --- |
| **M padding（头号风险）** | `TILE_M_DOWN=256` 后每 expert 最多浪费 255 行。397B@32K 浪费 20%，吃掉大半收益；小 batch（1K，20 行/expert）完全不可用 | host 侧 selector 必须按 `rows_per_expert` 门控；尾块混合派发到 `2x4`(BM=128) → TODO |
| LDS 81% | 余量小，epilogue 必须摊到两个 stage | 已在设计中处理；若 ATT 显示 LDS 阻塞，切 §3.3 备选 |
| `K=192` | `BK=128` 不整除。`BK=96` → nBK=2、MFMA/stage 降到 768 clk，此时 LDS 944>768 必须先切备选 epilogue | TODO |
| `K=128` | nBK=1，每个 N 块只有一个 mem stage，epilogue 无法摊分 → LDS ≈1088/1024，overrun 6% | 可接受，实测确认 |
| `K=384` | nBK=3，`frag_act` 48 regs，总计 ≈238 | 可行，H3 用；二期 |
| `K=512` | `frag_act` 64 regs → 总计 ≈254，且 ping-pong 不变仍 32 KiB | TODO |
| bf16 | `frag_act`/`frag_weight` 均翻倍 → 必然爆寄存器，需把 BN 降到 64 | TODO |
| 1 WG/CU | 没有第二个 WG 兜底，prologue 的 A gather（~1 µs）完全暴露；单 WG 约 73 µs，占比 ~2%。tail：1536/80=19.2 轮 → ~4% | 可选：套用 `moe_gemm_8wave_down` 的 persistent + atomic task queue |
| output 写成为第一大流量 | 16 B/clk/CU（43%） | 可用 `exec` mask 掉 padding 行的 store |
| H3 收益置信度 | H3 基线本就不受带宽约束（48.8%），收益来自 core 利用率而非带宽，§0.3 的预测不适用同一模型 | 实测优先级排后 |

---

## 10. 变更记录

| 日期 | 内容 |
| --- | --- |
| 2026-09-04 | 建档。完成 roofline 诊断（§0.2，确认 K=256 三个 case 卡在 HBM ~82%）、几何/寄存器/LDS 预算、两段反相遮盖论证（含 co-issue 实测依据）、伪代码。实现与实测未开始。 |
| 2026-09-04 | 落地 `gemm2_8x1.py` v1（正确性优先版：完整 8x1 拓扑 + LDS ping-pong + A 常驻 VGPR + CShuffle 写出；两段反相 barrier 留到 v2），并在 `gemm2.py` 注册 `down_path="8x1"`。推导出 preshuffle 权重的等价闭式 `addr(c,k)=(c/16)*16K+(k/16)*256+(c%16)*16+(k%16)`，据此确定 LDS 排布与免 swizzle 的 128-bit `ds_read`。 |
| 2026-09-04 | 因 §9.1-B1，将所有 `fx.buffer_ops.*` 改写为 v0.3.2 原生的 `fx.rocdl.make_buffer_tensor` + copy atom。修复两个 bug：（1）`load_w_scale` 缺双缓冲，预取 N+1 时覆盖了 N 的 epilogue 所需 scale；（2）weight 预取边界条件写错，最后一个 N 块的末尾 K chunk 被跳过（仅 nBK≥3 暴露）。删除 CShuffle 里多余的 workgroup barrier（该区 wave 私有），并将每个 K step 的 barrier 从 2 个减为 1 个（1.559 -> 1.389 ms）。首轮实测写入 §0.4。 |
| 2026-09-04 | 尝试两段反相（含滚动 `vmcnt` 版本）与 `frag_sorted_weight` 标量化，均回退。随后发现**测量协议缺陷**：同一份代码 run-to-run 漂移 8.9%，导致此前 3 个结论作废，详见 §0.5。补齐交接材料：把 harness 从 `/tmp` 落进仓库（`tests/contrib/moe/test_down_8x1.py` + `_env_workaround.py`），新增 §11。 |

---

## 11. 交接（换机器继续）

### 11.1 当前状态一句话

`gemm2_8x1` v1 **功能正确、资源达标、性能未达标**：稳态 1.389 ms / 197.9 useful TFLOPS
（Qwen3.5 35B K=256 @32K），约为 §0.1 历史基线的 0.55x。瓶颈定位尚未完成——
下一步必须用 ATT 实测，不要继续盲调（理由见 §0.5）。

### 11.2 代码位置

| 内容 | 路径 |
| --- | --- |
| kernel | `src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py` |
| path 注册 | `src/contrib/flydsl/moe_gemm_2stage/gemm2.py`（`_BUILDERS["8x1"]`） |
| 正确性 + bench harness | `tests/contrib/moe/test_down_8x1.py` |
| 环境修补（可选） | `tests/contrib/moe/_env_workaround.py` |
| 本文档 | `docs/design_moe_gemm2_8x1.md` |

分支 `luocheng/moe-down-8wave`。改动只在 `pyhip`，未动 `aiter` 源码与 `FlyDSL`。

### 11.3 环境要求与已知坑

| 项 | 要求 / 现象 |
| --- | --- |
| GPU | gfx942。本轮用 MI308X（80 CU）；§0 的 peak 常数 589.8 TFLOPS 与 5.3 TB/s 按此机型推导，换机型需重算 |
| FlyDSL | **v0.3.2**。`gemm2_8x1.py` 已按该版本 API 编写。注意现有 `gemm2_1x4/2x4/1x8` 仍在用**不存在**的 `fx.buffer_ops`，在 v0.3.2 上无法编译——想做同机 A/B 对比必须先同样改写它们 |
| aiter | 若 `import aiter` 报 `module_aiter_core has no attribute ...`，删掉 `aiter/jit/module_aiter_core.so` 让 JIT 重建。本轮重建后又遇 `gluon kernels require triton>=3.6.0`（实测 3.5.1），故 `test_moe.py` 全链路仍不可用 |
| pyhip | 若 editable 装在别的 checkout 上，`sys.path`/`PYTHONPATH` 盖不住（MetaPathFinder 优先）。`_env_workaround.py` 会改写 `__editable___pyhip_1_0_0_finder.MAPPING` |
| 时钟 | **必须锁频**，且**弃掉每轮首次运行**，见 §0.5 |

### 11.4 复现命令

```bash
# 正确性（单配置）
python3 tests/contrib/moe/test_down_8x1.py test --k 256 --n 256

# 正确性（全矩阵，pytest）
pytest tests/contrib/moe/test_down_8x1.py -k correctness

# 性能：repeat>=3 且丢弃首次
python3 tests/contrib/moe/test_down_8x1.py bench --case qwen35b_k256 --repeat 3

# 资源（LDS / scratch / VGPR / AGPR）
rocprofv3 --kernel-trace -f csv -o /tmp/k8x1 -- \
  python3 tests/contrib/moe/test_down_8x1.py bench --case qwen35b_k256
# 读 /tmp/k8x1_kernel_trace.csv 的 LDS_Block_Size / Scratch_Size / VGPR_Count / Accum_VGPR_Count
```

已验证基线：`K x N` ∈ {128,256,384} x {128,256,384,512} 全部 PASS，相对误差 0.37%–0.68%；
LDS 50176 B、scratch 0、VGPR 108 + AGPR 132 = 240。

### 11.5 下一步（按优先级）

1. **ATT 抓取**（`FlyDSL/.claude/skills/capture-kernel-trace`）。要回答两个问题：
   - mem stage（16 条 `ds_read_b128` + waitcnt）实际占多少 cycle？§5.1 假设 600–700 clk、
     compute stage 1024 clk，但反相实测变慢，说明这个假设错了；
   - §3.5 声称的「免 swizzle、零 bank conflict」`ds_read` 布局是否成立？该结论是纸上推导、
     从未验证，是当前最可疑的一环。
2. 依 ATT 结果决定：若确为 LDS conflict，改 LDS 排布（加 XOR swizzle）；
   若为延迟未隐藏，才考虑 `frag_weight` 双缓冲（需先腾出 64 VGPR）。
3. 反相改造（§5/§6）——**必须在 1、2 之后**，且要与 epilogue 摊入 mem stage 同时上线。
4. `raw_ptr_buffer_load/store` + `readfirstlane` soffset（§3.4），ISA 复查核心循环无地址 VALU。
5. host 侧 selector 按 `rows_per_expert` 门控（§9.2 的 M padding 风险）。

### 11.6 需要重新验证的结论

| 结论 | 状态 |
| --- | --- |
| §0.2 roofline 诊断（K=256 卡在 HBM ~82%） | 由 §0.1 历史实测推导，可信 |
| §0.3 8x1 目标预测 | 仍是**未验证的解析预测** |
| §4.1 寄存器预算 | 总数对（预测 222 / 实测 240），但**未区分 VGPR/AGPR 两池**，需按 §0.5 修正 |
| §5.1 每 stage 周期账 | **与反相实测矛盾，很可能是错的**，待 ATT |
| §3.5 免 swizzle 零 conflict | **纸上推导，从未验证**，待 ATT |
| §5.3 VALU co-issue 核算 | 依据是实测 co-issue 表，但本 kernel 内未验证 |
