# Paged Prefill 4-wave/8-wave 优化与性能报告

## 最新：8-wave 可选 persistent（2026-09-05）

当前8-wave工厂支持 `persistent=True`，默认仍static。设备端工作队列、每stream独立
8-byte header、CTA结束自动重置；热路径仍单kernel/零KV workspace。
最终 **236 passed / 6 skipped**，默认static六种汇编与第1阶段逐条一致。

同输入本轮 H3 BF16 page64：static **34648.100 µs**、persistent **31564.267 µs**、
4dynamic **31562.353 µs**；8-wave开启persistent降低8.90%，接近4dynamic。
SWA D128/D192分别 **101.649→98.352 / 116.698→113.855 µs**；full基本持平。
完整采样、真实PMC任务数、少量D192persistent spill与graph并发限制见
[persistent验收](../pa_8wave/persistent.md)。以下全矩阵保留原来static数据，不混入新结果。

## 8-wave tile API 重构（2026-09-05）

当前 direct-paged 8-wave 的 QK/PV 已改用 tiled-MMA fragment + `fx.gemm`，
保留 OPUS 八阶段流水。D128/D192 × full/causal/SWA 六种 specialization 的
**每条汇编指令及操作数、寄存器/LDS/spill 都与重构前一致**。
完整回归 **181 passed / 6 skipped**。首轮同进程延迟差 -0.16%～+1.09%，反序复测
短路径 -0.57%～+0.01%，未观察到可重复下降；不宣称 refactor 本身加速。
原始样本、逐指令核对及复现见 [tile 重构验收](../pa_8wave/tile_refactor.md)。

## 统一全矩阵重测（tile 重构前，2026-09-05）

本轮按本文的可运行 workload 矩阵重新测量：**22 种 workload × BF16/FP8 = 44 组配置，
127 个候选结果、635 个轮次样本**，并复核 **30 个 specialization 资源、4 个后端 PMC、
当前 8-wave ATT**。本轮**没有修改 attention 内核**。

- 当前 8-wave：[pa_8wave_950.py](../pa_8wave/pa_8wave_950.py)，已含 packed-V/控制等待优化，
  BF16 page64 direct-only，非 persistent；不能替换成旧 FP8/persistent 8-wave。
- 当前 4-wave：[pa_prefill_4wave.py](pa_prefill_4wave.py)，BF16/架构原生 FP8；batch1
  static 与强制 dynamic 分开测量，batch>1 只有 dynamic，不能将默认路径误标为 static。
- 统一入口：[benchmark_readme.py](benchmark_readme.py)。全部精确五轮采样、shape 定义、
  source hash、N/A 原因、回归与 profiler 记录见 [readme_retest_results.json](readme_retest_results.json)。
- **以下“当前”表均为本轮新测数据**；此前的版本 A/B、gather、gfx942 以及否决实验统一
  折叠在文末历史记录，不替换其历史样本、不混入当前速度结论。

### 计时、正确性和布局口径

MI350X gfx950、GPU0、256 CU，PyTorch 2.9.1 / ROCm 7.2，现有 auto-DPM，未改时钟或功耗。
本机可见的 8 张卡均为 gfx950，**没有 gfx942 硬件**。主配置为
`B1/Hq16/Hkv1/Dv128/page64`；额外配置在对应表中注明。

每个配置内同进程、相同逻辑 Q/K/V、同一原始 5D cache/随机物理页表、预分配输出，
无 LSE。100 轮共同预热，随后 **5 轮交替候选顺序，每候选每轮 20 warmup / 100 次迭代**，
`num_rotate_args=1`；`run_perftest` GPU profiler 每轮过滤均值的中位数。计时期间无
ATT/PMC，不含编译、首次分配、CPU dispatch、reference 或 linear KV 准备。

**每个已计时候选先对独立、分块 FP32 reference 检查全部输出**，包括 H3 长序列，
`checkAllclose` 错误比例必须为 0，并检查三次 bit-exact 重复。BF16 容差 `2e-2`，
FP8 容差 `1e-1`；FP8 reference 来自实际量化后输入的反量化值。性能输入尾页统一补零，
内核完整回归仍包含 NaN poison。未使用 `PA_SKIP_REFERENCE` 来获取当前表数据。

- `AITER linear` 使用 `flash_attn_varlen_func`：D128 full 是 ASM，D192 full 是 OPUS，
  SWA 是 CK。**不把整列称为 OPUS**。
- `OPUS linear` 显式调用专用入口：D128 dense4D、D192 packed3D group，profiling 已确认
  `gqa_d128_kernel` / `gqa_d192_v128_kernel`。它不支持对应 5D ABI，转换不计时，
  **不能称为 OPUS 分页端到端性能**。
- `AITER 5D` 在全部 22 组 BF16 配置中实际调用，page32/page64 均返回
  `no matching kernel found`，所以以下表不重复一整列 N/A；没有退回 linear。
- 当前 8-wave 的 FP8/page32 为不支持；D128 ragged H3、SWA/sink、FP8 无可比 OPUS。
  FP8 AITER 未配置与本轮 per-token Q scale 完全一致的 baseline，标 N/A，不推断其他
  AITER FP8 API 都不可用。FP8 4-wave 为 QK K64 / PV K16 当前分支。
- TFLOPS 按有效 QK/PV 计算；causal 精确计入对角线，SWA128 为最多 **129** 个可见 key。
  TB/s 按 Q/O 与页对齐可见 KV 并集的最小逻辑字节计算，**不是 HBM counter**。

### BF16 full attention：同轮 5D / linear 对照

时间单位 **µs**。所有 `5D` 列均含页表 lookup 和 attention，未扣除分页开销。

| 场景 | Dqk | 8-wave 5D | 4-wave static 5D | 4-wave dynamic 5D | AITER linear | 显式 OPUS linear |
|---|---:|---:|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 257.863 | 269.439 | 274.641 | 253.526 | 241.810 |
| noncausal Q10240/KV2583 | 192 | 296.015 | 318.116 | 325.063 | 292.603 | 292.013 |
| causal Q=KV32768 | 128 | 4499.220 | 4929.746 | 5509.998 | 4562.585 | 4713.127 |
| causal Q=KV32768 | 192 | 5347.772 | 6461.789 | 7142.503 | 5298.062 | 5298.313 |

D192 noncausal 当前 8-wave 比 4-wave static 延迟低 **6.95%**，比 OPUS linear 高
**1.37%**；causal32K 分别低 **17.24%**、高 **0.93%**。这些是本轮对应形状结果，
不是所有 head/序列长度的保证。

### BF16 SWA+sink：Q16K / window_left128

逐 head FP32 sink logit 从 -1 到 1。OPUS 不支持该语义，N/A。

| Dqk | Total KV | 8-wave 5D | 4-wave static 5D | 4-wave dynamic 5D | AITER linear |
|---:|---:|---:|---:|---:|---:|
| 128 | 32K | 101.598 | 82.887 | 89.050 | 116.719 |
| 128 | 64K | 101.254 | 83.370 | 88.191 | 117.799 |
| 128 | 128K | 101.281 | 82.288 | 89.314 | 116.376 |
| 192 | 32K | 116.690 | 96.668 | 104.010 | 127.453 |
| 192 | 64K | 116.777 | 96.783 | 103.360 | 126.525 |
| 192 | 128K | 115.637 | 96.405 | 102.715 | 127.488 |

**SWA 仍是 4-wave static 最快**；D192/KV128K 的 8-wave 比 static 慢 **19.95%**。
总 KV32K→128K 时耗时基本不变，符合可见页范围裁剪。

主 BF16 候选有效吞吐（`TFLOPS / 逻辑 TB/s`）：

| 场景 | Dqk | 8-wave | 4-wave static | 4-wave dynamic | AITER linear | OPUS linear |
|---|---:|---:|---:|---:|---:|---:|
| noncausal | 128 | 840.28 / 0.3304 | 804.18 / 0.3162 | 788.95 / 0.3103 | 854.66 / 0.3361 | 896.06 / 0.3524 |
| noncausal | 192 | 914.98 / 0.3598 | 851.41 / 0.3348 | 833.21 / 0.3277 | 925.65 / 0.3640 | 927.52 / 0.3647 |
| causal32K | 128 | 977.54 / 0.0634 | 892.17 / 0.0579 | 798.22 / 0.0518 | 963.97 / 0.0625 | 933.18 / 0.0605 |
| causal32K | 192 | 1028.04 / 0.0667 | 850.81 / 0.0552 | 769.72 / 0.0499 | 1037.69 / 0.0673 | 1037.64 / 0.0673 |
| SWA KV128K | 128 | 170.95 / 1.4087 | 210.41 / 1.7338 | 193.86 / 1.5974 | 148.78 / 1.2260 | N/A |
| SWA KV128K | 192 | 187.16 / 1.5422 | 224.50 / 1.8499 | 210.71 / 1.7363 | 169.76 / 1.3989 | N/A |

### FP8：当前 4-wave static/dynamic

OCP `float8_e4m3fn`，Q per-token/head scale、K/V scalar scale，输出 BF16。当前 8-wave
无 FP8 路径，不使用历史 8-wave 代替。时间 **µs**：

| 场景 | Dqk | KV | static | dynamic | static TFLOPS | static 逻辑 TB/s |
|---|---:|---:|---:|---:|---:|---:|
| noncausal Q10240 | 128 | 2583 | 200.050 | 216.693 | 1083.12 | 0.3178 |
| noncausal Q10240 | 192 | 2583 | 222.596 | 240.926 | 1216.77 | 0.3335 |
| causal Q32768 | 128 | 32768 | 3765.339 | 4202.290 | 1168.07 | 0.0557 |
| causal Q32768 | 192 | 32768 | 4341.750 | 4730.104 | 1266.25 | 0.0565 |
| SWA Q16K/W128 | 128 | 32K | 59.645 | 71.415 | 290.29 | 1.7586 |
| SWA Q16K/W128 | 128 | 64K | 59.404 | 71.040 | 291.46 | 1.7657 |
| SWA Q16K/W128 | 128 | 128K | 60.460 | 72.181 | 286.37 | 1.7349 |
| SWA Q16K/W128 | 192 | 32K | 68.799 | 80.266 | 314.58 | 1.7838 |
| SWA Q16K/W128 | 192 | 64K | 68.780 | 80.722 | 314.67 | 1.7843 |
| SWA Q16K/W128 | 192 | 128K | 68.705 | 80.619 | 315.01 | 1.7862 |

其余候选及配置的 TFLOPS/TB/s 在统一入口输出的完整 markdown 表中列出，亦可由原始
shape 和 latency 精确重算。BF16 与 FP8 的数值/吞吐不当作等精度的横向速度比较。

### page32、batch4、单 head 扩展矩阵

所有扩展场景仍在本轮 gfx950 上重测，**不是将历史 gfx942 数字更名**。D192 使用 Hq16/Hkv1；
单 head 与 batch4 D128 使用 Hq1/Hkv1。batch4 每序列 Q10240/KV2560；单 head full
为 Q=KV40960，causal 为 Q=KV32768；page32 的普通 NC/C 分别为 Q10240/KV2583 和 causal32K。
时间 **µs**，N/A 按上方支持规则解释：

| 场景 | page | dtype | 8-wave 5D | 4-wave static | 4-wave dynamic | AITER linear | OPUS linear |
|---|---:|---|---:|---:|---:|---:|---:|
| D192 noncausal | 32 | BF16 | N/A | 317.039 | 319.772 | 291.798 | 292.159 |
| D192 causal32K | 32 | BF16 | N/A | 6416.671 | 7001.488 | 5239.548 | 5242.735 |
| batch4 D192 | 32 | BF16 | N/A | N/A | 1274.182 | 1085.817 | 1085.394 |
| batch4 D128 H1 | 32 | BF16 | N/A | N/A | 88.969 | 77.792 | 76.971 |
| single-head full | 32 | BF16 | N/A | 1119.578 | 1112.622 | 1027.317 | 961.664 |
| single-head causal | 32 | BF16 | N/A | 634.201 | 620.533 | 592.812 | 533.133 |
| batch4 D192 | 64 | BF16 | 1086.783 | N/A | 1275.696 | 1087.917 | 1087.236 |
| batch4 D128 H1 | 64 | BF16 | 77.860 | N/A | 90.247 | 77.912 | 76.965 |
| single-head full | 64 | BF16 | 1006.462 | 1113.613 | 1109.627 | 1023.800 | 954.599 |
| single-head causal | 64 | BF16 | 612.997 | 623.598 | 628.860 | 596.321 | 534.068 |
| D192 noncausal | 32 | FP8 | N/A | 225.526 | 242.827 | N/A | N/A |
| D192 causal32K | 32 | FP8 | N/A | 4403.572 | 4726.341 | N/A | N/A |
| batch4 D192 | 32 | FP8 | N/A | N/A | 915.925 | N/A | N/A |
| batch4 D128 H1 | 32 | FP8 | N/A | N/A | 74.457 | N/A | N/A |
| single-head full | 32 | FP8 | N/A | 943.523 | 941.417 | N/A | N/A |
| single-head causal | 32 | FP8 | N/A | 581.083 | 586.964 | N/A | N/A |
| batch4 D192 | 64 | FP8 | N/A | N/A | 908.062 | N/A | N/A |
| batch4 D128 H1 | 64 | FP8 | N/A | N/A | 74.909 | N/A | N/A |
| single-head full | 64 | FP8 | N/A | 941.869 | 945.242 | N/A | N/A |
| single-head causal | 64 | FP8 | N/A | 567.372 | 572.345 | N/A | N/A |

### H3 长序列：segments=(63225,7)，Hq=Hkv14，D128

两段均 noncausal，完整 FP32 reference 通过；本轮采用统一 5轮/100迭代 protocol，
不沿用旧 H3 的 3warmup/10event 数字。时间 **µs**：

| page | dtype | 8-wave 5D | 4-wave dynamic 5D | AITER linear |
|---:|---|---:|---:|---:|
| 32 | BF16 | N/A | 31001.906 | 35201.709 |
| 64 | BF16 | 34665.584 | 31583.970 | 35167.792 |
| 32 | FP8 | N/A | 25426.733 | N/A |
| 64 | FP8 | N/A | 25094.570 | N/A |

H3 的 4-wave static 不支持 batch2，显式 OPUS D128 没有 ragged entry。
**page64 BF16 H3 的 8-wave 比 4-wave dynamic 慢 9.76%**，因此不能把主形状的优势
推广到 H3。当前 8-wave 无 persistent，batch4/H3 使用静态三维 grid。

### 全部正确性、兼容测试与失败记录

| 检查 | 本轮结果 |
|---|---|
| 当前 8-wave 完整 suite | **181 passed、6 skipped**；skip 为缺失 AITER5D page64 |
| 4-wave 完整默认 suite | **51 passed、2 skipped**；skip 为两个 opt-in 性能测试 |
| 统一 44组配置/127候选 | **全部 FP32 reference 错误比例0**；三次重复一致 |
| 旧 non-SWA opt-in | **失败：进程退出134，GPU memory fault**，不是 skip 或通过 |
| 旧 SWA opt-in（独立进程） | **1 passed**；旧 batch-prefill128K 已知 fault guard 未运行 |
| page32/64/128、短尾、batch、stream/counter复用 | 由完整 suites 覆盖；page128 未另设历史生产计时行 |
| gfx942 | 当前机器无此架构，**未重测**，旧记录保留为历史 |

旧 non-SWA opt-in 在 `run_dispatched_aiter` 的 **linear/page1 causal Q=KV32768** 调用后
同步处出现 GPU memory fault；无新计时可报告，不沿用此前“生产测试通过”的结论。
它与本轮成功的 `flash_attn_varlen_func` / 显式 OPUS 不同。原日志索引在结果 JSON；
故障后新进程的 GPU 健康检查成功。此任务未尝试通过改 kernel 或放宽校验来隐藏故障。

现有 [test_pa_prefill.py](test_pa_prefill.py) 的部分比较使用历史
[pa_prefill_8w32x32.py](../pa_8wave/pa_prefill_8w32x32.py)，因此以下旧 SWA opt-in 新测值
**单列为兼容记录**，不是当前 8-wave。它采用另一 seed、CUDA-event 20/100/5 与全量
Triton gather，不能与上方 GPU-profiler 表混算；单位 µs：

| KV | gather | batch-prefill only | gather+batch | varlen only | gather+varlen | 4-wave static | 4-wave dynamic | 历史8-wave |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 33.28 | 259.57 | 280.69 | 117.20 | 148.09 | 77.48 | 88.02 | 123.22 |
| 64K | 62.26 | 259.87 | 323.35 | 118.02 | 180.29 | 78.08 | 85.48 | 123.56 |
| 128K | 153.45 | 未运行：已知fault | 未运行：已知fault | 122.12 | 269.03 | 79.34 | 85.04 | 124.12 |

### 新资源：30 个 specialization

均为 page64、无 LSE，重新生成 ISA，表项为 **VGPR / SGPR / LDS bytes**；全部 private=0、
VGPR/SGPR spill=0。按 full-NC、causal32K、SWA128K 分别列出，不用某一个 shape 的资源
代表所有实例。

| dtype | 模式 | Dqk | 8-wave | 4-wave static | 4-wave dynamic |
|---|---|---:|---|---|---|
| BF16 | noncausal | 128 | 226 / 55 / 99840 | 180 / 74 / 16640 | 186 / 100 / 16644 |
| BF16 | noncausal | 192 | 256 / 59 / 149760 | 212 / 74 / 24960 | 218 / 100 / 24964 |
| BF16 | causal | 128 | 221 / 83 / 99840 | 182 / 74 / 16640 | 184 / 96 / 16644 |
| BF16 | causal | 192 | 254 / 89 / 149760 | 214 / 74 / 24960 | 216 / 96 / 24964 |
| BF16 | SWA | 128 | 228 / 69 / 99840 | 173 / 48 / 16640 | 174 / 68 / 16644 |
| BF16 | SWA | 192 | 256 / 71 / 149760 | 208 / 46 / 24960 | 209 / 68 / 24964 |
| FP8 | noncausal | 128 | N/A | 148 / 74 / 16384 | 156 / 98 / 16388 |
| FP8 | noncausal | 192 | N/A | 166 / 74 / 16384 | 176 / 99 / 16388 |
| FP8 | causal | 128 | N/A | 148 / 74 / 16384 | 153 / 96 / 16388 |
| FP8 | causal | 192 | N/A | 166 / 74 / 16384 | 172 / 96 / 16388 |
| FP8 | SWA | 128 | N/A | 145 / 46 / 16384 | 149 / 66 / 16388 |
| FP8 | SWA | 192 | N/A | 164 / 46 / 16384 | 168 / 66 / 16388 |

### 新 PMC / ATT：D192 noncausal Q10240/KV2583

[profile_readme.py](profile_readme.py) 使用同一 workload；每后端采预热后的第21–23次，
指令组和 utilization/cache 组独立运行。下列为三次中位数，不与性能表混用计时。

| 指标 | 8-wave 5D | 4-wave static | 4-wave dynamic | OPUS linear |
|---|---:|---:|---:|---:|
| `SQ_WAVES` | 5120 | 5120 | 2048 | 5120 |
| `SQ_INSTS_MFMA` | 8396800 | 8294400 | 8294400 | 8396800 |
| `SQ_INSTS_VALU` | 40509440 | 47052800 | 47639042 | 40092160 |
| `SQ_INSTS_VMEM_RD` | 1157120 | 4869120 | 4884480 | 1152000 |
| `SQ_INSTS_LDS` | 8458240 | 5160960 | 5167360 | 11816960 |
| `SQ_LDS_BANK_CONFLICT` | 0 | 491520 | 491520 | 0 |
| `MfmaUtil` | 57.62% | 53.88% | 52.09% | 60.17% |
| `MeanOccupancyPerActiveCU` | 1.9824 | 1.9082 | 1.8805 | 1.9814 |
| `TCC_MISS_sum` | 1171623 | 1081012 | 1133092 | 1123975 |

当前 8-wave 新 ATT：SE0/CU1、三次 capture 共 **72 个完整 wave**；每 wave 动态指令
**12217**、BFI **0**、waitcnt **179**、barrier **333**、MFMA **1640**。
稳定 phase 平均 **2783.72 cycles**。ATT 的 stage 时间/模型覆盖率不是全卡 utilization；
上表 `MfmaUtil` 是另行采集的硬件 metric。旧 ATT/优化推导见
[D192分析](../pa_8wave/d192_noncausal_att.md)，当前 capture 索引已存入结果 JSON。

### 复现当前矩阵

从 pyhip 根目录使用现有 GPU 环境，统一 benchmark 默认即为全部44组：

```bash
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python tests/flydsl/pa_4wave/benchmark_readme.py
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python -m pytest -q tests/flydsl/pa_8wave/test_pa_prefill.py
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python -m pytest -q tests/flydsl/pa_4wave/test_pa_prefill.py
```

`--case` / `--dtype` 可选择真实 workload/dtype 子矩阵；每项输出 `README_RETEST_RESULT`
JSON（五轮采样、精确 dispatch、错误比例与不可用原因）及完整 TFLOPS/TB/s markdown 表。
旧 opt-in 故障路径不属于统一 benchmark 的候选，不自动重试故障。

---

<details>
<summary>历史版本、旧计时协议、优化 A/B 与 gfx942 记录（不是本轮当前结果）</summary>

以下原样保留旧测量和优化过程；文内“当前/最新”均指当时版本。本轮结论只以上方
统一重测为准，旧 gather/core、旧8-wave FP8、已否决代码分支不重建为生产候选。

## 最新：packed-V 合入与控制/等待精简（2026-09-05）

[pa_8wave_950.py](../pa_8wave/pa_8wave_950.py) 已合入 packed-i32 V 拼接，不再是隔离实验。
先仅合入 packed-V，完成 **173 passed、6 skipped**，再修改控制/等待；最终
[完整回归](../pa_8wave/test_pa_prefill.py) **181 passed、6 skipped**。6 个跳过项仍仅为
AITER 缺少 page64 5D 实例。AITER/4-wave 内核、公开 ABI 和数值容差未修改。

### 已合入的改动与安全边界

- V LDS 结果先按 32 个 i32 word 拼接，再一次性转为 BF16；消除反复 BF16 slice store
  与 tail bitcast 之间的冗余 BFI。
- 将 phase S2 同一位置的 VM wait 与 page/LDS wait 合为一条
  `s_waitcnt vmcnt(KEEP_VMCNT) lgkmcnt(0)`。**未提高 VM 阈值、未删除 LDS 等待**。
- noncausal border 改为 `remaining=kv_len-tile*64; remaining<64`；合法 tile 的条件等价，
  不再组合 modulo、最后 tile 除法与布尔条件。mask predicate 仍在分支内物化。
- 正向 main phase 消费 `t-1 <= tiles-2`，必然不是最后一个实际 KV tile，因此编译期
  删除这两处 V tail 检查。**reverse main phase 与 epilogue 的 NaN-tail 清零及 fence
  全部保留**；不是全局禁用 V mask，也没有删减 workgroup barrier。
- 新增 8 项回归：D128/D192 × 正向/causal 首尾配对反向 × 有/无 LSE，同一编译结果
  依次使用 KV1/64/65/128/193/321/320，更新物理尾页 NaN poison，严格 FP32 reference
  与三次 bit-exact 重复，确保不依赖固定 runtime 长度或 padding 内容。
- 保持 **direct-paged、一个 attention kernel、零 KV workspace**。本轮未加入 persistent。

### 同进程分阶段性能

MI350X gfx950，BF16、B1/Hq16/Hkv1/Dv128/page64、无 LSE、预分配输出。三个 FlyDSL
候选使用**同一原始 5D cache/随机页表**；旧源码从已核对 SHA 的 Git blob 只读加载到
独立模块，仅用于本次比较。100 轮共同预热、5 轮交替候选顺序、每轮 20 warmup /
100 次 `run_perftest`，取五轮 GPU 时间中位数；不启用 ATT/PMC。所有候选 `err=0`。
单位 **µs**：

| 场景 | Dqk | 改动前 direct | 仅 packed-V | packed-V + 控制/等待（当前） | 显式 OPUS linear core |
|---|---:|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 266.395 | 259.345 | **258.495** | 241.272 |
| noncausal Q10240/KV2583 | 192 | 303.689 | 299.090 | **296.688** | 291.950 |
| causal Q=KV32768 | 128 | 4578.457 | 4471.397 | **4493.740** | 4697.385 |
| causal Q=KV32768 | 192 | 5443.132 | 5335.193 | **5317.539** | 5284.699 |
| SWA+sink Q16K/KV128K/W128 | 128 | 103.452 | 102.156 | **101.769** | N/A：不支持 |
| SWA+sink Q16K/KV128K/W128 | 192 | 119.056 | 117.129 | **116.521** | N/A：不支持 |

D192 noncausal 相比改动前下降 **2.31% / 7.001 µs**，其中控制/等待在 packed-V
基础上再减少 **2.402 µs**；与本轮 OPUS linear 的差距为 **1.62%**。
六个最终中位数均优于改动前，但 **D128 causal 比仅 packed-V 慢 0.50%**，不能宣称
控制改动对所有 shape 都加速；相比改动前该项仍下降 1.85%。D192 causal 比 OPUS 慢
0.62%，D128 noncausal 仍慢 7.14%。这里不混用不同时轮的中位数。

OPUS 使用同逻辑但预先构建的 linear KV，转换不计时，**不是 OPUS 的 5D 分页端到端
性能**；AITER 5D page64 仍无匹配实例。本轮未重测 4-wave，不把下方历史 4-wave 时间
拼成同轮对照。SWA 的小幅改善不构成超过 4-wave 的证据。

| 当前场景 | Dqk | 有效 TFLOPS | 逻辑 TB/s |
|---|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 838.23 | 0.3296 |
| noncausal Q10240/KV2583 | 192 | 912.90 | 0.3590 |
| causal 32K | 128 | 978.74 | 0.0635 |
| causal 32K | 192 | 1033.89 | 0.0670 |
| SWA+sink | 128 | 170.13 | 1.4019 |
| SWA+sink | 192 | 185.74 | 1.5305 |

SWA 仅按每行 129 个可见 key 计算有效 FLOPs；逻辑 TB/s 不是 HBM counter。
全部五轮样本、source/test SHA、回归与资源记录见
[packed_v_control_results.json](../pa_8wave/packed_v_control_results.json)。

### 最终 ATT 与资源复核

D192 noncausal，SE0/CU1、三个预热后 launch、64 个完整 wave；同一分析器复核：

| 每 wave 指标 | 改动前 | 当前 |
|---|---:|---:|
| 动态指令 | 13502.5 | 12217 |
| BFI | 656 | **0** |
| `s_waitcnt` | 218 | **179** |
| `s_barrier` | 333 | **333** |
| BF16 MFMA | 1640 | **1640** |
| `s_nop` | 652 | 752 |

稳定 phase 平均 **2913.6 → 2787.5 cycles**；这是采样 CTA 的 stage 时间，不作为
整 kernel 加速比例。部分 hazard padding 增加，未声称所有类别都减少。资源均取新生成
ISA metadata，无 LSE：

| Dqk | 模式 | VGPR | SGPR | LDS bytes | Private bytes | VGPR/SGPR spill |
|---:|---|---:|---:|---:|---:|---:|
| 128 | noncausal | 226 | 55 | 99840 | 0 | 0/0 |
| 192 | noncausal | 256 | 59 | 149760 | 0 | 0/0 |
| 128 | causal 配对 | 221 | 83 | 99840 | 0 | 0/0 |
| 192 | causal 配对 | 254 | 89 | 149760 | 0 | 0/0 |
| 128 | SWA+sink | 228 | 69 | 99840 | 0 | 0/0 |
| 192 | SWA+sink | 256 | 71 | 149760 | 0 | 0/0 |

复现入口与完整支持范围见 [8-wave README](../pa_8wave/README.md)。下方保留的是
**本次优化前**的 5D/OPUS、旧 core 和更早实验数据，不能当作当前资源或耗时。

---

## 历史对照：优化前 5D direct-paged 与显式 OPUS（2026-09-05）

[pa_8wave_950.py](../pa_8wave/pa_8wave_950.py) **仅保留 direct-paged attention**。
`prepare_kv`、`attend_linear`、gather kernel 和线性 KV workspace 均已移除；预分配输出
时一次公开调用只启动一个 attention kernel，不做任何 KV 预处理或数据缓存。
D128/D192、V128、BF16 page64、SWA、sink、LSE、ragged、strided Q/O 与 stream/graph
功能保持。当时只增加比较入口、测试和报告，未修改 attention 内核。
当时的[完整测试](../pa_8wave/test_pa_prefill.py) **173 passed、6 skipped**：原 165 项 direct
测试全部通过，另有 8 项比较回归通过；6 项 AITER 5D 测试因缺少 page64 实例跳过。

补充 [D192 noncausal ATT 分析](../pa_8wave/d192_noncausal_att.md)：当前 8-wave 与
OPUS group 均非 persistent。采样定位 V fragment 表示的 656 条额外 BFI/wave；保持
尾页保护的隔离改写在正常计时中为 298.784 µs，原始为 302.930 µs、OPUS 291.796 µs。
当时实验尚未合入；现在已合入并进一步优化，最新结果见本文开头。下表保留旧数据，证据见
[ATT 原始摘要](../pa_8wave/d192_noncausal_att_results.json)。

### 5D 输入、显式 OPUS 入口与计时口径

- MI350X gfx950，BF16，`B=1,Hq=16,Hkv=1,Dv=128,page_size=64`，不输出 LSE。
  5D K 为 `[pages,Hkv,Dqk/8,64,8]`，V 为 `[pages,Hkv,8,128,8]`。
  `flydsl_5d`、两种 `4wave_*_5d` 和 `aiter_5d` 使用**同一份物理 K/V、随机页表、
  实际尾页长度及 Q**；没有为某个 5D 候选重排或更换页大小。
- `aiter_5d` 直接调用 `mha_batch_prefill_func`，**不是 OPUS 入口**。当前构建在六个
  生产形状及六个小型正确性探测上均返回 `no matching kernel found`。
  已核对当前 CK 生成器：`SUPPORTED_PAGE_SIZE = [1, 16, 1024]`，page1 仅用于 linear，
  dispatch 按页大小精确匹配，**没有 page64 实例**。5D 输入通过 ABI 检查；错误附带的
  `>2GB/CDNA3` 通用提示不是本例原因。该列标 N/A，不回退到 linear，也没有改 AITER
  生成器或重新打包 cache 来制造可运行结果。
- `opus_linear` **绕过默认路由，显式调用 OPUS**：D128 使用
  `fmha_fwd_bf16_opus_fwd` 的 dense BSHD 入口，4D view 在计时前创建；D192 使用
  `fmha_fwd_bf16_opus_varlen_fwd` 的 packed 3D group 入口。Profiler 确认实际 kernel
  分别为 `gqa_d128_kernel` / `gqa_d192_v128_kernel`。当前 OPUS 没有对应 5D 页表 ABI，
  也不支持 SWA/sink。它使用由同一物理 cache 重建的相同逻辑 linear KV，**转换不计时**；
  因而这一列是 attention-only baseline，**不是 OPUS 的 5D / 分页端到端性能**。
- 每个形状内所有候选同进程、预分配输出，先对独立 FP32 reference 严格验算，`err=0`
  才测量。所有性能输入统一零填充尾页以兼容旧 4-wave；direct 正确性测试仍用 NaN poison。
  共同预热 100 轮，再做 **5 轮交替候选顺序、每轮 20 warmup / 100 次迭代**，
  `num_rotate_args=1`。每轮值是 `run_perftest` GPU profiler 的过滤均值（去首轮与 IQR
  异常点），下表取五轮中位数；不含编译、首次分配、CPU dispatch 或 reference 时间。

### 优化前性能：5D direct 与 OPUS linear core

单位 **µs**。最后一列为 `8-wave / OPUS - 1` 的延迟差，负值表示 8-wave 延迟更低；
它比较不同输入布局的 attention kernel，不代表同一个分页接口的端到端加速比。

| 场景 | Dqk | 8-wave 5D | 4-wave static 5D | 4-wave dynamic 5D | AITER 5D | OPUS linear core | 8-wave / OPUS 延迟差 |
|---|---:|---:|---:|---:|---|---:|---:|
| noncausal Q10240 / KV2583 | 128 | 264.067 | 269.462 | 274.031 | N/A：无 page64 实例 | 241.051 | +9.55% |
| noncausal Q10240 / KV2583 | 192 | 303.641 | 318.244 | 325.049 | N/A：无 page64 实例 | 292.581 | +3.78% |
| causal Q=KV=32768 | 128 | 4597.028 | 4954.043 | 5536.846 | N/A：无 page64 实例 | 4720.933 | -2.62% |
| causal Q=KV=32768 | 192 | 5444.459 | 6448.438 | 7125.702 | N/A：无 page64 实例 | 5283.595 | +3.04% |
| SWA+sink Q16K/KV128K/W128 | 128 | 101.813 | **81.945** | 87.121 | N/A：无 page64 实例 | N/A：不支持 SWA/sink | — |
| SWA+sink Q16K/KV128K/W128 | 192 | 118.101 | **95.748** | 101.849 | N/A：无 page64 实例 | N/A：不支持 SWA/sink | — |

对应的有效 TFLOPS / 逻辑 TB/s：

| 场景 | Dqk | 8-wave TFLOPS | 8-wave TB/s | OPUS TFLOPS | OPUS TB/s |
|---|---:|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 820.54 | 0.3227 | 898.89 | 0.3535 |
| noncausal Q10240/KV2583 | 192 | 892.00 | 0.3508 | 925.72 | 0.3640 |
| causal 32K | 128 | 956.74 | 0.0620 | 931.63 | 0.0604 |
| causal 32K | 192 | 1009.78 | 0.0655 | 1040.53 | 0.0675 |
| SWA+sink | 128 | 170.06 | 1.4013 | N/A | N/A |
| SWA+sink | 192 | 183.25 | 1.5101 | N/A | N/A |

SWA 的 `window_left=128` 是闭区间，每行 129 个可见 key，sink 为逐 head 的 FP32
logit（-1 到 1）。有效 FLOPs 只计可见 QK/PV；TB/s 按 Q/O 和页对齐的可见 KV 并集
计算最小逻辑流量，**不是 HBM counter**。**SWA 仍是 4-wave static 更快**。
D128 noncausal 比真正 OPUS 慢 9.55%，不能用此前默认 ASM 的时间或“接近旧 FlyDSL
core”来宣称与 OPUS 等速；D192 已测 full 场景仍有约 3–4% 差距。

基准还保留 `aiter_linear` 作为独立补充列：D128 full 实际是 ASM group/causal_group，
D192 full 是 OPUS group，SWA+sink 是 CK Tile；**不把这一整列称为 OPUS**。
精确 kernel 名、全部五轮样本、默认 AITER 补充结果、不可用原因、source/binary hash
及复现参数见 [paged_5d_opus_results.json](../pa_8wave/paged_5d_opus_results.json)。
没有挑选最快一轮；下节的旧 FlyDSL core 验收是另一次测量、另一个 baseline。

### 验收：与删除前 gather 分支的纯 core 比较

MI350X gfx950，B1/Hq16/Hkv1，Q/K/V 和预分配输出相同。**同进程**保留修改前源码
作为一次性外部 reference：只计旧 `attend_linear`，不计 gather；新实现计完整
direct-paged 调用。共同预热后，每候选 20 warmup / 100 次 GPU profiler 采样，
5 轮交替顺序中位数；不将历史不同轮次数字作为验收基准。

| 场景 | Dqk | 新 direct-paged µs | 旧纯 core µs | 新/旧延迟差 |
|---|---:|---:|---:|---:|
| noncausal Q10240 / KV2583 | 128 | **266.163** | 260.183 | +2.30% |
| noncausal Q10240 / KV2583 | 192 | **302.610** | 295.297 | +2.48% |
| causal Q=KV=32768 | 128 | **4558.574** | 4436.378 | +2.75% |
| causal Q=KV=32768 | 192 | **5381.104** | 5292.773 | +1.67% |
| SWA+sink Q16K/KV128K/W128 | 128 | **102.691** | 101.898 | +0.78% |
| SWA+sink Q16K/KV128K/W128 | 192 | **117.633** | 117.045 | +0.50% |

**已测六种代表形状都在旧纯 core 的 3% 内，SWA 在 1% 内。** 这是接近性能，不是
完全等速或任意 shape 的性能保证。原始每轮采样、source hash 与协议保存在
[direct_paged_core_results.json](../pa_8wave/direct_paged_core_results.json)。当前运行路径
没有保留旧 gather 分支。相对旧 core 最大绝对差分别为 noncausal 0.00048828125、
causal 0.0009765625、SWA 0.001953125；独立 FP32 reference 与重复确定性检查均通过。

### 优化前 direct-paged 实现变化与资源

- 仍为 BM256/BN64、512 threads、完整 constexpr STAGGER、八阶段 ping/pong。
- page ID 由 scalar load 提前读取，每个 KV tile 一次，跨 phase 复用到 K、V 和分段 K DMA。
  页表 load 由 `lgkmcnt(0)` 完成，不清空 in-flight KV VMEM 队列。
- K/V 从 SHUFFLE-5D 直接 coalesced 128-bit DMA 到 LDS。K 置换 token bits 2/3，
  score mask 同步置换，P/V operand 因而匹配连续 8-token；V 只用 `ds_read_b128`。
- 尾页 V 在已有访存等待阶段清零，防止 `0*NaN`；显式 LDS wait 后保留 scheduler fence，
  避免 mask 算术前移到异步 read 完成之前。
- SWA 直接跳过窗口外页；重复物理页和跨序列共享页支持。删除 Q16K/KV128K 的 D128
  **64 MiB** / D192 **80 MiB** 线性 KV workspace，辅助 GPU buffer 为 **0 B**。
- 无 LSE 性能 specialization：

| Dqk | 模式 | VGPR | SGPR | LDS bytes | Private bytes | VGPR/SGPR spill |
|---:|---|---:|---:|---:|---:|---:|
| 128 | noncausal | 228 | 74 | 99840 | 0 | 0/0 |
| 128 | causal 配对 | 221 | 83 | 99840 | 0 | 0/0 |
| 128 | SWA+sink | 230 | 71 | 99840 | 0 | 0/0 |
| 192 | noncausal | 256 | 77 | 149760 | 0 | 0/0 |
| 192 | causal 配对 | 254 | 89 | 149760 | 0 | 0/0 |
| 192 | SWA+sink | 256 | 75 | 149760 | 0 | 0/0 |

原 165 项 direct 测试包含 poisoned tails、SWA/sink/空行/LSE、runtime 页表与长度更新、reverse
causal 配对、strides/GQA、graph/stream，以及 **单 GPU launch / 零 workspace** 和共享页检查。
物理 cache byte span 暂限 signed 32-bit offset；其他支持范围和复现命令见
[8-wave README](../pa_8wave/README.md)。基准现在分别输出 `flydsl_5d`、`aiter_5d`、
`opus_linear`、`aiter_linear` 和适用的两种 `4wave_*_5d`；不可用项明确标记且不计时。
OPUS/default AITER 的 linear 数据只用于外部比较，当前 8-wave 实现没有 gather/core 分支。

---

## 历史：gather + OPUS 流水，D128 / SWA / sink（2026-09-05，已删除路径）

**以下整节仅保留改造前数据，不代表当前接口或实现。最新结果以上方 direct-paged 为准。**

本节替代下方 2026-09-03 的“当前”性能结论。实现为
[pa_8wave_950.py](../pa_8wave/pa_8wave_950.py)，测试与同输入多候选基准为
[test_pa_prefill.py](../pa_8wave/test_pa_prefill.py)。旧
[pa_prefill_8w32x32.py](../pa_8wave/pa_prefill_8w32x32.py) 不再代表本节的 8-wave。
本次没有修改 [pa_prefill_4wave.py](pa_prefill_4wave.py) 或 AITER 内核。

### 新实现的功能范围

| 项目 | 当前支持 |
|---|---|
| 架构 / dtype / page | gfx950 / BF16 / page64 SHUFFLE-5D |
| Head dimension | **QK=128 或 192，V=128**；不是把 D128 padding 到 D192 |
| Attention | full noncausal、bottom-right causal、**causal SWA** |
| SWA | `window_left=-1` 禁用；非负整数表示闭区间 `[i+KV-Q-window_left, i+KV-Q]`，128 对应最多 129 个 key |
| Sink | **可独立于 SWA 开启**；`has_sink=True` + 连续 FP32 `sink_ptr[Hq]`，值为未经 QK scale 缩放的自然 logit |
| Sink 语义 | 每 head 一个 value=0 的虚拟 key，只增加 softmax 分母；有限 logit 或 `-inf`（禁用） |
| 输出 | BF16 `out=`；可选 FP32 `LSE[total_q,Hq]`，包含 sink 分母，使用自然对数 |
| 其他 | GQA/MHA、ragged batch、非零 prefix、尾页、非连续 Q/O 的连续 head dimension、scalar/per-token-head Q scale、scalar K/V scale、stream、预热后的 graph |
| 空行 | 空 KV / 全遮罩行 `O=0`；无 sink 为 `LSE=-inf`，有 sink 为 `LSE=sink_logit` |

SWA 与 sink 可分别使用，也可同时使用；SWA 不支持 noncausal。FP8、其他 head dim、
V192、page32/128 和 sink-token 前缀不在这个新内核的范围内，不能套用下方旧实现的支持表。
GPU prefix/page ID/max-length 必须一致，scale 必须有限且为正；热路径不读回 GPU
metadata 做检查，KV byte span 暂限 signed 32-bit offset。

### 计时与数据口径

- **MI350X gfx950，PyTorch 2.9.1 / ROCm 7.2，B=1、Hq=16、Hkv=1、V128、page64**。
- 每行同进程、同逻辑 Q/K/V、随机物理页表、预分配输出；20 次预热、100 次采样、
  5 轮交替候选顺序取中位数。计时使用 `run_perftest` 的 **GPU profiler 时间**，
  不含编译和首次分配；不是下方历史 CUDA-event 数据的续测。
- `8-wave full` = **每次重新 gather + attention**，`core` 只计 attention；两者独立测量。
  不缓存 KV 内容或输出，只复用 workspace 和编译结果。`4-wave` 是 direct-paged，
  不需要 gather。不得拿 8-wave core 当作公开分页接口耗时。
- `AITER linear` 只计已有线性 KV 的 attention；`AITER full` 使用与 8-wave **同一个
  FlyDSL gather** 再调用 AITER。SWA 两边都只 gather 查询窗口并集覆盖的 KV 后缀，
  不是复制全部 128K KV。Full 中位数不是两个独立中位数之和。
- 每个候选先对独立 FP32 PyTorch reference 严格检查，`err=0` 才计时。性能输入的
  padding 统一为 0：旧 4-wave 对 NaN padding 会产生 `0*NaN`，不兼容 poison 输入；
  **新内核 correctness tests 仍使用 NaN poison 尾页**，未放宽测试容差。

### Full attention：当前同轮对照

时间单位均为 **µs**；括号中为 8-wave full 的有效 TFLOPS。

| 场景 | Dqk | 8-wave full | 8-wave core | gather | AITER linear | AITER full | 4-wave static | 4-wave dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| noncausal Q10240 / KV2583 | 128 | **263.273（823.01T）** | 260.286 | 3.581 | 253.762 | 258.071 | 268.635 | 273.778 |
| noncausal Q10240 / KV2583 | 192 | **299.365（904.74T）** | 295.001 | 3.820 | 291.320 | 295.888 | 317.724 | 324.786 |
| causal Q=KV=32768 | 128 | **4439.95（990.59T）** | 4436.34 | 6.695 | 4504.92 | 4557.40 | 4908.97 | 5501.55 |
| causal Q=KV=32768 | 192 | **5334.86（1030.53T）** | 5329.30 | 7.578 | 5281.14 | 5293.83 | 6442.38 | 7108.83 |

目标 D192 noncausal 的完整分页接口比 AITER linear 慢 **2.76%**，比相同分页适配的
AITER full 慢 **1.18%**，比本轮 4-wave static 快 **5.78%**。D128 noncausal full
比 AITER linear 慢 3.75%；D128 causal full 在本轮略快于 AITER linear。以上只说明已测形状，
不是全形状性能保证。

### SWA + sink：Q16K、window_left=128

使用每 head 不同的 FP32 sink（从 -1 到 1）。时间单位 **µs**：

| Dqk | Total KV | 8-wave full | 8-wave core | gather | AITER linear | AITER full | 4-wave static | 4-wave dynamic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32K | 107.234 | 102.030 | 5.168 | 115.626 | 122.071 | **81.934** | 87.280 |
| 128 | 64K | 106.905 | 101.441 | 5.260 | 116.421 | 122.220 | **81.902** | 87.331 |
| 128 | 128K | 107.027 | 102.373 | 5.142 | 116.993 | 121.916 | **82.289** | 87.695 |
| 192 | 32K | 121.993 | 116.671 | 5.864 | 126.538 | 130.533 | **96.057** | 101.626 |
| 192 | 64K | 122.071 | 116.660 | 6.096 | 126.611 | 131.565 | **95.731** | 101.307 |
| 192 | 128K | 122.969 | 116.913 | 5.891 | 125.643 | 131.914 | **95.864** | 102.523 |

128K 上 8-wave full：D128 **161.77 TFLOPS / 1.491 TB/s**，D192
**176.00 TFLOPS / 1.622 TB/s**。TB/s 为算法最小逻辑流量估计（Q/O、可见 KV 并集和
gather 读写），不是硬件 HBM counter；TFLOPS 仅计每行 129 个可见 key，未把被 mask
的 MFMA 工作算成有效 FLOPs。完整 benchmark 输出也列出所有候选的 TFLOPS/TB/s/err。

**SWA 当前仍是 4-wave static 更快。** 128K 上，新 8-wave full 比 AITER linear 快
约 8.52%（D128）/2.13%（D192），但比 4-wave static 慢约 30.06%/28.27%。8-wave
保留 BM256 八阶段 OPUS 流水；窄窗口下一个 WG 处理各 wave 窗口的并集，固定
prologue/barrier 与 masked work 的占比高。这里不把“新增支持”描述成超过 4-wave。
SWA 时间随总 KV 32K→128K 基本不变，验证了后缀 gather 和 tile 裁剪路径。

### AITER 实际路由（本轮 profiler 验证）

| 模式 | 公开入口 | 实际 kernel |
|---|---|---|
| D192 full | `flash_attn_varlen_func` | `gqa_d192_v128_kernel<...>`，OPUS group mode |
| D128 full | `flash_attn_varlen_func` | `aiter::fmha_fwd_hd128_bf16_group` / `...causal_group`，ASM |
| D128 SWA+sink | `flash_attn_varlen_func` | CK Tile `BlockFmhaPipelineQRKSVSAsyncTrload` |
| D192 SWA+sink | `flash_attn_varlen_func` | CK Tile `BlockFmhaPipelineQRKSVSAsync` |

所以 D128/SWA 列标为 **AITER**，不能全称为 OPUS；本次没有强制更改 AITER 的环境路由。
AITER full 的 gather 同样使用本次新实现，不应与历史全量 Triton gather 时间混比。

### 实现与资源

- D128 复用已验证的 D192 八阶段骨架，QK 每个 superunit 从 12 次 MFMA 改为 8 次；
  K DMA 从 3 改为 2，rolling `vmcnt` 从 5 改为 4，LDS 从 149760 B 改为 99840 B。
- SWA 按 query block 裁剪第一/最后 KV tile，mask 用闭区间检查。窄窗口不启用
  causal 首尾配对；full causal 保留配对和镜像反向遍历。
- Sink 只加入一次已完成跨 lane 归约的分母，并随 online maximum 一起重标定；
  无 QK descale 乘到 sink，也不增加 PV 分子。
- SWA gather 按序列裁去不可见 prefix，仍每次读取当前页表和 KV；workspace 保留
  绝对 KV 索引，**只减少搬运，不缩减分配容量**。B1/Hkv1/KV128K workspace 为
  D128 64 MiB、D192 80 MiB；本例只搬后缀 16512 tokens（258 页）。

以下是本轮 fresh ISA，均为无 LSE 性能 specialization：

| Dqk | 模式 | VGPR | SGPR | LDS bytes | Private bytes | VGPR/SGPR spill |
|---:|---|---:|---:|---:|---:|---:|
| 128 | noncausal | 220 | 68 | 99840 | 0 | 0/0 |
| 128 | causal 配对 | 220 | 74 | 99840 | 0 | 0/0 |
| 128 | SWA+sink | 222 | 62 | 99840 | 0 | 0/0 |
| 192 | noncausal | 256 | 71 | 149760 | 0 | 0/0 |
| 192 | causal 配对 | 256 | 80 | 149760 | 0 | 0/0 |
| 192 | SWA+sink | 256 | 65 | 149760 | 0 | 0/0 |

### 测试与复现

完整 strict suite **159 passed**。新增覆盖：D128/D192 × 奇偶/尾页，SWA 0/1/63/64/128/129/512，
有/无 sink，sink -80/0/80/-inf，空 KV/全遮罩 LSE，GQA/非连续 Q/O，scalar/per-token
scale，runtime 长度/页表/sink 改变，graph/stream，无 LSE 重复一致性和 causal 首尾配对。
窗口外页表可设成非法大 ID，证明不会读取不可见 prefix；另有 zero-QK/one-V 的精确
129-key+1-sink 测试，验证输出 `129/130` 与 `LSE=log(130)`，防止 off-by-one/double-sink。

从 pyhip 根目录，使用现有 GPU 环境 `/opt/venv/bin/python`：

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python -m pytest -q tests/flydsl/pa_8wave/test_pa_prefill.py
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python tests/flydsl/pa_8wave/test_pa_prefill.py --head-dim 128 192
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python tests/flydsl/pa_8wave/test_pa_prefill.py --q-len 32768 --kv-len 32768 --causal 1 --head-dim 128 192
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python tests/flydsl/pa_8wave/test_pa_prefill.py --q-len 16384 --kv-len 32768 65536 131072 --causal 1 --window-left 128 --sink 1 --head-dim 128 192
```

---

## 历史基线（以下为 2026-09-03，非新 8-wave 结果）

更新时间：2026-09-03。当前实现已完成gfx942/gfx950 BF16、架构原生FP8、
non-SWA及gfx950 SWA支持。本文件以gfx950当前结果为主；2026-08的gfx942数据统一
归档在“历史结果”中，不与当前表直接横向比较。

## 当前结果总览

当前表均在MI350X `gfx950`上测量，主配置为
`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。除特别说明外，协议均为同进程、
同逻辑输入、20次预热、100个CUDA event样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | AITER | 4-wave static | 4-wave dynamic | 8-wave | 结论 |
|---|---|---:|---:|---:|---:|---|
| non-causal `Q10240,KV2583` | BF16 | **282.27 us / 959.53T** | 304.07 us / 890.74T | 314.05 us / 862.43T（独立轮次） | 350.43 us / 772.89T native | 8-wave API已删除4-wave回退；native 8-wave同轮仍比4-wave慢15.24% |
| causal `Q=KV=32768` | BF16 | 15561.18 us / 353.29T | **5748.63 us / 956.32T** | 7032.47 us / 781.74T | 8462.96 us native（历史） | 8-wave不再自适应到4-wave；本行native数据为历史测量 |
| causal `Q=KV=32768` | OCP FP8 K64 | N/A | **4230.13 us / 1299.62T** | 4815.89 us / 1141.54T | 5163.99 us / 1064.60T | static最快；dynamic为static的0.878x且快于8-wave |
| SWA `Q16K,KV128K,window=128` | BF16 | 123.39 us varlen-only / 269.03 us含gather | **77.94 us / 277.68T有效** | 84.44 us / 256.31T有效 | 142.20 us native（历史） | 8-wave不再自适应到4-wave；本行native数据为历史测量 |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | N/A | **61.00 us / 354.78T有效** | 66.56 us / 325.14T有效 | 113.72 us / 190.31T有效 | static/dynamic分别较8-wave快1.864x/1.708x |

口径说明：

- non-causal FLOPs为`2 * Hq * Q * KV * (Dqk + Dv)`；causal按三角有效工作量减半；
- SWA有效FLOPs按每行最多`window_left + 1 = 129`个可见KV token计算；硬件执行
  TFLOPS另计被mask但仍进入MFMA的tile工作，详见
  [gfx950_swa_performance.md](../pa_8wave/gfx950_swa_performance.md)；
- 每张明细表内的数据可以直接比较；不同架构、dtype、shape或计时协议的数据不混算提升；
- dynamic为同一4-wave kernel的persistent ticket分支；总览中的BF16行来自包含全部候选
  的同轮次测试，FP8的static/dynamic来自同轮次配对测试；
- 8-wave API始终执行原生512线程kernel，不再导入或调用4-wave后端。AITER等长
  D192/V128使用linear/page-size-1 batch-prefill ABI，非等长使用linear THD；
  FlyDSL使用page64 vectorized cache。三者共享逻辑Q/K/V，计时不包含布局转换。

## 当前功能范围

| 架构 | BF16 | 原生FP8 | non-SWA | SWA + sink | page size |
|---|---|---|---|---|---|
| gfx942 | 支持 | `float8_e4m3fnuz` | 支持 | 未作为当前生产基线 | 32/64/128 |
| gfx950 | 支持 | OCP `float8_e4m3fn` | 支持 | BF16/FP8支持 | 32/64/128 |

launcher会拒绝与目标架构不匹配的FP8格式，避免相同8-bit载荷按另一种指数/NaN编码
解释。non-SWA回归覆盖Dqk/Dv为128/128、192/128和192/192、causal/non-causal、
page 32/64/128、per-token/per-tensor Q scale及ragged tail。

## gfx950当前明细

### Non-SWA AITER BF16路由

跨4-wave、8-wave和AITER的BF16结果只保留在顶部总览。对比测试显式按长度分流：

| 条件 | AITER公开入口 | profiler验证的实际事件 |
|---|---|---|
| `Q == KV` | `mha_batch_prefill_func` | `aiter::mha_batch_prefill` |
| `Q != KV` | `flash_attn_varlen_func` | `FlashAttnVarlenFunc` |

默认小shape回归校验non-SWA路由互斥、SWA两种线性AITER入口和三方输出，结果为
`3 passed`。生产性能分别由`PYHIP_RUN_PA_AITER_PERF=1`和
`PYHIP_RUN_PA_AITER_SWA_PERF=1`显式开启。

### Non-causal BF16 native 8-wave瓶颈分析（`Q10240,KV2583`）

AITER在该shape命中gfx950专用OPUS
`gqa_d192_v128_kernel<32,64,8,non-causal,group>`，不是通用CK fallback。下面的PMC
来自同一逻辑输入上的单kernel dispatch；profile延迟与顶部5轮中位数略有差异，但
相对关系一致。

| 指标 | 4-wave | native 8-wave | AITER OPUS |
|---|---:|---:|---:|
| Q tile / KV tile | 128 / 32 | 256 / 32 | 256 / 64 |
| 物理WG / 逻辑任务 | 1280 / 1280 | 256 / 640 | 640 / 640 |
| 物理wave启动数 | 5120 | 2048 | 5120 |
| BF16 MFMA | K16 | K16 | K16 |
| `SQ_INSTS_MFMA` | 8.294M | 8.397M | 8.397M |
| `SQ_INSTS_VALU` | 45.107M | 89.095M | 40.092M |
| `SQ_INSTS_VMEM_RD` | 4.869M | 1.395M | 1.152M |
| `SQ_INSTS_LDS` | 6.420M | 9.675M | 11.817M |
| LDS bank conflict | 0.492M / 1.60% | 14.418M / 33.50% | 0 / 0% |
| L2 miss | 1.180M | 1.139M | 1.124M |
| `MfmaUtil` | 53.64% | 31.90%-32.56% | 56.4%-56.6% |
| `MeanOccupancyPerActiveCU` | 1.72 | 2.00 | 1.98 |
| ISA VGPR / SGPR / LDS | 212 / 91 / 24960 B | 237 / 100 / 49940 B | 250 / 70 / 149760 B |
| 当前延迟 | 304.07 us | 350.43 us | 282.27 us |

设置`AITER_DISABLE_FMHA_OPUS=1`后，同一输入会落到AITER
`fmha_fwd_hd192_hd128_bf16_group` ASM，独立100-event中位为296.15 us。当前4-wave
以本轮数值作参考，4-wave相对该历史独立值慢1.18%，native 8-wave约为其1.37x。差距
不是Python dispatch，
剩余重点是KV64 softmax频率、barrier锁步和跨tile流水。

结论按影响排序：

1. **4-wave gfx950 BF16 K16已完成。** 动态MFMA从16.589M降到8.294M；输出与
  AITER保持`2e-2`容差内一致。K预取现使用buffer-to-LDS direct DMA，目标ISA为
  212 VGPR、24960 B LDS、0 scratch，最终同轮三次为299.29/299.65/300.59 us。
2. **gfx950 BF16 D192改用HW-slot互补priority。** 保持计算、访存、barrier和调度
  fence不变的三路A/B中，统一`p2/p0`、全程`p0`和slot-aware分别为316.71、322.13和
  312.87 us；slot-aware较统一`p2/p0`快1.23%，较全程`p0`快2.96%。wave slot 0的
  MFMA/softmax使用`p3/p1`，其他驻留wave使用`p2/p0`，避免所有wave同时以相同高
  priority竞争。fresh ATT中`p3/p1` wave的高/低阶段累计周期比中位数为2.04，统一
  `p2/p0`基线为2.54；资源仍为212 VGPR、24960 B LDS、0 scratch。
3. **BN32不会使概率EXP翻倍。** 每个wave在4-wave的一个BN32 tile持有16个score，
  AITER的一个BN64 tile持有32个score；按64 keys归一化，两者都是32个概率EXP。
  4-wave每个BN32还无条件执行1个在线max校正EXP，因此每64 keys比AITER约多2个
  TRANS。目标shape的PMC为4-wave 7.060M、AITER 6.733M，只差4.9%，EXP不是主要
  性能差距。每wave可以精确写成4-wave `81*16 + 81 + 2 = 1379`，AITER
  `41*32 + 3 = 1315`；其中前一项是概率EXP，其余为校正和epilogue TRANS。
  4-wave的逐lane条件被LLVM if-convert为无条件`v_exp + v_cndmask`；AITER先用
  wave ballot形成uniform分支，只在max真正推进时执行校正EXP。BN32真正重复的是
  row-max、在线max/sum更新、条件判断和布局控制。
4. **row-max和lazy-max的冗余VALU已压缩。** 16项row-max由LLVM通用归约改为5条
  `v_max3`初级归约、2条`v_max3`二级归约和1条`v_max`，cross-lane交换直接使用
  gfx950 `v_permlane32_swap`。二者组合A/B提速1.51%，并将VGPR从237降到230。
  在线状态改为AITER式阈值8的无偏lazy-max，直接选择`row_max/running_max`，再从
  选中的max计算correction；同时复用`max_advances`控制O重标定。每个展开块因此少
  1条ADD、1条compare和1条`cndmask`，correction改写单项提速0.31%-0.42%。
5. **page64 lookahead复用已删除重复页表工作。** pair内第一块的
  `table[(block+3)/2]`恒等于循环状态中的`prefetch_k_page_id`，只在第二块读取下一页。
  静态页表`buffer_load_dword`站点从5降到3，VMEM read从5.079M降到4.869M，INT32
  降到1.649M；同进程A/B提速0.82%-0.94%。
6. **8-wave V跨wave LDS复用已验证但未保留为目标默认路径。** D192/V128 BF16由
  512线程协作加载V到三槽LDS ring，VMEM read从4.319M降到1.395M，已接近AITER的
  1.152M；代价是LDS指令从5.881M增至9.675M、VALU增至89.095M。独立同进程A/B中，
  K-only为397.71 us，K+V LDS为410.95 us（回退3.33%）。其他head维度继续走原
  global-to-register V路径。
  当前native 8-wave为每个错相4-wave组使用独立两槽K LDS + direct-V，P@V位于
  MFMA stage。K通过buffer-to-LDS direct DMA写入pair-padded LDS，目标资源为
  233 VGPR / 49940 B LDS / 0 scratch。目标epilogue以direct
  store替代C-shuffle，静态barrier为31；公开API始终运行该512线程实现。
7. **K预取已改为direct LDS且保持低冲突。** 4-wave与8-wave均使用
  `raw_ptr_buffer_load_lds`，每条64-lane DMA搬两个32-row D-group，并在每个pair后留
  16 B padding；ISA中K侧`ds_write_b128`为0。4-wave在KV32和KV2583上的
  `SQ_LDS_BANK_CONFLICT`均固定为49152，说明新增80个K block带来0个K-LDS冲突，
  残余来自固定C-shuffle；8-wave KV2583整体为0冲突。8-wave必须为两个错相4-wave组
  分配独立双槽ring，否则KV96开始出现非确定性覆盖。
8. **AITER流水仍更完整。** 它保留两份score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`跨KV64 tile重叠，并将8个wave拆成两组错开一个stage。8-wave
  当前在每个BN32的QK、softmax和P@V阶段间用全workgroup barrier锁步；结果是相同
  MFMA数量下利用率只有约32%，AITER约56.5%。persistent ticket不是主要差异：
  8-wave虽只启动256个物理WG并循环处理640个逻辑任务，GPU利用率仍为100%，
  `MeanOccupancyPerActiveCU`也与AITER同为约2。
  原生8-wave的BN32双score一barrier原型已做到目标ragged shape逐bit一致，但三轮
  为547.20/546.10/549.66 us，对照native 390.53/389.99/391.89 us，回退约40%。
  它没有KV64带来的softmax频率减半，因此不能验证或替代真正的KV64设计。

#### 4-wave与AITER的VALU分类

同一`Q10240,KV2583,non-causal`单dispatch，二者均启动5120个wave。按每wave和每64
keys归一化后：

| PMC类别 | 4-wave / wave | AITER / wave | 4-wave / 64 keys | AITER / 64 keys | 差额 / 64 keys |
|---|---:|---:|---:|---:|---:|
| 全部VALU | 8810.0 | 7830.5 | 217.53 | 190.99 | +26.54 |
| TRANS F32 | 1379.0 | 1315.0 | 34.05 | 32.07 | +1.98 |
| FMA F32 | 1464.0 | 1352.0 | 36.15 | 32.98 | +3.17 |
| ADD F32 | 1378.0 | 1353.0 | 34.02 | 33.00 | +1.02 |
| MUL F32 | 68.0 | 84.0 | 1.68 | 2.05 | -0.37 |
| INT32 | 322.0 | 139.0 | 7.95 | 3.39 | +4.56 |
| INT64 | 41.0 | 4.0 | 1.01 | 0.10 | +0.91 |
| CVT | 682.0 | 700.0 | 16.84 | 17.07 | -0.23 |
| SALU（不计入VALU） | 548.0 | 948.5 | 13.53 | 23.13 | -9.60 |

差额来源按影响排序：

1. **gfx950 BF16转换已对齐native指令。** `_cvt_f32_to_bf16`在函数内部检查平台：
  gfx950使用vector cast并发射`v_cvt_pk_bf16_f32`，gfx942继续使用原有round-half-up
  的`+0x8000`、右移和`v_perm`。相对手写路径，gfx950每wave少1360条INT32、增加
  680条CVT，总VALU从59.407M降到52.444M（-11.7%），VGPR从237降到226。
  当前4-wave的CVT已与AITER基本相同；最终仍有322对139条INT32/wave，来自地址与控制。
2. **核心循环的`v_pk_add_f32`已消除。** 生成源有两处：BF16的vector
  `scaled_score - updated_max`被后端配成8条packed add/BN32；FP8的
  `llvm.vector.reduce.fadd.v16f32`被配成7条packed add/展开块。BF16现在直接使用
  16条`v_fma_f32(score, scale, -updated_max)`融合scale-sub；FP8使用逐元素
  `v_sub_f32`和15条标量add归约树。最终BF16/FP8静态ISA的`v_pk_add_f32`均为0。
  BF16同进程旧/新A-B为343.97/313.27 us（提速9.8%），总VALU从52.444M降到
  49.126M，ADD差额从+19.02降到+3.02条/64 keys；最大输出差0.000488。
3. **D192 K写已与EXP交织。** 原稳态每轮三条`ds_write_b128`连续发射，最后一条后
  仅约8条BF16 CVT便进入`lgkmcnt(0)+barrier`。单纯把K写整体移到softmax前会因提前
  等VMEM回退3.3%-3.8%；仅靠跨basic-block scheduler hint也不生效。最终方案延后
  output rescale，使17条EXP和3条K写进入同一调度区，并按
  `4 EXP -> write -> 4 EXP -> write -> 4 EXP -> write -> 5 EXP`交织。ATT确认稳态
  三条write两侧均有EXP，最后write到barrier约58条记录；同进程A/B提速0.92%，
  输出逐bit一致。
4. **显式max树、无偏lazy-max和谓词复用继续消除重复工作。** 最终总VALU进一步降到
  45.107M，ADD差额降到+1.02条/64 keys；静态ISA相对上一版每个展开块少1条ADD、
  1条compare和1条`cndmask`，仍为零`v_pk_add_f32`。
5. **BN32仍重复其他非EXP softmax工作。** 4-wave每64 keys做两次16项sum、row max、
  running max/sum更新和阈值选择；AITER对32项一次完成。sum reduction本身是
  `2*15`对`31`次add，数量近似相同；剩余差距来自row-max和在线状态更新等控制。
6. **未分类VALU主要是重复控制与布局操作。** 扣除转换净开销和细分浮点counter后，
  仍包含两次row-max后的`max/compare/cndmask`、两次在线状态选择、额外cross-lane
  交换、概率布局move/permute及地址计算。gfx950现有counter不能再将这些类别分开，
  因此不把该余量强行归到单一源码操作。AITER的SALU反而多9.60条/64 keys，说明
  它把更多统一控制留在scalar侧，而4-wave有更多逐lane vector控制。
7. **per-wave direct V主要造成VMEM差距。** 4-wave每个wave为自己的32行Q读取V，
  AITER由workgroup协作将BN64 K/V各搬入一次LDS。lookahead复用后4-wave的VMEM read为
  4.869M，AITER为1.152M（4.23倍）；这是issue/流水压力，不是上述INT32差额的主要来源。
8. **流水而非指令总量。** 即使移除上述冗余，4-wave仍是单score fragment，阶段间
  barrier锁步；AITER用双score、两组4-wave错相，将EXP/VALU隐藏在MFMA之后。
  本轮缩短非MFMA stage后4-wave `MfmaUtil`已从约46%升到53.64%，仍低于AITER约56.5%。

`-packed-fp32-ops`不是VALU差额原因。诊断A/B分别给转换前的同一BF16 specialization传入
`-packed-fp32-ops`和`+packed-fp32-ops`，MLIR属性确认相反，但最终ISA SHA256完全
一致：两者都是68个静态`v_exp`、192个静态`v_pk_*_f32`、237 VGPR、0 scratch。
也就是说当前编译器仍选择了packed add/mul；该target feature在这条ISA上没有产生
任何变化。

转换A/B中manual/native延迟为347.37/339.03 us（native快2.46%），两者对AITER最大
绝对误差均为0.0009766。首次组合测试的异步GPU fault来自前一个candidate尚未同步时
开始JIT/加载下一code object；在各候选首次launch后加入同步后，non-SWA与SWA两项
完整生产组合均通过。当前按8-wave相同方式保留平台门控。

当前优化状态与TODO：

1. [x] 4-wave gfx950 BF16 QK/P@V切换K16，并同步probability/K-row layout。
2. [x] 8-wave重做K-LDS padding/read permutation，将K-only冲突率降到3.76%。
3. [x] 8-wave V由workgroup协作加载到三槽LDS并跨8个wave复用；功能完成，当前性能
  取舍如上，后续优化不能只以冲突率作为验收指标。
4. [ ] 将8-wave内部KV tile提升到64并保留两个score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`重叠，同时把softmax频率减半；不再尝试已回退约40%的BN32
  双score版本。
5. [ ] 将8个wave拆成两组4-wave并错开一个pipeline stage；验收要求`MfmaUtil > 50%`、
  0 scratch，且不能恢复K-LDS冲突。除已保留的等价lookahead复用外，不再优先优化
  page-table或L2：剩余计数不支持该方向。

### 4-wave static/dynamic当前数据

同一输入、20次预热、100个CUDA event样本、5轮中位数；static与dynamic在每个case
按对应行注明的容差一致。`dynamic/static`为static延迟除以dynamic延迟，小于1表示dynamic较慢。

| 场景 | dtype | 4-wave static | 4-wave dynamic | dynamic/static |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | BF16 K16 direct-LDS | 312.73-314.21 us | 321.07-322.73 us | 0.974x |
| causal `Q=KV=32768` | BF16 K16 | 5748.63 us / 956.32T | 7032.47 us / 781.74T | 0.817x |
| SWA `Q16K,KV128K,window=128` | BF16 K16 | 77.94 us / 277.68T有效 | 84.44 us / 256.31T有效 | 0.923x |
| causal `Q=KV=32768` | OCP FP8 K64 | 4230.13 us / 1299.62T | 4815.89 us / 1141.54T | 0.878x |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | 61.00 us / 354.78T有效 | 66.56 us / 325.14T有效 | 0.916x |

slot-aware priority的最终同轮结果中，non-causal目标shape的dynamic比static慢
2.6%-2.7%；本轮dynamic相对static通过`rtol=atol=2e-2`。其他表项为此前同轮历史
数据，不据此外推。dynamic仍用于batch>1负载均衡，短non-causal/causal与SWA回归
均验证dynamic和static逐bit一致。

### Non-SWA FP8 K16/K64 A/B

生产shape为`Q=KV=32768,causal`，算法工作量为`5.497558 TFLOP`。相同量化输入下
K16与K64输出逐bit一致。

| Kernel | K16延迟 / TFLOPS | K64延迟 / TFLOPS | 延迟下降 |
|---|---:|---:|---:|
| 4-wave | 5237.05 us / 1049.74T | **4651.73 us / 1181.83T** | **11.18%** |
| 8-wave | 5611.21 us / 979.75T | **5101.77 us / 1077.58T** | **9.08%** |

gfx950 OCP FP8 QK使用`v_mfma_f32_32x32x64_f8f6f4`，并通过
`scale_a=scale_b=0`表达unity E8M0 scale。P@V reduction只有32，继续使用
`v_mfma_f32_32x32x16_fp8_fp8`。K64仅在Dqk可被64整除时启用。

### SWA

生产配置为`Q=16K,window_left=128,page=64,bottom-right causal`，带per-head FP32
sink。4-wave按query tile裁剪page table，窗口外page不会被读取；batch=1默认走
static，`force_dynamic_schedule=True`时走persistent。

BF16 scheduler sweep结果（K16/V-LDS前历史A/B）：

| Total KV | 4-wave static | 4-wave dynamic | 8-wave persistent | Dynamic vs 8-wave |
|---:|---:|---:|---:|---:|
| 32K | 103.95 us | 110.31 us | 141.64 us | 1.284x |
| 64K | 103.89 us | 110.34 us | 141.99 us | 1.287x |
| 128K | **103.94 us** | 110.26 us | 142.15 us | 1.289x |

128K历史dtype与K64 A/B结果（均早于本轮softmax/lookahead优化，仅用于记录K64收益）：

| dtype | 4-wave static | 8-wave persistent | 4-wave vs 8-wave |
|---|---:|---:|---:|
| BF16 | **103.15 / 104.01 us** / 209.82 / 208.08T有效 | 141.04 / 142.53 us / 153.45 / 151.85T有效 | 1.367x / 1.370x |
| OCP FP8 K16 | **71.64 us / 302.10T有效 / 599.52T执行** | 112.90 us / 191.70T有效 / 570.63T执行 | 1.576x |
| OCP FP8 K64 | **67.05 us / 322.78T有效 / 640.56T执行** | 112.35 us / 192.64T有效 / 573.43T执行 | **1.676x** |

SWA AITER有两种计时口径：

1. **含gather端到端**：从5D cache gather到linear K/V，再调用AITER；
2. **attention-only**：K/V已是linear布局，只计AITER attention kernel。

同进程、同输入、20次预热、100样本、5轮中位数；三行BF16均来自本轮最终实现：

| KV | gather + `mha_batch_prefill_func` | `mha_batch_prefill_func` only | gather + `flash_attn_varlen_func` | `flash_attn_varlen_func` only | 4-wave static | 8-wave persistent |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 281.37 us / 76.92T | 259.55 us / 83.39T | 146.72 us / 147.51T | 116.86 us / 185.20T | **78.32 us / 276.34T** | 143.45 us / 150.87T |
| 64K | 323.29 us / 66.94T | 260.41 us / 83.11T | 180.15 us / 120.14T | 118.20 us / 183.10T | **78.10 us / 277.11T** | 143.53 us / 150.79T |
| 128K | fault，未计时 | fault，未计时 | 269.03 us / 80.45T | 123.39 us / 175.40T | **77.94 us / 277.68T** | 142.20 us / 152.20T |

两种线性AITER API都正确支持SWA：`mha_batch_prefill_func`实际命中
`aiter::mha_batch_prefill`，`flash_attn_varlen_func`命中`aiter::mha_varlen_fwd`；
两者小shape最大绝对差为`0.0078125`。varlen更快且覆盖128K，因此是当前推荐的
attention-only AITER基线。真正从5D page64 cache直连时，只有
`mha_batch_prefill_func`具备paged ABI，但当前gfx950 BF16 D192/V128构建没有匹配
specialization；`flash_attn_varlen_func`是linear THD接口，不是direct-paged入口。

### 当前资源

gfx950 OCP FP8 D192当前可分发的K64 specialization：

| Kernel | 调度 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave K64 | static | 174 | 68 | 16384 B | 0 B | 0 | 0 |
| 4-wave K64 | dynamic | 178 | 93 | 16388 B | 0 B | 0 | 0 |
| 8-wave K64 | persistent | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64相对K16把QK静态MFMA站点减少4倍；P@V仍使用K16。

gfx950 BF16 D192/V128当前目标specialization：

| Kernel | 调度 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave direct K-LDS | static | 212 | 91 | 24960 B | 0 B | 0 | 0 |
| 4-wave direct K-LDS | dynamic | 218 | 93 | 24964 B | 0 B | 0 | 0 |
| 8-wave direct K-LDS + direct-V/direct-store | persistent | 233 | 100 | 49940 B | 0 B | 2 SGPR / 0 VGPR | 0 |

### 当前验证

| 范围 | 结果 |
|---|---|
| 4-wave合并测试文件 | `51 passed, 2 skipped`；skip为两项可选生产性能测试 |
| 4-wave dynamic生产输出 | non-SWA与SWA两项opt-in测试各`1 passed`，均包含static/dynamic逐bit检查 |
| 8-wave gfx950完整矩阵 | `67 passed, 1 skipped` |
| AITER non-SWA路由 + SWA双入口 + 三方正确性 | `3 passed` |
| non-SWA K16/K64 | 各kernel内逐bit一致 |
| SWA K16/K64 | 各kernel内逐bit一致；4/8-wave relative-L2 `6.3745e-5` |
| focused FP8 ISA | 0 private、0 spill、0 scratch |
| persistent counter复用 | 默认stream连续16次、两个非默认stream各8次均通过 |

## 当前实现与已完成优化

- 4-wave为`BM128 x BN32 x 256 threads`；batch=1走static grid，batch>1走
  atomic-ticket persistent grid；
- 8-wave为`BM256 x BN32 x 512 threads`，使用1 WG/CU persistent调度；
- 4-wave使用独立`k_lds0/k_lds1`字段构成pair-padded direct-DMA K LDS ping-pong，
  V保持global-to-register。稳态先以`vmcnt(8)`等待上一轮K DMA，同时保留当前V请求；
  gfx950 BF16 D192按HW wave slot为MFMA/softmax阶段选择`p3/p1`或`p2/p0`；进入MFMA
  阶段后以两条MFMA起步，再将K `ds_read_b128`与P@V MFMA逐条交织，读取完成后才向
  已消费槽发起下一次K DMA。prologue和epilogue的`vmcnt(0)`分别保护首次读取和
  K/output union复用；稳态剩余的机器级`vmcnt(0)`只保护最终V fragment消费；
- 8-wave native D192/V128 BF16为两个错相4-wave组分别使用direct-DMA双槽K LDS ring，
  并使用direct-V和direct output store；K的每两个D-group后padding 16 B，再由K16
  置换视图读取。P@V位于MFMA stage；公开8-wave API始终运行该原生路径；
- online softmax使用显式8指令max树、direct permlane32、阈值8的无偏lazy rebase和
  loop-carried max/sum，并复用推进谓词控制O重标定；
- page64的pair首块复用已携带的lookahead page id，只在pair第二块读取下一页；
- 32x32 MFMA减少row max/sum cross-lane开销，`v_permlane`避免经LDS交换；
- FP8采用gfx950 K64 QK、K16 P@V、score MUL split11和FP8-only fast-math；
- BF16/FP8共享pipeline骨架，dtype专属K搬运、scheduler、probability、V布局和epilogue
  地址封装在独立helper中。

Dynamic scheduler工作已全部完成：

- 消除persistent BF16 D192 spill，在消费点重物化C-shuffle地址和SWA mask坐标；
- 按device/stream/grid复用`work_counter`，最后退出的workgroup在device端复位；
- ticket改为4-byte LDS mailbox广播，每个ticket由双barrier缩减为单barrier；
- ticket barrier同时封闭前一work item的C-shuffle生命周期，删除重复re-entry barrier；
- 4-wave的1/2/3/4 WG-per-CU sweep选择2 WG/CU；8-wave选择1 WG/CU；
- 最终trace中static/dynamic kernel为97.34/105.52 us，稳态dynamic没有额外初始化dispatch。

版本：4-wave `7ed2215b0027881af91ad5e12f253288254396db8a21265c5d3bfd65e8d5d375`；
8-wave `160b38fcc74696fd05115d6ebfbe757b34592f53fde56c713b2e64b201bada7b`。

## 复现当前结果

```bash
cd /root/workspace/luocheng/pyhip
export HIP_VISIBLE_DEVICES=7
export FLYDSL_RUNTIME_ENABLE_CACHE=0

PA_CASE=tails PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=batch PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=noncausal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=causal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_DTYPE=bf16 PA_CASE=bf16_ref_short PA_NUM_ITERS=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py

python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_matches_dispatched_aiter \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_paths
PYHIP_RUN_PA_AITER_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_aiter_production_performance -s
PYHIP_RUN_PA_AITER_SWA_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_production_performance -s
```

测试前用`rocm-smi --showuse`选择空闲GPU。完整causal reference需要约64GB临时显存；
定频诊断流程见`tests/flydsl/H3_ATTENTION_THROTTLE_PROFILE.md`。

## 历史结果（gfx942）

以下数据来自MI308X/gfx942及2026-08版本，仅用于记录优化演进，不代表当前gfx950性能。

### 2026-08-14 BF16 D192 spill/occupancy验收

目标只有两项：

1. 为8-wave补齐BF16 `Dq=192,Dv=128`，修复其tail/causal mask和dtype编译缓存，
  并增加D128/D192 BF16回归；FP8路径保持兼容。
2. 修复4-wave batch>1 persistent BF16 D192的低occupancy分配。原
   `vgpr_count=265`实际为`256 VGPR + 9 AGPR`，并非普通VGPR越过硬件上限；
  将`rocdl.waves_per_eu=2`直接附着到persistent GPU kernel后，最终LLVM IR带有
  `amdgpu-waves-per-eu="2"`。page32/128变为`256 VGPR + 0 AGPR / 10 spill / 44B private`，
  page64 non-causal为`22 spill / 92B private`，causal为`16 spill / 68B private`；
  三种page size均为`256 VGPR + 0 AGPR`、2 waves/SIMD。

page64 causal的16个spill均为长寿命地址值，并非Q/K/V fragment或O accumulator：16个
store仅在workgroup入口执行一次；每个persistent work item分别有3个初始化reload、2个
masked-tail入口reload和11个C-shuffle epilogue reload。重复执行的KV fast/tail循环体内
没有scratch指令。

验收使用MI308X/gfx942、1300MHz、`Hq=16,Hkv=1,Q=KV=32768,causal,page_size=32`，
每组10次预热、两轮各50个event样本，顺序为`4w -> 8w -> 8w -> 4w`。

| shape | batch | 4-wave | 8-wave | 关键资源 |
|---|---:|---:|---:|---|
| BF16 D128 | 1 | 23.983 ms / 183.381T | 30.054 ms / 146.340T | 8-wave：8 spill / 36B private |
| BF16 D128 | 4 | 97.725 ms / 180.017T | 116.349 ms / 151.202T | 4-wave：228 VGPR / 0 spill |
| BF16 D192 | 1 | 34.468 ms / 159.497T | 37.770 ms / 145.555T | 8-wave：23 spill / 96B private |
| BF16 D192 | 4 | 137.813 ms / 159.566T | 146.258 ms / 150.352T | 4-wave：10 spill / 44B private |

4-wave D192 batch=4修复前为`113.419T`，修复后提升`40.69%`。四组4/8-wave输出
均finite；relative-L2为D128 batch=1/4 `3.7173e-5/3.8095e-5`，D192 batch=1/4
`3.8662e-5/3.8175e-5`。8-wave BF16 D192短尾、causal及FP8聚焦用例通过。

回归：8-wave `39 passed`、4-wave `15 passed`、公开API `11 passed`。

### 2026-08-11 4-wave/8-wave基线

当时的8-wave参考使用per-tensor Q量化；4-wave接收同一份FP8 Q和
等值descale。两者在同一进程使用10套buffer、各10次预热和50个CUDA event样本，采用
位置平衡顺序；“相对8-wave”为25组配对时间比中位数的倒数。

| 场景 | 时钟 | 实现 | 调度 | 中位延迟 | Actual TFLOPS | 相对8-wave |
|---|---|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | **4-wave** | static | **671.343 us** | **403.441** | **1.224x** |
| 同上 | auto | 8-wave | persistent | 821.883 us | 329.544 | 1.000x |
| causal `Q=KV=32768` | auto | **4-wave** | static | **17918.507 us** | **306.809** | **1.059x** |
| 同上 | auto | 8-wave | persistent | 18872.409 us | 291.301 | 1.000x |
| causal `Q=KV=32768` | 1300MHz | **4-wave** | static | **18836.170 us** | **291.862** | **1.061x** |
| 同上 | 1300MHz | 8-wave | persistent | 19992.191 us | 274.985 | 1.000x |

non-causal 25/25组获胜；causal auto 24/25组获胜；causal 1300MHz 25/25组获胜。
causal按三角有效FLOPs计数；auto-DPM存在双态，因此同时保留1300MHz结果。

下表沿用page32性能基线；page64/page128已通过功能精度回归，尚未单独建立性能基线。

主shape的4-wave static/persistent同代码对照：

| 场景 | 时钟 | static | persistent | static收益 |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | 670.102 us / 404.188T | 839.263 us / 322.720T | **25.18%** |
| causal `Q=KV=32768` | 1300MHz | 18836.068 us / 291.863T | 19555.952 us / 281.119T | **3.85%** |

因此non-causal的400T主要依赖batch=1 static调度；persistent路径仍约323T。causal也受益于
static，但收益明显较小。两组static/persistent输出逐元素一致。8-wave始终使用persistent。

### 2026-08-10 4-wave性能矩阵

除H3使用3次预热和10样本外，其余formal结果均为10套buffer、10次预热和50样本中位数。
causal括号内为当次快档min。

| dtype | Dq/Dv | 调度 | 场景 | shape | 中位延迟 | Actual TFLOPS |
|---|---:|---|---|---|---:|---:|
| FP8 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 672.883 us | 402.518 |
| FP8 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 3065.972 us | 350.213 |
| FP8 | 192/128 | static | causal | `H16,Q=KV=32768` | 17802.427 us (13973.054) | 308.809 |
| BF16 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 1323.445 us | 204.653 |
| BF16 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 7475.309 us | 143.638 |
| BF16 | 192/128 | static | causal | `H16,Q=KV=32768` | 35486.656 us (25454.100) | 154.919 |
| FP8 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 2500.491 us | 343.530 |
| FP8 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 208.721 us | 257.219 |
| FP8 | 128/128 | static | causal | `H1,Q=KV=32768` | 1137.484 us | 241.654 |
| BF16 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 3422.933 us | 250.952 |
| BF16 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 268.201 us | 200.175 |
| BF16 | 128/128 | static | causal | `H1,Q=KV=32768` | 1650.006 us | 166.592 |
| FP8 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 86.369 ms | 331.755 |
| BF16 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 179.958 ms | 159.223 |

### 2026-08-10 精度矩阵

`diff`为`pyhip.calc_diff`对PyTorch reference；全部通过`rtol=atol=0.1`和finite检查。

| dtype | Dq/Dv | ragged最大diff | batch=4 diff | small causal diff | 主shape/额外验证 |
|---|---:|---:|---:|---:|---|
| FP8 | 192/128 | `2.8836e-4` | `3.4356e-4` | `1.7518e-4` | 主non-causal `3.6652e-4` |
| BF16 | 192/128 | `2.5129e-6` | `2.7224e-6` | `1.9344e-6` | 主non-causal `2.8093e-6` |
| FP8 | 128/128 | `2.6076e-4` | `3.4029e-4` | `1.7112e-4` | H3 finite |
| BF16 | 128/128 | `2.4619e-6` | `2.7061e-6` | `1.8679e-6` | H3 finite |

ragged覆盖`KV=3/13/23/53/83`，small causal为`Q=KV=256`。4-wave/8-wave同输入的
non-causal与causal relative-L2分别为`1.17e-4`和`1.12e-4`。

### 2026-08-10 specialization资源

- block为`BM128 x BN32 x 256 threads`；每个workgroup 4个wave；
- K使用LDS ping-pong，V直接进入fragment，output使用两个半块C-shuffle；
- online softmax使用raw-max、lazy rebase和loop-carried max/sum；
- FP8使用QK `VMEM1 -> MFMA2`、score MUL split11和FP8-only fast-math；
- FP8 D192当时为168 combined VGPR、16KB LDS、0 scratch，自然达到3 waves/SIMD；
- BF16 D128使用专用scheduler/HW-slot priority，D192使用独立scheduler；
- BF16/FP8共享pipeline时序骨架，dtype专属K搬运、scheduler、probability写回、V布局、
  epilogue地址和compile hint均封装在独立helper。

refactor前后fresh执行ISA逐条一致：

| specialization | ISA资源 | MFMA |
|---|---|---:|
| FP8 D192 | 168 VGPR-form / 16KB / 0 scratch | 80 |
| FP8 D128 | 153 VGPR-form / 16KB / 0 scratch | 64 |
| BF16 D192 | 250 VGPR-form / 25KB / 0 scratch | 160 |
| BF16 D128 | 214 VGPR-form / 17KB / 0 scratch | 128 |

FP8 D192 dynamic persistent kernel的执行ISA同样逐条一致。

## 优化里程碑（2026-08）

以下记录保留改变实现或建立关键反证的里程碑，性能数字均为当时版本结果。

### 2026-08-10：4-wave pipeline与C-shuffle

- **改动**：建立MMA32骨架；K走LDS ping-pong，V直读；接入paged ABI、GQA、ragged和causal；
  output改为两个64x128半块C-shuffle。
- **验证**：反转page table、跨页和ragged尺寸通过。
- **结果**：约`1838 -> 1465 -> 1008 -> 915 us`；保留双缓冲和半块C-shuffle。

### 2026-08-10：static dispatch与causal均衡

- **改动**：batch=1改用static grid；batch>1保留persistent；causal使用
  `(251 * tile + 251) % 256`映射。
- **验证**：non-causal、batch=4和long causal通过。
- **结果**：short调用约`54 -> 10 us`；causal约`17.9 -> 16.7 ms`；保留static/仿射路径。

### 2026-08-10：双K流水与priority

- **改动**：形成`K(i+2)`预取、softmax、`K(i+1)`写入、PV/barrier/K-read跨回边流水；
  FP8统一stage priority为`0/2`。
- **验证**：双K统一priority反相稳定；HW-slot priority回退。
- **结果**：主路径约876--880us；保留双K与统一priority。

### 2026-08-10：raw-max与softmax调度

- **改动**：先对raw score做max/shuffle，再用score scaling覆盖等待；FP8增加固定切分。
- **验证**：FP8/BF16数值不变；shuffle wait由约55降至10.7/17.2 cycles。
- **结果**：raw-max约提升3%，split8再提升1.33%；BF16不采用split8。

### 2026-08-10：BF16与H3

- **改动**：加入D128/D192 BF16 MMA、K/V layout、128-bit copy、LDS padding和D128 scheduler。
- **验证**：BF16 ragged/batch/causal及真实H3通过。
- **结果**：当时BF16 D128为250.952T，D192为204.653T，H3为159.223T。

### 2026-08-10：FP8自然3-wave

- **改动**：epilogue重建C-shuffle地址，将资源从176降至168 combined VGPR；固定gap2、
  score MUL split11和FP8-only fast-math。
- **验证**：16KB LDS、0 scratch、80 MFMA；最终ATT三槽`2+1`混合相90.36%。
- **结果**：当时non-causal为402.518T；4-wave相对当时8-wave为1.222x。

### 2026-08-11：8-wave参考复测

- **改动**：使用当时的8-wave per-tensor Q和多page-size实现，共享输入位置平衡复测；
  causal额外使用1300MHz固定频率。
- **验证**：page32/64/128 short reference通过；定频结束后恢复auto。
- **结果**：non-causal为1.224x；causal auto/1300MHz分别为1.059x/1.061x。

### 2026-08-11：代码refactor

- **改动**：保留共享pipeline，将BF16/FP8 K搬运、scheduler、probability、V布局、epilogue和
  compile hint封装为helper；删除恒真/恒假参数并统一命名。
- **验证**：4种static specialization和FP8 dynamic persistent执行ISA逐条一致；完整精度矩阵通过。
- **结果**：性能矩阵与重构前一致；共享时序骨架、独立dtype细节，不复制两份pipeline。

## 已否决实验

| 类别 | 关键证据 | 保留方案 |
|---|---|---|
| K copy 128-bit | 隐式`vmcnt(0)`，回退15.4%--15.5% | 64-bit K copy |
| 阶段/HW-slot | 循环slot回退2.76%；入口复制pipeline回退16.71% | 单pipeline、FP8统一priority |
| barrier/PV/K写 | 隔页barrier等待约124增至403 cycles；PV切分增scratch | 每页barrier、完整PV |
| page-table pair load | VGPR 168升至172，失去3-wave | 标量lookahead |
| 映射/priority | tile-major回退0.5%；反向priority约回退3% | head-major、`0/2` |
| 数值调度 | sum多链/非均匀gap无收益；显式rcp仅397.112T | 单链sum、gap2、fast-math |
| BF16实验 | split8中性；D192 shape峰值约210.7T | 原BF16 softmax、独立D192优化 |
| V自然padding | 冲突率降到0.37%，但`ds_read_b128`串行化使延迟升到749.39 us | 128-bit swizzled V LDS |
| V预分区/专属hint | 256 VGPR且411.86 us；精确DS/MFMA分组为413.57 us | 消费点重物化V地址、原调度hint |
| BF16平衡sum树 | ADD仍为15条且VGPR不变，但目标case稳定回退0.65%-0.73% | 保留LLVM线性归约及现有EXP/DS-write交织 |
| page id `readfirstlane` | 少10条整数VALU、VGPR 230降到226，但VMEM地址依赖串行化并回退10.9%-11.2% | 仅复用等价lookahead，不强制页号进SGPR |
| 完整V预分区 | 输出逐bit一致，但静态地址指令净增1条 | 循环内按page/block分区 |
| 单结果permlane32 | 未初始化old-dst导致NaN；该指令需要双结果/旧目标语义 | 保留ROCDL双结果intrinsic |

## ATT证据

- 当前gfx950 BF16 K16、slot-aware priority、native CVT、显式max树、page64
  lookahead复用、零`v_pk_add_f32`：
  `tests/flydsl/pa_4wave/ui_output_agent_54855_dispatch_22`；源码快照与当前文件SHA256
  一致，trace含20个完整wave并同时覆盖`p3/p1`和`p2/p0`路径；
- 统一`p2/p0`基线：`tests/flydsl/pa_4wave/ui_output_agent_44323_dispatch_22`。其
  高/低阶段累计周期比中位数为2.54；slot-aware的`p3/p1` wave降至2.04。两份trace
  均保持MFMA起步、K DS-read/MFMA交织及读取完成后发射K DMA；
- FP8 D192：`tests/flydsl/pa_4wave/att_fp8_d192_3wave/ui_output_agent_28524_dispatch_66`；
- BF16 D192：`tests/flydsl/pa_4wave/att_bf16_d192/ui_output_agent_32152_dispatch_13`；
- FP8主要stall/MFMA：MFMA 36.674、VALU 12.619、barrier 7.413、VMEM-load 6.397、
  LDS-wait 5.900；两条barrier约128/145 cycles。

</details>
