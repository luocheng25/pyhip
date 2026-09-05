# Paged Prefill 8-wave

## 当前：gfx950 direct-paged attention（2026-09-05）

[pa_8wave_950.py](pa_8wave_950.py) 已整体重写；本节是当前实现与验证结果。
下方旧版记录对应 [pa_prefill_8w32x32.py](pa_prefill_8w32x32.py) 和早期实验，
**不适用于新实现**。原始 OPUS/AITER 内核未修改，执行路径不调用 AITER attention。
**仅保留 direct-paged 路径：一次调用、一个 attention kernel、零 KV workspace。**
`prepare_kv`、`attend_linear`、gather kernel 和线性 KV 缓冲均已删除。
最新同输入对照见 [4-wave/8-wave 主报告](../pa_4wave/README.md)。

### 最新：SWA+sink 分析与窄窗口优化（2026-09-05）

任务规模扫描和PMC/ATT表明，差距不是单纯任务数不足：BM256窗口并集使8-wave的MFMA
比4-wave多50%，短流水仍有53个barrier/wave（4-wave为13）。因此仅对实测有收益的
大grid窄窗口，跳过wave完全不可见的QK/PV；保留所有内存读取、NaN-tail保护与barrier。
小grid、宽窗口和D128/W128不启用，避免已测出的回退。

最终 **272 passed / 6 skipped**，4-wave **51 passed / 2 skipped**。D192/W128/KV32K–128K
static改善 **1.43%～2.17%**，persistent改善 **0.70%～1.11%**；KV128K最终为
**114.888 / 112.884 µs**（static/persistent），4static **95.082 µs**，仍未追平。
MFMA实测减半，但memory/softmax/同步成本仍在；full路径指令类别数量和资源不变。
完整根因、门限、否决实验、所有采样与ATT caveat见 [swa_analysis.md](swa_analysis.md)。

### 可选 persistent（2026-09-05，SWA优化前验收）

`PagedAttention(..., persistent=True)` 启用设备端任务队列，默认 `False` 不变。
每 device/stream/grid 首次分配8-byte header，最后一个CTA自动重置；预热后仍一次调用、
一个 attention kernel、零KV workspace。支持ragged、空请求、causal配对、SWA/sink、LSE、
独立stream捕获图的并发replay。共享同一个capture/header的图必须串行。

最终回归 **236 passed / 6 skipped**；默认static六种汇编逐条不变。
H3：static **34648.100 µs** → persistent **31564.267 µs**，4dynamic **31562.353 µs**；
SWA D192：**116.698→113.855 µs**，full路径基本持平。D192 persistent NC/SWA存在少量spill，
不宣称通用加速。完整接口、并发契约、资源、PMC及采样见 [persistent.md](persistent.md)。

### tile API 重构（2026-09-05）

Q/K/P、score/output 改用 tiled-MMA fragment API，QK/PV 改用 `fx.gemm`，
V 使用 layout-only `fx.select`；保留原八阶段流水与显式 LDS/wait 边界。
六种 specialization 的**汇编指令及操作数逐条一致**，资源不变、无 spill。
最终回归 **181 passed / 6 skipped**；反序复测未观察到可重复的性能下降。
完整逐指令证据、两轮原始性能采样和复现入口见 [tile_refactor.md](tile_refactor.md)。
这一阶段仍为 static 调度；下表是重构前的完整矩阵，不覆盖新测 A/B 样本。

### README 全矩阵重测（tile 重构前，2026-09-05）

按照 [4-wave/8-wave 主报告](../pa_4wave/README.md) 完成 **44 组配置 / 127 个候选结果**，
包括 BF16/FP8、SWA32K/64K/128K、page32/page64、batch4、单 head 与 H3；当前 8-wave
不支持的 FP8/page32 明确 N/A，不使用旧8-wave代替。全部计时候选先通过独立完整 FP32
reference 和三次重复一致性检查；五轮交替、每轮20warmup/100迭代的 GPU-profiler 中位数。
本轮没有修改 attention 内核。

主 BF16 page64 本轮结果，单位 **µs**：

| 场景 | Dqk | 当前8-wave | 4-wave static | 4-wave dynamic | 显式 OPUS linear |
|---|---:|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 257.863 | 269.439 | 274.641 | 241.810 |
| noncausal Q10240/KV2583 | 192 | 296.015 | 318.116 | 325.063 | 292.013 |
| causal Q=KV32768 | 128 | 4499.220 | 4929.746 | 5509.998 | 4713.127 |
| causal Q=KV32768 | 192 | 5347.772 | 6461.789 | 7142.503 | 5298.313 |
| SWA+sink Q16K/KV128K/W128 | 128 | 101.281 | 82.288 | 89.314 | N/A |
| SWA+sink Q16K/KV128K/W128 | 192 | 115.637 | 96.405 | 102.715 | N/A |

D192 noncausal 距 OPUS linear **+1.37%**、causal32K **+0.93%**；SWA 仍慢于 4-wave。
H3 page64 BF16 的当前8-wave **34665.584 µs**，4-wave dynamic **31583.970 µs**，
8-wave 慢 **9.76%**。OPUS 使用等价预建 linear KV、转换不计时，不是 5D端到端性能。

当前8-wave完整回归重跑 **181 passed / 6 skipped**；4-wave **51 passed / 2 skipped**。
另行执行的旧 non-SWA opt-in 在 AITER linear/page1 causal32K 出现 **GPU fault / exit134**，
已记录失败；旧 SWA opt-in 独立运行 **1 passed**。二者不混入当前内核的统一结果。
新 ATT 共72完整wave：BFI0、waitcnt179、barrier333、MFMA1640/wave，稳态phase **2783.72 cycles**。
全部采样、支持/失败原因、30组资源和新 PMC 见
[readme_retest_results.json](../pa_4wave/readme_retest_results.json)。

### packed-V / 控制等待合入记录（下表为上一轮 A/B，不是本轮重测）

先合入 packed-i32 V 拼接并完成 **173 passed、6 skipped**，再合入控制/等待精简；
最终完整回归 **181 passed、6 skipped**。新增 8 项 runtime 长度与 NaN-tail 回归，
覆盖 D128/D192、正向/首尾配对反向、LSE/无 LSE，同一编译结果反复改变真实尾页。

- `_read_v` 先拼接 packed words，再一次性 BF16 store，保留 V 尾页清零。
- phase S2 将 VM/page/LDS 等待合为一条，**等待阈值不变**。
- noncausal 用剩余 KV 长度判断边界；正向 main phase 消费的 V 必在最后 tile 前，
  因此编译期移除不可能命中的 tail 检查，**reverse 和 epilogue 的 NaN 保护不变**。
- 仍然 direct-only、一个 kernel、零 workspace，所有 stage barrier 保留。
  **没有新增 persistent 调度**；首尾配对仍不是工作队列。

相同 5D 输入、同进程三个 FlyDSL 版本与显式 OPUS 对照，5 轮交替、每轮 20 warmup /
100 次采样、中位 GPU 时间，单位 **µs**：

| 场景 | Dqk | 改动前 | 仅 packed-V | packed-V + 控制/等待（当轮） | OPUS linear core |
|---|---:|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 266.395 | 259.345 | **258.495** | 241.272 |
| noncausal Q10240/KV2583 | 192 | 303.689 | 299.090 | **296.688** | 291.950 |
| causal Q=KV32768 | 128 | 4578.457 | 4471.397 | **4493.740** | 4697.385 |
| causal Q=KV32768 | 192 | 5443.132 | 5335.193 | **5317.539** | 5284.699 |
| SWA+sink Q16K/KV128K/W128 | 128 | 103.452 | 102.156 | **101.769** | N/A |
| SWA+sink Q16K/KV128K/W128 | 192 | 119.056 | 117.129 | **116.521** | N/A |

D192 noncausal 较改动前下降 **2.31%**，距 OPUS linear **+1.62%**；D128 causal 本轮
较仅 packed-V 慢 **0.50%**，但仍比改动前快 1.85%，不宣称控制改动处处加速。
OPUS 只计预先准备的等价 linear KV attention，**不是 5D 分页端到端时间**。

最终 D192 noncausal ATT：BFI **656→0/wave**、waitcnt **218→179/wave**；barrier333、
MFMA1640 不变，稳定 phase **2913.6→2787.5 cycles**。动态指令 **13502.5→12217**；
部分 `s_nop` 增加，不把采样阶段改善直接当作全局加速比。新资源见下表，六种 specialization
均无 scratch/spill。完整协议、原始样本、回归和新 ATT 见
[packed_v_control_results.json](packed_v_control_results.json) 与主报告。
优化前根因分析仍保存在 [D192 ATT 分析](d192_noncausal_att.md)；其旧样本不重写。

### 优化前 5D / 显式 OPUS 比较（历史数据）

以下是 packed-V 合入前的一轮数据，不代表当前耗时。MI350X gfx950，BF16，
`B1/Hq16/Hkv1/Dv128/page64`，预分配输出、无 LSE；5 轮交替顺序、每轮 20 warmup /
100 次 `run_perftest` GPU profiler 迭代，取五轮过滤均值的中位数，单位 **µs**。

| 场景 | Dqk | 8-wave 5D direct | 显式 OPUS linear core | 8-wave / OPUS 延迟差 |
|---|---:|---:|---:|---:|
| noncausal Q10240 / KV2583 | 128 | 264.067 | 241.051 | +9.55% |
| noncausal Q10240 / KV2583 | 192 | 303.641 | 292.581 | +3.78% |
| causal Q=KV=32768 | 128 | 4597.028 | 4720.933 | -2.62% |
| causal Q=KV=32768 | 192 | 5444.459 | 5283.595 | +3.04% |
| SWA+sink Q16K/KV128K/W128 | 128 | 101.813 | N/A：不支持 SWA/sink | — |
| SWA+sink Q16K/KV128K/W128 | 192 | 118.101 | N/A：不支持 SWA/sink | — |

- 所有 `*_5d` 候选共享**完全相同的 SHUFFLE-5D cache、随机页表和实际尾页长度**。
  `aiter_5d` 已实际调用 `mha_batch_prefill_func`，六种场景均为 **N/A**：
  `no matching kernel found`。当前 CK 生成器只生成 page1/16/1024，且精确匹配页大小，
  没有 page64 实例；不是 5D ABI 错误，也不回退到 linear。
- **OPUS 当前没有该 5D ABI**。D128 通过 `fmha_fwd_bf16_opus_fwd` dense 4D 入口，
  D192 通过 `fmha_fwd_bf16_opus_varlen_fwd` packed 3D group 入口；profiler 确认
  `gqa_d128_kernel` / `gqa_d192_v128_kernel`。相同逻辑 KV 在计时前重建为 linear，
  不计转换开销，因此表中 OPUS **不是 5D / 分页端到端时间**。
- `aiter_linear` 仍是独立补充候选，D128 full 路由 ASM、D192 full 路由 OPUS、
  SWA+sink 路由 CK；不能统一标为 OPUS。SWA 同输入 4-wave static 为
  D128 **81.945 µs** / D192 **95.748 µs**，仍快于 8-wave。
- 全部候选先通过 FP32 reference（`err=0`）；样本、kernel 名、source/binary hash、
  不可用原因和复现参数见 [paged_5d_opus_results.json](paged_5d_opus_results.json)。
  完整 5D 对照表及有效 TFLOPS / 逻辑 TB/s 见主报告。旧 FlyDSL core 验收与本轮 OPUS
  是不同 baseline，不能据“旧 core 的 3% 内”宣称与 OPUS 全部等速。

### 范围与接口

- 支持 gfx950、BF16、`Dqk=128/192,Dv=128`、page64 SHUFFLE-5D K/V。
- 保留 `PagedAttention` 工厂与原调用参数；支持 GQA/MHA、causal/noncausal、
  ragged batch、非零 prefix 起点、尾页、非连续 Q/O（head dimension 连续）。
- 支持 scalar/per-token-head Q descale、scalar K/V descale；descale 必须有限且为正。
- `out=` 复用输出；`return_lse=True` 返回 `(out, lse)`，LSE 为 FP32
  `[total_q, Hq]`、自然对数；支持传入 `lse=` 和正的 `softmax_scale=`。
- SWA：`window_left>=0` 为 bottom-right causal 闭区间，128 表示最多 129 个可见 key。
  `window_left=-1` 为 full attention；SWA 不要求 sink，且不支持 noncausal。
- Sink：`has_sink=True` 要求同 GPU 的连续 FP32 `sink_ptr[Hq]`，可以独立于 SWA 使用。
  每 head 一个零 value 的虚拟 key，其 logit 不乘 QK scale，只计入一次分母和 LSE。
  支持有限 sink 值与 `-inf`（禁用），不支持 NaN/+inf。
- 空 KV / causal 全遮罩行写 `O=0`；无 sink 时 `LSE=-inf`，有 sink 时等于 sink logit。
  支持指定 stream 和预热后的 HIP graph。
- FP8、其他 head dimension、V192、其他 page size 和旧 fallback 仍未支持，显式报错。
- GPU prefix/page-table 值与 `max_seqlen_q/max_seqlen_k` 必须由调用方保证一致；
  热路径不复制 metadata 到 CPU。物理 KV cache 的 byte span 暂限有符号 32-bit offset。
  页表支持非连续页、重复页和跨请求共享物理页；物理 cache 容量不等于逻辑 KV 长度。

### OPUS 对应关系

- 512 threads、`BM=256/BN=64`；Q 保留 OPUS 布局，K/V 改为直接读取 SHUFFLE-5D page64。
- Q 在 prologue 中占用 V 区域；K 双槽、V 双槽；D192 LDS **149,760 B**，D128 **99,840 B**。
- 完整 `STAGGER=True/False` 编译期分体，仅在入口按 wave-group 分支；每阶段
  保留 scheduler fence、waitcnt 和 workgroup barrier。
- 每个 phase 的 8 个 stage 交织 `QK(t)` 与 `exp/sum/PV(t-1)`；两个 phase
  展开成 ping/pong，偶数 tile 独立收尾。QK/PV 均为原生 BF16 `32x32x16` MFMA，
  P 显式转为 BF16，PV 采用 K-major 双 accumulator 交织。
- 每个 KV tile 只查一次 scalar page table，提前两阶段读取 page ID，经 `lgkmcnt(0)`
  后复用到 K/V 与分段 K DMA。不会用 VMEM 页表读取把 KV DMA 队列清空。
- K 的逻辑 token bits 2/3 置换与 score mask 一致，使连续 P operand 对应连续
  8-token V；K/V 全局 DMA 都是 coalesced 128-bit，V 用 `ds_read_b128`，无 transpose。
- 尾页 V 在访存阶段显式清零，避免 `0*NaN`；清零前保留 LDS wait 与 scheduler fence。
  正向 main phase 的 V 已知是完整页，不做多余检查；reverse phase 和 epilogue 保留保护。
  V fragment 用 packed i32 拼接，避免 BF16 往返表示产生 BFI。
- D128 使用同一八阶段骨架，但每个 QK superunit 只有 8 次 MFMA、2 条 K DMA，
  rolling `vmcnt` 为 4；D192 对应 12 次 / 3 条 / 5。不把 D128 padding 到 D192。
- 复现 lazy-max 阈值 8、stage5 的 stagger 12 / nonstagger 6 个 scale-sub，以及
  OPUS 的 score、row-sum、probability materialization fence。
- causal 仅对需要遮罩的 wave/tile 做 mask；完整 grid 达到 512 WG 时配对首尾
  query block，镜像 block 反向遍历 KV，匹配 OPUS 的负载均衡策略。SWA 禁用首尾配对，
  按 query block 的窗口并集裁剪第一/最后 KV tile。
- 使用 `permlane32` 打包，attention epilogue 每 lane 8 条 128-bit store，无 C-shuffle。
- 布局、寄存器和 MMA 保留 FlyDSL 接口；显式 LDS-read、scalar FMA 和 pin 是局部 ISA
  边界。当前 copy lowering 会为单个共享数组添加多余 `vmcnt(0)`；显式 read 遵守
  调用点的 `lgkmcnt(0)` / barrier，保留 OPUS rolling VMEM waits。

**分页适配完全在 attention 内核内完成。** 输入 K/V 不重排、不预处理、不缓存内容；
只复用编译结果与可选scheduler header，不缓存KV内容。SWA 直接裁剪每个 query block 的
KV页范围，不访问更早的页表项。预热且传入 `out=`/`lse=` 后热路径不分配 GPU tensor。删除了原 Q16K/KV128K 下 D128
64 MiB / D192 80 MiB 的线性 KV workspace；LDS 用量保持不变。

### 优化前对旧 gather 分支纯 core 的验收（历史数据）

MI350X gfx950，`B=1,Hq=16,Hkv=1`，预分配输出、同输入比较；每候选 20 次预热、
100 次采样，5 轮交替顺序取中位数。下表为 `run_perftest` 的 **GPU 时间**，不含编译
和首次分配；PyTorch 仅用于参考，不参与计时。新旧实现在**同一个进程**使用相同
逻辑输入及输出布局，先共同预热，再交替采样。旧实现只计 `attend_linear`，不含 gather；
新实现计完整公开 direct-paged 调用，不扣除页表开销。

| 场景 | Dqk | 新 direct-paged | 旧纯 core | 延迟差 |
|---|---:|---:|---:|---:|
| noncausal Q10240 / KV2583 | 128 | **266.163 µs** | 260.183 µs | +2.30% |
| noncausal Q10240 / KV2583 | 192 | **302.610 µs** | 295.297 µs | +2.48% |
| causal Q32768 / KV32768 | 128 | **4558.574 µs** | 4436.378 µs | +2.75% |
| causal Q32768 / KV32768 | 192 | **5381.104 µs** | 5292.773 µs | +1.67% |
| SWA+sink Q16K / KV128K / W128 | 128 | **102.691 µs** | 101.898 µs | +0.78% |
| SWA+sink Q16K / KV128K / W128 | 192 | **117.633 µs** | 117.045 µs | +0.50% |

六个代表形状均在旧纯 core 的 **3% 内**，SWA 在 **1% 内**；仍有小幅差距，不能称为
完全等速。每轮样本、source SHA256、协议及资源见
[direct_paged_core_results.json](direct_paged_core_results.json)。旧源码仅在工作区外作为
一次性基准保存，当前实现没有 gather/linear fallback。相对旧 core 的最大绝对差：
noncausal 0.00048828125、causal 0.0009765625、SWA 0.001953125；均通过独立 FP32 reference。

以上旧 FlyDSL core 验收数据保持不变；最新 packed-V/control 对照在本文开头及主报告，
另行保存原始采样，不把旧 core 重新命名为 OPUS。以下为**当前源码**新生成的无 LSE 资源：

| attention specialization | VGPR | SGPR | LDS | Private | VGPR / SGPR spill |
|---|---:|---:|---:|---:|---:|
| D128 noncausal，无 LSE | 226 | 55 | 99840 B | 0 B | 0 / 0 |
| D128 causal 首尾配对，无 LSE | 221 | 83 | 99840 B | 0 B | 0 / 0 |
| D128 SWA+sink，无 LSE | 228 | 69 | 99840 B | 0 B | 0 / 0 |
| D192 noncausal，无 LSE | 256 | 59 | 149760 B | 0 B | 0 / 0 |
| D192 causal 首尾配对，无 LSE | 254 | 89 | 149760 B | 0 B | 0 / 0 |
| D192 SWA+sink，无 LSE | 256 | 71 | 149760 B | 0 B | 0 / 0 |

### 验证与复现

[test_pa_prefill.py](test_pa_prefill.py) 已替换为当前 specialization 的严格测试：
**181 passed、6 skipped**。其中原 165 项 direct 测试、8 项比较测试及本轮新增的
8 项 runtime NaN-tail 正向/反向回归全部通过；
6 项 AITER 5D 测试仅因缺失 page64 实例跳过，不吞其他运行错误或数值失败。
显式 OPUS 测试同时验证 FP32 reference 和实际 kernel 名，并拒绝 SWA/sink 语义。
direct 测试覆盖 D128/D192、1–6 页奇偶收尾、NaN poison 尾页、ragged batch、GQA、多种 Q/O stride、
descale 与 lazy-max rescale、LSE、空输入、每次更新 page table/KV、缓存后改变 runtime
长度、stream/graph、首尾配对的奇偶 query-block、默认无 LSE 分支与目标 shape。所有数值失败都会抛出，
不再使用吞异常的 `accuracy unknown`。新增 SWA 边界、有/无 sink、极端 sink、非法
窗口外 prefix 页表，以及精确的 `129/130` 输出 / `log(130)` LSE 检查。
direct-only 检查确保无 `prepare_kv` / `attend_linear` / KV workspace，预分配调用只有
一个 GPU attention event；另有重复物理页与跨请求共享页的回归。
完整测试使用独立 FP32 PyTorch reference；性能对照统一零填充尾页以兼容旧 4-wave。

当前运行环境使用 `/opt/venv/bin/python`（Python 3.10.12，ROCm 7.2）；从本目录运行：

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python -m pytest -q test_pa_prefill.py
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python test_pa_prefill.py --head-dim 128 192
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python test_pa_prefill.py --q-len 32768 --kv-len 32768 --causal 1 --head-dim 128 192
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python test_pa_prefill.py --q-len 16384 --kv-len 32768 65536 131072 --causal 1 --window-left 128 --sink 1 --head-dim 128 192
```

基准先严格检查可运行候选，再输出包含 `*_5d`、显式 `opus_linear` 和补充
`aiter_linear` 的 markdown 表；`err` 必须为 0。缺失 5D 实例与 OPUS 不支持的语义
显示 `unavailable`，不提供替代计时。每个形状还输出 `5D_OPUS_RESULT` JSON，含全部
五轮样本、实际 dispatch 和精确不可用原因。资源复核可在上述命令加
`FLYDSL_DUMP_IR=1` 查看生成 ISA。

---

## 历史记录（2026-09-03，旧实现）

更新时间：2026-09-03。完整的4-wave/8-wave/AITER统一数据、口径和gfx942历史结果见
[4-wave主报告](../pa_4wave/README.md)；本文件只维护8-wave实现及其当前摘要。

## 功能范围

- 支持gfx942/gfx950 BF16与架构原生FP8 vectorized paged-prefill；
- gfx942 FP8使用`torch.float8_e4m3fnuz`，gfx950使用OCP
  `torch.float8_e4m3fn`，launcher拒绝非原生编码；
- 覆盖D128/D192、causal/non-causal、page 32/64/128、per-token/per-tensor Q scale、
  ragged tail及gfx950 SWA + sink；
- gfx950完整回归：`67 passed, 1 skipped`。

## 当前性能摘要

平台为MI350X gfx950，主配置为`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。
除特别说明外，计时为20次预热、100样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | 8-wave结果 | 对照结论 |
|---|---|---:|---|
| non-causal `Q10240,KV2583` | BF16 native 512-thread | 350.43 us / 772.89T | API始终运行原生8-wave；同轮4-wave为304.07 us |
| causal `Q=KV=32768` | BF16 native 512-thread | 8392-8463 us（历史） | 已删除4-wave自适应回退 |
| causal `Q=KV=32768` | OCP FP8 K16 | 5611.21 us / 979.75T | K64前基线 |
| causal `Q=KV=32768` | OCP FP8 K64 | **5101.77 us / 1077.58T** | 较K16延迟下降9.08% |
| SWA `Q16K,KV128K,window=128` | BF16 native 512-thread | 140.49-142.20 us（历史） | 已删除4-wave自适应回退 |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | **112.35 us / 192.64T有效 / 573.43T执行** | 较K16下降0.49% |

SWA有效FLOPs按每行129个可见KV token计算；执行TFLOPS计入被mask但仍进入MFMA的
tile工作。完整SWA过程数据见[gfx950_swa_performance.md](gfx950_swa_performance.md)。

## AITER路由与SWA口径

non-SWA同输入BF16测试按长度显式选择并用profiler验证AITER入口：

| 条件 | 公开入口 | 实际事件 |
|---|---|---|
| `Q == KV` | `mha_batch_prefill_func` | `aiter::mha_batch_prefill` |
| `Q != KV` | `flash_attn_varlen_func` | `FlashAttnVarlenFunc` |

AITER等长D192/V128使用linear/page-size-1 ABI，非等长使用linear THD；8-wave使用
page64 vectorized cache。SWA上两种线性API都支持window与sink：
`mha_batch_prefill_func`命中`aiter::mha_batch_prefill`，
`flash_attn_varlen_func`命中`aiter::mha_varlen_fwd`。后者更快并覆盖128K。
真正direct-paged只有`mha_batch_prefill_func`具备ABI，但当前gfx950 BF16 D192/V128
page64构建没有匹配specialization。默认AITER回归为`3 passed`。

## 实现与优化

- 公开API始终运行原生512线程8-wave kernel，不再导入或调用4-wave后端；
- workgroup为`BM256 x BN32 x 512 threads`，8个wave分成两组交织QK MFMA与
  online softmax；
- 1 WG/CU persistent grid通过atomic ticket遍历`batch x query tile x Q head`任务；
- 原生gfx950 D192/V128 BF16为两个错相4-wave组分别使用两槽K LDS ring；K通过
  buffer-to-LDS direct DMA写入pair-padded视图，再由K16置换视图读取，V保持
  `global -> register`。两个组不能共享异步写入的ring，否则KV96起出现非确定覆盖；
- 原生BF16将P@V保留在MFMA stage，并显式拆成两个K16 atom组；目标ISA为233 VGPR、
  100 SGPR、49940 B LDS、2个SGPR spill、0 VGPR spill、0 scratch；目标specialization
  直接写VMEM，静态`S_BARRIER`
  从C-shuffle的46个降到31个；
- 32x32 MFMA减少row max/sum cross-lane操作，`v_permlane`避免经LDS交换；
- B@A `fx.gemm`使QK结果布局直接匹配P@V输入，K加载同时置换KV-length维；
- `-packed-fp32-ops`避免与MFMA有co-issue问题的packed FP32 VALU；
- `work_counter`按device/stream/grid复用并由最后退出的workgroup复位；
- lane 0通过独立4-byte LDS mailbox广播ticket，单barrier同时封闭上一work item的
  C-shuffle生命周期；
- 1/2/3/4 WG-per-CU sweep在B=4为295.49/298.49/300.95/302.72 us，保留1 WG/CU。

gfx950 OCP FP8 QK使用`v_mfma_f32_32x32x64_f8f6f4`和unity E8M0 scale；P@V
reduction只有32，保留K16 FP8 MFMA。Dqk不能被64整除时回退K16。

## 资源

gfx950 OCP FP8 D192 fresh ISA：

| 版本 | QK / P@V静态站点 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| K16 | 48 K16 / 32 K16 | 174 | 72 | 12292 B | 0 B | 0 | 0 |
| K64 | 12 K64 / 32 K16 | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64将QK静态MFMA站点减少4倍，仅增加2个VGPR。gfx950 BF16原生D192/V128 direct-LDS
specialization为233 VGPR、100 SGPR、49940 B LDS、0 private、2个SGPR spill、
0 VGPR spill、0 scratch。

## BF16 K/V LDS结果与TODO

- 历史register-to-LDS路径的自然padding写视图与K16置换读视图将K-only bank
  conflict从81.85%降到3.76%，目标shape独立A/B从约420 us降到397.71 us；
- V跨wave复用将`SQ_INSTS_VMEM_RD`从4.319M降到1.395M，但总LDS冲突为33.50%，
  K-only到K+V LDS从397.71 us回退到410.95 us；正式三方轮转协议为449.06 us；
- 历史共享K ring恢复direct-V并从三槽减到两槽后，LDS从62980 B降到25620 B、VGPR
  从241降到218；当前direct-LDS为保证两个错相4-wave组互不覆盖，改为独立双槽ring；
- 目标specialization用64-bit direct output store替代C-shuffle，输出最大差
  `0.0078125`，资源不变；三轮同进程A/B为409.14/407.66、412.17/411.33、
  412.32/411.79 us，对应稳定0.13%-0.36%提速；
- K预取改为`raw_ptr_buffer_load_lds`，每条64-lane DMA搬两个32-row D-group，每个
  pair后padding 16 B；K侧`ds_write_b128`静态站点归零。KV2583硬件计数为
  `SQ_INSTS_LDS=510528`、`SQ_LDS_BANK_CONFLICT=0`；双subgroup ring三轮为
  409.80/410.20/410.11 us；删除过时K地址重物化后的最终同轮中位为358.19 us。该改动减少K中转
  寄存器和DS write，但当前没有带来性能收益；
- 最终`MfmaUtil`为31.90%-32.56%，occupancy约2.00。自然V padding虽将冲突率降到
  0.37%，却使延迟升到749.39 us，因此冲突率不能作为V布局的单一验收指标。

后续TODO：

1. 将内部KV tile改为64并保留两个score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`重叠，同时将softmax频率减半；外部page ABI继续保持BN32。
  已验证的BN32双score不会降低softmax频率，不能替代该改造。
2. 将8个wave拆成两组4-wave并错开一个pipeline stage；要求`MfmaUtil > 50%`、
   0 scratch，且不得恢复K-LDS冲突。

已否决实验：

- D4将两个子组的K ring拆成四个独立LDS字段：`391.55 -> 412.52 us`，约回退5.4%，
  VGPR从237升到240；与D5组合后为434.64 us，因此恢复单数组双ring；
- D5按4-wave顺序将K DMA移到K DS-read之后并移除稳态`vmcnt(0)`：与D4组合后
  `391.55 -> 434.64 us`，约回退11.0%。8-wave全WG barrier下反而损失DMA/softmax覆盖；
- D11/D12最初在K地址重物化workaround仍存在时分别回退3.6%/2.9%，组合也回退；
  删除该workaround后重新评估，显式max树、direct permlane32、无偏lazy-max和
  fused scale-sub将VGPR从235降至233，同进程`347.79 -> 344.87 us`，提速0.85%，
  因此当前已保留；
- D17用现有128-bit C-shuffle替换64-bit direct store：输出逐bit一致，但
  `371.33 -> 373.39 us`，回退0.55%且增加barrier，因此保留direct store；
- 将score scale或半个softmax跨barrier前移分别回退约5%-6%和约14%；
- 在同一BN32内把前8个EXP与第一组P@V MFMA交织，ISA生效但回退约1%；
- 删除任一全WG barrier可做到正确的lockstep版本，但因失去两组4-wave错相，回退
  约7%-13%；gfx950不支持gfx1250 named subgroup barrier；
- BN32双score一barrier原型完成了K-ring代次、ragged-tail和最终MFMA收口验证，目标
  输出与native baseline逐bit一致；但三轮为547.20/546.10/549.66 us，对照native
  390.53/389.99/391.89 us，回退约40%，因此移除。真正有价值的方向仍是KV64，
  同时降低softmax频率并设计两组4-wave错相，而不是只延长score生命周期。

stage边界修复分两步：先将目标BF16的8条P@V MFMA从`p0`移回`p1`，同进程
`370.25 -> 367.39 us`；再保留K-DMA在softmax前提前发射，仅将`setprio(0)`边界后移
到3条K-DMA之后，同进程`345.09 -> 342.53 us`，提速0.75%。直接把K-DMA物理移动到
K DS-read之后会失去异步覆盖并回退`326.43 -> 370.09 us`，因此未保留。最终动态ATT
为`p0 = 17 EXP`、`p1 = 20 MFMA + 12 DS-read + 3 K-DMA`，与4-wave职责一致；两阶段
中位分别为736/1020 cycles，两个4-wave子组继续反相执行。
粗粒度
`optimize_native_bf16`已删除，拆为`use_split_bf16_pv`与
`use_direct_output_store`；拆分前后纯指令序列SHA256一致。

K LDS读取原先用side-effecting `v_mov(tid)`阻止后端外提地址计算；当前LLVM删除该
workaround后只保留一个循环不变量DS-read基址`v100`，没有恢复旧版的12个长寿命地址，
VGPR从241降至235，private/scratch不变。同进程A/B为`378.55 -> 340.91 us`，提速
11.04%，KV96与目标shape输出逐bit一致；此时三方测量为8-wave `358.19 us / 756.15T`、
4-wave `304.95 us / 888.17T`。随后对齐4-wave VALU后，最终三方测量为8-wave
`352.09 us / 769.25T`、4-wave `304.99 us / 888.05T`。stage边界继续对齐后，最终
三方测量为8-wave `350.43 us / 772.89T`、4-wave `304.07 us / 890.74T`。证据：
`tests/flydsl/pa_8wave/ui_output_agent_65497_dispatch_24`。按每wave每BN32归一化，
softmax核心两边均为`18 FMA + 17 EXP + 16 ADD + 7 MAX3 + 2 MAX + 1 permlane + 1 SUB`；
含MFMA stage地址控制后，4-wave/8-wave总VALU分别为90.10/87.41。

当前源码SHA256：`160b38fcc74696fd05115d6ebfbe757b34592f53fde56c713b2e64b201bada7b`。

## 复现

```bash
python -m pytest -q tests/flydsl/pa_8wave/test_pa_prefill.py
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

