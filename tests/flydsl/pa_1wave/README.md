# 单 wave SWA attention：gfx950 / BF16 / 16×16 MFMA

2026-09-06，MI350X gfx950 GPU0，256 CU。独立实现见
[swa_1wave.py](swa_1wave.py)，不是4/8-wave包装或fallback。

## 结果与适用范围

- **一个 workgroup 恰好一个 wave64，64线程；所有 MFMA 都是16×16。**
- BF16 Q/K/V/O，Dqk128/192、Dv128，page64 SHUFFLE-5D，bottom-right causal SWA。
- 支持独立sink、LSE、GQA/MHA、ragged/empty、strided Q/O、非单位scale、运行时页表更新、stream/graph。
- **无gather、无LDS分配/读写、无workgroup barrier、无辅助GPU workspace、无persistent队列。**
- W128、Q16K主矩阵：比4-wave static耗时降低 **11.25%–13.10%**，比当前已裁剪8-wave
  static降低 **24.51%–28.41%**。这里“降低”是相对耗时，不混用吞吐加速比。
- **244项测试通过，0失败/0跳过**；36次原生资源审计；最终4组benchmark文件共168个
  candidate结果、840个原始计时样本，全部参考误差比例为0。
- **不宣称所有shape全局最优。** W1024时，现有8-wave更快，D192还慢于4-wave约3.9%；
  本实现主要解决窄SWA，宽窗口数据完整保留在后文。

新内核没有修改现有[4-wave](../pa_4wave/pa_prefill_4wave.py)或
[8-wave](../pa_8wave/pa_8wave_950.py)，也没有修改之前的gfx942文件/压缩包。
本任务未commit/push。

## 为什么采用一 wave、两个16行子tile

[原SWA分析](../pa_8wave/swa_analysis.md)指出，W128表示每个query最多129个可见key：
4-wave CTA的128行query会形成256-token窗口并集；8-wave CTA的256行形成384-token并集。
现有8-wave虽然裁掉部分完全不可见的MFMA，但仍保留宽CTA的访存、LDS和同步成本。

本实现先尝试BM16：确实缩小了窗口并集，却将Q16K/H16的wave数翻倍到16384，且K/V
重复读取、细粒度softmax和访存等待过多，反而慢于4-wave。因此最终分为：

| 条件 | 每CTA query | KV tile | wave/CTA | 用途 |
|---|---:|---:|---:|---|
| `window_left <= 16` | 16 | 16 | 1 | 极窄窗口，优先少算 |
| 其他 | 32 = 2×16 | 32 | 1 | 同一wave的两组query共享K/V |

可以显式指定 `query_tile=16/32`、`block_n=16/32/64`；这些仍然全部是单wave、16×16 MFMA，
不是改变workgroup wave数。默认选择来自[最终tile扫描](final_tile_results.json)，没有运行时
autotune或隐含4-wave调用。W16的D128 Q16/Q32 BN16接近，默认选择计算并集更窄的Q16。

### 主路径流水

BM32/BN32，D192，每lane持有两组Q、两组O和小的softmax状态。逻辑顺序：

```text
readfirstlane(batch/page metadata)
load Q[0:16], Q[16:32]
initialize max, local_sum = sink_contribution / 4, O = 0
first = align_down(max(diagonal_start - window_left, 0), BN)
end   = max(min(diagonal_start + valid_query_rows, kv_length), 0)

for tile in [first, end), step BN:
    scalar load physical page ID
    load K into packed registers; wait K
    issue V loads
    QK for both 16-row query subtiles, using tiled 16×16×32 MFMA
    retire K before PV
    for each query subtile:
        mask only boundary tiles; full-visible interior needs no element mask
        row_max = max across each lane's scores and four lane groups
        lazy max: rescale O/local_sum only when max advances by >8 in log2 units
        P = exp2(score - max); keep denominator sum lane-local
        convert P to BF16
    wait V; clear invalid V-tail halves even if P is zero
    PV for both query subtiles, sharing V, using tiled 16×16 MFMA

reduce denominator across four lane groups once
normalize O, apply V scale, store BF16 O and optional FP32 LSE
```

这是源程序和依赖关系，**不声称LLVM一定把V load提前到所有QK之前**；最终ISA仍会做
寄存器压力驱动的调度。Q16采用更简单的逐tile K/V load→QK/softmax/PV流程，不保留
实验中失败的深预取或whole-window路径。

### 数据布局与数值约定

- K：`[physical_page, Hkv, Dqk/8, 64, 8]`；V：`[physical_page, Hkv, 8, 128, 8]`。
- Q/O最后一维连续，支持非重叠的padded/head-major布局及非零storage offset。
- QK使用 `MFMA(16,16,32,BF16)`。PV在BN16用K16 atom，BN32/64用K32 atom。
- 四个lane group共同负责同一query row；`lane & 15`是行，`lane >> 4`是列组。
  BN32/64的K行排列将P寄存器排成每lane连续8个token，直接喂PV，无LDS转置。
- packed K/V先写连续rmem storage，再用 `fx.select(...,[0,2,1])`得到MMA fragment视图。
  直接对置换layout做原始vector store曾导致错误，已用独立地址模型和GPU测试覆盖。
- 页内tile位移放在SGPR `soffset`，lane地址跨循环不变；页表值每次launch重新读取。
- SWA包含左右端点：`0 <= diagonal - key <= window_left`。用unsigned距离合并两项比较；
  有效query天然满足 `diagonal < kv_len`。只有边界tile才执行逐元素mask。
- sink是每head一个**未乘softmax scale的自然logit**，只进入分母；四组lane各计1/4份，
  epilogue归约后恰好一份。`sink=-inf`等价禁用。无key且无有效sink：O=0、LSE=-inf。
- sum跨lane归约延后到epilogue，max仍逐tile归约。BF16指数范围允许lazy-max阈值8，
  不必每次max微小改变就重缩放32个输出寄存器。大logit/非单位scale测试保留严格容差。
- K/V尾页用NaN污染测试；**V必须清零无效半字**，不能依赖 `P=0`，因为 `0×NaN`仍是NaN。

## 原生PMC：计算浪费实际减少多少

D192/Q16384/KV131072/H16/Hkv1/W128/sink，三次全grid采集，计数一致。
数据见[pmc_results.json](pmc_results.json)，基线MFMA形状由新鲜
[4-wave ISA资源](resource_4static.json)和[8-wave ISA资源](resource_8static.json)确认。

| 指标 | 新1-wave | 4-wave static | 当前裁剪8-wave static |
|---|---:|---:|---:|
| workgroup线程 | 64 | 256 | 512 |
| `SQ_WAVES` | 8192 | 8192 | 8192 |
| `SQ_INSTS_MFMA` | 1,638,400 | 1,310,720 | 983,040 |
| MFMA形状 | 16×16×32 | 32×32×16 | 32×32×16 |
| 每条MFMA FLOPs | 16,384 | 32,768 | 32,768 |
| 执行MFMA FLOPs / G | **26.844** | 42.950 | 32.212 |
| 有效FLOPs / 执行FLOPs | **80.625%** | 50.391% | 67.188% |
| `SQ_INSTS_VALU` | 9,060,637 | 14,926,304 | 15,520,129 |
| `SQ_INSTS_VMEM_RD` | 950,272 | 917,504 | 425,984 |
| `SQ_INSTS_LDS` | **0** | 1,081,344 | 2,064,384 |
| `SQ_LDS_BANK_CONFLICT` | **0** | 786,432 | 0 |

**不能直接比较MFMA条数：新16×16×32每条只有旧32×32×16一半的FLOPs。**
归一化后，新实现比4-wave少 **37.5%** MFMA计算，比已裁剪8-wave少 **16.67%**。
主shape的新BM32并集160token，比4-wave256和裁剪8-wave192更窄。

并不是“零浪费”：160-token矩形并集对每row只有129个有效token，剩余19.375%的矩形计算
包括两侧三角mask。BM16能将主shape并集缩到144，但实测数据复用/调度代价高于省下的计算。
选择BM32是实际吞吐权衡，不是把16×16 MFMA改成32×32。

VMEM**指令数不是HBM字节**。本实现甚至比4-wave多3.57%的VMEM读指令，却减少MFMA FLOPs、
VALU、LDS和同步，不能只归因为“HBM流量减少”。本任务不拿instrumented PMC时间代替benchmark，
也没有从历史8-wave ATT结构套推单wave延迟。

## 性能：同输入、预分配O、无LSE

全部：B1/Hq16/Hkv1/Dv128/page64/BF16，sink每head FP32 -1→1。同一shape的所有候选使用
**同一物理5D缓存和页表**，先完整chunked FP32 reference，再3次bit-exact重复检查。
100轮共同预热，每候选20 warmup/100 iterations，5轮交替正反顺序，报告中位数µs。
保持机器原有auto-DPM，未锁频或改功耗。原始异常偏快/偏慢样本不删除。

### W128，Q16K主矩阵

| Dqk | KV | 新1-wave | 4 static | 4 dynamic | 8 static | 8 persistent |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32768 | **72.098** | 81.811 | 87.680 | 99.993 | 98.100 |
| 128 | 65536 | **71.948** | 82.798 | 88.239 | 100.495 | 97.910 |
| 128 | 131072 | **72.385** | 82.135 | 87.944 | 100.603 | 97.921 |
| 192 | 32768 | **84.599** | 95.321 | 102.492 | 113.629 | 113.285 |
| 192 | 65536 | **85.219** | 97.108 | 102.783 | 113.357 | 113.045 |
| 192 | 131072 | **85.471** | 96.781 | 102.952 | 113.225 | 113.116 |

原始5轮数据、实际dispatch名字、err和source SHA256：[final_results.json](final_results.json)。

Q16K/KV128K/W128吞吐指标（有效attention FLOPs；TB/s是逻辑最低字节，不是测量HBM带宽）：

| Dqk | candidate | µs | 有效TFLOPS | 逻辑TB/s | err |
|---:|---|---:|---:|---:|---:|
| 128 | 1-wave | 72.385 | 239.194 | 1.971 | 0 |
| 128 | 4 static | 82.135 | 210.800 | 1.737 | 0 |
| 128 | 4 dynamic | 87.944 | 196.877 | 1.622 | 0 |
| 128 | 8 static | 100.603 | 172.104 | 1.418 | 0 |
| 128 | 8 persistent | 97.921 | 176.818 | 1.457 | 0 |
| 192 | 1-wave | 85.471 | 253.215 | 2.087 | 0 |
| 192 | 4 static | 96.781 | 223.625 | 1.843 | 0 |
| 192 | 4 dynamic | 102.952 | 210.221 | 1.732 | 0 |
| 192 | 8 static | 113.225 | 191.147 | 1.575 | 0 |
| 192 | 8 persistent | 113.116 | 191.331 | 1.577 | 0 |

### 窗口范围，Q16K/KV128K

| Dqk | W | 新1-wave | 4 static | 8 persistent |
|---:|---:|---:|---:|---:|
| 128 | 0 | **41.837** | 52.612 | 73.368 |
| 128 | 16 | **48.910** | 67.742 | 85.332 |
| 128 | 64 | **58.855** | 69.335 | 86.611 |
| 128 | 512 | **147.694** | 155.886 | 157.546 |
| 128 | 1024 | 246.389 | 248.876 | **233.565** |
| 192 | 0 | **49.945** | 62.944 | 83.015 |
| 192 | 16 | **57.332** | 80.437 | 96.080 |
| 192 | 64 | **67.961** | 81.190 | 98.171 |
| 192 | 512 | **177.598** | 179.021 | 180.346 |
| 192 | 1024 | 299.048 | 287.846 | **267.282** |

完整五候选数据：[final_window_results.json](final_window_results.json)。W512/D192对4-wave
只有约0.8%差距，不宣传为稳健大收益。W1024仍正确，但不是性能推荐区：长窗口重用/流水
成本重新占主导，单wave不是普遍最优；没有偷偷换到8-wave掩盖这一点。

### Query规模，KV128K/W128

| Dqk | Q | 新1-wave | 4 static | 8 persistent |
|---:|---:|---:|---:|---:|
| 128 | 256 | **8.922** | 13.223 | 17.951 |
| 128 | 2048 | **13.077** | 15.756 | 20.880 |
| 128 | 4096 | **20.463** | 21.337 | 28.413 |
| 128 | 65536 | **291.518** | 316.876 | 368.025 |
| 192 | 256 | **9.801** | 14.351 | 19.527 |
| 192 | 2048 | **15.744** | 17.869 | 23.604 |
| 192 | 4096 | **24.415** | 24.889 | 32.318 |
| 192 | 65536 | **346.921** | 371.445 | 422.293 |

完整五候选数据：[final_scaling_results.json](final_scaling_results.json)。没有把不同轮的
候选最低值拼成一个“最佳”表。Q4K/D192与4-wave差距较小，按实测报告，不外推未测shape。

## 寄存器与原生指令审计

所有36次单wave编译并实际执行均满足：64线程/workgroup、wavefront64、16×16 MFMA、
LDS0、scratch0、VGPR/SGPR spill0、`s_barrier`静态位置0。

默认路径资源，表内寄存器是ISA/ELF metadata，不使用profiler错误折算后的VGPR字段：

| Dqk | W | BM/BN | 无LSE VGPR/SGPR | 有LSE VGPR/SGPR | 默认AGPR |
|---:|---:|---|---|---|---:|
| 128 | 0 | 16/16 | 116/33 | 118/34 | 0 |
| 128 | 16 | 16/16 | 114/31 | 116/32 | 0 |
| 128 | 128 | 32/32 | 213/34 | 236/34 | 0 |
| 192 | 0 | 16/16 | 127/42 | 136/40 | 0 |
| 192 | 16 | 16/16 | 128/36 | 127/36 | 0 |
| 192 | 128 | 32/32 | 238/36 | 248/34 | 0 |

[default_resources.json](default_resources.json)有这12项；[resources.json](resources.json)有
W128下2个Dqk×2个query tile×3个KV tile×LSE开关=24项。
BN64部分候选的metadata `vgpr_count`超过256且使用AGPR；例如D192 BM32/BN64无LSE为
combined303、AGPR47，有LSE为374、AGPR118。**零spill不等于零AGPR。**这些候选仍正确，
但不是默认；完整区分见[validation.json](validation.json)。

原生审计保存ISA hash、实际路径、MFMA形状及资源。ISA文件在本机临时dump目录，JSON不是
内嵌完整ISA；换机器可用下节入口重生成。4-wave D192静态为208VGPR/46SGPR/LDS24960B，
当前8-wave为256/76/LDS149760B；均无scratch/spill。

## 优化过程：保留失败结果，不只展示赢家

下表是各阶段独立同进程A/B的D192主shape中位数µs。跨阶段不能当严格A/B；每个原始JSON
都保留该阶段同时采样的4/8-wave数据。

| 阶段 | 新候选µs | 同轮4 staticµs | 结论 / 原始数据 |
|---|---:|---:|---|
| 初版BM16/BN32，逐tile归约 | 118.233 | 95.570 | 省计算但更慢：[initial_results.json](initial_results.json) |
| whole-window，一次softmax | 120.928 | 95.995 | 得不偿失：[whole_window_results.json](whole_window_results.json) |
| whole-window三槽预取 | 126.131 | 94.788 | 寄存器压力加重：[whole_prefetch_results.json](whole_prefetch_results.json) |
| Q16拆K/V预取 | 127.442 | 95.109 | 没有解决核心复用：[split_pipeline_results.json](split_pipeline_results.json) |
| 同wave两组Q，BM32/BN32 | 96.127 | 95.600 | 接近4-wave：[query32_results.json](query32_results.json) |
| naive lazy-max | 110.561 | 95.653 | combined寄存器升275：[lazy_max_results.json](lazy_max_results.json) |
| 两组QK先于PV，缩短K生命周期 | 90.781 | 95.825 | 首次稳定领先：[query32_split_results.json](query32_split_results.json) |
| paired tiled GEMM | 91.060 | 95.448 | 交错收益小：[query32_interleaved_results.json](query32_interleaved_results.json) |
| 单比较mask、非负页索引 | 90.996 | 96.236 | 简化地址/掩码：[query32_mask_results.json](query32_mask_results.json) |
| lane-local sum，末尾归约 | 89.029 | 96.328 | 循环少两次跨lane归约：[query32_local_sum_results.json](query32_local_sum_results.json) |
| 仅边界逐元素mask | 88.852 | 95.491 | 内部tile快路径：[query32_boundary_mask_results.json](query32_boundary_mask_results.json) |
| tile偏移移入SGPR | 85.647 | 95.630 | 保留：[query32_scalar_offsets_results.json](query32_scalar_offsets_results.json) |
| 页表ID跨tile复用分支 | 87.051 | 95.350 | 拒绝，省读取却增加控制：[query32_page_reuse_results.json](query32_page_reuse_results.json) |

Q32下一K预取也没有稳定收益，D192明显回退，见
[query32_prefetch_results.json](query32_prefetch_results.json)。早期窗口扫描见
[window_tuning_results.json](window_tuning_results.json)。最终移除这些失败实现及其API，只保留
测得有用的Q16/Q32两条单wave路径。最终再次扫描6种tile：[final_tile_results.json](final_tile_results.json)。

**早期实验JSON不是最终验收：**部分没有source hash，名字含已删除的prefetch/whole-window；
它们仅记录优化过程，不能按当前源码复现出相同实验程序。最终4组结果均带准确kernel hash和
dispatch名。主/窗口/query扫描后只增加了CPU布局与MHA测试，benchmark函数和kernel没改，
原结果保留当时的test hash，不篡改为后来的文件hash。

## 验证与复现

### 测试范围

[test_swa_1wave.py](test_swa_1wave.py)包含244项：

- D128/D192×BM16/32×BN16/32/64；NaN尾页、所有mask行、query/KV边界。
- W0/1/15/16/17/31/32/33/63/64/65/127/128/129/512/1024。
- ragged空请求、GQA/MHA、head-major/padded Q/O、metadata offset、非零storage offset及guard。
- per-token/per-tensor scale、大logit、lazy max，sink ±80/0/-inf，严格LSE。
- W128零logit恰好129 key+1 sink，精确BF16输出129/130；禁用sink输出1。
- 运行时KV长度/页表/缓存/sink变化不重新编译；非法但完全排除的前缀页表不读取。
- 双stream graph replay、输出自动分配、重复bit-exact；warm路径无GPU额外分配且只有1次dispatch。
- CPU独立验证K/V地址覆盖双射、输出位置唯一、tile访问与逐row窗口集合严格相等。

误差要求：BF16 O `rtol=atol=0.02`，FP32 LSE `rtol=2e-4, atol=5e-4`；未放宽容差。
所有benchmark调用 `checkAllclose` 后还**assert返回err为0**，不是只看日志。
验收索引与log/hash：[validation.json](validation.json)。

### 环境和命令

本机Python3.10.12、FlyDSL0.3.1、PyTorch2.9.1+rocm7.2.0.git7e1940d4、HIP7.2.26015-fc0010cf6a，
pytest9.0.3，AITER使用现有workspace安装。可执行解释器是 `/opt/venv/bin/python`，不要误用
没有GPU依赖的workspace空venv。kernel本身仅依赖torch/FlyDSL；测试还依赖AITER、pytest、
pandas/tabulate及上级4/8-wave测试/reference。

```bash
cd /host_lc/pyhip
/opt/venv/bin/python -m pytest tests/flydsl/pa_1wave/test_swa_1wave.py -q
/opt/venv/bin/python tests/flydsl/pa_1wave/test_swa_1wave.py \
  --kv 32768 65536 131072 --output /tmp/swa_main_retest.json
/opt/venv/bin/python tests/flydsl/pa_1wave/test_swa_1wave.py \
  --candidates 1w_tiles 1w_auto 4w_static --query-tile 16 32 --bn 16 32 64 \
  --window 0 16 128 --output /tmp/swa_tile_retest.json
```

原生ISA及PMC入口：[profile_swa_1wave.py](profile_swa_1wave.py)，counter配置
[pmc.txt](pmc.txt)。dump目录必须是新的，避免读到旧ISA。

```bash
/opt/venv/bin/python tests/flydsl/pa_1wave/profile_swa_1wave.py --dq 192 \
  --dump-dir /tmp/swa_new_isa --output /tmp/swa_new_resource.json
/opt/rocm/bin/rocprofv3 -i tests/flydsl/pa_1wave/pmc.txt --output-format csv \
  --kernel-include-regex '_swa32_kernel' --kernel-iteration-range '[21-23]' \
  -d /tmp/swa_new_pmc -o counters -- \
  /opt/venv/bin/python tests/flydsl/pa_1wave/profile_swa_1wave.py --dq 192
```

4-wave过滤器是 `^attention_kernel_static`，8-wave是 `_attention_kernel`，用相应
`--candidate 4static/8static`。采集程序总共launch23次，选最后3次；rocprofv3本机默认输出
数据库，必须显式 `--output-format csv` 才有CSV。工具只核对正确性/机制，不输出benchmark。

### ABI与边界

工厂 `PagedAttention(...)` 与现有paged调用参数顺序兼容；可选 `out`、`sink_ptr`、`stream`、
`return_lse`、`lse`、`softmax_scale`。`cu_seqlens_k`可为None，实际KV长度由 `kv_indptr` 和
`kv_last_page_lens`推导。首次调用编译并launch，warm调用复用编译结果，元数据值不缓存。

调用方须保证device元数据值一致、可访问页号合法、actual Q长度不超过 `max_seqlen_q`，
Q/O是非重叠layout且不互相alias，scale/logit在FP32可表示范围内，sink为有限值或-inf。
host验证dtype/shape/device/连续性；不通过device→host同步逐项验证metadata。物理K/V各自
限制在signed-int32 byte offsets范围内。只针对gfx950，**不是gfx942兼容版本**；FP8、其他
head_dim/page/layout、非causal/full attention不在此专项实现范围。

最终kernel SHA256：`44b32d3bc5feb29e8088686397babb4552bfe5c3ed18653c5041e1657b07bf01`。
建议窄SWA优先评估此实现，宽W1024保留现有4/8-wave选择；“最佳”仅指已验证范围内的
实测候选，不保证未测shape、模型logit分布、不同软件/频率条件下全局最优。