# Batched GEMM core ceiling

[`probe-batched-gemm-core-ceiling.py`](probe-batched-gemm-core-ceiling.py)生成均衡batched GEMM的FP8 core co-issue skeleton。A/B为FP8，D为BF16。它用于快速测量候选tile在指定waves/WG、waves/SIMD和schedule下的VMEM/MFMA共发ceiling，不计算正确GEMM结果。

## V0边界

- 每个batch元素具有相同`M/N/K`；
- 每个WG处理一个`BM x (BN * n_tiles_per_wg)`输出tile组；
- `waves_m * waves_n`必须为4、8或16；
- 每wave覆盖`BM/waves_m x BN/waves_n`；
- A可由VMEM读取或用`--a-in-reg`仅预留寄存器；B始终由VMEM读取；
- D在全部K tile完成后连续写出；
- VMEM load进入独立sink，MFMA使用独立operand寄存器，D写独立payload寄存器；
- MFMA默认只保留一个4-AGPR写目标，C输入固定为立即数0；总MFMA按几何保持不变，但不存在accumulator RAW链；
- LDS仅用于实现请求的驻留waves/SIMD，不发出任何`ds_*`指令；
- 支持`2stage_0/2stage_prio/2stage_barrier/interleave`，语义与wave-stage probe一致；
- `--cross-n-prefetch`实验路径在K2预取下一个N tile的K0，仅支持`2stage_0/2stage_prio`；实测回退，因此默认关闭；
- `--cross-n-spread-stores`需配合cross-N预取，将上一N的8条store等距放入下一N的96条MFMA中；最后N因无后继仍集中写出；实测回退，默认关闭；
- 支持`batch_m_n/batch_n_m`两种WG线性顺序。
- `--n-tiles-per-wg`接受任意正整数；同WG串行处理多个BN tile，A-reg只保留一份，B/D按N tile递增；

因此结果名为`gemm_core_coissue_ceiling`。它是自动skeleton witness，不包含VMEM到MFMA的数据RAW依赖、LDS搬运、scale、activation、reduction或真实epilogue。

寄存器只用于保留候选资源和指令流，不做人工数据初始化：A-reg、独立MFMA operand和D payload均不要求数值正确。`--accumulator-destinations`支持1/2/3/4、默认1；当前五个case都使用一个4-AGPR write-only目标，所有MFMA的C输入均为0。这样不会把逐寄存器`v_mov`混入core ceiling；结果不可用于正确性验证。

## 工作推导

```text
waves/WG = waves_m * waves_n
wave_M = BM / waves_m
wave_N = BN / waves_n

M_tiles = ceil(M / BM)
N_tiles = ceil(N / BN)
N_tile_groups = ceil(N_tiles / n_tiles_per_wg)
K_tiles = ceil(K / BK)
workgroups = batch * M_tiles * N_tile_groups

MFMA/wave/K = (wave_M / 16) * (wave_N / 16) * (BK / 32)
A bytes/wave/K = wave_M * BK
B bytes/wave/K = wave_N * BK
D bytes/wave = wave_M * wave_N * 2
```

V0要求每类wave级VMEM字节数能被1024 B整除。A使用`--a-in-reg`时不发A read，但按完整K维的`wave_M * K_padded / (64 * 4)`预留VGPR；这是整个A wave tile在K循环期间保持寄存器常驻的资源代价。

## 地址语义

WG ID按所选grid order解码为`batch_id/m_tile/n_tile_group`：

```text
A stream key = (batch_id, m_tile, wave_m)
B stream key = (batch_id, n_tile_group * n_tiles_per_wg + local_n, wave_n)
D stream key = (batch_id, m_tile, n_tile_group * n_tiles_per_wg + local_n, wave)
```

因此同WG的多个N tile复用同一份A-reg；不同M tile WG复用相同B地址，D地址保持tile私有且连续。两种grid order只改变相同key的WG在dispatch中的相邻关系。N tile数不能整除`n_tiles_per_wg`时执行补齐tile，差异计入`executed_tflops`。

## 运行

均衡Hy3形状、full-N、10-buffer：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py bench \
  --physical-device 4 --device 0 \
  --batch 193 --m 1528 --n 4096 --k 192 \
  --bm 64 --bn 512 --bk 64 \
  --waves-m 1 --waves-n 8 --n-tiles-per-wg 8 \
  --waves-per-simd 4 --accumulator-destinations 1 \
  --a-in-reg --grid-order batch_m_n --schedule 2stage_0 --cache-policy temporal \
  --buffer-copies 10 --warmups 40 --samples 50 \
  --launches-per-sample 1 --sample-sync end \
  --json /tmp/ten-buffer-probe-hy3.json
```

对应生产kernel：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/profile-batched-gemm-production.py bench \
  --model hy3 --physical-device 4 --device 0 \
  --buffer-copies 10 --warmups 40 --samples 50 \
  --json /tmp/ten-buffer-production-hy3.json
```

当前Hy3 probe的SQ闭合：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 rocprofv3 \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_WAVES \
  -f csv -d /tmp/ten-buffer-pmc-hy3-probe-sq -o counters -- \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py pmc-run \
  --physical-device 4 --device 0 \
  --batch 193 --m 1528 --n 4096 --k 192 \
  --bm 64 --bn 512 --bk 64 \
  --waves-m 1 --waves-n 8 --n-tiles-per-wg 8 \
  --waves-per-simd 4 --accumulator-destinations 1 \
  --a-in-reg --grid-order batch_m_n --schedule 2stage_0 \
  --cache-policy temporal --buffer-copies 10 --warmups 40 \
  --json /tmp/ten-buffer-pmc-hy3-probe-expected.json
```

随后使用`pmc-analyze --expected /tmp/ten-buffer-pmc-hy3-probe-expected.json --csv /tmp/ten-buffer-pmc-hy3-probe-sq/counters_counter_collection.csv`验证四项SQ计数。L2/HBM只替换`--pmc`计数器组，生产侧把命令替换为`profile-batched-gemm-production.py profile --model hy3 --buffer-copies 10 --warmups 40`。

ATT使用相同profile命令，并增加：

```text
rocprofv3 --att --kernel-trace --kernel-include-regex <target-regex>
  --kernel-iteration-range '[1, [41-41]]'
  --att-target-cu 0 --att-shader-engine-mask 0xf --att-simd-select 0xf
```

该范围会同时解码同名kernel第1次和第41次迭代；数据表使用后一个dispatch。

## 输出判读

- `useful_tflops`按原始`batch*M*N*K`计算；
- `executed_tflops`按向BM/BN/BK补齐后的工作计算；
- `occupancy`记录最终ISA的VGPR/AGPR、HIP active WG/CU及实际waves/SIMD；
- 工具扫描最终ISA，发现任何`ds_*`指令即失败；
- 请求的waves/SIMD若不能由LDS和最终寄存器资源精确实现，候选失败，不静默降低驻留度。

## V0首轮验证

均衡Hy3使用`batch=193,M=1528,N=4096,K=192`；M是最接近生产平均useful rows/expert的整数，向`BM=64`补齐到1536，padding效率99.48%。`BN512/BK64`、8 waves/WG、A在reg时让完整K维A tile常驻，每wave预留48个A VGPR。旧实现按16个C tile保留64 AGPR，最终分配160个vector register/wave，只能驻留2 waves/SIMD。当前实现将MFMA的C输入固定为0，并只保留一个4-AGPR写目标；final ISA为`92 VGPR + 4 AGPR = 96`个vector register/wave、0 scratch。请求4 waves/SIMD时使用32 KiB occupancy-only LDS，HIP确认2 WG/CU；最终ISA没有`ds_*`。

单目标不是靠减少计算量换occupancy：Hy3 full-N配置的`NT/WG=8`，每wave仍有768条MFMA，C输入逐条为`0x0`。当前10-buffer SQ实测为28,459,008 MFMA、3,557,376 VMEM read和2,371,584 VMEM write，全部与公式精确相等。

## 可复现同口径主表

### 参数名与CLI映射

主表不再使用含义不清的`Q`和`n`：

- `B/M/N/K`分别对应`--batch/--m/--n/--k`；
- `BM/BN/BK`分别对应`--bm/--bn/--bk`；
- `WM/WN`分别对应`--waves-m/--waves-n`，`W/WG = WM * WN`为每WG的wave数；
- `W/SIMD`对应`--waves-per-simd`，表示请求的驻留waves/SIMD；
- `NT = ceil(N / BN)`为逻辑N tile总数，`NT/WG`对应`--n-tiles-per-wg`；
- `NG = ceil(NT / (NT/WG))`为N tile group数，不是命令行参数；
- `Dest`对应`--accumulator-destinations`；`A=reg`对应`--a-in-reg`；
- `Sched/Grid/Cache`分别对应`--schedule/--grid-order/--cache-policy`。
- ISA资源中的`V/A`分别表示VGPR/AGPR，LDS按KiB记录。

当前五个probe case全部使用`NT/WG=NT`，因此`NG=1`：一个WG处理该`BM`行块的完整逻辑N。

生产case的`--model`映射如下，`B`固定为32768：

| Case / `--model` | TopK | E | N | K | Quant | Path | active / launched WG |
|---|---:|---:|---:|---:|---|---|---:|
| Hy3 / `hy3` | 9 | 193 | 4096 | 192 | per-tensor | `true8_hy3` | 4,632 / 4,801 |
| H3 / `h3` | 4 | 128 | 6144 | 384 | per-token/per-channel | `physical_n256` | 1,024 / 1,152 |
| Xiaomi / `xiaomi` | 8 | 384 | 6144 | 256 | per-token/per-channel | `physical_n256` | 4,224 / 4,480 |
| Qwen3.5 35B / `q35` | 8 | 256 | 2048 | 512 | per-token/per-channel | `legacy` | 4,096 / 4,352 |
| Qwen3.5 397B / `q397` | 10 | 512 | 4096 | 512 | per-token/per-channel | `legacy` | 5,120 / 5,632 |

### 统一统计协议

主表统一使用gfx942 GPU4、650 W、1800 MHz performance-determinism目标。每个case分配10套地址并连续round-robin：先执行40个warmup dispatch，使每套地址预热4次；随后依次排队50对CUDA event和50个未插桩目标dispatch，使每套地址采样5次，最后只同步一次。每个event区间恰好包含一个down kernel；报告useful TFLOPS的中位数和`[P25--P75]`。

生产侧轮转activation、weight和output，固定routing metadata及scale；probe使用`--a-in-reg`，A只是未访问的1-byte占位，因此实际轮转B和D。两侧都使用`--buffer-copies 10 --warmups 40 --samples 50 --sample-sync end`。主表“差值”只是两个独立运行的中位数之差，不是逐样本配对置信区间。

本文所有当前数据表均来自2026-08-27这一轮10-buffer/full-N重测，没有混入此前single-buffer、partial-N、8 warmup + 48 sample或历史PMC/ATT数值。但“同一轮重测”不表示来自同一个物理dispatch：生产与probe吞吐是两个独立进程；SQ、L2、HBM read和HBM write是四个独立rocprof counter pass；ATT又是独立插桩运行。各profile run都固定为40次round-robin warmup后的同名kernel第41次迭代、buffer index 0。生产侧因初始化kernel更多，全局dispatch ID为72或73；probe为41。比较依据是相同目标kernel迭代、配置和地址轮转位置，不是要求全局dispatch ID相同。

### Probe复现命令

先从配置表选择一行并设置同名shell变量；下面命令显式给出所有影响候选和计时的probe参数：

```bash
CASE=hy3
B=193 M=1528 N=4096 K=192
BM=64 BN=512 BK=64
WM=1 WN=8 NT_WG=8 W_SIMD=4
SCHED=2stage_0 GRID=batch_m_n CACHE=temporal DEST=1

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py bench \
  --physical-device 4 --device 0 \
  --batch "$B" --m "$M" --n "$N" --k "$K" \
  --bm "$BM" --bn "$BN" --bk "$BK" \
  --waves-m "$WM" --waves-n "$WN" \
  --n-tiles-per-wg "$NT_WG" --waves-per-simd "$W_SIMD" \
  --accumulator-destinations "$DEST" --a-in-reg \
  --grid-order "$GRID" --schedule "$SCHED" --cache-policy "$CACHE" \
  --buffer-copies 10 --warmups 40 --samples 50 \
  --launches-per-sample 1 --sample-sync end \
  --json "/tmp/ten-buffer-probe-${CASE}.json"
```

| Case | `B x M x N x K` | `BM x BN x BK` | `WM x WN` (`W/WG`) | `W/SIMD` | `NT/WG` (`NT`; `NG`) | `Sched` | `Grid` | `A` | `Dest` | `Cache` | ISA；LDS |
|---|---|---|---|---:|---|---|---|---|---:|---|---|
| Hy3 | `193x1528x4096x192` | `64x512x64` | `1x8` (8) | 4 | 8 (8; 1) | `2stage_0` | `batch_m_n` | reg | 1 | `temporal` | 92V+4A；32 KiB |
| H3 | `128x1024x6144x384` | `128x256x128` | `2x4` (8) | 2 | 24 (24; 1) | `2stage_0` | `batch_m_n` | reg | 1 | `temporal` | 172V+4A；64 KiB |
| Xiaomi | `384x683x6144x256` | `64x256x128` | `1x4` (4) | 2 | 24 (24; 1) | `2stage_0` | `batch_m_n` | reg | 1 | `temporal` | 140V+4A；32 KiB |
| Qwen3.5 35B | `256x1024x2048x512` | `64x256x128` | `1x4` (4) | 2 | 8 (8; 1) | `2stage_0` | `batch_m_n` | reg | 1 | `temporal` | 204V+4A；32 KiB |
| Qwen3.5 397B | `512x640x4096x512` | `64x256x128` | `1x4` (4) | 2 | 16 (16; 1) | `2stage_0` | `batch_m_n` | reg | 1 | `temporal` | 204V+4A；32 KiB |

### 同协议结果

| Case | 生产kernel | 生产useful TFLOPS `[P25--P75]` | full-N probe useful TFLOPS `[P25--P75]` | probe - 生产 |
|---|---|---:|---:|---:|
| Hy3 | `true8_hy3` | 337.46 `[333.42--346.56]` | 369.77 `[368.70--379.44]` | +32.31T / +9.57% |
| H3 | `physical_n256` | 382.82 `[377.41--386.35]` | 430.43 `[425.30--431.23]` | +47.61T / +12.44% |
| Xiaomi | `physical_n256` | 358.48 `[354.21--360.34]` | 391.63 `[389.91--399.00]` | +33.15T / +9.25% |
| Qwen3.5 35B | `legacy` | 391.20 `[389.12--398.17]` | 487.65 `[486.34--489.27]` | +96.46T / +24.66% |
| Qwen3.5 397B | `legacy` | 412.13 `[410.39--413.58]` | 491.57 `[490.96--492.47]` | +79.44T / +19.28% |

生产shape为真实`B/TopK/E/N/K/quant`，probe把总useful rows均衡为`batch*M`；这是模型几何差异，不是统计口径差异。生产和probe都只计单个down dispatch，不包含sorting、gateup、reduction或完整MoE调用。原始记录为`/tmp/ten-buffer-production-*.json`和`/tmp/ten-buffer-probe-*.json`。

### 动态SQ计数

每侧均使用10套地址、40次round-robin warmup，再profile下一个目标dispatch。列顺序为`SQ_WAVES / SQ_INSTS_MFMA / SQ_INSTS_VMEM_RD / SQ_INSTS_VMEM_WR`。

| Case | 生产 | full-N probe |
|---|---|---|
| Hy3 | 38,408 / 28,459,008 / 3,964,992 / 2,371,584 | 37,056 / 28,459,008 / 3,557,376 / 2,371,584 |
| H3 | 9,216 / 37,748,736 / 5,775,360 / 1,572,864 | 8,192 / 37,748,736 / 4,718,592 / 1,572,864 |
| Xiaomi | 17,920 / 51,904,512 / 7,383,552 / 3,244,032 | 16,896 / 51,904,512 / 6,488,064 / 3,244,032 |
| Qwen3.5 35B | 17,408 / 33,554,432 / 5,177,344 / 1,048,576 | 16,384 / 33,554,432 / 4,194,304 / 1,048,576 |
| Qwen3.5 397B | 22,528 / 83,886,080 / 12,369,920 / 2,621,440 | 20,480 / 83,886,080 / 10,485,760 / 2,621,440 |

五个probe的四项SQ计数均与`derived`公式精确相等。生产与probe的MFMA和write总量逐项相同；生产包含真实load、scale、metadata及guard路径，因此waves和VMEM read更多。原始CSV为`/tmp/ten-buffer-pmc-<case>-<side>-sq/`，汇总为`/tmp/ten-buffer-pmc-summary.json`。

### PMC内存计数

每格按`生产 / probe`记录。L2命中率为`TCC_HIT/(TCC_HIT+TCC_MISS)`；HBM读写分别由EA read request和64B write request换算，单位GiB。

| Case | L2 hit | HBM read GiB | HBM write GiB | TCP read requests |
|---|---:|---:|---:|---:|
| Hy3 | 63.90% / 63.62% | 0.624 / 0.617 | 2.262 / 2.262 | 29,087,747 / 28,459,008 |
| H3 | 72.56% / 64.63% | 0.392 / 1.137 | 1.500 / 1.500 | 32,668,472 / 37,386,472 |
| Xiaomi | 55.49% / 52.95% | 2.553 / 2.730 | 3.094 / 3.094 | 54,045,137 / 51,904,512 |
| Qwen3.5 35B | 65.22% / 66.66% | 1.154 / 1.003 | 1.000 / 1.000 | 35,211,674 / 33,554,432 |
| Qwen3.5 397B | 52.35% / 54.79% | 4.698 / 4.232 | 2.500 / 2.500 | 86,297,707 / 83,886,080 |

四个counter pass都profile同一轮转位置；40个pass无错误日志。HBM write在五个case中生产/probe逐字节相同。

### ATT稳态

每侧在40次round-robin warmup后捕获同名kernel第41次迭代，target CU0、SE/SIMD mask=`0xf/0xf`。每个physical slot剔除首尾各一条wave；wave时长为`end-begin`；MFMA issue为`first_attempt+stall`；MFMA union按16-cycle执行窗口统计。十份trace均完整。

| Case | MFMA/wave | 生产wave cycles `[P25--P75]` | probe wave cycles `[P25--P75]` | cycles/MFMA 生产 / probe | MFMA union 生产 / probe |
|---|---:|---:|---:|---:|---:|
| Hy3 | 768 | 74,696 `[69,420--79,826]` | 66,746 `[59,826--72,316]` | 97.260 / 86.909 | 58.34% / 65.78% |
| H3 | 4,608 | 249,230 `[189,148--251,874]` | 173,272 `[149,661--194,586]` | 54.086 / 37.602 | 66.00% / 78.29% |
| Xiaomi | 3,072 | 145,358 `[137,225--150,888]` | 136,562 `[132,706--141,233]` | 47.317 / 44.454 | 66.79% / 71.01% |
| Qwen3.5 35B | 2,048 | 97,528 `[92,343--100,389]` | 75,120 `[73,374--77,139]` | 47.621 / 36.680 | 65.78% / 84.74% |
| Qwen3.5 397B | 4,096 | 181,752 `[176,923--190,108]` | 151,016 `[148,988--153,268]` | 44.373 / 36.869 | 69.04% / 85.69% |

当前两侧每个case的MFMA/wave相同，可以直接比较wave时长与cycles/MFMA。原始trace为`/tmp/ten-buffer-att-<case>-<side>/`，统一汇总为`/tmp/ten-buffer-att-steady.json`。

### Hy3差距与CU负载

10-buffer full-N下，Hy3 production/probe为337.46/369.77T，probe高9.57%。双方每wave都执行768条MFMA，ATT中probe的cycles/MFMA为86.909，相对生产97.260低10.64%；MFMA union从58.34%提高到65.78%，高7.44个百分点。墙钟、单位MFMA时长和physical-SIMD union三者方向一致，原先固定地址下“生产更快”的反转不再存在。

PMC不能把该差距解释成普通cache流量优势：生产/probe的L2 hit为63.90%/63.62%，HBM read为0.624/0.617 GiB，几乎相同。生产比probe多407,616条VMEM read和1,352条wave，来自真实A/weight、scale、metadata与guard路径；当前证据将差距定位为这些正确性工作及其引入的地址/调度背压，而不是HBM字节量。

#### Padding、guard与CU不均衡

Hy3共有`32768 * 9 = 294912`个useful row。轮转expert映射使每个expert有1528或1529行，均向M64补齐为1536行：

```text
active WG = 193 * 24 = 4632
executed rows = 4632 * 64 = 296448
row padding efficiency = 294912 / 296448 = 99.4819%
```

因此彻底消除行padding也最多只恢复0.52%，小于主表9.57%的差距。Balanced skeleton使用`193 * 1528 = 294904`个逻辑row，padding效率99.4792%，两侧几乎完全抵消。

生产metadata grid为4801个WG，其中4632个进入计算，169个WG guard退出；probe恰好为4632个WG。Fresh ATT在目标CU捕获208条无MFMA early-exit wave和1848条active wave；guard不能按完整GEMM WG计入负载。

生产有4632个active WG；当前full-N probe的`NT/WG=8`，同样有4632个WG。请求4 waves/SIMD时每轮最多160 WG计算：

- 生产最后量子为152/160个WG slot；
- full-N probe最后量子也为152/160个WG slot；
- 两者理想dispatch-tail效率完全相同，均为99.8276%，损失0.1724%。没有CU在整个dispatch中无任务，只有末轮部分CU先空闲。

当前两侧的active WG总数、每WG wave数和理想dispatch tail完全相同，所以理想任务量不解释差距。优先方向是缩减生产额外VMEM/metadata工作，或改善其与MFMA的重叠；padding、理想CU tail和HBM带宽不是优先项。

### 内存pattern复核

host self-test按实际WG线性顺序枚举每条wave的`(stream,k_tile,operation)`。对`NT/WG=1/2/4/8`及两种grid order，A/B/D地址多重集逐项相同：相同地址、相同重复次数，不重不漏。另有`NT/WG=24, NT=24, NG=1`的full-N几何检查。分组只改变时间顺序和WG边界。

#### 当前10-buffer复核

五个case的生产/probe MFMA和HBM write逐项完全相同，说明probe覆盖了目标矩阵计算量和BF16输出字节。读侧不是统一偏向：H3 probe的HBM read是生产的2.90倍，Xiaomi高6.9%；Hy3近似相同；Qwen3.5 35B/397B分别低13.1%/9.9%。因此不能用单一“probe cache更理想”解释五个case。

尽管读侧方向不同，五个case的probe都更快，并且ATT全部显示probe cycles/MFMA更低、MFMA union更高：cycles/MFMA改善10.64%/30.48%/6.05%/22.98%/16.91%，union提高7.44/12.29/4.22/18.95/16.66个百分点。共同差距来自probe省略VMEM到MFMA RAW、LDS搬运、scale、activation和真实epilogue后的core co-issue ceiling；PMC用于约束内存环境，不把ceiling解释成正确kernel可直接达到的下界。

TODO见[`batched-gemm-core-ceiling-todo.md`](batched-gemm-core-ceiling-todo.md)。