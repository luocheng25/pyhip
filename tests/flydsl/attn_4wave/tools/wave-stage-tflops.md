# Wave-stage TFLOPS探针

[`probe-wave-stage-tflops.py`](probe-wave-stage-tflops.py)用AsmJIT构造可控的VMEM/FP8-MFMA负载。模型只表达硬件层变量，不包含应用tile、routing、scale、CShuffle、LDS数据布局或后处理。

## 调度模型

支持四种schedule：

- `2stage_0`（默认）：stage 0批量发出`V0`条VMEM并等待/消费当前批；stage 1连续执行`C0`条`v_mfma_f32_16x16x32_fp8_fp8`。两个stage之间不插入priority或barrier指令。
- `2stage_prio`：与`2stage_0`工作量相同；进入stage 1执行`s_setprio 1`，离开stage 1执行`s_setprio 0`。
- `2stage_barrier`：仅允许8-wave及以上WG。WG后一半wave在首个预取stage前被阻塞；每个stage结束执行`s_barrier`，并在stage 1使用`setprio(1/0)`，使WG内上下半组严格反相。
- `interleave`：保留双VGPR bank；逐条发下一批VMEM，立即`vmcnt(V0)`并消费上一批对应read，再执行均分的一组MFMA。

所有schedule都采用有限流水：prologue先发第0批；最后一轮只`vmcnt(0)`、消费并计算，不再预取没有对应compute的下一批。因此`rounds`轮计算只发`rounds`批VMEM。

每条MFMA按`2 * 16 * 16 * 32 = 16384 FLOPs`计算。参数为：

- `W` / `--waves-per-workgroup`：每WG的wave数；
- `C0`：每轮、每wave的MFMA数；
- `V0`与`--vmem-op`：每轮VMEM数、read/write比例和顺序；
- `--schedule`：上述四种stage边界策略；
- `--cache-policy`：`temporal`（默认）或`non_temporal`；
- `--waves-per-simd`：请求的驻留waves/SIMD；
- `--pattern`：`private/simd/stage/workgroup/cu`地址共享范围。

## Round与workgroup

- **1 round**：每wave处理一批`V0` VMEM并执行`C0` MFMA。
- **1 workgroup**：一个实际launch WG，不是wave或round。

一个`W`-wave workgroup的静态工作量为：

```text
MFMA instructions/workgroup = W * rounds * C0
FLOPs/workgroup = W * rounds * C0 * 16384
VMEM batches/wave = rounds
VMEM instructions/workgroup = W * rounds * V0
logical VMEM bytes/workgroup = W * rounds * V0 * 1024
```

例如`rounds=128,W=4,C0=64,V0=8`时，每WG包含32,768条MFMA、536,870,912 FLOPs、4,096条wave-level VMEM和4,194,304逻辑VMEM bytes。

`--workgroups-per-cu 8`表示整次sample给每CU安排8个WG。未指定`--single-dispatch`时，工具按当前驻留WG/CU拆成多个residency-sized dispatch；TFLOPS分子包含所有WG。`--single-dispatch --workgroups N`可直接指定一次dispatch的精确WG数，用于复现真实case的active grid。

## 驻留与LDS

请求`Q = --waves-per-simd`时，每CU目标驻留`4Q`条wave，即`4Q/W`个W-wave WG。静态LDS为：

```text
LDS/WG = floor(64 KiB / (4Q/W), 256 B)
```

常用组合：

| waves/WG | threads/WG | Q | LDS/WG | 目标WG/CU |
|---:|---:|---:|---:|---:|
| 4 | 256 | 1 / 2 / 4 | 64 / 32 / 16 KiB | 1 / 2 / 4 |
| 8 | 512 | 2 / 4 | 64 / 32 KiB | 1 / 2 |
| 16 | 1024 | 4 | 64 KiB | 1 |

`4Q`必须能被W整除；例如8-wave WG本身已在每个SIMD放2条wave，所以`W=8,Q=1`会被拒绝。工具读取最终ISA的VGPR/AGPR/numbered-SGPR，并调用HIP module occupancy API。若编译资源只能达到更低Q，会发`RuntimeWarning`，JSON同时记录requested/actual Q，再以实际Q做生命周期驻留验证。

GPU4实测W4的Q1/Q2/Q4分别得到64/32/16 KiB LDS、1/2/4 WG/CU和1/2/4 waves/SIMD。true8等价case请求Q8时，`108 VGPR + 16 AGPR`只允许Q4，告警与运行时验证一致。

## 内存pattern

- `private`：每wave独立地址流；
- `simd`：同一物理SIMD的waves共享地址流；
- `stage`：同一CU内两个stage组各共享一条流；
- `workgroup`：同WG共享；
- `cu`：同物理CU共享。

物理pattern通过`XCC/SE/CU/SIMD/slot`硬件ID计算，不依赖block调度顺序。每个地址epoch内stream连续前进且不回绕。每条VMEM指令由64 lane各访问16 B，逻辑请求量为1024 B。

`--vmem-op`支持`read`、`write`、交替`mixed`，以及通用的`readN_writeM`；后者每批先严格发N条read，再发M条write，例如`read3_write2`、`read12_write8`和`read29_write8`。

## 运行

单case：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-wave-stage-tflops.py bench \
  --physical-device 4 --device 0 \
  --waves-per-workgroup 8 --waves-per-simd 4 \
  --c0 64 --v0 12 --schedule 2stage_0 \
  --pattern simd --vmem-op read \
  --json /tmp/wave-stage-w8-c64-v12.json
```

典型矩阵：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-wave-stage-tflops.py sweep \
  --physical-device 4 --device 0 --waves-per-simd 4 \
  --schedule 2stage_0 --cache-policy temporal \
  --json /tmp/wave-stage-typical.json
```

TFLOPS为：

```text
workgroups * waves_per_workgroup * rounds * C0 * 16384
/ kernel_seconds / 1e12
```

每个计时dispatch在kernel内读取`S_MEMTIME`和固定100 MHz的`S_MEMREALTIME`：

```text
effective_sclk_mhz = delta(s_memtime) * 100 / delta(s_memrealtime)
theoretical_fp8_tflops = 16384 / 16 * 80 CU * 4 SIMD/CU * effective_sclk_mhz / 1e6
end_to_end_attainment = measured_tflops / theoretical_fp8_tflops
```

工具检查GPU空闲、650 W power cap和驻留，成功或异常均恢复GPU状态。每个独立计时dispatch必须至少100 us。

## 新默认典型矩阵

以下数据于2026-08-25重新采集：gfx942 GPU4、1800 MHz performance determinism、650 W、`2stage_0 + temporal + Q4`、128 rounds、8 workgroups/CU、12 samples。表中为全sample中位值。

| waves/WG | C0 | V0 | pattern | TFLOPS | SCLK MHz | 理论TFLOPS | 达成率 | 最短dispatch us |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 64 | 8 | private | 483.80 | 1812.4 | 593.89 | 81.48% | 320.71 |
| 4 | 64 | 8 | simd | **513.78** | 1811.2 | 593.50 | 86.55% | 302.08 |
| 4 | 64 | 12 | private | 340.84 | 1814.4 | 594.56 | 57.32% | 468.01 |
| 4 | 64 | 12 | simd | 333.36 | 1813.7 | 594.30 | 56.10% | 473.74 |
| 4 | 32 | 12 | private | 171.30 | 1815.2 | 594.79 | 28.81% | 466.20 |
| 4 | 32 | 12 | simd | 165.99 | 1815.3 | 594.84 | 27.91% | 472.48 |
| 8 | 64 | 8 | private | 464.19 | 1645.7 | 539.28 | 88.18% | 322.36 |
| 8 | 64 | 8 | simd | 464.95 | 1533.2 | 502.40 | **91.05%** | 302.32 |
| 8 | 64 | 12 | private | 342.20 | 1538.3 | 504.09 | 68.28% | 478.96 |
| 8 | 64 | 12 | simd | 347.12 | 1646.6 | 539.56 | 64.14% | 464.60 |
| 8 | 32 | 12 | private | 172.83 | 1712.1 | 561.01 | 30.76% | 474.34 |
| 8 | 32 | 12 | simd | 170.25 | 1604.2 | 525.66 | 32.63% | 478.45 |
| 16 | 64 | 8 | private | 482.68 | 1705.4 | 558.82 | 86.43% | 324.83 |
| 16 | 64 | 8 | simd | 507.20 | 1727.8 | 566.16 | 89.35% | 302.12 |
| 16 | 64 | 12 | private | 342.09 | 1624.6 | 532.36 | 64.21% | 485.64 |
| 16 | 64 | 12 | simd | 351.66 | 1639.4 | 537.19 | 65.22% | 465.17 |
| 16 | 32 | 12 | private | 170.91 | 1641.4 | 537.86 | 31.72% | 483.19 |
| 16 | 32 | 12 | simd | 172.57 | 1648.0 | 540.03 | 31.91% | 473.54 |

完整JSON：`/tmp/wave-stage-typical-new.json`。

### 全schedule sweep

`interleave`和`2stage_prio`也完成同一18-case矩阵；`2stage_barrier`因只支持W8及以上，完成W8/W16的12-case矩阵。所有run均为temporal、Q4、128 rounds、8 workgroups/CU、12 samples，无资源告警或运行错误。

| schedule | cases | 最高绝对吞吐 | 最高频率归一化达成率 |
|---|---:|---|---|
| `2stage_0` | 18 | W4 64/8 simd：513.78T | W8 64/8 simd：91.05% |
| `interleave` | 18 | W4 64/8 simd：517.51T | W16 64/8 simd：90.83% |
| `2stage_prio` | 18 | W4 64/8 simd：514.64T | W8 64/8 simd：96.43% |
| `2stage_barrier` | 12 | W16 64/12 simd：524.22T | W8 64/8 private：92.15% |

### 通用参数case预测

下表汇总18个通用`W/C0/V0/pattern`组合和四种schedule，共66个适用预测，不代表具体模型case。数值均为TFLOPS中位值；`2stage_barrier`不支持W4，因此对应6项标为不适用。

| W | C0 | V0 | pattern | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | 最高预测 |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 64 | 8 | private | 483.80 | 483.65 | 不适用 | 480.70 | `2stage_0` 483.80 |
| 4 | 64 | 8 | simd | 513.78 | 514.64 | 不适用 | 517.51 | `interleave` 517.51 |
| 4 | 64 | 12 | private | 340.84 | 341.65 | 不适用 | 340.34 | `2stage_prio` 341.65 |
| 4 | 64 | 12 | simd | 333.36 | 333.34 | 不适用 | 351.91 | `interleave` 351.91 |
| 4 | 32 | 12 | private | 171.30 | 171.05 | 不适用 | 170.94 | `2stage_0` 171.30 |
| 4 | 32 | 12 | simd | 165.99 | 165.36 | 不适用 | 164.83 | `2stage_0` 165.99 |
| 8 | 64 | 8 | private | 464.19 | 454.89 | 468.95 | 465.99 | `2stage_barrier` 468.95 |
| 8 | 64 | 8 | simd | 464.95 | 462.70 | 507.85 | 504.58 | `2stage_barrier` 507.85 |
| 8 | 64 | 12 | private | 342.20 | 342.64 | 349.40 | 342.00 | `2stage_barrier` 349.40 |
| 8 | 64 | 12 | simd | 347.12 | 347.36 | 464.96 | 360.26 | `2stage_barrier` 464.96 |
| 8 | 32 | 12 | private | 172.83 | 172.98 | 174.98 | 171.87 | `2stage_barrier` 174.98 |
| 8 | 32 | 12 | simd | 170.25 | 169.77 | 231.62 | 169.62 | `2stage_barrier` 231.62 |
| 16 | 64 | 8 | private | 482.68 | 482.22 | 501.11 | 484.82 | `2stage_barrier` 501.11 |
| 16 | 64 | 8 | simd | 507.20 | 503.97 | 511.31 | 506.83 | `2stage_barrier` 511.31 |
| 16 | 64 | 12 | private | 342.09 | 342.39 | 352.84 | 343.82 | `2stage_barrier` 352.84 |
| 16 | 64 | 12 | simd | 351.66 | 352.27 | 524.22 | 366.17 | `2stage_barrier` 524.22 |
| 16 | 32 | 12 | private | 170.91 | 171.34 | 177.29 | 171.38 | `2stage_barrier` 177.29 |
| 16 | 32 | 12 | simd | 172.57 | 172.28 | 295.09 | 171.89 | `2stage_barrier` 295.09 |

完整结果：

- `/tmp/wave-stage-typical-new.json`
- `/tmp/wave-stage-typical-interleave-new.json`
- `/tmp/wave-stage-typical-2stage_prio-new.json`
- `/tmp/wave-stage-typical-2stage_barrier-new.json`

## MoE测试case预测与实测

以下五项来自[`test_moe.py`](../../../contrib/moe/test_moe.py)中的模型配置，batch固定为32768。先运行当前selector对应的真实down kernel，再用SQ PMC按active wave归一化，提取K-loop steady core的MFMA、read和write；固定的metadata、scale、分支与epilogue指令不并入`C0/V0`。探针统一使用temporal cache、SIMD共享、finite pipeline和真实active WG数。

`waves/WG`和`waves/SIMD`对probe与当前实测使用同一配置。它们已分别从rocprofv3的workgroup size及生产ISA VGPR/AGPR、LDS资源核对，并由probe最终ISA、HIP occupancy和运行时slot驻留再次验证：Hy3为`8/4`，H3为`8/2`，Xiaomi及两项Qwen为`4/2`。

预测与实测均为48个sample的useful TFLOPS中位值`[P25--P75]`。当前实测额外使用8次warmup，并固定GPU4为1800 MHz performance determinism。Hy3与Xiaomi的probe raw TFLOPS分别乘`294912/296448`和`32/33`以排除expert padding；其余三项raw即useful。最佳预测用粗体标出，误差定义为`最佳预测/当前实测-1`；W4不适用`2stage_barrier`。

| case | 当前path | waves/WG | waves/SIMD | ISA VGPR+AGPR | steady core x rounds | active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | 当前实测 | 最佳预测误差（差值） |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Hy3 | `true8_hy3` | 8 | 4 | 128+0 | (96 MFMA + 12R8W) x 8 | 4,632 | 260.50 [258.96--262.51] | 260.41 [258.92--262.41] | **358.44 [351.99--371.68]** | 267.40 [264.56--269.35] | 413.53 [411.57--415.76] | -13.32% (-55.09T) |
| H3 | `physical_n256` | 8 | 2 | 128+128 | (192 MFMA + 29R8W) x 24 | 1,024 | 319.27 [315.89--322.88] | 320.35 [316.00--322.38] | **368.57 [362.59--376.92]** | 332.20 [325.91--335.26] | 375.80 [375.37--376.36] | **-1.92% (-7.23T)** |
| Xiaomi | `physical_n256` | 4 | 2 | 80+128 | (128 MFMA + 18R8W) x 24 | 4,224 | 275.38 [273.45--325.41] | 278.07 [275.29--329.93] | 不适用 | **284.75 [282.83--285.79]** | 352.60 [351.82--353.06] | -19.24% (-67.86T) |
| Qwen3.5 35B | `legacy` | 4 | 2 | 108+132 | (64 MFMA + 9R2W) x 32 | 4,096 | **358.91 [347.98--405.60]** | 348.32 [345.12--350.96] | 不适用 | 353.39 [351.24--357.63] | 388.36 [386.96--389.22] | -7.58% (-29.45T) |
| Qwen3.5 397B | `legacy` | 4 | 2 | 108+132 | (64 MFMA + 9R2W) x 64 | 5,120 | 360.68 [358.15--362.94] | 362.68 [355.60--364.24] | 不适用 | **366.90 [362.80--368.17]** | 401.33 [400.35--402.09] | -8.58% (-34.44T) |

方向上，Hy3和H3的probe都指向`2stage_barrier`，其中H3已在2%内闭合；Hy3仍低估13.32%。Xiaomi的`interleave`预测最高但仍低估19.24%，应优先按wave角色拆分VMEM阶段。Qwen3.5 397B的`interleave`只比`2stage_0`高1.72%；Qwen3.5 35B各模式IQR重叠且存在频率双平台，暂无清晰schedule胜者。

H3已在2%内闭合；其余case全部低估，说明当前probe不是steady-core上限。主要偏差来源是把PMC总量平均成每条wave相同的`C0/V0`批次，未表达生产kernel的异构wave角色、分级`vmcnt`、LDS搬运与后处理重叠。下一步应先针对Xiaomi和Hy3按wave角色拆分VMEM阶段，再处理两项Qwen；不能使用一个跨case固定修正系数。

预测结果为`/tmp/moe-predict-<case>-<schedule>-final.json`，当前实测为`/tmp/moe-current-actual-final.jsonl`。

## 动态工作与PMC闭合

四种schedule使用同一W8/Q4、`C0=24,V0=5,read3_write2,rounds=2`负载采集SQ PMC，结果逐项相同：

| 指标 | 每个schedule |
|---|---:|
| waves | 1,280 |
| `SQ_INSTS_MFMA` | 61,440 |
| `SQ_INSTS_VMEM_RD` | 7,680 |
| `SQ_INSTS_VMEM_WR` | 5,120 |
| VMEM batches/wave | 2 |

因此stage模式只改变边界控制，不改变动态MFMA/VMEM工作量。

Private有限流水验证使用W16/Q4、`C0=64,V0=8,rounds=128`、80 WG、1280 waves：

```text
read instructions = 80 * 16 * 128 * 8 = 1,310,720
logical read bytes = 1,342,177,280
```

| 层级 | PMC结果 | 闭合 |
|---|---:|---:|
| `SQ_WAVES` | 1,280 | 100% |
| `SQ_INSTS_MFMA` | 10,485,760 | 100% |
| `SQ_INSTS_VMEM_RD` | 1,310,720 | 100% |
| `TCP_TCC_READ_REQ_sum` | 10,485,760 | 8 requests/VMEM |
| L2 hit rate | 0.02921% | 几乎全下沉HBM |
| HBM read bytes | 1,342,203,584 | 理论+0.001960% |

完整结果：`/tmp/wave-stage-private-pmc-new-validation.json`。

## Legacy校准

legacy N64参考核心为每wave每轮`24 MFMA + 3 read + 2 write`，ATT峰值4 waves/SIMD，cache为temporal，SIMD内存在地址复用。新配置为：

```text
--waves-per-workgroup 4 --waves-per-simd 4
--c0 24 --v0 5 --schedule interleave
--vmem-op read3_write2 --cache-policy temporal --pattern simd
--rounds 64 --workgroups-per-cu 58 --single-dispatch
```

真实physical4使用同源control/candidate、10 buffers、12轮down-only ABBA重新采集，得到`298.33T`；self-comparison ratio中位为`0.99868`，Q1--Q3为`0.98653--1.01696`。有限epilogue probe得到`286.34T`，误差为`-4.02%`，仍在5%门槛内。

结果：`/tmp/hy3-current-legacy-down-tflops-new.json`和`/tmp/wave-stage-legacy-interleave-new.json`。

## True8调度对照

true8等价配置固定为8 waves/WG、Q4、SIMD共享、temporal、每wave每轮`96 MFMA + 12 read + 8 write`、8 rounds、60 workgroups/CU、single dispatch。所有模式每wave均执行8批VMEM和768 MFMA。

真实single_n512同样用10 buffers、12轮down-only ABBA重新采集，得到`336.25T`；self-comparison ratio中位为`1.00184`，Q1--Q3为`0.98384--1.01703`。各probe统一排除首sample后：

| schedule/cache | TFLOPS | IQR | SCLK MHz | 相对真实 |
|---|---:|---:|---:|---:|
| `interleave` / non-temporal | 222.52 | 219.28--223.09 | 1810.2 | -33.82% |
| `interleave` / temporal | 270.48 | 267.68--274.39 | 1782.2 | -19.56% |
| `2stage_0` / temporal | 260.64 | 258.79--263.12 | 1808.7 | -22.48% |
| `2stage_prio` / temporal | 261.80 | 261.26--265.05 | 1539.9 | -22.14% |
| `2stage_barrier` / temporal | **364.41** | 351.19--374.08 | 1438.8 | **+8.38%** |

结果说明：在所有模式都采用有限epilogue、动态工作完全相同后，严格反相本身仍带来最大的全GPU MFMA供给改善；但当前简单barrier模型高估真实true8约8.4%，超过5%目标。真实kernel的分级`vmcnt(7..0)`、LDS读取、VALU后处理和延迟store会消耗这部分理想反相收益。

`2stage_barrier`的ATT采集观察到16个物理SIMD、每SIMD四个slot，WG内配对为`(0,1)`和`(2,3)`，0 residency failure、0 incomplete trace。控制流在首批预取前阻塞后一半wave，每个stage末尾barrier，并在排空时补齐低半组；这构成严格反相。当前trace decoder对AsmJIT展开体将循环PC index折叠为0，因此不报告伪精确的marker-overlap百分比。

结果文件：

- `/tmp/wave-stage-true8-interleave-new.json`
- `/tmp/wave-stage-true8-interleave-nontemporal-new.json`
- `/tmp/wave-stage-true8-2stage_0-new.json`
- `/tmp/wave-stage-true8-2stage_prio-new.json`
- `/tmp/wave-stage-true8-2stage_barrier-new-final.json`
- `/tmp/wave-stage-2stage-barrier-att-final`
- `/tmp/hy3-current-true8-down-tflops-new.json`

## 使用边界

该探针仍是简单VMEM/MFMA模型。推广到其他kernel时，应重新提取：

1. MFMA/VMEM比例与read/write顺序；
2. cache policy和地址共享范围；
3. 实际waves/SIMD与编译资源；
4. stage边界策略；
5. 是否存在会暴露在关键路径上的LDS、VALU或长依赖链。

Legacy误差为-4.02%，true8的理想barrier模型误差为+8.38%；不能把任一误差当作跨应用固定修正系数。
