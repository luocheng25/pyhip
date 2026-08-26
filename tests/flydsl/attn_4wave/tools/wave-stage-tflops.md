# Wave-stage TFLOPS探针

[`probe-wave-stage-tflops.py`](probe-wave-stage-tflops.py)用AsmJIT构造可控的VMEM/FP8-MFMA负载。模型只表达硬件层变量，不包含应用tile、routing、scale、CShuffle、LDS数据布局或后处理。

## 当前结论

2026-08-26使用当前代码完整重测后，五个生产case的拓扑匹配预测如下：

| case | 当前实测TFLOPS | 指定配置（仅schedule匹配） | 误差 |
|---|---:|---:|---:|
| Hy3 | 412.82 [410.80--413.91] | 238.60 [238.15--238.85] | -42.20% |
| H3 | 381.34 [380.68--381.66] | 273.00 [272.69--273.33] | -28.41% |
| Xiaomi | 352.70 [351.79--353.61] | 247.88 [247.74--248.18] | -29.72% |
| Qwen3.5 35B | 389.32 [388.11--390.03] | 302.07 [301.49--303.75] | -22.41% |
| Qwen3.5 397B | 400.65 [399.59--401.68] | 305.83 [305.63--306.15] | -23.67% |

用户指定的新配置全部低估生产实测22%--42%。这里“匹配”只表示选择了最接近生产控制流的schedule；指定的`private`地址范围和均匀VMEM分布并不等于生产kernel的缓存复用与写回位置。模型没有case分支或逐case修正系数，误差仅用于评价，不反馈到预测公式。

## 测量口径

本轮所有当前结论来自同一次重测批次：gfx942 GPU4、80 CU、ROCm 7.2.3、650 W power cap、1800 MHz performance-determinism目标、PTL `VECTOR,F8`。1800 MHz是请求目标，不是实测频率。

| 数据集 | 数量 | warmup / sample | 计时或采集方法 |
|---|---:|---:|---|
| 生产MoE | 5 | 8 / 48 | CUDA event，单dispatch/sample |
| 通用参数矩阵 | 84 | 3 / 12 | CUDA event，无instrumentation |
| MoE probe矩阵 | 135 | 8 / 48 | CUDA event，无instrumentation |
| schedule SQ闭合 | 5 | 1 / 1 | rocprofv3，读取最后dispatch |
| exact-grid guard闭合 | 1 | 1 / 1 | rocprofv3，读取最后dispatch |
| private SQ/L2/HBM闭合 | 6 counter pass | 1 / 1 | 2组exact-grid，rocprofv3读取最后dispatch |
| 频率/L2生产-probe对照 | 8 profile | 8 / 1 | rocprofv3，读取第9个dispatch |

采集时GPU4计算利用率为0%；存在容器外idle context占用显存，因此probe使用`--max-initial-vram-percent 90`，但每个测试的实际workspace均能完整驻留。正式吞吐kernel只使用CUDA event；`S_MEMTIME/S_MEMREALTIME`仅存在于独立短驻留验证kernel，不进入TFLOPS计时。

## 模型定义

### 调度

支持五种schedule：

- `2stage_0`（默认）：stage 0批量发出`V0`条VMEM并等待/消费当前批；stage 1连续执行`C0`条`v_mfma_f32_16x16x32_fp8_fp8`。两个stage之间不插入priority或barrier指令。
- `2stage_prio`：与`2stage_0`工作量相同；进入stage 1执行`s_setprio 1`，离开stage 1执行`s_setprio 0`。
- `2stage_barrier`：仅允许8-wave及以上WG。WG后一半wave在首个预取stage前被阻塞；每个stage结束执行`s_barrier`，并在stage 1使用`setprio(1/0)`，使WG内上下半组严格反相。
- `interleave`：保留双VGPR bank；逐条发下一批VMEM，立即`vmcnt(V0)`并消费上一批对应read，再执行均分的一组MFMA。
- `2stage_round_barrier`：与`2stage_0`工作量相同；有限流水前后各执行两个WG barrier，每个完整VMEM/MFMA round末尾执行一个barrier。它用于量化“全WG逐轮同步”的控制流成本，例如所有wave必须完成本轮才能复用LDS或进入下一输出轮；它不做WG内反相，也不是为了拟合某个case而增加的经验延迟。

所有schedule都采用有限流水：prologue先发第0批；最后一轮只`vmcnt(0)`、消费并计算，不再预取没有对应compute的下一批。因此`rounds`轮计算只发`rounds`批VMEM。

每条MFMA按`2 * 16 * 16 * 32 = 16384 FLOPs`计算。参数为：

- `W` / `--waves-per-workgroup`：每WG的wave数；
- `C0`：每轮、每wave的MFMA数；
- `V0`与`--vmem-op`：每轮VMEM数、read/write比例和顺序；
- `--schedule`：上述五种stage边界策略；
- `--cache-policy`：`temporal`（默认）或`non_temporal`；
- `--waves-per-simd`：请求的驻留waves/SIMD；
- `--pattern`：`private/simd/stage/workgroup/cu`地址共享范围。

### Round与workgroup

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

`--workgroups-per-cu 8`表示整次sample给每CU安排8个WG。未指定`--single-dispatch`时，工具按当前驻留WG/CU拆成多个residency-sized dispatch；TFLOPS分子包含所有WG。`--single-dispatch --workgroups N`指定总launch WG数；再加`--active-workgroups A`时，仅前A个WG执行负载，尾部WG统一早退，FLOPs和VMEM分子只计active WG。

### 驻留与LDS

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

GPU4实测W4的Q1/Q2/Q4分别得到64/32/16 KiB LDS、1/2/4 WG/CU和1/2/4 waves/SIMD。MoE矩阵同时记录请求Q与最终ISA/HIP occupancy允许的实际Q；H3请求Q4时实际为Q2，Xiaomi的W4/Q4实际为Q3、W8/Q4实际为Q2。

### 内存pattern

- `private`：每wave独立地址流；descriptor base使用64位`stream_id * bytes_per_stream`平移，因此单个exact-grid可覆盖超过4 GiB的总地址空间，同时每条wave的descriptor内offset保持32位；
- `simd`：同一物理SIMD的waves共享地址流；
- `stage`：同一CU内两个stage组各共享一条流；
- `workgroup`：同WG共享；
- `cu`：同物理CU共享。

物理pattern通过`XCC/SE/CU/SIMD/slot`硬件ID计算，不依赖block调度顺序。每个地址epoch内stream连续前进且不回绕。每条VMEM指令由64 lane各访问16 B，逻辑请求量为1024 B。

`--vmem-op`支持`read`、`write`、交替`mixed`，以及通用的`readN_writeM`；后者每批先严格发N条read，再发M条write，例如`read3_write2`、`read12_write8`和`read29_write8`。

TFLOPS为：

```text
workgroups * waves_per_workgroup * rounds * C0 * 16384
/ kernel_seconds / 1e12
```

正式吞吐计时使用CUDA event包围**无per-wave metadata**的kernel。短驻留验证kernel仍用`S_MEMTIME/S_MEMREALTIME`记录所有wave的生命周期，但不参与吞吐计时。这样避免每条wave四次时钟读取、48 B metadata写回和`lgkmcnt(0)`对被测kernel本身的扰动。

1800 MHz是performance-determinism目标，不再冒充实测频率：

```text
sclk_target_mhz = 1800
target_theoretical_fp8_tflops = 16384 / 16 * 80 CU * 4 SIMD/CU * 1800 / 1e6
target_theoretical_attainment = measured_tflops / target_theoretical_fp8_tflops
```

工具检查GPU计算空闲、650 W power cap和驻留，成功或异常均恢复GPU状态。每个独立计时dispatch必须至少100 us。

### 复现入口

单个probe使用：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
	/tmp/pyhip-flydsl024/bin/python \
	tests/flydsl/attn_4wave/tools/probe-wave-stage-tflops.py bench \
	--physical-device 4 --device 0 \
	--waves-per-workgroup 8 --waves-per-simd 4 \
	--c0 64 --v0 12 --rounds 128 \
	--schedule 2stage_0 --pattern simd --vmem-op read
```

通用矩阵使用`sweep`，MoE等工作量矩阵使用`bench --single-dispatch --workgroups N --active-workgroups A`。PMC先在rocprofv3下执行`pmc-run`生成expected JSON和CSV，再用`pmc-analyze`校验SQ/L2/HBM闭合。用户指定配置的artifact统一以`/tmp/wave-usercfg-20260826-*`命名。

## 通用参数矩阵

以下84项于2026-08-26用无侵入CUDA-event计时重新采集：temporal、Q4、128 rounds、8 workgroups/CU、3次warmup、12个sample。表中为TFLOPS中位值；`2stage_barrier`不支持W4。

| W | C0 | V0 | pattern | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `2stage_round_barrier` | 最高预测 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 32 | 12 | private | 176.82 | 174.77 | 不适用 | 176.29 | 169.10 | `2stage_0` 176.82 |
| 4 | 32 | 12 | simd | 169.34 | 165.80 | 不适用 | 169.60 | 172.39 | `2stage_round_barrier` 172.39 |
| 4 | 64 | 8 | private | 442.40 | 449.72 | 不适用 | 439.60 | 422.93 | `2stage_prio` 449.72 |
| 4 | 64 | 8 | simd | 455.35 | 464.35 | 不适用 | 467.73 | 467.36 | `interleave` 467.73 |
| 4 | 64 | 12 | private | 336.08 | 332.33 | 不适用 | 321.35 | 323.07 | `2stage_0` 336.08 |
| 4 | 64 | 12 | simd | 348.47 | 342.14 | 不适用 | 359.46 | 347.66 | `interleave` 359.46 |
| 8 | 32 | 12 | private | 161.55 | 161.85 | 163.92 | 161.36 | 163.84 | `2stage_barrier` 163.92 |
| 8 | 32 | 12 | simd | 160.30 | 160.22 | 245.98 | 159.71 | 241.27 | `2stage_barrier` 245.98 |
| 8 | 64 | 8 | private | 454.48 | 450.10 | 456.29 | 447.40 | 403.04 | `2stage_barrier` 456.29 |
| 8 | 64 | 8 | simd | 438.98 | 437.56 | 552.40 | 436.08 | 523.08 | `2stage_barrier` 552.40 |
| 8 | 64 | 12 | private | 321.35 | 320.81 | 326.08 | 319.69 | 317.62 | `2stage_barrier` 326.08 |
| 8 | 64 | 12 | simd | 326.17 | 325.50 | 466.59 | 338.49 | 435.92 | `2stage_barrier` 466.59 |
| 16 | 32 | 12 | private | 159.18 | 159.38 | 164.05 | 159.17 | 164.17 | `2stage_round_barrier` 164.17 |
| 16 | 32 | 12 | simd | 161.10 | 160.62 | 317.19 | 161.19 | 287.09 | `2stage_barrier` 317.19 |
| 16 | 64 | 8 | private | 448.51 | 448.83 | 456.87 | 448.87 | 402.27 | `2stage_barrier` 456.87 |
| 16 | 64 | 8 | simd | 429.33 | 442.13 | 578.70 | 465.91 | 525.07 | `2stage_barrier` 578.70 |
| 16 | 64 | 12 | private | 318.13 | 318.77 | 325.98 | 318.31 | 322.37 | `2stage_barrier` 325.98 |
| 16 | 64 | 12 | simd | 326.96 | 327.15 | 559.47 | 340.64 | 464.43 | `2stage_barrier` 559.47 |

完整结果：`/tmp/wave-rerun-20260826-typical-<schedule>.json`。

## MoE测试case预测与实测

以下五项来自[`test_moe.py`](../../../contrib/moe/test_moe.py)中的模型配置，batch固定为32768。模型本身不认识case：输入只包含W/Q、总/active WG、rounds、MFMA、VMEM mix、cache、地址共享范围和schedule。

用户给定配置记为`C0/read+write/rounds pattern`。例如`64/8+4/48 private`表示每wave每轮64条MFMA、8条read后4条write、48 rounds、每wave独立地址流，即`C0=64,V0=12,vmem_op=read8_write4,pattern=private`。

预测与实测均为8次warmup后48个sample的useful TFLOPS中位值`[P25--P75]`。Hy3与Xiaomi的probe raw TFLOPS分别乘`294912/296448`和`32/33`。跨W对比固定每个case的active waves、每wave的`C0/V0/rounds`、VMEM mix、cache和地址pattern，仅令active WG数按W反比变化；总waves不能整除W时向上补齐一个WG，新增尾wave只执行guard。各W行的动态MFMA/VMEM总工作量因此一致。

`Q请求->实际`同时给出命令请求值和最终ISA/HIP occupancy允许值。粗体表示与当前生产控制流最接近的通用schedule，不是五种候选中的最高值；误差定义为`拓扑匹配预测/当前实测-1`。`2stage_barrier`不支持W4；“不可驻留”表示最终ISA资源不足以形成W16/Q4 WG，不是零吞吐预测。

### Hy3

| 数据/配置 | W | Q请求->实际 | 总/active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `round_barrier` | 实测/匹配误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前实测（`true8_hy3`） | 8 | 4 | 4801/4632 | -- | -- | -- | -- | -- | 412.82 [410.80--413.91] |
| probe（`32/4+3/24; private`） | 4 | 1->1 | 9602/9264 | 215.54 [214.88--216.01] | **215.04 [214.67--215.41]** | 不适用 | 216.47 [216.08--217.02] | 214.63 [214.15--215.04] | -47.91% |
| probe（同配置） | 4 | 2->2 | 9602/9264 | 238.93 [238.58--239.23] | **239.09 [238.88--239.75]** | 不适用 | 236.14 [235.71--236.54] | 232.65 [231.77--233.25] | -42.08% |
| probe（同配置） | 4 | 4->4 | 9602/9264 | 239.72 [239.26--240.04] | **239.49 [227.91--240.08]** | 不适用 | 237.80 [237.48--238.08] | 228.73 [228.27--239.22] | -41.99% |
| probe（同配置） | 8 | 2->2 | 4801/4632 | 228.52 [228.06--229.05] | **228.66 [228.15--228.82]** | 235.04 [234.81--235.40] | 228.87 [228.68--229.29] | 224.53 [223.16--225.38] | -44.61% |
| probe（同配置，当前W/Q） | 8 | 4->4 | 4801/4632 | 239.05 [238.64--239.42] | **238.60 [238.15--238.85]** | 237.52 [237.17--237.80] | 236.65 [236.32--236.86] | 236.80 [236.25--237.08] | -42.20% |
| probe（同配置） | 16 | 4->4 | 2401/2316 | 233.31 [232.88--233.47] | **232.44 [232.05--232.86]** | 235.39 [234.97--235.81] | 230.91 [230.50--231.24] | 228.28 [228.02--228.90] | -43.70% |

### H3

| 数据/配置 | W | Q请求->实际 | 总/active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `round_barrier` | 实测/匹配误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前实测（`physical_n256`） | 8 | 2 | 1152/1024 | -- | -- | -- | -- | -- | 381.34 [380.68--381.66] |
| probe（`64/8+4/48; private`） | 4 | 1->1 | 2304/2048 | 278.32 [277.80--278.85] | 278.79 [277.86--279.57] | 不适用 | 269.93 [269.42--270.22] | 234.13 [233.69--234.23] | -- |
| probe（同配置） | 4 | 2->2 | 2304/2048 | 271.93 [271.44--272.54] | 282.94 [271.28--284.03] | 不适用 | 270.03 [269.37--270.44] | 270.21 [269.10--271.02] | -- |
| probe（同配置） | 4 | 4->4 | 2304/2048 | 272.36 [271.98--272.78] | 272.28 [271.92--272.64] | 不适用 | 269.12 [268.53--285.24] | 272.09 [271.33--283.18] | -- |
| probe（同配置，当前W/Q） | 8 | 2->2 | 1152/1024 | 278.90 [277.96--279.42] | 279.29 [278.63--279.82] | **273.00 [272.69--273.33]** | 268.63 [268.20--272.47] | 257.42 [256.54--258.37] | -28.41% |
| probe（同配置） | 8 | 4->4 | 1152/1024 | 272.14 [271.77--272.50] | 272.67 [271.81--272.95] | **273.67 [272.99--274.22]** | 269.03 [268.49--269.35] | 278.52 [278.11--279.23] | -28.23% |
| probe（同配置） | 16 | 4->4 | 576/512 | 273.93 [273.38--274.53] | 273.22 [272.73--273.65] | **275.56 [275.22--276.16]** | 274.03 [273.47--274.34] | 260.67 [260.35--261.30] | -27.74% |

### Xiaomi

| 数据/配置 | W | Q请求->实际 | 总/active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `round_barrier` | 实测/匹配误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前实测（`physical_n256`） | 4 | 2 | 4480/4224 | -- | -- | -- | -- | -- | 352.70 [351.79--353.61] |
| probe（`64/9+4/48; private`） | 4 | 1->1 | 4480/4224 | 249.11 [248.40--249.59] | **241.48 [240.98--247.33]** | 不适用 | 247.24 [247.02--247.40] | 216.64 [216.08--217.31] | -31.53% |
| probe（同配置，当前W/Q） | 4 | 2->2 | 4480/4224 | 248.12 [247.91--248.22] | **247.88 [247.74--248.18]** | 不适用 | 246.99 [246.84--247.14] | 246.52 [245.82--246.77] | -29.72% |
| probe（同配置） | 4 | 4->4 | 4480/4224 | 248.29 [248.10--248.47] | **248.37 [248.18--248.44]** | 不适用 | 246.26 [246.12--246.47] | 247.06 [246.95--247.28] | -29.58% |
| probe（同配置） | 8 | 2->2 | 2240/2112 | 243.17 [242.84--243.32] | **243.13 [243.05--243.34]** | 244.95 [244.75--245.17] | 244.21 [244.00--244.29] | 233.99 [233.55--234.90] | -31.07% |
| probe（同配置） | 8 | 4->4 | 2240/2112 | 246.18 [246.01--246.33] | **246.16 [246.03--246.29]** | 246.05 [245.95--246.18] | 244.14 [243.98--244.25] | 243.82 [243.69--243.97] | -30.21% |
| probe（同配置） | 16 | 4->4 | 1120/1056 | 239.95 [239.78--240.03] | **239.92 [239.66--240.06]** | 239.75 [239.61--239.97] | 239.15 [239.08--239.30] | 240.26 [239.89--240.83] | -31.98% |

### Qwen3.5 35B

| 数据/配置 | W | Q请求->实际 | 总/active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `round_barrier` | 实测/匹配误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前实测（`legacy`） | 4 | 2 | 4352/4096 | -- | -- | -- | -- | -- | 389.32 [388.11--390.03] |
| probe（`64/9+2/32; private`） | 4 | 1->1 | 4352/4096 | 287.19 [285.99--288.19] | 287.67 [287.13--289.19] | 不适用 | 300.62 [299.90--301.45] | **285.25 [283.21--286.92]** | -26.73% |
| probe（同配置，当前W/Q） | 4 | 2->2 | 4352/4096 | 303.91 [303.63--304.25] | 303.81 [303.44--304.18] | 不适用 | 304.65 [304.20--304.89] | **302.07 [301.49--303.75]** | -22.41% |
| probe（同配置） | 4 | 4->4 | 4352/4096 | 304.17 [303.66--304.41] | 304.18 [303.73--304.46] | 不适用 | 302.99 [302.62--303.23] | **303.70 [303.39--303.90]** | -21.99% |
| probe（同配置） | 8 | 2->2 | 2176/2048 | 298.82 [298.08--303.46] | 303.68 [302.67--304.72] | 302.11 [301.78--302.47] | 301.74 [301.34--302.08] | **297.80 [296.50--298.66]** | -23.51% |
| probe（同配置） | 8 | 4->4 | 2176/2048 | 303.87 [303.49--304.04] | 303.94 [303.44--304.26] | 304.87 [304.43--305.23] | 303.20 [302.94--303.33] | **302.68 [302.26--302.92]** | -22.26% |
| probe（同配置） | 16 | 4->4 | 1088/1024 | 301.74 [301.07--302.05] | 301.48 [301.24--301.90] | 301.07 [300.75--301.57] | 302.67 [302.27--302.83] | **299.83 [299.55--300.41]** | -22.99% |

### Qwen3.5 397B

| 数据/配置 | W | Q请求->实际 | 总/active WG | `2stage_0` | `2stage_prio` | `2stage_barrier` | `interleave` | `round_barrier` | 实测/匹配误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前实测（`legacy`） | 4 | 2 | 5632/5120 | -- | -- | -- | -- | -- | 400.65 [399.59--401.68] |
| probe（`64/9+2/64; private`） | 4 | 1->1 | 5632/5120 | 289.20 [287.67--290.15] | 289.18 [288.31--290.31] | 不适用 | 305.74 [305.55--305.95] | **286.31 [285.24--287.31]** | -28.54% |
| probe（同配置，当前W/Q） | 4 | 2->2 | 5632/5120 | 307.00 [306.81--307.16] | 306.93 [306.72--307.08] | 不适用 | 306.92 [306.67--307.03] | **305.83 [305.63--306.15]** | -23.67% |
| probe（同配置） | 4 | 4->4 | 5632/5120 | 307.02 [306.84--307.27] | 306.91 [306.62--307.10] | 不适用 | 305.83 [305.64--306.01] | **307.39 [307.11--307.58]** | -23.28% |
| probe（同配置） | 8 | 2->2 | 2816/2560 | 304.59 [304.33--304.82] | 304.59 [304.46--304.93] | 307.55 [307.40--307.80] | 306.84 [306.64--307.11] | **300.05 [299.53--300.39]** | -25.11% |
| probe（同配置） | 8 | 4->4 | 2816/2560 | 307.93 [307.73--308.16] | 307.78 [307.59--307.99] | 308.92 [308.72--309.08] | 306.85 [306.75--306.97] | **307.78 [307.52--307.89]** | -23.18% |
| probe（同配置） | 16 | 4->4 | 1408/1280 | 306.98 [306.73--307.26] | 306.97 [306.61--307.31] | 307.70 [307.52--307.86] | 307.19 [306.99--307.44] | **305.51 [305.08--305.84]** | -23.75% |

不能用五种schedule的最大值当作当前kernel预测；最大值只是候选调度上界。当前kernel必须选择控制拓扑最接近的模式。本轮当前W/Q误差为Hy3 -42.20%、H3 -28.41%、Xiaomi -29.72%、Qwen3.5 35B -22.41%、Qwen3.5 397B -23.67%。

本轮预测结果为`/tmp/wave-usercfg-20260826-<case>-w<W>q<Q>-<schedule>.json`，当前实测为`/tmp/moe-rerun-20260826-actual.jsonl`。

### Hy3循环核对

当前生产实测走`true8_hy3`：WG为8 waves、`BLOCK_M=64`、`BLOCK_N=512`、`BLOCK_K=64`，每wave覆盖`WAVE_N=64`。因此：

```text
nBN = 4096 / 512 = 8
nBK = 192 / 64 = 3
K-core次数/wave = nBN * nBK = 24
MFMA/K-core/wave = 64 * 64 * 64 / (16 * 16 * 32) = 32
MFMA/wave = 32 * 24 = 768
```

`4096/256*3=48`对应N256 WG，不是当前W8/N512生产路径。每个K-core的4条weight read是正确的；output则在一个N512块的3个K-core全部累加后写回。每wave负责M64xN64 bf16，即每个N块写`64*64*2/1024=8`条VMEM，而不是把`8/3`向上取整为每K-core 3条：

```text
核心weight read/wave = 4 * 3 * 8 = 96
output write/wave = 8 * 8 = 64
```

所以用户配置`32/4+3/24`保持了正确的768条MFMA和96条核心read，但发出72条write，比生产核心写回多8条。更合理的等计数macro-round表达是`96/12+8/8`；它仍只是统一VMEM/MFMA探针，不表达生产的精确写回PC。

8次warmup后的SQ/L2/HBM对照直接闭合了上述判断：

| 指标 | 生产`true8_hy3` | 指定private probe |
|---|---:|---:|
| 总/active waves | 38,408 / 37,056 | 38,408 / 37,056 |
| MFMA/active wave | 768 | 768 |
| VMEM read/active wave | 107 | 96 |
| VMEM write/active wave | 64 | 72 |
| L2 hit rate | 63.78% | 30.00% |
| HBM read bytes | 679,925,568 | 3,642,762,496 |
| SQ-derived SCLK | 1438.6 MHz | 1438.0 MHz |

MFMA没有少算，频率也一致；指定probe的HBM read是生产的5.36倍。生产中同expert weight被多条wave/WG复用，而`private`强制每wave使用独立地址流，主动消除了这种缓存复用，因而把探针变成内存压力测试。

四点A/B进一步分离了写回分组与地址共享，表中为同一probe FLOPs口径的raw TFLOPS：

| VMEM分组 | 地址pattern | TFLOPS |
|---|---|---:|
| `32/4+3/24` | private | 239.24 |
| `96/12+8/8` | private | 247.24 |
| `32/4+3/24` | simd | 386.24 |
| `96/12+8/8` | simd | 383.40 |

修正3个K-core后的store分组只恢复约8T；从private改为带复用的共享地址恢复约136--147T，是当前42.20%差距的主因。`simd`仅是通用共享近似，不代表已经复刻生产地址映射。

### 模型定位与业界方法

`simd`不等于真实expert访问。它以物理`CU/SIMD`作为stream key：同一SIMD上的wave无条件别名，不同SIMD上的wave永不共享。生产weight则以逻辑group key选择tensor基址。Hy3同轮`simd`的L2 hit/HBM read为34.95%/3,123 MiB，仍远离生产的63.78%/648 MiB；其386T是共享敏感性点，不是地址回放预测。

但默认方案也不应继续走向精确地址、phase DAG和scheduler回放。那类模型接近重新实现kernel，输入难以在新需求到来时可靠获得，维护成本也会超过直接写候选。业界更常见的工程组合是：

1. **Roofline/SOL做解析上界和瓶颈分类**；
2. **一次性微基准校准硬件可持续ceiling**，而不是每个case拟合系数；
3. **解析模型剪枝后只实测top-K**。Triton/CUTLASS/TVM类工作流的价值也主要来自缩小搜索空间，而不是用一个公式精确选出相差1%的winner。

因此模型应明确输出“上界与假设”，不承诺精确预测最终kernel墙钟。

### 两级可达模型

纯解析Roofline只给hard roof，确实可能比真正可达性能高很多；它不能单独承担规划目标。更根本的是：在算法族、tile和调度尚未选择时，“真正可达值”并不是仅由M/N/K决定的可识别量。在线模型因此分成两级，并增加一个自动生成的可执行witness，但仍不做精确kernel回放。

**Level 0：硬上界。** 只用shape、必要字节、硬件峰值、padding和dispatch tail，保证快速、可解释，用于否决不可能目标。

**Level 1：经验可达ceiling。** 查询由通用basis probes预先测得的可持续吞吐面，而不是使用架构峰值：

```text
P_mfma_sustained(W, Q, core_length)
BW_vmem_sustained(Q, read_write_mix, working_set/cache_size)
P_epilogue_sustained(operation_mix)
```

这些表只按GPU/ROCm和硬件特征分桶，不含模型名或shape修正。在线计算仍为max-plus/ECM式关键路径，不把各pipeline时间盲目相加：

**Executable core ceiling：实测候选上界。** 对解析模型筛出的top 3--5组合，用参数化skeleton生成器执行短测。均衡batch V0已实现为[`probe-batched-gemm-core-ceiling.py`](probe-batched-gemm-core-ceiling.py)，方法与边界见[`batched-gemm-core-ceiling.md`](batched-gemm-core-ceiling.md)。它保留候选的几何、VMEM、MFMA、寄存器和occupancy，但按约定不建立VMEM到MFMA及accumulator到D的依赖，因此结果是实测core ceiling，不是正确GEMM的`demonstrated reachable`下界。成本目标小于1分钟。

**Demonstrated floor：正确实现的已达点。** 只有执行正确数据依赖并通过正确性验证的现有kernel或候选，才能提供可达下界。若新需求尚无正确实现，该项为空，不能由无依赖ceiling替代。

新需求只要求以下输入：

```text
group_m_sizes（dense GEMM可退化为单个M）, N, K, dtype
候选BM/BN/BK, waves/WG
估算或骨架编译得到的VGPR/AGPR/LDS
```

由几何自动得到useful/executed FLOPs、padding、WG数、MFMA数、A/B/D必要字节和dispatch tail。性能上界取分层roof的最小值：

```text
T_compute = executed_flops / P_mfma_sustained(W, Q, core_length)
T_memory  = max(hbm_bytes / BW_hbm, l2_bytes / BW_l2)
T_steady  = max(T_compute, T_memory, T_epilogue_overlap)
T_total   = T_launch + T_prologue + repeats * T_steady + T_tail
P_ceiling = useful_flops / T_total
```

Q由资源档位推导；跨过Q档位时重新计算，不把“更高occupancy”本身当收益。只保留3--4种广义pipeline archetype，例如compute-major、streaming、producer-consumer和reduction；archetype再多就应停止扩模并实测。

模型输出四类结果而不是一个伪精确值：

```text
hard_roof          物理上界
measured_core_ceiling 无依赖参数化skeleton测得的候选core上界
demonstrated_floor 正确kernel已经达到的性能；新需求可为空
planning_band      交叉验证残差给出的可达区间[P10, P90]
calibrated_ceiling 当前archetype下的乐观可达目标
```

有正确实现时，`demonstrated_floor <= attainable <= min(measured_core_ceiling, hard_roof)`是证据边界；无正确实现时只有右侧上界。`planning_band`和`calibrated_ceiling`是条件估计，不伪装成证明。如果需求只允许纯解析、连短probe都不能运行，则只能可靠报告hard roof、瓶颈分类和较宽区间，不能承诺“真正可达TFLOPS”。

`planning_band`使用跨shape留一验证或conformal residual生成；禁止在预测当前shape时使用该shape的实测值。若新输入超出校准特征范围，或区间宽度超过10%--15%，模型必须返回`LOW_CONFIDENCE`并直接触发top-K实测。

离线硬件校准表只需按GPU/ROCm版本维护少量数据：

- FP8 MFMA在Q1/Q2/Q4及W4/W8/W16下的可持续TFLOPS；
- read-only、write-only和常见mixed stream的可持续HBM/L2带宽；
- launch固定成本和资源分配粒度。

这些是硬件ceiling，不含模型名、shape或逐case修正。当前通用矩阵已经说明经验层不可省略：固定W、指令mix和pattern后，不同schedule的attainment仍可从约68%跨到98%；纯Roofline无法区分这类可达性。

### 复用不确定性

缓存不应默认猜一个命中率，而应计算两个或三个场景：

```text
ideal-group-reuse: 每个group的weight只从HBM读取一次
task-reload:       每个BM任务重新读取weight
capacity-screen:   group working set能否容纳于目标cache slice
```

真实上限位于这些假设对应的场景之间。若所有场景都由compute roof限制，地址细节无需建模；若roof随复用场景跨越很大区间，则报告“cache-sensitive”，只增加一个廉价的reuse probe或直接测top-K候选，不升级成完整地址模拟器。区间宽度本身就是优化信号。

### 模型的决策职责

模型应该回答：

1. 目标TFLOPS在当前硬件上是否物理可行；
2. 候选主要受compute、HBM、资源档位还是tail限制；
3. 哪些tile明显不可能胜出，应在写kernel前淘汰；
4. 最值得实测的3--5个候选是什么。

它不应该负责预测相近schedule之间1%--3%的差异。此时scheduler相位、cache替换、功耗频率和依赖暴露已主导，直接ABBA比继续加模型维度便宜。

Hy3说明这种定位更稳健：在实测1438.6 MHz下，FP8 MFMA架构roof为471.40T；乘padding效率后为468.96T，生产412.82T达到该上界的88.03%，剩余理论余量13.60%。最低tensor流量对应的理想HBM roof高于compute roof，因此首先应判断为compute/流水受限，而不是从239T private压力测试推断生产上限。

建议验收指标为：hard roof不被实测显著突破、planning band覆盖率达到目标、瓶颈分类正确、实际最快配置进入预测top-K、模型运行时间远低于一次kernel实现。可以同时跟踪点估计误差，但它不是唯一通过条件。精确ATT/PMC模型保留为已有kernel的二级诊断工具，不进入新需求的默认路径。

工程停止线应明确：解析查询目标小于1秒；生成并运行top 3--5 skeleton小于1分钟；只保留top 3--5正式候选。若为了把区间再缩小几个百分点需要真实layout、完整phase DAG或scheduler回放，就已经越过模型的经济边界，应直接实现和ABBA测试候选。

## PMC闭合

### Schedule动态工作

五种schedule使用同一W8/Q4、`C0=24,V0=5,read3_write2,rounds=2`负载采集SQ PMC。每次先执行1个真实warmup，CSV分析读取第2个、即最后一个dispatch：

| schedule | waves | `SQ_INSTS_MFMA` | `SQ_INSTS_VMEM_RD` | `SQ_INSTS_VMEM_WR` |
|---|---:|---:|---:|---:|
| `2stage_0` | 1,280 | 61,440 | 7,680 | 5,120 |
| `2stage_prio` | 1,280 | 61,440 | 7,680 | 5,120 |
| `2stage_barrier` | 1,280 | 61,440 | 7,680 | 5,120 |
| `interleave` | 1,280 | 61,440 | 7,680 | 5,120 |
| `2stage_round_barrier` | 1,280 | 61,440 | 7,680 | 5,120 |

五种模式的动态MFMA/VMEM工作完全一致，差异来自控制和重叠拓扑，而非多发或少发指令。

### Total与active grid

W4/Q2、96个总WG、80个active WG的guarded精确grid结果：

| 指标 | PMC | 理论 |
|---|---:|---:|
| `SQ_WAVES` | 384 | 96 x 4 |
| `SQ_INSTS_MFMA` | 15,360 | 80 x 4 x 2 x 24 |
| `SQ_INSTS_VMEM_RD` | 1,920 | 80 x 4 x 2 x 3 |
| `SQ_INSTS_VMEM_WR` | 1,280 | 80 x 4 x 2 x 2 |

因此总wave按launch grid计数，MFMA/VMEM严格按active grid计数。

### Private有限流水

W16/Q4、`C0=64,V0=8,rounds=128`、80 WG、1,280 waves的三次counter pass闭合如下：

| 层级 | PMC结果 | 闭合 |
|---|---:|---:|
| `SQ_WAVES` | 1,280 | 100% |
| `SQ_INSTS_MFMA` | 10,485,760 | 100% |
| `SQ_INSTS_VMEM_RD` | 1,310,720 | 100% |
| `TCP_TCC_READ_REQ_sum` | 10,485,760 | 8 requests/VMEM |
| L2 hit rate | 0.02847% | 几乎全下沉HBM |
| HBM read bytes | 1,342,197,120 | 理论+0.001478% |

结果：`/tmp/wave-rerun-20260826-pmc-private-validation.json`。五种schedule和guard闭合结果位于`/tmp/wave-rerun-20260826-pmc-*`。

### 大于4 GiB的private exact-grid

用户配置中的H3 W16/Q4使用576个总WG、512个active WG；每wave独立地址，单dispatch覆盖4.5 GiB逻辑VMEM空间。buffer descriptor base按64位`stream_id * bytes_per_stream`平移，descriptor内offset不超过单wave范围。三次counter pass结果：

| 指标 | PMC结果 | 理论/闭合 |
|---|---:|---:|
| 总/active waves | 9,216 / 8,192 | 576 x 16 / 512 x 16 |
| `SQ_INSTS_MFMA` | 25,165,824 | 100% |
| `SQ_INSTS_VMEM_RD` | 3,145,728 | 100% |
| `SQ_INSTS_VMEM_WR` | 1,572,864 | 100% |
| `TCP_TCC_READ_REQ_sum` / VMEM read | 8 | 100% |
| HBM read bytes | 3,221,252,032 | 理论+0.000825% |

结果：`/tmp/wave-usercfg-20260826-large-private-pmc-validation.json`。该闭合排除了32位offset回绕或大地址alias导致的伪高吞吐。

## 差距分析

已由A/B和PMC确认：

1. **测量kernel必须无侵入。** 每条wave的时钟读取、metadata写回和额外`lgkmcnt(0)`会改变被测控制流。正式计时因此只使用无instrumentation kernel；全wave metadata仅用于短驻留验证。
2. **总wave与active wave必须分开。** `--workgroups/--active-workgroups`复现生产尾部早退WG；本轮guarded SQ闭合中总`SQ_WAVES=384`，MFMA/read/write严格按320条active wave闭合。
3. **控制拓扑仍需匹配，但不解释本轮主差距。** 64-round legacy当前W4/Q2中，`2stage_0`为307.00T，`2stage_round_barrier`为305.83T，仅差1.17T；五种schedule整体都远低于生产400.65T。取最大schedule仍会把候选上界误当当前kernel预测。
4. **频率不是生产/probe差距来源。** 8次warmup后，`SQ_CYCLES / 16 SE / dispatch time`得到短/长生产1438.5/1434.4 MHz，短/长private probe为1439.7/1435.5 MHz，分别接近。1800 MHz仅是目标值。
5. **private地址模式是本轮降速的主要新变量。** 两项Qwen保持`C0=64,V0=11,read9_write2`和原rounds，只把地址共享范围改为每wave独立；拓扑匹配预测降至302.07/305.83T。private probe的L2 hit均为15.39%，低于生产65.15%/50.38%，独立流显著增加下沉内存层级的请求。

Hy3的42.20%表面差距中，主项已经闭合为private地址模式：同指令工作改用`simd`共享近似后，useful口径误差缩小到约-7%。剩余部分仍包含生产真实expert-weight复用与`simd`近似的差异，以及非MFMA phase的**依赖和重叠拓扑**。用户给定private配置更接近无复用压力测试，不应直接等同于生产kernel的有效缓存共享。下一步应从生产地址映射提取通用共享范围，并在模型中加入多类指令交叠图或credit约束，而不是按case增加延迟或固定修正系数。

## 历史方法教训

修复前的探针在正式kernel内为每条wave执行四次时钟读取、48 B metadata写回和`lgkmcnt(0)`，会显著改变被测吞吐。旧绝对TFLOPS、旧SCLK和旧误差不再保留于主文档，也不能与本轮CUDA-event结果混用。ATT历史采集仍支持一个定性结论：严格反相确实改变WG内slot拓扑；它不提供当前性能校准系数。

## 使用边界

该探针仍是简单VMEM/MFMA模型。推广到其他kernel时，应重新提取：

1. MFMA/VMEM比例与read/write顺序；
2. cache policy和地址共享范围；
3. 实际waves/SIMD与编译资源；
4. stage边界策略；
5. 是否存在会暴露在关键路径上的LDS、VALU或长依赖链。

历史Legacy/true8误差不能作为当前模型精度，更不能作为跨应用固定修正系数。当前误差以“MoE测试case预测与实测”表中的拓扑匹配结果为准。
