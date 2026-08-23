# gfx942四wave异构stage优先级实验

本实验使用[`probe-setprio-antiphase.py`](probe-setprio-antiphase.py)验证：两个独立四wave
workgroup同驻一个CU时，按stage统一设置优先级能否令每个物理SIMD上的两个wave精确反相：

- **stage 0：** VMEM + LDS + VALU，所有wave优先级0；
- **stage 1：** MFMA，所有wave优先级1。

优先级不再按硬件slot区分。生成ISA中，每轮只有：

```text
s_setprio 0
stage 0: buffer_load + LDS + VALU
s_setprio 1
stage 1: MFMA
```

退出循环后恢复`s_setprio 0`。ISA中没有slot比较或条件优先级分支。

## stage定义

- **stage 0：** 每个tile依次执行non-temporal `buffer_load_dwordx4`、`vmcnt(0)`、
  `ds_write_b128`、`lgkmcnt(0)`、`ds_read_b128`、`lgkmcnt(0)`和4条依赖`v_xor_b32`；
- **stage 1：** 固定112条`v_mfma_f32_16x16x16_bf16`，在四个独立accumulator间轮转。

stage 0分别重复1/2/4个tile，得到小于、基本等于和大于stage 1的三档实测时长。

## 结论

**统一设置`stage0=0, MFMA=1`不能形成精确反相。** 三档1024-round ATT均有明显活动同stage重叠：

| 档位 | $T_0/T_1$ | run损失 | 活动同stage损失 | 观察窗中位数 | 永久零run损失pair |
|---|---:|---:|---:|---:|---:|
| stage 0 < stage 1 | 0.468 | 74.976% | 56.466% | 1542.029 us | 0/32 |
| stage 0 $\approx$ stage 1 | 0.977 | 57.843% | 38.871% | 2054.192 us | 0/32 |
| stage 0 > stage 1 | 2.027 | 1.485% | 35.655% | 3189.986 us | 24/32 |

stage 0较长时run序列仍能快速形成近一一配对，但活动同stage损失为35.66%，不是物理精确反相。

与等优先级`equal4`相比，新策略的活动损失只变化`-1.58 / -0.91 / +1.58`个百分点；基本处于同一
量级。这是因为同一时刻处于同一stage的两个ready wave获得相同优先级，`s_setprio`没有提供让它们
相互错开的slot差异或同步关系。

因此不存在本实验支持的精确稳定时间。架构语义上的结论不变：`s_setprio`只影响ready-wave仲裁，
没有跨workgroup happens-before关系，不能成为正确性依赖。

## 模式

- **barrier8：** 一个8-wave workgroup，每个SIMD两条wave，通过真实barrier建立已知精确反相。它是
  分析器校准基准，三档run损失和活动同stage损失均为0。
- **equal4：** 两个独立4-wave workgroup同驻一个CU，所有stage使用默认相同优先级。
- **priority4：** occupancy与equal4相同，但所有wave在stage 0设优先级0、进入MFMA stage设优先级1。

此前测试过的slot非对称策略`slot0: 3/1, slot1: 2/0`仅保留作历史对照，不再是脚本当前行为。

## 测量方法

hot loop只有两个静态stage、`s_setprio`和必要barrier；循环内没有时钟读取或用于观测的全局写回。
stage 0自身包含待测的LDS write。每个stage入口使用唯一`s_nop` marker，rocprofv3 ATT按PC区分stage，
并使用成功issue时刻：

$$
t_{\mathrm{issue}}=t_{\mathrm{first\ attempt}}+t_{\mathrm{stall}}
$$

分析器强制验证：

- 每条wave恰有$2R$个A/B marker和$2R$个stage-run；
- 每个stage 0 run恰有$7r$个目标issue，其中$r$为tile重复次数；
- 每个stage 1 run恰有112个MFMA issue；
- ATT `num_insts == num_stitched`；
- 每个采样SIMD恰有slot `(0, 1)`；
- dispatch/pair数量符合预期，capture log没有cutoff或incomplete标记。

### run损失

run损失用于观察stage序列是否一一配对。一个run只有在最近peer run与它互为最近邻、stage相反且
不存在等距歧义时才稳定：

$$
L_{\mathrm{run}}=
\frac{N_{\mathrm{unstable\ runs}}}{N_{\mathrm{observed\ runs}}}
$$

### 活动同stage损失

活动同stage损失是判断物理反相的主指标。只统计两条wave实际stage issue-span互相重叠的周期：

$$
L_{\mathrm{active}}=
\frac{C_{\mathrm{same\ stage}}}
{C_{\mathrm{same\ stage}}+C_{\mathrm{opposite\ stage}}}
$$

barrier8在三档的两个指标都严格为0%，证明分类器能识别已知精确反相。

对于没有补偿等待、连续交替的两个独立wave，stage时长不等还带来占空比下界：

$$
L_{\mathrm{duty}}=
\frac{|T_0-T_1|}{T_0+T_1}
$$

stage 0约两倍长时，新策略的$L_{\mathrm{active}}=35.66\%$，只比
$L_{\mathrm{duty}}=33.93\%$高1.73个百分点。run配对接近完美时，较长stage的剩余部分仍没有相反stage
可以覆盖，因此物理反相不可能为0损失。

## 受控结果

平台为MI308X `gfx942`，GPU4，80 CU，ROCm 7.2.3，rocprofv3 1.1.0，650 W power cap，PTL
`VECTOR,F8`，performance determinism 1800 MHz。每次capture选CU0、四个shader engine和四个SIMD；
两个dispatch合计32个物理SIMD pair。所有结果均为0 placement failure、0 trace failure，slot均为
`(0, 1)`。

priority4/equal4均使用1024 rounds；barrier8使用256 rounds校准三档时长。

| 模式 | stage 0 tiles | $T_0/T_1$ | $L_{run}$ | $L_{active}$ | 首个64-run零损失pair | 永久零run损失pair |
|---|---:|---:|---:|---:|---:|---:|
| barrier8 | 1 | 0.527 | 0.000% | 0.000% | 32/32 | 32/32 |
| barrier8 | 2 | 0.982 | 0.000% | 0.000% | 32/32 | 32/32 |
| barrier8 | 4 | 1.921 | 0.000% | 0.000% | 32/32 | 32/32 |
| priority4，新0/1策略 | 1 | 0.468 | 74.976% | 56.466% | 0/32 | 0/32 |
| priority4，新0/1策略 | 2 | 0.977 | 57.843% | 38.871% | 14/32 | 0/32 |
| priority4，新0/1策略 | 4 | 2.027 | 1.485% | 35.655% | 32/32 | 24/32 |
| equal4 | 1 | 0.462 | 74.218% | 58.047% | 0/32 | 0/32 |
| equal4 | 2 | 0.980 | 59.353% | 39.776% | 5/32 | 0/32 |
| equal4 | 4 | 2.011 | 1.405% | 34.079% | 32/32 | 29/32 |

### stage 0小于stage 1

run损失74.98%，活动损失56.47%，1.54 ms内没有任何64-run零损失窗口。新策略相对equal4只把活动
损失降低1.58个百分点，未产生锁相。

### 两stage基本相当

run损失57.84%，活动损失38.87%。14/32 pair曾出现64-run零损失窗口，首次出现时间中位959.51 us，
但所有pair之后都再次失锁，永久稳定pair为0/32。相较旧slot非对称策略，新策略的活动损失反而增加
7.54个百分点。

### stage 0大于stage 1

run损失1.48%，所有pair在中位2.59 us内出现64-run零损失窗口，24/32 pair形成永久零run损失后缀。
但活动损失仍为35.66%，且equal4为34.08%。低run损失主要来自约2:1的stage占空比自组织，不是MFMA
优先级1建立了反相同步。

## 与旧slot非对称策略对比

| stage档位 | 新0/1策略 $L_{active}$ | 旧`3/2,1/0`策略 | equal4 |
|---|---:|---:|---:|
| stage 0 < stage 1 | 56.466% | 51.496% | 58.047% |
| stage 0 $\approx$ stage 1 | 38.871% | 31.335% | 39.776% |
| stage 0 > stage 1 | 35.655% | 34.710% | 34.079% |

统一stage优先级没有优于旧slot非对称策略；在等长档退化最明显。原因是新策略只表达“MFMA比其他stage
优先”，却没有表达同一SIMD两条wave之间的相位关系。

## 意外与原因

1. **提高MFMA优先级不会自动产生反相。** 两条wave同时进入MFMA时都会变成优先级1，仲裁器没有信息
   决定哪一条应等待。
2. **低run损失不等于物理反相。** stage 0较长时，run中心可稳定配对，但较长stage的剩余活动仍会与
   peer同stage重叠。
3. **时长不对称可产生自然节奏。** `equal4`在stage 0较长档也接近新策略，证明低run损失不能归因于
   `s_setprio`。
4. **VMEM延迟有分布。** stage 0含non-temporal load，$T_0/T_1$在模式间略有变化，档位必须由ATT
   实测而不能只看静态指令数。
5. **没有跨WG纠错机制。** 启动偏斜、cache miss或某次stage超调后，统一0/1优先级不会强制恢复固定
   相位差。

## 复现

先运行分类器自测和placement检查：

```bash
/tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-setprio-antiphase.py self-test

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
/tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-setprio-antiphase.py run \
  --device 0 --mode priority4 --rounds 256 \
  --stage0-repeats 2 --stage1-mfmas 112 --data-mib 512 --dispatches 2 \
  --json /tmp/setprio-antiphase-placement.json
```

[`setprio-antiphase-att.yaml`](setprio-antiphase-att.yaml)默认写入
`/tmp/setprio-antiphase-att`并捕获第二次目标dispatch：

```bash
rm -rf /tmp/setprio-antiphase-att
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
rocprofv3 -i tests/flydsl/attn_4wave/tools/setprio-antiphase-att.yaml \
  --att-library-path /tmp/h3-rocprof-decoder/opt/rocm/lib -- \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-setprio-antiphase.py run \
  --device 0 --mode priority4 --rounds 1024 \
  --stage0-repeats 2 --stage1-mfmas 112 --data-mib 512 --dispatches 2 \
  --json /tmp/setprio-antiphase-placement.json \
  > /tmp/setprio-antiphase-capture.log 2>&1

/tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-setprio-antiphase.py analyze \
  --att-root /tmp/setprio-antiphase-att --rounds 1024 \
  --stage0-repeats 2 --stage1-mfmas 112 --stable-window-events 64 \
  --clock-mhz 1800 --expected-dispatches 2 --expected-pairs-per-dispatch 16 \
  --capture-log /tmp/setprio-antiphase-capture.log \
  --json /tmp/setprio-antiphase-analysis.json
```

`--clock-mhz 1800`只应用于已锁1800 MHz的capture。正式复现还应保存原始DPM/PTL状态并在`finally`
中恢复；本轮所有受控capture结束后均恢复到原始`auto + F8,VECTOR`状态。
