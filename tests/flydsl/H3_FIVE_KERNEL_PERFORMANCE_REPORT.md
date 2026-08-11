# H3 Attention 五实现逐 Dispatch 性能报告

日期：2026-08-10

本报告按 [H3_ATTENTION_THROTTLE_PROFILE.md](H3_ATTENTION_THROTTLE_PROFILE.md) 的逐 dispatch
协议和 [H3_AITER_ATTN_UNITTEST.md](pa_4wave/H3_AITER_ATTN_UNITTEST.md) 的真实 H3 shape、
AITER 入口及正确性口径，实测以下五项：

1. AITER Triton varlen
2. AITER ASM group RTNA，MI308 二进制
3. AITER ASM group RTNA，MI300 二进制运行在 MI308X 上
4. FlyDSL `8wave_32x32` dense 近似
5. FlyDSL `4wave_varlen` 真实 paged varlen

## 一页结论

平均吞吐排序为 `8wave_32x32 > 4wave_varlen > ASM MI300 > Triton > ASM MI308`；稳定性则相反，
Triton 和 ASM MI308 基本稳定，另外三项都有约 1 秒的 DPM 周期。因而不能只看平均 TFLOPS：

| 实现 | 70 次平均耗时 | 70 次平均 TFLOPS | 相对 Triton 耗时 | 完整周期持续均值 | 峰值 / 谷值 TFLOPS | 峰谷降幅 | 稳定周期 |
|---|---:|---:|---:|---:|---:|---:|---|
| Triton | 191.049 ms | 149.988 | 基线 | 不适用 | 150.779 / 144.019 | 4.48% | 未检测到 |
| ASM MI308 RTNA | 191.377 ms | 149.727 | +0.17% | 不适用 | 150.293 / 144.987 | 3.53% | 未检测到 |
| ASM MI300 RTNA | 168.773 ms | 172.129 | -11.66% | 169.266 ms / 171.635T | 204.168 / 151.085 | 26.00% | 每 6 次，1.0166 s |
| 8wave_32x32 | **158.359 ms** | **186.027** | **-17.11%** | 158.137 ms / 186.358T | 255.432 / 159.526 | **37.55%** | 每 6-7 次，1.0133 s |
| 4wave_varlen | 160.494 ms | 182.902 | -15.99% | 161.027 ms / 182.196T | 235.359 / 157.587 | 33.04% | 每 6-7 次，1.0112 s |

这里的两个平均值都是逐 dispatch 算术平均：分别对 70 个 event 耗时和 70 个逐次 TFLOPS 求平均；
没有排序、截断或使用中值。`完整周期持续均值`只对完整的 `[burst_i, burst_{i+1})` 区间按 dispatch
加权，排除末尾不完整周期，避免采样恰好停在高档而抬高结果。

对在线延迟的直观解释：

- `8wave_32x32` 的平均值和峰值最好，但单次延迟在 `112.176-179.615 ms` 间摆动，稳定性最差。
- `4wave_varlen` 的平均吞吐比 Triton 高 21.94%，但单次延迟在 `121.743-181.825 ms` 间摆动。
- ASM MI300 比 Triton 平均快 11.66%，但单次延迟仍在 `140.342-189.650 ms` 间周期摆动。
- ASM MI308 与 Triton 几乎打平，且两者本轮没有可重复的 6-7 dispatch 循环。
- 因此“最快实现”取决于目标：追求平均吞吐选 8wave；追求单次可预测性，本轮 Triton/MI308 更稳。

## 测试口径

### 输入与计时

- 逻辑 shape：BF16，`q=k=v=(63232,14,128)`，真实 segments `(63225,7)`，seed `1101`，
  `causal=False`，scale 为 $1/\sqrt{128}$。
- 真实 H3 FLOPs 为
  $\sum_i 4S_i^2DH=28,653,368,031,232$；所有表中 TFLOPS 均按该值归一。
- Triton、两个 ASM 和 `4wave_varlen` 执行真实 varlen pack。
- `8wave_32x32` 执行 `M=N=63232` dense 近似并使用 preshuffled V；它比真实 H3 多算
  0.022143%，但报告仍按真实 H3 FLOPs 归一。
- 每项先预热 3 次，再连续采集 70 次；CUDA event 只覆盖 kernel dispatch。
- sysfs 每 10 ms 采样 SCLK、PPT 功耗、junction 和 HBM 温度；每个 dispatch 都取得了至少一个点。
- 使用 auto DPM，没有固定 SCLK，没有调整 power cap、风扇、MCLK、FCLK 或 performance profile。

AITER 文档原协议是 `median of 10`。本报告改用降频手册的 70 次原始序列，因为 10 次中值会隐藏约
1 秒的控制周期；AITER 的 kernel 入口、RTNA 选择、真实 shape 和正确性检查保持不变。

### 硬件与软件

| 项目 | 本次值 |
|---|---|
| GPU | 物理 GPU 4，`0001:0b:00.0`，AMD Instinct MI308X，`gfx942`，80 CU，约 192 GiB |
| 功耗 / 时钟 | 650 W cap；测试前后 FCLK/MCLK 读回 1300/900 MHz；SCLK 由 auto DPM 控制 |
| 主机 | Linux `5.10.134-18.al8.x86_64`；`kernel.numa_balancing=1`；GPU NUMA node 1 |
| Python / Torch | Python 3.10.12；Torch `2.9.0a0+git7bcbafe`；HIP `7.0.51831-a3e329ad8` |
| AITER | `0.1.14rc1.dev240+g7d604afe5`，commit `7d604afe5fa7efba63c0dce323b95d9daf2db112` |
| FlyDSL | AITER 进程为 0.2.0；FlyDSL kernel 使用隔离环境 0.3.0 |
| PyHIP | commit `438e9faad7acf3b43ca9136764548c95a5f70a42` 加本次工作区修改 |

这不是降频手册参考环境的严格复现：参考环境是 Torch/ROCm 7.2、FlyDSL `0.3.0.dev765`、另一 PyHIP
commit。FlyDSL 0.2.0 无法编译当前两个 FlyDSL kernel，因此使用最接近参考的 PyPI FlyDSL 0.3.0；
AITER 仍在镜像原环境运行。五项共享同一张 GPU、Torch/HIP、shape、seed、FLOPs 和采样协议，本报告可用于
本机实现间比较，但跨环境引用时必须保留这些版本差异。

每轮 preflight 都在 CUDA 初始化前确认目标卡 `busy=0`、约 284 MiB 驱动保留显存、无 KFD 进程且
DPM=`auto`。正式 JSON 在测试结束时只看到采集进程自己的 CUDA context，没有第二个 KFD 进程。

## 正确性与二进制身份

AITER 正确性使用真实 `(63225,7)` pack，以逐 segment BF16 SDPA 为参考，并单独检查 7-token tail：

| 实现 | 整包 cosine | 整包 max abs | tail cosine | tail max abs |
|---|---:|---:|---:|---:|
| Triton | 1.000000000 | 0.000122 | 1.000000119 | 0.000122 |
| ASM MI308 group RTNA | 1.000000000 | 0.000122 | 1.000000119 | 0.000122 |
| ASM MI300 group RTNA | 1.000000000 | 0.000122 | 1.000000119 | 0.000000 |

MI308X 的 PCI chip ID 会让 AITER 固定从 `MI308/` 加载 `.co`。为测 MI300 二进制，本次在独立进程启动前
临时把 MI300 RTNA group `.co` 放到活动路径，进程退出后通过 trap 恢复原件；采集器同时校验并记录活动
文件内容 hash：

- MI308 `fwd_hd128_bf16_rtna_group.co`：
  `3687c5610a454572e4a615ec58f05e707fdf3995e4dc932cf2219ad2fa0052ff`
- MI300 `fwd_hd128_bf16_rtna_group.co`：
  `f8d7e1dfc5301edeb83e5520e8d710798c7641a52040c33dbed77c18115813c5`

正式采集结束后，活动 MI308 文件已恢复并再次通过第一个 hash。这里的“ASM MI300”指 MI300 `.co`
运行在 MI308X 硬件上，不是使用了一张 MI300 GPU。

## 详细性能与遥测

高档阈值按 `min + 0.65 * (max - min)`；只有检测到稳定周期的三项才计算高低档。稳定 Triton/MI308
中少数低点不构成重复周期，因此高低档栏不适用。

| 实现 | 平均 SCLK / 功耗 | 高档 SCLK / 功耗 | 低档 SCLK / 功耗 | 峰值功耗 | 最高 junction / HBM | TFLOPS-SCLK 相关性 | TFLOPS-power 相关性 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Triton | 1802 MHz / 591 W | 不适用 | 不适用 | 601 W | 83C / 49C | 0.4849 | 0.3240 |
| ASM MI308 | 1794 MHz / 538 W | 不适用 | 不适用 | 561 W | 84C / 49C | -0.0276 | 0.0674 |
| ASM MI300 | 1476 MHz / 496 W | 1586 MHz / 539 W | 1418 MHz / 474 W | 608 W | 79C / 48C | **0.8561** | **0.8081** |
| 8wave_32x32 | 1254 MHz / 432 W | 1477 MHz / 508 W | 1193 MHz / 411 W | 600 W | 69C / 46C | 0.7698 | 0.6861 |
| 4wave_varlen | 1321 MHz / 468 W | 1513 MHz / 540 W | 1250 MHz / 441 W | 616 W | 73C / 47C | 0.7962 | 0.7271 |

循环项的吞吐与 SCLK、功耗均高度正相关，说明性能摆动主要由运行频率变化直接驱动。三个循环项的高吞吐
burst 起点如下：

- ASM MI300：`0,6,12,18,24,30,36,42,48,54,60,66`
- 8wave_32x32：`0,7,13,19,26,32,38,45,51,57,64`
- 4wave_varlen：`0,6,13,19,25,31,38,44,50,56,63,69`

Triton 和 ASM MI308 的峰谷降幅只有 4.48%/3.53%，且候选高区间间隔不重复，按手册判定
`cycle_detected=False`。ASM MI300 的周期最规整，连续 11 个间隔都是 6 dispatch，实际周期仅在
`1.01596-1.01717 s` 间变化。

## 每个完整周期的平均值

一个完整周期定义为 `[当前高吞吐 burst 起点, 下一高吞吐 burst 起点)`。平均耗时和平均 TFLOPS 都保留
该周期内每个 dispatch 的等权算术平均；末尾没有下一个 burst 边界的区间不纳入。

| 实现 | 周期 | dispatch | 数量 | 周期时间 | 平均耗时 | 平均 TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| ASM MI300 | 0 | 0-5 | 6 | 1.0171 s | 169.340 ms | 171.929 |
| ASM MI300 | 1 | 6-11 | 6 | 1.0170 s | 169.305 ms | 171.877 |
| ASM MI300 | 2 | 12-17 | 6 | 1.0160 s | 169.203 ms | 171.850 |
| ASM MI300 | 3 | 18-23 | 6 | 1.0161 s | 169.229 ms | 171.708 |
| ASM MI300 | 4 | 24-29 | 6 | 1.0170 s | 169.336 ms | 171.551 |
| ASM MI300 | 5 | 30-35 | 6 | 1.0164 s | 169.252 ms | 171.598 |
| ASM MI300 | 6 | 36-41 | 6 | 1.0162 s | 169.185 ms | 171.620 |
| ASM MI300 | 7 | 42-47 | 6 | 1.0163 s | 169.242 ms | 171.515 |
| ASM MI300 | 8 | 48-53 | 6 | 1.0172 s | 169.337 ms | 171.414 |
| ASM MI300 | 9 | 54-59 | 6 | 1.0165 s | 169.267 ms | 171.483 |
| ASM MI300 | 10 | 60-65 | 6 | 1.0165 s | 169.234 ms | 171.440 |
| 8wave_32x32 | 0 | 0-6 | 7 | 1.0844 s | 154.725 ms | 192.129 |
| 8wave_32x32 | 1 | 7-12 | 6 | 0.9678 s | 161.195 ms | 182.629 |
| 8wave_32x32 | 2 | 13-18 | 6 | 0.9612 s | 160.001 ms | 184.628 |
| 8wave_32x32 | 3 | 19-25 | 7 | 1.0871 s | 155.138 ms | 190.612 |
| 8wave_32x32 | 4 | 26-31 | 6 | 0.9667 s | 160.920 ms | 182.751 |
| 8wave_32x32 | 5 | 32-37 | 6 | 0.9598 s | 159.758 ms | 184.218 |
| 8wave_32x32 | 6 | 38-44 | 7 | 1.0896 s | 155.456 ms | 189.429 |
| 8wave_32x32 | 7 | 45-50 | 6 | 0.9656 s | 160.802 ms | 182.687 |
| 8wave_32x32 | 8 | 51-56 | 6 | 0.9597 s | 159.746 ms | 183.612 |
| 8wave_32x32 | 9 | 57-63 | 7 | 1.0908 s | 155.573 ms | 188.371 |
| 4wave_varlen | 0 | 0-5 | 6 | 0.9647 s | 160.622 ms | 184.520 |
| 4wave_varlen | 1 | 6-12 | 7 | 1.1133 s | 158.850 ms | 184.361 |
| 4wave_varlen | 2 | 13-18 | 6 | 0.9780 s | 162.834 ms | 179.604 |
| 4wave_varlen | 3 | 19-24 | 6 | 0.9765 s | 162.591 ms | 180.795 |
| 4wave_varlen | 4 | 25-30 | 6 | 0.9679 s | 161.166 ms | 183.249 |
| 4wave_varlen | 5 | 31-37 | 7 | 1.1114 s | 158.572 ms | 184.577 |
| 4wave_varlen | 6 | 38-43 | 6 | 0.9775 s | 162.780 ms | 179.341 |
| 4wave_varlen | 7 | 44-49 | 6 | 0.9771 s | 162.679 ms | 179.962 |
| 4wave_varlen | 8 | 50-55 | 6 | 0.9692 s | 161.373 ms | 182.439 |
| 4wave_varlen | 9 | 56-62 | 7 | 1.1098 s | 158.325 ms | 185.001 |
| 4wave_varlen | 10 | 63-68 | 6 | 0.9774 s | 162.733 ms | 179.076 |

未纳入完整周期汇总的尾段：ASM MI300 `66-69`、8wave_32x32 `64-69`、4wave_varlen `69`。
Triton 和 ASM MI308 未检测到稳定周期，所以没有人为把偶发低点之间的区间标成周期。

## 名词与结论边界

### PPT

`PPT` 是 Package Power Tracking，可理解为 GPU/加速卡功率预算控制。固件不是只看某一个 10 ms
瞬时功耗点，而会在内部时间窗口内追踪功耗/能量；接近预算后，DPM 可能降低 SCLK，等窗口释放后再升频。
因此即使采样到的峰值低于 650 W cap，也不能排除更短尖峰或滑动窗口功率预算触发控制。

本机 sysfs 的功耗传感器标签为 PPT，但 AMDSMI 对 `ppt_accumulated`、PROCHOT 和各 thermal accumulated
counter 均返回字面量 `N/A`。所以本轮可以确认“吞吐、SCLK、功耗同步周期变化”，不能仅凭 counter
直接证明“PPT violation 发生了多少次”。

### DPM 与时钟

- `DPM`：Dynamic Power Management，驱动/固件根据功耗、温度、负载和策略动态选择工作频率/电压。
- `SCLK` 或 `gfxclk`：GPU 核心计算时钟；本轮与吞吐强相关，是循环降速的直接表现。
- `MCLK`：HBM 内存时钟；本轮读回 900 MHz。
- `FCLK`：fabric/互联时钟；本轮读回 1300 MHz。
- `performance determinism`：固定 SCLK 上限的对照模式，可隔离 auto DPM 相位，但不能替代 auto-DPM
  生产基线。本轮未改变 DPM，全部结果来自 `auto`。

### 温度与保护

- `junction`/`hotspot`：芯片内部热点温度，不等于环境温度或 HBM 温度。
- `HBM temperature`：显存温度。
- `PROCHOT`：Processor Hot，硬件过热/保护信号；触发时通常会强制降频。

循环项最高 junction 只有 69-79C，而没有稳定循环的 ASM MI308 反而达到 84C；这不符合简单的“达到固定
温度阈值就循环降频”解释，更倾向功率/控制窗口。但由于 thermal/PROCHOT counter 为 `N/A`，只能说
温度保护不是本轮首要嫌疑，不能声称已由硬件 counter 完全排除。

### CUDA event 与 TFLOPS

CUDA event 在 ROCm PyTorch 中用于记录 GPU stream 上两个事件之间的设备执行时间，sysfs 采样线程和 Python
打印不计入 `elapsed_ms`。本报告的 TFLOPS 是算法 FLOPs 除以 event 时间，不代表 GPU 所有真实执行指令数，
也不包含编译、输入构造或 host 调度开销。

## 复现

### 前提

- 从仓库根目录运行，ROCm/Torch/AITER环境与“硬件与软件”一节一致；默认AITER解释器为
  `/opt/venv/bin/python3`。
- AITER安装目录必须同时包含MI308和MI300的BF16 RTNA group code object，且SHA256与“正确性与
  二进制身份”一节一致。wrapper在运行前校验两者，任何不匹配都会停止。
- 默认使用物理GPU 4；可用`GPU=<index>`覆盖。profiler会在CUDA初始化前要求目标卡busy为0、无其他
  KFD进程、初始显存占用不超过1 GiB且DPM为`auto`。
- FlyDSL kernel需要0.3.0。不要提交虚拟环境，可在容器现有Torch环境上创建：

```bash
/opt/venv/bin/python3 -m venv --system-site-packages \
  artifacts/h3-five-kernel/venv-flydsl030
artifacts/h3-five-kernel/venv-flydsl030/bin/python -m pip install flydsl==0.3.0
artifacts/h3-five-kernel/venv-flydsl030/bin/python -m pip install -e .
```

### 相关代码

| 文件 | 职责 |
|---|---|
| [run_h3_five_kernel_auto.sh](run_h3_five_kernel_auto.sh) | 三轮采集、MI300 `.co`切换/恢复、离线分析与manifest |
| [h3_profile_common.sh](h3_profile_common.sh) | 在任何CUDA初始化前检查GPU进程、busy、显存和DPM |
| [profile_h3_attention_throttle.py](profile_h3_attention_throttle.py) | 构造五种实现并记录70条逐dispatch event和10 ms遥测 |
| [analyze_h3_attention_throttle.py](analyze_h3_attention_throttle.py) | 原序列均值、峰谷、相关性、burst和完整周期分析 |
| [h3_attn_kernel_test.py](pa_4wave/h3_attn_kernel_test.py) | Triton/ASM真实H3 pack正确性和7-token tail检查 |
| [test_attn_8wave_32x32_lkgv.py](test_attn_8wave_32x32_lkgv.py) | FlyDSL 8-wave dense入口 |
| [test_pa_prefill.py](pa_4wave/test_pa_prefill.py) | FlyDSL 4-wave真实paged-varlen入口 |

### 正式运行

一条命令复现AITER正确性、三轮性能采集、离线分析和校验和：

```bash
bash tests/flydsl/run_h3_five_kernel_auto.sh
```

[run_h3_five_kernel_auto.sh](run_h3_five_kernel_auto.sh)按以下顺序执行：

1. 用MI308 `.co`执行真实`(63225,7)` pack的AITER正确性，以及Triton和ASM MI308的
   `3 warmup + 70 dispatch`采集。
2. 在独立进程前把MI300 `.co`临时放入MI308活动路径，重新执行正确性和70次ASM MI300采集。
3. 恢复MI308 `.co`并验证hash，然后用FlyDSL 0.3.0环境采集`8wave_32x32,4wave_varlen`。
4. 对三份profile运行[analyze_h3_attention_throttle.py](analyze_h3_attention_throttle.py)，生成
   `analysis-auto-*.json/.log`和`SHA256SUMS`。

MI300替换由shell `trap`保护；正常退出或中途失败都会尝试恢复MI308原件。仍应在运行后检查：

```bash
sha256sum \
  /sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_rtna_group.co
# 预期：3687c5610a454572e4a615ec58f05e707fdf3995e4dc932cf2219ad2fa0052ff
```

### 快速smoke

smoke只验证三种进程、`.co`切换/恢复、JSON schema和分析链，不复现70次平均值：

```bash
GPU=4 \
H3_SKIP_CORRECTNESS=1 \
ATTN_PROFILE_WARMUP=0 \
ATTN_PROFILE_ITERS=1 \
ATTN_PROFILE_OUTPUT_DIR=/tmp/h3-five-kernel-smoke \
bash tests/flydsl/run_h3_five_kernel_auto.sh
```

### 结果验收

正式JSON应为schema v2、固定实现顺序、每项70条、index `0..69`、event时间和TFLOPS均为正，且每条
`sensor_count>0`。检查产物和活动二进制：

```bash
cd artifacts/h3-five-kernel
sha256sum -c SHA256SUMS
cd ../..

python3 - <<'PY'
import json
from pathlib import Path

expected = {
    "profile-auto-aiter-mi308.json": ["triton", "asm_mi308"],
    "profile-auto-aiter-mi300.json": ["asm_mi300"],
    "profile-auto-flydsl.json": ["8wave_32x32", "4wave_varlen"],
}
for name, implementations in expected.items():
    data = json.loads((Path("artifacts/h3-five-kernel") / name).read_text())
    assert data["schema_version"] == 2
    assert data["warmup"] == 3 and data["iters"] == 70
    assert [result["name"] for result in data["results"]] == implementations
    for result in data["results"]:
        assert [row["index"] for row in result["dispatches"]] == list(range(70))
        assert all(row["elapsed_ms"] > 0 and row["sensor_count"] > 0
                   for row in result["dispatches"])
print("H3 five-kernel profiles validated")
PY
```

复现的是同一实现、输入、采样和分析协议；auto-DPM初始相位、温度和软件版本会使性能值发生小幅变化，
不应要求新JSON与归档文件逐字节相同。

## 产物

原始顺序完整保留：

- AITER Triton + ASM MI308：
  [profile-auto-aiter-mi308.json](../../artifacts/h3-five-kernel/profile-auto-aiter-mi308.json)，
  [profile-auto-aiter-mi308.log](../../artifacts/h3-five-kernel/profile-auto-aiter-mi308.log)
- ASM MI300：
  [profile-auto-aiter-mi300.json](../../artifacts/h3-five-kernel/profile-auto-aiter-mi300.json)，
  [profile-auto-aiter-mi300.log](../../artifacts/h3-five-kernel/profile-auto-aiter-mi300.log)
- FlyDSL 8wave_32x32 + 4wave_varlen：
  [profile-auto-flydsl.json](../../artifacts/h3-five-kernel/profile-auto-flydsl.json)，
  [profile-auto-flydsl.log](../../artifacts/h3-five-kernel/profile-auto-flydsl.log)
- 离线分析：
  [analysis-auto-aiter-mi308.json](../../artifacts/h3-five-kernel/analysis-auto-aiter-mi308.json)，
  [analysis-auto-aiter-mi300.json](../../artifacts/h3-five-kernel/analysis-auto-aiter-mi300.json)，
  [analysis-auto-flydsl.json](../../artifacts/h3-five-kernel/analysis-auto-flydsl.json)
- 校验和：[SHA256SUMS](../../artifacts/h3-five-kernel/SHA256SUMS)

三轮正式测量使用相同 profiler source hash
`f0cef3590acbb72c9d56f87671d94e6501a932e6f9be72b1fa3ad45c6724a75e`。第一轮测量后，分析器仅收紧了
展示逻辑：当 `cycle_detected=False` 时不再把不规则高点间隔称为“周期”；原始数据、burst 检测阈值和
测量代码未变。全部正式结果均用最终 analyzer hash
`4ef90dd7364479828a80f56a8f92dbedcbba2b3d6ee3b523605702b5bb310f00` 重新离线分析。