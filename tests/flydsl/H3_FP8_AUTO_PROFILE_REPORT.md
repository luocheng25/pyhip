# H3 FP8 Attention Auto-DPM 综合报告

日期：2026-08-10

## 测试范围与比较口径

所有实现使用同一 H3 逻辑问题：segments `(63225,7)`，`Hq=Hkv=14`，`Dq=Dv=128`，
`causal=False`，seed `1101`，真实 FLOPs 为 `28,653,368,031,232`。正式性能均在物理 GPU 4
的 auto DPM 下采集，每实现预热 3 次，再连续保留 70 个 CUDA-event dispatch；不排序、不截断、
不使用中值作为正式汇总。

五项 FP8 实现分成两个**公平比较组**：

| 组 | 实现 | Q 量化 | K/V 量化 | 可严格组内比较 |
|---|---|---|---|---|
| A：FlyDSL paged | 8-wave、4-wave | E4M3FNUZ per-token | E4M3FNUZ per-tensor | 是 |
| B：AITER linear varlen | Triton、ASM MI308、ASM MI300 | E4M3FNUZ per-tensor | E4M3FNUZ per-tensor | 是 |

Triton/ASM 按 sequence/head 读取 descale，当前入口不支持 FlyDSL 所用的 Q per-token scale，因此两组
Q tensor 并不相同。下面的五实现总表便于观察量级和功率行为；**跨组的 1%-2% 差异不能解释为纯 kernel
差异**。

## 一页总览

| 组 | 实现 | 输出 | 平均耗时 | 平均 TFLOPS | min / max TFLOPS | CV | 峰谷降幅 | 稳定周期 |
|---|---|---|---:|---:|---:|---:|---:|---|
| A | FlyDSL 8-wave paged | BF16 | **88.849 ms** | **324.183** | 286.842 / 343.722 | 7.03% | 16.55% | 11-12 次，1.0099 s |
| A | FlyDSL 4-wave paged | BF16 | 89.014 ms | 323.392 | 286.527 / 339.916 | **6.60%** | **15.71%** | 11-12 次，1.0142 s |
| B | Triton | FP32 | 162.697 ms | 176.124 | 169.037 / 177.692 | 0.74% | 4.87% | 未检测到 |
| B | ASM MI308 `.co` | BF16 | 111.539 ms | 256.903 | 245.584 / 257.612 | **0.66%** | **4.67%** | 未检测到 |
| B | ASM MI300 `.co` | BF16 | **90.096 ms** | **319.019** | 283.178 / 330.317 | 5.35% | 14.27% | 每 11 次，0.9936 s |

### 主要结论

- **组 A**：8-wave 平均延迟比 4-wave 低 `0.185%`，吞吐高 `0.244%`，基本打平；4-wave
  波动略小但平均功耗更高。
- **组 B**：MI300 `.co` 比 MI308 `.co` 平均吞吐高 `24.18%`，比 Triton 高 `81.13%`；
  MI300 的代价是约 1 秒周期，Triton/MI308 本轮较稳定。
- **跨组观察**：MI300 per-tensor 为 `319.019T`，只比 FlyDSL 8-wave/4-wave per-token Q 低
  `1.59%/1.35%`，但差值同时包含量化粒度与 kernel 差异，不能写成 FlyDSL 固有领先。
- 吞吐接近 `319-324T` 的三个实现都出现约 1 秒功率循环；较慢的 Triton/MI308 没有检测到稳定循环。
- FP8 没有消除 DPM/PPT 周期。由于 FP8 kernel 单次约 89-90 ms，一个周期包含约 11 个 dispatch；
  此前 BF16 单次约 158-160 ms，同一时间周期表现为 6-7 个 dispatch。
- CU0 的 Advanced Thread Trace（ATT）进一步确认：三个循环实现只有 `43.9-53.7 stall cycles/MFMA`，
  而不循环的 MI308/Triton 为 `76.7/116.3`；差异来自动态流水线紧实度，不是第二 resident slot 空闲。

## 正确性

两组都从相同 seed 的 BF16 源输入开始，先按各自量化口径生成 FP8，再将量化后 Q/K/V 反量化到 BF16，
逐 segment 执行 BF16 SDPA 作为 reference。63225-token main 和 7-token tail 分开评分。门槛为所有区域
finite、cosine `>0.998`、relative L2 `<0.06`。

### 组 A：Q per-token，K/V per-tensor

| 实现 | whole cosine / rel-L2 | main cosine / rel-L2 | tail cosine / rel-L2 | max abs |
|---|---:|---:|---:|---:|
| FlyDSL 8-wave | 0.999739111 / 0.022850 | 0.999639273 / 0.026883 | 0.999885142 / 0.015163 | 0.044922 |
| FlyDSL 4-wave | 0.999739111 / 0.022850 | 0.999639273 / 0.026883 | 0.999885142 / 0.015163 | 0.044922 |

两套 FlyDSL kernel 的 whole/main/tail 输出逐元素完全相同，`max_abs=0`、relative L2 `=0`。
共享量化 helper 也已与 AITER `pertoken_quant/per_tensor_quant` 比较，FP8 位模式与 descale 逐元素相同。

### 组 B：Q/K/V 全 per-tensor

| 实现 | whole cosine / rel-L2 | main cosine / rel-L2 | tail cosine / rel-L2 | max abs |
|---|---:|---:|---:|---:|
| Triton | 0.999745965 / 0.022556 | 0.999657869 / 0.026194 | 0.999874711 / 0.015829 | 0.041521 |
| ASM MI308 | 0.999739110 / 0.022862 | 0.999643207 / 0.026750 | 0.999879062 / 0.015552 | 0.056641 |
| ASM MI300 | 0.999739110 / 0.022862 | 0.999643207 / 0.026750 | 0.999879062 / 0.015552 | 0.056641 |

三项均通过门槛；MI308 与 MI300 ASM 的正确性指标相同。Triton输出 FP32，ASM 输出 BF16，因此不以
逐元素完全相同作为要求。

## 性能组内比较

### 组 A：FlyDSL 8-wave 与 4-wave

| 指标 | 8-wave | 4-wave | 4-wave 相对 8-wave |
|---|---:|---:|---:|
| 70 次平均耗时 | 88.849 ms | 89.014 ms | +0.185% |
| 70 次平均吞吐 | 324.183T | 323.392T | -0.244% |
| CV | 7.03% | 6.60% | -0.43 pct |
| 平均 SCLK | 1686 MHz | 1695 MHz | +9 MHz |
| 平均功耗 | 541 W | 564 W | +22 W |
| 峰值功耗 | 600 W | 617 W | +17 W |

两项 burst 起点相同：`0,11,23,34,46,57,68`。8-wave 的 6 个完整周期平均为
`88.692-89.258 ms / 322.843-325.089T`；4-wave 为
`88.840-89.492 ms / 321.584-324.145T`。末尾 `68-69` 没有下一个 burst 边界，不纳入周期范围。

两个指定测试文件的内置 `3 warmup + 10 samples + median` 快照如下，仅用于确认入口，不替代 70 次均值：

| 入口 | median | min / max | TFLOPS |
|---|---:|---:|---:|
| `pa_8wave/test_pa_prefill.py` | 86.331 ms | 83.363 / 99.831 ms | 331.901 |
| `pa_4wave/test_pa_prefill.py` | 86.313 ms | 83.839 / 99.889 ms | 331.972 |

短序列 median 比完整周期算术平均乐观约 2%-3%。

### 组 B：Triton 与两套 ASM `.co`

| 指标 | Triton | ASM MI308 | ASM MI300 |
|---|---:|---:|---:|
| 70 次平均耗时 | 162.697 ms | 111.539 ms | **90.096 ms** |
| 70 次平均吞吐 | 176.124T | 256.903T | **319.019T** |
| 相对 Triton吞吐 | 基线 | +45.87% | +81.13% |
| CV | 0.74% | **0.66%** | 5.35% |
| 实际平均 SCLK | 1811 MHz | 1798 MHz | 1720 MHz |
| 平均功耗 | 522 W | 511 W | 553 W |

MI300 burst 起点为 `0,11,22,33,44,55,66`，6 个完整间隔全部为 11 dispatch；周期时间
`0.9931-0.9943 s`，周期平均 `318.396-318.778T`。Triton 和 MI308 的高区间间隔不重复，按同一规则
`cycle_detected=False`。

## 周期与遥测综合

| 实现 | 平均 / 峰值功耗 | 最高 junction / HBM | TFLOPS-SCLK | TFLOPS-power | 高档 SCLK / 功耗 | 低档 SCLK / 功耗 |
|---|---:|---:|---:|---:|---:|---:|
| Triton | 522 / 541 W | 76C / 46C | 0.1848 | -0.0232 | 不适用 | 不适用 |
| ASM MI308 | 511 / 526 W | 80C / 46C | 0.1742 | 0.0221 | 不适用 | 不适用 |
| ASM MI300 | 553 / 601 W | 88C / 46C | 0.6863 | 0.4923 | 1741 MHz / 562 W | 1648 MHz / 524 W |
| FlyDSL 8-wave | 541 / 600 W | 84C / 47C | 0.7642 | 0.6250 | 1736 MHz / 563 W | 1597 MHz / 502 W |
| FlyDSL 4-wave | 564 / 617 W | 85C / 47C | 0.7138 | 0.5927 | 1738 MHz / 584 W | 1602 MHz / 519 W |

三个循环实现都显示高档比低档有更高 SCLK 和功耗，且吞吐与 SCLK/功耗明显正相关。MI300 junction
达到 88C，但没有循环的 MI308 也达到 80C；本机 AMDSMI 的 PPT/PROCHOT/thermal accumulated counter
仍返回 `N/A`，因此只能确认频率、功耗、吞吐同步变化，不能给出硬件 violation 次数。

## ATT 指令级确认

为检验“稳定高频实现是否只是静态 occupancy 较低”以及“循环实现是否具有更紧实的动态流水线”，
对五项 FP8 kernel 各采集一次 `gfx942` CU0 Advanced Thread Trace。AITER/Triton 覆盖 SIMD0-3、slot0-1；
FlyDSL 的四 SIMD 原始 trace 超出 decoder 可处理范围，因此使用 512 MiB buffer 重采 SIMD0、slot0-1。
分析器要求每个 wave 的 `num_stitched == num_insts`；下表所用 1736 个 wave 全部通过，失败的四 SIMD
FlyDSL trace 不参与结果。

`Stall/MFMA` 与 `Idle/MFMA` 是 decoder 指令统计的累计 cycle 除以 FP8 MFMA hit 数；`issue density`
是每 SIMD 的 decoded dynamic instructions / wave-active cycles，再取 SIMD 中值。FlyDSL 只采一个 SIMD，
所以绝对 wave 数和 wave 时长不能与四 SIMD trace 横比；表中的每 MFMA 归一值、单 SIMD issue density
和同一 SIMD 的 slot overlap 才是比较口径。

| 实现 | 遥测分类 | ATT 范围 / 完整 wave | Inst/MFMA | Stall/MFMA | Idle/MFMA | Stall+Idle/MFMA | issue density | 双 slot overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Triton | 不循环 | SIMD0-3 / 348 | 18.718 | 116.319 | 6.063 | 122.382 | 0.088910 | 99.915% |
| ASM MI308 | 不循环 | SIMD0-3 / 712 | 8.499 | 76.734 | 1.954 | 78.689 | 0.067022 | 99.921% |
| ASM MI300 | 约 1 s 循环 | SIMD0-3 / 672 | 9.453 | **43.892** | 2.326 | **46.217** | **0.096445** | 99.941% |
| FlyDSL 8-wave | 约 1 s 循环 | SIMD0 / 2 | 9.559 | 53.383 | 11.224 | 64.608 | 0.088721 | 100.000% |
| FlyDSL 4-wave | 约 1 s 循环 | SIMD0 / 2 | 9.808 | 53.680 | 10.810 | 64.490 | 0.090402 | 100.000% |

循环组的 `Stall/MFMA` 完整落在 `43.9-53.7`，稳定组则为 `76.7-116.3`。相对稳定的 MI308，
MI300/8-wave/4-wave 分别低 `42.8%/30.4%/30.0%`；五项的双 slot overlap 均约 100%，因此可以
排除“采样 CU 没有第二 resident slot”这一简单静态驻留解释。Triton 的 raw issue density 虽接近 FlyDSL，
但每 MFMA 需要 `18.718` 条动态指令，约为其余实现的两倍，并承担最高的归一 stall，不能把 raw issue
density 单独解释为有效计算密度。

MI300/MI308 是最强的受控对照：两者使用同一 AITER symbol、相同 kernel resources、launch geometry、
输入、scale 与输出口径，只替换 code object。MI300 相对 MI308 的 ATT 变化如下：

| 指标 | ASM MI308 | ASM MI300 | MI300 变化 |
|---|---:|---:|---:|
| 总 Stall/MFMA | 76.734 | 43.892 | **-42.8%** |
| Stall+Idle/MFMA | 78.689 | 46.217 | **-41.3%** |
| issue density | 0.067022 | 0.096445 | **+43.9%** |
| median wave duration | 4,002,288 cycles | 3,093,882 cycles | **-22.7%** |
| MFMA stall/MFMA | 23.036 | 2.318 | -89.9% |
| VALU stall/MFMA | 13.594 | 2.124 | -84.4% |
| EXP stall/MFMA | 2.414 | 0.309 | -87.2% |
| waitcnt stall/MFMA | 6.587 | 4.101 | -37.7% |

MI300 的 barrier stall 从 `18.045` 增至 `23.129/MFMA`、SALU/branch stall 从 `2.828` 增至
`5.519/MFMA`，但被 MFMA/VALU/EXP 与 memory-wait 的大幅下降覆盖，故总 stall 仍低 42.8%。这不是
“所有 stall 都变少”的笼统结论，而是 code-object 调度改变了 stall 构成并显著压缩了总执行空隙。

Triton 的 `116.319 stall/MFMA` 主要来自 `waitcnt 42.646`（36.7%）、`VALU 30.987`（26.6%）和
`MFMA 23.931`（20.6%），三类合计 83.9%；最大单项是 `s_waitcnt vmcnt(0)`。这与其 162.7 ms 延迟、
522 W 平均功耗和没有稳定 DPM 周期相符：波前持续驻留，但大量 cycle 在等待或执行额外标量/向量工作，
单位有效 MFMA 的持续功率密度低于 MI300/FlyDSL。

因此，ATT 确认的是**执行机制**：MI300/FlyDSL 用更少的归一 stall、更密集的有效 issue 将相同 H3 工作
压入约 90 ms，持续功率升入 PPT/DPM 控制区；Triton/MI308 虽保持较高 SCLK，却因等待和额外指令较多，
没有进入同一周期区。约 1 秒的控制周期、SCLK 与功耗同步变化仍由 10 ms 遥测证明；ATT 本身不读取
PPT violation，也不能替代缺失的 AMDSMI accumulated counter，因此不把它表述为直接观测到硬件阈值。

## FP8 与历史 BF16 Auto 基线

| FP8 实现 | FP8 平均 | 历史 BF16 实现 | BF16 平均 | 变化 |
|---|---:|---|---:|---:|
| Triton FP8 per-tensor | 176.124T | Triton BF16 | 149.988T | +17.43% |
| ASM MI308 FP8 per-tensor | 256.903T | ASM MI308 BF16 | 149.727T | +71.58% |
| ASM MI300 FP8 per-tensor | 319.019T | ASM MI300 BF16 | 172.129T | +85.34% |
| FlyDSL 8-wave FP8 per-token Q | 324.183T | 8wave_32x32 dense BF16 | 186.027T | +74.27% |
| FlyDSL 4-wave FP8 per-token Q | 323.392T | 4wave_varlen BF16 | 182.902T | +76.81% |

该表仅反映同一 H3 FLOPs 口径下的历史量级，不是严格 dtype 消融：FP8 使用当前 commit
`aa324e1c752c0cef7edb1c1b73af25d4111869d6`，BF16 归档来自前一 commit `438e9fa`；此外 8-wave BF16
是 dense 近似而 FP8 是 paged varlen。不能把全部变化都归因于 FP8 数据类型。

## 支持路径与二进制身份

| 实现 | 实际入口 / 支持证据 |
|---|---|
| Triton | 低层 `_flash_attn_forward`；`_attn_fwd` 显式处理 `IS_FP8` 与 `descale_q/k/v` |
| ASM MI308 / MI300 | 低层 `_flash_attn_varlen_forward`；配置含 `fp8bf16,D128,group=1` |
| FlyDSL | 两个 paged MHA callable，Q per-token、K/V per-tensor scale |

AITER 的公开高层 `flash_attn_varlen_func` 没有完整暴露 FP8 descale，因此本测试使用已安装模块中的低层
forward 显式传 scale，不能省略 descale 后宣称支持。

ASM code object：

- MI308 `fwd_hd128_fp8_group.co`：
  `5a9cfe058a455734e8ac46e740f250631b0396eb785df8e5ab2b8df2ceacbe2e`
- MI300 `fwd_hd128_fp8_group.co`：
  `5e5b4b6891c600a0051ca0ebb3c14f415be7db8f3aa9607a18f057e244d65575`

MI300 `.co` 在独立进程前临时放入 AITER 按 PCI ID 选择的 MI308 活动路径；profiler 同时校验 active MI300
hash 与备份 MI308 hash，退出 trap 无条件恢复。最终活动文件已恢复为 MI308 hash。

## 环境与完整性

- GPU：物理 GPU 4，MI308X，`gfx942`，80 CU，650 W cap；
- DPM：全程 `auto`，没有设置 determinism；
- shape/FLOPs：五项完全相同；组内输入 tensor 和 scale 完全相同；
- 正式开始和结束：无 KFD 进程、busy=0、约 284-296 MiB 驱动保留显存、DPM=`auto`；
- 所有正式 JSON 为 schema v2，每项恰好 70 条、index `0..69`、全部 `sensor_count>0`；
- AMDSMI throttle counters 为 `N/A`，报告不伪造 PPT/thermal violation 次数。

## 复现

### 前提

- 从仓库根目录运行，使用报告“环境与完整性”中的ROCm/Torch/AITER环境。默认AITER解释器为
  `/opt/venv/bin/python3`，默认物理GPU为4，可用`GPU=<index>`覆盖。
- AITER安装目录必须同时包含MI308/MI300 FP8 group code object，且SHA256与“支持路径与二进制身份”
  一节一致。wrapper会在修改活动路径前验证源文件和备份。
- profiler会要求目标GPU在CUDA初始化前busy为0、无其他KFD进程、初始显存不超过1 GiB且DPM为
  `auto`。
- FlyDSL paged组需要0.3.0。与BF16报告共用同一隔离环境：

```bash
/opt/venv/bin/python3 -m venv --system-site-packages \
  artifacts/h3-five-kernel/venv-flydsl030
artifacts/h3-five-kernel/venv-flydsl030/bin/python -m pip install flydsl==0.3.0
artifacts/h3-five-kernel/venv-flydsl030/bin/python -m pip install -e .
```

### 相关代码

| 文件 | 职责 |
|---|---|
| [run_h3_fp8_report.sh](run_h3_fp8_report.sh) | 串联FlyDSL组、AITER组并生成统一manifest |
| [run_h3_fp8_auto.sh](run_h3_fp8_auto.sh) | FlyDSL 8/4-wave正确性、性能和BF16历史对比 |
| [run_h3_aiter_fp8_auto.sh](run_h3_aiter_fp8_auto.sh) | Triton/ASM正确性、性能和MI300 `.co`安全切换 |
| [h3_profile_common.sh](h3_profile_common.sh) | 在任何CUDA初始化前执行GPU空闲/DPM预检 |
| [h3_paged_inputs.py](h3_paged_inputs.py) | 两组共享H3输入、量化、反量化reference和比较指标 |
| [h3_aiter_fp8.py](h3_aiter_fp8.py) | 显式descale的AITER Triton/ASM FP8 launcher |
| [check_h3_fp8_paged.py](check_h3_fp8_paged.py) | FlyDSL组whole/main/tail正确性 |
| [check_h3_aiter_fp8.py](check_h3_aiter_fp8.py) | AITER组whole/main/tail正确性 |
| [profile_h3_attention_throttle.py](profile_h3_attention_throttle.py) | 五项逐dispatch性能和遥测采集 |
| [analyze_h3_attention_throttle.py](analyze_h3_attention_throttle.py) | DPM周期和完整周期分析 |
| [run_h3_fp8_att.sh](run_h3_fp8_att.sh) | 五项ATT采集、完整性检查和摘要生成 |
| [run_h3_fp8_att_target.py](run_h3_fp8_att_target.py) | 每次rocprof进程只发射一个目标dispatch |
| [analyze_h3_fp8_att.py](analyze_h3_fp8_att.py) | wave完整性、stall/MFMA、issue density和slot overlap |

### 正式性能与正确性

一条命令复现两组正确性、三轮`3 warmup + 70 dispatch`性能采集、离线分析、BF16历史对比和校验和：

```bash
bash tests/flydsl/run_h3_fp8_report.sh
```

[run_h3_fp8_report.sh](run_h3_fp8_report.sh)依次调用：

1. [run_h3_fp8_auto.sh](run_h3_fp8_auto.sh)：共享Q per-token、K/V per-tensor输入，检查FlyDSL
   8-wave/4-wave正确性并采集两项性能。
2. [run_h3_aiter_fp8_auto.sh](run_h3_aiter_fp8_auto.sh)：共享Q/K/V per-tensor输入，运行
   [check_h3_aiter_fp8.py](check_h3_aiter_fp8.py)和Triton/ASM MI308采集；随后临时切换MI300 `.co`，
   重跑正确性与ASM MI300采集，最后恢复MI308。
3. 对三份profile运行[analyze_h3_attention_throttle.py](analyze_h3_attention_throttle.py)，并用
   [analyze_h3_fp8_auto.py](analyze_h3_fp8_auto.py)生成FP8/BF16历史量级对比。

MI300替换由shell `trap`保护。运行后验证活动文件已恢复：

```bash
sha256sum \
  /sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_fp8_group.co
# 预期：5a9cfe058a455734e8ac46e740f250631b0396eb785df8e5ab2b8df2ceacbe2e
```

### 快速smoke

以下命令跳过reference并把每项减为1次dispatch，只验证入口、`.co`恢复、schema和分析链：

```bash
GPU=4 \
H3_SKIP_CORRECTNESS=1 \
ATTN_PROFILE_WARMUP=0 \
ATTN_PROFILE_ITERS=1 \
ATTN_FP8_OUTPUT_DIR=/tmp/h3-fp8-auto-smoke \
bash tests/flydsl/run_h3_fp8_report.sh
```

### ATT指令级复现

先按[GPU profiling说明](../../docs/profile-gpu.md)安装ROCprof trace decoder 0.1.6。不要提交decoder
安装包、解压目录或原始ATT；五条trace需要数GiB临时空间。若decoder已安装到`/opt/rocm/lib`：

```bash
GPU=4 \
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib \
H3_ATT_TRACE_ROOT=/tmp/h3-fp8-att-repro \
bash tests/flydsl/run_h3_fp8_att.sh
```

若decoder位于其他目录，将`ROCPROF_ATT_LIBRARY_PATH`设为包含
`librocprof-trace-decoder.so`的目录。wrapper采集Triton、ASM MI308、ASM MI300和两项FlyDSL：

- AITER/Triton：CU0、SIMD0-3、slot0-1、约2 GiB ATT buffer；
- FlyDSL：CU0、SIMD0、slot0-1、512 MiB buffer；
- 每条capture log不得出现`Stitch Incomplete`、`Wave incomplete`、cutoff或parser mismatch；
- [analyze_h3_fp8_att.py](analyze_h3_fp8_att.py)还会要求每个wave满足
  `num_stitched == num_insts`，只把约48 KiB的摘要写入`artifacts/h3-fp8-att/`。

### 结果验收

```bash
cd artifacts/h3-fp8-auto
sha256sum -c SHA256SUMS
cd ../h3-fp8-att
sha256sum -c SHA256SUMS
cd ../..

python3 - <<'PY'
import json
from pathlib import Path

expected = {
    "profile-auto-fp8.json": ["8wave_varlen_fp8", "4wave_varlen_fp8"],
    "profile-auto-aiter-mi308-fp8.json": ["triton_fp8", "asm_mi308_fp8"],
    "profile-auto-aiter-mi300-fp8.json": ["asm_mi300_fp8"],
}
for name, implementations in expected.items():
    data = json.loads((Path("artifacts/h3-fp8-auto") / name).read_text())
    assert data["schema_version"] == 2
    assert data["warmup"] == 3 and data["iters"] == 70
    assert [result["name"] for result in data["results"]] == implementations
    for result in data["results"]:
        assert [row["index"] for row in result["dispatches"]] == list(range(70))
        assert all(row["elapsed_ms"] > 0 and row["sensor_count"] > 0
                   for row in result["dispatches"])
print("H3 FP8 profiles validated")
PY
```

复现的是同一实现、量化口径、输入、采样和分析协议；auto-DPM初始相位、温度和软件版本会造成小幅
变化，不应要求新JSON与归档文件逐字节相同。

## 产物

- FlyDSL：[profile-auto-fp8.json](../../artifacts/h3-fp8-auto/profile-auto-fp8.json)，
  [analysis-auto-fp8.json](../../artifacts/h3-fp8-auto/analysis-auto-fp8.json)，
  [correctness-fp8.log](../../artifacts/h3-fp8-auto/correctness-fp8.log)
- Triton + MI308：
  [profile-auto-aiter-mi308-fp8.json](../../artifacts/h3-fp8-auto/profile-auto-aiter-mi308-fp8.json)，
  [analysis-auto-aiter-mi308-fp8.json](../../artifacts/h3-fp8-auto/analysis-auto-aiter-mi308-fp8.json)
- MI300：
  [profile-auto-aiter-mi300-fp8.json](../../artifacts/h3-fp8-auto/profile-auto-aiter-mi300-fp8.json)，
  [analysis-auto-aiter-mi300-fp8.json](../../artifacts/h3-fp8-auto/analysis-auto-aiter-mi300-fp8.json)
- 支持与正确性 probe：
  [aiter-fp8-probe-mi308.log](../../artifacts/h3-fp8-auto/aiter-fp8-probe-mi308.log)，
  [aiter-fp8-probe-mi300.log](../../artifacts/h3-fp8-auto/aiter-fp8-probe-mi300.log)
- 历史 BF16 对比：[fp8-vs-bf16.json](../../artifacts/h3-fp8-auto/fp8-vs-bf16.json)
- ATT 指令级摘要：[analysis.json](../../artifacts/h3-fp8-att/analysis.json)，
  [analysis.log](../../artifacts/h3-fp8-att/analysis.log)，
  [SHA256SUMS](../../artifacts/h3-fp8-att/SHA256SUMS)；单 dispatch 入口与分析器分别为
  [run_h3_fp8_att_target.py](run_h3_fp8_att_target.py) 和 [analyze_h3_fp8_att.py](analyze_h3_fp8_att.py)
- 全部校验和：[SHA256SUMS](../../artifacts/h3-fp8-auto/SHA256SUMS)
