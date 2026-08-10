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

## 复现与产物

FlyDSL per-token Q 组：

```bash
bash tests/flydsl/run_h3_fp8_auto.sh
```

AITER per-tensor Q/K/V 组：

```bash
bash tests/flydsl/run_h3_aiter_fp8_auto.sh
```

主要产物：

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
- 全部校验和：[SHA256SUMS](../../artifacts/h3-fp8-auto/SHA256SUMS)
