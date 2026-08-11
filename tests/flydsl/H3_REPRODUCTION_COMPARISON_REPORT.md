# H3 Attention srdc-52 与 srdc-7 对比报告

日期：2026-08-11

本报告对比两台不同服务器上的数据：

- `srdc-52`（`hjbog-srdc-52.amd.com`）：历史报告
  [H3_FIVE_KERNEL_PERFORMANCE_REPORT.md](H3_FIVE_KERNEL_PERFORMANCE_REPORT.md) 和
  [H3_FP8_AUTO_PROFILE_REPORT.md](H3_FP8_AUTO_PROFILE_REPORT.md) 中的数据；
- `srdc-7`（`hjbog-srdc-7.amd.com`）：2026-08-11 全量测试数据。

两台服务器均使用 MI308X，AITER code object 的 SHA256 也相同，但软件栈、驱动、固件和系统设置并不
完全一致。因此这是跨服务器实测对比，不能视为同一服务器上的单变量软件版本消融。

`srdc-7` 测试 BF16 与 FP8 全部 10 个实现。两类测试均执行正确性检查、每项
`3 warmup + 70 dispatch`，并以 1 ms 间隔采样 GPU SCLK、PPT 功耗和温度。全部测试使用 auto DPM，
未锁频，未修改 650 W power cap。

## 第一部分：BF16

### 性能对比

| 实现 | srdc-52 耗时 | srdc-7 耗时 | srdc-52 TFLOPS | srdc-7 TFLOPS | srdc-7 相对吞吐 | srdc-7 CV | srdc-7 稳态 SCLK | 周期 srdc-52 / srdc-7 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8-wave | 158.359 ms | **137.099 ms** | 186.027 | **212.753** | +14.37% | 13.744% | 1226-1714 MHz | 是 / 是 |
| 4-wave varlen | 160.494 ms | 137.983 ms | 182.902 | 210.785 | +15.24% | 12.406% | 1280-1717 MHz | 是 / 是 |
| ASM MI300 | 168.773 ms | 148.513 ms | 172.129 | 193.531 | +12.43% | 5.465% | 1453-1746 MHz | 是 / 是 |
| Triton | 191.049 ms | 183.496 ms | 149.988 | 156.156 | +4.11% | 0.453% | 1764-1802 MHz | 否 / 否 |
| ASM MI308 | 191.377 ms | 192.053 ms | 149.727 | 149.195 | -0.35% | **0.070%** | 1783-1790 MHz | 否 / 否 |

`srdc-7` 的 BF16 排名为 `8-wave > 4-wave > ASM MI300 > Triton > ASM MI308`。相对 Triton，
`srdc-7` 吞吐优势为：
8-wave `+36.24%`、4-wave `+34.98%`、ASM MI300 `+23.93%`。

### srdc-7 GPU 频率数据

稳态范围排除了每个实现最初两个 dispatch 的正常爬频。P05/中值/P95 和标准差基于每次 dispatch 内
1 ms SCLK 样本的算术平均值。

| 实现 | 全程观测范围 | 稳态观测范围 | 稳态均值 P05 / 中值 / P95 | 稳态均值标准差 | 频率点数 | 周期 |
|---|---:|---:|---:|---:|---:|---|
| 8-wave | 1226-1714 MHz | 1226-1714 MHz | 1251.8 / 1347.9 / 1670.1 MHz | 149.8 MHz | 7771 | 是，1.0110 s |
| 4-wave varlen | 1280-1717 MHz | 1280-1717 MHz | 1296.5 / 1409.9 / 1692.7 MHz | 147.7 MHz | 7749 | 是，1.0166 s |
| ASM MI300 | 1440-1746 MHz | 1453-1746 MHz | 1548.0 / 1694.0 / 1738.3 MHz | 71.8 MHz | 8403 | 是，1.0433 s |
| Triton | 1444-1802 MHz | 1764-1802 MHz | 1788.6 / 1790.6 / 1792.4 MHz | 2.2 MHz | 10216 | 否 |
| ASM MI308 | 1442-1790 MHz | 1783-1790 MHz | 1786.2 / 1786.9 / 1788.9 MHz | 0.9 MHz | 10818 | 否 |

频率结论：

1. 8-wave、4-wave 和 ASM MI300 仍有稳定的约 1 秒周期；
2. 三项吞吐与 SCLK 的相关系数为 `0.78-0.80`，峰谷降幅为 `14.66%-28.34%`；
3. 三项 `ppt_accumulated` 增量为 `2606/2993/6076`，稳定的 Triton/MI308 仅为 `5/0`；
4. 频率范围、SCLK 标准差、吞吐 CV 和周期检测同时证明 BF16 周期仍存在。

### srdc-7 正确性

| 实现 | srdc-7 whole cosine | srdc-7 max abs | tail cosine | 结果 |
|---|---:|---:|---:|---|
| Triton | 0.999999940 | 0.000244 | 1.000000119 | 通过 |
| ASM MI308 RTNA | 0.999999881 | 0.000244 | 1.000000119 | 通过 |
| ASM MI300 RTNA | 0.999999881 | 0.000244 | 1.000000119 | 通过 |

RTNE、RTNA、RTZ group/split 路径也全部在既定容差内。

## 第二部分：FP8

### 性能对比

FP8 保持原分组口径：组 A 的 FlyDSL 使用 Q per-token、K/V per-tensor；组 B 的 AITER 使用
Q/K/V per-tensor。跨组数值只用于观察量级。

| 组 | 实现 | srdc-52 耗时 | srdc-7 耗时 | srdc-52 TFLOPS | srdc-7 TFLOPS | srdc-7 相对吞吐 | srdc-7 CV | srdc-7 稳态 SCLK | 周期 srdc-52 / srdc-7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | FlyDSL 8-wave | 88.849 ms | **82.948 ms** | 324.183 | **345.437** | +6.56% | 0.083% | 1715-1776 MHz | 是 / **否** |
| A | FlyDSL 4-wave | 89.014 ms | 83.991 ms | 323.392 | 341.148 | +5.49% | 0.064% | 1712-1775 MHz | 是 / **否** |
| B | ASM MI300 | 90.096 ms | **87.666 ms** | 319.019 | **326.845** | +2.45% | 0.091% | 1714-1770 MHz | 是 / **否** |
| B | ASM MI308 | 111.539 ms | 112.081 ms | 256.903 | 255.649 | -0.49% | **0.053%** | 1754-1791 MHz | 否 / 否 |
| B | Triton | 162.697 ms | 161.945 ms | 176.124 | 176.933 | +0.46% | 0.176% | 1791-1807 MHz | 否 / 否 |

组内结论：

- 组 A 中 8-wave 比 4-wave 吞吐高 `1.26%`；
- 组 B 中 MI300 比 MI308 高 `27.85%`，比 Triton 高 `84.73%`；
- `srdc-7` 的组内排序与 `srdc-52` 一致；
- `srdc-52` 中三个高吞吐实现的约 1 秒周期，在 `srdc-7` 上全部消失。

### srdc-7 GPU 频率数据

| 实现 | 全程观测范围 | 稳态观测范围 | 稳态均值 P05 / 中值 / P95 | 稳态均值标准差 | 频率点数 |
|---|---:|---:|---:|---:|---:|
| FlyDSL 8-wave | 1443-1776 MHz | 1715-1776 MHz | 1773.8 / 1774.6 / 1775.4 MHz | 5.0 MHz | 4684 |
| FlyDSL 4-wave | 1441-1775 MHz | 1712-1775 MHz | 1772.1 / 1774.0 / 1774.8 MHz | 5.4 MHz | 4706 |
| ASM MI300 | 1442-1770 MHz | 1714-1770 MHz | 1766.9 / 1767.6 / 1769.9 MHz | 4.6 MHz | 4914 |
| ASM MI308 | 1444-1791 MHz | 1754-1791 MHz | 1789.0 / 1789.8 / 1790.3 MHz | 2.7 MHz | 6322 |
| Triton | 1445-1807 MHz | 1791-1807 MHz | 1804.3 / 1804.5 / 1805.1 MHz | 0.8 MHz | 9074 |

`srdc-52` 与 `srdc-7` 的稳态频率波动对比：

| 实现 | srdc-52 稳态范围 | srdc-7 稳态范围 | dispatch 均值标准差 srdc-52 / srdc-7 |
|---|---:|---:|---:|
| FlyDSL 8-wave | 1484-1788 MHz | 1715-1776 MHz | 97.8 / 5.0 MHz |
| FlyDSL 4-wave | 1500-1789 MHz | 1712-1775 MHz | 90.0 / 5.4 MHz |
| ASM MI300 | 1527-1786 MHz | 1714-1770 MHz | 68.6 / 4.6 MHz |
| ASM MI308 | 1725-1804 MHz | 1754-1791 MHz | 6.3 / 2.7 MHz |
| Triton | 1723-1819 MHz | 1791-1807 MHz | 6.3 / 0.8 MHz |

频率结论：

1. 五项在 1 ms 密集采样下均为 `cycle_detected=False`；
2. 三个在 `srdc-52` 上循环的高吞吐实现，dispatch 平均 SCLK 标准差在 `srdc-7` 上从
  `68.6-97.8 MHz` 降到 `4.6-5.4 MHz`；
3. 三者稳态 P05-P95 频带仅 `1.6-3.0 MHz`，没有约 1 秒的高低档切换；
4. `srdc-7` 峰谷降幅仅 `0.40%-0.99%`，候选高吞吐区间也没有重复周期；
5. FP8 的约 1 秒周期不是当前 ROCm 7.2.3 栈上的稳定现象。

### srdc-7 正确性

| 组 | 实现 | srdc-7 whole cosine | srdc-7 relative L2 | srdc-7 max abs | 结果 |
|---|---|---:|---:|---:|---|
| A | FlyDSL 8-wave | 0.999739051 | 0.022850 | 0.044922 | 通过 |
| A | FlyDSL 4-wave | 0.999739051 | 0.022850 | 0.044922 | 通过 |
| B | Triton | 0.999745905 | 0.022556 | 0.041521 | 通过 |
| B | ASM MI308 | 0.999739051 | 0.022863 | 0.056641 | 通过 |
| B | ASM MI300 | 0.999739051 | 0.022863 | 0.056641 | 通过 |

FlyDSL 8-wave 与 4-wave 输出逐元素相同；MI308 与 MI300 ASM 的正确性指标也相同。

## 公共测试口径与环境

- 服务器映射：`srdc-52` 为 `hjbog-srdc-52.amd.com`，`srdc-7` 为
  `hjbog-srdc-7.amd.com`；
- shape：segments `(63225,7)`，`Hq=Hkv=14`，`Dq=Dv=128`，seed `1101`；
- 真实 FLOPs：`28,653,368,031,232`；
- 两台服务器的测试目标均为物理 GPU 4、MI308X、`gfx942`、80 CU、650 W；
- 每项 `3 warmup + 70 dispatch`，CUDA event 仅覆盖 kernel；
- BF16 与 FP8 均以 1 ms 采样 SCLK、PPT、junction 和 HBM 温度；
- `srdc-7`：Python 3.12.13、Torch `2.10.0+git8514f05`、HIP `7.2.53211`、ROCm 7.2.3、
  AITER `0.22.1rc1.dev26`、FlyDSL 0.3.0；
- `srdc-52`：Python 3.10、Torch 2.9、ROCm 7.0、AITER 0.1.14、FlyDSL 0.3.0；
- `srdc-7` 的 AITER BF16/FP8 MI308、MI300 `.co` SHA256 均与 `srdc-52` 相同；
- 两台服务器的软件栈、驱动、固件和 `kernel.numa_balancing` 不完全相同，不能把性能或周期差异归因于
  单一组件。

## 数据与复现

`srdc-7` 测试产物：

- BF16：`/tmp/h3-retest-20260811-all/bf16`
- FP8：`/tmp/h3-retest-20260811-all/fp8`

BF16 14 个、FP8 17 个 manifest 文件均通过 `sha256sum -c`。所有 profile 均为 schema v2，
每项 70 条连续 dispatch，且 event 时间、TFLOPS 和传感器数量均为正。测试结束后两个活动 MI308 `.co`
均已恢复，GPU 为 `busy=0`、无 KFD 进程、DPM=`auto`。

BF16：

```bash
GPU=4 \
AITER_ROOT=/app/aiter \
AITER_PYTHON=/usr/bin/python \
FLYDSL_PYTHON=/tmp/h3-repro-flydsl030/bin/python \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_PROFILE_OUTPUT_DIR=/tmp/h3-retest-20260811-all/bf16 \
bash tests/flydsl/run_h3_five_kernel_auto.sh
```

FP8：

```bash
GPU=4 \
AITER_ROOT=/app/aiter \
AITER_PYTHON=/usr/bin/python \
FLYDSL_PYTHON=/tmp/h3-repro-flydsl030/bin/python \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_SENSOR_INTERVAL_MS=1 \
ATTN_FP8_OUTPUT_DIR=/tmp/h3-retest-20260811-all/fp8 \
bash tests/flydsl/run_h3_fp8_report.sh
```

[run_h3_five_kernel_auto.sh](run_h3_five_kernel_auto.sh)、[run_h3_fp8_auto.sh](run_h3_fp8_auto.sh) 和
[run_h3_aiter_fp8_auto.sh](run_h3_aiter_fp8_auto.sh) 均支持 `ATTN_PROFILE_SENSOR_INTERVAL_MS`；
[analyze_h3_attention_throttle.py](analyze_h3_attention_throttle.py) 输出全程/稳态 SCLK 范围、
P05/中值/P95 和采样点数。
