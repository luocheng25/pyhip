# gfx942严格8-wave反相：短MFMA与VMEM FIFO

[`probe-8wave-vmem-antiphase.py`](probe-8wave-vmem-antiphase.py)构造一个严格8-wave反相微基准：

- 每CU一个8-wave workgroup，固定64 KiB LDS；
- 每个物理SIMD恰有两个wave，硬件slot为`(0, 1)`；
- 条件入口/排空barrier让两个wave相差一个stage；
- stage 0连续发起non-temporal HBM `buffer_load_dwordx4`，随后`vmcnt(0)`并消费数据；
- stage 1只包含BF16 MFMA；
- 每个stage末尾使用真实barrier，不用`s_setprio`。

所有正式ATT均为32个物理SIMD pair、0 placement failure、0 trace failure，活动同stage重叠严格为0。
因此下面的stall来自已知反相下memory/compute时长不匹配，不是相位漂移。

## ATT口径

ATT记录为：

```text
[first_attempt, category, stall, duration, pc_index]
```

成功issue时刻为：

$$
t_{\mathrm{issue}}=t_{\mathrm{first\ attempt}}+t_{\mathrm{stall}}
$$

本工具区分：

- **VMEM-load issue stall：** `buffer_load_dwordx4`从首次尝试到成功发射的stall，反映VMEM issue pipe、
  request credit或FIFO backpressure；
- **wait-vmcnt stall：** `s_waitcnt vmcnt(0)`等待已发请求完成，反映consumer-side数据未返回；
- **compute barrier stall：** compute wave已完成MFMA，在barrier等待memory peer；
- **memory barrier stall：** memory wave已完成load/wait/consume，在barrier等待compute peer。

工具还计算`wait-vmcnt`或load-issue stall与peer compute-barrier stall的时间交集。严格反相下，该交集是
memory stall暴露到物理SIMD生命周期的直接见证。

## 实验环境

MI308X `gfx942`，GPU4，80 CU，ROCm 7.2.3，rocprofv3 1.1.0，650 W power cap，PTL
`VECTOR,F8`，performance determinism 1800 MHz。每个case为128 rounds，捕获两个dispatch；每次ATT选择
CU0、四个shader engine和四个SIMD。

资源：30 SGPR，64 KiB LDS，无scratch；VGPR随每wave outstanding load数从28增长到104，但编译器报告的
occupancy均为4 waves/SIMD，实际由64 KiB LDS限制为一个8-wave WG/CU。

## 1. MFMA不能遮盖HBM延迟

固定每memory stage一条HBM load，扫描16/32/64条MFMA：

| MFMA数 | MFMA issue span | load到`vmcnt`成功 | `vmcnt` stall | compute barrier stall | wait与peer barrier重叠比例 |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 cycles | 724 cycles | 708 cycles | 580 cycles | 76.36% |
| 32 | 512 cycles | 776 cycles | 760 cycles | 378 cycles | 46.21% |
| 64 | 1024 cycles | 690 cycles | 674 cycles | 8 cycles | 0.47% |

每条load自身的issue stall均值只有约18 cycles，占`load issue + vmcnt wait`的约2.2%--2.5%。因此该组
主要不是请求发不出去，而是请求已发出但数据尚未返回。

### 典型stall特征

**16条MFMA：**

- compute窗口只有256 cycles，比HBM可用时间短约468 cycles；
- compute wave先到barrier，中位stall 580 cycles；
- memory wave在`vmcnt(0)`中位stall 708 cycles；
- 约76%的wait stall与peer compute-barrier stall重叠；
- compute barrier的约92.4%可由该重叠解释。

**32条MFMA：**

- compute窗口512 cycles，仍比HBM可用时间短约264 cycles；
- compute barrier中位stall降至378 cycles；
- wait/peer-barrier重叠比例降至46.2%，但尾部仍明显暴露。

**64条MFMA：**

- compute窗口1024 cycles，超过本轮HBM可用时间；
- compute barrier只剩8 cycles基线，wait与peer barrier几乎不重叠；
- memory wave反而在memory barrier等待compute wave约256--276 cycles。

所以短compute的典型ATT签名是：

```text
memory wave: stall:wait-vmcnt
compute peer: stall:s_barrier
二者在绝对时间上大面积重叠
```

而不是显著的`stall:VMEM-load`。

## 2. VMEM FIFO/credit不足

固定128条MFMA（2048-cycle compute窗口），扫描每memory wave的load数。严格反相时每CU同时有4条
memory wave，因此每CU名义请求数为`4 * loads_per_wave`：

| 每wave load | 名义请求/CU | load issue span | load issue stall均值 | `vmcnt` stall中位 | issue stall占issue+wait |
|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 312 cycles | 24.42 cycles | 1242 cycles | 13.51% |
| 12 | 48 | 580 cycles | 31.36 cycles | 1298 cycles | 22.30% |
| 16 | 64 | 1160 cycles | 52.43 cycles | 1282 cycles | 39.42% |
| 20 | 80 | 1568 cycles | 58.52 cycles | 1486 cycles | 44.01% |

48请求以内，前12个load的issue stall均值约31 cycles。超过第12个load后：

- 16-load case的post-boundary均值为115.80 cycles，是pre-boundary的3.70倍；
- 20-load case的post-boundary均值为100.07 cycles，是pre-boundary的3.25倍；
- 第13/14个load最突出，也就是名义第52/56个CU请求；
- 16-load case第14个load平均stall约172 cycles，p95达到388--556 cycles；
- 20-load case第14个load平均stall约191 cycles，p95达到356--592 cycles。

### 典型stall特征

FIFO/credit压力的ATT签名与纯延迟不同：

```text
连续buffer_load中的后段load本身出现长stall
load issue span显著拉长
stall预算从wait-vmcnt转移到VMEM-load issue
```

从32到80个名义请求，load issue span由312增长到1568 cycles，load-issue stall预算占比从13.5%增长到
44.0%。这会阻塞同wave后续本可独立发射的指令，即使MFMA总量足够长，也不能把producer-side issue
backpressure简单视为普通HBM latency。

### 关于“48-entry”的边界

本实验把48 requests/CU作为待验证假设，而不是硬编码硬件规格。实测结论是：

- 12 loads/wave，即名义48 requests/CU时，未出现长尾陡升；
- 第13/14个load，即名义52/56 requests/CU时，stall明显跳变；
- 因此数据支持“可用credit约在48附近，超过后出现backpressure”，但不能仅凭ATT把物理FIFO精确宣称为
  48 entries。请求完成、四条memory wave的交错和trace采样粒度都会让拐点晚于名义边界。

## 两类问题的区别

| 特征 | compute太短 | FIFO/credit不足 |
|---|---|---|
| 主要stall位置 | `s_waitcnt vmcnt(0)` | 后段`buffer_load_dwordx4` |
| 请求状态 | 已成功发射，等待数据返回 | 请求尚未成功issue |
| peer表现 | compute wave长时间等barrier | issue span拉长后，wait和barrier也可能继续暴露 |
| 解决方向 | 更早预取或增加独立compute距离 | 限制outstanding请求、分批发射或穿插足够issue间隔 |

严格8-wave反相能用另一条wave遮盖部分stall，但不能消除同一个memory stage的producer credit或数据依赖。
16/32条MFMA不足时暴露consumer tail；超过约48个CU累计请求时又暴露producer backpressure。这两类stall
必须分别优化。

## 复现

ATT配置为[`strict8-vmem-antiphase-att.yaml`](strict8-vmem-antiphase-att.yaml)。例如短compute：

```bash
rm -rf /tmp/strict8-vmem-att
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
rocprofv3 -i tests/flydsl/attn_4wave/tools/strict8-vmem-antiphase-att.yaml \
  --att-library-path /tmp/h3-rocprof-decoder/opt/rocm/lib -- \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-8wave-vmem-antiphase.py run \
  --device 0 --rounds 128 --loads-per-stage 1 --mfmas-per-stage 16 \
  --data-mib 512 --dispatches 2 \
  --json /tmp/strict8-vmem-placement.json \
  > /tmp/strict8-vmem-capture.log 2>&1

/tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-8wave-vmem-antiphase.py analyze \
  --att-root /tmp/strict8-vmem-att --rounds 128 \
  --loads-per-stage 1 --mfmas-per-stage 16 \
  --capture-log /tmp/strict8-vmem-capture.log \
  --json /tmp/strict8-vmem-analysis.json
```

FIFO case只需改为例如：

```text
--loads-per-stage 16 --mfmas-per-stage 128
```

`--clock-mhz`不是本分析器必需项；所有stall均直接使用ATT shader cycles。本轮正式采集结束后，GPU4已恢复
到原始`auto + F8,VECTOR`状态。
