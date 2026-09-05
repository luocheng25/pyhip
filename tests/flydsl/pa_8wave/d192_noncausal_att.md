# D192 noncausal：ATT 分析与 persistent 状态

## 后续：packed-V 已合入并完成控制/等待精简

2026-09-05 后续实现已完成：packed-V 单独完整回归 **173 passed、6 skipped**，之后合入
同位置 wait 合并、noncausal border 简化和正向完整 V 页的检查裁剪，最终
**181 passed、6 skipped**。保留 reverse/epilogue NaN-tail 清零和所有 stage barrier。
本轮同进程 D192 noncausal 为 **303.689 → 299.090 → 296.688 µs**（改动前 / packed-only /
当前），OPUS linear **291.950 µs**。最终 ATT 的 BFI **656→0/wave**、waitcnt **218→179/wave**；
没有加入 persistent。最新协议、六形状采样、完整回归、新资源和 ATT 见
[packed_v_control_results.json](packed_v_control_results.json) 与 [主报告](../pa_4wave/README.md)。

---

## 原始诊断记录（以下为合入前数据）

2026-09-05。以下分析对象为合入前的 direct-only 版本，不是更早的 gather/旧 8-wave 实现。
**当时未修改生产 attention 内核**；新增采集/分析脚本并做进程内隔离实验。
原始摘要、每次 capture、PMC、计时样本、source hash 及完整 trace 目录索引见
[d192_noncausal_att_results.json](d192_noncausal_att_results.json)。

## 结论

1. **当前 8-wave 不支持 persistent 调度。** Noncausal 每 CTA 一个 Q block；causal
   首尾配对最多两个 Q block，不是工作队列。对比的 OPUS D192 group 也不是 persistent，
   所以这里不能将性能差归因于“OPUS 有 persist、FlyDSL 没有”。
2. 相同 Q10240/KV2583、D192/V128、Hq16/Hkv1、BF16 的确认轮：原始 direct
   **302.930 µs**，显式 OPUS linear **291.796 µs**，差 **11.134 µs / 3.82%**。
   OPUS 的等价 linear KV 预先准备，转换不计时；它不是 5D 分页端到端时间。
3. ATT 定位到 **V fragment 的 BF16 拼接/位转换生成了额外 BFI**，而非额外矩阵运算。
   保留页表、mask、NaN-tail 保护和所有 barrier，只将 V 拼接中间表示改为 packed i32
   的隔离实验，消除了 **656 条 BFI/wave**，同轮延迟变为 **298.784 µs**：降低
   **4.146 µs / 1.37%**，缩小约 **37%** 的本轮 OPUS 差距。仍比 OPUS 慢 **2.39%**。
4. 其余差异集中在分页地址/控制、tail 保护和 softmax/rescale 周围的流水间隙。
   两边 MFMA 工作量相同、LDS bank conflict 都为 0、scratch 都为 0。不能从单个
   `s_barrier` 的累计 stall 直接声称还能省多少全局 kernel 时间。

## 采集与证据范围

- MI350X gfx950，**256 CU、4 SIMD/CU、160 KiB LDS/CU**。Q/K/V 来自同一个 seed 和
  相同逻辑输入，物理页随机排列，性能尾页统一补零；无 LSE、预分配输出。
- [profile_noncausal_att.py](profile_noncausal_att.py) 分别调用公开 direct 入口和
  显式 `fmha_fwd_bf16_opus_varlen_fwd`，共 23 次，ATT 选取 **21–23** 次。
  采集后独立 FP32 reference 检查通过；两边最大绝对误差均为 **0.0009248**。
- ROCm 7.2，rocprofv3 1.1.0，decoder 0.1.6，所有 SIMD，256 MiB ATT buffer。
  分别采 CU1 与 CU2、每 CU 每候选 3 次：原始 FlyDSL **120 个完整 wave / 15 个 CTA**，
  OPUS **136 个完整 wave / 17 个 CTA**。每 wave 的 `num_stitched == num_insts`。
- 首轮申请 SE mask `0x3`，虽然存在两个 SE 的 raw ATT/occupancy，**指令级 wave JSON
  只有 SE0**；第二轮在 SE0 CU2 复核。不把这些数据描述成全卡/两个 SE 的指令覆盖。
- [analyze_noncausal_att.py](analyze_noncausal_att.py) 校验每个 CTA cohort 的 8 个 wave、
  两个 slot × 四个 SIMD、接近的起始时间，以及 **333 个 barrier 的完成时间对齐**。
  稳态取逻辑 phase 2–39，排除首个 phase 和尾页 phase；比较物理 barrier 间隔，
  每个间隔同时包含 stagger/nonstagger 两组不同的工作。
- ATT gfx9 的 `Latency = Stall + Issue`，**不是 MFMA 执行时长**。分析脚本另列的
  32-cycle MFMA interval union 是明确标记的模型，不是硬件 MFMA utilization counter。
  每 wave stall 相互重叠，不能简单相加作为 kernel-time 分解。ATT/PMC 下的 dispatch
  时间也没有拿来替代正常 benchmark 时间。

## 资源与全卡工作量：不是 occupancy/spill/bank-conflict 回退

| 项目 | FlyDSL direct | OPUS D192 group |
|---|---:|---:|
| CTA grid | 16 × 1 × 40 = 640 | 16 × 1 × 40 = 640 |
| threads / CTA | 512 | 512 |
| LDS / CTA（ELF） | 149760 B | 149760 B |
| VGPR / SGPR（ELF） | 256 / 77 | 250 / 70 |
| private/scratch | 0 | 0 |
| LDS 上限决定的 resident CTA/CU | 1 | 1 |
| 已采 CTA 内 resident waves/SIMD | 2 | 2 |
| `SQ_WAVES` | 5120 | 5120 |
| `SQ_INSTS_MFMA` | 8396800 | 8396800 |
| `SQ_INSTS_LDS` | 8458240 | 11816960 |
| `SQ_INSTS_SMEM` | 312320 | 76800 |
| `SQ_LDS_BANK_CONFLICT` | **0** | **0** |

PMC 是另行采集的全 grid 数据，两边各 3 次，以上计数每次相同。寄存器取真实
ELF/ISA metadata；本机 rocprofv3 CSV 的 `VGPR_Count=128/SGPR_Count=112` 不应当作
真实编译寄存器数，LDS 也会按分配粒度显示成 150016 B。

FlyDSL 的 V 使用 b128 LDS read，OPUS 使用 b64 transpose read，所以 FlyDSL 的 LDS
**指令数更少**，并无银行冲突。两边每 wave 都有 **1640 条 BF16 32×32×16 MFMA**、
**223 条全局 dwordx4 load**、**333 条 barrier**。同样的全局 load 指令数不等价于
同样的 HBM 流量，未据此声称 HBM/cache 行为完全相同。

## 稳态流水：差距落在哪里

以下是 CU1/CU2 共 6 次 capture 的等权均值，单位 **cycles / KV tile phase**。
每个单项是实际对齐的物理 barrier 间隔，不是某个 source 函数的独占时间。

| nonstagger / stagger 同时执行的 stage | FlyDSL | OPUS | 差值 |
|---|---:|---:|---:|
| S0 K-read / 前一 phase S7 | 297.0 | 273.1 | +23.9 |
| S1 QK+exp / S0 K-read | 398.5 | 376.2 | +22.3 |
| S2 K-read+page-ready / S1 QK+exp | 388.0 | 372.0 | +16.0 |
| S3 QK+exp/sum/P / S2 K-read | 412.0 | 412.7 | -0.6 |
| S4 V-read+K-DMA / S3 QK+exp/sum/P | 428.8 | 404.5 | +24.3 |
| S5 PV+max / S4 V-read+K-DMA | 320.8 | 288.0 | +32.8 |
| S6 V-read / S5 PV+max | 373.9 | 324.0 | **+49.9** |
| S7 PV+scale/rescale / S6 V-read | 294.6 | 275.7 | +19.0 |
| **合计** | **2913.6** | **2726.1** | **+187.5 / 6.88%** |

CU1 单独为 **2913.3 / 2726.3**，CU2 为 **2913.8 / 2726.0**，方向稳定。
这是采样 CTA 的稳态差，不是全 kernel 的 6.88% 减速承诺；正常多候选 benchmark
差为 3.82%。全局运行还受起停、CTA 分发、尾波和时钟影响。

### 1. 已通过隔离实验确认：V 拼接的额外 BFI

当前 [_read_v](pa_8wave_950.py#L191-L199) 将 8 次 LDS b128 结果作为 BF16 slice 写入
register tensor；随后 [_mask_v_tail](pa_8wave_950.py#L202-L217) 再 bitcast 成 packed
i32、经条件路径返回。这组表示/合并在当前编译结果中产生 **每 superunit 8 条 BFI**：
41 tiles × 2 superunits × 8 = **656 条/wave**。

这些动态 BFI 的两个 value operand 全部相同，因此只是同值 bit-select/搬运；其中
**336 条连 destination 都相同**，是 self-move。它们在正常 full-tile 路径也执行。
每 wave 总指令数 **13502.5 vs 12416**，增加约 **8.75%**；并非只在最后一个尾页付费。

隔离 `_read_v` 的中间表示为 i32 words、一次拼接后再转回 BF16，**保持尾页保护**：

| 实验 | 总指令 / wave | BFI / wave | 稳态 phase cycles |
|---|---:|---:|---:|
| 当前生产源码 | 13502.5 | 656 | 2913.6 |
| packed-i32 V 拼接（进程内实验） | 12844.5 | **0** | **2834.9** |
| 去掉 V tail 保护（仅诊断，语义不完整） | 12100.0 | 0 | 2788.6 |

第三行不能作为优化提交：mask P 后仍可能有 `0*NaN`，必须保留 V tail 清零。
IR 和 ISA 里实际存在 tail branch/phi，不是“编译器把全部 tail mask 变成无条件执行”。
**精确由哪个 LLVM pass 产生 BFI 尚未定位**；已确认的是该 V 表示改写消除了 BFI 并改善
实际延迟，而不是通过推测决定删掉正确性保护。

### 2. 页表与边界控制有额外成本，但不能将它们统称为 memory stall

- 每 wave SMEM 指令合计 **61 vs 15**，其中 `s_load_dword` 为 **50 vs 4**；
  `_prefetch_page` 的页 ID 与地址计算是 direct-paged 相对 linear 的新增工作。
- `s_waitcnt` 为 **218 vs 175**，对应等待的每 wave 累计约 **10456 vs 8400 cycles**。
  当前 page-ready 与 LDS 共用 LGKM 完成条件，这些等待和 wave-group barrier 会相互影响。
- 两边 barrier **条数相同**，但每 wave 累计 barrier stall 约 **22278 vs 19250 cycles**。
  这说明组间等待增加，不是增加了 barrier 数量，也不能直接把差值当作可删除时间。
- 还有不同的 max/rescale/边界选择和 hazard padding：FlyDSL `s_nop` 为 **652/wave**，
  OPUS **458/wave**。FlyDSL 使用 `v_maximum3_f32`，OPUS 使用 `v_max3_f32`；两边
  都是顺序 max reduction，不能解释为“OPUS 是树、FlyDSL 是链”。NaN/最大值语义
  和生成指令调度需单独验证，不应直接替换来换取不等价的 benchmark。

## 不启用 ATT 的验证轮

同一进程内加载原始模块与隔离 packed-V 模块，显式 OPUS 为第三候选；相同数据，
预分配输出，100 轮共同预热，然后 5 轮交替、每轮 20 warmup / 100 次 `run_perftest`。

| 候选 | 中位 GPU 时间 | 相对 OPUS |
|---|---:|---:|
| 当前 8-wave direct | 302.930 µs | +3.82% |
| packed-V 隔离实验 | 298.784 µs | +2.39% |
| 显式 OPUS linear core | 291.796 µs | baseline |

packed-V 另通过 **24 个独立 FP32-reference 用例**：D128/D192 × noncausal/causal/
SWA128+sink × KV64/65/128/777，Q257，NaN-poisoned 尾页及三次 bit-exact 重复。
**尚未将实验合入生产文件或替代完整 173 项回归**；本次是诊断，不把实验耗时写成
当前公共接口的新性能。

## Persistent：现状和收益边界

- 当前 [_launch_attention](pa_8wave_950.py#L674-L685) 使用硬件三维 grid；
  [_attention_kernel](pa_8wave_950.py#L628-L664) 按固定 head/batch/qb 执行后退出。
  没有 `persist` 开关、atomic work ticket 或外层工作循环。
- [_attention_sequence](pa_8wave_950.py#L589-L610) 的 causal MERGE 是固定首尾两块，
  不会不断取下一个 Q block。不能将它称为 persistent。
- 现有 4-wave 只有 [dynamic 路径](../pa_4wave/pa_prefill_4wave.py#L1482-L1560)
  使用 atomic ticket + 外层工作循环；[static 路径](../pa_4wave/pa_prefill_4wave.py#L1419-L1453)
  同样是一 CTA 一工作项。
- 此目标有 640 个 CTA、256 CU，LDS 决定每 CU 只能 resident 一个 CTA，约 2.5 轮
  全卡工作量。persistent 可以尝试减少 CTA 重建/metadata 重读/任务分发开销，
  但**不会自动增加 resident waves 或消除上面的每 tile BFI/wait/softmax 成本**。
  均匀 noncausal 的收益不应提前假定；ragged/causal 的动态均衡是另一项测试。

建议优先级：**先验证并合入 packed-V 表示修正 → 保留语义地精简 tail/控制与等待 →
再独立评估 persistent**。本次没有实现或测量 persistent 收益。