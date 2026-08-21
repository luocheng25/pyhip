# MoE Down优化文档入口

## 当前快照

- 分支：`luocheng/hy3-single-n512-handoff`
- HEAD：`9049ddb723a1428d8dfb4c75e352d9b65bc9db56`
- 历史范围：`a6a1632`（含）至`9049ddb`（含），共42个线性commit
- 当前工作树：包含六入口公共pipeline重构、N-split/persistent修复与测试，以及未提交的`docs/UNIFIED8_DOWN_TODO.md`
- 平台：MI308X / gfx942 / 80 CU / wave64，ROCm 7.2
- 正式性能态：1800MHz determinism，PTL `Enabled / VECTOR,F8`，10 rotating buffers，24轮ABBA

当前最优先事项是验证K512 P4 N-split的CShuffle容量门禁：代码按双M activation预算排除了一个理论51,200B的单M候选。第二优先级是完成Hy3 role-priority候选的clean ABBA24与fresh ATT；H3 PTPC保留late-scale paired auto，四算法报告里的旧`a452743` P2不能用于关闭当前selector。

## 阅读顺序

1. [优化经验与当前Handoff](OPTIMIZATION_LESSONS_AND_HANDOFF.md)
   - 先读。按重要性归纳可复用经验，给出当前六入口、shape选择矩阵、已知风险和下一步。
2. [重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)
   - 核对当前未提交重构在P1-P5代表case上是否保持性能，以及Qwen K512 scale竞争修复的收益；P0未单独计时。
3. [ATT与Stall分析方法](STALL_ANALYSIS_GUIDE.md)
   - 采集或解释trace前读。固定successful issue、`pc_index`、physical SIMD union和owner账本口径。
4. [42提交编年史](COMMIT_CHRONICLE.md)
   - 查询某个方案何时尝试、结果数字、为何保留或拒绝；末节另列当前未提交工作。

## 文档职责

| 文档 | 回答的问题 | 更新时机 |
| --- | --- | --- |
| `OPTIMIZATION_LESSONS_AND_HANDOFF.md` | 当前该走哪条路径？哪些经验最重要？下一步做什么？ | selector、当前赢家或优先级改变时 |
| `REFACTOR_PERFORMANCE_REPORT.md` | P1-P5代表case重构前后是否等价？修复收益是多少？ | 重构或共享pipeline行为改变时 |
| `STALL_ANALYSIS_GUIDE.md` | ATT如何采、如何避免误归因、如何从stall生成候选？ | 分析口径、脚本参数或新owner模式变化时 |
| `COMMIT_CHRONICLE.md` | 42个commit和未提交实验分别做了什么？ | 新commit或新的保留/拒绝实验出现时 |

## 证据优先级

当旧文档与新结果冲突时，按以下顺序裁决：

1. 相同当前源码、相同完整shape和相同数学输出的clean 10-buffer ABBA24。
2. 相同源码/shape但较短的clean paired ABBA，仅作晋级候选。
3. 历史commit中的clean ABBA24，仅对当时准确的量化、metadata和输出布局有效。
4. 外部负载下的同进程paired ratio，只可判断局部重构等价，不可比较历史绝对时间。
5. ATT/PMC、ISA和理论roof用于解释机制，不单独决定selector。

附件中的8个JSON没有独立`finite`字段；它们记录了finite的`rel_l2`和clean tail，显式`isfinite`门禁来自当前GPU回归。历史ISA报告中的“18条buffer-load”和“24条load”使用了未完整保留的不同统计口径，不应据此比较指令工作量。

不得跨量化推广。例如最新H3 clean复测是PTPC，不能直接推翻历史H3 per-tensor paired结论；应分别复测和选择。

## 当前算法速查

| 路径 | 入口 | 结构 | 当前用途 |
| --- | --- | --- | --- |
| P0 legacy | `moe_2stage_down_prefill_1x4` | 4 waves，M64xN64循环 | Qwen最新clean赢家；H3 PTPC是旧P2对照 |
| P1 N256 | `moe_2stage_down_prefill_N256_1x4` | 4 waves，M64xN256 | Xiaomi最新clean赢家 |
| P2 paired | `moe_2stage_down_prefill_2x4` | 两个4-wave组，M128xN256 | exact H3历史特化；PTPC late-scale auto待当前源码复验 |
| P3 true8 | `moe_2stage_down_prefill_1x8_2` | 统一8-wave M64xN512 | exact Hy3 K192赢家 |
| P4 N-split | `moe_2stage_down_prefill_1x8` | 两个4-wave N256组，共享M64 | forced实验；默认auto关闭 |
| P5 persistent | `moe_2stage_down_prefill_1x8_persistent` | P4串行两个M64 | Xiaomi实验；未胜过P1 |

## 当前必须保留的边界

- 原legacy `moe_2stage_down_prefill_1x4`保持与`e6fe8e934859...`一致。
- 新入口不调用`fxh`；共享逻辑留在本地device helper中。
- `expanded_m64_tasks`必须在gateup、down和split-k之间保持同一metadata/grid契约。
- N-split的两个N256 subgroup必须使用隔离的PTPC scale LDS。
- `MOE_DOWN_NSPLIT_N512=auto`保持forced-only，不因Qwen局部修复改成默认。
- H3 PTPC四算法报告中的P2来自`a452743`；当前late-scale P2以`2b50372`的clean ABBA24为准，不能跨源码代际关闭auto。
- K512 P4容量门禁必须按`paired_m_groups=1`计算；当前双M谓词高估activation LDS。
- accuracy必须显式拒绝NaN/Inf，不能让`diff=NaN`通过阈值比较。
- 正式测试前核验idle与PTL，结束后恢复`auto / F16,BF16`。

## 相关历史资料

这些文件保留完整实验细节，但部分selector结论已被后续复测替代；先以本目录handoff为准：

- [Down总TODO](../../../../docs/UNIFIED8_DOWN_TODO.md)
- [Hy3 single-M N512 handoff](../../../../docs/HY3_SINGLE_M_N512_HANDOFF_TODO.md)
- [8-wave PA中间上下文](../../../../docs/UNIFIED8_PA_INTERMEDIATE_CONTEXT.md)
- [MoE测试README](../README.md)
- [四算法clean复测](../results/down_variants_20260821_retest/REPORT.md)
- [N-split 59dd与9049对比](../results/current_vs_9049_20260821/REPORT.md)
- [Control K128 stall账本](../CONTROL_K128_STALL_EXPOSURE.md)

## 新实验的记录模板

每个候选至少记录：

```text
源码commit/SHA与patch
完整shape、量化、metadata、padding、输出契约
GPU/ROCm/FlyDSL、idle状态、DPM/PTL、buffer/round数
正确性：finite、tail、padding、bitwise或rel_l2
ISA：MFMA/load/store/barrier、VGPR/AGPR/LDS/private/scratch、occupancy
性能：control -> candidate、ratio median、IQR、wins/rounds
ATT：MFMA union、exclusive owner、same-reason witness、原因转移、热点PC
结论：保留/拒绝/待clean复测，以及下一可证伪检查
```

若没有clean性能或fresh ATT，应明确标记证据级别，不用推测补齐结论。
