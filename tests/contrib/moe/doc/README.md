# MoE Down优化文档入口

## 当前快照

- 分支：`luocheng/hy3-single-n512-handoff`
- HEAD基线：`de7887be185cd7acbf4b45c938d295e91eab49b7`
- 历史范围：`a6a1632`（含）至`9049ddb`（含），共42个线性commit
- 当前工作树：已从`de7887b`六入口实现收敛为P0、P12、P3三条路径。
- 平台：MI308X / gfx942 / 80 CU / wave64，ROCm 7.2
- 正式性能态：1800MHz determinism，PTL `Enabled / VECTOR,F8`，10 rotating buffers，24轮ABBA

当前实现：P0保持不动；P1/P2已合并为P12 physical N256；P3已抽为exact Hy3 true8独立主循环；P4/P5及`MOE_DOWN_NSPLIT_N512`已删除。

## 阅读顺序

1. [优化经验与当前Handoff](OPTIMIZATION_LESSONS_AND_HANDOFF.md)
   - 先读。按重要性归纳可复用经验，给出三路径实现、shape选择矩阵、资源门禁和后续验证。
2. [重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)
   - 核对`de7887b`重构在P1-P5代表case上是否保持性能，以及Qwen K512 scale竞争修复的收益；P0未单独计时。
3. [ATT与Stall分析方法](STALL_ANALYSIS_GUIDE.md)
   - 采集或解释trace前读。固定successful issue、`pc_index`、physical SIMD union和owner账本口径。
4. [42提交编年史](COMMIT_CHRONICLE.md)
   - 查询某个方案何时尝试、结果数字、为何保留或拒绝；末节另列历史范围之后的`de7887b`集成检查点。

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

## `de7887b`历史六路径

| 路径 | 入口 | 结构 | 当前用途 |
| --- | --- | --- | --- |
| P0 legacy | `moe_2stage_down_prefill_1x4` | 4 waves，M64xN64循环 | Qwen最新clean赢家；H3 PTPC是旧P2对照 |
| P1 N256 | `moe_2stage_down_prefill_N256_1x4` | 4 waves，M64xN256 | Xiaomi最新clean赢家 |
| P2 paired | `moe_2stage_down_prefill_2x4` | 两个4-wave组，M128xN256 | exact H3历史特化；PTPC late-scale auto待当前源码复验 |
| P3 true8 | `moe_2stage_down_prefill_1x8_2` | 统一8-wave M64xN512 | exact Hy3 K192赢家 |
| P4 N-split | `moe_2stage_down_prefill_1x8` | 两个4-wave N256组，共享M64 | forced实验；默认auto关闭 |
| P5 persistent | `moe_2stage_down_prefill_1x8_persistent` | P4串行两个M64 | Xiaomi实验；未胜过P1 |

## 当前三路径速查

| 目标路径 | 来源 | 生产case | 处理方式 |
| --- | --- | --- | --- |
| P0 legacy | P0 | Qwen、通用fallback | `moe_2stage_down_prefill_1x4`文本哈希保持`a6ac0289...`。 |
| P12 physical N256 | P1 + P2 | Xiaomi使用`m_groups=1`；H3使用`m_groups=2` | 唯一`_moe_2stage_down_prefill_physical_n256`主循环；入口线程数为`256*m_groups`。 |
| P3 true8 Hy3 | P3 | exact Hy3 K192 | 独立`_moe_2stage_down_prefill_true8_hy3`主循环。 |
| 已删除 | P4 + P5 | 无 | selector、环境变量、kernel和专用状态均移除；历史JSON保留。 |

公共编译API为：

```text
down_path = legacy | physical_n256 | true8_hy3
down_m_groups = 1 | 2
metadata_m_groups = 1 | 2
```

host由四字段`_FlyDownPlan(path, m_groups, metadata_m_groups, padding_bytes)`一次决定路径、任务展开和padding；output allocation与`sorted_sum`共用该plan。task map、output schedule和scale schedule由P12内部根据受限shape/量化唯一派生，不暴露为公共参数。

## 当前必须保留的边界

- 原legacy `moe_2stage_down_prefill_1x4`保持与`e6fe8e934859...`一致。
- P12与P3复用`fxh`中的`all_elements/all_copy_atoms/view/atom_tensor/eltwise_op/FlyObjCache`等叶子helper，不共享task map、K-loop、barrier或epilogue状态。仅`_down_copy_threads`因`fxh`无等价实现而保留本地。
- P12固定BM64/N256；P2要求`down_m_groups=metadata_m_groups=2`。
- H3 PTPC四算法报告中的P2来自`a452743`；当前late-scale P2以`2b50372`的clean ABBA24为准，不能跨源码代际关闭auto。
- P4/P5删除后不再推进K512 P4容量候选；51,200B结论仅保留为历史证据。
- accuracy必须显式拒绝NaN/Inf，不能让`diff=NaN`通过阈值比较。
- 正式测试前核验idle与PTL，结束后恢复`auto / F16,BF16`。

当前验证：完整down回归66项与selector/plan回归55项通过；新增BM64、小batch和H3短prefix门禁通过。fresh ISA资源为：P1 Xiaomi `208 VGPR/25KB LDS/0 scratch`，P2 K384 PTPC `256 VGPR/64KB/0 scratch`，P3 `128 VGPR/32KB/0 scratch`。规范化kernel符号后，P1/P2/P3 final ISA均与`de7887b`逐字一致；切换到`fxh`前后也逐字一致，哈希分别为`3b94b124...`/`b0e4b654...`/`e4e4fd6a...`。由于GPU显存被外部作业占78%，尚未执行正式10-buffer ABBA24。

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
