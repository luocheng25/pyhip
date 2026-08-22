# MoE Down优化经验与当前Handoff

## 一句话结论

MoE down不存在通用最优tile。当前最可靠的策略是：**按完整shape选择算法，以`down + sorted_sum`的clean ABBA决策；用physical SIMD stall账本解释结果；先守住VGPR/LDS occupancy门槛，再做局部调度。**

## 按重要性排序的经验

### 1. 以端到端combined和完整shape决策

这是最重要的结论。producer更快不代表系统更快，selector也不能只按`N/K/量化`的粗分类推广。

- Hy3历史上physical N256曾使down快9.80%，但`sorted_sum`慢45.70%，combined反而回退5.96%。
- H3 paired row-major比packed producer慢8.23%，却把consumer从3.6740ms降到0.7415ms，combined改善52.27%。
- R2、route-major和INT8压缩都曾明显改善某一侧，但完整链路分别因consumer gather或producer转换代价失败。
- 最新clean五shape复测比较的是59dd P4 N-split、P0/P1以及旧`a452743` P2：Hy3由P4胜出，Xiaomi由P1胜出，Qwen由P0胜出。H3结果不能代表`2b50372`之后的late-scale P2；后者对P0 Base的clean combined ratio为`0.95473`。另一个独立ABBA24证明9049 P3 true8快于Hy3 P4：P4/P3为`1.104034` down、`1.065036` combined。因此不存在“8-wave统一替代4-wave”，也不能混用不同源码代际决定selector。

决策单位必须包含：`B/TOPK/E/N/K/BM`、量化、padding、metadata排序单位、producer输出布局和consumer实现。正式指标为同进程24轮`down + sorted_sum` paired ratio。

### 2. Stall必须在physical SIMD union层分析

单wave stall不是physical空槽。Control K128中，single-wave `mfma_issue_unavailable`全部被peer MFMA execution覆盖，physical该项为0；真正缺口是两wave同相VMEM issue。

正确顺序是：

```text
successful issue -> 16-cycle MFMA window -> resident-wave union
-> physical idle -> exclusive owner -> joint phase/PC
```

必须使用`t_issue = first_attempt + stall`和`code[pc_index]`。目标owner下降后还要检查它转移到DS、tail还是residual，并同时要求MFMA union busy和墙钟改善。详见[stall分析方法](STALL_ANALYSIS_GUIDE.md)。

### 3. 单CU局部赢家可能输给整卡dispatch-tail

Physical-SIMD ATT只解释被采样CU内部的resident-wave气泡，不能看到80 CU之间由WG粒度造成的尾部不平衡。

- 零padding、相同1024个M64任务时，paired采样CU的steady MFMA union为90.73%，高于physical的86.29%，但整卡down/combined仍慢`1.01346/1.01946`。
- physical有1024个4-wave WG，critical CU承担13 waves/SIMD；paired只有512个8-wave WG，但critical CU承担14 waves/SIMD。ATT恰好采到了paired的轻载48-wave CU，漏掉56-wave重尾。
- 将任务数改为1280后，两边都恰好64 waves/CU，paired随即以`0.96455` down、`0.97949` combined胜出。
- 72-cell矩阵中，平衡任务使down ratio中位改善约5.20%，但60个原回退cell只有11个翻转；dispatch-tail重要，却不能解释短N rendezvous、输出转换和K512资源回退。

因此ATT之后还必须计算全芯片critical-wave分布并看整卡ABBA。对80 CU和`T`个M64任务，当前路径使用`ceil(T/80)`（physical）与`2*ceil(T/160)`（paired）waves/SIMD作为尾部下界。

### 4. 资源是离散门槛，不是连续成本

局部少几条指令常常不如跨过一个occupancy门槛重要，但“occupancy更高”也不是独立目标。

- K384将资源压到168 combined VGPR、20KB LDS后达到3 waves/SIMD，clean ABBA24改善4.06%。
- Hy3 single-M通过A swizzle shift3和`waves-per-eu=4,4`达到128 VGPR、32KB LDS、0 spill、4 waves/SIMD，成为K192 winner。
- K512 paired/N-split达到256 VGPR并出现80B左右private/scratch；这类路径即使功能正确，也在Qwen上远落后legacy。
- K192 8w-N256虽有4 resident waves，却因每wave工作太短、固定地址/load/CShuffle成本翻倍而回退51.26% down。

因此每次候选必须记录VGPR/AGPR、LDS、private/scratch和实际resident waves；跨档后必须重新看ATT和墙钟，不能用occupancy整数单独推断胜负。

### 5. 区分请求发射拥塞与完成等待

`VMEM issue stall`说明请求本身发不出去；`vmcnt wait`说明已发请求没有及时完成。两者需要相反的优化动作。

K192 single-M剩余physical idle中，VMEM issue占62.57%，completion wait仅4.60%；热点next-N K0 load到first consumer仍有约3.5K cycles距离。问题是weight load与output store同相争用，不是预取太晚。对应候选应改slot/role priority或请求相位，而不是继续提前load。

Control K128恢复slot-aware priority只改变仲裁，保持工作量和资源不变，ABBA24改善1.31%，两waveVMEM同因从9.8934%大幅下降。这是“先分类、再改代码”的标准案例。

### 6. 输出布局必须同时服务producer与consumer

MoE down的输出不是kernel终点，`sorted_sum`才是契约终点。

- wave-private XOR CShuffle把BN256结果恢复为标准row-major，在不引入bank conflict的前提下使combined优于R2。
- H3 paired从packed物理布局切到row-major，牺牲少量producer，换来数量级更好的consumer。
- direct atomic、route-major与压缩布局的共同失败原因，是producer、清零、转换或consumer中的另一端无法回本。
- padding必须按真实route测：Hy3 0B最佳；H3/Xiaomi的历史最优可为128B或256B，不能用随机loc benchmark替代真实TOPK路由。

任何新布局都必须同时报告down、sorted_sum、combined、输出容量和padding写入检查。

### 7. 先改变工作所有权，再榨局部调度

大收益通常来自tile/任务所有权改变，小收益才来自wait、priority和指令重排。

- M128 paired N512为两个相邻M64复制metadata并支付10个WG barrier；Hy3的突破来自改为single-M N512，一个WG只处理一个M64，8 waves沿N512连续展开。
- “8-wave”至少有三种不同算法：两个M64的`2x4`、一个M64的两个N256组`1x8`、统一N512 TiledMMA的true8 `1x8_2`。只比较线程数会误判算法。
- K越短，wave级地址、load、stage控制和CShuffle固定成本越重要。K192不能直接照搬K384的strict PA barrier和scheduler。

在局部ATT热点长期不收敛时，应重新问“谁拥有M/N/K与metadata”，而不是无限微调同一schedule。

### 8. 只有四层证据闭环才晋级

统一门禁：

1. **正确性**：physical/reduced、finite、inactive tail、padding、真实route和`rel_l2`。
2. **ISA/资源**：工作量、barrier、VGPR/AGPR、LDS、private/scratch、occupancy。
3. **性能**：clean 10-buffer ABBA24，报告ratio/IQR/wins；短测仅筛选。
4. **机制**：fresh ATT解释physical owner、原因转移和MFMA union。

外部作业占用79% VRAM时，重构等价验证只能采信同进程old/new ratio；绝对时间不能与历史clean窗口比较。idle gate拒绝ABBA24不是失败，而是保护证据质量。

### 9. 保留失败数字，防止重复实验

失败路线是可复用资产。除非fresh ATT owner或算法契约已改变，不要原样重复：

- R2/AoSoA、atomic scatter、route-major、FP8/INT8输出压缩；
- K128跨N BF16/LDS carry、packed FMA、tail反转、盲目增加VMEM/MFMA间隔；
- K192 paired barrier merge、K128+K64完整两stage、balanced CShuffle、K96 core；
- 强制2-wave、通用`4 VMEM/32 MFMA`、K64 core、weight permutation和过深consumer prefetch；
- 把H3 XCC map原样套到Hy3 K192；
- 将“8 waves/WG”当成可跨shape推广的统一算法。

详细commit、数字和保留/拒绝状态见[42提交编年史](COMMIT_CHRONICLE.md)。

## `de7887b`历史算法与入口

`de7887b`重构将算法拆为六个顶层入口；旧control中的这些路径原先都埋在`moe_2stage_down_prefill_1x4`中。

| ID | 算法 | 当前入口 | 线程/waves | 所有权 |
| --- | --- | --- | --- | --- |
| P0 | Legacy M64xN64 | `moe_2stage_down_prefill_1x4` | 256/4 | 4-wave统一M64xN64 |
| P1 | Physical M64xN256 | `moe_2stage_down_prefill_N256_1x4` | 256/4 | 4-wave统一N256 |
| P2 | Paired M128xN256 | `moe_2stage_down_prefill_2x4` | 512/8 | 两个4-wave组各处理一个M64 |
| P3 | True8 M64xN512 | `moe_2stage_down_prefill_1x8_2` | 512/8 | 统一`(8,1,1)`，8 waves连续覆盖N512 |
| P4 | N-split M64xN512 | `moe_2stage_down_prefill_1x8` | 512/8 | 两个4-wave N256组共享一个M64 |
| P5 | Xiaomi persistent | `moe_2stage_down_prefill_1x8_persistent` | 512/8 | P4沿N分组，同WG串行两个M64 |

新入口均为薄wrapper，不调用`fxh`；copy、scale、MMA、K-stage、CShuffle和任务映射下沉到公共设备helper。legacy入口保持与`e6fe8e934859...`的decorator+函数文本一致。

## 当前简化架构（已实施）

实现已收敛为三条生产路径。P1/P2使用同一个设备入口，`known_block_size`由
`256 * m_groups`编译期派生；P3拥有独立主循环。

| 目标路径 | 由当前路径演化 | 计算结构 | 保留case | 简化决策 |
| --- | --- | --- | --- | --- |
| P0 legacy | P0 | 4-wave M64xN64循环 | Qwen K512、通用fallback | **已保持源码与行为不动**，nested函数SHA256仍为`a6ac0289...`。 |
| P12 physical N256 | P1 + P2 | 每个4-wave group计算M64xN256；编译期`m_groups=1/2` | `m_groups=1`服务Xiaomi/通用N256；`m_groups=2`服务exact H3 | 已合并为唯一`_moe_2stage_down_prefill_physical_n256`主循环。 |
| P3 true8 Hy3 | P3 | 统一`(8,1,1)` TiledMMA计算M64xN512 | exact Hy3 K192 | 已抽为独立`_moe_2stage_down_prefill_true8_hy3`主循环。 |
| 已删除 | P4 + P5 | N-split M64xN512与persistent双M64 | 无自动生产赢家 | selector、环境变量、kernel和专用状态已移除；性能数据留在历史文档。 |

### 简化结论

1. **P0不动。** P0是Qwen K512和通用fallback的明确赢家，而且重构性能报告未单独计时P0；修改它既没有收益证据，也会扩大回归面。
2. **P1/P2合并。** 两者的基本计算单元都是4-wave M64xN256。P2只是并列两个group处理两个M64，适合用编译期`m_groups`表达，而不应形成第二套pipeline。
3. **P3独立。** P3使用统一8-wave MMA、512线程activation copy、shift3 swizzle、128 VGPR/32KB LDS和4 waves/SIMD约束；这些不变量与P12不同，强行共用主循环只会重新引入分支。
4. **P4删除。** P4从未自动选择；Hy3上P3比P4快10.40% down、6.50% combined，Qwen K512上P4又比P0慢约104%--111% combined。修正P4的51,200B CShuffle容量门禁不再进入生产计划。
5. **P5删除。** P5在Xiaomi仍比P1慢4.01% down、2.73% combined；persistent与early-prefetch状态不足以证明其复杂度合理。

### P12的最小配置面

host `_FlyDownPlan`只保留`path/m_groups/metadata_m_groups/padding_bytes`四个字段。
task map、output schedule和scale schedule不是公共配置；P12根据受限shape、量化和`m_groups`
在编译期唯一派生，避免把内部调度继续暴露为API。

| 字段 | P1模式 | P2模式 | 约束 |
| --- | ---: | ---: | --- |
| `m_groups` | 1 | 2 | 唯一决定M方向并组数量。 |
| `threads` | 256 | 512 | 必须由`256 * m_groups`派生。 |
| WG tile | M64xN256 | M128xN256 | 每个group始终是M64xN256。 |
| sorting tile | M64 | M128 | host一次决定，不再单独传`expanded_m64_tasks`开关。 |
| task map（内部派生） | linear M64 | generic pair swizzle或exact-H3 XCC/SE map | exact H3固定映射仅在至少2316 valid pairs时启用；短prefix线性回退。 |
| `padding_bytes` | shape决定0/128B | exact H3通常0/128B | host plan字段，同时驱动output allocation、`output_row_stride`和`sorted_sum`。 |
| output schedule（内部派生） | 通用N256 CShuffle | paired row-major CShuffle + delayed store | 由`m_groups`唯一决定，不公开枚举。 |
| scale schedule（内部派生） | default/preloaded | per-tensor default或exact-H3 PTPC late-load | exact H3 PTPC取消next-scale预取、MFMA后加载scale并改变loop-carried state。 |

当前kernel API为：

```text
compile_gemm(down_path=legacy | physical_n256 | true8_hy3,
			 down_m_groups=1 | 2,
			 metadata_m_groups=1 | 2,
			 down_output_padding_bytes=None | 0 | 32 | 64 | 128)
```

host `_FlyDownPlan`一次决定path、M-group、metadata展开和padding；
`down_physical_n512/down_paired_row_major/down_single_m_n512/down_nsplit_n512`
及`expanded_m64_tasks`已从生产API删除。

## 当前shape决策矩阵

下表以最新clean四算法复测为主；“历史成立”与“当前推荐”分开，避免旧selector结论覆盖新证据。

| Shape | 当前证据赢家 | 建议状态 | 说明 |
| --- | --- | --- | --- |
| Hy3：N4096/K192/E193/TopK9/per-tensor | **P3 true8 single-M N512** | 目标仍为独立P3 | 五shape复测中P4先胜P0/P1/P2；独立P3对P4 ABBA24又测得P4/P3为`1.104034` down、`1.065036` combined。role-priority仅有clean ABBA4，尚缺ABBA24和fresh ATT。 |
| Xiaomi：N6144/K256/E384/TopK8/PTPC | **P1 physical N256** | 迁入P12，`m_groups=1`；删除P5 | 最新clean复测P5比P1慢4.01% down、2.73% combined；persistent只比paired更好，不足以保留。 |
| H3：N6144/K384/E128/TopK4/PTPC | **P2 late-scale paired** | 迁入P12，`m_groups=2`、`H3_PTPC_LATE` | 旧`a452743` P2比P0慢56.87%/42.55%，但不含`2b50372` late-scale零spill修复；late-scale P2对P0 Base的clean combined ratio为`0.95473`，24/24胜。 |
| H3：N4096/K384/E193/TopK9/per-tensor | P2 paired曾在历史正式窗口胜出 | 迁入P12，`m_groups=2`；需当前代码clean复测 | 最新五shape复测使用N6144/E128/TopK4 PTPC，不能直接推翻per-tensor结果。 |
| Qwen3.5 35B：N2048/K512/E256/TopK8/PTPC | **P0 legacy** | 保持P0；删除P4 | P4修复scale LDS竞争后比59dd自身快0.92% combined，但最新clean绝对算法比较仍比P0慢110.98%。 |
| Qwen3.5 397B：N4096/K512/E512/TopK10/PTPC | **P0 legacy** | 保持P0；删除P4 | race修复后比59dd自身快2.73% combined，但最新clean仍比P0慢103.75%。 |

P4与`MOE_DOWN_NSPLIT_N512`已删除；历史force结果仅保留为否决证据，不再推广为K512生产路径。

## 当前实现基线

工作分支：`luocheng/hy3-single-n512-handoff`；父基线：
`de7887be185cd7acbf4b45c938d295e91eab49b7`，当前改动尚未提交。

该提交包含：

- `docs/UNIFIED8_DOWN_TODO.md`的实验记录；
- 六入口公共pipeline重构；
- down GPU回归、selector与metadata/grid回归；
- N-split PTPC双group scale LDS隔离、copy线程、短grid、expanded M64 task、K512寻址和finite检查修复。

已验证证据：

- down GPU回归70项通过；
- selector相关测试51项通过；
- P1-P5的六个原始代表case重构前后最大中位偏差：down 0.179%，combined 0.275%；P0 legacy未单独计时；
- Qwen K512修复组改善0.75%/0.92%和1.98%/2.73%（down/combined）；
- H3 4632任务静态映射严格双射；
- 详细协议、资源和8个case见[重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)。

注意：等价验证时外部作业占79% VRAM，使用2 buffers；ratio可用于重构等价判断，绝对时间不得替代clean 10-buffer数据。

TODO还记录了三组未保留在主line的实验资产：56KB共享CShuffle是小幅资源改善；真实`4N x 2M`原型在0 spill/2 barrier下仍输；跨CU dispatch-tail实验与72-cell falsification证明单CU ATT局部赢家可能输给整卡WG尾部。这些结论已纳入上面的第3条经验。

## 简化实施状态

### S0：冻结行为基线（完成）

1. 保存P0、P1、P2、P3当前源码与ISA SHA；P0只做字节一致检查。
2. 使用现有70项down回归和51项selector回归作为功能基线。
3. 对P1/P2/P3分别保存代表case的same-process ratio；不要把P4/P5作为新实现的性能门禁。

### S1：删除P4/P5（完成）

1. selector删除`MOE_DOWN_NSPLIT_N512`和`_select_fly_down_nsplit_layout()`。
2. 先让dispatcher无法选择`moe_2stage_down_prefill_1x8`与`moe_2stage_down_prefill_1x8_persistent`，但暂不从共享helper物理删除K-stage代码。
3. 跑P0/P1/P2/P3功能、ISA与性能基线，确认断开P4/P5可达性没有改变保留路径。
4. 完成S2/S3隔离后，再删除P4/P5入口、persistent second-M64、N-split偶奇N256映射、双group PTPC scale LDS与仅P4使用的immediate/delayed-store状态。
5. K384 weight-head、late PTPC scale、paired barrier/delayed store当前也服务P2；只有引用审计证明P12不再使用后才能删除，不能按“K384/K512分支”整块移除。
6. 保留P4/P5报告与JSON，不保留可执行生产代码。

删除P4/P5时仍需保留由本轮发现的通用修复：finite/NaN拒绝、合法copy线程选择，以及P1/P2/P3仍使用的短grid边界检查。不能因为最初在P4调试中发现它们就一起删除。

### S2：合并P1/P2为P12（完成）

1. 建立最小`_FlyDownPlan(path, m_groups, metadata_m_groups, padding_bytes)`；线程数、tile、metadata和输出分配从plan派生，task/output/scale schedule留在P12内部编译期推导。
2. 将P1/P2共同的copy、MMA、scale与CShuffle主循环搬入唯一`physical_n256`设备实现。
3. P2的paired row-major写回由`m_groups=2`唯一启用；PTPC late-scale由exact H3 PTPC shape唯一启用，不增加公共枚举。
4. exact H3 PTPC分支必须同时控制：不预取next scale、MFMA后加载当前scale、移除scale fragment的loop-carried state；per-tensor P2保持default scale schedule。
5. 若FlyDSL要求静态`known_block_size`，保留两个不含业务逻辑的256/512线程adapter；它们仍属于同一P12路径。
6. 第一阶段保持现有metadata ABI，只把`expanded_m64_tasks`变成`m_groups`的派生值，以便先验证P1/P2合并本身。
7. 第二阶段若移除`sorted_expert_ids.repeat_interleave(2)`，必须原子迁移四个独立计数：`metadata_blocks`（sorting产生的M64/M128记录）、`m64_tasks`（gateup/split-k消费单位）、`down_workgroups`（P2通常为`m64_tasks / 2`）和`physical_rows`（输出分配与`sorted_sum`边界）。
8. 稀疏路径的`M * TOPK * 2`覆盖、expert索引与输出行数必须在同一patch内切换到新ABI；禁止只改grid或只改metadata复制。
9. selector一次生成四字段plan；launch、output allocation和`sorted_sum`必须消费同一个plan，不能各自重新匹配shape。

### S3：抽出独立P3（完成）

1. 将P3的统一8-wave MMA、A-copy、swizzle、occupancy约束和role priority移入专用实现。
2. 入口保留exact Hy3 shape assert，不推广为通用N512。
3. P3与P12仅共享`fxh`提供的pointer/view、copy atom、fragment遍历和数值转换等叶子helper；不得共享含`m_groups`、barrier或epilogue状态的主循环。`fxh`无等价实现的`_down_copy_threads`保留本地。

### S4：收口selector与验证（功能/资源完成，性能待空闲GPU）

1. selector最终只能产生P0、P12或P3三种plan。
2. Qwen必须回到P0，Xiaomi走P12/`m_groups=1`，H3走P12/`m_groups=2`，Hy3走P3。
3. P1/P2/P3重构后要求代表ISA工作量、VGPR/LDS/private/scratch不退化；P0要求源码/ISA不变。
4. 先跑正确性与资源门禁，再用clean 10-buffer ABBA24确认P12/P3；只有重构等价，不再开展P4/P5调优。

已完成验证：

- 完整down回归66项，selector/plan回归55项。
- P0 nested函数文本与`de7887b`完全一致。
- P1/P2规范化kernel符号后的final ISA与`de7887b`逐字一致；SHA256分别为`bc6f0b760f65d145...`和`314fb83fba17f172...`。
- P3规范化final ISA与抽离前逐字一致：128 VGPR、32KB LDS、0 private/scratch、96 MFMA、2 barriers、24 loads、8 stores。
- 删除重复`_down_*`/`_DownOps`并切换到`fxh`后，P1/P2/P3 final ISA逐字不变；SHA256分别为`3b94b124...`、`b0e4b654...`、`e4e4fd6a...`。
- P2 K384 PTPC：256 VGPR、64KB LDS、0 private/scratch、192 MFMA、10 barriers。
- P1 Xiaomi K256 PTPC：208 VGPR、25KB LDS、0 private/scratch、128 MFMA、2 barriers。
- BM64、small-batch legacy回退、P2 metadata和M32768短valid-prefix门禁均有回归。

未完成项仅为正式性能复验：当前8卡均约78% VRAM占用，不满足`VRAM<=20%`门禁，未运行10-buffer ABBA24。

## 续跑检查表

### 修改前

- 记录branch、commit、kernel SHA和工作树diff。
- 选择`util<=5%`、`VRAM<=20%`空闲GPU。
- 确认初始performance level为`auto`。
- 明确control/candidate的metadata、padding和数学输出相同。

### 每个候选

- 先跑目标正确性，拒绝NaN/Inf。
- 导出ISA，记录资源、工作量和occupancy。
- 短ABBA只淘汰；晋级用10-buffer ABBA24。
- 有性能收益再采fresh ATT，不用旧trace解释新源码。
- 报告结束状态，并恢复`auto / F16,BF16`。

### 文档更新

- 在[编年史](COMMIT_CHRONICLE.md)记录方法、数字和保留/拒绝。
- 在[stall手册](STALL_ANALYSIS_GUIDE.md)补充新的owner判别或工具限制。
- 在本矩阵更新当前赢家、selector边界和下一优先级。
- 保存JSON、SHA256、ISA资源和trace目录来源；不要只写结论。
