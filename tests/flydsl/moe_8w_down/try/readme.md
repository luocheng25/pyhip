# MoE 8-wave down：完整优化历史与实验归档

更新：2026-09-12。本目录保存精简前的**完整实现、对照内核、测试、调优脚本和原始产物**；上级目录只维护胜出路径及用户要求保留的旧FlyDSL基线。

## 1. 归档范围与当前结论

- 精简前的27个顶层文件（25个Python文件、2个Markdown文件）先逐字节复制并核对，再整理上级目录。
- [README.md](README.md)是整理前主说明的原始副本；[README_8stage.md](README_8stage.md)保存四/八/十六阶段的详细设计及旧验证记录。本文件按时间与研究方向汇总全部已记录尝试，包括失败、慢样本和未采用方案。
- 原历史产物目录整体移入本目录，15组实验的日志、ATT UI、code objects、ISA和源码快照保持不变。下文均链接到新的相对位置。
- [moe_multistage_down.py](moe_multistage_down.py)、[moe_multistage_down_mfma32.py](moe_multistage_down_mfma32.py)、[moe_multistage_down_m64.py](moe_multistage_down_m64.py)、[moe_multistage_compact.py](moe_multistage_compact.py)保留完整可选参数；[moe_multistage_pipeline.py](moe_multistage_pipeline.py)、[moe_multistage_reduce.py](moe_multistage_reduce.py)、[moe_multistage_restore.py](moe_multistage_restore.py)保留全部布局/归约实验。
- 当前胜出配置仍为 **MFMA16、M256/N128/K256、OC4、PF3、256个persistent CTA、packed写出＋定制TOPK reduce**。compact和混合OC没有超过它，MFMA32的算子/浮点重新结合也未晋级。
- 用户补充要求保留上级的 [旧FlyDSL内核](../moe_8wave_down.py)、[旧DMA工具](../moe_8wave_down_utils.py)及 [主说明](../README.md)。旧BN32/BN64继续是主基准的默认对照；主目录唯一测试文件是 [test_blockscaled.py](../test_blockscaled.py)。

历史“当前源码”“默认候选”“最终”只指记录当时的状态。早期文档中的旧运行库路径、绝对路径和命令没有批量改写；它们不代表现有主路径接口。需要重放时使用本目录相应driver及配套快照，不能把旧参数传给已固定配置的上级模块。

## 2. 固定数学契约与比较口径

默认性能形状：GPU5、MI350X/gfx950、**256 CU、8 XCD**，tokens16384、TOPK8、E384、N6144、K256，seed1234。

- A为OCP FP8 E4M3FN `[tokens,topk,K]`，FP32 A scale物理K-major，每1×128一个scale。
- B为OCP FP8 `[experts,N,K]`，每128×128一个FP32 scale；MFMA16权重排列为(16,16)，MFMA32为(32,16)。
- 每K128 partial独立缩放并累加，再乘routing，最后BF16舍入。改变浮点结合顺序的实验须单独标明，不默认bitwise等价。
- routing ID编码为 `(slot << 24) | token`；AITER按M256补齐每个专家run。device valid计数是**padded有效前缀长度**，不是实际路由行数，metadata剩余容量未初始化。
- 原down输出是BF16 `[tokens,topk,N]`，TOPK为dim1。完整流程输出是BF16 `[tokens,N]`；不能误沿N求和。
- 默认实际路由行131072、原padded行196608、valid专家块768、去重专家384。默认packed分配容量896×256×6144×2，即2.625 GiB；未写padding不能当作有效数据。
- 性能沿用 `pyhip.run_perftest`、warmup2/iters10、counter reset计入。量化、排序、shuffle、参考、编译、统计和NaN投毒在稳态计时外；确认比较共享输入/输出地址，完整调用时间不由组件相加。
- 有效/padded TFLOPS分别使用真实/执行行数；逻辑带宽并列B每M块读取、每去重专家一次理想缓存两种模型。不是HBM硬件计数器，未计全部元数据和重复A加载。
- 数值容差一直为 `rtol=atol=0.01`。先对独立Torch参考检查BF16逐路输出，再以已验证基线的实际BF16 routes做TOPK sum参考；另有独立BF16 reducer检查，避免把不同中间舍入经相消放大的差异错误归因于reduce。
- 未通过调整GPU时钟/功耗/NUMA、丢弃慢轮次或弱化容差来获得结论。

## 3. 2026-09-11：从k128n蓝本到四/八阶段

最初要求参考8-wave的k128n分阶段实现，而不是旧down的整K/整N流水。目标是M256/N256/K256、8 waves、B直接global→LDS、Memory无VALU、原生MFMA16×16×128、MFMA间插入有用VALU，并以寄存器完成BF16转换/重排，保持十tensor接口。

1. 初版建立两组4-wave错相、B环形LDS、延迟写出及寄存器累加。最初BN256存在56B scratch，不能把这版资源数当作后续无spill版本。
2. 修正当前FlyDSL资源/指针接口、FP8 128-bit对齐load、vector<1xi32>标量intrinsic不匹配。`waves_per_eu=2`不等于禁止AGPR，必须通过LLVM passthrough显式限制AGPR并关闭packed-FP32 combine。
3. 通过任务内thread/lane重建、寄存器tuple pinning、A/B scale与routing缓存，缩短跨任务活跃区间并消除scratch/AGPR。A片段仍保留寄存器，C从不经LDS。
4. BN128/K256改为两个Memory/Compute对，共四阶段：每Compute计算N64×K256、16条MFMA；上一N的输出分别均分到两个Memory写出。旧八阶段→四阶段两次快测为658.704→625.110 μs、653.860→636.171 μs，约5.1%/2.7%时延改善，当时仍慢于旧BN64。

早期NT/AGPR配置的同场五候选时间：PyHIP564.728、旧BN32 677.364、旧BN64 579.973、新BN128四阶段627.887、新BN256八阶段706.969 μs。之后禁用NT/AGPR并调整缓存/寄存器，不能跨版本复用该排名。

证据：[初版五候选](ck_test/blockscaled_compare_20260911/README.md)、[BN128四阶段](ck_test/bn128_4stage_20260911/README.md)、[cached/VGPR-only](ck_test/cached_vgpr_4stage_8stage_20260911/README.md)。

## 4. 预取距离、整段Compute地址提前和stage粒度

### 4.1 PF1/2/3与地址提前

定义Memory q发Q[q+d]。必须同时调整请求前瞻和真实等待，不能仅放宽vmcnt。启动只发有消费者的packet，尾部静态裁掉无消费者DMA，任务边界彻底drain后才复用LDS。

- generic最终保留PF2，tuned组合后续选择PF3；不同stage粒度的“一拍”不是同一个字节量。
- lane地址全部移到task preparation；动态N只在Memory内做标量地址计算。为固定ring slot，主循环展开完整环周期。
- 动态B scale使用 `ds_read_addtid_b32`，保存/恢复M0，并在写M0后保留 `s_nop 1`。漏掉M0等待曾导致大量数值错误；修复后实际ISA整个Compute（含尾部）无地址运算。
- 地址提前前PF2快测BN128/BN256为566.600/565.380 μs；随后同场地址提前比较BN128 590.314→568.173、BN256 572.637→574.977 μs，BN256未获益。它们属于不同轮次，不串联计算累计加速。
- 当时ISA BN128为177 VGPR，BN256为239 VGPR；BN256有SGPR spill统计但未落scratch，不能笼统宣称所有历史版本SGPR spill均0。

证据：[PF2](ck_test/prefetch2_4stage_8stage_20260911/README.md)、[整段地址提前](ck_test/address_hoisted_4stage_8stage_20260911/README.md)、[详细流水与验证](README_8stage.md)。

### 4.2 BN256十六阶段：未采用

将N256分为八个Memory/Compute对，每Compute8条MFMA、每Memory两条延迟store＋一条N64/K128 direct-LDS DMA。八槽仍64KiB，比较vmcnt6/9对应的有效预取距离。

| 配置 | 同场时间 μs |
|---|---:|
| BN256八阶段 | 573.053 |
| BN256十六阶段 vmcnt6 | 577.605 |
| BN256十六阶段 vmcnt9 | 580.705 |

数值、graph、无scratch/AGPR、Memory无VALU、Compute无地址及七VALU审计通过，但没有性能收益。保留为本目录显式实验，不进入主路径。

### 4.3 BN64四阶段：未采用

每packet N32/K256、每Compute8条MFMA、Memory两store＋一DMA。PF1/2/3稳态分别是**vmcnt0/3/6**，不是旧选择器的3/6/9。

| PF距离 | 时间 μs |
|---:|---:|
| 1 | 1320.784 |
| 2 | 641.919 |
| 3 | 670.532 |

同场BN128/BN256对照为581.773/562.968 μs。三种BN64数值和ISA通过，PF2在该组较快但不胜。上述十六阶段和BN64细节见 [阶段实验记录](README_8stage.md)。

## 5. ATT驱动的BN128组合优化

早期BN256 sampled stall以barrier/store为主，促使实验聚焦写事务和Memory覆盖；这些stall比例不是整卡墙钟占比。

最终组合为BN128四阶段、OC4、PF3、LDS read先发、连续128B写出、输出aux2/NT、B aux16/SC1、不切换每stage优先级。DPP在寄存器交换一个row bit和column bit，保持原route/scales顺序与BF16舍入。热启动保留检查及counter清零，仅改为传新读取的裸指针/尺寸/当前stream，不缓存地址或routing内容。

未采用的同时期方向：

| 尝试 | 观察/结论 |
|---|---|
| 原scatter写单独开NT | 明显变慢，约700 μs；NT必须和连续写组合看待 |
| 每行256B覆盖 | 额外重排/mask抵消收益 |
| atomic swap覆盖输出 | 正确但约2.3–2.4 ms，拒绝 |
| 静态任务/静态XCD映射 | 当时组合未稳定胜出；后续另有独立nonpersistent研究 |
| 去错相/删barrier | 不作为合法无代价优化采纳 |
| B scale直接全局读取、routing提前并入scale | 未获稳定收益，后者改变浮点结合次序 |
| DMA/store顺序、N反向/轮转 | 无稳定赢家 |
| BN256组合 | 单轮曾超过10%，交错确认仅约1.068–1.077×，未选为最终结果 |

历史512-CTA tuned与PyHIP同址三轮均值 **522.627034 vs575.496832 μs**，速度比1.101162×（速度+10.12%，时延−9.19%）；首轮只有约5.4%，不是每轮保证10%。不同地址的早期更高结果不用于最终结论。

当时原生与ATT ISA均为185 VGPR、100 SGPR、70660B LDS、零private/spill/AGPR，12个Compute、165个稳态间隔各7条VALU、48条NT store。旧ATT/snapshot明确保留512 CTA，不重标为后来256 CTA。

证据：[组合结果与所有日志](ck_test/att_optimized_bn128_20260911/README.md)、[原始探索采集日志](ck_test/opt_profile_20260911/profile.log)。

## 6. 2026-09-12：CTA256与512单变量消融

保持OC4/PF3/NT/SC1/连续写等全部选项不变，四轮ABBA/BAAB：CTA256平均523.786890 μs，CTA512平均524.047935 μs，512时延约+0.05%，两轮略快两轮略慢，没有稳定收益。随后按用户要求固定256 CTA，每CTA仍512线程/8 waves。

证据：[CTA消融](ck_test/cta_ablation_20260912/README.md)。不把此前组合收益归因于CTA翻倍。

## 7. MFMA32与“再提升10%”：未达标

独立MFMA32×32×64原生OCP-FP8路径，权重(32,16)，两条K64先形成K128 partial再应用原block scale。四阶段、每Compute8条MFMA，目标每MFMA间14条有用VALU；寄存器BF16/permlane/DPP输出，无C LDS、AGPR或scratch。

组合探索包含：延迟K1退休、routing乘入FP32系数、row系数缓存、两个M任务成组、全padding wave跳算。无效wave仍参与所有必要DMA/wait/barrier。结构化分支曾引入scratch和等待/数值问题，不能忽略LLVM额外插入的VMEM；实验后改为uniform ASM guard。

四轮ABBA/BAAB对当前256-CTA MFMA16基线：**528.471989→523.609352 μs，速度+0.93%**，不是10%。实际ISA209 VGPR/97 SGPR/70660B LDS，77个稳态间隔各14条VALU；形式达标不代表吞吐目标达标。MFMA32的routing重新结合后来在完整TOPK reduce中失败，因此不是最终胜出路径。

其他邻域实验全部保留：

| 方向 | 结果概述 |
|---|---|
| B scale成对读/跨半块复用 | 约0.1%级别，无明确收益 |
| MFMA32原始32/64/128B行覆盖 | 约1081/686/538 μs，对照533 μs；128B覆盖必要但不充分 |
| PF2、PF4–7、cache/priority/splits | 未找到10%赢家；B NT约630 μs，明显回退 |
| 全padding wave跳Compute/跳LDS/同步变体 | 约0–2%级别，保留所有必要协作事件 |
| M512/N64、K128分段 | 高VGPR/spill导致约1.2–2.1 ms |
| M512/N32 | 无spill但64B写出效率差；合并128B后约545–550 μs，仍未胜 |
| M512把A放LDS | 零spill但162320B LDS，约540 μs vs522 μs |
| BM128/小packet提高并发 | 更少padding被B重复读取及启动成本抵消 |
| BM384 MFMA16 | 较少padding但整体持平，split6约1%级别 |
| 256B行/global store | 地址和重排/mask成本抵消收益 |
| task顺序、LDS布局、N轮转 | 小幅或无收益 |

各方向精确日志、快照及raw ATT见 [next10记录](ck_test/next10_20260912/README.md)；[配套旧版本重放](ck_test/next10_20260912/archive/replay_sweep.py)不覆盖当前模块。失败和编译错误日志未删除。

## 8. 连续写出＋restore：GEMM更快，端到端更慢

比较routed、sorted、packed、linear、linear_raw；后者减少GEMM重排，把恢复工作交给后继kernel。完整契约仍是输出原 `[tokens,topk,N]`，不能只报临时GEMM。

| 方案 | GEMM μs | Restore μs | 完整调用 μs |
|---|---:|---:|---:|
| MFMA32直接routed | — | 无 | 522.823 |
| sorted | 509.301 | 723.548 | 1230.460 |
| packed | 491.090 | 746.787 | 1231.731 |
| linear | 504.648 | 752.383 | 1253.649 |
| linear_raw | 496.472 | 1409.617 | 1903.403 |

两轮全部正确，packed GEMM约+6.46%，但完整路径约2.36倍慢。默认1.5GiB有效BF16中间数据的恢复至少额外读写3GiB，单独restore不值得采用。曾遇到FlyDSL动态分支局部变量作用域错误，以及sorted ID容量比expert块容量少几项的边界；修复时不读取无效容量，不放宽容差。

证据：[连续输出/restore](ck_test/sequential_20260912/README.md)。这促成后续让reduce直接消费packed，而非先恢复全中间张量。

## 9. Down＋TOPK reduce整体筛选：采纳的优化

沿用inverse＋sorted_sum的思路：packed down后清空并重建 `[token,slot] -> sorted_row` inverse，定制reduce直接gather各路、FP32累加后转BF16 `[tokens,N]`，省去独立restore。

尝试不同布局、MFMA16/32、torch/custom/reference reduce、线程/列tile、读取/写出cache、preload及求和方式。胜出为 **MFMA16 packed＋256线程/2048列/NT读/cached写/sequential FP32 sum**，每次包括inverse fill/invert。

- 四轮ABBA/BAAB：tuned routed＋torch.sum **928.345627 μs**，packed＋custom **901.830702 μs**，速度+2.94%、时延−2.86%，采纳；不是“再提升10%”达标。
- 另一组同custom reduce消融：routed909.769037 vs packed901.635636 μs，布局额外约+0.90%。不同组不能跨减来分解全部收益。
- MFMA32 folded routing在完整结果产生131/100663296不匹配，拒绝；关闭fold/defer的变体通过但未胜。逐route通过不等于TOPK相消后的结果通过。
- standalone reducer支持过五种布局，独立BF16输入对Torch sum验证；缺失inverse=-1贡献0，原地改变routing及空队列都不能复用旧输出。
- 正常组件诊断一次为down508.522、inverse9.372、reduce365.553 μs，完整910.489 μs；rocprof dispatch记录的down约565.185 μs有采集开销，不能和正常计时混用。

证据：[完整筛选与公平性](ck_test/reduce_20260912/README.md)、[逐kernel口径](ck_test/reduce_20260912/kernel_metrics/README.md)。

## 10. 历史520 μs和10%结果复现核查

用户直接运行routed down得到PyHIP557.588、tuned540.819 μs，参数正确。随后比较当前源码和冻结256-CTA源码，ABBA同址均值535.351196 vs534.495193 μs，仅0.16%差距，两者都未复现520 μs。

又重放当时**512-CTA原始源码＋原始benchmark＋原三轮driver**，并核对源码哈希；当前外部依赖下PyHIP554.610151、历史tuned534.491402 μs，速度+3.76%，未复现10%。未整体恢复旧外部环境/二进制，差异原因没有隔离，不能断言是某个时钟/缓存或特定进程造成，也不否定历史日志本身。

**历史522.627和本次540.819都是routed；约509是packed组件，不能混为同一输出契约。** 证据：[复现与原版重放](ck_test/repro_520_20260912/README.md)。

## 11. 旧FlyDSL BN32/BN64精度修复与目录迁移

默认比较改为down＋reduce后，旧BN32/BN64各出现62处最终不匹配，原因是两处跨K128累加使用分离FP32乘/加，而基线使用FMA。将两处改为显式 `llvm.fma.f32` 后，原容差下全部通过。

确定性例子：partial16和−5，scale2.3968749046325684，分离乘加得到4.015625，FMA得到4.015625476837158；routing0.5后BF16分别是2与2.015625，再加另一路−2就暴露0与0.015625差异。该回归连同graph保留，**没有删旧候选来掩盖失败**。

证据：[失败/修复、前后ISA](ck_test/blockscaled_reduce_20260912/README.md)。随后新增运行模块从原运行库目录移到测试目录、改为 `moe_multistage_*`，本地提交4c1a0cc包含9个必要文件，未push；当时两个指定测试不在提交中。迁移后六候选完整形状全部通过，单次tuned down508.834、完整909.632 μs，仅作为迁移验证。

本次再归档时，旧内核及工具按用户补充要求继续保留主目录，两个文件与归档副本逐字节一致；主测试继续默认比较PyHIP、旧BN32、旧BN64、tuned四项。

## 12. OC数量、nonpersistent与8 XCD

### 12.1 OC sweep

同址正反两轮、其余tuned参数固定：

| OC | Down μs | 完整流程 μs |
|---:|---:|---:|
| 1 | 560.158 | 960.265 |
| 2 | 529.429 | 923.917 |
| 4 | 511.343 | 907.483 |
| 8 | 568.321 | 953.242 |

OC1减少task切换但steady访存压力更大，仍是OC4最好。最初一组后半程down/inverse/reduce整体漂移的日志仍保留，不挑快值、不指定未经隔离的外部原因。

### 12.2 nonpersistent和swizzle

新增one-WG-per-capacity-task、device count guard及8 XCD映射。active前缀按8路transpose，余数和overlaunch保持identity；CPU验证双射，不宣称logical ID固定绑定physical CU。

- 不带swizzle的one-shot约down682.061/完整1061.755 μs，明显回退。
- 带8 XCD的one-shot正反两轮down512.973/完整906.903 μs；persistent509.402/908.577 μs。
- 更严格两轮ABBA/BAAB完整905.276→903.879 μs，仅约0.15%差异，down反而511.143→511.826 μs；未认定稳定优化，persistent默认不变。

证据：[OC/调度/ATT完整记录](ck_test/oc_compact_20260912/README.md)。

## 13. 严格physical-SIMD ATT：32 cycles与39%问题

使用successful issue=attempt+stall、complete=attempt+duration；code按显式PC索引匹配。同一physical SIMD resident waves的MFMA执行窗取并集，4-cycle tick只代表采样粒度。

- gfx950 MFMA16×16×128为**32 cycles**，MFMA32×32×64为64 cycles。参考方法里的gfx942 16-cycle不能直接套用；本轮从最初就按32计算，显式常量专项复核后数值不变，不能再把busy百分比乘二。
- 任务P/S/E与整个resident-wave-batch分开；persistent的长steady包含task切换，不能把它当作98%计算效率。
- 空闲周期按VMEM issue→VMEM wait→LDS issue→LDS wait→VALU→barrier→other互斥归属，并细分正常issue/stall、读写、PC和phase，账本闭合。
- 最终采集SE0/SE1、各CU0、SIMD0..3、两个resident slots；不是全卡capture。OC1/OC4/XCD4各16/16/192个完整wave，task数量3/12/1。

| 指标 | persistent OC1 | persistent OC4 | nonpersistent XCD4 |
|---|---:|---:|---:|
| Task prologue比例 | 6.371% | 21.036% | 18.070% |
| Task steady比例 | 92.733% | 75.967% | 79.135% |
| Task epilogue比例 | 0.896% | 2.997% | 2.796% |
| 内部steady MFMA union | 36.351% | 39.358% | 43.038% |
| 内部VMEM issue owner | 42.650% | 37.916% | 30.410% |

OC4内部steady独立区间复核为1474560/3746500=39.358334%。至少一个wave处于16-MFMA Compute core的并集覆盖56.094%，core内conditional busy70.164%，二者乘积就是39.358%。另外43.906%时间两个wave都在core之外，包含Memory/同步/尾处理，不等于整个GPU空闲。

同core MFMA successful-issue间隔min/p50=40 cycles、均值46.510、p9584；16条MFMA提供512 cycles工作，但core平均729.655 cycles。固定见证中首VALU等8 cycles后七条各4 cycles，下一MFMA相隔40，最后两条VALU没有被前一MFMA的32-cycle窗隐藏。B direct-LDS的2176-cycle raw issue stall仅被peer MFMA覆盖384，暴露1792 cycles。task间gap已被排除，不能拿22%的task gap再次解释内部39%。

结论是优先减少Memory发射停顿、改善错相覆盖，再核对真实VALU依赖/发射间隔；不把原始wave stall相加或局部busy映射为整卡收益。证据、witness和重算日志见 [第2、6、7节](ck_test/oc_compact_20260912/README.md)。

## 14. 动态M256＋M64 compact：正确但更慢

GPU构表直接扫描AITER M256 sorted前缀；4个占用M64 chunk组成一个M256任务，剩余1–3个进入M64表，193个有效行即可形成full任务。两核保持原物理sorted行和同一packed布局，不做重排。M256复用多阶段；M64由旧OCP块缩放内核适配为4 waves，不能直接套用FNUZ/PTPC的1×4数值ABI。

- table为 `[row_begin,expert]`；counts留device、每次重建；host只按shape计算容量，不按count选择grid，不缓存routing内容。
- 末轮利用率按 **full任务数×full OC splits、设备查询256 CU**计算，不复制MI308X参考的80。
- 默认384个M256＋731个M64，padded行145088，较原196608少26.204%；但B每M任务读取次数1115 vs768，增加45.182%。
- full/tail host容量679/1308，OC4 active1536/2924、launch2716/5232，其余early exit。
- 最终正反两轮compact down690.679/完整1081.501 μs，较persistent完整908.577 μs约慢19%。构表约9.24、full417.42、tail266.00 μs为独立诊断，不能相加替代完整计时。
- 后续分项报告对每核单独统计有效/padded FLOPs、实际分配路由行、B每task/每去重专家模型和自己的时间分母。两个核可能共用专家，理想B项不能简单相加冒充整个MoE一次读取。
- 单轮OC4分项：M256416.115 μs、有效/padded743.155 TF/s、逻辑4.415 TB/s；M64266.233 μs、有效387.177/padded552.785 TF/s、task模型5.863/ideal3.813 TB/s。

384 full在OC1是384 WG，末轮128/256=50%、总体等成本容量损失25%；OC4是1536 WG，正好六满轮，无此缺口。阈值0.6在OC1把128个full转为512个tail，得到256 full＋1243 tail，但当时全OC1 down741.622→743.842 μs未改善。模型不是实测occupancy，8-wave和4-wave任务也不能视为同成本。

证据：[compact及分项、CU核对](ck_test/oc_compact_20260912/README.md)。

## 15. M256 OC1＋M64 OC4混合：可行，仍未胜

新增 `tail_num_oc_splits`，默认None沿用full OC。packed的OC/local-N64可合并为global-N64，所以两核不同OC不会改变最终地址，无需restore或额外标签。构表和tail容量仍由full OC1决定，两核均8 XCD。

同址正反两轮：

| Full/tail OC | 阈值 | Full/tail任务 | Full μs | Tail μs | Down μs | 完整 μs |
|---|---:|---|---:|---:|---:|---:|
| 4/4 | 0.6 | 384/731 | 418.257 | 266.274 | 691.011 | 1082.543 |
| 1/1 | 0.6 | 256/1243 | 291.183 | 443.334 | 738.333 | 1141.359 |
| 1/4 | 0 | 384/731 | 451.124 | 266.106 | 725.730 | 1119.301 |
| 1/4 | 0.6 | 256/1243 | 291.476 | 422.301 | 718.026 | 1121.861 |

平衡混合比全OC1 down时延−2.75%、完整−1.71%，但比全OC4慢3.91%/3.63%。不能把291 μs full与原731个tail的266 μs拼接；平衡后1243个tail实际约422 μs。混合未采用为默认，八次大形状全正确，55项定向检查通过。

入口：[tune_schedule.py](tune_schedule.py)的 `compact1_4` / `compact0_1_4`；[日志与报告第9节](ck_test/oc_compact_20260912/README.md)。

## 16. 本次收敛：完整实验留档，主路径只留赢家和基线

主目录精简了多阶段模块内部的BN64/BN256/大K/其他OC/PF/nonpersistent/task-table/其他layout分支；不是仅删除文件后让winner继续依赖实验目录。主down只生成M256/N128/K256、OC4/PF3/persistent256/SC1/packed-NT，主reduce固定packed256线程/2048列；十tensor调用次序仍相同，最终pipeline输出 `[tokens,N]`。

按用户补充要求保留旧内核和工具**原样**，主基准默认四项：PyHIP、旧FlyDSL BN32、旧FlyDSL BN64、tuned。旧基线仍写routed结果并接预分配 `torch.sum(dim=1)`，保留数值、独立down/reduce/总时间、两种带宽及相对PyHIP/BN64的速度比。原本单独测试中的winner graph/empty/舍入/metric/ISA检查合并到主测试；全部旧实验测试则在本目录保留。

验证结果：

- 主目录统一测试 **25 passed**，包含旧BN32/BN64的跨K FMA相消及graph回归；[日志](cleanup_20260912/tests.log)。
- 归档compact/pipeline/ATT账本 **58 passed**；[日志](cleanup_20260912/archive_tests.log)。
- 同址ABBA、同一个归档benchmark比较归档winner与精简winner：down **509.256443 vs509.406599 μs**（约0.03%差异），完整 **908.283001 vs905.132800 μs**；四次大形状均正确，没有显示down退化，不把小幅整体差异当作新的优化收益。[独立复核入口](compare_winner_cleanup.py)、[原始日志](cleanup_20260912/paired_cleanup.log)。
- 新鲜精简版ISA：**185 VGPR、96 SGPR、70660B LDS、零private/VGPR spill/SGPR spill，无AGPR**，12个Memory均无VALU、12个整段Compute均无地址，165个稳态MFMA间隔各7条VALU，输出NT。reduce为41 VGPR/31 SGPR、零LDS/private/spill。[实际down ISA](cleanup_20260912/isa/moe_winner_cleanup_isa_20260912/moe_down_8stage_kernel_0/21_final_isa.s)、[reduce ISA](cleanup_20260912/isa/moe_winner_cleanup_isa_20260912/moe_down_sum_kernel_0/21_final_isa.s)。采集日志里的辅助资源提取曾误报SGPR8，以真实 `.sgpr_count:96` 为准，原日志不篡改。

主入口默认大形状单次验证（不是新多轮验收）：

| 候选 | Down μs | Reduce μs | 完整 μs | 最终不匹配 |
|---|---:|---:|---:|---|
| PyHIP | 552.508 | 397.330 | 1007.765 | 0/100663296 |
| 旧FlyDSL BN32 | 657.492 | 401.579 | 1082.695 | 0/100663296 |
| 旧FlyDSL BN64 | 574.077 | 399.594 | 1020.853 | 0/100663296 |
| 精简winner | 509.110 | 362.917 | 909.969 | 0/100663296 |

[完整日志](cleanup_20260912/default.log)。此次未新增commit/push，未改AITER、GPU设置、精度容差或计时器；没有重采ATT，旧UI与旧源哈希不重标为精简版。

## 17. 如何使用归档

- 主路径使用上级测试；本目录的 [test_blockscaled.py](test_blockscaled.py)是完整旧候选基准，不要和上级同名模块混在同一个Python进程/PYTHONPATH中。
- [pytest.ini](pytest.ini)为归档独立配置，使用 `--import-mode=importlib`，只收集本目录测试并排除历史快照。运行归档pytest时显式 `-c` 指向本配置；主目录配置会排除本目录，避免同名模块串用及重复运行历史样本。
- 脚本直接运行时Python首先搜索它所在目录，因此各调优driver的sibling imports仍有效；本目录保留完整依赖。目录名 `try` 是Python关键字，不使用普通点号import表达式导入该目录。
- [analyze_oc_att.py](analyze_oc_att.py)仅调整了共享ATT方法文件的父目录层数；复用的方法仍来自原attention分析目录。其余原始顶层代码保持归档时版本。
- 深层driver与它的本地数据整体移动，相对父层数不应机械加一；冻结source快照作为当时证据，不保证从任意目录直接执行。旧绝对路径需按实际新位置传参，不批量改写日志/二进制/source快照。
- 按实验选择 [tune_down.py](tune_down.py)、[tune_next10.py](tune_next10.py)、[compare_sequential.py](compare_sequential.py)、[compare_down_reduce.py](compare_down_reduce.py)、[tune_oc.py](tune_oc.py)、[tune_schedule.py](tune_schedule.py)。性能比较不要跨批次拼接最快组件。

## 18. 历史产物总索引

| 顺序 | 原始产物/说明 |
|---:|---|
| 1 | [初版五候选](ck_test/blockscaled_compare_20260911/README.md) |
| 2 | [BN128四阶段](ck_test/bn128_4stage_20260911/README.md) |
| 3 | [cached/VGPR-only](ck_test/cached_vgpr_4stage_8stage_20260911/README.md) |
| 4 | [PF2](ck_test/prefetch2_4stage_8stage_20260911/README.md) |
| 5 | [整段地址提前](ck_test/address_hoisted_4stage_8stage_20260911/README.md) |
| 6 | [早期BN64 PF2](ck_test/bn64_pf2_20260911/README.md) |
| 7 | [初步ATT优化采集日志](ck_test/opt_profile_20260911/profile.log) |
| 8 | [BN128最终组合与历史10%](ck_test/att_optimized_bn128_20260911/README.md) |
| 9 | [CTA256/512](ck_test/cta_ablation_20260912/README.md) |
| 10 | [MFMA32及所有邻域尝试](ck_test/next10_20260912/README.md) |
| 11 | [连续写出/restore](ck_test/sequential_20260912/README.md) |
| 12 | [down＋reduce筛选](ck_test/reduce_20260912/README.md) |
| 13 | [520 μs与历史原版复现](ck_test/repro_520_20260912/README.md) |
| 14 | [旧BN32/BN64 FMA修复](ck_test/blockscaled_reduce_20260912/README.md) |
| 15 | [OC/XCD/ATT/compact/混合OC](ck_test/oc_compact_20260912/README.md) |

当前优先改进方向来自已有证据：Memory发射停顿和两组wave覆盖，而非单纯继续增加stage、减少OC或只追求更少padded FLOPs。所有未采用方案均保留供后续复验，不把正确但更慢的实验删除出历史。