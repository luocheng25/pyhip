# moe down kernel - BlockScaled(128x128)

## 直接运行：down与完整流程分开报告

[test_blockscaled.py](test_blockscaled.py) **现在默认测down＋TOPK reduce**，形状和六个默认候选不变：

- PyHIP、旧FlyDSL BN32/BN64、generic四/八阶段：原routed down后接预分配输出的 `torch.sum(dim=1, out=...)`。物理中间布局是 `[tokens,topk,N]`，TOPK为dim1。
- `4stage_bn128_tuned`：采用已筛选的 **MFMA16 packed down＋定制reduce**，而不是routed down；256 CTA、split4/PF3，reduce为256线程/2048列/NT读取。每次完整调用都清空并重建inverse。
- `Down time (us)`、`Down effective/padded TF/s` 和两种 `Down TB/s`：**独立测量同一完整流程的down组件**，包含counter清零，不包含inverse和reduce。
- `Invert + fill time (us)`、`Reduce time (us)`：独立诊断，无inverse的候选显示N/A。
- `Reduce TB/s`：紧接reduce耗时，按 `(tokens * topk * N + tokens * N) * sizeof(BF16) / reduce_us / 1e6` 计算，只计有效中间数据读取与最终输出写入；不计padding容量、down权重、inverse构建或其他元数据流量。使用独立reduce耗时而非总耗时，未测量时为N/A，属于逻辑带宽估算。
- `Total Time (us)`、`Total Effective/Padded TF/s` 和两种 `Total TB/s`：**直接计时down＋reduce＋必要的inverse清空/重建**，不是将组件相加。TF/s只以GEMM工作量为分子；总带宽另外计入reduce读写和inverse主要流量，其他小元数据未计入，仍为逻辑估算而非HBM计数器。

默认候选共享同一份中间buffer和最终 `[tokens,N]` 输出分配；最终输出数值验证保留原容差。原 `run_perftest`、warmup2/iters10不变。完整调用先计时，组件随后分别计时，不能从独立组件的和推导总时间或收益。

`--mode down` 保留以前的纯down/routed比较；`--mode down-reduce` 显式选择新默认流程。`--profile` 不指定mode时仍为历史down-only的23次发射，指定 `--mode down-reduce` 才采集完整流程。
导入调用 `run_test` 的默认 `reduce_output=False` 不变，避免悄悄改变历史tuning脚本；传 `reduce_output=True` 时，返回的 `elapsed_us` 为总时间，`down_elapsed_us` 与 `down_*` 为组件性能。
MFMA32显式候选在完整流程模式关闭folded-routing/defer-K1重新结合，以保留已验证的TOPK-sum精度；不改变纯down实验配置。

**旧BN32/BN64的reduce校验失败已修复。** 两处跨K128块反量化累加由分离FP32乘/加改为显式FMA，与PyHIP/tuned舍入一致，避免BF16中点差异经TOPK相消后放大。默认大形状六候选全部PASS，旧候选各62处不匹配降为0；17项测试通过，未放宽容差或删除候选。[失败/修复日志、确定性回归及前后ISA](ck_test/blockscaled_reduce_20260912/README.md)。

## down＋最终TOPK reduce整体选择

**2026-09-12重新按整体筛选：采用显式 `compile_packed_down_reduce`。** 当前tuned写出＋torch.sum四轮平均
**928.346 μs**，MFMA16 packed写出＋定制reduce（含inverse清空/重建）**901.831 μs**，
速度 **+2.94%**、时延 **−2.86%**；四轮均领先，未达到此前10%目标。相同定制reduce下，packed比routed额外快约0.90%。
新入口输出BF16 `[tokens,N]`，内部已对TOPK求和；权重仍为(16,16)，保留原MFMA16浮点次序。
[整体API](moe_multistage_pipeline.py)、[比较入口](compare_down_reduce.py)、
[完整条件与日志](ck_test/reduce_20260912/README.md)。原down-only默认输出 `[tokens,topk,N]` 不变，不自动替换已有调用方。

## 直接比较当前与四/八阶段 kernel

地址提前实现现位于本目录的 [moe_multistage_down.py](moe_multistage_down.py)。测试直接使用该模块，不再向pyhip运行库安装实验实现；本地 [moe_8wave_down_8stage.py](moe_8wave_down_8stage.py) 仅保留旧测试导入路径兼容，接口及默认四/八阶段不变。

**最新推荐：`4stage_bn128_tuned` / `ATT_TUNED_BN128_CONFIG`**。BN128四阶段，split4、256 CTA、预取3拍、连续128B NT写、B cache SC1。历史512 CTA版本（2026-09-11）的同输入/同输出地址三轮平均 **522.627 μs vs PyHIP 575.497 μs，1.101×（速度+10.12%，时延−9.19%）**；单轮有波动，不保证每轮10%。该版本数值、6项定向测试及实际ISA通过；完整日志和当时ATT见 [优化结果](ck_test/att_optimized_bn128_20260911/README.md)，不重标为当前256 CTA配置。

[test_blockscaled.py](test_blockscaled.py) 的默认比较已包含该预设，原工厂默认不变。[tune_down.py](tune_down.py) 可运行三轮交错比较，但不保证复现历史时延，不改变计时器或容差。下列旧试验数字仅作历史参考。

**2026-09-12直接运行复核：当前未复现约520 μs。** 用户同轮输出为PyHIP **557.588 μs**、tuned **540.819 μs**（**1.031×**），选中的split4/CTA256/PF3/NT/SC1参数正确。随后同输入、同输出地址的current/frozen/frozen/current对照，当前源码平均 **535.351 μs**，冻结256 CTA源码 **534.495 μs**，相差 **0.16%**，全部正确；冻结版也未复现520 μs。未发现足以解释该差距的明显源码回退，未单独隔离时钟/缓存/主机提交等影响，不能将其断言为原因。历史 **522.627 μs同样是routed down**；另一次packed GEMM组件的 **508.522 μs** 不能替代它或完整down＋reduce时间。[复现脚本、完整日志与计时范围](ck_test/repro_520_20260912/README.md)。

**当时+10.12%的512 CTA原版也已重放：本次仅+3.76%。** 校验并加载历史kernel快照及配套benchmark/三轮driver，使用当前外部依赖，同GPU5、形状和原计时方法；三轮PyHIP平均 **554.610 μs**、历史512 CTA tuned **534.491 μs**（**1.037641×**），九次候选检查全部正确。不是拿后来的256 CTA版本代替原版，也未丢弃较慢轮次。该结论仅说明当前环境下未复现10%，不说明历史日志无效；旧环境或GPU二进制未整体还原，差异原因尚未隔离。[历史原版重放入口](ck_test/repro_520_20260912/reproduce_historical.py)、[本次完整日志](ck_test/repro_520_20260912/historical512_paired.log)。

带宽统计并列显示 `Valid expert blocks` 和 `Unique experts`。去重只读取 `sorted_expert_ids[:valid_expert_blocks]`，排除未初始化的容量尾部；`TB/s (B per M-block)` 保留原口径，`TB/s (B once/expert, ideal)` 假设每个参与专家的整份权重只从HBM读取一次、后续访问全部缓存命中。两者仅替换B流量项，A/输出及reduce流量口径不变，均不是实测HBM计数器。去重不改变实际计算量或TFLOPS。

**2026-09-12 CTA单变量验证：256→512没有稳定收益。** 固定tuned BN128的split4、预取3拍、NT/SC1及全部其他参数，四轮ABBA/BAAB平均为 **256 CTA 523.787 μs、512 CTA 524.048 μs**，512时延约 **+0.050%**；两轮略快、两轮略慢，不能把此前组合优化收益归因于CTA翻倍。全部正确；验证后按要求将tuned预设改为256 CTA，其余参数不变，历史512 CTA日志和ATT保留。[完整条件及原始日志](ck_test/cta_ablation_20260912/README.md)；[tune_down.py](tune_down.py) 的 `--compare-ctas 256 512 --rounds 4` 可重现该单变量比较。

**MFMA32再提升10%实验：未达标。** `32x32x64`、`shuffle(32,16)`、严格1 MFMA＋14 VALU、寄存器128B写出已实现，最终四轮同址交错比较 **528.472→523.609 μs，速度+0.93%**，不是10%。仅保留显式候选 `4stage_bn128_mfma32`，不加入默认比较或替换tuned；原MFMA16源码逐字节恢复。6项定向测试及零scratch/无AGPR/Memory无VALU/Compute无地址ISA通过。该实验组合使用FP32 routing-scale重新结合，非bitwise等价。[全部结果与新ATT](ck_test/next10_20260912/README.md)。

**旧连续写出＋完整restore实验：GEMM本体改善，端到端回退。** 两轮快测MFMA32直接写出 **522.823 μs**；N64 packed临时输出的GEMM诊断 **491.090 μs**，但恢复原布局需 **746.787 μs**，完整调用 **1231.731 μs**。该结论不包含后续TOPK sum；上面的新整体路径将恢复与sum合并，避免这次完整中间读写。保留 [compare_sequential.py](compare_sequential.py) 历史实验入口。[条件、日志及ISA](ck_test/sequential_20260912/README.md)。

**BN64 四阶段实验**改按 `prefetch_distance=1/2/3` 比较：Memory q 发 Q[q+d]，实际稳态 `vmcnt` 分别为 **0/3/6**，不是原来的3/6/9。GPU5 同轮快测为1320.784/641.919/670.532 μs；同场 BN128/BN256 对照为581.773/562.968 μs，全部正确。性能候选为 `4stage_bn64_pf1`、`4stage_bn64_pf2`、`4stage_bn64_pf3`；BN64 默认2拍，不改变原默认五项。

**BN256 十六阶段可选实验**：`bn256_stages=16, steady_vmcnt=6/9`，每个稳态 Memory 为2 store＋1 direct-LDS load。GPU5 同轮快测：八阶段573.053 μs，十六阶段 vmcnt6/9 为577.605/580.705 μs，均正确；暂未显示收益，保留八阶段默认。用 [test_blockscaled.py](test_blockscaled.py) 的 `--candidate 8stage_bn256 16stage_bn256_vmcnt6 16stage_bn256_vmcnt9` 比较，结构/等待审计和说明见 [README_8stage.md](README_8stage.md)。

**旧地址提前基线为2拍 B 预取**，NT/AGPR禁用；Memory无VALU、七条交织及零scratch保留，VGPR为177/239。当时GPU5默认形状单轮快测为568.173/574.977 μs；[旧地址提前版 ATT](ck_test/address_hoisted_4stage_8stage_20260911/README.md) 保留。此前566.600/565.380 μs及 [旧2拍 ATT](ck_test/prefetch2_4stage_8stage_20260911/README.md) 均为地址提前修改前的结果。

运行 [test_blockscaled.py](test_blockscaled.py) 即比较PyHIP、旧FlyDSL BN32/BN64、generic四/八阶段及优化BN128的down与完整down＋reduce，显示实际split/CTA/cache/输出布局、正确性和两类计时。新down均禁用AGPR；generic默认cached，优化候选显式启用NT及packed布局。所有默认候选使用同一输入/中间/最终输出分配及相同计时方法；默认形状不变，保留 `8stage_bn128` 选择键。以前的纯down数字需用 `--mode down` 对照。

新内核说明和本次结果见 [README_8stage.md](README_8stage.md)。[2拍 ATT](ck_test/prefetch2_4stage_8stage_20260911/README.md)、[3拍 cached/VGPR ATT](ck_test/cached_vgpr_4stage_8stage_20260911/README.md)、[旧四阶段 NT/AGPR 产物](ck_test/bn128_4stage_20260911/README.md) 和 [旧五候选产物](ck_test/blockscaled_compare_20260911/README.md) 保留为历史。使用 `--candidate` 可单选或多选；`--profile` 为单候选 ATT 采样模式，不报告性能时间。

FlyDSL语义复杂，需要使用COT逐步逼近目标。

Copilot帮我们实现了基本调度框架，但是实现高效的主Pipeline还是比较困难，因此我们需要逐个解决问题：

## workgroup 多线程协作加载scales

## 2026-09-12：迁回测试目录，整理本地提交

本阶段按实验代码管理，将此前新增的五个运行库文件移入本目录，统一使用 `moe_multistage_*` 名称：

| 模块 | 用途 |
|---|---|
| [moe_multistage_down.py](moe_multistage_down.py) | MFMA16四/八阶段down，含tuned与packed写出 |
| [moe_multistage_down_mfma32.py](moe_multistage_down_mfma32.py) | 独立MFMA32/14-VALU实验，不替换默认tuned |
| [moe_multistage_reduce.py](moe_multistage_reduce.py) | 各输出布局的定制TOPK reduce |
| [moe_multistage_pipeline.py](moe_multistage_pipeline.py) | down＋inverse清空/重建＋reduce及共享workspace |
| [moe_multistage_restore.py](moe_multistage_restore.py) | 连续写出实验的独立restore，非默认整体路径 |

此次仅迁移文件与更新模块导入；工厂/十tensor调用接口、tile、流水调度、cache参数、CTA数及计算顺序不变。
模块之间沿用测试目录的sibling import，脚本可直接运行；从其他程序导入时将本目录放入PYTHONPATH。
继续复用已存在的pyhip helper与inverse实现，不复制或移动它们。`test_blockscaled.py`不依赖两个仅保留本地的回归测试文件。

### 优化过程汇总与当前选择

1. 以k128n为蓝本实现8-wave direct-global→LDS流水，随后将BN128拆为两个Memory/Compute对；延迟写出均分到Memory，BF16打包留在寄存器。
2. 通过scale/routing缓存、地址提前与寄存器约束去除scratch/AGPR及Compute地址运算；tuned采用split4、PF3、连续128B NT写和B SC1。BN64 PF1/2/3、BN256十六阶段未显示稳定优势。
3. 历史512 CTA组合曾取得PyHIP575.497→522.627 μs（+10.12%）；CTA256/512单变量没有收益，最终保留256。当前环境重放历史原版为554.610→534.491 μs（+3.76%），不继续宣称稳定10%。
4. MFMA32/14-VALU进一步优化只有约+0.93%；packed GEMM本体更快，但独立restore使端到端回退。改按down＋TOPK reduce整体筛选后，采用MFMA16 packed＋定制reduce，四轮928.346→901.831 μs（+2.94%）。
5. 默认比较现同时报告down、reduce带宽和真实整体耗时；专家权重流量并列每M块读取与去重专家理想缓存两种口径，均非HBM计数器。旧BN32/BN64两处跨K反量化累加统一为显式FMA，修复TOPK相消后的62处不匹配。

迁移不重写或重标历史日志、ATT、ISA和源码快照；此前的哈希和性能数字只对应各自当时版本。
这些产物、额外调优/复现脚本和详细阶段说明均保留本地，不属于本次必要运行文件提交范围。
按要求，[test_8stage.py](test_8stage.py) 与 [test_down_reduce.py](test_down_reduce.py) 仅更新本地导入、用于验证，**不暂存、不提交**；依赖前者的MFMA32回归脚本也保留本地。

### 迁移后验证

- 五个旧运行库路径均已移除；导入检查确认五个新模块均来自本测试目录，benchmark不导入两个排除提交的回归文件。
- pipeline/舍入边界/CLI/统计及MFMA32定向回归：**23 passed**；tuned短N/split/graph定向检查：**6 passed**。
- packed restore的N512数值与graph检查通过。
- GPU5直接运行默认大形状（tokens16384/TOPK8/E384/N6144/K256）：**六候选全部PASS**，各 **0/100663296** 最终输出不匹配；down、reduce带宽及总时间均正常输出。

| 本次迁移验证（单轮） | PyHIP＋torch.sum | tuned packed＋custom reduce |
|---|---:|---:|
| Down μs（含counter reset） | 558.143 | 508.834 |
| Down有效 / padded TF/s | 738.729 / 1108.094 | 810.317 / 1215.476 |
| Inverse清空＋重建 μs | 不需要 | 9.292 |
| Reduce μs | 401.606 | 363.593 |
| Reduce逻辑TB/s | 4.512 | 4.983 |
| 完整调用实测 μs | 1015.491 | 909.632 |

这是迁移正确性及入口验证，不是新的多轮性能验收。组件分别计时，不能相加替代总耗时；没有修改warmup2/iters10、容差、时钟或硬件设置。

## 2026-09-12：OC数量、任务切换ATT与M64/M256动态拆分

观察到task/OC之间大空洞后，新增 [OC比较](tune_oc.py)、[调度比较](tune_schedule.py) 和 [physical SIMD分析器](analyze_oc_att.py)，保持同一数据/地址及原计时。OC1/2/4/8正反两轮的down为 **560.158/529.429/511.343/568.321 μs**，完整流程为 **960.265/923.917/907.483/953.242 μs**：减少OC虽减少切换次数，但当前形状仍是OC4最快。

按指定stall方法，fresh ATT使用successful issue、32-cycle gfx950 MFMA窗、4-cycle tick与同SIMD resident-wave并集。最终OC1/OC4/XCD4的每task头部占比为 **6.371%/21.036%/18.070%**、尾部 **0.896%/2.997%/2.796%**，内部steady MFMA busy为 **36.351%/39.358%/43.038%**。OC4 task间MFMA gap累计占采样活动时间22.247%，OC1降为5.199%；但OC1稳态VMEM issue更重，不能仅凭少切换推断墙钟更快。分析另列wave-batch与task两种生命周期、静态容量模型、七类互斥stall及完整断言；采样局部效率不冒充整卡利用率。

[M256内核](moe_multistage_down.py)新增显式 `persistent=False`、`xcd_swizzle=True` 和task-table输入，按MI350X **8 XCD**置换active前缀，其余identity；默认仍persistent。新 [compact构表/组合](moe_multistage_compact.py)参考8x1 compact，在device动态生成 `[row_begin,expert]` M256/M64表，保持原sorting物理行与packed输出；M256使用现有多阶段kernel，M64由本地OCP块缩放FlyDSL复制适配为 [M64尾核](moe_multistage_down_m64.py)，没有直接套用FNUZ/PTPC的1x4数值ABI。构表每次重建并计入时间，不读取device count来决定host grid。

默认shape为384个M256＋731个M64，计算padding从196608降至145088行（−26.20%），但权重task增加45.18%。最终正反两轮：persistent / XCD one-shot / compact的完整流程 **908.577/906.903/1081.501 μs**，compact仍回退约19%；M64构表/两个kernel带来的成本未被算量节省抵消。XCD单独两轮ABBA/BAAB仅约0.15%总时间差异，**不晋级为稳定收益**，当前默认预设不变。

51项定向检查通过，最终容量/slot更新后19项复验通过；M256任务表packed ISA继续满足无atomic/AGPR/scratch、Memory无VALU、Compute无地址和165个七VALU稳态间隔。完整结果、范围限制、异常轮次、最终fresh UI、源码快照和所有日志见 [本轮记录](ck_test/oc_compact_20260912/README.md)。本轮未提交或推送，两个指定本地测试文件继续不进入提交范围。

**32-cycle专项核对：** 当前MFMA执行窗为successful issue后32 cycles，等效槽为cycles/32；没有套用参考文档的gfx942 16-cycle窗口。分析器现显式固定32-cycle并拒绝16-cycle输入，表头/JSON明确区分32条MFMA/N与32-cycle/指令。三组既有ATT重算后busy和头尾周期均与前报一致，steady busy仍为36.351%/39.358%/43.038%，不能再次翻倍；3项CPU测试通过，未重采或重跑GPU。[逐项核对记录](ck_test/oc_compact_20260912/README.md#6-32-cycle口径专项核对)。

**低steady busy的进一步拆解：** 独立区间并集复核OC4仍为39.358%，但这是整个Memory＋Compute内部循环，不是纯Compute段。至少一条wave在16-MFMA Compute段内的时间覆盖56.094%，这些段的conditional MFMA busy为70.164%，乘积即39.358%；其余43.906%的时间两个wave均在Compute段之外。动态MFMA间隔中位数40 cycles、均值46.510 cycles，说明静态7-VALU交织并未全部隐藏。实际B direct-LDS issue-stall示例2176 cycles仅被peer MFMA覆盖384 cycles，仍暴露1792 cycles。task切换gap已排除，不能再次拿头尾开销解释该内部steady值。[分母、周期分解与固定N块见证](ck_test/oc_compact_20260912/README.md#7-为什么oc4稳态mfma-busy只有3936)。

## 2026-09-12：compact分项性能与256 CU末轮核对

[benchmark](test_blockscaled.py)现在为compact的M256、M64分别打印实际任务/有效行/padded行、独立耗时、有效/padded TF/s、两种逻辑TB/s，以及各自的OC/XCD、active/launch/early-exit WG和末轮容量模型。返回值新增 `component_metrics`，全部分子来自当前device任务表，分母是对应kernel自己的独立计时；统计读取在计时后，不用于host grid或任务决策。

GPU5查询为 **MI350X、256 CU**，生产构表与容量上界共同使用 `device_cu_count`，不写死80。**M256和M64均为nonpersistent，且都启用8 XCD swizzle**；不是只给M256加XCD。默认形状同前，warmup2/iters10，以下为单轮分项验证，不是新性能晋级：

| compact OC4 | M256 | M64 |
|---|---:|---:|
| M任务数 / 有效行 | 384 / 98304 | 731 / 32768 |
| Padded行 | 98304 | 46784 |
| 独立耗时 μs | 416.115 | 266.233 |
| 有效 / padded TF/s | 743.155 / 743.155 | 387.177 / 552.785 |
| TB/s：B每M任务读取 | 4.415 | 5.863 |
| TB/s：B每去重专家一次（理想） | 4.415 | 3.813 |

**384个M任务不一定等于384个WG。** OC1对应384 WG，等成本一WG/CU模型下末轮128/256=50%，两轮总体容量损失25%；阈值0.6会将最后128个M256改成512个M64，得到256个M256＋1243个M64。OC4对应1536 WG，恰好6个256-CU满轮，不应仅因384这个M任务数而再拆分。该模型不是实测occupancy或物理CU绑定。

OC1平衡前/后，M256独立时间443.927→294.482 μs，但M64为294.686→439.567 μs；含构表的down实测741.622→743.842 μs，未显示整体改善。OC4 down实测688.196 μs，完整流程1082.673 μs；组件不能相加替代这些独立整段计时。带宽计A一次、B每M任务或每子kernel去重专家一次、有效BF16写出，排除元数据及重复A加载，不是HBM计数器；两子kernel共享专家，理想B流量不能相加冒充全流程去重流量。

43项定向回归通过，覆盖384任务在256/80 CU、OC1/2/4下的拆分差异、分项公式、空任务及graph replay；三项默认大形状候选均0/100663296不匹配。未改默认OC4/persistent预设、GPU计时器、容差或硬件设置，未新建commit。[完整分项表、定义和原始日志](ck_test/oc_compact_20260912/README.md#8-compact分项指标与256-cu末轮)。

## 2026-09-12：M256 OC1＋M64 OC4混合配置

**可以混用，已实现并验证。** [compact工厂](moe_multistage_compact.py#L176)新增 `tail_num_oc_splits`，默认None沿用M256的 `num_oc_splits`；显式设置 `num_oc_splits=1, tail_num_oc_splits=4` 即M256 OC1＋M64 OC4。packed布局的OC/local-N64维可以合并为global-N64，最终地址与OC数无关，因此不需要重排、额外buffer或第二套inverse/reduce。两个kernel均保持8 XCD swizzle和原数值运算。

构表和tail容量仍按M256的OC数及硬件查询256 CU计算；默认阈值0.6时，384个full任务中的128个转为512个M64，最终256个M256＋1243个M64。分项统计使用各自真实OC数，不将M64的OC4误用作M256的末轮平衡依据。

同默认形状、同输入/中间/输出地址、warmup2/iters10，正反两轮均值：

| M256 OC / M64 OC | 末轮阈值 | M256 / M64任务数 | M256 μs | M64 μs | Down整段 μs | 完整流程 μs |
|---|---:|---:|---:|---:|---:|---:|
| 4 / 4 | 0.6 | 384 / 731 | 418.257 | 266.274 | **691.011** | **1082.543** |
| 1 / 1 | 0.6 | 256 / 1243 | 291.183 | 443.334 | 738.333 | 1141.359 |
| 1 / 4 | 0（不平衡） | 384 / 731 | 451.124 | 266.106 | 725.730 | 1119.301 |
| **1 / 4** | **0.6** | **256 / 1243** | **291.476** | **422.301** | **718.026** | **1121.861** |

混合且平衡后，M256有效/padded **707.290/707.290 TF/s**、逻辑带宽 **4.202 TB/s**；M64为 **488.179/592.584 TF/s**、B每M任务 **6.576 TB/s**、每去重专家一次理想 **3.377 TB/s**。这些按各组件两轮平均时间计算，不是实测HBM流量；整段时间仍单独测量，不相加。

混合比统一OC1的down时延降低 **2.75%**、完整流程降低 **1.71%**，但比统一OC4分别慢 **3.91% / 3.63%**。不能将M256的291 μs与旧731个tail任务的266 μs拼接：平衡后tail变成1243个，实测约422 μs。保留显式候选 `compact1_4`（平衡）和 `compact0_1_4`（不平衡），默认未改；[调度入口](tune_schedule.py)可比较这两项。55项定向测试、八次大形状检查全部通过，未改计时/容差、GPU设置或提交代码。[完整记录和日志](ck_test/oc_compact_20260912/README.md#9-m256-oc1与m64-oc4混合)。

