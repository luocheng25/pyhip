# 8-wave down：BN64/128 四阶段 / BN256 八阶段及十六阶段实验

当前实验实现：[moe_multistage_down.py](moe_multistage_down.py)，入口为 `moe_multistage_down.flydsl_moe_gemm_8wave_down`。2026-09-12统一迁回测试目录；本地 [moe_8wave_down_8stage.py](moe_8wave_down_8stage.py) 仅保留直接重导出，不再维护重复实现。默认四/八阶段不变，BN256 十六阶段实验需显式选择。

本地测试及 ISA 审计：[test_8stage.py](test_8stage.py)。性能比较：[test_blockscaled.py](test_blockscaled.py)；两者均直接导入同目录模块。下文历史迁移验证与快照记录保留当时口径。旧 [moe_8wave_down.py](moe_8wave_down.py) 及 [moe_8wave_down_utils.py](moe_8wave_down_utils.py) 包含当前 FlyDSL API 兼容性修正；随后另将两处跨K累加改为显式FMA以修复TOPK-sum舍入问题，未改变流水线。

## 最新 ATT 优化预设

`ATT_TUNED_BN128_CONFIG`：**BN128四阶段、OC split4、256 CTA、预取3拍、每行128B连续NT写、B读取SC1、先发LDS读、不切换stage优先级**。只改变访问布局和调度，保留原浮点运算次序、BF16舍入、错相barrier、B direct-LDS、Compute无地址计算；不使用AGPR或C LDS。

2026-09-12固定其余参数的四轮CTA单变量验证：256 CTA **523.787 μs**，512 CTA **524.048 μs**，没有测到512的稳定收益。因此tuned预设改为256 CTA，每CTA仍为512 threads / 8 waves；[验证记录](ck_test/cta_ablation_20260912/README.md) 保留全部轮次。

历史512 CTA版本（2026-09-11），GPU5默认形状的同输入/同输出地址三轮比较：PyHIP **575.497 μs**，优化版 **522.627 μs**，平均 **1.101162×（速度+10.12%，时延−9.19%）**；单轮1.054–1.129×，并非每轮保证10%。完整记录含未达标轮次与当时ATT见 [结果和产物索引](ck_test/att_optimized_bn128_20260911/README.md)。旧ATT及源码快照保持512 CTA，不重标为当前配置；本次未重新抓ATT。

- `--candidate 4stage_bn128_tuned` 选择优化预设；已加入直接运行的默认比较，原generic工厂默认不变。
- [tune_down.py](tune_down.py) 默认做3轮交错比较，计时器/容差不变，全部结果计入均值。所有候选共享输入和输出地址，NaN投毒在计时外，counter reset在计时内。
- 旧512 CTA版本的原生及ATT ISA均通过四阶段、整段Compute无地址、Memory零向量、七条交织、NT、无AGPR和零scratch审计；**185 VGPR、100 SGPR、70660 B LDS，无spill**。
- 6项新增短N/split/sorting/graph检查通过；原子覆盖写、静态任务、routing重新结合等无稳定收益实验已从正式代码移除并归档。

## down＋TOPK sum整体预设（2026-09-12）

[moe_multistage_pipeline.py](moe_multistage_pipeline.py) 提供 `compile_packed_down_reduce`
与 `PACKED_DOWN_REDUCE_CONFIG`：仍为tuned MFMA16 M256/N128/K256、8waves、256CTA、split4、PF3，
仅显式改为packed输出；256线程/2048列定制reduce从各TOPK路收集数据、FP32求和并输出BF16 `[tokens,N]`。
reduce读取aux2/NT、写回aux0；每次清空并重建inverse，计时包含全部步骤，无完整restore。

四轮ABBA/BAAB：原down＋torch.sum **928.346 μs**，packed＋定制reduce **901.831 μs**，
速度 **+2.94%**，按本次整体更快的准则采纳；不是此前10%目标达标。相同定制reduce下，packed再快0.90%。
9项定向数值/graph/动态routing/空队列检查通过，原容差不变。MFMA32重新结合方案在最终sum后出现131个超容差元素，
已拒绝；选中路径不改原MFMA16数学次序。[报告与日志](ck_test/reduce_20260912/README.md)。

原down工厂默认仍为 `output_layout="routed"`，原接口不变；`sorted/packed`是BN128/K256的显式选项，
中间tensor逻辑形状为 `[expert_blocks*256,N]`，不能直接torch.sum其物理布局。
整体入口的最终output必须为 `[tokens,N]`，之后不再另调用sum。当前测试TOPK在dim1，见实际shape而非硬编码dim2。

## 后续MFMA32实验（2026-09-12）

独立实现已迁为 [moe_multistage_down_mfma32.py](moe_multistage_down_mfma32.py)，
原tuned MFMA16不变。MFMA32使用(32,16)权重、两条K64累计为K128 partial、下一Compute退休K1反量化，
实际稳态严格1 MFMA＋14 VALU；209 VGPR、97 SGPR、70660B LDS，零scratch/无AGPR。
最终组合启用FP32 routing-scale重新结合、3个row系数缓存和全padding wave跳算，保持所有DMA/barrier及输出mask。
原参考和容差下6项定向测试通过；默认形状四轮同址ABBA/BAAB **528.472→523.609 μs，速度+0.93%**，
**没有达到相对当前tuned再提升10%的目标**。显式候选 `4stage_bn128_mfma32` 不进入默认列表。
[详细结果、失败试验与新ATT](ck_test/next10_20260912/README.md)。

### 连续写出实验

MFMA32另有显式 `output_layout=sorted/packed/linear/linear_raw` 临时布局，默认仍为 `routed`。
`linear`在完整有效wave上每条store连续1024B；`linear_raw`进一步去掉GEMM内permlane/DPP，
恢复步骤负责还原逻辑元素。两轮快测：MFMA32直接写出522.823 μs，packed的GEMM本体491.090 μs，
恢复746.787 μs、完整1231.731 μs，**没有端到端收益**。原MFMA16和默认候选不变。
[compare_sequential.py](compare_sequential.py) 报告独立组件和真实完整计时，不以组件之和替代总耗时。
[完整说明及结果](ck_test/sequential_20260912/README.md)。本次未抓新ATT。

## 地址提前基线（优化前历史）

> **基线默认2拍预取，整个 Compute 阶段无地址运算。** lane 地址统一在任务准备区计算并保留；必要时按两个 N 展开以固定 ring slot。动态 N 的 B scale 读取使用 Memory 内的 `ds_read_addtid_b32` 标量寻址，Compute 只广播系数、不计算地址。inline ASM 显式处理 M0 写后两周期等待，并保存恢复 M0，不影响 B direct-LDS DMA。

GPU5，tokens513/N6144/K256/E8/topk8，两种 BN 的数值及 graph 重放均 PASS；实际 ISA 的**整个 Compute（含尾部）**通过 `--require-no-compute-address`，Memory 无向量指令，七条交织、cached store、无 AGPR、零 scratch 均通过。

| 优化前基线 | 展开 Compute 段 | VGPR | SGPR / spill | LDS bytes |
|---|---:|---:|---:|---:|
| 四阶段 BN128 | 12 | 177 | 104 / 0 | 70660 |
| 八阶段 BN256 | 16 | 239 | 106 / 5 | 70660 |

随后 GPU5 默认形状单轮同进程快测：四阶段590.314→568.173 μs（−3.75%），八阶段572.637→574.977 μs（+0.41%），四项数值均 PASS。[当前地址提前版 ATT](ck_test/address_hoisted_4stage_8stage_20260911/README.md) 已采集，各8个 wave；未运行完整回归，较大 K 路径尚未重新验收。此前2拍快测的566.600/565.380 μs及 [旧2拍 ATT](ck_test/prefetch2_4stage_8stage_20260911/README.md) 均属于**地址提前修改之前**。下文详细流水线和旧验证数据保留为3拍基线。

## 接口和范围

- 工厂及返回的十个 tensor 参数与原实现同名、同序；默认 `block_m=256, block_n=256, num_oc_splits=1`。
- `ATT_TUNED_BN128_CONFIG` 是显式可选的BN128/K256预设，N须能整除512；不替换不支持形状的generic路径。
- BN64 实验参数：`block_n=64, prefetch_distance=1/2/3`（默认2）；仅支持 K256。BN64 不再通过 `steady_vmcnt` 反推预取距离。
- BN256 实验参数：`bn256_stages=16, steady_vmcnt=6` 或 `9`，仅支持 BN256/K256；默认仍为 `bn256_stages=8`。
- **gfx950、OCP FP8 E4M3FN 输入/权重、BF16 输出**。A 的 scale 是物理 K-major 的 `1×128`，B 的 scale 是 `128×128`，权重使用 `shuffle_weight(layout=(16,16))`。
- BN64/256 支持 K256；BN128 支持 K256/384/512/640。N/splits 必须整除 BN；支持只有 1/2 个 N tile，不要求至少 3 个 N tile。
- **BN128/K256 使用两个 Memory/Compute 对，共四阶段，每个 Compute 16 条 MFMA/wave。** BN256/K256 保留四对、八阶段；BN128 的较大 K 保留原 BK128 路径。
- **所有分支仍为VGPR-only**；generic默认cache aux0，优化预设输出aux2（NT）、B读取aux16（SC1）。`llvm.passthrough` 固定 `amdgpu-agpr-alloc=0,0`，显式 AGPR 约束和搬运已移除；`waves_per_eu=2` 本身不是禁用 AGPR 的保证。
- generic默认及当前优化预设均为persistent256 CTA，每CTA512 threads（8 waves）；counter在当前stream清零。输出已乘routing weight，TOPK不在此kernel内求和。
- 连续 tensor、32-bit buffer offset；显式检查 dtype、shape、device 和寻址范围，不静默 fallback。

## BN64 四阶段：提前1/2/3拍（2026-09-11，GPU5）

每个 wave 计算 M32×N64，完整 N tile 写4 KiB；B 的协作加载份额为2 KiB/wave。分成两个 Memory/Compute 对，每 Compute 8条 MFMA，每个稳态 Memory 写上一 N 的一半：**2条 store＋1条 B direct-LDS load**。B 包为 N32×K256=8 KiB/CTA；8槽共64 KiB。

这里一拍指一个 Memory/Compute 对，**Memory q 发出 Q[q+d]**，在本拍末等待下一消费者 Q[q+1]。每个较新的稳态 VMEM 组包含2 store＋1 DMA，因此等待值为 `3*(d-1)`。旧 BN64 选择器的 vmcnt3/6/9 实际对应 d=2/3/4，已替换为距离1/2/3，不能仅重命名旧三组结果。

| BN64 预取距离 | 本拍提交 | 实际稳态 ISA | 时间 (μs) |
|---|---|---|---:|
| 1拍 | Q[q+1] | `vmcnt(0)` | 1320.784 |
| 2拍 | Q[q+2] | `vmcnt(3)` | **641.919** |
| 3拍 | Q[q+3] | `vmcnt(6)` | 670.532 |

上述 vmcnt 已逐段检查实际 ISA，非仅按公式推算；启动/尾部按真实请求数收紧。三项均通过数值、graph、整段 Compute 无地址运算、Memory 无向量、七条交织、cached/no-AGPR 和零scratch检查。每项122 VGPR、70660 B LDS；SGPR分别87/86/86，无 SGPR/VGPR spill。

同进程默认形状 tokens16384/topk8/E384/N6144/K256、同输入地址、包含 counter reset、warmup2/iters10：对照 BN128 四阶段 **581.773 μs**，BN256 八阶段 **562.968 μs**；五项均 PASS，0/805306368 不匹配。BN64 本轮2拍最快，但仍慢于两个对照；单轮快测，不作统计显著性结论，原默认五项保持不变。

性能候选为 `4stage_bn64_pf1`、`4stage_bn64_pf2`、`4stage_bn64_pf3`。定向脚本使用 `--block-n 64 --prefetch-distance 1/2/3`，ISA 检查使用 `--require-bn64-four-stage --require-steady-vmcnt 0/3/6`；这里的斜线表示分别选择一个值。本次未跑完整回归、短N测试或抓 ATT。

## BN256 十六阶段实验（2026-09-11，GPU5）

- 每个 N256 分成 **8个 Memory/Compute 对**，每 Compute **8条 MFMA16×16×128**。
- 每个稳态 Memory **2条 cached store＋1条 B direct-LDS load**；B 包为 N64×K128=8 KiB，8槽仍共64 KiB。LDS scale/routing 读取另计，不将其冒充全局 B load。
- 按微拍计，`q=8*n+s`，B 的 N64 record 为 `2*(s//4)+s%2`，K128 block 为 `(s//2)%2`。分别预取 Q[q+3] / Q[q+4]，在本拍末确保下一拍消费者就绪；`wait_asyncmark` 在稳态留下2/3个较新 VMEM 组，每组2 store＋1 DMA，实际 ISA 分别为 **`vmcnt(6)` / `vmcnt(9)`**。启动和尾部按实际请求数收紧，不能单独放宽等待而不增加预取。
- 旧 C2/C3 的每行在前四个 Compute 打包，当前 C0/C1 在后四个 Compute 打包；上一 N 的输出均分到8个 Memory，最终 N 单独排空。所有 lane 地址仍在任务准备区物化，Compute 不计算地址。
- 两种模式均通过 tokens513/N6144/K256/E8/topk8 的数值及 graph 重放，以及实际 ISA 的16-stage结构、指定稳态 vmcnt、Compute无地址、Memory无向量、七条交织、cached/no-AGPR、零scratch检查。均为 **219 VGPR、106 SGPR（spill5但不落scratch）、70660 B LDS**；每份有32个展开 Compute，196个稳态 MFMA间隔严格7条向量指令。

默认形状 tokens16384/topk8/E384/N6144/K256，同进程共享输入地址，含 counter reset，warmup2/iters10，一轮快测：

| 候选 | 时间 (μs) | 相对八阶段时延变化 |
|---|---:|---:|
| 八阶段 BN256 | **573.053** | 基线 |
| 十六阶段，vmcnt6 | 577.605 | +0.79% |
| 十六阶段，vmcnt9 | 580.705 | +1.34% |

三项均 PASS，0/805306368 不匹配；vmcnt9 比 vmcnt6 慢约0.54%。单轮没有显示16-stage收益，不据此判断小差异有统计显著性，**保留八阶段默认**。

[test_blockscaled.py](test_blockscaled.py) 可选择 `--candidate 8stage_bn256 16stage_bn256_vmcnt6 16stage_bn256_vmcnt9`；两个实验候选不加入原默认五项。定向脚本使用 `--bn256-stages 16 --steady-vmcnt 6` 或 `9`；新增 `--require-sixteen-stage --require-steady-vmcnt 6` 或 `9` 用于 ISA 审计。已添加短 N 测试用例，本次未跑完整回归或重新抓 ATT；已有 ATT 是16-stage新增之前的四/八阶段快照。

## 流水线（3拍基线）

主蓝本是 [gemm2_8x1.py](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py) 的 `k128n`，不是原 down 的整 K/整 N tile 流水。

### BN128/K256：延迟写出、均分两个 Memory

一个 wave 的输出为 M32×N128，K256 共32条 MFMA16×16×128，按 N64 两半分成 **16+16条**。每个 B packet 为 N64×K256=16 KiB；四槽共64 KiB。

定义 `q=2*n+step`，`Q[q]=B[n, half=step, full K256]`。不同 K128 的 A/B scale 独立读取、独立反量化累加，不把整 K 合成一个 scale。

| step | Memory | Compute |
|---|---|---|
| 0 | 写 **上一 N 的低 N64**；读当前低 N64×K256 和两组 K scale；DMA Q[q+3] | 16 MFMA 计算当前低 N64；交织打包上一 N 的高 N64 |
| 1 | 写 **上一 N 的高 N64**；读当前高 N64×K256 和两组 K scale；DMA Q[q+3] | 16 MFMA 计算当前高 N64；交织打包当前低 N64 |

每个稳态 Memory 都是 **4条 cached `buffer_store_dwordx4`/线程（无 `nt`）**，8个wave共写32 KiB，即上一 N128 输出的一半。当前高 N64 的 FP32 结果和当前低 N64 的 BF16 结果跨 N 保留；下一 N 的 Compute0 打包高半，恰好赶上 Memory1。首 N 不写旧输出，最终 N 单独排空，**没有 C LDS 中转**。

启动只提交有消费者的 Q0/Q1/Q2。稳态预取 Q[q+3]，等待下一 Q 的 DMA 并退休本拍 LDS read 后过错相 barrier。`step1+3` 可落到两 N 之后，因此动态主循环后剥离最后两个 N，以静态裁掉无消费者的请求；N tile=1/2/3/4/5 均有测试。

### BN256/K256：四个 Compute

定义 `q=4*n+step`，`Q[q]=B[n, half=step//2, kb=step%2]`。每个 Compute 同样为16条 MFMA：

| step | B 消费 | Memory | Compute / C 退休 |
|---|---|---|---|
| 0 | 前 N128 / K0 | 写旧 C 的 row0/前半；读 B 和 scale；DMA Q[q+3] | MFMA + block dequant；pack 旧 C2 |
| 1 | 前 N128 / K1 | 写旧 C 的 row1/前半；读 B 和 scale；DMA Q[q+3] | MFMA + block dequant；pack 旧 C3 |
| 2 | 后 N128 / K0 | 写旧 C 的 row0/后半；读 B 和 scale；DMA Q[q+3] | MFMA + block dequant；pack 当前 C0 |
| 3 | 后 N128 / K1 | 写旧 C 的 row1/后半；读 B 和 scale；DMA Q[q+3] | MFMA + block dequant；pack 当前 C1，C2/C3留下一 N |

### 共同约束

- **8 waves 分两组 4 waves 错相**：group1 在首 Memory 前多一个 barrier，group0 在尾部补偿。所有退出条件位于错相区外。
- **B 四槽 direct-LDS ring**：BN128/K256 的槽为64×256 bytes，BN256/K256 为128×128 bytes，均为16 KiB。Memory q 发 Q[q+3]，预取距离3；仅在有真实消费者时发请求。
- B 用 `raw_ptr_buffer_load_async_lds`，生成 `buffer_load_dwordx4 ... lds`；没有 global→VGPR→LDS 中转。
- `wait_asyncmark` 等下一 Q 就绪；`lgkmcnt(0)` 在 Memory 末等本批 LDS read 完成后才过错相 barrier，保护跨组读和槽回收。任务退出时 drain async/VMEM/LDS，再复用 LDS。
- 地址在准备区或前一个 Compute 尾物化，跨动态 N loop 显式传递；Memory 只含 store / LDS read / DMA / SALU / wait。
- MFMA 为 **16×16×128**，物理 operands 为 B,A。不同 K128 部分先 FP32 点积，再用各自 A/B scale 累加；不能把 block scales 合成一个整 K scale。
- K256 的16-MFMA Compute 使用两条中间 MFMA 的延迟队列，按 **1 MFMA / 7条向量指令**交织反量化、独立已完成结果的 BF16 转换和 permlane；若有寄存器搬运也计入7条，不排除 `v_mov` 来凑计数。首拍没有可打包旧结果，尾部保留依赖排空及地址准备，不宣称所有周期都能满吞吐。
- A scale、B scale 和 routing weight **均缓存在 LDS**，每个 Memory 只取当前计算所需的值，避免这些值跨全部 N 常驻寄存器。A FP8 数据仍驻留寄存器。
- 使用原生 LLVM f32 算术/BF16转换保留 hazard 跟踪，`llvm.passthrough` 禁止 packed-FP32 combine。输出为 **`cvt_pk_bf16_f32` + `permlane16_swap` + cached `buffer_store_dwordx4`**；没有 C LDS scratch、CShuffle 或 LDS BF16 转换。

## 3拍基线验证（2026-09-11，MI350X / gfx950，GPU5）

- **26 项定向 pytest 通过**：五个支持的 K/BN 组合，两种 BN 的 N tile=1/2/3/4/5 和 split2、非均匀 block scales、padding guard、graph输出/counter投毒、真实 AITER sorting、257逻辑 task 的 persistent 复用、空队列、两个 BN 的多 seed cached/VGPR 回归以及旧 DMA helper。
- BN128/N6144 的实际 ISA：**10个展开 Memory、10个 Compute**（主循环仅含0/1两拍，另有启动和两 N 尾部代码）；每 Compute16条 MFMA，首 N 后每 Memory4条 cached store；所有 Memory无 `v_*`，135个非启动 MFMA间隔均为7条向量指令。
- BN128/N6144：LDS **70,020 B**，VGPR **174**（accum offset176），SGPR **95**；private segment、VGPR spill、SGPR spill **均为0**，无 scratch 指令、AGPR 操作数或 AGPR 搬运。
- BN256/N6144：16个展开 Memory/Compute 对，Memory零向量；64条输出 store 全部无 NT，210个稳态 MFMA间隔均为7条向量指令。VGPR **232**（accum offset232），SGPR106，SGPR spill5但不落scratch；private segment/VGPRspill均0，没有 AGPR 操作数或搬运。没有修改旧 BN64 的流水线来改善对比。
- 原生与 ATT 调试编译分别通过上述严格 ISA 审计。BN128 的 K384/512/640 另行通过 graph 数值及 cached/no-AGPR ISA 检查；较大 K 路径不套用 K256 的四阶段/16-MFMA 审计。
- 审计选项：`--require-four-stage` 检查两拍及4+4均分延迟写出，`--require-seven-valu` 检查全部向量指令而非只数数学指令，`--require-cached`、`--require-no-agpr`、`--require-no-scratch` 检查实际ISA/metadata。必须与 `--isa` 一起使用；`--require-nt` 仅保留用于检查历史 ISA，不是启用 NT 的运行开关。
- 最新四/八阶段 ATT 各8个 wave，默认大形状每份805,306,368个输出均无不匹配，轨迹源码与当前实现的 SHA256 一致。见 [最新产物和验证日志](ck_test/cached_vgpr_4stage_8stage_20260911/README.md)。

测试脚本可直接运行；`--graph --bench` 同时做数值/重放/快测，`--routing-mode aiter` 使用真实 sorting，`--isa` 审计实际生成的汇编。pytest 需将本目录加入 PYTHONPATH（沿用目录内 sibling import 约定）。原 kernel 的两个 Pylance `reportMissingImports` 警告在新文件中同样存在，但实际 editable 包导入、JIT、GPU 测试均通过；没有为此修改编辑器设置或依赖环境。

## 比较入口

直接运行 [test_blockscaled.py](test_blockscaled.py)，默认比较 PyHIP、旧 FlyDSL BN32/BN64、**4-stage BN128 / 8-stage BN256**。为兼容既有命令保留 `--candidate 8stage_bn128` 选择键，显示名称按实际阶段数标注。同一份量化 A/B、scales 和 routing 地址，全部包含 counter reset，统一 warmup2/iters10。表格同时显示相对 PyHIP 与旧 FlyDSL BN64 的加速比；不支持的形状显示 SKIP，实际编译/执行失败会报错并返回非零。

`--candidate` 可以选择一个或多个候选；`--profile` 要求只选一个候选，发射23次供 ATT 选择第21次，明确不显示性能时间。本次禁用 NT/AGPR 后只复核数值与采集轨迹，未重跑五候选计时。

### 历史性能：禁用 NT/AGPR 之前

以下是此前 NT/AGPR 版本的数据，不是当前 cached/VGPR-only 版本。默认形状 tokens16384/E384/topk8/N6144/K256，当时五项均 PASS，805,306,368 个输出逐项检查无不匹配：

| 候选 | 时间 (μs) | 有效 TF/s | 相对旧 FlyDSL BN64 |
|---|---:|---:|---:|
| PyHIP | 564.728 | 730.116 | 1.027x |
| FlyDSL BN32 | 677.364 | 608.708 | 0.856x |
| FlyDSL BN64 | 579.973 | 710.924 | 1.000x |
| 4-stage BN128 | 627.887 | 656.674 | 0.924x |
| 8-stage BN256 | 706.969 | 583.217 | 0.820x |

此前从 BN128 八阶段改为四阶段时，同场加载改前源码快照，正序测试 **658.704→625.110 μs**，反序测试 **653.860→636.171 μs**，时延分别减少 **5.1% / 2.7%**。这是两轮 quick benchmark，不是统计显著性结论；当时四阶段版仍未超过旧 BN64。

历史四阶段的 [ISA、ui_output、同场比较日志和源码快照](ck_test/bn128_4stage_20260911/README.md) 和 [五候选初版产物](ck_test/blockscaled_compare_20260911/README.md) 保持原样，不把旧 trace 重标成新实现。初版 BN256 的56B scratch、上述 NT/AGPR 计时均不作为当前状态；当前源码和两份轨迹以 [cached/VGPR-only 产物索引](ck_test/cached_vgpr_4stage_8stage_20260911/README.md) 为准。