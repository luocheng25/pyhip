# fused MOE g1u0 gelu

## fused MOE 基本流程

 - hidden_states: [num_tokens, model_dims]
 - 逐点乘以 smooth_scale + per-token 量化到 INT8 [[1]](#共享smooth_scales)
 - moe_sorting(block_m) : 每个token的topk个需要infer的复本，根据专家进行分组排序，每组内token的个数padding到block_m整数倍 [[2]](#最大化block_m)
 - GEMM1 + fused_gelu: 输出数据tensor bf16：[num_tokens, topk, inter_dims] [[3]](#HIPKitten's 8wave pipeline) [[4]](#指令优化gelu)
 - 逐点乘以 per-expert smooth_scale + per-token 量化到 INT8
 - GEMM2: [num_tokens, topk, model_dims] [[6]](#中间结果动态量化) [[7]](#XCD Swizzle)
 - ReduceSum : [num_tokens, model_dims] [[5]](#避免ATOM访存带宽限制)


## 共享smooth_scales
根据 [SmoothQuant论文](https://arxiv.org/abs/2211.10438)，outlier主要出现在激活input的某些channel，权重相对则要均匀很多，作为整个MOE的原始输入，应该可以共享同一份smooth scale, 相较于每个专家独享一份smooth scale的方案，这可以把第一个量化kernel的数据访问量至少降低TOPK倍。

## 最大化block_m

GEMM问题中，一个workgroup (threadblock)处理的输出矩阵越大，越接近正方形，越能够降低访存计算比 `(M+N)*K/(MNK)`, 在LDS和寄存器资源允许的情况下，配合jit的手工寄存器分配，该值可以增加到256。（目前aiter代码中最大128）

## HIPKitten's 8wave pipeline

[HipKittens](https://arxiv.org/abs/2511.08083) 针对AMDGPU引入了8-wave 排流水的创新性pipeline：
 - 按照4wave为单位，分为两组，两组交替执行计算和加载，遮盖访存延迟
 - 可以大大简化流水线的排布，并且获得几乎不输于4wave的性能
 - 相对传统的`4wave`排流水，可以避免手工决定使用AccVGPR还是普通VGPR的问题，全部分配为普通VGPR

## 指令优化gelu

[Gelu](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) 激活函数的计算使用到了[tanh](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)函数，HIP的C++ STL生成的汇编代码比较复杂并且可能出现thread-diverge（根据每个thread输入值的取值范围使用不同方式近似计算，以避免exp越界），可以更通过简单的方式保证exp的输入永远是负数来避免越界：
```python
    sign = np.sign(v)
    exp = np.exp(-2*sign*v)
    tanh = (sign - sign*exp)/(1 + exp)
```

## 避免ATOM访存带宽限制

目前Aiter中某些case下的moe gemm2的输出是使用 `global_atomic_pk_add_bf16` 指令直接累加到外存来避免写出巨量的中间结果和额外的ReduceSum kernel开销，但是实测表明这种 atomic 访存指令在 gfx942/gfx950 上的带宽只有普通写出访存指令的1/4，因此性能还不如`直接写出巨量数据，再使用ReduceSum读回做sum再存出最终结果`。

## 中间结果动态量化

gemm2写出巨量中间结果发生在gemm kernel的尾部，没有MFMA指令可以与其并行遮盖，因此拖累了性能，以1x32为单位的量化该中间结果到INT8再存出，可以显著降低写出消耗。

另外使用`sc1 nt`修饰符bypass cache，直接streaming到外存可以结果数据对L2-cache的污染，进一步提升gemm核心循环的性能


## XCD Swizzle

gfx950的256个CU分布在8个XCD上，每个XCD具有独立的L2-cache, 通过把访问相同专家权重的 gemm-block 计算任务分配到相同的 XCD上，当这些任务在gemm核心循环中，以一致的步调访问接近相同的权重和输入数据的某个K维度slice的时候，就可以从L2-cache受益，减少冗余的外存加载操作。

## H3 B=32768优化记录

固定测试配置：gfx942（MI308X），FP8 PTPC，`B=32768`，`H=6144`，本地
`I=384`（TP=8），`E=128`，`TOPK=4`，`BM=64`，`BN=256`。gateup和down都走
`prefill_1x4`。以下TFLOPS按未padding的有效token计算；sorting后实际执行行数多6.25%。

### 基线（2026-08-11）

| kernel | 有效FLOP | 稳态中位时间 | 有效TFLOPS | VGPR+AGPR | LDS/WG | wave/SIMD |
|---|---:|---:|---:|---:|---:|---:|
| gateup | 1.237 TFLOP | 2.734 ms | 452.5 | 56+128 | 16 KB | 2 |
| down | 0.618 TFLOP | 1.851 ms | 334.1 | 60+132 | 24 KB | 2 |

两者合计为1.855 TFLOP / 4.585 ms = 404.7 TFLOPS。完整`run()`还包含约
229 us输入量化、241 us中间激活量化、565 us sorted-sum和少量索引处理。

ATT稳态trace：

- gateup：79.3% stall；MFMA 55.3%，VMEM load+wait 26.7%，barrier 6.3%，
    LDS相关5.4%。
- down：77.9% stall；MFMA 38.3%，packed/其他VALU依赖23.8%，LDS相关15.9%，
    VMEM load+wait 13.2%，barrier 6.2%，VMEM store 2.6%。
- L2 hit：gateup 51.7%，down 60.2%。两者occupancy均被限制到2 wave/SIMD；
    down同时受24 KB LDS限制，最多2 WG/CU。

### MFMA stall解释

`v_mfma_f32_16x16x32_fp8_fp8`的issue间隔约4 cycles，同一accumulator的RAW延迟约
16 cycles，所以连续依赖MFMA在准备发射时理论上还需要等待约`16-4=12` cycles。
ATT的`stall_cycles`记录的是“该条MFMA准备发射但被阻塞”的周期，不是相邻两条MFMA的
总间隔。当前静态MFMA的stall/exec中位数为gateup 11.90、down 11.83 cycles，说明典型
accumulator链已经接近12-cycle RAW下限；但按动态执行加权的均值为16.73和13.91 cycles，
其中还包含操作数未就绪、pipeline争用和wave调度，不能解释成“MFMA总延迟接近16所以已最优”。

gfx942每条MFMA提供约12-cycle shadow，可以隐藏3条独立的普通4-cycle scalar VALU；依据
[`mfma-valu-coissue.md`](../../flydsl/attn_4wave/tools/mfma-valu-coissue.md)，
`v_pk_add_f32`/`v_pk_mul_f32`虽然吞吐也约4 cycles，但与MFMA的intra/inter full-coissue
容量都是0，不能利用这3条scalar VALU容量。

### down代码流程与首轮优化计划

down当前流程：

1. 按`sorted_ids`把FP8输入行搬到LDS A并同步；每个WG负责`64x64`输出块。
2. 从LDS加载完整A fragment；每次预取一个`N=64`权重块及PTPC scale到寄存器。
3. `gemm_compute`沿`K=384`执行FP8 MFMA，随后将scale和routing weight乘到FP32累加器。
4. FP32转BF16，写入双缓冲LDS C；同步后由上一缓冲读出并128-bit写全局内存。
5. kernel外通过`invert_sorted_ids + sorted_sum`把top-k中间结果归并为最终输出。

#### `gemm_compute`和4-wave划分

down的逻辑矩阵为`C[token, channel] = A[token, K] x W[channel, K]`，每个WG处理
`BLOCK_M x BLOCK_N = 64 token x 64 channel`。实现中为匹配MFMA寄存器布局，将MFMA的
M维映射为channel、N维映射为token：

```text
create_thr_mma(fp8, wave_mnk=(4, 1, 1))
thr_layout_mnk shape  = (4, 1, 1)
thr_layout_mnk stride = (1, 4, 0)
```

因此4个wave沿channel维划分，而不是沿token维划分：

| wave | thread | channel范围 | token范围 |
|---|---|---|---|
| 0 | 0..63 | 0..15 | 0..63 |
| 1 | 64..127 | 16..31 | 0..63 |
| 2 | 128..191 | 32..47 | 0..63 |
| 3 | 192..255 | 48..63 | 0..63 |

底层原子是`MFMA(16x16x32, FP8)`。一个wave覆盖`16 channel x 64 token`，即token方向有
4个16-wide accumulator tile。`BLOCK_K=32`，H3本地`K=384`，所以：

```text
nBK = K / BLOCK_K = 384 / 32 = 12
MFMA / wave / output-block = 12 K-step x 4 token-tile = 48
MFMA / WG / output-block   = 48 x 4 wave = 192
```

`gemm_compute(fragW, fragPCS, fragC)`的实际顺序：

1. 清零当前FP32 `fragC`。
2. 对12个K-step调用`fx.gemm`，每步每wave更新4组独立token accumulator。
3. K归约结束后，用PTPC weight scale逐元素乘`fragC`。
4. `postprocess_store2lds`再乘每token routing weight，FP32转BF16并写入LDS C。

外层`nBN=N/BLOCK_N=6144/64=96`。实现用两组`frag_weights/fragC/LDS-C`做N方向双缓冲：
当前块执行48条MFMA时预取后续channel块权重；上一块从LDS读出并写全局。每次外层循环推进
两个64-channel块。这里权重预取与MFMA存在重叠，但scale/routing-weight的packed ALU集中在
MFMA链之后，且`v_pk_mul_f32`不能利用MFMA的12-cycle scalar VALU shadow，正是首轮优化目标。

#### 权重和128-bit访存指令数

权重不经过LDS。`arg_p_weight`先包装成buffer descriptor，`load_tiled_mma_fragA`使用
`BufferCopy128b`直接执行`global/L2 -> VGPR`。对一个`64 token x 64 channel`输出块：

| 数据路径 | 每wave的wave-level指令 | 每WG的128-bit lane access | 备注 |
|---|---:|---:|---|
| weight global -> reg | 6 `buffer_load_dwordx4` | 1536 | 24 KB，4 wave各读自己的16 channel |
| A global -> reg | 6 `buffer_load_dwordx4` | 1536 | 每WG只在进入kernel时做一次 |
| A reg -> LDS | 6 `ds_write2_b64` | 1536 | 每WG只做一次 |
| A LDS -> reg | 24 `ds_read_b128` | 6144 | 每wave都需要完整`64x384` A，读量复制4倍 |
| C reg -> LDS | 2 `ds_write2st64_b64` | 512 | 每个输出块 |
| C LDS -> reg | 2 `ds_read_b128` | 512 | 每个输出块 |
| C reg -> global | 2 `buffer_store_dwordx4` | 512 | 每个输出块 |

这里“每wave指令”是一条wave指令控制64 lanes；“lane access”按每个active lane一次16-byte
访问计数，不等于合并后的物理L2/HBM transaction数。例如weight每wave读取
`16x384=6144 B`，所以是`6144/(64x16)=6`条wave指令；4 wave总计
`24x64=1536`个128-bit lane access。完整96个channel块中，每wave执行576条weight load和
4608条MFMA，整个WG读取权重2.25 MiB；权重没有对应的LDS/global写指令。

理论遮盖关系：

- 每条weight load包含两个K-step的wave operand，支持`2 K-step x 4 token-tile = 8`条MFMA。
    每个输出块是6条weight load对48条MFMA。
- 48条MFMA按4-cycle issue间隔提供约192 cycles的同wave计算窗口。当前scheduler按
    `6 MFMA + 1 memory group`排布，最早发出的weight load可获得接近整个窗口，最后几条load
    只有约24 cycles到下一次消费。
- 因此L2 hit通常可大部分遮盖，但300--500 cycle量级的HBM miss无法保证全部遮盖；2 wave/SIMD
    只能提供额外独立工作，不能消除同wave最后一批load到consumer的短距离。ATT中VMEM
    load+wait仍占down总stall的13.2%，说明weight/global读取是**大部分但未完全遮盖**。
- C的global store可与后续weight load/MFMA异步推进，ATT中VMEM-store只占2.6%，基本被遮盖。
- C的LDS write之后有barrier，下一阶段立即LDS read，存在硬同步依赖，不能由前一段MFMA
    向外遮盖；LDS相关stall占15.9%，是比global store更明确的未遮盖尾部。
- A的global->LDS->reg发生在96个输出块之前，没有MFMA并行，但只执行一次，成本被整个N循环
    摊薄；它不是当前首要优化目标。

首轮只改down，优先级如下，逐项小步验证：

1. **拆除packed FP32 ALU。** ATT中`v_pk_mul_f32`是最大的非MFMA热点之一，且微基准证明
     它不能和MFMA full-coissue。先把scale/routing-weight乘法改成明确的scalar FP32
     `v_mul_f32`/`v_fma_f32`，再检查ISA确认不被LLVM重新合并为`v_pk_*`。目标是把每条MFMA的
     12-cycle shadow用于最多3条scalar VALU。
2. **重排scalar ALU到MFMA窗口。** 仅拆包不够；需用调度barrier把独立scale/weight乘法
     分散到MFMA链之间，避免全部落在GEMM结束后的串行尾部。关注VGPR live range，不能让
     combined VGPR超过当前约192而降低occupancy。
3. **最后处理LDS epilogue。** down的`ds_write/lgkmcnt`平均暴露约47/46 cycles，当前
     `write LDS -> barrier -> read LDS -> global store`有硬依赖。先完成packed ALU实验，再考虑
     拉大LDS写读距离、将更多global store和下一块MFMA/权重预取交叠。

每次后续实验只在本节追加一行：`日期 | 改动 | 时间 | TFLOPS | VGPR+AGPR | LDS | ATT变化 | 结论`。

### 实验结果（2026-08-12）

以下时间均来自H3 B=32768的raw kernel trace，剔除首调用后取中位数；有效FLOP固定为
0.618475 TFLOP。当前gateup约451.8 TFLOPS，因此down若要追平，时间需从1.847 ms降到约
1.369 ms，即至少再降低25.9%。

| 改动 | down时间 | TFLOPS | VGPR+AGPR | 结果 |
|---|---:|---:|---:|---|
| N64原基线 | 1.8468 ms | 334.9 | 60+132 | 当前最快正确版本 |
| packed拆成scalar `llvm.fma.f32` | 1.8974 ms | 326.0 | 56+128 | ISA中`v_pk_mul_f32`清零，但scalar仍集中在MFMA之后，退化2.7%，已回退 |
| weight/scale三级预取 | 1.8784 ms | 329.3 | 108+132 | 消费距离从约48条MFMA增到96条，但VGPR和控制流成本更高，退化1.7%，已回退 |
| 跨块MFMA/后处理流水（仍packed） | 1.9821 ms | 312.0 | 96+128 | 槽位正确，但更长live range导致资源上升，退化6.8%，已回退 |
| N64每lane直接64-bit写出 | 3.2582 ms | 189.8 | 80+128 | 去掉C LDS，但每块store数量翻倍且串行，退化43.3%，已回退 |
| N128逻辑地址128-bit直写 | 2.4308 ms | 254.4 | 80+128 | 两个N64 pass在同一lane拼成连续8 BF16；无跨lane permute，`sorted_sum=0.5545 ms` |
| N128头部集中预取，无`setprio` | 2.0213 ms | 306.0 | 116+132 | 双份weight/scale寄存器；循环头集中发出12条weight和2条scale load |
| N128头部集中预取，slot反相`setprio` | 2.0142 ms | 307.1 | 116+132 | 当前N128最快；`sorted_sum=0.5726 ms`，完整H3 `diff=0.00105974` |

scalar实验补充结论：直接照搬PA的inline-asm `_fma_f32` 在down的SCF循环中不可靠。`=v`
输出允许寄存器覆盖尚未读取的输入；改为early-clobber后仍需显式处理VMEM依赖，尾块仍可能损坏。
改用LLVM scalar `llvm.fma.f32` intrinsic后，N=128/N=384有限值测试和完整H3测试均通过
（完整H3 `diff=0.00106`），且最终ISA生成普通`v_fma_f32`而不是packed指令，但因没有进入
MFMA shadow而出现上述性能退化。

#### N128直接128-bit写出结论

最初尝试保持标准连续N64权重布局，再从两个MFMA-M repeat拼接输出；该布局下同一lane的两组
4xBF16不连续，需要gfx942不支持的`permlane16_swap`或昂贵的`ds_bpermute`。最终方案不再对
结果做跨lane重排，而是重排weight的逻辑N遍历：

```text
pass 0: [0..3]  [8..11]  [16..19]  [24..27]
pass 1: [4..7] [12..15]  [20..23]  [28..31]
```

对wave `w`、lane-group `g`和N128 block `b`，两个pass分别产生：

```text
pass 0: 128*b + 32*w + 8*g + [0:4]
pass 1: 128*b + 32*w + 8*g + [4:8]
```

同一lane因此天然得到连续8个BF16，只需lane内拼接成一条`buffer_store_dwordx4`。每个物理
N128 block每wave执行4条128-bit store；不经过C LDS，不使用`bpermute`、`permlane`或其他
跨lane permute。写出地址直接采用真实逻辑顺序`wave*32 + lane_group*8`，所以`sorted_sum`
保持原128-bit连续读取和写出，不再需要channel解置换。

代价是weight的wave级N方向读取从连续16行改成“连续4行、跳过4行，下一pass补齐”。每个lane
自身仍执行连续16-byte的`buffer_load_dwordx4`，总weight字节数不变，但空间局部性下降。

#### N128头部集中预取与相位反转

原N128逻辑直写将next-pass0和next-pass1预取分散在两个GEMM之间，ATT中VMEM-load占40.0%、
VMEM-store占19.3%，down为2.4308 ms。新结构为每套current/next各保留两份weight和scale
fragment，主循环严格组织为：

```text
read/write stage:
    集中预取下一N128 block的两个pass（12条weight + 2条scale load）
compute stage:
    pass0 GEMM + scale/routing-weight后处理
    pass1 GEMM + scale/routing-weight后处理
read/write stage:
    4条128-bit global store
```

参考`tests/flydsl/pa_4wave/pa_prefill_4wave.py`读取`HW_REG_HW_ID[3:0]`，对同一SIMD的两个驻留
wave设置阶段相关优先级：read/write阶段slot0/slot1为`1/0`，compute阶段为`3/2`。最终ISA
确认12条weight load和2条scale load全部位于首条MFMA之前，随后是两组共96条MFMA及后处理，
最后才执行4条store。寄存器从逻辑直写版的80+128升至116+132，但仍保持2 wave/SIMD。

ATT对比（相同第5次稳态dispatch）：

| 指标 | N128逻辑直写 | 头部预取+`setprio` | 变化 |
|---|---:|---:|---:|
| total stall | 103.88M | 68.54M | -34.0% |
| stall rate | 85.6% | 76.3% | -9.3 pp |
| VMEM-load stall | 41.55M | 17.72M | -57.4% |
| VMEM-store stall | 20.03M | 12.66M | -36.8% |
| VMEM-wait stall | 3.34M | 2.27M | -32.0% |
| MFMA stall | 20.68M | 22.44M | +8.5% |

因此计算阶段确实部分掩盖了读写，down提升到2.0142 ms / 307.1 TFLOPS。关闭`setprio`但保留
完全相同的头部集中预取结构时为2.0213 ms / 306.0 TFLOPS，说明主要收益来自双份寄存器和
集中预取；slot反相本身额外约0.35%，接近运行波动，但ATT确认其最终阶段结构符合预期。

#### N256外循环 + 每wave 64x64x128核心

后续实验将WG输出块改为`M64xN256`，4个wave仍沿N分布，每个wave负责连续`M64xN64`；K方向
按128切成静态展开的内循环。H3的K=384因此每次N外循环包含3个`64x64x128`核心，共192条
FP8 MFMA。weight逻辑布局将TiledMMA的4个M-repeat映射为每wave连续64 channel，输出每个
token/lane使用两条128-bit store。

pipeline分为：

```text
stage0: 读weight/A、预取下一K（最后K预取下一N的K0）、写出
stage1: 当前64x64x128核心的MFMA计算
```

两个版本均通过N512/N1024有限值测试和完整H3测试（`diff=0.00105974`）：

| 版本 | down | TFLOPS | VGPR+AGPR | 结果 |
|---|---:|---:|---:|---|
| v1：当前K的weight/A读完立即计算 | 3.0773 ms | 201.0 | 56+128 | ATT暴露当前K依赖；该次stats受系统负载干扰，绝对时间波动较大 |
| v4：weight双槽ping-pong，A单槽，集中store | 2.3290 ms | 265.6 | 104+128 | 2 wave/SIMD；8条store集中在K循环之后 |
| v5：上一N块store按K-stage分成3/3/2 | 2.0559 ms | 300.8 | 120+128 | 当前N256/K128最佳；`sorted_sum=0.5645 ms` |

v1到v4的ATT变化证明K128 stage1能够掩盖stage0读取：

| Stall | v1 | v4 | 变化 |
|---|---:|---:|---:|
| total | 81.88M | 73.87M | -9.8% |
| VMEM-wait | 16.24M | 2.59M | -84.1% |
| LDS-wait | 15.01M | 1.67M | -88.9% |
| VMEM-load | 17.58M | 17.43M | -0.8% |
| VMEM-store | 12.97M | 16.81M | +29.6% |
| MFMA | 22.91M | 22.06M | -3.7% |

即计算确实掩盖了大部分weight/A等待，但没有减少实际load流量；每wave N64输出需要每token两条
128-bit store，store stall升至22.8%。尝试让A和weight都使用双槽时VGPR升至312，occupancy
降为1 wave/SIMD，因此淘汰。

v5将上一N块已完成的8条128-bit store作为BF16 loop-carried状态，在当前块的三个K128 stage0
中按3/3/2发出，最后再drain末块。每wave每个内层循环的128-bit指令预算为：

| K128 stage | weight `buffer_load_dwordx4` | A `ds_read_b128` | scale load | output `buffer_store_dwordx4` | MFMA |
|---|---:|---:|---:|---:|---:|
| K0 | 8 | 8 | 0 | 3 | 64 |
| K1 | 8 | 8 | 0 | 3 | 64 |
| K2 | 8（下一N的K0） | 8 | 4条`global_load_dwordx4` | 2 | 64 |

按WG计数时乘以4个wave：每K128为32条weight wave-instruction和32条A LDS wave-instruction；
每N256块为32条store wave-instruction。若按lane access计数，每wave每K128的weight/A各为
`8*64=512`次16-byte访问，每wave每N块store同样为512次16-byte访问。

相对集中store的v4，v5 down从2.3290降至2.0559 ms（-11.7%），TFLOPS从265.6升至300.8；
`down+sorted_sum=2.6204 ms`。VGPR从236升至250，仍保持2 wave/SIMD。ATT中store平均stall从
216.7降至191.1 cycles/exec（-11.8%），store stall总量从16.81M降至15.75M（-6.3%）；总stall
从73.87M降至72.81M（-1.4%）。因此分散store降低了单条store的暴露延迟，但ATT采样中的总周期
改善有限；绝对kernel时间的收益还包含更好的跨wave阶段配合。当前v5保留为未暂存实验，实验
开始前的全量工作区检查点在Git index中。

#### Scale 128-bit去重与LDS广播

原N256路径的scale已经生成4条`global_load_dwordx4`，但C-fragment地址只依赖`tid & 0xf0`，
同一16-lane group的16个lane读取完全相同的16个scale。每个N256块逻辑scale为1KB，原路径
4个wave各执行4条128-bit load，按lane流量为16KB，即16倍重复；K方向没有重复，每N块只加载
一次。

最终方案使用`fx.Array[fx.Float32, 256, 16]`额外分配1KB、显式16-byte对齐的独立LDS：每个wave
的前16个lane各用一条`buffer_load_dwordx4`加载4个连续scale到本wave的64-channel LDS片区，
然后4个wave分别用4条`ds_read_b128`按原C-fragment布局广播。各wave只读自己写入的片区，依靠
同wave`lgkmcnt`保序，不需要跨wave barrier。总LDS从24,576B增至25,600B，仍保持2 wave/SIMD。
BN128时上述容量和每wave参与lane数按tile缩半：scale LDS为512B、每wave前8个lane参与，
总LDS为25,088B。

指令/流量变化（每WG每N256块）：

| 路径 | Global scale load | Global lane流量 | LDS写 | LDS读 | Barrier |
|---|---:|---:|---:|---:|---:|
| 原direct scale | 16条wave-instruction | 16KB | 0 | 0 | 0 |
| 每wave私有LDS | 4条wave-instruction（每wave仅16 lanes active） | 1KB | 4条 | 16条 | 0 |

第一次仅wave0协作加载的版本虽然global只需1条wave-instruction，但需要全WG barrier；ATT中barrier
达到11.09M stall / 13.9%，因此放弃。每wave私有版本避免barrier，down为2.0288 ms。进一步将
下一N块scale的global load放在K2 stage0，把LDS提交推迟到K2的64条MFMA之后：最终ISA中稳态
scale load和`ds_write_b128`之间有64条MFMA，等待从`vmcnt(0)`放宽到`vmcnt(2)`。

| 版本 | down | TFLOPS | LDS | 结果 |
|---|---:|---:|---:|---|
| v5 direct scale | 2.0559 ms | 300.8 | 24,576B | 16倍global lane重复 |
| v7 每wave私有scale LDS，同步提交 | 2.0288 ms | 304.8 | 25,600B | 无barrier，但scale load后立即等待 |
| v8 异步scale load，K2后提交LDS | 2.0234 ms | 305.7 | 25,600B | 当前N256最佳；`sorted_sum=0.5646 ms` |

v7到v8的ATT中，scale helper stall从11.24M降至2.23M，VMEM-wait总量从11.87M降至3.06M，
总stall从76.44M降至72.61M。v8相对direct scale快1.58%，说明scale去重有效，但收益受新增LDS
广播和原有weight/store瓶颈限制。完整H3保持`diff=0.00105974`。

#### Tile N 128/192/256比较

physical down现支持`BLOCK_TILE_SIZE_N`为128/192/256，H3默认通过
`MOE_DOWN_PHYSICAL_TILE_N=256`选择BN256。BN128仍使用4个wave沿N：每wave负责32 channel，
每lane最终得到8个连续BF16，因此每token只需一条128-bit store；三个K128 stage的4条store
自动分成2/1/1。BN128与BN256总MFMA、weight字节数和输出字节数相同，仅改变N外循环粒度。

| Tile N | ISA VGPR | Stats VGPR+AGPR | LDS | 每N块MFMA | 每wave store/N块 |
|---|---:|---:|---:|---:|---:|
| 128 | 158（next-free 169） | 32+144 | 25,088B | 96 | 4 |
| 256 | 242 | 120+128 | 25,600B | 192 | 8 |

空闲card3按128 -> 192 -> 256 -> 192顺序完成最终sweep：

| Tile N | down | TFLOPS | sorted_sum | Stats VGPR+AGPR | LDS |
|---|---:|---:|---:|---:|---:|
| 128 | 2.2174 ms | 278.9 | 0.5679 ms | 32+144 | 25,088B |
| 192-A | 2.6780 ms | 231.0 | 0.5558 ms | 76+132 | 25,600B |
| **256** | **2.0369 ms** | **303.6** | **0.5530 ms** | 120+128 | 25,600B |
| 192-B | 2.6535 ms | 233.1 | 0.5634 ms | 76+132 | 25,600B |

BN128相对BN256慢8.9%。BN192在4-wave分布下每wave48 channel、3个MFMA-M repeat；每lane最终
有3组连续4 BF16，使用每token三条64-bit `buffer_store_dwordx2`，三个K128 stage各分4条。
N1536端到端正确，资源为ISA VGPR202、单N块144条MFMA和12条64-bit store，但两轮均比BN256
慢约30%。因此默认恢复BN256；BN128/192保留为环境开关和正确性覆盖。

#### Gateup/down pipeline、no-pk与scalar FMA

Gateup和physical down都以每wave `64x64x128` 为核心，每个核心均为64条
`v_mfma_f32_16x16x32_fp8_fp8`，但pipeline不同：

| | Gateup | Down BN256 |
|---|---|---|
| C流 | gate/up两条独立累加流 | 单条down累加流 |
| Activation | global gather -> register -> A LDS ping-pong | kernel头一次性global -> A LDS，K stage从LDS读 |
| Weight | gate/up各自global -> register双缓冲 | 单份weight global -> register双槽ping-pong |
| 调度 | 同一basic block内用`sched_group`交织VMEM/LDS/MFMA | `setprio`将stage0读写和stage1 MFMA分相，靠两驻留wave反相遮盖 |
| 输出 | CShuffle/LDS后写出，稳态store很少 | 上一N块8条128-bit store按2/2/4分散到三个K stage |

按最终ISA和ATT动态执行次数统一到“每wave每64条MFMA”后，128-bit指令预算为：

| | VMEM load | VMEM store | LDS read | LDS write |
|---|---:|---:|---:|---:|
| Gateup | 10.08 | 0.08 | 8.08 | 2.00 |
| Down BN256 | 8.54 | 2.78 | 9.39 | 0.35 |

Down稳态按当前源码stage精确计数为K0/K1/K2各8条weight load、8条A LDS read，store为2/2/4；
K2另有1条masked scale global load，N块尾有1条scale LDS write和4条scale LDS read。因而纯VMEM
平均为每wave每K128 `8 + 1/3`条load和`8/3`条store，共11条；4 wave合计44条，而不是将LDS
访问混入后得到的56条。LDS平均另有`8 + 4/3`条read和`1/3`条write，走独立管线，应单列。

在physical down launch上设置`target-features=-packed-fp32-ops`后，packed FP32指令清零；进一步将
routing scale与BF16 rounding bias合并为scalar `llvm.fma.f32`，最终ISA生成普通
`v_fmaak_f32`，明确没有`v_pk_fmac_f32`。BN256后处理从64条`v_pk_mul_f32`加32条
`v_pk_add_f32`变为64条scalar mul加64条scalar FMA；按每元素计算，原两轮packed共192个scalar
等价操作，现为128个，减少三分之一。空闲card3同口径结果：

| 版本 | Down | TFLOPS | Stats VGPR+AGPR | 结果 |
|---|---:|---:|---:|---|
| v8 packed | 2.0369 ms | 303.6 | 120+128 | 对照 |
| 仅禁pk | 2.0177 ms | 306.5 | 120+128 | +0.94% |
| 禁pk + scalar FMA融合 | **1.9966 ms** | **309.8** | **108+132** | 相对v8 +2.0%，保留 |
| 严格两相：stage1仅MFMA，stage0含读写与后处理 | **1.9928 ms** | **310.3** | **108+132** | 资源不变；相对FMA版基本持平，保留 |
| stage0保持slot `1/0`，stage1统一prio3 | **1.9719 ms** | **313.6** | **108+132** | 限制VMEM竞争、取消MFMA固定slot饥饿，保留 |
| 上一N块store按K stage改为`2/2/4` | **1.9504 ms** | **317.1** | **108+132** | 相对严格两相 +2.2%，当前最佳，保留 |

N256主循环现在是严格的两相状态机：prologue先进入stage0；每个K128迭代在stage0完成A读取、
下一份weight/scale预取和上一N块的分散store，随后切到stage1且只执行64条MFMA；MFMA结束后立即
切回stage0。因此最后K后的scale LDS提交、当前N块的weight/routing scale、BF16转换，以及末块
drain store都属于stage0。下一K迭代沿用当前stage0状态，不重复切换优先级。

FMA版ATT目录为`/tmp/moe_h3_n256_no_packed_fma_att/ui_output_agent_21912_dispatch_2596`。
总stall从72.61M降至70.91M；packed ALU的6.93M stall被0.70M scalar FP32 stall取代。三个K128
MFMA phase的64条MFMA stall分别为8.47M、8.31M、8.62M，首条仍有0.95M、0.99M、1.11M，说明
scalar后处理仍位于MFMA phase之外；当前主瓶颈回到VMEM load 20.80M、store 17.49M和MFMA
25.40M。尝试用`sched_group_barrier`指定`1 MFMA + 2 VALU`没有改变最终ISA顺序，已移除。

另试`MNK`遍历让weight源交替、activation源复用，计时仅从1.9966降至1.9901 ms（0.33%）；ATT中
MFMA平均stall几乎不变（13.64 -> 13.61 cycles/exec），总stall反而70.91M -> 72.87M，故按噪声
处理并恢复默认`NMK`。默认`NMK`已将K放在最外层，同一累加器复用距离约16条MFMA，单纯改变
M/N遍历无法进一步缩短MFMA依赖stall。

#### 严格两相ATT与遮盖机会

严格两相基线ATT为
`/tmp/moe_h3_n256_strict_two_stage_att/ui_output_agent_25881_dispatch_2596`，当前最佳`2/2/4`
ATT为`/tmp/moe_h3_n256_store_224_att/ui_output_agent_1692_dispatch_2596`。两者采样执行数一致；当前
最佳相对严格两相的stall变化为：

| Stall | 严格两相 | hybrid-prio3 + `2/2/4` | 变化 |
|---|---:|---:|---:|
| total | 70.00M | 68.25M | -2.5% |
| MFMA | 25.43M | 28.25M | +11.1% |
| VMEM load | 19.86M | 18.53M | -6.7% |
| VMEM store | 17.70M | 14.96M | -15.4% |
| VMEM wait | 3.62M（含统一wait分类） | 2.28M | 明显下降 |

Raw wave时间线显示，严格两相的双slot反相覆盖只有48.9%，两slot同时stage0占35.8%、同时stage1
占15.4%。这说明`setprio`不能构造严格的相位锁；更重要的是控制VMEM并发。原策略在stage0和
stage1都固定slot0比slot1高一级，slot1的首条weight load平均约328--370 cycles，slot0约162--173
cycles。最终策略在stage0保留slot `1/0`来抑制两个slot同时灌满VMEM队列，在stage1让两个slot都
使用prio3，避免固定压低slot1的MFMA。虽然反相覆盖降到约44%，但load/store stall下降更多，净性能
提升，因此不能把“反相比例最高”当作单一优化目标。

每个stage0的第一条weight load约245--259 cycles，后续7条通常仅27--47 cycles；它主要承接
MFMA->VMEM管线切换和队列背压，而不是8条load都暴露完整HBM延迟。K2原先集中9条load和2条store；
把8条输出store从`3/3/2`逐步后置后，`2/2/4`最佳。`1/2/5`退化到2.0001 ms，说明K2在4条store
附近已经达到承载边界。

已否证的遮盖实验：

- stage0或stage1单独反转固定slot：反相覆盖均降到约42%，MFMA平均stall从13.66升到约15.5
    cycles/exec，性能无收益，已回退。
- 两slot只按阶段统一prio0/prio2：控制指令更少，但ATT total stall增加2.6%，未保留。
- store-first再发weight/A load：相邻同卡A/B为2.0397 vs load-first 1.9853 ms，明显退化。
- 下一N scale load从K2提前到K1：隐藏距离从64增到128条MFMA，但live range改变寄存器分配，
    down中位数退化1.8%，已回退。
- 强制A LDS read先于weight load：最终ISA兑现但寄存器分配增加，外部GPU任务期间无法得到有效
    绝对性能，未保留。
- K2 `2R/1W` VMEM sched_group交织：最终ISA仍为全部load后全部store，且寄存器分配显著扰动，
    静态检查即淘汰。

#### BN256整wave连续写出

将BN256输出改为物理连续布局。对每条128-bit store，定义：

```text
physical_lane = (lane_id % 16) * 4 + lane_group
physical_chunk = ((block_n * 4 + wave_id) * 8 + store_index) * 64
                                 + physical_lane
byte_offset = physical_chunk * 16
```

因此lane `0/16/32/48`分别写相邻的第`0/1/2/3`个16B块，lane
`1/17/33/49`写第`4/5/6/7`个块；一个wave的64 lane恰好覆盖连续64个16B块，即连续1024B。
8条store依旧按K stage分为`2/2/4`，只改变物理地址。`sorted_sum`根据token location、逻辑column、
wave、lane group和store index执行逆映射。BN128/192保持原逻辑地址布局。

空闲card3同口径结果：

| 布局 | Down | TFLOPS | sorted_sum | VGPR+AGPR |
|---|---:|---:|---:|---:|
| 逻辑地址写出 | 1.9504 ms | 317.1 | 0.5669 ms | 108+132 |
| 整wave连续写出 | **1.5821 ms** | **390.9** | 1.2471 ms | 108+132 |

**核心结论：输出store是否能形成整wave顺序写出对down性能影响重大。** 在计算、资源和store
字节数不变时，仅将分散的逻辑地址写出改成每条wave连续1024B，down就降低18.9%；这说明原布局
的写合并、cache line/set映射或VMEM队列背压，而不是store指令数量本身，是主要性能因素。后续
padding实验将本次提交称为`base`，并以其1.5821 ms / 390.9 TFLOPS作为未改动最快基线。

连续写使down提升18.9%；`down + sorted_sum`从2.5173 ms变为2.8292 ms，但本实验目标只验证down
VMEM写出。最终ISA保持16条`buffer_store_dwordx4`，资源不变，地址计算中的乘加类指令减少16条。

连续写ATT为
`/tmp/moe_h3_n256_contiguous_store_att/ui_output_agent_61490_dispatch_2596`：

| Stall | 逻辑地址 | 连续地址 | 单次执行变化 |
|---|---:|---:|---:|
| total | 68.25M | 52.80M | 每MFMA采样归一后约-24% |
| VMEM store | 14.96M / 185.2 cycles | 4.74M / 57.5 cycles | **-68.9%** |
| VMEM load | 18.53M / 72.5 cycles | 6.99M / 26.8 cycles | **-63.0%** |
| MFMA | 28.25M / 15.2 cycles | 34.13M / 18.0 cycles | +18.5%，内存瓶颈解除后暴露 |

按阶段，K0/K1/K2的store stall分别从3.74M/3.52M/7.30M降到1.25M/1.26M/2.15M；末块drain
也从0.41M降到0.08M。连续store不仅减少store stall，也释放共享VMEM队列，显著降低后续weight
load stall。当前down已从VMEM主导转为MFMA主导。

#### BN256 sorted_sum优化

连续物理布局下，原始`sorted_sum`每个64-thread WG处理一个最终token。每线程处理12个128-bit
逻辑atom；TOPK=4时，每个atom发出4条随机`global_load_dwordx4`，转换到FP32后求和并写回。
基线为1.2471 ms；ATT目录为
`/tmp/moe_h3_sorted_sum_contiguous_baseline_att/ui_output_agent_4080_dispatch_2599`，总stall
174.72M / 93.0%，其中VMEM load 121.08M / 69.3%、VMEM wait 35.91M / 20.6%、VMEM store
14.29M / 8.2%。地址及其他算术只有1.6%，因此重点是load合并和MLP，而不是继续削减整数运算。

最终保留三项改动：

1. 将BN256源A包装成单个buffer resource，最终ISA使用`buffer_load_dwordx4`。
2. 重排8-lane内的逻辑column分工：

     ```text
     lane_in_octet = lane_id % 8
     physical_atom = (lane_id // 8) * 8
                                 + (lane_in_octet % 4) * 2
                                 + lane_in_octet // 4
     column = (atom_index * 64 + physical_atom) * 8
     ```

     原lane顺序对应的物理atom顺序为`[0,2,4,6,1,3,5,7]`；重排后lane `0..3`和`4..7`
     分别读取连续64B。每个lane仍将结果写回自己实际归约的逻辑column，不需要额外置换。
3. 将12个静态atom改成6次运行时循环；每轮先发两个atom共8条TOPK load，再依次FP32归约和
     写回。这样保持8路MLP，同时将H3 ISA从约1600行缩到571行，避免静态展开带来的长live range。

真实H3（B=32768、TOPK=4、N=6144）card3结果：

| 版本 | sorted_sum | Down | Down资源 | 结果 |
|---|---:|---:|---:|---|
| 连续布局原始逆映射 | 1.2471 ms | 1.5821 ms | 108+132 | 基线 |
| lane重排 + buffer load + 运行时双atom预取 | **1.1779 ms** | **1.5645 ms** | 108+132 | 保留 |

sorted_sum降低约5.5%；down没有代码改动，实测中位数仍优于1.5821 ms门槛。两者合计从
2.8292 ms降到约2.7424 ms，但仍高于原逻辑写出布局的2.5173 ms。

最终sorted_sum ATT为
`/tmp/moe_h3_sorted_sum_runtime_pair_att/ui_output_agent_26152_dispatch_2599`：56个arch VGPR、
64个SGPR、8 waves/SIMD。每轮动态body为8条`buffer_load_dwordx4`和2条
`buffer_store_dwordx4`；总stall 150.02M / 92.3%，VMEM load 121.22M / 80.8%、VMEM wait
12.85M / 8.6%、VMEM store 13.27M / 8.8%。双预取显著压低wait，但剩余瓶颈是物理布局下单逻辑行
最多只有4 lane组成连续64B读取，而不是occupancy或地址算术。

已否证方案：

- 单纯将global load替换为buffer load：相邻微基准1.4388 vs 1.4384 ms，指令形式改变但无性能收益。
- 预取深度3：随机微基准略快，但VGPR升到70、occupancy降至7 waves/SIMD；真实H3退化到1.2442 ms。
- 跨pair loop-carried预取：下一pair的8条load确实被移动到当前pair两次store之间，但ISA生成额外
    prologue/epilogue和`vmcnt(0)`边界，随机微基准从1.2294退化到1.2425 ms。
- 128/256线程：分别约1.3490/1.4424 ms，增加每token wave数反而增加调度和VMEM竞争。
- 512/1024/2048列分WG：随机loc下1024列块可到1.1170 ms、VGPR降到44，但真实MoE sorting分布
    退化到1.2344 ms，说明拆WG破坏了实际局部性。
- `ds_bpermute`恢复自然lane后连续store：增加48条交叉通道指令和`lgkmcnt`等待，仅提升约0.2%，
    按噪声回退。
- load cache modifier：SC0与默认等价，SC1绕L2退化约1.7%，NT与默认等价；保留默认缓存。
- 反向连续读后BF16 atomic scatter：仓库已有gfx942测试表明`global_atomic_pk_add_bf16`带宽约为
    普通写出的1/4，预计只会把随机读瓶颈换成更慢的原子写，未引入。

#### 原逻辑layout + N方向padding实验

后续将commit `a6a1632`称为`base`：它使用BN256整wave连续物理写出，down基线为
1.5821 ms / 390.9 TFLOPS。实验通过`MOE_DOWN_OUTPUT_PADDING_BYTES=0/32/64/128`恢复
`sorted_sum`原先要求的row-major逻辑layout，并在每个sorted row的N方向尾部增加padding；环境变量
未设置时仍运行`base`。

原逻辑layout中，down中间结果的第一维是sorting后的row，第二维是逻辑channel。每行的N个BF16
连续存放，`sorted_sum`使用`loc_ids[token, topk]`选择4行，并在相同channel位置求和：

```text
                         N / channel方向（地址递增）
                c=0                                      c=N-1
sorted row 0   [ x x x x x x x x | ... | x x x x x x x x ][ padding ]
sorted row 1   [ x x x x x x x x | ... | x x x x x x x x ][ padding ]
sorted row 2   [ x x x x x x x x | ... | x x x x x x x x ][ padding ]
                   ^ 一个sorted_sum lane读取8个BF16 = 16B ^

row_stride_bytes = N * sizeof(BF16) + padding_bytes
address(row, col) = base + row * row_stride_bytes + col * sizeof(BF16)

output[token, col:col+8]
    = sum(A[loc_ids[token, k], col:col+8], k=0..TOPK-1)
```

BN256 down的一条wave store在这个layout中的地址分布如下。`lane 0..15`写16个不同row的同一段N，
`lane 16/32/48`回到相同16个row并分别前进16/32/48个channel；因此一条wave指令不是连续1024B，
而是跨16行散布的64个16B transaction：

```text
lane       sorted row                 channel range（每lane 8 BF16）
0..15      token_repeat*16 + lane     c +  0 .. c +  7
16..31     token_repeat*16 + lane-16  c + 16 .. c + 23
32..47     token_repeat*16 + lane-32  c + 32 .. c + 39
48..63     token_repeat*16 + lane-48  c + 48 .. c + 55

下一条channel_piece store补齐每行的c+8..15、c+24..31、c+40..47、c+56..63。
```

H3的`N=6144`，无padding时row stride恰好为`12288B = 3 * 4096B = 192 * 64B`，相邻row
在4KB页内和64B cache-line编号上完全同相。若down退化主要来自cache set冲突，增加一个或两个
64B cache line应打散相邻row映射；32B则用于区分“任意错相”与“完整cache-line错相”。

在空闲card3上按`base -> 0 -> 32 -> 64 -> 128`和反向顺序各运行一轮；每轮剔除前两次warmup，
再合并两轮共18个稳态样本取中位数。五档完整H3均保持`diff=0.00105974`：

| 版本 | padding | row stride | stride mod 4KB | Down | TFLOPS | sorted_sum | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base`整wave连续物理layout | - | - | - | **1.5603 ms** | **396.4** | 1.1811 ms | 2.7414 ms |
| 原逻辑layout | 0B | 12288B | 0B | 1.9644 ms | 314.8 | **0.5661 ms** | 2.5305 ms |
| 原逻辑layout | 32B | 12320B | 32B | 1.9202 ms | 322.1 | 0.6124 ms | 2.5326 ms |
| 原逻辑layout | 64B | 12352B | 64B | 1.8619 ms | 332.2 | 0.5993 ms | 2.4612 ms |
| 原逻辑layout | **128B** | **12416B** | **128B** | **1.8266 ms** | **338.6** | **0.5766 ms** | **2.4032 ms** |

结果有两个层次：

- 对down单kernel，`base`整wave连续1024B写出仍最快；128B padded逻辑layout比`base`慢17.1%。
- 在原逻辑layout内部，padding收益随32/64/128B单调增加。128B相对0B让down降低7.0%；同时
    避免物理layout的昂贵逆映射，使`down + sorted_sum`比`base`降低12.3%，是本轮端到端最佳。

0B与128B的第5次稳态down ATT分别为：

- `/tmp/moe_pad_att_0/ui_output_agent_61597_dispatch_2596`
- `/tmp/moe_pad_att_128/ui_output_agent_5719_dispatch_2596`

两者ISA静态指令数、动态mix、资源和occupancy完全一致：192条MFMA、44条buffer load、16条
buffer store，`108 VGPR + 132 AGPR`、112 SGPR、25,600B LDS、2 waves/SIMD。只有地址stride
不同，因此可以直接比较stall：

| ATT指标 | 0B | 128B | 变化 |
|---|---:|---:|---:|
| total cycles | 89.38M | 84.77M | -5.2% |
| total stall | 69.47M | 64.66M | -6.9% |
| VMEM store stall | 15.34M | 12.25M | **-20.1%** |
| VMEM load stall | 18.61M | 15.46M | **-16.9%** |
| VMEM wait stall | 2.32M | 2.35M | +1.3% |
| MFMA stall | 29.23M | 30.61M | +4.7% |

PMC进一步排除了“padding减少写事务”的解释。第5次稳态dispatch中：

| PMC | 0B | 128B | 变化 |
|---|---:|---:|---:|
| `TCC_WRITE_sum` | 50,331,648 | 50,331,648 | 0 |
| `TCP_TCC_WRITE_REQ_sum` | 50,331,648 | 50,331,648 | 0 |
| `TCC_EA0_WRREQ_DRAM_sum` | 25,179,646 | 25,170,164 | -0.04% |

steady-state的`TCC_EA0_WRREQ_STALL_sum`和`TCC_TOO_MANY_EA_WRREQS_STALL_sum`均为0，所以不是
DRAM credit耗尽或pending写请求达到硬上限。综合来看，128B padding没有减少软件请求、L2写请求
或HBM写事务，却让同样的store请求完成得更快，并连带降低后续weight load stall。**这支持原始
`3 * 4KB`行stride造成cache set/TCC地址映射同相、局部排队或bank/channel争用的假设。** 当前
rocprof SDK将非sum TCC counter仍聚合为单值，尚不能直接展示16个TCC实例的负载倾斜。

#### 兼顾down与sorted_sum：tile-major AoSoA

![R2 tile-major AoSoA布局示意图](r2_tile_major_aosoa.png)

图片只画一个wave（W0，负责N0..63），并将过程拆成三个独立区域：

1. **逻辑值到线程。** 每个格子正好表示8个BF16，即一个128-bit/16B atom；格内`Txx/Sy`
    分别表示负责该atom的wave thread和动态store slot。主网格只展开`t0`的16行，`t1/t2/t3`
    使用相同64个thread，store slot依次变为`S2/S3`、`S4/S5`、`S6/S7`。
2. **Down写入物理内存。** 分开画出S0和S1两条wave store；每个R2双行组分别形成一个连续
    128B段，8个双行组合计为`8 x 128B = 1024B`。
3. **sorted_sum读取。** 以source row 0为例，蓝框分别标出两个连续64B读取组；灰色格属于配对
    row 1，读取row 0时跳过。箭头显示8个物理atom如何交给sum lane 0..7并恢复逻辑N顺序。

完全连续物理layout按`[N block, wave, store index, physical lane]`排列，down每条wave store覆盖
连续1024B，但同一逻辑row的数据被wave和store index拆散。row-major layout则反过来：每行完全
连续，但一条down wave store跨16行散布。两者不是只能二选一，可以按小组row构造AoSoA：

```text
tile-major顺序：
    [N256 block]
        [16-row token repeat]
            [R-row group]
                [wave 0..3，每wave负责N64]
                    [channel piece 0..1，每piece为8 channel]
                        [R rows]
                            [lane group 0..3]
```

令`R in {1,2,4,8,16}`，每个元素仍是8个BF16/16B atom。物理atom编号为：

```text
row_block   = row_in_16 // R
row_in_blk  = row_in_16 % R

physical_lane = ((row_block * 4 + wave_id) * (2 * R * 4)
                                 + channel_piece * (R * 4)
                                 + row_in_blk * 4
                                 + lane_group)

physical_chunk = block_n * (64 * 256 / 8)
                             + token_repeat * (16 * 256 / 8)
                             + physical_lane
```

这样一条down wave store仍然具有大粒度连续段，而同一逻辑row在一个N256 tile内的四个N64 wave
共512B也被聚在同一局部区域：

| R | down每条wave store | sorted_sum行局部性 |
|---:|---:|---:|
| 16 | 1段 x 1024B | 16行AoSoA；同row跨wave距离较大 |
| 8 | 2段 x 512B | 8行AoSoA |
| 4 | 4段 x 256B | 4行AoSoA |
| **2** | **8段 x 128B** | **2行AoSoA；同row N256共512B局部聚集** |
| 1 | 16段 x 64B | 单行N256连续512B，但down连续段最小 |

通过`MOE_DOWN_OUTPUT_ROW_GROUP=1/2/4/8/16`选择；环境变量未设置时严格保持`base`原地址公式。
五档地址在一个`64x256` tile上均穷举验证为完整双射，并通过down写出到sorted_sum恢复的GPU
正确性测试。第一次五档正反sweep结果：

| R | Down | sorted_sum | 合计 |
|---:|---:|---:|---:|
| 16 | 1.5690 ms | 1.2120 ms | 2.7811 ms |
| 8 | **1.5626 ms** | 1.1979 ms | 2.7605 ms |
| 4 | 1.5850 ms | 0.8912 ms | 2.4763 ms |
| **2** | 1.6037 ms | **0.7104 ms** | **2.3142 ms** |
| 1 | 1.9048 ms | 0.5628 ms | 2.4677 ms |

随后恢复“环境变量未设置=原base”约束，在完全空闲card3上对`base / R2 / 128B padded row-major`
按正反顺序公平复测，合并18个稳态样本：

| 版本 | Down | sorted_sum | 合计 | 相对base合计 |
|---|---:|---:|---:|---:|
| `base`整wave连续 | **1.5675 ms** | 1.1823 ms | 2.7498 ms | - |
| **tile-major R2** | **1.5953 ms** | **0.7116 ms** | **2.3069 ms** | **-16.1%** |
| 128B padded row-major | 1.8126 ms | **0.5757 ms** | 2.3883 ms | -13.1% |

R2相对base只牺牲1.8%的down，却让sorted_sum降低39.8%；端到端比base快16.1%，也比此前最佳
128B padded row-major快3.4%。这就是当前“兼得”方案：不增加中间结果字节数，不增加额外kernel，
只改变tile内物理地址排列。

R2 ATT：

- down：`/tmp/moe_r2_att/ui_output_agent_38743_dispatch_2596`
- sorted_sum：`/tmp/moe_r2_att/ui_output_agent_38743_dispatch_2599`

资源与base保持同档：down为`108 VGPR + 132 AGPR`、25,600B LDS、2 waves/SIMD；sorted_sum为
52 VGPR + 4 accum VGPR、8 waves/SIMD。R2 down ATT中VMEM store stall为5.88M、VMEM load为
8.45M；虽然不如base的整wave1024B极限，但仍远优于row-major。sorted_sum仍由VMEM load主导，
但tile-major行局部性将真实H3时间压到约0.71 ms。
