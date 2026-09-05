# Paged Prefill 4-wave/8-wave 优化与性能报告

更新时间：2026-09-03。当前实现已完成gfx942/gfx950 BF16、架构原生FP8、
non-SWA及gfx950 SWA支持。本文件以gfx950当前结果为主；2026-08的gfx942数据统一
归档在“历史结果”中，不与当前表直接横向比较。

## 当前结果总览

当前表均在MI350X `gfx950`上测量，主配置为
`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。除特别说明外，协议均为同进程、
同逻辑输入、20次预热、100个CUDA event样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | AITER | 4-wave static | 4-wave dynamic | 8-wave | 结论 |
|---|---|---:|---:|---:|---:|---|
| non-causal `Q10240,KV2583` | BF16 | **282.27 us / 959.53T** | 304.07 us / 890.74T | 314.05 us / 862.43T（独立轮次） | 350.43 us / 772.89T native | 8-wave API已删除4-wave回退；native 8-wave同轮仍比4-wave慢15.24% |
| causal `Q=KV=32768` | BF16 | 15561.18 us / 353.29T | **5748.63 us / 956.32T** | 7032.47 us / 781.74T | 8462.96 us native（历史） | 8-wave不再自适应到4-wave；本行native数据为历史测量 |
| causal `Q=KV=32768` | OCP FP8 K64 | N/A | **4230.13 us / 1299.62T** | 4815.89 us / 1141.54T | 5163.99 us / 1064.60T | static最快；dynamic为static的0.878x且快于8-wave |
| SWA `Q16K,KV128K,window=128` | BF16 | 123.39 us varlen-only / 269.03 us含gather | **77.94 us / 277.68T有效** | 84.44 us / 256.31T有效 | 142.20 us native（历史） | 8-wave不再自适应到4-wave；本行native数据为历史测量 |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | N/A | **61.00 us / 354.78T有效** | 66.56 us / 325.14T有效 | 113.72 us / 190.31T有效 | static/dynamic分别较8-wave快1.864x/1.708x |

口径说明：

- non-causal FLOPs为`2 * Hq * Q * KV * (Dqk + Dv)`；causal按三角有效工作量减半；
- SWA有效FLOPs按每行最多`window_left + 1 = 129`个可见KV token计算；硬件执行
  TFLOPS另计被mask但仍进入MFMA的tile工作，详见
  [gfx950_swa_performance.md](../pa_8wave/gfx950_swa_performance.md)；
- 每张明细表内的数据可以直接比较；不同架构、dtype、shape或计时协议的数据不混算提升；
- dynamic为同一4-wave kernel的persistent ticket分支；总览中的BF16行来自包含全部候选
  的同轮次测试，FP8的static/dynamic来自同轮次配对测试；
- 8-wave API始终执行原生512线程kernel，不再导入或调用4-wave后端。AITER等长
  D192/V128使用linear/page-size-1 batch-prefill ABI，非等长使用linear THD；
  FlyDSL使用page64 vectorized cache。三者共享逻辑Q/K/V，计时不包含布局转换。

## 当前功能范围

| 架构 | BF16 | 原生FP8 | non-SWA | SWA + sink | page size |
|---|---|---|---|---|---|
| gfx942 | 支持 | `float8_e4m3fnuz` | 支持 | 未作为当前生产基线 | 32/64/128 |
| gfx950 | 支持 | OCP `float8_e4m3fn` | 支持 | BF16/FP8支持 | 32/64/128 |

launcher会拒绝与目标架构不匹配的FP8格式，避免相同8-bit载荷按另一种指数/NaN编码
解释。non-SWA回归覆盖Dqk/Dv为128/128、192/128和192/192、causal/non-causal、
page 32/64/128、per-token/per-tensor Q scale及ragged tail。

## gfx950当前明细

### Non-SWA AITER BF16路由

跨4-wave、8-wave和AITER的BF16结果只保留在顶部总览。对比测试显式按长度分流：

| 条件 | AITER公开入口 | profiler验证的实际事件 |
|---|---|---|
| `Q == KV` | `mha_batch_prefill_func` | `aiter::mha_batch_prefill` |
| `Q != KV` | `flash_attn_varlen_func` | `FlashAttnVarlenFunc` |

默认小shape回归校验non-SWA路由互斥、SWA两种线性AITER入口和三方输出，结果为
`3 passed`。生产性能分别由`PYHIP_RUN_PA_AITER_PERF=1`和
`PYHIP_RUN_PA_AITER_SWA_PERF=1`显式开启。

### Non-causal BF16 native 8-wave瓶颈分析（`Q10240,KV2583`）

AITER在该shape命中gfx950专用OPUS
`gqa_d192_v128_kernel<32,64,8,non-causal,group>`，不是通用CK fallback。下面的PMC
来自同一逻辑输入上的单kernel dispatch；profile延迟与顶部5轮中位数略有差异，但
相对关系一致。

| 指标 | 4-wave | native 8-wave | AITER OPUS |
|---|---:|---:|---:|
| Q tile / KV tile | 128 / 32 | 256 / 32 | 256 / 64 |
| 物理WG / 逻辑任务 | 1280 / 1280 | 256 / 640 | 640 / 640 |
| 物理wave启动数 | 5120 | 2048 | 5120 |
| BF16 MFMA | K16 | K16 | K16 |
| `SQ_INSTS_MFMA` | 8.294M | 8.397M | 8.397M |
| `SQ_INSTS_VALU` | 45.107M | 89.095M | 40.092M |
| `SQ_INSTS_VMEM_RD` | 4.869M | 1.395M | 1.152M |
| `SQ_INSTS_LDS` | 6.420M | 9.675M | 11.817M |
| LDS bank conflict | 0.492M / 1.60% | 14.418M / 33.50% | 0 / 0% |
| L2 miss | 1.180M | 1.139M | 1.124M |
| `MfmaUtil` | 53.64% | 31.90%-32.56% | 56.4%-56.6% |
| `MeanOccupancyPerActiveCU` | 1.72 | 2.00 | 1.98 |
| ISA VGPR / SGPR / LDS | 212 / 91 / 24960 B | 237 / 100 / 49940 B | 250 / 70 / 149760 B |
| 当前延迟 | 304.07 us | 350.43 us | 282.27 us |

设置`AITER_DISABLE_FMHA_OPUS=1`后，同一输入会落到AITER
`fmha_fwd_hd192_hd128_bf16_group` ASM，独立100-event中位为296.15 us。当前4-wave
以本轮数值作参考，4-wave相对该历史独立值慢1.18%，native 8-wave约为其1.37x。差距
不是Python dispatch，
剩余重点是KV64 softmax频率、barrier锁步和跨tile流水。

结论按影响排序：

1. **4-wave gfx950 BF16 K16已完成。** 动态MFMA从16.589M降到8.294M；输出与
  AITER保持`2e-2`容差内一致。K预取现使用buffer-to-LDS direct DMA，目标ISA为
  212 VGPR、24960 B LDS、0 scratch，最终同轮三次为299.29/299.65/300.59 us。
2. **gfx950 BF16 D192改用HW-slot互补priority。** 保持计算、访存、barrier和调度
  fence不变的三路A/B中，统一`p2/p0`、全程`p0`和slot-aware分别为316.71、322.13和
  312.87 us；slot-aware较统一`p2/p0`快1.23%，较全程`p0`快2.96%。wave slot 0的
  MFMA/softmax使用`p3/p1`，其他驻留wave使用`p2/p0`，避免所有wave同时以相同高
  priority竞争。fresh ATT中`p3/p1` wave的高/低阶段累计周期比中位数为2.04，统一
  `p2/p0`基线为2.54；资源仍为212 VGPR、24960 B LDS、0 scratch。
3. **BN32不会使概率EXP翻倍。** 每个wave在4-wave的一个BN32 tile持有16个score，
  AITER的一个BN64 tile持有32个score；按64 keys归一化，两者都是32个概率EXP。
  4-wave每个BN32还无条件执行1个在线max校正EXP，因此每64 keys比AITER约多2个
  TRANS。目标shape的PMC为4-wave 7.060M、AITER 6.733M，只差4.9%，EXP不是主要
  性能差距。每wave可以精确写成4-wave `81*16 + 81 + 2 = 1379`，AITER
  `41*32 + 3 = 1315`；其中前一项是概率EXP，其余为校正和epilogue TRANS。
  4-wave的逐lane条件被LLVM if-convert为无条件`v_exp + v_cndmask`；AITER先用
  wave ballot形成uniform分支，只在max真正推进时执行校正EXP。BN32真正重复的是
  row-max、在线max/sum更新、条件判断和布局控制。
4. **row-max和lazy-max的冗余VALU已压缩。** 16项row-max由LLVM通用归约改为5条
  `v_max3`初级归约、2条`v_max3`二级归约和1条`v_max`，cross-lane交换直接使用
  gfx950 `v_permlane32_swap`。二者组合A/B提速1.51%，并将VGPR从237降到230。
  在线状态改为AITER式阈值8的无偏lazy-max，直接选择`row_max/running_max`，再从
  选中的max计算correction；同时复用`max_advances`控制O重标定。每个展开块因此少
  1条ADD、1条compare和1条`cndmask`，correction改写单项提速0.31%-0.42%。
5. **page64 lookahead复用已删除重复页表工作。** pair内第一块的
  `table[(block+3)/2]`恒等于循环状态中的`prefetch_k_page_id`，只在第二块读取下一页。
  静态页表`buffer_load_dword`站点从5降到3，VMEM read从5.079M降到4.869M，INT32
  降到1.649M；同进程A/B提速0.82%-0.94%。
6. **8-wave V跨wave LDS复用已验证但未保留为目标默认路径。** D192/V128 BF16由
  512线程协作加载V到三槽LDS ring，VMEM read从4.319M降到1.395M，已接近AITER的
  1.152M；代价是LDS指令从5.881M增至9.675M、VALU增至89.095M。独立同进程A/B中，
  K-only为397.71 us，K+V LDS为410.95 us（回退3.33%）。其他head维度继续走原
  global-to-register V路径。
  当前native 8-wave为每个错相4-wave组使用独立两槽K LDS + direct-V，P@V位于
  MFMA stage。K通过buffer-to-LDS direct DMA写入pair-padded LDS，目标资源为
  233 VGPR / 49940 B LDS / 0 scratch。目标epilogue以direct
  store替代C-shuffle，静态barrier为31；公开API始终运行该512线程实现。
7. **K预取已改为direct LDS且保持低冲突。** 4-wave与8-wave均使用
  `raw_ptr_buffer_load_lds`，每条64-lane DMA搬两个32-row D-group，并在每个pair后留
  16 B padding；ISA中K侧`ds_write_b128`为0。4-wave在KV32和KV2583上的
  `SQ_LDS_BANK_CONFLICT`均固定为49152，说明新增80个K block带来0个K-LDS冲突，
  残余来自固定C-shuffle；8-wave KV2583整体为0冲突。8-wave必须为两个错相4-wave组
  分配独立双槽ring，否则KV96开始出现非确定性覆盖。
8. **AITER流水仍更完整。** 它保留两份score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`跨KV64 tile重叠，并将8个wave拆成两组错开一个stage。8-wave
  当前在每个BN32的QK、softmax和P@V阶段间用全workgroup barrier锁步；结果是相同
  MFMA数量下利用率只有约32%，AITER约56.5%。persistent ticket不是主要差异：
  8-wave虽只启动256个物理WG并循环处理640个逻辑任务，GPU利用率仍为100%，
  `MeanOccupancyPerActiveCU`也与AITER同为约2。
  原生8-wave的BN32双score一barrier原型已做到目标ragged shape逐bit一致，但三轮
  为547.20/546.10/549.66 us，对照native 390.53/389.99/391.89 us，回退约40%。
  它没有KV64带来的softmax频率减半，因此不能验证或替代真正的KV64设计。

#### 4-wave与AITER的VALU分类

同一`Q10240,KV2583,non-causal`单dispatch，二者均启动5120个wave。按每wave和每64
keys归一化后：

| PMC类别 | 4-wave / wave | AITER / wave | 4-wave / 64 keys | AITER / 64 keys | 差额 / 64 keys |
|---|---:|---:|---:|---:|---:|
| 全部VALU | 8810.0 | 7830.5 | 217.53 | 190.99 | +26.54 |
| TRANS F32 | 1379.0 | 1315.0 | 34.05 | 32.07 | +1.98 |
| FMA F32 | 1464.0 | 1352.0 | 36.15 | 32.98 | +3.17 |
| ADD F32 | 1378.0 | 1353.0 | 34.02 | 33.00 | +1.02 |
| MUL F32 | 68.0 | 84.0 | 1.68 | 2.05 | -0.37 |
| INT32 | 322.0 | 139.0 | 7.95 | 3.39 | +4.56 |
| INT64 | 41.0 | 4.0 | 1.01 | 0.10 | +0.91 |
| CVT | 682.0 | 700.0 | 16.84 | 17.07 | -0.23 |
| SALU（不计入VALU） | 548.0 | 948.5 | 13.53 | 23.13 | -9.60 |

差额来源按影响排序：

1. **gfx950 BF16转换已对齐native指令。** `_cvt_f32_to_bf16`在函数内部检查平台：
  gfx950使用vector cast并发射`v_cvt_pk_bf16_f32`，gfx942继续使用原有round-half-up
  的`+0x8000`、右移和`v_perm`。相对手写路径，gfx950每wave少1360条INT32、增加
  680条CVT，总VALU从59.407M降到52.444M（-11.7%），VGPR从237降到226。
  当前4-wave的CVT已与AITER基本相同；最终仍有322对139条INT32/wave，来自地址与控制。
2. **核心循环的`v_pk_add_f32`已消除。** 生成源有两处：BF16的vector
  `scaled_score - updated_max`被后端配成8条packed add/BN32；FP8的
  `llvm.vector.reduce.fadd.v16f32`被配成7条packed add/展开块。BF16现在直接使用
  16条`v_fma_f32(score, scale, -updated_max)`融合scale-sub；FP8使用逐元素
  `v_sub_f32`和15条标量add归约树。最终BF16/FP8静态ISA的`v_pk_add_f32`均为0。
  BF16同进程旧/新A-B为343.97/313.27 us（提速9.8%），总VALU从52.444M降到
  49.126M，ADD差额从+19.02降到+3.02条/64 keys；最大输出差0.000488。
3. **D192 K写已与EXP交织。** 原稳态每轮三条`ds_write_b128`连续发射，最后一条后
  仅约8条BF16 CVT便进入`lgkmcnt(0)+barrier`。单纯把K写整体移到softmax前会因提前
  等VMEM回退3.3%-3.8%；仅靠跨basic-block scheduler hint也不生效。最终方案延后
  output rescale，使17条EXP和3条K写进入同一调度区，并按
  `4 EXP -> write -> 4 EXP -> write -> 4 EXP -> write -> 5 EXP`交织。ATT确认稳态
  三条write两侧均有EXP，最后write到barrier约58条记录；同进程A/B提速0.92%，
  输出逐bit一致。
4. **显式max树、无偏lazy-max和谓词复用继续消除重复工作。** 最终总VALU进一步降到
  45.107M，ADD差额降到+1.02条/64 keys；静态ISA相对上一版每个展开块少1条ADD、
  1条compare和1条`cndmask`，仍为零`v_pk_add_f32`。
5. **BN32仍重复其他非EXP softmax工作。** 4-wave每64 keys做两次16项sum、row max、
  running max/sum更新和阈值选择；AITER对32项一次完成。sum reduction本身是
  `2*15`对`31`次add，数量近似相同；剩余差距来自row-max和在线状态更新等控制。
6. **未分类VALU主要是重复控制与布局操作。** 扣除转换净开销和细分浮点counter后，
  仍包含两次row-max后的`max/compare/cndmask`、两次在线状态选择、额外cross-lane
  交换、概率布局move/permute及地址计算。gfx950现有counter不能再将这些类别分开，
  因此不把该余量强行归到单一源码操作。AITER的SALU反而多9.60条/64 keys，说明
  它把更多统一控制留在scalar侧，而4-wave有更多逐lane vector控制。
7. **per-wave direct V主要造成VMEM差距。** 4-wave每个wave为自己的32行Q读取V，
  AITER由workgroup协作将BN64 K/V各搬入一次LDS。lookahead复用后4-wave的VMEM read为
  4.869M，AITER为1.152M（4.23倍）；这是issue/流水压力，不是上述INT32差额的主要来源。
8. **流水而非指令总量。** 即使移除上述冗余，4-wave仍是单score fragment，阶段间
  barrier锁步；AITER用双score、两组4-wave错相，将EXP/VALU隐藏在MFMA之后。
  本轮缩短非MFMA stage后4-wave `MfmaUtil`已从约46%升到53.64%，仍低于AITER约56.5%。

`-packed-fp32-ops`不是VALU差额原因。诊断A/B分别给转换前的同一BF16 specialization传入
`-packed-fp32-ops`和`+packed-fp32-ops`，MLIR属性确认相反，但最终ISA SHA256完全
一致：两者都是68个静态`v_exp`、192个静态`v_pk_*_f32`、237 VGPR、0 scratch。
也就是说当前编译器仍选择了packed add/mul；该target feature在这条ISA上没有产生
任何变化。

转换A/B中manual/native延迟为347.37/339.03 us（native快2.46%），两者对AITER最大
绝对误差均为0.0009766。首次组合测试的异步GPU fault来自前一个candidate尚未同步时
开始JIT/加载下一code object；在各候选首次launch后加入同步后，non-SWA与SWA两项
完整生产组合均通过。当前按8-wave相同方式保留平台门控。

当前优化状态与TODO：

1. [x] 4-wave gfx950 BF16 QK/P@V切换K16，并同步probability/K-row layout。
2. [x] 8-wave重做K-LDS padding/read permutation，将K-only冲突率降到3.76%。
3. [x] 8-wave V由workgroup协作加载到三槽LDS并跨8个wave复用；功能完成，当前性能
  取舍如上，后续优化不能只以冲突率作为验收指标。
4. [ ] 将8-wave内部KV tile提升到64并保留两个score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`重叠，同时把softmax频率减半；不再尝试已回退约40%的BN32
  双score版本。
5. [ ] 将8个wave拆成两组4-wave并错开一个pipeline stage；验收要求`MfmaUtil > 50%`、
  0 scratch，且不能恢复K-LDS冲突。除已保留的等价lookahead复用外，不再优先优化
  page-table或L2：剩余计数不支持该方向。

### 4-wave static/dynamic当前数据

同一输入、20次预热、100个CUDA event样本、5轮中位数；static与dynamic在每个case
按对应行注明的容差一致。`dynamic/static`为static延迟除以dynamic延迟，小于1表示dynamic较慢。

| 场景 | dtype | 4-wave static | 4-wave dynamic | dynamic/static |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | BF16 K16 direct-LDS | 312.73-314.21 us | 321.07-322.73 us | 0.974x |
| causal `Q=KV=32768` | BF16 K16 | 5748.63 us / 956.32T | 7032.47 us / 781.74T | 0.817x |
| SWA `Q16K,KV128K,window=128` | BF16 K16 | 77.94 us / 277.68T有效 | 84.44 us / 256.31T有效 | 0.923x |
| causal `Q=KV=32768` | OCP FP8 K64 | 4230.13 us / 1299.62T | 4815.89 us / 1141.54T | 0.878x |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | 61.00 us / 354.78T有效 | 66.56 us / 325.14T有效 | 0.916x |

slot-aware priority的最终同轮结果中，non-causal目标shape的dynamic比static慢
2.6%-2.7%；本轮dynamic相对static通过`rtol=atol=2e-2`。其他表项为此前同轮历史
数据，不据此外推。dynamic仍用于batch>1负载均衡，短non-causal/causal与SWA回归
均验证dynamic和static逐bit一致。

### Non-SWA FP8 K16/K64 A/B

生产shape为`Q=KV=32768,causal`，算法工作量为`5.497558 TFLOP`。相同量化输入下
K16与K64输出逐bit一致。

| Kernel | K16延迟 / TFLOPS | K64延迟 / TFLOPS | 延迟下降 |
|---|---:|---:|---:|
| 4-wave | 5237.05 us / 1049.74T | **4651.73 us / 1181.83T** | **11.18%** |
| 8-wave | 5611.21 us / 979.75T | **5101.77 us / 1077.58T** | **9.08%** |

gfx950 OCP FP8 QK使用`v_mfma_f32_32x32x64_f8f6f4`，并通过
`scale_a=scale_b=0`表达unity E8M0 scale。P@V reduction只有32，继续使用
`v_mfma_f32_32x32x16_fp8_fp8`。K64仅在Dqk可被64整除时启用。

### SWA

生产配置为`Q=16K,window_left=128,page=64,bottom-right causal`，带per-head FP32
sink。4-wave按query tile裁剪page table，窗口外page不会被读取；batch=1默认走
static，`force_dynamic_schedule=True`时走persistent。

BF16 scheduler sweep结果（K16/V-LDS前历史A/B）：

| Total KV | 4-wave static | 4-wave dynamic | 8-wave persistent | Dynamic vs 8-wave |
|---:|---:|---:|---:|---:|
| 32K | 103.95 us | 110.31 us | 141.64 us | 1.284x |
| 64K | 103.89 us | 110.34 us | 141.99 us | 1.287x |
| 128K | **103.94 us** | 110.26 us | 142.15 us | 1.289x |

128K历史dtype与K64 A/B结果（均早于本轮softmax/lookahead优化，仅用于记录K64收益）：

| dtype | 4-wave static | 8-wave persistent | 4-wave vs 8-wave |
|---|---:|---:|---:|
| BF16 | **103.15 / 104.01 us** / 209.82 / 208.08T有效 | 141.04 / 142.53 us / 153.45 / 151.85T有效 | 1.367x / 1.370x |
| OCP FP8 K16 | **71.64 us / 302.10T有效 / 599.52T执行** | 112.90 us / 191.70T有效 / 570.63T执行 | 1.576x |
| OCP FP8 K64 | **67.05 us / 322.78T有效 / 640.56T执行** | 112.35 us / 192.64T有效 / 573.43T执行 | **1.676x** |

SWA AITER有两种计时口径：

1. **含gather端到端**：从5D cache gather到linear K/V，再调用AITER；
2. **attention-only**：K/V已是linear布局，只计AITER attention kernel。

同进程、同输入、20次预热、100样本、5轮中位数；三行BF16均来自本轮最终实现：

| KV | gather + `mha_batch_prefill_func` | `mha_batch_prefill_func` only | gather + `flash_attn_varlen_func` | `flash_attn_varlen_func` only | 4-wave static | 8-wave persistent |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 281.37 us / 76.92T | 259.55 us / 83.39T | 146.72 us / 147.51T | 116.86 us / 185.20T | **78.32 us / 276.34T** | 143.45 us / 150.87T |
| 64K | 323.29 us / 66.94T | 260.41 us / 83.11T | 180.15 us / 120.14T | 118.20 us / 183.10T | **78.10 us / 277.11T** | 143.53 us / 150.79T |
| 128K | fault，未计时 | fault，未计时 | 269.03 us / 80.45T | 123.39 us / 175.40T | **77.94 us / 277.68T** | 142.20 us / 152.20T |

两种线性AITER API都正确支持SWA：`mha_batch_prefill_func`实际命中
`aiter::mha_batch_prefill`，`flash_attn_varlen_func`命中`aiter::mha_varlen_fwd`；
两者小shape最大绝对差为`0.0078125`。varlen更快且覆盖128K，因此是当前推荐的
attention-only AITER基线。真正从5D page64 cache直连时，只有
`mha_batch_prefill_func`具备paged ABI，但当前gfx950 BF16 D192/V128构建没有匹配
specialization；`flash_attn_varlen_func`是linear THD接口，不是direct-paged入口。

### 当前资源

gfx950 OCP FP8 D192当前可分发的K64 specialization：

| Kernel | 调度 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave K64 | static | 174 | 68 | 16384 B | 0 B | 0 | 0 |
| 4-wave K64 | dynamic | 178 | 93 | 16388 B | 0 B | 0 | 0 |
| 8-wave K64 | persistent | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64相对K16把QK静态MFMA站点减少4倍；P@V仍使用K16。

gfx950 BF16 D192/V128当前目标specialization：

| Kernel | 调度 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave direct K-LDS | static | 212 | 91 | 24960 B | 0 B | 0 | 0 |
| 4-wave direct K-LDS | dynamic | 218 | 93 | 24964 B | 0 B | 0 | 0 |
| 8-wave direct K-LDS + direct-V/direct-store | persistent | 233 | 100 | 49940 B | 0 B | 2 SGPR / 0 VGPR | 0 |

### 当前验证

| 范围 | 结果 |
|---|---|
| 4-wave合并测试文件 | `51 passed, 2 skipped`；skip为两项可选生产性能测试 |
| 4-wave dynamic生产输出 | non-SWA与SWA两项opt-in测试各`1 passed`，均包含static/dynamic逐bit检查 |
| 8-wave gfx950完整矩阵 | `67 passed, 1 skipped` |
| AITER non-SWA路由 + SWA双入口 + 三方正确性 | `3 passed` |
| non-SWA K16/K64 | 各kernel内逐bit一致 |
| SWA K16/K64 | 各kernel内逐bit一致；4/8-wave relative-L2 `6.3745e-5` |
| focused FP8 ISA | 0 private、0 spill、0 scratch |
| persistent counter复用 | 默认stream连续16次、两个非默认stream各8次均通过 |

## 当前实现与已完成优化

- 4-wave为`BM128 x BN32 x 256 threads`；batch=1走static grid，batch>1走
  atomic-ticket persistent grid；
- 8-wave为`BM256 x BN32 x 512 threads`，使用1 WG/CU persistent调度；
- 4-wave使用独立`k_lds0/k_lds1`字段构成pair-padded direct-DMA K LDS ping-pong，
  V保持global-to-register。稳态先以`vmcnt(8)`等待上一轮K DMA，同时保留当前V请求；
  gfx950 BF16 D192按HW wave slot为MFMA/softmax阶段选择`p3/p1`或`p2/p0`；进入MFMA
  阶段后以两条MFMA起步，再将K `ds_read_b128`与P@V MFMA逐条交织，读取完成后才向
  已消费槽发起下一次K DMA。prologue和epilogue的`vmcnt(0)`分别保护首次读取和
  K/output union复用；稳态剩余的机器级`vmcnt(0)`只保护最终V fragment消费；
- 8-wave native D192/V128 BF16为两个错相4-wave组分别使用direct-DMA双槽K LDS ring，
  并使用direct-V和direct output store；K的每两个D-group后padding 16 B，再由K16
  置换视图读取。P@V位于MFMA stage；公开8-wave API始终运行该原生路径；
- online softmax使用显式8指令max树、direct permlane32、阈值8的无偏lazy rebase和
  loop-carried max/sum，并复用推进谓词控制O重标定；
- page64的pair首块复用已携带的lookahead page id，只在pair第二块读取下一页；
- 32x32 MFMA减少row max/sum cross-lane开销，`v_permlane`避免经LDS交换；
- FP8采用gfx950 K64 QK、K16 P@V、score MUL split11和FP8-only fast-math；
- BF16/FP8共享pipeline骨架，dtype专属K搬运、scheduler、probability、V布局和epilogue
  地址封装在独立helper中。

Dynamic scheduler工作已全部完成：

- 消除persistent BF16 D192 spill，在消费点重物化C-shuffle地址和SWA mask坐标；
- 按device/stream/grid复用`work_counter`，最后退出的workgroup在device端复位；
- ticket改为4-byte LDS mailbox广播，每个ticket由双barrier缩减为单barrier；
- ticket barrier同时封闭前一work item的C-shuffle生命周期，删除重复re-entry barrier；
- 4-wave的1/2/3/4 WG-per-CU sweep选择2 WG/CU；8-wave选择1 WG/CU；
- 最终trace中static/dynamic kernel为97.34/105.52 us，稳态dynamic没有额外初始化dispatch。

版本：4-wave `7ed2215b0027881af91ad5e12f253288254396db8a21265c5d3bfd65e8d5d375`；
8-wave `160b38fcc74696fd05115d6ebfbe757b34592f53fde56c713b2e64b201bada7b`。

## 复现当前结果

```bash
cd /root/workspace/luocheng/pyhip
export HIP_VISIBLE_DEVICES=7
export FLYDSL_RUNTIME_ENABLE_CACHE=0

PA_CASE=tails PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=batch PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=noncausal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=causal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_DTYPE=bf16 PA_CASE=bf16_ref_short PA_NUM_ITERS=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py

python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_matches_dispatched_aiter \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_paths
PYHIP_RUN_PA_AITER_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_aiter_production_performance -s
PYHIP_RUN_PA_AITER_SWA_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_production_performance -s
```

测试前用`rocm-smi --showuse`选择空闲GPU。完整causal reference需要约64GB临时显存；
定频诊断流程见`tests/flydsl/H3_ATTENTION_THROTTLE_PROFILE.md`。

## 历史结果（gfx942）

以下数据来自MI308X/gfx942及2026-08版本，仅用于记录优化演进，不代表当前gfx950性能。

### 2026-08-14 BF16 D192 spill/occupancy验收

目标只有两项：

1. 为8-wave补齐BF16 `Dq=192,Dv=128`，修复其tail/causal mask和dtype编译缓存，
  并增加D128/D192 BF16回归；FP8路径保持兼容。
2. 修复4-wave batch>1 persistent BF16 D192的低occupancy分配。原
   `vgpr_count=265`实际为`256 VGPR + 9 AGPR`，并非普通VGPR越过硬件上限；
  将`rocdl.waves_per_eu=2`直接附着到persistent GPU kernel后，最终LLVM IR带有
  `amdgpu-waves-per-eu="2"`。page32/128变为`256 VGPR + 0 AGPR / 10 spill / 44B private`，
  page64 non-causal为`22 spill / 92B private`，causal为`16 spill / 68B private`；
  三种page size均为`256 VGPR + 0 AGPR`、2 waves/SIMD。

page64 causal的16个spill均为长寿命地址值，并非Q/K/V fragment或O accumulator：16个
store仅在workgroup入口执行一次；每个persistent work item分别有3个初始化reload、2个
masked-tail入口reload和11个C-shuffle epilogue reload。重复执行的KV fast/tail循环体内
没有scratch指令。

验收使用MI308X/gfx942、1300MHz、`Hq=16,Hkv=1,Q=KV=32768,causal,page_size=32`，
每组10次预热、两轮各50个event样本，顺序为`4w -> 8w -> 8w -> 4w`。

| shape | batch | 4-wave | 8-wave | 关键资源 |
|---|---:|---:|---:|---|
| BF16 D128 | 1 | 23.983 ms / 183.381T | 30.054 ms / 146.340T | 8-wave：8 spill / 36B private |
| BF16 D128 | 4 | 97.725 ms / 180.017T | 116.349 ms / 151.202T | 4-wave：228 VGPR / 0 spill |
| BF16 D192 | 1 | 34.468 ms / 159.497T | 37.770 ms / 145.555T | 8-wave：23 spill / 96B private |
| BF16 D192 | 4 | 137.813 ms / 159.566T | 146.258 ms / 150.352T | 4-wave：10 spill / 44B private |

4-wave D192 batch=4修复前为`113.419T`，修复后提升`40.69%`。四组4/8-wave输出
均finite；relative-L2为D128 batch=1/4 `3.7173e-5/3.8095e-5`，D192 batch=1/4
`3.8662e-5/3.8175e-5`。8-wave BF16 D192短尾、causal及FP8聚焦用例通过。

回归：8-wave `39 passed`、4-wave `15 passed`、公开API `11 passed`。

### 2026-08-11 4-wave/8-wave基线

当时的8-wave参考使用per-tensor Q量化；4-wave接收同一份FP8 Q和
等值descale。两者在同一进程使用10套buffer、各10次预热和50个CUDA event样本，采用
位置平衡顺序；“相对8-wave”为25组配对时间比中位数的倒数。

| 场景 | 时钟 | 实现 | 调度 | 中位延迟 | Actual TFLOPS | 相对8-wave |
|---|---|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | **4-wave** | static | **671.343 us** | **403.441** | **1.224x** |
| 同上 | auto | 8-wave | persistent | 821.883 us | 329.544 | 1.000x |
| causal `Q=KV=32768` | auto | **4-wave** | static | **17918.507 us** | **306.809** | **1.059x** |
| 同上 | auto | 8-wave | persistent | 18872.409 us | 291.301 | 1.000x |
| causal `Q=KV=32768` | 1300MHz | **4-wave** | static | **18836.170 us** | **291.862** | **1.061x** |
| 同上 | 1300MHz | 8-wave | persistent | 19992.191 us | 274.985 | 1.000x |

non-causal 25/25组获胜；causal auto 24/25组获胜；causal 1300MHz 25/25组获胜。
causal按三角有效FLOPs计数；auto-DPM存在双态，因此同时保留1300MHz结果。

下表沿用page32性能基线；page64/page128已通过功能精度回归，尚未单独建立性能基线。

主shape的4-wave static/persistent同代码对照：

| 场景 | 时钟 | static | persistent | static收益 |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | 670.102 us / 404.188T | 839.263 us / 322.720T | **25.18%** |
| causal `Q=KV=32768` | 1300MHz | 18836.068 us / 291.863T | 19555.952 us / 281.119T | **3.85%** |

因此non-causal的400T主要依赖batch=1 static调度；persistent路径仍约323T。causal也受益于
static，但收益明显较小。两组static/persistent输出逐元素一致。8-wave始终使用persistent。

### 2026-08-10 4-wave性能矩阵

除H3使用3次预热和10样本外，其余formal结果均为10套buffer、10次预热和50样本中位数。
causal括号内为当次快档min。

| dtype | Dq/Dv | 调度 | 场景 | shape | 中位延迟 | Actual TFLOPS |
|---|---:|---|---|---|---:|---:|
| FP8 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 672.883 us | 402.518 |
| FP8 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 3065.972 us | 350.213 |
| FP8 | 192/128 | static | causal | `H16,Q=KV=32768` | 17802.427 us (13973.054) | 308.809 |
| BF16 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 1323.445 us | 204.653 |
| BF16 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 7475.309 us | 143.638 |
| BF16 | 192/128 | static | causal | `H16,Q=KV=32768` | 35486.656 us (25454.100) | 154.919 |
| FP8 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 2500.491 us | 343.530 |
| FP8 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 208.721 us | 257.219 |
| FP8 | 128/128 | static | causal | `H1,Q=KV=32768` | 1137.484 us | 241.654 |
| BF16 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 3422.933 us | 250.952 |
| BF16 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 268.201 us | 200.175 |
| BF16 | 128/128 | static | causal | `H1,Q=KV=32768` | 1650.006 us | 166.592 |
| FP8 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 86.369 ms | 331.755 |
| BF16 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 179.958 ms | 159.223 |

### 2026-08-10 精度矩阵

`diff`为`pyhip.calc_diff`对PyTorch reference；全部通过`rtol=atol=0.1`和finite检查。

| dtype | Dq/Dv | ragged最大diff | batch=4 diff | small causal diff | 主shape/额外验证 |
|---|---:|---:|---:|---:|---|
| FP8 | 192/128 | `2.8836e-4` | `3.4356e-4` | `1.7518e-4` | 主non-causal `3.6652e-4` |
| BF16 | 192/128 | `2.5129e-6` | `2.7224e-6` | `1.9344e-6` | 主non-causal `2.8093e-6` |
| FP8 | 128/128 | `2.6076e-4` | `3.4029e-4` | `1.7112e-4` | H3 finite |
| BF16 | 128/128 | `2.4619e-6` | `2.7061e-6` | `1.8679e-6` | H3 finite |

ragged覆盖`KV=3/13/23/53/83`，small causal为`Q=KV=256`。4-wave/8-wave同输入的
non-causal与causal relative-L2分别为`1.17e-4`和`1.12e-4`。

### 2026-08-10 specialization资源

- block为`BM128 x BN32 x 256 threads`；每个workgroup 4个wave；
- K使用LDS ping-pong，V直接进入fragment，output使用两个半块C-shuffle；
- online softmax使用raw-max、lazy rebase和loop-carried max/sum；
- FP8使用QK `VMEM1 -> MFMA2`、score MUL split11和FP8-only fast-math；
- FP8 D192当时为168 combined VGPR、16KB LDS、0 scratch，自然达到3 waves/SIMD；
- BF16 D128使用专用scheduler/HW-slot priority，D192使用独立scheduler；
- BF16/FP8共享pipeline时序骨架，dtype专属K搬运、scheduler、probability写回、V布局、
  epilogue地址和compile hint均封装在独立helper。

refactor前后fresh执行ISA逐条一致：

| specialization | ISA资源 | MFMA |
|---|---|---:|
| FP8 D192 | 168 VGPR-form / 16KB / 0 scratch | 80 |
| FP8 D128 | 153 VGPR-form / 16KB / 0 scratch | 64 |
| BF16 D192 | 250 VGPR-form / 25KB / 0 scratch | 160 |
| BF16 D128 | 214 VGPR-form / 17KB / 0 scratch | 128 |

FP8 D192 dynamic persistent kernel的执行ISA同样逐条一致。

## 优化里程碑（2026-08）

以下记录保留改变实现或建立关键反证的里程碑，性能数字均为当时版本结果。

### 2026-08-10：4-wave pipeline与C-shuffle

- **改动**：建立MMA32骨架；K走LDS ping-pong，V直读；接入paged ABI、GQA、ragged和causal；
  output改为两个64x128半块C-shuffle。
- **验证**：反转page table、跨页和ragged尺寸通过。
- **结果**：约`1838 -> 1465 -> 1008 -> 915 us`；保留双缓冲和半块C-shuffle。

### 2026-08-10：static dispatch与causal均衡

- **改动**：batch=1改用static grid；batch>1保留persistent；causal使用
  `(251 * tile + 251) % 256`映射。
- **验证**：non-causal、batch=4和long causal通过。
- **结果**：short调用约`54 -> 10 us`；causal约`17.9 -> 16.7 ms`；保留static/仿射路径。

### 2026-08-10：双K流水与priority

- **改动**：形成`K(i+2)`预取、softmax、`K(i+1)`写入、PV/barrier/K-read跨回边流水；
  FP8统一stage priority为`0/2`。
- **验证**：双K统一priority反相稳定；HW-slot priority回退。
- **结果**：主路径约876--880us；保留双K与统一priority。

### 2026-08-10：raw-max与softmax调度

- **改动**：先对raw score做max/shuffle，再用score scaling覆盖等待；FP8增加固定切分。
- **验证**：FP8/BF16数值不变；shuffle wait由约55降至10.7/17.2 cycles。
- **结果**：raw-max约提升3%，split8再提升1.33%；BF16不采用split8。

### 2026-08-10：BF16与H3

- **改动**：加入D128/D192 BF16 MMA、K/V layout、128-bit copy、LDS padding和D128 scheduler。
- **验证**：BF16 ragged/batch/causal及真实H3通过。
- **结果**：当时BF16 D128为250.952T，D192为204.653T，H3为159.223T。

### 2026-08-10：FP8自然3-wave

- **改动**：epilogue重建C-shuffle地址，将资源从176降至168 combined VGPR；固定gap2、
  score MUL split11和FP8-only fast-math。
- **验证**：16KB LDS、0 scratch、80 MFMA；最终ATT三槽`2+1`混合相90.36%。
- **结果**：当时non-causal为402.518T；4-wave相对当时8-wave为1.222x。

### 2026-08-11：8-wave参考复测

- **改动**：使用当时的8-wave per-tensor Q和多page-size实现，共享输入位置平衡复测；
  causal额外使用1300MHz固定频率。
- **验证**：page32/64/128 short reference通过；定频结束后恢复auto。
- **结果**：non-causal为1.224x；causal auto/1300MHz分别为1.059x/1.061x。

### 2026-08-11：代码refactor

- **改动**：保留共享pipeline，将BF16/FP8 K搬运、scheduler、probability、V布局、epilogue和
  compile hint封装为helper；删除恒真/恒假参数并统一命名。
- **验证**：4种static specialization和FP8 dynamic persistent执行ISA逐条一致；完整精度矩阵通过。
- **结果**：性能矩阵与重构前一致；共享时序骨架、独立dtype细节，不复制两份pipeline。

## 已否决实验

| 类别 | 关键证据 | 保留方案 |
|---|---|---|
| K copy 128-bit | 隐式`vmcnt(0)`，回退15.4%--15.5% | 64-bit K copy |
| 阶段/HW-slot | 循环slot回退2.76%；入口复制pipeline回退16.71% | 单pipeline、FP8统一priority |
| barrier/PV/K写 | 隔页barrier等待约124增至403 cycles；PV切分增scratch | 每页barrier、完整PV |
| page-table pair load | VGPR 168升至172，失去3-wave | 标量lookahead |
| 映射/priority | tile-major回退0.5%；反向priority约回退3% | head-major、`0/2` |
| 数值调度 | sum多链/非均匀gap无收益；显式rcp仅397.112T | 单链sum、gap2、fast-math |
| BF16实验 | split8中性；D192 shape峰值约210.7T | 原BF16 softmax、独立D192优化 |
| V自然padding | 冲突率降到0.37%，但`ds_read_b128`串行化使延迟升到749.39 us | 128-bit swizzled V LDS |
| V预分区/专属hint | 256 VGPR且411.86 us；精确DS/MFMA分组为413.57 us | 消费点重物化V地址、原调度hint |
| BF16平衡sum树 | ADD仍为15条且VGPR不变，但目标case稳定回退0.65%-0.73% | 保留LLVM线性归约及现有EXP/DS-write交织 |
| page id `readfirstlane` | 少10条整数VALU、VGPR 230降到226，但VMEM地址依赖串行化并回退10.9%-11.2% | 仅复用等价lookahead，不强制页号进SGPR |
| 完整V预分区 | 输出逐bit一致，但静态地址指令净增1条 | 循环内按page/block分区 |
| 单结果permlane32 | 未初始化old-dst导致NaN；该指令需要双结果/旧目标语义 | 保留ROCDL双结果intrinsic |

## ATT证据

- 当前gfx950 BF16 K16、slot-aware priority、native CVT、显式max树、page64
  lookahead复用、零`v_pk_add_f32`：
  `tests/flydsl/pa_4wave/ui_output_agent_54855_dispatch_22`；源码快照与当前文件SHA256
  一致，trace含20个完整wave并同时覆盖`p3/p1`和`p2/p0`路径；
- 统一`p2/p0`基线：`tests/flydsl/pa_4wave/ui_output_agent_44323_dispatch_22`。其
  高/低阶段累计周期比中位数为2.54；slot-aware的`p3/p1` wave降至2.04。两份trace
  均保持MFMA起步、K DS-read/MFMA交织及读取完成后发射K DMA；
- FP8 D192：`tests/flydsl/pa_4wave/att_fp8_d192_3wave/ui_output_agent_28524_dispatch_66`；
- BF16 D192：`tests/flydsl/pa_4wave/att_bf16_d192/ui_output_agent_32152_dispatch_13`；
- FP8主要stall/MFMA：MFMA 36.674、VALU 12.619、barrier 7.413、VMEM-load 6.397、
  LDS-wait 5.900；两条barrier约128/145 cycles。
