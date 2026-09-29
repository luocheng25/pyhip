# QSA direct跨query共享尝试：当前状态（2026-09-28）

本文件记录packed direct的填充T上限归因和“相邻query成对共享”原型的现状，完整过程见[opt.md](opt.md)末节。采用的版本已整理到[pair/](pair/__init__.py)留作参考（见第2节）；探索过程和其它变体只在本地研究目录[qsa_direct_roofline_20260928_01](../../../../mytest/mydata/qsa_direct_roofline_20260928_01)。两者都**未集成到生产代码**（direct.py、_direct_packed.py、qsa.py及路由均未改）。

环境为hjbog-srdc-52物理GPU3（MI308X/gfx942/80CU，PTL Enabled/VECTOR,F8）。本机PyHIP `.venv`没有torch，因此使用system `/usr/bin/python3`（Torch2.12.0+rocm7.2.4、Triton3.7.1+rocm7.2.4、FlyDSL0.3.4.1）。合成数据为每query选512块、相邻query随机换出δ块；真实capture由用户提供的`a.gz`解压到[qsa_real_study_20260925/capture/inputs](../../../../mytest/mydata/qsa_real_study_20260925/capture/inputs)（8份，按manifest核对sha256一致），结果见1.4节。

## 1. 当前性能

### 1.1 正式结果（合成M12000/H12/HK1，D256）

每实现10个独立buffer、2warmup、128sample，每轮ABCD/DCBA交错，原`cudaPerf`；计时后逐buffer检查实际输出。pair含每次pack，pair+planner另含每次重建共同块表。填充T使用direct口径，union使用自身的膨胀填充量。

| δ | 相邻共享 | direct µs/填充T | pair µs/填充T | pair+planner µs/填充T | forced union µs/自身填充T | 配对比 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 93.9% | 2507.8/134.7 | 1994.6/169.3 | 2052.6/164.5 | 1876.2/206.4 | 0.8207 |
| 100 | 82.4% | 2509.6/134.6 | 2040.8/165.5 | 2099.4/160.9 | 2628.8/213.1 | 0.8386 |
| 256 | 61.5% | 2511.2/134.5 | 2124.8/159.0 | 2180.5/154.9 | 3458.5/216.5 | 0.8705 |
| 450 | 41.2% | 2509.3/134.6 | 2256.9/149.7 | 2310.0/146.2 | 3875.7/217.4 | 0.9223 |

- raw在[formal_20260928.json](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/formal_20260928.json)，共2048条，长尾全部保留。
- 与reference最大误差6.8e-4；pair与direct全部行最大差1.95e-3（容差0.02）。
- 入口和4次采样前门禁通过；**结束门禁gfx=17%，失败**。该值在本进程刚结束工作后读取，复查为0%，但仍按失败记录，未重跑。

### 1.2 探索口径（单buffer、64sample，非正式）

- δ=100的时间构成：共享块阶段（82%工作量）1593µs，私有块阶段约455µs，planner 57µs。
- H6/H3（δ=100）：direct 2452.5/2446.0µs，pair 1993.8/1971.4µs。
- common≈0（只走私有阶段）为2546µs，比direct慢约2.7%，另加planner 57µs。
- 资源：pair kernel 250VGPR/48SGPR/16KiB LDS，spill和private均为0。

### 1.3 上限归因

| 项目 | 数值 |
|---|---:|
| direct每query每32token读取K+V | 32KiB，对应64条16x16x16 MFMA（16FLOP/B，按16头填充） |
| 实测向量访存（global/buffer load，TA/TCP路径）上限 | 约9.0TB/s（约61B/clk/CU） |
| direct实际请求带宽 | 8.52TB/s（95%），对应填充上限约142–145T |
| 去掉访存（结果无效）/只保留访存 | 1403µs（241T）/2717µs |
| LDS ds_read_b128实测 | 约126B/clk/CU |

### 1.4 真实capture（正式，TP0四份×TP2/4/8，另TP1四份×TP2复核）

协议同1.1（10buffer/2warmup/128sample，每样本正反交替顺序）。入口、每场采样前和结束门禁全部通过（读前先让本进程空闲3s）。共三次运行，32768条raw。所有实现（含生产direct/union/qsa）与capture输出的最大差均为0.03125，在rtol=atol=0.02内；各buffer输出bitwise一致。

相邻pair共同块比例（按整chunk）：全部pair为71.0%～81.7%，TP2下被路由到direct的pair为65.2%～77.3%。

| 输入（TP0） | TP | direct→pair+planner µs（填充T） | 现路由union+direct→union+pair µs | 完整调用：现状→稀疏全走pair µs |
|---|---:|---:|---:|---:|
| L3 M12000 | 2 | 2513.6→2120.0（134.4→159.3） | 2405.9→2234.5（0.929） | 2932.5→2649.7（0.904） |
| L47 M12000 | 2 | 2509.6→2111.2（134.6→160.0） | 2496.8→2313.7（0.928） | 3027.3→2644.2（0.873） |
| L3 M11888 | 2 | 2462.3→2076.3（135.6→160.8） | 2333.8→2287.9（0.980） | 2864.1→2605.4（0.910） |
| L47 M11888 | 2 | 2463.9→2090.5（135.5→159.7） | 2383.3→2268.5（0.952） | 2911.6→2617.3（0.899） |
| 四份 | 4 | 0.841～0.847（162.1～164.6T） | 1.002～1.025 | 1.024～1.170 |
| 四份 | 8 | 0.830～0.843（163.5～166.6T） | 1.004～1.005 | 1.542～1.624 |

括号内为配对比中位。完整调用为手工复现的qsa()设备工作（recover/校验、union rebuild、dense、稀疏分支），与生产`qsa()`计时差≤0.3%。

- **kernel层面**：全部12场快15–17%（配对比0.830～0.851），填充T从134–138升到159–167。
- **保持现1.7路由、只替换direct**：TP2快2–7%；TP4因direct行很少（0–608）、planner成为净开销，慢至多2.5%；TP8没有direct行，慢约0.5%。
- **TP2稀疏行全走pair**：完整调用快9–13%，比现路由更好，说明TP2下pair在union组上也更快；TP4/8时union明显更快，全走pair会慢至多62%。
- 这条路由结论来自这两层的同一批数据，没有留出集；TP1四份选择与TP0相同、仅head不同，复核结果一致（全走pair配对比0.871～0.903）。
- raw在[real_formal_tp0_20260928.json](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/real_formal_tp0_20260928.json)、[real_formal_tp0_full_20260928.json](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/real_formal_tp0_full_20260928.json)、[real_formal_tp1_full_20260928.json](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/real_formal_tp1_full_20260928.json)；脚本为[bench_real.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/bench_real.py)。

## 2. 原型文件

整理后的参考实现在[pair/](pair/__init__.py)（纳入git，qsa()不使用）：

- [kernel.py](pair/kernel.py)：FlyDSL kernel `pair_qsa_bf16_d256`，即pair5的默认配置，去掉了计时消融和环境变量开关；
- [planner.py](pair/planner.py)：Triton planner（1 warp）和plan构建；
- `qsa_all_pair()`（[\_\_init\_\_.py](pair/__init__.py)）：稀疏行全走pair、构表只保留planner所需部分的完整调用，即第8节的`pair_all_lean`；
- [bench.py](pair/bench.py)：正确性检查和简易计时（单buffer，非正式协议），支持合成输入和真实capture。

核对（2026-09-29，GPU3）：ungated与gated两种特化实际加载的code object与pair5逐字节相同，planner输出和kernel输出逐位相同，见[记录](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/extract_check_20260929b/compare.txt)。bench.py复测：合成δ=100时kernel配对比0.836、完整调用0.869；真实L47 M12000 TP2分别为0.841和0.842，与第1、8节一致。

研究目录中的原始文件（本地，未纳入git）：

| 文件 | 说明 |
|---|---|
| [pair5.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/pair5.py) | 当前最佳候选：原生LDS交换，ballot跳过rescale（默认开启） |
| pair2.py / pair3.py / pair4.py | 前序版本，pair5仍从中导入helper；hybrid.py提供Triton planner与共用helper |
| [pair6.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/pair6.py) | K双缓冲（循环展开2），268VGPR，未采用 |
| [bench_formal.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/bench_formal.py) | 正式对照脚本 |
| proto_run.py / phase1_ablate.py / debug_proto.py / debug_repeat.py | 探索计时、消融与正确性检查 |

## 3. 待验证的问题

1. ~~真实数据收益~~：已用真实capture测量，见1.4节；完整QSA逐kernel见第8节。仍待验证：
   - 按每卡head数选路径（TP2全走pair、TP4/8保持生产）只在这四份capture上得出，需用留出输入验证；
   - TP4/8空gate的planner＋pair约12.5µs，不比生产pack＋direct的约15.4µs贵；问题在于TP4少量direct行上pair比direct慢（608行时慢19%）。
2. **环境**：需在PyHIP `.venv`上复测；本轮结束门禁失败，需在空闲GPU上同场重测。
3. **集成路径**：
   - planner需在union rebuild之后运行，因为它依赖dense_membership；
   - 需支持gated混合路由（ACTIVE/GROUPS）；
   - 共同块表为M×512 int32（M12000时24.6MB），需确认graph capture和workspace生命周期。
4. **路由**：需按新的direct时延重新标定1.7阈值；common比例低时要回退原direct（慢2.7%，另加planner）。
5. **边界**：rows=1尾pair、奇数common chunk、HK2/ragged、多请求packed、NaN尾均需回归；raw（非packed）路径不适用。
6. **planner开销**：57µs能否融合进union rebuild或pack。
7. ~~LDS与向量访存的带宽和端口关系~~：已实测，见第6节。

## 4. 发现的问题

1. **inline-asm `ds_read_b128`是异步写回**，但LLVM认为asm结束时目的寄存器已定义。未使用的目的寄存器在`s_waitcnt`之前被VALU复用，迟到的LDS数据覆盖了buffer地址，产生只在某些寄存器分配下出现的确定性小误差。做法：asm读后立即wait并加`sched_barrier(0)`，不留dead目的寄存器，或改用原生LDS load。
   - 生产代码审计（2026-09-29）：qsa/mha测试编译出的全部特化共363个kernel（另含gfx950仅编译的11个）。在实际ISA中检查了inline asm发出的ds_read、ds_bpermute、buffer/global load和s_load，结果如下：
     - 未发现在covering wait之前读写其目的寄存器的指令；
     - 未发现在等待之前就已经dead、可能被复用的目的寄存器；
     - QSA抽查的12个code object与dump出的ISA逐条一致；MHA未做该核对。FlyDSL的ISA dump是对同一模块单独再编译一次，可能与实际加载的code object不同：pair kernel同一源码6次编译中，code object全部相同，dump有4次多1条指令、调度不同。
   - 工具与结果见[inline_asm_audit_20260929](../../../../mytest/mydata/inline_asm_audit_20260929/summary.txt)。
2. **inline-asm `ds_write`直接读MFMA结果**时不会插入hazard等待，得到非确定性的脏数据。需先经过一次VALU（如乘scale或不透明的1.0）。
   - 同类问题：mha_pa_bf16_942.py和mha_pa_fp8_942.py中的`_max3`（inline `v_max3_f32`）在最后一个MFMA之后只隔4–10个wait state就读取其结果：
     - 按LLVM `GFX940_XDL_N_PassWriteVgprVALUMemExpReadWaitStates`（NumPasses+3），gfx942上`v_mfma_f32_32x32x8_bf16`与`32x32x16_fp8_fp8`是8 pass，需要11个；本机ROCm 7.2.4编译普通VALU读取时插入`s_nop 10`，见[探针](../../../../mytest/mydata/inline_asm_audit_20260929/mfma_hazard_probe.hip)；
     - LLVM只对VALU/VMEM/DS/EXP检查该hazard，不检查inline asm；
     - 不足11个的读取：bf16_942全部109个特化共1711处，fp8_942 8个特化中4个共64处。
   - 已修复（2026-09-29，提交a7688442）：两个文件在prologue的row max之前调用`_mfma_hazard_wait()`（`sched_barrier(0)`、`s_nop 10`、`sched_barrier(0)`，共11个wait state）。inline `v_max3`树和KV循环不变；循环里的max与QK之间隔着访存阶段，本来就够。
     - 复查：修复后bf16_942 109个、fp8_942 8个特化中，不足11个wait state的读取为0；49项功能测试通过；性能用例的acc与修复前完全相同。
     - 性能（[按执行顺序校正](../../../../mytest/mydata/mha_max_hazard_fix_20260929/ab_nop_order_split.md)）：grid BF16 0.997–1.004，FP8 0.996–1.006。persistent BF16约+1%；把`s_nop 10`换成`s_nop 0`、代码布局相同的对照同样+1.07%，所以来自prologue变长后KV循环位置的变化（D128 persistent两份循环分别后移12和24字节），不是等待本身。
     - ATT（[脚本](../../../../mytest/mydata/mha_max_hazard_fix_20260929/att_analyze.py)，SE0 CU1，第3次调用）：每个`s_nop 10`耗时44 cycle（11×4），期间该SIMD的MFMA空闲；它所在的prologue stage每个task只执行一次，约870 cycle，增加16～60 cycle，约占task的0.02%。SIMD MFMA利用率：persistent 84.69%→83.80%（`s_nop 0`对照83.62%），grid 84.75%→84.74%，FP8 D192 71.38%→70.96%。persistent的下降来自KV循环stage每次迭代多20～52 cycle，`s_nop 0`对照完全相同；FP8多出的约1.07k cycle在第一个barrier之前的启动访存段，barrier stage合计反而少162 cycle。
     - 放弃的方案：把max树改成编译器可见的`llvm.maxnum`。hazard同样消除，但扰动了寄存器分配：FP8 D192的KV循环多了2条`v_add`，VGPR 220→224，慢1.1–1.6%。
     - 测量注意：`test_mha_pa.measure()`每轮第一个运行的候选读到的输入已被挤出cache，慢约1.2%；多于两个候选时比值有偏差。两个候选时取正反两种顺序的平均即可抵消（A/A为1.0000）。
   - QSA使用编译器可见的maxnum，不受影响。
3. **FlyDSL磁盘缓存不跟踪jit函数内读取的`os.environ`**，调试开关会复用旧kernel。开关必须写成模块级常量，或设置`FLYDSL_RUNTIME_ENABLE_CACHE=0`。
4. **循环头waitcnt合并**：prologue与循环体的load发射顺序不一致时，循环头会保守地等到`vmcnt(0)`，V的延迟被完全暴露。做法：prologue按循环相同顺序发射（K(1)先于V(0)），并用`sched_barrier`固定顺序。
5. **分支内的VMEM**（块缓存刷新）会在join处产生`vmcnt(0)`。改为无条件、提前一迭代加载。
6. 含inline-asm LDS和`s_barrier`的动态`if`，其后的值出错；很可能与第1条是同一race。
7. FlyDSL循环分析：
   - 循环前已有的同名变量若dtype/长度不同，会被当作carried变量而报错，需要改名；
   - 在`fx.const_expr`分支中定义的变量不会被循环分析识别。
8. K双缓冲使VGPR增至268，occupancy从2降到1，共享块阶段变慢到2330µs。
9. 启用packed FP32慢约2%，显式sched_group交织慢2–5%，无条件rescale慢3%。
10. **共享块阶段MFMA利用率约57%**，限制来自2wave/SIMD、每chunk一次的barrier耦合和访存延迟。16KiB LDS/CTA已是4CTA/CU的上限，再增加VGPR或LDS都会降低occupancy。
11. 2-wave workgroup的两个wave实测总在不同SIMD上（960个均如此，多数跨LDS端口），所以不是“同SIMD锁步”导致的低重叠。
12. rescale的代价主要来自ballot分支切分调度区域，而不是乘法本身：去掉rescale省约79µs，而lazy rescale（τ=4/8）只省约13µs（0.6%）。
13. 按K分片提前发射K(t+2)（[pair7.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/pair7.py)）慢了0.7%，未采用。
14. planner改为1 warp：57→35µs，输出一致，已采用。

## 5. 仅计时消融（δ=100，共享块阶段）

| 条件 | µs |
|---|---:|
| 当前（ballot跳过rescale） | 1592 |
| 去掉K载入 | 1448 |
| 去掉V载入 | 1511 |
| 去掉K/V载入 | 1366 |
| 无条件rescale版 | 1640 |
| 无条件rescale版，去掉barrier | 1574 |
| 无条件rescale版，去掉rescale | 1495 |
| 无条件rescale版，去掉载入/barrier/rescale | 1260 |

## 6. LDS与global load带宽实测（GPU3，1.83–1.85GHz）

使用[lds_bw_kernel.cpp](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/lds_bw_kernel.cpp)和[lds_bw.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/lds_bw.py)测量：每CU一个1024线程workgroup（64KiB LDS）；按HW_ID只让指定SIMD上的前N个wave工作；以s_memtime计周期，并用s_memrealtime（100MHz）换算时钟；80个CU、每SIMD 4 wave的放置均已核对。表中每格为每个活跃SIMD上1/2/4 wave时的B/clk/CU。

| 指令 | SIMD0 | SIMD0+1 | SIMD2+3 | SIMD0+2 | 全部4个 |
|---|---|---|---|---|---|
| ds_read_b32 | 17/33/51 | 29/57/58 | – | 33/66/102 | 58/117/116 |
| ds_read_b64 | 28/55/48 | 55/64/64 | 55/64/64 | 55/111/109 | 111/120/124 |
| ds_read_b128 | 39/64/59 | 64/64/64 | 64/64/64 | 79/128/122 | 128/128/128 |
| ds_write_b128 | 27/35/37 | 34/37/38 | 38/39/39 | 55/71/74 | 67/74/76 |
| global_load_dwordx4（L1命中） | 30/62/64 | 43/64/64 | 43/64/64 | 43/64/64 | 64/64/64 |
| global_load_dwordx4（L2命中） | 12/24/48 | 21/42/64 | – | 21/42/64 | 48/64/64 |

- **LDS峰值为128B/clk/CU**（32bank×4B），墙钟18.6TB/s，与80CU×1.845GHz×128B≈18.9TB/s一致。
- **LDS分成两个各64B/clk的端口**：SIMD0/1共用一个，SIMD2/3共用另一个。
  - 单个SIMD最多64B/clk；SIMD0+1或SIMD2+3合计仍为64；SIMD0+2、0+3、1+2、1+3均可达128。
  - 三个SIMD（1+2+3）时，按SIMD分别为64/32/32。
  - 因此饱和需要两个端口上都有活跃SIMD，每个SIMD至少2个wave做b128。b32受指令发射率限制，四个SIMD也只有约117。
- ds_write_b128每端口约37B/clk，全CU最多约76B/clk（约11.2TB/s）。
- **global load（TA/TCP）全CU只有一个64B/clk通路**：单个SIMD带4 wave即可饱和，约9.3–9.4TB/s，只有LDS的一半。packed direct的K/V走这条通路，不经过LDS。

## 7. pair设计的上限与剩余差距（真实L3 M12000 TP2）

正式复测（planner 1 warp）：pair+planner 2096.9µs（161.1T），比direct 2513.4µs快16.3%。四份TP2的配对比为0.834～0.841，稀疏行全走pair时完整调用配对比为0.867～0.902；raw见[real_formal_tp0_planner1_20260928.json](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/real_formal_tp0_planner1_20260928.json)。

探索口径分解见[breakdown_real_20260928.jsonl](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/breakdown_real_20260928.jsonl)。

| 项 | L3 M12000 | L47 M12000 |
|---|---:|---:|
| 共享块占比 | 71.0% | 73.3% |
| pair kernel（含pack） | 2073µs | 2061µs |
| 只跑共享阶段 / 只跑私有阶段 | 1405 / 897µs | 1443 / 862µs |
| 共享阶段去掉K/V载入 | 1218µs（−13%） | 1264µs（−12%） |
| 共享阶段去掉barrier / 去掉rescale | 1350 / 1326µs | 1395 / 1373µs |
| 共享阶段去掉载入、barrier、rescale | 1120µs（214T，71% MFMA） | 1161µs |

- **资源利用率（L3，整kernel 2062µs）**：请求字节为13.61GB（共享16KiB、私有32KiB/query-chunk），即**6.6TB/s，是global load峰值的71%**；MFMA为163.8T，是峰值301.5T的54%、纯计算结构上限241T的68%。
- **设计上限**（两阶段完全重叠、无任何停顿）：
  - MFMA侧：共享阶段纯计算1120µs＋私有阶段计算406µs，共1526µs；
  - 访存侧：13.61GB÷9.33TB/s＝1459µs；
  - 故上限约1526µs＋35µs planner，约216T。当前2097µs，达到该上限的74%。
- **差距来源**：共享阶段的访存停顿约187µs、barrier约55µs、rescale分支约79µs；私有阶段单独运行只有6.8TB/s，没有流化的一次性开销占比高。
- **为什么在该设计内难以消除**：
  - VGPR已用250/256，LDS 4CTA×16KiB＝64/64KiB，每SIMD只能放2个wave，在途访存只够提前约半个chunk；
  - 多一份K缓冲（+32VGPR）就触发occupancy减半，共享阶段慢了42%（pair6）；LDS被交换区占满，没有暂存K/V的空间；
  - gfx942没有分离式barrier，每chunk的交换必须全CTA同步；
  - rescale分支改成无条件乘法更慢，lazy τ只有−0.6%。
- **结论**：在当前寄存器/LDS预算下，pair kernel实际上限约160–167填充T，只剩几个百分点的调优空间。要到200T需要另一种设计。

## 8. 完整QSA：TP2/4/8端到端与逐kernel（新kernel `pair_qsa_bf16_d256`）

pair5的kernel改名为`pair_qsa_bf16_d256`，以便trace与生产direct区分（代码未变）。每个capture×TP比较5种完整调用（[qsa_variants.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/qsa_variants.py)）：

- `qsa`：生产`qsa()`，源码未改；
- `replica`：手工复现qsa()的设备工作（recover/校验、union rebuild、dense、union、pack+direct），与qsa逐位一致，计时差≤0.2%；
- `pair_routed`：保持1.7路由，只把direct换成planner+pair；
- `pair_all`：稀疏行全走pair，union rebuild照常；
- `pair_all_lean`：稀疏行全走pair，构表只保留planner需要的membership清零+scatter，跳过union专用的compact/score_masks/order_tasks。

计时：[bench_qsa_tp.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/bench_qsa_tp.py)，协议同1.1（10buffer/2warmup/128sample，正反交替）。14次门禁全部通过，7680条raw；每个variant的输出在计时前后都与capture比对（最大差0.03125，在.02容差内），并与buffer0逐位一致；pair_all_lean与pair_all逐位一致。

逐kernel：[trace_qsa_tp.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/trace_qsa_tp.py)在rocprofv3 kernel trace下运行，每场每variant 10个buffer各trace一次；每个call前后各插一个spin kernel作为边界（前一个约2.2ms，确保整个call在第一个kernel开始前已全部入队）。600个call全部对上，没有未知kernel，call内kernel间无空隙。由[parse_qsa_trace.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/parse_qsa_trace.py)统计每kernel每call的中位数。有两个call受干扰（L3 M11888 TP4 pair_routed的union 7185µs、L47 M11888 TP2 lean的dense 773µs，同kernel其余call为1656/180µs），因此用中位数，raw未删。完整表见[qsa_tp_report_20260928.md](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/qsa_tp_report_20260928.md)。

### 8.1 端到端（cudaPerf中位µs，括号为相对qsa的配对比中位）

| 输入 | TP | dense/union/direct行 | qsa | pair_routed | pair_all | pair_all_lean |
|---|---:|---|---:|---:|---:|---:|
| L3 M11888 | 2 | 2051/7909/1928 | 2863.3 | 2811.5（0.982） | 2580.1（0.901） | **2509.4（0.877）** |
| L3 M12000 | 2 | 2051/3929/6020 | 2932.7 | 2751.4（0.938） | 2628.2（0.896） | **2561.5（0.874）** |
| L47 M11888 | 2 | 2051/5599/4238 | 2907.1 | 2782.5（0.957） | 2593.4（0.892） | **2529.0（0.869）** |
| L47 M12000 | 2 | 2051/4269/5680 | 3026.3 | 2831.6（0.936） | 2620.8（0.867） | **2551.4（0.843）** |
| L3 M11888 | 4 | 2051/9837/0 | 2116.6 | 2112.7（0.998） | 2449.0（1.157） | 2375.1（1.122） |
| L3 M12000 | 4 | 2051/9341/608 | 2482.2 | 2526.1（1.018） | 2519.9（1.015） | 2442.6（0.984） |
| L47 M11888 | 4 | 2051/9613/224 | 2391.5 | 2408.6（1.007） | 2488.9（1.040） | 2414.6（1.009） |
| L47 M12000 | 4 | 2051/9933/16 | 2464.1 | 2467.6（1.001） | 2511.5（1.019） | 2434.5（0.988） |
| L3 M11888 | 8 | 2051/9837/0 | 1518.7 | 1514.3（0.997） | 2436.0（1.604） | 2352.7（1.548） |
| L3 M12000 | 8 | 2051/9949/0 | 1639.8 | 1639.2（1.000） | 2514.9（1.534） | 2421.6（1.477） |
| L47 M11888 | 8 | 2051/9837/0 | 1624.3 | 1621.8（0.998） | 2481.3（1.527） | 2387.1（1.470） |
| L47 M12000 | 8 | 2051/9949/0 | 1608.7 | 1608.6（1.000） | 2500.9（1.555） | 2412.5（1.501） |

### 8.2 逐kernel（每call中位µs，L3 M12000；其它三份见完整表）

| kernel | TP2 qsa | TP2 lean | TP4 qsa | TP4 routed | TP4 lean | TP8 qsa | TP8 lean |
|---|---:|---:|---:|---:|---:|---:|---:|
| recover | 219.4 | 219.7 | 220.0 | 220.2 | 220.2 | 220.3 | 219.9 |
| check_errors＋assert | 12.1 | 12.1 | 11.7 | 12.0 | 12.0 | 12.0 | 12.0 |
| membership_zero＋scatter | 46.6 | 45.7 | 45.5 | 45.4 | 45.5 | 43.8 | 47.0 |
| compact＋score_masks＋order_tasks | 66.7 | – | 83.1 | 84.6 | – | 100.7 | – |
| dense | 179.7 | 179.0 | 107.4 | 107.1 | 106.8 | 106.2 | 105.1 |
| union | 849.8 | – | 1841.1 | 1842.1 | – | 1154.2 | – |
| pack | 10.5 | 9.5 | 8.7 | 8.5 | 9.8 | 4.4 | 9.7 |
| direct | 1552.8 | – | 176.0 | – | – | 11.0 | – |
| planner | – | 33.8 | – | 10.3 | 33.7 | – | 33.4 |
| pair | – | 2063.3 | – | 209.5 | 2022.6 | – | 2004.0 |
| kernel合计 | 2938.1 | 2563.5 | 2493.6 | 2541.3 | 2450.2 | 1652.7 | 2432.1 |

四份capture汇总：

- **与新kernel无关的固定项**：recover 218–221µs，check_errors＋assert约12µs，dense在TP2约180µs、TP4/8约106µs。TP2改用lean后，pair约占80%，recover升为第二大项（约9%），dense约7%。
- **构表**：membership清零＋scatter 43–47µs；compact/score_masks/order_tasks在TP2约67–77µs、TP4约78–85µs、TP8约88–101µs，lean省下的正是这部分。
- **pair全部稀疏行**：TP2 2013–2063µs、TP4 1966–2023µs、TP8 1943–2004µs，几乎不随TP变化；planner 31–34µs。
- **union**：TP8只需1046–1154µs就能处理全部稀疏行。

### 8.3 结论

- **TP2**：lean全走pair使整个QSA快12.3%–15.7%（省354–475µs），四份一致。按行摊算，union选中的那些行，pair约0.18µs/行，union约0.22µs/行，pair在这些行上也更快，所以不应保留union。
- **TP4**：没有稳定收益。
  - pair_routed为0.998–1.018：被路由到direct的行很少（0–608），L3 M12000的608行上pair 209.5µs，比direct 176.0µs慢19%。
  - 推测原因：行数少时处于单任务延迟主导区，pair每个任务2个wave、需要LDS交换和barrier，关键路径比direct长。这些行的共同块比例仍有61%，所以不是共享不足。该推测未验证。
  - lean为0.984–1.122：union行上两者每行代价基本持平（pair相对union −1.5%～+3%），但L3 M11888上pair贵19%。
- **TP8**：没有direct行，pair_routed与生产相同（空gate的planner＋pair约12.5µs，生产的pack＋direct约15.4µs）；全走pair慢47%–60%。
- **原因**：pair时间几乎与TP无关。它受每行K/V全局载入限制，每行需要载入的K/V量不随TP减少，而union在16行tile内复用K/V，因此head越少union越有优势。
- **集成建议**：按每卡head数（host已知，图安全）选路径：TP2全部稀疏行走pair（lean构表）；TP4/TP8保持生产union＋direct，不启动pair。
- **后续可做**：
  - recover约220µs（TP2改用lean后占约9%）是最大的非attention项；
  - TP4上union与pair每行代价接近，按tile选union/pair还要保留union构表，预计收益很小（未测）；
  - 少行时pair的延迟问题。
