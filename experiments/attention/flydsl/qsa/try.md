# QSA direct跨query共享尝试：当前状态（2026-09-28）

本文件记录packed direct的填充T上限归因和“相邻query成对共享”原型的现状，完整过程见[opt.md](opt.md)末节。原型只在本地研究目录[qsa_direct_roofline_20260928_01](../../../../mytest/mydata/qsa_direct_roofline_20260928_01)，未纳入git，也**未集成到生产代码**（direct.py、_direct_packed.py、qsa.py及路由均未改）。

环境为hjbog-srdc-52物理GPU3（MI308X/gfx942/80CU，PTL Enabled/VECTOR,F8）。本机PyHIP `.venv`没有torch，因此使用system `/usr/bin/python3`（Torch2.12.0+rocm7.2.4、Triton3.7.1+rocm7.2.4、FlyDSL0.3.4.1）。**本机没有真实capture**，全部数据来自合成选择：每query选512块，相邻query随机换出δ块。

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

## 2. 原型文件（本地，未纳入git）

| 文件 | 说明 |
|---|---|
| [pair5.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/pair5.py) | 当前最佳候选：原生LDS交换，ballot跳过rescale（默认开启） |
| pair2.py / pair3.py / pair4.py | 前序版本，pair5仍从中导入helper；hybrid.py提供Triton planner与共用helper |
| [pair6.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/pair6.py) | K双缓冲（循环展开2），268VGPR，未采用 |
| [bench_formal.py](../../../../mytest/mydata/qsa_direct_roofline_20260928_01/bench_formal.py) | 正式对照脚本 |
| proto_run.py / phase1_ablate.py / debug_proto.py / debug_repeat.py | 探索计时、消融与正确性检查 |

## 3. 待验证的问题

1. **真实数据收益**：需要用TP2 L3/L47 capture测相邻query的共同块比例，并复测原型。前文另一环境中真实L3 union为2790µs，介于合成δ=100（2629µs）和δ=256（3459µs）之间，推测真实相邻共享率略低于82%，收益可能介于13%与16%之间，但尚未证实。
2. **环境**：需在PyHIP `.venv`上复测；本轮结束门禁失败，需在空闲GPU上同场重测。
3. **集成路径**：
   - planner需在union rebuild之后运行，因为它依赖dense_membership；
   - 需支持gated混合路由（ACTIVE/GROUPS）；
   - 共同块表为M×512 int32（M12000时24.6MB），需确认graph capture和workspace生命周期。
4. **路由**：需按新的direct时延重新标定1.7阈值；common比例低时要回退原direct（慢2.7%，另加planner）。
5. **边界**：rows=1尾pair、奇数common chunk、HK2/ragged、多请求packed、NaN尾均需回归；raw（非packed）路径不适用。
6. **planner开销**：57µs能否融合进union rebuild或pack。
7. **LDS与向量访存的带宽和端口关系**：在同一SIMD配置下直接测量。

## 4. 发现的问题

1. **inline-asm `ds_read_b128`是异步写回**，但LLVM认为asm结束时目的寄存器已定义。未使用的目的寄存器在`s_waitcnt`之前被VALU复用，迟到的LDS数据覆盖了buffer地址，产生只在某些寄存器分配下出现的确定性小误差。做法：asm读后立即wait并加`sched_barrier(0)`，不留dead目的寄存器，或改用原生LDS load。
2. **inline-asm `ds_write`直接读MFMA结果**时不会插入hazard等待，得到非确定性的脏数据。需先经过一次VALU（如乘scale或不透明的1.0）。
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
