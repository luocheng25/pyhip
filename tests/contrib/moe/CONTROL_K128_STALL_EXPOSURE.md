# MFMA 气泡细分暴露分析

所有原因行都是互斥的owner归因，用来闭合当前时间账本，不是可以直接相加的加速预算。
Physical `any` / `all`计数是不可加的见证量，joint state才表示精确的联合状态。

本文的`cycle/N`统一按一个wave完成一个N块的工作量归一：每N固定执行192条MFMA，每条按
16-cycle execution window计，因此一个等价wave-N包含3072个MFMA execution cycles。Physical
steady窗口是resident waves生命周期的交集，不一定由完整N块组成，所以physical表用
`MFMA execution mass / 3072`得到等价wave-N数，不能直接拿完整N块数作分母。
表内`cycle/N`保留两位小数，逐行相加可能与未舍入总计相差`0.01 cycle/N`；JSON中的原始周期严格闭合。

## 表格口径

### “两wave同因/总时间”是什么意思

这一列只检查physical MFMA-idle tick。若两条resident wave在同一个4-cycle tick里具有完全相同的
互斥原因，就把这4 cycles记到该原因，最后除以整个physical steady时间。例如Control的
`vmem_issue_stall`同因值为9.8934%，表示这部分总时间里两条wave同时无法发射VMEM请求，没有另一条
wave能用MFMA遮盖它。

它是严格的同因见证量，不是额外周期，**不能再与owner列相加**。如果一个tick里wave0是
`vmem_issue_stall`、wave1是`ds_wait_stall`，owner账本会把4 cycles拆成2+2 cycles分别归因，但该tick
不会进入任何“两wave同因”行。

### 为什么五类不能直接加回MFMA busy

记`U_MFMA`为MFMA union busy、`E_5`为五类owner之和、`R`为其余互斥residual。表中恒等式
`U_MFMA + E_5 + R = 100%`仅用于闭合当前trace。

它不表示“删除某类后，MFMA busy会增加相同百分点”。一次调度改动会同时改变请求发射、未来wait、
tail、两wave相位和总分母；而且带阻塞标签的wave不保证此刻已有ready MFMA可填。例如某tick分别是
VMEM发射阻塞和DS等待阻塞，即使把VMEM提前，DS等待仍可能占满该tick，并不会凭空产生4 cycles的
MFMA union execution。

实际Control到9aa595d也否证了线性加法：VMEM issue/wait合计下降7.846个百分点，但MFMA union busy
只增加4.819个百分点；其余周期转移为DS、tail和residual。因而优化必须同时检查physical union busy、
原因转移和墙钟时间。

## Control K128

### 单wave视角

单wave平均为`7938.53 cycle/N`，MFMA busy为`38.70%`。

| 单wave状态 | cycle/N | 占总时间 | 占MFMA-idle |
|---|---:|---:|---:|
| **`MFMA execution`（MFMA执行）** | **3072.00** | **38.70%** | - |
| `vmem_issue_stall`（VMEM发射阻塞） | 1143.83 | 14.41% | 23.50% |
| `vmem_wait_stall`（VMEM等待阻塞） | 98.23 | 1.24% | 2.02% |
| `ds_issue_stall`（DS发射阻塞） | 237.45 | 2.99% | 4.88% |
| `ds_wait_stall`（DS等待阻塞） | 904.95 | 11.40% | 18.60% |
| `structural_tail`（结构尾部） | 1242.36 | 15.65% | 25.53% |
| `mixed_wait_stall`（混合等待阻塞） | 68.86 | 0.87% | 1.41% |
| `mfma_issue_unavailable`（MFMA发射暂不可用） | 1074.58 | 13.54% | 22.08% |
| `other_dependency_stall`（其他依赖阻塞） | 0.32 | 0.00% | 0.01% |
| `normal_issue_exposure`（正常发射暴露） | 80.11 | 1.01% | 1.65% |
| `scheduler_ready`（调度器就绪残差） | 15.84 | 0.20% | 0.33% |
| `other_exposure`（其他暴露） | 0.00 | 0.00% | 0.00% |

### MFMA发射暂不可用的程序证明

- 共检查`534,528`个动态相邻MFMA对；背靠背accumulator RAW为`0`对。
- Accumulator复用距离min/P50/P99为`16/16/16`条MFMA；successful-issue距离为
  `256/256/2488 cycles`。
- Physical SIMD上的MFMA successful-issue间隔min/P50/P99为`16/16/296 cycles`；小于16 cycles的
  间隔为`0`个。
- A/source0：`534,528/534,528`条VMEM producer边被计数器顺序有效的`s_waitcnt vmcnt`覆盖。
- B/source1：`534,528/534,528`条DS producer边被计数器顺序有效的`s_waitcnt lgkmcnt`覆盖。
- `mfma_issue_unavailable`共`2,991,620`个wave-cycles；peer MFMA execution-window覆盖率为`100.00%`，
  与peer MFMA同tick成功发射的比例为`24.55%`，没有peer execution覆盖的周期为`0`。
- 原始`stall:MFMA`共`8,512,120 cycles`：仅本wave窗口`5,353,028`，仅peer窗口`3,159,092`，
  两者同时`0`，两者都不在`0`。

因此，本桶可以排除背靠背accumulator RAW和未被wait覆盖的A/B读取依赖。它表示本wave的MFMA/VALU
发射类别暂不可用，而另一条resident wave占用了同一SIMD的16-cycle MFMA硬件时隙。ATT中的4-cycle
`issue_cost`是trace记录成本，不是物理MFMA initiation interval。Priority可以改变wave仲裁，但只有当
这种暴露进一步造成physical union空槽时才是调度缺陷；已被peer覆盖的single-wave stall不是可直接相加
的性能差距。

### Physical union视角

Physical union合并同一SIMD上的2条resident wave。按实际MFMA execution mass归一后，等价wave-N数为
`1310.34375`，总时间为`4280.01 cycle/N = 3072.00 union execution + 1208.01 idle`。

| Physical owner状态 | cycle/N | 占总时间 | 占MFMA-idle | 两wave同因/总时间 |
|---|---:|---:|---:|---:|
| **`MFMA union execution`（MFMA并集执行）** | **3072.00** | **71.78%** | - | - |
| `vmem_issue_stall`（VMEM发射阻塞） | 557.92 | 13.04% | 46.19% | 9.8934% |
| `vmem_wait_stall`（VMEM等待阻塞） | 29.37 | 0.69% | 2.43% | 0.0019% |
| `ds_issue_stall`（DS发射阻塞） | 85.35 | 1.99% | 7.07% | 0.5684% |
| `ds_wait_stall`（DS等待阻塞） | 217.59 | 5.08% | 18.01% | 1.3912% |
| `structural_tail`（结构尾部） | 280.34 | 6.55% | 23.21% | 1.9217% |
| `mixed_wait_stall`（混合等待阻塞） | 13.78 | 0.32% | 1.14% | 0.0016% |
| `mfma_issue_unavailable`（MFMA发射暂不可用） | 0.00 | 0.00% | 0.00% | 0.0000% |
| `other_dependency_stall`（其他依赖阻塞） | 0.02 | 0.0005% | 0.0016% | 0.0000% |
| `normal_issue_exposure`（正常发射暴露） | 20.10 | 0.47% | 1.66% | 0.0014% |
| `scheduler_ready`（调度器就绪残差） | 3.55 | 0.08% | 0.29% | 0.0000% |
| `other_exposure`（其他暴露） | 0.00 | 0.00% | 0.00% | 0.0000% |

Oracle目标集合为`ds_issue_stall, ds_wait_stall, structural_tail, vmem_issue_stall`。若假设任一目标
标签都能立即填入MFMA，乐观可恢复上界是steady总时间的`28.19%`，剩余`0.03%`。该oracle会凭空
假设ready MFMA，只用于说明线性加法过于乐观，不是性能预测。严格VMEM-wait见证量为`0.0019%`。

## 9aa595d

### 单wave视角

单wave平均为`7567.66 cycle/N`，MFMA busy为`40.59%`。

| 单wave状态 | cycle/N | 占总时间 | 占MFMA-idle |
|---|---:|---:|---:|
| **`MFMA execution`（MFMA执行）** | **3072.00** | **40.59%** | - |
| `vmem_issue_stall`（VMEM发射阻塞） | 573.33 | 7.58% | 12.75% |
| `vmem_wait_stall`（VMEM等待阻塞） | 52.90 | 0.70% | 1.18% |
| `ds_issue_stall`（DS发射阻塞） | 235.90 | 3.12% | 5.25% |
| `ds_wait_stall`（DS等待阻塞） | 921.75 | 12.18% | 20.50% |
| `structural_tail`（结构尾部） | 1398.55 | 18.48% | 31.11% |
| `mixed_wait_stall`（混合等待阻塞） | 94.17 | 1.24% | 2.09% |
| `mfma_issue_unavailable`（MFMA发射暂不可用） | 983.20 | 12.99% | 21.87% |
| `other_dependency_stall`（其他依赖阻塞） | 57.54 | 0.76% | 1.28% |
| `normal_issue_exposure`（正常发射暴露） | 145.75 | 1.93% | 3.24% |
| `scheduler_ready`（调度器就绪残差） | 32.56 | 0.43% | 0.72% |
| `other_exposure`（其他暴露） | 0.00 | 0.00% | 0.00% |

### MFMA发射暂不可用的程序证明

- 共检查`534,528`个动态相邻MFMA对；背靠背accumulator RAW为`0`对。
- Accumulator复用距离min/P50/P99为`15/16/16`条MFMA；successful-issue距离为
  `256/256/2304 cycles`。
- Physical SIMD上的MFMA successful-issue间隔min/P50/P99为`16/16/188 cycles`；小于16 cycles的
  间隔为`0`个。
- A/source0：`534,528/534,528`条VMEM producer边被计数器顺序有效的`s_waitcnt vmcnt`覆盖。
- B/source1：`534,528/534,528`条DS producer边被计数器顺序有效的`s_waitcnt lgkmcnt`覆盖。
- `mfma_issue_unavailable`共`2,737,228`个wave-cycles；peer MFMA execution-window覆盖率为`100.00%`，
  与peer MFMA同tick成功发射的比例为`24.71%`，没有peer execution覆盖的周期为`0`。
- 原始`stall:MFMA`共`8,546,024 cycles`：仅本wave窗口`5,623,060`，仅peer窗口`2,922,964`，
  两者同时`0`，两者都不在`0`。

同理，本桶可以排除背靠背accumulator RAW和未被wait覆盖的A/B读取依赖。Priority能改变wave仲裁，
但已被peer覆盖的single-wave stall不是可直接相加的性能差距。

### Physical union视角

Physical union合并同一SIMD上的2条resident wave。等价wave-N数为`1326.05729`，总时间为
`4010.73 cycle/N = 3072.00 union execution + 938.73 idle`。

| Physical owner状态 | cycle/N | 占总时间 | 占MFMA-idle | 两wave同因/总时间 |
|---|---:|---:|---:|---:|
| **`MFMA union execution`（MFMA并集执行）** | **3072.00** | **76.59%** | - | - |
| `vmem_issue_stall`（VMEM发射阻塞） | 220.97 | 5.51% | 23.54% | 3.4205% |
| `vmem_wait_stall`（VMEM等待阻塞） | 14.68 | 0.37% | 1.56% | 0.0000% |
| `ds_issue_stall`（DS发射阻塞） | 82.77 | 2.06% | 8.82% | 0.5171% |
| `ds_wait_stall`（DS等待阻塞） | 226.46 | 5.65% | 24.12% | 1.5233% |
| `structural_tail`（结构尾部） | 309.63 | 7.72% | 32.98% | 2.7796% |
| `mixed_wait_stall`（混合等待阻塞） | 21.44 | 0.53% | 2.28% | 0.0001% |
| `mfma_issue_unavailable`（MFMA发射暂不可用） | 0.00 | 0.00% | 0.00% | 0.0000% |
| `other_dependency_stall`（其他依赖阻塞） | 14.25 | 0.36% | 1.52% | 0.0005% |
| `normal_issue_exposure`（正常发射暴露） | 39.43 | 0.98% | 4.20% | 0.0076% |
| `scheduler_ready`（调度器就绪残差） | 9.11 | 0.23% | 0.97% | 0.0003% |
| `other_exposure`（其他暴露） | 0.00 | 0.00% | 0.00% | 0.0000% |

Oracle目标集合相同。乐观可恢复上界是steady总时间的`23.35%`，剩余`0.06%`；它不是性能预测。
严格VMEM-wait见证量为`0.0000%`。

## Control与9aa595d实测对照

| Physical owner状态 | 变化（9aa - Control，百分点） |
|---|---:|
| `vmem_issue_stall`（VMEM发射阻塞） | -7.526 |
| `vmem_wait_stall`（VMEM等待阻塞） | -0.320 |
| `ds_issue_stall`（DS发射阻塞） | +0.070 |
| `ds_wait_stall`（DS等待阻塞） | +0.563 |
| `structural_tail`（结构尾部） | +1.170 |
| residual（其余互斥原因合计） | +1.225 |
| **`MFMA union execution`** | **+4.819** |

VMEM issue减少7.526个百分点，并没有一比一变成MFMA busy；约2.7个百分点转移到了其他暴露类别。
所以某一类下降只说明当前schedule的暴露发生了转移，最终是否优化成功必须同时检查physical union busy、
原因转移和墙钟时间。

## 从暴露账本推导的优化思路

### 总体判断

当前最值得优化的不是单wave最大的某一行，而是**两条resident wave同时暴露、且已被同工作量消融证明
可以改变的physical空槽**。Control最强的这类信号是`vmem_issue_stall`：owner占总时间13.04%，两wave
同因占9.8934%；9aa595d分别降到5.51%和3.4205%，同时MFMA union busy从71.78%升到76.59%。因此
首要目标是恢复两wave的VMEM反相，而不是继续压缩单wave内的MFMA/VMEM间隔。

建议按以下顺序推进：

| 优先级 | 目标 | 当前证据 | 候选改动 | 主要验收指标 |
|---|---|---|---|---|
| P0 | 两wave VMEM错相 | Control VMEM issue owner 13.04%，同因9.8934%；9aa已证明可降 | 恢复/引入slot-aware priority，或按hardware slot生成非对称VMEM相位 | VMEM同因和owner同时下降，MFMA union busy与ABBA墙钟同时改善 |
| P1 | 保持消费距离，只移动请求发射相位 | VMEM wait仅0.69%，真正主项是VMEM issue 13.04% | 旋转8组`VMEM1 -> MFMA4`的位置，避免两slot同tick灌入VMEM队列 | `vmem_issue_stall`下降且`vmem_wait_stall`不反弹 |
| P2 | 隐藏重新暴露的DS wait | Control DS issue/wait为1.99%/5.08%；9aa后DS wait升到5.65% | 拆分头部`DSRD8`、前移DS read或增加DS read到consumer的距离 | DS wait下降，LDS指令数/资源档位不变，VMEM收益不丢失 |
| P3 | 缩短structural tail | Control tail owner 6.55%，两wave同因1.9217%；局部重排不能生成下一N的MFMA | 分片退休accumulator，使`postprocess/CShuffle(N)`与`MFMA(N+1)`重叠 | tail与总`cycle/N`下降，仍保持2 waves/SIMD和全shape正确 |
| 暂不做 | 追逐single-wave MFMA stall或普通VALU | `mfma_issue_unavailable`在physical中为0；normal issue仅0.47% | 不增加accumulator链，不增加wait，不优先搬普通VALU | 防止优化被已遮盖的single-wave统计误导 |

### P0：先恢复physical VMEM反相

最低风险候选是复现9aa595d的slot priority行为。当前K128 physical sched-group的静态顺序为
`DSRD8 -> 8 x (VMEM1 -> MFMA4) -> MFMA32`；同一静态顺序会施加到两个resident wave，容易让两条wave
在相同相位同时提交VMEM。Priority本身不增加MFMA或VMEM带宽，但可以改变哪条wave先获得发射机会，
进而打破长期同相。

当前代码中K128会令`use_physical_sched_group=True`，而`hw_wave_slot`读取、`enter_read_write_stage()`和
`enter_compute_stage()`里的现有priority逻辑都受`not use_physical_sched_group`保护，因此K128实际上绕开了
这条slot-aware路径。P0实现应把hardware slot读取和stage边界priority安全地扩展到K128专用路径，不能只
切换`_PHYSICAL_N256_USE_SETPRIO`常量。

建议先做两个独立实验，避免一次混入多个变量：

1. **slot-aware priority实验。** 在K128路径的read/write与compute stage边界复用现有hardware-slot
  priority机制；只改变仲裁，不改变MFMA、VMEM、DS指令数和资源。
2. **无priority的slot-aware相位实验。** 如果必须保持0 `s_setprio`，根据hardware wave slot让一条wave
  从VMEM组开始、另一条wave从MFMA组开始，或对8个`VMEM1 -> MFMA4`组做固定轮转。相同
  `sched_group`复制到所有wave不会自然产生这种相位差，因此必须显式引入slot相关非对称性。

第一阶段目标不是把VMEM类清零，而是至少接近已实测的9aa点：

- physical `vmem_issue_stall`从13.04%向5.51%下降；
- “两wave同为VMEM issue”从9.8934%向3.4205%下降；
- MFMA union busy从71.78%向76.59%上升；
- 同工作量、同资源的多buffer ABBA墙钟稳定胜出。

如果VMEM同因下降但MFMA union busy和墙钟不改善，说明空槽只是转移到了DS、tail或residual，不能保留
该候选。

### P1：移动VMEM发射相位，不盲目提前预取

Control的VMEM wait owner只有0.69%，两wave同时VMEM wait仅0.0019%；相比之下VMEM issue owner为
13.04%。这说明当前主要问题不是“数据发得太晚、consumer等不到”，而是“两wave在同一时刻争用VMEM
发射/队列”。另外，所有A/source0 producer边已经被有效`vmcnt`覆盖，因此不应通过增加`s_waitcnt`解决。

具体做法是保留现有load到consumer的长距离，只在这段距离内移动请求：

- 对8个VMEM组尝试有限的phase rotation，而不是继续细化成更多相同的`VMEM1 -> MFMA4`；
- 优先把一条wave的VMEM组放到另一条wave的MFMA硬件时隙附近，而不是放到另一条wave的VMEM组附近；
- 每次只移动1--2组，检查`vmem_issue_stall -> vmem_wait/tail/DS wait`的转移矩阵；
- 若`vmem_wait_stall`明显上升，说明消费距离被压短，应回退该移动而不是再放宽wait阈值掩盖问题。

### P2：在VMEM相位改善后处理DS

9aa把VMEM暴露压低后，DS wait从5.08%升到5.65%，说明DS是下一层会被重新暴露的瓶颈。它应在P0
候选稳定后再处理，否则容易用DS重排破坏已经获得的VMEM反相。

建议按风险从低到高尝试：

1. 将头部`DSRD8`拆成两个或更多小组，在中间插入已有的独立MFMA；
2. 前移将被下一批MFMA消费的DS read，增加`ds_read -> lgkmcnt -> MFMA`距离；
3. 只在ATT仍显示`stall:DS-*`时检查bank/address模式；正常DS issue是服务成本，不应当作bank conflict删掉。

验收时同时报告DS issue、DS wait和两wave同因。只降低单wave `lgkmcnt`而physical union不变，不算有效。

### P3：tail只能通过跨N结构重写

`structural_tail`表示当前N块的ready MFMA已经耗尽，不能靠重新排列同一N中的指令填满。要继续提高上限，
需要更早释放部分accumulator，让下一N的MFMA与当前N的postprocess/CShuffle/store重叠。

建议从单slice probe开始：

1. 先退休一个完成的accumulator slice；
2. 只把该slice的scale/routing、BF16 pack和CShuffle提前；
3. 验证下一N的首批MFMA能在当前tail中真实successful issue；
4. 再逐步扩大slice数，禁止第一步就增加完整的第二套LDS CShuffle缓冲。

每一步都必须检查VGPR/AGPR、LDS、scratch和occupancy。只要资源跨档导致低于2 waves/SIMD，或全shape
正确性失败，即使局部tail下降也应拒绝。

### 当前不应投入的方向

- **不要继续增加accumulator链来“修复RAW”。** 534,528个相邻MFMA对中背靠背RAW为0，复用距离已经
  足够；single-wave `mfma_issue_unavailable`又被peer MFMA 100%覆盖，physical暴露为0。
- **不要为A/B operand额外插入wait。** 所有A/VMEM与B/DS producer边已经由计数器顺序有效的
  `vmcnt`/`lgkmcnt`覆盖；新增wait只会缩短可调度窗口。
- **不要优先移动普通VALU。** `normal_issue_exposure`在physical中仅0.47%，而且正常issue不是stall。
- **不要按五类百分比估算收益。** 例如VMEM下降7.846个百分点只带来4.819个百分点MFMA busy提升，
  目标类别下降后必须检查它转移到了哪里。

### 候选的统一验收流程

每个候选按“小步改动 -> 正确性 -> 资源 -> 性能 -> ATT”的顺序验收：

1. **正确性。** 覆盖现有全shape测试，先看逐元素/`rel_l2`，不能只测单一K128 shape。
2. **资源。** 固定记录VGPR、AGPR、LDS、scratch和实际waves/SIMD；目标是维持Control的2 waves/SIMD。
3. **稳定性能。** 在相同PTL/DPM/NUMA状态下做多buffer配对ABBA，报告中位数和配对比，不使用单次计时。
4. **physical ATT。** 使用同一N2--N13窗口重新生成本账本，至少报告MFMA union busy、五类owner、
  两wave同因和原因转移。
5. **保留条件。** 只有正确性与资源通过、ABBA墙钟改善，并且physical账本能解释收益时才保留；
  “目标行下降但busy/墙钟不变”视为原因转移，不视为优化成功。

## 优化进展

### P0完成：K128恢复slot-aware priority

P0保留Control的`DSRD8 -> 8 x (VMEM1 -> MFMA4) -> MFMA32` sched-group，只把K128原先绕开的
hardware-slot priority边界接回：read/write stage按slot设为`1/0`，compute stage设为`3`，尾部恢复为`0`。

最终ISA的工作量和资源与`0f0a14c`基线完全相同：

| 项目 | Control | P0 |
|---|---:|---:|
| MFMA | 192 | 192 |
| buffer load / store | 38 / 8 | 38 / 8 |
| DS read / write | 32 / 22 | 32 / 22 |
| `setprio` / `s_getreg` | 0 / 0 | 12 / 1 |
| next-free VGPR / accum offset | 190 / 192 | 190 / 192 |
| LDS / scratch | 28,672B / 0 | 28,672B / 0 |
| 实际资源 | 64V + 128A，2 waves/SIMD | 64V + 128A，2 waves/SIMD |

在GPU4、`VECTOR,F8`、1800MHz determinism、10-buffer共同gateup上下文中进行24轮正反ABBA；完整输出
逐bit一致。结果为：

| 版本 | 中位时延 | 有效TFLOPS | 配对口径 |
|---|---:|---:|---:|
| `0f0a14c` Control | 2.221609ms | 417.586T | - |
| **P0 slot priority** | **2.192790ms** | **423.074T** | candidate/control中位`0.986906` |

配对时延降低`1.309%`、等价吞吐提升`1.327%`，ratio IQR为`0.985008--0.988416`，不跨1。

P0 fresh ATT仍使用单SE、CU2、全4 SIMD、N2--N13窗口；232/232 wave完整，资源为
`64V+128A/28,672B/0 scratch`。与Control的physical变化为：

| Physical状态 | P0 - Control（百分点） |
|---|---:|
| `vmem_issue_stall` | **-7.395** |
| `vmem_wait_stall` | **-0.375** |
| `ds_issue_stall` | -0.046 |
| `ds_wait_stall` | +0.290 |
| `structural_tail` | +0.863 |
| residual | +1.175 |
| **`MFMA union execution`** | **+5.487** |

其中VMEM issue两wave同因从`9.8934%`降到`3.5875%`，说明收益确实来自恢复physical VMEM反相，
不是偶然墙钟波动。P0已达到保守目标带下沿；下一步以P0为新基线，先尝试在不损失VMEM反相的前提下
降低重新暴露的DS wait，再评估tail结构重写。

### P1完成：删除CShuffle写侧冗余wait

P0的每个M8 CShuffle slice原来是：

`DS write -> lgkmcnt(0) -> DS read -> lgkmcnt(0) -> buffer store`

gfx942会保持同一wave内DS操作的发射顺序，因此第一条write-side `lgkmcnt(0)`不是保证本wave
write-before-read所必需的；真正必须保留的是第二条wait，它保证DS read结果在被buffer store消费前已经
ready。P1只删除第一条wait，不改变CShuffle地址、bank映射、DS read或结果消费顺序。

最终ISA逐行diff只有8条write-side `s_waitcnt lgkmcnt(0)`消失，对应8个M8 slice；其余指令顺序完全
相同：

| 项目 | P0 | P1 |
|---|---:|---:|
| MFMA | 192 | 192 |
| buffer / global load | 38 / 10 | 38 / 10 |
| DS read / write | 32 / 22 | 32 / 22 |
| buffer store | 8 | 8 |
| `lgkmcnt(0)` | 24 | 16 |
| next-free VGPR / SGPR | 190 / 96 | 190 / 96 |
| LDS / scratch | 28,672B / 0 | 28,672B / 0 |
| 实际资源 | 64V + 128A，2 waves/SIMD | 64V + 128A，2 waves/SIMD |

正确性分两层验证：正式ABBA的完整输出逐bit一致；另外在随机route、activation、weight和scale下，
`N=512`与`N=4096`各连续执行20次，40次完整有效输出全部逐bit一致。这同时检查了异步LDS顺序在
重复执行中没有非确定性结果。

在与P0相同的GPU4、`VECTOR,F8`、1800MHz determinism、10-buffer共同gateup上下文中进行24轮正反
ABBA，24/24轮候选均胜出：

| 版本 | 中位时延 | 有效TFLOPS | 配对口径 |
|---|---:|---:|---:|
| P0 slot priority | 2.190950ms | 423.430T | - |
| **P1去除write wait** | **2.163370ms** | **428.828T** | candidate/control中位`0.986942` |

配对时延降低`1.306%`、等价吞吐提升`1.323%`，ratio IQR为`0.983107--0.987691`，不跨1。
与P0的分阶段配对比例相乘，Control到P1的累计时延比例为`0.974019`，即约降低`2.598%`。

P1 fresh ATT继续使用单SE、CU2、全4 SIMD、N2--N13窗口；P0/P1均为232/232条完整wave，资源均为
`64V+128A/112S/28,672B/0 scratch`。按MFMA execution mass归一后的physical总时间从
`3976.05 cycle/N`降到`3964.70 cycle/N`，减少`11.36 cycle/N`；MFMA union busy增加`0.221pp`。
Single-wave总时间从`7483.49 cycle/N`降到`7325.00 cycle/N`，减少`158.49 cycle/N`（`2.118%`）。

| Physical状态 | P1 - P0（百分点） | owner周期变化 |
|---|---:|---:|
| `vmem_issue_stall` | +1.285 | +66,502 |
| `vmem_wait_stall` | +0.109 | +5,636 |
| `ds_issue_stall` | +0.495 | +25,638 |
| **`ds_wait_stall`** | **-2.052** | **-106,612** |
| `structural_tail` | +0.126 | +6,354 |
| residual | -0.185 | -9,594 |
| **`MFMA union execution`** | **+0.221** | **+9,472** |

两wave同为DS wait的见证量从`1.4230%`降到`0.5712%`，`DS wait + structural tail`联合状态从
`4.4809%`降到`3.3231%`。节省的周期没有一比一转成MFMA：两wave同为VMEM issue从`3.5875%`回升到
`4.5855%`，DS issue和tail也略有重新暴露。因此P1是成功的**DS wait优化**，不是P3 tail优化；下一步
仍应从当前P1重新采样并处理新主项，不能继续按旧P0百分比线性估算收益。

冻结产物：

- 24轮ABBA：`/tmp/moe-p1-cshuffle-nowritewait-abba24.log`，SHA256
  `4de553dce03a19f46d346ef218df3978b70f6c3b3ff21c75f8d4af2b67e52bc1`；
- 随机正确性：`/tmp/moe-p1-cshuffle-nowritewait-random20.log`，SHA256
  `1fb302710a4a9c0807ed8f1329c5edb6a1f6399023919f094f88ddf057d10326`；
- 最终ISA：`/tmp/moe-p1-cshuffle-nowritewait-dump/moe_2stage_down_prefill_1x4_0/21_final_isa.s`，
  SHA256 `0253d3e8ca489b06058ded09957ec2fae994b3c1dfb1b190c5d81da65d5d5bcd`；
- P0/P1 exposure账本：`/tmp/CONTROL_K128_P1_STALL_EXPOSURE.json`，SHA256
  `68a55e4169ea68f9d1aa2f2a0edc7d5956b698f8d4e8d0adc0f6d41c271e0895`。

### P2完成：用下一slice写入隐藏CShuffle读取等待

P1剩余的CShuffle序列仍是逐slice串行：

`write_i -> read_i -> lgkmcnt(0) -> store_i -> write_(i+1)`

P2先写slice0；前7个slice在`read_i`之后立即完成下一slice的BF16 pack和两条独立DS write，再执行
完整`lgkmcnt(0)`并store当前read结果。最后一个slice仍直接`read_7 -> lgkmcnt(0) -> store_7`：

`read_i -> pack/write_(i+1) -> lgkmcnt(0) -> store_i`

因此P2没有删除read-result wait，也没有依赖非零`lgkmcnt`阈值；它只增加DS read到完整wait之间的
独立工作距离。最终ISA确认每个前7个slice的两条next-write都位于当前read和wait之间。MFMA、VMEM、
DS和store工作量保持不变：

| 项目 | P1 | P2 |
|---|---:|---:|
| MFMA | 192 | 192 |
| buffer / global load | 38 / 10 | 38 / 10 |
| DS read / write | 32 / 22 | 32 / 22 |
| buffer store | 8 | 8 |
| `lgkmcnt(0)` | 16 | 16 |
| next-free VGPR / accum offset | 190 / 192 | 192 / 192 |
| LDS / scratch | 28,672B / 0 | 28,672B / 0 |
| 权威资源 | 64V + 128A，2 waves/SIMD | 64V + 128A，2 waves/SIMD |

next-free VGPR增加2，但P1本来就按192档分配，P2没有跨资源档；ATT资源CSV仍为
`64V+128A/112S/28,672B/0 scratch`。随机route、activation、weight和scale下，`N=512`与
`N=4096`各连续执行20次，40次完整有效输出全部逐bit一致；正式ABBA的完整输出同样逐bit一致。

在相同GPU4、`VECTOR,F8`、1800MHz determinism和10-buffer共同gateup上下文中，以提交`4e5b914`
的P1源码快照为control进行24轮正反ABBA，24/24轮候选均胜出：

| 版本 | 中位时延 | 有效TFLOPS | 配对口径 |
|---|---:|---:|---:|
| P1去除write wait | 2.158549ms | 429.785T | - |
| **P2增加read隐藏距离** | **2.138729ms** | **433.768T** | candidate/control中位`0.991734` |

配对时延降低`0.827%`、等价吞吐提升`0.833%`，ratio IQR为`0.984884--0.993821`，不跨1。
将P0、P1、P2三步配对比例相乘，Control到P2的累计时延比例约为`0.965968`，即降低`3.403%`、
等价吞吐提升`3.523%`。

P2 fresh ATT继续使用单SE、CU2、全4 SIMD、N2--N13窗口；P1/P2均为232/232条完整wave。按MFMA
execution mass归一后的physical总时间从`3964.70 cycle/N`降到`3929.22 cycle/N`，减少
`35.48 cycle/N`（`0.895%`）；MFMA union busy从`77.484%`升到`78.183%`，增加`0.700pp`。
Single-wave总时间从`7325.00 cycle/N`降到`7263.15 cycle/N`，减少`61.86 cycle/N`（`0.844%`）。

| Physical状态 | P2 - P1（百分点） | owner周期变化 |
|---|---:|---:|
| `vmem_issue_stall` | +0.299 | -2,638 |
| `vmem_wait_stall` | +0.104 | +4,090 |
| `ds_issue_stall` | +0.150 | +1,274 |
| **`ds_wait_stall`** | **-1.184** | **-66,796** |
| `structural_tail` | -0.028 | -20,346 |
| residual | -0.041 | -6,708 |
| **`MFMA union execution`** | **+0.700** | - |

两wave同为DS wait的见证量从`0.5712%`降到`0.1263%`，`DS wait + structural tail`联合状态从
`3.3231%`降到`2.2886%`。`code.json`会把重复基本块的相同PC合并，不能用静态row数代表8个source
slice；在可识别的CShuffle区域记录中，read-side wait stall从`1.704M`降到`0.916M`，每次动态命中的
平均stall从`65.58`降到`41.14 cycles`。主归因仍以闭合的physical owner账本为准：P2继续降低了
DS wait，但没有消除新主项`structural_tail 7.51%`和`vmem_issue_stall 7.22%`。

冻结产物：

- 24轮ABBA：`/tmp/moe-p2-cshuffle-readpipe-abba24.log`，SHA256
  `8d9df432288975da1cc764c289e644d935a2a05e070f3d7956c29f4890513791`；
- 随机正确性：`/tmp/moe-p2-cshuffle-readpipe-random20.log`，SHA256
  `1fb302710a4a9c0807ed8f1329c5edb6a1f6399023919f094f88ddf057d10326`；
- 当前源码对应最终ISA：
  `/tmp/moe-p2-cshuffle-readpipe-final-dump/moe_2stage_down_prefill_1x4_0/21_final_isa.s`，SHA256
  `433917dea86889d09afcf4badbb782ff17633213b099ddf2ff9b6bce6dfc7ae8`；
- P1/P2 exposure账本：`/tmp/CONTROL_K128_P2_STALL_EXPOSURE.json`，SHA256
  `39c92985d70fa3e4f864475f324cb83bb3d34dae09c6c35283bb932aa7684072`。

### P3完成：gfx942 non-temporal输出store

P2 fresh ATT的physical `vmem_issue_stall`为`7.22%`，两wave同因达`4.62%`；opcode细分又显示8条
输出store累计`1.256M` stall，并会反压下一轮weight load。输出是一次写入、随后由同stream中的独立
sorted_sum kernel读取，适合non-temporal写策略。

LLVM `AMDGPURawPtrBufferStore`的aux定义在gfx942与旧架构不同：bit0/bit1/bit4分别是
`sc0/nt/sc1`。P3把physical N256输出store的aux从`0`改为`2`。最终ISA逐行diff严格只有8条
`buffer_store_dwordx4`增加`nt`后缀；指令数、顺序和资源完全不变：

| 项目 | P2 | P3 |
|---|---:|---:|
| MFMA | 192 | 192 |
| buffer / global load | 38 / 10 | 38 / 10 |
| DS read / write | 32 / 22 | 32 / 22 |
| buffer store | 8个default | 8个`nt` |
| next-free VGPR / accum offset | 192 / 192 | 192 / 192 |
| LDS / scratch | 28,672B / 0 | 28,672B / 0 |
| 权威资源 | 64V + 128A，2 waves/SIMD | 64V + 128A，2 waves/SIMD |

随机route、activation、weight和scale下，`N=512`与`N=4096`各连续执行20次，40次完整有效输出
全部逐bit一致；正式ABBA的完整输出也逐bit一致。相同stream的kernel边界保证后续消费者看到已经完成
的store，`nt`只改变缓存策略，不改变程序顺序或输出地址。最终运行完整
`tests/contrib/moe/test_flydsl_moe_down.py`设备回归，结果为`11 passed`。

在相同GPU4、`VECTOR,F8`、1800MHz determinism和10-buffer共同gateup上下文中，以提交`406906d`
的P2源码快照为control进行24轮正反ABBA，24/24轮候选均胜出：

| 版本 | 中位时延 | 有效TFLOPS | 配对口径 |
|---|---:|---:|---:|
| P2增加read隐藏距离 | 2.137749ms | 433.967T | - |
| **P3 non-temporal store** | **2.110829ms** | **439.502T** | candidate/control中位`0.987811` |

配对时延降低`1.219%`、等价吞吐提升`1.234%`，ratio IQR为`0.981261--0.990510`，不跨1。
将P0--P3四步配对比例相乘，Control到P3的累计时延比例约为`0.954194`，即降低`4.581%`、
等价吞吐提升`4.801%`。

P3 fresh ATT继续使用单SE、CU2、全4 SIMD、N2--N13窗口；P2/P3均为232/232条完整wave，权威资源
均为`64V+128A/112S/28,672B/0 scratch`。按MFMA execution mass归一后的physical总时间从
`3929.22 cycle/N`降到`3825.60 cycle/N`，减少`103.62 cycle/N`（`2.637%`）；MFMA union busy从
`78.183%`升到`80.301%`，增加`2.118pp`。Single-wave总时间从`7263.15 cycle/N`降到
`7094.46 cycle/N`，减少`168.68 cycle/N`（`2.322%`）。

| Physical状态 | P3 - P2（百分点） | owner周期变化 |
|---|---:|---:|
| **`vmem_issue_stall`** | **-1.608** | **-92,980** |
| `vmem_wait_stall` | -0.128 | -7,264 |
| `ds_issue_stall` | -0.146 | -13,132 |
| `ds_wait_stall` | -0.064 | -8,152 |
| `structural_tail` | -0.129 | -24,216 |
| residual | -0.043 | -6,420 |
| **`MFMA union execution`** | **+2.118** | - |

两wave同为VMEM issue的见证量从`4.6191%`降到`3.5092%`。动态opcode细分进一步闭合了原因：

| 动态类别 | P2 stall | P3 stall | stall / hit | 变化 |
|---|---:|---:|---:|---:|
| 8条输出store | 1.256M | 1.022M | 42.30 -> 34.41 cycles | -18.66% |
| weight/global load | 1.672M | 1.406M | 17.66 -> 14.86 cycles | -15.88% |

因此P3不仅降低了store自身发射阻塞，也释放了共享VMEM队列、降低后续weight load阻塞；physical
VMEM owner、MFMA union busy和ABBA墙钟三者方向一致。P3后的首要剩余项变为
`structural_tail 7.38%`和`vmem_issue_stall 5.62%`，而不是继续追逐已经降到`2.07%`的DS wait。

冻结产物：

- 24轮ABBA：`/tmp/moe-p3-store-nt-abba24.log`，SHA256
  `60708478a5189ffef6120ff8fa824fd7befdbb2ee957f9f1416cf17e6ba625fb`；
- 随机正确性：`/tmp/moe-p3-store-nt-random20.log`，SHA256
  `1fb302710a4a9c0807ed8f1329c5edb6a1f6399023919f094f88ddf057d10326`；
- 最终ISA：`/tmp/moe-p3-store-nt-dump/moe_2stage_down_prefill_1x4_0/21_final_isa.s`，SHA256
  `416d8ee48b0537822a760420ef4025d51a9f40d2fa04d4a158a65d469865387a`；
- P2/P3 exposure账本：`/tmp/CONTROL_K128_P3_STALL_EXPOSURE.json`，SHA256
  `4e2de5c5ef4a86d79bdab85890b4a6856afd9c031b37f43ef758a019546f8e8e`。

### P3 tail内部成本分解

为避免继续把`structural_tail`误当成单一wait，按每条wave的
`last MFMA execution end -> next N first MFMA successful issue`切出N2--N13窗口，并将ATT记录拆成
`attempt -> successful issue`的显式stall和`successful issue -> complete`的正常service。P3共有
232条完整计算wave、2,784个窗口，平均tail为`2316.14 cycles/N`。

| Tail类别 | 动态次数/N | stall cycles/N | service cycles/N | span cycles/N |
|---|---:|---:|---:|---:|
| 16条`ds_write_b128` | 16 | 106.73 | 417.55 | **524.29** |
| 64条`v_fmaak_f32` | 64 | 37.66 | 256.63 | 294.29 |
| 8条`lgkmcnt(0)` | 8 | **329.87** | 0 | 329.87 |
| 8条NT store | 8 | 288.96 | 35.26 | 324.22 |
| 32条`v_perm_b32` | 32 | 22.99 | 128.28 | 151.27 |

这些span会跨管线重叠，不能相加为线性speedup预算，但足以定位工作量主项：16条CShuffle
`ds_write_b128`的正常service加stall已经超过8条read wait。逐PC结果在N6后稳定，slot/SIMD差异很小，
说明它是每N固定工作量，而不是偶发bank conflict。因此下一步应先减少DS write指令数，而不是重复
已经否证的双read、read-ahead、cache和priority扫描。

P3之后还补测了两个邻域，均通过`N=512/4096`各20次随机完整输出逐bit验证，但未通过性能门槛：

- 逐局部fragment执行packed FMA/BF16：资源进入约256 combined档，4轮ratio中位`1.021923`，
  后三轮全部退化`2.0%--3.5%`；
- output store `nt -> sc0 nt`：ISA只给8条store增加`sc0`，4轮ratio中位`0.999144`、IQR
  `0.948416--1.007361`，2/4轮退化，未稳定胜出。

### P6完成：全64-lane M16 row-pair CShuffle

P3每个M8 slice只有32个lane生产数据，每个slice需要两条masked `ds_write_b128`；8个slice共16条。
P6将相邻两个M8合并为一个M16 row-pair，让64个lane都参与同一条DS write：

```text
row_half     = (lane % 16) // 8
row_in_8     = lane % 8
logical_atom = 2 * (lane // 16) + channel_piece
physical_atom = logical_atom ^ row_in_8
```

每个row-pair只需两条全wave `ds_write_b128`，4个pair共8条。程序化双射检查证明每个pair的128个
128-bit atom恰好覆盖`M16 x N64 x BF16`的1,024个元素，且读取端沿用相同XOR逆映射。CShuffle LDS
由每wave 1KB扩大为2KB，即每WG 4KB扩大为8KB；整个kernel LDS从28,672B到32,768B，仍恰好允许
2 WG/CU，也就是2 waves/SIMD。读取、完整`lgkmcnt(0)`和NT store仍按单个M8依次退休，没有引入
已否证的双read成组或双store突发。

| 项目 | P3 | P6 |
|---|---:|---:|
| MFMA | 192 | 192 |
| buffer / global load | 38 / 10 | 38 / 10 |
| DS read / write总数 | 32 / 22 | 32 / 14 |
| CShuffle `ds_write_b128` | 16 | **8** |
| CShuffle read wait / NT store | 8 / 8 | 8 / 8 |
| next-free VGPR / accum offset | 192 / 192 | 194 / 196 |
| LDS / private scratch | 28,672B / 0 | 32,768B / 0 |
| ATT权威资源 | 64V + 128A / 112S | 68V + 132A / 112S |
| occupancy | 2 waves/SIMD | 2 waves/SIMD |

随机route、activation、weight和scale下，`N=512`与`N=4096`各连续20次，40次完整有效输出均与P3
逐bit一致；正式ABBA完整输出同样逐bit一致。完整
`tests/contrib/moe/test_flydsl_moe_down.py`设备回归结果为`11 passed`。

在相同GPU4、`VECTOR,F8`、1800MHz determinism、10-buffer轮换和共同gateup上下文中，以提交
`027c4b7`的P3源码为control进行24轮正反ABBA，24/24轮candidate/control均小于1：

| 版本 | 中位时延 | 有效TFLOPS | Q1--Q3 |
|---|---:|---:|---:|
| P3 non-temporal store | 2.110069ms | 439.660T | 2.105999--2.115989ms |
| **P6 M16 row-pair** | **2.053649ms** | **451.739T** | 2.049659--2.066369ms |

配对ratio中位为`0.974461`，IQR为`0.969446--0.976308`：时延降低`2.554%`，等价吞吐提升
`2.621%`。将P0--P3累计ratio与P6相乘，分析Control到P6的时延约降低`7.017%`，等价吞吐约提升
`7.547%`。

P6 fresh ATT仍使用单SE、CU2、全4 SIMD和N2--N13。原始trace包含244条完整stitched wave：
228条各有3,072条MFMA的计算wave，以及16条各9条控制指令、0 MFMA的valid边界early-exit wave，
后者在4个SIMD上各4条。账本只过滤这16条完整early-exit wave；不存在`1..3071`条MFMA的部分wave。
P3/P6分别按2,784/2,736个动态计算N窗口归一，避免直接比较不同采样wave数的总cycle。

Single-wave总时间从`7094.46`降到`6989.75 cycles/N`，减少`1.476%`。Physical MFMA union busy
从`80.301%`升到`80.619%`，增加`0.318pp`。互斥owner显示这是明显的成本转移，而不是各类别一比一
变成MFMA busy：

| Physical owner | P3 | P6 | P6-P3 |
|---|---:|---:|---:|
| `ds_issue_stall` | 2.447% | **0.839%** | **-1.608pp** |
| `ds_wait_stall` | 2.074% | 3.061% | +0.986pp |
| `structural_tail` | 7.382% | **4.954%** | **-2.429pp** |
| `vmem_issue_stall` | 5.616% | 8.115% | +2.499pp |
| `vmem_wait_stall` | 0.397% | 0.466% | +0.069pp |
| **MFMA union execution** | **80.301%** | **80.619%** | **+0.318pp** |

tail窗口平均长度从`2316.14`降到`2152.80 cycles/N`，降低`7.05%`。其中CShuffle
`ds_write_b128`从16条降到8条后，stall从`106.73`降到`20.33 cycles/N`，正常service从
`417.55`降到`184.08 cycles/N`；masked producer的8组`saveexec/branch/or`也全部消失。与此同时，
8条`lgkmcnt(0)`的stall从`329.87`升到`514.10 cycles/N`，8条NT store的stall从`288.96`升到
`396.43 cycles/N`，weight load阻塞也重新暴露。这解释了为什么DS write减半带来稳定`2.55%`墙钟
收益，但MFMA union只增加`0.318pp`：节省的DS service首先缩短tail，然后部分转移为共享DS/VMEM
队列阻塞，不能按类别下降量线性外推。

冻结产物：

- 24轮ABBA：`/tmp/moe-p6-rowpair-abba24.log`，SHA256
  `b438d22c68fcf9563e894ee95b3fe972973941926dbaae27837f587c52bd4e2c`；
- 随机正确性：`/tmp/moe-p6-rowpair-random20.log`，SHA256
  `1fb302710a4a9c0807ed8f1329c5edb6a1f6399023919f094f88ddf057d10326`；
- 完整down设备回归：`/tmp/moe-p6-rowpair-regression11.log`，SHA256
  `b3bb326aeb61ee351538b1bf79e13eb58019bfa322a435ae63eaa9498f1b457b`；
- 最终ISA：`/tmp/moe-p6-rowpair-dump/moe_2stage_down_prefill_1x4_0/21_final_isa.s`，SHA256
  `83893be8c8b69ab127d07d2ac62d49d907eacf1fcc7cc13c333d1812275458ee`；
- P6 exposure账本：`/tmp/moe-p6-rowpair-exposure.json`，SHA256
  `92535e6dcd69c11c698e9147c914b287d290da758be34f508824645df05768cf`；
- P3/P6 successful-issue slot账本：`/tmp/moe-p3-p6-slots.json`，SHA256
  `507a1048538ecfdeb0993cd222feef5676b30217913795434817ef9b002254d3`。

### 已否证的低风险单变量

以下候选都已恢复，不应在没有新ATT证据时重复扫描。短筛均以P0为control并保持2 waves/SIMD：

| 候选 | 4轮candidate/control中位 | 判定 |
|---|---:|---|
| 头部`DSRD8 -> DSRD4 + DSRD4` | `0.999463` | IQR跨1，未稳定降低DS wait |
| read/write priority `1/0 -> 2/0` | `1.001519` | 退化 |
| 取消compute阶段priority | `1.044018` | 明确退化约4.4% |
| read/write slot priority反转为`0/1` | `0.997456` | IQR跨1，无稳定收益 |
| compute priority `3 -> 2` | `1.003133` | 后三轮全部退化 |
| CShuffle双slice成组、8次wait降为4次 | `1.002490` | 后三轮两次约退化0.9%，DS/store突发抵消收益 |
| CShuffle单read-ahead、read与前一store重叠 | `0.998282` | 最后两轮退化，IQR跨1 |
| 跨N延迟1个M8 slice，携带BF16 fragment | `1.006517` | 218/220 VGPR档，后三轮全部退化 |
| 跨N延迟1个M8 slice，仅携带LDS内容 | `0.997889` | 220/220 VGPR档，IQR跨1 |
| 32条packed FMA替代64条scalar FMA | `1.013091` | 254/256 VGPR硬边界，后三轮全部退化 |
| 16条packed + 32条scalar FMA | `1.016452` | 222/224 VGPR档，后三轮全部退化 |
| 逐fragment FMA/BF16/CShuffle退休 | `1.015805` | 资源与工作量不变，但后三轮全部退化 |
| output store `nt -> nt sc1` | `1.018043` | ISA仅多8个`sc1`，后三轮全部退化 |
| output store `nt -> sc0 nt` | `0.999144` | IQR跨1，2/4轮退化，无稳定收益 |
| tail-only slot priority `1/0 -> 0/1` | `1.010352` | 资源/工作量不变，后三轮全部退化 |
| VMEM/MFMA间隔`4 -> 5` | - | 后端生成与P3逐字相同的ISA，无有效旋钮 |
| packed-local逐fragment后处理 | `1.021923` | 约256 combined档，后三轮全部退化2.0%--3.5% |

上述P4候选均通过`N=512/4096`各20次随机完整输出逐bit验证，并保持2 waves/SIMD，但都没有通过
稳定性能门槛。Priority实验说明P0的完整阶段切换必须保留；简单拆分DSRD也没有把single-wave变化
转化为physical收益。更重要的是，当前tail重写的第一约束已经不是“能否保持2-wave”，而是即使仍在
2-wave档，combined寄存器从192升到220--256也会增加调度压力；不能只看occupancy整数值。

因此P6取代P3成为当前已验证的2-wave最优点：24轮吞吐为`451.739T`，相对分析Control的分阶段
累计吞吐约提升`7.547%`，fresh ATT的MFMA union busy为`80.619%`。剩余最大owner已转为
`vmem_issue_stall 8.115%`，其次是`structural_tail 4.954%`和`ds_wait_stall 3.061%`。P6 LDS恰好
32KB、权威combined资源为200，下一步不能再增加LDS，也不应重复双read成组、read-ahead、packed
FMA或cache/priority扫描；必须针对新暴露的VMEM队列状态提出保持2-wave且不增加长期live state的方案。
