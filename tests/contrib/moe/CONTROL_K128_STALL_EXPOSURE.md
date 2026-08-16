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
