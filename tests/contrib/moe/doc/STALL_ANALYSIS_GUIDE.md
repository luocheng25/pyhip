# MoE Down ATT与Stall分析方法

## 目标

本手册把一次stall观察变成可复验的优化闭环：

```text
正确性 -> ISA/资源 -> clean ABBA -> fresh ATT
         -> physical SIMD union -> exclusive owner -> 单变量候选 -> 重新闭环
```

ATT用于解释“时间在哪里、候选为何生效”，不能替代墙钟。只有目标stall下降、physical MFMA union busy改善且同进程ABBA墙钟同时改善，候选才可晋级。

## 四条不可违反的口径

### 1. Successful issue不是first attempt

gfx9 ATT动态记录为：

```text
[first_attempt, category, stall, duration, pc_index]
```

真实成功发射时刻是：

$$
t_{issue}=t_{first\_attempt}+stall
$$

MFMA execution window从`t_issue`开始，gfx942本项目按16 cycles建模。若把`first_attempt`当issue，会把阻塞期误画成执行期，夸大跨wave overlap并高估MFMA busy。

### 2. 指令映射必须使用`code[pc_index]`

`pc_index`已经是`code.json`中该动态事件的索引。使用`code[pc_index - 1]`会把事件整体错配到前一条ISA，进而颠倒load、wait、MFMA和store归因。每次更换rocprof版本后都应抽样检查动态事件的opcode与源码行。

### 3. 单wave stall不能直接相加

同一physical SIMD上的resident waves共享MFMA执行资源。一条wave的`stall:MFMA`可能被peer的16-cycle MFMA execution完全覆盖；这种stall对physical墙钟没有直接缺口。

必须按以下key合并resident waves：

```text
(shader_engine, cu, simd)
```

并计算MFMA execution window并集：

$$
U_{MFMA}=\frac{|\bigcup_i W_i^{MFMA}|}{T_{steady}}
$$

只有并集之外的physical idle才需要归因。slot数必须从trace动态识别；本项目已有2、3、4 resident-wave案例，不能硬编码为2。

### 4. Physical SIMD账本不覆盖跨CU dispatch-tail

ATT解释一个被采样physical SIMD内部的resident-wave overlap，但整卡墙钟还受workgroup调度粒度和各CU任务尾部影响。目标CU可能恰好落在轻载组，因此“采样CU的MFMA union更高”不能推出整个dispatch更快。

对80 CU和`T`个逻辑M64任务，当前4-wave physical与8-wave paired路径的critical waves/SIMD分别是：

$$
W_{physical}=\left\lceil\frac{T}{80}\right\rceil,\qquad
W_{paired}=2\left\lceil\frac{T}{160}\right\rceil
$$

当`T mod 160`落在`1..80`时，paired会多支付一个wave batch。实测`T=1024`时paired采样CU的steady MFMA union为90.73%，高于physical的86.29%，但整卡combined ratio仍为`1.01946`；将任务平衡为`T=1280`后，paired combined ratio转为`0.97949`。因此每次还必须报告全SE/CU的captured-wave或WG分布、critical-wave公式和整卡ABBA。Physical owner账本解释局部气泡，不能替代dispatch-tail模型。

## 互斥Owner账本

对每个4-cycle physical idle tick，先判定各resident wave的blocker，再把该tick等分给同时存在的owner。优先级如下：

1. VMEM instruction issue stall：`buffer/global/flat load/store`自身未能发射。
2. VMEM completion wait：`s_waitcnt vmcnt(...)`等待已发请求完成。
3. LDS instruction issue stall：`ds_read/ds_write`自身未能发射。
4. LDS completion wait：`s_waitcnt lgkmcnt(...)`等待DS结果。
5. mixed wait：同一wait同时含`vmcnt`与`lgkmcnt`。
6. structural tail：当前N的ready MFMA已耗尽，处在postprocess/CShuffle/store尾部。
7. MFMA issue unavailable：排除显式memory blocker后，本wave暂不能发MFMA。
8. normal issue、scheduler ready和其他残差。

“正常发射一条DS/VMEM”是服务成本，不等于stall；“VMEM issue stall”与“vmcnt wait”也不是同一问题：

| 观察 | 更可能的原因 | 首选动作 | 不应做的事 |
| --- | --- | --- | --- |
| VMEM issue高、vmcnt wait低 | resident waves同相提交请求，队列/发射口争用 | 改wave间相位或role/slot priority，保持消费距离 | 盲目把load继续提前 |
| vmcnt wait高 | 请求到consumer距离不足或请求过晚 | 预取、增加独立计算距离 | 用更多wait掩盖问题 |
| DS issue高 | DS发射拥塞或地址模式问题 | 检查同相、bank PMC、拆分请求 | 把所有DS cycles都称为bank conflict |
| lgkmcnt wait高 | read到consumer距离不足 | 前移read、在read/wait间插独立工作 | 只删必要wait |
| structural tail高 | 当前N没有ready MFMA | 分片退休或跨N overlap | 在同一N内重排普通VALU并宣称可填满tail |
| 单waveMFMA stall高、union为0 | peer已覆盖、共享MFMA资源正常仲裁 | 不优化该桶 | 增加accumulator链 |

### Witness不是可加速预算

`all_waves_same_reason`只证明所有resident waves在同一tick暴露相同原因，不能再加到owner账本。Oracle recoverable值假设目标blocker可立即替换为ready MFMA，只是上界见证，不是性能预测。

账本恒等式是：

$$
U_{MFMA}+\sum_r E_r=100\%
$$

它不表示删除$E_r$就能等量增加$U_{MFMA}$。Control K128与早期`9aa595d`对照中，VMEM issue/wait下降7.846个百分点，但MFMA union只增加4.819个百分点，其余时间转移到DS、tail和residual。

## 标准工作流

### 0. 固定问题和控制变量

在采集前记录：

- commit、kernel SHA256和候选patch；
- GPU、gfx、ROCm/FlyDSL版本、80 CU与power cap；
- shape：`B/TOPK/E/N/K/BM`、量化、padding和metadata排序单位；
- tile、threads/WG、waves/WG、TiledMMA结构；
- 初始GPU idle状态、1800MHz determinism、PTL `Enabled / VECTOR,F8`；
- control与candidate是否使用同一数学输出契约。

若PTL、外部负载或metadata不同，先修复实验，不分析绝对时间。

### 1. 正确性与ISA资源门禁

ATT前至少确认：

- valid physical rows、padding和inactive tail；
- reduced row-major输出、finite检查和`rel_l2`；
- MFMA/load/store/barrier数量符合算法；
- VGPR/AGPR、LDS、private/scratch、实际waves/SIMD；
- 没有意外spill或跨occupancy台阶。

例如K192 single-M基线为128 VGPR、32KB LDS、0 scratch、96 MFMA、2 barriers、4 waves/SIMD。若候选跨到3 waves/SIMD，后续stall分布已不是同资源实验。

### 2. 先用clean ABBA确定是否值得解释

正式协议使用10 rotating buffers、24轮ABBA；每版本有48个绝对样本和24个paired ratio。报告：

```text
control ms -> candidate ms
candidate/control median
ratio Q1..Q3
wins/rounds
```

短ABBA只淘汰明显回退。外部任务占用时只可将同进程ratio标成stress证据，不能替代clean结果；idle gate拒绝运行是正确行为。

### 3. 采集fresh ATT

现有入口：

- [profile_unified8_att.py](../profile_unified8_att.py)：构造shape、连续dispatch并检查finite。
- [unified8_att.yaml](../unified8_att.yaml)：ATT job模板。

先用kernel trace确认真实kernel name，再更新YAML中的`kernel_include_regex`和新的`output_directory`。多入口重构后，旧的`.*moe_2stage_down_prefill_1x4_0.*`不一定匹配目标入口。

示例：

```bash
HIP_VISIBLE_DEVICES=4 \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
rocprofv3 -i tests/contrib/moe/unified8_att.yaml -- \
  python tests/contrib/moe/profile_unified8_att.py \
  --path single_n512 --k 192 --dispatches 6 --exact-valid-grid
```

注意：`profile_unified8_att.py`只负责launch和finite检查，不负责设置或恢复GPU/PTL状态。采集前后必须独立核验并恢复`auto / F16,BF16`。每个候选使用新目录，不能复用旧`ui_output_agent_*`。

### 4. 生成physical MFMA slot ledger

分析器：[analyze_down_mfma_slots.py](../analyze_down_mfma_slots.py)。

K192 single-M N512示例（8个N512块、每个K64 core每wave 32条MFMA）：

```bash
python tests/contrib/moe/analyze_down_mfma_slots.py \
  --trace control=/tmp/control/ui_output_agent_*_dispatch_5 \
  --trace candidate=/tmp/candidate/ui_output_agent_*_dispatch_5 \
  --n-blocks 8 --mfma-per-core 32 --workers 4 \
  --json /tmp/k192-slots.json \
  --markdown /tmp/k192-slots.md \
  --svg /tmp/k192-slots.svg
```

K384 N256用`--n-blocks 16 --mfma-per-core 64`。参数必须来自真实tile和动态MFMA计数，不能照抄示例。当前分析器假设每N有3个K core；K512 direct-store等不同pipeline在使用前必须先验证模型能闭合，不能为了得到报告强套参数。

检查项：

- 每条完整wave的MFMA数与预期一致；
- physical group的resident slot数与资源推导一致；
- steady窗口不含slot replacement、prologue和drain；
- `successful_issue_formula`为`first_attempt + stall`；
- control/candidate使用相同steady定义。

### 5. 生成exclusive stall ledger

分析器：[analyze_down_stall_exposure.py](../analyze_down_stall_exposure.py)。

```bash
python tests/contrib/moe/analyze_down_stall_exposure.py \
  --trace control=/tmp/control/ui_output_agent_*_dispatch_5 \
  --trace candidate=/tmp/candidate/ui_output_agent_*_dispatch_5 \
  --n-blocks 8 --mfma-per-core 32 \
  --first-n 1 --last-n-exclusive 7 \
  --resident-waves 4 \
  --json /tmp/k192-exposure.json \
  --markdown /tmp/k192-exposure.md
```

N窗口应去掉首尾非稳态块，并用timeline确认所有slot完整覆盖；不能机械套`1..7`。至少报告：

- steady与lifecycle MFMA union busy；
- owner占总时间、占idle时间；
- all-waves-same-reason witness；
- VMEM issue与vmcnt wait分开；
- LDS issue、lgkmcnt wait与normal issue分开；
- 原因转移表；
- top PC、opcode、源码行和发生phase。

### 6. 用joint phase定位可改代码

owner只回答“谁占了空槽”，joint state回答“哪些wave阶段同时发生”。对热点PC检查：

1. 它是请求发射、completion wait还是普通issue？
2. peer waves处于`core0/core1/core2/boundary/tail`哪一阶段？
3. producer到first consumer还有多少successful-issue距离？
4. 是否所有producer边都已被计数器顺序有效的wait覆盖？
5. 该PC能否移动而不增加指令、wait、寄存器或barrier？

K192 single-M示例中，剩余physical idle的62.57%由VMEM issue拥有，VMEM completion wait仅4.60%；四wave同时VMEM窗口主要是“1条next-N weight load + 3条output store”。热点K0 load到first consumer仍有约3.5K cycles中位距离，因此问题是load/store队列相位，而不是数据到达太晚。这才支持role priority，而不是继续提前预取。

### 7. 只做一个可证伪改动

候选应直接对应一个owner假设：

| 假设 | 最小候选 | 可证伪检查 |
| --- | --- | --- |
| 两slot VMEM同相 | slot/role priority或固定phase rotation | same-reason VMEM下降，wait不反弹 |
| DS read消费距离不足 | 在read与wait间放独立pack/write | lgkmcnt owner下降，指令数不增 |
| CShuffle tail过长 | 单slice分片退休probe | 下一N首批MFMA真实进入旧tail |
| occupancy是主因 | 仅跨一个VGPR/LDS门槛 | resident waves增加且墙钟改善 |

禁止一次同时改tile、priority、store policy和padding；否则ATT即使变好也无法归因。

### 8. 重新执行四层闭环

保留候选必须同时满足：

1. 正确性不退化，finite/tail/padding全部通过。
2. ISA工作量与资源变化已解释，没有意外spill。
3. clean ABBA24的ratio/IQR/wins稳定改善。
4. fresh ATT中目标owner、physical union busy和墙钟方向一致。

若owner下降但union/墙钟不变，结论是“暴露转移”，不是成功。若墙钟改善但ATT未解释，先检查trace dispatch、PTL、代码SHA和steady窗口，不补故事。

## 已验证的分析案例

### Control K128：physical union纠正了错误优先级

Control的single-wave `mfma_issue_unavailable`很大，但100%被peer MFMA execution覆盖，physical该项为0；真正空槽是VMEM issue owner 13.04%，两wave同因9.8934%。恢复slot-aware priority后：

- ABBA24 ratio `0.986906`；
- VMEM同因降到3.5875%；
- MFMA union约提高5.49个百分点；
- MFMA/load/store/DS数量与2-wave资源档位不变。

这证明优化对象是跨wave请求相位，不是本waveMFMA RAW。

### P7：局部wait优化必须看墙钟

删除冗余write-side wait、pipeline next slice、减半DS write和row-pair read分别通过独立ABBA晋级。反序pair虽有ratio `0.992165`，但Q3为`1.015624`且LLVM同时改写FMA/寄存器，因此拒绝。ATT只提供假设，稳定ratio决定是否保留。

### K192 single-M：issue contention不是completion latency

single-M把physical MFMA union从N256的64.42%提高到70.32%，但剩余idle中VMEM issue占62.57%，completion wait仅4.60%。四wavejoint state进一步定位到next-N weight load与output-store尾部重叠。role priority短ABBA4改善down 2.50%、combined 1.91%，但仍需clean ABBA24和fresh ATT才能正式晋升。

## 常见失误清单

- 用`first_attempt`画MFMA window。
- 用`code[pc_index - 1]`映射ISA。
- 把所有wave的stall cycles直接求和。
- 把`duration`或4-cycle trace issue cost当MFMA initiation interval。
- 把normal DS/VMEM issue当stall。
- 把VMEM issue stall与`vmcnt` completion wait合并。
- 只看steady busy，不看包含prologue/drain/tail的lifecycle busy。
- 把owner、same-reason witness和oracle上界重复相加。
- 目标owner下降后不报告转移到哪里。
- 用旧trace解释新ISA，或复用旧output目录。
- 采集时不记录PTL、DPM、外部负载和源码SHA。
- 在模型不闭合时继续解读百分比。

## 相关证据

- [Control K128完整账本](../CONTROL_K128_STALL_EXPOSURE.md)
- [42提交编年史](COMMIT_CHRONICLE.md)
- [Hy3 single-M handoff](../../../../docs/HY3_SINGLE_M_N512_HANDOFF_TODO.md)
- [重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)
