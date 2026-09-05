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

## 七层分层报告

报告必须按下面七层自外向内展开。每层有自己的分母，不能把不同层的百分比相加。
前两层回答“工作是否均匀地送到硬件”，第3--5层回答“一批resident waves的生命周期
花在哪里”，第6层才回答“稳态MFMA空槽由什么占用”。

| 层级 | 分析对象 | 分母 | 核心输出 | 何时继续下钻 |
| ---: | --- | --- | --- | --- |
| 1 | 全设备CU | 静态active tile/WG分配容量 | `Z_CU`、`I_CU`、tasks/CU | `I_CU`显著时先修grid/mapping |
| 2 | 每CU内SIMD | 静态wave或resident-batch分配容量 | `Z_SIMD`、`I_SIMD`、waves/SIMD | 不均衡显著时先修wave映射 |
| 3 | 单SIMD prologue | active-batch lifecycle | cycles/batch、占比、分位数 | 首MFMA过晚时分析初始化关键路径 |
| 4 | 单SIMD epilogue | active-batch lifecycle | cycles/batch、占比、分位数 | 末MFMA后尾巴长时分析store/drain |
| 5 | 单SIMD steady | active-batch lifecycle | steady占比、内部N窗口覆盖率 | 只有steady足够长才进入第6层 |
| 6 | steady MFMA union | 选定的内部steady窗口 | busy/idle及七类互斥stall | 按最大可控类别选择一个实验 |
| 7 | 决策闭环 | 同shape的control/candidate | 原因转移、ABBA24、结论 | 三者同向才保留候选 |

第1--2层是由任务数、tile划分和硬件资源静态求出的容量口径，不依赖ATT时间戳；
第3--5层是生命周期口径，第6层是内部steady窗口口径。
只有第3--5层内部、以及第6层的`MFMA busy + 七类stall`各自要求加和闭合。

### 1. CU任务不均衡

该层必须在读取ATT前静态完成。先由shape和tile计算逻辑任务数，再由launch grid计算
启动的workgroup数。例如MoE down中：

$$
T_M=\left\lceil\frac{valid\_rows}{BM}\right\rceil,
\qquad
T_N=\left\lceil\frac{N}{N_{per\ WG}}\right\rceil,
\qquad
T=T_M T_N.
$$

若kernel在WG内部遍历完整N，则$T_N=1$。launch出来但被uniform early-exit的WG不计入
$T$，应静态报告$G_{launch}-T$及其占launch grid的比例。

根据kernel的确定性task-to-CU映射，直接计算每个CU的$n_c$。均匀商余分配时：

$$
q=\left\lfloor\frac{T}{C}\right\rfloor,
\qquad r=T\bmod C,
$$

即$r$个CU各$q+1$个任务，其余$C-r$个CU各$q$个任务。若代码有XCC/SE/CU重排，按
源码映射逐个枚举task，而不是假设硬件自然分配。

令$T$为active workgroup数，$C$为物理CU数，$n_c$为CU $c$承担的active workgroup数：

$$
n_{max}=\max_c n_c,
\qquad
I_{CU}=1-\frac{\sum_c n_c}{C\,n_{max}}.
$$

同时单列完全没有active workgroup的CU比例：

$$
Z_{CU}=\frac{|\{c:n_c=0\}|}{C}.
$$

`I_CU`是相对critical CU的静态容量损失；均匀商余分配可直接写成：

$$
I_{CU}=1-\frac{T}{C\lceil T/C\rceil}.
$$

若任务等成本，对应的理想到实际critical-path膨胀为$n_{max}/(T/C)-1$。必须报告：

- launch WG、active WG、uniform early-exit WG；
- CU数、零任务CU数；
- 静态`tasks/CU`直方图、$Z_{CU}$和$I_{CU}$；
- 使用的task-to-CU映射公式或枚举程序。

ATT在本层只做sanity check，例如确认被采样CU的wave数量与静态预测一致；不能用只采样
CU1的ATT反推全部CU分布。任务成本不等时，静态计数仍必须报告，但它只是容量模型；
另用全设备timeline报告每CU工作时长偏差，不能替换或混入$I_{CU}$。

### 2. CU内SIMD wave不均衡

该层同样必须静态完成。令每个WG包含$W$个wave，每个CU有$S$个SIMD；由kernel的
wave-to-SIMD规则计算一个WG对各SIMD的贡献$a_s$。对CU $c$：

$$
w_{c,s}=n_c a_s.
$$

若$W$能整除$S$且wave均匀映射，则$a_s=W/S$；例如8-wave WG和4个SIMD时，每个WG
静态贡献2 waves/SIMD。若不能整除或起始SIMD会旋转，必须按tile和映射规则枚举每个WG，
不能用ATT反推。wave-count不均衡为：

$$
I_{SIMD}=1-\frac{\sum_{c,s}w_{c,s}}
{\sum_c S\max_s w_{c,s}}.
$$

同时报告零active-wave SIMD比例：

$$
Z_{SIMD}=\frac{|\{(c,s):w_{c,s}=0\}|}{C\,S}.
$$

该式包含零wave SIMD，并按每个CU自己的critical SIMD归一。还要单列resident batch之间
的静态不均衡。令$R$为资源决定的resident waves/SIMD，静态batch数为：

$$
b_{c,s}=\left\lceil\frac{w_{c,s}}{R}\right\rceil,
\qquad
I_{SIMD,batch}=1-\frac{\sum_{c,s}b_{c,s}}
{\sum_c S\max_s b_{c,s}}.
$$

报告静态`waves/SIMD`和`resident batches/SIMD`直方图、零wave SIMD数、
$I_{SIMD}$与$I_{SIMD,batch}$。实际wave duration不同不会改变静态分配；若需要观察
相邻resident batch之间的动态供给空洞，将其作为第3--5层的补充指标$I_{gap}$报告：

$$
I_{gap}=\frac{\sum \text{inter-batch gap cycles}}
{\sum(\text{batch lifetime}+\text{inter-batch gap})}.
$$

ATT在第2层也只用于核对采样CU是否符合静态`waves/SIMD`结果，不能作为主要计算来源。

### 3. 单SIMD prologue

先在每个physical SIMD上按resident slot和生命周期重建wave batch。对batch $b$定义：

$$
t_0=\min_i t_{begin,i},\qquad
t_1=\min_i t_{first\ MFMA,i}.
$$

prologue为$P_b=t_1-t_0$。报告$\sum P_b/\sum L_b$、cycles/batch及
min/p50/p95/max。这里是“physical SIMD首次有MFMA前”的时间，不是把各wave prologue
相加。

### 4. 单SIMD epilogue

令$t_2=\max_i(t_{last\ MFMA\ issue,i}+16)$，$t_3=\max_i t_{end,i}$，则：

$$
E_b=t_3-t_2.
$$

同样报告$\sum E_b/\sum L_b$及分布。它只包含最后一条MFMA结束后的物理尾部；当前N
内部的CShuffle/store空洞属于steady中的`other/structural tail`，不能重复计入epilogue。

### 5. 单SIMD steady

resident batch的物理生命周期和steady span定义为：

$$
L_b=t_3-t_0,
\qquad
S_b=t_2-t_1,
\qquad
P_b+S_b+E_b=L_b.
$$

报告$P/S/E$三项占比，必须精确闭合到100%。随后为第6层选取内部稳定N窗口；通常去掉
首尾N块，但必须由timeline确认，而不是固定照抄`N2..N13`。同时报告该窗口覆盖完整
steady span的比例。

### 6. Steady physical MFMA-union与stall构成

ATT以4 cycles为最小时间粒度。先在每条MFMA的successful issue后标记16-cycle执行窗，
再合并同一SIMD所有resident wave：

$$
B(t)=\bigvee_i MFMA_i(t).
$$

`B(t)=1`记MFMA busy；只有`B(t)=0`才分类。为避免任意16-cycle对齐切断MFMA窗口，先按
4-cycle tick精确累计，最后除以16，报告为“16-cycle等效槽”；因此汇总值允许是小数。

单条ATT记录中的`stall`只是该wave从attempt到successful issue的原始等待，不能直接
当成physical stall。指令$r$对physical账本的贡献必须先与steady union idle求交：

$$
E_r=\left|[t_{attempt,r},t_{issue,r})\cap T_{steady}\cap
\overline{\bigcup_i W_i^{MFMA}}\right|.
$$

例如某条`ds_read2st64_b64`为`attempt=98540, issue=98580`，原始stall是40 cycles；
但peer在98548开始执行MFMA，且98540--98544也已被前一条MFMA覆盖。真正未被MFMA union
覆盖的只有`[98544,98548)`，因此该实例对physical `LDS issue`只贡献4 cycles，而不是
40 cycles。热点PC必须累计这种交集后的贡献，禁止按原始`record.stall`排序后直接解释
为物理损失。

当一个idle tick上多个wave有不同状态时，**整段4 cycles只归给以下最高优先级类别**，
不再在多个owner间等分：

| 优先级 | 主类别 | 包含内容 | 必须附带的子分解 |
| ---: | --- | --- | --- |
| 1 | VMEM issue | VMEM正常发射或发射前stall | service / issue-stall；load / store |
| 2 | VMEM wait | `s_waitcnt vmcnt(...)`；mixed wait也在此归类 | wait阈值、PC、producer距离 |
| 3 | LDS issue | DS/LDS正常发射或发射前stall | service / issue-stall；read / write |
| 4 | LDS wait | `s_waitcnt lgkmcnt(...)` | wait阈值、PC、producer距离 |
| 5 | VALU execution | 成功发射的VALU/TRANS | opcode、PC、所在phase |
| 6 | barrier | `s_barrier`发射或等待 | barrier代次、两wave phase |
| 7 | other | 以下所有剩余状态 | 必须继续展开，不能只报other |

主类别选定后，只在命中该主类别的wave之间等分4 cycles，用于生成该类别内部的
service/stall、load/store、opcode、PC和phase子表；未命中主类别的wave不参与子表。
因此每个子表之和必须等于对应主类别，而七个主类别之和必须等于union idle。

实现时可直接使用以下伪代码，避免把wave级stall求和：

```text
for each (SE, CU, SIMD):
  discover resident slots from trace
  reconstruct resident-wave batches
  for each 4-cycle tick in the selected internal steady window:
    if any resident wave has a 16-cycle MFMA execution window at tick:
      mfma_busy += 4
      continue
    categories = classify_each_active_wave(tick)
    owner = first_present(categories, PRIORITY_ORDER)
    stall[owner] += 4
    split 4 cycles among waves matching owner for opcode/PC/phase details
```

`other`至少拆成：structural tail、VALU dependency、SALU/control、MFMA unavailable、
scheduler ready、SMEM/其他service和无法解释的residual。每个主类别报告：

```text
cycles
16-cycle equivalent slots = cycles / 16
share of steady stall = cycles / union_idle_cycles
share of steady = cycles / steady_cycles
```

主表必须满足：

$$
U_{MFMA}+E_{VMEM\ issue}+E_{VMEM\ wait}+E_{LDS\ issue}
+E_{LDS\ wait}+E_{VALU}+E_{barrier}+E_{other}=100\%.
$$

每次报告必须执行四个断言：

```text
prologue + steady + epilogue == active-batch lifecycle
MFMA busy + MFMA idle == selected steady window
sum(seven exclusive categories) == MFMA idle
sum(subcategories of category r) == category r
```

第6层中的“stall”泛指没有MFMA执行的slot。VMEM/LDS正常issue和VALU execution是必要
服务成本，不是硬件阻塞，因此必须在子分解中与真正的issue stall分开。

### 7. 一页报告与快速判瓶颈

固定按以下顺序输出，不要先展示几十个opcode：

```text
1. CU: active/zero CU，tasks/CU分布，I_CU
2. SIMD: waves/SIMD、batches/SIMD分布，Z_SIMD、I_SIMD、I_SIMD,batch
3. prologue: cycles/batch，占active-batch lifecycle；补充动态inter-batch gap
4. epilogue: cycles/batch，占active-batch lifecycle
5. steady: cycles/batch，占active-batch lifecycle，内部N窗口覆盖率
6. MFMA union: busy/idle cycles、百分比、16-cycle等效槽
7. steady stall: 七类互斥主表；最大两项各给top opcode/PC/phase
8. witness: top joint state与all-waves-same，仅作定位
9. control -> candidate原因转移表
10. clean ABBA24与结论
```

快速决策顺序：

| 最大损失层/类别 | 首先检查 | 首选单变量实验 |
| --- | --- | --- |
| CU不均衡 | active WG数、mapping、zero-task CU | 改tile/grid/task mapping |
| SIMD不均衡 | waves/SIMD、batches/SIMD、resident容量 | 改waves/WG或SIMD映射 |
| inter-batch gap | 相邻batch的end到begin | 检查调度供给或slot replacement |
| prologue | 首个MFMA前的A/B/metadata关键路径 | 提前首批load、合并初始化wait |
| epilogue | 最后MFMA后的CShuffle/store | 分片退休、提前最后store |
| VMEM issue | service与issue-stall、load/store同相 | 错相请求或role/slot priority |
| VMEM wait | load到consumer距离 | 增加预取深度 |
| LDS issue | read/write同相、bank PMC | 拆分DS burst或调整地址/相位 |
| LDS wait | read到consumer距离 | 前移read并插入独立工作 |
| VALU execution | top VALU PC和peer phase | 将短生命周期VALU移入peer MFMA窗 |
| barrier | 两边到达phase、生产消费关系 | 先平衡前置工作；确认安全前不删barrier |
| structural tail | 最后ready MFMA位置 | 跨N overlap或分片退休 |
| VALU dependency | producer/consumer和VGPR生命周期 | 改依赖链，而非仅移动整组VALU |

选择候选时只处理占steady总周期最大的可控项。若该项下降但MFMA union或墙钟没有改善，
说明气泡转移，不算成功。

## 当前权威案例：8x1 K256

本节是当前仓库唯一的MoE 8x1 K256 ATT结果入口。历史候选、旧owner账本和旧脚本结果
不再作为依据。

### 输入身份

- raw trace：`ui_output_moe_8x1_k256_current_dispatch_16/`
- kernel：`moe_2stage_down_prefill_8x1_0`，GPU7，dispatch 16
- 采样：CU1、4个SE、每SE 4个SIMD；408条active wave、128条uniform early-exit wave
- shape：B32768、TOPK8、E256、N2048、K256、BM256、BN128、8 waves/WG
- geometry：16个N block，每N 4个core，每core 32条MFMA，每wave共2048条MFMA
- 源码SHA256：`85a13a748104baae3b9fd73f3936ca611a1e978e4d0e4ece1fe11a7e86de4d9b`
- ISA SHA256：`739617672892eb8035e997d493298e5617c0e1ddfa5f4350011f7aee2777c7df`
- `code.json` SHA256：`e87e0f766ac594dbcbfba27f49f6eb0155fa2232fdded806a56695d84c09560d`

单次ATT dispatch为0.717243ms，资源为60 regular + 132 accum VGPR、112 trace SGPR、
48KiB LDS、0 scratch。该单次时延只标识trace，不替代clean ABBA墙钟结果。

### 唯一复算命令

分析器与本文同目录：[analyze_mfma_stall.py](analyze_mfma_stall.py)。从仓库根目录运行：

```bash
python tests/flydsl/attn_4wave/tools/analyze_mfma_stall.py \
  ui_output_moe_8x1_k256_current_dispatch_16 \
  --n-blocks 16 --cores-per-n 4 --mfma-per-core 32 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 \
  --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 1 \
  --record-attempt 98540 --record-pc-index 1213 \
  --json ui_output_moe_8x1_k256_current_dispatch_16/mfma_stall_report.json
```

脚本只依赖Python标准库和项目已有的NumPy，不依赖`/tmp`文件。它在一个进程中完成：

1. 解析`code.json`和所有wave JSON，校验每条active wave的MFMA数。
2. 静态计算第1--2层，按resident slot重建第3--5层生命周期。
3. 按successful issue和16-cycle MFMA窗计算第6层physical union。
4. 按固定优先级生成七类互斥账本及PC/opcode/phase子表。
5. 可选重算一个32-MFMA stage及一条动态记录的physical贡献。

权威机器可读输出为`mfma_stall_report.json`；本文是唯一人读报告。重新运行后必须得到
下面的关键数值，否则先停止解释并核对trace、geometry和源码hash。

### 七层结果

第1层：launch 1280 WG，其中1024个active、256个uniform early-exit。80个CU中64个各
13个active WG、16个各12个，无零任务CU：

$$
I_{CU}=1-\frac{1024}{80\times13}=1.538\%,
\qquad I_{critical}=1.5625\%.
$$

第2层：每WG的8个wave均匀落到4个SIMD。256个SIMD各26 waves，64个SIMD各24 waves，
对应13或12个resident batch；$I_{SIMD}=I_{SIMD,batch}=Z_{SIMD}=0$。

204个采样resident batch的生命周期为：

| 阶段 | cycles/batch | p50 | p95 | lifecycle占比 |
| --- | ---: | ---: | ---: | ---: |
| prologue | 16,237.35 | 14,082 | 26,475.4 | 16.690% |
| steady | 78,885.82 | 76,628 | 98,764 | 81.083% |
| epilogue | 2,166.55 | 2,152 | 2,375.4 | 2.227% |

三项闭合为19,847,104 cycles。inter-batch gap共99,452 cycles，平均529 cycles，占观察
horizon的0.499%。N2--N13内部窗口为11,909,224 cycles，覆盖完整steady的74.004%。

| 类别 | cycles | 16-cycle等效槽 | idle占比 | steady占比 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 9,922,560 | 620,160.00 | - | 83.318% |
| VMEM issue | 189,156 | 11,822.25 | 9.521% | 1.588% |
| VMEM wait | 354,564 | 22,160.25 | 17.847% | 2.977% |
| LDS issue | 696,084 | 43,505.25 | 35.038% | 5.845% |
| LDS wait | 98,420 | 6,151.25 | 4.954% | 0.826% |
| VALU execution | 276,624 | 17,289.00 | 13.924% | 2.323% |
| barrier | 289,212 | 18,075.75 | 14.558% | 2.428% |
| other | 82,604 | 5,162.75 | 4.158% | 0.694% |

MFMA idle为1,986,664 cycles，七个类别精确闭合。LDS issue进一步拆成352,436 cycles
issue-stall和343,648 cycles正常服务；VMEM issue拆成104,616 cycles issue-stall和
84,540 cycles正常服务。`other`由60,430 SALU/control、20,120 structural tail、
1,250 scheduler ready和804 MFMA unavailable cycles组成。

### 具体32-MFMA stage

选取`se0_sm0_sl0_wv0.json`中的N2/core1，即slot0的MFMA ordinal 288--319：

- physical位置：SE0/CU1/SIMD0/slot0；partner为slot1。
- 动态窗口：`[99052,99648)`，共596 cycles。
- 首条MFMA：PC index 1239，ISA第1632行，源码`gemm2_8x1.py:1024`。
- 末条MFMA：PC index 1316，ISA第1757行，源码`gemm2_8x1.py:1024`。
- 32条MFMA的单wave raw stall之和为196 cycles；它不是physical idle。

| physical union状态 | cycles | 16-cycle等效槽 | stage占比 |
| --- | ---: | ---: | ---: |
| MFMA busy | 512 | 32.00 | 85.906% |
| VMEM issue | 8 | 0.50 | 1.342% |
| VMEM wait | 4 | 0.25 | 0.671% |
| LDS issue | 20 | 1.25 | 3.356% |
| LDS wait | 4 | 0.25 | 0.671% |
| VALU execution | 44 | 2.75 | 7.383% |
| barrier | 4 | 0.25 | 0.671% |

该stage的84-cycle idle由脚本逐tick唯一归类。20-cycle LDS issue全部是正常DS服务，
此stage内没有LDS issue-stall。主要位置是：

- VMEM issue：`[99324,99328)`、`[99360,99364)`；
- LDS issue：`[99364,99380)`、`[99384,99388)`；
- VMEM wait：`[99380,99384)`；LDS wait：`[99512,99516)`；
- barrier：`[99548,99552)`；其余44 cycles为零散VALU服务。

### 98540记录的交集反例

同一wave的PC index 1213为`ds_read2st64_b64 v[4:7], v49 offset1:4`，源码
`gemm2_8x1.py:895`。它在N2的core0->core1边界执行，raw stall区间为
`[98540,98580)`，共40 cycles。前一条MFMA覆盖98540--98544，peer下一条MFMA从98548
开始，因此：

$$
[98540,98580)\cap T_{steady}\cap\overline{\bigcup_iW_i^{MFMA}}
=[98544,98548).
$$

该记录对physical union idle和exclusive LDS owner都只贡献4 cycles。它不属于上面的
`[99052,99648)` stage窗口，不能把40 cycles加进stage或全局physical账本。

### 第7层决策闭环

当前K256 ISA包含4+4 B-read与prologue额外half-B两级carry。两级carry相对4+4基线的
10-buffer clean primed ABBA24为：

```text
4+4 baseline: 0.774583 ms / 354.87 useful TFLOPS
two-level carry: 0.751384 ms / 365.83 useful TFLOPS
candidate/control: 0.96702, IQR [0.96372, 0.96969], 24/24 wins
```

有效工作量为$2\times32768\times8\times2048\times256$
$=274,877,906,944$ FLOP。两级carry的K256 ISA SHA256与当前多K源码的K256 ISA相同，
均为`73961767...`；正确性、资源、墙钟和ATT方向均已闭合。当前trace的83.318% union
低于同ISA另一轮采样的85.241%，说明ATT采样存在波动；它不推翻24轮墙钟结论。

### Witness不是可加速预算

`joint state`和`all_waves_same_reason`只用于定位同时发生的状态，不能再加到主账本。
两者必须在上述七类映射之后统计；若需要展示原始`stall:DS-read`等细粒度状态，只能作为
对应主类别的子表，不能与七类主表混算。
Oracle recoverable假设目标blocker可立即替换为ready MFMA，只是上界见证，不是性能
预测。每次候选必须给出control到candidate的原因转移，确认不是把气泡搬到另一类别。

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

若候选跨越VGPR/LDS occupancy门槛，后续stall分布已不是同资源实验，必须单独解释。

### 2. 先用clean ABBA确定是否值得解释

正式协议使用10 rotating buffers、24轮ABBA；每版本有48个绝对样本和24个paired ratio。报告：

```text
control ms -> candidate ms
candidate/control median
ratio Q1..Q3
wins/rounds
```

短ABBA只淘汰明显回退。外部任务占用时只可将同进程ratio标成stress证据，不能替代clean结果；idle gate拒绝运行是正确行为。

### 3. 采集或选择raw ATT

后处理的输入目录必须至少包含`code.json`和完整的
`se<SE>_sm<SIMD>_sl<SLOT>_wv<WAVE>.json`。每个候选使用新目录，禁止用旧trace解释新ISA。
先记录kernel名、dispatch、源码/ISA/code哈希、shape、GPU/PTL/DPM和外部负载，再分析。

当前仓库完整保存了本节K256 raw trace，因此**从raw trace到报告可独立复现**。重新采集
同一workload仍依赖机器上的MoE launch环境，不由本分析脚本负责；采集时使用的ATT配置为：

```yaml
kernel_include_regex: "^moe_2stage_down_prefill_8x1_0$"
kernel_iteration_range: "[2]"
advanced_thread_trace: true
att_target_cu: 1
att_shader_engine_mask: "0xf"
att_simd_select: "0xf"
att_buffer_size: "0x60000000"
```

采集前必须通过全机idle gate，GPU7固定1800MHz determinism、PTL
`Enabled / VECTOR,F8`、650W、NUMA off；无论成功与否都恢复原状态。先用kernel trace
确认匹配序号，不能假设`[2]`对不同launch脚本仍指向同一dispatch。

### 4. 运行唯一分析器

使用上文的[统一复算命令](#唯一复算命令)。参数必须来自实际geometry，不能照抄K256：

- `n-blocks * cores-per-n * mfma-per-core`必须等于每条active wave的动态MFMA数；
- `resident-waves`必须与资源和trace中的同时active slot数一致；
- `first-n..last-n-exclusive`应排除首尾非稳态块，并报告其steady覆盖率；
- control和candidate必须使用相同geometry和steady定义。

脚本会强制检查active wave MFMA数、lifecycle闭合、union busy/idle闭合、七类owner闭合和
类别子项闭合。任何断言失败都意味着模型不适用或输入不完整，不能继续解读百分比。

### 5. 用joint phase定位可改代码

owner只回答“谁占了空槽”，joint state回答“哪些wave阶段同时发生”。对热点PC检查：

1. 它是请求发射、completion wait还是普通issue？
2. peer waves处于`core0/core1/core2/boundary/tail`哪一阶段？
3. producer到first consumer还有多少successful-issue距离？
4. 是否所有producer边都已被计数器顺序有效的wait覆盖？
5. 该PC能否移动而不增加指令、wait、寄存器或barrier？

只有在单条记录与union idle求交后，热点PC才可进入候选列表。先区分service与真正stall，
再结合peer phase判断是请求同相、completion latency、依赖链还是结构性tail。

### 6. 只做一个可证伪改动

候选应直接对应一个owner假设：

| 假设 | 最小候选 | 可证伪检查 |
| --- | --- | --- |
| 两slot VMEM同相 | slot/role priority或固定phase rotation | same-reason VMEM下降，wait不反弹 |
| DS read消费距离不足 | 在read与wait间放独立pack/write | lgkmcnt owner下降，指令数不增 |
| CShuffle tail过长 | 单slice分片退休probe | 下一N首批MFMA真实进入旧tail |
| occupancy是主因 | 仅跨一个VGPR/LDS门槛 | resident waves增加且墙钟改善 |

禁止一次同时改tile、priority、store policy和padding；否则ATT即使变好也无法归因。

### 7. 重新执行四层闭环

保留候选必须同时满足：

1. 正确性不退化，finite/tail/padding全部通过。
2. ISA工作量与资源变化已解释，没有意外spill。
3. clean ABBA24的ratio/IQR/wins稳定改善。
4. fresh ATT中目标owner、physical union busy和墙钟方向一致。

若owner下降但union/墙钟不变，结论是“暴露转移”，不是成功。若墙钟改善但ATT未解释，先检查trace dispatch、PTL、代码SHA和steady窗口，不补故事。

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

## 文件职责

- 本文：唯一方法、复现命令、当前权威K256结果和具体样本。
- [analyze_mfma_stall.py](analyze_mfma_stall.py)：唯一后处理实现。
- `ui_output_moe_8x1_k256_current_dispatch_16/`：raw trace及脚本生成的唯一JSON账本。
- [design_moe_gemm_8wave_down.md](../../../../design_moe_gemm_8wave_down.md)：kernel设计与性能演进；不再复制ATT账本。
