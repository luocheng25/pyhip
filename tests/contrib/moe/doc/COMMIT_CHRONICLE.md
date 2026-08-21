# MoE Down优化提交编年史

## 范围与读法

本文覆盖线性历史 `a6a1632a40d2b520261b0380d8d2896e671f6df3^..9049ddb723a1428d8dfb4c75e352d9b65bc9db56`，即从`a6a1632`（含）到当前分支HEAD（含）的42个commit；最后一节补充当前未提交工作，不把工作树内容伪装成commit。

状态含义：

- **保留**：方法进入后续主线，可能又被更优实现替代。
- **替代**：当时有效，但后续实现改变了输出契约、tile或调度。
- **拒绝**：正确性、资源或稳定ABBA门禁未通过。
- **分析**：主要增加证据、模型或handoff，不直接改变生产算法。

性能数字只在原提交协议内比较；不同harness、PTL或外部负载下的绝对时间不可横向相减。

## 阶段A：N256、输出布局与shape selector（2026-08-13至08-15）

| Commit | 方法与目标 | 关键结果 | 状态 |
| --- | --- | --- | --- |
| `a6a1632` optimize H3 down output pipeline | 引入4-wave BN256 physical down、K-stage流水、wave-contiguous输出与physical `sorted_sum`，消除N64 producer固定开销。 | down由约`1.9504ms`降到`1.5821ms`，但consumer增大；优化consumer后约`1.5645+1.1779=2.7424ms`。H3 `diff=0.00105974`。 | **保留骨架**；wave-contiguous输出后来被row-major CShuffle替代。 |
| `5311831` balance down and reduce layouts | 扫描0/32/64/128B row padding；设计R2 tile-major AoSoA，平衡producer写合并与consumer gather。 | 128B row-major比physical base combined快12.3%；R2正式约`2.3069ms`，比base快16.1%、比pad128快3.4%。 | **替代**；padding按shape选择的原则保留。 |
| `8bee001` optimize down LDS and document roof | 用wave-private XOR CShuffle恢复标准row-major；A-LDS改`Swizzle(3,4,4)`；scale load先于weight load。 | CShuffle combined `2.230630ms`，比R2快2.84%；A swizzle ratio `0.974474`；scale-first ratio `0.984869`；最终down约`1.516768ms/407.8T`。CShuffle没有新增bank-conflict计数。 | **保留**。 |
| `8a4b49c` select down layouts by shape | 按完整gateup tile、量化和实际LDS occupancy选base/physical；修复Hy3不完整BN256 gate tile。 | Hy3 physical combined当时快6.44%；Qwen K512因LDS跨32KB台阶回退，继续base；Xiaomi physical优于legacy。 | **保留selector框架**；具体边界后续多次修正。 |
| `00b01c3` select down layouts by end-to-end cost | 用生产式10-buffer和48个配对ratio，以`down+sorted_sum`而非down-only决策。 | Hy3 physical down快9.80%，但sum慢45.70%，combined回退5.96%；Xiaomi/H3 combined改善11.04%/6.68%。 | **保留决策原则**；Hy3暂回legacy。 |
| `39c5390` tune Hy3 down row padding | 用真实TOPK9路由比较padding，区分随机loc与真实跨行gather。 | Hy3 0B使combined `3.081979→2.919034ms`，改善5.41%；32/64/128B分别回退13.74%/12.68%/5.86%。 | **保留**：Hy3用0B，H3/Xiaomi多用128B。 |
| `1d21377` remove unused down layouts | 删除BN128/192、direct physical、R2、physical inverse/prefetch等未晋升路径，收敛为BN256+CShuffle row-major。 | `102 insertions/624 deletions`，无新性能变化。 | **保留清理**。 |
| `d32016c` analyze Hy3 down efficiency | 用K192/K384同协议ATT/PMC分解固定成本、occupancy和MFMA利用率。 | K192 `1.834509ms/252.85T`，K384 `3.039535ms/305.22T`；MFMA均16 cycles/inst，HBM仅峰值约20%至30%；固定截距约`0.629483ms`。 | **分析**：K192短core固定成本主导。 |
| `f3b0525` correct Hy3 resource counts | 纠正“160 VGPR”为最终ISA next-free 150/accum offset 152，运行资源24V+128A。 | 0 spill；无性能变化。 | **分析纠错**。 |
| `b9c2750` extend Hy3 K scaling analysis | 扫K192至K1024，并在同K640、同LDS/occupancy下比较K128与K64 core。 | K640 K128比K64时延低7.34%、吞吐高7.93%；K512出现1 wave/SIMD资源台阶；K1024超64KB LDS编译失败。 | **分析**：较长K128 core更能摊薄stage/operand-ready开销。 |
| `d73ea19` fuse Hy3 down scales | 融合activation/weight/routing scale，消除64条逐输出MUL；配合K128+K64两stage。 | `1.833768→1.760709ms`，paired ratio `0.96677`；ISA 628→555，VGPR 150/152→144/144；完整11项通过。 | **保留**。 |
| `9aa595d` rename physical down N256 path | 将误导性的N128名称改为真实N256，统一API、环境变量和测试。 | 51行对51行纯重命名，无行为变化。 | **保留命名修正**。 |
| `1e637f6` optimize Hy3 K384 down for three waves | 先用`sched_group`改K128 interleave；再将K384改为6×K64，K0-K3常驻16KB，K4/K5借4KB CShuffle scratch，跨过3-wave门槛。 | 稳定`VECTOR,F8` ABBA24：`2.222207→2.131987ms`，ratio `0.959404`，吞吐+4.23%；20,480B LDS、168 VGPR、0 spill。 | **保留**；受污染窗口的+10.95%不作为主结论。 |

## 阶段B：从ATT空槽到K128 P7（2026-08-16）

| Commit | Stall假设与候选 | 关键证据/结果 | 状态 |
| --- | --- | --- | --- |
| `0f0a14c` analyze Control K128 stall exposure | 修正ATT successful issue语义，合并同SIMD两resident wave，建立physical MFMA-union/owner/joint-state账本。 | Control MFMA union 71.78%；VMEM issue owner 13.04%，两wave同因9.8934%；单waveMFMA stall被peer 100%覆盖。 | **分析基线**：优先恢复两waveVMEM错相。 |
| `46bcf9e` restore K128 slot priority | read/write阶段按hardware slot设`1/0`，compute设3，尾部归0；只改变仲裁。 | 24轮 ratio `0.986906`，时延-1.31%；VMEM同因9.8934%→3.5875%，MFMA union约+5.49pp。 | **保留P0**。 |
| `4e5b914` Optimize K128 CShuffle LDS wait | 删除同wave DS write后的冗余`lgkmcnt(0)`，保留read结果wait。 | `lgkmcnt(0)` 24→16；ratio `0.986942`；DS wait约-2.05pp。 | **保留P1**。 |
| `406906d` Pipeline K128 CShuffle reads | 在当前LDS read和wait之间放入下一slice pack/write，拉长read消费距离。 | ratio `0.991734`；DS同因0.5712%→0.1263%；资源档位不变。 | **保留P2**。 |
| `b154b2d` Document rejected CShuffle schedules | 扫双slice分组与单read-ahead。 | 双slice ratio `1.002490`；read-ahead `0.998282`但IQR跨1。 | **拒绝**。 |
| `027c4b7` Use non-temporal K128 stores | gfx942 raw-buffer store加NT aux，减store及后续weight-load队列背压。 | ratio `0.987811`；store stall约-18.66%，后续load stall约-15.88%。 | **保留P3**。 |
| `afa6864` Document rejected K128 tail rewrites | 扫跨N BF16 carry、LDS carry、packed FMA、半packed与分片退休。 | ratios依次约`1.0065/0.9979/1.0131/1.0165/1.0158`；部分虽维持2-wave但寄存器压力或调度退化。 | **拒绝**。 |
| `833fa1c` Close K128 two-wave optimization scan | 扫NT sc1、tail slot反转和VMEM/MFMA间隔。 | `1.018043/1.010352`；间隔4→5生成相同ISA。P3约439.5T。 | **阶段收口**。 |
| `1d1d60c` Record K128 full down regression | 对P3执行完整设备回归。 | 11 passed。 | **验证**。 |
| `7fb9107` Halve K128 CShuffle DS writes | 64 lanes共同生产M16 row-pair，DS write 16→8。 | ratio `0.974461`，24/24胜；tail约-163 cycles/N；LDS 28,672→32,768B但仍2-wave。 | **保留P6**。 |
| `755f72c` Pipeline K128 row-pair reads | 每pair先发两条read，再`lgkmcnt(1)->store0->lgkmcnt(0)->store1`，避免store burst。 | ratio `0.990328`，20/24胜；MFMA union 80.62%→81.67%；read wait 514→273 cycles/N。 | **保留P7**。 |
| `43bd4da` Close post-P7 scheduling scan | 扫整K0/K1预取、首atom、合并core、next-pair write、equal priority。 | 多数ratio `>1`；两个约0.99候选IQR上缘>1。 | **拒绝并收口**。 |
| `c17a1c2` Close P7 pair-order scan | 根据首/末pair wait差异反序CShuffle pair。 | ratio `0.992165`但Q3 `1.015624`，LLVM同时重排FMA/寄存器。 | **拒绝**。 |
| `031177b` document P7 R2 rejection | P7 producer改R2 AoSoA并配consumer逆映射。 | down `1.006482`，consumer `2.319782`，combined `1.359892`。 | **拒绝**：consumer局部性损失主导。 |
| `faedc0e` document P7 atomic scatter rejection | packed-BF16 atomic直接scatter，省reduce。 | atomic down约7.41ms，另需清零；P7 down+sum约2.777ms。 | **拒绝**。 |
| `333cdc9` document P7 route-major rejection | producer直接按`[token,topk,N]`scatter，consumer identity读。 | consumer快约4.26%，但producer约2.94至3.19ms，完整链路慢于P7。 | **拒绝**。 |
| `b13a3d9` document P7 INT8 compression rejection | per-32 INT8把行宽降到BF16的53.125%。 | consumer快约33%，但combined ratio `1.180187`；rel_l2约0.604%。 | **拒绝**。 |
| `23d4e52` close P7 packed INT8 scan | 扫packed FP8、packed-U8 per-32/per-8及转换微基准。 | FP8误差2.38%淘汰；per-32 ratio `1.134499`；per-8误差0.483%但ratio `1.029974`。 | **拒绝并关闭压缩方向**。 |

## 阶段C：8-wave、paired、XCC与single-M（2026-08-17至08-21）

| Commit | 方法与wave/tile结构 | 关键结果 | 状态 |
| --- | --- | --- | --- |
| `238f871` unified 8-wave checkpoint | 512线程，统一`(8,1,1)`，M64xN512；严格per-K反相；CShuffle阶段`0/0/4`。 | strict→0/0/4：down ratio `0.986555`，combined `0.984826`；100V+132A、40KB、0 scratch、2 waves/SIMD。 | **检查点**；0/1/3、defer2未晋升。 |
| `7600ec0` strict PA XCC-local tuning | 转为paired M128：两个4-wave M64xN256组；按XCC/SE rotation-2映射2316 WG。 | 对physical4 ratio `0.891783`，483.27T；但比固定绝对线慢24.154us。 | **候选保留**，未过绝对线。 |
| `c82e2df` promote strict PA exact-map best | exact-map+drain priority，非目标compile-time fallback。 | 对physical4 ratio `0.890060`、24/24；254 VGPR、49,152B、0 scratch；仍比固定线慢17.494us。 | **晋升packed exact-H3 producer**。 |
| `b415273` optimize and validate all down paths | expert-safe row-major direct epilogue、physical4多M block修复、batch1/splitk VMEM crossing。 | direct row-major相对physical4 `1.232053`、相对packed `1.363678`；batch1 FP8 `0.956752`，splitk `1.865054`。 | **仅保留batch1 crossing和索引修复**。 |
| `bbaae3e` expand paired 8-wave | paired覆盖K64-K512和三种FP8量化；修PTPC scale ownership；K512 direct scale。 | 48/48正确；generic combined无赢家；K512 256 VGPR、19 spills、80B private，不auto。 | **保留显式能力，不做generic auto**。 |
| `a452743` optimize H3 paired row-major pipeline | wave-private CShuffle流式转row-major，保持4+4 stage，不增加WG barrier。 | producer比packed慢8.23%，但consumer 3.6740→0.7415ms，combined降52.27%；相对physical N256 combined ratio `0.96644`。 | **保留exact H3 per-tensor**。 |
| `2b50372` defer H3 PTPC scale loads | K384 direct-scale移到MFMA后epilogue即时读，移除loop-carried scale fragment。 | 18 spills/76B private→0；对Base combined ratio `0.95473`，对physical N256 `0.79605`。 | **保留exact H3 PTPC**。 |
| `5161224` optimize Hy3 single-M N512 | M64 sorting；一个512-thread WG处理一个M64；8 waves连续覆盖N512；无paired metadata复制与10个barrier。 | 128 VGPR、32KB、0 scratch、4 waves/SIMD；N256→single-M down ratio `0.906541`，combined `0.935520`。 | **保留exact Hy3 K192**。 |
| `f133cc0` add recovery handoff | 记录算法、资源、失败项、命令、GPU状态协议；状态工具支持AMDSMI fallback。 | busy GPU在改状态前拒绝，`state_unchanged=true`。 | **分析/handoff**。 |
| `4abc3cb` record clean Hy3 retest | 独立clean 10-buffer ABBA24复现single-M。 | down ratio `0.909806`，combined `0.937691`，consumer `1.000015`。 | **验证**：收益来自down。 |
| `9049ddb` tune Hy3 N512 | role priority：read/prefetch=1、MFMA=3、完整epilogue=0；slot分析扩展2/4 resident waves。 | clean ABBA4 down `0.975036`、combined `0.980920`；128 VGPR、32KB、0 scratch不变。ABBA24被外部作业idle gate拒绝。 | **当前HEAD候选**，相对516缺clean ABBA24正式晋升。 |

## 当前未提交检查点（不属于上述42个commit）

当前工作树还包含以下未提交成果，必须与commit历史一起阅读：

1. **M64xN512 N-split实验（59dd stash）**：两个独立4-wave N256组共享M64 activation，构成M64xN512；增加row-major/XCC映射、K192 immediate-store、K384 delayed CShuffle、K512 direct packed-store和Xiaomi双M64 persistent。
2. **六个顶层入口重构**：legacy `_1x4`、`_N256_1x4`、`_2x4`、`_1x8_2`、`_1x8`、`_1x8_persistent`，共享本地copy/MMA/scale/epilogue/K-stage helper；新入口不调用`fxh`。
3. **修复项**：N-split PTPC双group scale LDS隔离、K64-K512合法copy线程、短H3前缀映射、sparse grid的M64任务单位、gateup/down metadata展开契约、K512偶奇N256地址、finite/NaN显式门禁。
4. **性能等价与修复收益**：P1-P5的六个原始代表case重构前后最大中位偏差0.275%；P0 legacy未单独计时。Qwen K512因scale竞争修复，35B改善0.75%/0.92%，397B改善1.98%/2.73%。详见[重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)。
5. **H3 PTPC四算法报告不代表当前P2**：报告中慢于legacy 56.87% down/42.55% combined的paired源码是`a452743`，早于`2b50372` late-scale零spill修复。当前适用的clean证据仍是late-scale P2/Base combined `0.95473`、24/24胜；重构后P2与9049等价。保留paired auto，后续只需用当前源码clean复验，不能据旧P2结果关闭selector。
6. **H3 PTPC低风险资源缩减**：两个反相4-wave组共享streaming CShuffle scratch，LDS从64KB降到56KB；两次ABBA24的combined ratio为`0.99607/0.99618`，但IQR上缘略跨1，因此只作为资源改善，不定义新吞吐档位。
7. **真实`4N x 2M`原型被否证**：M128xN256使用4条N-wave乘2条M-wave、2 barriers、8 stores且0 spill，仍在H3和Xiaomi分别输`1.01448` combined与`1.09436/1.06540` down/combined。少barrier和0 spill不足以弥补独立WG latency hiding减少。
8. **单CU ATT不代表整卡尾部**：在零padding、同任务量实验中，paired采样CU的MFMA union优于physical（90.73%对86.29%），整卡却慢`1.01346/1.01946`。根因是1024个M64任务形成13条physical与14条paired critical waves/SIMD；改为1280个平衡任务后，paired转而以`0.96455/0.97949`胜出。
9. **72-cell falsification限定dispatch-tail解释力**：平衡任务数使down ratio中位从`1.1280`改善到`1.0600`，但60个原回退cell仅11个翻转；dispatch-tail通常贡献约5个百分点，却不是短N rendezvous、输出转换和K512资源回退的唯一原因。
10. **K512 P4 CShuffle被保守门禁误排除**：当前判定固定使用`2 * BM * K`的双M activation预算，但N-split运行时`paired_m_groups=1`。P4 K512实际估算为32KB A + 16KB CShuffle + 2KB双组PTPC scale = 51,200B，低于64KB；当前34,816B direct-store ISA不是容量下限。该row-major CShuffle候选尚未编译、验证或计时。

## 来源索引

- [主README历史](../README.md)
- [当前未提交8-wave TODO](../../../../docs/UNIFIED8_DOWN_TODO.md)
- [Hy3 single-M handoff](../../../../docs/HY3_SINGLE_M_N512_HANDOFF_TODO.md)
- [8-wave PA中间上下文](../../../../docs/UNIFIED8_PA_INTERMEDIATE_CONTEXT.md)
- [四算法clean复测](../results/down_variants_20260821_retest/REPORT.md)
- [59dd与9049直接对比](../results/current_vs_9049_20260821/REPORT.md)
- [重构性能报告](REFACTOR_PERFORMANCE_REPORT.md)
