# SWA+sink：8-wave 与4-wave差距分析

2026-09-05，MI350X gfx950 GPU0，256 CU，BF16 page64 SHUFFLE-5D；除任务数扫描外，
`B1/Hq16/Hkv1/Q16384/KV131072/Dv128/window_left128`。sink逐head FP32 -1→1。
分析基线是第二阶段提交 `3633441`，不使用历史8-wave。

## 结论：不是单纯任务数造成的不均衡

1. **增加任务数后差距仍在。** Q从4096增到65536，8-wave static仍慢16.7%～21.4%。
   Q16K有1024个8-wave CTA（256 CU的4轮），4-wave有2048 CTA，但两者都是8192个wave。
   不能只用“8-wave任务数少一半”解释差距。
2. **每行进入MFMA的窗口并集更宽。** BM256覆盖384个KV token，BM128覆盖256个。
   同一wave32行的真实窗口并集最多160个token，但8-wave执行完整CTA并集。
   三次全grid PMC确认8-wave **1966080** 条MFMA，4-wave **1310720**，准确为**1.5倍**。
3. **短窗口无法充分摊薄OPUS长流水的固定成本。** 8-wave只有6个BN64 tile，但仍有
   prologue、5个主phase和epilogue；ATT每wave **53 barrier**，4-wave为**13**。
   8-wave prologue/epilogue约24.9K cycles，占采样CTA约44.85K cycles的一半以上。
   这是对该短窗口的测量，不外推到长full attention。
4. **不是8-wave LDS bank conflict或static spill。** 8-wave此配置无spill、bank conflict0。
   小任务量确有调度/填充成本，但大任务、persistent及ATT证据不支持把全部差距归于尾波不均。

因此按照任务要求，尝试了不改任务划分、不删除barrier的内部工作量优化。没有修改4-wave，
也没有恢复gather、切换到4-wave fallback或以缺失功能换速度。

## 任务规模扫描（修改前）

固定D192/KV128K/W128/H16，完整FP32 reference后，同进程5轮20/100 GPU-profiler采样。
五轮中位数，**µs**；100轮共同预热，预分配输出，无LSE。

| Q | 8 static | 8 persistent | 4 static | 4 dynamic | 8static比4static慢 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 22.005 | 23.447 | 17.502 | 21.864 | 25.7% |
| 4096 | 28.934 | 31.985 | 24.607 | 31.329 | 17.6% |
| 8192 | 59.681 | 60.574 | 49.506 | 52.805 | 20.6% |
| 16384 | 116.045 | 113.638 | 95.590 | 102.360 | 21.4% |
| 32768 | 232.734 | 227.819 | 199.378 | 207.589 | 16.7% |
| 65536 | 444.517 | 428.260 | 368.904 | 389.788 | 20.5% |

原始样本：[swa_scaling_results.json](swa_scaling_results.json)。不能跨轮拼接这些数与后文新测
优化样本计算加速比，A/B在每次测量内比较。

## PMC 与 ATT

| counter，Q16K | 8 static | 8 persistent | 4 static | 4 dynamic |
|---|---:|---:|---:|---:|
| SQ_WAVES | 8192 | 2048 | 8192 | 2048 |
| SQ_INSTS_MFMA | 1966080 | 1966080 | 1310720 | 1310720 |
| SQ_INSTS_LDS | 2064384 | 2073600 | 1081344 | 1091584 |
| SQ_INSTS_VMEM_RD | 425984 | 466944 | 917504 | 950272 |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 786432 | 786432 |

每项三次相同。VMEM指令数不是HBM字节，不据此声称8-wave HBM流量更小。
两边相同的8192wave也不意味着occupancy相同：8-wave LDS限制每CU一个CTA，4-wave资源更小。

原始ATT在SE0/CU1、SIMD0–3各三次，8wave/4wave分别96个完整wave；逐文件验证
`num_stitched == num_insts`。8wave还验证每CTA8wave起始/slot及53个barrier完成时间对齐。
全部PMC原始三次样本及CSV hash见 [swa_pmc_results.json](swa_pmc_results.json)。

| ATT每wave或每CTA观测 | 8 static | 4 static |
|---|---:|---:|
| MFMA / wave | 240 | 160 |
| barrier / wave | 53 | 13 |
| 动态指令 / wave | 2928 | 2841.5 |
| mean wave span / cycles（3次） | 42464 / 44190 / 43337 | 31976 / 32235 / 32345 |
| 各SIMD sampled resident union / observed span | 99.45%～99.78% | 100% |
| 8wave CTA prologue / main / epilogue均值cycles | 16711 / 19996 / 8143 | 不作8wave式分组 |

驻留覆盖指在**采样wave首次开始到最后结束**范围内至少一个wave驻留，不是全卡occupancy
counter，也不包含完整dispatch前后尾部。结合多任务规模扫描，只排除“主要由任务数不足
导致”的解释，不宣称全卡没有任何不均衡。

ATT `Latency=Stall+Issue` 不是MFMA执行延迟；wave stall互相重叠，**不能加总归因全局耗时**。
32cycle MFMA union仅为明确标记的模型。仪器化span不拿来替代正常benchmark。
分析入口：[analyze_swa_att.py](analyze_swa_att.py)；采集入口
[profile_scheduling.py](profile_scheduling.py)。原始分析：
[swa_att_8wave_results.json](swa_att_8wave_results.json)、
[swa_att_4wave_results.json](swa_att_4wave_results.json)。

## 尝试及收敛范围

仅在某个wave的32个query行与当前KV tile完全不相交时，跳过该wave的QK/PV `fx.gemm`。
predicate是wave-uniform，score继续使用原mask，PV保持原accumulator。所有全局/LDS读取、
V尾页NaN清理、scale/softmax、scheduler fence及workgroup barrier均保留。

不能无条件启用：

- D192W128大grid初轮static约快1%～3%，persistent约快1%；D128W128没有稳定收益。
- 宽W512/W1024全启用会慢约5%～14%；分支破坏流水调度，mask工作占比也降低。
- Q2K/Q4K全启用会慢约6%～12%，跳过部分wave的MFMA并不能让CTA barrier更早完成。

所以最终编译期门限为：**B=1、`ceil(MAX_Q/256) * H >= 1024`，D128窗口0–64或D192窗口
0–128**。其他情形保留原来的无predicate GEMM，不修改任务数/负载均衡算法。
这是当前实测范围的保守策略；不能保证未测试所有shape都有收益。

初期不受限实验（不是最终dispatch策略）的原始数据保留在
[swa_gate_window_results.json](swa_gate_window_results.json)、
[swa_fix_scaling_results.json](swa_fix_scaling_results.json)；后者清楚保留小grid回退证据。
最终门限复测见 [swa_fix_window_results.json](swa_fix_window_results.json)、
[swa_fix_small_grid_results.json](swa_fix_small_grid_results.json)。
早期相同D192计算门控、但尚未限制任务数时的KV扫描与ATT单独保留在
[swa_fix_kv_results.json](swa_fix_kv_results.json)、
[swa_att_gated_results.json](swa_att_gated_results.json)，其source hash不是最终版本；
最终结论使用下一节重新采集的结果。

## 最终验收与收益

最终源码SHA256 `975757800802f8b4d30ebd325a0fc0763a8b0d5346c329836b2e7c2b8b553c69`。
完整回归 **272 passed / 6 skipped**；4-wave默认回归 **51 passed / 2 skipped**。
新增36项覆盖窗口门限两侧、1024任务门限两侧、非对齐diagonal、NaN-tail、全遮罩wave、
static/persistent、nonunit scale/lazy max，以及大grid非法窗口外页表和精确sink分母。
没有修改容差或跳过数值失败。原始验收索引见 [swa_validation.json](swa_validation.json)。

最终同进程A/B，Q16K、W128、BF16 page64、原始5D输入，**µs**：

| Dqk | KV | 修改前static | 最终static | 修改前persistent | 最终persistent | 4static |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 32K | 100.329 | 100.882 | 97.410 | 97.398 | 82.135 |
| 128 | 64K | 100.367 | 101.294 | 98.427 | 98.189 | 82.042 |
| 128 | 128K | 100.688 | 100.783 | 98.189 | 97.926 | 82.040 |
| 192 | 32K | 116.148 | 113.629 | 113.697 | 112.604 | 95.896 |
| 192 | 64K | 116.239 | 114.028 | 113.873 | 113.078 | 95.240 |
| 192 | 128K | 116.549 | 114.888 | 114.151 | 112.884 | 95.082 |

**D192 static降低1.43%～2.17%，persistent降低0.70%～1.11%**。D128/W128不启用此优化，
其小幅正负差为测量波动，不计为收益。所有五轮样本、err、源码hash、有效TFLOPS/逻辑字节
见 [swa_fix_final_results.json](swa_fix_final_results.json)。更窄窗口的反序确认轮：

| Dqk / window_left | static前→后 µs | persistent前→后 µs |
|---|---:|---:|
| 128 / 0 | 78.991 → 76.619 | 76.171 → 73.105 |
| 128 / 64 | 90.330 → 87.518 | 87.184 → 87.585 |
| 192 / 0 | 91.031 → 86.614 | 88.794 → 83.050 |
| 192 / 64 | 104.235 → 99.786 | 101.479 → 98.137 |

D128/W64 persistent这一轮+0.46%，前轮-0.35%，不宣称所有门限内配置必定加速。
最终排除的小grid波动范围约-1.20%～+0.99%；宽W512约-0.37%～+0.20%，已消除全启用
实验的明显回退。源代码门限是静态选择；运行时metadata必须符合原公开接口的最大长度约束。

NC/causal主路径A/B及新鲜ISA在 [swa_fix_primary_results.json](swa_fix_primary_results.json)：
static四种full配置指令及操作数逐条一致；persistent三种逐条一致，D192causal仅一条
`s_add_i32`两个可交换source操作数对调。所有full的mnemonic数量与资源完全不变。
D128/W128仅少量mask地址计算重排，mnemonic数量/资源不变。full计时变化均在约1%内。

最终D192/W128全grid PMC：MFMA **1966080→983040**（-50%），static LDS2064384、
VMEM425984、wave8192、bank conflict0均不变；persistent MFMA同样减半，仍2048wave。
最终static为256VGPR/76SGPR、LDS149760B、private0；persistent为256VGPR/100SGPR、
LDS149764B、private0，原persistent的4个VGPR spill不再出现。**只减少无用MFMA，
不是把全部memory/softmax/同步工作减半。**

最终ATT再采三次共96完整wave，MFMA **240→120/wave**，barrier保持53。
仪器化wave span约**50.2–51.1K cycles**，反而高于修改前42.5–44.2K；动态指令约
3075–3076（原2928），分支/移位及barrier等待抵消了不少收益。保留这一观测，不用“MFMA减半”
或ATT span推断全局加速；正常profiler A/B才是上述1%～2%的性能证据。完整记录见
[swa_att_final_results.json](swa_att_final_results.json)。

## 剩余差距与建议

跳过完全不可见的MFMA不会消除CTA范围的KV DMA、LDS read、softmax固定调度、prologue或
53个barrier，因此只获得小幅改善，**没有达到4-wave SWA性能**。盲目删workgroup barrier
会破坏cooperative LDS流水，不采用这种“修复”。

建议：SWA优先使用现有4-wave；8-wave主要用于长full attention。H3可启用persistent。
若继续专项SWA优化，应另行设计更窄query tile/分组独立的短流水，并对所有stage数据依赖
重新验证，而不是只增加grid或复制4-wave调度参数。任务数不足的小请求保留原static路径。