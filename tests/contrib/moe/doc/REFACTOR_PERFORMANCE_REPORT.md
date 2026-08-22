# MoE down 重构前后性能等价验证（2026-08-21）

## 1. 结论摘要

| 范围 | 结论 |
| --- | --- |
| 六个原始代表case | 覆盖P1-P5；P0 legacy未单独计时。配对中位数最大偏差：Down 0.179%，Combined（Down + `sorted_sum`）0.275%；均小于1%，性能一致。 |
| 两个Qwen K512 case | Qwen3.5 35B：Down改善0.75%，Combined改善0.92%；Qwen3.5 397B：Down改善1.98%，Combined改善2.73%。 |
| 正确性 | 8个JSON的reduced `rel_l2`均为finite，inactive tail clean（未写）；当前70项GPU回归另有显式`isfinite`门禁。JSON未单列finite字段。 |

两个Qwen case的收益不是纯重构噪声，而是重构期间修复N-split PTPC双group scale LDS竞争后的收益。

## 2. 测试基线与协议

| 源码版本 | SHA | 覆盖路径 |
| --- | --- | --- |
| 9049 control | `4951a4878bbd290a8dce702180675a545b9478ba85039c4be6421ba360cb280c` | P1 `n256`、P2 `paired`、P3 `true8` |
| 59dd control | `59ddf290a4820a1e02bfe3cec80ff7748e2db2c9b55e30b2a1dd2c5448188398` | P4 `nsplit192`/`nsplit`/`qwen35`/`qwen397`、P5 `persistent` |
| 重构后 | `b93faa818a602d19645e30cb6fbe27f2231ca3ebbf892c93fc7037df5c196b6c` | P0-P5；性能case覆盖P1-P5 |

- gfx942 / 80 CU；GPU1、GPU2、GPU7；B32768。
- 24轮ABBA：偶数轮old-new-new-old，奇数轮new-old-old-new。
- 每个版本、每个phase采集48个绝对时间样本和24个配对ratio。
- 每次down前运行相同gateup；使用2个轮换buffer。
- 测试时进入1800MHz performance determinism，PTL为`Enabled / VECTOR,F8`。
- 每个case计时前GPU busy不超过2%，计时后瞬时busy不超过5%；结束后恢复`auto`。

初始VRAM为79%，超过正式门禁20%，且外部作业占用使本次无法使用正式10-buffer协议。因此绝对时间受污染，不能与历史绝对时间横向比较；同进程、同输入、交错顺序的old/new配对ratio仍然有效，并由ISA资源和关键指令对照辅助判断。

## 3. 算法与Kernel总览

表中的M/N分别指逻辑输出矩阵的token维和channel维；gfx942上一个wave为64线程。旧control的全部算法分支均埋在旧`moe_2stage_down_prefill_1x4`中，下表的“重构后Kernel”是拆分后的独立顶层入口。

| 路径ID | 路径 | 重构后Kernel | 线程 / Waves | wave分布 | WG tile | 本报告case |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | Legacy M64xN64 | `moe_2stage_down_prefill_1x4` | 256 / 4 | 统一`(4,1,1)` TiledMMA；4 waves沿N64划分，每wave为M64xN16 | 每个N循环计算M64xN64 | 未单独计时 |
| P1 | N256 4-wave | `moe_2stage_down_prefill_N256_1x4` | 256 / 4 | 统一`(4,1,1)` TiledMMA；每wave为M64xN64 | M64xN256 | `n256` |
| P2 | M128xN256 8-wave | `moe_2stage_down_prefill_2x4` | 512 / 8 | 两个独立4-wave组分别负责相邻的一个M64；每组使用`(4,1,1)`，每wave为M64xN64 | 两组并行组成M128xN256 | `paired` |
| P3 | true8 M64xN512 | `moe_2stage_down_prefill_1x8_2` | 512 / 8 | 统一`(8,1,1)` TiledMMA；wave 0..7连续覆盖N512，每wave为M64xN64 | M64xN512 | `true8` |
| P4 | N-split M64xN512 | `moe_2stage_down_prefill_1x8` | 512 / 8 | 两个独立4-wave组共享一个M64；group 0/1分别处理`2*b+0`/`2*b+1`号N256块，每wave为M64xN64 | 两个N256组并行组成M64xN512 | `nsplit192`/`nsplit`/`qwen35`/`qwen397` |
| P5 | Xiaomi persistent | `moe_2stage_down_prefill_1x8_persistent` | 512 / 8 | N方向与P4相同；同一WG先后计算两个相邻M64，第二个M64复用当前weight/scale流水 | 串行两次M64xN512，等效覆盖M128xN512；grid减半 | `persistent` |

Qwen性能测试显式设置`down_nsplit_n512=True`。生产默认`MOE_DOWN_NSPLIT_N512=auto`不会自动选择P4，只有设置`MOE_DOWN_NSPLIT_N512=1`时才走该Kernel。

## 4. N-split按K特化

> 简化决策（2026-08-22）：本节记录的是P4 N-split内部的历史特化，不是后续生产计划。
> 目标架构删除P4与P5；P0保持不动，P1/P2合并为physical N256路径，P3保持独立。
> 下列K192/K384/K512结果继续用于解释删除依据，不再派生新的P4优化任务。

三个版本均为512线程/8 waves，拆成两个独立4-wave组；两个组共享一个M64 activation，分别处理偶数/奇数N256块并合成M64xN512。

| K / Case | K core | 同步与写回 | 资源 | 性能结论 |
| --- | --- | --- | --- | --- |
| K192<br>`nsplit192` / Hy3 | 3 x K64 | `K<=256`特化：两个组不做逐K workgroup rendezvous，当前N块计算完立即写回；activation由384线程搬运768个128-bit atom | 144 VGPR，96 SGPR，32KB LDS，0B private，0 scratch；96 MFMA，2 barrier | 性能一致 |
| K384<br>`nsplit` / H3 | 3 x K128 | 两组在read/write和compute阶段同步；启用K384 weight-head预取；结果延迟一个N块，通过row-major CShuffle写回 | 256 VGPR，96 SGPR，43,008B LDS，84B private；192 MFMA，10 barrier，存在scratch访问 | 性能一致；LDS比旧版多1KB，用于双scale |
| K512<br>`qwen35`/`qwen397` | 4 x K128 | 两组同步并延迟写回；不启用K384 weight-head；当前保守门禁按双M activation预算禁用CShuffle，走direct packed row-major store | 256 VGPR，96 SGPR，34,816B LDS，80B private；256 MFMA，12 barrier，存在scratch访问 | Qwen改善：35B为0.75%/0.92%，397B为1.98%/2.73%（Down/Combined） |

K192采用最轻的barrier-free/immediate-store流水；K384进入同步、延迟CShuffle流水；K512增加一个K128 core，并被当前容量谓词切换到direct packed-store，同时承受最高的寄存器与spill压力。该谓词使用`2 * BM * K`，适用于双M paired，却高估了`paired_m_groups=1`的P4 N-split：32KB A + 16KB CShuffle + 2KB双组PTPC scale合计51,200B，理论上低于64KB。这个CShuffle变体尚未编译、验证或计时，不能把“当前direct-store”写成真实LDS容量下限。

## 5. 性能结果

`new / old < 1`表示重构后更快。

### 5.1 配对比值（判定主表）

| Case | 模型 / Shape | 路径 / Kernel | 旧版本 | Down new/old (IQR) | Combined new/old (IQR) | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| `n256` | Hy3<br>B32768/N4096/K192/E193/TopK9/per-tensor | P1 N256 4-wave<br>`moe_2stage_down_prefill_N256_1x4` | 9049 | 1.000281 (0.998581..1.002820) | 1.000317 (0.999379..1.001670) | 性能一致 |
| `paired` | H3<br>B32768/N6144/K384/E128/TopK4/PTPC | P2 M128xN256 8-wave<br>`moe_2stage_down_prefill_2x4` | 9049 | 0.998212 (0.993165..1.004437) | 1.002748 (0.997252..1.006173) | 性能一致 |
| `true8` | Hy3<br>B32768/N4096/K192/E193/TopK9/per-tensor | P3 true8 M64xN512<br>`moe_2stage_down_prefill_1x8_2` | 9049 | 0.999512 (0.998476..1.000355) | 1.000201 (0.998829..1.000773) | 性能一致 |
| `nsplit192` | Hy3<br>B32768/N4096/K192/E193/TopK9/per-tensor | P4 N-split K192<br>`moe_2stage_down_prefill_1x8` | 59dd | 1.000163 (0.998628..1.000829) | 0.999517 (0.999039..1.000457) | 性能一致 |
| `nsplit` | H3<br>B32768/N6144/K384/E128/TopK4/PTPC | P4 N-split K384<br>`moe_2stage_down_prefill_1x8` | 59dd | 0.999575 (0.995239..1.006855) | 1.000318 (0.992955..1.006852) | 性能一致 |
| `persistent` | Xiaomi<br>B32768/N6144/K256/E384/TopK8/PTPC | P5 Xiaomi persistent<br>`moe_2stage_down_prefill_1x8_persistent` | 59dd | 0.999958 (0.998327..1.001455) | 0.999782 (0.998321..1.000845) | 性能一致 |
| `qwen35` | Qwen3.5 35B<br>B32768/N2048/K512/E256/TopK8/PTPC | P4 N-split K512<br>`moe_2stage_down_prefill_1x8` | 59dd | 0.992536 (0.989500..0.999330) | 0.990840 (0.987403..0.992950) | 改善0.75%/0.92% |
| `qwen397` | Qwen3.5 397B<br>B32768/N4096/K512/E512/TopK10/PTPC | P4 N-split K512<br>`moe_2stage_down_prefill_1x8` | 59dd | 0.980250 (0.968768..0.987650) | 0.972731 (0.962418..0.982878) | 改善1.98%/2.73% |

### 5.2 绝对中位时间（仅备查）

以下绝对时间受79% VRAM污染，不能与历史绝对时间横向比较。

| Case | Down old/new ms | Combined old/new ms |
| --- | --- | --- |
| `n256` | 1.562227 / 1.562008 | 2.300632 / 2.302652 |
| `paired` | 1.669688 / 1.667288 | 2.221071 / 2.224431 |
| `true8` | 1.394147 / 1.393267 | 2.133931 / 2.133911 |
| `nsplit192` | 1.539027 / 1.539048 | 2.278731 / 2.277811 |
| `nsplit` | 3.610818 / 3.613778 | 4.167761 / 4.168001 |
| `persistent` | 2.595013 / 2.595272 | 3.577477 / 3.575278 |
| `qwen35` | 3.351174 / 3.331254 | 3.690896 / 3.655076 |
| `qwen397` | 7.885973 / 7.709613 | 8.721737 / 8.464696 |

## 6. 正确性与ISA

| 类别 | Case | 正确性结果 | finite / tail |
| --- | --- | --- | --- |
| 逐bit组 | `n256`/`paired`/`true8`/`nsplit192`/`persistent` | physical和reduced逐bit一致，rel_l2=0 | 均finite；inactive tail clean（未写） |
| 修复组 | `nsplit`/`qwen35`/`qwen397` | 旧59dd存在PTPC双group scale LDS竞争，故不要求逐bit；新旧reduced relative-L2依次为`2.77561e-4`/`6.61701e-4`/`2.35524e-4` | 均finite；inactive tail clean（未写） |

| 路径 | 旧/新一致资源与关键指令 | LDS差异与备注 |
| --- | --- | --- |
| true8 M64xN512 | 128 VGPR，96 SGPR，32KB LDS，0 private；96 MFMA，18条buffer-load，8 store，2 barrier | `current_vs_9049`另记通用`load/store=24/8`，但未保留计数规则和ISA明细；两个load口径不可直接比较。 |
| Xiaomi persistent | 212 VGPR，96 SGPR，51,200B LDS，0 private；256 MFMA，38 buffer load，16 store，2 barrier | 无差异 |
| N-split K384 | 256 VGPR，96 SGPR，84B private，存在scratch访问；192 MFMA，42 buffer load，16 store，48 ds_read，21 ds_write，10 barrier，8 setprio | 重构后43,008B，旧版41,984B；增加1KB为两个N256 subgroup分别保留PTPC scale LDS，修复scale竞争；occupancy和性能未变化 |

## 7. 结果文件

以下附件保存在`refactor-performance-data/`，由同目录`SHA256SUMS`校验：

- `n256.json`
- `paired.json`
- `true8.json`
- `nsplit192.json`
- `nsplit.json`
- `persistent.json`
- `qwen35.json`
- `qwen397.json`
- `tool-final.sha256`
- `tool-qwen-final.sha256`
- `compare_refactor_perf.py`（与`tool-qwen-final.sha256`匹配的最终harness快照）

`tool-final.sha256`记录前六个代表case使用的早期harness哈希；该前驱脚本未保留，不能仅凭哈希重建。`compare_refactor_perf.py`及两个tool SHA均为`de7887b`之前的历史证据快照，必须保持字节不变；它仍使用已删除的P4/P5旧编译参数，不能对当前三路径API执行。复核历史结果时应重建9049与59dd control，并在独立历史worktree运行。
