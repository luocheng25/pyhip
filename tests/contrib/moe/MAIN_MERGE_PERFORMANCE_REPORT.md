# MoE down 合并到 main 前后性能对比（2026-08-22）

## 合并范围

本次在 `main` 上合并以下内容：

- `luocheng/hy3-single-n512-handoff@3591fd0`：
  - `src/contrib/flydsl/moe_gemm_splitk.py`
  - `tests/contrib/moe/test_moe.py`
- `luocheng/try-opt-down-308@709b39b5e1`新增的四个VMEM工具文件：
  - `tests/flydsl/attn_4wave/tools/vmem-bandwidth.md`
  - `tests/flydsl/attn_4wave/tools/vmem-bandwidth.py`
  - `tests/flydsl/attn_4wave/tools/vmem-fifo.md`
  - `tests/flydsl/attn_4wave/tools/vmem-fifo.py`
- `src/contrib/flydsl/helpers.py`的最小依赖扩展：`create_thr_mma(..., tid=)`与TiledMMA copy默认沿用`mm.thr_idx`，用于P2两个4-wave子组的局部thread id。

合并前 `main` 为 `649e1242f1c9369967e1eb8dfe0dbc3e1d823688`。

| 版本 | `moe_gemm_splitk.py` SHA256 |
| --- | --- |
| 合并前main | `7a2c8a9753bda118ccdfff59e656394d1bada4f9972040fb6991a06389300b36` |
| 原始ABBA24实测版本（HEAD `c82dc79`） | `34c35868bd9356c14fe8317c7f16912b38d06a751391ba53123239f6c5fa19ab` |
| 当前路径重命名版（与重命名前final ISA逐条一致） | `029813384f6563045e1e4fae647fe2bf7743042545aa31e2478f26efbec627c5` |

当前kernel已完成host测试最小化，并简化down路径校验、P12主循环与launcher分派；
P12 down直接消费M128 sorting产生的原始expert metadata。Gateup保持原有固定4-wave
`prefill_1x4`/`splitk`实现，不在kernel内引入8-wave BM128分支。

## 显式配置

性能case不调用自动down selector。`TILE_M_DOWN`、`TILE_M_GATEUP`、host `TILE_N`、
down路径和padding均直接写在 `test_moe.py` 的各模型args中。`TILE_M_DOWN`控制
sorting和down工作组覆盖范围，`TILE_M_GATEUP`控制gateup任务大小。

H3显式设置`TILE_M_DOWN=128`、`TILE_M_GATEUP=64`：host只传原始metadata任务数和
`METADATA_TILE_SIZE_M`；kernel文件校验两个M tile的关系、计算每个metadata block
对应的任务数并扩展launch grid。Gateup按任务索引映射原始expert metadata，P12 down
仍以一个M128工作组消费同一原始metadata。若显式设置
`TILE_M_GATEUP=128`，gateup直接消费原始metadata并启动一个BM128任务，但仍使用
现有4-wave kernel，每个wave串行覆盖两倍token行。只有`prefill_1x4`允许gateup/down
M tile不同；batch1和split-k要求两者相等。

Xiaomi 8-wave是附加实验配置，不修改生产缺省值。这里的8-wave特指P12
M128xN256：`TILE_M_DOWN=128`，一个down工作组包含两个4-wave子组；gateup仍使用
`TILE_M_GATEUP=64`并由kernel从原始M128 metadata展开两个任务。它不是Hy3专用的
P3 M64xN512路径。

| Case | Shape | 合并前显式配置 | 合并后显式配置 |
| --- | --- | --- | --- |
| Hy3 | B32768/N4096/K192/E193/TopK9/per-tensor | BM64，host TILE_N256，gateup BN128，P0 legacy | BM64，gateup BM64/BN128，P3 `8wave_1x8`，padding 0B |
| Qwen3.5 397B | B32768/N4096/K512/E512/TopK10/PTPC | BM64，host TILE_N256，gateup BN256，P0 legacy | 相同 |
| Qwen3.5 35B | B32768/N2048/K512/E256/TopK8/PTPC | BM64，host TILE_N256，gateup BN256，P0 legacy | 相同 |
| Xiaomi | B32768/N6144/K256/E384/TopK8/PTPC | BM64，host TILE_N256，gateup BN256，P0 legacy | BM64，gateup BM64/BN256，P12 `4wave_n256`，padding 128B |
| Xiaomi（8-wave附加） | B32768/N6144/K256/E384/TopK8/PTPC | BM64，host TILE_N256，gateup BN256，P0 legacy | sorting/down BM128，gateup BM64×2/BN256，P12 `4wave_n256`，padding 128B |
| H3 | B32768/N6144/K384/E128/TopK4/PTPC | BM64，host TILE_N256，gateup BN256，P0 legacy | sorting/down BM128，gateup BM64×2/BN256，P12 `4wave_n256`，padding 0B |

## 测试协议

- GPU：AMD Instinct MI308X / gfx942，80 CU，GPU1。
- 1800MHz performance determinism；PTL `Enabled / VECTOR,F8`；650W power cap。
- 每个case开始前要求GPU busy不超过5%、VRAM不超过20%、performance level为`auto`。
- 10组轮换buffer；24轮ABBA；每版每phase 48个绝对样本和24个配对ratio。
- 四个phase：`gateup`、`down`、`down + sorted_sum`（Combined）、`gateup + down + sorted_sum`（Full）。
- `ratio = 合并后 / 合并前`；小于1表示合并后更快。
- 每个case结束后恢复原performance level、PTL和NUMA状态。
- 正式矩阵包含五个生产case及一组Xiaomi 8-wave附加case。

## 性能结果

时间和TFLOPS均按“合并前 / 合并后”列出。TFLOPS按Down有效FLOPs
`2 * B * TopK * N * K`计算。提升率定义为`1 - after / before`，正值表示合并后更快；
括号内保留原始`after / before`配对ratio。

| Case | Down ms 前/后 | Combined ms 前/后 | Full ms 前/后 | Down TFLOPS 前/后 | Down提升率 (ratio) | Combined提升率 (ratio) | Full提升率 (ratio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hy3 | 1.643546 / 1.340005 | 2.428970 / 2.079488 | 4.803359 / 4.440317 | 282.229 / 346.160 | **+18.299%** (0.817010) | **+14.281%** (0.857188) | **+7.490%** (0.925102) |
| Qwen3.5 397B | 3.333133 / 3.336873 | 4.133956 / 4.140577 | 10.117620 / 10.127681 | 412.342 / 411.879 | -0.127% (1.001272) | -0.108% (1.001082) | -0.425% (1.004246) |
| Qwen3.5 35B | 1.388626 / 1.386025 | 1.727087 / 1.724927 | 4.227376 / 4.223156 | 395.899 / 396.642 | +0.128% (0.998720) | +0.035% (0.999647) | -0.004% (1.000040) |
| Xiaomi | 2.973011 / 2.236128 | 3.975436 / 3.237912 | 7.772731 / 6.983928 | 277.373 / 368.777 | **+24.692%** (0.753076) | **+18.803%** (0.811969) | **+10.424%** (0.895761) |
| Xiaomi（8-wave附加） | 2.967412 / 2.610890 | 3.956396 / 3.573375 | 7.629130 / 7.465930 | 277.897 / 315.844 | **+12.273%** (0.877266) | **+9.585%** (0.904151) | **+2.426%** (0.975744) |
| H3 | 1.734247 / 1.595206 | 2.300329 / 2.158828 | 4.936820 / 4.787199 | 356.625 / 387.709 | **+7.957%** (0.920433) | **+6.280%** (0.937202) | **+3.203%** (0.967972) |

## 结论

- Hy3切换到P3后：Down改善18.30%，Combined改善14.28%，完整链改善7.49%；三项均24/24轮胜。
- Xiaomi缺省切换到P12/TILE_M64后：Down改善24.69%，Combined改善18.80%，完整链改善10.42%；三项均24/24轮胜。
- Xiaomi 8-wave M128xN256附加配置：Down改善12.27%，Combined改善9.59%，完整链改善2.43%，三项均24/24轮胜；但gateup因M128 sorting增加padding工作量而慢6.89%（ratio `1.068913`，仅2/24轮胜）。相对当前缺省M64xN256，Down、Combined和Full绝对中位数分别慢16.76%、10.36%和6.90%，因此不替代生产缺省配置。
- H3切换到P12/`TILE_M_DOWN=128`、`TILE_M_GATEUP=64`后：Down改善7.96%，Combined改善6.28%，完整链改善3.20%；Down和Combined均24/24轮胜，Full为20/24轮胜。
- 两个Qwen case前后均固定P0 legacy，所有phase的IQR跨1，性能一致。
- 五个生产case的gateup ratio均处于噪声区间；Xiaomi 8-wave附加配置是唯一例外。生产配置的收益来自显式down路径。

H3 down去适配前后直接ABBA24中，reduced输出逐bit一致；原始M128 metadata契约的Down为
`1.595546 ms`，旧适配为`1.618886 ms`，改善1.32%（ratio `0.986796`，23/24轮胜）；
Combined改善1.01%（ratio `0.989935`，22/24轮胜），Full差异处于噪声区间。当前
路径不再构造重复expert视图；task扩展和原始metadata映射均由kernel文件负责。

可选的现有4-wave BM128 gateup也完成ABBA24：Gateup由`2.544369 ms`增至
`3.565993 ms`（ratio `1.396820`，慢39.68%），Full由`4.788898 ms`增至
`5.733141 ms`（ratio `1.203159`，慢20.32%）。因此H3缺省明确设为
BM64任务展开；`TILE_M_GATEUP=128`仅作为可配置的功能选项保留。

## 正确性与原始证据

- 六组case的gateup与down输出均finite。
- Gateup最大relative-L2为0；Down最大relative-L2为`1.4517762e-05`（Hy3）。
- 所有inactive tail和padding区域均保持未写。
- 当前候选的66项GPU回归全部通过；`prefill_1x4`的BM64×2/BM128×1对照逐值
  一致（relative-L2为0），batch1和split-k均会拒绝不一致的gateup/down M tile。
- 原始JSON：`/tmp/current-34c35868-vs-main649e124-abba24/{hy3,qwen397,qwen35,xiaomi,xiaomi8,h3}.json`。
- H3去适配专项JSON：`/tmp/noadapter-vs-adapter-h3-abba24.json`，SHA256
  `574244ac6cc2f26c3633e862b19b8ccf9b68f83e54d51ca49465def4325810a9`。
- 4-wave BM128专项JSON：`/tmp/h3-true-bm128-vs-two-bm64-abba24.json`。
- Final-API临时harness：`/tmp/compare_main_merge_finalapi.py`，SHA256 `397ad420deff0e80f496aba0dab4bde317d8e69724ba1277a3fb41ff3fa2bd6a`。
- JSON SHA256：
  - Hy3：`1fd5c917e36f04a8e28d9aefae5c67f80b1f27ce40a37db8516e9dc3ae07a352`
  - Qwen3.5 397B：`313c1d205129d17815a94851524c4a899b5514a18f26d801ecbd2067ba7d7c81`
  - Qwen3.5 35B：`f2b5e507279b9af5ebd86484d3a641e29ad501559907323f6f53a46ceb0a066e`
  - Xiaomi：`9a93b33a6b2d35e633608b4c27f7f7110348bca08f894fded19e74a3848d5a2c`
  - Xiaomi 8-wave：`e3af0c06c8b2d5e6bf0599bbcdc2a9b5b1e6b9cde4ecb0f105a5d633e60b9219`
  - H3：`dbc4b1eb70de16dc7b446f229279dfac72fbc1579040cce24da7a71ce3822649`

最终GPU状态恢复为`auto / PTL Enabled VECTOR,F8 / NUMA=1`，VRAM占用0%。

## 新增性能开关简化（2026-08-23）

本轮逐项关闭P12新增的存储、scale、调度和任务映射选择。删除门槛为候选相对
当前实现的最差Down中位回退不超过0.7%，并要求reduced输出逐bit一致、finite、
inactive tail和padding均正确。测试使用gfx942、1800MHz performance determinism、
PTL `Enabled / VECTOR,F8`、10组轮换buffer和对称ABBA；中性候选升至24或48轮确认。

最终删除两项：

- K384 `use_postprocess_weight_head`：移除首个weight atom单独预取、tail补载、
  `frag_weight_head`以及贯穿`scf.for`的额外状态，统一为完整weight fragment预取。
  48轮Down ratio分别为exact-map `0.998443`、H3 PTPC `0.990179`、Xiaomi PTPC
  `0.980829`、Xiaomi per-tensor `0.978369`；没有回退，后三组反而改善约
  0.98%至2.16%。
- `use_physical_sched_group`：删除`8 dsrd + 64 MFMA/VMEM`的scheduler hint和开关。
  K128/K256/K384/K512、M64/M128共八个K/wave组合，另加exact-map shape；
  Down最差ratio为`1.002958`（+0.296%），其余中性或更快；Combined最差ratio为
  `1.000876`（+0.088%）。

最终组合版相对清理前的48轮结果：

| Case | Down ratio | Combined ratio | 结论 |
| --- | ---: | ---: | --- |
| K256 M128 per-tensor | `0.995256` | `0.999831` | 中性 |
| H3 K384 PTPC | `0.990567` | `0.993106` | 改善0.94% / 0.69% |
| Xiaomi K384 per-tensor | `0.981540` | `0.985208` | 改善1.85% / 1.48% |
| exact-map K384 per-tensor | `0.999420` | `0.998753` | 中性 |

以下选择经单变量A/B确认超过0.7%门槛或属于容量约束，因此保留：

| 保留项 | 候选变化 | Down / Combined ratio | 原因 |
| --- | --- | ---: | --- |
| exact任务映射 | 改用通用4-way映射 | `1.037858 / 1.034018` | 明确回退 |
| `late_direct_ptpc_scale` | 改为early scale | `1.691935 / 1.529408` | 明确回退 |
| PTPC scale LDS路径 | 全部改为direct global | `1.039507 / 1.028472` | 明确回退 |
| paired LDS CShuffle | 改为direct store | `1.343881 / 1.253747` | 明确回退 |
| NT store policy | cache modifier `2 -> 0` | P1 `1.041361 / 1.036030`；P2 `1.018629 / 1.027226` | 明确回退 |
| K192阶段合并 | 关闭前两K-core合并 | M64 `1.010081 / 1.007300` | Down回退1.01% |
| delayed output/分段store | 改为立即写回 | 历史正式A/B收益约0.86%至2.03% | 超过门槛 |
| direct PTPC / row-major CShuffle容量分支 | 强制统一会超过gfx942 64KB LDS | 不适用 | 正确性/容量约束 |

正确性方面，所有本轮A/B结果均reduced逐bit一致；最终又运行M64、M128 K384 PTPC、
M128 K384双per-tensor和P3 Hy3四项focused GPU测试，均通过。P1/P2/P3 fresh compile
均为0 scratch；最终源码SHA256为
`dd8f4532d1bf37daaef34d57a34ba083d6a6aca31971e76f27ac14ad4c4e40d3`。
随后将P3的纯转发wrapper收口为真实kernel入口，并将steady state显式组织为
activation LDS staging/sync、weight prefetch + MFMA、scale + BF16转换、
CShuffle wait/store四阶段；P3 final ISA逐字不变，SHA256仍为
`43373c05d39d0b3e9af7aa3c6d583c1e2da9e23affd009be3296d310e6dbdd78`。
P12也移除纯转发wrapper，并内联N-loop的状态打包、恢复、单层`run_n_block`、
postprocess和drain helper；主循环直接呈现状态恢复、K-core预取/计算、scale/转换、
即时或延迟store、carry和最终drain。P1/P2 final ISA同样逐字不变。
通用A/B harness为`/tmp/compare_moe_switch_abba.py`，SHA256
`4379c4af42ecf3979cf8685263c311991d0f34540d2edbb7a142a92cbcec401f`；
32份JSON的清单位于`/tmp/moe-switch-results/SHA256SUMS`，清单SHA256为
`0fefc81422b310c0fcd3f0c3b823c2de3315319f2c43edac09de713bcb0fab2a`。
GPU4恢复为`auto / PTL Disabled`，GPU5-7恢复为
`auto / PTL Enabled VECTOR,F8`；四卡VRAM均为0%，NUMA保持1。

## XCC任务映射去shape限制（2026-08-23）

P2 exact XCC/SE映射原先由`M/E/N/K/TOPK`的特定组合选中。枚举验证表明，
该映射在前2316个paired任务上恰好覆盖`[0, 2315]`且无重复；2316之后保持
identity，因此正确性只依赖任务拓扑，不依赖模型shape。最终选择条件收敛为
`task_num >= 2316`，不再检查`M/E/N/K/TOPK`。

不能把generic与exact算术直接合入一个runtime分支kernel：虽然资源仍为
VGPR256/SGPR96/64KB LDS/0 scratch，但P2静态指令增加28条，Xiaomi 2304-task
的48轮Down ratio达到`1.009710`，超过0.7%删除门槛。直接把host `task_num`
传入单kernel更差，12轮Down/Combined ratio为`1.173741 / 1.125607`，也否决。

最终在同一个原生`JitFunction`中生成generic/exact两个独立`gpu.func`，launcher
只按runtime `task_num`选择其中一个`gpu.launch_func`。每次重新trace前及两次
kernel实例化之间均清空`FlyObjCache`，避免跨MLIR context或`gpu.func`复用SSA对象；
不同pointer alignment强制两个JIT cache key连续编译的专项测试通过。公开参数列表
不变，生产路径的`flyc.compile()`仍返回`CompiledFunction`并可重复热调用；没有
新增用户开关，也不会按每个`task_num`值产生JIT cache变体。

最终48轮ABBA结果如下。control为旧shape门控，candidate为任务拓扑门控；四组
reduced输出均逐bit一致、finite，inactive tail和padding均保持未写。

| Case | task_num | Down ratio | Combined ratio | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原exact shape | 2497 | `0.997113` | `0.996812` | 中性 |
| H3 K384 PTPC | 1152 | `0.999991` | `1.000423` | 短grid中性 |
| Xiaomi K256 PTPC | 2304 | `0.998879` | `0.999486` | 阈值下方中性 |
| 非原shape B38400/N2048/K256/E384/TopK8 | 2784 | `0.888156` | `0.916243` | 改善11.18% / 8.38% |

P2 generic kernel final ISA与改动前1113条指令逐字一致；exact kernel为1095条。
两者均为VGPR256/SGPR96/64KB LDS/0 scratch。P1/P3不受映射选择影响，P3 final
ISA仍与主流程简化后的471条指令逐字一致。路径重命名前源码SHA256为
`b2941bf5b6cf9af464980c1af9009e377ab619d66dbc0bdcd4f6b8b6d8c9ee3e`。

最终JSON及SHA256：

- `/tmp/moe-switch-results/xcc-launcher-select-exact-final-abba48.json`：
  `90e116f76a0788c3761cc952bcfd92de21e6a8a1ffe5be4db820df5ed3558534`
- `/tmp/moe-switch-results/xcc-launcher-select-h3-final-abba48.json`：
  `41f0fd0375459d310a0debe7cc4d42bcba905dca61374de75cd7fd1f99e1ab75`
- `/tmp/moe-switch-results/xcc-launcher-select-xiaomi-final-abba48.json`：
  `8efc2469869befd06fa82956e0e702a612b96a7b634f0defc99d0bbc7fc296c9`
- `/tmp/moe-switch-results/xcc-launcher-select-nonshape-final-abba48.json`：
  `d121b0cb7d2d32517b38ee8feb59019233543123ef2dd39810259280815a0e38`

## Down路径命名统一（2026-08-24）

为让配置名直接表达执行拓扑，本轮仅进行语义重命名：

- `physical_n256`改为`4wave_n256`：每个4-wave group计算M64xN256；BM128时同一
  workgroup包含两个4-wave group。
- `true8_hy3`改为`8wave_1x8`：单个8-wave workgroup计算M64xN512。
- 关联的选择变量、任务映射函数、kernel入口、线程/group计数和host显式配置同步
  使用`4wave_n256`/`8wave_1x8`命名；不保留旧字符串别名。

P1、P2、P3均完成fresh compile和GPU执行。重命名前后final ISA逐条比较结果：

| 路径 | 指令数 | 比较结果 | VGPR / SGPR / LDS / scratch |
| --- | ---: | --- | --- |
| P1 `4wave_n256` M64 | 932 | 逐条一致 | 250 / 96 / 25,600B / 0 |
| P2 `4wave_n256` generic | 1113 | 逐条一致 | 256 / 96 / 65,536B / 0 |
| P2 `4wave_n256` exact | 1095 | 逐条一致 | 256 / 96 / 65,536B / 0 |
| P3 `8wave_1x8` | 471 | 逐条一致 | 128 / 96 / 32,768B / 0 |

因此本报告已有ABBA性能数值无需重测，路径名称已更新为新枚举值；数值、ratio、
正确性和环境证据均保持有效。重命名后`moe_gemm_splitk.py` SHA256为
`029813384f6563045e1e4fae647fe2bf7743042545aa31e2478f26efbec627c5`。
