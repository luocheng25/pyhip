# MoE A8W4（fp8 ptpc 激活 × mxfp4 权重）gfx942 新算子：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-22 19:57）· 目标：Qwen3.8-Flash-Next MoE（H2560，E513=512 routed+1 融合 shared，topk11，I=320/192，SiLU）在 MI308X 上实现 mxfp4 权重的两阶段 FlyDSL MoE（激活按 token 量化 fp8，decode 可用 bf16）：功能正确，decode 接近访存上限、prefill 接近算力上限，各档基本达到 aiter fp8 ptpc（用 aiter 调优表），达不到说明原因（1.72× 相对冻结基线）· 已完成 3 个里程碑（含本提交 M2） · 最佳累计 1.576× · 下一步：M3 访存档路径（M 128–2048）：sorted split-K / 小 BM kernel 高带宽流式读 mxfp4 权重；之后：M4 prefill 算力档（M≥4096）：恢复指令排布、调 tile、中间量化并入 gate/up epilogue、减规约开销（承诺）
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
本任务在 MI308X（gfx942）上做 Qwen3.8-Flash-Next MoE 的 A8W4 FlyDSL 算子；M≤32 走免排序的 decode 路径（已快于 aiter），M>32 走排序的两阶段路径（尚未优化）。
- 形状：H2560，E513（512 routed + 1 个融合 shared，每个 token 的第 11 个槽位固定选它），topk11，I=320（TP2）/ 192（TP4，160 补零），SiLU。
- 输入输出：bf16 进出，输入须为有限值；权重为标准 mxfp4（fp4x2 + e8m0 scale），加载时一次性转成私有布局；激活按行动态量化成 fp8 e4m3fnuz。
- 底座：aiter gfx9_fp8 两阶段 FlyDSL MoE（供 `tuned_moe` 用）；A8W4 的 prefill kernel 由它改写、decode kernel 新写，都在 `a8w4_*.py`，底座文件不改。
- 性能验收：每个 case ≤ 1.05× aiter fp8 ptpc（`aiter.fused_moe` + aiter 自带 qwen38 调优表），争取接近访存 / 算力上限。
- 精度验收：对照同量化点的独立 fp32 Torch 参考，calc_diff ≤ 1e-3 且 max|out−ref| ≤ 2%·max|ref|，无 NaN / Inf。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、接入 tuned_moe / SGLang、gfx950；底座不再提供 MXFP4（A16W4）、SiTU / SiTUv2、单独 8x1 Down。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。
     完整的任务划分与逐拍流水线写在同目录 kernels.md（模板 templates/KERNELS.md），这里链接过去。 -->
权重在寄存器里按 per-group LUT 解成 fp8 走 fp8 MFMA；M≤32 免排序，M>32 走排序的 BM64 两阶段；方括号内的行量化按分档决定是否单独做。
```text
加载时 prepare_a8w4_weights：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
fused_moe_a8w4（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  M ≤ 32：[行量化(x)] → Gate/Up decode（每 WG 一条路由的 16 个 gate 与 16 个 up 通道）→ mid f32 [M,topk,I]
    → [行量化(mid)] → Down decode（每 WG 一个 token 的 16 个输出通道，topk 个 expert 在 LDS 内相加）→ out
  M > 32：moe_sorting(Opus, BM64) → 行量化(x) → Gate/Up（gather xq 行）→ mid f32 → 行量化(mid)
    → Down（× 通道因子 × ms × 路由权重）→ routes bf16 [排序行, H] → sorted_sum（底座）按 loc_ids 相加 → out
```

| 输入范围 | 路径 | 为什么 |
|---|---|---|
| M ≤ 32 | decode：Gate/Up decode → Down decode，不排序、不规约 | M 小时排序块几乎全是填充（M=1 每块 1 行），launch 也多 |
| （M≤32 内）M·topk·I/16 ≥ 2560：TP2 M≥12，TP4 M≥20 | x 先单独行量化，Gate/Up 读 fp8 | 否则 M·topk·I/16 个 WG 各量化一遍整行 x |
| （M≤32 内）M ≥ 4 | mid 先单独行量化，Down 用槽对 kernel | 否则每行 mid 被 H/16 个 WG 各量化一遍；独立 kernel 固定约 4.5 µs |
| M > 32 | prefill：Gate/Up BM64·BN128·TK256 → Down BM64 | 同一 expert 的多行按块共用权重；BM64 是可用的最小块 |

门槛取自三种组合在 M=2–16 的实测交叉点；w2 ≥ 4 GiB 时 Down 一律用核内量化版（槽对版按 32 位偏移寻址）；M≤32 不看 tile 参数。
核心 kernel 骨架（完整任务划分与逐拍流水线见 [kernels.md](kernels.md)；decode 的块对 / 槽对 lane 映射见 [a8w4_decode.py](a8w4_decode.py) docstring）：
```text
prefill Gate/Up：grid (2I/128, 排序块)，4 wave 管 64 个排序行 × 64 gate 与同号 64 up 通道（每 wave 16 通道）
  for tile in H/256（x 经 LDS 双缓冲、权重进 VGPR，都提前 1 个 tile 预取）：LUT 解码 → gate、up 各 MFMA 16x16x32
  epilogue：× 通道因子 × xs → SiLU(gate)·up → f32 写 mid（填充行丢弃）
prefill Down：grid (1, 排序块)，64 行 mq 整块驻留 VGPR；逐个 64 列块（权重提前 1 块预取）
  LUT 解码 → MFMA → × 通道因子 × ms × 路由权重 → bf16 → 经 LDS 合并写 routes
decode Gate/Up：grid (I/16, M·topk)，4 wave = gate / up × K 前后半；序言一次发出本 wave 的全部权重加载
  每 lane 一条 16 B 加载覆盖相邻两个 64-K 块；x 行（核内量化或读预量化）→ LDS，广播作 A 操作数
  4 个 wave 的部分和经 LDS 相加 → × 通道因子 × xs → SiLU(gate)·up → f32 写 mid
decode Down：grid (H/16, M)，wave 按槽取（槽对版一条 16 B 加载覆盖两个槽的同一 64-K 块）；mid 行 → LDS
  MFMA → × 通道因子 × 行 scale × 路由权重；各 wave 的部分和经 LDS 相加 → bf16 写 out（不需清零）
```
行量化：一个 wave 一行，scale = amax/240、q = RNE(x/scale)（两处 IEEE 除法），与参考逐位一致；prefill 版逐元素访存，decode 版 16 B 访存。
底座（供 `tuned_moe`，A8W4 只用其 `common.py`、Opus 排序与 `sorted_sum`）：aiter gfx9_fp8 逐字节移植，autotune 选路径。

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
关键取舍：fp4 在寄存器里解成 fp8 走 fp8 MFMA，激活量化与参考逐位一致；M≤32 不排序，按访存指令条数优化。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 经 LUT 解成 fp8 走 fp8 MFMA（decode 同） | 解成 bf16（A16W4） | 无 fp4 MFMA；bf16 的 MFMA 与解码 VALU 都翻倍 | delta > 11 的组按 0 处理 |
| 私有布局按 MFMA K 序预排，加载时转一次 | kernel 内重排标准布局 | 每 lane 每 64-K 读 8 B 即 2 条 MFMA 操作数 | 需 prepare；布局绑定 MFMA 形状 |
| 自写行量化（IEEE 除法），Gate/Up 写 f32 中间 | aiter HIP 量化 / epilogue 融合量化 | 与参考逐位一致；实测误差 ≤1.24%，判据 2% | 多 1 次 launch 与 f32 中间读写 |
| M≤32 不排序，Down 在 WG 内加完 topk 直接写 bf16 | 小 M 也走排序 BM64 | BM64 时 M=1 的 Down 只有 11 个 CTA | 多一套 kernel；另有形状上限 |
| decode 每 lane 一条 16 B 加载（两个块或两个槽） | 每 lane 8 B（同 prefill） | 受访存指令条数限制，指令数减半 | 只用一半 MFMA 输出；槽对要屏蔽 A 操作数 |
| 底座原样取自 aiter，A8W4 prefill 沿用其结构 | pyhip 分叉版 / PR #64 | 与对照同源；PR #64 要 I%256 | 删 pyhip 独有功能；无 sched_group |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面：每档一行“本实现 ÷ 对照”与一句结论；对照的来源（实现、版本或提交、调用方式、
     各档实际选中的 kernel、是否与指标同进程同协议计时）；精度参考（参考实现与容差）。冻结基线由生成块第一行给出。 -->
M≤32 的 decode 档已快于 aiter（耗时 0.58–0.82×）；M≥128 仍走未优化的排序路径，慢于 aiter（1.13–1.60×）。
| 档（case） | 本实现 ÷ aiter（TP2 / TP4；bench 的 `MF_REF ratio`，同卡 4 次中位数） | 结论 |
|---|---|---|
| 单 token（m1） | 0.77 / 0.82 | 达标；两次 launch 的固定开销为主，HBM 约 15% / 11% |
| decode（m16） | 0.59 / 0.58 | 达标；受访存指令条数限制，HBM 约 36% / 28% |
| 访存 / 过渡（m128、m1024） | 1.60、1.57 / 1.60、1.50 | 未达标：仍走排序 BM64 功能版 |
| prefill（m4096、m16384） | 1.35、1.21 / 1.16、1.13 | 未达标：同上 |
- 对照：aiter fp8 ptpc，/opt/aiter @43ce3d6bf2（gfx9_fp8 kernel、`fused_moe_gfx942.py` 与调优表都同 ca6bc0f851）；与指标同进程同协议交替计时。
- 调用：`aiter.fused_moe`（per_Token；同一份 BF16 权重按通道量化成 fp8、shuffle 16x16，TP4 补零到 192）；只用 qwen38 调优表并断言命中。
- 对照所选：m1 FlyDSL batch1；m16 split-K / asm（TP2 / TP4）；m128–1024 asm 1 阶段；m4096 asm / FlyDSL；m16384 FlyDSL prefill。
- 计时：每次先写 512 MiB 冷缓存并让 GPU 空转（≥200 µs 且 ≥2× host 下发），只计 GPU 时间；30 次取中位。
- 精度参考：`pyhip.testing.moe_a8w4.reference`，独立 fp32 Torch，mxfp4 精确反量化，输入与中间按行 fp8（同形 tensor 做 IEEE 除法）。
- 判据：calc_diff ≤ 1e-3 且最大误差 ≤ 2%·max|ref|、无 NaN / Inf；bench 每组输出都校验，最大误差 decode ≤0.35%、prefill ≤1.16%。
- 底座的精度参考：`pyhip.testing.moe.torch_reference`，calc_diff ≤ 0.02；FP32→BF16 默认 RTE（同 aiter）。

<!-- mf:perf -->
基线来源（mf 记录）：11-22 13:41 冻结于提交 `78962639bc`（里程碑 M1，tag `moe-ptpc-a8w4-r2-M1`），bench 跑 3 次取中位数，GPU0，噪声 ±0.17%；bench 命令 `/opt/lc/work/qsa-fp8/venv-fly050/bin/python benchmarks/moe/bench_moe_a8w4.py`。对照实现与精度参考的来源见本节手写部分。

最近的性能里程碑 M2：相对冻结基线 1.576×（mf 复核 V1：同卡 ABBA 4 对，GPU0，噪声 ±0.30%）。

<details><summary>逐 case（12 个，mf 生成）</summary>

| case | 基线 | 当前 | 累计加速 | 增量加速 |
|---|---|---|---|---|
| tp2_m1 | 103.4 | 18.26 | 5.674× | — |
| tp2_m1024 | 760.44 | 759.78 | 1.001× | — |
| tp2_m128 | 648.93 | 649.22 | 0.999× | — |
| tp2_m16 | 251.97 | 101.2 | 2.488× | — |
| tp2_m16384 | 4330.28 | 4339.44 | 0.998× | — |
| tp2_m4096 | 1528.3 | 1529.64 | 0.999× | — |
| tp4_m1 | 93.64 | 14.75 | 6.326× | — |
| tp4_m1024 | 544.48 | 540 | 1.007× | — |
| tp4_m128 | 460.52 | 457.87 | 1.007× | — |
| tp4_m16 | 189.73 | 73.49 | 2.580× | — |
| tp4_m16384 | 2974.13 | 2969.07 | 1.002× | — |
| tp4_m4096 | 1048.02 | 1044.38 | 1.006× | — |

</details>
<!-- /mf:perf -->

## 5. 测试覆盖
A8W4 与底座各有基本用例；12 个 bench case 覆盖 A8W4 两条路径的全部分档；分派交界、变体、逐位、报错与 graph 在其他覆盖。

**基本功能测试**（每次改动先跑：`commands.correctness` = `test_moe_a8w4.py` + `test_fly_2stage_basic.py`，9 例，约 20 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| A8W4 TP2（I320，E33）：M=3（decode，激活在 kernel 内量化）与 M=1031（prefill，多块 + 尾块） | `test_tp2` ×2 | 通过 |
| A8W4 TP4：I160 补零到 192，M=257（prefill） | `test_tp4_padded` | 通过 |
| 底座 decode：batch1（M=1）、split-K（M=16），FP8 PTPC，H2560 / I320 | `test_batch1`、`test_splitk` | 通过 |
| 底座 prefill：FP8 × 三种 Down（compact 用 I192）、BF16 × default | `test_prefill` ×3、`test_prefill_bf16` | 通过 |

**性能测试**（`commands.bench` = `bench_moe_a8w4.py`，12 个 case，每组输出都按判据校验；底座是逐字节移植，没有 case）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| E513 decode：M=1（2 个 kernel）、TP4 M=16（mid 预量化）、TP2 M=16（x、mid 都预量化） | `tp*_m1`、`tp*_m16` | 0.58–0.82× aiter |
| E513 prefill：M=128 / 1024 / 4096 / 16384 | `tp2_m128` … `tp4_m16384` | 1.13–1.60× aiter |

**其他覆盖测试**（只测前两类覆盖不了的；`commands.verify_extra`，62 例，约 65 秒；用例名省略 `test_`）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| decode 分派交界 M=4、TP2 11/12、TP4 19/20、32/33 | 基本只有 M=3，bench 只有 M=1、16 | `decode_dispatch_boundaries` ×7 | 通过 |
| decode topk=8（槽对整对无效、WG 数恰为 2560）与 topk=17（Down 4 wave 轮流取槽） | 前两类都是 topk=11 | `decode_topk_variants` ×2 | 通过 |
| decode 同 token 重复 expert、int64 非连续 topk_ids、非连续 topk_weight | 生成的路由不重复、张量连续 | `decode_duplicate_and_int64_ids` | 通过 |
| decode 在 w2 ≥ 4 GiB 时改用核内量化 Down（stride 0 视图冒充 4 GiB） | 前两类的 w2 都远小于 4 GiB | `decode_down_fallback_w2_4gib` | 通过 |
| CUDA graph 捕获与重放：prefill M=64，decode M=1 / 5 / 16 | 前两类都是 eager 调用 | `graph_replay`、`decode_graph_replay` ×3 | 通过 |
| prefill 行量化与参考逐位一致（BF16 中点、全零行、无 -0） | 少量 fp8 翻转端到端未必发现 | `row_quant_bitexact` ×2 | 通过 |
| decode 行量化与 a8w4_quant、参考逐位一致：下溢、-0、次正规、尾片、越界行不写 | 同上；bench 只有随机数据 | `decode_row_quant_bitexact` ×6 | 通过 |
| A8W4 prepare 布局逐字节：delta 0–15、-0、补零、通道因子 | 端到端误差看不出布局与文档不符 | `prepare_layout_bitexact` | 通过 |
| A8W4 delta 11（精确）/ 12（清零）边界，全部 16 个 E2M1 码 | 随机权重同一行 E8M0 只差 0–3 | `delta_flush_boundary` | 通过 |
| A8W4 E513 下 prefill grid 收缩交界 M=46 / 47 | 基本用 E33，bench 的 M 远离交界 | `grid_shrink_boundary_e513` | 通过 |
| A8W4 prefill 非默认 tile：TILE_K128、BM128 + BN256 | 前两类只走默认 tile | `tile_variants` ×2 | 通过 |
| A8W4 报错：dtype、形状、E8M0 NaN、H / I 不对齐、BM·I_pad 非 4096 倍数、hidden 不符 | 前两类只给合法输入 | `rejects_bad_inputs` | 通过 |
| 底座 per-tensor / BF16 decode scale、SwiGLU | 基本只测 PTPC、SiLU | `splitk_variants`、`prefill_variants`、`swiglu` ×2 | 通过 |
| 底座形状交界：8x1 的 K 调度族、N512、K640 + pad 0、M·topk ≤ E | 基本只用 Qwen 形状与大 M | `prefill_variants`、`splitk_variants` | 通过 |
| 底座 batch1 清零写满与 graph replay（排序、loc_ids、任务表） | 基本只跑 eager、不查输出两侧 | `batch1_clear_graph` ×4、`graph_replay` ×2 | 通过 |
| 底座报错：K 不是 256 的倍数、Gate/Up 与 Down 的 M 不同 | 主路径只走合法配置 | `gateup_rejects_incomplete_k` ×4、`split_m_tile_rejected` | 通过 |
| 底座逐位：FP32→BF16 舍入、Down 打包前只舍入一次 | calc_diff 看不出舍入方式 | `bf16_rounding_bits`、`down_rounds_before_packing` ×3 | 通过 |
| 底座候选门槛：不完整 N tile、默认 Down 超 64 KiB；恰 64 KiB 可 launch | 前两类用固定配置，不枚举候选 | `candidate_gates` ×5、`lds_64k_boundary` | 通过 |

未覆盖 / 跳过：
- decode 的形状上限（H > 4096；M ≥ 4 时 I_pad > 1024·min(⌈topk/2⌉, 3)）：目标形状碰不到，构建 kernel 时报 AssertionError，未测。
- w2 真超过 4 GiB：只用 stride 0 视图验证分派与单槽 Down，没在大权重上跑过 64 位 expert 寻址；这个回退也没有性能 case。
- 含 NaN 的输入：约定输入为有限值（decode 与 prefill 行量化对 NaN 的处理不同），不测。
- prefill 路径的同 token 重复 expert id：生成的路由不重复，只在 decode 上验证过。
- M·topk·I_pad·4 ≥ 2^31（I320 约 15 万 token）：输入过大，入口也不检查。
- gfx950 与同进程多卡：只在单卡 gfx942 上跑。
- 底座 per-tensor 的 Gate/Up 输出预清零：只在排序丢掉路由时起作用，正常 top-k 路由触发不到（做法同 aiter）。
- tuned_moe 的调优缓存、剪枝与非 FlyDSL 路径在 `test_tuned_moe.py` / `test_moe.py`，不在上面两条命令里。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
decode 档已达标；主要风险是 M≥128 未达标与精度余量小。
- M≥128 仍走排序 BM64 功能版（1.13–1.60× aiter）→ 这 8 个 case 未达 1.05× → 按路线图做访存档与 prefill 算力档。
- 2% 最大误差判据对 fp8 舍入翻转敏感（prefill 3 个种子 0.33–1.24%）→ 改动量化前的算术会超判据 → 保持与参考逐位一致，由逐位用例把关。
- host 下发：decode 约 0.1–0.17 ms、prefill 约 0.32–0.38 ms/次 → eager 小 M 受 host 限制 → 实用靠 CUDA graph（两条路径都已验证）。
- 约束 H%512、I%32、BM·I_pad%4096、M·topk·I_pad·4 < 2^31，decode 另有形状上限（见第 5 节），入口都不查 → 新形状可能报错或越界 → 接新模型先核对。
- 底座：编译缓存不按设备（A8W4 同）、gfx950 未验证、SiTUv2 无 FlyDSL 候选、两个旧用例起点即失败 → 每卡一进程、只用 gfx942。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../moe_a8w4.py) 的 `prepare_a8w4_weights`（加载时一次）与 `fused_moe_a8w4`（接口与约束见 docstring，分派门槛是文件头的 `DECODE_*`）。
- kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)；私有布局与 LUT 解码见 [a8w4_common.py](a8w4_common.py)，decode 的块对 / 槽对映射见 [a8w4_decode.py](a8w4_decode.py) 的 docstring。
- 参考与 bench：[pyhip.testing.moe_a8w4](../../../../testing/moe_a8w4.py)、[bench_moe_a8w4.py](../../../../../../benchmarks/moe/bench_moe_a8w4.py)；用例说明见 [tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)。
- 底座：[tuned_moe.py](../../tuned_moe.py) 的 `_run_fly` / `_configs`；kernel 同 [aiter gfx9_fp8 @ca6bc0f851](https://github.com/ROCm/aiter/tree/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8)，调用链见 [fused_moe_gfx942.py](https://github.com/ROCm/aiter/blob/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/fused_moe_gfx942.py)。
- 里程碑（每个提交写了改了什么、为什么）：

<!-- mf:roadmap -->
- M0 ✅ 已推送 · 功能 · 底座：aiter gfx9_fp8 替换 pyhip moe_gemm_2st… · [提交](https://github.com/luocheng25/pyhip/commit/aca23ec23286bb8136d4c1644491dc7d132489a0)
- M1 ✅ 已推送 · 功能 · A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重 · [提交](https://github.com/luocheng25/pyhip/commit/78962639bc4e8cdde5b337b6e05ee7ecef02dc99)
- M2 ✅ 本提交 · 性能 · 小 M 直通路径（M≤32）：免排序小 tile kernel，激活可用 bf… · 1.576× 对基线（预期 1.5×）
- M3 ⏳ 承诺 · 性能 · 访存档路径（M 128–2048）：sorted split-K / 小 BM… · 预期 1.4×
- M4 ⏳ 承诺 · 性能 · prefill 算力档（M≥4096）：恢复指令排布、调 tile、中间量化并… · 预期 1.09×

下一步（本提交的承诺）：M3 访存档路径（M 128–2048）：sorted split-K / 小 BM kernel 高带宽流式读 mxfp4 权重；之后：M4 prefill 算力档（M≥4096）：恢复指令排布、调 tile、中间量化并入 gate/up epilogue、减规约开销（承诺）
<!-- /mf:roadmap -->
