# MoE A8W4（fp8 ptpc 激活 × mxfp4 权重）gfx942 新算子：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-22 13:37）· 目标：Qwen3.8-Flash-Next MoE（H2560，E513=512 routed+1 融合 shared，topk11，I=320/192，SiLU）在 MI308X 上实现 mxfp4 权重的两阶段 FlyDSL MoE（激活按 token 量化 fp8，decode 可用 bf16）：功能正确，decode 接近访存上限、prefill 接近算力上限，各档基本达到 aiter fp8 ptpc（用 aiter 调优表），达不到说明原因（1.2× 相对冻结基线）· 已完成 2 个里程碑（含本提交 M1） · 最佳累计 — · 下一步：等你决定
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
本任务在 MI308X（gfx942）上做 Qwen3.8-Flash-Next MoE 的 A8W4 两阶段 FlyDSL 算子；功能版可用，所有 M 走同一条路径，性能尚未优化。
- 形状：H2560，E513（512 routed + 1 个融合 shared，每个 token 的第 11 个槽位固定选它），topk11，I=320（TP2）/ 192（TP4，160 补零），SiLU。
- 输入输出：bf16 进出；权重为标准 mxfp4（fp4x2 + e8m0 scale），加载时一次性转成私有布局；激活按行动态量化成 fp8 e4m3fnuz。
- 底座：aiter gfx9_fp8 两阶段 FlyDSL MoE（供 `tuned_moe` 用）；A8W4 kernel 由它的 prefill kernel 改写成新文件，底座文件不改。
- 性能验收：每个 case ≤ 1.05× aiter fp8 ptpc（`aiter.fused_moe` + aiter 自带 qwen38 调优表），争取接近访存 / 算力上限。
- 精度验收：对照同量化点的独立 fp32 Torch 参考，calc_diff ≤ 1e-3 且 max|out−ref| ≤ 2%·max|ref|，无 NaN / Inf。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、接入 tuned_moe / SGLang、gfx950；底座不再提供 MXFP4（A16W4）、SiTU / SiTUv2、单独 8x1 Down。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。
     完整的任务划分与逐拍流水线写在同目录 kernels.md（模板 templates/KERNELS.md），这里链接过去。 -->
A8W4 是把底座 prefill 两段 kernel 的权重换成 mxfp4：权重在寄存器里按 per-group LUT 解成 fp8，走 fp8 MFMA；所有 M 走同一条路径。
```text
加载时 prepare_a8w4_weights：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
fused_moe_a8w4（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  moe_sorting(Opus, block_m=64) → sorted_ids / expert_ids / loc_ids
  行量化(x) → xq fp8 [M,H]、xs [M]
  A8W4 Gate/Up：gather xq 行 × 解码(w1) → × 通道因子 × xs → SiLU(gate)·up → mid f32 [M,topk,I]
  行量化(mid) → mq fp8、ms [M·topk]
  A8W4 Down：gather mq 行 × 解码(w2) → × 通道因子 × ms × 路由权重 → routes bf16 [排序行, H]
  sorted_sum（底座）：按 loc_ids 把每个 token 的 topk 行相加 → out
```

| 输入范围 | 路径 | 为什么 |
|---|---|---|
| 所有 M（单 token、decode、过渡、prefill） | Gate/Up BM64·BN128·TK256 → Down BM64 | 先让一条路径正确；BM32 配 I320/192 不合法 |
| decode 专用（split-K / batch1 / A16W4） | 未实现 | 冻结基线后按性能里程碑加 |

Gate/Up 骨架（[a8w4_gemm1.py](a8w4_gemm1.py)；完整任务划分与流水线见 [kernels.md](kernels.md)）：
```text
grid (2I/128, 排序块)；CTA 256 线程 = 64 个排序行 × 64 个 gate 与同号 64 个 up 通道；4 个 wave 各管 16 个通道
for tile in H/256（x 经 LDS 双缓冲，权重直接进 VGPR，两者都提前 1 个 tile 预取）:
    预取下一 tile：按 sorted id gather x 行；本 wave 的权重（每 lane 每 64-K 8 B）与 delta
    for 64-K 块 in 4：从 LDS 取 x；delta 选 LUT，每 8 个 fp4 用 2 次 v_perm 解成 fp8；gate、up 各做 MFMA 16x16x32
    预取到的 x 写入另一块 LDS；barrier
epilogue：× 通道因子 × xs[token] → SiLU(gate)·up → f32 写 mid[token, slot]（填充行丢弃）
```
Down 骨架（[a8w4_gemm2.py](a8w4_gemm2.py)）：
```text
grid (1, 排序块)；CTA 256 线程 = 64 个排序行 × 全部 H 列；4 个 wave 各管每个 64 列块中的 16 列
gather 64 行 mq（64×I fp8）→ LDS → 整块读进 VGPR 作 MFMA 操作数
for 64 列块 in H/64（权重提前 1 块预取，输出经 LDS 双缓冲）:
    本 wave 16 行权重（每 lane 每 64-K 8 B）与 delta → LUT 解码 → I/32 步 MFMA 16x16x32 → × 通道因子
    × ms[token, slot] × 路由权重 → bf16 → LDS → 128 bit 合并写 routes 的这一列块
```
行量化（[a8w4_quant.py](a8w4_quant.py)）：一个 wave 一行，scale = amax/240、q = RNE(x/scale)，两处都是 IEEE 除法，与参考逐位一致。

底座（供 `tuned_moe`，A8W4 不经过）：aiter gfx9_fp8 逐字节移植，batch1 / split-K / prefill（三种 Down）由 autotune 选。
A8W4 只复用底座的 `common.py` 工具、Opus 排序的用法与 `sorted_sum`；自己的 kernel 都在 `a8w4_*.py`。

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
关键取舍：权重在寄存器里解成 fp8 走 fp8 MFMA，激活量化与参考逐位一致，先用一条 BM64 路径保证正确。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 按 per-group LUT 解成 fp8 走 fp8 MFMA | 解成 bf16 走 bf16 MFMA | 无 fp4 MFMA；fp8 算力是 bf16 的 2 倍 | delta > 11 的组按 0 处理 |
| 私有布局按 MFMA K 序预排，加载时转一次 | kernel 内重排标准布局 | 每 lane 每 64-K 读 8 B 即 2 条 MFMA 操作数 | 需 prepare；布局绑定 MFMA 形状 |
| 自写行量化（IEEE 除法），Gate/Up 写 f32 中间 | aiter HIP 量化 / epilogue 融合量化 | 与参考逐位一致；实测误差 ≤1.24%，判据 2% | 多 1 次 launch 与 f32 中间读写 |
| 所有 M 走一条 BM64 路径 | 同时做 decode 专用路径 | BM·I_pad 须是 4096 的倍数，BM64 最小可用 | M=1 时 Down 只有 11 个 CTA |
| 沿用底座 prefill_1x4 + 默认 Down 的结构 | 搬上游 PR #64 的 TP kernel | PR #64 要求 I%256，320 / 192 不满足 | 去掉了 sched_group 指令排布 |
| 底座 9 个 kernel 文件原样取自 aiter | 保留 pyhip 分叉版 | 与对照同源，差异 0 字节 | pyhip 独有功能删除 |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面：每档一行“本实现 ÷ 对照”与一句结论；对照的来源（实现、版本或提交、调用方式、
     各档实际选中的 kernel、是否与指标同进程同协议计时）；精度参考（参考实现与容差）。冻结基线由生成块第一行给出。 -->
尚无性能结论：冻结基线还没测（测后见生成块第一行），逐档“本实现 ÷ 对照”待冻结后补。
- 对照：aiter fp8 ptpc，/opt/aiter @43ce3d6bf2（gfx9_fp8 kernel、`fused_moe_gfx942.py` 与调优表都同 ca6bc0f851）。
- 调用：`aiter.fused_moe`（per_Token；同一份 BF16 权重按通道量化成 fp8、shuffle 16x16，TP4 补零到 192）。
- 调优表：只用 `qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv`；bench 每个 case 断言 dispatcher 用的是表中那一行并打印 kernel 名。
- 对照所选（TP2）：M=1 FlyDSL batch1；M=16 FlyDSL split-K；M=128–4096 asm 1 阶段 32x320；M=16384 FlyDSL prefill（Down 1x4_64x256）。
- 对照所选（TP4）：M=1 FlyDSL batch1；M=16–1024 asm 1 阶段 32x192；M=4096、16384 FlyDSL prefill（Down 1x4_64x256）。
- 计时：两者同进程同协议交替；每次先写 512 MiB 冷缓存并让 GPU 空转（≥200 µs 且 ≥2× host 下发），只计 GPU 时间；30 次取中位。
- 精度参考：`pyhip.testing.moe_a8w4.reference`，独立 fp32 Torch，mxfp4 精确反量化，输入与中间按行 fp8（同形 tensor 做 IEEE 除法）。
- 判据：calc_diff ≤ 1e-3 且 max 误差 ≤ 2%·max|ref|、无 NaN / Inf；bench 对每组计时输出都校验。
- 底座的精度参考：`pyhip.testing.moe.torch_reference`，calc_diff ≤ 0.02；FP32→BF16 默认 RTE（同 aiter）。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
A8W4 与底座各有基本用例；12 个 bench case 覆盖 A8W4 全部分档；布局、边界、报错、逐位与 graph 在其他覆盖。

**基本功能测试**（每次改动先跑：`commands.correctness` = `test_moe_a8w4.py` + `test_fly_2stage_basic.py`，9 例，约 20 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| A8W4 TP2（I320，E33）：M=3（M·topk = E）与 M=1031（多块 + 尾块） | `test_tp2` ×2 | 通过 |
| A8W4 TP4：I160 补零到 192，M=257 | `test_tp4_padded` | 通过 |
| 底座 decode：batch1（M=1）、split-K（M=16），FP8 PTPC，H2560 / I320 | `test_batch1`、`test_splitk` | 通过 |
| 底座 prefill：FP8 × 三种 Down（compact 用 I192）、BF16 × default | `test_prefill` ×3、`test_prefill_bf16` | 通过 |

**性能测试**（`commands.bench` = `bench_moe_a8w4.py`，12 个 case；底座是逐字节移植，没有 case）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| E513 的 TP2 / TP4 × 单 token、decode、过渡、prefill，同场对照 aiter | `tp2_m1` … `tp4_m16384` | 可跑，输出校验全过；数字待冻结 |

**其他覆盖测试**（只测前两类覆盖不了的；`commands.verify_extra`，42 例，约 1 分钟；用例名省略 `test_`）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| A8W4 prepare 布局逐字节：delta 0–15、-0、补零、通道因子 | 端到端误差看不出布局与文档不符 | `prepare_layout_bitexact` | 通过 |
| A8W4 delta 11（精确）/ 12（清零）边界，全部 16 个 E2M1 码 | 随机权重同一行 E8M0 只差 0–3 | `delta_flush_boundary` | 通过 |
| A8W4 行量化与参考逐位一致（BF16 中点、全零行、无 -0） | 少量 fp8 翻转端到端未必发现 | `row_quant_bitexact` ×2 | 通过 |
| A8W4 E513 下 grid 收缩交界 M=46 / 47 | 基本用 E33，bench 的 M 远离交界 | `grid_shrink_boundary_e513` | 通过 |
| A8W4 非默认 tile：TILE_K128、BM128 + BN256 | 前两类只走默认 tile | `tile_variants` ×2 | 通过 |
| A8W4 CUDA graph 捕获与重放 | 前两类都是 eager 调用 | `graph_replay` | 通过 |
| A8W4 报错：dtype、形状、E8M0 NaN、H / I 不对齐、BM·I_pad 非 4096 倍数、hidden 不符 | 前两类只给合法输入 | `rejects_bad_inputs` | 通过 |
| 底座 per-tensor / BF16 decode scale、SwiGLU | 基本只测 PTPC、SiLU | `splitk_variants`、`prefill_variants`、`swiglu` ×2 | 通过 |
| 底座形状交界：8x1 的 K 调度族、N512、K640 + pad 0、M·topk ≤ E | 基本只用 Qwen 形状与大 M | `prefill_variants`、`splitk_variants` | 通过 |
| 底座 batch1 清零写满与 graph replay（排序、loc_ids、任务表） | 基本只跑 eager、不查输出两侧 | `batch1_clear_graph` ×4、`graph_replay` ×2 | 通过 |
| 底座报错：K 不是 256 的倍数、Gate/Up 与 Down 的 M 不同 | 主路径只走合法配置 | `gateup_rejects_incomplete_k` ×4、`split_m_tile_rejected` | 通过 |
| 底座逐位：FP32→BF16 舍入、Down 打包前只舍入一次 | calc_diff 看不出舍入方式 | `bf16_rounding_bits`、`down_rounds_before_packing` ×3 | 通过 |
| 底座候选门槛：不完整 N tile、默认 Down 超 64 KiB；恰 64 KiB 可 launch | 前两类用固定配置，不枚举候选 | `candidate_gates` ×5、`lds_64k_boundary` | 通过 |

未覆盖 / 跳过：
- decode 专用路径（split-K / batch1 / A16W4）：尚未实现。
- 同一 token 重复 expert id：输入生成的路由不重复，A8W4 未验证。
- M·topk·I_pad·4 ≥ 2^31（I320 约 15 万 token）：输入过大，入口也不检查。
- gfx950 与同进程多卡：只在单卡 gfx942 上跑。
- 底座 per-tensor 的 Gate/Up 输出预清零：只在排序丢掉路由时起作用，正常 top-k 路由触发不到（做法同 aiter）。
- tuned_moe 的调优缓存、剪枝与非 FlyDSL 路径在 `test_tuned_moe.py` / `test_moe.py`，不在上面两条命令里。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
功能已过验收；主要风险是精度余量小、decode 没有专用路径。
- 2% 最大误差判据对 fp8 舍入翻转敏感（3 个种子 0.33–1.24%）→ 改动量化前的算术会超判据 → 保持与参考逐位一致，由逐位用例把关。
- 所有 M 走 BM64 路径（M=1 时 Down 只有 11 个 CTA）→ decode 档可能达不到 1.05× aiter → 冻结后加 split-K / batch1 / A16W4。
- host 下发约 0.3 ms/次（排序 + 5 个 FlyDSL kernel）→ eager 小 M 受 host 限制 → decode 实用靠 CUDA graph（已验证可捕获）。
- 约束 H%512、I%32、BM·I_pad%4096、M·topk·I_pad·4 < 2^31（末条入口不检查）→ 新形状可能报错或越界 → 接新模型先核对。
- 底座：编译缓存不按设备（A8W4 同）、gfx950 未验证、SiTUv2 无 FlyDSL 候选、两个旧用例起点即失败 → 每卡一进程、只用 gfx942。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../moe_a8w4.py) 的 `prepare_a8w4_weights`（加载时一次）与 `fused_moe_a8w4`（接口与约束见 docstring）。
- kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)；私有布局与 LUT 解码见 [a8w4_common.py](a8w4_common.py) 的 docstring。
- 参考与 bench：[pyhip.testing.moe_a8w4](../../../../testing/moe_a8w4.py)、[bench_moe_a8w4.py](../../../../../../benchmarks/moe/bench_moe_a8w4.py)；用例说明见 [tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)。
- 底座：[tuned_moe.py](../../tuned_moe.py) 的 `_run_fly` / `_configs`；kernel 同 [aiter gfx9_fp8 @ca6bc0f851](https://github.com/ROCm/aiter/tree/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8)，调用链见 [fused_moe_gfx942.py](https://github.com/ROCm/aiter/blob/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/fused_moe_gfx942.py)。
- 里程碑（每个提交写了改了什么、为什么）：

<!-- mf:roadmap -->
- M0 ✅ 已推送 · 功能 · 底座：aiter gfx9_fp8 替换 pyhip moe_gemm_2st… · [提交](https://github.com/luocheng25/pyhip/commit/aca23ec23286bb8136d4c1644491dc7d132489a0)
- M1 ✅ 本提交 · 功能 · A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重

下一步（本提交的承诺）：等你决定
<!-- /mf:roadmap -->
