# MoE A8W4（fp8 ptpc 激活 × mxfp4 权重）gfx942 新算子：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-22 12:12）· 目标：Qwen3.8-Flash-Next MoE（H2560，E513=512 routed+1 融合 shared，topk11，I=320/192，SiLU）在 MI308X 上实现 mxfp4 权重的两阶段 FlyDSL MoE（激活按 token 量化 fp8，decode 可用 bf16）：功能正确，decode 接近访存上限、prefill 接近算力上限，各档基本达到 aiter fp8 ptpc（用 aiter 调优表），达不到说明原因（1.2× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M1 A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
本任务在 MI308X（gfx942）上做 Qwen3.8-Flash-Next MoE 的 A8W4 两阶段 FlyDSL 算子；**A8W4 尚未实现**，目前只有底座。
- 形状：H2560，E513（512 routed + 1 个融合 shared，每个 token 的第 11 个槽位固定选它），topk11，I=320（TP2）/ 192（TP4，160 补零），SiLU。
- 输入输出：bf16 激活进出；权重 mxfp4（fp4x2 + e8m0 scale）；激活按 token 量化为 fp8 e4m3fnuz，decode 可直接用 bf16。
- 底座（已有）：aiter gfx9_fp8 的两阶段 FlyDSL MoE，支持 BF16 与 FP8 per-token / per-tensor 权重的 SiLU / SwiGLU。
- 性能验收：每个 case ≤ 1.05× aiter fp8 ptpc（`aiter.fused_moe` + aiter 自带 qwen38 调优表），争取接近访存 / 算力上限。
- 精度验收：对照独立 fp32 Torch 参考，calc_diff ≤ 1e-3 且 max|out−ref| ≤ 2%·max|ref|，无 NaN / Inf。
- 不做：EP / expert_mask、bias、预量化输入、接入 SGLang、gfx950；底座不再提供 MXFP4（A16W4）、SiTU / SiTUv2、单独 8x1 Down。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。
     完整的任务划分与逐拍流水线写在同目录 kernels.md（模板 templates/KERNELS.md），这里链接过去。 -->
底座逐字节取自上游：`tuned_moe.fused_moe` 按 batch 桶枚举三条 FlyDSL 路径的候选，autotune 选出最快且通过精度检查的配置。
```text
fused_moe ──autotune 选配置──→ _run_fly（bf16 进、bf16 出）
  batch1 ：Gate/Up(batch1，顺带清零 out) → Down(batch1，atomic 加到 out)
  split-K：moe_sorting(清零 out) → Gate/Up(splitk) → Down(splitk，atomic 加到 out)
  prefill：moe_sorting(Opus aux 给出 loc_ids) → [FP8] 量化 x → Gate/Up(prefill_1x4) → mid
           → [FP8] 量化 mid → Down(default | 1x4_64x256 | 8x1_compact) → 每条路由一行
           → sorted_sum(按 loc_ids 把 topk 行相加) → out
```

| 输入范围（batch 桶） | 路径 | 为什么 |
|---|---|---|
| ≤ 32 | batch1：每条路由一个 task | 不排序，只有两次 launch |
| 任意 | split-K：排序成 M16 / 32 / 64 块 | Down 直接 atomic 累加，无规约 kernel |
| ≥ 64，Down = default | Gate/Up 与 Down 同 M，BF16 / FP8 | 通用；M×I 激活整块放进 LDS |
| ≥ 64，Down = 1x4_64x256 | FP8，M64，N256 tile | 专用 Down，调优候选之一 |
| ≥ 64，Down = 8x1_compact | FP8，M64 排序后拆成 M256 full + M64 tail | 大 M 用 M256 tile 复用权重 |

kernel 与上游逐字节相同（结构见第 7 节的上游链接）：Gate/Up 在 `gemm1.py`；Down 在 `gemm2.py`（batch1 / split-K / default）、
`gemm2_1x4.py`、`gemm2_8x1.py` 与 `gemm2_8x1_compact.py`（任务表每次 launch 由 GPU 重建）；规约 `moe_reduce.py`；量化 `quant.py`。

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
底座与上游保持逐字节一致，pyhip 只在调用层（`tuned_moe`）做适配。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| 9 个 kernel 文件原样取自 aiter | 保留 pyhip 分叉版 | 与对照 aiter fp8 ptpc 同源，差异 0 字节 | pyhip 独有功能删除 |
| 删 MXFP4、SiTU / SiTUv2、单独 8x1、不同 M | 在新底座上重做 | 都依赖已删的旧 kernel 接口 | SiTUv2 只剩 aiter 候选 |
| prefill 规约用 Opus 排序给的 loc_ids | 自写反向索引 kernel | 与 aiter 调用链一致，少 1 次 launch | 依赖 aiter `AUX_SORT_OPUS` |
| 候选层排除两类 launch 期错误 | 只靠编译失败剪枝 | 不完整 N tile、激活 > 64 KiB 编译不报错 | 与 kernel 限制重复维护 |
| 编译缓存只按配置、不按设备 | 按设备包装缓存 | 与 aiter 一致 | 同进程多卡未验证 |
| 调优 model key 升到 7 | 沿用 6 | 旧 winner 可能指向已删的 8x1 | 旧调优结果作废 |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面：每档一行“本实现 ÷ 对照”与一句结论；对照的来源（实现、版本或提交、调用方式、
     各档实际选中的 kernel、是否与指标同进程同协议计时）；精度参考（参考实现与容差）。冻结基线由生成块第一行给出。 -->
当前没有性能数据：底座与 aiter 逐字节一致，不做性能主张；冻结基线与 aiter 对照尚未建立。
- 精度参考：独立 Torch 实现 `pyhip.testing.moe.torch_reference`，fp32 累加，FP8 时输入与中间激活按所选方式量化。
- 容差：`calc_diff ≤ 0.02`（能量归一化误差，不是逐元素 2%）；两组逐位用例要求 0 误差。
- FP32→BF16 默认 RTE（与 aiter 一致）；导入前设 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=1` 可换简化 RTA。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
底座的三条路径与三种 Down 各有基本用例，边界、报错、逐位与 graph 在其他覆盖；A8W4 尚无用例与 bench。

**基本功能测试**（每次改动先跑：`commands.correctness` = `test_fly_2stage_basic.py`，6 例，约 16 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| batch1 decode：FP8 PTPC，M1，H2560 / I320 | `test_batch1` | 通过 |
| split-K decode：M16 | `test_splitk` | 通过 |
| prefill：FP8 × 三种 Down（compact 用 I192）、BF16 × default | `test_prefill` ×3、`test_prefill_bf16` | 通过 |

**性能测试**（`commands.bench`）：无（M0 为逐字节移植）。

**其他覆盖测试**（`commands.verify_extra` = `test_fly_2stage_extra.py`，33 例，约 45 秒；用例名省略 `test_`）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| per-tensor 与 decode BF16 的 scale 分支 | 基本用例只测 PTPC 与 BF16 prefill | `splitk_variants`、`prefill_variants` | 通过 |
| SwiGLU 的 clamp 与 (up + 1)：split-K、prefill | 基本用例只测 SiLU | `swiglu` ×2 | 通过 |
| 形状交界：8x1 的 K 调度族、N512、K640 + pad 0、M·topk ≤ E | 基本用例只用 Qwen 形状与大 M | `prefill_variants`、`splitk_variants` | 通过 |
| batch1 融合清零写满输出、不越界 | 基本用例不查输出两侧 | `batch1_clear_graph` ×4 | 通过 |
| graph replay：排序、loc_ids、任务表在设备端重建 | 基本用例只跑 eager | `graph_replay` ×2、`batch1_clear_graph` | 通过 |
| 报错：K 不是 256 的倍数、Gate/Up 与 Down 的 M 不同 | 主路径只走合法配置 | `gateup_rejects_incomplete_k` ×4、`split_m_tile_rejected` | 通过 |
| 逐位：FP32→BF16 舍入、Down 打包前只舍入一次 | calc_diff 看不出舍入方式 | `bf16_rounding_bits`、`down_rounds_before_packing` ×3 | 通过 |
| 候选门槛：不完整 N tile、默认 Down 超 64 KiB；恰 64 KiB 可 launch | 前两类用固定配置，不枚举候选 | `candidate_gates` ×5、`lds_64k_boundary` | 通过 |

未覆盖 / 跳过：
- gfx950 与同进程多卡：用例只在单卡 gfx942 上跑（底座目标 gfx942）。
- per-tensor 的 Gate/Up 输出预清零：只在排序丢掉路由时起作用，正常 top-k 路由触发不到（做法同 aiter）。
- tuned_moe 的调优缓存、剪枝与非 FlyDSL 路径在 `test_tuned_moe.py` / `test_moe.py`，不在上面两条命令里。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
底座可用，风险集中在被删功能的调用方和未验证的平台；任务目标 A8W4 还没开始。
- SiTUv2 经 `fused_moe` 只剩 aiter 候选，本机 aiter 的 SiTUv2 CK 模块构建失败 → 报 No MoE config passed validation → 不在本任务范围。
- 编译缓存不按设备区分 → 同进程多卡可能复用别卡的 kernel（未验证）→ 与 aiter 一致，按每卡一进程使用。
- gfx950 上的 FlyDSL 路径未验证 → 只在 gfx942 上有保证 → 本任务不做 gfx950。
- A8W4 尚未实现 → 任务目标未达成 → 下一步做功能版：mxfp4 prepare、新入口、参考与 bench。
- 起点即失败的旧用例（`test_aiter_signature`、`test_asm_batch_loopn`）→ 全量跑旧用例会有无关失败 → 暂不处理。

## 7. 进一步阅读
- 代码入口：[tuned_moe.py](../../tuned_moe.py) 的 `_run_fly`（调用链）与 `_configs`（候选）；[__init__.py](__init__.py) 的 `compile_gemm`。
- kernel 实现：与上游 [aiter gfx9_fp8 @ca6bc0f851](https://github.com/ROCm/aiter/tree/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8) 逐字节相同。
- 上游调用链：[fused_moe_gfx942.py](https://github.com/ROCm/aiter/blob/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/fused_moe_gfx942.py)。
- 用例说明：[tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)。
- 里程碑（每个提交写了改了什么、为什么）：

<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · 底座：aiter gfx9_fp8 替换 pyhip moe_gemm_2st…
- M1 ⏳ 承诺 · 功能 · A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重

下一步（本提交的承诺）：M1 A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重
<!-- /mf:roadmap -->
