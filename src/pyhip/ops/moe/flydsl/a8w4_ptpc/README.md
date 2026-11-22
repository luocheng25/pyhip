# MoE PTPC A8W4（Qwen3.8-Flash-Next，MI308X）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-22 05:42）· 目标：pyhip 新增 MoE 算子：bf16 激活按 token 量化成 fp8、权重 mxfp4；功能正确，decode 接近访存上限、prefill 接近算力上限（TP2/TP4 单卡形状）（1.094× 相对冻结基线）· 已完成 8 个里程碑（含本提交 M7） · 最佳累计 1.322× · 下一步：写报告
<!-- /mf:summary -->

## 1. 目标与范围
在 pyhip 提供 Qwen3.8-Flash-Next 的 MoE 算子：权重 mxfp4，在 MI308X 上每个 case 基本达到同形状 FP8 ptpc 的速度。
- 输入：BF16 `x [T, 2560]` 与路由结果 `topk_ids` / `topk_weight [T, 10]`；输出 BF16 `[T, 2560]` = Σ_k w_k·Down(SiLU(Gate x)·Up x)。
- 权重：512 个专家的 mxfp4（E2M1 + 每 32 个 K 一个 E8M0），单卡分片 TP2（I=320）/ TP4（I=160），加载时转换一次。
- 激活：prefill 按 token 量化成 FP8（A8W4）；decode 保持 BF16（W4A16）。目标硬件 MI308X（gfx942，80 CU），T 从 1 到 65536。
- 验收：bench 16 个 case 的耗时都 ≤1.053× aiter_ptpc（Aiter 的 FP8 ptpc：激活按 token、权重按通道量化）；精度对 FP32 参考满足上游容差（§4）。
- 底座：`moe_gemm_2stage` 的 gfx942 BF16 / FP8 kernel 换成上游 aiter gfx9_fp8 同名文件（旧实现保留为 legacy），a8w4 kernel 由它派生。
- 不做：shared expert 融合、top-k 选择、SiLU 以外的激活、gfx950、TP4 T>8192 的 compact Down；运行时仍调 Aiter 的 `moe_sorting`。

## 2. 总体设计
按 T 分三档：T≤16 每条路由一组 CTA 做 VALU gemv，17–128 排序后 split-K，≥129 激活量化成 FP8 走两级 1x4 GEMM（gfx942 无 FP4 MFMA，fp4 在寄存器里解成 FP8）。
```text
加载时一次  prepare_weights：checkpoint 的 mxfp4（E2M1 + 每 32 K 一个 E8M0）→ MXFP4Weights（fd2 格式，I 补零到 64 倍数）
            fd2：c_max 为通道内最大的组 E8M0，因子 2^(c_max−132)；每组只存 d = c_max − code（d ≥ 12 置 0）
每次调用    fused_moe(x, weights, topk_weight, topk_ids)：T=0 直接返回，否则 select_config(T) 查 CONFIGS
  T≤16      Gate/Up gemv（顺带清零 out）→ mid（BF16）→ Down gemv（BF16 原子加 out）
  17–128    moe_sorting（BM16，清零 out）→ Gate/Up split-K → mid → Down split-K（原子加 out）
  ≥129      moe_sorting（M64）→ 量化 x（FP8）→ Gate/Up 1x4 → mid → 量化 mid → Down 1x4（每个排序行写一行）
            → invert_sorted_ids + sorted_sum（每个 token 把 top-k 行相加）→ out
```
| T（`CONFIGS`） | 路径 | 为什么 |
|---|---|---|
| 1 | W4A16 gemv（gn32 / dn64） | 只有 10 条路由，小 BN 才能多开 CTA |
| 2–16 | W4A16 gemv（gn128 / dn64） | 每路由 1 行激活；VALU 乘加不触发降频 |
| 17–128 | W4A16 split-K（BM16，BF16 MFMA） | 排序后 MFMA 划算（T=32 比 gemv 快 8%） |
| ≥129 | A8W4 1x4（M64，BK128，FP8 MFMA） | 算力受限；FP8 MFMA 吞吐是 BF16 的 2 倍 |

档位只看 `CONFIGS`（交界沿用 FP8 配置表），`config=` 可传表外配置；两档都可进 CUDA graph（capture 前先调用一次以完成编译）。
数值：两档 kernel 用的权重都与“mxfp4 反量化 + flush”逐位相同；输出与参考只差累加顺序和 BF16 舍入（中间结果、输出）。
```text
# decode gemv（T≤16，decode.py）：grid (N/BN, topk, T)；Gate/Up 4 个 wave 沿 K 切分，Down 1 个 wave
for k in range(K / (64·waves)):      # 激活先展开成 f32 放进 LDS（每 CTA 一次）；nibble + delta 双缓冲，提前 1 步读
    w ← fd2 查表(nibble, delta) → FP8 → f32；acc += w · x[k]（v_pk_fma_f32，不发 MFMA）
lane / wave 间归约 → × 通道因子 → Gate/Up：SiLU(gate)·up 写 BF16 mid；Down：× 路由权重后 BF16 原子加 out
# split-K（17–128）骨架相同：CTA 处理 16 个排序行，乘加换成 BF16 MFMA 16×16×16
# prefill Gate/Up 1x4（prefill.py）：grid (2I/BN, 排序块)；CTA = 64 排序行 × 64 对 gate/up 通道，4 个 wave 沿通道
tile_m ← 32 if 本块第 33 行是补齐行 else 64       # half_m
序言：A[0]（FP8 激活，按排序行 gather）→ LDS；B[0]（nibble + delta）→ 寄存器
for k in range(K / BK):                          # BK=128；A 经 LDS ping/pong；B 提前 1 步，32 行 tile 提前 2 步
    发起 A[k+1]、B[k+1] 的读；fd2 解码 B[k] → FP8；gate、up 各做 FP8 MFMA 16×16×32；A[k+1] 写 LDS
收尾：× 通道因子 × token scale → SiLU(gate)·up → BF16 → LDS 转置（CShuffle）→ 按 (token, k) 写 mid
# prefill Down 1x4：CTA = 64 排序行 × 全部 H 列（N 循环 H/256 次），4 个 wave 沿列，half_m 同上；FP8 mid 先 gather 进 LDS
# 每步 fd2 解码 + FP8 MFMA，权重提前 2 个 64-K 块读并跨 N tile 携带；× 因子 × 行系数后每个排序行写一行
```

## 3. 关键决策
核心取舍：decode 保住激活精度、按访存优化；prefill 用 FP8 MFMA 吃算力，fp4 在寄存器里精确解码。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| decode（T≤128）用 W4A16 | A8W4（激活也量化成 FP8） | 激活无量化误差；访存受限，已 ≤1.03× 对照 | 放弃 A8W4 约 45% 的 VALU 节省（估计） |
| T≤16 用 VALU gemv（f32 FMA） | batch1（BF16 MFMA，16 列只用 1 列） | 持续 BF16 MFMA 致降频 13%–21% | 与 batch1 不逐位（累加顺序） |
| prefill 用 fd2 把 fp4 解成 FP8 | 解成 BF16 走 BF16 MFMA | FP8 MFMA 吞吐是 BF16 的 2 倍，且精确 | scale 不到通道最大 2^-11 的组置 0 |
| half_m：后 32 行全补齐的块只算 32 行 | M32 排序（要改冻结用例） | T=1024 约 2/3 的 MFMA 落在补齐行 | 比纯 M32 慢 2.5%–3%；依赖补齐在块尾 |
| 上游 gfx9_fp8 与 legacy 并存，按配置分流 | 纯上游；合并进上游结构 | 纯上游要删改约 55 个受保护用例 | 代码 8006→14158 行，修 bug 可能改两处 |
| decode / prefill 交界保持 128/129 | 下移到 T≈32 | 下移会改冻结 bench 的计时口径与数值 | T=32–128 少拿 9%–14%（TP2） |

## 4. 性能与精度
16 个 case 全部达到验收（耗时 ≤1.053× aiter_ptpc），几何平均 0.766×；指标（冻结时未达标的 3 个 case）相对冻结基线 1.322×（目标 1.094×）。
| T 档 | a8w4 ÷ aiter_ptpc 耗时比 | 结论 |
|---|---|---|
| 1（gemv） | 0.34 | Aiter 在 T=1 慢 |
| 2–16（gemv） | 0.75–0.88 | 达标；gemv 避开降频 |
| 17–128（split-K） | 0.95–1.03 | 余量最小（TP2 T=32 为 1.028） |
| ≥129（prefill） | 0.75–0.88 | 达标；T=1024 主要靠 half_m |

- 对照 aiter_ptpc：`aiter.fused_moe.fused_moe(..., quant_type=per_Token)`，同形状 FP8 权重，/opt/aiter `43ce3d6bf292`。
- Aiter 自选 kernel：这些形状没有调优行，按默认规则各档都走 1-stage ASM（block_m=32）；T=1024 由日志确认，其余档按源码推断。
- 计时：与 a8w4 同一 bench 进程、同一[协议](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py#L1)、同一组输入；decode 档 CUDA graph replay，prefill eager。
- 对照 fly_ptpc：`tuned_moe` 的 FlyDSL FP8 ptpc（上游 gfx9_fp8 kernel），固定为 `FLY_CONFIGS`，同进程同协议，只作参考。
- 精度参考：[`reference_moe`](../../../../testing/moe_a8w4_mxfp4.py#L91)，FP32，Gate/Up 结果先舍入成 BF16；权重为 mxfp4 精确反量化 + 同一 flush；prefill 档两级输入按行量化成 FP8。
- 容差（上游 aiter `op_tests/test_flydsl_moe_gfx942.py`）：无元素偏差超过该 token 参考 RMS，logits_diff < 1e-3，最差 token 相对 L2 < 0.2。

<!-- mf:perf -->
基线来源（mf 记录）：11-21 17:35 冻结于提交 `2e5a428be2`（里程碑 M2，tag `moe-ptpc-a8w4-M2`），bench 跑 3 次取中位数，GPU0，噪声 ±0.19%；bench 命令 `cd {wt} && PYTHONPATH={wt}/src /opt/lc/work/qsa-fp8/venv-fly050/bin/python benchmarks/moe/bench_moe_a8w4_ptpc.py --mf`。对照实现与精度参考的来源见本节手写部分。

最近的性能里程碑 M4：相对冻结基线 1.322×，相对 M3 1.074×（mf 复核 V3：同卡 ABBA 4 对，GPU0，噪声 ±0.24%）。

<details><summary>逐 case（16 个，mf 生成）</summary>

| case | 基线 | 当前 | 累计加速 | 增量加速 |
|---|---|---|---|---|
| tp2_t1 | 23.2405 | 22.3905 | 1.039× | 1.043× |
| tp2_t1024 | 552.923 | 402.162 | 1.373× | 0.994× |
| tp2_t128 | 392.032 | 391.921 | 1.000× | 1.001× |
| tp2_t16 | 140.641 | 117.491 | 1.197× | 1.244× |
| tp2_t32 | 210.291 | 210.091 | 1.000× | 1.000× |
| tp2_t65536 | 12828.4 | 11686.6 | 1.099× | 1.000× |
| tp2_t8 | 74.4005 | 66.11 | 1.125× | 1.127× |
| tp2_t8192 | 1800.94 | 1568.62 | 1.148× | 0.999× |
| tp4_t1 | 19.56 | 19.25 | 1.016× | 1.021× |
| tp4_t1024 | 399.612 | 284.321 | 1.405× | 1.002× |
| tp4_t128 | 251.341 | 251.121 | 1.001× | 1.000× |
| tp4_t16 | 86.8305 | 76.7105 | 1.132× | 1.133× |
| tp4_t32 | 137.951 | 137.921 | 1.000× | 0.999× |
| tp4_t65536 | 9176.55 | 7923.48 | 1.156× | 0.999× |
| tp4_t8 | 50.2305 | 45.38 | 1.108× | 1.109× |
| tp4_t8192 | 1300.26 | 1085.64 | 1.199× | 1.000× |

</details>
<!-- /mf:perf -->

## 5. 测试覆盖
最近一次全量（代码与用例同本提交）：2141 例中 1408 通过、733 跳过（任务前已有、本机不具备条件的用例），无失败。
类别按内容分。`commands.correctness`（约 260 s）现在跑全部 2141 例，`verify_extra` 为空；
按三类原则 correctness 应只跑基本功能、其余放 `verify_extra`，改命令要重新冻结，留给后续任务。

**基本功能测试**（a8w4 主路径，TP2 / TP4 对 FP32 参考，并断言走的是配置表这一档；18 例）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| decode 各档：T=1、8、16、32、128 | 10 | 通过 |
| prefill：T=129、1024、8192、65536 | 8 | 通过 |

**性能测试**（`commands.bench`：TP2 / TP4 × T=1、8、16、32、128、1024、8192、65536，16 个 case）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| 三档各 2–3 个 T | 16 | 全部 ≤1.053× aiter_ptpc（§4） |

**其他覆盖测试**（只测前两类覆盖不了的；2123 例）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| prepare 的逐位格式、按专家分块与报错 | 容差盖不住个别字节错；报错不跑 kernel | 7 | 通过 |
| 输入检查报错、T=0、查表规则、各档下界 T=2 / 17 | 主路径只用合法输入与各档代表 T | 9 | 通过 |
| 表外配置与 gemv / batch1 互验 | 配置表不选它们 | 11 | 通过 |
| half_m 交界、预取 2 步、按行量化逐位 | 均匀路由碰不到交界；容差盖住 1 ulp | 9 | 通过 |
| CUDA graph replay 新输入（三档） | 主路径是 eager | 6 | 通过 |
| 编译缓存、构建断言与 prefill 形状边界 | 不改变输出；Qwen3.8 形状碰不到 | 4 | 通过 |
| 上游 / legacy 分流规则与 launcher 适配 | 不经过 a8w4，只能断言分流结果 | 94 | 通过 |
| 上游 FP8 路径端到端、graph、RTE | a8w4 的用例不经过这些 kernel | 34 | 通过 |
| 任务前已有的 MoE 用例（ASM、FlyDSL 等） | 回归：替换会影响 a8w4 以外的路径 | 1949 | 1216 通过 / 733 跳过 |

未覆盖 / 跳过（一行一条）：
- 真实权重：本机没有 Qwen3.8 的 mxfp4 checkpoint，测试权重是随机 nibble 与随机 E8M0 code（含 d=0..11 与 flush）。
- kernel 内的 fp4 解码没有单独的逐位用例：逐位只验证到 Python 模型与 prepare 字节，解码由端到端容差间接验证。
- 端到端只测 Qwen3.8 形状与均匀随机路由；T=65536 只比 1/16 的 token；prefill 的 graph 只测 T=1024。
- 未测：`topk_ids` 越界（不检查）、prefill `H % 512 ≠ 0` 要到 kernel 启动后才报错、按行量化遇 NaN / Inf、BM≠64 的 half_m。
- 降频没有用例，只在 bench 里可见；split-K 档是否降频只看过 T=32。
- 排除 9 个任务前就失败的旧用例：`test_asm_batch_loopn` 8 个（JIT 缓存文件名过长）、`test_aiter_signature` 1 个（/opt/aiter 的 `fused_moe` 多 3 个参数）。

## 6. 风险与待定
- 真实权重的精度未验证：flush 在 Qwen3.8 checkpoint 上的比例与误差没统计 → 拿到权重后统计 flush 比例并对 FP32 参考复测。
- split-K 档余量小（TP2 T=32 为 1.028×，门限 1.053×），仍发 BF16 MFMA，是否降频只看过 T=32 → 交界下移可一并解决。
- 交界下移待定：prefill 档在 T=32–128 已快 9%–14%（TP2；TP4 0%–7%）→ 会改冻结 bench 的口径与这些 T 的数值，需重新冻结，等用户决定。
- 运行时依赖 Aiter `moe_sorting`，half_m 还依赖它把补齐放在专家末块的末尾 → 升级 Aiter 后重跑用例；自带排序留作后续。
- 形状范围：只调过 Qwen3.8；TP4 T>8192 无 compact Down，prefill 要求 `H % 512 == 0`、补零后 I ≤ 832 → 换模型前要扩展。

## 7. 进一步阅读
- 代码入口：[`fused_moe`](moe.py#L85)、配置表 [`CONFIGS`](moe.py#L36)（注释里有各档选择的实测依据）、[`prepare_weights`](weights.py#L130)。
- 格式与 kernel：[weights.py 模块说明](weights.py#L1)（fd2 与精确性论证）；[decode.py](decode.py#L1)、[prefill.py](prefill.py#L1) 文件头（相对上游的改动）。
- 接口细节：[`compile_decode_gemm`](decode.py#L1459)、[`compile_prefill_gemm`](prefill.py#L1266) 的 docstring（launcher 参数、可选 tile）。
- 上游替换：[深度参考](m0_upstream_replacement.md)（`tuned_moe` FP8 接口与调用链、同名文件改动、分流规则、已知限制）。
- 用例与 bench：[tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)、[benchmarks/moe/README.md](../../../../../../benchmarks/moe/README.md)。
- 里程碑（每个提交写了改了什么、为什么）：

<!-- mf:roadmap -->
- M0 ✅ 已推送 · 功能 · 把上游 aiter gfx9_fp8 的 kernel 更新到 pyhip m… · [提交](https://github.com/luocheng25/pyhip/commit/e433e8d9184cdc5fd387641a00bc5686abec37cc)
- M1 ✅ 已推送 · 功能 · W4A16 decode：mxfp4 权重格式 + decode 路径（配置表… · [提交](https://github.com/luocheng25/pyhip/commit/cc271e29d151356c40607da4db96af8ea068e459)
- M2 ✅ 已推送 · 功能 · A8W4 prefill：fd2 解码接入 fp8 MFMA（配置表 pref… · [提交](https://github.com/luocheng25/pyhip/commit/2e5a428be2a332b3e45fe3879b4391f26fee0b8c)
- M3 ✅ 已推送 · 性能 · T=1024 prefill 追上 aiter_ptpc（TP2/TP4） · 1.238× 对基线（预期 1.1×） · [提交](https://github.com/luocheng25/pyhip/commit/a294918e917414d1b8b72dfce070209fb3565920)
- M4 ✅ 已推送 · 性能 · TP2 T=16 decode 追上 aiter_ptpc（假设 H1：w1 … · 1.074× 对 M3（预期 1.05×） · [提交](https://github.com/luocheng25/pyhip/commit/5c562dc418d5cf405015786fdd938e43756a6e8c)
- M5 ✅ 已推送 · 维护 · 入口文档按最终结果更新（顶部摘要、各档路径一览、M4 标题更正） · [提交](https://github.com/luocheng25/pyhip/commit/f48b61744b2383f7827ba3da5bbd2d1746b57225)
- M6 ✅ 已推送 · 维护 · 入口文档补“基线与对照的来源”，并按篇幅预算收拢 · [提交](https://github.com/luocheng25/pyhip/commit/4c0c98887e7e002d4525e33d935badf344ceeef6)
- M7 ✅ 本提交 · 维护 · 入口文档按设计文档标准重写（给 tech lead 的设计说明）

下一步（本提交的承诺）：写报告
<!-- /mf:roadmap -->
