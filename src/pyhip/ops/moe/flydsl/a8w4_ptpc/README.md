# MoE PTPC A8W4（Qwen3.8-Flash-Next，MI308X）：入口

<!-- mf:summary -->
> 由 mf 生成（11-22 02:57）· 目标：pyhip 新增 MoE 算子：bf16 激活按 token 量化成 fp8、权重 mxfp4；功能正确，decode 接近访存上限、prefill 接近算力上限（TP2/TP4 单卡形状）（1.094× 相对冻结基线）· 已完成 7 个里程碑（含本提交 M6） · 最佳累计 1.322× · 下一步：写报告
<!-- /mf:summary -->

> - **这是什么**：Qwen3.8-Flash-Next MoE（H=2560、E=512、top-10、SiLU，TP2 / TP4 单卡分片）在 MI308X 上的 A8W4 算子：mxfp4 权重 + 按 token 量化的 FP8 激活（decode 档保持 BF16 激活，见下一条）。入口是 [`prepare_weights`](weights.py#L130)（加载时把 checkpoint 的 mxfp4 转成 fd2 格式，只做一次）和 [`fused_moe`](moe.py#L85)，按配置表 [`CONFIGS`](moe.py#L36) 分档（各档路径与对照见 §2）。kernel 由 M0 移植到 `moe_gemm_2stage/` 的上游 aiter `gfx9_fp8` 派生，M0 只是底座。
> - **为什么这样设计**：decode（T≤128）用 W4A16（用户 11-21 选 A）：激活保持 BF16、mxfp4 在寄存器里精确展开，T=17–128 先排序走 split-K 的 BF16 MFMA；T≤16 每条路由只有一行激活，改用 VALU 上 f32 乘加的 gemv，因为持续的 BF16 MFMA 会让 MI308X 降频（M4）。prefill（T≥129）用 A8W4：gfx942 没有 FP4 MFMA，fp4 经 fd2 查表精确解成 FP8 进 FP8 MFMA（M2），M64 排序块的后 32 行全是补齐行时两级只算前 32 行、否则算满 64 行（M3）。两档 kernel 用的权重都与“mxfp4 反量化 + flush”逐位相同，输出对 FP32 参考满足 M0 的容差（§4）。
> - **现在最大的风险**：精度只在随机 mxfp4 权重上验证过：fd2 会把 scale 不到本通道最大组 scale 的 2^-11 的组整组置 0（flush），本机没有 Qwen3.8 的 mxfp4 checkpoint，真实权重上被置 0 的比例和由此带来的误差都没统计（[M1](history.md#m1)、[M2](history.md#m2) 已知问题）。性能余量最小的是 split-K 档（T=17–128）：TP2 T=32 / 128 为 aiter_ptpc 的 1.028 / 1.018×（门限 1.053×），它仍发 BF16 MFMA，是否降频只看过 T=32（[M4 已知问题](history.md#m4)）。

## 1. 路线图
<!-- mf:roadmap -->
| 里程碑 | 状态 | 类型 | 标题 | 增量（预期） | 累计 | 用例 | 复核 | 提交 |
|---|---|---|---|---|---|---|---|---|
| M0 | ✅ 已推送 | 功能 | 把上游 aiter gfx9_fp8 的 kernel 更新到 pyhip moe_gemm_2stage（同名文件替换 / 更新） | — | — | ✓ | 冷读✓ 3分 | [moe-ptpc-a8w4-M0](https://github.com/luocheng25/pyhip/commit/e433e8d9184cdc5fd387641a00bc5686abec37cc) |
| M1 | ✅ 已推送 | 功能 | W4A16 decode：mxfp4 权重格式 + decode 路径（配置表 decode 档，当前 T≤128） | — | — | ✓ | ✓ | [moe-ptpc-a8w4-M1](https://github.com/luocheng25/pyhip/commit/cc271e29d151356c40607da4db96af8ea068e459) |
| M2 | ✅ 已推送 | 功能 | A8W4 prefill：fd2 解码接入 fp8 MFMA（配置表 prefill 档，当前 T≥129） | — | — | ✓ | ✓ | [moe-ptpc-a8w4-M2](https://github.com/luocheng25/pyhip/commit/2e5a428be2a332b3e45fe3879b4391f26fee0b8c) |
| M3 | ✅ 已推送 | 性能 | T=1024 prefill 追上 aiter_ptpc（TP2/TP4） | 1.238× 对基线（预期 1.1×） | 1.238× | ✓ | ✓ | [moe-ptpc-a8w4-M3](https://github.com/luocheng25/pyhip/commit/a294918e917414d1b8b72dfce070209fb3565920) |
| M4 | ✅ 已推送 | 性能 | TP2 T=16 decode 追上 aiter_ptpc（假设 H1：w1 每 lane 每轮读 16 B） | 1.074× 对 M3（预期 1.05×） | 1.322× | ✓ | ✓ | [moe-ptpc-a8w4-M4](https://github.com/luocheng25/pyhip/commit/5c562dc418d5cf405015786fdd938e43756a6e8c) |
| M5 | ✅ 已推送 | 维护 | 入口文档按最终结果更新（顶部摘要、各档路径一览、M4 标题更正） | — | — | ✓ | ✓ | [moe-ptpc-a8w4-M5](https://github.com/luocheng25/pyhip/commit/f48b61744b2383f7827ba3da5bbd2d1746b57225) |
| M6 | ✅ 本提交 | 维护 | 入口文档补“基线与对照的来源”，并按篇幅预算收拢 | — | — | ✓ | ✓ | 本提交 |

下一步（本提交的承诺）：写报告
<!-- /mf:roadmap -->

## 2. 性能

**各 T 档路径与对照**（最终状态，即 M4 提交；a8w4 是 mf 复核 V3 的同卡 ABBA 4 对、GPU0，aiter_ptpc 取同次 8 次运行的中位数，见任务目录 `notes/r1-final-percase.txt`；括号里是 a8w4 ÷ aiter_ptpc 的耗时比，<1 为 a8w4 更快）：

| T 档（配置表） | 算法 / tile | bench T | TP2 µs（÷ aiter_ptpc） | TP4 µs（÷ aiter_ptpc） |
|---|---|---|---|---|
| 1 | decode W4A16 · gemv（VALU f32 乘加）：Gate/Up gn32、Down dn64 | 1 | 22.4（0.340） | 19.2（0.342） |
| 2–16 | decode W4A16 · gemv：gn128 / dn64 | 8 | 66.1（0.819） | 45.4（0.748） |
| 〃 | 〃 | 16 | 117.5（0.876） | 76.7（0.769） |
| 17–128 | decode W4A16 · split-K（BM16 排序，BF16 MFMA）：gn128 / dn128 | 32 | 210.1（1.028） | 137.9（0.971） |
| 〃 | 〃 | 128 | 391.9（1.018） | 251.1（0.946） |
| ≥129 | prefill A8W4 · M64 排序，Gate/Up 1x4 gn128 BK128 → Down 1x4 M64×N256（FP8 MFMA）；后 32 行全补齐的块只算 32 行 | 1024 | 402.2（0.863） | 284.3（0.861） |
| 〃 | 〃 | 8192 | 1568.6（0.844） | 1085.6（0.749） |
| 〃 | 〃 | 65536 | 11686.6（0.878） | 7923.5（0.747） |

- 结论：16 个 case 全部 ≤1.053× aiter_ptpc（SPEC 的每 case ≥0.95×；最慢的是 TP2 T=32，1.028×），几何平均 0.766×。
- 后续：把 decode/prefill 交界下移可让 T=32..128 再快 9%–14%（TP2，L1 自测；TP4 最多约 7%，见 [M3 已知问题](history.md#m3)），但会改变冻结 bench 的计时口径、这些 T 也会从 W4A16 变成 A8W4，要重新冻结，等用户决定。

<!-- mf:perf -->
基线来源（mf 记录）：11-21 17:35 冻结于提交 `2e5a428be2`（里程碑 M2，tag `moe-ptpc-a8w4-M2`），bench 跑 3 次取中位数，GPU0，噪声 ±0.19%；bench 命令 `cd {wt} && PYTHONPATH={wt}/src /opt/lc/work/qsa-fp8/venv-fly050/bin/python benchmarks/moe/bench_moe_a8w4_ptpc.py --mf`。对照实现与精度参考的来源见“基线与对照的来源”一节。

M4 的复核（V3：同卡 ABBA 4 对，GPU0，噪声 ±0.24%，冻结基线 `2e5a428be2`）

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
<!-- /mf:perf -->

- 解读：上面的生成块是 M4 的 mf 复核，累计一列是 V3 相对冻结基线（M2），增量一列是 V4 相对 M3；M0–M2、M5、M6 没有性能主张。验收口径（用户 11-21 确认）是每个 case 的 `latency_us` 基本达到同形状 aiter_ptpc 的水平（耗时 ≤1.053×）；指标只计冻结时未达到的 3 个 case（TP2 T=16、TP2 / TP4 T=1024），目标是相对基线加权几何平均 1.094×（任务目录 `plan.md`），实际 1.322×，其余 case 由每 case 回退 ≤2% 把关。
- 各里程碑的自测、M0 替换前后与 `bench_tuned_moe` 的对比见 [history.md §2](history.md#perf)。

### 基线与对照的来源
- **冻结基线**：见上方生成块第一行（mf 记录）。
- **性能对照 aiter_ptpc**（bench 的 `REF impl=aiter_ptpc` 行；验收按它换算，见任务目录的 SPEC 与 `plan.md`）：
  - 调用：`aiter.fused_moe.fused_moe(hidden, w1, w2, topk_weight, topk_ids, activation=Silu, quant_type=per_Token, w1_scale, w2_scale)`，不指定 kernel，由 Aiter 自选（[bench 的 `make_call`](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py#L157)）。
  - 版本：Aiter 来自 `/opt/aiter`（bench 与用例的解释器都从这里导入），提交 `43ce3d6bf292`（2026-11-14），任务期间没有换过（git reflog）；工作区另有未提交改动（`3rdparty/composable_kernel` 子模块不在记录的提交、几个未跟踪的 tuned gemm csv）。
  - 计时：与指标 a8w4 同一个 bench 进程、同一协议、同一组输入（[bench 文件头](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py#L1)）：≥2 轮预热后连续运行 1 s 让频率稳定，再取 ≥20 个样本的中位数；输入至少轮换 4 组（小 T 加组，使两次复用之间触达 ≥512 MiB 专家权重）；每个样本前放一个 sleep kernel 挡住主机下发；decode 档（按 a8w4 配置表，T≤128）用 CUDA graph replay，prefill 为 eager。§2 表里的 aiter_ptpc 是 V3 的 8 次运行（`runs/20261121-201616-int-verify-8814`）取中位数。
  - 选中的 kernel：T=1024 没有调优行，走 Aiter 默认的 1-stage ASM、block_m=32 排序：TP2 `fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x320`，TP4 `fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x192`（`runs/20261121-174458-L1-run-c3cd/out.log`，L1 的 profiler；另见 `mf show` 的笔记、`notes/L1-report.md`、`notes/m3-tile-survey.md`）。其余档未逐档核对：bench 带 `--mf` 时把 `AITER_LOG_LEVEL` 设成 WARNING（[L415](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py#L415)），设 `AITER_LOG_LEVEL=INFO` 再跑即可看到 Aiter 打印的 `[fused_moe] using …`、`run_1stage = … block_m = …` 与 `LoadKernel: …`。
  - 权重：与 fly_ptpc 同一份 FP8（E4M3FNUZ，按输出通道量化，经 `shuffle_weight`）；TP4 用补零到 I=192 的这一份（[`make_weights`](../../../../testing/moe_a8w4_ptpc.py#L114)）。
- **性能对照 fly_ptpc**（`REF impl=fly_ptpc`）：`tuned_moe` 的 FlyDSL FP8 ptpc 候选（M0 路径），经 `make_moe_runner` 固定为 [`FLY_CONFIGS`](../../../../testing/moe_a8w4_ptpc.py#L41)（不调优、不回退 Aiter），同进程、同协议。与 a8w4 比时注意：a8w4 的 prefill 排序用 `accumulate=False`，省掉 fly_ptpc 的两次 T×H 清零，比值略偏向 a8w4；TP4 T=65536 的 fly_ptpc 用 compact Down，a8w4 只有 1x4。
- **精度参考**：FP32 计算，Gate/Up 结果先舍入成 BF16（kernel 以 BF16 存中间结果）。a8w4 用 [`moe_a8w4_mxfp4.reference_moe`](../../../../testing/moe_a8w4_mxfp4.py#L91)：权重为 mxfp4 精确反量化 E2M1·2^(code−127)，并施加 kernel 的 flush（比本通道最大 code 小 12 以上的组为 0）；decode 档激活为 BF16，prefill 档两级输入按行量化成 FP8。FP8（fly_ptpc、aiter_ptpc 与 `Q` 用例）用 [`moe_a8w4_ptpc.reference_moe`](../../../../testing/moe_a8w4_ptpc.py#L200)（权重为反量化的 FP8）。容差沿用上游 aiter `op_tests/test_flydsl_moe_gfx942.py`（/opt/aiter 的 L64–L67），以 [`check_output`](../../../../testing/moe_a8w4_ptpc.py#L253) 与常量（[L24–L26](../../../../testing/moe_a8w4_ptpc.py#L24)）为准：输出全部有限；没有元素偏差超过该 token 参考值 RMS 的 1 倍；`logits_diff`（`1 − 2·Σxy / Σ(x² + y²)`）< 1e-3；最差 token 的相对 L2 误差 < 0.2。bench `--check` 用同一组常量判定（[`output_errors`](../../../../testing/moe_a8w4_ptpc.py#L242)）。

## 3. 接口与行为（a8w4）
入口 `pyhip.ops.moe.flydsl.a8w4_ptpc`（[`__init__.py`](__init__.py)）。M0 的 FP8 `tuned_moe` 接口、`compile_gemm` 分流规则（B1–B30）及其环境变量见 [history.md §3.2](history.md#interfaces)。

| 符号 / 环境变量 | 签名 / 取值 | 约束 / 报错 | 说明 |
|---|---|---|---|
| [`prepare_weights`](weights.py#L130) | `(w1, w1_scale, w2, w2_scale, *, chunk=32)` → `MXFP4Weights` | dtype 不对 → `TypeError`；维数 / 形状不一致、`H % 256 ≠ 0`、`I % 32 ≠ 0` 或 I=0、E8M0 code 255 → `ValueError` | 加载时调用一次。输入是单卡分片的 checkpoint 自然布局：w1 uint8 `[E, 2I, H/2]`（一字节两个 E2M1，低半字节是偶数 k；前 I 行 gate、后 I 行 up），w1_scale E8M0 `[E, 2I, H/32]`，w2 `[E, H, I/2]`，w2_scale `[E, H, I/32]`；也接受 `float4_e2m1fn_x2` / `float8_e8m0fnu`。I 补零到 64 的倍数（TP4 160→192） |
| [`MXFP4Weights`](weights.py#L48) | frozen dataclass：`w1` / `w2`（fd1 排列的 nibble）、`*_delta`（decode 用）、`*_pdelta`（prefill 用）、`*_factor`、`inter`、`inter_padded`；属性 `experts`、`hidden` | 只由 `prepare_weights` 构造 | 格式与精确性论证见 [weights.py 模块说明](weights.py#L1) |
| [`fused_moe`](moe.py#L85) | `(hidden_states, weights, topk_weight, topk_ids, *, output=None, config=None)` → `output` | 见下面“输入检查” | `Σ_k topk_weight · Down(SiLU(Gate x) · Up x)`，只有 SiLU；不调优、不回退 Aiter |
| [`select_config`](moe.py#L46) / [`decode_max_tokens`](moe.py#L54) / [`CONFIGS`](moe.py#L36) | `select_config(tokens, configs=CONFIGS)` → 配置副本；`decode_max_tokens()` → decode 档上界（当前 128） | 不报错；T 超过所有上界时取最后一行 | 档位只看 `CONFIGS`，代码里不写交界 |
| [`compile_decode_gemm`](decode.py#L1459) / [`compile_prefill_gemm`](prefill.py#L1266) / [`compile_prefill_quant`](prefill.py#L1295) | 内部，由 `fused_moe` 调用 | 不支持的 tile / 形状由 builder 的 `assert` 拒绝（`AssertionError`） | 按设备缓存，同一设备同一配置只构建一次 |
| 环境变量 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED` | 默认 `1` | 导入 `moe_gemm_2stage/common.py` 时读取，进程内再改不生效 | FP32→BF16 舍入：1 为简化 RTA，0 为 RTE；a8w4 的 kernel 与 M0 共用这一开关 |

输入检查（[`_check`](moe.py#L67) 与 [`fused_moe`](moe.py#L85) 开头，在任何 kernel 启动之前；不满足就报错，不转交 Aiter）：
- `weights` 不是 `MXFP4Weights` → `TypeError`；`hidden_states` 不是 BF16 `[T, H]` CUDA 张量、H 或设备与权重不符、`topk_ids` / `topk_weight` 不是 int32 / fp32 `[T, topk]`、任一张量不连续或不在同一设备、`output` 不是与 `hidden_states` 同形状的 BF16 → `ValueError`；`config` 的 path 不是 decode / prefill → `ValueError`。
- 不检查：`0 ≤ topk_ids < E` 由调用方保证。prefill 档要求 `H % 512 == 0`、补零后 I ≤ 880（即 I ≤ 832），是编译期 `AssertionError`：I 超限在启动 Gate/Up 之前报；H 不满足要到 Gate/Up、Down 已启动后构建 `sorted_sum` 时才报，`out` 不写。Qwen3.8（H=2560，I=320 / 192）都满足。

dispatch（[`select_config`](moe.py#L46) 查 [`CONFIGS`](moe.py#L36)；各档耗时与对照见 §2）：
- T=0 → 不编译、不启动，直接返回 `output`（不传就新建）。
- T=1 → decode W4A16 · gemv gn32 / dn64；T=2–16 → gemv gn128 / dn64：每条路由一组 CTA、不排序，在 VALU 上做 f32 乘加（不发 MFMA，避开降频），Gate/Up 顺带清零 `out`，Down 原子加。
- T=17–128 → decode W4A16 · split-K gn128 / dn128：Aiter `moe_sorting`（BM16，同时清零 `out`），BF16 MFMA，Down 原子加。
- T≥129 → prefill A8W4（[`_fused_moe_prefill`](moe.py#L142)）：Aiter `moe_sorting`（M64，`accumulate=False`）→ FlyDSL 按行把 `x` 量化成 FP8 → Gate/Up 1x4 gn128 BK128（fp4 经 fd2 查表解成 FP8，FP8 MFMA）写 BF16 `mid` → `mid` 按 (token, k) 量化 → Down 1x4 M64×N256，每个排序行写一行（行距补 128 B）→ `invert_sorted_ids` + `sorted_sum` 写全部 `out`。后 32 行全是补齐的排序块两级只算前 32 行（`half_m`）；Gate/Up 的 32 行 tile 提前 2 个 K tile、Down 提前 2 个 K core 读权重。
- `config=` 可走表外配置：decode `alg` ∈ {`gemv`, `batch1`, `splitk`}（batch1 只经 `config=` 可达），Gate/Up BN：gemv / batch1 ∈ {32, 64, 128}、split-K ∈ {64, 128}，Down ∈ {64, 128}；prefill Gate/Up BN、BK ∈ {128, 256}，Down 只有 1x4；`quant` 不是 `"fly"` 时用 Aiter 量化（不报错）；缺 `half_m` / `prefetch_*` / `quant` 键时按 M2（`False`、1、1、Aiter 量化）。
- 运行时依赖 Aiter 的 `moe_sorting`（split-K 与 prefill），kernel 文件不导入 Aiter；两档都可进 CUDA graph（capture 前先调用一次完成编译）。

## 4. 测试覆盖
用例在 `tests/ops/moe/a8w4_ptpc/`（共 192 个）：`U` = [test_moe_gemm_2stage_upstream.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_moe_gemm_2stage_upstream.py)（94），`Q` = [test_qwen38_ptpc.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_qwen38_ptpc.py)（34），`D` = [test_a8w4_decode.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_decode.py)（17），`DE` = [test_a8w4_decode_edges.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_decode_edges.py)（12），`G4` = [test_a8w4_decode_gemv.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_decode_gemv.py)（4），`PF` = [test_a8w4_prefill.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_prefill.py)（14），`PE` = [test_a8w4_prefill_edges.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_prefill_edges.py)（7），`P3` = [test_a8w4_prefill_m3.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_a8w4_prefill_m3.py)（10）；`TM` / `TT` = `tests/ops/moe/test_moe.py` / `test_tuned_moe.py`（任务前已有）。精度用例的参考与容差见 §2“基线与对照的来源”，没有放宽。本表累计 M0–M6（M5、M6 只改文档）；各里程碑的原始覆盖表见 [history.md §4](history.md#tests)。

| 情况 | 用例（文件::名字） | 类型 | 结果 |
|---|---|---|---|
| decode 各档端到端：TP2 / TP4 × T ∈ {1（gemv gn32）, 8, 16（gemv gn128）, 32, 128（split-K）}，断言编译的两级 kernel 就是配置表这一档 | D::test_decode_matches_reference（10） | 精度 | 通过 |
| decode 档内下界与交界另一侧：T=2（gemv，TP4 不传 `output`）、T=17（split-K） | DE::test_decode_tier_lower_edges（4） | 精度 | 通过 |
| decode 表外 tile（`config=`）：batch1 gn128 / dn128、split-K gn64 / dn64、gemv gn64 + dn128 | DE::test_config_override_tiles（2）、G4::test_gemv_config_tiles（2） | 精度 | 通过 |
| gemv 与 batch1 同一输入都满足容差；每个 token 只留一条路由时逐元素 ≤1 个 BF16 ulp、≥99% 逐位相同 | G4::test_gemv_matches_batch1（2） | 精度 | 通过 |
| prefill 端到端：TP2 / TP4 × T ∈ {129（交界）, 1024, 8192, 65536（每 16 个 token 比 1 个）}，断言编译的是配置表 prefill 档、不经过 decode kernel | PF::test_prefill_matches_reference（8） | 精度 | 通过 |
| half_m 交界：专家行数 31 / 32 / 1 → 只算前 32 行，33 → 整块，64 / 65 / 96 / 97 → 整块加按余数的末块，同一次调用两种块并存；逐块断言 Down 的路径，`mid` 与输出和 M2 路径逐位相同 | P3::test_half_m_dispatch_boundary（2） | 精度 | 通过 |
| 预取 2 步在 K tile 少的形状（Down K=64 / 128，Gate/Up BK256 / 128），与 M2 路径逐位相同 | P3::test_prefetch_two_with_few_k_tiles（2） | 精度 | 通过 |
| FlyDSL 按行量化对参考 `quantize_rows` 逐位：宽度 2560 / 320 / 192 / 256 / 128 × 行数 1、10、33、1027；全零行、±BF16 最大值、FP8 中点与次正规数；越过 `rows` 的行不写 | P3::test_quant_rows_matches_reference_bitwise（5） | 精度 | 通过 |
| prefill 表外配置：Gate/Up BK256、小形状上 BN256 + BK256、`quant="aiter"` | PF::test_prefill_other_tiles、P3::test_quant_aiter_option | 精度 | 通过 |
| prefill 用于小 T（`config=`）：T=1、51（T·topk ≤ E 的分支）、52 | PE::test_prefill_override_small_tokens（3） | 精度 | 通过 |
| prefill 形状边界：H=512 × 补零后 I=832（BK128 / 256）；I=896 在启动前报 `AssertionError`、不写输出 | PE::test_prefill_min_hidden_max_inter | 精度 / 其他 | 通过 |
| prepare 逐位：解回的权重 = mxfp4 反量化 + flush + 补零；字节布局（nibble 顺序、−0→+0、flush、delta、因子）与 prefill 的 delta 排列逐字节；按专家分块（含最后一块不满）结果不变 | D::test_prepare_matches_dequant_bitwise、test_prepare_byte_layout，PF::test_prefill_delta_layout_bitwise，DE::test_prepare_chunk_does_not_change_result，PE::test_prefill_delta_partial_last_chunk | 精度 | 通过 |
| prepare 报错：code 255、形状 / 维数不一致、2I 为奇数、`H % 256`、`I % 32`、int16 → `TypeError`；fp4x2 / e8m0 dtype 与 uint8 结果相同 | D::test_prepare_rejects_invalid、DE::test_prepare_rejects_other_invalid | 其他 | 通过 |
| `fused_moe` 输入检查：16 种非法输入在启动前报错（不编译、不写输出）；`config` 的 path 未知 → `ValueError` | DE::test_fused_moe_rejects_invalid_inputs、PF::test_unknown_path_rejected_before_launch | 其他 | 通过 |
| 空输入 T=0：decode 档与 `config=` 指定的 prefill 档都不编译、不启动 | DE::test_empty_input_returns_without_launch、PE::test_prefill_override_empty_input | 其他 | 通过 |
| 配置表查表：各行上下界、超过所有上界取最后一行、返回副本 | DE::test_select_config_rules | 其他 | 通过 |
| CUDA graph：gemv gn32（TP2 T=1）、gemv gn128（TP4 T=8）、split-K（TP2 T=32、TP4 T=128）、prefill（TP2 / TP4 T=1024），replay 两组新输入与路由 | D::test_decode_cuda_graph（4）、PF::test_prefill_cuda_graph（2） | 其他 | 通过 |
| 编译缓存与构建断言：同一设备同一配置只构建一次、换设备 key 重建；decode BM ≠ 16、prefill BK ∉ {128, 256}、Down tile ≠ 64×256、Gate/Up 的 K % 256、K tile 数为奇数、N % BN，Down 的 N % 256、LDS 超限被拒绝 | DE::test_compile_decode_gemm_cached_per_device、PF::test_compile_prefill_gemm_cached_per_device、PE::test_compile_prefill_gemm_shape_asserts | 其他 | 通过 |
| BF16 舍入切到 RTE（`AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=0`） | D、DE、G4（33）；PF、PE（21） | 精度 | 通过（复核：`runs/20261121-205636-_task-run-376f`、`runs/20261121-171057-_task-run-48ee`） |
| 性能：TP2 / TP4 × T ∈ {1, 8, 16}（gemv）、{32, 128}（split-K）、{1024, 8192, 65536}（prefill）共 16 个 case；decode 档 graph 计时，aiter_ptpc、fly_ptpc 为对照行 | [bench_moe_a8w4_ptpc.py](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py) `--mf`（mf 复核 V3 / V4，同卡 ABBA 4 对） | 性能 | 16 个 case 全部 ≤1.053× aiter_ptpc，几何平均 0.766×；相对冻结基线加权 1.322×（§2） |
| M0 分流规则 B1–B29：每条至少一例，断言判定结果、实际调用的 builder，以及上游 builder 能否构建 | U::test_compile_gemm_routes（68） | 其他 | 通过 |
| M0 适配与环境：launcher 包装（12）、每设备只判定一次、`MOE_PREFILL_TILE_K` / `CU_NUM`、原名符号、屏蔽 aiter 后能导入、`_run_compiled` 失败后可恢复、compact 容量、舍入开关与位型、quant 契约 | U 的其余 26 个 | 其他 / 精度 | 通过 |
| M0 FP8 端到端（`tuned_moe` 固定 `FLY_CONFIGS`）：TP2 / TP4 × T ∈ {1, 8, 16, 32, 128, 129, 256, 257, 1024, 2000, 8192, 65536}，TP4 另加 8200；断言两级都判给上游、是配置表这一档 | Q::test_matches_reference（25） | 精度 | 通过 |
| M0 FP8 decode 进 graph、RTE 下同一容差、`has_num_local_tokens` 计数之后的行不写 | Q::test_decode_cuda_graph（4）、test_rte_rounding_matches_reference（4）、test_num_local_tokens_skips_padding_rows | 其他 / 精度 | 通过 |
| 任务前已有的 MoE 用例：gfx942 上 bf16 / ptpc / per_tensor × SiLU / SwiGLU 判给上游；SiTUv2、独立 8x1、compact K=320、legacy 报错文本走 legacy | TM、TT | 精度 / 其他 | 通过（两卡、FP4 / gfx950 用例本机跳过） |
| 全部功能用例（task.json 的正确性命令，`-m "not perf"`） | TM、TT 与 `a8w4_ptpc/` 全部 | 精度 / 其他 | 1408 通过 / 733 跳过 / 191 deselected（M5 收尾，与本提交同一份代码与用例，`runs/20261121-212555-_ms-close-2918`） |

未覆盖 / 跳过（一行一条）：
- flush 对真实权重的影响没统计：本机没有 Qwen3.8 的 mxfp4 checkpoint；测试权重是随机 nibble 与随机 E8M0 code（覆盖 16 个编码、d=0..11 与 flush），不是按 OCP 规则量化的权重。
- kernel 在寄存器里的 fp4 解码（→BF16 / FP8）没有单独的逐位用例：逐位只验证到 Python 模型 `decode_prepared` 与 `*_pdelta` 字节，解码本身由端到端容差间接验证。
- 只测 Qwen3.8 形状（H=2560、E=512、topk=10、I=320 / 160→192）；其它形状只出现在 CPU 的 prepare、prefill 小形状与边界用例里；Gate/Up 的 K tile 数为奇数的收尾只在 Down 上测到。
- prefill 的 `H % 512 ≠ 0`（如 H=256）到构建 `sorted_sum` 时才报 `AssertionError`，没有用例固定这一行为（复核实测，`runs/20261121-170845-_task-run-0d5c`）。
- compact Down（TP4 T>8192）没有 A8W4 版本，prefill 一律 1x4（推迟到性能阶段，M3、M4 都没做）。
- T=65536 只比较 1/16 的 token；prefill 的 graph 只测 T=1024；路由只测均匀随机，少数专家占满的不均匀路由（含 half_m 的收益）没测。
- half_m 只在 BM=64 测：BM=128 / 256、32 行 tile 配 `prefetch_gate=1`、`prefetch_down` ≥ 3 只经 `compile_prefill_gemm` 可达，未测；half_m 依赖 Aiter `moe_sorting` 把补齐放在每个专家末块的末尾（P3 断言了这一点，换排序实现要重查）。
- 按行量化：scale 成为 fp32 次正规数（|x| 约 1e-36 以下）、NaN / Inf 输入、`cols % 8 ≠ 0` 的断言未测；P3 不在 RTE 下复跑（M3 没改 BF16 舍入）。
- 带全部路由的输出不逐元素比较 gemv 与 batch1：Down 的 BF16 原子加顺序不定，同一算法连跑两次也约一半元素不同（`runs/20261121-205153-_task-run-af65`），只断言两者都满足容差。
- `topk_ids` 越界（≥E 或负数）不检查、未测；多卡上的编译缓存只用替换设备 key 模拟；FlyDSL 磁盘编译缓存在用例里关闭；非 gfx942、两卡、FP4 / gfx950 的用例本机跳过。
- 降频本身没有用例，只在 bench 与 L2 的 `tools/decode_prof.py`（任务目录）里可见；split-K 档是否降频只看过 T=32。
- bench `--check` 超出容差时返回 1 没有用例（bench 不在用例范围）。
- 正确性命令排除 9 个替换前（8027faad）就失败的用例：`test_asm_batch_loopn` 8 个（ASM JIT 缓存文件名含 worktree 长路径，Errno 36）、`test_aiter_signature` 1 个（/opt/aiter 的 `fused_moe` 多 3 个参数）。
- 冗余（不删）：DE::test_select_config_rules 与 D 里的 `select_config` 断言；PE 的 Down LDS 断言与 I=896；PF 的 t129 与 P3 里“只算前 32 行”的一侧。

运行：`python -m pytest -q tests/ops/moe/a8w4_ptpc`（192 个）；RTE 下复跑加 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=0`；按文件运行见 [history.md §4 命令](history.md#commands)。

## 5. 里程碑（最新在上）
### M6 · 入口文档补“基线与对照的来源”，并按篇幅预算收拢（本提交）
- **做了什么**：§2 新增“基线与对照的来源”（冻结基线、aiter_ptpc、fly_ptpc、精度参考，各给出处）；M0–M5 的细节（M0 的 FP8 `tuned_moe` 设计、各里程碑的自测、覆盖明细、决策与死路、原 §7 各节）原样移到新建的 [history.md](history.md)，只加小标题与锚点；本文 §3 收成当前 a8w4 的接口与 dispatch，§4 合成一张覆盖表加未覆盖清单，§5 每个旧里程碑一行；[m0_upstream_replacement.md](m0_upstream_replacement.md) 与 [moe_gemm_2stage/README.md](../moe_gemm_2stage/README.md) 里指向旧章节的说法改指 history.md。
- **为什么**：用户 11-22 要求入口文档写明基线与对照的来源，并把入口文档控制在 250 行以内（不含 mf 生成块）：原来 593 行，现在 140 行。只改文档，代码、用例、bench 都没改，数值不变。
- **与计划 / 上游的差异**：无。
- **已知问题**：aiter_ptpc 在 T=1024 以外各档选的 kernel 未逐档核对（查看方法见 §2 来源一节）；history.md 里的代码行号是写入时的，当前接口以 §3 为准。

更早的里程碑（详情是 history.md 里的原文）：
- M5 · 入口文档按最终结果更新（顶部摘要、各档路径一览、M4 标题更正） · 只改文档：顶部 3 行与 §2 的各档路径表按最终状态重写 · [提交](https://github.com/luocheng25/pyhip/commit/f48b61744b2383f7827ba3da5bbd2d1746b57225) · [详情](history.md#m5)
- M4 · TP2 T=16 decode 追上 aiter_ptpc（T≤16 改用 VALU gemv，避开 BF16 MFMA 降频；计划卡标题里的假设 H1 已否定） · tp2_t16 相对 M3 1.244×（117.5 µs，为 aiter_ptpc 的 0.876），相对冻结基线累计 1.322× · [提交](https://github.com/luocheng25/pyhip/commit/5c562dc418d5cf405015786fdd938e43756a6e8c) · [详情](history.md#m4)
- M3 · T=1024 prefill 追上 aiter_ptpc（TP2/TP4） · half_m、预取 2 步与 FlyDSL 按行量化，tp2 / tp4_t1024 相对冻结基线 1.375× / 1.405×（V2） · [提交](https://github.com/luocheng25/pyhip/commit/a294918e917414d1b8b72dfce070209fb3565920) · [详情](history.md#m3)
- M2 · A8W4 prefill：fd2 解码接入 fp8 MFMA（配置表 prefill 档，当前 T≥129） · prefill 档可用（Down 一律 1x4），对 FP32 参考满足容差；基线在此冻结 · [提交](https://github.com/luocheng25/pyhip/commit/2e5a428be2a332b3e45fe3879b4391f26fee0b8c) · [详情](history.md#m2)
- M1 · W4A16 decode：mxfp4 权重格式 + decode 路径（配置表 decode 档，当前 T≤128） · `prepare_weights` 与 4 个 W4A16 kernel，kernel 用的权重与 mxfp4 反量化 + flush 逐位相同 · [提交](https://github.com/luocheng25/pyhip/commit/cc271e29d151356c40607da4db96af8ea068e459) · [详情](history.md#m1)
- M0 · 把上游 aiter gfx9_fp8 的 kernel 更新到 pyhip moe_gemm_2stage（同名文件替换 / 更新） · 上游与 legacy 两套并存、按配置分流，FP8 ptpc 走 `tuned_moe` 的 FlyDSL 候选 · [提交](https://github.com/luocheng25/pyhip/commit/e433e8d9184cdc5fd387641a00bc5686abec37cc) · [详情](history.md#m0)

## 6. 深度参考
- [history.md](history.md)：M0–M5 的完整小节（做了什么、为什么、kernel 伪代码、与计划的差异、已知问题）、M0 的 FP8 `tuned_moe` 设计（术语、调用链、tile 划分、流水线）、各里程碑的覆盖明细与未覆盖清单、决策与死路、原始数据清单。
- [m0_upstream_replacement.md](m0_upstream_replacement.md)：M0 的补充术语与 Qwen3.8 配置全表、逐文件相对上游的改动、与上游的行为差异、分流规则全表（B1–B29）、覆盖项全表（63 项）、测量来源。
- 代码里的说明：[weights.py 模块说明](weights.py#L1)（fd2 权重格式与精确性论证）、[decode.py](decode.py#L1) 与 [prefill.py](prefill.py#L1) 的文件头（相对上游 gfx9_fp8 的改动）、[moe.py](moe.py#L1) 的文件头与配置表注释（各档选择的实测依据）。
- [bench_moe_a8w4_ptpc.py 文件头](../../../../../../benchmarks/moe/bench_moe_a8w4_ptpc.py#L1)（计时协议、`--check` / `--sweep`）、[benchmarks/moe/README.md](../../../../../../benchmarks/moe/README.md)、[tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)、[moe_gemm_2stage/README.md](../moe_gemm_2stage/README.md)（替换前的 pyhip 实现，现为 legacy）。
- 原始数据在 myflow 任务目录（不在仓库）：`runs/`、`notes/`、`plan.md`，逐条见 [history.md §6](history.md#refs)。
