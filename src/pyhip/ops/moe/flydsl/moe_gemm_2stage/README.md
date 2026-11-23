# MoE A8W4（fp8 ptpc 激活 × mxfp4 权重）gfx942 新算子：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-23 02:53）· 目标：Qwen3.8-Flash-Next MoE（H2560，E513=512 routed+1 融合 shared，topk11，I=320/192，SiLU）在 MI308X 上实现 mxfp4 权重的两阶段 FlyDSL MoE（激活按 token 量化 fp8，decode 可用 bf16）：功能正确，decode 接近访存上限、prefill 接近算力上限，各档基本达到 aiter fp8 ptpc（用 aiter 调优表），达不到说明原因；第二轮（用户 11-22）：M=1–32 本实现耗时 ≤ aiter fp8 ptpc 的 0.5×（fp4 权重数据量约为 fp8 的一半），TP2 与 TP4（1.11× 相对冻结基线）· 已完成 7 个里程碑（含本提交 M7） · 最佳累计 — · 下一步：等你决定
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
本任务在 MI308X（gfx942）上做 Qwen3.8-Flash-Next MoE 的 A8W4 FlyDSL 算子；分 decode / 流式 / prefill 三条路径，各档都在 aiter 的 1.05× 以内。
- 形状：H2560，E513（512 routed + 1 个融合 shared，每个 token 的第 11 个槽位固定选它），topk11，I=320（TP2）/ 192（TP4，160 补零），SiLU。
- 输入输出：bf16 进出，输入须为有限值；权重为标准 mxfp4（fp4x2 + e8m0 scale），加载时一次性转成私有布局；激活按行动态量化成 fp8 e4m3fnuz。
- 底座：aiter gfx9_fp8 两阶段 FlyDSL MoE（供 `tuned_moe` 用）；A8W4 的 prefill kernel 由它改写，decode 与流式新写，底座文件不改。
- 性能验收：每个 case ≤ 1.05× aiter fp8 ptpc（`aiter.fused_moe` + aiter 自带 qwen38 调优表），争取接近访存 / 算力上限。
- 精度验收：对照同量化点的独立 fp32 Torch 参考，calc_diff ≤ 1e-3 且 max|out−ref| ≤ 2%·max|ref|，无 NaN / Inf。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、接入 tuned_moe / SGLang、gfx950；底座不再提供 MXFP4（A16W4）、SiTU / SiTUv2、单独 8x1 Down。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。
     完整的任务划分与逐拍流水线写在同目录 kernels.md（模板 templates/KERNELS.md），这里链接过去。 -->
权重在寄存器里按 per-group LUT 解成 fp8 走 fp8 MFMA；按 M 分 decode / 流式 / prefill 三条路径，方括号内的步骤按分档决定。
```text
加载时 prepare_a8w4_weights：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
fused_moe_a8w4（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  decode：[行量化(x)] → Gate/Up decode（WG 管一条路由）→ mid f32 → [行量化(mid)] → Down decode（WG 管一个 token，LDS 内相加）→ out
  流式：moe_sorting(Opus, BM16/32) → 行量化(x) → 流式 Gate/Up → mid（f32，或在尾部直接量化成 fp8）
    → [行量化(mid)] → 流式 Down（× 通道因子 × ms × 路由权重）→ routes bf16 [排序行, H] → route_sum → out
  prefill：moe_sorting(Opus, BM64) → 行量化(x) → Gate/Up（gather xq 行）→ mid f32 → 行量化(mid)
    → Down（同上）→ routes → sorted_sum（底座）按 loc_ids 相加 → out
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| M ≤ 32（形状受支持时） | decode：不排序、不规约 | 排序块几乎全是填充，launch 也多 |
| （decode）M·topk·I/16 ≥ 2560；M ≥ 4 | 先量化 x；先量化 mid、Down 用槽对版 | 免得每个 WG 各量化一遍整行 |
| M > 32 且每 expert 平均 ≤ 22 行（M·topk ≤ 22E） | 流式：小排序块，权重直读 VGPR | 每 expert 几到二十几行，耗时在读权重 |
| （流式）每 expert 平均 ≤ 12 / ≤ 22 行 | 排序块 16（路由 < 4E/3 时 32）/ 32 | 少填充；块 16 要 aiter 走 Opus 排序 |
| （流式，BM32）累加器 8·(I/64)·2 ≤ 96 | Gate/Up 一个 WG 管全部 I、尾部量化 | 省 f32 中间读写与一次 launch |
| 每 expert 平均 > 22 行 | prefill：BM64·BN128·TK128 两阶段 | 多行按块共用权重；TK128 占用率更高 |
| 给了 tile 参数；decode 不支持的 M ≤ 32 | prefill；TK 默认 M>2048 取 128，否则 256 | tile 参数只对 prefill 有意义 |

门槛取自实测交叉点。decode 只在 H ≤ 4096 且 Down 的 mid 行放得进 LDS 时用；槽对 Down 放不下或 w2 ≥ 4 GiB 时用单槽版。
核心 kernel 骨架（完整任务划分与逐拍流水线见 [kernels.md](kernels.md)；decode 的块对 / 槽对映射见 [a8w4_decode.py](a8w4_decode.py)）：
```text
prefill Gate/Up：grid (2I/128, 排序块)，4 wave 管 64 行 × 64 gate 与同号 64 up；x 经 LDS 双缓冲、权重进 VGPR，提前 1 个 tile 预取
  每 128-K 步建一次 LUT → 解码 → MFMA 16x16x32（解码与 LDS 读排进上一块的 MFMA 间）；epilogue → SiLU(gate)·up → f32 写 mid
prefill Down：grid (1, 排序块)，64 行 mq 驻留 VGPR；逐列块解码 → MFMA → bf16 经 LDS 写 routes；第 32 行是填充的块两者只算 32 行
decode Gate/Up：grid (I/16, M·topk)，4 wave = gate / up × K 前后半；每 lane 16 B 覆盖两个 64-K 块，部分和经 LDS 相加
decode Down：grid (H/16, M)，wave 按槽取（槽对版一条 16 B 覆盖两个槽）；各 wave 部分和经 LDS 相加 → bf16 写 out
流式 Gate/Up：grid (I/(16·G·W), 排序块)，每 wave G 个 gate 块与同号 up 块；权重与激活直读 VGPR，提前 P 个 64-K 步
  按有效 16 行子块数特化（纯填充块退出）；W>1 且 G≤4 时激活经 LDS 两槽环共享；尾部可经 LDS 交换行 amax 后量化
流式 Down：grid (H/(16·R·W), 排序块)，每 wave R 个 16 通道块 × 全部 I；本块激活驻留 VGPR（BM≥32 时 4 wave 经 LDS 共享）
  逐个通道块：权重提前 4 块 → 解码 → 两条 MFMA 链 → × 通道因子 × 行 scale → bf16 写 routes（填充行丢弃）
```
底座（供 `tuned_moe`，A8W4 只用其 `common.py`、Opus 排序与 `sorted_sum`）：aiter gfx9_fp8 逐字节移植，autotune 选路径。

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
关键取舍：fp4 在寄存器里解成 fp8 走 fp8 MFMA，激活量化与参考逐位一致；按 M 与每 expert 行数分三条路径，小 M 按访存指令条数、中 M 按读权重优化。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 经 LUT 解成 fp8 走 fp8 MFMA | 解成 bf16（A16W4） | 无 fp4 MFMA；bf16 的 MFMA 与解码 VALU 都翻倍 | delta > 11 的组按 0；流式档受解码 VALU 限制 |
| 私有布局按 MFMA K 序预排，加载时转一次 | kernel 内重排标准布局 | 每 lane 每 64-K 读 8 B 即 2 条 MFMA 操作数 | 需 prepare；布局绑定 MFMA 形状 |
| 行量化的商逐位等于 IEEE 除法 | aiter HIP 量化（乘倒数） | 判据 2%，prefill 实测最大 1.88%，翻转吃余量 | 倒数 + 两次 FMA 修正；流式尾部夹逼 + 回退 |
| M≤32 不排序，每 expert ≤22 行走小块流式 | 全走 BM64；split-K | BM64 时 M=1 的 Down 只 11 个 CTA，M=128 块内约 95% 是填充 | 三套 kernel |
| decode 每 lane 一条 16 B 加载（两个块或两个槽） | 每 lane 8 B（同 prefill） | 受访存指令条数限制，指令数减半 | 只用一半 MFMA 输出；槽对要屏蔽 A 操作数 |
| 底座取自 aiter，A8W4 prefill 沿用其结构 | pyhip 分叉版 / PR #64；量化并入 Gate/Up | 与对照同源；一行 I 跨 3–5 个 WG | 删 pyhip 独有功能；中间另行量化 |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面：每档一行“本实现 ÷ 对照”、本实现与对照的 TB/s、TFLOPS（占峰值）与一句结论；对照的来源（实现、版本或提交、调用方式、
     各档实际选中的 kernel、是否与指标同进程同协议计时）；精度参考（参考实现与容差）。冻结基线由生成块第一行给出。
     逐 case 详细数据（主规模按 2 的幂细扫，含两边的耗时、TB/s、TFLOPS）在下面的 mf 生成块里，正文写一句指向它。 -->
本实现耗时 ÷ aiter：decode 0.575–0.830、流式 0.772–1.000、prefill 0.898–1.025（34 个 case 最大 1.025，tp2_m4096）；M≤32 未达 0.5 的第二轮目标。
|档（M）|÷ aiter（TP2 / TP4）|本实现 TB/s（占峰值）|aiter TB/s（占峰值）|本实现 TFLOPS（占峰值）|aiter TFLOPS（占峰值）|结论|
|---|---|---|---|---|---|---|
|1–2 decode|0.741–0.775 / 0.782–0.830|0.59–1.03（11–19%）|0.91–1.41（17–27%）|2.2–4.0（≤0.7%）|1.8–2.9（≤0.5%）|launch 开销为主|
|4–32 预量化 decode|0.598–0.648 / 0.575–0.699|1.07–1.94（20–37%）|1.39–2.28（26–43%）|4.3–10.3（≤1.7%）|3.0–6.5（≤1.1%）|访存指令条数受限|
|48–1024 流式|0.772–1.000 / 0.783–0.990|1.22–2.00（23–38%）|2.22–3.05（42–58%）|9.0–115（≤18.9%）|7.5–115（≤18.9%）|解码 VALU 受限|
|≥2048 prefill|0.955–1.025 / 0.898–0.983|0.15–1.04（≤19.6%）|0.22–1.82（≤34.3%）|129–256（21–42%）|127–254（21–42%）|持平或略快；功耗墙|
- TB/s 按各自的最小数据量算（mxfp4 权重约为 fp8 的一半）：本实现 TB/s 都低于 aiter，decode 更快靠数据量减半，带宽利用率不如 aiter，所以没到 2×。
- 逐 M（1…32768 的 2 的幂与 48，× TP2/TP4，34 个 case）的耗时、TB/s、TFLOPS 见下方 mf 生成块（展开“逐 case”）。
- 对照：/opt/aiter @43ce3d6bf2 的 `aiter.fused_moe`（per_Token；同一份 BF16 权重按通道量化成 fp8，TP4 同样补零到 192），只用 qwen38 调优表并断言命中。
- 对照所选：FlyDSL batch1（M1–8）、split-K（tp2_m16、tp4_m32–64）、prefill（M≥2048，tp2_m4096 除外）；CK 两段（tp2_m48–64）；其余 asm 1 阶段。
- 计时：与指标同进程同协议交替；每次先写 512 MiB 冷缓存并让 GPU 空转（≥200 µs 且 ≥2× host 下发），只计 GPU 时间；30 次取中位。
- 精度参考：`pyhip.testing.moe_a8w4.reference`，独立 fp32 Torch，mxfp4 精确反量化，输入与中间按行 fp8（同形 tensor 做 IEEE 除法）。
- 判据：calc_diff ≤ 1e-3 且最大误差 ≤ 2%·max|ref|、无 NaN / Inf；bench 每组输出都校验，最大误差 decode 0.34%、流式 0.64%、prefill 1.88%。
- 底座的精度参考：`pyhip.testing.moe.torch_reference`，calc_diff ≤ 0.02；FP32→BF16 默认 RTE（同 aiter）。

<!-- mf:perf -->
基线来源（mf 记录）：11-23 02:31 冻结于提交 `9ded05a260`，bench 跑 3 次取中位数，GPU0，噪声 ±0.14%；bench 命令 `/opt/lc/work/qsa-fp8/venv-fly050/bin/python benchmarks/moe/bench_moe_a8w4.py`。对照实现与精度参考的来源见本节手写部分。

最近的性能里程碑 M4：相对当时的冻结基线 2.061×（mf 复核 V4，旧 bench）。之后 bench 改了（重新冻结于 `9ded05a260`，含 M4 的代码）：下表是当前代码在新 bench 上的逐 case 数据。

<details><summary>逐 case（34 个，mf 冻结 3 次中位，当前代码）</summary>

| case | 当前 µs | 对照 µs | 当前/对照 | 当前 TB/s | 对照 TB/s | 当前 TFLOPS | 对照 TFLOPS |
|---|---|---|---|---|---|---|---|
| tp2_m1 | 18.2 | 23.50 | 0.775 | 0.81 (15.3%) | 1.16 (21.9%) | 2.99 (0.5%) | 2.30 (0.4%) |
| tp2_m2 | 27.32 | 36.72 | 0.741 | 1.03 (19.4%) | 1.41 (26.6%) | 3.98 (0.7%) | 2.92 (0.5%) |
| tp2_m4 | 40.3 | 62.20 | 0.648 | 1.33 (25.1%) | 1.60 (30.2%) | 5.38 (0.9%) | 3.49 (0.6%) |
| tp2_m8 | 65.32 | 109.52 | 0.598 | 1.59 (30.0%) | 1.75 (33.1%) | 6.61 (1.1%) | 3.92 (0.6%) |
| tp2_m16 | 100.06 | 166.84 | 0.600 | 1.85 (34.9%) | 2.05 (38.7%) | 8.62 (1.4%) | 5.19 (0.9%) |
| tp2_m32 | 168.26 | 264.52 | 0.636 | 1.94 (36.6%) | 2.28 (43.0%) | 10.3 (1.7%) | 6.52 (1.1%) |
| tp2_m48 | 278.92 | 301.48 | 0.925 | 1.54 (29.0%) | 2.63 (49.7%) | 9.30 (1.5%) | 8.60 (1.4%) |
| tp2_m64 | 257.52 | 327.80 | 0.787 | 1.85 (34.9%) | 2.68 (50.6%) | 13.4 (2.2%) | 10.6 (1.7%) |
| tp2_m128 | 315.34 | 383.68 | 0.826 | 2.00 (37.8%) | 3.05 (57.5%) | 21.9 (3.6%) | 18.0 (3.0%) |
| tp2_m256 | 343.96 | 448.56 | 0.772 | 1.98 (37.4%) | 2.81 (53.0%) | 40.2 (6.6%) | 30.9 (5.1%) |
| tp2_m512 | 391.06 | 462.42 | 0.851 | 1.76 (33.2%) | 2.75 (51.9%) | 70.8 (11.7%) | 59.9 (9.9%) |
| tp2_m1024 | 482.56 | 483.26 | 1.000 | 1.44 (27.2%) | 2.64 (49.9%) | 115 (18.9%) | 115 (18.9%) |
| tp2_m2048 | 676.34 | 708.44 | 0.955 | 1.04 (19.6%) | 1.82 (34.3%) | 164 (27.0%) | 156 (25.8%) |
| tp2_m4096 | 1159.39 | 1132.01 | 1.025 | 0.63 (11.9%) | 1.16 (21.9%) | 191 (31.5%) | 196 (32.3%) |
| tp2_m8192 | 1921.23 | 1936.69 | 0.991 | 0.40 (7.5%) | 0.70 (13.2%) | 231 (38.0%) | 229 (37.7%) |
| tp2_m16384 | 3589.32 | 3613.24 | 0.994 | 0.24 (4.5%) | 0.40 (7.5%) | 247 (40.7%) | 245 (40.5%) |
| tp2_m32768 | 6930.6 | 6967.72 | 0.995 | 0.15 (2.8%) | 0.23 (4.3%) | 256 (42.2%) | 254 (42.0%) |
| tp4_m1 | 14.88 | 17.92 | 0.830 | 0.59 (11.2%) | 0.91 (17.2%) | 2.19 (0.4%) | 1.81 (0.3%) |
| tp4_m2 | 21.24 | 27.08 | 0.782 | 0.79 (14.9%) | 1.15 (21.7%) | 3.08 (0.5%) | 2.40 (0.4%) |
| tp4_m4 | 29.9 | 42.76 | 0.699 | 1.07 (20.2%) | 1.39 (26.2%) | 4.32 (0.7%) | 3.02 (0.5%) |
| tp4_m8 | 45.54 | 72.40 | 0.630 | 1.32 (24.9%) | 1.55 (29.2%) | 5.70 (0.9%) | 3.59 (0.6%) |
| tp4_m16 | 74.66 | 129.74 | 0.575 | 1.52 (28.6%) | 1.62 (30.6%) | 6.98 (1.2%) | 4.00 (0.7%) |
| tp4_m32 | 115.72 | 178.12 | 0.650 | 1.70 (32.1%) | 2.05 (38.7%) | 8.99 (1.5%) | 5.81 (1.0%) |
| tp4_m48 | 173.6 | 206.76 | 0.838 | 1.45 (27.3%) | 2.25 (42.5%) | 8.98 (1.5%) | 7.52 (1.2%) |
| tp4_m64 | 187.46 | 232.34 | 0.803 | 1.57 (29.6%) | 2.35 (44.4%) | 11.1 (1.8%) | 8.92 (1.5%) |
| tp4_m128 | 223.76 | 275.40 | 0.812 | 1.68 (31.7%) | 2.53 (47.8%) | 18.6 (3.1%) | 15.1 (2.5%) |
| tp4_m256 | 248.76 | 317.18 | 0.783 | 1.65 (31.1%) | 2.39 (45.1%) | 33.4 (5.5%) | 26.2 (4.3%) |
| tp4_m512 | 295.42 | 336.70 | 0.879 | 1.41 (26.6%) | 2.28 (43.0%) | 56.2 (9.3%) | 49.3 (8.1%) |
| tp4_m1024 | 345.34 | 348.68 | 0.990 | 1.22 (23.0%) | 2.22 (41.9%) | 96.2 (15.9%) | 95.3 (15.7%) |
| tp4_m2048 | 514.04 | 522.80 | 0.983 | 0.84 (15.8%) | 1.50 (28.3%) | 129 (21.3%) | 127 (21.0%) |
| tp4_m4096 | 802.36 | 893.78 | 0.898 | 0.56 (10.6%) | 0.90 (17.0%) | 166 (27.3%) | 149 (24.5%) |
| tp4_m8192 | 1349.09 | 1444.41 | 0.934 | 0.37 (6.9%) | 0.59 (11.1%) | 197 (32.5%) | 184 (30.4%) |
| tp4_m16384 | 2480.89 | 2636.75 | 0.941 | 0.23 (4.4%) | 0.35 (6.6%) | 214 (35.3%) | 202 (33.3%) |
| tp4_m32768 | 4707.89 | 5084.67 | 0.925 | 0.16 (3.0%) | 0.22 (4.1%) | 226 (37.3%) | 209 (34.5%) |

TB/s、TFLOPS = bench 给出的每次调用数据量 / 运算量 ÷ 中位耗时，括号内占峰值（带宽 5.3 TB/s、算力 606 TFLOPS）；对照 = aiter_fp8_ptpc（MF_REF，同一 bench 进程、同一协议）

</details>
<!-- /mf:perf -->

## 5. 测试覆盖
A8W4 与底座各有基本用例；34 个 bench case（M=1…32768 的 2 的幂与 48，× TP2 / TP4）覆盖 A8W4 三条路径的各分派；交界、形状回退、变体、逐位、报错与 graph 在其他覆盖。

**基本功能测试**（每次改动先跑：`commands.correctness` = `test_moe_a8w4.py` + `test_fly_2stage_basic.py`，10 例，约 35 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| A8W4 TP2（I320，E33）：M=3 decode、M=64 流式（BM32，Gate/Up 尾部量化）、M=1031 prefill | `test_tp2` ×3 | 通过 |
| A8W4 TP4：I160 补零到 192，M=257 prefill | `test_tp4_padded` | 通过 |
| 底座 decode：batch1（M=1）、split-K（M=16），FP8 PTPC，H2560 / I320 | `test_batch1`、`test_splitk` | 通过 |
| 底座 prefill：FP8 × 三种 Down（compact 用 I192）、BF16 × default | `test_prefill` ×3、`test_prefill_bf16` | 通过 |

**性能测试**（`commands.bench` = `bench_moe_a8w4.py`，34 个 case：E513，M=1、2、4…32768 与 48 × TP2 / TP4；每组输出按判据校验；底座逐字节移植，无 case）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| decode 核内量化（2 个 kernel，M1–2） | `tp*_m1`、`tp*_m2` | 0.741–0.830×，0.59–1.03 TB/s |
| decode 先量化 mid，较大 M 也先量化 x | `tp*_m4`…`tp*_m32` | 0.575–0.699×，1.07–1.94 TB/s |
| 流式 BM16（M64–512）、BM32+尾部量化（其余） | `tp*_m48`…`tp*_m1024` | 0.772–1.000×，1.22–2.00 TB/s |
| prefill BM64·TK128（M≥2048） | `tp*_m2048`…`tp*_m32768` | 0.898–1.025×，129–256 TFLOPS |

**其他覆盖测试**（只测前两类覆盖不了的；`commands.verify_extra`，77 例，约 140 秒；用例名省略 `test_`）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| decode 分派交界 M=4、TP2 11/12、TP4 19/20、32/33 | 基本仅 M=3；bench 仅 2 的幂与 48，不断言路径 | `decode_dispatch_boundaries` ×7 | 通过 |
| 流式与 prefill 的交界 M=66 / 67（E33，每 expert 22 / 22.3 行） | 基本与 bench 的 M 都不在交界上 | `stream_dispatch_boundary` ×2 | 通过 |
| prefill 半块交界：尾块恰为 BM/2、BM/2+1 行（BM64、BM128） | 随机路由碰不到恰好的行数 | `prefill_half_block_boundary` ×3 | 通过 |
| prefill 的 TK：默认 128；给了 tile 参数时 M=2048 取 256、M=2049 取 128 | 前两类不给 tile 参数 | 同上 | 通过 |
| 形状回退：H 4096 / 4608、槽对 Down I 1024 / 1088、I1024 流式 Down 共享与否（BM32 / 64） | 前两类只有目标形状 | `shape_fallbacks` ×6 | 通过 |
| decode topk=8（槽对整对无效、WG 数恰为 2560）与 topk=17（Down 4 wave 轮流取槽） | 前两类都是 topk=11 | `decode_topk_variants` ×2 | 通过 |
| decode 同 token 重复 expert、int64 非连续 topk_ids、非连续 topk_weight | 生成的路由不重复、张量连续 | `decode_duplicate_and_int64_ids` | 通过 |
| decode 在 w2 ≥ 4 GiB 时改用核内量化 Down（stride 0 视图冒充 4 GiB） | 前两类的 w2 都远小于 4 GiB | `decode_down_fallback_w2_4gib` | 通过 |
| CUDA graph 捕获与重放：流式 M=64、prefill、decode M=1 / 5 / 16 | 都是 eager 调用 | `graph_replay` ×2、`decode_graph_replay` ×3 | 通过 |
| 行量化逐位：中点、-0、次正规、越界行、乘倒数会舍错的商、scale<2^-64 | fp8 翻转看不出 | `row_quant_bitexact` ×2、`decode_row_quant_bitexact` ×6 | 通过 |
| 流式输入量化与 Gate/Up 尾部量化逐位（含回退分支、次正规 scale） | 同上 | `stream_row_quant_bitexact` ×2、`stream_fused_quant_bitexact` ×2 | 通过 |
| A8W4 prepare 布局逐字节：delta 0–15、-0、补零、通道因子 | 端到端误差看不出布局与文档不符 | `prepare_layout_bitexact` | 通过 |
| A8W4 delta 11（精确）/ 12（清零）边界，全部 16 个 E2M1 码 | 随机权重同一行 E8M0 只差 0–3 | `delta_flush_boundary` | 通过 |
| E513 流式 grid 收缩交界 M=46 / 47 | 基本用 E33；bench 的 M=48 已不收缩 | `grid_shrink_boundary_e513` | 通过 |
| 给了 tile 参数走 prefill（M=300）：BM128 + BN256，TK 取 256 | 前两类只走默认 tile | `tile_variants` | 通过 |
| A8W4 报错：dtype、形状、E8M0 NaN、H / I 不对齐、BM·I_pad 非 4096 倍数、hidden 不符 | 前两类只给合法输入 | `rejects_bad_inputs` | 通过 |
| 底座 per-tensor / BF16 decode scale、SwiGLU | 基本只测 PTPC、SiLU | `splitk_variants`、`prefill_variants`、`swiglu` ×2 | 通过 |
| 底座形状交界：8x1 的 K 调度族、N512、K640 + pad 0、M·topk ≤ E | 基本只用 Qwen 形状与大 M | `prefill_variants`、`splitk_variants` | 通过 |
| 底座 batch1 清零写满与 graph replay（排序、loc_ids、任务表） | 基本只跑 eager、不查输出两侧 | `batch1_clear_graph` ×4、`graph_replay` ×2 | 通过 |
| 底座报错：K 不是 256 的倍数、Gate/Up 与 Down 的 M 不同 | 主路径只走合法配置 | `gateup_rejects_incomplete_k` ×4、`split_m_tile_rejected` | 通过 |
| 底座逐位：FP32→BF16 舍入、Down 打包前只舍入一次 | calc_diff 看不出舍入方式 | `bf16_rounding_bits`、`down_rounds_before_packing` ×3 | 通过 |
| 底座候选门槛：不完整 N tile、默认 Down 超 64 KiB；恰 64 KiB 可 launch | 前两类用固定配置，不枚举候选 | `candidate_gates` ×5、`lds_64k_boundary` | 通过 |

未覆盖 / 跳过：
- I_pad > 1024：prefill Down 的 64·I_pad 超过 64 KiB，走 prefill 时（每 expert > 22 行、给 tile 参数、decode 不支持）构建即报错，入口不检查。
- 同 token 重复 expert id：排序路径（流式、prefill）只保留最后一个槽（aiter 排序丢弃其余槽与权重），与 decode、参考不同。
- 流式排序块 64 只在调大 `STREAM_MAX_ROWS` 时用：只有形状回退与尾部量化逐位各一例，没有性能 case。
- 排序块 16 / 32 的 Opus 交界只靠与 aiter 同一公式，未测。
- w2 真超过 4 GiB：只用 stride 0 视图验证分派与单槽 Down，没在大权重上跑过 64 位 expert 寻址；这个回退也没有性能 case。
- 含 NaN 的输入：约定输入为有限值（decode 与 prefill 行量化对 NaN 的处理不同），不测。
- M·topk·I_pad·4 ≥ 2^31（I320 约 15 万 token）：输入过大，入口也不检查。
- gfx950 与同进程多卡：只在单卡 gfx942 上跑。
- 底座 per-tensor 的 Gate/Up 输出预清零：只在排序丢掉路由时起作用，正常 top-k 路由触发不到（做法同 aiter）。
- tuned_moe 的调优缓存、剪枝与非 FlyDSL 路径在 `test_tuned_moe.py` / `test_moe.py`（两个旧用例起点即失败），不在上面两条命令里。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
各档都在 aiter 的 1.05× 以内（M≤32 未达 0.5× 的第二轮目标）；主要风险是 prefill 受功耗墙限制、流式离访存上限远与精度余量小。
- prefill 持续负载与 aiter 一样顶在 650 W 功耗墙（sclk 约 1.55–1.6 GHz）→ TP2 M16384 只到 41% fp8 峰值，减 VALU 不再减耗时 → 不再优化。
- Down 的 LDS bank conflict 占 LDS 活跃周期 59%（未处理）；流式 HBM 只到 23–38%，瓶颈是 LUT 解码 VALU → 再快要减解码指令。
- 2% 最大误差判据余量小：prefill 实测最大 1.88%（tp4_m32768）→ 改量化算术会超判据 → 保持逐位一致。
- 约束 H%512、I%32、M·topk·I_pad·4 < 2^31；I_pad > 1024 时 prefill 不支持；重复 expert id 各路径不一致；门槛按 E513 实测 → 接新模型先核对。
- host 下发 decode 约 0.1–0.17 ms、流式与 prefill 约 0.29–0.36 ms/次，编译缓存不按设备 → 实用靠 CUDA graph（三条路径已验证）、每卡一进程。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../moe_a8w4.py) 的 `prepare_a8w4_weights`（加载时一次）与 `fused_moe_a8w4`（接口与约束见 docstring，分派门槛在文件头与 `_stream_*`）。
- kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)；私有布局与 LUT 解码见 [a8w4_common.py](a8w4_common.py)，decode 的块对 / 槽对映射见 [a8w4_decode.py](a8w4_decode.py) 的 docstring。
- 参考与 bench：[pyhip.testing.moe_a8w4](../../../../testing/moe_a8w4.py)、[bench_moe_a8w4.py](../../../../../../benchmarks/moe/bench_moe_a8w4.py)；用例说明见 [tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)。
- 底座：[tuned_moe.py](../../tuned_moe.py) 的 `_run_fly` / `_configs`；kernel 同 [aiter gfx9_fp8 @ca6bc0f851](https://github.com/ROCm/aiter/tree/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8)，调用链见 [fused_moe_gfx942.py](https://github.com/ROCm/aiter/blob/ca6bc0f8518796024a0e7856211394a1a8c24820/aiter/ops/flydsl/fused_moe_gfx942.py)。
- 里程碑（每个提交写了改了什么、为什么）：

<!-- mf:roadmap -->
- M0 ✅ 已推送 · 功能 · 底座：aiter gfx9_fp8 替换 pyhip moe_gemm_2st… · [提交](https://github.com/luocheng25/pyhip/commit/aca23ec23286bb8136d4c1644491dc7d132489a0)
- M1 ✅ 已推送 · 功能 · A8W4 功能版：底座 prefill 两段 kernel 换 mxfp4 权重 · [提交](https://github.com/luocheng25/pyhip/commit/78962639bc4e8cdde5b337b6e05ee7ecef02dc99)
- M2 ✅ 已推送 · 性能 · 小 M 直通路径（M≤32）：免排序小 tile kernel，激活可用 bf… · 1.576× 对基线（预期 1.5×） · [提交](https://github.com/luocheng25/pyhip/commit/4cff0b0fbddf7afbc808c54b911d75dbfc7cf49b)
- M3 ✅ 已推送 · 性能 · 访存档路径（M 128–2048）：sorted split-K / 小 BM… · 1.211× 对 M2（预期 1.4×） · [提交](https://github.com/luocheng25/pyhip/commit/5f081cca12aedca048ae3e502cf5f159203cf6c9)
- M4 ✅ 已推送 · 性能 · prefill 算力档（M≥4096）：恢复指令排布、调 tile、中间量化并… · 1.080× 对 M3（预期 1.09×） · [提交](https://github.com/luocheng25/pyhip/commit/9dbf952e4d27b52477d2cc89e10948035e1a806a)
- M5 ✖ 放弃 · 性能 · decode 档（M≤32）再提速：单次 launch 的小 M 融合 dec… · 预期 1.11×
- M6 ✅ 已推送 · 维护 · 性能数据细扫：bench 改为 M=1…32768（2 的幂）× TP2/TP… · [提交](https://github.com/luocheng25/pyhip/commit/21a02e7e9969a26a4fd8d1d20c50068f8bd290ca)
- M7 ✅ 本提交 · 维护 · 补充 M=48 性能 case：覆盖 E513 下 M=33–62 的排序块 …

下一步（本提交的承诺）：等你决定
<!-- /mf:roadmap -->
