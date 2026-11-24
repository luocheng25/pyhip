# MoE A8W4（fp8 ptpc 激活 × mxfp4 权重）gfx942 新算子：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-24 14:27）· 目标：Qwen3.8-Flash-Next MoE（H2560，E513=512 routed+1 融合 shared，topk11，I=320/192，SiLU）在 MI308X 上实现 mxfp4 权重的两阶段 FlyDSL MoE（激活按 token 量化 fp8）：功能正确，decode 接近访存上限、prefill 接近算力上限，各档基本达到 aiter fp8 ptpc。第三轮（用户 11-23）：Qwen TP2/TP4 与 MiMo-V2.6-Flash TP8/TP4（H4096 E256 topk8，I=256/512）每个 M（1…32768 的 2 的幂及 48）的耗时 ≤ min(R1 M8, tingqli refactor/a8w4 最快配置) × 1.02，以 cmp3 同卡同协议对比验收；误差门限 err≤4.5%（与 tingqli 同级）。DeepSeek 支持（用户 11-24）：DeepSeek-V4.1-Flash-1017 TP4（H5120 E384 topk6，I=576，swiglu_limit=10）加 SwiGLU 截断与 hidden 5120 decode，逐点 ≤ tingqli 最快配置（I 补零到 768）× 1.02；算子级精度评测（真实 FP4 权重）中本算子误差 ≤ 每 32 元素 UE8M0 FP8 参考的 1.5 倍（1.003× 相对冻结基线）· 已完成 16 个里程碑（含本提交 M15） · 最佳累计 — · 下一步：M16 hidden 5120 的 decode：DeepSeek TP4 M≤32 不再回退 prefill（Gate/Up decode 每线程多段 16 B，Down decode 核对 I=576）；之后：M17 DeepSeek TP4 调参：流式 / prefill 的分界、排序块与 tile 在 H5120 上重标定（M48–32768）（暂定）
<!-- /mf:summary -->
<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->
## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
本任务在 MI308X（gfx942）上做 Qwen3.8-Flash-Next、MiMo-V2.6-Flash 与 DeepSeek-V4.1-Flash MoE 的 A8W4 FlyDSL 算子，按 M 分三条路径。
- Qwen3.8：H2560，E513（512 routed + 1 个融合 shared，每个 token 的第 11 个槽位固定选它），topk11，I=320（TP2）/ 192（TP4，160 补零）。
- MiMo-V2.6-Flash：H4096，E256（均匀路由，无 shared），topk8，I=256（TP8）/ 512（TP4）。
- DeepSeek-V4.1-Flash：只做 TP4，H5120，E384（shared 不融合），topk6，I=576；SiLU 前截断 gate ≤ 10、|up| ≤ 10（`swiglu_limit`）。
- 输入输出：bf16 进出，输入与权重须为有限值；权重为标准 mxfp4（fp4x2 + e8m0 scale），加载时一次性转成私有布局；激活按行动态量化成 fp8 e4m3fnuz。
- 底座：aiter gfx9_fp8 两阶段 FlyDSL MoE（逐字节移植，供 `tuned_moe` 用）；A8W4 prefill 由它改写，decode 与流式新写。
- 性能验收：Qwen 每个 case ≤ 1.05× aiter fp8 ptpc；每个 M ≤ min(R1 第一轮实现, tingqli refactor/a8w4 最快配置) × 1.02（DeepSeek 只比 tingqli）。
- 精度验收：对照同量化点的独立 fp32 参考，calc_diff ≤ 1e-3、max 误差 ≤ 4.5%·max|ref|、无 NaN / Inf；DeepSeek 另要求真实权重下误差 ≤ 模型参考的 1.5 倍。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、接入 tuned_moe / SGLang、gfx950；底座不再提供 MXFP4（A16W4）、SiTU / SiTUv2、单独 8x1 Down。
## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。
     完整的任务划分与逐拍流水线写在同目录 kernels.md（模板 templates/KERNELS.md），这里链接过去。 -->
权重在寄存器里按 per-group LUT 解成 fp8 走 fp8 MFMA；按 M 分 decode / 流式 / prefill 三条路径，方括号内的步骤按分档决定，门槛取自实测交叉点。
```text
加载时 prepare_a8w4_weights：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
fused_moe_a8w4（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  decode：[行量化(x)] → Gate/Up decode（WG 管一条路由）→ mid f32 → [行量化(mid)] → Down decode（WG 管一个 token，LDS 内相加）→ out
  decode 去重：行量化(x) + 计划 + 清零 out → Gate/Up 与 Down（组首 WG 管同 expert 的 2–3 条路由）→ bf16x2 原子加进 out
  流式：prep（排序 BM16/32/64 + 行量化(x) [+ 清零 out]）→ Gate/Up → [行量化(mid)] → Down → 原子加进 out，或 routes → route_sum
  prefill：[流式 prep（H≥2560 且 M·topk≤24576）| Opus 排序 BM64 → 行量化(x，loc_ids 置 −1)] → Gate/Up（gather xq 行）
    → mid f32 → 行量化(mid) → Down（× 通道因子 × ms × 路由权重，写 loc_ids）→ routes → sorted_sum（底座）按 loc_ids 相加 → out
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| M ≤ 32，H ≤ 4096 且 mid 行放得进 LDS | decode：不排序、不规约 | 排序块几乎全是填充，launch 也多 |
| （decode）M·topk·I/16 ≥ 2560；M ≥ 4 | 先量化 x；先量化 mid、Down 用槽对版（放得下且 w2 < 4 GiB） | 免得每个 WG 各量化一遍整行 |
| （decode）H4096、I%128=0、M≥2（路由 ≤1.5E 时到 M64） | 按 expert 去重；M·I<3072 时两个 kernel | E256 M16/32 有 20%/36% 路由重读 expert |
| M>32 且每 expert 平均 ≤22 行（H≥4096 为 48，H2560 BM64 尾部量化为 56） | 流式：排序块 16（≤12 行且路由 ≥4E/3）、32（≤28 行）、64 | 读权重为主；上限取实测交叉 |
| （流式）H < 4096 且累加器 8·(I/64)·(BM/16) ≤ 96 | Gate/Up 一个 WG 管全部 I、尾部量化 | 省一次 launch；H4096 时轻 wave 更快 |
| （流式）I < 512 | Down 原子加进 prep 清零的 out | 免 routes 与 route_sum；I512 寄存器多反慢 |
| （流式）BM64、非原子且 H ≥ 4096 | Down 换 prefill Down + route_sum | MiMo TP4 M1024 Down 303→289 µs |
| 超过流式上限 | prefill：BM64·BN128·TK128；块少时 Down 把 H 切给 2 / 4 个 WG | 多行按块共用权重；切分补满末轮 WG |
| （prefill）H≥2560 且 M·topk≤24576 | 排序与 x 量化改用流式 prep（一次 launch） | TP2 M2048 快 2.1%；路由多则 prep 慢 |
| 给了 tile 参数；decode 不支持的 M ≤ 32 | prefill；TK 默认 M>2048 取 128，否则 256 | tile 参数只对 prefill 有意义 |
- DeepSeek TP4（H5120、I576）：M≤32 decode 放不下 → prefill（TK256），33–3072 流式（BM16–64，Down 不用原子），≥3073 prefill（TK128）。
- 核心 kernel 骨架（[kernels.md](kernels.md) 每节先给 Gluon 风格的分阶段写法，写明放置、预取与 barrier，再给任务划分与逐拍流水线；decode 的块对 / 槽对映射见 [a8w4_decode.py](a8w4_decode.py)）：
```text
prefill Gate/Up：grid (2I/128, 排序块)，4 wave 管 64 行 × 64 gate 与同号 64 up；权重进 VGPR 提前 2 tile，x 经 LDS 双缓冲
  每 lane 16 B 读相邻两行 → 每 128-K 步建 LUT → 解码 → MFMA；epilogue 跨 lane 换回 → SiLU(gate)·up → f32 写 mid
prefill Down：grid (切分 1/2/4, 排序块)，64 行 mq 驻留 VGPR；逐列块解码 → MFMA → bf16 经 LDS 写 routes，最后写 loc_ids
decode Gate/Up：grid (I/16, M·topk)，4 wave = gate / up × K 前后半；每 lane 16 B 覆盖两个 64-K 块，部分和经 LDS 相加
decode Down：grid (H/16, M)，wave 按槽取（槽对版一条 16 B 覆盖两个槽）；各 wave 部分和经 LDS 相加 → bf16 写 out
去重版：Gate/Up 同 grid、只有组首 WG 计算（组员 x 行进 LDS，A 的行 m、8+m 取组员 m）；Down grid (H/128, M·topk)，结果原子加
流式 prep：前 ⌈M·topk/256⌉ 个 WG 各统计全部路由（LDS 原子）后放置自己的 256 条，其余 WG 量化 x 行（并清零 out）
流式 Gate/Up：grid (I/(16·G·W), 排序块)，每 wave G 个 gate 块与同号 up 块；权重（non-temporal）与激活直读 VGPR，提前 P 个 64-K 步
  按有效 16 行子块数特化（纯填充块退出）；W>1 时激活经 LDS 环共享（轻 wave 每 2 步一次 barrier）；尾部量化时 4 wave 管全部 I
流式 Down：grid (H/(16·R·W), 排序块)，每 wave R 个 16 通道块 × 全部 I；本块激活驻留 VGPR（BM≥32 或 H4096·I≤256 时 4 wave 经 LDS 共享）
  逐个通道块：权重提前 4 块 → 解码 → 两条 MFMA 链 → × 通道因子 × 行 scale → bf16 写 routes，或两块凑一次原子加进 out
```
截断（`swiglu_limit` L>0）：四类 Gate/Up 收尾在乘完通道因子与 xs 后、SiLU 前令 gate ≤ L、|up| ≤ L；L 是编译期常量（kernel 名加 `_swiglu`），L=0 时 IR 不变。
## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
关键取舍：fp4 在寄存器里解成 fp8 走 fp8 MFMA，激活量化与参考逐位一致；按 M 与每 expert 行数分三条路径，小 M 按访存指令条数、中 M 按读权重优化。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 经 LUT 解成 fp8 走 fp8 MFMA | 解成 bf16（A16W4） | 无 fp4 MFMA；bf16 的 MFMA 与解码 VALU 都翻倍 | delta > 11 的组按 0；流式档受解码 VALU 限制 |
| 私有布局按 MFMA K 序预排，加载时转一次 | kernel 内重排标准布局 | 每 lane 每 64-K 读 8 B 即 2 条 MFMA 操作数 | 需 prepare；布局绑定 MFMA 形状 |
| 行量化的商逐位等于 IEEE 除法 | aiter HIP 量化（乘倒数） | 量化不翻转，prefill 最大 2.26%（判据 4.5%） | 倒数 + 两次 FMA 修正；流式尾部夹逼 + 回退 |
| M≤32 不排序，每 expert ≤22–56 行走小块流式 | 全走 BM64；split-K | BM64 时 M=1 的 Down 只 11 个 CTA，M=128 块内约 95% 是填充 | 三套 kernel |
| decode 每 lane 16 B 加载；H4096 按 expert 去重 | 每 lane 8 B；逐路由读权重 | 访存指令减半；E256 M32 有 36% 路由重读 | 只用半数 MFMA 输出；原子加不逐位确定 |
| 底座取自 aiter，A8W4 prefill 沿用其结构 | pyhip 分叉版 / PR #64；量化并入 Gate/Up | 与对照同源；一行 I 跨 3–5 个 WG | 中间另行量化；辅助 kernel 多 |
## 4. 性能与精度
<!-- ≤15 行，写在生成块前面：每档一行“本实现 ÷ 对照”、本实现与对照的 TB/s、TFLOPS（占峰值）与一句结论；对照的来源（实现、版本或提交、调用方式、
     各档实际选中的 kernel、是否与指标同进程同协议计时）；精度参考（参考实现与容差）。冻结基线由生成块第一行给出。
     逐 case 详细数据（主规模按 2 的幂细扫，含两边的耗时、TB/s、TFLOPS）在下面的 mf 生成块里，正文写一句指向它。 -->
Qwen 耗时 ÷ aiter：decode 0.569–0.825、流式 0.573–0.908、prefill 0.811–0.905；逐点目标未达 7 / 83 点（MiMo TP8 6、DeepSeek M4）。
|档（M）|÷ aiter（TP2 / TP4）|本实现 TB/s（占峰值）|aiter TB/s（占峰值）|本实现 TFLOPS（占峰值）|aiter TFLOPS（占峰值）|结论|
|---|---|---|---|---|---|---|
|1–2 decode|0.749–0.779 / 0.785–0.825|0.60–1.03（11–20%）|0.92–1.43（17–27%）|2.2–4.0（≤0.7%）|1.8–3.0（≤0.5%）|launch 开销为主|
|4–32 预量化 decode|0.607–0.660 / 0.569–0.700|1.07–1.94（20–37%）|1.39–2.32（26–44%）|4.3–10.3（≤1.7%）|3.0–6.6（≤1.1%）|访存指令条数受限|
|48–1024 流式（TP4 到 2048）|0.597–0.908/0.573–0.849|0.98–2.43（18–46%）|1.50–2.93（28–55%）|10–150（≤25%）|7.3–127（≤21%）|VALU 受限|
|≥2048 prefill（TP4 ≥4096）|0.864–0.905/0.811–0.851|0.16–1.15（≤22%）|0.22–1.82（≤34%）|181–282（30–47%）|148–255（25–42%）|功耗墙|
- MiMo 只测本实现：decode（M≤48）0.69–2.37 TB/s（13–45%），流式（M64–1024）1.30–2.55 TB/s（25–48%），prefill 190–330 TFLOPS（31–55%）。
- DeepSeek 本实现：M≤32 回退 prefill 0.41–1.67 TB/s（8–32%），流式（M48–2048）1.29–2.72 TB/s（24–51%），prefill 203–263 TFLOPS（33–43%）。
- DeepSeek ÷ tingqli：M≤32 0.943–1.156（只有 M4 超 1.02，是权重读取下限的 3.2–13 倍），流式 0.609–0.878，prefill 0.806–0.853，几何平均 0.820。
- TB/s 按各自的最小数据量算（mxfp4 权重约为 fp8 的一半）：本实现都低于 aiter，decode 更快靠数据量减半；85 个 case 的逐 M 数据见下方生成块。
- aiter 对照只有 Qwen（其余模型无调优表）：aiter @43ce3d6bf2 的 `aiter.fused_moe` per_Token，同一份 BF16 权重、TP4 补零到 192，断言命中 qwen38 表。
- 对照所选：FlyDSL batch1（M1–8）、split-K（tp2_m16、tp4_m32–64）、prefill（M≥2048，tp2_m4096 除外）；CK 两段（tp2_m48–64）；其余 asm 1 阶段。
- 计时：与指标同进程同协议交替；每次先写 512 MiB 冷缓存并让 GPU 空转（≥200 µs 且 ≥2× host 下发），只计 GPU 时间；30 次取中位。
- R1（tag moe-ptpc-a8w4-M8 @2683967c3f）与 tingqli（@5c70a4d，逐 M 取所有配置里最快的）用仓库外 cmp3 脚本：同卡同输入同协议，各自进程轮换 4 轮（DeepSeek 3 轮）。
- DeepSeek 对照两边都不截断（截断只多约 2 条 VALU）；tingqli 要 I 为 256 的倍数，补零到 768（等价），其 tuner 为该形状选 M≤1024 tp2k bm32、M≥2048 tp1k ks3。
- 精度参考：A8W4 为 `pyhip.testing.moe_a8w4.reference`（独立 fp32，mxfp4 精确反量化，输入与中间按行 fp8），底座为 `torch_reference`。
- 判据：A8W4 calc_diff ≤ 1e-3、最大误差 ≤ 4.5%·max|ref|（底座 calc_diff ≤ 0.02）；最大误差 decode 1.33%、流式 1.19%、prefill 2.26%。
- DeepSeek 精度用真实 V4.1 权重评测（layer 2 / 37 路由专家的 TP4 切片、真实路由，激活为随机方向 × ffn_norm 增益加压力 token；未做端到端，脚本与数据在仓库外的任务目录）：
  按 token FP8 的每 token 相对 L2 误差是模型参考（每 32 元素 UE8M0 FP8）的 0.48–0.98 倍（UE8M0 向上取整丢精度），满足 ≤1.5 倍判据，不改激活 scale 粒度。
<!-- mf:perf -->
基线来源（mf 记录）：11-24 13:48 冻结于提交 `b3848fd6cb`，bench 跑 3 次取中位数，GPU0，噪声 ±0.14%；bench 命令 `/opt/lc/work/qsa-fp8/venv-fly050/bin/python tests/ops/moe/test_moe_a8w4.py`。对照实现与精度参考的来源见本节手写部分。

最近的性能里程碑 M13：相对当时的冻结基线 1.123×（mf 复核 V12，旧 bench）。之后 bench 改了（重新冻结于 `b3848fd6cb`，含 M13 的代码）：下表是当前代码在新 bench 上的逐 case 数据。

<details><summary>逐 case（85 个，mf 冻结 3 次中位，当前代码）</summary>

| case | 当前 µs | 对照 µs | 当前/对照 | 当前 TB/s | 对照 TB/s | 当前 TFLOPS | 对照 TFLOPS |
|---|---|---|---|---|---|---|---|
| ds_tp4_m1 | 68.86 | — | — | 0.41 (7.8%) | — | 1.52 (0.3%) | — |
| ds_tp4_m2 | 78.16 | — | — | 0.73 (13.8%) | — | 2.71 (0.4%) | — |
| ds_tp4_m4 | 129.06 | — | — | 0.89 (16.7%) | — | 3.30 (0.5%) | — |
| ds_tp4_m8 | 179.96 | — | — | 1.18 (22.2%) | — | 4.71 (0.8%) | — |
| ds_tp4_m16 | 275.4 | — | — | 1.49 (28.1%) | — | 6.18 (1.0%) | — |
| ds_tp4_m32 | 438.3 | — | — | 1.67 (31.5%) | — | 7.78 (1.3%) | — |
| ds_tp4_m48 | 420 | — | — | 2.30 (43.4%) | — | 12.1 (2.0%) | — |
| ds_tp4_m64 | 498.3 | — | — | 2.35 (44.3%) | — | 13.6 (2.2%) | — |
| ds_tp4_m128 | 580.76 | — | — | 2.72 (51.3%) | — | 23.4 (3.9%) | — |
| ds_tp4_m256 | 661.1 | — | — | 2.72 (51.4%) | — | 41.1 (6.8%) | — |
| ds_tp4_m512 | 710.44 | — | — | 2.59 (48.8%) | — | 76.5 (12.6%) | — |
| ds_tp4_m1024 | 908.87 | — | — | 2.03 (38.3%) | — | 120 (19.7%) | — |
| ds_tp4_m2048 | 1444.33 | — | — | 1.29 (24.4%) | — | 151 (24.8%) | — |
| ds_tp4_m4096 | 2145.97 | — | — | 0.89 (16.8%) | — | 203 (33.4%) | — |
| ds_tp4_m8192 | 3573.08 | — | — | 0.56 (10.5%) | — | 243 (40.2%) | — |
| ds_tp4_m16384 | 6765.36 | — | — | 0.32 (6.0%) | — | 257 (42.4%) | — |
| ds_tp4_m32768 | 13247.2 | — | — | 0.19 (3.6%) | — | 263 (43.3%) | — |
| mimo_tp4_m1 | 28.36 | — | — | 0.95 (17.9%) | — | 3.52 (0.6%) | — |
| mimo_tp4_m2 | 40.58 | — | — | 1.31 (24.7%) | — | 4.98 (0.8%) | — |
| mimo_tp4_m4 | 65.3 | — | — | 1.58 (29.9%) | — | 6.18 (1.0%) | — |
| mimo_tp4_m8 | 100.38 | — | — | 1.89 (35.7%) | — | 8.01 (1.3%) | — |
| mimo_tp4_m16 | 157.34 | — | — | 2.24 (42.2%) | — | 10.2 (1.7%) | — |
| mimo_tp4_m32 | 230.38 | — | — | 2.37 (44.7%) | — | 14.0 (2.3%) | — |
| mimo_tp4_m48 | 303.32 | — | — | 2.29 (43.2%) | — | 15.9 (2.6%) | — |
| mimo_tp4_m64 | 304.6 | — | — | 2.51 (47.4%) | — | 21.2 (3.5%) | — |
| mimo_tp4_m128 | 332.8 | — | — | 2.55 (48.1%) | — | 38.7 (6.4%) | — |
| mimo_tp4_m256 | 363.12 | — | — | 2.38 (44.9%) | — | 71.0 (11.7%) | — |
| mimo_tp4_m512 | 422.64 | — | — | 2.06 (38.8%) | — | 122 (20.1%) | — |
| mimo_tp4_m1024 | 587.46 | — | — | 1.49 (28.2%) | — | 175 (29.0%) | — |
| mimo_tp4_m2048 | 892.76 | — | — | 1.00 (18.9%) | — | 231 (38.1%) | — |
| mimo_tp4_m4096 | 1481.39 | — | — | 0.63 (11.9%) | — | 278 (45.9%) | — |
| mimo_tp4_m8192 | 2657.09 | — | — | 0.37 (7.0%) | — | 310 (51.2%) | — |
| mimo_tp4_m16384 | 5087.07 | — | — | 0.22 (4.2%) | — | 324 (53.5%) | — |
| mimo_tp4_m32768 | 9991.84 | — | — | 0.14 (2.6%) | — | 330 (54.5%) | — |
| mimo_tp8_m1 | 19.48 | — | — | 0.69 (13.1%) | — | 2.59 (0.4%) | — |
| mimo_tp8_m2 | 27.44 | — | — | 0.97 (18.3%) | — | 3.68 (0.6%) | — |
| mimo_tp8_m4 | 39.66 | — | — | 1.29 (24.3%) | — | 5.09 (0.8%) | — |
| mimo_tp8_m8 | 62.48 | — | — | 1.51 (28.5%) | — | 6.42 (1.1%) | — |
| mimo_tp8_m16 | 98.08 | — | — | 1.73 (32.6%) | — | 8.21 (1.4%) | — |
| mimo_tp8_m32 | 141.36 | — | — | 1.94 (36.6%) | — | 11.4 (1.9%) | — |
| mimo_tp8_m48 | 170.12 | — | — | 1.97 (37.2%) | — | 14.2 (2.3%) | — |
| mimo_tp8_m64 | 170.1 | — | — | 2.18 (41.1%) | — | 18.9 (3.1%) | — |
| mimo_tp8_m128 | 191.62 | — | — | 2.22 (41.9%) | — | 33.6 (5.5%) | — |
| mimo_tp8_m256 | 200.6 | — | — | 2.18 (41.1%) | — | 64.2 (10.6%) | — |
| mimo_tp8_m512 | 240.54 | — | — | 1.83 (34.6%) | — | 107 (17.7%) | — |
| mimo_tp8_m1024 | 344.42 | — | — | 1.30 (24.6%) | — | 150 (24.7%) | — |
| mimo_tp8_m2048 | 543.8 | — | — | 0.86 (16.2%) | — | 190 (31.3%) | — |
| mimo_tp8_m4096 | 872.38 | — | — | 0.57 (10.8%) | — | 236 (39.0%) | — |
| mimo_tp8_m8192 | 1526.67 | — | — | 0.37 (7.0%) | — | 270 (44.6%) | — |
| mimo_tp8_m16384 | 2849.26 | — | — | 0.25 (4.7%) | — | 289 (47.8%) | — |
| mimo_tp8_m32768 | 5635.21 | — | — | 0.17 (3.2%) | — | 293 (48.3%) | — |
| tp2_m1 | 18.12 | 23.28 | 0.779 | 0.81 (15.3%) | 1.17 (22.1%) | 2.99 (0.5%) | 2.31 (0.4%) |
| tp2_m2 | 27.12 | 36.18 | 0.749 | 1.03 (19.5%) | 1.43 (27.0%) | 3.99 (0.7%) | 2.99 (0.5%) |
| tp2_m4 | 40.34 | 61.14 | 0.660 | 1.33 (25.1%) | 1.63 (30.7%) | 5.38 (0.9%) | 3.52 (0.6%) |
| tp2_m8 | 65.44 | 106.90 | 0.612 | 1.59 (30.0%) | 1.80 (33.9%) | 6.60 (1.1%) | 4.02 (0.7%) |
| tp2_m16 | 100.08 | 164.84 | 0.607 | 1.85 (34.9%) | 2.08 (39.2%) | 8.62 (1.4%) | 5.22 (0.9%) |
| tp2_m32 | 168.26 | 260.48 | 0.647 | 1.94 (36.6%) | 2.32 (43.8%) | 10.3 (1.7%) | 6.62 (1.1%) |
| tp2_m48 | 250.98 | 315.32 | 0.798 | 1.71 (32.3%) | 2.52 (47.5%) | 10.3 (1.7%) | 8.22 (1.4%) |
| tp2_m64 | 203.1 | 340.38 | 0.597 | 2.34 (44.2%) | 2.59 (48.8%) | 17.0 (2.8%) | 10.2 (1.7%) |
| tp2_m128 | 259.92 | 407.28 | 0.638 | 2.43 (45.9%) | 2.87 (54.2%) | 26.6 (4.4%) | 17.0 (2.8%) |
| tp2_m256 | 288.76 | 429.56 | 0.672 | 2.36 (44.5%) | 2.93 (55.3%) | 47.9 (7.9%) | 32.2 (5.3%) |
| tp2_m512 | 323.62 | 443.80 | 0.729 | 2.13 (40.2%) | 2.87 (54.1%) | 85.5 (14.1%) | 62.4 (10.3%) |
| tp2_m1024 | 433.42 | 477.66 | 0.908 | 1.60 (30.2%) | 2.68 (50.5%) | 128 (21.1%) | 116 (19.1%) |
| tp2_m2048 | 611.42 | 707.60 | 0.864 | 1.15 (21.7%) | 1.82 (34.3%) | 181 (29.9%) | 156 (25.8%) |
| tp2_m4096 | 1020.2 | 1131.63 | 0.902 | 0.71 (13.4%) | 1.16 (21.9%) | 217 (35.8%) | 196 (32.3%) |
| tp2_m8192 | 1714.29 | 1943.81 | 0.881 | 0.45 (8.5%) | 0.70 (13.2%) | 258 (42.6%) | 228 (37.6%) |
| tp2_m16384 | 3206.26 | 3609.28 | 0.888 | 0.27 (5.1%) | 0.40 (7.5%) | 276 (45.6%) | 245 (40.5%) |
| tp2_m32768 | 6276.16 | 6940.32 | 0.905 | 0.16 (3.0%) | 0.23 (4.3%) | 282 (46.6%) | 255 (42.1%) |
| tp4_m1 | 14.72 | 17.84 | 0.825 | 0.60 (11.3%) | 0.92 (17.3%) | 2.20 (0.4%) | 1.81 (0.3%) |
| tp4_m2 | 21.18 | 26.94 | 0.785 | 0.79 (14.9%) | 1.16 (21.9%) | 3.08 (0.5%) | 2.40 (0.4%) |
| tp4_m4 | 29.88 | 42.68 | 0.700 | 1.07 (20.2%) | 1.39 (26.3%) | 4.32 (0.7%) | 3.02 (0.5%) |
| tp4_m8 | 45.34 | 72.56 | 0.625 | 1.33 (25.1%) | 1.54 (29.1%) | 5.71 (0.9%) | 3.59 (0.6%) |
| tp4_m16 | 74.08 | 130.08 | 0.569 | 1.53 (28.8%) | 1.62 (30.5%) | 7.00 (1.2%) | 4.00 (0.7%) |
| tp4_m32 | 114.9 | 178.42 | 0.644 | 1.71 (32.3%) | 2.05 (38.7%) | 9.02 (1.5%) | 5.81 (1.0%) |
| tp4_m48 | 143.22 | 214.50 | 0.669 | 1.75 (33.0%) | 2.17 (41.0%) | 10.9 (1.8%) | 7.28 (1.2%) |
| tp4_m64 | 140.56 | 245.56 | 0.573 | 2.10 (39.6%) | 2.23 (42.0%) | 14.8 (2.4%) | 8.48 (1.4%) |
| tp4_m128 | 173.1 | 288.18 | 0.600 | 2.17 (41.0%) | 2.42 (45.7%) | 24.0 (4.0%) | 14.4 (2.4%) |
| tp4_m256 | 193.04 | 327.56 | 0.590 | 2.12 (40.0%) | 2.32 (43.8%) | 43.0 (7.1%) | 25.4 (4.2%) |
| tp4_m512 | 222.3 | 342.42 | 0.648 | 1.87 (35.3%) | 2.24 (42.3%) | 74.7 (12.3%) | 48.5 (8.0%) |
| tp4_m1024 | 276.4 | 356.70 | 0.775 | 1.52 (28.7%) | 2.17 (40.9%) | 120 (19.8%) | 93.1 (15.4%) |
| tp4_m2048 | 442.3 | 521.12 | 0.849 | 0.98 (18.4%) | 1.50 (28.3%) | 150 (24.8%) | 127 (21.0%) |
| tp4_m4096 | 725.54 | 895.46 | 0.811 | 0.62 (11.7%) | 0.90 (17.0%) | 183 (30.2%) | 148 (24.5%) |
| tp4_m8192 | 1216.27 | 1444.67 | 0.843 | 0.41 (7.7%) | 0.59 (11.1%) | 218 (36.1%) | 184 (30.4%) |
| tp4_m16384 | 2220.51 | 2610.51 | 0.851 | 0.26 (4.9%) | 0.36 (6.8%) | 239 (39.5%) | 204 (33.6%) |
| tp4_m32768 | 4259.34 | 5101.67 | 0.833 | 0.18 (3.4%) | 0.22 (4.1%) | 250 (41.2%) | 208 (34.4%) |

TB/s、TFLOPS = bench 给出的每次调用数据量 / 运算量 ÷ 中位耗时，括号内占峰值（带宽 5.3 TB/s、算力 606 TFLOPS）；对照 = aiter_fp8_ptpc（MF_REF，同一 bench 进程、同一协议）

</details>
<!-- /mf:perf -->
## 5. 测试覆盖
三类测试都在 [test_moe_a8w4.py](../../../../../../tests/ops/moe/test_moe_a8w4.py) 一个文件：A8W4 与底座各有基本用例；85 个性能 case 覆盖 A8W4 三个模型三条路径的各分派（底座无）；交界、形状回退、变体、逐位、报错、截断与 graph 在其他覆盖。

**基本功能测试**（区域 1–2，每次改动先跑：`commands.correctness` = `pytest tests/ops/moe/test_moe_a8w4.py -m basic`，15 例，约 48 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| Qwen：TP2（I320，E33）M3 decode、M64 流式、M1031 prefill；TP4 I160→192、M257 prefill | `test_tp2` ×3、`test_tp4_padded` | 通过 |
| MiMo（H4096，E256，topk8）：TP8 M5 去重 decode（2 kernel）、M300 流式；TP4 M24 去重（4 kernel）、M1600 prefill | `test_mimo` ×4 | 通过 |
| DeepSeek（H5120，E24，topk6，I576）M64 流式 BM32，swiglu_limit=10、输入 ×10 使截断大量触发 | `test_deepseek_swiglu` | 通过 |
| 底座 decode（batch1、split-K）、prefill（三种 Down、BF16） | `test_fp8_batch1`、`test_fp8_splitk`、`test_fp8_prefill*` ×4 | 通过 |

**性能测试**（区域 5，`commands.bench` = `python tests/ops/moe/test_moe_a8w4.py`；85 个 case：5 个形状 × M=1…32768（2 的幂）与 48）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| decode：M1–2 核内量化（2 个 kernel），M4–32 先量化 mid（较大 M 也先量化 x） | `tp*_m1`…`tp*_m32` | 0.569–0.825×，0.60–1.94 TB/s |
| 流式 BM16（M64–512）、BM32/64 尾部量化（M48、1024，TP4 2048）、原子加 | `tp*_m48`…`m1024`、`tp4_m2048` | 0.573–0.908×，0.98–2.43 TB/s |
| prefill BM64·TK128（TP2 ≥2048、TP4 ≥4096；M2048 用 prep） | `tp2_m2048`…、`tp4_m4096`… | 0.811–0.905×，181–282 TFLOPS |
| MiMo：M1 decode、2–48 去重、64–1024 流式、≥2048 prefill（2048 用 prep） | `mimo_*` | 无对照 0.69–2.55 TB/s、190–330 TFLOPS |
| DeepSeek（swiglu_limit=10）：M≤32 回退 prefill、48–2048 流式、≥4096 prefill | `ds_tp4_*` | 无对照 0.41–2.72 TB/s、203–263 TFLOPS |

**其他覆盖测试**（区域 3–4，只测前两类覆盖不了的；`commands.verify_extra` = 同一文件 `-m "not basic"`，124 例；用例名省略 `test_`）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| decode 交界 M=4、TP2 11/12、TP4 19/20、32/33 | 基本 M=3/5/24；bench 无 11/12、19/20、33 | `decode_dispatch_boundaries` ×7 | 通过 |
| 去重交界：M·I=3072（I256 M11/12、I512 M5/6）、M16/17 组 2→3 与 8→4 wave、I384 | 组少满员；bench 无 | `decode_dedup_dispatch` ×8 | 通过 |
| 去重 decode 与流式的交界 E256 M=48 / 49 | bench 只有 48、64，不断言 kernel | `decode_dedup_stream_boundary` ×2 | 通过 |
| 流式→prefill E33 M66/67、grid 收缩 E513 M46/47 | bench 无 | `stream_dispatch_boundary` ×2、`grid_shrink_boundary_e513` | 通过 |
| 流式上限（56 / 48 行）两侧；换 prefill Down 及其切分 | bench 无交界、切分单一 | `stream_rows_limits` ×5、`stream_prefill_down_split` ×3 | 通过 |
| prefill 半块交界；给 tile 时的 TK 与 BM128·BN256 | 随机路由碰不到；不给 tile | `prefill_half_block_boundary` ×3、`tile_variants` | 通过 |
| prefill 排序用 prep / Opus 的路由上限两侧；prep 时重复 expert 逐槽相加 | 不在交界、不断言排序 kernel；路由不重复 | `prefill_prep_routes_limit` ×3 | 通过 |
| 形状回退：H4608、H4096·I64 走原 decode、槽对 Down I1024/1088、I1024 流式 Down 共享与否 | 前两类 I≤576、不断言路径 | `shape_fallbacks` ×6 | 通过 |
| decode topk=8 时 WG 数恰为 2560；topk=17 时 Down 4 wave 轮流取槽 | 无 topk 8 的槽对、无 topk>16 | `decode_topk_variants` ×2 | 通过 |
| decode 同 token 重复 expert、int64 非连续 topk_ids、非连续 topk_weight | 生成的路由不重复、张量连续 | `decode_duplicate_and_int64_ids` | 通过 |
| 重复 expert 在去重 decode、流式（原子加 / route_sum）也逐槽相加 | 同上 | `duplicate_ids_keep_every_route` ×4 | 通过 |
| Opus 排序时 prefill 重复 expert 只留后一槽，loc −1 不越界 | 同上 | `prefill_duplicate_ids_keep_last_route`、`row_quant_fill` ×2 | 通过 |
| 原地输出 out=x：去重 decode 改走四个 kernel，流式 prep 先读完 x 再清零 | 前两类的 output 都另行分配 | `inplace_output` ×2 | 通过 |
| decode 在 w2 ≥ 4 GiB 时改用核内量化 Down（stride 0 视图冒充 4 GiB） | 前两类的 w2 都远小于 4 GiB | `decode_down_fallback_w2_4gib` | 通过 |
| graph 捕获与重放：流式 M64、prefill、decode M1/5/16、去重 M5/24 | 都是 eager 调用 | `graph_replay` ×2、`decode_*graph_replay` ×5 | 通过 |
| 行量化逐位：中点、-0、次正规、越界行、乘倒数会舍错的商、scale<2^-64 | fp8 翻转看不出 | `row_quant_bitexact` ×2、`decode_row_quant_bitexact` ×6 | 通过 |
| 去重计划 kernel：x 量化逐位、清零 out、计划的条数与组员（跨段、尾组不满） | 同上；随机路由少碰跨段 | `decode_plan_quant_bitexact` | 通过 |
| 流式 prep 量化（QW4 / 1）、清零、排序格式；Gate/Up 尾部量化 | 同上 | `stream_row_quant_bitexact` ×4、`stream_fused_quant_bitexact` ×2 | 通过 |
| A8W4 prepare 布局逐字节；delta 11 / 12 边界 | 端到端看不出；随机权重碰不到 | `prepare_layout_bitexact`、`delta_flush_boundary` | 通过 |
| A8W4 报错：dtype、形状、E8M0 NaN、H/I 不对齐、BM·I_pad 非 4096 倍数、hidden 不符、swiglu_limit 非法 | 只给合法输入 | `rejects_bad_inputs` | 通过 |
| 截断：decode 两种、去重两种、流式三种、prefill 两种、H5120 小 M | 基本只有流式 BM32；bench 几乎不触发 | `swiglu_limit_kernels` ×10 | 通过 |
| 截断只限 gate 上侧：gate 全为 −20 时不夹到 −10（流式、H5120 小 M） | 下侧误夹只差 <5e-4，端到端看不出 | `swiglu_gate_upper_only` ×2 | 通过 |
| 底座变体：per-tensor/BF16、SwiGLU、8x1 族、N512、K640、M·topk≤E | 只测 PTPC、SiLU、Qwen 形状 | `fp8_*_variants`、`fp8_swiglu` ×2 | 通过 |
| 底座 batch1 清零写满、graph 重放（排序、loc_ids、任务表） | 基本只跑 eager、不查两侧 | `fp8_batch1_clear_graph` ×4、`fp8_graph_replay` ×2 | 通过 |
| 底座报错：K 非 256 倍数、Gate/Up、Down 的 M 不同 | 只走合法配置 | `fp8_gateup_rejects_incomplete_k` ×4、`fp8_split_m_tile_rejected` | 通过 |
| 底座逐位：FP32→BF16 舍入、Down 打包前只舍一次 | calc_diff 看不出舍入 | `fp8_bf16_rounding_bits`、`fp8_down_rounds_before_packing` ×3 | 通过 |
| 底座候选门槛：不完整 N tile、默认 Down 超 64 KiB；恰 64 KiB 可用 | 前两类不枚举候选 | `fp8_candidate_gates` ×5、`fp8_lds_64k_boundary` | 通过 |
#### 未覆盖 / 跳过
- I_pad > 1024：prefill Down 的 64·I_pad 超过 64 KiB，走 prefill 时（超过流式上限、给 tile 参数、decode 不支持）构建即报错，入口不检查。
- 未测的交界：流式排序块 16 / 32（路由 4E/3）与 32 / 64（28 行）、去重 decode 的 M ≤ 64 上限（要 E ≥ 342）；去重 Down 在 WG 内量化 mid 只靠端到端判据。
- w2 真超过 4 GiB：只用 stride 0 视图验证分派与单槽 Down，没在大权重上跑过 64 位 expert 寻址；这个回退也没有性能 case。
- 含 NaN 的输入或权重（截断用 min/max，与 torch.clamp 不同）、M·topk·I_pad·4 ≥ 2^31（入口不检查）、gfx950 与同进程多卡、底座 per-tensor 预清零（正常路由触发不到）：不测。
- tuned_moe 的调优缓存、剪枝与非 FlyDSL 路径在 `test_tuned_moe.py` / `test_moe.py`（两个旧用例起点即失败），不在本文件里。
## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
主要风险是逐点目标未达 7 / 83 点：DeepSeek M4 落后 tingqli 15.6%（回退 prefill → 做 decode）；MiMo TP8 M16、M512–8192 差 ≤4.8%（→ 合并辅助 kernel）。
- prefill 持续负载与 aiter 一样顶在 650 W 功耗墙（sclk 约 1.55–1.6 GHz）→ TP2 M16384 只到 46% fp8 峰值，减 VALU 不再减耗时。
- Down 的 LDS bank conflict 占 LDS 活跃周期 59%（未处理）；流式 HBM 只到 18–47%，瓶颈是 LUT 解码 VALU → 再快要减解码指令。
- 约束 H%512、I%32、M·topk·I_pad·4 < 2^31，I_pad > 1024 时 prefill 不支持；Opus 排序丢重复 expert 的槽；流式排序块规则沿用 aiter 排序的限制、未复核。
- host 下发中位 decode 约 0.14 ms、流式 0.24 ms、prefill 0.33 ms，编译缓存不按设备 → 实用靠 CUDA graph（各路径已验证）、每卡一进程。
## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../moe_a8w4.py) 的 `prepare_a8w4_weights`（加载时一次）与 `fused_moe_a8w4`（接口与约束见 docstring，分派门槛在文件头与 `_stream_*`）。
- kernel 设计：[kernels.md](kernels.md)；私有布局与 LUT 解码见 [a8w4_common.py](a8w4_common.py)，块对、槽对与去重计划见 [a8w4_decode.py](a8w4_decode.py)、[a8w4_decode_dedup.py](a8w4_decode_dedup.py) docstring。
- 参考与测试：[pyhip.testing.moe_a8w4](../../../../testing/moe_a8w4.py)；三类测试（含性能测试）在 [test_moe_a8w4.py](../../../../../../tests/ops/moe/test_moe_a8w4.py)，说明见 [tests/ops/moe/README.md](../../../../../../tests/ops/moe/README.md)。
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
- M7 ✅ 已推送 · 维护 · 补充 M=48 性能 case：覆盖 E513 下 M=33–62 的排序块 … · [提交](https://github.com/luocheng25/pyhip/commit/e8b7260961cf47f1c4565ecf95017e50573f3a88)
- M8 ✅ 已推送 · 维护 · 测试合并为单一文件、kernels.md 增加 Triton 风格写法 · [提交](https://github.com/luocheng25/pyhip/commit/3aa7bee2d8c65dd59a9aef896f3fcf54ad8b8db6)
- M9 ✅ 已推送 · 维护 · kernels.md：Triton 风格改为 Gluon 风格的分阶段写法（显… · [提交](https://github.com/luocheng25/pyhip/commit/fb345e1a5f7519c225e69b4f488467dec2d8a41f)
- M10 ✅ 已推送 · 维护 · 第三轮度量契约：bench 与用例加入 MiMo-V2.6-Flash TP8… · [提交](https://github.com/luocheng25/pyhip/commit/00b617f1e9623b1f66fd22dc37e96724b654ca65)
- M11 ✅ 已推送 · 性能 · MiMo 小 / 中 M（8–2048）追上 tingqli：decode /… · 1.056× 对 M10a（预期 1.04×） · [提交](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0)
- M12 ✅ 已推送 · 性能 · 大 M prefill（≥1024）追上 R1 / tingqli：Gate/… · 1.063× 对 M11（预期 1.03×） · [提交](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)
- M10a ✅ 已推送 · 维护 · 用例的路径断言改为检查关键 kernel 顺序（第三轮去掉 route_sum… · [提交](https://github.com/luocheng25/pyhip/commit/4a637c36d275f778cf6eeab43a4c62190a4f1cf4)
- M13 ✅ 已推送 · 维护 · prefill M≈2048 排序与 x 量化合并（复用流式 prep），针对… · 1.000× 对 M12（预期 1×） · [提交](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a)
- M14 ✅ 已推送 · 功能 · SwiGLU 截断：fused_moe_a8w4 加 swiglu_limit… · [提交](https://github.com/luocheng25/pyhip/commit/bcec12e99c467236ae739254a118f35cd54f9cfa)
- M15 ✅ 本提交 · 维护 · 度量契约：bench 与用例加入 DeepSeek-V4.1-Flash TP…
- M16 ⏳ 承诺 · 性能 · hidden 5120 的 decode：DeepSeek TP4 M≤32 … · 预期 1.08×
- M17 · 暂定 · 性能 · DeepSeek TP4 调参：流式 / prefill 的分界、排序块与 t… · 预期 1.015×

下一步（本提交的承诺）：M16 hidden 5120 的 decode：DeepSeek TP4 M≤32 不再回退 prefill（Gate/Up decode 每线程多段 16 B，Down decode 核对 I=576）；之后：M17 DeepSeek TP4 调参：流式 / prefill 的分界、排序块与 tile 在 H5120 上重标定（M48–32768）（暂定）
<!-- /mf:roadmap -->
