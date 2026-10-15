# Qwen3.8 MonoKernel：使用说明与最终结果

MonoKernel 把 Qwen3.8-Flash-Next（`qwen4_exp`）一个 decode 层合成一次常驻 kernel 启动。本页只维护当前使用说明、最终有效结果和验收状态；修改与优化过程追加到[源码readme](../../src/pyhip/ops/monokernel/qwen38/readme.txt)。

当前阶段为 P1：GDN 层（线性注意力层）的阶段版内核（每个阶段一次启动，两次 all-reduce 之间分三段）已实现，并在 TP2/TP4 真实采集上与 SGLang 对齐；单次启动融合、full attention 层（P2/P3）和服务接入（P4）尚未完成，因此还没有性能结论。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 模型几何与 TP 分片 | [config.py](../../src/pyhip/ops/monokernel/qwen38/config.py) |
| checkpoint 读取、SGLang 一致的分片、SGLang 服务内格式、合成权重 | [weights.py](../../src/pyhip/ops/monokernel/qwen38/weights.py) |
| decode 层 golden（按阶段拆分） | [reference.py](../../src/pyhip/ops/monokernel/qwen38/reference.py) |
| 阶段内核（FlyDSL 0.3.2）：量化与线性层、HC、GDN、MoE | [flydsl/](../../src/pyhip/ops/monokernel/qwen38/flydsl/) |
| 阶段版整层驱动（按 all-reduce 分三段） | [flydsl/layer.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/layer.py) |
| golden 与权重格式单元测试 | [test_qwen38_reference.py](../../tests/ops/monokernel/test_qwen38_reference.py)、[test_qwen38_weights.py](../../tests/ops/monokernel/test_qwen38_weights.py) |
| 阶段内核单元测试（对 golden） | [test_qwen38_stages.py](../../tests/ops/monokernel/test_qwen38_stages.py)、[test_qwen38_moe.py](../../tests/ops/monokernel/test_qwen38_moe.py) |
| SGLang 解码采集（自启服务 + 插件），采完自动检查 | [capture.py](capture.py)、[capture_plugin](capture_plugin/qwen38_capture_plugin.py) |
| golden 与采集逐阶段比对、阈值 | [check_reference.py](check_reference.py) |
| 阶段内核与采集比对（逐阶段 + 整层链式） | [check_kernels.py](check_kernels.py) |

目标为 **ROCm gfx942（MI308X）**。验证环境：Python3.10.12、Torch2.12/ROCm7.14、FlyDSL0.3.2、SGLang `1d383c5`、AITER `4752a40`。golden 和单元测试只依赖 PyTorch；采集另需 SGLang、AITER 和 `/models/Qwen3.8-Flash-Next-PTPC-FP8`。AITER 固定 `flydsl==0.3.2`，不要在同一环境安装 FlyDSL main：它会让 AITER 的 FlyDSL MoE（生产 decode MoE）无法导入或编译。

本机示例统一使用 PyHIP 的 .venv 解释器。新数据目录在 mytest/mydata 下且必须尚不存在。修改 PyHIP 中的 FlyDSL 代码后，测试前设置 `export FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`。

## 1. 单元测试

```bash
.venv/bin/python -m pytest -q tests/ops/monokernel
```

使用 16 个专家的合成权重，覆盖 TP2/4/8：各 rank 的 GDN 投影、conv、递推和状态，以及 attention 的 Q/K/V/gate 和稀疏注意力，都与 TP1 结果的对应切片一致（容差 1 个 BF16 ulp）；负槽位输出 0 且不更新状态；整层与 TP1 接近（各 rank 独立做 FP8 再量化，不会逐位相同）；路由并列时选较小的 expert id；HC mix 的近 tie 解析只翻转最接近 tie 的那一个低秩激活。权重格式：16x16 shuffle 与 AITER `shuffle_weight` 逐位相同，E4M3FN→E4M3FNUZ 反量化值不变，TP2/TP4 下 `sglang_format` 覆盖全部 SGLang 参数，padding 行为 0。

阶段内核测试（S=1/5/16/33/64）用 SGLang 服务内格式的权重：逐 token FP8 量化与 golden 逐位相同；W8A8/BF16 线性层、HC mix 与 combine+norm、qkvz+conv（含 conv 状态）、delta rule（含 SSM 状态）与门控 norm、路由 top-k 与分组计划、W8A16 MoE 均与 golden 在 1 个 BF16 ulp 内（判据：≥99.9% 元素 ≤1 ulp、rel_l2 ≤ 1e-3；MoE 中间量与输出放宽到 99.5%）。MoE 用完整 512+1 专家，覆盖 TP2（中间维 320）与 TP4（160 补到 192）。

## 2. 整体正确性：与 SGLang 对齐

```bash
.venv/bin/python benchmarks/monokernel/capture.py --tp 2 4 --devices 0,1,2,3

# 只重新检查已有采集
.venv/bin/python benchmarks/monokernel/check_reference.py \
  --capture <dir>/tp2/capture --output <dir>/tp2/check_reference.json
```

`capture.py` 先建 checkpoint 的 4 层视图（第 0–2 层 GDN，第 3 层 full attention+QSA；去掉 PLE 和 MTP），再按每个 TP 起一个自有 SGLang 服务：AITER attention/MoE（`AITER_MOE_PADDING_SIZE=64`，同启动脚本）、eager（`--disable-cuda-graph`）、`--skip-server-warmup`，原生 QSA，不加载 PyHIP 加速。负载是 PyHIP 文档的连续 token 片段：一批 4 个请求（3600/2100/900/3000，两长两短于 QSA 的 2048 预算）加 1 个单请求（2600），各生成 3 个贪心 token。插件记录每个 rank 的前 4 个 decode 步：3 步 S=4（overlap 调度会多跑一步）和 1 步 S=1；第一步还写出各层参数与状态池的布局（`layout.json`）和每个参数的原始位等间隔采样（`param_samples.pt`）。`--devices` 用当前进程可见的 GPU 编号，TP N 取前 N 张；入口检查这些卡的显存，被占用则拒绝启动。省略 `--output` 时在 mytest/mydata 下新建目录。

每个 TP 目录输出 `capture/`、`server.log`、`manifest.json`（命令、环境、FlyDSL/Torch 版本与 PyHIP/SGLang/AITER 的 git HEAD）和 `check_reference.json`；只要有一个 TP 未采全或检查失败，退出码就非 0。

`check_reference.py` 的检查分四类：

- 逐阶段：每个 SGLang 阶段从它自己采到的输入重算，包括 HC norm/mix/combine、GDN 投影/conv+递推（含状态）/输出投影、attention 的 Q/K/V/gate、稀疏注意力、输出投影，以及路由和 MoE。
- all-reduce：各 rank 采到的部分和按 rank 顺序求和。
- 链式：每层从层输入重算整层，所有 rank 在同一进程，状态从采集值出发。MoE 和层输出只比较路由与 SGLang 一致的行；路由不同的行另计，并要求是近似并列。
- 权重格式（`served_format`）：checkpoint 经 `sglang_format` 转换后，与服务内参数的原始位采样逐位比较（一个 GDN 层和一个 attention 层，每个 rank），任何不同都判失败。

指标为 rel_l2、max_abs、逐位相等比例 `exact`、1 个 BF16 ulp 以内比例 `ulp1`；阈值见 `THRESHOLDS`，约为下表最坏值的 2–3 倍。HC mix 的 320 个低秩激活先舍入到 BF16 再参与上投影，一个落在 BF16 tie 附近的激活随 FP32 累加顺序舍入到另一侧，就会使整行约 2.5% 的输出差 1 ulp 以上。因此 HC mix 检查对距 tie 不超过 32 个 FP32 ulp 的激活尝试另一侧舍入（每行最多 2 个，只保留能减少不一致的翻转），并报告翻转数 `tie_flips` 与距离 `tie_ulps`。SGLang 各 rank 上本应相同的张量出现差异时只报告、不判失败，原因是 SGLang 的 HC mix 用原子累加。

### 当前结果

2026-10-13，MI308X，生产 FlyDSL decode MoE，TP2（GPU0/1）与 TP4（GPU0/1/4/5）均判定 PASS。下表为全部 step、层、rank 中的最坏值，格式为 TP2 / TP4。

| 检查 | rel_l2 | exact | ulp1 |
|---|---:|---:|---:|
| HC norm / mix / combine（attn、mlp 两组） | ≤ 4.5e-5 / 5.6e-5 | ≥ 0.9996 / 0.9997 | 1.000 / 1.000 |
| HC mix 近 tie 翻转 | 每处最多 1 个，距 tie ≤ 30.9 / 28.1 个 FP32 ulp | — | — |
| GDN qkvz / ba / conv 状态 / 递推输出 / ssm 状态 | ≤ 5.3e-5 / 7.0e-5 | ≥ 0.305 / 0.199 | 1.000 / 1.000 |
| GDN 输出投影 | 4.1e-3 / 9.1e-5 | 0.492 / 0.9996 | 0.987 / 0.9999 |
| attention q / k / v / gate | ≤ 5.6e-5 / 7.1e-5 | ≥ 0.9997 / 0.9998 | 1.000 / 1.000 |
| 稀疏注意力 | 2.7e-4 / 1.1e-4 | 0.996 / 0.996 | 0.9995 / 0.9998 |
| attention 输出投影 | 4.1e-3 / < 1e-6 | 0.493 / 0.9999 | 0.9998 / 0.9999 |
| 两次 all-reduce | ≤ 3.6e-3 / 4.0e-3 | ≥ 0.713 / 0.624 | 1.000 / 1.000 |
| 路由 logits | 1.2e-4 / 0 | 0.9995 / 1.000 | 1.000 / 1.000 |
| 路由选择（采集的 logits） | 0 行不同 / 0 行不同 | — | — |
| MoE（采集的路由） | 5.1e-3 / 5.4e-3 | 0.362 / 0.355 | 0.758 / 0.745 |
| 链式：attention 块 / MoE / 层输出 | 1.5e-2 / 1.0e-2 / 6.3e-3；4.0e-3 / 9.7e-3 / 4.9e-3 | — | — |
| 链式：路由 | 52 行中 1 行不同（概率完全并列）；2 行不同（概率差 ≤ 3.1%） | — | — |
| 链式：conv / ssm 状态 | 8.9e-5 / 2.2e-5；2.9e-5 / 6.5e-5 | — | — |
| 权重格式 | 每 rank 两层各 23 个张量，0 处不同 | — | — |

GDN 的 ssm 状态是 FP32，`exact` 只反映 FP32 末位差异。TP2 S=4 的 GDN 和 attention 输出投影 4e-3 来自 AITER 为 (M=4, N=2560, K=3072) 调优选中的 cktile GEMM 截断到 BF16；TP4（K=1536）和 S=1 不走该内核，逐位一致。两次 all-reduce 的差异来自 AITER all-reduce 截断，不超过 1 ulp。MoE 剩余约 0.5% 来自 AITER 把每个专家输出舍入到 BF16 再用 BF16 原子加累加。

### SGLang 服务内的权重与状态格式

MonoKernel 直接读取 SGLang 加载后的张量，不另存权重。`sglang_format` 由 checkpoint 生成同样的格式，供单元测试和离线对齐使用；上表的权重格式检查确认两者逐位一致。

| 张量 | 形状（TP2 / TP4） | 格式 |
|---|---|---|
| 稠密 FP8：GDN `in_proj_qkvz` / `out_proj`，attention `qkv_proj` / `o_proj` | [N, K]，如 qkvz [8192, 2560] / [4096, 2560] | E4M3FNUZ，按 AITER `shuffle_weight(w, (16, 16))` 预排（16 行 × 16 字节一块，块内行主序，16 行组内沿 K 排列）；scale 为 FP32 [N, 1]，是 checkpoint 的 2 倍 |
| MoE `w13` / `w2`（513 个专家，最后一个为共享专家） | [513, 2·I, 2560] / [513, 2560, I]，I = 320 / 192 | 同上；中间维每半补零到 64 的倍数（TP4 的 160 补到 192）；scale 为 FP32 [513, 2·I, 1] / [513, 2560, 1]，补零行的 scale 为 0 |
| BF16：HC 四组、`in_proj_ba`、`conv1d`、norm、路由、共享专家门、indexer | 与 checkpoint 相同 | 行主序，未预排；`in_proj_ba` 为 [b; a]；`A_log` 为 FP32 |
| GDN 状态 | conv [槽位, conv_dim, 3] BF16；temporal [槽位, HV, V, K] FP32 | 连续；槽位由 `mamba_cache_indices` 给出 |
| attention KV | [token 槽位, KV 头, 256] BF16 | 连续；`req_to_token` 为 int32 [请求, 262148] |

### MonoKernel 阶段内核与 SGLang 对齐

```bash
.venv/bin/python benchmarks/monokernel/check_kernels.py \
  --capture <dir>/tp2/capture --output <dir>/tp2/check_kernels.json
```

读取同一份采集，内核使用 checkpoint 经 `sglang_format` 得到的服务内格式权重（与服务逐位相同）和 `pack_layer` 生成的 BF16 预排副本，目前覆盖 GDN 层（第 0–2 层）。两种模式：

- `kernel.*`：每个内核阶段从 SGLang 采到的输入出发，只衡量该内核。GDN 递推用 SGLang 自己的 conv 输入、门控和 z；out_proj 用 SGLang 的 core 经 golden 门控 norm 后的结果；MoE 用内核自己的路由，只比较专家集合与 SGLang 一致的行。
- `chain.*`：阶段版整层（`StagedLayer`）在同一进程内跑完所有 rank，从采集的层输入出发，两次 all-reduce 按 rank 顺序求和（同 `golden_allreduce`）。

阈值沿用 `check_reference.py` 中同一物理量的阈值；HC mix 不做近 tie 解析，`ulp1` 下限为 0.995。2026-10-13 结果（`qwen38_mono_p0_20261013_06`，最坏值，TP2 / TP4，均 PASS）：

| 检查 | rel_l2 | ulp1 |
|---|---:|---:|
| HC norm / mix / combine（attn、mlp） | ≤ 8.5e-5 / 8.5e-5 | 1.000 / 1.000 |
| GDN qkvz / ba / conv 状态 / 递推输出 | ≤ 5.3e-5 / 2.9e-5 | 1.000 / 1.000 |
| GDN ssm 状态 | 2.5e-7 / 4.5e-7 | 1.000 / 1.000 |
| GDN 输出投影 | 4.1e-3 / 9.1e-5 | 0.987 / 1.000 |
| 路由 logits / 选择 | 1.1e-4 / 1.0e-4；0 行不同 | 1.000 / 1.000 |
| MoE | 5.0e-3 / 5.2e-3 | 0.761 / 0.756 |
| 链式：attention 块 / MoE / 层输出 | 6.3e-3 / 1.0e-2 / 6.3e-3；4.0e-3 / 9.5e-3 / 4.9e-3 | — |
| 链式：路由 | 39 行 0 行不同；39 行 1 行不同（概率差 3.1%） | — |
| 链式：conv / ssm 状态 | 1.5e-5 / 4.1e-5；2.9e-5 / 4.8e-5 | — |

与 golden 一样，TP2 输出投影的 4e-3 来自 SGLang 侧 cktile GEMM 的 BF16 截断，MoE 的约 0.5% 来自 AITER 的逐专家 BF16 舍入与 BF16 原子累加；内核与 golden 的差距都在 1 个 BF16 ulp 量级。

## 3. 集成方法

采集不修改 SGLang 源码，通过 SGLang 通用插件接入：

- `capture_plugin/` 里的 dist-info 注册入口点 `sglang.srt.plugins: qwen38_capture`。`capture.py` 把该目录加入服务的 PYTHONPATH，并设置 `SGLANG_PLUGINS=qwen38_capture`（白名单）和 `QWEN38_CAPTURE_DIR`（不设置则插件不生效）。
- 插件用 `HookRegistry` 的 AROUND hook 包住 14 个方法：decode 层 forward、HC mix/combine、GDN forward/输入投影/线性注意力、attention 的 self_attention/QKV 准备/QSA topk/RadixAttention，以及 MoE 的 forward/共享专家追加/路由专家。每个 decode 步在 CPU 上复制一份，写成 `tp{T}_rank{R}_step{S}.pt`。
- pool 和 backend 通过 `sglang.srt.model_executor.forward_context` 取得：`ForwardBatch` 已不再带 `req_to_token_pool`/`token_to_kv_pool`；GDN 槽位取自 `get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices`。
- 必须 eager 运行：CUDA graph 重放时不会调用被包装的 Python 方法。

golden 跟随的 SGLang/AITER 数值语义：MRoPE 按 BF16 逐运算舍入；稀疏注意力按 32 个 key 一块做在线 softmax；FP8 线性层逐 token 量化，`scale = amax·f32(1/240)`，`q = x·rcp(scale)`；路由并列时取较小 id；MoE decode 为 W8A16（BF16 激活、BF16 中间量）。AITER 的 BF16 截断和 BF16 原子累加不模仿，golden 用就近舍入和 FP32 累加。

MonoKernel 内核实现后，按同一插件机制在 decode 时替换整个 decoder layer forward，并保留原生路径作回退（P4），届时在此补充接入步骤。

## 4. 性能

暂无：MonoKernel 内核尚未实现。从 P1 起，在本节记录 TP2/TP4 下各 S 的 decode 层耗时，并与 SGLang 原生实现对比。
