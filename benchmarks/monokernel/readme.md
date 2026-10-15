# Qwen3.8 MonoKernel：使用说明与最终结果

MonoKernel 在 SGLang 服务中把 Qwen3.8-Flash-Next（`qwen4_exp`）的 decode 层换成每层一次常驻 kernel 启动。一次启动内完成以下全部计算：attention 分支的 HC mix、mixer（GDN，或 full attention + QSA indexer）、两次 TP all-reduce、MLP 分支的 HC mix、MoE、层末 combine，以及下一层的 HC norm。本页只维护当前使用说明、最终有效结果和验收状态；修改与优化过程追加到[源码readme](../../src/pyhip/ops/monokernel/qwen38/readme.md)。

## 当前状态

| 项目 | 状态 |
|---|---|
| 替换范围 | decode 批（`ForwardMode.DECODE`，1–64 行）主模型的第 0、2–47 层，即 35 个 GDN 层和 12 个 full attention 层，每层一次启动 |
| 仍走 SGLang 原生 | 第 1 层（PLE）、prefill/extend、MTP（draft、verify 与 MTP 层）、超过 64 行或带 ReplaySSM 的 decode 批、embedding、lm head、采样 |
| TP | TP2、TP4 已完成服务验证和性能测试。要求 attention TP 等于 TP，不支持 DP attention。TP8 未测 |
| 正确性 | 整层启动与拆分启动逐位一致（TP1/2/4）；服务内逐层对照原生层，差距与原生层自身两次运行之间的差距同级（§6） |
| 性能（中位 ITL） | TP2：C1 13.40→7.17 ms，C8 22.90→13.31 ms；TP4：C1 12.81→6.42 ms，C8 20.99→11.16 ms（§7） |
| 推荐配置 | 同时开启 PyHIP QSA prefill（`SGLANG_USE_PYHIP_QSA=1`）：TTFT 与 PyHIP QSA 相同，ITL 与 MonoKernel 相同；TP4 C8 输出吞吐 298 tok/s（原生 196） |

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 模型几何与 TP 分片 | [config.py](../../src/pyhip/ops/monokernel/qwen38/config.py) |
| checkpoint 读取、与 SGLang 一致的分片、SGLang 服务内格式、MonoKernel 自有的 BF16 预排副本、合成权重 | [weights.py](../../src/pyhip/ops/monokernel/qwen38/weights.py) |
| decode 层 golden（按阶段拆分，含 QSA indexer） | [reference.py](../../src/pyhip/ops/monokernel/qwen38/reference.py) |
| 阶段内核（FlyDSL 0.3.2）：量化与线性层、HC、GDN、MoE | [flydsl/](../../src/pyhip/ops/monokernel/qwen38/flydsl/) 的 linear/hc/gdn/moe.py |
| 阶段版整层驱动（每阶段一次启动） | [flydsl/layer.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/layer.py) |
| 常驻启动：阶段函数（含 attention 与 indexer）、GEMV 轮转调度、整层 kernel 与宿主接口 | [phases.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/phases.py)、[attention.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/attention.py)、[_phase.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/_phase.py)、[fused.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/fused.py) |
| grid barrier、启动内 all-reduce、跨卡共享显存 | [sync.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/sync.py)、[xgmi.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/xgmi.py)、[peer.py](../../src/pyhip/ops/monokernel/qwen38/peer.py) |
| SGLang decode hook 与插件入口 | [sglang_plugin.py](../../src/pyhip/ops/monokernel/qwen38/sglang_plugin.py)、[plugin](plugin/qwen38_monokernel_plugin.py) |
| 单元测试（golden、权重格式、阶段内核） | [tests/ops/monokernel](../../tests/ops/monokernel/) |
| SGLang 解码采集（自启服务 + 插件），采完自动检查 | [capture.py](capture.py)、[capture_plugin](capture_plugin/qwen38_capture_plugin.py) |
| golden 与采集逐阶段比对、阈值 | [check_reference.py](check_reference.py) |
| 阶段内核与采集比对（逐阶段 + 整层链式） | [check_kernels.py](check_kernels.py) |
| 整层启动与拆分启动逐位比对（多卡） | [check_layer.py](check_layer.py) |
| 单层 GPU 时间 | [bench_layer.py](bench_layer.py) |
| 自启服务：逐层校验与性能测试 | [serve.py](serve.py) |

目标为 **ROCm gfx942（MI308X）**。验证环境：Python3.10.12、Torch2.12/ROCm7.14、FlyDSL0.3.2、SGLang `1d383c5`、AITER `4752a40`。golden 和单元测试只依赖 PyTorch；采集、整层检查与服务另需 SGLang、AITER 和 `/models/Qwen3.8-Flash-Next-PTPC-FP8`。AITER 固定 `flydsl==0.3.2`，不要在同一环境安装 FlyDSL main：它会让 AITER 的 FlyDSL MoE（生产 decode MoE）无法导入或编译。

本机示例统一使用 PyHIP 的 .venv 解释器。新数据目录在 mytest/mydata 下且必须尚不存在。修改 PyHIP 中的 FlyDSL 代码后，测试前设置 `export FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`：FlyDSL 0.3.2 的磁盘缓存键只含依赖源码、标量闭包值和 compile hint，只改辅助函数时会取回旧 kernel。

## 1. 单元测试

```bash
.venv/bin/python -m pytest -q tests/ops/monokernel
```

使用 16 个专家的合成权重，覆盖 TP2/4/8：各 rank 的 GDN 投影、conv、递推和状态，以及 attention 的 Q/K/V/gate 和稀疏注意力，都与 TP1 结果的对应切片一致（容差 1 个 BF16 ulp）；负槽位输出 0 且不更新状态；整层与 TP1 接近（各 rank 独立做 FP8 再量化，不会逐位相同）；路由并列时选较小的 expert id；HC mix 的近 tie 解析只翻转最接近 tie 的那一个低秩激活。权重格式：16x16 shuffle 与 AITER `shuffle_weight` 逐位相同，E4M3FN→E4M3FNUZ 反量化值不变，TP2/TP4 下 `sglang_format` 覆盖全部 SGLang 参数，padding 行为 0。

阶段内核测试（S=1/5/16/33/64）用 SGLang 服务内格式的权重：逐 token FP8 量化与 golden 逐位相同；W8A8/BF16 线性层、HC mix 与 combine+norm、qkvz+conv（含 conv 状态）、delta rule（含 SSM 状态）与门控 norm、路由 top-k 与分组计划、W8A16 MoE 均与 golden 在 1 个 BF16 ulp 内（判据：≥99.9% 元素 ≤1 ulp、rel_l2 ≤ 1e-3；MoE 中间量与输出放宽到 99.5%）。MoE 用完整 512+1 专家，覆盖 TP2（中间维 320）与 TP4（160 补到 192）。

## 2. golden 与阶段内核：与 SGLang 对齐（离线采集）

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

## 3. 采集插件与 golden 数值语义

采集不修改 SGLang 源码，通过 SGLang 通用插件接入：

- `capture_plugin/` 里的 dist-info 注册入口点 `sglang.srt.plugins: qwen38_capture`。`capture.py` 把该目录加入服务的 PYTHONPATH，并设置 `SGLANG_PLUGINS=qwen38_capture`（白名单）和 `QWEN38_CAPTURE_DIR`（不设置则插件不生效）。
- 插件用 `HookRegistry` 的 AROUND hook 包住 14 个方法：decode 层 forward、HC mix/combine、GDN forward/输入投影/线性注意力、attention 的 self_attention/QKV 准备/QSA topk/RadixAttention，以及 MoE 的 forward/共享专家追加/路由专家。每个 decode 步在 CPU 上复制一份，写成 `tp{T}_rank{R}_step{S}.pt`。
- pool 和 backend 通过 `sglang.srt.model_executor.forward_context` 取得：`ForwardBatch` 已不再带 `req_to_token_pool`/`token_to_kv_pool`；GDN 槽位取自 `get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices`。
- 必须 eager 运行：CUDA graph 重放时不会调用被包装的 Python 方法。

golden 跟随的 SGLang/AITER 数值语义：MRoPE 按 BF16 逐运算舍入；稀疏注意力按 32 个 key 一块做在线 softmax；FP8 线性层逐 token 量化，`scale = amax·f32(1/240)`，`q = x·rcp(scale)`；路由并列时取较小 id；MoE decode 为 W8A16（BF16 激活、BF16 中间量）。AITER 的 BF16 截断和 BF16 原子累加不模仿，golden 用就近舍入和 FP32 累加。

## 4. 整层启动

### 结构

每层一次常驻启动：每个 CU 一个 512 线程 CTA（MI308X 为 80 个），按固定顺序执行一串阶段，阶段之间用 grid barrier 同步（[sync.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/sync.py)）。gfx942 上普通存储在同一启动内对其它 XCD 不可见，所以跨 CTA 传递的中间量用 sc1 写穿存储写出、sc1 加载读入。KV cache 与 index logits 写穿后由其它 CTA 用普通加载读取，这依赖一个前提：同一启动内，这些地址被写之前不会被其它 XCD 读进 L2。各阶段的数值与阶段内核相同（§1、§2）。

| 层 | 阶段（输入的 attention 分支 norm 由上一层给出） |
|---|---|
| GDN（18 个） | HC down、lowrank、mix up；激活量化 + ba 投影；qkvz 投影 + causal conv；delta rule + 门控 norm；量化；out_proj（推送 all-reduce 分片）；MLP 分支 combine + norm；HC down、lowrank、mix up；路由；top-k；分组计划；MoE up；MoE down（推送分片）；combine + 下一层 norm |
| full attention（24 个） | HC down、lowrank、mix up；激活量化 + index q/k 投影；qkv 投影；q/k norm、MRoPE、KV 写入，以及 index q 的 norm 与 RoPE、index key 写入 pending ring；完成组的 4:1 压缩；分页 index logits；CTA top-k 与展开；两遍稀疏注意力（scores、values）；合并与门控；量化；o_proj（推送分片）；其后 10 个阶段同 GDN 层 |

- GEMV 阶段按静态轮次调度：下一轮的权重在本轮计算前发出，第 0 轮的权重在 grid barrier 的到达与等待之间发出；单 split、单 token 块时激活每阶段只读一次。
- 上一层的 combine 同时算出下一层 attention 分支的 norm，SGLang 把它作为 `residual` 传给下一层；本层直接读取，省去 norm 阶段。没有该输入时（如第 0 层）启动内先做 norm。
- 指针参数在使用处才从 kernarg 段加载（`KernelArgs`）。AMDGPU 后端会在入口把全部 kernel 参数装进 SGPR，整层 kernel 的参数多，SGPR 因此溢出 647 处；改为按需加载后，attention 层降到 376 处、GDN 层降到 165 处。

### 启动内 all-reduce

两次 all-reduce 在启动内经 XGMI 完成（[xgmi.py](../../src/pyhip/ops/monokernel/qwen38/flydsl/xgmi.py)）：

- 每个 rank 用 `hipExtMallocWithFlags(hipDeviceMallocUncached)` 分配一块 uncached 显存（[peer.py](../../src/pyhip/ops/monokernel/qwen38/peer.py)），IPC 句柄经 TP 的 CPU group 交换，每个 rank 都映射所有 rank 的区域。
- 产出阶段（out_proj、MoE down）把本 rank 的每个 BF16 输出块写进所有 rank 的 `recv[site][rank]`（系统一致性存储）。阶段末每个 CTA 等自己的存储确认后，与各 peer 的同号 CTA 握手（每 CTA 一个标志和 epoch，按 2^32 取模比较）。握手后的 grid barrier 保证所有分片都已到达本地。
- 消费阶段从本地区域按 rank 顺序读分片，FP32 求和后截断到 BF16，与 AITER one-stage all-reduce 逐位相同。两次 all-reduce 交替使用两个 site，不需要清零。
- 一次握手约 2.4 µs；64 行分片的推送约多 6.6 µs（约 50 GB/s）。

## 5. 服务接入

通过 SGLang 插件接入，不修改 SGLang 源码：

- [plugin](plugin/) 里的 dist-info 注册入口点 `sglang.srt.plugins: qwen38_monokernel`，入口只导入 `pyhip.ops.monokernel.qwen38.sglang_plugin.register`。
- 启用方法：在服务环境中设置 `PYTHONPATH=<pyhip>/benchmarks/monokernel/plugin:<pyhip>/src`、`SGLANG_PLUGINS=qwen38_monokernel`、`PYHIP_MONOKERNEL=1`。如果 .venv 中的 pyhip 已是 editable 安装，可以不加 `src`。
- 插件用 AROUND hook 包住 `Qwen4ExpLinearDecoderLayer.forward` 与 `Qwen4ExpAttentionDecoderLayer.forward`。不在替换范围内的调用（见“当前状态”）原样交给原生实现。
- 权重直接引用 SGLang 加载后的参数。另外生成每层的 BF16 预排副本（HC、路由与共享专家门、ba、indexer），每 rank 约 1.5 GB，在首次调用时分配，不计入 SGLang 的显存规划（KV 容量与原生相同）。
- 启动可被 CUDA graph 捕获。kernel 在 SGLang 的 eager warmup 中编译，服务就绪时间比原生多约 40 s。

| 环境变量 | 作用 |
|---|---|
| `PYHIP_MONOKERNEL=1` | 启用 hook；未设置时插件不做任何事 |
| `PYHIP_MONOKERNEL_VALIDATE=1` | 只作用于 eager 批：每次层调用都在同一输入和 GDN 状态上再跑一次原生层，日志 `MonoKernel validate {...}` 记录逐层指标，保留 MonoKernel 的结果 |
| `PYHIP_MONOKERNEL_AR=aiter` | 每层在两次 all-reduce 处拆开，使用 SGLang（AITER）的 all-reduce |
| `PYHIP_MONOKERNEL_ATTENTION=native` | full attention 层只融合 HC mix，attention 本身走 SGLang 原生实现（拆分路径） |
| `SGLANG_USE_PYHIP_QSA=1` | SGLang 自带开关：prefill 的 QSA attention 与 indexer 走 PyHIP。与 MonoKernel 同时开启即推荐配置 |

attention TP 与 TP 不等时（DP attention）自动退到拆分路径，该情形未测试。

[serve.py](serve.py) 自启一个服务，参数与用户启动脚本相同：AITER attention/MoE、chunked prefill 16384、关闭 radix cache、最多 32 个并发请求、decode graph 到 32、不卸载 PLE embedding。可见卡由 `--devices` 指定：

```bash
.venv/bin/python benchmarks/monokernel/serve.py --arm mono --mode validate --tp 2 --devices 0,1
.venv/bin/python benchmarks/monokernel/serve.py --arm mono_pyhip --mode bench --tp 4 --devices 4,5,6,7
```

- `--arm`：`native`（原生）、`pyhip`（`SGLANG_USE_PYHIP_QSA=1`）、`mono`（MonoKernel）、`mono_pyhip`（两者同开）。
- 非原生臂先把 `src/pyhip` 与插件复制到运行目录的 `source/`，再从那里加载，运行中修改代码不会影响服务；每次运行使用新的 FlyDSL 缓存。
- 输出目录（省略 `--output` 时在 mytest/mydata 下新建）包含 `manifest.json`（命令、环境、git 状态、前后 GPU 状态、结果）和 `server.log`。validate 模式另有 `outputs.json`，bench 模式另有 `bench_c{1,2,4,8}.jsonl/.log`。
- 服务就绪后先发一个冒烟请求（要求只回答一个数字），答对才继续。

## 6. 正确性验收

### 整层启动与拆分启动逐位一致

```bash
.venv/bin/python benchmarks/monokernel/check_layer.py --gpus 0,1 --rows 1 5 16 33 64
```

每个 GPU 一个进程，GPU 数即 TP。使用真实权重（第 0 层 GDN、第 3 层 full attention）和合成的输入、GDN 状态、分页 KV cache 与 QSA 状态。比较的两条路径：

- 整层启动，all-reduce 在启动内。
- 拆分启动，加上精确 all-reduce（按 rank 顺序 FP32 求和、截断到 BF16）和单独的 combine。

比较对象为层输出、下一层 norm、conv/SSM 状态、KV cache、index-key ring、ring 坐标和压缩 key。每个行数分别在给出和不给出输入 norm 时各跑一次。2026-10-13（`eb9615d`）在 TP1（GPU4）、TP2（GPU4/7）、TP4（GPU4–7）上全部逐位相同。拆分启动各阶段与 golden、采集的对齐见 §1、§2。

### 服务内逐层对照原生

`serve.py --mode validate`：eager 服务，`PYHIP_MONOKERNEL_VALIDATE=1`。用真实文本 prompt：先发 1 个 900 token 的单请求，再发 4 个请求的批（900/2100/3600/3000 token），每个请求生成 64 个贪心 token。对每次层调用，在同一输入和状态上再跑原生层，计算 MonoKernel 输出与原生输出的 rel_l2。

- 原生层在同一输入上执行两次的差（`native_repeat`）作为原生自身的噪声基线：HC mix 与 MoE 的原子累加使原生结果不确定。
- 路由与原生不同的行单独统计；`out_same_route` 只比较路由相同的行。路由不同的行来自近似并列的专家分数，这些行的输出差距更大（全部行的 p99 约 0.02）。

2026-10-13，`eef32ec`，全部层调用、全部 rank 的统计：

| 运行 | 层 | 调用数 | 路由不同的行 | out_same_route 中位 / p99 | native_repeat 中位 / p99 |
|---|---|---:|---:|---|---|
| TP2 MonoKernel | attention | 3312 | 2.58% | 0.00198 / 0.00828 | 0.00165 / 0.0147 |
| | GDN | 9660 | 1.47% | 0.00191 / 0.00451 | 0.00159 / 0.00299 |
| TP4 MonoKernel | attention | 6624 | 1.89% | 0.00202 / 0.00847 | 0.00177 / 0.0123 |
| | GDN | 19320 | 0.09% | 0.00164 / 0.00337 | 0.00159 / 0.00299 |
| TP2 MonoKernel + PyHIP QSA prefill | attention | 3312 | 3.16% | 0.00207 / 0.00881 | 0.00169 / 0.0187 |
| | GDN | 9660 | 1.83% | 0.00188 / 0.00446 | 0.00153 / 0.00288 |
| TP4 MonoKernel + PyHIP QSA prefill | attention | 6624 | 1.99% | 0.00200 / 0.00871 | 0.00176 / 0.0187 |
| | GDN | 19320 | 0.10% | 0.00167 / 0.00339 | 0.00163 / 0.00307 |

GDN 的 conv 状态 rel_l2 中位为 0、SSM 状态中位约 1.3e-7。所有运行的冒烟请求都答对，全部请求完成。

不同运行生成的 token 序列会在前几个 token 内分叉，原生 prefill 的两次运行之间也是如此。原因是贪心解码在近似并列处翻转，不作为正确性判据；上面的逐层指标才是判据。没有测模型质量指标（如 GSM8K）。

## 7. 性能

### 服务

`serve.py --mode bench`：CUDA graph，关闭校验。`sglang.bench_serving` 用 random 数据，每档 32 个请求，输入 12000、输出 350 token（range ratio 1.0），并发 1/2/4/8 依次运行。2026-10-13，MI308X；TP2 用 GPU0/1，TP4 用 GPU4–7；MonoKernel 代码为 `eef32ec`（组合臂为 `096b438`，内核相同）。

中位 ITL（ms）：

| 臂 | TP2 C1 | C2 | C4 | C8 | TP4 C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 | 13.40 | 15.28 | 18.92 | 22.90 | 12.81 | 14.45 | 16.60 | 20.99 |
| PyHIP QSA | 10.86 | 11.86 | 14.35 | 15.44 | 10.34 | 11.16 | 12.02 | 13.60 |
| MonoKernel | 7.17 | 8.14 | 10.07 | 13.31 | 6.42 | 7.30 | 8.58 | 11.16 |
| MonoKernel + PyHIP QSA prefill | 7.21 | 8.21 | 10.19 | 13.45 | 6.39 | 7.26 | 8.51 | 11.04 |

中位 TTFT（ms）：

| 臂 | TP2 C1 | C2 | C4 | C8 | TP4 C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 | 1051 | 2055 | 4230 | 6852 | 846 | 1645 | 3399 | 5519 |
| PyHIP QSA | 899 | 1426 | 3177 | 4771 | 674 | 1091 | 2699 | 3668 |
| MonoKernel | 1048 | 2052 | 4226 | 6841 | 841 | 1634 | 3400 | 5518 |
| MonoKernel + PyHIP QSA prefill | 898 | 1462 | 3176 | 4772 | 676 | 1094 | 2707 | 3683 |

输出吞吐（tok/s，含 prefill 时间）：

| 臂 | TP2 C1 | C2 | C4 | C8 | TP4 C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 | 61.0 | 94.6 | 127.0 | 168.8 | 65.7 | 104.2 | 145.5 | 195.9 |
| PyHIP QSA | 74.5 | 118.6 | 160.0 | 222.5 | 81.6 | 134.3 | 194.1 | 273.1 |
| MonoKernel | 98.5 | 142.7 | 176.5 | 211.6 | 113.6 | 166.8 | 208.6 | 258.1 |
| MonoKernel + PyHIP QSA prefill | 101.8 | 151.4 | 191.9 | 235.2 | 120.4 | 181.7 | 231.1 | 298.4 |

- MonoKernel 只替换 decode：TTFT 与原生相同。中位 ITL 比原生低 46%/50%（TP2/TP4，C1）和 42%/47%（C8）。
- PyHIP QSA 只替换 QSA attention 与 indexer，prefill 与 decode 都受益；组合臂的 decode 层由 MonoKernel 完成，因此 TTFT 取 PyHIP、ITL 取 MonoKernel。
- 12000 token 输入下，C8 的吞吐主要由 prefill 决定，单开 MonoKernel 在 C8 吞吐低于 PyHIP QSA；组合臂在所有档位最高。

### 单层 GPU 时间

```bash
.venv/bin/python benchmarks/monokernel/bench_layer.py --impl staged fused layer --tp 2 4 --rows 1 4 16 64
```

测量方式：

- 单 GPU，只算一个 rank 的分片；6 个真实 GDN 层轮换（权重超出 256 MiB MALL，每层都从 HBM 读权重），10 组输入。
- 一个 CUDA graph 含 10 次层调用，128 次重放取中位。
- 不含跨卡 all-reduce。`staged` 为每阶段一次启动；`fused` 为每层两次常驻启动；`layer` 为服务使用的一次启动整层，用单 rank 通信区，含分片的推送与读取，但没有跨卡等待。

2026-10-13，GPU5/6（`eb9615d`），µs/层：

| 实现 | TP2 S=1 | S=4 | S=16 | S=64 | TP4 S=1 | S=4 | S=16 | S=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| staged | 122.7 | 174.9 | 337.9 | 1600 | 110.1 | 138.7 | 260.8 | 1012 |
| fused | 117.7 | 175.7 | 360.0 | 2207 | 99.3 | 144.4 | 280.5 | 1383 |
| layer | 126.1 | 180.0 | 354.3 | 2301 | 105.4 | 148.9 | 272.3 | 1446 |

- S=1 时一个 GDN 层每 rank 读取约 80 MiB（TP2）/ 58 MiB（TP4）权重，按 4 TB/s 计只需约 21/15 µs。实测是它的 6–7 倍：约 20 个相互依赖的阶段，每个阶段都受延迟限制。
- 在 CUDA graph 下，一次 grid barrier（约 1.5–2 µs）与一个 kernel 边界的代价相近，因此单次启动本身几乎不带来收益。在服务中，整层单次启动（含启动内 all-reduce、省去输入 norm）比“拆分启动 + AITER all-reduce”的 ITL 只低约 1%（TP2 C1 7.22→7.17 ms，TP4 C1 6.48→6.42 ms）。相对原生的收益主要来自各阶段内核本身：qkvz 投影内的 conv、W8A16 MoE 与分组计划、HC、两遍稀疏注意力与 indexer。
- S≥16 时常驻启动慢于阶段内核：MoE 阶段只有 80 个 CTA，而合成输入几乎激活全部专家，阶段内核则按专家数展开网格。服务测试的并发不超过 8，每步最多 8 行。

## 8. 限制与注意事项

- 只覆盖 decode；每步最多 64 行（用户启动脚本的 decode graph 最多 32 行）。MTP 保持原生。
- 只测了 TP2、TP4。分片代码支持 TP8，但未验证。
- 依赖 XGMI 对等访问与 HIP IPC；所有 rank 必须以相同的网格运行同一组启动（SGLang 的 TP rank 满足这一条件）。
- S 较大（≥16）时整层启动慢于阶段内核，见 §7；需要大批 decode 时应先优化 MoE 阶段，或按行数切换实现。
- 每 rank 额外占用约 1.5 GB 预排副本，见 §5。
