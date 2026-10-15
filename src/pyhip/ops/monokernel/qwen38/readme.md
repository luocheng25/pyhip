# Qwen3.8-Flash-Next MonoKernel（gfx942）修改与优化记录

只追加，不改写已有条目。当前用法与最终结果见 [benchmarks/monokernel/readme.md](../../../../../benchmarks/monokernel/readme.md)。从 SGLang 接口到 kernel 的调用链伪代码见文末“调用链伪代码”一节。

## 2026-10-13 P0：golden、TP 分片与 SGLang 对齐

### 内容

- `config.py`：48 层（36 GDN、12 full attention+QSA）、HC 4x2560、MoE 512 路由专家+1 共享（top10）；TP1/2/4/8 分片。
- `weights.py`：checkpoint 读取；与 SGLang 一致的分片（GDN q/k/v/z、attention 头、专家中间维连续切分，TP>2 复制 KV 头）；合成权重。
- `reference.py`：单 rank decode 层 golden，按阶段拆分（HC norm/mix/combine、GDN proj/core/out、attention proj/core/out、路由、MoE、all-reduce、整层）。QSA indexer 不在其中（按计划 P3），attention 的 topk 由调用方给出。
- 单元测试 `tests/ops/monokernel/test_qwen38_reference.py`（13 项）；集成检查 `benchmarks/monokernel/`。

### 由 SGLang/AITER 实际内核确定并写入 golden 的数值语义

离线运行 SGLang 自身内核或读 AITER 源码核实：

1. MRoPE：BF16 表下逐运算舍入，`bf(bf(a*cos) - bf(b*sin))`。FP32 一次舍入版有 5.6% 元素差 1 ulp。
2. QSA decode 稀疏注意力：选中 key 按 32 个一块在线 softmax，P 在当前最大值下转 BF16。全局 softmax 版 rel 9e-4，分块后 1.3e-4。
3. FP8 线性层逐 token 量化同 AITER：`scale = amax*f32(1/240)`，`q = x*rcp(scale)`。用除法时个别行 FP8 码翻转（TP4 o_proj 一行 rel 4e-3）。
4. 路由：BF16 logits 常有并列，取较小 expert id（稳定排序）；修正后逐阶段路由 0 行不一致。
5. MoE decode：AITER FlyDSL batch1（调优表 token 1/4/8、inter 320/192 都走此路径）是 W8A16：激活保持 BF16，FP8 权重反量化，SiLU 门控中间量为 BF16。最初按 W8A8（激活逐 token FP8）写，对生产内核 moe.partial rel 4.57e-2；改后 5.08e-3。

### 未模仿的差异（golden 保持就近舍入和 FP32 累加）

- AITER all-reduce 与调优表 (M=4, N=2560, K=3072) 的 cktile GEMM 截断到 BF16，差不超过 1 ulp。
- AITER MoE 每个专家输出加 0x8000 舍入到 BF16，再用 BF16 原子加累加。逐专家最小二乘比例系数都是 1±0.1%，剩余约 0.5% 是这部分舍入噪声；模拟截断或固定顺序 BF16 累加反而更差。
- SGLang `fused_hc_mix` 用原子累加，同一步各 rank 的 `hc_mlp.out`、logits 偶尔不同。
- Triton sigmoid/rsqrt 偶有 1 ulp 差，经 FP8 再量化会翻转个别码（TP4 gdn.partial 一行）。

### 结果（check_reference：4 层视图，3 步 S=4 + 1 步 S=1）

- TP2，生产 FlyDSL MoE（`qwen38_mono_p0_20261013_03`）：PASS。逐阶段除 MoE 外 1 BF16 ulp 内比例为 1.000；moe.partial rel 5.1e-3；整层 out rel 6.3e-3、MoE 1.15e-2，路由 1/52 行不同且为近似并列（概率差 3.1%）。
- TP4（`qwen38_mono_p0_20261013_01`）：采集时 FlyDSL main 与 AITER 不兼容，MoE 为 AITER 回退的 1-stage asm 内核（W8A8），非 MoE 阶段同样在 1 ulp 内；MoE 与整层检查对最终 golden 不适用。GPU2-7 被其它会话占用，生产 MoE 的 TP4 尚未重采。
- 合成权重的 TP2/4/8 分片与 TP1 一致（单元测试）。

### 数据

- `mytest/mydata/qwen38_mono_p0_20261013_01`：首轮 TP2/TP4 采集；`source/` 冻结了当时的 FlyDSL 版代码、采集脚本和探针。
- `mytest/mydata/qwen38_mono_p0_20261013_03`：TP2 生产 MoE 采集；`check_reference.json` 为 W8A8 golden（FAIL，据此发现问题），`check_reference_moe_w8a16.json` 为最终 golden（PASS）。
- `mytest/mydata/qwen38_mono_p0_20261013_02_failed_relative_model_path`：驱动把相对模型路径交给服务导致启动失败，已修正。
- 探针脚本：`mytest/qwen38_mono_probe_{kernels,gemm,moe,tp4_proj,quant,attention}.py`。

### 环境

- 代码最初写在 `/opt/lc/FlyDSL/kernels/monokernel/qwen38`（未提交），10-13 迁入 pyhip，FlyDSL 副本与 README 改动已删除。
- venv 的 FlyDSL 从 editable 0.4.0.dev959（`/opt/lc/FlyDSL`，与 AITER 4752a408 不兼容）恢复为 AITER 固定的 0.3.2：由系统 pip 安装重建 wheel，147 个载荷文件 sha256 一致，pip check 无冲突；pyhip QSA 83 项测试通过，AITER FlyDSL MoE 编译和运行正常。记录在 `mytest/mydata/flydsl_032_restore_20261013_01`。

## 2026-10-13 P0 补充：TP4 生产 MoE、HC mix 近 tie、SGLang 服务内权重格式

### TP4 生产 MoE 重采

- GPU4-7 空出后重采（`qwen38_mono_p0_20261013_04` 为 TP2，`_05` 为 TP4）。TP4 MoE（中间维 160 补到 192）rel 5.3e-3，与 TP2 同级。
- 首次 TP2→TP4 连续采集时 TP4 入口端口探测失败：上一服务留下 TIME_WAIT，探测 socket 未设 `SO_REUSEADDR`。已修正。

### HC mix 近 tie（TP4 step3 layer1 hc_mlp.mix：exact 0.975、ulp1 0.998，超出 0.999）

- 离线对同一输入重复运行生产 `fused_hc_mix` 200 次，输出完全相同且逐位复现采集值，不是运行间不确定性。
- 逐个把 golden 的 BF16 低秩激活翻转 1 ulp，只有 r=289 一处翻转使 golden 与生产逐位一致；该激活的 FP32 值距 BF16 tie 1.8 个 FP32 ulp。生产内核的 FP32 累加顺序（原子累加 + `tl.dot`）与 torch 不同，把它舍入到另一侧，整行输出随之移动。
- 处理：`reference.resolve_hc_mix_ties` 对距 tie ≤ 32 个 FP32 ulp 的激活尝试另一侧舍入（每行最多 2 个，只保留能减少不一致的翻转），check_reference 的 HC mix 逐阶段检查和链式 hc_attn_out 都用它，并报告 `tie_flips`/`tie_ulps`。之后的采集中最远一次为 30.9 ulp；窗口按相对 ulp 计，激活很小时累加误差相对更大，若以后出现 >32 的合理翻转需改为按 `sum|x·w|` 的绝对窗口。
- 探针：`mytest/qwen38_mono_probe_hcmix.py`。

### SGLang 服务内权重格式（MonoKernel 直接读取，不另存权重）

- checkpoint 为 compressed-tensors（权重 per-channel、激活 per-token 动态），走 `CompressedTensorsW8A8Fp8` / `...Fp8MoEMethod`：`normalize_e4m3fn_to_e4m3fnuz`（同位读作 FNUZ、scale×2、0x80→0），稠密层与专家都 `shuffle_weight(w, (16, 16))`；专家中间维按 `AITER_MOE_PADDING_SIZE=64` 每半补零；BF16 权重保持 checkpoint 布局。
- `weights.py` 新增 `to_fnuz`/`shuffle_16x16`/`unshuffle_16x16`/`moe_inter_padded`/`sglang_format` 和 `SGLANG_NAMES`（与 SGLang 参数名对应）。shuffle 与 AITER `shuffle_weight` 逐位相同（单元测试），fnuz 反量化值不变。
- 采集插件第一步写出各层参数/状态池布局（`layout.json`）与每个参数 4096 点原始位采样（`param_samples.pt`）；check_reference 的 served_format 用 `sglang_format(checkpoint)` 逐位比对。`qwen38_mono_p0_20261013_06` 中 TP2/TP4 每 rank 的 layer0/layer3 各 23 个张量 0 处不同。SGLang 另有派生 buffer（`q/k_norm.gemma_weight`、indexer norm 的 `gemma_weight`、rotary `cos_sin_cache`）不在其中。
- 状态池：mamba conv `[33, conv_dim, 3]` BF16 连续，temporal `[33, HV, 128, 128]` FP32；KV `[token 槽位, KV 头, 256]` BF16；`req_to_token` int32 `[33, 262148]`；QSA pool 另有 key_state/compressed_k/rope_position 缓冲（P3 再用）。

### 结果（`qwen38_mono_p0_20261013_06`，check_reference 含 served_format）

- TP2 PASS、TP4 PASS；最坏值见 [benchmarks/monokernel/readme.md](../../../../../benchmarks/monokernel/readme.md)。TP2 链式 attention 块 rel 1.5e-2（S=4 cktile GEMM 截断经 FP8 再量化放大），TP4 链式路由 2/52 行不同、概率差 ≤3.1%。
- 单元测试 21 项（新增 HC mix 近 tie 1 项、权重格式 7 项）。
- `_06` 首次运行显示 complete=False：完整性检查只认 step 文件，新增的 layout/param_samples 被当成多余文件；已改为同时要求这两类文件，采集数据本身完整，检查直接在其上运行。

### P4 跳过的层

- 用户要求 P4 第一步跳过“第一层（有 PLE）”。PLE 权重在 `model.language_model.layers.1.ple.*`（`config.ple_layer_ids=[2]` 按 1 起算），即 layer_id 1；layer 0 只需把 embedding 复制到 4 个分支（SGLang 在 `_prepare_qwen4_exp_attn` 中做），可正常替换。

## 2026-10-13 P1-1：阶段版 GDN 层内核（每阶段一次启动）

### 内核

位于 `src/pyhip/ops/monokernel/qwen38/flydsl/`，FlyDSL 0.3.2 低层写法：raw buffer 描述符 + MFMA intrinsic + LDS 地址。

- `linear.py`：`quant_rows`（AITER 公式，逐位）；`Gemm`（16 行权重块 × 最多 64 行 token，CTA 内 4 个 wave 分 K，LDS 归约）；`make_linear`：W8A8（直接读 SGLang 的 FP8 shuffle 权重与 scale）、BF16（读 `pack_layer` 的 shuffle 副本）、FP32 split-K。
- `hc.py`：down（split-K=16，附 4 行 inject）→ lowrank（FP32 归约、`bf16(silu(t/4))`、`gain=2·sigmoid`）→ up+mix（up 行重排，使一个 lane 持有同一列的 4 个分支）；combine+norm（gr_write 写法：每 (行, 分支) 一个 5-wave CTA）。
- `gdn.py`：qkvz GEMM 的 epilogue 内做 causal conv（BF16 舍入的抽头乘积、SiLU）并更新 conv 状态，省去单独的 conv 阶段；delta rule 每 (行, v 头) 一个 CTA，线程对 = 状态一行的两个 K 半段，DPP 交换求和；门控 RMSNorm 同 CTA 完成。
- `moe.py`：top-k 与分组计划在一个 1024 线程 CTA 内：每专家 64 位行掩码，路由位置 = 前缀偏移 + popcount（行序确定）；W8A16 up（FP8→BF16 精确转换后 BF16 MFMA，gate/up 成对）；down 每 16 列一个 CTA，4 个 wave 轮流处理活跃专家，按 LDS 每 wave 累加器、固定顺序求和（确定性，无原子）。
- `layer.py`：`StagedLayer` 按两次 all-reduce 分 `attention_phase` / `mlp_phase` / `finish` 三段。

### 验证

- 单元测试 76 项（阶段 40、MoE 10 含 512+1 专家 TP2/TP4、GDN 5 等）。`check_kernels.py` 用真实采集（`qwen38_mono_p0_20261013_06`）：TP2、TP4 均 PASS，各阶段与 SGLang 的差距与 golden 同级（结果表见 [benchmarks/monokernel/readme.md](../../../../../benchmarks/monokernel/readme.md)）。

### 问题与修正

- split-K 的 f32 输出：token 超出 rows 的填充行写入了下一个 split 的区域（descriptor 只防整体越界），改为显式屏蔽。
- `flyc.compile` 本身会执行一次调用，`_call` 编译后又调用一次，conv/ssm 状态被更新两次；改为编译调用即返回。
- FlyDSL 0.3.2 的缓存键只含依赖源码、标量闭包值和 compile hints；`Gemm` 对象里的 K 不在键中，TP4 的 out_proj（K=1536）从磁盘缓存取到了 TP2 的 K=3072 kernel，越界读导致 GPU memory fault（时序相关、加打印即消失，也可能静默算错）。`_call` 现在把完整配置放进 compile hint。
- TP4 的 ba 只有 24 行：BF16 线性层支持输出列数为 4 的倍数（权重补零到 16 行，存储按真实列数屏蔽）。

### 初测（CUDA graph 重放，GPU0，非正式）

qkvz（N=8192, K=2560）S=1 7.5 µs（2.8 TB/s）、S=16 14.9 µs、S=64 47.6 µs。S 大时每个 16 行块都从 L2 重读整份激活，是下一步优化点（多块共用激活、LDS 暂存）。Python 逐次启动开销约 20 µs/次，计时必须用 graph。

## 2026-10-13 P1-2：GDN 层的常驻启动（0646468、b8ff9a1）

### 内容

- `fused.py`：GDN 层的两半（attention 半：norm .. out_proj；MLP 半：combine/norm .. MoE down）各一次常驻启动，每个 CU 一个 512 线程 CTA，阶段函数之间用 grid barrier（`sync.py`）。跨 CTA 数据用 sc1 存取，不依赖 L2 写回。
- `_phase.py` 的 `RoundGemv`：GEMV 阶段按静态轮次调度，下一轮权重在本轮计算前发出，第 0 轮权重在 barrier 的 arrive 与 wait 之间发出（split grid barrier）；单 split、单 token 块时激活每阶段只读一次。
- in_proj_ba 挪到激活量化旁；路由与共享专家门合成一块权重（`router_shared`）。
- 递推在 wait 前预取第一个状态、计算时预取下一个，用 packed FP32 FMA，每个向量只算一次倒数（对 golden 验证）。
- `bench_layer.py` 增加 `--impl fused`；可选 `s_memrealtime` 阶段时间线。

### 验证

fused 与 staged 在 TP2/TP4、rows 1–64 逐位相同（探针 `mytest/qwen38_mono_probe_fused.py`）。

## 2026-10-13 P4：SGLang decode 插件与自启服务驱动（5b23d26）

### 内容

- `sglang_plugin.py` + `benchmarks/monokernel/plugin`：AROUND hook 包住两类 decoder layer 的 forward。满足条件时（DECODE、≤64 行、非 PLE 层、非 MTP 层、无 ReplaySSM）走 MonoKernel，否则原样调用原生实现。权重直接引用 SGLang 参数，另加 `pack_layer` 的 BF16 副本。
- `PYHIP_MONOKERNEL_VALIDATE=1`：eager 批在同输入、同 GDN 状态上再跑原生层，记录 out、out_same_route（只比较路由相同的行）、native_repeat（原生两次运行之差，作噪声基线）以及 conv/ssm。
- `serve.py`：自启服务（参数同用户启动脚本，可见卡作参数），validate 与 bench 两种模式。bench 协议同此前 QSA 服务研究：random 32×12000→350，C1/2/4/8。
- 最初只替换 GDN 层（attention 层原生）：TP2 C1 ITL 13.40→12.14 ms。再给 attention 层加上融合的 HC mix 和 MLP 半（attention 本身原生）：11.75 ms。

### 问题

- `mono_bench_tp2_warm_failed_concurrent_edit`：服务运行中编辑了源码，新 import 读到半成品而失败。此后 `serve.py` 把 `src/pyhip` 与插件复制到运行目录的 `source/` 再加载，每次运行用新的 FlyDSL 缓存。

## 2026-10-13 P2：full attention mixer 融合（a2397ff）

### 内容

- attention 层拆成两次常驻启动。prep：HC mix、qkv、q/k norm、MRoPE、KV 写入；core：稀疏注意力、门控、o_proj。QSA indexer 仍用 SGLang 的（在两次启动之间调用）。
- 稀疏注意力最初用单遍在线 softmax，服务校验 attention 层 out_same_route 中位 0.0034（`mono_validate_tp2_p2b`），明显高于原生自身的 0.0016。SGLang decode 先算每块最大值再做两遍；改为同样的两遍并使用相同的块最大值后，中位降到 0.0020（`_p2b_twopass`），复测 0.0019（`_p2b_repeat`）。
- 服务 TP2 C1 ITL 11.75→9.88 ms。

## 2026-10-13 P3：QSA decode indexer 融合（3260819）

### 内容

- `attention.py` 加入 indexer：index q/k 投影（BF16，与激活量化同阶段）、index q 的 Gemma norm 与 RoPE、index key 写入 SGLang 的 pending ring；新 token 补齐一组时做 4:1 压缩写入 compressed 池；对每行全部压缩块算分页 relu logits；CTA 内 top-k（与 SGLang `indexer_topk` 相同的升序输出）并展开为 token 位置。
- prep 末尾产出 index logits，core 开头做 top-k；SGLang 的 indexer 不再调用。
- 服务：TP2 C1–C8 ITL 7.22/8.12/10.07/13.41 ms，TP4 6.48/7.35/8.67/11.27 ms。校验（out_same_route 中位，attention/GDN）：TP2 0.00199/0.0019，TP4 0.00194/0.00168；原生自身 0.00165/0.00156、0.00172/0.00164。

## 2026-10-13 P4b：启动内 all-reduce（e9ac268）

### 设计（`xgmi.py`、`peer.py`）

- 每 rank 一块 `hipExtMallocWithFlags(Uncached)` 显存，IPC 句柄经 TP 的 CPU group 交换（`all_gather_object`），每个 rank 映射所有区域；ctypes 调用当前进程已加载的 libamdhip64（从 `/proc/self/maps` 找），与 torch 共用 context。
- 推送式：产出阶段把本 rank 的每个 BF16 输出块用 `sc0|sc1` 存储写进所有 rank 的 `recv[site][rank]`。阶段末等 vmcnt 归零后做 CTA barrier，wave 0 的各 lane 写 peer 的标志、再等自己的标志（每 CTA 一个 epoch，按 2^32 取模比较）。之后的 grid barrier 保证分片已全部到达。
- 消费阶段按 rank 顺序 FP32 求和并截断到 BF16，与 AITER one-stage all-reduce 逐位相同。site 0/1 在两次 all-reduce 间交替，无需清零。
- 微测（`mytest/qwen38_mono_xgmi_probe.py`、`mytest/qwen38_mono_launch_floor.py`）：0 错；握手约 2.4 µs；推送 64 行约多 6.6 µs（约 50 GB/s）；空常驻启动 1.96 µs；一次 grid barrier 约 1.4–1.8 µs。
- GDN 层一次启动；attention 层为 prep + 其余（core、两次 all-reduce、MLP 半、combine）两次启动。

### 结果与问题

- 服务 TP2 C1–C8 ITL 7.30/8.24/10.18/13.49 ms，反而比 P3（AITER all-reduce）略慢。
- 诊断：AMDGPU 后端在 kernel 入口把全部参数装进 SGPR，整层 kernel 参数多，SGPR 溢出 647 处、v_readlane 1675 条，抵消了省下的 all-reduce 启动。
- 计时探针在一个地址上做约 4 万次原子加，扭曲了计时；改为只在出错数非 0 时才原子加。
- 两种 kernel 同名，ISA dump 目录互相覆盖；改名为 `gdn_layer` / `attn_layer`。

## 2026-10-13 每层一次启动、上一层 norm 作输入、参数按需加载（eef32ec）

### 内容

- `KernelArgs`：Int64 指针参数在使用处从 `llvm.amdgcn.kernarg.segment.ptr + 8·index` 加载，缓冲区在 lambda 里懒构造。SGPR 溢出 647 → 376（attn_layer）/165（gdn_layer）。
- attention 层的 prep 并入同一启动。KV 写入改为写穿（aux），后续阶段在其它 XCD 上用普通加载读取；同一启动内这些地址写前不会被其它 XCD 读进 L2。
- SGLang 传给下一层的 residual 就是上一层 combine 算出的 attention 分支 norm：作为 hn 输入，省去 norm 阶段。
- 服务（TP2 GPU0/1、TP4 GPU4–7，并发运行于不同卡）：TP2 C1–C8 ITL 7.17/8.14/10.07/13.31 ms，TP4 6.42/7.30/8.58/11.16 ms；相对 P3 约低 1%。校验 out_same_route 中位（attention/GDN）：TP2 0.00198/0.00191，TP4 0.00202/0.00164；原生自身 0.00165/0.00159、0.00177/0.00159。
- 逐位探针（`mytest/qwen38_mono_probe_layer_ar.py`，此后固化为 `benchmarks/monokernel/check_layer.py`）：整层启动对“拆分启动 + 精确 all-reduce + 单独 combine”，TP1/TP2/TP4 全部逐位相同。

### 时间线

rows=1、TP2、计时模式，GDN 层约 144 µs。较大的延迟项如下；握手时间计入 out_proj / moe_down。

| 阶段 | 耗时（µs） | 其后 barrier 等待（µs） |
|---|---:|---:|
| quant_ba | 6.5 | 7.9 |
| quant_y | 1.1 | 2.9 |
| mlp_norm | 6.2 | 7.3 |
| topk | 5.5 | 6.4 |
| plan | 2.8 | 3.6 |
| combine | 6.0 | — |

## 2026-10-13 收尾：组合臂、单层计时、整层逐位检查固化（096b438、eb9615d）

### 组合臂

`serve.py` 增加 mono_pyhip 臂（MonoKernel decode + `SGLANG_USE_PYHIP_QSA=1` 的 prefill），校验与单开 MonoKernel 同级：

| 指标 | TP2 C1 | C2 | C4 | C8 | TP4 C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 中位 ITL（ms） | 7.21 | 8.21 | 10.19 | 13.45 | 6.39 | 7.26 | 8.51 | 11.04 |
| 中位 TTFT（ms） | 898 | 1462 | 3176 | 4772 | 676 | 1094 | 2707 | 3683 |

TP4 C8 输出吞吐 298.4 tok/s，高于 PyHIP QSA 的 273.1 和 MonoKernel 的 258.1。PyHIP QSA 臂 TP2 复测 ITL 10.86/11.86/14.35/15.44 ms，与 10-04 的 QSA 记录一致。

### 整层逐位检查

`check_layer.py`：TP1（GPU4）、TP2（GPU4/7）、TP4（GPU4–7）全部逐位相同。

### 单层计时

`bench_layer.py --impl layer`（单 rank 通信区，无跨卡等待），µs/层：

| 实现 | TP2 S=1 | S=4 | S=16 | S=64 | TP4 S=1 | S=4 | S=16 | S=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| staged | 122.7 | 174.9 | 337.9 | 1600 | 110.1 | 138.7 | 260.8 | 1012 |
| fused | 117.7 | 175.7 | 360.0 | 2207 | 99.3 | 144.4 | 280.5 | 1383 |
| layer | 126.1 | 180.0 | 354.3 | 2301 | 105.4 | 148.9 | 272.3 | 1446 |

结论：

- S=1 时 GDN 层每 rank 只读约 80/58 MiB 权重（按 4 TB/s 约 21/15 µs），实测是它的 6–7 倍，约 20 个相互依赖的阶段都受延迟限制。
- CUDA graph 下 grid barrier 与 kernel 边界代价相近，常驻启动本身对 S=1 几乎没有收益。相对原生的收益主要来自阶段内核本身。
- S≥16 时 MoE 阶段只有 80 个 CTA，合成输入激活几乎所有专家，慢于按专家数展开网格的阶段内核。

### 其它

- 生成序列在不同运行间会在前几个 token 分叉，原生 prefill 自身两次运行也是如此（近似并列处的贪心翻转），不作判据。
- 跨 XCD 一致性探针 `mytest/qwen38_mono_xcd_coherence.py` 写好但未运行，当前用法不依赖它所测的情形。

### 未做的优化（估计值，均未实现）

- 激活量化并入 FP8 GEMV（每个 CTA 自算整行 amax，与 quant 阶段逐位相同），ba / index q/k 投影与 qkvz/qkv 同阶段错开 CTA：约 10–15 µs/层。
- top-k 与分组计划合并：约 3 µs/层。
- combine 类阶段在 barrier 等待前预取 h、gain 与 norm 权重：约 4 µs/层。
- 大 S 下按行数切换到阶段内核，或给 MoE 阶段更多并行度。

### 数据

`mytest/mydata/qwen38_mono_p4_20261013_01`：各运行目录的 `manifest.json` 记录代码版本；`src_3260819`、`src_096b438` 为当时用于服务的 git archive 导出。

## 2026-10-14 调用链伪代码：SGLang 接口 → kernel（cd45202）

本节描述服务路径上一个 decode 层从 SGLang 调用到 GPU kernel 的全过程。服务路径指 `PYHIP_MONOKERNEL=1`，`PYHIP_MONOKERNEL_AR` 与 `PYHIP_MONOKERNEL_ATTENTION` 取默认值，`PYHIP_MONOKERNEL_VALIDATE=0`。

- 常量代入 Qwen3.8-Flash-Next（`config.QWEN38_FLASH_NEXT`）和 MI308X（80 个 CU）。随 TP 变化的量在伪代码里写作 `名字(TP2值|TP4值)`。
- 伪代码里的每个符号都在下面的表格或伪代码中定义。函数名与源码同名，所在文件写在注释里；路径相对 `src/pyhip/ops/monokernel/qwen38/`，SGLang 路径相对 `python/sglang/srt/`。

### 记号

| 记号 | 含义 | 来源 |
|---|---|---|
| `bf16(x)` | FP32 按最近偶数舍入到 BF16 | `flydsl/_ir.py` 的 `bf16_round`、`bf16_word`、`rne_bits` |
| `fp8(x)` | FP32 按最近偶数转 E4M3FNUZ | `_ir.fp8_pack` |
| `σ(x)` | 快速 sigmoid：`rcp(1 + exp2(-x·log2e))` | `_ir.sigmoid` |
| `silu(x)` | `x·σ(x)` | `_ir.silu` |
| `σ_acc(x)`、`exp_acc`、`sqrt_acc` | 精确版：`1 / (1 + exp_acc(-x))`、LLVM `exp`、`sqrt` | `_ir.sigmoid_accurate`、`exp_accurate`、`sqrt_accurate` |
| `silu_acc(x)` | `x / (1 + exp_acc(-x))` | `phases.moe_up`、`phases.qkvz_conv` 中的写法 |
| `softplus(x)` | `F.softplus`（beta 1，threshold 20） | `_ir.softplus` |
| `rsq(x)` | 硬件倒数平方根 | `rocdl.rsq` |
| `exp2(x)` | 硬件 2 的幂 | `_ir.exp2` |
| `gemma(v, w)` | `bf16(v · rsq(Σv²/n + eps) · (1 + w))`，n 为该头或该分支的通道数 | `phases.combine_norm`、`attention.qk_prep`、`attention._gemma_norm2` |
| `ref(m)` | `m` 为 `-inf` 时取 0，否则取 `m` | `attention._guard` |
| `wave_sum`、`wave_max` | 64 lane 归约（DPP、swizzle、bpermute），每个 lane 得到结果 | `_ir.wave_reduce` |
| `load.sc1`、`store.sc1` | aux=16，设备一致的加载、存储（`ctx.aux`） | `_phase.SC1` |
| `store.sys`、`load.sys` | aux=17，系统一致（sc0、sc1） | `xgmi.SYS` |
| `Σ_{i=a..b}` | 按写出的顺序做 FP32 累加 | — |
| `S.名字` | scratch 中的中间量，见下面 scratch 表 | `fused.scratch_layout` |
| `a("X")` | 第 X 个指针参数，在使用处从 kernarg 段加载 | `fused.KernelArgs` |
| `W_QKVZ` 等大写名 | 即 `a("W_QKVZ")` 指向的张量 | 见“kernel 参数与来源”表 |
| `relu(x)` | `max(x, 0)` | `attention.index_logits` |

### 模型常量

| 符号 | 值 | 来源 |
|---|---|---|
| H | 2560 | `Qwen38Config.hidden`；`phases.H` |
| C | 4 | `Qwen38Config.hc_count`；`phases.C` |
| WIDTH | 10240 | `Qwen38Config.hc_width = C·H`；`phases.WIDTH` |
| LOWRANK | 320 | `Qwen38Config.hc_lowrank` |
| eps | 1e-6 | `Qwen38Config.eps`（`FusedLayer.eps`） |
| D | 128 | `Qwen38Config.gdn_head_dim`；`phases.D` |
| KERNEL | 4 | `Qwen38Config.conv_kernel` |
| DA | 256 | `Qwen38Config.head_dim`；`attention.DA` |
| ROT | 64 | `Qwen38Config.rotary_dim`；`attention.ROT` |
| IH、ID | 4、128 | `Qwen38Config.index_heads`、`index_head_dim`；`attention.IH`、`attention.ID` |
| RATIO | 4 | `Qwen38Config.index_compress_ratio` |
| E | 512 | `Qwen38Config.n_experts`；共享专家 id = 512（`shared_expert`） |
| TOP_K | 10 | `Qwen38Config.top_k` |
| MOE_SLOTS | 11 | `Qwen38Config.moe_slots = top_k + 1`（`fused.shard_dims` 第 6 项）；10 个路由专家 + 1 个共享专家 |
| 层类型 | `(layer_id + 1) % 4 == 0` 为 full attention（第 3、7、…、47 层），其余为 GDN | `Qwen38Config.layer_kind` |

### 每 rank 的量

| 符号 | TP2 | TP4 | 来源 |
|---|---:|---:|---|
| tp | 2 | 4 | `work.comm.world`（`PeerMemory.world`） |
| k_heads | 8 | 4 | `Qwen38Shard.gdn_k_heads = 16 / tp` |
| v_heads | 24 | 12 | `Qwen38Shard.gdn_v_heads = 48 / tp` |
| key_dim | 1024 | 512 | `Qwen38Shard.gdn_key_dim = k_heads·D` |
| value_dim | 3072 | 1536 | `Qwen38Shard.gdn_value_dim = v_heads·D` |
| conv_dim | 5120 | 2560 | `Qwen38Shard.conv_dim = 2·key_dim + value_dim` |
| N_QKVZ | 8192 | 4096 | `fused._gdn_steps`：`n_qkvz = conv_dim + value_dim` |
| GROUP | 3 | 3 | `phases.recurrent`：`group = v_heads // k_heads` |
| BA_OUT | 48 | 24 | `2·v_heads`（`phases.quant_ba` 的 `ba_out`） |
| BA_ROWS | 48 | 32 | `packed["gdn_ba"].shape[0]`，`weights.pack_layer` 补到 16 的倍数 |
| BA_CTAS | 2 | 1 | `phases.quant_ba`：`(BA_ROWS/16 + 1) // 2` |
| heads | 12 | 6 | `Qwen38Shard.heads = 24 / tp` |
| kv_heads | 1 | 1 | `Qwen38Shard.kv_heads = max(1, 2 // tp)`（TP4 两两复制） |
| N_QKV | 6656 | 3584 | `attention.qkv_columns(heads) = (2·heads + 2)·DA` |
| K_O | 3072 | 1536 | `fused._core_steps`：`k = heads·DA` |
| INTER_P | 320 | 192 | `weights.moe_inter_padded = ceil(640/tp/64)·64`（`MOE_PADDING = 64`） |
| REGION_BYTES | 1,318,912 | 2,629,632 | `xgmi.region_bytes(tp, 64) = 8192 + 2·tp·64·H·2` |

### 启动与阶段常量

| 符号 | 值 | 来源 |
|---|---|---|
| CTAS | 80 | `fused.ctas_for = multi_processor_count`（MI308X） |
| THREADS | 512（8 个 wave） | `_phase.THREADS`、`_phase.WAVES` |
| HALF | 256（4 个 wave） | `_phase.HALF`、`_phase.HALF_WAVES` |
| 全部 wave 数 | 640 | `Ctx.waves = CTAS·8` |
| LDS | 64 KiB/CTA | `_phase.LDS_WORDS·4` |
| HINTS | `{"waves_per_eu": 2}` | `fused.HINTS` |
| CAPACITY | 64 | `sglang_plugin.CAPACITY` |
| rows | 1..64 | `hidden_states.shape[0]` |
| tiles | rows≤16 为 1，≤32 为 2，否则 4 | `linear.tiles_for` |
| DOWN_ROWS、KSPLIT | 336、16 | `phases.DOWN_ROWS`（320 低秩 + 4 inject + 12 零行）、`phases.KSPLIT` |
| FP8_MAX | 240 | `_ir.FP8_MAX` |
| L2NORM_EPS | 1e-6 | `phases.L2NORM_EPS` |
| TOPK_TOK | 2051 | `attention.TOPK = index_topk + RATIO − 1 = 2048 + 3` |
| CHUNK、CHUNKS | 16、129 | `attention.CHUNK`、`attention.CHUNKS = ceil(2051/16)` |
| TOPK_BLK | 512 | `pyhip/ops/qsa/flydsl/indexer_topk.TOPK` |
| RING | 8 | `attention.RING = 2·RATIO`（SGLang 的 `qsa_ring_slots_per_request`） |
| INDEX_COLUMNS | 640 | `attention.INDEX_COLUMNS = (IH + 1)·ID` |
| Q_SCALE | 0.0901684 | `attention.Q_SCALE = DA^-0.5·log2e` |
| INDEX_SCALE | 0.0883883 | `attention.INDEX_SCALE = ID^-0.5` |
| MROPE_SECTION | (11, 11, 10) | `attention.MROPE_SECTION` |
| SEG_CHUNKS（tiles 1/2/4） | 2/4/8 | `attention.group_for` |
| SEGS（tiles 1/2/4） | 65/33/17 | `attention.segments = ceil(129 / SEG_CHUNKS)` |
| BLOCK_CHUNKS（tiles 1/2/4） | 2/2/4 | `attention.block_keys / 16`（SGLang decode 的 BLOCK_N 32/32/64） |
| MAX_BLOCKS | 65552 | `_Runtime.workspace`：`ceil(262148/64)·16`，262148 = `req_to_token.shape[1]` |
| MAX_RANKS、MAX_CTAS | 8、128 | `peer.MAX_RANKS`、`xgmi.MAX_CTAS` |
| FLAGS、EPOCHS、RECV | 0、4096、8192 | `xgmi.py`，区域内的字节偏移 |

### kernel 参数与来源

“served”指 SGLang 加载后的参数，原地引用（`_Runtime.layer` 经 `weights.SGLANG_NAMES` 取得）；“packed”指 `weights.pack_layer` 生成的 BF16 16x16 预排副本。形状写作“TP2；TP4”。

| 参数 | 张量与来源 | dtype、形状 |
|---|---|---|
| `H_IN` | `h`：`hidden_states`；第 0 层是 embedding，复制成 4 个分支（`_mono_layer`） | BF16 [rows, 10240] |
| `HN_IN` | `hn`：SGLang 的 `residual`；没有时传 0，启动内先做 norm | BF16 [rows, 10240] |
| `NORM_A` | served `hc_attn_norm` ← `attn_hyper_connection.hc_norm.weight` | BF16 [10240] |
| `SCRATCH` | `work.scratch`（`FusedWorkspace`） | uint8，按 `scratch_layout` 分区 |
| `HC_DN_A`、`HC_UP_A` | packed `hc_attn_dn`（down 320 行 + inject 4 行 + 零行）、`hc_attn_up`（`pack_hc_up` 重排） | BF16 [336, 10240]、[10240, 320] |
| `W_QKVZ`、`S_QKVZ` | served `gdn_qkvz` ← `linear_attn.in_proj_qkvz.weight`、`.weight_scale` | FP8 [8192；4096, 2560]、FP32 [8192；4096, 1] |
| `CONV_W` | served `gdn_conv` ← `linear_attn.conv1d.weight` | BF16 [5120；2560, 1, 4] |
| `SLOTS` | `_gdn_state`：`linear_attn_backend.forward_metadata.mamba_cache_indices` | int32 [rows]，<0 为填充行 |
| `CONV_STATE` | `_gdn_state`：`req_to_token_pool.mamba2_layer_cache(id).conv[0]` | BF16 [槽位, 5120；2560, 3] |
| `W_BA` | packed `gdn_ba` ← `linear_attn.in_proj_ba.weight`（[b; a]） | BF16 [48；32, 2560] |
| `A_LOG`、`DT_BIAS` | served `gdn_a_log`、`gdn_dt_bias` ← `linear_attn.A_log`、`linear_attn.dt_bias` | FP32、BF16 [24；12] |
| `GDN_NORM` | served `gdn_norm` ← `linear_attn.norm.weight` | BF16 [128] |
| `SSM` | `_gdn_state`：`mamba2_layer_cache(id).temporal` | FP32 [槽位, 24；12, 128, 128] |
| `W_OUT`、`S_OUT` | served `gdn_out` ← `linear_attn.out_proj.weight`、`.weight_scale` | FP8 [2560, 3072；1536]、FP32 [2560, 1] |
| `W_QKV`、`S_QKV` | served `attn_qkv` ← `qkv_proj.weight`、`.weight_scale` | FP8 [6656；3584, 2560]、FP32 [6656；3584, 1] |
| `Q_NORM`、`K_NORM` | served `attn_q_norm`、`attn_k_norm` ← `q_norm.weight`、`k_norm.weight` | BF16 [256] |
| `TABLE` | `layer.rotary_emb.cos_sin_cache`（先 `_match_cos_sin_cache_dtype`） | BF16 [位置, 64]：前 32 列 cos，后 32 列 sin |
| `POSITIONS` | attention 层 forward 的 `positions` | int64 [3, ≥rows]（MRoPE 三个轴），`pos_stride = positions.stride(0)` |
| `CACHE_LOC` | `forward_batch.out_cache_loc`（转 int64） | int64 [rows] |
| `K_CACHE`、`V_CACHE` | `get_token_to_kv_pool().get_key_buffer(id)`、`get_value_buffer(id)` | BF16 [token 槽位, 1, 256] |
| `W_INDEX` | packed `index_qk` ← `indexer.index_qk_proj.weight` | BF16 [640, 2560]：q 头 0–3 各 128 行，然后 k 128 行 |
| `INDEX_Q_NORM`、`INDEX_K_NORM` | served `index_q_norm`、`index_k_norm` ← `indexer.q_layernorm.weight`、`indexer.k_layernorm.weight` | BF16 [128] |
| `REQ_POOL`、`SEQ_LENS` | `forward_batch.req_pool_indices`、`forward_batch.seq_lens`（转 int64） | int64 [rows]；seq 含本步新 token |
| `REQ_TO_TOKEN` | `get_req_to_token_pool().req_to_token` | int32 [请求槽位, 262148] |
| `RING`、`RING_POS` | `pool.get_qsa_key_state_buffer(id)`、`pool.qsa_rope_position_buffer` | BF16 [请求槽位·8, 1, 128]、int64 [请求槽位·8, 3] |
| `COMPRESSED` | `pool.get_qsa_compressed_k_buffer(id)` | BF16 [·, 1, 128]，按 token 槽位 // 4 索引 |
| `W_O`、`S_O` | served `attn_o` ← `o_proj.weight`、`.weight_scale` | FP8 [2560, 3072；1536]、FP32 [2560, 1] |
| `NORM_M` | served `hc_mlp_norm` ← `mlp_hyper_connection.hc_norm.weight` | BF16 [10240] |
| `HC_DN_M`、`HC_UP_M` | packed `hc_mlp_dn`、`hc_mlp_up` | 同 `HC_DN_A`、`HC_UP_A` |
| `W_ROUTER` | packed `router_shared` ← `mlp.gate.weight` [512, 2560] 与 `mlp.shared_expert_gate.weight` 补到 16 行 | BF16 [528, 2560] |
| `W13`、`S13` | served `moe_w13` ← `mlp.experts.w13_weight`、`w13_weight_scale`（gate、up 两半各补零到 INTER_P） | FP8 [513, 640；384, 2560]、FP32 [513, 640；384, 1] |
| `W2`、`S2` | served `moe_w2` ← `mlp.experts.w2_weight`、`w2_weight_scale` | FP8 [513, 2560, 320；192]、FP32 [513, 2560, 1] |
| `NORM_NEXT` | `hc_consumer.hc_norm.weight`：下一层的 `attn_hyper_connection`，第 47 层为 `hyper_connection_mixer`；`hc_consumer` 为 None（第 0 层，下一层带 PLE）时用本层 `hc_attn_norm` | BF16 [10240] |
| `OUT`、`NORMED` | `_mono_layer` 新分配，返回给 SGLang 作下一层的 `hidden_states`、`residual` | BF16 [rows, 10240] |
| `COMM` | `work.comm.table`（`PeerMemory.table`：8 个区域基址 + 本 rank 号） | int64 [16] |
| `TIMELINE` | 0（服务路径不计时） | — |
| 标量 | GDN：`rows`、`conv_state_bytes = conv_state.numel()·2`；attention：`rows`、`pos_stride`、`table_rows = table.shape[0]`、`rtt_stride = req_to_token.stride(0)`、`rtt_bytes = req_to_token.numel()·4`、`ring_slots = ring.shape[0]`、`compressed_bytes = compressed.numel()·2` | Int32 / Int64 |

### scratch 中间量（`S.名字`）

按 CAPACITY = 64 行分配，每项按 256 字节对齐（`fused.scratch_layout`）。

| 名字 | dtype、形状 | 写入阶段 |
|---|---|---|
| `hn` | BF16 [rows, 10240] | norm、mlp_norm |
| `partial` | FP32 [16, rows, 336] | hc_down |
| `low` | BF16 [rows, 320] | hc_lowrank |
| `gain_a`、`gain_m` | FP32 [rows, 4] | attention 分支、MLP 分支的 hc_lowrank |
| `x` | BF16 [rows, 2560] | hc_mix_up |
| `xq`、`xs` | FP8 [rows, 2560]、FP32 [rows] | quant_ba、quant_index |
| `ba` | BF16 [rows, BA_OUT] | quant_ba |
| `qkvz`、`conv` | BF16 [rows, N_QKVZ]、[rows, conv_dim] | qkvz_conv |
| `core` | BF16 [rows, value_dim] | recurrent（未归一化的输出，供调试） |
| `y`、`yq`、`ys` | BF16 [rows, value_dim 或 K_O]、FP8、FP32 [rows] | recurrent 或 combine_gate；quant_y |
| `index_qk` | BF16 [rows, 640] | quant_index |
| `qkv` | BF16 [rows, N_QKV] | qkv |
| `q` | BF16 [rows, heads, 256] | qk_prep |
| `index_q` | BF16 [rows, 4, 128] | qk_prep（index_prep） |
| `index_logits` | FP32 [rows, MAX_BLOCKS] | logits |
| `topk` | int32 [rows, 2051] | topk |
| `scores`、`chunk_max` | FP32 [rows, 129·16, 16]、[rows, 129, 16] | scores |
| `part_ml`、`part_acc` | FP32 [records, 16, 2]、[records, heads, 256]，records ≤ 1088 | values |
| `logits`、`shared` | BF16 [rows, 512]、FP32 [rows, 16] | router |
| `ids`、`weights` | int32、FP32 [rows, 11] | mlp_topk |
| `plan` | int32 [1 + 3·513]：`[num_active, expert[513], offset[513], count[513]]` | plan |
| `route_row`、`route_weight` | int32、FP32 [rows·11] | plan |
| `mid` | BF16 [rows·11, INTER_P] | moe_up |
| `h1` | BF16 [rows, 10240] | mlp_norm |
| `sync` | 两行 128 字节计数器（到达、离开） | grid barrier |

### 1. 接口层：SGLang 如何进入插件

#### 1.1 进程启动：注册并打上 hook

每个 TP rank 都是一个 scheduler 子进程，各自执行一遍下面的流程，所以每个 rank 的 decoder layer 都被包装。

```python
# SGLang：managers/scheduler.py 中 scheduler 子进程的入口在最开始调用 load_plugins()
def load_plugins():                                           # plugins/__init__.py（只读）
    for ep in entry_points(group="sglang.srt.plugins"):      # GENERAL_PLUGINS_GROUP
        if SGLANG_PLUGINS and ep.name not in SGLANG_PLUGINS.split(","): continue   # 白名单，为空时全部加载
        ep.load()()                                           # 调用 register()；serve.py 设 SGLANG_PLUGINS=qwen38_monokernel
    HookRegistry.apply_hooks()                                # 把登记过的方法替换为包装函数（plugins/hook_registry.py）

# 入口点由 benchmarks/monokernel/plugin/qwen38_monokernel_plugin-0.1.dist-info/entry_points.txt 声明：
#   [sglang.srt.plugins] qwen38_monokernel = qwen38_monokernel_plugin:register
# 该目录需在 PYTHONPATH 中；qwen38_monokernel_plugin.py 只有一行：
#   from pyhip.ops.monokernel.qwen38.sglang_plugin import register
def register():                                               # sglang_plugin.py
    if not PYHIP_MONOKERNEL: return                           # 未设置：不注册任何 hook，后面全部是原生实现
    _RUNTIME = _Runtime()
    #   validate = PYHIP_MONOKERNEL_VALIDATE；native_attention = (PYHIP_MONOKERNEL_ATTENTION == "native")
    #   fused_ar = (PYHIP_MONOKERNEL_AR != "aiter")；layers = {}，workspaces = {}（懒初始化，见 2.1）
    HookRegistry.register("sglang.srt.models.qwen4_exp.Qwen4ExpLinearDecoderLayer.forward",
                          gdn_layer_forward, AROUND)
    HookRegistry.register("sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer.forward",
                          attention_layer_forward, AROUND)
    if validate:                                              # 只在验证模式（场景 D）
        HookRegistry.register("sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._append_shared_to_topk_output",
                              append_shared, AROUND)
# AROUND：apply_hooks 后，Layer.forward(*args) 变成 hook(original_forward, *args)（hook_registry._wrap_fn）
```

#### 1.2 每次前向：从模型循环到分派

```python
# SGLang：models/qwen4_exp.py Qwen4ExpModel.forward（只读，未修改）
hidden_states = embed_tokens(input_ids)                       # BF16 [rows, 2560]
residual = None                                               # 层间：上一层融合出的本层 hc_norm，或 None
for i in range(48):
    hidden_states, residual = layers[i](positions=positions, hidden_states=hidden_states, residual=residual,
                                        forward_batch=forward_batch, hc_consumer=_hc_consumers[i], ...)
    # layers[i](...) 实际进入下面的 hook：GDN 层 → gdn_layer_forward，full attention 层 → attention_layer_forward
hidden_states, _ = hyper_connection_mixer.mix(hidden_states, residual)
# _hc_consumers[i]：下一层的 attn_hyper_connection；第 47 层为 hyper_connection_mixer；
#   下一层带 PLE 时为 None（只有第 0 层，因为第 1 层带 PLE）

def gdn_layer_forward(original, layer, hidden_states, residual=None, **kw):          # sglang_plugin.py
    rt = _RUNTIME
    if rt is None or not rt.accepts(layer, kw["forward_batch"], hidden_states):
        return original(layer, hidden_states, residual, **kw)                         # 场景 E
    rt.calls += 1
    if rt.validate and not torch.cuda.is_current_stream_capturing():
        return _validated(rt, lambda: original(layer, hidden_states, residual, **kw),
                          layer, hidden_states, kw["hc_consumer"], hn=residual)       # 场景 D
    return _mono_layer(rt, layer, hidden_states, kw["hc_consumer"], hn=residual)     # 场景 A、B

def attention_layer_forward(original, layer, positions, hidden_states, residual, forward_batch, **kw):
    # 与 gdn_layer_forward 相同，只是多传 attention = (positions, forward_batch)：
    #   不接受 → original(layer, positions, hidden_states, residual, forward_batch, **kw)        # 场景 E
    #   验证   → _validated(rt, <原生调用>, layer, hidden_states, kw["hc_consumer"], attention, hn=residual)   # 场景 D
    #   其余   → _mono_layer(rt, layer, hidden_states, kw["hc_consumer"], attention, hn=residual)              # A、B、C

def _Runtime.accepts(layer, batch, hidden_states):                                   # sglang_plugin.py
    return (batch is not None and batch.forward_mode.is_decode()                      # 只接受普通 DECODE
            and layer.layer_id != 1 and layer.ple is None                             # PLE_LAYER = 1
            and layer.config.num_hidden_layers != 1                                    # MTP draft 模型只有 1 层
            and 0 < hidden_states.shape[0] <= 64                                       # CAPACITY
            and get_attn_backend().linear_attn_backend.forward_metadata.replayssm_write_pos is None)
# get_attn_backend、get_token_to_kv_pool、get_req_to_token_pool：model_executor/forward_context.py
# LINEAR、FULL：config.LayerKind；_int64：sglang_plugin._int64（转 int64）

def _mono_layer(rt, layer, hidden_states, hc_consumer, attention=None, hn=None):     # sglang_plugin.py
    kind = LINEAR if attention is None else FULL
    mono = rt.layer(layer, kind)                                                      # 2.1
    h = hidden_states if hidden_states.shape[-1] == 10240 else cat([hidden_states] * 4, dim=-1)   # 第 0 层
    if hn is not None and (hn.shape != h.shape or not hn.is_contiguous()):
        hn = None
    work = rt.workspace(mono, h.device)                                               # 2.1
    norm_next = hc_consumer.hc_norm.weight if hc_consumer is not None else mono.served["hc_attn_norm"]
    out, normed = empty_like(h), empty_like(h)                                        # BF16 [rows, 10240]
    if work.comm is None or (kind == FULL and rt.native_attention):
        _split_layer(rt, mono, layer, h, work, norm_next, out, normed, attention)    # 场景 B、C
    else:
        mono.layer(...)                                                                # 场景 A
    return out, (normed if hc_consumer is not None else None)                         # 下一层的 (hidden_states, residual)
```

### 2. 各场景的调用链

#### 2.0 场景总表

| 场景 | 触发条件 | 每层的 GPU 启动 | SGLang 仍执行的部分 |
|---|---|---|---|
| A mono 模式（服务默认） | `PYHIP_MONOKERNEL=1`；`PYHIP_MONOKERNEL_AR` 不是 aiter；attention TP 等于 TP；attention 层还要求 `PYHIP_MONOKERNEL_ATTENTION` 不是 native | 1 次：`gdn_layer` 或 `attn_layer` | 无（整层替换，含两次 all-reduce） |
| B 拆分模式（AITER all-reduce） | `PYHIP_MONOKERNEL_AR=aiter`，或 attention TP 不等于 TP（DP attention，未测）：`work.comm is None` | GDN 层：`attention`、`mlp`、`hc_combine_norm`；attention 层：`prep`、`core`、`mlp`、`hc_combine_norm`；另加 SGLang 的 2 次 all-reduce | `attn_tp_all_reduce`、`tensor_model_parallel_all_reduce` |
| C 拆分模式（原生 attention） | `PYHIP_MONOKERNEL_ATTENTION=native`；只影响 attention 层，GDN 层按 A 或 B | `mix`、SGLang 的 attention、all-reduce、`mlp`、all-reduce、`hc_combine_norm` | `self_attention`（qkv、q/k norm、RoPE、KV 写入、QSA indexer、稀疏注意力、o_proj）与 2 次 all-reduce |
| D 验证模式 | `PYHIP_MONOKERNEL_VALIDATE=1`，且当前批不在 CUDA graph 捕获中 | 先按 A、B 或 C 跑一遍 MonoKernel，再跑两遍原生层 | 原生层整层两遍（参考值、重复值） |
| E 原生 | 插件未启用，或 `accepts` 为假 | 无 MonoKernel 启动 | 整层 |

SGLang 自己的 `SGLANG_USE_PYHIP_QSA=1`（serve.py 的 mono_pyhip 臂）只改变 SGLang 原生路径的实现：prefill，以及场景 C、E 中原生 attention 的实现。它不改变 A、B 的进入方式。

#### 2.1 所有 MonoKernel 场景共用：懒初始化与编译

第一次进入 `_mono_layer` 时才构造，之后复用：

- `_Runtime.layer` 每层构造一次；
- `_Runtime.workspace` 每个设备构造一次，其中 `PeerMemory` 是 TP 组的集合操作，各 rank 在同一层的同一次调用里同时到达；
- `_call` 对每个新的 key 编译一次。

SGLang 在捕获 decode CUDA graph 前，会对每个 batch 大小先 eager 跑两遍前向（`model_executor/runner_backend/` 下的 graph 后端，例如 `full_cuda_graph_backend.py` 的 `for _ in range(2): forward_fn()`），编译就发生在这两遍里。如果 graph 捕获时遇到没编译过的 key，`_call` 直接报错。

```python
def _Runtime.layer(layer, kind):                    # sglang_plugin.py；每层首次调用时构造并缓存在 rt.layers
    # get_tensor_model_parallel_*、get_tp_group、get_attn_tp_group：SGLang distributed/
    rank_shard = shard(tp=get_tensor_model_parallel_world_size(), rank=get_tensor_model_parallel_rank())  # config.shard
    params = dict(layer.named_parameters())
    served = {name: params[path] for name, path in SGLANG_NAMES[kind].items()}       # weights.SGLANG_NAMES，原地引用
    packed = pack_layer(served, kind, QWEN38_FLASH_NEXT)
    return (FusedGdnLayer if kind == LINEAR else FusedAttentionLayer)(rank_shard, served, packed)

def pack_layer(served, kind, config):               # weights.py；shuffle = shuffle_16x16，pad16 = 行补零到 16 的倍数
    for p in ("hc_attn", "hc_mlp"):
        packed[p + "_dn"] = shuffle(pad16(cat(served[p + "_down"], served[p + "_inject"])))   # [320+4 → 336, 10240]
        packed[p + "_up"] = pack_hc_up(served[p + "_up"])   # 第 16b+4jj+g 行 = 分支 g 第 4b+jj 列；[10240, 320]
    packed["router_shared"] = cat(shuffle(served["router"]), shuffle(pad16(served["shared_gate"])))  # [528, 2560]
    if kind == LINEAR: packed["gdn_ba"] = shuffle(pad16(served["gdn_ba"]))         # [BA_ROWS(48|32), 2560]
    else:              packed["index_qk"] = shuffle(served["index_qk"])            # [640, 2560]
    # 另有 router、shared_gate 的单独副本，供阶段内核使用

def _Runtime.workspace(mono, device):               # sglang_plugin.py；每设备一次，缓存在 rt.workspaces
    max_blocks = ceil(get_req_to_token_pool().req_to_token.shape[1] / 64) * 16          # 262148 → 65552
    tp_group = get_tp_group()
    comm = (PeerMemory(region_bytes(tp, 64), tp_group.cpu_group, device)
            if fused_ar and get_attn_tp_group().world_size == tp_group.world_size else None)   # None → 场景 B
    return FusedWorkspace(64, mono.shard, device, max_blocks, comm)
    # FusedWorkspace（fused.py）：dims = shard_dims(shard, max_blocks)
    #   = (conv_dim, value_dim, v_heads, INTER_P, 512, 11, heads, 65552)；scratch = zeros(scratch_layout(dims, 64) 字节)

def PeerMemory(size, group, device):                # peer.py；每个 rank 执行
    local = hipExtMallocWithFlags(size, 0x3)        # hipDeviceMallocUncached
    hipMemset(local, 0, size)
    handles = all_gather_object(hipIpcGetMemHandle(local), group)        # 经 TP 的 CPU group
    bases = [local if r == rank else hipIpcOpenMemHandle(handles[r], 0x1) for r in range(tp)]   # LAZY_PEER_ACCESS
    table = int64[16]([*bases, 0 补到 8 项, rank, 0 ...])                  # TABLE_WORDS = 16
    dist.barrier(group)                             # 所有区域清零并映射后才会有 kernel 写入

def _call(key, builder, args, hints):                # linear.py；所有 MonoKernel 启动都经过这里
    if key not in _COMPILED:
        if torch.cuda.is_current_stream_capturing(): raise RuntimeError("warm MonoKernel stages before graph capture")
        with CompilationContext.compile_hints({"monokernel_config": repr(key), **hints}):
            _COMPILED[key] = flyc.compile(builder, *args)      # 编译并执行这一次（不再重复启动）
    else:
        _COMPILED[key](*args)    # @flyc.jit launch：kernel(...).launch(grid, block, stream)
```

#### 2.2 场景 A：mono 模式（服务默认）

调用链如下（每层一次 GPU 启动，两次 all-reduce 都在启动内）：

```text
Qwen4ExpModel.forward → layers[i].forward（包装）→ gdn_layer_forward / attention_layer_forward
  → _mono_layer → FusedGdnLayer.layer / FusedAttentionLayer.layer → _call → kernel gdn_layer / attn_layer（§3、§4）
```

```python
# _mono_layer 中 work.comm 非 None、且不是 native attention 时：
if kind == LINEAR:
    slots, conv_state, ssm_state = _gdn_state(layer)
    mono.layer(h, slots, conv_state, ssm_state, norm_next, out, normed, work, hn=hn)
else:
    positions, fb = attention
    pool, k_cache, v_cache, req_to_token, req_pool, seq_lens = _attention_pools(layer, fb)
    layer.rotary_emb._match_cos_sin_cache_dtype(h)
    mono.layer(h, positions, _int64(fb.out_cache_loc), k_cache, v_cache, layer.rotary_emb.cos_sin_cache,
               req_pool, seq_lens, req_to_token, pool.get_qsa_key_state_buffer(layer.layer_id),
               pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer.layer_id),
               norm_next, out, normed, work, hn=hn)

def _gdn_state(layer):                                                                # sglang_plugin.py
    backend = get_attn_backend().linear_attn_backend
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    return backend.forward_metadata.mamba_cache_indices.to(int32), cache.conv[0], cache.temporal

def _attention_pools(layer, fb):                                                      # sglang_plugin.py
    pool = get_token_to_kv_pool()
    return (pool, pool.get_key_buffer(layer.layer_id), pool.get_value_buffer(layer.layer_id),
            get_req_to_token_pool().req_to_token, _int64(fb.req_pool_indices), _int64(fb.seq_lens))

def FusedGdnLayer.layer(h, slots, conv_state, ssm_state, norm_next, out, normed, work, hn=None):   # fused.py
    rows = h.shape[0]; tiles = tiles_for(rows); tp = work.comm.world; normed_in = hn is not None
    all_phases = tuple(range(len(layer_phases(GDN_LAYER_PHASES, normed_in))))      # 18，或 19（含 norm）
    args = (h, hn or 0, hc_attn_norm, work.scratch, hc_attn_dn, hc_attn_up, gdn_qkvz, gdn_qkvz_scale, gdn_conv,
            slots, conv_state, packed gdn_ba, gdn_a_log, gdn_dt_bias, gdn_norm, ssm_state, gdn_out, gdn_out_scale,
            *_tail(norm_next, out, normed, work), 0, rows, conv_state.numel() * 2, current_stream)
    # _tail = (hc_mlp_norm, hc_mlp_dn, hc_mlp_up, router_shared, moe_w13, moe_w13_scale, moe_w2, moe_w2_scale,
    #          norm_next, out, normed, work.comm.table)；指针顺序 = GDN_LAYER_ARGS
    key = ("gdn_layer", device, dims, k_heads, BA_ROWS, tiles, 64, eps, 80, tp, normed_in, all_phases, False)
    _call(key, make_gdn_layer(dims, k_heads, BA_ROWS, tiles, 64, eps, 80, tp, normed_in, all_phases), args, HINTS)
    # launch：gdn_layer(...).launch(grid=(80,1,1), block=(512,1,1), stream=current_stream)

def FusedAttentionLayer.layer(h, positions, cache_loc, k_cache, v_cache, table, req_pool, seq_lens, req_to_token,
                              ring, ring_pos, compressed, norm_next, out, normed, work, hn=None):    # fused.py
    pos_stride = positions.stride(0) if positions.dim() == 2 else 0
    all_phases = tuple(range(len(layer_phases(ATTN_LAYER_PHASES, normed_in))))     # 24，或 25（含 norm）；rows、tiles、tp、normed_in 同上
    args = (h, hn or 0, hc_attn_norm, work.scratch, hc_attn_dn, hc_attn_up, attn_qkv, attn_qkv_scale, attn_q_norm,
            attn_k_norm, table, positions, cache_loc, k_cache, v_cache, packed index_qk, index_q_norm, index_k_norm,
            req_pool, seq_lens, req_to_token, ring, ring_pos, compressed, attn_o, attn_o_scale,
            *_tail(norm_next, out, normed, work), 0, rows, pos_stride, table.shape[0], req_to_token.stride(0),
            req_to_token.numel() * 4, ring.shape[0], compressed.numel() * 2, current_stream)   # 顺序 = ATTN_LAYER_ARGS
    key = ("attn_layer", device, dims, tiles, 64, eps, 80, tp, normed_in, all_phases, False)
    _call(key, make_attn_layer(dims, tiles, 64, eps, 80, tp, normed_in, all_phases), args, HINTS)
```

- **层间衔接**：第 0 层没有 `residual`，`normed_in=False`，启动内先做 norm；输入是 embedding 复制成的 4 个分支。第 0 层的 `hc_consumer` 为 None，返回 `(out, None)` 给原生的第 1 层。第 2 层起，`residual` 是上一层（第 1 层由原生实现）融合出的本层 norm，`normed_in=True`。第 47 层的 `NORM_NEXT` 是 `hyper_connection_mixer` 的 norm，返回的 `normed` 交给最后的 `hyper_connection_mixer.mix`。
- **CUDA graph**：捕获时 hook 照常执行，launch 被录进 graph，`out`、`normed` 从 graph 的内存池分配。重放时不经过 Python，直接重放录下的 `gdn_layer`/`attn_layer`，指针都指向 SGLang 的静态输入缓冲和池。

#### 2.3 场景 B：拆分模式（AITER all-reduce）

调用链：`_mono_layer`（`work.comm is None`）→ `_split_layer`。每层在两次 all-reduce 处拆开，两次 all-reduce 由 SGLang（AITER custom all-reduce）完成。这个模式不使用 `residual`（`hn`），第一个启动自己从 `h` 做 norm。

```python
def _split_layer(rt, mono, layer, h, work, norm_next, out, normed, attention):        # sglang_plugin.py
    rows = h.shape[0]
    gains = empty(2, rows, 4, FP32)                 # [0]：attention 分支 combine 增益；[1]：MLP 分支
    if attention is None:                           # GDN 层
        slots, conv, ssm = _gdn_state(layer)
        part_a = empty(rows, 2560, BF16)
        mono.attention(h, slots, conv, ssm, part_a, work, gain_out=gains[0])    # FusedGdnLayer.attention → kernel attention
    else:                                           # full attention 层
        part_a = _attention_core(rt, mono, layer, h, work, gains[0], *attention)
    attn = attn_tp_all_reduce(part_a)               # SGLang layers/dp_attention.py
    h1, part_m = empty_like(h), empty_like(attn)
    mono.mlp(h, attn, gains[0], h1, gains[1], part_m, work)                      # FusedLayer.mlp → kernel mlp
    moe = tensor_model_parallel_all_reduce(part_m)  # SGLang distributed/
    hc_combine_norm(h1, moe, gains[1], norm_next, out, normed)                   # flydsl/hc.py → kernel hc_combine_norm

def _attention_core(rt, mono, layer, h, work, gain, positions, fb):              # sglang_plugin.py
    x, part = empty(rows, 2560, BF16), empty(rows, 2560, BF16)
    if rt.native_attention: ...                     # 场景 C
    k_cache, v_cache, req_to_token, req_pool, seq_lens = _attention_prep(mono, layer, h, x, work, gain, positions, fb)
    #   _attention_prep：_attention_pools + _match_cos_sin_cache_dtype(x)
    #   + mono.prep(h, x, positions, _int64(fb.out_cache_loc), k_cache, v_cache, cos_sin_cache, work, gain,
    #               req_pool, seq_lens, req_to_token, ring, ring_pos, compressed)  # FusedAttentionLayer.prep → kernel prep
    mono.core(req_to_token, req_pool, seq_lens, k_cache, v_cache, part, work)   # FusedAttentionLayer.core → kernel core
    return part
```

拆分模式的各个 kernel 与 §4 的阶段对应如下。前五个都是 80 个 CTA × 512 线程的常驻启动，阶段之间用 grid barrier。它们与 §3 的整层启动有三处不同：

- 输出写进宿主传入的缓冲（`OUT`），不推送到其它 rank，也没有握手；
- `mlp_norm` 的 y 来自 SGLang 归约后的 `attn`，不是 `comm.sources(0)`；
- 增益、`h1` 是宿主传入的张量，不在 scratch 中。

| kernel（构造函数） | 宿主方法 | 阶段（§4） | 写出 |
|---|---|---|---|
| `attention`（`fused.make_attention`） | `FusedGdnLayer.attention` | norm、hc_down、hc_lowrank、hc_mix_up、quant_ba、qkvz_conv、recurrent、quant_y、out_proj | `part_a`（out_proj 部分和）、`gains[0]`、conv/SSM 状态 |
| `prep`（`fused.make_attn_prep`） | `FusedAttentionLayer.prep` | norm、hc_down、hc_lowrank、hc_mix_up、quant_index、qkv、qk_prep、compress、logits | `x`、`gains[0]`、KV/ring/compressed；scratch 中的 `S.qkv`、`S.q`、`S.index_logits` 供随后的 `core` 读取 |
| `core`（`fused.make_attn_core`） | `FusedAttentionLayer.core` | topk、scores、values、combine_gate、quant_y、o_proj | `part_a`（o_proj 部分和） |
| `mix`（`fused.make_mix`） | `FusedLayer.mix` | norm、hc_down、hc_lowrank、hc_mix_up | `x`、`gains[0]`（场景 C） |
| `mlp`（`fused.make_mlp`） | `FusedLayer.mlp` | mlp_norm、hc_down、hc_lowrank、hc_mix_up、router、topk、plan、moe_up、moe_down | `h1`、`gains[1]`、`part_m`（MoE 部分和） |
| `hc_combine_norm`（`hc.make_combine_norm`） | `hc.hc_combine_norm` | 同 §4 的 combine，但每个（行，分支）一个 320 线程 CTA，共 rows·4 个 CTA | `out`、`normed` |

#### 2.4 场景 C：拆分模式（原生 attention）

调用链：`attention_layer_forward` → `_mono_layer`（`kind == FULL and rt.native_attention`）→ `_split_layer` → `_attention_core` 的原生分支。GDN 层不受影响，仍按场景 A 或 B。

```python
# _attention_core 中 rt.native_attention 为真时
mono.mix(h, x, work, gain)                                   # FusedLayer.mix → kernel mix：norm + HC mix → x、gains[0]
part = layer.self_attention(positions=positions, hidden_states=x, forward_batch=fb)
#   SGLang 原生：qkv_proj、q/k norm、RoPE、KV 写入、QSA indexer、稀疏注意力、o_proj，返回本 rank 的部分和
#   （SGLANG_USE_PYHIP_QSA=1 时这里的 QSA 用 PyHIP 实现）
return part
# 之后同场景 B：attn_tp_all_reduce → mono.mlp → tensor_model_parallel_all_reduce → hc_combine_norm
```

#### 2.5 场景 D：验证模式（`PYHIP_MONOKERNEL_VALIDATE=1`）

只在 eager 批上比较：graph 捕获的批照常走 A、B 或 C，不比较。serve.py 的 validate 模式加 `--disable-cuda-graph`，所以每个 decode 批都比较。MonoKernel 一侧调用 `_mono_layer`，按当前环境变量走 A、B 或 C，所以验证覆盖的就是当前配置的模式。

```python
def _validated(rt, run_native, layer, hidden_states, hc_consumer, attention=None, hn=None):   # sglang_plugin.py
    gdn = attention is None
    if gdn:                                          # GDN 状态会被两边各更新一次，先保存
        slots, conv, ssm = _gdn_state(layer); index = slots.long()
        conv_before, ssm_before = conv[index].clone(), ssm[index].clone()
    out, normed = _mono_layer(rt, layer, hidden_states, hc_consumer, attention, hn)          # MonoKernel（A、B、C）
    ids = work.view("ids", (rows, 11), int32).clone()                                         # MonoKernel 的路由（S.ids）
    if gdn:
        conv_mono, ssm_mono = conv[index].clone(), ssm[index].clone()
        conv[index], ssm[index] = conv_before, ssm_before                                     # 恢复，让原生层同起点
    rt.capture_ids = True
    ref_out, ref_normed = run_native()               # 原生层；append_shared 把原生 topk_ids（含共享专家）存进 rt.native_ids
    rt.capture_ids = False
    report = {layer, rows, out: _metrics(out, ref_out)}                                       # rel_l2、max_abs、ulp1
    if gdn:
        report.conv, report.ssm = _metrics(conv_mono, conv[index]), _metrics(ssm_mono, ssm[index])
        conv[index], ssm[index] = conv_before, ssm_before
    repeat_out, _ = run_native()                     # 原生层再跑一遍：原生自身的波动（HC mix、MoE 原子累加等）
    report.native_repeat = _metrics(repeat_out, ref_out)
    if normed is not None and ref_normed is not None: report.normed = _metrics(normed, ref_normed)
    same = 每行排序后的 ids 与 rt.native_ids 相同
    report.route_mismatch_rows = 行数(~same);  report.out_same_route = _metrics(out[same], ref_out[same])
    logger.info("MonoKernel validate %s", report)
    if gdn: conv[index], ssm[index] = conv_mono, ssm_mono                                      # 后续步骤沿用 MonoKernel 的状态
    return out, normed                               # 下一层用 MonoKernel 的结果
# attention 层的 KV、ring、compressed 由 MonoKernel 写入后，又被两次原生运行在读取前覆盖，最终是原生写入的值
```

#### 2.6 场景 E：原生（不进入 MonoKernel）

以下情况由 hook 直接调用 `original(...)`，整层走 SGLang 原生实现：

- 没有设置 `PYHIP_MONOKERNEL=1`（此时根本没有注册 hook），或 `SGLANG_PLUGINS` 不含 `qwen38_monokernel`；
- 不是普通 DECODE：prefill/extend、MTP 的 draft extend 与 target verify、idle 批；
- 第 1 层（PLE）；MTP draft 模型的层（`num_hidden_layers == 1`）；
- 本批行数为 0 或超过 64（CAPACITY）；
- 线性注意力后端带 ReplaySSM（`replayssm_write_pos` 不为 None）。

### 3. kernel 骨架

以下是场景 A 的整层 kernel。场景 B、C 的 kernel 使用同一组阶段函数，区别见 2.3。

```python
# fused.make_gdn_layer → @flyc.kernel gdn_layer(<GDN_LAYER_ARGS>, rows, conv_state_bytes)
# fused.make_attn_layer → @flyc.kernel attn_layer(<ATTN_LAYER_ARGS>, rows, pos_stride, table_rows, rtt_stride,
#                                                  rtt_bytes, ring_slots, compressed_bytes)
# 以下为每个线程的视角；80 个 CTA 同时常驻（每 CU 一个），每 CTA 512 线程
ctx = Ctx(rows, coherent=True)          # _phase.Ctx：tid、lane = tid % 64、wave = tid // 64、cta、
                                        # wave_id = wave*80 + cta、half = tid // 256、half_wave = wave % 4、
                                        # aux = SC1、shared = LDS 64 KiB
comm = Comm(ctx, COMM, tp, 64)          # xgmi.Comm：rank = table[8]，mine = table[rank]
a = KernelArgs(GDN_LAYER_ARGS or ATTN_LAYER_ARGS)

steps = _mix_steps(H_IN=a("H_IN"), NORM_W=a("NORM_A"), GAIN_OUT=S.gain_a, HC_DN=a("HC_DN_A"), HC_UP=a("HC_UP_A"),
                   X=S.x, combine=False, hn_in=a("HN_IN") if normed_in else None)
#   = [norm（仅 normed_in=False）, hc_down, hc_lowrank, hc_mix_up]
if GDN:
    steps += _gdn_steps(..., out=None, push=comm.targets(0, rows), finish=comm.handshake)
    #   = [quant_ba, qkvz_conv, recurrent, quant_y, out_proj]
else:
    steps += _prep_steps(...)           # = [quant_index, qkv, qk_prep（含 index_prep）, compress, logits]
    steps += _core_steps(..., out=None, push=comm.targets(0, rows), finish=comm.handshake)
    #   = [topk, scores, values, combine_gate, quant_y, o_proj]
steps += _tail_steps(comm, ...)
#   = _mlp_steps(H_IN=a("H_IN"), y=Reduced(comm.sources(0, rows)), GAIN_IN=S.gain_a, NORM_W=a("NORM_M"), H_OUT=S.h1,
#                GAIN_OUT=S.gain_m, HC_DN=a("HC_DN_M"), HC_UP=a("HC_UP_M"), push=comm.targets(1, rows),
#                finish=comm.handshake)
#     = [mlp_norm, mlp_hc_down, mlp_hc_lowrank, mlp_hc_mix_up, router, topk, plan, moe_up, moe_down]
#   + [combine]
_run(ctx, steps, sync=S.sync, ctas=80)

def _run(ctx, steps, sync, ctas):       # fused.py
    for i, (prefetch, run) in enumerate(steps):
        if i: grid_arrive(sync)         # §5
        pre = prefetch() if prefetch else None      # 本阶段第 0 轮的权重，不依赖前一阶段
        if i: grid_wait(sync, ordinal=i - 1)
        run(pre)
    grid_exit(sync, exits=sync + 128)   # 最后离开的 CTA 把计数器清零，graph 可以重放
```

阶段序号（`normed_in=True`，即有 `residual` 时；否则在最前面多一个 norm 阶段）：

| 层 | 阶段 |
|---|---|
| GDN（18 个） | 0 hc_down、1 hc_lowrank、2 hc_mix_up、3 quant_ba、4 qkvz_conv、5 recurrent、6 quant_y、7 out_proj、8 mlp_norm、9 mlp_hc_down、10 mlp_hc_lowrank、11 mlp_hc_mix_up、12 router、13 topk、14 plan、15 moe_up、16 moe_down、17 combine |
| full attention（24 个） | 0 hc_down、1 hc_lowrank、2 hc_mix_up、3 quant_index、4 qkv、5 qk_prep、6 compress、7 logits、8 topk、9 scores、10 values、11 combine_gate、12 quant_y、13 o_proj、14–23 同 GDN 层的 8–17 |

### 4. 阶段函数

```python
# ---- 通用：GEMV 阶段（_phase.RoundGemv + _phase.HalfGemm，phases.Gemv）
# 权重 W [N, K] 为 16x16 shuffle；N 按 16 行分块，块数 = items。
# half item：第 r 轮里 CTA c 的 half h 处理 item = 2*(c + 80*r) + h，r < rounds = ceil(ceil(items/2) / 80)。
# 一个 half（4 个 wave）算一个 16 行块：wave w 取该块 K 方向的第 w, w+4, … 个 1 KiB 权重片（16 行 × 64 字节），
#   用 MFMA 乘对应的激活，得到 FP32 部分和 [16 输出, 16*tiles token]；4 个 wave 的部分和经 LDS 按 0+1+2+3 相加。
# 第 r+1 轮的权重在第 r 轮计算前发出；第 0 轮的权重由 prefetch 在 grid_arrive 与 grid_wait 之间发出。
# 单 split 且 tiles=1 时，激活每阶段只读一次（RoundGemv.cache）。
# 尾声：token = 16*t + lane%16，n0 = 16*block + 4*(lane//16)，lane 拿到 acc[token][n0 .. n0+3]；由 wave t%4 收集 tile t。
# acc[token][n] = Σ_k W[n, k] * X[token, k]（FP32；X 为该阶段的激活，token < rows 的结果才写出）。

# ---- norm / mlp_norm / combine（phases.combine_norm）
# wave item (row, branch)，共 rows*4 个，item 从 wave_id 起、步长 640；lane l 负责分支内的列 col = 8*(l + 64*i) + [0,8)，i < 5
def combine_norm(H_IN, y, GAIN_IN, NORM_W, H_OUT, HN_OUT, combine):
    v = H_IN[row, 2560*branch + col]                                     # load.sc1
    if combine:
        yv = reduce(y, (row*2560 + col)*2) if y 是 all-reduce 来源 else y[row, col]     # §5
        v = bf16(v + yv * GAIN_IN[row, branch])
        H_OUT[row, 2560*branch + col] = v
    total = Σ_{i=0..4} wave_sum(Σ_{8 列} v²)
    rstd = rsq(total / 2560 + eps)
    HN_OUT[row, 2560*branch + col] = bf16(v * rstd * (1 + NORM_W[2560*branch + col]))
# norm：     combine_norm(a("H_IN"), -, -, a("NORM_A"), -, S.hn, combine=False)
# mlp_norm： combine_norm(a("H_IN"), comm.sources(0), S.gain_a, a("NORM_M"), S.h1, S.hn, True)
# combine：  combine_norm(S.h1, comm.sources(1), S.gain_m, a("NORM_NEXT"), a("OUT"), a("NORMED"), True)
# HN = a("HN_IN")（normed_in）或 S.hn，供 hc_down 与 hc_mix_up 读取

# ---- hc_down（phases.hc_down）：BF16 GEMV，items = 21 块 × 16 split = 336，每 split K = 640，3 轮
for item: block, split = item % 21, item // 21
    S.partial[split, token, 16*block + j] = Σ_{k ∈ split} HN[token, k] * HC_DN[16*block + j, k]          # j < 16

# ---- hc_lowrank（phases.hc_lowrank）：wave item (row, third)，共 rows*3 个
i = 64*third + lane                                                  # 有效：i < 162
col = 2*i if i < 160 else 320 + 2*(i - 160)
A, B = (Σ_{s=0..15} S.partial[s, row, col]) * 0.25, (Σ_{s=0..15} S.partial[s, row, col+1]) * 0.25
if i < 160: S.low[row, col], S.low[row, col+1] = bf16(silu(A)), bf16(silu(B))
else:       GAIN_OUT[row, col-320], GAIN_OUT[row, col-319] = 2*σ(A), 2*σ(B)     # 4 个分支的 combine 增益

# ---- hc_mix_up（phases.hc_mix_up）：BF16 GEMV，items = 640 块，K = 320，4 轮
for block: c = 4*block + lane//16                                    # 这个 lane 的 4 个输出 = 列 c 的 4 个分支
    tot[g] = Σ_k S.low[token, k] * HC_UP[16*block + 4*(lane//16) + g, k]                                # g < 4
    S.x[token, c] = bf16(0.25 * Σ_{g=0..3} σ(tot[g]) * HN[token, 2560*g + c])

# ---- quant（phases.quant；与 AITER 的逐 token FP8 相同）：wave item 每行
def quant(src, dst, scale_out, K):   # K = 2560 / value_dim / K_O；lane l 取 8*(l + 64*c) + [0,8)，c < K/512
    amax = wave_max(max |src[row, :]|)
    scale = amax * (1/240) if amax > 0 else 1.0
    dst[row, :] = fp8(clamp(src[row, :] * (1/scale), -240, 240));  scale_out[row] = scale

# ---- quant_ba（phases.quant_ba，GDN）
if cta < BA_CTAS(2|1):               # ba 投影：BF16 GEMV，items = BA_ROWS/16 (3|2)，在 BA_CTAS 个 CTA 上 1 轮
    S.ba[token, n] = bf16(Σ_k S.x[token, k] * W_BA[n, k])                                               # n < BA_OUT(48|24)
else:                                # 其余 CTA 按行量化 x
    quant(S.x, S.xq, S.xs, 2560)     # 行从 wave*(80-BA_CTAS) + cta - BA_CTAS 开始，步长 (80-BA_CTAS)*8

# ---- qkvz_conv（phases.qkvz_conv）：FP8 GEMV，items = N_QKVZ/16 (512|256)，4|2 轮
# S.qkvz 的列：[q key_dim | k key_dim | v value_dim | z value_dim]
for block: n0 = 16*block + 4*(lane//16)
    xv[c] = bf16(acc[token][n0+c] * S.xs[token] * S_QKVZ[n0+c])                                         # c < 4
    S.qkvz[token, n0+c] = xv[c]
    if 16*block < conv_dim(5120|2560):                               # q/k/v 列做 causal conv
        slot = SLOTS[token]; active = slot >= 0
        win = [CONV_STATE[slot, n0+c, 0], CONV_STATE[slot, n0+c, 1], CONV_STATE[slot, n0+c, 2], xv[c]]
        t = Σ_{j=0..3} bf16(win[j] * CONV_W[n0+c, 0, j])
        S.conv[token, n0+c] = bf16(silu_acc(t)) if active else 0
        if active: CONV_STATE[slot, n0+c, 0..2] = win[1..3]            # 普通存储（跨启动）

# ---- recurrent（phases.recurrent）：CTA item (row, head)，共 rows*v_heads 个，item 从 cta 起、步长 80
# 第一个 item 的 SSM 状态由 recurrent_prefetch 在 grid_wait 之前读入；计算时预取下一个 item 的状态
# 线程 t：状态行 v = t // 4（0..127），列段 seg = [32*(t%4), 32*(t%4) + 32)
kv = head // GROUP(3)
q = S.conv[row, 128*kv + seg];  k = S.conv[row, key_dim + 128*kv + seg]
val = S.conv[row, 2*key_dim + 128*head + v]
b = S.ba[row, head];  a_ = S.ba[row, v_heads + head];  z = S.qkvz[row, conv_dim + 128*head + v]
qn = q / sqrt_acc(Σ_{128} q² + 1e-6) * 128^-0.5;  kn = k / sqrt_acc(Σ_{128} k² + 1e-6)   # Σ 经同一行的 4 个线程 DPP 合并
beta = bf16(σ_acc(b));  g = -exp_acc(A_LOG[head]) * softplus(a_ + DT_BIAS[head])
St = SSM[slot, head] * exp_acc(g)                                     # FP32 [128, 128]，本线程持有 St[v, seg]
delta = (val - St[v, :] · kn) * beta;  St[v, :] += delta * kn         # packed FP32 FMA
o = bf16(St[v, :] · qn) if slot >= 0 else 0
if slot >= 0: SSM[slot, head][v, seg] = St[v, seg]                    # 普通存储（跨启动）
rstd = rsq(Σ_{v<128} o² / 128 + eps)                                  # 8 个 wave 经 LDS 合并
S.y[row, 128*head + v] = bf16(o * rstd * GDN_NORM[v] * σ_acc(z));  S.core[row, 128*head + v] = o

# ---- quant_y：quant(S.y, S.yq, S.ys, value_dim(3072|1536))；attention 层 K = K_O(3072|1536)

# ---- out_proj / o_proj（phases.out_proj）：FP8 GEMV，items = 160，1 轮，K = value_dim 或 K_O
for block:
    o4 = bf16(acc[token][n0 .. n0+3] * S.ys[token] * W_SCALE[n0 .. n0+3])        # W_SCALE = S_OUT 或 S_O
    for r in range(tp): store.sys(comm.targets(0, rows)[r][token, n0 .. n0+3], o4)   # 推送到每个 rank 的 RECV[0][本 rank]
comm.handshake()                                                       # §5

# ---- quant_index（phases.quant_ba，W = W_INDEX，items = 40，ba_ctas = 20）
if cta < 20: S.index_qk[token, n] = bf16(Σ_k S.x[token, k] * W_INDEX[n, k])        # n < 640：[q0|q1|q2|q3|k]，各 128
else:        quant(S.x, S.xq, S.xs, 2560)

# ---- qkv（attention.qkv）：FP8 GEMV，items = N_QKV/16 (416|224)，3|2 轮
S.qkv[token, n] = bf16(acc[token][n] * S.xs[token] * S_QKV[n])
# S.qkv 的列：头 j < heads 为 [q_j 256 | gate_j 256]，之后 k 256、v 256

# ---- qk_prep（attention.qk_prep，同一阶段接着执行 attention.index_prep）
# qk_prep：wave item (row, j)，j < heads+2：j < heads 为 q 头 j，j = heads 为 k，j = heads+1 为 v；lane l 管通道 4l..4l+3
col = 512*j if j < heads else 512*heads + 256*(j - heads)
w4 = S.qkv[row, col + 4l .. +4]
nm = gemma(w4, Q_NORM if j < heads else K_NORM)                      # n = 256；整头 Σ 经 wave_sum
if lane < 16:                        # 通道 < 64 做交错 MRoPE：pair p = 4*(l%8) + e（e < 4），通道 p 与 p+32 成对
    axis = 1 if p%3 == 1 and p < 33 else 2 if p%3 == 2 and p < 30 else 0           # MROPE_SECTION
    pos = POSITIONS[axis, row];  cs, sn = TABLE[pos, p], TABLE[pos, 32 + p]
    nm[p], nm[p+32] = bf16(bf16(nm[p]*cs) - bf16(nm[p+32]*sn)), bf16(bf16(nm[p+32]*cs) + bf16(nm[p]*sn))
if j < heads:   S.q[row, j, 4l..] = bf16(nm * Q_SCALE)               # 预乘 256^-0.5·log2e
elif j == heads: K_CACHE[CACHE_LOC[row], 0, 4l..] = nm               # store.sc1：同一启动内其它 XCD 要读
else:            V_CACHE[CACHE_LOC[row], 0, 4l..] = w4               # 原值
# index_prep：wave item (row, j)，j < 5：j < 4 为 index q 头 j，j = 4 为 index key；lane l 管通道 2l、2l+1
w2 = S.index_qk[row, 128*j + 2l .. +2]
nm = gemma(w2, INDEX_Q_NORM)                                          # n = 128
if lane < 32: RoPE（pair p = 2*(l%16) + e，与通道 p+32 成对，位置同上；交换伙伴 lane ^ 16）
if j < 4: S.index_q[row, j, 2l..] = nm
else:
    slot = REQ_POOL[row]*8 + (SEQ_LENS[row] - 1) % 8
    RING[slot, 0, 2l..] = w2                                          # 原始投影，不做 norm
    RING_POS[slot, 0..2] = POSITIONS[0..2, row]

# ---- compress（attention.index_compress）：wave item 每行
seq, req = SEQ_LENS[row], REQ_POOL[row]
if seq % 4 == 0:                     # 新 token 补齐了一组
    pooled = bf16(Σ_{m=0..3} RING[req*8 + (seq - 4 + m) % 8] / 4)
    nm = gemma(pooled, INDEX_K_NORM)
    first = req*8 + (seq - 4) % 8;  RoPE 位置取 RING_POS[first, axis(p)]
    COMPRESSED[CACHE_LOC[row] // 4, 0, :] = nm

# ---- logits（attention.index_logits）：每行的页由全部 wave 分摊，一页 = 16 个压缩块
for row in range(rows):
    blocks = SEQ_LENS[row] // 4
    for page in range((wave_id + 61*row) % 640, ceil(blocks / 16), 640):
        blk = 16*page + lane%16;  kb = COMPRESSED[REQ_TO_TOKEN[req, 4*blk] // 4]           # blk < blocks
        S.index_logits[row, blk] = 128^-0.5 * Σ_{h=0..3} relu(S.index_q[row, h] · kb)    # MFMA + DPP 求头和

# ---- topk（attention.index_topk，内部用 PyHIP indexer_topk 的 decode 选择）：CTA item 每行，item 从 cta 起、步长 80
count = SEQ_LENS[row] // 4;  n_blk = min(count, 512)
if count <= 512: S.topk[row, t] = t，t < 4*count
else:            S.index_logits[row, :count] 的精确前 512 块（并列取较小块号），按块号升序，块 b 展开为 token 4b..4b+3
for col in range(4*n_blk, 2051):     # 末尾不满一组的 token，每行共 2051 列
    o = col - 4*n_blk;  tok = 4*count + o
    S.topk[row, col] = tok if o < 3 and tok < seq else -1

# ---- scores（attention.attention_scores）：wave item (row, 4 个 chunk)，每行 ceil(129/4) = 33 个
# 以下 attention 阶段中：req = REQ_POOL[row]，seq = SEQ_LENS[row]；key < 16 为 chunk 内的位置，
#   slot(key) = REQ_TO_TOKEN[req, S.topk[row, 16c + key]]
# lane：g = l // 16，r = l % 16（head = r 为本 rank 的 query 头，r < heads 有效）；一个 chunk = 16 个选中位置
for chunk c:
    idx = S.topk[row, 16c + key];  ok = 0 <= idx < seq
    K16[key] = K_CACHE[slot(key)]（ok 时）
    s[key, head] = K16[key] · S.q[row, head]                         # MFMA：S^T = K Q^T，K = 256
    S.scores[row, 16c + key, head] = s if ok else -inf
    S.chunk_max[row, c, head] = max_key S.scores[row, 16c + key, head]

# ---- values（attention.attention_values）：wave item (row, segment, 通道半区)，每行 SEGS*2 个
# 一个 segment = SEG_CHUNKS 个 chunk；BLOCK_CHUNKS = SGLang decode 一块的 chunk 数
m = max_{c < 段首} S.chunk_max[row, c, head];  total = 0;  acc = 0       # 段前的运行最大值
for c in segment:
    if c % BLOCK_CHUNKS == 0:
        m_new = max(m, 本块各 chunk 的 chunk_max);  α = exp2(m - ref(m_new));  total *= α;  acc *= α;  m = m_new
    p[key] = exp2(S.scores[row, 16c + key, head] - ref(m))           # q 已预乘 log2e，故用 exp2
    total += Σ_key p[key]
    acc[head, ch] += Σ_key V_CACHE[slot(key), 0, ch] * bf16(p[key])   # ch 为本半区的 128 个通道；MFMA
record = row*SEGS + segment
S.part_ml[record, head] = (m, total);  S.part_acc[record, head, ch] = acc

# ---- combine_gate（attention.combine_gate）：wave item (row, head)；lane l 管通道 4l..4l+3
m_last = S.part_ml[row*SEGS + SEGS-1, head].m
w_s = exp2(S.part_ml[row*SEGS + s, head].m - ref(m_last))            # s < SEGS
o = bf16((Σ_s w_s * S.part_acc[row*SEGS + s, head, ch]) / (Σ_s w_s * S.part_ml[row*SEGS + s, head].total))
S.y[row, head, ch] = bf16(o * σ_acc(S.qkv[row, 512*head + 256 + ch]))          # sigmoid 输出门

# ---- router（phases.router）：BF16 GEMV，items = 33（32 个路由块 + 1 个共享门块），1 轮
block < 32：S.logits[token, n] = bf16(Σ_k S.x[token, k] * W_ROUTER[n, k])         # n < 512
block = 32：S.shared[token, j] = Σ_k S.x[token, k] * W_ROUTER[512 + j, k]         # FP32，只用 j = 0

# ---- topk（phases.topk_rows）：wave item 每行；lane l 持有专家 8l..8l+7
p = softmax(S.logits[row, :512])                                      # FP32，exp_acc，经 wave_max、wave_sum
key(e) = (有序化(logit 的 BF16 位) << 16) | (0xFFFF - e)               # 有序化 = moe._ordered；相等时较小的 id 胜出
重复 10 次：e* = argmax key（wave 内 umax），记下 e* 与 p[e*]，再把它移出候选
S.ids[row, 0..9] = e*；S.weights[row, 0..9] = p[e*] / Σ_{10} p[e*]
S.ids[row, 10] = 512；S.weights[row, 10] = σ_acc(S.shared[row, 0])             # 共享专家

# ---- plan（phases.plan）：只在 CTA 0，全部数据经 LDS
routes = rows*11；按 S.ids 统计 count[e]（e < 513）与 64 位行掩码 mask[e]
offset[e] = Σ_{e' < e} count[e']
S.plan = [num_active, 活跃专家（按 id 升序）, 对应 offset, 对应 count]
for 路由 (row, k)，e = S.ids[row, k]：
    pos = offset[e] + popcount(mask[e] 中低于 row 的位)               # 同一专家内按行号排序
    S.route_row[pos] = row;  S.route_weight[pos] = S.weights[row, k]

# ---- moe_up（phases.moe_up）：half item = (活跃专家序号 a, 中间维 16 行块 blk)，blk < INTER_P/16 (20|12)
# items = (INTER_P/16) * num_active；for pair in range(cta, ceil(items/2), 80)：item = 2*pair + half
e, off, cnt = S.plan 中第 a 个活跃专家的 id、offset、count
for pos < cnt（pos = 16*t + lane%16）：tok = S.route_row[off + pos]
    gt[c] = Σ_k bf16(W13[e, 16*blk + c, k]) * S.x[tok, k]                       # FP8 → BF16 精确转换，BF16 MFMA（W8A16）
    up[c] = Σ_k bf16(W13[e, INTER_P + 16*blk + c, k]) * S.x[tok, k]             # K = 2560 由 4 个 wave 分段，LDS 按 0..3 求和
    S.mid[off + pos, 16*blk + c] = bf16(silu_acc(gt[c] * S13[e, 16*blk + c]) * (up[c] * S13[e, INTER_P + 16*blk + c]))

# ---- moe_down（phases.moe_down）：half item = 16 列输出块 blk < 160；CTA c 处理 blk = 2c + half
# half 内 wave w 依次处理活跃专家 a = w, w+4, …，各自在 LDS 中累加 acc_w[row, 16 列]（每个 item 开始时清零）
# e、off、cnt 同 moe_up（第 a 个活跃专家），pos < cnt，c < 16
acc = Σ_{k < INTER_P} bf16(W2[e, 16*blk + c, k]) * S.mid[off + pos, k]
acc_w[S.route_row[off + pos], c] += acc * S2[e, 16*blk + c] * S.route_weight[off + pos]
res[row, c] = bf16(acc_0[row, c] + acc_1[row, c] + acc_2[row, c] + acc_3[row, c])     # row < rows；按 wave 顺序相加，无原子
for r in range(tp): store.sys(comm.targets(1, rows)[r][row, 16*blk + c], res)    # 推送 site 1
comm.handshake()
```

### 5. grid barrier 与启动内 all-reduce

```python
# sync.py；sync = S.sync，exits = S.sync + 128 字节
def grid_arrive(sync):
    s_waitcnt vmcnt(0); fence(release, workgroup); CTA barrier       # 本 CTA 的存储都已确认
    if tid == 0: atomic_add(sync, 1)                                  # monotonic，agent 范围
def grid_wait(sync, ordinal):
    if tid == 0: while load(sync) < 80*(ordinal + 1): s_sleep(1)      # 计数在一次启动内只增不减
    CTA barrier
def grid_exit(sync, exits):
    if tid == 0 and atomic_add(exits, 1) == 79: sync = 0; exits = 0

# xgmi.py：rank s 的区域 R_s（PeerMemory，uncached，每个 rank 都映射了所有区域）
#   R.FLAGS  u32 [128][8]，偏移 0：   FLAGS[c][s] = rank s 的 CTA c 最近一次握手写来的 epoch
#   R.EPOCHS u32 [128]，偏移 4096：   本 rank 的 CTA c 已完成的握手次数（只本 rank 读写）
#   R.RECV   BF16 [2][tp][64][2560]，偏移 8192：RECV[site][s] = rank s 的分片
comm.targets(site, rows) = [R_r.RECV[site][rank] for r in range(tp)]  # 产出阶段 store.sys 写入
comm.sources(site, rows) = [R_rank.RECV[site][s] for s in range(tp)]  # 消费阶段按 rank 顺序读
def comm.handshake():                 # 每个 CTA 在产出阶段末执行，其后的 grid barrier 保证分片已全部到达
    s_waitcnt vmcnt(0); CTA barrier
    if tp > 1:
        if wave == 0:
            epoch = R_rank.EPOCHS[cta] + 1
            lane s（s < tp 且 s != rank）：R_s.FLAGS[cta][rank] = epoch；
                                           spin until R_rank.FLAGS[cta][s] - epoch >= 0（按 2^32 取模）
            lane 0：R_rank.EPOCHS[cta] = epoch
        CTA barrier
def reduce(sources, at):             # combine_norm 的 y：8 个 BF16
    v = Σ_{s=0..tp-1} load.sys(sources[s], at)                        # FP32，从 rank 0 起按顺序
    return v & 0xFFFF0000            # 截断到 BF16（tp > 1），与 AITER one-stage all-reduce 逐位相同
# site 0（mixer 输出）与 site 1（MoE 输出）交替使用：一个 rank 要到下一次另一 site 的握手后才会再写同一 site，
# 而任何 peer 都要在本 site 的消费阶段之后的 grid barrier 才会进入那次握手，所以无需清零
```
