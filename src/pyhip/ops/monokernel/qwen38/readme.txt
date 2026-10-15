Qwen3.8-Flash-Next MonoKernel（gfx942）修改与优化记录
只追加，不改写已有条目。当前用法与最终结果见 benchmarks/monokernel/readme.md。


== 2026-10-13  P0：golden、TP 分片与 SGLang 对齐 ==

内容
- config.py：48 层（36 GDN、12 full attention+QSA）、HC 4x2560、MoE 512 路由专家+1 共享（top10）；TP1/2/4/8 分片。
- weights.py：checkpoint 读取；与 SGLang 一致的分片（GDN q/k/v/z、attention 头、专家中间维连续切分，TP>2 复制 KV 头）；合成权重。
- reference.py：单 rank decode 层 golden，按阶段拆分（HC norm/mix/combine、GDN proj/core/out、attention proj/core/out、
  路由、MoE、all-reduce、整层）。QSA indexer 不在其中（按计划 P3），attention 的 topk 由调用方给出。
- 单元测试 tests/ops/monokernel/test_qwen38_reference.py（13 项）；集成检查 benchmarks/monokernel/。

由 SGLang/AITER 实际内核确定并写入 golden 的数值语义（离线运行 SGLang 自身内核或读 AITER 源码核实）
1. MRoPE：BF16 表下逐运算舍入，bf(bf(a*cos) - bf(b*sin))。FP32 一次舍入版有 5.6% 元素差 1 ulp。
2. QSA decode 稀疏注意力：选中 key 按 32 个一块在线 softmax，P 在当前最大值下转 BF16。全局 softmax 版 rel 9e-4，分块后 1.3e-4。
3. FP8 线性层逐 token 量化同 AITER：scale = amax*f32(1/240)，q = x*rcp(scale)。用除法时个别行 FP8 码翻转（TP4 o_proj 一行 rel 4e-3）。
4. 路由：BF16 logits 常有并列，取较小 expert id（稳定排序）；修正后逐阶段路由 0 行不一致。
5. MoE decode：AITER FlyDSL batch1（调优表 token 1/4/8、inter 320/192 都走此路径）是 W8A16：激活保持 BF16，FP8 权重
   反量化，SiLU 门控中间量为 BF16。最初按 W8A8（激活逐 token FP8）写，对生产内核 moe.partial rel 4.57e-2；改后 5.08e-3。

未模仿的差异（golden 保持就近舍入和 FP32 累加）
- AITER all-reduce 与调优表 (M=4, N=2560, K=3072) 的 cktile GEMM 截断到 BF16，差不超过 1 ulp。
- AITER MoE 每个专家输出加 0x8000 舍入到 BF16，再用 BF16 原子加累加。逐专家最小二乘比例系数都是 1±0.1%，
  剩余约 0.5% 是这部分舍入噪声；模拟截断或固定顺序 BF16 累加反而更差。
- SGLang fused_hc_mix 用原子累加，同一步各 rank 的 hc_mlp.out、logits 偶尔不同。
- Triton sigmoid/rsqrt 偶有 1 ulp 差，经 FP8 再量化会翻转个别码（TP4 gdn.partial 一行）。

结果（check_reference：4 层视图，3 步 S=4 + 1 步 S=1）
- TP2，生产 FlyDSL MoE（qwen38_mono_p0_20261013_03）：PASS。逐阶段除 MoE 外 1 BF16 ulp 内比例为 1.000；moe.partial rel 5.1e-3；
  整层 out rel 6.3e-3、MoE 1.15e-2，路由 1/52 行不同且为近似并列（概率差 3.1%）。
- TP4（qwen38_mono_p0_20261013_01）：采集时 FlyDSL main 与 AITER 不兼容，MoE 为 AITER 回退的 1-stage asm 内核（W8A8），
  非 MoE 阶段同样在 1 ulp 内；MoE 与整层检查对最终 golden 不适用。GPU2-7 被其它会话占用，生产 MoE 的 TP4 尚未重采。
- 合成权重的 TP2/4/8 分片与 TP1 一致（单元测试）。

数据
- mytest/mydata/qwen38_mono_p0_20261013_01：首轮 TP2/TP4 采集；source/ 冻结了当时的 FlyDSL 版代码、采集脚本和探针。
- mytest/mydata/qwen38_mono_p0_20261013_03：TP2 生产 MoE 采集；check_reference.json 为 W8A8 golden（FAIL，据此发现问题），
  check_reference_moe_w8a16.json 为最终 golden（PASS）。
- mytest/mydata/qwen38_mono_p0_20261013_02_failed_relative_model_path：驱动把相对模型路径交给服务导致启动失败，已修正。
- 探针脚本：mytest/qwen38_mono_probe_{kernels,gemm,moe,tp4_proj,quant,attention}.py。

环境
- 代码最初写在 /opt/lc/FlyDSL/kernels/monokernel/qwen38（未提交），10-13 迁入 pyhip，FlyDSL 副本与 README 改动已删除。
- venv 的 FlyDSL 从 editable 0.4.0.dev959（/opt/lc/FlyDSL，与 AITER 4752a408 不兼容）恢复为 AITER 固定的 0.3.2：
  由系统 pip 安装重建 wheel，147 个载荷文件 sha256 一致，pip check 无冲突；pyhip QSA 83 项测试通过，
  AITER FlyDSL MoE 编译和运行正常。记录在 mytest/mydata/flydsl_032_restore_20261013_01。


== 2026-10-13  P0 补充：TP4 生产 MoE、HC mix 近 tie、SGLang 服务内权重格式 ==

TP4 生产 MoE 重采
- GPU4-7 空出后重采（qwen38_mono_p0_20261013_04 为 TP2，_05 为 TP4）。TP4 MoE（中间维 160 补到 192）rel 5.3e-3，与 TP2 同级。
- 首次 TP2→TP4 连续采集时 TP4 入口端口探测失败：上一服务留下 TIME_WAIT，探测 socket 未设 SO_REUSEADDR。已修正。

HC mix 近 tie（TP4 step3 layer1 hc_mlp.mix：exact 0.975、ulp1 0.998，超出 0.999）
- 离线对同一输入重复运行生产 fused_hc_mix 200 次，输出完全相同且逐位复现采集值，不是运行间不确定性。
- 逐个把 golden 的 BF16 低秩激活翻转 1 ulp，只有 r=289 一处翻转使 golden 与生产逐位一致；该激活的 FP32 值距 BF16 tie 1.8 个
  FP32 ulp。生产内核的 FP32 累加顺序（原子累加 + tl.dot）与 torch 不同，把它舍入到另一侧，整行输出随之移动。
- 处理：reference.resolve_hc_mix_ties 对距 tie ≤ 32 个 FP32 ulp 的激活尝试另一侧舍入（每行最多 2 个，只保留能减少不一致的
  翻转），check_reference 的 HC mix 逐阶段检查和链式 hc_attn_out 都用它，并报告 tie_flips/tie_ulps。之后的采集中最远一次为
  30.9 ulp；窗口按相对 ulp 计，激活很小时累加误差相对更大，若以后出现 >32 的合理翻转需改为按 sum|x·w| 的绝对窗口。
- 探针：mytest/qwen38_mono_probe_hcmix.py。

SGLang 服务内权重格式（MonoKernel 直接读取，不另存权重）
- checkpoint 为 compressed-tensors（权重 per-channel、激活 per-token 动态），走 CompressedTensorsW8A8Fp8 / ...Fp8MoEMethod：
  normalize_e4m3fn_to_e4m3fnuz（同位读作 FNUZ、scale×2、0x80→0），稠密层与专家都 shuffle_weight(w, (16, 16))；
  专家中间维按 AITER_MOE_PADDING_SIZE=64 每半补零；BF16 权重保持 checkpoint 布局。
- weights.py 新增 to_fnuz/shuffle_16x16/unshuffle_16x16/moe_inter_padded/sglang_format 和 SGLANG_NAMES（与 SGLang 参数名对应）。
  shuffle 与 AITER shuffle_weight 逐位相同（单元测试），fnuz 反量化值不变。
- 采集插件第一步写出各层参数/状态池布局（layout.json）与每个参数 4096 点原始位采样（param_samples.pt）；check_reference 的
  served_format 用 sglang_format(checkpoint) 逐位比对。qwen38_mono_p0_20261013_06 中 TP2/TP4 每 rank 的 layer0/layer3 各 23 个
  张量 0 处不同。SGLang 另有派生 buffer（q/k_norm.gemma_weight、indexer norm 的 gemma_weight、rotary cos_sin_cache）不在其中。
- 状态池：mamba conv [33, conv_dim, 3] BF16 连续，temporal [33, HV, 128, 128] FP32；KV [token 槽位, KV 头, 256] BF16；
  req_to_token int32 [33, 262148]；QSA pool 另有 key_state/compressed_k/rope_position 缓冲（P3 再用）。

结果（qwen38_mono_p0_20261013_06，check_reference 含 served_format）
- TP2 PASS、TP4 PASS；最坏值见 benchmarks/monokernel/readme.md。TP2 链式 attention 块 rel 1.5e-2（S=4 cktile GEMM 截断经
  FP8 再量化放大），TP4 链式路由 2/52 行不同、概率差 ≤3.1%。
- 单元测试 21 项（新增 HC mix 近 tie 1 项、权重格式 7 项）。
- _06 首次运行显示 complete=False：完整性检查只认 step 文件，新增的 layout/param_samples 被当成多余文件；已改为同时要求这两类文件，
  采集数据本身完整，检查直接在其上运行。

P4 跳过的层
- 用户要求 P4 第一步跳过“第一层（有 PLE）”。PLE 权重在 model.language_model.layers.1.ple.*（config.ple_layer_ids=[2] 按 1 起算），
  即 layer_id 1；layer 0 只需把 embedding 复制到 4 个分支（SGLang 在 _prepare_qwen4_exp_attn 中做），可正常替换。


== 2026-10-13  P1-1：阶段版 GDN 层内核（每阶段一次启动） ==

内核（src/pyhip/ops/monokernel/qwen38/flydsl/，FlyDSL 0.3.2 低层写法：raw buffer 描述符 + MFMA intrinsic + LDS 地址）
- linear.py：quant_rows（AITER 公式，逐位）；Gemm（16 行权重块 × 最多 64 行 token，CTA 内 4 个 wave 分 K，LDS 归约）；
  make_linear：W8A8（直接读 SGLang 的 FP8 shuffle 权重与 scale）、BF16（读 pack_layer 的 shuffle 副本）、FP32 split-K。
- hc.py：down（split-K=16，附 4 行 inject）→ lowrank（FP32 归约、bf16(silu(t/4))、gain=2·sigmoid）→ up+mix（up 行重排，
  使一个 lane 持有同一列的 4 个分支）；combine+norm（gr_write 写法：每 (行, 分支) 一个 5-wave CTA）。
- gdn.py：qkvz GEMM 的 epilogue 内做 causal conv（BF16 舍入的抽头乘积、SiLU）并更新 conv 状态，省去单独的 conv 阶段；
  delta rule 每 (行, v 头) 一个 CTA，线程对 = 状态一行的两个 K 半段，DPP 交换求和；门控 RMSNorm 同 CTA 完成。
- moe.py：top-k 与分组计划在一个 1024 线程 CTA 内：每专家 64 位行掩码，路由位置 = 前缀偏移 + popcount（行序确定）；
  W8A16 up（FP8→BF16 精确转换后 BF16 MFMA，gate/up 成对）；down 每 16 列一个 CTA，4 个 wave 轮流处理活跃专家，
  按 LDS 每 wave 累加器、固定顺序求和（确定性，无原子）。
- layer.py：StagedLayer 按两次 all-reduce 分 attention_phase / mlp_phase / finish 三段。

验证
- 单元测试 76 项（阶段 40、MoE 10 含 512+1 专家 TP2/TP4、GDN 5 等）。check_kernels.py 用真实采集（qwen38_mono_p0_20261013_06）：
  TP2、TP4 均 PASS，各阶段与 SGLang 的差距与 golden 同级（结果表见 benchmarks/monokernel/readme.md）。

问题与修正
- split-K 的 f32 输出：token 超出 rows 的填充行写入了下一个 split 的区域（descriptor 只防整体越界），改为显式屏蔽。
- flyc.compile 本身会执行一次调用，_call 编译后又调用一次，conv/ssm 状态被更新两次；改为编译调用即返回。
- FlyDSL 0.3.2 的缓存键只含依赖源码、标量闭包值和 compile hints；Gemm 对象里的 K 不在键中，TP4 的 out_proj（K=1536）
  从磁盘缓存取到了 TP2 的 K=3072 kernel，越界读导致 GPU memory fault（时序相关、加打印即消失，也可能静默算错）。
  _call 现在把完整配置放进 compile hint。
- TP4 的 ba 只有 24 行：BF16 线性层支持输出列数为 4 的倍数（权重补零到 16 行，存储按真实列数屏蔽）。

初测（CUDA graph 重放，GPU0，非正式）：qkvz（N=8192,K=2560）S=1 7.5us（2.8TB/s）、S=16 14.9us、S=64 47.6us；
S 大时每个 16 行块都从 L2 重读整份激活，是下一步优化点（多块共用激活、LDS 暂存）。Python 逐次启动开销约 20us/次，
计时必须用 graph。
