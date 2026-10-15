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


== 2026-10-13  P1-2：GDN 层的常驻启动（0646468、b8ff9a1） ==

内容
- fused.py：GDN 层的两半（attention 半：norm .. out_proj；MLP 半：combine/norm .. MoE down）各一次常驻启动，
  每个 CU 一个 512 线程 CTA，阶段函数之间用 grid barrier（sync.py）。跨 CTA 数据用 sc1 存取，不依赖 L2 写回。
- _phase.py RoundGemv：GEMV 阶段按静态轮次调度，下一轮权重在本轮计算前发出，第 0 轮权重在 barrier 的
  arrive 与 wait 之间发出（split grid barrier）；单 split、单 token 块时激活每阶段只读一次。
- in_proj_ba 挪到激活量化旁；路由与共享专家门合成一块权重（router_shared）。
- 递推在 wait 前预取第一个状态、计算时预取下一个，用 packed FP32 FMA，每个向量只算一次倒数（对 golden 验证）。
- bench_layer.py 增加 --impl fused；可选 s_memrealtime 阶段时间线。
验证：fused 与 staged 在 TP2/TP4、rows 1–64 逐位相同（探针 mytest/qwen38_mono_probe_fused.py）。


== 2026-10-13  P4：SGLang decode 插件与自启服务驱动（5b23d26） ==

内容
- sglang_plugin.py + benchmarks/monokernel/plugin：AROUND hook 包住两类 decoder layer 的 forward。满足条件时
  （DECODE、≤64 行、非 PLE 层、非 MTP 层、无 ReplaySSM）走 MonoKernel，否则原样调用原生实现。权重直接引用
  SGLang 参数，另加 pack_layer 的 BF16 副本。
- PYHIP_MONOKERNEL_VALIDATE=1：eager 批在同输入、同 GDN 状态上再跑原生层，记录 out、out_same_route（只比较路由
  相同的行）、native_repeat（原生两次运行之差，作噪声基线）以及 conv/ssm。
- serve.py：自启服务（参数同用户启动脚本，可见卡作参数），validate 与 bench 两种模式。bench 协议同此前 QSA
  服务研究：random 32×12000→350，C1/2/4/8。
- 最初只替换 GDN 层（attention 层原生）：TP2 C1 ITL 13.40→12.14 ms。再给 attention 层加上融合的 HC mix 和
  MLP 半（attention 本身原生）：11.75 ms。

问题
- mono_bench_tp2_warm_failed_concurrent_edit：服务运行中编辑了源码，新 import 读到半成品而失败。此后 serve.py
  把 src/pyhip 与插件复制到运行目录的 source/ 再加载，每次运行用新的 FlyDSL 缓存。


== 2026-10-13  P2：full attention mixer 融合（a2397ff） ==

内容
- attention 层拆成两次常驻启动。prep：HC mix、qkv、q/k norm、MRoPE、KV 写入；core：稀疏注意力、门控、o_proj。
  QSA indexer 仍用 SGLang 的（在两次启动之间调用）。
- 稀疏注意力最初用单遍在线 softmax，服务校验 attention 层 out_same_route 中位 0.0034（mono_validate_tp2_p2b），
  明显高于原生自身的 0.0016。SGLang decode 先算每块最大值再做两遍；改为同样的两遍并使用相同的块最大值后，
  中位降到 0.0020（_p2b_twopass），复测 0.0019（_p2b_repeat）。
- 服务 TP2 C1 ITL 11.75→9.88 ms。


== 2026-10-13  P3：QSA decode indexer 融合（3260819） ==

内容
- attention.py 加入 indexer：index q/k 投影（BF16，与激活量化同阶段）、index q 的 Gemma norm 与 RoPE、
  index key 写入 SGLang 的 pending ring；新 token 补齐一组时做 4:1 压缩写入 compressed 池；
  对每行全部压缩块算分页 relu logits；CTA 内 top-k（与 SGLang indexer_topk 相同的升序输出）并展开为 token 位置。
- prep 末尾产出 index logits，core 开头做 top-k；SGLang 的 indexer 不再调用。
- 服务：TP2 C1–C8 ITL 7.22/8.12/10.07/13.41 ms，TP4 6.48/7.35/8.67/11.27 ms。
  校验（out_same_route 中位，attention/GDN）：TP2 0.00199/0.0019，TP4 0.00194/0.00168；
  原生自身 0.00165/0.00156、0.00172/0.00164。


== 2026-10-13  P4b：启动内 all-reduce（e9ac268） ==

设计（xgmi.py、peer.py）
- 每 rank 一块 hipExtMallocWithFlags(Uncached) 显存，IPC 句柄经 TP 的 CPU group 交换（all_gather_object），
  每个 rank 映射所有区域；ctypes 调用当前进程已加载的 libamdhip64（从 /proc/self/maps 找），与 torch 共用 context。
- 推送式：产出阶段把本 rank 的每个 BF16 输出块用 sc0|sc1 存储写进所有 rank 的 recv[site][rank]。阶段末等
  vmcnt 归零后做 CTA barrier，wave 0 的各 lane 写 peer 的标志、再等自己的标志（每 CTA 一个 epoch，按 2^32
  取模比较）。之后的 grid barrier 保证分片已全部到达。
- 消费阶段按 rank 顺序 FP32 求和并截断到 BF16，与 AITER one-stage all-reduce 逐位相同。site 0/1 在两次
  all-reduce 间交替，无需清零。
- 微测（mytest/qwen38_mono_xgmi_probe.py、qwen38_mono_launch_floor.py）：0 错；握手约 2.4 µs；推送 64 行约多
  6.6 µs（约 50 GB/s）；空常驻启动 1.96 µs；一次 grid barrier 约 1.4–1.8 µs。
- GDN 层一次启动；attention 层为 prep + 其余（core、两次 all-reduce、MLP 半、combine）两次启动。

结果与问题
- 服务 TP2 C1–C8 ITL 7.30/8.24/10.18/13.49 ms，反而比 P3（AITER all-reduce）略慢。
- 诊断：AMDGPU 后端在 kernel 入口把全部参数装进 SGPR，整层 kernel 参数多，SGPR 溢出 647 处、v_readlane 1675 条，
  抵消了省下的 all-reduce 启动。
- 计时探针在一个地址上做约 4 万次原子加，扭曲了计时；改为只在出错数非 0 时才原子加。
- 两种 kernel 同名，ISA dump 目录互相覆盖；改名为 gdn_layer / attn_layer。


== 2026-10-13  每层一次启动、上一层 norm 作输入、参数按需加载（eef32ec） ==

内容
- KernelArgs：Int64 指针参数在使用处从 llvm.amdgcn.kernarg.segment.ptr + 8·index 加载，缓冲区在 lambda 里
  懒构造。SGPR 溢出 647 → 376（attn_layer）/165（gdn_layer）。
- attention 层的 prep 并入同一启动。KV 写入改为写穿（aux），后续阶段在其它 XCD 上用普通加载读取；同一启动内
  这些地址写前不会被其它 XCD 读进 L2。
- SGLang 传给下一层的 residual 就是上一层 combine 算出的 attention 分支 norm：作为 hn 输入，省去 norm 阶段。
- 服务（TP2 GPU0/1、TP4 GPU4–7，并发运行于不同卡）：
  TP2 C1–C8 ITL 7.17/8.14/10.07/13.31 ms，TP4 6.42/7.30/8.58/11.16 ms；相对 P3 约低 1%。
  校验 out_same_route 中位（attention/GDN）：TP2 0.00198/0.00191，TP4 0.00202/0.00164；
  原生自身 0.00165/0.00159、0.00177/0.00159。
- 逐位探针（mytest/qwen38_mono_probe_layer_ar.py，此后固化为 benchmarks/monokernel/check_layer.py）：
  整层启动对“拆分启动 + 精确 all-reduce + 单独 combine”，TP1/TP2/TP4 全部逐位相同。

时间线（rows=1、TP2、计时模式，GDN 层约 144 µs）中较大的延迟项，括号内为其后 barrier 等待：
quant_ba 6.5（7.9）、quant_y 1.1（2.9）、mlp_norm 6.2（7.3）、topk 5.5（6.4）、plan 2.8（3.6）、combine 6.0；
握手时间计入 out_proj / moe_down。


== 2026-10-13  收尾：组合臂、单层计时、整层逐位检查固化（096b438、eb9615d） ==

- serve.py 增加 mono_pyhip 臂（MonoKernel decode + SGLANG_USE_PYHIP_QSA=1 的 prefill）。TP2 C1–C8 ITL
  7.21/8.21/10.19/13.45 ms、TTFT 898/1462/3176/4772 ms；TP4 6.39/7.26/8.51/11.04 ms、676/1094/2707/3683 ms；
  TP4 C8 输出吞吐 298.4 tok/s，高于 PyHIP QSA 的 273.1 和 MonoKernel 的 258.1。校验与单开 MonoKernel 同级。
  PyHIP QSA 臂 TP2 复测 10.86/11.86/14.35/15.44 ms，与 10-04 的 QSA 记录一致。
- check_layer.py：TP1（GPU4）、TP2（GPU4/7）、TP4（GPU4–7）全部逐位相同。
- bench_layer.py --impl layer（单 rank 通信区，无跨卡等待），µs/层：
  TP2 S=1/4/16/64：staged 122.7/174.9/337.9/1600，fused 117.7/175.7/360.0/2207，layer 126.1/180.0/354.3/2301；
  TP4：staged 110.1/138.7/260.8/1012，fused 99.3/144.4/280.5/1383，layer 105.4/148.9/272.3/1446。
  结论：
  - S=1 时 GDN 层每 rank 只读约 80/58 MiB 权重（按 4 TB/s 约 21/15 µs），实测是它的 6–7 倍，约 20 个相互依赖的
    阶段都受延迟限制。
  - CUDA graph 下 grid barrier 与 kernel 边界代价相近，常驻启动本身对 S=1 几乎没有收益。相对原生的收益主要来自
    阶段内核本身。
  - S≥16 时 MoE 阶段只有 80 个 CTA，合成输入激活几乎所有专家，慢于按专家数展开网格的阶段内核。
- 生成序列在不同运行间会在前几个 token 分叉，原生 prefill 自身两次运行也是如此（近似并列处的贪心翻转），
  不作判据。
- 未做的优化（估计值，均未实现）：
  - 激活量化并入 FP8 GEMV（每个 CTA 自算整行 amax，与 quant 阶段逐位相同），ba / index q/k 投影与 qkvz/qkv
    同阶段错开 CTA：约 10–15 µs/层。
  - top-k 与分组计划合并：约 3 µs/层。
  - combine 类阶段在 barrier 等待前预取 h、gain 与 norm 权重：约 4 µs/层。
  - 大 S 下按行数切换到阶段内核，或给 MoE 阶段更多并行度。
  - 跨 XCD 一致性探针 mytest/qwen38_mono_xcd_coherence.py 写好但未运行，当前用法不依赖它所测的情形。

数据：mytest/mydata/qwen38_mono_p4_20261013_01（各运行目录的 manifest.json 记录代码版本；src_3260819、src_096b438
为当时用于服务的 git archive 导出）。
