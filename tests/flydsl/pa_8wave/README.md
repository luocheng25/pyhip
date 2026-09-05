# Paged Prefill 8-wave

## 当前：gfx950 OPUS 流水复现（2026-09-05）

[pa_8wave_950.py](pa_8wave_950.py) 已整体重写；本节是当前实现与验证结果。
下方旧版记录对应 [pa_prefill_8w32x32.py](pa_prefill_8w32x32.py) 和早期实验，
**不适用于新实现**。原始 OPUS/AITER 内核未修改，执行路径不调用 AITER attention。

### 范围与接口

- 仅支持 gfx950、BF16、`Dqk=192/Dv=128`、page64 SHUFFLE-5D K/V。
- 保留 `PagedAttention` 工厂与原调用参数；支持 GQA/MHA、causal/noncausal、
  ragged batch、非零 prefix 起点、尾页、非连续 Q/O（head dimension 连续）。
- 支持 scalar/per-token-head Q descale、scalar K/V descale；descale 必须有限且为正。
- `out=` 复用输出；`return_lse=True` 返回 `(out, lse)`，LSE 为 FP32
  `[total_q, Hq]`、自然对数；支持传入 `lse=` 和正的 `softmax_scale=`。
- 空 KV 和 causal 全遮罩行写 `O=0, LSE=-inf`；支持指定 stream 和预热后的 HIP graph。
- FP8、D128、其他 page size、SWA、sink 及旧 fallback 在本实现中已移除，显式报错。
- GPU prefix/page-table 值与 `max_seqlen_q/max_seqlen_k` 必须由调用方保证一致；
  热路径不复制 metadata 到 CPU。KV byte span 暂限有符号 32-bit offset，未实现 OPUS 的大地址 rebasing。

### OPUS 对应关系

- 512 threads、`BM=256/BN=64`；Q/K/V 的 LDS 写入与寄存器读取布局按 OPUS 复现。
- Q 在 prologue 中占用 V 区域；K 双槽、V 双槽，LDS 共 **149,760 B**。
- 完整 `STAGGER=True/False` 编译期分体，仅在入口按 wave-group 分支；每阶段
  保留 scheduler fence、waitcnt 和 workgroup barrier。
- 每个 phase 的 8 个 stage 交织 `QK(t)` 与 `exp/sum/PV(t-1)`；两个 phase
  展开成 ping/pong，偶数 tile 独立收尾。QK/PV 均为原生 BF16 `32x32x16` MFMA，
  P 显式转为 BF16，PV 采用 K-major 双 accumulator 交织。
- 复现 lazy-max 阈值 8、stage5 的 stagger 12 / nonstagger 6 个 scale-sub，以及
  OPUS 的 score、row-sum、probability materialization fence。
- causal 仅对需要遮罩的 wave/tile 做 mask；完整 grid 达到 512 WG 时配对首尾
  query block，镜像 block 反向遍历 KV，匹配 OPUS 的负载均衡策略。
- 使用 `permlane32` 打包，attention epilogue 每 lane 8 条 128-bit store，无 C-shuffle。
- 布局、寄存器和 MMA 保留 FlyDSL 接口；显式 LDS-read、scalar FMA 和 pin 是局部 ISA
  边界。当前 copy lowering 会为单个共享数组添加多余 `vmcnt(0)`；显式 read 遵守
  调用点的 `lgkmcnt(0)` / barrier，保留 OPUS rolling VMEM waits。

**分页适配是独立 FlyDSL gather，不是 direct-paged attention。** 每次公开调用都先
从当前 page table/KV 数据重新 gather，再运行 attention；只缓存 workspace 和编译结果，
不缓存 attention 数据。Workspace 按 device/stream/batch/capacity 隔离，大小为
`B * ceil(max_KV/64) * 64 * Hkv * (192+128) * 2` 字节。目标形状约 **1.60 MiB**。

### 实测性能

MI350X gfx950，`B=1,Hq=16,Hkv=1`，预分配输出、同输入比较；每候选 20 次预热、
100 次采样，5 轮交替顺序取中位数。下表为 `run_perftest` 的 **GPU 时间**，不含编译
和首次分配；PyTorch 仅用于参考，不参与计时。Full 是单独计时的 gather+attention，
不是两个独立中位数相加。OPUS 对照使用线性 KV，不含 gather。

| 场景 | FlyDSL full（含 gather） | FlyDSL core | gather | OPUS linear | full / OPUS 差距 |
|---|---:|---:|---:|---:|---:|
| noncausal Q10240 / KV2583 | **299.239 µs** | **294.696 µs** | 3.827 µs | **291.345 µs** | +2.71% |
| causal Q8192 / KV8192 | **345.690 µs** | 339.730 µs | 4.328 µs | **335.805 µs** | +2.94% |
| causal Q32768 / KV32768 | **5284.93 µs** | 5279.93 µs | 7.874 µs | **5209.49 µs** | +1.45% |

目标 noncausal full 为 **905.1 TFLOPS**，core 为 919.1 TFLOPS；core 比 OPUS 慢约
**1.15%**。这些结果是接近性能，不是更快或全形状完全等速的声明。短序列的 gather
启动开销占比更大。历史旧版本目标约 331.9 µs，不作为跨进程精确 A/B 基准。

| attention specialization | VGPR | SGPR | LDS | Private | VGPR / SGPR spill |
|---|---:|---:|---:|---:|---:|
| 目标 noncausal，无 LSE | 256 | 71 | 149760 B | 0 B | 0 / 0 |
| causal 首尾配对，无 LSE | 256 | 80 | 149760 B | 0 B | 0 / 0 |

### 验证与复现

[test_pa_prefill.py](test_pa_prefill.py) 已替换为当前 specialization 的严格测试：
**50 passed**。覆盖 1–6 页奇偶收尾、NaN poison 尾页、ragged batch、GQA、多种 Q/O stride、
descale 与 lazy-max rescale、LSE、空输入、每次更新 page table/KV、缓存后改变 runtime
长度、stream/graph、首尾配对的奇偶 query-block、默认无 LSE 分支与目标 shape。所有数值失败都会抛出，
不再使用吞异常的 `accuracy unknown`。与 OPUS 的目标固定随机输入最大绝对差为
`0.00048828125`；完整测试使用独立 FP32 PyTorch reference。

当前运行环境使用 `/opt/venv/bin/python`（Python 3.10.12，ROCm 7.2）；从本目录运行：

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python -m pytest -q test_pa_prefill.py
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python test_pa_prefill.py
FLYDSL_RUNTIME_ENABLE_CACHE=0 /opt/venv/bin/python test_pa_prefill.py --q-len 32768 --kv-len 32768 --causal 1
```

基准会先严格检查 gather 和每个 attention candidate，再输出完整 markdown 表；`err`
必须为 0。资源复核可在上述命令加 `FLYDSL_DUMP_IR=1` 查看生成 ISA。

---

## 历史记录（2026-09-03，旧实现）

更新时间：2026-09-03。完整的4-wave/8-wave/AITER统一数据、口径和gfx942历史结果见
[4-wave主报告](../pa_4wave/README.md)；本文件只维护8-wave实现及其当前摘要。

## 功能范围

- 支持gfx942/gfx950 BF16与架构原生FP8 vectorized paged-prefill；
- gfx942 FP8使用`torch.float8_e4m3fnuz`，gfx950使用OCP
  `torch.float8_e4m3fn`，launcher拒绝非原生编码；
- 覆盖D128/D192、causal/non-causal、page 32/64/128、per-token/per-tensor Q scale、
  ragged tail及gfx950 SWA + sink；
- gfx950完整回归：`67 passed, 1 skipped`。

## 当前性能摘要

平台为MI350X gfx950，主配置为`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。
除特别说明外，计时为20次预热、100样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | 8-wave结果 | 对照结论 |
|---|---|---:|---|
| non-causal `Q10240,KV2583` | BF16 native 512-thread | 350.43 us / 772.89T | API始终运行原生8-wave；同轮4-wave为304.07 us |
| causal `Q=KV=32768` | BF16 native 512-thread | 8392-8463 us（历史） | 已删除4-wave自适应回退 |
| causal `Q=KV=32768` | OCP FP8 K16 | 5611.21 us / 979.75T | K64前基线 |
| causal `Q=KV=32768` | OCP FP8 K64 | **5101.77 us / 1077.58T** | 较K16延迟下降9.08% |
| SWA `Q16K,KV128K,window=128` | BF16 native 512-thread | 140.49-142.20 us（历史） | 已删除4-wave自适应回退 |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | **112.35 us / 192.64T有效 / 573.43T执行** | 较K16下降0.49% |

SWA有效FLOPs按每行129个可见KV token计算；执行TFLOPS计入被mask但仍进入MFMA的
tile工作。完整SWA过程数据见[gfx950_swa_performance.md](gfx950_swa_performance.md)。

## AITER路由与SWA口径

non-SWA同输入BF16测试按长度显式选择并用profiler验证AITER入口：

| 条件 | 公开入口 | 实际事件 |
|---|---|---|
| `Q == KV` | `mha_batch_prefill_func` | `aiter::mha_batch_prefill` |
| `Q != KV` | `flash_attn_varlen_func` | `FlashAttnVarlenFunc` |

AITER等长D192/V128使用linear/page-size-1 ABI，非等长使用linear THD；8-wave使用
page64 vectorized cache。SWA上两种线性API都支持window与sink：
`mha_batch_prefill_func`命中`aiter::mha_batch_prefill`，
`flash_attn_varlen_func`命中`aiter::mha_varlen_fwd`。后者更快并覆盖128K。
真正direct-paged只有`mha_batch_prefill_func`具备ABI，但当前gfx950 BF16 D192/V128
page64构建没有匹配specialization。默认AITER回归为`3 passed`。

## 实现与优化

- 公开API始终运行原生512线程8-wave kernel，不再导入或调用4-wave后端；
- workgroup为`BM256 x BN32 x 512 threads`，8个wave分成两组交织QK MFMA与
  online softmax；
- 1 WG/CU persistent grid通过atomic ticket遍历`batch x query tile x Q head`任务；
- 原生gfx950 D192/V128 BF16为两个错相4-wave组分别使用两槽K LDS ring；K通过
  buffer-to-LDS direct DMA写入pair-padded视图，再由K16置换视图读取，V保持
  `global -> register`。两个组不能共享异步写入的ring，否则KV96起出现非确定覆盖；
- 原生BF16将P@V保留在MFMA stage，并显式拆成两个K16 atom组；目标ISA为233 VGPR、
  100 SGPR、49940 B LDS、2个SGPR spill、0 VGPR spill、0 scratch；目标specialization
  直接写VMEM，静态`S_BARRIER`
  从C-shuffle的46个降到31个；
- 32x32 MFMA减少row max/sum cross-lane操作，`v_permlane`避免经LDS交换；
- B@A `fx.gemm`使QK结果布局直接匹配P@V输入，K加载同时置换KV-length维；
- `-packed-fp32-ops`避免与MFMA有co-issue问题的packed FP32 VALU；
- `work_counter`按device/stream/grid复用并由最后退出的workgroup复位；
- lane 0通过独立4-byte LDS mailbox广播ticket，单barrier同时封闭上一work item的
  C-shuffle生命周期；
- 1/2/3/4 WG-per-CU sweep在B=4为295.49/298.49/300.95/302.72 us，保留1 WG/CU。

gfx950 OCP FP8 QK使用`v_mfma_f32_32x32x64_f8f6f4`和unity E8M0 scale；P@V
reduction只有32，保留K16 FP8 MFMA。Dqk不能被64整除时回退K16。

## 资源

gfx950 OCP FP8 D192 fresh ISA：

| 版本 | QK / P@V静态站点 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| K16 | 48 K16 / 32 K16 | 174 | 72 | 12292 B | 0 B | 0 | 0 |
| K64 | 12 K64 / 32 K16 | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64将QK静态MFMA站点减少4倍，仅增加2个VGPR。gfx950 BF16原生D192/V128 direct-LDS
specialization为233 VGPR、100 SGPR、49940 B LDS、0 private、2个SGPR spill、
0 VGPR spill、0 scratch。

## BF16 K/V LDS结果与TODO

- 历史register-to-LDS路径的自然padding写视图与K16置换读视图将K-only bank
  conflict从81.85%降到3.76%，目标shape独立A/B从约420 us降到397.71 us；
- V跨wave复用将`SQ_INSTS_VMEM_RD`从4.319M降到1.395M，但总LDS冲突为33.50%，
  K-only到K+V LDS从397.71 us回退到410.95 us；正式三方轮转协议为449.06 us；
- 历史共享K ring恢复direct-V并从三槽减到两槽后，LDS从62980 B降到25620 B、VGPR
  从241降到218；当前direct-LDS为保证两个错相4-wave组互不覆盖，改为独立双槽ring；
- 目标specialization用64-bit direct output store替代C-shuffle，输出最大差
  `0.0078125`，资源不变；三轮同进程A/B为409.14/407.66、412.17/411.33、
  412.32/411.79 us，对应稳定0.13%-0.36%提速；
- K预取改为`raw_ptr_buffer_load_lds`，每条64-lane DMA搬两个32-row D-group，每个
  pair后padding 16 B；K侧`ds_write_b128`静态站点归零。KV2583硬件计数为
  `SQ_INSTS_LDS=510528`、`SQ_LDS_BANK_CONFLICT=0`；双subgroup ring三轮为
  409.80/410.20/410.11 us；删除过时K地址重物化后的最终同轮中位为358.19 us。该改动减少K中转
  寄存器和DS write，但当前没有带来性能收益；
- 最终`MfmaUtil`为31.90%-32.56%，occupancy约2.00。自然V padding虽将冲突率降到
  0.37%，却使延迟升到749.39 us，因此冲突率不能作为V布局的单一验收指标。

后续TODO：

1. 将内部KV tile改为64并保留两个score fragment，使`QK(t)`与
  `softmax/P@V(t-1)`重叠，同时将softmax频率减半；外部page ABI继续保持BN32。
  已验证的BN32双score不会降低softmax频率，不能替代该改造。
2. 将8个wave拆成两组4-wave并错开一个pipeline stage；要求`MfmaUtil > 50%`、
   0 scratch，且不得恢复K-LDS冲突。

已否决实验：

- D4将两个子组的K ring拆成四个独立LDS字段：`391.55 -> 412.52 us`，约回退5.4%，
  VGPR从237升到240；与D5组合后为434.64 us，因此恢复单数组双ring；
- D5按4-wave顺序将K DMA移到K DS-read之后并移除稳态`vmcnt(0)`：与D4组合后
  `391.55 -> 434.64 us`，约回退11.0%。8-wave全WG barrier下反而损失DMA/softmax覆盖；
- D11/D12最初在K地址重物化workaround仍存在时分别回退3.6%/2.9%，组合也回退；
  删除该workaround后重新评估，显式max树、direct permlane32、无偏lazy-max和
  fused scale-sub将VGPR从235降至233，同进程`347.79 -> 344.87 us`，提速0.85%，
  因此当前已保留；
- D17用现有128-bit C-shuffle替换64-bit direct store：输出逐bit一致，但
  `371.33 -> 373.39 us`，回退0.55%且增加barrier，因此保留direct store；
- 将score scale或半个softmax跨barrier前移分别回退约5%-6%和约14%；
- 在同一BN32内把前8个EXP与第一组P@V MFMA交织，ISA生效但回退约1%；
- 删除任一全WG barrier可做到正确的lockstep版本，但因失去两组4-wave错相，回退
  约7%-13%；gfx950不支持gfx1250 named subgroup barrier；
- BN32双score一barrier原型完成了K-ring代次、ragged-tail和最终MFMA收口验证，目标
  输出与native baseline逐bit一致；但三轮为547.20/546.10/549.66 us，对照native
  390.53/389.99/391.89 us，回退约40%，因此移除。真正有价值的方向仍是KV64，
  同时降低softmax频率并设计两组4-wave错相，而不是只延长score生命周期。

stage边界修复分两步：先将目标BF16的8条P@V MFMA从`p0`移回`p1`，同进程
`370.25 -> 367.39 us`；再保留K-DMA在softmax前提前发射，仅将`setprio(0)`边界后移
到3条K-DMA之后，同进程`345.09 -> 342.53 us`，提速0.75%。直接把K-DMA物理移动到
K DS-read之后会失去异步覆盖并回退`326.43 -> 370.09 us`，因此未保留。最终动态ATT
为`p0 = 17 EXP`、`p1 = 20 MFMA + 12 DS-read + 3 K-DMA`，与4-wave职责一致；两阶段
中位分别为736/1020 cycles，两个4-wave子组继续反相执行。
粗粒度
`optimize_native_bf16`已删除，拆为`use_split_bf16_pv`与
`use_direct_output_store`；拆分前后纯指令序列SHA256一致。

K LDS读取原先用side-effecting `v_mov(tid)`阻止后端外提地址计算；当前LLVM删除该
workaround后只保留一个循环不变量DS-read基址`v100`，没有恢复旧版的12个长寿命地址，
VGPR从241降至235，private/scratch不变。同进程A/B为`378.55 -> 340.91 us`，提速
11.04%，KV96与目标shape输出逐bit一致；此时三方测量为8-wave `358.19 us / 756.15T`、
4-wave `304.95 us / 888.17T`。随后对齐4-wave VALU后，最终三方测量为8-wave
`352.09 us / 769.25T`、4-wave `304.99 us / 888.05T`。stage边界继续对齐后，最终
三方测量为8-wave `350.43 us / 772.89T`、4-wave `304.07 us / 890.74T`。证据：
`tests/flydsl/pa_8wave/ui_output_agent_65497_dispatch_24`。按每wave每BN32归一化，
softmax核心两边均为`18 FMA + 17 EXP + 16 ADD + 7 MAX3 + 2 MAX + 1 permlane + 1 SUB`；
含MFMA stage地址控制后，4-wave/8-wave总VALU分别为90.10/87.41。

当前源码SHA256：`160b38fcc74696fd05115d6ebfbe757b34592f53fde56c713b2e64b201bada7b`。

## 复现

```bash
python -m pytest -q tests/flydsl/pa_8wave/test_pa_prefill.py
python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_matches_dispatched_aiter \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_paths
PYHIP_RUN_PA_AITER_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_pa_aiter_production_performance -s
PYHIP_RUN_PA_AITER_SWA_PERF=1 \
  python -m pytest -q \
  tests/flydsl/pa_4wave/test_pa_prefill.py::test_swa_aiter_production_performance -s
```

