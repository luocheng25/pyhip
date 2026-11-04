# gfx942 FP8 block-scale MoE（prefill）：使用、集成与当前性能

本页只记录使用方法、sglang集成方式、当前有效性能和方案建议。实现细节、数据流和优化/回退历史见[源码readme](../../src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/readme.md)。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| 公开接口 `fused_moe_fp8_blockscale_942`（aiter fused_moe调用约定） | [moe.py](../../src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/moe.py) |
| gate_up / 短K down（I=256）/ dense down / inverse / sum五个FlyDSL kernel | [kernels.py](../../src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/kernels.py) |
| 实验：M64 gate_up与1-stage（gemm1 + requant + down，`PYHIP_MOE_DOWN=fused`，见4.15） | [kernels_m64.py](../../src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/kernels_m64.py) |
| 复用的256x256 8-wave FP8 block-scale流水线 | [gemm_fp8_blockscale_8w_942.py](../../src/pyhip/ops/gemm/flydsl/gemm_fp8_blockscale_8w_942.py) |
| 正确性测试（TP8/TP4/EP8、FP8输入、padding为主的小批次） | [test_moe_fp8_blockscale_942.py](../../tests/ops/moe/test_moe_fp8_blockscale_942.py) |
| 单卡性能对比（PyHIP vs aiter，可选per-kernel拆分） | [bench_moe_fp8_blockscale.py](bench_moe_fp8_blockscale.py) |
| 8卡通信方案（AR vs RS+AG vs FP8 AG） | [comm_bench.py](comm_bench.py) |
| TBO可行性（MoE与另一stream上的all-reduce并发） | [tbo_overlap.py](tbo_overlap.py) |
| sglang aiter runner分发检查（需sglang源码在PYTHONPATH） | [sglang_dispatch_check.py](sglang_dispatch_check.py) |
| HIP微基准：Y store模式、原子累加、MALL驻留、atomic/store与MFMA同CU并发（corun）；per-kernel PMC；down去store消融；有损传输的精度；融合sum + two-shot AR（sum_ar）、按列组的版本与CU-masked并发（sum_ar_cols、down_ar_overlap、cu_mask/residency）；down + sum + AR单kernel融合（down_ar_fused、down_ar_comm）及其atomic epilogue版本（down_atomic、execz_vmcnt）；整层对比（down_ar_fused `LAYER=1`、tq_layer_compare）；token分两半流水（halves_pipeline）；1-stage M64与2-stage的逐位对比和计时（moe64_check）；tingqli代码的验证与route-store补丁 | [probes/](probes/) |

目标为 **gfx942（MI300X/MI325X）、FP8 E4M3FNUZ权重与激活、128x128 block scale、BF16输出**。依赖torch(ROCm)、FlyDSL、aiter（复用`moe_sorting`和per-1x128输入量化）。验证环境：MI300X×8，ROCm 7.2.4，torch 2.11.0+rocm7.2，FlyDSL 0.3.2，aiter `2b81433bb`，Python 3.12。

本地未安装`pyhip.ops`时，在import前把源码目录加到`pyhip.__path__`（benchmark与测试已自动处理；sglang用`SGLANG_PYHIP_ROOT`）。修改源码后的实验设置`FLYDSL_RUNTIME_ENABLE_CACHE=0`，避免FlyDSL磁盘缓存返回旧kernel。

```bash
python -m pytest -q tests/ops/moe/test_moe_fp8_blockscale_942.py
python benchmarks/moe_fp8_blockscale/bench_moe_fp8_blockscale.py --check            # 单卡，默认tp8/tp4/ep8 x 4K-32K
python benchmarks/moe_fp8_blockscale/bench_moe_fp8_blockscale.py --configs ep8 --tokens 32768 --breakdown
torchrun --nproc_per_node 8 benchmarks/moe_fp8_blockscale/comm_bench.py 8192 32768   # 需8卡空闲
torchrun --nproc_per_node 8 benchmarks/moe_fp8_blockscale/tbo_overlap.py 16384 [--tp8]
python benchmarks/moe_fp8_blockscale/probes/pmc_kernels.py pyhip 32768             # 或aiter；需rocprofv3在PATH
python benchmarks/moe_fp8_blockscale/probes/corun_probe.py 32768 18000 0,16,12,8 [store]   # 见4.5；默认atomic
torchrun --nproc_per_node 8 benchmarks/moe_fp8_blockscale/probes/sum_ar.py 8192 16384 32768   # 见4.7，需8卡空闲
```

## 1. 接口

```python
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import fused_moe_fp8_blockscale_942

out = fused_moe_fp8_blockscale_942(hidden_states, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale,
                                   a1_scale=None, out=None, expert_mask=None)
```

| 参数 | 约定（与sglang aiter runner传给`aiter.fused_moe`的一致） |
|---|---|
| hidden_states | `[T, H]` BF16（内部用aiter per-1x128量化），或FP8 + `a1_scale`（字节布局为`[H/128, T]`，即aiter的transposed scale） |
| w13 / w2 | `[E, 2I, H]`（gate行在前）/ `[E, H, I]` FP8，aiter `shuffle_weight(16, 16)` |
| w13_scale / w2_scale | `[E, 2I/128, H/128]` / `[E, H/128, I/128]` FP32 |
| topk_ids / topk_weights | `[T, topk]` int32 / FP32（已归一化） |
| expert_mask | 可选，`[E_global]` int32；EP不走all-to-all时标出本rank的专家，w13/w2只含本地专家，输出只累加本地路由 |

限制：I % 128 == 0，H % 256 == 0，gfx942，BF16输出，SiLU gate-up；不支持bias、swiglu clamp、MXFP4权重。排序后行数×H×2超过4 GiB时（TP下约>60K token）自动按token二分执行。workspace按设备缓存、只增不减；CUDA graph capture前须用最大token数预热一次（prefill默认不capture）。

## 2. sglang集成

已接入sglang的aiter MoE runner（`python/sglang/srt/layers/moe/moe_runner/aiter.py`，`AiterRunnerCore.run`入口处，默认关闭），开关定义在`python/sglang/srt/environ.py`：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_USE_PYHIP_FP8_MOE` | 0 | 打开PyHIP FP8 block-scale MoE |
| `SGLANG_PYHIP_FP8_MOE_MIN_TOKENS` | 8192 | 单次MoE调用token数下限，少于时仍走aiter |
| `SGLANG_PYHIP_ROOT` | - | PyHIP源码checkout（扩展已安装的pyhip） |

满足以下全部条件才分发到PyHIP，否则原样调用aiter：`per_128x128`、SiLU、无bias/swiglu clamp/a2 scale、非FP4、权重已shuffle、**每rank intermediate ≥ 512**、token数≥下限、输入为BF16或已transposed的FP8 scale、gfx942、不在graph capture中。

**目标部署（[launch_mimo_v26_flash_dp_attention.sh](../../../sglang/evaluation/mimo_v26_flash/launch_mimo_v26_flash_dp_attention.sh)：`--tp-size 8 --dp-size 2 --ep-size 1 --enable-dp-attention`）的MoE是TP8，每rank intermediate=256。**经过4.1节的短K优化，PyHIP在该形状下与aiter持平（16K–64K为0.99–1.01x，8K为0.92x），没有收益，因此I=256仍走aiter，打开开关也不替换。开关只对I≥512（如TP4形状，1.02–1.06x）有效。

```bash
export SGLANG_USE_PYHIP_FP8_MOE=1 SGLANG_PYHIP_ROOT=/path/to/pyhip   # 只影响 I >= 512 的MoE
```

`--moe-a2a-backend none`时sglang的standard dispatcher对aiter runner传全局`topk_ids`（EP时另传`expert_mask_gpu`），本kernel支持这两种ABI。

已验证：[分发检查](sglang_dispatch_check.py)覆盖TP4 BF16输入、TP4 FP8输入（transposed scale）、EP8（expert_mask）走PyHIP，TP8 I=256和低于token下限回退aiter，输出与aiter的rel_l2为0.55%–0.58%。

## 3. 当前性能

2026-11-02，GPU0（MI300X，750 W功耗墙），测前8卡use均为0%。各方案（含1-stage、tingqli、8卡整层）的同日对比见4.16；在sglang新分支`feat/mimo-dpa-prefill-opt`上的端到端结果（PyHIP FP8 GEMM + 序列并行MoE）见4.17，之后的A2A o_proj、SWA序列并行等见4.20。每格为一次完整调用的持续循环均值（CUDA event，2 s），包含输入量化、排序、两个GEMM和top-k求和；两种实现使用相同输入与权重。TFLOPS按有效路由数计（不计padding行）；rel为与aiter输出的相对L2误差（与FP32参考的误差：PyHIP 0.57%–0.62%，aiter 0.68%–0.77%）。TP8为目标部署的形状；TP4、EP8只作为I≥512时kernel能力的参考，不是部署建议。

| 配置（每rank） | token | PyHIP µs | aiter µs | 加速 | PyHIP TFLOPS | aiter TFLOPS | rel vs aiter |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TP8：256专家，I=256（目标部署，11-03 compact M64尾块）** | 4096 | 613 | 613 | 1.00x | 336 | 336 | 0.40% |
| | 8192 | 938–950 | 903–918 | 0.96–0.97x | 434–440 | 449–456 | 0.40% |
| | 16384 | 1658–1669 | 1693–1721 | 1.01–1.04x | 494–497 | 479–487 | 0.40% |
| | 32768 | 3134–3158 | 3166–3168 | 1.00x | 522 | 521 | 0.40% |
| | 65536 | 6363 | 6353 | 1.00x | 518 | 519 | 0.40% |
| TP4：256专家，I=512（参考） | 4096 | 1198 | 1155 | 0.96x | 344 | 357 | 0.58% |
| | 8192 | 1754 | 1791 | 1.02x | 470 | 460 | 0.58% |
| | 16384 | 2980 | 3129 | 1.05x | 553 | 527 | 0.58% |
| | 32768 | 5431 | 5755 | 1.06x | 607 | 573 | 0.58% |
| | 65536 | 10909 | 11246 | 1.03x | 605 | 587 | 0.58% |
| EP8：32专家，I=2048，全局top-8（参考） | 4096 | 560 | 593 | 1.06x | 360 | 340 | 0.55% |
| | 8192 | 845 | 1034 | 1.22x | 483 | 394 | 0.55% |
| | 16384 | 1396 | 1818 | 1.30x | 593 | 456 | 0.55% |
| | 32768 | 2432 | 3464 | 1.42x | 680 | 478 | 0.55% |
| | 65536 | 4908 | 6763 | 1.38x | 674 | 489 | 0.55% |

32K token逐kernel GPU时间（torch profiler，µs）：

| 配置 | 输入量化(aiter) | 排序(aiter) | gate_up | down | inverse(含-1填充) | sum | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP8 PyHIP（v1，dense down） | 112 | 82 | 1358 | 1165 | 7 | 729 | 3453 |
| **TP8 PyHIP（短K down + nt sum）** | 112–178 | 82 | 1417 | 936 | 7 | 592 | 3160–3200 |
| **TP8 PyHIP（当前，+ compact M64尾块，tile表11 µs）** | 164–176 | 84 | 1361 | 901 | 8 | 597–608 | 3134–3158 |
| TP8 aiter（1-stage asm，含量化/排序） | 140 | 117 | 2865（gemm1+act+gemm2+atomic） | | | | 3122 |
| EP8 PyHIP（参考） | 131 | 141 | 1404 | 621 | 6 | 154 | 2457 |

MiMo 128K prefill的生产profile（dummy权重，每32K token一步）中，该aiter kernel为2783 µs/层；与上表的差来自数据相关的功耗（dummy权重功耗更低、时钟更高），比较时用同一份数据。MoE循环时GPU持续在750 W，SCLK约1150 MHz。

## 4. 方案建议（部署固定为DP2/TP8 DP-attention，2026-11-02重新评估）

前提：不改部署参数，attention为DP2×TP4，MoE为TP8（每rank I=256），每层的MoE调用处理DP gather后的全局token（chunk 65536时每DP组32K，全局最多64K）。每层GPU时间由三部分组成（32K全局token）：

| 部分 | 当前 | 本轮可得 | 说明 |
|---|---:|---:|---|
| MoE（aiter 1-stage） | 3.19 ms | 3.16 ms（PyHIP，≈持平） | 见4.1，TP8下kernel侧已接近2-stage的带宽下限 |
| 两次8卡all-reduce | 3.39 ms | 2.62 ms（−0.77 ms） | 见4.2，代码级改动，不改部署 |
| MoE入口的全token量化 | 0.11–0.18 ms | 0 | 随4.2的FP8 gather一并去掉 |

**结论：在固定部署下，收益主要来自通信路径（4.2），不是MoE kernel替换。**此前“改EP8 + PyHIP”的建议（MoE 1.30x）需要改`--ep-size`，不在本轮范围内，只作为参考数据保留在第3节。

### 4.1 1-stage还是2-stage（I=256）

| 方案 | 32K实测/推算 | 依据 |
|---|---|---|
| aiter 1-stage asm（现状） | 3.19 ms | gemm1+act+gemm2+BF16原子累加在一个kernel内 |
| PyHIP 2-stage（当前） | 3.16–3.20 ms（1.00–1.01x） | gate_up 1417 + 短K down 936 + sum 592 + 量化/排序约260；8K时0.92x |
| 2-stage的下限 | 约2.7 ms | Y写出+读回各2.15 GB（4.5 TB/s下约0.95 ms），gate_up执行行在功耗墙下约1.3 ms，其余0.3 ms |
| 1-stage（BF16 atomic代替Y+sum，含M64、每CU 2个WG） | 约3.16–3.25 ms（±0，见4.5） | 按行连续的`global_atomic_pk_add_bf16`上限1.52 TB/s（2.15 GB需1.41 ms），MFMA输出布局下0.36 TB/s，FP32原子0.70 TB/s；同CU另一WG做gemm1（同时读内存）时，atomic有70–77%无法掩盖 |

因此TP8下两条路径的上限相近，继续在MoE kernel上投入的空间约5–10%（compact M64尾块减少padding、量化前移）。**建议MoE保持aiter（sglang的分发门槛仍为I≥512）**，PyHIP的短K down与nt sum作为I=256的可用实现保留。

### 4.2 通信：从每层2次all-reduce到序列并行+FP8 gather（主要收益）

当前每层两次8卡all-reduce都作用于全局token缓冲（DP2的两组token拼接、未填部分为0）：attention之后`dp_gather_partial`（同时完成o_proj的TP4归约和DP gather），MoE之后的all-reduce（再`dp_scatter`取本组token）。MoE输入在DP-attention下是BF16（融合RMSNorm+量化只在`attn_dp_size == 1`时启用），因此还有一次全token的RMSNorm和量化在8张卡上重复执行。

8卡RCCL实测（[comm_bench.py](comm_bench.py)，`NCCL_MIN_NCHANNELS=112`，N为全局token，µs，取8卡最大值的中位数）：

| 集合通信 | N=8192 | 16384 | 32768 | 65536 |
|---|---:|---:|---:|---:|
| AR8 [N,H] BF16 | 468 | 884 | 1695 | 3318 |
| RS8 [N,H] BF16 | 275 | 456 | 826 | 1586 |
| AG8 [N,H] BF16 | 265 | 467 | 868 | 1673 |
| AG8 [N,H] FP8+scale | 165 | 271 | 477 | 887 |
| RS4 [N/2,H] BF16（两组并行） | 270 | 467 | 869 | 1671 |
| AG4 [N/2,H] BF16 | 245 | 434 | 813 | 1563 |
| AG4 [N/2,H] FP8+scale | 151 | 247 | 452 | 833 |
| A2A8 [N,H] int8+scale/32（代替RS8，接收方反量化求和，未计） | — | 310 | 528 | 985 |
| A2A4 [N/2,H] int8+scale/32（代替RS4） | — | 286 | 480 | 891 |
| **每层合计：当前 2×AR8** | 935 | 1767 | 3390 | 6635 |
| sglang `SGLANG_DP_HIERARCHICAL_COMM=1`（RS4+AG8+RS8+AG4，BF16） | 1055（1.13x） | 1824（1.03x） | 3376（1.00x） | 6493（0.98x） |
| **序列并行+FP8 gather**（RS4+AG8 fp8+RS8+AG4 fp8） | 861（0.92x） | 1440（0.82x） | **2624（0.77x）** | **4976（0.75x）** |
| 序列并行+FP8 gather+int8 RS（有损，见4.6） | — | 1131（0.64x） | 1939（0.57x） | 3587（0.54x） |

结论与建议方案：

1. **sglang现有的hierarchical开关在MI300X上无收益**：4卡子组的集合通信每卡只用到7条xGMI链路中的3条，传一半数据的RS4/AG4与8卡RS8/AG8同样慢。
2. **改为序列并行（SP）+ FP8 gather，通信−23%~−25%（不改部署参数，只改sglang的DP-attention通信路径）**：残差流按token切到8卡；归约用reduce-scatter（BF16，保持精度）；RMSNorm和量化只在本卡1/8 token上做；消费者（MoE和QKV GEMM）本来就吃FP8，所以all-gather传FP8+1x128 scale（字节约为BF16的52%）。附带去掉8卡重复的全token RMSNorm、MoE入口单独的量化kernel（112–178 µs/层），以及DP gather的fill/memcpy（profile中约114 ms/step）。32K token每层省约0.77 ms（加上量化约0.9 ms），47层约36–42 ms/step。
3. **MoE并行方式不影响这部分通信量**：tokens已经在每张卡上（DP gather），MoE输出归约是同一个8卡缓冲。对MiMo（top-8、8卡）而言，DeepEP/MORI式all-to-all的dispatch字节≈FP8 all-gather，combine≈BF16 reduce-scatter，没有量上的优势，只增加负载不均和依赖，不建议。
4. **con=1长上下文**：只有一个DP组有token时，全局缓冲的AR有一半是0；SP+gather只搬真实token。单请求128K prefill下attention占比50%，DP2只用一半attention算力。
5. 次数层面：每层边界仍是一次RS和一次AG（共4次集合），无法再少，因为归约之后必须先做norm+quant，才能gather。进一步可把“RS→norm→quant→AG”做成一个基于IPC buffer的自定义kernel，去掉RCCL占用的CU（112 channel）和kernel间隙。
6. **不改代码的无损选项：大消息的AR改走QuickReduce FP。** 当前启动脚本`unset ROCM_QUICK_REDUCE_QUANTIZATION`，QuickReduce处于关闭状态，prefill的AR（≥64 MiB）全部走RCCL；aiter custom AR默认只处理≤64 MiB（`AITER_CUSTOM_AR_MAX_SIZE`）。

   8卡实测（[ar_large_bench.py](probes/ar_large_bench.py)，[N, 4096] BF16，最慢rank的中位数，µs；rel_l2为与FP32参考的误差）：

   | 全局token（大小） | RCCL | aiter CA（上限调到512 MiB） | QR FP（BF16） | QR FP（转FP16） |
   |---|---:|---:|---:|---:|
   | 8192（64 MiB） | 487 | 467 | 483 | 499 |
   | 16384（128 MiB） | 928 | 977 | 892 | 904 |
   | 32768（256 MiB） | 1782 | 1891 | 1697 | 1695 |
   | 65536（512 MiB） | 3541 | 3760 | 3288 | 3262 |
   | rel_l2 | 3.68e-3 | 3.21e-3 | 3.68e-3 | 1.74e-3 |

   - 设`ROCM_QUICK_REDUCE_QUANTIZATION=FP`（不量化）和`ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0`：误差与RCCL相同，128–512 MiB快4%–7%。
   - sglang先试aiter CA（≤64 MiB），再试QR（BF16、8卡时≥16 MiB），所以64 MiB以下的路径不变。
   - 转FP16的版本误差更小，但值超过65504会溢出；在确认激活范围之前不用。
   - aiter CA把上限调大后，128 MiB以上反而比RCCL慢5%–6%，保持默认64 MiB。
   - 自研two-shot（sum_ar的只AR模式，FP32按rank顺序累加，4.7）约为RCCL的0.92x，与QR FP相当；它的价值在于与sum融合（0.80x）。

### 4.3 如何验证达到硬件极限

本机实测上限（单卡，750 W）：MFMA FP8纯计算1724 TFLOPS（1443 MHz，748 W，所有CU 2 wave/SIMD，`mfma_power.py`）；功耗墙下带数据搬运的GEMM实测上限约1040 TFLOPS（dense 8-wave kernel，K=4096）；HBM读4.28、copy 3.73、写5.06 TB/s；同一缓冲写后立即读4.44 TB/s（≤256 MB时4.78）。MoE整链运行时GPU在750 W墙上，SCLK约1150 MHz。

| 方法 | 用途 |
|---|---|
| 逐kernel roofline：`max(有效FLOPs/峰值, 最少字节/带宽)` | TP8 32K：Y写+读4.3 GB→≥0.95 ms（down+sum实测1.53 ms）；gate_up执行行1.23 TFLOP→功耗墙下约1.3 ms（实测1.42） |
| 有效 vs 执行FLOPs | TFLOPS只按有效路由计；padding（256行块，平均每专家128行）单列，不能用执行FLOPs抬高结论 |
| 功耗与时钟（`amd-smi metric -p -c`持续采样，PMC `GRBM_GUI_ACTIVE`/kernel时长） | 所有FP8 GEMM都在750 W墙上，用1443 MHz峰值估算会高估约25% |
| PMC（rocprofv3：`SQ_VALU_MFMA_BUSY_CYCLES`、`SQ_INSTS_VALU/MFMA`、`SQ_WAIT_INST_ANY`、`TCC_EA0_WRREQ`） | MFMA忙比例、VALU/MFMA比、等待、写请求粒度（短K down：MFMA忙50%，去store后67%） |
| 消融（编译期开关去掉store/退休/同步/DMA） | 定位瓶颈：短K down去store 936→约600 µs，去store+退休+同步约500 µs |
| 微基准（HIP） | store模式（半行2.7 vs 整行4.5 TB/s）、原子累加（1.53 TB/s）、MALL驻留（≤8%）、atomic与MFMA同CU并发（4.5）、通信（[comm_bench.py](comm_bench.py)，32K时8卡AR/RS/AG的busbw约280/285/270 GB/s） |

TP8、32K token、单卡的per-kernel PMC（[pmc_kernels.py](probes/pmc_kernels.py)，rocprofv3下kernel串行；时钟=`GRBM_GUI_ACTIVE`/8/时长，MFMA忙=`SQ_VALU_MFMA_BUSY_CYCLES`/(304×4×周期)，读写为L2之外的字节）：

| kernel | 时长 | 时钟 | MFMA忙 | TFLOPS（有效/执行） | 读 | 写 | 判定 |
|---|---:|---:|---:|---|---|---|---|
| gate_up | 1397 µs | 1.12 GHz | 63% | 787 / 882 | 2.04 GB（1.46 TB/s） | 0.08 GB | 计算受限：执行TFLOPS为功耗墙下GEMM上限1040的85%；其余为padding（11%）和7.55轮任务的尾部（约6%） |
| down（短K） | 934 µs | 1.10 GHz | 48% | 589 / 660 | 0.49 GB | 2.15 GB（2.30 TB/s） | 处在拐点（执行233 FLOP/B）：MFMA约600 µs、Y写出约480 µs，只重叠约1/3 |
| sum | 547 µs | 1.53 GHz | 0 | — | 2.15 GB（3.93 TB/s） | 0.27 GB | 带宽受限：读+写2.42 GB，4.4 TB/s，达到HBM读上限 |
| aiter 1-stage（参考） | 2905 µs | 1.37 GHz | 34% | 568 / 585 | 2.90 GB（1.00 TB/s） | 2.15 GB（BF16 atomic） | atomic 3.36e7次×64 B，全部在L2之外执行（`TCC_EA0_ATOMIC`=`TCC_ATOMIC`）；受atomic与读的争用限制（4.5） |

三者各自的下限之和约2.45 ms（gate_up约1.3 ms、down约0.6 ms、sum约0.55 ms），实测2.88 ms；差距主要来自down的写出与MFMA重叠不足（约0.33 ms）。

### 4.4 与TBO的关系

TBO（two-batch overlap）用两个micro-batch和多stream，把一个batch的通信藏到另一个batch的计算后面。它与“gemm2+all-reduce融合”（mega_moe_tp）针对的是同一段通信时间，**二者是重复的：选其一，不叠加**。通信减量（4.2）与两者正交，可叠加。

在MI300X上实测（[tbo_overlap.py](tbo_overlap.py)，8卡，16K token micro-batch，RCCL AR [16K, 4096] BF16放在第二个stream上与MoE并发）：

| MoE实现 | MoE单独 | AR单独 | 串行 | 并发 | 被隐藏的AR |
|---|---:|---:|---:|---:|---:|
| TP8 aiter（目标部署） | 1830 µs | 857 µs | 2515 µs | 2920 µs | 0%（慢16%） |
| EP8 PyHIP（参考） | 1514 µs | 859 µs | 2360 µs | 2434 µs | 0%（并发比串行慢3%） |

RCCL kernel占用CU，与功耗墙下的GEMM争用，stream间几乎没有真正的并发（本轮down/sum双stream切片同样变慢）。TBO还会把每个GEMM的批次减半（TP8：16K 0.48 vs 32K 0.52 PFLOPS有效）。**建议prefill不开TBO**。要做重叠，应在kernel内部完成（持久化kernel划分CU给通信，或通信融合进GEMM epilogue），并且应在4.2把通信量降下来之后再评估。

### 4.5 评估：M64 + 1-stage atomic + 每CU 2个WG（tingqli方案）

参考[tingqli/aiter doc-mimo](https://github.com/tingqli/aiter/tree/tq/mimo-moe/doc-mimo)（A8W4，MI325X，TP8 I=256）。其中`fmoe_bm64`是BM=64的单kernel：gemm1 → 寄存器内量化 → gemm2 + BF16 atomic，256 VGPR、25 KB LDS、每CU 2个CTA。8K/16K token比他的2-kernel快18%/17%，但atomic仍多花140–365 µs。`moe_2k_vs_1k`：I=256、≥4096 token时快4–15%；I=512只能每CU 1个CTA，没有收益。两个要点分开评估：

**(1) M64减少padding：成立，且与是否1-stage无关。** 本bench路由（top-8，256专家），用`moe_sorting`实测执行行数：

| 全局token | 真实路由 | M256执行行 | M64执行行 | M64减少 |
|---:|---:|---:|---:|---:|
| 8192 | 65536 | 96000（+46.5%） | 73152（+11.6%） | −23.8% |
| 16384 | 131072 | 162304（+23.8%） | 138880（+6.0%） | −14.4% |
| 32768 | 262144 | 293888（+12.1%） | 270144（+3.1%） | −8.1% |
| 65536 | 524288 | 557568（+6.3%） | 532288（+1.5%） | −4.5% |

纯M64 tile的W复用只有M256的1/4，每个M64块都要再读一遍2 MB的W13[e]。因此在2-stage中用compact任务实现：专家的满256行仍走M256 tile，只把尾部按64行处理（参考[moe_gemm_2stage的compact实现](../../src/pyhip/ops/moe/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py)）。32K时gate_up与down的MFMA量减少8.1%，约−0.15 ms（−5%）；16K约−0.16 ms。

**(2) 1-stage + BF16 atomic + 每CU 2个WG：本场景（FP8权重、MI300X、16K–64K token）预计无收益。** 方案的前提是：一个WG做atomic时，同CU的另一个WG在做gemm1，从而掩盖atomic。用[corun_probe.py](probes/corun_probe.py)直接测这个前提。每CU恰好2个WG：一个做MFMA，可选同时读内存（每wave 4个读在途）；另一个把全部真实路由按行连续用`global_atomic_pk_add_bf16`累加到OUT。表中为“均匀”模式，即atomic均匀分布在MFMA WG的整个时长内，对应1-stage的稳态。32K token，atomic共2.15 GB，在1.52 TB/s下需1413 µs：

| MFMA WG同时的读 | MFMA单独 | 加均匀atomic后 | atomic暴露 | 暴露比例 |
|---|---:|---:|---:|---:|
| 无 | 1571 µs | 1910 µs | 339 µs | 24% |
| 1.40 GB（约0.8 TB/s） | 1679–1813 µs | 2556–2609 µs | 795–877 µs | 56–62% |
| 1.87 GB（约1.0 TB/s） | 1715–1866 µs | 2670–2769 µs | 903–955 µs | 64–68% |
| 2.80 GB（约1.5 TB/s） | 1767–1984 µs | 2859–3026 µs | 1041–1093 µs | 74–77% |

（每行两个值分别为读1 GB流式缓冲和读常驻MALL的128 MB缓冲；16K token时同样的读速率下暴露51–64%。）

- 只有另一个WG不访存时，atomic才大部分被掩盖（76%）。gemm1必须持续读W和A：PyHIP的gate_up+down共读2.5 GB，aiter 1-stage读2.9 GB，约1.3–1.5 TB/s。这时atomic有70–77%（约1.0 ms）无法掩盖。
- 读常驻MALL与读HBM的结果相同，每wave在途读1个或4个的结果也相同。所以这不是HBM带宽或延迟问题，而是atomic与读请求在fabric/EA上的吞吐争用：atomic全部在L2之外执行。每CU 2个WG能掩盖延迟，不能消除这部分争用。
- 交叉验证：tingqli的FP4权重读量约为FP8的一半，他测得的atomic暴露（8K/16K下占atomic时间的39–51%）与本探针在0.7–0.9 TB/s读速率下的51–58%一致。
- aiter现有的`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_64x256`就是M64+BF16 atomic，但512 VGPR、64 KB LDS，每CU 1个WG：2905 µs，MFMA忙34%（4.3）。

32K估算。执行行按M64；1-stage的计算效率按PyHIP当前gate_up/down实测，偏乐观：

| 方案 | gate_up | down | Y+sum / atomic | 其余 | 合计 |
|---|---:|---:|---:|---:|---:|
| 当前2-stage（M256） | 1397 | 934（MFMA约600，写出暴露约330） | sum 547 | 约260 | 约3.14 ms（实测3.16） |
| 2-stage + compact M64尾块 | 约1284 | 约885 | sum 547 | 约270 | 约2.99 ms（−5%） |
| 1-stage + atomic + 每CU 2个WG | 约1284 | 约551（只算MFMA） | atomic暴露约1000–1090，输出清零约55 | 约260 | 约3.15–3.25 ms（±0） |

其余风险：

- 资源：每CU 2个WG要求每wave ≤256个VGPR（含AGPR）、每WG ≤32 KB LDS。FP8的W13 tile（N512×K128，每K步64 KB）是FP4的2倍，放不进LDS，只能直接进VGPR。blockscale每个K128还要单独的partial累加再按scale折叠，寄存器更紧。tingqli在FP4下已经遇到spill（用`waves_per_eu`强制提高占用率时慢约2倍）。
- 数值：BF16 atomic每条路由舍入一次（top-8共8次），累加顺序不确定，结果不可复现。aiter 1-stage的rel_l2为0.68–0.77%，PyHIP为0.57–0.62%。

**结论：采纳(1)，在2-stage中做compact M64尾块；不做(2)。** 本实现的store+sum代价（down写出暴露约0.33 ms + sum 0.55 ms ≈ 0.88 ms）与1-stage的atomic暴露（约1.0 ms）相当。

曾经的解释“tingqli的1-stage把gemm2的atomic藏到了gemm1下面、store+sum没有这段损失”经本机实测是错的。在MI300X上跑他的代码（`tq/mimo-moe`@`8f43399`，A8W4，TP8 I=256，含sort+quant，µs）：

| tokens | 2-kernel atomic | 2-kernel store+sum | 2-kernel不写回 | 1-stage | 1-stage不写atomic |
|---:|---:|---:|---:|---:|---:|
| 8192 | 909 | 1052 | 768 | 794 | 639 |
| 16384 | 1684 | 1950 | 1375 | 1471 | 1218 |
| 32768 | 3269 | 3763 | 2579 | 2901 | 2093 |

- store+sum（他的`OUT_MODE="sorted"`）比atomic更慢：相对不写回多花284/575/1184 µs，约为atomic（141/309/690 µs）的2倍。sum要把top-k份数据再读一遍，且无法与GEMM重叠。他自己的结论也是“保留atomic”。
- 1-stage里atomic同样多花155/253/808 µs，并没有被藏起来。1-stage的收益来自GEMM部分：不写回时为2093 µs，2-kernel为2579 µs。原因是A2不落地、M64少padding、按XCD连续映射让同专家权重在L2复用。
- 因此对本实现的启示只有一点：得在GEMM侧（M64尾块、XCD映射），而不在写出方式。脚本：[tq_store_sum_check.py](probes/tq_store_sum_check.py)（需要他的fork）。

**(3) 1-stage + 普通store（Y写出与另一个WG的gemm1并行，sum保留）：也不能解决。** 先定位down写出的代价：把short-K down的行数参数T传0，所有store越界丢弃，MFMA工作不变（[down_nostore.py](probes/down_nostore.py)）：

| down（32K） | 时长（连续运行 / rocprofv3） | 时钟 | MFMA忙 |
|---|---:|---:|---:|
| 正常 | 912–920 / 875 µs | 1.23 GHz | 46% |
| store全部丢弃 | 608–632 / 570 µs | 1.29 GHz | 67.5% |

写出使down多花约300 µs。其中时钟只差5%，主要是store的回压卡住了MFMA发射。推测机制是：gfx942的vmcnt把读写按发出顺序一起计数，等DMA时要连带等它之前的store（zhaobo的gfx942移植版R29在GEMM2里观察到同一现象）。接着用[corun_probe.py](probes/corun_probe.py)的`store`模式模拟1-stage：另一个WG在整个MFMA时长内均匀写Y（2.15 GB，单独写约470 µs，4.5 TB/s），时钟用kernel内`s_memtime`/`s_memrealtime`测持续值：

| MFMA WG同时的读 | MFMA单独 | 加均匀store后 | store暴露 |
|---|---|---|---:|
| 无 | 1590 µs，1.60 GHz | 1787 µs，1.44 GHz | 197 µs（42%） |
| 1.40 GB（常驻MALL / 流式） | 1686–1725 µs，1.55–1.60 GHz | 2107–2209 µs，1.29–1.37 GHz | 420–483 µs（90–101%） |
| 2.80 GB | 1803–1999 µs，1.41–1.56 GHz | 2248–2512 µs，1.24–1.41 GHz | 445–513 µs（94–108%） |

rocprofv3确认加store前后MFMA忙不变（89% → 89%，86% → 86%），只是时钟下降。也就是说，放在另一个WG的store确实不再卡MFMA发射，但750 W功耗墙下，写HBM的能耗只能靠降频支付，重叠省不下能量。gemm1同时在读内存时，store几乎全部暴露（约450 µs），反而比现在down里的约300 µs更多。atomic的情况不同：时钟基本不变（1.50 → 1.48 GHz），损失来自fabric争用。结论：down的写出问题不在重叠方式，而在字节数和功耗。可行的方向是：减少Y的字节（4.6的int8 Y），或在down内拆开store与DMA等待的耦合（例如只由一个wave发DMA且不发store），后者最多收回约300 µs中的发射停顿部分。

### 4.6 评估：aiter PR #6514（mega_moe_tp的big-M路径）

[PR #6514](https://github.com/ROCm/aiter/pull/6514)（已合入，本地见`aiter_moe308`的`aiter/ops/flydsl/kernels/mega_moe_tp/`）把整个TP MoE层连同通信做成一个常驻kernel，每CU一个CTA。大batch（超过本地约2048 token）走big-M路径，按token分块流水，每块依次是：

1. SDMA gather输入；
2. kernel内做路由规划；
3. 单元（专家×128行）内完成GEMM1 + 激活 + 量化 + GEMM2；
4. route rows用store写出，可选MXFP8（`R8`）；
5. kernel内top-k求和，跨XCD可见性靠每个单元的`buffer_wbl2 sc1`和`sc1`读；
6. MoE输出的reduce-scatter以int8+FP32 scale/32（可选int4，`E4`）经SDMA发送；
7. 尾部融合：残差 + 下一层RMSNorm + FP8量化 + FP8 all-gather。

限制：只支持gfx950（`mega_moe_tp_supported`），按160 KB LDS设计，权重为MXFP4，big-M还需要带CCO SDMA的MORI，且不能graph capture。调优表中MI355X TP4、32K token的整层时间（含通信）为2.9–4.7 ms。

本机`/dev/shm/zhaobo/mimo`已有人将其移植到gfx942，并支持MiMo的FP8 128×128 blockscale，共23版补丁，记录在`megamoe_issues.md`。big-M路径未移植；移植的是LB路径。部署口径（TP8 MiMo、调优后的aiter基线）的结果：

- 32K下整层5445 µs，基线（fused MoE + AR）4858 µs，慢约12%。
- GEMM部分比asm fmoe慢约36%（4634 vs 3406 µs），通信尾部反而快（770 vs AR 1635 µs）。
- 他也试过用atomic代替route rows，慢了5–13%，与4.5的结论一致。

逐项看能否在gfx942、sglang DP2/TP8下借鉴（精度为本bench TP8 4K token的模拟，[transport_numerics.py](probes/transport_numerics.py)；通信为[comm_bench.py](comm_bench.py)的all-to-all代理）：

| PR中的做法 | gfx942可用性 | 本机实测/估算 | 建议 |
|---|---|---|---|
| SP输入/输出 + 尾部融合（RS → 残差+norm+FP8量化 → FP8 AG） | 可用，纯计算+集合通信 | 即第5节优先级1：32K时每层通信 3390 → 2624 µs | 采纳（与现有方案一致） |
| RS载荷压缩为int8 + FP32 scale/32 | 可用，纯VALU | 4.2：再降到 1939 µs（−0.7 ms/层，发送方量化若不融合进上游约+0.1 ms/次）。8卡求和rel_l2：BF16 ring 0.71% → int8 0.83% | 候选，有损，需确认并做模型精度验证 |
| RS载荷int4（`E4`） | 可用 | rel_l2 9.7% | 不采纳 |
| route rows用MXFP8（`R8`） | 可用 | MoE输出rel_l2 0.60% → 2.73% | 不采纳；改用int8 + FP32 scale/32：0.81%，字节为BF16的56%，预计down+sum −0.25~0.3 ms（待实现验证） |
| SDMA拷贝引擎做AG（不占CU） | 本机MORI未编译CCO SDMA（无`libmori_cco_device.bc`），gfx942未验证；SDMA不能做归约 | 未测 | 先测SDMA带宽；只用于AG，要配合分块流水才有重叠 |
| 128行单元 / 按token分块流水 | 128行在gfx942的64 KB LDS下放不下（zhaobo估算）；64行可行 | 与4.5(1)的M64一致 | 只取M64尾块 |
| 整个常驻mega kernel | 需重写（gfx950专属指令、LDS、MXFP4） | zhaobo移植版比基线慢约12% | 不采纳 |

### 4.7 gemm2/sum + all-reduce融合

**tingqli的gemm2 + two-shot AR融合**（`doc-mimo/kernels/gemm2_ar_fused.py`）。comm WG放在gemm2的grid里，gemm2按列chunk顺序完成，每个chunk完成就开始two-shot（IPC push、写穿、单调epoch标志）。他只测到8K，本机8卡按TP8 I=256补测大token（`CHUNK=256 CWG=64`，结果正确，µs）：

| tokens | gemm2 | 自研two-shot | 依次执行 | 融合 | AR被隐藏 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 526 | 466 | 981 | 727 | 54% |
| 16384 | 996 | 852 | 1878 | 1393 | 57% |
| 32768 | 1957 | 1637 | 3691 | 2788 | 55% |

**本实现的对应做法：sum + two-shot AR融合**（[sum_ar.hip](probes/sum_ar.hip)、[sum_ar.py](probes/sum_ar.py)）。本实现的MoE输出由sum kernel产生，所以把two-shot融进sum。结构：

- 常驻WG（512线程）内按wave分角色。4个sum wave只读Y、做top-k求和，写入LDS环；4个push wave从环里取数据，用sc0|sc1写穿推给owner rank。这样sum wave从不发global store，Y的读不会在按序的vmcnt里排在远端store后面。
- RS完成后全体wave做owner归约（按rank顺序fp32求和，各rank结果逐位相同），再push给所有rank。
- 最后等全部行到齐，做grid barrier后由每个XCD做一次system acquire（失效L2）。

3个规模、多个epoch都与参考逐位一致。结果（µs，取8卡最慢）：

| tokens | sum | RCCL AR | sum + RCCL | 自研two-shot（同一kernel只做AR） | **融合sum+AR** |
|---:|---:|---:|---:|---:|---:|
| 8192 | 151 | 472 | 597 | 432 | **465（0.78x）** |
| 16384 | 284 | 873 | 1141 | 806 | **890（0.78x）** |
| 32768 | 620 | 1686 | 2213 | 1563–1587 | **1771（0.80x，−0.44 ms）** |

开发中的两个坑：

1. **vmcnt串行化。** 第一版一个wave既读Y又远端写，编译器在运行时循环入口插了`vmcnt(0)`，每行都要等上一行的远端store确认。展开循环、按wave分角色解决。
2. **L2外的读与xGMI写争用fabric。** 32K、grid 304时：单独sum 557 µs，单独远端push 878 µs，两者同时却要用到2771 µs。把Y的工作集缩到L2内（2 MB）后融合即为1901 µs，缩到128 MB（常驻MALL）仍是3560 µs。解决办法是限流：grid 304/152/96/76/64/38对应3629/2713/1771/1775/1854/2497 µs。tingqli的gemm2主要读L2，所以没有遇到这个问题。

结论：

- 对本实现，融合点应选sum而不是down。融合sum + AR在32K下每层省约0.44 ms，其中two-shot本身比RCCL快约0.1 ms，其余来自sum被掩盖66–70%。
- 把AR进一步藏到down下面：双stream、CU分区的做法（4.8）没有收益；单kernel内按wave分工的做法（4.10）有小幅收益，为0.92–0.97x。
- 与第5节优先级1的关系：改成序列并行后，MoE之后只需RS，不再需要AR。同样的结构去掉AG半段就是“sum + RS”，仍然适用（未测）。要接进sglang，还需要处理对称内存的分配、CUDA graph下的epoch，以及与DP gather缓冲的对接。

### 4.8 把AR藏到down的计算下：不可行（争用）

做法（[down_ar_overlap.py](probes/down_ar_overlap.py)、[sum_ar_cols.hip](probes/sum_ar_cols.hip)）：

- down的4个1024列组分成4次启动（现有kernel的`CG0`/`GROUPS`参数，不改FlyDSL代码），放在CU-masked stream上。
- 另一个互不相交的masked stream上，`sum_ar_cols`等第g组的event后，对这1024列做sum + two-shot AR。
- 普通的双 stream基本没有并发，所以用`hipExtStreamCreateWithCUMask`硬分CU。实测mask的bit i对应XCD i % 8，连续bit段在各XCD间均衡；两个互斥mask的kernel可以真正并发（[cu_mask_probe.py](probes/cu_mask_probe.py)）。

结果正确，与参考逐位一致。但在所有CU划分下都比依次执行慢（32K，µs）：

| comm CU / grid | down（4次启动，mask后） | 通信（mask后） | 重叠 | 无依赖并发（上限） | down + 融合sum_ar（全GPU依次） |
|---|---:|---:|---:|---:|---:|
| 48 / 32 | 1253–1329 | 3058–3066 | 4153–4177 | 4028 | 2794–2798 |
| 96 / 64 | 1322–1336 | 2058–2073 | 3213–3225 | 3071 | 2803–2807 |
| 144 / 96 | 1554 | 1812 | 3340 | — | 2839 |

- 即使完全去掉依赖，并发也只把1336 + 2058 = 3394 µs压到3071 µs，仍慢于全GPU依次执行的约2800 µs。说明损失不在依赖等待，而是争用：down要写Y（2.15 GB），通信要读Y并做xGMI推送，两者在HBM/fabric和功耗墙上互相拖慢。把down限制在部分CU上也让它变慢了约40%。
- 同样的通信与gate_up无依赖并发时却有收益：gate_up单独1654 µs、通信单独2058 µs（都在mask后），并发2811 µs；全GPU依次执行是1397 + 1771 = 3168 µs。gate_up是计算密集、读L2为主，与通信抢的资源少。所以更有希望的是“token分两半、前一半的sum+AR藏到后一半的gate_up下”，上限约省 0.36 ms；尚未实现。
- 踩到的坑：masked queue上，这个8-wave、172 VGPR的通信kernel只有约2/3的CU能同时常驻。例如在 48 CU的mask上，grid 为 48 时WG只落到34–42个CU，部分WG要等别的WG退出才启动；而WG之间又互相等待，结果死锁。同样资源的空转kernel却能全部常驻（[residency_probe.py](probes/residency_probe.py)）。grid取CU数的2/3即可正常工作；所有自旋等待都加了20 s超时，避免挂死GPU。

### 4.9 尝试：去掉atomic的1-stage（route store + 融合sum+AR）

4.5说“1-stage并没有把atomic藏起来”，指的是tingqli的实际实现：用`G2_NO_ATOMIC`开关对比，atomic在他的1-stage里仍多花155/253/808 µs（8K/16K/32K）。他自己也试过错开相位、工作队列混合，都无效。

这里试了一种他没有的组合：1-stage不再做atomic，而是把每条route行用普通store写到它的sorted位置。做法是给他的fused kernel加了实验开关`FMOE_ROUTE_STORE`（[tq_fmoe_route_store.patch](probes/tq_fmoe_route_store.patch)）。top-k求和交给`sum_ar`在通信里完成。结果正确（与atomic版本的cos diff为1e-5量级）。8卡，含sort+quant，µs（[tq_1stage_route_ar.py](probes/tq_1stage_route_ar.py)）：

| tokens | 1-stage（atomic） | 1-stage（route store） | atomic + RCCL AR | atomic + two-shot | route store + 融合sum+AR |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1017 | 895 | 1560 | 1529 | 1479–1506 |
| 16384 | 1587 | 1537 | 2528 | 2504 | 2597–2614 |

去掉atomic后1-stage的确快了50–122 µs，但通信那边多出了99–143 µs：route行写出又读回的字节，与xGMI推送争用。整体基本持平（−3% / +4%）。与4.5中atomic、store都受争用和功耗限制的结论一致：MI300X上这部分字节不管放在哪个阶段都要付出代价。

### 4.10 down + sum + AR融合成一个kernel（mega_moe_tp式wave分工）

做法（[down_ar_fused.py](probes/down_ar_fused.py)、[down_ar_comm.hip](probes/down_ar_comm.hip)）：一个常驻kernel，计算wave与通信wave分享同一批CU。

- WG为256线程，每CU 2个。down的计算部分直接复用`_down_direct_body`，它需要206 VGPR和32 KB LDS，所以每CU最多放2个4-wave的WG。gfx942的`s_barrier`作用于整个WG，计算wave要用它同步W2的LDS环，所以通信wave不能放进计算WG（barrier次数对不上会死锁），而是单独成WG，与计算WG同处一个CU。
- 角色在运行时分配：CU上先到的WG做计算（4个down wave）；后到的WG在本XCD的配额（NCOMM/8）内做通信（2个sum wave + 2个push wave），超出配额的也做计算。
- 计算：任务是(列组, M-block)，列组优先，每个XCD一个原子队列，同expert的W2仍在本XCD的L2里复用。XCD上某列组的最后一个任务做一次agent-scope release（`buffer_wbl2 sc1`，写回整个XCD的L2），再给ready[列组]加1。
- 通信：某列组在8个XCD都完成后，按`sum_ar_cols`的方式做这一组的top-k sum + two-shot AR（RS推给owner；owner按rank顺序fp32求和后AG推回各卡），同时计算WG继续算后面的列组。
- 实现：计算部分是FlyDSL，通信部分是HIP C++，编译成LLVM bitcode后用`flydsl.compiler.extern_link.link_extern`链接进同一个kernel并内联。

结果与参考逐位一致（8卡取最慢rank的中位数，µs；列组512列、通信WG 160个为最优配置）：

| tokens | down + sum + RCCL | down + 融合sum_ar（依次） | **单kernel融合** | 比依次 | 融合kernel只算 / 只通信 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 946 | 841 | **788** | 0.94x | 399 / 505 |
| 16384 | 1750 | 1519 | **1394** | 0.92x | 635 / 966 |
| 32768 | 3299 | 2865 | **2786** | 0.97x | 1197 / 1838 |

多次运行的范围：8K为782–797 vs 832–845，16K为1385–1394 vs 1462–1519，32K为2744–2801 vs 2797–2865。

收益不大的原因可以拆开看（32K，同一配置的探针）：

| 通信做什么 | 与down并发 | 通信单独 |
|---|---:|---:|
| 正常：读Y做top-k sum + AR，按列组等down | 2786 | 1838 |
| 只做AR（输入是现成的partial，不读Y），down写Y | 2340 | 1672 |
| 只做AR，down不写Y | **1949** | 1667 |
| 读Y做sum + AR，不等down（只为测时间） | 2684 | 1780 |

- xGMI通信本身几乎可以完全藏到不写Y的down下面：1949 µs只比通信单独多约280 µs。
- 加上down写Y（2.15 GB）多出约390 µs，再加上sum读Y（2.15 GB）又多出约350–450 µs。也就是说，限制重叠的是Y的写出与读回，不是xGMI或MFMA。这与4.5、4.8、4.9的结论一致。

调优中的发现（32K）：

| 选项 | 结果 |
|---|---|
| 每个任务都做release（`buffer_wbl2`） | 只算1063 → 1356 µs（+290）；改成每个XCD每个列组一次 |
| Y用sc1写穿、不做wbl2 | 只算1198–1298 µs，也比每XCD一次wbl2慢 |
| 通信WG数（列组512列） | 96/128/160/192/224/256/304个：3197/2930/2744–2790/2854/3161/3159/3140 µs。更多的通信WG让通信本身变慢，与`sum_ar`的grid扫描一致（L2之外的读与xGMI写在fabric上争用） |
| 列组1024列 vs 512列 | 512列在8K/16K更好（流水线的头尾更短），32K持平 |
| 限制每XCD的计算WG数、通信wave用`s_setprio` | 无影响 |

开发中的坑：

1. 32K时Y有2.4 GB（293888行×8 KB），buffer resource的num_records必须给满4 GB；按2 GB设置时，2 GB之后的行全读成0，结果错误。
2. `inv`和peer地址必须用标量load（address_space(4)）。如果用vector load，它们会在按序的vmcnt里排在Y的load或远端store后面。
3. 在展开的sum循环里加运行时模式分支，让通信慢了50%（1855 → 2873 µs）；改成模板参数后恢复。
4. FlyDSL与HIP链接：HIP代码不能调用ockl（用`__builtin_amdgcn_workitem_id_x`等builtin）；编译时加`-nogpulib -fgpu-rdc -mcode-object-version=6`，并用`-target-feature -packed-fp32-ops`与FlyDSL kernel的特性一致，否则被调函数不会内联。

结论：单kernel融合在本机可行、结果逐位正确，比down + 融合sum_ar依次执行快3–8%（每层约−0.05~−0.12 ms）。剩下的损失来自Y的往返，xGMI通信本身可以藏到down下面。所以要继续减少这部分时间，应减少Y的字节数（4.6的int8 Y，需确认精度），而不是继续调整重叠方式。与mega_moe_tp的区别：它每CU 1个CTA，内含4个计算wave + 4个通信/辅助wave；本实现因为down需要206 VGPR，改成每CU 2个WG分角色。zhaobo的gfx942移植版整层比基线慢约12%（4.6）。

### 4.11 整层对比：aiter、PyHIP、tingqli（含all-reduce）

8卡，每层MoE + all-reduce，都含sort和输入量化，输出都是all-reduce后的[T, H] BF16（µs）。各方案在同一轮里交替计时：每轮每个方案连续跑4次取平均，共15轮，取中位数，再取8卡中最慢的。这样可以避开功耗墙下时钟漂移和host launch延迟的影响：逐次同步计时时，同一方案的结果最多相差约一倍，aiter的Python路径里小launch较多，受影响最大。

tingqli的方案是A8W4，只作参考。它的权重是MXFP4，读权重的字节是FP8的一半；gfx942没有FP4 MFMA，要反量化成FP8再算，所以MFMA工作量相同。脚本：[down_ar_fused.py](probes/down_ar_fused.py)（`LAYER=1`）、[tq_layer_compare.py](probes/tq_layer_compare.py)。

| 方案 | 权重 | 8192 | 16384 | 32768 |
|---|---|---:|---:|---:|
| aiter fused_moe（1-stage asm）+ RCCL AR（当前部署） | FP8 | 1474 | 2833 | 5356 |
| aiter + two-shot AR（sum_ar的只AR模式） | FP8 | 1407 | 2729 | 5001 |
| PyHIP 2-stage + RCCL AR | FP8 | 1558 | 2812 | 5350 |
| PyHIP + 融合sum + two-shot AR（4.7） | FP8 | 1430 | 2509 | **4501** |
| PyHIP + down/sum/AR单kernel（4.10） | FP8 | 1387 | 2374 | 4518 |
| tingqli单kernel（fmoe bm64，atomic）+ RCCL AR | A8W4 | 1364 | 2650 | 5106 |
| tingqli单kernel + 他的two-shot | A8W4 | 1315 | 2516 | 4729 |
| tingqli 2-kernel（bm64 gemm1 + gemm2 nt4）+ two-shot | A8W4 | 1411 | 2698 | 5102 |
| tingqli 2-kernel（bm64 gemm1 + gemm2融合two-shot，CHUNK 256、CWG 64） | A8W4 | **1193** | **2271** | 4605 |

不含all-reduce的MoE本身，以及all-reduce暴露的时间（带AR减去不带AR）：

| 方案 | MoE本身（8K / 16K / 32K） | AR暴露（8K / 16K / 32K） |
|---|---|---|
| aiter + RCCL | 956 / 1820 / 3365 | 518 / 1013 / 1991 |
| PyHIP + 融合sum_ar | 1062 / 1814 / 3310（含sum） | 368 / 695 / 1191 |
| PyHIP + 单kernel | 同上 | 325 / 560 / 1208 |
| tingqli单kernel + two-shot | 834 / 1537 / 3034 | 481 / 979 / 1695 |
| tingqli 2-kernel + gemm2融合AR | 984 / 1808 / 3597（gemm2 base） | 209 / 463 / 1008 |

- 相对当前部署（aiter + RCCL）：PyHIP的单kernel版本在8K/16K/32K为0.94/0.84/0.84x，融合sum_ar版本为0.97/0.89/0.84x。32K下两者相同，每层约−0.85 ms。
- tingqli的2-kernel + gemm2融合AR在8K/16K最快（1193/2271 µs）。32K下比PyHIP慢约2%（4605 vs 4501）。小token下读权重占比高：每卡W13+W2的FP8字节约0.8 GB，约200 µs，FP4少一半。这部分来自权重格式，不完全是方案本身的差异。
- 它的AR暴露最少（32K为1008 µs，PyHIP约1200 µs）。原因是gemm2在GEMM内部用atomic把top-k直接累加到[T, H]，通信只需读T×H（32K下268 MB）；PyHIP的通信要读回Y（2.15 GB），而4.10已经测出，这部分读写正是重叠的瓶颈。代价是gemm2本身变慢：它2-kernel的MoE本身在32K比PyHIP多287 µs（3597 vs 3310），而且它的权重字节只有PyHIP的一半。
- 启示：做融合AR时，在GEMM内用atomic做top-k归约（通信只读T×H）比store + sum（通信要读Y）更利于重叠。PyHIP的down可以试“atomic累加到[T, H] + 融合AR”；4.5测得atomic本身暴露约1 ms，而且BF16 atomic的累加顺序不确定、数值变差，两者相抵后的结果需要实测。4.12已实测：在PyHIP的down上更慢。

### 4.12 尝试：down用BF16 atomic累加到[T, H] + 融合AR（实测更慢）

做法（[down_atomic.py](probes/down_atomic.py)，`down_ar_fused.py`加`ATOMIC=1`）：MFMA、W2环和调度与`_down_direct_body`相同，只换epilogue。retire后每个lane有一行的4个dword（行内跨16 B），用两次DPP quad交换（寄存器位0↔lane位0、寄存器位1↔lane位1）重排，使每条`global_atomic_pk_add_bf16`覆盖4行×64 B的完整片段。padding行用inline asm的`s_and_saveexec_b64`屏蔽（gfx942没有`buffer_atomic_pk_add_bf16`，不能靠越界丢弃）。通信用4.10的partial模式，只读[T, H]，不再读Y、不做top-k sum。累加器在整层里由aiter `moe_sorting`（`accumulate=True, output=`）清零。

atomic布局的微基准（[atomic_probe.py](probes/atomic_probe.py)，32K，单卡，2.15 GB的route行）：

| 每条atomic指令覆盖 | µs | TB/s |
|---|---:|---:|
| 1行×256 B（上限参考） | 1404 | 1.53 |
| 4行×64 B（一个32列unit；lane顺序连续或用DPP交换后的顺序都一样） | 1510–1548 | 1.39–1.42 |
| 2行×128 B（偶/奇unit配对） | 1460 | 1.47 |
| 4行×64 B，列组优先，列组宽1024/512/256列 | 1433–1532 | 1.40–1.50 |
| 不重排（每lane 4个dword跨行） | 5970 | 0.36 |
| 4行×64 B，加上down每个任务的A2/W2读 | 1892–1922 | 1.12–1.13 |
| 参考：普通store | 483 | 4.99 |

8卡结果（µs，最慢rank中位数；通信WG 160个，列组512列）：

| tokens | atomic down单独（通信WG空闲） | store down单独 | atomic融合 | store融合（4.10） | 通信单独（partial / 4.10） |
|---:|---:|---:|---:|---:|---:|
| 8192 | 500 | 353 | 814–821 | 788 | 506 / 505 |
| 16384 | 1097 | 572 | 1696 | 1394 | 920 / 966 |
| 32768 | 2472 | 1016 | 3463 | 2786 | 1752 / 1838 |

整层（`LAYER=1`，同一轮交替计时）：8K为1416 vs 1377（store融合），16K为2658 vs 2362；32K单kernel已慢677 µs，未再测整层。

数值（相对FP32真值：同样的BF16 route行，top-k和跨卡求和都用FP32）：atomic版rel_l2为4.03e-3，store + sum路径为2.36e-3；两次运行之间58–66%的元素不同。

慢在哪里（32K）：

| 探针 | atomic down单独（µs） |
|---|---:|
| 正常（608个计算WG） | 2390 |
| atomic全部EXEC=0（指令照发，不访存） | 782（比store down的1016还快） |
| W2环的vmcnt等待放宽到48/63（结果错误，只为计时） | 2395 / 2401 |
| 每XCD计算WG数20/38/50/76 | 3256 / 2615 / 2497 / 2390 |

- GEMM、DPP重排和等待都不是瓶颈，瓶颈是atomic本身的吞吐。kernel里的有效吞吐约0.9 TB/s（2.15 GB / 2.39 ms）。微基准里单独的atomic有1.4–1.5 TB/s，加上down的读降到1.1 TB/s，再与MFMA交错又降一截。gfx942的atomic都在memory侧做读改写（4.5：`TCC_EA0_ATOMIC == TCC_ATOMIC`）。store + 读回要多搬一倍字节，但store和load有4–5 TB/s。
- partial模式的通信单独只比读Y快约5%（32K为1752 vs 1838）：xGMI本身才是通信的瓶颈。4.10里限制重叠的是down写Y与通信抢带宽，不是通信读Y的字节数。
- tingqli的gemm2也用atomic，它的MoE本身在32K比PyHIP多287 µs（4.11）。他的AR暴露少，可能部分是因为gemm2更慢、能藏住更多通信（推断，未单独测量）。

验证过的细节：

1. EXEC=0的VMEM指令在gfx942上照样计入vmcnt（[execz_vmcnt.py](probes/execz_vmcnt.py)：发出一个冷load，再发20条EXEC=0的atomic，`s_waitcnt vmcnt(20)`总能等到load；不发atomic的对照则等不到）。所以用EXEC屏蔽padding行时，W2环里按指令数写死的vmcnt仍然正确。
2. atomic指令内的lane顺序不影响合并（上表“连续”与“DPP交换”的顺序相同），64 B一段就够，配对成128 B只多5%。

结论：不可行。atomic down单独（32K 2.47 ms）已经接近整个store融合kernel（2.79 ms），融合后8K/16K/32K为1.03/1.22/1.24x，整层也更慢，数值误差约1.7倍且不确定。继续保留store + 融合sum + AR。

### 4.13 与mega_moe_tp、tingqli单kernel的对照

down + reduce + 通信的单kernel已经实现（4.10）。它的结构是mega_moe_tp的gemm2 + AR部分：常驻CTA、任务用atomic认领、按列chunk顺序完成、route行在通信阶段做top-k求和。

| | mega_moe_tp（gfx950） | tingqli `gemm2_ar_fused` | PyHIP `down_ar_fused`（4.10） |
|---|---|---|---|
| 单kernel范围 | 整层：gemm1 + gemm2 + top-k + RS/AG，按token分块流水 | gemm2 + two-shot AR | down + top-k sum + two-shot AR |
| CTA | 常驻，每CU 1个，wave分角色（160 KB LDS） | 非常驻；grid前`cwg`个WG做通信 | 常驻，每CU 2个WG，运行时分角色，每XCD队列认领 |
| top-k归约 | route行，通信阶段求和 | atomic进moe_buf（route模式他测过更慢） | route行，通信阶段求和（atomic版更慢，4.12） |
| 32K依次 → 融合（µs） | — | 3691 → 2788（gemm2单独1957） | 2865 → 2786（down单独1016） |

- 两个单kernel在32K都是约2.79 ms（16K为1393 vs 1394，8K为727 vs 788；两者用各自的脚本计时）。tingqli“AR藏掙55%”是因为他的gemm2慢一倍，可藏的计算多；PyHIP的down快，依次执行已接近下限。
- 下限由通信和字节争用决定：two-shot AR单独就有1.56–1.67 ms（xGMI），比down更长；4.10的探针是AR单独1667、加不写Y的down 1949、加写Y 2340、再加sum读Y 2786。tingqli在≤2048 token、完全无依赖时也只藏住35–65%（他的`moe-new-tp.md`第9节）。

mega_moe_tp多出来的部分是把gemm1也放进同一个kernel，按token分块流水，让前一块的通信藏在后一块的gemm1下。4.8已测到通信与gate_up并发有收益（3168 → 2811 µs），但在本机有三个障碍：

1. gate_up的WG是8个wave、用满64 KB LDS，每CU只能放1个；`s_barrier`作用于整个WG，通信wave不能放进去。通信只能占独立的CU，或者缩小tile。zhaobo的gfx942移植版GEMM慢了36%（4.6）。
2. 按token分块要分块排序，每块都要为每个专家补齐。用本bench的路由实测行数（相对不分块）：

   | 块行数 | 分块 | 8K | 16K | 32K |
   |---:|---|---:|---:|---:|
   | 256 | 2块 | +36.5% | +21.1% | +9.8% |
   | 256 | 4块 | +173% | +61.5% | +32.2% |
   | 64 | 2块 | +10.3% | +6.2% | +2.7% |
   | 64 | 4块 | +31.7% | +17.2% | +8.7% |

   当前256行的M-block下，32K分2块要多算9.8%的行，约+0.24 ms（gate_up + down约2.4 ms），几乎抵消通信藏在gate_up下的上限（−0.36 ms）；小token下分块直接亏。按尾块M64补齐时只多2.7%。
3. tingqli也评估过改成mega_moe_tp式的常驻CTA（他的`moe-new-tp.md`第12节）：静态常驻的gemm2慢了6–24%，LL协议慢约2倍，让计算WG顺手做通信也更慢，结论是不值得投入。

`gemm1_ag_tp.py`是另一种TP切分：两个GEMM都按N切，gemm1的epilogue把FP8中间激活推给所有卡（AG1），MoE之后再AG输出。tingqli的结论是不值得做：中间激活宽topk×inter = 16384，是输出4096的4倍，AG1即使用FP8也和整个two-shot AR一样多；推送只有19–36%被gemm1掩盖；整层持平或更慢。

结论：down与通信的单kernel已经做到，限制在通信和字节争用，不在是否单kernel。mega_moe_tp真正多出的收益是把通信藏到gemm1下，前提是先把补齐降下来（优先级3的M64尾块），再做token分块（3b）。两者已实测（4.14）：M64尾块有小幅收益，token分两半更慢。

### 4.14 compact M64尾块（优先级3，已实现）与token分两半（3b，实测更慢）

**compact M64尾块**（源码已更新）：

- 短K路径（I=256）改用64行排序（aiter `moe_sorting` block 64）。
- 单WG的`_tiles_kernel`在设备端把每个专家的块切成256行tile，再加一个64–192行的尾tile，不需要host同步。
- gate_up和down按tile取（首行、行数、专家）：
  - gate_up中超出行数的不足64行的band跳过MFMA（`scf.if`），DMA和barrier照常；
  - down中没有行的wave只做W2 ring的DMA和barrier。
- I≥512的dense down仍用256行排序。

单卡TP8（µs）：

| token | 之前 | M64尾块 | aiter | gate_up | down | tile表 |
|---:|---:|---:|---:|---|---|---:|
| 4096 | 648 | 613 | 613 | | | |
| 8192 | 980–994 | 938–950 | 903–918 | 502 → 476 | 319 → 299 | 7 |
| 16384 | 1714–1717 | 1658–1669 | 1693–1721 | 833 → 742–762 | 502 → 467–477 | 10 |
| 32768 | 3160–3196 | 3134–3158 | 3166–3168 | 1382–1399 → 1361 | 923–929 → 901 | 11 |
| 65536 | 6432 | 6363 | 6353 | | | |

- 执行行减少了8.1%（32K）和14.4%（16K），但gate_up在这两档只快了2–3%和7–9%。
  - 8-wave流水里，band到SIMD的映射是固定的：band 0/2在SIMD 0/1上，band 1/3在SIMD 2/3上。
  - 所以只有1个band的尾tile仍要占一半时间，3个band的尾tile不省时间；DMA和barrier也照常。
- tile表kernel：初版逐块load，32K时19 µs；改成每线程用dwordx4扫16块后为11 µs。

8卡整层（测法同4.11，µs，结果仍与sum_ar逐位一致）：

| 方案 | 8K | 16K | 32K |
|---|---:|---:|---:|
| aiter + RCCL（当前部署） | 1444 | 2869 | 5363 |
| aiter + two-shot | 1385 | 2760 | 4962 |
| PyHIP（不含AR） | 1032 | 1784 | 3268 |
| PyHIP + RCCL | 1500 | 2804 | 5249 |
| PyHIP + 融合sum_ar | 1366 | 2455 | **4401** |
| PyHIP + 单kernel down_ar | **1338** | **2379** | 4417 |

- 相对4.11：PyHIP本身减少30/30/42 µs，sum_ar版减少64/54/100 µs，down_ar版为−49/+5/−101 µs。
- 相对当前部署为0.93/0.83/0.82x。

**token分两半（3b）**（[halves_pipeline.py](probes/halves_pipeline.py)）：

- 两半各自排序、切tile。
- A的down完成后，A的sum_ar放在CU-masked的comm stream上，B的gate_up放在其余CU上；之后在整卡上跑B的down和sum_ar。
- 结果与一次性整层逐位一致。

8卡结果（µs）：

| 方案 | 16K | 32K |
|---|---:|---:|
| 一次性 + 融合sum_ar | 2446 | 4432 |
| 两半依次执行 | 2765 | 4867 |
| 两半重叠（comm 32 / 64 / 96 CU） | 3155 / 3461 / 2739 | 5829 / 5667 / 4643 |

- 两半依次执行就慢了319/435 µs，原因有三：
  - 每半都要把全部专家的W13+W2（每卡共768 MB）再读一遍；
  - 小一半的GEMM效率更低：8K的gate_up是476 µs，两次就是952，而16K只要762；
  - 多一次排序和tile表，补齐也更多（M64分2块时多6.2%/2.7%的行）。
- 重叠最多收回约130–220 µs（96 CU最好），抵不上拆分的代价。
- 这也是mega_moe_tp式token分块在本部署不划算的原因：TP8下每卡权重768 MB，每多一块就多读一遍。
- 踩到的坑：`sum_ar.hip`结尾的grid barrier按`epoch × grid`计数，同一块对称缓冲不能换grid大小用，否则会挂死；所以每种grid各用一块缓冲。主stream也不能用legacy default stream。

**把gemm1也融进同一个kernel（1-stage，M64单元）：不建议投入（之后已实现并实测，见4.15）。**

- 现成参照：aiter的FP8 1-stage asm（M64，`fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_64x256`）在同一测法下32K为3362 µs，已慢于PyHIP 2-stage（3268）。用FlyDSL重写只能接近它。
- 1-stage里一个token的输出要等它的8个专家都算完，4.10的down列组流水就不存在了。要与通信重叠只能按token分块，又回到上面每块重读权重的问题。
- 1-stage省下的只有A2的往返（32K约69 MB，约30 µs）和tile表kernel；用atomic代替Y+sum时还能省inverse kernel。
- gate_up单独改M64单元时，每64行要读一遍W13[e]（2 MB），权重复用只有M256的1/4。尾块已按M64处理，满块保持M256。

### 4.15 1-stage M64 kernel：gemm1 + requant + down（参考tingqli `fmoe_bm64`；逐位正确，实测更慢）

实现：[kernels_m64.py](../../src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/kernels_m64.py)，`PYHIP_MOE_DOWN=fused`在整个MoE里启用（仅I=256、H为1024的倍数，默认关闭）。

- 每CTA一个64行排序块（一个专家）。默认8 wave：218 VGPR、26.5 KB LDS、无spill，每CU 1个CTA。4 wave、每CU 2个CTA的版本见本节后半。
- gemm1：wave w负责gate和up各32列，一个CTA算完全部I=256。
  - X行按token gather，每个K-block 8 KB，经2槽LDS环（行XOR swizzle）。
  - W13片段从预shuffle权重直接读进VGPR，提前一个K-block。
  - block scale的partial晚两块用FMA退休，按K顺序累加，与2-stage gate_up逐位一致。
- requant：同一128列组的4个wave经LDS交换行最大值。FP8 A2和scale留在LDS，不写回：前128列放在单独的8 KB区，后128列复用X槽。
- down：wave w负责输出列512w .. +511。
  - A2行和scale读进VGPR，W2片段直接从global读，提前一个32列unit。
  - MFMA、退休和成对store与`_down_direct_body`相同，Y与2-stage逐位一致。
- 之后仍要inverse和sum（或sum_ar）。省掉的是tile表kernel和A2/A2S的往返。

单卡TP8，整个MoE（bench `--breakdown`，µs）：

| token | 2-stage合计 | fused合计 | 2-stage gate_up + down + tile表 | moe64 | moe64 / 2-stage |
|---:|---:|---:|---|---:|---:|
| 8192 | 937 | 965 | | | |
| 16384 | 1684 | 1783 | 805 + 482 + 7 = 1294 | 1425 | 1.10x |
| 32768 | 3149 | 3425 | 1365 + 900 + 12 = 2277 | 2583 | 1.13x |

逐位对比与背靠背计时（[moe64_check.py](probes/moe64_check.py)，µs；A2和Y的有效行均0 mismatch）：

| token | gate_up（tile） | gate_up64 | down（tile） | gate_up+down | moe64 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 445–455 | 432–442 | 300–308 | 757–762 | 770–772 |
| 16384 | 764–782 | 792–809 | 531–542 | 1292–1311 | 1345–1357 |
| 32768 | 1332–1348 | 1452–1480 | 987–998 | 2370 | 2554–2566 |

慢的原因：

- **权重复用只有1/4。**
  - gemm1：每64行读一遍W13[e]（2 MB），32K共8.4 GB；tile gate_up每CTA（256行 × 半个I）读1 MB，共2.1 GB。
  - down：每个wave自己从global取W2片段，每64行读一遍W2[e]（1 MB），32K共4.2 GB；短K down是4个wave共用LDS里的W2环，共1.1 GB。
  - 8K时满256行的tile很少，所以gate_up64与tile版持平；token越多差距越大（32K gemm1部分+110–140 µs，down部分+90–115 µs）。
- **每CU只有1个CTA。** 限制是VGPR，不是LDS。
  - 每个SIMD有512个VGPR（VGPR与AGPR统一）。moe64每wave 218个（按8分配为224），每SIMD只能放2个wave。一个CTA有8个wave，每个SIMD正好2个，VGPR已经用满。
  - LDS 26.5 KB本可以放2个CTA。
  - 再放一个CTA需要每wave ≤128 VGPR。现有寄存器占用（每wave）：
    - gemm1：累加器64（64行 × gate/up各32列）、W13片段32、下一K-block预取32、X片段32，再加partial、scale和地址；
    - down：A2行64（整个K）、W2片段32 + 预取32、两组MFMA partial 32，再加store暂存和scale。
  - W13不能改走LDS：一个CTA每个K-block的W13是512行 × 128 B = 64 KB。
  - 强制`waves_per_eu=4`（≤128 VGPR）的实测结果：
    - moe64 spill 601个VGPR、scratch 940 B/lane，8K 770 → 3782 µs，32K 2554 → 13812 µs；
    - gate_up64 spill 260个，8K 432 → 2385 µs；
    - moe64的Y还会出错：inline-asm ds_read的结果在`s_waitcnt lgkmcnt`之前就被spill出去，存下的是旧值。
  - tingqli的每CU 2个CTA是每CTA 4个wave，每SIMD仍是2个wave，与这里的总占用率相同，区别只是两个CTA可以处在不同阶段。为了放进256 VGPR，他把gemm1按intermediate分两半各跑一遍（X读两次）；FP4的W片段寄存器也只要FP8的一半。本实现的同样结构实测见下文。
  - 两个CTA即使能错开阶段，4.5(3)的同CU并发探针也显示，在功耗墙下down的store与另一个CTA的gemm1重叠没有收益。
- **省下的很少。** A2往返在32K约69 MB（约30 µs），tile表11 µs。
- W2预取改为提前两个unit（+32 VGPR）：8K 770 → 782，32K 2566 → 2559 µs，在噪声内，已回退。

**改成4-wave CTA × 每CU 2个（tingqli的结构）：逐位正确，但更慢。**

- `launch_moe64(..., NW=4)`：每CTA 4个wave，gemm1分两遍，每遍128个intermediate列（column wave = 4 × 遍号 + wave），X要gather两次。down中每个wave负责1024列。
- 资源：248 VGPR、26.5 KB LDS、无spill，每CU 2个CTA，每SIMD仍是2个wave。
- 第一版把第0遍的A2 code留在寄存器里跨过第1遍，结果到256 VGPR并spill，Y出错。改为每遍结束就写进LDS，并在两遍之间加`sched_barrier`，降到248。

背靠背计时（moe64_check.py，µs，两种wave数都0 mismatch）：

| token | 2-stage gate_up+down | gate_up64 x8 / x4 | moe64 x8 / x4 |
|---:|---:|---|---|
| 8192 | 766 | 442 / 489 | 783 / 845 |
| 16384 | 1299 | 803 / 839 | 1363 / 1514 |
| 32768 | 2382 | 1450 / 1527 | 2568 / 2804 |

整个MoE（bench `--breakdown`，µs）：

| token | 2-stage | fused x8 | fused x4 |
|---:|---:|---:|---:|
| 8192 | 937 | 964 | 1042 |
| 16384 | 1684 | 1782 | 1883 |
| 32768 | 3149 | 3443 | 3557 |

32K的PMC（[pmc_kernels.py](probes/pmc_kernels.py)，fused时加`PYHIP_MOE_DOWN=fused PYHIP_MOE_FUSED_WAVES=8/4`；L2未命中读即`FETCH_SIZE`，由MALL或HBM提供）：

| kernel | µs | 时钟 | MFMA忙 | L2未命中读 | 写 | L2命中 |
|---|---:|---|---|---:|---:|---|
| 2-stage gate_up + down | 1400 + 905 | 1.10 / 1.13 GHz | 58.9% / 44.5% | 2.11 + 0.49 GB | 0.07 + 2.15 GB | 66.5% / 54.4% |
| moe64 x8 | 2588 | 1.22 GHz | 43.4% | 3.17 GB | 2.15 GB | 72.7% |
| moe64 x4 | 2642 | 1.13 GHz | 45.5% | 6.08 GB | 2.15 GB | 61.2% |
| moe64 x4，LDS补到40 KB（每CU 1个，`FUSED_LDS=40960`） | 2811 | 1.52 GHz | 31.9% | 5.15 GB | 2.15 GB | 66.4% |

- x8与2-stage的MFMA周期数相同（MFMA忙 × 时间 × 时钟：1370 vs 1362 k），差别在MFMA忙更低，以及多出0.57 GB的L2未命中读（每64行重读权重）。
- x4的L2未命中读几乎翻倍（6.08 vs 3.17 GB）：
  - X要gather两次；
  - 每CU 2个CTA，每个XCD同时有76个块在算（约4.6个专家，x8约2.3个），W13/W2的工作集翻倍，L2命中从73%降到61%。
  - 多出的数据搬运增加功耗，时钟从1.22降到1.13 GHz。
- 多出的2.91 GB可以拆开（用每CU 1个的x4做中间点）：
  - 两遍gemm1占+1.98 GB（3.17 → 5.15）。每个wave的寄存器只够64行 × gate/up各32列（累加器64），4个wave一遍只能覆盖128个intermediate列，X要再读一遍（真实行约1.07 GB），其余约0.9 GB是权重分两遍扫时L2复用变差。一遍算完256列需要累加器128 + W13片段64 + 预取64 + X片段32，超过256。
  - 每CU 2个CTA占+0.93 GB（5.15 → 6.08），L2命中66% → 61%。
  - x8里每个wave的寄存器占用与x4相同，只是8个wave共用同一份X，一遍算完全部256列。
  - tingqli在I=256时也是两遍。他的代码里`NH = INTER // 128`，每遍128列（4个wave × 32列），每遍都重新gather A；调优表在8K–32K选的是`ks1`，即一个CTA做完两遍。I=512时是4块：16K以下用`ks2`（每个块2个CTA，各做两遍），32K用`ks1`加`waves_per_eu=1`（一个CTA做四遍）。他每遍的寄存器约为累加器64 + W 32：FP4的W片段只有FP8的一半，而且读完就地装下一个tile，不另设预取缓冲。
  - 一遍版x4要放进256 VGPR，得改gemm1的内循环：W13片段用完就地重装（省掉64的预取缓冲），X按M子块流式读（32 → 16）。估算约250，几乎没有余量。即使做到，也只能消掉两遍的+1.98 GB，每CU 2个CTA的+0.93 GB仍在，最好情况与x8持平，仍慢于2-stage，所以不做。
- 第二个CTA几乎没有掩盖延迟：每周期的MFMA忙只从43.4%升到45.5%。把LDS补到40 KB、每CU只放一个4-wave CTA时，moe64 x4为2881 µs，两个CTA为2804 µs，只差3%。
- 所以fused默认用8 wave（`PYHIP_MOE_FUSED_WAVES`默认8），4 wave可设为4（kernel参数`NW=4`）。

与tingqli `fmoe_bm64`（`doc-mimo/kernels/test_a8w4_fmoe_bm64.py`）及其他实现的结构差异：

| | tingqli fmoe_bm64 | PyHIP 1-stage x4 | PyHIP 1-stage x8 | PyHIP 2-stage | aiter 1-stage asm |
|---|---|---|---|---|---|
| 权重 | MXFP4（字节为FP8的一半），寄存器内解码成FP8 | FP8 128×128 block scale | 同左 | 同左 | 同左 |
| 每CTA | 64行，4 wave，每CU 2个 | 64行，4 wave，每CU 2个 | 64行，8 wave，每CU 1个 | gate_up 256行×128列（8 wave，每CU 1个）；down 256行×1024列（4 wave，每CU 2个） | 64×256，每CU 1个 |
| gemm1 | 分两半各一遍，K tile 256，单槽A + 寄存器预取；单个累加器，scale在K循环后乘 | 分两半各一遍，K128两槽DMA环；每个K128的partial单独按scale用FMA退休 | 一遍，同左 | W经LDS，4倍复用（M256） | — |
| 中间激活 | LDS（bf16暂存 + FP8半块） | LDS | LDS | HBM（A2、A2S） | — |
| gemm2 | 32个128列n-tile，W2进寄存器；单个累加器在K组边界按s0/s1 rescale | 每wave 1024列，W2进寄存器，两个partial用FMA退休 | 每wave 512列 | W2经LDS环由4个wave共用 | — |
| 输出 | BF16 atomic到[T, H]（不确定） | Y route行 + sum（逐位确定） | 同左 | 同左 | BF16 atomic |
| VGPR / LDS | 256 / 24.5 KB | 248 / 26.5 KB | 218 / 26.5 KB | 246 / 64 KB；194 / 32 KB | 512 / 64 KB |

- block scale让每个K128都要有单独的partial和scale，tingqli的格式只要一个累加器。所以同样的4-wave结构，FP8下寄存器贴到248，VALU也更多。
- M64重读的权重，FP8每块3 MB（W13 2 MB + W2 1 MB），FP4约1.5 MB。x4下每个XCD约4.6个专家在算，FP8的工作集约14 MB，FP4约7 MB，L2为4 MB。
- tingqli的1-stage是和他自己的2-kernel比（32K 2901 vs 3269 µs）；他2-kernel的gemm1同样是M64，gemm2慢约一倍。PyHIP 2-stage的gate_up是M256加M64尾块，down的W2经LDS共享，已在功耗墙下GEMM上限的85%左右。

**与tingqli 1-stage的差距拆分**

数据来源：tingqli的逐kernel时间用[tq_store_sum_check.py](probes/tq_store_sum_check.py)测，`G2_NO_ATOMIC=1`得到不写回的版本；他的PMC用[tq_pmc.py](probes/tq_pmc.py)测；PyHIP的GEMM部分是gate_up加不写Y的down，用[moe64_check.py](probes/moe64_check.py)测。单位µs：

| 部分（8K / 16K / 32K） | tingqli 1-stage（A8W4） | PyHIP 2-stage（FP8） | PyHIP多出 |
|---|---|---|---|
| 量化 + 排序 | 81 / 133 / 275 | 64 / 112 / 258 | −17 / −21 / −17 |
| GEMM（不写回） | 566 / 983 / 1818 | 686 / 1117 / 1921 | +120 / +134 / +103 |
| 写回 + top-k | atomic：133 / 365 / 800 | store + inverse + sum：191 / 455 / 970 | +58 / +90 / +170 |
| 合计 | 780 / 1481 / 2893 | 941 / 1684 / 3149 | +161 / +203 / +256 |

逐个GEMM对比（不写回）：

| token | tingqli gemm1 | PyHIP gate_up | tingqli gemm2 | PyHIP down | tingqli 1-stage |
|---:|---:|---:|---:|---:|---:|
| 8192 | 424 | 473–480 | 285 | 203–216 | 566 |
| 16384 | 727 | 791 | 516 | 339 | 983 |
| 32768 | 1326 | 1340 | 986 | 594 | 1818 |

32K的PMC，tingqli的fused kernel：不写回时1941 µs、1.21 GHz、MFMA忙57.0%、L2未命中读2.95 GB、L2命中68.1%；带atomic时2527 µs、1.16 GHz、MFMA忙46.0%。

- 他的1-stage比自己的2-kernel快，是因为fused kernel里gemm2的时间有一半藏到了另一个CTA的gemm1下面：1-stage减去gemm1，剩142 / 256 / 492，正好是他单独gemm2的50%。
- 他能藏住，是因为访存少。他同样做两遍、X读两次，但L2未命中读只有2.95 GB（PyHIP x4为6.08 GB），时钟保持在1.21 GHz。区别在FP4：每64行重读的权重字节少一半，L2能容下更多同时在算的专家。PyHIP用同样结构时被访存和功耗拖住（x4为1.13 GHz）。
- 32K时PyHIP的gate_up与他的gemm1相同（1340 vs 1326），GEMM部分的差全在down：他有效的gemm2约492，PyHIP不写Y的down是594。8K/16K的差主要在gate_up：小token时读权重的占比高，FP8每卡768 MB，FP4约400 MB，按HBM带宽算约差90 µs。
- 写回：atomic只在gemm2本身慢、有计算可藏时才划算。PyHIP的down快，4.12实测换成atomic反而更慢：down单独500 / 1097 / 2472 vs 353 / 572 / 1016 µs。
- 结论：差距来自权重格式（FP4字节减半，同时影响HBM下限和L2命中），以及atomic与慢gemm2的搭配，不在CTA结构。同为FP8时，tingqli的结构（x4）在本机更慢。

结论：1-stage在本部署比2-stage慢3%/6%/9%（8K/16K/32K整个MoE；4-wave × 2 CTA版本慢11%/12%/13%），保留为实验开关，不作默认。

- 它也不利于与通信融合：每个M64块写出全部4096列，任何列组都要等所有块完成。4.10的down列组流水因此不存在，要与通信重叠只能按token分块，而每块都要重读768 MB权重（4.14的3b）。
- tingqli的1-stage比他的2-kernel快，是因为fused kernel把他较慢的gemm2藏掉了一半（见上面的差距拆分）。PyHIP的down本身就比他的gemm2快1.3–1.7倍，没有这部分可藏。

### 4.16 各方案汇总（2026-11-03同日重测）

单卡MoE：TP8，GPU0，含输入量化和排序，不含AR。单位µs，用bench的2 s循环计时（tingqli的用[tq_store_sum_check.py](probes/tq_store_sum_check.py)的`loop`列，方法相同）：

| 方案 | 权重 | 写回 | 8192 | 16384 | 32768 |
|---|---|---|---:|---:|---:|
| aiter fused_moe（1-stage asm，当前部署） | FP8 | atomic | 902–906 | 1692–1696 | 3156–3162 |
| **PyHIP 2-stage（默认）** | FP8 | Y + sum | 947 | 1652 | 3154 |
| PyHIP 1-stage x8（`PYHIP_MOE_DOWN=fused`） | FP8 | Y + sum | 968 | 1798 | 3457 |
| PyHIP 1-stage x4（另加`PYHIP_MOE_FUSED_WAVES=4`） | FP8 | Y + sum | 1042 | 1897 | 3559 |
| tingqli 1-stage（fmoe_bm64） | A8W4 | atomic | 797 | 1509 | 2965 |
| tingqli 2-kernel（bm64 + gemm2 nt4） | A8W4 | atomic | 921 | 1732 | 3349 |
| tingqli 2-kernel（bm128） | A8W4 | atomic | 925 | 1714 | 3316 |
| tingqli 2-kernel（bm128） | A8W4 | store + sum | 1073 | 1997 | 3831 |

8卡整层：每层MoE + all-reduce，都含量化和排序，取最慢rank的中位数，各方案同一轮交替计时，单位µs。我们和aiter的用[down_ar_fused.py](probes/down_ar_fused.py) `LAYER=1`测，tingqli的用[tq_layer_compare.py](probes/tq_layer_compare.py)测，两个脚本先后运行。所有PyHIP变体的输出与sum_ar逐位一致。

| 方案 | 权重 | 8192 | 16384 | 32768 | 相对当前部署 | 不含AR的MoE | AR暴露 |
|---|---|---:|---:|---:|---|---|---|
| aiter + RCCL（当前部署） | FP8 | 1516 | 2799 | 5354 | 1 | 964 / 1817 / 3352 | 552 / 982 / 2002 |
| aiter + two-shot | FP8 | 1437 | 2736 | 4971 | 0.95 / 0.98 / 0.93 | 同上 | 473 / 919 / 1619 |
| PyHIP 2-stage + RCCL | FP8 | 1552 | 2726 | 5241 | 1.02 / 0.97 / 0.98 | 1054 / 1774 / 3268 | 498 / 952 / 1973 |
| **PyHIP 2-stage + 融合sum_ar**（4.7） | FP8 | 1404 | 2429 | **4394** | 0.93 / 0.87 / **0.82** | 同上 | 350 / 655 / 1126 |
| **PyHIP 2-stage + down_ar单kernel**（4.10） | FP8 | 1369 | 2351 | 4478 | 0.90 / 0.84 / 0.84 | 同上 | 315 / 576 / 1210 |
| PyHIP 1-stage x8 + 融合sum_ar | FP8 | 1401 | 2588 | 4891 | 0.92 / 0.92 / 0.91 | 1007 / 1835 / 3551 | 395 / 753 / 1341 |
| tingqli 1-stage + RCCL | A8W4 | 1341 | 2627 | 5132 | 0.88 / 0.94 / 0.96 | 824 / 1519 / 3023 | 517 / 1108 / 2109 |
| tingqli 1-stage + two-shot | A8W4 | 1320 | 2520 | 4729 | 0.87 / 0.90 / 0.88 | 同上 | 496 / 1001 / 1706 |
| tingqli 2-kernel nt4 + two-shot | A8W4 | 1422 | 2704 | 5127 | 0.94 / 0.97 / 0.96 | 944 / 1753 / 3405 | 478 / 951 / 1722 |
| **tingqli 2-kernel + gemm2融合AR**（CHUNK 256、CWG 64） | A8W4 | **1218** | **2258** | 4534 | **0.80 / 0.81** / 0.85 | 992 / 1811 / 3608（gemm2 base） | 226 / 448 / 926 |

32K逐kernel PMC（[pmc_kernels.py](probes/pmc_kernels.py)、[tq_pmc.py](probes/tq_pmc.py)；aiter一行为11-02的4.3数据）：

| kernel | µs | 时钟 | MFMA忙 | L2未命中读 | 写 | L2命中 |
|---|---:|---|---:|---:|---:|---:|
| aiter 1-stage | 2905 | 1.37 GHz | 34% | 2.90 GB | 2.15 GB（atomic） | — |
| PyHIP gate_up / down / sum | 1400 / 905 / 543 | 1.10 / 1.13 / 1.53 GHz | 58.9% / 44.5% / 0 | 2.11 / 0.49 / 2.15 GB | 0.07 / 2.15 / 0.27 GB | 66.5% / 54.4% / 10.0% |
| PyHIP 1-stage x8 | 2588 | 1.22 GHz | 43.4% | 3.17 GB | 2.15 GB | 72.7% |
| PyHIP 1-stage x4 | 2642 | 1.13 GHz | 45.5% | 6.08 GB | 2.15 GB | 61.2% |
| tingqli 1-stage | 2527 | 1.16 GHz | 46.0% | 2.94 GB | 2.15 GB（atomic） | 46.9% |
| tingqli 1-stage（不写回） | 1941 | 1.21 GHz | 57.0% | 2.95 GB | 0.02 GB | 68.1% |

- FP8权重下，32K最快的是PyHIP 2-stage + 融合sum_ar（当前部署的0.82x），8K/16K最快的是down_ar单kernel（0.90/0.84x）。
- 包括A8W4在内，8K/16K最快的是tingqli的2-kernel + gemm2融合AR（0.80/0.81x），差距主要来自FP4的权重字节（4.15）。
- 1-stage的两种结构在FP8下都不如2-stage，保留为实验开关。

### 4.17 sglang新分支`feat/mimo-dpa-prefill-opt`：base、序列并行实现与端到端结果（2026-11-03）

环境：worktree `sglang_dpa`，本地分支`feat/mimo-dpa-prefill-opt-pyhip`（基于b75b5fc215，改动已`git add`）；aiter快进到37c1a5b8d（新分支需要`plan_pa_decode`）。启动脚本`launch_mimo_v26_flash_dp2_tp8_ep1_dpa_accuracy_baseline.sh`：DP2 × attention TP4，MoE TP8 EP1，FP8 experts，DFlash，chunked prefill 32K（每DP组16K）。负载为ISL 64K / OSL 20，max-concurrency 2（每DP组1个请求）。完整数据：`/root/workspace/data/dpa_prefill/results.md`。

**1. 分支的“每层两次AR改为序列并行”是否合理**

分支里的序列并行只在EP8 + MORI配置（`launch_mimo_v26_flash_dp2_tp4_ep8_dflash_accuracy_baseline.sh`）中生效：残差流按attention-TP切分，attention后RS4，QKV前用packed FP8 AG4（`Fp8BlockwisePacket`），MoE走MORI的dispatch/combine。TP8配置仍然每层两次8卡AR。实测（真实路由`PERFORMANCE_MODE=0`，rank 0每步GPU时间，ms）：

| chunk | base（TP8，2×AR8） | 分支EP8 + MORI |
|---|---:|---:|
| 0 / 1 / 2 | 469 / 527 / 597 | 522 / 551 / 602 |
| 每层通信 | AR8 1.71–1.88 ms × 2 | RS4 0.95 + AG4 0.48 + dispatch 0.48 + **combine 2.8–3.4** |

- 序列并行本身合理：attention侧的RS4 + FP8 AG4比AR8快约25%，而且去掉了8卡重复的全token norm/量化、全局router GEMM和DP gather的fill/memcpy（这几项从54 ms/step降到14 ms/step）。
- 但MoE改EP8 + MORI不合适：combine每层2.8–3.4 ms（含等最慢rank的expert；各rank MoE为131–158 ms/step，不均衡1.10），整步比base慢5%。与4.2第3点的判断一致：MiMo top-8、8卡下all-to-all在字节上没有优势，只增加不均衡。
- 因此保留attention侧的序列并行，MoE保持TP8：all-gather FP8输入（+ 1x128 scale + top-k）、MoE后reduce-scatter，即4.2的方案。

**2. 实现（sglang，`SGLANG_DPA_SP_MOE=1`）**

| 改动 | 说明 |
|---|---|
| `communicator.py` | `_LayerModeComputationContext.sp_tp_moe`：模型声明自己处理token切分时，稀疏层的MLP模式为`SCATTERED`，attention侧复用分支已有的RS4和packed FP8 AG4 |
| `models/mimo_v2.py` | `MiMoV2MoE.forward_sp`：本卡shard上做router、top-k和FP8 1x128量化（与`fused_moe`内部同一个`per_group_quant_hip`），用`all_gatherv`在TP组gather FP8、scale、top-k，scale拼成列主序（`a1_scale_is_transposed`），TP8 expert算全部token，`reduce_scatter`回shard；空shard不发量化kernel |
| 每次前向选择 | 每层建两个`LayerCommunicator`；全局token ≥ `SGLANG_DPA_SP_MOE_MIN_TOKENS`（默认2048）且不在graph capture时走序列并行，否则走原来的AR路径（decode、CUDA graph不变）。第一版decode也走序列并行时，每层4次小集合通信，GSM8K decode吞吐低约15% |
| `fp8_utils.py` | 合入PyHIP FP8 GEMM（`SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM`），并接受行跨度的FP8输入 |
| PyHIP GEMM | `gemm_a8w8_blockscale_942`增加X的行跨度`lda`（constexpr），packed AG4输出（行宽4224字节）直接作为GEMM输入，与连续X逐位相同、速度相同（[bench_mimo_fp8_linear_942.py](../gemm/bench_mimo_fp8_linear_942.py)） |

**3. PyHIP FP8 GEMM与分支CK bpreshuffle（单卡）**

| 形状（M × N × K） | CK µs | PyHIP µs | PyHIP/CK |
|---|---:|---:|---:|
| SWA QKV 16384×3712×4096（39层） | 604 | 502 | 0.83 |
| full QKV 16384×3392×4096（9层） | 686 | 443 | 0.65 |
| dense gate_up 32768×4096×4096 | 1295 | 1027 | 0.79 |
| dense down 32768×4096×2048 | 695 | 555 | 0.80 |

与FP32参考的rel_rms两者相同（1.66e-3）。端到端QKV/dense GEMM从34.8 ms/step（723 TFLOPS）降到26.2 ms/step（960 TFLOPS）。

**4. 端到端结果**

prefill每步GPU时间（rank 0，两个DP组都在做16K chunk，ms）：

| chunk | base | + PyHIP GEMM | + 序列并行MoE | 分支EP8 |
|---|---:|---:|---:|---:|
| 0 | 469.1 | 481.6 | 413.4 | 521.5 |
| 1 | 527.4 | 506.8 | 439.6 | 550.6 |
| 2 | 597.4 | 558.0 | 488.9 | 602.2 |
| 均值 | 531.3 | 515.5（−3.0%） | **447.3（−15.8%）** | 558.1（+5.0%） |

每层MoE的通信：base 2 × AR8 = 3.4–3.8 ms；序列并行RS4 0.89 + AG8（FP8 + scale + top-k）0.52 + RS8 0.88 + AG4（FP8 packet）0.44 = **2.73 ms**。每步分类（ms）：通信182 → 145（RCCL 121 + aiter AG4 24），norm/quant 17.9 → 4.4，copy/fill 16.2 → 3.0，router GEMM 9.7 → 1.5，QKV/dense GEMM 34.8 → 25.4，MoE GEMM 140 → 144（不变）。

bench（32个请求）：输入吞吐45,431 → 49,926 tok/s（+9.9%），总时长46.2 → 42.0 s，平均/中位/P99 TTFT 2053/2047/2645 → 1943/1882/2443 ms，平均E2E 2879 → 2622 ms。GSM8K（5-shot，200题）：base 0.865，最终配置0.870（标准误约±2.4%）。

其他：

- QuickReduce FP（4.2第6点）：序列并行后prefill只剩第0层（dense TP8）的2次大AR（约4 ms/step），收益不到0.2 ms/step，未启用。
- 冷启动：分支在`flydsl_pa.py`/`flydsl_helpers.py`里`setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")`，PyHIP的FlyDSL kernel每次起服务首次使用都要编译，首批大prefill停顿50–90 s（base约15 s）。部署时显式设`FLYDSL_RUNTIME_ENABLE_CACHE=1`或预热。

### 4.18 与DP2 EP8最佳模式（attention后scatter → megaMoE（EP，非TP）→ all-gather）对比（估算）

两者attention侧相同（RS4 BF16 + FP8 packet AG4），差别只在MoE段：当前为AG8（FP8 + scale + top-k）→ TP8 expert（每rank I=256）→ RS8；EP8为一个megaMoE kernel内完成dispatch → 本rank 32个expert（I=2048）→ combine。

每卡每层传输量（32K全局token，每卡shard 4096 token，MiB）。EP8按均匀路由：每个token平均落到5.29张卡（P = 1 − C(224,8)/C(256,8) = 0.662），即4.63张远端卡；dispatch按目标卡去重，combine前在expert所在卡先按路由权重求和：

| 段 | 基线2×AR8 | 当前序列并行TP8 | EP8 megaMoE |
|---|---:|---:|---:|
| attention后 | AR8 448 | RS4 96（4卡，3条链路） | RS4 96 |
| MoE输入 | — | AG8 117 | dispatch 78 |
| MoE输出 | AR8 448 | RS8 224 | combine 148 |
| attention前 | — | AG4 49.5（4卡） | AG4 49.5 |
| **合计** | **896** | **487** | **371（−24%）** |
| 其中MoE段（8卡全链路） | — | 341 | 226（−34%） |

- dispatch不去重、combine不预先求和时为117 + 224 = 341 MiB，与TP8的AG8 + RS8完全相同；EP的字节优势全部来自去重和预求和（top-8落到8卡，平均只有5.29张不同的卡）。
- 按实测带宽（AG8 236、RS8 267 GB/s；4.2中RCCL A2A8约230 GB/s），EP8的226 MiB不重叠时约0.9–1.0 ms，当前AG8 + RS8实测1.40 ms。

MoE段每层时间（ms）。每卡FLOP相同：2 × 32768 × 8 × (4096 × 4096 + 2048 × 4096) / 8 = 1.649 TFLOP；EP按1000 TFLOPS的70%–80%即2.06–2.36 ms（PyHIP的EP8 kernel单卡实测2.43 ms，即68%，见第3节）；当前TP8 aiter实测3.06 ms（54%）：

| 场景 | MoE计算 | 通信暴露 | MoE段 | 每层省 | prefill每步（当前447 ms，base 531 ms） |
|---|---:|---:|---:|---:|---:|
| 当前序列并行TP8（实测） | 3.06 | 1.40（AG8 0.52 + RS8 0.88） | 4.46 | — | 447 |
| EP8乐观：80%、负载均衡、通信几乎全藏 | 2.06 | 0.1 | 2.16 | 2.30 | 339（−24%；比base −36%） |
| EP8中性：75%、不均衡1.10、通信藏一半 | 2.42 | 0.47 | 2.89 | 1.57 | 373（−17%；比base −30%） |
| EP8保守：70%、不均衡1.10、通信不重叠 | 2.59 | 0.95 | 3.54 | 0.92 | 404（−10%；比base −24%） |

- 每层省0.9–2.3 ms，其中计算效率（I=256的TP8只有54%）占0.5–1.0 ms，通信（字节−34%加上与计算重叠）占0.45–1.3 ms。
- 不均衡1.10来自本次EP8实测（random-ids，各rank MoE 131–158 ms/step）；真实文本可能更大，需要EPLB冗余expert。
- 前提是gfx942上有EP的megaMoE（dispatch去重、combine预求和、与expert计算重叠）：aiter的mega_moe只支持gfx950；分支的MORI实测dispatch 0.48 + combine 2.8–3.4 ms/层（4.17），离上表很远。可行路线是PyHIP EP8 kernel（68%）加IPC的dispatch/combine。

### 4.19 与pyhip提交45afb765（luwei，gfx942 FP8 block-scale dense GEMM + MoE GateUp/Down LDS kernel）对比（2026-11-03）

单卡GPU 0，输入相同。脚本在`/root/workspace/data/luwei45/`（`bench_dense.py`、`bench_moe.py`，45afb765的`src/contrib/flydsl`导出到`/root/workspace/data/pyhip_45afb76`）。

Dense GEMM（kernel时间，µs）：

| M×N×K | CK | PyHIP 8w | 45afb 8x1 | 45afb 4x2-lds | 45afb 2x4 | 45afb最好 / PyHIP |
|---|---:|---:|---:|---:|---:|---:|
| 16384×3712×4096（SWA QKV） | 609 | 500 | 550 | 557 | 566 | 1.10 |
| 16384×3392×4096（full QKV） | 691 | 435 | 462 | 461 | 467 | 1.06 |
| 32768×4096×4096（第0层gate_up） | 1283 | 1023 | 1093 | 1077 | 1095 | 1.05 |
| 32768×4096×2048（第0层down） | 699 | 552 | 586 | 592 | 603 | 1.06 |
| 8192×3712×4096 | 314 | 256 | 272 | 275 | 281 | 1.06 |
| 4096×3712×4096 | 189 | 123 | 127 | 131 | 133 | 1.04 |
| 4096³ / 8192³ | 170 / 1723 | 130 / 1021 | 137 / 1093 | 140 / 1084 | 140 / 1118 | 1.05 / 1.06 |

- 精度：CK、PyHIP、45afb 2x4的rel_rms都是1.66e-3；45afb 8x1和4x2-lds是3.31e-3（BF16输出截断，不是RNE）。2x4除SWA QKV（PyHIP走split-K尾块）外与PyHIP逐位相同。

MoE整次调用（入口量化 + aiter moe_sorting + GEMM1 + GEMM2 + top-k求和，µs；括号为相对PyHIP）。45afb的Down检查capacity × H × 2 < 2 GiB（capacity按aiter sorting的T × 8 + E × 256行），32K无法单次调用，表中*为分两次16K：

| 配置 | token | PyHIP | aiter | 45afb atomic（最好layout） | 45afb rows + 求和（最好layout） |
|---|---:|---:|---:|---:|---:|
| TP8（I=256，256 expert） | 8K | 948 | 905 | 1259（1.33x） | 1546（1.63x） |
| | 16K | 1645 | 1715 | 2174（1.32x） | 2682（1.63x） |
| | 32K | 3098 | 3152 | 4352*（1.40x） | 5357*（1.73x） |
| TP4（I=512） | 8K | 1746 | 1796 | 1559（0.89x） | 1615（0.93x） |
| | 16K | 2984 | 3110 | 2727（0.91x） | 2856（0.96x） |
| | 32K | 5417 | 5745 | 5406*（1.00x） | 5817*（1.07x） |
| EP8（I=2048，本卡32 expert） | 8K | 839 | 1040 | 801（0.96x） | 805（0.96x） |
| | 16K | 1405 | 1816 | 1332（0.95x） | 1363（0.97x） |
| | 32K | 2420 | 3454 | 2610*（1.08x） | 2659*（1.10x） |

- 已安装的FlyDSL 0.3.2没有`rocdl.s_buffer_load`，45afb的`reduce="kernel"`（`moe_down_reduce`）和GateUp `pair_blocks=True`跑不了；“rows + 求和”用它的rows kernel（同样的rows/inverse）加PyHIP的FP32求和kernel代替。
- 精度（8K，相对FP32参考的rel_l2）：PyHIP 0.616% / 0.612% / 0.594%（TP8/TP4/EP8），45afb rows + 求和相同（与PyHIP相差7e-5），45afb atomic 0.698% / 0.694% / 0.599%（BF16 atomic逐route舍入，与aiter的0.697%相当；EP8每token本卡约1个route，所以几乎不变）。

16K分kernel（µs）：

| kernel | TP8 PyHIP | TP8 45afb | TP4 PyHIP | TP4 45afb | EP8 PyHIP | EP8 45afb |
|---|---:|---:|---:|---:|---:|---:|
| 入口量化 + sorting | 113 | 102 | 122 | 111 | 143 | 140 |
| GEMM1（GateUp + FP8重量化） | 828 | 732（−12%） | 1513 | 1318（−13%） | 788 | 718（−9%） |
| GEMM2（Down） | 498 | 1274（含atomic求和） | 973 | 1268（含atomic求和） | 373 | 419（含atomic求和） |
| top-k求和 | 266 | — | 294 | — | 79 | — |
| 清零/tiles/inverse | 19 | 26 | 19 | 28 | 18 | 29 |

- 45afb的GateUp在三种形状上都比PyHIP的gate_up快9%–13%，而且TP8时它按256行pad（约25%空行），PyHIP用compact M64尾块（约6%），按实际算的tile计差距更大。
- 45afb的Down在I=256（K=256）时很慢：1274 µs，对比PyHIP的down_direct + 求和764 µs。I=2048时与PyHIP的down + 求和相当。
- 结论：dense保留PyHIP 8w；当前部署的TP8 MoE不换（45afb慢32%–40%）；EP8在≤16K时45afb快3%–5%，32K因为要分两次反而慢8%。可借鉴的是它的GateUp：换进PyHIP的gate_up（保留compact尾块和down_direct），估计整个MoE −5%–6%（TP8 16K 828 → 约730 µs）。
- 顺带修复：4.17给`gemm_fp8_blockscale_8w_942._body`加了`LDA`参数，MoE dense down（I≥512，TP4/EP8）的调用当时没有改，会报TypeError；已改为传`LDA=K`，`tests/ops/moe/test_moe_fp8_blockscale_942.py`共5项通过。I=256的部署路径不受影响。

### 4.20 TP8序列并行之后的继续优化（2026-11-03）

配置同4.17（DP2 × attention TP4、MoE TP8、DFlash、chunked prefill 32K即每组16K、ISL/OSL 64K/20、并发2、32条请求）。
代码在sglang_dpa的`feat/mimo-dpa-prefill-opt-pyhip`上（本地commit），新增开关：

| 开关 | 作用 |
|---|---|
| `SGLANG_ENABLE_EMBED_REPLICATION=1` + `--moe-dense-tp-size 1` | embedding每卡全量（MiMo与MTP改用`get_embedding_tp_kwargs`），第0层dense MLP在token shard上用全量权重：去掉每步3次AR（embedding AR4、dp_gather AR8、MLP AR8），换成1次RS4 + 1次AG4 |
| `SGLANG_DPA_SP_OPROJ=1` | o_proj之前把attention输出按token做all-to-all（每卡发48 MiB，RS4要发96 MiB），o_proj用全量权重（K=8192）直接得到本卡shard；去掉RS4，且o_proj从4份BF16部分和变为单次FP32累加 |
| `SGLANG_DPA_SP_SWA=1` | 39个SWA层（窗口128）在本卡token shard上做全部64头：全量QKV（加载后从TP分片拼成[Q\|K\|V]）、本卡8个KV头里属于其它rank的2头经all-to-all写回按头切分的cache、128 token窗口halo（前面slice的尾部 + 前一chunk的cache）经小all-gather、临时page-64 SHUFFLE-5D页上跑aiter FlyDSL SWA kernel、全量o_proj。SWA层不再有AG4和o_proj的RS4/A2A |
| `--enable-prefill-delayer --prefill-delayer-max-delay-passes 100` | 两个DP组的prefill对齐（sglang已有功能，要求overlap schedule） |
| `SGLANG_USE_PYHIP_FP8_MOE=1` + `--ep-size 2` | MoE改EP2×TP4（I=512），PyHIP MoE接入aiter runner（从`feat/mimo-ck-linear-prefill`移植）；顺带修了`should_use_dp_reduce_scatterv`在EP==DP且attention TP>1时的断言 |

2026-11-04整理（分支改名`feat/mimo-dpa-prefill-opt-comm`，在`/root/workspace/sglang`，rebase到`origin/mimo-opt-mxfp4-dflash`）：
FP8 GEMM kernel移进sglang（`layers/quantization/pyhip_fp8_blockscale_gemm.py`，与本仓库cab0421的kernel源码一致，只用Torch和FlyDSL），
不再需要`SGLANG_PYHIP_ROOT`；PyHIP MoE接入（EP2实验用，未采用）删除。新增环境变量只剩两个：
`SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM`，以及`SGLANG_ENABLE_DPA_SP`（= 原`SGLANG_DPA_SP_MOE` + `_OPROJ` + `_SWA`）；2048 token门槛改为代码常量。

每步prefill（rank 0，chunk 0/1/2；“净”= 非通信kernel + 去掉>5 ms一次性等待的通信）与端到端：

| 配置 | 每步净值 ms | 通信 ms/step | bench input tok/s | Mean E2E / TTFT / TPOT ms | GSM8K |
|---|---:|---:|---:|---|---:|
| base（4.17） | 531.3（含等待） | 182 | 45,431 | — | 0.865 |
| sp（4.17，SP + PyHIP GEMM） | 435.9 | 136–141 | 49,926 | 2622 / 1943 / 35.8 | 0.870 |
| + dense TP1 + embedding复制（dense1） | 433.2 | 135–137 | 45,394* | 2881 / 1716 / 61.4 | 0.875 |
| sp + prefill delayer（spd） | 439.8 | 137–146 | 60,943 | 2147 / 1978 / 8.9 | — |
| dense1 + EP2×TP4 + PyHIP MoE + delayer（ep2d） | 441.2 | 150–156 | 61,410 | 2131 / 1969 / 8.5 | 0.900（无delayer那次） |
| dense1 + A2A o_proj + delayer（oprojd） | 410.3 | 112.5–113 | 65,186 | 2003 / 1842 / 8.4 | 0.895 |
| oprojd + SWA序列并行（swad） | **387.8** | **87–91** | **67,466** | **1938 / 1771 / 8.8** | 0.895 |

*没有prefill delayer时两个DP组的相位随机：一组decode时要陪另一组的prefill（SP每层8卡同步），TPOT从20 ms拉到60–90 ms，
端到端吞吐在±10%内随机波动（dense1的每步prefill其实更快）。所以之后都开prefill delayer。

- swad相对sp：每步净值−48 ms（−11%），端到端吞吐+35%（其中prefill delayer约+22%，o_proj A2A约+7%，SWA序列并行约+3.5%）；相对base +48.5%。
- o_proj A2A：4卡单测T=16K，GEMM+RS4 1472 µs → A2A+GEMM 1054 µs（RS4 886 → A2A 450 µs）；o_proj对FP32参考rel_l2 3.16e-3 → 1.66e-3。
- SWA序列并行：4卡单测（`check_sp_swa.py`：两请求带前缀、slice小于窗口、16K+32K前缀、三请求跨slice）与TP路径同一kernel结果差≤1 BF16 ulp，cache写入逐位一致，padding行为0。
  e2e每步通信再−22~26 ms；非通信+1~4 ms（临时页的cat/index/permute，copy/fill 2.6 → 9.3 ms/step，可融合）。
- EP2×TP4：MoE kernel 2.87 ms/层（aiter TP8 3.19），但两个专家半区逐层负载不同（每步平均只差1.5%），RS8要等慢的一半，
  通信+15~20 ms/step，净值反而+8 ms。单卡均匀路由32K：tp8 PyHIP/aiter 3133/3163，ep2 2839/3669，ep4 2591/3560 µs（aiter带expert_mask很慢）。不采用。
- full attention kernel：sglang的`pyhip_pa.py`移植自PyHIP `f55f549`，kernel源码与PyHIP HEAD相同。PyHIP MHA readme 3.6的约600T是Q=32768（全局chunk 64K）；
  本配置Q=16384，单卡同形状KV 16K/32K/48K为567/554/537T（平均546T），e2e 541T，与单卡一致。Q=32768时590T。差距是1024个任务只有3.4波（304 CU）的尾波。
- o_proj：checkpoint的`ignored_layers`共49项全是`self_attn.o_proj`，权重本来就是BF16，走hipBLASLt。M4096 N4096 K8192单卡487–493 µs（557–564T）。
  PyHIP的asm `gemm_8wave_fp8bf16fp16`有bf16模式，但按gfx950写（256×256 tile超过gfx942的64 KB LDS，小tile断言失败），gfx942上没有可用的PyHIP BF16 GEMM。
  假设在线量化成FP8（未部署）：1×128激活量化57–74 µs + PyHIP FP8 GEMM 246–267 µs（1031–1117T），每次省150–190 µs，每步约−7~−9 ms；
  但对FP32参考rel 3.7e-2（BF16为1.7e-3），属于降精度，不做。
- PyHIP FP8 GEMM在最终配置形状上（单卡，CK bpreshuffle → PyHIP，rel_rms同为1.66e-3）：SWA QKV（SP，4096×14848×4096，39层）797 → 502 µs（625 → 992T），
  full QKV（TP4，16384×3392×4096，9层）719 → 463（633 → 982T），dense gate_up（TP1，4096×32768×4096）1632 → 1018（674 → 1080T），
  dense down（TP1，4096×4096×16384）917 → 507（600 → 1084T）。每步合计CK 40.1 → PyHIP 25.3 ms（e2e实测24.0），−14.8 ms/step（约占swad每步3.7%）；
  SP-SWA把QKV变成M4096 N14848后，CK从826T降到625T，PyHIP仍约990T，所以收益比第1轮（−8.6 ms/step）大。

swad每步构成（rank 0，ms/step）：MoE GEMM 141.5（aiter 3.01 ms/层）、RCCL 90.4（A2A 48次 + AG8 47 + RS8 47）、full attention 68.9、o_proj 29.3（全量K后561T）、
QKV/dense 24.0、aiter AG 10.0、copy/fill 9.3、MoE route/sort 9.2、SWA 6.2、norm/quant 4.3、rope/KV 3.8；另有DFlash的FC（20480→4096，每组16K token在4个rank上重复算）约5 ms。

每步集合通信（rank 0，chunk 1，`comm_volume.py`）。发出量按每卡计：AG为(n−1)×本卡分片，RS/A2A为(n−1)/n×输入，AR为2(n−1)/n×输入。

| 集合通信 | 每次发出 | base | sp | swad |
|---|---:|---|---|---|
| AR8 BF16（attention后、MoE后） | 448 MiB | 96次，178.3 ms | 2次（第0层），4.9 ms | — |
| AR4 BF16（embedding） | 192 MiB | 1次，1.8 ms | 1次，2.0 ms | — |
| AG4 FP8包（attention输入，4224 B/token） | 49.5 MiB | — | 46次，20.3 ms | 8次，3.6 ms |
| RS4 BF16（o_proj部分和） | 96 MiB | — | 47次，41.9 ms | — |
| A2A4 BF16（o_proj） | 48 MiB | — | — | 9次，3.7 ms |
| A2A4 BF16（SWA K/V） | 15 MiB | — | — | 39次，6.3 ms |
| AG4 BF16（SWA halo） | 1.86 + 0.47 MiB | — | — | 78次，1.6 ms |
| AG4 BF16（DFlash抓取、末层拼回） | 96 MiB | — | 4次，3.3 ms | 5次，4.8 ms |
| AG8（FP8 + scale + top-k，4288 B/token） | 117 MiB | — | 47次，24.5 ms | 47次，25.0 ms |
| RS8 BF16（MoE输出） | 224 MiB | — | 47次，40.8 ms | 47次，42.3 ms |
| 合计 | | 42.2 GiB，180.1 ms | 23.7 GiB，137.6 ms | 17.6 GiB，87.3 ms |

8卡集合等效230–260 GB/s，4卡组内100–125 GB/s（3条xGMI vs 7条）。swad里77%的通信时间是MoE的AG8 + RS8。

### 4.21 继续减少通信：DFlash投影放到shard上、MoE的top-k sum融进RS8（2026-11-04）

代码在sglang `feat/mimo-dpa-prefill-opt-comm`（`/root/workspace/sglang`，e3ef41b187之后，未commit），配置同4.20最终配置（swad）。

**① DFlash上下文投影在token shard上算。** 原来抓取的5个层输入（第1/12/24/36层 + 末层）各AG4成[16K, 4096]，拼成[16K, 20480]后
attention-TP组4卡各自对全部16K行算FC（20480→4096）+ hidden_norm。改为各卡在自己的4K行上抓取、拼接、投影，只AG4一次投影后的[16K, 4096]
（target注册draft的`project_target_hidden`；draft见到4096宽的输入就不再投影）。只在EXTEND且SP生效时走；decode / verify不变。
每步（rank 0，chunk 1）：AG4 [16K, 4096] 5次4.42 ms → 2次1.77 ms，FC GEMM 5.21 → 1.44 ms，拼接1.59 → 1.19 ms，合计约−7.2 ms/step；
GSM8K阶段accept len不变（5.15 vs 5.14），GSM8K 0.895，bench 66,142 → 66,847 tok/s（+1.1%）。

**② MoE：PyHIP MoE（gate_up + down_direct）+ 融合的top-k sum / RS8。** 序列并行下MoE每层是AG8 → MoE（全部32K token，本卡I=256切片）→ RS8。
aiter 1-stage MoE自带top-k求和，没法在它的输出上融合；PyHIP的direct路径本来就先写route行Y再单独求和，所以把求和与RS8合成一个kernel
（[sum_rs.hip](probes/sum_rs.hip)，在[sum_ar.hip](probes/sum_ar.hip)基础上改）：
- 常驻grid 96个WG × 512线程。wave 0–3对token（unit u，目的rank s按u轮转）的8个route行做FP32求和、转BF16，写进LDS环（4格）；
  wave 4–7把每个四分之一行用write-through store推到rank s的inbox[parity][本rank][u]，每个unit结束后在目的rank上置flag（单调epoch，不清零）。
- reduce阶段：等本rank各行的8×4个flag，按rank顺序FP32求和8个inbox行（绕过L2的load），BF16写出本卡shard。
- RS阶段从不等别的rank，所以WG全部常驻时不会死锁；对方最多领先一个epoch，inbox按epoch奇偶双缓冲。
- shard大小可不同（两个DP组token数不同、某组空闲）：unit取最大shard，u超出s的shard就跳过；目的rank的inbox / flag行号为`src × size[s] + u`。
- 与参考逐位一致：每rank部分和 = BF16(FP32 Σk)，跨rank FP32按rank顺序求和、一次BF16舍入（RCCL的RS是BF16逐跳舍入，这里不比它差）。

8卡kernel（`sum_rs.py`，grid 96，最慢rank中位数，µs）：

| 全局token | shard | sum | RCCL RS | sum+RS | 融合 | 融合/(sum+RS) |
|---:|---|---:|---:|---:|---:|---:|
| 16K | 2048×8 | 285 | 473 | 720 | 540 | 0.75 |
| 32K | 4096×8 | 618 | 842 | 1357 | 1119 | 0.82 |
| 16K | 2730×4 / 1366×4 | 285 | 713* | 918 | 717 | 0.78 |
| 32K | 5461×4 / 2731×4 | 617 | 1225* | 1773 | 1407 | 0.79 |

*不等长时RCCL按最大shard补齐再RS。grid越大越慢（16K/32K：96 → 543/1091，128 → 757/1434，304 → 1461/2860 µs）：
远端写并发太多时xGMI拥塞；只推本卡（不走xGMI）时304 WG最快，所以瓶颈是远端写的并发度。

整层MoE + RS（`moe_rs.py`，8卡同时跑，均匀路由，预量化FP8输入，µs）：

| 全局token | aiter + RCCL RS | PyHIP（含sum）+ RCCL RS | PyHIP + 融合sum/RS | 融合/aiter |
|---:|---:|---:|---:|---:|
| 16K | 2419 | 2319 | 2157 | 0.89 |
| 32K | 4259 | 4019 | 3765 | 0.88 |

与aiter + RCCL的rel_l2为5.7e-3（两套FP8实现之差；各自对FP32参考都是4.0e-3）。

接入sglang：`SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM=1`且`SGLANG_ENABLE_DPA_SP=1`时，MiMo的SP MoE改走
`layers/moe/pyhip_fp8_blockscale_moe.py`（从cab0421移植tiles / gate_up / inverse / down_direct，与本仓库源码AST一致，只删了dense down和sum；
融合sum + RS是同文件里的FlyDSL kernel `_sum_rs_kernel`，算法同`sum_rs.hip`，不需要hipcc / tvm-ffi）。对称内存用hipMalloc + IPC handle，
首次使用时在TP组的device communicator上all-gather交换；容量 = 每DP组chunked prefill / attention TP（这里4096行，每卡512 MiB），
更大的shard退回aiter + RCCL。层检查（gfx942、FP8 128×128、aiter shuffle、I=256、top-8、TP8、无EP / 共享专家 / clamp / bias）不过也退回。
8卡核对（`check_pyhip_moe_rs.py`）：与PyHIP kernel + 探针kernel逐位一致（等长、不等长、一组空闲，各3个epoch）。

端到端（rank 0，chunk 0/1/2）：

| 指标 | dfl1（①） | moers2（① + ②，HIP JIT） | moers4（HIP JIT，最终版） | moers5（FlyDSL版） | moers7（FlyDSL，无asm，全buffer） |
|---|---:|---:|---:|---:|---:|
| 每步span ms | 382.3 / 422.9 / 573.6 | 328.1 / 378.2 / 445.0 | 326.4 / 375.0 / 424.5 | 320.6 / 387.0 / 440.2 | 346.0 / 423.2 / 424.2（含等另一DP组） |
| 每步净值 ms | 352.0 / 391.3 / 433.8 | — | 315.8 / 368.5 / 419.3 | 313.8 / 370.2 / 415.3 | — |
| 每步非通信 ms | 245.4 / 301.6 / 344.3 | 211.9 / 263.1 / 312.6 | 212.7 / 262.0 / 312.1 | — | — |
| MoE kernel ms/step（chunk 1） | aiter 144.0 | gate_up 65.1 + down 44.5 + tiles/inverse 0.8 = 110.4 | 同左 | 同左 | 同左 |
| RCCL ms/step（chunk 1） | 104.4（含RS8与等待） | 38.1（RS8没了） | 38.4 | 53.4（AG8等待多） | 92.5（含20 ms等待） |
| 融合sum + RS ms/step | — | 63.0（含等待） | 60.3 | 57.4 | 60.1（单次中位数1.22 ms） |
| input tok/s | 66,847 | 70,157（moers3：70,682） | 71,590（+7.1%） | 70,634 | 70,847 |
| Mean TTFT / TPOT ms | 1763 / 10.17 | 1697 / 8.83 | 1674 / 8.10 | 1685 / 8.73 | 1674 / 9.04 |
| GSM8K | 0.895 | 0.900 | — | 0.890 | 0.885 |

- MoE + RS每步约210 → 170 ms（−40 ms，约−0.85 ms/层）。相对e3ef41（head）：66,142 → 70.6K–71.6K tok/s（+7%–8%）。
- 净值 = 非通信kernel + 去掉>5 ms一次性等待的通信（span还含等另一个DP组的时间）。PR描述里chunk 1的389 ms是swad的净值，dfl1同口径391.3 ms。
- 通信字节数不变：RS8每层仍是每卡发224 MiB（10.28 GiB/step），只是不走RCCL。`comm_volume.py`的“RCCL send total”只算带record_param_comms的
  RCCL调用（RS8 + A2A，不含AG8和aiter AG），所以从11.27变成0.99 GiB；全部集合通信每卡每步17.6 → 17.3 GiB（只有①省下DFlash的3次AG4）。
- 第一次e2e（moers1，旧版按需扩容的对称内存）吞吐63,741、TPOT 15.75 ms（ITL全程偏高，accept len正常，最大ITL 41 ms），GSM8K 0.875；
  第二次（moers2，固定容量）起正常（moers2–4）。按需扩容会在forward里用CPU组做all_gather_object，已改成固定容量 + device组交换。
- sglang里第一版（HIP，tvm-ffi JIT）融合kernel比探针的同一算法慢7–15%：route行用一个带边界的buffer resource、行偏移在VGPR里算（为让缺失route（−1）越界读0）。
  交叉计时（各自kernel读对方的Y）确认是kernel本身而不是缓冲区。改回每行一个scalar resource（缺失route给0字节resource，无分支）后，
  16K/32K为532 / 1155 µs，与探针521 / 1133 µs差≤2%，结果仍逐位一致。
- 之后改写成FlyDSL（`_sum_rs_kernel`），HIP的.cuh和JIT封装删除；新kernel不用inline asm，向量访存全是buffer指令（ISA里没有`flat_*` / `global_*`）：
  - 数据：route行`nt`，inbox写`sc0 sc1`（write-through），inbox读`sc0 sc1`；每个resource都在4 GiB内（inbox ≤ 512 MiB，flag区 ≤ 1 MiB）；
  - flag：读用`llvm.amdgcn.raw.ptr.atomic.buffer.load`（`buffer_load_dword … sc0 sc1`；这个intrinsic对LLVM不是只读，
    自旋循环每轮都重新读，普通buffer load会被外提成死循环），写用`buffer_atomic_swap … sc1`（每个目的rank一个scalar resource，
    lane 0依次写8个，shard外的给0字节resource丢弃）；写前`fence release workgroup` + `s_waitcnt vmcnt(0)`，等到后`fence acquire workgroup`，
    都不产生`buffer_wbl2` / `buffer_inv`；
  - LDS环的数据用普通LLVM load/store（`ds_read_b128` / `ds_write_b128`），计数器用`workgroup-one-as`域的acquire load / release store
    （LDS按序执行，编译器没有插lgkmcnt等待）；INV的8个route行一条`s_load_dwordx8`（invariant load）；
  - 求和循环每次处理两行、两组寄存器乒乓：只用一组时，回边要把新预取的16个向量拷进循环携带寄存器，先`s_waitcnt vmcnt(0)`等它们到齐。
  与探针逐位一致（15个用例）。交替计时16K / 32K为557 / 1111 µs（探针523 / 1126 µs）；VGPR 142，SGPR 101。

### 4.22 1f：MoE gate_up / down的上限评估与优化（2026-11-04）

单卡GPU 7，MiMo TP8形状（H 4096，I 256，256专家，top-8），32K token。脚本[route_rows_bench.py](probes/route_rows_bench.py)：
与sglang `moe_route_rows`相同的调用，`--check`按路由逐位比较A2/A2S/Y，`--routing`切换路由分布，`--ablate`做消融。

**两个kernel都顶在750 W功耗墙上。** 循环跑kernel时用`rocm-smi`采样：

| kernel | 功耗 | sclk | 实测 | 按当时时钟的FP8峰值 |
|---|---:|---:|---:|---:|
| gate_up | 750 W | 1.22–1.23 GHz | 约900 TFLOPS（路由行）/ 935（含compact padding的执行行） | 1532 TFLOPS，利用率59%（PMC的MFMA忙也是59%） |
| down | 750 W | 1.05–1.07 GHz | 2.5–2.6 TB/s（写Y 2.0 GiB + 读A2/W2） | — |

**gate_up分解**（µs）：

| 变体 | gate_up | 差值 | 含义 |
|---|---:|---:|---|
| 随机路由（基线） | 1271 | — | |
| 均衡路由（每专家正好1024行，无尾tile） | 1182 | −89 | 尾tile和更多CTA（2296 → 2048）的代价 |
| 均衡，且每专家的token连续 | 1166 | −16 | X gather的局部性，影响很小 |
| epilogue换成一次store（消融） | 1217 | −54 | SiLU、量化、写出合计 |
| 只丢弃A2/A2S store（消融） | 1210 | −61 | 主要是写出本身，计算几乎不占时间 |

对照：
- 同一台机器上dense FP8 GEMM 32768×4096×4096（同样在功耗墙下）1023 µs，约1075 TFLOPS。gate_up的执行行速率935 T约为它的87%。
- 45afb765的GateUp（`fp8_out`，256行pad，293888执行行）在32K下1272 µs，本实现1211 µs（270144执行行），本实现快5%；
  按执行行算45afb只快约3.5%。4.19在16K看到的9%–13%差距在32K不存在，1c（换成45afb的GateUp）不再值得做。

**down分解**：store全部丢弃时556 µs（MFMA约束），正常时973 µs，写出多花417 µs。4.5(3)已测过把store挪到另一个WG也省不下来：
功耗墙下写HBM的能耗只能靠降频支付。只减半W2的DMA（消融，结果错误）可省49 µs。

**尝试过的改动**（都与基线逐位一致；单卡为交替3轮的中位数，e2e为moers8/moers9 rank 0全部141次调用的中位数）：

| 改动 | 单卡32K | e2e | 结论 |
|---|---|---|---|
| gate_up的A2先经LDS中转（行r放在区域r>>5的4 KiB处，16 B chunk按r&7异或），每条`buffer_store_dwordx4`写8个完整的128 B行；原来每lane 16条dword store，每行分8次各写16 B | 1232.9 → 1223.6（−9） | 1350.8 → 1333.8（−17） | 保留 |
| 短K down改8-wave CTA × 512行tile：W2的DMA与L2→LDS流量减半，`_tiles_kernel`一次launch写出256行和512行两张表 | 随机路由976.8 → 965.0（−12），偏斜路由（每专家logit偏置std 0.5）也是−11 | 943.9 → 968.5（+25）；排除与另一stream的kernel重叠的调用后仍是+25 | 回退。真实路由下尾tile更多，只有1个wave在算的512行tile CTA独占整个CU |

e2e（`run_case.sh`，ACC=1，32条请求）：

| run | 内容 | GSM8K | 输入吞吐 tok/s | 平均TTFT ms | chunk 1 span ms |
|---|---|---:|---:|---:|---:|
| moers7 | 基线 | 0.885 | 70847 | 1673.5 | 423.2 |
| moers8 | LDS中转 + 8-wave down | 0.895 | 72201 | 1653.9 | 398.6 |
| moers9 | 只保留LDS中转 | 0.890 | 70451 | 1693.2 | 393.9 |

- 每层只省约17 µs（约0.8 ms/step），在e2e的±2%噪声以内。
- 三次之间chunk 1 span相差约30 ms，几乎全部来自通信算子里的等待：AG8、A2A4、RS8的总时间减去“调用次数 × 中位数”，
  moers7约49 ms，moers8约3.5 ms，moers9约18 ms（rank之间的偏斜，见第5节1k）。

**结论**：gate_up已达到同功耗下dense GEMM的87%，down的写出代价是能耗（2 GiB的Y），重叠或换写法都省不下来。
原先估计的“gate_up到1000 T、down到4 TB/s，约−28 ms/step”不成立。剩下能做的只有：
- gate_up的尾tile：band到SIMD的映射固定，64行尾tile仍要1/2时间，192行要整块时间。需要为尾tile换一种wave划分，估计−25~−45 µs/层；
- 两个kernel都要减少能耗，也就是少搬字节。

### 4.23 1g：通信改走SDMA（xGMI copy engine）的评估（2026-11-04）

- RCCL 2.27.7（ROCm 7.2）的集合通信都由CU kernel完成，没有SDMA路径（只有`RCCL_P2P_BATCH_*`之类开关）。
- 本机已装的MORI（`/sgl-workspace/mori`@879983bd）有SDMA集合通信：`mori.ccl.AllgatherSdma` / `All2allSdma`。
  - 需要设置`MORI_ENABLE_SDMA=1`，并初始化mori shmem。
  - 由一个1×512线程的小kernel提交SDMA包并等待完成。
  - 结果可以留在symmetric transit buffer里（零拷贝），也可以再拷到用户buffer。

脚本`/root/workspace/data/dpa_prefill/sdma_ag_bench.py`（`MORI_ENABLE_SDMA=1 torchrun --nproc_per_node 8`），8卡，每rank 14.625 MiB（等于MoE AG8的117 MiB）。
“与GEMM并发”指另一stream上同时跑BF16 16384×4096×4096 GEMM（单独约920 µs）：

| 实现 | 单独 | 每rank发出 | 与GEMM并发时的AG | GEMM变慢 | 并发总时间 vs 串行 |
|---|---:|---:|---:|---:|---:|
| RCCL，单tensor | 388–390 µs | 275 GB/s | 778 µs | +92% | +5%（没有收益） |
| RCCL，4个tensor一组（模型现状：top-k权重、id、FP8 hidden、scale） | 454 µs | 236 GB/s | — | — | — |
| e2e中的AG8（moers7–9中位数） | 535–556 µs | — | — | — | — |
| MORI SDMA，拷回用户buffer | 415–505 µs | 212–258 GB/s | 643 µs | +35% | −32% |
| MORI SDMA，零拷贝 | 349 µs | 308 GB/s | 504 µs | +19% | −28% |

32 MiB每rank时：RCCL 816–832 µs，SDMA零拷贝721–732 µs（321 GB/s）。与GEMM并发时RCCL严重退化（GEMM +98%~+384%），
SDMA零拷贝GEMM只慢18%，总时间−26%。

结论：
- SDMA可用，而且确实不占CU。零拷贝时AG8单独就比单tensor的RCCL快10%，比模型现在的4-tensor分组快23%。
  与计算并发时，GEMM只慢约19%（HBM与功耗的争用），RCCL则会让GEMM慢近一倍。
- MORI的API会阻塞host：同步调用的host耗时约等于传输时间，异步API的`wait_async`也阻塞。sglang的CPU要领先GPU发kernel，
  不能直接用，需要stream有序、不阻塞host的版本（或我们自己提交SDMA包）。
- 关键路径：AG8 → MoE是依赖链。要把通信藏起来，必须把MoE按token分块流水，而4.14已测过token分两半本身就多花319–435 µs。
  即使用SDMA（重叠时GEMM约+19%），隐藏约0.55 ms的AG8也基本收不回成本。所以SDMA的实际收益来自“直接替换”：
  e2e的AG8从535–556 µs降到约350 µs（打包成一个buffer + 零拷贝），约−0.2 ms/层、−9 ms/step。
  其中4-tensor分组 → 单buffer这一步不需要SDMA，RCCL也能拿到约−60~−100 µs/层。
- RS8的求和必须用CU，不适合SDMA。A2A4 K/V（15 MiB，约210 µs）可以同样改用`All2allSdma`，但量小，收益约−1~−2 ms/step。

### 4.24 1k：rank间偏斜；1j：AG8打包 / SDMA（2026-11-04）

详细数据见`/root/workspace/data/dpa_prefill/results.md`第7轮。

1k：8个rank的trace时钟对齐（同一次集合通信的结束时间相差中位数11–13 µs），按“最晚开始 − 自己开始”统计每次集合通信里的等待。
- 可修的来源：每个prefill前向开头有两处host同步。一处是aiter prefill元数据，`kv_indices[kv_indptr[-1]:]`的隐式`.item()`和`paged_kv_indptr[0] = 0`；
  另一处是SP-SWA元数据的两个同步`.to(device)`。host在这里失去领先，之后SWA层的小kernel launch跟不上GPU，
  各rank主stream的空闲每step 1.6–14.6 ms（最坏42.8 ms），其他rank在下一次集合通信里等。
  sglang里改成device上的掩码和`non_blocking`（结果逐位相同），chunk 1/2的这类空闲变成0。
  扣掉冻结后，非首块每步约−11~−15 ms（rank 0 span：chunk 1约375 → 364.6 ms，chunk 2约426 → 411 ms）。
- 另一个来源是约15–24 ms的GPU冻结，每3个prefill step出现0–3次。表现为某张卡的一个计算kernel变长（down 4次、GEMM 2次），
  或所有卡的集合通信一起多等约15–20 ms；期间没有其他stream kernel，也没有host malloc / free。
  原因是主机`kernel.numa_balancing`被外部周期性打开：NUMA采样改页表 → MMU notifier → KFD驱逐整个进程的GPU queue。
  sglang各rank的`evicted_ms`每秒增长约2.5 ms，只在numa_balancing=1时增长。
  用显式`MPOL_LOCAL`内存策略启动整个进程树（`/root/workspace/data/dpa_prefill/no_numa_scan.py`），NUMA均衡会跳过这些进程，驱逐为0，profile里没有冻结。
  两项合计：每步342.7 / 393.9 / 438.0 → 317.4 / 365.0 / 412.4 ms（−7%），e2e 70451 → 73042 tok/s（+3.7%），GSM8K 0.905（全量1319题 0.896）。
  根治需要主机上`numa_balancing=0`。本机偶尔也有其他租户的作业共用这8张卡。
- 4.22里怀疑的SWA回收merge sort不是来源：它只在profile结束时出现长kernel。

1j：
- HIP runtime的SDMA（`hipMemcpyAsync`写对方IPC buffer，`hipStreamWriteValue32` / `hipStreamWaitValue32`做flag）：数据正确，全部走copy engine，
  但整条记录一次890–1150 µs（RCCL 444 µs），host入队每次250–550 µs；关掉推荐engine时数据出错。
- MORI的SDMA在实现里调用`hipStreamSynchronize`，要用只能自己基于它的device端API写。
- RCCL：4个一组508–518 µs，整条记录[n, 4288]一次444 µs，只有x8 440 µs，只有小字段59 µs；“x8 + 小字段”无论一组还是两次都没有收益。
  整条打包净约−45~−50 µs/层（−2 ms/step），需要gate_up按行跨度读X、量化输出写成记录。暂不做。

## 5. 后续工作（固定部署下的优先级）

| 优先级 | 事项 | 预期（32K全局token，每层） |
|---|---|---|
| 0a | ~~Profile（moers7配置，chunk 0–2）：每个算子统计时间、TFLOPS、带宽，与理论值对比，给出优化建议和优先级~~（results.md第5轮） | 已完成 |
| 1f | ~~MoE gate_up / down评估并优化~~（4.22）：两个kernel都受750 W功耗墙限制，gate_up是同功耗dense GEMM的87%，down的写出代价是能耗。A2经LDS中转保留（e2e −17 µs/层），8-wave down在真实路由下更慢，已回退 | 约−0.8 ms/step；剩余只有gate_up尾tile（约−25~−45 µs/层） |
| 1g | ~~通信走SDMA的评估~~（4.23）：MORI的SDMA零拷贝AG8单独349 µs（RCCL单tensor 390，4-tensor分组454，e2e 535–556），与GEMM并发时GEMM只慢约19%；MORI的API会阻塞host。AG8在关键路径上，分块重叠收不回成本，收益只来自直接替换（见1j） | 已评估 |
| 1j | ~~AG8打包 / SDMA~~（4.24）：HIP runtime的SDMA慢（890–1150 µs）且host开销大，MORI阻塞host；RCCL整条记录一次444 µs vs 现状508–518 µs，净约−2 ms/step，需要gate_up按行跨度读X、量化输出写成记录 | 已评估，暂不做（−2 ms/step） |
| 1k | ~~rank间偏斜~~（4.24）：去掉prefill前向开头的两处host同步（aiter prefill元数据、SP-SWA元数据），chunk 1/2的host迟到空闲降到0；15–24 ms的GPU冻结是主机NUMA自动均衡触发的KFD queue驱逐，用显式`MPOL_LOCAL`启动规避。合计每步−7%，e2e +3.7% | sglang改动未commit；主机应设`numa_balancing=0` |
| 1h | 融合MoE的IPC缓冲（512 MiB）和route行workspace（32K时2.19 GiB）改到启动时预分配，计入KV pool规划（现在在第一次SP prefill时分配） | 稳定性，性能不变 |
| 1i | DFlash：prompt里距末尾超过窗口1024的token跳过FC和draft KV（要处理prefix cache共享，未验证） | 约−2~−3 ms/step（估算） |
| 0 | ~~启动脚本改为`ROCM_QUICK_REDUCE_QUANTIZATION=FP`、`ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0`~~（4.2第6点）：序列并行落地后prefill只剩第0层的2次大AR，收益<0.2 ms/step，未启用（4.17） | 被1取代 |
| 1 | ~~sglang DP-attention通信改为序列并行+FP8 gather~~：已在`feat/mimo-dpa-prefill-opt`上实现（`SGLANG_DPA_SP_MOE`，4.17），只用于prefill大前向；每层通信3.4–3.8 → 2.73 ms，prefill每步−15.8%（含PyHIP FP8 GEMM），bench吞吐+9.9%，GSM8K不变 | 已完成 |
| 1b | ~~attention侧RS4/AG4~~：SWA层序列并行 + 其余层A2A o_proj（4.20），每步通信136 → 87–91 ms；另开prefill delayer、dense TP1、embedding复制 | 已完成（−48 ms/step，e2e +35%） |
| 1d | SP-SWA的临时页打包（cat/index/permute约7 ms/step）融合成一个kernel；~~DFlash的FC放到shard上算~~（4.21①，−7.2 ms/step）；FlyDSL冷启动编译（4.17） | −8~−11 ms/step |
| 1c | ~~MoE GEMM1：借鉴45afb765的GateUp LDS kernel~~：32K下45afb比本实现慢5%，按执行行只快约3.5%（4.22） | 关闭 |
| 1e | o_proj BF16 GEMM（M4096 N4096 K8192，hipBLASLt 561T）：调hipBLASLt/aiter tuned GEMM，或写gfx942 BF16 PyHIP kernel（目前没有，4.20） | 到650T约−3 ms/step（估算） |
| 2 | 把“RS→norm→quant→AG”做成IPC自定义kernel（A2A o_proj之后attention侧已没有RS4，要按现在的流程重新评估）；~~MoE侧把RS融进sum kernel~~（4.21②：PyHIP MoE + 融合sum/RS8已接入sglang，MoE + RS每步−40 ms，e2e +7.1%） | attention侧待测 |
| 3 | ~~MoE：compact M64尾块~~（已实现，4.14）：单卡8K/16K/32K为−3%/−3%/−1%，整层（down_ar版）−49/+5/−101 µs | 已完成 |
| 3c | ~~MoE侧用down + sum + AR单kernel（4.10）代替down + 融合sum_ar~~：SP布局下用RS8不用AR，已由4.21②取代 | 不适用 |
| 4 | ~~MoE：FP8 gather落地后，PyHIP直接吃FP8输入，去掉入口量化~~（4.21②：SP路径直接传FP8 + transposed scale） | 已完成 |
| 不做 | down改用BF16 atomic累加到[T, H] + 融合AR（4.12：kernel内atomic只有约0.9 TB/s，down单独32K 2.47 ms；融合后1.03/1.22/1.24x，数值误差约1.7倍且不确定） | 更慢 |
| 不做 | token分两半、前半通信藏到后半gate_up下（3b，4.14：两半依次就+319/+435 µs，重叠收不回）；gemm1融进同一kernel的1-stage M64（4.15：已实现、逐位正确，整个MoE慢3%/6%/9%，W13/W2复用只有1/4；改成tingqli式4-wave × 每CU 2个CTA更慢，慢11%–13%，L2未命中读翻倍；保留为`PYHIP_MOE_DOWN=fused`） | 更慢 |
| 不做 | 1-stage + BF16 atomic + 每CU 2个WG（4.5：atomic与gemm1的读在fabric上争用，约1.0 ms无法掩盖，与store+sum相当） | 约±0，开发量与spill风险大 |
| 不做 | 1-stage + 普通store（4.5(3)：store不再卡MFMA，但功耗墙下时钟下降，暴露约450 µs）、移植mega_moe_tp（4.6） | 无收益 |
| 不做 | AR与down双stream并发（4.8：争用，比依次执行慢）；去掉atomic的1-stage + 融合sum+AR（4.9：持平） | 无收益 |
| 不做 | TBO / 双stream重叠AR（4.4：藏住0%）；`SGLANG_DP_HIERARCHICAL_COMM`（4.2：1.00x）；aiter CA上限调大到prefill尺寸（4.2：比RCCL慢5%–6%） | 无收益 |
| 不做 | MoE EP2×TP4 / EP4×TP2（4.20：kernel −0.32 ms/层，但专家分组逐层不均衡让RS8等待+15~20 ms/step） | 净值更慢 |
| 待确认（有损） | o_proj在线量化FP8 + PyHIP FP8 GEMM（4.20：303–340 µs vs 490 µs，rel 1.7e-3 → 3.7e-2；ckpt有意不量化o_proj） | −7~−9 ms/step |
| 待确认（有损） | RS载荷int8 + FP32 scale/32（4.6，rel_l2 0.71% → 0.83%） | 通信再 −0.7 ms/层（32K） |
| 待确认（有损） | Y（route rows）用int8 + FP32 scale/32（4.6，rel_l2 0.60% → 0.81%） | down+sum −0.25~0.3 ms（估算） |
