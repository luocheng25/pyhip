# Paged Prefill 4-wave/8-wave 优化与性能报告

更新时间：2026-09-02。当前实现已完成gfx942/gfx950 BF16、架构原生FP8、
non-SWA及gfx950 SWA支持。本文件以gfx950当前结果为主；2026-08的gfx942数据统一
归档在“历史结果”中，不与当前表直接横向比较。

## 当前结果总览

当前表均在MI350X `gfx950`上测量，主配置为
`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。除特别说明外，协议均为同进程、
同逻辑输入、20次预热、100个CUDA event样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | AITER | 4-wave | 8-wave | 结论 |
|---|---|---:|---:|---:|---|
| non-causal `Q10240,KV2583` | BF16 | **259.93 us / 1042.00T** | 514.22 us / 526.72T | 425.36 us / 636.75T | AITER varlen最快；4/8-wave分别为0.505x/0.611x |
| causal `Q=KV=32768` | BF16 | 15454.06 us / 355.74T | 8917.36 us / 616.50T | **8283.14 us / 663.70T** | 8-wave最快，较AITER快1.866x |
| causal `Q=KV=32768` | OCP FP8 K64 | N/A | **4651.73 us / 1181.83T** | 5101.77 us / 1077.58T | 4-wave较8-wave快1.097x |
| SWA `Q16K,KV128K,window=128` | BF16 | 121.74 us varlen-only / 266.53 us含gather | **94.44 us static** | 139.28 us | 4-wave较两种AITER口径快1.289x/2.822x；8-wave较含gather快1.914x |
| SWA `Q16K,KV128K,window=128` | OCP FP8 K64 | N/A | **67.05 us / 322.78T有效** | 112.35 us / 192.64T有效 | 4-wave较8-wave快1.676x |

口径说明：

- non-causal FLOPs为`2 * Hq * Q * KV * (Dqk + Dv)`；causal按三角有效工作量减半；
- SWA有效FLOPs按每行最多`window_left + 1 = 129`个可见KV token计算；硬件执行
  TFLOPS另计被mask但仍进入MFMA的tile工作，详见
  [gfx950_swa_performance.md](../pa_8wave/gfx950_swa_performance.md)；
- 每张明细表内的数据可以直接比较；不同架构、dtype、shape或计时协议的数据不混算提升；
- AITER等长D192/V128使用linear/page-size-1 batch-prefill ABI，非等长使用linear THD；
  FlyDSL使用page64 vectorized cache。三者共享逻辑Q/K/V，计时不包含布局转换。

## 当前功能范围

| 架构 | BF16 | 原生FP8 | non-SWA | SWA + sink | page size |
|---|---|---|---|---|---|
| gfx942 | 支持 | `float8_e4m3fnuz` | 支持 | 未作为当前生产基线 | 32/64/128 |
| gfx950 | 支持 | OCP `float8_e4m3fn` | 支持 | BF16/FP8支持 | 32/64/128 |

launcher会拒绝与目标架构不匹配的FP8格式，避免相同8-bit载荷按另一种指数/NaN编码
解释。non-SWA回归覆盖Dqk/Dv为128/128、192/128和192/192、causal/non-causal、
page 32/64/128、per-token/per-tensor Q scale及ragged tail。

## gfx950当前明细

### Non-SWA AITER BF16路由

跨4-wave、8-wave和AITER的BF16结果只保留在顶部总览。对比测试显式按长度分流：

| 条件 | AITER公开入口 | profiler验证的实际事件 |
|---|---|---|
| `Q == KV` | `mha_batch_prefill_func` | `aiter::mha_batch_prefill` |
| `Q != KV` | `flash_attn_varlen_func` | `FlashAttnVarlenFunc` |

默认小shape回归校验non-SWA路由互斥、SWA两种线性AITER入口和三方输出，结果为
`3 passed`。生产性能分别由`PYHIP_RUN_PA_AITER_PERF=1`和
`PYHIP_RUN_PA_AITER_SWA_PERF=1`显式开启。

### Non-SWA FP8 K16/K64 A/B

生产shape为`Q=KV=32768,causal`，算法工作量为`5.497558 TFLOP`。相同量化输入下
K16与K64输出逐bit一致。

| Kernel | K16延迟 / TFLOPS | K64延迟 / TFLOPS | 延迟下降 |
|---|---:|---:|---:|
| 4-wave | 5237.05 us / 1049.74T | **4651.73 us / 1181.83T** | **11.18%** |
| 8-wave | 5611.21 us / 979.75T | **5101.77 us / 1077.58T** | **9.08%** |

gfx950 OCP FP8 QK使用`v_mfma_f32_32x32x64_f8f6f4`，并通过
`scale_a=scale_b=0`表达unity E8M0 scale。P@V reduction只有32，继续使用
`v_mfma_f32_32x32x16_fp8_fp8`。K64仅在Dqk可被64整除时启用。

### SWA

生产配置为`Q=16K,window_left=128,page=64,bottom-right causal`，带per-head FP32
sink。4-wave按query tile裁剪page table，窗口外page不会被读取；batch=1默认走
static，`force_dynamic_schedule=True`时走persistent。

BF16 scheduler sweep结果：

| Total KV | 4-wave static | 4-wave dynamic | 8-wave persistent | Dynamic vs 8-wave |
|---:|---:|---:|---:|---:|
| 32K | 103.95 us | 110.31 us | 141.64 us | 1.284x |
| 64K | 103.89 us | 110.34 us | 141.99 us | 1.287x |
| 128K | **103.94 us** | 110.26 us | 142.15 us | 1.289x |

128K final-source dtype与K64结果：

| dtype | 4-wave static | 8-wave persistent | 4-wave vs 8-wave |
|---|---:|---:|---:|
| BF16 | **103.15 / 104.01 us** / 209.82 / 208.08T有效 | 141.04 / 142.53 us / 153.45 / 151.85T有效 | 1.367x / 1.370x |
| OCP FP8 K16 | **71.64 us / 302.10T有效 / 599.52T执行** | 112.90 us / 191.70T有效 / 570.63T执行 | 1.576x |
| OCP FP8 K64 | **67.05 us / 322.78T有效 / 640.56T执行** | 112.35 us / 192.64T有效 / 573.43T执行 | **1.676x** |

SWA AITER有两种计时口径：

1. **含gather端到端**：从5D cache gather到linear K/V，再调用AITER；
2. **attention-only**：K/V已是linear布局，只计AITER attention kernel。

同进程、同输入、20次预热、100样本、5轮中位数：

| KV | gather + `mha_batch_prefill_func` | `mha_batch_prefill_func` only | gather + `flash_attn_varlen_func` | `flash_attn_varlen_func` only | 4-wave static | 8-wave persistent |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 275.37 us / 78.59T | 248.67 us / 87.03T | 145.51 us / 148.74T | 115.32 us / 187.67T | **92.08 us / 235.04T** | 138.43 us / 156.34T |
| 64K | 316.81 us / 68.31T | 248.39 us / 87.13T | 181.83 us / 119.03T | 117.52 us / 184.16T | **93.34 us / 231.87T** | 139.57 us / 155.07T |
| 128K | fault，未计时 | fault，未计时 | 266.53 us / 81.20T | 121.74 us / 177.78T | **94.44 us / 229.17T** | 139.28 us / 155.39T |

两种线性AITER API都正确支持SWA：`mha_batch_prefill_func`实际命中
`aiter::mha_batch_prefill`，`flash_attn_varlen_func`命中`aiter::mha_varlen_fwd`；
两者小shape最大绝对差为`0.0078125`。varlen更快且覆盖128K，因此是当前推荐的
attention-only AITER基线。真正从5D page64 cache直连时，只有
`mha_batch_prefill_func`具备paged ABI，但当前gfx950 BF16 D192/V128构建没有匹配
specialization；`flash_attn_varlen_func`是linear THD接口，不是direct-paged入口。

### 当前资源

gfx950 OCP FP8 D192 fresh ISA：

| Kernel | QK / P@V静态站点 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave K16 | 24 K16 / 16 K16 | 180 | 37 | 16384 B | 0 B | 0 | 0 |
| 4-wave K64 | 6 K64 / 16 K16 | 180 | 37 | 16384 B | 0 B | 0 | 0 |
| 8-wave K16 | 48 K16 / 32 K16 | 174 | 72 | 12292 B | 0 B | 0 | 0 |
| 8-wave K64 | 12 K64 / 32 K16 | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64把QK静态MFMA站点减少4倍；4-wave资源不变，8-wave仅增加2个VGPR。

gfx950 BF16 SWA specialization：

| Kernel | 调度 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave | static | 249 | 91 | 25600 B | 0 B | 0 | 0 |
| 4-wave | dynamic | 254 | 91 | 25604 B | 0 B | 0 | 0 |
| 8-wave | persistent | 228 | 88 | 36868 B | 0 B | 0 | 0 |

### 当前验证

| 范围 | 结果 |
|---|---|
| 4-wave合并测试文件 | `49 passed, 2 skipped`；skip为两项可选生产性能测试 |
| 8-wave gfx950完整矩阵 | 原矩阵`65 passed, 1 skipped` |
| AITER non-SWA路由 + SWA双入口 + 三方正确性 | `3 passed` |
| non-SWA K16/K64 | 各kernel内逐bit一致 |
| SWA K16/K64 | 各kernel内逐bit一致；4/8-wave relative-L2 `6.3745e-5` |
| focused FP8 ISA | 0 private、0 spill、0 scratch |
| persistent counter复用 | 默认stream连续16次、两个非默认stream各8次均通过 |

## 当前实现与已完成优化

- 4-wave为`BM128 x BN32 x 256 threads`；batch=1走static grid，batch>1走
  atomic-ticket persistent grid；
- 8-wave为`BM256 x BN32 x 512 threads`，使用1 WG/CU persistent调度；
- K使用LDS ping-pong，V从global直接进入register，output使用半块C-shuffle；
- online softmax使用raw-max、lazy rebase和loop-carried max/sum；
- 32x32 MFMA减少row max/sum cross-lane开销，`v_permlane`避免经LDS交换；
- FP8采用gfx950 K64 QK、K16 P@V、score MUL split11和FP8-only fast-math；
- BF16/FP8共享pipeline骨架，dtype专属K搬运、scheduler、probability、V布局和epilogue
  地址封装在独立helper中。

Dynamic scheduler工作已全部完成：

- 消除persistent BF16 D192 spill，在消费点重物化C-shuffle地址和SWA mask坐标；
- 按device/stream/grid复用`work_counter`，最后退出的workgroup在device端复位；
- ticket改为4-byte LDS mailbox广播，每个ticket由双barrier缩减为单barrier；
- ticket barrier同时封闭前一work item的C-shuffle生命周期，删除重复re-entry barrier；
- 4-wave的1/2/3/4 WG-per-CU sweep选择2 WG/CU；8-wave选择1 WG/CU；
- 最终trace中static/dynamic kernel为97.34/105.52 us，稳态dynamic没有额外初始化dispatch。

版本：4-wave `a1b7e861c9373103994e295412aa9cf6b09da2758515070fafb14bf4ec4a0b69`；
8-wave `f9e68913a906d6b4c20935f920bc528d3ec6341f085a642b1e7be7e6f2a2f815`。

## 复现当前结果

```bash
cd /root/workspace/luocheng/pyhip
export HIP_VISIBLE_DEVICES=7
export FLYDSL_RUNTIME_ENABLE_CACHE=0

PA_CASE=tails PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=batch PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=noncausal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=causal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_DTYPE=bf16 PA_CASE=bf16_ref_short PA_NUM_ITERS=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py

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

测试前用`rocm-smi --showuse`选择空闲GPU。完整causal reference需要约64GB临时显存；
定频诊断流程见`tests/flydsl/H3_ATTENTION_THROTTLE_PROFILE.md`。

## 历史结果（gfx942）

以下数据来自MI308X/gfx942及2026-08版本，仅用于记录优化演进，不代表当前gfx950性能。

### 2026-08-14 BF16 D192 spill/occupancy验收

目标只有两项：

1. 为8-wave补齐BF16 `Dq=192,Dv=128`，修复其tail/causal mask和dtype编译缓存，
  并增加D128/D192 BF16回归；FP8路径保持兼容。
2. 修复4-wave batch>1 persistent BF16 D192的低occupancy分配。原
   `vgpr_count=265`实际为`256 VGPR + 9 AGPR`，并非普通VGPR越过硬件上限；
  将`rocdl.waves_per_eu=2`直接附着到persistent GPU kernel后，最终LLVM IR带有
  `amdgpu-waves-per-eu="2"`。page32/128变为`256 VGPR + 0 AGPR / 10 spill / 44B private`，
  page64 non-causal为`22 spill / 92B private`，causal为`16 spill / 68B private`；
  三种page size均为`256 VGPR + 0 AGPR`、2 waves/SIMD。

page64 causal的16个spill均为长寿命地址值，并非Q/K/V fragment或O accumulator：16个
store仅在workgroup入口执行一次；每个persistent work item分别有3个初始化reload、2个
masked-tail入口reload和11个C-shuffle epilogue reload。重复执行的KV fast/tail循环体内
没有scratch指令。

验收使用MI308X/gfx942、1300MHz、`Hq=16,Hkv=1,Q=KV=32768,causal,page_size=32`，
每组10次预热、两轮各50个event样本，顺序为`4w -> 8w -> 8w -> 4w`。

| shape | batch | 4-wave | 8-wave | 关键资源 |
|---|---:|---:|---:|---|
| BF16 D128 | 1 | 23.983 ms / 183.381T | 30.054 ms / 146.340T | 8-wave：8 spill / 36B private |
| BF16 D128 | 4 | 97.725 ms / 180.017T | 116.349 ms / 151.202T | 4-wave：228 VGPR / 0 spill |
| BF16 D192 | 1 | 34.468 ms / 159.497T | 37.770 ms / 145.555T | 8-wave：23 spill / 96B private |
| BF16 D192 | 4 | 137.813 ms / 159.566T | 146.258 ms / 150.352T | 4-wave：10 spill / 44B private |

4-wave D192 batch=4修复前为`113.419T`，修复后提升`40.69%`。四组4/8-wave输出
均finite；relative-L2为D128 batch=1/4 `3.7173e-5/3.8095e-5`，D192 batch=1/4
`3.8662e-5/3.8175e-5`。8-wave BF16 D192短尾、causal及FP8聚焦用例通过。

回归：8-wave `39 passed`、4-wave `15 passed`、公开API `11 passed`。

### 2026-08-11 4-wave/8-wave基线

当时的8-wave参考使用per-tensor Q量化；4-wave接收同一份FP8 Q和
等值descale。两者在同一进程使用10套buffer、各10次预热和50个CUDA event样本，采用
位置平衡顺序；“相对8-wave”为25组配对时间比中位数的倒数。

| 场景 | 时钟 | 实现 | 调度 | 中位延迟 | Actual TFLOPS | 相对8-wave |
|---|---|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | **4-wave** | static | **671.343 us** | **403.441** | **1.224x** |
| 同上 | auto | 8-wave | persistent | 821.883 us | 329.544 | 1.000x |
| causal `Q=KV=32768` | auto | **4-wave** | static | **17918.507 us** | **306.809** | **1.059x** |
| 同上 | auto | 8-wave | persistent | 18872.409 us | 291.301 | 1.000x |
| causal `Q=KV=32768` | 1300MHz | **4-wave** | static | **18836.170 us** | **291.862** | **1.061x** |
| 同上 | 1300MHz | 8-wave | persistent | 19992.191 us | 274.985 | 1.000x |

non-causal 25/25组获胜；causal auto 24/25组获胜；causal 1300MHz 25/25组获胜。
causal按三角有效FLOPs计数；auto-DPM存在双态，因此同时保留1300MHz结果。

下表沿用page32性能基线；page64/page128已通过功能精度回归，尚未单独建立性能基线。

主shape的4-wave static/persistent同代码对照：

| 场景 | 时钟 | static | persistent | static收益 |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | 670.102 us / 404.188T | 839.263 us / 322.720T | **25.18%** |
| causal `Q=KV=32768` | 1300MHz | 18836.068 us / 291.863T | 19555.952 us / 281.119T | **3.85%** |

因此non-causal的400T主要依赖batch=1 static调度；persistent路径仍约323T。causal也受益于
static，但收益明显较小。两组static/persistent输出逐元素一致。8-wave始终使用persistent。

### 2026-08-10 4-wave性能矩阵

除H3使用3次预热和10样本外，其余formal结果均为10套buffer、10次预热和50样本中位数。
causal括号内为当次快档min。

| dtype | Dq/Dv | 调度 | 场景 | shape | 中位延迟 | Actual TFLOPS |
|---|---:|---|---|---|---:|---:|
| FP8 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 672.883 us | 402.518 |
| FP8 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 3065.972 us | 350.213 |
| FP8 | 192/128 | static | causal | `H16,Q=KV=32768` | 17802.427 us (13973.054) | 308.809 |
| BF16 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 1323.445 us | 204.653 |
| BF16 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 7475.309 us | 143.638 |
| BF16 | 192/128 | static | causal | `H16,Q=KV=32768` | 35486.656 us (25454.100) | 154.919 |
| FP8 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 2500.491 us | 343.530 |
| FP8 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 208.721 us | 257.219 |
| FP8 | 128/128 | static | causal | `H1,Q=KV=32768` | 1137.484 us | 241.654 |
| BF16 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 3422.933 us | 250.952 |
| BF16 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 268.201 us | 200.175 |
| BF16 | 128/128 | static | causal | `H1,Q=KV=32768` | 1650.006 us | 166.592 |
| FP8 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 86.369 ms | 331.755 |
| BF16 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 179.958 ms | 159.223 |

### 2026-08-10 精度矩阵

`diff`为`pyhip.calc_diff`对PyTorch reference；全部通过`rtol=atol=0.1`和finite检查。

| dtype | Dq/Dv | ragged最大diff | batch=4 diff | small causal diff | 主shape/额外验证 |
|---|---:|---:|---:|---:|---|
| FP8 | 192/128 | `2.8836e-4` | `3.4356e-4` | `1.7518e-4` | 主non-causal `3.6652e-4` |
| BF16 | 192/128 | `2.5129e-6` | `2.7224e-6` | `1.9344e-6` | 主non-causal `2.8093e-6` |
| FP8 | 128/128 | `2.6076e-4` | `3.4029e-4` | `1.7112e-4` | H3 finite |
| BF16 | 128/128 | `2.4619e-6` | `2.7061e-6` | `1.8679e-6` | H3 finite |

ragged覆盖`KV=3/13/23/53/83`，small causal为`Q=KV=256`。4-wave/8-wave同输入的
non-causal与causal relative-L2分别为`1.17e-4`和`1.12e-4`。

### 2026-08-10 specialization资源

- block为`BM128 x BN32 x 256 threads`；每个workgroup 4个wave；
- K使用LDS ping-pong，V直接进入fragment，output使用两个半块C-shuffle；
- online softmax使用raw-max、lazy rebase和loop-carried max/sum；
- FP8使用QK `VMEM1 -> MFMA2`、score MUL split11和FP8-only fast-math；
- FP8 D192当时为168 combined VGPR、16KB LDS、0 scratch，自然达到3 waves/SIMD；
- BF16 D128使用专用scheduler/HW-slot priority，D192使用独立scheduler；
- BF16/FP8共享pipeline时序骨架，dtype专属K搬运、scheduler、probability写回、V布局、
  epilogue地址和compile hint均封装在独立helper。

refactor前后fresh执行ISA逐条一致：

| specialization | ISA资源 | MFMA |
|---|---|---:|
| FP8 D192 | 168 VGPR-form / 16KB / 0 scratch | 80 |
| FP8 D128 | 153 VGPR-form / 16KB / 0 scratch | 64 |
| BF16 D192 | 250 VGPR-form / 25KB / 0 scratch | 160 |
| BF16 D128 | 214 VGPR-form / 17KB / 0 scratch | 128 |

FP8 D192 dynamic persistent kernel的执行ISA同样逐条一致。

## 优化里程碑（2026-08）

以下记录保留改变实现或建立关键反证的里程碑，性能数字均为当时版本结果。

### 2026-08-10：4-wave pipeline与C-shuffle

- **改动**：建立MMA32骨架；K走LDS ping-pong，V直读；接入paged ABI、GQA、ragged和causal；
  output改为两个64x128半块C-shuffle。
- **验证**：反转page table、跨页和ragged尺寸通过。
- **结果**：约`1838 -> 1465 -> 1008 -> 915 us`；保留双缓冲和半块C-shuffle。

### 2026-08-10：static dispatch与causal均衡

- **改动**：batch=1改用static grid；batch>1保留persistent；causal使用
  `(251 * tile + 251) % 256`映射。
- **验证**：non-causal、batch=4和long causal通过。
- **结果**：short调用约`54 -> 10 us`；causal约`17.9 -> 16.7 ms`；保留static/仿射路径。

### 2026-08-10：双K流水与priority

- **改动**：形成`K(i+2)`预取、softmax、`K(i+1)`写入、PV/barrier/K-read跨回边流水；
  FP8统一stage priority为`0/2`。
- **验证**：双K统一priority反相稳定；HW-slot priority回退。
- **结果**：主路径约876--880us；保留双K与统一priority。

### 2026-08-10：raw-max与softmax调度

- **改动**：先对raw score做max/shuffle，再用score scaling覆盖等待；FP8增加固定切分。
- **验证**：FP8/BF16数值不变；shuffle wait由约55降至10.7/17.2 cycles。
- **结果**：raw-max约提升3%，split8再提升1.33%；BF16不采用split8。

### 2026-08-10：BF16与H3

- **改动**：加入D128/D192 BF16 MMA、K/V layout、128-bit copy、LDS padding和D128 scheduler。
- **验证**：BF16 ragged/batch/causal及真实H3通过。
- **结果**：当时BF16 D128为250.952T，D192为204.653T，H3为159.223T。

### 2026-08-10：FP8自然3-wave

- **改动**：epilogue重建C-shuffle地址，将资源从176降至168 combined VGPR；固定gap2、
  score MUL split11和FP8-only fast-math。
- **验证**：16KB LDS、0 scratch、80 MFMA；最终ATT三槽`2+1`混合相90.36%。
- **结果**：当时non-causal为402.518T；4-wave相对当时8-wave为1.222x。

### 2026-08-11：8-wave参考复测

- **改动**：使用当时的8-wave per-tensor Q和多page-size实现，共享输入位置平衡复测；
  causal额外使用1300MHz固定频率。
- **验证**：page32/64/128 short reference通过；定频结束后恢复auto。
- **结果**：non-causal为1.224x；causal auto/1300MHz分别为1.059x/1.061x。

### 2026-08-11：代码refactor

- **改动**：保留共享pipeline，将BF16/FP8 K搬运、scheduler、probability、V布局、epilogue和
  compile hint封装为helper；删除恒真/恒假参数并统一命名。
- **验证**：4种static specialization和FP8 dynamic persistent执行ISA逐条一致；完整精度矩阵通过。
- **结果**：性能矩阵与重构前一致；共享时序骨架、独立dtype细节，不复制两份pipeline。

## 已否决实验

| 类别 | 关键证据 | 保留方案 |
|---|---|---|
| K copy 128-bit | 隐式`vmcnt(0)`，回退15.4%--15.5% | 64-bit K copy |
| 阶段/HW-slot | 循环slot回退2.76%；入口复制pipeline回退16.71% | 单pipeline、FP8统一priority |
| barrier/PV/K写 | 隔页barrier等待约124增至403 cycles；PV切分增scratch | 每页barrier、完整PV |
| page-table pair load | VGPR 168升至172，失去3-wave | 标量lookahead |
| 映射/priority | tile-major回退0.5%；反向priority约回退3% | head-major、`0/2` |
| 数值调度 | sum多链/非均匀gap无收益；显式rcp仅397.112T | 单链sum、gap2、fast-math |
| BF16实验 | split8中性；D192 shape峰值约210.7T | 原BF16 softmax、独立D192优化 |

## ATT证据

- FP8 D192：`tests/flydsl/pa_4wave/att_fp8_d192_3wave/ui_output_agent_28524_dispatch_66`；
- BF16 D192：`tests/flydsl/pa_4wave/att_bf16_d192/ui_output_agent_32152_dispatch_13`；
- FP8主要stall/MFMA：MFMA 36.674、VALU 12.619、barrier 7.413、VMEM-load 6.397、
  LDS-wait 5.900；两条barrier约128/145 cycles。
