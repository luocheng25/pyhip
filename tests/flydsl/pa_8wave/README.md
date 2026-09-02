# Paged Prefill 8-wave

更新时间：2026-09-02。完整的4-wave/8-wave/AITER统一数据、口径和gfx942历史结果见
[4-wave主报告](../pa_4wave/README.md)；本文件只维护8-wave实现及其当前摘要。

## 功能范围

- 支持gfx942/gfx950 BF16与架构原生FP8 vectorized paged-prefill；
- gfx942 FP8使用`torch.float8_e4m3fnuz`，gfx950使用OCP
  `torch.float8_e4m3fn`，launcher拒绝非原生编码；
- 覆盖D128/D192、causal/non-causal、page 32/64/128、per-token/per-tensor Q scale、
  ragged tail及gfx950 SWA + sink；
- gfx950完整回归：`65 passed, 1 skipped`。

## 当前性能摘要

平台为MI350X gfx950，主配置为`B=1,Hq=16,Hkv=1,Dqk=192,Dv=128,page=64`。
除特别说明外，计时为20次预热、100样本、5轮中位数；TFLOPS为算法有效FLOPs。

| 场景 | dtype | 8-wave结果 | 对照结论 |
|---|---|---:|---|
| non-causal `Q10240,KV2583` | BF16 | 425.36 us / 636.75T | AITER varlen为259.93 us；8-wave为0.611x |
| causal `Q=KV=32768` | BF16 | **8283.14 us / 663.70T** | AITER batch-prefill为15454.06 us；8-wave快1.866x |
| causal `Q=KV=32768` | OCP FP8 K16 | 5611.21 us / 979.75T | K64前基线 |
| causal `Q=KV=32768` | OCP FP8 K64 | **5101.77 us / 1077.58T** | 较K16延迟下降9.08% |
| SWA `Q16K,KV128K,window=128` | BF16 | 139.28 us | AITER varlen-only 121.74 us；含gather 266.53 us |
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

- workgroup为`BM256 x BN32 x 512 threads`，8个wave分成两组交织QK MFMA与
  online softmax；
- 1 WG/CU persistent grid通过atomic ticket遍历`batch x query tile x Q head`任务；
- K走`global -> LDS -> register` ping-pong，V走`global -> register`，降低LDS压力；
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

K64将QK静态MFMA站点减少4倍，仅增加2个VGPR。gfx950 BF16 SWA specialization为
228 VGPR、88 SGPR、36868 B LDS、0 private、0 spill、0 scratch。

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

