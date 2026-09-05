# gfx950 SWA paged-prefill results

## Environment

- GPU: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`
- ROCm: 7.2
- rocprofv3: 1.1.0
- Input: BF16 or OCP FP8 (`torch.float8_e4m3fn`), Hq=16, Hkv=1,
  Dqk=192, Dv=128, page size 64
- SWA: bottom-right causal, `window_left=128`, per-head FP32 sink
- Benchmark: Q=16384, 20 warmups, 100 timed iterations, 5 runs; table uses median

gfx950 must use OCP E4M3 FN. gfx942 uses E4M3 FNUZ instead; the encodings are not
interchangeable, and both launchers reject the non-native format.

## Performance

| Total KV | gfx942-style K8 baseline | gfx950 initial K16 | Compute-optimized K16 | Scheduler-optimized K16 | Speedup vs K8 |
|---:|---:|---:|---:|---:|---:|
| 32K | 257.65 us | 177.07 us | 162.21 us | 141.55 us | 1.820x |
| 64K | 253.02 us | 177.03 us | 161.42 us | 141.65 us | 1.786x |
| 128K | 249.94 us | 176.61 us | 162.17 us | 141.39 us | 1.768x |

Final K16 plus persistent-scheduler optimization runs, in us:

| Total KV | Five runs | Median |
|---:|---|---:|
| 32K | 155.86, 141.52, 141.58, 141.44, 141.55 | 141.55 |
| 64K | 155.20, 141.45, 141.65, 141.70, 141.52 | 141.65 |
| 128K | 154.65, 141.04, 141.39, 141.09, 141.54 | 141.39 |

Replacing packed FP32 score/output arithmetic with scalar `v_mul_f32` and
`v_sub_f32` reduced the 128K control from about 177 us to about 170 us. Moving
the matured K-ring wait/store from the softmax phase to the shorter PV/MFMA
phase reduced it again to about 161 us. Relative to the scalar pre-balance
control, this is a 5.50% reduction at 128K.

Porting the 4-wave scheduler changes then removed per-call counter allocation,
fill and seed copy, replaced the global mailbox and two ticket barriers with a
four-byte LDS mailbox and one barrier, and reused that barrier for final
C-shuffle re-entry. This reduced the compute-optimized 8-wave latency by
12.25%-12.81%. A 1/2/3/4 workgroup-per-CU sweep retained one: B=4 measured
295.49/298.49/300.95/302.72 us, while B=1's preference for two was only 0.4%.

The historical gather+AITER comparison below was measured before the gfx950
instruction update and is retained only for chronology:

| Total KV | Gather | Gather+AITER | Optimized FlyPA | Historical end-to-end speedup |
|---:|---:|---:|---:|---:|
| 32K | 31.77 us | 268.34 us | 161.04 us | 1.666x |
| 64K | 60.28 us | 301.89 us | 160.89 us | 1.876x |
| 128K | 115.54 us | 389.46 us | 160.68 us | 2.424x |

Those historical end-to-end ratios use the earlier stable reference values.
The current same-process results are reported below. Direct 5D FlyPA removes
20/40/80 MiB of temporary K/V and 40/80/160 MiB of gather read+write traffic
at 32K/64K/128K.

## 4-wave versus AITER

The gfx950 SWA-enabled 4-wave, 8-wave, and current AITER baselines use the same
production shape and protocol as above: Q=16384, 20 warmups, 100 timed
iterations, five runs, with the table reporting the median. All candidates use
the same seed-1 logical tensors in one process and alternate measurement order.
Outputs were compared with `rtol=atol=2e-2`.

By default, batch=1 selects the 4-wave static grid. The `force_dynamic_schedule`
control sends the identical B=1 workload through the ticket-based persistent
kernel used for batch>1.

Current same-process AITER baselines distinguish end-to-end gather cost from
attention-only cost. All entries below use the same logical Q/K/V, 20 warmups,
100 samples, and five round medians:

| Total KV | Gather + batch-prefill | Batch-prefill only | Gather + varlen | Varlen only | 4-wave static | 8-wave persistent | Direct paged AITER |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32K | 275.37 us / 78.59T | 248.67 us / 87.03T | 145.51 us / 148.74T | 115.32 us / 187.67T | **92.08 us / 235.04T** | 138.43 us / 156.34T | unsupported |
| 64K | 316.81 us / 68.31T | 248.39 us / 87.13T | 181.83 us / 119.03T | 117.52 us / 184.16T | **93.34 us / 231.87T** | 139.57 us / 155.07T | unsupported |
| 128K | fault / not timed | fault / not timed | 266.53 us / 81.20T | 121.74 us / 177.78T | **94.44 us / 229.17T** | 139.28 us / 155.39T | unsupported |

Here batch-prefill means `mha_batch_prefill_func`, which emits
`aiter::mha_batch_prefill`; varlen means `flash_attn_varlen_func`, which emits
`aiter::mha_varlen_fwd`. Both linear paths match numerically, but varlen is
faster and remains valid at 128K. Attention-only assumes linear K/V already
exist; it is not a direct 5D-cache result. Only `mha_batch_prefill_func` accepts
the vectorized paged ABI, and this AITER build has no matching gfx950 BF16
D192/V128 page64 specialization. `flash_attn_varlen_func` accepts linear THD,
not vectorized 5D cache.

Historical isolated-process scheduler comparison (superseded by the
same-process table above):

| Total KV | Gather+AITER | 4-wave static | 4-wave dynamic | 8-wave | AITER vs dynamic | Dynamic vs 8-wave | Dynamic slowdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 267.79 us | 103.95 us | 110.31 us | 141.64 us | 2.428x | 1.284x | 6.12% |
| 64K | 300.57 us | 103.89 us | 110.34 us | 141.99 us | 2.724x | 1.287x | 6.21% |
| 128K | fault | 103.94 us | 110.26 us | 142.15 us | N/A | 1.289x | 6.08% |

After eliminating spill and scheduler setup/synchronization overhead, forced
dynamic is only 6.1%-6.2% slower than static and is 1.284x-1.289x faster than
8-wave on these inputs.

### FP8 SWA

The same Q=16K, total-KV=128K production input was quantized to gfx950-native
OCP FP8 with per-token Q scale and independent per-tensor K/V scales. Both
direct-5D kernels retain BF16 output and include the same causal window and
sink logits.

| Kernel | BF16 repeat medians | BF16有效TFLOPS | OCP FP8 K64 median | FP8有效TFLOPS | FP8 speedup range |
|---|---:|---:|---:|---:|---:|
| 4-wave static | 103.15 / 104.01 us | 209.82 / 208.08T | 67.05 us | 322.78T | 1.538x-1.551x |
| 8-wave persistent | 141.04 / 142.53 us | 153.45 / 151.85T | 112.35 us | 192.64T | 1.255x-1.269x |

Both outputs are finite and pass the dequantized-BF16 reference tolerance.
K64 and K16 are bitwise identical within each kernel; 4-wave/8-wave
relative-L2 remains `6.3745e-5`. K64 FP8 4-wave is 1.676x faster than K64 FP8
8-wave for this SWA shape.

Same-process, same-input K16/K64 A/B results:

| Workload | Kernel | K16延迟 / TFLOPS | K64 QK延迟 / TFLOPS | Reduction |
|---|---|---:|---:|---:|
| non-SWA Q=KV=32K | 4-wave | 5237.05 us / 1049.74T | 4651.73 us / 1181.83T | 11.18% |
| non-SWA Q=KV=32K | 8-wave | 5611.21 us / 979.75T | 5101.77 us / 1077.58T | 9.08% |
| SWA Q=16K, KV=128K | 4-wave | 71.64 us / 302.10T有效 / 599.52T执行 | 67.05 us / 322.78T有效 / 640.56T执行 | 6.41% |
| SWA Q=16K, KV=128K | 8-wave | 112.90 us / 191.70T有效 / 570.63T执行 | 112.35 us / 192.64T有效 / 573.43T执行 | 0.49% |

non-SWA TFLOPS使用测试中的causal三角有效FLOPs，共`5.497558 TFLOP`。SWA
“有效”口径按每个query实际可见的129个KV token计算，共`0.021643 TFLOP`；
“执行”口径计入tile/page粒度上虽被mask但仍进入MFMA的矩阵工作，4-wave/8-wave
分别为`0.042950/0.064425 TFLOP`。因此跨4-wave/8-wave比较硬件利用率时应看
执行TFLOPS，比较模型有效工作时应看有效TFLOPS。

A fresh isolated compile of the gfx950 OCP FP8 D192 specialization reports
180 VGPR / 37 SGPR / 16384-byte LDS for 4-wave and 176 VGPR / 72 SGPR /
12292-byte LDS for 8-wave. Both have zero private bytes, zero reported spills,
and no static scratch instructions.

| Kernel | QK / P@V静态站点 | VGPR | SGPR | LDS | Private | Spill | Scratch |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-wave K16 | 24 K16 / 16 K16 | 180 | 37 | 16384 B | 0 B | 0 | 0 |
| 4-wave K64 | 6 K64 / 16 K16 | 180 | 37 | 16384 B | 0 B | 0 | 0 |
| 8-wave K16 | 48 K16 / 32 K16 | 174 | 72 | 12292 B | 0 B | 0 | 0 |
| 8-wave K64 | 12 K64 / 32 K16 | 176 | 72 | 12292 B | 0 B | 0 | 0 |

K64把QK静态MFMA站点缩减4倍；P@V站点不变。4-wave资源完全不变，8-wave仅
增加2个VGPR，未改变当前1 WG/CU选择，也未引入private、spill或scratch。

Final-source five-run details, in us. BF16 was repeated in two independent
processes; its median spread was 0.83% for 4-wave and 1.06% for 8-wave.

| Dtype | Run | Kernel | Samples |
|---|---|---|---|
| BF16 | A | 4-wave | 118.04, 103.46, 104.01, 104.03, 103.78 |
| BF16 | A | 8-wave | 143.66, 142.53, 142.48, 142.54, 142.52 |
| BF16 | B | 4-wave | 117.13, 102.99, 103.15, 103.00, 103.54 |
| BF16 | B | 8-wave | 141.83, 140.91, 140.88, 141.04, 141.04 |
| OCP FP8 K64 | A | 4-wave | 69.09, 66.73, 65.88, 67.33, 67.05 |
| OCP FP8 K64 | A | 8-wave | 112.25, 112.32, 117.01, 112.75, 112.35 |

Five-run details, in us:

| Total KV | Component | Runs |
|---:|---|---|
| 32K | Gather | 31.80, 31.52, 31.52, 31.52, 31.50 |
| 32K | AITER linear | 238.30, 236.22, 236.84, 235.84, 237.20 |
| 32K | Gather+AITER | 267.79, 268.03, 267.79, 268.05, 267.73 |
| 32K | 4-wave static | 117.08, 103.71, 103.95, 103.76, 104.12 |
| 32K | 4-wave dynamic | 111.60, 109.94, 110.19, 110.31, 110.40 |
| 32K | 8-wave | 141.50, 141.17, 141.82, 141.64, 142.09 |
| 64K | Gather | 60.27, 59.73, 59.83, 59.85, 59.77 |
| 64K | AITER linear | 237.87, 235.98, 237.30, 236.06, 237.16 |
| 64K | Gather+AITER | 300.35, 300.57, 300.17, 300.59, 300.57 |
| 64K | 4-wave static | 114.67, 103.50, 103.81, 103.89, 104.02 |
| 64K | 4-wave dynamic | 110.84, 110.11, 110.14, 110.34, 110.39 |
| 64K | 8-wave | 141.99, 141.86, 141.99, 141.76, 142.16 |
| 128K | Gather | 115.89, 116.85, 115.51, 116.58, 115.50 |
| 128K | 4-wave static | 115.45, 103.94, 103.85, 103.82, 104.28 |
| 128K | 4-wave dynamic | 112.73, 110.06, 110.41, 110.26, 110.24 |
| 128K | 8-wave | 142.64, 141.87, 142.50, 141.90, 142.15 |

On the current AITER checkout, the 128K Dqk=192/Dv=128 linear attention call
aborts its isolated process with a GPU memory-access fault before timing. The
GPU recovered and the isolated 128K gather and 4-wave measurements remained
finite. The earlier stable 128K Gather+AITER result in the historical table is
389.46 us; comparing that historical value with the current 110.67 us dynamic
4-wave median would give 3.519x, but it is not reported as a current-run
speedup.

### Static versus dynamic ATT analysis

The captures use the same seed-1 Q=16K/KV=128K input and the same target CU.
Static produces one workgroup per query-tile/head work item; dynamic launches
two workgroups/CU and lets each persistent workgroup fetch multiple tickets.
The target CU completes the same eight work items in both traces.

| ATT metric on target CU | Static | Dynamic | Change |
|---|---:|---:|---:|
| Traced waves | 32 short waves | 8 persistent waves | topology change |
| Dynamic MFMA | 10240 | 10240 | identical attention work |
| LDS reads / writes | 3712 / 1376 | 3744 / 1384 | +32 / +8 for ticket LDS |
| CU makespan | 166476 cycles | 178776 cycles | +7.4% |
| Effective cycles/work item | 20810 | 22347 | +1538 |
| Sum of wave lifetimes | 1279292 | 1369448 cycles | +7.0% |

Kernel tracing reports 2048 static workgroups (`16 heads * 128 Q tiles`) versus
512 dynamic workgroups (`256 CUs * 2`). Thus each dynamic workgroup processes
four work items on average; each of its four waves executes the equivalent of
four static waves serially. Final slot completion skew is 10908 cycles for
static and 13720 cycles for dynamic, so tail imbalance is not the dominant
source of the 12300-cycle makespan increase.

The instruction-class ATT totals below are sums across resident waves, so they
identify pressure sources but must not be added directly to CU makespan:

| Stall class | Static | Dynamic | Delta |
|---|---:|---:|---:|
| `s_waitcnt` | 75272 | 113716 | +38444 (+51.1%) |
| `s_barrier` | 37440 | 83796 | +46356 (+123.8%) |
| VMEM load | 138444 | 150512 | +12068 (+8.7%) |
| MFMA | 218040 | 216464 | -1576 (-0.7%) |
| Scratch | 0 | 0 | 0 |

The optimized ticket path performs one agent-scope atomic, broadcasts its
result through a four-byte LDS mailbox, and executes one workgroup barrier per
ticket. That barrier accounts for 24740 stall cycles in this target-CU trace.
The former second ticket barrier and epilogue re-entry barrier are absent.

Splitting each persistent wave at the two ticket barriers in the pre-fix trace
gave a direct per-work-item comparison:

| Wave interval | Median | Mean | Range |
|---|---:|---:|---:|
| Static work item, pre-fix diagnostic | 38658 cycles | 39766 cycles | 36404-47824 |
| Dynamic work-item compute, pre-fix diagnostic | 44002 cycles | 45344 cycles | 36228-58328 |
| Dynamic ticket handoff, pre-fix diagnostic | 1612 cycles | 2386 cycles | 224-10924 |

This pre-fix interval split first showed that ticket handoff was only part of
the gap and led to the spill investigation. The optimized ATT comparison above
supersedes its absolute timings. Dynamic now has only 32 additional barrier
hits: eight work items on the target CU times four waves times one ticket
barrier.

Consumer-side rematerialization now keeps both specializations spill-free:
static uses 249 VGPRs, while dynamic uses 254 VGPRs; both have zero private
bytes and zero scratch. Before this change, dynamic used 256 VGPRs, 60 private
bytes, 14 spills, and 560 executed scratch operations. Removing those spills
reduced dynamic latency from 146.73/147.43/150.01 us to
136.76/136.46/136.50 us, a 6.8%-9.0% improvement. In a matched dynamic ATT
capture, scratch operations fell from 560 to zero and summed `s_waitcnt` stall
fell by 113640 cycles. The target-CU makespan varied by +0.6% between the two
individual traces, so the five-run CUDA-event medians are the performance
decision metric.

The subsequent scheduler changes cache one counter per device/stream/grid,
reset its ticket/completion header from the last exiting workgroup, move ticket
broadcast into LDS, and remove the redundant barriers. They reduce the
spill-free dynamic medians further to 110.19/109.99/110.67 us, another
18.9%-19.4%. A 1/2/3/4 workgroup-per-CU sweep selected the existing value of
two for both B=1 and B=4.

Kernel-dispatch tracing gives the wall-time view for steady samples:

| GPU dispatch | Median |
|---|---:|
| Static 4-wave attention kernel | 97.34 us |
| Dynamic 4-wave attention kernel | 105.52 us |
| 8-wave attention kernel | 156.75 us |
| Per-call dynamic setup dispatches | 0 |

The final dynamic attention dispatch is about 8.18 us slower than static and
the steady dynamic section contains no counter-fill or seed-copy dispatches.
The residual gap is therefore inside the persistent kernel, primarily its
atomic ticket acquisition and one LDS-broadcast barrier per work item.

### Why 32K and 64K are flat

The direct kernels prune the page table per query tile. For a page-aligned,
sufficiently long prefix, the number of pages visited by one tile is

`(window_left + block_m) / page_size`.

With Q=16384, `window_left=128`, and page size 64:

| Kernel | Q tile | Tiles/head | Pages/tile | Page visits/head |
|---|---:|---:|---:|---:|
| 4-wave | 128 | 128 | 4 | 512 |
| 8-wave | 256 | 64 | 6 | 384 |

These counts are identical at total KV 32K, 64K, and 128K. Increasing the
prefix only changes `first_page`; it does not add attention work. Therefore the
direct-5D latency stays near 103 us for static 4-wave, 136-137 us for dynamic
4-wave before scheduler optimization, about 110 us after it, and about 141 us
for scheduler-optimized 8-wave. The gather kernel still reads and
writes every logical KV token, so it grows from about 32 us to 60 us and 116 us.

## ISA verification

Final ISA for the production 128K shape:

| Instruction/resource | Result |
|---|---:|
| `v_mfma_f32_32x32x16_bf16` static sites | 80 |
| `v_mfma_f32_32x32x8_bf16` static sites | 0 |
| `v_cvt_pk_bf16_f32` static sites | 64 |
| conversion-path `v_perm_b32` sites | 0 |
| `v_permlane32_swap_b32_e32` static sites | 5 |
| packed FP32 arithmetic sites | 0 |
| VGPR allocation | 228 |
| SGPR allocation | 88 |
| LDS | 36868 bytes |
| private segment | 0 bytes |
| scratch instructions in steady KV loop | 0 |
| total static scratch instructions | 0 |

The BF16 QK/PV paths directly build K16 tiled MMA objects with
`fx.make_mma_atom`, `fx.make_tiled_mma`, and `thr_slice`. gfx950 OCP FP8 builds
a K64 `MFMA_Scale` object for QK and a separate K16 object for P@V because
P@V reduces over only 32 probabilities. For D192, the 4-wave static ISA has
6 K64 QK and 16 K16 P@V sites; 8-wave has 12 K64 QK and 32 K16 P@V sites.
Compute uses `fx.gemm`; there are no direct `rocdl.mfma_*` calls. The
gfx950 path stores K lookahead blocks in a three-stage LDS ring instead of
carrying two eight-VGPR register fragments across the loop. Scalar FP32
hot-path operations and consumer-side address rematerialization keep the
current BF16 production kernel at 228 VGPRs and zero scratch over the whole kernel. The
xor-32 reductions lower to the gfx950 lane-swap instruction. The F32 vector
`.to(fx.BFloat16)` conversion lowers to deterministic packed RNE.

## TILE_N=64 experiment

A native physical `BN=64` variant was compiled and tested. For Dqk=192 it
required four b128 K atoms per participating thread and a two-slot K ring to fit
LDS. It compiled to 256 VGPRs, 64 bytes of private storage, 47 VGPR spills, and
43 static scratch instructions; the focused SWA case also produced non-finite
output. It was rejected before performance measurement.

The fallback evaluated each 64-token page as two sequential 32-column QK/PV
substeps while reusing `fragS`, `fragK`, and `fragV`. It restored 228 VGPRs and
zero scratch and passed correctness. Its 32K/64K/128K medians were
161.70/161.46/161.24 us, statistically the same as the existing page-local pair
of 32-column substeps. Because it generated the same 80 static MFMA sites and
provided no throughput gain, the extra logical-TILE_N layer was not kept.

## ATT latency hiding

Capture workload: Q=16K, total KV=128K, eight FlyPA-only launches. The final
trace selected warmed-up kernel iterations `[5,6]`, producing PID 65503,
dispatch 61 on SE0/CU0 with all four SIMDs and two wave slots per SIMD.

The statistics use the repository's physical-SIMD union-MFMA method:

- Successful issue time is `first_attempt + stall`.
- Each gfx950 `v_mfma_f32_32x32x16_bf16` contributes a 32-cycle busy window.
- The two resident wave slots on each physical SIMD are combined before measuring interval coverage.
- Percentages are weighted by the four physical SIMD lifetimes.

| Metric | Before scheduler port | Final source | Change |
|---|---:|---:|---:|
| Weighted physical-SIMD span | 1085944 cycles | 1063100 cycles | -2.10% |
| Target-CU makespan | 271504 cycles | 266280 cycles | -1.92% |
| Dynamic MFMA | 7680 | 7680 | identical work |
| Serialized / union MFMA cycles | 245760 / 245760 | 245760 / 245760 | identical |
| MFMA union coverage | 22.63% | 23.12% | +0.49 pp |
| Non-MFMA issue inside MFMA shadow | 60.89% | 61.18% | +0.29 pp |
| Physical no-issue | 38.84% | 37.56% | -1.28 pp |
| Barrier hits / summed stall | 1760 / 519304 | 1696 / 520024 | -64 / +0.14% |
| VMEM-load hits / summed stall | 4656 / 223328 | 4624 / 198204 | -32 / -11.25% |

The MFMA union still equals the serialized 32-cycle total, so resident slots do
not overlap MFMA busy windows. The final-source kernel-side span is 1.92%
shorter than the pre-scheduler capture. The historical scheduler-stage
12.25%-12.81% CUDA-event gain also includes deleting per-call counter
allocation/fill/seed work. Current kernel tracing measured a 156.75 us
instrumented steady median with no non-attention dispatches in the steady
section; two uninstrumented final-source CUDA-event runs measured 141.04 and
142.53 us at 128K.

## Artifacts

- Same-input static 4-wave UI: `att_ui_same_input_4wave_static_128k/`
	(PID 8398, dispatch 59; source SHA256 `a1b7e861c9373103994e295412aa9cf6b09da2758515070fafb14bf4ec4a0b69`)
- Same-input dynamic 4-wave UI: `att_ui_same_input_4wave_dynamic_128k/`
	(PID 23897, dispatch 61; source SHA256 `a1b7e861c9373103994e295412aa9cf6b09da2758515070fafb14bf4ec4a0b69`)
- Same-input 8-wave UI: `att_ui_same_input_8wave_128k/`
	(PID 65503, dispatch 61; source SHA256 `f9e68913a906d6b4c20935f920bc528d3ec6341f085a642b1e7be7e6f2a2f815`)
- UI input: seed 1, Q=16384, prefix=114688, total KV=131072
- Union analyzer: `analyze_att_union_mfma.py`
- Generated ISA: `my_ir_dumps/attn_kernel_0/22_final_isa.s`

## Correctness

Current gfx950 modules:

- Consolidated 4-wave + AITER module: `49 passed, 2 skipped`; the skips are the
	opt-in non-SWA and SWA production benchmarks.
- 8-wave: `65 passed, 1 skipped`

The matrices include native-FP8 non-SWA D128/D192 cases and native-FP8 SWA
single/ragged-batch cases in addition to the BF16 scheduler regressions.