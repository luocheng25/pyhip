# MoE GEMM 8-wave pipeline

This note describes the `wg_M=256`, `wg_N=256`, FP8 block-scale, gate-up
instance of `moe_gemm_8wave_g1u1`. Counts are ISA instructions issued by one
wave, not tensor elements. A workgroup contains eight waves.

This kernel asserts `gate_up`; down projection remains owned by the specialized
`moe_gemm_down_tp` path. The balanced cross-iteration B-register schedule is
enabled only for `wg_M=256`. The `wg_M=128` body retains the original schedule
because its operand/register progression differs.

The runtime `wg_M=256` to `wg_M=128` adaptive branch is controlled separately:

```bash
# Default: disable the adaptive branch and always execute the compiled wg_M.
MOE_8WAVE_ADAPTIVE_WG_M=0 python tests/contrib/moe/test_fused_moe.py ...

# Enable sorted_ids[128]-based dispatch to the generated wg_M=128 body.
MOE_8WAVE_ADAPTIVE_WG_M=1 python tests/contrib/moe/test_fused_moe.py ...
```

Only `0` and `1` are accepted. The value is passed as the compile-time
`adaptive_wg_m` argument, so enabled and disabled kernels have separate JIT
cache keys.

## Compile-time dimensions

```text
HALF_BLOCK_SIZE_ROW = 128
HALF_BLOCK_SIZE_COL = 128
nrM = (wg_M / 16) / WARPS_ROW / 2 = 4
nrN = (wg_N / 16) / WARPS_COL / 2 = 2
loop_cnt = IC / 128 = 48                 # IC=6144
```

## Activation scale layout

FP8 activations use `QuantType.per_1x128`: one scale per row and K-block of
128 values. The gate-up kernel keeps two 8192-byte LDS buffers, each covering
eight K stages. Weight scales remain 128x128. The specialized
`moe_gemm_down_tp` kernel is unchanged and consumes the transposed stage-2
scale layout.

## LDS usage

The gfx950 LDS capacity is 163840 bytes per CU. The default 1x128,
`wg_M=256` path allocates:

| Allocation | Bytes |
|---|---:|
| Sorted IDs | 1024 |
| Sorted weights | 1024 |
| Four A matrix tiles | 65536 |
| Four B matrix tiles | 65536 |
| Two eight-stage scaleA buffers | 16384 |
| All scaleB values | 384 |
| **Raw total** | **149888** |

Rocprof reports 150016 bytes after hardware/compiler allocation rounding.

The eight-stage path emits six scaleA VM instructions per wave for 48 K stages.

### Full block-scale cache limit

One logical scaleA stage for `wg_M=256` occupies `256 * 4 = 1024` bytes.
Preloading all 48 stages would require 49152 bytes and raise the raw total to
182656 bytes, which does not fit. The retained implementation therefore uses
two rolling eight-stage buffers totaling 16384 bytes.

An experiment used 12 stages per chunk:

```text
12 * 2048 B scaleA + 133504 B other LDS = 158080 B
```

This left 5760 bytes of raw headroom. The experiment processed 48 K blocks as
four 12-block chunks, batching scale VM loads before each chunk so that
`loop_body` contained only LDS-to-register scale reads. It was reverted after
profiling showed a 10.86% regression; the retained implementation uses the
rolling eight-stage cache described above.

One `mfma(c_index)` emits:

```text
8  v_mfma_f32_16x16x128_f8f6f4          # nrM * nrN
32 v_fmac_f32                            # nrM * nrN * 4 accumulators
4  v_mul_f32                             # scaleA * scaleB, once per m
```

`yield 16` is a `J.emit` scheduling hint and does not emit an instruction.

## Buffer progress

At the entry of `loop_body(k)`:

```text
tic LDS: A[k,row0/row1], B[k,gate/up]
toc LDS: A[k+1,row0],    B[k+1,gate/up]
scale LDS: current eight-stage scaleA group; the other buffer holds the next group
registers: B[k,gate] was preloaded by the previous iteration
           (or by the prologue for k=0)
VM offsets: point to k+1 before Region 0
```

The four accumulator indices represent:

```text
c0 = A[k,row0] * B[k,gate]
c1 = A[k,row0] * B[k,up]
c2 = A[k,row1] * B[k,gate]
c3 = A[k,row1] * B[k,up]
```

## Balanced steady-state pseudocode

```python
# Prologue: scaleA[0] is the oldest of nine VM loads issued by each wave.
# vmcnt(4) retires it; the k=0 barrier preserves the staggered wave-group phase.
wait_first_batch_vmcnt4(); staggered_barrier()
Breg[gate] = lds_read_B(tic, gate)             # 4 ds_read_b128, B[k=0,gate]

def loop_body(k):
    # Region 0 --------------------------------------------------------------
    # Consume k/row0/gate. Finish the k+1 A tile while MFMA works on k.
    Areg[row0] = lds_read_A(tic, row0)          # 8 ds_read_b128, A[k,row0]
    scaleA[row0] = lds_read_scaleA(tic, row0)   # 4 ds_read_b32, scaleA[k,row0]
    scaleB[:] = lds_read_scaleB(k)              # 2 ds_read_b32, gate/up
    vm_to_lds_A(toc, row1)                      # 2 buffer_load_dwordx4, A[k+1,row1]
    advance_vm_offsets_to(k + 2)
    wait_lgkm(); barrier()
    mfma_fifo_step(c0)                          # 8 MFMA + 32 FMAC + 4 MUL
    barrier()

    # Region 1 --------------------------------------------------------------
    # Consume k/row0/up. Reuse the old tic A-row0 storage for k+2.
    Breg[up] = lds_read_B(tic, up)               # 4 ds_read_b128, B[k,up]
    vm_to_lds_A(tic, row0)                       # 2 buffer_load_dwordx4, A[k+2,row0]
    barrier(); wait_lgkm()
    mfma_fifo_step(c1)                           # 8 MFMA + 32 FMAC + 4 MUL
    barrier()

    # Region 2 --------------------------------------------------------------
    # Consume k/row1/gate. Reuse old tic B-gate storage for k+2.
    Areg[row1] = lds_read_A(tic, row1)           # 8 ds_read_b128, A[k,row1]
    scaleA[row1] = lds_read_scaleA(tic, row1)    # 4 ds_read_b32, scaleA[k,row1]
    vm_to_lds_B(tic, gate)                       # 2 buffer_load_dwordx4, B[k+2,gate]
    barrier(); wait_lgkm()
    mfma_fifo_step(c2)                           # 8 MFMA + 32 FMAC + 4 MUL
    barrier()

    # Region 3 --------------------------------------------------------------
    # Finish the next tile before reading it. MFMA c3 reads Breg[up], so the
    # independent Breg[gate] bank can receive the next iteration's operand.
    vm_to_lds_B(tic, up)                          # 2 buffer_load_dwordx4, B[k+2,up]
    if k % 8 == 7 and k + 9 < loop_cnt:
        vm_to_lds_scaleA(old_scale_buffer, k + 9) # 1 dwordx4, eight K stages
    wait_vmcnt_threshold(); barrier()
    Breg[gate] = lds_read_B(toc, gate)            # 4 ds_read_b128, B[k+1,gate]
    mfma_fifo_step(c3)                            # 8 MFMA + 32 FMAC + 4 MUL
    if scaleA_prefetched:
        wait_scale_vmcnt0()                       # direct-to-LDS completion
    barrier()

    swap(tic, toc)                                # next loop computes k+1
```

The Region 3 LDS read must occur after the VM wait and barrier: `ldsB[toc,gate]`
is part of the tile whose global-to-LDS loads are completed by that wait. It is
then safe to write `mfma_B[0]` because `mfma(3)` reads `mfma_B[1]`. Completion
of the DS read is covered by the Region 0 `lgkmcnt(0)` wait in the next
iteration.

## Instruction counts per loop body

| Category | ISA | Instructions/wave |
|---|---|---:|
| LDS to A registers | `ds_read_b128` | 16 |
| LDS to B registers | `ds_read_b128` | 8 |
| LDS to scaleA registers | `ds_read_b32` | 8 |
| LDS to scaleB registers | `ds_read_b32` | 2 |
| **All LDS reads** | | **34** |
| MFMA | `v_mfma_f32_16x16x128_f8f6f4` | 32 |
| Accumulation | `v_fmac_f32` | 128 |
| Scale products | `v_mul_f32` | 16 |
| **FMAC including scale MUL** | | **144** |
| VM to LDS A | `buffer_load_dwordx4` | 4 |
| VM to LDS B | `buffer_load_dwordx4` | 4 |
| VM to LDS scaleA | `buffer_load_dwordx4` | 1 every eight loops |
| VM to LDS scaleB | preloaded before K loop | 0 |
| Barrier | `s_barrier` | 8 |
| LDS wait | `s_waitcnt lgkmcnt(...)` | 4 |
| VM wait | `s_waitcnt vmcnt(...)` | 1 |

## Forty-eight steady-state iterations

| Category | Instructions/wave |
|---|---:|
| LDS to A registers | 768 |
| LDS to B registers | 384 |
| LDS to scaleA registers | 384 |
| LDS to scaleB registers | 96 |
| MFMA | 1536 |
| FMAC | 6144 |
| Scale MUL | 768 |
| FMAC including scale MUL | 6912 |
| VM load A | 192 |
| VM load B | 192 |
| VM load scaleA inside loop bodies | 4 |
| VM load scaleB | 0 |
| Barrier | 384 |

The prologue adds one `ds_read_B` group (4 instructions) to seed `k=0`. The
last Region 3 also preloads an unused `B[k=48,gate]`; retaining the uniform
steady-state schedule costs four LDS instructions once per kernel path.

The prologue emits two scaleA loads and the loop emits four refills, for six
`buffer_load_dwordx4 ... lds` instructions per wave across all 48 K stages.
Before the first prologue wait, each wave issues nine VM loads in this order:
one scaleA load, two B-gate loads, two A-row0 loads, two B-up loads, and two
A-row1 loads. `vmcnt(4)` therefore retires the oldest five, including
`scaleA[0]`; an additional `vmcnt(0)` is not required before its first LDS
read.

For the adaptive `wg_M=128` body, only the leading four waves issue scaleA
loads. A producer wave issues seven requests and waits at `vmcnt(3)`, retiring
the oldest four including scaleA. The lagging four waves issue no scaleA load;
they acquire its LDS contents through the following barrier generation.

The conditional barrier executed by `warp_m=1` staggers the two four-wave
groups. At the second prologue wait, the four requests left by the first batch
are followed by seven requests from the second batch. `vmcnt(7)` retires only
the old requests; all of `scaleA[1]`, `A[toc,row0]`, and the four B requests may
still be pending.

The k=0 barrier is not required by the scale index: k=0 reads `scaleA[0]`, and
`scaleA[1]` is first read at k=8. Its role in the retained schedule is to pace
the leading four-wave group while the lagging group issues the second prefetch
batch. Controlled waits locate the sensitive boundary at `A[toc,row0]`:
`vmcnt(6)` (scaleA only) and `vmcnt(5)` (half of A-row0) remain unstable, while
`vmcnt(4)` (all of A-row0) is stable. An `s_barrier` is only an execution
barrier on gfx950 and does not formally guarantee DMA completion, so this is a
performance-oriented scheduling dependency rather than a scaleA memory fence.
Each later scaleA refill is newer than the matrix loads in Region 3, so its
partial wait intentionally leaves it pending; that refill still requires the
post-`mfma(3)` `vmcnt(0)` before the existing end-of-iteration barrier.

The MFMA FIFO tail adds 32 `v_fmac_f32` instructions after the K loop.

## Validation and performance

Correctness command for the profiled FP8 gate-up path:

```bash
python tests/contrib/moe/test_fused_moe.py \
    -dim 6144,256 -t 16384 -a silu -s f -e 384 -k 8 -p t -q 5 -j
```

Result after balancing the LDS reads:

```text
logits_diff = 4.3373e-06
```

Additional coverage:

```text
BF16, preshuffle=true:  logits_diff = 3.45017e-06
FP8,  preshuffle=false: logits_diff = 4.3373e-06
```

The latter paths verify that the compile-time guard does not change the
`wg_M=128` body.

Rocprof command used for both measurements:

```bash
rocprofv3 --stats --kernel-trace -f csv \
    -d <output-directory> -o <output-prefix> -- \
    python tests/contrib/moe/test_fused_moe.py \
        -dim 6144,256 -t 16384 -a silu -s f -e 384 -k 8 -p t -q 5 -j
```

Seven `moe_gemm_8wave_g1u1` dispatches were measured:

| Schedule | Mean (us) | Median (us) | Min (us) | Max (us) | Stddev (us) |
|---|---:|---:|---:|---:|---:|
| Original | 541.068 | 543.685 | 528.485 | 560.365 | 10.646 |
| Balanced LDS reads | 540.788 | 540.325 | 533.685 | 549.605 | 5.944 |
| 1x128 row-major four-stage dwordx4 scaleA | 519.645 | 504.885 | 502.805 | 585.166 | 29.738 |
| 1x128 row-major eight-stage dwordx4 scaleA | 519.125 | 511.486 | 502.885 | 569.686 | 21.361 |
| 12-stage scaleA chunks (rejected) | 599.497 | 601.165 | 583.925 | 618.445 | 12.588 |

The mean change is `-0.280 us` (`-0.05%`) and the median change is `-3.360 us`
(`-0.62%`). This is not a statistically convincing speedup; treat it as
performance-neutral until a larger repeated benchmark shows otherwise. Kernel
resources are unchanged: 128 reported VGPRs, 112 SGPRs, and 137728 bytes LDS
per workgroup.

### Eight-stage dwordx4 scaleA result

The default 1x128 gate-up path stores its stage-1 scales row-major as
`[token, K/128]`. Four adjacent K-block scales for one row are therefore loaded
with one `buffer_load_dwordx4 ... lds`. All 512 threads participate: each
thread loads four FP32 values, so one workgroup load transfers 2048 scales,
covering eight K stages for 256 rows. Two 8192-byte LDS buffers ping-pong, and
the next group is loaded after every eighth `loop_body` call.

A hardware probe verified that `buffer_load_dwordx4 ... lds` writes
`[wave][lane][four components]`, with the four components contiguous in LDS.
The consumer uses this measured lane-major mapping. The generated `IC=6144`
kernel contains six scaleA `buffer_load_dwordx4` instructions for 48 K stages,
replacing 12 four-stage loads or 48 scalar stage loads. This layout is limited
to gate-up. Stage 2 retains the transposed scale layout and continues to use
the unchanged `moe_gemm_down_tp` kernel.

Direct-to-LDS VM loads require a wait that retires the specific producer and a
workgroup barrier before cross-wave consumption; the wait need not always be
`vmcnt(0)`. In the prologue, the earlier `vmcnt(4)` retires `scaleA[0]`. Later
refills are the newest VM requests and still use `vmcnt(0)`. Removing both the
prologue wait and its barrier caused divergent outputs because it also removed
the required barrier generation; removing only the redundant wait preserves
bitwise-identical results across seven repeated gate-up runs.

### Controlled scalar versus dwordx4 rocprof comparison

On 2026-07-23, scaleA load width was measured with ordinary rocprof kernel
tracing. This experiment did **not** enable ATT, advanced thread trace, PMC
counters, or an ATT input YAML. Each run used:

```bash
HIP_VISIBLE_DEVICES=0 \
PYHIP_CACHE_DIR=<precompiled-variant-cache> \
rocprofv3 --stats --kernel-trace -f csv \
    -d <output-directory> -o trace -- \
    python tests/contrib/moe/test_fused_moe.py \
        -dim 6144,256 -t 16384 -a silu -s f \
        -e 384 -k 8 -p t -q 5 -j
```

Both variants retained the same two eight-stage scaleA buffers, 2048 scales
transferred per prefetch batch, prefetch cadence, LDS capacity, and scale
consumption points. The only intended difference was the VMEM load width:

- scalar: four `buffer_load_dword ... lds` instructions per scaleA batch;
- dwordx4: one `buffer_load_dwordx4 ... lds` instruction per scaleA batch.

The scalar variant used an equivalent stage-major LDS layout and changed the
scaleA contribution to `vmcnt` thresholds from one request to four. Generated
assembly was checked to contain four scalar scale loads per batch and no
compiler-recombined dwordx4 load. The dwordx4 kernel contained six scaleA
dwordx4 loads per wave for all 48 K stages.

Both variants were compiled before profiling into separate JIT cache
directories, so compilation time was excluded. Four process-level rounds were
run in the interleaved order `D-S / S-D / D-S / S-D` to reduce temperature and
clock-drift bias. Each process produced seven profiled gate-up dispatches (two
warmups and five measured calls in the test wrapper), giving 28
`moe_gemm_8wave_g1u1` samples per variant. Kernel duration was calculated from
`trace_kernel_trace.csv` as:

```text
(End_Timestamp - Start_Timestamp) / 1000  # microseconds
```

Per-round results:

| Round | Variant | Samples | Mean (us) | Median (us) | Min (us) | Max (us) | Stddev (us) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | dwordx4 | 7 | 498.291 | 489.484 | 484.805 | 520.965 | 14.642 |
| 1 | scalar | 7 | 519.273 | 512.924 | 505.365 | 534.605 | 11.685 |
| 2 | dwordx4 | 7 | 496.759 | 492.124 | 486.644 | 511.365 | 9.614 |
| 2 | scalar | 7 | 522.936 | 513.405 | 507.965 | 551.685 | 18.131 |
| 3 | dwordx4 | 7 | 498.142 | 488.924 | 488.445 | 543.765 | 20.330 |
| 3 | scalar | 7 | 514.113 | 509.804 | 504.765 | 526.285 | 9.017 |
| 4 | dwordx4 | 7 | 491.518 | 488.164 | 485.404 | 511.245 | 8.892 |
| 4 | scalar | 7 | 518.787 | 509.564 | 507.164 | 551.645 | 17.278 |

Combined results:

| Variant | Samples | Mean (us) | Median (us) | Min (us) | Max (us) | Stddev (us) | P25 (us) | P75 (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar | 28 | 518.778 | 512.185 | 504.765 | 551.685 | 14.071 | 508.544 | 526.945 |
| dwordx4 | 28 | 496.177 | 489.304 | 484.805 | 543.765 | 13.619 | 487.924 | 503.985 |

The dwordx4 variant reduced median kernel time by `22.881 us` (`4.47%`) and
mean time by `22.600 us` (`4.36%`). The paired per-round median advantages were
`23.440`, `21.281`, `20.880`, and `21.400 us`, all in favor of dwordx4. A
20,000-resample bootstrap over dispatch samples gave a 95% confidence interval
of `[17.000, 34.101] us` for the median advantage.

Both variants produced `logits_diff=4.3373e-06` in all four profiling rounds.
Rocprof reported identical resources for both: 128 VGPRs, 112 SGPRs, 150016
bytes LDS, zero scratch, 512 threads/workgroup, and grid size 917504. Therefore
the measured difference is not explained by occupancy or resource allocation;
the current dwordx4 scaleA load should be retained.

Raw `trace_kernel_trace.csv`, rocprof stats, run logs, and `summary.csv` from
this experiment are stored under:

```text
/tmp/pyhip-scale-load-compare-20260723/
```

The temporary scalar implementation and profiling switch were removed after
collection; the repository source remains dwordx4-only.

### Historical scaleA comparisons

An earlier single-process, seven-dispatch comparison measured scalar,
four-stage dwordx4, and eight-stage dwordx4 implementations that also differed
in scaleA layout or prefetch cadence:

```text
scalar scaleA:          530.616 us mean, 527.045 us median
four-stage dwordx4:     519.645 us mean, 504.885 us median
eight-stage dwordx4:    519.125 us mean, 511.486 us median
eight vs scalar:        -11.491 us, -2.17% mean
eight vs four-stage:     -0.520 us, -0.10% mean
```

Correctness coverage:

```text
eight-stage, wg_M=256:      logits_diff = 4.33730e-06
eight-stage, adaptive mode: logits_diff = 4.33730e-06
seven repeated gate outputs: bitwise identical
```

The row-major and transposed quantizer outputs were verified element-for-
element: quantized activations and scale values are identical after reshaping;
only scale storage order differs.

### Rejected scaleA chunk experiment

The 12-stage variant was numerically correct (`logits_diff=4.3373e-06`) and
compiled to 253 VGPRs with 158080 raw LDS bytes. Rocprof reported 158208 bytes
after allocation rounding, so occupancy remained 2 waves/SIMD. Its mean kernel
time increased from 540.788 us to 599.497 us (`+58.709 us`, `+10.86%`).

The concentrated scale loads require a full `vmcnt(0)` and barrier at every
chunk boundary, exposing their VM latency instead of hiding one load inside
each loop iteration. Future work should retain overlap, for example by
prefetching the next chunk into a second compact chunk buffer while computing
the current chunk; simply moving all scale loads before a chunk is a negative
optimization.