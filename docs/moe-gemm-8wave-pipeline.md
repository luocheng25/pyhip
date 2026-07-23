# MoE GEMM 8-wave pipeline

This note describes the `wg_M=256`, `wg_N=256`, FP8 block-scale, gate-up
instance of `moe_gemm_8wave_g1u1`. Counts are ISA instructions issued by one
wave, not tensor elements. A workgroup contains eight waves.

The balanced cross-iteration B-register schedule is enabled only for this
`wg_M=256 && gate_up` layout. The `wg_M=128` and ordinary down variants retain
the original schedule because their operand/register progression differs.

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

## Activation scale mode

`MOE_ACT_SCALE_PER_TOKEN` selects the activation quantization layout for FP8
MoE with 128x128 block-scaled weights:

```bash
# Default: one scale per 128 activation values
MOE_ACT_SCALE_PER_TOKEN=0 python tests/contrib/moe/test_fused_moe.py ...

# Optional: one scale per activation row
MOE_ACT_SCALE_PER_TOKEN=1 python tests/contrib/moe/test_fused_moe.py ...
```

Only `0` and `1` are accepted. The value is read when
`pyhip.contrib.fused_moe` is imported, so set it before starting Python. It is
passed to the JIT kernels as the compile-time `act_scale_per_token` argument;
the two layouts therefore use different binary-cache keys.

| Environment value | Quantizer | Activation scale layout | 8-wave scaleA cache |
|---:|---|---|---|
| `0` (default) | `QuantType.per_1x128` | one scale per row and K-block of 128 | two 2048 B rolling LDS stages |
| `1` | `QuantType.per_Token` | `a1[token,1]`, `a2[token,topk,1]` | one compact `wg_M * 4` B LDS cache |

Weight scales remain 128x128 in both modes.
The specialized `moe_gemm_down_tp` kernel is left unchanged and continues to
serve only the default 1x128 layout. Per-token mode routes stage 2 through
`moe_gemm_8wave_g1u1` instead.

## LDS usage

The gfx950 LDS capacity is 163840 bytes per CU. The original `wg_M=256` path
allocates:

| Allocation | Bytes |
|---|---:|
| Sorted IDs | 1024 |
| Sorted weights | 1024 |
| Four A matrix tiles | 65536 |
| Four B matrix tiles | 65536 |
| One per-row scaleA cache | 1024 |
| All scaleB values | 384 |
| **Raw total** | **134528** |

Rocprof reports 134656 bytes after hardware/compiler allocation rounding.

With `MOE_ACT_SCALE_PER_TOKEN=1`, the activation contract is one FP32 scale per
input row:

```text
stage 1: a1_scale[token, 1]
stage 2: a2_scale[token, topk, 1]
```

The weight contract remains 128x128 block scale. At kernel entry, the first
`wg_M` threads cooperatively gather one scale for every sorted row into a
`wg_M * 4` byte LDS buffer. Padding rows are redirected to scale row zero.
Every K iteration then reads the same row scale from LDS, so there are no
scaleA VM loads inside `loop_body`.

Compared with `MOE_ACT_SCALE_PER_TOKEN=0`, this saves 3072 LDS
bytes. The old pipeline emitted 50 scaleA VM instructions per wave: stages 0
and 1 in the prologue, then stages `k+2` for all 48 loop iterations (the final
two were harmless out-of-range buffer loads). The compact per-row cache emits
one load in each of the first four waves and none in waves 4-7: 400 dynamic
wave-instructions per workgroup become 4.

### Historical block-scale cache limit

Under the previous one-scale-per-128-values contract, one scaleA stage
physically reserved `8 waves * 64 lanes * 4 B = 2048 B`.
Preloading all 48 stages with the current layout would require 98304 bytes and
raise total LDS to 231808 bytes, which does not fit. Only the first 256 values
of each physical stage are consumed by `wg_M=256`; even a future compact
1024-byte layout would require 49152 bytes for all stages and a total of 182656
bytes, still above the limit.

An experiment used 12 stages per chunk:

```text
12 * 2048 B scaleA + 133504 B other LDS = 158080 B
```

This left 5760 bytes of raw headroom. The experiment processed 48 K blocks as
four 12-block chunks, batching scale VM loads before each chunk so that
`loop_body` contained only LDS-to-register scale reads. It was reverted after
profiling showed a 10.86% regression; the retained implementation uses the
per-row cache described above.

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
scale LDS: one scale for every sorted row, reused by all k
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
# Prologue: seed the operand carried across loop iterations.
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
    wait_vmcnt_threshold(); barrier()
    Breg[gate] = lds_read_B(toc, gate)            # 4 ds_read_b128, B[k+1,gate]
    mfma_fifo_step(c3)                            # 8 MFMA + 32 FMAC + 4 MUL
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
| VM to LDS scaleA | cached before K loop | 0 |
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
| VM load scaleA inside loop bodies | 0 |
| VM load scaleB | 0 |
| Barrier | 384 |

The prologue adds one `ds_read_B` group (4 instructions) to seed `k=0`. The
last Region 3 also preloads an unused `B[k=48,gate]`; retaining the uniform
steady-state schedule costs four LDS instructions once per kernel path.

Kernel entry emits one `buffer_load_dword ... lds` per wave to cache all row
scales. This replaces the previous 48 scaleA VM instructions across the K
loop.

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

The latter paths use the original schedule; they verify that the compile-time
guard does not change the `wg_M=128` or ordinary down variants.

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
| Per-row scaleA, duplicated 2 KiB cache | 493.199 | 495.645 | 476.444 | 506.645 | 10.922 |
| Per-row scaleA, compact 1 KiB cache | 486.187 | 487.124 | 477.844 | 494.525 | 7.088 |
| 12-stage scaleA chunks (rejected) | 599.497 | 601.165 | 583.925 | 618.445 | 12.588 |

The mean change is `-0.280 us` (`-0.05%`) and the median change is `-3.360 us`
(`-0.62%`). This is not a statistically convincing speedup; treat it as
performance-neutral until a larger repeated benchmark shows otherwise. Kernel
resources are unchanged: 128 reported VGPRs, 112 SGPRs, and 137728 bytes LDS
per workgroup.

### Per-row scaleA result

The final compact per-row implementation measured:

```text
gate-up 8-wave: 540.788 -> 486.187 us  (-10.10%)
gate LDS:        137728  -> 134656 B (rocprof rounded)
gate JIT VGPRs:  256     -> 252
```

The activation quantization producer changed from 1x128 group quantization to
per-token quantization. Its combined stage-1 plus stage-2 cost increased from
about 90.4 us to 133.8 us per fused-MoE invocation. End-to-end performance must
therefore include both that cost and the ordinary 8-wave stage-2 fallback; the
historical down-TP comparison is not applicable to the retained routing.

Correctness with both contracts:

```text
per-token, preshuffle=true:  logits_diff = 4.36141e-06
per-token, preshuffle=false: logits_diff = 4.36141e-06
per-token, wg_M=128:          logits_diff = 4.43035e-06  (token=4096)
1x128, wg_M=256:              logits_diff = 4.33730e-06  (token=16384)
1x128, wg_M=128:              logits_diff = 4.36227e-06  (token=4096)
```

The gate kernel itself benefits substantially, but end-to-end improvement now
depends on optimizing or fusing the per-row quantization producer.

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