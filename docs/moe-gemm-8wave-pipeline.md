# MoE GEMM 8-wave pipeline

This note describes the `wg_M=256`, `wg_N=256`, FP8 block-scale, gate-up
instance of `moe_gemm_8wave_g1u1`. Counts are ISA instructions issued by one
wave, not tensor elements. A workgroup contains eight waves.

The balanced cross-iteration B-register schedule is enabled only for this
`wg_M=256 && gate_up` layout. The `wg_M=128` and ordinary down variants retain
the original schedule because their operand/register progression differs.

## Compile-time dimensions

```text
HALF_BLOCK_SIZE_ROW = 128
HALF_BLOCK_SIZE_COL = 128
nrM = (wg_M / 16) / WARPS_ROW / 2 = 4
nrN = (wg_N / 16) / WARPS_COL / 2 = 2
loop_cnt = IC / 128 = 48                 # IC=6144
```

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
tic LDS: A[k,row0/row1], B[k,gate/up], scaleA[k]
toc LDS: A[k+1,row0],    B[k+1,gate/up], scaleA[k+1]
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
    vm_to_lds_scaleA(tic, k + 2)                  # 1 buffer_load_dword
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
| VM to LDS scaleA | `buffer_load_dword` | 1 |
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
| VM load scaleA | 48 |
| VM load scaleB | 0 |
| Barrier | 384 |

The prologue adds one `ds_read_B` group (4 instructions) to seed `k=0`. The
last Region 3 also preloads an unused `B[k=48,gate]`; retaining the uniform
steady-state schedule costs four LDS instructions once per kernel path.

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

The mean change is `-0.280 us` (`-0.05%`) and the median change is `-3.360 us`
(`-0.62%`). This is not a statistically convincing speedup; treat it as
performance-neutral until a larger repeated benchmark shows otherwise. Kernel
resources are unchanged: 128 reported VGPRs, 112 SGPRs, and 137728 bytes LDS
per workgroup.