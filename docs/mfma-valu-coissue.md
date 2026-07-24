# MFMA and VALU co-issue on gfx950

## Result

`v_pk_fma_f32` cannot issue while `v_mfma_f32_16x16x128_f8f6f4` is busy on
the tested gfx950 system. Two scalar `v_fmac_f32` instructions can issue in the
same MFMA shadow. This explains the measured packed-FMA regression in
`moe_gemm_8wave_g1u1` even though the packed version has fewer static
instructions.

The production comparison used physical GPU 6, ROCm 7.2.3, rocprofv3 1.1.0,
and the workload:

```text
-dim 6144,256 -t 16384 -a silu -s f -e 384 -k 8 -p t -q 5 -j
```

Both variants were built from `5f6589e`; only the scalar FIFO accumulation was
changed. Every run produced `logits_diff=4.3373e-06`.

| Metric | Scalar `v_fmac_f32` | Packed `v_pk_fma_f32` |
|---|---:|---:|
| Dispatch samples | 21 | 21 |
| Kernel mean | 514.422 us | 586.825 us |
| Kernel median | 509.645 us | 579.166 us |
| `SQ_INSTS_MFMA` | 18,874,368 | 18,874,368 |
| MFMA busy cycles / MFMA | 32.000 | 32.000 |
| Co-exec cycles / MFMA | 16.920 | 2.465 |
| Co-exec / MFMA busy | 52.875% | 7.703% |

Packed FMA reduces MFMA-shadow coverage by 45.172 percentage points, or 85.4%
relative, while the median kernel time regresses by 69.521 us (13.64%). The
MFMA count and MFMA busy time are unchanged, so this is not a change in matrix
work or matrix-unit latency.

## Counter meaning

The installed ROCm counter database defines:

| Counter | gfx950 SQ event | Scope | Meaning |
|---|---:|---|---|
| `SQ_VALU_MFMA_COEXEC_CYCLES` | 94 | per-SIMD, nondeterministic | Cycles where MFMA VALU is busy and a normal VALU instruction is issued |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 93 | per-SIMD | Cycles where the MFMA ALU is busy |
| `SQ_INSTS_MFMA` | 58 | per-SE aggregate | MFMA instructions issued |
| `SQ_INSTS_VALU` | 26 | per-SE aggregate | VALU instructions issued |
| `SQ_CYCLES` | 2 | per-SIMD | SQ clocks |

The useful ratios are:

$$
\text{MFMA shadow coverage} =
\frac{\text{SQ\_VALU\_MFMA\_COEXEC\_CYCLES}}
     {\text{SQ\_VALU\_MFMA\_BUSY\_CYCLES}}
$$

$$
\text{co-exec cycles per MFMA} =
\frac{\text{SQ\_VALU\_MFMA\_COEXEC\_CYCLES}}
     {\text{SQ\_INSTS\_MFMA}}
$$

The first ratio is the safest comparison because numerator and denominator have
the same per-SIMD scope. The counter is explicitly nondeterministic, so compare
multiple dispatches and report distributions rather than treating one value as
an architectural constant.

`SQ_VALU_MFMA_COEXEC_CYCLES` is an overlap counter, not an instruction latency
counter. It cannot by itself identify which VALU instruction co-issued or how
long a blocked instruction waited. The microbenchmark below fixes the instruction
stream and scans the MFMA-to-VALU distance to recover that boundary.

## Microbenchmark

[mfma-coissue.py](../archive/gemm/mfma-coissue.py) emits four independent
`v_mfma_f32_16x16x128_f8f6f4` instructions per loop iteration. Each MFMA is
followed by one of:

- no VALU operation;
- one `v_add_f32`;
- one `v_fmac_f32`;
- two independent `v_fmac_f32` operations;
- one `v_pk_fma_f32 op_sel_hi:[1,0,1]`.

The two-FMAC case performs the same two FP32 updates as one packed FMA. The
generated loop was checked to contain exactly four MFMA plus either eight
scalar FMAC or four packed FMA instructions. Both variants use zero scratch and
reach eight waves/SIMD.

With no inserted wait states:

| VALU sequence after each MFMA | Co-exec cycles / MFMA | Co-exec / busy | Device cycles / MFMA |
|---|---:|---:|---:|
| One `v_add_f32` | 4.000 | 12.500% | 285.017 |
| One `v_fmac_f32` | 4.000 | 12.500% | 285.017 |
| Two `v_fmac_f32` | 7.999 | 24.998% | 285.022 |
| One `v_pk_fma_f32` | 0.000 | 0.000% | 317.018 |

The fixed-period, equivalent-work distance scan was:

| `s_nop` wait states after MFMA | Packed minus two-FMAC cycles / MFMA | Two-FMAC co-exec / MFMA | Packed co-exec / MFMA |
|---:|---:|---:|---:|
| 0 | +31.996 | 7.999 | 0.000 |
| 2 | +32.000 | 7.999 | 0.000 |
| 4 | +24.000 | 7.999 | 0.000 |
| 6 | +16.000 | 4.000 | 0.000 |
| 8 | +8.000 | 0.000 | 0.000 |
| 9 | +4.000 | 0.000 | 0.000 |
| 10 | +0.000 | 0.000 | 0.000 |
| 12 | +0.000 | 0.000 | 0.000 |

On this one-wave-per-CU probe, one wait state corresponds to approximately four
device clocks because the CU scheduler visits four SIMDs round-robin. The packed
instruction is therefore delayed until the 32-clock MFMA busy window has passed;
the final issue boundary appears between 9 and 10 wait states after accounting
for issue alignment. This is an issue-blocking penalty, not the data dependency
latency of `v_pk_fma_f32`.

Run the complete probe with:

```bash
GPU_ID=6 archive/gemm/profile-mfma-coissue.sh
```

The script refuses to run if the selected GPU is not idle. It collects the five
PMC counters in one rocprof pass for every control and gap, then invokes
[analyze-mfma-coissue.py](../archive/gemm/analyze-mfma-coissue.py).

## Compiler connection

LLVM independently encodes the same restriction. `SIPreEmitPeephole.cpp` says
it unpacks `V_PK_MUL_F32`, `V_PK_ADD_F32`, and `V_PK_FMA_F32` adjacent to MFMAs
"such that they can be co-issued." It maps `V_PK_FMA_F32` to two
`V_FMA_F32_e64` instructions and only transforms packed instructions inside the
MFMA schedule-model window. The pass also checks register overlap, clobbering,
terminators, mode/EXEC changes, and never-coissue instructions.

That pass cannot repair this PyHIP kernel: PyHIP has already serialized the
instructions into inline assembly before HIP compilation, and inline assembly
is opaque to LLVM machine-level instruction transforms. The packed instruction
therefore reaches the final ISA unchanged. The practical options are:

1. Keep scalar FMAC in MFMA-shadow regions. This is the best choice for the
   current kernel and is supported by both production PMC and wall time.
2. Split packed FP32 operations into scalar FMA before final assembly, matching
   LLVM's pre-emit transform.
3. Move packed operations beyond the MFMA busy window. This removes the direct
   block but usually loses the overlap the pipeline was designed to exploit.
4. Use packed FMA only in tails or regions without in-flight MFMA work.

## Related material

- [ROCm profiler counter definitions](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/counter_defs.yaml): gfx950 events 93 and 94.
- [ROCm Compute Profiler pipeline descriptions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html#matrix-fused-multiply-add-mfma): MFMA shadow execution and scheduler model.
- [ROCm Compute Profiler IPC and utilization tutorial](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/tutorial/instructions-per-cycle-and-utilizations.html): example using MFMA execution cycles and possible VALU co-execution cycles.
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator): defines `coexec` and `coexec_delay`; its public release currently covers CDNA1-CDNA3, not this CDNA4 instruction.
- [LLVM `SIPreEmitPeephole.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIPreEmitPeephole.cpp): packed-FP32-to-scalar transform for MFMA co-issue.
- [LLVM gfx950 schedule model](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SISchedule.td): MFMA resource-cycle model.
- [LLVM gfx950 llvm-mca test](https://github.com/llvm/llvm-project/blob/main/llvm/test/tools/llvm-mca/AMDGPU/gfx950.s): `v_mfma_f32_16x16x128_f8f6f4` uses four modeled MFMA resource cycles.
- [CK packed-math warning](https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp): documents a gfx950 regression from un-coexecutable packed math.
- [MI355X MFMA ILP study](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html): uses the same `16x16x128` MFMA and separates ILP from occupancy.

Raw data from the 2026-07-24 experiment is stored in:

```text
/tmp/pyhip-mfma-coissue-controls-20260724/
/tmp/pyhip-mfma-coissue-gapscan-20260724/
/tmp/pyhip-mfma-coissue-equal-work-20260724/
/tmp/pyhip-moe-coissue-pmc-20260724/
```