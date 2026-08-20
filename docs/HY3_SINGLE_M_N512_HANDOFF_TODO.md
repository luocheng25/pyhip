# Hy3 single-M N512 handoff TODO

> Recovery branch: `luocheng/hy3-single-n512-handoff`
>
> Code checkpoint: `5161224` (`optimize Hy3 down with single-M N512`)
>
> Parent checkpoint: `2b50372` (`defer H3 PTPC scale loads to epilogue`)
>
> Kernel source SHA256: `352c7b0235f4937452ce16275ab4b1da7c7b28fbcea588b90b532afffd55a8f6`
>
> Target: MI308X / gfx942, ROCm 7.2.3, FlyDSL 0.2.4.

This file is the self-contained restart point for the Hy3 down optimization. Read it before changing the kernel. The broader H3 paired history remains in `docs/UNIFIED8_PA_INTERMEDIATE_CONTEXT.md`; current all-down boundaries remain in `docs/UNIFIED8_DOWN_TODO.md` and `tests/contrib/moe/README.md`.

## 1. Goal and current result

Production Hy3 shape:

```text
B=32768, TOPK=9, E=193, N=4096, K=192
FP8 weight/activation, per-tensor weight and activation scales
BM=64, row-major BF16 down output, sorted_sum consumer
```

The previous automatic winner was physical N256 with four waves and 0B row padding. The attempted paired M128 N512 path was about 3% slower than N256 because it duplicated two M64 groups inside one 512-thread workgroup and paid ten workgroup barriers.

The retained solution is a different algorithm: **single-M N512**.

- Keep M64 sorting and the original M64 task count.
- One 512-thread workgroup handles one M64 task.
- Eight waves span N512, with each wave owning N64.
- Do not duplicate expert metadata or pair two M64 tasks.
- Write ordinary row-major output through the existing wave-private XOR CShuffle.
- Keep 0B row padding.

Clean 10-buffer ABBA24 on GPU4, 1800MHz, PTL `VECTOR,F8`:

| Phase | Physical N256 -> single-M N512 | Candidate/control ratio | IQR | Wins |
| --- | ---: | ---: | ---: | ---: |
| down | `1.673130 -> 1.513829 ms` | `0.906541` | `0.903067--0.910706` | 24/24 |
| down + sorted_sum | `2.511415 -> 2.347834 ms` | `0.935520` | `0.931968--0.939437` | 24/24 |

This is a 9.35% down latency reduction and 6.45% combined reduction relative to N256.

Independent clean retest using the tracked formal harness on the same source:

| Phase | Physical N256 -> single-M N512 | Candidate/control ratio | IQR | Wins |
| --- | ---: | ---: | ---: | ---: |
| down | `1.557529 -> 1.417749 ms` | `0.909806` | `0.908210--0.912926` | 24/24 |
| sorted_sum | `0.712085 -> 0.711784 ms` | `1.000015` | `0.998096--1.001698` | 12/24 |
| down + sorted_sum | `2.283913 -> 2.141253 ms` | `0.937691` | `0.934980--0.940317` | 24/24 |

The second clean run confirms that the gain comes from down; the row-major consumer is neutral. Its output JSON is `/tmp/hy3_single_idle_retest.json` on the source node with SHA256 `de86615dba85f616e87facc4d08a2e30fe0238011ab8e1e490feaaf93997e651`. The harness checked ten buffers, all valid physical and reduced outputs bitwise, untouched inactive tails, idle initial state, managed `VECTOR,F8` state, and restored `auto`/`F16,BF16` state.

Two later retests ran while an external all-GPU job held every card at 100% busy and 83% VRAM. They are stress evidence only, not replacements for the clean result:

| Retest | Down ratio / wins | Combined ratio / wins |
| --- | --- | --- |
| stress ABBA24 #1 | `0.945073`, 19/24 | `0.936736`, 22/24 |
| stress ABBA24 #2 | `0.971739`, 16/24, IQR crosses 1 | `0.945147`, 22/24 |

All comparisons use same-process paired ratios. Do not compare absolute medians across separate windows.

## 2. Implementation details

Primary code: `src/contrib/flydsl/moe_gemm_splitk.py`.

New compile option:

```python
down_single_m_n512=True
```

It is valid only with:

```python
down_physical_n512=True
down_paired_row_major=True
N=4096
K=192
TOPK=9
E=193
weight_quant_type="per_tensor"
act_quant_type="per_tensor"
BLOCK_TILE_SIZE_M=64
BLOCK_TILE_SIZE_N=512
```

Important ownership variables:

- `use_paired_m128`: existing two-M64 M128 path; false for single-M.
- `paired_m_groups`: 1 for single-M, 2 for M128 paired.
- `pair_e_idx` and launch grid remain M64-sized for single-M.
- `BLOCK_N=512`, `WAVE_N=64`, tiled MMA wave shape `(8, 1, 1)`.
- Weight N layout uses the historical validated N512 permutation `(64, 2, 2, 2)` with strides `(1, 256, 64, 128)`.

Activation copy:

- A tile is `64 x 192 x FP8` = 768 128-bit atoms.
- All 512 threads copy one atom.
- Threads 0--255 copy the remaining 256 atoms.
- This replaces the historical 384-thread copy that left two waves idle.

Occupancy/resource fix:

- Single-M A LDS uses swizzle mode 3.
- Launch attaches `amdgpu-waves-per-eu=4,4` only for this exact path.
- Swizzle mode 4 under this constraint produced 6 spills / 28B private memory.
- Swizzle mode 3 produces zero spill and reaches the occupancy boundary.

Final resource/ISA gate:

```text
128 VGPR
32,768B LDS
0 private bytes
0 VGPR spills
0 scratch loads/stores
96 MFMA
2 s_barrier
18 buffer loads
8 buffer stores
ISA SHA256: 8e085db5ce25cff89dc106ff3800756c97618c02896d59ab27a878a4963d17d4
ISA lines/bytes: 703 / 24,116
```

Physical N256 and single-M N512 both use M64 sorting. Their padding ratio for production Hy3 is `1.0052083333333333`.

## 3. Dispatcher behavior

Selector: `_select_fly_down_single_m_layout()` in `tests/contrib/moe/test_moe.py`.

Automatic selection is exact-shape only. Disable it with:

```bash
MOE_DOWN_SINGLE_M_N512=0
```

`MOE_DOWN_PAIRED_N512=1` has precedence over automatic single-M. Forced paired M128 therefore still receives M128 sorting and duplicated expert metadata; single-M receives M64 sorting and unmodified metadata.

Production accuracy check already passed:

```text
fly_splitk_2s[B=64 weight_type=torch.float8_e4m3fnuz @ per_tensor]
acc OK(diff=0.00016577)
```

## 4. Correctness and regression evidence

Persistent test: `test_down_prefill_single_m_n512_hy3` in `tests/contrib/moe/test_flydsl_moe_down.py`.

It covers:

- two active experts;
- exact three-task grid (no `M * TOPK <= E` launch override);
- encoded TOPK=9 token/slot sorted IDs;
- physical N256 vs single-M valid and padded rows, bitwise equal;
- M64 zero-padding rows;
- inactive allocated tail remains sentinel;
- PyTorch reference relative-L2 threshold.

Validated commands/results:

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  /tmp/pyhip-flydsl024/bin/python -m pytest -q \
  tests/contrib/moe/test_flydsl_moe_down.py
# 62 passed, 1 existing allocator warning

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  /tmp/pyhip-flydsl024/bin/python -m pytest -q \
  tests/contrib/moe/test_moe.py \
  -k 'select_fly_down_padding_bytes or select_fly_down_paired_layout or select_fly_down_single_m_layout or select_fly_down_n512_paired_precedence'
# 21 passed
```

Fresh final ISA was byte-identical before/after for these non-target production kernels:

- exact H3 per-tensor paired N512;
- H3 PTPC paired N512;
- Xiaomi PTPC physical N256;
- Qwen3.5-397B PTPC physical N256.

## 5. Formal benchmark on a new node

Tracked harness: `tests/contrib/moe/compare_unified8_candidates.py`.

It now supports:

```text
--k 192
--control-path physical4
--candidate-path single_n512
```

The harness requires GPU idle (`busy <= 5%`, VRAM below its initial threshold), checks MI308X/gfx942/80 CUs/650W, sets deterministic/PTL state, uses ten buffers and ABBA ordering, verifies all valid physical and reduced outputs bitwise, checks inactive tails, and restores GPU state.

GPU state management uses the pinned `AMDSMI_ROOT` CLI when present and otherwise falls back to the system `amd-smi` on `PATH`. The fallback was verified with system AMDSMI 26.2.2 / ROCm 7.2.3. A live safety probe on a 100%-busy GPU rejected the run before changing state and confirmed `state_unchanged=true`.

Create a control source from the parent checkpoint, then run:

```bash
cd /host_root/pyhip

git show 2b50372:src/contrib/flydsl/moe_gemm_splitk.py \
  > /tmp/hy3_single_n512_control.py

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  /tmp/pyhip-flydsl024/bin/python \
  tests/contrib/moe/compare_unified8_candidates.py \
  --control /tmp/hy3_single_n512_control.py \
  --candidate src/contrib/flydsl/moe_gemm_splitk.py \
  --control-path physical4 \
  --candidate-path single_n512 \
  --k 192 \
  --rounds 24 \
  --physical-device 4 \
  --output /tmp/hy3_single_n512_clean_abba24.json
```

Reporting format is mandatory:

```text
Control ms -> Candidate ms
candidate/control median
ratio Q1--Q3
wins/rounds
```

Never claim a clean result if another workload is active. Never compare absolute medians from different benchmark processes when the paired ratio is available.

Current local benchmark log provenance (these `/tmp` files do not survive moving nodes):

```text
clean final:
  /tmp/hy3-single-final-clean-abba24.log
  SHA256 434bc4c08276a94db8f9ccc1a761a4abe2deb874e382978080ec214525f0252c
stress retest #1:
  /tmp/hy3-single-retest-contended-abba24.log
  SHA256 9353bf3949242b0e1f44270586ec762ed5d4a872d5efa377f00e02c7152a55d8
stress retest #2:
  /tmp/hy3-single-retest-2-contended-abba24.log
  SHA256 664d4fd07469372e179d4080320e287306b0e76bc8550425993cf80b07be5a47
```

## 6. Attempts that must not be repeated unchanged

These were correct or compilable but did not solve the N256 gap:

- Apply M128 K192 barrier merge directly: down regressed about 14%.
- Complete K128+K64 two-stage paired schedule: regressed about 5%.
- Balanced K128+K64 CShuffle split: large regression under stress screening.
- K96 cores: backend padded work to 128 MFMA instead of the required 96.
- BF16 accumulator carry across N: reduced VGPR 202 -> 168 but did not reliably improve combined time.
- Move paired CShuffle from 0/2/2 to 0/0/4: regressed for short K192 windows.
- Remove only one paired rendezvous while retaining local priorities: large regression.
- LDS weight sharing: safe implementation requires double buffering beyond 64KB; prior partial sharing also lost to added DS/wait cost.
- gfx942 split `s_barrier_signal` / `s_barrier_wait`: assembler reports unsupported.
- Reuse exact H3 XCC/SE map for K192: same-path effect was near neutral and insufficient.

The breakthrough was changing ownership from M128 pairing to single-M N512 and then crossing the 128-VGPR occupancy boundary without spills.

## 7. Remaining TODO

1. [x] Run the tracked formal benchmark independently and confirm reproducibility: `0.909806` down / `0.937691` combined versus the original `0.906541` / `0.935520`.
2. Run production Hy3 at B=32768 when enough VRAM is free; confirm accuracy and capture full-pipeline latency if required.
3. Optionally collect fresh ATT for physical N256 vs single-M N512. Verify the expected occupancy change and attribute remaining stalls; do not change code solely from single-wave counters.
4. Decide whether to merge this branch back into `luocheng/try-opt-down-308` after the new-node clean run.
5. Keep `tests/flydsl/flash_attn_api/` untouched; it is user-owned and intentionally absent from commits.

## 8. New-node recovery checklist

```bash
git fetch lc
git checkout luocheng/hy3-single-n512-handoff

git log -3 --oneline
git status --short
sha256sum src/contrib/flydsl/moe_gemm_splitk.py
```

Expected kernel SHA256:

```text
352c7b0235f4937452ce16275ab4b1da7c7b28fbcea588b90b532afffd55a8f6
```

Environment:

```text
FlyDSL Python: /tmp/pyhip-flydsl024/bin/python
PYTHONPATH=src
FLYDSL_RUNTIME_ENABLE_CACHE=0
Target GPU: MI308X gfx942, 80 CUs
Formal clock: 1800MHz determinism
Formal PTL: VECTOR,F8
Buffers: 10
Order: ABBA
Rounds: 24
```

Before timing:

```bash
rocm-smi --showuse --showmemuse --showpids -d 4
```

Stop if GPU busy is above 5% or another process owns substantial VRAM. The formal tracked harness will also reject a non-idle initial state.
