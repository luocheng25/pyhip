# FlyDSL MoE down optimization TODO

> Packed-best promotion commit: `c82e2df`; current integrated kernel source SHA256:
> `929b36f77ebcbe282f01251f1105748cecdfe8f2911ffa78153875c4dafd184d`.
>
> Fixed H3 target: `1.895517 ms`; promoted result: `1.913011 ms`; gap: `17.494 us` (`0.9145%`).
>
> Do not promote a candidate without the path-specific correctness gate, resource/ISA gate, and controlled ABBA gate listed below.

## 2026-08-19 all-down checkpoint

- The packed H3 formal best is unchanged: its rebuilt final ISA is byte-identical to the promoted artifact. It still has 192 MFMA, 39 128-bit loads, 16 stores, 10 real barriers, 254 VGPR, 49,152B LDS, and zero scratch.
- The generic dispatcher cannot consume the packed layout directly. The direct row-major paired epilogue is mathematically correct, but realistic random-route ABBA8 regressed versus physical4 (`1.232053`, 0/8 wins) and versus packed paired (`1.363678`, 0/8 wins). Exact H3 therefore defaults to repaired physical4 with 0B row padding; `MOE_DOWN_PAIRED_N512=1` retains the row-major paired route as an explicit experiment only.
- The batch1 down path now allows VMEM reads to cross its compiler scheduling barriers. H3-dimension ABBA24 improved FP8 by `4.32%` (`0.956752`, 24/24 wins) and BF16 by `0.24%` (`0.997616`, 21/24 wins). BF16 and FP8 dispatcher accuracy gates pass.
- Applying the same mask to splitk is rejected: `0.872255 -> 1.626202 ms`, ratio `1.865054`, 0/8 wins. Splitk remains unchanged.
- The focused selector suite passes 14/14; the down suite now has persistent paired two-M64, repaired physical4 two-block, and batch1 BF16/FP8 atomic coverage.

## Path matrix

| Down path | Current implementation | Strict paired method applicability | Required gate |
| --- | --- | --- | --- |
| H3 fp8 per-tensor `prefill_1x4`, N4096/K384/TOPK9/E193 | packed paired formal best; generic dispatcher defaults to physical4/0B | Strict paired kernel remains the performance reference; row-major adapter is opt-in because it regresses | exact-grid physical/reduced/tail bitwise gate; 254 VGPR, 49,152B LDS, 10 barriers; ABBA8/24 |
| Generic fp8 per-tensor `prefill_1x4` | physical4 fallback | Direct row-major pairing passes correctness but fails performance; do not expose packed block-major stores to the generic consumer | N512/N1024 PyTorch oracle plus odd logical-block tail |
| fp8 PTPC `prefill_1x4` | physical4 | Pairing still needs scale-LDS ownership and cross-group synchronization proof; first group-local scale prototype crashed at launch and was reverted | PTPC per-token scale oracle for K192/K384/K512 and inactive tail |
| Non-physical `prefill_1x4` (bf16/fp8) | original 4-wave path | Reuse scheduling ideas only; physical pairing changes its output and LDS contract | existing prefill tests plus shape matrix |
| `splitk` down | one-wave compute with scatter/atomic output | Do not apply 4+4 anti-phase directly; only test load scheduling and mapping independently | atomic and non-atomic oracle, multi-expert routing |
| `batch1` down | one-wave compute with BF16 atomic reduction; VMEM-read crossing enabled | Pairing is structurally inapplicable; the compatible scheduling idea is promoted | batch1 TOPK reduction oracle |

## P0: correctness and contract

- [x] Compare H3 paired output against an independently correct physical4/PyTorch path, not only against a descendant of the same packed layout. Random unique-top-k H3 produced zero reduced-output mismatches; the real dispatcher reports `diff=0.00017182`.
- [x] Determine whether the H3 packed store is row-major or requires a matching consumer/decode step. It is block-major packed; the repaired `packed_direct` decoder is correct but too slow for production consumption.
- [ ] Add an exact-H3 regression that checks mathematical output and inactive tail independently of the candidate module.
- [x] Keep the generic dispatcher on physical4 unless explicitly opted into row-major pairing. N512/N1024 and real-H3 correctness pass, but the row-major performance gate fails.

## P1: generic per-tensor prefill pairing

- [x] Prototype a row-major direct-store adapter for the paired accumulator fragment without extra real barriers. It preserves 192 MFMA, 39 loads, 16 stores, 10 barriers, 254 VGPR, 49,152B LDS, and zero scratch after final-store deduplication.
- [x] Test transpose alternatives after direct stores proved expensive. A 64KB CShuffle version reached 256 VGPR with 27 spills/88B private memory; a wave-local bpermute version reached 256 VGPR with 4 spills/12B private memory. Both were reverted.
- [ ] Validate N512, N1024, N4096, K192/K384, one and two logical M64 blocks, and a non-multiple-of-128 valid tail.
- [x] Require zero scratch and no regression in the down suite before timing. The retained direct adapter has zero scratch; persistent paired and physical4 tests cover two M64 blocks.

## P2: PTPC physical4 and paired paths

- [ ] Apply the split weight-head prefetch only after proving PTPC scale ordering remains correct.
- [ ] Test K192/K384/K512 with per-token activation scale and per-channel weight scale.
- [ ] Measure against physical4 using 10-buffer ABBA8; advance only if paired ratio is below `0.997`.

## P3: zero-memory scheduling probes

- [x] Allow VMEM reads to cross only the paired A-stage compiler scheduling barrier (`sched_barrier(0x20)`); keep all real barriers unchanged. The rebuilt final ISA was byte-identical to control, so this global placement has no executable effect and was removed.
- [ ] If global `0x20` contaminates a Stage B interval, test isolated K0->K1 and K1->K2 masks.
  Expected gain per isolated mask: `0.1%-0.35%` (`1.9-6.7 us`), mutually exclusive with the global probe.
- [ ] Require all three Stage B intervals to remain exactly 64 MFMA and no other instructions.

## P4: splitk and batch1

- [x] Capture path-specific baseline timings and ISA/resource counts before changing scheduling.
- [x] For `splitk`, test only VMEM issue ordering and output scatter grouping; do not add workgroup barriers. VMEM-read crossing regressed by `86.5%` and was removed.
- [x] For `batch1`, test weight-load issue order and BF16 atomic grouping; do not introduce paired workgroups. VMEM-read crossing passed BF16/FP8 accuracy and ABBA24, and is now batch1-only.
- [x] Keep these results separate from the H3 fixed target because their shapes and output contracts differ.

## Closed routes

- [x] LDS weight sharing: 1 atom ratio `1.286286`; 3 atoms `1.099853`; 4 atoms reached 256 VGPR.
- [x] Pure permute rescheduling: ratio `1.020263`, 0/8 wins.
- [x] Slot priority 1/2: ratio `1.029335`, 0/8 wins.
- [x] Store26: ratio `1.019395`, 0/8 wins.
- [x] Expert stripe2: ratio `1.005637`, 1/8 wins.
- [x] Grid3d/grid2d: ratios `1.005300`/`1.000011`; no stable gain.
- [x] Direct row-major paired stores: ratio `1.232053` versus physical4 and `1.363678` versus packed paired, 0/8 wins in both comparisons.
- [x] Paired row-major CShuffle: correct at N512/N1024, but 65,536B LDS caused 256 VGPR, 27 spills, and 88B private memory.
- [x] Paired row-major bpermute: correct at N512/N1024, but 128 `ds_bpermute` instructions caused 256 VGPR, 4 spills, and 12B private memory.
- [x] Splitk VMEM-read crossing: ratio `1.865054`, 0/8 wins.
- [x] PTPC paired group-local scale LDS: legal padded metadata still crashed at kernel launch; reverted before timing.