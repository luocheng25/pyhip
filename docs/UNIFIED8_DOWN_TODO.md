# FlyDSL MoE down optimization TODO

> Packed-best promotion commit: `c82e2df`; current integrated kernel source SHA256:
> `ceb45548316522b0dc2c316aa6a8d23114da1a4af36dd0d4e164fbd5e99891a8`.
>
> Fixed H3 target: `1.895517 ms`; promoted result: `1.913011 ms`; gap: `17.494 us` (`0.9145%`).
>
> Do not promote a candidate without the path-specific correctness gate, resource/ISA gate, and controlled ABBA gate listed below.

## 2026-08-19 all-down checkpoint

### Expanded 8-wave support and three-way selection

- Paired 8-wave now accepts every requested FP8 down-prefill combination with `BM=64`, `N % 512 == 0`, `K % 64 == 0`, and `64 <= K <= 512`: PTPC weight + PTPC activation, or per-tensor weight + PTPC/per-tensor activation. The exhaustive persistent matrix crosses N512/N1024, all eight K points, and all three quant combinations (48/48), while checking M128 padding and inactive tails. The real dispatcher also passes forced PTPC K256/K512.
- PTPC scale ownership is fixed by indexing both four-wave groups with `local_tid`; both groups stage the same N256 scale block. K512 cannot fit the extra 1KB scale LDS, so it uses direct global-to-C-fragment scale loads. It is numerically correct, but final ISA has 256 VGPR, 19 spills, and 80B private memory, so it is never auto-selected.
- Controlled 10-buffer ABBA8 compared base, physical N256, and row-major paired 8-wave, including `sorted_sum`. Generic shapes still select the measured best of base and physical N256. A later streaming wave-private CShuffle removed the exact-H3 consumer penalty and all epilogue spills, so exact H3 per-tensor now auto-selects paired N512; `MOE_DOWN_PAIRED_N512=0/1` remains the disable/force override.
- Automatic winner boundaries: physical N256 for K64-K320; physical N256 for K384 only with per-tensor weights; base for PTPC K384 and all K448/K512 cases. Shape-specific row padding preserves measured combined winners: Hy3 K192 and per-tensor H3 K384 use 0B, Xiaomi PTPC K256 uses 0B, generic physical cases use 128B except per-tensor N4096 K64/K128.

- The packed H3 formal best is unchanged: its rebuilt final ISA is byte-identical to the promoted artifact. It still has 192 MFMA, 39 128-bit loads, 16 stores, 10 real barriers, 254 VGPR, 49,152B LDS, and zero scratch.
- The generic dispatcher does not consume the packed layout directly. The promoted row-major epilogue streams each staged BF16 vector pair through wave-private LDS, retaining the original K-stage overlap and delaying global address generation until after DS reads. Exact-H3 ABBA24 versus physical4 improved down `2.4743 -> 2.3552 ms` (24/24 wins) and full pipeline `8.6356 -> 8.5859 ms` (22/24 wins), with bitwise-equal reduced output.
- Timing-contract clarification: historical `1.913011 ms` is the packed producer only, while streamed row-major down includes the layout conversion needed by standard `sorted_sum`. The formal same-process exact-grid ABBA24 harness measured packed `1.9350 + 3.6740 ms` versus streamed row-major `2.0943 + 0.7415 ms` for down + consumer. The producer cost is `+8.23%`, but combined drops from `5.8432` to `2.7887 ms`. Against physical N256 in the same formal harness, streamed row-major has down ratio `0.94760` and combined ratio `0.96644`. The packed ISA remains byte-identical.
- The batch1 down path now allows VMEM reads to cross its compiler scheduling barriers. H3-dimension ABBA24 improved FP8 by `4.32%` (`0.956752`, 24/24 wins) and BF16 by `0.24%` (`0.997616`, 21/24 wins). BF16 and FP8 dispatcher accuracy gates pass.
- Applying the same mask to splitk is rejected: `0.872255 -> 1.626202 ms`, ratio `1.865054`, 0/8 wins. Splitk remains unchanged.
- The focused selector suite covers all 24 K/quant winner cells plus boundaries; the complete down suite includes the exhaustive 48-case paired matrix, repaired physical4 two-block, and batch1 BF16/FP8 atomic coverage.

## Path matrix

| Down path | Current implementation | Strict paired method applicability | Required gate |
| --- | --- | --- | --- |
| H3 fp8 per-tensor `prefill_1x4`, N4096/K384/TOPK9/E193 | packed paired formal best plus auto-selected streaming row-major adapter | Paired N512 streams through wave-private CShuffle; packed default remains unchanged | exact-grid physical/reduced/tail bitwise gate; 256 VGPR, 65,536B LDS, 0 scratch; ABBA8/24 |
| Generic fp8 per-tensor `prefill_1x4` | physical4 fallback | Direct row-major pairing passes correctness but fails performance; do not expose packed block-major stores to the generic consumer | N512/N1024 PyTorch oracle plus odd logical-block tail |
| fp8 PTPC `prefill_1x4` | auto-selects base or physical4; paired8 is explicit | Paired8 supports K64-K512; staged scale uses group-local thread numbering, K512 uses direct scale loads | all eight K points, multi-N, and inactive tail |
| Non-physical `prefill_1x4` (bf16/fp8) | original 4-wave path | Reuse scheduling ideas only; physical pairing changes its output and LDS contract | existing prefill tests plus shape matrix |
| `splitk` down | one-wave compute with scatter/atomic output | Do not apply 4+4 anti-phase directly; only test load scheduling and mapping independently | atomic and non-atomic oracle, multi-expert routing |
| `batch1` down | one-wave compute with BF16 atomic reduction; VMEM-read crossing enabled | Pairing is structurally inapplicable; the compatible scheduling idea is promoted | batch1 TOPK reduction oracle |

## P0: correctness and contract

- [x] Compare H3 paired output against an independently correct physical4/PyTorch path, not only against a descendant of the same packed layout. Random unique-top-k H3 produced zero reduced-output mismatches; the real dispatcher reports `diff=0.00017182`.
- [x] Determine whether the H3 packed store is row-major or requires a matching consumer/decode step. It is block-major packed; the repaired `packed_direct` decoder is correct but too slow for production consumption.
- [ ] Add an exact-H3 regression that checks mathematical output and inactive tail independently of the candidate module.
- [x] Keep generic shapes on physical4/base; auto-promote only exact H3 after the streaming row-major performance gate passed.

## P1: generic per-tensor prefill pairing

- [x] Prototype a row-major direct-store adapter for the paired accumulator fragment without extra real barriers. It preserves 192 MFMA, 39 loads, 16 stores, 10 barriers, 254 VGPR, 49,152B LDS, and zero scratch after final-store deduplication.
- [x] Test transpose alternatives after direct stores proved expensive. A 64KB CShuffle version reached 256 VGPR with 27 spills/88B private memory; a wave-local bpermute version reached 256 VGPR with 4 spills/12B private memory. Both were reverted.
- [ ] Validate N512, N1024, N4096, K192/K384, one and two logical M64 blocks, and a non-multiple-of-128 valid tail.
- [x] Require zero scratch and no regression in the down suite before timing. The retained direct adapter has zero scratch; persistent paired and physical4 tests cover two M64 blocks.

## P2: PTPC physical4 and paired paths

- [x] Keep the split weight-head prefetch restricted to the original per-tensor K384 schedule; PTPC uses the generic K-core path.
- [x] Test K64-K512 with per-token activation scale and per-channel weight scale, including two N256 blocks and inactive tails.
- [x] Measure base, physical4, and paired8 using 10-buffer ABBA8. The original direct row-major adapter did not beat the best existing path and remained explicit-only; the later zero-spill streaming adapter supersedes that result for exact H3 only.

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
- [x] Initial PTPC paired scale prototype crashed because waves 4-7 indexed beyond the shared N256 scale block. Fixed with `local_tid`; K64-K384 staged-scale and K512 direct-scale paths now pass.