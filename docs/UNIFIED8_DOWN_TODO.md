# FlyDSL MoE down optimization TODO

> Packed-best promotion commit: `c82e2df`; historical integrated-kernel checkpoint SHA256:
> `a8a017e81f52699cd33149266639fb14743412b419cf757a7c118c2d03946de7`.
>
> Current uncommitted six-entry refactor source SHA256 (2026-08-21):
> `b93faa818a602d19645e30cb6fbe27f2231ca3ebbf892c93fc7037df5c196b6c`.
>
> The historical `a8a017...` source snapshot is not retained in the current
> Git/stash set; keep that hash as provenance, not as a rebuildable artifact.
>
> Fixed H3 target: `1.895517 ms`; promoted result: `1.913011 ms`; gap: `17.494 us` (`0.9145%`).
>
> Do not promote a candidate without the path-specific correctness gate, resource/ISA gate, and controlled ABBA gate listed below.

## 2026-08-19 all-down checkpoint

### Expanded 8-wave support and three-way selection

- Paired 8-wave now accepts every requested FP8 down-prefill combination with `BM=64`, `N % 512 == 0`, `K % 64 == 0`, and `64 <= K <= 512`: PTPC weight + PTPC activation, or per-tensor weight + PTPC/per-tensor activation. The exhaustive persistent matrix crosses N512/N1024, all eight K points, and all three quant combinations (48/48), while checking M128 padding and inactive tails. The real dispatcher also passes forced PTPC K256/K512.
- PTPC scale ownership is fixed by indexing both four-wave groups with `local_tid`; both groups stage the same N256 scale block. K512 cannot fit the extra 1KB scale LDS, so it uses direct global-to-C-fragment scale loads. It is numerically correct, but final ISA has 256 VGPR, 19 spills, and 80B private memory, so it is never auto-selected.
- Controlled 10-buffer ABBA8 compared base, physical N256, and row-major paired 8-wave, including `sorted_sum`. Generic shapes still select the measured best of base and physical N256. A later streaming wave-private CShuffle removed the exact-H3 consumer penalty and all epilogue spills, so exact H3 per-tensor now auto-selects paired N512; `MOE_DOWN_PAIRED_N512=0/1` remains the disable/force override.
- Automatic winner boundaries: physical N256 for K64-K320; physical N256 for K384 only with per-tensor weights; base for PTPC K384 and all K448/K512 cases. Shape-specific row padding preserves measured combined winners: Hy3 K192 and per-tensor H3 K384 use 0B; Xiaomi PTPC K256 and generic physical cases use 128B except per-tensor N4096 K64/K128.

- The packed H3 formal best is unchanged: its rebuilt final ISA is byte-identical to the promoted artifact. It still has 192 MFMA, 39 128-bit loads, 16 stores, 10 real barriers, 254 VGPR, 49,152B LDS, and zero scratch.
- The generic dispatcher does not consume the packed layout directly. The promoted row-major epilogue streams each staged BF16 vector pair through wave-private LDS, retaining the original K-stage overlap and delaying global address generation until after DS reads. Exact-H3 ABBA24 versus physical4 improved down `2.4743 -> 2.3552 ms` (24/24 wins) and full pipeline `8.6356 -> 8.5859 ms` (22/24 wins), with bitwise-equal reduced output.
- Timing-contract clarification: historical `1.913011 ms` is the packed producer only, while streamed row-major down includes the layout conversion needed by standard `sorted_sum`. The formal same-process exact-grid ABBA24 harness measured packed `1.9350 + 3.6740 ms` versus streamed row-major `2.0943 + 0.7415 ms` for down + consumer. The producer cost is `+8.23%`, but combined drops from `5.8432` to `2.7887 ms`. Against physical N256 in the same formal harness, streamed row-major has down ratio `0.94760` and combined ratio `0.96644`. The packed ISA remains byte-identical.
- The batch1 down path now allows VMEM reads to cross its compiler scheduling barriers. H3-dimension ABBA24 improved FP8 by `4.32%` (`0.956752`, 24/24 wins) and BF16 by `0.24%` (`0.997616`, 21/24 wins). BF16 and FP8 dispatcher accuracy gates pass.
- Applying the same mask to splitk is rejected: `0.872255 -> 1.626202 ms`, ratio `1.865054`, 0/8 wins. Splitk remains unchanged.
- The focused selector suite covers all 24 K/quant winner cells plus boundaries; the complete down suite includes the exhaustive 48-case paired matrix, repaired physical4 two-block, and batch1 BF16/FP8 atomic coverage.
- 2026-08-20 streaming-paired expansion gate: 72 generic cells (`N=512/1024/4096`, K64-K512, three FP8 quant combinations) plus four production shapes compared base, physical N256 (0/128B), and paired M128 (0/128B); every reduced output was bitwise equal. No generic M128 cell survived a reliable direct ABBA24 versus the current winner; H3 PTPC was promoted only after the later spill-removal specialization below. Xiaomi PTPC K256 padding changed from 0B to the clean-window ABBA24-winning 128B (`combined ratio 0.99756`, 18/24 wins). N512/N1024 base/physical boundary promotion remains pending because an external all-GPU workload contaminated the follow-up microsecond-scale timings.
- Aggregate best-paired/best-non-paired combined ratios were `1.1831` at N512, `1.1282` at N1024, and `1.0574` at generic N4096 (72-cell median `1.1212`). ABBA4 produced four apparent paired winners; direct ABBA24 rejected all four (`1.00255`, `1.02040`, `1.01782`, and `0.99842` with only 14/24 wins). Thus the generic table gained no automatic M128 promotion.
- H3 PTPC K384 was subsequently promoted after removing current/next direct-scale fragments from the MFMA loop-carried state. Fresh ISA changed from 18 VGPR spills / 76B private / 25+17 scratch load/store to zero spill/private/scratch. Full paired correctness (48/48) and production `diff=0.00019035` pass. ABBA24 combined ratios are `0.60734` versus the old spilling 8-wave, `0.95473` versus Base, and `0.79605` versus physical N256; all are 24/24 wins. Non-target exact-H3 and Qwen-K512 old/new ratios are `0.99930` and `0.99772` with IQRs crossing 1; Hy3/Xiaomi final ISA is byte-identical. Auto M128 now covers exact H3 per-tensor and H3 PTPC.
- Hy3 K192 now has a separate single-M N512 specialization rather than M128 pairing: M64 sorting is unchanged, eight waves span N512, and a balanced 512-thread A copy removes idle waves. K64 LDS swizzle plus `amdgpu-waves-per-eu=4,4` reaches 128 VGPR / 32KB LDS / zero scratch, allowing two 512-thread workgroups; final ISA has 96 MFMA and only two barriers. Physical N256 and single-M outputs/tails are bitwise equal, production accuracy is `diff=0.00016577`, and the full down suite is 62/62. Clean 10-buffer/1800MHz ABBA24 improved down `1.673130 -> 1.513829 ms` (`0.906541`, IQR `0.903067--0.910706`) and combined `2.511415 -> 2.347834 ms` (`0.935520`, IQR `0.931968--0.939437`), both 24/24 wins.
- Independent tracked-harness clean ABBA24 reproduced the Hy3 result: down `1.557529 -> 1.417749 ms`, ratio `0.909806` (IQR `0.908210--0.912926`, 24/24); combined `2.283913 -> 2.141253 ms`, ratio `0.937691` (IQR `0.934980--0.940317`, 24/24). Consumer ratio `1.000015` had an IQR crossing one, confirming the improvement is in down.
- H3 PTPC can safely share the streaming CShuffle scratch between its anti-phased four-wave groups. This reduces LDS from 64KB to 56KB without changing occupancy, MFMA, stores, barriers, or output. Two 10-buffer ABBA24 runs against the old H3 PTPC path produced down ratios `0.99538` and `0.99393`, and combined ratios `0.99607` and `0.99618`; both runs were bitwise equal. The gain is small and the combined IQR upper edge remains slightly above one, so treat this as a low-risk resource improvement rather than a new throughput tier.
- A genuine `(4 N waves) x (2 M waves)` M128xN256 tiled-MMA prototype was validated for H3 PTPC and Xiaomi PTPC. It used M64 sorting, supported different experts in the two M64 halves, passed production-size bitwise checks and odd-tail checks, and reduced the core to two barriers/eight stores with zero spill. It still lost: H3 combined ratio `1.01448` versus the promoted paired path, and Xiaomi down/combined ratios `1.09436`/`1.06540` versus physical N256. Slot-staggering Xiaomi reproduced `1.09393`/`1.05675`. The remaining gap is not padding, spill, or barrier count; one 48KB/64KB 8-wave workgroup cannot match the latency hiding of two resident physical-N256 workgroups. Do not repeat this layout unchanged.
- A fresh padding-neutral comparison isolated the apparent physical-N256 advantage. `B8192/TOPK8/E128/N4096/K384`, double per-tensor, gives exactly 512 rows/expert: M64 and M128 sorting ratios are both `1.0`, row padding is 0B, and both paths launch exactly 1024 M64 tasks. ABBA24 still showed paired slower (`1.01346` down, `1.01946` combined), but fresh ATT showed its sampled core was better: steady MFMA-union busy `90.73%` versus physical `86.29%`, lifecycle busy `86.08%` versus `82.63%`, and median wave lifetime `113457` versus `115393` cycles. Paired idle was chiefly barrier imbalance (`54.14%` of its smaller idle budget); physical idle was chiefly structural tail (`45.01%`) and VMEM stall/wait (`27.24%`). Raw traces are `/tmp/padding-neutral-{physical,paired}-att`; unified reports are `/tmp/padding-neutral-n256-vs-paired-{slots,exposure}.{json,md}`.
- The remaining difference is cross-CU workgroup granularity, which single-CU ATT does not measure. Equal total work is 1024 four-wave physical WGs versus 512 eight-wave paired WGs. Across 80 CUs this is `12/13` physical WGs (`48/52` waves) but `6/7` paired WGs (`48/56` waves); the paired critical CU carries 7.69% more waves. The ATT target happened to sample only 48-wave paired CUs, while physical samples contained both 48- and 52-wave CUs. A second zero-padding shape with 1280 M64 tasks (`B10240`) balances exactly at 16 physical or 8 paired WGs per CU, 64 waves each; paired then won ABBA24 by `0.96455` down and `0.97949` combined, 24/24. Thus strict 8-wave anti-phase works; the apparent reversal is predominantly dispatch-tail load imbalance caused by doubling the WG scheduling quantum.
- A second paired ATT capture on another target CU directly observed the missing heavy tail: shader engines had `48/56/56/48` captured waves and full spans `700780/804632/796092/706748` cycles, versus the first paired target's uniform 48 waves and roughly 691K--699K cycles. The physical target had `48/52/52/48` waves and roughly 738K--795K cycles. For `T` logical M64 tasks on 80 CUs, critical waves/SIMD are `ceil(T/80)` for physical and `2*ceil(T/160)` for paired. Paired pays one extra wave batch exactly when `T mod 160` is in `1..80`; `T=1024` gives 13 versus 14, while `T=1280`, exact H3 (`T=4632`), and H3 PTPC (`T=2048`) have equal critical-wave counts.
- Full-matrix falsification shows that this dispatch-tail effect is common but not the sole cause of generic 8-wave regressions. The current source was tested for all 72 `N=512/1024/4096`, K64--K512, and quantization cells with exact valid grids, 0B row padding, and direct `paired-pad0 / physical-pad0` ABBA8. Moving from `T=1024` (13 physical versus 14 paired critical waves/SIMD) to balanced `T=1280` (16 versus 16) improved the down ratio by a median `5.20%`, from `1.1280` to `1.0600`. Of the 60 cells that were slower at `T=1024`, only 11 flipped below one; 49 remained slower. Combined ratios improved by a median `4.81%`, but only 8 of 60 regressions flipped and 52 remained. This direct physical comparison does not imply dispatcher promotion: the current Base path is still the relevant winner in several K448/K512 cells.
- The effect is dominant mainly for long-N, near-break-even cells. At N4096, balancing moved the median down ratio `1.0487 -> 0.9982` and flipped 9/20 regressions. At N512 and N1024 it moved `1.1912 -> 1.1475` and `1.1619 -> 1.0902`, but flipped only 1/20 regressions at each N. Thus the larger WG scheduling quantum contributes about five percentage points broadly, while short-N paired rendezvous, output conversion, and reduced independent-WG latency-hiding remain insufficiently amortized. A current N512/K128 double-per-tensor paired ISA has 184 VGPR, 32KB LDS, and zero scratch, yet balanced ABBA24 still loses `1.14147` down and `1.06581` combined, ruling out spills as the explanation for that class.
- Resource-specific regressions form a separate class. Balanced PTPC/PTPC remains slower in 18/24 cells. Current generic K512 PTPC paired ISA has 256 VGPR, 64KB LDS, 80B private memory, 17 scratch loads, and 17 scratch stores; balanced ABBA24 still loses `1.38757` down and `1.34403` combined. Conversely, N4096/K384 double-per-tensor changes from `1.03263` down at `T=1024` to `0.97846` at `T=1280` (24/24 balanced down wins), which is the clean dispatch-tail-dominated class. N4096/K64 double-per-tensor is a boundary: balanced ABBA24 down is `0.97922` with an IQR crossing one and combined is `1.00596`, also crossing one.
- Reproduction logs and SHA256: `/tmp/unbalanced-matrix-all-k-exact-abba8.log` (`a8242e321784081cf4c24c179ffd4164fec42403a149ba5bcf745da4c61a896e`), `/tmp/unbalanced-small-all-k-exact-abba8.log` (`c3c7f849ea545405188810d4c3e4d41c0bad30839129aba80d676209ba0708b5`), `/tmp/balanced-matrix-all-k-abba8.log` (`99ab47365861dadf4e7dac528092baecedb6f6e8a41cdb7d1fe7cb3e1bf5e3d4`), `/tmp/balanced-small-all-k-abba8.log` (`50dce038efa130694bdabbdc31b0f2df81326949f4ad04dffdaa401888bbb405`), and representative ABBA24 `/tmp/tail-classification-representative-abba24.log` (`428cf2442bba08aa1ca2e147985b6fdb243584cded17ec6010572a0205ed947c`).

## Path matrix

| Down path | Current implementation | Strict paired method applicability | Required gate |
| --- | --- | --- | --- |
| H3 fp8 per-tensor `prefill_1x4`, N4096/K384/TOPK9/E193 | packed paired formal best plus auto-selected streaming row-major adapter | Paired N512 streams through wave-private CShuffle; packed default remains unchanged | exact-grid physical/reduced/tail bitwise gate; 256 VGPR, 65,536B LDS, 0 scratch; ABBA8/24 |
| H3 fp8 PTPC `prefill_1x4`, N6144/K384/TOPK4/E128 | paired N512 streaming row-major with epilogue-time scale load | Late scale removes all 18 spills while preserving 64KB LDS | 48-case paired matrix, production accuracy, Base/old-paired ABBA24 |
| Hy3 fp8 per-tensor `prefill_1x4`, N4096/K192/TOPK9/E193 | single-M N512 with M64 sorting and row-major CShuffle | Eight waves span N; no M128 metadata duplication or paired rendezvous | bitwise physical/reduced/tail gate; 128 VGPR, 32KB LDS, 0 scratch; clean ABBA24 combined ratio 0.935520 |
| Generic fp8 per-tensor `prefill_1x4` | Existing base/physical4 winner table | Streaming row-major pairing passes correctness but no non-H3 cell passed reliable direct ABBA24 performance promotion | N512/N1024/N4096 K/quant matrix plus odd logical-block tail |
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