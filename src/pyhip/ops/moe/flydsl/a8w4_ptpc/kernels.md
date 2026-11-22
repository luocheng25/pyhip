# MoE PTPC A8W4：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。 -->

本文覆盖 prefill（T≥129）的两个 1x4 kernel；decode 的 gemv / split-K 目前只有[入口文档](README.md)第 2 节的骨架。
形状按 Qwen3.8 单卡（[测试形状](../../../../testing/moe_a8w4_ptpc.py#L18)）：H=2560、topk=10、E=512；TP2 I=320，TP4 I=160 补零到 192。
tile 取 [`CONFIGS`](moe.py#L36) 的 prefill 行：tile_m=64、Gate/Up BN=128 / BK=128、Down 64×256、padding=128、half_m=True。
预取配置：prefetch_gate=2（只作用于 32 行 tile），prefetch_down=2。
记号：A 是激活（FP8，按排序行 gather），B 是权重（mxfp4）；MFMA 的 M 维是权重通道、N 维是排序行；`# L…` 指 prefill.py 的行。
任务 = 一个排序块（同一专家的 64 个排序行）；任务数 = `sorted_expert_ids` 的长度，即排序分配的 M64 块数上限。
只有前 ⌈num_valid_ids[0]/64⌉ 个任务有效，其余 CTA 直接返回；排序把每个专家的补齐行放在它末块的末尾。

**与上游 FP8 1x4 的差异**（[prefill.py 文件头](prefill.py#L1)；只保留 ptpc + SiLU）
- 权重是 mxfp4：每 lane 每 (M repeat, 64-K 块) 读 8 B nibble（上游读 16 B FP8），外加 w1_pdelta / w2_pdelta 的 delta dword。
- MFMA 之前用 fd2 查表把 nibble 解成上游读到的那些 FP8 字节；激活路径、FP8 MFMA 与收尾不变，w_scale 换成通道因子。
- Down 固定 BLOCK_K=64（去掉上游 BLOCK_K=128 时的延迟写出与 task table）；权重提前 PREFETCH 个 K 步、跨 N tile 携带（上游 1 个）。
- half_m（两级）：排序块第 33 行是补齐行时只算前 32 行；Gate/Up 的 32 行 tile 权重提前 2 个 K 步（上游 1 个）。

## Gate/Up 1x4（moe_a8w4_gateup_prefill_1x4）
第一级 GEMM：FP8 x（按 token 量化）× w1 → 每个路由行的 SiLU(gate)·up，写 BF16 `mid [T, topk, I]`，随后按行量化成 FP8 进 Down。
代码：[kernel](prefill.py#L648) · [`_gateup_tile`](prefill.py#L532) · [`gemm_1x4`](prefill.py#L204) · [launch（grid）](prefill.py#L714)；
上游对应 `moe_gemm_2stage/gemm1.py` 的 [`gemm_1x4`](../moe_gemm_2stage/gemm1.py#L310)。

**任务划分**（N = 2I：TP2 640、TP4 384；K = H = 2560；BM=64、BN=128、BK=128）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (2I/128, 任务数)：x 在 TP2 为 5、TP4 为 3 | x = 通道组 blk_n，y = 任务 e |
| CTA（256 线程） | 64 行（half_m 时 32）× 64 对通道 × 全部 K | gate 通道 64·blk_n+[0,64) 与同号 up 通道；写 mid 的这 64 列 |
| wave（4 个） | 16 对通道 × 全部行 | 通道 64·blk_n+16w+[0,16)；4 个 wave 读同一份 A |
| lane | gate、up 各 4 个 16×16 块（32 行时各 2 个），每块 4 个 fp32 | 权重行 64·blk_n+16w+lane%16；A 取每个 16 行块的第 lane%16 行 |
| K 步 | 2560/128 = 20 步（须为偶数，两拍一轮） | 每 lane 读 gate、up 各 16 B nibble 与 1 个 delta dword；A 每线程 tile_m/32 个 16 B |

blockIdx.y = e：expert = sorted_expert_ids[e]，行 = sorted_ids[64e : 64e+64]（token = id & 0xFFFFFF，k = id >> 24）。
launcher 在 M·topk ≤ E 时把 grid y 缩到 M·topk（T≥129 时碰不到）。
lane 在每个 64-K 块里取 16·(lane//16) 起的 16 个 K（权重与 A 相同），分两次 K32 MFMA。
一个 w1_pdelta dword 覆盖 256 K（每 64 K 一个字节）；BK=128 时相邻两步读同一个 dword，各用其中 2 个字节。

**流水线**（拍 t 算第 t 个 K 步，r = t % 2；预取：A 总是 1 步，B 为 B_AHEAD 步）
```text
# 一个 CTA：任务 e = blockIdx.y，通道组 blk_n = blockIdx.x
if e·64 ≥ num_valid_ids[0]: return                                                  # L671
LDS sorted_lds[0:64] ← sorted_ids[64e : 64e+64]（tid < 64 各搬一个）；barrier       # L693–L698
tile_m ← 64 if token(sorted_lds[32]) < M else 32（half_m：readfirstlane 后分支）    # L702–L709
_gateup_tile(tile_m)：同一份代码按 64 / 32 行各展开一次，只改行数                   # L532
  从 sorted_lds 读 mid 写出行号、token scale 行号、A 的 gather 行号进寄存器         # L558、L581、L248
  barrier（此后 a_ping 覆盖 sorted_lds）                                            # L263
  B_AHEAD ← PREFETCH（配置 2）if tile_m ≤ 32 else 1（64 行 tile 每个权重配的 MFMA 多）  # L592
  序言：发起 A[0]（gather）与 B[0] → 槽 0；B_AHEAD=2 时 B[1] → 槽 1                 # L448–L451
        s_waitcnt(vmcnt=0)；A[0] → LDS[0]（a_ping）；barrier                        # L452–L454
  for t in 0, 2, …, 16：拍(t)；拍(t+1)（scf.for 9 轮，携带 gate / up 累加器）       # L458–L464
  拍(18)（B_AHEAD=2 时不再发 B）；拍(19)（不发读、不写 LDS）                        # L467–L469
拍(t)：                                                                             # L408
  发起 A[t+1] 的 gather 读（global → 寄存器，每线程 tile_m/32 个 16 B）             # L413
  B_AHEAD=1：发起 B[t+1] → 槽 1−r（nibble + delta）                                 # L414–L415
  fd2 解码：槽 r 的 B[t] → gate / up 的 FP8 fragment（本拍 MFMA 之前）              # L416
  B_AHEAD=2：发起 B[t+2] → 槽 r（刚解码完的槽）                                     # L417–L418
  for ki in 0, 1（各 64 K）：                                                       # L419
    A[t] 的这 64 K：LDS[r] → 寄存器（ds_read 128 b）                                # L420–L424
    gate、up 各 2·n_reps 次 MFMA 16×16×32 FP8（n_reps = tile_m/16）                 # L425–L441
  A[t+1]：寄存器 → LDS[1−r]（ds_write 128 b）；hot_loop_scheduler 把读写穿插进 MFMA  # L442–L444
  sched_barrier；barrier（LDS[1−r] 写完、LDS[r] 读完）                              # L445–L446
收尾（_gateup_tile）：
  acc × 通道因子（w1_factor）× token scale（x 的按行 scale）                        # L595
  SiLU(gate)·up（f32）→ BF16                                                        # L608
  barrier；BF16 tile 转置写进 LDS（CShuffle，复用 a_ping）；barrier；读回 128 b     # L635–L644
  按排序行写 mid[token, k, 64·blk_n …]，每次 16 B（补齐行 token ≥ M，越界写被丢弃）  # L645
```
fd2 解码（[`_decode_rep`](prefill.py#L70)）：v_perm 广播本组 delta 字节，|E2M1|→FP8 查找表整体减去它，再按 nibble 查表并补符号位。
每个 (M repeat, 64-K 块) 的 16 个 nibble 解成 16 个 FP8，排列与上游从 global 读到的 FP8 相同。

**资源**
- LDS 16 KiB：A 的 ping / pong 各 64×128 B（32 行 tile 用一半）；sorted_lds（256 个 int32）与它们共用（union）。
- CShuffle 复用 a_ping：64×64 BF16 = 8 KiB（32 行时 4 KiB）；代码断言它不超过 ping + pong。
- 按每 CU 64 KiB LDS 算，每 CU 最多 4 个 CTA。
- 寄存器（VGPR 总数未记录）：每 lane 累加器 32 个 fp32（32 行 tile 16 个）；B 两个槽，各有 gate、up 各 4 个 nibble dword 与 2 个 delta dword。

## Down 1x4（moe_a8w4_down_prefill_1x4_64x256）
第二级 GEMM：FP8 mid × w2 → 每个排序行一行 BF16（已乘路由权重），写 `routes [任务数·64, H+64]`；随后 sorted_sum 按 token 相加成 out。
代码：[kernel](prefill.py#L1035) · [`_down_tile`](prefill.py#L799) · [`_map_down_task`](prefill.py#L779) · [launch（grid）](prefill.py#L1129)；
上游对应 [`gemm2_1x4.py`](../moe_gemm_2stage/gemm2_1x4.py#L227)，gather、行系数与 CShuffle 写出的 helper 直接导入。

**任务划分**（N = H = 2560；K = I：TP2 320、TP4 192；tile 64×256；BLOCK_K = 64）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (1, 任务数) | y 经 `_map_down_task` 换成任务 e（见下） |
| CTA（256 线程） | 64 行（half_m 时 32）× 2560 列，N 循环 10 个 tile | 任务 e 的行：FP8 mid 全部 K 先进 LDS；写 routes 的这些行 |
| wave（4 个） | 每个 N tile 64 列 × 全部行 | 列 256·n + 64w + [0,64) |
| lane | 4×4 个 16×16 块（32 行时 4×2），每块 4 个 fp32 | 累加 16 个连续列 × 每 16 行块 1 行；每 K 步读 4 列 × 16 个 K 的权重 |
| K 步 | 64 K：TP2 5 步、TP4 3 步 | 每 lane 读 32 B nibble 与 1 个 delta dword；A 从 LDS 读 |

blockIdx.y = b：c = ⌊有效任务数 / 4⌋（4 = MI308X 的 XCD 数）；b < 4c 时 e = (b%4)·c + b//4，否则 e = b。
即 b%4 相同的 CTA 处理连续的 c 个任务（代码注释：在 XCD 间交错）；e·64 ≥ num_valid_ids[0] 的 CTA 返回。
lane l 的累加器是本 wave 第 16·(l//16) 起的 16 列：权重视图按上游的 N256 行序排列，使这 16 列连续，打包成 16 B 写进 CShuffle。
一个 w2_pdelta dword 属于 (64 列块, 32-K 组, lane%16)，4 个字节是 lane 的 4 个 M repeat；lane//32 选组的奇偶。
routes 行距 H + 64 个 BF16（padding 128 B）；任务 e 写第 64e 起的行，补齐行 sorted_sum 不读。

**流水线**（K 步 k = 64 K；权重提前 pf = min(PREFETCH, K/64) = 2 步，32 / 64 行 tile 相同）
```text
# 一个 CTA：任务 e 的 64 个排序行 × 全部 2560 列；N tile n = 256 列
e ← _map_down_task(blockIdx.y)；if e·64 ≥ num_valid_ids[0]: return                  # L1049–L1052
second ← sorted_ids[64e + 32]（global，给 half_m）                                  # L1109
gather：64 行 × K 的 FP8 mid 按 (token, k) 读进 LDS（16 B 一次）；barrier           # L1111
tile_m ← 64 if token(second) < M else 32（readfirstlane；gather 总是 64 行）        # L1118–L1123
_down_tile(tile_m)：                                                                # L799
  行系数 ← sorted_weights × mid 的行 scale（按 (token, k)；C 布局，整个 kernel 不变）  # L900–L911
  序言：N tile 0 的前 pf 个 K 步 → pf 个权重槽（各 8 个 nibble dword + 1 个 delta dword）  # L978–L981
        N tile 0 的通道因子：global → LDS（每 wave 自己的 64 个）→ 寄存器           # L982
  for n in 0..9（scf.for；携带下一 tile 的 pf 个权重槽与通道因子）：                # L1013–L1019
    acc ← 0；发起 N tile n+1 通道因子的 global 读（K 步 0 之前）                    # L1021、L990
    for k in 0..K/64−1（展开）：                                                    # L992
      A[k]：LDS → 寄存器                                                            # L993
      发起 K 步 k+pf 的权重读；越过本 tile 时读 N tile n+1 的前 pf 步（带进下一轮，n=9 时照发不用）  # L996–L1002
      fd2 解码 K 步 k 的槽 → FP8 fragment（本步 MFMA 之前）                         # L1003
      s_setprio 3；acc += W·A：4×4 块（32 行时 4×2）× 2 次 MFMA 16×16×32；s_setprio 0  # L1004–L1006
    N tile n+1 的通道因子 → LDS → 寄存器                                            # L1007
    acc × 通道因子 × 行系数 → BF16                                                  # L1025–L1027
    每 16 行：BF16 写本 wave 的 CShuffle LDS（16×64），读回，每 lane 写 2 × 16 B    # L1028
# gather 之后没有 CTA barrier：通道因子与 CShuffle 的 LDS 都按 wave 分区
```
gather 用 [`_gather_sorted_rows`](../moe_gemm_2stage/gemm2_1x4.py#L34)；写出走 non-temporal 的 buffer store（[`_store_cshuffle_row_pair`](../moe_gemm_2stage/gemm2_1x4.py#L140)）。

**资源**
- LDS = A 64·K B + 通道因子 256×4 B + CShuffle 4×16×64 BF16：TP2 20480 + 1024 + 8192 = 29696 B，TP4 21504 B。
- 代码断言合计 ≤ 64 KiB（即 K ≤ 880）；按每 CU 64 KiB LDS 算，每 CU 最多 TP2 2 个、TP4 3 个 CTA。
- 寄存器（VGPR 总数未记录）：每 lane 累加器 64 个 fp32（32 行 tile 32 个）；权重最多 pf + 1 = 3 个槽同时在寄存器。
- 编译时关闭 packed-fp32-ops（同上游）。
