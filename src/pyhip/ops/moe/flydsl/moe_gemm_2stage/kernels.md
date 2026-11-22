# MoE A8W4：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。 -->

两段 GEMM 共用 [a8w4_common.py](a8w4_common.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；lane `16b + r` 读第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32 的操作数。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](a8w4_common.py#L58) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](a8w4_common.py#L69) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼。
- 解出的值是 |e2m1| × 2^(5−delta)，delta ≤ 11 时精确；通道因子 s = 2^(cmax−132)（f32）与激活 scale 在 epilogue 一起乘。

## A8W4 Gate/Up（[a8w4_gemm1.py](a8w4_gemm1.py#L344)）
按排序块算 mid[token, slot, :] = SiLU(x·w1_gate)·(x·w1_up)，f32 写出；结构同底座 `gemm1.py` 的 prefill_1x4，权重换成 mxfp4。

**任务划分**（默认 BM64、BN128、TILE_K256；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (2I/128, 排序块数) | x：64 gate + 同号 64 up 通道；y：1 个排序块 |
| CTA（256 线程） | 4 个 wave | 64 个排序行（同一 expert）× 128 通道 × 全部 H |
| wave | 1 个 16 通道块 | gate、up 各 16 通道 × 64 行（4 个 token 块） |
| lane | l = 16b + r | 权重行 r 的 K 16b…16b+15；token l%16 的 4 个累加 |
| K 步 | tile 256 = 4 个 64-K 块 | 每个 64-K 块每 wave 16 条 MFMA 16x16x32 |

`e·64 ≥ num_valid` 的 CTA 直接退出；M·topk ≤ E 时 grid 的 y 维取 M·topk（每条路由至多占一个排序块）。

**流水线**（逐拍；x 的 LDS 双缓冲 A_ping / A_pong，权重 VGPR 两槽 0 / 1，预取距离 1 个 tile）
```text
序言
  tid < 64：sorted_ids[e·64 + tid] → LDS；barrier
  各线程从 LDS 取 x 的 gather 行号与 epilogue 的 (token, slot)；barrier（LDS 让给 A_ping）
  预取 tile 0：gather x（每线程 4 × 16 B buffer load）→ VGPR；本 wave 权重 → 槽 0
  s_waitcnt vmcnt(0)；x → A_ping；barrier
主循环 i = 0 … H/512 − 2（每次两拍；H2560 为 4 次）
  拍 0（读 A_ping，算 tile 2i）：
    发预取 tile 2i+1：gather x → VGPR；权重 → 槽 1（每 lane 8 × dwordx2 + 2 × dword）
    for ki in 0 … 3（64-K 块）：
      A_ping 第 ki 块 → B 操作数（64 行 × 64-K，每 lane 64 B）
      解码：gate、up 各取 delta 第 ki 字节 → LUT；各 2 个权重 dword → 16 个 fp8
      MFMA 16x16x32 fp8 × 16（gate/up × 4 个 token 块 × 2 个 K32）
    预取的 x：VGPR → A_pong；sched_barrier；barrier
  拍 1（读 A_pong，算 tile 2i+1）：同上，预取 tile 2i+2 → 槽 0 与 A_ping
收尾：拍 0（tile H/256 − 2，预取最后一个 tile）；拍 1（最后一个 tile，不预取）
epilogue（累加器留在 VGPR，不过 LDS）
  acc ×= 通道因子 s[ch]（gate、up 各自）× xs[token]
  SiLU(gate)·up = gate · rcp(1 + exp2(−gate·log2e)) · up
  每个 token 块 4 个 f32 一次 16 B buffer store → mid[token, slot, ch … ch+3]
  填充行（token ≥ M）的偏移改成越界值，store 被硬件丢弃
```
关键代码：[排序行号进 LDS](a8w4_gemm1.py#L381)、[序言预取](a8w4_gemm1.py#L276)、[一拍](a8w4_gemm1.py#L242)、
[权重加载](a8w4_gemm1.py#L218)、[解码](a8w4_gemm1.py#L230)、[主循环](a8w4_gemm1.py#L284)、
[反量化](a8w4_gemm1.py#L298)、[SiLU 与写出](a8w4_gemm1.py#L65)。

**资源**：LDS 32 KiB（A_ping / A_pong 各 64 × 256 B，排序行号的 1 KiB 与之复用）；寄存器与占用未统计。

## A8W4 Down（[a8w4_gemm2.py](a8w4_gemm2.py#L50)）
按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同底座 `gemm2.py` 的默认 prefill Down。

**任务划分**（默认 BM64；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (1, 排序块数) | 1 个排序块 × 全部 H 列 |
| CTA（256 线程） | 4 个 wave | 64 行 mq 整块驻留；逐个 64 列块走完 H |
| wave | 每个 64 列块中的 16 列 | 列 n·64 + 16w … +15 × 64 行 |
| lane | l = 16b + r | 权重列 r 的 K 16b…16b+15；B 操作数的 1/64 |
| K 步 | I/64 个 64-K 块 × 2 个 K32 | 每个列块一次算完全部 K（I320 为 10 步） |

退出与 grid 收缩同 Gate/Up。H2560 共 40 个列块；M=1 时只有 11 个 CTA（每条路由一个排序块）。

**流水线**（逐拍；权重 VGPR 两槽 0 / 1，预取距离 1 个列块；输出 LDS 双缓冲 C[0] / C[1]）
```text
序言
  全 CTA 按 sorted id gather mq 的 64 行（每次 16 B，共 64·I/4096 轮）→ LDS A；barrier
  LDS A → VGPR：64 × I 整块作 B 操作数；barrier（之后 LDS 改作输出缓冲 C）
  行 scale = ms[token, slot] × 路由权重，按累加器布局放在 VGPR
  load_block(0) → 槽 0：本 wave 16 行权重 I/64 × dwordx2 + ⌈I/256⌉ × dword，及 64 个通道因子
  compute(槽 0) = 第 0 块；load_block(1) → 槽 1；第 0 块 × 行 scale → bf16 → C[0]；barrier
主循环 n = 0, 2, …, H/64 − 4（每次两拍；H2560 为 19 次）
  拍 0：C[0] → routes 第 n 块；load_block(n+2) → 槽 0
        compute(槽 1) = 第 n+1 块 → × 行 scale → bf16 → C[1]；barrier
  拍 1：C[1] → routes 第 n+1 块；load_block(n+3) → 槽 1
        compute(槽 0) = 第 n+2 块 → × 行 scale → bf16 → C[0]；barrier
收尾：C[0] → 第 H/64 − 2 块；compute(槽 1) = 最后一块 → C[1]；barrier；C[1] → 最后一块
compute(槽)
  for kb in 0 … I/64 − 1：delta dword[kb/4] 第 kb%4 字节 → LUT；2 个权重 dword → 16 个 fp8
  for kb：MFMA 16x16x32 fp8 × 8（4 个 token 块 × 2 个 K32）；结果 × 通道因子
C → routes：bf16 结果经 LDS 转成按行连续，再 16 B buffer store 合并写出
```
关键代码：[gather](a8w4_gemm2.py#L119)、[激活进 VGPR](a8w4_gemm2.py#L139)、[行 scale](a8w4_gemm2.py#L166)、
[load_block](a8w4_gemm2.py#L189)、[compute](a8w4_gemm2.py#L206)、[写 LDS](a8w4_gemm2.py#L229)、
[写 routes](a8w4_gemm2.py#L240)、[主循环](a8w4_gemm2.py#L244)。

**资源**：LDS = max(64·I B 激活, 2 × 64 × 64 × 2 B 输出) = 20 KiB（I320）/ 16 KiB（I192）；
64·I 须 ≤ 64 KiB 且是 4096 的倍数（gather 每轮 256 线程 × 16 B），所以 BM32 配 I320 / 192 不可用。

## 行量化（[a8w4_quant.py](a8w4_quant.py#L23)）
把 bf16 输入 [M, H] 或 f32 中间 [M·topk, I] 按行量化成 fp8 与每行 scale，算术与参考逐位一致。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows/4⌉ | 4 行 |
| CTA（256 线程） | 4 个 wave | 每 wave 1 行 |
| lane | cols/64 个元素 | 列 j·64 + lane（H2560 为 40 个，I320 为 5 个） |

**流水线**
```text
读本 lane 的 cols/64 个元素 → f32；lane 内取 max|x|，再 6 次 shuffle_xor（32 … 1）得整行 amax
scale = amax / 240（IEEE 除法；全零行取 1）；lane 0 写 scale
q = clamp(x / scale, ±240)（IEEE 除法）→ cvt_pk_fp8_f32（RNE）→ -0（0x80）改 0 → 逐字节写
```
关键代码：[amax 规约](a8w4_quant.py#L46)、[scale](a8w4_quant.py#L47)、[量化](a8w4_quant.py#L52)。

**资源**：不用 LDS。
