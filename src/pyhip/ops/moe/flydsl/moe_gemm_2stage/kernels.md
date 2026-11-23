# MoE A8W4：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。 -->

A8W4 的 GEMM kernel（prefill 两个、decode 三个、流式两个）共用 [a8w4_common.py](a8w4_common.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；lane `16b + r` 读第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32 的操作数。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](a8w4_common.py#L59) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](a8w4_common.py#L70) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼（流式 kernel 用同样的两次 v_perm）。
- 解出的值是 |e2m1| × 2^(5−delta)，delta ≤ 11 时精确；通道因子 s = 2^(cmax−132)（f32）与激活 scale 在 epilogue 一起乘。
- decode kernel（[a8w4_decode.py](a8w4_decode.py)）每 lane 一条 16 B 加载，取布局里相邻两个 lane 的 8 B（行 2(l%8) 与 +1）；
  LUT 预先算好放在 LDS（delta 0–11 各 8 B，用 delta 字节直接寻址），两个 dword 的移位合成一次 64 位移位。

## 先读：Triton 风格的写法
每个 kernel 一节，先给一段 Triton 风格的函数：一个 program = 一个 CTA，只写它**算什么**（读哪些数据、怎么累加、写到哪里），
LDS、预取、指令排布等**怎么搬**留给后面的逐拍流水线。张量下标代替指针运算，省略 mask 与类型转换；参数用 host 侧的张量名。
共用的记号：
```python
# M：token 数；H：hidden；I：中间维（补零后）；E：expert 数；TOPK：每个 token 的路由数。路由 route = token·TOPK + slot。
# x / out：bf16 [M, H] 的输入与输出。xq / xs：x 按行量化后的 fp8 [M, H] 与 f32 scale [M]。
# mid：f32 [M, TOPK, I]（= SiLU(gate)·up）；mq / ms：mid 按行量化后的 fp8 [M·TOPK, I] 与 f32 [M·TOPK]。
# aiter moe_sorting 的输出（int32，权重 f32）：排序块 e = 同一 expert（sorted_expert_ids[e]）的 BM 个排序行；
#   sorted_ids[排序行] = token | slot << 24，token ≥ M 的是填充行（只在每个 expert 的末块尾部）；
#   sorted_weights[排序行] = 该路由的路由权重；num_valid = 排序行总数（含填充行）：grid 按排序块数的上界发射，
#   e·BM ≥ num_valid 的 CTA 直接退出；loc_ids[token, slot] = 该路由排序后的行号（< 0 表示没有这条路由，求和时跳过；
#   同一 token 重复选同一 expert 时，aiter 排序只保留最后一个槽，其余槽为 −1）。
def dequant(w, w_delta, expert, rows, k0, nk):  # 权重 → fp8 [len(rows), nk]：|e2m1| × 2^(5−delta) 带符号（上面的 LUT 解码）
def quant_row(v):                               # → (q, s)：s = max|v| / 240（为 0 时取 1：全零行或下溢）；q = RNE(clamp(v / s, ±240))，-0 存成 +0
def silu(g):                                    # g · sigmoid(g)，实现为 g · rcp(1 + exp2(−g·log2e))
# tl.dot：fp8 MFMA 16x16x32，f32 累加。w_scale[expert, ch] 是每个通道的因子 2^(cmax − 132)，在 epilogue 才乘。
# 行量化 kernel 都按 quant_row 把一行量化成 fp8，三者逐位一致：row_quant_fp8 通用（输入 bf16 或 f32），
#   row_quant_bf16 是流式档专用（输入 bf16 的 x），row_quant_decode 是 decode 档用的版本。
# sorted_sum / route_sum：out[t, :] = Σ_slot routes[loc_ids[t, slot], :]（f32 按槽顺序相加后转 bf16），不读填充行。
# 三条路径（host 见 moe_a8w4.py；给了 tile 参数时一律 prefill）：
#   decode（M ≤ 32 且 decode kernel 放得下；Gate/Up 按路由、Down 按 token 直接分 CTA，不排序，也就没有 sorted_sum）：
#     [row_quant_decode(x)] → gateup_decode → [row_quant_decode(mid)] → down_decode(_pq) → out
#   流式（M > 32 且 M·TOPK ≤ 22·E）：moe_sorting → row_quant_bf16(x) → gateup_stream（带尾部量化，或另跑 row_quant_fp8）
#     → down_stream → route_sum → out
#   prefill（其余，含 decode 放不下的 M ≤ 32）：moe_sorting → row_quant_fp8(x) → gateup_prefill → row_quant_fp8(mid)
#     → down_prefill → sorted_sum → out
```

## A8W4 Gate/Up prefill（[a8w4_gemm1.py](a8w4_gemm1.py#L426)）
prefill 档（M > 32 且每个 expert 平均超过 22 行、给了 tile 参数，或 decode 放不下的 M ≤ 32）按排序块算
mid[token, slot, :] = SiLU(x·w1_gate)·(x·w1_up)，f32 写出；结构同底座 `gemm1.py` 的 prefill_1x4，权重换成 mxfp4。

**Triton 风格**
```python
# grid = (2·I/BN, 排序块数；M·TOPK ≤ E 时取 M·TOPK)，256 线程；默认 BM=64、BN=128、TILE_K=128
# BN = 本 CTA 的 gate + up 通道总数，输出只有 BN/2 列（每列由同号的一个 gate、一个 up 通道得到）
@triton.jit
def a8w4_gateup_prefill(xq, xs, w1, w1_delta, w1_scale, sorted_ids, sorted_expert_ids, num_valid, mid):
    n_blk, e = tl.program_id(0), tl.program_id(1)    # n_blk：BN/2 个 gate 通道及同号 up 通道；e：排序块
    if e * BM >= num_valid: return                   # 多发的 CTA 直接退出
    ids = sorted_ids[e * BM + tl.arange(0, BM)]
    token, slot = ids & 0xFFFFFF, ids >> 24
    TILE_M = BM // 2 if token[BM // 2] >= M else BM  # 半块：后一半全是填充时只算前一半（两份特化）
    token, slot, expert = token[:TILE_M], slot[:TILE_M], sorted_expert_ids[e]
    ch = n_blk * (BN // 2) + tl.arange(0, BN // 2)
    g = tl.zeros([TILE_M, BN // 2], tl.float32)
    u = tl.zeros([TILE_M, BN // 2], tl.float32)
    for k in range(0, H, TILE_K):                    # 实际：x 经 LDS 双缓冲，权重提前 1 个 tile 读进 VGPR
        a = xq[token, k:k + TILE_K]                  # fp8 [TILE_M, TILE_K]，按 token gather
        g += tl.dot(a, dequant(w1, w1_delta, expert, ch, k, TILE_K).T)
        u += tl.dot(a, dequant(w1, w1_delta, expert, I + ch, k, TILE_K).T)
    g *= w1_scale[expert, ch] * xs[token][:, None]   # 通道因子 × 每个 token 的激活 scale
    u *= w1_scale[expert, I + ch] * xs[token][:, None]
    mid[token, slot, ch] = silu(g) * u               # f32；填充行（token ≥ M）的写被硬件丢弃
```

**任务划分**（默认 BM64、BN128、TILE_K128；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (2I/128, 排序块数) | x：64 gate + 同号 64 up 通道；y：1 个排序块 |
| CTA（256 线程） | 4 个 wave | 本块 TILE_M 个排序行（同一 expert）× 128 通道 × 全部 H |
| wave | 1 个 16 通道块 | gate、up 各 16 通道 × TILE_M 行（TILE_M/16 个 token 块） |
| lane | g = l/16、r = l%16 | 权重行 r 在每个 128-K 步里的第 g 个 32-K 组（两个半块各 8 B）；token r 的累加 |
| K 步 | 128-K 步（两个 64-K 半块） | 每个半块每 wave 4·TILE_M/16 条 MFMA 16x16x32 |

TILE_M = BM；排序块第 BM/2 行是填充时（有效行在块首）取 BM/2，两份特化由 readfirstlane 后的标量分支选。
`e·BM ≥ num_valid` 的 CTA 直接退出；M·topk ≤ E 时 grid 的 y 维取 M·topk（每条路由至多占一个排序块）。
K 置换：A 的 LDS 读视图把步内 MFMA 侧的 K 64h + 16g + 8c + v 映射到存储的 32g + 16h + 8c + v，权重按同一置换取址，
所以 lane 组 g 在步内两个半块都用同一个 32-K 组：每步只用一个 delta 字节（第 g/2 + 2·(步号%2) 个）建一次 LUT。

**流水线**（逐拍；x 的 LDS 双缓冲 A_ping / A_pong，权重 VGPR 两槽 0 / 1，预取距离 1 个 tile）
```text
序言
  tid < BM：sorted_ids[e·BM + tid] → LDS；barrier；各线程取 epilogue 的 (token, slot)
  LDS 第 BM/2 行（readfirstlane）是填充 → 以下 TILE_M = BM/2，否则 TILE_M = BM
  各线程从 LDS 取 x 的 gather 行号；barrier（LDS 让给 A_ping）
  预取 tile 0：gather x（每线程 TILE_M·TILE_K/4096 条 16 B buffer load）→ VGPR；本 wave 权重 → 槽 0
  s_waitcnt vmcnt(0)；x → A_ping；解码 tile 0 的第 0 个 64-K 块；barrier
主循环 i = 0 … K/(2·TILE_K) − 2（每次两拍；H2560、TILE_K128 为 9 次）
  拍 0（读 A_ping，算 tile 2i）：
    发预取 tile 2i+1：gather x → VGPR；权重 → 槽 1（每半块 1 条 dwordx2，每 tile 1 条 delta dword）
    for ki in 0 … TILE_K/64 − 1：
      A_ping 第 ki 块按置换视图读成 B 操作数（TILE_M/16 条 16 B LDS 读）
      ki > 0：解码第 ki 块（ki 为偶数时由 delta 字节建 LUT，奇数块复用）
      MFMA 16x16x32 fp8 × 4·TILE_M/16（gate / up × token 块 × 2 个 K32）
    解码槽 1 的第 0 块（下一 tile）；预取的 x：VGPR → A_pong；barrier
    指令排布（sched_group）：预取的 vmem 随前几条 MFMA 发出；下一块的 LDS 读与解码 VALU 均摊在当前块的
      MFMA 之间（最后一块摊的是下一 tile 第 0 块的解码）；A_pong 的 LDS 写排在最后几条 MFMA 之间
  拍 1（读 A_pong，算 tile 2i+1）：同上，预取 tile 2i+2 → 槽 0 与 A_ping
收尾：拍 0（倒数第二个 tile，预取最后一个）；拍 1（最后一个 tile，不预取、不解码下一块）
epilogue（累加器留在 VGPR，不过 LDS）
  acc ×= 通道因子 s[ch]（gate、up 各自）× xs[token]
  SiLU(gate)·up = gate · rcp(1 + exp2(−gate·log2e)) · up
  每个 token 块 4 个 f32 一次 16 B buffer store → mid[token, slot, ch … ch+3]
  填充行（token ≥ M）的偏移改成越界值，store 被硬件丢弃
```
关键代码：[排序行号进 LDS](a8w4_gemm1.py#L463)、[半块分支](a8w4_gemm1.py#L487)、[K 置换读视图](a8w4_gemm1.py#L171)、
[lane 取址](a8w4_gemm1.py#L220)、[权重加载](a8w4_gemm1.py#L254)、[解码与 LUT 复用](a8w4_gemm1.py#L268)、
[指令排布](a8w4_gemm1.py#L294)、[一拍](a8w4_gemm1.py#L320)、[序言](a8w4_gemm1.py#L357)、[主循环](a8w4_gemm1.py#L366)、
[反量化](a8w4_gemm1.py#L380)、[SiLU 与写出](a8w4_gemm1.py#L65)。

**资源**：LDS 2·BM·TILE_K（A_ping / A_pong，排序行号的 1 KiB 与之复用）：BM64 时 TILE_K128 为 16 KiB、TILE_K256 为 32 KiB；
TILE_K128 的 LDS 与权重寄存器都减半、占用率更高，所以按行数选 prefill 时默认用它。

## A8W4 Down prefill（[a8w4_gemm2.py](a8w4_gemm2.py#L59)）
prefill 档按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同底座 `gemm2.py` 的默认 prefill Down。

**Triton 风格**
```python
# grid = (1, 排序块数)，256 线程；默认 BM=64
@triton.jit
def a8w4_down_prefill(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                      routes):
    e = tl.program_id(1)                             # 一个排序块 × 全部 H 列
    if e * BM >= num_valid: return
    ids = sorted_ids[e * BM + tl.arange(0, BM)]
    bm = BM // 2 if (ids[BM // 2] & 0xFFFFFF) >= M else BM   # 半块，同 Gate/Up
    rows = e * BM + tl.arange(0, bm)                 # 本块的排序行号
    route = (ids[:bm] & 0xFFFFFF) * TOPK + (ids[:bm] >> 24)
    expert = sorted_expert_ids[e]
    a = mq[route, :]                                 # fp8 [bm, I]：先 gather 进 LDS，再整块留在 VGPR
    row_scale = ms[route] * sorted_weights[rows]     # 中间激活 scale × 路由权重
    for n in range(0, H, 64):                        # 实际：权重提前 1 块读进 VGPR，结果经 LDS 双缓冲合并写出
        c = tl.dot(a, dequant(w2, w2_delta, expert, n + tl.arange(0, 64), 0, I).T)   # [bm, 64] f32
        c *= w2_scale[expert, n:n + 64] * row_scale[:, None]
        routes[rows, n:n + 64] = c.to(tl.bfloat16)   # RNE；填充行也写（值无意义），之后 sorted_sum 只按 loc_ids 读有效行
```

**任务划分**（默认 BM64；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (1, 排序块数) | 1 个排序块 × 全部 H 列 |
| CTA（256 线程） | 4 个 wave | bm 行 mq 整块驻留；逐个 64 列块走完 H |
| wave | 每个 64 列块中的 16 列 | 列 n·64 + 16w … +15 × bm 行 |
| lane | l = 16b + r | 权重列 r 的 K 16b…16b+15；B 操作数的 1/64 |
| K 步 | I/64 个 64-K 块 × 2 个 K32 | 每个列块一次算完全部 K（I320 为 10 步） |

bm = BM；排序块第 BM/2 行是填充时取 BM/2（同 Gate/Up，两份特化各用一份构建缓存）。退出与 grid 收缩同 Gate/Up。H2560 共 40 个列块。

**流水线**（逐拍；权重 VGPR 两槽 0 / 1，预取距离 1 个列块；输出 LDS 双缓冲 C[0] / C[1]）
```text
序言
  全 CTA 按 sorted id gather mq 的 BM 行（每次 16 B，共 BM·I/4096 轮）→ LDS A；barrier
  sorted_ids[BM/2]（readfirstlane）是填充 → 以下 bm = BM/2，否则 bm = BM
  LDS A 的前 bm 行 → VGPR（bm × I 整块作 B 操作数）；barrier（之后 LDS 改作输出缓冲 C）
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
  for kb：MFMA 16x16x32 fp8 × bm/8（bm/16 个 token 块 × 2 个 K32）；结果 × 通道因子
→ bf16：f32 位加 0x7FFF 与第 16 位（RNE，输入须有限），一条 v_perm 取两个值的高半字拼成一个字
C → routes：bf16 结果经 LDS 转成按行连续，再 16 B buffer store 合并写出 bm 行
```
关键代码：[gather](a8w4_gemm2.py#L120)、[半块分支](a8w4_gemm2.py#L281)、[激活进 VGPR](a8w4_gemm2.py#L163)、
[行 scale](a8w4_gemm2.py#L177)、[load_block](a8w4_gemm2.py#L200)、[compute](a8w4_gemm2.py#L217)、[bf16 打包](a8w4_gemm2.py#L46)、
[写 LDS](a8w4_gemm2.py#L240)、[写 routes](a8w4_gemm2.py#L251)、[主循环](a8w4_gemm2.py#L260)。

**资源**：LDS = max(64·I B 激活, 2 × 64 × 64 × 2 B 输出) = 20 KiB（I320）/ 16 KiB（I192）；
64·I 须 ≤ 64 KiB 且是 4096 的倍数（gather 每轮 256 线程 × 16 B），所以 BM32 配 I320 / 192 不可用，I > 1024 时整个 prefill 不可用。

## 行量化 prefill（[a8w4_quant.py](a8w4_quant.py#L64)）
prefill 档把 bf16 输入 [M, H] 与 f32 中间 [M·topk, I]、流式档把不在 Gate/Up 尾部量化的 f32 中间按行量化成 fp8 与每行 scale，
结果与参考（IEEE 除法）逐位一致。

**Triton 风格**
```python
# grid = ⌈rows / (256/L)⌉，256 线程：每行 L 个相邻 lane（L 见任务划分）
@triton.jit
def a8w4_row_quant_fp8(x, q, scale, rows):           # x：bf16 [M, H]（输入）或 f32 [M·TOPK, I]（mid）
    r = tl.program_id(0) * (256 // L) + tl.arange(0, 256 // L)   # 行号 ≥ rows 的读到 0、不写
    q[r, :], scale[r] = quant_row(x[r, :])           # 商 = 倒数乘 + 两次 FMA 修正（scale < 2^-64 时逐元素除法），
                                                     # 与 IEEE 的 x / scale 逐位一致
```

**任务划分**（L 取 64、32、16、8、4 中第一个整除 cols、且每 lane 恰为整数个 16 B 的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows / (256/L)⌉ | 256/L 行 |
| CTA（256 线程） | 4 个 wave | 每行 L 个相邻 lane |
| lane | 本行连续 cols/L 个元素 | H2560 bf16：L64、40 个；I320 / I192 f32：L16、20 / 12 个 |

**流水线**
```text
lane 连续读 cols/L 个元素（每次 16 B buffer load；rows 以外的行读到 0，写被丢弃）
lane 内 max|x| → log2(L) 次 shuffle_xor 得整行 amax；scale = amax / 240（IEEE；为 0 时取 1）；每行首个 lane 写 scale
scale ≥ 2^-64：y = 1/scale（IEEE，每行一次）；q = x·y，再两次 q += (x − q·scale)·y（FMA），得正确舍入的商
scale < 2^-64（残差可能下溢）：q = x / scale，逐元素 IEEE 除法
clamp ±240 → 每 4 个 cvt_pk_fp8（RNE）成一个字 → 0x80（-0）改 0 → 字数为偶数时每 2 个字一次 8 B 写，否则 4 B 写
```
关键代码：[lane 数](a8w4_quant.py#L33)、[16 B 读](a8w4_quant.py#L89)、[amax 与 scale](a8w4_quant.py#L101)、
[倒数与修正](a8w4_quant.py#L125)、[打包与 -0](a8w4_quant.py#L54)、[写回](a8w4_quant.py#L111)。

**资源**：不用 LDS。

## A8W4 Gate/Up decode（[a8w4_decode.py](a8w4_decode.py#L197)）
M ≤ 32 且形状受支持（H ≤ 4096，即 H/16 个 16 B 片不超过 256 个线程；见 [gateup_decode_fits](a8w4_decode.py#L54)）时不排序：
每个 WG 算一条路由 (token, slot) 的 16 个 gate 与同号 16 个 up 通道，写 mid[token, slot, ch]（f32）。

**Triton 风格**
```python
# grid = (I/16, M·TOPK)，256 线程（4 个 wave）；不排序
@triton.jit
def a8w4_gateup_decode(x, xq, xs, topk_ids, w1, w1_delta, w1_scale, mid):
    j, route = tl.program_id(0), tl.program_id(1)    # j：16 个 gate 通道及同号 up 通道；route：一条路由
    token, slot, expert = route // TOPK, route % TOPK, topk_ids[route]
    if PREQUANT:                                     # 编译期开关：M·TOPK·I/16（CTA 数）≥ 2560 时为真，
        a, s = xq[token, :], xs[token]               #   host 先用 row_quant_decode 量化 x，这里读 fp8 [H] 与 f32
    else:                                            # CTA 少时每个 CTA 自己量化 bf16 的 x 行（4 个 wave 求 max，经 LDS），
        a, s = quant_row(x[token, :])                #   省掉一次 launch
    ch = 16 * j + tl.arange(0, 16)
    g = tl.sum(dequant(w1, w1_delta, expert, ch, 0, H) * a[None, :], axis=1)       # [16] f32
    u = tl.sum(dequant(w1, w1_delta, expert, I + ch, 0, H) * a[None, :], axis=1)
    g *= w1_scale[expert, ch] * s
    u *= w1_scale[expert, I + ch] * s
    mid[token, slot, ch] = silu(g) * u               # f32
    # 实际：wave w 算 gate（w 偶）或 up（w 奇）的 K 前半（w < 2）或后半，4 个 wave 的部分和在 LDS 相加；
    #       这个 gemv 用 MFMA 16x16x32 做（A 的 16 行都是同一 token 的 x，结果怎么取见下面的说明与代码 docstring
    #       “Block pairs”）。本 wave 的全部权重加载在序言一次发出。
```

**任务划分**（数字按 H2560、I320）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·topk) | x：16 个 gate 通道块 j 与同号 up 块；y：一条路由 |
| CTA（256 线程） | 4 个 wave | 该路由 expert 的 32 行权重 × 全部 H |
| wave | w%2：gate / up；w/2：K 前半 / 后半 | 16 行 × H/2（10 个块对） |
| lane | l | 块对中第 l/32 块的行 2(l%8)、2(l%8)+1，K 组 (l%32)/8，共 16 B |
| K 步 | 块对（相邻两个 64-K 块） | 每 wave 4 条 MFMA 16x16x32（2 行 × 2 个 K32） |

A 操作数的 16 行都是同一 token 的 x；输出取行 0 的列 0–7 与行 8 的列 8–15（各含一半 K 组），
列 m 与 m+8 相加得行 2m+h 的全 K（映射见代码 docstring 的 “Block pairs”）。

**流水线**（不分 tile：序言一次发出本 wave 的全部加载，主循环按序消费；无双缓冲）
```text
序言
  tid ≤ 11：LUT 表 → LDS lut（delta d 在字节 8d：|e2m1| 0–3、4–7 的 fp8 各一个 dword）
  expert = topk_ids[route]；本 wave 的 16 行块 rb 与 K 起点 kb0
  发 x 行加载：预量化版 tid < H/16 各读 16 B fp8；核内量化版每线程读 H/512 个 bf16 对
  for p in 块对（sched_barrier 固定顺序）：p 偶时 delta dwordx2（两行 × 4 块）；权重 dwordx4
  tid < 16：通道因子 ws[gate][ch]、ws[up][ch]
x 行 → LDS xq
  预量化版：16 B → LDS；barrier；scale = x_scale[token]
  核内量化版：lane 内 max|x| → DPP 4 步 + readlane 4 行 → LDS red[wave]；barrier
              scale = max(red) / 240（IEEE，0 取 1）；q = RNE(clamp(x/scale)) → 每线程 2 B → LDS；barrier
主循环 p = 0 … H/256 − 1（第 p 个块对只等到自己的加载）
  xop = LDS xq 第 kb0+2p+l/32 块、K 组 (l%32)/8 的 16 B（2 个 K32 的 A 操作数）
  每行 h：LUT 对 = LDS lut[delta 字节]；fp4 经 64 位移位 + v_perm 查表 → 2 个 fp8 i64
  MFMA 16x16x32 × 4：acc[h] += xop[c] · w[h][c]（c = 0, 1）
收尾
  lane 0–7（行 0 列 0–7）、lane 40–47（行 8 列 8–15）：acc[h][0] → LDS part[wave, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8；gate = wave 0 + wave 2，up = wave 1 + wave 3
  × 通道因子 × scale → gate · rcp(1 + exp2(−gate·log2e)) · up → f32 写 mid[route, j·16 + tid]
```
关键代码：[LUT 表](a8w4_decode.py#L141)、[x 行加载](a8w4_decode.py#L257)、[权重加载](a8w4_decode.py#L280)、
[核内量化](a8w4_decode.py#L301)、[主循环](a8w4_decode.py#L320)、[块解码](a8w4_decode.py#L168)、[收尾](a8w4_decode.py#L335)。

**资源**：LDS 约 3.2 KiB（xq H B、part 512 B、red 16 B、LUT 96 B）；本 wave 的权重全部同时在途（H2560 每 lane 10 条 16 B）。

## A8W4 Down decode（核内量化，[a8w4_decode.py](a8w4_decode.py#L374)）
M ≤ 3、w2 ≥ 4 GiB 或槽对版放不下时用：每个 WG 算一个 token 的 16 个输出通道，在 LDS 里把 topk 个 expert 相加后直接写 bf16 out。
它的 LDS 放不下时（[down_decode_fits](a8w4_decode.py#L60)）decode 整体不用，M ≤ 32 走 prefill。

**Triton 风格**
```python
# grid = (H/16, M)，64·W 线程：W = TOPK 个 wave（TOPK > 16 时 W = 4）
@triton.jit
def a8w4_down_decode(mid, topk_ids, topk_weight, w2, w2_delta, w2_scale, out):
    j, token = tl.program_id(0), tl.program_id(1)    # j：16 个输出通道
    n = 16 * j + tl.arange(0, 16)
    total = tl.zeros([16], tl.float32)
    for slot in range(TOPK):                         # 实际：wave w 管槽 w, w+W, …，各 wave 并行
        expert = topk_ids[token, slot]
        a, s = quant_row(mid[token, slot, :])        # 本槽的中间激活在 wave 内按行量化成 fp8
        acc = tl.sum(dequant(w2, w2_delta, expert, n, 0, I) * a[None, :], axis=1)
        total += acc * w2_scale[expert, n] * s * topk_weight[token, slot]
    out[token, n] = total.to(tl.bfloat16)            # 各 wave 的部分和在 LDS 树形相加后 RTE；
                                                     # 直接写 out，不需要清零，也没有 sorted_sum
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/16, M) | x：16 个输出通道块 j；y：一个 token |
| CTA（64·W 线程） | W = topk 个 wave（topk > 16 时 W = 4） | 该 token 的全部 topk 个槽 |
| wave | 槽 w, w+W, …（越界槽钳到 topk−1、贡献置零） | 该槽 expert 的 16 行 × 全部 I |
| lane | l = 16b + r | 行 r 的 K 组 b（每块 8 B，同 prefill）；mid 列 l + 64u |
| K 步 | 64-K 块 | 2 条 MFMA 16x16x32（I320 为 5 块） |

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；每轮 r：slot = w + W·r、expert、路由权重
  发 mid 行加载：lane 读 I/64 个 f32（列 l + 64u）
  发权重加载：每 64-K 块 1 条 dwordx2，每 4 块 1 条 delta dword（排在其前）；通道因子 ws[ch]
wave 内量化本槽 mid 行
  amax：lane 内 → DPP + readlane；scale = amax / 240（IEEE，0 取 1）
  q = RNE(clamp(mid/scale)) → 逐字节写 LDS mq[w, r]；barrier
主循环 r、kb = 0 … I/64 − 1
  xop = LDS mq[w, r] 第 kb 块、K 组 l/16 的 16 B；LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64
  MFMA 16x16x32 × 2 → acc；行 0（lane < 16 的 acc[0]）即本槽的 16 个通道
  total += 有效 ? acc[0] × 通道因子 × scale × 路由权重 : 0
收尾
  lane < 16：total → LDS part[w]；barrier
  tid < 16：W 个 wave 的部分和树形相加 → bf16（RTE）→ out[token, j·16 + tid]（不需清零与 sorted_sum）
```
关键代码：[槽与 expert](a8w4_decode.py#L429)、[加载](a8w4_decode.py#L439)、[wave 内量化](a8w4_decode.py#L463)、
[主循环](a8w4_decode.py#L478)、[收尾](a8w4_decode.py#L489)。

**资源**：LDS = W·⌈topk/W⌉·I B（mq）+ 64·W B（part）+ 96 B，TP2 topk11 约 4.2 KiB。

## A8W4 Down decode 槽对（预量化，[a8w4_decode.py](a8w4_decode.py#L521)）
M ≥ 4、w2 < 4 GiB 且放得下（每行 I/16 个片 ≤ 64·W 个 lane，LDS 够）时用：读预量化的 mq fp8 与 ms；
一条 16 B 加载覆盖两个路由槽的同一 64-K 块，其余同核内量化版。

**Triton 风格**
```python
# grid = (H/16, M)，64·W 线程：W = min(⌈TOPK/2⌉, 3) 个 wave，按轮取槽对
@triton.jit
def a8w4_down_decode_pq(mq, ms, topk_ids, topk_weight, w2, w2_delta, w2_scale, out):
    j, token = tl.program_id(0), tl.program_id(1)
    n = 16 * j + tl.arange(0, 16)
    total = tl.zeros([16], tl.float32)
    for p in range((TOPK + 1) // 2):                 # 槽对 (2p, 2p+1)；实际低槽、高槽各占半个 wave 的 lane（0–31 / 32–63），
        for slot in (2 * p, 2 * p + 1):              # 所以一条 16 B 加载、一组 MFMA 同时推进两个槽；slot ≥ TOPK 的贡献置零
            route, expert = token * TOPK + slot, topk_ids[token, slot]
            acc = tl.sum(dequant(w2, w2_delta, expert, n, 0, I) * mq[route, :][None, :], axis=1)
            total += acc * w2_scale[expert, n] * ms[route] * topk_weight[token, slot]
    out[token, n] = total.to(tl.bfloat16)            # 同 a8w4_down_decode
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/16, M) | 同核内量化版 |
| CTA（64·W 线程） | W = min(⌈topk/2⌉, 3)；topk11 为 3 wave × 2 轮 | 该 token 的全部槽对 |
| wave | 槽对 p = w + W·r：低槽 2p、高槽 2p+1 | 越界槽钳到 topk−1、贡献置零 |
| lane | l：0–31 低槽、32–63 高槽 | 本槽块的行 2(l%8)、2(l%8)+1，K 组 (l%32)/8，共 16 B |
| K 步 | 64-K 块 | 4 条 MFMA 16x16x32（2 行 × 2 个 K32） |

A 操作数：l%2 == l/32 的 lane 取本槽本 K 组的激活，其余取 LDS 的全零行；
同一 MFMA 的输出行 0/8 只含低槽、行 1/9 只含高槽（两槽的通道因子不同，须分开乘；见 docstring “Slot pairs”）。

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；tid < I/16：LDS 末尾写一行全零（被屏蔽 lane 的 A 操作数）
  每轮 r、半 s：slot、expert（readfirstlane 成标量）、路由权重、ms[token, slot]
  发 mq 行加载：lane 读本半槽的第 l%32 + 32u 个 16 B 片（越界片钳到最后一片）
  发权重加载：整张量 buffer，每 lane 偏移 = 本半槽 expert × 每 expert 字节数（32 位）
    每 64-K 块 1 条 dwordx4；每 4 块 1 条 delta dwordx2；两槽的通道因子各 1 条 dwordx2
mq 片 → LDS 行 [w, r, 半]（只写有效片）；barrier
主循环 r、kb = 0 … I/64 − 1
  xop = 保留的 lane ? LDS 本槽行第 kb 块、K 组 (l%32)/8 的 16 B : 零行
  每行 h：LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64；MFMA × 4：acc[h] += xop[c] · w[h][c]
  vals[h] += Σ 有效槽 s：acc[h][s] × 通道因子[s][h] × ms[s] × 路由权重[s]（[0] 低槽、[1] 高槽）
收尾
  lane 0–7、40–47：vals[h] → LDS part[w, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8 → 树形相加 → bf16（RTE）→ out[token, j·16 + tid]
```
关键代码：[零行](a8w4_decode.py#L571)、[槽对](a8w4_decode.py#L580)、[mq 片](a8w4_decode.py#L592)、
[整张量寻址](a8w4_decode.py#L604)、[屏蔽 A 操作数](a8w4_decode.py#L645)、[主循环](a8w4_decode.py#L647)、[收尾](a8w4_decode.py#L668)。

**资源**：LDS = (2·W·轮数 + 1)·I B + 128·W B + 96 B，TP2 topk11 约 4.5 KiB；
每 lane 偏移是 32 位，w2 总字节数须 < 4 GiB（否则 host 改用核内量化版）。

## 行量化 decode（[a8w4_decode.py](a8w4_decode.py#L706)）
槽对 Down 前量化 mid（f32 [M·topk, I]），M·topk·I/16 ≥ 2560 时量化 x（bf16 [M, H]）；逐元素 IEEE 除法，结果与 [a8w4_quant.py](a8w4_quant.py) 逐位一致。

**Triton 风格**
```python
# grid = ⌈rows/4⌉，256 线程：每个 wave 一行
@triton.jit
def a8w4_row_quant_decode(x, q, scale, rows):        # x：bf16 [M, H] 或 f32 mid [M·TOPK, I]
    r = 4 * tl.program_id(0) + tl.arange(0, 4)       # 行号 ≥ rows 的 wave 不读不写
    q[r, :], scale[r] = quant_row(x[r, :])           # 逐元素 IEEE 除法，与 a8w4_row_quant_fp8 逐位一致
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows/4⌉ | 4 行 |
| CTA（256 线程） | 4 个 wave | 每 wave 1 行（rows 以外的 wave 不读不写） |
| lane | 16 B 片 l + 64m | bf16 8 个、f32 4 个元素 |

**流水线**
```text
lane 读 16 B 片 l + 64m（越过行尾的 lane 重读最后一片，不改 amax）
lane 内 max|x| → DPP 4 步 + readlane 4 行得整行 amax；scale = amax / 240（IEEE，0 取 1）
q = clamp(x / scale, ±240)（IEEE）→ cvt_pk_fp8（RNE）→ -0 改 +0 → 每 4 个打包成 32 位字
有效片写回：bf16 每片 8 B、f32 每片 4 B；lane 0 写 scale
```
关键代码：[读片](a8w4_decode.py#L726)、[scale](a8w4_decode.py#L744)、[写回](a8w4_decode.py#L746)、[-0 处理](a8w4_decode.py#L124)。

**资源**：不用 LDS。

## A8W4 Gate/Up 流式（[a8w4_stream.py](a8w4_stream.py#L262)）
M > 32 且每个 expert 平均不超过 22 行时按排序块（同一 expert 的 16 / 32 行；64 行只在调大 `STREAM_MAX_ROWS` 时用）
算 SiLU(x·w1_gate)·(x·w1_up)：权重与激活直接从 global 读进 VGPR，不经 LDS 双缓冲；写 f32 mid[token, slot, :]，
或在尾部量化成 fp8 mq[route, :] 与 ms[route]。

**Triton 风格**
```python
# grid = (I/(16·G·W), 排序块数)，64·W 线程；排序块 BM = 16 / 32，G、W、P 见任务划分与流水线
@triton.jit
def a8w4_gateup_stream(xq, xs, w1, w1_delta, w1_scale, sorted_ids, sorted_expert_ids, num_valid, mid, mq, ms):
    c, e = tl.program_id(0), tl.program_id(1)        # c：W·G 个 16 通道 gate 块及同号 up 块；e：排序块
    ids = sorted_ids[e * BM + tl.arange(0, BM)]
    S = valid_subblocks(ids, e, num_valid, M)        # 首行是真实行（token < M）的 16 行子块数；有效行在块首，所以是前缀；
                                                     #   e·BM ≥ num_valid 时为 0。kernel 为 S = 1…BM/16 各编一份代码，运行时
                                                     #   按 S 分支进入其中一份，S = 0 直接退出
    token, slot = ids[:16 * S] & 0xFFFFFF, ids[:16 * S] >> 24
    expert = sorted_expert_ids[e]
    ch = c * (16 * G * W) + tl.arange(0, 16 * G * W)
    g = tl.zeros([16 * S, 16 * G * W], tl.float32)
    u = tl.zeros([16 * S, 16 * G * W], tl.float32)
    for k in range(0, H, 64):                        # 权重直接 global → VGPR，提前 P 个 64-K 步发出；
        a = xq[token, k:k + 64]                      # 激活也直接进 VGPR，只有 W > 1 且 G ≤ 4 时经 LDS 两槽环在 wave 间共享
                                                     #   （没有 prefill 那种整块双缓冲）
        g += tl.dot(a, dequant(w1, w1_delta, expert, ch, k, 64).T)
        u += tl.dot(a, dequant(w1, w1_delta, expert, I + ch, k, 64).T)
    v = silu(g * w1_scale[expert, ch] * xs[token][:, None]) * (u * w1_scale[expert, I + ch] * xs[token][:, None])
    if QUANT:                                        # BM > 16 且累加器放得下（8·(I/64)·(BM/16) ≤ 96 个 VGPR）时分派打开：
                                                     #   G = I/64、W = 4，一个 CTA 恰好算完一行的全部 I 个通道
        mq[token * TOPK + slot, :], ms[token * TOPK + slot] = quant_row(v)
    else:
        mid[token, slot, ch] = v                     # f32；host 随后用 row_quant_fp8 量化。填充行的写都被丢弃
```

**任务划分**（G、W 由分派选：排序块 16 为 G2·W2；尾部量化时 G = I/64、W = 4，一个 WG 管全部 I；其余 G4·W1）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/(16·G·W), 排序块数) | x：W·G 个 gate 16 行块与同号 up 块；y：1 个排序块 |
| CTA（64·W 线程） | W 个 wave | 本块 expert 的 2·G·W 个 16 行权重块 × 全部 H |
| wave | G 个 gate 块 + 同号 G 个 up 块 | 本块的 S 个有效 16 行子块（S = 1 … BM/16） |
| lane | l = 16b + t | 权重行 t 的 K 16b…16b+15（每 64-K 8 B）；子块第 t 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每个权重块：1 次 LUT、2 个 dword 解码，每子块 2 条 MFMA 16x16x32 |

MFMA 的 A 是权重（16 个通道行），B 是 16 个排序行；输出 lane l 持有通道 4(l/16) … +3、排序行 l%16。
M·topk ≤ E 时 grid 的 y 维取 M·topk；排序块里有效行在前，所以有效子块是前缀，kernel 按 S 生成 BM/16 份特化。

**流水线**（预取距离 P 个 64-K 步：S = 1 时 PREFETCH[0]，否则 PREFETCH[1]；W>1 且 G≤4 时激活经 LDS 两槽环共享）
```text
序言（一轮访存）：num_valid、本块 expert、本 lane 各子块的 sorted id → token / slot
  S = 首行有效的子块数（越过 num_valid 为 0）；进入第 S 份特化体，S = 0 直接退出
发 kb = 0 … P−1：激活（共享时每 lane 读 BM·64/(64·W) B，否则每子块 16 B）→ 2G 个权重块各 8 B → 每 4 步 delta 1 dword
共享：激活(0) → LDS 槽 0；lgkmcnt(0) + barrier
主循环 kb = 0 … H/64 − 1：
  发 kb+P 的激活与权重（先激活后权重，等激活时不必等后发的权重）
  共享：激活(kb+1) → LDS 槽 (kb+1)%2；从槽 kb%2 读本 lane 的 16 B 作 B（行内 16 B 片按 (行号>>1)&3 异或错开）
  不共享：B = 寄存器里的激活(kb)
  for 2G 个权重块：LUT（delta 第 kb%4 字节）→ 2 个 dword 解成 2 个 fp8 i64 → 每子块 MFMA ×2
  共享：lgkmcnt(0) + barrier（槽 kb%2 读完才会被 kb+2 覆盖）；sched_barrier
收尾
  acc × 通道因子（gate、up 各自）× xs[token] → g · rcp(1 + exp2(−g·log2e)) · u
  不量化：每子块 4 个 f32 一次 16 B store → mid[token, slot, ch … ch+3]（填充行偏移越界，store 被丢弃）
  量化：行 amax（lane 内 4G 个 → xor 16、32）→ LDS [子块][行][wave]；barrier；取 W 个 wave 的最大值
        scale = amax/240（0 取 1）；q = v·rcp(scale) 在 ×(1∓2^-20) 两端 RNE 一致即用，否则整 wave 回退 IEEE 除法
        每 4 个 fp8 一次 4 B store → mq[route, ch … ch+3]；wave 0 的 lane < 16 写 ms[route]
```
关键代码：[前导](a8w4_stream.py#L241)、[按 S 特化](a8w4_stream.py#L466)、[共享激活](a8w4_stream.py#L367)、
[预取](a8w4_stream.py#L404)、[主循环](a8w4_stream.py#L412)、[SiLU](a8w4_stream.py#L292)、[尾部量化](a8w4_stream.py#L324)、
[夹逼与回退](a8w4_stream.py#L150)。

**资源**：LDS = 共享时 2·BM·64 B 激活环（BM16 2 KiB、BM64 8 KiB）+ 量化时 BM·W·4 B amax；
累加器 8·G·S 个 VGPR，分派只在它 ≤ 96 时让一个 WG 管全部 I 并尾部量化（TP2 BM32 为 80，BM64 为 160 故不量化）。

## A8W4 Down 流式（[a8w4_stream.py](a8w4_stream.py#L609)）
M > 32 且每个 expert 平均不超过 22 行时按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 route_sum。

**Triton 风格**
```python
# grid = (H/(16·R·W), 排序块数)，64·W 线程（R8·W4 或 R16·W1，见任务划分）
@triton.jit
def a8w4_down_stream(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                     routes):
    c, e = tl.program_id(0), tl.program_id(1)        # c：W·R 个 16 通道输出块；e：排序块
    ids = sorted_ids[e * BM + tl.arange(0, BM)]
    S = valid_subblocks(ids, e, num_valid, M)        # 同 a8w4_gateup_stream
    rows = e * BM + tl.arange(0, 16 * S)             # 本块的排序行号
    route = (ids[:16 * S] & 0xFFFFFF) * TOPK + (ids[:16 * S] >> 24)
    expert = sorted_expert_ids[e]
    a = mq[route, :]                                 # fp8 [16·S, I]，整块留在 VGPR（R8·W4 时先经 LDS 在 wave 间共享）
    row_scale = ms[route] * sorted_weights[rows]     # 中间激活 scale × 路由权重
    for r in range(R * W):                           # 实际：每个 wave 依次算 R 块，权重提前 min(4, R) 块发出
        n = (c * R * W + r) * 16 + tl.arange(0, 16)
        acc = tl.dot(a, dequant(w2, w2_delta, expert, n, 0, I).T)            # [16·S, 16]，实际分两条累加链
        routes[rows, n] = (acc * w2_scale[expert, n] * row_scale[:, None]).to(tl.bfloat16)   # 之后 route_sum
```

**任务划分**（R、W 由分派选：排序块 ≥ 32 且共享激活行放得进 64 KiB LDS 时 R8·W4，否则 R16·W1）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/(16·R·W), 排序块数) | x：W·R 个 16 通道输出块；y：1 个排序块 |
| CTA（64·W 线程） | W 个 wave | 共享时一起把本块 S·16 行 mq 装入 LDS |
| wave | R 个 16 通道块 | 每块全部 I；本块激活 S × I/64 × 16 B 驻留 VGPR |
| lane | l = 16b + t | 权重行 t 的 K 16b…16b+15；子块第 t 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每子块 2 条 MFMA，分属两条累加链（I320 为 5 步） |

**流水线**（预取距离 P = min(4, R) 个通道块）
```text
序言：前导同 Gate/Up；进入第 S 份特化体，S = 0 退出
共享：本块 S·16 行 mq（每行 I B）切成 16 B 片分给全 WG 的 lane 发加载（先于权重发出）
通道因子：本 wave R·16 个 f32，一条 16 B 加载（lane·16 < R·64）
发前 P 个通道块的权重：每块 I/64 × 8 B + delta（一条 dword 取两组：奇数 16-lane 组取奇数组）
通道因子 → LDS 本 wave 的 1 KiB 区
共享：激活片 → LDS（行距 I+16，16 行的 16 B 读落在不同 bank）；barrier；每子块读 I/64 个 16 B → VGPR
不共享：每子块直接从 global 读 I/64 个 16 B → VGPR
行 scale = ms[route] × sorted_weights[排序行]
for r in 0 … R−1：
  发 r+P 块的权重；r+1 块的 delta 经 xor-16 交换拼出两组（提前一块，藏住交换延迟）
  for kb：LUT → 2 个 dword 解码 → 每子块 2 条 MFMA，分别累加到链 0 / 链 1
  两链相加 × 通道因子（LDS）× 行 scale → 两两 RNE 打包成 bf16 → 8 B store → routes[排序行, ch … ch+3]（填充行丢弃）
```
关键代码：[共享激活](a8w4_stream.py#L674)、[通道因子](a8w4_stream.py#L693)、[权重与 delta](a8w4_stream.py#L655)、
[delta 交换](a8w4_stream.py#L662)、[主循环](a8w4_stream.py#L715)、[写 routes](a8w4_stream.py#L738)、[LDS 预算](a8w4_stream.py#L603)。

**资源**：LDS = W·1 KiB 通道因子 + 共享时 BM·(I+16) B 激活 + W·1 KiB 越界片槽（构建时断言 ≤ 64 KiB；TP2 BM32 为 18.5 KiB，
I1024 BM64 要 73 KiB 所以不共享）；激活占 S·I/64·4 个 VGPR（BM64、I320 为 80）。

## 流式路径的输入量化与 route_sum（[a8w4_stream.py](a8w4_stream.py#L499)）
流式档代替 a8w4_quant 的输入量化与底座 `sorted_sum`；结果与它们相同（量化逐位一致，求和同顺序同舍入）。

**Triton 风格**
```python
@triton.jit  # grid = ⌈M/4⌉，256 线程：每个 wave 一行
def a8w4_row_quant_bf16(x, xq, xs, rows):
    r = 4 * tl.program_id(0) + tl.arange(0, 4)
    xq[r, :], xs[r] = quant_row(x[r, :])             # IEEE 除法，与 a8w4_row_quant_fp8 逐位一致

@triton.jit  # grid = ⌈M·H/512 / 4⌉，256 线程：4 个 wave 各管一段 = 一个 token 的 512 列
def a8w4_route_sum(loc_ids, routes, out, M):
    for w in range(4):                               # 实际：4 个 wave 并行
        seg = 4 * tl.program_id(0) + w
        t, cols = seg // (H // 512), (seg % (H // 512)) * 512 + tl.arange(0, 512)
        acc = tl.zeros([512], tl.float32)
        for slot in range(TOPK):                     # 按槽顺序相加（与 sorted_sum 同序同舍入）
            if loc_ids[t, slot] >= 0:                # loc < 0：没有这条路由，不读
                acc += routes[loc_ids[t, slot], cols]
        out[t, cols] = acc.to(tl.bfloat16)           # RNE
```

**任务划分**
| kernel | grid × CTA | wave / lane |
|---|---|---|
| `a8w4_row_quant_bf16` | ⌈M/4⌉ × 256 线程 | wave 一行；lane 读 16 B 片 c·64 + lane（共 H/512 片） |
| `a8w4_route_sum` | ⌈M·H/512 / 4⌉ × 256 线程 | wave 管一个 token 的 512 列；lane 16 B |

**流水线**
```text
row_quant_bf16：读片 → lane 内 max|x| → 6 次 shuffle_xor 得整行 amax → scale = amax/240（IEEE，0 取 1）
  q = clamp(x/scale, ±240)（IEEE）→ RNE fp8 → -0 改 +0 → 每片 8 B 写回；lane 0 写 scale
route_sum：lane < topk 读 loc_ids[token] → readlane 逐槽广播 → topk 条 16 B 加载同时在途（loc < 0 不读、加 +0）
  f32 按槽顺序相加 → RNE 成 bf16 → 16 B 写 out[token, 列]
```
关键代码：[输入量化](a8w4_stream.py#L499)、[route_sum](a8w4_stream.py#L550)。

**资源**：都不用 LDS。
