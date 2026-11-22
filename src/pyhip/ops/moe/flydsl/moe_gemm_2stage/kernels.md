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

## A8W4 Gate/Up prefill（[a8w4_gemm1.py](a8w4_gemm1.py#L344)）
prefill 档（M > 2048、给了 tile 参数或 decode 不支持的形状）按排序块算 mid[token, slot, :] = SiLU(x·w1_gate)·(x·w1_up)，f32 写出；
结构同底座 `gemm1.py` 的 prefill_1x4，权重换成 mxfp4。

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

## A8W4 Down prefill（[a8w4_gemm2.py](a8w4_gemm2.py#L50)）
prefill 档按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同底座 `gemm2.py` 的默认 prefill Down。

**任务划分**（默认 BM64；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (1, 排序块数) | 1 个排序块 × 全部 H 列 |
| CTA（256 线程） | 4 个 wave | 64 行 mq 整块驻留；逐个 64 列块走完 H |
| wave | 每个 64 列块中的 16 列 | 列 n·64 + 16w … +15 × 64 行 |
| lane | l = 16b + r | 权重列 r 的 K 16b…16b+15；B 操作数的 1/64 |
| K 步 | I/64 个 64-K 块 × 2 个 K32 | 每个列块一次算完全部 K（I320 为 10 步） |

退出与 grid 收缩同 Gate/Up。H2560 共 40 个列块。

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
64·I 须 ≤ 64 KiB 且是 4096 的倍数（gather 每轮 256 线程 × 16 B），所以 BM32 配 I320 / 192 不可用，I > 1024 时整个 prefill 不可用。

## 行量化 prefill（[a8w4_quant.py](a8w4_quant.py#L23)）
prefill 档把 bf16 输入 [M, H] 与 f32 中间 [M·topk, I]、流式档把不在 Gate/Up 尾部量化的 f32 中间按行量化成 fp8 与每行 scale，
算术与参考逐位一致。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows/4⌉ | 4 行 |
| CTA（256 线程） | 4 个 wave | 每 wave 1 行 |
| lane | cols/64 个元素 | 列 j·64 + lane（H2560 为 40 个，I320 为 5 个） |

**流水线**
```text
读本 lane 的 cols/64 个元素 → f32；lane 内取 max|x|，再 6 次 shuffle_xor（32 … 1）得整行 amax
scale = amax / 240（IEEE 除法；为 0 时取 1，含全零行与 amax/240 下溢）；lane 0 写 scale
q = clamp(x / scale, ±240)（IEEE 除法）→ cvt_pk_fp8_f32（RNE）→ -0（0x80）改 0 → 逐字节写
```
关键代码：[amax 规约](a8w4_quant.py#L46)、[scale](a8w4_quant.py#L47)、[量化](a8w4_quant.py#L52)。

**资源**：不用 LDS。

## A8W4 Gate/Up decode（[a8w4_decode.py](a8w4_decode.py#L197)）
M ≤ 32 且形状受支持（H ≤ 4096，即 H/16 个 16 B 片不超过 256 个线程；见 [gateup_decode_fits](a8w4_decode.py#L54)）时不排序：
每个 WG 算一条路由 (token, slot) 的 16 个 gate 与同号 16 个 up 通道，写 mid[token, slot, ch]（f32）。

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
槽对 Down 前量化 mid（f32 [M·topk, I]），M·topk·I/16 ≥ 2560 时量化 x（bf16 [M, H]）；算术同 [a8w4_quant.py](a8w4_quant.py)，结果逐位一致。

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

## A8W4 Gate/Up 流式（[a8w4_stream.py](a8w4_stream.py#L257)）
33 ≤ M ≤ 2048 时按排序块（同一 expert 的 16 / 32 / 64 行）算 SiLU(x·w1_gate)·(x·w1_up)：权重与激活直接从 global 读进 VGPR，
不经 LDS 双缓冲；写 f32 mid[token, slot, :]，或在尾部量化成 fp8 mq[route, :] 与 ms[route]。

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
关键代码：[前导](a8w4_stream.py#L236)、[按 S 特化](a8w4_stream.py#L461)、[共享激活](a8w4_stream.py#L362)、
[预取](a8w4_stream.py#L399)、[主循环](a8w4_stream.py#L407)、[SiLU](a8w4_stream.py#L287)、[尾部量化](a8w4_stream.py#L319)、
[夹逼与回退](a8w4_stream.py#L145)。

**资源**：LDS = 共享时 2·BM·64 B 激活环（BM16 2 KiB、BM64 8 KiB）+ 量化时 BM·W·4 B amax；
累加器 8·G·S 个 VGPR，分派只在它 ≤ 96 时让一个 WG 管全部 I 并尾部量化（TP2 BM32 为 80，BM64 为 160 故不量化）。

## A8W4 Down 流式（[a8w4_stream.py](a8w4_stream.py#L604)）
33 ≤ M ≤ 2048 时按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 route_sum。

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
关键代码：[共享激活](a8w4_stream.py#L669)、[通道因子](a8w4_stream.py#L688)、[权重与 delta](a8w4_stream.py#L650)、
[delta 交换](a8w4_stream.py#L657)、[主循环](a8w4_stream.py#L710)、[写 routes](a8w4_stream.py#L733)、[LDS 预算](a8w4_stream.py#L598)。

**资源**：LDS = W·1 KiB 通道因子 + 共享时 BM·(I+16) B 激活 + W·1 KiB 越界片槽（构建时断言 ≤ 64 KiB；TP2 BM32 为 18.5 KiB，
I1024 BM64 要 73 KiB 所以不共享）；激活占 S·I/64·4 个 VGPR（BM64、I320 为 80）。

## 流式路径的输入量化与 route_sum（[a8w4_stream.py](a8w4_stream.py#L494)）
33 ≤ M ≤ 2048 时代替 a8w4_quant 的输入量化与底座 `sorted_sum`；算术与它们相同（量化逐位一致，求和同顺序同舍入）。

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
关键代码：[输入量化](a8w4_stream.py#L494)、[route_sum](a8w4_stream.py#L545)。

**资源**：都不用 LDS。
