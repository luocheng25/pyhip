# gfx942 FP8 block-scale MoE：实现与优化记录

本目录是gfx942（MI300X/MI325X）FP8 block-scale MoE的prefill实现。使用方法、sglang集成、当前性能与方案建议见[benchmark说明](../../../../../../benchmarks/moe_fp8_blockscale/readme.md)。第1–3节对应当前源码（2026-11-02），第4节是优化与回退记录（4.1节为DP2/TP8 DP-attention部署下I=256的专项优化）。

## 1. 数据流

```text
out[t] = sum_k w[t,k] * W2[e_k] @ ( silu(W13g[e_k] @ x[t]) * (W13u[e_k] @ x[t]) )      128x128 FP8 block scale
```

一次`fused_moe_fp8_blockscale_942`调用依次发出（同一stream，无host同步）：

| # | Kernel | grid x block | 作用 |
|---|---|---|---|
| 0 | aiter per-1x128量化（仅BF16输入） | - | x → FP8 [T,H] + scale [H/128,T] |
| 1 | aiter `moe_sorting`：I=256（短K路径）用block 64，其余block 256 | - | 路由按专家排序，每专家补齐到block的倍数；padding行token=T；EP时只保留本地专家，专家号为本地序号 |
| 1b | `_tiles_kernel` | 1 x 256 | compact tile表：每专家的块切成256行tile加一个64–192行尾tile（block 256时每块一个tile），`tiles[t] = (首行, 专家 \| 行数/64 << 16)`，tile数写在设备端 |
| 2 | `_gate_up_kernel` | 最坏tile数×(I/128) x 512 | 每(tile, 128个intermediate列)：gather X行 × [gate 128行 \| up 128行]，K=H；SiLU·up后按行1x128量化写A2/A2S；tile行数之外的band不发MFMA |
| 3 | `_inverse_kernel` | ⌈R/256⌉ x 256 | inv[t,k] = 路由(t,k)的排序行（先填-1，只扫有效前缀） |
| 4a | `_down_direct_kernel`（I=256） | 最坏tile数×(H/1024) x 256 | 短K专用：每(tile, 1024列)，前行数/64个wave各持64行A2（整个K）在寄存器，W2按64行块流过2槽LDS，其余wave只做DMA与barrier；输出Y [R,H]，padding行不写 |
| 4b | `_down_kernel`（I≥512） | 最坏块数×(H/256) x 512 | 稠密kernel的`_body`原样运行在(A2行, W2[e])上，输出排序顺序的BF16行Y [R,H] |
| 5 | `_sum_kernel` | T x 256 | 每CTA一个token：out[t] = BF16(Σ_k Y[inv[t,k]])，FP32按k顺序累加；inv为-1时读越界得0 |

`PYHIP_MOE_DOWN=fused`（实验，I=256且H为1024的倍数）时，1b/2/4a换成`kernels_m64._moe64_kernel`（每个64行块一个CTA：gemm1 → requant到LDS → down，直接写Y；见2.6），3和5不变。

| 缓冲区 | 写 | 读 | 内容 |
|---|---|---|---|
| A2 [R, I] FP8 | gate_up | down | 排序行的SiLU(gate)·up，按(行,128列)量化 |
| A2S [I/128, R] FP32 | gate_up | down | amax/240 × 路由权重；down的activation scale，权重因此不必在down里再乘 |
| Y [R, H] BF16 | down | sum | 已乘路由权重的每路由输出（排序顺序，按M块连续） |
| inv [T, topk] int32 | inverse | sum | 路由到排序行的映射，非本地路由为-1 |

A2/A2S/Y按设备缓存、只增不减（`_workspace`）。R×H×2必须小于4 GiB（buffer的32位偏移），超出时按token递归二分。

## 2. Kernel

### 2.1 公共流水（复用dense kernel）

两个GEMM都运行[gemm_fp8_blockscale_8w_942](../../../gemm/flydsl/gemm_fp8_blockscale_8w_942.py)的结构：256x256 tile，8 wave分G0/G1两组错开一个stage，每K-block四个stage M1/C1/M2/C2（计算stage各64条16x16x32 FP8 MFMA），LDS四个16 KiB槽X0/X1/P1/P2（XOR swizzle，preshuffled W保持aiter 16x16布局），dword LDS-DMA，block scale的partial晚两个block用FMA退休。块号按XCD连续分配：块b落在XCD b%8，`_xcd_order`让每个XCD拿到逻辑顺序中连续的一段，即连续的M块（基本是同一专家），权重在L2中共享。有效块数在设备端从`num_valid_ids`读，超出的CTA直接退出。

### 2.2 gate_up（`_gate_up_body`）

```python
# CTA: M块 mblk（专家 e = sorted_expert_ids[mblk]），intermediate列 128*jt .. +127
table[lane]  = token(row(lane)) * H          # row(l) = 128*(l>>5) + 2q + 8*((l&31)>>1) + (l&1)，padding行为越界偏移
xs_off[mb]   = token(wm*64 + 16*mb + r16)*4  # 该lane四个MFMA token的activation scale
# W tile行：P1 = gate[128*jt .. +127]，P2 = up[128*jt .. +127]
#   => wave (wm, wn) 的N-block 0-3是gate的64列，4-7是同样64列的up；C1用gate的weight scale，C2用up的
for kb in K-blocks:                           # 与dense kernel同一stage表
    G1的X DMA t：lo = readlane(table, 32*half+2t)，hi = readlane(table, 32*half+2t+1)
                 voffset = (lane<32 ? lo : hi) + swizzled chunk，soffset = kb*128
    G0的W DMA：preshuffled 16行组 (gate16|up16) + (t>>1)，chunk q + 4*(t&1)
    C1: acc[0..3] += gate partial * xs*ws_gate；C2: acc[4..7] += up partial * xs*ws_up
act = silu(acc[nbl]) * acc[4+nbl]             # lane: token 16mb+r16，列 64wn+16nbl+4g+e
amax = 每token在本wave 64列上的最大值（shuffle_xor 16/32），与wn^1的wave经LDS交换得到128列最大值
A2[row, 128jt + 64wn + 16nbl + 4g .. +3] = fp8(act * 240/amax)（v_cvt_pk_fp8_f32，每lane每块1个dword）
    # 先ds_write_b32进LDS：tile行r在区域r>>5的4 KiB处（避开行最大值），16 B chunk按r&7异或；barrier后
    # wave w读行32w .. +31（每lane一个ds_read_b128），每条buffer_store_dwordx4写8个完整的128 B行
A2S[jt, row] = amax/240 * sorted_weight[row]  # wn=0的wave写；amax=0（padding行）时为0
```

- 每个DMA用2条`v_readlane`加1次select取两行偏移，table只占1个VGPR（16个VGPR的打包方案会把寄存器推到上限）。在每次DMA前重新pin一次table，防止循环不变的readlane被提出循环、占用64个SGPR。
- epilogue的跨wave最大值交换复用wave私有的LDS区域，与dense epilogue的论证相同：G0复用X0/X1，G1复用P1/P2，最后一次读取在之前的barrier之前已完成。之后一次CTA barrier同时完成两组交换。
- 数值：K128 partial → FP32乘(xs·ws) → FMA累加；SiLU用exp2/rcp；量化与参考实现同为amax/240。与FP32参考的rel_l2为0.57%–0.62%，同条件下aiter为0.68%–0.77%。

### 2.3 down（I≥512）、inverse

I≥512时down直接调用dense kernel的`_body`：X=A2（排序行连续，无需gather），XS=A2S（stride为R），W/WS传入按专家偏移的view，N=H，K=I。输出与dense kernel相同（LDS转置后的BF16 store），行写到Y的排序位置。

### 2.4 短K down（`_down_direct_body`，I=256）

K=256时dense流水每个256x256 tile只有2个K-block，prologue/epilogue占约一半时间。短K down改为“A在寄存器、W流过LDS”：

```python
# CTA（4 wave，LDS 2×16 KiB，每CU 2个CTA）：M块 mblk × 列组 cg（1024列）；wave w 持有行 64w .. 64w+63
x[mb][kb]  = A2[16mb + r16, 128kb + 32g .. +31]       # 64个VGPR，整个K；A2S、行有效性一次读入
for 128列对 it（共用一个weight block scale）:           # 2个LDS槽：块2it在槽0，2it+1在槽1
    U0 U1 | barrier，DMA块2it+2 → 槽0 | U2 U3 | barrier，DMA块2it+3 → 槽1
    unit（32列）：64条MFMA = 4个M子块 × 2(v) × 2(kb) × 4 step
        MFMA v 的A行r16 = unit行 8(r16>>2) + 4v + (r16&3)   → 输出lane(r16, g)恰为8个连续列 8g .. 8g+7
        每个M子块的partial在下一个M子块的MFMA期间退休：p0*s0 + p1*s1（FP32 FMA）→ BF16 RNE → 4个dword
        偶数unit的结果暂存，与下一个奇数unit同时store：一行的128 B cache line由相邻两条dwordx4写满
    下一unit的W片段（8条ds_read_b128）在本unit开始时发出
```

- 行置换使MFMA输出不需要lane间交换；对应的LDS读用DMA时的行XOR（16行组内后8行的位置异或4）消除bank冲突。
- `s_waitcnt vmcnt(8)`依赖gfx942的VMEM按序完成：每个DMA之后恰有8条store。
- padding行（token=T）的store地址越界丢弃，Y只写真实路由（32K token时省12%写带宽）。
- 194 VGPR、无spill；MFMA与退休VALU按1:3交错（`sched_group_barrier`）。

### 2.5 sum

每CTA一个token：8个inv用标量load读入，wave w负责H/4列，lane每条指令读16 B，一条指令覆盖一行的1 KB连续区；8路FP32按k顺序求和后软件RNE转BF16。Y的load带`nt`（只读一次的流式数据，不占L2/MALL）。`launch_sum`的C0/NS参数可只处理一段列。

### 2.6 实验：M64 gate_up与1-stage（`kernels_m64.py`）

```python
# CTA（NW = 8：218 VGPR，每CU 1个；NW = 4：248 VGPR，每CU 2个；LDS 26.5 KB）：排序行row0 .. +63（一个64行块，专家e）
# gemm1：column wave cw = gate列 32cw .. +31 与 up同样的32列（N-block g0 g1 u0 u1）
#        NW = 8一遍算完I=256（cw = wave）；NW = 4分两遍（cw = 4p + wave），每遍X重新gather
for kb in K-blocks:                     # 2槽LDS环，每槽 = 64行 X的一个K-block（8 KB，16 B chunk c 在 c ^ (row & 7)）
    DMA X(kb+1) → 另一槽；W13片段(kb+1)直接从global读进VGPR；算kb：4 N-block × 4 M子块 × 4 step
    partial晚两块用FMA退休（与_gate_up_body同序，结果逐位一致）
act = silu(gate) * up；128列组 = 物理wave 4(w>>2) .. +3，行最大值经LDS交换 → FP8 codes + scale（× 路由权重）
# gate_up64：写A2/A2S（与_gate_up_kernel同布局）
# moe64：每遍结束即把A2写进LDS：列组0在A2H0（8 KB），列组1在X槽0；行 r 的chunk c 在 c ^ ((r >> 1) & 7)；
#        scale写在A2SC；之后：
x[mb][kb] = A2行16mb + r16的列组kb → VGPR（整个K）
wave w：输出列 (N/NW)w .. ，每128列一个W2 block scale，4个32列unit；
    下一unit的W2片段直接从global读（无LDS），MFMA/退休/成对store同_down_direct_body → Y逐位一致
```

- 与2-stage比，W13和W2每64行就要读一遍（复用只有1/4），整个MoE慢3%–9%（benchmark说明4.15）。
- NW = 4（tingqli式，每CU 2个CTA）更慢：X gather两次，每XCD同时在算的专家翻倍，L2未命中读3.17 → 6.08 GB（32K）。`PYHIP_MOE_FUSED_WAVES`默认8。
- NW = 4时把第0遍的A2 code留在寄存器里会到256 VGPR并spill（Y出错）；每遍结束即写LDS、两遍之间加`sched_barrier(0)`后为248。
- 静态循环必须用`range_constexpr`：`@flyc.jit`函数里的`for ... in range(n)`会被改写成`scf.for`，循环变量变成IR值，不能再做Python list下标。

## 3. 文件与限制

| 文件 | 内容 |
|---|---|
| [kernels.py](kernels.py) | gate_up、短K down、dense down、inverse、sum五个kernel，gather/DMA/epilogue helper，launch函数 |
| [kernels_m64.py](kernels_m64.py) | 实验：M64 gate_up（`launch_gate_up64`）与1-stage（`launch_moe64`），每CTA 8或4个wave（`NW`），见2.6 |
| [moe.py](moe.py) | 公开接口、输入量化与排序（aiter）、workspace、token二分、down/sum分发（`PYHIP_MOE_DOWN=dense`强制用dense down，`=fused`用实验的1-stage，`PYHIP_MOE_FUSED_WAVES`为其每CTA wave数（8或4，默认8）；`PYHIP_MOE_SPLIT`为列切片双stream实验开关，默认1） |
| [\_\_init\_\_.py](__init__.py) | 导出`fused_moe_fp8_blockscale_942`、`moe_sorting`、`quantize_input` |

- 仅gfx942、FP8 E4M3FNUZ、SiLU gate-up、gate行在前、aiter 16x16 shuffle、BF16输出；无bias、swiglu clamp、MXFP4权重；I%128==0，H%256==0。
- EP模式要求w13/w2的专家顺序与expert_mask中置1的顺序一致（与sglang standard dispatcher一致）。非本地路由在sum中按0计。
- 256行tile：I=256时专家只补齐到64行（compact尾tile）；I≥512时仍补齐到256行（大batch平均浪费约12%）。小batch仍以padding为主（<8K token不建议使用，sglang默认阈值8192）。
- 第一次调用会编译（4个kernel，约数十秒）；CUDA graph capture前须eager预热到最大token数，以免workspace扩容。

## 4. 优化与回退记录

| 日期 | 改动 | 结果 | 结论 |
|---|---|---|---|
| 11-02 | v1：2-stage，gate_up复用dense流水+readlane gather，down复用`_body`，路由权重并入A2S | 一次编译即正确（rel 0.59%）；32K token：TP8 3439 µs（aiter 3156），TP4 5478（aiter 5729） | gate_up约920 TFLOPS（执行行），TP8瓶颈在down（1178 µs）+sum（742 µs） |
| 11-02 | down消融：store改越界丢弃 | 1178 → 968 µs | store约210 µs；其余是K=256时每tile只有2个K-block的固定开销 |
| 11-02 | sum带宽探针：顺序行 vs 随机行 | 3.73 vs 3.69 TB/s | gather顺序不是问题，sum已接近读带宽 |
| 11-02 | N切片：down/sum按256–1024列分片交替，使中间结果留在256 MB Infinity Cache | sum 729 → 592 µs，down 1163 → 1437 µs，合计无收益 | 回退 |
| 11-02 | 切片的sum放到第二个stream与下一片down并发 | 慢3–5% | ROCm上两个stream几乎没有并发，回退 |
| 11-02 | EP8形状（32专家，I=2048，真实top-8路由，expert_mask） | 32K token 2399–2451 µs，同条件aiter 3488；对生产TP8 aiter为1.30x | 方案建议的核心：同通信量下改EP8 |
| 11-02 | gate_up/down按G个M块分组的tile顺序（G=4/8/16） | ±2%以内 | 噪声范围内，回退（XCD连续分配已经让同专家的块共享L2） |
| 11-02 | 超过4 GiB时按token二分 | 65536 token：TP8 0.91x、TP4 1.03x、EP8 1.39x | 保留 |
| 11-03 | compact M64尾tile（I=256）：64行排序 + 设备端tile表，gate_up/down按tile，行数外的band/wave不发MFMA | TP8单卡4K/8K/16K/32K：648→613、980→938–950、1717→1658–1669、3160–3196→3134–3158 µs | 保留。gate_up尾tile受band→SIMD固定映射限制（1 band仍占1/2时间，3 band不省） |
| 11-03 | 1-stage M64（`kernels_m64.py`）：gemm1 + requant到LDS + down在同一CTA，W13/W2片段直接进VGPR | A2、Y与2-stage逐位一致；整个MoE 8K/16K/32K：937→965、1684→1783、3149→3425 µs（moe64 1425/2583 vs gate_up+down+tile表 1294/2277） | 保留为实验开关`PYHIP_MOE_DOWN=fused`，不作默认。权重复用只有1/4，每CU 1个CTA；W2预取提前2个unit无变化，已回退 |
| 11-03 | 1-stage改tingqli式4-wave CTA × 每CU 2个（`NW=4`：两遍gemm1、A2分两半写LDS） | 逐位一致，248 VGPR；moe64 8K/16K/32K：783/1363/2568 → 845/1514/2804 µs；整个MoE 964/1782/3443 → 1042/1883/3557 | 更慢，`PYHIP_MOE_FUSED_WAVES`默认8。L2未命中读翻倍（3.17 → 6.08 GB）、时钟1.22 → 1.13 GHz；LDS补到40 KB（每CU 1个）只慢了3% |
| 11-04 | gate_up的A2经LDS中转，每条store写完整128 B行（原来每lane 16条dword store，每行分8次写16 B） | 逐位一致；32K单卡交替3轮：1228.7/1230.0/1240.1 → 1227.8/1226.4/1216.7 µs（约−9 µs）；e2e（moers8，141次调用中位数）1350.8 → 1339.3 µs | 保留。消融显示A2写出约占60 µs，但主要是写流量本身的代价，部分行写只占一小部分 |
| 11-04 | 短K down改8-wave CTA × 512行tile（每CU 1个CTA，W2的DMA与L2→LDS流量减半）；512行tile表与256行表同一次`_tiles_kernel`写出 | 逐位一致；32K单卡交替3轮（随机路由）：976.8 → 965.0 µs（−1.2%），偏斜路由也是−11 µs；但e2e（moers8，真实路由）中位数943.9 → 968.5 µs（+2.6%，排除与另一stream的rocprim排序重叠的调用仍是+25 µs）。只减半W2 DMA（结果错误的消融）为−49 µs | 回退。真实路由下尾tile更多，512行尾tile的CTA常只有1个wave在算却独占整个CU |

### 4.1 DP2/TP8 DP-attention专项（MoE TP8，每rank I=256）

约束：不改部署（`--tp-size 8 --dp-size 2 --ep-size 1 --enable-dp-attention`），MoE保持TP8，每次调用是DP gather后的全局token（16K–64K）。v1在该形状下只有aiter的0.81–0.92x。下表均为32K token、GPU0单卡、同一输入，单位µs（aiter同条件3160–3190）：

| 步骤 | 改动 | down | sum | 合计 | 结论 |
|---|---|---:|---:|---:|---|
| 0 | v1：dense `_body`做down，sum每线程16 B | 1165 | 729 | 3453 | 基线 |
| 1 | 短K down初版：A在寄存器、W流过LDS，ds_swizzle做lane对交换 | NaN | | | `_wait(lgkmcnt)`后缺`sched_barrier`，MFMA被调度到inline-asm ds_read之前；补上后正确 |
| 2 | 同上（已修正） | 1230 | 761 | 3542 | 比dense还慢：ISA中每个16x16块是“4条相关MFMA→s_nop→22条串行VALU→ds_swizzle等待”，没有重叠 |
| 3 | 行置换代替lane交换；每M子块的退休VALU插到下一M子块的MFMA之间；W片段预取；每块中间barrier | 1008 | 761 | 3355 | 消融：去store 630，再去退休VALU与同步500（≈功耗墙下的MFMA下限） |
| 4 | 按M块旋转列扫描顺序（想分散并发store） | 2577 | | | 慢2.5x，回退；说明store模式是主瓶颈 |
| 5 | HIP store探针（同样的Y写入量） | | | | 64 B半行、另一半约1 µs后再写：2.7 TB/s；同样64 B但两条store相邻：4.2；整行：4.5 TB/s |
| 6 | 偶数unit结果暂存，与奇数unit背靠背store（每条cache line在相邻两条store内写满） | 914 | 761 | 3295 | 保留 |
| 7 | 打包FP32：inline asm `v_pk_*` / LLVM `<2 x float>` | 1064 / 1011 | | | asm不在hazard识别范围内，结果错误且更慢；LLVM把向量运算标量化。回退 |
| 8 | BF16改为+0x8000舍入（VALU减半） | 943 | | | 无变化：VALU不是瓶颈，保留RNE |
| 9 | 去掉vmcnt等待 / 去掉DMA与barrier | 920–948 / 773 | | | 等待不是瓶颈；DMA与store共用VMEM路径 |
| 10 | sum：每CTA一个token，inv用标量load，每条load读1 KB连续区 | | 741 | | 小幅改善 |
| 11 | sum的Y load加`nt`（流式，不进L2/MALL） | | 592–614 | 3160–3218 | 保留，sum −18%（读约3.95 TB/s） |
| 12 | Y store加`nt`/`sc0`/`sc1` | ±1% | | | 无效，回退 |
| 13 | MALL驻留探针：写后立即读 | | | | ≤256 MB时写+读4.78 TB/s，2.4 GB时4.44：MALL最多省约8%，列切片不值得 |
| 14 | 列切片：第s片的sum放第二个stream，与第s+1片down并发（2/4片） | | | 3231 / 3487 | 变慢，默认关闭（`PYHIP_MOE_SPLIT=1`） |
| 15 | 原子累加探针（替代Y写回+sum） | | | | `global_atomic_pk_add_bf16`按行连续1.53 TB/s（1406 µs），MFMA布局0.36，FP32原子0.70：都不如写Y+sum（约0.48+0.56 ms下限） |

其他观察：

- 功耗墙：MoE循环中GPU一直在750 W，SCLK约1150 MHz（`amd-smi`），PMC的`GRBM_GUI_ACTIVE`换算为0.95–1.1 GHz。按1443 MHz的MFMA峰值估算会高估约25%。
- 跨XCD的L2不一致：在kernel内由“最后到达的CTA”做top-k归约，需要`buffer_wbl2 sc1`（整个L2写回）+`buffer_inv sc1`，代价大；down与sum之间的kernel边界正好完成这件事。
- down写出的代价（T=0使全部store越界丢弃，32K）：875 → 570 µs，MFMA忙46% → 67.5%，时钟1.23 → 1.29 GHz，即主要是store回压造成的MFMA发射停顿。把store移到同CU的另一个WG（1-stage）后，MFMA忙不变但功耗墙下时钟下降，store暴露约90–108%，没有收益（benchmark说明4.5(3)）。
- 用tingqli的代码在本机验证：他的2-kernel里store+sum比atomic更慢（32K：3763 vs 3269 µs）。1-stage的收益来自GEMM部分（不写回时2093 vs 2579 µs），不是把atomic藏在gemm1下面（benchmark说明4.5）。
- sum与two-shot all-reduce融合（`probes/sum_ar.hip`，按wave分为sum/push角色，IPC写穿）：32K为1771 µs，sum + RCCL为2213 µs（0.80x）。grid需限流到约96；不限流时，L2之外的读与xGMI远端写争用fabric，反而更慢（benchmark说明4.7）。
- 结果：TP8从0.81–0.92x提升到0.92x（8K）、1.00–1.01x（16K–64K）。剩余时间：gate_up 1417（约870 TFLOPS执行行），down 936，sum 592，量化/排序/inverse约260。Y写出+读回（各2.15 GB）在4.5 TB/s下就要约0.95 ms，是I=256时2-stage的硬下限。1-stage用BF16 atomic代替Y+sum（tingqli方案：M64、每CU 2个WG）也没有余量：同CU并发探针显示，另一个WG的gemm1在读内存时，atomic有70–77%（约1.0 ms）无法掩盖，与store+sum的代价相当。M64减少padding与1-stage无关，可以在2-stage中用compact尾块实现（32K执行行−8.1%）。TP8下MoE kernel剩余空间约5–10%（见benchmark说明4.5与第5节）。
- 2026-11-03，sglang新分支`feat/mimo-dpa-prefill-opt`：MoE kernel不变（I=256仍走aiter），收益来自层间通信改为序列并行（`SGLANG_DPA_SP_MOE`：本卡shard上router/量化，TP组all-gather FP8输入 + reduce-scatter输出，每层通信3.4–3.8 → 2.73 ms）和QKV/dense换用PyHIP FP8 GEMM（新增X行跨度，直接吃packed FP8 all-gather）；prefill每步−15.8%，bench吞吐+9.9%，GSM8K不变（benchmark说明4.17）。
- 2026-11-03，与pyhip提交45afb765（GateUp/Down LDS kernel）对比：它的GateUp比本实现的gate_up快9%–13%；它的Down在I=256时慢很多（1274 vs 764 µs，后者为down_direct + 求和），所以整个MoE在TP8时慢32%–40%，EP8（≤16K）时快3%–5%（benchmark说明4.19）。同时修复：dense down调用`_body`时补上了`LDA=K`（GEMM新增该参数后，I≥512会报TypeError）。
- 2026-11-03，TP8继续优化（benchmark说明4.20）：sglang侧o_proj改A2A + 全量权重、SWA层序列并行、dense TP1、embedding复制和prefill delayer，每步prefill净值435.9 → 387.8 ms，bench吞吐49,926 → 67,466 tok/s；I=512的EP2×TP4在sglang里MoE kernel −0.32 ms/层，但专家分组逐层不均衡使RS8等待更多，不采用。
