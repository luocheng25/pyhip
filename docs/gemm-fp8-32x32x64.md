# gemm_8wave_fp8bf16fp16：从 16x16x128 迁移到 32x32x64 f8f6f4

## 背景与目标

`src/contrib/gemm_fp8.py` 中的 `gemm_8wave_fp8bf16fp16` 是 HipKittens 8-wave 风格的 fp8 GEMM，
原本使用 `v_mfma_f32_16x16x128_f8f6f4`（M=16, N=16, K=128）作为核心矩阵指令。

本次改动把 **fp8 的两条路径**（非 scale 与带 block-scale 的 `use_f32_blockscales_128`）的核心 MFMA
换成 `v_mfma_f32_32x32x64_f8f6f4`（M=32, N=32, K=64）；`bf16` / `fp16` 分支仍保持 16x16x128。

硬件：gfx950（CDNA4 / MI350），支持 `v_mfma_f32_32x32x64_f8f6f4`。

> 本文主体描述非 scale 路径；带 scale 路径的特殊处理见文末「带 block-scale 路径」一节。

## 两个 MFMA 指令的布局差异

数据来自 CK（`archive/ck_tile/.../warp_gemm_attribute_mfma_impl.hpp`）并经实测验证：

| 指令 | A/B operand | C operand | kAMLane/kBNLane | kABKLane | kABKPerLane | kCMLane/kCNLane |
|------|-------------|-----------|-----------------|----------|-------------|-----------------|
| 16x16x128 | fp8[32] = 8 VGPR | f32[4] = 4 VGPR | 16 / 16 | 4 | 32 | 4 / 16 |
| 32x32x64  | fp8[32] = 8 VGPR | f32[16] = 16 VGPR | 32 / 32 | 2 | 32 | 2 / 32 |

关键点：
- **A/B 每个 lane 的寄存器数相同（都是 8 VGPR / 32 个 fp8）**，因此每个 warp 加载 A/B 的寄存器总量不变。
- **C 每个 lane 从 4 变成 16 个 f32**，但每个 32x32 tile 是 16x16 的 4 倍面积，tile 数量减少为 1/4，
  故每个 warp 的 C 寄存器总量不变（仍是 128 VGPR/lane）。
- 一次 32x32x64 只吃 K=64（16x16x128 吃 K=128），因此 `BLOCK_K=128` 内需要 **2 个 K window**。

实测确认（`v_mfma_f32_32x32x64_f8f6f4(cReg16, aReg8, bReg8, cReg16)`）：
- **A operand**：lane `(r=lane%32, klane=lane//32)` 持有 `A[r, klane*32 : klane*32+32]`（连续 32 个 fp8）。
- **C output**：`c_reg[i*4+j] = C[row=(lane//32)*4 + i*8 + j, col=lane%32]`，i,j ∈ 0..3。
  其中 `row` 对应 srcA 的行、`col` 对应 srcB 的行。

## Tiling 结构（保持 8-wave pipeline 不变）

`wg_M = wg_N = 256`，8 warp（`WARPS_ROW=2 × WARPS_COL=4`），沿用 4 个 128x128 half-block（`cindex`，`cm=cindex//2`, `cn=cindex%2`）。

| 量 | 16x16x128 | 32x32x64 |
|----|-----------|----------|
| 每 warp 每 half-block 的 tile 数 | nrM=4, nrN=2 | nrM=2, nrN=1 |
| 每个 tile 的 K window 数 | 1（K=128） | 2（每个 K=64） |
| 每次 `mfma()` 的 MFMA 条数 | nrM*nrN = 8 | nrM*nrN*2 = 4 |

因每条 32x32x64 做 2 倍 MAC，总算力不变。8-wave 软件流水线（`loop_body` 的 4 次 `mfma()` + s_barrier 结构）完全复用。

## 关键洞见

1. **K 排列自由**：MFMA 沿 K 求和，只要 A 和 B 用**相同的 (klane, 寄存器位置) → K** 映射即可，
   K 的具体顺序不必与 CK 的"连续"约定一致。这样才能用 row-major loader 的 swizzle 布局直接喂给 MFMA。

2. **srcA 决定输出行、srcB 决定输出列（lane%32）**。C 写回按 `col=lane%32=N` 组织（保证 N 方向 coalesced），
   因此 `mfma()` 必须传 **srcA=mfma_A(M)、srcB=mfma_B(N)**。
   > 注意：16x16 的 non-rotate 路径用的是相反顺序（srcA=mfma_B），直接照搬会导致输出 M/N 互换。这是本次实现中最隐蔽的一个 bug。

3. **A 始终 row-major，B 受 `bpreshuffle` 影响**。两种情况都要为 32x32 写新的 LDS 读取：
   16x16 的 `ds_read`（lane%16=行、lane//16=列组）与 32x32（lane%32=行、lane//32=K 组）lane 映射不兼容，无法复用。

## 主要代码改动

### `src/contrib/gemm_fp8.py`
- 新增标志 `mfma_32x32 = (AB_dtype=="fp8") and (not use_f32_blockscales_128)`。
- `nrM //= 2`、`nrN //= 2`（4→2, 2→1）。
- 寄存器声明分支：
  - `mfma_A = J.gpr(nrM, 2, 2, 4, "vfp8x4")`（`[m_tile, window, colgroup, 4]`）
  - `mfma_B = J.gpr(2, nrN, 2, 2, 4, "vfp8x4")`（`[b_index, n_tile, window, colgroup, 4]`）
  - `mfma_C = J.gpr(4, nrM, nrN, 16, "vf32")`
- `mfma()` 用 `v_mfma_f32_32x32x64_f8f6f4`，循环 `kk(2 window) / m_tile / n_tile`（kk 外层，
  交错两个 tile 的 window 以掩盖 w0->w1 的累加 RAW），srcA=mfma_A、srcB=mfma_B。
- `ds_readA`：新的 row-major 32x32 读取 `make_ds_read_32(warp_m*64)`，用无 bank conflict 的 swizzle
  **`swizzle(row, cg) = (cg ^ (row>>1)) & 7`**（为 32x32 的 `lane%32` 访问专门选定，详见下面性能节），
  按 `cg = kk*4 + klane*2 + ci` 取 2 个 col_group。
- `ds_readB`：按 `bpreshuffle` 分支——非 preshuffle 复用 row-major 读取；preshuffle 用专门的
  `ds_read_bps32`（block = `warp_n*nbK_ps + kk*2 + klane`，块内 `klane_ps` 偏移 0 / 512）。
- B loader 调用传 `mfma_MN=(32 if mfma_32x32 else 16)`。
- C 写回新增 32x32 分支：`col=lane%32=N`（coalesced），`(lane//32)*4 + i*8 + j` 为 tile 内 M 偏移，
  用 `buffer_store_short` / `buffer_store_short_d16_hi` 逐 bf16 写回。

### `src/contrib/common/loaders.py`
- `get_mfma_loader_preshuffled` / `get_mfma_loader` 增加 `mfma_MN=16` 参数：
  `mfma_K_bytes = (64//mfma_MN)*16`，`stride_1kb = vm_stride/mfma_K_bytes`，`nbK = K/mfma_K_bytes`，
  `vm_load_cnt` 中的 `16` 改为 `mfma_MN`。默认 `mfma_MN=16` 保持对既有调用方的兼容。
- `get_mfma_loader_row_major` 也增加 `mfma_MN=16`：当 `mfma_MN==32` 时将写放置/读查找的 swizzle
  从 `(col ^ row)` 改为 **`(col ^ (row>>1))`**，消除 32x32 `lane%32` 读取的 LDS bank conflict（详见性能节）。

### `tests/contrib/gemm/test_fp8_8wave.py`
- preshuffle 分支按 dtype 选择布局：`pre_shuffle(w, mfma_MN=(32 if AB_dtype=="fp8" else 16))`。

## 带 block-scale 路径（use_f32_blockscales_128）

块缩放参数 `scale_BM, scale_BN, scale_BK = 1, 128, 128`：`scaleA` 是 per-token（每 M 行、每 128-K），
`scaleB` 是每 128-N-block、每 128-K-block。kernel 用「延迟一拍」的 fifo：当前 K-block 的未缩放乘积
先写入 fifo，下一拍再乘上该 K-block 的 scale 累加进 `mfma_C`。

关键点：**scale 路径的 MFMA 用 `srcA=mfma_B`、`srcB=mfma_A`**（与非 scale 路径相反，也与 16x16 一致），
使输出的 **M = lane%32（每 lane 固定）**。这样一个 tile 的 16 个 `c_reg` 共享同一个
`scaleA[M]`（M 固定）与 `scaleB[N-block]`（N 均在同一 128-N half-block），只需**单个缩放**即可。

> 若沿用非 scale 的 `srcA=mfma_A`，则 `M=(lane//32)*4+i*8+j` 会随 `c_reg` 变化，需要 16 个 per-token
> `scaleA`，叠加 `mfma_fifo_scale` 后寄存器会超 256 而 spill。因此 scale 路径选择与非 scale 相反的算子顺序。

相应的代码变动：
- `ds_read_scaleA` 改用 `lane%32` 索引、每 tile 32 行（`scale_mrow = 32`）。
- `mfma_fifo` 每项 16 个 f32（`C_PER_TILE=16`）；每 tile 先累加 2 个 K window 再延迟 `v_fmac`（16 元素）应用 scale。
- 32x32 scale 路径仿照 16x16 使用**跨 tile/c_index 的双槽流式环形 FIFO**，不是整拍 ping-pong：
  当前 tile 写 free 槽，两条 MFMA 之间分两批各 8 条 `v_fmac` flush 上一个 pending tile；flush 完后旧槽变 free、
  当前槽变 pending。同一调用的第二个 tile 会 flush 第一个 tile，最后仅留一个 pending tile 给下一调用。
  因此只需原有 2×16 VGPR FIFO，就能让所有 `v_mfma_f32_32x32x64` 之间都有 8 条 `v_fmac`。
- scale 路径的 C 写回是「转置」版（`M=lane%32`、`N=(lane//32)*4+i*8+j`）。lane 与 lane+32
  分别持有同一行相邻的 4 列，因此参考 16x16 写回，用两次 `v_permlane32_swap_b32` 把两个 half-wave
  交织为连续 8 列，再用一次 `store_dwordx4` 写 16B。不能直接照搬 `v_permlane16_swap_b32`：微内核
  实测它配对 lane+16，会混合不同 M 行；`v_permlane32` 才配对所需的 lane+32。
- 该改动把写回从 64 个 `buffer_store_dword` site 降到 16 个 `buffer_store_dwordx4` site，VGPR 保持 248、
  无 spill、occupancy=2；非 scale 路径仍使用原有写回分支。

scale 路径功能验证（自写 torch 参考，块缩放 1/128/128，bpreshuffle True/False 各 M/N/K 组合）均 `calc_diff = 0.00000`。

## 功能验证

`python -m pytest tests/contrib/gemm/test_fp8_8wave.py` → **36 passed**
（覆盖 `fp8 / bf16 / fp16` × `bpreshuffle True/False` × `M∈{32,256,2400}` × `N∈{256,1536}` × `K=256`，
fp8 各例 `calc_diff = 0.00000`）。

## 环境与复现（venv）

硬件/软件：Ubuntu 22.04、gfx950（MI350）、Python 3.12、PyTorch 2.11（ROCm 定制构建）、HIP 7.2、aiter（scale 对比需）。
`torch`/`aiter` 为 gfx950 定制编译版（非 PyPI），因此 venv 需继承系统 ROCm 栈：

```bash
cd /host_lc/pyhip-gemm
python -m venv .venv --system-site-packages   # 继承系统 torch/aiter/ROCm
.venv/bin/python -m pip install -e .          # 将工作区 pyhip 以 editable 装入 venv
.venv/bin/python -c "import pyhip.contrib.gemm_fp8 as g; print(g.__file__)"  # -> 工作区 src
```

## 如何在 32x32 与 16x16 间切换

`gemm_8wave_fp8bf16fp16` 增加了**编译期参数** `use_mfma_32x32`（仅对 fp8 生效）：
`True` 用 `v_mfma_f32_32x32x64`，`False` 用 `v_mfma_f32_16x16x128`。

> **重要（对比方法学）**：该开关必须是**编译期参数**（会进入 JIT 的 kernel cache key，即 `.co` 文件名含
> `use_mfma_32x32=True/False`），这样两种实现是**不同的编译产物**。若改用环境变量/全局变量切换，因缓存 key 不含它，
> 16x16 与 32x32 会**复用同一份 `.co`**，导致对比无效（表现为两者数值完全相同）。preshuffle 时 `use_mfma_32x32`
> 还决定 `pre_shuffle` 的 `mfma_MN`（32 或 16），务必与内核一致。

## 性能记录（gfx950 / MI350，bf16 输出，2026-07-24）

以下为**有效对比**（通过编译期参数 `use_mfma_32x32` 产出两份不同 `.co` 内核，同进程/交替多轮取最好，
以抵消 GPU 热节流对绝对值的影响；比值 32x32/16x16 才是可靠指标）：

### 当前 scale 路径：16384×3584×6144

测试条件：`HIP_VISIBLE_DEVICES=6`，同一进程内 32x32/16x16 交替 3 轮，每轮 50 次并取最好值；
block-scale 为 `1×128×128`，输出 bf16。

| preshuffle | 32x32x64 | 16x16x128 | 32x32/16x16 |
|:----------:|:---------:|:----------:|:------------:|
| False | **2368.5 TFLOPS** | 2455.2 TFLOPS | **96.5%** |
| True  | **2374.1 TFLOPS** | 2469.4 TFLOPS | **96.1%** |

当前 32x32 scale 实现包含：无 LDS bank conflict 的 `(col ^ (row>>1))` swizzle、双槽流式 FIFO、
`v_fmac_f32` 以及 `v_permlane32_swap_b32 + buffer_store_dwordx4` 写回。资源为 248 VGPR、0 spill、
occupancy=2。完整回归为 42 tests passed；大 shape 与 torch 参考的 `calc_diff=0.000000`。

### 非 scale 路径历史记录：8192×8192×8192

| preshuffle | 32x32/16x16 |
|:----------:|:------------:|
| False | **约 95%** |
| True  | **约 94%** |

演进：非 scale 修复 LDS bank conflict 从 91%→95%；带 scale 修 conflict 到 79%、把 `v_pk_fma`→`v_fmac`
后旧调度历史最好约 85%；全 MFMA/fmac 交织后约 83%；最终用 `v_permlane32_swap_b32 + store_dwordx4`
优化写回后达到 96%（见下）。

所有用例 `calc_diff = 0.00000`（功能均正确）。

> ⚠️ 更早的文档曾用环境变量 `GEMM_MFMA16` 切换，但**缓存 key 未含开关导致 16x16/32x32 复用同一 `.co`**，
> 是无效对比（非 scale 甚至出现完全相同的数值）。改成编译期参数后才是真实对比。

### 根因定位（rocprofv3）与修复

用 `rocprofv3 --pmc SQ_LDS_BANK_CONFLICT ...` 直接对比两个内核，发现真正的瓶颈：

| 计数器（8192³，preshuffle） | 32x32（修复前） | 16x16 |
|------|:---:|:---:|
| `SQ_LDS_BANK_CONFLICT` | **33,554,432** | **0** |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 536,870,912 | 536,870,912（相同） |

- **纯 MFMA 吞吐并非瓶颈**：单独微基准（A/B 常驻寄存器、多累加器、无内存/barrier）测得
  `v_mfma_f32_32x32x64_f8f6f4` = 4733 TFLOPS，`v_mfma_f32_16x16x128_f8f6f4` = 4582 TFLOPS，
  32x32 反而快 ~3%。所以差距来自**内核结构**而非指令本身。
- **真因 = LDS bank conflict**：CDNA4 的 `ds_read_b128` 把 64 lane 分成 **4 组各 16 lane、共 64 个 bank
  （16 个 lane-bank）**，冲突只在组内看。原 swizzle `(col ^ row)` 是为 16x16 的 `lane%16` 访问调的；
  32x32 用 `lane%32`（32 行）访问时，同组内多 lane 落到同一 lane-bank，产生 **4-way bank conflict**。
- **修复**：仅对 32x32 的 row-major loader 改用 swizzle **`(col ^ (row>>1))`**（读写一致、仍是按列组的
  自反置换），使每组 16 lane 落到 16 个不同 lane-bank → **冲突归零**。修复后 `SQ_LDS_BANK_CONFLICT=0`，
  非 scale 从 91% 提升到 95%。（B 的 preshuffle 读 `lane_bank=L&15` 本就无冲突。）

### scale 路径写回优化与当前性能

修复 bank conflict 后（scale 路径也受益），scale-32x32 仍慢于 16x16。profile 显示：

| 计数器（scale，8192³） | 32x32(旧v_fmac调度) | 32x32(双槽流式FIFO) | 32x32(v_pk_fma) | 16x16 |
|------|:---:|:---:|:---:|:---:|
| `SQ_LDS_BANK_CONFLICT` | 0 | 0 | 0 | 0 |
| VALU/MFMA **co-execution** 占比 | 41% | **52%** | 9% | **53%** |
| 32x32/16x16 性能 | ~85% | **~83%** | ~79% | — |

- scale 版每个 K-block 后需用 `v_fmac_f32` 把 fifo（32x32 每 tile **16 个 f32**）乘 block-scale 累加进
  `mfma_C`。这些 VALU 本应与 MFMA **并行执行（co-execute，不同执行单元）**藏在 MFMA 之下。
- 旧实现按整拍保存两个 tile，最后一个 tile 的 w0/w1 会连续执行；改成跨 tile 的双槽流式 FIFO 后，
  汇编中所有相邻 MFMA 之间均有 **8 条 `v_fmac`**，无需增加 FIFO、VGPR=248、无 spill、occupancy=2。
  `SQ_VALU_MFMA_COEXEC_CYCLES` 从 41% 升到 52%（接近 16x16 的 53%），但端到端性能未同步提升，
  反而从旧调度历史最好约 85% 降到约 83%；说明剩余时间受其它 pipeline/依赖边界主导，而非 MFMA/VALU 并行率。
- ATT 随后确认旧写回的 64 个 `buffer_store_dword` 是最大 stall 热点：按采样 wave 归一化约
  **28,965 stall/wave**。改成 32 次 `v_permlane32_swap_b32` + 16 次 `buffer_store_dwordx4` 后，
  写回降到 **6,031 stall/wave（-79.2%）**；其它 MFMA、fmac、barrier、LDS 热点变化均在约 1% 内。
  同进程交替测试（16384×3584×6144）达到 noPre=96.5%、pre=96.1% of 16x16。
- **`v_pk_fma` 是负优化（已回退）**：曾把 flush 改用 `v_pk_fma_f32`（打包 2×f32，指令减半 16→8/tile），
  但打包 scale 需 `v_mov` 广播 `(sc,sc)`（引入串行依赖）+ 打包对 `e:e+1` 约束调度，使 co-exec **反降到 9%**、
  性能从 85% 掉到 79%。**“更少指令 ≠ 更快”**——细粒度独立的 16 条 `v_fmac` 调度更灵活、co-exec 更高，净更快。
- 其它试过的改法也**不可行/更慢**：
  - **即时 flush**（用下一 tile 的 MFMA 掩盖当前 tile 的 flush）：RAW 比 WAR 更糟，co-exec 降到 4%。
  - **fifo 按 c_index 奇偶双缓冲**（消除 WAR，理论最佳）：需 +32 VGPR，**超过 256 硬上限、无法编译**
    （scale-32x32 已用 252 VGPR）。
  - 暂存 1/4/8 个旧 fifo 元素虽然也能打断最后两条 MFMA，但需要额外 `v_mov`，实测均慢于纯槽位轮转。
- 结论：双槽流式 FIFO 已解决最后两条连续 MFMA，剩余差距来自其它 pipeline/tiling 成本。

## 备注 / 后续

- `bf16` / `fp16` 分支仍走 16x16x128（不属于 f8f6f4，本次不涉及）。
- 非 scale fp8 已基本追平 16x16（~95%），推荐 32x32；带 scale 路径 16x16 仍更优。通过编译期参数
  `use_mfma_32x32` 可切换。
- **进一步优化 scale 的方向**（需要更大改动）：
  1. 重构 warp tiling 让每 warp 覆盖更多 32x32 tile（提升 ILP）；
  2. 或把 flush 移出 `mfma()` 到 `loop_body` 的 ds_read 阶段，藏在访存延迟里。
