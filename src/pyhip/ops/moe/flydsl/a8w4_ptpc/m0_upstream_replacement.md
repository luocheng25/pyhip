# M0 深度参考：`moe_gemm_2stage` 换成上游 gfx9_fp8

入口文档是 [README.md](README.md)，M0 的设计说明在 M6 移到了 [history.md](history.md)。本文放它们装不下的细节：补充术语、Qwen3.8 配置全表、逐文件相对上游的改动、
行为差异、分流规则全表（B1–B29）、覆盖项全表（63 项）和测量来源。

“上游”= ROCm/aiter@df3e0b2f 的 `aiter/ops/flydsl/kernels/moe_gemm_2stage/gfx9_fp8/`；
“legacy”= 替换前（8027faad）的 pyhip 实现，现在在 `*_legacy.py` 与 `gemm2_default.py`。
用例前缀：`U` = [test_moe_gemm_2stage_upstream.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_moe_gemm_2stage_upstream.py)，
`Q` = [test_qwen38_ptpc.py](../../../../../../tests/ops/moe/a8w4_ptpc/test_qwen38_ptpc.py)，
`TM` = `tests/ops/moe/test_moe.py`，`TT` = `tests/ops/moe/test_tuned_moe.py`；
`U::routes[x]` = `test_compile_gemm_routes[x]`，`U::adapt[x]` = `test_compile_gemm_adapts_launchers[x]`。

<a id="terms"></a>

## 0. 补充术语与配置全表

[history.md §3.1](history.md#terms) 只列 10 条主要术语，其余在这里：
- **`METADATA_TILE_SIZE_M`**：`moe_sorting` 补齐用的块行数（tuned_moe 里取 Down 的 BM）。Gate/Up 的 BM 比它小时，一个排序块拆成多个任务；只有 legacy 支持，上游要求两者相等（B6）。
- **full / tail 容量**：compact 预先分配的任务表行数（[`task_capacities`](../moe_gemm_2stage/gemm2_8x1_compact.py#L27)）。full 任务每个 M256（4 个 M64 排序块），tail 任务每个 M64；超出 GPU 上实际任务数的 CTA 直接返回。
- **padding 128 B**：prefill Down 写中间结果时每行末尾多留 128 字节（行跨度 = N + 64 个 BF16），`sorted_sum` 按同一跨度读；由配置的 `padding=128`（即 `down_output_padding_bytes`）指定。
- **shuffle（权重）**：aiter `shuffle_weight(w)`（默认 layout (16, 16)）把每个专家的权重按 16 行 × 16 字节的小块重排，并给返回的张量加 `is_shuffled = True`；`tuned_moe` 靠这个属性判断能否走本地 kernel。
- **`jit_*`**：`tuned_moe` 的 ASM 候选（`pyhip.ops.moe.asm`），如 `jit_batch1`、`jit_batch`、`jit_splitk`（两阶段 split-K）。
- **mimo_ptpc**：`pyhip.testing.moe_shapes` 里的 MiMo 形状（H=6144、每卡 I=256、E=384、top-8、FP8 ptpc），`bench_tuned_moe.py` 用它计时。
- **fd2 解码**（M2 计划，细节以实现为准）：pyhip PR #64 的 mxfp4 权重格式。每个输出通道取各 32-K 组 E8M0 scale 的最大值作通道因子（在 epilogue 乘，位置与 FP8 的每通道 scale 相同），组内只存相对它的指数差（差太大的组置 0）；K 循环里查表把 E2M1 精确转成 FP8，再做 FP8 MFMA（gfx942 没有 FP4 MFMA）。
- **logits_diff**：`1 − 2·Σ(x·y) / Σ(x² + y²)`，x 为参考、y 为输出，对全部元素求和，0 表示完全一致；与 aiter `calc_diff` 是同一公式（TM、TT 与 tuned_moe 调优用 `calc_diff` ≤ 0.02，Q 用 < 1e-3）。
- **中位比值**：替换后 ÷ 替换前的耗时比，每轮算一次，取各轮的中位数；大于 1 表示变慢。

<a id="configs"></a>

Qwen3.8 配置全表（[`FLY_CONFIGS`](../../../../testing/moe_a8w4_ptpc.py#L41)；只在用例和 bench 里经 `make_moe_runner` 强制走 FlyDSL，生产调用由 `tuned_moe` 自动调优，见 [history.md §3.3](history.md#callchain)）。T 取第一条“上界 ≥ T”的行，超出最后一行时用最后一行；decode / prefill 交界由这张表决定，当前是 128/129，M1、M2 跟随它（用户 11-21 确认）。

| T | TP2（I=320） | TP4（I=160 补零到 192） |
|---|---|---|
| 1 | decode batch1：gn32，Down dm16 dn64 | 同左 |
| 2–16 | decode batch1：gn64，Down dm16 dn64 | 同左 |
| 17–128 | decode split-K：gm16 gn128，Down dm16 dn128 | 同左 |
| 129–1024 | prefill：gm64 gn128 gk128，Down 1x4 dm64 dn256，padding 128 B | prefill：gk256，其余同左 |
| 1025–8192 | prefill：gk256，Down 1x4 | 同左 |
| >8192 | prefill：gk256，Down 1x4 | prefill：gm64 gn128 gk256，Down compact dm64 dn128，padding 128 B |

TP4 补零：FlyDSL Down 要求 I % 64 == 0（tuned_moe 也按 `i % 64` 判断能否走本地 kernel），[`make_weights`](../../../../testing/moe_a8w4_ptpc.py#L114) 在加载权重时给 gate、up 各补 32 行零、w2 补 32 列零、w1 scale 补 1；补出的通道输出 0，结果与 I=160 等价（只差累加顺序）。未补零的 I=160 在 tuned_moe 里没有本地候选，只走 aiter。

<a id="files"></a>

## 1. 同名文件相对上游的改动

增删行数是对照上游同名文件 `diff` 的结果（含空行）；每个文件头都写明了来源和差异。

| 文件 | 相对上游 | 改了什么 |
|---|---|---|
| `__init__.py` | +8/−62 | 删去上游不分设备的 `compile_gemm`，改从 `gemm.py` 导入；多导出 `compile_moe_reduction` |
| `common.py` | +79/−2 | `aiter.jit.utils.chip_info` 换成本地 `get_gfx_runtime` / `get_cu_num`；舍入开关默认值 "0" 改 "1"；末尾加 `_run_compiled` 和 pyhip 独有函数 |
| `gemm1.py` | +8/−0 | 内容不变；末尾转导出 legacy 的 `_build_moe_gemm1` |
| `gemm2.py` | +6/−2 | 只改 compact builder 的名字 |
| `gemm2_1x4.py` | +3/−0 | 只加文件头 |
| `gemm2_8x1.py` | +10/−1 | 上游 `_build_moe_gemm2_8x1` 改名 `_build_moe_gemm2_8x1_upstream`；原名转导出 legacy 的独立 8x1 builder |
| `gemm2_8x1_compact.py` | +18/−4 | builder 与 `allocate_task_buffers` 加 `_upstream` 后缀；原名转导出 legacy 版（pyhip 签名） |
| `moe_reduce.py` | +178/−4 | 上游 `sorted_sum` 改名 `sorted_sum_upstream`；末尾保留 pyhip 的 `sorted_sum`、`compile_moe_reduction`、`invert_sorted_ids` |
| `quant.py` | +4/−3 | `_run_compiled` 改从 `common.py` 导入 |

legacy 文件相对替换前同名文件的改动（只改 import 与文件头）：`gemm1_legacy.py` +5/−2、`gemm2_legacy.py` +6/−4、
`gemm2_1x4_legacy.py` +5/−3、`gemm2_8x1_legacy.py` +4/−2、`gemm2_8x1_compact_legacy.py` +6/−3、`gemm2_default.py` +6/−2；
`common_legacy.py` +11/−56，多删的是它自己的 BF16 舍入实现，改为从 `common.py` 导入同一实现与开关。

<a id="differences"></a>

## 2. 与上游的行为差异

1. 按设备缓存：上游 `compile_gemm` 与 `sorted_sum` 不分设备；pyhip 的 `compile_gemm`、`sorted_sum`、`invert_sorted_ids`
   每个设备各编一份（D1、D6）。
2. 分流与包装只在 pyhip 的 `gemm.py`：上游 batch1 launcher 多一个 token 数指针、取路由数 `batch·TOPK`，compact 多一个
   `cu_count`，`gemm.py` 把它们包成 pyhip 原来的参数（A1、A2）。
3. BF16 舍入默认简化 RTA（上游默认 RTE），legacy 与上游共用同一个开关（E2）。
4. `MOE_PREFILL_TILE_K`：`compile_gemm` 在分流前解析，只作用于 Gate/Up；上游 Down 不读，legacy Down 仍读。替换前
   default、1x4 Down 也读它（BF16 default Down 的 BK 由它决定），这是行为差异（A13）。
5. `CU_NUM`：只影响上游 `get_cu_num()`。上游 Down（1x4、compact full）按 CU 数是否为 80 判定 MI308X 拓扑，legacy 按
   设备名；compact 包装与 `allocate_task_buffers` 都用设备真实 CU 数，不受 `CU_NUM` 影响（A2、A11）。
6. `_run_compiled` 是上游 tensor_shim 版的最小替代：没有 AOT 仅编译模式和特化键；编译失败时先弹出泄漏的 `ir.Context`
   再抛出（D4）。`get_gfx_runtime` / `get_cu_num` 读当前 torch 设备，不支持 AOT 回放与 rocminfo。
7. quant：上游 `flydsl_absmax` 不在内部清零 Amax（调用方清零，否则结果是旧值与新 absmax 的较大者）；amax=0 时
   `flydsl_quant_per_tensor` 输出 0（A12）。pyhip 内没有调用方，`moe_gemm_splitk` 仍导出二者。
8. `has_num_local_tokens=True` 只有上游支持，且只允许 batch1：`compile_gemm` 不做包装，原样返回上游原生 launcher（多一个 token 数指针，接收路由数 T·TOPK）；计数之后的行不计算，输出保持 Gate/Up 清零后的 0（A4、A6）。pyhip 内只有用例调用。
9. `sorted_sum_upstream`、`allocate_task_buffers_upstream` 只为与上游一致而保留，pyhip 不调用（A3、A7）。

<a id="routes"></a>

## 3. 分流规则全表（B1–B29）

[`gemm.gemm_implementation`](../moe_gemm_2stage/gemm.py#L39) 按表中顺序判定，首个命中即返回。例子是 `U::routes[...]`
的参数 id，基准形状是 Qwen3.8 TP2 的 FP8 ptpc（Gate/Up N=640、K=2560；Down N=2560、K=320；compact 例子 K=192）。
每个用例断言三件事：`gemm_implementation` 的结果；`compile_gemm` 只调用了该实现、该阶段的 builder；判给上游的配置
上游 builder 能构建。

| id | 规则 | 走上游的例子 | 走 legacy 的例子 |
|---|---|---|---|
| B1 | `has_num_local_tokens=True` → 上游，先于其它条件 | num-local-tokens-first（SiTUv2 也判上游，由上游 builder 拒绝） | — |
| B2 | `force_batch1_path` 先把 alg 改成 batch1 | force-batch1 | 对照 gateup-odd-k-tiles |
| B3 | 非 gfx942 → legacy | — | gfx950-gateup、gfx950-down |
| B4 | 权重不是 bf16 / fp8 → legacy | bf16 | fp4 |
| B5 | 激活不是 silu / swiglu → legacy | swiglu | situv2 |
| B6 | `METADATA_TILE_SIZE_M` 不是 None 或 BM → legacy | metadata-none、gateup-prefill | metadata-ne-bm（256） |
| B7 | alg 不认识 → legacy | — | alg-unknown |
| B8 | Gate/Up 带 `down_path` 或非 0 padding → legacy | gateup-padding-0 | gateup-down-path、gateup-padding |
| B9 | Gate/Up prefill 要 `N % BN == 0` | gateup-prefill | gateup-n-tile |
| B10 | Gate/Up prefill 要 `K % (2·BK) == 0` | gateup-bk256 | gateup-odd-k-tiles（K=2304、BK256） |
| B11 | Gate/Up prefill 的 BK 默认值：FP8 128、BF16 64 | gateup-bf16-default-bk64 | gateup-fp8-default-bk128（K=384） |
| B12 | BF16 Gate/Up 要 BK ∈ {64, 128} | — | gateup-bf16-bk256 |
| B13 | BF16 Gate/Up 要 `2·BM·BK·2 ≤ 64 KiB` | gateup-bf16-lds-64k（恰好 64 KiB） | gateup-bf16-lds-over |
| B14 | Gate/Up split-K / batch1 不查形状 → 上游 | gateup-splitk、gateup-batch1 | — |
| B15 | stage 不认识 → legacy | — | stage-unknown |
| B16 | default Down 带 padding → legacy | down-padding-0 | down-padding |
| B17 | default Down prefill 要 `K % 64 == 0`、`N % 256 == 0`；split-K 不查 N | down-splitk-n128 | down-prefill-k64、down-prefill-n256 |
| B18 | default Down BF16 要 `BM·K·2 ≤ 64 KiB` | down-bf16-lds-64k | down-bf16-lds-over |
| B19 | 其余 default Down（prefill / split-K / batch1）→ 上游 | down-prefill、down-splitk、down-batch1 | — |
| B20 | 1x4 / compact 要 prefill、FP8、BM64 | down-1x4 | special-splitk、special-bf16、special-m128 |
| B21 | 1x4 / compact 要 padding ∈ {0, 32, 64, 128}（None 视为 0） | special-padding-none、special-padding-32 | special-padding-96 |
| B22 | 1x4 / compact 要 `K % 64 == 0`、`N % 256 == 0` | — | special-k64、special-n256 |
| B23 | 1x4 / compact 要 `64·K + scale + 8 KiB ≤ 64 KiB`（ptpc 的 scale 1 KiB） | special-lds-ptpc-k832、special-lds-per-tensor-k896 | special-lds-ptpc-k896、special-lds-per-tensor-k960 |
| B24 | 1x4 要 BN256 | down-1x4 | 1x4-n128 |
| B25 | compact 要 K ∈ {192, 256, 384, 512, 640} | down-compact、compact-k256/384/512/640、compact-per-tensor | compact-k320（Qwen3.8 TP2）、compact-k448 |
| B26 | compact 要 N ≥ 512 | compact-n512 | compact-n256 |
| B27 | compact 要 BN128 | down-compact | compact-n256-tile |
| B28 | compact 要 0 < E ≤ 2048 | compact-e2048 | compact-e2049、compact-e0、compact-e-none |
| B29 | 独立 8x1、未知 path → legacy | — | 8x1、8x1-m64、unknown-path |

<a id="coverage"></a>

## 4. 覆盖项全表（63 项）

测试负责人整理（M0 收尾），每项至少一个用例；B32、D6 的用例在本机（单张 gfx942）跳过。B1–B29 见上一节。

### A 接口契约

| id | 覆盖项 | 用例 |
|---|---|---|
| A1 | 上游 batch1 包装成 pyhip 的 8 参数约定（`topk_ids` 充当 token 数指针，M→M·TOPK） | U::adapt[upstream-gateup-batch1, upstream-down-batch1]；Q::test_matches_reference[tp*-t1/t8/t16] |
| A2 | compact 包装多传 `cu_count` = 设备真实 CU 数（`CU_NUM=40` 时也不变），照抄 compile_hints | U::adapt[upstream-down-compact]；Q::test_matches_reference[tp4-t8200, tp4-t65536] |
| A3 | legacy 分配的任务缓冲容量 = 上游按同一 CU 数的拆分容量 | U::test_compact_task_buffers_match_upstream_split |
| A4 | 不包装的出口原样返回 builder 的 launcher：上游 prefill / default / 1x4、`has_num_local_tokens`、legacy | U::adapt |
| A5 | 参数透传：上游 Down 的 padding None→0；legacy 收到原值和独有参数 | U::adapt[upstream-down-default, upstream-down-batch1, legacy-*] |
| A6 | `has_num_local_tokens=True` 的上游原生 launcher：计数之后的行不写 | Q::test_num_local_tokens_skips_padding_rows |
| A7 | `sorted_sum` 是 pyhip 按设备的版本（padding 默认 None），上游版改名不用 | U::test_pyhip_symbols_keep_their_modules；TM::test_fly_down_paths |
| A8 | pyhip 独有、测试与 experiments 用到的符号仍在原模块；上游同名实现带 `_upstream` | U::test_pyhip_symbols_keep_their_modules；TM::test_down_device_config |
| A9 | 包导出按设备缓存的 `compile_gemm`（不是上游 `__init__` 的版本） | U::test_pyhip_symbols_keep_their_modules |
| A10 | kernel 模块不导入 aiter | U::test_kernel_modules_import_without_aiter |
| A11 | `get_gfx_runtime`、`get_cu_num`（`CU_NUM` 覆盖）、上游 `get_down_device_config`（非 gfx942 报 RuntimeError） | U::test_chip_info_stand_ins |
| A12 | quant：absmax 不清零、量化数值 | U::test_flydsl_quant_per_tensor_contract |
| A13 | `MOE_PREFILL_TILE_K` 在分流前解析，只作用于 Gate/Up | U::test_compile_gemm_prefill_tile_k_env |

### B 分支与变体（B1–B29 之外）

| id | 覆盖项 | 用例 |
|---|---|---|
| B30 | Qwen3.8 的生产配置两级 GEMM 都判给上游，且实际编译的 alg 与 Down path 就是配置表这一档 | Q::test_matches_reference（全部） |
| B31 | legacy 功能不变：SiTUv2、独立 8x1、compact K=320、METADATA≠BM、上游 Gate/Up 配 legacy Down、legacy 报错文本 | TM::test_fly_situv2_*、TM::test_fly_down_paths、TM::test_fly_down_8x1_*、TT::test_fly_prefill_k_tiles、TT::test_fly_compile_failure_pruning、TT::test_fly_gateup_rejects_incomplete_k |
| B32 | FP4 / MXFP4、gfx950 的 legacy 功能（本机跳过，需 gfx950） | TM::test_fly_mxfp4*、TT::test_fly_mxfp4_direct_candidates、TT::test_fly_default_down_large_lds 等 |

### C 边界

| id | 覆盖项 | 用例 |
|---|---|---|
| C1 | Qwen3.8 配置表的每一档与各档上界：1、8、16、32、128、129、256、257、1024、2000（非 64 整倍数）、8192，TP4 另加 8200；decode / prefill 交界由配置表决定，当前 128/129，两侧都有用例（T=129 为第二轮新增） | Q::test_matches_reference（25 个） |
| C2 | 最大 T=65536：路由中间结果超过 2 GiB，字节偏移超出 int32 | Q::test_matches_reference[tp2-t65536, tp4-t65536] |
| C3 | TP4 的 I=160 补零到 192 | Q::test_matches_reference[tp4-*]、Q::test_decode_cuda_graph[tp4-*] |
| C4 | 64 KiB LDS 恰好相等 / 超出 | U::routes（B13、B18、B23） |
| C5 | compact 任务容量：0 块、少于 4 块、CU 1…304、E 1…2048 | U::test_compact_task_buffers_match_upstream_split |
| C6 | 空与零：token 数为 0；amax 为 0 | Q::test_num_local_tokens_skips_padding_rows；U::test_flydsl_quant_per_tensor_contract |

### D 执行模式

| id | 覆盖项 | 用例 |
|---|---|---|
| D1 | 分流在按设备的编译缓存里只判定一次；换设备键重新判定 | U::test_compile_gemm_routes_once_per_device |
| D2 | decode 的 CUDA graph（batch1 gn32 / gn64、split-K）replay 新输入 | Q::test_decode_cuda_graph |
| D3 | `has_num_local_tokens` 的计数在 graph replay 之间可变 | Q::test_num_local_tokens_skips_padding_rows |
| D4 | `_run_compiled` 首次编译并执行、之后复用；编译失败后可恢复 | U::test_run_compiled_compiles_once_and_recovers |
| D5 | 非默认 stream 加 graph 的上游 batch1 | TT::test_fly_direct_clear_graph |
| D6 | 多卡 launcher 隔离（需两张卡，单卡跳过） | TT::test_fly_reduce_device_cache_stream_graph |

### E 数值

| id | 覆盖项 | 用例 |
|---|---|---|
| E1 | Qwen3.8 上游 kernel 对 FP32 参考，容差同上游 aiter 用例 | Q::test_matches_reference、Q::test_decode_cuda_graph |
| E2 | 舍入开关：未设、"1"、"True" 为简化 RTA，"0" 为 RTE；legacy 用同一函数 | U::test_bf16_rounding_switch |
| E3 | `_f32_to_bf16` 逐位：RTA = `(bits + 0x8000) >> 16`，RTE = 就近偶数（NaN 仍为 NaN） | U::test_f32_to_bf16_follows_switch；TT::test_fly_bf16_rounding_bits |
| E4 | RTE 下四类上游 kernel 仍满足同一参考与容差 | Q::test_rte_rounding_matches_reference |
| E5 | Down 打包前的舍入逐位正确 | TT::test_fly_down_rounds_before_packing |
| E6 | 上游 per-tensor 量化的数值 | U::test_flydsl_quant_per_tensor_contract |

<a id="sources"></a>

## 5. 测量来源（都不是 mf 复核的数据）

路径都在 myflow 任务目录（`/root/.myflow/tasks/moe-ptpc-a8w4/`）下，不在仓库里。

| 内容 | 谁测的 | 位置 |
|---|---|---|
| Qwen3.8 bench 16 个 case：替换前、替换后 RTE、替换后 RTA 三版本轮换 6 轮 | 编排者 | `runs/20261121-094917-_task-run-8e2d/out.log` |
| `bench_tuned_moe.py` 替换前后初筛：7 个 FP8 模型、2 个 BF16 模型共 43 个 case，9 个慢 3% 以上（mimo_ptpc M=256 为 1.136） | 编排者 | `runs/20261121-104005-_task-run-7733/out.log` |
| 9 个初筛 case 第一轮复测：mimo_ptpc M=256 为 1.119（胜出从 jit_splitk 换成 aiter）；胜出配置相同的 qwen35_35B M=16 为 1.066、qwen35_397B_k256 M=16 为 1.035，另 4 个 ≤ 1.018；qwen35_35B_k256 M=1、qwen35_397B_k256 M=1 换了胜出配置（1.064、1.047） | 编排者 | `runs/20261121-104604-_task-run-af22/out.log` |
| 第二轮复测，两边强制用同一调优配置：5 个 case 为 0.938–1.027，mimo_ptpc M=256 为 1.007 | 编排者 | `runs/20261121-105710-_task-run-57d8/out.log` |
| `bench_moe_a8w4_ptpc.py --mf --check`（f97327ce）：fly_ptpc 与 aiter_ptpc 参考行 | 实现者 | `runs/20261121-101211-_task-run-bebe/out.log` |
| 替换后重扫 `FLY_CONFIGS`（`--sweep`）：全部 case；RTA 下 TP4 T=65536 等 | 实现者 | `runs/20261121-091729-_task-run-8cc9/out.log`、`runs/20261121-092025-_task-run-1f27/out.log` |
| 正确性命令（补 T=129 后的最终状态）：1344 通过 / 733 跳过 / 191 deselected | 测试负责人 | `runs/20261121-132445-_task-run-0738` |
| RTE 环境下 `tests/ops/moe/a8w4_ptpc` 126 个用例通过 | 测试负责人 | `runs/20261121-110533-_task-run-46be` |
| 22 处变异（改分流条件、去掉 compact 的 CU 数、RTA 默认改 0、把 split-K 配置跑成 batch1 等）全部被新用例检出 | 测试负责人 | `milestones/M0/coverage.md` |
