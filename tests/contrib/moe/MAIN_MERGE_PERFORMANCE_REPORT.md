# MoE down 合并最终性能报告（2026-09-02）

## 范围与最终状态

本次合并基于 `hy3-single-n512-handoff@3591fd0`。公开兼容入口保留在
`src/contrib/flydsl/moe_gemm_splitk.py`；实现参考FlyDSL上游`kernels/moe`的分层，
拆分到`moe_gemm_2stage/`：`gemm.py`负责stage分发，`gemm1.py`负责gate-up，
`gemm2.py`按路径分发到`gemm2_{default,1x4,1x8,2x4,4x1}.py`，其余公共能力位于
`{layout_helpers,moe_reduce,quant,common}.py`。新子包与FlyDSL上游一致，只公开
`compile_moe_gemm1`、`compile_moe_gemm2`和`compile_moe_reduction`三项keyword-only
缓存API；原`moe_gemm_splitk.py`保留兼容入口。重构前算法基线源码
SHA256为`efbf7b566968f2ce69696e05d4334370521454a63185185b32632624144d459b`；
原三条专用down路径均与该基线逐bit一致；新增`4x1`使用独立BF16 CShuffle舍入路径，
按数值门禁验证。以下为完成编译期cache修复后的最终工作树源码：

| 文件 | SHA256 |
| --- | --- |
| `helpers.py` | `1f9a0940b172e5bd7fcfd7e940455f7d67a230792372c2902f370c8a8521bee6` |
| `moe_gemm_splitk.py` | `92912d1d5c26470ec839fb8ae953d8691c9f2a2a142c3a2d9353ce33553da5e3` |
| `moe_gemm_2stage/__init__.py` | `7377ded7c4849dc903e4b5e18dcd0d61c071cfa14ae9ef2cfc173d9e93e1ae6b` |
| `moe_gemm_2stage/common.py` | `85e2153dce37859947577223959f7958b78e8147441e53d3ed4f4314d5fe4f0a` |
| `moe_gemm_2stage/gemm.py` | `74aaa6b012dda3962b3671d70f95aa6dfba175867e6daacc8cc6f509896e2a73` |
| `moe_gemm_2stage/gemm1.py` | `ad41f7692e4c35396ff625053b4f4c891f47fd714dc017593927c35eff74da3d` |
| `moe_gemm_2stage/gemm2.py` | `3e67678c361e3033d8fc175ef6324c0a3452669943856e4ef55ed06630e892d8` |
| `moe_gemm_2stage/gemm2_1x4.py` | `075f8c5e2cb6223c2ebeeb6e77806f0d04c18b7c63063cfd88d6e478bf7e021a` |
| `moe_gemm_2stage/gemm2_1x8.py` | `48ea64dec1ed9d92103f956bb818f39aa409d7a87c56fe107d74abb97bdb486a` |
| `moe_gemm_2stage/gemm2_2x4.py` | `b19857cccbdef0378622330d290d132dc65f3e4dc187b0968ef723f222db9f34` |
| `moe_gemm_2stage/gemm2_4x1.py` | `3d5af103996c00ba919512896daf944ad72677ca08e0ddac71ca161240348602` |
| `moe_gemm_2stage/gemm2_default.py` | `79ebe7ce2921c5e84034327d3ebdbcb4dcdf6d06d3cfc90bf8d740d13f06ac20` |
| `moe_gemm_2stage/layout_helpers.py` | `a72dc9ac9caa74b54d6027418ad5b9e68833cafda5a7141d2767d9b74018a42c` |
| `moe_gemm_2stage/moe_reduce.py` | `743dfe9dc1c4e1bade0a3771ff375ed183f5539090f713bc9aad112674136aa2` |
| `moe_gemm_2stage/quant.py` | `6876e2299806e2a0b9615bde122c3a9dcb6583b82bb4019532e7185aa0a55698` |

原42点4x1矩阵和3点ABBA48使用的`gemm2_4x1.py` SHA256为
`1d2ee22f85ead04ddc2529a2269108fb3f856b4158b4c038e52a4c553f89739f`；六项K256
流水优化对应`d1267f39f751680f790a30a7b4e4a4047e950621aaa8e7b1d4f883d32fc031af`。
本轮在`d1267f...`上否证defer1，再为Qwen35 K512 BM128晋级单B寄存器槽特化，得到
`0c2c5d...`；随后为Qwen35 K256/BM256合入balance93专属流水，并恢复非目标
CShuffle的原始生成顺序，得到最终`3d5af103...`。同一FlyDSL 0.3.2环境下fresh编译
证明K192、K384和K512规范化机器指令与`0c2c5d...`逐条一致，因此既有K512矩阵仍有效；
Qwen35 K256则由本报告后文的final-source矩阵覆盖。原矩阵与final-source复测记录的
`gemm2_1x8.py`和`gemm2_2x4.py` SHA256分别为`48ea64dc...`和`b19857cc...`；
这两个版本已经为generic/topology MLIR region使用独立`FlyObjCache`，未改kernel算法、
布局或运行时门禁。两条路径已另做fresh隔离复验并逐bit一致，结果见“正确性与结论”。

当前公开 `down_path` 及固定拓扑为：

| `down_path` | Kernel | 拓扑 |
| --- | --- | --- |
| `1x4_64x256` | `moe_2stage_down_prefill_1x4_64x256` | BM64、BN256、4 waves |
| `2x4` | `moe_2stage_down_prefill_2x4` | BM128、BN256、两个独立4-wave子组 |
| `1x8` | `moe_2stage_down_prefill_1x8` | BM64、BN512、8 waves |
| `4x1` | `moe_2stage_down_prefill_4x1` | BM128或BM256、BN64、4 waves沿M |
| `default` | 原有路径 | Qwen K=512生产配置保持不变 |

旧路径名不保留兼容别名。`TILE_M_DOWN`控制sorting与down任务覆盖范围，
`TILE_M_GATEUP`控制gateup任务大小；`2x4`和`4x1`允许down与gateup使用不同M tile。
BM128/BM256的`4x1`仍由down kernel直接消费对应metadata；gateup复用已有
`gateup_tasks_per_metadata = METADATA_TILE_SIZE_M / BLOCK_TILE_SIZE_M`机制，把一条
metadata任务展开成2/4个BM64任务，没有修改gateup kernel算法。

## 生产配置

性能case均显式指定tile、路径和padding，不调用自动down selector。

| Case | B / Hidden / Inter-TP / E / TopK | Quant | `TILE_M_DOWN / TILE_M_GATEUP / TILE_N` | `down_path` | Padding |
| --- | --- | --- | --- | --- | ---: |
| Hy3 | 32768 / 4096 / 192 / 193 / 9 | per-tensor | 64 / 64 / 128 | `1x8` | 0B |
| Qwen3.5 397B K=512 | 32768 / 4096 / 512 / 512 / 10 | PTPC | 64 / 64 / 256 | `default` | 默认 |
| Qwen3.5 397B K=256 | 32768 / 4096 / 256 / 512 / 10 | PTPC | 64 / 64 / 256 | `1x4_64x256` | 128B |
| Qwen3.5 35B K=512 | 32768 / 2048 / 512 / 256 / 8 | PTPC | 64 / 64 / 256 | `default` | 默认 |
| Qwen3.5 35B K=256 | 32768 / 2048 / 256 / 256 / 8 | PTPC | 256 / 64 / 256 | `4x1` | 128B |
| Xiaomi | 32768 / 6144 / 256 / 384 / 8 | PTPC | 64 / 64 / 256 | `1x4_64x256` | 128B |
| H3 | 32768 / 6144 / 384 / 128 / 4 | PTPC | 128 / 64 / 256 | `2x4` | 0B |

`Inter-TP`为tensor-parallel切分后的K维；`TILE_N`是host/gateup配置，专用down
kernel的固定BN见上一节。两个Qwen3.5 35B case使用TP1，其余均使用TP8。
Qwen3.5 35B K=256这一行仅描述B=32768的显式生产预设；1K--16K继续使用
`1x4_64x256`，边界依据见final-source矩阵和16K ABBA48。

## 测试协议

- GPU：AMD Instinct MI308X / gfx942，80 CU。
- 1800MHz performance determinism；PTL `Enabled / VECTOR,F8`；650W power cap。
- 10组buffer轮换，避免固定地址偏置；同进程平衡ABBA顺序。
- 最终path矩阵使用ABBA12；接近边界的case升至ABBA48。每轮先对同一版本的两个
  样本求均值，再计算同轮提升率，最后取各轮提升率中位数。
- ABBA48中候选相对当前或统一路径的配对提升率IQR跨0时视为持平，不因微小绝对
  中位数差异切换路径。
- phase包括`down`、`down + sorted_sum`（Combined）和完整链（Full）；完整链内包含gateup。
- `提升率 = 1 - candidate / control`；正值表示candidate更快，负值表示回退。
- 每轮检查reduced输出、finite、inactive tail及padding；测试后恢复performance
  level、PTL与NUMA状态。
- Down有效带宽是模型字节口径，不是PMC读取的HBM物理流量。设
  `R = B * TopK`、`A = min(R, E)`，固定计入FP8 activation输入`R * K`、
  BF16路由输出`2 * R * N`和实际活跃专家FP8权重`A * N * K`。PTPC额外计入
  per-channel weight scale `4 * A * N`和per-token activation scale`4 * R`；
  per-tensor额外计入`4 * A + 4`。不计sorting metadata、输出padding、量化和
  `sorted_sum`。`effective GB/s = effective bytes / Down秒数 / 1e9`。
- 本轮重新执行七个case的完整ABBA12矩阵，共42个Batch点、每路径/phase 24个样本；
  七份JSON清单SHA256为
  `134e948e3e3cd9ac0068d087d51b9bf3d9c249bd71887bccf2d27dffb5bfb336`。
  三个边界点另以ABBA48复核，每路径/phase 96个样本，JSON清单SHA256为
  `740aa34173bbcfc560e67f7ad6f0a5d0601f47204e0851849d1fddb61c903786`。
  临时矩阵harness SHA256为
  `59c7fc5aeb5e1081fca4ed781d6e06070e2754084de1de69de71dccb34c0232c`，未加入仓库。

## 1K-32K最终path矩阵

下表是七个正式case的最终选择，全部使用上节源码重新执行。主矩阵使用ABBA12；
Qwen 397B K=512 1K、Xiaomi 4K和H3 8K再以ABBA48复核，并以长采样结果覆盖表中
对应行。
ABBA48中，Qwen 397B 1K的`1x4`相对`default`在Down/Combined分别提升
`1.64% [0.67%, 2.35%]`和`1.12% [0.01%, 1.96%]`，但Full为
`0.60% [-5.34%, 6.76%]`，按端到端保守tie-break仍选择`default`。Xiaomi 4K的
`2x4`相对`1x4`在Down提升`0.93% [0.05%, 1.64%]`，Combined和Full的IQR跨0，
继续与同模型其余Batch统一使用`1x4_64x256`。H3 8K的`2x4`相对`1x4`在
Down/Combined分别提升`2.24% [1.81%, 2.53%]`和
`1.77% [1.38%, 2.28%]`，保留`2x4`。

final-source补测中，Qwen35 K256 16K的BM256 `4x1`相对`1x4_64x256`在ABBA48的
Down/Combined分别提升`2.65% [2.14%, 3.50%]`和
`1.84% [1.40%, 2.16%]`，但E2E为`0.87% [-1.45%, 3.38%]`、仅24/48轮胜，
因此16K仍保留`1x4_64x256`。32K direct ABBA24的Down/Combined/E2E IQR均不跨0，
三者均24/24轮由BM256 `4x1`胜出。

`Down`列为“绝对中位延迟ms / 有效TFLOPS / effective GB/s /
相对同轮default的配对提升率”；
`Full`列为“绝对中位延迟ms / 配对提升率”；`Combined`为down与`sorted_sum`的绝对
中位延迟。完整链包括sorting、两次activation quant、gateup、down、invert和
`sorted_sum`。Down有效FLOPs按`2 * B * TopK * Hidden * Inter-TP`计算；default相对
自身的提升率固定为0。

| Case | Batch | Path | Down ms / TFLOPS / effective GB/s / 提升率 | Combined ms | Full ms / 提升率 |
| --- | ---: | --- | ---: | ---: | ---: |
| Hy3 K=192 | 1K | `1x8` | 0.1173 / 123.6 / 1952.3 / 12.5% | 0.1527 | 0.4085 / 2.2% |
|  | 2K | `1x8` | 0.1874 / 154.7 / 1634.2 / 14.8% | 0.2468 | 0.5825 / 5.3% |
|  | 4K | `1x8` | 0.2402 / 241.4 / 1918.4 / 21.0% | 0.3438 | 0.8245 / 7.9% |
|  | 8K | `1x8` | 0.4072 / 284.8 / 1890.8 / 21.1% | 0.5976 | 1.4340 / 6.8% |
|  | 16K | `1x8` | 0.7263 / 319.3 / 1911.2 / 20.9% | 1.1001 | 2.6205 / 7.8% |
|  | 32K | `1x8` | 1.3513 / 343.3 / 1942.1 / 18.0% | 2.0575 | 5.1749 / 7.1% |
| Qwen3.5 397B K=512 | 1K | `default` | 0.5387 / 79.7 / 2174.4 / 0.0% | 0.5727 | 1.3009 / 0.0% |
|  | 2K | `default` | 0.5433 / 158.1 / 2319.9 / 0.0% | 0.6075 | 1.4512 / 0.0% |
|  | 4K | `default` | 0.8257 / 208.1 / 1742.5 / 0.0% | 0.9445 | 2.3421 / 0.0% |
|  | 8K | `default` | 1.1840 / 290.2 / 1516.5 / 0.0% | 1.3933 | 3.4071 / 0.0% |
|  | 16K | `default` | 1.7617 / 390.1 / 1424.1 / 0.0% | 2.1651 | 5.5929 / 0.0% |
|  | 32K | `default` | 3.3656 / 408.4 / 1169.3 / 0.0% | 4.2034 | 10.9304 / 0.0% |
| Qwen3.5 397B K=256 | 1K | `1x4_64x256` | 0.2832 / 75.8 / 2231.3 / 24.2% | 0.3208 | 0.6984 / 11.2% |
|  | 2K | `1x4_64x256` | 0.2855 / 150.5 / 2516.5 / 23.6% | 0.3469 | 0.7962 / 8.9% |
|  | 4K | `1x4_64x256` | 0.4765 / 180.3 / 1870.8 / 27.4% | 0.6046 | 1.3652 / 11.0% |
|  | 8K | `1x4_64x256` | 0.6477 / 265.2 / 1910.8 / 27.0% | 0.8923 | 2.0602 / 10.1% |
|  | 16K | `1x4_64x256` | 1.0134 / 339.1 / 1904.6 / 25.6% | 1.4840 | 3.4910 / 8.4% |
|  | 32K | `1x4_64x256` | 1.8804 / 365.5 / 1762.8 / 26.9% | 2.8139 | 6.6479 / 8.5% |
| Qwen3.5 35B K=512 | 1K | `default` | 0.1484 / 115.8 / 2077.9 / 0.0% | 0.1625 | 0.3782 / 0.0% |
|  | 2K | `default` | 0.1510 / 227.5 / 2292.0 / 0.0% | 0.1726 | 0.4410 / 0.0% |
|  | 4K | `default` | 0.2354 / 292.0 / 1791.5 / 0.0% | 0.2777 | 0.7248 / 0.0% |
|  | 8K | `default` | 0.3912 / 351.3 / 1464.2 / 0.0% | 0.4727 | 1.3067 / 0.0% |
|  | 16K | `default` | 0.7163 / 383.7 / 1221.6 / 0.0% | 0.8928 | 2.5048 / 0.0% |
|  | 32K | `default` | 1.3824 / 397.7 / 1070.3 / 0.0% | 1.7292 | 4.8909 / 0.0% |
| Qwen3.5 35B K=256 | 1K | `1x4_64x256` | 0.0808 / 106.3 / 2128.2 / 26.5% | 0.0910 | 0.2266 / 11.7% |
|  | 2K | `1x4_64x256` | 0.0825 / 208.1 / 2516.2 / 25.5% | 0.0998 | 0.2665 / 10.2% |
|  | 4K | `1x4_64x256` | 0.1383 / 248.5 / 2018.3 / 22.8% | 0.1760 | 0.4522 / 9.2% |
|  | 8K | `1x4_64x256` | 0.2367 / 290.4 / 1782.3 / 21.7% | 0.3185 | 0.8124 / 6.9% |
|  | 16K | `1x4_64x256` | 0.4350 / 316.0 / 1626.0 / 17.8% | 0.6036 | 1.5512 / 5.7% |
|  | 32K | `4x1` BM256 | 0.7601 / 361.7 / 1681.7 / 20.6% | 1.0855 | 2.9368 / 7.6% |
| Xiaomi K=256 | 1K | `1x4_64x256` | 0.3037 / 84.8 / 2358.0 / 27.2% | 0.3461 | 0.7543 / 12.7% |
|  | 2K | `1x4_64x256` | 0.3006 / 171.4 / 2724.2 / 27.0% | 0.3753 | 0.8951 / 10.0% |
|  | 4K | `1x4_64x256` | 0.5187 / 198.7 / 1975.3 / 29.0% | 0.6582 | 1.4999 / 12.3% |
|  | 8K | `1x4_64x256` | 0.7401 / 278.5 / 1939.8 / 27.2% | 1.0077 | 2.2751 / 11.1% |
|  | 16K | `1x4_64x256` | 1.3241 / 311.4 / 1705.4 / 28.6% | 1.8393 | 4.2060 / 10.5% |
|  | 32K | `1x4_64x256` | 2.2256 / 370.5 / 1753.6 / 27.0% | 3.2858 | 7.7623 / 10.8% |
| H3 K=384 | 1K | `1x4_64x256` | 0.1631 / 118.5 / 2188.9 / 14.7% | 0.1858 | 0.4161 / 6.0% |
|  | 2K | `1x4_64x256` | 0.1626 / 237.7 / 2514.6 / 14.5% | 0.2049 | 0.4934 / 6.0% |
|  | 4K | `1x4_64x256` | 0.3024 / 255.7 / 1695.9 / 9.2% | 0.3843 | 0.8605 / 3.3% |
|  | 8K | `2x4` | 0.5123 / 301.8 / 1406.5 / 4.5% | 0.6534 | 1.5442 / 1.3% |
|  | 16K | `2x4` | 0.8631 / 358.3 / 1316.0 / 7.8% | 1.1443 | 2.8019 / 3.0% |
|  | 32K | `2x4` | 1.5850 / 390.2 / 1240.7 / 10.1% | 2.1409 | 5.3202 / 3.7% |

最终选择因此不是单一全局path：Hy3全Batch选`1x8`，Xiaomi全Batch选
`1x4_64x256`；H3在1K-4K选`1x4_64x256`、8K-32K选`2x4`；两个Qwen K=512
case保持`default`，Qwen397 K=256全Batch选`1x4_64x256`；Qwen35 K=256在1K--16K
选`1x4_64x256`，32K选BM256 `4x1`加128B padding。Qwen3.5 397B K=256的Down
提升率为23.6%-27.4%，Full提升率为8.4%-11.2%；final-source Qwen3.5 35B K=256
的Down提升率为17.8%-26.5%，Full提升率为5.7%-11.7%。H3 8K的`2x4` Down和
Full提升率分别为`4.5%`和`1.3%`。最终path的Down有效带宽范围为
`1070.3-2724.2 GB/s`。

## 4x1 down-projection新增路径

### 实现与调度

新增生产`down_path="4x1"`，固定BN64、256 threads和4 waves沿M，支持BM128/BM256、
BK128/BK192、PTPC/per-tensor。A的完整K维在进入N循环前以64-bit load预取到VGPR；
B由256线程协作执行128-bit global load，再写入两个LDS slot做BK级双缓冲。当前
Qwen35 N2048/K512/BM128例外使用单个B寄存器槽，避免四个K stage展开时两个完整B块
同时驻留。Epilogue复用B的LDS存储做CShuffle：同lane的两个N group先缩放并打包成
128-bit LDS record，consumer读取两个64-bit half并拼成128-bit全局写；相对直接64-bit
store，单tile全局store数从256降至128。

LDS别名生命周期由CShuffle前后barrier保护；MFMA后的`sched_barrier(0)`阻止LLVM把
后续`vmcnt(0)`提前合并到MFMA前。下一N tile提交B之前只等待必要的输出VMEM额度，
允许上一tile的store继续在飞。MI308X任务映射按4 XCC、每XCC 4 SE、每SE 5 CU做
双射，不能整除16 SE的尾部保持identity，并把logical XCC旋转2位以改善权重L2局部性。

K512的N深度调度保持不变：BM128在`task_num < 768`时每WG处理8个N tile，否则处理
完整N；BM256固定每WG处理16个N tile。full/short两个MLIR region使用独立
`FlyObjCache`。六项流水的双B寄存器槽使Qwen35 K512 BM128 full/short上升到
176/176 VGPR并跌到Q2；当前专属单槽流水恢复为168/164 VGPR、16,896B LDS、0 private，
重新达到Q3。该条件只覆盖`N=2048 && K=512 && BM=128`；同驱动fresh compile证明
Qwen397 K512 BM128、K512 BM256和K256 BM256的非目标机器码保持不变。K192/K256/K384
保持full-N；把short-N开放到这些K的实验在Qwen35 K256 1K/32K均无收益，已撤回。

正式矩阵首次fresh compile旧`2x4`/`1x8`的generic和topology双特化时，还暴露了两者
共用`FlyObjCache`导致跨MLIR region复用SSA的问题。本轮只把两个特化拆为独立cache；
kernel算法、布局和运行时门禁不变，独立`hy3/p2`与`hy3/p3`编译执行均通过。

### 历史42点ABBA12矩阵（`1d2ee...`）

协议与前文一致，使用GPU0、650W、1800MHz performance determinism、PTL
`Enabled / VECTOR,F8`、10组buffer。七个case各测1K--32K六档Batch，共42点；
每路径/phase为ABBA12的24个event样本。`D/C/F`依次表示Down、Combined和Full中位
延迟ms；“相对既有”使用每轮两个样本均值计算配对提升率，正值表示4x1更快。最佳4x1
按Down选择BM128或BM256；useful/executed TFLOPS及effective GB/s沿用前文口径。
带`*`的三行由下一节ABBA48结果覆盖。

| Case | B | 最快既有path / Down ms | BM128 D/C/F ms | BM256 D/C/F ms | 最佳4x1 D: useful/executed TFLOPS, GB/s | 相对既有 D/C/F |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Hy3 K=192 | 1K | `1x8` / 0.1145 | 0.2844/0.3143/0.6129 | 0.4658/0.4975/0.9201 | **BM128**: 51.0/136.6, 805 | -148.1%/-129.8%/-52.2% |
|  | 2K | `1x8` / 0.1812 | 0.2808/0.3343/0.6758 | 0.4638/0.5162/0.9750 | **BM128**: 103.2/138.4, 1091 | -55.2%/-41.6%/-15.8% |
|  | 4K | `1x8` / 0.2495 | 0.4105/0.5122/1.0550 | 0.4715/0.5738/1.1087 | **BM128**: 141.3/189.3, 1123 | -64.6%/-46.8%/-23.9% |
|  | 8K | `1x8` / 0.4246 | 0.5420/0.7337/1.5794 | 0.6779/0.8841/1.8169 | **BM128**: 214.0/215.1, 1421 | -28.8%/-18.8%/-8.2% |
|  | 16K | `1x8` / 0.7711 | 1.0321/1.4277/2.9699 | 0.9211/1.3080/2.8526 | **BM256**: 251.8/253.1, 1507 | -19.3%/-12.7%/-5.4% |
|  | 32K | `1x8` / 1.3789 | 1.8450/2.6179/5.7748 | 1.7758/2.5420/5.6856 | **BM256**: 261.2/262.6, 1478 | -28.8%/-17.9%/-6.9% |
| Qwen397 K=512 | 1K | `1x4_64x256` / 0.4927 | 0.9450/0.9887/2.1454 | 1.6019/1.6517/3.7760 | **BM128**: 45.5/290.9, 1240 | -90.6%/-85.3%/-77.5% |
|  | 2K | `1x4_64x256` / 0.5009 | 0.9912/1.0645/2.2680 | 1.6475/1.7123/3.9147 | **BM128**: 86.7/277.3, 1272 | -96.9%/-87.3%/-69.0% |
|  | 4K | `default` / 0.8049 | 0.9807/1.1051/2.5080 | 1.5910/1.7111/4.1478 | **BM128**: 175.2/280.3, 1467 | -21.2%/-19.7%/-8.4% |
|  | 8K | `default` / 1.1553 | 1.6885/1.9067/4.4486 | 1.6051/1.8171/4.3757 | **BM256**: 214.1/342.5, 1119 | -39.1%/-33.3%/-29.6% |
|  | 16K | `default` / 1.7924 | 2.4476/2.8401/6.7105 | 3.1793/3.6042/8.5638 | **BM128**: 280.8/336.9, 1025 | -37.3%/-30.5%/-19.1% |
|  | 32K* | `default` / 3.3603 | 4.0007/4.8164/11.6170 | 4.8636/5.6938/13.5325 | **BM128**: 343.5/343.5, 984 | -19.4%/-15.7%/-5.9% |
| Qwen35 K=512 | 1K | `default` / 0.1478 | 0.2453/0.2587/0.5996 | 0.4291/0.4441/1.0767 | **BM128**: 70.0/280.2, 1257 | -66.0%/-60.7%/-57.9% |
|  | 2K | `default` / 0.1513 | 0.2599/0.2813/0.6544 | 0.4378/0.4614/1.1345 | **BM128**: 132.2/264.4, 1332 | -72.1%/-63.2%/-49.9% |
|  | 4K | `default` / 0.2291 | 0.2781/0.3140/0.7649 | 0.4430/0.4869/1.2007 | **BM128**: 247.1/247.1, 1516 | -21.2%/-15.7%/-5.9% |
|  | 8K | `default` / 0.3914 | 0.5003/0.5824/1.4134 | 0.4733/0.5534/1.3841 | **BM256**: 290.4/290.4, 1210 | -20.2%/-16.6%/-6.5% |
|  | 16K* | `default` / 0.7108 | 0.8287/1.0081/2.6078 | 0.8258/1.0047/2.5959 | **BM256**: 332.8/332.8, 1060 | -16.2%/-13.2%/-4.5% |
|  | 32K* | `default` / 1.3660 | 1.5475/1.8890/5.0192 | 1.5838/1.9238/5.0516 | **BM128**: 355.2/355.2, 956 | -13.2%/-10.4%/-3.9% |
| Qwen397 K=256 | 1K | `1x4_64x256` / 0.2672 | 0.6357/0.6736/1.2690 | 1.0629/1.0947/2.2120 | **BM128**: 33.8/216.2, 994 | -135.2%/-119.9%/-85.3% |
|  | 2K | `1x4_64x256` / 0.2725 | 0.6416/0.7052/1.3471 | 1.0658/1.1358/2.3014 | **BM128**: 66.9/214.2, 1120 | -134.9%/-111.3%/-74.2% |
|  | 4K | `1x4_64x256` / 0.4650 | 0.6395/0.7552/1.5173 | 1.0646/1.1815/2.4938 | **BM128**: 134.3/214.9, 1394 | -37.7%/-27.8%/-12.0% |
|  | 8K | `1x4_64x256` / 0.6788 | 1.1177/1.3457/2.7655 | 1.0656/1.2999/2.7240 | **BM256**: 161.2/258.0, 1161 | -58.4%/-42.1%/-32.1% |
|  | 16K | `1x4_64x256` / 1.0938 | 1.5671/1.9914/4.1535 | 1.9121/2.3581/5.0908 | **BM128**: 219.3/263.1, 1232 | -45.2%/-28.0%/-18.3% |
|  | 32K | `1x4_64x256` / 1.9481 | 2.4467/3.2862/7.0907 | 2.7826/3.6227/7.9485 | **BM128**: 280.9/280.9, 1355 | -25.9%/-13.6%/-5.2% |
| Qwen35 K=256 | 1K | `1x4_64x256` / 0.0791 | 0.2026/0.2136/0.4113 | 0.3165/0.3313/0.6606 | **BM128**: 42.4/169.6, 849 | -155.3%/-138.6%/-77.7% |
|  | 2K | `1x4_64x256` / 0.0819 | 0.2050/0.2233/0.4560 | 0.3200/0.3427/0.7100 | **BM128**: 83.8/167.6, 1013 | -150.6%/-124.5%/-69.7% |
|  | 4K | `1x4_64x256` / 0.1346 | 0.2112/0.2478/0.5254 | 0.3282/0.3718/0.7805 | **BM128**: 162.7/162.7, 1321 | -55.9%/-41.8%/-16.5% |
|  | 8K | `1x4_64x256` / 0.2361 | 0.3279/0.4082/0.9067 | 0.3388/0.4187/0.9152 | **BM128**: 209.6/209.6, 1286 | -38.1%/-28.4%/-11.4% |
|  | 16K | `1x4_64x256` / 0.4330 | 0.5474/0.7269/1.6777 | 0.5545/0.7343/1.6885 | **BM128**: 251.1/251.1, 1292 | -26.9%/-20.7%/-7.9% |
|  | 32K | `1x4_64x256` / 0.8081 | 0.9861/1.3300/3.2114 | 0.9598/1.3124/3.1948 | **BM256**: 286.4/286.4, 1332 | -19.2%/-14.1%/-5.9% |
| Xiaomi K=256 | 1K | `1x4_64x256` / 0.2908 | 0.6859/0.7257/1.3831 | 1.1919/1.2387/2.4441 | **BM128**: 37.6/225.4, 1044 | -135.2%/-117.1%/-85.3% |
|  | 2K | `1x4_64x256` / 0.2879 | 0.6887/0.7581/1.4762 | 1.2056/1.2824/2.5731 | **BM128**: 74.8/224.5, 1189 | -137.9%/-108.4%/-73.4% |
|  | 4K | `1x4_64x256` / 0.5289 | 0.6932/0.8365/1.6722 | 1.1992/1.3568/2.8019 | **BM128**: 148.7/223.0, 1478 | -30.4%/-24.6%/-11.2% |
|  | 8K | `1x4_64x256` / 0.7573 | 1.2780/1.5707/3.1187 | 1.1870/1.4811/3.0358 | **BM256**: 173.7/260.5, 1210 | -57.8%/-43.3%/-32.9% |
|  | 16K | `1x4_64x256` / 1.3711 | 1.8091/2.3253/4.6996 | 2.0671/2.6020/5.5555 | **BM128**: 227.9/256.4, 1248 | -32.9%/-22.0%/-8.7% |
|  | 32K | `1x4_64x256` / 2.3121 | 3.4374/4.4101/9.1512 | 3.2119/4.1894/8.8136 | **BM256**: 256.7/288.8, 1215 | -38.1%/-25.8%/-12.5% |
| H3 K=384 | 1K | `1x4_64x256` / 0.1573 | 0.3445/0.3675/0.7037 | 0.5470/0.5698/1.2124 | **BM128**: 56.1/224.4, 1037 | -121.1%/-109.6%/-70.3% |
|  | 2K | `1x4_64x256` / 0.1589 | 0.3428/0.3853/0.7764 | 0.5517/0.5909/1.2680 | **BM128**: 112.7/225.5, 1193 | -118.1%/-98.8%/-59.5% |
|  | 4K | `1x4_64x256` / 0.2980 | 0.3470/0.4174/0.9033 | 0.5542/0.6368/1.3619 | **BM128**: 222.8/222.8, 1478 | -16.5%/-13.7%/-6.6% |
|  | 8K | `1x4_64x256` / 0.5208 | 0.6759/0.8169/1.7097 | 0.5615/0.7009/1.5933 | **BM256**: 275.4/275.4, 1283 | -8.0%/-6.4%/-3.3% |
|  | 16K | `1x4_64x256` / 0.9526 | 1.2253/1.5050/3.1547 | 1.0231/1.3081/2.9533 | **BM256**: 302.3/302.3, 1110 | -7.9%/-6.3%/-2.2% |
|  | 32K | `default` / 1.7360 | 2.2080/2.7556/6.0113 | 1.8639/2.4216/5.6401 | **BM256**: 331.8/331.8, 1055 | -6.7%/-4.9%/-2.0% |

七份正式JSON按“文件名 + SHA256”排序后的清单SHA256为
`5d69ca7c5b60284e8b9cc004a4259546ca55ff58efd63e072bc1cf741f7f7617`；临时harness
SHA256为`f5e3d66786f4707e14849763f1517e2c2f7b7b20fd9125db333b5b8117db7c7c`，未加入
仓库。JSON内记录的facade和package源码哈希已逐文件与原矩阵快照复核，七个case均
MATCH；当前4x1源码已由后文六项流水专项结果取代。42点全部为24样本，Down最大
relative-L2为`0.004914`，Full最大值为
`0.005431`；全部finite，inactive tail和padding均未写。H3的Full门禁使用`0.006`，
只容纳两次FP8量化与归约后的累计差异；Down门禁仍为`0.005`。五组正式4x1参数化
正确性测试覆盖BM128/BM256、BK128/BK192、PTPC/per-tensor，以及新增的
BM128/N2048/K512/PTPC单槽特化，结果`5 passed`。

### Qwen35 K=256：4x1预测与实测

4x1 core ceiling使用`256x4096x2048x256`均衡shape、`BM256xBN64xBK128`、
`WMxWN=4x1`、Q2和`NT/WG=8`；4个wave各加载四分之一B。模拟忽略B写入LDS、
跨wave barrier、LDS读回、真实MFMA累加依赖、scale、routing metadata和CShuffle。
50个无插桩样本得到`2.3241ms / 473.10 useful TFLOPS`；独立PMC双pass得到
`52.1/1590.5/1642.6 GB/s`（物理读/写/总）。该值是core co-issue上界，不是生产
kernel的逐Batch性能预测。

生产矩阵使用10组buffer和ABBA12，每个path/phase 24个event样本。下表的1x4带宽和
“最佳4x1带宽”均为模型effective GB/s，不可与上述PMC物理带宽直接相除；4x1列依次为
`ms / useful TFLOPS / executed TFLOPS`。

| Batch | 1x4 ms / useful TFLOPS / effective GB/s | 4x1 BM128 | 4x1 BM256 | 最佳4x1 / effective GB/s | 最佳4x1相对1x4 TFLOPS | ceiling达到率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.0791 / 108.6 / 2174 | 0.2026 / 42.4/169.6 | 0.3165 / 27.1/217.1 | BM128 / 849 | -61.0% | 9.0% |
| 2K | 0.0819 / 209.7 / 2535 | 0.2050 / 83.8/167.6 | 0.3200 / 53.7/214.8 | BM128 / 1013 | -60.0% | 17.7% |
| 4K | 0.1346 / 255.3 / 2073 | 0.2112 / 162.7/162.7 | 0.3282 / 104.7/209.4 | BM128 / 1321 | -36.3% | 34.4% |
| 8K | 0.2361 / 291.1 / 1787 | 0.3279 / 209.6/209.6 | 0.3388 / 202.9/202.9 | BM128 / 1286 | -28.0% | 44.3% |
| 16K | 0.4330 / 317.4 / 1633 | 0.5474 / 251.1/251.1 | 0.5545 / 247.9/247.9 | BM128 / 1292 | -20.9% | 53.1% |
| 32K | 0.8081 / 340.1 / 1582 | 0.9861 / 278.8/278.8 | 0.9598 / 286.4/286.4 | BM256 / 1332 | -15.8% | 60.5% |

若只按core ceiling，4x1应比32K的1x4高`39.1%`；实际BM256反而低`15.8%`、延迟高
`18.8%`，只达到ceiling的`60.5%`，相对上界仍有`39.5%`缺口。低Batch还叠加BM
padding：1K时1x4、BM128和BM256分别执行useful行数的2x、4x和8x；因此BM128虽比
BM256快，useful TFLOPS仍只有42.4T。4K以后最佳4x1的useful/executed相等，剩余差距
已不是padding造成。

#### 性能不及预测原因

对最终最快的32K BM256路径重新采集ATT。目标dispatch为
`moe_2stage_down_prefill_4x1_0`，ATT运行测得`0.9592ms / 286.58T`，与正式矩阵
`0.9598ms / 286.38T`一致；数值、finite、tail和padding门禁全部通过。权威资源为
`80 VGPR + 128 AGPR`、112 SGPR、17,408B LDS、0 scratch；combined VGPR为208，
因此只能驻留2 waves/SIMD，达到3 waves需要不超过170。

ATT覆盖12,815条指令、37.73M cycles和28.68M stall cycles，总stall率为76.0%。stall
分类为MFMA/FMA 29.9%、VMEM wait/load/store 15.7%/10.7%/7.1%、LDS wait/LDS/barrier
12.1%/8.5%/6.1%，其余10.1%。因此差距不是单一HBM带宽问题，而是以下成本叠加：

- 模拟用4个write-only AGPR目标且C输入固定为0；生产保留128个真实累加AGPR，MFMA
  存在RAW链，形成最大单类stall并把occupancy锁在Q2；
- 模拟只计四分之一B VMEM读取；生产需要`global -> VGPR -> LDS`批量提交、跨wave同步，
  再执行LDS读回供MFMA消费。VMEM与LDS路径合计贡献约54.1%的stall；
- BN64令完整N包含32个细粒度tile，反复执行B双缓冲交接、wait和barrier；模拟省略这些
  生产者/消费者依赖；
- 生产还包含PTPC scale、routing weight、BF16舍入与CShuffle。CShuffle已把输出合并为
  128-bit store，但仍增加LDS读写和同步；模拟D payload与MFMA结果完全独立。

最终UI已复制到仓库当前目录
[`ui_output_agent_29195_dispatch_147`](../../../ui_output_agent_29195_dispatch_147)，
并附带`out_kernel_trace.csv`、`benchmark.json`和`input.yaml`，可直接运行热点分析器。
`code.json`和benchmark的SHA256分别为
`080bcb503ea42e60c55c6b490f9a4c0c604967aad8ce0605a20a3a1ed1d2d643`和
`1d124a4512f3714ed9db074adc8a1c680cb4e13ffa8e953d58faf4163784b8c2`。

#### 六项流水优化复测

在上述稳定基线上继续完成六项流水修正：routing/A scale在A预载前读取；最后一个K
stage把`W(N+1,K0)`提前到当前B LDS wait之前并依靠buffer descriptor做末尾OOB裁剪；
B global预取使用两个VGPR槽和partial `vmcnt`；MFMA/非MFMA区间显式使用
`s_setprio 3/0`；weight scale与BF16转换使用标量`v_fma_f32`/`v_fmaak_f32`，不再生成
`v_pk_mul_f32`；CShuffle由同lane两个N group组成128-bit LDS record，再由两次64-bit
LDS读拼成128-bit global store。最终ISA中`ds_write_b128=96`、`ds_write_b64=0`、
`ds_bpermute=0`、`v_pk_mul_f32=0`。重复N/K热段使用`vmcnt(2)`等partial wait；仅metadata、
A预载、首块B和最终drain各保留一次`vmcnt(0)`。

相同GPU0、1800MHz、PTL `Enabled / VECTOR,F8`、10 buffer、ABBA12协议下，Qwen35 K256
32K BM256的Down从历史`0.959164ms / 286.58T`降到
`0.876084ms / 313.76T`，时延降低`8.66%`、吞吐提高`9.48%`；相对1x4的吞吐差距从
`-15.8%`缩小到`-7.76%`，core ceiling达到率从`60.5%`提高到`66.32%`。24个样本范围
为`0.871324--0.888244ms`，Down/E2E finite、tail和padding门禁全部通过。结果JSON
SHA256为`fee7e187cc1ac1bbaad107741901a1593c6cbbe246a37783ae0e761be4c09d5a`。

fresh ATT对应`0.871123ms / 315.54T`，权威资源为`100 VGPR + 132 AGPR`、112 SGPR、
17,408B LDS、0 scratch，仍维持2 waves/SIMD。相同156个完整wave、每wave 4,096条MFMA
下，总cycle从37.73M降到34.48M，总stall从28.68M降到25.45M，stall率从76.0%降到
73.8%。`vmcnt` stall从4.49M降到0.97M；旧`v_pk_mul_f32`的2.60M stall被标量FMA的
0.23M替代。物理SIMD双驻留wave账本显示steady MFMA并集从`59.04%`升到`66.86%`，
VALU与peer MFMA共发射从0.44M增至1.59M cycles，证明MFMA/非MFMA双stage确实形成。
剩余steady idle以structural tail `53.91%`和LDS stall/wait `24.99%`为主。

优化后UI位于
[`ui_output_agent_30308_dispatch_147`](../../../ui_output_agent_30308_dispatch_147)，
附带`benchmark.json`、`out_kernel_trace.csv`、`input.yaml`和`physical_slots.json`；
`code.json` SHA256为
`6b110ba1ed99c4a208b6ecd0ff7129ab35a69ec048246fa7e9978789e6dbb49f`。

基于新ATT还实现了只把最后一个CShuffle row-pair延迟到下一N非MFMA stage的8KB额外LDS
原型，源码SHA256为
`7c90cec0d3f6765edcb70ed8cc1273ee477a0404c7c26fea3e567f74a78bc3ca`；BM128
column-code为`relative_l2=0`，BM256 K256 random与BK192回归分别为`0.003307`和
`0.003310`，LDS为25,600B且仍为Q2。GPU恢复空闲后，以`d1267f...`为control执行同进程
10-buffer ABBA4：10份physical/reduced output全部逐bit一致，但Down从`0.877323ms`
退化到`0.902684ms`，paired ratio为`1.028506`，IQR `1.013007--1.032123`，0/4胜。
因此defer1被正式否证，不采ATT、不进入生产；结果JSON SHA256为
`25e9eabaeab7908debc57b061b20a9e4a813987d2fdeba479d0b8a017ca5d70d`。

#### balance93 K256 production晋级

最终`3d5af103...`只在`N=2048、K=256、BM256、BK128、PTPC`启用专属A128读取、
partial LDS wait、weight-scale流水、CShuffle `read -> pack -> consume`交叠和NT输出；
非目标路径恢复原始CShuffle函数体。同一FlyDSL 0.3.2环境下，K192、K384、K512
相对pre-balance93的规范化ISA分别为13,142、47,145、14,057条，逐条相同且资源不变；
目标K256为13,435条、240 combined VGPR、25,600B LDS、0 private/scratch、Q2，
并含256条静态`buffer_store_dwordx4 nt`。

GPU7上的final-source direct ABBA24使用1800MHz determinism、PTL
`Enabled / VECTOR,F8`、650W、NUMA off和10 buffers。Qwen35 K256的Down有效工作量为
`2 * B * TopK * Hidden * Inter-TP = 274,877,906,944 FLOP`，完整E2E三个GEMM为
`824,633,720,832 FLOP`；Combined的TFLOPS仍只按Down工作量计算。结果如下：

| Phase | `1x4_64x256` ms / useful TFLOPS | BM256 `4x1` ms / useful TFLOPS | 配对提升率中位 / IQR | 胜率 |
| --- | ---: | ---: | ---: | ---: |
| Down | 0.819303 / 335.50 | 0.766544 / 358.59 | 6.35% / `[6.13%, 6.58%]` | 24/24 |
| Combined | 1.151685 / 238.67 | 1.079325 / 254.68 | 6.09% / `[5.88%, 6.36%]` | 24/24 |
| E2E | 3.005093 / 274.41 | 2.927592 / 281.68 | 2.59% / `[1.23%, 3.93%]` | 24/24 |

三路径六档ABBA12使用相同协议，每路径/phase 24个event样本；1K--16K最终选择
`1x4_64x256`，32K选择BM256 `4x1`加128B padding。16K另由上文ABBA48保守判为
E2E持平；32K另由direct ABBA24确认三个phase稳定胜出。三份结果JSON SHA256分别为：

- 六档`default/p1/p4`矩阵：
  `b252471dea1c826cc2a84f9e869c68e77e85753c22a1542b868273c9ac31ba74`；
- 32K direct ABBA24：
  `89b915471aac54c7eb540df768f81370c5d0955c4e24341a2149d6218cc0c9a4`；
- 16K ABBA48：
  `b3257dede277a738f2343b263b4cc7069cb3f4faddda60a5ef00032489f21a57`。

column-code为`relative_l2=0`；随机routing、activation scale和weight scale专项为
`relative_l2=0.003316`、全finite；正式`4x1`参数化正确性5/5通过。矩阵与direct
ABBA的Down/E2E相对default均`relative_l2=0`，inactive tail和padding保持未写。

final-source fresh ATT位于
[`ui_output_agent_42238_dispatch_147`](../../../ui_output_agent_42238_dispatch_147)，
旧`ui_output_agent_30308_dispatch_147`保留。目标dispatch 147为`0.738603ms`，资源为
`112 VGPR + 128 AGPR`、112 SGPR、25,600B LDS、0 scratch；13,418条指令中13,417条
有源码映射，268个wave文件覆盖四个shader engine。物理slot账本的steady MFMA union为
`74.34%`、lifecycle busy为`66.42%`；K0/A1/K1/A0 span中位数为
`1024/440/2136/992 cycles`。剩余steady idle主要是VALU调度`28.44%`、structural tail
`17.11%`和LDS stall/wait `16.67%`。UI内附benchmark、原始kernel trace、slot/stage
账本、hotspot报告和三份性能JSON；`code.json`与ATT benchmark SHA256分别为
`06ea15dbf893a517dd9ffdab416787d94868ff02001952faa8daa61daf3858d0`和
`570d548a231799cf7f96b0146e6b9bb08ac1b01bb0dc2f626efaa8564acf2e66`。

所有final-source性能与ATT运行结束后，GPU7均恢复为`auto`、PTL
`Enabled / VECTOR,F8`、650W、NUMA off；sysfs busy为0且无遗留设备枚举进程。

### 历史ABBA48边界复核（`1d2ee...`）

对Qwen35 K512 16K/32K及Qwen397 K512 32K使用相同10组buffer升至ABBA48，
每路径/phase为96个样本。下表是候选相对`1x4_64x256`的配对提升率中位数和IQR；
最后一列是相对`default`的D/C/F中位提升率。

| Case / B | 4x1 | Down vs 1x4 | Combined vs 1x4 | Full vs 1x4 | vs default D/C/F |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen35 K512 / 16K | BM256 | +2.04% `[+1.70%, +2.71%]` | +0.60% `[+0.00%, +1.01%]` | +0.09% `[-0.84%, +1.10%]` | -16.15%/-13.24%/-4.52% |
| Qwen35 K512 / 32K | BM128 | +6.43% `[+6.25%, +6.61%]` | +4.51% `[+4.37%, +4.65%]` | +1.57% `[+0.98%, +2.03%]` | -13.17%/-10.37%/-3.91% |
| Qwen397 K512 / 32K | BM128 | -1.16% `[-1.93%, -0.68%]` | +0.91% `[+0.17%, +1.54%]` | +1.04% `[+0.67%, +1.24%]` | -19.41%/-15.70%/-5.90% |

两份ABBA48 JSON清单SHA256为
`d4d921131821805591f568fdf3f0de5f8cf2c3b5d0279b2e9cad312b482c8a00`。因此4x1
在Qwen35 K512 32K稳定超过`1x4_64x256`，16K的Down也稳定更快但Full持平；
Qwen397 K512 32K的Down略慢、Combined/Full略快。三点均仍慢于`default`，所以不应
把“超过1x4”表述为超过当前全局最快生产路径。

### 当前哈希K512全矩阵复测

K512表原始测量源码为`0c2c5d...`；final-source `3d5af103...`的K512规范化机器指令
与其逐条相同，因此结果继续有效。协议为相同1800MHz、PTL `Enabled / VECTOR,F8`、
10-buffer ABBA12，每路径/phase有24个event样本。`D/C/F`依次为Down、Combined和完整
E2E中位延迟ms；两组矩阵的Down和E2E相对default均逐bit一致，全部finite、tail clean、
padding clean。

| Qwen35 K512 Batch | default D/C/F | `1x4_64x256` D/C/F | 4x1 BM128 D/C/F | 4x1 BM256 D/C/F | D/C/F胜者 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1K | 0.1482/0.1608/0.3816 | 0.1517/0.1640/0.3816 | 0.2324/0.2451/0.5868 | 0.5817/0.5969/1.2314 | default/default/1x4 |
| 2K | 0.1477/0.1693/0.4390 | 0.1521/0.1740/0.4431 | 0.2424/0.2637/0.6433 | 0.5894/0.6137/1.2878 | default/default/default |
| 4K | 0.2329/0.2763/0.7261 | 0.2470/0.2850/0.7330 | 0.2739/0.3102/0.7580 | 0.6052/0.6509/1.3664 | default/default/default |
| 8K | 0.3905/0.4738/1.3119 | 0.4412/0.5237/1.3561 | 0.4768/0.5632/1.3906 | 0.6379/0.7214/1.5518 | default/default/default |
| 16K | 0.7290/0.9070/2.4833 | 0.8538/1.0210/2.5963 | 0.8157/0.9942/2.5751 | 1.1494/1.3297/2.9080 | default/default/default |
| 32K | 1.3787/1.7212/4.8575 | 1.6553/1.9748/5.1094 | 1.4792/1.8188/4.9699 | 2.2621/2.5986/5.7273 | default/default/default |

| Qwen397 K512 Batch | default D/C/F | `1x4_64x256` D/C/F | 4x1 BM128 D/C/F | 4x1 BM256 D/C/F | D/C/F胜者 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1K | 0.4916/0.5267/1.2519 | 0.4742/0.5075/1.2144 | 0.9965/1.0415/2.1995 | 2.1890/2.2190/4.3658 | 1x4/1x4/1x4 |
| 2K | 0.5082/0.5660/1.3254 | 0.4845/0.5436/1.3064 | 1.0206/1.0888/2.3587 | 2.1909/2.2889/4.5205 | 1x4/1x4/1x4 |
| 4K | 0.7897/0.9117/2.2905 | 0.8323/0.9638/2.3622 | 1.0230/1.1444/2.5731 | 2.1970/2.3162/4.7868 | default/default/default |
| 8K | 1.1570/1.3579/3.3584 | 1.2355/1.4666/3.5176 | 1.7774/1.9932/4.5322 | 2.2322/2.4562/5.0275 | default/default/default |
| 16K | 1.7507/2.1632/5.5860 | 1.9945/2.4446/5.8886 | 2.6561/3.0585/6.9208 | 4.4377/4.8559/9.8409 | default/default/default |
| 32K | 3.3390/4.1515/10.9413 | 3.9555/4.8482/11.6594 | 4.3102/5.1104/11.8908 | 6.7503/7.5700/15.3581 | default/default/default |

Qwen35专属单槽修复相对`d1267f...`的32K BM128直接ABBA12为
`1.725987 -> 1.496465ms`，paired ratio `0.867689`，IQR `0.861945--0.869843`，
12/12胜，10份physical/reduced output逐bit一致。合入后的工作树单点复验为
`1.483344ms`；相对`1x4`的D/C/F分别快`10.55%/8.14%/2.66%`，但相对default仍慢
`8.59%/6.60%/2.40%`。16K时BM128也比`1x4`快`4.5%/2.6%/0.8%`；8K及以下不占优。
Qwen397维度不启用该特化，1K/2K保持`1x4`胜出，4K起保持default胜出。

Qwen35和Qwen397六Batch结果JSON SHA256分别为
`e9cdb892817f82a0f702b830735b25f6f96d056635fc55a9a9882ad2b281a6d1`和
`0dfd27d68cfa536b897aa935c8dec3c7b7073e4075e4a45a31b766b951c684dc`；直接
ABBA12 JSON SHA256为
`16aa0544150b658c5bb7f3da47d1c42c381bcd5564283b02237eb3db4807f8ae`。

### 差距与资源分析

低Batch首先受BM padding支配。以1K为例，Qwen35 K256的`1x4`执行16,384行
（useful的2.0倍），BM128 4x1执行32,768行（4.0倍）；Qwen397 K256从3.2倍增至
6.4倍，Xiaomi从3.0倍增至6.0倍。此时4x1 executed TFLOPS仍有169--225T，
但useful吞吐被额外一倍的expert padding直接折损，无法靠N方向调度修复。

高Batch padding基本消失后仍有本体差距：BN64让每个N tile的MFMA工作量更小，
而B的LDS commit、跨wave barrier和CShuffle固定成本更频繁。BM128单tile只有
BM128xBN64，输出工作量是BM64xBN256 `1x4`的一半；BM256虽补回工作量，却增加A/C
live range。当前fresh ISA显示Qwen35 K512 BM128 full/short为168/164 VGPR、
metadata SGPR 38（next-free SGPR 96）、16,896B LDS、0 private/scratch，对应Q3；
其ISA SHA256为`a278c3aa6ef92e5311fc7b090804871f9d7d200e26221e224d75a568f221a761`。
非目标Qwen397 BM128 full/short保持178/176 VGPR、Q2；K512 BM256保持276 VGPR
（其中20 AGPR）、17,408B LDS、Q1，因此BM128始终是K512的最佳4x1。K256 BM256在
收窄修复前后最终ISA逐字节一致，SHA256为
`597d57b44c2da9e9d21ac20c9f96defdce0beea2049d58d8bf51275024634ae2`。

本轮已验证并保留的修正包括LDS CShuffle 128-bit输出、MFMA后调度屏障、partial VMEM
wait、MI308X XCC/SE/CU映射、显式MFMA/非MFMA `setprio 3/0`、K512 scale后载和
Qwen35 K512 BM128单B寄存器槽。已实测否定并撤回的方案包括full-K LDS、早期stage
priority变体、NT32、批量两个row-pair CShuffle、就地BF16转换、defer1及把short-N扩展到
K192/K256/K384。最终判断是：4x1实现满足请求的生产接口与数据路径，在Qwen35 K512
高Batch可超过`1x4`，但没有达到`default`；Qwen397和其余case也未普遍达到最快既有
路径。当前保留为显式path；Qwen35 K256仅在32K使用，K512仍不进入自动selector。
整个实现与实验未修改legacy JIT。

## Batched GEMM core ceiling预测与实测

本轮从`8c1a86965b2a65b69036291f9b95533044c2d81f`只移植
`tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py`到当前同名`tools`
目录；原文件SHA256为
`e393589fa1f49a0ede20ccd5df0f3aff2ad8fab7ed7a9fa9917dc59fba56bbcf`。正式测量版本
SHA256为`baca74ae95a564f98b14cfadd3f7f75665a3a7d5795d3362f109c4d6b3fe22a2`。测量后执行
Black/Ruff机械格式化，并把内聚occupancy helper的返回值对齐来源语义
`min(requested, achievable)`；本轮七个配置的`requested == achievable`，结果不受该修正
影响。最终版本SHA256为`80da30297540083b75eaafa22347ceb8a5379a3274c93a0c9e02065f7d952299`。
没有移植同提交的生产profile、TODO、wave-stage工具，也没有修改`src/core/asmjit.py`。
当前版本仅把原工具依赖的GPU状态、统计和occupancy helper内聚进单文件，并局部哈希
JIT compile key以规避文件名长度限制。内建`self-test`的几何、padding、MFMA/VMEM
工作量和地址多重集检查全部通过。完整CLI与复现命令见
[`batched-gemm-core-ceiling.md`](../../flydsl/attn_4wave/tools/batched-gemm-core-ceiling.md)。

这里的“预测”是该工具实测的`gemm_core_coissue_ceiling`：均衡batch skeleton保留与
目标矩阵相同的MFMA、B read和BF16 D write工作量，但MFMA operand、VMEM结果和D payload
彼此独立；A用`--a-in-reg`只预留完整K维寄存器，不访问A buffer；LDS只限制occupancy，
最终ISA不得包含`ds_*`。因此它是候选tile的core co-issue上界，不是正确GEMM实现，也
不是仅按峰值算力计算的静态理论值。

### 统一协议与配置

- GPU4，AMD Instinct MI308X / gfx942 / 80 CU；650W、1800MHz performance
  determinism、PTL `Enabled / VECTOR,F8`。
- 预测和生产实测均使用10套地址、40次round-robin warmup、50个CUDA-event样本、
  `sample-sync=end`；报告中位数和`[P25--P75]`。
- ceiling侧轮转B/D，生产侧轮转activation/weight/output；每个event只包含一次目标
  down dispatch，不包含sorting、gateup、reduction或完整MoE链。
- 生产侧使用当前`168808caeacf7e0d7cb336df25554a0bf778d6dc`源码。临时生产测量
  harness SHA256为`f8ba0964023d2e27beb50a83250772bdd7b9055d5efdc06ec6ac316c12d8fc3e`，
  未加入仓库。
- `差值 = ceiling - 生产`；`达到率 = 生产 / ceiling`。两侧为独立进程，差值不是
  配对置信区间。

| Case | ceiling `B x M x N x K` | `BM x BN x BK` | `WM x WN` (`W/WG`) | `W/SIMD` | `NT/WG` | ISA；LDS | ceiling WG | 生产active/launched WG |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Hy3 K=192 | `193x1528x4096x192` | `64x512x64` | `1x8` (8) | 4 | 8 | 92V+4A；32KiB | 4,632 | 4,632/4,801 |
| Qwen3.5 397B K=512 | `512x640x4096x512` | `64x256x128` | `1x4` (4) | 2 | 16 | 204V+4A；32KiB | 5,120 | 5,120/5,632 |
| Qwen3.5 397B K=256 | `512x640x4096x256` | `64x256x128` | `1x4` (4) | 2 | 16 | 140V+4A；32KiB | 5,120 | 5,120/5,632 |
| Qwen3.5 35B K=512 | `256x1024x2048x512` | `64x256x128` | `1x4` (4) | 2 | 8 | 204V+4A；32KiB | 4,096 | 4,096/4,352 |
| Qwen3.5 35B K=256 | `256x1024x2048x256` | `64x256x128` | `1x4` (4) | 2 | 8 | 140V+4A；32KiB | 4,096 | 4,096/4,352 |
| Xiaomi K=256 | `384x683x6144x256` | `64x256x128` | `1x4` (4) | 2 | 24 | 140V+4A；32KiB | 4,224 | 4,224/4,480 |
| H3 K=384 | `128x1024x6144x384` | `128x256x128` | `2x4` (8) | 2 | 24 | 172V+4A；64KiB | 1,024 | 1,024/1,152 |

每个ceiling WG处理完整N维，即`NT/WG=ceil(N/BN)`且N tile group数为1。均衡M取
`B*TopK/E`的整数近似；Hy3和Xiaomi分别向BM补齐，ceiling useful/executed效率为
99.48%和97.02%，其余五个case为100%。所有七个最终ISA的请求occupancy均与HIP
driver返回值一致，且均为0 scratch、无`ds_*`。

### 预测与实测结果

| Case | 生产 ms / useful TFLOPS `[P25--P75]` | ceiling ms / useful TFLOPS `[P25--P75]` | ceiling - 生产 | 达到率 |
| --- | ---: | ---: | ---: | ---: |
| Hy3 K=192 | 1.3747 / 337.43 `[334.42--353.63]` | 1.2659 / 366.42 `[365.78--367.85]` | +28.99T / +8.59% | 92.09% |
| Qwen3.5 397B K=512 | 3.4214 / 401.71 `[400.25--402.76]` | 2.7981 / 491.19 `[490.94--491.68]` | +89.49T / +22.28% | 81.78% |
| Qwen3.5 397B K=256 | 1.8616 / 369.14 `[360.19--372.05]` | 1.6991 / 404.45 `[394.15--405.37]` | +35.31T / +9.57% | 91.27% |
| Qwen3.5 35B K=512 | 1.4113 / 389.54 `[388.78--391.27]` | 1.1284 / 487.21 `[486.40--488.41]` | +97.66T / +25.07% | 79.95% |
| Qwen3.5 35B K=256 | 0.7665 / 358.62 `[352.33--359.46]` | 0.6738 / 407.96 `[407.01--410.09]` | +49.34T / +13.76% | 87.91% |
| Xiaomi K=256 | 2.2746 / 362.53 `[361.38--363.79]` | 2.1124 / 390.57 `[389.52--391.95]` | +28.03T / +7.73% | 92.82% |
| H3 K=384 | 1.5777 / 392.01 `[390.50--395.17]` | 1.4609 / 423.34 `[422.91--424.88]` | +31.33T / +7.99% | 92.60% |

七份ceiling JSON按“文件名 + SHA256”排序后的清单SHA256为
`1c3f5cd5b5c4b4f4d9b51f5852972d12ed7c82b0e6253c84a5705507fdc59f00`；七份生产
JSON的清单SHA256为
`478e08d30d7f729f5c7d4a0143e1950953558afa9a3932ef03067eaef8a318c7`。

生产达到ceiling的范围为79.95%--92.82%。Hy3、Xiaomi和H3均超过92%，说明当前专用
路径已经接近这个不含正确性依赖的core co-issue上界；两个Qwen K=256路径分别达到
91.27%和87.91%。两个K=512 `default`路径只有81.78%和79.95%，同时拥有最高的
204V+4A资源档位，是后续优先优化对象。该差距不能直接解释为某一种cache或指令瓶颈；
ceiling还省略了VMEM到MFMA RAW、LDS搬运、scale、metadata和真实epilogue，需结合
PMC/ATT再做归因。

## 专用down路径运行时拓扑映射

`1x4_64x256`、`2x4`和`1x8`共享`_map_down_task`，不包含Batch、E、N、K或TopK
专用映射常量。host编译时通过当前device name识别MI308；只有MI308启用4 XCC、
每XCC 4 SE、每SE 5 CU的topology特化。generic连续分段数也由设备决定：MI308为
4 XCC，非MI308为8 XCD；非MI308不生成topology `gpu.func`，只保留generic kernel。

MI308 topology kernel从sorting有效行数运行时计算：

```text
valid_tasks     = ceil(sorting_valid_rows / task_rows)
tasks_per_se    = floor(valid_tasks / 16)
mapped_tasks    = tasks_per_se * 16
short_cu_tasks  = floor(tasks_per_se / 5)
long_cu_count   = tasks_per_se % 5
```

其中`1x4_64x256`和`1x8`的`task_rows=64`，`2x4`的`task_rows=128`。
完整映射依次完成XCC连续分段、XCC内SE分段及每SE的5-CU ragged列转置；不能均分
到16个SE的尾部保持identity。generic映射按设备做4-way或8-way连续分段，不能完整
均分的尾部保持identity。两种映射对任意非负任务数都是双射。

`1x4_64x256`固定使用generic。MI308上的`2x4`要求精确padded task数至少为80；
`1x8`要求精确padded task数位于闭区间`[160, 2880]`，即每CU `[2, 36]`个任务。
host使用`task_num`及`M * TopK`做保守预选，最终由`_map_down_task`读取
`p_num_valid_ids[0]`，按对应`task_rows`计算sorting/expert padding后的精确任务数；
device端结果是topology/generic选择的权威门禁。

### 1x4_64x256 topology对比

control为当前重构package的生产generic映射，源码集合SHA256为
`f37254683b2cc4778b5628bf8dcf20ceeb437ce038dac136066fdd4a339a1888`；candidate集合
SHA256为`1e60d160f847b3dc677ea4137e07c704d2e5bba24122d2e590c6ce438efee94a`，唯一差异是
`gemm2_1x4.py`将`_map_down_task(..., False, ...)`改为`True`，该文件的control/candidate
SHA256分别为`075f8c5e2cb6223c2ebeeb6e77806f0d04c18b7c63063cfd88d6e478bf7e021a`和
`6434b6267d0538df36cdd298421172f3690f4f6e69ac60eeab50d4914852436f`。

四个生产配置均使用10组buffer和ABBA12。表格每格依次为
“Down提升率 [IQR] / Combined提升率 [IQR]”；正值表示topology更快，负值表示回退。

| Batch | Xiaomi K=256 | Qwen 397B K=256 | Qwen 35B K=256 | H3 K=384 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | +1.4% [-1.4%, +1.9%] / +0.7% [-1.5%, +1.5%] | +4.4% [+3.6%, +5.5%] / +6.5% [-4.5%, +21.4%] | +4.0% [-3.7%, +7.7%] / +3.0% [-3.2%, +4.6%] | -2.1% [-5.1%, -0.3%] / -1.6% [-3.6%, -0.8%] |
| 2K | -0.0% [-1.6%, +1.4%] / +0.8% [-1.3%, +1.5%] | +4.6% [+4.4%, +5.5%] / +4.0% [+3.5%, +4.5%] | +2.1% [-4.3%, +8.5%] / +1.8% [-2.7%, +4.6%] | -1.7% [-4.5%, -0.1%] / -0.8% [-1.5%, -0.3%] |
| 4K | -1.2% [-1.6%, +0.3%] / -0.9% [-1.8%, +0.2%] | +0.0% [-1.5%, +3.4%] / +0.2% [-1.2%, +2.1%] | +1.8% [+0.4%, +3.2%] / +1.5% [-0.1%, +2.9%] | +0.2% [-0.3%, +0.6%] / +0.2% [-0.3%, +0.4%] |
| 8K | -1.3% [-2.7%, +0.4%] / -0.9% [-2.2%, +0.7%] | +1.0% [-1.5%, +4.1%] / +0.8% [-0.5%, +1.6%] | +1.3% [-0.1%, +2.2%] / +0.9% [+0.0%, +1.1%] | +0.2% [-0.2%, +0.4%] / +0.2% [-0.2%, +0.4%] |
| 16K | -1.2% [-2.7%, +1.1%] / -1.2% [-2.5%, +0.9%] | +2.6% [+1.7%, +3.9%] / +1.8% [+0.0%, +2.2%] | +3.2% [+1.9%, +4.4%] / +1.5% [-2.6%, +4.4%] | -1.1% [-1.6%, -0.6%] / -1.4% [-1.8%, -0.6%] |
| 32K | +0.2% [-0.4%, +0.5%] / -0.2% [-0.3%, +0.1%] | -0.0% [-0.6%, +1.2%] / -0.6% [-0.7%, -0.0%] | +0.6% [+0.3%, +1.3%] / +0.6% [+0.3%, +1.0%] | -1.3% [-1.5%, -1.0%] / -1.2% [-1.3%, -1.0%] |

topology存在局部收益，但不能作为`1x4_64x256`的统一策略。相同的2,048个最小活跃
任务下，Qwen 35B 16K的Down提升3.2%，Xiaomi 16K回退1.2%，H3 32K回退1.3%；
收益显然不只由任务密度决定。Qwen 397B在1K/2K稳定提升约4%-5%，但其余Batch大多
中性；为避免引入模型或shape专用门禁，生产路径继续固定使用generic映射。

当前生产实例资源如下，均为0 scratch：

| 路径 | 指令数 | VGPR / SGPR / LDS |
| --- | ---: | --- |
| `1x4_64x256` | 930 | 250 / 96 / 25,600B |
| `2x4` generic | 1,092 | 256 / 96 / 65,536B |
| `2x4` topology | 1,119 | 256 / 96 / 65,536B |
| `1x8` generic | 490 | 128 / 96 / 28,672B |
| `1x8` topology | 521 | 128 / 96 / 28,672B |

资源使用当前重构源码和`COMPILE_ONLY=1` fresh dump；三份最终ISA SHA256依次为
`8a4515d8e7a5545e284780657d1fde8460fd0cdfa5be549cb141a0db66c56aa0`（1x4）、
`435dbb14bd0ab5601594e9536cf730512f229d3b9c0cd83711b88af94639f1fc`（2x4）和
`bb507d0b1df5856df45745aba516e9b0917e9f48657c9e17abda986bccba6525`（1x8）。
指令数按每个函数体内非标签、非directive的机器指令行统计；`_0`为先生成的generic，
`_1`为topology。


## 正确性与结论

- 原最终path矩阵的输出均finite，Down最大relative-L2为
  `2.065696389763616e-05`；新增4x1矩阵的Down/Full最大relative-L2分别为
  `0.004914`和`0.005431`，inactive tail及padding均保持未写。
- generic/topology映射整数模型对`valid_tasks=0..10000`穷举无重复或漏项；不能完整
  分段的尾部保持identity。
- `1x4_64x256` topology在Xiaomi、两个Qwen K=256和H3共24组A/B中均逐bit一致、
  `rel_l2=0`、finite、tail clean且padding clean；性能收益依赖模型shape，未进入生产。
- `2x4`与`1x8`本轮均完成fresh compile；正式矩阵实际执行了两者的
  generic/topology运行时门禁，所有结果finite且满足relative-L2阈值。最终源码另以
  Hy3 1K单路径fresh隔离复验：两者Down/Full均`rel_l2=0`、finite、tail/padding clean，
  每phase 4个样本且状态完整恢复；结果JSON SHA256分别为
  `694c6cdf23b95c4e6693fb7c213b264601e04ed2a8c08269e1dc11c587dc332c`和
  `33926877e5c9d2dd3871f91de83b105716e58c3a770989bb30afb25fce74ce5b`。
- MI308精确门禁覆盖padding跨界：`1x8`的原始/精确任务数`100/160`进入topology，
  `2800/2881`在topology kernel内回退generic；两者均与强制4-XCD generic逐bit一致。
  专项JSON SHA256为`edde066551505877e1d70713e72eb89fa9cb466e48db898605a3469e4c4b776f`。
- 模拟非MI308配置下，`2x4`和`1x8`的8-XCD generic均与MI308 topology逐bit一致，
  `rel_l2=0`且finite；运行时专项JSON SHA256为
  `b8d2aecc66f59ff2912c201a519b690e05ecbcfdc227ab208d8635500030c104`。fresh IR各只包含一个
  generic `gpu.func`，且确认使用8-XCD分段；本轮fresh ISA SHA256分别为
  `68311777e8d8fcede30a8507d793a518563692cf10428217bee0feed2916a6a8`和
  `68c20585ded3ee2b92cd0e2a717a7a7c0bedcf6d16c2980f561e7fa5a9aeac14`。
  device name配置单测3组全部通过。
- `1x8`当前源码六点Down、Combined及Full矩阵均重新执行，输出finite、tail clean；
  相对default的最大relative-L2包含在上述全矩阵上界内。
- 本轮Hy3 metadata显示`1x8`精确有效行数从12,352增至296,448：1K-16K的精确任务数
  位于`[160, 2880]`并进入topology，32K为4,632个任务，在kernel内回退generic。
- fresh dump确认`1x8`包含490/521指令的generic/topology两个`gpu.func`，资源均为
  128 VGPR、96 SGPR、28,672B LDS和0 scratch；`2x4`两个特化资源均为256 VGPR、
  96 SGPR、65,536B LDS和0 scratch，对应1,092/1,119条指令；`1x4_64x256`为
  930条指令、250 VGPR、96 SGPR、25,600B LDS和0 scratch。
- `4x1`五组正式参数化正确性测试全部通过，包含BM128/N2048/K512/PTPC单槽特化；
  42点ABBA12和3点ABBA48均完成finite、
  tail、padding及数值门禁。六项流水优化后，Qwen35 K256 BM256专项ABBA12达到
  `0.876084ms / 313.76T`，fresh ATT确认热循环partial wait、无`v_pk_mul_f32`、
  128-bit LDS写以及MFMA/非MFMA双stage。defer1通过数值但ABBA4稳定回退2.85%，已否证。
- final-source `3d5af103...`已晋级Qwen35 K256/BM256 balance93流水；32K direct
  ABBA24的Down/Combined/E2E分别提升6.35%/6.09%/2.59%，均24/24胜。六档矩阵和
  16K ABBA48确定1K--16K仍选`1x4_64x256`，仅32K选BM256 `4x1`加128B padding。
  K192/K384/K512 fresh ISA逐条不变，目标资源为240 combined VGPR、25,600B LDS、
  0 scratch和256条NT store；final ATT steady MFMA union为74.34%。
- `0c2c5d...`已完成Qwen35/Qwen397 K512各六个Batch的24-sample全矩阵；final-source
  K512机器码逐条相同。Qwen35
  BM128单槽特化相对`d1267f...`的直接ABBA12 ratio为`0.867689`、12/12胜；32K相对
  `1x4_64x256`的Down/Combined/Full提升10.55%/8.14%/2.66%，但相对default仍回退
  8.59%/6.60%/2.40%。Qwen397不启用该特化，非目标ISA逐字节不变。所有矩阵输出逐bit
  一致，因此K512的4x1继续只注册显式path，不进入自动selector。
- 七个当前生产case完成同协议batched GEMM core ceiling预测与真实down实测；生产达到率
  为79.95%--92.82%。两个K=512 `default`路径达到率最低（81.78%和79.95%），是后续
  优先优化对象；该差距只表示相对core co-issue上界的剩余空间，不单独用于瓶颈归因。
- 最终生产选择见1K-32K矩阵；MI308的`2x4`和`1x8`保留topology特化与运行时门禁，
  其余设备只生成8-XCD generic；`1x4_64x256`始终使用设备相关generic swizzle。
  Qwen K=512维持`default`；Qwen397 K=256维持`1x4_64x256`；Qwen35 K=256在
  1K--16K选`1x4_64x256`，32K选BM256 `4x1`加128B padding。
- 各性能/ATT harness结束时均验证performance level、PTL和NUMA恢复原状态。defer1首次
  门禁曾被外部作业占满8张GPU阻止；空闲窗口恢复后已完成并否证。最终矩阵结束后外部
  作业再次占用全部GPU约46% VRAM；本轮未终止或干扰外部进程，所有测试卡在交还前均已
  恢复原`auto`与PTL状态。

## FlyDSL 0.3.2最终验收进度（2026-09-03）

本轮在system FlyDSL 0.3.2环境修复raw-buffer cache modifier与compile hint传播后，
重新验证Qwen3.5 397B K256和Hy3 K192。FlyDSL定向回归结果为`19 passed, 2 skipped`。
以下性能均使用MI308X、1800MHz performance determinism、PTL
`Enabled / VECTOR,F8`、650W、NUMA off和10组buffer；有效FLOPs仍按
`2 * B * TopK * Hidden * Inter-TP`计算。

Hy3最终只移动`1x8` activation WG barrier：先发出routing weight及per-tensor scale
读取，再同步activation LDS，首次LDS读取仍严格位于barrier之后。最终
`gemm2_1x8.py` SHA256为
`405884d53ea9b7bd14cdb0c157db57f0b97d4bfee3f90912a47abfb668b725ae`；production
ISA与隔离winner逐字节一致，SHA256为
`640c7897ec788b51f870ee8d8b28c062cfdc026064faa257396842f789b3e88e`，资源保持
128 VGPR、96 SGPR、28,672B LDS、0 private/scratch。

同地址、同输入的direct ABBA24中，candidate/control配对ratio中位为`0.996439`，
IQR为`[0.994063, 0.997618]`，22/24轮胜；active区逐bit一致，control/candidate
inactive tail均未写。JSON SHA256为
`988dd0055b71fd84b4d564e2b7ef1974cd38931c05b7faae5d5711b3be37de43`。
按原报告相同六路径分配及ABBA12协议重跑六档矩阵，1K--32K的Down/Combined/Full
均继续选择`1x8`；32K Down为`1.346566ms / 344.47T`，低于原报告
`1.3513ms / 343.3T`，时延改善0.35%。32K Combined/Full分别为
`2.075629/5.190804ms`；六档输出均finite、tail/padding clean，Down相对default的
最大`rel_l2=2.066e-5`。矩阵JSON SHA256为
`5148ea3fa689fae59504d8bffd4105535cc5f65ff61a5cfefcba5dffc4021450`。

Qwen3.5 397B K256 production `1x4_64x256`已完成当前源码ABBA12：32K Down为
`1.866767ms / 368.12T`，相对原报告`1.8804ms / 365.5T`时延改善0.73%；
Combined/Full分别为`2.783471/6.683827ms`。Down和Full相对default逐bit一致，
finite、tail/padding clean。JSON SHA256为
`a212b2c6044c26341b9f8afacc6889b13ea134a694a98916408da26472d26d18`。
production compile-only ISA与formal dump逐字节一致，SHA256为
`e824a84582629078b852daa673f22cbb5cd7e74a9d2a1346a954127a733adb1d`；资源为
250 VGPR、96 SGPR、25,600B LDS、0 private/scratch，含16条NT store且无
`v_pk_mul_f32`/`v_pk_fma_f32`。

尚未完成的最终门禁是Qwen397 K256 ABBA24、Qwen六档当前源码矩阵，以及Qwen/Hy3
winner fresh ATT。启动Qwen ABBA24时idle gate观测到全部8张GPU均为100% busy、
约76% VRAM，测试在改变PTL/performance state前被拒绝；未终止外部进程，也未降低
idle门禁。fresh ATT配置已固定实际发射符号
`moe_2stage_down_prefill_1x4_64x256_0`和`moe_2stage_down_prefill_1x8_1`，待空闲
GPU窗口补齐后再更新本节，不能把此前control ATT代替winner ATT。
