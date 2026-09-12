# moe down kernel - BlockScaled(128x128)

## 保留范围

本目录维护**胜出路径及原有基线对照**。旧内核、工具和本文档保留；全部优化尝试、完整参数实现、其他测试和ATT产物归档到 [try/readme.md](try/readme.md)。整理前的说明另存于 [try/README.md](try/README.md)，历史记录未丢弃。

| 文件 | 用途 |
|---|---|
| [moe_multistage_down.py](moe_multistage_down.py) | 固定胜出配置的MFMA16 packed down |
| [moe_multistage_reduce.py](moe_multistage_reduce.py) | 固定packed TOPK reduce |
| [moe_multistage_pipeline.py](moe_multistage_pipeline.py) | down＋inverse清空/重建＋reduce与workspace |
| [moe_8wave_down.py](moe_8wave_down.py) | 原有FlyDSL BN32/BN64基线，原样保留 |
| [moe_8wave_down_utils.py](moe_8wave_down_utils.py) | 原有FlyDSL DMA工具，原样保留 |
| [test_blockscaled.py](test_blockscaled.py) | 唯一主测试：四候选比较、数值/graph/舍入/统计/ISA检查 |

主路径不导入归档代码。MFMA32、compact、混合OC、其他stage/tile/cache及restore实验只在归档内保留，不作为主路径可选配置。

## 胜出配置

**MFMA16、M256/N128/K256、OC4、PF3、256个persistent CTA，packed down＋定制TOPK reduce。**

| 部分 | 固定配置 |
|---|---|
| Down线程与流水 | 512线程/8 waves，两个4-wave组错相；两个Memory/Compute对，共四阶段 |
| 计算 | 原生MFMA16×16×128，每Compute16条；K128独立scale、FP32累加、routing乘法后BF16舍入 |
| B读取 | 直接global→LDS，四槽64KiB ring，预取3拍，cache aux16/SC1 |
| Scale/routing | 多线程协作缓存于LDS；A FP8片段留寄存器 |
| C写出 | 寄存器BF16/permlane/DPP，连续128B NT写，aux2；无C LDS |
| 调度 | 256个persistent CTA，每次调用重置counter；不使用nonpersistent/XCD/task-table变体 |
| Reduce | 256线程、2048列/CTA、NT读取，FP32顺序求和后BF16，cached写回 |
| Inverse | 每次fill(-1)并重建，空/缺失route贡献0，无routing内容缓存 |

## 调用与张量契约

`compile_packed_down_reduce(n=..., k=256, topk=..., num_experts=..., workspace=...)` 返回十tensor callable，顺序为：最终output、A、B、A scales、B scales、sorted IDs、sorted routing weights、expert IDs、valid IDs、task counter。

- 支持gfx950；**K固定256、N为512的正倍数**，TOPK不超过专家数且不超过255。非胜出配置显式拒绝，不静默fallback。
- A为连续OCP FP8 E4M3FN `[tokens,topk,256]`；A scale为物理K-major FP32，B scale为FP32 `[experts,N/128,2]`。
- B为现有 `shuffle_weight(layout=(16,16))` 的FP8 `[experts,N,256]`；不使用MFMA32的权重排列。
- sorting为AITER M256 padded专家run。`num_valid_ids[0]` 是padded有效前缀长度，expert ID只读取有效前缀；不能用未初始化的capacity尾部统计专家。
- **Pipeline最终output为BF16 `[tokens,N]`**，内部已完成TOPK求和，调用方不要再sum。
- 单独 `flydsl_moe_gemm_8wave_down` 写BF16 packed `[expert_capacity*256,N]`，物理顺序为 `[expert_block256,global_N64,row256,col64]`，不是普通2-D行主序；必须配套packed reducer。
- 保留的旧FlyDSL工厂及十tensor接口不变，输出仍为routed `[tokens,topk,N]`，TOPK在dim1。
- `DownReduceWorkspace` 复用中间buffer及inverse。首次分配/编译需在graph capture前预热；不同stream/并发调用使用不同workspace。当前buffer offset为32-bit，超范围会报错。

## 测试与基线比较

[test_blockscaled.py](test_blockscaled.py) 默认比较以下四项，默认形状tokens16384/TOPK8/E384/N6144/K256、seed1234：

| 选择键 | Down | 最终求和 |
|---|---|---|
| `pyhip` | 原PyHIP BN64/OC1 routed | 预分配 `torch.sum(dim=1, out=...)` |
| `flydsl_bn32` | 原FlyDSL BN32/OC1 routed | 同上 |
| `flydsl_bn64` | 原FlyDSL BN64/OC1 routed | 同上 |
| `4stage_bn128_tuned` | 胜出BN128/OC4 packed | inverse清空/重建＋定制reduce |

旧BN32/BN64候选、输出检查（最大/平均绝对误差、`calc_diff`、不匹配数）、独立down/reduce/完整时间、两种带宽和相对PyHIP/BN64速度比均保留。旧两处跨K累加的显式FMA修复也保留，防止BF16中点差异经TOPK相消后放大；确定性舍入与graph回归已合并到主测试。

- 直接运行脚本默认测试完整down＋reduce；`--candidate` 可以单选或多选上述四项。
- `--mode down` 测单独down：旧基线仍为routed，胜出路径为packed；通过**计时外Torch解码**验证，不能将该packed数字声称为原routed接口性能。
- `--profile` 要求单候选，23次直接launch且不报性能；未指定mode时为down-only，显式 `--mode down-reduce` 才采完整流程。
- 仍可设置tokens/model-dim/experts/topk/seed；本主比较限定胜出形状范围。历史完整可调接口在归档基准中。
- pytest使用 [pytest.ini](pytest.ini)，例如从仓库运行pytest并以 `-c tests/flydsl/moe_8w_down/pytest.ini` 选择配置；只收集主测试，排除归档。归档测试使用 [try/pytest.ini](try/pytest.ini)，分开进程执行以避免同名模块串用。

### 数值和计时口径

实际量化使用AITER BF16→OCP-FP8，真实sorting/shuffle；各候选共享同一A/B/scales/routing及中间/最终buffer地址。先对独立Torch K128块缩放参考检查逐路BF16结果，再以已验证PyHIP的实际BF16 routes做最终sum参考；独立packed reducer测试另与Torch sum比较。容差仍为 `rtol=atol=0.01`。

- `Down time` 含counter清零、排除inverse/reduce；`Total Time` 直接测完整调用，不是组件相加。量化、排序、编译、参考、解码、统计及NaN投毒在稳态计时外。
- 使用原 `pyhip.run_perftest`、warmup2/iters10、同址零参数闭包；`run_test` 现在也默认 `reduce_output=True`。归档旧driver保留原默认口径。
- 有效TF/s以真实路由行计，padded TF/s以device有效专家块×256计；总TF/s也仅计GEMM工作量。
- Down字节数计A一次、有效BF16输出写出、B每M块读取；理想模型仅把B换成每去重专家读取一次。二者均非实测HBM流量，未计全部元数据和OC重复A读取。
- Reduce TB/s为有效BF16中间读取＋最终输出写入除以自己的独立耗时；Total另计reduce和inverse主要流量，不计未写padding容量。未测组件显示N/A。

## 性能与本次整理验证

历史四轮同址验收：tuned routed＋torch.sum **928.346 μs**，胜出packed＋custom **901.831 μs**，速度+2.94%、时延−2.86%。后续整体复测约905–909 μs；这不等于此前routed down历史10%结果的稳定复现。

本次精简前后用**同一归档基准和同一buffer做ABBA**：归档winner down509.256443 μs、精简版509.406599 μs，约0.03%差异；完整908.283001 vs905.132800 μs。未显示down退化，不把小幅整体差异当成新的优化收益。[复核日志](try/cleanup_20260912/paired_cleanup.log)。

以下为本次主入口单轮验证，不是新的多轮性能验收：

| 指标 | PyHIP | 旧FlyDSL BN32 | 旧FlyDSL BN64 | 胜出packed |
|---|---:|---:|---:|---:|
| Down μs | 552.508 | 657.492 | 574.077 | **509.110** |
| Down有效TF/s | 746.264 | 627.106 | 718.226 | **809.877** |
| Reduce μs | 397.330 | 401.579 | 399.594 | **362.917** |
| Reduce逻辑TB/s | 4.560 | 4.512 | 4.534 | **4.993** |
| 完整实测 μs | 1007.765 | 1082.695 | 1020.853 | **909.969** |
| 最终不匹配 | 0/100663296 | 0/100663296 | 0/100663296 | 0/100663296 |

- 主测试 **25 passed**，涵盖短N、真实padding、persistent复用、graph replay、空/原地改变routing、BF16相消、基线比较和统计；[最终测试日志](try/cleanup_20260912/final_tests.log)。
- 新鲜胜出down ISA：185 VGPR、96 SGPR、70660B LDS、零private/VGPR spill/SGPR spill，无AGPR。12个Memory无VALU、整段Compute无地址，165个稳态间隔各7条VALU，输出NT。reduce为41 VGPR/31 SGPR、零LDS/private/spill；[down ISA](try/cleanup_20260912/isa/moe_winner_cleanup_isa_20260912/moe_down_8stage_kernel_0/21_final_isa.s)。
- [完整四候选日志](try/cleanup_20260912/default.log)。本次未更改旧基线源码、计时器、容差、GPU时钟或设置，未新增commit/push。

其他优化方向及全部成功/失败记录统一见 [try/readme.md](try/readme.md)，不再在主说明列实验配置。

