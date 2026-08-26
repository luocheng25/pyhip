# Batched GEMM core ceiling TODO

以下项目不进入均衡batch V0。只有V0在多个shape上通过SQ闭合、资源闭合并能稳定筛出合理top-K后再实施。

## 验证后优先级

1. **候选tile/wave搜索**
   - 任意正整数`n_tiles_per_wg`已实现；五个current case均按`n_tiles_per_wg=ceil(N/BN)`、单write-only目标和10-buffer协议正式重测；
   - 扫BM/BN/BK、waves_m/n、waves/SIMD和A模式，寻找能覆盖Hy3/Qwen的候选族；
   - 先做资源/公式剪枝，只短测合法top候选。
2. **非均衡grouped GEMM**
   - 输入`group_ms`或task-to-group映射；
   - 计算每group的M padding、active WG和dispatch tail；
   - 保留同group A/B复用，不加入模型名或expert特判。
3. **候选批量搜索**
   - 对BM/BN/BK、waves_m/n、waves/SIMD、A模式、grid order、schedule做合法性剪枝；
   - 只短测解析模型选出的top 3--5；
   - 输出hard roof、best skeleton、资源档位和置信标记。
4. **真实layout敏感性**
   - 增加A/B stride和有限的layout permutation；
   - 不复制单个生产kernel的专用swizzle；
   - 用L2/HBM PMC判断是否值得继续建模。

## 明确延后

- LDS数据读写及VMEM->LDS->MFMA依赖；
- VMEM load结果作为MFMA operand的RAW依赖；
- persistent WG和跨M/N动态tile scheduler；
- split-K、atomic和跨WG reduction；
- scale、dequant、activation、routing weight和完整CShuffle epilogue；
- 非连续D store、padding stride和output reduction；
- 精确scheduler/phase DAG回放；
- 真实功耗上下文及跨kernel融合成本。

## V0通过标准

- [x] 纯推导self-test覆盖padding、A-reg/A-VMEM、waves/WG、waves/SIMD和非法组合；
- [x] 所有四种schedule的`SQ_WAVES/MFMA/VMEM_RD/VMEM_WR`与公式100%闭合；
- [x] 最终ISA没有`ds_*`，LDS仅控制occupancy；
- [x] 请求的waves/SIMD与HIP occupancy完全一致；单write-only MFMA目标使Hy3从160降到96 vector registers/wave，并达到4 waves/SIMD；
- [x] `batch_m_n/batch_n_m`动态工作相同；
- [x] 五个均衡MoE shape完成稳定短测；
- [x] `n_tiles_per_wg=1/2/4/8`保持SQ总工作闭合，full-N的8/16/24 tile配置通过派生自测和五shape实测；
- [x] 五个生产/probe case完成10-buffer、40 warmup、50 sample正式吞吐复测；
- [x] 五个生产/probe case完成当前full-N的SQ、L2/HBM PMC和fresh ATT；
- 已知高性能tile或其邻域进入skeleton预测top-K；
- 完整候选筛选明显快于人工实现一个正式kernel。