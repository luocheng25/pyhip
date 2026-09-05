# Tile API 重构验收（2026-09-05）

本阶段只重构当前 direct-paged 内核，不恢复旧 attention 算法、gather 或 fallback。
基线为工作区重构前 Git blob `40e83659b4f752ce1424ca598dd75a1827141c07`，
SHA256 `d75b971423b763b6968d143b008845ad4d100e64b69f10ba0c6e2671cd4ff79c`。
重构后 SHA256 `3c340d9c36f8b54f6ed473730e20e6759dc273aaf3ac25165b6fc066edfa5b29`。

## 实现

- Q/K/P 与 score/output 使用 `make_tiled_mma`、`make_fragment_A/B/C`。
- QK/PV 使用 `fx.gemm`，删除逐 atom 的手写 MMA 循环。
- V 的 `fx.select(v, [0, 2, 1])` 只是 layout view，不搬运寄存器；packed-i32 V
  拼接及 NaN-tail 保护保持不变。
- **`traversal_order` 是最快变化维度优先**。PV 的 `mnk` 对应原来的 K 外层 / M
  内层交织；`kmn` 虽然数值正确且指令数量相同，却会串行使用同一个 accumulator。
  以实际 MFMA operand 顺序验证后采用 `mnk`，不凭名称推断顺序。
- 保留已验证的八阶段调度、page-prefetch、显式 LDS-read 与 wait/fence。通用 copy
  对 aliased ring 的保守同步不是本次 refactor 的替代路径。

## 汇编与正确性

六种 D128/D192 × noncausal / causal-paired / SWA+sink specialization 新鲜编译：
**去掉注释和基本块编号后的每一条指令及操作数均与基线一致**，不只是总数相同。
全部 mnemonic 计数、VGPR/SGPR/LDS/private/spill 一致，private/spill 均为 0。
静态指令总数不应与 ATT 每 wave 动态计数混淆。

最终完整回归：**181 passed / 6 skipped**；6 项为 AITER 缺失 page64 实例。
同进程对照的全部输出也通过完整 FP32 reference (`err=0`) 与三次 bit-exact 重复。

## 同进程性能

MI350X gfx950 GPU0，原有 auto-DPM。相同原始 5D 输入/随机页表、预分配输出、无 LSE。
100 轮共同预热，5 轮交替顺序，每候选每轮20 warmup/100采样，`run_perftest`
GPU profiler 时间；以下为五轮中位数，单位 **µs**。

| 场景 | Dqk | 重构前 | tile API | 延迟变化 |
|---|---:|---:|---:|---:|
| noncausal Q10240/KV2583 | 128 | 256.517 | 258.132 | +0.63% |
| noncausal Q10240/KV2583 | 192 | 295.670 | 296.550 | +0.30% |
| causal Q=KV32768 | 128 | 4454.812 | 4448.558 | -0.14% |
| causal Q=KV32768 | 192 | 5286.584 | 5282.314 | -0.08% |
| SWA+sink Q16K/KV128K/W128 | 128 | 100.716 | 101.812 | +1.09% |
| SWA+sink Q16K/KV128K/W128 | 192 | 116.974 | 116.791 | -0.16% |

对短路径再反转候选初始顺序复测：

| 场景 | Dqk | 重构前 | tile API | 延迟变化 |
|---|---:|---:|---:|---:|
| noncausal | 128 | 258.145 | 256.663 | -0.57% |
| noncausal | 192 | 296.731 | 296.748 | +0.01% |
| SWA+sink | 128 | 100.843 | 100.849 | +0.01% |
| SWA+sink | 192 | 116.379 | 116.351 | -0.03% |

结论：**未观察到可重复的性能下降**，且六种代码生成完全等价。保留首轮小幅正差，
不把计时波动抹成零，也不把本阶段重构称为加速优化。

## 复现与证据

- [benchmark_revisions.py](benchmark_revisions.py)：`--baseline` 接受 Git blob 或
  `revision:path`，只在基准进程加载旧源码，不引入生产 fallback。输出 µs/有效
  TFLOPS/逻辑 TB/s、严格 error ratio、真实 dispatch 及资源。
- [tile_refactor_results.json](tile_refactor_results.json)：六配置全部采样、源码和 ISA
  hash、全部 mnemonic 计数与资源。
- [tile_refactor_repeat_results.json](tile_refactor_repeat_results.json)：反序复测全部采样。
- [tile_refactor_isa_validation.json](tile_refactor_isa_validation.json)：六配置逐指令 hash
  与等价性断言。完整 IR/ISA 目录在结果中记录。

原始工作区基准参数：`--candidate baseline static --baseline 40e83659b4f752ce1424ca598dd75a1827141c07`；
该重构前 blob 当时在 index 中，不是独立提交；全新 clone 不保证包含这个对象。
新鲜 ISA 捕获时使用 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 及 `--dump-root` 独立目录。
后续提交后也可用第一阶段提交的源码作为 baseline 重跑。