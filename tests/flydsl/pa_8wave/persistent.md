# 可选 persistent 调度

`PagedAttention(..., persistent=True)` 启用设备端工作队列；默认 `False` 保留 static。
不改变输入 ABI、attention 算术、SWA/sink 语义或 causal 首尾配对，也不调用 4-wave。

## 调度与资源生命周期

- persistent grid 为 `min(CU 数, 最大任务数)`。当前8-wave的 LDS 限制为每 CU 一个 CTA，
  不照搬4-wave的 `2 × CU` grid。每个 CTA 先处理自己的 `block_id`，随后 atomic add
  领取后续 ticket，直到所有实际任务完成；不是只有首尾两块的“伪 persistent”。
- ticket 在各 batch 的**实际** Q-block 数中映射，head 是最快维度。空 batch 跳过，
  不处理 `max_seqlen_q` padding；prefix/page table/last-page 元数据每次从设备读取。
- full causal 仍按原规则将首尾 Q-block 配对，后半块反向遍历；SWA 不配对。
- 每任务完成后清空 outstanding VM/LDS 操作，再通过 workgroup barrier 广播下一张
  ticket，并结束原任务的 LDS 生命周期。全体8wave退出 stagger 分支后才取下一任务。
- 每个 CTA 恰好提交一次 completion；最后一个将 header 恢复成 `[grid, 0]`。
  同一 stream 的 kernel 完成顺序保证下次调用看见重置状态，无跨 CTA 自旋或全局 barrier。
- header 是每 `(device, stream handle, grid)` 独立的 **8 byte** device tensor，首次
  调用初始化；LDS 仅增加 **4 byte mailbox**。这不是 KV workspace，不缓存任何输入内容。
- 预热后、预分配 `out`/`lse` 的每次公开调用只有一个 attention dispatch，无初始化
  kernel、counter fill、host metadata copy 或临时 GPU allocation。

## Stream 与 graph 契约

每个 specialization 必须先在捕获 stream 上预热。首次 header 创建发生于 capture 内会
明确报错。图保留捕获时的 counter 地址，**不同 capture stream 独立预热/捕获的图**可以
并发 replay；同一 header 对应的图与普通调用必须串行。不支持把同一 capture/header 的
两个图在不同 stream 同时 replay。只改变当前 stream 不会替换已捕获的 kernel 参数。

所有路径都要求 caller 正确管理跨 stream 数据依赖：共享只读 KV/metadata 可并发读取，
但不可在另一个仍使用它们的 kernel 执行期间修改 page table、prefix、Q/K/V 或 sink。

## 验证与性能证据

最终回归 **236 passed / 6 skipped**（原181项 +55项 scheduler覆盖；6项缺失 AITER
page64 实例）。回归在 [test_pa_prefill.py](test_pa_prefill.py)：

- static/persistent NaN tail、runtime KV 长度、causal odd/even 配对与独立 sink。
- ragged GQA、非零 prefix、padded/head-major Q/O、空Q/空KV/all-masked、共享物理页。
- 大于CU数的工作量、连续重复、counter `[grid,0]`、固定 storage 下改变Q映射、全空任务。
- 两个 stream 分别捕获、每图两个调用、八轮并发 replay，输出/LSE 对完整FP32 reference。
- warmed call 禁止 tensor allocation，验证只有 `_attention_persistent_kernel_0` dispatch。

同进程测试工具 [benchmark_revisions.py](benchmark_revisions.py) 的候选包括
`static persistent 4static 4dynamic`。独立捕获入口为
[profile_scheduling.py](profile_scheduling.py)，用于 ATT/PMC，不用其仪器化时间计性能。
正式采样见 [persistent_results.json](persistent_results.json)、
[persistent_batch4_results.json](persistent_batch4_results.json)、
[persistent_h3_results.json](persistent_h3_results.json)。

MI350X gfx950、同进程/相同5D输入、预分配输出、完整FP32 reference、3次重复、
100轮共同预热后5轮交替20/100 GPU-profiler采样，五轮中位数 **µs**：

| 场景 | Dqk | static | persistent | 4-wave dynamic |
|---|---:|---:|---:|---:|
| NC Q10240/KV2583 | 128 | 259.082 | 257.610 | — |
| NC Q10240/KV2583 | 192 | 297.757 | 298.028 | — |
| causal Q=KV32768 | 128 | 4481.362 | 4497.694 | — |
| causal Q=KV32768 | 192 | 5294.663 | 5286.994 | — |
| SWA+sink Q16K/KV128K/W128 | 128 | 101.649 | 98.352 | — |
| SWA+sink Q16K/KV128K/W128 | 192 | 116.698 | 113.855 | — |
| batch4 Q10240/KV2560/H16 | 128 | 920.327 | 911.605 | 1020.024 |
| batch4 Q10240/KV2560/H16 | 192 | 1073.348 | 1057.581 | 1262.603 |
| H3 segments63225/7、Hq=Hkv14 | 128 | 34648.100 | 31564.267 | 31562.353 |

H3 比 static 降低 **8.90%**；与4dynamic差 **0.006%**，视为等速而非宣称胜出。
SWA改善约2.44%/3.24%，full attention基本持平，短小任务额外队列成本可能更慢。

默认 static 与第1阶段提交基线的新鲜编译：六种 specialization **逐条指令/操作数 hash
一致、全部 mnemonic与资源一致**，无 spill。persistent 资源单独记录：

| 场景 | Dqk | VGPR | SGPR | LDS B | private B | VGPR spill |
|---|---:|---:|---:|---:|---:|---:|
| NC | 128 | 232 | 81 | 99844 | 0 | 0 |
| NC | 192 | 256 | 86 | 149764 | 12 | 2 |
| causal paired | 128 | 232 | 104 | 99844 | 0 | 0 |
| causal paired | 192 | 256 | 106 | 149764 | 0 | 0 |
| SWA+sink | 128 | 240 | 90 | 99844 | 0 | 0 |
| SWA+sink | 192 | 256 | 94 | 149764 | 20 | 4 |

三个全grid PMC样本均确认，D192 SWA **SQ_WAVES 8192→2048**（1024→256 CTA），
**SQ_INSTS_MFMA保持1966080**，LDS bank conflict均0；persistent确实在一个CTA内
处理后续任务，没有漏算。完整其他counter见 [persistent_pmc_results.json](persistent_pmc_results.json)。

persistent 并非总是更快。小工作量会多出队列/finalizer 开销；已有 full causal 配对的
均衡场景收益可能很小。D192原先已达256VGPR，外层工作队列可能引入少量spill；
以实际 ISA metadata 报告，不把 optional persistent 宣称为无成本或自动选择。