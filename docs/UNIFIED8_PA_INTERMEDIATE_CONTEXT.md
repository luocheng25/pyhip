# Unified 8-wave PA MoE Down 中间状态与续跑手册

> 状态：实验继续；当前最佳在同轮ABBA24中比physical4快12.13%，但仍未达到固定绝对时间线。
>
> 更新时间：2026-08-19。
>
> 目标平台：AMD Instinct MI308X / gfx942，ROCm 7.2。
>
> 本文档是当前实验的唯一跨机器handoff入口。生产源码保持严格per-K PA反相与
> `A0/A1/A2 = 0/0/4` CShuffle延迟写回；不要把下文待测patch直接当成生产实现。

## 0. 最新结论（覆盖下文旧待测描述）

本轮要求同时满足：真实512-thread/8-wave workgroup、两组logical 4-wave严格反相、group1
落后一rendezvous、三个K128、每个Stage B源码只有对应K的MFMA，以及相对physical4快10%以上。
实验没有找到同时满足正确性、结构约束和性能目标的候选，因此production kernel未修改。

### 0.1 2026-08-19 checkpoint：suffix XCC + SE-local + XCC rotation-2

production源码保持未修改，SHA256为
`959dd745328c54506e73ec9cbd1aebd91ffa995b6478df328faf714e1bb2a674`。当前最佳候选源码
SHA256为`6ccc253d57367a2c2fa45ef802288c325ebc50c0bbbe762bae9f122b78110171`，可由
`tests/contrib/moe/candidates/unified8_strict_pa_xccrot2.patch`从本checkpoint父提交重建；patch
SHA256为`07c31b20773f84dd3413d2cd0c50f85c740d8e2d2b9c13bd0fbf0e7ccf049448`。

候选保留512 threads、严格4+4反相、group1落后一rendezvous、3个K128、10个真实barrier和
三个各64条MFMA的纯Stage B。资源为254 VGPR、36 SGPR、49,152B LDS、0 scratch；静态ISA为
192条MFMA、39条`buffer_load_dwordx4`和16条output store。full-H3验证覆盖296,448个有效行：
physical output和完整reduced output均逐bit一致，inactive tail未被写入。

H3 exact-valid-grid把2,316个paired WG映射为：前2,304个WG按XCC连续分成四段，每XCC 576个；
XCC内先按四个名义SE槽分组，再把每SE的五个名义CU列转置为长度`29/29/29/29/28`的连续列。
最后把四个48-expert逻辑段相对物理XCC循环移动2位；E192的12个tail WG仍以`3/3/3/3`
分布。核心rank公式为：

```python
se_slot = xcc_local_idx % 4
within_se = xcc_local_idx // 4
cu_column = within_se % 5
cohort = within_se // 5
se_rank = select(cu_column < 4, 29 * cu_column + cohort, 116 + cohort)
logical_xcc = (physical_xcc + 2) % 4
logical_pair = logical_xcc * 576 + 144 * se_slot + se_rank
```

正式结果保存在
`/tmp/production-physical4-vs-xcc-suffix-selocal-xccrot2-clean-abba24.json`：

| 路径 | down中位数 | 有效TFLOPS | paired ratio |
| --- | ---: | ---: | ---: |
| 同轮production physical4 | 2.158432 ms | 429.81 | 1.000000 |
| strict-PA rotation-2 | 1.919671 ms | 483.27 | 0.891783 |

24/24轮candidate均快于control；paired latency降低10.82%，对应吞吐提高12.13%。但是固定验收
仍以早先正式physical4基线`2.085069 ms`为准：候选只提高8.62%，固定目标为`1.895517 ms`，
当前还慢24.154 us（1.274%）。因此候选不得合入production。

natural SE-local的正式结果为`1.920991 ms`，rotation-2只额外降低约1.32 us；该微增益在短测
中为paired ratio `0.994366`，但绝对中位受10-buffer allocation和同轮调用位置影响明显。fresh
SE-local ATT dispatch为`1.823170 ms`（仅解释性，不能作为计时），steady MFMA busy由suffix的
90.95%提高到95.14%，fixed-window physical union busy由90.49%提高到95.00%。剩余idle主要为
`other_dependency_stall 37.13%`、`structural_tail 24.64%`、`vmem_issue_stall 23.23%`，严格
VMEM-wait witness仍为0。

本checkpoint已关闭：SE列逆序/相位、20-column与dual-column映射、运行时CU-ID映射、exact-H3
双kernel、packed FMA、K2 high-first、VMEM credit、output store split/order的大部分扫描、K2
VMEM/DS burst拆分和read interleave、compute-stage mask不对称、group1 `5/3` store split。
当前scratch worktree中存在尚未编译/计时的group1 `3/5`草稿，它不是有效候选，也不属于本提交。

重建当前最佳：

```bash
git worktree add --detach /tmp/unified8-strict-pa-xccrot2 HEAD
git -C /tmp/unified8-strict-pa-xccrot2 apply \
  "$PWD/tests/contrib/moe/candidates/unified8_strict_pa_xccrot2.patch"
sha256sum \
  /tmp/unified8-strict-pa-xccrot2/src/contrib/flydsl/moe_gemm_splitk.py
```

预期源码SHA256为
`6ccc253d57367a2c2fa45ef802288c325ebc50c0bbbe762bae9f122b78110171`。

### 0.2 2026-08-18增量：complementary packed store（历史）

当前最佳有效候选位于`/tmp/unified8-paired-sharedw`，源码SHA256为
`647dd44e1d57f7b2c274ebf21560858a79b05fa371b3c9f79af2f04520e90945`。它保持真实
512-thread/8-wave、严格反相和10个真实barrier，只改变上一N块的packed MUBUF写回相位：

```text
group0 A0/A1/A2 = 0/2/6
group1 A0/A1/A2 = 0/6/2
```

对应control位于`/tmp/unified8-paired-store26`，两组均为`0/2/6`，源码SHA256为
`182f59b38ca1ce931107ab573522718d8de617f98cc1b8736c15f6b86746e631`。受控
10-buffer ABBA16结果为：

| 版本 | down中位数 | paired ratio中位数 | 有效TFLOPS |
| --- | ---: | ---: | ---: |
| paired control | 2.072853 ms | 1.000000 | 447.55 |
| complementary store | 2.027153 ms | 0.979634 | 457.64 |

10份有效physical rows、完整reduced outputs均逐bit一致，inactive tail未被写入。候选资源为
`96 VGPR + 128 AGPR`、49,152B LDS、0 scratch；静态ISA仍为192条MFMA和10个barrier。
相对正式physical4的2.085069 ms，它仅降低约2.78%延迟；10%目标仍是1.895517 ms，当前候选
还需降低约6.9%。因此不得合入production。

complementary性能扫描中，`A1/A2`分别为`0/8 vs 8/0`、`1/7 vs 7/1`、
`2/6 vs 6/2`、`3/5 vs 5/3`时，短测paired ratio依次为`1.017577`、`0.994213`、
`0.979634`、`0.992727`；只有最优的`2/6 vs 6/2`继续完成了完整正确性和ABBA16门禁。
下文“组间非对称store被否证”只描述更早的vector-major producer-only布局，不适用于这个保留
原physical/reduced输出契约的packed MUBUF调度。

fresh ATT的匹配对比为：

| 指标 | paired control | complementary store |
| --- | ---: | ---: |
| steady MFMA busy | 81.82% | 83.23% |
| lifecycle MFMA busy | 76.47% | 77.71% |
| physical-union MFMA busy | 80.43% | 81.89% |
| `VMEM-store + barrier` idle cycles | 1,924,796 | 556,528 |
| `barrier + barrier` idle cycles | 3,391,892 | 3,165,404 |
| `VMEM-load + barrier` idle cycles | 2,764,584 | 2,919,520 |

store错峰共减少约1,006,524 steady MFMA-idle cycles，但剩余空洞已迁移到barrier/barrier、
load/barrier和wait-vmcnt。fixed-window归因中，complementary的MFMA-idle由
`other_dependency_stall 41.51%`、`vmem_issue_stall 24.74%`、`structural_tail 20.77%`
主导；oracle上界由control的16.31%降到13.48%，residual由3.26%升到4.63%，严格
VMEM-wait witness仍为0。因此继续只扫描A1/A2 store比例没有证据支持达到目标。

唯一尚待受控判定的store探针位于`/tmp/unified8-paired-a0`，源码SHA256为
`f7e69e30469d8996ae76806da5b578fd39f91e6d70302b9e906df2c03dfc0d48`：

```text
group0 A0/A1/A2 = 1/1/6
group1 A0/A1/A2 = 0/6/2
```

它已通过AST/Pylance检查、真实FlyDSL编译和正式shape的paired单buffer门禁：296,448个有效
physical rows、完整reduced output均与complementary候选逐bit一致，10,816个inactive tail rows
未被写入。旧`check_unified8_n4096.py`也显示两候选逐bit相同，但其单组PyTorch oracle不适用于
paired双组输出契约，因此该旧脚本整体退出。A0仍未通过10-buffer门禁；正式ABBA2在任何编译前
被硬件门禁阻止：当时全部8张卡均由外部作业占用约79% VRAM。待GPU4恢复到不超过20% VRAM后，
必须从同一ABBA2命令重跑，不得绕过空闲门禁或引用污染状态下的计时。

所有早期N-split timing均无效：相关变体存在两组工作重叠和越界写，未实现同一输出契约。
不要引用这些计时；任何N-split路线必须先重新证明边界、完整写覆盖和inactive-tail不变。

正式24轮、10-buffer、ABBA基线为：

| 路径 | down中位数 | 相对physical4 |
| --- | ---: | ---: |
| physical4 | 2.085069 ms | 1.000000 |
| production unified8 | 2.286270 ms | 1.094621 |

consumer ratio约为`0.99945`，combined ratio为`1.072512`。目标时间是
`2.085069 / 1.10 = 1.895517 ms`，production仍需缩短约17.1%。

strict-PA oracle证明计算本身不是硬件MFMA吞吐瓶颈：

| oracle | down中位数 | 相对production control |
| --- | ---: | ---: |
| compute-only（无output store） | 1.743390 ms | 0.767612 |
| compute + routing/BF16（无output store） | 1.807410 ms | 0.797958 |

这意味着达到目标只剩约`0.088 ms`的完整输出预算；当前可实现的输出路径远超该预算。

最接近的producer-only下界是vector-major packed输出，布局为
`[N512 block][vector_index][thread_id][8 BF16]`。它保留所有真实barrier、seed/drain、三个K128
和MFMA-only Stage B，但没有matching consumer，正式运行使用`--skip-correctness --down-only`，
因此不能采纳：

| vector-major候选 | down约值 | 相对production |
| --- | ---: | ---: |
| MUBUF、policy2、统一A2 `0/0/8` | 2.1833 ms | 0.96024 |
| group0 A2 / group1 A1 | 2.2197 ms | 0.97311 |
| group0 A1 / group1 A2 | 2.2026 ms | 0.96775 |
| `global_store_dwordx4 ... nt` | 2.2355 ms | 0.98216 |

最后一个候选的最终ISA已确认是16条静态`global_store_dwordx4`，并非后端折回MUBUF。
组间非对称store和global-store指令族都被ABBA8否证；即使忽略consumer和正确性，最佳producer
仍比目标慢约15.2%。因此没有为vector-major布局实现consumer，也没有对这些下界做ABBA24。

fresh vector-packed ATT显示steady MFMA busy由production的`77.51%`提高到`81.29%`，但idle
的`61.25%`已迁移为VMEM stall/wait，`26.44%`为barrier imbalance。最大joint blocker是
`VMEM-store + peer barrier`（5,569,168 cycles，占idle 54.65%）；非对称store的实测退化说明
简单错峰不能抵消缩短store retirement距离的代价。

本轮已关闭的其他路线包括CShuffle重分布、pair barrier、dual4、K192、priority、提前K0/K1
prefetch、MFMA32、transposed MFMA/DPP、register CShuffle和cache-policy扫描。除非底层约束改变
或出现新的physical-union ATT证据，不要重复这些实验。第7、8节的“待测”描述仅保留为早期
检查点历史，已由本节覆盖。

## 1. 五分钟续跑

### 1.1 确认版本和生产源码

本检查点提交包含本文档；在新机器上应先检出该提交，再核对生产源码哈希：

```bash
cd /path/to/pyhip
git rev-parse HEAD
sha256sum src/contrib/flydsl/moe_gemm_splitk.py
```

本文档生成时的版本边界：

| 仓库/组件 | 版本 |
| --- | --- |
| pyhip父提交 | `238f871` |
| pyhip分支 | `luocheng/try-opt-down-308` |
| 生产源码SHA256 | `959dd745328c54506e73ec9cbd1aebd91ffa995b6478df328faf714e1bb2a674` |
| FlyDSL提交 | `eb7d69c18f8675c4aa26e8fa01b3277f35a3b57f` |
| llvm-project提交 | `7f77ca0dbda4abbf9af06537b2c475f20ccd6007` |

`pyhip父提交`是中间检查点提交前的HEAD。实际handoff提交应使用“包含本文档的提交”，
由`git log -1 --oneline -- docs/UNIFIED8_PA_INTERMEDIATE_CONTEXT.md`获得。

### 1.2 先做功能门禁

选择一张空闲卡，以下示例沿用物理GPU 4：

```bash
cd /path/to/pyhip
export HIP_VISIBLE_DEVICES=4
export PYTHONPATH=src:.
export FLYDSL_RUNTIME_ENABLE_CACHE=0

pytest -q tests/contrib/moe/test_flydsl_moe_down.py
python tests/contrib/moe/check_unified8_n4096.py
```

预期：MoE down完整文件`13 passed`；N4096输出包含：

```text
N=4096 blocks=8 physical_bit_equal=True rel_l2=0.00405590
```

N512覆盖单N块drain，N1024覆盖跨N稳态延迟写回，N4096覆盖8个N512块。
任何候选都必须先通过这三层，再看性能。

### 1.3 选择空闲GPU

```bash
rocm-smi --showuse --showmemuse --csv
```

正式计时前要求目标卡GPU利用率不高于5%，VRAM占用不高于20%，且同一XCD/主机没有
明显外部干扰。当前会话末尾曾出现外部任务占用全部8卡约84% VRAM，因此两个新候选
没有可信性能数字。不要在该状态下补测或引用短测结果。

### 1.4 重建第一个待测候选

先测`0/1/3`，它改动小、仍为40KB LDS。用独立worktree保留当前生产源码作为control：

```bash
cd /path/to/pyhip
rm -rf /tmp/unified8-013

git worktree add --detach /tmp/unified8-013 HEAD
git -C /tmp/unified8-013 apply \
  "$PWD/tests/contrib/moe/candidates/unified8_cshuffle_013.patch"

sha256sum /tmp/unified8-013/src/contrib/flydsl/moe_gemm_splitk.py
```

预期候选SHA256：

```text
bdff6229dce252ee3a886fd21b6e4dcbd6781626bf5be1c71e96f09103715936
```

先做逐bit验证：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. \
python tests/contrib/moe/check_unified8_n4096.py \
  --candidate /tmp/unified8-013/src/contrib/flydsl/moe_gemm_splitk.py
```

### 1.5 对production做正式ABBA

脚本会保存原GPU状态，设置`VECTOR,F8`和1800MHz determinism，运行后恢复原状态：

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python tests/contrib/moe/compare_unified8_candidates.py \
  --physical-device 4 \
  --rounds 24 \
  --control src/contrib/flydsl/moe_gemm_splitk.py \
  --candidate /tmp/unified8-013/src/contrib/flydsl/moe_gemm_splitk.py \
  --output /tmp/unified8-013-vs-production-abba24.json
```

采用门槛不是单次更快，而是：

1. 10份全部有效physical rows和完整reduced output逐bit一致，inactive tail未被写入；
2. down与combined的配对ratio中位数均小于1；
3. IQR与逐轮胜率支持收益，不由单个离群点驱动；
4. 资源未跨occupancy档；
5. fresh ATT显示原因迁移符合假设，而不是只改变了采样噪声。

若`0/1/3`被否证，再按第8节测试`defer2`，不要同时修改两个变量。

## 2. 当前生产状态

### 2.1 问题形状和线程组织

正式Hunyuan/H3 down性能形状：

| 参数 | 值 |
| --- | --- |
| Batch | 32768 |
| TopK | 9 |
| Experts | 193 |
| N | 4096 |
| K | 384 |
| BLOCK_M | 64 |
| physical BLOCK_N | 512 |
| threads/workgroup | 512 |
| waves/workgroup | 8个wave64 |
| TiledMMA | `(8, 1, 1)` |
| K cores/N block | 3个K128 |
| MFMA/K core/wave | 64 |

生产入口是`compile_gemm(..., down_physical_n512=True)`。N512路径复用N256 CShuffle
基础设施，但一个workgroup是真正的8-wave统一kernel，不是两个独立4-wave kernel。

### 2.2 严格per-K PA反相

8 waves分为两个4-wave组。每个K都由真实barrier分为：

- Stage A：当前K的全部非MFMA工作，包括LDS读、weight预取、必要的CShuffle工作；
- Stage B：当前K的64条MFMA及其scheduler约束，不放入CShuffle或其他语义工作；
- A/B边界：`sched_barrier -> s_barrier -> s_setprio -> sched_barrier`。

启动和收尾：

- seed：group1在N0/K0准备前等待，让group0先进入A0；
- steady state：两组始终相差一个rendezvous，一组A时另一组B；
- drain：最后B后由group0补一次barrier，再写最后N块。

“Stage”必须按动态`barrier-to-barrier`区间定义，不能按Python循环、源码基本块或主观标签定义。

### 2.3 当前`0/0/4`跨N时序

生产CShuffle row-pair分布为`A0/A1/A2 = 0/0/4`：

```text
A0(N): 读取A(N,K0)，预取W(N,K1)，其他非MFMA
B0(N): 64 MFMA only
A1(N): 读取A(N,K1)，预取W(N,K2)，其他非MFMA
B1(N): 64 MFMA only
A2(N): 读取A(N,K2)，先预取W(N+1,K0)，再写回N-1的4个CShuffle row-pair
B2(N): 64 MFMA only
post-B2(N): scale/postprocess N，并把bf16结果携带到下一N
A2(N+1): 写回N的4个row-pair
final drain: 写回最后一个N
```

关键不变量：

- `W(N+1,K0)`在`B2(N)`计算K2之前发出，位于`A2(N)`；
- 不能把它推迟到`B2(N)`之后，否则下一个N的K0启动被拉长；
- CShuffle写回发生在weight预取之后；
- 每个Stage B仍严格只有MFMA；
- 最后一块不能依赖不存在的下一N，必须由drain后的epilogue写出。

### 2.4 资源与ISA边界

production N4096/gfx942最终资源：

- `100V + 132A`，metadata/descriptor曾显示`228/228`；
- 40960B（40KB）LDS；
- 0 scratch；
- 2 resident waves/SIMD；
- 192 static MFMA；
- 36 VMEM loads、16 stores、40 DS reads、20 DS writes。

N256 specialization未被N512改动污染：与strict control规范化机器指令逐条一致，共767条。

## 3. 已验证结果

### 3.1 正确性

- N512、N1024 focused回归通过；
- N4096的8个N512块与production/reference逐bit一致；
- N4096相对PyTorch参考`rel_l2=0.00405590`；
- MoE down测试文件完整结果：`13 passed`；
- 正式ABBA的10份全部有效physical rows及完整reduced outputs逐bit一致。

### 3.2 正式24轮性能

协议：GPU4、1800MHz determinism、`VECTOR,F8`、10 buffers、共同gateup、ABBA顺序、
每轮两次control和两次candidate，最后按轮计算candidate/control ratio。

production `0/0/4`相对修正前strict control：

| 指标 | production | strict control | 配对结果 |
| --- | ---: | ---: | ---: |
| down中位数 | 2.287409 ms | 2.335649 ms | ratio 0.986555 |
| useful down吞吐 | 405.574 TFLOP/s | 397.197 TFLOP/s | 24/24轮胜出 |
| down ratio IQR | - | - | 0.979249..0.992356 |
| combined中位数 | 3.085252 ms | 3.139091 ms | ratio 0.984826 |
| combined ratio IQR | - | - | 0.974966..0.993268 |
| combined胜率 | - | - | 22/24 |

独立的另一轮24-round实验一致：down ratio `0.990233`，combined ratio `0.984732`。

本机历史证据（`/tmp`不属于跨机器资产）：

```text
/tmp/unified8-pa-cshuffle-004-compact-vs-strict-abba24.log
SHA256 5bf0352ac301df5ce832b76e7d04eb7d608e2e40ac94469a3331fc3aa936df7a
```

### 3.3 Fresh ATT

production正式trace：

```text
/tmp/moe-unified8-pa-cshuffle-004-att/ui_output_agent_43471_dispatch_18
code.json SHA256 88c6773bd4b6884a5989f2f75d3256e95552e1cf85fa053d8b81b5dfb8170e4a
dispatch duration 2.259369 ms
1856 waves
```

相对strict：

| 动态量 | strict | production 0/0/4 |
| --- | ---: | ---: |
| A0 median | 2328 cycles | 1252 cycles |
| A1 median | 1580 cycles | 1080 cycles |
| A2 median | 1032 cycles | 2084 cycles |
| 所有B span | 1024 cycles | 1024 cycles |
| barrier stall | 33.71M | 31.01M |
| VMEM-load stall | 15.73M | 12.55M |
| total stall | 99.58M | 97.43M |

production总stall分类：MFMA/FMA 33.0%、barrier 31.8%、VMEM-load 13.0%、LDS 6.1%、
LDS-wait 5.8%、VMEM-wait 4.8%、store 4.0%。

slot不平衡仍存在：

- slot0 A0/A1/A2均值：`1739/1258/1777` cycles；
- slot1 A0/A1/A2均值：`1152/1140/2417` cycles；
- 条件seed PC已证明logical wave_group1恰好映射hardware slot1。

这说明下一步应围绕A2工作迁移做单变量实验，但不能仅凭single-wave A2数字决定方向。

## 4. Wave/SIMD stall分解方法论

### 4.1 ATT时间语义

rocprof ATT的gfx9记录为：

```text
[first_attempt, category, stall, duration, pc_index]
```

真正成功issue时间是：

```text
successful_issue = first_attempt + stall
```

不能把`first_attempt`当issue时间；否则会虚增跨wave重叠并把等待错误归到别的阶段。

### 4.2 两层账本

必须同时看两个层次：

1. single-wave ledger：解释某个wave为什么没有发MFMA，定位依赖、wait和阶段边界；
2. physical-SIMD ledger：按真实`(SE, CU, SIMD)`合并两个resident wave slot，回答SIMD是否仍有MFMA可执行。

性能相关的是第二层。single-wave stall不能线性相加，因为两个resident wave的stall可能重叠，
也可能一个wave stall时另一个wave正在发MFMA。

### 4.3 MFMA execution union

本实验把每个成功issue的MFMA绘制为16-cycle执行窗口，并在每个physical SIMD上取union：

```text
busy = union([issue, issue + 16) for both resident waves)
idle = observed SIMD window - busy
```

然后只在idle窗口做原因归属。这样避免把两个wave的相同周期重复计算。

### 4.4 互斥owner ledger

每个非MFMA tick只能归入一个owner，优先级如下：

1. 明确的VMEM issue stall；
2. `wait-vmcnt`；
3. DS/LDS issue stall；
4. `wait-lgkmcnt`；
5. mixed wait；
6. structural tail；
7. MFMA operand/dependency unavailable；
8. 其他dependency stall；
9. 正常非MFMA issue；
10. scheduler/ready或其他。

使用互斥分类的目的不是宣称唯一根因，而是保证账本守恒，避免同一周期被多种stall重复计费。

### 4.5 Joint witness

只有两个resident slots在同一physical SIMD、同一周期都不能提供MFMA时，才形成SIMD idle的
joint witness。优化问题应写成：

```text
哪个owner同时阻塞了两个slot，且通过软件移动能缩短physical idle union？
```

不要写成“哪个single-wave stall总数最大”。例如把A2工作移到A1，如果只是把阻塞从slot1
迁到slot0而没有缩短union，性能不会提升。

### 4.6 原因迁移而非单计数下降

每次候选需要比较：

- physical MFMA busy fraction；
- physical idle及joint witness分布；
- A0/A1/A2和B0/B1/B2动态span；
- barrier、VMEM、LDS、wait、tail之间的迁移；
- resource/occupancy是否变化。

某一stall计数下降但总idle不降，通常只是原因迁移。某个Stage变短但peer Stage变长，也可能
没有收益。ATT负责解释和生成假设，24轮ABBA负责采纳或否证。

### 4.7 实验闭环

固定流程：

1. 从fresh production ATT找最大的physical joint idle owner；
2. 提一个只改变一个因素、可被反证的局部假设；
3. 修改临时候选，不替换production；
4. N512/N1024/N4096逐bit；
5. 检查VGPR/AGPR、LDS、scratch、occupancy和最终ISA；
6. 空闲GPU上4轮短筛只用于淘汰明显退化；
7. 候选看似有益时做24轮10-buffer ABBA；
8. 只有正式ABBA支持后才采fresh ATT；
9. 根据原因迁移决定采纳、否证或提出下一候选；
10. 更新本文档，禁止依靠`/tmp`文件名或聊天记忆继续。

## 5. 性能Profile与机器配置

### 5.1 固定软件/硬件环境

当前机器：8x AMD Instinct MI308X，gfx942，80 CU/卡，约192GiB VRAM/卡。

主机CPU为2x Intel Xeon Platinum 8480C，每socket 56核、每核2线程，共224逻辑CPU；
主机内存约2TiB，2个NUMA node。开发容器为Ubuntu 22.04，容器可见的宿主kernel为
`5.10.134-18.al8.x86_64`。

| 软件 | 版本 |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.9.1+rocm7.2.0.git7e1940d4 |
| torch HIP | 7.2.26015-fc0010cf6a |
| FlyDSL Python包 | 0.2.4 |
| rocprofv3 | 1.1.0 |
| ROCm | 7.2.0 |
| amdgpu driver | 6.16.13 |

### 5.2 GPU状态协议

`compare_unified8_candidates.py`复用`probe_control_k128_hardware.py`：

- 运行前读取GPU利用率、VRAM、clock、perf level、PTL状态/格式、power cap和NUMA状态；
- 保存原状态；
- 要求原performance level为`auto`，并校验`HIP_VISIBLE_DEVICES`恰好映射到被管理的物理卡；
- 启用PTL；
- 设置PTL格式为`VECTOR,F8`；
- `rocm-smi --setperfdeterminism 1800`；
- 执行正确性与ABBA；
- `finally`中reset determinism、恢复auto perf level及原PTL格式/状态；
- 输出initial/managed/restored state，并逐字段核验perf level、PTL格式/状态和NUMA恢复结果。

AMDSMI 26.2.2的默认安装根是：

```text
/tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3
```

新机器路径不同则传：

```bash
--amdsmi-root /path/to/opt/rocm-7.2.3
```

设置clock/PTL通常需要容器拥有对应设备与管理权限。不要用无法恢复原状态的手工命令做正式测试。

NUMA balancing当前是1。脚本允许新机器原值为0或1，但要求实验前后不变化；若设置不同，
control和candidate必须在同一设置下重测，不能与本文数值直接横比。

### 5.3 ABBA为什么这样测

- 10份activation、weight、scale和output轮换，避免同地址缓存成为候选特权；
- 每次down前运行同一份common gateup，模拟真实前序并稳定cache状态；
- 偶数轮`A B B A`，奇数轮`B A A B`，抵消慢漂移与顺序偏差；
- 每轮先求候选/control，再统计24个paired ratios；
- 同时计down-only和down+sorted_sum combined；
- 比较全部有效physical rows及完整reduced output；control/candidate使用不同sentinel，
  同时确认没有漏写，且physical allocation的inactive tail未被写入，不做抽样正确性。

正式性能必须记录raw ratios、median、IQR、wins、源码SHA、GPU状态和恢复状态。

### 5.4 采集fresh ATT

先确认GPU空闲，清理旧目录，然后执行：

```bash
cd /path/to/pyhip
rm -rf /tmp/moe-unified8-att

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. FLYDSL_RUNTIME_ENABLE_CACHE=0 \
rocprofv3 \
  -i tests/contrib/moe/unified8_att.yaml \
  -- python tests/contrib/moe/profile_unified8_att.py \
       --module src/contrib/flydsl/moe_gemm_splitk.py \
       --dispatches 6
```

采集选项以YAML为准；采集后必须从`out_kernel_trace.csv`确认实际被trace的dispatch是
N512 down kernel，而不是sorting、gateup或JIT warmup。

使用官方`rocprof-trace-decoder`将`.att`解码为`ui_output_agent_*_dispatch_*`目录。
仓库不提交decoder二进制；版本应与rocprof 1.1.0/ROCm 7.2兼容。

### 5.5 分析N4096 trace

N4096每个完整wave执行8个N512块，因此必须传`--n-blocks 8`。省略该参数会沿用历史默认16，
阶段ordinal和steady-state窗口都会错误。

```bash
TRACE=/tmp/moe-unified8-att/ui_output_agent_PID_dispatch_ID

python tests/contrib/moe/analyze_down_mfma_slots.py \
  --trace "production=$TRACE" \
  --n-blocks 8 \
  --workers 4 \
  --json /tmp/unified8-production-slots.json \
  --markdown /tmp/unified8-production-slots.md \
  --svg /tmp/unified8-production-slots.svg

python tests/contrib/moe/analyze_down_stall_exposure.py \
  --trace "production=$TRACE" \
  --n-blocks 8 \
  --first-n 2 \
  --last-n-exclusive 7 \
  --json /tmp/unified8-production-exposure.json \
  --markdown /tmp/unified8-production-exposure.md
```

`last-n-exclusive`必须不超过`n_blocks - 1`，因为最后块属于drain边界，不应混入steady-state。
分析器应报告约1856 waves、192 static MFMA，并从occupancy/metadata解析`100V+132A`、
40960B LDS、2 waves/SIMD。

本trace上的已验证输出是：slot分析器`steady_mfma_busy_fraction=0.791456`；exposure分析器
在显式`N2..N6`窗口上`physical_union.mfma_busy_fraction=0.759479`。两者不能直接横比：前者
自动纳入两个resident slot都处于core/core-boundary/tail的全部steady N-loop周期；后者只纳入
指定N范围且恰有完整resident pair的周期。比较候选时必须固定同一分析器、同一窗口和同一参数，
不要把这两个数字当成矛盾或性能变化。

## 6. 已否证方向

不要重复以下实验，除非有新ATT证据或同时改变了明确的底层约束：

| 方向 | 结果 | 结论 |
| --- | --- | --- |
| CShuffle `0/2/2`拆分 | 约退化4% | 平均拆分不等于physical union平衡 |
| production `0/0/4` vs `0/3/1` | down ratio 0.956626，combined 0.977206 | 集中在A2明显优于把3对放A1 |
| CShuffle read-distance/readpipe | 24轮down ratio 1.005482，combined 0.996415 | down明确退化，combined近中性，不采纳 |
| slot1 A2前`setprio 1` | down ratio 1.011678，combined 1.008820 | 提权反而压制peer，不采纳 |
| 各类更早/更晚weight prebarrier | 无稳定收益或破坏正确时序 | 保持`W(N+1,K0)`在A2/B2前 |
| 只看single-wave stall总数选方向 | 与ABBA不一致 | 必须看physical union和joint witness |

`0/0/4`对`0/3/1`的24轮直接ABBA中，down 19/24轮胜、IQR
`0.944449..0.987214`；combined 21/24轮胜。该结果也是“不要因A2最长就盲目把大量工作
搬到A1”的直接证据。

## 7. 当前待测候选一：`0/1/3`

patch：

```text
tests/contrib/moe/candidates/unified8_cshuffle_013.patch
```

语义：将一个CShuffle row-pair从A2移到A1，其余三个仍在A2；B0/B1/B2保持MFMA-only。

已完成：

- patch可从本检查点production干净应用；
- N4096 8-block逐bit通过；
- `rel_l2=0.00405590`；
- 资源约`232/232`，40KB LDS，仍为2 waves/SIMD；
- 工作量与production相同。

仓库patch重建SHA256：

```text
bdff6229dce252ee3a886fd21b6e4dcbd6781626bf5be1c71e96f09103715936
```

未完成：空闲GPU上的4轮筛选、24轮正式ABBA和fresh ATT。它是下一项优先任务。

## 8. 当前待测候选二：`defer2`

patch：

```text
tests/contrib/moe/candidates/unified8_cshuffle_defer2.patch
```

语义：A2立即完成前两个row-pair；后两个只写入两个wave-private LDS槽，跨B2后在A0读回并
store。B2本身仍只有MFMA。

代价与假设：

- CShuffle wave-private LDS由16KB增至32KB；
- workgroup总LDS由40KB增至56KB；
- 资源约`236/236`，仍为2 waves/SIMD；
- production本来就只有一个8-wave WG驻留，因此56KB没有再降低occupancy；
- 目标是把约两份read/wait/store从过长A2移到较短A0。

已完成：patch应用、N4096逐bit、`rel_l2=0.00405590`、静态资源和指令工作量检查。

仓库patch重建SHA256：

```text
5d1bf549121c2dcdc15839d34aa4ab371481ca705af38853e0c0f32c6376b39e
```

未完成：可信计时和fresh ATT。只有`0/1/3`被否证或defer2有更强的新证据时再测它。

重建：

```bash
cd /path/to/pyhip
rm -rf /tmp/unified8-defer2
git worktree add --detach /tmp/unified8-defer2 HEAD
git -C /tmp/unified8-defer2 apply \
  "$PWD/tests/contrib/moe/candidates/unified8_cshuffle_defer2.patch"

HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. \
python tests/contrib/moe/check_unified8_n4096.py \
  --candidate /tmp/unified8-defer2/src/contrib/flydsl/moe_gemm_splitk.py
```

## 9. 修改与采纳规则

- production保持本检查点`0/0/4`，不要为了测试patch直接改主工作树；
- 每次只改一个可解释变量；
- 候选源码、结果JSON、ATT code.json和ISA都记录SHA256；
- 首先逐bit，随后资源/ISA，再性能；顺序不可反转；
- 4轮只淘汰明显回归，不能作为采纳证据；
- 最终采纳至少需要24轮ABBA与fresh ATT；
- 任一候选若让Stage B混入CShuffle/VALU/VMEM语义工作，视为违反当前实验定义；
- 任一候选若把`W(N+1,K0)`移到`B2(N)`之后，视为时序错误；
- 不提交`/tmp`trace或decoder大文件，只提交脚本、patch、文档和必要的小型结果摘要；
- 外部负载出现时停止计时，不“校正”污染数据。

## 10. 本检查点文件边界

本次中间提交应只包含：

```text
src/contrib/flydsl/moe_gemm_splitk.py
tests/contrib/moe/test_flydsl_moe_down.py
tests/contrib/moe/analyze_down_mfma_slots.py
tests/contrib/moe/analyze_down_stall_exposure.py
tests/contrib/moe/check_unified8_n4096.py
tests/contrib/moe/compare_unified8_candidates.py
tests/contrib/moe/profile_unified8_att.py
tests/contrib/moe/unified8_att.yaml
tests/contrib/moe/candidates/unified8_cshuffle_013.patch
tests/contrib/moe/candidates/unified8_cshuffle_defer2.patch
docs/UNIFIED8_PA_INTERMEDIATE_CONTEXT.md
```

明确排除：

- `src/contrib/flydsl/helpers.py`的`thread_idx_override`改动，本路径未引用；
- 已暂存但与本任务无关的`profile_compiled_breakdown.py`和`profile_moe_compile.py`；
- `.vscode`、gpucore、备份源码、decoder、scratch测试及其他未跟踪文件。

## 11. 已知限制

- 本文的formal strict control只保存在本机历史`/tmp`产物，未作为跨机器源码资产提交；
  后续候选应直接以本检查点production为control。
- ATT采集/decoder目录可能因rocprof版本产生不同命名，必须通过kernel trace确认dispatch。
- `profile_unified8_att.py`为确定性trace workload，不是性能计时器；性能结论只来自ABBA脚本。
- production SHA包含工作树中的统一8-wave实现；检出handoff提交后应重新核对SHA。
- 两个候选只有正确性和资源结论，没有性能结论。
