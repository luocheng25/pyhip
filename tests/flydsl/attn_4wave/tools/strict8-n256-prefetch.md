# 严格8-wave N256 PTPC：store split与K128预取实验

[`probe-8wave-n256-prefetch.py`](probe-8wave-n256-prefetch.py)将physical N256 PTPC的稳态工作量抽成
严格8-wave反相模型，用来验证：

1. delayed store由`2/2/4`改为`3/3/2`是否更均衡；
2. K128 weight预取拆成K64 head/tail是否有利；
3. K2在tail前加入`vmcnt(6)` credit水位是否有利。

## 模型

一个64 KiB、8-wave WG由两个4-wave子组组成，真实barrier令两个子组严格反相。每个K128 core、
每条memory wave固定执行：

- 8条FP8 weight `buffer_load_dwordx4`，对应每wave `N64 x K128`；
- 8条activation `ds_read_b128`；
- 64条生产同款`v_mfma_f32_16x16x32_fp8_fp8`；
- 上一N块共8条delayed output store，在K0/K1/K2间分配；
- K2额外1条PTPC scale load。

总工作量在所有variant间完全相同：三core合计25条load、8条store、24条LDS read和192条MFMA。
资源为30 SGPR、104 VGPR、64 KiB LDS、0 scratch；实际为一个8-wave WG/CU、每SIMD两个wave。

## 请求账本

严格反相保证任一时刻只有4条memory wave。

### `2/2/4`

| core | 每wave请求 | 名义请求/CU |
|---|---:|---:|
| K0 | 8 weight + 2 store = 10 | 40 |
| K1 | 8 weight + 2 store = 10 | 40 |
| K2 | 8 weight + 4 store + 1 scale = 13 | 52 |

### `3/3/2`

| core | 每wave请求 | 名义请求/CU |
|---|---:|---:|
| K0 | 8 weight + 3 store = 11 | 44 |
| K1 | 8 weight + 3 store = 11 | 44 |
| K2 | 8 weight + 2 store + 1 scale = 11 | 44 |

所以从credit账本看，**`3/3/2`确实更均衡**：它把K2从52降到44，代价是K0/K1从40升到44；三者均
保持在约48-request拐点以下。

## 测试variant

| schedule | 含义 |
|---|---|
| `burst` | store、scale、8条weight load连续发射，最后`vmcnt(0)` |
| `split` | store、scale、4条head load，穿插8条LDS read，再发4条tail load，最后`vmcnt(0)` |
| `credit` | 与split相同，但`2/2/4`的K2在tail前加入`vmcnt(6)`；`3/3/2`也做相同消融 |

`credit`只在K2插入额外wait。所有case都保持严格反相；ATT中的活动同stage重叠均为0。

## 实验环境

MI308X `gfx942`，GPU4，ROCm 7.2.3，rocprofv3 1.1.0，650 W power cap，PTL `VECTOR,F8`，
performance determinism 1800 MHz。ATT每case为128 rounds、两个dispatch、32个物理SIMD pair；全部
0 placement failure、0 trace failure。

无ATT wall-time采用同进程六配置、三轮预热、24轮正序/逆序交错。

## ATT结果

三项总和为：VMEM issue stall + 最终`vmcnt(0)` stall + peer compute-barrier stall。

| variant | issue stall | wait stall | peer barrier | 三项总和 / `224-burst` |
|---|---:|---:|---:|---:|
| `224-burst` | 4.152M | 16.600M | 16.002M | 1.0000 |
| `332-burst` | 4.155M | 15.044M | 14.661M | **0.9212** |
| `224-split` | 2.534M | 17.171M | 16.186M | 0.9765 |
| `332-split` | 2.564M | 17.811M | 16.847M | 1.0127 |
| `224-credit` | 2.501M | 17.620M | 16.676M | 1.0012 |
| `332-credit` | 2.571M | 17.387M | 16.350M | 0.9878 |

### `3/3/2 burst`对`2/2/4 burst`

- issue stall几乎不变：`1.0006x`；
- completion wait下降9.38%；
- peer compute-barrier下降8.38%；
- 三项stall账本下降7.88%。

core级变化符合请求账本：`2/2/4`的K2有52个名义请求，stage span中位2436 cycles、wait中位1456
cycles；`3/3/2`将K2降到44个请求，stage span降至2158 cycles、wait降至1288 cycles。K0/K1虽从40
增到44请求，但三core整体更接近同一时长，barrier不均衡下降。

因此，**在严格8-wave反相模型中，`3/3/2`的ATT结果优于`2/2/4`。**

## 稳态ATT stall账本

为排除入口prologue和末尾drain，稳态分析对每条wave丢弃前8轮和后8轮，只保留128轮中的112轮。
归一口径分为：

- **owner cycles/wave/round：** 两条wave各自stall的累计owner账本；
- **physical joint cycles/pair/round：** memory wave的VMEM blocker与same-SIMD compute peer的barrier
   stall在绝对时间上的交集，只计一次物理周期。

### burst的core级物理joint blocker

| split | K0 | K1 | K2 | 三core总量 | max-min | CV |
|---|---:|---:|---:|---:|---:|---:|
| `2/2/4` | 1089.44 | 1027.28 | **1375.56** | 3492.29 | 348.29 | 13.03% |
| `3/3/2` | 1054.06 | 1015.07 | **1105.88** | **3175.00** | **90.81** | **3.51%** |

`3/3/2 burst`相对`2/2/4 burst`：

- 三core物理joint blocker总量下降9.09%；
- core间max-min下降73.9%；
- CV从13.03%降到3.51%；
- owner三项总量下降7.74%。

这直接证明`3/3/2`在严格反相下不仅静态请求数更均衡，动态稳态stall也更均衡。

### `2/2/4 burst`稳态owner账本

单位为cycles/wave/round：

| core | 请求/CU | VMEM issue | `vmcnt` wait | compute barrier | physical joint/pair/round |
|---|---:|---:|---:|---:|---:|
| K0 | 40 | 289.98 | 1295.31 | 1213.80 | 1089.44 |
| K1 | 40 | 293.97 | 1294.33 | 1343.78 | 1027.28 |
| K2 | 52 | **428.14** | **1448.38** | 1339.90 | **1375.56** |

K2同时表现出更高的producer-side issue stall和consumer-side completion wait，符合52请求跨过credit拐点
的预期。

### 六variant稳态汇总

均相对`224-burst`：

| variant | physical joint ratio | owner三项ratio |
|---|---:|---:|
| `224-burst` | 1.0000 | 1.0000 |
| `332-burst` | **0.9091** | **0.9226** |
| `224-split` | 0.9523 | 0.9771 |
| `332-split` | 0.9981 | 1.0141 |
| `224-credit` | 0.9780 | 1.0003 |
| `332-credit` | 0.9724 | 0.9894 |

稳态下仍是`332-burst`最佳；head/tail拆分的issue收益被更晚tail带来的wait/barrier抵消。

### 为什么head/tail拆分失败

K64 head/tail拆分把连续VMEM issue stall明显降低约38%--40%，但tail发得更晚：

- `224-split` wait增加3.44%，peer barrier增加1.15%；
- `332-split` wait增加7.30%，peer barrier增加5.28%；
- 最终没有净收益。

这里8条LDS read并没有提供足够的有用隐藏距离；它只是把最后四条weight load推迟到更靠近
`vmcnt(0)`的位置。**预取距离应以最后一批load到consumer的距离衡量，而不是首条head load的位置。**

### 为什么`vmcnt(6)`失败

K2的credit wait中位只有4 cycles，说明插入点通常尚未因credit耗尽而长时间阻塞；但这条wait增加了
控制依赖，并继续推迟tail。`224-credit`的三项stall总量回到基线附近，wall-time反而退化。

## 无ATT ABBA24

| variant | median | 相对`224-burst` | 同轮胜场 |
|---|---:|---:|---:|
| `224-burst` | 0.969104 ms | 1.000000 | control |
| `332-burst` | 0.968803 ms | **0.999690** | 11/24 |
| `224-split` | 0.994503 ms | 1.026209 | 0/24 |
| `332-split` | 0.998603 ms | 1.030440 | 0/24 |
| `224-credit` | 0.996444 ms | 1.028212 | 1/24 |
| `332-credit` | 0.999524 ms | 1.031390 | 1/24 |

`332-burst / 224-burst`的同轮ratio中位为`1.000397`，IQR为`0.998341--1.003073`，11/24胜。
因此wall-time结论是**性能中性**，不能仅凭ATT账本宣称提速。ATT说明它消除了K2重尾，但总kernel时间还
包含MFMA、LDS、控制流和固定开销，约8%的局部stall账本差异没有形成可分辨的端到端收益。

## 为什么历史4-wave选择`2/2/4`

历史physical N256不是严格反相：两slot同时stage0约35.8%，反相覆盖约44%--49%。在该条件下，后置
store可改变两个独立WG的VMEM竞争与phase关系，实测`2/2/4`优于`3/3/2`。

本实验则由单个8-wave WG和barrier保证4 memory + 4 compute waves。相位机制不同，所以“账本更均衡”
在ATT中兑现，但wall-time只达到中性。不能把本微基准的`3/3/2`直接晋级到现有4-wave生产kernel。

## 推荐

针对严格8-wave N256 PTPC：

1. **store split优先用`3/3/2 burst`作为候选。** 它把三个core统一为44 requests/CU，ATT账本最佳，
   wall-time不劣于`2/2/4`但尚无显著提升。
2. **不要采用当前K64 head/tail实现。** 它降低issue stall却缩短tail到consumer的距离，wall-time稳定
   退化2.6%--3.0%。
3. **不要在此位置加入`vmcnt(6)`。** 实测wait本身只有约4 cycles，没有解决主瓶颈。
4. 真正的下一步是保持8条weight load尽早发出，同时把独立LDS/VALU/store调度到load之后、最终
   `vmcnt(0)`之前；不要把后4条load推迟。
5. 若迁移到真实M128xN256 kernel，必须重新做正确性和ABBA；该模型不含完整CShuffle、routing scale
   后处理和expert边界。

## 复现

ATT配置：[`strict8-n256-prefetch-att.yaml`](strict8-n256-prefetch-att.yaml)。例如：

```bash
rm -rf /tmp/strict8-n256-prefetch-att
HIP_VISIBLE_DEVICES=4 PYTHONPATH=src:. PYHIP_JIT_LOG=0 \
rocprofv3 -i tests/flydsl/attn_4wave/tools/strict8-n256-prefetch-att.yaml \
  --att-library-path /tmp/h3-rocprof-decoder/opt/rocm/lib -- \
  /tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-8wave-n256-prefetch.py run \
  --device 0 --rounds 128 --store-split 332 --schedule burst \
  --data-mib 512 --dispatches 2 \
  --json /tmp/strict8-n256-placement.json \
  > /tmp/strict8-n256-capture.log 2>&1

/tmp/pyhip-flydsl024/bin/python \
  tests/flydsl/attn_4wave/tools/probe-8wave-n256-prefetch.py analyze \
  --att-root /tmp/strict8-n256-prefetch-att --rounds 128 \
  --store-split 332 --schedule burst \
  --capture-log /tmp/strict8-n256-capture.log \
  --json /tmp/strict8-n256-analysis.json
```

所有受控实验结束后，GPU4均恢复到原始`auto + F8,VECTOR`状态。
