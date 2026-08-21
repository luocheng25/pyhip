# 当前N512与9049ddb性能对比（2026-08-21）

## 对比定义

- 旧实现：commit `9049ddb723a1428d8dfb4c75e352d9b65bc9db56`，kernel SHA256 `4951a4878bbd290a8dce702180675a545b9478ba85039c4be6421ba360cb280c`。
- 当前实现：实验worktree kernel SHA256 `59ddf290a4820a1e02bfe3cec80ff7748e2db2c9b55e30b2a1dd2c5448188398`。
- Hy3：9049使用其exact-shape single-M N512；当前使用M64 N-split N512。
- 其余case：9049使用原通用M128-paired N512；当前使用M64 N-split N512，Xiaomi额外启用persistent pair与early prefetch。
- 两边分别使用其正确metadata：Hy3为9049 M64 sorting 4801 tasks、当前BM128 sorting展开4994 tasks；其他case两边均使用BM128 metadata。

## 协议

- B32768，10 rotating buffers，24轮ABBA。
- 每实现每phase 48个绝对样本，24个配对ratio。
- 每样本前运行相同model-shaped gateup。
- 1800MHz performance determinism，PTL `Enabled / VECTOR,F8`。
- 每个case要求初始GPU `auto`、busy<=5%、VRAM<=20%，结束恢复原状态。
- 比较最终标准row-major `sorted_sum`数学输出；五个case最大relative-L2均<=6.62e-4。

## 结果

`当前/9049`小于1表示当前实现更快。

| Case | Phase | 9049 ms | 当前 ms | 9049 TFLOPS | 当前 TFLOPS | 当前/9049 | IQR | 当前胜轮 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Hy3 | Down | **1.384806** | 1.531986 | **334.961** | 302.781 | 1.104034 | 1.094697..1.113589 | 0/24 |
| Hy3 | Down+sum | **2.125708** | 2.256949 | **218.213** | 205.524 | 1.065036 | 1.056117..1.074230 | 0/24 |
| Qwen3.5 397B | Down | **5.992322** | 7.925470 | **229.358** | 173.414 | 1.321078 | 1.313857..1.325224 | 0/24 |
| Qwen3.5 397B | Down+sum | **6.771806** | 8.721273 | **202.958** | 157.590 | 1.284032 | 1.272947..1.291860 | 0/24 |
| Qwen3.5 35B | Down | **2.484910** | 3.338614 | **221.238** | 164.666 | 1.340841 | 1.333348..1.345419 | 1/24 |
| Qwen3.5 35B | Down+sum | **2.824912** | 3.669516 | **194.610** | 149.817 | 1.298384 | 1.290702..1.302648 | 0/24 |
| Xiaomi | Down | 2.617870 | **2.553090** | 315.002 | **322.994** | **0.974734** | 0.959372..0.983373 | 24/24 |
| Xiaomi | Down+sum | 3.621434 | **3.555254** | 227.709 | **231.948** | **0.976588** | 0.967418..0.994060 | 20/24 |
| H3 | Down | **1.662267** | 3.585555 | **372.067** | 172.491 | 2.160334 | 2.142964..2.175531 | 0/24 |
| H3 | Down+sum | **2.201710** | 4.138897 | **280.907** | 149.430 | 1.878333 | 1.869378..1.886348 | 0/24 |

Combined TFLOPS使用Down有效FLOPs除以`Down+sorted_sum`时间，是等效吞吐。

## 结论

- `9049ddb`在Hy3、Qwen3.5 397B、Qwen3.5 35B、H3上更快。
- 当前实现只在Xiaomi胜出：down快2.53%，combined快2.34%。
- Hy3应使用9049的真8-wave single-M N512 specialization：当前慢10.40% down、6.50% combined。
- 对非Hy3，9049与当前比较的是两种不同的通用N512分解。9049保留M128-paired N512；当前是M64 N-split，因此K512和H3 PTPC上当前明显更慢。
- 跨五case配对ratio几何平均为down 1.3272、combined 1.2664，但该平均受H3算法差异影响，不应用作统一dispatch策略；应按shape选择。

## Hy3 ISA差异

| 指标 | 9049 single-M | 当前N-split |
| --- | ---: | ---: |
| Tiled MMA | 8-wave N512 `(8,1,1)` | 2个4-wave N256 `(4,1,1)` |
| VGPR | 128 | 144 |
| LDS | 32KB | 32KB |
| Scratch | 0 | 0 |
| Occupancy | 4 waves/SIMD | 3 waves/SIMD |
| MFMA/load/store | 96/24/8 | 96/24/8 |
| Barrier | 2 | 2 |
| setprio | 8 | 1 |

9049还使用M64 sorting、512-thread activation copy、A swizzle shift3和`amdgpu-waves-per-eu=4,4`；当前Hy3使用BM128 sorting展开、384-thread activation copy、shift4、small-K barrier-free与immediate-store。
