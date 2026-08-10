# AITER ASM FlashAttention 单元测试 — MiniMax-H3 @ MI308X (gfx942)

镜像 `sabreshao/sglang:mxh3_mi308x_0805`
（`sha256:15d00daf75a643245885d962d7244f12f4545475ab8057016ad728693f4d2278`），
其中 AITER `7d604afe5fa7efba63c0dce323b95d9daf2db112`。

shape 取线上真实值：`q=k=v=(63232, 14, 128) bf16`，`cu_seqlens=[0, 63225, 63232]`，
`causal=False`，`scale=0.08838835`。

```bash
HIP_VISIBLE_DEVICES=4 python -m pytest -q /home/niding/h3_attn_kernel_test.py
```

## Kernel

| | GPU kernel | Python 入口 | 二进制 |
|---|---|---|---|
| SDPA | `attn_fwd` | `torch.nn.functional.scaled_dot_product_attention` | `torch/lib/libaotriton_v2.so.0.10.0` + `aotriton.images/` |
| Triton（线上现状） | `_attn_fwd_IS_CAUSAL_0_NUM_Q_HEADS_14_NUM_K_HEADS_14_BLOCK_M_128_BLOCK_N_64_BLOCK_DMODEL_128_RETURN_SCORES_0_ENABLE_DROPOUT_0_IS_FP8_0_VARLEN_1_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0` | `aiter.ops.triton.attention.mha.flash_attn_varlen_func` | Triton JIT，磁盘无 .co |
| ASM（候选） | `aiter::fmha_fwd_hd128_bf16_rtna_group` | `aiter.flash_attn_varlen_func` | `hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_rtna_group.co` |

kernel 名都是 torch profiler 实测读回来的。SDPA 在这个镜像里落到 PyTorch ROCm
的 flash backend（AOTriton），不是 math。

## 正确性

基准 = segment-wise bf16 SDPA（pack 拆开逐段过
`F.scaled_dot_product_attention`，同为 bf16）。

```
kernel     cvt     cos_vs_sdpa   max_abs     tail_cos  cos_vs_triton   max_abs
triton     -       1.000000000  0.000122  1.000000119         (self)
asm_group  RTNE    1.000000000  0.000122  1.000000119    1.000000000  0.000122
asm_group  RTNA    1.000000000  0.000122  1.000000119    1.000000000  0.000122
```

与 SDPA、与 Triton 均为 `cos = 1.000000000`，`max_abs = 1.22e-4`。换过去输出不变。

`tail_cos` 单列：63232 个 token 里 63225 个属于第一段，尾段整个算错整包 cosine
仍有 0.99999，必须单独打分。

`7 passed, 2 skipped in 11.05s`

## 性能

GPU 4，CUDA event，median of 10，28.65 TFLOP。

ASM 二进制有 MI300 和 MI308 两套，aiter 在 `csrc/cpp_itfs/mha_fwd.cu:63-79` 靠
`is_mi308_device()`（PCI chip id）硬判该加载哪套。两套都测：

| 二进制 | kernel | cvt | median | min | max | TFLOPS | vs triton |
|---|---|---|---|---|---|---|---|
| — | triton | - | 191.68 ms | 191.63 | 191.75 | 149.5 | — |
| MI308 | asm_group | RTNE | 205.15 ms | 205.11 | 205.17 | 139.7 | +7.0% |
| MI308 | asm_group | RTNA | 192.52 ms | 192.44 | 192.58 | 148.8 | +0.4% |
| MI300 | asm_group | RTNE | 178.24 ms | 153.12 | 178.72 | 160.8 | −7.0% |
| MI300 | asm_group | RTNA | 173.63 ms | 148.89 | 175.46 | 165.0 | **−9.4%** |

**MI308 二进制没有收益** —— RTNA 打平 triton，RTNE 反而慢 7%。−9.4% 完全
依赖换成 MI300 的 .co：

```bash
cp /sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI300/*.co \
   /sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308/       # 原件备份在 MI308.orig/
```

## 与 FlyDSL 4-wave 的同机复测

`test_pa_prefill.py`新增`PA_CASE=h3`，使用相同segments、heads、dtype、scale和
`3 warmup + 10 CUDA-event samples + statistics.median`协议。两边TFLOPS均按
`sum(4*S_i^2*D*H)=28.653368 TFLOP`计算。

当前物理GPU4（`fclk=1300MHz,mclk=900MHz`）同一进程、同一输入结果：

| 实现 | median | TFLOPS |
|---|---:|---:|
| FlyDSL 4-wave paged varlen | 181.464 ms | 157.901 |
| AITER Triton varlen | 202.460 ms | 141.526 |
| AITER ASM group RTNA | 190.815 ms | 150.163 |
| AITER ASM split RTNA | 182.247 ms | 157.223 |

按吞吐计算，FlyDSL比Triton高11.57%、比group RTNA高5.15%、比split RTNA高0.43%；
按延迟计算则分别降低10.37%、4.90%和0.43%。绝对值与上表历史数据不可直接横比：镜像、
AITER二进制和GPU时钟不同；同机表用于实现间公平比较。

仅作跨环境参考：当前FlyDSL `157.901T`比上表历史Triton `149.5T`高5.6%、比历史
MI308 group RTNA `148.8T`高6.1%，但比替换MI300 `.co`后的历史group RTNA
`165.0T`低4.3%。该比较不控制镜像、二进制和时钟，不能替代同机表。

接入过程中修复了FlyDSL K手工寻址的多KV-head page stride：物理page跨度需要乘
`num_kv_heads`。修复后真实H3整包cosine为`0.9999973`、max-abs为`2.44e-4`，7-token
tail cosine为`1.0000001`、max-abs为`0`。
