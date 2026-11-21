"""mxfp4 MoE 权重的加载时准备（``prepare_weights``，每份权重只做一次）。

输入是 checkpoint 的自然布局（单卡分片后），mxfp4 = E2M1 + 每 32 个 K 一个 E8M0：
  w1 uint8 [E, 2I, H/2]（一字节两个 E2M1，低半字节是偶数 k；前 I 行 gate、后 I 行 up），
  w1_scale uint8 [E, 2I, H/32]；w2 uint8 [E, H, I/2]，w2_scale uint8 [E, H, I/32]。
  也接受 torch.float4_e2m1fn_x2 / torch.float8_e8m0fnu，按字节读取。

输出 ``MXFP4Weights``，decode（M1）与 prefill（M2）共用。对每个 GEMM（N 个输出通道、归约维 K）：
  w       uint8 [E, N/16, K/16, 16, 8]：与 FP8 权重的 (16,16) preshuffle 元素顺序相同，一行的 16 个 K
          占 8 字节；每个 dword 的 8 个 nibble 改存为 字节 j = k_j | k_{j+4} << 4（PR #64 的 fd1），
          -0 改为 +0，被 flush 的组整组写 0。
  delta   uint8 [E, N/64, K/32, 16, 4]：min(c_max - code, 11) * 8；dword (n64, g, l) 的字节 j 属于
          第 64*n64 + 16*j + l 行的第 g 组。c_max 是该行所有组 E8M0 code 的最大值。
  factor  float32 [E, N]：2^(c_max - 132)（c_max < 6 时为 fp32 次正规数，精确）；补零的通道为 1。

数值（与 PR #64 的 fd2 相同）：kernel 把 nibble 的 |E2M1| 经查找表变成 FP8 E4M3FNUZ 值
|E2M1| * 2^(5 - d)（d = c_max - code <= 11 时都是正规数，精确），再精确展开成 BF16；fp32 累加结果
乘 factor 得 E2M1 * 2^(code - 127)，即 mxfp4 精确反量化。d >= 12 的组（不到该通道最大组 scale 的
2^-11）解码为 0（flush），参考实现施加同一规则。

I 补零到 64 的倍数（TP4 的 160 -> 192；Down 的 K 与 delta 的 64 行分块都要求）：补的 nibble 为 0，
补的组 code 记 0（不改变真实通道的 c_max），补的通道 factor 为 1，结果与不补零精确等价。
E8M0 的 255（NaN）直接报错。
"""

import dataclasses

import torch

GROUP_K = 32
FLUSH_DELTA = 12
MAX_DELTA = FLUSH_DELTA - 1
FACTOR_EXP_OFFSET = 132  # 127（E8M0 偏置）+ 5（查找表里 |E2M1| 乘了 2^5）
INTER_ALIGN = 64

_FP4_DTYPES = tuple(getattr(torch, name) for name in ("float4_e2m1fn_x2",) if hasattr(torch, name))
_E8M0_DTYPES = tuple(getattr(torch, name) for name in ("float8_e8m0fnu",) if hasattr(torch, name))


@dataclasses.dataclass(frozen=True)
class MXFP4Weights:
    """``prepare_weights`` 的结果；字段含义见模块说明。``inter`` 是真实 I，``inter_padded`` 是 kernel 的 I。"""

    w1: torch.Tensor
    w1_delta: torch.Tensor
    w1_factor: torch.Tensor
    w2: torch.Tensor
    w2_delta: torch.Tensor
    w2_factor: torch.Tensor
    inter: int
    inter_padded: int

    @property
    def experts(self):
        return self.w1.shape[0]

    @property
    def hidden(self):
        return self.w2.shape[1] * 16


def _as_bytes(t, dtypes, name):
    if t.dtype in dtypes:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8 (or {[str(d) for d in dtypes]}), got {t.dtype}")
    return t.contiguous()


def channel_factor(c_max):
    """2^(c_max - 132) 的 fp32 位型（c_max 为 int32 张量，0..254）。"""
    normal = (c_max - (FACTOR_EXP_OFFSET - 127)).clamp_min(1) << 23
    subnormal = torch.ones_like(c_max) << (c_max + 149 - FACTOR_EXP_OFFSET).clamp(0, 22)
    return torch.where(c_max >= FACTOR_EXP_OFFSET - 126, normal, subnormal).view(torch.float32)


def _prepare_gemm(w, codes, real_rows, chunk):
    """一个 GEMM：自然布局 w [E, N, K/2]、codes [E, N, K/32] -> (w, delta, factor)。"""
    experts, n, kb = w.shape
    k = 2 * kb
    groups = k // GROUP_K
    out_w = torch.empty((experts, n // 16, k // 16, 16, 8), dtype=torch.uint8, device=w.device)
    out_d = torch.empty((experts, n // 64, groups, 16, 4), dtype=torch.uint8, device=w.device)
    out_f = torch.empty((experts, n), dtype=torch.float32, device=w.device)
    for begin in range(0, experts, chunk):
        end = min(experts, begin + chunk)
        e = end - begin
        code = codes[begin:end].to(torch.int32)
        c_max = code.amax(dim=-1)
        delta = c_max.unsqueeze(-1) - code
        b = w[begin:end]
        nib = torch.stack((b & 0xF, b >> 4), dim=-1).view(e, n, groups, GROUP_K)
        nib = nib.masked_fill(nib == 8, 0).masked_fill((delta >= FLUSH_DELTA).unsqueeze(-1), 0)
        nib = nib.view(e, n, k // 8, 2, 4)
        fd1 = nib[..., 0, :] | (nib[..., 1, :] << 4)
        out_w[begin:end] = fd1.view(e, n // 16, 16, k // 16, 8).permute(0, 1, 3, 2, 4)
        d8 = (delta.clamp(max=MAX_DELTA) * 8).to(torch.uint8)
        out_d[begin:end] = d8.view(e, n // 64, 4, 16, groups).permute(0, 1, 4, 3, 2)
        out_f[begin:end] = channel_factor(c_max)
    out_f[:, ~real_rows] = 1.0
    return out_w, out_d, out_f


def prepare_weights(w1, w1_scale, w2, w2_scale, *, chunk=32):
    """自然布局的 mxfp4 MoE 权重 -> ``MXFP4Weights``（kernel 格式，见模块说明）。"""
    w1 = _as_bytes(w1, _FP4_DTYPES, "w1")
    w2 = _as_bytes(w2, _FP4_DTYPES, "w2")
    w1_scale = _as_bytes(w1_scale, _E8M0_DTYPES, "w1_scale")
    w2_scale = _as_bytes(w2_scale, _E8M0_DTYPES, "w2_scale")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(f"w1/w2 must be [E, 2I, H/2] / [E, H, I/2], got {tuple(w1.shape)} / {tuple(w2.shape)}")
    experts, n1, hb = w1.shape
    inter, hidden = n1 // 2, 2 * hb
    if (n1 % 2 or tuple(w2.shape) != (experts, hidden, inter // 2)
            or tuple(w1_scale.shape) != (experts, n1, hidden // GROUP_K)
            or tuple(w2_scale.shape) != (experts, hidden, inter // GROUP_K)):
        raise ValueError(f"inconsistent mxfp4 shapes: w1 {tuple(w1.shape)}, w1_scale {tuple(w1_scale.shape)}, "
                         f"w2 {tuple(w2.shape)}, w2_scale {tuple(w2_scale.shape)}")
    if hidden % 256 or inter % GROUP_K or inter == 0:
        raise ValueError(f"need H % 256 == 0 and I % 32 == 0, got H={hidden}, I={inter}")
    for name, s in (("w1_scale", w1_scale), ("w2_scale", w2_scale)):
        if bool((s == 255).any()):
            raise ValueError(f"{name} contains E8M0 code 255 (NaN)")
    padded = -(-inter // INTER_ALIGN) * INTER_ALIGN
    device = w1.device
    if padded != inter:
        pad1 = torch.zeros((experts, 2 * padded, hb), dtype=torch.uint8, device=device)
        pad1[:, :inter], pad1[:, padded:padded + inter] = w1[:, :inter], w1[:, inter:]
        pads1 = torch.zeros((experts, 2 * padded, hidden // GROUP_K), dtype=torch.uint8, device=device)
        pads1[:, :inter], pads1[:, padded:padded + inter] = w1_scale[:, :inter], w1_scale[:, inter:]
        pad2 = torch.zeros((experts, hidden, padded // 2), dtype=torch.uint8, device=device)
        pad2[:, :, :inter // 2] = w2
        pads2 = torch.zeros((experts, hidden, padded // GROUP_K), dtype=torch.uint8, device=device)
        pads2[:, :, :inter // GROUP_K] = w2_scale
        w1, w1_scale, w2, w2_scale = pad1, pads1, pad2, pads2
    rows1 = torch.zeros(2 * padded, dtype=torch.bool, device=device)
    rows1[:inter] = rows1[padded:padded + inter] = True
    rows2 = torch.ones(hidden, dtype=torch.bool, device=device)
    k1 = _prepare_gemm(w1, w1_scale, rows1, chunk)
    k2 = _prepare_gemm(w2, w2_scale, rows2, chunk)
    return MXFP4Weights(*k1, *k2, inter=inter, inter_padded=padded)
