"""Qwen3.8-Flash-Next 单卡分片的 mxfp4（A8W4）MoE：权重生成、mxfp4 反量化与 FP32 参考。

tests/ops/moe/a8w4_ptpc 与 benchmarks/moe/bench_moe_a8w4_ptpc.py 共用；形状、输入、容差沿用
``pyhip.testing.moe_a8w4_ptpc``（H=2560，E=512，top-10，SiLU；TP2 I=320、TP4 I=160）。导入本模块不初始化 GPU。

- 权重：自然布局（checkpoint 形式）的随机 mxfp4。nibble 在 16 个 E2M1 编码上均匀取（含 -0）；每行一个基准
  E8M0 code，各 32-K 组再随机减 0..20，使“通道最大 code − 组 code”覆盖查找表平移的 0..11 和会被 flush 的 >= 12。
- 参考：权重为 mxfp4 精确反量化 E2M1 * 2^(code-127)，并施加 kernel 的 flush 规则（比通道最大 code 小 12 以上的
  组为 0）；FP32 计算，Gate/Up 结果先舍入成 BF16（kernel 以 BF16 保存中间结果）。decode 的数据流（M1）激活为 BF16，
  与 M0 的 ``reference_moe(act_quant=False)`` 相同；prefill 的数据流（M2，``act_quant=True``）把两级 GEMM 的输入
  都按行量化成 FP8，与 M0 的 ``reference_moe(act_quant=True)`` 相同。
"""

from pyhip.testing import moe_a8w4_ptpc as M

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FLUSH_DELTA = 12
# 每组相对行基准 code 的减量分布：多数在 0..2，约 1/4 在 3..11，约 5% >= 12（flush）。
_DROP_BINS = ((0.70, 0, 3), (0.25, 3, 12), (0.05, 12, 21))


def _torch():
    import torch

    return torch


def e2m1_table(device):
    torch = _torch()
    mags = torch.tensor(E2M1, dtype=torch.float32, device=device)
    return torch.cat([mags, -mags])


def random_codes(shape, *, gen, device, base=(118, 125)):
    """E8M0 code [..., rows, groups]：每行基准 code 在 ``base`` 内，组减量按 ``_DROP_BINS``。"""
    torch = _torch()
    rows = torch.randint(base[0], base[1], shape[:-1] + (1,), generator=gen, device=device)
    pick = torch.rand(shape, generator=gen, device=device)
    drop = torch.zeros(shape, dtype=torch.int64, device=device)
    edge = 0.0
    for share, low, high in _DROP_BINS:
        sel = (pick >= edge) & (pick < edge + share)
        drop = torch.where(sel, torch.randint(low, high, shape, generator=gen, device=device), drop)
        edge += share
    return (rows - drop).to(torch.uint8)


def make_mxfp4_weights(tp, *, seed, device="cuda", chunk=64, prepare=True):
    """``(raw, prepared)``：raw = (w1 [E, 2I, H/2], w1_scale [E, 2I, H/32], w2 [E, H, I/2], w2_scale [E, H, I/32])，
    uint8 自然布局、未补零；prepared = ``prepare_weights(*raw)``（``prepare=False`` 时为 None）。"""
    torch = _torch()
    inter = M.INTER[tp]
    gen = torch.Generator(device=device).manual_seed(seed)
    shapes = (((M.E, 2 * inter, M.H // 2), (M.E, 2 * inter, M.H // 32)),
              ((M.E, M.H, inter // 2), (M.E, M.H, inter // 32)))
    raw = []
    for w_shape, s_shape in shapes:
        w = torch.empty(w_shape, dtype=torch.uint8, device=device)
        s = torch.empty(s_shape, dtype=torch.uint8, device=device)
        for begin in range(0, M.E, chunk):
            end = min(M.E, begin + chunk)
            w[begin:end] = torch.randint(0, 256, (end - begin, *w_shape[1:]), generator=gen,
                                         device=device, dtype=torch.int32).to(torch.uint8)
            s[begin:end] = random_codes((end - begin, *s_shape[1:]), gen=gen, device=device)
        raw += [w, s]
    if not prepare:
        return tuple(raw), None
    from pyhip.ops.moe.flydsl.a8w4_ptpc import prepare_weights

    return tuple(raw), prepare_weights(*raw)


def dequant_mxfp4(w, codes, *, flush=True):
    """mxfp4 精确反量化：w uint8 [..., N, K/2]、codes uint8 [..., N, K/32] -> fp32 [..., N, K]。

    ``flush``：施加 kernel 的规则，code 比该行最大 code 小 ``FLUSH_DELTA`` 及以上的组为 0。
    """
    torch = _torch()
    nib = torch.stack((w & 0xF, w >> 4), dim=-1).flatten(-2).long()
    values = e2m1_table(w.device)[nib]
    code = codes.to(torch.int32)
    # 2^(code-127) 的 fp32 位型；code 0 是次正规数 2^-127。
    scale = torch.where(code > 0, code << 23, torch.full_like(code, 1 << 22)).view(torch.float32)
    grouped = values.unflatten(-1, (codes.shape[-1], 32)) * scale.unsqueeze(-1)
    if flush:
        drop = (code.amax(dim=-1, keepdim=True) - code) >= FLUSH_DELTA
        grouped = grouped.masked_fill(drop.unsqueeze(-1), 0.0)
    return grouped.flatten(-2)


def reference_moe(hidden, raw, topk_weight, topk_ids, *, act_quant=False):
    """自然布局 mxfp4 权重上的 FP32 参考，返回 ``[T, H]`` FP32。

    ``act_quant=False``：decode 的数据流（BF16 激活，M1）；``True``：prefill 的数据流（M2），两级 GEMM 的输入
    都按行量化成 FP8 再反量化（``moe_a8w4_ptpc.quantize_rows``，同 M0 的 ``reference_moe(act_quant=True)``）。
    """
    torch = _torch()
    w1, s1, w2, s2 = raw
    tokens, topk = topk_ids.shape
    inter = w2.shape[2] * 2
    routes = topk_ids.reshape(-1).long()
    order = torch.argsort(routes, stable=True)
    counts = torch.bincount(routes, minlength=w1.shape[0]).tolist()
    a1 = _dequant_rows(hidden) if act_quant else hidden.float()
    act = torch.empty((tokens * topk, inter), dtype=torch.bfloat16, device=hidden.device)
    down = torch.empty((tokens * topk, w2.shape[1]), dtype=torch.float32, device=hidden.device)
    begin = 0
    for expert, count in enumerate(counts):
        if count:
            rows = order[begin:begin + count]
            gate_up = a1[rows // topk] @ dequant_mxfp4(w1[expert], s1[expert]).t()
            gate, up = gate_up[:, :inter], gate_up[:, inter:]
            act[rows] = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            a2 = _dequant_rows(act[rows]) if act_quant else act[rows].float()
            down[rows] = a2 @ dequant_mxfp4(w2[expert], s2[expert]).t()
            begin += count
    down = down.view(tokens, topk, -1) * topk_weight.float().unsqueeze(-1)
    return down.sum(dim=1)


def _dequant_rows(x):
    """按行量化成 FP8（per-token）再反量化成 fp32。"""
    q, scale = M.quantize_rows(x)
    return q.float() * scale


def decode_prepared(w, delta, factor):
    """按 kernel 的算术把 ``prepare_weights`` 的一个 GEMM（w [E, N/16, K/16, 16, 8]、delta、factor）解回
    fp32 [E, N, K]：fd1 nibble -> 查找表平移 delta 的 FP8 E4M3FNUZ 字节 -> 乘 factor。"""
    torch = _torch()
    experts, nb16, kb16 = w.shape[:3]
    n, k = nb16 * 16, kb16 * 16
    b = w.permute(0, 1, 3, 2, 4).reshape(experts, n, k // 8, 4).to(torch.int32)
    nib = torch.cat((b & 0xF, b >> 4), dim=-1).reshape(experts, n, k)
    d8 = delta.permute(0, 1, 4, 3, 2).reshape(experts, n, k // 32).to(torch.int32)
    table = torch.tensor([0x80, 0x60, 0x68, 0x6C, 0x70, 0x74, 0x78, 0x7C], dtype=torch.int32, device=w.device)
    byte = table[nib & 7] - d8.repeat_interleave(32, dim=-1)
    byte = torch.where((nib & 7) == 0, torch.zeros_like(byte), byte) | ((nib & 8) << 4)
    fp8 = byte.to(torch.uint8).view(torch.float8_e4m3fnuz).float()
    return fp8 * factor.unsqueeze(-1)
