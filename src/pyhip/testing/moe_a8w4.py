"""A8W4 MoE 的输入生成、独立 FP32 参考与验收判据（任务冻结后受保护）。

测试与 bench 共用；导入本模块不初始化 GPU。参考只用 torch，不调用被测 kernel：
MXFP4 精确反量化；``activation="fp8"`` 时输入和中间激活按行量化成 FP8 E4M3FNUZ
（scale = amax / 240，与 A8W4 路径的量化点一致），``"bf16"`` 时激活不量化（给 A16W4 路径）。
"""

FP8_MAX = 240.0
CALC_DIFF_MAX = 1e-3
MAX_ERR_RATIO = 0.045  # 与 tingqli refactor/a8w4 同级（它对 FP8 激活参考的 err 约 4.2%）
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def make_weights(experts, hidden, inter, *, seed=0, fp8_inter=None, device="cuda"):
    """随机 BF16 MoE 权重（逐 expert 生成）量化成标准 MXFP4；可选同源 FP8 per-channel 版本。

    返回 dict：``w1`` float4_e2m1fn_x2 [E, 2I, H/2]（行 [0, I) gate、[I, 2I) up）、
    ``w1_scale`` float8_e8m0fnu [E, 2I, H/32]、``w2`` [E, H, I/2]、``w2_scale`` [E, H, I/32]。
    给出 ``fp8_inter``（>= inter）时另含把同一份 BF16 权重的 I 补零到 ``fp8_inter`` 后按输出通道
    量化的 FP8 E4M3FNUZ 权重（未 shuffle）：``w1_fp8`` [E, 2*fp8_inter, H]、``w1_fp8_scale``
    [E, 2*fp8_inter, 1]、``w2_fp8`` [E, H, fp8_inter]、``w2_fp8_scale`` [E, H, 1]（float32）。
    标准差：w1 为 1/sqrt(H)，w2 为 1/sqrt(I)。
    """
    import aiter
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    mx = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    h, i = hidden, inter
    out = dict(
        w1=torch.empty((experts, 2 * i, h // 2), dtype=torch.uint8, device=device),
        w1_scale=torch.empty((experts, 2 * i, h // 32), dtype=torch.uint8, device=device),
        w2=torch.empty((experts, h, i // 2), dtype=torch.uint8, device=device),
        w2_scale=torch.empty((experts, h, i // 32), dtype=torch.uint8, device=device),
    )
    fp8 = torch.float8_e4m3fnuz
    if fp8_inter is not None:
        assert fp8_inter >= inter
        p = fp8_inter
        pt = aiter.get_torch_quant(aiter.QuantType.per_Token)
        out.update(
            w1_fp8=torch.empty((experts, 2 * p, h), dtype=fp8, device=device),
            w1_fp8_scale=torch.empty((experts, 2 * p, 1), dtype=torch.float32, device=device),
            w2_fp8=torch.empty((experts, h, p), dtype=fp8, device=device),
            w2_fp8_scale=torch.empty((experts, h, 1), dtype=torch.float32, device=device),
        )
    for e in range(experts):
        v1 = torch.randn((2 * i, h), generator=gen, device=device) * h ** -0.5
        v2 = torch.randn((h, i), generator=gen, device=device) * i ** -0.5
        v1, v2 = v1.to(torch.bfloat16), v2.to(torch.bfloat16)
        for name, value in (("w1", v1), ("w2", v2)):
            q, s = mx(value, quant_dtype=torch.float4_e2m1fn_x2)
            out[name][e].copy_(q.view(torch.uint8))
            out[f"{name}_scale"][e].copy_(s.view(torch.uint8).reshape(out[f"{name}_scale"][e].shape))
        if fp8_inter is not None:
            z = v1.new_zeros((p - i, h))
            v1p = torch.cat([v1[:i], z, v1[i:], z])
            v2p = torch.cat([v2, v2.new_zeros((h, p - i))], dim=1)
            for name, value in (("w1_fp8", v1p), ("w2_fp8", v2p)):
                q, s = pt(value, quant_dtype=fp8)
                out[name][e].copy_(q)
                out[f"{name}_scale"][e].copy_(s.view(-1, 1))
    for name in ("w1", "w2"):
        out[name] = out[name].view(torch.float4_e2m1fn_x2)
        out[f"{name}_scale"] = out[f"{name}_scale"].view(torch.float8_e8m0fnu)
    return out


def make_inputs(tokens, hidden, experts, topk, *, seed=0, shared=True, device="cuda"):
    """返回 ``(hidden_states, topk_weight, topk_ids)``。

    hidden_states BF16 [M, H] ~ N(0, 1)。``shared``（Qwen3.8）：最后一个 expert（E-1）是融合的 shared expert，
    每个 token 的最后一个槽位必选；其余 topk-1 个在 [0, E-1) 中均匀随机且不重复。topk_weight float32：
    routed 为 topk-1 个随机 logit 的 softmax，shared 为 sigmoid(randn)。否则（MiMo）topk 个 expert 在 [0, E)
    中均匀随机且不重复，权重为 topk 个随机 logit 的 softmax。
    """
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=gen, device=device).to(torch.bfloat16)
    if not shared:
        ids = torch.rand((tokens, experts), generator=gen, device=device).topk(topk, dim=-1).indices
        logits = torch.randn((tokens, topk), generator=gen, device=device)
        return x, torch.softmax(logits, dim=-1).float().contiguous(), ids.to(torch.int32).contiguous()
    routed = torch.rand((tokens, experts - 1), generator=gen, device=device).topk(topk - 1, dim=-1).indices
    shared = torch.full((tokens, 1), experts - 1, dtype=routed.dtype, device=device)
    ids = torch.cat([routed, shared], dim=1).to(torch.int32).contiguous()
    logits = torch.randn((tokens, topk - 1), generator=gen, device=device)
    shared_w = torch.sigmoid(torch.randn((tokens, 1), generator=gen, device=device))
    weights = torch.cat([torch.softmax(logits, dim=-1), shared_w], dim=1).float().contiguous()
    return x, weights, ids


def dequant_mxfp4(q, scale):
    """标准 MXFP4 [..., N, K/2] + E8M0 [..., N, K/32] -> 精确的 float32 [..., N, K]。"""
    import torch

    b = q.view(torch.uint8)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).long()
    value = torch.tensor(_E2M1, dtype=torch.float32, device=b.device)[nib]
    code = scale.view(torch.uint8).to(torch.float64)
    factor = torch.exp2(code - 127).float()
    return (value.unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)).flatten(-2)


def quant_rows_fp8(t):
    """按行（最后一维）量化成 FP8 E4M3FNUZ 再反量化。

    scale = amax / 240，q = RNE(x / scale)，两次都是 IEEE f32 除法（全零行 scale 取 1）。
    注意 torch 的 ``tensor / python 标量`` 会改成乘倒数（可差 1 ulp），这里除以同形 tensor。
    """
    import torch

    t = t.float()
    amax = t.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (t / scale).to(torch.float8_e4m3fnuz).float() * scale


def reference(x, w1, w1_scale, w2, w2_scale, topk_weight, topk_ids, *, activation="fp8"):
    """独立 FP32 MoE 参考，返回 float32 [M, H]。

    权重为标准 MXFP4（``make_weights`` 的格式），SiLU(gate) * up，routing 权重在 Down 后乘、
    按 token 累加。``activation``：``"fp8"`` 量化输入与中间激活（每个 token、每个 token×槽位
    一行一个 scale）；``"bf16"`` 不量化激活。
    """
    import torch

    if activation not in ("fp8", "bf16"):
        raise ValueError(f"activation must be 'fp8' or 'bf16', got {activation!r}")
    fp8 = activation == "fp8"
    M, H = x.shape
    E, n1 = w1.shape[0], w1.shape[1]
    I, topk = n1 // 2, topk_ids.shape[1]
    xa = quant_rows_fp8(x) if fp8 else x.float()
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E).tolist()
    tokens, slots = order // topk, order % topk
    groups, start = [], 0
    for e, n in enumerate(counts):
        if n:
            groups.append((e, tokens[start:start + n], slots[start:start + n]))
        start += n
    mid = torch.zeros((M, topk, I), dtype=torch.float32, device=x.device)
    for e, t, s in groups:
        h = xa[t] @ dequant_mxfp4(w1[e], w1_scale[e]).T
        mid[t, s] = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    if fp8:
        mid = quant_rows_fp8(mid)
    out = torch.zeros((M, H), dtype=torch.float32, device=x.device)
    for e, t, s in groups:
        y = mid[t, s] @ dequant_mxfp4(w2[e], w2_scale[e]).T
        out.index_add_(0, t, y * topk_weight[t, s].float().unsqueeze(-1))
    return out


def path_activation(path):
    """由算子记录的 kernel 序列（``fused_moe_a8w4.last_path``）判断该走哪种参考。

    算子的 kernel 以 ``"a8w4_"``（FP8 激活）或 ``"a16w4_"``（BF16 激活）开头；至少要有一个，
    且只能是一种，否则断言失败（防止静默回退到别的实现）。返回 ``"fp8"`` 或 ``"bf16"``。
    """
    modes = {"fp8" if name.startswith("a8w4_") else "bf16"
             for name in path if name.startswith(("a8w4_", "a16w4_"))}
    assert len(modes) == 1, f"expected A8W4 or A16W4 kernels of one kind, got path {list(path)}"
    return modes.pop()


def calc_diff(x, y):
    """1 - 2·Σxy / Σ(x² + y²)（双精度）；两边全零时为 0。"""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def check(out, ref):
    """验收：无 NaN/Inf，calc_diff <= 1e-3，且 max|out - ref| <= 4.5% · max|ref|。返回明细 dict。"""
    import torch

    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return dict(ok=False, reason="output contains NaN/Inf")
    diff = calc_diff(o, r)
    err = (o - r).abs().max().item()
    top = r.abs().max().item()
    ratio = err / top if top else (0.0 if err == 0 else float("inf"))
    ok = diff <= CALC_DIFF_MAX and ratio <= MAX_ERR_RATIO
    return dict(ok=ok, calc_diff=diff, max_err=err, max_ref=top, err_ratio=ratio)


def assert_close(out, ref):
    result = check(out, ref)
    assert result["ok"], f"A8W4 output out of tolerance: {result}"
    return result
