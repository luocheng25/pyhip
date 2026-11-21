"""Qwen3.8-Flash-Next 单卡分片的 per-token FP8（ptpc）MoE：形状、输入、FP32 参考和 FlyDSL 固定配置。

tests/ops/moe/a8w4_ptpc 与 benchmarks/moe/bench_moe_a8w4_ptpc.py 共用；导入本模块不初始化 GPU。

- 形状：H=2560，E=512，top-10，SiLU；TP2 每卡 I=320，TP4 每卡 I=160。FlyDSL 的 Down
  要求 I 是 64 的倍数（tuned_moe 也按 ``i % 64`` 判断能否走本地 kernel），所以 TP4 在加载
  权重时补零到 192：gate、up 各补 32 行，w2 补 32 列。补出的 gate/up 输出为 0，
  SiLU(0)·0=0，per-token amax 也不变，结果与 I=160 精确等价（只差累加顺序）。
- 路径：经 tuned_moe 的 FlyDSL 候选，用 ``make_moe_runner`` 固定配置（不调优、不回退 Aiter）。
  ``fly_decode`` 读 BF16 激活；``fly_prefill`` 把两级 GEMM 的输入都按 token 量化成 FP8。
  ``FLY_CONFIGS`` 由 ``bench_moe_a8w4_ptpc.py --sweep`` 在 MI308X 上从 tuned_moe 的 FlyDSL
  候选里实测选出（规则见其注释）。
- 参考：FP32 计算、权重为反量化的 FP8；``act_quant`` 与所选路径的数据流一致；
  Gate/Up 结果先舍入成 BF16（kernel 以 BF16 保存中间结果）。
- 容差：沿用上游 aiter ``op_tests/test_flydsl_moe_gfx942.py``，以每个 token 参考值的 RMS 为单位。
"""

H, E, TOPK = 2560, 512, 10
INTER = {2: 320, 4: 160}
KERNEL_INTER = {2: 320, 4: 192}
BENCH_TOKENS = (1, 8, 16, 32, 128, 1024, 8192, 65536)
FP8_MAX = 240.0

ELEMENT_ERR_LIMIT = 1.0
TOKEN_ERR_LIMIT = 0.2
LOGITS_DIFF_LIMIT = 1e-3

FLY_IMPLS = ("fly_decode", "fly_prefill")
_BATCH1 = dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16, tile_n_down=64, decode_alg="batch1")
_SPLITK = dict(_impl="fly_decode", tile_m_gate=16, tile_m_down=16, tile_n_gate=128, tile_n_down=128)
_PREFILL_1X4 = dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                    tile_n_down=256, down_path="1x4_64x256", padding=128)
_COMPACT = dict(_impl="fly_prefill", tile_m_gate=64, tile_m_down=64, tile_n_gate=128,
                tile_n_down=128, down_path="8x1_compact", padding=128)
# (tp, 最大 T) -> 配置；T 取第一条“最大 T >= T”的行，超出最后一行时用最后一行。
# 由 `bench_moe_a8w4_ptpc.py --sweep` 在 MI308X 上实测选出，只取 gemm.py 分流到上游 gfx9_fp8 的候选
# （这是 gfx942 ptpc 的生产路径）；相差不到 1%（10 次采样的噪声内）时沿用相邻 T 的配置。
# 替换前（8027faad，pyhip 实现）与替换后（上游实现，BF16 舍入 RTE 或 RTA）各扫一次，结论一致，
# 只有 TP4 T=65536 在替换后改选上游 compact（比 1x4 快 3%~4%）。TP2 T=8192 最快的是 legacy 的
# compact（K=320，上游不支持），比 1x4 快约 1%，不取。
FLY_CONFIGS = {
    (2, 1): dict(_BATCH1, tile_n_gate=32),
    (2, 16): dict(_BATCH1, tile_n_gate=64),
    (2, 128): dict(_SPLITK),
    (2, 1024): dict(_PREFILL_1X4, tile_k_gate=128),
    (2, 65536): dict(_PREFILL_1X4, tile_k_gate=256),
    (4, 1): dict(_BATCH1, tile_n_gate=32),
    (4, 16): dict(_BATCH1, tile_n_gate=64),
    (4, 128): dict(_SPLITK),
    (4, 8192): dict(_PREFILL_1X4, tile_k_gate=256),
    (4, 65536): dict(_COMPACT, tile_k_gate=256),
}


def _torch():
    import torch

    return torch


def fp8_dtype():
    return _torch().float8_e4m3fnuz


def case_name(tp, tokens):
    return f"tp{tp}_t{tokens}"


def fly_config(tp, tokens):
    """``tokens`` 个 token 在 TP ``tp`` 上使用的 FlyDSL 配置（tuned_moe 的 Config 参数）。"""
    rows = sorted((limit, config) for (row_tp, limit), config in FLY_CONFIGS.items() if row_tp == tp)
    if not rows:
        raise KeyError(f"no FlyDSL config for TP{tp}")
    for limit, config in rows:
        if tokens <= limit:
            return dict(config)
    return dict(rows[-1][1])


def act_quant(config):
    """prefill 路径按 token 量化两级输入；decode 路径读 BF16 激活。"""
    return config["_impl"] == "fly_prefill"


def describe(config):
    """配置的简短描述，用于日志和报告。"""
    if config["_impl"] == "fly_decode":
        if config.get("decode_alg") == "batch1":
            return f"decode/batch1 gn{config['tile_n_gate']} dn{config['tile_n_down']}"
        return (f"decode/splitk m{config['tile_m_down']} "
                f"gn{config['tile_n_gate']} dn{config['tile_n_down']}")
    path = config.get("down_path", "default")
    return (f"prefill gm{config['tile_m_gate']} gn{config['tile_n_gate']} gk{config['tile_k_gate']} "
            f"down={path} dm{config['tile_m_down']} dn{config['tile_n_down']} "
            f"pad{config.get('padding') or 0}")


def quantize_rows(x):
    """沿最后一维按行量化成 FP8，返回 ``(q, scale[..., 1])``。

    与 Aiter per-token HIP kernel 的算术一致：``scale = amax * fp32(1/240)``，
    ``q = sat(x / scale)`` 就近舍入（kernel 乘近似倒数，极少数值会差一个 FP8 步长）；
    全零行的 scale 为 0。
    """
    torch = _torch()
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = (xf * inv).clamp(-FP8_MAX, FP8_MAX).to(fp8_dtype())
    return q, scale


def make_weights(tp, *, seed, keep_raw=True, device="cuda", chunk=64):
    """一组专家权重，返回 ``(raw, kernel)``。

    ``raw = (w1 [E, 2I, H], w2 [E, H, I], w1_scale [E, 2I], w2_scale [E, H])`` 是未补零、
    未 shuffle 的自然布局，供参考实现使用；``keep_raw=False`` 时为 None。
    ``kernel`` 是同样四项的 kernel 布局：I 补零到 ``KERNEL_INTER[tp]``、权重经 Aiter
    ``shuffle_weight``（带 ``is_shuffled``）、scale 形状为 ``[E, rows, 1]``。
    权重取标准正态的 BF16 值，按输出通道量化成 FP8。
    """
    torch = _torch()
    from aiter.ops.shuffle import shuffle_weight

    inter, padded = INTER[tp], KERNEL_INTER[tp]
    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.empty((E, 2 * inter, H), dtype=fp8_dtype(), device=device)
    w2 = torch.empty((E, H, inter), dtype=fp8_dtype(), device=device)
    s1 = torch.empty((E, 2 * inter), dtype=torch.float32, device=device)
    s2 = torch.empty((E, H), dtype=torch.float32, device=device)
    for begin in range(0, E, chunk):
        end = min(E, begin + chunk)
        for w, s, shape in ((w1, s1, (2 * inter, H)), (w2, s2, (H, inter))):
            value = torch.randn((end - begin, *shape), generator=gen, device=device)
            q, scale = quantize_rows(value.to(torch.bfloat16))
            w[begin:end] = q
            s[begin:end] = scale.squeeze(-1)
    if padded == inter:
        k1, k2, ks1 = w1, w2, s1
    else:
        k1 = torch.zeros((E, 2 * padded, H), dtype=w1.dtype, device=device)
        k1[:, :inter] = w1[:, :inter]
        k1[:, padded:padded + inter] = w1[:, inter:]
        k2 = torch.zeros((E, H, padded), dtype=w2.dtype, device=device)
        k2[:, :, :inter] = w2
        ks1 = torch.ones((E, 2 * padded), dtype=torch.float32, device=device)
        ks1[:, :inter] = s1[:, :inter]
        ks1[:, padded:padded + inter] = s1[:, inter:]
    kernel = (shuffle_weight(k1), shuffle_weight(k2),
              ks1.reshape(E, -1, 1).contiguous(), s2.reshape(E, -1, 1).contiguous())
    return ((w1, w2, s1, s2) if keep_raw else None), kernel


def make_inputs(tokens, *, seed, device="cuda"):
    """``(hidden bf16 [T, H], topk_weight fp32 [T, K], topk_ids int32 [T, K])``。

    每个 token 均匀随机选 ``TOPK`` 个不同专家；权重随机且每个 token 和为 1。
    """
    torch = _torch()
    gen = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn((tokens, H), generator=gen, device=device).to(torch.bfloat16)
    keys = torch.rand((tokens, E), generator=gen, device=device)
    ids = keys.topk(TOPK, dim=1).indices.to(torch.int32).contiguous()
    logits = torch.randn((tokens, TOPK), generator=gen, device=device)
    return hidden, logits.softmax(dim=-1).contiguous(), ids


def moe_call(kernel, hidden, topk_weight, topk_ids, output):
    """tuned_moe 公共接口（及 ``make_moe_runner``）的参数。"""
    import aiter

    w1, w2, w1_scale, w2_scale = kernel
    return dict(hidden_states=hidden, w1=w1, w2=w2, topk_weight=topk_weight, topk_ids=topk_ids,
                w1_scale=w1_scale, w2_scale=w2_scale, quant_type=aiter.QuantType.per_Token,
                activation=aiter.ActivationType.Silu, output=output)


def fly_runner(config):
    """固定配置的 tuned_moe FlyDSL 调用：``runner(**moe_call(...))``。"""
    from pyhip.testing.moe import make_moe_runner

    if config["_impl"] not in FLY_IMPLS:
        raise ValueError(f"not a FlyDSL config: {config}")
    return make_moe_runner(config)


def fly_candidates(call):
    """tuned_moe 对这组输入生成的全部 FlyDSL 候选（Config 参数字典）。"""
    import inspect

    from pyhip.ops.moe import tuned_moe as tm

    bound = inspect.signature(tm.fused_moe).bind(**call)
    bound.apply_defaults()
    configs = tm._configs(**tm._make_tune_args(bound.arguments))
    return [c.all_kwargs() for c in configs if c.all_kwargs()["_impl"] in FLY_IMPLS]


def reference_moe(hidden, raw, topk_weight, topk_ids, *, act_quant):
    """自然布局权重上的 FP32 参考，返回 ``[T, H]`` FP32。"""
    torch = _torch()
    w1, w2, w1_scale, w2_scale = raw
    tokens = hidden.shape[0]
    topk = topk_ids.shape[1]
    inter = w2.shape[2]
    if act_quant:
        q, s = quantize_rows(hidden)
        a1 = q.float() * s
    else:
        a1 = hidden.float()
    routes = topk_ids.reshape(-1).long()
    order = torch.argsort(routes, stable=True)
    counts = torch.bincount(routes, minlength=w1.shape[0]).tolist()
    act = torch.empty((tokens * topk, inter), dtype=torch.bfloat16, device=hidden.device)
    begin = 0
    for expert, count in enumerate(counts):
        if count:
            rows = order[begin:begin + count]
            w = w1[expert].float() * w1_scale[expert].reshape(-1, 1)
            gate_up = a1[rows // topk] @ w.t()
            gate, up = gate_up[:, :inter], gate_up[:, inter:]
            act[rows] = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            begin += count
    if act_quant:
        q, s = quantize_rows(act)
        a2 = q.float() * s
    else:
        a2 = act.float()
    down = torch.empty((tokens * topk, w2.shape[1]), dtype=torch.float32, device=hidden.device)
    begin = 0
    for expert, count in enumerate(counts):
        if count:
            rows = order[begin:begin + count]
            w = w2[expert].float() * w2_scale[expert].reshape(-1, 1)
            down[rows] = a2[rows] @ w.t()
            begin += count
    down = down.view(tokens, topk, -1) * topk_weight.float().unsqueeze(-1)
    return down.sum(dim=1)


def output_errors(reference, output):
    """上游的三项精度指标：``(偏差超过 ELEMENT_ERR_LIMIT 个 token RMS 的元素比例,
    logits_diff, 最差 token 的相对 L2 误差)``。"""
    rms = reference.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    beyond = ((output.float() - reference.float()).abs() / rms) > ELEMENT_ERR_LIMIT
    x, y = reference.double(), output.double()
    logits_diff = float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())
    token_err = float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max())
    return float(beyond.float().mean()), logits_diff, token_err


def check_output(reference, output, msg=""):
    """按上游容差断言，返回 ``output_errors``。"""
    torch = _torch()
    assert bool(torch.isfinite(output).all()), f"{msg}: non-finite output"
    beyond, logits_diff, token_err = output_errors(reference, output)
    assert beyond == 0, f"{msg}: {beyond:.3%} of elements beyond {ELEMENT_ERR_LIMIT} RMS"
    assert logits_diff < LOGITS_DIFF_LIMIT, f"{msg}: logits_diff={logits_diff}"
    assert token_err < TOKEN_ERR_LIMIT, f"{msg}: worst token error {token_err}"
    return beyond, logits_diff, token_err


def touched_experts(tokens):
    """均匀路由下一次调用触达的专家数期望。"""
    return E * (1 - (1 - TOPK / E) ** tokens)
