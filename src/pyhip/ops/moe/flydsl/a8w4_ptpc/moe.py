"""A8W4 MoE 的入口：mxfp4 权重（``prepare_weights`` 的结果）+ BF16 激活，按配置表分档。

- decode 档：W4A16，权重在寄存器里精确展开成 BF16，与 BF16 激活做 BF16 MFMA（``decode.py``）。
  T 很小时 batch1（不排序，每条路由一组 CTA，Gate/Up 顺带清零输出）；否则 split-K（先排序）。
- prefill 档：A8W4，数据流与 M0 FP8 ptpc 的 prefill 相同（``prefill.py``）：按 M64 排序，输入按 token 量化成
  FP8，Gate/Up（1x4）写 BF16 中间结果，中间结果再按 (token, k) 量化成 FP8，Down（1x4，M64xN256）每个排序行
  写一行，最后按 token 把 top-k 行相加；权重在寄存器里精确解码成 FP8，做 FP8 MFMA。

档位与 tile 只看 ``CONFIGS``（与 M0 FP8 ptpc 的配置表同一交界，当前 T<=128 为 decode），代码里不写交界。
运行时依赖 Aiter 的 ``moe_sorting``（split-K 与 prefill）和 per-token 量化（prefill，与 M0 相同）；
kernel 本身不导入 Aiter。
"""

import torch

from .decode import compile_decode_gemm
from .prefill import compile_prefill_gemm
from .weights import MXFP4Weights

BLOCK_M = 16
# (最大 T, 配置)：T 取第一条“最大 T >= T”的行，None 表示没有上界。decode 的 tile 沿用 M0 FP8 ptpc 配置表
# （pyhip.testing.moe_a8w4_ptpc.FLY_CONFIGS）的 decode 行：T=1 batch1 gn32，2..16 batch1 gn64，17..128 split-K。
# prefill 的 Gate/Up 与 FP8 表相同（M64、gn128），Down 一律 1x4（M64xN256，行距补 128 B）；FP8 表在 TP4
# T>8192 用的 compact Down 留到性能阶段。Gate/Up 的 BK 取 128：MI308X 上按 bench 协议实测，TP2/TP4 的
# T=1024/8192/65536 都比 256 快 3%~8%（FP8 表在 T>1024 或 TP4 取 256）。
CONFIGS = (
    (1, dict(path="decode", alg="batch1", tile_n_gate=32, tile_n_down=64)),
    (16, dict(path="decode", alg="batch1", tile_n_gate=64, tile_n_down=64)),
    (128, dict(path="decode", alg="splitk", tile_n_gate=128, tile_n_down=128)),
    (None, dict(path="prefill", tile_m=64, tile_n_gate=128, tile_k_gate=128, tile_n_down=256, padding=128)),
)
FP8 = torch.float8_e4m3fnuz


def select_config(tokens, configs=CONFIGS):
    """``tokens`` 个 token 使用的配置（字典副本）。"""
    for limit, config in configs:
        if limit is None or tokens <= limit:
            return dict(config)
    return dict(configs[-1][1])


def decode_max_tokens(configs=CONFIGS):
    """decode 档的最大 T：配置表里 decode 行的最大上界。"""
    return max(limit for limit, config in configs if config["path"] == "decode")


def _ptr(t):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    elem = {torch.bfloat16: fx.BFloat16, torch.float32: fx.Float32, torch.int32: fx.Int32}.get(t.dtype, fx.Uint8)
    return flyc.from_c_void_p(elem, t.data_ptr())


def _check(hidden_states, weights, topk_weight, topk_ids, output):
    if not isinstance(weights, MXFP4Weights):
        raise TypeError("weights must come from a8w4_ptpc.prepare_weights")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2 or not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a BF16 [T, H] CUDA tensor")
    tokens, hidden = hidden_states.shape
    if hidden != weights.hidden or weights.w1.device != hidden_states.device:
        raise ValueError(f"hidden size / device mismatch: H={hidden}, weights H={weights.hidden}")
    if (topk_ids.dtype != torch.int32 or topk_weight.dtype != torch.float32 or topk_ids.ndim != 2
            or topk_ids.shape[0] != tokens or topk_weight.shape != topk_ids.shape):
        raise ValueError("topk_ids / topk_weight must be int32 / fp32 [T, topk]")
    tensors = (hidden_states, topk_weight, topk_ids) + (() if output is None else (output,))
    if not all(t.is_contiguous() and t.device == hidden_states.device for t in tensors):
        raise ValueError("inputs must be contiguous and on one device")
    if output is not None and (output.shape != hidden_states.shape or output.dtype != torch.bfloat16):
        raise ValueError("output must be a BF16 tensor shaped like hidden_states")


def fused_moe(hidden_states, weights, topk_weight, topk_ids, *, output=None, config=None):
    """MoE 前向：``sum_k topk_weight[t, k] * Down(SiLU(Gate x) * Up x)``，写入并返回 ``output``（BF16 [T, H]）。

    ``weights``：``prepare_weights`` 的结果；``topk_ids`` int32 [T, topk]（0 <= id < E），``topk_weight`` fp32。
    ``config`` 默认按 ``CONFIGS`` 选（decode 或 prefill 档）。
    """
    from ..moe_gemm_2stage.common import _run_compiled

    _check(hidden_states, weights, topk_weight, topk_ids, output)
    tokens, hidden = hidden_states.shape
    config = select_config(tokens) if config is None else dict(config)
    if config["path"] not in ("decode", "prefill"):
        raise ValueError(f"unknown A8W4 path: {config}")
    out = torch.empty_like(hidden_states) if output is None else output
    if tokens == 0:
        return out
    stream = torch.cuda.current_stream(hidden_states.device)
    if config["path"] == "prefill":
        return _fused_moe_prefill(hidden_states, weights, topk_weight, topk_ids, out, config, stream)
    experts, topk, inter = weights.experts, topk_ids.shape[1], weights.inter_padded
    mid = torch.empty((tokens, topk, inter), dtype=torch.bfloat16, device=hidden_states.device)
    common = dict(TOPK=topk, BLOCK_TILE_SIZE_M=BLOCK_M, alg=config["alg"], E=experts)
    gateup = compile_decode_gemm(stage="gateup", N=2 * inter, K=hidden,
                                 BLOCK_TILE_SIZE_N=config["tile_n_gate"], **common)
    down = compile_decode_gemm(stage="down", N=hidden, K=inter,
                               BLOCK_TILE_SIZE_N=config["tile_n_down"], **common)
    w = weights
    if config["alg"] == "batch1":
        # Gate/Up 顺带清零 out，同一 stream 上的 Down 再原子加。
        _run_compiled(gateup, *map(_ptr, (hidden_states, w.w1, mid, topk_ids, out, w.w1_factor, w.w1_delta)),
                      tokens, stream)
        _run_compiled(down, *map(_ptr, (mid, w.w2, out, topk_ids, topk_weight, w.w2_factor, w.w2_delta)),
                      tokens, stream)
        return out
    from ..._aiter_compat import moe_sorting

    # moe_sorting 同时清零 out。
    routing = moe_sorting(topk_ids, topk_weight, experts, hidden, hidden_states.dtype, BLOCK_M, output=out)[:4]
    tasks = routing[2].numel()
    _run_compiled(gateup, *map(_ptr, (hidden_states, w.w1, mid, *routing, w.w1_factor, w.w1_delta)),
                  tokens, tasks, stream)
    _run_compiled(down, *map(_ptr, (mid, w.w2, out, *routing, w.w2_factor, w.w2_delta)), tokens, tasks, stream)
    return out


def _fused_moe_prefill(hidden_states, w, topk_weight, topk_ids, out, config, stream):
    """prefill 档：数据流同 tuned_moe 的 FlyDSL FP8 ptpc prefill（M0 的 ``_run_fly``），权重换成 mxfp4。"""
    import aiter

    from ..._aiter_compat import moe_sorting
    from ..moe_gemm_2stage.common import _run_compiled
    from ..moe_gemm_2stage.moe_reduce import invert_sorted_ids, sorted_sum

    tokens, hidden = hidden_states.shape
    experts, topk, inter = w.experts, topk_ids.shape[1], w.inter_padded
    block_m, padding, device = config["tile_m"], config["padding"], hidden_states.device
    # Down 按排序行写各自的行、不做 atomic，out 由 sorted_sum 整体写出：排序不必分配、清零 moe_buf。
    routing = moe_sorting(topk_ids, topk_weight, experts, hidden, hidden_states.dtype, block_m,
                          accumulate=False)[:4]
    tasks = routing[2].numel()
    quantize = aiter.get_hip_quant(aiter.QuantType.per_Token)
    common = dict(TOPK=topk, BLOCK_TILE_SIZE_M=block_m, E=experts)
    gateup = compile_prefill_gemm(stage="gateup", N=2 * inter, K=hidden, BLOCK_TILE_SIZE_N=config["tile_n_gate"],
                                  tile_k=config["tile_k_gate"], **common)
    down = compile_prefill_gemm(stage="down", N=hidden, K=inter, BLOCK_TILE_SIZE_N=config["tile_n_down"],
                                padding=padding, **common)
    x, x_scale = quantize(hidden_states, quant_dtype=FP8)
    mid = torch.empty((tokens, topk, inter), dtype=torch.bfloat16, device=device)
    _run_compiled(gateup, *map(_ptr, (x, w.w1, mid, *routing, w.w1_factor, w.w1_pdelta, x_scale)),
                  tokens, tasks, stream)
    mid_q, mid_scale = quantize(mid.view(tokens * topk, inter), quant_dtype=FP8)
    routes = torch.empty((tasks * block_m, hidden + padding // 2), dtype=torch.bfloat16, device=device)
    _run_compiled(down, *map(_ptr, (mid_q, w.w2, routes, *routing, w.w2_factor, w.w2_pdelta, mid_scale)),
                  tokens, tasks, stream)
    # 每个 (token, k) 的排序行号，再按 token 把 top-k 行相加写入 out。
    loc = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    invert_sorted_ids(topk)(routing[0], loc, routing[3], routing[0].numel(), tokens)
    sorted_sum(topk, hidden, padding)(loc, routes, out, tokens)
    return out
