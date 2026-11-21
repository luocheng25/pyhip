"""A8W4 MoE 的入口：mxfp4 权重（``prepare_weights`` 的结果）+ BF16 激活，按配置表分档。

- decode 档：W4A16，权重在寄存器里精确展开成 BF16，与 BF16 激活做 BF16 MFMA（``decode.py``）。
  T 很小时 batch1（不排序，每条路由一组 CTA，Gate/Up 顺带清零输出）；否则 split-K（先排序）。
- prefill 档：A8W4（M2 实现），现在调用会报 ``NotImplementedError``。

档位与 tile 只看 ``CONFIGS``（与 M0 FP8 ptpc 的配置表同一交界，当前 T<=128 为 decode），代码里不写交界。
运行时依赖 Aiter 的 ``moe_sorting``（只用于 split-K，与 M0 的 decode 相同）；kernel 本身不导入 Aiter。
"""

import torch

from .decode import compile_decode_gemm
from .weights import MXFP4Weights

BLOCK_M = 16
# (最大 T, 配置)：T 取第一条“最大 T >= T”的行，None 表示没有上界。decode 的 tile 沿用 M0 FP8 ptpc 配置表
# （pyhip.testing.moe_a8w4_ptpc.FLY_CONFIGS）的 decode 行：T=1 batch1 gn32，2..16 batch1 gn64，17..128 split-K。
CONFIGS = (
    (1, dict(path="decode", alg="batch1", tile_n_gate=32, tile_n_down=64)),
    (16, dict(path="decode", alg="batch1", tile_n_gate=64, tile_n_down=64)),
    (128, dict(path="decode", alg="splitk", tile_n_gate=128, tile_n_down=128)),
    (None, dict(path="prefill")),
)


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
    ``config`` 默认按 ``CONFIGS`` 选；prefill 档报 ``NotImplementedError``（M2 实现）。
    """
    from ..moe_gemm_2stage.common import _run_compiled

    _check(hidden_states, weights, topk_weight, topk_ids, output)
    tokens, hidden = hidden_states.shape
    config = select_config(tokens) if config is None else dict(config)
    if config["path"] != "decode":
        raise NotImplementedError(f"A8W4 prefill tier (T={tokens}) is not implemented yet (milestone M2)")
    out = torch.empty_like(hidden_states) if output is None else output
    if tokens == 0:
        return out
    experts, topk, inter = weights.experts, topk_ids.shape[1], weights.inter_padded
    stream = torch.cuda.current_stream(hidden_states.device)
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
