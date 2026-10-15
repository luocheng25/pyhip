"""Torch goldens for one TP rank of a Qwen3.8-Flash-Next decode layer.

The goldens follow SGLang's ROCm decode path for ``qwen4_exp``: BF16 tensors
between operators, FP32 math inside an operator, per-token dynamic FP8
activations for the FP8 linear projections (the MoE experts keep BF16
activations), and the BF16 rounding points inside the fused kernels (HC mix,
RoPE, conv taps, GDN beta, sparse-attention probabilities).
Where AITER truncates to BF16 (its all-reduce and some tuned FP8 GEMMs), the
goldens still round to nearest.
Partial outputs reduce through a caller-supplied ``allreduce`` so a multi-rank
run checks every rank against its own shard; ``golden_layer_tp`` runs all ranks
in one process and sums BF16 partials in rank order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from pyhip.ops.monokernel.qwen38.config import LayerKind, Qwen38Config, Qwen38Shard

# gfx942 activations are E4M3FNUZ (max 240); gfx950 uses OCP E4M3FN (max 448).
ACTIVATION_FP8 = torch.float8_e4m3fnuz
LOG2E = 1.4426950408889634
L2NORM_EPS = 1.0e-6


def bf(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and widen back to FP32."""

    return x.to(torch.bfloat16).float()


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """RMSNorm with a ``1 + weight`` gain, optionally over contiguous groups."""

    xf = x.float()
    shape = xf.shape
    if group is not None:
        xf = xf.reshape(*shape[:-1], shape[-1] // group, group)
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return normed.reshape(shape) * (1.0 + weight.float())


def quantize_rows(x: torch.Tensor, fp8_dtype: torch.dtype = ACTIVATION_FP8):
    """Dynamic per-row FP8 quantization: (FP8 values widened to FP32, FP32 row scales).

    Like AITER's per-token quant, the scale is ``amax * float32(1 / fp8_max)`` and
    values are ``x * (1 / scale)``; a division instead flips some FP8 codes.
    """

    fp8_max = torch.finfo(fp8_dtype).max
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    scale = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=xf.device)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    return (xf * torch.reciprocal(scale)).clamp(-fp8_max, fp8_max).to(fp8_dtype).float(), scale


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> torch.Tensor:
    """W8A8 projection with per-token activation and per-row weight scales; BF16 output."""

    values, scale = quantize_rows(x, fp8_dtype)
    return bf((values @ weight.float().T) * scale * weight_scale.float())


def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return bf(x.float() @ weight.float().T)


def rope_cos_sin(positions: torch.Tensor, config: Qwen38Config, table_dtype: torch.dtype = torch.bfloat16):
    """NeoX RoPE factors of the rotary prefix; SGLang's ROCm cos/sin table is BF16."""

    exponent = torch.arange(0, config.rotary_dim, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / config.rope_theta ** (exponent / config.rotary_dim)
    angles = positions.float()[:, None] * inv_freq[None]
    return angles.cos().to(table_dtype).float(), angles.sin().to(table_dtype).float()


def apply_neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Rotate channel pairs (i, i + rotary_dim / 2) of ``x`` [rows, heads, dim].

    SGLang's ROCm MRoPE kernel runs in the BF16 table dtype, so every product
    and every sum rounds to BF16.
    """

    half = rotary_dim // 2
    x = x.float()
    first, second = x[..., :half], x[..., half:rotary_dim]
    cos, sin = cos[:, None, :], sin[:, None, :]
    rotated_first = bf(bf(first * cos) - bf(second * sin))
    rotated_second = bf(bf(second * cos) + bf(first * sin))
    return torch.cat((rotated_first, rotated_second, x[..., rotary_dim:]), dim=-1)


# --------------------------------------------------------------------- HC


def golden_hc_norm(h: torch.Tensor, weight: torch.Tensor, config: Qwen38Config) -> torch.Tensor:
    return bf(gemma_rmsnorm(h, weight, config.eps, group=config.hidden))


def golden_hc_lowrank(hn: torch.Tensor, down: torch.Tensor, config: Qwen38Config) -> torch.Tensor:
    """FP32 SiLU low-rank activations of the HC mix, before their BF16 rounding."""

    return F.silu((hn.float() @ down.float().T) / config.hc_count)


def hc_mix_from_lowrank(hn: torch.Tensor, lowrank: torch.Tensor, up: torch.Tensor, config: Qwen38Config) -> torch.Tensor:
    rows, hc, hidden = hn.shape[0], config.hc_count, config.hidden
    gate = torch.sigmoid(lowrank.float() @ up.float().T)
    mixed = (gate.view(rows, hc, hidden) * hn.float().view(rows, hc, hidden)).sum(1) / hc
    return bf(mixed)


def golden_hc_mix(hn: torch.Tensor, down: torch.Tensor, up: torch.Tensor, config: Qwen38Config) -> torch.Tensor:
    """Low-rank gated branch mean of the normalized hyper state ``hn`` [rows, hc * hidden]."""

    return hc_mix_from_lowrank(hn, bf(golden_hc_lowrank(hn, down, config)), up, config)


def resolve_hc_mix_ties(
    hn: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    config: Qwen38Config,
    target: torch.Tensor,
    window_ulps: float = 32.0,
    max_flips: int = 2,
) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    """``golden_hc_mix`` with near-tie low-rank roundings flipped where that reproduces ``target``.

    A low-rank activation within ``window_ulps`` FP32 ulps of a BF16 tie rounds by FP32 summation
    order, and one flip moves every output of its row. Per mismatching row, flips (nearest tie
    first, at most ``max_flips`` kept) stay when they reduce the mismatches with ``target``.
    Returns the output and the kept flips as ``(row, index, distance to the tie in FP32 ulps)``.
    """

    act = golden_hc_lowrank(hn, down, config)
    lowrank = act.to(torch.bfloat16)
    out = hc_mix_from_lowrank(hn, lowrank, up, config)
    target = target.to(out.device).float().reshape(out.shape)
    # Adding one to the int16 bits of a BF16 value grows its magnitude for either sign.
    step = torch.where(act.abs() > lowrank.float().abs(), 1, -1).to(torch.int16)
    other = lowrank.view(torch.int16) + step
    tie = (lowrank.float() + other.view(torch.bfloat16).float()) / 2
    ulps = ((act - tie).abs() / (act.abs() * 2.0**-23)).nan_to_num(nan=torch.inf)
    flips = []
    for row in (out != target).any(-1).nonzero().flatten().tolist():
        bits = lowrank[row].view(torch.int16).clone()
        missing = int((out[row] != target[row]).sum())
        kept = 0
        for index in ulps[row].argsort().tolist():
            distance = float(ulps[row, index])
            if distance > window_ulps or kept == max_flips or missing == 0:
                break
            trial = bits.clone()
            trial[index] = other[row, index]
            mixed = hc_mix_from_lowrank(hn[row : row + 1], trial.view(torch.bfloat16)[None], up, config)[0]
            trial_missing = int((mixed != target[row]).sum())
            if trial_missing < missing:
                bits, missing, out[row] = trial, trial_missing, mixed
                kept += 1
                flips.append((row, index, distance))
    return out, flips


def golden_hc_combine(
    h: torch.Tensor,
    hn: torch.Tensor,
    block_out: torch.Tensor,
    inject: torch.Tensor,
    config: Qwen38Config,
) -> torch.Tensor:
    """Inject the reduced block output into every branch with a per-branch sigmoid gain."""

    rows, hc, hidden = h.shape[0], config.hc_count, config.hidden
    gain = 2.0 * torch.sigmoid((hn.float() @ inject.float().T) / hc)
    out = h.float().view(rows, hc, hidden) + block_out.float()[:, None, :] * gain[:, :, None]
    return bf(out.view(rows, hc * hidden))


# -------------------------------------------------------------------- GDN


def golden_gdn_proj(
    x: torch.Tensor, t: dict, fp8_dtype: torch.dtype = ACTIVATION_FP8
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 ``[q | k | v | z]`` and BF16 ``[b | a]`` input projections."""

    return fp8_linear(x, t["gdn_qkvz"], t["gdn_qkvz_scale"], fp8_dtype), bf16_linear(x, t["gdn_ba"])


def golden_gdn_core(
    mixed_qkv: torch.Tensor,
    ba: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    *,
    state_indices: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal conv plus gated delta rule of ``mixed_qkv`` [rows, q | k | v].

    ``conv_state`` [slots, conv_dim, kernel - 1] and ``ssm_state``
    [slots, v_heads, V, K] update in place like the kernel; rows whose slot is
    negative produce zeros and leave every state untouched. Returns
    ``conv_out`` [rows, conv_dim] and ``core`` [rows, v_heads, V].
    """

    d = shard.config.gdn_head_dim
    key_dim, value_dim, hv, hk = shard.gdn_key_dim, shard.gdn_value_dim, shard.gdn_v_heads, shard.gdn_k_heads
    mixed, ba = mixed_qkv.float(), ba.float()
    beta_logit, a = ba[:, :hv], ba[:, hv:]
    conv_weight = t["gdn_conv"].float()
    decay_rate = torch.exp(t["gdn_a_log"].float())
    dt_bias = t["gdn_dt_bias"].float()
    conv_out = torch.zeros_like(mixed)
    core = torch.zeros(mixed.shape[0], hv, d, device=mixed.device)
    for row in range(mixed.shape[0]):
        slot = int(state_indices[row])
        if slot < 0:
            continue
        window = torch.cat((conv_state[slot].float(), mixed[row, :, None]), dim=1)
        # The Triton conv rounds every BF16 tap product before the FP32 sum.
        conv_out[row] = bf(F.silu(bf(window * conv_weight).sum(-1)))
        conv_state[slot] = window[:, 1:].to(conv_state.dtype)
        q, k, v = conv_out[row].split((key_dim, key_dim, value_dim))
        q = q.view(hk, d)
        k = k.view(hk, d)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + L2NORM_EPS) * d**-0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + L2NORM_EPS)
        q = q.repeat_interleave(hv // hk, dim=0)
        k = k.repeat_interleave(hv // hk, dim=0)
        g = -decay_rate * F.softplus(a[row] + dt_bias)
        beta = bf(torch.sigmoid(beta_logit[row]))
        state = ssm_state[slot].float() * torch.exp(g)[:, None, None]
        delta = (v.view(hv, d) - (state * k[:, None, :]).sum(-1)) * beta[:, None]
        state = state + delta[:, :, None] * k[:, None, :]
        core[row] = bf((state * q[:, None, :]).sum(-1))
        ssm_state[slot] = state.to(ssm_state.dtype)
    return conv_out, core


def golden_gdn_out(
    core: torch.Tensor,
    z: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm of ``core`` gated by ``sigmoid(z)``, then the row-parallel out projection."""

    rows, hv, d = core.shape
    core = core.float()
    normed = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + shard.config.eps) * t["gdn_norm"].float()
    normed = bf(normed * torch.sigmoid(z.float().reshape(rows, hv, d)))
    partial = fp8_linear(normed.reshape(rows, hv * d), t["gdn_out"], t["gdn_out_scale"], fp8_dtype)
    return normed, partial


def golden_gdn(
    x: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    *,
    state_indices: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> dict[str, torch.Tensor]:
    """Gated DeltaNet decode for independent requests; see ``golden_gdn_core`` for the states."""

    qkvz, ba = golden_gdn_proj(x, t, fp8_dtype)
    conv_out, core = golden_gdn_core(
        qkvz[:, : shard.conv_dim],
        ba,
        t,
        shard,
        state_indices=state_indices,
        conv_state=conv_state,
        ssm_state=ssm_state,
    )
    normed, partial = golden_gdn_out(core, qkvz[:, shard.conv_dim :], t, shard, fp8_dtype)
    return dict(qkvz=qkvz, ba=ba, conv_out=conv_out, core=core, normed=normed, partial=partial)


# -------------------------------------------------------------- attention


def golden_attention_proj(
    x: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    positions: torch.Tensor,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> dict[str, torch.Tensor]:
    """FP8 ``[q, gate | k | v]`` projection, Gemma QK norm, and NeoX RoPE at ``positions`` [rows].

    Returns ``q``/``gate`` [rows, heads, d] and ``k``/``v`` [rows, kv_heads, d].
    """

    c = shard.config
    rows, d, heads, kv_heads = x.shape[0], c.head_dim, shard.heads, shard.kv_heads
    q_width = heads * 2 * d
    qkv = fp8_linear(x, t["attn_qkv"], t["attn_qkv_scale"], fp8_dtype)
    q_gate = qkv[:, :q_width].reshape(rows, heads, 2 * d)
    q, gate = q_gate[..., :d], q_gate[..., d:]
    k = qkv[:, q_width : q_width + kv_heads * d].reshape(rows, kv_heads, d)
    v = qkv[:, q_width + kv_heads * d :].reshape(rows, kv_heads, d)
    q = bf(gemma_rmsnorm(q, t["attn_q_norm"], c.eps))
    k = bf(gemma_rmsnorm(k, t["attn_k_norm"], c.eps))
    cos, sin = rope_cos_sin(positions, c)
    q = bf(apply_neox_rope(q, cos, sin, c.rotary_dim))
    k = bf(apply_neox_rope(k, cos, sin, c.rotary_dim))
    return dict(qkv=qkv, q=q, k=k, v=v, gate=gate)


def golden_attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shard: Qwen38Shard,
    *,
    positions: torch.Tensor,
    k_cache: Sequence[torch.Tensor],
    v_cache: Sequence[torch.Tensor],
    selected: torch.Tensor,
    key_block: int = 32,
) -> torch.Tensor:
    """Sparse GQA decode of ``q`` [rows, heads, d] for independent requests.

    ``positions`` [rows] hold each new token's logical position. ``k_cache[row]``
    / ``v_cache[row]`` [capacity, kv_heads, d] hold that request's keys and
    values and receive ``k``/``v`` [rows, kv_heads, d] at ``positions[row]``.
    ``selected`` [rows, topk] lists logical positions; entries outside
    ``[0, position]`` are ignored. Like SGLang's decode kernel (32-key blocks
    up to 32 rows), the softmax runs online over ``key_block`` selected keys at
    a time and rounds probabilities to BF16 at the running max before the PV
    product. Returns the BF16-valued output [rows, heads, d].
    """

    d, heads, kv_heads = shard.config.head_dim, shard.heads, shard.kv_heads
    group = heads // kv_heads
    out = torch.zeros(q.shape[0], heads, d, device=q.device)
    for row in range(q.shape[0]):
        position = int(positions[row])
        k_cache[row][position] = k[row].to(k_cache[row].dtype)
        v_cache[row][position] = v[row].to(v_cache[row].dtype)
        keys_at = selected[row].long()
        keys_at = keys_at[(keys_at >= 0) & (keys_at <= position)]
        keys = k_cache[row][keys_at].float()
        values = v_cache[row][keys_at].float()
        # The decode kernel pre-scales q into the exp2 domain and rounds it to BF16.
        q_scaled = bf(q[row].float() * (d**-0.5 * LOG2E))
        for head in range(kv_heads):
            heads_of = slice(head * group, (head + 1) * group)
            running_max = torch.full((group,), float("-inf"), device=q.device)
            normalizer = torch.zeros(group, device=q.device)
            acc = torch.zeros(group, d, device=q.device)
            for start in range(0, keys.shape[0], key_block):
                scores = q_scaled[heads_of] @ keys[start : start + key_block, head].T
                next_max = torch.maximum(running_max, scores.amax(-1))
                alpha = torch.exp2(running_max - next_max)
                probs = torch.exp2(scores - next_max[:, None])
                acc = acc * alpha[:, None] + bf(probs) @ values[start : start + key_block, head]
                normalizer = normalizer * alpha + probs.sum(-1)
                running_max = next_max
            out[row, heads_of] = acc / normalizer[:, None]
    return bf(out)


def golden_attention_out(
    core: torch.Tensor,
    gate: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid output gate, then the row-parallel output projection: ``(gated, partial)``."""

    rows = core.shape[0]
    gated = bf(core.float() * torch.sigmoid(gate.float()))
    partial = fp8_linear(gated.reshape(rows, -1), t["attn_o"], t["attn_o_scale"], fp8_dtype)
    return gated, partial


def golden_attention(
    x: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    *,
    positions: torch.Tensor,
    k_cache: Sequence[torch.Tensor],
    v_cache: Sequence[torch.Tensor],
    selected: torch.Tensor,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> dict[str, torch.Tensor]:
    """Gated sparse-attention decode; see ``golden_attention_core`` for the cache inputs."""

    proj = golden_attention_proj(x, t, shard, positions, fp8_dtype)
    core = golden_attention_core(
        proj["q"],
        proj["k"],
        proj["v"],
        shard,
        positions=positions,
        k_cache=k_cache,
        v_cache=v_cache,
        selected=selected,
    )
    gated, partial = golden_attention_out(core, proj["gate"], t, shard, fp8_dtype)
    return dict(proj, core=core, gated=gated, partial=partial)


# --------------------------------------------------------------- QSA indexer


def golden_index_q(x: torch.Tensor, t: dict, positions: torch.Tensor, config: Qwen38Config):
    """SGLang's decode indexer queries (its unfused path once the RoPE table is BF16): the BF16
    ``index_qk_proj`` ``qk`` [rows, (heads + 1) * d], then Gemma-normalized and RoPE-rotated queries
    [rows, heads, d] (BF16 per operator)."""

    c = config
    rows = x.shape[0]
    qk = bf16_linear(x, t["index_qk"])
    q = qk[:, : c.index_heads * c.index_head_dim].reshape(rows, c.index_heads, c.index_head_dim)
    q = bf(gemma_rmsnorm(q, t["index_q_norm"], c.eps))
    cos, sin = rope_cos_sin(positions, c)
    return qk, apply_neox_rope(q, cos, sin, c.rotary_dim)


def golden_index_compress(members: torch.Tensor, k_norm: torch.Tensor, first_positions: torch.Tensor,
                          config: Qwen38Config) -> torch.Tensor:
    """Compressed index keys of complete groups: ``members`` [groups, ratio, d] raw BF16 keys -> the FP32 mean
    rounded to BF16, Gemma norm, RoPE at the group's first position; [groups, d]."""

    pooled = bf(members.float().mean(1))
    normed = bf(gemma_rmsnorm(pooled, k_norm, config.eps))
    cos, sin = rope_cos_sin(first_positions, config)
    return apply_neox_rope(normed[:, None], cos, sin, config.rotary_dim)[:, 0]


def golden_index_logits(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Decode scores of one row: ``relu(q_h . k)`` summed over the heads, over sqrt(d); q [heads, d],
    keys [blocks, d] -> FP32 [blocks]."""

    return torch.relu(q.float() @ keys.float().T).sum(0) / q.shape[-1] ** 0.5


def golden_index_select(logits: torch.Tensor, block_topk: int) -> torch.Tensor:
    """The ``block_topk`` best blocks in ascending order; ties keep the lower block (SGLang's ``fast_topk``
    returns the same set up to ties, in an unspecified order)."""

    if logits.numel() <= block_topk:
        return torch.arange(logits.numel(), device=logits.device)
    best = torch.sort(logits, descending=True, stable=True).indices[:block_topk]
    return best.sort().values


def golden_index_expand(blocks: torch.Tensor, seq_len: int, config: Qwen38Config) -> torch.Tensor:
    """SGLang's ``expand_qsa_block_indices`` for a decode row: the blocks' tokens, then the pending tail of the
    incomplete group, then -1; int32 [index_topk + ratio - 1]."""

    ratio = config.index_compress_ratio
    out = torch.full((config.index_topk + ratio - 1,), -1, dtype=torch.int32, device=blocks.device)
    tokens = (blocks[:, None] * ratio + torch.arange(ratio, device=blocks.device)).reshape(-1)[: config.index_topk]
    tokens = tokens[tokens < seq_len]
    tail = torch.arange((seq_len // ratio) * ratio, seq_len, device=blocks.device)
    out[: tokens.numel()] = tokens.to(torch.int32)
    out[tokens.numel() : tokens.numel() + tail.numel()] = tail.to(torch.int32)
    return out


# -------------------------------------------------------------------- MoE


def golden_route(x: torch.Tensor, t: dict, shard: Qwen38Shard):
    """Softmax top-k with renormalization, then the sigmoid-gated shared expert slot.

    BF16 router logits tie often; like SGLang's top-k, ties select the lower expert id.
    """

    c = shard.config
    logits = bf16_linear(x, t["router"])
    weights, ids = torch.softmax(logits, dim=-1).sort(dim=-1, descending=True, stable=True)
    weights, ids = weights[:, : c.top_k], ids[:, : c.top_k]
    weights = weights / weights.sum(-1, keepdim=True)
    shared = torch.sigmoid(x.float() @ t["shared_gate"].float())
    ids = torch.cat((ids, torch.full_like(ids[:, :1], c.shared_expert)), dim=-1)
    weights = torch.cat((weights, shared[:, None]), dim=-1)
    return logits, ids, weights


def golden_moe(
    x: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    *,
    ids: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Routed plus shared experts of one rank's intermediate shard.

    Like AITER's FlyDSL decode MoE (``batch1``), activations stay BF16 against
    the dequantized FP8 weights and the SiLU-gated intermediate is a BF16
    tensor; weighted expert outputs accumulate in FP32. ``ids``/``weights``
    [rows, top_k + 1] override routing so expert math can be checked from
    another implementation's selection.
    """

    logits = None
    if ids is None:
        logits, ids, weights = golden_route(x, t, shard)
    rows, inter = x.shape[0], shard.inter
    xf = x.float()
    out = torch.zeros(rows, shard.config.hidden, device=x.device)
    mid = torch.zeros(rows, ids.shape[1], inter, device=x.device)
    for row in range(rows):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            gate_up = (xf[row] @ t["moe_up"][expert].float().T) * t["moe_up_scale"][expert]
            mid[row, slot] = bf(F.silu(gate_up[:inter]) * gate_up[inter:])
            down = (mid[row, slot] @ t["moe_down"][expert].float().T) * t["moe_down_scale"][expert]
            out[row] += weights[row, slot].float() * down
    return dict(logits=logits, ids=ids, weights=weights, mid=mid, partial=bf(out))


# ------------------------------------------------------------------ layer


def _layer(
    kind: LayerKind,
    h: torch.Tensor,
    ranks: Sequence[tuple[dict, Qwen38Shard, dict]],
    reduce: Callable[[list[torch.Tensor]], torch.Tensor],
    fp8_dtype: torch.dtype,
) -> dict:
    c = ranks[0][1].config
    if h.shape[-1] == c.hidden:
        # Layer 0 starts from the token embeddings replicated into every branch.
        h = h.repeat(1, c.hc_count)
    t0 = ranks[0][0]
    hn = golden_hc_norm(h, t0["hc_attn_norm"], c)
    x = golden_hc_mix(hn, t0["hc_attn_down"], t0["hc_attn_up"], c)
    mixer = golden_gdn if kind is LayerKind.LINEAR else golden_attention
    blocks = [mixer(x, t, s, fp8_dtype=fp8_dtype, **inputs) for t, s, inputs in ranks]
    attn = reduce([b["partial"] for b in blocks])
    h1 = golden_hc_combine(h, hn, attn, t0["hc_attn_inject"], c)
    hn1 = golden_hc_norm(h1, t0["hc_mlp_norm"], c)
    xm = golden_hc_mix(hn1, t0["hc_mlp_down"], t0["hc_mlp_up"], c)
    logits, ids, weights = golden_route(xm, t0, ranks[0][1])
    moes = [golden_moe(xm, t, s, ids=ids, weights=weights) for t, s, _ in ranks]
    mlp = reduce([m["partial"] for m in moes])
    out = golden_hc_combine(h1, hn1, mlp, t0["hc_mlp_inject"], c)
    return dict(
        hc_attn_in=h,
        hc_attn_normed=hn,
        hc_attn_out=x,
        blocks=blocks,
        attn=attn,
        hc_attn_combined=h1,
        hc_mlp_normed=hn1,
        hc_mlp_out=xm,
        logits=logits,
        ids=ids,
        weights=weights,
        moes=moes,
        mlp=mlp,
        out=out,
    )


def golden_layer(
    kind: LayerKind,
    h: torch.Tensor,
    t: dict,
    shard: Qwen38Shard,
    allreduce: Callable[[torch.Tensor], torch.Tensor],
    *,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
    **mixer_inputs,
) -> dict:
    """One rank's view of a decode layer; ``allreduce`` sums BF16 partials across TP ranks.

    ``mixer_inputs`` are the keyword inputs of ``golden_gdn`` or ``golden_attention``.
    """

    return _layer(kind, h, [(t, shard, mixer_inputs)], lambda parts: allreduce(parts[0]), fp8_dtype)


def golden_allreduce(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum per-rank BF16 partials in rank order in FP32, then round to BF16."""

    total = parts[0].float()
    for part in parts[1:]:
        total = total + part.float()
    return bf(total)


def golden_layer_tp(
    kind: LayerKind,
    h: torch.Tensor,
    weights: Sequence[dict],
    shards: Sequence[Qwen38Shard],
    mixer_inputs: Sequence[dict],
    *,
    fp8_dtype: torch.dtype = ACTIVATION_FP8,
) -> dict:
    """Every TP rank of a decode layer in one process; reductions sum in rank order."""

    return _layer(kind, h, list(zip(weights, shards, mixer_inputs)), golden_allreduce, fp8_dtype)
