"""SGLang hook: run Qwen3.8-Flash-Next decode layers through the fused MonoKernel launches.

Enabled with ``PYHIP_MONOKERNEL=1`` (plugin ``qwen38_monokernel``, see
``benchmarks/monokernel/plugin``). For a normal decode batch (``ForwardMode.DECODE``, at most 64
rows, no ReplaySSM) every main-model layer other than the PLE layer runs as one launch: the mixer (GDN:
qkvz/ba projections, causal conv, delta rule, gated norm, ``out_proj``; full attention: qkv, q/k norm,
MRoPE, the KV store, the QSA index queries, the pending index-key ring and the compression of completed
groups in SGLang's pools, the index logits and top-k, sparse attention over the selected positions,
``o_proj``) from the previous layer's attention-branch norm (SGLang's ``residual``), the attention
all-reduce, the MLP half .. MoE down, the MoE all-reduce, and the combine with the next layer's
attention-branch norm. The all-reduces run inside the launch over XGMI (``flydsl/xgmi.py``, regions in
``peer.PeerMemory``), bit for bit AITER's one-stage all-reduce. ``PYHIP_MONOKERNEL_AR=aiter`` splits a
layer at the all-reduces instead (the GDN attention half / the attention ``prep`` + ``core``, SGLang's
attention all-reduce, the MLP half, SGLang's MoE all-reduce, ``hc_combine_norm``). With
``PYHIP_MONOKERNEL_ATTENTION=native`` the full-attention layer runs the fused mix, then SGLang's own
``self_attention`` (split path). Everything else (prefill, MTP verify/draft, the MTP draft layer, layer 1)
stays native. The launches are CUDA-graph capturable; kernels compile during SGLang's eager warmup.

``PYHIP_MONOKERNEL_VALIDATE=1`` (eager batches only) also runs the native layer on the same input
(and GDN states), logs per-layer metrics against it and keeps the MonoKernel results.
"""

from __future__ import annotations

import logging
import os

import torch

from .config import LayerKind, shard
from .flydsl.fused import FusedAttentionLayer, FusedGdnLayer, FusedLayer, FusedWorkspace
from .flydsl.hc import hc_combine_norm
from .flydsl.xgmi import region_bytes
from .peer import PeerMemory
from .weights import SGLANG_NAMES, pack_layer

logger = logging.getLogger(__name__)
CAPACITY = 64
PLE_LAYER = 1


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


class _Runtime:
    def __init__(self):
        self.layers = {}
        self.workspaces = {}
        self.validate = _enabled("PYHIP_MONOKERNEL_VALIDATE")
        self.native_attention = os.environ.get("PYHIP_MONOKERNEL_ATTENTION", "mono") == "native"
        self.fused_ar = os.environ.get("PYHIP_MONOKERNEL_AR", "fused") != "aiter"
        self.calls = 0
        self.capture_ids = False
        self.native_ids = None

    def accepts(self, layer, batch, hidden_states) -> bool:
        if batch is None or not batch.forward_mode.is_decode():
            return False
        if layer.layer_id == PLE_LAYER or getattr(layer, "ple", None) is not None:
            return False
        # The MTP draft model is a one-layer Qwen4ExpModel.
        if layer.config.num_hidden_layers == 1 or not 0 < hidden_states.shape[0] <= CAPACITY:
            return False
        from sglang.srt.model_executor.forward_context import get_attn_backend

        metadata = get_attn_backend().linear_attn_backend.forward_metadata
        return getattr(metadata, "replayssm_write_pos", None) is None

    def layer(self, layer, kind: LayerKind) -> FusedLayer:
        mono = self.layers.get(layer.layer_id)
        if mono is None:
            from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

            rank_shard = shard(get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank())
            params = dict(layer.named_parameters())
            served = {name: params[path] for name, path in SGLANG_NAMES[kind].items()}
            with torch.no_grad():
                packed = pack_layer(served, kind, rank_shard.config)
            mono = (FusedGdnLayer if kind is LayerKind.LINEAR else FusedAttentionLayer)(rank_shard, served, packed)
            self.layers[layer.layer_id] = mono
            logger.info("MonoKernel %s layer %d ready (tp=%d rank=%d)", kind.value, layer.layer_id, rank_shard.tp,
                        rank_shard.rank)
        return mono

    def workspace(self, mono: FusedLayer, device) -> FusedWorkspace:
        work = self.workspaces.get(device)
        if work is None:
            from sglang.srt.distributed import get_attn_tp_group, get_tp_group
            from sglang.srt.model_executor.forward_context import get_req_to_token_pool

            # index logits for every compressed block a request row can hold
            positions = get_req_to_token_pool().req_to_token.shape[1]
            max_blocks = -(-positions // (4 * 16)) * 16
            comm = None
            tp_group = get_tp_group()
            # in-launch all-reduces: both all-reduces of the layer over the whole tensor-parallel group
            if self.fused_ar and get_attn_tp_group().world_size == tp_group.world_size:
                comm = PeerMemory(region_bytes(tp_group.world_size, CAPACITY), tp_group.cpu_group, device)
            work = FusedWorkspace(CAPACITY, mono.shard, device, max_blocks=max_blocks, comm=comm)
            self.workspaces[device] = work
            logger.info("MonoKernel workspace on %s: all-reduce %s", device,
                        "in launch" if comm is not None else "AITER")
        return work


_RUNTIME = None


def _gdn_state(layer):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend().linear_attn_backend
    slots = backend.forward_metadata.mamba_cache_indices
    cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    return slots.to(torch.int32), cache.conv[0], cache.temporal


def _int64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.int64 else tensor.to(torch.int64)


def _attention_pools(layer, forward_batch):
    """SGLang's KV and QSA pools of a full-attention layer and the batch's request tables."""
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool, get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    k_cache, v_cache = pool.get_key_buffer(layer.layer_id), pool.get_value_buffer(layer.layer_id)
    req_to_token = get_req_to_token_pool().req_to_token
    req_pool, seq_lens = _int64(forward_batch.req_pool_indices), _int64(forward_batch.seq_lens)
    return pool, k_cache, v_cache, req_to_token, req_pool, seq_lens


def _attention_prep(mono: FusedAttentionLayer, layer, h, x, work, gain, positions, forward_batch):
    pool, k_cache, v_cache, req_to_token, req_pool, seq_lens = _attention_pools(layer, forward_batch)
    rotary = layer.rotary_emb
    rotary._match_cos_sin_cache_dtype(x)
    mono.prep(h, x, positions, _int64(forward_batch.out_cache_loc), k_cache, v_cache, rotary.cos_sin_cache, work,
              gain, req_pool, seq_lens, req_to_token, pool.get_qsa_key_state_buffer(layer.layer_id),
              pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer.layer_id))
    return k_cache, v_cache, req_to_token, req_pool, seq_lens


def _attention_core(rt: _Runtime, mono: FusedAttentionLayer, layer, h, work, gain, positions, forward_batch):
    """``prep`` and ``core`` launches (QSA indexer included, its state in SGLang's pool): this rank's ``o_proj``
    partial of a full-attention layer."""
    rows, hidden = h.shape[0], mono.shard.config.hidden
    x = torch.empty(rows, hidden, dtype=torch.bfloat16, device=h.device)
    if rt.native_attention:
        mono.mix(h, x, work, gain)
        return layer.self_attention(positions=positions, hidden_states=x, forward_batch=forward_batch)
    k_cache, v_cache, req_to_token, req_pool, seq_lens = _attention_prep(mono, layer, h, x, work, gain, positions,
                                                                         forward_batch)
    part = torch.empty(rows, hidden, dtype=torch.bfloat16, device=h.device)
    mono.core(req_to_token, req_pool, seq_lens, k_cache, v_cache, part, work)
    return part


def _mono_layer(rt: _Runtime, layer, hidden_states, hc_consumer, attention=None, hn=None):
    """One decode layer; ``attention = (positions, forward_batch)`` for a full-attention layer; ``hn``: the
    attention-branch norm of ``hidden_states`` from the previous layer (SGLang's ``residual``), if any."""
    mono = rt.layer(layer, LayerKind.LINEAR if attention is None else LayerKind.FULL)
    width = mono.shard.config.hc_width
    h = hidden_states if hidden_states.shape[-1] == width else torch.cat([hidden_states] * mono.shard.config.hc_count,
                                                                         dim=-1)
    h = h.contiguous()
    if hn is not None and (hn.shape != h.shape or not hn.is_contiguous()):
        hn = None
    work = rt.workspace(mono, h.device)
    weight = hc_consumer.hc_norm.weight if hc_consumer is not None else mono.served["hc_attn_norm"]
    out, normed = torch.empty_like(h), torch.empty_like(h)
    if work.comm is None or (attention is not None and rt.native_attention):
        _split_layer(rt, mono, layer, h, work, weight, out, normed, attention)
    elif attention is None:
        slots, conv, ssm = _gdn_state(layer)
        mono.layer(h, slots, conv, ssm, weight, out, normed, work, hn=hn)
    else:
        positions, forward_batch = attention
        pool, k_cache, v_cache, req_to_token, req_pool, seq_lens = _attention_pools(layer, forward_batch)
        rotary = layer.rotary_emb
        rotary._match_cos_sin_cache_dtype(h)
        mono.layer(h, positions, _int64(forward_batch.out_cache_loc), k_cache, v_cache, rotary.cos_sin_cache, req_pool,
                   seq_lens, req_to_token, pool.get_qsa_key_state_buffer(layer.layer_id),
                   pool.qsa_rope_position_buffer, pool.get_qsa_compressed_k_buffer(layer.layer_id), weight, out,
                   normed, work, hn=hn)
    return out, (normed if hc_consumer is not None else None)


def _split_layer(rt: _Runtime, mono, layer, h, work, weight, out, normed, attention):
    """The layer split at SGLang's (AITER) all-reduces."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.dp_attention import attn_tp_all_reduce

    rows = h.shape[0]
    hidden = mono.shard.config.hidden
    gains = torch.empty(2, rows, mono.shard.config.hc_count, dtype=torch.float32, device=h.device)
    if attention is None:
        slots, conv, ssm = _gdn_state(layer)
        part_a = torch.empty(rows, hidden, dtype=torch.bfloat16, device=h.device)
        mono.attention(h, slots, conv, ssm, part_a, work, gain_out=gains[0])
    else:
        part_a = _attention_core(rt, mono, layer, h, work, gains[0], *attention)
    attn = attn_tp_all_reduce(part_a)
    h1 = torch.empty_like(h)
    part_m = torch.empty_like(attn)
    mono.mlp(h, attn, gains[0], h1, gains[1], part_m, work)
    moe = tensor_model_parallel_all_reduce(part_m)
    hc_combine_norm(h1, moe, gains[1], weight, out, normed)


def _metrics(got: torch.Tensor, expected: torch.Tensor) -> dict:
    got, expected = got.float(), expected.float()
    delta = (got - expected).abs()
    scale = expected.norm().clamp_min(1e-30)
    magnitude = torch.maximum(got.abs(), expected.abs()).clamp_min(torch.finfo(torch.bfloat16).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
    return dict(rel_l2=float(delta.norm() / scale), max_abs=float(delta.max()),
                ulp1=float((delta <= ulp).float().mean()))


def _validated(rt: _Runtime, run_native, layer, hidden_states, hc_consumer, attention=None, hn=None):
    """MonoKernel layer, then the native layer twice from the same input (GDN: and states), compared; the
    second native run (``native_repeat``) measures SGLang's own run-to-run spread (atomic HC mix and MoE
    accumulation, top-k collection order). A full-attention layer's KV, indexer ring and compressed-key
    writes are rewritten by a native run before it reads them, so they need no restore; they end up native."""
    gdn = attention is None
    if gdn:
        slots, conv, ssm = _gdn_state(layer)
        index = slots.long()
        conv_before, ssm_before = conv[index].clone(), ssm[index].clone()
    out, normed = _mono_layer(rt, layer, hidden_states, hc_consumer, attention, hn)
    rows = out.shape[0]
    mono = rt.layers[layer.layer_id]
    slots_per_row = mono.shard.config.top_k + 1
    ids = rt.workspace(mono, out.device).view("ids", (rows, slots_per_row), torch.int32).clone()
    if gdn:
        conv_mono, ssm_mono = conv[index].clone(), ssm[index].clone()
        conv[index], ssm[index] = conv_before, ssm_before
    rt.capture_ids, rt.native_ids = True, None
    try:
        ref_out, ref_normed = run_native()
    finally:
        rt.capture_ids = False
    report = dict(layer=layer.layer_id, rows=int(rows), out=_metrics(out, ref_out))
    if gdn:
        report.update(conv=_metrics(conv_mono, conv[index]), ssm=_metrics(ssm_mono, ssm[index]))
        conv[index], ssm[index] = conv_before, ssm_before
    repeat_out, _ = run_native()
    report["native_repeat"] = _metrics(repeat_out, ref_out)
    if normed is not None and ref_normed is not None:
        report["normed"] = _metrics(normed, ref_normed)
    if rt.native_ids is not None and rt.native_ids.shape == ids.shape:
        same = (ids.sort(dim=1).values == rt.native_ids.to(torch.int32).sort(dim=1).values).all(dim=1)
        report["route_mismatch_rows"] = int((~same).sum())
        if bool(same.any()):
            report["out_same_route"] = _metrics(out[same], ref_out[same])
    logger.info("MonoKernel validate %s", report)
    if gdn:
        conv[index], ssm[index] = conv_mono, ssm_mono
    return out, normed


def append_shared(original, block, topk_output, hidden_states):
    """Validation only: keep the native routing (with the shared expert) of the layer being compared."""
    out = original(block, topk_output, hidden_states)
    rt = _RUNTIME
    if rt is not None and rt.capture_ids:
        rt.native_ids = out.topk_ids.clone()
    return out


def gdn_layer_forward(original, layer, hidden_states, residual=None, **kwargs):
    rt = _RUNTIME
    batch = kwargs.get("forward_batch")
    if rt is None or not rt.accepts(layer, batch, hidden_states):
        return original(layer, hidden_states, residual, **kwargs)
    rt.calls += 1
    consumer = kwargs.get("hc_consumer")
    if rt.validate and not torch.cuda.is_current_stream_capturing():
        return _validated(rt, lambda: original(layer, hidden_states, residual, **kwargs), layer, hidden_states,
                          consumer, hn=residual)
    return _mono_layer(rt, layer, hidden_states, consumer, hn=residual)


def attention_layer_forward(original, layer, positions, hidden_states, residual, forward_batch, **kwargs):
    rt = _RUNTIME
    if rt is None or not rt.accepts(layer, forward_batch, hidden_states):
        return original(layer, positions, hidden_states, residual, forward_batch, **kwargs)
    rt.calls += 1
    consumer = kwargs.get("hc_consumer")
    attention = (positions, forward_batch)
    if rt.validate and not torch.cuda.is_current_stream_capturing():
        return _validated(rt, lambda: original(layer, positions, hidden_states, residual, forward_batch, **kwargs),
                          layer, hidden_states, consumer, attention, hn=residual)
    return _mono_layer(rt, layer, hidden_states, consumer, attention, hn=residual)


_HOOKS = {
    "sglang.srt.models.qwen4_exp.Qwen4ExpLinearDecoderLayer.forward": gdn_layer_forward,
    "sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer.forward": attention_layer_forward,
}


def register():
    global _RUNTIME
    if not _enabled("PYHIP_MONOKERNEL"):
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _RUNTIME = _Runtime()
    hooks = dict(_HOOKS)
    if _RUNTIME.validate:
        hooks["sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._append_shared_to_topk_output"] = append_shared
    for target, hook in hooks.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    logger.info("MonoKernel decode hooks registered (validate=%s, attention=%s)", _RUNTIME.validate,
                "native" if _RUNTIME.native_attention else "mono")
