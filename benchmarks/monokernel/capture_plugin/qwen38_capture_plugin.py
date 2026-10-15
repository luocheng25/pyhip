"""SGLang general plugin: capture Qwen3.8 (qwen4_exp) decode-layer tensors for the MonoKernel golden check.

Discovered through the sglang.srt.plugins entry point in the sibling dist-info directory when this
folder is on PYTHONPATH (benchmarks/monokernel/capture.py does this), and active only when
QWEN38_CAPTURE_DIR is set. Every scheduler process writes tp{tp}_rank{rank}_step{step:03d}.pt per
decode step (first QWEN38_CAPTURE_STEPS steps) with CPU copies of each layer's operator inputs and
outputs, GDN states before/after, and the KV rows of every request. Run with --disable-cuda-graph:
replayed graphs never call the wrapped Python methods.
"""

import json
import os

import torch

_CAPTURE = None


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().contiguous().to('cpu', copy=True)
    return value


def _describe(value):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                    offset=value.storage_offset(), contiguous=value.is_contiguous(),
                    shuffled=value.__dict__.get('is_shuffled'))
    if isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
        return dict(count=len(value), first=_describe(value[0]))
    return None


def _describe_object(obj):
    found = {name: _describe(value) for name, value in vars(obj).items()}
    return dict(type=type(obj).__name__, tensors={k: v for k, v in found.items() if v is not None})


def _sample(value, count=4096):
    """Raw bits at evenly spaced flat indices, to compare a parameter's exact format offline."""
    flat = value.detach().reshape(-1)
    index = torch.linspace(0, flat.numel() - 1, min(count, flat.numel()), dtype=torch.float64,
                           device=flat.device).long()
    picked = flat[index]
    if picked.element_size() == 1:
        picked = picked.view(torch.uint8)
    elif picked.dtype in (torch.bfloat16, torch.float16):
        picked = picked.view(torch.int16)
    return dict(index=index.cpu(), bits=picked.cpu(), shape=tuple(value.shape), dtype=str(value.dtype))


def _layer_samples(layer):
    named = dict(layer.named_parameters())
    named.update(layer.named_buffers())
    return {name: _sample(value) for name, value in named.items()}


def _layer_layout(layer):
    """Parameter and state-pool layouts as SGLang holds them after weight post-processing."""
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

    params = {name: _describe(p) for name, p in layer.named_parameters()}
    params.update({name: _describe(b) for name, b in layer.named_buffers()})
    if hasattr(layer, 'linear_attn'):
        pool = get_attn_backend().linear_attn_backend.req_to_token_pool
        cache = pool.mamba2_layer_cache(layer.layer_id)
        state = dict(pool=_describe_object(pool), conv=_describe(cache.conv), temporal=_describe(cache.temporal))
    else:
        kv = get_token_to_kv_pool()
        state = dict(pool=_describe_object(kv), key=_describe(kv.get_key_buffer(layer.layer_id)),
                     value=_describe(kv.get_value_buffer(layer.layer_id)))
    return dict(params=params, state=state)


class _Capture:
    def __init__(self, out_dir, max_steps):
        self.out_dir = out_dir
        self.max_steps = max_steps
        self.step = 0
        self.meta = None
        self.layers = {}
        self.record = None
        self.hc_names = {}
        self.layout = {}
        self.samples = {}

    def put(self, key, value):
        if self.record is not None and value is not None:
            self.record[key] = _cpu(value)

    def flush(self):
        from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        meta = dict(self.meta, tp_size=tp, tp_rank=rank, step=self.step)
        path = os.path.join(self.out_dir, f'tp{tp}_rank{rank}_step{self.step:03d}.pt')
        torch.save(dict(meta=meta, layers=self.layers), path + '.tmp')
        os.replace(path + '.tmp', path)
        if self.step == 0:
            with open(os.path.join(self.out_dir, f'tp{tp}_rank{rank}_layout.json'), 'w') as f:
                json.dump(self.layout, f, indent=1)
            torch.save(self.samples, os.path.join(self.out_dir, f'tp{tp}_rank{rank}_param_samples.pt'))
        self.step += 1
        self.layers = {}


def _layer_forward(original, layer, *args, **kwargs):
    cap = _CAPTURE
    batch = kwargs.get('forward_batch')
    if cap.step >= cap.max_steps or batch is None or not batch.forward_mode.is_decode():
        return original(layer, *args, **kwargs)
    layer_id = layer.layer_id
    if layer_id == 0:
        cap.layers = {}
        cap.meta = dict(batch_size=batch.batch_size, seq_lens=_cpu(batch.seq_lens),
                        req_pool_indices=_cpu(batch.req_pool_indices), positions=_cpu(kwargs.get('positions')),
                        out_cache_loc=_cpu(batch.out_cache_loc), forward_mode=str(batch.forward_mode),
                        num_layers=layer.config.num_hidden_layers)
    if cap.step == 0:
        cap.layout[layer_id] = _layer_layout(layer)
        cap.samples[layer_id] = _layer_samples(layer)
    record = dict(kind='linear_attention' if hasattr(layer, 'linear_attn') else 'full_attention',
                  hidden_in=_cpu(kwargs.get('hidden_states')), residual_in=_cpu(kwargs.get('residual')))
    cap.hc_names = {id(layer.attn_hyper_connection): 'hc_attn', id(layer.mlp_hyper_connection): 'hc_mlp'}
    cap.record = record
    try:
        out = original(layer, *args, **kwargs)
    finally:
        cap.record = None
    record['out'] = _cpu(out[0])
    record['residual_out'] = _cpu(out[1])
    cap.layers[layer_id] = record
    if layer_id == layer.config.num_hidden_layers - 1:
        cap.flush()
    return out


def _hc_mix(original, module, *args, **kwargs):
    cap = _CAPTURE
    name = cap.hc_names.get(id(module)) if cap.record is not None else None
    if name is not None:
        cap.put(f'{name}.in', args[0] if args else kwargs.get('hyper_input'))
    mixed, residuals = original(module, *args, **kwargs)
    if name is not None:
        cap.put(f'{name}.normed', residuals[1])
        cap.put(f'{name}.out', mixed)
    return mixed, residuals


def _hc_combine(original, module, block_output, residuals):
    cap = _CAPTURE
    name = cap.hc_names.get(id(module)) if cap.record is not None else None
    if name is not None:
        cap.put(f'{name}.block', block_output)
    out = original(module, block_output, residuals)
    if name is not None:
        cap.put(f'{name}.combined', out)
    return out


def _gdn_forward(original, module, hidden_states, forward_batch):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    cap = _CAPTURE
    if cap.record is None:
        return original(module, hidden_states, forward_batch)
    backend = get_attn_backend().linear_attn_backend
    slots = backend.forward_metadata.mamba_cache_indices.long()
    cache = backend.req_to_token_pool.mamba2_layer_cache(module.layer_id)
    cap.put('gdn.x', hidden_states)
    cap.put('gdn.state_indices', slots)
    cap.put('gdn.conv_before', cache.conv[0][slots])
    cap.put('gdn.ssm_before', cache.temporal[slots])
    out = original(module, hidden_states, forward_batch)
    cap.put('gdn.conv_after', cache.conv[0][slots])
    cap.put('gdn.ssm_after', cache.temporal[slots])
    cap.put('gdn.partial', out)
    return out


def _gdn_input_proj(original, module, hidden_states):
    out = original(module, hidden_states)
    _CAPTURE.put('gdn.qkvz', out[0])
    _CAPTURE.put('gdn.ba', out[1])
    return out


def _linear_attn(original, module, forward_batch, mixed_qkv, a, b):
    cap = _CAPTURE
    cap.put('gdn.mixed_qkv', mixed_qkv)
    cap.put('gdn.a', a)
    cap.put('gdn.b', b)
    out = original(module, forward_batch, mixed_qkv, a, b)
    cap.put('gdn.core', out)
    return out


def _self_attention(original, layer, *args, **kwargs):
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool, get_token_to_kv_pool

    cap = _CAPTURE
    if cap.record is None:
        return original(layer, *args, **kwargs)
    names = ('positions', 'hidden_states', 'forward_batch')
    values = dict(zip(names, args), **kwargs)
    batch = values['forward_batch']
    cap.put('attn.positions', values['positions'])
    cap.put('attn.x', values['hidden_states'])
    out = original(layer, *args, **kwargs)
    cap.put('attn.partial', out)
    kv = get_token_to_kv_pool()
    keys, vals = kv.get_key_buffer(layer.attn.layer_id), kv.get_value_buffer(layer.attn.layer_id)
    rows = get_req_to_token_pool().req_to_token
    k_rows, v_rows = [], []
    for row in range(batch.batch_size):
        slots = rows[batch.req_pool_indices[row], : int(batch.seq_lens[row])].long()
        k_rows.append(_cpu(keys[slots]))
        v_rows.append(_cpu(vals[slots]))
    cap.record['attn.k_cache'] = k_rows
    cap.record['attn.v_cache'] = v_rows
    return out


def _prepare_qkv_gate(original, layer, *args, **kwargs):
    out = original(layer, *args, **kwargs)
    for name, value in zip(('q', 'k', 'v', 'gate'), out):
        _CAPTURE.put(f'attn.{name}', value)
    return out


def _qsa_topk(original, layer, *args, **kwargs):
    out = original(layer, *args, **kwargs)
    _CAPTURE.put('attn.topk', out)
    return out


def _radix_attention(original, module, *args, **kwargs):
    out = original(module, *args, **kwargs)
    _CAPTURE.put('attn.core', out)
    return out


def _moe_forward(original, block, *args, **kwargs):
    _CAPTURE.put('moe.x', args[0] if args else kwargs.get('hidden_states'))
    out = original(block, *args, **kwargs)
    _CAPTURE.put('moe.out', out)
    return out


def _append_shared(original, block, topk_output, hidden_states):
    cap = _CAPTURE
    cap.put('moe.logits', topk_output.router_logits)
    cap.put('moe.routed_ids', topk_output.topk_ids)
    cap.put('moe.routed_weights', topk_output.topk_weights)
    out = original(block, topk_output, hidden_states)
    cap.put('moe.ids', out.topk_ids)
    cap.put('moe.weights', out.topk_weights)
    return out


def _router_experts(original, block, hidden_states):
    out = original(block, hidden_states)
    _CAPTURE.put('moe.partial', out)
    return out


_HOOKS = {
    'sglang.srt.models.qwen4_exp.Qwen4ExpLinearDecoderLayer.forward': _layer_forward,
    'sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer.forward': _layer_forward,
    'sglang.srt.layers.hyperconnection.GatedResidual.mix': _hc_mix,
    'sglang.srt.layers.hyperconnection.GatedResidual.combine': _hc_combine,
    'sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward': _gdn_forward,
    'sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj': _gdn_input_proj,
    'sglang.srt.layers.radix_linear_attention.RadixLinearAttention.forward': _linear_attn,
    'sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer.self_attention': _self_attention,
    'sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer._prepare_qkv_gate': _prepare_qkv_gate,
    'sglang.srt.models.qwen4_exp.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices': _qsa_topk,
    'sglang.srt.layers.radix_attention.RadixAttention.forward': _radix_attention,
    'sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock.forward': _moe_forward,
    'sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._append_shared_to_topk_output': _append_shared,
    'sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts': _router_experts,
}


def register():
    global _CAPTURE
    out_dir = os.environ.get('QWEN38_CAPTURE_DIR')
    if not out_dir:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    os.makedirs(out_dir, exist_ok=True)
    _CAPTURE = _Capture(out_dir, int(os.environ.get('QWEN38_CAPTURE_STEPS', '4')))
    for target, hook in _HOOKS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
