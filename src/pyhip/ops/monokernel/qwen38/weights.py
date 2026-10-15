"""Checkpoint names, SGLang-equivalent TP sharding, SGLang's served format, and synthetic weights.

Per-rank golden tensors keep the checkpoint's E4M3FN values with FP32 per-row
scales. ``sglang_format`` converts them to what SGLang holds after loading on
gfx942 with AITER, which the MonoKernel reads in place: E4M3FNUZ bits with
doubled scales (every dequantized value is unchanged), AITER's 16x16 weight
shuffle, and expert intermediate rows zero-padded to ``MOE_PADDING``.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from pyhip.ops.monokernel.qwen38.config import LayerKind, Qwen38Config, Qwen38Shard

CHECKPOINT_FP8 = torch.float8_e4m3fn
KERNEL_FP8 = torch.float8_e4m3fnuz
# AITER_MOE_PADDING_SIZE of the serving launcher; SGLang zero-pads each expert's local intermediate dim to it.
MOE_PADDING = 64
LAYER_PREFIX = "model.language_model.layers.{}."
EXPERT_TENSORS = ("moe_up", "moe_up_scale", "moe_down", "moe_down_scale")

# Served tensor name -> SGLang parameter name under the decoder layer module.
SGLANG_COMMON = {
    **{
        f"{prefix}_{part}": f"{module}.{name}.weight"
        for prefix, module in (("hc_attn", "attn_hyper_connection"), ("hc_mlp", "mlp_hyper_connection"))
        for part, name in (
            ("norm", "hc_norm"),
            ("down", "input_mix_weight_down"),
            ("up", "input_mix_weight_up"),
            ("inject", "block_inject_weight"),
        )
    },
    "router": "mlp.gate.weight",
    "shared_gate": "mlp.shared_expert_gate.weight",
    "moe_w13": "mlp.experts.w13_weight",
    "moe_w13_scale": "mlp.experts.w13_weight_scale",
    "moe_w2": "mlp.experts.w2_weight",
    "moe_w2_scale": "mlp.experts.w2_weight_scale",
}
SGLANG_NAMES = {
    LayerKind.LINEAR: {
        **SGLANG_COMMON,
        "gdn_qkvz": "linear_attn.in_proj_qkvz.weight",
        "gdn_qkvz_scale": "linear_attn.in_proj_qkvz.weight_scale",
        "gdn_ba": "linear_attn.in_proj_ba.weight",
        "gdn_conv": "linear_attn.conv1d.weight",
        "gdn_a_log": "linear_attn.A_log",
        "gdn_dt_bias": "linear_attn.dt_bias",
        "gdn_norm": "linear_attn.norm.weight",
        "gdn_out": "linear_attn.out_proj.weight",
        "gdn_out_scale": "linear_attn.out_proj.weight_scale",
    },
    LayerKind.FULL: {
        **SGLANG_COMMON,
        "attn_qkv": "qkv_proj.weight",
        "attn_qkv_scale": "qkv_proj.weight_scale",
        "attn_q_norm": "q_norm.weight",
        "attn_k_norm": "k_norm.weight",
        "attn_o": "o_proj.weight",
        "attn_o_scale": "o_proj.weight_scale",
        "index_qk": "indexer.index_qk_proj.weight",
        "index_q_norm": "indexer.q_layernorm.weight",
        "index_k_norm": "indexer.k_layernorm.weight",
    },
}
_DENSE_FP8 = ("gdn_qkvz", "gdn_out", "attn_qkv", "attn_o")

# get(name, dim=None, start=0, stop=None) returns a checkpoint-named layer tensor,
# sliced along ``dim`` when given.
Getter = Callable[..., torch.Tensor]


def _local(total: int, shard: Qwen38Shard) -> tuple[int, int]:
    size = total // shard.tp
    return shard.rank * size, (shard.rank + 1) * size


def _rows_and_scale(get: Getter, name: str, ranges) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.cat([get(f"{name}.weight", 0, lo, hi) for lo, hi in ranges])
    scale = torch.cat([get(f"{name}.weight_scale", 0, lo, hi) for lo, hi in ranges])
    return weight, scale.float().reshape(-1)


def _gdn_ranges(shard: Qwen38Shard):
    """Rows of this rank's q, k, and v blocks in ``in_proj_qkv`` and ``conv1d``."""

    c = shard.config
    key_dim = c.gdn_k_heads * c.gdn_head_dim
    value_dim = c.gdn_v_heads * c.gdn_head_dim
    klo, khi = _local(key_dim, shard)
    vlo, vhi = _local(value_dim, shard)
    return ((klo, khi), (key_dim + klo, key_dim + khi), (2 * key_dim + vlo, 2 * key_dim + vhi)), (vlo, vhi)


def _shard_gdn(get: Getter, shard: Qwen38Shard) -> dict[str, torch.Tensor]:
    c = shard.config
    qkv_ranges, (vlo, vhi) = _gdn_ranges(shard)
    hlo, hhi = _local(c.gdn_v_heads, shard)
    qkv, qkv_scale = _rows_and_scale(get, "linear_attn.in_proj_qkv", qkv_ranges)
    z, z_scale = _rows_and_scale(get, "linear_attn.in_proj_z", ((vlo, vhi),))
    conv = torch.cat([get("linear_attn.conv1d.weight", 0, lo, hi) for lo, hi in qkv_ranges])
    return {
        "gdn_qkvz": torch.cat((qkv, z)),
        "gdn_qkvz_scale": torch.cat((qkv_scale, z_scale)),
        # SGLang's merged in_proj_ba stores b before a.
        "gdn_ba": torch.cat(
            (get("linear_attn.in_proj_b.weight", 0, hlo, hhi), get("linear_attn.in_proj_a.weight", 0, hlo, hhi))
        ),
        "gdn_conv": conv.reshape(-1, c.conv_kernel),
        "gdn_a_log": get("linear_attn.A_log", 0, hlo, hhi).float(),
        "gdn_dt_bias": get("linear_attn.dt_bias", 0, hlo, hhi),
        "gdn_norm": get("linear_attn.norm.weight"),
        "gdn_out": get("linear_attn.out_proj.weight", 1, vlo, vhi),
        "gdn_out_scale": get("linear_attn.out_proj.weight_scale").float().reshape(-1),
    }


def _shard_attention(get: Getter, shard: Qwen38Shard) -> dict[str, torch.Tensor]:
    c = shard.config
    qlo, qhi = _local(c.heads * 2 * c.head_dim, shard)
    kv_lo = shard.kv_head_start * c.head_dim
    kv_hi = kv_lo + shard.kv_heads * c.head_dim
    q, q_scale = _rows_and_scale(get, "self_attn.q_proj", ((qlo, qhi),))
    k, k_scale = _rows_and_scale(get, "self_attn.k_proj", ((kv_lo, kv_hi),))
    v, v_scale = _rows_and_scale(get, "self_attn.v_proj", ((kv_lo, kv_hi),))
    olo, ohi = _local(c.heads * c.head_dim, shard)
    return {
        # Each query head stores its 256 query rows followed by 256 output-gate rows.
        "attn_qkv": torch.cat((q, k, v)),
        "attn_qkv_scale": torch.cat((q_scale, k_scale, v_scale)),
        "attn_q_norm": get("self_attn.q_norm.weight"),
        "attn_k_norm": get("self_attn.k_norm.weight"),
        "attn_o": get("self_attn.o_proj.weight", 1, olo, ohi),
        "attn_o_scale": get("self_attn.o_proj.weight_scale").float().reshape(-1),
        "index_qk": get("self_attn.indexer.index_qk_proj.weight"),
        "index_q_norm": get("self_attn.indexer.q_layernorm.weight"),
        "index_k_norm": get("self_attn.indexer.k_layernorm.weight"),
    }


def _expert_prefix(config: Qwen38Config, expert: int) -> str:
    return f"mlp.experts.{expert}" if expert < config.n_experts else "mlp.shared_expert"


def _shard_expert(get: Getter, shard: Qwen38Shard, expert: int):
    prefix = _expert_prefix(shard.config, expert)
    rows = (_local(shard.config.inter, shard),)
    gate, gate_scale = _rows_and_scale(get, f"{prefix}.gate_proj", rows)
    up, up_scale = _rows_and_scale(get, f"{prefix}.up_proj", rows)
    lo, hi = rows[0]
    down = get(f"{prefix}.down_proj.weight", 1, lo, hi)
    down_scale = get(f"{prefix}.down_proj.weight_scale").float().reshape(-1)
    return torch.cat((gate, up)), torch.cat((gate_scale, up_scale)), down, down_scale


def shard_experts(get: Getter, shard: Qwen38Shard, experts: Iterable[int]) -> dict[str, dict[int, torch.Tensor]]:
    """``moe_*`` banks keyed by expert id; the shared expert is id ``n_experts``."""

    banks = {name: {} for name in EXPERT_TENSORS}
    for expert in sorted({int(e) for e in experts}):
        for name, value in zip(EXPERT_TENSORS, _shard_expert(get, shard, expert)):
            banks[name][expert] = value
    return banks


def shard_layer_weights(
    get: Getter,
    shard: Qwen38Shard,
    kind: LayerKind,
    experts: Iterable[int] | None = None,
) -> dict:
    """Slice checkpoint-named layer tensors into one rank's golden inputs.

    ``experts=None`` stacks every routed expert plus the shared expert (id
    ``n_experts``); an iterable loads only those ids into dicts keyed by id.
    """

    t = {}
    for prefix, module in (("hc_attn", "attn_hyper_connection"), ("hc_mlp", "mlp_hyper_connection")):
        t[f"{prefix}_norm"] = get(f"{module}.hc_norm.weight")
        t[f"{prefix}_down"] = get(f"{module}.input_mix_weight_down.weight")
        t[f"{prefix}_up"] = get(f"{module}.input_mix_weight_up.weight")
        t[f"{prefix}_inject"] = get(f"{module}.block_inject_weight.weight")
    t.update(_shard_gdn(get, shard) if kind is LayerKind.LINEAR else _shard_attention(get, shard))
    t["router"] = get("mlp.gate.weight")
    t["shared_gate"] = get("mlp.shared_expert_gate.weight").reshape(-1)
    if experts is None:
        ids = range(shard.config.n_experts + 1)
        banks = shard_experts(get, shard, ids)
        t.update({name: torch.stack([bank[e] for e in ids]) for name, bank in banks.items()})
    else:
        t.update(shard_experts(get, shard, experts))
    return t


def shard_gdn_state(conv_state: torch.Tensor, ssm_state: torch.Tensor, shard: Qwen38Shard):
    """Copy one rank's slice of TP1 GDN state [slots, conv_dim, K - 1] / [slots, v_heads, V, K]."""

    qkv_ranges, _ = _gdn_ranges(shard)
    hlo, hhi = _local(shard.config.gdn_v_heads, shard)
    conv = torch.cat([conv_state[:, lo:hi] for lo, hi in qkv_ranges], dim=1)
    return conv, ssm_state[:, hlo:hhi].clone(memory_format=torch.contiguous_format)


def shard_kv_heads(cache: torch.Tensor, shard: Qwen38Shard) -> torch.Tensor:
    """Copy this rank's KV heads from a TP1 cache [tokens, kv_heads, head_dim]."""

    heads = cache[:, shard.kv_head_start : shard.kv_head_start + shard.kv_heads]
    return heads.clone(memory_format=torch.contiguous_format)


def to_fnuz(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang's ``normalize_e4m3fn_to_e4m3fnuz``: the same bits read as E4M3FNUZ (half the value)
    with doubled scales; E4M3FN's -0 (0x80, NaN in E4M3FNUZ) becomes 0."""

    bits = weight.view(torch.int8).clone()
    bits[bits == -128] = 0
    return bits.view(KERNEL_FP8), scale.float() * 2.0


def shuffle_16x16(weight: torch.Tensor) -> torch.Tensor:
    """AITER's ``shuffle_weight(w, (16, 16))`` over the last two dims [..., N, K]: tiles of 16 rows
    by 16 bytes, stored row-major within a tile and K-major across tiles of a 16-row block."""

    n, k = weight.shape[-2:]
    pack = 16 // weight.element_size()
    if n % 16 or k % 32:
        raise ValueError(f"shuffle_16x16 needs N % 16 == 0 and K % 32 == 0, got {n}x{k}")
    tiles = weight.reshape(-1, n // 16, 16, k // pack, pack).permute(0, 1, 3, 2, 4)
    return tiles.contiguous().reshape(weight.shape)


def unshuffle_16x16(weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape[-2:]
    pack = 16 // weight.element_size()
    tiles = weight.reshape(-1, n // 16, k // pack, 16, pack).permute(0, 1, 3, 2, 4)
    return tiles.contiguous().reshape(weight.shape)


def moe_inter_padded(shard: Qwen38Shard) -> int:
    local = shard.config.inter // shard.tp
    return -(-local // MOE_PADDING) * MOE_PADDING


def sglang_format(t: dict, shard: Qwen38Shard, kind: LayerKind) -> dict[str, torch.Tensor]:
    """One rank's golden tensors (experts stacked, ``experts=None``) in SGLang's served format.

    Keys are those of ``SGLANG_NAMES[kind]``; values match the SGLang parameters bit for bit.
    """

    out = {}
    for name in SGLANG_NAMES[kind]:
        if name in _DENSE_FP8:
            weight, scale = to_fnuz(t[name], t[f"{name}_scale"])
            out[name], out[f"{name}_scale"] = shuffle_16x16(weight), scale.reshape(-1, 1)
        elif name == "gdn_conv":
            out[name] = t[name].reshape(-1, 1, shard.config.conv_kernel)
        elif name == "shared_gate":
            out[name] = t[name].reshape(1, -1)
        elif name in t and not name.endswith("_scale"):
            out[name] = t[name]
    inter, padded = shard.config.inter // shard.tp, moe_inter_padded(shard)
    up, up_scale = to_fnuz(t["moe_up"], t["moe_up_scale"])
    down, down_scale = to_fnuz(t["moe_down"], t["moe_down_scale"])
    pad = padded - inter

    def pad_halves(value: torch.Tensor) -> torch.Tensor:
        # Zero rows after the gate half and after the up half of every expert.
        widths = (0, 0) * (value.dim() - 2) + (0, pad)
        return torch.cat([torch.nn.functional.pad(half, widths) for half in value.split(inter, dim=1)], dim=1)

    out["moe_w13"] = shuffle_16x16(pad_halves(up.view(torch.int8)).view(KERNEL_FP8))
    out["moe_w13_scale"] = pad_halves(up_scale[..., None])
    out["moe_w2"] = shuffle_16x16(torch.nn.functional.pad(down.view(torch.int8), (0, pad)).view(KERNEL_FP8))
    out["moe_w2_scale"] = down_scale[..., None]
    return out


def _pad_rows(weight: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    return torch.nn.functional.pad(weight, (0, 0, 0, -weight.shape[0] % multiple))


def pack_hc_up(up: torch.Tensor, config: Qwen38Config) -> torch.Tensor:
    """HC up [hc * hidden, lowrank] with each 16-row block holding columns 4b..4b+3 of all branches
    (row ``16 b + 4 jj + g`` = branch ``g`` column ``4 b + jj``), 16x16-shuffled."""

    rows = torch.arange(up.shape[0], device=up.device)
    block, column, branch = rows // 16, rows % 16 // config.hc_count, rows % config.hc_count
    return shuffle_16x16(up[branch * config.hidden + 4 * block + column].contiguous())


def pack_layer(served: dict, kind: LayerKind, config: Qwen38Config) -> dict[str, torch.Tensor]:
    """MonoKernel-owned BF16 copies of one layer, 16x16-shuffled for the stage kernels.

    ``hc_*_dn`` stacks the low-rank down rows, the 4 inject rows and zero rows to 336;
    ``router_shared`` stacks the router rows and the shared-expert gate padded to 16 rows (fused path).
    """

    packed = {}
    for prefix in ("hc_attn", "hc_mlp"):
        down = torch.cat((served[f"{prefix}_down"], served[f"{prefix}_inject"]))
        packed[f"{prefix}_dn"] = shuffle_16x16(_pad_rows(down))
        packed[f"{prefix}_up"] = pack_hc_up(served[f"{prefix}_up"], config)
    packed["router"] = shuffle_16x16(served["router"])
    packed["shared_gate"] = shuffle_16x16(_pad_rows(served["shared_gate"]))
    packed["router_shared"] = torch.cat((packed["router"], packed["shared_gate"]))
    if kind is LayerKind.LINEAR:
        packed["gdn_ba"] = shuffle_16x16(_pad_rows(served["gdn_ba"]))
    else:
        packed["index_qk"] = shuffle_16x16(served["index_qk"])
    return packed


def tensor_getter(tensors: dict[str, torch.Tensor]) -> Getter:
    def get(name: str, dim: int | None = None, start: int = 0, stop: int | None = None) -> torch.Tensor:
        tensor = tensors[name]
        return tensor if dim is None else tensor.narrow(dim, start, stop - start)

    return get


def synthetic_tensors(
    config: Qwen38Config,
    kind: LayerKind,
    *,
    seed: int = 1234,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """Checkpoint-named TP1 tensors with checkpoint dtypes, deterministic per name."""

    tensors: dict[str, torch.Tensor] = {}
    hidden, width = config.hidden, config.hc_width

    def generator(name: str) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))

    def bf16(name: str, *shape: int, std: float, mean: float = 0.0) -> None:
        value = torch.randn(*shape, generator=generator(name), device=device) * std + mean
        tensors[name] = value.to(torch.bfloat16)

    def fp8(name: str, rows: int, cols: int) -> None:
        value = torch.randn(rows, cols, generator=generator(name), device=device) / cols**0.5
        scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-12) / torch.finfo(CHECKPOINT_FP8).max
        tensors[f"{name}.weight"] = (value / scale).to(CHECKPOINT_FP8)
        tensors[f"{name}.weight_scale"] = scale.to(torch.bfloat16)

    for module in ("attn_hyper_connection", "mlp_hyper_connection"):
        bf16(f"{module}.hc_norm.weight", width, std=0.1)
        bf16(f"{module}.input_mix_weight_down.weight", config.hc_lowrank, width, std=width**-0.5)
        bf16(f"{module}.input_mix_weight_up.weight", width, config.hc_lowrank, std=config.hc_lowrank**-0.5)
        bf16(f"{module}.block_inject_weight.weight", config.hc_count, width, std=width**-0.5)
    if kind is LayerKind.LINEAR:
        key_dim = config.gdn_k_heads * config.gdn_head_dim
        value_dim = config.gdn_v_heads * config.gdn_head_dim
        fp8("linear_attn.in_proj_qkv", 2 * key_dim + value_dim, hidden)
        fp8("linear_attn.in_proj_z", value_dim, hidden)
        bf16("linear_attn.in_proj_a.weight", config.gdn_v_heads, hidden, std=hidden**-0.5)
        bf16("linear_attn.in_proj_b.weight", config.gdn_v_heads, hidden, std=hidden**-0.5)
        bf16("linear_attn.conv1d.weight", 2 * key_dim + value_dim, 1, config.conv_kernel, std=0.5)
        decay = torch.rand(config.gdn_v_heads, generator=generator("linear_attn.A_log"), device=device) * 15 + 1
        tensors["linear_attn.A_log"] = decay.log().to(torch.bfloat16)
        bf16("linear_attn.dt_bias", config.gdn_v_heads, std=0.5)
        bf16("linear_attn.norm.weight", config.gdn_head_dim, std=0.1, mean=1.0)
        fp8("linear_attn.out_proj", hidden, value_dim)
    else:
        fp8("self_attn.q_proj", config.heads * 2 * config.head_dim, hidden)
        fp8("self_attn.k_proj", config.kv_heads * config.head_dim, hidden)
        fp8("self_attn.v_proj", config.kv_heads * config.head_dim, hidden)
        bf16("self_attn.q_norm.weight", config.head_dim, std=0.1)
        bf16("self_attn.k_norm.weight", config.head_dim, std=0.1)
        fp8("self_attn.o_proj", hidden, config.heads * config.head_dim)
        index_rows = (config.index_heads + config.index_kv_heads) * config.index_head_dim
        bf16("self_attn.indexer.index_qk_proj.weight", index_rows, hidden, std=hidden**-0.5)
        bf16("self_attn.indexer.q_layernorm.weight", config.index_head_dim, std=0.1)
        bf16("self_attn.indexer.k_layernorm.weight", config.index_head_dim, std=0.1)
    bf16("mlp.gate.weight", config.n_experts, hidden, std=hidden**-0.5)
    bf16("mlp.shared_expert_gate.weight", 1, hidden, std=hidden**-0.5)
    for expert in range(config.n_experts + 1):
        prefix = _expert_prefix(config, expert)
        fp8(f"{prefix}.gate_proj", config.inter, hidden)
        fp8(f"{prefix}.up_proj", config.inter, hidden)
        fp8(f"{prefix}.down_proj", hidden, config.inter)
    return tensors


def make_weights(
    shard: Qwen38Shard,
    kind: LayerKind,
    *,
    seed: int = 1234,
    device: torch.device | str = "cuda",
) -> dict:
    """One rank's synthetic layer weights; every rank slices the same TP1 tensors."""

    tensors = synthetic_tensors(shard.config, kind, seed=seed, device=device)
    return shard_layer_weights(tensor_getter(tensors), shard, kind)


class CheckpointLayer:
    """Getter over one decoder layer of a safetensors checkpoint directory."""

    def __init__(self, model_dir: str | Path, layer_id: int, device: torch.device | str = "cuda"):
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.prefix = LAYER_PREFIX.format(layer_id)
        self.device = torch.device(device)
        self._handles = {}

    def __call__(self, name: str, dim: int | None = None, start: int = 0, stop: int | None = None) -> torch.Tensor:
        from safetensors import safe_open

        full_name = self.prefix + name
        file_name = self.weight_map[full_name]
        handle = self._handles.get(file_name)
        if handle is None:
            handle = safe_open(str(self.model_dir / file_name), framework="pt", device="cpu")
            self._handles[file_name] = handle
        if dim is None:
            tensor = handle.get_tensor(full_name)
        else:
            view = handle.get_slice(full_name)
            index = [slice(None)] * len(view.get_shape())
            index[dim] = slice(start, stop)
            tensor = view[tuple(index)]
        return tensor.to(self.device)


def load_layer_weights(
    model_dir: str | Path,
    layer_id: int,
    shard: Qwen38Shard,
    *,
    device: torch.device | str = "cuda",
    experts: Iterable[int] | None = None,
) -> dict:
    """One rank's weights for checkpoint layer ``layer_id``, sharded like SGLang."""

    kind = shard.config.layer_kind(layer_id)
    return shard_layer_weights(CheckpointLayer(model_dir, layer_id, device), shard, kind, experts)
