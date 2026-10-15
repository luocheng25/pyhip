"""Qwen3.8-Flash-Next decoder geometry and one tensor-parallel rank's shard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SUPPORTED_TP = (1, 2, 4, 8)


class LayerKind(str, Enum):
    """Token mixer of one decoder layer; both kinds end in the same MoE."""

    LINEAR = "linear_attention"
    FULL = "full_attention"


@dataclass(frozen=True)
class Qwen38Config:
    """Text-decoder geometry of ``Qwen/Qwen3.8-Flash-Next`` (``qwen4_exp``)."""

    hidden: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    eps: float = 1e-6
    num_layers: int = 48
    full_attention_interval: int = 4
    gdn_k_heads: int = 16
    gdn_v_heads: int = 48
    gdn_head_dim: int = 128
    conv_kernel: int = 4
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1.0e7
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_compress_ratio: int = 4
    index_topk: int = 2048
    n_experts: int = 512
    top_k: int = 10
    inter: int = 640

    @property
    def hc_width(self) -> int:
        return self.hc_count * self.hidden

    @property
    def shared_expert(self) -> int:
        """Expert id of the shared expert, fused after the routed experts."""

        return self.n_experts

    @property
    def moe_slots(self) -> int:
        return self.top_k + 1

    def layer_kind(self, layer_id: int) -> LayerKind:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"layer_id must be in [0, {self.num_layers}), got {layer_id}")
        return LayerKind.FULL if (layer_id + 1) % self.full_attention_interval == 0 else LayerKind.LINEAR


QWEN38_FLASH_NEXT = Qwen38Config()


@dataclass(frozen=True)
class Qwen38Shard:
    """Local sizes of one TP rank, matching SGLang's ``qwen4_exp`` sharding.

    GDN q/k/v/z, attention query heads, and expert intermediates are split
    contiguously; the two KV heads are replicated once TP exceeds two.
    """

    config: Qwen38Config
    tp: int
    rank: int

    def __post_init__(self) -> None:
        if self.tp not in SUPPORTED_TP:
            raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {self.tp}")
        if not 0 <= self.rank < self.tp:
            raise ValueError(f"rank must be in [0, {self.tp}), got {self.rank}")
        c = self.config
        for name, count in (
            ("gdn_k_heads", c.gdn_k_heads),
            ("gdn_v_heads", c.gdn_v_heads),
            ("heads", c.heads),
            ("inter", c.inter),
        ):
            if count % self.tp:
                raise ValueError(f"{name}={count} is not divisible by tp={self.tp}")
        if c.gdn_v_heads % c.gdn_k_heads or c.heads % c.kv_heads:
            raise ValueError("value/query heads must be a multiple of key heads")
        if max(self.tp, c.kv_heads) % min(self.tp, c.kv_heads):
            raise ValueError(f"tp={self.tp} cannot shard or replicate {c.kv_heads} KV heads")

    @property
    def gdn_k_heads(self) -> int:
        return self.config.gdn_k_heads // self.tp

    @property
    def gdn_v_heads(self) -> int:
        return self.config.gdn_v_heads // self.tp

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_k_heads * self.config.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_v_heads * self.config.gdn_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def heads(self) -> int:
        return self.config.heads // self.tp

    @property
    def kv_heads(self) -> int:
        return max(1, self.config.kv_heads // self.tp)

    @property
    def kv_head_start(self) -> int:
        return self.rank * self.config.kv_heads // self.tp

    @property
    def inter(self) -> int:
        return self.config.inter // self.tp


def shard(tp: int, rank: int, config: Qwen38Config = QWEN38_FLASH_NEXT) -> Qwen38Shard:
    return Qwen38Shard(config, tp, rank)
