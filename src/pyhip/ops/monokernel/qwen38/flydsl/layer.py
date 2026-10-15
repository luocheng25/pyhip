"""Staged (multi-launch) MonoKernel decode layer: the stage kernels of one TP rank in the layer's order.

The layer splits at its two all-reduces: ``attention_phase`` (HC mix of the attention branch, the
GDN mixer, its out projection partial), ``mlp_phase`` (combine with the reduced mixer output and
the MLP-branch norm, HC mix, routing, MoE partial) and ``finish`` (combine with the reduced MoE
output and the next layer's attention-branch norm). A serving rank calls them in sequence with its
collective; the capture check runs every rank's phase in one process and sums the partials.
"""

from __future__ import annotations

import torch

from ..config import LayerKind, Qwen38Shard
from ..reference import ACTIVATION_FP8
from ..weights import moe_inter_padded
from .gdn import gdn_recurrent, qkvz_conv
from .hc import HCWorkspace, hc_combine_norm, hc_mix
from .linear import linear_bf16, linear_fp8, quant_rows
from .moe import MoEWorkspace, moe_down, moe_up, topk_plan


class LayerWorkspace:
    """Intermediate buffers of one rank for ``rows`` tokens (every stage output stays readable)."""

    def __init__(self, rows: int, shard: Qwen38Shard, device: torch.device):
        c = shard.config
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.rows = rows
        self.x = torch.empty(rows, c.hidden, **bf16)
        self.xq = torch.empty(rows, c.hidden, dtype=ACTIVATION_FP8, device=device)
        self.xs = torch.empty(rows, dtype=torch.float32, device=device)
        self.hc_attn = HCWorkspace(rows, device)
        self.hc_mlp = HCWorkspace(rows, device)
        self.qkvz = torch.empty(rows, shard.conv_dim + shard.gdn_value_dim, **bf16)
        self.conv = torch.empty(rows, shard.conv_dim, **bf16)
        self.ba = torch.empty(rows, 2 * shard.gdn_v_heads, **bf16)
        self.core = torch.empty(rows, shard.gdn_v_heads, c.gdn_head_dim, **bf16)
        self.y = torch.empty(rows, shard.gdn_value_dim, **bf16)
        self.yq = torch.empty(rows, shard.gdn_value_dim, dtype=ACTIVATION_FP8, device=device)
        self.ys = torch.empty(rows, dtype=torch.float32, device=device)
        self.partial = torch.empty(rows, c.hidden, **bf16)
        self.h1 = torch.empty(rows, c.hc_width, **bf16)
        self.hn1 = torch.empty(rows, c.hc_width, **bf16)
        self.xm = torch.empty(rows, c.hidden, **bf16)
        self.moe = MoEWorkspace(rows, c.n_experts, c.top_k, moe_inter_padded(shard), device)
        self.partial_m = torch.empty(rows, c.hidden, **bf16)


class StagedLayer:
    """One rank's decode layer over SGLang's served tensors (``served``) and MonoKernel copies (``packed``)."""

    def __init__(self, kind: LayerKind, shard: Qwen38Shard, served: dict, packed: dict):
        if kind is not LayerKind.LINEAR:
            raise NotImplementedError("the staged full-attention layer is P2")
        self.kind, self.shard, self.served, self.packed = kind, shard, served, packed

    def attention_phase(self, hn: torch.Tensor, slots: torch.Tensor, conv_state: torch.Tensor,
                        ssm_state: torch.Tensor, work: LayerWorkspace) -> torch.Tensor:
        """hn: the attention-branch normalized hyper state [rows, 10240]; returns this rank's partial."""
        s, w, p = self.shard, self.served, self.packed
        hc_mix(hn, p["hc_attn_dn"], p["hc_attn_up"], work.x, work.hc_attn)
        quant_rows(work.x, work.xq, work.xs)
        qkvz_conv(work.xq, work.xs, w["gdn_qkvz"], w["gdn_qkvz_scale"], w["gdn_conv"], slots, conv_state, work.qkvz,
                  work.conv)
        linear_bf16(work.x, p["gdn_ba"], work.ba)
        gdn_recurrent(work.conv, work.qkvz, work.ba, w["gdn_a_log"], w["gdn_dt_bias"], w["gdn_norm"], ssm_state, slots,
                      work.core, work.y, s.gdn_v_heads, s.gdn_k_heads)
        quant_rows(work.y, work.yq, work.ys)
        linear_fp8(work.yq, work.ys, w["gdn_out"], w["gdn_out_scale"], work.partial)
        return work.partial

    def mlp_phase(self, h: torch.Tensor, attn: torch.Tensor, work: LayerWorkspace) -> torch.Tensor:
        """h: the layer input hyper state; attn: the reduced mixer output; returns the MoE partial."""
        w, p = self.served, self.packed
        rows = h.shape[0]
        hc_combine_norm(h, attn, work.hc_attn.gain, w["hc_mlp_norm"], work.h1, work.hn1)
        hc_mix(work.hn1, p["hc_mlp_dn"], p["hc_mlp_up"], work.xm, work.hc_mlp)
        linear_bf16(work.xm, p["router"], work.moe.logits)
        linear_bf16(work.xm, p["shared_gate"], work.moe.shared)
        topk_plan(work.moe, rows)
        moe_up(work.xm, w["moe_w13"], w["moe_w13_scale"], work.moe)
        moe_down(w["moe_w2"], w["moe_w2_scale"], work.moe, work.partial_m)
        return work.partial_m

    def finish(self, mlp: torch.Tensor, next_norm: torch.Tensor, out: torch.Tensor, out_normed: torch.Tensor,
               work: LayerWorkspace) -> None:
        """out = combined hyper state; out_normed = its attention-branch norm for the next layer."""
        hc_combine_norm(work.h1, mlp, work.hc_mlp.gain, next_norm, out, out_normed)
