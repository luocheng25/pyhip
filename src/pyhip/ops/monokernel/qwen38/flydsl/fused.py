"""Fused GDN decode layer: each half of one rank's layer is one persistent launch.

The attention half runs [combine with the previous layer's reduced MLP output +] the attention
branch norm, HC down, low-rank, HC up mix, quant x, ``in_proj_qkvz`` + conv with ``in_proj_ba``, the
recurrence, quant y and ``out_proj``; the MLP half runs the combine with the reduced mixer output +
the MLP branch norm, HC down, low-rank, HC up mix, router + shared gate, top-k, the routing plan,
MoE gate/up and MoE down. Phases (``phases.py``) are separated by grid barriers (``sync.py``) and
reproduce the stage kernels bit for bit; ``phases=(i,)`` launches a single phase for debugging.

Kernel arguments are device addresses: weights and pools in place, and one scratch buffer for the
intermediates at offsets fixed by ``scratch_layout(shard, capacity)``.
"""

from __future__ import annotations

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm

from ..config import QWEN38_FLASH_NEXT, Qwen38Shard
from ..weights import moe_inter_padded
from . import attention as att
from . import phases as ph
from ._ir import store, words
from ._phase import THREADS, Ctx, buffer_at
from .linear import _call, tiles_for
from .sync import STRIPE_WORDS, _if, grid_arrive, grid_exit, grid_wait

C, H, WIDTH, LOWRANK = ph.C, ph.H, ph.WIDTH, ph.LOWRANK
MIX_PHASES = ("norm", "hc_down", "hc_lowrank", "hc_mix_up")
ATTENTION_PHASES = MIX_PHASES + ("quant_ba", "qkvz_conv", "recurrent", "quant_y", "out_proj")
MLP_PHASES = MIX_PHASES + ("router", "topk", "plan", "moe_up", "moe_down")
# Full-attention mixer: ``prep`` before SGLang's QSA indexer (which reads x), ``core`` after it.
PREP_PHASES = MIX_PHASES + ("quant_x", "qkv", "qk_prep")
CORE_PHASES = ("scores", "values", "combine_gate", "quant_y", "o_proj")
SHARED_LOGITS = 16
# One resident CTA of 8 waves per CU: 2 waves per SIMD may use 256 registers each.
HINTS = {"waves_per_eu": 2}


def ctas_for(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _attention_records(capacity: int) -> int:
    """Most (row, segment) partials of the sparse attention over the row buckets up to ``capacity``."""
    return max(min(capacity, 16 * tiles) * att.segments(tiles) for tiles in (1, 2, 4))


def scratch_layout(dims: tuple, capacity: int) -> tuple[dict, int]:
    """Byte offset and size of every intermediate for ``capacity`` rows (``dims`` from ``shard_dims``)."""
    conv_dim, value_dim, v_heads, inter, n_experts, slots, heads = dims
    records = _attention_records(capacity)
    y_dim = max(value_dim, heads * att.DA)
    sizes = dict(
        hn=capacity * WIDTH * 2,
        partial=ph.KSPLIT * capacity * ph.DOWN_ROWS * 4,
        low=capacity * LOWRANK * 2,
        x=capacity * H * 2,
        xq=capacity * H,
        xs=capacity * 4,
        qkvz=capacity * (conv_dim + value_dim) * 2,
        conv=capacity * conv_dim * 2,
        ba=capacity * 2 * v_heads * 2,
        core=capacity * value_dim * 2,
        y=capacity * y_dim * 2,
        yq=capacity * y_dim,
        ys=capacity * 4,
        qkv=capacity * att.qkv_columns(heads) * 2,
        q=capacity * heads * att.DA * 2,
        scores=capacity * att.CHUNKS * att.CHUNK * 16 * 4,
        chunk_max=capacity * att.CHUNKS * 16 * 4,
        part_ml=records * 16 * 8,
        part_acc=records * heads * att.DA * 4,
        logits=capacity * n_experts * 2,
        shared=capacity * SHARED_LOGITS * 4,
        ids=capacity * slots * 4,
        weights=capacity * slots * 4,
        plan=(1 + 3 * (n_experts + 1)) * 4,
        route_row=capacity * slots * 4,
        route_weight=capacity * slots * 4,
        mid=capacity * slots * inter * 2,
        sync=2 * STRIPE_WORDS * 4,
    )
    layout, at = {}, 0
    for name, size in sizes.items():
        layout[name] = at
        at += -(-size // 256) * 256
    return layout, at


def shard_dims(shard: Qwen38Shard) -> tuple:
    c = shard.config
    return (shard.conv_dim, shard.gdn_value_dim, shard.gdn_v_heads, moe_inter_padded(shard), c.n_experts,
            c.top_k + 1, shard.heads)


def _stamp(ctx, timeline, slot, columns):
    """Thread 0 writes the 100 MHz realtime counter to ``timeline[cta, slot]`` (int64)."""

    def write():
        now = fx.Int64(llvm.call_intrinsic(fx.Int64.ir_type, "llvm.amdgcn.s.memrealtime", [], [], []))
        store(buffer_at(timeline, ctx.ctas * columns * 8), (ctx.cta * columns + slot) * 8,
              words([fx.Int32(now), fx.Int32(now >> 32)]))

    _if(ctx.tid == 0, write)


def _run(ctx, steps, selected, sync_address, ctas, timeline=None):
    """Run the selected ``(prefetch, run)`` phases with grid barriers between them: ``prefetch`` (weights
    of the phase's first round) is issued between the barrier's arrive and wait and handed to ``run``.
    With ``timeline``, slot ``2 k`` is the start of phase ``k`` and ``2 k + 1`` the end of its work."""
    chosen = [steps[i] for i in selected]
    columns = 2 * len(chosen)
    for i, (prefetch, run) in enumerate(chosen):
        if i:
            grid_arrive(sync_address, ctx.tid)
        pre = prefetch() if prefetch is not None else None
        if i:
            grid_wait(sync_address, i - 1, ctas, ctx.tid)
        if timeline is not None:
            _stamp(ctx, timeline, 2 * i, columns)
        run(pre)
        if timeline is not None:
            _stamp(ctx, timeline, 2 * i + 1, columns)
    if len(chosen) > 1:
        grid_exit(sync_address, sync_address + STRIPE_WORDS * 4, ctas, ctx.tid)


def _mix_steps(ctx, at, rows, h_in, y_in, gain_in, norm_w, h_out, gain_out, hc_dn, hc_up, x, combine, eps, tiles,
               ctas):
    """The ``MIX_PHASES`` steps: [combine +] branch norm into ``hn``, HC down, low-rank (gains to ``gain_out``)
    and HC up mix into ``x``."""
    hn = at("hn", rows * WIDTH * 2)
    partial = at("partial", ph.KSPLIT * rows * ph.DOWN_ROWS * 4)
    low = at("low", rows * LOWRANK * 2)
    w_dn = buffer_at(hc_dn, ph.DOWN_ROWS * WIDTH * 2)
    w_up = buffer_at(hc_up, WIDTH * LOWRANK * 2)
    return [
        (None, lambda pre: ph.combine_norm(ctx, buffer_at(h_in, rows * WIDTH * 2), buffer_at(y_in, rows * H * 2),
                                           buffer_at(gain_in, rows * C * 4), buffer_at(norm_w, WIDTH * 2),
                                           buffer_at(h_out, rows * WIDTH * 2), hn, combine, eps)),
        (lambda: ph.hc_down_gemv(ctx, hn, w_dn, tiles, ctas).prefetch(),
         lambda pre: ph.hc_down(ctx, hn, w_dn, partial, tiles, ctas, pre)),
        (None, lambda pre: ph.hc_lowrank(ctx, partial, low, buffer_at(gain_out, rows * C * 4))),
        (lambda: ph.hc_mix_up_gemv(ctx, low, w_up, tiles, ctas).prefetch(),
         lambda pre: ph.hc_mix_up(ctx, low, w_up, hn, x, tiles, ctas, pre)),
    ]


@cache
def make_mix(dims, tiles, capacity, combine, eps, ctas, selected, timed=False):
    """``MIX_PHASES`` alone: the attention-branch mix of a full-attention layer into ``X``."""
    layout, _ = scratch_layout(dims, capacity)
    fused = len(selected) > 1

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def mix(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
            GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, X: fx.Int64, TIMELINE: fx.Int64,
            rows: fx.Int32):
        ctx = Ctx(rows, fused)

        def at(name, num_bytes):
            return buffer_at(SCRATCH + layout[name], num_bytes)

        steps = _mix_steps(ctx, at, rows, H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, HC_DN, HC_UP,
                           buffer_at(X, rows * H * 2), combine, eps, tiles, ctas)
        _run(ctx, steps, selected, SCRATCH + layout["sync"], ctas, TIMELINE if timed else None)

    @flyc.jit
    def launch(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
               GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, X: fx.Int64,
               TIMELINE: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        mix(H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, SCRATCH, HC_DN, HC_UP, X, TIMELINE,
            rows).launch(grid=(ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


@cache
def make_attn_prep(dims, tiles, capacity, combine, eps, ctas, selected, timed=False):
    """``PREP_PHASES``: the attention-branch mix into ``X`` (read by SGLang's indexer), the per-token FP8
    quantization of x, ``qkv_proj``, and the q/k norm + MRoPE with the KV store (``attention.qk_prep``)."""
    heads = dims[6]
    layout, _ = scratch_layout(dims, capacity)
    fused = len(selected) > 1
    n_qkv = att.qkv_columns(heads)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def prep(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
             GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, X: fx.Int64, W_QKV: fx.Int64,
             S_QKV: fx.Int64, Q_NORM: fx.Int64, K_NORM: fx.Int64, TABLE: fx.Int64, POSITIONS: fx.Int64,
             CACHE_LOC: fx.Int64, K_CACHE: fx.Int64, V_CACHE: fx.Int64, TIMELINE: fx.Int64, rows: fx.Int32,
             pos_stride: fx.Int32, table_rows: fx.Int32):
        ctx = Ctx(rows, fused)

        def at(name, num_bytes):
            return buffer_at(SCRATCH + layout[name], num_bytes)

        x = buffer_at(X, rows * H * 2)
        xq, xs = at("xq", rows * H), at("xs", rows * 4)
        qkv = at("qkv", rows * n_qkv * 2)
        w_qkv = buffer_at(W_QKV, n_qkv * H)
        steps = _mix_steps(ctx, at, rows, H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, HC_DN, HC_UP, x, combine,
                           eps, tiles, ctas) + [
            (None, lambda pre: ph.quant(ctx, x, xq, xs, H)),
            (lambda: att.qkv_gemv(ctx, xq, w_qkv, heads, tiles, ctas).prefetch(),
             lambda pre: att.qkv(ctx, xq, xs, w_qkv, buffer_at(S_QKV, n_qkv * 4), qkv, heads, tiles, ctas, pre)),
            (None, lambda pre: att.qk_prep(ctx, qkv, at("q", rows * heads * att.DA * 2),
                                           buffer_at(Q_NORM, att.DA * 2), buffer_at(K_NORM, att.DA * 2),
                                           buffer_at(TABLE, table_rows * att.ROT * 2),
                                           buffer_at(POSITIONS, (2 * pos_stride + rows) * 8), pos_stride,
                                           buffer_at(CACHE_LOC, rows * 8), K_CACHE, V_CACHE, heads, eps)),
        ]
        _run(ctx, steps, selected, SCRATCH + layout["sync"], ctas, TIMELINE if timed else None)

    @flyc.jit
    def launch(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
               GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, X: fx.Int64,
               W_QKV: fx.Int64, S_QKV: fx.Int64, Q_NORM: fx.Int64, K_NORM: fx.Int64, TABLE: fx.Int64,
               POSITIONS: fx.Int64, CACHE_LOC: fx.Int64, K_CACHE: fx.Int64, V_CACHE: fx.Int64, TIMELINE: fx.Int64,
               rows: fx.Int32, pos_stride: fx.Int32, table_rows: fx.Int32, stream: fx.Stream):
        prep(H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, SCRATCH, HC_DN, HC_UP, X, W_QKV, S_QKV, Q_NORM, K_NORM,
             TABLE, POSITIONS, CACHE_LOC, K_CACHE, V_CACHE, TIMELINE, rows, pos_stride,
             table_rows).launch(grid=(ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


@cache
def make_attn_core(dims, tiles, capacity, ctas, selected, timed=False):
    """``CORE_PHASES``: sparse attention over the selected positions, segment merge + output gate, the
    per-token FP8 quantization and ``o_proj`` (this rank's partial into ``OUT``)."""
    heads = dims[6]
    layout, _ = scratch_layout(dims, capacity)
    fused = len(selected) > 1
    n_qkv = att.qkv_columns(heads)
    k = heads * att.DA
    records = _attention_records(capacity)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def core(SCRATCH: fx.Int64, TOPK: fx.Int64, REQ_TO_TOKEN: fx.Int64, REQ_POOL: fx.Int64, SEQ_LENS: fx.Int64,
             K_CACHE: fx.Int64, V_CACHE: fx.Int64, W_O: fx.Int64, S_O: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64,
             rows: fx.Int32, rtt_stride: fx.Int32, rtt_bytes: fx.Int32):
        ctx = Ctx(rows, fused)

        def at(name, num_bytes):
            return buffer_at(SCRATCH + layout[name], num_bytes)

        part_ml, part_acc = at("part_ml", records * 16 * 8), at("part_acc", records * heads * att.DA * 4)
        scores = at("scores", rows * att.CHUNKS * att.CHUNK * 16 * 4)
        chunk_max = at("chunk_max", rows * att.CHUNKS * 16 * 4)
        topk = buffer_at(TOPK, rows * att.TOPK * 4)
        req_to_token = buffer_at(REQ_TO_TOKEN, rtt_bytes)
        req_pool, seq_lens = buffer_at(REQ_POOL, rows * 8), buffer_at(SEQ_LENS, rows * 8)
        qkv = at("qkv", rows * n_qkv * 2)
        y, yq, ys = at("y", rows * k * 2), at("yq", rows * k), at("ys", rows * 4)
        w_o = buffer_at(W_O, H * k)
        steps = [
            (None, lambda pre: att.attention_scores(ctx, at("q", rows * heads * att.DA * 2), topk, req_to_token,
                                                    req_pool, seq_lens, K_CACHE, scores, chunk_max, heads,
                                                    rtt_stride)),
            (None, lambda pre: att.attention_values(ctx, scores, chunk_max, topk, req_to_token, req_pool, seq_lens,
                                                    V_CACHE, part_ml, part_acc, heads, tiles, rtt_stride)),
            (None, lambda pre: att.combine_gate(ctx, part_ml, part_acc, qkv, y, heads, tiles)),
            (None, lambda pre: ph.quant(ctx, y, yq, ys, k)),
            (lambda: ph.out_proj_gemv(ctx, yq, w_o, k, tiles, ctas).prefetch(),
             lambda pre: ph.out_proj(ctx, yq, ys, w_o, buffer_at(S_O, H * 4), buffer_at(OUT, rows * H * 2), k,
                                     tiles, ctas, pre)),
        ]
        _run(ctx, steps, selected, SCRATCH + layout["sync"], ctas, TIMELINE if timed else None)

    @flyc.jit
    def launch(SCRATCH: fx.Int64, TOPK: fx.Int64, REQ_TO_TOKEN: fx.Int64, REQ_POOL: fx.Int64, SEQ_LENS: fx.Int64,
               K_CACHE: fx.Int64, V_CACHE: fx.Int64, W_O: fx.Int64, S_O: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64,
               rows: fx.Int32, rtt_stride: fx.Int32, rtt_bytes: fx.Int32, stream: fx.Stream):
        core(SCRATCH, TOPK, REQ_TO_TOKEN, REQ_POOL, SEQ_LENS, K_CACHE, V_CACHE, W_O, S_O, OUT, TIMELINE, rows,
             rtt_stride, rtt_bytes).launch(grid=(ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


@cache
def make_attention(dims, k_heads, ba_rows, tiles, capacity, combine, eps, ctas, selected, timed=False):
    conv_dim, value_dim, v_heads, _, _, _, _ = dims
    layout, _ = scratch_layout(dims, capacity)
    n_qkvz = conv_dim + value_dim
    fused = len(selected) > 1

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def attention(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
                  GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, W_QKVZ: fx.Int64,
                  S_QKVZ: fx.Int64, CONV_W: fx.Int64, SLOTS: fx.Int64, CONV_STATE: fx.Int64, W_BA: fx.Int64,
                  A_LOG: fx.Int64, DT_BIAS: fx.Int64, GDN_NORM: fx.Int64, SSM: fx.Int64, W_OUT: fx.Int64,
                  S_OUT: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64, rows: fx.Int32, conv_state_bytes: fx.Int64):
        ctx = Ctx(rows, fused)

        def at(name, num_bytes):
            return buffer_at(SCRATCH + layout[name], num_bytes)

        x, xq, xs = at("x", rows * H * 2), at("xq", rows * H), at("xs", rows * 4)
        qkvz, conv = at("qkvz", rows * n_qkvz * 2), at("conv", rows * conv_dim * 2)
        ba = at("ba", rows * 2 * v_heads * 2)
        core, y = at("core", rows * value_dim * 2), at("y", rows * value_dim * 2)
        yq, ys = at("yq", rows * value_dim), at("ys", rows * 4)
        slots = buffer_at(SLOTS, rows * 4)
        w_ba = buffer_at(W_BA, ba_rows * H * 2)
        w_qkvz = buffer_at(W_QKVZ, n_qkvz * H)
        w_out = buffer_at(W_OUT, H * value_dim)
        steps = _mix_steps(ctx, at, rows, H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, HC_DN, HC_UP, x, combine,
                           eps, tiles, ctas) + [
            (lambda: ph.ba_gemv(ctx, x, w_ba, ba_rows // 16, tiles).prefetch(),
             lambda pre: ph.quant_ba(ctx, x, xq, xs, w_ba, ba, ba_rows // 16, 2 * v_heads, tiles, ctas, pre)),
            (lambda: ph.qkvz_gemv(ctx, xq, w_qkvz, n_qkvz, tiles, ctas).prefetch(),
             lambda pre: ph.qkvz_conv(ctx, xq, xs, w_qkvz, buffer_at(S_QKVZ, n_qkvz * 4),
                                      buffer_at(CONV_W, conv_dim * 4 * 2), slots,
                                      buffer_at(CONV_STATE, conv_state_bytes), qkvz, conv, n_qkvz, conv_dim, tiles,
                                      ctas, pre)),
            (lambda: ph.recurrent_prefetch(ctx, slots, SSM, v_heads),
             lambda pre: ph.recurrent(ctx, conv, qkvz, ba, buffer_at(A_LOG, v_heads * 4),
                                      buffer_at(DT_BIAS, v_heads * 2), buffer_at(GDN_NORM, ph.D * 2), SSM, slots,
                                      core, y, v_heads, k_heads, conv_dim, eps, pre)),
            (None, lambda pre: ph.quant(ctx, y, yq, ys, value_dim)),
            (lambda: ph.out_proj_gemv(ctx, yq, w_out, value_dim, tiles, ctas).prefetch(),
             lambda pre: ph.out_proj(ctx, yq, ys, w_out, buffer_at(S_OUT, H * 4), buffer_at(OUT, rows * H * 2),
                                     value_dim, tiles, ctas, pre)),
        ]
        _run(ctx, steps, selected, SCRATCH + layout["sync"], ctas, TIMELINE if timed else None)

    @flyc.jit
    def launch(H_IN: fx.Int64, Y_IN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
               GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, W_QKVZ: fx.Int64,
               S_QKVZ: fx.Int64, CONV_W: fx.Int64, SLOTS: fx.Int64, CONV_STATE: fx.Int64, W_BA: fx.Int64,
               A_LOG: fx.Int64, DT_BIAS: fx.Int64, GDN_NORM: fx.Int64, SSM: fx.Int64, W_OUT: fx.Int64,
               S_OUT: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64, rows: fx.Int32, conv_state_bytes: fx.Int64,
               stream: fx.Stream):
        attention(H_IN, Y_IN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, SCRATCH, HC_DN, HC_UP, W_QKVZ, S_QKVZ, CONV_W,
                  SLOTS, CONV_STATE, W_BA, A_LOG, DT_BIAS, GDN_NORM, SSM, W_OUT, S_OUT, OUT, TIMELINE, rows,
                  conv_state_bytes).launch(grid=(ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


@cache
def make_mlp(dims, tiles, capacity, eps, ctas, selected, timed=False):
    _, _, _, inter, n_experts, slots, _ = dims
    top_k = slots - 1
    layout, _ = scratch_layout(dims, capacity)
    fused = len(selected) > 1

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def mlp(H_IN: fx.Int64, ATTN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
            GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, W_ROUTER: fx.Int64,
            W13: fx.Int64, S13: fx.Int64, W2: fx.Int64, S2: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64,
            rows: fx.Int32):
        ctx = Ctx(rows, fused)

        def at(name, num_bytes):
            return buffer_at(SCRATCH + layout[name], num_bytes)

        routes = rows * slots
        x = at("x", rows * H * 2)
        logits, shared = at("logits", rows * n_experts * 2), at("shared", rows * SHARED_LOGITS * 4)
        ids, weights = at("ids", routes * 4), at("weights", routes * 4)
        plan = at("plan", (1 + 3 * (n_experts + 1)) * 4)
        route_row, route_weight = at("route_row", routes * 4), at("route_weight", routes * 4)
        mid = at("mid", routes * inter * 2)
        w_router = buffer_at(W_ROUTER, (n_experts + 16) * H * 2)
        steps = _mix_steps(ctx, at, rows, H_IN, ATTN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, HC_DN, HC_UP, x, True, eps,
                           tiles, ctas) + [
            (lambda: ph.router_gemv(ctx, x, w_router, n_experts, tiles, ctas).prefetch(),
             lambda pre: ph.router(ctx, x, w_router, logits, shared, n_experts, tiles, ctas, pre)),
            (None, lambda pre: ph.topk_rows(ctx, logits, shared, ids, weights, n_experts, top_k)),
            (None, lambda pre: ph.plan(ctx, ids, weights, plan, route_row, route_weight, n_experts, top_k)),
            (None, lambda pre: ph.moe_up(ctx, x, W13, S13, plan, route_row, mid, n_experts, top_k, inter, tiles)),
            (None, lambda pre: ph.moe_down(ctx, mid, W2, S2, plan, route_row, route_weight,
                                           buffer_at(OUT, rows * H * 2), n_experts, top_k, inter, tiles)),
        ]
        _run(ctx, steps, selected, SCRATCH + layout["sync"], ctas, TIMELINE if timed else None)

    @flyc.jit
    def launch(H_IN: fx.Int64, ATTN: fx.Int64, GAIN_IN: fx.Int64, NORM_W: fx.Int64, H_OUT: fx.Int64,
               GAIN_OUT: fx.Int64, SCRATCH: fx.Int64, HC_DN: fx.Int64, HC_UP: fx.Int64, W_ROUTER: fx.Int64,
               W13: fx.Int64, S13: fx.Int64, W2: fx.Int64, S2: fx.Int64, OUT: fx.Int64, TIMELINE: fx.Int64,
               rows: fx.Int32, stream: fx.Stream):
        mlp(H_IN, ATTN, GAIN_IN, NORM_W, H_OUT, GAIN_OUT, SCRATCH, HC_DN, HC_UP, W_ROUTER, W13, S13, W2, S2, OUT,
            TIMELINE, rows).launch(grid=(ctas, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------- host


class FusedWorkspace:
    """Scratch for ``capacity`` rows; ``view(name, rows, ...)`` exposes an intermediate."""

    def __init__(self, capacity: int, shard: Qwen38Shard, device: torch.device):
        self.capacity, self.shard, self.device = capacity, shard, device
        self.dims = shard_dims(shard)
        self.layout, size = scratch_layout(self.dims, capacity)
        self.scratch = torch.zeros(size, dtype=torch.uint8, device=device)

    def view(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        count = 1
        for n in shape:
            count *= n
        start = self.layout[name]
        return self.scratch[start:start + count * dtype.itemsize].view(dtype).view(shape)


class FusedLayer:
    """Launches shared by both layer kinds of one rank over SGLang's served and MonoKernel-packed tensors:
    the attention-branch HC mix alone (``mix``) and the MLP half (``mlp``)."""

    def __init__(self, shard: Qwen38Shard, served: dict, packed: dict, eps: float = QWEN38_FLASH_NEXT.eps):
        self.shard, self.served, self.packed, self.eps = shard, served, packed, float(eps)
        self.ctas = ctas_for(served["hc_attn_norm"].device)

    def mix(self, h, x, work, gain_out, combine=None, h_out=None, selected=None, timeline=None):
        """``x`` [rows, hidden] = the attention-branch HC mix of ``h`` (``combine`` None) or of
        ``h_out = bf16(h + y * gain)`` with ``combine = (y, gain)``; ``gain_out`` [rows, 4] FP32 receives
        the attention-branch combine gains."""
        w, p = self.served, self.packed
        rows = h.shape[0]
        tiles = tiles_for(rows)
        selected = tuple(range(len(MIX_PHASES))) if selected is None else tuple(selected)
        y_in, gain_in = combine if combine is not None else (h, h)
        h_out = h if h_out is None else h_out
        ptr = lambda t: t.data_ptr()  # noqa: E731
        stream = torch.cuda.current_stream(h.device)
        args = (ptr(h), ptr(y_in), ptr(gain_in), ptr(w["hc_attn_norm"]), ptr(h_out), ptr(gain_out),
                ptr(work.scratch), ptr(p["hc_attn_dn"]), ptr(p["hc_attn_up"]), ptr(x),
                0 if timeline is None else ptr(timeline), rows, stream)
        timed = timeline is not None
        key = ("fused_mix", h.device, work.dims, tiles, work.capacity, combine is not None, self.eps, self.ctas,
               selected, timed)
        builder = make_mix(work.dims, tiles, work.capacity, combine is not None, self.eps, self.ctas, selected,
                           timed)
        with torch.cuda.device(h.device):
            _call(key, builder, args, HINTS)

    def mlp(self, h, attn, gain_attn, h1, gain_out, out, work, selected=None, timeline=None):
        """``h1 = bf16(h + attn * gain_attn)``; ``out`` = this rank's MoE partial; ``gain_out`` the
        MLP-branch combine gains."""
        w, p = self.served, self.packed
        rows = h.shape[0]
        tiles = tiles_for(rows)
        selected = tuple(range(len(MLP_PHASES))) if selected is None else tuple(selected)
        ptr = lambda t: t.data_ptr()  # noqa: E731
        stream = torch.cuda.current_stream(h.device)
        args = (ptr(h), ptr(attn), ptr(gain_attn), ptr(w["hc_mlp_norm"]), ptr(h1), ptr(gain_out), ptr(work.scratch),
                ptr(p["hc_mlp_dn"]), ptr(p["hc_mlp_up"]), ptr(p["router_shared"]), ptr(w["moe_w13"]),
                ptr(w["moe_w13_scale"]), ptr(w["moe_w2"]), ptr(w["moe_w2_scale"]), ptr(out),
                0 if timeline is None else ptr(timeline), rows, stream)
        timed = timeline is not None
        key = ("fused_mlp", h.device, work.dims, tiles, work.capacity, self.eps, self.ctas, selected, timed)
        builder = make_mlp(work.dims, tiles, work.capacity, self.eps, self.ctas, selected, timed)
        with torch.cuda.device(h.device):
            _call(key, builder, args, HINTS)


class FusedGdnLayer(FusedLayer):
    """One rank's GDN layer as two persistent launches: ``attention`` and ``mlp``."""

    def attention(self, h, slots, conv_state, ssm_state, out, work, combine=None, h_out=None, gain_out=None,
                  selected=None, timeline=None):
        """``out`` = this rank's ``out_proj`` partial for the layer input ``h`` (``combine`` None) or for
        ``h_out = bf16(h + y * gain)`` with ``combine = (y, gain)``; ``gain_out`` [rows, 4] FP32 receives
        the attention-branch combine gains. ``timeline``: int64 [ctas, 2 * phases] stamps (see ``_run``)."""
        s, w, p = self.shard, self.served, self.packed
        rows = h.shape[0]
        tiles = tiles_for(rows)
        selected = tuple(range(len(ATTENTION_PHASES))) if selected is None else tuple(selected)
        y_in, gain_in = combine if combine is not None else (h, h)
        h_out = h if h_out is None else h_out
        ptr = lambda t: t.data_ptr()  # noqa: E731
        stream = torch.cuda.current_stream(h.device)
        args = (ptr(h), ptr(y_in), ptr(gain_in), ptr(w["hc_attn_norm"]), ptr(h_out), ptr(gain_out),
                ptr(work.scratch), ptr(p["hc_attn_dn"]), ptr(p["hc_attn_up"]), ptr(w["gdn_qkvz"]),
                ptr(w["gdn_qkvz_scale"]), ptr(w["gdn_conv"]), ptr(slots), ptr(conv_state), ptr(p["gdn_ba"]),
                ptr(w["gdn_a_log"]), ptr(w["gdn_dt_bias"]), ptr(w["gdn_norm"]), ptr(ssm_state), ptr(w["gdn_out"]),
                ptr(w["gdn_out_scale"]), ptr(out), 0 if timeline is None else ptr(timeline), rows,
                conv_state.numel() * 2, stream)
        timed = timeline is not None
        key = ("fused_attention", h.device, work.dims, s.gdn_k_heads, p["gdn_ba"].shape[0], tiles, work.capacity,
               combine is not None, self.eps, self.ctas, selected, timed)
        builder = make_attention(work.dims, s.gdn_k_heads, p["gdn_ba"].shape[0], tiles, work.capacity,
                                 combine is not None, self.eps, self.ctas, selected, timed)
        with torch.cuda.device(h.device):
            _call(key, builder, args, HINTS)


class FusedAttentionLayer(FusedLayer):
    """One rank's full-attention layer: ``prep`` (attention-branch mix .. q/k norm, MRoPE and KV store), the
    caller's QSA indexer on ``x``, ``core`` (sparse attention .. ``o_proj`` partial) and ``mlp``."""

    def prep(self, h, x, positions, cache_loc, k_cache, v_cache, table, work, gain_out, combine=None, h_out=None,
             selected=None, timeline=None):
        """``positions``: int64 [3, *] MRoPE (or [rows]) positions; ``cache_loc``: int64 [rows] KV slots;
        ``k_cache``/``v_cache``: BF16 [slots, 1, 256] of this layer; ``table``: SGLang's BF16 cos/sin cache
        [positions, 64]."""
        w, p = self.served, self.packed
        rows = h.shape[0]
        tiles = tiles_for(rows)
        selected = tuple(range(len(PREP_PHASES))) if selected is None else tuple(selected)
        y_in, gain_in = combine if combine is not None else (h, h)
        h_out = h if h_out is None else h_out
        pos_stride = positions.stride(0) if positions.dim() == 2 else 0
        ptr = lambda t: t.data_ptr()  # noqa: E731
        stream = torch.cuda.current_stream(h.device)
        args = (ptr(h), ptr(y_in), ptr(gain_in), ptr(w["hc_attn_norm"]), ptr(h_out), ptr(gain_out),
                ptr(work.scratch), ptr(p["hc_attn_dn"]), ptr(p["hc_attn_up"]), ptr(x), ptr(w["attn_qkv"]),
                ptr(w["attn_qkv_scale"]), ptr(w["attn_q_norm"]), ptr(w["attn_k_norm"]), ptr(table), ptr(positions),
                ptr(cache_loc), ptr(k_cache), ptr(v_cache), 0 if timeline is None else ptr(timeline), rows,
                pos_stride, table.shape[0], stream)
        timed = timeline is not None
        key = ("fused_attn_prep", h.device, work.dims, tiles, work.capacity, combine is not None, self.eps,
               self.ctas, selected, timed)
        builder = make_attn_prep(work.dims, tiles, work.capacity, combine is not None, self.eps, self.ctas,
                                 selected, timed)
        with torch.cuda.device(h.device):
            _call(key, builder, args, HINTS)

    def core(self, topk, req_to_token, req_pool, seq_lens, k_cache, v_cache, out, work, selected=None,
             timeline=None):
        """``topk``: int32 [rows, 2051] selected logical positions (-1 = none); ``req_to_token``: int32
        [requests, stride]; ``req_pool``/``seq_lens``: int64 [rows]; ``out``: BF16 [rows, hidden] partial."""
        w = self.served
        rows = topk.shape[0]
        tiles = tiles_for(rows)
        selected = tuple(range(len(CORE_PHASES))) if selected is None else tuple(selected)
        ptr = lambda t: t.data_ptr()  # noqa: E731
        stream = torch.cuda.current_stream(topk.device)
        args = (ptr(work.scratch), ptr(topk), ptr(req_to_token), ptr(req_pool), ptr(seq_lens), ptr(k_cache),
                ptr(v_cache), ptr(w["attn_o"]), ptr(w["attn_o_scale"]), ptr(out),
                0 if timeline is None else ptr(timeline), rows, req_to_token.stride(0),
                req_to_token.numel() * 4, stream)
        timed = timeline is not None
        key = ("fused_attn_core", topk.device, work.dims, tiles, work.capacity, self.ctas, selected, timed)
        builder = make_attn_core(work.dims, tiles, work.capacity, self.ctas, selected, timed)
        with torch.cuda.device(topk.device):
            _call(key, builder, args, HINTS)
