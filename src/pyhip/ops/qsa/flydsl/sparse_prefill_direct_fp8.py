# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA direct attention over FP8 (E4M3FNUZ) K/V: one query wave over its own
selected blocks, FP8 MFMAs.

Each call packs every request's K/V into four-token blocks in FP8 MFMA operand order (with a
union plan only when K3 routes some tile to direct, direct_flag[0]), so the direct kernel's
loads are whole, coalesced operands (no LDS, no permutes):
  PK[block][i * 256 + (token * 4 + g) * 16 + 0..15]
      = K[token][64i + 8g + 0..7] ++ K[token][64i + 32 + 8g + 0..7]
  PV[block][p * 256 + d * 16 + (m - 4p) * 4 + token] = V[token][16d + m]   (p = m >> 2)
Per-token scales are stored per block as one 64-byte record of k_scale, log2(v_scale) and
1 / v_scale (clamped as in the
union kernel); tokens past a request's end read 0 (a slot table's later pool rows too).

A wave runs one query row and KV head: its G <= 16 Q heads are the columns of 16x16x32 FP8
MFMAs. Its key list (complete blocks in block_indices order, then the causal tail block)
runs in 32-key chunks of eight blocks: QK(kg) for blocks 8t + 4kg + jj; S lane (head, g)
holds blocks 8t + g and 8t + 4 + g, which are also its PV K-dim, so P needs no exchange.
PV MFMA m writes dims 16d + m: a lane (head, g) ends with dims 64g .. 64g + 63. Softmax:
the union kernel's lazy max (S'' = S c - m, a rescale only when some S'' passes LAZY or a
fresh row sees its first key, the row max then at S'' = P_BIAS), uniform per row across its
four lanes. Q comes from
sparse_prefill_union_fp8.quantize_q.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl._mlir import ir
from flydsl.expr import rocdl

from .sparse_prefill_common import (
    _MAX_FLOOR,
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _maximum,
    _min,
    _target_features,
    _uniform,
)
from .sparse_prefill_direct_packed import _enabled
from .sparse_prefill_prepare import allocate_scratch, upload
from .sparse_prefill_union_fp8 import (
    LAZY,
    P_BIAS,
    LOG2E,
    SCALE_MAX,
    SCALE_MIN,
    _as_f32,
    _as_i32,
    _cvt_fp8,
    _perm,
    _scaled_bf16,
)

__all__ = ["prepare", "scratch_specs", "run", "launcher"]

D = 256
BLOCK_BYTES = 1024  # one packed four-token block of one KV head
CHUNK = 8  # blocks per 32-key chunk


def _mfma16(a_words, b_words, acc):
    """16x16x32 FP8 MFMA: a, b hold 8 bytes per lane, acc FP32[4]."""
    a = fx.Vector.from_elements(list(a_words), fx.Int32).bitcast(fx.Int64)[0]
    b = fx.Vector.from_elements(list(b_words), fx.Int32).bitcast(fx.Int64)[0]
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8_(
            ir.VectorType.get([4], fx.Float32.ir_type),
            a.ir_value(),
            b.ir_value(),
            fx.Vector(acc).ir_value(),
            0,
            0,
            rocdl._blgp_attr(0),
        )
    )


def _transpose4(w):
    """w[t] holds bytes (d0..d3) of token t; returns o[d] holding bytes (t0..t3)."""
    x_lo = _perm(w[1], w[0], 0x05010400)
    x_hi = _perm(w[1], w[0], 0x07030602)
    y_lo = _perm(w[3], w[2], 0x05010400)
    y_hi = _perm(w[3], w[2], 0x07030602)
    return [
        _perm(y_lo, x_lo, 0x05040100),
        _perm(y_lo, x_lo, 0x07060302),
        _perm(y_hi, x_hi, 0x05040100),
        _perm(y_hi, x_hi, 0x07060302),
    ]


def _scale_terms(sv):
    sv = fx.minnumf(
        fx.maxnumf(fx.Float32(sv), fx.Float32(SCALE_MIN)), fx.Float32(SCALE_MAX)
    )
    return (
        fx.Float32(rocdl.log(fx.Float32.ir_type, sv.ir_value())),
        fx.Float32(rocdl.rcp(fx.Float32.ir_type, sv.ir_value())),
    )


# ---- Pack ----


def _pack_v(vr, pv, target, start, valid, head, lane, hk, oob):
    """V^T piece of lane p * 16 + d: dims 16d + 4p .. + 3 of the four tokens, transposed."""
    p, d = lane >> 4, lane & 15
    words = []
    for t in range(4):
        source = ((start + t) * hk + head) * D + 16 * d + 4 * p
        source = (fx.Int32(t) < valid).select(source, oob)
        words.append(_buffer_words(vr, source, 1)[0])
    _buffer_store_words(
        fx.Vector.from_elements(_transpose4(words), fx.Int32), pv, target
    )


def _pack_scales(KS, VS, PSC, start, valid, block, head, lane, inside, n, packed, hk):
    """Scale record of the block: k_scale, log2(v_scale), 1 / v_scale of its four tokens
    (lanes 0..3), 64 bytes."""
    scale_extent = fx.Uint32(n) * hk * 4
    ks, vs = _buffer(KS, scale_extent), _buffer(VS, scale_extent)
    t = lane & 3
    source = (t < valid).select(((start + t) * hk + head) * 4, fx.Int32(scale_extent))
    sk = _as_f32(_buffer_words(ks, source, 1)[0])
    lsv, rsv = _scale_terms(_as_f32(_buffer_words(vs, source, 1)[0]))
    record_extent = fx.Uint32(packed) * hk * 64
    record = (inside & (lane < 4)).select(
        (block * hk + head) * 64 + t * 4, fx.Int32(record_extent)
    )
    records = _buffer(PSC, record_extent)
    for part, value in enumerate((sk, lsv, rsv)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            records,
            record + 16 * part,
        )


@flyc.jit
def _pack_body(
    K,
    V,
    KS,
    VS,
    PK,
    PV,
    PSC,
    SOURCES,
    VALID,
    SLOTS,
    pair,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
):
    lane = fx.Int32(fx.thread_idx.x) & 63
    block, head = pair // HK, pair % HK
    inside = pair < PACKED * HK
    start = _uniform(fx.Int32(SOURCES[_min(block, PACKED - 1)]))
    valid = inside.select(
        _uniform(fx.Int32(VALID[_min(block, PACKED - 1)])), fx.Int32(0)
    )
    if fx.const_expr(SLOTTED):
        # A block occupies four consecutive pool rows from its first token's slot.
        start = _uniform(fx.Int32(SLOTS[start]))
    extent = fx.Uint32(N) * HK * D
    kr, vr = _buffer(K, extent), _buffer(V, extent)
    oob = fx.Int32(extent)
    out_extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    pk, pv = _buffer(PK, out_extent), _buffer(PV, out_extent)
    target = inside.select(
        (block * HK + head) * BLOCK_BYTES + lane * 16, fx.Int32(out_extent)
    )
    # K: piece (i, token, g) of lane i * 16 + token * 4 + g.
    i, token, g = lane >> 4, (lane >> 2) & 3, lane & 3
    row = ((start + token) * HK + head) * D
    row = (token < valid).select(row, oob)
    lo = _buffer_words(kr, row + 64 * i + 8 * g, 2)
    hi = _buffer_words(kr, row + 64 * i + 32 + 8 * g, 2)
    _buffer_store_words(
        fx.Vector.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int32), pk, target
    )
    _pack_v(vr, pv, target, start, valid, head, lane, HK, oob)
    if fx.const_expr(PER_TOKEN):
        _pack_scales(
            KS,
            VS,
            PSC,
            start,
            valid,
            block,
            head,
            lane,
            inside,
            N,
            PACKED,
            HK,
        )


@flyc.kernel(name="sparse_prefill_pack_kv_fp8_d256", known_block_size=[256, 1, 1])
def _pack_kernel(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
):
    pair = fx.Int32(fx.block_idx.x) * 4 + _uniform(fx.Int32(fx.thread_idx.x) >> 6)
    args = (K, V, KS, VS, PK, PV, PSC, SOURCES, VALID, SLOTS, pair, N, PACKED)
    if fx.const_expr(GATED):
        # K3 sets direct_flag[0] when some tile goes direct; otherwise nothing reads PK/PV.
        if _uniform(fx.Int32(DIRECT_FLAG[0])) != 0:
            _pack_body(*args, HK, PER_TOKEN, SLOTTED)
    else:
        _pack_body(*args, HK, PER_TOKEN, SLOTTED)


@flyc.jit
def _pack_launch(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
    stream: fx.Stream,
):
    _pack_kernel(
        K,
        V,
        KS,
        VS,
        PK,
        PV,
        PSC,
        SOURCES,
        VALID,
        SLOTS,
        DIRECT_FLAG,
        N,
        PACKED,
        HK,
        PER_TOKEN,
        SLOTTED,
        GATED,
        value_attrs={"llvm.target_features": _target_features()},
    ).launch(grid=((PACKED * HK + 3) // 4, 1, 1), block=(256, 1, 1), stream=stream)


# ---- Direct ----


def _slot_base(env, slot):
    """Packed byte base of key-list slot `slot`: complete blocks in block_indices order, the
    causal tail block, then padding (block 0, masked)."""
    index = _min(slot, fx.Int32(env.topk - 1))
    listed = fx.Int32(
        _buffer_words(env.gblocks, (env.row * env.topk + index) * 4, 1)[0]
    )
    block = (slot < env.complete).select(
        listed, (slot == env.complete).select(env.tail_block, fx.Int32(0))
    )
    return ((env.p_start + block) * env.hk + env.hkv) * BLOCK_BYTES


def _chunk_bases(env, t):
    """(K bases of blocks 8t + jj, 8t + 4 + jj; V/scale bases of blocks 8t + g, 8t + 4 + g)."""
    first = t * CHUNK
    return [
        _slot_base(env, first + env.jj),
        _slot_base(env, first + 4 + env.jj),
        _slot_base(env, first + env.g),
        _slot_base(env, first + 4 + env.g),
    ]


def _chunk_loads(env, bases):
    """K and V words of a chunk (and per-token scale terms): 8 + 8 b128 loads."""
    k = [
        _buffer_words(env.gk, bases[kg] + 256 * i + env.k_lane, 4)
        for kg in range(2)
        for i in range(4)
    ]
    v = [
        _buffer_words(env.gv, bases[2 + b] + 256 * p + env.v_lane, 4)
        for b in range(2)
        for p in range(4)
    ]
    scales = []
    if env.per_token:
        # Records of blocks 8t + g, 8t + 4 + g: k_scale, log2(v_scale), 1 / v_scale.
        for part in range(3):
            for b in range(2):
                scales.append(
                    _buffer_words(env.gsc, (bases[2 + b] >> 10) * 64 + 16 * part, 4)
                )
    return k, v, scales


def _chunk_words(env, bases, zero4):
    k, v, sc = _chunk_loads(env, bases)
    if not env.per_token:
        sc = [zero4] * 6
    return _flatten(k), _flatten(v), _flatten(sc)


def _probabilities(vals, sc, per_token):
    """FP8 P' words of the lane's 8 keys and its row-sum term (P' / v_scale per token)."""
    p = [_exp(vals[i]) for i in range(8)]
    terms = (
        [p[i] * _as_f32(sc[4 + i // 4][i % 4]) for i in range(8)] if per_token else p
    )
    total = terms[0]
    for x in terms[1:]:
        total = total + x
    return (_cvt_fp8(p[0:4]), _cvt_fp8(p[4:8])), total


def _pv16(o, v, pw):
    """O^T += V^T P for the 16 PV MFMAs (MFMA m writes dims 16d + m)."""
    o = fx.Vector(o)
    out = []
    for m in range(16):
        a = (v[m >> 2][m & 3], v[4 + (m >> 2)][m & 3])
        acc = fx.Vector.from_elements([o[4 * m + r] for r in range(4)], fx.Float32)
        acc = _mfma16(a, pw, acc)
        out += [acc[r] for r in range(4)]
    return fx.Vector.from_elements(out, fx.Float32)


def _store_o(env, o, inv):
    """O / l as BF16: lane (head, g) holds dims 64g .. 64g + 63 (dim 64g + 16r + m in
    o[4m + r])."""
    inv = fx.Vector.from_elements([fx.Float32(inv), fx.Float32(inv)], fx.Float32)
    words = []
    for r in range(4):
        values = [o[4 * m + r] for m in range(16)]
        for k in range(4):
            words += _scaled_bf16(values[4 * k : 4 * k + 4], inv)
    for s in range(8):
        _buffer_store_words(
            fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
            env.gout,
            env.out_lane + 16 * s,
        )


def _flatten(vectors, dtype=fx.Int32):
    return fx.Vector.from_elements([v[i] for v in vectors for i in range(4)], dtype)


def _split(vector, count):
    vector = fx.Vector(vector)
    return [
        fx.Vector.from_elements([vector[4 * j + i] for i in range(4)], vector.dtype)
        for j in range(count)
    ]


def _scores(env, k, sc, t, m_run):
    """S''(t) of the lane's 8 keys (blocks 8t + g, 8t + 4 + g; tokens 0..3), masked past the
    key list, with the lane-local max and the lazy-max ballot."""
    s = []
    for kg in range(2):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for c in range(8):
            w = k[4 * kg + (c >> 1)]
            q = env.q[c]
            acc = _mfma16((w[2 * (c & 1)], w[2 * (c & 1) + 1]), (q[0], q[1]), acc)
        s.append(acc)
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    neg = fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run)
    vals = []
    for b in range(2):
        slot = t * CHUNK + 4 * b + env.g
        for r in range(4):
            x = s[b][r]
            if env.per_token:
                x = x * _as_f32(sc[b][r])
            x = fx.Float32(fx.fma(x, env.c_scale, neg))
            if env.per_token:
                x = x + _as_f32(sc[2 + b][r])
            keep = (slot < env.complete) | (
                (slot == env.complete) & (fx.Int32(r) < env.tail)
            )
            vals.append(keep.select(x, fx.Float32(-math.inf)))
    local = vals[0]
    for x in vals[1:]:
        local = _maximum(local, x)
    trigger = (local > LAZY) | (fresh & (local > fx.Float32(-math.inf)))
    ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, trigger.ir_value()))
    return fx.Vector.from_elements(vals, fx.Float32), local, ballot


@flyc.jit
def _rescale_direct(o, l_run, m_run, vals, local, ballot):
    """Lazy max of a chunk: the row max over the row's four lanes (g); O, l and S'' follow."""
    o, l_run, m_run, vals = (
        fx.Vector(o),
        fx.Float32(l_run),
        fx.Float32(m_run),
        fx.Vector(vals),
    )
    if fx.Int64(ballot) != fx.Int64(0):
        local = fx.Float32(local)
        local = _maximum(local, local.shuffle_xor(16, 64))
        local = _maximum(local, local.shuffle_xor(32, 64))
        fresh = m_run == fx.Float32(_MAX_FLOOR)
        # A fresh row without a key keeps FLOOR (its S'' stay -inf).
        delta = _maximum(
            local - P_BIAS, fresh.select(fx.Float32(_MAX_FLOOR), fx.Float32(0.0))
        )
        corr = fresh.select(fx.Float32(0.0), _exp(fx.Float32(0.0) - delta))
        o = o * corr
        l_run = l_run * corr
        vals = vals - delta
        m_run = fresh.select(fx.Float32(0.0), m_run) + delta
    return o, l_run, m_run, vals


@flyc.kernel(name="sparse_prefill_direct_fp8_d256", known_block_size=[64, 1, 1])
def _direct_kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    PER_TOKEN: fx.Constexpr[bool],
):
    group = H // HK
    work = fx.Int32(fx.block_idx.x)
    tile, hkv = work // HK, work % HK
    row = _uniform(META[tile * 5])
    p_start, position = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 4])
    on = _enabled(fx.Int32(0), fx.Int32(1), row, QUERY_TILES, ACTIVE, GATED, NQ)
    lane = fx.Int32(fx.thread_idx.x) & 63
    n, g = lane & 15, lane >> 4
    visible = position + 1
    complete = _min(visible >> 2, fx.Int32(TOPK))
    tail = visible & 3
    blocks = complete + (tail != 0).select(fx.Int32(1), fx.Int32(0))
    chunks = on.select((blocks + CHUNK - 1) // CHUNK, fx.Int32(0))
    q_rows = fx.Uint32(NQ) * H
    head_ok = on & (n < group)
    qrow = row * H + hkv * group + n
    extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    scale_extent = fx.Uint32(PACKED) * HK * 64
    gq8 = _buffer(Q8, q_rows * D)
    q_base = head_ok.select(qrow * D + 8 * g, fx.Int32(q_rows * D))
    qs_offset = head_ok.select(qrow * 4, fx.Int32(q_rows * 4))
    qs = _as_f32(_buffer_words(_buffer(QS, q_rows * 4), qs_offset, 1)[0])
    c_scale = qs * (SCALE * LOG2E)
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(KS[0])
    env = type("_Env", (), {})()
    env.__dict__.update(
        row=row,
        hkv=hkv,
        hk=HK,
        topk=TOPK,
        p_start=p_start,
        complete=complete,
        tail=tail,
        tail_block=visible >> 2,
        jj=(lane >> 2) & 3,
        g=g,
        per_token=PER_TOKEN,
        c_scale=c_scale,
        # Q (B operand): head n, dims 32c + 8g .. + 7 for MFMA c.
        q=[_buffer_words(gq8, q_base + 32 * c, 2) for c in range(8)],
        gblocks=_buffer(BLOCKS, fx.Uint32(ROWS) * TOPK * 4),
        gk=_buffer(PK, extent),
        gv=_buffer(PV, extent),
        gsc=_buffer(PSC, scale_extent),
        k_lane=((lane & 3) * 4 + g) * 16,
        v_lane=n * 16,
        gout=_buffer(OUT, q_rows * D * 2),
        out_lane=head_ok.select(qrow * (D * 2) + g * 128, fx.Int32(q_rows * D * 2)),
    )
    zero4 = fx.Vector.filled(4, 0, fx.Int32)

    def loads(bases):
        return _chunk_words(env, bases, zero4)

    first_k, first_v, first_sc = loads(_chunk_bases(env, fx.Int32(0)))
    state = (
        fx.Vector.filled(64, 0.0, fx.Float32),
        fx.Float32(_MAX_FLOOR),
        fx.Float32(0.0),
        first_k,
        first_v,
        first_sc,
        fx.Vector.from_elements(_chunk_bases(env, fx.Int32(1)), fx.Int32),
    )

    @flyc.jit
    def step(o, m_run, l_run, kw, vw, scw, bases, t):
        """Chunk t from its prefetched words; issues chunk t + 1's loads first."""
        bases = fx.Vector(bases)
        k_next, v_next, sc_next = loads([bases[i] for i in range(4)])
        bases_next = _chunk_bases(env, fx.Int32(t) + 2)
        k, v, sc = _split(kw, 8), _split(vw, 8), _split(scw, 6)
        vals, local, ballot = _scores(env, k, sc, fx.Int32(t), fx.Float32(m_run))
        o, l_run, m_run, vals = _rescale_direct(o, l_run, m_run, vals, local, ballot)
        pw, total = _probabilities(vals, sc, PER_TOKEN)
        l_run = fx.Float32(l_run) + total
        return (
            _pv16(o, v, pw),
            m_run,
            l_run,
            k_next,
            v_next,
            sc_next,
            fx.Vector.from_elements(bases_next, fx.Int32),
        )

    for t in range(fx.Int32(0), chunks, fx.Int32(1)):
        state = step(*state, t)
    o, l_run = fx.Vector(state[0]), fx.Float32(state[2])
    total = l_run + l_run.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    if fx.const_expr(not PER_TOKEN):
        inv = inv * fx.Float32(VS[0])
    _store_o(env, [o[i] for i in range(64)], inv)


@flyc.jit
def _direct_launch(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    PER_TOKEN: fx.Constexpr[bool],
    GRID: fx.Int32,
    stream: fx.Stream,
):
    _direct_kernel(
        Q8,
        QS,
        PK,
        PV,
        PSC,
        KS,
        VS,
        OUT,
        BLOCKS,
        META,
        ACTIVE,
        QUERY_TILES,
        H,
        HK,
        NQ,
        ROWS,
        PACKED,
        TOPK,
        GATED,
        SCALE,
        PER_TOKEN,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "llvm.target_features": _target_features(),
        },
    ).launch(grid=(GRID, 1, 1), block=(64, 1, 1), stream=stream)


# ---- Host ----


class DirectPlan:
    """Packed-KV FP8 direct plan: one wave per (query row, KV head)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def scratch_specs(prepared):
    blocks, hk = prepared.packed_blocks, prepared.hk
    specs = [
        ("packed_key", (blocks * hk * BLOCK_BYTES,), torch.uint8),
        ("packed_value", (blocks * hk * BLOCK_BYTES,), torch.uint8),
    ]
    if prepared.per_token:
        specs.append(("scale_records", (blocks * hk * 16,), torch.float32))
    return specs


def prepare(*, inputs, union=None, per_token, scratch=True):
    """Plan the FP8 direct path: every request's context packed per call, one wave per row."""
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    k_starts, p_starts = np.cumsum(lengths) - lengths, np.cumsum(blocks) - blocks
    request = np.repeat(np.arange(len(q_lens)), q_lens)
    local = np.arange(len(request)) - np.repeat(np.cumsum(q_lens) - q_lens, q_lens)
    rows = len(request)
    metadata = np.stack(
        (
            np.arange(rows),
            np.ones(rows, dtype=np.int64),
            p_starts[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    owner = np.repeat(np.arange(len(blocks)), blocks)
    index = np.arange(len(owner)) - p_starts[owner]
    sources = k_starts[owner] + 4 * index
    valid = np.minimum(lengths[owner] - 4 * index, 4)
    device = inputs.q.device
    standin = inputs.query_positions
    prepared = DirectPlan(
        metadata=upload(metadata, device),
        rows=rows,
        hk=hk,
        packed_blocks=int(blocks.sum()),
        pack_sources=upload(sources, device),
        pack_valid=upload(valid, device),
        per_token=per_token,
        slotted=inputs.kv_slots is not None,
        gated=union is not None,
        direct_flag=standin if union is None else union.direct_flag,
        active=standin if union is None else union.active,
        query_tiles=standin if union is None else union.query_tiles,
        source_blocks=inputs.block_indices,
    )
    if scratch:
        allocate_scratch(prepared, scratch_specs(prepared), device)
    return prepared


_PACK, _DIRECT = {}, {}


def _pack_args(inputs, prepared, stream):
    empty = prepared.packed_key[:4].view(torch.float32)
    return (
        inputs.k,
        inputs.v,
        inputs.k_scale.view(-1),
        inputs.v_scale.view(-1),
        prepared.packed_key,
        prepared.packed_value,
        prepared.scale_records if prepared.per_token else empty,
        prepared.pack_sources,
        prepared.pack_valid,
        inputs.kv_slots if prepared.slotted else prepared.pack_sources,
        prepared.direct_flag,
        inputs.k.shape[0],
        prepared.packed_blocks,
        prepared.hk,
        prepared.per_token,
        prepared.slotted,
        prepared.gated,
        stream,
    )


def _direct_args(inputs, prepared, out, stream):
    empty = prepared.packed_key[:4].view(torch.float32)
    return (
        inputs.q8.view(-1),
        inputs.qs.view(-1),
        prepared.packed_key,
        prepared.packed_value,
        prepared.scale_records if prepared.per_token else empty,
        inputs.k_scale.view(-1),
        inputs.v_scale.view(-1),
        out.view(-1),
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.query_tiles,
        inputs.q.shape[1],
        prepared.hk,
        inputs.q.shape[0],
        prepared.rows,
        prepared.packed_blocks,
        inputs.topk,
        prepared.gated,
        inputs.scale,
        prepared.per_token,
        prepared.rows * prepared.hk,
        stream,
    )


def _compile(cache, key, launch, args, device, stream):
    compiled = cache.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(device), torch.cuda.stream(stream):
            compiled = cache[key] = flyc.compile(launch, *args)
    return compiled


def _compiled(inputs, prepared, out, stream):
    device = inputs.q.device
    pack = _pack_args(inputs, prepared, stream)
    direct = _direct_args(inputs, prepared, out, stream)
    # Specializations: pack (HK, PER_TOKEN, SLOTTED, GATED); direct (H, HK, TOPK, GATED,
    # SCALE, PER_TOKEN).
    pack_fn = _compile(
        _PACK, (device, *pack[13:17]), _pack_launch, pack, device, stream
    )
    key = (device, *direct[12:14], *direct[17:21])
    direct_fn = _compile(_DIRECT, key, _direct_launch, direct, device, stream)
    return pack, direct, pack_fn, direct_fn


def run(*, inputs, prepared, out):
    """Pack the context, then the direct waves. inputs carries q8/qs (quantize_q) and
    k_scale/v_scale like the union kernel."""
    if prepared.rows == 0 or prepared.packed_blocks == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    with torch.cuda.device(inputs.q.device):
        pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
        pack_fn(*pack)
        direct_fn(*direct)


def launcher(*, inputs, prepared, out):
    """launch(q8, qs, k, v, out, k_scale, v_scale, slots) with flat Q8/QS/out/scales and
    unflattened K/V; other arguments prebuilt."""
    if prepared.rows == 0 or prepared.packed_blocks == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
    slotted = prepared.slotted

    def launch(q8, qs, k, v, o, ks, vs, slots):
        pack_fn(k, v, ks, vs, *pack[4:9], slots if slotted else pack[9], *pack[10:])
        direct_fn(q8, qs, *direct[2:5], ks, vs, o, *direct[8:])

    return launch
