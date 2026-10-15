"""Phase device functions of the full-attention mixer of a Qwen3.8 decode layer (QSA sparse attention).

``qkv``: the W8A8 ``qkv_proj`` (``[q | gate]`` per query head, then ``k``, ``v``) on ``RoundGemv``.
``qk_prep``: wave item (row, j) for the query heads, the key and the value: Gemma RMSNorm of q/k, the
interleaved MRoPE of the 64 rotary channels from SGLang's BF16 cos/sin table (every product and sum
rounded to BF16 like SGLang's ROCm MRoPE), the query pre-scaled into the exp2 domain and rounded to BF16
(SGLang's decode kernel), and the key/value stored into the paged KV cache at ``out_cache_loc``.
``attention_scores``: wave item (row, 16-key chunk of the selected positions): ``S^T = K Q^T`` on MFMA
(keys are the M rows, the 16 head columns hold the rank's query heads); masked FP32 scores and the
chunk's per-head maximum go to scratch. ``attention_values``: wave item (row, segment of ``group``
chunks): SGLang's decode kernel processes the front-packed selections sequentially in blocks of
``block_keys`` keys with the running maximum, rounding probabilities to BF16 at it; the segment starts
from the running maximum of the earlier chunks, so every block uses SGLang's maximum and the BF16
probabilities are SGLang's. ``out^T += V^T P^T`` runs on MFMA with V transposed in registers; the
segment's (running max, sum, FP32 accumulator) go to scratch. Selected positions outside
``[0, seq_len)`` are masked; keys and values are read through ``req_to_token`` with 64-bit addresses.
``combine_gate``: wave item (row, head): merges the segments, rounds the attention output to BF16 and
applies the sigmoid output gate (``fused_sigmoid_mul``).

The rank holds one KV head (TP 2 and 4).
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl

from ..config import QWEN38_FLASH_NEXT
from . import phases as ph
from pyhip.ops.qsa.flydsl import indexer_topk as qtopk

from ._ir import (DPP_ROR8, DPP_XOR1, DPP_XOR2, DROP, LOG2E, _swap16, _swap32, bf16_pair, bf16_round, bf16_word,
                  bf16x4_half, dpp, exp2, fmax, lds_load, lds_store, load, mfma_bf16, scalar, scalar_f32,
                  sigmoid_accurate, store, wave_sum, words, zeros4)
from ._phase import STAGED_WAVES, buffer_at, device, pipelined

_C = QWEN38_FLASH_NEXT
DA = _C.head_dim
ROT = _C.rotary_dim
TOPK = _C.index_topk + _C.index_compress_ratio - 1
CHUNK = 16
CHUNKS = -(-TOPK // CHUNK)
MROPE_SECTION = (11, 11, 10)
# SGLang's decode kernel: (q * scaling) * 1.4426950408 in FP32 (scaling = 1/16 is exact).
Q_SCALE = DA ** -0.5 * LOG2E
F32 = fx.Float32
NEG_INF = float("-inf")


def group_for(tiles):
    """16-key chunks per attention segment (more rows -> longer segments, fewer partials); a multiple of
    the chunks of an SGLang block."""
    return {1: 2, 2: 4, 4: 8}[tiles]


def block_keys(tiles):
    """SGLang's decode ``BLOCK_N`` for the row count of ``tiles`` (``_get_best_config``: <= 32 rows -> 32,
    <= 64 -> 64)."""
    return 64 if tiles == 4 else 32


def segments(tiles):
    return -(-CHUNKS // group_for(tiles))


def qkv_columns(heads):
    return (2 * heads + 2) * DA


def _global_load(address, words_count=4):
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.Int64(address).ir_value())
    return fx.Vector(llvm.load(ir.VectorType.get([words_count], fx.Int32.ir_type), pointer,
                               alignment=4 * words_count))


def _perm(a, b, selector):
    """``v_perm_b32``: selector bytes 0-3 pick bytes of ``b``, 4-7 bytes of ``a``."""
    return fx.Int32(llvm.call_intrinsic(fx.Int32.ir_type, "llvm.amdgcn.perm",
                                        [fx.Int32(a).ir_value(), fx.Int32(b).ir_value(),
                                         fx.Int32(selector).ir_value()], [], []))


def _mrope_axis(pair):
    """Position axis of rotary pair ``pair`` under the interleaved MRoPE (SGLang's ``axis_map``)."""
    m3 = pair % 3
    axis1 = (m3 == 1) & (pair < 3 * MROPE_SECTION[1])
    axis2 = (m3 == 2) & (pair < 3 * MROPE_SECTION[2])
    return axis1.select(fx.Int32(1), axis2.select(fx.Int32(2), fx.Int32(0)))


# ---------------------------------------------------------------- projections


def qkv_gemv(ctx, xq, w, heads, tiles, ctas):
    return ph.Gemv(ctx, ph.H, tiles, "fp8", qkv_columns(heads) // 16, ctas, lambda item: (w, item, 0), xq)


@device
def qkv(ctx, xq, xs, w, ws, out, heads, tiles, ctas, pre=None):
    """Half item per 16 rows of the W8A8 ``qkv_proj``; BF16 ``acc * x_scale * w_scale``."""
    gemv = qkv_gemv(ctx, xq, w, heads, tiles, ctas)
    gather = gemv.gemm.gather
    lane, wave = ctx.lane, ctx.half_wave
    n = qkv_columns(heads)

    def epilogue(block, valid, region):
        for t in range_constexpr(tiles):
            if wave == t % STAGED_WAVES:
                ph._store_tile(ctx, out, gather(region, t, lane), block, t, n, "scaled", ctx.aux, valid, xs, ws)

    gemv.run(epilogue, pre)


# ---------------------------------------------------------------- q/k norm, rope, KV store


@device
def qk_prep(ctx, qkv_out, q_out, q_norm, k_norm, table, positions, pos_stride, cache_loc, k_cache, v_cache, heads,
            eps):
    """Wave item (row, j): query head ``j < heads``, the key (``j == heads``) or the value. Lane ``l`` owns
    channels ``4 l .. 4 l + 3``; rotary pair ``i`` (channels ``i``, ``i + 32``) sits in lanes ``i // 4`` and
    ``8 + i // 4``, which swap halves with a row rotation by 8."""
    rows, lane = ctx.rows, ctx.lane
    n = qkv_columns(heads)
    per_row = heads + 2
    low = lane < 8
    rotary = lane < 16
    for item_iv in range(ctx.wave_id, rows * per_row, ctx.waves):
        item = fx.Int32(item_iv)
        row = item // per_row
        j = item % per_row
        is_q = j < heads
        is_k = j == heads
        is_v = j == heads + 1
        column = is_q.select(j * 2 * DA, heads * 2 * DA + (j - heads) * DA)
        word = load(qkv_out, (row * n + column + lane * 4) * 2, words=2, aux=ctx.aux)
        values = list(bf16_pair(word[0])) + list(bf16_pair(word[1]))
        wq = load(q_norm, lane * 8, words=2)
        wk = load(k_norm, lane * 8, words=2)
        gains = list(bf16_pair(is_q.select(wq[0], wk[0]))) + list(bf16_pair(is_q.select(wq[1], wk[1])))
        squares = values[0] * values[0]
        for e in range_constexpr(1, 4):
            squares = squares + values[e] * values[e]
        total = scalar_f32(wave_sum(squares, lane))
        rstd = F32(rocdl.rsq(F32.ir_type, (total * (1.0 / DA) + eps).ir_value()))
        normed = [bf16_round((values[e] * rstd) * (1.0 + gains[e])) for e in range_constexpr(4)]
        roped = []
        for e in range_constexpr(4):
            partner = dpp(normed[e], DPP_ROR8)
            pair = 4 * (lane & 7) + e
            position = load(positions, (_mrope_axis(pair) * pos_stride + row) * 8, words=1)
            cos = ph._half_word(table, position * ROT + pair, 0)
            sin = ph._half_word(table, position * ROT + ROT // 2 + pair, 0)
            first = low.select(normed[e], partner)
            second = low.select(partner, normed[e])
            rotated_first = bf16_round(bf16_round(first * cos) - bf16_round(second * sin))
            rotated_second = bf16_round(bf16_round(second * cos) + bf16_round(first * sin))
            roped.append(rotary.select(low.select(rotated_first, rotated_second), normed[e]))
        scaled = [bf16_round(roped[e] * Q_SCALE) for e in range_constexpr(4)]
        q_words = words([bf16_word(scaled[0], scaled[1]), bf16_word(scaled[2], scaled[3])])
        store(q_out, is_q.select(((row * heads + j) * DA + lane * 4) * 2, fx.Int32(DROP)), q_words, aux=ctx.aux)
        slot = load(cache_loc, row * 8, words=1)
        k_row = buffer_at(k_cache + fx.Int64(slot) * (DA * 2), DA * 2)
        v_row = buffer_at(v_cache + fx.Int64(slot) * (DA * 2), DA * 2)
        k_words = words([bf16_word(roped[0], roped[1]), bf16_word(roped[2], roped[3])])
        store(k_row, is_k.select(lane * 8, fx.Int32(DROP)), k_words)
        store(v_row, is_v.select(lane * 8, fx.Int32(DROP)), word)


# ---------------------------------------------------------------- sparse attention


def _guard(value):
    """The running maximum as an exp2 reference (0 while nothing is valid yet)."""
    return (value == F32(NEG_INF)).select(F32(0.0), value)


def _value_operands(vv):
    """A operands of ``out^T += V^T P^T`` for the 4 channel tiles of one 64-channel block: ``vv[i]`` holds the
    two words (4 BF16 channels ``64 mm + 4 r + [0, 4)``) of key ``4 g + i``; tile ``t`` takes channel ``t`` of
    the 4 keys."""
    out = []
    for t in range_constexpr(4):
        d = t // 2
        selector = 0x05040100 if t % 2 == 0 else 0x07060302
        out.append(words([_perm(vv[1][d], vv[0][d], selector), _perm(vv[3][d], vv[2][d], selector)]).bitcast(fx.Int16))
    return out


SCORE_CHUNKS = 4  # chunks of one scores item (their loads are issued together)
VALUE_BATCH = 4  # chunks of a values item whose loads are issued together


@device
def attention_scores(ctx, q, topk, req_to_token, req_pool, seq_lens, k_cache, scores, chunk_max, heads, rtt_stride):
    """Wave item (row, ``SCORE_CHUNKS`` chunks); the selections, slots and keys of all its chunks are loaded
    before the math. Lane ``l`` (``g = l // 16``, ``r = l % 16``) loads key ``r`` of a chunk for the A operand
    of ``S^T = K Q^T`` (channels ``32 jj + 8 g + [0, 8)``; Q^T uses the same channel order) and owns the
    scores of keys ``4 g + [0, 4)`` for head ``r``: FP32 ``scores[row, key, head]`` (-inf where masked) and
    ``chunk_max[row, chunk, head]``."""
    rows, lane = ctx.rows, ctx.lane
    g = lane >> 4
    r = lane & 15
    head_ok = r < heads
    quads = -(-CHUNKS // SCORE_CHUNKS)
    for item_iv in range(ctx.wave_id, rows * quads, ctx.waves):
        item = fx.Int32(item_iv)
        row = item // quads
        first = (item % quads) * SCORE_CHUNKS
        req = load(req_pool, row * 8, words=1)
        seq = load(seq_lens, row * 8, words=1)
        qv = [load(q, head_ok.select(((row * heads + r) * DA + 32 * jj + 8 * g) * 2, fx.Int32(DROP)), aux=ctx.aux)
              for jj in range_constexpr(8)]
        inside_a, idx_a, idx4 = [], [], []
        for c in range_constexpr(SCORE_CHUNKS):
            base = (first + c) * CHUNK
            inside = (base + r) < TOPK
            inside_a.append(inside)
            idx_a.append(load(topk, inside.select((row * TOPK + base + r) * 4, fx.Int32(DROP)), words=1,
                              aux=ctx.aux))
            idx4.append(load(topk, ((first + c) < CHUNKS).select((row * TOPK + base + 4 * g) * 4, fx.Int32(DROP)),
                             words=4, aux=ctx.aux))
        ok_a, slot_a = [], []
        for c in range_constexpr(SCORE_CHUNKS):
            ok = inside_a[c] & (idx_a[c] >= 0) & (idx_a[c] < seq)
            ok_a.append(ok)
            slot_a.append(load(req_to_token, ok.select((req * rtt_stride + idx_a[c]) * 4, fx.Int32(DROP)), words=1))
        kv = []
        for c in range_constexpr(SCORE_CHUNKS):
            k_base = k_cache + fx.Int64(ok_a[c].select(slot_a[c], fx.Int32(0))) * (DA * 2)
            kv.append([_global_load(k_base + (32 * jj + 8 * g) * 2) for jj in range_constexpr(8)])
        for c in range_constexpr(SCORE_CHUNKS):
            base = (first + c) * CHUNK
            s = zeros4()
            for jj in range_constexpr(8):
                for h in range_constexpr(2):
                    s = mfma_bf16(bf16x4_half(kv[c][jj], h), bf16x4_half(qv[jj], h), s)
            values = []
            for i in range_constexpr(4):
                index = fx.Int32(idx4[c][i])
                ok = ((base + 4 * g + i) < TOPK) & (index >= 0) & (index < seq)
                values.append(ok.select(s[i], F32(NEG_INF)))
            peak = fmax(fmax(values[0], values[1]), fmax(values[2], values[3]))
            peak = fmax(peak, _swap16(peak))
            peak = fmax(peak, _swap32(peak, lane))
            live = (first + c) < CHUNKS
            for i in range_constexpr(4):
                at = live.select(((row * CHUNKS * CHUNK + base + 4 * g + i) * 16 + r) * 4, fx.Int32(DROP))
                store(scores, at, values[i].bitcast(fx.Int32), aux=ctx.aux)
            store(chunk_max, (live & (g == 0)).select(((row * CHUNKS + first + c) * 16 + r) * 4, fx.Int32(DROP)),
                  peak.bitcast(fx.Int32), aux=ctx.aux)


@device
def attention_values(ctx, scores, chunk_max, topk, req_to_token, req_pool, seq_lens, v_cache, part_ml, part_acc,
                     heads, tiles, rtt_stride):
    """Wave item (row, segment of ``group`` chunks, channel half). The running maximum before the segment
    comes from the earlier chunk maxima (lane group ``g`` takes chunks ``g, g + 4, ...``); each SGLang block
    (``block_keys``) raises it, rescales the sum and the accumulator, and its probabilities ``exp2(s - max)``
    are rounded to BF16 for ``out^T += V^T P^T`` over the half's 128 channels: lane ``l`` loads the values of
    its keys ``4 g + [0, 4)`` at channels ``64 mm + 4 r + [0, 4)`` and transposes them in registers
    (``_value_operands``). Chunks run in batches of ``VALUE_BATCH`` whose loads are issued together."""
    rows, lane = ctx.rows, ctx.lane
    group = group_for(tiles)
    segs = segments(tiles)
    per_block = block_keys(tiles) // CHUNK
    lookback = -(-CHUNKS // 4)
    g = lane >> 4
    r = lane & 15
    head_ok = r < heads
    for item_iv in range(ctx.wave_id, rows * segs * 2, ctx.waves):
        item = fx.Int32(item_iv)
        half = item & 1
        row = (item >> 1) // segs
        seg = (item >> 1) % segs
        first = seg * group
        req = load(req_pool, row * 8, words=1)
        seq = load(seq_lens, row * 8, words=1)
        prior = F32(NEG_INF)
        for j in range_constexpr(lookback):
            c = 4 * j + g
            earlier = c < first
            value = load(chunk_max, earlier.select(((row * CHUNKS + c) * 16 + r) * 4, fx.Int32(DROP)), words=1,
                         aux=ctx.aux).bitcast(F32)
            prior = fmax(prior, earlier.select(value, F32(NEG_INF)))
        prior = fmax(prior, _swap16(prior))
        m = fmax(prior, _swap32(prior, lane))
        m_ref = _guard(m)
        total = F32(0.0)
        acc = [zeros4() for _ in range_constexpr(8)]
        for b0 in range_constexpr(0, group, VALUE_BATCH):
            batch = list(range(b0, min(b0 + VALUE_BATCH, group)))
            block_peaks, sc, idx4 = {}, {}, {}
            for c in batch:
                chunk = first + c
                inside = chunk < CHUNKS
                if c % per_block == 0:
                    peaks = []
                    for b in range_constexpr(per_block):
                        member = (chunk + b) < CHUNKS
                        value = load(chunk_max, member.select(((row * CHUNKS + chunk + b) * 16 + r) * 4,
                                                              fx.Int32(DROP)), words=1, aux=ctx.aux).bitcast(F32)
                        peaks.append(member.select(value, F32(NEG_INF)))
                    block_peaks[c] = peaks
                base = chunk * CHUNK
                sc[c] = [load(scores, inside.select(((row * CHUNKS * CHUNK + base + 4 * g + i) * 16 + r) * 4,
                                                    fx.Int32(DROP)), words=1, aux=ctx.aux)
                         for i in range_constexpr(4)]
                idx4[c] = load(topk, inside.select((row * TOPK + base + 4 * g) * 4, fx.Int32(DROP)), words=4, aux=ctx.aux)
            oks, slots = {}, {}
            for c in batch:
                chunk = first + c
                base = chunk * CHUNK
                oks[c], slots[c] = [], []
                for i in range_constexpr(4):
                    index = fx.Int32(idx4[c][i])
                    ok = (chunk < CHUNKS) & ((base + 4 * g + i) < TOPK) & (index >= 0) & (index < seq)
                    oks[c].append(ok)
                    slots[c].append(load(req_to_token, ok.select((req * rtt_stride + index) * 4, fx.Int32(DROP)),
                                         words=1))
            vv = {}
            for c in batch:
                vv[c] = []
                for j in range_constexpr(2):
                    keys = []
                    for i in range_constexpr(4):
                        v_base = v_cache + fx.Int64(oks[c][i].select(slots[c][i], fx.Int32(0))) * (DA * 2)
                        pair = _global_load(v_base + (64 * (2 * half + j) + 4 * r) * 2, 2)
                        keys.append([oks[c][i].select(fx.Int32(pair[0]), fx.Int32(0)),
                                     oks[c][i].select(fx.Int32(pair[1]), fx.Int32(0))])
                    vv[c].append(keys)
            for c in batch:
                if c % per_block == 0:
                    block_max = block_peaks[c][0]
                    for b in range_constexpr(1, per_block):
                        block_max = fmax(block_max, block_peaks[c][b])
                    m_new = fmax(m, block_max)
                    m_ref = _guard(m_new)
                    alpha = exp2(m - m_ref)
                    total = total * alpha
                    scale4 = words([alpha] * 4, F32)
                    acc = [a * scale4 for a in acc]
                    m = m_new
                inside = (first + c) < CHUNKS  # chunks past the end hold no scores
                probs = [exp2(inside.select(sc[c][i].bitcast(F32), F32(NEG_INF)) - m_ref)
                         for i in range_constexpr(4)]
                psum = (probs[0] + probs[1]) + (probs[2] + probs[3])
                psum = psum + _swap16(psum)
                total = total + (psum + _swap32(psum, lane))
                p_bf16 = words([bf16_word(probs[0], probs[1]), bf16_word(probs[2], probs[3])]).bitcast(fx.Int16)
                for j in range_constexpr(2):
                    operands = _value_operands(vv[c][j])
                    for t in range_constexpr(4):
                        acc[4 * j + t] = mfma_bf16(operands[t], p_bf16, acc[4 * j + t])
        record = row * segs + seg
        store(part_ml, ((half == 0) & (g == 0) & head_ok).select((record * 16 + r) * 8, fx.Int32(DROP)),
              words([m, total], F32).bitcast(fx.Int32), aux=ctx.aux)
        for j in range_constexpr(2):
            for i in range_constexpr(4):
                values = words([acc[4 * j + t][i] for t in range_constexpr(4)], F32).bitcast(fx.Int32)
                at = ((record * heads + r) * DA + 64 * (2 * half + j) + 16 * g + 4 * i) * 4
                store(part_acc, head_ok.select(at, fx.Int32(DROP)), values, aux=ctx.aux)


@device
def combine_gate(ctx, part_ml, part_acc, qkv_out, y, heads, tiles):
    """Wave item (row, head): the segments' weights ``exp2(m_s - m_last)`` (lane ``s``) and the merged sum,
    then lane ``l`` merges channels ``4 l .. 4 l + 3`` (weights broadcast through LDS), rounds the attention
    output to BF16 and applies the sigmoid gate."""
    rows, lane = ctx.rows, ctx.lane
    segs = segments(tiles)
    n = qkv_columns(heads)
    weights_at = ctx.shared + ctx.wave * 512
    for item_iv in range(ctx.wave_id, rows * heads, ctx.waves):
        item = fx.Int32(item_iv)
        row = item // heads
        head = item % heads
        last = load(part_ml, ((row * segs + segs - 1) * 16 + head) * 8, words=1, aux=ctx.aux).bitcast(F32)
        m_ref = _guard(last)
        weighted = F32(0.0)
        for part_i in range_constexpr(-(-segs // 64)):
            s = lane + 64 * part_i
            live = s < segs
            ml = load(part_ml, live.select(((row * segs + s) * 16 + head) * 8, fx.Int32(DROP)), words=2,
                      aux=ctx.aux).bitcast(F32)
            w = live.select(exp2(ml[0] - m_ref), F32(0.0))
            weighted = weighted + w * ml[1]
            lds_store(weights_at + s * 4, w.bitcast(fx.Int32), words=1)
        total = scalar_f32(wave_sum(weighted, lane))
        merged = [F32(0.0)] * 4

        def issue(group):
            out = []
            for k in range_constexpr(len(group)):
                s = group[k]
                out.append((lds_load(weights_at + s * 4, words=1).bitcast(F32),
                            load(part_acc, (((row * segs + s) * heads + head) * DA + 4 * lane) * 4,
                                 aux=ctx.aux).bitcast(F32)))
            return out

        def consume(group, loaded):
            for k in range_constexpr(len(loaded)):
                w, part = loaded[k]
                for e in range_constexpr(4):
                    merged[e] = merged[e] + part[e] * w

        pipelined([list(range(start, min(start + 8, segs))) for start in range(0, segs, 8)], issue, consume)
        out = [bf16_round(merged[e] / total) for e in range_constexpr(4)]
        gate = load(qkv_out, (row * n + head * 2 * DA + DA + lane * 4) * 2, words=2, aux=ctx.aux)
        gates = list(bf16_pair(gate[0])) + list(bf16_pair(gate[1]))
        gated = [out[e] * sigmoid_accurate(gates[e]) for e in range_constexpr(4)]
        store(y, ((row * heads + head) * DA + lane * 4) * 2,
              words([bf16_word(gated[0], gated[1]), bf16_word(gated[2], gated[3])]), aux=ctx.aux)


# ---------------------------------------------------------------- QSA indexer

IH = _C.index_heads
ID = _C.index_head_dim
RATIO = _C.index_compress_ratio
RING = 2 * RATIO  # SGLang's qsa_ring_slots_per_request
INDEX_COLUMNS = (IH + 1) * ID
INDEX_SCALE = ID ** -0.5


def _gemma_norm2(values, gains, lane, eps):
    """Gemma RMSNorm of a 128-channel head held 2 channels per lane, rounded to BF16."""
    squares = values[0] * values[0] + values[1] * values[1]
    total = scalar_f32(wave_sum(squares, lane))
    rstd = F32(rocdl.rsq(F32.ir_type, (total * (1.0 / ID) + eps).ir_value()))
    return [bf16_round((values[e] * rstd) * (1.0 + gains[e])) for e in range_constexpr(2)]


def _index_rope(normed, lane, table, position_of):
    """RoPE (BF16 per operator) of a 128-channel head held 2 channels per lane: lane ``l`` has channels
    ``2 l, 2 l + 1``, so rotary pair ``i`` (channels ``i``, ``i + 32``) sits in lanes ``i // 2`` and ``16 + i // 2``,
    which swap with ``_swap16``; ``position_of(pair)`` gives the pair's MRoPE coordinate."""
    low = lane < 16
    rotary = lane < 32
    out = []
    for e in range_constexpr(2):
        partner = _swap16(normed[e])
        pair = 2 * (lane & 15) + e
        position = position_of(pair)
        cos = ph._half_word(table, position * ROT + pair, 0)
        sin = ph._half_word(table, position * ROT + ROT // 2 + pair, 0)
        first = low.select(normed[e], partner)
        second = low.select(partner, normed[e])
        rotated_first = bf16_round(bf16_round(first * cos) - bf16_round(second * sin))
        rotated_second = bf16_round(bf16_round(second * cos) + bf16_round(first * sin))
        out.append(rotary.select(low.select(rotated_first, rotated_second), normed[e]))
    return out


@device
def index_prep(ctx, index_qk, q_out, q_norm, table, positions, pos_stride, req_pool, seq_lens, ring, ring_pos, eps):
    """Wave item (row, j): index query head ``j < 4`` (Gemma norm and RoPE into ``q_out``) or the index key:
    the raw projection into the request's pending ring slot ``req * 8 + (seq - 1) % 8`` and its MRoPE
    coordinates into the ring's position rows (SGLang's ``set_qsa_key_state_buffer`` and
    ``set_qsa_rope_position_buffer``)."""
    rows, lane = ctx.rows, ctx.lane
    per_row = IH + 1
    gains = list(bf16_pair(load(q_norm, lane * 4, words=1)))
    for item_iv in range(ctx.wave_id, rows * per_row, ctx.waves):
        item = fx.Int32(item_iv)
        row = item // per_row
        j = item % per_row
        word = load(index_qk, (row * INDEX_COLUMNS + j * ID + lane * 2) * 2, words=1, aux=ctx.aux)
        normed = _gemma_norm2(list(bf16_pair(word)), gains, lane, eps)

        def position_of(pair):
            return load(positions, (_mrope_axis(pair) * pos_stride + row) * 8, words=1)

        roped = _index_rope(normed, lane, table, position_of)
        store(q_out, (j < IH).select(((row * IH + j) * ID + lane * 2) * 2, fx.Int32(DROP)),
              bf16_word(roped[0], roped[1]), aux=ctx.aux)
        seq = load(seq_lens, row * 8, words=1)
        req = load(req_pool, row * 8, words=1)
        slot = req * RING + (seq - 1) % RING
        is_key = j == IH
        store(ring, is_key.select(slot * (ID * 2) + lane * 4, fx.Int32(DROP)), word, aux=ctx.aux)
        coordinate = load(positions, ((lane & 3) * pos_stride + row) * 8, words=2)
        store(ring_pos, (is_key & (lane < 3)).select((slot * 3 + lane) * 8, fx.Int32(DROP)), coordinate, aux=ctx.aux)


@device
def index_compress(ctx, ring, ring_pos, k_norm, table, req_pool, seq_lens, cache_loc, compressed, eps):
    """Wave item per row; a row whose new token completes a group (``seq % 4 == 0``) compresses the group's 4
    ring members: FP32 mean rounded to BF16, Gemma norm, RoPE at the first member's coordinates, stored at
    compressed slot ``out_cache_loc // 4`` (SGLang's ``update_key_state_and_compress``)."""
    rows, lane = ctx.rows, ctx.lane
    gains = list(bf16_pair(load(k_norm, lane * 4, words=1)))
    for row_iv in range(ctx.wave_id, rows, ctx.waves):
        row = fx.Int32(row_iv)
        seq = load(seq_lens, row * 8, words=1)
        req = load(req_pool, row * 8, words=1)
        boundary = (seq % RATIO) == 0
        total = [F32(0.0), F32(0.0)]
        for m in range_constexpr(RATIO):
            slot = req * RING + (seq - RATIO + m) % RING
            member = list(bf16_pair(load(ring, boundary.select(slot * (ID * 2) + lane * 4, fx.Int32(DROP)), words=1,
                                         aux=ctx.aux)))
            total = [total[e] + member[e] for e in range_constexpr(2)]
        pooled = [bf16_round(total[e] * (1.0 / RATIO)) for e in range_constexpr(2)]
        normed = _gemma_norm2(pooled, gains, lane, eps)
        first = req * RING + (seq - RATIO) % RING

        def position_of(pair):
            return load(ring_pos, boundary.select((first * 3 + _mrope_axis(pair)) * 8, fx.Int32(DROP)), words=1,
                        aux=ctx.aux)

        roped = _index_rope(normed, lane, table, position_of)
        target = load(cache_loc, row * 8, words=1) // RATIO
        store(compressed, boundary.select(target * (ID * 2) + lane * 4, fx.Int32(DROP)), bf16_word(roped[0], roped[1]),
              aux=ctx.aux)


@device
def index_logits(ctx, q, compressed, req_to_token, req_pool, seq_lens, logits, rtt_stride, max_blocks):
    """Wave item (row, 16 compressed blocks): ``relu(q_h . k)`` summed over the 4 index heads, over sqrt(128), into
    ``logits[row, block]`` for the row's ``seq // 4`` blocks. Block ``b``'s key sits at compressed slot
    ``req_to_token[req, 4 b] // 4``. ``S^T = K Q^T`` on MFMA as in ``attention_scores`` (heads 4..15 read zero);
    the head sum is a quad reduction."""
    rows, lane = ctx.rows, ctx.lane
    g = lane >> 4
    r = lane & 15
    head_ok = r < IH
    for row_iv in range(0, rows, 1):
        row = fx.Int32(row_iv)
        seq = load(seq_lens, row * 8, words=1)
        req = load(req_pool, row * 8, words=1)
        blocks = seq // RATIO
        pages = (blocks + 15) // 16
        qv = [load(q, head_ok.select(((row * IH + r) * ID + 32 * jj + 8 * g) * 2, fx.Int32(DROP)), aux=ctx.aux)
              for jj in range_constexpr(4)]
        start = (ctx.wave_id + row * 61) % ctx.waves
        for page_iv in range(start, pages, ctx.waves):
            page = fx.Int32(page_iv)
            block = page * 16 + r
            ok = block < blocks
            token_slot = load(req_to_token, ok.select((req * rtt_stride + block * RATIO) * 4, fx.Int32(DROP)),
                              words=1)
            base = (token_slot // RATIO) * (ID * 2)
            kv = [load(compressed, ok.select(base + (32 * jj + 8 * g) * 2, fx.Int32(DROP)), aux=ctx.aux)
                  for jj in range_constexpr(4)]
            s = zeros4()
            for jj in range_constexpr(4):
                for h in range_constexpr(2):
                    s = mfma_bf16(bf16x4_half(kv[jj], h), bf16x4_half(qv[jj], h), s)
            bits = s.bitcast(fx.Int32)
            relu = [((bits[i] > fx.Int32(0)) & head_ok).select(bits[i], fx.Int32(0)).bitcast(F32)
                    for i in range_constexpr(4)]
            summed = []
            for i in range_constexpr(4):
                pair_sum = relu[i] + dpp(relu[i], DPP_XOR1)
                summed.append((pair_sum + dpp(pair_sum, DPP_XOR2)) * INDEX_SCALE)
            at = (row * max_blocks + page * 16 + 4 * g) * 4
            store(logits, (r == 0).select(at, fx.Int32(DROP)), words(summed, F32).bitcast(fx.Int32), aux=ctx.aux)


def _put(output, index, value, aux):
    store(output, fx.Int32(index) * 4, fx.Int32(value), aux=aux)


def _emit_ascending(output, chosen, merged, split, lane, threads, sync, aux):
    """``indexer_topk._emit_ascending`` with a cache policy for the token stores."""
    per = qtopk.TOPK // threads
    diagonal = lane * per
    count = fx.Int32(qtopk.TOPK) - split
    low = qtopk._max(diagonal - count, fx.Int32(0))
    high = qtopk._min(diagonal, split)
    for _ in range_constexpr(10):
        active = low < high
        mid = (low + high) >> 1
        a = qtopk._load16(chosen + qtopk._min(mid, fx.Int32(qtopk.TOPK - 1)) * 2)
        b = qtopk._load16(chosen + qtopk._min(qtopk._max(split + diagonal - mid - 1, fx.Int32(0)),
                                              fx.Int32(qtopk.TOPK - 1)) * 2)
        right = a < b
        low = (active & right).select(mid + 1, low)
        high = active.select(right.select(high, mid), high)
    i, j = low, diagonal - low
    for k in range_constexpr(per):
        a = (i < split).select(qtopk._load16(chosen + qtopk._min(i, fx.Int32(qtopk.TOPK - 1)) * 2), fx.Int32(1 << 16))
        b = (j < count).select(qtopk._load16(chosen + qtopk._min(split + j, fx.Int32(qtopk.TOPK - 1)) * 2),
                               fx.Int32(1 << 16))
        first = a < b
        qtopk._store16(merged + (diagonal + k) * 2, first.select(a, b))
        i = first.select(i + 1, i)
        j = first.select(j, j + 1)
    sync()
    for k in range_constexpr(qtopk.TOPK // threads):
        index = lane + k * threads
        value = qtopk._load16(merged + index * 2)
        tokens = words([value * RATIO + t for t in range_constexpr(RATIO)])
        store(output, index * (RATIO * 4), tokens, aux=aux)


@device
def _write_row(logits, stride, row, count, visible, length, output, space, aux):
    """``indexer_topk._write_row`` (decode CTA of 8 waves) over raw buffers: the row's top-512 blocks ascending
    (ties keep the lower block), 4 tokens each, then the causal tail, then -1."""
    blocks = count
    if count <= qtopk.TOPK:
        for token in range(space.thread, count * RATIO, space.threads):
            _put(output, token, token, aux)
    else:
        blocks = fx.Int32(qtopk.TOPK)
        base = row * stride * 4
        low, high, nan = qtopk._bounds(logits, base, count, space, None, row)
        split = qtopk._select(logits, base, count, space, low, high, nan)
        qtopk._row_sync(space)
        _emit_ascending(output, space.chosen, space.merged, split, space.thread, space.threads,
                        qtopk._sync_of(space), aux)
    tail_start = visible // RATIO * RATIO
    tail_count = visible - tail_start
    for index in range(blocks * RATIO + space.thread, fx.Int32(TOPK), fx.Int32(space.threads)):
        column = fx.Int32(index)
        offset = column - blocks * RATIO
        token = tail_start + offset
        value = ((offset < RATIO - 1) & (offset < tail_count) & (token < length)).select(token, fx.Int32(-1))
        _put(output, column, value, aux)


@device
def index_topk(ctx, logits, max_blocks, seq_lens, topk_out):
    """CTA item per row: SGLang's ``fast_topk`` + ``expand_qsa_block_indices`` (same selection up to ties, in
    ascending block order) with PyHIP's decode top-k (``indexer_topk``) on the 8-wave CTA."""
    rows = ctx.rows
    space = qtopk._space(qtopk.DECODE_WAVES, ctx.wave, ctx.lane, ctx.shared + ctx.wave * qtopk.HIST,
                         ctx.shared + qtopk.D_CAND, ctx.shared + qtopk.D_CHOSEN, ctx.shared, hist_base=ctx.shared,
                         red=ctx.shared + qtopk.D_RED)
    for row_iv in range(ctx.cta, rows, ctx.ctas):
        row = fx.Int32(row_iv)
        seq = scalar(load(seq_lens, row * 8, words=1))
        output = buffer_at(topk_out + fx.Int64(row) * (TOPK * 4), TOPK * 4)
        _write_row(logits, max_blocks, row, seq // RATIO, seq, seq, output, space, ctx.aux)
        qtopk._cta_sync()
