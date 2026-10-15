"""Phase device functions of the MonoKernel GDN-layer launches.

Each phase is one stage of the layer on the resident grid of ``_phase``; the GEMV phases keep the
FP32 summation order of the 4-wave stage kernels of ``hc``/``linear``/``gdn``/``moe`` and the
recurrence is shared with the stage launch (``gdn.gdn_recurrent``), so the fused launches match the
stage kernels bit for bit. Values written and read inside one launch use ``ctx.aux``
(device-coherent when fused); values that cross launches use plain accesses.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl

from ..config import QWEN38_FLASH_NEXT
from ._ir import (DPP_XOR1, DPP_XOR2, DROP, FP8_MAX, barrier, bf16_pair, bf16_round, bf16_word, dpp, exp_accurate,
                  fabs, fmax, fmin, fp8_pack, fp8_to_bf16_words, lds_load, lds_store, load, mfma_bf16, rne_bits, scalar,
                  scalar_f32, sigmoid, sigmoid_accurate, silu, softplus, sqrt_accurate, store, store_short, wave_max,
                  wave_sum, words, zeros4)
from ._phase import (HALF_WAVES, REGION_BYTES, STAGED_WAVES, THREADS, HalfGemm, act_rows, batches, buffer_at, device,
                     half_region, pipelined)
from .moe import MAX_ROWS, _bf16x4, _intrinsic, _lds_atomic, _ordered, _readlane, _wave_umax

C, H = QWEN38_FLASH_NEXT.hc_count, QWEN38_FLASH_NEXT.hidden
WIDTH, LOWRANK = C * H, QWEN38_FLASH_NEXT.hc_lowrank
DOWN_ROWS, KSPLIT = 336, 16
D, KERNEL = QWEN38_FLASH_NEXT.gdn_head_dim, QWEN38_FLASH_NEXT.conv_kernel
L2NORM_EPS = 1.0e-6
F32 = fx.Float32


def _bf16_values(word):
    out = []
    for i in range_constexpr(4):
        out += list(bf16_pair(word[i]))
    return out


def _pack_bf16(values):
    return words([bf16_word(values[2 * i], values[2 * i + 1]) for i in range_constexpr(len(values) // 2)])


def _half_word(rsrc, index, aux):
    """BF16 element ``index`` of a buffer as FP32."""
    lo, hi = bf16_pair(load(rsrc, (index >> 1) * 4, words=1, aux=aux))
    return ((index & 1) == 1).select(hi, lo)


# ---------------------------------------------------------------- hyper-connection


@device
def combine_norm(ctx, h_in, y, gain, nw, h_out, hn_out, combine, eps):
    """Wave item (row, branch): ``h' = bf16(h + y * gain)`` if ``combine`` and ``hn' = norm(h')``.
    Lane ``l`` covers the 8 columns of stage threads ``l + 64 i`` (``i < 5``)."""
    rows, lane = ctx.rows, ctx.lane
    for item_iv in range(ctx.wave_id, rows * C, ctx.waves):
        item = fx.Int32(item_iv)  # induction variables are bare arith values
        row = item >> 2
        branch = item & 3
        values = []
        for i in range_constexpr(5):
            values.append(_bf16_values(load(h_in, (row * WIDTH + branch * H + (lane + 64 * i) * 8) * 2)))
        if fx.const_expr(combine):
            g = load(gain, (row * C + branch) * 4, words=1).bitcast(F32)
            for i in range_constexpr(5):
                ys = _bf16_values(load(y, (row * H + (lane + 64 * i) * 8) * 2))
                values[i] = [bf16_round(values[i][e] + ys[e] * g) for e in range_constexpr(8)]
                store(h_out, (row * WIDTH + branch * H + (lane + 64 * i) * 8) * 2, _pack_bf16(values[i]))
        sums = []
        for i in range_constexpr(5):
            squares = F32(0.0)
            for v in values[i]:
                squares = squares + v * v
            # lane 0's sum, as the stage kernel's LDS hand-off (other lanes may round differently)
            sums.append(scalar_f32(wave_sum(squares, lane)))
        total = sums[0]
        for i in range_constexpr(1, 5):
            total = total + sums[i]
        mean = total * F32(1.0 / H)
        rstd = F32(rocdl.rsq(F32.ir_type, (mean + eps).ir_value()))
        for i in range_constexpr(5):
            column = branch * H + (lane + 64 * i) * 8
            gains = _bf16_values(load(nw, column * 2))
            normed = [(values[i][e] * rstd) * (1.0 + gains[e]) for e in range_constexpr(8)]
            store(hn_out, (row * WIDTH + column) * 2, _pack_bf16(normed), aux=ctx.aux)


def _pairs(ctx, items):
    """Half-item loop bounds: pair ``p`` holds items ``2 p`` (half 0) and ``2 p + 1`` (half 1)."""
    return ctx.cta, (items + 1) // 2 if isinstance(items, int) else (items + 1) >> 1, ctx.ctas


@device
def hc_down(ctx, hn, w, partial, tiles):
    """``partial[s] = hn[:, K_s] @ [down; inject]^T`` in FP32 for the 16 K splits (half item per
    16-row block and split)."""
    gemm = HalfGemm(WIDTH, tiles, "bf16", KSPLIT)
    run, stash, gather = gemm.partial, gemm.stash, gemm.gather  # loops may not call methods of outer locals
    blocks = DOWN_ROWS // 16
    items = blocks * KSPLIT
    rows, lane, wave = ctx.rows, ctx.lane, ctx.half_wave
    read = act_rows(hn, gemm.row_bytes, rows, lane, ctx.aux)
    first, stop, step = _pairs(ctx, items)
    for pair_iv in range(first, stop, step):
        pair = fx.Int32(pair_iv)
        item = pair * 2 + ctx.half
        valid = item < items
        block = item % blocks
        split = item // blocks
        region = half_region(ctx, pair)
        stash(region, run(read, w, block, split, lane, wave, valid), lane, wave)
        barrier()
        n0 = block * 16 + (lane >> 4) * 4
        for t in range_constexpr(tiles):
            if wave == t % STAGED_WAVES:
                total = gather(region, t, lane)
                token = 16 * t + (lane & 15)
                live = valid & (token < rows)
                at = live.select(((split * rows + token) * DOWN_ROWS + n0) * 4, fx.Int32(DROP))
                store(partial, at, total.bitcast(fx.Int32), aux=ctx.aux)


@device
def hc_lowrank(ctx, partial, low, gain):
    """Wave item (row, third): stage thread ``i = 64 * third + lane`` owns low-rank columns 2i, 2i + 1
    (``i < 160``) or inject columns (``i = 160, 161``)."""
    rows, lane = ctx.rows, ctx.lane
    for item_iv in range(ctx.wave_id, rows * 3, ctx.waves):
        item = fx.Int32(item_iv)
        row = item // 3
        i = (item % 3) * 64 + lane
        column = (i < LOWRANK // 2).select(2 * i, LOWRANK + 2 * (i - LOWRANK // 2))
        valid = i < (LOWRANK + C) // 2
        a, b = F32(0.0), F32(0.0)
        for s in range_constexpr(KSPLIT):
            pair = load(partial, valid.select(((s * rows + row) * DOWN_ROWS + column) * 4, fx.Int32(DROP)), words=2,
                        aux=ctx.aux).bitcast(F32)
            a, b = a + pair[0], b + pair[1]
        a, b = a * 0.25, b * 0.25
        low_at = (i < LOWRANK // 2).select((row * LOWRANK + column) * 2, fx.Int32(DROP))
        store(low, low_at, bf16_word(silu(a), silu(b)), aux=ctx.aux)
        gains = words((sigmoid(a) * 2.0, sigmoid(b) * 2.0), F32).bitcast(fx.Int32)
        gain_at = (valid & (i >= LOWRANK // 2)).select((row * C + column - LOWRANK) * 4, fx.Int32(DROP))
        store(gain, gain_at, gains)


@device
def hc_mix_up(ctx, low, w, hn, x, tiles):
    """Half item per 16 packed up rows: gates of column ``4 block + l // 16`` for its 4 branches, then
    ``x = bf16(sum_g sigmoid(gate_g) hn_g / 4)``."""
    gemm = HalfGemm(LOWRANK, tiles, "bf16")
    run, stash, gather = gemm.partial, gemm.stash, gemm.gather
    items = WIDTH // 16
    rows, lane, wave = ctx.rows, ctx.lane, ctx.half_wave
    read = act_rows(low, gemm.row_bytes, rows, lane, ctx.aux)
    first, stop, step = _pairs(ctx, items)
    for pair_iv in range(first, stop, step):
        pair = fx.Int32(pair_iv)
        block = pair * 2 + ctx.half
        region = half_region(ctx, pair)
        stash(region, run(read, w, block, 0, lane, wave, block < items), lane, wave)
        barrier()
        column = block * 4 + (lane >> 4)
        for t in range_constexpr(tiles):
            if wave == t % STAGED_WAVES:
                total = gather(region, t, lane)
                token = 16 * t + (lane & 15)
                mixed = F32(0.0)
                for g in range_constexpr(C):
                    mixed = mixed + sigmoid(total[g]) * _half_word(hn, token * WIDTH + g * H + column, ctx.aux)
                at = (token < rows).select((token * H + column) * 2, fx.Int32(DROP))
                store_short(x, at, rne_bits(mixed * 0.25).bitcast(fx.Int32) >> 16, aux=ctx.aux)


# ---------------------------------------------------------------- linear helpers


@device
def quant(ctx, src, dst, scales, k):
    """Wave item per row: AITER's per-token FP8 quantization (``linear.make_quant_rows``)."""
    chunks = k // 512
    rows, lane = ctx.rows, ctx.lane
    for row_iv in range(ctx.wave_id, rows, ctx.waves):
        row = fx.Int32(row_iv)
        values = []
        for c in range_constexpr(chunks):
            values += _bf16_values(load(src, (row * k + (c * 64 + lane) * 8) * 2, aux=ctx.aux))
        amax = F32(0.0)
        for value in values:
            amax = fmax(amax, fabs(value))
        amax = wave_max(amax, lane)
        scale = amax * F32(1.0 / FP8_MAX)
        scale = (amax > 0.0).select(scale, F32(1.0))
        inverse = F32(1.0) / scale
        for c in range_constexpr(chunks):
            packed = []
            for j in range_constexpr(2):
                quads = [fmin(fmax(values[c * 8 + 4 * j + e] * inverse, F32(-FP8_MAX)), F32(FP8_MAX))
                         for e in range_constexpr(4)]
                packed.append(fp8_pack(*quads))
            store(dst, row * k + (c * 64 + lane) * 8, words(packed), aux=ctx.aux)
        store(scales, (lane == 0).select(row * 4, fx.Int32(DROP)), scale.bitcast(fx.Int32), aux=ctx.aux)


def _store_tile(ctx, out, total, block, t, n_out, epilogue, aux, valid, xs=None, ws=None):
    """Linear epilogues of ``make_linear`` for token tile ``t`` of a 16-row block."""
    lane = ctx.lane
    n0 = block * 16 + (lane >> 4) * 4
    token = 16 * t + (lane & 15)
    live = valid & (token < ctx.rows) & (n0 < n_out)
    if fx.const_expr(epilogue == "f32"):
        store(out, live.select((token * n_out + n0) * 4, fx.Int32(DROP)), total.bitcast(fx.Int32), aux=aux)
    else:
        vals = [total[i] for i in range_constexpr(4)]
        if fx.const_expr(epilogue == "scaled"):
            x_scale = load(xs, token * 4, words=1, aux=ctx.aux).bitcast(F32)
            w_scale = load(ws, n0 * 4).bitcast(F32)
            vals = [vals[i] * x_scale * w_scale[i] for i in range_constexpr(4)]
        store(out, live.select((token * n_out + n0) * 2, fx.Int32(DROP)), _pack_bf16(vals), aux=aux)


# ---------------------------------------------------------------- gated deltanet


@device
def qkvz_ba(ctx, xq, xs, w, ws, cw, slots, state, qkvz, conv, x, w_ba, ba, n, conv_dim, ba_blocks, ba_out, tiles):
    """Half items: the ``n // 16`` blocks of ``in_proj_qkvz`` with the conv epilogue
    (``gdn.make_qkvz_conv``), then the ``ba_blocks`` blocks of the BF16 ``in_proj_ba``."""
    gemm = HalfGemm(H, tiles, "fp8")
    gemm_ba = HalfGemm(H, tiles, "bf16")
    run, run_ba, stash, gather = gemm.partial, gemm_ba.partial, gemm.stash, gemm.gather
    rows, lane, wave = ctx.rows, ctx.lane, ctx.half_wave
    read = act_rows(xq, gemm.row_bytes, rows, lane, ctx.aux)
    read_x = act_rows(x, gemm_ba.row_bytes, rows, lane, ctx.aux)
    blocks = n // 16
    items = blocks + ba_blocks
    first, stop, step = _pairs(ctx, items)
    for pair_iv in range(first, stop, step):
        pair = fx.Int32(pair_iv)
        item = pair * 2 + ctx.half
        valid = item < items
        region = half_region(ctx, pair)
        # the CTA barrier stays outside the (half-dependent) branches
        if item < blocks:
            stash(region, run(read, w, item, 0, lane, wave, valid), lane, wave)
        else:
            stash(region, run_ba(read_x, w_ba, item - blocks, 0, lane, wave, valid), lane, wave)
        barrier()
        if item < blocks:
            block = item
            n0 = block * 16 + (lane >> 4) * 4
            w_scale = load(ws, n0 * 4).bitcast(F32)
            for t in range_constexpr(tiles):
                if wave == t % STAGED_WAVES:
                    total = gather(region, t, lane)
                    token = 16 * t + (lane & 15)
                    live = token < rows
                    x_scale = load(xs, token * 4, words=1, aux=ctx.aux).bitcast(F32)
                    xv = [bf16_round(total[i] * x_scale * w_scale[i]) for i in range_constexpr(4)]
                    store(qkvz, live.select((token * n + n0) * 2, fx.Int32(DROP)), _pack_bf16(xv), aux=ctx.aux)
                    if block * 16 < conv_dim:
                        slot = load(slots, live.select(token * 4, fx.Int32(DROP)), words=1)
                        active = live & (slot >= 0)
                        offset = active.select((slot * conv_dim + n0) * 6, fx.Int32(DROP))
                        first_words = load(state, offset, words=4)
                        second = load(state, offset + 16, words=2)
                        taps = _bf16_values(first_words) + list(bf16_pair(second[0])) + list(bf16_pair(second[1]))
                        weights = (_bf16_values(load(cw, n0 * KERNEL * 2))
                                   + _bf16_values(load(cw, n0 * KERNEL * 2 + 16)))
                        outs, shifted = [], []
                        for c in range_constexpr(4):
                            window = [taps[3 * c], taps[3 * c + 1], taps[3 * c + 2], xv[c]]
                            total_c = bf16_round(window[0] * weights[KERNEL * c])
                            for j in range_constexpr(1, KERNEL):
                                total_c = total_c + bf16_round(window[j] * weights[KERNEL * c + j])
                            activated = total_c / (1.0 + exp_accurate(F32(0.0) - total_c))
                            outs.append(active.select(activated, F32(0.0)))
                            shifted += window[1:]
                        store(conv, live.select((token * conv_dim + n0) * 2, fx.Int32(DROP)), _pack_bf16(outs),
                              aux=ctx.aux)
                        packed = [bf16_word(shifted[2 * i], shifted[2 * i + 1]) for i in range_constexpr(6)]
                        store(state, offset, words(packed[:4]))
                        store(state, active.select(offset + 16, fx.Int32(DROP)), words(packed[4:]))
        else:
            for t in range_constexpr(tiles):
                if wave == t % STAGED_WAVES:
                    _store_tile(ctx, ba, gather(region, t, lane), item - blocks, t, ba_out, "bf16", ctx.aux, valid)


RECURRENT_THREADS = 4  # threads per state row


@device
def recurrent(ctx, conv, qkvz, ba, a_log, dtb, nw, ssm_address, slots, core, y, v_heads, k_heads, conv_dim, eps):
    """CTA item (row, v-head) of the gated delta rule: thread ``t`` owns state row ``v = t // 4`` and
    columns ``32 (t % 4) + [0, 32)``; the 4 threads of a row combine their dot products with DPP."""
    rows, lane = ctx.rows, ctx.lane
    key_dim = k_heads * D
    width = conv_dim + v_heads * D
    group = v_heads // k_heads
    scale = D ** -0.5
    span = D // RECURRENT_THREADS
    v = ctx.tid >> 2
    quarter = ctx.tid & 3
    items = rows * v_heads

    def quad_sum(value):
        value = value + dpp(value, DPP_XOR1)
        return value + dpp(value, DPP_XOR2)

    for item_iv in range(ctx.cta, items, ctx.ctas):
        item = fx.Int32(item_iv)
        row = item // v_heads
        head = item % v_heads
        kv = head // group
        slot = scalar(load(slots, row * 4, words=1))
        active = slot >= 0
        state_at = fx.Int64(active.select(slot, fx.Int32(0))) * (v_heads * D * D) + fx.Int64(head) * (D * D)
        ssm = buffer_at(ssm_address + state_at * 4, D * D * 4)
        row_offset = (v * D + quarter * span) * 4
        chunks = [load(ssm, active.select(row_offset + i * 16, fx.Int32(DROP))) for i in range_constexpr(span // 4)]
        base = row * conv_dim * 2
        q, k = [], []
        for i in range_constexpr(span // 8):
            q += _bf16_values(load(conv, base + (kv * D + quarter * span) * 2 + i * 16, aux=ctx.aux))
            k += _bf16_values(load(conv, base + (key_dim + kv * D + quarter * span) * 2 + i * 16, aux=ctx.aux))
        value = _half_word(conv, row * conv_dim + 2 * key_dim + head * D + v, ctx.aux)
        b = _half_word(ba, row * 2 * v_heads + head, ctx.aux)
        a = _half_word(ba, row * 2 * v_heads + v_heads + head, ctx.aux)
        dt = _half_word(dtb, head, 0)
        a_log_value = load(a_log, head * 4, words=1).bitcast(F32)
        q_sq, k_sq = F32(0.0), F32(0.0)
        for i in range_constexpr(span):
            q_sq = q_sq + q[i] * q[i]
            k_sq = k_sq + k[i] * k[i]
        q_den, k_den = sqrt_accurate(quad_sum(q_sq) + L2NORM_EPS), sqrt_accurate(quad_sum(k_sq) + L2NORM_EPS)
        qn = [q[i] / q_den * scale for i in range_constexpr(span)]
        kn = [k[i] / k_den for i in range_constexpr(span)]
        beta = bf16_round(sigmoid_accurate(b))
        g = (F32(0.0) - exp_accurate(a_log_value)) * softplus(a + dt)
        decay = exp_accurate(g)
        state = []
        for chunk in chunks:
            words4 = chunk.bitcast(F32)
            state += [words4[j] * decay for j in range_constexpr(4)]
        ks = F32(0.0)
        for i in range_constexpr(span):
            ks = ks + state[i] * kn[i]
        delta = (value - quad_sum(ks)) * beta
        state = [state[i] + delta * kn[i] for i in range_constexpr(span)]
        out = F32(0.0)
        for i in range_constexpr(span):
            out = out + state[i] * qn[i]
        core_value = active.select(bf16_round(quad_sum(out)), F32(0.0))
        for i in range_constexpr(span // 4):
            store(ssm, active.select(row_offset + i * 16, fx.Int32(DROP)),
                  words([state[4 * i + j] for j in range_constexpr(4)], F32).bitcast(fx.Int32))
        squares = wave_sum((quarter == 0).select(core_value * core_value, F32(0.0)), lane)
        if lane == 0:
            lds_store(ctx.shared + ctx.wave * 4, squares.bitcast(fx.Int32), words=1)
        barrier()
        total = lds_load(ctx.shared, words=1).bitcast(F32)
        for w in range_constexpr(1, THREADS // 64):
            total = total + lds_load(ctx.shared + w * 4, words=1).bitcast(F32)
        rstd = F32(rocdl.rsq(F32.ir_type, (total * (1.0 / D) + eps).ir_value()))
        weight = _half_word(nw, v, 0)
        z = _half_word(qkvz, row * width + conv_dim + head * D + v, ctx.aux)
        normed = core_value * rstd * weight * sigmoid_accurate(z)
        at = (quarter == 0).select((item * D + v) * 2, fx.Int32(DROP))
        store_short(core, at, core_value.bitcast(fx.Int32) >> 16)
        store_short(y, at, bf16_word(normed, normed).bitcast(fx.Int32) & 0xFFFF, aux=ctx.aux)
        barrier()  # the next item reuses the LDS words


@device
def out_proj(ctx, yq, ys, w, ws, out, k, tiles):
    """Half item per 16 output columns of the W8A8 ``out_proj`` (``linear.make_linear``)."""
    gemm = HalfGemm(k, tiles, "fp8")
    run, stash, gather = gemm.partial, gemm.stash, gemm.gather
    lane, wave = ctx.lane, ctx.half_wave
    read = act_rows(yq, gemm.row_bytes, ctx.rows, lane, ctx.aux)
    items = H // 16
    first, stop, step = _pairs(ctx, items)
    for pair_iv in range(first, stop, step):
        pair = fx.Int32(pair_iv)
        block = pair * 2 + ctx.half
        valid = block < items
        region = half_region(ctx, pair)
        stash(region, run(read, w, block, 0, lane, wave, valid), lane, wave)
        barrier()
        for t in range_constexpr(tiles):
            if wave == t % STAGED_WAVES:
                _store_tile(ctx, out, gather(region, t, lane), block, t, H, "scaled", 0, valid, ys, ws)


# ---------------------------------------------------------------- mixture of experts


@device
def router(ctx, x, w_router, w_shared, logits, shared_logit, n_experts, tiles):
    """Half items: the router blocks (BF16 logits), then the shared-expert gate block (FP32)."""
    gemm = HalfGemm(H, tiles, "bf16")
    run, stash, gather = gemm.partial, gemm.stash, gemm.gather
    lane, wave = ctx.lane, ctx.half_wave
    read = act_rows(x, gemm.row_bytes, ctx.rows, lane, ctx.aux)
    blocks = n_experts // 16
    items = blocks + 1
    first, stop, step = _pairs(ctx, items)
    for pair_iv in range(first, stop, step):
        pair = fx.Int32(pair_iv)
        item = pair * 2 + ctx.half
        valid = item < items
        region = half_region(ctx, pair)
        if item < blocks:
            stash(region, run(read, w_router, item, 0, lane, wave, valid), lane, wave)
        else:
            stash(region, run(read, w_shared, 0, 0, lane, wave, valid), lane, wave)
        barrier()
        if item < blocks:
            for t in range_constexpr(tiles):
                if wave == t % STAGED_WAVES:
                    _store_tile(ctx, logits, gather(region, t, lane), item, t, n_experts, "bf16", ctx.aux, valid)
        else:
            for t in range_constexpr(tiles):
                if wave == t % STAGED_WAVES:
                    _store_tile(ctx, shared_logit, gather(region, t, lane), 0, t, 16, "f32", ctx.aux, valid)


@device
def topk_rows(ctx, logits, shared_logit, ids, weights, n_experts, top_k):
    """Wave item per row: softmax, top-k (ties to the lower id), renormalized weights and the shared
    expert (``moe.make_topk_plan``)."""
    slots = top_k + 1
    rows, lane = ctx.rows, ctx.lane
    for row_iv in range(ctx.wave_id, rows, ctx.waves):
        row = fx.Int32(row_iv)
        word = load(logits, (row * n_experts + lane * 8) * 2, aux=ctx.aux)
        values, keys = [], []
        for i in range_constexpr(4):
            lo, hi = bf16_pair(word[i])
            values += [lo, hi]
            bits = word[i].bitcast(fx.Uint32)
            for h in range_constexpr(2):
                raw_bits = ((bits >> (16 * h)) & 0xFFFF).bitcast(fx.Int32)
                keys.append((_ordered(raw_bits) << 16) | (0xFFFF - (lane * 8 + 2 * i + h)))
        peak = values[0]
        for value in values[1:]:
            peak = (value > peak).select(value, peak)
        peak = wave_max(peak, lane)
        exps = [exp_accurate(value - peak) for value in values]
        local = exps[0]
        for value in exps[1:]:
            local = local + value
        total = wave_sum(local, lane)
        probs = [value / total for value in exps]
        chosen, chosen_p = [], []
        for _ in range_constexpr(top_k):
            best = keys[0]
            for key in keys[1:]:
                best = _intrinsic("llvm.umax.i32", fx.Int32, best, key)
            best = _wave_umax(best, lane)
            expert = 0xFFFF - (best & 0xFFFF)
            owner, index = expert >> 3, expert & 7
            mine = F32(0.0)
            for i in range_constexpr(8):
                mine = (index == i).select(probs[i], mine)
            chosen_p.append(_readlane(mine.bitcast(fx.Int32), owner).bitcast(F32))
            chosen.append(expert)
            keys = [((lane == owner) & (index == i)).select(fx.Int32(0), keys[i]) for i in range_constexpr(8)]
        mass = chosen_p[0]
        for p in chosen_p[1:]:
            mass = mass + p
        shared_value = load(shared_logit, row * 64, words=1, aux=ctx.aux).bitcast(F32)
        my_id, my_weight = fx.Int32(n_experts), sigmoid_accurate(shared_value)
        for k in range_constexpr(top_k):
            my_id = (lane == k).select(chosen[k], my_id)
            my_weight = (lane == k).select(chosen_p[k] / mass, my_weight)
        at = (lane < slots).select((row * slots + lane) * 4, fx.Int32(DROP))
        store(ids, at, my_id, aux=ctx.aux)
        store(weights, at, my_weight.bitcast(fx.Int32), aux=ctx.aux)


@device
def plan(ctx, ids, weights, plan_out, route_row, route_weight, n_experts, top_k):
    """CTA 0: group the routes by expert (``moe.make_topk_plan``): PLAN = [num_active, expert[],
    offset[], count[]], ``route_row``/``route_weight`` by position."""
    experts = n_experts + 1
    slots = top_k + 1
    per_lane = -(-experts // 64)
    ids_at, weights_at = 0, MAX_ROWS * slots
    counts_at = 2 * MAX_ROWS * slots
    masks_at = counts_at + experts
    offsets_at = masks_at + 2 * experts
    partial_at = offsets_at + experts
    shared, tid, lane = ctx.shared, ctx.tid, ctx.lane
    routes = ctx.rows * slots
    if ctx.cta == 0:
        for r_iv in range(tid, routes, THREADS):
            r = fx.Int32(r_iv)
            lds_store(shared + (ids_at + r) * 4, load(ids, r * 4, words=1, aux=ctx.aux), words=1)
            lds_store(shared + (weights_at + r) * 4, load(weights, r * 4, words=1, aux=ctx.aux), words=1)
        for e_iv in range(tid, 3 * experts, THREADS):
            lds_store(shared + (counts_at + fx.Int32(e_iv)) * 4, fx.Int32(0), words=1)
        barrier()
        for r_iv in range(tid, routes, THREADS):
            r = fx.Int32(r_iv)
            expert = lds_load(shared + (ids_at + r) * 4, words=1)
            row = r // slots
            _lds_atomic(llvm.AtomicBinOp.add, shared + (counts_at + expert) * 4, fx.Int32(1))
            _lds_atomic(llvm.AtomicBinOp._or, shared + (masks_at + 2 * expert + (row >> 5)) * 4,
                        fx.Int32(1) << (row & 31))
        barrier()
        if ctx.wave == 0:
            count_sum, active_sum = fx.Int32(0), fx.Int32(0)
            for i in range_constexpr(per_lane):
                e = lane * per_lane + i
                c = (e < experts).select(lds_load(shared + (counts_at + e) * 4, words=1), fx.Int32(0))
                count_sum = count_sum + c
                active_sum = active_sum + (c > 0).select(fx.Int32(1), fx.Int32(0))
            lds_store(shared + (partial_at + lane) * 4, count_sum, words=1)
            lds_store(shared + (partial_at + 64 + lane) * 4, active_sum, words=1)
            rocdl.wave_barrier()
            count_base, active_base = fx.Int32(0), fx.Int32(0)
            for j in range_constexpr(64):
                before = fx.Int32(j) < lane
                count_base = count_base + before.select(lds_load(shared + (partial_at + j) * 4, words=1), fx.Int32(0))
                active_base = active_base + before.select(lds_load(shared + (partial_at + 64 + j) * 4, words=1),
                                                          fx.Int32(0))
            for i in range_constexpr(per_lane):
                e = lane * per_lane + i
                c = (e < experts).select(lds_load(shared + (counts_at + e) * 4, words=1), fx.Int32(0))
                lds_store(shared + (offsets_at + e) * 4, count_base, words=1)
                is_active = c > 0
                store(plan_out, is_active.select((1 + active_base) * 4, fx.Int32(DROP)), fx.Int32(e), aux=ctx.aux)
                store(plan_out, is_active.select((1 + experts + active_base) * 4, fx.Int32(DROP)), count_base,
                      aux=ctx.aux)
                store(plan_out, is_active.select((1 + 2 * experts + active_base) * 4, fx.Int32(DROP)), c,
                      aux=ctx.aux)
                count_base = count_base + c
                active_base = active_base + is_active.select(fx.Int32(1), fx.Int32(0))
            store(plan_out, (lane == 63).select(fx.Int32(0), fx.Int32(DROP)), active_base, aux=ctx.aux)
        barrier()
        for r_iv in range(tid, routes, THREADS):
            r = fx.Int32(r_iv)
            expert = lds_load(shared + (ids_at + r) * 4, words=1)
            row = r // slots
            low_mask = lds_load(shared + (masks_at + 2 * expert) * 4, words=1)
            high_mask = lds_load(shared + (masks_at + 2 * expert + 1) * 4, words=1)
            below_low = (row >= 32).select(low_mask, low_mask & ((fx.Int32(1) << row) - 1))
            below_high = (row >= 32).select(high_mask & ((fx.Int32(1) << (row - 32)) - 1), fx.Int32(0))
            rank = _intrinsic("llvm.ctpop.i32", fx.Int32, below_low) + _intrinsic("llvm.ctpop.i32", fx.Int32,
                                                                                  below_high)
            position = lds_load(shared + (offsets_at + expert) * 4, words=1) + rank
            store(route_row, position * 4, row, aux=ctx.aux)
            store(route_weight, position * 4, lds_load(shared + (weights_at + r) * 4, words=1), aux=ctx.aux)


@device
def moe_up(ctx, x, w_address, ws_address, plan_in, route_row, mid, n_experts, top_k, inter, tiles):
    """Half item (active expert, 16-row block of the intermediate dim), K split over the half's 4
    waves like ``moe.make_moe_up``: W8A16 gate/up of the expert's rows, ``mid = bf16(silu(g s_g) (u s_u))``."""
    experts = n_experts + 1
    per_wave = H // 64 // STAGED_WAVES
    blocks = inter // 16
    groups = batches(per_wave, tiles, 2)
    lane, wave = ctx.lane, ctx.half_wave
    region = ctx.shared + ctx.half * (2 * REGION_BYTES)  # gate and up partials: 4 x 2 x tiles KiB
    num_active = scalar(load(plan_in, 0, words=1, aux=ctx.aux))
    items = blocks * num_active
    for pair_iv in range(ctx.cta, (items + 1) >> 1, ctx.ctas):
        pair = fx.Int32(pair_iv)
        item = pair * 2 + ctx.half
        valid = item < items
        active = valid.select(item // blocks, fx.Int32(0))
        block = item % blocks
        expert = scalar(load(plan_in, (1 + active) * 4, words=1, aux=ctx.aux))
        offset = scalar(load(plan_in, (1 + experts + active) * 4, words=1, aux=ctx.aux))
        count = valid.select(scalar(load(plan_in, (1 + 2 * experts + active) * 4, words=1, aux=ctx.aux)), fx.Int32(0))
        w = buffer_at(w_address + fx.Int64(expert) * (2 * inter * H), 2 * inter * H)
        token_rows = []
        for t in range_constexpr(tiles):
            position = 16 * t + (lane & 15)
            live = position < count
            row = load(route_row, live.select((offset + position) * 4, fx.Int32(DROP)), words=1, aux=ctx.aux)
            token_rows.append(live.select(row * (H * 2), fx.Int32(DROP)))
        gate = [zeros4() for _ in range_constexpr(tiles)]
        up = [zeros4() for _ in range_constexpr(tiles)]

        def issue(group):
            weights, acts = [], []
            for j in group:
                q = wave + STAGED_WAVES * j
                g_at = valid.select(block * (16 * H) + q * 1024 + lane * 16, fx.Int32(DROP))
                u_at = valid.select((inter // 16 + block) * (16 * H) + q * 1024 + lane * 16, fx.Int32(DROP))
                weights.append((load(w, g_at), load(w, u_at)))
            for j in group:
                q = wave + STAGED_WAVES * j
                per_tile = []
                for t in range_constexpr(tiles):
                    at = token_rows[t] + (q * 64 + (lane >> 4) * 16) * 2
                    a0 = load(x, at, aux=ctx.aux)
                    a1 = load(x, (token_rows[t] < DROP).select(at + 16, fx.Int32(DROP)), aux=ctx.aux)
                    per_tile.append([a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]])
                acts.append(per_tile)
            return weights, acts

        def consume(group, loaded):
            weights, acts = loaded
            for i in range_constexpr(len(group)):
                g_word, u_word = weights[i]
                g_bf = [fp8_to_bf16_words(g_word[e]) for e in range_constexpr(4)]
                u_bf = [fp8_to_bf16_words(u_word[e]) for e in range_constexpr(4)]
                for t in range_constexpr(tiles):
                    a = acts[i][t]
                    for s in range_constexpr(4):
                        operand = _bf16x4(a[2 * s], a[2 * s + 1])
                        gate[t] = mfma_bf16(_bf16x4(*g_bf[s]), operand, gate[t])
                        up[t] = mfma_bf16(_bf16x4(*u_bf[s]), operand, up[t])

        pipelined(groups, issue, consume)
        for t in range_constexpr(tiles):
            lds_store(region + (((wave * 2 + 0) * tiles + t) * 64 + lane) * 16, gate[t].bitcast(fx.Int32))
            lds_store(region + (((wave * 2 + 1) * tiles + t) * 64 + lane) * 16, up[t].bitcast(fx.Int32))
        barrier()
        channel = block * 16 + (lane >> 4) * 4
        scales = buffer_at(ws_address + fx.Int64(expert) * (2 * inter * 4), 2 * inter * 4)
        g_scale = load(scales, channel * 4).bitcast(F32)
        u_scale = load(scales, (inter + channel) * 4).bitcast(F32)
        for t in range_constexpr(tiles):
            if wave == t % STAGED_WAVES:
                g_total = lds_load(region + ((0 * tiles + t) * 64 + lane) * 16).bitcast(F32)
                u_total = lds_load(region + ((1 * tiles + t) * 64 + lane) * 16).bitcast(F32)
                for other in range_constexpr(1, STAGED_WAVES):
                    g_total = g_total + lds_load(region + (((other * 2) * tiles + t) * 64 + lane) * 16).bitcast(F32)
                    u_total = u_total + lds_load(
                        region + (((other * 2 + 1) * tiles + t) * 64 + lane) * 16).bitcast(F32)
                values = []
                for i in range_constexpr(4):
                    gv = g_total[i] * g_scale[i]
                    uv = u_total[i] * u_scale[i]
                    values.append(gv / (1.0 + exp_accurate(F32(0.0) - gv)) * uv)
                position = 16 * t + (lane & 15)
                target = (position < count).select(((offset + position) * inter + channel) * 2, fx.Int32(DROP))
                store(mid, target, _pack_bf16(values), aux=ctx.aux)
        barrier()  # the next item reuses the region


@device
def moe_down(ctx, mid, w_address, ws_address, plan_in, route_row, route_weight, out, n_experts, top_k, inter,
             tiles):
    """Half item per 16 output columns (``moe.make_moe_down``): the half's wave ``w`` walks active
    experts ``w, w + 4, ...`` into its LDS accumulator, summed in wave order."""
    experts = n_experts + 1
    loads = inter // 64
    acc_words = MAX_ROWS * 16
    rows, lane = ctx.rows, ctx.lane
    region = ctx.shared + ctx.half * (HALF_WAVES * acc_words * 4)
    mine = region + ctx.half_wave * (acc_words * 4)
    num_active = scalar(load(plan_in, 0, words=1, aux=ctx.aux))
    blocks = H // 16
    for pair_iv in range(ctx.cta, blocks >> 1, ctx.ctas):
        block = fx.Int32(pair_iv) * 2 + ctx.half
        for i in range_constexpr(HALF_WAVES * acc_words // (256 * 4)):
            lds_store(region + ((i * 256 + ctx.half_tid) * 4) * 4, words([fx.Int32(0)] * 4))
        barrier()
        column = block * 16 + (lane >> 4) * 4
        for active_iv in range(ctx.half_wave, num_active, HALF_WAVES):
            active = fx.Int32(active_iv)
            expert = scalar(load(plan_in, (1 + active) * 4, words=1, aux=ctx.aux))
            offset = scalar(load(plan_in, (1 + experts + active) * 4, words=1, aux=ctx.aux))
            count = scalar(load(plan_in, (1 + 2 * experts + active) * 4, words=1, aux=ctx.aux))
            w = buffer_at(w_address + fx.Int64(expert) * (H * inter), H * inter)
            scale = load(buffer_at(ws_address + fx.Int64(expert) * (H * 4), H * 4), column * 4).bitcast(F32)
            acc = [zeros4() for _ in range_constexpr(tiles)]
            for q in range_constexpr(loads):
                w_word = load(w, block * (16 * inter) + q * 1024 + lane * 16)
                w_bf = [fp8_to_bf16_words(w_word[i]) for i in range_constexpr(4)]
                for t in range_constexpr(tiles):
                    position = 16 * t + (lane & 15)
                    at = (position < count).select(((offset + position) * inter + q * 64 + (lane >> 4) * 16) * 2,
                                                   fx.Int32(DROP))
                    a0 = load(mid, at, aux=ctx.aux)
                    a1 = load(mid, (position < count).select(at + 16, fx.Int32(DROP)), aux=ctx.aux)
                    acts = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]]
                    for s in range_constexpr(4):
                        acc[t] = mfma_bf16(_bf16x4(*w_bf[s]), _bf16x4(acts[2 * s], acts[2 * s + 1]), acc[t])
            for t in range_constexpr(tiles):
                position = 16 * t + (lane & 15)
                valid = position < count
                route = (offset + position) * 4
                row = load(route_row, valid.select(route, fx.Int32(DROP)), words=1, aux=ctx.aux)
                weight = load(route_weight, valid.select(route, fx.Int32(DROP)), words=1, aux=ctx.aux).bitcast(F32)
                address = mine + (row * 16 + (lane >> 4) * 4) * 4
                if valid:
                    old = lds_load(address).bitcast(F32)
                    new = [old[i] + acc[t][i] * scale[i] * weight for i in range_constexpr(4)]
                    lds_store(address, words(new, F32).bitcast(fx.Int32))
                rocdl.wave_barrier()
        barrier()
        for i in range_constexpr(MAX_ROWS * 16 // 256):
            cell = i * 256 + ctx.half_tid
            row, col = cell >> 4, cell & 15
            total = lds_load(region + cell * 4, words=1).bitcast(F32)
            for other in range_constexpr(1, HALF_WAVES):
                total = total + lds_load(region + (other * acc_words + cell) * 4, words=1).bitcast(F32)
            target = (row < rows).select((row * H + block * 16 + col) * 2, fx.Int32(DROP))
            store_short(out, target, bf16_word(total, total).bitcast(fx.Int32) & 0xFFFF)
        barrier()  # the next item clears the accumulators
