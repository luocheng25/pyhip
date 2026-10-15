"""MoE decode stages: routing plan, W8A16 gate/up and down over SGLang's shuffled expert banks.

``topk_plan`` (one 1024-thread CTA, rows <= 64): per row (one wave) softmax of the BF16 router logits,
top-k by logit with ties to the lower id (equivalently by probability, like a stable sort),
renormalized weights, and the shared expert ``n_experts`` with weight ``sigmoid(shared_logit)``.
Then the routes are grouped by expert: per expert a 64-bit row mask gives each route the position
``offset[e] + popcount(mask[e] & ((1 << row) - 1))``, so every expert lists its rows in increasing
order; active experts are listed in increasing id.

``moe_up``: one CTA per (16-row block of the intermediate dim, active expert); BF16 activations of the
expert's rows (MFMA columns) against the FP8 gate and up rows converted to BF16 (exact), then
``mid = bf16(silu(g * s_g) * (u * s_u))`` per route.

``moe_down``: one CTA per 16 output columns; wave ``w`` walks active experts ``w, w + 4, ...``, adds
``(mid @ down^T) * s_d * route_weight`` into a per-wave LDS accumulator row of each route, and the
4 accumulators are summed in a fixed order into the BF16 partial output.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl

from ._ir import (DROP, barrier, bf16_pair, bf16_word, buffer, exp_accurate, fp8_to_bf16_words, lds_base,
                  lds_load, lds_store, load, mfma_bf16, scalar, sigmoid_accurate, store, store_short, wave_max,
                  wave_sum, words, zeros4)
from .linear import WAVES, _call, tiles_for

TOPK_THREADS = 1024
MAX_ROWS = 64


def _lds_ptr(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_atomic(op, address, value):
    llvm.AtomicRMWOp(op, _lds_ptr(address), fx.Int32(value).ir_value(), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup")


def _intrinsic(name, dtype, *args):
    values = [a.ir_value() if hasattr(a, "ir_value") else fx.arith.unwrap(a) for a in args]
    return dtype(llvm.call_intrinsic(dtype.ir_type, name, values, [], []))


def _readlane(value, lane):
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, fx.Int32(value).ir_value(), fx.Int32(lane).ir_value()))


def _wave_umax(value, lane):
    """Unsigned max of an i32 over the 64 lanes (all lanes get it)."""
    for offset in (1, 2, 4, 8, 16, 32):
        other = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, ((lane ^ offset) * 4).ir_value(), value.ir_value()))
        value = _intrinsic("llvm.umax.i32", fx.Int32, value, other)
    return value


def _ordered(bits):
    """Monotone unsigned 16-bit key of BF16 bits (-0 ranks as +0)."""
    bits = (bits == 0x8000).select(fx.Int32(0), bits)
    negative = (bits & 0x8000) != 0
    return negative.select((~bits) & 0xFFFF, bits | 0x8000)


@cache
def make_topk_plan(n_experts, top_k):
    experts = n_experts + 1  # with the shared expert
    slots = top_k + 1
    assert n_experts == 512 and slots <= 64
    per_lane = -(-experts // 64)
    # LDS words: ids, weights, counts, masks (2 per expert), offsets, lane partials (2 x 64).
    ids_at, weights_at = 0, MAX_ROWS * slots
    counts_at = 2 * MAX_ROWS * slots
    masks_at = counts_at + experts
    offsets_at = masks_at + 2 * experts
    partial_at = offsets_at + experts
    lds_words = partial_at + 2 * 64

    @flyc.kernel(known_block_size=[TOPK_THREADS, 1, 1])
    def topk_plan(LOGITS: fx.Tensor, SHARED: fx.Tensor, IDS: fx.Tensor, WEIGHTS: fx.Tensor, PLAN: fx.Tensor,
                  ROUTE_ROW: fx.Tensor, ROUTE_WEIGHT: fx.Tensor, rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_words, 16]).peek()
        shared = lds_base(storage, lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        routes = rows * slots
        for row in range(wave, rows, TOPK_THREADS // 64):
            word = load(buffer(LOGITS, rows * n_experts * 2), (row * n_experts + lane * 8) * 2)
            logits, keys = [], []
            for i in range_constexpr(4):
                lo, hi = bf16_pair(word[i])
                logits += [lo, hi]
                bits = word[i].bitcast(fx.Uint32)
                for h in range_constexpr(2):
                    raw = ((bits >> (16 * h)) & 0xFFFF).bitcast(fx.Int32)
                    keys.append((_ordered(raw) << 16) | (0xFFFF - (lane * 8 + 2 * i + h)))
            peak = logits[0]
            for value in logits[1:]:
                peak = (value > peak).select(value, peak)
            peak = wave_max(peak, lane)
            exps = [exp_accurate(value - peak) for value in logits]
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
                mine = fx.Float32(0.0)
                for i in range_constexpr(8):
                    mine = (index == i).select(probs[i], mine)
                chosen_p.append(_readlane(mine.bitcast(fx.Int32), owner).bitcast(fx.Float32))
                chosen.append(expert)
                keys = [((lane == owner) & (index == i)).select(fx.Int32(0), keys[i]) for i in range_constexpr(8)]
            mass = chosen_p[0]
            for p in chosen_p[1:]:
                mass = mass + p
            shared_logit = load(buffer(SHARED, rows * 16 * 4), row * 64, words=1).bitcast(fx.Float32)
            my_id, my_weight = fx.Int32(n_experts), sigmoid_accurate(shared_logit)
            for k in range_constexpr(top_k):
                my_id = (lane == k).select(chosen[k], my_id)
                my_weight = (lane == k).select(chosen_p[k] / mass, my_weight)
            route = row * slots + lane
            in_row = lane < slots
            store(buffer(IDS, routes * 4), in_row.select(route * 4, fx.Int32(DROP)), my_id)
            store(buffer(WEIGHTS, routes * 4), in_row.select(route * 4, fx.Int32(DROP)), my_weight.bitcast(fx.Int32))
            if lane < slots:
                lds_store(shared + (ids_at + route) * 4, my_id, words=1)
                lds_store(shared + (weights_at + route) * 4, my_weight.bitcast(fx.Int32), words=1)
        for e in range(tid, 3 * experts, TOPK_THREADS):
            lds_store(shared + (counts_at + e) * 4, fx.Int32(0), words=1)
        barrier()
        for r in range(tid, routes, TOPK_THREADS):
            expert = lds_load(shared + (ids_at + r) * 4, words=1)
            row = r // slots
            _lds_atomic(llvm.AtomicBinOp.add, shared + (counts_at + expert) * 4, fx.Int32(1))
            _lds_atomic(llvm.AtomicBinOp._or, shared + (masks_at + 2 * expert + (row >> 5)) * 4,
                        fx.Int32(1) << (row & 31))
        barrier()
        if wave == 0:
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
            plan = buffer(PLAN, (1 + 3 * experts) * 4)
            for i in range_constexpr(per_lane):
                e = lane * per_lane + i
                c = (e < experts).select(lds_load(shared + (counts_at + e) * 4, words=1), fx.Int32(0))
                lds_store(shared + (offsets_at + e) * 4, count_base, words=1)
                active = c > 0
                # PLAN = [num_active, expert[...], offset[...], count[...]] (experts entries each).
                store(plan, active.select((1 + active_base) * 4, fx.Int32(DROP)), fx.Int32(e))
                store(plan, active.select((1 + experts + active_base) * 4, fx.Int32(DROP)), count_base)
                store(plan, active.select((1 + 2 * experts + active_base) * 4, fx.Int32(DROP)), c)
                count_base = count_base + c
                active_base = active_base + active.select(fx.Int32(1), fx.Int32(0))
            store(plan, (lane == 63).select(fx.Int32(0), fx.Int32(DROP)), active_base)
        barrier()
        for r in range(tid, routes, TOPK_THREADS):
            expert = lds_load(shared + (ids_at + r) * 4, words=1)
            row = r // slots
            low = lds_load(shared + (masks_at + 2 * expert) * 4, words=1)
            high = lds_load(shared + (masks_at + 2 * expert + 1) * 4, words=1)
            below_low = (row >= 32).select(low, low & ((fx.Int32(1) << row) - 1))
            below_high = (row >= 32).select(high & ((fx.Int32(1) << (row - 32)) - 1), fx.Int32(0))
            rank = _intrinsic("llvm.ctpop.i32", fx.Int32, below_low) + _intrinsic("llvm.ctpop.i32", fx.Int32,
                                                                                  below_high)
            position = lds_load(shared + (offsets_at + expert) * 4, words=1) + rank
            store(buffer(ROUTE_ROW, routes * 4), position * 4, row)
            store(buffer(ROUTE_WEIGHT, routes * 4), position * 4, lds_load(shared + (weights_at + r) * 4, words=1))

    @flyc.jit
    def launch(LOGITS: fx.Tensor, SHARED: fx.Tensor, IDS: fx.Tensor, WEIGHTS: fx.Tensor, PLAN: fx.Tensor,
               ROUTE_ROW: fx.Tensor, ROUTE_WEIGHT: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            topk_plan(LOGITS, SHARED, IDS, WEIGHTS, PLAN, ROUTE_ROW, ROUTE_WEIGHT, rows).launch(
                grid=(1, 1, 1), block=(TOPK_THREADS, 1, 1), stream=stream)

    return launch


def _bf16x4(word_a, word_b):
    return words((word_a, word_b)).bitcast(fx.Int16)


@cache
def make_moe_up(n_experts, top_k, inter, hidden, tiles):
    experts = n_experts + 1
    slots = top_k + 1
    row_bytes = hidden  # FP8 weights
    loads = row_bytes // 64
    assert loads % WAVES == 0 and inter % 16 == 0
    per_wave = loads // WAVES
    lds_words = WAVES * 2 * tiles * 64 * 4

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def moe_up(X: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, PLAN: fx.Tensor, ROUTE_ROW: fx.Tensor, MID: fx.Tensor,
               rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_words, 16]).peek()
        shared = lds_base(storage, lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        active = fx.Int32(fx.block_idx.y)
        plan = buffer(PLAN, (1 + 3 * experts) * 4)
        if active < scalar(load(plan, 0, words=1)):
            expert = scalar(load(plan, (1 + active) * 4, words=1))
            offset = scalar(load(plan, (1 + experts + active) * 4, words=1))
            count = scalar(load(plan, (1 + 2 * experts + active) * 4, words=1))
            routes = rows * slots
            route_rows = buffer(ROUTE_ROW, routes * 4)
            x = buffer(X, rows * hidden * 2)
            w = buffer(W, fx.Int32(2 * inter * hidden), fx.Int64(expert) * (2 * inter * hidden))
            token_rows = []
            for t in range_constexpr(tiles):
                position = 16 * t + (lane & 15)
                valid = position < count
                row = load(route_rows, valid.select((offset + position) * 4, fx.Int32(DROP)), words=1)
                token_rows.append(valid.select(row * (hidden * 2), fx.Int32(DROP)))
            gate = [zeros4() for _ in range_constexpr(tiles)]
            up = [zeros4() for _ in range_constexpr(tiles)]
            for j in range_constexpr(per_wave):
                q = wave + WAVES * j
                g_word = load(w, block * (16 * row_bytes) + q * 1024 + lane * 16)
                u_word = load(w, (inter // 16 + block) * (16 * row_bytes) + q * 1024 + lane * 16)
                g_bf = [fp8_to_bf16_words(g_word[i]) for i in range_constexpr(4)]
                u_bf = [fp8_to_bf16_words(u_word[i]) for i in range_constexpr(4)]
                for t in range_constexpr(tiles):
                    base = token_rows[t] + (q * 64 + (lane >> 4) * 16) * 2
                    a0 = load(x, base)
                    a1 = load(x, (token_rows[t] < DROP).select(base + 16, fx.Int32(DROP)))
                    acts = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]]
                    for s in range_constexpr(4):
                        operand = _bf16x4(acts[2 * s], acts[2 * s + 1])
                        gate[t] = mfma_bf16(_bf16x4(*g_bf[s]), operand, gate[t])
                        up[t] = mfma_bf16(_bf16x4(*u_bf[s]), operand, up[t])
            for t in range_constexpr(tiles):
                lds_store(shared + (((wave * 2 + 0) * tiles + t) * 64 + lane) * 16, gate[t].bitcast(fx.Int32))
                lds_store(shared + (((wave * 2 + 1) * tiles + t) * 64 + lane) * 16, up[t].bitcast(fx.Int32))
            barrier()
            channel = block * 16 + (lane >> 4) * 4
            scales = buffer(WS, fx.Int32(2 * inter * 4), fx.Int64(expert) * (2 * inter))
            g_scale = load(scales, channel * 4).bitcast(fx.Float32)
            u_scale = load(scales, (inter + channel) * 4).bitcast(fx.Float32)
            mid = buffer(MID, routes * inter * 2)
            for t in range_constexpr(tiles):
                if wave == t % WAVES:
                    g_total = lds_load(shared + ((0 * tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
                    u_total = lds_load(shared + ((1 * tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
                    for other in range_constexpr(1, WAVES):
                        g_total = g_total + lds_load(shared + (((other * 2) * tiles + t) * 64 + lane) * 16).bitcast(
                            fx.Float32)
                        u_total = u_total + lds_load(
                            shared + (((other * 2 + 1) * tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
                    values = []
                    for i in range_constexpr(4):
                        g = g_total[i] * g_scale[i]
                        u = u_total[i] * u_scale[i]
                        values.append(g / (1.0 + exp_accurate(fx.Float32(0.0) - g)) * u)
                    position = 16 * t + (lane & 15)
                    target = (position < count).select(((offset + position) * inter + channel) * 2, fx.Int32(DROP))
                    store(mid, target, words((bf16_word(values[0], values[1]), bf16_word(values[2], values[3]))))

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, PLAN: fx.Tensor, ROUTE_ROW: fx.Tensor, MID: fx.Tensor,
               rows: fx.Int32, max_active: fx.Int32, stream: fx.Stream):
        if rows > 0:
            moe_up(X, W, WS, PLAN, ROUTE_ROW, MID, rows).launch(grid=(inter // 16, max_active, 1),
                                                                block=(64 * WAVES, 1, 1), stream=stream)

    return launch


@cache
def make_moe_down(n_experts, top_k, inter, hidden, tiles):
    experts = n_experts + 1
    slots = top_k + 1
    row_bytes = inter  # FP8 weights [hidden, inter]
    loads = row_bytes // 64 if inter % 64 == 0 else None
    assert loads is not None
    acc_words = MAX_ROWS * 16  # one wave's FP32 accumulator [row][16 columns]
    lds_words = WAVES * acc_words

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def moe_down(MID: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, PLAN: fx.Tensor, ROUTE_ROW: fx.Tensor,
                 ROUTE_WEIGHT: fx.Tensor, OUT: fx.Tensor, rows: fx.Int32):
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_words, 16]).peek()
        shared = lds_base(storage, lds_words)
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid & 63, scalar(tid >> 6)
        block = fx.Int32(fx.block_idx.x)
        for i in range_constexpr(lds_words // (64 * WAVES * 4)):
            lds_store(shared + ((i * 64 * WAVES + tid) * 4) * 4, words([fx.Int32(0)] * 4))
        barrier()
        plan = buffer(PLAN, (1 + 3 * experts) * 4)
        num_active = scalar(load(plan, 0, words=1))
        routes = rows * slots
        route_rows = buffer(ROUTE_ROW, routes * 4)
        route_weights = buffer(ROUTE_WEIGHT, routes * 4)
        mid = buffer(MID, routes * inter * 2)
        column = block * 16 + (lane >> 4) * 4
        mine = shared + wave * (acc_words * 4)
        for active in range(wave, num_active, WAVES):
            expert = scalar(load(plan, (1 + active) * 4, words=1))
            offset = scalar(load(plan, (1 + experts + active) * 4, words=1))
            count = scalar(load(plan, (1 + 2 * experts + active) * 4, words=1))
            w = buffer(W, fx.Int32(hidden * inter), fx.Int64(expert) * (hidden * inter))
            scale = load(buffer(WS, fx.Int32(hidden * 4), fx.Int64(expert) * hidden), column * 4).bitcast(fx.Float32)
            acc = [zeros4() for _ in range_constexpr(tiles)]
            for q in range_constexpr(loads):
                w_word = load(w, block * (16 * row_bytes) + q * 1024 + lane * 16)
                w_bf = [fp8_to_bf16_words(w_word[i]) for i in range_constexpr(4)]
                for t in range_constexpr(tiles):
                    position = 16 * t + (lane & 15)
                    base = (position < count).select(((offset + position) * inter + q * 64 + (lane >> 4) * 16) * 2,
                                                     fx.Int32(DROP))
                    a0 = load(mid, base)
                    a1 = load(mid, (position < count).select(base + 16, fx.Int32(DROP)))
                    acts = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]]
                    for s in range_constexpr(4):
                        acc[t] = mfma_bf16(_bf16x4(*w_bf[s]), _bf16x4(acts[2 * s], acts[2 * s + 1]), acc[t])
            for t in range_constexpr(tiles):
                position = 16 * t + (lane & 15)
                valid = position < count
                route = (offset + position) * 4
                row = load(route_rows, valid.select(route, fx.Int32(DROP)), words=1)
                weight = load(route_weights, valid.select(route, fx.Int32(DROP)), words=1).bitcast(fx.Float32)
                address = mine + (row * 16 + (lane >> 4) * 4) * 4
                if valid:
                    old = lds_load(address).bitcast(fx.Float32)
                    new = [old[i] + acc[t][i] * scale[i] * weight for i in range_constexpr(4)]
                    lds_store(address, words(new, fx.Float32).bitcast(fx.Int32))
                rocdl.wave_barrier()
        barrier()
        out = buffer(OUT, rows * hidden * 2)
        for i in range_constexpr(MAX_ROWS * 16 // (64 * WAVES)):
            cell = i * 64 * WAVES + tid
            row, col = cell >> 4, cell & 15
            total = lds_load(shared + cell * 4, words=1).bitcast(fx.Float32)
            for other in range_constexpr(1, WAVES):
                total = total + lds_load(shared + (other * acc_words + cell) * 4, words=1).bitcast(fx.Float32)
            target = (row < rows).select((row * hidden + block * 16 + col) * 2, fx.Int32(DROP))
            store_short(out, target, bf16_word(total, total).bitcast(fx.Int32) & 0xFFFF)

    @flyc.jit
    def launch(MID: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, PLAN: fx.Tensor, ROUTE_ROW: fx.Tensor,
               ROUTE_WEIGHT: fx.Tensor, OUT: fx.Tensor, rows: fx.Int32, stream: fx.Stream):
        if rows > 0:
            moe_down(MID, W, WS, PLAN, ROUTE_ROW, ROUTE_WEIGHT, OUT, rows).launch(
                grid=(hidden // 16, 1, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------- host


class MoEWorkspace:
    def __init__(self, rows, n_experts, top_k, inter, device):
        slots = top_k + 1
        routes = rows * slots
        self.n_experts, self.top_k, self.inter = n_experts, top_k, inter
        self.logits = torch.empty(rows, n_experts, dtype=torch.bfloat16, device=device)
        self.shared = torch.empty(rows, 16, dtype=torch.float32, device=device)
        self.ids = torch.empty(rows, slots, dtype=torch.int32, device=device)
        self.weights = torch.empty(rows, slots, dtype=torch.float32, device=device)
        self.plan = torch.empty(1 + 3 * (n_experts + 1), dtype=torch.int32, device=device)
        self.route_row = torch.empty(routes, dtype=torch.int32, device=device)
        self.route_weight = torch.empty(routes, dtype=torch.float32, device=device)
        self.mid = torch.empty(routes, inter, dtype=torch.bfloat16, device=device)


def topk_plan(work, rows):
    stream = torch.cuda.current_stream(work.logits.device)
    args = (work.logits.view(-1), work.shared.view(-1), work.ids.view(-1), work.weights.view(-1), work.plan,
            work.route_row, work.route_weight, rows, stream)
    with torch.cuda.device(work.logits.device):
        _call(("topk_plan", work.logits.device, work.n_experts, work.top_k), make_topk_plan(work.n_experts, work.top_k),
              args)


def moe_up(x, w13, w13_scale, work):
    rows, hidden = x.shape
    tiles = tiles_for(rows)
    max_active = min(rows * (work.top_k + 1), work.n_experts + 1)
    stream = torch.cuda.current_stream(x.device)
    args = (x.view(-1), w13.view(torch.uint8).view(-1), w13_scale.view(-1), work.plan, work.route_row,
            work.mid.view(-1), rows, max_active, stream)
    with torch.cuda.device(x.device):
        _call(("moe_up", x.device, work.n_experts, work.top_k, work.inter, hidden, tiles),
              make_moe_up(work.n_experts, work.top_k, work.inter, hidden, tiles), args)


def moe_down(w2, w2_scale, work, out):
    rows, hidden = out.shape
    tiles = tiles_for(rows)
    stream = torch.cuda.current_stream(out.device)
    args = (work.mid.view(-1), w2.view(torch.uint8).view(-1), w2_scale.view(-1), work.plan, work.route_row,
            work.route_weight, out.view(-1), rows, stream)
    with torch.cuda.device(out.device):
        _call(("moe_down", out.device, work.n_experts, work.top_k, work.inter, hidden, tiles),
              make_moe_down(work.n_experts, work.top_k, work.inter, hidden, tiles), args)
