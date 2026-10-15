"""Phase scaffolding shared by the per-phase launches and the fused per-layer-half launch.

A *phase* is a device function over a list of items. Every launch runs a resident grid of one CTA of
``THREADS`` threads per CU; a per-phase launch runs one phase, a fused launch runs the phases of a
layer half with ``grid_barrier`` between them. Items are scheduled at three granularities:

- wave items (GEMV blocks, rows): wave ``w`` of CTA ``c`` takes ``w * ctas + c, ... + waves``, so
  consecutive items land on different CUs;
- half items (256 threads, both halves of a CTA in lockstep so they may share CTA barriers);
- CTA items.

Phases are module-level functions rewritten by FlyDSL's AST transform (``@device``) so they may use
dynamic ``for``/``if``. Values that cross CTAs inside a fused launch move with device-coherent (sc1)
loads/stores (``ctx.aux``), so the grid barrier needs no L2 write-back or invalidate.

``HalfGemm`` is the K loop of one 16-row weight block split over the 4 waves of a half exactly like
the 4-wave stage GEMVs (wave ``w`` takes loads ``q = w (mod 4)``, the partials are summed 0 + 1 + 2 + 3
through LDS), so the phases reproduce the validated stage kernels bit for bit. Loads are emitted in
batches ahead of the math (``pipelined``): the compiler does not hoist loads across device-coherent
ones by itself.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr, rocdl

from ._ir import (DROP, barrier, bf16x4_half, i64_half, lds_base, lds_load, lds_store, load, mfma_bf16, mfma_fp8,
                  scalar, zeros4)

device = ASTRewriter.transform
THREADS = 512
WAVES = THREADS // 64
HALF = 256
HALF_WAVES = HALF // 64
LDS_WORDS = 16 * 1024  # 64 KiB arena shared by all phases; also caps the occupancy at one CTA per CU
SC1 = 16  # gfx942 cache policy: device-coherent load/store
STAGED_WAVES = 4  # summation order of the stage GEMVs


def buffer_at(address, num_bytes):
    """Raw buffer descriptor at an i64 device address."""
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), fx.Int64(address).ir_value())
    return rocdl.make_buffer_rsrc(ir.Type.parse("!llvm.ptr<8>"), pointer, fx.Int16(0).ir_value(),
                                  fx.Int64(num_bytes).ir_value(), fx.Int32(0x27000).ir_value())


class Ctx:
    """Per-thread launch context while tracing a phase."""

    def __init__(self, rows, coherent, lds_words=LDS_WORDS):
        self.tid = fx.Int32(fx.thread_idx.x)
        self.lane = self.tid & 63
        self.wave = scalar(self.tid >> 6)
        self.cta = fx.Int32(fx.block_idx.x)
        self.ctas = fx.Int32(fx.grid_dim.x)
        self.wave_id = self.wave * self.ctas + self.cta
        self.waves = self.ctas * WAVES
        self.half = scalar(self.tid >> 8)
        self.half_tid = self.tid & (HALF - 1)
        self.half_wave = self.wave & (HALF_WAVES - 1)
        self.rows = rows
        self.coherent = coherent
        self.aux = SC1 if coherent else 0
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_words, 16]).peek()
        self.shared = lds_base(storage, lds_words)


REGION_BYTES = LDS_WORDS * 4 // 4  # one half's GEMV reduction buffer (two halves x double buffering)


def batches(count, tiles, words_per_load=1, budget=64):
    """Load batches of a wave's K loop: a batch holds ``words_per_load * (1 + tiles)`` 16-byte words per
    load step within about ``budget`` registers; two batches are in flight."""
    size = max(1, budget // (4 * words_per_load * (1 + tiles)))
    return [list(range(start, min(start + size, count))) for start in range(0, count, size)]


def pipelined(groups, issue, consume):
    """Program-order software pipeline: the loads of batch ``i + 1`` are emitted before the math of
    batch ``i`` (the compiler does not hoist loads across device-coherent ones)."""
    pending = issue(groups[0])
    for i in range_constexpr(len(groups)):
        following = issue(groups[i + 1]) if i + 1 < len(groups) else None
        consume(groups[i], pending)
        pending = following


class HalfGemm:
    """One 16-row block per half item, K split over the half's 4 waves exactly like the stage GEMVs:
    wave ``w`` accumulates loads ``q = w, w + 4, ...`` (masked past the end, as in the stages); the
    4 partials go through LDS (``stash``, CTA barrier) and wave ``t % 4`` sums them in order 0..3
    (``gather``) for token tile ``t``."""

    def __init__(self, k, tiles, weight="fp8", ksplit=1):
        esize = 1 if weight == "fp8" else 2
        self.row_bytes = k * esize
        self.loads = self.row_bytes // 64
        assert self.loads % ksplit == 0 and tiles in (1, 2, 4)
        self.split_loads = self.loads // ksplit
        self.per_wave = -(-self.split_loads // STAGED_WAVES)
        self.tiles = tiles
        self.mfma = mfma_fp8 if weight == "fp8" else mfma_bf16
        self.operand = i64_half if weight == "fp8" else bf16x4_half
        self.groups = batches(self.per_wave, tiles)

    def stash(self, region, acc, lane, wave):
        for t in range_constexpr(self.tiles):
            lds_store(region + ((wave * self.tiles + t) * 64 + lane) * 16, acc[t].bitcast(fx.Int32))

    def gather(self, region, t, lane):
        total = lds_load(region + (t * 64 + lane) * 16).bitcast(fx.Float32)
        for other in range_constexpr(1, STAGED_WAVES):
            total = total + lds_load(region + ((other * self.tiles + t) * 64 + lane) * 16).bitcast(fx.Float32)
        return total

    # ---- whole-item pieces for RoundGemv

    def _ok(self, wave, j, valid):
        local = wave + STAGED_WAVES * j
        inside = local < self.split_loads
        return local, (inside if valid is True else valid & inside)

    def weights(self, w, block, split, lane, wave, valid):
        """All of the wave's weight words of one item (0 past the end or for an invalid item)."""
        out = []
        for j in range_constexpr(self.per_wave):
            local, ok = self._ok(wave, j, valid)
            q = split * self.split_loads + local
            out.append(load(w, ok.select(block * (16 * self.row_bytes) + q * 1024 + lane * 16, fx.Int32(DROP))))
        return out

    def acts(self, act, split, wave, valid, steps=None):
        out = []
        for j in (range_constexpr(self.per_wave) if steps is None else steps):
            local, ok = self._ok(wave, j, valid)
            out.append([act(split * self.split_loads + local, t, ok) for t in range_constexpr(self.tiles)])
        return out

    def product(self, weights, act, split, wave, valid, cached=None):
        """FP32 partials of the wave's loads in order; activations from ``cached`` or loaded in batches."""
        acc = [zeros4() for _ in range_constexpr(self.tiles)]

        def consume(group, acts):
            for i in range_constexpr(len(group)):
                for t in range_constexpr(self.tiles):
                    for h in range_constexpr(2):
                        acc[t] = self.mfma(self.operand(weights[group[i]], h), self.operand(acts[i][t], h), acc[t])

        if cached is not None:
            consume(list(range(self.per_wave)), cached)
        else:
            pipelined(self.groups, lambda group: self.acts(act, split, wave, valid, group), consume)
        return acc


class RoundGemv:
    """Static half-item schedule of a GEMV phase on ``ctas`` resident CTAs: round ``r`` of CTA ``c``
    holds items ``2 (c + r ctas) + half``. The weights of round ``r + 1`` are issued before the math of
    round ``r`` (round 0's by ``prefetch``, between the grid barrier's arrive and wait); single-split
    GEMVs with one token tile load their activations once per phase. Partials go through
    double-buffered LDS regions, so a round needs one CTA barrier.

    ``locate(item) -> (w, block, split)``; ``epilogue(item, valid, region)`` runs after the barrier
    (wave ``t % 4`` gathers tile ``t`` with ``gemm.gather``)."""

    def __init__(self, gemm, items, ctas):
        self.gemm, self.items, self.ctas = gemm, items, ctas
        self.rounds = -(-((items + 1) // 2) // ctas)
        self.cache = gemm.split_loads == gemm.loads and gemm.tiles == 1

    def _item(self, ctx, r):
        item = (ctx.cta + r * self.ctas) * 2 + ctx.half
        return item, item < self.items

    def prefetch(self, ctx, locate):
        item, valid = self._item(ctx, 0)
        w, block, split = locate(item)
        return self.gemm.weights(w, block, split, ctx.lane, ctx.half_wave, valid)

    def run(self, ctx, locate, read, epilogue, pre=None):
        gemm, lane, wave = self.gemm, ctx.lane, ctx.half_wave
        cached = gemm.acts(read, 0, wave, True) if self.cache else None
        current = self.prefetch(ctx, locate) if pre is None else pre
        for r in range_constexpr(self.rounds):
            item, valid = self._item(ctx, r)
            _, _, split = locate(item)
            following = self.prefetch_round(ctx, locate, r + 1) if r + 1 < self.rounds else None
            region = ctx.shared + ((r % 2) * 2 + ctx.half) * REGION_BYTES
            gemm.stash(region, gemm.product(current, read, split, wave, valid, cached), lane, wave)
            barrier()
            epilogue(item, valid, region)
            current = following

    def prefetch_round(self, ctx, locate, r):
        item, valid = self._item(ctx, r)
        w, block, split = locate(item)
        return self.gemm.weights(w, block, split, ctx.lane, ctx.half_wave, valid)


def act_rows(a, row_bytes, rows, lane, aux):
    """Activation reader over a row-major [rows, K] buffer (``ok`` masks the load)."""

    def read(q, t, ok=True):
        token = 16 * t + (lane & 15)
        live = (token < rows) if ok is True else ok & (token < rows)
        return load(a, live.select(token * row_bytes + q * 64 + (lane >> 4) * 16, fx.Int32(DROP)), aux=aux)

    return read
