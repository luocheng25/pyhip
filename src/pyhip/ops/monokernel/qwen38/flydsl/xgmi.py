"""In-launch all-reduce over XGMI for the fused MonoKernel layer launches.

Every rank owns one uncached region (``peer.PeerMemory``) that every rank has mapped:

    FLAGS   u32 flags[MAX_CTAS][MAX_RANKS]: flags[c][s] = the last handshake CTA c of rank s signalled here
    EPOCHS  u32 epoch[MAX_CTAS]: the handshakes CTA c of this rank has done (read by this rank only)
    RECV    BF16 recv[2][tp][capacity][H]: recv[site][s] = rank s's partial of all-reduce ``site``

All-reduce ``site`` (0: the mixer output, 1: the MoE output) is the producer phase storing every BF16 tile
of this rank's partial into ``recv[site][rank]`` of every rank (``targets``), a per-CTA ``handshake`` at its
end and the grid barrier behind it: CTA c leaves the handshake once CTA c of every peer has stored all of
its tiles, so after the barrier every rank's partial is complete in the local region. The consumer reads
the ``tp`` partials of its local region in rank order (``sources``) and ``reduce`` returns
``trunc_bf16(sum in FP32 from rank 0)``, AITER's one-stage all-reduce bit for bit. The sites alternate:
a rank stores into a site again only after its next handshake at the other site, which no peer enters
before the grid barrier behind its consumer of this site. Every rank runs the same launches with the same
grid, as SGLang's tensor-parallel ranks do; handshake numbers compare modulo 2^32.

Stores and loads of the region are system-coherent (sc0 sc1) and the region is uncached, so the partials
cross neither a stale L2 line nor a write-back; a flag is written only after the CTA's partial stores are
acknowledged.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, scf
from flydsl.expr import range_constexpr, rocdl

from ..config import QWEN38_FLASH_NEXT
from ..peer import MAX_RANKS, TABLE_WORDS
from ._ir import HIGH_HALF, barrier, bf16_pair, load, raw, scalar, words
from ._phase import buffer_at
from .sync import VMCNT0, _global, _if

H = QWEN38_FLASH_NEXT.hidden
MAX_CTAS = 128
FLAGS = 0
EPOCHS = MAX_CTAS * MAX_RANKS * 4
RECV = 8192
SYS = 17  # gfx942 cache policy sc0 | sc1: system-coherent
SITES = 2


def region_bytes(tp: int, capacity: int) -> int:
    return RECV + SITES * tp * capacity * H * 2


def _i64(word):
    """Two i32 words (low first) as one i64."""
    return words((word[0], word[1])).bitcast(fx.Int64)[0]


def _wait_reached(pointer, target):
    """Spin until the u32 at ``pointer`` has reached ``target`` (modulo 2^32)."""
    while_op = scf.WhileOp([], [])
    before = while_op.regions[0].blocks.append()
    with ir.InsertionPoint(before):
        value = llvm.load(fx.Int32.ir_type, pointer, alignment=4, ordering=llvm.AtomicOrdering.monotonic)
        behind = arith.cmpi(arith.CmpIPredicate.slt, arith.subi(value, raw(target)), raw(fx.Int32(0)))
        scf.ConditionOp(behind, [])
    after = while_op.regions[1].blocks.append()
    with ir.InsertionPoint(after):
        rocdl.s_sleep(1)
        scf.YieldOp([])


class Comm:
    """Kernel-side view of the regions; ``table`` is the i64 address of ``PeerMemory.table``."""

    def __init__(self, ctx, table, tp, capacity):
        assert 1 <= tp <= MAX_RANKS
        self.ctx, self.tp, self.capacity = ctx, tp, capacity
        self.table = buffer_at(table, TABLE_WORDS * 8)
        self.rank = scalar(load(self.table, MAX_RANKS * 8, words=1))
        self.mine = self._base(self.rank)

    def _base(self, rank):
        word = load(self.table, rank * 8, words=2)
        return _i64((scalar(word[0]), scalar(word[1])))

    def _slot(self, site, rank):
        return fx.Int64(RECV + (site * self.tp + rank) * (self.capacity * H * 2))

    def targets(self, site, rows):
        """``recv[site][rank]`` of every rank: where the producer stores each tile of this rank's partial."""
        offset = self._slot(site, self.rank)
        return [buffer_at(self._base(r) + offset, rows * H * 2) for r in range_constexpr(self.tp)]

    def sources(self, site, rows):
        """``recv[site][0 .. tp)`` of the local region, in rank order."""
        return [buffer_at(self.mine + self._slot(site, r), rows * H * 2) for r in range_constexpr(self.tp)]

    def handshake(self):
        """End of a producer phase (every thread of every CTA): this CTA's stores are acknowledged, then it
        waits for CTA ``cta`` of every peer to get there."""
        rocdl.s_waitcnt(VMCNT0)
        barrier()
        if self.tp > 1:
            _if(self.ctx.wave == 0, self._exchange)
            barrier()

    def _exchange(self):
        ctx, lane = self.ctx, self.ctx.lane
        epoch_at = self.mine + fx.Int64(EPOCHS + ctx.cta * 4)
        epoch = fx.Int32(llvm.load(fx.Int32.ir_type, _global(epoch_at), alignment=4,
                                   ordering=llvm.AtomicOrdering.monotonic)) + 1

        def signal():
            peer = _i64(load(self.table, lane * 8, words=2))
            llvm.store(raw(epoch), _global(peer + fx.Int64(FLAGS + (ctx.cta * MAX_RANKS + self.rank) * 4)),
                       alignment=4, ordering=llvm.AtomicOrdering.monotonic)
            _wait_reached(_global(self.mine + fx.Int64(FLAGS + (ctx.cta * MAX_RANKS + lane) * 4)), epoch)

        _if((lane < self.tp) & (lane != self.rank), signal)
        _if(lane == 0, lambda: llvm.store(raw(epoch), _global(epoch_at), alignment=4,
                                          ordering=llvm.AtomicOrdering.monotonic))


def reduce(sources, offset):
    """The 8 BF16 values at byte ``offset`` summed over ``sources`` in order in FP32 and truncated to BF16
    (as FP32 values)."""
    total = None
    for src in sources:
        word = load(src, offset, aux=SYS)
        values = []
        for i in range_constexpr(4):
            values += list(bf16_pair(word[i]))
        total = values if total is None else [a + b for a, b in zip(total, values)]
    if len(sources) == 1:
        return total
    return [(v.bitcast(fx.Uint32) & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32) for v in total]
