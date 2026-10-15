"""Launch-wide synchronization for persistent MonoKernel launches (all CTAs co-resident).

``grid_barrier`` counts arrivals on 32-bit global counters that only grow during a launch: barrier
``k`` (0-based) waits until they reach ``(k + 1) * ctas``. Every thread first waits for its stores to
be acknowledged, then thread 0 arrives and spins with ``s_sleep``. Without ``coherent_data`` it also
releases/acquires at agent scope (L2 write-back and invalidate on gfx942). ``grid_arrive`` and
``grid_wait`` split the barrier so independent loads (weights) can be issued in between.
``grid_exit`` lets the last CTA reset the counters, so a captured graph replays from zero.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, scf
from flydsl.expr import rocdl

from ._ir import barrier, raw


def address(tensor):
    return fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))


def _global(address_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), raw(fx.Int64(address_i64)))


def _first_thread(tid, body):
    if_op = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, raw(fx.Int32(tid)), raw(fx.Int32(0))))
    with ir.InsertionPoint(if_op.then_block):
        body()
        scf.YieldOp([])


def _spin_until(pointer, target):
    while_op = scf.WhileOp([], [])
    before = while_op.regions[0].blocks.append()
    with ir.InsertionPoint(before):
        value = llvm.load(fx.Int32.ir_type, pointer, alignment=4, ordering=llvm.AtomicOrdering.monotonic,
                          syncscope="agent")
        scf.ConditionOp(arith.cmpi(arith.CmpIPredicate.ult, value, raw(fx.Int32(target))), [])
    after = while_op.regions[1].blocks.append()
    with ir.InsertionPoint(after):
        rocdl.s_sleep(1)
        scf.YieldOp([])


def _if(condition, body):
    if_op = scf.IfOp(raw(condition))
    with ir.InsertionPoint(if_op.then_block):
        body()
        scf.YieldOp([])


STRIPE_WORDS = 32  # one 128-byte line per arrival counter
VMCNT0 = 0x0F70  # gfx9 s_waitcnt: vmcnt(0), expcnt/lgkmcnt unconstrained


def grid_barrier(counter, ordinal, ctas, tid, coherent_data=False, stripes=1):
    """Barrier number ``ordinal`` of this launch over ``ctas`` CTAs; ``counter`` is an i64 address of
    ``stripes`` arrival counters ``STRIPE_WORDS`` words apart (CTA ``c`` arrives on ``c % stripes``;
    thread ``s`` polls stripe ``s``).

    With ``coherent_data`` the caller moves every cross-CTA value with device-coherent (sc1)
    loads/stores, so the barrier skips the agent-scope L2 write-back and invalidate; every thread
    still waits for its stores to be acknowledged before the CTA arrives."""
    grid_arrive(counter, tid, coherent_data, stripes)
    grid_wait(counter, ordinal, ctas, tid, coherent_data, stripes)


def grid_arrive(counter, tid, coherent_data=True, stripes=1):
    """First half of ``grid_barrier``: the CTA's stores are acknowledged, then thread 0 arrives. Loads
    that do not depend on other CTAs (weights) may be issued before ``grid_wait``."""
    rocdl.s_waitcnt(VMCNT0)
    llvm.fence(llvm.AtomicOrdering.release, syncscope="workgroup")
    barrier()
    cta = fx.Int32(fx.block_idx.x)

    def arrive():
        if not coherent_data:
            llvm.fence(llvm.AtomicOrdering.release, syncscope="agent")
        mine = counter + fx.Int64((cta % stripes) * (STRIPE_WORDS * 4))
        llvm.atomicrmw(llvm.AtomicBinOp.add, _global(mine), raw(fx.Int32(1)), llvm.AtomicOrdering.monotonic,
                       syncscope="agent", alignment=4)

    _if(fx.Int32(tid) == 0, arrive)


def grid_wait(counter, ordinal, ctas, tid, coherent_data=True, stripes=1):
    """Second half of ``grid_barrier`` number ``ordinal``: wait for every CTA's arrival."""

    def wait():
        members = (fx.Int32(tid) < ctas % stripes).select(fx.Int32(ctas // stripes + 1), fx.Int32(ctas // stripes))
        _spin_until(_global(counter + fx.Int64(fx.Int32(tid) * (STRIPE_WORDS * 4))), members * (ordinal + 1))
        if not coherent_data:
            llvm.fence(llvm.AtomicOrdering.acquire, syncscope="agent")

    _if(fx.Int32(tid) < stripes, wait)
    barrier()


def grid_exit(counter, exits, ctas, tid, stripes=1):
    """After the last barrier: the last CTA to leave zeroes the ``stripes`` counters and ``exits``."""

    def leave():
        ticket = llvm.atomicrmw(llvm.AtomicBinOp.add, _global(exits), raw(fx.Int32(1)),
                                llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
        last = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, ticket, raw(fx.Int32(ctas - 1))))
        with ir.InsertionPoint(last.then_block):
            for s in range(stripes):
                llvm.store(raw(fx.Int32(0)), _global(counter + fx.Int64(s * STRIPE_WORDS * 4)), alignment=4,
                           ordering=llvm.AtomicOrdering.monotonic, syncscope="agent")
            llvm.store(raw(fx.Int32(0)), _global(exits), alignment=4, ordering=llvm.AtomicOrdering.monotonic,
                       syncscope="agent")
            scf.YieldOp([])

    _first_thread(tid, leave)
