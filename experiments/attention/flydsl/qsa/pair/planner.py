"""Per-call pair plan: blocks both queries of a pair selected go first in each row's list."""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from .. import direct, union

W = 2  # queries (and waves) per CTA


@triton.jit
def pair_qsa_split_common(Blocks, Dense, Meta, Groups, Active, Lists, Common,
                          MAX_BLOCKS: tl.constexpr, W: tl.constexpr):
    cta = tl.program_id(0)
    local = tl.program_id(1)
    first = tl.load(Meta + cta * 5)
    rows = tl.load(Meta + cta * 5 + 1)
    position0 = tl.load(Meta + cta * 5 + 4)
    group = tl.load(Groups + cta * 2)
    offset = tl.load(Groups + cta * 2 + 1)
    if local < rows:
        if tl.load(Active + group) == 0:
            row = first + local
            cols = tl.arange(0, 512)
            blocks = tl.load(Blocks + row * 512 + cols)
            valid = blocks >= 0
            bits = tl.load(Dense + group * MAX_BLOCKS + blocks, valid, 0)
            mask = (1 << rows) - 1
            # Complete for every query of the CTA; one-query CTAs gain no reuse.
            common = valid & (((bits >> offset) & mask) == mask) & (blocks * 4 + 3 <= position0) & (rows > 1)
            rank = tl.cumsum(common.to(tl.int32), 0) - 1
            count = tl.sum(common.to(tl.int32), 0)
            # Phase 1 takes whole 8-block (32-token) chunks; the remainder stays private.
            shared_count = (count // 8) * 8
            shared = common & (rank < shared_count)
            rest = valid & ~shared
            destination = tl.where(shared, rank, shared_count + tl.cumsum(rest.to(tl.int32), 0) - 1)
            destination = tl.where(valid, destination, cols)
            tl.store(Lists + row * 512 + destination, tl.where(valid, blocks, -1))
            if local == 0:
                tl.store(Common + cta, shared_count)


def prepare(inputs, uplan, gated=True):
    """Split every union group into query pairs; gated runs only groups union leaves to direct."""
    packed_bytes = inputs.k.numel() * inputs.k.element_size() + inputs.v.numel() * inputs.v.element_size()
    if not (len(inputs.query_lens) == 1 and inputs.k.shape[0] % 4 == 0
            and packed_bytes <= direct.MAX_PACKED_KV_BYTES
            and inputs.k.numel() * 2 + inputs.k.shape[1] * 1536 + 512 < 2**32):
        raise ValueError("The pair kernel needs the packed-KV layout (direct.prepare's packed conditions)")
    rows, groups = [], []
    for g, (first, count, k0, kv_len, position0) in enumerate(uplan.metadata.cpu().numpy()):
        for local in range(0, count, W):
            rows.append((first + local, min(W, count - local), k0, kv_len, position0 + local))
            groups.append((g, local))
    device = inputs.q.device
    return SimpleNamespace(
        metadata=torch.tensor(rows, dtype=torch.int32, device=device).reshape(-1, 5),
        groups=torch.tensor(groups, dtype=torch.int32, device=device).reshape(-1, 2),
        num_tiles=len(rows),
        common=torch.zeros(len(rows), dtype=torch.int32, device=device),
        lists=torch.empty((inputs.q.shape[0], 512), dtype=torch.int32, device=device),
        packed_key=torch.empty_like(inputs.k),
        packed_value=torch.empty_like(inputs.v),
        gated=gated,
        uplan=uplan,
        active_source=uplan.active if gated else torch.zeros_like(uplan.active),
        query_tiles=uplan.query_tiles,
    )


def rebuild_membership(inputs, uplan):
    """The part of union.rebuild_plan the planner reads; union-only kernels are skipped."""
    if uplan.num_tiles == 0:
        return
    uplan.dense_membership.zero_()
    union.union_qsa_scatter_membership[(uplan.num_tiles, uplan.query_tile)](
        inputs.block_indices, inputs.query_positions, uplan.metadata, uplan.dense_membership,
        uplan.query_tile, uplan.max_blocks, 512, 512, num_warps=4)


def rebuild(inputs, plan):
    # One warp: 35 us vs 57 us with four (real L3 M12000 TP2).
    pair_qsa_split_common[(plan.num_tiles, W)](
        inputs.block_indices, plan.uplan.dense_membership, plan.metadata, plan.groups, plan.active_source,
        plan.lists, plan.common, plan.uplan.max_blocks, W, num_warps=1)
