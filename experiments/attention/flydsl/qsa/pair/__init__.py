"""Reference prototype, not used by qsa(): query-pair sharing for the direct branch.

Two adjacent queries read the blocks they both selected once (each of two waves
loads one D half). Measured results, limits and integration notes: ../try.md
sections 1, 7 and 8; ../opt.md, 2026-09-28. `bench.py` checks and times it.
"""

import torch

from .. import dense
from ..qsa import _qsa_check_errors, _qsa_recover_blocks
from .kernel import run
from .planner import prepare, rebuild, rebuild_membership

__all__ = ["prepare", "qsa_all_pair", "rebuild", "rebuild_membership", "run"]


def qsa_all_pair(workspace, inputs, plan, out):
    """qsa()'s device work with every sparse row on the pair kernel (the TP2 option).

    workspace is a qsa._Workspace, inputs its bind(); plan is
    prepare(inputs, workspace.union, gated=False), or None when every row is dense.
    """
    rows = inputs.q.shape[0]
    _qsa_recover_blocks[(rows,)](
        inputs.indices, inputs.query_positions, inputs.kv_lens,
        inputs.query_sequence_ids, inputs.block_indices, workspace.errors, num_warps=4)
    _qsa_check_errors[(1,)](workspace.errors, workspace.valid, rows, 1024, num_warps=4)
    torch._assert_async(workspace.valid, "Invalid compressed QSA token/block/tail ABI")
    if plan is not None:
        rebuild_membership(inputs, workspace.union)
    dense.run(inputs=inputs, prepared=workspace.dense, out=out)
    if plan is not None:
        rebuild(inputs, plan)
        run(inputs, plan, out)
    return out
