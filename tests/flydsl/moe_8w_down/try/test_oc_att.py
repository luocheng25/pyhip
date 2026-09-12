# SPDX-License-Identifier: MIT
"""CPU regressions for successful-issue timing and exclusive physical owners."""

import json

import pytest

import analyze_oc_att as analysis


def test_successful_issue_pc_and_union(tmp_path):
    code = [
        ["s_nop 0", 0, 1, ""],
        ["v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], 0", 0, 2, ""],
        ["buffer_load_dwordx4 v[4:7], v0, s[0:3], 0 offen", 0, 3, ""],
        ["s_barrier", 0, 4, ""],
    ]
    (tmp_path / "code.json").write_text(json.dumps({"code": code}))
    # MFMA attempts at20/100, stalls20: execute[40,72),[120,152).
    # Both waves' MFMA executions overlap and must count only once.
    for slot, blocker in ((0, 3), (1, 4)):
        records = [[0, 1, 0, 4, 1], [20, 6, 20, 24, 2], [72, 3, 16, 20, blocker], [100, 6, 20, 24, 2]]
        payload = {"num_insts": 4, "num_stitched": 4, "wave": {
            "begin": 0, "end": 160, "cu": 0, "simd": 0, "slot": slot, "instructions": records}}
        (tmp_path / f"se0_sm0_sl{slot}_wv0.json").write_text(json.dumps(payload))
    table, groups, counts = analysis.parse(tmp_path, 2, 32, False)
    assert counts["active_waves"] == 2
    assert groups[(0,0,0)][0]["mfmas"][0]["issue"] == 40
    result = analysis.ledger(table, groups, 2, 1, 32, 0, 2)
    assert analysis.MFMA_EXEC_CYCLES == 32
    assert result["mfma_execution_cycles"] == result["equivalent_slot_cycles"] == 32
    assert result["selected_cycles"] == 112
    assert result["busy_cycles"] == 64  # not128 and not attempt-based windows
    assert result["idle_cycles"] == 48
    assert result["selected_mfma_slots"] == 3.5
    assert result["busy_mfma_slots"] == 2
    assert result["idle_mfma_slots"] == 1.5
    assert result["categories"]["VMEM issue"]["cycles"] == 20
    assert result["categories"]["VMEM issue"]["equivalent_mfma_slots"] == 0.625
    assert result["categories"]["barrier"]["cycles"] == 0  # VMEM wins exclusive priority
    assert result["task_lifecycle"]["prologue"]["cycles"] == 40
    assert result["task_lifecycle"]["epilogue"]["cycles"] == 8


def test_reject_gfx942_16cycle_model(tmp_path):
    with pytest.raises(ValueError, match="32-cycle"):
        analysis.parse(tmp_path, 2, 16, False)
    with pytest.raises(ValueError, match="32-cycle"):
        analysis.ledger([], {}, 2, 1, 16, 0, 2)


def test_static_swizzle_not_claimed_as_measured():
    model = analysis.method.static_distribution(3584, 3072, 256, 8, 4, 2)
    assert model["uniform_early_exit_workgroups"] == 512
    assert model["tasks_per_cu_histogram"] == {"12": 256}