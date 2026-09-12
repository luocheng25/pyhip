# SPDX-License-Identifier: MIT

"""Correctness/ISA checks for BN64/128 four-stage and BN256 eight/sixteen-stage kernels."""

import argparse
from pathlib import Path
import re

import pytest
import torch
from aiter.ops.shuffle import shuffle_weight

from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down


def make_case(tokens, n, k, experts, topk, seed, routing_mode="manual", weight_layout=(16, 16)):
    """Real preshuffle + encoded/padded routing; independent nonunit K scales."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(tokens, topk, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    w = torch.randn(experts, n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    a_scales = torch.rand(k // 128, tokens * topk, device="cuda", generator=generator) + 0.25
    b_scales = torch.rand(experts, (n + 127) // 128, k // 128, device="cuda", generator=generator) + 0.25
    scores = torch.rand(tokens, experts, device="cuda", generator=generator)
    assignments = scores.topk(topk, dim=1).indices
    routing = torch.rand(tokens, topk, device="cuda", generator=generator)
    routing /= routing.sum(dim=1, keepdim=True)
    if routing_mode == "aiter":
        from aiter.fused_moe import moe_sorting
        ids, routes, eids, valid, _ = moe_sorting(
            assignments.to(torch.int32), routing, experts, n, torch.bfloat16, 256,
            None, None, 0,
        )
    else:
        assert routing_mode == "manual"
        sorted_ids, sorted_routes, sorted_experts = [], [], []
        sentinel = (topk << 24) | tokens
        for expert in range(experts):
            rows = torch.where(assignments.reshape(-1) == expert)[0]
            count = rows.numel()
            padded = (count + 255) // 256 * 256
            ids = torch.full((padded,), sentinel, dtype=torch.int32, device="cuda")
            ids[:count] = ((rows % topk) << 24 | (rows // topk)).to(torch.int32)
            routes = torch.zeros(padded, device="cuda")
            routes[:count] = routing.reshape(-1)[rows]
            sorted_ids.append(ids)
            sorted_routes.append(routes)
            sorted_experts.extend([expert] * (padded // 256))
        ids = torch.cat(sorted_ids)
        routes = torch.cat(sorted_routes)
        eids = torch.tensor(sorted_experts, dtype=torch.int32, device="cuda")
        valid = torch.tensor([ids.numel()], dtype=torch.int32, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    ref = torch.empty(tokens * topk, n, dtype=torch.float32, device="cuda")
    for expert in range(experts):
        rows = torch.where(assignments.reshape(-1) == expert)[0]
        accum = torch.zeros(rows.numel(), n, dtype=torch.float32, device="cuda")
        for kb in range(k // 128):
            partial = a.float().reshape(tokens * topk, k)[rows, kb * 128:(kb + 1) * 128] @ w[
                expert, :, kb * 128:(kb + 1) * 128
            ].float().T
            scales = a_scales[kb, rows, None] * b_scales[expert, :, kb].repeat_interleave(128)[None, :n]
            accum += partial * scales
        ref[rows] = accum * routing.reshape(-1)[rows, None]
    ref = ref.to(torch.bfloat16).reshape(tokens, topk, n)

    # Guards prove padded rows never write outside the caller's output buffer.
    storage = torch.full((ref.numel() + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = storage[n:-n].view_as(ref)
    args = (output, a, shuffle_weight(w, layout=weight_layout), a_scales, b_scales,
            ids, routes, eids, valid, counter)
    return args, ref, storage


def run_case(*, tokens=257, n=256, k=256, experts=1, topk=1, block_n=256, splits=1,
             seed=1234, graph=False, bench=False, routing_mode="manual",
             bn256_stages=8, steady_vmcnt=6, prefetch_distance=None, tuned=False):
    args, ref, guard = make_case(tokens, n, k, experts, topk, seed, routing_mode)
    kernel_options = dict(
        n=n, k=k, topk=topk, num_experts=experts, block_n=block_n, num_oc_splits=splits,
        bn256_stages=bn256_stages, steady_vmcnt=steady_vmcnt,
        prefetch_distance=prefetch_distance,
    )
    if tuned:
        assert k == 256, "ATT-tuned path requires K256"
        kernel_options.update({**ATT_TUNED_BN128_CONFIG, "num_oc_splits": splits})
    kernel = flydsl_moe_gemm_8wave_down(**kernel_options)
    returned = kernel(*args)
    assert returned is args[0]
    torch.cuda.synchronize()

    def check():
        actual = args[0].float()
        expected = ref.float()
        error = (actual - expected).abs()
        mismatches = (error > 0.01 + 0.01 * expected.abs()).sum().item()
        rel_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-20)).item()
        assert torch.isfinite(actual).all(), "non-finite/unwritten output"
        assert not mismatches, f"{mismatches} mismatches, max_abs={error.max().item()}, rel_l2={rel_l2}"
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all(), "output guard overwritten"
        return rel_l2, error.max().item()

    rel_l2, max_abs = check()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            kernel(*args)
        for _ in range(3):
            args[0].fill_(torch.nan)
            args[-1].fill_(0x123456)
            capture.replay()
            torch.cuda.synchronize()
            check()
    elapsed = None
    if bench:
        from aiter.test_common import run_perftest
        _, elapsed = run_perftest(kernel, *args, num_warmup=5, num_iters=30, num_rotate_args=1)
        check()
    config = getattr(kernel, "config", kernel_options)
    print(
        f"PASS tokens={tokens} N={n} K={k} E={experts} topk={topk} BN={config['block_n']} "
        f"splits={splits} graph={graph} routing={routing_mode} tuned={tuned} "
        f"output_aux={config.get('output_cache_policy', 0)} agpr=False "
        f"bn256_stages={bn256_stages} steady_vmcnt={steady_vmcnt if bn256_stages == 16 else 'auto'} "
        f"prefetch_distance={config.get('prefetch_distance', prefetch_distance)} "
        f"rel_l2={rel_l2:.6g} max_abs={max_abs:.6g} "
        f"time_us={elapsed if elapsed is not None else 'unchecked'}",
        flush=True,
    )
    return args, rel_l2


def audit_isa(path, *, require_no_scratch=False, require_nt=False, require_seven_valu=False,
              require_four_stage=False, require_cached=False, require_no_agpr=False,
              require_no_compute_address=False, require_sixteen_stage=False,
              require_steady_vmcnt=None, require_bn64_four_stage=False):
    """Check actual marked Memory stages, not just the Python operation list."""
    text = Path(path).read_text()
    memory = re.findall(r"MOE8_MEMORY_BEGIN_(\d+)(.*?)MOE8_MEMORY_END_\1", text, re.S)
    compute = re.findall(r"MOE8_COMPUTE_BEGIN_(\d+)(.*?)MOE8_COMPUTE_END_\1", text, re.S)
    assert memory and compute, "no schedule markers in assembly"
    assert [stage for stage, _ in memory] == [stage for stage, _ in compute], "Memory/Compute stage mismatch"
    stages = max(int(stage) for stage, _ in compute) + 1
    if require_four_stage or require_bn64_four_stage:
        assert not (require_four_stage and require_bn64_four_stage), "select BN64 or BN128 audit, not both"
        assert stages == 2 and len(compute) % 2 == 0, "expected two Memory/Compute pairs per N tile"
        assert [int(stage) for stage, _ in compute] == [0, 1] * (len(compute) // 2)
        stores_per_stage = 2 if require_bn64_four_stage else 4
        mfmas_per_stage = 8 if require_bn64_four_stage else 16
        for index, (_, body) in enumerate(memory):
            stores = re.findall(r"^\s*buffer_store_dwordx4\b", body, re.M)
            assert len(stores) == (0 if index < 2 else stores_per_stage), f"Memory{index} delayed stores: {len(stores)}"
            if require_bn64_four_stage:
                loads = re.findall(r"^\s*buffer_load_dwordx4\b[^\n]*\blds\b", body, re.M)
                assert len(loads) <= 1, f"Memory{index} direct-LDS packets: {len(loads)}"
        for index, (_, body) in enumerate(compute):
            mfmas = re.findall(r"^\s*v_mfma_(?:scale_)?f32_16x16x128_f8f6f4\b", body, re.M)
            assert len(mfmas) == mfmas_per_stage, f"Compute{index} MFMA count: {len(mfmas)}"
        print(f"FOUR-STAGE PASS: 2 Compute x {mfmas_per_stage} MFMA, delayed output split "
              f"{stores_per_stage}+{stores_per_stage} stores across 2 Memory stages")
    if require_sixteen_stage:
        assert stages == 8 and len(compute) % 8 == 0, "expected eight Memory/Compute pairs per N256"
        assert [int(stage) for stage, _ in compute] == list(range(8)) * (len(compute) // 8)
        for index, (_, body) in enumerate(memory):
            stores = re.findall(r"^\s*buffer_store_dwordx4\b", body, re.M)
            loads = re.findall(r"^\s*buffer_load_dwordx4\b[^\n]*\blds\b", body, re.M)
            assert len(stores) == (0 if index < 8 else 2), f"Memory{index} delayed stores: {len(stores)}"
            assert len(loads) <= 1, f"Memory{index} direct-LDS packets: {len(loads)}"
            if 8 <= index < len(memory) - 8:
                assert len(loads) == 1, f"steady Memory{index} must issue one B packet"
        for index, (_, body) in enumerate(compute):
            mfmas = re.findall(r"^\s*v_mfma_(?:scale_)?f32_16x16x128_f8f6f4\b", body, re.M)
            assert len(mfmas) == 8, f"Compute{index} MFMA count: {len(mfmas)}"
        print("SIXTEEN-STAGE PASS: 8 Compute x 8 MFMA; each steady Memory has 2 stores + 1 direct-LDS load")
    if require_steady_vmcnt is not None:
        assert require_sixteen_stage or require_bn64_four_stage
        assert require_steady_vmcnt in ((0, 3, 6) if require_bn64_four_stage else (6, 9))
        if require_bn64_four_stage:
            marked = re.search(r"MOE8_STEADY_BEGIN(.*?)MOE8_STEADY_END", text, re.S)
            assert marked, "BN64 vmcnt audit needs a dynamic steady loop"
            steady = re.findall(r"MOE8_MEMORY_BEGIN_(\d+)(.*?)MOE8_MEMORY_END_\1", marked[1], re.S)
        else:
            # Skip the first two N tiles (prologue/transition) and final N.
            steady = memory[2 * stages:-stages]
        assert steady, "vmcnt audit needs a shape with a dynamic steady loop"
        for index, (stage, body) in enumerate(steady):
            waits = re.findall(r"\bs_waitcnt\b[^\n]*\bvmcnt\((\d+)\)", body)
            assert waits == [str(require_steady_vmcnt)], f"steady Memory{index}/step{stage} vmcnts: {waits}"
            if require_bn64_four_stage:
                stores = re.findall(r"^\s*buffer_store_dwordx4\b", body, re.M)
                loads = re.findall(r"^\s*buffer_load_dwordx4\b[^\n]*\blds\b", body, re.M)
                assert len(stores) == 2 and len(loads) == 1, f"steady Memory{index} is not 2 stores + 1 B load"
        print(f"VMCNT PASS: all {len(steady)} dynamic steady Memory stages use vmcnt({require_steady_vmcnt})")
    bad = []
    for index, (stage, body) in enumerate(memory):
        instructions = re.findall(r"^\s*(v_\w+|ds_write\w*)\b.*$", body, re.M)
        if instructions:
            bad.append((index, stage, instructions))
    assert not bad, f"Memory contains vector instructions/LDS writes: {bad}"
    assert re.search(r"buffer_load_\w+.*\blds\b", text), "B DMA missing"
    assert "v_cvt_pk_bf16_f32" in text and "v_permlane16_swap_b32" in text
    assert "v_pk_mul_f32" not in text and "v_pk_add_f32" not in text
    for stage, body in compute:
        assert re.search(r"v_mfma_(?:scale_)?f32_16x16x128_f8f6f4", body), f"MFMA missing in Compute{stage}"
        assert re.search(r"v_(mul|fma|fmac)_f32", body), f"VALU missing in Compute{stage}"
        assert not re.search(r"\bds_(read|write)\w*\b", body), f"LDS operation in Compute{stage}"
    if require_no_compute_address:
        # Inspect the WHOLE marked stage, including its post-MFMA tail.
        # Coefficient readfirstlane and value moves are not address arithmetic.
        arithmetic = re.compile(
            r"^\s*((?:v_(?:(?:and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|mbcnt|bit|brev|alignbit|cmp)\w*"
            r"|(?:add|sub|mul|mad)\w*_(?:u|i)\d+\w*)"
            r"|s_(?:add|sub|mul|and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|brev)\w*))\b.*$",
            re.M,
        )
        bad = [(index, stage, arithmetic.findall(body))
               for index, (stage, body) in enumerate(compute) if arithmetic.search(body)]
        assert not bad, f"Compute contains integer address/coordinate arithmetic: {bad}"
        print(f"COMPUTE-ADDRESS PASS: all {len(compute)} full Compute stages have no integer address arithmetic")
    if require_no_scratch:
        private = re.findall(r"\.private_segment_fixed_size:\s*(\d+)", text)
        assert private and all(int(value) == 0 for value in private), f"private segment: {private}"
        assert not re.search(r"^\s*scratch_\w+", text, re.M), "scratch load/store in ISA"
        spills = re.findall(r"\.vgpr_spill_count:\s*(\d+)", text)
        assert spills and all(int(value) == 0 for value in spills), f"VGPR spills: {spills}"
        print("SCRATCH PASS: private segment=0, VGPR spill=0, no scratch instructions")
    if require_nt:
        stores = re.findall(r"^\s*buffer_store_\w+.*$", text, re.M)
        assert stores and all(re.search(r"\bnt\b", store) for store in stores), "non-NT output store"
        print(f"NT PASS: all {len(stores)} output stores are non-temporal")
    if require_cached:
        assert not require_nt, "cached and NT checks are mutually exclusive"
        stores = re.findall(r"^\s*buffer_store_\w+.*$", text, re.M)
        assert stores and all(not re.search(r"\bnt\b", store) for store in stores), "NT output store in cached variant"
        print(f"CACHED PASS: all {len(stores)} output stores have NT disabled")
    if require_no_agpr:
        instructions = "\n".join(re.findall(r"^\s*(?:v_|s_|ds_|buffer_|global_|scratch_)\w+.*$", text, re.M))
        assert not re.search(r"\bv_accvgpr_\w+|\ba(?:\d+|\[\d+)(?:\b|:)", instructions), "AGPR operand or transfer in ISA"
        count = re.findall(r"\.vgpr_count:\s*(\d+)", text)
        offset = re.findall(r"\.amdhsa_accum_offset\s+(\d+)", text)
        assert count and offset and int(count[0]) <= int(offset[0]), f"AGPR allocated: total={count}, offset={offset}"
        print(f"NO-AGPR PASS: no AGPR instructions/operands; total={count[0]}, accum_offset={offset[0]}")
    if require_seven_valu:
        intervals = 0
        # BN128 first Compute / BN256 first two Computes have no completed
        # record to pack.  Do not skip BN128's second (already full) Compute.
        # The later stages interleave independent dequant, BF16 conversion,
        # permutation and any register copies; do not exclude moves from ISA.
        assert stages in (2, 4, 8) and len(compute) >= stages, "expected BN64/128/256 K256 schedule"
        startup = stages // 2
        mfma_count = 8 if stages == 8 or require_bn64_four_stage else 16
        for index, (stage, body) in enumerate(compute):
            vector_ops = re.findall(r"^\s*(v_\w+)\b", body, re.M)
            mfmas = [i for i, op in enumerate(vector_ops) if op.startswith("v_mfma_")]
            assert len(mfmas) == mfma_count, f"expected {mfma_count} MFMA per Compute"
            if index < startup:
                continue
            gaps = [right - left - 1 for left, right in zip(mfmas, mfmas[1:])]
            assert gaps == [7] * (mfma_count - 1), f"Compute{index}/step{stage} vector gaps: {gaps}"
            intervals += len(gaps)
        print(f"INTERLEAVE PASS: {intervals} steady MFMA intervals each contain exactly 7 vector instructions")
    metadata = re.findall(r"^\s*\.(?:vgpr_count|sgpr_count|group_segment_fixed_size|private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count):.*$", text, re.M)
    print(f"ISA PASS: {len(memory)} Memory stages without vector instructions; {len(compute)} MFMA/VALU Compute stages")
    print("\n".join(metadata))


@pytest.mark.parametrize("k,block_n", [(256, 256), (256, 128), (384, 128), (512, 128), (640, 128)])
def test_blockscaled_8stage(k, block_n):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=129, n=block_n * 5, k=k, experts=4, topk=2, block_n=block_n, graph=True)


@pytest.mark.parametrize("block_n", [128, 256])
@pytest.mark.parametrize("tiles,splits", [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (3, 2)])
def test_n_boundaries(tiles, splits, block_n):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=257, n=block_n * tiles * splits, k=256, experts=4, topk=2,
             block_n=block_n, splits=splits, graph=True)


@pytest.mark.parametrize("block_n", [128, 256])
def test_persistent_reuse_and_empty_work(block_n):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    # >256 logical tasks forces at least one physical CTA to reuse its LDS.
    args, _, _ = make_case(128, 256, 256, 257, 1, 42)
    a, w, a_scales, b_scales = args[1:5]
    ids = torch.full((257 * 256,), (1 << 24) | 128, dtype=torch.int32, device="cuda")
    routes = torch.zeros(ids.numel(), device="cuda")
    ids[:128] = torch.arange(128, dtype=torch.int32, device="cuda")
    routes[:128] = 1
    expert_ids = torch.arange(257, dtype=torch.int32, device="cuda")
    valid = torch.tensor([ids.numel()], dtype=torch.int32, device="cuda")
    kernel = flydsl_moe_gemm_8wave_down(n=256, k=256, topk=1, num_experts=257, block_n=block_n)
    output = torch.full((128, 1, 256), torch.nan, dtype=torch.bfloat16, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    # Give the only active expert an independent, known pre-shuffle matrix.
    w0 = (torch.arange(256 * 256, device="cuda").reshape(256, 256) % 17 - 8).to(torch.float8_e4m3fn)
    w[0].copy_(shuffle_weight(w0[None], layout=(16, 16))[0])
    ref = torch.zeros(128, 256, device="cuda")
    for kb in range(2):
        partial = a[:, 0, kb * 128:(kb + 1) * 128].float() @ w0[:, kb * 128:(kb + 1) * 128].float().T
        scale = a_scales[kb, :, None] * b_scales[0, :, kb].repeat_interleave(128)[None, :]
        ref += partial * scale
    kernel(output, a, w, a_scales, b_scales, ids, routes, expert_ids, valid, counter)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:, 0], ref.to(torch.bfloat16), rtol=0.01, atol=0.01)
    assert counter.item() == 257 + 256
    valid.zero_()
    output.fill_(torch.nan)
    kernel(output, a, w, a_scales, b_scales, ids, routes, expert_ids, valid, counter)
    torch.cuda.synchronize()
    assert torch.isnan(output).all()
    assert counter.item() == 256


@pytest.mark.parametrize("block_n", [128, 256])
def test_aiter_sorting_contract(block_n):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=513, n=1280, k=256, experts=16, topk=4, block_n=block_n,
             routing_mode="aiter", graph=True)


@pytest.mark.parametrize("block_n", [128, 256])
@pytest.mark.parametrize("seed", [2026, 42])
def test_cached_vgpr_paths(block_n, seed):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=257, n=block_n * 5, k=256, experts=4, topk=2, block_n=block_n,
             seed=seed, graph=True)


@pytest.mark.parametrize("steady_vmcnt", [6, 9])
@pytest.mark.parametrize("tiles", [1, 2, 5])
def test_bn256_sixteen_stage(tiles, steady_vmcnt):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=257, n=256 * tiles, k=256, experts=4, topk=2, block_n=256,
             bn256_stages=16, steady_vmcnt=steady_vmcnt, graph=True)


@pytest.mark.parametrize("prefetch_distance", [1, 2, 3])
@pytest.mark.parametrize("n,splits", [(64, 1), (128, 1), (192, 1), (640, 1), (384, 2)])
def test_bn64_four_stage(n, splits, prefetch_distance):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=257, n=n, k=256, experts=4, topk=2, block_n=64,
             splits=splits, prefetch_distance=prefetch_distance, graph=True)


@pytest.mark.parametrize("n,splits", [(128, 1), (256, 1), (384, 1), (512, 4), (1536, 4), (6144, 4)])
def test_att_tuned_bn128(n, splits):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("native gfx950 MFMA is required")
    run_case(tokens=257, n=n, k=256, experts=4, topk=2, block_n=128, splits=splits,
             seed=2026, tuned=True, graph=True, routing_mode="aiter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--bn256-stages", type=int, choices=(8, 16), default=8)
    parser.add_argument("--steady-vmcnt", type=int, choices=(6, 9), default=6)
    parser.add_argument("--prefetch-distance", type=int, choices=(1, 2, 3))
    parser.add_argument("--tuned", action="store_true", help="use the ATT-tuned BN128 kernel options; --splits remains explicit")
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--routing-mode", choices=("manual", "aiter"), default="manual")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--isa", type=Path)
    parser.add_argument("--require-no-scratch", action="store_true")
    parser.add_argument("--require-nt", action="store_true")
    parser.add_argument("--require-seven-valu", action="store_true")
    parser.add_argument("--require-four-stage", action="store_true")
    parser.add_argument("--require-bn64-four-stage", action="store_true")
    parser.add_argument("--require-sixteen-stage", action="store_true")
    parser.add_argument("--require-steady-vmcnt", type=int, choices=(0, 3, 6, 9))
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--require-no-agpr", action="store_true")
    parser.add_argument("--require-no-compute-address", action="store_true")
    options = vars(parser.parse_args())
    isa = options.pop("isa")
    audit_options = {name: options.pop(name) for name in (
        "require_no_scratch", "require_nt", "require_seven_valu", "require_four_stage",
        "require_cached", "require_no_agpr", "require_no_compute_address",
        "require_sixteen_stage", "require_steady_vmcnt", "require_bn64_four_stage",
    )}
    if isa:
        audit_isa(isa, **audit_options)
    else:
        if any(value is not None and value is not False for value in audit_options.values()):
            parser.error("--require-* options need --isa")
        run_case(**options)


if __name__ == "__main__":
    main()