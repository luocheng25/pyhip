# SPDX-License-Identifier: MIT
"""Winning packed path plus retained PyHIP/FlyDSL BN32/BN64 baselines."""

import argparse
from pathlib import Path
import re

import aiter
import pytest
import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import pyhip
from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_down
from moe_8wave_down import flydsl_moe_gemm_8wave_down as legacy_flydsl_down
from moe_multistage_down import ATT_TUNED_BN128_CONFIG, flydsl_moe_gemm_8wave_down
from moe_multistage_pipeline import DownReduceWorkspace, compile_packed_down_reduce
from moe_multistage_reduce import make_moe_sum


CANDIDATES = {
    "pyhip": ("PyHIP", 64, None),
    "flydsl_bn32": ("FlyDSL BN32", 32, legacy_flydsl_down),
    "flydsl_bn64": ("FlyDSL BN64", 64, legacy_flydsl_down),
    "4stage_bn128_tuned": ("4-stage BN128 tuned", 128, flydsl_moe_gemm_8wave_down),
}
DEFAULT_CANDIDATES = tuple(CANDIDATES)
ACTIVATION_QUANT = aiter.get_hip_quant(aiter.QuantType.per_1x128)


def make_pyhip_down(*, n, k, topk, num_experts):
    """Existing PyHIP BN64/OC1 is a test baseline, not another shipped kernel."""
    def down(output, input_q, weight, input_scales, weight_scales,
             ids, routes, experts, valid, counter):
        counter.zero_()
        moe_gemm_8wave_down(
            [256], [512], output.numel() * output.element_size() > (1 << 32),
            "fp8", 256, 64, num_experts, n, k, 1, False, True, topk,
            ids.data_ptr(), routes.data_ptr(), experts.data_ptr(), valid.data_ptr(),
            weight.data_ptr(), weight_scales.data_ptr(), input_q.data_ptr(), input_scales.data_ptr(),
            output.data_ptr(), input_q.shape[0], counter,
        )
        return output
    down.config = {"block_m": 256, "block_n": 64, "num_oc_splits": 1,
                   "persistent_workgroups": 256, "output_layout": "routed"}
    return down


def with_torch_sum(down, workspace):
    """Retain routed baseline + preallocated TOPK sum on the shared storage."""
    def buffers(args):
        output, input_q, expert_ids = args[0], args[1], args[7]
        storage, _ = workspace.prepare(output, input_q, expert_ids)
        tokens, topk = input_q.shape[:2]
        return storage[:tokens * topk].view(tokens, topk, output.shape[1])

    def launch(*args):
        middle = buffers(args)
        down(middle, *args[1:])
        torch.sum(middle, dim=1, out=args[0])
        return args[0]

    def components(*args):
        middle = buffers(args)
        return {"gemm": lambda: down(middle, *args[1:]),
                "reduce": lambda: torch.sum(middle, dim=1, out=args[0])}

    launch.benchmark_components = components
    launch.poison_workspace = lambda *args: buffers(args).fill_(torch.nan)
    launch.config = {**getattr(down, "config", {}), "output_layout": "routed",
                     "reduction": "torch", "includes_inverse": False, "includes_reduce": True}
    return launch


def make_routing(tokens, topk, experts, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(tokens, experts, generator=generator, device="cuda", dtype=torch.float32)
    ids = scores.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)
    weights = torch.rand(tokens, topk, generator=generator, device="cuda", dtype=torch.float32)
    return ids, weights / weights.sum(dim=1, keepdim=True)


def torch_reference_down(input_q, input_scales_k_major, weight_q, weight_scales, topk_ids, topk_weights):
    """Independent K128 block-scale reference, routing multiply, then BF16."""
    tokens, topk, k = input_q.shape
    experts, n, _ = weight_q.shape
    rows = tokens * topk
    a = input_q.float().reshape(rows, k // 128, 128)
    scales = input_scales_k_major.view(k // 128, rows).t().float()
    output = torch.empty((rows, n), dtype=torch.bfloat16, device=input_q.device)
    expert_per_row, routing = topk_ids.reshape(-1), topk_weights.reshape(-1)
    for expert in range(experts):
        row_ids = torch.where(expert_per_row == expert)[0]
        if row_ids.numel() == 0:
            continue
        accum = torch.zeros((row_ids.numel(), n), dtype=torch.float32, device=input_q.device)
        for kb in range(k // 128):
            for bn in range(n // 128):
                w = weight_q[expert, bn * 128:(bn + 1) * 128, kb * 128:(kb + 1) * 128]
                partial = a[row_ids, kb] @ w.float().t()
                factor = scales[row_ids, kb, None] * weight_scales[expert, bn, kb]
                accum[:, bn * 128:(bn + 1) * 128] += partial * factor
        output[row_ids] = (accum * routing[row_ids, None]).to(torch.bfloat16)
    return output.view(tokens, topk, n)


def make_case(tokens, n, experts, topk, seed):
    """Real BF16→OCP-FP8 quantization, (16,16) weights and AITER M256 sorting."""
    assert n % 512 == 0 and 0 < topk <= min(experts, 255)
    torch.manual_seed(seed)
    a_bf16 = torch.randn((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn((experts, n, 256), dtype=torch.bfloat16, device="cuda")
    a, sa = ACTIVATION_QUANT(a_bf16, quant_dtype=dtypes.fp8, transpose_scale=True)
    blocks = w_bf16.view(experts, n // 128, 128, 2, 128).permute(0, 1, 3, 2, 4).contiguous()
    qblocks, sb = aiter.pertoken_quant(blocks.view(experts, -1, 128 * 128), quant_dtype=dtypes.fp8)
    w = qblocks.view(experts, n // 128, 2, 128, 128).permute(0, 1, 3, 2, 4).contiguous().view(experts, n, 256)
    sb = sb.view(experts, n // 128, 2)
    assignments, routing = make_routing(tokens, topk, experts, seed + 1)
    ids, routes, eids, valid, _ = moe_sorting(assignments, routing, experts, n, torch.bfloat16, 256, None, None, 0)
    reference = torch_reference_down(a, sa, w, sb, assignments, routing)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    return (a, shuffle_weight(w, layout=(16, 16)), sa, sb, ids, routes, eids, valid, counter), reference


def error_stats(output, reference):
    # Chunking avoids an extra full-shape FP32 allocation in the large case.
    max_abs, total_abs, mismatches, nonfinite = 0.0, 0.0, 0, False
    dot, denominator = 0.0, 0.0
    for begin in range(0, output.shape[0], 256):
        actual, expected = output[begin:begin + 256].float(), reference[begin:begin + 256].float()
        error = (actual - expected).abs()
        max_abs = max(max_abs, error.max().item())
        total_abs += error.sum().item()
        mismatches += (error > 0.01 + 0.01 * expected.abs()).sum().item()
        nonfinite |= not torch.isfinite(actual).all().item()
        # Keep the original pyhip.calc_diff definition without allocating
        # two full default-shape float64 tensors. Reporting is outside timers.
        actual64, expected64 = actual.double(), expected.double()
        dot += (actual64 * expected64).sum().item()
        denominator += (actual64.square() + expected64.square()).sum().item()
    return {"status": "FAIL" if mismatches or nonfinite else "PASS", "mismatch_count": mismatches,
            "mismatches": f"{mismatches}/{output.numel()}", "max_abs": max_abs,
            "mean_abs": total_abs / output.numel(), "diff": 1 - 2 * dot / denominator if denominator else 0.0}


def unpack_routes(source, ids, valid, tokens, topk):
    """Test-only Torch decoding, outside timers; never a production restore."""
    limit = int(valid[0].item())
    n = source.shape[1]
    sorted_values = source[:limit].view(-1, n // 64, 256, 64).permute(0, 2, 1, 3).reshape(limit, n)
    encoded = ids[:limit].to(torch.int64)
    token, slot = encoded & 0xFFFFFF, (encoded >> 24) & 0xFF
    live = (token < tokens) & (slot < topk)
    result = torch.empty((tokens, topk, n), dtype=source.dtype, device=source.device)
    result[token[live], slot[live]] = sorted_values[live]
    return result


def print_markdown_table(headers, rows):
    widths = [max(len(str(header)), *(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
    def formatted(row):
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
    print(formatted(headers))
    print(formatted(["-" * width for width in widths]))
    for row in rows:
        print(formatted(row))


def performance(us, flops, padded_flops, nbytes, ideal_nbytes):
    return {"elapsed_us": us, "us": f"{us:.3f}" if us is not None else "N/A",
            **{name: f"{work / us / 1e6:.3f}" if us is not None else "N/A" for name, work in (
                ("effective_tflops", flops), ("padded_tflops", padded_flops),
                ("tb_per_s", nbytes), ("tb_per_s_ideal", ideal_nbytes))}}


def run_test(tokens=16384, model_dim=6144, inter_dim=256, experts=384, topk=8,
             block_m=256, num_oc_splits=1, seed=1234, candidates=None, profile=False,
             reduce_output=True, component_diagnostics=True):
    """Compare the winner and retained baselines. Down-only is packed vs routed.

    num_oc_splits describes the baselines and must remain1; the winner
    is always OC4. BF16 routes are checked before TOPK sum so rounding near
    cancellation is not confused with a reduction correctness regression.
    """
    assert torch.cuda.is_available() and torch.cuda.get_device_properties().gcnArchName.startswith("gfx950")
    assert inter_dim == block_m == 256 and model_dim % 512 == 0 and num_oc_splits == 1
    candidates = list(DEFAULT_CANDIDATES if candidates is None else candidates)
    assert candidates and all(key in DEFAULT_CANDIDATES for key in candidates)
    assert not profile or len(candidates) == 1
    args, reference_routes = make_case(tokens, model_dim, experts, topk, seed)
    a, b, sa, sb, ids, routes, eids, valid, counter = args
    baseline = make_pyhip_down(n=model_dim, k=256, topk=topk, num_experts=experts)
    baseline_routes = torch.empty_like(reference_routes)
    baseline(baseline_routes, *args)
    torch.cuda.synchronize()
    baseline_error = error_stats(baseline_routes, reference_routes)
    assert baseline_error["status"] == "PASS", f"PyHIP down vs Torch: {baseline_error}"
    reference = baseline_routes.sum(dim=1) if reduce_output else reference_routes
    del baseline_routes
    output = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    workspace = DownReduceWorkspace()
    storage, _ = workspace.prepare(output, a, eids)
    routed = storage[:tokens * topk].view(tokens, topk, model_dim)
    packed = storage[:eids.numel() * 256]
    pipeline = compile_packed_down_reduce(n=model_dim, k=256, topk=topk, num_experts=experts, workspace=workspace)
    down = flydsl_moe_gemm_8wave_down(n=model_dim, k=256, topk=topk, num_experts=experts)
    inputs = (output, *args)
    valid_blocks = int(valid[0].item()) // 256
    unique_experts = torch.unique(eids[:valid_blocks]).numel()
    flops = 2 * tokens * topk * model_dim * 256
    padded_flops = 2 * valid_blocks * 256 * model_dim * 256
    weight_bytes = model_dim * 256 * b.element_size()
    intermediate_bytes = tokens * topk * model_dim * 2
    other_bytes = a.numel() * a.element_size() + intermediate_bytes
    down_bytes = other_bytes + valid_blocks * weight_bytes
    down_ideal = other_bytes + unique_experts * weight_bytes
    reduce_bytes = intermediate_bytes + output.numel() * 2 if reduce_output else 0
    results = []

    for key in candidates:
        name, block_n, factory = CANDIDATES[key]
        winner = key == "4stage_bn128_tuned"
        storage.fill_(torch.nan)
        output.fill_(torch.nan)
        if winner:
            pipeline.poison_workspace(*inputs)
            launch = (lambda: pipeline(*inputs)) if reduce_output else (lambda: down(packed, *args))
            config = pipeline.config if reduce_output else down.config
            components = pipeline.benchmark_components(*inputs) if reduce_output else {}
        else:
            routed_down = baseline if factory is None else factory(
                n=model_dim, k=256, topk=topk, num_experts=experts,
                block_m=256, block_n=block_n, num_oc_splits=1,
            )
            routed_down.config = {"block_m": 256, "block_n": block_n, "num_oc_splits": 1,
                                  "persistent_workgroups": 256, "output_layout": "routed"}
            routed_pipeline = with_torch_sum(routed_down, workspace)
            launch = (lambda: routed_pipeline(*inputs)) if reduce_output else (lambda: routed_down(routed, *args))
            config = routed_pipeline.config if reduce_output else routed_down.config
            components = routed_pipeline.benchmark_components(*inputs) if reduce_output else {}
        elapsed, times = None, {}
        inverse_bytes = (valid_blocks * 256 + 2 * tokens * topk) * 4 if reduce_output and winner else 0
        total_bytes, ideal_bytes = down_bytes + reduce_bytes + inverse_bytes, down_ideal + reduce_bytes + inverse_bytes
        if profile:
            for _ in range(23):
                launch()
                torch.cuda.synchronize()
        else:
            _, elapsed = pyhip.run_perftest(launch, num_warmup=2, num_iters=10, num_copies=1,
                num_flops=padded_flops, num_bytes=total_bytes, num_verbose=1,
                num_name=key + ("_total" if reduce_output else "_down"),
                num_spec_tag=f"M={tokens * topk},N={model_dim},K=256")
            if component_diagnostics:
                for label, component in components.items():
                    _, times[label] = pyhip.run_perftest(component, num_warmup=2, num_iters=10,
                                                       num_copies=1, num_verbose=0, num_name=f"{key}_{label}")
            launch()
        torch.cuda.synchronize()
        if winner:
            decoded = unpack_routes(packed, ids, valid, tokens, topk)
            down_error = error_stats(decoded, reference_routes)
            assert down_error["status"] == "PASS", f"packed down vs Torch: {down_error}"
        actual = output if reduce_output else decoded if winner else routed
        errors = error_stats(actual, reference)
        down_us = times.get("gemm") if reduce_output else elapsed
        reduce_us = times.get("reduce")
        stats = {"name": name, "candidate": key, "block_n": block_n, "config": config,
                 "valid_expert_blocks": valid_blocks, "unique_experts": unique_experts,
                 "rw_bytes": total_bytes, "ideal_rw_bytes": ideal_bytes,
                 "down_rw_bytes": down_bytes, "down_ideal_rw_bytes": down_ideal,
                 "reduce_rw_bytes": reduce_bytes, "inverse_rw_bytes": inverse_bytes,
                 "measurement_scope": "down+reduce+inverse_if_needed" if reduce_output else "down",
                 "components_us": times, **errors,
                 **performance(elapsed, flops, padded_flops, total_bytes, ideal_bytes),
                 **{"down_" + name: value for name, value in performance(down_us, flops, padded_flops, down_bytes, down_ideal).items()},
                 "reduce_tb_per_s": f"{reduce_bytes / reduce_us / 1e6:.3f}" if reduce_us is not None else "N/A"}
        results.append(stats)
        if winner:
            del decoded

    props = torch.cuda.get_device_properties(a.device)
    print(f"\nShape: tokens={tokens}, TOPK={topk}, E={experts}, N={model_dim}, K=256; {props.name}, CUs={props.multi_processor_count}")
    print("Winner: M256/N128, OC4, persistent256, PF3, B SC1, packed NT output; custom reduce256/2048/NT.")
    print("Reference: independent Torch block-scale routes; final sum uses validated PyHIP BF16 routes. rtol=atol=0.01.")
    print("Timing: same A/B/scales/routing/intermediate/final addresses, counter reset included, warmup2/iters10.")
    print("Down is independently timed; Total includes inverse rebuild and reduce, never a component sum.")
    print("Bandwidth: A once + B per M-block (or once per unique expert, ideal) + valid BF16 writes; not HBM counters.")
    if not reduce_output:
        print("Down-only compares winner PACKED output with baseline ROUTED output after untimed Torch decoding.")
    if profile:
        print("Profile:23 direct launches, no performance timings.")
    rows = [[title, *(r["config"].get(field, "N/A") for r in results)] for title, field in (
        ("Block N", "block_n"), ("OC splits", "num_oc_splits"),
        ("Persistent CTAs", "persistent_workgroups"), ("Down output layout", "output_layout"),
        ("Reduce implementation", "reduction"),
    )]
    for title, field in (("Status", "status"), ("Valid expert blocks", "valid_expert_blocks"), ("Unique experts", "unique_experts"),
                         ("Down time (us)", "down_us"), ("Down effective TF/s", "down_effective_tflops"),
                         ("Down padded TF/s", "down_padded_tflops"), ("Down TB/s (B per M-block)", "down_tb_per_s"),
                         ("Down TB/s (B once/expert, ideal)", "down_tb_per_s_ideal")):
        rows.append([title, *(r[field] for r in results)])
    if reduce_output:
        for label, title in (("inverse", "Invert + fill time (us)"), ("reduce", "Reduce time (us)")):
            rows.append([title, *(f"{r['components_us'][label]:.3f}" if label in r["components_us"] else "N/A" for r in results)])
        for title, field in (("Reduce TB/s", "reduce_tb_per_s"), ("Total Time (us)", "us"),
                             ("Total effective TF/s", "effective_tflops"), ("Total padded TF/s", "padded_tflops"),
                             ("Total TB/s (B per M-block)", "tb_per_s"), ("Total TB/s (B once/expert, ideal)", "tb_per_s_ideal")):
            rows.append([title, *(r[field] for r in results)])
    by_name = {r["name"]: r for r in results}
    for baseline_name in ("PyHIP", "FlyDSL BN64"):
        for field, scope in (("down_elapsed_us", "Down"), ("elapsed_us", "Total")):
            if scope == "Total" and not reduce_output:
                continue
            base = by_name.get(baseline_name, {}).get(field)
            rows.append([f"{scope} speedup vs {baseline_name}", *(
                f"{base / r[field]:.3f}x" if base is not None and r[field] is not None else "N/A" for r in results)])
    rows.extend([["Max abs error", *(f"{r['max_abs']:.6g}" for r in results)],
                 ["Mean abs error", *(f"{r['mean_abs']:.6g}" for r in results)],
                 ["calc_diff", *(f"{r['diff']:.6g}" for r in results)],
                 ["Mismatches", *(r["mismatches"] for r in results)]])
    print_markdown_table(["Metric", *(r["name"] for r in results)], rows)
    assert all(r["status"] == "PASS" for r in results), "correctness failed"
    print("PASS: all executed kernels match the validated reference")
    return results


def audit_winner_isa(path):
    text = Path(path).read_text()
    memory = re.findall(r"MOE8_MEMORY_BEGIN_(\d+)(.*?)MOE8_MEMORY_END_\1", text, re.S)
    compute = re.findall(r"MOE8_COMPUTE_BEGIN_(\d+)(.*?)MOE8_COMPUTE_END_\1", text, re.S)
    assert memory and [i for i, _ in memory] == [i for i, _ in compute]
    assert set(i for i, _ in compute) == {"0", "1"}
    for _, body in memory:
        assert not re.search(r"^\s*(?:v_\w+|ds_write\w*)\b", body, re.M), "VALU/C-LDS in Memory"
    address_ops = re.compile(r"^\s*(?:v_(?:(?:and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|mbcnt|bit|brev|alignbit|cmp)\w*|(?:add|sub|mul|mad)\w*_(?:u|i)\d+\w*)|s_(?:add|sub|mul|and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|brev)\w*)\b", re.M)
    intervals = 0
    for index, (_, body) in enumerate(compute):
        ops = re.findall(r"^\s*(v_\w+)\b", body, re.M)
        mfmas = [i for i, op in enumerate(ops) if "mfma" in op]
        assert len(mfmas) == 16 and all("16x16x128" in ops[i] for i in mfmas)
        assert not address_ops.search(body) and not re.search(r"\bds_(?:read|write)\w*", body)
        if index:
            assert [b - a - 1 for a, b in zip(mfmas, mfmas[1:])] == [7] * 15
            intervals += 15
    assert re.search(r"buffer_load_\w+.*\blds\b", text)
    stores = re.findall(r"^\s*buffer_store_\w+.*$", text, re.M)
    assert stores and all(re.search(r"\bnt\b", op) for op in stores)
    assert "v_cvt_pk_bf16_f32" in text and "v_permlane16_swap_b32" in text
    assert not re.search(r"^\s*(?:scratch_|v_accvgpr_)\w+", text, re.M)
    for field in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"):
        values = re.findall(rf"\.{field}:\s*(\d+)", text)
        assert values and all(int(v) == 0 for v in values), (field, values)
    vgprs = re.findall(r"\.vgpr_count:\s*(\d+)", text)
    offset = re.findall(r"\.amdhsa_accum_offset\s+(\d+)", text)
    assert vgprs and offset and int(vgprs[0]) <= int(offset[0]), "AGPR allocation"
    print(f"ISA PASS: {len(memory)} pure Memory / address-free Compute stages, {intervals} seven-VALU intervals, no scratch/AGPR, NT output")


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")


@pytest.mark.parametrize("tokens,n,experts,topk", [
    (65, 512, 4, 1), (257, 512, 4, 2), (513, 1024, 8, 8), (129, 6144, 4, 2),
    (257, 1536, 4, 2), (257, 2048, 4, 2), (257, 2560, 4, 2), (128, 512, 257, 1),
])
def test_packed_pipeline(tokens, n, experts, topk):
    require_gpu()
    args, ref = make_case(tokens, n, experts, topk, 2026)
    baseline = torch.empty_like(ref)
    make_pyhip_down(n=n, k=256, topk=topk, num_experts=experts)(baseline, *args)
    assert error_stats(baseline, ref)["status"] == "PASS"
    expected = baseline.sum(dim=1)
    guard = torch.full((expected.numel() + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = guard[n:-n].view_as(expected)
    inputs = (output, *args)
    pipeline = compile_packed_down_reduce(n=n, topk=topk, num_experts=experts)
    def check():
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.isfinite(output).all() and torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
    pipeline.poison_workspace(*inputs)
    assert pipeline(*inputs) is output
    torch.cuda.synchronize()
    check()
    decoded = unpack_routes(pipeline.workspace.data, args[4], args[7], tokens, topk)
    assert error_stats(decoded, ref)["status"] == "PASS"
    assert args[-1].item() == (args[7][0].item() // 256) * 4 + 256
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        pipeline(*inputs)
    for _ in range(2):
        pipeline.poison_workspace(*inputs)
        output.fill_(torch.nan)
        args[-1].fill_(0x123456)
        capture.replay()
        torch.cuda.synchronize()
        check()
    old_ids, old_valid = args[4].clone(), args[7].clone()
    args[4].fill_((topk << 24) | tokens)
    pipeline.poison_workspace(*inputs)
    capture.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    args[7].zero_()
    capture.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0 and args[-1].item() == 256
    args[4].copy_(old_ids)
    args[7].copy_(old_valid)
    pipeline.poison_workspace(*inputs)
    capture.replay()
    torch.cuda.synchronize()
    check()


@pytest.mark.parametrize("n,topk", [(512, 1), (512, 8), (6144, 8)])
def test_packed_reducer(n, topk):
    require_gpu()
    tokens, rows = 19, 256
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn((tokens, topk, n), generator=generator, device="cuda").to(torch.bfloat16)
    permutation = torch.randperm(tokens * topk, generator=generator, device="cuda")
    sorted_values = torch.full((rows, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    sorted_values[:tokens * topk] = values.reshape(-1, n)[permutation]
    source = sorted_values.view(1, 256, n // 64, 64).permute(0, 2, 1, 3).contiguous().view(rows, n)
    inverse = torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
    inverse.view(-1)[permutation] = torch.arange(tokens * topk, dtype=torch.int32, device="cuda")
    output = torch.full((tokens, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    reduce = make_moe_sum(n=n, topk=topk)
    reduce(output, source, inverse)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, values.sum(dim=1), rtol=0.01, atol=0.01)
    inverse.fill_(-1)
    reduce(output, source, inverse)
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("candidate", ["4stage_bn128_tuned", "flydsl_bn32", "flydsl_bn64"])
def test_cross_k_fma_cancellation(candidate):
    require_gpu()
    tokens, topk, n = 3, 2, 512
    a = torch.zeros((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    a[..., 0] = a[..., 128] = 1
    w = torch.zeros((2, n, 256), dtype=torch.bfloat16, device="cuda")
    w[0, :, 0], w[0, :, 128], w[1, :, 0] = 16, -5, -4
    sa = torch.ones((2, tokens * topk), device="cuda")
    sb = torch.ones((2, n // 128, 2), device="cuda")
    sb[0, :, 1] = 2.3968749046325684
    ids = torch.full((512,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    routes = torch.zeros(512, device="cuda")
    for expert in range(2):
        ids[expert * 256:expert * 256 + tokens] = torch.arange(tokens, dtype=torch.int32, device="cuda") | (expert << 24)
        routes[expert * 256:expert * 256 + tokens] = 0.5
    output = torch.empty((tokens, n), dtype=torch.bfloat16, device="cuda")
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(16, 16)),
            sa, sb, ids, routes, torch.arange(2, dtype=torch.int32, device="cuda"),
            torch.tensor([512], dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda"))
    if candidate == "4stage_bn128_tuned":
        pipeline = compile_packed_down_reduce(n=n, topk=topk, num_experts=2)
    else:
        pipeline = with_torch_sum(legacy_flydsl_down(n=n, k=256, topk=topk, num_experts=2,
                                                    block_n=CANDIDATES[candidate][1]), DownReduceWorkspace())
    pipeline.poison_workspace(*args)
    pipeline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0.01, atol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=0.01, atol=0.01)


@pytest.mark.parametrize("reduce_output", [False, True])
def test_down_and_total_metrics(reduce_output, capsys):
    require_gpu()
    results = run_test(tokens=513, model_dim=512, experts=4, topk=2, reduce_output=reduce_output)
    printed = capsys.readouterr().out
    assert [r["candidate"] for r in results] == list(DEFAULT_CANDIDATES)
    assert all(r["status"] == "PASS" for r in results)
    for row in results:
        assert row["elapsed_us"] > 0 and row["down_elapsed_us"] > 0
        assert row["unique_experts"] == 4 and row["valid_expert_blocks"] > 4
        assert row["rw_bytes"] == row["down_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["ideal_rw_bytes"] == row["down_ideal_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["down_effective_tflops"] == f"{2 * 513 * 2 * 512 * 256 / row['down_elapsed_us'] / 1e6:.3f}"
        if reduce_output:
            assert row["down_elapsed_us"] == row["components_us"]["gemm"]
            assert row["reduce_tb_per_s"] == f"{row['reduce_rw_bytes'] / row['components_us']['reduce'] / 1e6:.3f}"
            assert "Total Time (us)" in printed
        else:
            assert row["elapsed_us"] == row["down_elapsed_us"] and row["reduce_rw_bytes"] == row["inverse_rw_bytes"] == 0
    assert results[-1]["config"]["output_layout"] == "packed"
    assert "FlyDSL BN32" in printed and "FlyDSL BN64" in printed and "speedup vs FlyDSL BN64" in printed
    assert "Mean abs error" in printed and "calc_diff" in printed
    for row in results[1:3]:
        assert row["config"]["output_layout"] == "routed"
        assert row["config"]["block_n"] == (32 if row["candidate"] == "flydsl_bn32" else 64)
        if reduce_output:
            assert row["config"]["reduction"] == "torch" and not row["config"]["includes_inverse"]


@pytest.mark.parametrize("kwargs", [{"n": 384}, {"n": 512, "k": 384}, {"n": 512, "num_oc_splits": 1}])
def test_reject_nonwinner_configuration(kwargs):
    with pytest.raises((AssertionError, TypeError)):
        flydsl_moe_gemm_8wave_down(topk=1, num_experts=1, **kwargs)


@pytest.mark.parametrize("argv,scope", [([], True), (["--mode", "down"], False),
    (["--candidate", "flydsl_bn32", "flydsl_bn64"], True),
    (["--mode", "down", "--candidate", "flydsl_bn32", "flydsl_bn64"], False),
    (["--profile", "--candidate", "4stage_bn128_tuned"], False),
    (["--profile", "--mode", "down-reduce", "--candidate", "4stage_bn128_tuned"], True)])
def test_cli_scope(monkeypatch, argv, scope):
    import sys
    module = sys.modules[__name__]
    calls = []
    monkeypatch.setattr(sys, "argv", [__file__, *argv])
    monkeypatch.setattr(module, "run_test", lambda **kw: calls.append(kw))
    main()
    assert len(calls) == 1 and calls[0]["reduce_output"] is scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--model-dim", type=int, default=6144)
    parser.add_argument("--inter-dim", type=int, choices=[256], default=256)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--candidate", choices=DEFAULT_CANDIDATES, nargs="+", default=None)
    parser.add_argument("--mode", choices=("down-reduce", "down"), default=None)
    parser.add_argument("--profile", action="store_true", help="23 direct launches; defaults to packed down-only")
    args = parser.parse_args()
    if args.profile and (args.candidate is None or len(args.candidate) != 1):
        parser.error("--profile requires exactly one --candidate")
    mode = args.mode or ("down" if args.profile else "down-reduce")
    run_test(tokens=args.tokens, model_dim=args.model_dim, inter_dim=args.inter_dim,
             experts=args.experts, topk=args.topk, seed=args.seed, candidates=args.candidate,
             profile=args.profile, reduce_output=mode == "down-reduce")


if __name__ == "__main__":
    main()