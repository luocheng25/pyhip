# SPDX-License-Identifier: MIT

import argparse
from functools import partial

import aiter
import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import pyhip
from pyhip.contrib.moe_gemm_8wave import moe_gemm_8wave_down

from moe_8wave_down import flydsl_moe_gemm_8wave_down
from moe_multistage_down import (
    ATT_TUNED_BN128_CONFIG,
    flydsl_moe_gemm_8wave_down as flydsl_moe_gemm_8wave_down_8stage,
)
from moe_multistage_down_mfma32 import (
    MFMA32_EXPERIMENT_CONFIG, flydsl_moe_gemm_8wave_down_mfma32,
)
from moe_multistage_pipeline import (
    DownReduceWorkspace, compile_packed_down_reduce,
)


CANDIDATES = {
    "pyhip": ("PyHIP", 64, None),
    "flydsl_bn32": ("FlyDSL BN32", 32, flydsl_moe_gemm_8wave_down),
    "flydsl_bn64": ("FlyDSL BN64", 64, flydsl_moe_gemm_8wave_down),
    # Keep the existing selection key; K256 now has two Memory/Compute pairs.
    "8stage_bn128": ("4-stage BN128", 128, flydsl_moe_gemm_8wave_down_8stage),
    "8stage_bn256": ("8-stage BN256", 256, flydsl_moe_gemm_8wave_down_8stage),
    "4stage_bn128_tuned": ("4-stage BN128 tuned", 128, flydsl_moe_gemm_8wave_down_8stage),
}
DEFAULT_CANDIDATES = tuple(CANDIDATES)
CANDIDATES["4stage_bn128_mfma32"] = ("4-stage BN128 MFMA32 experimental", 128, flydsl_moe_gemm_8wave_down_mfma32)
CANDIDATE_OPTIONS = {"4stage_bn128_tuned": ATT_TUNED_BN128_CONFIG,
                     "4stage_bn128_mfma32": MFMA32_EXPERIMENT_CONFIG}
CANDIDATE_WEIGHT_LAYOUTS = {"4stage_bn128_mfma32": (32, 16)}
# The stage-count/prefetch experiments remain opt-in.
CANDIDATES.update({
    f"16stage_bn256_vmcnt{count}": (
        f"16-stage BN256 vmcnt{count}", 256,
        partial(flydsl_moe_gemm_8wave_down_8stage, bn256_stages=16, steady_vmcnt=count),
    )
    for count in (6, 9)
})
CANDIDATES.update({
    f"4stage_bn64_pf{distance}": (
        f"4-stage BN64 PF{distance}", 64,
        partial(flydsl_moe_gemm_8wave_down_8stage, prefetch_distance=distance),
    )
    for distance in (1, 2, 3)
})

ACTIVATION_QUANT = aiter.get_hip_quant(aiter.QuantType.per_1x128)


def make_pyhip_down(*, n, k, topk, num_experts, block_m=256, block_n=64, num_oc_splits=1):
    """Adapt the existing PyHIP launch to the same ten-tensor down interface."""
    def down(output, input_q, weight_shuffled, input_scales, weight_scales,
             sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, counter):
        counter.zero_()
        moe_gemm_8wave_down(
            [256], [512], output.numel() * output.element_size() > (1 << 32),
            "fp8", block_m, block_n, num_experts, n, k, num_oc_splits,
            False, True, topk, sorted_ids.data_ptr(), sorted_weights.data_ptr(),
            sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(),
            weight_shuffled.data_ptr(), weight_scales.data_ptr(),
            input_q.data_ptr(), input_scales.data_ptr(), output.data_ptr(), input_q.shape[0], counter,
        )
        return output

    down.config = {"block_m": block_m, "block_n": block_n, "num_oc_splits": num_oc_splits,
                   "persistent_workgroups": 256, "output_layout": "routed"}
    return down


def with_torch_sum(down, workspace):
    """Compose any routed down with TOPK sum using shared, preallocated buffers."""
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
        return {
            "gemm": lambda: down(middle, *args[1:]),
            "reduce": lambda: torch.sum(middle, dim=1, out=args[0]),
        }

    def poison(*args):
        buffers(args).fill_(torch.nan)

    launch.benchmark_components = components
    launch.poison_workspace = poison
    launch.config = {**getattr(down, "config", {}), "output_layout": "routed",
                     "reduction": "torch", "includes_inverse": False, "includes_reduce": True}
    return launch


def print_markdown_table(headers, rows):
    """Print a compact Markdown table without an additional dependency."""
    widths = [len(header) for header in headers]
    for row in rows:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(str(value)))

    def format_row(row):
        return "| " + " | ".join(
            str(value).ljust(widths[column])
            for column, value in enumerate(row)
        ) + " |"

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def make_routing(tokens, topk, experts, seed):
    """Create random top-k IDs and normalized routing weights."""
    assert topk <= experts
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    # Random scores followed by topk gives independent random routing per token
    # while preserving the real router invariant that expert IDs are unique.
    scores = torch.rand(tokens, experts, generator=generator, dtype=torch.float32)
    topk_ids = torch.topk(scores, topk, dim=-1, sorted=False).indices.to(torch.int32)
    topk_weights = torch.rand(tokens, topk, generator=generator, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return topk_ids, topk_weights


def torch_reference_down(
    input_q,
    input_scales_k_major,
    weight_q,
    weight_scales,
    topk_ids,
    topk_weights,
):
    """Block-scale FP8 MoE down reference matching the kernel output layout."""
    tokens, topk, k = input_q.shape
    experts, n, weight_k = weight_q.shape
    assert weight_k == k

    rows = tokens * topk
    input_blocks = input_q.float().reshape(rows, k // 128, 128)
    # transpose_scale=True keeps the logical tensor shape but stores scales in
    # K-major order; recover the row-major [tokens*topk, K/128] view.
    input_scales = input_scales_k_major.view(k // 128, rows).t().float()
    output = torch.empty(rows, n, dtype=torch.bfloat16)
    expert_per_row = topk_ids.reshape(-1)
    routing_per_row = topk_weights.reshape(-1)

    for expert in range(experts):
        row_ids = torch.where(expert_per_row == expert)[0]
        if row_ids.numel() == 0:
            continue

        accum = torch.zeros(row_ids.numel(), n, dtype=torch.float32)
        for bk in range(k // 128):
            a = input_blocks[row_ids, bk, :]
            for bn in range(n // 128):
                w = weight_q[expert, bn * 128 : (bn + 1) * 128, bk * 128 : (bk + 1) * 128]
                partial = a @ w.float().t()
                scale = input_scales[row_ids, bk, None] * weight_scales[expert, bn, bk]
                accum[:, bn * 128 : (bn + 1) * 128] += partial * scale

        # Kernel multiplies in FP32 and then converts each result to BF16.
        output[row_ids] = (accum * routing_per_row[row_ids, None]).to(torch.bfloat16)

    return output.reshape(tokens, topk, n)


def run_test(
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    block_m,
    num_oc_splits,
    seed,
    candidates=None,
    profile=False,
    reduce_output=False,
    component_diagnostics=True,
):
    """Compare matching outputs; imported callers remain down-only by default.

    With reduce_output=True, elapsed_us is the directly measured complete call;
    down_elapsed_us is its separately timed GEMM component (counter reset included).
    CLI defaults to this mode, while historical tuning callers keep their scope.
    """
    torch.set_default_device("cuda")
    assert block_m == 256
    assert model_dim % 128 == 0
    assert inter_dim % 128 == 0
    assert model_dim % num_oc_splits == 0
    assert (model_dim // num_oc_splits) % 64 == 0
    assert (model_dim // num_oc_splits) // 64 >= 3
    candidates = list(DEFAULT_CANDIDATES) if candidates is None else list(candidates)
    assert candidates and all(name in CANDIDATES for name in candidates)
    assert not profile or len(candidates) == 1, "--profile requires one --candidate"

    torch.manual_seed(seed)
    # Match the real fused-MoE path: Aiter's HIP FP8 quantizers consume BF16.
    input_bf16 = torch.randn(tokens, topk, inter_dim, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(experts, model_dim, inter_dim, dtype=torch.bfloat16)

    input_q, input_scales = ACTIVATION_QUANT(
        input_bf16,
        quant_dtype=dtypes.fp8,
        transpose_scale=True,
    )

    # Re-layout each 128x128 weight tile as one row, then use Aiter's existing
    # per-token quantizer so each row receives exactly one FP8 block scale.
    weight_blocks = weight_bf16.view(
        experts, model_dim // 128, 128, inter_dim // 128, 128
    ).permute(0, 1, 3, 2, 4).contiguous()
    weight_q_blocks, weight_scales = aiter.pertoken_quant(
        weight_blocks.view(experts, -1, 128 * 128),
        quant_dtype=dtypes.fp8,
    )
    weight_q = weight_q_blocks.view(
        experts, model_dim // 128, inter_dim // 128, 128, 128
    ).permute(0, 1, 3, 2, 4).contiguous().view(experts, model_dim, inter_dim)
    weight_scales = weight_scales.view(
        experts, model_dim // 128, inter_dim // 128
    )
    weight_layouts = {(16, 16), *(CANDIDATE_WEIGHT_LAYOUTS.get(name, (16, 16)) for name in candidates)}
    shuffled_weights = {layout: shuffle_weight(weight_q, layout=layout) for layout in sorted(weight_layouts)}
    # Reuse this allocation even when a candidate needs a different prepack.
    weight_shuffled = shuffled_weights[(16, 16)].clone() if len(weight_layouts) > 1 else shuffled_weights[(16, 16)]
    topk_ids, topk_weights = make_routing(tokens, topk, experts, seed + 1)
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        block_m,
        None,
        None,
        0,
    )

    reference = torch_reference_down(
        input_q,
        input_scales,
        weight_q,
        weight_scales,
        topk_ids,
        topk_weights,
    )
    intermediate_bytes = reference.numel() * reference.element_size()
    if reduce_output:
        # TOPK is dim1 in [tokens,topk,N]. Validate the baseline's BF16 routes
        # before summing them; summing independently rounded Torch routes can
        # amplify tiny differences near cancellation. The tolerance is unchanged.
        baseline_routes = torch.empty_like(reference)
        baseline_counter = torch.zeros(1, dtype=torch.int32)
        if inter_dim in (256, 384, 512, 640):
            baseline_options = (dict(ATT_TUNED_BN128_CONFIG)
                                if inter_dim == 256 and model_dim % 512 == 0 else
                                {"block_n": 128, "num_oc_splits": 1})
            baseline_down = flydsl_moe_gemm_8wave_down_8stage(
                n=model_dim, k=inter_dim, topk=topk, num_experts=experts, **baseline_options,
            )
        else:
            baseline_down = make_pyhip_down(n=model_dim, k=inter_dim, topk=topk, num_experts=experts,
                                           num_oc_splits=num_oc_splits)
        baseline_down(baseline_routes, input_q, shuffled_weights[(16, 16)], input_scales, weight_scales,
                      sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, baseline_counter)
        torch.cuda.synchronize()
        baseline_mismatches = 0
        for begin in range(0, tokens, 256):
            actual = baseline_routes[begin:begin + 256].float()
            expected = reference[begin:begin + 256].float()
            assert torch.isfinite(actual).all(), "non-finite baseline down output"
            baseline_mismatches += ((actual - expected).abs() > 0.01 + 0.01 * expected.abs()).sum().item()
        assert baseline_mismatches == 0, f"baseline down: {baseline_mismatches} mismatches against Torch reference"
        reference = baseline_routes.sum(dim=1)
        del baseline_routes, baseline_counter
        print("Reduced reference: baseline down BF16 output + torch.sum(TOPK); baseline down validated against Torch at original tolerance.")

    num_cus = torch.cuda.get_device_properties().multi_processor_count
    flops1 = 2 * tokens * topk * model_dim * inter_dim
    valid_eblocks = num_valid_ids[0].item() // block_m
    # Only the valid prefix is initialized; unused expert-ID capacity must
    # not contribute to the ideal once-per-expert weight-read model.
    unique_experts = torch.unique(sorted_expert_ids[:valid_eblocks]).numel()
    flops2 = valid_eblocks * block_m * model_dim * inter_dim * 2
    weight_bytes_per_expert = model_dim * inter_dim * weight_q.element_size()
    non_weight_rw_bytes = input_q.numel() * input_q.element_size() + intermediate_bytes
    down_rw_bytes = valid_eblocks * weight_bytes_per_expert + non_weight_rw_bytes
    down_ideal_rw_bytes = unique_experts * weight_bytes_per_expert + non_weight_rw_bytes
    reduce_rw_bytes = intermediate_bytes + reference.numel() * reference.element_size() if reduce_output else 0
    rw_bytes = down_rw_bytes + reduce_rw_bytes
    ideal_rw_bytes = down_ideal_rw_bytes + reduce_rw_bytes
    traffic_stats = {
        "valid_expert_blocks": valid_eblocks,
        "unique_experts": unique_experts,
        "rw_bytes": rw_bytes,
        "ideal_rw_bytes": ideal_rw_bytes,
        "down_rw_bytes": down_rw_bytes,
        "down_ideal_rw_bytes": down_ideal_rw_bytes,
        "reduce_rw_bytes": reduce_rw_bytes,
    }

    ref_f32 = reference.float()
    threshold = 1.0e-2 + 1.0e-2 * ref_f32.abs()

    def performance(elapsed_us, nbytes, ideal_nbytes):
        return {
            "elapsed_us": elapsed_us,
            "us": f"{elapsed_us:.3f}" if elapsed_us is not None else "N/A",
            "effective_tflops": f"{flops1 / elapsed_us / 1e6:.3f}" if elapsed_us is not None else "N/A",
            "padded_tflops": f"{flops2 / elapsed_us / 1e6:.3f}" if elapsed_us is not None else "N/A",
            "tb_per_s": f"{nbytes / elapsed_us / 1e6:.3f}" if elapsed_us is not None else "N/A",
            "tb_per_s_ideal": f"{ideal_nbytes / elapsed_us / 1e6:.3f}" if elapsed_us is not None else "N/A",
        }

    def make_result(name, block_n, output, candidate_us, unavailable=None, *, down_us=None, reduce_us=None, inverse_bytes=0):
        stats = {**traffic_stats, "rw_bytes": rw_bytes + inverse_bytes,
                 "ideal_rw_bytes": ideal_rw_bytes + inverse_bytes, "inverse_rw_bytes": inverse_bytes,
                 "measurement_scope": "down+reduce+inverse_if_needed" if reduce_output else "down"}
        metrics = {**performance(candidate_us, stats["rw_bytes"], stats["ideal_rw_bytes"]),
                   **{"down_" + key: value for key, value in
                      performance(down_us, down_rw_bytes, down_ideal_rw_bytes).items()},
                   "reduce_tb_per_s": f"{reduce_rw_bytes / reduce_us / 1e6:.3f}"
                   if reduce_output and reduce_us is not None else "N/A"}
        if unavailable is not None:
            return {
                **stats, **metrics,
                "name": name,
                "block_n": block_n,
                "status": unavailable,
                "max_abs": "N/A",
                "mean_abs": "N/A",
                "diff": "N/A",
                "mismatches": "N/A",
                "failed": False,
            }

        output_f32 = output.float()
        abs_error = (output_f32 - ref_f32).abs()
        mismatch_count = (abs_error > threshold).sum().item()
        nonfinite = not torch.isfinite(output_f32).all().item()
        return {
            **stats, **metrics,
            "name": name,
            "block_n": block_n,
            "status": "FAIL" if nonfinite or mismatch_count else "PASS",
            "max_abs": f"{abs_error.max().item():.6g}",
            "mean_abs": f"{abs_error.mean().item():.6g}",
            "diff": f"{pyhip.calc_diff(ref_f32, output_f32):.6g}",
            "mismatches": f"{mismatch_count}/{abs_error.numel()}",
            "failed": nonfinite or mismatch_count > 0,
        }

    # All candidates write the exact same allocation, not merely equal-shaped
    # outputs with different physical placement/cache behavior.  Poison it
    # outside the timer before each candidate; validate before the next reuse.
    output = torch.empty_like(reference)
    workspace = DownReduceWorkspace() if reduce_output else None
    results, candidate_errors, skipped = [], {}, {}
    for candidate in candidates:
        name, block_n, factory = CANDIDATES[candidate]
        factory_options = dict(
            n=model_dim, k=inter_dim, topk=topk, num_experts=experts,
            block_m=block_m, block_n=block_n, num_oc_splits=num_oc_splits,
        )
        factory_options.update(CANDIDATE_OPTIONS.get(candidate, {}))
        candidate_splits = factory_options["num_oc_splits"]
        if candidate == "8stage_bn128" and inter_dim != 256:
            name = f"{4 * (inter_dim // 128)}-stage BN128"
        reason = None
        if candidate in ("4stage_bn128_tuned", "4stage_bn128_mfma32") and (inter_dim != 256 or model_dim % (4 * 128)):
            reason = "ATT-tuned BN128 requires K256 and N divisible512"
        elif candidate.startswith(("4stage_", "8stage_", "16stage_")):
            if inter_dim not in (256, 384, 512, 640):
                reason = "8-stage supports K=256/384/512/640"
            elif block_n in (64, 256) and inter_dim != 256:
                reason = "BN64/BN256 require K256; BN128 supports larger K"
            elif (model_dim // candidate_splits) % block_n:
                reason = f"N per split must be divisible by BN{block_n}"
        if reason:
            skipped[name] = reason
            results.append(make_result(name, block_n, None, None, "SKIP"))
            continue

        weight_layout = CANDIDATE_WEIGHT_LAYOUTS.get(candidate, (16, 16))
        if len(weight_layouts) > 1:
            weight_shuffled.copy_(shuffled_weights[weight_layout])
        output.fill_(torch.nan)
        counter = torch.zeros(1, dtype=torch.int32 if factory is not None else torch.uint32)
        candidate_us = None
        component_times = {}
        inverse_bytes = 0
        launch_config = {"num_oc_splits": candidate_splits, "persistent_workgroups": 256,
                 "weight_layout": weight_layout}
        try:
            if factory is None:
                kernel = make_pyhip_down(**factory_options)
            elif reduce_output and candidate == "4stage_bn128_tuned":
                # Select the measured whole-pipeline winner, not routed down
                # or the faster-but-numerically-reassociated MFMA32 experiment.
                kernel = compile_packed_down_reduce(workspace=workspace, **factory_options)
            else:
                if reduce_output and candidate == "4stage_bn128_mfma32":
                    # Folded routing failed the original TOPK-sum tolerance.
                    factory_options.update(fold_routing=False, defer_k1=False)
                kernel = factory(**factory_options)
            if reduce_output and not getattr(kernel, "config", {}).get("includes_reduce", False):
                kernel = with_torch_sum(kernel, workspace)
            launch_config.update(getattr(kernel, "config", {}))
            if launch_config.get("includes_inverse", False):
                # Logical inverse traffic: fill + scan valid padded IDs +
                # write one int32 index per route. Small other metadata omitted.
                inverse_bytes = (valid_eblocks * block_m + 2 * tokens * topk) * sorted_ids.element_size()
            if hasattr(kernel, "poison_workspace"):
                kernel.poison_workspace(
                    output, input_q, weight_shuffled, input_scales, weight_scales,
                    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, counter,
                )

            def launch():
                return kernel(
                    output, input_q, weight_shuffled, input_scales, weight_scales,
                    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, counter,
                )

            if profile:
                # ATT iteration21 sees a warmed kernel, with no timing/spin
                # kernels in between.  ATT durations are not benchmark times.
                for _ in range(23):
                    launch()
                    torch.cuda.synchronize()
            else:
                # Zero-argument wrappers prevent run_perftest from cloning
                # only the FlyDSL inputs: all candidates use the SAME A/B/
                # scales/routing addresses and include the counter reset.
                _, candidate_us = pyhip.run_perftest(
                    launch, num_warmup=2, num_iters=10, num_copies=1,
                    num_flops=flops2, num_verbose=1, num_bytes=rw_bytes + inverse_bytes,
                    num_name=candidate + ("_total" if reduce_output else ""),
                    num_spec_tag=f"M={tokens * topk},N={model_dim},K={inter_dim}",
                )
                if component_diagnostics and hasattr(kernel, "benchmark_components"):
                    components = kernel.benchmark_components(
                        output, input_q, weight_shuffled, input_scales, weight_scales,
                        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, counter,
                    )
                    for label, component in components.items():
                        _, elapsed = pyhip.run_perftest(
                            component, num_warmup=2, num_iters=10, num_copies=1,
                            num_verbose=0, num_name=f"{candidate}_{label}",
                        )
                        component_times[label] = elapsed
                    # Validate the complete public-output path, not only a
                    # diagnostic component's result. Timings are not summed.
                    launch()
            torch.cuda.synchronize()
        except Exception as e:
            candidate_errors[name] = f"{type(e).__name__}: {e}"
            results.append(make_result(name, block_n, None, None, "ERROR"))
        else:
            down_us = component_times.get("gemm") if reduce_output else component_times.get("gemm", candidate_us)
            result = make_result(name, block_n, output, candidate_us, down_us=down_us,
                                 reduce_us=component_times.get("reduce"), inverse_bytes=inverse_bytes)
            result["config"] = launch_config
            result["components_us"] = component_times
            results.append(result)

    by_name = {result["name"]: result for result in results}

    def speedup(result, baseline, metric="elapsed_us"):
        base, elapsed = by_name.get(baseline, {}).get(metric), result[metric]
        return f"{base / elapsed:.3f}x" if base is not None and elapsed is not None else "N/A"

    print(
        f"\nShape: tokens={tokens}, topk={topk}, experts={experts}, "
        f"N={model_dim}, K={inter_dim}"
    )
    print(
        f"Config: valid_expert_blocks={valid_eblocks}, "
        f"unique_experts={unique_experts}, "
        f"reference_oc_splits={num_oc_splits}, device_CUs={num_cus}; actual launch config per candidate below"
    )
    if profile:
        scope = "down+reduce pipeline" if reduce_output else "down"
        print(f"Profile mode: 23 {scope} launches, no performance timings.")
    else:
        print("Timing: identical input/output addresses, counter reset included, warmup=2, iterations=10.")
    print("New FlyDSL paths: AGPR disabled. Cache-policy integers are raw ROCDL aux values; output aux2 means NT on gfx950.")
    print(
        "Bandwidth models: B read per M-block vs once per unique expert (ideal cache reuse); "
        "A/output accounting is unchanged. Neither is measured HBM traffic."
    )
    print("Tolerance: rtol=1e-2, atol=1e-2\n")
    if reduce_output:
        print("Output [tokens,N]: down + TOPK sum(dim=1), sharing intermediate and final allocations across built-in candidates.")
        print("Down time includes counter reset but excludes inverse/reduce; Total measures the complete call, not a sum of components.")
        print("TF/s counts GEMM work only. Total bytes add reduce R/W and inverse fill/scan/write when needed; other metadata omitted.")
        print("Reduce TB/s = (valid BF16 intermediate reads + final output writes) / independently measured reduce time; metadata excluded.")
        if "4stage_bn128_tuned" in candidates:
            print("Tuned selects MFMA16 packed down + custom reduce (256 threads, 2048 columns, NT reads); inverse rebuilt every call.")
    headers = ["Metric", *(result["name"] for result in results)]
    rows = [
        ["Status", *(result["status"] for result in results)],
        ["OC splits", *(result.get("config", {}).get("num_oc_splits", "N/A") for result in results)],
        ["Persistent CTAs", *(result.get("config", {}).get("persistent_workgroups", "N/A") for result in results)],
        ["Weight layout", *(result.get("config", {}).get("weight_layout", "N/A") for result in results)],
        ["B prefetch distance", *(result.get("config", {}).get("prefetch_distance", "N/A") for result in results)],
        ["Output cache aux", *(result.get("config", {}).get("output_cache_policy", "N/A") for result in results)],
        ["B cache aux", *(result.get("config", {}).get("weight_cache_policy", "N/A") for result in results)],
        ["Coalesced output", *(result.get("config", {}).get("coalesced_output", "N/A") for result in results)],
        ["Valid expert blocks", *(result["valid_expert_blocks"] for result in results)],
        ["Unique experts", *(result["unique_experts"] for result in results)],
    ]
    if reduce_output:
        rows.extend([
            ["Down output layout", *(result.get("config", {}).get("output_layout", "N/A") for result in results)],
            ["Reduce implementation", *(result.get("config", {}).get("reduction", "N/A") for result in results)],
            ["Reduce threads", *(result.get("config", {}).get("reduce_threads", "N/A") for result in results)],
            ["Reduce block columns", *(result.get("config", {}).get("reduce_cols", "N/A") for result in results)],
            ["Reduce read cache aux", *(result.get("config", {}).get("read_policy", "N/A") for result in results)],
            ["Inverse rebuilt in timer", *(result.get("config", {}).get("includes_inverse", False) for result in results)],
            ["Down time (us)", *(result["down_us"] for result in results)],
            ["Down effective TF/s", *(result["down_effective_tflops"] for result in results)],
            ["Down padded TF/s", *(result["down_padded_tflops"] for result in results)],
            ["Down TB/s (B per M-block)", *(result["down_tb_per_s"] for result in results)],
            ["Down TB/s (B once/expert, ideal)", *(result["down_tb_per_s_ideal"] for result in results)],
            ["Down speedup vs PyHIP", *(speedup(result, "PyHIP", "down_elapsed_us") for result in results)],
        ])
    if any(result.get("components_us") for result in results):
        if not reduce_output:
            rows.append(["GEMM output layout", *(
                result.get("config", {}).get("output_layout", "routed") for result in results
            )])
        component_labels = sorted({label for result in results for label in result.get("components_us", {})})
        for label in component_labels:
            if reduce_output and label == "gemm":
                continue  # Already reported with down TF/s and bandwidth above.
            title = {"inverse": "Invert + fill time (us)", "reduce": "Reduce time (us)"}.get(label, f"Diagnostic {label} (us)")
            rows.append([title, *(
                f"{result['components_us'][label]:.3f}" if label in result.get("components_us", {}) else "N/A"
                for result in results
            )])
            if reduce_output and label == "reduce":
                rows.append(["Reduce TB/s", *(result["reduce_tb_per_s"] for result in results)])
        print("Component times are independently timed diagnostics, not summed. N/A means absent or not timed.")
        print("TF/s and TB/s use the stated work model, not hardware traffic counters.")
    prefix = "Total " if reduce_output else ""
    rows.extend([
        [prefix + "Time (us)", *(result["us"] for result in results)],
        [prefix + "Effective TF/s", *(result["effective_tflops"] for result in results)],
        [prefix + "Padded TF/s", *(result["padded_tflops"] for result in results)],
        [prefix + "TB/s (B per M-block)", *(result["tb_per_s"] for result in results)],
        [prefix + "TB/s (B once/expert, ideal)", *(result["tb_per_s_ideal"] for result in results)],
        [prefix + "Speedup vs PyHIP", *(speedup(result, "PyHIP") for result in results)],
        [prefix + "Speedup vs FlyDSL BN64", *(speedup(result, "FlyDSL BN64") for result in results)],
        ["Max abs error", *(result["max_abs"] for result in results)],
        ["Mean abs error", *(result["mean_abs"] for result in results)],
        ["calc_diff", *(result["diff"] for result in results)],
        ["Mismatches", *(result["mismatches"] for result in results)],
    ])
    print_markdown_table(headers, rows)

    if skipped:
        print("\nSkipped configurations:")
        for name, reason in skipped.items():
            print(f"- {name}: {reason}")
    if candidate_errors:
        print("\nCandidate errors:")
        for name, error in candidate_errors.items():
            summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
            print(f"- {name}: {summary}")

    failed_results = [result["name"] for result in results if result["failed"]]
    if failed_results:
        raise AssertionError(
            "correctness check failed for: " + ", ".join(failed_results)
        )
    if candidate_errors:
        raise RuntimeError("candidate execution failed: " + ", ".join(candidate_errors))
    print("\nPASS: all executed kernels match the Torch block-scale reference")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare down kernels and complete down+TOPK-reduce pipelines")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--model-dim", type=int, default=6144)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--num-oc-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), nargs="+", default=None)
    parser.add_argument("--mode", choices=("down-reduce", "down"), default=None,
                        help="default: down-reduce; --profile defaults to down for historical ATT compatibility")
    parser.add_argument("--profile", action="store_true", help="23 direct launches of one candidate for ATT; no timings")
    args = parser.parse_args()
    if args.profile and (args.candidate is None or len(args.candidate) != 1):
        parser.error("--profile requires exactly one --candidate")
    mode = args.mode or ("down" if args.profile else "down-reduce")
    run_test(
        args.tokens,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        args.block_m,
        args.num_oc_splits,
        args.seed,
        args.candidate,
        args.profile,
        reduce_output=mode == "down-reduce",
    )


if __name__ == "__main__":
    main()
