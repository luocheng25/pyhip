"""Check the Qwen3.8 goldens against SGLang decode captures that use real weights.

``--capture`` holds ``tp{T}_rank{R}_step{S}.pt`` files written by the capture
plugin (see capture.py), one per decode step and TP rank: batch metadata plus,
per decoder layer, every operator's inputs and outputs, the GDN states before
and after the step, and each request's KV rows. Each SGLang stage is recomputed
from its own captured inputs (``stage``), the reductions from the captured
per-rank partials (``allreduce``), and every layer end to end from its captured
input with all ranks in one process (``chain``). The JSON report lists, per
check, relative L2 error, max absolute error, and the bit-exact fraction against
SGLang; the exit status is nonzero when a check exceeds ``THRESHOLDS``.
``served_format`` compares the checkpoint converted by ``sglang_format`` with
the plugin's raw-bit samples of SGLang's parameters (one GDN and one attention
layer, every rank); any difference fails.

    .venv/bin/python benchmarks/monokernel/check_reference.py \\
        --capture DIR/tp2/capture --output DIR/tp2/check_reference.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, Qwen38Shard, shard
from pyhip.ops.monokernel.qwen38.reference import (
    golden_allreduce,
    golden_attention_core,
    golden_attention_out,
    golden_attention_proj,
    golden_gdn_core,
    golden_gdn_out,
    golden_gdn_proj,
    golden_hc_combine,
    golden_hc_norm,
    golden_layer_tp,
    golden_moe,
    golden_route,
    resolve_hc_mix_ties,
)
from pyhip.ops.monokernel.qwen38.weights import (
    EXPERT_TENSORS,
    SGLANG_NAMES,
    CheckpointLayer,
    sglang_format,
    shard_experts,
    shard_layer_weights,
)

MODEL_DIR = Path("/models/Qwen3.8-Flash-Next-PTPC-FP8")
CAPTURE_NAME = re.compile(r"tp(\d+)_rank(\d+)_step(\d+)\.pt")
# Tensors SGLang computes redundantly on every rank. Divergence is reported, not failed: its HC
# mix accumulates the low-rank GEMV with atomics, so ranks can differ by an ulp and route apart.
REPLICATED = (
    "hc_attn.in",
    "hc_attn.normed",
    "hc_attn.out",
    "hc_attn.block",
    "hc_attn.combined",
    "hc_mlp.normed",
    "hc_mlp.out",
    "moe.logits",
    "moe.ids",
    "moe.weights",
    "moe.out",
    "hc_mlp.combined",
    "out",
)

# Bounds per check: about twice to three times the worst value of the 2026-10-13 captures (TP2 with the
# production FlyDSL decode MoE; TP2/TP4 non-MoE stages). Stage checks start from captured inputs and hold
# to one BF16 ulp, except where AITER epilogues truncate to BF16 (all-reduce, the tuned M=4 cktile GEMM),
# a one-ulp Triton sigmoid difference flips an FP8 code, or AITER's MoE rounds every expert output to BF16
# and sums them with BF16 atomics. Chained checks compound these through the FP8 requantizations of each
# layer; ``tie_gap`` bounds routes that differ to near-ties. HC mix stages flip low-rank activations that
# sit within 32 FP32 ulps of a BF16 tie (``tie_flips``), since one such rounding moves a whole row.
STAGE_ULP = dict(rel_l2=1e-3, ulp1=0.999)
PROJECTION = dict(rel_l2=1e-2, ulp1=0.98)
THRESHOLDS = {
    **{
        name: STAGE_ULP
        for name in (
            "hc_attn.norm",
            "hc_attn.combine",
            "hc_mlp.norm",
            "hc_mlp.combine",
            "gdn.qkvz",
            "gdn.ba",
            "gdn.conv_state",
            "gdn.core",
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.gate",
            "attn.core",
            "moe.logits",
        )
    },
    "hc_attn.mix": dict(rel_l2=2e-3, ulp1=0.999),
    "hc_mlp.mix": dict(rel_l2=2e-3, ulp1=0.999),
    "gdn.ssm_state": dict(rel_l2=1e-5, ulp1=0.999),
    "gdn.partial": PROJECTION,
    "attn.partial": PROJECTION,
    "attn.allreduce": dict(rel_l2=1e-2, ulp1=0.999),
    "moe.allreduce": dict(rel_l2=1e-2, ulp1=0.999),
    "moe.route": dict(tie_gap=1e-2, ulp1=0.999),
    "moe.partial": dict(rel_l2=1.5e-2, ulp1=0.6),
    "chain.hc_attn_in": dict(rel_l2=0.0),
    "chain.hc_attn_out": dict(rel_l2=2e-3, ulp1=0.999),
    "chain.attn": dict(rel_l2=3e-2),
    "chain.hc_attn_combined": dict(rel_l2=1.5e-2),
    "chain.hc_mlp_out": dict(rel_l2=1.5e-2),
    "chain.route": dict(tie_gap=0.1, mismatched=0.2),
    "chain.mlp": dict(rel_l2=4e-2),
    "chain.out": dict(rel_l2=2e-2),
    "chain.conv_state": dict(rel_l2=3e-3),
    "chain.ssm_state": dict(rel_l2=1e-3),
}


def evaluate(summary: dict) -> list[str]:
    """Threshold violations of a report summary; checks a capture cannot exercise are skipped."""

    failures = []
    for name, bounds in THRESHOLDS.items():
        value = summary.get(name)
        if value is None:
            continue
        for key, bound in bounds.items():
            if key == "mismatched":
                got = value["mismatched_rows"] / max(value["rows"], 1)
                ok = got <= bound
            elif key == "ulp1":
                got = value[key]
                ok = got >= bound
            else:
                got = value[key]
                ok = got <= bound
            if not ok:
                failures.append(f"{name}.{key}={got:.4g} (bound {bound:g})")
    return failures


class _ExpertBank(dict):
    """Expert id -> tensor; a missing id loads that expert into all four ``moe_*`` banks."""

    def __init__(self, load):
        super().__init__()
        self._load = load

    def __missing__(self, expert):
        self._load(expert)
        return dict.__getitem__(self, expert)


def load_rank_weights(model_dir: Path, layer_id: int, rank_shard: Qwen38Shard, device: torch.device) -> dict:
    """One rank's layer weights with experts loaded on first use."""

    get = CheckpointLayer(model_dir, layer_id, device)
    t = shard_layer_weights(get, rank_shard, rank_shard.config.layer_kind(layer_id), experts=())

    def load(expert: int) -> None:
        for name, bank in shard_experts(get, rank_shard, (expert,)).items():
            t[name].update(bank)

    t.update({name: _ExpertBank(load) for name in EXPERT_TENSORS})
    return t


def metrics(got: torch.Tensor, expected: torch.Tensor) -> dict:
    """Relative L2, max abs error, bit-exact fraction, and fraction within one BF16 ulp.

    ``ulp1`` absorbs round-to-nearest versus truncating BF16 epilogues (AITER's
    all-reduce and some tuned FP8 GEMMs truncate) and FP32 summation order.
    """

    got = got.float()
    expected = expected.to(got.device).float().reshape(got.shape)
    if got.numel() == 0:
        return dict(rel_l2=0.0, max_abs=0.0, exact=1.0, ulp1=1.0)
    delta = (got - expected).abs()
    magnitude = torch.maximum(got.abs(), expected.abs()).clamp_min(torch.finfo(torch.bfloat16).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
    return dict(
        rel_l2=float(delta.norm() / expected.norm().clamp_min(1e-30)),
        max_abs=float(delta.max()),
        exact=float((delta == 0).float().mean()),
        ulp1=float((delta <= ulp).float().mean()),
    )


def same_route(ids: torch.Tensor, ref_ids: torch.Tensor) -> torch.Tensor:
    """Rows whose selected expert sets agree."""

    ids, ref_ids = ids.long(), ref_ids.to(ids.device).long()
    return (ids.sort(-1).values == ref_ids.sort(-1).values).all(-1)


def route_metrics(
    ids: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_weights: torch.Tensor,
    top_k: int,
) -> dict:
    """Expert-set agreement, weights of agreeing rows (sorted by id), and the tie gap of the others.

    ``tie_gap`` is the largest relative probability gap, under ``logits``, between
    this route's weakest routed expert and a routed expert only the reference chose.
    """

    ids, ref_ids = ids.long(), ref_ids.to(ids.device).long()
    same = same_route(ids, ref_ids)
    order, ref_order = ids.argsort(-1), ref_ids.argsort(-1)
    sorted_weights = weights.float().gather(-1, order)[same]
    ref_sorted = ref_weights.to(weights.device).float().gather(-1, ref_order)[same]
    probs = torch.softmax(logits.float(), -1)
    tie_gap = 0.0
    for row in (~same).nonzero().flatten().tolist():
        weakest = probs[row, ids[row, :top_k]].min()
        others = sorted(set(ref_ids[row, :top_k].tolist()) - set(ids[row, :top_k].tolist()))
        tie_gap = max(tie_gap, float((weakest - probs[row, others].max()) / weakest))
    return dict(
        rows=int(same.numel()),
        mismatched_rows=int((~same).sum()),
        tie_gap=tie_gap,
        **metrics(sorted_weights, ref_sorted),
    )


def worst(items: list[dict]) -> dict:
    """Combine per-rank or per-step results into their worst case; row counts add up."""

    merged = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if key in ("rows", "mismatched_rows"):
            merged[key] = sum(values)
        else:
            merged[key] = min(values) if key in ("exact", "ulp1") else max(values)
    return merged


def _to(record: dict, name: str, device: torch.device) -> torch.Tensor:
    return record[name].to(device)


def _mix_metrics(hn: torch.Tensor, t: dict, prefix: str, target: torch.Tensor) -> dict:
    mixed, flips = resolve_hc_mix_ties(hn, t[f"{prefix}_down"], t[f"{prefix}_up"], QWEN38_FLASH_NEXT, target)
    return dict(
        metrics(mixed, target),
        tie_flips=len(flips),
        tie_ulps=max((distance for _, _, distance in flips), default=0.0),
    )


def check_hc(record: dict, t: dict, device: torch.device) -> dict:
    c = QWEN38_FLASH_NEXT
    out = {}
    for name in ("hc_attn", "hc_mlp"):
        h, hn = _to(record, f"{name}.in", device), _to(record, f"{name}.normed", device)
        out[f"{name}.norm"] = metrics(golden_hc_norm(h, t[f"{name}_norm"], c), hn)
        out[f"{name}.mix"] = _mix_metrics(hn, t, name, record[f"{name}.out"])
        combined = golden_hc_combine(h, hn, _to(record, f"{name}.block", device), t[f"{name}_inject"], c)
        out[f"{name}.combine"] = metrics(combined, record[f"{name}.combined"])
    return out


def check_gdn(record: dict, t: dict, rank_shard: Qwen38Shard, device: torch.device) -> dict:
    qkvz, ba = golden_gdn_proj(_to(record, "gdn.x", device), t)
    out = {"gdn.qkvz": metrics(qkvz, record["gdn.qkvz"]), "gdn.ba": metrics(ba, record["gdn.ba"])}
    rows = qkvz.shape[0]
    conv = _to(record, "gdn.conv_before", device).clone()
    ssm = _to(record, "gdn.ssm_before", device).clone()
    _, core = golden_gdn_core(
        _to(record, "gdn.mixed_qkv", device),
        _to(record, "gdn.ba", device),
        t,
        rank_shard,
        state_indices=torch.arange(rows),
        conv_state=conv,
        ssm_state=ssm,
    )
    out["gdn.conv_state"] = metrics(conv, record["gdn.conv_after"])
    out["gdn.ssm_state"] = metrics(ssm, record["gdn.ssm_after"])
    out["gdn.core"] = metrics(core, record["gdn.core"])
    captured_core = _to(record, "gdn.core", device).reshape(core.shape)
    z = _to(record, "gdn.qkvz", device)[:, rank_shard.conv_dim :]
    _, partial = golden_gdn_out(captured_core, z, t, rank_shard)
    out["gdn.partial"] = metrics(partial, record["gdn.partial"])
    return out


def _kv_inputs(record: dict, meta: dict, device: torch.device) -> dict:
    return dict(
        positions=(meta["seq_lens"] - 1).to(device),
        k_cache=[rows.to(device).clone() for rows in record["attn.k_cache"]],
        v_cache=[rows.to(device).clone() for rows in record["attn.v_cache"]],
        selected=_to(record, "attn.topk", device),
    )


def check_attention(record: dict, meta: dict, t: dict, rank_shard: Qwen38Shard, device: torch.device) -> dict:
    d = rank_shard.config.head_dim
    rope_positions = _to(record, "attn.positions", device)[0]
    proj = golden_attention_proj(_to(record, "attn.x", device), t, rank_shard, rope_positions)
    out = {f"attn.{name}": metrics(proj[name], record[f"attn.{name}"]) for name in ("q", "k", "v", "gate")}
    rows = proj["q"].shape[0]
    q, k, v, gate = (_to(record, f"attn.{name}", device).reshape(rows, -1, d) for name in ("q", "k", "v", "gate"))
    core = golden_attention_core(q, k, v, rank_shard, **_kv_inputs(record, meta, device))
    out["attn.core"] = metrics(core, record["attn.core"])
    captured_core = _to(record, "attn.core", device).reshape(core.shape)
    _, partial = golden_attention_out(captured_core, gate, t, rank_shard)
    out["attn.partial"] = metrics(partial, record["attn.partial"])
    return out


def check_moe(record: dict, t: dict, rank_shard: Qwen38Shard, device: torch.device, route: bool) -> dict:
    x = _to(record, "moe.x", device)
    out = {}
    if route:
        logits, ids, weights = golden_route(x, t, rank_shard)
        out["moe.logits"] = metrics(logits, record["moe.logits"])
        out["moe.route"] = route_metrics(
            ids, weights, logits, record["moe.ids"], record["moe.weights"], rank_shard.config.top_k
        )
    moe = golden_moe(x, t, rank_shard, ids=_to(record, "moe.ids", device), weights=_to(record, "moe.weights", device))
    out["moe.partial"] = metrics(moe["partial"], record["moe.partial"])
    return out


def check_chain(
    kind: LayerKind,
    records: list[dict],
    meta: dict,
    weights: list[dict],
    shards: list[Qwen38Shard],
    device: torch.device,
) -> dict:
    """Recompute the whole layer from its captured input; mixer states start from the capture.

    MLP and layer outputs compare only rows whose expert set matches SGLang's;
    ``chain.route`` counts the others and their tie gap.
    """

    if kind is LayerKind.LINEAR:
        rows = records[0]["gdn.x"].shape[0]
        inputs = [
            dict(
                state_indices=torch.arange(rows),
                conv_state=_to(record, "gdn.conv_before", device).clone(),
                ssm_state=_to(record, "gdn.ssm_before", device).clone(),
            )
            for record in records
        ]
    else:
        inputs = [_kv_inputs(record, meta, device) for record in records]
    result = golden_layer_tp(kind, _to(records[0], "hidden_in", device), weights, shards, inputs)
    first = records[0]
    same = same_route(result["ids"], first["moe.ids"]).cpu()
    out = {
        "chain.hc_attn_in": metrics(result["hc_attn_in"], first["hc_attn.in"]),
        "chain.hc_attn_out": _mix_metrics(result["hc_attn_normed"], weights[0], "hc_attn", first["hc_attn.out"]),
        "chain.attn": metrics(result["attn"], first["hc_attn.block"]),
        "chain.hc_attn_combined": metrics(result["hc_attn_combined"], first["hc_attn.combined"]),
        "chain.hc_mlp_out": metrics(result["hc_mlp_out"], first["hc_mlp.out"]),
        "chain.route": route_metrics(
            result["ids"],
            result["weights"],
            result["logits"],
            first["moe.ids"],
            first["moe.weights"],
            shards[0].config.top_k,
        ),
        "chain.mlp": metrics(result["mlp"][same.to(device)], first["moe.out"][same]),
        "chain.out": metrics(result["out"][same.to(device)], first["out"][same]),
    }
    if kind is LayerKind.LINEAR:
        out["chain.conv_state"] = worst(
            [metrics(i["conv_state"], record["gdn.conv_after"]) for i, record in zip(inputs, records)]
        )
        out["chain.ssm_state"] = worst(
            [metrics(i["ssm_state"], record["gdn.ssm_after"]) for i, record in zip(inputs, records)]
        )
    return out


def check_step(
    records: list[dict], weights: dict, shards: list[Qwen38Shard], device: torch.device
) -> tuple[dict, list[str]]:
    """Checks of one decode step (``records[rank]``), and replicated tensors that differ across ranks."""

    meta = records[0]["meta"]
    results, divergent = {}, []
    for layer_id in sorted(records[0]["layers"]):
        layer = [record["layers"][layer_id] for record in records]
        kind = LayerKind(layer[0]["kind"])
        rank_weights = [weights[layer_id, rank] for rank in range(len(records))]
        out = check_hc(layer[0], rank_weights[0], device)
        per_rank = defaultdict(list)
        for rank, (record, t, rank_shard) in enumerate(zip(layer, rank_weights, shards)):
            if kind is LayerKind.LINEAR:
                mixer = check_gdn(record, t, rank_shard, device)
            else:
                mixer = check_attention(record, meta, t, rank_shard, device)
            mixer.update(check_moe(record, t, rank_shard, device, route=rank == 0))
            for name, value in mixer.items():
                per_rank[name].append(value)
        out.update({name: worst(values) for name, values in per_rank.items()})
        partial = "gdn.partial" if kind is LayerKind.LINEAR else "attn.partial"
        out["attn.allreduce"] = metrics(golden_allreduce([r[partial] for r in layer]), layer[0]["hc_attn.block"])
        out["moe.allreduce"] = metrics(golden_allreduce([r["moe.partial"] for r in layer]), layer[0]["moe.out"])
        out.update(check_chain(kind, layer, meta, rank_weights, shards, device))
        results[layer_id] = out
        divergent += [
            f"layer{layer_id}.{name}"
            for name in REPLICATED
            if any(not torch.equal(record[name], layer[0][name]) for record in layer[1:])
        ]
    return results, divergent


def load_captures(folder: Path) -> tuple[int, dict[int, list[dict]]]:
    """``(tp, {step: [record per rank]})`` from one capture directory."""

    found = {}
    for path in folder.glob("*.pt"):
        match = CAPTURE_NAME.fullmatch(path.name)
        if match:
            tp, rank, step = map(int, match.groups())
            found[tp, rank, step] = path
    tps = {key[0] for key in found}
    if len(tps) != 1:
        raise ValueError(f"{folder} must hold captures of exactly one TP size, found {sorted(tps)}")
    tp = tps.pop()
    steps = sorted({key[2] for key in found})
    missing = [(rank, step) for step in steps for rank in range(tp) if (tp, rank, step) not in found]
    if missing:
        raise ValueError(f"{folder} lacks (rank, step) captures {missing}")
    return tp, {
        step: [torch.load(found[tp, rank, step], map_location="cpu", weights_only=True) for rank in range(tp)]
        for step in steps
    }


def _sample_bits(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    picked = value.reshape(-1)[index.to(value.device)]
    if picked.element_size() == 1:
        picked = picked.view(torch.uint8)
    elif picked.dtype in (torch.bfloat16, torch.float16):
        picked = picked.view(torch.int16)
    return picked.cpu()


def check_format(capture: Path, tp: int, model_dir: Path, device: torch.device, layer_ids: list[int]) -> dict:
    """Compare ``sglang_format`` of the checkpoint with SGLang's served parameters, sampled by the plugin.

    Per rank and layer: tensors whose shape, dtype, or sampled bits differ, and SGLang tensors
    the served format does not cover.
    """

    report = {}
    for rank in range(tp):
        path = capture / f"tp{tp}_rank{rank}_param_samples.pt"
        if not path.exists():
            return {}
        samples = torch.load(path, map_location="cpu", weights_only=True)
        rank_shard = shard(tp, rank)
        for layer_id in layer_ids:
            kind = rank_shard.config.layer_kind(layer_id)
            t = shard_layer_weights(CheckpointLayer(model_dir, layer_id, device), rank_shard, kind)
            served = sglang_format(t, rank_shard, kind)
            del t
            live = samples[layer_id]
            names = SGLANG_NAMES[kind]
            differs = {}
            for name, value in served.items():
                sample = live[names[name]]
                if tuple(value.shape) != tuple(sample["shape"]) or str(value.dtype) != sample["dtype"]:
                    differs[name] = f"{tuple(value.shape)} {value.dtype} vs {tuple(sample['shape'])} {sample['dtype']}"
                    continue
                mismatched = int((_sample_bits(value, sample["index"]) != sample["bits"]).sum())
                if mismatched:
                    differs[name] = f"{mismatched}/{sample['index'].numel()} sampled values differ"
            report[f"rank{rank}.layer{layer_id}"] = dict(
                checked=len(served),
                differs=differs,
                not_served=sorted(set(live) - {names[name] for name in served}),
            )
            del served
    return report


def run(capture: Path, model_dir: Path, device: torch.device) -> dict:
    started = time.time()
    tp, steps = load_captures(capture)
    shards = [shard(tp, rank) for rank in range(tp)]
    layer_ids = sorted(next(iter(steps.values()))[0]["layers"])
    weights = {
        (layer_id, rank): load_rank_weights(model_dir, layer_id, shards[rank], device)
        for layer_id in layer_ids
        for rank in range(tp)
    }
    checks, divergent, batch = {}, {}, {}
    for step, records in steps.items():
        checks[step], divergent[step] = check_step(records, weights, shards, device)
        meta = records[0]["meta"]
        batch[step] = dict(seq_lens=meta["seq_lens"].tolist(), forward_mode=meta["forward_mode"])
    summary = {}
    for name in {name for step in checks.values() for layer in step.values() for name in layer}:
        summary[name] = worst([layer[name] for step in checks.values() for layer in step.values() if name in layer])
    failures = evaluate(summary)
    # One GDN and one attention layer cover every served tensor kind.
    kinds = {QWEN38_FLASH_NEXT.layer_kind(layer_id): layer_id for layer_id in reversed(layer_ids)}
    served_format = check_format(capture, tp, model_dir, device, sorted(kinds.values()))
    failures += [f"format.{key}.{name}: {why}" for key, item in served_format.items() for name, why in item["differs"].items()]
    return dict(
        capture=str(capture),
        model_dir=str(model_dir),
        tp=tp,
        layers={layer_id: QWEN38_FLASH_NEXT.layer_kind(layer_id).value for layer_id in layer_ids},
        batch=batch,
        passed=not failures,
        failures=failures,
        summary=dict(sorted(summary.items())),
        checks=checks,
        divergent_replicated=divergent,
        served_format=served_format,
        seconds=round(time.time() - started, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with torch.no_grad():
        report = run(args.capture, args.model_dir, torch.device(args.device))
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(f"tp={report['tp']} steps={len(report['batch'])} seconds={report['seconds']}")
    for name, value in report["summary"].items():
        print(f"  {name:24s} " + " ".join(f"{key}={value[key]:.3g}" for key in value))
    for step, items in report["divergent_replicated"].items():
        if items:
            print(f"  step {step}: replicated tensors differ across ranks: {items}")
    print("PASS" if report["passed"] else "FAIL: " + "; ".join(report["failures"]))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
