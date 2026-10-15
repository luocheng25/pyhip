"""Check the MonoKernel stage kernels against SGLang decode captures (real weights, real inputs).

Same captures as check_reference.py. Kernels read the checkpoint converted to SGLang's served
format (``sglang_format``, verified bit-identical to the server) plus the MonoKernel's packed BF16
copies. Two modes, per decode step and layer:

* ``stage``: every kernel stage starts from its captured SGLang input (HC norm, HC mix, the GDN
  block from ``gdn.x`` with the captured states, the reduction of the captured partials, combine,
  routing, MoE from ``moe.x`` with SGLang's route), so each metric isolates one kernel.
* ``chain``: the staged layer (``flydsl.layer.StagedLayer``) runs every rank from the captured layer
  input, summing the rank partials in rank order like ``golden_allreduce``.

Bounds follow check_reference.THRESHOLDS for the same quantity (the kernels must stay as close to
SGLang as the goldens); MLP and layer outputs compare rows whose expert set matches SGLang's.

    .venv/bin/python benchmarks/monokernel/check_kernels.py --capture DIR/tp2/capture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.monokernel.check_reference import (  # noqa: E402
    MODEL_DIR,
    THRESHOLDS,
    load_captures,
    metrics,
    route_metrics,
    same_route,
    worst,
)
from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard  # noqa: E402
from pyhip.ops.monokernel.qwen38.reference import golden_allreduce, golden_gdn_core, golden_gdn_out  # noqa: E402
from pyhip.ops.monokernel.qwen38.weights import (  # noqa: E402
    CheckpointLayer,
    pack_layer,
    sglang_format,
    shard_layer_weights,
)

GOLDEN_GDN = ("gdn_conv", "gdn_a_log", "gdn_dt_bias", "gdn_norm", "gdn_out", "gdn_out_scale")

KERNEL_THRESHOLDS = {
    "kernel.hc_attn.norm": THRESHOLDS["hc_attn.norm"],
    "kernel.hc_attn.mix": dict(rel_l2=2e-3, ulp1=0.995),
    "kernel.gdn.qkvz": THRESHOLDS["gdn.qkvz"],
    "kernel.gdn.ba": THRESHOLDS["gdn.ba"],
    "kernel.gdn.conv_state": THRESHOLDS["gdn.conv_state"],
    "kernel.gdn.core": THRESHOLDS["gdn.core"],
    "kernel.gdn.ssm_state": THRESHOLDS["gdn.ssm_state"],
    "kernel.gdn.partial": THRESHOLDS["gdn.partial"],
    "kernel.hc_attn.combine": THRESHOLDS["hc_attn.combine"],
    "kernel.hc_mlp.norm": THRESHOLDS["hc_mlp.norm"],
    "kernel.hc_mlp.mix": dict(rel_l2=2e-3, ulp1=0.995),
    "kernel.moe.logits": THRESHOLDS["moe.logits"],
    "kernel.moe.route": THRESHOLDS["moe.route"],
    "kernel.moe.partial": THRESHOLDS["moe.partial"],
    "kernel.hc_mlp.combine": THRESHOLDS["hc_mlp.combine"],
    "chain.hc_attn_out": dict(rel_l2=2e-3, ulp1=0.995),
    "chain.attn": THRESHOLDS["chain.attn"],
    "chain.hc_attn_combined": THRESHOLDS["chain.hc_attn_combined"],
    "chain.hc_mlp_out": THRESHOLDS["chain.hc_mlp_out"],
    "chain.route": THRESHOLDS["chain.route"],
    "chain.mlp": THRESHOLDS["chain.mlp"],
    "chain.out": THRESHOLDS["chain.out"],
    "chain.conv_state": THRESHOLDS["chain.conv_state"],
    "chain.ssm_state": THRESHOLDS["chain.ssm_state"],
}


def evaluate(summary: dict) -> list[str]:
    failures = []
    for name, bounds in KERNEL_THRESHOLDS.items():
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


def load_layer(model_dir: Path, layer_id: int, rank_shard, device: torch.device) -> tuple[dict, dict, dict]:
    """Served and packed tensors of one rank's layer, plus the checkpoint-format GDN tensors the goldens use."""
    kind = rank_shard.config.layer_kind(layer_id)
    t = shard_layer_weights(CheckpointLayer(model_dir, layer_id, device), rank_shard, kind)
    served = sglang_format(t, rank_shard, kind)
    golden = {name: t[name] for name in GOLDEN_GDN}
    del t
    return served, pack_layer(served, kind, rank_shard.config), golden


def _states(record: dict, device: torch.device):
    rows = record["gdn.x"].shape[0]
    slots = torch.arange(rows, dtype=torch.int32, device=device)
    return slots, record["gdn.conv_before"].to(device).clone(), record["gdn.ssm_before"].to(device).clone()


def check_layer(records: list[dict], layers: list, goldens: list[dict], shards: list, device: torch.device) -> dict:
    """``records[rank]`` of one GDN layer; ``layers[rank]`` the StagedLayer of that rank and ``goldens[rank]``
    the checkpoint-format tensors the golden GDN core and gated norm need."""

    from pyhip.ops.monokernel.qwen38.flydsl.gdn import gdn_recurrent
    from pyhip.ops.monokernel.qwen38.flydsl.hc import hc_combine_norm, hc_mix
    from pyhip.ops.monokernel.qwen38.flydsl.layer import LayerWorkspace
    from pyhip.ops.monokernel.qwen38.flydsl.linear import linear_bf16, linear_fp8, quant_rows
    from pyhip.ops.monokernel.qwen38.flydsl.moe import moe_down, moe_up, topk_plan

    c = QWEN38_FLASH_NEXT
    first = records[0]
    rows = first["gdn.x"].shape[0]
    to = lambda name, record=first: record[name].to(device)  # noqa: E731
    out = {}
    works = [LayerWorkspace(rows, s, device) for s in shards]
    w0, p0 = layers[0].served, layers[0].packed

    # ---- stage mode
    h = to("hc_attn.in")
    normed = torch.empty_like(h)
    hc_combine_norm(h, None, None, w0["hc_attn_norm"], None, normed)
    out["kernel.hc_attn.norm"] = metrics(normed, first["hc_attn.normed"])
    hc_mix(to("hc_attn.normed"), p0["hc_attn_dn"], p0["hc_attn_up"], works[0].x, works[0].hc_attn)
    out["kernel.hc_attn.mix"] = metrics(works[0].x, first["hc_attn.out"])
    per_rank = defaultdict(list)
    for record, layer, work, golden in zip(records, layers, works, goldens):
        s, w = layer.shard, layer.served
        slots, conv, ssm = _states(record, device)
        work.x.copy_(record["gdn.x"].to(device))
        _gdn_from_x(layer, work, slots, conv, ssm)
        per_rank["kernel.gdn.qkvz"].append(metrics(work.qkvz, record["gdn.qkvz"]))
        per_rank["kernel.gdn.ba"].append(metrics(work.ba, record["gdn.ba"]))
        per_rank["kernel.gdn.conv_state"].append(metrics(conv, record["gdn.conv_after"]))
        # The recurrence from SGLang's own conv input, gates and z, so it is isolated from qkvz ulps.
        slots, conv, ssm = _states(record, device)
        conv_out, _ = golden_gdn_core(record["gdn.mixed_qkv"].to(device), record["gdn.ba"].to(device), golden, s,
                                      state_indices=torch.arange(rows), conv_state=conv.clone(), ssm_state=ssm.clone())
        qkvz = record["gdn.qkvz"].to(device)
        gdn_recurrent(conv_out.to(torch.bfloat16), qkvz, record["gdn.ba"].to(device), w["gdn_a_log"], w["gdn_dt_bias"],
                      w["gdn_norm"], ssm, slots, work.core, work.y, s.gdn_v_heads, s.gdn_k_heads)
        per_rank["kernel.gdn.ssm_state"].append(metrics(ssm, record["gdn.ssm_after"]))
        per_rank["kernel.gdn.core"].append(metrics(work.core, record["gdn.core"]))
        # The out projection from SGLang's core: golden gated norm, then the kernels' quant + W8A8.
        normed, _ = golden_gdn_out(record["gdn.core"].to(device).reshape(work.core.shape), qkvz[:, s.conv_dim:],
                                   golden, s)
        quant_rows(normed.reshape(rows, -1).to(torch.bfloat16), work.yq, work.ys)
        linear_fp8(work.yq, work.ys, w["gdn_out"], w["gdn_out_scale"], work.partial)
        per_rank["kernel.gdn.partial"].append(metrics(work.partial, record["gdn.partial"]))
    out.update({name: worst(values) for name, values in per_rank.items()})
    # combine from the captured reduced mixer output and the captured gains' source (the HC mix above)
    hc_mix(to("hc_attn.normed"), p0["hc_attn_dn"], p0["hc_attn_up"], works[0].x, works[0].hc_attn)
    hc_combine_norm(h, to("hc_attn.block"), works[0].hc_attn.gain, w0["hc_mlp_norm"], works[0].h1, works[0].hn1)
    out["kernel.hc_attn.combine"] = metrics(works[0].h1, first["hc_attn.combined"])
    out["kernel.hc_mlp.norm"] = metrics(works[0].hn1, first["hc_mlp.normed"])
    hc_mix(to("hc_mlp.normed"), p0["hc_mlp_dn"], p0["hc_mlp_up"], works[0].xm, works[0].hc_mlp)
    out["kernel.hc_mlp.mix"] = metrics(works[0].xm, first["hc_mlp.out"])
    moe = works[0].moe
    x_moe = to("moe.x")
    linear_bf16(x_moe, p0["router"], moe.logits)
    linear_bf16(x_moe, p0["shared_gate"], moe.shared)
    topk_plan(moe, rows)
    out["kernel.moe.logits"] = metrics(moe.logits, first["moe.logits"])
    out["kernel.moe.route"] = route_metrics(moe.ids, moe.weights, moe.logits, first["moe.ids"], first["moe.weights"],
                                            c.top_k)
    same = same_route(moe.ids, first["moe.ids"]).cpu()
    for record, layer, work in zip(records, layers, works):
        if work is not works[0]:
            linear_bf16(x_moe, layer.packed["router"], work.moe.logits)
            linear_bf16(x_moe, layer.packed["shared_gate"], work.moe.shared)
            topk_plan(work.moe, rows)
        moe_up(x_moe, layer.served["moe_w13"], layer.served["moe_w13_scale"], work.moe)
        moe_down(layer.served["moe_w2"], layer.served["moe_w2_scale"], work.moe, work.partial_m)
        per_rank["kernel.moe.partial"].append(metrics(work.partial_m[same.to(device)], record["moe.partial"][same]))
    out["kernel.moe.partial"] = worst(per_rank["kernel.moe.partial"])
    combined, next_normed = torch.empty_like(h), torch.empty_like(h)
    hc_mix(to("hc_mlp.normed"), p0["hc_mlp_dn"], p0["hc_mlp_up"], works[0].xm, works[0].hc_mlp)
    works[0].h1.copy_(to("hc_attn.combined"))
    layers[0].finish(to("moe.out"), w0["hc_attn_norm"], combined, next_normed, works[0])
    out["kernel.hc_mlp.combine"] = metrics(combined, first["hc_mlp.combined"])

    # ---- chain mode
    works = [LayerWorkspace(rows, s, device) for s in shards]
    hn = torch.empty_like(h)
    hc_combine_norm(h, None, None, w0["hc_attn_norm"], None, hn)
    states = [_states(record, device) for record in records]
    partials = [layer.attention_phase(hn, *state, work) for layer, state, work in zip(layers, states, works)]
    attn = golden_allreduce(partials).to(torch.bfloat16)
    partials = [layer.mlp_phase(h, attn, work) for layer, work in zip(layers, works)]
    mlp = golden_allreduce(partials).to(torch.bfloat16)
    final = torch.empty_like(h)
    layers[0].finish(mlp, w0["hc_attn_norm"], final, torch.empty_like(h), works[0])
    same = same_route(works[0].moe.ids, first["moe.ids"]).cpu()
    out["chain.hc_attn_out"] = metrics(works[0].x, first["hc_attn.out"])
    out["chain.attn"] = metrics(attn, first["hc_attn.block"])
    out["chain.hc_attn_combined"] = metrics(works[0].h1, first["hc_attn.combined"])
    out["chain.hc_mlp_out"] = metrics(works[0].xm, first["hc_mlp.out"])
    out["chain.route"] = route_metrics(works[0].moe.ids, works[0].moe.weights, works[0].moe.logits, first["moe.ids"],
                                       first["moe.weights"], c.top_k)
    out["chain.mlp"] = metrics(mlp[same.to(device)], first["moe.out"][same])
    out["chain.out"] = metrics(final[same.to(device)], first["out"][same])
    out["chain.conv_state"] = worst([metrics(state[1], record["gdn.conv_after"])
                                     for state, record in zip(states, records)])
    out["chain.ssm_state"] = worst([metrics(state[2], record["gdn.ssm_after"])
                                    for state, record in zip(states, records)])
    return out


def _gdn_from_x(layer, work, slots, conv, ssm):
    from pyhip.ops.monokernel.qwen38.flydsl.gdn import gdn_recurrent, qkvz_conv
    from pyhip.ops.monokernel.qwen38.flydsl.linear import linear_bf16, linear_fp8, quant_rows

    s, w, p = layer.shard, layer.served, layer.packed
    quant_rows(work.x, work.xq, work.xs)
    qkvz_conv(work.xq, work.xs, w["gdn_qkvz"], w["gdn_qkvz_scale"], w["gdn_conv"], slots, conv, work.qkvz, work.conv)
    linear_bf16(work.x, p["gdn_ba"], work.ba)
    gdn_recurrent(work.conv, work.qkvz, work.ba, w["gdn_a_log"], w["gdn_dt_bias"], w["gdn_norm"], ssm, slots,
                  work.core, work.y, s.gdn_v_heads, s.gdn_k_heads)
    quant_rows(work.y, work.yq, work.ys)
    linear_fp8(work.yq, work.ys, w["gdn_out"], w["gdn_out_scale"], work.partial)
    return work.partial.clone()


def run(capture: Path, model_dir: Path, device: torch.device) -> dict:
    from pyhip.ops.monokernel.qwen38.flydsl.layer import StagedLayer

    started = time.time()
    tp, steps = load_captures(capture)
    shards = [shard(tp, rank) for rank in range(tp)]
    layer_ids = [i for i in sorted(next(iter(steps.values()))[0]["layers"])
                 if QWEN38_FLASH_NEXT.layer_kind(i) is LayerKind.LINEAR]
    layers, goldens = {}, {}
    for layer_id in layer_ids:
        for rank in range(tp):
            served, packed, goldens[layer_id, rank] = load_layer(model_dir, layer_id, shards[rank], device)
            layers[layer_id, rank] = StagedLayer(LayerKind.LINEAR, shards[rank], served, packed)
    checks = {}
    for step, records in steps.items():
        checks[step] = {}
        for layer_id in layer_ids:
            layer_records = [record["layers"][layer_id] for record in records]
            checks[step][layer_id] = check_layer(layer_records, [layers[layer_id, r] for r in range(tp)],
                                                 [goldens[layer_id, r] for r in range(tp)], shards, device)
    summary = {}
    for name in {name for step in checks.values() for layer in step.values() for name in layer}:
        summary[name] = worst([layer[name] for step in checks.values() for layer in step.values() if name in layer])
    failures = evaluate(summary)
    return dict(capture=str(capture), model_dir=str(model_dir), tp=tp, layers=layer_ids, passed=not failures,
                failures=failures, summary=dict(sorted(summary.items())), checks=checks,
                seconds=round(time.time() - started, 1))


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
    print(f"tp={report['tp']} layers={report['layers']} seconds={report['seconds']}")
    for name, value in report["summary"].items():
        print(f"  {name:26s} " + " ".join(f"{key}={value[key]:.3g}" for key in value))
    print("PASS" if report["passed"] else "FAIL: " + "; ".join(report["failures"]))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
