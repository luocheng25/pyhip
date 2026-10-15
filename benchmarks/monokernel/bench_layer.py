"""Per-rank GPU time of the staged MonoKernel GDN decode layer (no all-reduce), real weights.

For each TP size and batch S, rotates over ``--layers`` distinct GDN layers of one rank (their
weights exceed the 256 MiB MALL, so every layer streams its weights from HBM) and over 10 input
buffer sets. One CUDA graph holds 10 layer calls (attention phase, MLP phase with the rank partial
standing in for the reduced output, finish); after 2 warmup replays, 128 timed replays give the
median time per layer. ``--profile`` adds a per-stage breakdown from the torch profiler.

    .venv/bin/python benchmarks/monokernel/bench_layer.py --tp 2 4 --rows 1 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.monokernel.check_reference import MODEL_DIR  # noqa: E402
from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard  # noqa: E402
from pyhip.ops.monokernel.qwen38.flydsl.hc import hc_combine_norm  # noqa: E402
from pyhip.ops.monokernel.qwen38.flydsl.layer import LayerWorkspace, StagedLayer  # noqa: E402
from pyhip.ops.monokernel.qwen38.weights import CheckpointLayer, pack_layer, sglang_format, shard_layer_weights  # noqa: E402

BUFFERS, WARMUP, SAMPLES = 10, 2, 128


def load(model_dir, layer_ids, rank_shard, device):
    layers = []
    for layer_id in layer_ids:
        t = shard_layer_weights(CheckpointLayer(model_dir, layer_id, device), rank_shard, LayerKind.LINEAR)
        served = sglang_format(t, rank_shard, LayerKind.LINEAR)
        del t
        layers.append(StagedLayer(LayerKind.LINEAR, rank_shard, served, pack_layer(served, LayerKind.LINEAR,
                                                                                   rank_shard.config)))
    return layers


def bench(layers, rows, device):
    c = QWEN38_FLASH_NEXT
    s = layers[0].shard
    gen = torch.Generator(device=device).manual_seed(rows)
    work = LayerWorkspace(rows, s, device)
    hs = [torch.randn(rows, c.hc_width, generator=gen, device=device).to(torch.bfloat16) for _ in range(BUFFERS)]
    hns = [torch.empty_like(h) for h in hs]
    for h, hn in zip(hs, hns):
        hc_combine_norm(h, None, None, layers[0].served["hc_attn_norm"], None, hn)
    slots = torch.arange(rows, dtype=torch.int32, device=device)
    states = [((0.1 * torch.randn(rows, s.conv_dim, 3, generator=gen, device=device)).to(torch.bfloat16),
               0.01 * torch.randn(rows, s.gdn_v_heads, c.gdn_head_dim, c.gdn_head_dim, generator=gen, device=device))
              for _ in layers]
    out, out_normed = torch.empty_like(hs[0]), torch.empty_like(hs[0])

    def call(i):
        layer = layers[i % len(layers)]
        conv, ssm = states[i % len(layers)]
        partial = layer.attention_phase(hns[i % BUFFERS], slots, conv, ssm, work)
        partial = layer.mlp_phase(hs[i % BUFFERS], partial, work)
        layer.finish(partial, layer.served["hc_attn_norm"], out, out_normed, work)

    for i in range(BUFFERS):  # compile every kernel outside the graph
        call(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(BUFFERS):
            call(i)
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(SAMPLES):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / BUFFERS)
    return statistics.median(times), call


def profile(call, device):
    from torch.profiler import ProfilerActivity, profile as torch_profile

    for i in range(BUFFERS):
        call(i)
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(BUFFERS):
            call(i)
        torch.cuda.synchronize()
    totals = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = event.name.split("_0")[0]
            totals[name] = totals.get(name, 0.0) + event.device_time / BUFFERS
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 5, 6, 8])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    report = []
    with torch.no_grad():
        for tp in args.tp:
            layers = load(args.model_dir, args.layers, shard(tp, 0), device)
            for rows in args.rows:
                us, call = bench(layers, rows, device)
                item = dict(tp=tp, rows=rows, layer_us=round(us, 2))
                if args.profile:
                    item["stages_us"] = {k: round(v, 2) for k, v in profile(call, device).items()}
                report.append(item)
                print(json.dumps(item), flush=True)
            del layers
            torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
