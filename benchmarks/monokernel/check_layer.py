"""The one-launch MonoKernel layers (``FusedGdnLayer.layer``, ``FusedAttentionLayer.layer``) with their in-launch
XGMI all-reduces against the split launches (GDN attention half / attention ``prep`` + ``core``, MLP half) with
an exact all-reduce (FP32 sum in rank order truncated to BF16, as AITER's one-stage kernel) and the separate
``hc_combine_norm``, bit for bit: layer output, next attention-branch norm, GDN conv/SSM states, KV cache and
the QSA index-key ring, ring coordinates and compressed keys.

One process per GPU (``--gpus``; the tensor-parallel size is their count, one GPU checks the launch over a
single-rank region). Real weights of layer 0 (GDN) and layer 3 (full attention) of every rank; synthetic
inputs, GDN states, paged KV cache and QSA state (the ring holds the pending index keys and the compressed
slots the golden compression of every completed group of a random key history). Every row count runs with
and without the attention-branch norm of the input given (``hn``, the served path).

    .venv/bin/python benchmarks/monokernel/check_layer.py --gpus 0,1 --rows 1 5 16 33 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.monokernel.check_reference import MODEL_DIR  # noqa: E402

PAGE = 64
RTT_STRIDE = 1 << 14
REQUESTS = 72
MAX_BLOCKS = RTT_STRIDE // 4
CAPACITY = 64


def exact_all_reduce(part, group):
    if group is None:
        return part.clone()
    parts = [torch.empty_like(part, device="cpu") for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, part.cpu(), group=group)
    total = parts[0].float()
    for p in parts[1:]:
        total = total + p.float()
    return (total.view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16).to(part.device)


def differing_bytes(report, name, got, expected):
    report[name] = int((got.contiguous().view(torch.uint8) != expected.contiguous().view(torch.uint8)).sum())
    return report[name] == 0


def cos_sin_table(positions, device):
    from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT as c

    inv_freq = 1.0 / c.rope_theta ** (torch.arange(0, c.rotary_dim, 2, dtype=torch.float32) / c.rotary_dim)
    freqs = torch.arange(positions, dtype=torch.float32)[:, None] * inv_freq[None]
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16).to(device)


def attention_norm(layer, h):
    from pyhip.ops.monokernel.qwen38.flydsl.hc import hc_combine_norm

    hn = torch.empty_like(h)
    hc_combine_norm(h, None, None, layer.served["hc_attn_norm"], None, hn)
    return hn


def check_gdn(layer, work, rows, device, group, with_hn):
    from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT as c
    from pyhip.ops.monokernel.qwen38.flydsl.hc import hc_combine_norm

    s = layer.shard
    h = torch.randn(rows, c.hc_width, generator=torch.Generator(device=device).manual_seed(rows),
                    device=device).to(torch.bfloat16)  # the same on every rank
    rank_gen = torch.Generator(device=device).manual_seed(1000 * rows + s.rank)
    pool = rows + 3
    slots = torch.randperm(pool, generator=torch.Generator().manual_seed(rows))[:rows].to(torch.int32).to(device)
    if rows > 2:
        slots[1] = -1
    conv0 = (0.1 * torch.randn(pool, s.conv_dim, 3, generator=rank_gen, device=device)).to(torch.bfloat16)
    ssm0 = 0.01 * torch.randn(pool, s.gdn_v_heads, c.gdn_head_dim, c.gdn_head_dim, generator=rank_gen, device=device)
    norm_next = layer.served["hc_attn_norm"]

    conv_r, ssm_r = conv0.clone(), ssm0.clone()
    part_a = torch.empty(rows, c.hidden, dtype=torch.bfloat16, device=device)
    gains = torch.empty(2, rows, c.hc_count, dtype=torch.float32, device=device)
    layer.attention(h, slots, conv_r, ssm_r, part_a, work, gain_out=gains[0])
    attn = exact_all_reduce(part_a, group)
    h1, part_m = torch.empty_like(h), torch.empty_like(attn)
    layer.mlp(h, attn, gains[0], h1, gains[1], part_m, work)
    moe = exact_all_reduce(part_m, group)
    out_r, normed_r = torch.empty_like(h), torch.empty_like(h)
    hc_combine_norm(h1, moe, gains[1], norm_next, out_r, normed_r)

    conv_n, ssm_n = conv0.clone(), ssm0.clone()
    out_n, normed_n = torch.empty_like(h), torch.empty_like(h)
    hn = attention_norm(layer, h) if with_hn else None
    layer.layer(h, slots, conv_n, ssm_n, norm_next, out_n, normed_n, work, hn=hn)
    torch.cuda.synchronize()
    report = {}
    ok = [differing_bytes(report, name, got, expected) for name, got, expected in (
        ("out", out_n, out_r), ("normed", normed_n, normed_r), ("conv", conv_n, conv_r), ("ssm", ssm_n, ssm_r))]
    return all(ok), report


def check_attention(layer, t, work, rows, device, group, table, k_cache, v_cache, with_hn):
    from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT as c
    from pyhip.ops.monokernel.qwen38.flydsl import attention as att
    from pyhip.ops.monokernel.qwen38.flydsl.hc import hc_combine_norm
    from pyhip.ops.monokernel.qwen38.reference import golden_index_compress

    cpu = torch.Generator().manual_seed(rows)
    h = torch.randn(rows, c.hc_width, generator=torch.Generator(device=device).manual_seed(rows),
                    device=device).to(torch.bfloat16)
    seq = torch.randint(2100, 12000, (rows,), generator=cpu)
    seq[0] = 1200  # fewer blocks than the top-k: every block selected
    seq[1::3] = (seq[1::3] // 4) * 4  # the new token completes a group
    req = torch.randperm(REQUESTS - 1, generator=cpu)[:rows] + 1
    slots_total = k_cache.shape[0]
    pages = torch.randperm(slots_total // PAGE - 1, generator=cpu) + 1
    rtt = torch.zeros(REQUESTS, RTT_STRIDE, dtype=torch.int32)
    used = 0
    for i in range(rows):
        count = -(-int(seq[i]) // PAGE)
        base = pages[used:used + count] * PAGE
        used += count
        rtt[req[i], :count * PAGE] = (base[:, None] + torch.arange(PAGE)).reshape(-1).to(torch.int32)
    positions = (seq - 1).repeat(3, 1).to(device)
    cache_loc = rtt[req, seq - 1].long().to(device)
    ring = torch.zeros(REQUESTS * att.RING, 1, att.ID, dtype=torch.bfloat16, device=device)
    ring_pos = torch.zeros(REQUESTS * att.RING, 3, dtype=torch.int64, device=device)
    compressed = torch.zeros(slots_total // att.RATIO + 1, 1, att.ID, dtype=torch.bfloat16, device=device)
    for i in range(rows):
        keys = torch.randn(int(seq[i]), att.ID, generator=cpu).to(torch.bfloat16).to(device)
        done = (int(seq[i]) - 1) // att.RATIO
        if done:
            group_pos = torch.arange(done, device=device) * att.RATIO
            ck = golden_index_compress(keys[:done * att.RATIO].view(done, att.RATIO, att.ID), t["index_k_norm"],
                                       group_pos, c)
            compressed[(rtt[req[i], :done * att.RATIO:att.RATIO] // att.RATIO).long(), 0] = ck.bfloat16()
        for p in range(done * att.RATIO, int(seq[i]) - 1):
            ring[req[i] * att.RING + p % att.RING, 0] = keys[p]
            ring_pos[req[i] * att.RING + p % att.RING] = p
    rtt_d, req_d, seq_d = rtt.to(device), req.long().to(device), seq.long().to(device)
    k_run, v_run = k_cache.clone(), v_cache.clone()
    k_new, v_new = k_cache.clone(), v_cache.clone()
    ring_new, ring_pos_new, compressed_new = ring.clone(), ring_pos.clone(), compressed.clone()
    norm_next = layer.served["hc_attn_norm"]

    x = torch.empty(rows, c.hidden, dtype=torch.bfloat16, device=device)
    gain_a = work.view("gain_a", (rows, c.hc_count), torch.float32)
    layer.prep(h, x, positions, cache_loc, k_run, v_run, table, work, gain_a, req_d, seq_d, rtt_d, ring, ring_pos,
               compressed)
    part_a = torch.empty(rows, c.hidden, dtype=torch.bfloat16, device=device)
    layer.core(rtt_d, req_d, seq_d, k_run, v_run, part_a, work)
    attn = exact_all_reduce(part_a, group)
    h1, part_m = torch.empty_like(h), torch.empty_like(attn)
    gain_m = torch.empty(rows, c.hc_count, dtype=torch.float32, device=device)
    layer.mlp(h, attn, gain_a.clone(), h1, gain_m, part_m, work)
    moe = exact_all_reduce(part_m, group)
    out_r, normed_r = torch.empty_like(h), torch.empty_like(h)
    hc_combine_norm(h1, moe, gain_m, norm_next, out_r, normed_r)

    out_n, normed_n = torch.empty_like(h), torch.empty_like(h)
    hn = attention_norm(layer, h) if with_hn else None
    layer.layer(h, positions, cache_loc, k_new, v_new, table, req_d, seq_d, rtt_d, ring_new, ring_pos_new,
                compressed_new, norm_next, out_n, normed_n, work, hn=hn)
    torch.cuda.synchronize()
    report = {}
    ok = [differing_bytes(report, name, got, expected) for name, got, expected in (
        ("out", out_n, out_r), ("normed", normed_n, normed_r), ("k_cache", k_new, k_run), ("v_cache", v_new, v_run),
        ("ring", ring_new, ring), ("ring_pos", ring_pos_new, ring_pos), ("compressed", compressed_new, compressed))]
    return all(ok), report


def worker(rank, gpus, port, rows_list, model_dir, queue):
    world = len(gpus)
    torch.cuda.set_device(gpus[rank])
    device = torch.device("cuda", gpus[rank])
    group = None
    if world > 1:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
        group = dist.group.WORLD
    from pyhip.ops.monokernel.qwen38.config import QWEN38_FLASH_NEXT, LayerKind, shard
    from pyhip.ops.monokernel.qwen38.flydsl import attention as att
    from pyhip.ops.monokernel.qwen38.flydsl.fused import FusedAttentionLayer, FusedGdnLayer, FusedWorkspace
    from pyhip.ops.monokernel.qwen38.flydsl.xgmi import region_bytes
    from pyhip.ops.monokernel.qwen38.peer import PeerMemory
    from pyhip.ops.monokernel.qwen38.weights import CheckpointLayer, pack_layer, sglang_format, shard_layer_weights

    s = shard(world, rank)
    results, ok = {}, True
    with torch.no_grad():
        comm = PeerMemory(region_bytes(world, CAPACITY), group, device)
        work = FusedWorkspace(CAPACITY, s, device, max_blocks=MAX_BLOCKS, comm=comm)
        layers = {}
        for layer_id, kind in ((0, LayerKind.LINEAR), (3, LayerKind.FULL)):
            t = shard_layer_weights(CheckpointLayer(model_dir, layer_id, device), s, kind)
            served = sglang_format(t, s, kind)
            packed = pack_layer(served, kind, QWEN38_FLASH_NEXT)
            layers[kind] = ((FusedGdnLayer if kind is LayerKind.LINEAR else FusedAttentionLayer)(s, served, packed),
                            t)
        table = cos_sin_table(RTT_STRIDE, device)
        slots = REQUESTS * 192 * PAGE  # up to 12288 positions per request
        gen = torch.Generator(device=device).manual_seed(7 + rank)
        k_cache = torch.randn(slots, 1, att.DA, generator=gen, device=device).to(torch.bfloat16)
        v_cache = torch.randn(slots, 1, att.DA, generator=gen, device=device).to(torch.bfloat16)
        for rows in rows_list:
            for with_hn in (False, True):
                tag = f"rows={rows} hn={int(with_hn)}"
                good, report = check_gdn(layers[LayerKind.LINEAR][0], work, rows, device, group, with_hn)
                results[f"gdn {tag}"] = report
                ok &= good
                layer, t = layers[LayerKind.FULL]
                good, report = check_attention(layer, t, work, rows, device, group, table, k_cache, v_cache, with_hn)
                results[f"attention {tag}"] = report
                ok &= good
    queue.put((rank, ok, results))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0,1", help="comma-separated visible GPUs, one per TP rank")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 5, 16, 33, 64])
    parser.add_argument("--port", type=int, default=29652)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert max(args.rows) <= CAPACITY
    gpus = [int(g) for g in args.gpus.split(",")]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, gpus, args.port, args.rows, args.model_dir, queue))
             for r in range(len(gpus))]
    for p in procs:
        p.start()
    collected = {}
    for _ in procs:
        rank, ok, results = queue.get(timeout=3600)
        collected[rank] = dict(ok=ok, results=results)
    for p in procs:
        p.join(timeout=120)
    passed = all(entry["ok"] for entry in collected.values())
    report = dict(gpus=gpus, rows=args.rows, passed=passed, ranks=dict(sorted(collected.items())))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    bad = {f"rank{rank} {case}": {k: v for k, v in counts.items() if v}
           for rank, entry in report["ranks"].items() for case, counts in entry["results"].items()
           if any(counts.values())}
    print(json.dumps(bad, indent=1) if bad else "every case bit-identical")
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
