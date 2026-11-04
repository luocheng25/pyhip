"""Whole TP8 MoE + reduce-scatter on 8x MI300X (sequence-parallel layout, MiMo-V2.6-Flash: H 4096, 256 experts,
top-8, I = 256 per rank, FP8 128x128 block scales, pre-quantized FP8 input with aiter's transposed scales).

torchrun --nproc_per_node 8 moe_rs.py [tokens...]   (env ITERS=20, SKEW=0 as in sum_rs.py)

  aiter+rs   : aiter fused_moe (1-stage) -> [T, H] -> RCCL reduce_scatter (current sglang SP path)
  pyhip+rs   : PyHIP direct MoE with its top-k sum -> RCCL reduce_scatter
  pyhip+fused: PyHIP sorting / gate_up / down_direct -> Y route rows -> sum_rs (top-k sum + reduce-scatter)
Same gathered tokens and routing on every rank, each rank its own expert I-slice. Prints the slowest rank's
median per call and the rel. L2 difference of the shards against aiter+rs.
"""

import os
import sys

import torch
import torch.distributed as dist

import pyhip

HERE = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(HERE, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, HERE)
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as K  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M  # noqa: E402
from sum_rs import NPES, FusedSumRS, shard_sizes, timed  # noqa: E402

FP8 = torch.float8_e4m3fnuz
H, E, TOPK, I = 4096, 256, 8, 256


def main():
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
    rank = dist.get_rank()
    tokens = [int(t) for t in sys.argv[1:]] or [16384, 32768]
    skew, iters = float(os.environ.get("SKEW", "0")), int(os.environ.get("ITERS", "20"))
    rs_fused = FusedSumRS(rank, max(max(shard_sizes(t, skew)) for t in tokens))
    stream = torch.cuda.current_stream()
    g = torch.Generator(device="cuda").manual_seed(100 + rank)  # this rank's I-slice
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * 0.5).to(FP8)
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) * 0.5).to(FP8)
    s13 = torch.rand(E, 2 * I // 128, H // 128, device="cuda", generator=g) * 0.02 + 0.005
    s2 = torch.rand(E, H // 128, I // 128, device="cuda", generator=g) * 0.02 + 0.005
    w13, w2 = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    w13.is_shuffled = w2.is_shuffled = True
    u8 = torch.uint8
    for T in tokens:
        sizes = shard_sizes(T, skew)
        T = sum(sizes)
        off = [sum(sizes[:i]) for i in range(NPES + 1)]
        g = torch.Generator(device="cuda").manual_seed(7)  # gathered batch: same on every rank
        x = torch.randn(T, H, device="cuda", generator=g).to(torch.bfloat16)
        wts, ids = torch.topk(torch.sigmoid(torch.randn(T, E, device="cuda", generator=g)), TOPK, -1)
        wts, ids = (wts / wts.sum(-1, keepdim=True)).float(), ids.to(torch.int32)
        x8, xs = M.quantize_input(x)  # xs: [H / 128, T] view of the transposed scale bytes
        a1 = torch.as_strided(xs, (T, H // 128), (1, T))
        out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
        equal = len(set(sizes)) == 1
        shard = torch.empty(max(sizes), H, dtype=torch.bfloat16, device="cuda")
        padded = torch.zeros(NPES * max(sizes), H, dtype=torch.bfloat16, device="cuda")
        fused_out = torch.empty(sizes[rank], H, dtype=torch.bfloat16, device="cuda")

        def rccl_rs(full):
            if equal:
                dist.reduce_scatter_tensor(shard, full)
            else:
                for s in range(NPES):
                    padded[s * max(sizes):s * max(sizes) + sizes[s]].copy_(full[off[s]:off[s + 1]])
                dist.reduce_scatter_tensor(shard, padded)
            return shard[:sizes[rank]]

        def aiter_rs():
            full = fused_moe(x8, w13, w2, wts, ids, quant_type=QuantType.per_128x128,
                             activation=ActivationType.Silu, w1_scale=s13, w2_scale=s2, a1_scale=a1,
                             a1_scale_is_transposed=True, dtype=torch.bfloat16)
            return rccl_rs(full)

        def pyhip_rs():
            M.fused_moe_fp8_blockscale_942(x8, w13, w2, wts, ids, s13, s2, a1_scale=xs, out=out)
            return rccl_rs(out)

        def pyhip_fused():
            sid, sw, seid, nv = M.moe_sorting(ids, wts, E, H, None, M.COMPACT_BLOCK)
            rows = sid.numel()
            a2, a2s, y = M._workspace(x8.device, rows, I, H)
            inv = torch.full((T, TOPK), -1, dtype=torch.int32, device=x8.device)
            tiles, count, cap = M._tiles(seid, nv, E, M.COMPACT_BLOCK, stream)
            M._run(("gate_up", x8.device, I, H), K.launch_gate_up, x8.view(u8).view(-1), w13.view(u8).view(-1),
                   xs.reshape(-1), s13.reshape(-1), sid, tiles, count, sw, a2, a2s, T, rows, cap * (I // 128), I, H,
                   stream)
            M._run(("inverse", x8.device, TOPK), K.launch_inverse, sid, nv, inv.view(-1), T, rows,
                   (rows + K.SUM_THREADS - 1) // K.SUM_THREADS, TOPK, stream)
            per = H // K.DOWN_COLS
            M._run(("down_direct", x8.device, H, I, per), K.launch_down_direct, a2, w2.view(u8).view(-1), a2s,
                   s2.reshape(-1), tiles, count, sid, y, rows, T, 0, cap * per, H, I, per, stream)
            return rs_fused(y, inv, fused_out, sizes)

        ref = aiter_rs().clone()
        diffs = {}
        for name, fn in (("pyhip+rs", pyhip_rs), ("pyhip+fused", pyhip_fused)):
            got = fn().float()
            diffs[name] = ((got - ref.float()).norm() / ref.float().norm()).item()
        res = {name: timed(fn, iters) for name, fn in
               (("aiter+rs", aiter_rs), ("pyhip+rs", pyhip_rs), ("pyhip+fused", pyhip_fused))}
        d = torch.tensor([max(diffs.values())], device="cuda")
        dist.all_reduce(d, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"T={T:6d} sizes {sizes[0]}/{sizes[-1]}: " + "  ".join(f"{k} {v:7.1f}" for k, v in res.items())
                  + f" us | fused vs aiter+rs {res['pyhip+fused'] / res['aiter+rs']:.3f}x"
                  + "  rel vs aiter+rs " + " ".join(f"{k} {v:.2e}" for k, v in diffs.items()), flush=True)
    rs_fused.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
