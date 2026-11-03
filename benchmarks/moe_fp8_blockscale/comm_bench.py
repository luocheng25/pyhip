"""RCCL collective timings for MiMo prefill communication schemes (8x MI300X, torchrun --nproc_per_node 8).

Global tokens N (both DP groups), H = 4096. Attention groups: ranks 0-3 and 4-7 (DP2 x TP4).
  current     : 2 x all_reduce(8, [N, H] BF16)                          (dp_gather_partial AR + MoE AR)
  hierarchical: RS4([N/2, H] BF16) + AG8([N/8, H] BF16) + RS8([N, H] BF16) + AG4([N/8, H] BF16)
                (sglang SGLANG_DP_HIERARCHICAL_COMM=1)
  sp_fp8      : as hierarchical, but both all-gathers carry FP8 activations + 1x128 scales (the consumers,
                MoE and QKV GEMMs, quantize anyway; norm + quant run on the token shard)
  sp_fp8_i8rs : as sp_fp8, but both reduce-scatters are all-to-alls of int8 partial rows + one FP32 scale
                per 32 values (H + H/8 bytes per row, aiter PR 6514 big-M "E rows"); the owner dequantizes
                and sums (local kernel not timed). Lossy: changes numerics, needs an accuracy check.
usage: torchrun --nproc_per_node 8 benchmarks/moe_fp8_blockscale/comm_bench.py [tokens...]
"""

import os
import sys

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
attn = [dist.new_group([0, 1, 2, 3]), dist.new_group([4, 5, 6, 7])]
mine = attn[rank // 4]
H = 4096


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


for n in [int(t) for t in sys.argv[1:]] or [8192, 32768]:
    full = torch.randn(n, H, device="cuda").to(torch.bfloat16)
    half = full[: n // 2].clone()
    shard8 = torch.empty(n // 8, H, device="cuda", dtype=torch.bfloat16)
    shard8_fp8 = torch.empty(n // 8, H + 128, device="cuda", dtype=torch.uint8)  # + 32 FP32 scales per token
    full_fp8 = torch.empty(n, H + 128, device="cuda", dtype=torch.uint8)
    half_fp8 = torch.empty(n // 2, H + 128, device="cuda", dtype=torch.uint8)
    half_bf16 = torch.empty(n // 2, H, device="cuda", dtype=torch.bfloat16)
    full_i8 = torch.empty(n, H + H // 8, device="cuda", dtype=torch.uint8)
    recv_i8 = torch.empty_like(full_i8)
    half_i8 = torch.empty(n // 2, H + H // 8, device="cuda", dtype=torch.uint8)
    recv_half_i8 = torch.empty_like(half_i8)
    recv_full = torch.empty_like(full)
    r = {
        "AR8 [N,H] bf16": bench(lambda: dist.all_reduce(full)),
        "RS8 [N,H] bf16": bench(lambda: dist.reduce_scatter_tensor(shard8, full)),
        "A2A8 [N,H] bf16": bench(lambda: dist.all_to_all_single(recv_full, full)),
        "A2A8 [N,H] i8+s": bench(lambda: dist.all_to_all_single(recv_i8, full_i8)),
        "AG8 [N,H] bf16": bench(lambda: dist.all_gather_into_tensor(full, shard8)),
        "AG8 [N,H] fp8+s": bench(lambda: dist.all_gather_into_tensor(full_fp8, shard8_fp8)),
        "RS4 [N/2,H] bf16": bench(lambda: dist.reduce_scatter_tensor(shard8, half, group=mine)),
        "A2A4 [N/2,H] i8+s": bench(lambda: dist.all_to_all_single(recv_half_i8, half_i8, group=mine)),
        "AG4 [N/2,H] bf16": bench(lambda: dist.all_gather_into_tensor(half_bf16, shard8, group=mine)),
        "AG4 [N/2,H] fp8+s": bench(lambda: dist.all_gather_into_tensor(half_fp8, shard8_fp8, group=mine)),
    }
    if rank == 0:
        print(f"== N={n} global tokens ({n * H * 2 >> 20} MiB BF16 buffer), NCCL_MIN_NCHANNELS="
              f"{os.environ.get('NCCL_MIN_NCHANNELS', '-')}")
        for k, v in r.items():
            print(f"   {k:20s} {v:9.1f} us")
        cur = 2 * r["AR8 [N,H] bf16"]
        hier = r["RS4 [N/2,H] bf16"] + r["AG8 [N,H] bf16"] + r["RS8 [N,H] bf16"] + r["AG4 [N/2,H] bf16"]
        sp8 = r["RS4 [N/2,H] bf16"] + r["AG8 [N,H] fp8+s"] + r["RS8 [N,H] bf16"] + r["AG4 [N/2,H] fp8+s"]
        i8 = r["A2A4 [N/2,H] i8+s"] + r["AG8 [N,H] fp8+s"] + r["A2A8 [N,H] i8+s"] + r["AG4 [N/2,H] fp8+s"]
        print(f"   per layer: current 2xAR {cur:.0f} us | hierarchical {hier:.0f} us ({hier / cur:.2f}x) | "
              f"SP+FP8 gathers {sp8:.0f} us ({sp8 / cur:.2f}x) | + int8 RS {i8:.0f} us ({i8 / cur:.2f}x)", flush=True)
dist.destroy_process_group()
