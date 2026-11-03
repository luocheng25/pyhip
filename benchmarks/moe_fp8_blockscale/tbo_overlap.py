"""TBO feasibility on 8x MI300X: one micro-batch's MoE (main stream) concurrent with the other's all-reduce.

Each rank runs its EP8 share (32 experts, I=2048; or TP8 aiter with --tp8) of a T-token micro-batch while an
RCCL all-reduce of another [T, 4096] BF16 buffer runs on a second stream.
usage: torchrun --nproc_per_node 8 benchmarks/moe_fp8_blockscale/tbo_overlap.py [tokens] [--tp8]
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_fp8_blockscale as b  # noqa: E402

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
tokens = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16384
tp8 = "--tp8" in sys.argv
b.CONFIGS["ep8"] = (2048, 32, rank)
case = b.make_case("tp8" if tp8 else "ep8", tokens, seed=rank)
moe = (lambda: b.run_aiter(case)) if tp8 else (lambda: b.run_pyhip(case))
buf = torch.randn(tokens, 4096, device="cuda").to(torch.bfloat16)
side = torch.cuda.Stream()


def ar():
    with torch.cuda.stream(side):
        dist.all_reduce(buf)


def timed(fn, reps=20):
    # Fixed warm-up count: every rank must issue the same number of collectives.
    for _ in range(60):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
        torch.cuda.current_stream().wait_stream(side)
    end.record()
    end.synchronize()
    t = torch.tensor(start.elapsed_time(end) / reps * 1e3, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


t_moe = timed(moe)
t_ar = timed(ar)
t_seq = timed(lambda: (moe(), ar()))


def both():
    side.wait_stream(torch.cuda.current_stream())
    ar()
    moe()


t_both = timed(both)
t_moe = min(t_moe, timed(moe))
t_ar = min(t_ar, timed(ar))
t_seq = min(t_seq, timed(lambda: (moe(), ar())))
t_both = min(t_both, timed(both))
if rank == 0:
    hidden = t_moe + t_ar - t_both
    print(f"{'TP8 aiter' if tp8 else 'EP8 pyhip'} MoE T={tokens}: moe {t_moe:.0f} us, all-reduce {t_ar:.0f} us, "
          f"sequential {t_seq:.0f} us, concurrent {t_both:.0f} us -> hidden {hidden:.0f} us "
          f"({hidden / t_ar * 100:.0f}% of the all-reduce)", flush=True)
dist.destroy_process_group()
