"""Large BF16 all-reduce on 8x MI300X, lossless backends only: RCCL vs aiter custom AR (cutoff raised with
AITER_CUSTOM_AR_MAX_SIZE) vs sglang QuickAllReduce in FP mode (ROCM_QUICK_REDUCE_QUANTIZATION=FP) with the
BF16->FP16 cast on and off (readme 4.2).

PYTHONPATH=<sglang>/python:<aiter> NCCL_MIN_NCHANNELS=112 \
  torchrun --nproc_per_node 8 --tee 3 --log-dir /tmp/arlog ar_large_bench.py [tokens...]   (rank 0 output in the
  log dir; [tokens, 4096] BF16, tokens = global tokens)

Prints per size the slowest-rank median time of each backend and its rel_l2 against an FP32 all-reduce of the same
inputs.
"""
import os
import sys

os.environ.setdefault("AITER_CUSTOM_AR_MAX_SIZE", str(512 * 1024 * 1024))
os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "FP"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

H = 4096
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", device_id=torch.device("cuda", local))
rank, world = dist.get_rank(), dist.get_world_size()
cpu_group = dist.new_group(backend="gloo")
dev = torch.device("cuda", local)


def make_qr(cast):
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1" if cast else "0"
    from sglang.srt.distributed.device_communicators.quick_all_reduce import QuickAllReduce
    qr = QuickAllReduce(group=cpu_group, device=dev)
    return None if qr.disabled else qr


def rccl(x):
    y = x.clone()
    dist.all_reduce(y)
    return y


methods = {"RCCL": rccl}
try:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce as AiterCA
    ca = AiterCA(group=cpu_group, device=dev)
    if not ca.disabled:
        methods["aiter CA"] = ca.custom_all_reduce
except Exception as e:  # noqa: BLE001
    if rank == 0:
        print("aiter CA unavailable:", repr(e)[:200], flush=True)
for cast in (True, False):
    qr = make_qr(cast)
    if qr is not None:
        methods[f"QR FP ({'fp16 cast' if cast else 'bf16'})"] = qr.quick_all_reduce


def timed(fn, x, iters=20):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(x)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3)
    t = torch.tensor(sorted(times)[len(times) // 2], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


for tokens in [int(a) for a in sys.argv[1:]] or [8192, 16384, 32768, 65536]:
    torch.manual_seed(1234 + rank)
    x = (torch.randn(tokens, H, device=dev) * 2.0).to(torch.bfloat16)
    ref = x.float()
    dist.all_reduce(ref)
    line = []
    for name, fn in methods.items():
        y = fn(x).float()
        err = torch.tensor(float((y - ref).norm() / ref.norm()), device=dev)
        dist.all_reduce(err, op=dist.ReduceOp.MAX)
        line.append(f"{name} {timed(fn, x):7.1f} us (rel_l2 {float(err):.2e})")
    if rank == 0:
        print(f"tokens {tokens:6d} ({tokens * H * 2 >> 20} MiB): " + "  ".join(line), flush=True)
dist.destroy_process_group()
