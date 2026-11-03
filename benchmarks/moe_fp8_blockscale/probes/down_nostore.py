"""Short-K down (TP8, I=256) with and without its Y traffic: T=0 drops every store, MFMA work unchanged.

python down_nostore.py [tokens]            torch-profiler times of both variants (back-to-back MoE calls)
DOWN_T0=0|1 python down_nostore.py run     12 calls of one variant (wrap with rocprofv3 for clock counters)
"""

import os
import sys

import torch

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as m  # noqa: E402

drop = {"on": False}
_run = m._run


def patched(key, launcher, *args):
    if key[0] == "down_direct" and drop["on"]:
        args = args[:9] + (0,) + args[10:]  # T: every row counts as padding -> stores out of range
    return _run(key, launcher, *args)


m._run = patched
tokens = [a for a in sys.argv[1:] if a != "run"]
c = b.make_case("tp8", int(tokens[0]) if tokens else 32768)
b.run_pyhip(c)
torch.cuda.synchronize()

if "run" in sys.argv[1:]:
    drop["on"] = os.environ.get("DOWN_T0") == "1"
    for _ in range(12):
        b.run_pyhip(c)
    torch.cuda.synchronize()
    sys.exit(0)

from torch.profiler import ProfilerActivity, profile  # noqa: E402

for rnd in range(2):
    for on in (False, True):
        drop["on"] = on
        for _ in range(3):
            b.run_pyhip(c)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                b.run_pyhip(c)
            torch.cuda.synchronize()
        times = {e.key[:20]: e.device_time_total / e.count for e in prof.key_averages()
                 if any(s in e.key for s in ("_down_direct", "_gate_up", "_sum_kernel"))}
        print(f"round {rnd} {'T=0 (no Y stores)' if on else 'normal          '}: "
              + "  ".join(f"{k} {v:7.1f} us" for k, v in sorted(times.items())))
