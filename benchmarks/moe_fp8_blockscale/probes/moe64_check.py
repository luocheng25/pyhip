"""1-stage M64 MoE kernel (kernels_m64.moe64: gemm1 + requant + down per 64-row block, readme 4.15) against the
2-stage short-K path (tile gate_up + direct down) on one GPU, for 8-wave CTAs (one per CU) and 4-wave CTAs
(two per CU).

python moe64_check.py [tokens...]        (TP8 shard: H 4096, I 256, 256 experts, top-8)

Checks that gate_up64's A2 bytes and moe64's Y rows (valid routes) are bit-identical to the 2-stage path, and
times each kernel back to back (20 launches after 3 warm-ups).
"""

import os
import sys

import flydsl.compiler as flyc
import torch

import pyhip

here = os.path.dirname(os.path.abspath(__file__))
pyhip.__path__.insert(0, os.path.join(here, "..", "..", "..", "src", "pyhip"))
sys.path.insert(0, os.path.dirname(here))
import bench_moe_fp8_blockscale as b  # noqa: E402

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as K  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels_m64 as K64  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as M  # noqa: E402

H, I, E = 4096, 256, 256
u8 = torch.uint8
compiled = {}


def run(key, launcher, *args):
    if key not in compiled:
        compiled[key] = flyc.compile(launcher, *args)
    else:
        compiled[key](*args)


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def main():
    dev = torch.device("cuda", 0)
    for T in [int(t) for t in sys.argv[1:]] or [8192, 32768]:
        c = b.make_case("tp8", T)
        stream = torch.cuda.current_stream()
        x8, xs = M.quantize_input(c["x"])
        sid, sw, seid, nv = M.moe_sorting(c["ids"], c["wts"], E, H, None, M.COMPACT_BLOCK)
        tiles, count, cap = M._tiles(seid, nv, E, M.COMPACT_BLOCK, stream)
        rows = sid.numel()
        a2 = [torch.zeros(rows * I, dtype=u8, device=dev) for _ in range(2)]
        a2s = [torch.zeros(2 * rows, dtype=torch.float32, device=dev) for _ in range(2)]
        y = [torch.zeros(rows * H, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        g1 = (x8.view(u8).view(-1), c["w13_sh"].view(u8).view(-1), xs.reshape(-1), c["s13"].reshape(-1))
        w2, ws2 = c["w2_sh"].view(u8).view(-1), c["s2"].reshape(-1)
        groups = H // K.DOWN_COLS

        def gate_up():
            M._run(("gate_up", dev, I, H), K.launch_gate_up, *g1, sid, tiles, count, sw, a2[0], a2s[0], T, rows,
                   cap * (I // 128), I, H, stream)

        def down(t=T):
            M._run(("down_direct", dev, H, I, groups), K.launch_down_direct, a2[0], w2, a2s[0], ws2, tiles, count,
                   sid, y[0], rows, t, 0, cap * groups, H, I, groups, stream)

        def gate_up64(nw):
            run(("gate_up64", nw), K64.launch_gate_up64, *g1, sid, seid, sw, nv, a2[1], a2s[1], T, rows,
                seid.numel(), I, H, nw, stream)

        def moe64(nw):
            run(("moe64", nw), K64.launch_moe64, *g1, w2, ws2, sid, seid, sw, nv, y[1], T, rows, seid.numel(), I, H,
                H, nw, stream)

        gate_up(); down()
        torch.cuda.synchronize()
        n = int(nv[0])
        valid = (sid[:n] & 0xFFFFFF) < T
        bad = []
        for nw in (8, 4):
            a2[1].zero_(); y[1].zero_()
            gate_up64(nw); moe64(nw)
            torch.cuda.synchronize()
            bad.append((nw, int((a2[0][: n * I] != a2[1][: n * I]).sum()),
                        int((y[0].view(rows, H)[:n][valid] != y[1].view(rows, H)[:n][valid]).sum())))
        times = {"gate_up": timed(gate_up), "down": timed(down), "gate_up+down": timed(lambda: (gate_up(), down())),
                 "down (no store)": timed(lambda: down(0))}  # T=0: every Y store is dropped, same MFMA work
        for nw in (8, 4):
            times[f"gate_up64 x{nw}"] = timed(lambda: gate_up64(nw))
            times[f"moe64 x{nw}"] = timed(lambda: moe64(nw))
        print(f"T={T}: sorted rows {n}, mismatches (waves, A2, Y) {bad} | "
              + "  ".join(f"{k} {v:.1f}" for k, v in times.items()) + " us", flush=True)


if __name__ == "__main__":
    main()
