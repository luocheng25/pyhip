"""Per-kernel timing of the I = 256 route-row path (sort, tiles, gate_up, inverse, down_direct), MiMo TP8 shapes, one GPU.

Same calls as sglang's pyhip_fp8_blockscale_moe.moe_route_rows. Outputs are compared per route (independent of
the sorting order): A2 / A2S / Y rows of route (t, k) are gathered through inv.

python benchmarks/moe_fp8_blockscale/probes/route_rows_bench.py [--tokens 32768] [--reps 20]
       [--save /dev/shm/rr_base | --check /dev/shm/rr_base] [--only gate_up down_direct]
       [--routing random|balanced|blocks|skewed] [--ablate gate_up_epilogue|gate_up_stores|down_stores|...]

routing: random = sigmoid top-8 of random logits (default); balanced = every expert exactly T * 8 / E routes from
random tokens (no compact tails); blocks = balanced with each expert's tokens in 8 contiguous runs (X locality);
skewed = random logits plus a per-expert bias (std 0.5).
ablate (FLYDSL_RUNTIME_ENABLE_CACHE=0; timing only, wrong outputs): gate_up_epilogue = one store of the summed
accumulators instead of SiLU / requant / A2 stores; gate_up_stores / down_stores = stores past the buffer end;
down_dma_half = half of every W2 chunk's LDS-DMA pieces.
gate_up_direct_stores (bit-exact A/B reference) = the earlier epilogue without LDS staging.
"""

import argparse
import os
import statistics

import torch

import pyhip

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "pyhip")
if os.path.isdir(_SRC) and os.path.abspath(_SRC) not in pyhip.__path__:
    pyhip.__path__.insert(0, os.path.abspath(_SRC))

from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import kernels as _k  # noqa: E402
from pyhip.ops.moe.flydsl.moe_fp8_blockscale_942 import moe as _m  # noqa: E402

FP8 = torch.float8_e4m3fnuz
H, E, I, TOPK = 4096, 256, 256, 8


def make_inputs(tokens, seed=0, routing="random"):
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, H, device="cuda", generator=g).to(torch.bfloat16)
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * 0.5).to(FP8)
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) * 0.5).to(FP8)
    s13 = torch.rand(E, 2 * I // 128, H // 128, device="cuda", generator=g) * 0.02 + 0.005
    s2 = torch.rand(E, H // 128, I // 128, device="cuda", generator=g) * 0.02 + 0.005
    wts, ids = torch.topk(torch.sigmoid(torch.randn(tokens, E, device="cuda", generator=g)), TOPK, -1)
    if routing == "skewed":
        # Per-expert popularity bias (std 0.5 on the logits): uneven expert loads, as with a real router.
        bias = torch.randn(E, device="cuda", generator=g) * 0.5
        wts, ids = torch.topk(torch.sigmoid(torch.randn(tokens, E, device="cuda", generator=g) + bias), TOPK, -1)
    elif routing != "random":
        # expert(t, k) = (sigma(t) + k * E / TOPK) % E: distinct per token, every expert T * TOPK / E routes
        sigma = torch.arange(tokens, device="cuda") % E
        if routing == "balanced":
            sigma = sigma[torch.randperm(tokens, device="cuda", generator=g)]
        else:
            sigma = torch.arange(tokens, device="cuda") // (tokens // E)
        ids = (sigma[:, None] + torch.arange(TOPK, device="cuda")[None, :] * (E // TOPK)) % E
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    x8, xs = _m.quantize_input(x)
    return dict(x8=x8, xs=xs, w13=shuffle_weight(w13, (16, 16)), w2=shuffle_weight(w2, (16, 16)), s13=s13, s2=s2,
                wts=wts, ids=ids.to(torch.int32))


class RouteRows:
    """sglang moe_route_rows split into separately launchable steps."""

    def __init__(self, c):
        self.c = c
        x8 = c["x8"]
        self.tokens = x8.shape[0]
        self.stream = torch.cuda.current_stream()
        self.sort()

    def sort(self):
        c = self.c
        self.sid, self.sw, self.seid, self.nv = _m.moe_sorting(c["ids"], c["wts"], E, H, None, _m.COMPACT_BLOCK)
        self.rows, self.blocks = self.sid.numel(), self.seid.numel()
        self.tiles, self.count, self.cap = _m._tiles(self.seid, self.nv, E, _m.COMPACT_BLOCK, self.stream)
        self.a2, self.a2s, self.y = _m._workspace(x8_device(c), self.rows, I, H)
        self.inv = torch.full((self.tokens, TOPK), -1, dtype=torch.int32, device="cuda")

    def gate_up(self):
        c, u8 = self.c, torch.uint8
        _m._run(("gate_up", self.sid.device, I, H), _k.launch_gate_up,
                c["x8"].view(u8).view(-1), c["w13"].view(u8).view(-1), c["xs"].reshape(-1), c["s13"].reshape(-1),
                self.sid, self.tiles, self.count, self.sw, self.a2, self.a2s, self.tokens, self.rows,
                self.cap * (I // 128), I, H, self.stream)

    def inverse(self):
        _m._run(("inverse", self.sid.device, TOPK), _k.launch_inverse,
                self.sid, self.nv, self.inv.view(-1), self.tokens, self.rows,
                (self.rows + _k.SUM_THREADS - 1) // _k.SUM_THREADS, TOPK, self.stream)

    def down_direct(self):
        c, u8 = self.c, torch.uint8
        groups = H // _k.DOWN_COLS
        _m._run(("down_direct", self.sid.device, H, I, groups), _k.launch_down_direct,
                self.a2, c["w2"].view(u8).view(-1), self.a2s, c["s2"].reshape(-1), self.tiles, self.count, self.sid,
                self.y, self.rows, self.tokens, 0, self.cap * groups, H, I, groups, self.stream)

    def per_route(self):
        """(A2 [T*8, I] bytes, A2S [I/128, T*8], Y [T*8, H]) of each route, sorted-order independent."""
        idx = self.inv.view(-1).long()
        assert bool((idx >= 0).all())
        a2 = self.a2.view(self.rows, I)[idx]
        a2s = self.a2s.view(I // 128, self.rows)[:, idx]
        y = self.y.view(self.rows, H)[idx]
        return a2, a2s, y


def x8_device(c):
    return c["x8"].device


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3)
    return statistics.median(out), min(out), max(out)


def ablate(what):
    """Patch kernel helpers before the first compile (timing experiments only)."""
    import flydsl.expr as fx

    if what == "gate_up_epilogue":
        def epilogue(acc, storage, shared, wave, lane, wm, wn, r16, g, SW, A2, A2S, R, row0, rows, jt, I):
            total = acc[0][0]
            for v in acc:
                for i in range(16):
                    total = total + v[i]
            res = _k._buffer(A2S, (I // 128) * R * 4)
            _k._store(total, res, (fx.Int32(rows) < 0).select(fx.Int32(0), fx.Int32(_k._DROP)))
        _k._act_epilogue = epilogue
    elif what == "gate_up_stores":
        orig_epilogue = _k._act_epilogue

        def epilogue(*args):
            # A2 / A2S stores only (tiles / inverse keep _store)
            orig_store = _k._store
            _k._store = lambda value, resource, voffset: _store_dropped(value, resource)
            try:
                orig_epilogue(*args)
            finally:
                _k._store = orig_store
        _k._act_epilogue = epilogue
    elif what == "down_stores":
        orig = _k._store_pair
        _k._store_pair = lambda first, second, y_res, voffset, soffset: orig(first, second, y_res,
                                                                              fx.Int32(_k._DROP), soffset)
    elif what == "down_dma_half":
        # Half of every W2 chunk's DMA pieces (stale LDS for the rest): W2 traffic / DMA issue of 512-row tasks.
        def dma_chunk(resource, storage, wave, lane_off, slot, chunk, K):
            pieces = _k.DOWN_CHUNK * K // 1024
            for t in range(pieces // 2):
                p = wave * pieces + t
                _k._dma(resource, storage, slot * (_k.DOWN_CHUNK * K) + p * 256, lane_off,
                        chunk * (_k.DOWN_CHUNK * K) + p * 256)
        _k._dma_chunk = dma_chunk
    elif what == "gate_up_direct_stores":
        # Pre-LDS-staging epilogue (A/B reference): 16 dword stores per lane, 16 bytes per row and store.
        def epilogue(acc, storage, shared, wave, lane, wm, wn, r16, g, SW, A2, A2S, R, row0, rows, jt, I):
            act = [[_k._silu_mul(acc[nbl][i], acc[4 + nbl][i]) for i in range(16)] for nbl in range(4)]
            amax = []
            for mb in range(4):
                m = _k._abs(act[0][4 * mb])
                for i in range(1, 16):
                    m = _k._max(m, _k._abs(act[i // 4][4 * mb + i % 4]))
                m = _k._max(m, fx.Float32(m.shuffle_xor(16, 64)))
                amax.append(_k._max(m, fx.Float32(m.shuffle_xor(32, 64))))
            own = (g == 0).select(amax[0], (g == 1).select(amax[1], (g == 2).select(amax[2], amax[3])))
            _k._write_lds_f32(shared + wave * 8192 + lane * 4, own)
            _k._wait(lgkmcnt=0)
            _k._stage_end()
            other = shared + (wave ^ 1) * 8192
            partner = [_k._read_lds_f32(other + (16 * mb + r16) * 4) for mb in range(4)]
            partner_own = _k._read_lds_f32(other + lane * 4)
            sw_res = _k._buffer(fx.make_view(fx.get_iter(SW) + fx.Int64(row0), fx.make_layout(_k.BM, 1)), _k.BM * 4)
            route = _k._load_f32(sw_res, (wm * 64 + lane) * 4)
            _k._wait(lgkmcnt=0)
            a2_res = _k._buffer(fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * I, fx.make_layout(_k.BM * I, 1)),
                                rows * I)
            column = jt * 128 + wn * 64 + g * 4
            for mb in range(4):
                top = _k._max(amax[mb], partner[mb])
                inv = (top > fx.Float32(0.0)).select(fx.Float32(_k.FP8_MAX) / top, fx.Float32(0.0))
                row = wm * 64 + 16 * mb + r16
                for nbl in range(4):
                    word = _k._cvt_fp8x4([act[nbl][4 * mb + e] * inv for e in range(4)])
                    _k._store(word, a2_res, row * I + column + nbl * 16)
            top = _k._max(own, partner_own)
            scale = (top > fx.Float32(0.0)).select(top * fx.Float32(1.0 / _k.FP8_MAX) * route, fx.Float32(0.0))
            a2s_res = _k._buffer(A2S, (I // 128) * R * 4)
            _k._store(scale, a2s_res, ((wn == 0) & (wm * 64 + lane < rows)).select(
                (jt * R + row0 + wm * 64 + lane) * 4, fx.Int32(_k._DROP)))
        _k._act_epilogue = epilogue
    else:
        raise ValueError(what)


def _store_dropped(value, resource):
    import flydsl.expr as fx
    from flydsl.expr import rocdl

    rocdl.raw_ptr_buffer_store(value.ir_value(), resource, fx.Int32(_k._DROP).ir_value(), fx.Int32(0).ir_value())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--save")
    p.add_argument("--check")
    p.add_argument("--only", nargs="+", default=["sort", "gate_up", "inverse", "down_direct"])
    p.add_argument("--routing", default="random", choices=["random", "balanced", "blocks", "skewed"])
    p.add_argument("--ablate")
    a = p.parse_args()
    if a.ablate:
        ablate(a.ablate)
    c = make_inputs(a.tokens, routing=a.routing)
    rr = RouteRows(c)
    rr.gate_up()
    rr.inverse()
    rr.down_direct()
    torch.cuda.synchronize()
    routes = a.tokens * TOPK
    flops = {"gate_up": 2.0 * routes * 2 * I * H, "down_direct": 2.0 * routes * H * I}
    nbytes = {"down_direct": routes * H * 2 + E * H * I + routes * I}
    print(f"T={a.tokens} rows(capacity)={rr.rows} tiles={int(rr.count.item())}")
    for name in a.only:
        med, lo, hi = timed(getattr(rr, name), a.reps)
        extra = ""
        if name in flops:
            extra += f"  {flops[name] / med / 1e6:7.0f} TFLOPS"
        if name in nbytes:
            extra += f"  {nbytes[name] / med / 1e6:6.2f} TB/s"
        print(f"{name:12s} median {med:8.1f} us (min {lo:8.1f}, max {hi:8.1f}){extra}", flush=True)
    if a.save or a.check:
        rr.gate_up()
        rr.inverse()
        rr.down_direct()
        torch.cuda.synchronize()
        a2, a2s, y = rr.per_route()
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            torch.save({"a2": a2.cpu(), "a2s": a2s.cpu(), "y": y.cpu()}, os.path.join(a.save, f"t{a.tokens}.pt"))
            print("saved", a.save)
        else:
            ref = torch.load(os.path.join(a.check, f"t{a.tokens}.pt"))
            for key, got in (("a2", a2), ("a2s", a2s), ("y", y)):
                want = ref[key].cuda()
                same = torch.equal(got.view(torch.uint8) if got.dtype == torch.uint8 else got, want)
                diff = 0 if same else int((got.view(-1).view(torch.uint8) != want.view(-1).view(torch.uint8)).sum())
                print(f"check {key}: {'bit-exact' if same else f'MISMATCH ({diff} bytes differ)'}")


if __name__ == "__main__":
    main()
