# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE down `8x1` (BM=256 / 8 wave) 的独立正确性与性能 harness。

不依赖 aiter：sorting metadata 与 weight preshuffle 都在纯 torch 里构造，
因此在 aiter 无法 import 的环境里仍可运行。设计与实测记录见
``docs/design_moe_gemm2_8x1.md``。

    python3 tests/contrib/moe/test_down_8x1.py test
    python3 tests/contrib/moe/test_down_8x1.py bench --case qwen35b_k256

⚠️ 测量协议：单进程首次运行会因时钟未爬满而偏慢约 8%。对比性能时必须弃掉每轮
第一次运行、每个配置重复 >= 3 次，详见文档 §0.5。
"""

import argparse
import math
import os
import statistics
import sys

# 异常环境（editable pyhip 指向别的 checkout / aiter 无法 import）下的可选修补。
if os.environ.get("PYHIP_8X1_ENV_FIX", "1") != "0":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _env_workaround  # noqa: F401

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

FP8 = torch.float8_e4m3fnuz
DEV = "cuda"
FP8_PEAK_TFLOPS = 589.8  # MI308X: 80 CU x 4096 FLOP/clk x 1.8 GHz

_FX_DTYPE = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    FP8: fx.Uint8,
}

# name -> (B, Hidden(N), Inter-TP(K), E, TopK)
CASES = {
    "qwen35b_k256": (32768, 2048, 256, 256, 8),
    "qwen397b_k256": (32768, 4096, 256, 512, 10),
    "xiaomi_k256": (32768, 6144, 256, 384, 8),
    "h3_k384": (32768, 6144, 384, 128, 4),
}


def _compile(N, K, TOPK, E, BM=256, BN=128):
    from pyhip.contrib.flydsl.moe_gemm_2stage.gemm2 import compile_moe_gemm2

    return compile_moe_gemm2(
        N=N,
        K=K,
        weight_dtype="fp8",
        weight_quant_type="ptpc",
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BM,
        BLOCK_TILE_SIZE_N=BN,
        alg="prefill_1x4",
        E=E,
        act_quant_type="ptpc",
        down_path="8x1",
        down_output_padding_bytes=0,
    )


def to_ptr(t):
    return flyc.from_c_void_p(_FX_DTYPE[t.dtype], t.data_ptr())


def preshuffle(w):
    """[E,N,K] -> addr(c,k) = (c/16)*16K + (k/16)*256 + (c%16)*16 + (k%16).

    与 `moe_gemm_8wave_down` 的 bpreshuffle 等价：按 `[N/16 通道组][K/16][16 通道][16 k]`
    排布，使 global load / ds_write / ds_read 三处同时连续。
    """
    E, N, K = w.shape
    return (
        w.view(E, N // 16, 16, K // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(E, N * K)
    )


def build_metadata(topk_ids, E, BM):
    """纯 torch 复刻 moe_sorting 的输出（按 expert 分组 + 向 BM 补齐）。"""
    ntok, topk = topk_ids.shape
    flat_e = topk_ids.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e, minlength=E)
    sorted_rows, sorted_expert = [], []
    pos = 0
    for e in range(E):
        n = int(counts[e])
        rows = order[pos : pos + n]
        pos += n
        nblk = max(1, (n + BM - 1) // BM)
        pad = nblk * BM - n
        sorted_rows.append(
            torch.cat([rows, torch.full((pad,), -1, dtype=torch.long, device=rows.device)])
        )
        sorted_expert += [e] * nblk
    src = torch.cat(sorted_rows)
    valid = src >= 0
    tok = torch.where(valid, src // topk, torch.zeros_like(src))
    slot = torch.where(valid, src % topk, torch.zeros_like(src))
    enc = (slot.to(torch.int32) << 24) | tok.to(torch.int32)
    return (
        enc.contiguous(),
        torch.tensor(sorted_expert, dtype=torch.int32, device=DEV),
        valid,
        src,
    )


def run_correctness(K=256, N=256, ntok=256, topk=4, E=8, BM=256, BN=128, tol=0.02):
    torch.manual_seed(0)
    topk_ids = torch.randint(0, E, (ntok, topk), device=DEV)
    sorted_ids, sorted_expert, valid, src = build_metadata(topk_ids, E, BM)
    nrows, ntask = sorted_ids.numel(), sorted_expert.numel()

    a = (torch.randn(ntok, topk, K, device=DEV) / 8).to(FP8)
    w = (torch.randn(E, N, K, device=DEV) / 8).to(FP8)
    a_scale = torch.rand(ntok, topk, device=DEV, dtype=torch.float32) * 0.5 + 0.5
    w_scale = torch.rand(E, N, device=DEV, dtype=torch.float32) * 0.5 + 0.5
    routing = torch.rand(nrows, device=DEV, dtype=torch.float32) * 0.5 + 0.5
    out = torch.zeros(nrows, N, device=DEV, dtype=torch.bfloat16)
    num_valid = torch.tensor([nrows, ntok], dtype=torch.int32, device=DEV)

    _compile(N, K, topk, E, BM, BN)(
        to_ptr(a),
        to_ptr(preshuffle(w)),
        to_ptr(out),
        to_ptr(sorted_ids),
        to_ptr(routing),
        to_ptr(sorted_expert),
        to_ptr(num_valid),
        to_ptr(w_scale),
        to_ptr(a_scale),
        ntok,
        ntask,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    af, wf = a.float().view(-1, K), w.float()
    e_of_row = sorted_expert.repeat_interleave(BM).long()
    ref = torch.zeros(nrows, N, device=DEV, dtype=torch.float32)
    idx = torch.nonzero(valid).squeeze(1)
    for i in idx.tolist():
        r, e = int(src[i]), int(e_of_row[i])
        ref[i] = (af[r] @ wf[e].t()) * a_scale.view(-1)[r] * w_scale[e] * routing[i]

    got_v = out.float()[idx]
    ref_v = ref.to(torch.bfloat16).float()[idx]
    rel = (got_v - ref_v).abs().max() / ref_v.abs().max().clamp(min=1e-6)
    ok = bool(rel < tol)
    print(f"K={K} N={N} rows={nrows} valid={idx.numel()} rel={rel:.5f} {'PASS' if ok else 'MISMATCH'}")
    if not ok:
        # 通道/行置换诊断：恒等即说明布局正确，问题在数值。
        r0, g0 = ref_v[:64], got_v[:64]
        d = (g0.t()[:, None, :] - r0.t()[None, :, :]).abs().mean(-1)
        print("perm(got_ch -> ref_ch):", d.argmin(1)[:32].tolist())
        dr = (g0[:, None, :] - r0[None, :, :]).abs().mean(-1)
        print("perm(got_row -> ref_row):", dr.argmin(1)[:24].tolist())
    return ok


def run_bench(case, BM=256, BN=128, buffers=3, warmup=40, samples=50):
    B, N, K, E, TOPK = CASES[case]
    rows_per_expert = B * TOPK // E
    tiles = math.ceil(rows_per_expert / BM)
    ntask = E * tiles
    nrows = ntask * BM
    useful_flop = 2 * (B * TOPK) * N * K
    exec_flop = 2 * nrows * N * K

    torch.manual_seed(0)
    src = torch.arange(B * TOPK, device=DEV).view(E, -1)
    enc = ((src % TOPK).to(torch.int32) << 24) | (src // TOPK).to(torch.int32)
    sorted_ids = torch.cat(
        [enc, torch.zeros(E, tiles * BM - rows_per_expert, dtype=torch.int32, device=DEV)],
        dim=1,
    ).reshape(-1)
    sorted_expert = torch.arange(E, device=DEV, dtype=torch.int32).repeat_interleave(tiles).contiguous()
    num_valid = torch.tensor([nrows, B], dtype=torch.int32, device=DEV)
    routing = torch.rand(nrows, device=DEV, dtype=torch.float32)

    acts = [(torch.randn(B, TOPK, K, device=DEV) / 8).to(FP8) for _ in range(buffers)]
    ws = [preshuffle((torch.randn(E, N, K, device=DEV) / 8).to(FP8)) for _ in range(buffers)]
    a_scales = [torch.rand(B, TOPK, device=DEV, dtype=torch.float32) for _ in range(buffers)]
    w_scales = [torch.rand(E, N, device=DEV, dtype=torch.float32) for _ in range(buffers)]
    outs = [torch.zeros(nrows, N, device=DEV, dtype=torch.bfloat16) for _ in range(buffers)]

    launch = _compile(N, K, TOPK, E, BM, BN)

    def run(i):
        j = i % buffers
        launch(
            to_ptr(acts[j]), to_ptr(ws[j]), to_ptr(outs[j]), to_ptr(sorted_ids),
            to_ptr(routing), to_ptr(sorted_expert), to_ptr(num_valid),
            to_ptr(w_scales[j]), to_ptr(a_scales[j]), B, ntask,
            torch.cuda.current_stream().cuda_stream,
        )

    for i in range(warmup):
        run(i)
    torch.cuda.synchronize()

    times = []
    for i in range(samples):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        run(i)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med = statistics.median(times)

    print(f"case={case} B={B} N={N} K={K} E={E} TOPK={TOPK} BM={BM}")
    print(f"rows/expert={rows_per_expert} tiles={tiles} WG={ntask} pad={nrows/(B*TOPK):.3f}x")
    print(f"ms median={med:.4f} [P25={times[len(times)//4]:.4f} P75={times[3*len(times)//4]:.4f}]")
    print(f"useful TFLOPS={useful_flop/med*1e-9:.2f}  executed TFLOPS={exec_flop/med*1e-9:.2f}")
    print(f"peak_frac(executed)={exec_flop/med*1e-9/FP8_PEAK_TFLOPS*100:.1f}%")
    return med


def test_down_8x1_correctness():
    for K, N in [(128, 128), (128, 384), (256, 128), (256, 256), (256, 512), (384, 256), (384, 384)]:
        assert run_correctness(K=K, N=N)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "bench"])
    ap.add_argument("--case", default="qwen35b_k256", choices=list(CASES))
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1, help=">=3 并弃掉首次，见文档 §0.5")
    args = ap.parse_args()
    if args.mode == "test":
        sys.exit(0 if run_correctness(K=args.k, N=args.n) else 1)
    for r in range(args.repeat):
        med = run_bench(args.case)
        if args.repeat > 1:
            print(f"  ^ repeat {r} {'(丢弃：时钟未爬满)' if r == 0 else ''}")
