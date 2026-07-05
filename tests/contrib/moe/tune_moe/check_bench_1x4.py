#!/usr/bin/env python3
"""Correctness check for the bench 1x4 core against torch matmul reference.

Usage:
  python3 check_bench_1x4.py [tk=64] [tiles=64x128,64x256]
"""
import os
import sys

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import bench_gemm_core as B  # noqa: E402
import flydsl.compiler as flyc  # noqa: E402


def _reconstruct_logical_weight(w_raw, N, K, contiguous_n):
    """Reconstruct the logical (N, K) weight from raw data through
    composition(preshuffle, group_layout_silu) — vectorized with torch."""
    import numpy as np
    w_flat = w_raw.reshape(-1).cpu().view(torch.uint16).numpy()
    element_num = 8  # 16 bytes / 2 bytes per bf16

    # Build index arrays for all (n, k) positions
    n_idx = np.arange(N)[:, None].repeat(K, axis=1)  # (N, K)
    k_idx = np.arange(K)[None, :].repeat(N, axis=0)  # (N, K)

    # group_layout_silu: ((cn, 2, N//(cn*2)), K) → ((1, N//2, cn), N)
    n0 = n_idx % contiguous_n
    n1 = (n_idx // contiguous_n) % 2
    n2 = n_idx // (contiguous_n * 2)
    addr_silu = n0 + n1 * (N // 2) + n2 * contiguous_n + k_idx * N

    # preshuffle: ((16, N//16), (elem, K//elem)) → ((elem, 16K), (1, 16*elem))
    pn = addr_silu % N
    pk = addr_silu // N
    pn0 = pn % 16
    pn1 = pn // 16
    pk0 = pk % element_num
    pk1 = pk // element_num
    phys = pn0 * element_num + pn1 * 16 * K + pk0 + pk1 * 16 * element_num

    w_logical_flat = w_flat[phys.ravel()]
    w_logical = torch.from_numpy(w_logical_flat.view(np.float16).astype(np.float32)).reshape(N, K).to(
        dtype=torch.bfloat16, device=w_raw.device
    )
    # bf16 reinterpret: numpy doesn't have bf16, use uint16 view
    w_logical_u16 = torch.from_numpy(w_logical_flat).reshape(N, K)
    w_logical = w_logical_u16.view(torch.bfloat16).to(device=w_raw.device)
    return w_logical


def run_and_check(wave, BM, BN, tk, M=256):
    """Run bench kernel bf16 output and compare against torch A @ W^T reference.
    Weight is shuffled via shuffle_weight before passing to the kernel."""
    torch.manual_seed(0)
    N, K = B.N1, B.K1
    a = (torch.randn([M, K], dtype=torch.bfloat16) + 1) * 0.001
    w_raw = torch.randn([N, K], dtype=torch.bfloat16)

    # Torch reference: A @ W^T using un-shuffled weight
    ref_out = a @ w_raw.t()

    # Shuffle weight for kernel (same as _bench_fly / preshuffle_gemm_v2)
    flydsl_root = os.path.join(os.path.dirname(__file__), "../../../../../FlyDSL")
    if flydsl_root not in sys.path:
        sys.path.insert(0, flydsl_root)
    from tests.utils import shuffle_weight
    w_shuffled = shuffle_weight(w_raw, layout=(16, 16))

    # Run bench kernel with shuffled weight — output is (M, N) bf16
    out_bf16 = torch.zeros([M, N], dtype=torch.bfloat16)
    jit_bf16 = B.build(wave, N, K, BM, BN, tk)
    args_bf16 = (B._ptr(a), B._ptr(w_shuffled), B._ptr(out_bf16), M, B.STREAM)
    fn_bf16 = flyc.compile(jit_bf16, *args_bf16)
    fn_bf16(*args_bf16)
    torch.cuda.synchronize()
    bench_out = out_bf16

    # Compare with tolerance (bf16 GEMM vs f32 torch)
    diff = (ref_out.float() - bench_out.float()).abs()
    max_diff = diff.max().item()
    ref_abs = ref_out.float().abs()
    rel_err = (diff / (ref_abs + 1e-6)).mean().item()
    nz = (bench_out != 0).sum().item()
    return max_diff, rel_err, nz, bench_out.numel()


def main():
    tk = 64
    cfgs = [(64, 128), (64, 256), (128, 128)]
    for arg in sys.argv[1:]:
        if arg.startswith("tk="):
            tk = int(arg.split("=")[1])
        elif arg.startswith("tiles="):
            cfgs = [tuple(int(x) for x in t.split("x")) for t in arg.split("=")[1].split(",")]

    print(f"Checking 1x4 B-first kernel against torch matmul (tk={tk})")
    all_ok = True
    for BM, BN in cfgs:
        max_diff, rel_err, nz, total = run_and_check("1x4", BM, BN, tk)
        # bf16 GEMM vs f32 torch: rel_err < 1% is excellent; max_diff < 1.0 (bf16 ULP range)
        status = "OK" if rel_err < 0.01 and max_diff < 1.0 else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  BM={BM} BN={BN} tk={tk}: {status} (max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, nonzero={nz}/{total})")
    print("ALL OK" if all_ok else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
