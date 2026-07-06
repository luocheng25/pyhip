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
    dummy_scale = torch.empty(0, dtype=torch.float32, device="cuda")
    jit_bf16 = B.build(wave, N, K, BM, BN, tk)
    args_bf16 = (B._ptr(a), B._ptr(w_shuffled), B._ptr(out_bf16), B._ptr(dummy_scale), B._ptr(dummy_scale), M, B.STREAM)
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


def _preshuffle_fp8(w_fp8, N, K):
    """Apply preshuffle layout for fp8: ((16, N//16), (16, K//16)), strides ((16, 16K), (1, 256)).
    Input: w_fp8 [N, K] in fp8, Output: shuffled flat tensor in fp8."""
    import numpy as np
    w_flat = w_fp8.reshape(-1).view(torch.uint8).cpu().numpy()
    element_num = 16  # 16 bytes / 1 byte per fp8
    n_idx = np.arange(N)[:, None].repeat(K, axis=1)
    k_idx = np.arange(K)[None, :].repeat(N, axis=0)
    n0 = n_idx % 16
    n1 = n_idx // 16
    k0 = k_idx % element_num
    k1 = k_idx // element_num
    phys = n0 * element_num + n1 * 16 * K + k0 + k1 * 16 * element_num
    result = np.empty(N * K, dtype=np.uint8)
    result[phys.ravel()] = w_flat
    return torch.from_numpy(result).view(torch.float8_e4m3fnuz).to(device=w_fp8.device).reshape(N, K)


def _preshuffle_silu_fp8(w_fp8, N, K, contiguous_n):
    """Apply composition(preshuffle, group_layout_silu) for fp8 weight."""
    import numpy as np
    w_flat = w_fp8.reshape(-1).view(torch.uint8).cpu().numpy()
    element_num = 16
    n_idx = np.arange(N)[:, None].repeat(K, axis=1)
    k_idx = np.arange(K)[None, :].repeat(N, axis=0)
    # group_layout_silu: ((cn, 2, N//(cn*2)), K) → strides ((1, N//2, cn), N)
    n0 = n_idx % contiguous_n
    n1 = (n_idx // contiguous_n) % 2
    n2 = n_idx // (contiguous_n * 2)
    addr_silu = n0 + n1 * (N // 2) + n2 * contiguous_n + k_idx * N
    # preshuffle on top
    pn = addr_silu % N
    pk = addr_silu // N
    pn0 = pn % 16
    pn1 = pn // 16
    pk0 = pk % element_num
    pk1 = pk // element_num
    phys = pn0 * element_num + pn1 * 16 * K + pk0 + pk1 * 16 * element_num
    result = np.empty(N * K, dtype=np.uint8)
    result[phys.ravel()] = w_flat
    return torch.from_numpy(result).view(torch.float8_e4m3fnuz).to(device=w_fp8.device).reshape(N, K)


def run_and_check_fp8_ptpc(wave, BM, BN, tk, M=256):
    """Run bench kernel fp8 ptpc and compare against dequantized torch reference."""
    torch.manual_seed(0)
    N, K = B.N1, B.K1

    # Generate bf16 data, quantize to fp8
    a_bf16 = (torch.randn([M, K], dtype=torch.bfloat16) + 1) * 0.001
    w_bf16 = torch.randn([N, K], dtype=torch.bfloat16)
    a_fp8, a_scale = B._quantize_to_fp8(a_bf16)
    w_fp8, w_scale = B._quantize_weight_to_fp8(w_bf16)

    # Reference: fp8 GEMM with per-channel/per-token dequant, silu column reorder
    # Build silu column mapping: kernel output col → original weight row
    cn = BN // 2
    col_map = torch.empty(N, dtype=torch.long)
    for n in range(N // 2):
        col_map[n] = 2 * cn * (n // cn) + n % cn
        col_map[N // 2 + n] = 2 * cn * (n // cn) + cn + n % cn

    # Compute raw fp8 GEMM in silu order, then apply scales
    raw_gemm = a_fp8.float() @ w_fp8[col_map].float().t()  # (M, N)
    # Dequant: C_real[m, n] = raw_gemm[m, n] * a_scale[m] * w_scale[col_map[n]]
    ref_out = (raw_gemm * a_scale.unsqueeze(-1) * w_scale[col_map].unsqueeze(0)).to(torch.bfloat16)

    # Shuffle weight for kernel using fp8 preshuffle+silu composition
    contiguous_n = BN // 2
    w_shuffled_fp8 = _preshuffle_silu_fp8(w_fp8, N, K, contiguous_n)

    out = torch.zeros([M, N], dtype=torch.bfloat16)
    jit = B.build(wave, N, K, BM, BN, tk, in_dtype="fp8", quant_type="ptpc")
    args = (B._fp8_ptr(a_fp8), B._fp8_ptr(w_shuffled_fp8), B._ptr(out),
            B._f32_ptr(w_scale), B._f32_ptr(a_scale), M, B.STREAM)
    fn = flyc.compile(jit, *args)
    fn(*args)
    torch.cuda.synchronize()

    diff = (ref_out.float() - out.float()).abs()
    max_diff = diff.max().item()
    ref_abs = ref_out.float().abs()
    rel_err = (diff / (ref_abs + 1e-6)).mean().item()
    nz = (out != 0).sum().item()
    return max_diff, rel_err, nz, out.numel()


def main():
    tk = 64
    dtype = "bf16"
    cfgs = [(64, 128), (64, 256), (128, 128)]
    for arg in sys.argv[1:]:
        if arg.startswith("tk="):
            tk = int(arg.split("=")[1])
        elif arg.startswith("tiles="):
            cfgs = [tuple(int(x) for x in t.split("x")) for t in arg.split("=")[1].split(",")]
        elif arg.startswith("dtype="):
            dtype = arg.split("=")[1]

    if dtype == "fp8_ptpc":
        if tk == 64:
            tk = 128  # fp8 default
        print(f"Checking 1x4 B-first fp8 ptpc kernel (tk={tk})")
        all_ok = True
        for BM, BN in cfgs:
            max_diff, rel_err, nz, total = run_and_check_fp8_ptpc("1x4", BM, BN, tk)
            # fp8 GEMM: rel_err < 15% is expected (fp8 e4m3 precision ~12.5%, MFMA vs torch accum order)
            status = "OK" if rel_err < 0.15 and max_diff < 0.5 else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  BM={BM} BN={BN} tk={tk}: {status} (max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, nonzero={nz}/{total})")
        print("ALL OK" if all_ok else "*** MISMATCH ***")
    else:
        print(f"Checking 1x4 B-first kernel against torch matmul (tk={tk})")
        all_ok = True
        for BM, BN in cfgs:
            max_diff, rel_err, nz, total = run_and_check("1x4", BM, BN, tk)
            status = "OK" if rel_err < 0.01 and max_diff < 1.0 else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  BM={BM} BN={BN} tk={tk}: {status} (max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, nonzero={nz}/{total})")
        print("ALL OK" if all_ok else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
