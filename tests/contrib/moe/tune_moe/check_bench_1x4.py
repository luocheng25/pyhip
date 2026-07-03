#!/usr/bin/env python3
"""Correctness check for the bench 1x4 core by SELF-CONSISTENCY: the a_cp_frag ping-pong
restructure is a pure pipeline/scheduling change (same per-tile K accumulation order), so
the raw C accumulator dump MUST be identical to the pre-change (known-correct) kernel.

Workflow:
  1. BEFORE editing _gemm_1x4:  python3 check_bench_1x4.py save ref_tk64.pt 64
                                python3 check_bench_1x4.py save ref_tk128.pt 128
  2. AFTER editing:             python3 check_bench_1x4.py compare ref_tk64.pt 64
                                python3 check_bench_1x4.py compare ref_tk128.pt 128
"""
import os
import sys

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import bench_gemm_core as B  # noqa: E402  (adds ../../../../src to sys.path on import)
import flydsl.compiler as flyc  # noqa: E402


def run(wave, BM, BN, tk, M=4096):
    torch.manual_seed(0)
    a = (torch.randn([M, B.K1], dtype=torch.bfloat16) + 1) * 0.001
    w = torch.randn([B.N1, B.K1], dtype=torch.bfloat16)
    nblk = (B.N1 // BN) * (M // BM)
    out = torch.zeros([nblk * 256 * B._MAX_VALS_PER_THREAD], dtype=torch.float32)
    jit = B.build(wave, B.N1, B.K1, BM, BN, tk)
    args = (B._ptr(a), B._ptr(w), B._ptr(out), M, B.STREAM)
    fn = flyc.compile(jit, *args)
    fn(*args)
    torch.cuda.synchronize()
    return out.clone()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "save"
    path = sys.argv[2] if len(sys.argv) > 2 else "ref.pt"
    tk = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    cfgs = [(64, 128), (64, 256), (128, 128), (128, 256)]
    res = {c: run("1x4", c[0], c[1], tk) for c in cfgs}
    if mode == "save":
        torch.save(res, path)
        print(f"saved {path} (tk={tk}, {len(cfgs)} configs)")
    else:
        ref = torch.load(path)
        ok = True
        for c in cfgs:
            md = (ref[c] - res[c]).abs().max().item()
            same = torch.equal(ref[c], res[c])
            status = "IDENTICAL" if same else ("OK" if md < 1e-2 else "MISMATCH")
            if md >= 1e-2:
                ok = False
            print(f"  BM={c[0]} BN={c[1]} tk={tk}: {status} (maxdiff={md:.3e})")
        print("ALL OK" if ok else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
