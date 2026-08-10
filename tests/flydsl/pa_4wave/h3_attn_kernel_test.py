"""Unit tests for the three varlen-attention kernels used by MiniMax-H3 on gfx942.

Kernels under test
------------------
  triton     aiter.ops.triton.attention.mha.flash_attn_varlen_func
  asm_group  aiter.flash_attn_varlen_func            (grouped-varlen ASM FMHA v3)
  asm_split  aiter.ops.mha._flash_attn_forward       (one non-varlen launch per doc)

Two things are checked, and they are deliberately separated:

  correctness  run on the real 63225+7 pack only, against *segment-wise bf16
               torch SDPA*: the pack is unpacked and each document run through
               F.scaled_dot_product_attention at bf16, the same dtype the
               candidates use, so a mismatch means the kernel differs rather
               than that bf16 rounds. Triton is reported as a second column
               since it is the incumbent production path, and the 7-token tail
               is scored separately from the pack. --ref fp32 swaps in a chunked
               fp32 oracle for absolute rather than SDPA-relative correctness.

  performance  CUDA-event timing, median of N. Reported per kernel, plus a
               pack-balance sweep that isolates how much each kernel wastes on
               the padded tail of an unbalanced pack.

Device selection matters on this box: GPU 0-3 run at fclk 1300 / mclk 900 while
GPU 4-7 run at 1550 / 1100. Benchmarking attention (memory-bound) on GPU 0 while
serving runs on GPU 4 gives numbers that are wrong in the *ordering*, not just
the magnitude. The default device is therefore cuda:0 of whatever
HIP_VISIBLE_DEVICES exposes, and the harness prints the clocks it measured on so
a stale comparison is obvious in the log.

Run as a test:      pytest -q h3_attn_kernel_test.py
Run correctness:    python h3_attn_kernel_test.py --check
                    python h3_attn_kernel_test.py --check --ref fp32
Run the benchmark:  python h3_attn_kernel_test.py --bench
Balance sweep:      python h3_attn_kernel_test.py --sweep-balance
Kernel identity:    python h3_attn_kernel_test.py --kernel-names
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import statistics
from dataclasses import dataclass

import torch

import aiter
from aiter.ops import mha as aiter_mha
from aiter.ops.triton.attention.mha import (
    flash_attn_varlen_func as triton_flash_attn_varlen_func,
)

# The real H3 shape, taken from H3_ATTN_DEBUG_SHAPE=1 on a live 4-GPU run:
#   q=k=v=(63232, 14, 128) bf16, cu_seqlens=[0, 63225, 63232], causal=False
H3_SEGMENTS = (63225, 7)
H3_HEADS = 14
H3_HEAD_DIM = 128

KERNELS = ("triton", "asm_group", "asm_split")
BF16_CVT = {0: "RTNE", 1: "RTNA", 2: "RTZ"}

# Full identity of what each label actually dispatches. The ASM entries load a
# distinct .co per (cvt, grouped) combination -- asm_group and asm_split are two
# separately compiled binaries, not two ways of calling one kernel. Confirmed at
# runtime by AITER_LOG_MORE=1 and by --kernel-names below.
_HSA_DIR = "/sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308"
CVT_SUFFIX = {0: "rtne", 1: "rtna", 2: "rtz"}
_MANGLE_LEN = {  # Itanium mangling embeds the identifier length
    "fmha_fwd_hd128_bf16_rtne": 24, "fmha_fwd_hd128_bf16_rtne_group": 30,
    "fmha_fwd_hd128_bf16_rtna": 24, "fmha_fwd_hd128_bf16_rtna_group": 30,
    "fmha_fwd_hd128_bf16_rtz": 23, "fmha_fwd_hd128_bf16_rtz_group": 29,
}
assert all(len(k) == v for k, v in _MANGLE_LEN.items())


def kernel_identity(kernel: str, cvt: int) -> dict:
    """Full name of the binary a (kernel, cvt) pair dispatches to."""
    if kernel == "triton":
        return {
            "entry": "aiter.ops.triton.attention.mha.flash_attn_varlen_func",
            "symbol": "_attn_fwd (Triton JIT, name resolved at runtime)",
            "binary": "JIT-compiled, no .co on disk",
        }
    suffix = CVT_SUFFIX[cvt]
    group = "_group" if kernel == "asm_group" else ""
    # The on-disk file drops the "fmha_" prefix that the mangled symbol carries.
    filename = f"fwd_hd128_bf16_{suffix}{group}.co"
    symbol_id = f"fmha_fwd_hd128_bf16_{suffix}{group}"
    binary = f"{_HSA_DIR}/{filename}"
    if not os.path.exists(binary):
        binary += "   <-- MISSING"
    return {
        "entry": (
            "aiter.flash_attn_varlen_func"
            if kernel == "asm_group"
            else "aiter.ops.mha._flash_attn_forward"
        ),
        "symbol": f"_ZN5aiter{_MANGLE_LEN[symbol_id]}{symbol_id}E",
        "binary": binary,
    }


# --------------------------------------------------------------------------
# inputs / reference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One attention problem: a packed sequence split into `segments`."""

    name: str
    segments: tuple[int, ...]
    heads: int = H3_HEADS
    head_dim: int = H3_HEAD_DIM

    @property
    def total(self) -> int:
        return sum(self.segments)

    @property
    def max_seqlen(self) -> int:
        return max(self.segments)

    @property
    def cu_seqlens_host(self) -> list[int]:
        out = [0]
        for s in self.segments:
            out.append(out[-1] + s)
        return out

    def flops(self) -> float:
        """2 GEMMs of S*S*D per head, per segment."""
        return sum(4.0 * s * s * self.head_dim * self.heads for s in self.segments)


def make_inputs(case: Case, device: torch.device, seed: int = 1101):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (case.total, case.heads, case.head_dim)
    q, k, v = (
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)
        for _ in range(3)
    )
    cu = torch.tensor(case.cu_seqlens_host, device=device, dtype=torch.int32)
    return q, k, v, cu


def reference_sdpa_bf16(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: Case, scale: float
) -> torch.Tensor:
    """Default reference: segment-wise bf16 torch SDPA.

    The varlen pack is unpacked and each document run through
    F.scaled_dot_product_attention at bf16 -- the same dtype the kernels under
    test use, so the comparison isolates *kernel* differences instead of folding
    in the bf16 rounding that every candidate shares. That makes the tolerance
    meaningfully tighter than against an fp32 oracle.
    """
    out = torch.empty_like(q)
    bounds = case.cu_seqlens_host
    for start, stop in zip(bounds, bounds[1:]):
        if stop <= start:
            continue
        # (S, H, D) -> (1, H, S, D), the layout SDPA wants.
        qs, ks, vs = (
            t[start:stop].transpose(0, 1).unsqueeze(0) for t in (q, k, v)
        )
        att = torch.nn.functional.scaled_dot_product_attention(
            qs, ks, vs, dropout_p=0.0, is_causal=False, scale=scale
        )
        out[start:stop] = att.squeeze(0).transpose(0, 1)
    return out


def reference_fp32_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    case: Case,
    scale: float,
    q_chunk: int = 512,
) -> torch.Tensor:
    """Optional stricter oracle: non-flash fp32 attention, chunked over queries.

    A 63225^2 fp32 score matrix is 16 GB per head, so queries go in blocks.
    Accumulation is fp32 throughout; only the final store is bf16. Use via
    --ref fp32 when you want to know whether a kernel is *right* in absolute
    terms rather than equivalent to SDPA.
    """
    out = torch.empty_like(q)
    bounds = case.cu_seqlens_host
    for start, stop in zip(bounds, bounds[1:]):
        if stop <= start:
            continue
        # (S, H, D) -> (H, S, D) so the matmul batches over heads.
        kt = k[start:stop].transpose(0, 1).float()
        vt = v[start:stop].transpose(0, 1).float()
        for qs in range(start, stop, q_chunk):
            qe = min(qs + q_chunk, stop)
            qt = q[qs:qe].transpose(0, 1).float()
            scores = torch.bmm(qt, kt.transpose(1, 2)) * scale
            probs = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.bmm(probs, vt).transpose(0, 1).to(out.dtype)
            del scores, probs
        del kt, vt
    return out


REFERENCES = {"sdpa": reference_sdpa_bf16, "fp32": reference_fp32_chunked}


def reference_attention(q, k, v, case, scale, kind: str = "sdpa"):
    return REFERENCES[kind](q, k, v, case, scale)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def run_triton(q, k, v, cu, case, scale, cvt):
    del cvt  # Triton path has no bf16-conversion knob.
    return _first(
        triton_flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=case.max_seqlen,
            max_seqlen_k=case.max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
        )
    )


def run_asm_group(q, k, v, cu, case, scale, cvt):
    return _first(
        aiter.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=case.max_seqlen,
            max_seqlen_k=case.max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            how_v3_bf16_cvt=cvt,
        )
    )


def run_asm_split(q, k, v, cu, case, scale, cvt):
    out = torch.empty_like(q)
    bounds = case.cu_seqlens_host
    for start, stop in zip(bounds, bounds[1:]):
        if stop <= start:
            continue
        aiter_mha._flash_attn_forward(
            q=q[start:stop].unsqueeze(0),
            k=k[start:stop].unsqueeze(0),
            v=v[start:stop].unsqueeze(0),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=False,
            return_softmax=False,
            how_v3_bf16_cvt=cvt,
            out=out[start:stop].unsqueeze(0),
        )
    return out


RUNNERS = {
    "triton": run_triton,
    "asm_group": run_asm_group,
    "asm_split": run_asm_split,
}


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float()
    cand = candidate.float()
    diff = (ref - cand).abs()
    denom = ref.abs().max().clamp_min(1e-12)
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            ref.flatten(), cand.flatten(), dim=0
        ).item(),
        "max_abs": diff.max().item(),
        "rel": (diff.max() / denom).item(),
        "finite": bool(torch.isfinite(cand).all().item()),
    }


def bench(fn, warmup: int = 3, iters: int = 10) -> dict:
    """CUDA-event timing. Median, not min.

    The ASM kernels sit closer to the power limit than Triton does, so their
    first iterations are optimistic; min-of-N flatters them by ~15% relative to
    what a sustained 50-step denoise loop actually sees.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def physical_index(index: int) -> int:
    """Map a torch device index back to the physical GPU rocm-smi reports.

    Under HIP_VISIBLE_DEVICES=4 the torch device is cuda:0 but the card is
    GPU[4]; reading GPU[0]'s clocks would report the wrong -- and on this box,
    differently clocked -- device.
    """
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not visible:
        return index
    try:
        return int(visible.split(",")[index])
    except (ValueError, IndexError):
        return index


def device_clocks(index: int) -> str:
    """Best-effort fclk/mclk readback so a downclocked-GPU run is visible."""
    index = physical_index(index)
    try:
        raw = subprocess.run(
            ["rocm-smi", "--showclocks"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except Exception:
        return "clocks=unavailable"
    want = f"GPU[{index}]"
    bits = [f"phys=GPU[{index}]"]
    for line in raw.splitlines():
        if want in line and ("fclk" in line or "mclk" in line):
            bits.append(line.split(":")[-1].strip())
    return ", ".join(bits) or "clocks=unavailable"


def resolve_device(spec: str) -> torch.device:
    dev = torch.device(spec)
    torch.cuda.set_device(dev)
    return dev


# --------------------------------------------------------------------------
# case grid
# --------------------------------------------------------------------------

# Correctness runs on the real shape only. A synthetic grid of small cases was
# tried first and every kernel passed every one of them identically, so it added
# runtime without adding information -- the only case that has ever discriminated
# between these kernels is the production 63225+7 pack.
H3_CASE = Case("h3_real", H3_SEGMENTS)


# --------------------------------------------------------------------------
# pytest
# --------------------------------------------------------------------------

try:
    import pytest
except ImportError:  # standalone use without pytest installed
    pytest = None


if pytest is not None:

    @pytest.fixture(scope="module")
    def device():
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        dev = resolve_device(os.environ.get("H3_TEST_DEVICE", "cuda:0"))
        print(f"\n[device] {torch.cuda.get_device_name(dev)} {device_clocks(dev.index)}")
        return dev

    @pytest.fixture(scope="module")
    def problem(device):
        """Inputs and reference for the real shape, built once."""
        scale = 1.0 / math.sqrt(H3_CASE.head_dim)
        q, k, v, cu = make_inputs(H3_CASE, device)
        ref = reference_attention(q, k, v, H3_CASE, scale)
        return q, k, v, cu, scale, ref

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("cvt", sorted(BF16_CVT))
    def test_matches_reference(kernel, cvt, problem):
        """Match segment-wise bf16 SDPA on the real 63225+7 pack.

        Scored twice. The whole-pack cosine is dominated by the 63225-token head,
        so a kernel returning garbage for the 7-token tail would still score
        ~0.99999 -- the tail gets its own assertion. That is not paranoia:
        --kernel-names shows asm_split routes the tail to a ck_tile kernel rather
        than the ASM one, so it is a genuinely separate code path.
        """
        if kernel == "triton" and cvt != 1:
            pytest.skip("triton has no bf16-cvt knob; one run is enough")
        q, k, v, cu, scale, ref = problem
        got = RUNNERS[kernel](q, k, v, cu, H3_CASE, scale, cvt)
        tag = f"{kernel}/{BF16_CVT[cvt]}"

        whole = compare(ref, got)
        assert whole["finite"], f"{tag} produced non-finite output"
        assert whole["cosine"] > 0.9999, f"{tag} cosine={whole['cosine']:.8f}"
        # Same dtype on both sides, so this is tighter than an fp32 comparison.
        # RTZ still costs a full bf16 ulp relative to RTNE, by design.
        assert whole["rel"] < 0.02, f"{tag} rel={whole['rel']:.5f}"

        head = H3_CASE.segments[0]
        tail = compare(ref[head:], got[head:])
        assert tail["cosine"] > 0.9999, f"{tag} tail cosine={tail['cosine']:.8f}"


# --------------------------------------------------------------------------
# standalone driver
# --------------------------------------------------------------------------


def cmd_check(args, dev):
    """Score every kernel against two references.

    Segment-wise bf16 SDPA is the primary reference -- same dtype as the
    candidates, so a mismatch means the *kernel* differs, not that bf16 rounds.
    Triton is the incumbent -- it says whether swapping to ASM changes what
    production has been emitting. They answer different questions, so both are
    reported. --ref fp32 swaps the first column for a stricter absolute oracle.
    """
    case = H3_CASE
    head = case.segments[0]
    scale = 1.0 / math.sqrt(case.head_dim)
    q, k, v, cu = make_inputs(case, dev)
    ref = reference_attention(q, k, v, case, scale, args.ref)
    tri = RUNNERS["triton"](q, k, v, cu, case, scale, 1)

    print(f"segments={case.segments} heads={case.heads} dim={case.head_dim}")
    print(
        f"{'kernel':<11}{'cvt':<6}{'cos_vs_' + args.ref:>13}{'max_abs':>10}"
        f"{'tail_cos':>13}{'cos_vs_triton':>15}{'max_abs':>10}"
    )
    bad = 0
    for kernel in KERNELS:
        for cvt in sorted(BF16_CVT):
            if kernel == "triton" and cvt != 1:
                continue
            got = tri if kernel == "triton" else RUNNERS[kernel](
                q, k, v, cu, case, scale, cvt
            )
            m = compare(ref, got)
            # The 7-token tail is 0.01% of the pack; without its own column a
            # kernel could return garbage there and still score 0.99999 overall.
            tail = compare(ref[head:], got[head:])
            t = compare(tri, got)
            ok = m["finite"] and m["cosine"] > 0.9999 and tail["cosine"] > 0.9999
            flag = "" if ok else "  <-- FAIL"
            bad += bool(flag)
            label = "-" if kernel == "triton" else BF16_CVT[cvt]
            tcos = "       (self)" if kernel == "triton" else f"{t['cosine']:>15.9f}"
            tabs = "" if kernel == "triton" else f"{t['max_abs']:>10.6f}"
            print(
                f"{kernel:<11}{label:<6}{m['cosine']:>13.9f}{m['max_abs']:>10.6f}"
                f"{tail['cosine']:>13.9f}{tcos:>15}{tabs:>10}{flag}"
            )
    print(f"\n{'FAILURES: %d' % bad if bad else 'all kernels within tolerance'}")
    return bad


def cmd_kernel_names(args, dev):
    """Print the GPU kernel actually dispatched, read back from the profiler.

    The ASM names are also derivable statically (kernel_identity), but the
    Triton name is JIT-generated and its tile/XCD parameters are baked into the
    symbol -- the only honest way to report it is to launch and look.
    """
    case = Case("names", tuple(args.segments), args.heads, args.head_dim)
    scale = 1.0 / math.sqrt(case.head_dim)
    q, k, v, cu = make_inputs(case, dev)
    from torch.profiler import ProfilerActivity, profile

    for kernel in KERNELS:
        for cvt in sorted(BF16_CVT):
            if kernel == "triton" and cvt != 1:
                continue
            ident = kernel_identity(kernel, cvt)
            RUNNERS[kernel](q, k, v, cu, case, scale, cvt)  # warm/JIT
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                RUNNERS[kernel](q, k, v, cu, case, scale, cvt)
                torch.cuda.synchronize()
            launched = [
                (e.key, e.device_time)
                for e in prof.key_averages()
                if e.device_time > 0 and "memcpy" not in e.key.lower()
            ]
            launched.sort(key=lambda x: -x[1])
            print(f"\n[{kernel} / {BF16_CVT[cvt] if kernel != 'triton' else '-'}]")
            print(f"  entry   {ident['entry']}")
            print(f"  binary  {ident['binary']}")
            for name, us in launched[:3]:
                print(f"  launched  {us / 1e3:8.2f} ms  {name}")
    return 0


def cmd_bench(args, dev):
    case = Case("bench", tuple(args.segments), heads=args.heads, head_dim=args.head_dim)
    scale = 1.0 / math.sqrt(case.head_dim)
    q, k, v, cu = make_inputs(case, dev)
    tflop = case.flops() / 1e12
    print(f"segments={case.segments} heads={case.heads} dim={case.head_dim} "
          f"total={case.total} flops={tflop:.2f} TFLOP")
    print(f"{'kernel':<11}{'cvt':<6}{'median_ms':>11}{'min_ms':>10}{'max_ms':>10}{'TFLOPS':>10}")
    for kernel in KERNELS:
        for cvt in sorted(BF16_CVT):
            if kernel == "triton" and cvt != 1:
                continue
            fn = lambda: RUNNERS[kernel](q, k, v, cu, case, scale, cvt)  # noqa: E731
            t = bench(fn, args.warmup, args.iters)
            label = "-" if kernel == "triton" else BF16_CVT[cvt]
            print(
                f"{kernel:<11}{label:<6}{t['median']:>11.2f}{t['min']:>10.2f}"
                f"{t['max']:>10.2f}{tflop / (t['median'] / 1e3):>10.1f}"
            )
    return 0


def cmd_sweep_balance(args, dev):
    """Hold total tokens fixed, vary where the pack is cut.

    A kernel whose grid is sized batch * ceil(max_seqlen / BLOCK) * heads pays
    for the tail segment as if it were as long as the head segment. Such a
    kernel gets *faster* as the split becomes balanced, even though the real
    FLOPs drop by 2x going the other way. A kernel that sizes work per segment
    tracks the FLOPs curve instead. This sweep separates the two.
    """
    total = args.total
    print(f"total={total} heads={args.heads} dim={args.head_dim}  (median ms)")
    print(f"{'cut':>8}{'tail':>8}{'TFLOP':>9}{'triton':>10}{'asm_group':>11}{'asm_split':>11}")
    for cut in args.cuts:
        if not 0 < cut < total:
            continue
        case = Case(f"cut{cut}", (cut, total - cut), args.heads, args.head_dim)
        scale = 1.0 / math.sqrt(case.head_dim)
        q, k, v, cu = make_inputs(case, dev)
        row = {}
        for kernel in KERNELS:
            fn = lambda: RUNNERS[kernel](q, k, v, cu, case, scale, args.cvt)  # noqa: E731
            row[kernel] = bench(fn, args.warmup, args.iters)["median"]
        print(
            f"{cut:>8}{total - cut:>8}{case.flops() / 1e12:>9.2f}"
            f"{row['triton']:>10.2f}{row['asm_group']:>11.2f}{row['asm_split']:>11.2f}"
        )
        del q, k, v
        torch.cuda.empty_cache()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=os.environ.get("H3_TEST_DEVICE", "cuda:0"))
    p.add_argument("--check", action="store_true", help="correctness check")
    p.add_argument("--ref", choices=sorted(REFERENCES), default="sdpa",
                   help="primary reference: segment-wise bf16 SDPA (default) or fp32")
    p.add_argument("--bench", action="store_true", help="per-kernel timing")
    p.add_argument("--sweep-balance", action="store_true", help="pack-imbalance sweep")
    p.add_argument("--kernel-names", action="store_true",
                   help="print the GPU kernel each label dispatches to")
    p.add_argument("--segments", type=int, nargs="+", default=list(H3_SEGMENTS))
    p.add_argument("--heads", type=int, default=H3_HEADS)
    p.add_argument("--head-dim", type=int, default=H3_HEAD_DIM)
    p.add_argument("--cvt", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--total", type=int, default=63232)
    p.add_argument(
        "--cuts",
        type=int,
        nargs="+",
        default=[31616, 47424, 55328, 59280, 61248, 62240, 62736, 63225],
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    dev = resolve_device(args.device)
    print(f"[device] {args.device} -> {torch.cuda.get_device_name(dev)} "
          f"{device_clocks(dev.index)}")

    if not (args.check or args.bench or args.sweep_balance or args.kernel_names):
        args.check = args.bench = True

    rc = 0
    if args.kernel_names:
        print("\n=== dispatched kernels ===")
        rc |= cmd_kernel_names(args, dev)
    if args.check:
        print(f"\n=== correctness (primary reference = {args.ref}) ===")
        rc |= cmd_check(args, dev)
    if args.bench:
        print("\n=== performance ===")
        rc |= cmd_bench(args, dev)
    if args.sweep_balance:
        print("\n=== pack-balance sweep ===")
        rc |= cmd_sweep_balance(args, dev)
    raise SystemExit(1 if rc else 0)


if __name__ == "__main__":
    main()
