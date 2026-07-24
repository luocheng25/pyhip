import argparse
import json
import statistics

import pyhip
import torch


VALU_KINDS = ("none", "add", "fmac", "fmac2", "pk_fma")


def emit_delay(J, wait_states):
    while wait_states >= 16:
        J.s_nop(15)
        wait_states -= 16
    if wait_states:
        J.s_nop(wait_states - 1)


@pyhip.jit(no_pass=["pass_dse", "pass_dce"])
def mfma_valu_coissue(
    J: pyhip.JIT,
    valu_kind,
    gap_wait_states,
    period_wait_states,
    repeat_count,
    p_cycles: "void*",
):
    assert valu_kind in VALU_KINDS
    assert 0 <= gap_wait_states <= period_wait_states
    assert repeat_count > 0

    mfma_a = J.gpr(8, "vu32")
    mfma_b = J.gpr(8, "vu32")
    mfma_d = J.gpr(4, 4, "vf32", align=4)
    mfma_a[...] = 0x01010101
    mfma_b[...] = 0x01010101
    mfma_d[...] = 0

    scalar_src = J.gpr("vf32", 1.0)
    scalar_dst = J.gpr("vf32", 1.0)
    scalar_src2 = J.gpr("vf32", 1.0)
    scalar_dst2 = J.gpr("vf32", 1.0)
    packed_src0 = J.gpr(2, "vf32", align=2)
    packed_src1 = J.gpr(2, "vf32", align=2)
    packed_dst = J.gpr(2, "vf32", align=2)
    packed_src0[...] = 1.0
    packed_src1[...] = 1.0
    packed_dst[...] = 1.0

    p_cycles[:] += J.blockIdx.x[0] * 16

    start = J.gpr(2, "su32", align=2)
    J.s_memtime(start)

    loop = J.gpr("su32", 0)
    with J.While(loop[0] < repeat_count):
        for bank in range(4):
            J.v_mfma_f32_16x16x128_f8f6f4(
                mfma_d[bank], mfma_a, mfma_b, 0
            )
            emit_delay(J, gap_wait_states)

            if valu_kind == "add":
                J.v_add_f32(scalar_dst, scalar_dst, scalar_src)
            elif valu_kind in ("fmac", "fmac2"):
                J.v_fmac_f32(scalar_dst, scalar_src, scalar_src)
                if valu_kind == "fmac2":
                    J.v_fmac_f32(scalar_dst2, scalar_src2, scalar_src2)
            elif valu_kind == "pk_fma":
                J.v_pk_fma_f32(
                    packed_dst,
                    packed_src0,
                    packed_src1,
                    packed_dst,
                    mod="op_sel_hi:[1,0,1]",
                )

            emit_delay(J, period_wait_states - gap_wait_states)

        loop[0] += 1

    sink_v = J.gpr("vf32")
    J.v_add_f32(sink_v, mfma_d[0, 0], scalar_dst)
    if valu_kind == "fmac2":
        J.v_add_f32(sink_v, sink_v, scalar_dst2)
    if valu_kind == "pk_fma":
        J.v_add_f32(sink_v, sink_v, packed_dst[0])

    sink_s = J.gpr("su32")
    J.v_readfirstlane_b32(sink_s, sink_v)

    stop = J.gpr(2, "su32", align=2)
    J.s_memtime(stop)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_sub_u32(stop[0], stop[0], start[0])
    J.s_subb_u32(stop[1], stop[1], start[1])
    J.s_store_dwordx2(stop, p_cycles, 0, mod="glc")
    J.s_store_dword(sink_s, p_cycles, 8, mod="glc")


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(
        description="Measure gfx950 MFMA/VALU issue overlap with a fixed ISA sequence."
    )
    parser.add_argument("--kind", choices=VALU_KINDS, required=True)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--period", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4096)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--dispatches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.gap <= args.period:
        parser.error("--gap must be between zero and --period")

    torch.cuda.set_device(args.device)
    arch = torch.cuda.get_device_properties(args.device).gcnArchName
    if "gfx950" not in arch:
        raise RuntimeError(f"This probe requires gfx950, got {arch}")

    warmup_cycles = torch.zeros(args.grid * 2, dtype=torch.uint64, device="cuda")
    for _ in range(args.warmup):
        mfma_valu_coissue(
            [args.grid],
            [64],
            args.kind,
            args.gap,
            args.period,
            args.repeats,
            warmup_cycles.data_ptr(),
        )
    torch.cuda.synchronize()

    all_cycles = torch.zeros(
        (args.dispatches, args.grid, 2), dtype=torch.uint64, device="cuda"
    )
    stride_bytes = args.grid * 2 * all_cycles.element_size()
    for dispatch in range(args.dispatches):
        mfma_valu_coissue(
            [args.grid],
            [64],
            args.kind,
            args.gap,
            args.period,
            args.repeats,
            all_cycles.data_ptr() + dispatch * stride_bytes,
        )
    torch.cuda.synchronize()

    samples = all_cycles[:, :, 0].cpu().flatten().tolist()
    result = {
        "kernel": "mfma_valu_coissue",
        "arch": arch,
        "kind": args.kind,
        "gap_wait_states": args.gap,
        "period_wait_states": args.period,
        "repeat_count": args.repeats,
        "mfma_per_repeat": 4,
        "valu_per_repeat": {
            "none": 0,
            "add": 4,
            "fmac": 4,
            "fmac2": 8,
            "pk_fma": 4,
        }[args.kind],
        "grid_workgroups": args.grid,
        "workgroup_size": 64,
        "dispatches": args.dispatches,
        "device_cycle_samples": len(samples),
        "device_cycles_mean": statistics.mean(samples),
        "device_cycles_median": statistics.median(samples),
        "device_cycles_p25": percentile(samples, 0.25),
        "device_cycles_p75": percentile(samples, 0.75),
        "device_cycles_min": min(samples),
        "device_cycles_max": max(samples),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()