import argparse
import os
import statistics

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("PYHIP_DEBUG_LOG", "")

import pyhip
import torch

THREADS = 64
PATTERNS = {
    "balanced_unique": 0,
    "broadcast_4": 1,
    "same_bank_stride128": 2,
}
VOID_POINTER = "void*"


@pyhip.jit(no_pass=["pass_dse", "pass_dce"])
def measure_lds_pattern(J, pattern, repeats, output: VOID_POINTER):
    lds_base = J.alloc_lds(16 * 1024, align=16)
    lane = J.lane_id

    # 0: 64 unique contiguous 16B atoms (balanced, 1024B/wave).
    # 1: four 16B broadcast groups selected by lane[5:4] (A-read-like).
    # 2: 64 distinct addresses with 128B stride (same four banks per lane).
    address = J.gpr("vu32")
    if pattern == 0:
        address[0] = lds_base + lane[0] * 16
    elif pattern == 1:
        address[0] = lds_base + (lane[0] >> 4) * 16
    else:
        assert pattern == 2
        address[0] = lds_base + lane[0] * 128

    seed = J.gpr(4, "vu32", 0x12345678, align=4)
    value = J.gpr(4, "vu32", align=4)
    sink = J.gpr("vu32", 0)

    # Initialize every address used by the measurement.
    J.ds_write_b128(address, seed)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_barrier()

    start = J.gpr(2, "su32", align=2)
    stop = J.gpr(2, "su32", align=2)
    J.s_memtime(start)

    loop = J.gpr("su32", 0)
    with J.While(loop[0] < repeats):
        J.ds_read_b128(value, address)
        J.s_waitcnt(mod="lgkmcnt(0)")
        sink[0] ^= value[0]
        loop[0] += 1

    J.s_memtime(stop)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_sub_u32(stop[0], stop[0], start[0])
    J.s_subb_u32(stop[1], stop[1], start[1])
    J.s_store_dwordx2(stop, output, 0, mod="glc")
    sink_scalar = J.gpr("su32")
    J.v_readfirstlane_b32(sink_scalar, sink[0])
    J.s_store_dword(sink_scalar, output, 8, mod="glc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        choices=("all", *PATTERNS),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    output = torch.zeros(2, dtype=torch.uint64)
    names = PATTERNS if args.pattern == "all" else (args.pattern,)

    for name in names:
        pattern = PATTERNS[name]
        samples = []
        elapsed_samples = []
        for _ in range(args.samples):
            output.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            measure_lds_pattern(
                [1], [THREADS], pattern, args.repeats, output.data_ptr()
            )
            end.record()
            torch.cuda.synchronize()
            samples.append(int(output[0].item()))
            elapsed_samples.append(start.elapsed_time(end))
        median = statistics.median(samples)
        elapsed_median = statistics.median(elapsed_samples)
        print(
            name,
            f"median_cycles={median}",
            f"cycles_per_ds_read={median / args.repeats:.6f}",
            f"event_ms={elapsed_median:.6f}",
            f"effective_ghz={median / (elapsed_median * 1e6):.6f}",
            f"min_max={min(samples)}..{max(samples)}",
            f"samples={args.samples}",
        )


if __name__ == "__main__":
    main()
