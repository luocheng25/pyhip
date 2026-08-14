import argparse
import os
import statistics

os.environ.setdefault("PYHIP_JIT_LOG", "0")
os.environ.setdefault("PYHIP_DEBUG_LOG", "")

import pyhip
import torch

VOID_POINTER = "void*"


@pyhip.jit(no_pass=["pass_dse", "pass_dce"])
def measure_fp8_mfma(J, chains, repeats, output: VOID_POINTER):
    operand_a = J.gpr(2, "vu32", 0x40404040, align=2)
    operand_b = J.gpr(2, "vu32", 0x40404040, align=2)
    accumulators = J.gpr(chains, 4, "vf32", align=4)
    accumulators[...] = 0.0

    start = J.gpr(2, "su32", align=2)
    stop = J.gpr(2, "su32", align=2)
    J.s_memtime(start)

    loop = J.gpr("su32", 0)
    with J.While(loop[0] < repeats):
        for chain in range(chains):
            J.v_mfma_f32_16x16x32_fp8_fp8(
                accumulators[chain], operand_a, operand_b, accumulators[chain]
            )
        loop[0] += 1

    J.s_memtime(stop)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_sub_u32(stop[0], stop[0], start[0])
    J.s_subb_u32(stop[1], stop[1], start[1])
    J.s_store_dwordx2(stop, output, 0, mod="glc")

    sink = J.gpr("vf32")
    sink[0] = accumulators[0, 0]
    sink_scalar = J.gpr("su32")
    J.v_readfirstlane_b32(sink_scalar, sink[0])
    J.s_store_dword(sink_scalar, output, 8, mod="glc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    output = torch.zeros(2, dtype=torch.uint64)
    for chains in args.chains:
        samples = []
        elapsed_samples = []
        for _ in range(args.samples):
            output.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            measure_fp8_mfma([1], [64], chains, args.repeats, output.data_ptr())
            end.record()
            torch.cuda.synchronize()
            samples.append(int(output[0].item()))
            elapsed_samples.append(start.elapsed_time(end))
        median = statistics.median(samples)
        elapsed_median = statistics.median(elapsed_samples)
        instruction_count = args.repeats * chains
        print(
            f"chains={chains}",
            f"median_cycles={median}",
            f"cycles_per_mfma={median / instruction_count:.6f}",
            f"event_ms={elapsed_median:.6f}",
            f"effective_ghz={median / (elapsed_median * 1e6):.6f}",
            f"min_max={min(samples)}..{max(samples)}",
            f"samples={args.samples}",
        )


if __name__ == "__main__":
    main()
