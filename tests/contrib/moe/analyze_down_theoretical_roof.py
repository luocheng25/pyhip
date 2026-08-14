#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


USEFUL_FLOPS = 2 * 32768 * 4 * 384 * 6144
MFMA_FLOPS = 2 * 16 * 16 * 32
SIMDS = 80 * 4
NAIVE_BYTES = 6_543_114_240
MFMA_PER_N256_WAVE = 192
POSTPROCESS_MUL_PER_N256_WAVE = 64
POSTPROCESS_FMA_PER_N256_WAVE = 64
POSTPROCESS_PERM_PER_N256_WAVE = 32
ARCH_MFMA_CYCLES = 16.0


def read_counter_csv(paths, kernel, dispatch_id):
    values = defaultdict(list)
    for path in paths:
        matching_rows = []
        with open(path) as csv_file:
            for row in csv.DictReader(csv_file):
                if kernel not in row["Kernel_Name"]:
                    continue
                matching_rows.append(row)
        if not matching_rows:
            continue
        selected_dispatch = (
            max(int(row["Dispatch_Id"]) for row in matching_rows)
            if dispatch_id == "latest"
            else int(dispatch_id)
        )
        selected_rows = [
            row for row in matching_rows if int(row["Dispatch_Id"]) == selected_dispatch
        ]
        if not selected_rows:
            raise ValueError(f"dispatch {selected_dispatch} did not match {kernel!r} in {path}")
        first_row = selected_rows[0]
        duration_ms = (
            int(first_row["End_Timestamp"]) - int(first_row["Start_Timestamp"])
        ) / 1e6
        print(
            f"PMC_FILE path={path} dispatch={selected_dispatch}",
            f"duration_ms={duration_ms:.6f}",
            f"vgpr={first_row['VGPR_Count']}",
            f"agpr={first_row['Accum_VGPR_Count']}",
            f"lds={first_row['LDS_Block_Size']}",
        )
        for row in selected_rows:
            values[row["Counter_Name"]].append(float(row["Counter_Value"]))
    if not values:
        raise ValueError(f"no counters matched kernel {kernel!r} in {paths}")
    return {name: statistics.median(samples) for name, samples in values.items()}


def classify_instruction(assembly):
    assembly = assembly.strip()
    if assembly.startswith("v_mfma"):
        return "MFMA"
    if assembly.startswith(("buffer_load", "global_load", "flat_load")):
        return "VMEM-load"
    if assembly.startswith(
        ("buffer_store", "global_store", "flat_store", "global_atomic")
    ):
        return "VMEM-store"
    if assembly.startswith("ds_"):
        return "LDS"
    if assembly.startswith("s_waitcnt"):
        has_vmem = "vmcnt" in assembly
        has_lds = "lgkmcnt" in assembly
        if has_vmem and has_lds:
            return "VMEM/LDS-wait"
        if has_vmem:
            return "VMEM-wait"
        if has_lds:
            return "LDS-wait"
    if assembly.startswith("s_barrier"):
        return "barrier"
    if assembly.startswith("v_"):
        return "VALU"
    return "other"


def read_att(path):
    with open(path) as json_file:
        rows = json.load(json_file)["code"]
    mfma = [row for row in rows if str(row[0]).strip().startswith("v_mfma")]
    stall_by_class = defaultdict(int)
    cycles_by_class = defaultdict(int)
    exec_by_class = defaultdict(int)
    for row in rows:
        instruction_class = classify_instruction(str(row[0]))
        exec_by_class[instruction_class] += row[6]
        cycles_by_class[instruction_class] += row[7]
        stall_by_class[instruction_class] += row[8]
    return {
        "instruction_count": len(rows),
        "mfma_static": len(mfma),
        "mfma_exec": sum(row[6] for row in mfma),
        "mfma_cycles": sum(row[7] for row in mfma),
        "mfma_stall": sum(row[8] for row in mfma),
        "total_cycles": sum(row[7] for row in rows),
        "total_stall": sum(row[8] for row in rows),
        "exec_by_class": dict(exec_by_class),
        "cycles_by_class": dict(cycles_by_class),
        "stall_by_class": dict(stall_by_class),
    }


def intersection_bounds(left_count, right_count, universe_count):
    return (
        max(0.0, left_count + right_count - universe_count),
        min(left_count, right_count),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-ms", type=float, required=True)
    parser.add_argument("--effective-ghz", type=float, required=True)
    parser.add_argument("--mfma-cycles", type=float, default=17.613525)
    parser.add_argument(
        "--postprocess-mul", type=int, default=POSTPROCESS_MUL_PER_N256_WAVE
    )
    parser.add_argument(
        "--postprocess-fma", type=int, default=POSTPROCESS_FMA_PER_N256_WAVE
    )
    parser.add_argument(
        "--postprocess-perm", type=int, default=POSTPROCESS_PERM_PER_N256_WAVE
    )
    parser.add_argument("--mul-cycles", type=float, default=4.0)
    parser.add_argument("--fma-cycles", type=float, default=4.0)
    parser.add_argument("--perm-cycles", type=float, default=4.0)
    parser.add_argument("--hbm-peak-tbps", type=float, default=5.3)
    parser.add_argument("--kernel", default="moe_2stage_down_prefill_1x4_0")
    parser.add_argument("--dispatch-id", default="latest")
    parser.add_argument("--att-code-json")
    parser.add_argument("--sq-csv", nargs="+")
    parser.add_argument("--l2-csv", nargs="+")
    parser.add_argument("--hbm-csv", nargs="+")
    args = parser.parse_args()

    architecture_roof = (
        MFMA_FLOPS / ARCH_MFMA_CYCLES * SIMDS * args.effective_ghz * 1e9 / 1e12
    )
    measured_mfma_roof = (
        MFMA_FLOPS / args.mfma_cycles * SIMDS * args.effective_ghz * 1e9 / 1e12
    )
    serialized_postprocess_cycles = (
        args.postprocess_mul * args.mul_cycles
        + args.postprocess_fma * args.fma_cycles
        + args.postprocess_perm * args.perm_cycles
    )
    architecture_mfma_cycles = MFMA_PER_N256_WAVE * ARCH_MFMA_CYCLES
    measured_mfma_cycles = MFMA_PER_N256_WAVE * args.mfma_cycles
    architecture_schedule_efficiency = architecture_mfma_cycles / (
        architecture_mfma_cycles + serialized_postprocess_cycles
    )
    measured_schedule_efficiency = measured_mfma_cycles / (
        measured_mfma_cycles + serialized_postprocess_cycles
    )
    architecture_schedule_roof = architecture_roof * architecture_schedule_efficiency
    measured_schedule_roof = measured_mfma_roof * measured_schedule_efficiency
    useful_tflops = USEFUL_FLOPS / (args.kernel_ms * 1e9)
    arithmetic_intensity = USEFUL_FLOPS / NAIVE_BYTES
    all_hbm_roof = arithmetic_intensity * args.hbm_peak_tbps
    print(f"useful_tflops={useful_tflops:.3f}")
    print(f"serialized_postprocess_cycles={serialized_postprocess_cycles:.3f}")
    print(f"architecture_mfma_roof_tflops={architecture_roof:.3f}")
    print(f"architecture_mfma_efficiency={useful_tflops / architecture_roof:.4%}")
    print(f"measured_mfma_roof_tflops={measured_mfma_roof:.3f}")
    print(f"measured_mfma_efficiency={useful_tflops / measured_mfma_roof:.4%}")
    print(f"architecture_schedule_roof_tflops={architecture_schedule_roof:.3f}")
    print(f"architecture_schedule_efficiency={useful_tflops / architecture_schedule_roof:.4%}")
    print(f"measured_schedule_roof_tflops={measured_schedule_roof:.3f}")
    print(f"measured_schedule_efficiency={useful_tflops / measured_schedule_roof:.4%}")
    print(f"naive_arithmetic_intensity_flop_per_byte={arithmetic_intensity:.3f}")
    print(f"all_hbm_traffic_roof_tflops={all_hbm_roof:.3f}")
    print(f"naive_l2_side_tbps={NAIVE_BYTES / (args.kernel_ms * 1e-3) / 1e12:.3f}")

    if args.att_code_json:
        att = read_att(Path(args.att_code_json))
        print(
            "ATT",
            {
                key: value
                for key, value in att.items()
                if not key.endswith("_by_class")
            },
        )
        if att["mfma_exec"]:
            print(
                "att_mfma_cycles_per_exec="
                f"{att['mfma_cycles'] / att['mfma_exec']:.6f}"
            )
            print(
                "att_mfma_stall_per_exec="
                f"{att['mfma_stall'] / att['mfma_exec']:.6f}"
            )
        for instruction_class, stall in sorted(
            att["stall_by_class"].items(), key=lambda item: item[1], reverse=True
        ):
            print(
                f"att_stall_{instruction_class}={stall}",
                f"share={stall / att['total_stall']:.4%}",
                f"exec={att['exec_by_class'][instruction_class]}",
                f"cycles={att['cycles_by_class'][instruction_class]}",
            )

    for label, paths in (
        ("SQ", args.sq_csv),
        ("L2", args.l2_csv),
        ("HBM", args.hbm_csv),
    ):
        if paths:
            counters = read_counter_csv(paths, args.kernel, args.dispatch_id)
            print(label, counters)
            if label == "SQ":
                if "SQ_WAVES" in counters:
                    waves = counters["SQ_WAVES"]
                    for counter_name in (
                        "SQ_INSTS_MFMA",
                        "SQ_INSTS_VALU",
                        "SQ_INSTS_VMEM",
                        "SQ_INSTS_LDS",
                        "SQ_WAVE_CYCLES",
                    ):
                        if counter_name in counters:
                            print(
                                f"{counter_name.lower()}_per_wave="
                                f"{counters[counter_name] / waves:.6f}"
                            )
                if (
                    "SQ_VALU_MFMA_BUSY_CYCLES" in counters
                    and "SQ_INSTS_MFMA" in counters
                ):
                    print(
                        "sq_mfma_busy_cycles_per_inst="
                        f"{counters['SQ_VALU_MFMA_BUSY_CYCLES'] / counters['SQ_INSTS_MFMA']:.6f}"
                    )
            if label == "L2":
                denominator = counters["TCC_HIT_sum"] + counters["TCC_MISS_sum"]
                print(f"l2_hit_rate={counters['TCC_HIT_sum'] / denominator:.4%}")
            if label == "HBM":
                read_requests = counters["TCC_EA0_RDREQ_DRAM_sum"]
                write_requests = counters["TCC_EA0_WRREQ_DRAM_sum"]
                if "TCC_EA0_RDREQ_sum" in counters:
                    read_32b_min, read_32b_max = intersection_bounds(
                        counters["TCC_EA0_RDREQ_32B_sum"],
                        read_requests,
                        counters["TCC_EA0_RDREQ_sum"],
                    )
                else:
                    read_32b_min, read_32b_max = 0.0, read_requests
                if "TCC_EA0_WRREQ_sum" in counters:
                    write_64b_min, write_64b_max = intersection_bounds(
                        counters["TCC_EA0_WRREQ_64B_sum"],
                        write_requests,
                        counters["TCC_EA0_WRREQ_sum"],
                    )
                else:
                    write_64b_min, write_64b_max = 0.0, write_requests

                read_bytes_min = read_32b_max * 32 + (read_requests - read_32b_max) * 64
                read_bytes_max = read_32b_min * 32 + (read_requests - read_32b_min) * 64
                write_bytes_min = write_64b_min * 64 + (write_requests - write_64b_min) * 32
                write_bytes_max = write_64b_max * 64 + (write_requests - write_64b_max) * 32
                hbm_bytes_min = read_bytes_min + write_bytes_min
                hbm_bytes_max = read_bytes_max + write_bytes_max
                seconds = args.kernel_ms * 1e-3
                hbm_tbps_min = hbm_bytes_min / seconds / 1e12
                hbm_tbps_max = hbm_bytes_max / seconds / 1e12
                print(
                    "estimated_hbm_read_gb_range="
                    f"{read_bytes_min / 1e9:.3f}..{read_bytes_max / 1e9:.3f}"
                )
                print(
                    "estimated_hbm_write_gb_range="
                    f"{write_bytes_min / 1e9:.3f}..{write_bytes_max / 1e9:.3f}"
                )
                print(f"estimated_hbm_tbps_range={hbm_tbps_min:.3f}..{hbm_tbps_max:.3f}")
                print(
                    "estimated_hbm_peak_fraction_range="
                    f"{hbm_tbps_min / args.hbm_peak_tbps:.4%}.."
                    f"{hbm_tbps_max / args.hbm_peak_tbps:.4%}"
                )
                print(
                    "measured_hbm_arithmetic_intensity_range="
                    f"{USEFUL_FLOPS / hbm_bytes_max:.3f}.."
                    f"{USEFUL_FLOPS / hbm_bytes_min:.3f}"
                )


if __name__ == "__main__":
    main()
