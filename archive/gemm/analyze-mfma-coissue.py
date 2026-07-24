import argparse
import collections
import csv
import glob
import json
import os
import statistics


COUNTERS = (
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_CYCLES",
)


def profile_rows(directory, kernel_substring):
    paths = glob.glob(os.path.join(directory, "*counter_collection.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one counter CSV in {directory}, got {paths}")

    dispatches = collections.defaultdict(dict)
    durations = {}
    with open(paths[0], newline="") as handle:
        for row in csv.DictReader(handle):
            if kernel_substring not in row["Kernel_Name"]:
                continue
            dispatch_id = int(row["Dispatch_Id"])
            dispatches[dispatch_id][row["Counter_Name"]] = float(
                row["Counter_Value"]
            )
            durations[dispatch_id] = (
                int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            ) / 1000

    for dispatch_id, values in dispatches.items():
        missing = set(COUNTERS) - values.keys()
        if missing:
            raise RuntimeError(f"dispatch {dispatch_id} is missing {sorted(missing)}")
    return list(dispatches.values()), list(durations.values())


def median_counter(dispatches, counter):
    return statistics.median(row[counter] for row in dispatches)


def load_runner_result(directory):
    path = os.path.join(directory, "run.log")
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.startswith("{")]
    if len(records) != 1:
        raise RuntimeError(f"expected one JSON record in {path}, got {len(records)}")
    return records[0]


def micro_row(directory):
    name = os.path.basename(directory)
    kind, gap = name.rsplit("-g", 1)
    dispatches, durations = profile_rows(directory, "mfma_valu_coissue")
    runner = load_runner_result(directory)
    mfma = median_counter(dispatches, "SQ_INSTS_MFMA")
    busy = median_counter(dispatches, "SQ_VALU_MFMA_BUSY_CYCLES")
    coexec = median_counter(dispatches, "SQ_VALU_MFMA_COEXEC_CYCLES")
    valu = median_counter(dispatches, "SQ_INSTS_VALU")
    sq_cycles = median_counter(dispatches, "SQ_CYCLES")
    mfma_count = runner["repeat_count"] * runner["mfma_per_repeat"]
    return {
        "kind": kind,
        "gap_wait_states": int(gap),
        "profiled_dispatches": len(dispatches),
        "kernel_time_median_us": statistics.median(durations),
        "mfma_count": mfma,
        "valu_count": valu,
        "mfma_busy_cycles_per_mfma": busy / mfma,
        "coexec_cycles_per_mfma": coexec / mfma,
        "coexec_over_busy_percent": 100 * coexec / busy,
        "sq_cycles_per_mfma": sq_cycles / mfma,
        "device_cycles_per_mfma": runner["device_cycles_median"] / mfma_count,
    }


def production_rows(root):
    rows = []
    for directory in sorted(glob.glob(os.path.join(root, "profiles", "r*-*"))):
        name = os.path.basename(directory)
        round_text, variant = name.split("-", 1)
        dispatches, durations = profile_rows(directory, "moe_gemm_8wave_g1u1")
        mfma = median_counter(dispatches, "SQ_INSTS_MFMA")
        busy = median_counter(dispatches, "SQ_VALU_MFMA_BUSY_CYCLES")
        coexec = median_counter(dispatches, "SQ_VALU_MFMA_COEXEC_CYCLES")
        valu = median_counter(dispatches, "SQ_INSTS_VALU")
        sq_cycles = median_counter(dispatches, "SQ_CYCLES")
        rows.append(
            {
                "round": int(round_text[1:]),
                "variant": variant,
                "profiled_dispatches": len(dispatches),
                "kernel_time_mean_us": statistics.mean(durations),
                "kernel_time_median_us": statistics.median(durations),
                "mfma_count": mfma,
                "valu_count": valu,
                "mfma_busy_cycles_per_mfma": busy / mfma,
                "coexec_cycles_per_mfma": coexec / mfma,
                "coexec_over_busy_percent": 100 * coexec / busy,
                "sq_cycles_per_mfma": sq_cycles / mfma,
            }
        )
    return rows


def production_combined_rows(root):
    rows = []
    for variant in ("scalar", "packed"):
        dispatches = []
        durations = []
        for directory in sorted(
            glob.glob(os.path.join(root, "profiles", f"r*-{variant}"))
        ):
            profile_dispatches, profile_durations = profile_rows(
                directory, "moe_gemm_8wave_g1u1"
            )
            dispatches.extend(profile_dispatches)
            durations.extend(profile_durations)

        if not dispatches:
            continue
        mfma = median_counter(dispatches, "SQ_INSTS_MFMA")
        busy = median_counter(dispatches, "SQ_VALU_MFMA_BUSY_CYCLES")
        coexec = median_counter(dispatches, "SQ_VALU_MFMA_COEXEC_CYCLES")
        valu = median_counter(dispatches, "SQ_INSTS_VALU")
        sq_cycles = median_counter(dispatches, "SQ_CYCLES")
        rows.append(
            {
                "variant": variant,
                "profiled_dispatches": len(dispatches),
                "kernel_time_mean_us": statistics.mean(durations),
                "kernel_time_median_us": statistics.median(durations),
                "mfma_count": mfma,
                "valu_count": valu,
                "mfma_busy_cycles_per_mfma": busy / mfma,
                "coexec_cycles_per_mfma": coexec / mfma,
                "coexec_over_busy_percent": 100 * coexec / busy,
                "sq_cycles_per_mfma": sq_cycles / mfma,
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-root", action="append", default=[])
    parser.add_argument("--production-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    micro = []
    for root in args.micro_root:
        for directory in glob.glob(os.path.join(root, "*-g*")):
            micro.append(micro_row(directory))
    micro.sort(key=lambda row: (row["kind"], row["gap_wait_states"]))
    write_csv(os.path.join(args.output_dir, "microbenchmark.csv"), micro)

    comparisons = []
    indexed = {(row["kind"], row["gap_wait_states"]): row for row in micro}
    gaps = sorted(
        gap
        for kind, gap in indexed
        if kind == "pk_fma" and ("fmac2", gap) in indexed
    )
    for gap in gaps:
        scalar = indexed["fmac2", gap]
        packed = indexed["pk_fma", gap]
        comparisons.append(
            {
                "gap_wait_states": gap,
                "packed_minus_two_fmac_device_cycles_per_mfma": (
                    packed["device_cycles_per_mfma"]
                    - scalar["device_cycles_per_mfma"]
                ),
                "two_fmac_coexec_cycles_per_mfma": scalar[
                    "coexec_cycles_per_mfma"
                ],
                "packed_coexec_cycles_per_mfma": packed[
                    "coexec_cycles_per_mfma"
                ],
            }
        )
    write_csv(os.path.join(args.output_dir, "microbenchmark-comparison.csv"), comparisons)

    if args.production_root:
        write_csv(
            os.path.join(args.output_dir, "production.csv"),
            production_rows(args.production_root),
        )
        write_csv(
            os.path.join(args.output_dir, "production-combined.csv"),
            production_combined_rows(args.production_root),
        )

    print(
        json.dumps(
            {
                "microbenchmark_rows": len(micro),
                "comparison_rows": len(comparisons),
                "production_root": args.production_root,
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()