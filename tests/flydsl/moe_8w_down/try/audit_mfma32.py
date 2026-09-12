# SPDX-License-Identifier: MIT
"""Check actual MFMA32 scheduling, memory purity and resource metadata."""

import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isa", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    text = args.isa.read_text()
    memory = re.findall(r"MOE32_MEMORY_BEGIN_(\d+)(.*?)MOE32_MEMORY_END_\1", text, re.S)
    compute = re.findall(r"MOE32_COMPUTE_BEGIN_(\d+)(.*?)MOE32_COMPUTE_END_\1", text, re.S)
    assert memory and compute and len(memory) == len(compute)
    bad_memory, bad_compute, bad_gaps = [], [], []
    for index, (_, body) in enumerate(memory):
        bad = re.findall(r"^\s*(v_\w+|ds_write\w*)\b", body, re.M)
        if bad:
            bad_memory.append((index, bad))
    integer = re.compile(r"^\s*(?:v_(?:(?:and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|mbcnt|bit|brev|alignbit|cmp)\w*|(?:add|sub|mul|mad)\w*_(?:u|i)\d+\w*)|s_(?:add|sub|mul|and|or|xor|not|lshl|lshr|ashr|bfe|bfi|bcnt|brev)\w*)\b", re.M)
    for index, (_, body) in enumerate(compute):
        vector = re.findall(r"^\s*(v_\w+)\b", body, re.M)
        positions = [i for i, op in enumerate(vector) if op.startswith("v_mfma_")]
        assert len(positions) == 8 and all("32x32x64_f8f6f4" in vector[i] for i in positions)
        gaps = [b - a - 1 for a, b in zip(positions, positions[1:])]
        print(f"COMPUTE {index} gaps={gaps} tail_valu={len(vector) - positions[-1] - 1}")
        if index and gaps != [14] * 7:
            bad_gaps.append((index, gaps))
        if integer.search(body) or re.search(r"\bds_(read|write)\w*", body):
            bad_compute.append(index)
    metadata = re.findall(r"^\s*\.(?:vgpr_count|sgpr_count|group_segment_fixed_size|private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count):.*$", text, re.M)
    print("\n".join(metadata))
    print(f"AUDIT memory_valu={bad_memory} compute_address={bad_compute} bad_gaps={bad_gaps}")
    if args.strict:
        assert not bad_memory and not bad_compute and not bad_gaps
        assert re.search(r"\.private_segment_fixed_size:\s*0\b", text)
        assert re.search(r"\.vgpr_spill_count:\s*0\b", text)
        assert not re.search(r"^\s*scratch_\w+|\bv_accvgpr_\w+", text, re.M)
        instructions = "\n".join(re.findall(r"^\s*(?:v_|s_|ds_|buffer_|global_)\w+.*$", text, re.M))
        assert not re.search(r"\ba(?:\d+|\[\d+)(?:\b|:)", instructions)
        print("STRICT_PASS: Memory no VALU; Compute no addresses/LDS; 1 MFMA +14 VALU; no scratch/AGPR")


if __name__ == "__main__":
    main()