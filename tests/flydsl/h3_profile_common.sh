#!/usr/bin/env bash

h3_require_idle_gpu() {
    local gpu=$1
    local python=$2
    local max_vram_mib=${ATTN_PROFILE_MAX_INITIAL_VRAM_MIB:-1024}
    local bdf
    local process_count
    local busy
    local vram_used
    local dpm

    command -v amd-smi >/dev/null
    [[ -x "$python" ]]
    bdf=$(
        amd-smi list --json | "$python" -c '
import json
import sys

gpu = int(sys.argv[1])
devices = json.load(sys.stdin)
print(next(device["bdf"] for device in devices if device["gpu"] == gpu))
' "$gpu"
    )
    process_count=$(
        amd-smi process -g "$gpu" -G --json | "$python" -c '
import json
import sys

entries = json.load(sys.stdin)[0]["process_list"]
print(sum(isinstance(entry.get("process_info"), dict) for entry in entries))
'
    )
    busy=$(<"/sys/bus/pci/devices/$bdf/gpu_busy_percent")
    vram_used=$(<"/sys/bus/pci/devices/$bdf/mem_info_vram_used")
    dpm=$(<"/sys/bus/pci/devices/$bdf/power_dpm_force_performance_level")

    if (( process_count != 0 || busy != 0 || vram_used > max_vram_mib * 1024 * 1024 )) \
        || [[ "$dpm" != auto ]]; then
        printf 'GPU preflight failed before CUDA initialization: gpu=%s bdf=%s processes=%s busy=%s vram_mib=%s dpm=%s\n' \
            "$gpu" "$bdf" "$process_count" "$busy" "$((vram_used / 1024 / 1024))" "$dpm" >&2
        return 1
    fi
    printf 'preflight,physical_gpu=%s,bdf=%s,busy=0,vram_mib=%s,processes=0,dpm=auto\n' \
        "$gpu" "$bdf" "$((vram_used / 1024 / 1024))"
}
