#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(1100|1200|1300|1400|1500|1600|1700|1800)$ ]]; then
    echo "usage: $0 {1100|1200|1300|1400|1500|1600|1700|1800}" >&2
    exit 2
fi

frequency=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

gpu=${GPU:-4}
python=${PYTHON:-python3}
flydsl_python=${FLYDSL_PYTHON:-artifacts/h3-five-kernel/venv-flydsl030/bin/python}
output_dir=${ATTN_FREQUENCY_OUTPUT_DIR:-$root/artifacts/h3-frequency-sweep}
active=${AITER_MI308_RTNA_GROUP:-/sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_rtna_group.co}
mi300=${AITER_MI300_RTNA_GROUP:-/sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtna_group.co}
backup="$output_dir/fwd_hd128_bf16_rtna_group.mi308.original.co"
mi308_sha=3687c5610a454572e4a615ec58f05e707fdf3995e4dc932cf2219ad2fa0052ff
mi300_sha=f8d7e1dfc5301edeb83e5520e8d710798c7641a52040c33dbed77c18115813c5

mkdir -p "$output_dir"

sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

cleanup() {
    cp --preserve=mode,timestamps "$backup" "$active" 2>/dev/null || true
    sudo -n amd-smi reset -g "$gpu" -d >/dev/null 2>&1 || true
}
trap cleanup EXIT

[[ "$(sha256 "$active")" == "$mi308_sha" ]]
[[ "$(sha256 "$mi300")" == "$mi300_sha" ]]
if [[ ! -f "$backup" ]]; then
    cp --preserve=mode,timestamps "$active" "$backup"
fi
[[ "$(sha256 "$backup")" == "$mi308_sha" ]]

sudo -n amd-smi set -g "$gpu" -d "$frequency" | tee "$output_dir/set-${frequency}.log"

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=triton,asm_mi308 \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1 \
ATTN_PROFILE_REQUESTED_SCLK_MHZ="$frequency" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-${frequency}-aiter-mi308.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-${frequency}-aiter-mi308" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-${frequency}-aiter-mi308.log"

cp --preserve=mode,timestamps "$mi300" "$active"
[[ "$(sha256 "$active")" == "$mi300_sha" ]]

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=asm_mi300 \
ATTN_PROFILE_MI308_REFERENCE="$backup" \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1 \
ATTN_PROFILE_REQUESTED_SCLK_MHZ="$frequency" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-${frequency}-aiter-mi300.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-${frequency}-aiter-mi300" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-${frequency}-aiter-mi300.log"

cp --preserve=mode,timestamps "$backup" "$active"
[[ "$(sha256 "$active")" == "$mi308_sha" ]]

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=8wave_32x32,4wave_varlen \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_ALLOW_NON_AUTO_DPM=1 \
ATTN_PROFILE_REQUESTED_SCLK_MHZ="$frequency" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-${frequency}-flydsl.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-${frequency}-flydsl" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$flydsl_python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-${frequency}-flydsl.log"

cleanup
trap - EXIT

[[ "$(sha256 "$active")" == "$mi308_sha" ]]
[[ "$(cat /sys/bus/pci/devices/$(GPU="$gpu" "$python" - <<'PY'
import json
import os
import subprocess

devices = json.loads(subprocess.check_output(["amd-smi", "list", "--json"], text=True))
print(next(device["bdf"] for device in devices if device["gpu"] == int(os.environ["GPU"])))
PY
)/power_dpm_force_performance_level)" == "auto" ]]

echo "completed frequency=${frequency}MHz"