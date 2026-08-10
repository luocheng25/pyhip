#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

gpu=${GPU:-4}
output_dir=${ATTN_FP8_OUTPUT_DIR:-$root/artifacts/h3-fp8-auto}
active=/sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_fp8_group.co
mi300=/sgl-workspace/aiter/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8_group.co
backup="$output_dir/fwd_hd128_fp8_group.mi308.original.co"
mi308_sha=5a9cfe058a455734e8ac46e740f250631b0396eb785df8e5ab2b8df2ceacbe2e
mi300_sha=5e5b4b6891c600a0051ca0ebb3c14f415be7db8f3aa9607a18f057e244d65575

mkdir -p "$output_dir"

sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

restore() {
    cp --preserve=mode,timestamps "$backup" "$active" 2>/dev/null || true
}
trap restore EXIT

[[ "$(sha256 "$active")" == "$mi308_sha" ]]
[[ "$(sha256 "$mi300")" == "$mi300_sha" ]]
if [[ ! -f "$backup" ]]; then
    cp --preserve=mode,timestamps "$active" "$backup"
fi
[[ "$(sha256 "$backup")" == "$mi308_sha" ]]

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=triton_fp8,asm_mi308_fp8 \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-aiter-mi308-fp8.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal-aiter-mi308-fp8" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-aiter-mi308-fp8.log"

cp --preserve=mode,timestamps "$mi300" "$active"
[[ "$(sha256 "$active")" == "$mi300_sha" ]]

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=asm_mi300_fp8 \
ATTN_PROFILE_MI308_FP8_REFERENCE="$backup" \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-aiter-mi300-fp8.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal-aiter-mi300-fp8" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-aiter-mi300-fp8.log"

restore
trap - EXIT
[[ "$(sha256 "$active")" == "$mi308_sha" ]]

python3 tests/flydsl/analyze_h3_attention_throttle.py \
    "$output_dir/profile-auto-aiter-mi308-fp8.json" \
    --no-samples \
    --analysis-json "$output_dir/analysis-auto-aiter-mi308-fp8.json" \
    | tee "$output_dir/analysis-auto-aiter-mi308-fp8.log"

python3 tests/flydsl/analyze_h3_attention_throttle.py \
    "$output_dir/profile-auto-aiter-mi300-fp8.json" \
    --no-samples \
    --analysis-json "$output_dir/analysis-auto-aiter-mi300-fp8.json" \
    | tee "$output_dir/analysis-auto-aiter-mi300-fp8.log"