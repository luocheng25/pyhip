#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
source tests/flydsl/h3_profile_common.sh

gpu=${GPU:-4}
aiter_python=${AITER_PYTHON:-/opt/venv/bin/python3}
flydsl_python=${FLYDSL_PYTHON:-artifacts/h3-five-kernel/venv-flydsl030/bin/python}
aiter_root=${AITER_ROOT:-/sgl-workspace/aiter}
output_dir=${ATTN_PROFILE_OUTPUT_DIR:-$root/artifacts/h3-five-kernel}
warmup=${ATTN_PROFILE_WARMUP:-3}
iters=${ATTN_PROFILE_ITERS:-70}
sensor_interval_ms=${ATTN_PROFILE_SENSOR_INTERVAL_MS:-10}

active=$aiter_root/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_bf16_rtna_group.co
mi300=$aiter_root/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtna_group.co
backup="$output_dir/fwd_hd128_bf16_rtna_group.mi308.original.co"
mi308_sha=3687c5610a454572e4a615ec58f05e707fdf3995e4dc932cf2219ad2fa0052ff
mi300_sha=f8d7e1dfc5301edeb83e5520e8d710798c7641a52040c33dbed77c18115813c5

mkdir -p "$output_dir"

sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "missing executable: $1" >&2
        exit 1
    fi
}

restore() {
    cp --preserve=mode,timestamps "$backup" "$active" 2>/dev/null || true
}
trap restore EXIT

require_executable "$aiter_python"
require_executable "$flydsl_python"
h3_require_idle_gpu "$gpu" "$aiter_python"
[[ "$(sha256 "$active")" == "$mi308_sha" ]]
[[ "$(sha256 "$mi300")" == "$mi300_sha" ]]
if [[ ! -f "$backup" ]]; then
    cp --preserve=mode,timestamps "$active" "$backup"
fi
[[ "$(sha256 "$backup")" == "$mi308_sha" ]]

if [[ ${H3_SKIP_CORRECTNESS:-0} != 1 ]]; then
    HIP_VISIBLE_DEVICES="$gpu" \
    PYTHONPYCACHEPREFIX="$output_dir/pycache-correctness-aiter-mi308" \
    "$aiter_python" -B tests/flydsl/pa_4wave/h3_attn_kernel_test.py --check \
        2>&1 | tee "$output_dir/correctness-aiter-mi308.log"
fi

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=triton,asm_mi308 \
ATTN_PROFILE_WARMUP="$warmup" \
ATTN_PROFILE_ITERS="$iters" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-aiter-mi308.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS="$sensor_interval_ms" \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal-aiter-mi308" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$aiter_python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-aiter-mi308.log"

cp --preserve=mode,timestamps "$mi300" "$active"
[[ "$(sha256 "$active")" == "$mi300_sha" ]]

if [[ ${H3_SKIP_CORRECTNESS:-0} != 1 ]]; then
    HIP_VISIBLE_DEVICES="$gpu" \
    PYTHONPYCACHEPREFIX="$output_dir/pycache-correctness-aiter-mi300" \
    "$aiter_python" -B tests/flydsl/pa_4wave/h3_attn_kernel_test.py --check \
        2>&1 | tee "$output_dir/correctness-aiter-mi300.log"
fi

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=asm_mi300 \
ATTN_PROFILE_MI308_REFERENCE="$backup" \
ATTN_PROFILE_WARMUP="$warmup" \
ATTN_PROFILE_ITERS="$iters" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-aiter-mi300.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS="$sensor_interval_ms" \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal-aiter-mi300" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$aiter_python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-aiter-mi300.log"

restore
trap - EXIT
[[ "$(sha256 "$active")" == "$mi308_sha" ]]

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=8wave_32x32,4wave_varlen \
ATTN_PROFILE_WARMUP="$warmup" \
ATTN_PROFILE_ITERS="$iters" \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-flydsl.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS="$sensor_interval_ms" \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal-flydsl" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$flydsl_python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-flydsl.log"

for name in aiter-mi308 aiter-mi300 flydsl; do
    "$aiter_python" tests/flydsl/analyze_h3_attention_throttle.py \
        "$output_dir/profile-auto-$name.json" \
        --no-samples \
        --analysis-json "$output_dir/analysis-auto-$name.json" \
        | tee "$output_dir/analysis-auto-$name.log"
done

(
    cd "$output_dir"
    files=(
        profile-auto-aiter-mi308.json profile-auto-aiter-mi308.log \
        profile-auto-aiter-mi300.json profile-auto-aiter-mi300.log \
        profile-auto-flydsl.json profile-auto-flydsl.log \
        analysis-auto-aiter-mi308.json analysis-auto-aiter-mi308.log \
        analysis-auto-aiter-mi300.json analysis-auto-aiter-mi300.log \
        analysis-auto-flydsl.json analysis-auto-flydsl.log
    )
    for correctness in correctness-aiter-mi308.log correctness-aiter-mi300.log; do
        if [[ -f "$correctness" ]]; then
            files+=("$correctness")
        fi
    done
    sha256sum "${files[@]}" > SHA256SUMS
)