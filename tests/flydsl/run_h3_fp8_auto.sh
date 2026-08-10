#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

gpu=${GPU:-4}
python=${FLYDSL_PYTHON:-artifacts/h3-five-kernel/venv-flydsl030/bin/python}
output_dir=${ATTN_FP8_OUTPUT_DIR:-$root/artifacts/h3-fp8-auto}

mkdir -p "$output_dir"

HIP_VISIBLE_DEVICES="$gpu" \
PYTHONPYCACHEPREFIX="$output_dir/pycache-correctness" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$python" -B tests/flydsl/check_h3_fp8_paged.py \
    2>&1 | tee "$output_dir/correctness-fp8.log"

HIP_VISIBLE_DEVICES="$gpu" \
ATTN_PROFILE_IMPLS=8wave_varlen_fp8,4wave_varlen_fp8 \
ATTN_PROFILE_WARMUP=3 \
ATTN_PROFILE_ITERS=70 \
ATTN_PROFILE_OUTPUT="$output_dir/profile-auto-fp8.json" \
ATTN_PROFILE_SENSOR_INTERVAL_MS=10 \
PYTHONPYCACHEPREFIX="$output_dir/pycache-formal" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
"$python" -B tests/flydsl/profile_h3_attention_throttle.py \
    2>&1 | tee "$output_dir/profile-auto-fp8.log"

"$python" tests/flydsl/analyze_h3_attention_throttle.py \
    "$output_dir/profile-auto-fp8.json" \
    --no-samples \
    --analysis-json "$output_dir/analysis-auto-fp8.json" \
    | tee "$output_dir/analysis-auto-fp8.log"

"$python" tests/flydsl/analyze_h3_fp8_auto.py \
    "$output_dir/profile-auto-fp8.json" \
    --bf16-profile artifacts/h3-five-kernel/profile-auto-flydsl.json \
    --output "$output_dir/fp8-vs-bf16.json" \
    | tee "$output_dir/fp8-vs-bf16.log"