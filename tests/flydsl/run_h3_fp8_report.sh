#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

output_dir=${ATTN_FP8_OUTPUT_DIR:-$root/artifacts/h3-fp8-auto}
mkdir -p "$output_dir"

ATTN_FP8_OUTPUT_DIR="$output_dir" bash tests/flydsl/run_h3_fp8_auto.sh
ATTN_FP8_OUTPUT_DIR="$output_dir" bash tests/flydsl/run_h3_aiter_fp8_auto.sh

(
    cd "$output_dir"
    files=(
        profile-auto-fp8.json profile-auto-fp8.log
        analysis-auto-fp8.json analysis-auto-fp8.log
        profile-auto-aiter-mi308-fp8.json profile-auto-aiter-mi308-fp8.log
        analysis-auto-aiter-mi308-fp8.json analysis-auto-aiter-mi308-fp8.log
        profile-auto-aiter-mi300-fp8.json profile-auto-aiter-mi300-fp8.log
        analysis-auto-aiter-mi300-fp8.json analysis-auto-aiter-mi300-fp8.log
    )
    if [[ -f fp8-vs-bf16.json && -f fp8-vs-bf16.log ]]; then
        files+=(fp8-vs-bf16.json fp8-vs-bf16.log)
    fi
    for correctness in \
        correctness-fp8.log \
        aiter-fp8-probe-mi308.log \
        aiter-fp8-probe-mi300.log; do
        if [[ -f "$correctness" ]]; then
            files+=("$correctness")
        fi
    done
    sha256sum "${files[@]}" > SHA256SUMS
)