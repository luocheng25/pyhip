#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
source tests/flydsl/h3_profile_common.sh

gpu=${GPU:-4}
aiter_python=${AITER_PYTHON:-/opt/venv/bin/python3}
flydsl_python=${FLYDSL_PYTHON:-artifacts/h3-five-kernel/venv-flydsl030/bin/python}
aiter_root=${AITER_ROOT:-/sgl-workspace/aiter}
decoder_dir=${ROCPROF_ATT_LIBRARY_PATH:-/opt/rocm/lib}
trace_root=${H3_ATT_TRACE_ROOT:-/tmp/h3-fp8-att-repro}
output_dir=${H3_ATT_OUTPUT_DIR:-$root/artifacts/h3-fp8-att}

active=$aiter_root/hsa/gfx942/fmha_v3_fwd/MI308/fwd_hd128_fp8_group.co
mi300=$aiter_root/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8_group.co
backup="$trace_root/fwd_hd128_fp8_group.mi308.original.co"
mi308_sha=5a9cfe058a455734e8ac46e740f250631b0396eb785df8e5ab2b8df2ceacbe2e
mi300_sha=5e5b4b6891c600a0051ca0ebb3c14f415be7db8f3aa9607a18f057e244d65575

mkdir -p "$trace_root" "$output_dir"

sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

restore() {
    cp --preserve=mode,timestamps "$backup" "$active" 2>/dev/null || true
}
trap restore EXIT

command -v rocprofv3 >/dev/null
[[ -x "$aiter_python" ]]
[[ -x "$flydsl_python" ]]
[[ -f "$decoder_dir/librocprof-trace-decoder.so" ]]
h3_require_idle_gpu "$gpu" "$aiter_python"
[[ "$(sha256 "$active")" == "$mi308_sha" ]]
[[ "$(sha256 "$mi300")" == "$mi300_sha" ]]
cp --preserve=mode,timestamps "$active" "$backup"

write_config() {
    local config=$1
    local regex=$2
    local trace_dir=$3
    local simd_select=$4
    local buffer_size=$5
    {
        printf '%s\n' 'jobs:'
        printf '%s\n' '  -'
        printf '    kernel_include_regex: "%s"\n' "$regex"
        printf '%s\n' '    kernel_iteration_range: "[1]"'
        printf '%s\n' '    output_file: out'
        printf '    output_directory: "%s"\n' "$trace_dir"
        printf '%s\n' '    output_format: [json]'
        printf '%s\n' '    truncate_kernels: true'
        printf '%s\n' '    sys_trace: false'
        printf '%s\n' '    advanced_thread_trace: true'
        printf '%s\n' '    att_target_cu: 0'
        printf '%s\n' '    att_shader_engine_mask: "0x1"'
        printf '    att_simd_select: "%s"\n' "$simd_select"
        printf '    att_buffer_size: "%s"\n' "$buffer_size"
    } > "$config"
}

capture() {
    local label=$1
    local regex=$2
    local impl=$3
    local python=$4
    local simd_select=$5
    local buffer_size=$6
    local trace_dir="$trace_root/$label"
    local config="$trace_root/$label.yaml"
    local log="$trace_root/$label-capture.log"

    rm -rf "$trace_dir"
    write_config "$config" "$regex" "$trace_dir" "$simd_select" "$buffer_size"
    HIP_VISIBLE_DEVICES="$gpu" \
    H3_ATT_IMPL="$impl" \
    PYTHONPYCACHEPREFIX="$trace_root/pycache-$label" \
    FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    rocprofv3 -i "$config" --att-library-path "$decoder_dir" -- \
        "$python" -B tests/flydsl/run_h3_fp8_att_target.py \
        2>&1 | tee "$log"
    if grep -Eiq 'Stitch Incomplete|Wave incomplete|cutoff|parser mismatch' "$log"; then
        echo "incomplete ATT capture: $label" >&2
        exit 1
    fi
}

capture triton '.*_attn_fwd.*' triton_fp8 "$aiter_python" 0xf 0x7f000000
capture asm-mi308 '.*fmha_fwd_hd128_fp8_group.*' asm_fp8 "$aiter_python" 0xf 0x7f000000

cp --preserve=mode,timestamps "$mi300" "$active"
[[ "$(sha256 "$active")" == "$mi300_sha" ]]
capture asm-mi300 '.*fmha_fwd_hd128_fp8_group.*' asm_fp8 "$aiter_python" 0xf 0x7f000000
restore
[[ "$(sha256 "$active")" == "$mi308_sha" ]]

capture flydsl-8wave '.*attn_kernel.*' flydsl_8wave_fp8 "$flydsl_python" 0x1 0x20000000
capture flydsl-4wave '.*attention_kernel.*' flydsl_4wave_fp8 "$flydsl_python" 0x1 0x20000000

"$aiter_python" tests/flydsl/analyze_h3_fp8_att.py \
    "$trace_root/triton" \
    "$trace_root/asm-mi308" \
    "$trace_root/asm-mi300" \
    "$trace_root/flydsl-8wave" \
    "$trace_root/flydsl-4wave" \
    --labels Triton ASM-MI308 ASM-MI300 FlyDSL-8wave FlyDSL-4wave \
    --output "$output_dir/analysis.json" \
    | tee "$output_dir/analysis.log"

(
    cd "$output_dir"
    sha256sum analysis.json analysis.log > SHA256SUMS
)