#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

gpu="${GPU_ID:-6}"
stamp="$(date +%Y%m%d-%H%M%S)"
result="${RESULT_DIR:-$PWD/mfma-coissue-${stamp}}"
cache="${PYHIP_CACHE_DIR:-/tmp/pyhip-mfma-coissue-${stamp}}"
import_root="${PYHIP_IMPORT_ROOT:-/tmp/pyhip-mfma-coissue-import}"
counters='SQ_VALU_MFMA_COEXEC_CYCLES,SQ_VALU_MFMA_BUSY_CYCLES,SQ_INSTS_MFMA,SQ_INSTS_VALU,SQ_CYCLES'

if [[ -e "$result" ]]; then
    echo "Result path already exists: $result" >&2
    exit 1
fi

if pgrep -f 'rocprofv3|rocprof-compute|mfma-coissue.py' >/dev/null; then
    echo "A profiler or co-issue probe is already running" >&2
    exit 1
fi

gpu_line="$(rocm-smi --showuse --showmemuse 2>/dev/null)"
gpu_use="$(printf '%s\n' "$gpu_line" | sed -n "s/^GPU\[$gpu\].*GPU use (%): \([0-9][0-9]*\).*/\1/p" | head -n1)"
vram_use="$(printf '%s\n' "$gpu_line" | sed -n "s/^GPU\[$gpu\].*VRAM%): \([0-9][0-9]*\).*/\1/p" | head -n1)"
if [[ -z "$gpu_use" || -z "$vram_use" ]]; then
    echo "Could not parse utilization for physical GPU $gpu" >&2
    exit 1
fi
if [[ "$gpu_use" != 0 || "$vram_use" != 0 ]]; then
    echo "Physical GPU $gpu is not idle: gpu_use=$gpu_use vram_use=$vram_use" >&2
    exit 1
fi

mkdir -p "$result/controls" "$result/gap-scan" "$result/summary" "$cache" "$import_root"
ln -sfn "$PWD/src" "$import_root/pyhip"

run_profile() {
    local out=$1
    local kind=$2
    local gap=$3
    local repeats=$4
    local dispatches=$5
    mkdir -p "$out"
    HIP_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$import_root" \
    PYHIP_CACHE_DIR="$cache" \
    rocprofv3 \
        --pmc "$counters" \
        --kernel-trace \
        --kernel-include-regex '.*mfma_valu_coissue.*' \
        --output-directory "$out" \
        --output-file trace \
        --output-format csv \
        -- \
        /usr/bin/python archive/gemm/mfma-coissue.py \
            --kind "$kind" \
            --gap "$gap" \
            --period 64 \
            --repeats "$repeats" \
            --grid 256 \
            --dispatches "$dispatches" \
            --warmup 1 \
            --device 0 \
        > "$out/run.log" 2>&1
}

for kind in none add fmac fmac2 pk_fma; do
    run_profile "$result/controls/${kind}-g0" "$kind" 0 1024 3
done

for kind in fmac2 pk_fma; do
    for gap in 2 4 6 8 9 10 12; do
        run_profile "$result/gap-scan/${kind}-g${gap}" "$kind" "$gap" 512 2
    done
done

/usr/bin/python archive/gemm/analyze-mfma-coissue.py \
    --micro-root "$result/controls" \
    --micro-root "$result/gap-scan" \
    --output-dir "$result/summary"

find "$result" -type f ! -name manifest.txt -printf '%P\t%s bytes\n' \
    | sort > "$result/manifest.txt"
echo "Result: $result"