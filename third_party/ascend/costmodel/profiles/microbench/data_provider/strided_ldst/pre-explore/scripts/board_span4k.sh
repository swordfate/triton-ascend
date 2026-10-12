#!/usr/bin/env bash
# Controlled comparison at a constant ~4 KB span.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/results/dtype_calibration_v12_wide12/raw2"
mkdir -p "$OUT"
sed 's/if stride < 2:/if stride < 1:/' scripts/dtype_stride_measure_probe.py > scripts/_wide12_probe.py
run() {
  local DT=$1 B=$2 S=$3
  REPS=5 MAX_ATTEMPTS=5 bash scripts/board_env.sh python scripts/_wide12_probe.py \
    --op load --path simd --dtype $DT --blocks $B --strides $S --num-warps 1 \
    --rotate-step 8192 --rotate-mask 16383 --reps 5 --max-attempts 5 \
    --out "$OUT/span4k_${DT}_b${B}_s${S}.json" 2>&1 | grep -E "WROTE|Error" | tail -1
}
run float16 2048 1
run float16 1024 2
run float32 1024 1
run float32 512 2
run int64 512 1
run int64 256 2
run int8 4096 1
run int8 2048 2
echo "ALLDONE_SPAN4K $(date +%H:%M)"
