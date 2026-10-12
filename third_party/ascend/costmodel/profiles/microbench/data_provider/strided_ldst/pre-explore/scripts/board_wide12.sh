#!/usr/bin/env bash
# Controlled stride=1 vs stride=2 measurement for the SIMD wide branch.
#   stride=1 : contiguous DMA of n*eb bytes, no extraction
#   stride=2 : DMA of ~2n*eb contiguous bytes, then extract_slice of n elements
# Choosing (stride=1, n=2k) against (stride=2, n=k) makes the SPAN equal, which
# separates the transfer cost from the extraction cost.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/results/dtype_calibration_v12_wide12/raw"
mkdir -p "$OUT"
BLOCKS="4 8 16 32 64 128 256 512 1024"
run() {
  local DT=$1 OP=$2 PATHN=$3 W=$4 STR=$5
  REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op $OP --path $PATHN --dtype $DT --blocks $BLOCKS --strides $STR --num-warps $W \
    --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
    --out "$OUT/1d_${OP}_${PATHN}_${DT}_W${W}_s${STR}.json" 2>&1 | grep -E "WROTE|Error" | tail -1
}
for DT in int8 float16 float32 int64; do
  run $DT load simd 1 "1 2"
  run $DT store simd 1 "1 2"
done
echo "ALLDONE_WIDE12 $(date +%H:%M)"
