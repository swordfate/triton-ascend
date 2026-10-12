#!/usr/bin/env bash
# Expanded 2D shape grid, focused on the weak slots.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/results/dtype_calibration_v10_expand2d/raw"
mkdir -p "$OUT"
SHAPES="1x16 1x32 1x64 2x8 2x16 2x32 8x4 16x4 16x8 8x32 16x32 32x8 32x16 4x32 2x64"
RS="2 4"
CS="2 4 8 16"
run() {
  local OP=$1 PATHN=$2 DT=$3 W=$4 TAG=$5
  REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op $OP --path $PATHN --dtype $DT \
    --shapes $SHAPES --row-strides $RS --inner-strides $CS --num-warps $W \
    --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
    --out "$OUT/2d_${OP}_${PATHN}_${DT}_W${W}_${TAG}.json" 2>&1 | grep -E "WROTE|Error" | tail -1
}
for DT in int8 float16 float32 int64; do
  run store simt_only $DT 1 "expand"
  run store simt_only $DT 4 "expand"
done
echo "PHASE1_DONE"
for DT in int8 float16 float32 int64; do
  run load simd $DT 1 "expand"
done
echo "PHASE2_DONE"
for DT in int8 float16 float32 int64; do
  run store simd $DT 1 "expand"
  run load simt_only $DT 1 "expand"
done
echo "ALLDONE_EXPAND2D"
