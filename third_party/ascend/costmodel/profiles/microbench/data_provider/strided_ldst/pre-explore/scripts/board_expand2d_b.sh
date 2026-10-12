#!/usr/bin/env bash
# Reduced supplementary campaign -- must finish before 22:00.
#   A. R-scan  : fix n, sweep every (R,C) split  -> the per-row cost regularity
#   B. W-scan  : 2D SIMT with many num_warps     -> W had only 32 points each
# C (wide branch) and D (dense strides) are deferred: wide is already accurate
# (0.74% rank-1, 15.4% rank-2) and the stride curve is already good.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
while pgrep -f board_expand2d.sh > /dev/null; do sleep 60; done
echo "=== campaign A/B start $(date +%H:%M) ==="

OUT="$PRE/results/dtype_calibration_v11_expand2d_b/raw"
mkdir -p "$OUT"

A_SHAPES="2x8 8x2 16x1 2x16 8x4 16x2 32x1 32x2 64x1 2x64 64x2 2x128 4x64 64x4 8x64 64x8"
B_SHAPES="4x4 4x8 4x16 8x8 8x16 16x16 2x16 8x4 16x4 16x8"

run() {
  local OP=$1 PATHN=$2 DT=$3 W=$4 RS=$5 CS=$6 SH=$7 TAG=$8
  REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op $OP --path $PATHN --dtype $DT \
    --shapes $SH --row-strides $RS --inner-strides $CS --num-warps $W \
    --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
    --out "$OUT/2d_${OP}_${PATHN}_${DT}_W${W}_${TAG}.json" 2>&1 | grep -E "WROTE|Error" | tail -1
}

# --- A: R-scan (strides trimmed to 3 to fit the deadline) ---
for DT in int8 float16 float32 int64; do
  run store simt_only $DT 1 "2 4" "2 4 16" "$A_SHAPES" A_rs
  run store simt_only $DT 4 "2 4" "2 4 16" "$A_SHAPES" A_rs
  run load  simt_only $DT 1 "2 4" "2 4 16" "$A_SHAPES" A_rs
done
echo "PHASE_A_DONE $(date +%H:%M)"

# --- B: W sweep on 2D SIMT store (int8 + float16, W=2,8,16,32,64) ---
for DT in int8 float16; do
  for W in 2 8 16 32 64; do
    run store simt_only $DT $W "2 4" "2 4 8 16" "$B_SHAPES" B_ws
  done
done
echo "ALLDONE_CAMPAIGN_B $(date +%H:%M)"
