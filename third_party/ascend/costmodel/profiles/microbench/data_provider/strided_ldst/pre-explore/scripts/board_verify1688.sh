#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/verify1688
for reps in 4 8; do
echo "===== reps=$reps ====="
REPS=$reps MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
  --op load --path simd --dtype float16 \
  --shapes 4x16 8x8 --row-strides 2 --inner-strides 64 --num-warps 1 \
  --rotate-step 8192 --rotate-mask 16383 --reps $reps --max-attempts 4 \
  --out runs/verify1688/v_$reps.json 2>&1 | grep -E "^2d_|WROTE|Error|error" | tail -6
done
sleep 15
echo "===== 第三次（1D 对照）====="
REPS=6 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
  --op load --path simd --dtype float16 --blocks 64 --strides 64 --num-warps 1 \
  --rotate-step 8192 --rotate-mask 16383 --reps 6 --max-attempts 4 \
  --out runs/verify1688/r1d.json 2>&1 | grep -E "^load|WROTE" | tail -3
