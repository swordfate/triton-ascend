#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/repro2d
CFG=(
 "float16 8x8 4 64" "float16 4x4 16 64" "float16 4x16 3 64"
 "float32 4x4 8 16" "float32 4x4 4 16" "float16 8x8 4 16"
 "float16 8x8 3 16" "float16 8x8 8 16"
)
i=0
for c in "${CFG[@]}"; do
  set -- $c
  i=$((i+1))
  for rep in 1 2 3 4 5; do
    REPS=4 MAX_ATTEMPTS=3 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
      --op load --path simd --dtype $1 --shapes $2 --row-strides $3 --inner-strides $4 \
      --num-warps 1 --rotate-step 8192 --rotate-mask 255 --reps 4 --max-attempts 3 \
      --out runs/repro2d/c${i}_r${rep}.out > /dev/null 2>&1
    echo "cfg$i rep$rep done"
  done
done
echo REPRODONE
