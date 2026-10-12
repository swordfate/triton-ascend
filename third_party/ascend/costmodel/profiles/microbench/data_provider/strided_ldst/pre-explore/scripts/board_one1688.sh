#!/usr/bin/env bash
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/verify1688
REPS=4 MAX_ATTEMPTS=3 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
  --op load --path simd --dtype float16 \
  --shapes "4x16 8x8" --row-strides 2 --inner-strides 64 --num-warps 1 \
  --rotate-step 8192 --rotate-mask 16383 --reps 4 --max-attempts 3 \
  --out runs/verify1688/v1.json 2>&1 | tail -25
