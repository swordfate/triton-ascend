#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/rot2d
CASES=(
  "float16 8x8 2 64"
  "float16 8x8 8 64"
  "float16 4x16 2 64"
  "float16 4x8 16 64"
  "float16 8x8 8 16"
)
for c in "${CASES[@]}"; do
  set -- $c
  DT=$1; SH=$2; RS=$3; CS=$4
  for STEP in 8192 65536 524288; do
    r=$(REPS=4 MAX_ATTEMPTS=3 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
      --op load --path simd --dtype $DT \
      --shapes $SH --row-strides $RS --inner-strides $CS --num-warps 1 \
      --rotate-step $STEP --rotate-mask 255 --reps 4 --max-attempts 3 \
      --out runs/rot2d/${DT}_${SH}_rs${RS}_cs${CS}_st${STEP}.json 2>&1 | grep -E "WROTE" | tail -1)
    echo "  dt=$DT sh=$SH rs=$RS cs=$CS step=$STEP -> $r"
  done
done
