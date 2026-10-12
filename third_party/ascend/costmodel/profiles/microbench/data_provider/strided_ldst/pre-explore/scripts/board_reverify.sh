#!/usr/bin/env bash
# 严格复验 4x16 vs 8x8：多种 rotate-mask / step / reps，看结论是否稳定
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/reverify
run() {
  local DT=$1 SH=$2 RS=$3 CS=$4 MASK=$5 STEP=$6 REPS=$7
  REPS=$REPS MAX_ATTEMPTS=5 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op load --path simd --dtype $DT --shapes $SH --row-strides $RS --inner-strides $CS \
    --num-warps 1 --rotate-step $STEP --rotate-mask $MASK --reps $REPS --max-attempts 5 \
    --out runs/reverify/${DT}_${SH}_rs${RS}_cs${CS}_m${MASK}_s${STEP}.out 2>&1 | grep -E "WROTE" | tail -1
}
for MASK in 63 255 1023 16383; do
  for STEP in 8192 65536; do
    for CFG in "float16 4x16 2 64" "float16 8x8 2 64" "float16 8x8 8 64" "float16 4x16 8 64"; do
      set -- $CFG
      echo -n "mask=$MASK step=$STEP $2: "
      run $1 $2 $3 $4 $MASK $STEP 5
    done
  done
done
echo "ALLDONE"
