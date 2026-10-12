#!/usr/bin/env bash
# stride=1 vs stride=2 for the SIMD wide branch.
# The shipped probe refuses stride==1 by design (it is a strided-access probe),
# so we run an unmodified copy with only that guard relaxed.  The original file
# is left untouched.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/results/dtype_calibration_v12_wide12/raw"
mkdir -p "$OUT"
# local copy with the guard relaxed
sed 's/if stride < 2:/if stride < 1:/' scripts/dtype_stride_measure_probe.py > scripts/_wide12_probe.py
grep -n "if stride < 1:" scripts/_wide12_probe.py | head -1
BLOCKS="4 8 16 32 64 128 256 512 1024"
run() {
  local DT=$1 OP=$2 PATHN=$3 W=$4 STR=$5
  REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_wide12_probe.py \
    --op $OP --path $PATHN --dtype $DT --blocks $BLOCKS --strides $STR --num-warps $W \
    --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
    --out "$OUT/1d_${OP}_${PATHN}_${DT}_W${W}_s${STR// /_}.json" 2>&1 | grep -E "WROTE|Error|error" | tail -1
}
for DT in int8 float16 float32 int64; do
  run $DT load simd 1 "1 2"
done
echo "PHASE1_DONE $(date +%H:%M)"
for DT in int8 float16 float32 int64; do
  run $DT store simd 1 "1 2"
done
echo "ALLDONE_WIDE12B $(date +%H:%M)"
