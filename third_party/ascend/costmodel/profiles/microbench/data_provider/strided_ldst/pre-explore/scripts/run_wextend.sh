#!/usr/bin/env bash
# W-EXTENSION: give a chosen num_warps the SAME full grid that num_warps=4 already has.
# W in {2,8,16,32,64} keep the reduced grid (num_warps=2 was dropped by request).
#   1D full grid = 8 elements x 10 strides = 80
#   2D full grid = 6 shapes x 4 rs x 5 cs = 120
#
# W (env, default 1) selects the num_warps column of every invocation below.
# W=2 reproduces the historical run_wextend_w2.sh, which used to be a
# byte-identical copy of this file differing only in that column.
# Usage:  W=2 bash scripts/run_wextend.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v9}"
OUT="${OUT_DIR:-results/dtype_calibration_v9_wsweep_full}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
W="${W:-1}"
WIN="--rotate-step 8192 --rotate-mask 16383"

m1d() { local op="$1" dtype="$2" warps="$3" blocks="$4" strides="$5"
  local sig="$(echo $blocks|tr ' ' '-')_$(echo $strides|tr ' ' '-')"
  local outfile="$OUT/raw/${op}_simt_only_${dtype}_W${warps}_wfull_${sig}.json"
  echo "----- 1D ${op} simt_only ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op $op --path simt_only --dtype $dtype --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m2d() { local op="$1" dtype="$2" warps="$3" shapes="$4" rs="$5" cs="$6"
  local sig="$(echo $shapes|tr ' x' '--')_$(echo $rs|tr ' ' '-')_$(echo $cs|tr ' ' '-')"
  local outfile="$OUT/raw/2d_${op}_simt_only_${dtype}_W${warps}_wfull_${sig}.json"
  echo "----- 2D ${op} simt_only ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op $op --path simt_only --dtype $dtype --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m1d load int8 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d load float16 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d load float32 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d load int64 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d store int8 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d store float16 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d store float32 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d store int64 $W "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m2d load int8 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d load float16 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d load float32 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d load int64 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d store int8 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d store float16 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d store float32 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d store int64 $W "16x16 4x16 4x4 4x8 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
echo "===== W extension done ($(date -Is)) ====="
