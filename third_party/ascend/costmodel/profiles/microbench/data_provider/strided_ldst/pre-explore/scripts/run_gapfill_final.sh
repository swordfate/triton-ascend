#!/usr/bin/env bash
# GAP FILL: the last missing cells so every (slot,dtype,W) reaches its target grid.
# Gaps are exact rectangles, so each invocation measures precisely what is missing.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v8}"
OUT="${OUT_DIR:-results/dtype_calibration_v8_gapfill}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"

m1d() { local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6"
  local sig="$(echo $blocks|tr ' ' '-')_$(echo $strides|tr ' ' '-')"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_W${warps}_gap_${sig}.json"
  echo "----- 1D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op $op --path $path --dtype $dtype --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m2d() { local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7"
  local sig="$(echo $shapes|tr ' x' '--')_$(echo $rs|tr ' ' '-')_$(echo $cs|tr ' ' '-')"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_W${warps}_gap_${sig}.json"
  echo "----- 2D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op $op --path $path --dtype $dtype --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m1d load simd float16 1 "16 32 64 128 256" "2 3 4 8 16"
m1d load simd float32 1 "16 32 64 128 256 512" "2 3 4 8"
m1d load simd int64 1 "16 32 64 128 256" "2 3 4 8 16 32 64"
m1d load simd int8 1 "16 32 64 128 256" "2 3 4 8 16 32 64"
m1d load simt_only float32 1 "4 16 64 256" "2 4 16 256"
m1d load simt_only float32 16 "4 16 64 256" "2 4 16 256"
m1d load simt_only float32 2 "4 16 64 256" "2 4 16 256"
m1d load simt_only float32 32 "4 16 64 256" "2 4 16 256"
m1d load simt_only float32 4 "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256"
m1d load simt_only float32 64 "4 16 64 256" "2 4 16 256"
m1d load simt_only float32 8 "4 16 64 256" "2 4 16 256"
m1d load simt_only int64 4 "256" "2 4"
m1d load simt_only int8 4 "256" "4 8 16"
m1d store simd float16 1 "16 32 64 128 256" "2 3 4 8 16"
m1d store simd float32 1 "32 64 128 256" "2 3 4 6 8 16 32 64 128 256"
m1d store simd int64 1 "16 32 64 128 256" "2 3 4 8 16 32 64"
m1d store simd int8 1 "16 32 64 128 256" "2 3 4 8 16 32 64"
m1d store simt_only float32 1 "64 256" "2 4 16 256"
m1d store simt_only float32 16 "64 256" "2 4 16 256"
m1d store simt_only float32 2 "64 256" "2 4 16 256"
m1d store simt_only float32 32 "64 256" "2 4 16 256"
m1d store simt_only float32 4 "32 64 128 256" "2 3 4 6 8 16 32 64 128 256"
m1d store simt_only float32 64 "64 256" "2 4 16 256"
m1d store simt_only float32 8 "64 256" "2 4 16 256"
m1d store simt_only int64 4 "256" "2 4"
m1d store simt_only int8 4 "256" "4 8 16"
m2d load simd int8 1 "16x16 4x16 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d load simt_only int8 1 "16x16" "2 8" "2 16"
m2d load simt_only int8 16 "16x16" "2 8" "2 16"
m2d load simt_only int8 2 "16x16" "2 8" "2 16"
m2d load simt_only int8 32 "16x16" "2 8" "2 16"
m2d load simt_only int8 4 "16x16 4x16 8x16 8x8" "2 3 4 8" "2 3 4 8 16"
m2d load simt_only int8 64 "16x16" "2 8" "2 16"
m2d load simt_only int8 8 "16x16" "2 8" "2 16"
echo "===== gapfill done ($(date -Is)) ====="
