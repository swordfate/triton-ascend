#!/usr/bin/env bash
# DELTA of the final campaign: only cells that have no valid data yet.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v4}"
OUT="${OUT_DIR:-results/dtype_calibration_v4_final}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"

m1d() {  # op path dtype warps blocks strides tag
  local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6" tag="$7"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 1D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60; }

m2d() {  # op path dtype warps shapes rs cs tag
  local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7" tag="$8"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 2D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60; }

if [[ "$PASS" == "core1d" ]]; then
  m1d load simd int8 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simd int8 1 "16 32 64 128 256" "6 128 256" core
  m1d load simd float16 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simd float16 1 "16 32 64 128 256" "6 32 64 128 256" core
  m1d load simd float32 1 "4 8" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simd float32 1 "16 32 64 128 256 512" "6 16 32 64 128 256" core
  m1d load simd int64 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simd int64 1 "16 32 64 128 256" "6 128 256" core
  m1d load simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simt_only int8 4 "256" "2 3 6 32 64 128 256" core
  m1d load simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d load simt_only int64 4 "256" "3 6 8 16 32 64 128 256" core
  m1d store simd int8 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simd int8 1 "16 32 64 128 256" "6 128 256" core
  m1d store simd float16 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simd float16 1 "16 32 64 128 256" "6 32 64 128 256" core
  m1d store simd float32 1 "4 8 16 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simd float32 1 "32 64 128 256" "" core
  m1d store simd int64 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simd int64 1 "16 32 64 128 256" "6 128 256" core
  m1d store simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simt_only int8 4 "256" "2 3 6 32 64 128 256" core
  m1d store simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simt_only float32 4 "4 8 16 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simt_only float32 4 "32 64 128 256" "" core
  m1d store simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" core
  m1d store simt_only int64 4 "256" "3 6 8 16 32 64 128 256" core
fi
if [[ "$PASS" == "wsweep1d" ]]; then
  m1d load simt_only int8 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int8 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int8 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int8 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int8 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int8 64 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only float16 64 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d load simt_only int64 64 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int8 64 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float16 64 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only float32 1 "4 16" "2 4 16 256" wsweep
  m1d store simt_only float32 2 "4 16" "2 4 16 256" wsweep
  m1d store simt_only float32 8 "4 16" "2 4 16 256" wsweep
  m1d store simt_only float32 16 "4 16" "2 4 16 256" wsweep
  m1d store simt_only float32 32 "4 16" "2 4 16 256" wsweep
  m1d store simt_only float32 64 "4 16" "2 4 16 256" wsweep
  m1d store simt_only int64 1 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int64 2 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int64 8 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int64 16 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int64 32 "4 16 64 256" "2 4 16 256" wsweep
  m1d store simt_only int64 64 "4 16 64 256" "2 4 16 256" wsweep
fi
if [[ "$PASS" == "d2" ]]; then
  m2d load simd int8 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only int8 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only int8 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int8 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int8 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int8 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int8 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int8 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simd int8 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int8 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simd float16 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only float16 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only float16 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float16 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float16 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float16 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float16 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float16 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simd float16 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float16 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simd float32 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only float32 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only float32 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float32 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float32 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float32 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float32 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only float32 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simd float32 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only float32 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simd int64 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only int64 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d load simt_only int64 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int64 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int64 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int64 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int64 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simt_only int64 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d load simd int64 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d load simt_only int64 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simd int8 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only int8 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only int8 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int8 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int8 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int8 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int8 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int8 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simd int8 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int8 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simd float16 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only float16 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only float16 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float16 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float16 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float16 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float16 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float16 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simd float16 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float16 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simd float32 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only float32 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only float32 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float32 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float32 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float32 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float32 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only float32 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simd float32 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only float32 64 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simd int64 1 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only int64 4 "4x4 4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 3 4 8 16 64" core
  m2d store simt_only int64 1 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int64 2 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int64 8 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int64 16 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int64 32 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simt_only int64 64 "4x8 16x16" "2 8" "2 64" wsweep
  m2d store simd int64 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 1 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 2 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 4 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 8 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 16 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 32 "4x1 16x1" "2 4 8 16" "1" n1
  m2d store simt_only int64 64 "4x1 16x1" "2 4 8 16" "1" n1
fi
echo "===== final delta done ($(date -Is)) ====="
