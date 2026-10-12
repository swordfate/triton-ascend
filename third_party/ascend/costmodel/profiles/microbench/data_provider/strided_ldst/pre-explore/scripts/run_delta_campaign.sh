#!/usr/bin/env bash
# DELTA campaign: measure ONLY the grid cells not already measured.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_delta}"
OUT="${OUT_DIR:-results/dtype_calibration_v3_delta}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"

measure() {  # <op> <path> <dtype> <warps> <blocks> <strides> <tag>
  local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6" tag="$7"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} W=${warps} n=$(echo $blocks|wc -w)x$(echo $strides|wc -w) $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}

measure load simd int8 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simd int8 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 40 48 96 128 192 256" part
measure load simd float16 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simd float16 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 32 40 48 64 96 128 192 256" part
measure load simd float32 1 "4 8" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simd float32 1 "16 32 64 128 256 512" "5 6 7 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" part
measure load simd int64 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simd int64 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 40 48 96 128 192 256" part
measure load simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simt_only int8 4 "256" "2 3 5 6 7 9 10 11 12 20 24 28 32 40 48 64 96 128 192 256" part
measure load simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simt_only float32 4 "4 8 16 32 64 128 256 512" "28" part
measure load simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure load simt_only int64 4 "256" "3 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" part
measure store simd int8 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simd int8 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 40 48 96 128 192 256" part
measure store simd float16 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simd float16 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 32 40 48 64 96 128 192 256" part
measure store simd float32 1 "4 8 16 32 128 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simd float32 1 "64 256" "5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" part
measure store simd int64 1 "4 8 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simd int64 1 "16 32 64 128 256" "6 7 9 10 11 12 20 24 28 40 48 96 128 192 256" part
measure store simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simt_only int8 4 "256" "2 3 5 6 7 9 10 11 12 20 24 28 32 40 48 64 96 128 192 256" part
measure store simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simt_only float32 4 "4 8 16 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simt_only float32 4 "32 64 128 256" "28" part
measure store simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" full
measure store simt_only int64 4 "256" "3 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256" part
echo "===== delta campaign done ($(date -Is)) ====="
