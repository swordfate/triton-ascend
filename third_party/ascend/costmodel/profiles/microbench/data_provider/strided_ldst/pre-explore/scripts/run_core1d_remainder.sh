#!/usr/bin/env bash
# core1d REMAINDER: the first run aborted because one generated
# invocation had an EMPTY --strides list (blocks with no missing stride were
# wrongly placed in the partial group).  Generator fixed; this covers only
# what is still missing.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v4}"
OUT="${OUT_DIR:-results/dtype_calibration_v4_final}"
mkdir -p "$OUT/raw" "$OUT/measure_asm"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
m1d() { local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6" tag="$7"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 1D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60; }

  m1d load simd int8 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simd float16 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simd float16 1 "16 32 64 128 256" "32 64 128 256" fix1
  m1d load simd float32 1 "4 8" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simd float32 1 "16 32 64 128 256 512" "16 32 64 128 256" fix1
  m1d load simd int64 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simd int64 1 "16 32 64 128 256" "6 128 256" fix1
  m1d load simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simt_only int8 4 "256" "2 3 6 32 64 128 256" fix1
  m1d load simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d load simt_only int64 4 "256" "3 6 8 16 32 64 128 256" fix1
  m1d store simd int8 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simd int8 1 "16 32 64 128 256" "6 128 256" fix1
  m1d store simd float16 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simd float16 1 "16 32 64 128 256" "6 32 64 128 256" fix1
  m1d store simd float32 1 "4 8 16 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simd int64 1 "4 8 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simd int64 1 "16 32 64 128 256" "6 128 256" fix1
  m1d store simt_only int8 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simt_only int8 4 "256" "2 3 6 32 64 128 256" fix1
  m1d store simt_only float16 4 "4 8 16 32 64 128 256 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simt_only float32 4 "4 8 16 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simt_only int64 4 "4 8 16 32 64 128 512" "2 3 4 6 8 16 32 64 128 256" fix1
  m1d store simt_only int64 4 "256" "3 6 8 16 32 64 128 256" fix1
echo "===== core1d remainder done ($(date -Is)) ====="
