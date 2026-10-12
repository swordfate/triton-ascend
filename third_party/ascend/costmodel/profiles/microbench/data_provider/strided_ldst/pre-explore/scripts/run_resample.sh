#!/usr/bin/env bash
# RE-MEASURE the noisy points: every case whose spread exceeded 0.40 on all
# attempts.  79% of repeated measurements agree, 12% are STABLY noisy (the
# access pattern itself is jittery -- re-measuring will not help) and 9% are
# transient.  This pass re-runs all of them with MORE samples per attempt and
# MORE attempts, so a quiet window has a much better chance of being found.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v7}"
OUT="${OUT_DIR:-results/dtype_calibration_v7_resample}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-6}"; ATT="${ATT:-5}"
WIN="--rotate-step 8192 --rotate-mask 16383"

m1d() { local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6" tag="$7"
  local sig="$(echo $blocks | tr ' ' '-')_$(echo $strides | tr ' ' '-')"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_W${warps}_${tag}_${sig}.json"
  echo "----- 1D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype $dtype --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m2d() { local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7" tag="$8"
  local sig="$(echo $shapes | tr ' x' '--')_$(echo $rs | tr ' ' '-')_$(echo $cs | tr ' ' '-')"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_W${warps}_${tag}_${sig}.json"
  echo "----- 2D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
    --op $op --path $path --dtype $dtype --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40; }

m1d load simd float16 1 "16 32 64 128 256 512" "32 64 128 256" resample
m1d load simd float32 1 "16 32 64 128 256 512" "16 32 64 128 256" resample
m1d load simd int64 1 "16 32 64 128 256" "128 256" resample
m1d load simd int8 1 "4 8" "2 16" resample
m1d load simt_only float16 1 "16 64 256" "16 256" resample
m1d load simt_only float16 16 "16 256" "16 256" resample
m1d load simt_only float16 2 "16 64 256" "16 256" resample
m1d load simt_only float16 4 "16 32 64 128 256 512" "6 8 16 32 64 128 256" resample
m1d load simt_only float16 8 "16 64 256" "16 256" resample
m1d load simt_only int64 1 "16 64" "256" resample
m1d load simt_only int64 2 "16 64" "256" resample
m1d load simt_only int64 4 "16 32 64 128" "128 256" resample
m1d load simt_only int64 8 "16" "256" resample
m1d store simd float16 1 "16 32 64 128 256 512" "16 32 64 128 256" resample
m1d store simd float32 1 "16 512" "8 16 32 64 128 256" resample
m1d store simd int64 1 "16 32 64 128 256" "128 256" resample
m1d store simt_only float16 1 "4 64 256" "2 4 256" resample
m1d store simt_only float16 2 "4 64" "2 4 16 256" resample
m1d store simt_only float16 4 "4 8 16 32 64 128" "2 3 4 6 8 16 32 128" resample
m1d store simt_only float16 8 "4 16 64" "2 4 16 256" resample
m1d store simt_only float32 1 "16" "4" resample
m1d store simt_only float32 16 "4" "2" resample
m1d store simt_only float32 2 "16" "256" resample
m1d store simt_only float32 4 "4 8 16" "2 4 6 8 16 128" resample
m1d store simt_only float32 8 "16" "2 4" resample
m1d store simt_only int64 2 "4 16" "4 16 256" resample
m1d store simt_only int64 4 "4 8 16 32" "3 4 6 16" resample
m1d store simt_only int8 1 "16" "16" resample
m1d store simt_only int8 2 "16 64" "2" resample
m1d store simt_only int8 4 "8 16 32 128" "3 4 6 8 16 32 64" resample
m1d store simt_only int8 8 "4 16" "4 16" resample
m2d load simd float16 1 "4x8 4x16 8x8 8x16 16x16" "2 3 4 8 16" "2 16 64" resample
m2d load simd float32 1 "8x16 16x16" "2 3 4 8" "2 8 16" resample
m2d load simd int64 1 "4x16 8x8 8x16 16x16" "2 3 4" "3 4 8 16" resample
m2d load simt_only float16 1 "16x16" "8" "2" resample
m2d load simt_only float16 16 "16x16" "8" "2" resample
m2d load simt_only float16 2 "16x16" "8" "2" resample
m2d load simt_only float16 32 "16x16" "8" "2" resample
m2d load simt_only float16 4 "16x16" "2 3 4 8" "2 3 4 8 16" resample
m2d load simt_only float16 64 "16x16" "8" "2" resample
m2d load simt_only float16 8 "16x16" "8" "2" resample
m2d load simt_only float32 1 "16x16" "2 8" "2 16" resample
m2d load simt_only float32 16 "16x16" "2 8" "2 16" resample
m2d load simt_only float32 2 "16x16" "2 8" "2 16" resample
m2d load simt_only float32 32 "16x16" "2 8" "2 16" resample
m2d load simt_only float32 4 "4x16 8x8 8x16 16x16" "2 3 4 8" "2 3 4 8 16" resample
m2d load simt_only float32 64 "16x16" "8" "2" resample
m2d load simt_only float32 8 "16x16" "2 8" "2 16" resample
m2d load simt_only int64 1 "16x16" "8" "2" resample
m2d load simt_only int64 2 "16x16" "8" "2" resample
m2d load simt_only int64 32 "16x16" "8" "2" resample
m2d load simt_only int64 4 "4x16 8x8 8x16 16x16" "2 3 4 8" "2 3 4 8 16" resample
m2d load simt_only int64 8 "16x16" "8" "2" resample
m2d store simd float32 1 "4x16 8x8 8x16 16x16" "2 3" "8 16" resample
m2d store simd int64 1 "4x16 8x8 8x16 16x16" "2 3 4" "8 16" resample
m2d store simt_only float16 1 "4x8" "2" "2" resample
m2d store simt_only float16 4 "4x8 4x16" "3 8" "2 4" resample
m2d store simt_only float32 4 "4x4 4x8 4x16 8x8 8x16" "3 4 8" "2 3 8" resample
m2d store simt_only int64 1 "4x8" "8" "2" resample
m2d store simt_only int64 4 "4x4 4x8 8x8" "2 3 4 8" "2 3 4 8" resample
m2d store simt_only int8 1 "4x8" "2" "16" resample
m2d store simt_only int8 2 "4x8" "2" "16" resample
m2d store simt_only int8 4 "4x4 4x8" "2 3 8" "3 4 8 16" resample
echo "===== resample done ($(date -Is)) ====="
