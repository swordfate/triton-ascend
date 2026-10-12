#!/usr/bin/env bash
# Phase F top-up: SIMD load is the sparse slot (the historical SIMD data is
# single-shot, not rotate-loop).  Re-measure it in the rotate-loop domain with
# the 512 MB window, across element widths and both ranks.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_simdfill}"
OUT="${OUT_DIR:-results/dtype_calibration_v2_rotate512}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-simdfill}"
REPS1="${REPS1:-2}"; ATT1="${ATT1:-2}"
REPS2="${REPS2:-3}"; ATT2="${ATT2:-1}"
WIN="--rotate-step 8192 --rotate-mask 16383"

measure() {
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6"
  local tag="$7" reps="$8" attempts="$9"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} ${tag} blocks='$blocks' strides='$strides' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40
}

measure2d() {
  local op="$1" path="$2" shapes="$3" warps="$4" forms="$5"
  local tag="$6" reps="$7" attempts="$8"
  local slug="${shapes// /-}_${forms// /-}"
  local outfile="$OUT/raw/2d_${op}_${path}_float32_${slug}_${tag}.json"
  echo "----- 2d ${op} ${path} ${tag} shapes='$shapes' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype float32 \
    --shapes $shapes --num-warps $warps --forms $forms $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -30
}

if [[ "$PASS" == "simdfill" ]]; then
  for pass in "pass1 $REPS1 $ATT1" "pass2 $REPS2 $ATT2"; do
    set -- $pass; tag="$1"; r="$2"; a="$3"
    measure load simd float32 "64 256 512" "2 3 4 8"   1 "$tag" "$r" "$a"
    measure load simd int8    "64 256"     "4 8 16 32" 1 "$tag" "$r" "$a"
    measure load simd float16 "64 256"     "2 4 8 16"  1 "$tag" "$r" "$a"
    measure load simd int64   "64 256"     "2 4"       1 "$tag" "$r" "$a"
    measure2d load simd "1x64 8x16 16x32" 1 \
      "both_s2_tight both_s2_gap both_s3_tight both_s3_gap both_s4_tight both_s5_tight" "$tag" "$r" "$a"
    measure2d load simd "1x256 8x16 16x32" 1 \
      "both_s3_tight both_s3_gap both_s5_tight both_s5_gap" "$tag" "$r" "$a"
  done
fi

echo "===== simdfill campaign done ($PASS) $(date -Is) ====="
