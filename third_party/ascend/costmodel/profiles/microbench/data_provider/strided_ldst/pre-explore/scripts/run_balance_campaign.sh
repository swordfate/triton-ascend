#!/usr/bin/env bash
# Phase F top-up 2: balance the point counts across element widths.
#
# The historical datasets are f32-only, so after pooling, f32 has ~1250 points
# per slot while int8 / float16 / int64 have 2-8.  SIMD load in particular has
# only 15+42 points in total.  This campaign measures a dense grid for the
# under-represented (path, op, dtype) combinations so every element width gets
# a comparable number of points, and the byte-domain pooling claim can be
# tested rather than assumed.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_balance}"
OUT="${OUT_DIR:-results/dtype_calibration_v2_rotate512}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-balance}"
REPS1="${REPS1:-2}"; ATT1="${ATT1:-2}"
REPS2="${REPS2:-3}"; ATT2="${ATT2:-1}"
WIN="--rotate-step 8192 --rotate-mask 16383"

# 5 blocks x 8 strides = 40 cases per (op, path, dtype).
BLOCKS="16 32 64 128 256"
STRIDES="2 3 4 5 8 16 32 64"
# template only fires for static non-power-of-two stride >= 3.
TSTRIDES="3 5 6 7 9 12 24 48"

measure() {  # <op> <path> <dtype> <blocks> <strides> <warps> <tag> <reps> <attempts>
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6"
  local tag="$7" reps="$8" attempts="$9"
  local outfile="$OUT/raw/bal_${op}_${path}_${dtype}_${tag}.json"
  echo "----- bal ${op} ${path} ${dtype} ${tag} n=$(echo $blocks|wc -w)x$(echo $strides|wc -w) $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -50
}

measure2d() {  # <op> <path> <dtype> <shapes> <warps> <forms> <tag> <reps> <attempts>
  local op="$1" path="$2" dtype="$3" shapes="$4" warps="$5" forms="$6"
  local tag="$7" reps="$8" attempts="$9"
  local slug="${shapes// /-}_${forms// /-}"
  local outfile="$OUT/raw/bal2d_${op}_${path}_${dtype}_${slug}_${tag}.json"
  echo "----- bal2d ${op} ${path} ${dtype} ${tag} $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --shapes $shapes --num-warps $warps --forms $forms $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40
}

if [[ "$PASS" == "balance" ]]; then
  for pass in "pass1 $REPS1 $ATT1" "pass2 $REPS2 $ATT2"; do
    set -- $pass; tag="$1"; r="$2"; a="$3"
    # --- SIMD load: the sparsest slot, all four widths
    for d in int8 float16 float32 int64; do
      measure load simd "$d" "$BLOCKS" "$STRIDES" 1 "$tag" "$r" "$a"
    done
    # --- SIMD store: non-f32 only (f32 already has 112 points)
    for d in int8 float16 int64; do
      measure store simd "$d" "$BLOCKS" "$STRIDES" 1 "$tag" "$r" "$a"
    done
    # --- pure SIMT: non-f32 only
    for d in int8 int64; do
      measure load  simt_only "$d" "$BLOCKS" "$STRIDES" 1 "$tag" "$r" "$a"
      measure store simt_only "$d" "$BLOCKS" "$STRIDES" 1 "$tag" "$r" "$a"
    done
    # --- template: non-f32 only, template-eligible strides
    for d in int8 float16 int64; do
      measure load  template "$d" "$BLOCKS" "$TSTRIDES" 32 "$tag" "$r" "$a"
      measure store template "$d" "$BLOCKS" "$TSTRIDES" 32 "$tag" "$r" "$a"
    done
    # --- 2-D for the non-f32 widths (tests the byte-domain claim on rank-2)
    for d in int8 float16; do
      measure2d load  simd      "$d" "1x64 8x16 16x32" 1  "both_s3_tight both_s3_gap both_s5_tight" "$tag" "$r" "$a"
      measure2d load  template  "$d" "1x64 8x16 16x32" 32 "both_s3_tight both_s3_gap both_s5_tight" "$tag" "$r" "$a"
      measure2d store simd      "$d" "1x64 8x16 16x32" 1  "both_s3_tight both_s3_gap both_s5_tight" "$tag" "$r" "$a"
    done
  done
fi

echo "===== balance campaign done ($PASS) $(date -Is) ====="
