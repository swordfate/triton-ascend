#!/usr/bin/env bash
# Phase R driver: re-measure with the HISTORICAL rotation window (512 MB).
#
# Today's Phase A/B probes use a 64 MB byte budget for the rotation window;
# the historical SIMT/store/template probes used step=8192 mask=16383, i.e.
# 134,217,728 elements = 512 MB for f32.  This driver re-runs the SAME probe
# code with the historical window so the two datasets are directly comparable
# cell by cell.  Measurement method, gates and grid are unchanged.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_dtype_v2_rotate512}"
OUT="${OUT_DIR:-results/dtype_calibration_v2_rotate512}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"

PASS="${PASS:-rot}"
REPS1="${REPS1:-2}"; ATT1="${ATT1:-2}"
REPS2="${REPS2:-3}"; ATT2="${ATT2:-1}"
WIN="--rotate-step 8192 --rotate-mask 16383"

measure() {  # measure <op> <path> <dtype> <blocks> <strides> <warps> <tag> <reps> <attempts>
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6"
  local tag="$7" reps="$8" attempts="$9"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} ${tag} blocks='$blocks' strides='$strides' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps \
    $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -30
}

measure_win() {  # <op> <path> <dtype> <blocks> <strides> <warps> <tag> <reps> <attempts> <step> <mask>
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6"
  local tag="$7" reps="$8" attempts="$9" st="${10}" mk="${11}"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} ${tag} step=$st mask=$mk $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps \
    --rotate-step "$st" --rotate-mask "$mk" \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -20
}

measure2d() {  # measure2d <op> <path> <shapes> <warps> <forms> <tag> <reps> <attempts>
  local op="$1" path="$2" shapes="$3" warps="$4" forms="$5"
  local tag="$6" reps="$7" attempts="$8"
  local slug="${shapes}_${forms// /-}"
  local outfile="$OUT/raw/2d_${op}_${path}_float32_${slug}_${tag}.json"
  echo "----- 2d ${op} ${path} ${tag} shapes='$shapes' forms='$forms' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype float32 \
    --shapes $shapes --num-warps $warps --forms $forms \
    $WIN \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -25
}

if [[ "$PASS" == "rot" ]]; then
  # The grid is a SUBSET of the v1 grid (cell-by-cell comparable) plus the
  # f32 anchors that v1 lacked for the SIMT and template paths.
  for op in load store; do
    for pass in "pass1 $REPS1 $ATT1" "pass2 $REPS2 $ATT2"; do
      set -- $pass; tag="$1"; r="$2"; a="$3"
      measure "$op" simd      float32 "64"  "2 3 4 8" 1 "$tag" "$r" "$a"
      measure "$op" simd      float32 "256" "2 3"     1 "$tag" "$r" "$a"
      measure "$op" simd      int8    "64"  "4 8"     1 "$tag" "$r" "$a"
      measure "$op" simt_only float32 "64"  "2 3"     1 "$tag" "$r" "$a"
      measure "$op" simt_only int8    "256" "4"       1 "$tag" "$r" "$a"
      measure "$op" template  float32 "64"  "3"       1 "$tag" "$r" "$a"
      measure "$op" template  int8    "64"  "3"       1 "$tag" "$r" "$a"
    done
  done
fi

if [[ "$PASS" == "rot2d" ]]; then
  for op in load store; do
    for pass in "pass1 $REPS1 $ATT1" "pass2 $REPS2 $ATT2"; do
      set -- $pass; tag="$1"; r="$2"; a="$3"
      measure2d "$op" simd      "8x16" 1 "both_s3_tight both_s3_gap" "$tag" "$r" "$a"
      measure2d "$op" simd      "1x64" 1 "both_s3_tight"             "$tag" "$r" "$a"
      measure2d "$op" simt_only "8x16" 1 "both_s3_tight"             "$tag" "$r" "$a"
      measure2d "$op" template  "8x16" 1 "both_s3_tight"             "$tag" "$r" "$a"
      measure2d "$op" template  "1x64" 1 "both_s3_tight"             "$tag" "$r" "$a"
    done
  done
fi

if [[ "$PASS" == "stepiso" ]]; then
  # Isolate STEP from WINDOW on the two SIMT int8 points whose v1/v2 values
  # differ by 70% / 156%.  v1 used step=4096 mask=16383 (64 MB for int8);
  # v2 used step=8192 mask=16383 (134 MB).  Here we hold mask and vary step,
  # and hold step=4096 and double the mask, so the two effects separate.
  for op in load store; do
    measure_win "$op" simt_only int8 "256" "4" 1 "step4096_mask16383" 2 2 4096 16383
    measure_win "$op" simt_only int8 "256" "4" 1 "step4096_mask32767" 2 2 4096 32767
    measure_win "$op" simt_only int8 "256" "4" 1 "step2048_mask32767" 2 2 2048 32767
  done
fi

echo "===== rotate512 campaign done ($PASS) $(date -Is) ====="
