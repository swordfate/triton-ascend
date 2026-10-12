#!/usr/bin/env bash
# Phase A driver: element dtype x static stride calibration for the three
# strided lowerings (SIMD / pure SIMT / SIMT stride template).
#
# All measurements run on NPU0 through scripts/board_env.sh, which uses the
# installed wj_autoscope package and preserves the CANN PYTHONPATH.
#
# The measurement grid is a BYTE-GAP CROSSING design: strides are chosen per
# dtype so that the same byte gap (element_bytes * stride) is realised by
# several different element types.  That is what decides whether the SIMD
# wide/gather boundary lives in the element domain (current model) or in the
# byte domain (physical MTE transaction width).
#
#   gap  4B : int8 s4,  f16 s2
#   gap  6B : int8 s6,  f16 s3
#   gap  8B : int8 s8,  f16 s4,  f32 s2
#   gap 10B : int8 s10, f16 s5
#   gap 12B : int8 s12, f16 s6,  f32 s3
#   gap 16B : int8 s16, f16 s8,  f32 s4,  i64 s2
#   gap 24B : int8 s24, f16 s12, f32 s6,  i64 s3
#   gap 32B : int8 s32, f16 s16, f32 s8,  i64 s4
#
# stride == 1 is never measured (contiguous is out of the strided model).
#
# Usage:
#   PASS=path      bash scripts/run_dtype_calibration.sh   # compile-only checks
#   PASS=simd      bash scripts/run_dtype_calibration.sh
#   PASS=simt      bash scripts/run_dtype_calibration.sh
#   PASS=template  bash scripts/run_dtype_calibration.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_dtype_v1}"
OUT="${OUT_DIR:-results/dtype_calibration_v1}"
mkdir -p "$OUT/ir_evidence/path_check_asm" "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"

PASS="${PASS:-path}"
REPS1="${REPS1:-2}"; ATT1="${ATT1:-2}"
REPS2="${REPS2:-3}"; ATT2="${ATT2:-1}"

measure() {  # measure <op> <path> <dtype> <blocks> <strides> <warps> <tag> <reps> <attempts>
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6"
  local tag="$7" reps="$8" attempts="$9"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} ${tag} blocks='$blocks' strides='$strides' warps='$warps' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" \
    --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40
}

measure2d() {  # measure2d <op> <path> <dtype> <shapes> <warps> <forms> <tag> <reps> <attempts>
  local op="$1" path="$2" dtype="$3" shapes="$4" warps="$5" forms="$6"
  local tag="$7" reps="$8" attempts="$9"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_${tag}.json"
  echo "----- 2d ${op} ${path} ${dtype} ${tag} shapes='$shapes' forms='$forms' $(date -Is) -----"
  REPS="$reps" MAX_ATTEMPTS="$attempts" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --shapes $shapes --num-warps $warps --forms $forms \
    --reps "$reps" --max-attempts "$attempts" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -30
}

if [[ "$PASS" == "pathlite" ]]; then
  # Boundary-focused grid: covers bail(1), bail(2), template(3), bail(4),
  # template(5), bail(8), template(9), bail(16), template(17), template(255)
  # for four element widths.  This is the acceptance grid for
  # "path check mismatches = 0".
  echo "===== path check (lite): template dispatch across dtypes $(date -Is) ====="
  bash scripts/board_env.sh python scripts/dtype_stride_path_check.py \
    --dtypes int8 float16 float32 int64 \
    --modes template \
    --blocks 64 \
    --strides 1 2 3 4 5 8 9 16 17 255 \
    --out "$OUT/ir_evidence/dtype_stride_path_check_template.json" \
    --asm-dir "$OUT/ir_evidence/path_check_asm" --asm-limit 12 \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" \
    | tee "$OUT/ir_evidence/path_check_template.log"

  echo "===== path check: simd / simt_only compile support $(date -Is) ====="
  bash scripts/board_env.sh python scripts/dtype_stride_path_check.py \
    --dtypes int8 float16 float32 int64 \
    --modes simd simt_only \
    --blocks 64 \
    --strides 3 8 \
    --out "$OUT/ir_evidence/dtype_stride_path_check_modes.json" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" \
    | tee "$OUT/ir_evidence/path_check_modes.log"
fi

if [[ "$PASS" == "path" ]]; then
  echo "===== path check: template dispatch across dtypes $(date -Is) ====="
  bash scripts/board_env.sh python scripts/dtype_stride_path_check.py \
    --dtypes int8 float16 bfloat16 float32 int32 int64 \
    --modes template \
    --blocks 64 \
    --strides 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 31 32 63 64 65 255 \
    --out "$OUT/ir_evidence/dtype_stride_path_check_template.json" \
    --asm-dir "$OUT/ir_evidence/path_check_asm" --asm-limit 12 \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" \
    | tee "$OUT/ir_evidence/path_check_template.log"

  echo "===== path check: simd / simt_only compile support $(date -Is) ====="
  bash scripts/board_env.sh python scripts/dtype_stride_path_check.py \
    --dtypes int8 float16 bfloat16 float32 int32 int64 \
    --modes simd simt_only \
    --blocks 64 \
    --strides 3 8 \
    --out "$OUT/ir_evidence/dtype_stride_path_check_modes.json" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" \
    | tee "$OUT/ir_evidence/path_check_modes.log"
fi

if [[ "$PASS" == "simd" ]]; then
  # BYTE-GAP CROSSING design.  Compile time is the binding constraint
  # (~40 s per distinct (path, dtype, block, stride) constexpr set), so the
  # grid is the minimal one that still realises the same byte gap with several
  # element widths:
  #   gap  4B: int8 s4,  f16 s2
  #   gap  8B: int8 s8,  f16 s4,  f32 s2
  #   gap 12B: int8 s12, f16 s6,  f32 s3
  #   gap 16B: int8 s16, f16 s8,  f32 s4,  i64 s2
  #   gap 32B: int8 s32, f16 s16, f32 s8,  i64 s4
  for op in load store; do
    measure "$op" simd int8    "64"     "4 8 12 16 32" 1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd float16 "64"     "2 4 6 8 16"   1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd float32 "64"     "2 3 4 8"      1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd int64   "64"     "2 4"          1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd int8    "256"    "4 8 16"       1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd float32 "256"    "2 3 4"        1 pass1 "$REPS1" "$ATT1"
    measure "$op" simd int8    "64"     "4 8 16 32"    1 pass2 "$REPS2" "$ATT2"
    measure "$op" simd float16 "64"     "2 4 8 16"     1 pass2 "$REPS2" "$ATT2"
    measure "$op" simd float32 "64"     "2 3 4 8"      1 pass2 "$REPS2" "$ATT2"
    measure "$op" simd int64   "64"     "2 4"          1 pass2 "$REPS2" "$ATT2"
  done
fi

if [[ "$PASS" == "simt" ]]; then
  # Pure-SIMT transfer check: widest and narrowest element, two warp counts.
  for op in load store; do
    measure "$op" simt_only int8  "256" "4 8 16" 1 pass1 "$REPS1" "$ATT1"
    measure "$op" simt_only int8  "256" "4 8 16" 4 pass1 "$REPS1" "$ATT1"
    measure "$op" simt_only int64 "256" "2 4"    1 pass1 "$REPS1" "$ATT1"
    measure "$op" simt_only int64 "256" "2 4"    4 pass1 "$REPS1" "$ATT1"
    measure "$op" simt_only int8  "256" "4 8 16" 1 pass2 "$REPS2" "$ATT2"
    measure "$op" simt_only int8  "256" "4 8 16" 4 pass2 "$REPS2" "$ATT2"
    measure "$op" simt_only int64 "256" "2 4"    1 pass2 "$REPS2" "$ATT2"
    measure "$op" simt_only int64 "256" "2 4"    4 pass2 "$REPS2" "$ATT2"
  done
fi

if [[ "$PASS" == "template" ]]; then
  # Template v2 transfer: static non-power-of-two strides only (the trigger).
  for op in load store; do
    measure "$op" template int8    "16 256 1024" "3 9" 32 pass1 "$REPS1" "$ATT1"
    measure "$op" template float16 "16 256 1024" "3 9" 32 pass1 "$REPS1" "$ATT1"
    measure "$op" template int64   "16 256 1024" "3 9" 32 pass1 "$REPS1" "$ATT1"
    measure "$op" template int8    "16 256 1024" "3 9" 32 pass2 "$REPS2" "$ATT2"
    measure "$op" template float16 "16 256 1024" "3 9" 32 pass2 "$REPS2" "$ATT2"
    measure "$op" template int64   "16 256 1024" "3 9" 32 pass2 "$REPS2" "$ATT2"
  done
fi

if [[ "$PASS" == "2dpath" ]]; then
  echo "===== 2D path check (all forms x all modes) $(date -Is) ====="
  bash scripts/board_env.sh python scripts/stride2d_path_check.py \
    --shapes 1x64 8x16 \
    --modes template simd simt_only \
    --dtype float32 \
    --out "$OUT/ir_evidence/stride2d_path_check.json" \
    --asm-dir "$OUT/ir_evidence/path2d_asm" --asm-limit 12 \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" \
    | tee "$OUT/ir_evidence/path2d.log"
fi

if [[ "$PASS" == "2d" ]]; then
  # 2D-strided family: BOTH axes carry holes (both_sK_tight / both_sK_gap).
  # R == 1 shapes are included on purpose: a rank-2 tile with a singleton row
  # axis is a real lowering shape and must be modelled.
  BOTH_FORMS="both_s2_tight both_s2_gap both_s3_tight both_s3_gap both_s5_tight both_s5_gap"
  TEMPLATE_FORMS="both_s3_tight both_s3_gap both_s5_tight both_s5_gap"
  SHAPES="1x64 1x256 8x16 16x32"
  for op in load store; do
    measure2d "$op" simd      float32 "$SHAPES" 1  "$BOTH_FORMS"     pass1 "$REPS1" "$ATT1"
    measure2d "$op" simt_only float32 "$SHAPES" 4  "$BOTH_FORMS"     pass1 "$REPS1" "$ATT1"
    measure2d "$op" template  float32 "$SHAPES" 32 "$TEMPLATE_FORMS" pass1 "$REPS1" "$ATT1"
    measure2d "$op" simd      float32 "$SHAPES" 1  "$BOTH_FORMS"     pass2 "$REPS2" "$ATT2"
    measure2d "$op" simt_only float32 "$SHAPES" 4  "$BOTH_FORMS"     pass2 "$REPS2" "$ATT2"
    measure2d "$op" template  float32 "$SHAPES" 32 "$TEMPLATE_FORMS" pass2 "$REPS2" "$ATT2"
  done
fi

if [[ "$PASS" == "all_a" ]]; then
  echo "########## PHASE A CAMPAIGN START $(date -Is) ##########"
  PASS=pathlite bash "$0"
  PASS=simd     bash "$0"
  PASS=simt     bash "$0"
  PASS=template bash "$0"
  echo "########## PHASE A CAMPAIGN DONE $(date -Is) ##########"
fi

if [[ "$PASS" == "all_b" ]]; then
  echo "########## PHASE B CAMPAIGN START $(date -Is) ##########"
  PASS=2dpath bash "$0"
  PASS=2d     bash "$0"
  echo "########## PHASE B CAMPAIGN DONE $(date -Is) ##########"
fi

echo "===== dtype calibration driver done ($PASS) $(date -Is) ====="
