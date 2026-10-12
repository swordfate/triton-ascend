#!/usr/bin/env bash
# FINAL campaign: 1-D + 2-D grids with the SIMT num_warps sweep.
#
# Design (after the 7x cost increase from W in {1,2,4,8,16,32,64}):
#   * strides trimmed 23 -> 16, dropping the rare odd values 7 9 11 28 40 96 192
#   * the num_warps sweep is measured on a REDUCED (block x stride) grid and
#     pooled with the full W=4 grid, instead of running everything at all 7 W
#
#   PASS=core1d    1D full grid, SIMD W=1, SIMT W=4            (2048)
#   PASS=wsweep1d  1D reduced grid, SIMT W in {1,2,8,16,32,64} (1152)
#   PASS=core2d    2D main, SIMD W=1, SIMT W=4                 (6912)
#   PASS=wsweep2d  2D reduced, SIMT W in {1,2,8,16,32,64}      (1296)
#   PASS=n1        2D n x 1, SIMT W in all 7                   (1008)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v4}"
OUT="${OUT_DIR:-results/dtype_calibration_v4_final}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-core1d}"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
E="4 8 16 32 64 128 256 512"
S16="2 3 4 6 8 16 32 64 128 256"
EB="4 16 64 256"
SB="2 4 16 256"
WS="1 2 8 16 32 64"
SHAPES="4x4 4x8 4x16 8x8 8x16 16x16"
RS="2 3 4 6 8 16"
CS="2 3 4 8 16 64"
SH2="4x8 16x16"
RS2="2 8"
CS2="2 64"

m1d() {  # <op> <path> <dtype> <warps> <blocks> <strides> <tag>
  local op="$1" path="$2" dtype="$3" warps="$4" blocks="$5" strides="$6" tag="$7"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 1D ${op} ${path} ${dtype} W=${warps} n=$(echo $blocks|wc -w)x$(echo $strides|wc -w) $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}
m2d() {  # <op> <path> <dtype> <warps> <shapes> <rs> <cs> <tag>
  local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7" tag="$8"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 2D ${op} ${path} ${dtype} W=${warps} shapes=${shapes} rs='${rs}' cs='${cs}' $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py --op "$op" --path "$path" --dtype "$dtype" \
    --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}

if [[ "$PASS" == "core1d" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    m1d "$op" simd      "$d" 1 "$E" "$S16" core
    m1d "$op" simt_only "$d" 4 "$E" "$S16" core
  done; done
fi
if [[ "$PASS" == "wsweep1d" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    for w in $WS; do m1d "$op" simt_only "$d" "$w" "$EB" "$SB" wsweep; done
  done; done
fi
if [[ "$PASS" == "core2d" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    m2d "$op" simd      "$d" 1 "$SHAPES" "$RS" "$CS" core
    m2d "$op" simt_only "$d" 4 "$SHAPES" "$RS" "$CS" core
  done; done
fi
if [[ "$PASS" == "wsweep2d" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    for w in $WS; do m2d "$op" simt_only "$d" "$w" "$SH2" "$RS2" "$CS2" wsweep; done
  done; done
fi
if [[ "$PASS" == "n1" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    m2d "$op" simd "$d" 1 "4x1 16x1" "2 4 8 16" "1" n1
    for w in 1 2 4 8 16 32 64; do m2d "$op" simt_only "$d" "$w" "4x1 16x1" "2 4 8 16" "1" n1; done
  done; done
fi
echo "===== campaign done ($PASS) $(date -Is) ====="
