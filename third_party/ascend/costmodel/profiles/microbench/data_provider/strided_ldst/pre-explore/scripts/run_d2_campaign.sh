#!/usr/bin/env bash
# 2D campaign in the (row_stride, col_stride) parameterisation.
#
#   PASS=d2main : both axes strided
#       R,C in {4,8,16} (9 shapes)  x  rs in {2,3,4,6,8,16}  x  cs in {2,3,4,8,16,32,64,128}
#       = 9 x 6 x 8 = 432 cases per (op, path, dtype);  x16 = 6912
#       shape = (R*rs) rows x (C*cs) cols ;  OS = rs*C*cs
#   PASS=d2n1   : single column, row jumps
#       shapes 4x1 8x1 16x1  x  rs in {2,3,4,6,8,16}  x  cs = 1
#       = 18 cases per (op, path, dtype);  x16 = 288
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_d2}"
OUT="${OUT_DIR:-results/dtype_calibration_v3_delta}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-d2main}"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
SHAPES="4x4 4x8 4x16 8x4 8x8 8x16 16x4 16x8 16x16"
RS="2 3 4 6 8 16"
CS="2 3 4 8 16 32 64 128"

measure2d() {  # <op> <path> <dtype> <warps> <shapes> <rs> <cs> <tag>
  local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7" tag="$8"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} W=${warps} shapes=${shapes} rs='${rs}' cs='${cs}' $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps \
    $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}

if [[ "$PASS" == "d2main" ]]; then
  for op in load store; do
    for d in int8 float16 float32 int64; do
      measure2d "$op" simd      "$d" 1 "$SHAPES" "$RS" "$CS" d2main
      measure2d "$op" simt_only "$d" 4 "$SHAPES" "$RS" "$CS" d2main
    done
  done
fi

if [[ "$PASS" == "d2n1" ]]; then
  for op in load store; do
    for d in int8 float16 float32 int64; do
      measure2d "$op" simd      "$d" 1 "4x1 8x1 16x1" "$RS" "1" d2n1
      measure2d "$op" simt_only "$d" 4 "4x1 8x1 16x1" "$RS" "1" d2n1
    done
  done
fi

echo "===== d2 campaign done ($PASS) $(date -Is) ====="
