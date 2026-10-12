#!/usr/bin/env bash
# FINAL-v2 campaign (reduced 2-D).  Changes vs the previous round:
#   * core2d : rs {2,3,4,8,16} -> {2,3,4,8};  cs {2,3,4,8,16,64} -> {2,3,4,8,16}
#              (180 -> 120 forms; max address span 0.94 MB -> 0.12 MB)
#   * n1     : only the 'load simd' slice (4 invocations) is kept.  The n1 sweep
#              confirmed that an (R,1) tile has the SAME address list as a 1-D
#              block: 7 of 8 comparable points agreed within 4%.  The one
#              exception (simd, R=16, OS=2, +107%) is why the simd slice stays.
#   * fix1   : the core1d remainder lost to the empty --strides bug.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v5}"
OUT="${OUT_DIR:-results/dtype_calibration_v5_final}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-core2d}"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
SH="4x4 4x8 4x16 8x8 8x16 16x16"
RS="2 3 4 8"
CS="2 3 4 8 16"
SH2="4x8 16x16"; RS2="2 8"; CS2="2 16"
WS="1 2 8 16 32 64"

m2d() {  # op path dtype warps shapes rs cs tag
  local op="$1" path="$2" dtype="$3" warps="$4" shapes="$5" rs="$6" cs="$7" tag="$8"
  local outfile="$OUT/raw/2d_${op}_${path}_${dtype}_W${warps}_${tag}.json"
  echo "----- 2D ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py --op "$op" --path "$path" --dtype "$dtype" \
    --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}

if [[ "$PASS" == "fix1" ]]; then
  OUT_DIR="$OUT" bash scripts/run_core1d_remainder.sh
fi
if [[ "$PASS" == "core2d" ]]; then
  for op in load store; do for d in int8 float16 float32 int64; do
    m2d "$op" simd      "$d" 1 "$SH" "$RS" "$CS" core
    m2d "$op" simt_only "$d" 4 "$SH" "$RS" "$CS" core
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
  done; done
fi
echo "===== campaign v2 done ($PASS) $(date -Is) ====="
