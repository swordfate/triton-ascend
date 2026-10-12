#!/usr/bin/env bash
# Re-measure the 2-D int8 LOAD cells with the fixed probe.
#
# The old probe wrote the load result into 'out' (the ELEMENT dtype).  The
# kernel stores tl.sum(acc), which is float32, so int8 overflowed as soon as a
# tile held ~43+ elements: every such case was flagged wrong (448 cases) while
# the timings themselves were fine.  The probe now uses the float32 sink,
# matching the 1-D probe.  Only int8 load is affected, so only that slice is
# re-measured.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_v6}"
OUT="${OUT_DIR:-results/dtype_calibration_v6_int8fix}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
SH="4x4 4x8 4x16 8x8 8x16 16x16"
RS="2 3 4 8"
CS="2 3 4 8 16"

m2d() {  # path warps shapes rs cs tag
  local path="$1" warps="$2" shapes="$3" rs="$4" cs="$5" tag="$6"
  local outfile="$OUT/raw/2d_load_${path}_int8_W${warps}_${tag}.json"
  echo "----- 2D load ${path} int8 W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/stride2d_measure_probe.py --op load --path "$path" --dtype int8 \
    --shapes $shapes --row-strides $rs --inner-strides $cs --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40
}

m2d simd      1 "$SH" "$RS" "$CS" core
m2d simt_only 4 "$SH" "$RS" "$CS" core
for w in 1 2 8 16 32 64; do
  m2d simt_only "$w" "4x8 16x16" "2 8" "2 16" wsweep
done
m2d simd 1 "4x1 16x1" "2 4 8 16" "1" n1
echo "===== int8 load re-measure done $(date -Is) ====="
