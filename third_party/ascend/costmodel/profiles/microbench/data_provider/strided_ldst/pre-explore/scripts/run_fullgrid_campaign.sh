#!/usr/bin/env bash
# FULL 1-D measurement grid, four element widths, all four slots.
#
# Grid (per the agreed design):
#   elements : 4 8 16 32 64 128 256 512                       (8)
#   strides  : 2..12 16 20 24 28 32 40 48 64 96 128 192 256   (23)
#   => 184 cells per (slot, dtype); 4 slots x 4 dtypes = 2944 cells
#
# Protocol change vs the earlier rounds: reps 2 -> 3 and attempts 2 -> 3, so a
# single unlucky high-spread attempt no longer kills a cell.  The only invalid
# reason observed so far was spread (16 of 642 cases = 2.49%; witness and
# frequency never fired).
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_fullgrid}"
OUT="${OUT_DIR:-results/dtype_calibration_v3_fullgrid}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-simd}"
REPS="${REPS:-3}"; ATT="${ATT:-3}"
WIN="--rotate-step 8192 --rotate-mask 16383"
BLOCKS="4 8 16 32 64 128 256 512"
STRIDES="2 3 4 5 6 7 8 9 10 11 12 16 20 24 28 32 40 48 64 96 128 192 256"

measure() {  # <op> <path> <dtype> <warps> <tag>
  local op="$1" path="$2" dtype="$3" warps="$4" tag="$5"
  local outfile="$OUT/raw/${op}_${path}_${dtype}_${tag}.json"
  echo "----- ${op} ${path} ${dtype} W=${warps} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $BLOCKS --strides $STRIDES --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -60
}

if [[ "$PASS" == "simd" ]]; then
  for op in load store; do
    for d in int8 float16 float32 int64; do measure "$op" simd "$d" 1 full; done
  done
fi

if [[ "$PASS" == "simt" ]]; then
  # num_warps 4 is the contract used by the shipped SIMT model
  for op in load store; do
    for d in int8 float16 float32 int64; do measure "$op" simt_only "$d" 4 full; done
  done
fi

echo "===== fullgrid campaign done ($PASS) $(date -Is) ====="
