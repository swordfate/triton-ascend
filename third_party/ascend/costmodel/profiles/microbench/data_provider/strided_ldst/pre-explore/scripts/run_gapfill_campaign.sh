#!/usr/bin/env bash
# Phase F top-up 3: fill the remaining coverage gaps.
#
# After the balance round the coverage matrix still had:
#   * pure SIMT float16 : ZERO points (the balance grid only listed int8/int64)
#   * pure SIMT int8 / int64 : 4 / 2 points
#   * SIMD non-f32 : 33-40 points (wanted more)
# template is deliberately NOT topped up in this round.
#
# Single pass on purpose: the dataset takes the min over whatever passes exist,
# and one pass already yields valid+correct points.  A second pass roughly
# doubles the wall clock, and the SIMT grids cost ~25 s per case.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_gapfill}"
OUT="${OUT_DIR:-results/dtype_calibration_v2_rotate512}"
mkdir -p "$OUT/raw" "$OUT/measure_asm" "$TRITON_CACHE_DIR"
PASS="${PASS:-gap}"
REPS="${REPS:-2}"; ATT="${ATT:-2}"
WIN="--rotate-step 8192 --rotate-mask 16383"

measure() {  # <op> <path> <dtype> <blocks> <strides> <warps> <tag>
  local op="$1" path="$2" dtype="$3" blocks="$4" strides="$5" warps="$6" tag="$7"
  local outfile="$OUT/raw/gap_${op}_${path}_${dtype}_${tag}.json"
  echo "----- gap ${op} ${path} ${dtype} ${tag} $(date -Is) -----"
  REPS="$REPS" MAX_ATTEMPTS="$ATT" bash scripts/board_env.sh python \
    scripts/dtype_stride_measure_probe.py \
    --op "$op" --path "$path" --dtype "$dtype" \
    --blocks $blocks --strides $strides --num-warps $warps $WIN \
    --reps "$REPS" --max-attempts "$ATT" \
    --asm-dir "$OUT/measure_asm" --out "$outfile" \
    2>&1 | grep --line-buffered -v "UserWarning\|warnings.warn" | tail -40
}

if [[ "$PASS" == "gap" ]]; then
  # ---- 1. pure SIMT float16 (the outright hole) + int8/int64 top-up.
  # Small blocks first so that a partial run still yields usable points.
  for op in load store; do
    measure "$op" simt_only float16 "32 64 128 256" "2 3 4 5 8 16 32 64" 1 gap1
    measure "$op" simt_only int8    "64 128 256"    "2 4 8 16 32"         1 gap1
    measure "$op" simt_only int64   "64 128 256"    "2 4 8 16 32"         1 gap1
  done
  # ---- 2. SIMD non-f32 top-up (cheap: ~4.6 s/case).
  for op in load store; do
    for d in int8 float16 int64; do
      measure "$op" simd "$d" "16 32 64 128 256" "2 3 4 5 8 16 32 64" 1 gap1
    done
  done
  # ---- 3. SIMD load wide branch needs more stride==2 blocks.
  for d in int8 float16 int64 float32; do
    measure load simd "$d" "8 16 32 48 64 96 128 256 512" "2" 1 gap2
  done
fi

echo "===== gapfill campaign done ($PASS) $(date -Is) ====="
