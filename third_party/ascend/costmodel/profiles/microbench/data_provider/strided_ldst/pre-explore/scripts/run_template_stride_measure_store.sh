#!/usr/bin/env bash
# 1D rank1 template store measurement (Event slope on rotate loop).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
mkdir -p results/model_template_stride_v2/raw
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_template_store}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
OUT_JSON="${OUT_JSON:-results/model_template_stride_v2/raw/board_template_stride_store.json}"
REPS="${REPS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
NUM_WARPS="${NUM_WARPS:-32}"
BLOCKS="${BLOCKS:-4 8 16 32 64 128 256 512 1024 2048}"
STRIDES="${STRIDES:-3 5 6 7 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 40 48 63 65 80 96 127 129 160 192 255}"
bash scripts/board_env.sh python scripts/template_stride_store_probe.py \
  --blocks $BLOCKS \
  --strides $STRIDES \
  --num-warps $NUM_WARPS \
  --iters 2000 8000 \
  --reps "$REPS" \
  --max-attempts "$MAX_ATTEMPTS" \
  --device 0 \
  --asm-dir results/model_template_stride_v2/ir_evidence/measure_asm \
  --out "$OUT_JSON"
