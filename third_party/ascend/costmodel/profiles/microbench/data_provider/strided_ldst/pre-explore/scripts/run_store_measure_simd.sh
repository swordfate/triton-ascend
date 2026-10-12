#!/usr/bin/env bash
# Measure the SIMD store target grid with the frequency-gated probe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
mkdir -p results/model_store_v1/raw
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_store}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
bash scripts/board_env_latest.sh python scripts/store_measure_probe.py \
  --mode simd \
  --blocks 32 64 128 256 \
  --strides 1 2 3 4 5 6 7 8 9 10 11 12 16 20 24 32 40 48 64 96 128 192 256 \
  --iters 2000 8000 \
  --reps 5 \
  --max-attempts 2 \
  --device 0 \
  --out results/model_store_v1/raw/board_store_simd.json
