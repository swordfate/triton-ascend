#!/usr/bin/env bash
# Measure the six missing large strides for W=1/2/4/8/16 on physical NPU1.
set -eo pipefail
umask 077
cd "$(dirname "$0")/.."

RESULTS_DIR="results/v4_aligned_23"
mkdir -p "$RESULTS_DIR"
for W in 1 2 4 8 16; do
  echo "=== v4 aligned_23 W=$W $(date -Is)"
  ASCEND_RT_VISIBLE_DEVICES=1 bash scripts/board_env.sh \
    python3 scripts/simt_frequency_retry_probe_aligned.py \
    --blocks 4 8 16 32 64 128 256 512 \
    --num-warps "$W" \
    --strides 20 24 40 48 96 192 \
    --iters 2000 8000 \
    --reps 3 \
    --rotate-step 8192 \
    --rotate-mask 16383 \
    --warmup-launches 1000 \
    --max-retries 2 \
    --retry-sleep 2.0 \
    --freq-device 1 \
    --out "$RESULTS_DIR/board_v4_aligned_23_w${W}.json"
done
echo "ALL DONE $(date -Is)" > "$RESULTS_DIR/done.txt"
