#!/usr/bin/env bash
# N2 dense-stride measurement for the key underfilled num_warps values (32, 64).
set -eo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_EXPLORE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE_EXPLORE"

set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
OUT_DIR="${OUT_DIR:-$PWD/results/board_v2_simt_warps}"
mkdir -p "$OUT_DIR"

STRIDES="1 2 3 4 5 6 7 8 9 10 11 12 16 20 24 32 40 48 64 96 128 192 256"
for W in 32 64; do
  python3 scripts/simt_frequency_retry_probe.py \
    --num-warps "$W" \
    --blocks 32 64 128 256 \
    --strides $STRIDES \
    --iters 2000 8000 \
    --reps 2 \
    --rotate-step 8192 \
    --rotate-mask 16383 \
    --warmup-launches 2000 \
    --max-retries 3 \
    --retry-sleep 2.0 \
    --out "$OUT_DIR/dense_w${W}.json" 2>&1 | tee "$OUT_DIR/dense_w${W}.log"
done
echo "DONE $OUT_DIR"
