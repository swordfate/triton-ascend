#!/usr/bin/env bash
# Frequency-aware dense SIMT Event-slope measurement (protocol v2).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_EXPLORE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE_EXPLORE"
set +u
source ~/env_ascend.sh >/dev/null 2>&1
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u

export PYTHONPATH="$HOME/sb64_latest_env${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

BLOCKS=(16 32 64 128 256)
STRIDES=(1 2 3 4 5 6 7 8 9 10 11 12 16 20 24 32 40 48 64 96 128 192 256)

mkdir -p results/board_v2_simt
python3 scripts/simt_frequency_retry_probe.py \
  --blocks "${BLOCKS[@]}" --strides "${STRIDES[@]}" \
  --iters 2000 8000 --reps 2 \
  --rotate-step 8192 --rotate-mask 16383 \
  --warmup-launches 1000 --max-retries 2 \
  --out results/board_v2_simt/simt_frequency_retry_results.json

echo DONE
