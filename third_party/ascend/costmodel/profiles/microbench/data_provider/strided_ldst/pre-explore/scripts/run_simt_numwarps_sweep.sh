#!/usr/bin/env bash
# Coarse num_warps sweep for the SIMT strided-load target (v2 frequency-aware protocol).
set -eo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_EXPLORE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE_EXPLORE"

# CANN set_env.sh touches optional unset variables; source with -u disabled.
set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
OUT_DIR="${OUT_DIR:-$PWD/results/board_v2_simt_warps}"
mkdir -p "$OUT_DIR"

python3 scripts/simt_frequency_retry_probe.py \
  --blocks 32 64 128 256 \
  --strides 1 4 8 16 32 64 128 256 \
  --num-warps 1 2 4 8 16 32 64 \
  --iters 2000 8000 \
  --reps 2 \
  --rotate-step 8192 \
  --rotate-mask 16383 \
  --warmup-launches 2000 \
  --max-retries 3 \
  --retry-sleep 2.0 \
  --out "$OUT_DIR/frequency_retry_results.json" 2>&1 | tee "$OUT_DIR/run.log"

echo "WROTE $OUT_DIR/frequency_retry_results.json" | tee -a "$OUT_DIR/run.log"
