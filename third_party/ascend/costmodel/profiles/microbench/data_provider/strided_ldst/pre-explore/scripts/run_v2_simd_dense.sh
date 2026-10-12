#!/usr/bin/env bash
# Frequency-aware dense SIMD single-load measurement (protocol v2).
#
# Reuses frequency_retry_probe.py:
#   * npu-smi current-frequency logging,
#   * busy no-load warm-up when the core is throttled,
#   * retry when the matched no-load reference is abnormal,
#   * fallback normalization recorded in the JSON.
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
STRIDES=($(seq 1 32))
STRIDES+=(40 48 56 64 80 96 112 128 160 192 224 256)

mkdir -p results/board_v2_simd
python3 scripts/frequency_retry_probe.py \
  --blocks "${BLOCKS[@]}" --strides "${STRIDES[@]}" \
  --reps 10 --nbuf 10 --warmup-launches 300 --max-retries 2 \
  --out results/board_v2_simd/frequency_retry_results.json

echo DONE
