#!/usr/bin/env bash
# Supplementary block sweep for the semi-white-box/black-box fit.
# Adds intermediate/large BLOCK points so the black-box surface is not built
# from only five block sizes.
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

SIMD_BLOCKS=(8 24 48 96 192 384 512)
# Triton SIMT lowering requires tl.arange(0, N) to have a power-of-two range,
# so the non-power-of-two SIMD blocks above cannot be measured natively in
# simt_only mode without a masked padding kernel.  The SIMT supplement
# therefore only adds native power-of-two block points (the base v2 matrix
# already contains 16/32/64/128/256).
SIMT_BLOCKS=(4 8 512)
STRIDES_SIMD=($(seq 1 32))
STRIDES_SIMD+=(40 48 56 64 80 96 112 128 160 192 224 256)
STRIDES_SIMT=(1 2 3 4 5 6 7 8 9 10 11 12 16 20 24 32 40 48 64 96 128 192 256)

mkdir -p results/board_v2_simd_more_blocks results/board_v2_simt_more_blocks
if [ ! -f results/board_v2_simd_more_blocks/frequency_retry_results.json ]; then
  python3 scripts/frequency_retry_probe.py \
    --blocks "${SIMD_BLOCKS[@]}" --strides "${STRIDES_SIMD[@]}" \
    --reps 10 --nbuf 10 --warmup-launches 300 --max-retries 2 \
    --out results/board_v2_simd_more_blocks/frequency_retry_results.json
else
  echo "skip existing results/board_v2_simd_more_blocks/frequency_retry_results.json"
fi

python3 scripts/simt_frequency_retry_probe.py \
  --blocks "${SIMT_BLOCKS[@]}" --strides "${STRIDES_SIMT[@]}" \
  --iters 2000 8000 --reps 2 --rotate-step 8192 --rotate-mask 16383 \
  --warmup-launches 1000 --max-retries 2 \
  --out results/board_v2_simt_more_blocks/simt_frequency_retry_results.json

echo DONE
