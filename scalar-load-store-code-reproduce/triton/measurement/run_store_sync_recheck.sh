#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0
python3 store_probe_variants.py --modes simd simt_only --ks 1 2 4 8 \
  --variants 1 2 --rounds 6 --reps 80 --warmup 5 --device 0 \
  --out results_sync_recheck 2>&1 | tee run_sync_recheck.log
