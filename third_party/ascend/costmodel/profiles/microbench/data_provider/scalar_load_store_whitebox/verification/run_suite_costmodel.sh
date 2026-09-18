#!/usr/bin/env bash
# Costmodel report for one tiny padded kernel: run_suite_costmodel.sh <kernel>
set -eo pipefail
KERNEL="${1:?usage: $0 padded_copy_gather|padded_copy_scatter|padded_copy_wgrad}"
OUT=/home/c00946898/stage_vs_camodel_20260917
source /home/c00946898/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576

export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_ASCEND_COMPILE_MODE=simd_simt
export TRITON_ASCEND_AUTO_SIMT_SCOPE=report
export TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP="$OUT/costmodel_${KERNEL}.json"
rm -f "$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
export COSTMODEL_LOG_LEVEL=2
export COSTMODEL_LOG_FILE="$OUT/costmodel_${KERNEL}.log"
rm -f "$COSTMODEL_LOG_FILE"
export TRITON_CACHE_DIR="$OUT/cache_costmodel_${KERNEL}"
rm -rf "$TRITON_CACHE_DIR"

cd "$OUT"
python3 tiny_padded_suite_runner.py --kernel "$KERNEL" --mode costmodel
echo "REPORT=$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
