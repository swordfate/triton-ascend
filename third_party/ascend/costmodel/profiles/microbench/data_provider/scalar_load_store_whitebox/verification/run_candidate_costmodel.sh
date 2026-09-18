#!/usr/bin/env bash
# run_candidate_costmodel.sh <kernel> <profile_json> <tag>
set -eo pipefail
KERNEL="${1:?kernel}"
PROFILE="$(realpath "${2:?profile}")"
TAG="${3:?tag}"
OUT=/home/c00946898/stage_vs_camodel_20260917
source /home/c00946898/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576

export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_ASCEND_COMPILE_MODE=simd_simt
export TRITON_ASCEND_AUTO_SIMT_SCOPE=report
export TRITON_ASCEND_AUTO_SIMT_PROFILE="$PROFILE"
export TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP="$OUT/costmodel_${KERNEL}__${TAG}.json"
rm -f "$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
export COSTMODEL_LOG_LEVEL=0
unset COSTMODEL_LOG_FILE
export TRITON_CACHE_DIR="$OUT/cache_candidate_${TAG}_${KERNEL}"
rm -rf "$TRITON_CACHE_DIR"

cd "$OUT"
python3 tiny_padded_suite_runner.py --kernel "$KERNEL" --mode costmodel > "candidate_${TAG}_${KERNEL}.run.log" 2>&1
echo "REPORT=$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
