#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
KERNEL="${1:?usage: $0 KERNEL}"
OUT="$PWD/out"
mkdir -p "$OUT"
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_ASCEND_COMPILE_MODE=simd_simt
export TRITON_ASCEND_AUTO_SIMT_SCOPE=report
export TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP="$OUT/costmodel_${KERNEL}.json"
export TRITON_ASCEND_AUTO_SIMT_PROFILE="$PWD/profile.json"
rm -f "$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
export COSTMODEL_LOG_LEVEL=2
export COSTMODEL_LOG_FILE="$OUT/costmodel_${KERNEL}.log"
rm -f "$COSTMODEL_LOG_FILE"
export TRITON_CACHE_DIR="$OUT/cache_costmodel_${KERNEL}"
rm -rf "$TRITON_CACHE_DIR"

python3 run_scalar_dominate_one.py --kernel "$KERNEL"
echo "REPORT=$TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"
