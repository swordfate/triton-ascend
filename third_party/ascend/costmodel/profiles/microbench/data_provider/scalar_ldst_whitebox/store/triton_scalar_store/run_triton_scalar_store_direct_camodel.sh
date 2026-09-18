#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_ASCEND_COMPILE_MODE=simd
export TRITON_ASCEND_AUTO_SIMT_SCOPE=off
unset TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP

MODE="${MODE:-const}"
GRID="${GRID:-4}"
LOG="${LOG:-run_camodel_${MODE}.log}"
export TRITON_CACHE_DIR="$PWD/cache_${MODE}"
rm -rf "$TRITON_CACHE_DIR"

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=1 --timeout=10 --kernel-name="_triton_scalar_store_${MODE}" \
  python3 triton_scalar_store_direct_demo.py --mode "$MODE" --grid "$GRID" 2>&1 | tee "$LOG"
