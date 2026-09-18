#!/usr/bin/env bash
# CAModel run for one tiny padded kernel/mode:
#   run_suite_camodel.sh <kernel> <simt_only|simd>
set -eo pipefail
KERNEL="${1:?usage: $0 padded_copy_gather|padded_copy_scatter|padded_copy_wgrad SIMT_ONLY_OR_SIMD}"
MODE="${2:?usage: $0 KERNEL SIMT_ONLY_OR_SIMD}"
OUT=/home/c00946898/stage_vs_camodel_20260917
source /home/c00946898/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576

export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_ASCEND_COMPILE_MODE="$MODE"
export TRITON_ASCEND_AUTO_SIMT_SCOPE=off
unset TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP
export TRITON_CACHE_DIR="$OUT/cache_camodel_${KERNEL}_${MODE}"
rm -rf "$TRITON_CACHE_DIR"

if [ "$MODE" = "simt_only" ]; then
  PY_MODE=camodel_simt_only
else
  PY_MODE=camodel_simd
fi
KERNEL_PREFIX="_${KERNEL}"

cd "$OUT"
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=1 --timeout=10 --kernel-name="$KERNEL_PREFIX" \
  python3 tiny_padded_suite_runner.py --kernel "$KERNEL" --mode "$PY_MODE" 2>&1 | tee "camodel_${KERNEL}_${MODE}.log"
