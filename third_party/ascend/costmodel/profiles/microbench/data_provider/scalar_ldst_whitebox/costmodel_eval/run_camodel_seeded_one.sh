#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
KERNEL="${1:?usage: $0 KERNEL MODE}"
MODE="${2:?usage: $0 KERNEL MODE}"
case "$KERNEL" in
  padded_copy_gather) PREFIX=_padded_copy_gather ;;
  padded_copy_scatter) PREFIX=_padded_copy_scatter ;;
  padded_copy_wgrad) PREFIX=_padded_copy_wgrad ;;
  binned_copy_gather) PREFIX=_binned_copy_gather ;;
  binned_copy_scatter) PREFIX=_binned_copy_scatter ;;
  binned_copy_wgrad) PREFIX=_binned_copy_wgrad ;;
  *) echo "unknown kernel $KERNEL" >&2; exit 2 ;;
esac
OUT="$PWD/out"
mkdir -p "$OUT"
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_ASCEND_COMPILE_MODE="$MODE"
export TRITON_ASCEND_AUTO_SIMT_SCOPE=off
unset TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP || true
export TRITON_CACHE_DIR="$OUT/cache_camodel_${KERNEL}_${MODE}"
rm -rf "$TRITON_CACHE_DIR"

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=1 --timeout=20 --kernel-name="$PREFIX" \
  python3 run_scalar_dominate_seeded.py --kernel "$KERNEL" --seed "${SEED:-0}" 2>&1 | tee "$OUT/camodel_${KERNEL}_${MODE}.log"
