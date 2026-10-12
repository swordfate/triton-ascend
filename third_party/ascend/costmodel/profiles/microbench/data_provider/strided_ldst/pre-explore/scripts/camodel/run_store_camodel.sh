#!/usr/bin/env bash
# CAModel simulator driver for the strided-store probe.
#
# Run on the Ascend server from this directory.  It uses the latest
# feature/simd-simt-compile-mode runtime assembled in ~/sb64_latest_env,
# falling back to the conda site-packages only for torch/torch_npu.
set -eo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_EXPLORE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE_EXPLORE"

# CANN set_env.sh accesses unset optional variables; keep -u disabled while sourcing.
set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$HOME/sb64_latest_env${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-$PWD/runs_store/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
chmod 700 "$OUT_DIR"
cp -f "$PRE_EXPLORE/scripts/strided_store_probe.py" "$OUT_DIR/"

BLOCK="${BLOCK:-32}"
MODES="${MODES:-simd simt_only}"
STRIDES="${STRIDES:-1 2 4 8 16 32 64 128 256}"
NUM_WARPS="${NUM_WARPS:-1}"
NUM_CASES=0
for _mode in $MODES; do
  for _stride in $STRIDES; do
    NUM_CASES=$((NUM_CASES + 1))
  done
done

LOG="${LOG:-$OUT_DIR/run.log}"
CACHE_DIR="${CACHE_DIR:-$OUT_DIR/triton_cache}"
rm -rf "$CACHE_DIR"
export TRITON_CACHE_DIR="$CACHE_DIR"

echo "=== strided store CAModel run ===" | tee "$LOG"
echo "OUT_DIR=$OUT_DIR" | tee -a "$LOG"
echo "BLOCK=$BLOCK MODES='$MODES' STRIDES='$STRIDES' NUM_WARPS=$NUM_WARPS NUM_CASES=$NUM_CASES" | tee -a "$LOG"
echo "PYTHONPATH=$PYTHONPATH" | tee -a "$LOG"
python3 -c 'import triton; print("triton=", triton.__file__, triton.__version__)' | tee -a "$LOG"

cd "$OUT_DIR"
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count="$NUM_CASES" --timeout=2880 \
  --kernel-name=strided_store_kernel \
  python3 strided_store_probe.py \
    --modes $MODES --strides $STRIDES --block "$BLOCK" \
    --num-warps "$NUM_WARPS" \
    --manifest launch_manifest.json --asm-dir asm 2>&1 | tee -a "$LOG"

echo "=== done: $OUT_DIR ===" | tee -a "$LOG"
find "$OUT_DIR" -maxdepth 1 -type d -name 'OPPROF_*' -print
