#!/usr/bin/env bash
# CAModel simulator driver for the rank-2 strided-load probe.
set -eo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_EXPLORE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE_EXPLORE"

set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$HOME/sb64_latest_env${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-$PWD/runs/stride2d_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"; chmod 700 "$OUT_DIR"
cp -f "$PRE_EXPLORE/scripts/stride2d_camodel_probe.py" "$OUT_DIR/"

MODES="${MODES:-simd}"
SHAPES="${SHAPES:-1x64 8x16 16x32}"
INNER="${INNER:-3}"
NUM_CASES=0
for _m in $MODES; do for _s in $SHAPES; do for _k in $INNER; do
  NUM_CASES=$((NUM_CASES + 2))
done; done; done

LOG="$OUT_DIR/run.log"
export TRITON_CACHE_DIR="$OUT_DIR/triton_cache"
echo "=== rank-2 strided load CAModel run ===" | tee "$LOG"
echo "OUT_DIR=$OUT_DIR NUM_CASES=$NUM_CASES MODES=$MODES SHAPES=$SHAPES INNER=$INNER" | tee -a "$LOG"
python3 -c 'import triton; print("triton=", triton.__file__, triton.__version__)' | tee -a "$LOG"

cd "$OUT_DIR"
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count="$NUM_CASES" --timeout=2880 \
  --kernel-name=stride2d_load_kernel \
  python3 stride2d_camodel_probe.py \
    --modes $MODES --shapes $SHAPES --inner-strides $INNER \
    --manifest launch_manifest.json --asm-dir asm 2>&1 | tee -a "$LOG"
echo "=== done: $OUT_DIR ===" | tee -a "$LOG"
find "$OUT_DIR" -maxdepth 1 -type d -name 'OPPROF_*' -print
