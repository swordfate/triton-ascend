#!/usr/bin/env bash
# Run one alignment geometry under the CAModel simulator and keep the trace.
#
#   bash run_align_camodel.sh <rows> <cols> [dtype]
#
# Output: ~/align_camodel/r<rows>_c<cols>/OPPROF_*/
#   dump/core0.mte_log.dump              DMA instructions with burst parameters
#   simulator/core0.veccore0/*_instr_exe.csv   per-instruction cycles
#
# PITFALL - do not add SIMLIB / BISHLIB to LD_LIBRARY_PATH here.  The Triton
# ascend driver compiles a small helper extension at import time with hard-coded
# include paths from another install; polluting LD_LIBRARY_PATH makes that build
# fail, and then the kernel never executes inside the simulator while msprof
# still writes ~1 GB of empty *_dump files.  The only reliable check that the
# kernel really ran is the ALIGN_RUN_OK line plus 'Profiling data parse
# finished' in the log.
set -u
source "$HOME/env_ascend.sh" >/dev/null 2>&1
source /data/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wj_autoscope 2>/dev/null || true
ulimit -n 1048576

# resolve the runner location before any cd
HERE=$(cd "$(dirname "$0")" && pwd)
ROWS="${1:?usage: run_align_camodel.sh <rows> <cols> [dtype]}"
COLS="${2:?usage: run_align_camodel.sh <rows> <cols> [dtype]}"
DTYPE="${3:-fp32}"
TAG="r${ROWS}_c${COLS}"
DIR="$HOME/align_camodel/$TAG"
SOC="${SOC_VERSION:-Ascend950PR_9599}"

rm -rf "$DIR"; mkdir -p "$DIR"
export TRITON_CACHE_DIR="$DIR/cache"
cd "$DIR" || exit 1

echo "== $TAG (rows=$ROWS cols=$COLS dtype=$DTYPE soc=$SOC) =="
timeout 900 msprof op simulator --soc-version="$SOC" \
    --kernel-name=align_kernel --launch-count=1 --timeout=900 \
    python3 "$HERE/align_camodel.py" "$ROWS" "$COLS" "$DTYPE" \
    > msprof.log 2>&1
rc=$?
echo "rc=$rc"
grep -q ALIGN_RUN_OK msprof.log && echo "kernel ran inside the simulator" \
    || echo "WARNING: no ALIGN_RUN_OK - the kernel did NOT execute; the dumps are empty"
OPPROF=$(ls -d OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF" ]; then
  echo "trace: $DIR/$OPPROF"
  echo "--- DMA instructions (burst parameters) ---"
  grep -E "burst_num|MOV_SRC_TO_DST" "$OPPROF/dump/core0.mte_log.dump" 2>/dev/null | head -8
  echo "--- bus request sizes ---"
  grep -oE "size: [0-9]+" "$OPPROF/dump/core0.mte_log.dump" 2>/dev/null \
      | sort | uniq -c | sort -rn | head -5
fi
