#!/usr/bin/env bash
# CAModel white-box for the pure-SIMT strided load, targeting the regime where
# the semi-white-box formula has its worst error:
#   * error grows with block      (b4 9.7% -> b512 27.0%)
#   * worst num_warps are 2 / 1 / 64, best are 8 / 16
#   * worst strides are 64 / 128 / 256, best are 10 / 11 / 12 / 24 / 40 / 48
# So we sweep stride at block=32, W=2 (the bad regime) plus a W=16 control.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"
set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$HOME/sb64_latest_env${PYTHONPATH:+:$PYTHONPATH}"

run_one() {  # <outdir> <num_warps> <strides...>
  local OUT="$1"; shift
  local W="$1"; shift
  mkdir -p "$OUT"; chmod 700 "$OUT"
  cd "$OUT"
  local N=$#
  echo "=== CAModel SIMT load: block=32 num_warps=$W strides=$* ($N kernels) ==="
  TRITON_CACHE_DIR="$OUT/triton_cache" msopprof simulator \
    --soc-version=Ascend950PR_9599 --core-id=0 \
    --launch-count="$N" --timeout=2880 \
    --kernel-name=strided_load_kernel \
    python3 "$PRE/scripts/strided_load_probe.py" \
      --modes simt_only --strides "$@" --block 32 --num-warps "$W" \
      --manifest launch_manifest.json --asm-dir asm 2>&1 | tail -5
  ls -d OPPROF_* 2>/dev/null || true
}

RUNS="${RUNS:-$PRE/runs/simt_camodel}"
run_one "$RUNS/w2" 2 2 4 8 16 32
run_one "$RUNS/w16" 16 2 8 32
echo "=== simt camodel done: $RUNS ==="
