#!/usr/bin/env bash
# v2 in-sample calibration driver for the SIMT local-scope strided
# triton_stride_load/store template path.
#
# All measurements run on NPU0 through scripts/board_env.sh, which uses the
# installed wj_autoscope package and preserves the CANN PYTHONPATH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PRE"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PRE/triton_cache_template_v2}"
mkdir -p results/model_template_stride_v2/raw results/model_template_stride_v2/ir_evidence/measure_asm
mkdir -p "$TRITON_CACHE_DIR"

PASS="${PASS:-all}"
if [[ "$PASS" == "all" || "$PASS" == "path" ]]; then
  echo "===== path check $(date -Is) ====="
  bash scripts/board_env.sh python scripts/template_stride_path_check.py \
    --blocks 4 8 16 32 64 128 256 512 1024 2048 \
    --strides 3 5 6 7 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
              1 2 4 8 16 32 64 128 256 \
    --num-warps 32 \
    --out results/model_template_stride_v2/ir_evidence/template_stride_path_check_v2.json \
    --asm-dir results/model_template_stride_v2/ir_evidence/path_check_asm \
    2>&1 | tee results/model_template_stride_v2/ir_evidence/path_check_v2.log
fi

if [[ "$PASS" == "all" || "$PASS" == "load" ]]; then
  echo "===== load pass1 $(date -Is) ====="
  REPS=2 MAX_ATTEMPTS=2 \
  OUT_JSON=results/model_template_stride_v2/raw/board_template_stride_load_pass1.json \
    bash scripts/run_template_stride_measure_load.sh \
    2>&1 | tee results/model_template_stride_v2/raw/load_pass1.log
  echo "===== load pass2 $(date -Is) ====="
  REPS=3 MAX_ATTEMPTS=1 \
  OUT_JSON=results/model_template_stride_v2/raw/board_template_stride_load_pass2.json \
    bash scripts/run_template_stride_measure_load.sh \
    2>&1 | tee results/model_template_stride_v2/raw/load_pass2.log
fi

if [[ "$PASS" == "all" || "$PASS" == "store" ]]; then
  echo "===== store pass1 $(date -Is) ====="
  REPS=2 MAX_ATTEMPTS=2 \
  OUT_JSON=results/model_template_stride_v2/raw/board_template_stride_store_pass1.json \
    bash scripts/run_template_stride_measure_store.sh \
    2>&1 | tee results/model_template_stride_v2/raw/store_pass1.log
  echo "===== store pass2 $(date -Is) ====="
  REPS=3 MAX_ATTEMPTS=1 \
  OUT_JSON=results/model_template_stride_v2/raw/board_template_stride_store_pass2.json \
    bash scripts/run_template_stride_measure_store.sh \
    2>&1 | tee results/model_template_stride_v2/raw/store_pass2.log
fi

echo "===== calibration v2 driver done $(date -Is) ====="
