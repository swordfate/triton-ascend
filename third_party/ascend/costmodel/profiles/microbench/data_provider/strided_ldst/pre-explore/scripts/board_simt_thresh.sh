#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/simtth
# 找断点：int8 W=4 n=64, ISB 从 2 扫到 20
REPS=4 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
  --op store --path simt_only --dtype int8 --blocks 64 \
  --strides 2 3 4 5 6 7 8 9 10 12 14 16 20 --num-warps 4 \
  --rotate-step 8192 --rotate-mask 16383 --reps 4 --max-attempts 4 \
  --out runs/simtth/int8_W4.out 2>&1 | grep -E "^store|WROTE" | tail -16
echo "--- 再验 dtype 依赖 (W=4,n=64,ISB=8/16) ---"
for DT in float16 float32 int64; do
  REPS=4 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
    --op store --path simt_only --dtype $DT --blocks 64 \
    --strides 4 8 16 --num-warps 4 \
    --rotate-step 8192 --rotate-mask 16383 --reps 4 --max-attempts 4 \
    --out runs/simtth/${DT}_W4.out 2>&1 | grep -E "^store" | tail -3
done
