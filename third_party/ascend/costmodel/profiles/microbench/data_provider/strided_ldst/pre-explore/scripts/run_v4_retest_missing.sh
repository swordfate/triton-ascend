#!/usr/bin/env bash
# 5-allocation retest for the 240 missing W=1/2/4/8/16 large-stride cases.
set -eo pipefail
umask 077
cd "$(dirname "$0")/.."

ASCEND_RT_VISIBLE_DEVICES=1 bash scripts/board_env.sh \
  python3 scripts/retest_strided_simt_cases.py \
  --cases results/v4_missing_cases.json \
  --out results/v4_retest_23_missing.json \
  --allocations 5 \
  --reps 3 \
  --freq-device 1
echo "RETEST DONE $(date -Is)"
