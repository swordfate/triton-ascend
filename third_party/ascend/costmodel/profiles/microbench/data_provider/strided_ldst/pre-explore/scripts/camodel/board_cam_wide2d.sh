#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/runs/wide2d_$(date +%H%M%S)"
MODES=simd SHAPES="4x16 8x8" INNER="1 2 3 6" OUT_DIR="$OUT" \
  bash scripts/run_stride2d_camodel.sh 2>&1 | tail -4
bash scripts/board_env.sh python scripts/analyze_stride2d_camodel.py "$OUT" \
  -o "$OUT/parsed.json" --output-csv "$OUT/parsed.csv" 2>&1 | tail -12
echo "OUT=$OUT"
