#!/usr/bin/env bash
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
# 板卡案例：f16, ISB=128 -> IS=64 elements;  4x16 rowpitch 4096B, 8x8 rowpitch 2048B
# CAModel 用 f32，等价元素数：4096/4=1024, 2048/4=512
OUT_DIR="$PRE/runs/is64_$(date +%H%M%S)"
MODES=simd SHAPES="4x16 8x8 8x16 16x16" INNER="64" OUT_DIR="$OUT_DIR" \
  bash scripts/run_stride2d_camodel.sh 2>&1 | tail -6
bash scripts/board_env.sh python scripts/analyze_stride2d_camodel.py "$OUT_DIR" \
  -o "$OUT_DIR/parsed.json" --output-csv "$OUT_DIR/parsed.csv" 2>&1 | tail -12
echo "OUT=$OUT_DIR"
