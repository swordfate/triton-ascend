#!/usr/bin/env bash
PRE="$HOME/strided-template-calibration-v2"
RUN=$(ls -d "$PRE"/runs/shapes_* | tail -1)
echo "RUN=$RUN"
cd "$PRE"
bash scripts/board_env.sh python scripts/analyze_stride2d_camodel.py "$RUN" \
  -o "$RUN/parsed.json" --output-csv "$RUN/parsed.csv" 2>&1 | tail -40
