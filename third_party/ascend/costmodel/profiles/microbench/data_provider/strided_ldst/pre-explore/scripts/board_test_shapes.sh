#!/usr/bin/env bash
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
mkdir -p runs/newshapes
REPS=3 MAX_ATTEMPTS=3 bash scripts/board_env.sh python scripts/stride2d_measure_probe.py \
  --op store --path simt_only --dtype int8 \
  --shapes 1x16 2x8 2x16 8x4 16x4 16x8 8x32 16x32 32x8 2x32 1x32 4x32 \
  --row-strides 2 4 --inner-strides 4 8 --num-warps 4 \
  --rotate-step 8192 --rotate-mask 255 --reps 3 --max-attempts 3 \
  --out runs/newshapes/test.out 2>&1 | tail -6
echo "---"
python3 -c "
import json
d=json.load(open('runs/newshapes/test.out'))
print('cases:', len(d['cases']))
print('shapes seen:', sorted({(c['R'],c['C']) for c in d['cases']}))
print('valid:', sum(1 for c in d['cases'] if c.get('valid')), '/', len(d['cases']))
"
