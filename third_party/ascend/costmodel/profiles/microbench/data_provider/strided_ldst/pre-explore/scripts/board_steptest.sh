#!/usr/bin/env bash
# Decisive test: is the rotate STEP (in ELEMENTS) confounding the dtype
# comparison?  Hold everything fixed except --rotate-step.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
echo "--- W2 campaign state ---"
tmux ls 2>&1 | head -3
ls "$PRE/w2full_done.txt" 2>/dev/null && echo "W2 DONE" || echo "W2 running"
echo ""
echo "=== int8 store simd b64 s2: 变 rotate-step（元素单位）==="
for st in 256 1024 4096 8192 32768; do
  r=$(REPS=2 MAX_ATTEMPTS=2 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
      --op store --path simd --dtype int8 --blocks 64 --strides 2 --num-warps 1 \
      --rotate-step $st --rotate-mask 63 --reps 2 --max-attempts 2 \
      --out /tmp/steptest_int8_$st.json 2>&1 | grep -E "^store" | tail -1)
  echo "  step=$st (字节 $((st*1))) : $r"
done
echo ""
echo "=== int64 store simd b64 s2: 同样的元素步长 ==="
for st in 256 1024 4096 8192 32768; do
  r=$(REPS=2 MAX_ATTEMPTS=2 bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py \
      --op store --path simd --dtype int64 --blocks 64 --strides 2 --num-warps 1 \
      --rotate-step $st --rotate-mask 63 --reps 2 --max-attempts 2 \
      --out /tmp/steptest_int64_$st.json 2>&1 | grep -E "^store" | tail -1)
  echo "  step=$st (字节 $((st*8))) : $r"
done
