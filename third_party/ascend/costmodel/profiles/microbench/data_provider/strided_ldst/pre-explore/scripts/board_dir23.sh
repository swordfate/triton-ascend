#!/usr/bin/env bash
# Direction 2 supplement : int64 contiguous store (stride=1), plus a matched
#                          stride=2 set for the same blocks.
# Direction 3            : re-measure large-inner-stride points with a MUCH
#                          larger rotation window.  The standard window is
#                          step=8192 elements x mask=16383; when n*stride
#                          exceeds 8192 elements consecutive iterations reuse
#                          the same lines and the measured slope can be
#                          depressed by cache hits.
set -uo pipefail
PRE="$HOME/strided-template-calibration-v2"
cd "$PRE"
OUT="$PRE/results/dtype_calibration_v13_dir23/raw"
mkdir -p "$OUT"
sed 's/if stride < 2:/if stride < 1:/' scripts/dtype_stride_measure_probe.py > scripts/_dir23_probe.py

# ---- direction 2 supplement: int64 contiguous store -------------------------
for OP in store; do
  REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_dir23_probe.py \
    --op $OP --path simd --dtype int64 --blocks 4 8 16 32 64 128 256 512 1024 --strides 1 2 \
    --num-warps 1 --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
    --out "$OUT/d2_${OP}_simd_int64_W1.json" 2>&1 | grep -E "WROTE|Error" | tail -1
done
echo "D2_DONE $(date +%H:%M)"

# ---- direction 3: large inner stride, small vs large rotation window --------
# small window (the historical setting that may alias)
REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_dir23_probe.py \
  --op store --path simt_only --dtype float32 --blocks 64 256 --strides 32 128 512 1024 \
  --num-warps 4 --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
  --out "$OUT/d3_store_simt_f32_W4_winsmall.json" 2>&1 | grep -E "WROTE|Error" | tail -1
REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_dir23_probe.py \
  --op store --path simt_only --dtype float32 --blocks 64 256 --strides 32 128 512 1024 \
  --num-warps 4 --rotate-step 65536 --rotate-mask 65535 --reps 3 --max-attempts 4 \
  --out "$OUT/d3_store_simt_f32_W4_winbig.json" 2>&1 | grep -E "WROTE|Error" | tail -1
REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_dir23_probe.py \
  --op load --path simd --dtype float32 --blocks 64 256 --strides 32 128 512 1024 \
  --num-warps 1 --rotate-step 8192 --rotate-mask 16383 --reps 3 --max-attempts 4 \
  --out "$OUT/d3_load_simd_f32_W1_winsmall.json" 2>&1 | grep -E "WROTE|Error" | tail -1
REPS=3 MAX_ATTEMPTS=4 bash scripts/board_env.sh python scripts/_dir23_probe.py \
  --op load --path simd --dtype float32 --blocks 64 256 --strides 32 128 512 1024 \
  --num-warps 1 --rotate-step 65536 --rotate-mask 65535 --reps 3 --max-attempts 4 \
  --out "$OUT/d3_load_simd_f32_W1_winbig.json" 2>&1 | grep -E "WROTE|Error" | tail -1
echo "ALLDONE_DIR23 $(date +%H:%M)"
