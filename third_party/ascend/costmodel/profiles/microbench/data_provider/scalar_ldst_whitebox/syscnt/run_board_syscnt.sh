#!/usr/bin/env bash
set -eo pipefail
# Run the SYS_CNT probes on the real Ascend board (no msopprof wrapper).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ~/env_ascend.sh
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
HOST="$ROOT/syscnt/syscnt_rep_host"
OUT="$ROOT/syscnt/results/board"
mkdir -p "$OUT"
REPS="${REPS:-10}"

run_case () {
  local obj="$1"; shift
  for k in "$@"; do
    echo "=== board $(basename "$obj" .o) $k ==="
    "$HOST" "$obj" "$k" "$REPS" 1 > "$OUT/$(basename "$obj" .o)__$k.log" 2>&1
  done
}

run_case "$ROOT/load/scalar_o1/load_scalar_o1_syscnt.o" \
  simd_nop_syscnt simd_main_ld_o1_syscnt simt_nop_syscnt \
  simt_ld_uniform_o1_syscnt simt_ld_uniform_o1_t1_syscnt

run_case "$ROOT/load/scalar_o4/load_scalar_o4_syscnt.o" \
  simd_nop_syscnt simd_main_ld_same_o4_syscnt simd_main_ld_diff_o4_syscnt \
  simt_nop_syscnt simt_ld_uniform_same_o4_syscnt simt_ld_uniform_diff_o4_syscnt

run_case "$ROOT/store/scalar_o1/store_scalar_o1_syscnt.o" \
  simd_nop_syscnt simd_main_st_o1_syscnt simd_main_st_o1_dcci_syscnt \
  simd_main_st_o1_dcci_rd_syscnt simt_nop_syscnt simt_st_uniform_o1_syscnt \
  simt_st_uniform_o1_dcci_syscnt

run_case "$ROOT/store/scalar_o4/store_scalar_o4_syscnt.o" \
  simd_nop_syscnt simd_main_st_same_o4_syscnt simd_main_st_same_o4_dcci_syscnt \
  simd_main_st_diff_o4_syscnt simd_main_st_diff_o4_dcci_syscnt \
  simt_nop_syscnt simt_st_uniform_same_o4_syscnt simt_st_uniform_same_o4_dcci_syscnt \
  simt_st_uniform_diff_o4_syscnt simt_st_uniform_diff_o4_dcci_syscnt

run_case "$ROOT/load_vec_compute/scalar_o1_o4/load_vec_compute_syscnt.o" \
  lvc_nop_syscnt simd_lc_o1_syscnt simd_lc_o4_same_syscnt simd_lc_o4_diff_syscnt \
  simt_lc_o1_syscnt simt_lc_o4_same_syscnt simt_lc_o4_diff_syscnt

# compute ILP16: use the existing scalar_add_host (args: K nlane nwarp iters mode)
COMPUTE_HOST="$ROOT/compute/scalar_add/scalar_add_host"
for k in simd_add_nop_loop_syscnt simd_add_ilp16_syscnt \
         simt_add_nop_loop_syscnt simt_add_ilp16_syscnt simt_add_uniform_ilp16_syscnt; do
  for it in 1000 4000; do
    echo "=== board compute $k iters=$it ==="
    "$COMPUTE_HOST" "$ROOT/compute/scalar_add/scalar_add_syscnt.o" "$k" 1 32 1 "$it" 1 \
      > "$OUT/scalar_add_syscnt__${k}_it${it}.log" 2>&1
  done
done

echo "SYSCNT_BOARD_DONE"
