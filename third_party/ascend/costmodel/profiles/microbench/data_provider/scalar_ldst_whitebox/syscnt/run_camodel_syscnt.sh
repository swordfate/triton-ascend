#!/usr/bin/env bash
set -eo pipefail
# Run the SYS_CNT probes under CAModel, one kernel per msopprof process.
# Running each kernel alone avoids the SIMT VF run-order sensitivity.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/syscnt"
source ~/env_ascend.sh
ulimit -n 1048576
OUT="$ROOT/syscnt/results/camodel_ind_all"
mkdir -p "$OUT"

run_one () {
  local obj="$1" k="$2" tag
  tag="$(basename "$obj" .o)__$k"
  echo "=== CAModel individual $tag ==="
  msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
    --launch-count=1 --timeout=5 \
    ./syscnt_runner.sh "$obj" "$k" > "$OUT/$tag.log" 2>&1
}

O1="$ROOT/load/scalar_o1/load_scalar_o1_syscnt.o"
O4="$ROOT/load/scalar_o4/load_scalar_o4_syscnt.o"
S1="$ROOT/store/scalar_o1/store_scalar_o1_syscnt.o"
S4="$ROOT/store/scalar_o4/store_scalar_o4_syscnt.o"

# nop baselines (also used by board runs; keep the same object/kernel names)
for obj in "$O1" "$O4" "$S1" "$S4"; do
  run_one "$obj" simd_nop_syscnt
  run_one "$obj" simt_nop_syscnt
done

run_one "$O1" simd_main_ld_o1_syscnt
run_one "$O1" simt_ld_uniform_o1_syscnt
run_one "$O1" simt_ld_uniform_o1_t1_syscnt
run_one "$O4" simd_main_ld_same_o4_syscnt
run_one "$O4" simd_main_ld_diff_o4_syscnt
run_one "$O4" simt_ld_uniform_same_o4_syscnt
run_one "$O4" simt_ld_uniform_diff_o4_syscnt
run_one "$S1" simd_main_st_o1_syscnt
run_one "$S1" simd_main_st_o1_dcci_syscnt
run_one "$S1" simd_main_st_o1_dcci_rd_syscnt
run_one "$S1" simt_st_uniform_o1_syscnt
run_one "$S1" simt_st_uniform_o1_dcci_syscnt
run_one "$S4" simd_main_st_same_o4_syscnt
run_one "$S4" simd_main_st_same_o4_dcci_syscnt
run_one "$S4" simd_main_st_diff_o4_syscnt
run_one "$S4" simd_main_st_diff_o4_dcci_syscnt
run_one "$S4" simt_st_uniform_same_o4_syscnt
run_one "$S4" simt_st_uniform_same_o4_dcci_syscnt
run_one "$S4" simt_st_uniform_diff_o4_syscnt
run_one "$S4" simt_st_uniform_diff_o4_dcci_syscnt

LVC="$ROOT/load_vec_compute/scalar_o1_o4/load_vec_compute_syscnt.o"
run_one "$LVC" lvc_nop_syscnt
run_one "$LVC" simd_lc_o1_syscnt
run_one "$LVC" simd_lc_o4_same_syscnt
run_one "$LVC" simd_lc_o4_diff_syscnt
run_one "$LVC" simt_lc_o1_syscnt
run_one "$LVC" simt_lc_o4_same_syscnt
run_one "$LVC" simt_lc_o4_diff_syscnt

# compute ILP16: two iters points per kernel, fit slope later
COUT="$ROOT/syscnt/results/camodel_compute"
mkdir -p "$COUT"
COMPUTE_OBJ="$ROOT/compute/scalar_add/scalar_add_syscnt.o"
for k in simd_add_nop_loop_syscnt simd_add_ilp16_syscnt \
         simt_add_nop_loop_syscnt simt_add_ilp16_syscnt simt_add_uniform_ilp16_syscnt; do
  # 100/400 keeps the CAModel serial run short; the big board points are
  # covered by run_board_syscnt.sh.
  for it in 100 400; do
    echo "=== CAModel compute $k iters=$it ==="
    msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
      --launch-count=1 --timeout=5 \
      ./syscnt_compute_runner.sh "$COMPUTE_OBJ" "$k" 1 32 1 "$it" 1 \
      > "$COUT/${k}_it${it}.log" 2>&1
  done
done
echo "SYSCNT_CAMODEL_DONE"
