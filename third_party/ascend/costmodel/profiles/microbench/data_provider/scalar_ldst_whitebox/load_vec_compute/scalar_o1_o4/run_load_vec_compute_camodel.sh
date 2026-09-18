#!/usr/bin/env bash
set -eo pipefail
# Run one kernel per msopprof process so each case is cold and standalone.
# On the shared server, launch this script through tmux; see 01 guide §10.2.
cd "$(dirname "$0")"
source ~/env_ascend.sh
ulimit -n 1048576

run_one () {
  local kernel="$1" tag="$2"
  echo "=== CAModel $tag: $kernel ==="
  msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
    --launch-count=1 --timeout=5 \
    ./load_vec_compute_runner.sh load_vec_compute.o "$kernel" \
    > "camodel_${tag}.log" 2>&1
}

run_one simd_lc_o1       simd_lc_o1
run_one simd_lc_o4_same  simd_lc_o4_same
run_one simd_lc_o4_diff  simd_lc_o4_diff
run_one simt_lc_o1       simt_lc_o1
run_one simt_lc_o4_same  simt_lc_o4_same
run_one simt_lc_o4_diff  simt_lc_o4_diff
