#!/usr/bin/env bash
set -e
# One kernel per msopprof process so each case is standalone.
cd "$(dirname "$0")"
source ~/env_ascend.sh
ulimit -n 1048576
ITERS="${ITERS:-64}"

run_one () {
  local kernel="$1" tag="$2" iters="$3"
  echo "=== CAModel $tag: $kernel iters=$iters ==="
  msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
    --launch-count=1 --timeout=5 \
    ./scalar_add_runner.sh scalar_add.o "$kernel" 1 32 1 "$iters" 0 2>&1 \
    | tee "camodel_${tag}.log"
}

for k in simd_add_ilp1 simd_add_ilp2 simd_add_ilp4 simd_add_ilp8 simd_add_ilp16 \
         simt_add_ilp1 simt_add_ilp2 simt_add_ilp4 simt_add_ilp8 simt_add_ilp16; do
  run_one "$k" "$k" "$ITERS"
done

# uniform add: ILP16 all-lane-same + same GM scalar address semantics
run_one simt_add_uniform_ilp16     simt_add_uniform_ilp16        "$ITERS"
run_one simt_add_uniform_same_addr simt_add_uniform_same_addr_o1  1
