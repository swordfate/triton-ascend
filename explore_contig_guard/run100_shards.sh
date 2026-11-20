#!/bin/bash
# run100_shards.sh <DEVICE> <SHARD_LIST> <OUT_ROOT>
# e.g. bash run100_shards.sh 3 "0,2,4,6" /tmp/RUN100V2C3
set -u
DEV=$1; SHARDS=$2; OUTROOT=$3
export ASCEND_RT_VISIBLE_DEVICES=$DEV
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
# set_env puts CANN's own bishengir first; re-export the pinned one AFTER
export PATH=/home/c00946898/bishengir-install/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/bishengir-install/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages
ulimit -n 1048576
cd /root/w/cc-harness/costmodel_generalization
mkdir -p "$OUTROOT"
IFS=',' read -ra SL <<< "$SHARDS"
for s in "${SL[@]}"; do
  export TRITON_CACHE_DIR="$OUTROOT/cache_s$s"
  rm -rf "$TRITON_CACHE_DIR"
  echo "=== shard $s on device $DEV ==="
  python3 run_generalization_matrix.py \
    --manifest generalization_matrix_v140_all_100.json \
    --output-root "$OUTROOT" \
    --profile /usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json \
    --expected-bishengir /home/c00946898/bishengir-install/bin/bishengir-compile \
    --warmup 3 --repeats 7 \
    --measurement-processes 1 --disable-profiler \
    --shard-index "$s" --shard-count 8 \
    > "$OUTROOT/shard$s.out" 2>&1
  echo "shard $s rc=$?"
done
echo "SHARDS_DONE $OUTROOT"
