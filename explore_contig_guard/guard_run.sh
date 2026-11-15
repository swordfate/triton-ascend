#!/bin/bash
# guard_run.sh <DEVICE> <TAG> [CASES...]
set -u
DEV=$1; TAG=$2; shift 2
export ASCEND_RT_VISIBLE_DEVICES=$DEV
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PATH=/home/c00946898/bishengir-install/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/bishengir-install/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages
export TRITON_CACHE_DIR=/tmp/GUARD_CONTIG/cache_$TAG
rm -rf "$TRITON_CACHE_DIR"
ulimit -n 1048576
cd /root/work/explore_contig
mkdir -p /tmp/GUARD_CONTIG
OUT=/tmp/GUARD_CONTIG/guard_table_$TAG.json
if [ $# -gt 0 ]; then
  python3 bench_guard_contig.py --out "$OUT" --cases "$@"
else
  python3 bench_guard_contig.py --out "$OUT"
fi
echo GUARD_RUN_DONE_$TAG
