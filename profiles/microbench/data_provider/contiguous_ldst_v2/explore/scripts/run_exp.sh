#!/bin/bash
# Explore-experiment runner wrapper (37 server).
# usage: bash run_exp.sh <script.py> <card> <out.jsonl> [script args...]
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export PATH=/home/c00946898/bishengir-install/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/bishengir-install/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages
export ASCEND_RT_VISIBLE_DEVICES=$2
ulimit -n 1048576
mkdir -p /root/work/explore_contig/results
cd /root/work/explore_contig
python3.11 "/root/work/explore_contig/$1" --out "/root/work/explore_contig/results/$3" "${@:4}"
