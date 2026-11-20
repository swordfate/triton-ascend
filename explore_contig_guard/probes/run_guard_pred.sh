#!/bin/bash
# Guard prediction-only rerun on card 1 with the final gate wheel.
export PATH=/home/c00946898/bishengir-install/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/bishengir-install/lib:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages
export ASCEND_RT_VISIBLE_DEVICES=1
ulimit -n 1048576
cd /root/work/triton-ascend/explore_contig_guard/probes
rm -rf /tmp/GUARD_GATE
sed 's/torch.npu.set_device(0)/torch.npu.set_device(0)/; s|/tmp/GUARD_20261119|/tmp/GUARD_GATE|' guard_pred_20261119.py > guard_pred_gate.py
python3 guard_pred_gate.py
