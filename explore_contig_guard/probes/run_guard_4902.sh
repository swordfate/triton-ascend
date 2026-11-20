#!/bin/bash
# Guard prediction-only rerun with the 4902ae4a baseline wheel (0868cdae).
export PATH=/home/c00946898/bishengir-install/bin:$PATH
export LD_LIBRARY_PATH=/home/c00946898/bishengir-install/lib:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/python/site-packages:/usr/local/Ascend/cann-9.1.0/python/site-packages
export ASCEND_RT_VISIBLE_DEVICES=1
ulimit -n 1048576
cd /root/work/triton-ascend/explore_contig_guard/probes
rm -rf /tmp/GUARD_4902
sed 's/torch.npu.set_device(0)/torch.npu.set_device(0)/; s|/tmp/GUARD_20261119|/tmp/GUARD_4902|' guard_pred_20261119.py > guard_pred_4902.py
python3 guard_pred_4902.py
