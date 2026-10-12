#!/usr/bin/env bash
# Board wrapper that prefers the assembled sb64_latest_env Triton runtime while
# preserving CANN/torch_npu PYTHONPATH entries.
set -eo pipefail
set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$HOME/sb64_latest_env${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
