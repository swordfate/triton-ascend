#!/usr/bin/env bash
# Common board-side environment wrapper for store probes.
#
# Usage:
#   bash scripts/board_env.sh python scripts/<probe>.py <args...>
#
# This intentionally does NOT set PYTHONPATH to the assembled sb64 env: the
# board-validation runtime is the installed wj_autoscope site-packages build.
set -eo pipefail
set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Preserve the CANN site-packages entries already exported by set_env.sh.
# Do NOT unset PYTHONPATH: torch_npu custom-kernel launch needs them on A5.
exec "$@"
