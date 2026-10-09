#!/usr/bin/env bash
# Wait for the in-flight v4_retest_all job, then run the missing 240-case
# aligned + retest passes, merge the retest JSONs, build the v4 dataset and
# fit the v8-simple model.  All stages are logged separately.
set -eo pipefail
umask 077
cd "$(dirname "$0")/.."

set +u
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
set -u
ulimit -n 1048576

ROOT="$PWD"
echo "=== v4 followup start $(date -Is)"
while true; do
  n=$(grep -c '^\[' v4_retest_all_outer.log || true)
  if [ -s results/v4_retest_all.json ] && [ "$n" -ge 1048 ]; then
    echo "retest_all complete: n=$n $(date -Is)"
    break
  fi
  echo "waiting retest_all: n=$n $(date -Is)"
  sleep 60
done

bash scripts/run_v4_aligned_23.sh > v4_aligned_23_outer.log 2>&1
echo "=== aligned_23 done $(date -Is)"
bash scripts/run_v4_retest_missing.sh > v4_retest_23_missing_outer.log 2>&1
echo "=== retest_23_missing done $(date -Is)"

python3 scripts/merge_v4_retest.py \
  --main results/v4_retest_all.json \
  --missing results/v4_retest_23_missing.json \
  --out results/v4_retest_full_23.json > v4_merge_retest.log 2>&1
echo "=== merge done $(date -Is)"

python3 scripts/build_v4_numwarps_dataset.py > v4_build_dataset.log 2>&1
echo "=== dataset done $(date -Is)"

python3 scripts/evaluate_strided_load_v7_on_v4.py > v4_eval_v7.log 2>&1
echo "=== v7 eval done $(date -Is)"

python3 scripts/fit_strided_model_v8.py > v4_fit_v8.log 2>&1
echo "=== fit v8 done $(date -Is)"

echo "ALL DONE $(date -Is)"
