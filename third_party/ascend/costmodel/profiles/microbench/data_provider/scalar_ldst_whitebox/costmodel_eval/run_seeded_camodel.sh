#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p out
run_one() {
  local k="$1" m="$2" seed="$3"
  echo "=== $(date +%T) $k $m seed=$seed ==="
  SEED="$seed" timeout 3600 bash run_camodel_seeded_one.sh "$k" "$m" > "out/${k}_${m}.run.log" 2>&1
  echo "$(date +%T) $k $m rc=$?"
}
run_one padded_copy_gather    simt_only 12
run_one padded_copy_scatter   simt_only 12
run_one padded_copy_wgrad     simt_only 12
run_one binned_copy_gather    simt_only 0
run_one binned_copy_scatter   simt_only 0
run_one binned_copy_wgrad     simt_only 0
echo CAMODEL_SEEDED_DONE
