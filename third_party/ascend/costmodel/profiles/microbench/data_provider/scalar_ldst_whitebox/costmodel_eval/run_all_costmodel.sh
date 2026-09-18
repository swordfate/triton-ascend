#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
STATUS="out/costmodel_status.log"
: > "$STATUS"
for k in padded_copy_gather padded_copy_scatter padded_copy_wgrad \
         binned_copy_gather binned_copy_scatter binned_copy_wgrad; do
  echo "=== $(date +%T) $k costmodel ===" | tee -a "$STATUS"
  timeout 900 bash run_costmodel_one.sh "$k" > "out/${k}_costmodel.run.log" 2>&1
  rc=$?
  echo "$(date +%T) $k costmodel rc=$rc" | tee -a "$STATUS"
done
echo ALL_COSTMODEL_DONE | tee -a "$STATUS"
