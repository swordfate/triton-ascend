#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
STATUS="out/camodel_status.log"
: > "$STATUS"
for k in padded_copy_gather padded_copy_scatter padded_copy_wgrad \
         binned_copy_gather binned_copy_scatter binned_copy_wgrad; do
  for m in simd simt_only; do
    echo "=== $(date +%T) $k $m ===" | tee -a "$STATUS"
    timeout 3600 bash run_camodel_one.sh "$k" "$m" > "out/${k}_${m}.run.log" 2>&1
    rc=$?
    echo "$(date +%T) $k $m rc=$rc" | tee -a "$STATUS"
  done
done
echo ALL_CAMODEL_DONE | tee -a "$STATUS"
