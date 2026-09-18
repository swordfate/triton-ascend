#!/usr/bin/env bash
# Run CAModel for the remaining padded kernels (scatter/wgrad) in both modes.
set -u
cd /home/c00946898/stage_vs_camodel_20260917
STATUS="suite_camodel_status.log"
: > "$STATUS"
for k in padded_copy_scatter padded_copy_wgrad; do
  for m in simt_only simd; do
    timeout 1800 ./run_suite_camodel.sh "$k" "$m" > "camodel_${k}_${m}_run.log" 2>&1
    rc=$?
    echo "$(date +%T) $k $m rc=$rc" | tee -a "$STATUS"
  done
done
echo "ALL_DONE" | tee -a "$STATUS"
