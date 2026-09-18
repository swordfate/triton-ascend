#!/usr/bin/env bash
# Generate candidate profiles and run costmodel reports for the 3 padded kernels.
set -u
cd /home/c00946898/stage_vs_camodel_20260917
python3 make_candidate_profiles.py
PROFDIR=profiles
STATUS=candidate_status.log
: > "$STATUS"
for tag in base simt_fill470_l48 simt_fill480_l48 simt_fill440_l0 simd_fill410_l10; do
  for kernel in padded_copy_gather padded_copy_scatter padded_copy_wgrad; do
    timeout 600 ./run_candidate_costmodel.sh "$kernel" "$PROFDIR/$tag.json" "$tag" >/dev/null 2>&1
    rc=$?
    echo "$(date +%T) $tag $kernel rc=$rc" | tee -a "$STATUS"
  done
done
echo ALL_DONE | tee -a "$STATUS"
