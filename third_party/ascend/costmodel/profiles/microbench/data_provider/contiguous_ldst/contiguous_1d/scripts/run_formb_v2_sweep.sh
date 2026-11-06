#!/bin/bash
set -u
C="${HOME}/contig_calib"
M="${C}/formb_v2.jsonl"
LOG="${C}/formb_v2.log"
HL="${C}/formb_v2_health.log"
: > "${M}"
source ~/_pre.sh >/dev/null 2>&1
unset NPU_DEVICE_LIMIT
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE:-0}
TOTAL=728
echo "start $(date +%H:%M:%S) device=$ASCEND_RT_VISIBLE_DEVICES total=$TOTAL" > "${LOG}"
: > "${HL}"
n=0
health() {
  local v
  v=$(python3 "${C}/probe_speed.py" --json 2>/dev/null | tail -1)
  echo "$(date +%H:%M:%S) after=$n $v" >> "${HL}"
}
for B in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do
  for dir in load store; do
    for dt in int8 fp16 fp32 int64; do
      for W in 1 2 4 8 16 32 64; do
        n=$((n+1))
        if timeout 900 python3 "${C}/probe_mask_kind.py" --mode simt_only --direction "${dir}" \
             --dtype "${dt}" --block "${B}" --valid "${B}" --dynamic 1 --warps "${W}" \
             --k-low 2048 --k-high 8192 --repeats 5 --manifest "${M}" > /dev/null 2>&1; then
          echo "OK   [$n/$TOTAL] B=$B $dir $dt W=$W $(date +%H:%M:%S)" >> "${LOG}"
        else
          echo "FAIL [$n/$TOTAL] B=$B $dir $dt W=$W $(date +%H:%M:%S)" >> "${LOG}"
        fi
        if [ $((n % 100)) -eq 0 ]; then health; fi
      done
    done
  done
done
echo "ALLDONE $(date +%H:%M:%S)" >> "${LOG}"
touch "${C}/FORMB_DONE"
