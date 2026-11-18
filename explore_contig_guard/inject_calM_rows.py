#!/usr/bin/env python3
"""inject_calM_rows.py - inject calM/calM2 measured rows into both profile
copies (repo + site-packages).

Convention (calJ-compatible): row value = whole-kernel profiler median in ns,
divided by waves for grids > 56.

Derived rows (fully measured-based, no free parameters):
  fused   n1_s1_b2048 g8 k{1,4,16,64}: T = full_kernel_g8(k) - k*186.5
          (subtract the calibrated rowwise-reduction term per iteration)
  masked80 n1_s1_b48000 g8 k{4,8,16,18}: T = full_kernel_g8(k) - k*(1004.7+219.0)
          (subtract vecExec and the per-iteration completion floors)
Straight measured rows:
  rowred  simt n1_s0_b8192 g8 k{4,16,64} + g56 k{4,8,16,64} (multibuffer=True
          loads-only bodies; replaces the pre-multibuffer under-measurements)
  permute simd n2_s0_b2048 g{8,32,56} k{4,8,12,16} (loads-only bodies)
  masked80 simd n1_s1_b65536 g8 k{4,8,16,64} + g56 k{4,8,16} (bucket bodies)
"""
import json
import shutil
from pathlib import Path

SP = Path('/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/'
          'ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json')
REPO = Path('/root/work/triton-ascend/third_party/ascend/costmodel/profiles/'
            'simd_simt/ascend_950pr_simd_simt_v1.json')

calM = json.load(open('/root/work/explore_contig/calM_rows.json'))
calM2 = json.load(open('/root/work/explore_contig/calM2_rows.json'))

FUSED_RED = 186.5          # stage_4 vecExec per iteration ([stg] trace)
MSM_VECEXEC = 1004.7       # stage_5 vecExec per iteration
MSM_FLOORS = 219.0         # per-iteration completion floors

new_simd = {}
new_simt = {}

# --- fused: derived in-context memory rows (g8 = final-cost bracket) ---
for k in (1, 4, 16, 64):
    key = 'n1_s1_b2048full_k%d_g8_m0' % k
    if key in calM2:
        new_simd['n1_s1_b2048_k%d_g8_m0' % k] = round(
            calM2[key] - k * FUSED_RED, 1)
if 'n1_s1_b2048full_k4_g32_m0' in calM2:
    pass  # full g32 k4 measured as attr in calM (mbTrue) - route-eval row:
if 'attr_fsoft_full_k4_g32_mbTrue' in calM:
    new_simd['n1_s1_b2048_k4_g32_m0'] = round(
        calM['attr_fsoft_full_k4_g32_mbTrue'] - 4 * FUSED_RED, 1)

# --- masked80: derived in-context rows at the exact 48KB chunk ---
for k in (4, 8, 16, 18):
    key = 'n1_s1_b48000full_k%d_g8_m0' % k
    if key in calM2:
        new_simd['n1_s1_b48000_k%d_g8_m0' % k] = round(
            calM2[key] - k * (MSM_VECEXEC + MSM_FLOORS), 1)

# --- masked80: straight bucket-body rows ---
for k in (4, 8, 16):
    for g in (8, 56):
        key = 'n1_s1_b65536_k%d_g%d_m0' % (k, g)
        if key in calM:
            new_simd['n1_s1_b65536_k%d_g%d_m0' % (k, g)] = calM[key]
if 'n1_s1_b65536_k64_g8_m0' in calM2:
    new_simd['n1_s1_b65536_k64_g8_m0'] = calM2['n1_s1_b65536_k64_g8_m0']
# masked80 real-body g56 rows (route-eval bracket; memory-only measurement)
for k in (4, 8, 16, 18):
    key = 'n1_s1_b48000_k%d_g56_m0' % k
    if key in calM:
        new_simd[key] = calM[key]

# --- rowred: simt loads-only multibuffer rows ---
for k in (4, 16, 64):
    key = 'n1_s0_b8192_k%d_g56_m0' % k
    if key in calM:
        new_simt[key] = calM[key]
    key = 'n1_s0_b8192_k%d_g8_m0' % k
    if key in calM2:
        new_simt[key] = calM2[key]
if 'n1_s0_b8192_k8_g56_m0' in calM2:
    new_simt['n1_s0_b8192_k8_g56_m0'] = calM2['n1_s0_b8192_k8_g56_m0']

# --- permute: x/y loads-only rows ---
for k in (4, 8, 12, 16):
    for g in (8, 32, 56):
        key = 'n2_s0_b2048_k%d_g%d_m0' % (k, g)
        if key in calM:
            new_simd[key] = calM[key]

for p in (SP, REPO):
    d = json.load(open(p))
    for mode, add in (('simd', new_simd), ('simt', new_simt)):
        v2 = d[mode]['memory']['contiguous_ldst_v2']
        direct = v2['direct']
        for key, val in add.items():
            direct[key] = val
    json.dump(d, open(p, 'w'), indent=1)
    print('updated', p, 'simd+%d simt+%d' % (len(new_simd), len(new_simt)))
print('ROWS:')
for k, v in sorted(new_simd.items()):
    print('  simd %-30s %s' % (k, v))
for k, v in sorted(new_simt.items()):
    print('  simt %-30s %s' % (k, v))
