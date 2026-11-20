#!/usr/bin/env python3
"""inject_perm_n3.py - permute combined-load rows (x,y contiguous + z strided,
mixed-stream admission): full-body measured values keyed n3_s0_b2048."""
import json

calM3 = json.load(open('/root/work/explore_contig/calM3_rows.json'))
add = {}
for k in (8, 12, 16):
    key = 'attr_perm_full_k%d_g32' % k
    if key in calM3:
        add['n3_s0_b2048_k%d_g32_m0' % k] = calM3[key]

SP = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/"
      "costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json")
REPO = ("/root/work/triton-ascend/third_party/ascend/costmodel/profiles/"
        "simd_simt/ascend_950pr_simd_simt_v1.json")
for p in (SP, REPO):
    d = json.load(open(p))
    d['simd']['memory']['contiguous_ldst_v2']['direct'].update(add)
    json.dump(d, open(p, 'w'), indent=1)
for k, v in sorted(add.items()):
    print(k, v)
print('permute n3_s0 rows injected')
