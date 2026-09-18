#!/usr/bin/env python3
"""Create profile.json from the installed david_v100_simd_simt_v1.json snapshot.

This is the single white-box profile used by README.md section 11 validation.

Load coefficients (README sections 1-6 CAModel measurements):
  SIMD MainScalar  : o1 447, o4-same 493, o4-diff 956
  SIMT warp-uniform: o1 530, o4-same 1923, o4-diff 526

Store coefficients (white-box CAModel active windows):
  SIMD Triton store lowers to MTE3: prep 20 + fill 450 + (K-1)*480.
  SIMT uniform store: same-line K>=2 555+(K-1)*480; first-store/diff 450+(K-1)*20.
"""
import json
import os

src = os.path.expanduser(
    "~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/ascend/"
    "costmodel_profiles/simd_simt/david_v100_simd_simt_v1.json"
)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.json")
d = json.load(open(src))

simd = d["simd"]["stage_resources"]["scalar_memory"]
simt = d["simt"]["stage_resources"]["scalar_memory"]

# SIMD load: prep 7 + fill 440 = 447; hit=(493-447-3*3)/3=37/3; issue=3.
simd["main_load_prep_system_cycles"] = 7.0
simd["main_load_fill_system_cycles"] = 440.0
simd["main_load_hit_system_cycles"] = 37.0 / 3.0
simd["main_load_issue_system_cycles"] = 3.0
# keep outstanding=2, low=250, high=350, threshold=4

# SIMT load: prep 6 + fill 524 = 530; same-line serial=(1923-530)/3; diff issue 0.001.
simt["uniform_load_prep_system_cycles"] = 6.0
simt["uniform_load_fill_system_cycles"] = 524.0
simt["uniform_load_same_line_serial_system_cycles"] = (1923.0 - 530.0) / 3.0
simt["uniform_load_diff_line_issue_system_cycles"] = 0.001  # >0 keeps structured branch

# SIMD Triton scalar store = MTE3 UB -> OUT.  Remove the obsolete MainScalar
# store fields (this target does not use ST_XD_XN_IMM -> GM) and use the
# white-box MTE3 window: prep + fill + (K - 1) * serial.
for key in [k for k in simd if k.startswith("main_store_")]:
    del simd[key]
simd["mte3_store_prep_system_cycles"] = 20.0
simd["mte3_store_fill_system_cycles"] = 450.0
simd["mte3_store_serial_system_cycles"] = 480.0

# SIMT uniform store white-box CAModel window (not board marginal throughput):
# same-line K>=2: 555+(K-1)*480; first-store/diff: 450+(K-1)*20.
simt["uniform_store_same_line_base_system_cycles"] = 555.0
simt["uniform_store_same_line_serial_system_cycles"] = 480.0
simt["uniform_store_diff_line_base_system_cycles"] = 450.0
simt["uniform_store_diff_line_issue_system_cycles"] = 20.0

d["microbenchmark_profile"] = os.path.expanduser(
    "~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/ascend/"
    "costmodel_profiles/microbench/ascend_davidv100_v1.json"
)
json.dump(d, open(out, "w"), indent=2)
print("WROTE", out)
