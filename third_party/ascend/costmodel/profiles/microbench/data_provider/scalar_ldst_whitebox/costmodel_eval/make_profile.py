#!/usr/bin/env python3
"""Write the scalar load/store white-box fields into a local profile.json.

Source formulas (raw CAModel core cycles, 1.8 GHz):
  SIMD MainScalar load, diff-line: T = 7 + 440 + max(0,K-2)*extra(K) + (K-1)*3
  SIMT warp-uniform load, diff-line: T = 6 + 480 + (K-1)*1
  SIMD Triton scalar store, K=1: T = 20 + 450
  SIMT scalar store, K=1: T = 450

The profile stores SYS_CNT cycles, so all cycle constants are multiplied by
988.9/1800 = 0.5493889.  Same-line and dependency branches are not exercised
by the six target kernels and are intentionally not modeled here.
"""
import json
import os

SIM_TO_SYS = 988.9 / 1800.0
PROFILE = os.path.expanduser("~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/"
                             "ascend/costmodel_profiles/simd_simt/david_v100_simd_simt_v1.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.json")

profile = json.load(open(PROFILE))
simd = profile["simd"]["stage_resources"]
simt = profile["simt"]["stage_resources"]

simd["scalar_memory"] = {
    "description": "SIMD scalar white-box fields in SYS_CNT cycles.",
    "main_load_prep_system_cycles": 7.0 * SIM_TO_SYS,
    "main_load_fill_system_cycles": 440.0 * SIM_TO_SYS,
    "main_load_issue_system_cycles": 3.0 * SIM_TO_SYS,
    "main_load_outstanding_line_count": 2,
    "main_load_extra_line_low_system_cycles": 250.0 * SIM_TO_SYS,
    "main_load_extra_line_high_system_cycles": 350.0 * SIM_TO_SYS,
    "main_load_extra_line_high_threshold": 4,
    "mte3_store_prep_system_cycles": 20.0 * SIM_TO_SYS,
    "mte3_store_fill_system_cycles": 450.0 * SIM_TO_SYS,
}
simt["scalar_memory"] = {
    "description": "SIMT warp-uniform scalar white-box fields in SYS_CNT cycles.",
    "uniform_load_prep_system_cycles": 6.0 * SIM_TO_SYS,
    "uniform_load_fill_system_cycles": 480.0 * SIM_TO_SYS,
    "uniform_load_diff_line_issue_system_cycles": 1.0 * SIM_TO_SYS,
    "uniform_store_base_system_cycles": 450.0 * SIM_TO_SYS,
}
profile["microbenchmark_profile"] = os.path.expanduser(
    "~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/"
    "ascend/costmodel_profiles/microbench/ascend_davidv100_v1.json")
json.dump(profile, open(OUT, "w"), indent=2)
print("WROTE", OUT)
