#!/usr/bin/env python3
"""Extract >10% error cases from _100case_baseline_vs_commit_gridhint.md,
join with the CURRENT run's route.json predictions, emit a working table.

Usage: python3 fail_analysis.py
"""
import glob
import json
import re

MD = ("/root/work/triton-ascend/explore_contig_guard/"
      "_100case_baseline_vs_commit_gridhint.md")
RUN = "/tmp/RUN_PATH"

# 不可评（无实测）
NO_MEAS = {"histogram_i32_r8192_b256_uniform", "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix", "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}

row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([+-][0-9.]+)%\s*\|\s*(\S+)\s*\|"
    r"\s*(\S+)\s*\|\s*([0-9.]+)\s*\|\s*([+-][0-9.]+)%\s*\|\s*(\S+)\s*\|")

fails = {}
meas = {}
for line in open(MD):
    m = row_re.match(line)
    if not m:
        continue
    name = m.group(2)
    m_meas = float(m.group(4))
    b_err = float(m.group(6))
    f_err = float(m.group(10))
    meas[name] = m_meas
    if name in NO_MEAS:
        continue
    if abs(b_err) > 10.0 or abs(f_err) > 10.0:
        fails[name] = (m_meas, b_err, f_err)

print("union >10%% cases: %d" % len(fails))

# current run predictions
cur = {}
for f in glob.glob(RUN + "/s*/*.route.json"):
    name = f.split("/")[-1].replace("__auto_f1.route.json", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    route = d.get("effective_decision_kind") or "?"
    costs = d.get("candidate_elapsed_costs") or {}
    pred_us = costs.get(route)
    if pred_us is not None:
        cur[name] = (route, pred_us / 1000.0)

print("\n%-38s %8s %8s %8s %8s  %s" %
      ("case", "meas_us", "base_err", "final_err", "cur_err", "cur_route"))
for name in sorted(fails):
    m, b, fl = fails[name]
    if name in cur:
        route, p = cur[name]
        ce = (p - m) / m * 100.0
        print("%-38s %8.3f %7.1f%% %7.1f%% %7.1f%%  %s" %
              (name, m, b, fl, ce, route))
    else:
        print("%-38s %8.3f %7.1f%% %7.1f%% %8s  %s" %
              (name, m, b, fl, "MISSING", "-"))
