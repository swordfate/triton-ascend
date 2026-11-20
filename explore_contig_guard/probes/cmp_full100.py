#!/usr/bin/env python3
"""Full 100-case comparison: git467c2d83 baseline vs E33 wheel, errors
vs the v140 measured table (parsed from the gridhint md)."""
import glob
import json
import re

MD = ("/root/work/triton-ascend/explore_contig_guard/"
      "_100case_baseline_vs_commit_gridhint.md")
NO_MEAS = {"histogram_i32_r8192_b256_uniform", "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix", "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}
row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+)\s*\|")

meas = {}
for line in open(MD):
    m = row_re.match(line)
    if m:
        meas[m.group(2)] = float(m.group(4))


def load(root):
    out = {}
    for f in glob.glob(root + "/s*/*.route.json"):
        name = f.split("/")[-1].replace("__auto_f1.route.json", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        route = d.get("effective_decision_kind") or "?"
        costs = d.get("candidate_elapsed_costs") or {}
        if route in costs:
            out[name] = (route, costs[route] / 1000.0)
    return out


base = load("/tmp/RUN_PATH_BEFORE_E33")
e33 = load("/tmp/RUN_PATH")

regress, improve, both_pass, rows = [], [], [], []
for name, m in sorted(meas.items()):
    if name in NO_MEAS:
        continue
    b = base.get(name)
    a = e33.get(name)
    if not b or not a:
        rows.append((name, m, None, None, "MISSING"))
        continue
    be = (b[1] - m) / m * 100
    ae = (a[1] - m) / m * 100
    bp = abs(be) < 10
    ap = abs(ae) < 10
    tag = "PASS" if ap else "FAIL"
    if bp and not ap:
        regress.append((name, be, ae, b[0], a[0], a[1]))
    elif ap and not bp:
        improve.append((name, be, ae, b[0], a[0], a[1]))
    rows.append((name, m, be, ae, tag))

npass = sum(1 for r in rows if r[4] == "PASS")
neval = sum(1 for r in rows if r[4] != "MISSING")
print("eval=%d PASS=%d FAIL=%d" % (neval, npass, neval - npass))
print("回归 (PASS->FAIL): %d" % len(regress))
for r in regress:
    print("  REG %s base_err=%+.1f%% -> %+.1f%%" % (r[0], r[1], r[2]))
print("改善 (FAIL->PASS): %d" % len(improve))
for r in improve:
    print("  IMP %s base_err=%+.1f%% -> %+.1f%%" % (r[0], r[1], r[2]))

errs = sorted(abs(r[3]) for r in rows if r[3] is not None)
if errs:
    n = len(errs)
    print("median |err|=%.2f%% p90=%.2f%%" %
          (errs[n // 2], errs[int(n * 0.9)]))
