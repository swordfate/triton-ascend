#!/usr/bin/env python3
"""Compare gitc380f6da (current) vs gridhint-doc baseline git2e9efcac.
Measured = v140 table (gridhint md column).  Errors = (pred-meas)/meas."""
import glob
import json
import re

MD = "/tmp/gridhint.md"
NO_MEAS = {"histogram_i32_r8192_b256_uniform", "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix", "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}
row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([+-][0-9.]+)%\s*\|\s*(\S+)\s*\|"
    r"\s*(\S+)\s*\|\s*([0-9.]+)\s*\|\s*([+-][0-9.]+)%\s*\|\s*(\S+)\s*\|")

base = {}
for line in open(MD):
    m = row_re.match(line)
    if m:
        base[m.group(2)] = (float(m.group(4)), float(m.group(5)),
                            float(m.group(6)), m.group(3))

cur = {}
for f in glob.glob("/tmp/RUN_PATH/s*/*.route.json"):
    name = f.split("/")[-1].replace("__auto_f1.route.json", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    route = d.get("effective_decision_kind") or "?"
    costs = d.get("candidate_elapsed_costs") or {}
    if route in costs:
        cur[name] = (route, costs[route] / 1000.0)

rows, regress, improve = [], [], []
npass_b, npass_c, neval = 0, 0, 0
for name, (m, bpred, berr, broute) in sorted(base.items()):
    if name in NO_MEAS:
        continue
    if name not in cur:
        rows.append((name, m, berr, None, broute, "-", "MISSING"))
        continue
    route, p = cur[name]
    ce = (p - m) / m * 100.0
    bp = abs(berr) < 10
    cp = abs(ce) < 10
    neval += 1
    npass_b += bp
    npass_c += cp
    rows.append((name, m, berr, ce, route, p, tag := ("PASS" if cp else
                                                      "FAIL")))
    if bp and not cp:
        regress.append((name, bpred, berr, p, ce))
    elif cp and not bp:
        improve.append((name, bpred, berr, p, ce))

print("eval=%d  2e9efcac基线 PASS=%d  gitc380f6da PASS=%d" %
      (neval, npass_b, npass_c))
eb = sorted(abs(r[2]) for r in rows if r[2] is not None)
ec = sorted(abs(r[3]) for r in rows if r[3] is not None)
if eb:
    n = len(eb)
    print("2e9efcac  median=%.2f%% p90=%.2f%%" %
          (eb[n // 2], eb[min(n - 1, int(n * 0.9))]))
if ec:
    n = len(ec)
    print("c380f6da  median=%.2f%% p90=%.2f%%" %
          (ec[n // 2], ec[min(n - 1, int(n * 0.9))]))
print("\n回归 (基线PASS -> 现在FAIL): %d" % len(regress))
for name, bp, be, p, ce in regress:
    print("  REG %-46s 基线pred=%7.3f err=%+6.1f%% -> pred=%7.3f err=%+6.1f%%"
          % (name, bp, be, p, ce))
print("\n改善 (基线FAIL -> 现在PASS): %d" % len(improve))
for name, bp, be, p, ce in improve:
    print("  IMP %-46s 基线pred=%7.3f err=%+6.1f%% -> pred=%7.3f err=%+6.1f%%"
          % (name, bp, be, p, ce))

print("\n== 当前 >10%% 的 case（含基线已FAIL的）==")
for r in rows:
    if r[4] == "FAIL":
        print("  FAIL %-46s meas=%7.3f pred=%7.3f err=%+6.1f%% "
              "(基线 %+6.1f%%)" % (r[0], r[1], r[3], r[3] and r[2], r[2]))
