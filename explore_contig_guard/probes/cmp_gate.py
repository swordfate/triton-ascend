#!/usr/bin/env python3
"""Compare the FRESH clean 100-case run (/tmp/RUN_1119CLEAN, wheel
gitc380f6da, cache cleared) against the v140 measured table and the
4902ae4a baseline column from _100case_4902ae4a_vs_1119.md.
Outputs: PASS count, median/p90, new-FAIL list (4902 ok -> fresh fail),
improvements, and a fresh 1119-column table fragment."""
import glob
import json
import re
import statistics

DOC = ("/root/work/triton-ascend/explore_contig_guard/"
       "_100case_4902ae4a_vs_1119.md")
RUN = "/tmp/RUN_GATE"
NO_MEAS = {"histogram_i32_r8192_b256_uniform",
           "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix",
           "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}

# doc rows: | # | `case` | route | meas | p4 | e4 | mk4 | r11 | p11 | e11 | mk11 |
row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+|None)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"
    r"\s*([^|]+)\|\s*([^|]+)\|")

base = {}
for line in open(DOC):
    m = row_re.match(line)
    if not m:
        continue
    name = m.group(2).strip()
    try:
        meas = float(m.group(4))
    except ValueError:
        meas = None
    p4 = float(m.group(5))
    try:
        e4 = float(m.group(6).strip().replace("%", "").strip())
    except ValueError:
        e4 = None
    base[name] = {"idx": int(m.group(1)), "route2e": m.group(3),
                  "meas": meas, "p4": p4, "e4": e4}


def load_run(root):
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
            tag = route.replace("all_", "").replace("_only", "")
            factor = d.get("selected_superblock_factor") or \
                d.get("effective_runtime_factor") or 1
            out[name] = ("%s+F%s" % (tag, factor), costs[route] / 1000.0)
    return out


fresh = load_run(RUN)
print("fresh routes:", len(fresh))

new_fail, improve, rows = [], [], []
npass = neval = 0
for name, b in sorted(base.items(), key=lambda kv: kv[1]["idx"]):
    if name in NO_MEAS or b["meas"] is None or b["e4"] is None:
        continue
    e = fresh.get(name)
    if e is None:
        rows.append((b["idx"], name, b["meas"], b["e4"], None, None,
                     "MISSING"))
        continue
    route, pred = e
    ce = (pred - b["meas"]) / b["meas"] * 100.0
    ok4 = abs(b["e4"]) < 10
    okF = abs(ce) < 10
    neval += 1
    npass += okF
    rows.append((b["idx"], name, b["meas"], b["e4"], ce, route,
                 "PASS" if okF else "FAIL"))
    if ok4 and not okF:
        new_fail.append((b["idx"], name, b["e4"], ce, route, pred))
    elif okF and not ok4:
        improve.append((b["idx"], name, b["e4"], ce))

errs = sorted(abs(r[4]) for r in rows if r[4] is not None)
n = len(errs)
print("eval=%d  fresh PASS=%d  FAIL=%d" % (neval, npass, neval - npass))
print("fresh median=%.2f%% p90=%.2f%%" %
      (errs[n // 2], errs[min(n - 1, int(n * 0.9))]))
print("\n== 新增FAIL (4902 PASS -> fresh FAIL): %d ==" % len(new_fail))
for idx, name, e4, ce, route, pred in new_fail:
    print("  REG %-46s 4902=%+6.1f%% -> fresh=%+6.1f%%  (%s %.3fus)" %
          (name, e4, ce, route, pred))
print("\n== 改善 (4902 FAIL -> fresh PASS): %d ==" % len(improve))
for idx, name, e4, ce in improve:
    print("  IMP %-46s 4902=%+6.1f%% -> fresh=%+6.1f%%" % (name, e4, ce))
print("\n== fresh 全部>10%% ==")
for idx, name, meas, e4, ce, route, tag in rows:
    if tag == "FAIL":
        print("  FAIL %-46s meas=%8.3f fresh=%+6.1f%% (4902 %+6.1f%%)" %
              (name, meas, ce, e4))
