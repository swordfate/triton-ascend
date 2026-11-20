#!/usr/bin/env python3
"""guard_table_md.py - render guard_table.json into the PR markdown table."""
import json
import sys
from collections import defaultdict

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/GUARD_CONTIG/guard_table.json"
rows = json.load(open(src))

order = ["simd_f1", "simt_f1", "simt_f2", "simt_f4"]
by_case = defaultdict(dict)
for r in rows:
    by_case[r["case"]][r["combo"]] = r

print("| case | mode | 实测(us) | 预测(us) | 误差 | 整核误差<10% |")
print("|---|---|---:|---:|---:|---|")
n_pass = n_all = 0
errs = []
for case in sorted(by_case):
    for combo in order:
        r = by_case[case].get(combo)
        if not r:
            continue
        n_all += 1
        meas = r["meas_ns"] / 1000.0 if r["meas_ns"] else None
        pred = r["pred_ns"] / 1000.0 if r["pred_ns"] else None
        err = r["err_pct"]
        ok = "是" if r["pass"] else "否"
        if r["pass"]:
            n_pass += 1
        if err is not None:
            errs.append(abs(err))
        print("| `%s` | %s | %s | %s | %s | %s |" % (
            case, combo,
            "%.3f" % meas if meas else "n/a",
            "%.3f" % pred if pred else "n/a",
            ("%+.2f%%" % err) if err is not None else "n/a",
            ok))
print()
print("GUARD %d/%d within 10%%" % (n_pass, n_all))
if errs:
    errs.sort()
    print("abs-err median %.2f%% mean %.2f%% max %.2f%%" %
          (errs[len(errs) // 2], sum(errs) / len(errs), errs[-1]))
