#!/usr/bin/env python3
"""Fit te>=8 SIMT stream formulas from the paired-control calibration.

Loads: cost = slope(A) - slope(B); cells where both kernels carry a
spilling accumulator (2*te*eb > 192 bytes/thread of live state) are
excluded from the fit - the formula extrapolates smoothly there.
Stores: direct slope, all cells (no accumulator in the store loop).
"""
import json, math, itertools
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "te8", "raw", "te8_stream_paired_950pr.jsonl")
EB = {"int8":1,"fp16":2,"fp32":4,"int64":8}
rows = []
for line in open(RAW):
    r = json.loads(line)
    if r.get("status") != "ok" or not r.get("cost"): continue
    W, eb, te, by = r["W"], r["eb"], float(r["te"]), r["by"]
    BUDGET = {1:120, 2:120, 4:120, 8:120, 16:60, 32:28}
    if r["dir"] == "load":
        if W == 64: continue                      # 16-reg regime: marginal spills
        if te * eb / 2.0 > BUDGET[W]: continue    # acc+temp over the RF budget
        if eb == 1 and te >= 64: continue         # byte-load slow path (separate regime)
        if eb == 2 and te == 64: continue         # marginal pressure
    rows.append(dict(dir=r["dir"], dtype=r["dtype"], eb=eb, W=W, te=te, by=by,
                     ns=float(r["cost"]),
                     nI=max(1.0, math.ceil((by/(32.0*W))/16.0))))
print("fit rows: %d (load %d, store %d)" % (
    len(rows), sum(1 for r in rows if r["dir"]=="load"),
    sum(1 for r in rows if r["dir"]=="store")))

F = {
 "one":  lambda r: 1.0,
 "nI":   lambda r: r["nI"],
 "te":   lambda r: r["te"],
 "teq":  lambda r: max(0.0, r["te"]-8.0),
 "bpt":  lambda r: r["te"]*r["eb"],
 "eb":   lambda r: float(r["eb"]),
 "log2eb": lambda r: math.log2(r["eb"]),
 "bw":   lambda r: r["by"]/256.0,
 "lines": lambda r: r["by"]/128.0,
 "lnW4": lambda r: (r["by"]/128.0)/max(1.0, r["W"]/4.0),
 "pw":   lambda r: (r["by"]/256.0)/r["W"],
 "pwsq": lambda r: (r["by"]/256.0)/math.sqrt(r["W"]),
 "l2w":  lambda r: math.log2(r["W"]),
 "nIl2w": lambda r: r["nI"]*math.log2(r["W"]),
 "sqte": lambda r: math.sqrt(r["te"]),
 "nImW": lambda r: r["nI"]*max(1.0, r["W"]/4.0),
}
KEYS = [k for k in F if k != "one"]

def best_subset(fam, size=3):
    results = []
    for combo in itertools.combinations(KEYS, size):
        keys = ["one"] + list(combo)
        X = np.array([[F[k](r) for k in keys] for r in fam])
        y = np.array([r["ns"] for r in fam])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        ape = np.abs(X@coef - y)/y
        results.append((100*ape.mean(), keys, coef, 100*np.percentile(ape,95)))
    results.sort()
    return results[0]

out = {}
print("%-14s %-26s %5s %5s" % ("family","best subset","MAPE%","p95%"))
for direction in ("load","store"):
    famall = [r for r in rows if r["dir"]==direction]
    m, keys, coef, p95 = best_subset(famall)
    print("%-14s %-26s %5.1f %5.1f %s" % (direction, "+".join(keys), m, p95,
          "PASS-merged" if m<=15 else ""))
    if m <= 15.0:
        out[direction] = dict(features=keys, coefficients=[round(c,3) for c in coef],
                              n=len(famall), mape=round(m,1))
    else:
        for dt in ("int8","fp16","fp32","int64"):
            fam = [r for r in rows if r["dir"]==direction and r["dtype"]==dt]
            if len(fam) < 8: continue
            m, keys, coef, p95 = best_subset(fam)
            print("  %-12s %-24s %5.1f %5.1f %s" % (dt, "+".join(keys), m, p95,
                  "PASS" if m<=15 else ""))
            out["%s.%s" % (direction, dt)] = dict(
                features=keys, coefficients=[round(c,3) for c in coef],
                n=len(fam), mape=round(m,1),
                eb_key="%s.eb%d" % (direction, EB[dt]))
# emit with .ebN keys as the C++ lookup expects
emit = {}
for k, v in out.items():
    emit[v.pop("eb_key", k)] = v
json.dump({"te8_formulas": emit}, open(os.path.join(HERE, "..", "te8", "fitted", "te8_formulas.json"), "w"), indent=1)
print("\nwrote /root/work/te8_formulas.json:", sorted(emit.keys()))
