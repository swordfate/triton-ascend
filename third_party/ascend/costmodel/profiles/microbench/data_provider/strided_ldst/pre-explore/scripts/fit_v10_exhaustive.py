#!/usr/bin/env python3
"""Two-stage search: rank every pool feature on its own, keep the best N, then
enumerate all 1..4 term subsets of that shortlist (intercept always included).
Non-negative, relative-weighted (1/T) least squares throughout."""
from __future__ import annotations
import sys, json, itertools, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_v9 import load, wnnls, mape
import fit_v9_pool as P

P.POOL.update({
    "nOverEpl": lambda r: r["n"] / max(1.0, math.floor(128.0 / max(1e-9, r["ISB"]))),
    "LoverEb": lambda r: r["L"] / r["eb"],
    "nmodW": lambda r: r["n"] % max(1.0, 32.0 * r["W"]),
    "nl": lambda r: r["n"] * r["L"],
    "nperthread": lambda r: r["n"] / (32.0 * r["W"]),
    "threads": lambda r: 32.0 * r["W"],
    "spanperthread": lambda r: r["span"] / (32.0 * r["W"]),
    "Lperthread": lambda r: r["L"] / (32.0 * r["W"]),
    "nlW": lambda r: r["n"] * r["L"] / (32.0 * r["W"]),
    # ---- the elements-per-line family (the key dtype-absorbing interaction) --
    "epl": lambda r: 128.0 / max(1e-9, r["ISB"]),
    "epl1": lambda r: max(1.0, 128.0 / max(1e-9, r["ISB"])),
    "nEpl": lambda r: r["n"] * 128.0 / max(1e-9, r["ISB"]),
    "nEpl1": lambda r: r["n"] * max(1.0, 128.0 / max(1e-9, r["ISB"])),
    "n2L": lambda r: r["n"] * r["n"] / max(1.0, r["L"]),
    "nEpl_eb": lambda r: r["n"] * 128.0 / max(1e-9, r["ISB"]) * r["eb"],
    "Lepl": lambda r: r["L"] * 128.0 / max(1e-9, r["ISB"]),
    "neplW": lambda r: r["n"] * 128.0 / max(1e-9, r["ISB"]) / (32.0 * r["W"]),
    # ---- piecewise-linear hinges on elements-per-line (concave f(epl)) ----
    "he4": lambda r: max(0.0, 128.0 / max(1e-9, r["ISB"]) - 4.0),
    "he8": lambda r: max(0.0, 128.0 / max(1e-9, r["ISB"]) - 8.0),
    "he16": lambda r: max(0.0, 128.0 / max(1e-9, r["ISB"]) - 16.0),
    "he32": lambda r: max(0.0, 128.0 / max(1e-9, r["ISB"]) - 32.0),
    "nhe4": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 4.0),
    "nhe8": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 8.0),
    "nhe16": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 16.0),
    "nhe32": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 32.0),
    "nhe8W": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 8.0) / (32.0 * r["W"]),
    "nhe16W": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 16.0) / (32.0 * r["W"]),
    "nhe32W": lambda r: r["n"] * max(0.0, 128.0 / max(1e-9, r["ISB"]) - 32.0) / (32.0 * r["W"]),
    "hepl_n": lambda r: r["n"] * min(128.0 / max(1e-9, r["ISB"]), 8.0),
    "hepl_n32": lambda r: r["n"] * min(128.0 / max(1e-9, r["ISB"]), 32.0),
    "ep_cap": lambda r: min(128.0 / max(1e-9, r["ISB"]), 8.0),
    # ---- hinges on L/eb, the SIMT "flat then ramp" driver ----
    "minLeb4": lambda r: min(r["L"] / r["eb"], 4.0),
    "minLeb8": lambda r: min(r["L"] / r["eb"], 8.0),
    "minLeb16": lambda r: min(r["L"] / r["eb"], 16.0),
    "minLeb32": lambda r: min(r["L"] / r["eb"], 32.0),
    "hLeb4": lambda r: max(0.0, r["L"] / r["eb"] - 4.0),
    "hLeb8": lambda r: max(0.0, r["L"] / r["eb"] - 8.0),
    "hLeb16": lambda r: max(0.0, r["L"] / r["eb"] - 16.0),
    "hLeb32": lambda r: max(0.0, r["L"] / r["eb"] - 32.0),
    "hLeb2": lambda r: max(0.0, r["L"] / r["eb"] - 2.0),
    "hLeb1": lambda r: max(0.0, r["L"] / r["eb"] - 1.0),
    "minL16": lambda r: min(r["L"], 16.0),
    "minL64": lambda r: min(r["L"], 64.0),
    "hL2": lambda r: max(0.0, r["L"] - 2.0),
    "hL4": lambda r: max(0.0, r["L"] - 4.0),
    "hL8": lambda r: max(0.0, r["L"] - 8.0),
    "hL16": lambda r: max(0.0, r["L"] - 16.0),
    "hL64": lambda r: max(0.0, r["L"] - 64.0),
    "hL256": lambda r: max(0.0, r["L"] - 256.0),
    "spanEb": lambda r: r["span"] / r["eb"],
    "minSpanEb": lambda r: min(r["span"] / r["eb"], 1024.0),
    "hSpanEb1k": lambda r: max(0.0, r["span"] / r["eb"] - 1024.0),
    "hSpanEb4k": lambda r: max(0.0, r["span"] / r["eb"] - 4096.0),
    "hSpanEb16k": lambda r: max(0.0, r["span"] / r["eb"] - 16384.0),
})


def sub(rows, slot):
    if slot == "simd_load_wide":
        return [r for r in rows if r["op"] == "load" and r["path"] == "simd" and r["ISB"] == 2.0 * r["eb"]]
    if slot == "simd_load_gather":
        return [r for r in rows if r["op"] == "load" and r["path"] == "simd" and r["ISB"] != 2.0 * r["eb"]]
    pth, op = slot.split("_", 1)
    pth = "simt_only" if pth == "simt" else pth
    return [r for r in rows if r["op"] == op and r["path"] == pth]


def solve(rr, keys, K=4, topn=3, shortlist=34):
    t = np.array([r["T"] for r in rr]); W = 1.0 / np.maximum(t, 1e-9)
    C = {k: np.array([P.POOL[k](r) for r in rr]) for k in keys}
    one = np.ones(len(rr))
    # stage 1: single-feature ranking (with intercept)
    rank = []
    for k in keys:
        A = np.column_stack([one, C[k]])
        rank.append((mape(t, A @ wnnls(A, t, W)), k))
    rank.sort()
    short = [k for _, k in rank[:shortlist]]
    # stage 2: exhaustive over the shortlist
    res = []
    for k in range(1, K + 1):
        for combo in itertools.combinations(short, k):
            A = np.column_stack([one] + [C[c] for c in combo])
            x = wnnls(A, t, W)
            res.append((mape(t, A @ x), combo, x))
    res.sort(key=lambda z: z[0])
    return res[:topn], rank[:8], short


def main():
    ds = str(HERE.parent / "results/final/dataset_new_only_v2.csv")
    rows = load(ds)
    KEYS = list(P.POOL.keys())
    print("dataset rows=%d pool=%d" % (len(rows), len(KEYS)))
    out = {}
    for slot in ("simd_load_wide", "simd_load_gather", "simd_store", "simt_load", "simt_store"):
        rr = sub(rows, slot)
        best, rank, short = solve(rr, KEYS, K=4, topn=3)
        print("\n=== %s n=%d ===" % (slot, len(rr)))
        print("  单特征前 5: " + ", ".join("%s=%.1f%%" % (k, m) for m, k in rank[:5]))
        for m, combo, x in best:
            print("  %6.2f%%  %s" % (m, " + ".join("%.4g*%s" % (c, k) for k, c in zip(combo, x[1:]))))
        m, combo, x = best[0]
        A = np.column_stack([np.ones(len(rr))] + [np.array([P.POOL[c](r) for r in rr]) for c in combo])
        pr = A @ x
        t = np.array([r["T"] for r in rr])
        perd = {}
        for d in ("int8", "float16", "float32", "int64"):
            i = [j for j, r in enumerate(rr) if r["dtype"] == d]
            if i: perd[d] = round(mape(t[i], pr[i]), 2)
        out[slot] = {"n": len(rr), "mape": round(m, 3),
                     "terms": [{"feature": "intercept", "coef": round(float(x[0]), 6)}] +
                              [{"feature": k, "coef": round(float(c), 6)} for k, c in zip(combo, x[1:])],
                     "per_dtype": perd}
        print("  per-dtype:", perd)
    T = sum(v["mape"] * v["n"] for v in out.values()); N = sum(v["n"] for v in out.values())
    out["_overall"] = round(T / N, 3)
    json.dump(out, open(HERE.parent / "results/final/fit_v10_exhaustive.json", "w"), indent=1)
    print("\nOVERALL weighted MAPE = %.2f%%" % (T / N))


if __name__ == "__main__":
    main()
