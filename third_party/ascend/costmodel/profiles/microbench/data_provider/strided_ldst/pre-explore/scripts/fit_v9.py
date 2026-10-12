#!/usr/bin/env python3
"""Byte-domain exploration harness for the strided ld/st cost model.

Everything here is a simple LINEAR combination with NON-NEGATIVE weights
(Lawson-Hanson NNLS).  No log, no exp, no powers.  Element width enters
STRUCTURALLY through byte-domain features, so all four dtypes and both ranks
share one fit per slot -- there is never a per-dtype formula.
"""
from __future__ import annotations
import csv, math, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def nnls(A, b, tol=1e-11, maxiter=600):
    """Lawson-Hanson non-negative least squares."""
    A = np.asarray(A, float); b = np.asarray(b, float)
    m, n = A.shape
    P = np.zeros(n, bool); x = np.zeros(n)
    w = A.T @ (b - A @ x)
    it = 0
    while (not P.all()) and (w[~P] > tol).any() and it < maxiter:
        it += 1
        cand = np.where(~P)[0]
        j = cand[np.argmax(w[cand])]
        P[j] = True
        while True:
            Ap = A[:, P]
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(Ap, b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z; break
            mask = P & (z <= 0)
            alpha = (x[mask] / (x[mask] - z[mask])).min()
            x = x + alpha * (z - x)
            P[np.isclose(x, 0.0, atol=1e-12)] = False
        w = A.T @ (b - A @ x)
    return x


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        eb = float(r["element_bytes"]); n = float(r["elements"])
        ISB = float(r["inner_stride_bytes"]); W = max(1.0, float(r["num_warps"]))
        span = float(r["f_span_bytes"])          # already includes the row structure
        Lc = float(r["f_L"])                     # 128B lines touched
        threads = 32.0 * W
        d = dict(r)
        d["T"] = float(r["target_ns"])
        d.update(eb=eb, n=n, ISB=ISB, W=W, span=span, threads=threads,
                 L=Lc, Lf=span / 128.0,
                 pages=span / 4096.0, pagec=math.floor(span / 4096.0),
                 iters=math.ceil(n / threads), per_lane=n / threads,
                 rowgap=float(r.get("f_row_gap_bytes") or 0.0))
        # line count implied by ONE element per line when stride is wide
        d["Lelem"] = n if ISB >= 128.0 else math.ceil(span / 128.0)
        d["lines_per_elem"] = d["L"] / n
        d["cmd"] = math.ceil(d["L"] / max(1.0, math.floor(128.0 / max(1.0, eb))))
        out.append(d)
    return out


def design(rows, feats):
    return np.column_stack([[f(r) for r in rows] for _, f in feats])


def wnnls(A, t, w):
    """NNLS on the RELATIVE problem: minimise sum ((Ax-t)/t)^2.
    Still a plain linear model -- only the fitting weights change, so every
    point counts equally in relative terms (which is what MAPE measures)."""
    w = np.asarray(w, float)
    return nnls(A * w[:, None], np.asarray(t, float) * w)


def mape(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    return float(np.mean(np.abs(p - t) / np.maximum(t, 1e-9)) * 100.0)


def fit(rows, feats, name, report=True):
    A = design(rows, feats)
    t = np.array([r["T"] for r in rows])
    x = nnls(A, t)
    pred = A @ x
    m = mape(t, pred)
    if report:
        print("  %-22s MAPE=%6.2f%%  " % (name, m) +
              " ".join("%s=%.4g" % (f[0], c) for f, c in zip(feats, x) if abs(c) > 1e-9))
    return m, x, pred


def per(rows, pred, key):
    g = {}
    for r, p in zip(rows, pred):
        g.setdefault(key(r), []).append((r["T"], p))
    return {k: mape([a for a, _ in v], [b for _, b in v]) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


CONST = ("1", lambda r: 1.0)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "results/final/dataset_new_only.csv")
    rows = load(ds)
    print("dataset:", ds, " rows:", len(rows))
    SLOTS = {
        "simd_load":   [r for r in rows if r["op"] == "load" and r["path"] == "simd"],
        "simd_store":  [r for r in rows if r["op"] == "store" and r["path"] == "simd"],
        "simt_load":   [r for r in rows if r["op"] == "load" and r["path"] == "simt_only"],
        "simt_store":  [r for r in rows if r["op"] == "store" and r["path"] == "simt_only"],
    }
    for s, rr in SLOTS.items():
        print("\n=== %s  n=%d ===" % (s, len(rr)))
        print("  per-dtype:", {k: round(v, 1) for k, v in
              per(rr, np.array([r["T"] * 0 + np.mean([x["T"] for x in rr]) for r in rr]), lambda r: r["dtype"]).items()})


if __name__ == "__main__":
    main()
