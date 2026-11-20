#!/usr/bin/env python3
"""E28 analysis: identify the mixed-body composition law.

Pure anchors -> L(lB), S(sB) curves (per-(n,bytes) direct table: the anchor
grid is dense enough to price every mixed point's two sides exactly by its
(nl,lB)/(ns,sB) key when present, else nearest same-n point with matched
per-stream bytes).

Candidate laws evaluated on the 26 mixed points (MD+MI, /ITERS):
  H_max : max(L,S)
  H_add : L+S
  H_bw  : shared-bandwidth  lat + (lB+sB)/BW
  H_sum : max(L,S) + alpha * harmonic(lB,sB)      (contention term)
  H_sk  : max(L,S) + alpha * min-lines (sB/128)   (store-slot term)
with per-law LSQ where applicable + MAPE.
"""
import json
import math
from pathlib import Path

ITERS = 64
data = [json.loads(l) for l in
        Path("/tmp/e28.jsonl").read_text().splitlines()]
for d in data:
    d["per"] = d["meas"] / ITERS

L = {(d["nl"], d["lB"]): d["per"] for d in data if d["kind"] == "L"}
S = {(d["ns"], d["sB"]): d["per"] for d in data if d["kind"] == "S"}
mix = [d for d in data if d["kind"] in ("MD", "MI")]

print("== pure anchors (per-iter ns) ==")
for k in sorted(L):
    print("  L n=%d %7.0fB -> %7.1f" % (k[0], k[1], L[k]))
for k in sorted(S):
    print("  S n=%d %7.0fB -> %7.1f" % (k[0], k[1], S[k]))


def price(tab, n, b):
    """exact key, else same-n neighbour with closest per-stream bytes."""
    if (n, b) in tab:
        return tab[(n, b)]
    same = [(k, v) for k, v in tab.items() if k[0] == n]
    if same:
        return min(same, key=lambda kv: abs(kv[0][1] - b))[1]
    return min(tab.items(), key=lambda kv: abs(kv[0][1] - b))[1]


def lsq(X, y):
    import numpy as np
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


print("\n== mixed points: meas vs laws ==")
print("%-18s %7s %7s %7s %7s %7s %7s %7s" %
      ("point", "meas", "max", "add", "bw", "sum", "sk", "meas-max"))
rows = []
for d in mix:
    Lv = price(L, d["nl"], d["lB"])
    Sv = price(S, d["ns"], d["sB"])
    d["L"] = Lv
    d["S"] = Sv
    d["rmax"] = d["per"] - max(Lv, Sv)
    rows.append(d)
    print("%-18s %7.1f %7.1f %7.1f %7s %7s %7s %7.1f  (nl=%d ns=%d)" %
          (d["tag"], d["per"], max(Lv, Sv), Lv + Sv, 0, 0, 0, d["rmax"],
           d["nl"], d["ns"]))

# ---- LSQ candidate fits ----
import numpy as np
y = np.array([d["per"] for d in rows], dtype=float)
X_max = np.array([[max(d["L"], d["S"])] for d in rows])
X_add = np.array([[d["L"] + d["S"]] for d in rows])
# H_bw: lat + (lB+sB)/BW  -> features [1, total]
X_bw = np.array([[1.0, d["lB"] + d["sB"]] for d in rows])
# H_sum: max + alpha*harmonic
X_sum = np.array([[max(d["L"], d["S"]),
                   (d["lB"] * d["sB"]) / max(1.0, d["lB"] + d["sB"])]
                  for d in rows])
# H_sk: max + alpha*store lines + beta*load lines
X_sk = np.array([[max(d["L"], d["S"]),
                  d["sB"] / 128.0, d["lB"] / 128.0] for d in rows])
# H_kline: max + alpha * min(lB,sB)/128
X_kl = np.array([[max(d["L"], d["S"]),
                  min(d["lB"], d["sB"]) / 128.0] for d in rows])


def mape(pred):
    return 100.0 * sum(abs(p - d["per"]) / d["per"]
                       for p, d in zip(pred, rows)) / len(rows)


print("\n== law fits (26 mixed points) ==")
for name, X in (("H_max", X_max), ("H_add", X_add), ("H_bw", X_bw),
                ("H_sum(harmonic)", X_sum), ("H_sk(lines l+s)", X_sk),
                ("H_kline(min lines)", X_kl)):
    b = lsq(X, y)
    pred = X @ b
    coefs = ", ".join("%s=%.2f" % (nm, v) for nm, v in
                      zip(["c%d" % i for i in range(len(b))], b))
    print("%-18s MAPE=%6.1f%%  [%s]" % (name, mape(pred), coefs))

# residual structure of H_max: rmax vs features
print("\n== H_max residual structure (rmax per mixed point) ==")
for d in sorted(rows, key=lambda d: d["rmax"]):
    print("%-16s rmax=%7.1f  lB=%6d sB=%6d nl=%d ns=%d  "
          "minKB=%5.1f harm=%6.0f" %
          (d["tag"], d["rmax"], d["lB"], d["sB"], d["nl"], d["ns"],
           min(d["lB"], d["sB"]) / 1024.0,
           (d["lB"] * d["sB"]) / max(1, d["lB"] + d["sB"]) / 1024.0))
