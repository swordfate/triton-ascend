#!/usr/bin/env python3
"""E27e fit v6: per-n curves + joint issue floor + line tax.

  L = curve_nL(lB)   (per-nL measured piecewise-linear curve)
  S = curve_nS(sB)   (simd measured per-nS; simt = E17i drain curve)
  floor    = max(stepL[nL], stepS[nS], stepT[nL+nS])
  base     = max(floor, L, S)                    (simd)
           = L + S                               (simt, E17g additive)
  mixed tax (simd, both directions, non-tiny):
             + c_l*(lB/128) + c_s*(sB/128)
  tiny excess/chain: delta tables vs the load-only base (unchanged)

stepT[n] fitted from small-byte MIXED points (lB,sB <= 2KB) with
nL+nS == n; stepL/stepS from the single-direction small-byte points.
"""
import json
import math
import statistics
from pathlib import Path

DTYPES = ("int8", "fp16", "fp32", "int64")
SIMT_LINE_L2 = (6.22, 6.46, 6.08, 6.08)
SIMT_LINE_BYTES = (4096.0, 8192.0, 16384.0, 32768.0)


def curve_eval(pts, b):
    if not pts or b <= 0:
        return 0.0
    if b <= pts[0][0]:
        return pts[0][1]
    if len(pts) < 2:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        if pts[i][0] <= b <= pts[i + 1][0]:
            t = (b - pts[i][0]) / max(1e-9, pts[i + 1][0] - pts[i][0])
            return pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
    a, bb = pts[-2], pts[-1]
    slope = (bb[1] - a[1]) / max(1.0, bb[0] - a[0])
    return bb[1] + slope * (b - bb[0])


def simt_store_curve(sb):
    if sb <= 0:
        return 0.0
    lines = sb / 128.0
    b = min(max(sb, SIMT_LINE_BYTES[0]), SIMT_LINE_BYTES[-1])
    for i in range(len(SIMT_LINE_BYTES) - 1):
        if SIMT_LINE_BYTES[i] <= b <= SIMT_LINE_BYTES[i + 1]:
            t = ((b - SIMT_LINE_BYTES[i]) /
                 (SIMT_LINE_BYTES[i + 1] - SIMT_LINE_BYTES[i]))
            lc = (SIMT_LINE_L2[i] +
                  t * (SIMT_LINE_L2[i + 1] - SIMT_LINE_L2[i]))
            return lines * lc
    return lines * SIMT_LINE_L2[-1]


def nlerp(curves, n, b):
    keys = sorted(curves.keys())
    if not keys or n <= 0:
        return 0.0
    if n in curves:
        return curve_eval(curves[n], b)
    lo = max([k for k in keys if k <= n] or [None])
    hi = min([k for k in keys if k >= n] or [None])
    if lo is None:
        return curve_eval(curves[hi], b) * n / hi
    if hi is None:
        return curve_eval(curves[lo], b)
    w = (math.log2(n) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return (curve_eval(curves[lo], b) +
            w * (curve_eval(curves[hi], b) - curve_eval(curves[lo], b)))


def fit(mode, dtype, data):
    Lf = [d for d in data if d["kind"] == "L"]
    Sf = [d for d in data if d["kind"] == "S"]
    M = [d for d in data if d["kind"] == "M"]
    T0 = [d for d in data if d["kind"] == "T" and not d["red"]]
    T1 = [d for d in data if d["kind"] == "T" and d["red"]]

    load_curves = {}
    for n in sorted(set(d["nl"] for d in Lf)):
        pts = sorted([(d["lB"], d["meas"]) for d in Lf if d["nl"] == n])
        load_curves[n] = [(b, round(v, 1)) for b, v in pts]
    if mode == "simt_only":
        store_curves = {1: [(b, round(simt_store_curve(b), 1))
                            for b in (1024.0, 4096.0, 8192.0, 16384.0,
                                      32768.0)]}
    else:
        store_curves = {}
        for n in sorted(set(d["ns"] for d in Sf)):
            pts = sorted([(d["sB"], d["meas"]) for d in Sf if d["ns"] == n])
            store_curves[n] = [(b, round(v, 1)) for b, v in pts]

    # ---- floors ----
    def small_of(fam, nkey, bkey):
        out = {}
        for d in fam:
            if d[bkey] <= 2048:
                n = d[nkey]
                out.setdefault(n, []).append(d["meas"])
        return {n: statistics.median(v) for n, v in out.items()}

    stepL = small_of(Lf, "nl", "lB")
    stepS = small_of(Sf, "ns", "sB")
    stepT = {}
    if mode == "simd":
        for d in M:
            if d["lB"] <= 2048 and d["sB"] <= 2048:
                n = d["nl"] + d["ns"]
                stepT.setdefault(n, []).append(d["meas"])
        stepT = {n: round(statistics.median(v), 1)
                 for n, v in stepT.items()}

    def loadside(d):
        return nlerp(load_curves, d["nl"], d["lB"]) if d["nl"] else 0.0

    def storeside(d):
        if d["ns"] == 0:
            return 0.0
        if mode == "simt_only":
            return simt_store_curve(d["sB"])
        return nlerp(store_curves, d["ns"], d["sB"])

    def floor(d):
        f = 0.0
        if d["nl"] in stepL:
            f = max(f, stepL[d["nl"]])
        if d["ns"] in stepS:
            f = max(f, stepS[d["ns"]])
        n = d["nl"] + d["ns"]
        if n in stepT:
            f = max(f, stepT[n])
        return f

    # ---- mixed line tax (simd) ----
    c_l = c_s = 0.0
    mixed = [d for d in M if d["seb"] > 128] if mode == "simd" else []
    if len(mixed) >= 4:
        import numpy as np
        X = np.array([[d["lB"] / 128.0, d["sB"] / 128.0] for d in mixed])
        y = np.array([d["meas"] - max(floor(d), loadside(d), storeside(d))
                      for d in mixed])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        c_l, c_s = round(float(beta[0]), 3), round(float(beta[1]), 3)

    def base(d):
        b = max(floor(d), loadside(d), storeside(d))
        if mode == "simt_only":
            return loadside(d) + storeside(d)
        if d["nl"] > 0 and d["ns"] > 0:
            return b + c_l * d["lB"] / 128.0 + c_s * d["sB"] / 128.0
        return b

    def deltas_of(fam):
        out = {}
        for domk in sorted(set(d["dom"] for d in fam)):
            pts = sorted([d for d in fam if d["dom"] == domk],
                         key=lambda d: d["ns"])
            dl = []
            for d in pts:
                dd = dict(d)
                dd["ns"] = 0
                dd["nl"] = d["nl"]
                dl.append((d["ns"], round(d["meas"] - base(dd), 1)))
            out[domk] = dl
        return out

    excess = deltas_of(T0)
    chain = deltas_of(T1)

    def interp_pairs(pairs, n):
        if not pairs:
            return 0.0
        if n <= pairs[0][0]:
            return pairs[0][1] * n / max(1, pairs[0][0])
        for (n0, v0), (n1, v1) in zip(pairs, pairs[1:]):
            if n <= n1:
                return v0 + (v1 - v0) * (n - n0) / (n1 - n0)
        (n0, v0), (n1, v1) = pairs[-2], pairs[-1]
        return v1 + (v1 - v0) / max(1, n1 - n0) * (n - n1)

    def domkey(dom):
        keys = sorted(set(list(excess.keys()) + list(chain.keys())))
        if not keys:
            return None
        return min(keys, key=lambda k: abs(math.log2(max(1, k)) -
                                           math.log2(max(1, dom))))

    def m3(d):
        b = base(d)
        if d["ns"] > 0 and d["seb"] <= 128:
            tab = chain.get(domkey(d["dom"])) if d["red"] else \
                excess.get(domkey(d["dom"]))
            return b + interp_pairs(tab or [], d["ns"])
        return b

    def mape(pred, subset=None):
        ds = [d for d in data if not subset or subset(d)]
        if not ds:
            return float("nan")
        return 100.0 * sum(abs(pred(d) - d["meas"]) / d["meas"]
                           for d in ds) / len(ds)

    sec = {
        "load_curves": {str(k): {"b": [p[0] for p in v],
                                 "ns": [p[1] for p in v]}
                        for k, v in load_curves.items()},
        "store_curves": {str(k): {"b": [p[0] for p in v],
                                  "ns": [p[1] for p in v]}
                         for k, v in store_curves.items()},
        "step_l": {str(k): v for k, v in stepL.items()},
        "step_s": {str(k): v for k, v in stepS.items()},
        "step_t": {str(k): v for k, v in stepT.items()},
        "c_l": c_l, "c_s": c_s,
        "additive": 1 if mode == "simt_only" else 0,
        "excess": {"%d" % k: v for k, v in excess.items()},
        "chain": {"%d" % k: v for k, v in chain.items()},
        "MAPE_all": {"M3": round(mape(m3), 1)},
        "MAPE_loads": {"M3": round(mape(m3, lambda d: d["kind"] == "L"), 1)},
        "MAPE_stores": {"M3": round(mape(m3, lambda d: d["kind"] == "S"), 1)},
        "MAPE_mix": {"M3": round(mape(m3, lambda d: d["kind"] == "M"), 1)},
        "MAPE_tiny": {"M3": round(mape(m3, lambda d: d["kind"] == "T"), 1)},
    }
    return sec, data, m3


def main():
    for mode in ("simd", "simt_only"):
        for dtype in DTYPES:
            data = []
            p = Path("/tmp/e27b_%s_%s.jsonl" % (dtype, mode))
            if not p.exists():
                print("MISSING", p)
                continue
            for line in p.read_text().splitlines():
                o = json.loads(line)
                if "data" in o:
                    data = o["data"]
            sec, _, m3 = fit(mode, dtype, data)
            Path("/tmp/loopfit6_%s_%s.json" % (mode, dtype)).write_text(
                json.dumps(sec, indent=1))
            print("%s %s: all=%.1f loads=%.1f stores=%.1f mix=%.1f "
                  "tiny=%.1f  c_l=%.3f c_s=%.3f" %
                  (mode, dtype, sec["MAPE_all"]["M3"],
                   sec["MAPE_loads"]["M3"], sec["MAPE_stores"]["M3"],
                   sec["MAPE_mix"]["M3"], sec["MAPE_tiny"]["M3"],
                   sec["c_l"], sec["c_s"]), flush=True)
    print("FIT6 DONE")


main()
