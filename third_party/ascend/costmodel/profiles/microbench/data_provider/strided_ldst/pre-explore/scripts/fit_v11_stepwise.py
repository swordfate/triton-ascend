#!/usr/bin/env python3
"""Forward-backward stepwise selection (features CAN be swapped back out),
non-negative relative-weighted least squares, up to 10 terms."""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_v9 import load, wnnls, mape
import fit_v10_exhaustive as EX
import fit_v9_pool as P

# ---- register EVERY f_* column the dataset carries (the v8 formulas used
#      several of these and the hand-written pool had missed them) ----------
import csv as _csv
_ds = HERE.parent / "results/final/dataset_new_only_v2.csv"
with _ds.open(newline="") as _fh:
    _cols = [c for c in next(_csv.reader(_fh)) if c.startswith("f_")]
_extra = {}
for _c in _cols:
    _k = _c[2:]
    if _k in P.POOL:
        continue
    _extra[_k] = (lambda k: (lambda r: float(r.get("f_" + k) or 0.0)))(_k)
P.POOL.update(_extra)

# ---- ratios / products between the fundamental quantities -------------------
# Physically meaningful compositions: how many elements a warp covers, how many
# lines per thread, how many rows per iteration, bytes per line, ...
def _Q(r):
    import math
    n = float(r["n"]); W = max(1.0, float(r["W"])); T = 32.0 * W
    R = float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0
    C = float(r.get("f_cols") or n) if int(r["rank"]) == 2 else n
    L = float(r["L"]); eb = float(r["eb"]); ISB = float(r["ISB"])
    span = float(r["span"]); it = float(r["iters"])
    mwl = float(r.get("f_mean_warp_lines") or 0.0)
    pg = float(r["pages"])
    return dict(n=n, R=R, C=C, L=L, W=W, T=T, eb=eb, ISB=ISB, spanb=span * eb,
                it=it, mwl=mwl, pg=pg, rl=float(r.get("f_row_lines") or 0.0),
                rs=float(r.get("f_row_span_bytes") or 0.0))
_RP = {}
def _mk(a, b, mode):
    if mode == "ratio":
        return lambda r, a=a, b=b: _Q(r)[a] / max(1e-9, _Q(r)[b])
    return lambda r, a=a, b=b: _Q(r)[a] * _Q(r)[b]
_KEYS = ["n", "R", "C", "L", "W", "T", "eb", "ISB", "spanb", "it", "mwl", "pg", "rl", "rs"]
for _i in range(len(_KEYS)):
    for _j in range(_i + 1, len(_KEYS)):
        a, b = _KEYS[_i], _KEYS[_j]
        _RP["r_%s_%s" % (a, b)] = _mk(a, b, "ratio")
        _RP["p_%s_%s" % (a, b)] = _mk(a, b, "prod")
P.POOL.update(_RP)


# ---- SIMT LSU: a 32-thread warp covers 32 consecutive flattened elements, so it
#      spans ceil(32/C) rows.  When C < 32 one warp straddles several rows and
#      the access turns from (strided but single-row) into multi-row -> ~1.5-2x.
#      Verified on the full grid: controlled ratio median 1.50, p90 2.44. ----
def _C(r):
    return float(r.get("f_cols") or 1.0) if int(r["rank"]) == 2 else float(r["n"])
def _rowsPerWarp(r):
    import math
    return math.ceil(32.0 / max(1.0, _C(r)))
P.POOL.update({
    "C":        _C,
    "warpUnder":lambda r: max(0.0, 32.0 - _C(r)),
    "warpUnderR":lambda r: max(0.0, 32.0 - _C(r)) * _rowsPerWarp(r),
    "rowsPerWarp": _rowsPerWarp,
    "warpFlag": lambda r: 1.0 if _C(r) < 32.0 else 0.0,
    "wuL":      lambda r: max(0.0, 32.0 - _C(r)) * r["L"] / 32.0,
    "wuN":      lambda r: max(0.0, 32.0 - _C(r)) * r["n"] / 32.0,
    "wuSeg":    lambda r: max(0.0, 32.0 - _C(r)) * (float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0),
    "wuIters":  lambda r: max(0.0, 32.0 - _C(r)) * r["iters"] / 32.0,
    "mwlpR":    lambda r: float(r.get("f_mean_warp_lines") or 0.0) * _rowsPerWarp(r),
})


# ---- systematic pairwise max/min / hinge family -------------------------------
# Each of these is a physically meaningful composition:  two resources in
# series (max), a saturated resource (min), or the excess of one over the other.
_BASEQ = {}
for _n, _f in (("L", lambda r: r["L"]), ("n", lambda r: r["n"]),
               ("seg", lambda r: float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0),
               ("mwl", lambda r: float(r.get("f_mean_warp_lines") or 0.0)),
               ("iters", lambda r: r["iters"]), ("W", lambda r: float(r["W"])),
               ("pages", lambda r: r["pages"]),
               ("cshortN", lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) * r["n"] / 32.0),
               ("rowlines", lambda r: float(r.get("f_row_lines") or 0.0))):
    _BASEQ[_n] = _f
_PAIRS = [("L", "seg"), ("L", "mwl"), ("L", "iters"), ("L", "W"), ("L", "pages"),
          ("n", "seg"), ("n", "mwl"), ("n", "W"), ("iters", "W"), ("iters", "mwl"),
          ("seg", "mwl"), ("seg", "W"), ("cshortN", "L"), ("rowlines", "seg"),
          ("L", "cshortN"), ("mwl", "seg")]
_EXTRA2 = {}
for _a, _b in _PAIRS:
    _EXTRA2["mx_%s_%s" % (_a, _b)] = (lambda r, a=_a, b=_b: max(_BASEQ[a](r), _BASEQ[b](r)))
    _EXTRA2["mn_%s_%s" % (_a, _b)] = (lambda r, a=_a, b=_b: min(_BASEQ[a](r), _BASEQ[b](r)))
    _EXTRA2["ov_%s_%s" % (_a, _b)] = (lambda r, a=_a, b=_b: max(0.0, _BASEQ[a](r) - _BASEQ[b](r)))
    _EXTRA2["rt_%s_%s" % (_a, _b)] = (lambda r, a=_a, b=_b: _BASEQ[a](r) / max(1e-9, _BASEQ[b](r)))
P.POOL.update(_EXTRA2)


# ---- the compiler switches the SIMD load to a CONTIGUOUS-BLOCK fetch when
#      inner stride <= 2 elements (verified in CAModel: biu_cmds 64 -> 1..9,
#      bytes/cmd 4 -> 28..512).  So for those cases the cost driver is the
#      contiguous footprint, not the number of touched lines. ----
def _spanL(r):
    import math
    return math.ceil((r["span"] * r["eb"]) / 128.0)
def _packedL(r):
    import math
    return math.ceil((r["n"] * r["eb"]) / 128.0)
P.POOL.update({
    "spanL":   _spanL,        # contiguous footprint in 128B lines
    "packedL": _packedL,      # payload-only lines
    "wasteL":  lambda r: max(0.0, _spanL(r) - _packedL(r)),
    "spanperN":lambda r: _spanL(r) / max(1.0, r["n"]),
    "spanL2":  lambda r: _spanL(r) * _spanL(r),
})


def _g(r, k, d=0.0):
    try:
        v = r.get(k)
        return float(v) if v not in (None, "") else d
    except Exception:
        return d
# ---- regime indicator family: [quantity <= threshold] and its interactions ----
_TH = {
    "le_n128":   ("n", 128.0), "le_n256": ("n", 256.0), "le_n64": ("n", 64.0),
    "le_R8":     ("rows", 8.0), "le_R4": ("rows", 4.0), "le_R16": ("rows", 16.0),
    "le_mwl8":   ("f_mean_warp_lines", 8.0), "le_mwl16": ("f_mean_warp_lines", 16.0),
    "le_lem2":   ("f_line_elems_max", 2.0), "le_lem16": ("f_line_elems_max", 16.0),
    "le_isb32":  ("ISB", 32.0), "le_isb64": ("ISB", 64.0),
    "le_L32":    ("L", 32.0), "le_L64": ("L", 64.0), "le_L128": ("L", 128.0),
    "le_span16k":("span", 16384.0), "le_rowl8": ("f_row_lines", 8.0),
}
_IND = {}
for _nm, (_q, _t) in _TH.items():
    if _q == "rows":
        _IND[_nm] = (lambda r, t=_t: 1.0 if (_g(r, "f_rows", 1.0) if int(r["rank"]) == 2 else 1.0) <= t else 0.0)
    elif _q == "ISB":
        _IND[_nm] = (lambda r, t=_t: 1.0 if r["ISB"] <= t else 0.0)
    elif _q == "n":
        _IND[_nm] = (lambda r, t=_t: 1.0 if r["n"] <= t else 0.0)
    elif _q == "L":
        _IND[_nm] = (lambda r, t=_t: 1.0 if r["L"] <= t else 0.0)
    elif _q == "span":
        _IND[_nm] = (lambda r, t=_t: 1.0 if r["span"] <= t else 0.0)
    else:
        _IND[_nm] = (lambda r, t=_t, q=_q: 1.0 if _g(r, q) <= t else 0.0)
P.POOL.update(_IND)
# interactions of the regime flags with the main cost drivers
for _nm in list(_IND):
    P.POOL[_nm + "xL"] = (lambda r, f=_IND[_nm]: f(r) * r["L"])
    P.POOL[_nm + "xN"] = (lambda r, f=_IND[_nm]: f(r) * r["n"])


def _W(r):
    return float(r["W"])
# T = max(warp_launch_k*W, memory_cost)  =>  k*W + max(0, memory - k*W).
# Represent the second half with 2-D hinges on (memory proxy, W).
_HINGE_W = {}
for _a in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    for _src in ("L", "n"):
        _HINGE_W["h%sW%s" % (_src, str(_a).replace(".", ""))] = (
            lambda r, a=_a, s=_src: max(0.0, float(r[s]) - a * _W(r)))
_HINGE_W["hLW"]   = lambda r: max(0.0, r["L"] - _W(r))
_HINGE_W["hNW32"] = lambda r: max(0.0, r["n"] / 32.0 - _W(r))
_HINGE_W["hLdivW"]= lambda r: r["L"] / max(1.0, _W(r))
_HINGE_W["hNdivW"]= lambda r: r["n"] / max(1.0, _W(r))
_HINGE_W["warpNs"]= lambda r: 2.09 * _W(r)
_HINGE_W["hLmwl"] = lambda r: max(0.0, r["L"] - float(r.get("f_mean_warp_lines") or 0.0) * _W(r))
P.POOL.update(_HINGE_W)


def _thr(r):
    return 32.0 * float(r["W"])
def _ept(r):
    return r["n"] / max(1.0, _thr(r))
P.POOL.update({
    # ---- threads = 32*num_warps; relation between elements and threads ----
    "thr":     _thr,
    "ept":     _ept,                                    # elements per thread
    "rounds":  lambda r: float(np.ceil(_ept(r))),       # issue rounds per thread
    "active":  lambda r: min(r["n"], _thr(r)),          # threads that get work
    "idle":    lambda r: max(0.0, _thr(r) - r["n"]),    # threads that get none
    "util":    lambda r: min(1.0, _ept(r)),             # thread utilisation 0..1
    "dupcap":  lambda r: max(0.0, 1.0 / _ept(r) - 1.0) if _ept(r) > 0 else 0.0,
    "idual":   lambda r: 1.0 - min(1.0, _ept(r)),       # idle fraction
    "eptL":    lambda r: _ept(r) * r["L"],
    "eptN":    lambda r: _ept(r) * r["n"],
    "roundsL": lambda r: float(np.ceil(_ept(r))) * r["L"],
    "roundsMwl": lambda r: float(np.ceil(_ept(r))) * float(r.get("f_mean_warp_lines") or 0.0),
    "thrL":    lambda r: _thr(r) * r["L"],
    "lowthread": lambda r: 1.0 if _ept(r) >= 1.0 else 0.0,   # >=1 element per thread
    "hiept1":  lambda r: max(0.0, _ept(r) - 1.0),
    "hiept2":  lambda r: max(0.0, _ept(r) - 2.0),
})


def _mwl(r):
    return float(r.get("f_mean_warp_lines") or 0.0)
P.POOL.update({
    # ---- SIMT LSU: floor + linear ramp in mean warp lines (verified on board:
    #      flat ~25ns then ~7-9ns per extra line, breakpoint near mwl=2) ----
    "mwl":      _mwl,
    "hmwl1":    lambda r: max(0.0, _mwl(r) - 1.0),
    "hmwl2":    lambda r: max(0.0, _mwl(r) - 2.0),
    "hmwl3":    lambda r: max(0.0, _mwl(r) - 3.0),
    "hmwl4":    lambda r: max(0.0, _mwl(r) - 4.0),
    "mwllog":   lambda r: _mwl(r),
    "mwlL":     lambda r: _mwl(r) * r["L"],
    "mwlN":     lambda r: _mwl(r) * r["n"],
    "xmwl":     lambda r: float(r.get("f_max_warp_lines") or 0.0),
    "hxmwl2":   lambda r: max(0.0, float(r.get("f_max_warp_lines") or 0.0) - 2.0),
    "minmwl":   lambda r: float(r.get("f_min_warp_lines") or 0.0),
})


P.POOL.update({
    # ---- MTE 32-byte minimum-burst THRESHOLD (verified: SIMD 2-4x slower when
    #      C*eb < 32, no effect on the SIMT LSU path) ----
    "burstShort": lambda r: 1.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 0.0,
    "bsL":        lambda r: (1.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 0.0) * r["L"],
    "bsN":        lambda r: (1.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 0.0) * r["n"],
    "bsPages":    lambda r: (1.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 0.0) * r["pages"],
    "bsSeg":      lambda r: (1.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 0.0) * (float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0),
    "burstLong":  lambda r: 0.0 if float(r.get("f_cols") or 1.0) * r["eb"] < 32.0 else 1.0,
})


P.POOL.update({
    # ---- physical composites: dtype/wp adaptation baked into the DEFINITION ----
    # MTE: a row is one burst, padded up to the 32-byte minimum
    "bursts":   lambda r: float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0,
    "padB":     lambda r: max(32.0, float(r.get("f_cols") or 1.0) * r["eb"]) / 32.0,
    "burstVol": lambda r: (float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0)
                          * max(32.0, float(r.get("f_cols") or 1.0) * r["eb"]) / 32.0,
    "padWaste": lambda r: (float(r.get("f_rows") or 1.0) if int(r["rank"]) == 2 else 1.0)
                          * max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) / 32.0,
    # SIMT: warp rounds x distinct lines per wave = memory transactions
    "trans":    lambda r: r["iters"] * float(r.get("f_mean_warp_lines") or 1.0),
    "transX":   lambda r: r["iters"] * float(r.get("f_max_warp_lines") or 1.0),
    "transL":   lambda r: r["iters"] * min(r["L"], 32.0),
    # bytes actually requested vs bytes fetched
    "reqBytes": lambda r: r["n"] * r["eb"] / 32.0,
    "padBytes": lambda r: r["n"] * max(32.0, float(r.get("f_cols") or 1.0) * r["eb"]) / 32.0,
    "txBytes":  lambda r: r["iters"] * float(r.get("f_mean_warp_lines") or 1.0) * r["eb"],
})


P.POOL.update({
    # ---- MTE burst-length physics: a row is one burst of C*eb bytes ----
    "cbytes":  lambda r: float(r.get("f_cols") or 1.0) * r["eb"],
    "cshort":  lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]),
    "cshortL": lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) * r["L"] / 128.0,
    "cburst":  lambda r: float(r.get("f_cols") or 1.0) * r["eb"] / 32.0,
    "minCB":   lambda r: min(float(r.get("f_cols") or 1.0) * r["eb"], 32.0),
    "cshortR": lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) * float(r.get("f_rows") or 1.0),
    "cshortN": lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) * r["n"] / 32.0,
    "cshortL2":lambda r: max(0.0, 32.0 - float(r.get("f_cols") or 1.0) * r["eb"]) * r["L"] / 32.0,
    "burstEff":lambda r: min(1.0, float(r.get("f_cols") or 1.0) * r["eb"] / 32.0),
    "cbL":     lambda r: float(r.get("f_cols") or 1.0) * r["eb"] * r["L"] / 32.0,
    "cbR":     lambda r: float(r.get("f_cols") or 1.0) * r["eb"] * float(r.get("f_rows") or 1.0) / 32.0,
})


P.POOL.update({
    # span/eb = n * element-stride, the SIMT ramp variable; breakpoints from
    # the per-subgroup fits (threshold sits near 128-512, not 1024+).
    "se": lambda r: r["span"] / r["eb"],
    "seEb": lambda r: r["span"] / r["eb"] * r["eb"],
    "seDEb": lambda r: r["span"] / r["eb"] / r["eb"],
    **{("hse%d" % k): (lambda kk: (lambda r: max(0.0, r["span"] / r["eb"] - kk)))(k)
       for k in (32, 64, 128, 256, 512, 1024, 2048, 4096)},
    **{("mse%d" % k): (lambda kk: (lambda r: min(r["span"] / r["eb"], kk)))(k)
       for k in (32, 64, 128, 256, 512, 1024, 2048, 4096)},
    **{("hseE%d" % k): (lambda kk: (lambda r: r["eb"] * max(0.0, r["span"] / r["eb"] - kk)))(k)
       for k in (128, 256, 512, 1024)},
    **{("hseD%d" % k): (lambda kk: (lambda r: max(0.0, r["span"] / r["eb"] - kk) / r["eb"]))(k)
       for k in (128, 256, 512, 1024)},
    **{("hseW%d" % k): (lambda kk: (lambda r: max(0.0, r["span"] / r["eb"] - kk) / (32.0 * r["W"])))(k)
       for k in (128, 256, 512, 1024)},
    **{("hLebe%d" % k): (lambda kk: (lambda r: max(0.0, r["L"] / r["eb"] - kk)))(k)
       for k in (0.5, 1, 2, 4, 8, 16, 32)},
    **{("mLebe%d" % k): (lambda kk: (lambda r: min(r["L"] / r["eb"], kk)))(k)
       for k in (0.5, 1, 2, 4, 8, 16, 32)},
    **{("hLebW%d" % k): (lambda kk: (lambda r: max(0.0, r["L"] / r["eb"] - kk) / (32.0 * r["W"])))(k)
       for k in (1, 2, 4, 8)},
});


P.POOL.update({
    # ---- rank-2 comb geometry ----
    "LperR": lambda r: r["L"] / max(1.0, float(r.get("f_rows") or 1.0)),
    "R": lambda r: float(r.get("f_rows") or 1.0),
    "Rgapline": lambda r: float(r.get("f_rows") or 1.0) * float(r.get("f_gap_per_line") or 0.0),
    "duty": lambda r: (float(r.get("f_row_span_bytes") or 0.0)
                       / max(1.0, float(r.get("f_row_pitch_bytes") or 1.0))),
    "pitchL": lambda r: float(r.get("f_row_pitch_bytes") or 0.0) / 128.0,
    "RspanL": lambda r: float(r.get("f_rows") or 1.0) * float(r.get("f_row_span_bytes") or 0.0) / 128.0,
    "R2": lambda r: float(r.get("f_rows") or 1.0) ** 2,
    "RL": lambda r: float(r.get("f_rows") or 1.0) * r["L"],
    "RpitchL": lambda r: float(r.get("f_rows") or 1.0) * float(r.get("f_row_pitch_bytes") or 0.0) / 128.0,
    "combgap": lambda r: (float(r.get("f_comb_lines") or 0.0) - r["L"]),
    "LperPitch": lambda r: r["L"] / max(1.0, float(r.get("f_row_pitch_bytes") or 1.0) / 128.0),
    "rowOverL": lambda r: float(r.get("f_rows") or 1.0) / max(1.0, r["L"]),
    "spanPerPitch": lambda r: r["span"] / max(1.0, float(r.get("f_row_pitch_bytes") or 1.0)),
    "pitchGap": lambda r: float(r.get("f_row_pitch_bytes") or 0.0) / max(1.0, float(r.get("f_row_span_bytes") or 1.0)),
    "Rcols": lambda r: float(r.get("f_cols") or 1.0),
    "RC": lambda r: float(r.get("f_rows") or 1.0) * float(r.get("f_cols") or 1.0),
    "LmR": lambda r: max(0.0, r["L"] - float(r.get("f_rows") or 1.0)),
})


def stepwise(rr, keys, maxterms=10, tol=1e-4):
    t = np.array([r["T"] for r in rr]); W = 1.0 / np.maximum(t, 1e-9)
    C = {k: np.array([P.POOL[k](r) for r in rr]) for k in keys}
    one = np.ones(len(rr))

    def score(sel):
        A = np.column_stack([one] + [C[k] for k in sel])
        x = wnnls(A, t, W)
        return mape(t, A @ x), x

    cur = []
    best, _ = score(cur)                      # intercept only
    improved = True
    while improved and len(cur) < maxterms:
        improved = False
        # forward
        cand, cbest = None, best
        for k in keys:
            if k in cur:
                continue
            m, _ = score(cur + [k])
            if m < cbest - tol:
                cbest, cand = m, k
        if cand is not None:
            cur.append(cand); best = cbest; improved = True
        # backward
        changed = True
        while changed and cur:
            changed = False
            for k in list(cur):
                s2 = [c for c in cur if c != k]
                m, _ = score(s2)
                if m <= best + tol:
                    cur = s2; best = m; changed = True; break
    m, x = score(cur)
    return m, cur, x


def main():
    ds = str(HERE.parent / "results/final/dataset_new_only_v2.csv")
    rows = load(ds)
    KEYS = list(P.POOL.keys())
    print('pool', len(KEYS))
    print("dataset rows=%d pool=%d" % (len(rows), len(KEYS)))
    out = {}
    for slot in ("simd_load_wide", "simd_load_gather", "simd_store", "simt_load", "simt_store"):
        rr = EX.sub(rows, slot)
        m, combo, x = stepwise(rr, KEYS, maxterms=10)
        print("\n=== %s n=%d  MAPE=%.2f%% ===" % (slot, len(rr), m))
        print("   " + " + ".join("%.4g*%s" % (c, k) for k, c in zip(combo, x[1:])))
        A = np.column_stack([np.ones(len(rr))] + [np.array([P.POOL[c](r) for r in rr]) for c in combo])
        pr = A @ x; t = np.array([r["T"] for r in rr])
        perd = {}
        for d in ("int8", "float16", "float32", "int64"):
            i = [j for j, r in enumerate(rr) if r["dtype"] == d]
            if i: perd[d] = round(mape(t[i], pr[i]), 2)
        out[slot] = {"n": len(rr), "mape": round(m, 3),
                     "terms": [{"feature": "intercept", "coef": round(float(x[0]), 6)}] +
                              [{"feature": k, "coef": round(float(c), 6)} for k, c in zip(combo, x[1:])],
                     "per_dtype": perd}
        print("   per-dtype:", perd)
    T = sum(v["mape"] * v["n"] for v in out.values()); N = sum(v["n"] for v in out.values())
    out["_overall"] = round(T / N, 3)
    json.dump(out, open(HERE.parent / "results/final/fit_v11_stepwise.json", "w"), indent=1)
    print("\nOVERALL weighted MAPE = %.2f%%" % (T / N))


if __name__ == "__main__":
    main()
