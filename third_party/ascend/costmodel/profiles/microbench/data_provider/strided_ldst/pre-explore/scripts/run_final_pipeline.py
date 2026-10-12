#!/usr/bin/env python3
"""End-to-end refit pipeline.

  1. read every raw 2D probe file from both campaigns
  2. build the per-campaign 2D datasets, then the unified dataset
  3. fit ONE formula per (slot, rank), <= 10 terms, optionally wrapped in max()
  4. report in-sample MAPE per formula and overall

The slot split follows the compiler behaviour verified in CAModel:

  simd load : inner stride <= 2 elements  -> contiguous-block fetch
                                           (biu_cmds 64 -> 1..9, B/cmd 4 -> 28..512)
              otherwise                   -> gather, one BIU command per element
  simd store, simt load, simt store       -> one formula each
  rank 1 vs rank 2                        -> separate formulas

Allowed forms: plain linear and max(); no logs, powers or exponentials.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_v9 import load, mape
import fit_v11_stepwise as SW
import fit_v9_pool as P

NAMES = list(P.POOL.keys())


def wls(A, t, W):
    return np.linalg.lstsq(A * W[:, None], t * W, rcond=None)[0]


def greedy(rr, t, mt, names=NAMES):
    """forward selection plus a backward drop; returns (features, coefs, mape)."""
    W = 1.0 / np.maximum(t, 1e-9)
    C = {k: np.array([P.POOL[k](r) for r in rr]) for k in names}
    one = np.ones(len(rr))

    def sc(s):
        A = np.column_stack([one] + [C[k] for k in s]); x = wls(A, t, W); return mape(t, A @ x)

    cur = []; best = sc(cur)
    while len(cur) < mt:
        cand, cb = None, best
        for k in names:
            if k in cur:
                continue
            m = sc(cur + [k])
            if m < cb - 1e-4:
                cb, cand = m, k
        if cand is None:
            break
        cur.append(cand); best = cb
        ch = True
        while ch and cur:
            ch = False
            for k in list(cur):
                m = sc([c for c in cur if c != k])
                if m <= best + 1e-4:
                    cur = [c for c in cur if c != k]; best = m; ch = True
                    break
    A = np.column_stack([one] + [C[k] for k in cur])
    return cur, wls(A, t, W), mape(t, A @ wls(A, t, W))


def slot_rows(rr0, slot, rk):
    if slot == "simd_load_wide":
        return [r for r in rr0 if r["path"] == "simd" and r["op"] == "load"
                and r["ISB"] <= 2.0 * r["eb"] and int(r["rank"]) == rk]
    if slot == "simd_load_gather":
        return [r for r in rr0 if r["path"] == "simd" and r["op"] == "load"
                and r["ISB"] > 2.0 * r["eb"] and int(r["rank"]) == rk]
    if slot == "simd_store":
        return [r for r in rr0 if r["path"] == "simd" and r["op"] == "store" and int(r["rank"]) == rk]
    return [r for r in rr0 if r["path"] == "simt_only" and r["op"] == slot.split("_")[1]
            and int(r["rank"]) == rk]


def fit_slot(rr):
    """Returns (mape, entry).  Tries plain linear and max(limit, linear)."""
    t = np.array([r["T"] for r in rr]); Wv = np.array([float(r["W"]) for r in rr])
    c, x, m_lin = greedy(rr, t, 10)
    best = (m_lin, {"form": "linear", "intercept": round(float(x[0]), 6),
                    "terms": [{"feature": k, "coef": round(float(v), 6)} for k, v in zip(c, x[1:])]})
    # max(c * num_warps, memory): the board shows a per-warp launch cost that is
    # flat in the element count for num_warps >= 16.
    if len(set(Wv)) > 1 and (Wv >= 16).sum() >= 8:
        a = float(np.median(t[Wv >= 16] / Wv[Wv >= 16]))
        cc = c
        for _ in range(4):
            memdom = t > a * Wv * 1.05
            if memdom.sum() < 40:
                break
            cc, xc, _ = greedy([r for i, r in enumerate(rr) if memdom[i]], t[memdom], 8)
            Ac = np.column_stack([np.ones(len(rr))] + [[P.POOL[f](r) for r in rr] for f in cc])
            memall = Ac @ wls(Ac, t, 1 / np.maximum(t, 1e-9))
            wdom = a * Wv > memall
            if wdom.sum() >= 8:
                a = float(np.median(t[wdom] / Wv[wdom]))
        Ac = np.column_stack([np.ones(len(rr))] + [[P.POOL[f](r) for r in rr] for f in cc])
        xc = wls(Ac, t, 1 / np.maximum(t, 1e-9))
        m_max = mape(t, np.maximum(a * Wv, Ac @ xc))
        if m_max < m_lin - 0.3:
            best = (m_max, {"form": "max(%.4g*num_warps, linear)" % a, "a_ns_per_warp": round(a, 3),
                            "intercept": round(float(xc[0]), 6),
                            "terms": [{"feature": k, "coef": round(float(v), 6)} for k, v in zip(cc, xc[1:])]})
    return best


def build_datasets(raw_dirs, outdir):
    """raw_dirs: list of (raw_dir, tag).  Produces <outdir>/dataset_2d_<tag>.csv."""
    made = []
    for raw, tag in raw_dirs:
        raw = Path(raw)
        if not raw.exists() or not list(raw.glob("*.json")):
            print("  skip (empty): %s" % raw)
            continue
        dst = Path(outdir) / ("build_" + tag)
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(HERE / "build_2d_dataset.py"),
                        "--raw-dir", str(raw), "--out-dir", str(dst)], check=True,
                       stdout=subprocess.DEVNULL)
        made.append(dst / "dataset_2d.csv")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild the 2D + unified datasets first")
    ap.add_argument("--unified", type=Path,
                    default=HERE.parent / "results/final/dataset_unified_v4.csv")
    args = ap.parse_args()

    if args.build:
        made = build_datasets(
            [(HERE.parent / "results/expand2d/raw", "c1"),
             (HERE.parent / "results/expand2d_b/raw", "c2")],
            HERE.parent / "results/_build")
        cmd = [sys.executable, str(HERE / "build_unified_dataset.py"),
               "--out", str(args.unified), "--exclude-template",
               "--v1-1d", str(HERE.parent / "results/final/dataset.csv"),
               "--v1-2d", str(HERE.parent / "results/final/dataset_2d.csv")]
        for i, p in enumerate(made):
            cmd += ["--v%s-2d" % (3 + i), str(p)]
        subprocess.run(cmd, check=True)
        # new-only view used for fitting
        rows = list(csv.DictReader(args.unified.open(newline="")))
        new = [r for r in rows if r["source"].startswith(("v1_", "v2_", "v3_", "v4_"))]
        cols = list(rows[0].keys())
        npath = args.unified.parent / "dataset_new_only_v5.csv"
        with npath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(new)
        print("new-only rows = %d -> %s" % (len(new), npath))

    npath = args.unified.parent / "dataset_new_only_v5.csv"
    if not npath.exists():
        import csv as _csv
        _rows = list(_csv.DictReader(args.unified.open(newline="")))
        _new = [r for r in _rows if r["source"].startswith(("v1_", "v2_", "v3_", "v4_"))]
        _cols = list(_rows[0].keys())
        with npath.open("w", newline="") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=_cols); _w.writeheader(); _w.writerows(_new)
        print("new-only rows = %d -> %s" % (len(_new), npath))
    rows = load(str(npath))
    rr0 = [r for r in rows if str(r.get("valid")) == "1"]
    print("fitting on %d probe-valid points" % len(rr0))
    out = {}; TOT = 0.0; N = 0
    print("%-24s %6s %6s %9s   %s" % ("formula", "n", "terms", "MAPE", "form"))
    for slot in ("simd_load_wide", "simd_load_gather", "simd_store", "simt_load", "simt_store"):
        for rk in (1, 2):
            rr = slot_rows(rr0, slot, rk)
            if len(rr) < 30:
                continue
            m, entry = fit_slot(rr)
            key = "%s_r%d" % (slot, rk)
            entry.update({"n": len(rr), "mape": round(m, 2)})
            out[key] = entry
            TOT += m * len(rr); N += len(rr)
            print("%-24s %6d %6d %8.2f%%   %s" % (key, len(rr), len(entry["terms"]) + 1, m, entry["form"]),
                  flush=True)
    out["_overall"] = round(TOT / N, 2)
    outp = HERE.parent / "results/final/MODEL_FINAL_V7.json"
    json.dump(out, outp.open("w"), indent=1)
    print("\noverall in-sample MAPE = %.2f%%   -> %s" % (TOT / N, outp))


if __name__ == "__main__":
    main()
