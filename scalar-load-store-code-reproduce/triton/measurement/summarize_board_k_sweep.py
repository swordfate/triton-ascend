#!/usr/bin/env python3
"""Summarize diff-line board K-sweeps against the refit cycle formulas.

Inputs:
  results/{simd,simt_only}_{load,store}.json  (2026-09-27 multi-ptr K sweep)
  store_sync_results/store_variant_results.json
      (2026-09-28 store retest, variant 0 = matched sink baseline)
  store_recheck_results/*.json
      (2026-09-28 fresh SIMD/SIMT store reruns)

Metric choice:
  load  : Duration(us)_delta for both modes
  store : SIMT Duration(us)_delta; SIMD aiv_mte3_time(us)_delta
          (the SIMD scalar store is MTE3-async, so kernel Duration has no signal)

Store K=8: the original retest run saw a non-reproducible 229 ns (min 222 / max
236).  Three fresh runs (66.5/57.5/61.5 ns) and the matched K=8 probe (61.5 ns)
are used instead.  All other store K use the cross-run median.

Output: results/scalar_k_board_summary.{csv,md}
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STORE_FILES = [HERE / "store_sync_results" / "store_variant_results.json"]
STORE_FILES += sorted((HERE / "store_recheck_results").glob("*.json"))


def pred_cycle(mode, kind, k):
    """Refit formulas in raw CAModel cycle domain (README-diff-k-sweep.md)."""
    k = max(1, k)
    if kind == "load":
        if mode == "simd":
            return (7.0 + 263.9 + max(0.0, k - 1.0) * 278.74285714285713 +
                    (k - 1.0) * 1.0)
        return (6.0 + 306.75 + max(0.0, k - 2.0) * 74.775 +
                (k - 1.0) * 1.0)
    # Stores: SIMD uses the aiv_mte3_time activation model; SIMT uses the
    # cross-run matched-sink linear model.
    if mode == "simd":
        fill = 11.7
        if k <= 1:
            return fill
        return fill + 95.4 + (k - 2.0) * 4.5
    return 30.4 + (k - 1.0) * 12.348  # subsequent


def load_rows():
    rows = []
    for name in ["simd_load", "simt_load", "simd_store", "simt_store"]:
        data = json.loads((RESULTS / f"{name}.json").read_text())
        for c in data["cases"]:
            if c.get("variant", "diff") != "diff":
                continue
            # Store rows here are the old probe; replaced by the retest data.
            if c["kind"] == "store":
                continue
            rows.append(c)
    out = []
    for c in rows:
        mode, kind, k = c["mode"], c["kind"], c["K"]
        measured_us = c["Duration(us)_delta"]
        measured_cycle = measured_us * 1000.0 * 1.8
        pred = pred_cycle(mode, kind, k)
        err = (measured_cycle - pred) / pred * 100.0
        if abs(err) < 0.05:
            err = 0.0
        out.append({
            "mode": mode,
            "kind": kind,
            "K": k,
            "metric": "Duration(us)_delta",
            "measured_us": round(measured_us, 4),
            "measured_cycle": round(measured_cycle, 1),
            "pred_cycle": round(pred, 1),
            "err_pct": f"{err:+.1f}%",
            "note": "",
        })
    return out


def store_runs():
    runs = []
    for path in STORE_FILES:
        data = json.loads(path.read_text())
        for c in data["cases"]:
            if c.get("variant") == 0:
                runs.append((path, c))
    return runs


def store_rows():
    runs = store_runs()
    out = []
    for mode in ("simd", "simt_only"):
        metric = "aiv_mte3_time(us)_delta" if mode == "simd" else "Duration(us)_delta"
        metric_label = ("aiv_mte3_time(us)_delta" if mode == "simd"
                        else "Duration(us)_delta (matched sink)")
        for k in (1, 2, 4, 8):
            vals = []
            note = ""
            for path, c in runs:
                if c["mode"] != mode or c["K"] != k:
                    continue
                # The original K=8 run is a non-reproducible runtime outlier.
                if mode == "simt_only" and k == 8 and path.parent.name == "store_sync_results":
                    note = ("original K=8 229 ns outlier excluded; "
                            "fresh runs + matched K=8 used")
                    continue
                value = c[metric]
                if value is not None:
                    vals.append(value)
            med_us = statistics.median(vals)
            measured_cycle = med_us * 1000.0 * 1.8
            pred = pred_cycle(mode, "store", k)
            err = (measured_cycle - pred) / pred * 100.0
            if abs(err) < 0.05:
                err = 0.0
            out.append({
                "mode": mode,
                "kind": "store",
                "K": k,
                "metric": metric_label,
                "measured_us": round(med_us, 4),
                "measured_cycle": round(measured_cycle, 1),
                "pred_cycle": round(pred, 1),
                "err_pct": f"{err:+.1f}%",
                "note": note,
            })
    return out


def main():
    out = load_rows() + store_rows()
    out.sort(key=lambda r: (r["mode"], r["kind"], r["K"]))

    csv_path = RESULTS / "scalar_k_board_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    lines = [
        "# Diff-line board K-sweep vs refit cycle formulas",
        "",
        "`measured_cycle = measured_us * 1000 * 1.8`; `err = (measured - pred) / pred`.",
        "Loads use `Duration(us)_delta`; SIMT store uses matched-sink `Duration(us)_delta`;",
        "SIMD store uses `aiv_mte3_time(us)_delta` because the scalar store is MTE3-async.",
        "Store K=8 original run (229 ns) is a non-reproducible outlier and is excluded;",
        "cross-run/fresh medians are used (SIMT K=8 median 61.5 ns).",
        "",
        "| mode | kind | K | metric | measured us | measured cycle | pred cycle | err | note |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in out:
        lines.append(
            f"| {r['mode']} | {r['kind']} | {r['K']} | {r['metric']} | "
            f"{r['measured_us']:.4f} | {r['measured_cycle']:.1f} | "
            f"{r['pred_cycle']:.1f} | {r['err_pct']} | {r['note']} |")
    md_path = RESULTS / "scalar_k_board_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("WROTE", csv_path)
    print("WROTE", md_path)
    for r in out:
        print(f"{r['mode']:9s} {r['kind']:5s} K={r['K']}: "
              f"meas={r['measured_cycle']:.1f} cyc pred={r['pred_cycle']:.1f} "
              f"({r['err_pct']}) {r['note']}")


if __name__ == "__main__":
    main()
