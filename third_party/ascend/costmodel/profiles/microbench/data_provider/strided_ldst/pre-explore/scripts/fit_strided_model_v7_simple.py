#!/usr/bin/env python3
"""Simple non-negative additive strided-load model v7.

This is the user-facing simplification after v6:

  * no target log;
  * no powers / squares / cubes;
  * no negative coefficients;
  * all fits are raw-domain weighted non-negative least squares.

SIMD wide/gather keep the v6 formulas.  SIMT is replaced by four simple
derived terms:

    T_simt = c0
           + c1 * dupL
           + c2 * min(L, 64)
           + c3 * max(0, W*L - 3072)
           + c4 * dup * cross

where dupL = dup*L is the replicated-line request count, min(L,64) is the
line work before the saturated regime, max(0, W*L-3072) is warp-line work
above the in-flight saturation threshold, and dup*cross is underfilled
replication on page-crossing loads.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_strided_model_common import (
    ELEM_BYTES,
    LINE_BYTES,
    PAGE_BYTES,
    SIMD_DATASET,
    SIMT_DATASET,
    error_metrics,
    fit_weighted,
    model_payload,
    simd_gather_design,
    simd_wide_design,
    write_error_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent  # pre-explore root (scripts/ lives directly under it)
DEFAULT_OUT = HERE / "results/model_v7_simple"


def simt_simple_design(rows):
    block = rows["block"].to_numpy(dtype=float)
    stride = rows["stride"].to_numpy(dtype=float)
    num_warps = rows["num_warps"].to_numpy(dtype=float)
    cross = rows["cross_page"].to_numpy(dtype=float)
    lines = rows["L"].to_numpy(dtype=float)

    span = (block - 1.0) * stride * ELEM_BYTES
    e = block / (32.0 * num_warps)
    dup = np.maximum(0.0, 1.0 / np.maximum(e, 1e-12) - 1.0)

    dup_l = dup * lines
    l_cap = np.minimum(lines, 64.0)
    wl_tail = np.maximum(0.0, (num_warps * lines) - 3072.0)
    dup_page = dup * cross

    X = np.column_stack([
        np.ones(len(rows)),
        dup_l,
        l_cap,
        wl_tail,
        dup_page,
    ])
    names = ["1", "dupL", "min_L_64", "max_WL_minus_3072", "dup_cross"]
    return X, names


def fit_group(rows, design_fn, alpha):
    X, names = design_fn(rows)
    y = rows["target"].to_numpy(dtype=float)
    theta = fit_weighted(X, y, alpha)
    pred = X @ theta
    return {
        "rows": rows,
        "design": X,
        "names": names,
        "theta": theta,
        "pred": pred,
        "metrics": error_metrics(y, pred),
        "alpha": alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    simd = pd.read_csv(SIMD_DATASET)
    simd = simd[simd["valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    simd["target"] = simd["target_ns"].astype(float)
    wide_rows = simd[simd["path"] == "simd_wide"].reset_index(drop=True)
    gather_rows = simd[simd["path"] == "simd_gather"].reset_index(drop=True)

    wide_fit = fit_group(wide_rows, simd_wide_design, alpha=1.0)
    gather_fit = fit_group(gather_rows, simd_gather_design, alpha=1.0)

    simt = pd.read_csv(SIMT_DATASET)
    simt = simt[simt["valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    simt["target"] = simt["target_ns"].astype(float)
    simt_rows = simt.reset_index(drop=True)
    simt_fit = fit_group(simt_rows, simt_simple_design, alpha=1.5)

    payload = {
        "model_version": "strided_load_v7_simple_nonneg",
        "description": "No-log, no-power, non-negative additive strided-load model.",
        "simd_target": "ns",
        "simt_target": "ns/iteration",
        "alignment": "aligned base only; misalignment deferred",
        "data": {
            "simd": "results/model_v2/dataset.csv (simd rows)",
            "simt": "results/model_v3_numwarps/dataset.csv",
        },
        "simd_wide": model_payload("simd_wide", wide_fit),
        "simd_gather": model_payload("simd_gather", gather_fit),
        "simt": model_payload("simt", simt_fit, {
            "term_note": (
                "dupL = dup*L; min(L,64) is the line-count term before "
                "saturation; max(0, W*L-3072) is the warp-line work above the "
                "in-flight saturation threshold; dup*cross is underfilled "
                "replication on page-crossing loads."
            ),
            "derived_features": {
                "dup": "max(0, 1/E - 1), E = block/(32*num_warps)",
                "L": "aligned distinct 128B lines",
                "W": "num_warps",
                "cross": "1 if (block-1)*stride*4 >= 4096 else 0",
            },
        }),
    }
    (out_dir / "model_v7_simple.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    metrics_rows = []
    for group, fit in (("simd_wide", wide_fit), ("simd_gather", gather_fit),
                       ("simt", simt_fit)):
        row = {"group": group}
        row.update(fit["metrics"])
        metrics_rows.append(row)
    with (out_dir / "metrics_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8")

    all_rows = []
    for group, fit, extra_fields in (
            ("simd_wide", wide_fit, ()),
            ("simd_gather", gather_fit, ()),
            ("simt", simt_fit, ("num_warps",))):
        rows = list(fit["rows"].to_dict("records"))
        write_error_csv(out_dir / f"errors_{group}.csv", rows,
                        fit["pred"], extra_fields)
        for i, r in enumerate(rows):
            err = fit["pred"][i] / r["target"] - 1.0
            rec = {
                "group": group,
                "index": i,
                "block": int(r["block"]),
                "stride": int(r["stride"]),
                "target_ns": f"{r['target']:.6f}",
                "pred_ns": f"{fit['pred'][i]:.6f}",
                "error_pct": f"{err * 100.0:.6f}",
                "abs_error_pct": f"{abs(err) * 100.0:.6f}",
            }
            if "num_warps" in r:
                rec["num_warps"] = int(r["num_warps"])
            all_rows.append(rec)
    with (out_dir / "errors_all.csv").open("w", newline="") as f:
        fields = ["group", "index", "block", "stride", "num_warps",
                  "target_ns", "pred_ns", "error_pct", "abs_error_pct"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("simd_wide  ", wide_fit["metrics"])
    print("simd_gather", gather_fit["metrics"])
    print("simt       ", simt_fit["metrics"])
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
