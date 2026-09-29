#!/usr/bin/env python3
"""Fit the raw-domain non-negative strided-store model v1.

Branches
--------
* ``simd_wide``   : stride == 1, MTE3 wide transaction
* ``simd_gather`` : stride >= 2, per-element MTE3 commands
* ``simt``        : SIMT_STG + 128B line writes

All coefficients are non-negative (Lawson-Hanson NNLS).  No target log,
power, or negative term enters the final predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_store_common import metrics, nnls, read_dataset  # noqa: E402

BRANCH_SPECS = {
    "simd_wide": {
        "features": ["block"],
        "selector": lambda r: r["mode"] == "simd" and int(r["stride"]) == 1,
    },
    "simd_gather": {
        "features": ["line_request_size", "bank_pairs", "worst_G32768"],
        "selector": lambda r: r["mode"] == "simd" and int(r["stride"]) >= 2,
    },
    "simt": {
        "features": ["K_stg", "L", "line_request_size"],
        "selector": lambda r: r["mode"] == "simt_only",
    },
}

FEATURE_NOTES = {
    "block": "logical f32 elements in the tile; wide MTE3 transaction payload",
    "line_request_size": "useful bytes carried by one 128B line (4..128B)",
    "bank_pairs": "pairs of tile elements landing in the same 2KB/8-bank bucket",
    "worst_G32768": "max tile elements in one 32KB region minus 1",
    "K_stg": "SIMT warp-level STG instruction count per program",
    "L": "distinct 128B lines touched by the strided tile",
}


def design(rows, feature_names):
    A = np.zeros((len(rows), len(feature_names) + 1), dtype=np.float64)
    A[:, 0] = 1.0
    for j, name in enumerate(feature_names, start=1):
        A[:, j] = [float(r[name]) for r in rows]
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    dataset = args.dataset or root / "results/model_store_v1/dataset.csv"
    outdir = args.outdir or root / "results/model_store_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_dataset(dataset)
    branches = {}
    preds_by_index = [None] * len(rows)
    branch_by_index = [None] * len(rows)

    for branch, spec in BRANCH_SPECS.items():
        selected = [r for r in rows if spec["selector"](r)]
        names = spec["features"]
        A = design(selected, names)
        b = np.array([float(r["target_ns"]) for r in selected], dtype=np.float64)
        coef = nnls(A, b)
        pred = A @ coef
        branches[branch] = {
            "n": len(selected),
            "features": names,
            "intercept_ns": float(coef[0]),
            "coefficients_ns": [float(v) for v in coef[1:]],
            "metrics": metrics(b, pred),
            "feature_notes": {name: FEATURE_NOTES.get(name, "") for name in names},
        }
        for row, p in zip(selected, pred):
            idx = rows.index(row)
            preds_by_index[idx] = float(p)
            branch_by_index[idx] = branch

    # Write errors.
    error_fields = ["mode", "block", "stride", "num_warps", "branch",
                    "target_ns", "pred_ns", "relative_error", "abs_pct_error"]
    totals = {"target": [], "pred": []}
    group_errors = {name: {"target": [], "pred": []} for name in BRANCH_SPECS}
    with (outdir / "errors_all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=error_fields)
        writer.writeheader()
        for row, branch, pred in zip(rows, branch_by_index, preds_by_index):
            target = float(row["target_ns"])
            rel = pred / target - 1.0
            writer.writerow({
                "mode": row["mode"], "block": row["block"], "stride": row["stride"],
                "num_warps": row["num_warps"], "branch": branch,
                "target_ns": target, "pred_ns": pred,
                "relative_error": rel, "abs_pct_error": abs(rel) * 100.0,
            })
            totals["target"].append(target)
            totals["pred"].append(pred)
            group_errors[branch]["target"].append(target)
            group_errors[branch]["pred"].append(pred)

    for branch, bucket in group_errors.items():
        with (outdir / f"errors_{branch}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=error_fields)
            writer.writeheader()
            for row, br, pred in zip(rows, branch_by_index, preds_by_index):
                if br != branch:
                    continue
                target = float(row["target_ns"])
                rel = pred / target - 1.0
                writer.writerow({
                    "mode": row["mode"], "block": row["block"], "stride": row["stride"],
                    "num_warps": row["num_warps"], "branch": branch,
                    "target_ns": target, "pred_ns": pred,
                    "relative_error": rel, "abs_pct_error": abs(rel) * 100.0,
                })

    overall = metrics(totals["target"], totals["pred"])
    metrics_summary = {
        "branch_order": list(BRANCH_SPECS.keys()),
        "branches": {name: data["metrics"] for name, data in branches.items()},
        "overall": overall,
        "target_definition": ("min observed Event time per iteration in a "
                              "2000/8000-iteration rotate loop, ns/iteration"),
        "cycle_domain": "board Event ns",
    }
    model = {
        "model_name": "strided_store_v1",
        "target_kind": "store_rotate_min_observed_ns_per_iter",
        "target_definition": metrics_summary["target_definition"],
        "dataset": str(dataset),
        "feature_notes": FEATURE_NOTES,
        "branches": branches,
        "metrics": metrics_summary,
    }
    (outdir / "model_store_v1.json").write_text(
        json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True), encoding="utf-8")

    with (outdir / "metrics_summary.csv").open("w", newline="") as f:
        fieldnames = ["branch", "n", "mape_pct", "p50_pct", "p90_pct",
                      "p95_pct", "max_pct", "bias_pct", "rmse_pct", "rmse_ns"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in list(BRANCH_SPECS.keys()) + ["overall"]:
            data = branches[name]["metrics"] if name in branches else overall
            writer.writerow({"branch": name, **{k: data[k] for k in fieldnames[1:]}})
    print(f"wrote {outdir}/model_store_v1.json")
    for name, data in branches.items():
        m = data["metrics"]
        print(f"{name}: n={data['n']} MAPE={m['mape_pct']:.2f}% "
              f"p50={m['p50_pct']:.2f}% p90={m['p90_pct']:.2f}% "
              f"max={m['max_pct']:.2f}%")
    print(f"overall: n={overall['n']} MAPE={overall['mape_pct']:.2f}% "
          f"p50={overall['p50_pct']:.2f}% p90={overall['p90_pct']:.2f}% "
          f"max={overall['max_pct']:.2f}%")


if __name__ == "__main__":
    main()
