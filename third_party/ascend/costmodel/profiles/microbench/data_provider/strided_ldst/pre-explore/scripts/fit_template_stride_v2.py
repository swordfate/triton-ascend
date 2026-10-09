#!/usr/bin/env python3
"""Fit the v2 in-sample semi-white models for the template stride paths.

Model family (same as v1, re-fitted on the full BLOCK=4..2048 / W=32 matrix):

    T = intercept + sum_j c_j * f_j,  c_j >= 0
    target = ns/iteration

Only in-sample metrics are reported.  Candidate features come from the
1024-thread scalar loop in SIMTStrideLoad.cpp / SIMTStrideStore.cpp through
``template_stride_features.py``.  Forward selection rejects candidates that are
almost collinear with an already selected feature, then Lawson-Hanson NNLS is
used for the non-negative coefficients.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_store_common import metrics as base_metrics
from fit_store_common import nnls
from template_stride_features import FEATURE_ORDER

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = HERE / "results/model_template_stride_v2/dataset.csv"
DEFAULT_OUT_DIR = HERE / "results/model_template_stride_v2"

# num_warps is fixed to 32 in this calibration; this control feature must not
# be selected into the physical model.
EXCLUDED_FEATURES = {"num_warps_minus1"}

# Candidate-feature subsets.
#   profile - restrict the pool to the features the C++ StageCostModels template
#             formulas and the hardware profile already expose, so the fitted
#             coefficients drop into simt.stage_resources.template_strided_memory
#             with no code change.  This is the adopted model.
#   all     - re-open the full pool.  Diagnostic only: it additionally selects
#             tail_lanes_gt0 / stride_gt_line / active_warps for ~0.1-0.2 pp of
#             in-sample MAPE and would require new profile fields.
PROFILE_FEATURE_SETS = {
    "triton_stride_load": ("L", "bucket_worst_32k16", "mean_warp_lines", "tail_elems"),
    "triton_stride_store": ("L", "bucket_worst_32k16", "mean_warp_lines", "iters_per_thread"),
}


def fit_weighted(X, y, alpha):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha == 0.0:
        return nnls(X, y)
    w = np.power(np.maximum(y, 1e-9), -alpha)
    return nnls(X * np.sqrt(w)[:, None], y * np.sqrt(w))


def design(rows, feature_names, include_intercept=True):
    cols = []
    if include_intercept:
        cols.append(np.ones(len(rows), dtype=float))
    for name in feature_names:
        cols.append(np.asarray([float(r[f"f_{name}"]) for r in rows], dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(rows), 0))


def weighted_obj(X, y, alpha):
    if X.shape[1] == 0:
        return float("inf"), np.zeros(0), np.zeros_like(y)
    coef = fit_weighted(X, y, alpha)
    pred = X @ coef
    if alpha == 0.0:
        w = np.ones_like(y)
    else:
        w = np.power(np.maximum(y, 1e-9), -alpha)
    return float(np.sum(w * (pred - y) ** 2)), coef, pred


def correlation_guard(rows, selected, remaining, threshold):
    """Drop candidates almost collinear with an already selected feature."""
    if not selected:
        return remaining
    xs = np.asarray([[float(r[f"f_{name}"]) for name in remaining] for r in rows], dtype=float)
    ys = np.asarray([[float(r[f"f_{name}"]) for name in selected] for r in rows], dtype=float)
    keep = []
    for j, name in enumerate(remaining):
        col = xs[:, j]
        if np.std(col) < 1e-12:
            continue
        max_abs_corr = 0.0
        for k in range(ys.shape[1]):
            ref = ys[:, k]
            if np.std(ref) < 1e-12:
                continue
            corr = abs(float(np.corrcoef(col, ref)[0, 1]))
            max_abs_corr = max(max_abs_corr, corr)
        if max_abs_corr <= threshold:
            keep.append(name)
        elif name in selected:
            keep.append(name)
    return keep


def forward_select(rows, alpha, max_terms, improvement_tol, corr_threshold,
                   candidates=None):
    y = np.asarray([float(r["target_ns"]) for r in rows], dtype=float)
    selected = []
    X = design(rows, selected, include_intercept=True)
    obj, coef, pred = weighted_obj(X, y, alpha)
    history = []
    pool = FEATURE_ORDER if candidates is None else list(candidates)
    remaining = [name for name in pool
                 if name != "intercept" and name not in EXCLUDED_FEATURES]
    while remaining and len(selected) < max_terms:
        candidates = correlation_guard(rows, selected, remaining, corr_threshold)
        if not candidates:
            break
        best = None
        for name in candidates:
            cand = selected + [name]
            X_cand = design(rows, cand, include_intercept=True)
            cand_obj, cand_coef, cand_pred = weighted_obj(X_cand, y, alpha)
            if best is None or cand_obj < best[1]:
                best = (name, cand_obj, cand_coef, cand_pred)
        improvement = (obj - best[1]) / max(obj, 1e-12)
        history.append({
            "candidate": best[0],
            "objective": best[1],
            "relative_improvement": improvement,
            "n_candidates": len(candidates),
        })
        if improvement < improvement_tol:
            break
        selected.append(best[0])
        remaining.remove(best[0])
        obj = best[1]
    X = design(rows, selected, include_intercept=True)
    _, coef, pred = weighted_obj(X, y, alpha)
    return selected, coef, pred, history


def fit_path(rows, path, alpha, max_terms, improvement_tol, corr_threshold,
             candidates=None):
    valid = [
        r for r in rows
        if r["path"] == path and int(r["valid"]) == 1 and int(r["correctness_ok"]) == 1
    ]
    if not valid:
        raise SystemExit(f"no valid rows for path={path}")
    selected, coef, _, history = forward_select(
        valid, alpha, max_terms, improvement_tol, corr_threshold, candidates
    )
    # Prune zero coefficients that NNLS may leave behind.
    keep = [i for i, c in enumerate(coef[1:]) if c > 1e-9]
    selected = [selected[i] for i in keep]
    coef = np.asarray([coef[0]] + [coef[i + 1] for i in keep], dtype=float)
    X = design(valid, selected, include_intercept=True)
    pred = X @ coef
    y = np.asarray([r["target_ns"] for r in valid], dtype=float)
    metrics = base_metrics(y, pred)
    metrics["tag"] = "in_sample"
    model = {
        "model_version": "template_stride_v2",
        "path": path,
        "target": "ns/iteration",
        "target_definition": (
            "torch.npu.Event slope over a 2000/8000-iteration rotate loop; "
            "minimum Event per point across reps, then minimum valid/correct "
            "target across pass1/pass2 for the same (path, block, stride, num_warps)"
        ),
        "compile_mode": "simd_simt_template",
        "ir_evidence": "results/model_template_stride_v2/ir_evidence/",
        "form": "T = intercept + sum(c_i * feature_i), c_i >= 0",
        "fit_method": (
            "weighted Lawson-Hanson NNLS + forward selection with a "
            f"{corr_threshold:.2f} absolute-correlation guard and a "
            f"{improvement_tol} relative-improvement floor (in-sample only)"
        ),
        "fit_weights_alpha": float(alpha),
        "fit_max_terms": int(max_terms),
        "fit_improvement_tol": float(improvement_tol),
        "fit_corr_threshold": float(corr_threshold),
        "feature_names": list(selected),
        "intercept": float(coef[0]),
        "coefficients": [float(v) for v in coef[1:]],
        "n_rows": len(valid),
        "training_ranges": {
            "block": [int(min(r["block"] for r in valid)), int(max(r["block"] for r in valid))],
            "stride": [int(min(r["stride"] for r in valid)), int(max(r["stride"] for r in valid))],
            "num_warps": sorted({int(r["num_warps"]) for r in valid}),
            "n_blocks": len({int(r["block"]) for r in valid}),
            "n_strides": len({int(r["stride"]) for r in valid}),
        },
        "selection_history": history,
        "metrics": metrics,
        "coefficients_nonnegative": bool(np.all(np.asarray(coef) >= -1e-12)),
    }
    return model, valid, selected, coef


def write_errors(path, rows, selected, coef):
    fields = ["path", "block", "stride", "num_warps", "target_ns", "pred_ns",
              "relative_error", "abs_pct_error", "split"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            X = design([r], selected, include_intercept=True)
            pred = float((X @ coef)[0])
            rel = pred / float(r["target_ns"]) - 1.0
            writer.writerow({
                "path": r["path"], "block": int(r["block"]),
                "stride": int(r["stride"]), "num_warps": int(r["num_warps"]),
                "target_ns": float(r["target_ns"]), "pred_ns": pred,
                "relative_error": rel, "abs_pct_error": abs(rel) * 100.0,
                "split": "in_sample",
            })


def load_dataset(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["block"] = int(r["block"])
        r["stride"] = int(r["stride"])
        r["num_warps"] = int(r["num_warps"])
        r["valid"] = int(r["valid"])
        r["correctness_ok"] = int(r["correctness_ok"])
        r["target_ns"] = float(r["target_ns"])
        for name in FEATURE_ORDER:
            r[f"f_{name}"] = float(r[f"f_{name}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--max-terms", type=int, default=6)
    ap.add_argument("--improvement-tol", type=float, default=0.002)
    ap.add_argument("--corr-threshold", type=float, default=0.98)
    ap.add_argument("--feature-subset", choices=("profile", "all"),
                    default="profile",
                    help="profile = only the features the hardware profile "
                         "already exposes (default, adopted model); "
                         "all = full candidate pool (diagnostic)")
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    all_error_rows = []
    for path in ("triton_stride_load", "triton_stride_store"):
        # Select alpha by in-sample MAPE; ties naturally keep the first alpha.
        best = None
        for alpha in args.alphas:
            candidates = (PROFILE_FEATURE_SETS[path]
                          if args.feature_subset == "profile" else None)
            model, valid, selected, coef = fit_path(
                rows, path, alpha, args.max_terms, args.improvement_tol,
                args.corr_threshold, candidates
            )
            mape = model["metrics"]["mape_pct"]
            if best is None or mape < best[0]:
                best = (mape, model, valid, selected, coef)
        _, model, valid, selected, coef = best
        model["feature_subset"] = args.feature_subset
        stem = "load" if path == "triton_stride_load" else "store"
        (args.out_dir / f"model_template_stride_{stem}_v2.json").write_text(
            json.dumps(model, indent=2), encoding="utf-8"
        )
        write_errors(args.out_dir / f"errors_{stem}.csv", valid, selected, coef)
        summary[path] = {
            "model_file": f"model_template_stride_{stem}_v2.json",
            "n_rows": model["n_rows"],
            "feature_names": selected,
            "intercept": model["intercept"],
            "coefficients": model["coefficients"],
            "alpha": model["fit_weights_alpha"],
            "in_sample_mape_pct": model["metrics"]["mape_pct"],
            "in_sample_p50_pct": model["metrics"]["p50_pct"],
            "in_sample_p90_pct": model["metrics"]["p90_pct"],
            "in_sample_p95_pct": model["metrics"]["p95_pct"],
            "in_sample_max_pct": model["metrics"]["max_pct"],
            "in_sample_bias_pct": model["metrics"]["bias_pct"],
            "in_sample_rmse_pct": model["metrics"]["rmse_pct"],
            "in_sample_rmse_ns": model["metrics"]["rmse_ns"],
        }
        with (args.out_dir / f"errors_{stem}.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                all_error_rows.append(row)

    (args.out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "metrics_summary.csv").open("w", newline="") as f:
        fields = ["path", "n_rows", "feature_names", "intercept", "coefficients",
                  "alpha", "in_sample_mape_pct", "in_sample_p50_pct",
                  "in_sample_p90_pct", "in_sample_p95_pct",
                  "in_sample_max_pct", "in_sample_bias_pct",
                  "in_sample_rmse_pct", "in_sample_rmse_ns"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path, s in summary.items():
            writer.writerow({
                "path": path, "n_rows": s["n_rows"],
                "feature_names": ";".join(s["feature_names"]),
                "intercept": s["intercept"],
                "coefficients": ";".join(str(v) for v in s["coefficients"]),
                "alpha": s["alpha"],
                "in_sample_mape_pct": s["in_sample_mape_pct"],
                "in_sample_p50_pct": s["in_sample_p50_pct"],
                "in_sample_p90_pct": s["in_sample_p90_pct"],
                "in_sample_p95_pct": s["in_sample_p95_pct"],
                "in_sample_max_pct": s["in_sample_max_pct"],
                "in_sample_bias_pct": s["in_sample_bias_pct"],
                "in_sample_rmse_pct": s["in_sample_rmse_pct"],
                "in_sample_rmse_ns": s["in_sample_rmse_ns"],
            })
    with (args.out_dir / "errors_all.csv").open("w", newline="") as f:
        fields = ["path", "block", "stride", "num_warps", "target_ns",
                  "pred_ns", "relative_error", "abs_pct_error", "split"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_error_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
