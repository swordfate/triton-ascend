#!/usr/bin/env python3
"""Fit the v1 semi-white models for the 1D rank1 template load/store paths.

Model family:
    T = intercept + sum_j c_j * f_j,  c_j >= 0
in the raw ``ns/iteration`` domain.

The candidate features come from ``template_stride_features.py``, which mirrors
the 1024-thread scalar loop in ``SIMTStrideLoad.cpp`` / ``SIMTStrideStore.cpp``.
We use forward selection with weighted non-negative least squares instead of
dumping all correlated features into one NNLS, then report an out-of-sample
split by BLOCK.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fit_store_common import metrics as base_metrics
from fit_store_common import nnls
from template_stride_features import FEATURE_ORDER

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = HERE / "results/model_template_stride_v1/dataset.csv"
OUT_DIR = HERE / "results/model_template_stride_v1"


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
        return float("inf")
    coef = fit_weighted(X, y, alpha)
    pred = X @ coef
    w = np.ones_like(y) if alpha == 0.0 else np.power(np.maximum(y, 1e-9), -alpha)
    return float(np.sum(w * (pred - y) ** 2)), coef, pred


def forward_select(rows, alpha, max_terms, improvement_tol):
    y = np.asarray([float(r["target_ns"]) for r in rows], dtype=float)
    selected = []
    X_base = design(rows, selected, include_intercept=True)
    obj, _, _ = weighted_obj(X_base, y, alpha)
    history = []
    remaining = [name for name in FEATURE_ORDER if name != "intercept"]
    while remaining and len(selected) < max_terms:
        best = None
        for name in remaining:
            cand = selected + [name]
            X = design(rows, cand, include_intercept=True)
            cand_obj, coef, pred = weighted_obj(X, y, alpha)
            if best is None or cand_obj < best[1]:
                best = (name, cand_obj, coef, pred)
        improvement = (obj - best[1]) / max(obj, 1e-12)
        history.append({
            "candidate": best[0],
            "objective": best[1],
            "relative_improvement": improvement,
        })
        if improvement < improvement_tol:
            break
        selected.append(best[0])
        remaining.remove(best[0])
        obj = best[1]
    X = design(rows, selected, include_intercept=True)
    _, coef, pred = weighted_obj(X, y, alpha)
    return selected, coef, pred, obj, history


def metric_row(tag, target, pred):
    m = base_metrics(target, pred)
    m["tag"] = tag
    return m


def split_rows(rows, holdout_blocks):
    holdout = [r for r in rows if int(r["block"]) in holdout_blocks]
    train = [r for r in rows if int(r["block"]) not in holdout_blocks]
    return train, holdout


def leave_group_out_cv(rows, selected, alpha, group_key, held_values):
    """Fit coefficients on all groups except one, predict the held group."""
    targets, preds = [], []
    for value in held_values:
        train = [r for r in rows if r[group_key] != value]
        test = [r for r in rows if r[group_key] == value]
        if not train or not test:
            continue
        X_train = design(train, selected, include_intercept=True)
        y_train = np.asarray([r["target_ns"] for r in train], dtype=float)
        coef = fit_weighted(X_train, y_train, alpha)
        X_test = design(test, selected, include_intercept=True)
        preds.extend((X_test @ coef).tolist())
        targets.extend([r["target_ns"] for r in test])
    if not targets:
        return {}
    return metric_row(f"cv_by_{group_key}", np.asarray(targets), np.asarray(preds))


def fit_path(rows, path, holdout_blocks, alphas, max_terms, improvement_tol,
             selection_scope="all"):
    valid = [
        r for r in rows
        if r["path"] == path and int(r["valid"]) == 1 and int(r["correctness_ok"]) == 1
    ]
    if not valid:
        raise SystemExit(f"no valid rows for path={path}")
    if selection_scope == "all":
        train, holdout = valid, []
    else:
        train, holdout = split_rows(valid, holdout_blocks)
        if not holdout:
            raise SystemExit(f"holdout blocks {holdout_blocks} not present for {path}")
    # choose alpha by forward-selection objective on the selection split
    best_alpha = None
    best_sel = None
    for alpha in alphas:
        selected, coef, pred, obj, history = forward_select(
            train, alpha, max_terms, improvement_tol
        )
        if best_alpha is None or obj < best_alpha[0]:
            best_alpha = (obj, alpha)
            best_sel = (selected, coef, pred, history)
    selected, coef, train_pred, history = best_sel
    alpha = best_alpha[1]
    # Drop features that ended with a zero coefficient.  Forward selection can
    # add a collinear candidate whose final NNLS weight is zero; it has no
    # inference effect and should not appear in the integration contract.
    keep = [i for i, c in enumerate(coef[1:]) if c > 1e-9]
    selected = [selected[i] for i in keep]
    coef = np.asarray([coef[0]] + [coef[i + 1] for i in keep], dtype=float)
    X_train = design(train, selected, include_intercept=True)
    train_pred = X_train @ coef
    y_train = np.asarray([r["target_ns"] for r in train], dtype=float)

    stride_cv_values = [5, 11, 21, 48, 129, 255]
    cv_by_block = leave_group_out_cv(valid, selected, alpha, "block", holdout_blocks)
    cv_by_stride = leave_group_out_cv(valid, selected, alpha, "stride", stride_cv_values)
    # In all-data selection mode there is no separate holdout fit; report the
    # block out-of-sample CV as the primary holdout metric and keep stride CV
    # as a second view.
    holdout_metrics = cv_by_block
    n_holdout = int(cv_by_block.get("n", 0)) if cv_by_block else 0
    model = {
        "model_version": "template_stride_v1",
        "path": path,
        "target": "ns/iteration",
        "target_definition": "torch.npu.Event slope over a 2000/8000-iteration rotate loop; min event per point across reps/passes",
        "compile_mode": "simd_simt_template",
        "ir_evidence": "results/model_template_stride_v1/ir_evidence/",
        "form": "T = intercept + sum(c_i * feature_i), c_i >= 0",
        "fit_method": "weighted non-negative least squares + forward selection",
        "selection_scope": selection_scope,
        "fit_weights_alpha": float(alpha),
        "feature_names": list(selected),
        "intercept": float(coef[0]),
        "coefficients": [float(v) for v in coef[1:]],
        "n_rows": len(valid),
        "n_train": len(train),
        "n_holdout": n_holdout,
        "training_ranges": {
            "block": [int(min(r["block"] for r in valid)), int(max(r["block"] for r in valid))],
            "stride": [int(min(r["stride"] for r in valid)), int(max(r["stride"] for r in valid))],
            "num_warps": sorted({int(r["num_warps"]) for r in valid}),
            "holdout_blocks": sorted(int(b) for b in holdout_blocks),
        },
        "selection_history": history,
        "metrics": metric_row("in_sample", y_train, train_pred),
        "holdout_metrics": holdout_metrics,
        "cv_by_block": cv_by_block,
        "cv_by_stride": cv_by_stride,
        "cv_by_stride_values": stride_cv_values,
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
            x = design([r], selected, include_intercept=True)
            pred = float((x @ coef)[0])
            rel = pred / float(r["target_ns"]) - 1.0
            writer.writerow({
                "path": r["path"], "block": int(r["block"]), "stride": int(r["stride"]),
                "num_warps": int(r["num_warps"]), "target_ns": float(r["target_ns"]),
                "pred_ns": pred, "relative_error": rel,
                "abs_pct_error": abs(rel) * 100.0,
                "split": "in_sample",
            })


def write_cv_errors(path, rows, selected, alpha, group_key, held_values):
    """Leave-one-group-out errors: refit coefficients per fold."""
    fields = ["path", "block", "stride", "num_warps", "target_ns", "pred_ns",
              "relative_error", "abs_pct_error", "split"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for value in held_values:
            train = [r for r in rows if r[group_key] != value]
            test = [r for r in rows if r[group_key] == value]
            if not train or not test:
                continue
            X_train = design(train, selected, include_intercept=True)
            y_train = np.asarray([r["target_ns"] for r in train], dtype=float)
            coef = fit_weighted(X_train, y_train, alpha)
            X_test = design(test, selected, include_intercept=True)
            preds = X_test @ coef
            for r, pred in zip(test, preds):
                rel = float(pred) / float(r["target_ns"]) - 1.0
                writer.writerow({
                    "path": r["path"], "block": int(r["block"]),
                    "stride": int(r["stride"]), "num_warps": int(r["num_warps"]),
                    "target_ns": float(r["target_ns"]), "pred_ns": float(pred),
                    "relative_error": rel, "abs_pct_error": abs(rel) * 100.0,
                    "split": f"cv_{group_key}_{value}",
                })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--holdout-blocks", nargs="+", type=int, default=[64, 512, 2048])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.5])
    ap.add_argument("--max-terms", type=int, default=6,
                    help="deprecated compatibility knob; ignored when --max-terms-candidates is used")
    ap.add_argument("--max-terms-candidates", nargs="+", type=int,
                    default=[4, 5, 6, 8])
    ap.add_argument("--improvement-tol", type=float, default=0.005)
    ap.add_argument("--selection-scope", choices=["all", "train"], default="all")
    args = ap.parse_args()

    import csv as _csv
    with args.dataset.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    for r in rows:
        r["block"] = int(r["block"])
        r["stride"] = int(r["stride"])
        r["num_warps"] = int(r["num_warps"])
        r["valid"] = int(r["valid"])
        r["correctness_ok"] = int(r["correctness_ok"])
        r["target_ns"] = float(r["target_ns"])
        for name in FEATURE_ORDER:
            r[f"f_{name}"] = float(r[f"f_{name}"])
        r["_split"] = "holdout" if int(r["block"]) in args.holdout_blocks else "train"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for path in ("triton_stride_load", "triton_stride_store"):
        candidates = []
        for max_terms in args.max_terms_candidates:
            model, valid, selected, coef = fit_path(
                rows, path, args.holdout_blocks, args.alphas, max_terms,
                args.improvement_tol, args.selection_scope,
            )
            cv_block = model.get("cv_by_block", {}).get("mape_pct", float("inf"))
            cv_stride = model.get("cv_by_stride", {}).get("mape_pct", float("inf"))
            # Primary selection: block CV MAPE; break near-ties by stride CV.
            score = cv_block + 0.25 * cv_stride
            candidates.append((score, model, valid, selected, coef, max_terms))
        _, model, valid, selected, coef, chosen_terms = min(
            candidates, key=lambda item: item[0]
        )
        model["selected_max_terms"] = chosen_terms
        model["max_terms_candidates"] = list(args.max_terms_candidates)
        stem = "load" if path == "triton_stride_load" else "store"
        (args.out_dir / f"model_template_stride_{stem}_v1.json").write_text(
            json.dumps(model, indent=2), encoding="utf-8"
        )
        write_errors(args.out_dir / f"errors_{stem}.csv", valid, selected, coef)
        write_cv_errors(args.out_dir / f"errors_cv_block_{stem}.csv", valid,
                        selected, model["fit_weights_alpha"], "block",
                        model["training_ranges"]["holdout_blocks"])
        write_cv_errors(args.out_dir / f"errors_cv_stride_{stem}.csv", valid,
                        selected, model["fit_weights_alpha"], "stride",
                        model["cv_by_stride_values"])
        summary[path] = {
            "model_file": f"model_template_stride_{stem}_v1.json",
            "n_rows": model["n_rows"],
            "n_train": model["n_train"],
            "n_holdout": model["n_holdout"],
            "feature_names": selected,
            "intercept": model["intercept"],
            "coefficients": model["coefficients"],
            "alpha": model["fit_weights_alpha"],
            "in_sample_mape_pct": model["metrics"]["mape_pct"],
            "holdout_mape_pct": model["holdout_metrics"]["mape_pct"],
            "holdout_p50_pct": model["holdout_metrics"]["p50_pct"],
            "holdout_max_pct": model["holdout_metrics"]["max_pct"],
            "cv_by_block_mape_pct": model.get("cv_by_block", {}).get("mape_pct"),
            "cv_by_block_max_pct": model.get("cv_by_block", {}).get("max_pct"),
            "cv_by_stride_mape_pct": model.get("cv_by_stride", {}).get("mape_pct"),
            "cv_by_stride_max_pct": model.get("cv_by_stride", {}).get("max_pct"),
        }
    (args.out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "metrics_summary.csv").open("w", newline="") as f:
        fields = ["path", "n_rows", "n_train", "n_holdout", "in_sample_mape_pct",
                  "holdout_mape_pct", "holdout_p50_pct", "holdout_max_pct",
                  "cv_by_block_mape_pct", "cv_by_block_max_pct",
                  "cv_by_stride_mape_pct", "cv_by_stride_max_pct",
                  "intercept", "feature_names", "coefficients"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path, s in summary.items():
            writer.writerow({
                "path": path, "n_rows": s["n_rows"], "n_train": s["n_train"],
                "n_holdout": s["n_holdout"],
                "in_sample_mape_pct": s["in_sample_mape_pct"],
                "holdout_mape_pct": s["holdout_mape_pct"],
                "holdout_p50_pct": s["holdout_p50_pct"],
                "holdout_max_pct": s["holdout_max_pct"],
                "cv_by_block_mape_pct": s["cv_by_block_mape_pct"],
                "cv_by_block_max_pct": s["cv_by_block_max_pct"],
                "cv_by_stride_mape_pct": s["cv_by_stride_mape_pct"],
                "cv_by_stride_max_pct": s["cv_by_stride_max_pct"],
                "intercept": s["intercept"],
                "feature_names": ";".join(s["feature_names"]),
                "coefficients": ";".join(str(v) for v in s["coefficients"]),
            })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
