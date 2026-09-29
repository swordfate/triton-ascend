#!/usr/bin/env python3
"""Shared NNLS / metrics / feature helpers for the store model."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ELEM_BYTES = 4
WARP = 32


def nnls(A, b, max_iter=None, tol=1e-12):
    """Lawson-Hanson non-negative least squares, works with float64 numpy."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n
    P = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=np.float64)
    w = A.T @ (b - A @ x)
    it = 0
    while (not P.all()) and np.any(w[~P] > tol) and it < max_iter:
        candidates = np.where(~P, w, -np.inf)
        j = int(np.argmax(candidates))
        P[j] = True
        while True:
            Ap = A[:, P]
            s = np.zeros(n, dtype=np.float64)
            try:
                s[P] = np.linalg.lstsq(Ap, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                s[P] = 0.0
            if np.all(s[P] > tol):
                break
            ratios = []
            for idx in np.where(P)[0]:
                if s[idx] <= tol:
                    denom = x[idx] - s[idx]
                    if denom > 0:
                        ratios.append(x[idx] / denom)
            if not ratios:
                s[P] = np.maximum(s[P], 0.0)
                break
            alpha = min(ratios)
            x = x + alpha * (s - x)
            P = x > tol
            if not np.any(P):
                x = np.zeros(n, dtype=np.float64)
                break
        x = np.where(P, np.maximum(s, 0.0), 0.0)
        w = A.T @ (b - A @ x)
        it += 1
    return np.maximum(x, 0.0)


def metrics(target, pred):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    rel = pred / target - 1.0
    abs_rel = np.abs(rel)
    return {
        "n": int(len(target)),
        "mape_pct": float(np.mean(abs_rel) * 100.0),
        "p50_pct": float(np.percentile(abs_rel, 50) * 100.0),
        "p90_pct": float(np.percentile(abs_rel, 90) * 100.0),
        "p95_pct": float(np.percentile(abs_rel, 95) * 100.0),
        "max_pct": float(np.max(abs_rel) * 100.0),
        "bias_pct": float(np.mean(rel) * 100.0),
        "rmse_pct": float(math.sqrt(np.mean((pred - target) ** 2)) / np.mean(target) * 100.0),
        "rmse_ns": float(math.sqrt(np.mean((pred - target) ** 2))),
    }


def read_dataset(path: Path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in list(row.keys()):
            if key in {"mode", "target_kind"}:
                continue
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass
        row["block"] = int(row["block"])
        row["stride"] = int(row["stride"])
        row["num_warps"] = int(row["num_warps"])
        row["valid"] = int(row["valid"])
    return rows


def write_errors(path: Path, rows, branches, targets, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "block", "stride", "num_warps", "branch",
              "target_ns", "pred_ns", "relative_error", "abs_pct_error"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, branch, target, pred in zip(rows, branches, targets, preds):
            rel = pred / target - 1.0
            writer.writerow({
                "mode": row["mode"], "block": row["block"], "stride": row["stride"],
                "num_warps": row["num_warps"], "branch": branch,
                "target_ns": target, "pred_ns": pred,
                "relative_error": rel, "abs_pct_error": abs(rel) * 100.0,
            })
