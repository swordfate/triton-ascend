#!/usr/bin/env python3
"""Final non-negative additive strided-load models (raw target domain).

This script replaces the old log-domain v2/v5 reference models with three
explicit additive models:

  simd_wide   : target ns,       all-data weighted non-negative LS
  simd_gather : target ns,       all-data weighted non-negative LS
  simt        : target ns/iter,  all-data weighted non-negative LS

Model family:  T = c0 + sum_i c_i * f_i,  c_i >= 0

No log target transform is used.  The weights are target^-alpha so that
relative error receives more attention while the fitted equation remains a
plain additive formula in the original target units.

Outputs:
  results/model_v6_nonneg/model_v6_nonneg.json
  results/model_v6_nonneg/metrics_summary.json / .csv
  results/model_v6_nonneg/errors_all.csv
  results/model_v6_nonneg/errors_simd_wide.csv
  results/model_v6_nonneg/errors_simd_gather.csv
  results/model_v6_nonneg/errors_simt.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent  # pre-explore root (scripts/ lives directly under it)
SIMD_DATASET = HERE / "results/model_v2/dataset.csv"
SIMT_DATASET = HERE / "results/model_v3_numwarps/dataset.csv"
OUT_DIR = HERE / "results/model_v6_nonneg"

LINE_BYTES = 128
PAGE_BYTES = 4096
ELEM_BYTES = 4


# ---------------------------------------------------------------------------
# Non-negative least squares (Lawson-Hanson active set)
# ---------------------------------------------------------------------------

def nonneg_lstsq(A: np.ndarray, b: np.ndarray, tol: float = 1e-10,
                 max_iter: int = 10000):
    """Solve min_x ||A x - b||_2 subject to x >= 0."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    x = np.zeros(n, dtype=float)
    passive = []          # indices with x_i > 0
    active = list(range(n))
    w = A.T @ (b - A @ x)
    it = 0
    while active and w[active].max() > tol and it < max_iter:
        it += 1
        j = int(active[int(np.argmax(w[active]))])
        passive.append(j)
        active.remove(j)
        while True:
            Ap = A[:, passive]
            xp, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if np.all(xp > tol):
                x[passive] = xp
                break
            # Step to the boundary: x + alpha * (xp_full - x)
            alpha = np.inf
            for idx, p in enumerate(passive):
                if xp[idx] <= tol and x[p] - xp[idx] > 1e-15:
                    alpha = min(alpha, x[p] / (x[p] - xp[idx]))
            if not np.isfinite(alpha):
                x[passive] = np.maximum(xp, 0.0)
                break
            xp_full = np.zeros(n, dtype=float)
            xp_full[passive] = xp
            x = x + alpha * (xp_full - x)
            keep = []
            for p in passive:
                if x[p] > tol:
                    keep.append(p)
                else:
                    x[p] = 0.0
                    active.append(p)
            passive = sorted(keep)
            if not passive:
                break
        w = A.T @ (b - A @ x)
    # polish on final passive set
    if passive:
        idx = np.asarray(passive, dtype=int)
        x[idx] = np.maximum(np.linalg.lstsq(A[:, idx], b, rcond=None)[0], 0.0)
    return x


def fit_weighted(X: np.ndarray, y: np.ndarray, alpha: float):
    """Non-negative LS with weights y^-alpha (alpha=0 => ordinary NNLS)."""
    y = np.asarray(y, dtype=float)
    if alpha == 0.0:
        return nonneg_lstsq(X, y)
    w = np.power(y, -alpha)
    return nonneg_lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w))


# ---------------------------------------------------------------------------
# Metrics / reporting
# ---------------------------------------------------------------------------

def error_metrics(target, pred):
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rel = pred / target - 1.0
    ae = np.abs(rel) * 100.0
    return OrderedDict([
        ("n", int(len(ae))),
        ("mape_pct", float(ae.mean())),
        ("p50_pct", float(np.percentile(ae, 50))),
        ("p90_pct", float(np.percentile(ae, 90))),
        ("p95_pct", float(np.percentile(ae, 95))),
        ("max_pct", float(ae.max())),
        ("bias_pct", float(rel.mean() * 100.0)),
        ("rmse_pct", float(np.sqrt(np.mean(rel ** 2)) * 100.0)),
        ("within_5pct", float((ae <= 5.0).mean() * 100.0)),
        ("within_10pct", float((ae <= 10.0).mean() * 100.0)),
        ("within_20pct", float((ae <= 20.0).mean() * 100.0)),
        ("within_30pct", float((ae <= 30.0).mean() * 100.0)),
        ("within_50pct", float((ae <= 50.0).mean() * 100.0)),
    ])


def write_error_csv(path: Path, rows, pred, extra_fields=()):
    with path.open("w", newline="") as f:
        fields = ["index", "block", "stride", "target_ns", "pred_ns",
                  "error_pct", "abs_error_pct"] + list(extra_fields)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            err = pred[i] / r["target"] - 1.0
            row = {
                "index": i,
                "block": int(r["block"]),
                "stride": int(r["stride"]),
                "target_ns": f"{r['target']:.6f}",
                "pred_ns": f"{pred[i]:.6f}",
                "error_pct": f"{err * 100.0:.6f}",
                "abs_error_pct": f"{abs(err) * 100.0:.6f}",
            }
            for field in extra_fields:
                row[field] = r.get(field)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# SIMD model features
# ---------------------------------------------------------------------------

def gather_bucket_features(block: int, stride: int, g: int = 2048, banks: int = 8):
    """Feature helpers for one SIMD gather tile.

    The SIMD gather path issues one 4B MTE command per element.  The commands
    map into `banks` buckets by ((addr // g) % banks); `pairs` counts unordered
    pairs that share a bucket, and `worst` is (largest bucket - 1).
    """
    counts = [0] * banks
    for i in range(int(block)):
        counts[((i * int(stride) * ELEM_BYTES) // g) % banks] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1
    return float(pairs), float(worst)


def simd_wide_design(rows):
    block = rows["block"].to_numpy(dtype=float)
    stride = rows["stride"].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(rows)),
        stride - 1.0,
        (block > 64.0).astype(float),
        np.maximum(0.0, block - 64.0),
    ])
    names = ["1", "stride_minus_1", "large_tile_gt64", "tail_elems_gt64"]
    return X, names


def simd_gather_design(rows):
    block = rows["block"].to_numpy(dtype=int)
    stride = rows["stride"].to_numpy(dtype=int)
    pairs = np.empty(len(rows), dtype=float)
    worst = np.empty(len(rows), dtype=float)
    for i, (b, s) in enumerate(zip(block, stride)):
        pairs[i], worst[i] = gather_bucket_features(int(b), int(s))
    cross_count = np.floor(((block - 1) * stride * ELEM_BYTES) / PAGE_BYTES)
    X = np.column_stack([
        np.ones(len(rows)),
        block.astype(float),
        pairs,
        worst,
        cross_count,
    ])
    names = ["1", "n_cmd_block", "bucket_pairs_g2048_b8",
             "bucket_worst_g2048_b8", "page_cross_count"]
    return X, names


# ---------------------------------------------------------------------------
# SIMT model features
# ---------------------------------------------------------------------------

def aligned_lines(block: int, stride: int):
    stride_bytes = stride * ELEM_BYTES
    if stride_bytes >= LINE_BYTES:
        return float(block)
    return float(((block - 1) * stride_bytes) // LINE_BYTES + 1)


def simt_design(rows):
    block = rows["block"].to_numpy(dtype=float)
    stride = rows["stride"].to_numpy(dtype=float)
    num_warps = rows["num_warps"].to_numpy(dtype=float)
    cross = rows["cross_page"].to_numpy(dtype=float)
    lines = rows["L"].to_numpy(dtype=float)

    span = (block - 1.0) * stride * ELEM_BYTES
    e = block / (32.0 * num_warps)
    dup = np.maximum(0.0, 1.0 / np.maximum(e, 1e-12) - 1.0)
    l0 = lines * (1.0 - cross)
    l1 = lines * cross
    dup_l = dup * lines

    # Natural normalizers fixed once from the data 95th percentile.  These
    # keep the units of every term explicit and avoid huge cubic values.
    s_k, s_e, s_l0, s_l1, s_dup_l = 64.0, 4.0, 32.0, 256.0, 1920.0
    s_dup = 63.0
    f_k = rows["K"].to_numpy(dtype=float) / s_k
    f_e = e / s_e
    f_l0 = l0 / s_l0
    f_l1 = l1 / s_l1
    f_dup = dup / s_dup
    f_dup_l = dup_l / s_dup_l

    X = np.column_stack([
        np.ones(len(rows)),
        cross,
        f_dup_l ** 3,
        f_l0 ** 2,
        f_k * (f_l1 ** 2),
        f_dup_l,
        f_e ** 2,
        f_e * f_dup_l,
    ])
    names = [
        "1",
        "cross",
        "dupL_norm_cube",
        "L0_norm_square",
        "K_norm_x_L1_norm_square",
        "dupL_norm",
        "E_norm_square",
        "E_norm_x_dupL_norm",
    ]
    scale_info = {
        "K": s_k,
        "E": s_e,
        "L0": s_l0,
        "L1": s_l1,
        "dup": s_dup,
        "dupL": s_dup_l,
    }
    return X, names


# ---------------------------------------------------------------------------
# Main fitting
# ---------------------------------------------------------------------------

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


def model_payload(group, fit, extra=None):
    payload = OrderedDict([
        ("group", group),
        ("n_rows", int(len(fit["rows"]))),
        ("fit_domain", "raw target"),
        ("fit_weights", f"target^(-{fit['alpha']:g})"),
        ("form", "T = c0 + sum(c_i * f_i), c_i >= 0"),
        ("feature_names", fit["names"]),
        ("coefficients", [float(v) for v in fit["theta"]]),
        ("metrics", fit["metrics"]),
        ("training_ranges", {
            "block": [int(fit["rows"]["block"].min()), int(fit["rows"]["block"].max())],
            "stride": [int(fit["rows"]["stride"].min()), int(fit["rows"]["stride"].max())],
        }),
    ])
    if extra:
        payload.update(extra)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SIMD -------------------------------------------------------------
    simd = pd.read_csv(SIMD_DATASET)
    simd = simd[simd["valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    simd["target"] = simd["target_ns"].astype(float)
    wide_rows = simd[simd["path"] == "simd_wide"].reset_index(drop=True)
    gather_rows = simd[simd["path"] == "simd_gather"].reset_index(drop=True)

    wide_fit = fit_group(wide_rows, simd_wide_design, alpha=1.0)
    gather_fit = fit_group(gather_rows, simd_gather_design, alpha=1.0)

    # --- SIMT -------------------------------------------------------------
    simt = pd.read_csv(SIMT_DATASET)
    simt = simt[simt["valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    simt["target"] = simt["target_ns"].astype(float)
    simt_rows = simt.reset_index(drop=True)
    simt_fit = fit_group(simt_rows, simt_design, alpha=1.5)

    # --- raw-unit coefficient conversion for README / JSON ---------------
    wide_raw = {
        "intercept": float(wide_fit["theta"][0]),
        "stride_minus_1": float(wide_fit["theta"][1]),
        "large_tile_gt64": float(wide_fit["theta"][2]),
        "tail_elems_gt64": float(wide_fit["theta"][3]),
    }
    gather_raw = {
        "intercept": float(gather_fit["theta"][0]),
        "n_cmd_block": float(gather_fit["theta"][1]),
        "bucket_pairs_g2048_b8": float(gather_fit["theta"][2]),
        "bucket_worst_g2048_b8": float(gather_fit["theta"][3]),
        "page_cross_count": float(gather_fit["theta"][4]),
    }
    # Convert normalized SIMT coefficients back to natural units; the exact
    # formula in normal form is kept as the canonical predictor form.
    s = {"K": 64.0, "E": 4.0, "L0": 32.0, "L1": 256.0,
         "dup": 63.0, "dupL": 1920.0}
    th = simt_fit["theta"]
    simt_raw = {
        "intercept": float(th[0]),
        "cross": float(th[1]),
        "dupL_cube": float(th[2] / (s["dupL"] ** 3)),
        "L0_square": float(th[3] / (s["L0"] ** 2)),
        "K_x_L1_square": float(th[4] / (s["K"] * (s["L1"] ** 2))),
        "dupL": float(th[5] / s["dupL"]),
        "E_square": float(th[6] / (s["E"] ** 2)),
        "E_x_dupL": float(th[7] / (s["E"] * s["dupL"])),
    }

    payload = OrderedDict([
        ("model_version", "strided_load_v6_nonneg_additive"),
        ("description", "Non-negative additive strided-load models, no log target."),
        ("simd_target", "ns"),
        ("simt_target", "ns/iteration"),
        ("alignment", "aligned base only; misalignment deferred"),
        ("data", {
            "simd": "results/model_v2/dataset.csv (simd rows)",
            "simt": "results/model_v3_numwarps/dataset.csv",
        }),
        ("simd_wide", model_payload("simd_wide", wide_fit, {"raw_coefficients": wide_raw})),
        ("simd_gather", model_payload("simd_gather", gather_fit, {"raw_coefficients": gather_raw})),
        ("simt", model_payload("simt", simt_fit, {
            "normalizers": s,
            "raw_coefficients": simt_raw,
            "feature_note": "K,E,L0,L1,dup,dupL are normalised by the listed constants before the powers/products are formed.",
        })),
    ])
    (out_dir / "model_v6_nonneg.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

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

    # --- per-point errors -------------------------------------------------
    all_error_rows = []
    group_info = {
        "simd_wide": (wide_fit, ()),
        "simd_gather": (gather_fit, ()),
        "simt": (simt_fit, ("num_warps",)),
    }
    combined_fields = ["group"] + [f for info in group_info.values() for f in info[1]]
    for group, (fit, extra_fields) in group_info.items():
        fit["rows"].assign(target=fit["rows"]["target"])
        out_path = out_dir / f"errors_{group}.csv"
        write_error_csv(out_path, list(fit["rows"].to_dict("records")),
                        fit["pred"], extra_fields)
        for i, r in enumerate(fit["rows"].to_dict("records")):
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
            all_error_rows.append(rec)
    with (out_dir / "errors_all.csv").open("w", newline="") as f:
        fields = ["group", "index", "block", "stride", "num_warps",
                  "target_ns", "pred_ns", "error_pct", "abs_error_pct"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_error_rows)

    print("simd_wide  ", wide_fit["metrics"])
    print("simd_gather", gather_fit["metrics"])
    print("simt       ", simt_fit["metrics"])
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
