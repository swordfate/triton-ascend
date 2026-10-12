#!/usr/bin/env python3
"""Fit the v8 semi-white-box SIMT strided-load model on the v4 dataset.

The model family is additive, raw-domain, and non-negative:

    T(ns/iteration) = c0 + sum_i c_i * f_i,  c_i >= 0

Only facts derivable from ``(block, stride, num_warps)`` are used.  The
candidate set contains the v7 baseline, linear/quadratic interaction controls,
and a preferred family built from the half-white-box terms

    L      = aligned distinct 128B lines
    K      = W * max(1, ceil(block / (32 W)))   warp instructions
    dup    = max(0, 1/E - 1), E = block / (32 W)
    cross  = 1 if span >= 4096 else 0

plus bounded line caps and a page-cross/line interaction.  Selection uses
pooled leave-one-block-out (LOBO) and leave-one-num_warps-out (LOWO) metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
DEFAULT_DATASET = HERE / "results/model_v4_numwarps/dataset.csv"
DEFAULT_OUTDIR = HERE / "results/model_v4_simple"

ELEM_BYTES = 4
LINE_BYTES = 128
PAGE_BYTES = 4096
WARP = 32
TARGET_FLOOR_NS = 120.0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def nonneg_lstsq(A: np.ndarray, b: np.ndarray, tol: float = 1e-10,
                 max_iter: int = 10000):
    """Lawson-Hanson active-set NNLS (same solver as the v7 fit)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    _, n = A.shape
    x = np.zeros(n, dtype=float)
    passive = []
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
    if passive:
        idx = np.asarray(passive, dtype=int)
        x[idx] = np.maximum(np.linalg.lstsq(A[:, idx], b, rcond=None)[0], 0.0)
    return x


def fit_weighted(X, y, alpha):
    """NNLS with sample weights y^-alpha (alpha=0 => ordinary NNLS)."""
    y = np.asarray(y, dtype=float)
    if alpha == 0.0:
        return nonneg_lstsq(X, y)
    w = np.power(y, -alpha)
    return nonneg_lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w))


def error_metrics(target, pred) -> dict:
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rel = pred / target - 1.0
    ae = np.abs(rel) * 100.0
    return {
        "n": int(len(ae)),
        "mape_pct": float(ae.mean()),
        "p50_pct": float(np.percentile(ae, 50)),
        "p90_pct": float(np.percentile(ae, 90)),
        "p95_pct": float(np.percentile(ae, 95)),
        "max_pct": float(ae.max()),
        "bias_pct": float(rel.mean() * 100.0),
        "rmse_pct": float(np.sqrt(np.mean(rel ** 2)) * 100.0),
        "within_5pct": float((ae <= 5.0).mean() * 100.0),
        "within_10pct": float((ae <= 10.0).mean() * 100.0),
        "within_20pct": float((ae <= 20.0).mean() * 100.0),
        "within_30pct": float((ae <= 30.0).mean() * 100.0),
        "within_50pct": float((ae <= 50.0).mean() * 100.0),
    }


# ---------------------------------------------------------------------------
# Dataset / feature construction
# ---------------------------------------------------------------------------

def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_flagged(row) -> bool:
    return (float(row.get("target_spread", 0.0)) > 1.4 or
            int(float(row.get("n_samples", 0.0))) < 3 or
            int(float(row.get("n_dropped", 0.0))) > 0)


def col(rows, name) -> np.ndarray:
    return np.array([float(r[name]) for r in rows], dtype=float)


def derived_terms(rows):
    block = col(rows, "block")
    warps = col(rows, "num_warps")
    lines = col(rows, "L")
    k = col(rows, "K")
    cross = col(rows, "cross_page")
    e = block / (WARP * warps)
    dup = np.maximum(0.0, 1.0 / np.maximum(e, 1e-12) - 1.0)
    return block, warps, lines, k, cross, dup


def line_request_size(rows):
    stride_bytes = col(rows, "stride_bytes")
    elems_per_line = np.maximum(1.0, np.floor(LINE_BYTES / stride_bytes))
    return np.where(stride_bytes >= LINE_BYTES, ELEM_BYTES,
                    np.minimum(LINE_BYTES, elems_per_line * ELEM_BYTES))


def feature_vector(rows, name) -> np.ndarray:
    block, warps, lines, k, cross, dup = derived_terms(rows)
    if name == "1":
        return np.ones(len(rows))
    if name == "K":
        return k
    if name == "W":
        return warps
    if name == "L":
        return lines
    if name == "dup":
        return dup
    if name == "dupL":
        return dup * lines
    if name == "dupL_L":
        return dup * lines * lines
    if name == "cross":
        return cross
    if name == "dup_cross":
        return dup * cross
    if name == "page_count":
        return col(rows, "page_count")
    if name == "line_req":
        return line_request_size(rows)
    if name == "line_req_L":
        return line_request_size(rows) * lines
    if name == "dupL_line_req":
        return dup * lines * line_request_size(rows)
    if name in ("bank_pairs_g2048_b8", "bank_worst_g2048_b8",
                "bank_pairs_g32768_b16", "bank_worst_g32768_b16"):
        return col(rows, name)
    if name.startswith("min_L_"):
        cap = float(name[len("min_L_"):])
        return np.minimum(lines, cap)
    if name.startswith("max_L_minus_"):
        cap = float(name[len("max_L_minus_"):])
        return np.maximum(0.0, lines - cap)
    if name.startswith("max_WL_minus_"):
        cap = float(name[len("max_WL_minus_"):])
        return np.maximum(0.0, warps * lines - cap)
    if name.startswith("dupL_minL_"):
        cap = float(name[len("dupL_minL_"):])
        return dup * lines * np.minimum(lines, cap)
    if name == "dupL_minL":  # compatibility name used by earlier candidates
        return dup * lines * np.minimum(lines, 64.0)
    if name.startswith("cross_minL_"):
        cap = float(name[len("cross_minL_"):])
        return cross * np.minimum(lines, cap)
    if name == "cross_minL":  # compatibility name
        return cross * np.minimum(lines, 64.0)
    if name.startswith("lowW_") and name.endswith("_L"):
        cap = float(name[len("lowW_"):-2])
        return np.maximum(0.0, cap - warps) * lines
    if name.startswith("lowW_") and "_minL_" in name:
        rest = name[len("lowW_"):]
        cap_text, line_text = rest.split("_minL_", 1)
        cap = float(cap_text)
        line_cap = float(line_text)
        return np.maximum(0.0, cap - warps) * np.minimum(lines, line_cap)
    raise KeyError(f"unknown feature {name!r}")


def design_matrix(rows, features):
    return np.column_stack([feature_vector(rows, f) for f in features])


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def candidate_specs():
    """Return an ordered dict name -> feature list."""
    specs = {}

    # Committed v7 reference (re-fit on the v4 data for fair comparison).
    specs["v7_reference"] = ["1", "dupL", "min_L_64",
                             "max_WL_minus_3072", "dup_cross"]

    # Linear, cross, and quadratic controls.
    specs["v8_linear"] = ["1", "K", "min_L_64", "max_L_minus_64",
                          "max_WL_minus_3072", "dupL", "cross", "dup_cross"]
    specs["v8_cross"] = ["1", "K", "min_L_64", "max_L_minus_64",
                         "max_WL_minus_3072", "dup", "dupL_minL", "cross"]
    specs["v8_nocross"] = ["1", "K", "min_L_64", "max_L_minus_64",
                           "max_WL_minus_3072", "dup", "dupL_minL"]
    specs["v8_quadratic"] = ["1", "K", "min_L_64", "max_L_minus_64",
                             "max_WL_minus_3072", "dup", "dupL_L", "cross"]

    # Preferred family: bounded first-line term, bounded line tail,
    # underfilled-warp replication interaction, page-cross line interaction,
    # plus optional in-flight-saturation and low-warp parallelism terms.
    #
    #  * line/tail caps 32 and 64 lines are fixed physical probes
    #    (32 lines = one 4KB page of 128B lines);
    #  * dup cap 64 vs 128 tests where replicated-line pressure saturates;
    #  * page cap 32 vs 64 tests the page-cross interaction width;
    #  * lowW_T remains only for W < T: with T=4 it is a candidate
    #    "fewer than four warp groups cannot hide line fill latency" term,
    #    and T=2/8/16 are stress controls rather than a hard-coded constant.
    for line_cap in (32, 64):
        for tail_cap in (32, 64):
            for dup_cap in (64, 128):
                for page_cap in (32, 64):
                    for wl_cap in (0, 3072):
                        for low_cap in (0, 2, 4, 8, 16):
                            features = [
                                "1",
                                f"min_L_{line_cap}",
                                f"max_L_minus_{tail_cap}",
                                f"dupL_minL_{dup_cap}",
                                f"cross_minL_{page_cap}",
                            ]
                            name = (f"v8_l{line_cap}_t{tail_cap}_"
                                    f"d{dup_cap}_p{page_cap}")
                            if wl_cap:
                                features.append(f"max_WL_minus_{wl_cap}")
                                name += f"_wl{wl_cap}"
                            if low_cap:
                                features.append(f"lowW_{low_cap}_L")
                                name += f"_lowW{low_cap}"
                            specs[name] = features

    # Optional explicable screens: page count, bank proxies.  They are kept in
    # the candidate list so NNLS can zero them if they do not help.
    specs["v8_screen_page"] = ["1", "min_L_32", "max_L_minus_32",
                               "dupL_minL_128", "cross_minL_32",
                               "lowW_4_L", "page_count"]
    specs["v8_screen_bank"] = ["1", "min_L_32", "max_L_minus_32",
                               "dupL_minL_128", "cross_minL_32",
                               "lowW_4_L", "bank_pairs_g2048_b8",
                               "bank_worst_g2048_b8"]
    specs["v8_screen_K"] = ["1", "K", "min_L_32", "max_L_minus_32",
                            "dup", "dupL_minL_128", "cross_minL_32",
                            "lowW_4_L"]
    return specs


# ---------------------------------------------------------------------------
# Cross validation
# ---------------------------------------------------------------------------

def target_values(rows, target_col):
    return np.array([float(r[target_col]) for r in rows], dtype=float)


def fit_cv(rows, features, alpha, target_col, group_key):
    X = design_matrix(rows, features)
    y = target_values(rows, target_col)
    groups = np.array([int(float(r[group_key])) for r in rows], dtype=int)
    pred = np.zeros(len(rows), dtype=float)
    folds = []
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        theta = fit_weighted(X[train], y[train], alpha)
        pred[test] = X[test] @ theta
        folds.append({
            "fold": int(group),
            "group_key": group_key,
            "n": int(test.sum()),
            "metrics": error_metrics(y[test], pred[test]),
        })
    pooled = error_metrics(y, pred)
    fold_mapes = [f["metrics"]["mape_pct"] for f in folds]
    return pooled, folds, float(np.mean(fold_mapes)), pred


def candidate_row(spec_name, features, alpha, ins, lobo, lowo):
    row = {
        "spec": spec_name,
        "features": ";".join(features),
        "n_features": len(features),
        "alpha": alpha,
    }
    for scope, metrics in (("in", ins), ("lobo", lobo), ("lowo", lowo)):
        for key in ("mape_pct", "p50_pct", "p90_pct", "p95_pct",
                    "max_pct", "bias_pct", "rmse_pct"):
            row[f"{scope}_{key}"] = metrics[key]
    row["cv_score_pct"] = 0.5 * (lobo["mape_pct"] + lowo["mape_pct"])
    row["cv_max_score_pct"] = 0.5 * (lobo["max_pct"] + lowo["max_pct"])
    return row


def select_candidate(rows, specs, alphas, target_col):
    """Evaluate every candidate and return (rows, chosen dict)."""
    rows_out = []
    detailed = {}
    for spec_name, features in specs.items():
        for alpha in alphas:
            X = design_matrix(rows, features)
            y = target_values(rows, target_col)
            theta = fit_weighted(X, y, alpha)
            pred = X @ theta
            ins = error_metrics(y, pred)
            lobo, lobo_folds, lobo_mean, _ = fit_cv(
                rows, features, alpha, target_col, "block")
            lowo, lowo_folds, lowo_mean, _ = fit_cv(
                rows, features, alpha, target_col, "num_warps")
            row = candidate_row(spec_name, features, alpha, ins, lobo, lowo)
            row["lobo_mean_fold_mape_pct"] = lobo_mean
            row["lowo_mean_fold_mape_pct"] = lowo_mean
            row["nonnegative"] = all(v >= 0.0 for v in theta)
            row["nonzero_terms"] = int(np.count_nonzero(theta > 1e-12))
            rows_out.append(row)
            detailed[(spec_name, alpha)] = {
                "spec": spec_name,
                "features": features,
                "alpha": alpha,
                "theta": [float(v) for v in theta],
                "ins": ins,
                "lobo": lobo,
                "lobo_folds": lobo_folds,
                "lowo": lowo,
                "lowo_folds": lowo_folds,
            }

    # Treat numerically identical CV scores as ties, then prefer the most
    # compact formula.  The screens deliberately contain zero-valued extra
    # terms, so they must not win merely because NNLS introduced 1e-13 noise.
    best_cv = min(r["cv_score_pct"] for r in rows_out)
    tolerance = 1e-9 * max(1.0, abs(best_cv))
    near = [r for r in rows_out
            if abs(r["cv_score_pct"] - best_cv) <= tolerance]
    best_row = min(
        near,
        key=lambda r: (r["nonzero_terms"], r["n_features"],
                       r["cv_max_score_pct"], r["in_max_pct"],
                       r["spec"], r["alpha"]))
    chosen = detailed[(best_row["spec"], best_row["alpha"])]
    return rows_out, chosen


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_metrics(rows, target, pred, group_key):
    groups = np.array([int(float(r[group_key])) for r in rows], dtype=int)
    out = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        metrics = error_metrics(target[mask], pred[mask])
        row = {"group_key": group_key, "group": int(group)}
        row.update(metrics)
        row["mean_target_ns"] = float(target[mask].mean())
        row["mean_pred_ns"] = float(pred[mask].mean())
        out.append(row)
    return out


def formula_text(features, theta):
    names = ["intercept"] + features[1:]
    terms = []
    for name, value in zip(names, theta):
        terms.append(f"{float(value):.9g}*{name}")
    return " + ".join(terms)


def feature_notes():
    return {
        "1": "fixed SIMT start/control/first-LDG cost, ns",
        "K": "W * max(1, ceil(block/(32W))) SIMT_LDG warp instructions",
        "L": "distinct aligned 128B lines touched by the strided tile",
        "W": "num_warps, i.e. logical SIMT warp groups",
        "dup": "max(0, 1/E-1), E=block/(32W); underfilled-warp replication factor",
        "dupL": "dup * L; replicated line requests",
        "dupL_L": "dup * L^2; quadratic replication control",
        "dupL_minL_32": "dup * L * min(L,32); replicated line requests capped at 32 lines",
        "dupL_minL_64": "dup * L * min(L,64); replicated line requests capped at 64 lines",
        "dupL_minL_128": "dup * L * min(L,128); replicated line requests capped at 128 lines (16KB)",
        "dupL_minL_256": "dup * L * min(L,256); replicated line requests capped at 256 lines (32KB)",
        "cross": "1 if (block-1)*stride*4 >= 4096 else 0; tile crosses a 4KB page",
        "cross_minL_32": "cross * min(L,32); page-cross penalty on at most 32 lines (one 4KB page of 128B lines)",
        "cross_minL_64": "cross * min(L,64); page-cross penalty on at most 64 lines",
        "lowW_2_L": "max(0,2-W)*L; fewer than two warps cannot hide line fill latency",
        "lowW_4_L": "max(0,4-W)*L; fewer than four warps cannot hide line fill latency",
        "lowW_8_L": "max(0,8-W)*L; fewer than eight warps cannot hide line fill latency",
        "lowW_16_L": "max(0,16-W)*L; fewer than sixteen warps cannot hide line fill latency",
        "lowW_4_minL_32": "max(0,4-W)*min(L,32); low-warp deficit bounded to one 4KB page of lines",
        "page_count": "floor(((block-1)*stride*4)/4096); number of fully crossed 4KB pages",
        "min_L_32": "min(L,32); first 32 128B lines of fill/queue work",
        "min_L_64": "min(L,64); first 64 128B lines of fill/queue work",
        "max_L_minus_32": "max(0,L-32); additional lines past 32, cheaper due to overlap",
        "max_L_minus_64": "max(0,L-64); additional lines past 64, cheaper due to overlap",
        "max_WL_minus_3072": "max(0,W*L-3072); warp-line requests above the in-flight saturation threshold",
        "max_WL_minus_4096": "max(0,W*L-4096); warp-line requests above a higher saturation threshold",
        "line_req": "useful bytes carried by one 128B line (4..128B) from aligned-base geometry",
        "line_req_L": "line_req * L; line-density/request-size interaction",
        "dupL_line_req": "dup * L * line_req; underfilled replication weighted by request size",
        "bank_pairs_g2048_b8": "pairs of tile elements sharing a 2KB/8-bank bucket",
        "bank_worst_g2048_b8": "largest 2KB/8-bank bucket count minus one",
        "bank_pairs_g32768_b16": "pairs sharing a 32KB/16-group bucket",
        "bank_worst_g32768_b16": "largest 32KB/16-group bucket count minus one",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--target-col", default="target_ns",
                    help="target_ns (median) or target_p25_ns (robust)")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1.0, 1.5, 2.0])
    ap.add_argument("--max-spread", type=float, default=0.0,
                    help="optional hard filter: drop target_spread > value")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="exclude rows flagged as discordant before fitting")
    ap.add_argument("--chosen-spec", default=None,
                    help="force a candidate spec name (for reproducibility)")
    ap.add_argument("--chosen-alpha", type=float, default=None,
                    help="force an alpha (requires --chosen-spec)")
    args = ap.parse_args()

    if args.chosen_alpha is not None and args.chosen_spec is None:
        ap.error("--chosen-alpha requires --chosen-spec")

    rows = load_rows(args.dataset)
    if "valid" in rows[0]:
        rows = [r for r in rows
                if str(r["valid"]).strip().lower() in ("true", "1", "yes")]
    if args.max_spread and args.max_spread > 0:
        before = len(rows)
        rows = [r for r in rows
                if float(r["target_spread"]) <= args.max_spread]
        print(f"spread filter: kept {len(rows)}/{before} rows")
    if args.exclude_flagged:
        before = len(rows)
        rows = [r for r in rows if not is_flagged(r)]
        print(f"flagged filter: kept {len(rows)}/{before} rows")
    if not rows:
        raise SystemExit("no rows after filtering")
    if args.target_col not in rows[0]:
        raise SystemExit(f"dataset has no target column {args.target_col!r}")

    specs = candidate_specs()
    candidate_rows, chosen = select_candidate(
        rows, specs, args.alphas, args.target_col)

    if args.chosen_spec is not None:
        matches = [c for c in candidate_rows
                   if c["spec"] == args.chosen_spec and
                   (args.chosen_alpha is None or
                    abs(c["alpha"] - args.chosen_alpha) < 1e-12)]
        if not matches:
            raise SystemExit(
                f"requested candidate {args.chosen_spec} alpha={args.chosen_alpha} not found")
        best_row = matches[0]
        # Recompute the full detail for the forced candidate.
        _, chosen = select_candidate(
            rows, {args.chosen_spec: specs[args.chosen_spec]},
            [best_row["alpha"]], args.target_col)
    else:
        best_cv = min(r["cv_score_pct"] for r in candidate_rows)
        tolerance = 1e-9 * max(1.0, abs(best_cv))
        near = [r for r in candidate_rows
                if abs(r["cv_score_pct"] - best_cv) <= tolerance]
        best_row = min(
            near,
            key=lambda r: (r["nonzero_terms"], r["n_features"],
                           r["cv_max_score_pct"], r["in_max_pct"],
                           r["spec"], r["alpha"]))

    features = chosen["features"]
    features = chosen["features"]
    theta = np.array(chosen["theta"], dtype=float)
    X = design_matrix(rows, features)
    y = target_values(rows, args.target_col)
    pred = X @ theta
    ins_metrics = error_metrics(y, pred)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Candidate table, one row per (spec, alpha).
    write_csv(outdir / "cv_candidates.csv", candidate_rows)

    # Chosen candidate CV details and fold-level metrics.
    lobo_pooled = chosen["lobo"]
    lowo_pooled = chosen["lowo"]
    lobo_folds = chosen["lobo_folds"]
    lowo_folds = chosen["lowo_folds"]
    for scope, pooled, folds in (
            ("lobo", lobo_pooled, lobo_folds),
            ("lowo", lowo_pooled, lowo_folds)):
        payload = {
            "target_col": args.target_col,
            "spec": chosen["spec"],
            "alpha": chosen["alpha"],
            "group_key": "block" if scope == "lobo" else "num_warps",
            "pooled": pooled,
            "mean_fold_mape_pct": float(np.mean(
                [f["metrics"]["mape_pct"] for f in folds])),
            "folds": folds,
        }
        (outdir / f"cv_{scope}_folds.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    # Errors on the fitting rows.
    error_rows = []
    for row, p in zip(rows, pred):
        target = float(row[args.target_col])
        rel = float(p) / target - 1.0
        error_rows.append({
            "mode": "simt",
            "num_warps": int(float(row["num_warps"])),
            "block": int(float(row["block"])),
            "stride": int(float(row["stride"])),
            "target_col": args.target_col,
            "target_ns": target,
            "pred_ns": float(p),
            "relative_error": rel,
            "abs_pct_error": abs(rel) * 100.0,
            "target_spread": float(row.get("target_spread", 0.0)),
            "n_samples": int(float(row.get("n_samples", 0.0))),
            "n_dropped": int(float(row.get("n_dropped", 0.0))),
            "flagged_discordant": int(is_flagged(row)),
        })
    write_csv(outdir / "errors_all.csv", error_rows)

    # Grouped error tables.
    for key in ("block", "num_warps", "stride"):
        group_rows = group_metrics(rows, y, pred, key)
        write_csv(outdir / f"group_metrics_by_{key}.csv", group_rows)

    # Compact metrics summary.
    summary_rows = [
        {"scope": "in_sample", **ins_metrics},
        {"scope": "lobo_pooled", **lobo_pooled},
        {"scope": "lowo_pooled", **lowo_pooled},
        {"scope": "lobo_mean_fold",
         **{"n": int(np.mean([f["n"] for f in lobo_folds])),
            "mape_pct": float(np.mean([f["metrics"]["mape_pct"] for f in lobo_folds])),
            "p50_pct": float(np.mean([f["metrics"]["p50_pct"] for f in lobo_folds])),
            "p90_pct": float(np.mean([f["metrics"]["p90_pct"] for f in lobo_folds])),
            "p95_pct": float(np.mean([f["metrics"]["p95_pct"] for f in lobo_folds])),
            "max_pct": float(np.mean([f["metrics"]["max_pct"] for f in lobo_folds])),
            "bias_pct": float(np.mean([f["metrics"]["bias_pct"] for f in lobo_folds])),
            "rmse_pct": float(np.mean([f["metrics"]["rmse_pct"] for f in lobo_folds]))}},
        {"scope": "lowo_mean_fold",
         **{"n": int(np.mean([f["n"] for f in lowo_folds])),
            "mape_pct": float(np.mean([f["metrics"]["mape_pct"] for f in lowo_folds])),
            "p50_pct": float(np.mean([f["metrics"]["p50_pct"] for f in lowo_folds])),
            "p90_pct": float(np.mean([f["metrics"]["p90_pct"] for f in lowo_folds])),
            "p95_pct": float(np.mean([f["metrics"]["p95_pct"] for f in lowo_folds])),
            "max_pct": float(np.mean([f["metrics"]["max_pct"] for f in lowo_folds])),
            "bias_pct": float(np.mean([f["metrics"]["bias_pct"] for f in lowo_folds])),
            "rmse_pct": float(np.mean([f["metrics"]["rmse_pct"] for f in lowo_folds]))}},
    ]
    write_csv(outdir / "metrics_summary.csv", summary_rows)
    (outdir / "metrics_summary.json").write_text(
        json.dumps({
            "target_col": args.target_col,
            "spec": chosen["spec"],
            "alpha": chosen["alpha"],
            "features": features,
            "intercept": float(theta[0]),
            "coefficients": [float(v) for v in theta[1:]],
            "in_sample": ins_metrics,
            "lobo_pooled": lobo_pooled,
            "lowo_pooled": lowo_pooled,
            "lobo_mean_fold_mape_pct": summary_rows[3]["mape_pct"],
            "lowo_mean_fold_mape_pct": summary_rows[4]["mape_pct"],
        }, indent=2), encoding="utf-8")

    # Model JSON consumed by the README and by any later profile mapping.
    notes = feature_notes()
    model = {
        "model_version": "strided_load_v8_semi_white_box_v4",
        "description": (
            "Additive non-negative SIMT strided-load model fitted on the v4 "
            "1288-case page-aligned dataset.  Coefficients are ns/iteration; "
            "each term is a white-box line/duplication/page fact."
        ),
        "target": {
            "column": args.target_col,
            "unit": "ns/iteration",
            "domain": "board Event slope",
            "target_ns_definition": "median of cleaned allocation samples for target_ns",
            "target_p25_definition": "lower-quartile of cleaned allocation samples for target_p25_ns",
            "cleaning": {
                "floor_ns": TARGET_FLOOR_NS,
                "drop_non_positive": True,
                "flagged_rule": "spread>1.4 or n_samples<3 or n_dropped>0",
            },
        },
        "dataset": str(args.dataset),
        "n_rows": len(rows),
        "selection": {
            "criterion": (
                "pooled CV mean(LOBO MAPE, LOWO MAPE), then CV max, then "
                "in-sample max; the documented v4 final model is forced to "
                "the simpler no-WL + dup cap128 + lowW4 candidate"
                if args.chosen_spec else
                "pooled CV mean(LOBO MAPE, LOWO MAPE), then CV max, then "
                "in-sample max, with ties broken by formula compactness"
            ),
            "mode": "forced" if args.chosen_spec else "cv_auto",
            "chosen_spec": chosen["spec"],
            "chosen_alpha": chosen["alpha"],
            "lobo_pooled_mape_pct": lobo_pooled["mape_pct"],
            "lowo_pooled_mape_pct": lowo_pooled["mape_pct"],
        },
        "formula": formula_text(features, theta),
        "intercept_ns": float(theta[0]),
        "coefficients_ns": {features[i]: float(theta[i])
                            for i in range(1, len(features))},
        "feature_names": features,
        "feature_notes": {name: notes.get(name, "") for name in features},
        "metrics": {
            "in_sample": ins_metrics,
            "lobo_pooled": lobo_pooled,
            "lowo_pooled": lowo_pooled,
        },
        "profile_mapping": {
            "note": (
                "The coefficients are raw ns and map to the existing "
                "nanoseconds_to_system_cycles conversion.  Do not change the "
                "production profile until route-level validation passes."
            ),
            "formula_terms": [
                {"model_term": name, "quantity": notes.get(name, name)}
                for name in features[1:]
            ],
        },
        "alignment": "aligned base only; misalignment deferred",
    }
    (outdir / "model_v4_simple.json").write_text(
        json.dumps(model, indent=2, sort_keys=False), encoding="utf-8")

    # Console summary.
    print(f"dataset rows: {len(rows)} target={args.target_col} "
          f"flagged_in_fit={sum(is_flagged(r) for r in rows)}")
    print("top candidates by pooled CV:")
    for row in sorted(candidate_rows,
                      key=lambda r: (r["cv_score_pct"],
                                     r["cv_max_score_pct"]))[:10]:
        print(f"  {row['spec']:36s} a={row['alpha']:.1f} "
              f"in={row['in_mape_pct']:.2f}/{row['in_max_pct']:.1f} "
              f"cv={row['cv_score_pct']:.2f} "
              f"lobo={row['lobo_mape_pct']:.2f}/{row['lobo_max_pct']:.1f} "
              f"lowo={row['lowo_mape_pct']:.2f}/{row['lowo_max_pct']:.1f}")
    print(f"chosen: {chosen['spec']} alpha={chosen['alpha']} "
          f"features={features}")
    print(f"  in-sample MAPE={ins_metrics['mape_pct']:.3f}% "
          f"p50={ins_metrics['p50_pct']:.3f}% max={ins_metrics['max_pct']:.3f}%")
    print(f"  LOBO pooled MAPE={lobo_pooled['mape_pct']:.3f}% "
          f"max={lobo_pooled['max_pct']:.3f}%")
    print(f"  LOWO pooled MAPE={lowo_pooled['mape_pct']:.3f}% "
          f"max={lowo_pooled['max_pct']:.3f}%")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
