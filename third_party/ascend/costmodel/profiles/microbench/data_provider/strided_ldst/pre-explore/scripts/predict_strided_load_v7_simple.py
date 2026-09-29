#!/usr/bin/env python3
"""Reference predictor for the strided-load model v7-simple.

Model family (same as v6 for SIMD, simplified for SIMT):

    T = c0 + c1*f1 + c2*f2 + ...        all c_i >= 0
    T is raw ns for SIMD single-cold-load
    T is raw ns/iteration for SIMT rotate-loop

SIMD:
  * wide   (stride=1/2): [1, stride-1, I(block>64), max(0, block-64)]
  * gather (stride>=3): [1, block, bucket_pairs(2048,8),
                            bucket_worst(2048,8), floor(span/4096)]

SIMT (v7 simple, four derived terms, no powers, no log):
    E      = block / (32 * num_warps)
    dup    = max(0, 1/E - 1)
    L      = aligned distinct 128B lines
    cross  = 1 if (block-1)*stride*4 >= 4096 else 0
    T_simt = c0
           + c1*dupL
           + c2*min(L, 64)
           + c3*max(0, num_warps*L - 3072)
           + c4*dup*cross

Use ``--verify`` to re-run the model on the committed ``errors_all.csv`` and
compare every prediction and metric against ``model_v7_simple.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent  # pre-explore root (scripts/ lives directly under it)
DEFAULT_MODEL = HERE / "results/model_v7_simple/model_v7_simple.json"
DEFAULT_ERRORS = HERE / "results/model_v7_simple/errors_all.csv"

ELEM_BYTES = 4
LINE_BYTES = 128
PAGE_BYTES = 4096


# ---------------------------------------------------------------------------
# Model loading / feature helpers
# ---------------------------------------------------------------------------

def load_model(path: Path = DEFAULT_MODEL):
    return json.loads(Path(path).read_text())


def _bucket_pair_worst(block: int, stride: int, g: int = 2048, banks: int = 8):
    """Gather-command 2KB/8-bank conflict features (same as v6)."""
    counts = [0] * banks
    for i in range(int(block)):
        counts[((i * int(stride) * ELEM_BYTES) // g) % banks] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1
    return float(pairs), float(worst)


def aligned_lines(block: int, stride: int) -> float:
    """Aligned-base distinct 128B lines touched by a strided tile."""
    stride_bytes = stride * ELEM_BYTES
    if stride_bytes >= LINE_BYTES:
        return float(block)
    return float(((block - 1) * stride_bytes) // LINE_BYTES + 1)


def predict_simd_wide(model, block: int, stride: int) -> float:
    coeff = model["simd_wide"]["coefficients"]
    f = [
        1.0,
        float(stride - 1),
        1.0 if block > 64 else 0.0,
        float(max(0, block - 64)),
    ]
    return float(np.dot(coeff, f))


def predict_simd_gather(model, block: int, stride: int) -> float:
    coeff = model["simd_gather"]["coefficients"]
    pairs, worst = _bucket_pair_worst(block, stride)
    span = (block - 1) * stride * ELEM_BYTES
    cross_count = math.floor(span / PAGE_BYTES)
    f = [1.0, float(block), pairs, worst, float(cross_count)]
    return float(np.dot(coeff, f))


def predict_simd(model, block: int, stride: int) -> float:
    if stride in (1, 2):
        return predict_simd_wide(model, block, stride)
    return predict_simd_gather(model, block, stride)


def predict_simt(model, block: int, stride: int, num_warps: int) -> float:
    coeff = model["simt"]["coefficients"]

    span = (block - 1) * stride * ELEM_BYTES
    cross = 1.0 if span >= PAGE_BYTES else 0.0
    lines = aligned_lines(block, stride)
    e = block / (32.0 * num_warps)
    dup = max(0.0, 1.0 / e - 1.0) if e > 0 else 0.0
    dup_l = dup * lines

    f = [
        1.0,
        dup_l,
        min(lines, 64.0),
        max(0.0, float(num_warps) * lines - 3072.0),
        dup * cross,
    ]
    return float(np.dot(coeff, f))


# ---------------------------------------------------------------------------
# Verification against committed fit outputs / model JSON
# ---------------------------------------------------------------------------

EXPECTED_FEATURES = {
    "simd_wide": ["1", "stride_minus_1", "large_tile_gt64",
                  "tail_elems_gt64"],
    "simd_gather": ["1", "n_cmd_block", "bucket_pairs_g2048_b8",
                    "bucket_worst_g2048_b8", "page_cross_count"],
    "simt": ["1", "dupL", "min_L_64", "max_WL_minus_3072", "dup_cross"],
}


def error_metrics(target, pred):
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
    }


def verify(model_path: Path = DEFAULT_MODEL,
           errors_path: Path = DEFAULT_ERRORS,
           verbose: bool = True) -> int:
    model = load_model(model_path)
    problems = []

    # 1) formula shape: only expected simple features and non-negative weights.
    for group, expected in EXPECTED_FEATURES.items():
        if group not in model:
            problems.append(f"missing model group {group}")
            continue
        names = list(model[group].get("feature_names", []))
        coeff = list(model[group].get("coefficients", []))
        if names != expected:
            problems.append(f"{group}: feature names {names} != {expected}")
        if len(coeff) != len(expected):
            problems.append(f"{group}: {len(coeff)} coefficients for {len(expected)} features")
        if any((not math.isfinite(c)) or c < 0.0 for c in coeff):
            problems.append(f"{group}: negative/non-finite coefficient {coeff}")

    # 2) predictor must reproduce every committed per-point prediction.
    with Path(errors_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        problems.append(f"no rows in {errors_path}")

    grouped_targets = {}
    grouped_preds = {}
    max_abs_pred_diff = 0.0
    max_rel_pred_error = 0.0
    for row in rows:
        group = row["group"]
        block = int(row["block"])
        stride = int(row["stride"])
        if group == "simd_wide":
            pred = predict_simd_wide(model, block, stride)
        elif group == "simd_gather":
            pred = predict_simd_gather(model, block, stride)
        elif group == "simt":
            pred = predict_simt(model, block, stride, int(row["num_warps"]))
        else:
            problems.append(f"unknown group {group!r} in errors csv")
            continue
        fit_pred = float(row["pred_ns"])
        max_abs_pred_diff = max(max_abs_pred_diff, abs(pred - fit_pred))
        scale = max(abs(fit_pred), 1e-12)
        max_rel_pred_error = max(max_rel_pred_error, abs(pred - fit_pred) / scale)
        grouped_targets.setdefault(group, []).append(float(row["target_ns"]))
        grouped_preds.setdefault(group, []).append(pred)

    # 3) metrics recomputed from the committed errors CSV must match the model JSON.
    metric_diffs = {}
    for group, targets in grouped_targets.items():
        preds = grouped_preds[group]
        metrics = error_metrics(targets, preds)
        stored = model[group]["metrics"]
        for key, value in metrics.items():
            if key == "n":
                if int(stored[key]) != int(value):
                    problems.append(f"{group}.n: {stored[key]} != {value}")
                continue
            diff = abs(float(stored[key]) - float(value))
            metric_diffs.setdefault(group, {})[key] = diff
            # errors_all.csv stores 6 decimal places, so accept small
            # floating-point reporting tolerance.
            if diff > 1e-4:
                problems.append(f"{group}.{key}: {stored[key]} != {value} (diff {diff})")

    if verbose:
        print(f"model: {model_path}")
        print(f"errors: {errors_path} ({len(rows)} rows)")
        for group in ("simd_wide", "simd_gather", "simt"):
            c = model[group]["coefficients"]
            print(f"  {group:12s} n={model[group]['n_rows']:4d} "
                  f"MAPE={model[group]['metrics']['mape_pct']:.4f}% "
                  f"coeff={['%.9g' % v for v in c]}")
        print(f"  max |predict - errors.pred_ns| = {max_abs_pred_diff:.3e}")
        print(f"  max predictor relative diff    = {max_rel_pred_error:.3e}")
        if metric_diffs:
            worst = max(d for group in metric_diffs.values() for d in group.values())
            print(f"  max recomputed metric diff     = {worst:.3e} percentage points")

    if problems:
        print("VERIFY FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("VERIFY OK: predictor matches model_v7_simple.json / errors_all.csv")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["simd", "simt", "simt_only"])
    ap.add_argument("block", nargs="?", type=int)
    ap.add_argument("stride", nargs="?", type=int)
    ap.add_argument("num_warps", nargs="?", type=int, default=1)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--verify", action="store_true",
                    help="recompute all predictions/metrics and compare with the model JSON")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(verify(args.model))

    if args.mode is None or args.block is None or args.stride is None:
        ap.error("mode, block and stride are required unless --verify is used")

    model = load_model(args.model)
    if args.mode == "simd":
        value = predict_simd(model, args.block, args.stride)
        print(f"{value:.6f} ns")
    else:
        value = predict_simt(model, args.block, args.stride, args.num_warps)
        print(f"{value:.6f} ns/iteration")


if __name__ == "__main__":
    main()
