#!/usr/bin/env python3
"""Reference predictor and self-check for the v8 SIMT strided-load model.

The model JSON is the source of truth: feature names and ns coefficients are
read from it, so this script follows whatever the fit selected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "results/model_v4_simple/model_v4_simple.json"
DEFAULT_ERRORS = ROOT / "results/model_v4_simple/errors_all.csv"

ELEM_BYTES = 4
LINE_BYTES = 128
PAGE_BYTES = 4096
WARP = 32


def compute_features(block: int, stride: int, num_warps: int):
    block = float(block)
    stride = float(stride)
    warps = float(num_warps)
    stride_bytes = ELEM_BYTES * stride
    span = (block - 1.0) * stride_bytes
    lines = (block if stride_bytes >= LINE_BYTES
             else math.floor(((block - 1.0) * stride_bytes) / LINE_BYTES) + 1.0)
    k = warps * max(1.0, math.ceil(block / (WARP * warps)))
    e = block / (WARP * warps)
    dup = max(0.0, 1.0 / e - 1.0) if e > 0 else 0.0
    cross = 1.0 if span >= PAGE_BYTES else 0.0
    page_count = math.floor(span / PAGE_BYTES)
    elems_per_line = max(1.0, math.floor(LINE_BYTES / stride_bytes))
    line_req = (ELEM_BYTES if stride_bytes >= LINE_BYTES
                else min(LINE_BYTES, elems_per_line * ELEM_BYTES))
    features = {
        "K": k,
        "W": warps,
        "L": lines,
        "dup": dup,
        "dupL": dup * lines,
        "dupL_L": dup * lines * lines,
        "dupL_minL": dup * lines * min(lines, 64.0),
        "cross": cross,
        "dup_cross": dup * cross,
        "page_count": page_count,
        "line_req": line_req,
        "line_req_L": line_req * lines,
        "dupL_line_req": dup * lines * line_req,
    }
    for cap in (32, 64, 128, 256):
        features[f"min_L_{cap}"] = min(lines, float(cap))
        features[f"max_L_minus_{cap}"] = max(0.0, lines - float(cap))
        features[f"dupL_minL_{cap}"] = dup * lines * min(lines, float(cap))
        features[f"cross_minL_{cap}"] = cross * min(lines, float(cap))
    for cap in (2048, 3072, 4096, 6144):
        features[f"max_WL_minus_{cap}"] = max(0.0, warps * lines - float(cap))
    for cap in (2, 4, 8, 16, 32):
        low_warp = max(0.0, float(cap) - warps)
        features[f"lowW_{cap}_L"] = low_warp * lines
        features[f"lowW_{cap}_minL_32"] = low_warp * min(lines, 32.0)
    # Bank features are not computed here; the files are only needed for
    # candidate screening, and the selected model does not use them.
    return features


def load_model(path: Path = DEFAULT_MODEL):
    return json.loads(Path(path).read_text())


def predict(model, block, stride, num_warps):
    features = compute_features(block, stride, num_warps)
    value = float(model["intercept_ns"])
    for name, coefficient in model["coefficients_ns"].items():
        if name not in features:
            raise KeyError(
                f"model term {name!r} is not implemented by the v8 predictor")
        value += float(coefficient) * float(features[name])
    return value


def metrics(target, pred):
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


def verify(model_path: Path = DEFAULT_MODEL, errors_path: Path = DEFAULT_ERRORS):
    model = load_model(model_path)
    rows = list(csv.DictReader(Path(errors_path).open(newline="")))
    if not rows:
        raise SystemExit(f"no rows in {errors_path}")
    targets = []
    preds = []
    max_abs = 0.0
    max_rel = 0.0
    for row in rows:
        target = float(row["target_ns"])
        pred = predict(model, int(float(row["block"])),
                       int(float(row["stride"])),
                       int(float(row["num_warps"])))
        targets.append(target)
        preds.append(pred)
        old = float(row["pred_ns"])
        max_abs = max(max_abs, abs(pred - old))
        max_rel = max(max_rel, abs(pred - old) / max(abs(old), 1e-12))
    recomputed = metrics(targets, preds)
    stored = model["metrics"]["in_sample"]
    max_metric = max(abs(recomputed[k] - float(stored[k]))
                     for k in ("mape_pct", "p50_pct", "p90_pct", "p95_pct",
                               "max_pct", "bias_pct", "rmse_pct"))
    print(f"model: {model_path}")
    print(f"errors: {errors_path} ({len(rows)} rows)")
    print(f"max |predict - errors.pred_ns| = {max_abs:.3e}")
    print(f"max predictor relative diff    = {max_rel:.3e}")
    print(f"max recomputed metric diff     = {max_metric:.3e} percentage points")
    if max_abs > 1e-6 or max_metric > 1e-6:
        raise SystemExit("VERIFY FAILED")
    print("VERIFY OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["simt", "simt_only"])
    ap.add_argument("block", nargs="?", type=int)
    ap.add_argument("stride", nargs="?", type=int)
    ap.add_argument("--num-warps", type=int, default=1)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args.model, args.errors)
        return
    if args.mode is None or args.block is None or args.stride is None:
        ap.error("provide mode block stride, or --verify")
    value = predict(load_model(args.model), args.block, args.stride,
                    args.num_warps)
    print(f"{value:.6f} ns/iteration")


if __name__ == "__main__":
    main()
