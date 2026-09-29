#!/usr/bin/env python3
"""Predictor and self-check for the strided-store model v1.

Examples
--------
  python3 scripts/predict_strided_store_v1.py simd 128 16
  python3 scripts/predict_strided_store_v1.py simt_only 128 16 --num-warps 4
  python3 scripts/predict_strided_store_v1.py --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_store_dataset as bsd  # noqa: E402
from fit_store_common import metrics  # noqa: E402


def branch_for(mode, stride):
    if mode == "simt":
        mode = "simt_only"
    if mode == "simd":
        return "simd_wide" if int(stride) == 1 else "simd_gather"
    if mode == "simt_only":
        return "simt"
    raise ValueError(f"unsupported mode {mode!r}")


def predict_row(model, mode, block, stride, num_warps):
    raw = {"mode": mode, "block": int(block), "stride": int(stride),
           "num_warps": int(num_warps)}
    row = bsd.add_features(raw)
    branch_name = branch_for(mode, stride)
    branch = model["branches"][branch_name]
    value = float(branch["intercept_ns"])
    for name, coef in zip(branch["features"], branch["coefficients_ns"]):
        value += float(coef) * float(row[name])
    return branch_name, value, row


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify(model_path, dataset_path, errors_path):
    model = json.loads(Path(model_path).read_text())
    dataset = read_csv(dataset_path)
    errors = read_csv(errors_path)
    if len(dataset) != len(errors):
        raise SystemExit(f"dataset/errors length mismatch: {len(dataset)} vs {len(errors)}")
    max_pred_abs = 0.0
    max_pred_rel = 0.0
    recomputed = {}
    for row, err in zip(dataset, errors):
        mode = row["mode"]
        block = int(float(row["block"]))
        stride = int(float(row["stride"]))
        num_warps = int(float(row["num_warps"]))
        branch, pred, _ = predict_row(model, mode, block, stride, num_warps)
        old = float(err["pred_ns"])
        max_pred_abs = max(max_pred_abs, abs(pred - old))
        if old:
            max_pred_rel = max(max_pred_rel, abs(pred - old) / abs(old))
        if branch != err["branch"]:
            raise SystemExit(f"branch mismatch for {mode} b{block} s{stride}")
        recomputed.setdefault(branch, {"target": [], "pred": []})
        recomputed[branch]["target"].append(float(row["target_ns"]))
        recomputed[branch]["pred"].append(pred)
    # Overall metrics.
    all_target = [t for bucket in recomputed.values() for t in bucket["target"]]
    all_pred = [p for bucket in recomputed.values() for p in bucket["pred"]]
    recomputed["overall"] = {"target": all_target, "pred": all_pred}
    max_metric_diff = 0.0
    for branch, bucket in recomputed.items():
        fresh = metrics(bucket["target"], bucket["pred"])
        if branch == "overall":
            stored = model["metrics"]["overall"]
        else:
            stored = model["branches"][branch]["metrics"]
        for key in ("mape_pct", "p50_pct", "p90_pct", "p95_pct", "max_pct",
                    "bias_pct", "rmse_pct", "rmse_ns"):
            max_metric_diff = max(max_metric_diff, abs(fresh[key] - float(stored[key])))
    # Formula checks.
    for branch_name, branch in model["branches"].items():
        for name, coef in zip(branch["features"], branch["coefficients_ns"]):
            lowered = name.lower()
            if any(token in lowered for token in ("log", "^", "**", "sqrt", "pow")):
                raise SystemExit(f"forbidden feature {name!r} in {branch_name}")
            if float(coef) < 0.0:
                raise SystemExit(f"negative coefficient {name!r} in {branch_name}")
        if float(branch["intercept_ns"]) < 0.0:
            raise SystemExit(f"negative intercept in {branch_name}")
    print(f"max |predict - errors.pred_ns| = {max_pred_abs:.3e}")
    print(f"max predictor relative diff   = {max_pred_rel:.3e}")
    print(f"max recomputed metric diff    = {max_metric_diff:.3e} percentage points")
    print("coefficients non-negative, no power/log feature names")
    print("VERIFY OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?")
    ap.add_argument("block", nargs="?", type=int)
    ap.add_argument("stride", nargs="?", type=int)
    ap.add_argument("--num-warps", type=int, default=1)
    ap.add_argument("--model", type=Path,
                    default=ROOT / "results/model_store_v1/model_store_v1.json")
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "results/model_store_v1/dataset.csv")
    ap.add_argument("--errors", type=Path,
                    default=ROOT / "results/model_store_v1/errors_all.csv")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args.model, args.dataset, args.errors)
        return
    if args.mode is None or args.block is None or args.stride is None:
        ap.error("provide mode block stride, or --verify")
    model = json.loads(args.model.read_text())
    branch, value, _ = predict_row(model, args.mode, args.block, args.stride,
                                   args.num_warps)
    print(f"{branch}: {value:.6f} ns")


if __name__ == "__main__":
    main()
