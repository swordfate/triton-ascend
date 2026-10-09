#!/usr/bin/env python3
"""Compare the template strided model against the existing pure-SIMT models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import predict_strided_load_v7_simple as load_predict
import predict_strided_store_v1 as store_predict
from template_stride_features import FEATURE_ORDER, feature_vector

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DIR = HERE / "results/model_template_stride_v1"


def template_predict(model, block, stride, num_warps=1):
    vec = dict(zip(FEATURE_ORDER, feature_vector(block, stride, num_warps)))
    return float(model["intercept"]) + sum(
        float(coef) * vec[name]
        for name, coef in zip(model["feature_names"], model["coefficients"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--suffix", default="v1",
                    help="model file suffix, e.g. v1 or v2")
    args = ap.parse_args()
    out_dir = args.dir
    load_model = json.loads((HERE / "results/model_v7_simple/model_v7_simple.json").read_text())
    store_model = json.loads((HERE / "results/model_store_v1/model_store_v1.json").read_text())
    tload = json.loads((out_dir / f"model_template_stride_load_{args.suffix}.json").read_text())
    tstore = json.loads((out_dir / f"model_template_stride_store_{args.suffix}.json").read_text())
    rows = []
    with (out_dir / "dataset.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["valid"]) != 1 or int(row["correctness_ok"]) != 1:
                continue
            block = int(row["block"])
            stride = int(row["stride"])
            warps = int(row["num_warps"])
            if row["path"] == "triton_stride_load":
                pure = load_predict.predict_simt(load_model, block, stride, warps)
                template = template_predict(tload, block, stride, warps)
            else:
                _, pure, _ = store_predict.predict_row(
                    store_model, "simt_only", block, stride, warps
                )
                template = template_predict(tstore, block, stride, warps)
            rows.append({
                "path": row["path"], "block": block, "stride": stride,
                "num_warps": warps, "template_ns": template,
                "pure_simt_ns": pure, "ratio": template / pure,
            })
    with (out_dir / "template_vs_pure_simt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for path in ("triton_stride_load", "triton_stride_store"):
        subset = [r for r in rows if r["path"] == path]
        ratios = [r["ratio"] for r in subset]
        summary[path] = {
            "n": len(subset),
            "median_ratio": float(np.median(ratios)),
            "mean_ratio": float(np.mean(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
        }
    (out_dir / "template_vs_pure_simt_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
