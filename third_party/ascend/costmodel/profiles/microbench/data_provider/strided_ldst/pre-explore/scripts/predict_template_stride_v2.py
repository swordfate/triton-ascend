#!/usr/bin/env python3
"""Predictor and verifier for the template stride v2 models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from template_stride_features import FEATURE_ORDER, feature_vector

HERE = Path(__file__).resolve().parent.parent
MODEL_DIR = HERE / "results/model_template_stride_v2"


def load_model(path: Path):
    return json.loads(Path(path).read_text())


def predict(model, block, stride, num_warps=32):
    vec = dict(zip(FEATURE_ORDER, feature_vector(block, stride, num_warps)))
    total = float(model["intercept"])
    for name, coef in zip(model["feature_names"], model["coefficients"]):
        total += float(coef) * vec[name]
    return total


def verify(model_path: Path, errors_path: Path):
    model = load_model(model_path)
    max_diff = 0.0
    n = 0
    with errors_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pred = predict(model, int(row["block"]), int(row["stride"]),
                           int(row["num_warps"]))
            max_diff = max(max_diff, abs(pred - float(row["pred_ns"])))
            n += 1
    print(f"verified {n} rows; max |predict - errors.pred_ns| = {max_diff:.6e}")
    if max_diff > 1e-6:
        raise SystemExit("VERIFY FAILED: predictor differs from errors file")
    print("VERIFY OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", choices=["load", "store"], nargs="?", default=None)
    ap.add_argument("block", type=int, nargs="?", default=None)
    ap.add_argument("stride", type=int, nargs="?", default=None)
    ap.add_argument("--num-warps", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        for stem in ("load", "store"):
            verify(MODEL_DIR / f"model_template_stride_{stem}_v2.json",
                   MODEL_DIR / f"errors_{stem}.csv")
        return
    if args.path is None or args.block is None or args.stride is None:
        raise SystemExit(
            "usage: predict_template_stride_v2.py <load|store> <block> <stride> "
            "[--num-warps 32]"
        )
    model = load_model(MODEL_DIR / f"model_template_stride_{args.path}_v2.json")
    print(f"{predict(model, args.block, args.stride, args.num_warps):.6f} ns/iteration")


if __name__ == "__main__":
    main()
