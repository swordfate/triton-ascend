#!/usr/bin/env python3
"""Evaluate the existing v7-simple SIMT strided-load model on the v4 dataset.

This reuses the committed v7 predictor and writes overall/grouped error tables
under ``results/model_v4_numwarps/v7_eval/``.  numpy + stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from predict_strided_load_v7_simple import error_metrics, load_model, predict_simt


SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
DEFAULT_DATASET = HERE / "results/model_v4_numwarps/dataset.csv"
DEFAULT_MODEL = HERE / "results/model_v7_simple/model_v7_simple.json"
DEFAULT_OUTDIR = HERE / "results/model_v4_numwarps/v7_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    model = load_model(args.model)
    with args.dataset.open(newline="") as f:
        rows = list(csv.DictReader(f))
    targets = np.array([float(r["target_ns"]) for r in rows], dtype=float)
    pred = np.array([
        predict_simt(model, int(r["block"]), int(r["stride"]),
                     int(r["num_warps"]))
        for r in rows
    ], dtype=float)
    overall = error_metrics(targets, pred)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "v7_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8")

    with (outdir / "v7_errors_all.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["num_warps", "block", "stride", "target_ns", "pred_ns",
                         "relative_error", "abs_pct_error"])
        for row, p in zip(rows, pred):
            target = float(row["target_ns"])
            rel = float(p) / target - 1.0
            writer.writerow([int(row["num_warps"]), int(row["block"]),
                             int(row["stride"]), f"{target:.6f}",
                             f"{float(p):.6f}", f"{rel:.9f}",
                             f"{abs(rel) * 100.0:.6f}"])

    for key in ("block", "num_warps", "stride"):
        groups = sorted({int(r[key]) for r in rows})
        with (outdir / f"v7_by_{key}.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["group", "n", "mape_pct", "p50_pct", "p90_pct",
                             "p95_pct", "max_pct", "bias_pct", "rmse_pct"])
            for group in groups:
                mask = np.array([int(r[key]) == group for r in rows])
                metrics = error_metrics(targets[mask], pred[mask])
                writer.writerow([group, metrics["n"], metrics["mape_pct"],
                                 metrics["p50_pct"], metrics["p90_pct"],
                                 metrics["p95_pct"], metrics["max_pct"],
                                 metrics["bias_pct"], metrics["rmse_pct"]])

    print(f"v7 on v4 dataset: n={overall['n']} "
          f"MAPE={overall['mape_pct']:.3f}% "
          f"p50={overall['p50_pct']:.3f}% max={overall['max_pct']:.3f}%")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
