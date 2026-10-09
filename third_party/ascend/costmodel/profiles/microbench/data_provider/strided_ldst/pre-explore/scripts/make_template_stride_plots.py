#!/usr/bin/env python3
"""Generate parity/error plots for the template stride v1 models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT_DIR = HERE / "results/model_template_stride_v1"


def read_errors(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            row["block"] = int(row["block"])
            row["stride"] = int(row["stride"])
            row["target_ns"] = float(row["target_ns"])
            row["pred_ns"] = float(row["pred_ns"])
            row["abs_pct_error"] = float(row["abs_pct_error"])
            row["relative_error"] = float(row["relative_error"])
            row["split"] = row.get("split", "all")
            rows.append(row)
    return rows


def plot_pair(lines, out_dir):
    plt.figure(figsize=(6, 6))
    for label, rows in lines:
        x = [r["target_ns"] for r in rows]
        y = [r["pred_ns"] for r in rows]
        plt.scatter(x, y, s=8, alpha=0.6, label=label)
    lo = min(min(r["target_ns"] for _, rs in lines for r in rs),
             min(r["pred_ns"] for _, rs in lines for r in rs)) * 0.8
    hi = max(max(r["target_ns"] for _, rs in lines for r in rs),
             max(r["pred_ns"] for _, rs in lines for r in rs)) * 1.2
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("target ns/iteration")
    plt.ylabel("prediction ns/iteration")
    plt.title("Template strided load/store parity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "parity_loglog.png", dpi=150)
    plt.close()


def plot_error_hist(lines, out_dir):
    plt.figure(figsize=(7, 4))
    for label, rows in lines:
        signed = np.asarray([r["relative_error"] * 100.0 for r in rows])
        plt.hist(signed, bins=30, alpha=0.5, label=label)
    plt.xlabel("signed relative error (%)")
    plt.ylabel("count")
    plt.title("Template strided load/store error histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_hist.png", dpi=150)
    plt.close()


def plot_error_vs(series, group, out_dir):
    plt.figure(figsize=(7, 4))
    for label, rows in series:
        xs = [r[group] for r in rows]
        ys = [r["abs_pct_error"] for r in rows]
        plt.scatter(xs, ys, s=8, alpha=0.6, label=label)
    plt.xscale("log" if group == "stride" else "linear")
    plt.xlabel(group)
    plt.ylabel("absolute error (%)")
    plt.title(f"Template strided error vs {group}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"error_vs_{group}.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    out_dir = args.dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        ("load", read_errors(args.dir / "errors_load.csv")),
        ("store", read_errors(args.dir / "errors_store.csv")),
    ]
    plot_pair(lines, out_dir)
    plot_error_hist(lines, out_dir)
    plot_error_vs(lines, "stride", out_dir)
    plot_error_vs(lines, "block", out_dir)
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
