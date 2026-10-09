#!/usr/bin/env python3
"""Generate parity and error plots for the strided-store model v1."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/model_store_v1"
PLOTS = RES / "plots"


def load_errors(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["target_ns"] = float(r["target_ns"])
        r["pred_ns"] = float(r["pred_ns"])
        r["relative_error"] = float(r["relative_error"])
        r["block"] = int(r["block"])
        r["stride"] = int(r["stride"])
        r["num_warps"] = int(r["num_warps"])
    return rows


def scatter_style(branch):
    return {"simd_wide": ("tab:green", "o"),
            "simd_gather": ("tab:blue", "o"),
            "simt": ("tab:red", "o")}[branch]


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = load_errors(RES / "errors_all.csv")
    branches = sorted({r["branch"] for r in rows})
    order = [b for b in ("simd_wide", "simd_gather", "simt") if b in branches]

    # 1) log-log parity: one panel per branch.
    fig, axes = plt.subplots(1, len(order), figsize=(14, 4.6), squeeze=False)
    for ax, branch in zip(axes[0], order):
        sub = [r for r in rows if r["branch"] == branch]
        color, _ = scatter_style(branch)
        target = [r["target_ns"] for r in sub]
        pred = [r["pred_ns"] for r in sub]
        ax.scatter(target, pred, s=14, alpha=0.65, c=color, edgecolors="none")
        lo = max(1e-3, min(target + pred) * 0.8)
        hi = max(target + pred) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("target (ns/iteration, log)")
        ax.set_ylabel("prediction (ns/iteration, log)")
        err = [abs(r["relative_error"]) for r in sub]
        ax.set_title(f"{branch}\nn={len(sub)}  MAPE="
                     f"{sum(err) / len(err) * 100.0:.2f}%")
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("Strided store: measured vs predicted (log-log)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "store_parity_loglog.png", dpi=160)
    plt.close(fig)

    # 2) signed relative error histogram: one panel per branch.
    order = [b for b in ("simd_wide", "simd_gather", "simt") if b in branches]
    fig, axes = plt.subplots(1, len(order), figsize=(14, 4.4), squeeze=False)
    for ax, branch in zip(axes[0], order):
        sub = [r["relative_error"] * 100.0 for r in rows if r["branch"] == branch]
        color, _ = scatter_style(branch)
        bins = 30
        if sub and min(sub) != max(sub):
            bins = np.linspace(min(sub), max(sub), 30)
        ax.hist(sub, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        abs_err = [abs(v) for v in sub]
        mape = sum(abs_err) / len(abs_err) if abs_err else 0.0
        p50 = float(np.percentile(abs_err, 50)) if abs_err else 0.0
        mx = max(abs_err) if abs_err else 0.0
        ax.set_title(f"{branch}\nn={len(sub)}  MAPE={mape:.2f}%  "
                     f"p50={p50:.2f}%  max={mx:.2f}%")
        ax.set_xlabel("signed relative error: pred / target - 1 (%)")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("Strided store: signed relative-error distribution", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "store_error_hist.png", dpi=160)
    plt.close(fig)

    # 3) absolute error vs stride: one panel per branch.
    fig, axes = plt.subplots(1, len(order), figsize=(14, 4.6), squeeze=False)
    for ax, branch in zip(axes[0], order):
        sub = [r for r in rows if r["branch"] == branch]
        color, _ = scatter_style(branch)
        ax.scatter([r["stride"] for r in sub],
                   [abs(r["relative_error"]) * 100.0 for r in sub],
                   s=14, alpha=0.65, c=color, edgecolors="none")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stride (elements, log2)")
        ax.set_ylabel("absolute relative error (%)")
        err = [abs(r["relative_error"]) for r in sub]
        ax.set_title(f"{branch}\nn={len(sub)}  MAPE="
                     f"{sum(err) / len(err) * 100.0:.2f}%")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Strided store: error vs stride", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "store_error_vs_stride.png", dpi=160)
    plt.close(fig)

    # 4) absolute error vs target: one panel per branch.
    fig, axes = plt.subplots(1, len(order), figsize=(14, 4.6), squeeze=False)
    for ax, branch in zip(axes[0], order):
        sub = [r for r in rows if r["branch"] == branch]
        color, _ = scatter_style(branch)
        ax.scatter([r["target_ns"] for r in sub],
                   [abs(r["relative_error"]) * 100.0 for r in sub],
                   s=14, alpha=0.65, c=color, edgecolors="none")
        ax.set_xscale("log")
        ax.set_xlabel("target (ns/iteration, log)")
        ax.set_ylabel("absolute relative error (%)")
        err = [abs(r["relative_error"]) for r in sub]
        ax.set_title(f"{branch}\nn={len(sub)}  MAPE="
                     f"{sum(err) / len(err) * 100.0:.2f}%")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Strided store: error vs target", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOTS / "store_error_vs_target.png", dpi=160)
    plt.close(fig)
    print(f"wrote 4 plots under {PLOTS}")


if __name__ == "__main__":
    main()
