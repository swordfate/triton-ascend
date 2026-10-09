#!/usr/bin/env python3
"""Generate parity / residual plots for the final v7-simple strided-load model.

Input:
  results/model_v7_simple/errors_all.csv

Output:
  results/model_v7_simple/plots/
    v7_parity.png                 target vs prediction, log-log, per group
    v7_error_hist.png             signed relative-error histogram, per group
    v7_error_vs_stride.png        signed error vs stride, colored by block/W
    v7_error_vs_target.png        signed error vs target, colored by block/W

The plots consume the same errors_all.csv that the predictor verification uses,
so they cannot silently drift away from the committed model output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRE_EXPLORE = SCRIPT_DIR.parent
DEFAULT_ERRORS = PRE_EXPLORE / "results/model_v7_simple/errors_all.csv"
DEFAULT_OUT = PRE_EXPLORE / "results/model_v7_simple/plots"

GROUP_TITLES = {
    "simd_wide": "SIMD wide (stride=1/2)",
    "simd_gather": "SIMD gather (stride>=3)",
    "simt": "SIMT (num_warps=1..64)",
}
GROUP_ORDER = ["simd_wide", "simd_gather", "simt"]


def load_errors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "target_ns", "pred_ns", "stride", "block"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    df = df[df["group"].isin(GROUP_ORDER)].copy()
    if df.empty:
        raise SystemExit(f"no rows for {GROUP_ORDER} in {path}")
    return df


def groups(df: pd.DataFrame):
    for group in GROUP_ORDER:
        sub = df[df["group"] == group].copy()
        if not sub.empty:
            yield group, sub


def group_title(group: str, sub: pd.DataFrame) -> str:
    err = sub["error_pct"].to_numpy(dtype=float)
    return (
        f"{GROUP_TITLES[group]}\n"
        f"n={len(sub)}  MAPE={np.mean(np.abs(err)):.2f}%  "
        f"p50={np.percentile(np.abs(err), 50):.2f}%  "
        f"max={np.max(np.abs(err)):.2f}%"
    )


def color_values(sub: pd.DataFrame):
    """Color by block for SIMD, by num_warps for SIMT."""
    if sub["group"].iloc[0] == "simt" and "num_warps" in sub.columns:
        values = sub["num_warps"].to_numpy(dtype=float)
        label = "num_warps"
    else:
        values = sub["block"].to_numpy(dtype=float)
        label = "block"
    return values, label


def plot_parity(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (group, sub) in zip(axes, groups(df)):
        target = sub["target_ns"].to_numpy(dtype=float)
        pred = sub["pred_ns"].to_numpy(dtype=float)
        ax.scatter(target, pred, s=12, alpha=0.45, edgecolors="none")
        lo = max(1e-3, min(target.min(), pred.min()) * 0.8)
        hi = max(target.max(), pred.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("measured target (ns; ns/iter for SIMT)")
        ax.set_ylabel("prediction")
        ax.set_title(group_title(group, sub))
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("v7-simple: measured vs predicted", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "v7_parity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", path)


def plot_error_hist(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (group, sub) in zip(axes, groups(df)):
        err = sub["error_pct"].to_numpy(dtype=float)
        bins = np.linspace(err.min(), err.max(), 40)
        ax.hist(err, bins=bins, color="#4c72b0", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_xlabel("signed relative error: pred / target - 1 (%)")
        ax.set_ylabel("count")
        ax.set_title(group_title(group, sub))
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("v7-simple: signed relative-error distribution", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "v7_error_hist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", path)


def plot_error_scatter(df: pd.DataFrame, x_col: str, out_path: Path,
                       xlabel: str, title: str, log_x: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (group, sub) in zip(axes, groups(df)):
        cvalues, clabel = color_values(sub)
        norm = Normalize(vmin=np.log2(max(1.0, cvalues.min())),
                         vmax=np.log2(max(1.0, cvalues.max())))
        x = sub[x_col].to_numpy(dtype=float)
        y = sub["error_pct"].to_numpy(dtype=float)
        scatter = ax.scatter(x, y, c=np.log2(np.maximum(cvalues, 1.0)),
                             cmap="viridis", norm=norm, s=16, alpha=0.65,
                             edgecolors="none")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("signed relative error (%)")
        ax.set_title(group_title(group, sub))
        ax.grid(True, alpha=0.2)
        cb = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"log2({clabel})")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_errors(args.errors)
    plot_parity(df, args.out_dir)
    plot_error_hist(df, args.out_dir)
    plot_error_scatter(
        df, "stride", args.out_dir / "v7_error_vs_stride.png",
        "stride (elements)", "v7-simple: signed error vs stride",
        log_x=True)
    plot_error_scatter(
        df, "target_ns", args.out_dir / "v7_error_vs_target.png",
        "measured target (ns)", "v7-simple: signed error vs measured target",
        log_x=False)


if __name__ == "__main__":
    main()
