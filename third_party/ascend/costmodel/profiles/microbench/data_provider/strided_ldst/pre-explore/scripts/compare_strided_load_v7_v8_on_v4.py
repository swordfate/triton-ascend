#!/usr/bin/env python3
"""Build a compact v7-vs-v8 grouped error comparison on the v4 dataset."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V7 = ROOT / "results/model_v4_numwarps/v7_eval/v7_errors_all.csv"
DEFAULT_V8 = ROOT / "results/model_v4_simple/errors_all.csv"
DEFAULT_OUT = ROOT / "results/model_v4_simple/v7_vs_v8_group_metrics.csv"


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metrics(targets, preds):
    target = np.asarray(targets, dtype=float)
    pred = np.asarray(preds, dtype=float)
    rel = pred / target - 1.0
    ae = np.abs(rel) * 100.0
    return {
        "n": int(len(ae)),
        "mape_pct": float(ae.mean()),
        "p50_pct": float(np.percentile(ae, 50)),
        "p90_pct": float(np.percentile(ae, 90)),
        "max_pct": float(ae.max()),
        "bias_pct": float(rel.mean() * 100.0),
        "rmse_pct": float(np.sqrt(np.mean(rel ** 2)) * 100.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", type=Path, default=DEFAULT_V7)
    ap.add_argument("--v8", type=Path, default=DEFAULT_V8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    v7 = read_rows(args.v7)
    v8 = read_rows(args.v8)
    key_fields = ("num_warps", "block", "stride")
    index = {}
    for row in v8:
        index[(int(row["num_warps"]), int(row["block"]), int(row["stride"]))] = row
    if len(v7) != len(v8):
        raise SystemExit(f"row count mismatch: v7={len(v7)} v8={len(v8)}")

    rows = []
    for group_key in ("overall", "block", "num_warps", "stride"):
        if group_key == "overall":
            groups = [None]
        elif group_key == "block":
            groups = sorted({int(r["block"]) for r in v8})
        elif group_key == "num_warps":
            groups = sorted({int(r["num_warps"]) for r in v8})
        else:
            groups = sorted({int(r["stride"]) for r in v8})
        for group in groups:
            selected_v7 = []
            selected_v8 = []
            for row7 in v7:
                key = (int(row7["num_warps"]), int(row7["block"]),
                       int(row7["stride"]))
                if group_key != "overall" and str(row7[group_key]) != str(group):
                    continue
                row8 = index[key]
                selected_v7.append((float(row7["target_ns"]),
                                    float(row7["pred_ns"])))
                selected_v8.append((float(row8["target_ns"]),
                                    float(row8["pred_ns"])))
            if not selected_v7:
                continue
            m7 = metrics([x[0] for x in selected_v7],
                         [x[1] for x in selected_v7])
            m8 = metrics([x[0] for x in selected_v8],
                         [x[1] for x in selected_v8])
            rows.append({
                "group_key": group_key,
                "group": "" if group is None else group,
                **{f"v7_{k}": v for k, v in m7.items()},
                **{f"v8_{k}": v for k, v in m8.items()},
                "mape_reduction_pct_points": m7["mape_pct"] - m8["mape_pct"],
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")
    for row in rows:
        if row["group_key"] == "overall":
            print("overall: "
                  f"v7 MAPE={row['v7_mape_pct']:.2f}% "
                  f"v8 MAPE={row['v8_mape_pct']:.2f}% "
                  f"reduction={row['mape_reduction_pct_points']:.2f} pp")


if __name__ == "__main__":
    main()
