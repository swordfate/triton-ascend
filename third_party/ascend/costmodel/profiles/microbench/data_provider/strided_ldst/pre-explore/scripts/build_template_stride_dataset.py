#!/usr/bin/env python3
"""Build the 1D rank1 template load/store dataset from raw probe JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from template_stride_features import FEATURE_ORDER, feature_vector

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "results/model_template_stride_v1/raw"
DEFAULT_LOAD = [
    RAW / "board_template_stride_load.json",
    RAW / "board_template_stride_load_pass2.json",
    RAW / "board_template_stride_load_rerun_outliers.json",
    RAW / "board_template_stride_load_rerun2_outliers.json",
]
DEFAULT_STORE = [
    RAW / "board_template_stride_store.json",
    RAW / "board_template_stride_store_pass2.json",
    RAW / "board_template_stride_store_rerun_outliers.json",
]
DEFAULT_OUT = HERE / "results/model_template_stride_v1/dataset.csv"


def read_path(path: Path, expected: str):
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if payload.get("path") != expected:
        raise ValueError(f"{path}: expected {expected}, got {payload.get('path')}")
    return [(path, case) for case in payload.get("cases", [])]


def build_rows(load_paths, store_paths):
    raw_cases = []
    for path in load_paths:
        raw_cases += read_path(path, "triton_stride_load")
    for path in store_paths:
        raw_cases += read_path(path, "triton_stride_store")
    grouped = {}
    for source, case in raw_cases:
        key = (case["path"], int(case["block"]), int(case["stride"]),
               int(case["num_warps"]))
        grouped.setdefault(key, []).append((source, case))
    rows = []
    for (path, block, stride, num_warps), candidates in sorted(grouped.items()):
        valid = [
            (s, c) for s, c in candidates
            if c.get("valid") and c.get("correctness_ok")
        ]
        pool = valid if valid else candidates
        source, chosen = min(pool, key=lambda sc: float(sc[1]["target_ns"]))
        features = feature_vector(block, stride, num_warps)
        row = {
            "row_id": len(rows),
            "path": path,
            "source_file": source.name,
            "case_index": len(rows),
            "block": int(block),
            "stride": int(stride),
            "num_warps": int(num_warps),
            "target_ns": float(chosen["target_ns"]),
            "target_kind": "min_across_probe_passes",
            "valid": int(bool(chosen.get("valid", False)
                              and chosen.get("correctness_ok", False))),
            "correctness_ok": int(bool(chosen.get("correctness_ok", False))),
            "n_candidates": len(candidates),
            "n_valid_candidates": len(valid),
        }
        for name, value in zip(FEATURE_ORDER, features):
            row[f"f_{name}"] = float(value)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-json", nargs="+", type=Path, default=DEFAULT_LOAD)
    ap.add_argument("--store-json", nargs="+", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = build_rows(args.load_json, args.store_json)
    fields = (
        ["row_id", "path", "source_file", "case_index", "block", "stride",
         "num_warps", "target_ns", "target_kind", "valid", "correctness_ok",
         "n_candidates", "n_valid_candidates"]
        + [f"f_{name}" for name in FEATURE_ORDER]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
