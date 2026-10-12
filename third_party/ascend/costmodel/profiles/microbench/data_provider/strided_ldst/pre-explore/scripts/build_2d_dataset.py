#!/usr/bin/env python3
"""Build the Phase B (rank-2 strided) dataset from raw 2D probe JSON.

Scope, as clarified by the task owner:

  * a rank-2 tile counts as 2D-strided only when BOTH axes carry holes
    (inner_stride >= 2 AND outer_stride > (C-1)*inner_stride + 1);
  * a rank-2 tile whose ROW dimension is 1 is still a real lowering shape and
    MUST be modelled -- it is classified as singleton_row_strided and kept.

Rules mirror build_dtype_dataset.py: valid = witness AND frequency AND spread
plus numeric correctness; the delivered target per case is the minimum over
valid passes; every feature comes from stride_feature_reference.py.

Usage:
    python3 scripts/build_2d_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import stride_feature_reference as sfr  # noqa: E402

DEFAULT_RAW = HERE / "results/dtype_calibration_v1/raw"
DEFAULT_OUT = HERE / "results/dtype_calibration_v1"

FEATURE_COLUMNS = [
    "f_elements", "f_L", "f_line_elems_max", "f_line_elems_min", "f_line_density",
    "f_mean_warp_lines", "f_min_warp_lines", "f_max_warp_lines",
    "f_bucket_pairs_2k8", "f_bucket_worst_2k8",
    "f_bucket_pairs_32k16", "f_bucket_worst_32k16",
    "f_span_bytes", "f_page_cross", "f_cross",
    "f_inner_stride_bytes", "f_row_gap_bytes", "f_row_gap_elements",
    "f_tail_elems_2d", "f_iters_per_thread_2d", "f_K_2d",
    "f_line_request_size_2d", "f_tile_bytes",
]


def feature_row(case):
    R, C = int(case["R"]), int(case["C"])
    OS, IS = int(case["OS"]), int(case["IS"])
    eb = int(case["element_bytes"])
    warps = int(case["num_warps"])
    f = sfr.tile_features(R, C, OS, IS, eb)
    elements = R * C
    return {
        "f_elements": f["elements"],
        "f_L": f["L"],
        "f_line_elems_max": f["line_elems_max"],
        "f_line_elems_min": f["line_elems_min"],
        "f_line_density": f["line_density"],
        "f_mean_warp_lines": f["mean_warp_lines"],
        "f_min_warp_lines": f["min_warp_lines"],
        "f_max_warp_lines": f["max_warp_lines"],
        "f_bucket_pairs_2k8": f["bucket_pairs_2k8"],
        "f_bucket_worst_2k8": f["bucket_worst_2k8"],
        "f_bucket_pairs_32k16": f["bucket_pairs_32k16"],
        "f_bucket_worst_32k16": f["bucket_worst_32k16"],
        "f_span_bytes": f["span_bytes"],
        "f_page_cross": f["page_cross"],
        "f_cross": f["cross"],
        "f_inner_stride_bytes": f["inner_stride_bytes"],
        "f_row_gap_bytes": f["row_gap_bytes"],
        "f_row_gap_elements": f["row_gap_elements"],
        "f_tail_elems_2d": float(elements % sfr.TEMPLATE_THREADS),
        "f_iters_per_thread_2d": float(
            -(-elements // sfr.TEMPLATE_THREADS)),
        "f_K_2d": float(max(1, warps) * max(
            1.0, -(-elements // (sfr.WARP * max(1, warps))))),
        "f_line_request_size_2d": f["line_request_size"],
        "f_tile_bytes": f["tile_bytes"],
    }


def invalid_reason(case):
    if not case.get("correctness_ok", False):
        return "correctness"
    if case.get("error"):
        return "compile_or_assert"
    attempts = case.get("attempts") or []
    if not attempts:
        return "no_attempt"
    if all(not a["witness_ok"] for a in attempts):
        return "witness"
    if all(not a["frequency_ok"] for a in attempts):
        return "frequency"
    if all(not a["stable_ok"] for a in attempts):
        return "spread"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-contig", action="store_true",
                    help="keep the fully contiguous control rows (never fitted)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    order = []
    sources = {}
    for path in sorted(args.raw_dir.glob("*2d_*.json")):
        payload = json.loads(path.read_text())
        for case in payload.get("cases", []):
            if "R" not in case:
                continue
            key = (case.get("op"), case.get("path"), case.get("dtype"),
                   case.get("form"), int(case["R"]), int(case["C"]),
                   int(case["OS"]), int(case["IS"]), int(case["num_warps"]))
            bucket = grouped.setdefault(key, {"good": [], "all": []})
            bucket["all"].append(case)
            if case.get("valid") and case.get("correctness_ok"):
                bucket["good"].append(case)
                sources.setdefault(key, set()).add(path.name)
            if key not in order:
                order.append(key)

    rows = []
    invalid_counts = {}
    total = valid = 0
    for key in order:
        op, path_name, dtype_name, form, R, C, OS, IS, warps = key
        bucket = grouped[key]
        total += 1
        # Same policy as build_dtype_dataset.py: keep every numerically
        # correct point; a high spread is a quality flag, not a reason to drop
        # it (the probe already reports the median over attempts).
        cands = [c for c in bucket["all"] if c.get("correctness_ok")]
        if not cands:
            reason = invalid_reason(bucket["all"][0])
            tag = (op, path_name, form)
            invalid_counts.setdefault(tag, {})
            invalid_counts[tag][reason] = invalid_counts[tag].get(reason, 0) + 1
            continue
        good = [c for c in cands if c.get("valid")]
        if good:
            valid += 1
        best = min(good or cands, key=lambda c: c["target_ns"])
        targets = sorted(float(c["target_ns"]) for c in cands)
        nn = len(targets)
        median = (targets[nn // 2] if nn % 2 else
                  0.5 * (targets[nn // 2 - 1] + targets[nn // 2]))
        spreads = [a.get("spread") for c in cands
                   for a in (c.get("attempts") or [])
                   if a.get("spread") is not None]
        row = {
            "op": op, "path": path_name, "dtype": dtype_name,
            "element_bytes": int(best["element_bytes"]),
            "form": form, "axis_class": best.get("axis_class", ""),
            "R": R, "C": C, "OS": OS, "IS": IS,
            "elements": R * C,
            "contiguous": int(bool(best.get("contiguous"))),
            "num_warps": warps,
            "target_ns": float(median),
            "target_min_ns": float(targets[0]),
            "target_max_ns": float(targets[-1]),
            "target_spread": float(max(spreads)) if spreads else 0.0,
            "quality": "ok" if good else invalid_reason(cands[0]),
            "target_kind": best.get("target_kind", ""),
            "valid": 1 if good else 0, "correctness_ok": 1,
            "n_passes": len(bucket["all"]),
            "n_valid_passes": len(good),
            "contains_template_call": int(bool(best.get("contains_template_call"))),
            "source_files": ";".join(sorted(sources.get(key, []))),
        }
        row.update(feature_row(best))
        rows.append(row)

    if not args.include_contig:
        rows = [r for r in rows if not r["contiguous"]]

    columns = ["op", "path", "dtype", "element_bytes", "form", "axis_class",
               "R", "C", "OS", "IS", "elements", "contiguous", "num_warps",
               "target_ns", "target_min_ns", "target_max_ns", "target_spread",
               "quality", "target_kind", "valid", "correctness_ok",
               "n_passes", "n_valid_passes", "contains_template_call",
               "source_files"] + FEATURE_COLUMNS
    with (args.out_dir / "dataset_2d.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    coverage = {}
    for row in rows:
        tag = (row["op"], row["path"], row["axis_class"])
        c = coverage.setdefault(tag, {"n": 0, "forms": set(), "shapes": set()})
        c["n"] += 1
        c["forms"].add(row["form"])
        c["shapes"].add(f"{row['R']}x{row['C']}")
    meta = {
        "kind": "stride2d_dataset",
        "raw_files": sorted(p.name for p in args.raw_dir.glob("*2d_*.json")),
        "total_cases": total, "valid_cases": valid,
        "invalid_cases": total - valid,
        "invalid_by_bucket": {"|".join(k): v for k, v in
                              sorted(invalid_counts.items(), key=str)},
        "scope": ("rank-2 tiles with holes on BOTH axes; R == 1 rank-2 tiles "
                  "are included as singleton_row_strided"),
        "feature_source": "scripts/stride_feature_reference.py",
        "coverage": {"|".join(k): {"n": v["n"], "forms": sorted(v["forms"]),
                                   "shapes": sorted(v["shapes"])}
                     for k, v in sorted(coverage.items(), key=str)},
    }
    (args.out_dir / "dataset_2d_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} (valid {valid}/{total}) -> {args.out_dir / 'dataset_2d.csv'}")
    for tag, value in sorted(coverage.items(), key=str):
        print(f"  {tag}: n={value['n']} forms={sorted(value['forms'])} "
              f"shapes={sorted(value['shapes'])}")


if __name__ == "__main__":
    main()
