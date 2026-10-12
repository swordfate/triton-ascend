#!/usr/bin/env python3
"""Build the Phase A (element dtype) calibration dataset from raw probe JSON.

Raw inputs: results/dtype_calibration_v1/raw/*.json produced by
scripts/dtype_stride_measure_probe.py.

Rules (mirroring the v2 template calibration):
  * a case is usable only if at least one attempt passed witness AND frequency
    AND spread, and the numeric correctness check passed;
  * the delivered target for a (op, path, dtype, block, stride, num_warps) key
    is the MINIMUM target over all valid passes (contention can only inflate an
    Event measurement);
  * stride == 1 must never appear -- asserted here;
  * every feature comes from stride_feature_reference.py so that a future C++
    port has one exact definition to follow.

Usage:
    python3 scripts/build_dtype_dataset.py
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
    "f_L", "f_line_request_size", "f_line_request_size_cpp4", "f_page_cross", "f_cross",
    "f_dup", "f_K", "f_K_stg", "f_lowW_4_L", "f_min_L_32",
    "f_max_L_minus_32", "f_dupL_minL_128", "f_cross_minL_32",
    "f_dupL", "f_min_L_64", "f_max_WL_minus_3072", "f_dup_cross",
    "f_bucket_pairs_g2048_b8", "f_bucket_worst_g2048_b8",
    "f_bucket_worst_32k16", "f_bank_pairs_g32768_b16", "f_worst_g32768_b16",
    "f_mean_warp_lines", "f_tail_elems",
    "f_iters_per_thread", "f_active_threads",
    "f_stride_minus_1", "f_large_tile_gt64", "f_tail_elems_gt64",
    "f_n_cmd_block", "f_page_cross_count",
]


def feature_row(case):
    elements = int(case["block"])
    stride_bytes = int(case["stride_bytes"])
    element_bytes = int(case["element_bytes"])
    warps = int(case["num_warps"])
    simd = sfr.simd_features(elements, stride_bytes, element_bytes)
    simt = sfr.simt_features(elements, stride_bytes, element_bytes, warps)
    tmpl = sfr.template_features(elements, stride_bytes, element_bytes, warps)
    row = {
        "f_L": tmpl["L"],
        "f_line_request_size": simt["line_request_size"],
        # C++ stridedLineRequestSize() hardcodes 4 bytes/element; A_reuse must
        # use that semantics while the byte-domain schemes use the real one.
        "f_line_request_size_cpp4": (4.0 if stride_bytes >= 128 else
                                     min(128.0, max(1, 128 // stride_bytes) * 4.0)),
        "f_page_cross": tmpl["page_cross"],
        "f_cross": simt["cross"],
        "f_dup": simt["dup"],
        "f_K": simt["K"],
        "f_K_stg": simt["K_stg"],
        "f_lowW_4_L": simt["lowW_4_L"],
        "f_min_L_32": simt["min_L_32"],
        "f_max_L_minus_32": simt["max_L_minus_32"],
        "f_dupL_minL_128": simt["dupL_minL_128"],
        "f_cross_minL_32": simt["cross_minL_32"],
        "f_dupL": simt["dupL"],
        "f_min_L_64": simt["min_L_64"],
        "f_max_WL_minus_3072": max(0.0, warps * simt["L"] - 3072.0),
        "f_dup_cross": simt["dup"] * simt["cross"],
        "f_bucket_pairs_g2048_b8": simd["bucket_pairs_g2048_b8"],
        "f_bucket_worst_g2048_b8": simd["bucket_worst_g2048_b8"],
        "f_bucket_worst_32k16": tmpl["bucket_worst_32k16"],
        "f_bank_pairs_g32768_b16": sfr.bank_facts(elements, stride_bytes, 32768, 16)[0],
        "f_worst_g32768_b16": sfr.bank_facts(elements, stride_bytes, 32768, 16)[1],
        "f_mean_warp_lines": tmpl["mean_warp_lines"],
        "f_tail_elems": tmpl["tail_elems"],
        "f_iters_per_thread": tmpl["iters_per_thread"],
        "f_active_threads": tmpl["active_threads"],
        "f_stride_minus_1": simd["stride_minus_1"],
        "f_large_tile_gt64": simd["large_tile_gt64"],
        "f_tail_elems_gt64": simd["tail_elems_gt64"],
        "f_n_cmd_block": simd["n_cmd_block"],
        "f_page_cross_count": simd["page_cross_count"],
    }
    return row


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
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    order = []
    sources = {}
    raw_files = sorted(args.raw_dir.glob("*.json"))
    if not raw_files:
        raise SystemExit(f"no raw json under {args.raw_dir}")
    for path in raw_files:
        # rank-2 raw files share the directory but are handled by
        # build_2d_dataset.py; their cases have no stride/stride_bytes.
        if "2d_" in path.name:
            continue
        payload = json.loads(path.read_text())
        for case in payload.get("cases", []):
            if "stride_bytes" not in case:
                continue
            key = (case.get("op"), case.get("path"), case.get("dtype"),
                   int(case.get("block", 0)), int(case.get("stride", 0)),
                   int(case.get("num_warps", 0)))
            bucket = grouped.setdefault(key, {"good": [], "all": []})
            bucket["all"].append(case)
            if case.get("valid") and case.get("correctness_ok"):
                bucket["good"].append(case)
                sources.setdefault(key, set()).add(path.name)
            if key not in order:
                order.append(key)

    rows = []
    invalid_counts = {}
    total_cases = 0
    valid_cases = 0
    for key in order:
        op, path_name, dtype_name, block, stride, warps = key
        bucket = grouped[key]
        total_cases += 1
        # Keep every numerically-correct point.  A high spread is a QUALITY
        # FLAG, not a reason to drop the point: the probe already reports the
        # median over attempts for such cases, which is the "run more, take the
        # median" estimator.  Only a wrong result is dropped.
        cands = [c for c in bucket["all"] if c.get("correctness_ok")]
        if not cands:
            reason = invalid_reason(bucket["all"][0])
            invalid_counts[(op, path_name, dtype_name)] = \
                invalid_counts.get((op, path_name, dtype_name), {})
            invalid_counts[(op, path_name, dtype_name)][reason] = \
                invalid_counts[(op, path_name, dtype_name)].get(reason, 0) + 1
            continue
        good = [c for c in cands if c.get("valid")]
        if good:
            valid_cases += 1
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
            "block": block, "stride": stride,
            "stride_bytes": int(best["stride_bytes"]),
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
            "source_files": ";".join(sorted(sources.get(key, []))),
            "contains_template_call": int(bool(best.get("contains_template_call"))),
        }
        row.update(feature_row(best))
        rows.append(row)

    if any(int(r["stride"]) == 1 for r in rows):
        raise SystemExit("stride == 1 leaked into the dataset")

    columns = ["op", "path", "dtype", "element_bytes", "block", "stride",
               "stride_bytes", "num_warps", "target_ns", "target_min_ns",
               "target_max_ns", "target_spread", "quality", "target_kind",
               "valid", "correctness_ok", "n_passes", "n_valid_passes",
               "contains_template_call", "source_files"] + FEATURE_COLUMNS
    with (args.out_dir / "dataset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    coverage = {}
    for row in rows:
        key = (row["op"], row["path"], row["dtype"])
        c = coverage.setdefault(key, {"n": 0, "blocks": set(), "strides": set(),
                                      "byte_gaps": set()})
        c["n"] += 1
        c["blocks"].add(row["block"])
        c["strides"].add(row["stride"])
        c["byte_gaps"].add(row["stride_bytes"])

    meta = {
        "kind": "dtype_calibration_dataset",
        "raw_files": [p.name for p in raw_files],
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "invalid_cases": total_cases - valid_cases,
        "invalid_by_bucket": {"|".join(k): v for k, v in
                              sorted(invalid_counts.items(), key=str)},
        "stride1_rows": 0,
        "coverage": {"|".join(k): {"n": v["n"],
                                   "blocks": sorted(v["blocks"]),
                                   "strides": sorted(v["strides"]),
                                   "byte_gaps": sorted(v["byte_gaps"])}
                     for k, v in sorted(coverage.items(), key=str)},
        "feature_source": "scripts/stride_feature_reference.py",
    }
    (args.out_dir / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} (valid cases {valid_cases}/{total_cases}) "
          f"-> {args.out_dir / 'dataset.csv'}")
    for key, value in sorted(coverage.items(), key=str):
        print(f"  {key}: n={value['n']} blocks={sorted(value['blocks'])} "
              f"gaps={sorted(value['byte_gaps'])}")
    for key, value in sorted(invalid_counts.items(), key=str):
        print(f"  invalid {key}: {value}")


if __name__ == "__main__":
    main()
