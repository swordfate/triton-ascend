#!/usr/bin/env python3
"""Analyze residual between measured route latencies and C++ costmodel scores.

Input: JSONL produced by run_triton_benchmark.py.  Each line must contain at
least ``case``, ``route`` and ``latency_ms``.  The ``simd_simt_report`` route
line should also contain ``report_json`` (captured from the native pass).

Output:
  1. per-case table: measured latency(s), predicted raw candidate costs,
     measured simd/simt ratio vs predicted raw ratio;
  2. component-share table for every case;
  3. suspicious-component ranking: for each scoring component, how strongly its
     case-to-case variation explains the ratio residual.  The top components
     are the ones whose underlying cce rate should be re-measured first.

Usage:
    python bench/simt_autoscope/analyze_residuals.py \
        --input results/simt_autoscope_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"skip bad json line: {exc}", file=sys.stderr)
    return rows


def pick(rows, case, route):
    for row in rows:
        if row.get("case") == case and row.get("route") == route:
            return row
    return None


def num(obj, *path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def safe_div(a, b):
    if b is None or a is None or abs(b) < 1e-12:
        return None
    return a / b


def pearson(xs, ys):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    ys = [y for y in ys if y is not None and not math.isnan(y)]
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx < 1e-12 or vy < 1e-12:
        return None
    return cov / math.sqrt(vx * vy)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/simt_autoscope_bench.jsonl")
    ap.add_argument("--min-samples", type=int, default=3,
                    help="minimum cases needed before ranking components")
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit(f"no valid rows in {args.input}")

    cases = sorted({row.get("case") for row in rows if row.get("case")})

    # NOTE: SimdSimtCostReport::toJSON flattens the breakdown into the
    # report top-level, e.g. report["memory"], report["simt_execution"].
    components = [
        ("simd_memory_share", "simd memory roofline share",
         lambda r: safe_div(
             num(r, "memory", "simd_roofline_system_cycles"),
             num(r, "simt_execution", "simd_issue_payload_system_cycles"))),
        ("simd_dot_share", "simd dot share",
         lambda r: safe_div(
             num(r, "compute_only", "simd_dot"),
             num(r, "simt_execution", "simd_issue_payload_system_cycles"))),
        ("simd_penalty_share", "simd structural penalty share",
         lambda r: safe_div(
             num(r, "structure", "simd_structural_penalty_system_cycles"),
             num(r, "analytical_candidate_costs", "all_simd"))),
        ("simt_compute_share", "simt compute share",
         lambda r: safe_div(
             num(r, "compute_only", "simt"),
             num(r, "simt_execution", "simt_issue_payload_system_cycles"))),
        ("simt_dot_share", "simt dot share",
         lambda r: safe_div(
             num(r, "compute_only", "simt_dot"),
             num(r, "simt_execution", "simt_issue_payload_system_cycles"))),
        ("simt_memory_share", "simt memory share",
         lambda r: safe_div(
             num(r, "memory", "simt_roofline_system_cycles"),
             num(r, "simt_execution", "simt_issue_payload_system_cycles"))),
        ("simt_shuffle_share", "simt shuffle share",
         lambda r: safe_div(
             num(r, "simt_execution", "shuffle_system_cycles"),
             num(r, "simt_execution", "simt_issue_payload_system_cycles"))),
        ("simt_predicate_share", "simt predicate share",
         lambda r: safe_div(
             num(r, "simt_execution", "predicate_system_cycles"),
             num(r, "simt_execution", "simt_issue_payload_system_cycles"))),
    ]

    print("=" * 100)
    print("PER-CASE: measured latency vs model ratio")
    print("=" * 100)
    header = (f"{'case':28s} {'simd_ms':>10s} {'simt_ms':>10s} {'report_ms':>10s} "
              f"{'meas_ratio':>11s} {'pred_raw_ratio':>15s} {'pred_cal_ratio':>15s}")
    print(header)

    case_data = []
    for case in cases:
        simd_row = pick(rows, case, "simd")
        simt_row = pick(rows, case, "simt_only")
        rep_row = pick(rows, case, "simd_simt_report")
        if not simd_row or not simt_row:
            print(f"{case:28s}  missing simd/simt_only rows")
            continue
        simd_ms = simd_row.get("latency_ms")
        simt_ms = simt_row.get("latency_ms")
        report_ms = rep_row.get("latency_ms") if rep_row else None
        meas_ratio = safe_div(simd_ms, simt_ms)

        report = (rep_row or {}).get("report_json") or {}
        raw_simd = num(report, "event_route_calibration", "raw_candidate_costs", "all_simd")
        raw_simt = num(report, "event_route_calibration", "raw_candidate_costs", "all_simt_only")
        cal_simd = num(report, "candidate_costs", "all_simd")
        cal_simt = num(report, "candidate_costs", "all_simt_only")
        pred_raw_ratio = safe_div(raw_simd, raw_simt)
        pred_cal_ratio = safe_div(cal_simd, cal_simt)

        print(f"{case:28s} {simd_ms:10.4f} {simt_ms:10.4f} {report_ms if report_ms is not None else float('nan'):10.4f} "
              f"{meas_ratio if meas_ratio is not None else float('nan'):11.3f} "
              f"{pred_raw_ratio if pred_raw_ratio is not None else float('nan'):15.3f} "
              f"{pred_cal_ratio if pred_cal_ratio is not None else float('nan'):15.3f}")

        case_data.append({
            "case": case,
            "report": report,
            "meas_ratio": meas_ratio,
            "pred_raw_ratio": pred_raw_ratio,
            "pred_cal_ratio": pred_cal_ratio,
            "delta_raw": (pred_raw_ratio - meas_ratio) if pred_raw_ratio is not None and meas_ratio is not None else None,
            "delta_cal": (pred_cal_ratio - meas_ratio) if pred_cal_ratio is not None and meas_ratio is not None else None,
        })

    print()
    print("=" * 100)
    print("COMPONENT SHARES")
    print("=" * 100)
    comp_header = f"{'case':28s} " + " ".join(f"{name:>18s}" for name, _, _ in components)
    print(comp_header)
    for item in case_data:
        report = item["report"]
        values = []
        for name, _, fn in components:
            v = fn(report) if report else None
            values.append(v)
        print(f"{item['case']:28s} " + " ".join(
            (f"{v:18.3f}" if v is not None else f"{'na':>18s}") for v in values))

    print()
    print("=" * 100)
    print("SUSPICIOUS COMPONENT RANKING (correlation between component share and ratio residual)")
    print("=" * 100)
    print("Residual definition: predicted_raw_ratio - measured_ratio.  ")
    print("Positive residual means the model overestimates SIMD (or underestimates SIMT).")
    print("A component whose share co-moves with the residual is likely mis-calibrated.")
    print()
    valid_cases = [c for c in case_data if c["delta_raw"] is not None]
    ranked = []
    for name, desc, fn in components:
        shares = [fn(c["report"]) for c in valid_cases]
        deltas = [c["delta_raw"] for c in valid_cases]
        corr = pearson(shares, deltas)
        ranked.append((name, desc, corr))
    ranked = [r for r in ranked if r[2] is not None]
    ranked.sort(key=lambda r: abs(r[2]), reverse=True)

    if len(valid_cases) < args.min_samples:
        print(f"need at least {args.min_samples} cases for correlation, "
              f"currently {len(valid_cases)}; collect more shapes first.")
    else:
        print(f"{'component':24s} {'corr_with_residual':>20s}  interpretation")
        for name, desc, corr in ranked:
            direction = "overestimates SIMD / underestimates SIMT" if corr > 0 else "overestimates SIMT / underestimates SIMD"
            print(f"{name:24s} {corr:20.3f}  {direction}")
        print()
        print("Next step: take the top 1-2 components, find their underlying cce rate(s),")
        print("and run the corresponding cce sweep (see docs/simt-costmodel-dataset-plan.md).")


if __name__ == "__main__":
    main()
