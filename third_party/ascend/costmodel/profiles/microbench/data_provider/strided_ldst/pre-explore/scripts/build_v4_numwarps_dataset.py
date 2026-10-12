#!/usr/bin/env python3
"""Build the strided-load v4 SIMT dataset from page-aligned measurements.

Inputs (all relative to the pre-explore root):
  results/v4_aligned_dev1/*.json
  results/v4_aligned_dev1_extra/*.json
  results/v4_aligned_23/*.json
  results/v4_retest_full_23.json       # 5-allocation retest medians, preferred

Outputs:
  results/model_v4_numwarps/dataset.csv
  results/model_v4_numwarps/flagged_discordant.csv

Every row is generated from ``derive_facts(block, stride, num_warps)``; raw
rows are only used to supply target metadata (min/max/spread/source).  This
keeps feature construction independent of which measurement pass happened to
contain a case.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
OUT_DIR = HERE / "results/model_v4_numwarps"

LINE_BYTES = 128
LANES_PER_WARP = 32
ELEM_BYTES = 4
PAGE_BYTES = 4096
TARGET_FLOOR_NS = 120.0

RAW_GLOBS = [
    "results/v4_aligned_dev1/*.json",
    "results/v4_aligned_dev1_extra/*.json",
    "results/v4_aligned_23/*.json",
]
DEFAULT_RETEST = "results/v4_retest_full_23.json"

REPORT_FIELDS = [
    "mode", "num_warps", "block", "stride",
    "target_ns", "target_p25_ns", "target_min_ns", "target_max_ns",
    "target_spread", "n_samples", "n_allocations", "n_dropped",
    "target_kind", "source_json", "valid",
    "elem_bytes", "stride_bytes", "span_bytes", "page_count", "cross_page",
    "elements_per_thread", "dup", "dup_times_L", "K", "L", "W_times_L",
    "useful_bytes", "line_utilization",
    "bank_pairs_g2048_b8", "bank_worst_g2048_b8",
    "bank_pairs_g32768_b16", "bank_worst_g32768_b16",
]


def bank_facts(block: int, stride_bytes: int, group_bytes: int, banks: int):
    counts = [0] * banks
    for i in range(int(block)):
        counts[((i * stride_bytes) // group_bytes) % banks] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1 if counts else 0
    return pairs, worst


def derive_facts(block: int, stride: int, num_warps: int):
    stride_bytes = stride * ELEM_BYTES
    span_bytes = (block - 1) * stride_bytes
    elements_per_thread = block / (num_warps * LANES_PER_WARP)
    duplication = (
        max(0.0, 1.0 / elements_per_thread - 1.0)
        if elements_per_thread > 0
        else 0.0
    )
    lines = (
        block
        if stride_bytes >= LINE_BYTES
        else ((block - 1) * stride_bytes) // LINE_BYTES + 1
    )
    k = num_warps * max(1, math.ceil(block / (num_warps * LANES_PER_WARP)))
    pairs_g2048_b8, worst_g2048_b8 = bank_facts(block, stride_bytes, 2048, 8)
    pairs_g32768_b16, worst_g32768_b16 = bank_facts(block, stride_bytes, 32768, 16)
    return dict(
        elem_bytes=ELEM_BYTES,
        stride_bytes=stride_bytes,
        span_bytes=span_bytes,
        page_count=span_bytes // PAGE_BYTES,
        cross_page=1 if span_bytes >= PAGE_BYTES else 0,
        elements_per_thread=elements_per_thread,
        dup=duplication,
        dup_times_L=duplication * lines,
        K=k,
        L=lines,
        W_times_L=num_warps * lines,
        useful_bytes=block * ELEM_BYTES,
        line_utilization=(block * ELEM_BYTES) / (lines * LINE_BYTES),
        bank_pairs_g2048_b8=pairs_g2048_b8,
        bank_worst_g2048_b8=worst_g2048_b8,
        bank_pairs_g32768_b16=pairs_g32768_b16,
        bank_worst_g32768_b16=worst_g32768_b16,
    )


def read_raw_cases(path: Path):
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    out = []
    for case in payload.get("cases", []):
        if not case.get("valid", False):
            continue
        target = case.get("event_slope_normalized_ns_per_iter")
        if target is None:
            target = case.get("event_slope_ns_per_iter")
        if target is None:
            continue
        out.append({
            "num_warps": int(case.get("num_warps", 1)),
            "block": int(case["block"]),
            "stride": int(case["stride"]),
            "target_ns": float(target),
            "source": str(path.relative_to(HERE)),
            "aicore_freq_before_mhz": case.get("aicore_freq_before_mhz"),
            "aicore_freq_after_measure_mhz": case.get("aicore_freq_after_measure_mhz"),
            "buffer_data_ptr": case.get("buffer_data_ptr"),
            "buffer_page_offset": case.get("buffer_page_offset"),
            "buffer_line_offset": case.get("buffer_line_offset"),
        })
    return out


def target_stats(values, floor_ns=TARGET_FLOOR_NS):
    """Robust target summary with a measured positive floor.

    The board Event slope is occasionally negative or far below the physical
    floor (~130 ns/iteration for every observed legal configuration).  Such
    allocation samples are treated as timing artifacts.  ``target_ns`` remains
    the instructed median of the cleaned samples; ``target_p25_ns`` is a
    lower-quartile robust target for model fitting.
    """
    raw = [float(v) for v in values if v is not None]
    cleaned = sorted(v for v in raw if v >= floor_ns)
    if not cleaned:
        cleaned = sorted(v for v in raw if v > 0.0)
    if not cleaned:
        return None
    n = len(cleaned)
    median = cleaned[n // 2] if n % 2 else 0.5 * (cleaned[n // 2 - 1] + cleaned[n // 2])
    p25 = cleaned[(n - 1) // 4]
    return {
        "target_ns": float(median),
        "target_p25_ns": float(p25),
        "target_min_ns": float(cleaned[0]),
        "target_max_ns": float(cleaned[-1]),
        "target_spread": (cleaned[-1] / cleaned[0] if cleaned[0] > 0 else 0.0),
        "n_samples": n,
        "n_dropped": len(raw) - len(cleaned),
    }


def build_rows(raw_rows, retest_rows):
    raw_groups = defaultdict(list)
    for row in raw_rows:
        if float(row["target_ns"]) > 0.0:
            raw_groups[(row["num_warps"], row["block"], row["stride"])].append(row)

    retest_groups = defaultdict(list)
    for row in retest_rows:
        values = [t for t in row.get("targets_ns", []) if t is not None]
        if values:
            retest_groups[(int(row["num_warps"]), int(row["block"]),
                           int(row["stride"]))].extend(values)

    rows = []
    flagged = []
    for key in sorted(set(raw_groups) | set(retest_groups)):
        num_warps, block, stride = key
        record = {
            "mode": "simt",
            "num_warps": num_warps,
            "block": block,
            "stride": stride,
            "valid": True,
        }
        record.update(derive_facts(block, stride, num_warps))

        if key in retest_groups:
            stats = target_stats(retest_groups[key])
            if stats is None:
                continue
            record.update(stats)
            record.update({
                "n_allocations": len(retest_groups[key]),
                "target_kind": "simt_rotate_aligned_event_median",
                "source_json": DEFAULT_RETEST,
            })
            if key in raw_groups:
                raw = raw_groups[key][0]
                record.update({
                    "aicore_freq_before_mhz": raw.get("aicore_freq_before_mhz"),
                    "aicore_freq_after_measure_mhz": raw.get("aicore_freq_after_measure_mhz"),
                    "buffer_data_ptr": raw.get("buffer_data_ptr"),
                    "buffer_page_offset": raw.get("buffer_page_offset"),
                    "buffer_line_offset": raw.get("buffer_line_offset"),
                })
        else:
            candidates = raw_groups[key]
            stats = target_stats([r["target_ns"] for r in candidates])
            if stats is None:
                continue
            template = candidates[0]
            record.update(stats)
            record.update({
                "n_allocations": len(candidates),
                "target_kind": "simt_rotate_aligned_event_fallback",
                "source_json": ";".join(sorted({r["source"] for r in candidates})),
                "aicore_freq_before_mhz": template.get("aicore_freq_before_mhz"),
                "aicore_freq_after_measure_mhz": template.get("aicore_freq_after_measure_mhz"),
                "buffer_data_ptr": template.get("buffer_data_ptr"),
                "buffer_page_offset": template.get("buffer_page_offset"),
                "buffer_line_offset": template.get("buffer_line_offset"),
            })

        rows.append(record)
        if (record["target_spread"] > 1.4 or record["n_samples"] < 3 or
                record.get("n_dropped", 0) > 0):
            flagged.append({
                "num_warps": record["num_warps"],
                "block": record["block"],
                "stride": record["stride"],
                "target_ns": record["target_ns"],
                "target_p25_ns": record["target_p25_ns"],
                "min_ns": record["target_min_ns"],
                "max_ns": record["target_max_ns"],
                "spread": record["target_spread"],
                "n_samples": record["n_samples"],
                "n_allocations": record["n_allocations"],
                "n_dropped": record.get("n_dropped", 0),
            })
    return rows, flagged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR / "dataset.csv")
    ap.add_argument("--retest", type=Path, default=HERE / DEFAULT_RETEST)
    args = ap.parse_args()

    raw_rows = []
    for pattern in RAW_GLOBS:
        for path in sorted(HERE.glob(pattern)):
            raw_rows.extend(read_raw_cases(path))

    retest_rows = []
    if args.retest.exists():
        retest_rows = json.loads(args.retest.read_text()).get("results", [])
    else:
        raise SystemExit(f"missing retest file: {args.retest}")

    rows, flagged = build_rows(raw_rows, retest_rows)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = REPORT_FIELDS + [
        "aicore_freq_before_mhz", "aicore_freq_after_measure_mhz",
        "buffer_data_ptr", "buffer_page_offset", "buffer_line_offset",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    flag_path = out_path.with_name("flagged_discordant.csv")
    with flag_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_warps", "block", "stride", "target_ns",
                        "target_p25_ns", "min_ns", "max_ns", "spread",
                        "n_samples", "n_allocations", "n_dropped"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flagged)

    print(f"wrote {out_path} rows={len(rows)} flagged={len(flagged)} "
          f"(flags: {flag_path})")


if __name__ == "__main__":
    main()
