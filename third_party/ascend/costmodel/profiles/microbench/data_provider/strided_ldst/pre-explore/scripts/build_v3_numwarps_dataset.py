#!/usr/bin/env python3
"""Build the SIMT strided-load v3 num_warps dataset from the coarse board sweep.

Source: results/board_v2_simt_warps/frequency_retry_results_merged_v2.json
(224 coarse cases plus dense W=32/64 N2 cases and 29 higher-rep retries).

The dataset contains 344 valid cases:
    num_warps = 1,2,4,8,16,32,64
    block     = 32,64,128,256
    stride    = 1,4,8,16,32,64,128,256
All targets use the v2 frequency-aware Event-slope protocol.

The output keeps only physically derivable facts.  The current 224-row sweep
is self-contained; the older dense v2 num_warps=1 table is written next to it
as a reference file, not mixed into the fitting table, because the two
measurement sessions differ by up to ~10% on overlapping points.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent  # pre-explore root (scripts/ lives directly under it)
DEFAULT_SOURCE = HERE / "results/board_v2_simt_warps/frequency_retry_results_merged_v2.json"
DEFAULT_OUT = HERE / "results/model_v3_numwarps/dataset.csv"
DEFAULT_V2_REFERENCE = HERE / "results/model_v3_numwarps/dataset_v2_w1_reference.csv"

LINE_BYTES = 128
LANES_PER_WARP = 32
ELEM_BYTES = 4


def derive_facts(block: int, stride: int, num_warps: int):
    stride_bytes = stride * ELEM_BYTES
    span_bytes = (block - 1) * stride_bytes
    elements_per_thread = block / (num_warps * LANES_PER_WARP)
    duplication = max(0.0, 1.0 / elements_per_thread - 1.0) if elements_per_thread > 0 else 0.0
    # Total number of SIMT_LDG instructions per logical program and iteration.
    # CAModel shows this is num_warps * ceil(block/(num_warps*32)) for the
    # shapes measured here (each warp gets at least one instruction).
    k = num_warps * max(1, math.ceil(block / (num_warps * LANES_PER_WARP)))
    lines = block if stride_bytes >= LINE_BYTES else ((block - 1) * stride_bytes) // LINE_BYTES + 1
    return dict(
        elem_bytes=ELEM_BYTES,
        stride_bytes=stride_bytes,
        span_bytes=span_bytes,
        log_span_bytes=math.log(max(1.0, span_bytes)),
        cross_page=1 if span_bytes >= 4096 else 0,
        elements_per_thread=elements_per_thread,
        dup=duplication,
        dup_times_L=duplication * lines,
        dup_times_log_span=duplication * math.log(max(1.0, span_bytes)),
        K=k,
        L=lines,
        K_times_cross=k * (1 if span_bytes >= 4096 else 0),
        L_times_cross=lines * (1 if span_bytes >= 4096 else 0),
        W_times_cross=num_warps * (1 if span_bytes >= 4096 else 0),
        W_times_L=num_warps * lines,
        log_L=math.log2(max(1, lines)),
        useful_bytes=block * ELEM_BYTES,
        bytes_per_line=(block * ELEM_BYTES) / lines,
        line_utilization=(block * ELEM_BYTES) / (lines * LINE_BYTES),
    )


def load_sweep(path: Path):
    payload = json.loads(path.read_text())
    rows = []
    for case in payload["cases"]:
        if not case.get("valid", False):
            continue
        target = case.get("event_slope_normalized_ns_per_iter")
        if target is None:
            target = case["event_slope_ns_per_iter"]
        block = int(case["block"])
        stride = int(case["stride"])
        num_warps = int(case["num_warps"])
        row = {
            "mode": "simt",
            "num_warps": num_warps,
            "block": block,
            "stride": stride,
            "target_ns": float(target),
            "target_kind": "simt_rotate_iter",
            "aicore_freq_before_mhz": case.get("aicore_freq_before_mhz"),
            "aicore_freq_after_mhz": case.get("aicore_freq_after_measure_mhz"),
            "retries": case.get("retries"),
            "source_json": "board_v2_simt_warps",
            "valid": True,
        }
        row.update(derive_facts(block, stride, num_warps))
        rows.append(row)
    rows.sort(key=lambda r: (r["num_warps"], r["block"], r["stride"]))
    return rows


def load_v2_w1_reference(dataset: Path, out: Path):
    if not dataset.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode", "num_warps", "block", "stride", "target_ns", "target_kind",
        "source_json", "elem_bytes", "stride_bytes", "span_bytes",
        "log_span_bytes", "cross_page", "elements_per_thread", "dup",
        "dup_times_L", "dup_times_log_span", "K", "L", "useful_bytes",
        "bytes_per_line", "line_utilization",
    ]
    rows = []
    with dataset.open(newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("mode") != "simt" or raw.get("valid", "True").lower() not in ("true", "1", "yes"):
                continue
            block = int(raw["block"])
            stride = int(raw["stride"])
            row = dict(raw)
            row.update(derive_facts(block, stride, 1))
            row["mode"] = "simt"
            row["num_warps"] = 1
            row["target_ns"] = float(raw["target_ns"])
            row["source_json"] = "model_v2_reference"
            rows.append(row)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} v2 W=1 reference rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--v2-dataset", type=Path, default=HERE / "results/model_v2/dataset.csv")
    ap.add_argument("--v2-reference", type=Path, default=DEFAULT_V2_REFERENCE)
    args = ap.parse_args()

    rows = load_sweep(args.source)
    fields = [
        "mode", "num_warps", "block", "stride", "target_ns", "target_kind",
        "aicore_freq_before_mhz", "aicore_freq_after_mhz", "retries",
        "source_json", "valid",
        "elem_bytes", "stride_bytes", "span_bytes", "log_span_bytes",
        "cross_page", "elements_per_thread", "dup", "dup_times_L",
        "dup_times_log_span", "K", "L", "K_times_cross", "L_times_cross",
        "W_times_cross", "W_times_L", "log_L", "useful_bytes", "bytes_per_line",
        "line_utilization",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    load_v2_w1_reference(args.v2_dataset, args.v2_reference)


if __name__ == "__main__":
    main()
