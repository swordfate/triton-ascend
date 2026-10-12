#!/usr/bin/env python3
"""Build the strided-store fitting dataset from board measurement JSONs.

Inputs are the JSON payloads produced by ``store_measure_probe.py``:

  results/model_store_v1/raw/board_store_simd.json
  results/model_store_v1/raw/board_store_simt.json

Output:

  results/model_store_v1/dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ELEM_BYTES = 4
WARP = 32
PAGE_BYTES = 4096
BANK_BYTES = 2048
BANK_COUNT = 8


def distinct_lines(block: int, stride_bytes: int) -> int:
    if stride_bytes >= 128:
        return block
    return ((block - 1) * stride_bytes) // 128 + 1


def line_request_size(block: int, stride_bytes: int) -> int:
    if stride_bytes >= 128:
        return ELEM_BYTES
    elems_per_line = max(1, 128 // stride_bytes)
    return min(128, elems_per_line * ELEM_BYTES)


def bank_facts(block: int, stride_bytes: int):
    counts = [0] * BANK_COUNT
    for i in range(block):
        bucket = ((i * stride_bytes) // BANK_BYTES) % BANK_COUNT
        counts[bucket] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1
    return pairs, worst


def add_features(row):
    block = int(row["block"])
    stride = int(row["stride"])
    num_warps = int(row["num_warps"])
    stride_bytes = stride * ELEM_BYTES
    elements_per_thread = block / (WARP * num_warps)
    e = elements_per_thread
    dup = max(0.0, 1.0 / e - 1.0) if e > 0 else 0.0
    lines = distinct_lines(block, stride_bytes)
    span = (block - 1) * stride_bytes + ELEM_BYTES
    k_stg = num_warps * max(1, (block + WARP * num_warps - 1) // (WARP * num_warps))
    pairs, worst = bank_facts(block, stride_bytes)
    # SIMD gather candidate collision features (aligned-base convention).
    counts_s512 = [0] * 32
    counts_g32768 = [0] * 16
    for i in range(block):
        counts_s512[((i * stride_bytes) // 512) % 32] += 1
        counts_g32768[((i * stride_bytes) // 32768) % 16] += 1
    pairs_s512 = sum(c * (c - 1) // 2 for c in counts_s512)
    occ_g512 = sum(1 for c in counts_s512 if c)
    worst_g32768 = max(counts_g32768) - 1 if counts_g32768 else 0
    row.update({
        "pairs_S512": pairs_s512,
        "occ_G512": occ_g512,
        "worst_G32768": worst_g32768,
        "elem_bytes": ELEM_BYTES,
        "stride_bytes": stride_bytes,
        "elements": block,
        "span_bytes": span,
        "elements_per_thread": elements_per_thread,
        "E": e,
        "K_stg": k_stg,
        "dup": dup,
        "L": lines,
        "line_request_size": line_request_size(block, stride_bytes),
        "cross_page": 1 if span >= PAGE_BYTES else 0,
        "page_count": span // PAGE_BYTES,
        "n_cmd": 1 if (row["mode"] == "simd" and stride == 1) else block,
        "wide": 1 if (row["mode"] == "simd" and stride == 1) else 0,
        "bank_pairs": pairs,
        "bank_worst": worst,
        "dupL": dup * lines,
        "W_times_L": num_warps * lines,
        "L_min_64": min(lines, 64),
        "W_times_L_minus_3072": max(0, num_warps * lines - 3072),
        "dup_cross": dup * (1 if span >= PAGE_BYTES else 0),
    })
    return row


def pick_target(case):
    """Minimum observed Event time per iteration.

    Each short-kernel repetition is either at boost or at a lower DVFS state;
    the minimum is the best estimate of the boosted steady-state rate and
    rejects slow-frequency samples.  It is a ``ns/iteration`` throughput target
    with the same rotate-loop semantics for SIMD and SIMT.
    """
    best = None
    for attempt in case.get("attempts", []):
        for row in attempt.get("rows", []):
            iters = float(row.get("iters", 0) or 0)
            if iters <= 0:
                continue
            values = row.get("event_ns_all") or [row.get("event_ns_min")]
            for value in values:
                target = float(value) / iters
                if best is None or target < best[0]:
                    best = (target, attempt, row, float(value))
    return best


def load_raw(path: Path, out):
    payload = json.loads(path.read_text())
    mode = payload.get("mode")
    freq = payload.get("frequency_check", {})
    for case in payload.get("cases", []):
        chosen = pick_target(case)
        if chosen is None:
            target = float(case.get("target_ns", 0.0))
            witness_ok = 0
            source = "fallback_case_target"
            iters = None
        else:
            target, attempt, raw_row, raw_event = chosen
            witness_ok = 1 if attempt.get("witness_ok") else 0
            source = f"min_observed_iters{int(raw_row.get('iters', 0))}"
            iters = int(raw_row.get("iters", 0))
        row = {
            "mode": mode,
            "block": int(case["block"]),
            "stride": int(case["stride"]),
            "num_warps": int(case["num_warps"]),
            "target_ns": float(target),
            "target_kind": "store_rotate_min_observed_ns_per_iter",
            "target_source": source,
            "target_iters": iters,
            "valid": int(bool(target > 0.0 and freq.get("inflight_freq_mhz") is not None
                              and freq.get("inflight_freq_mhz") >= 1500)),
            "raw_valid": int(bool(case.get("valid"))),
            "freq_check_mhz": freq.get("inflight_freq_mhz"),
            "witness_ok": int(witness_ok),
            "target_spread": None,
        }
        out.append(add_features(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--simd", type=Path, default=None)
    ap.add_argument("--simd-extra", type=Path, nargs="*", default=None)
    ap.add_argument("--simt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    simd = args.simd or root / "results/model_store_v1/raw/board_store_simd.json"
    simd_extra = args.simd_extra or [
        root / "results/model_store_v1/raw/board_store_simd_wide_more.json"
    ]
    simt = args.simt or root / "results/model_store_v1/raw/board_store_simt.json"
    out = args.out or root / "results/model_store_v1/dataset.csv"
    rows = []
    input_paths = [simd, *simd_extra, simt]
    for path in input_paths:
        if path.exists():
            load_raw(path, rows)
        else:
            print(f"warning: missing {path}")
    if not rows:
        raise SystemExit("no measurement rows found")
    # De-duplicate (mode, block, stride, num_warps).  The supplement file is
    # read after the main SIMD file, so the most recent target wins if the same
    # case ever appears twice (the expected wide/gather block sets are disjoint).
    dedup = {}
    for row in rows:
        key = (row["mode"], row["block"], row["stride"], row["num_warps"])
        dedup[key] = row
    rows = [dedup[key] for key in sorted(dedup)]
    fieldnames = list(rows[0].keys())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
