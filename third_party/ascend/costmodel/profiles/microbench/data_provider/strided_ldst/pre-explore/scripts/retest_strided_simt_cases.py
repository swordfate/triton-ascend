#!/usr/bin/env python3
"""Retest selected SIMT strided-load cases with independent buffer allocations.

Each case is measured with several independently allocated page-aligned buffers
held alive simultaneously.  A transient outlier (host/device contention, bad
allocator address, throttling) usually affects only one allocation, so the
median across allocations is a robust target.  Results also record every
buffer address and the per-allocation target for audit.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import syscnt_event_compare_aligned as sec

MIN_AICORE_MHZ = 1500


def read_aicore_freq_mhz(device: int):
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", str(device)],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
    except Exception:
        return None
    m = re.search(r"Aicore curFreq\(MHZ\)\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def measure_one(case, args):
    mode = "simt_only"
    block = int(case["block"])
    stride = int(case["stride"])
    num_warps = int(case["num_warps"])
    allocations = []
    # Keep all buffers alive so the allocator cannot hand the same address to
    # the next allocation in this case.
    held_buffers = []
    for alloc in range(args.allocations):
        attempt_rows = []
        accepted = None
        for attempt in range(args.max_retries + 1):
            x = sec.make_buffer(block, stride, args.rotate_step, args.rotate_mask)
            held_buffers.append(x)
            freq_before = read_aicore_freq_mhz(args.freq_device)
            result = sec.run_case(
                mode, block, stride, args.rotate_step, args.rotate_mask,
                args.iters, args.reps, num_warps=num_warps, x=x)
            freq_after = read_aicore_freq_mhz(args.freq_device)
            row = dict(result)
            row["allocation"] = alloc
            row["attempt"] = attempt
            row["aicore_freq_before_mhz"] = freq_before
            row["aicore_freq_after_measure_mhz"] = freq_after
            row["valid"] = bool(
                result.get("event_slope_ns_per_iter") is not None
                and (freq_after or 0) >= MIN_AICORE_MHZ)
            attempt_rows.append(row)
            if row["valid"]:
                accepted = row
                break
            time.sleep(args.retry_sleep)
        if accepted is None:
            accepted = attempt_rows[-1]
        allocations.append(accepted)
    targets = [a["event_slope_ns_per_iter"] for a in allocations
               if a.get("event_slope_ns_per_iter") is not None]
    return {
        "block": block,
        "stride": stride,
        "num_warps": num_warps,
        "targets_ns": targets,
        "median_ns": statistics.median(targets) if targets else None,
        "min_ns": min(targets) if targets else None,
        "max_ns": max(targets) if targets else None,
        "allocations": allocations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True,
                    help="JSON list: [{block, stride, num_warps}, ...]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allocations", type=int, default=3)
    ap.add_argument("--iters", nargs="+", type=int, default=[2000, 8000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rotate-step", type=int, default=8192)
    ap.add_argument("--rotate-mask", type=int, default=16383)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--freq-device", type=int, default=1)
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    results = []
    for index, case in enumerate(cases):
        result = measure_one(case, args)
        results.append(result)
        print(
            f'[{index + 1}/{len(cases)}] W={result["num_warps"]} '
            f'b{result["block"]} s{result["stride"]} '
            f'targets={[round(t, 2) for t in result["targets_ns"]]} '
            f'median={result["median_ns"]}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2),
                        encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
