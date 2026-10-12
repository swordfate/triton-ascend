#!/usr/bin/env python3
"""Frequency-aware retry probe for the dense strided-load measurement.

The v1 dense run showed a bimodal core-frequency state:
  * normal   : matched no-load kernel ~250-400 SYS_CNT cycles
  * throttled: same kernel ~1700-2200 SYS_CNT cycles

This driver re-uses the existing SIMD SYS_CNT kernels and adds:
  * npu-smi current-frequency logging,
  * a busy warm-up loop to force the core back to boost,
  * per-case retry when the matched no-load reference is abnormal,
  * a fallback frequency normalization using the in-kernel no-load cycles
    (and the npu-smi frequency as provenance).

It is intentionally small: use it to validate the protocol on a few cases
before running the full dense matrix.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch_npu

from simd_one_load_syscnt import SYS_CNT_MHZ, load_kernel, noload_kernel, simd_opts


def read_aicore_freq_mhz(device: int):
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", str(device)],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
    except Exception:
        return None
    match = re.search(r"Aicore curFreq\(MHZ\)\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def warm_up_npu(opts, out, t0, t1, block, launches=500):
    for _ in range(launches):
        noload_kernel[(1, )](out, t0, t1, BLOCK=block, **opts)
    torch.npu.synchronize()


def run_measurement(opts, xs, out, t0, t1, block, stride):
    vals_load = []
    vals_noload = []
    for x in xs:
        load_kernel[(1, )](x, out, t0, t1, BLOCK=block, STRIDE=stride, **opts)
        torch.npu.synchronize()
        vals_load.append(int(t1.cpu().item()) - int(t0.cpu().item()))
        noload_kernel[(1, )](out, t0, t1, BLOCK=block, **opts)
        torch.npu.synchronize()
        vals_noload.append(int(t1.cpu().item()) - int(t0.cpu().item()))
    delta = [a - b for a, b in zip(vals_load, vals_noload)]
    return vals_load, vals_noload, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=[1, 2, 3, 6, 9, 16, 32, 56, 80, 192])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--nbuf", type=int, default=15)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-aicore-mhz", type=int, default=1500)
    ap.add_argument("--normal-noload-max", type=int, default=800)
    ap.add_argument("--nominal-noload", type=int, default=350)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--warmup-launches", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("frequency_retry_probe.json"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    opts = simd_opts()
    rows = []
    for block in args.blocks:
        out = torch.zeros(1, dtype=torch.float32, device="npu")
        t0 = torch.zeros(1, dtype=torch.int64, device="npu")
        t1 = torch.zeros(1, dtype=torch.int64, device="npu")
        noload_kernel.warmup(out, t0, t1, BLOCK=block, grid=(1, ), **opts)
        for stride in args.strides:
            total = args.nbuf * block * max(stride, 1) + 4096
            xs = [(torch.arange(total, dtype=torch.float32) % 7.0).contiguous().to("npu")
                  for _ in range(min(args.nbuf, args.reps))]
            load_kernel.warmup(xs[0], out, t0, t1, BLOCK=block, STRIDE=stride,
                               grid=(1, ), **opts)
            attempts = []
            accepted = None
            for attempt in range(args.max_retries + 1):
                freq_before = read_aicore_freq_mhz(args.device)
                if freq_before is None or freq_before < args.min_aicore_mhz:
                    warm_up_npu(opts, out, t0, t1, block, args.warmup_launches)
                    freq_after_warm = read_aicore_freq_mhz(args.device)
                else:
                    freq_after_warm = freq_before
                vals_load, vals_noload, delta = run_measurement(
                    opts, xs, out, t0, t1, block, stride)
                freq_after = read_aicore_freq_mhz(args.device)
                noload_median = statistics.median(vals_noload)
                row_attempt = {
                    "attempt": attempt,
                    "aicore_freq_before_mhz": freq_before,
                    "aicore_freq_after_warmup_mhz": freq_after_warm,
                    "aicore_freq_after_measure_mhz": freq_after,
                    "noload_cycles_median": noload_median,
                    "load_cycles_median": statistics.median(vals_load),
                    "delta_median": statistics.median(delta),
                    "delta_min": min(delta),
                    "delta_max": max(delta),
                }
                attempts.append(row_attempt)
                # The in-kernel no-load reference is the primary health check;
                # the npu-smi query is context/provenance and may race.
                if noload_median <= args.normal_noload_max:
                    accepted = (vals_load, vals_noload, delta, row_attempt)
                    break
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep)
            if accepted is not None:
                vals_load, vals_noload, delta, accepted_attempt = accepted
                row = {
                    "block": block,
                    "stride": stride,
                    "valid": True,
                    "retries": accepted_attempt["attempt"],
                    "reps": len(vals_load),
                    "load_cycles_median": statistics.median(vals_load),
                    "noload_cycles_median": statistics.median(vals_noload),
                    "delta_median": statistics.median(delta),
                    "delta_min": min(delta),
                    "delta_max": max(delta),
                    "aicore_freq_before_mhz": accepted_attempt["aicore_freq_before_mhz"],
                    "aicore_freq_after_measure_mhz": accepted_attempt["aicore_freq_after_measure_mhz"],
                    "normalization_factor": 1.0,
                    "normalized_delta": statistics.median(delta),
                    "attempts": attempts,
                }
            else:
                last = attempts[-1]
                # Fallback: normalize via the in-kernel no-load reference.
                normalization = args.nominal_noload / max(1, last["noload_cycles_median"])
                row = {
                    "block": block,
                    "stride": stride,
                    "valid": False,
                    "retries": args.max_retries,
                    "reps": args.reps,
                    "load_cycles_median": last["load_cycles_median"],
                    "noload_cycles_median": last["noload_cycles_median"],
                    "delta_median": last["delta_median"],
                    "delta_min": last["delta_min"],
                    "delta_max": last["delta_max"],
                    "aicore_freq_before_mhz": last["aicore_freq_before_mhz"],
                    "aicore_freq_after_measure_mhz": last["aicore_freq_after_measure_mhz"],
                    "normalization_factor": normalization,
                    "normalized_delta": last["delta_median"] * normalization,
                    "attempts": attempts,
                }
            rows.append(row)
            status = "OK" if row["valid"] else "FALLBACK"
            print(f'{status} b{block} s{stride}: '
                  f'load={row["load_cycles_median"]:.0f} '
                  f'noload={row["noload_cycles_median"]:.0f} '
                  f'delta={row["delta_median"]:.0f} '
                  f'norm={row["normalized_delta"]:.0f} '
                  f'freq_before={row["aicore_freq_before_mhz"]} '
                  f'freq_after={row["aicore_freq_after_measure_mhz"]} '
                  f'retries={row["retries"]}', flush=True)
    args.out.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
