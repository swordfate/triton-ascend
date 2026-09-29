#!/usr/bin/env python3
"""Frequency-aware dense SIMT Event-slope measurement (protocol v2).

The SIMT path cannot use the inline SYS_CNT asm, so the measured quantity is
the `torch.npu.Event` slope over an in-kernel iteration loop.  The same
frequency instability that affects the SIMD SYS_CNT probe also affects Event
slopes, therefore this driver:

  * queries `npu-smi info -t common -i <dev>` for Aicore current frequency,
  * runs a small `loop_noasm` warm-up burst when the core is throttled,
  * retries a case when frequency is not in boost after warm-up/measurement,
  * records all frequency readings, retries and (if ever needed) a
    frequency-normalized fallback value.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch_npu

import syscnt_event_compare as sec


def read_aicore_freq_mhz(device: int):
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", str(device)],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
    except Exception:
        return None
    match = re.search(r"Aicore curFreq\(MHZ\)\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def warm_up_simt(device, opts, block, stride, step, mask, launches, device_id):
    # opts already carries num_warps; keep this helper dumb on purpose.
    x = torch.ones(max(4096, block * max(1, stride) * 4), dtype=torch.float32, device="npu")
    out = torch.zeros(1, dtype=torch.float32, device="npu")
    sec.loop_noasm.warmup(x, out, 64, BLOCK=block, STRIDE=stride,
                          STEP=step, MASK=mask, grid=(1, ), **opts)
    for _ in range(launches):
        sec.loop_noasm[(1, )](x, out, 64, BLOCK=block, STRIDE=stride,
                              STEP=step, MASK=mask, **opts)
    torch.npu.synchronize()
    return read_aicore_freq_mhz(device_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[1])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=list(range(1, 17)) + [20, 24, 32, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--iters", nargs="+", type=int, default=[2000, 8000])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rotate-step", type=int, default=8192)
    ap.add_argument("--rotate-mask", type=int, default=16383)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--nominal-mhz", type=int, default=1650)
    ap.add_argument("--min-aicore-mhz", type=int, default=1500)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--warmup-launches", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("simt_v2_results.json"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    rows = []
    for num_warps in args.num_warps:
        o = sec.opts("simt_only", num_warps=num_warps)
        for block in args.blocks:
            for stride in args.strides:
                attempts = []
                accepted = None
                for attempt in range(args.max_retries + 1):
                    freq_before = read_aicore_freq_mhz(args.device)
                    freq_after_warm = freq_before
                    if freq_before is None or freq_before < args.min_aicore_mhz:
                        freq_after_warm = warm_up_simt(
                            args.device, o, block, stride, args.rotate_step,
                            args.rotate_mask, args.warmup_launches, args.device)
                    if (freq_after_warm or 0) < args.min_aicore_mhz:
                        attempts.append({
                            "attempt": attempt,
                            "aicore_freq_before_mhz": freq_before,
                            "aicore_freq_after_warmup_mhz": freq_after_warm,
                            "reason": "warmup_not_boosted",
                        })
                        if attempt < args.max_retries:
                            time.sleep(args.retry_sleep)
                        continue
                    # run_case compiles/warm-ups and measures the Event slope.
                    result = sec.run_case(
                        "simt_only", block, stride, args.rotate_step,
                        args.rotate_mask, args.iters, args.reps,
                        num_warps=num_warps)
                    freq_after = read_aicore_freq_mhz(args.device)
                    attempt_row = {
                        "attempt": attempt,
                        "aicore_freq_before_mhz": freq_before,
                        "aicore_freq_after_warmup_mhz": freq_after_warm,
                        "aicore_freq_after_measure_mhz": freq_after,
                        "event_slope_ns_per_iter": result["event_slope_ns_per_iter"],
                    }
                    attempts.append(attempt_row)
                    if freq_after is not None and freq_after >= args.min_aicore_mhz:
                        accepted = result
                        break
                    if attempt < args.max_retries:
                        time.sleep(args.retry_sleep)
                if accepted is None:
                    last = attempts[-1]
                    result = {
                        "mode": "simt_only",
                        "num_warps": int(num_warps),
                        "block": block,
                        "stride": stride,
                        "step": args.rotate_step,
                        "mask": args.rotate_mask,
                        "iters": args.iters,
                        "reps": args.reps,
                        "rows": [],
                        "event_slope_ns_per_iter": last.get("event_slope_ns_per_iter"),
                        "event_slope_syscnt_cycles_per_iter": (
                            last.get("event_slope_ns_per_iter", 0.0) * sec.SYS_CNT_MHZ
                            if last.get("event_slope_ns_per_iter") is not None else None),
                        "valid": False,
                        "retries": args.max_retries,
                        "aicore_freq_before_mhz": last.get("aicore_freq_before_mhz"),
                        "aicore_freq_after_measure_mhz": last.get("aicore_freq_after_measure_mhz"),
                        "normalization_factor": (
                            (last.get("aicore_freq_after_measure_mhz") or args.nominal_mhz)
                            / args.nominal_mhz),
                        "event_slope_normalized_ns_per_iter": (
                            last.get("event_slope_ns_per_iter", 0.0) *
                            ((last.get("aicore_freq_after_measure_mhz") or args.nominal_mhz)
                             / args.nominal_mhz)
                            if last.get("event_slope_ns_per_iter") is not None else None),
                        "attempts": attempts,
                    }
                else:
                    result["valid"] = True
                    result["num_warps"] = int(num_warps)
                    result["retries"] = len(attempts) - 1
                    result["aicore_freq_before_mhz"] = attempts[-1]["aicore_freq_before_mhz"]
                    result["aicore_freq_after_measure_mhz"] = attempts[-1]["aicore_freq_after_measure_mhz"]
                    result["normalization_factor"] = 1.0
                    result["event_slope_normalized_ns_per_iter"] = result["event_slope_ns_per_iter"]
                    result["attempts"] = attempts
                rows.append(result)
                print(f'{"OK" if result["valid"] else "FALLBACK"} '
                      f'W={num_warps} b{block} s{stride}: '
                      f'event={result.get("event_slope_ns_per_iter")} ns/iter '
                      f'freq_before={result.get("aicore_freq_before_mhz")} '
                      f'freq_after={result.get("aicore_freq_after_measure_mhz")} '
                      f'retries={result.get("retries")}', flush=True)
    args.out.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
