#!/usr/bin/env python3
"""Board store target probe (short-kernel Event slope + boost witness).

This is the target used for the store dataset.  A single long kernel is not
used because sustained MTE3/STG traffic can push AICore out of boost; instead
each case is a slope over short rotate-loop kernels:

    T_slope = d(Event_min) / d(iters)   [ns/iteration]

Before every short measured launch the probe runs a short ALU busy burst to
keep DVFS in boost.  A post-case ALU witness must match the boost reference
within ``--witness-tol`` or the case is retried.  A separate dedicated
frequency check samples ``npu-smi`` while a long ALU kernel is in flight; no
``npu-smi`` process runs during a timed Event window.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def busy_kernel(out_ptr, iters, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(iters):
        acc += 1.0
    tl.store(out_ptr + tl.arange(0, BLOCK), acc)


@triton.jit
def store_loop_kernel(out_ptr, iters,
                      BLOCK: tl.constexpr, STRIDE: tl.constexpr,
                      STEP: tl.constexpr, MASK: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    value = tl.arange(0, BLOCK).to(tl.float32) + 1.0
    for i in range(iters):
        tl.store(out_ptr + (i & MASK) * STEP + offs, value + i)


def read_aicore_freq_mhz(device: int):
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", str(device)],
            stderr=subprocess.DEVNULL, text=True, timeout=15)
    except Exception:
        return None
    match = re.search(r"Aicore curFreq\(MHZ\)\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def simd_store_opts():
    return {
        "num_warps": 1,
        "logical_program_count_hint": 1,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd",
        "enable_auto_blockify": False,
    }


def simt_store_opts(num_warps: int):
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": 1,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simt_only",
        "enable_auto_blockify": True,
    }


def measure_event_us(launch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end)  # ms


def slope_ns_per_iter(iters_list, event_ns_list):
    xs = [float(i) for i in iters_list]
    ys = [float(v) for v in event_ns_list]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def freq_check(device, busy_out):
    launch_iters = 150_000_000
    busy_kernel[(1,)](busy_out, launch_iters, BLOCK=1024, num_warps=1,
                      compile_mode="simd", enable_auto_blockify=False,
                      auto_simt_scope_mode="off")
    t0 = time.monotonic()
    freq = read_aicore_freq_mhz(device)
    query_seconds = time.monotonic() - t0
    torch.npu.synchronize()
    return freq, query_seconds


def run_case(mode, block, stride, num_warps, buffer, busy_out, step, mask,
             iters_list, reps, device, reference_witness_ns,
             witness_tol=0.15, max_attempts=3, busy_iters=2_000_000,
             max_spread=0.40):
    opts = simd_store_opts() if mode == "simd" else simt_store_opts(num_warps)
    grid = 1

    def ramp():
        busy_kernel[(grid,)](busy_out, busy_iters, BLOCK=1024, num_warps=1,
                             compile_mode="simd", enable_auto_blockify=False,
                             auto_simt_scope_mode="off")
        torch.npu.synchronize()

    store_loop_kernel.warmup(buffer, 16, BLOCK=block, STRIDE=stride,
                             STEP=step, MASK=mask, grid=(grid,), **opts)
    busy_kernel.warmup(busy_out, 16, BLOCK=1024, grid=(grid,),
                       num_warps=1, compile_mode="simd",
                       enable_auto_blockify=False, auto_simt_scope_mode="off")
    store_loop_kernel[(grid,)](buffer, 16, BLOCK=block, STRIDE=stride,
                               STEP=step, MASK=mask, **opts)
    torch.npu.synchronize()

    attempts = []
    accepted = None
    for attempt in range(max_attempts):
        event_ns_list = []
        raw_rows = []
        for iters in iters_list:
            evs = []
            for _ in range(reps):
                ramp()
                evs.append(measure_event_us(
                    lambda: store_loop_kernel[(grid,)](buffer, iters,
                                                       BLOCK=block, STRIDE=stride,
                                                       STEP=step, MASK=mask,
                                                       **opts)))
            med_ms = statistics.median(evs)
            event_ns_list.append(med_ms * 1e6)  # ms -> ns
            raw_rows.append({"iters": iters,
                             "event_ns_min": min(evs) * 1e6,
                             "event_ns_median": med_ms * 1e6,
                             "event_ns_all": [v * 1e6 for v in evs]})
        target = slope_ns_per_iter(iters_list, event_ns_list)
        per_iter = [event / max(1, it) for event, it in zip(event_ns_list, iters_list)]
        spread = ((max(per_iter) - min(per_iter)) / min(per_iter)
                  if min(per_iter) > 0 else float("inf"))
        ramp()
        post_witness_ns = measure_event_us(
            lambda: busy_kernel[(grid,)](busy_out, busy_iters, BLOCK=1024,
                                         num_warps=1, compile_mode="simd",
                                         enable_auto_blockify=False,
                                         auto_simt_scope_mode="off")) * 1e6
        witness_ok = (reference_witness_ns > 0 and
                      abs(post_witness_ns / reference_witness_ns - 1.0) <= witness_tol)
        stable_ok = spread <= max_spread
        row = {
            "attempt": attempt,
            "rows": raw_rows,
            "target_ns": target,
            "post_witness_ns": post_witness_ns,
            "witness_ok": bool(witness_ok),
            "spread": spread,
            "stable_ok": bool(stable_ok),
        }
        attempts.append(row)
        if witness_ok and stable_ok:
            accepted = row
            break
    valid = accepted is not None
    return {
        "mode": mode,
        "block": int(block),
        "stride": int(stride),
        "num_warps": int(num_warps),
        "target_ns": float(accepted["target_ns"]) if accepted is not None
        else float(statistics.median(r["target_ns"] for r in attempts)),
        "target_kind": "store_rotate_loop_slope_ns_per_iter",
        "valid": bool(valid),
        "step": step,
        "mask": mask,
        "iters_list": list(iters_list),
        "reps": reps,
        "attempts": attempts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simd", "simt_only"], required=True)
    ap.add_argument("--blocks", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=list(range(1, 13)) +
                    [16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[1])
    ap.add_argument("--iters", nargs="+", type=int, default=[2000, 8000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--busy-iters", type=int, default=2_000_000)
    ap.add_argument("--step", type=int, default=8192)
    ap.add_argument("--mask", type=int, default=16383)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--witness-tol", type=float, default=0.15)
    ap.add_argument("--max-spread", type=float, default=0.40)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    max_tile = max(args.blocks) * max(args.strides)
    total = args.mask * args.step + max_tile + 4096
    buffer = torch.empty(total, dtype=torch.float32, device="npu")
    busy_out = torch.empty(4096, dtype=torch.float32, device="npu")

    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, num_warps=1,
                          compile_mode="simd", enable_auto_blockify=False,
                          auto_simt_scope_mode="off")
        torch.npu.synchronize()
    freq_mhz, query_seconds = freq_check(args.device, busy_out)
    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, num_warps=1,
                          compile_mode="simd", enable_auto_blockify=False,
                          auto_simt_scope_mode="off")
        torch.npu.synchronize()
    reference_witness_ns = measure_event_us(
        lambda: busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024,
                                  num_warps=1, compile_mode="simd",
                                  enable_auto_blockify=False,
                                  auto_simt_scope_mode="off")) * 1e6

    cases = []
    t0 = time.monotonic()
    for block in args.blocks:
        for stride in args.strides:
            warps = [1] if args.mode == "simd" else args.num_warps
            for num_warps in warps:
                case = run_case(args.mode, block, stride, num_warps, buffer,
                                busy_out, args.step, args.mask, args.iters,
                                args.reps, args.device, reference_witness_ns,
                                args.witness_tol, args.max_attempts,
                                args.busy_iters, args.max_spread)
                cases.append(case)
                print(f'{args.mode} b{block} s{stride} W{num_warps}: '
                      f'{case["target_ns"]:.3f} ns/iter valid={case["valid"]} '
                      f'witness={[round(a["post_witness_ns"]/1e3, 2) for a in case["attempts"]]}us',
                      flush=True)
    elapsed = time.monotonic() - t0
    valid = sum(1 for c in cases if c["valid"])
    payload = {
        "mode": args.mode,
        "cases": cases,
        "frequency_check": {
            "inflight_freq_mhz": freq_mhz,
            "query_seconds": query_seconds,
            "reference_witness_ns": reference_witness_ns,
            "witness_tolerance": args.witness_tol,
        },
        "elapsed_seconds": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.out} valid={valid}/{len(cases)} "
          f"freq_check={freq_mhz}MHz query={query_seconds:.3f}s "
          f"ref_witness={reference_witness_ns/1e3:.2f}us elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
