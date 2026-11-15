#!/usr/bin/env python3
"""Card health check for the contiguous calibration probes.

Why this exists: on a shared machine the NPU can end up running ~20x slower than
normal (another tenant saturating the device, a leaked device context left behind by
a killed process, or the card sitting in an Alarm state).  Every measurement taken in
that state is silently wrong in the same self-consistent way: the per-record spread
stays under 0.1% and the k-linearity stays at 1.000, so no per-record gate can see it.

This script runs one fixed tiny kernel and prints cycles per iteration.  Run it before
and after a campaign; anything above ~2x the healthy value means the numbers taken in
between are not comparable with the rest of the calibration.

Healthy reference (Ascend 950PR, default clocks):  ~90-110 ns/iteration
Degraded example seen on 2026-10-31:                ~1250 ns/iteration  (20x)

Usage:
    ASCEND_RT_VISIBLE_DEVICES=1 python3 probe_speed.py
    python3 probe_speed.py --json          # machine readable
"""
import argparse, json, sys, time

import torch, torch_npu, triton
import triton.language as tl

HEALTHY_NS = 150.0      # above this the card is not producing usable numbers
DEGRADED_NS = 300.0


@triton.jit(do_not_specialize=["iters"])
def speed_kernel(src, dst, iters, BLOCK: tl.constexpr, WINMASK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + (step & WINMASK) * BLOCK + lane)
    tl.store(dst + lane, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--k-low", type=int, default=2048)
    ap.add_argument("--k-high", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    src = torch.zeros(1 << 20, dtype=torch.int8, device="npu")
    dst = torch.zeros(1 << 20, dtype=torch.int8, device="npu")

    def once(k):
        speed_kernel[(1,)](src, dst, k, args.block, args.k_low - 1)

    for _ in range(args.warmup):
        once(args.k_low)
    torch.npu.synchronize()

    best = {}
    for k in (args.k_low, args.k_high):
        mn = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            once(k)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1e6
            mn = dt if mn is None else min(mn, dt)
        best[k] = mn

    ratio = best[args.k_high] / best[args.k_low] if best[args.k_low] else 0.0
    expect = args.k_high / args.k_low
    # 线性度校验：k_high 的耗时应当约为 k_low 的 (k_high/k_low) 倍。
    # 没有这道门时，单次 t(k_high) 偏小就会把差值压扁，报出假的 healthy
    # （2026-10-31 14:04 就是这样误判了 0 号卡，让看护脚本在降频期开了采集）。
    linear = abs(ratio / expect - 1.0) <= 0.25
    ns_per_iter = (best[args.k_high] - best[args.k_low]) * 1000.0 / (
        args.k_high - args.k_low)

    if not linear:
        verdict = "invalid"
    elif ns_per_iter <= HEALTHY_NS:
        verdict = "healthy"
    elif ns_per_iter <= DEGRADED_NS:
        verdict = "degraded"
    else:
        verdict = "unusable"

    out = {"ns_per_iter": ns_per_iter, "verdict": verdict,
           "linearity": ratio / expect if expect else 0.0,
           "healthy_ns": HEALTHY_NS, "k_low_us": best[args.k_low],
           "k_high_us": best[args.k_high]}
    if args.json:
        print(json.dumps(out))
    else:
        print("block=%d  k=%d/%d  repeats=%d" % (args.block, args.k_low,
                                                 args.k_high, args.repeats))
        print("  k_low %.1f us   k_high %.1f us" % (best[args.k_low], best[args.k_high]))
        print("  %.1f ns/iteration   -> %s (healthy <= %.0f, unusable > %.0f)"
              % (ns_per_iter, verdict, HEALTHY_NS, DEGRADED_NS))
    return 0 if verdict == "healthy" else 1


if __name__ == "__main__":
    sys.exit(main())
