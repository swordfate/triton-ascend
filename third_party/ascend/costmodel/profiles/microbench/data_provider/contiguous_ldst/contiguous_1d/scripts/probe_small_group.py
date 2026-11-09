#!/usr/bin/env python3
"""Sub-line same-shape load group calibration (u-amortization).

Real kernels issue groups of u same-shape small loads per iteration (the
4-way discrete-mask loads, the 4x64B indirect-index loads).  The small and
lines<=2 segments were calibrated at u=1 (isolated latency), so the model
charges u x full latency while the group shares the LSU issue rounds: a
4x64B SIMT group measures flat ~120ns in N.

Kernel: per iteration, u loads of BLOCK elements from u distinct
BLOCK-strided segments of one flat tensor (window rotation by step & MASK),
accumulated elementwise.  cost = two-trip slope = ns per GROUP of u loads.
The elementwise add of a sub-line block biases each load by ~1-3ns; the
u=1 column anchors against the independently calibrated segments.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
      "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}


@triton.jit(do_not_specialize=["iters"])
def group_loads(src, out, iters, BLOCK: tl.constexpr, U: tl.constexpr,
                MASK: tl.constexpr):
    program = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        base = (step & MASK) * (U * BLOCK)
        for u in tl.static_range(U):
            acc += tl.load(src + base + u * BLOCK + lane)
    tl.store(out + program * BLOCK + lane, acc)


def slope_ns(launch, lo, hi, reps, trace_dir):
    launches = [(launch, lo)] * reps + [(launch, hi)] * reps
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1,
                                                 active=len(launches)),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(trace_dir))) as prof:
        for fn, trip in launches:
            fn(trip)
            prof.step()
    torch.npu.synchronize()
    durations = []
    for path in sorted(trace_dir.rglob("kernel_details.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)") and "group_loads" in (
                        row.get("Name") or ""):
                    durations.append(float(row["Duration(us)"]))
    if len(durations) < 2:
        return None
    half = len(durations) // 2
    lo_vals, hi_vals = durations[:half], durations[half:]
    if not lo_vals or not hi_vals:
        return None
    return (min(hi_vals) - min(lo_vals)) * 1000.0 / (hi - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtypes", default="fp16,fp32,int64")
    args = parser.parse_args()
    lo, hi, reps, window = 1024, 4096, 4, 64
    mask = window - 1
    out = open(args.out, "a")
    count = 0
    for dtype in args.dtypes.split(","):
        eb = EB[dtype]
        for by in (32, 64, 128, 256):
            block = by // eb
            if block < 1:
                continue
            n = window * 8 * block + 8 * block
            src = (torch.rand(n) * 7).to(DT[dtype]).to("npu")
            dst = torch.zeros(n, dtype=DT[dtype], device="npu")
            for warps in (1, 4, 16):
                for u in (1, 2, 3, 4, 6, 8):
                    opts = dict(compile_mode="simt_only", num_warps=warps,
                                superblock_factor=1)
                    fa = lambda t: group_loads[(1,)](
                        src, dst[:block].contiguous(), t, block, u, mask,
                        **opts)
                    for _ in range(3):
                        fa(lo)
                    torch.npu.synchronize()
                    cost = slope_ns(fa, lo, hi, reps, Path("/tmp/sg"))
                    rec = dict(v=1, mode="simt_only", dir="load", dtype=dtype,
                               eb=eb, W=warps, u=u, by=by, block=block,
                               lo=lo, hi=hi, window=window,
                               cost=cost,
                               status="ok" if cost and cost > 0 else "bad")
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    count += 1
                    if count % 24 == 0:
                        print("... %d pts" % count, flush=True)
    print("DONE %d pts" % count, flush=True)


if __name__ == "__main__":
    main()
