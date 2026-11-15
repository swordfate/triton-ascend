#!/usr/bin/env python3
"""te>=8 SIMT stream (loop-carried / pipelined) calibration, v2 methodology.

Replaces the accumulator-contaminated load probe.  The old body

    acc += tl.load(src + base + (step & MASK) * BLOCK + lane)

carries a te-wide accumulator whose live footprint (acc + load temp) crosses
the 127-register budget at te>=64 (4B lanes) / te>=128 (2B lanes): the loop
then spills on every iteration, and the two-trip slope folded those spills
plus the elementwise add into the "memory" table.  The step regime was the
probe, not the memory system.

Paired control.  Kernel B keeps the same accumulator shape and the same
per-iteration scalar work, replacing only the block load with a tiny
8-element table load whose address depends on the loop index, so neither
LICM nor a closed form can collapse the loop:

    A:  acc += tl.load(src + base + (step & MASK) * BLOCK + lane)
    B:  acc += value + tl.load(scalars + (step & 7))

    cost_per_op = slope(A) - slope(B),  slope over trips 1024 -> 4096

Stores keep the direct slope: a loop of back-to-back stores sustains the
drain rate (the store queue fills and throttles the loop), which is exactly
the per-op cost the stream pricing needs.  No control is required.

    A:  tl.store(dst + base + (step & MASK) * BLOCK + lane, value + step)

Grid: dtypes x {load,store} x W{1..64} x te{8..512} with BLOCK = 32*W*te
capped at 16384 elements, one program, rotating 1024-block address window.
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
def ld_stream(src, scalars, out, iters, BLOCK: tl.constexpr,
              MASK: tl.constexpr):
    program = tl.program_id(0)
    base = program * (MASK + 1) * BLOCK
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + base + (step & MASK) * BLOCK + lane)
    tl.store(out + program * BLOCK + lane, acc)


@triton.jit(do_not_specialize=["iters"])
def ld_control(scalars, out, iters, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 3, out.dtype.element_ty)
    acc = tl.zeros((BLOCK,), out.dtype.element_ty)
    for step in range(iters):
        acc += value + tl.load(scalars + (step & 7))
    tl.store(out + program * BLOCK + lane, acc)


@triton.jit(do_not_specialize=["iters"])
def st_stream(dst, iters, BLOCK: tl.constexpr, MASK: tl.constexpr):
    program = tl.program_id(0)
    base = program * (MASK + 1) * BLOCK
    lane = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 1, dst.dtype.element_ty)
    for step in range(iters):
        tl.store(dst + base + (step & MASK) * BLOCK + lane, value + step)


def paired_slopes(name_a, name_b, fa, fb, lo, hi, reps, trace_dir):
    """One profiler session per point: [A,B]x reps at LO then at HI.

    Returns (slope_a, slope_b) in ns per operation, using the per-name
    session minima at each trip count.  The chronological row split lands
    the LO launches in the first half and the HI launches in the second.
    """
    launches = []
    for trip in (lo, hi):
        for _ in range(reps):
            launches.append((fa, trip))
            if fb is not None:
                launches.append((fb, trip))
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
    rows = []
    for path in sorted(trace_dir.rglob("kernel_details.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)"):
                    rows.append((row.get("Name") or "",
                                 float(row["Duration(us)"])))
    if not rows:
        return None, None

    def slope(match):
        seq = [d for name, d in rows if match in name]
        if not seq:
            return None
        half = len(seq) // 2
        lo_vals, hi_vals = seq[:half], seq[half:]
        if not lo_vals or not hi_vals:
            return None
        return (min(hi_vals) - min(lo_vals)) * 1000.0 / (hi - lo)

    return slope(name_a), (slope(name_b) if fb is not None else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtypes", default="int8,fp16,fp32,int64")
    parser.add_argument("--dirs", default="load,store")
    args = parser.parse_args()
    lo, hi, reps = 1024, 4096, 4
    window = 1024
    mask = window - 1
    out = open(args.out, "a")
    count = 0
    for dtype in args.dtypes.split(","):
        eb = EB[dtype]
        scalars = (torch.arange(8) % 5).to(DT[dtype]).to("npu")
        for direction in args.dirs.split(","):
            for warps in (1, 2, 4, 8, 16, 32, 64):
                for te in (8, 16, 32, 64, 128, 256, 512):
                    block = int(te * 32 * warps)
                    if block > 16384:
                        continue
                    n = window * block + block
                    src = (torch.rand(n) * 7).to(DT[dtype]).to("npu")
                    dst = torch.zeros(n, dtype=DT[dtype], device="npu")
                    head = dst[:block].contiguous()
                    opts = dict(compile_mode="simt_only", num_warps=warps,
                                superblock_factor=1)
                    rec = dict(v=2, mode="simt_only", dir=direction,
                               dtype=dtype, eb=eb, W=warps, te=te, block=block,
                               by=block * eb, lo=lo, hi=hi, window=window)
                    try:
                        if direction == "load":
                            fa = lambda t: ld_stream[(1,)](
                                src, scalars, head, t, block, mask, **opts)
                            fb = lambda t: ld_control[(1,)](
                                scalars, head, t, block, **opts)
                            slope_a, slope_b = paired_slopes(
                                "ld_stream", "ld_control", fa, fb, lo, hi,
                                reps, Path("/tmp/te8_pair"))
                        else:
                            fa = lambda t: st_stream[(1,)](
                                dst, t, block, mask, **opts)
                            slope_a, slope_b = paired_slopes(
                                "st_stream", None, fa, None, lo, hi,
                                reps, Path("/tmp/te8_pair"))
                        cost = None
                        if slope_a is not None and slope_b is not None:
                            cost = slope_a - slope_b
                        rec.update(slope_a=slope_a, slope_b=slope_b,
                                   cost=cost,
                                   status="ok" if cost and cost > 0 else "bad")
                    except Exception as exc:  # noqa: BLE001
                        rec.update(status="err", note=str(exc)[:200])
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    count += 1
                    if count % 20 == 0:
                        print("... %d pts" % count, flush=True)
    print("DONE %d pts" % count, flush=True)


if __name__ == "__main__":
    main()
