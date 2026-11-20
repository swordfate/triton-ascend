#!/usr/bin/env python3
"""Masked block probe: does the cost depend on whether the true-count is known at compile time?

A prefix mask `arange < M` has two very different lowerings:

  static  (M is a constexpr)  ->  ttir: `arith.constant dense<M>`  - the compiler knows exactly
                                  how many lanes are live
  dynamic (M is a kernel arg) ->  ttir: `tt.splat %M`               - the count is unknown at
                                  compile time

This probe measures the same block with the same M both ways, so the difference between the two
rows is the cost of not knowing.  Everything else matches the contiguous probes: the rotate wrap
is `step & MASK` (power-of-two window), the load is consumed by an elementwise accumulate.
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

DTYPES = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
          "int64": torch.int64}
ELEMENT_BYTES = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}


def next_pow2(n):
    return 1 << max(0, (n - 1).bit_length())


@triton.jit
def kernel_static(source, output, iters, M: tl.constexpr, BLOCK: tl.constexpr,
                  WINMASK: tl.constexpr, direction: tl.constexpr):
    base = 0
    lane = tl.arange(0, BLOCK)
    keep = lane < M
    if direction == 0:
        acc = tl.zeros((BLOCK, ), source.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(source + base + (step & WINMASK) * BLOCK + lane,
                           mask=keep, other=0)
        tl.store(output + lane, acc, mask=keep)
    else:
        value = tl.full((BLOCK, ), 1, output.dtype.element_ty)
        for step in range(iters):
            tl.store(output + base + (step & WINMASK) * BLOCK + lane, value + step,
                     mask=keep)


@triton.jit(do_not_specialize=["iters", "M"])
def kernel_dyn(source, output, iters, M, BLOCK: tl.constexpr,
               WINMASK: tl.constexpr, direction: tl.constexpr):
    base = 0
    lane = tl.arange(0, BLOCK)
    keep = lane < M
    if direction == 0:
        acc = tl.zeros((BLOCK, ), source.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(source + base + (step & WINMASK) * BLOCK + lane,
                           mask=keep, other=0)
        tl.store(output + lane, acc, mask=keep)
    else:
        value = tl.full((BLOCK, ), 1, output.dtype.element_ty)
        for step in range(iters):
            tl.store(output + base + (step & WINMASK) * BLOCK + lane, value + step,
                     mask=keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--direction", choices=("load", "store"), required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--valid", type=int, required=True, help="true-count M")
    parser.add_argument("--dynamic", type=int, required=True, help="1 = M is a runtime arg")
    parser.add_argument("--warps", type=int, required=True)
    parser.add_argument("--k-low", type=int, default=2048)
    parser.add_argument("--k-high", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    window = args.k_low
    assert window & (window - 1) == 0
    winmask = window - 1
    block, m = args.block, args.valid
    assert m <= block
    dtype = DTYPES[args.dtype]
    torch.manual_seed(23)
    total = window * block + block
    source = (torch.rand(total, dtype=torch.float32) * 7).to(dtype).to("npu")
    output = torch.zeros(total, dtype=dtype, device="npu")
    direction = 0 if args.direction == "load" else 1
    options = {"compile_mode": args.mode, "num_warps": args.warps}
    if args.mode == "simt_only":
        options["superblock_factor"] = 1

    def launch(trips):
        if args.dynamic:
            return kernel_dyn[(1, )](source, output, trips, m, block, winmask,
                                     direction, **options)
        return kernel_static[(1, )](source, output, trips, m, block, winmask,
                                    direction, **options)

    medians, spread = {}, 0.0
    for trips in (args.k_low, args.k_high):
        launch(trips)
        torch.npu.synchronize()
        trace = Path("/tmp/mask_traces") / ("%s_%s_%s_b%d_m%d_d%d_w%d_%d"
                                            % (args.mode, args.direction, args.dtype,
                                               block, m, args.dynamic, args.warps, trips))
        if trace.exists():
            shutil.rmtree(trace)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=args.repeats),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace))
        ) as profiler:
            for _ in range(args.repeats):
                launch(trips)
                profiler.step()
        torch.npu.synchronize()
        durations = []
        for path in sorted(trace.rglob("kernel_details.csv")):
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("Duration(us)") and "kernel_" in (row.get("Name") or ""):
                        durations.append(float(row["Duration(us)"]))
        if not durations:
            with args.manifest.open("a") as stream:
                stream.write(json.dumps({"status": "fail", "tag": "mask"}) + chr(10))
            return
        medians[trips] = min(durations)
        spread = max(spread, (max(durations) - min(durations)) / min(durations))

    slope = (medians[args.k_high] - medians[args.k_low]) * 1000.0 / (
        args.k_high - args.k_low)
    rec = {"status": "ok", "mode": args.mode, "direction": args.direction,
           "dtype": args.dtype, "block": block, "valid": m, "dynamic": args.dynamic,
           "bytes_valid": m * ELEMENT_BYTES[args.dtype], "warps": args.warps,
           "iters": [args.k_low, args.k_high], "min_us": medians,
           "cost_per_operation": slope, "spread": spread,
           "tag": "msk2_%s_%s_%s_b%d_m%d_%s_w%d"
                  % (args.mode, args.direction, args.dtype, block, m,
                     "dyn" if args.dynamic else "sta", args.warps)}
    with args.manifest.open("a") as stream:
        stream.write(json.dumps(rec, sort_keys=True) + chr(10))
    print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
