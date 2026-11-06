#!/usr/bin/env python3
"""Pure contiguous probe: a one-dimensional block, no reduction and no modulo.

Two things were wrong with the earlier probe and are fixed here.

1. The rotate loop wrapped with `step % window`.  A remainder is an integer division,
   which on the vector path costs far more than the add it replaces, and it sat inside
   the loop that is supposed to contain nothing but the transfer.  `window` is now a
   power of two and the wrap is `step & (window - 1)`.

2. The load was consumed with `tl.sum(block, axis=0)`.  Any consumption costs something;
   the cheapest one that keeps the accumulator inside the register budget is an
   elementwise accumulate over a one-dimensional block, which is what the strided
   calibration uses as well.  There is no reduction anywhere.

The loop body is therefore exactly:

    acc += tl.load(source + (step & MASK) * BLOCK + lane)

which is one contiguous block load plus one add per element, with no division and no
reduction.
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


@triton.jit(do_not_specialize=["iters"])
def block_kernel(source, output, iters, BLOCK: tl.constexpr,
                 MASK: tl.constexpr, direction: tl.constexpr):
    program = tl.program_id(0)
    base = program * (MASK + 1) * BLOCK
    lane = tl.arange(0, BLOCK)
    if direction == 0:
        acc = tl.zeros((BLOCK, ), source.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(source + base + (step & MASK) * BLOCK + lane)
        tl.store(output + program * BLOCK + lane, acc)
    else:
        value = tl.full((BLOCK, ), 1, output.dtype.element_ty)
        for step in range(iters):
            tl.store(output + base + (step & MASK) * BLOCK + lane, value + step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--direction", choices=("load", "store"), required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument("--block", type=int, required=True, help="elements per operation")
    parser.add_argument("--warps", type=int, required=True)
    parser.add_argument("--k-low", type=int, default=2048)
    parser.add_argument("--k-high", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--dump-asm", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    window = args.k_low                      # power of two, so the wrap is an AND
    assert window & (window - 1) == 0, "window must be a power of two"
    mask = window - 1
    block = args.block
    dtype = DTYPES[args.dtype]
    torch.manual_seed(23)
    n = window * block + block
    source = (torch.rand(n, dtype=torch.float32) * 7).to(dtype).to("npu")
    output = torch.zeros(n, dtype=dtype, device="npu")
    direction = 0 if args.direction == "load" else 1
    options = {"compile_mode": args.mode, "num_warps": args.warps}
    if args.mode == "simt_only":
        options["superblock_factor"] = 1

    def launch(trips):
        return block_kernel[(1, )](source, output, trips, block, mask, direction,
                                   **options)

    handle = launch(args.k_low)
    torch.npu.synchronize()
    if args.dump_asm:
        for key in ("ttir", "ttadapter", "source"):
            text = handle.asm.get(key)
            if isinstance(text, str):
                print(json.dumps({"asm": key, "chars": len(text),
                                  "remsi": text.count("remsi"),
                                  "remainder": text.count(" % "),
                                  "reduce": text.count("reduce"),
                                  "sum": text.count("sum"),
                                  "divsi": text.count("divsi"),
                                  "load": text.count("load")}))

    medians, spread = {}, 0.0
    for trips in (args.k_low, args.k_high):
        launch(trips)
        torch.npu.synchronize()
        trace = Path("/tmp/block_traces") / ("%s_%s_%s_%d_%d_%d"
                                             % (args.mode, args.direction, args.dtype,
                                                block, args.warps, trips))
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
                    if row.get("Duration(us)") and "block_kernel" in (row.get("Name") or ""):
                        durations.append(float(row["Duration(us)"]))
        if not durations:
            record = {"status": "fail", "reason": "no kernel rows", "tag": "blk"}
            with args.manifest.open("a") as stream:
                stream.write(json.dumps(record) + chr(10))
            print(json.dumps(record))
            return
        medians[trips] = min(durations)
        spread = max(spread, (max(durations) - min(durations)) / min(durations))

    slope = (medians[args.k_high] - medians[args.k_low]) * 1000.0 / (
        args.k_high - args.k_low)
    record = {
        "status": "ok", "mode": args.mode, "direction": args.direction,
        "dtype": args.dtype, "rows": 1, "cols": block,
        "bytes": block * ELEMENT_BYTES[args.dtype],
        "warps": args.warps, "iters": [args.k_low, args.k_high],
        "window_blocks": window, "min_us": medians,
        "cost_per_operation": slope, "spread": spread,
        "tag": "blk_%s_%s_%s_b%d_w%d" % (args.mode, args.direction, args.dtype,
                                         block * ELEMENT_BYTES[args.dtype], args.warps),
    }
    with args.manifest.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + chr(10))
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
