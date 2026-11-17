#!/usr/bin/env python3
"""Microbenchmark where one target operation dominates the arithmetic.

Per element: one load, ONE target op, one store.  The target op is the only
arithmetic in the loop, so the kernel is priced almost entirely by that op.
--op selects exp / log / rem / div, --route forces simd or simt_only, and the
cost-model report (route + predicted us) is dumped via auto_simt_scope_dump.
Kernel duration comes from the NPU profiler.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from profile_utils import profile_kernel


@triton.jit
def op_kernel(x_ptr, d_ptr, out_ptr, n_elements, n_blocks, BLOCK_SIZE: tl.constexpr,
              OP: tl.constexpr, UNSIGNED: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        if OP == 0:          # exp
            r = tl.exp(x)
        elif OP == 1:        # log
            r = tl.log(x)
        elif OP == 2:        # div
            r = x / tl.load(d_ptr + (offsets % 7), mask=mask)
        else:                # rem
            d = tl.load(d_ptr + (offsets % 7), mask=mask)
            if UNSIGNED:
                r = (x.to(tl.uint32, bitcast=True) % d.to(tl.uint32, bitcast=True)).to(tl.int32, bitcast=True)
            else:
                r = x % d
        if OP >= 2:
            tl.store(out_ptr + offsets, r, mask=mask)
        else:
            tl.store(out_ptr + offsets, r, mask=mask)


OP_NAMES = ("exp", "log", "div", "rem")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("auto", "simd", "simt_only"), required=True)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--op", choices=OP_NAMES, required=True)
    ap.add_argument("--unsigned", action="store_true")
    ap.add_argument("--n-elements", type=int, default=1 << 20)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--profile-dir", type=Path)
    args = ap.parse_args()
    opid = OP_NAMES.index(args.op)
    is_int = args.op in ("rem", )

    torch.manual_seed(0)
    if is_int:
        x = torch.randint(-2**31, 2**31 - 1, (args.n_elements, ), dtype=torch.int32, device="npu")
        d = torch.tensor([97, 101, 103, 107, 109, 113, 127], dtype=torch.int32, device="npu")
        out = torch.zeros(args.n_elements, dtype=torch.int32, device="npu")
    else:
        base = torch.rand(args.n_elements, dtype=torch.float32, device="npu")
        x = (base + 0.5) if args.op in ("exp", ) else (base + 1.5)   # exp: [0.5,1.5]; log: [1.5,2.5]
        d = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=torch.float32, device="npu")
        out = torch.zeros(args.n_elements, dtype=torch.float32, device="npu")

    n_blocks = (args.n_elements + args.block_size - 1) // args.block_size
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(n_blocks, vector_cores)

    def launch() -> None:
        op_kernel[(grid, )](
            x, d, out, args.n_elements, n_blocks,
            BLOCK_SIZE=args.block_size, OP=opid, UNSIGNED=bool(args.unsigned),
            num_warps=args.num_warps,
            compile_mode=("simd_simt" if args.mode == "auto" else args.mode), superblock_factor=args.factor,
            logical_program_count_hint=grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch(); torch.npu.synchronize()
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(args.batch):
            launch()
        end.record(); torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000.0 / args.batch)
    samples.sort()
    prof = profile_kernel(launch, args.profile_dir, "op_kernel") if args.profile_dir else None
    print(json.dumps({
        "case": "%s_%s_n%d" % (args.op, "uns" if args.unsigned else "sgn", args.n_elements),
        "route_requested": args.mode,
        "num_warps": args.num_warps,
        "grid": grid, "logical_programs": n_blocks, "vector_cores": vector_cores,
        "elapsed_us": {"median": statistics.median(samples), "minimum": samples[0]},
        "report": str(args.report) if args.report else None,
        "profiler": prof,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
