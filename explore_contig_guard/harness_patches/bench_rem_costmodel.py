#!/usr/bin/env python3
"""Microbenchmark where the tensor remainder dominates the arithmetic.

One load, one multiply-add, one rem, one store per element.  The --route flag
forces the route so both SIMD and SIMT lowering of arith.remsi/remui can be
priced.  The cost-model report (route + predicted us) is dumped through
auto_simt_scope_dump, and the kernel duration comes from the NPU profiler.
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
def rem_kernel(x_ptr, d_ptr, out_ptr, n_elements, n_blocks, BLOCK_SIZE: tl.constexpr,
               UNSIGNED: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        d = tl.load(d_ptr + (offsets % 7), mask=mask)
        if UNSIGNED:
            xu = x.to(tl.uint32, bitcast=True)
            r = xu % d.to(tl.uint32, bitcast=True)
            tl.store(out_ptr + offsets, r.to(tl.int32, bitcast=True), mask=mask)
        else:
            r = x % d
            tl.store(out_ptr + offsets, r, mask=mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("simd", "simt_only"), required=True)
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

    torch.manual_seed(0)
    x = torch.randint(-2**31, 2**31 - 1, (args.n_elements, ), dtype=torch.int32, device="npu")
    d = torch.tensor([97, 101, 103, 107, 109, 113, 127], dtype=torch.int32, device="npu")
    out = torch.zeros(args.n_elements, dtype=torch.int32, device="npu")
    n_blocks = (args.n_elements + args.block_size - 1) // args.block_size
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(n_blocks, vector_cores)

    def launch() -> None:
        rem_kernel[(grid, )](
            x, d, out, args.n_elements, n_blocks,
            BLOCK_SIZE=args.block_size, UNSIGNED=bool(args.unsigned),
            num_warps=4, compile_mode=args.route, superblock_factor=args.factor,
            logical_program_count_hint=grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch(); torch.npu.synchronize()
    if args.unsigned:
        expected = torch.remainder(x, d[torch.arange(args.n_elements, device="npu") % 7])
        expected = torch.where(expected < 0, expected + 2**32, expected).to(torch.int32)
    else:
        expected = torch.remainder(x, d[torch.arange(args.n_elements, device="npu") % 7])
    max_abs = int((out - expected).abs().max().item())
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
    prof = profile_kernel(launch, args.profile_dir, "rem_kernel") if args.profile_dir else None
    print(json.dumps({
        "case": "rem_%s_n%d" % ("uns" if args.unsigned else "sgn", args.n_elements),
        "route_requested": args.route,
        "n_elements": args.n_elements,
        "block_size": args.block_size,
        "grid": grid,
        "logical_programs": n_blocks,
        "vector_cores": vector_cores,
        "max_abs": max_abs,
        "elapsed_us": {"median": statistics.median(samples), "minimum": samples[0]},
        "report": str(args.report) if args.report else None,
        "profiler": prof,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
