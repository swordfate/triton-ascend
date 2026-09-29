#!/usr/bin/env python3
"""Board-side event-timing cross-check for the strided-load probe.

The preferred measurement is ``torch_npu.profiler`` (``kernel_details.csv``),
but on the shared Ascend950PR server other users' ``msprof`` sessions hold the
hardware profiling lock, so ``torch_npu.profiler`` returns no task data.  This
script therefore uses the fallback:

  * launch the load kernel and a structurally-matched no-load kernel once each;
  * wrap each single launch with ``torch.npu.Event`` on the current stream;
  * repeat with rotating input buffers (``NBUF``) and alternating order;
  * report the median/robust-mean of the paired difference ``t_load - t_noload``.

The absolute event times are dominated by launch/event overhead (~10-20 us for
one tiny kernel).  The paired difference cancels that fixed overhead and is the
board estimate of the extra device time caused by the load.  It is not a
replacement for a clean ``kernel_details.csv`` measurement.
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK) * STRIDE
    v = tl.load(x_ptr + pid * BLOCK * STRIDE + offs)
    tl.store(out_ptr + pid, tl.sum(v, axis=0))


@triton.jit
def noload_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.arange(0, BLOCK).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v, axis=0))


def opts_for(mode, grid):
    common = {
        "num_warps": 1,
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "auto_simt_scope_mode": "off",
    }
    if mode == "simd":
        return {**common, "compile_mode": "simd", "enable_auto_blockify": False}
    return {**common, "compile_mode": "simt_only", "enable_auto_blockify": True}


def measure_us(launch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0  # us


def run_case(mode, block, stride, grid, reps, nbuf, baseline="noload"):
    opts = opts_for(mode, grid)
    total = grid * block * max(1, stride) + 4096
    xs = [(torch.arange(total, dtype=torch.float32) % 7.0).contiguous().to("npu")
          for _ in range(nbuf)]
    out = torch.zeros(grid, dtype=torch.float32).to("npu")
    load_kernel.warmup(xs[0], out, BLOCK=block, STRIDE=stride, grid=(grid,), **opts)
    if baseline == "noload":
        noload_kernel.warmup(out, BLOCK=block, grid=(grid,), **opts)
    for _ in range(5):
        load_kernel[(grid,)](xs[0], out, BLOCK=block, STRIDE=stride, **opts)
        if baseline == "noload":
            noload_kernel[(grid,)](out, BLOCK=block, **opts)
    torch.npu.synchronize()

    loads, noloads, deltas = [], [], []
    for i in range(reps):
        x = xs[i % nbuf]
        if baseline == "noload":
            base_launch = lambda: noload_kernel[(grid,)](out, BLOCK=block, **opts)
        else:
            base_launch = lambda: load_kernel[(grid,)](x, out, BLOCK=block, STRIDE=0, **opts)
        if i % 2 == 0:
            tl_us = measure_us(lambda: load_kernel[(grid,)](x, out, BLOCK=block, STRIDE=stride, **opts))
            nl_us = measure_us(base_launch)
        else:
            nl_us = measure_us(base_launch)
            tl_us = measure_us(lambda: load_kernel[(grid,)](x, out, BLOCK=block, STRIDE=stride, **opts))
        loads.append(tl_us)
        noloads.append(nl_us)
        deltas.append(tl_us - nl_us)
    deltas_sorted = sorted(deltas)
    trim = deltas_sorted[len(deltas_sorted) // 10:9 * len(deltas_sorted) // 10]
    return {
        "load_median_us": statistics.median(loads),
        "noload_median_us": statistics.median(noloads),
        "delta_median_us": statistics.median(deltas),
        "delta_trim10_mean_us": statistics.mean(trim),
        "delta_p10_us": deltas_sorted[len(deltas_sorted) // 10],
        "delta_p90_us": deltas_sorted[9 * len(deltas_sorted) // 10],
        "delta_min_us": deltas_sorted[0],
        "delta_max_us": deltas_sorted[-1],
        "reps": reps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--blocks", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--nbuf", type=int, default=256)
    ap.add_argument("--baseline", choices=["noload", "uniform"], default="uniform")
    ap.add_argument("--out", type=Path, default=Path("board_event_results"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for block in args.blocks:
        for mode in args.modes:
            for stride in args.strides:
                result = run_case(mode, block, stride, args.grid, args.reps, args.nbuf, args.baseline)
                row = {"mode": mode, "block": block, "stride": stride, "grid": args.grid, "baseline": args.baseline}
                row.update(result)
                rows.append(row)
                print(f'{mode} b{block} s{stride}: delta={result["delta_median_us"]:.3f} us '
                      f'(trim={result["delta_trim10_mean_us"]:.3f}, '
                      f'p10..p90={result["delta_p10_us"]:.3f}..{result["delta_p90_us"]:.3f})',
                      flush=True)
    (args.out / "board_event_results.json").write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    with (args.out / "board_event_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
