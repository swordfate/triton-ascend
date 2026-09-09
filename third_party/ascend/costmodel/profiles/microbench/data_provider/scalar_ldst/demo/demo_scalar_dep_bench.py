#!/usr/bin/env python3
"""Uniform dependent scalar load/store marginal benchmark.

Addresses are random (precomputed random permutation) and each iteration's
address depends on the value loaded by the previous iteration.
"""
import argparse
import statistics

import torch
import torch_npu
import triton
import triton.language as tl

DEFAULT_CORES = 56
DEFAULT_FREQ_MHZ = 988.9


@triton.jit
def dep_load_kernel(next_ptr, start_ptr, out, N: tl.constexpr):
    pid = tl.program_id(0)
    cur = tl.load(start_ptr + pid)
    s = 0
    for i in tl.static_range(N):
        cur = tl.load(next_ptr + cur)
        s += cur
    tl.store(out + pid, s)


@triton.jit
def dep_store_kernel(next_ptr, data_ptr, start_ptr, out, N: tl.constexpr):
    pid = tl.program_id(0)
    cur = tl.load(start_ptr + pid)
    s = 0
    for i in tl.static_range(N):
        cur = tl.load(next_ptr + cur)
        tl.store(data_ptr + cur, s + i)
        s += cur
    tl.store(out + pid, s)


def launch_opts(mode, sf, grid):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": 2048,
        }
    return {
        "num_warps": 1,
        "compile_mode": "simt_only",
        "auto_simt_scope_mode": "off",
        "enable_auto_blockify": True,
        "superblock_factor": sf,
        "logical_program_count_hint": 2048,
    }


def make_launch(variant, grid, ops, mode, sf):
    total = max(1, grid * ops)
    next_ptr = torch.randperm(total, dtype=torch.int32).npu()
    start_ptr = torch.randint(0, total, (grid,), dtype=torch.int32).npu()
    data = torch.zeros(total, dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    opts = launch_opts(mode, sf, grid)
    kernel = dep_load_kernel if variant == "depl" else dep_store_kernel

    def launch():
        if variant == "depl":
            kernel[(grid,)](next_ptr, start_ptr, out, N=ops, **opts)
        else:
            kernel[(grid,)](next_ptr, data, start_ptr, out, N=ops, **opts)

    return launch


def measure_paired_delta_ms(launch_n, launch_0, reps):
    for _ in range(3):
        launch_n()
        launch_0()
    torch.npu.synchronize()
    deltas = []
    for _ in range(reps):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch_n()
        en.record()
        torch.npu.synchronize()
        tn = st.elapsed_time(en)
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch_0()
        en.record()
        torch.npu.synchronize()
        t0 = st.elapsed_time(en)
        deltas.append(tn - t0)
    return statistics.median(deltas)


def theil_sen_slope(points):
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx:
                slopes.append(dy / dx)
    return statistics.median(slopes) if slopes else 0.0


def fit_per_op(variant, ops, mode, sf, grids, reps):
    points = []
    for grid in grids:
        ln = make_launch(variant, grid, ops, mode, sf)
        l0 = make_launch(variant, grid, 0, mode, sf)
        points.append((grid, measure_paired_delta_ms(ln, l0, reps)))
    slope = theil_sen_slope(points)  # ms per logical grid
    cycles = slope * DEFAULT_CORES * DEFAULT_FREQ_MHZ * 1000.0
    return cycles / ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=4)
    ap.add_argument("--grids", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    print("# uniform dependent scalar load/store marginal cycles per op")
    print("variant,mode,sf,cycles_per_op")
    for variant in ["depl", "deps"]:
        for mode, sf in [("simd", 1), ("simt_only", 1), ("simt_only", 4)]:
            value = fit_per_op(variant, args.ops, mode, sf, args.grids, args.reps)
            print(f"{variant},{mode},{sf},{value:.3f}", flush=True)


if __name__ == "__main__":
    main()
