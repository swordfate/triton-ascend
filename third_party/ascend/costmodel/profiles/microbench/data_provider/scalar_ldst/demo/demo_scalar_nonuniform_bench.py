#!/usr/bin/env python3
"""Non-uniform scalar load SIMD vs SIMT F1/F4 marginal benchmark.

Each program loads OPS scalar indices from an index tensor and then loads
OPS scalar values from data_ptr + idx.  This models indirect/non-uniform
scalar loads.
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
def nonuniform_load_kernel(idx, data, out, N: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    for i in tl.static_range(N):
        j = tl.load(idx + pid * N + i)
        s += tl.load(data + j)
    tl.store(out + pid, s)


def launch_opts(mode, sf, grid):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    return {
        "num_warps": 1,
        "compile_mode": "simt_only",
        "auto_simt_scope_mode": "off",
        "enable_auto_blockify": True,
        "superblock_factor": sf,
        "logical_program_count_hint": grid,
    }


def measure(launch, reps):
    for _ in range(3):
        launch()
    torch.npu.synchronize()
    times = []
    for _ in range(reps):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch()
        en.record()
        torch.npu.synchronize()
        times.append(st.elapsed_time(en))
    return statistics.median(times)


def fit_slope(grids, times):
    n = len(grids)
    mg = sum(grids) / n
    mt = sum(times) / n
    cov = sum((g - mg) * (t - mt) for g, t in zip(grids, times))
    var = sum((g - mg) ** 2 for g in grids)
    return cov / var if var else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=4)
    ap.add_argument("--grids", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    grids = args.grids
    print("# non-uniform scalar load marginal cycles/op")
    for mode, sf in [("simd", 1), ("simt_only", 1), ("simt_only", 4)]:
        points = []
        for grid in grids:
            idx = torch.randint(0, grid * args.ops, (grid * args.ops,), dtype=torch.int32, device="npu")
            data = torch.randn(grid * args.ops, device="npu")
            out = torch.zeros(grid, dtype=torch.float32, device="npu")
            opts = launch_opts(mode, sf, grid)

            def launch():
                nonuniform_load_kernel[(grid,)](idx, data, out, N=args.ops, **opts)

            t = measure(launch, args.reps)
            points.append((grid, t))
        slope = fit_slope(grids, [t for _, t in points])
        per_prog_ms = slope * DEFAULT_CORES
        per_op_cycles = per_prog_ms * DEFAULT_FREQ_MHZ * 1000 / args.ops
        print(f"{mode},{sf},{per_op_cycles:.3f}", flush=True)


if __name__ == "__main__":
    main()
