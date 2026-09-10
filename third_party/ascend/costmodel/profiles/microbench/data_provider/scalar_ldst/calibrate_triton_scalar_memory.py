#!/usr/bin/env python3
"""All-Triton scalar memory calibration for SIMD / SIMT F1/F2/F4.

This is a comparison-oriented replacement for the CCE-derived SIMT scalar
memory throughput.  It keeps the same Triton kernel for every mode, so the
only difference is the backend path selected by the compiler options.

Outputs marginal cycles per scalar op from paired N=0 / N=ops grid sweeps.
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
def scalar_mem_kernel(a, b, off, out, N_LD: tl.constexpr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    for i in tl.static_range(N_LD):
        idx = tl.load(off + pid * N_LD + i)
        s += tl.load(a + idx)
    for i in tl.static_range(N_ST):
        idx = tl.load(off + pid * N_ST + i)
        tl.store(b + idx, s + i)
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


def make_launch(variant, grid, ops, mode, sf):
    nld = {"load": ops, "store": 0}[variant]
    nst = {"load": 0, "store": ops}[variant]
    n = max(1, max(nld, nst))
    a = torch.randint(0, 100, (grid * max(nld, 1),), dtype=torch.int32, device="npu")
    b = torch.zeros(grid * max(nst, 1), dtype=torch.int32, device="npu")
    # Pre-generate random offsets as 64B-spaced slots (16 int32 elements).
    total = grid * n
    n_slots = max(1, total // 16)
    off = (torch.randint(0, n_slots, (total,), dtype=torch.int32, device="npu") * 16)
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    opts = launch_opts(mode, sf, grid)

    def launch():
        scalar_mem_kernel[(grid,)](a, b, off, out, N_LD=nld, N_ST=nst, **opts)

    return launch


def measure(launch, reps, warmup=3):
    for _ in range(warmup):
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


def paired_delta(launch_n, launch_0, reps):
    for _ in range(3):
        launch_n()
        launch_0()
    torch.npu.synchronize()
    deltas = []
    for _ in range(reps):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record(); launch_n(); en.record(); torch.npu.synchronize()
        tn = st.elapsed_time(en)
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record(); launch_0(); en.record(); torch.npu.synchronize()
        t0 = st.elapsed_time(en)
        deltas.append(tn - t0)
    return statistics.median(deltas)


def theil_sen(points):
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx:
                slopes.append(dy / dx)
    return statistics.median(slopes) if slopes else 0.0


def per_op_cycles(variant, ops, mode, sf, grids, reps):
    pts = []
    for grid in grids:
        ln = make_launch(variant, grid, ops, mode, sf)
        l0 = make_launch(variant, grid, 0, mode, sf)
        pts.append((grid, paired_delta(ln, l0, reps)))
    slope = theil_sen(pts)
    # slope is ms per logical grid; convert to per-program cycles.
    cycles = slope * DEFAULT_CORES * DEFAULT_FREQ_MHZ * 1000
    return cycles / ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--grids", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    print("variant,mode,sf,ops,cycles_per_op")
    for variant in ["load", "store"]:
        for ops in args.ops:
            for mode, sf in [("simd", 1), ("simt_only", 1), ("simt_only", 2), ("simt_only", 4)]:
                value = per_op_cycles(variant, ops, mode, sf, args.grids, args.reps)
                print(f"{variant},{mode},{sf},{ops},{value:.3f}", flush=True)


if __name__ == "__main__":
    main()
