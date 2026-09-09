#!/usr/bin/env python3
"""Unified scalar SIMD/SIMT benchmark with robust marginal measurement.

Covers:
    load-only
    store-only
    load+store

For each scenario and each execution mode (SIMD, SIMT F1, SIMT F4), it performs
a grid sweep and fits per-core per-program marginal cycles.  It uses paired
N=0/N=ops measurements and Theil-Sen slope to remove fixed launch/setup.

Run:
    source ~/env_ascend.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 demo_scalar_bench.py --ops 4 --grids 2048 4096 8192 16384 --reps 30
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
def scalar_ldst_kernel(a, b, ld_off, st_off, out,
                       N_LD: tl.constexpr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    # Pre-generated random addresses, read from offset tables like CCE probes.
    for i in tl.static_range(N_LD):
        off = tl.load(ld_off + pid * N_LD + i)
        s += tl.load(a + off)
    for i in tl.static_range(N_ST):
        off = tl.load(st_off + pid * N_ST + i)
        tl.store(b + off, s + i)
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
    nld = {"load": ops, "store": 0, "ldst": ops}[variant]
    nst = {"load": 0, "store": ops, "ldst": ops}[variant]
    a = torch.arange(grid * max(nld, 1), dtype=torch.int32, device="npu") % 7
    b = torch.zeros(grid * max(nst, 1), dtype=torch.int32, device="npu")
    ld_off = torch.randint(0, grid * max(nld, 1), (grid * max(nld, 1),),
                           dtype=torch.int32, device="npu")
    st_off = torch.randint(0, grid * max(nst, 1), (grid * max(nst, 1),),
                           dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    opts = launch_opts(mode, sf, grid)

    def launch():
        scalar_ldst_kernel[(grid,)](
            a, b, ld_off, st_off, out, N_LD=nld, N_ST=nst, **opts
        )

    return launch


def measure_paired_delta_ms(launch_n, launch_0, reps=30, warmup=3):
    """Return median((time(N) - time(0))) over paired launches on one grid."""
    for _ in range(warmup):
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
        t_n = st.elapsed_time(en)

        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch_0()
        en.record()
        torch.npu.synchronize()
        t_0 = st.elapsed_time(en)
        deltas.append(t_n - t_0)
    return statistics.median(deltas)


def theil_sen_slope(points):
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx != 0:
                slopes.append(dy / dx)
    if not slopes:
        return 0.0
    return statistics.median(slopes)

def fit_slope_ms_per_grid(points):
    return theil_sen_slope(points)


def slope_to_per_program_cycles(slope_ms_per_grid):
    # total time scales as (grid / cores) * per_program_time + fixed
    # slope_ms_per_grid = per_program_ms / cores
    per_program_ms = slope_ms_per_grid * DEFAULT_CORES
    return per_program_ms * DEFAULT_FREQ_MHZ * 1e3  # ms -> cycles at MHz

def fit_marginal_per_op_cycles(variant, ops, mode, sf, grids, reps):
    """Fit paired N=0 / N=ops marginal cycles, normalized per scalar op."""
    points = []
    for grid in grids:
        launch_n = make_launch(variant, grid, ops, mode, sf)
        launch_0 = make_launch(variant, grid, 0, mode, sf)
        delta_ms = measure_paired_delta_ms(launch_n, launch_0, reps)
        points.append((grid, delta_ms))
    slope = fit_slope_ms_per_grid(points)
    total_marginal_program_cycles = slope_to_per_program_cycles(slope)
    if variant == "ldst":
        return total_marginal_program_cycles / (2 * ops)
    return total_marginal_program_cycles / ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=32)
    ap.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
    )
    ap.add_argument("--min-grid", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    grids = [g for g in args.grids if g >= args.min_grid]
    if len(grids) < 2:
        raise SystemExit(
            f"Not enough grids >= --min-grid {args.min_grid}: {args.grids}"
        )
    if len(grids) < 3:
        print(f"# warning: only {len(grids)} grids used; slope will be noisy",
              file=__import__("sys").stderr)

    variants = ["load", "store", "ldst"]
    modes = [("simd", 1), ("simt_only", 1), ("simt_only", 4)]

    print("# scalar bench: marginal per-core cycles per operation")
    print("# grids:", ",".join(str(g) for g in grids))
    print("# reps:", args.reps)
    print("variant,mode,sf,cycles_per_operation_or_program")
    for variant in variants:
        for mode, sf in modes:
            per_op = fit_marginal_per_op_cycles(
                variant, args.ops, mode, sf, grids, args.reps
            )
            print(f"{variant},{mode},{sf},{per_op:.3f}", flush=True)


if __name__ == "__main__":
    main()
