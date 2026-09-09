#!/usr/bin/env python3
"""Calibrate SIMD Triton scalar GM store marginal cost from real NPU sweeps.

Why this script exists
----------------------
The CCE scalar-store probe measures direct MainScalar scalar GM stores.
In real Triton SIMD scalar-heavy kernels, scalar GM stores are lowered through
MTE3-like operations and are much more expensive.  This script measures the
same Triton scalar_ldst kernel used by demo_scalar_scenarios.py and reports
the marginal cycles per extra scalar store after subtracting a no-op baseline.

Run on the Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 calibrate_triton_simd_scalar_store.py --grid 4096
"""
import argparse
import statistics

import torch
import torch_npu
import triton
import triton.language as tl

# Rough calibration constants; adjust to the measured device.
DEFAULT_CORES = 56
DEFAULT_FREQ_MHZ = 988.9


@triton.jit
def scalar_ldst_kernel(a, b, out, N_LD: tl.constexpr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    for i in tl.static_range(N_LD):
        s += tl.load(a + pid * N_LD + i)
    for i in tl.static_range(N_ST):
        tl.store(b + pid * N_ST + i, s + i)
    tl.store(out + pid, s)


def measure(grid, nld, nst, reps=30):
    a = torch.arange(grid * max(nld, 1), dtype=torch.int32, device="npu") % 7
    b = torch.zeros(grid * max(nst, 1), dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    opts = {
        "num_warps": 1,
        "compile_mode": "simd",
        "auto_simt_scope_mode": "off",
        "enable_auto_blockify": False,
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
    }

    def launch():
        scalar_ldst_kernel[(grid,)](
            a, b, out, N_LD=nld, N_ST=nst, **opts
        )

    for _ in range(5):
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


def per_program_cycles(ms, grid, cores=DEFAULT_CORES, freq_mhz=DEFAULT_FREQ_MHZ):
    # ms is wall time for the whole grid.  All cores run in parallel; the
    # serial per-core per-program cost is:
    #   (ms / 1000 * freq_hz) / (grid / cores)
    return ms * 1e-3 * freq_mhz * 1e6 * cores / grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--ops", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    grid = args.grid
    base_ms = measure(grid, 0, 0, reps=args.reps)
    print(f"# grid={grid} base_ms={base_ms:.6f}")
    print("ops,store_marginal_cycles_per_op")
    for n in args.ops:
        store_ms = measure(grid, 0, n, reps=args.reps)
        store_cycles = per_program_cycles(store_ms - base_ms, grid)
        print(f"{n},{store_cycles / n:.3f}", flush=True)


if __name__ == "__main__":
    main()
