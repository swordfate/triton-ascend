#!/usr/bin/env python3
"""Device-side launch-overhead microbenchmark for both routes.

Targets P1 item C in docs/simt-costmodel-dataset-plan.md: replace the
min_kernel_cycles floor (SIMD 11000 / SIMT 12500, hand-tuned on the 5-case
minima) with a measured, additive fixedOverhead(grid, num_warps, route).

The kernel is the smallest possible real kernel: one scalar store per CTA.
Latency is measured as N back-to-back launches between one Event pair, so
the number is device-side per-launch cost (host launch is excluded, and the
report's grid_wave_count gap is exactly what the grid sweep captures).

Sweep: grid {1, 4, 16, 64, 256, 1024, 4096} x num_warps {1, 4, 8, 32}.
Fitting step (done offline): latency(grid, warps) = a(warps) + b(warps) *
ceil(grid / cores_per_wave), separately per route.

Usage on A5 (SIMD route):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_launch_overhead.py \
        --out ascend_results/launch_overhead_simd_microbench.jsonl

SIMT route:

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_launch_overhead.py \
        --out ascend_results/launch_overhead_simt_microbench.jsonl

Output is JSONL with latency_ms per (grid, num_warps) config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def overhead_kernel(ptr):
    pid = tl.program_id(0)
    tl.store(ptr + pid, pid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/launch_overhead_microbench.jsonl")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_grid = 4096
    ptr = torch.zeros(max_grid, dtype=torch.int32, device=device)

    for grid in [1, 4, 16, 64, 256, 1024, 4096]:
        for num_warps in [1, 4, 8, 32]:
            g = (grid,)
            label = {"grid": grid, "num_warps": num_warps}
            print(f"=== running {label} ===", flush=True)
            overhead_kernel[g](ptr, num_warps=num_warps, num_stages=2)
            if device == "npu":
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize()
            for _ in range(args.warmup):
                overhead_kernel[g](ptr, num_warps=num_warps, num_stages=2)
            if device == "npu":
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(args.reps):
                    overhead_kernel[g](ptr, num_warps=num_warps, num_stages=2)
                end.record()
                torch.npu.synchronize()
                latency_ms = start.elapsed_time(end) / args.reps
            else:
                torch.cuda.synchronize()
                import time
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    overhead_kernel[g](ptr, num_warps=num_warps, num_stages=2)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

            row = {
                "label": label,
                "latency_ms": latency_ms,
                "grid": grid,
                "num_warps": num_warps,
                "per_cta_latency_ms": latency_ms / grid,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
