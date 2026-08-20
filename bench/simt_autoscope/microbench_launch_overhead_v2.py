#!/usr/bin/env python3
"""Launch-overhead microbenchmark v2: sweep kernel SHAPE x grid x num_warps.

v1 swept only a scalar-store kernel and produced a contradiction: minimal
SIMT kernel measured ~11920 cycles for every grid, yet real small kernels
(silu_mul 3422, seg_indptr 3323, mx4 4450 cycles) are 3x faster than the
minimal kernel.  Hypothesis: the device-side fixed cost depends on the
kernel's codegen shape (scalar-only vs vector tensor ops vs indirect
loads), not only on grid/num_warps.  This script sweeps four shapes that
map onto costmodel features:

    scalar    : one scalar store per CTA (v1 kernel, scalarOps only)
    vector    : BLOCK-wide contiguous load+store per CTA (tensor ops)
    silu      : BLOCK-wide load + silu + store (elementwise, like
                silu_mul_static_tensorwise_quant)
    indirect  : contiguous index load + data load through the index
                (loaded-index-dependent, like _count_expert_num_tokens)

Each config is measured twice:
    back-to-back : N launches between one Event pair (per-launch device
                   cost, the launch-inclusive methodology of the 5-case)
    single       : one launch per Event pair, repeated (diagnoses whether
                   back-to-back pipelining hides part of the cost)

Fit target (offline): fixedOverhead(shape_class, grid, num_warps, route)
where shape_class maps to features: hasIndirectMemory -> indirect,
tensor load ops -> vector/silu, else scalar.

Usage on A5 (SIMD route):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_launch_overhead_v2.py \
        --out ascend_results/launch_overhead_v2_simd.jsonl

SIMT route:

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_launch_overhead_v2.py \
        --out ascend_results/launch_overhead_v2_simt.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def scalar_store_kernel(ptr):
    pid = tl.program_id(0)
    tl.store(ptr + pid, pid)


@triton.jit
def vector_copy_kernel(src, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs)
    tl.store(dst + offs, x)


@triton.jit
def silu_elementwise_kernel(src, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs)
    y = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(dst + offs, y)


@triton.jit
def indirect_load_kernel(src, idx, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx + offs)
    x = tl.load(src + i, other=0.0)
    tl.store(dst + offs, x)


def time_back_to_back(kernel, grid, kargs, constexpr_kwargs, num_warps,
                      warmup, reps, device):
    kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                 num_stages=2)
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    for _ in range(warmup):
        kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                     num_stages=2)
    if device == "npu":
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=2)
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / reps
    else:
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(reps):
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=2)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / reps


def time_single(kernel, grid, kargs, constexpr_kwargs, num_warps,
                reps, device):
    times = []
    for _ in range(reps):
        if device == "npu":
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=2)
            end.record()
            torch.npu.synchronize()
            times.append(start.elapsed_time(end))
        else:
            torch.cuda.synchronize()
            import time
            t0 = time.perf_counter()
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=2)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/launch_overhead_v2.jsonl")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--single-reps", type=int, default=10)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    BLOCK = 1024
    max_grid = 4096
    ptr = torch.zeros(max_grid, dtype=torch.int32, device=device)
    src = torch.randn(max_grid * BLOCK, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    idx = torch.randint(0, max_grid * BLOCK, (max_grid * BLOCK,),
                        device=device).to(torch.int32)

    shapes = {
        "scalar": (scalar_store_kernel, (ptr,), {}),
        "vector": (vector_copy_kernel, (src, dst), {"BLOCK": BLOCK}),
        "silu": (silu_elementwise_kernel, (src, dst), {"BLOCK": BLOCK}),
        "indirect": (indirect_load_kernel, (src, idx, dst),
                     {"BLOCK": BLOCK}),
    }

    for shape, (kernel, kargs, constexpr_kwargs) in shapes.items():
        for grid in [1, 16, 64, 256, 1024, 4096]:
            for num_warps in [1, 4, 32]:
                label = {"shape": shape, "grid": grid,
                         "num_warps": num_warps}
                print(f"=== running {label} ===", flush=True)
                g = (grid,)
                bt = time_back_to_back(kernel, g, kargs, constexpr_kwargs,
                                       num_warps, args.warmup, args.reps,
                                       device)
                single = time_single(kernel, g, kargs, constexpr_kwargs,
                                     num_warps, args.single_reps, device)
                row = {
                    "label": label,
                    "back_to_back_latency_ms": bt,
                    "single_launch_latency_ms": single,
                }
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
