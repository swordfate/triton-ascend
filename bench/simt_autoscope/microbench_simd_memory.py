#!/usr/bin/env python3
"""SIMD memory microbenchmark for Ascend.

This script is intended to run with TRITON_ASCEND_COMPILE_MODE=simd.  It
measures effective SIMD global-memory bandwidth for four access patterns that
are easy to map to TTIR features:

    contiguous : dst[i] = src[i]
    strided    : dst[i] = src[(i * stride) % n]
    gather     : dst[i] = src[idx[i]]
    masked     : dst[i] = src[i] if mask[i] else 0

Usage on A5:

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_simd_memory.py \
        --out ascend_results/simd_memory_microbench.jsonl

The output is JSONL.  Each line contains the effective read+write bytes and
latency; the fitting step will convert them into bytes/system_cycle using the
SYS_CNT frequency in the microbenchmark profile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

@triton.jit
def contiguous_copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)

@triton.jit
def strided_copy_kernel(src, dst, n, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = (offs * stride) % n
    x = tl.load(src + idx, mask=mask)
    tl.store(dst + offs, x, mask=mask)

@triton.jit
def gather_copy_kernel(src, idx, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i = tl.load(idx + offs, mask=mask, other=0)
    x = tl.load(src + i, mask=mask, other=0.0)
    tl.store(dst + offs, x, mask=mask)

@triton.jit
def masked_copy_kernel(src, mask_tensor, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    m = tl.load(mask_tensor + offs, mask=mask, other=0).to(tl.int1)
    x = tl.load(src + offs, mask=mask & m, other=0.0)
    tl.store(dst + offs, x, mask=mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ascend_results/simd_memory_microbench.jsonl")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"

    def bench(kernel, grid, kargs, constexpr_kwargs, num_warps, num_stages,
              label):
        label = dict(label)
        full_tensor_read_bytes = label["n"] * 4
        if "read_bytes" not in label and "active_ratio" in label:
            # Report requested read bytes first, but keep the full-tensor
            # byte count too: the C++ costmodel currently charges full tensor
            # bytes for masked tt.load, so both definitions are useful.
            label["read_bytes"] = int(full_tensor_read_bytes * label["active_ratio"])
            label["full_tensor_read_bytes"] = full_tensor_read_bytes
        else:
            label["full_tensor_read_bytes"] = full_tensor_read_bytes
        kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                     num_stages=num_stages)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
        for _ in range(args.warmup):
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=num_stages)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.reps):
                kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                             num_stages=num_stages)
            end.record()
            torch.npu.synchronize()
            elapsed_ms = start.elapsed_time(end)
            latency_ms = elapsed_ms / args.reps
        else:
            torch.cuda.synchronize()
            import time
            t0 = time.perf_counter()
            for _ in range(args.reps):
                kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                             num_stages=num_stages)
            torch.cuda.synchronize()
            import time
            latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

        read_bytes = label.get("read_bytes", label["n"] * 4)
        write_bytes = label["n"] * 4
        total_bytes = read_bytes + write_bytes
        num_ctas = 1
        if isinstance(grid, tuple):
            for dim in grid:
                num_ctas *= int(dim)
        else:
            num_ctas = int(grid)
        row = {
            "label": label,
            "num_ctas": num_ctas,
            "latency_ms": latency_ms,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "total_bytes": total_bytes,
            "per_cta_read_bytes": read_bytes / num_ctas,
            "per_cta_write_bytes": write_bytes / num_ctas,
            "per_cta_total_bytes": total_bytes / num_ctas,
            "bytes_per_second": total_bytes / (latency_ms / 1000.0),
            "per_cta_bytes_per_second": (total_bytes / num_ctas) / (latency_ms / 1000.0),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))

    # Sweep matrix: total elements and patterns.
    for n in [1024 * 1024, 4 * 1024 * 1024]:
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        bench(contiguous_copy_kernel, grid, (src, dst, n), {"BLOCK": BLOCK},
              num_warps=4, num_stages=2, label={"pattern": "contiguous", "n": n})

        for stride in [2, 4, 8, 16]:
            bench(strided_copy_kernel, grid, (src, dst, n, stride),
                  {"BLOCK": BLOCK}, num_warps=4, num_stages=2,
                  label={"pattern": "strided", "n": n, "stride": stride})

        idx = torch.randint(0, n, (n,), device=device).to(torch.int32)
        bench(gather_copy_kernel, grid, (src, idx, dst, n), {"BLOCK": BLOCK},
              num_warps=4, num_stages=2,
              label={"pattern": "gather", "n": n})

        mask_tensor = (torch.rand(n, device=device) > 0.5).to(torch.int8)
        bench(masked_copy_kernel, grid, (src, mask_tensor, dst, n),
              {"BLOCK": BLOCK}, num_warps=4, num_stages=2,
              label={"pattern": "masked", "n": n, "active_ratio": 0.5})


if __name__ == "__main__":
    main()
