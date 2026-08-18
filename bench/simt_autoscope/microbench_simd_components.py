#!/usr/bin/env python3
"""SIMD compute/dot microbenchmark for Ascend.

Run with TRITON_ASCEND_COMPILE_MODE=simd and AUTO_SIMT_SCOPE=off.

Usage on A5:

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_simd_components.py \
        --out ascend_results/simd_components_microbench.jsonl

Output JSONL rows: op, dtype, n, latency_ms, elements_per_second.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    if OP == 0:
        out = x + y
    elif OP == 1:
        out = x * y
    elif OP == 2:
        out = x / (y + 1.0)
    elif OP == 3:
        out = tl.exp(x)
    elif OP == 4:
        out = (x > y).to(tl.float32)
    elif OP == 5:
        out = tl.where(x > y, x, y)
    else:
        out = x
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ascend_results/simd_components_microbench.jsonl")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"

    def sync():
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()

    def measure(kernel, grid, kargs, constexpr_kwargs, num_warps, num_stages,
                row):
        kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                     num_stages=num_stages)
        sync()
        for _ in range(args.warmup):
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=num_stages)
        sync()
        if hasattr(torch, "npu") and hasattr(torch.npu, "Event"):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.reps):
                kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                             num_stages=num_stages)
            end.record()
            sync()
            latency_ms = start.elapsed_time(end) / args.reps
        else:
            t0 = time.perf_counter()
            for _ in range(args.reps):
                kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                             num_stages=num_stages)
            sync()
            latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

        row = dict(row)
        row["latency_ms"] = latency_ms
        num_ctas = 1
        if isinstance(grid, tuple):
            for dim in grid:
                num_ctas *= int(dim)
        else:
            num_ctas = int(grid)
        row["num_ctas"] = num_ctas
        if "elements" in row and row["elements"]:
            row["elements_per_cta"] = row["elements"] / num_ctas
        if "flops" in row and row["flops"]:
            row["flops_per_cta"] = row["flops"] / num_ctas
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))

    op_names = {0: "add", 1: "mul", 2: "div", 3: "exp", 4: "cmp", 5: "select"}
    BLOCK = 1024
    for dtype in [torch.float32, torch.float16]:
        for n in [1024 * 1024, 4 * 1024 * 1024]:
            x = torch.randn(n, dtype=dtype, device=device)
            y = torch.randn(n, dtype=dtype, device=device)
            out = torch.empty_like(x)
            grid = (triton.cdiv(n, BLOCK),)
            for op_id, op_name in op_names.items():
                measure(
                    elementwise_kernel, grid, (x, y, out, n),
                    {"OP": op_id, "BLOCK": BLOCK},
                    num_warps=4, num_stages=2,
                    row={"kind": "elementwise", "op": op_name, "dtype": str(dtype),
                         "n": n, "elements": n, "elements_per_second": None},
                )
                # Fill elements_per_second after measuring latency
                last = None
                out_path = Path(args.out)
                lines = out_path.read_text().strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    last["elements_per_second"] = n / (last["latency_ms"] / 1000.0)
                    lines[-1] = json.dumps(last, sort_keys=True)
                    out_path.write_text("\n".join(lines) + "\n")

    # Dot sweeps: K must be constexpr and divisible by BLOCK_K.
    dot_shapes = [(128, 128, 64), (256, 256, 128), (512, 512, 128)]
    for M, N, K in dot_shapes:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        a = torch.randn(M, K, dtype=torch.float32, device=device)
        b = torch.randn(K, N, dtype=torch.float32, device=device)
        c = torch.empty(M, N, dtype=torch.float32, device=device)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        measure(
            matmul_kernel, grid,
            (a, b, c, M, N, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)),
            {"K": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            num_warps=4, num_stages=2,
            row={"kind": "dot", "op": "matmul_f32", "dtype": "f32",
                 "M": M, "N": N, "K": K, "flops": 2 * M * N * K},
        )
        out_path = Path(args.out)
        lines = out_path.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            last["flops_per_second"] = 2 * M * N * K / (last["latency_ms"] / 1000.0)
            lines[-1] = json.dumps(last, sort_keys=True)
            out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
