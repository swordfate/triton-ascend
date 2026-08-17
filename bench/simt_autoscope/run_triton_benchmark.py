#!/usr/bin/env python3
"""Run representative Triton kernels under multiple SIMD/SIMT routes on Ascend.

This harness intentionally uses a small set of built-in Triton kernels first.
It is the methodology skeleton: each (case, route) runs in a fresh process with
the proper TRITON_ASCEND_* environment, measures NPU Event latency, and for
the ``simd_simt_report`` route also captures the native C++ costmodel report.

Usage on an Ascend A5 machine with triton-ascend installed:

    python bench/simt_autoscope/run_triton_benchmark.py \
        --case elementwise_silu_mul --route simd \
        --out results/simt_autoscope_bench.jsonl

    python bench/simt_autoscope/run_triton_benchmark.py \
        --case all --route all \
        --out results/simt_autoscope_bench.jsonl \
        --report-dir results/reports \
        --warmup 10 --reps 50

Output JSONL columns:
    case, route, shape, dtype, num_warps, num_stages,
    latency_ms, compile_ms, report_json_path
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUTES = ("simd", "simt_only", "simd_simt_report")


def route_env(route: str) -> dict:
    env = os.environ.copy()
    if route == "simd":
        env["TRITON_ASCEND_COMPILE_MODE"] = "simd"
        env["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "off"
    elif route == "simt_only":
        env["TRITON_ASCEND_COMPILE_MODE"] = "simt_only"
        env["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "off"
    elif route == "simd_simt_report":
        env["TRITON_ASCEND_COMPILE_MODE"] = "simd_simt"
        env["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "report"
    else:
        raise ValueError(f"unknown route: {route}")
    env["TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"] = env.get(
        "TRITON_ASCEND_SIMT_SCOPE_DUMP_OVERRIDE", ""
    )
    return env


def builtin_case_names():
    return [
        "elementwise_silu_mul",
        "rowwise_reduce_masked",
        "indirect_elementwise",
        "block_matmul",
        "single_block_cumsum",
    ]


def run_builtin_case(name: str, route: str, args) -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def elementwise_silu_mul_kernel(
        x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = x * tl.sigmoid(x) * y
        tl.store(out_ptr + offs, z, mask=mask)

    @triton.jit
    def rowwise_reduce_masked_kernel(
        x_ptr,
        mask_ptr,
        out_ptr,
        R,
        C,
        BLOCK_C: tl.constexpr,
        NEG_INF: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        c_mask = cols < C
        offs = row * C + cols
        m = tl.load(mask_ptr + offs, mask=c_mask, other=0).to(tl.int1)
        x = tl.load(x_ptr + offs, mask=c_mask, other=0.0)
        x = tl.where(m, x, NEG_INF)
        row_max = tl.max(x, axis=0)
        tl.store(out_ptr + row, row_max)

    @triton.jit
    def indirect_elementwise_kernel(
        src_ptr, idx_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        val = tl.load(src_ptr + idx, mask=mask, other=0.0)
        out = val * 2.0 + 1.0
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def block_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
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

    @triton.jit
    def single_block_cumsum_kernel(
        in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask, other=0.0)
        y = tl.cumsum(x, axis=0)
        tl.store(out_ptr + offs, y, mask=mask)

    device = "npu" if hasattr(torch, "npu") else "cuda"

    def make_elementwise_silu_mul():
        n_elements = 1024 * 1024
        BLOCK = 1024
        x = torch.randn(n_elements, dtype=torch.float32, device=device)
        y = torch.randn(n_elements, dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, BLOCK),)
        return (
            elementwise_silu_mul_kernel,
            grid,
            (x, y, out, n_elements),
            {"BLOCK": BLOCK, "num_warps": 4, "num_stages": 2},
            {"n_elements": n_elements, "BLOCK": BLOCK, "dtype": "f32", "shape": f"[{n_elements}]"},
        )

    def make_rowwise_reduce_masked():
        R, C, BLOCK_C = 128, 32, 32
        x = torch.randn(R, C, dtype=torch.float32, device=device)
        mask = (torch.rand(R, C, device=device) > 0.3).to(torch.int8)
        out = torch.empty(R, dtype=torch.float32, device=device)
        grid = (R,)
        return (
            rowwise_reduce_masked_kernel,
            grid,
            (x, mask, out, R, C),
            {"BLOCK_C": BLOCK_C, "NEG_INF": -1e30, "num_warps": 1, "num_stages": 1},
            {"R": R, "C": C, "dtype": "f32", "shape": f"[{R},{C}]"},
        )

    def make_indirect_elementwise():
        n_elements = 256 * 1024
        src_size = 1024 * 1024
        BLOCK = 1024
        src = torch.randn(src_size, dtype=torch.float32, device=device)
        idx = torch.randint(0, src_size, (n_elements,), device=device).to(torch.int32)
        out = torch.empty(n_elements, dtype=torch.float32, device=device)
        grid = (triton.cdiv(n_elements, BLOCK),)
        return (
            indirect_elementwise_kernel,
            grid,
            (src, idx, out, n_elements),
            {"BLOCK": BLOCK, "num_warps": 4, "num_stages": 2},
            {"n_elements": n_elements, "src_size": src_size, "BLOCK": BLOCK, "dtype": "f32", "shape": f"[{n_elements}]"},
        )

    def make_block_matmul():
        M, N, K = 256, 256, 128
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        a = torch.randn(M, K, dtype=torch.float32, device=device)
        b = torch.randn(K, N, dtype=torch.float32, device=device)
        c = torch.empty(M, N, dtype=torch.float32, device=device)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        return (
            block_matmul_kernel,
            grid,
            (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)),
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "num_warps": 4, "num_stages": 2},
            {"M": M, "N": N, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "dtype": "f32", "shape": f"[{M},{K}]x[{K},{N}]"},
        )

    def make_single_block_cumsum():
        n_elements = 4096
        BLOCK = 4096
        x = torch.randn(n_elements, dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        grid = (1,)
        return (
            single_block_cumsum_kernel,
            grid,
            (x, out, n_elements),
            {"BLOCK": BLOCK, "num_warps": 4, "num_stages": 1},
            {"n_elements": n_elements, "BLOCK": BLOCK, "dtype": "f32", "shape": f"[{n_elements}]"},
        )

    cases = {
        "elementwise_silu_mul": make_elementwise_silu_mul,
        "rowwise_reduce_masked": make_rowwise_reduce_masked,
        "indirect_elementwise": make_indirect_elementwise,
        "block_matmul": make_block_matmul,
        "single_block_cumsum": make_single_block_cumsum,
    }

    kernel, grid, kargs, launch_kwargs, meta = cases[name]()
    grid_shape = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
    constexpr_kwargs = {k: v for k, v in launch_kwargs.items() if k not in ("num_warps", "num_stages")}
    num_warps = launch_kwargs.get("num_warps", 4)
    num_stages = launch_kwargs.get("num_stages", 1)

    t0 = time.perf_counter()
    kernel[grid_shape](*kargs, **constexpr_kwargs, num_warps=num_warps, num_stages=num_stages)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(args.warmup):
        kernel[grid_shape](*kargs, **constexpr_kwargs, num_warps=num_warps, num_stages=num_stages)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

    if hasattr(torch, "npu") and hasattr(torch.npu, "Event"):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.reps):
            kernel[grid_shape](*kargs, **constexpr_kwargs, num_warps=num_warps, num_stages=num_stages)
        end.record()
        torch.npu.synchronize()
        elapsed_ms = start.elapsed_time(end)
        latency_ms = elapsed_ms / args.reps
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            kernel[grid_shape](*kargs, **constexpr_kwargs, num_warps=num_warps, num_stages=num_stages)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

    return {
        "case": name,
        "route": route,
        "meta": meta,
        "latency_ms": latency_ms,
        "compile_ms": compile_ms,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def child_main(args):
    env = route_env(args.route)
    report_path = None
    if args.report_dir:
        report_path = Path(args.report_dir) / f"{args.case}_{args.route}.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            report_path.unlink()
        env["TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"] = str(report_path)
    os.environ.update(env)

    result = run_builtin_case(args.case, args.route, args)

    if report_path and report_path.exists():
        last = report_path.read_text().strip().splitlines()[-1]
        try:
            result["report_json"] = json.loads(last)
        except json.JSONDecodeError:
            result["report_json"] = {"parse_error": True}
        result["report_json_path"] = str(report_path)
    else:
        result["report_json"] = None
        result["report_json_path"] = None

    write_jsonl(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def parent_main(args):
    cases = builtin_case_names() if args.case == "all" else [args.case]
    routes = list(ROUTES) if args.route == "all" else [args.route]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and args.fresh:
        out_path.unlink()

    for case in cases:
        for route in routes:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--case", case,
                "--route", route,
                "--out", str(out_path),
                "--report-dir", str(Path(args.report_dir)) if args.report_dir else "",
                "--warmup", str(args.warmup),
                "--reps", str(args.reps),
                "--no-fresh",
            ]
            print(f"[run] {case} {route}")
            subprocess.run(cmd, check=True, env=route_env(route))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all", help="built-in case name or 'all'")
    ap.add_argument("--route", default="all", help="simd, simt_only, simd_simt_report, or 'all'")
    ap.add_argument("--out", default="results/simt_autoscope_bench.jsonl")
    ap.add_argument("--report-dir", default="results/reports")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--fresh", action="store_true", default=True)
    ap.add_argument("--no-fresh", action="store_false", dest="fresh")
    args = ap.parse_args()

    if args.case == "all" or args.route == "all":
        parent_main(args)
    else:
        child_main(args)


if __name__ == "__main__":
    main()
