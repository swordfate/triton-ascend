#!/usr/bin/env python3
"""Strided-access microbenchmark mirroring real target kernels.

This script reproduces the strided access patterns found in the 25 real
target kernels (see docs/simt-costmodel-dataset-plan.md section 3), so that
the C++ feature extractor can learn how to estimate a stride from TTIR:

    rope_interleaved : FBGEMM _rope_padded_kernel
                       cols_re = (offset + c) * 2 ; cols_im = cols_re + 1
                       (stride-2 interleaved load AND store, stride is a
                        compile-time constant baked into the TTIR)
    column_access    : read a column of a 2D tensor, i.e. one element per
                       row: idx = pid + arange(0, BLOCK) * row_stride
                       (stride is a runtime scalar argument)
    strided_store    : contiguous load, strided store
                       idx = pid + arange(0, BLOCK) * stride
                       (stride is a runtime scalar argument)
    strided_general  : dst[i] = src[(i * stride) % n]
                       (the existing microbench pattern, for continuity)

For every kernel/constexpr shape this script also saves the TTIR text
(compiled.asm["ttir"]) into --ttir-dir, so the C++ feature side can inspect
exactly how each pattern shows up in Ascend TTIR.

Usage on A5 (SIMD route):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_simd_strided_patterns.py \
        --out ascend_results/simd_strided_patterns_microbench.jsonl \
        --ttir-dir ascend_results/ttir_strided_patterns

Optionally also run the SIMT route for cross validation:

    TRITON_ASCEND_COMPILE_MODE=simt_only \
    python bench/simt_autoscope/microbench_simd_strided_patterns.py \
        --out ascend_results/simt_strided_patterns_microbench.jsonl \
        --ttir-dir ascend_results/ttir_strided_patterns_simt

Output is JSONL with effective bytes/second; the fitting step converts to
bytes/system_cycle using the SYS_CNT frequency in the microbenchmark profile
(988.9 MHz on the current A5 configuration).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def rope_interleaved_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    c = tl.arange(0, BLOCK // 2)
    cols_re = base + c * 2
    cols_im = cols_re + 1
    mask = cols_im < n
    re_x = tl.load(src + cols_re, mask=mask, other=0.0)
    im_x = tl.load(src + cols_im, mask=mask, other=0.0)
    y_re = re_x - im_x
    y_im = re_x + im_x
    tl.store(dst + cols_re, y_re, mask=mask)
    tl.store(dst + cols_im, y_im, mask=mask)


@triton.jit
def column_access_kernel(src, dst, rows, cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < rows
    idx = pid + offs * cols
    x = tl.load(src + idx, mask=mask, other=0.0)
    tl.store(dst + pid * BLOCK + offs, x, mask=mask)


@triton.jit
def strided_store_kernel(src, dst, n, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    idx = offs * stride
    tl.store(dst + idx, x, mask=mask)


@triton.jit
def strided_general_kernel(src, dst, n, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = (offs * stride) % n
    x = tl.load(src + idx, mask=mask, other=0.0)
    tl.store(dst + offs, x, mask=mask)


def dump_ttir(kernel, grid, kargs, constexpr_kwargs, num_warps, num_stages,
              ttir_dir: Path, label: dict):
    try:
        compiled = kernel.warmup(
            *kargs, grid=grid, **constexpr_kwargs,
            num_warps=num_warps, num_stages=num_stages,
        )
        ttir_text = compiled.asm.get("ttir")
        if not ttir_text:
            print(f"  [ttir] no ttir stage in asm for {label}")
            return
        name = "_".join(
            f"{k}{v}" for k, v in sorted(label.items())
            if k in ("pattern", "stride")
        )
        out = ttir_dir / f"{name}.ttir.mlir"
        out.write_text(ttir_text, encoding="utf-8")
        print(f"  [ttir] saved {out} ({len(ttir_text)} bytes)")
    except Exception as exc:
        print(f"  [ttir] failed for {label}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/simd_strided_patterns_microbench.jsonl")
    ap.add_argument("--ttir-dir",
                    default="ascend_results/ttir_strided_patterns")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument(
        "--pattern",
        default=None,
        help="only run this pattern: rope_interleaved, column_access, "
             "strided_store, strided_general (default: all)",
    )
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"
    ttir_dir = Path(args.ttir_dir)
    ttir_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def bench(kernel, grid, kargs, constexpr_kwargs, num_warps, num_stages,
              label, dump=False):
        label = dict(label)
        if args.pattern and label["pattern"] != args.pattern:
            return
        print(f"=== running {label} ===", flush=True)
        kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                     num_stages=num_stages)
        if device == "npu":
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
        for _ in range(args.warmup):
            kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                         num_stages=num_stages)
        if device == "npu":
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.reps):
                kernel[grid](*kargs, **constexpr_kwargs, num_warps=num_warps,
                             num_stages=num_stages)
            end.record()
            torch.npu.synchronize()
            latency_ms = start.elapsed_time(end) / args.reps
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
        write_bytes = label.get("write_bytes", label["n"] * 4)
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
            "bytes_per_second": total_bytes / (latency_ms / 1000.0),
            "per_cta_bytes_per_second": (total_bytes / num_ctas) / (latency_ms / 1000.0),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))
        if dump:
            dump_ttir(kernel, grid, kargs, constexpr_kwargs, num_warps,
                      num_stages, ttir_dir, label)

    BLOCK = 1024
    num_warps = 4
    num_stages = 2

    # --- rope_interleaved: stride-2 interleave (constant stride in TTIR) ---
    for n in [512 * 1024, 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024]:
        grid = (triton.cdiv(n, BLOCK),)
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        label = {"pattern": "rope_interleaved", "n": n, "stride": 2,
                 "read_bytes": n * 4, "write_bytes": n * 4}
        bench(rope_interleaved_kernel, grid, (src, dst, n), {"BLOCK": BLOCK},
              num_warps, num_stages, label, dump=(n == 1024 * 1024))

    # --- column_access: stride is runtime `cols`.  Keep sizes small:
    # reading one element per row is extremely slow (one cache line per
    # 4-byte read when cols is large). ---
    for rows, cols in [(256, 4096), (1024, 1024), (1024, 4096)]:
        BLOCK_C = min(1024, triton.next_power_of_2(rows))
        grid = (cols,)
        src = torch.randn(rows, cols, dtype=torch.float32, device=device)
        dst = torch.empty(rows * cols, dtype=torch.float32, device=device)
        n = rows * cols
        label = {"pattern": "column_access", "n": n, "rows": rows,
                 "stride": cols, "read_bytes": n * 4, "write_bytes": n * 4}
        bench(column_access_kernel, grid, (src, dst, rows, cols),
              {"BLOCK": BLOCK_C}, num_warps, num_stages, label,
              dump=(rows == 1024 and cols == 4096))

    # --- strided_store: contiguous load, strided store ---
    for n, stride in [(1024 * 1024, 2), (1024 * 1024, 8),
                      (4 * 1024 * 1024, 4), (4 * 1024 * 1024, 16)]:
        grid = (triton.cdiv(n, BLOCK),)
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty(n * stride, dtype=torch.float32, device=device)
        label = {"pattern": "strided_store", "n": n, "stride": stride,
                 "read_bytes": n * 4, "write_bytes": n * 4}
        bench(strided_store_kernel, grid, (src, dst, n, stride),
              {"BLOCK": BLOCK}, num_warps, num_stages, label,
              dump=(n == 1024 * 1024 and stride == 2))

    # --- strided_general: existing pattern, for continuity ---
    for n, stride in [(4 * 1024 * 1024, 2), (4 * 1024 * 1024, 4),
                      (4 * 1024 * 1024, 8), (4 * 1024 * 1024, 16)]:
        grid = (triton.cdiv(n, BLOCK),)
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        label = {"pattern": "strided_general", "n": n, "stride": stride,
                 "read_bytes": n * 4, "write_bytes": n * 4}
        bench(strided_general_kernel, grid, (src, dst, n, stride),
              {"BLOCK": BLOCK}, num_warps, num_stages, label,
              dump=(stride == 2))


if __name__ == "__main__":
    main()
