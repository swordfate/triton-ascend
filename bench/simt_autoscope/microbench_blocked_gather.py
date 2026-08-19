#!/usr/bin/env python3
"""Blocked-gather (row-contiguous + random row offset) microbenchmark.

Targets P1 item D in docs/simt-costmodel-dataset-plan.md: the costmodel
currently charges every loaded-index-dependent load at the pure gather rate
(2.27 B/cycle SIMD), but real kernels like VLLM _causal_conv1d_fwd_kernel
load W-wide stride-1 rows whose *row start* is data-dependent.  Measured
effective bandwidth there is ~268 B/cycle, i.e. the access is coalesced
within the row, not random.

This script sweeps the contiguous run width W:

    row = tl.load(row_ids + base + i)      # scalar, data-dependent
    x   = tl.load(src + row * W + offs)    # W-wide stride-1 row
    tl.store(dst + ... , x)

W=1 degenerates to pure per-element gather; W=1024 degenerates to a
contiguous copy (one row per program).  Every config moves the same BLOCK
(1024) elements per CTA so per-CTA overhead stays comparable across W.

The TTIR of the data load is exactly the loaded-index-dependent pointer
shape the costmodel anchors on (`addptr(loaded_scalar * W, make_range(W))`),
so the per-W TTIR dumps also serve to validate the future feature-side
row-width detection.

Usage on A5 (SIMD route, the primary data):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_blocked_gather.py \
        --out ascend_results/blocked_gather_simd_microbench.jsonl \
        --ttir-dir ascend_results/ttir_blocked_gather

Cross-check on the SIMT route (SIMT GM rates need the same locality bins):

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_blocked_gather.py \
        --out ascend_results/blocked_gather_simt_microbench.jsonl \
        --ttir-dir ascend_results/ttir_blocked_gather_simt

Output is JSONL with bytes/second; convert to bytes/system_cycle with the
SYS_CNT frequency in the microbenchmark profile (988.9 MHz on the current
A5 configuration).  TTIR text is saved per dumped W into --ttir-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def blocked_gather_kernel(src, row_ids, dst, R: tl.constexpr,
                          W: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, W)
    row_base = pid * R
    out_base = pid * R * W
    for i in range(R):
        row = tl.load(row_ids + row_base + i)
        x = tl.load(src + row * W + offs)
        tl.store(dst + out_base + i * W + offs, x)


@triton.jit
def contiguous_copy_kernel(src, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs)
    tl.store(dst + offs, x)


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
        name = "blocked_gather_W{}.ttir.mlir".format(label["W"])
        out = ttir_dir / name
        out.write_text(ttir_text, encoding="utf-8")
        print(f"  [ttir] saved {out} ({len(ttir_text)} bytes)")
    except Exception as exc:
        print(f"  [ttir] failed for {label}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/blocked_gather_microbench.jsonl")
    ap.add_argument("--ttir-dir",
                    default="ascend_results/ttir_blocked_gather")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"
    ttir_dir = Path(args.ttir_dir)
    ttir_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def bench(kernel, grid, kargs, constexpr_kwargs, num_warps, num_stages,
              label, dump=False):
        label = dict(label)
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
            latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

        read_bytes = label["n"] * 4
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

    # Full W sweep at one working set, then a working-set sweep at three
    # representative W to catch cache-size dependence.
    configs = [(4 * 1024 * 1024, w) for w in
               [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]]
    for n in [1 * 1024 * 1024, 16 * 1024 * 1024]:
        for w in [32, 256, 1024]:
            configs.append((n, w))

    for n, w in configs:
        n_rows = n // w
        R = max(1, BLOCK // w)
        grid = (triton.cdiv(n_rows, R),)
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        row_ids = torch.randint(0, n_rows, (n_rows,), device=device).to(torch.int32)
        label = {"pattern": "blocked_gather", "n": n, "W": w, "R": R,
                 "read_bytes": n * 4, "write_bytes": n * 4}
        bench(blocked_gather_kernel, grid, (src, row_ids, dst),
              {"R": R, "W": w}, num_warps, num_stages, label,
              dump=(n == 4 * 1024 * 1024 and w in (32, 256, 1024)))

    # Contiguous baseline for reference (W=infinity).
    n = 4 * 1024 * 1024
    grid = (triton.cdiv(n, BLOCK),)
    src = torch.randn(n, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    label = {"pattern": "contiguous", "n": n, "W": -1, "R": 1,
             "read_bytes": n * 4, "write_bytes": n * 4}
    bench(contiguous_copy_kernel, grid, (src, dst), {"BLOCK": BLOCK},
          num_warps, num_stages, label, dump=True)


if __name__ == "__main__":
    main()
