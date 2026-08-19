#!/usr/bin/env python3
"""Dynamic-bound loop per-trip cost microbenchmark.

Targets the unknown-trip loop gap found on SGLang
silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe
(docs/simt-costmodel-dataset-plan.md section 10): the costmodel charges an
scf.for with a runtime bound as ONE trip, so per-tile work is invisible and
all three route scores collapse onto the min_kernel_cycles floor.  This
script measures the real per-trip cost of a dynamic loop on both routes to
size a future trip-count proxy.

Two kernels:
    dynamic_loop_silu      : per-tile load + silu + store, runtime trips
    dynamic_loop_computed  : same, but the load address per element is
                             (offs // S) * S + (offs % S) with a computed
                             index, mirroring the silu_mul kernel's
                             ids // intermediate_size addressing

trips is a runtime argument, so the loop bound is unknown at compile time
(has_unknown_trip_count in the costmodel).  Sweep trips to separate fixed
overhead from per-trip slope.

Usage on A5 (SIMD route):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_dynamic_loop.py \
        --out ascend_results/dynamic_loop_simd_microbench.jsonl \
        --ttir-dir ascend_results/ttir_dynamic_loop

SIMT route:

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_dynamic_loop.py \
        --out ascend_results/dynamic_loop_simt_microbench.jsonl \
        --ttir-dir ascend_results/ttir_dynamic_loop_simt

Output is JSONL with latency_ms per (pattern, trips, grid) config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def dynamic_loop_silu_kernel(src, dst, trips, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(trips):
        offs = (pid * trips + i) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(src + offs)
        y = x * (1.0 / (1.0 + tl.exp(-x)))
        tl.store(dst + offs, y)


@triton.jit
def dynamic_loop_computed_kernel(src, dst, trips, S: tl.constexpr,
                                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(trips):
        offs = (pid * trips + i) * BLOCK + tl.arange(0, BLOCK)
        idx = (offs // S) * S + (offs % S)
        x = tl.load(src + idx)
        y = x * (1.0 / (1.0 + tl.exp(-x)))
        tl.store(dst + offs, y)


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
        name = "{}.ttir.mlir".format(label["pattern"])
        out = ttir_dir / name
        out.write_text(ttir_text, encoding="utf-8")
        print(f"  [ttir] saved {out} ({len(ttir_text)} bytes)")
    except Exception as exc:
        print(f"  [ttir] failed for {label}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/dynamic_loop_microbench.jsonl")
    ap.add_argument("--ttir-dir",
                    default="ascend_results/ttir_dynamic_loop")
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

        row = {
            "label": label,
            "latency_ms": latency_ms,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if dump:
            dump_ttir(kernel, grid, kargs, constexpr_kwargs, num_warps,
                      num_stages, ttir_dir, label)

    BLOCK = 1024
    grid = 16
    num_warps = 4
    num_stages = 2
    trips_list = [1, 4, 16, 64, 256]

    for trips in trips_list:
        n = grid * trips * BLOCK
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        label = {"pattern": "dynamic_loop_silu", "trips": trips,
                 "grid": grid, "n": n}
        bench(dynamic_loop_silu_kernel, (grid,), (src, dst, trips),
              {"BLOCK": BLOCK}, num_warps, num_stages, label,
              dump=(trips == 4))

        S = 8
        label = {"pattern": "dynamic_loop_computed", "trips": trips,
                 "grid": grid, "n": n, "S": S}
        bench(dynamic_loop_computed_kernel, (grid,), (src, dst, trips),
              {"S": S, "BLOCK": BLOCK}, num_warps, num_stages, label,
              dump=(trips == 4))


if __name__ == "__main__":
    main()
