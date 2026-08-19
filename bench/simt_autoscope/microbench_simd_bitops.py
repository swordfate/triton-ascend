#!/usr/bin/env python3
"""Integer bit-op throughput microbenchmark for both routes.

Targets P0 item A3 calibration in docs/simt-costmodel-dataset-plan.md: the
costmodel now classifies i32 bit ops (bitop/min/rem/bitcast) but charges
them with placeholder factors (SIMD bitop factor 96, i.e. ~1 element/cycle
scalarization hypothesis, derived from the silu_quantize_mx4 gap of ~70k
cycles / 59k element-ops).  This script measures the real throughput of
each op class so the placeholder factors can be replaced.

Ops (i32 tensors, load -> op -> store):
    and / or / xor / shl / shr / min / rem
plus a f32->i32 tensor bitcast kernel.

Usage on A5 (SIMD route, primary):

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_simd_bitops.py \
        --out ascend_results/bitops_simd_microbench.jsonl \
        --ttir-dir ascend_results/ttir_bitops

SIMT route cross-check:

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_simd_bitops.py \
        --out ascend_results/bitops_simt_microbench.jsonl \
        --ttir-dir ascend_results/ttir_bitops_simt

Output is JSONL with elements_per_second; the fitting step converts to
vector_instructions/element per system_cycle (SYS_CNT = 988.9 MHz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def int_bitop_kernel(src, dst, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0)
    if OP == 0:
        y = x & 7
    elif OP == 1:
        y = x | 7
    elif OP == 2:
        y = x ^ 7
    elif OP == 3:
        y = x << 3
    elif OP == 4:
        y = x >> 3
    elif OP == 5:
        y = tl.minimum(x, 7)
    else:
        y = x % 7
    tl.store(dst + offs, y, mask=mask)


@triton.jit
def bitcast_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0.0)
    y = x.to(tl.int32, bitcast=True)
    tl.store(dst + offs, y, mask=mask)


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
        name = "{}_{}.ttir.mlir".format(label["pattern"], label["op"])
        out = ttir_dir / name
        out.write_text(ttir_text, encoding="utf-8")
        print(f"  [ttir] saved {out} ({len(ttir_text)} bytes)")
    except Exception as exc:
        print(f"  [ttir] failed for {label}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="ascend_results/bitops_microbench.jsonl")
    ap.add_argument("--ttir-dir",
                    default="ascend_results/ttir_bitops")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=30)
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

        n = label["n"]
        row = {
            "label": label,
            "latency_ms": latency_ms,
            "elements_per_second": n / (latency_ms / 1000.0),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if dump:
            dump_ttir(kernel, grid, kargs, constexpr_kwargs, num_warps,
                      num_stages, ttir_dir, label)

    BLOCK = 1024
    num_warps = 4
    num_stages = 2
    op_names = ["and", "or", "xor", "shl", "shr", "min", "rem"]

    for n in [1 * 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]:
        grid = (triton.cdiv(n, BLOCK),)
        src_i32 = torch.randint(-1000, 1000, (n,), device=device,
                                dtype=torch.int32)
        dst_i32 = torch.empty_like(src_i32)
        for op in range(7):
            label = {"pattern": "int_bitop", "op": op_names[op], "n": n,
                     "dtype": "i32"}
            bench(int_bitop_kernel, grid, (src_i32, dst_i32, n),
                  {"OP": op, "BLOCK": BLOCK}, num_warps, num_stages, label,
                  dump=(op == 3 and n == 1 * 1024 * 1024))

        src_f32 = torch.randn(n, dtype=torch.float32, device=device)
        label = {"pattern": "bitcast", "op": "bitcast", "n": n,
                 "dtype": "f32_to_i32"}
        bench(bitcast_kernel, grid, (src_f32, dst_i32, n),
              {"BLOCK": BLOCK}, num_warps, num_stages, label,
              dump=(n == 1 * 1024 * 1024))


if __name__ == "__main__":
    main()
