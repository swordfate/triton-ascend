#!/usr/bin/env python3
"""Compile a real costmodel-controlled strided kernel and inspect its route.

This probe runs the Python backend with compile_mode="simd_simt" and
auto_simt_scope_mode="auto".  It does not need a CAModel run: the goal is to
confirm which route the native CostModel selects and whether the emitted
TTAdapter contains the local-SIMT triton_stride_load/store template call when
the route is mixed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def strided_load_route_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr,
                              STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK) * STRIDE
    value = tl.load(x_ptr + pid * BLOCK * STRIDE + offs)
    tl.store(out_ptr + pid, tl.sum(value, axis=0))


@triton.jit
def strided_store_route_kernel(out_ptr, BLOCK: tl.constexpr,
                               STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK) * STRIDE
    value = tl.arange(0, BLOCK).to(tl.float32)
    tl.store(out_ptr + pid * BLOCK * STRIDE + offs, value)


def asm_text(compiled):
    chunks = []
    for value in (getattr(compiled, "asm", {}) or {}).values():
        if isinstance(value, bytes):
            chunks.append(value.decode("latin1", errors="replace"))
        elif isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks)


def compile_one(kernel, args, *, block, stride, report_path: Path, outdir: Path):
    report_path.unlink(missing_ok=True)
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "compile_on_910_95": True,
        "enable_auto_blockify": True,
        "superblock_factor": 1,
        "logical_program_count_hint": 1,
        "physical_vector_core_count_hint": 1,
    }
    compiled = kernel.warmup(*args, BLOCK=block, STRIDE=stride, grid=(1,),
                             **opts)
    text = asm_text(compiled)
    outdir.mkdir(parents=True, exist_ok=True)
    for key, value in (getattr(compiled, "asm", {}) or {}).items():
        path = outdir / f"b{block}_s{stride}.{key}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8", errors="replace")
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text())
    return compiled, text, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--strides", nargs="+", type=int, default=[3])
    ap.add_argument("--outdir", type=Path,
                    default=Path("/tmp/mixed_route_probe"))
    args = ap.parse_args()

    torch_npu.npu.set_device(0)
    total = args.block * max(args.strides) + 4096
    x = (torch.arange(total, dtype=torch.float32) % 7.0).to("npu")
    out = torch.zeros(1, dtype=torch.float32).to("npu")

    for stride in args.strides:
        for name, kernel, kernel_args, symbol in (
            ("load", strided_load_route_kernel, (x, out),
             "triton_stride_load"),
            ("store", strided_store_route_kernel, (out,),
             "triton_stride_store"),
        ):
            report_path = args.outdir / f"report_{name}_s{stride}.json"
            compiled, text, report = compile_one(
                kernel, kernel_args, block=args.block, stride=stride,
                report_path=report_path, outdir=args.outdir / name)
            has_template = symbol in text
            effective = None
            if report:
                effective = report.get("stage_model", {}).get(
                    "selected_route", {}).get("kind")
                if effective is None:
                    effective = report.get("stage_model", {}).get(
                        "effective_decision_kind")
            print(f"{name} s{stride}: effective={effective} "
                  f"template_call={has_template} report={report_path}")
            if has_template:
                for line in text.splitlines():
                    if "call @" + symbol in line:
                        print("   ", line.strip()[:240])


if __name__ == "__main__":
    main()
