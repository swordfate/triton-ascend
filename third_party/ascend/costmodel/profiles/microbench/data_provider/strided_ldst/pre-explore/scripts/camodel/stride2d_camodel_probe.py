#!/usr/bin/env python3
"""Rank-2 strided tl.load probe for CAModel white-box exploration.

One kernel contains exactly one shaped 2D strided load:

    offs = arange(0,R)[:,None] * OS + arange(0,C)[None,:] * IS

The result is reduced to a scalar so the load cannot be removed and the store
traffic is negligible compared with the load under test.

Launched under msopprof simulator.  The manifest carries the usual 1-D keys the
committed analyzer needs (mode / stride / block / grid / x_ptr / x_bytes, with
stride = IS and block = R*C) PLUS the rank-2 fields R / C / OS / IS, so
analyze_stride2d_camodel.py can post-join them by index.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def stride2d_load_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr,
                         OS: tl.constexpr, IS: tl.constexpr,
                         SPAN: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, R)[:, None] * OS + tl.arange(0, C)[None, :] * IS
    v = tl.load(x_ptr + pid * SPAN + offs)
    tl.store(out_ptr + pid, tl.sum(v))


def launch_options(mode: str, grid: int, num_warps: int):
    common = {"num_warps": num_warps, "superblock_factor": 1,
              "logical_program_count_hint": grid,
              "physical_vector_core_count_hint": 1,
              "auto_simt_scope_mode": "off"}
    if mode == "simd":
        return {**common, "compile_mode": "simd", "enable_auto_blockify": False}
    if mode == "simt_only":
        return {**common, "compile_mode": "simt_only", "enable_auto_blockify": True}
    raise ValueError(mode)


def forms(shapes, inner_strides, outer_pad, include_contig):
    out = []
    for shape in shapes:
        rows, cols = (int(v) for v in shape.split("x"))
        if include_contig:
            out.append(("contig", rows, cols, cols, 1))
        for k in inner_strides:
            out.append((f"both_s{k}_tight", rows, cols, cols * k, k))
            out.append((f"both_s{k}_gap", rows, cols, cols * k + outer_pad, k))
    return out


def save_asm(compiled, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for key, value in (getattr(compiled, "asm", None) or {}).items():
        if not isinstance(value, (str, bytes)):
            continue
        path = out_dir / f"{label}.{key}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value, encoding="utf-8", errors="replace")
        keys.append({"key": key, "path": path.name})
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd"])
    ap.add_argument("--shapes", nargs="+", default=["1x64", "8x16", "16x32"])
    ap.add_argument("--inner-strides", nargs="+", type=int, default=[3])
    ap.add_argument("--outer-pad", type=int, default=7)
    ap.add_argument("--include-contig", action="store_true")
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--num-warps", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("launch_manifest.json"))
    ap.add_argument("--asm-dir", type=Path, default=Path("asm"))
    ap.add_argument("--warmup-only", action="store_true")
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    all_forms = forms(args.shapes, args.inner_strides, args.outer_pad,
                      args.include_contig)
    rows = []
    for mode in args.modes:
        opts = launch_options(mode, args.grid, args.num_warps)
        for name, R, C, OS, IS in all_forms:
            span = (R - 1) * OS + (C - 1) * IS + 1
            total = args.grid * (span + 4096)
            x = (torch.arange(total, dtype=torch.float32) % 7.0).contiguous().to("npu")
            out = torch.zeros(args.grid, dtype=torch.float32).to("npu")
            label = f"{mode}_{name}_{R}x{C}_os{OS}_is{IS}"
            compiled = stride2d_load_kernel.warmup(
                x, out, R=R, C=C, OS=OS, IS=IS, SPAN=span,
                grid=(args.grid,), **opts)
            save_asm(compiled, args.asm_dir, label)
            if not args.warmup_only:
                t0 = time.time()
                stride2d_load_kernel[(args.grid,)](x, out, R=R, C=C, OS=OS, IS=IS,
                                                   SPAN=span, **opts)
                torch_npu.npu.synchronize()
                elapsed = time.time() - t0
            else:
                elapsed = None
            rows.append({
                "order": len(rows), "mode": mode, "form": name,
                "R": R, "C": C, "OS": OS, "IS": IS,
                "stride": IS, "block": R * C, "grid": args.grid,
                "num_warps": args.num_warps,
                "options": {k: v for k, v in opts.items()},
                "kernel_symbol": getattr(compiled, "name", None),
                "launched": not args.warmup_only,
                "host_launch_seconds": elapsed,
                "x_ptr": hex(x.data_ptr()),
                "out_ptr": hex(out.data_ptr()),
                "x_bytes": int(x.numel()) * x.element_size(),
                "used_bytes": args.grid * span * 4,
                "span_elements": span,
                "contiguous": bool(IS == 1 and OS == C),
            })
    args.manifest.write_text(json.dumps(
        {"probe": "stride2d_camodel_probe", "cases": rows}, indent=2),
        encoding="utf-8")
    print(f"wrote {args.manifest} cases={len(rows)}")


if __name__ == "__main__":
    main()
