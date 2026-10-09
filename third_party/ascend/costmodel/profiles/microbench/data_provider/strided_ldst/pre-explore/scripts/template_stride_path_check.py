#!/usr/bin/env python3
"""Compile-only boundary check for the template strided-load/store dispatch.

Expected from ``StridedLoadStoreRewrite.cpp``:
  * static non-power-of-two stride >= 3 -> ``call @triton_stride_load/store``
  * stride 1, 2 and powers of two       -> structured/deinterleave path,
    i.e. no template call (memref.copy remains in TTAdapter).

This script also records TTAdapter snippets as IR evidence.  It does not launch
kernels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from template_stride_common import asm_text, save_asm, template_opts


@triton.jit
def check_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.arange(0, BLOCK), v)


@triton.jit
def check_store_kernel(out_ptr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    v = tl.arange(0, BLOCK).to(tl.float32)
    tl.store(out_ptr + offs, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128, 255, 256])
    ap.add_argument("--out", type=Path, default=Path("template_stride_path_check.json"))
    ap.add_argument("--asm-dir", type=Path, default=Path("template_stride_path_check_asm"))
    args = ap.parse_args()

    torch_npu.npu.set_device(0)
    x = torch.randn(1 << 18, dtype=torch.float32, device="npu")
    y = torch.zeros(1 << 18, dtype=torch.float32, device="npu")
    opts = template_opts(num_warps=1, grid=1)
    results = []
    for path, kernel, kernel_args in [
        ("triton_stride_load", check_load_kernel, (x, y)),
        ("triton_stride_store", check_store_kernel, (y,)),
    ]:
        expect_template = lambda stride: stride >= 3 and (stride & (stride - 1)) != 0
        for block in args.blocks:
            for stride in args.strides:
                compiled = kernel.warmup(
                    *kernel_args, BLOCK=block, STRIDE=stride, grid=(1,), **opts
                )
                text = asm_text(compiled)
                contains = path in text
                expected = expect_template(stride)
                keys = save_asm(compiled, args.asm_dir, f"{path.split('_')[-1]}_b{block}_s{stride}")
                results.append({
                    "path": path,
                    "block": int(block),
                    "stride": int(stride),
                    "expected_template": bool(expected),
                    "contains_template_call": bool(contains),
                    "match": bool(expected == contains),
                    "ttadapter_has_memref_copy": "memref.copy" in text,
                    "asm_keys": keys,
                })
                print(f'{path} b{block} s{stride}: expected={expected} '
                      f'contains={contains} match={expected == contains}', flush=True)
    args.out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    bad = [r for r in results if not r["match"]]
    print(f"wrote {args.out}; mismatches={len(bad)}")


if __name__ == "__main__":
    main()
