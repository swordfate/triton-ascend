#!/usr/bin/env python3
"""Compile-only path check: element dtype x static stride x lowering mode.

Answers, without launching anything:

  1. Which (dtype, stride) combinations reach the SIMT strided template
     (call @triton_stride_load / @triton_stride_store)?
  2. Is the template bail boundary element-based or byte-based?
     StridedLoadStoreRewrite.cpp decides on the per-axis ELEMENT stride:
         stride <= 1                 -> bail (contiguous)
         stride == 2                 -> bail (deinterleave / strided DMA)
         stride power-of-two >= 4    -> bail (strided DMA)
         static non-power-of-two >= 3 -> StrideLoadOp/StrideStoreOp
     so the expectation encoded here is element-based and dtype independent.
  3. Does the same kernel compile at all under compile_mode=simd and
     compile_mode=simt_only for that dtype (useful before spending board time)?

stride == 1 is deliberately part of the grid: it is the BAIL control that must
NOT produce a template call.

This script never launches a kernel and never writes into any pre-existing
results directory other than the one passed with --out.
"""
from __future__ import annotations

import argparse
import json
import traceback
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
    v = tl.arange(0, BLOCK) % 7
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty))


DTYPES = [
    ("int8", torch.int8),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
    ("int32", torch.int32),
    ("int64", torch.int64),
]

ELEMENT_BYTES = {
    "int8": 1, "float16": 2, "bfloat16": 2,
    "float32": 4, "int32": 4, "int64": 8,
}


def simd_opts(num_warps=1, grid=1):
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd",
        "compile_on_910_95": True,
        "enable_auto_blockify": False,
    }


def simt_opts(num_warps=32, grid=1):
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simt_only",
        "compile_on_910_95": True,
        "enable_auto_blockify": True,
    }


def simd_simt_opts(num_warps=1, grid=1):
    """compile_mode=simd_simt: the production mixed mode.

    Question under test: does a template-eligible strided access still reach the
    triton_stride_load/store template here, or does it go through the ordinary
    ttgir pipeline?  StridedLoadStoreRewrite activates on
        compile_on_910_95 && (compile_mode==SimdSimtTemplate || hasModelSelectedMemoryOp)
    so the answer is not obvious from compile_mode alone.
    """
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd_simt",
        "parallel_mode": "mix_simd_simt",
        "compile_on_910_95": True,
        "enable_auto_blockify": False,
    }


def opts_for(mode, num_warps):
    if mode == "simd_simt":
        return simd_simt_opts(num_warps=num_warps, grid=1)
    if mode == "template":
        return template_opts(num_warps=num_warps, grid=1)
    if mode == "simd":
        return simd_opts(num_warps=1)
    if mode == "simt_only":
        return simt_opts(num_warps=num_warps)
    raise ValueError(mode)


def expect_template(stride: int) -> bool:
    """Element-domain dispatch rule from StridedLoadStoreRewrite.cpp."""
    if stride <= 1:
        return False
    if stride == 2:
        return False
    if (stride & (stride - 1)) == 0:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+",
                    default=["int8", "float16", "bfloat16", "float32", "int32", "int64"])
    ap.add_argument("--modes", nargs="+", default=["template"])
    ap.add_argument("--blocks", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 255])
    ap.add_argument("--num-warps", type=int, default=32)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--asm-dir", type=Path, default=None)
    ap.add_argument("--asm-limit", type=int, default=0,
                    help="save at most N asm dumps (0 = none)")
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    dtype_map = {name: torch_dtype for name, torch_dtype in DTYPES}
    for name in args.dtypes:
        if name not in dtype_map:
            raise SystemExit(f"unknown dtype {name!r}; have {sorted(dtype_map)}")

    results = []
    asm_saved = 0
    for dtype_name in args.dtypes:
        torch_dtype = dtype_map[dtype_name]
        elem_bytes = ELEMENT_BYTES[dtype_name]
        total = 1 << 22
        x = torch.zeros(total, dtype=torch_dtype, device="npu")
        y = torch.zeros(total, dtype=torch_dtype, device="npu")
        for mode in args.modes:
            opts = opts_for(mode, args.num_warps)
            for path, kernel, kernel_args in (
                    ("triton_stride_load", check_load_kernel, (x, y)),
                    ("triton_stride_store", check_store_kernel, (y,))):
                for block in args.blocks:
                    for stride in args.strides:
                        label = (f"{dtype_name}_{mode}_"
                                 f"{path.split('_')[-1]}_b{block}_s{stride}")
                        record = {
                            "dtype": dtype_name,
                            "element_bytes": elem_bytes,
                            "mode": mode,
                            "path": path,
                            "block": int(block),
                            "stride": int(stride),
                            "byte_gap": elem_bytes * int(stride),
                            "expected_template": bool(expect_template(stride)),
                        }
                        try:
                            compiled = kernel.warmup(
                                *kernel_args, BLOCK=block, STRIDE=stride,
                                grid=(1,), **opts)
                            text = asm_text(compiled)
                            contains = path in text
                            record["compiled"] = True
                            record["contains_template_call"] = bool(contains)
                            record["match"] = bool(
                                (contains == record["expected_template"])
                                if mode == "template" else not contains)
                            record["ttadapter_has_memref_copy"] = "memref.copy" in text
                            if (args.asm_dir and args.asm_limit
                                    and asm_saved < args.asm_limit):
                                record["asm_keys"] = save_asm(compiled, args.asm_dir, label)
                                asm_saved += 1
                        except Exception as exc:  # compile failure is a result here
                            record["compiled"] = False
                            record["contains_template_call"] = None
                            record["match"] = None
                            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
                            record["traceback_tail"] = traceback.format_exc()[-600:]
                        results.append(record)
                        print(f"{dtype_name:9s} {mode:9s} {path.split('_')[-1]:6s} "
                              f"b{block:5d} s{stride:4d} gapB={elem_bytes*stride:4d} "
                              f"expected={record['expected_template']} "
                              f"contains={record.get('contains_template_call')} "
                              f"match={record.get('match')}"
                              + ("" if record["compiled"] else "  COMPILE_FAILED"),
                              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "dtype_stride_path_check",
        "compile_only": True,
        "dispatch_rule": ("element stride: <=1 bail, ==2 bail, power-of-two >=4 bail, "
                          "static non-power-of-two >=3 -> template"),
        "dtypes": args.dtypes,
        "modes": args.modes,
        "blocks": args.blocks,
        "strides": args.strides,
        "num_warps": args.num_warps,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    template_rows = [r for r in results if r["mode"] == "template"]
    mismatches = [r for r in template_rows if r.get("match") is False]
    failed = [r for r in results if not r["compiled"]]
    print(f"wrote {args.out}")
    print(f"cases={len(results)} template_cases={len(template_rows)} "
          f"mismatches={len(mismatches)} compile_failures={len(failed)}")
    if mismatches:
        for r in mismatches[:20]:
            print(f"  MISMATCH {r['dtype']} {r['path']} b{r['block']} "
                  f"s{r['stride']} expected={r['expected_template']} "
                  f"contains={r['contains_template_call']}")
    if failed:
        seen = set()
        for r in failed:
            key = (r["dtype"], r["mode"], r["error"][:80])
            if key in seen:
                continue
            seen.add(key)
            print(f"  COMPILE_FAIL {r['dtype']} {r['mode']} {r['path']} "
                  f"b{r['block']} s{r['stride']}: {r['error'][:200]}")


if __name__ == "__main__":
    main()
