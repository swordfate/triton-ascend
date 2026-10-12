#!/usr/bin/env python3
"""Compile-only path check for 2D (rank-2) strided tiles.

Address set of one tile:

    addr(r, c) = r * OS + c * IS      r in [0,R), c in [0,C)

Contiguity of the whole tile (the rule used in the README):

    fully contiguous  <=>  IS == 1 and OS == C
    strided           <=>  anything else (holes in the inner axis, in the
                           outer axis, or both)

The template dispatch in StridedLoadStoreRewrite.cpp looks only at the
INNERMOST per-axis stride:

    lastStride <= 1              -> bail (contiguous inner axis)
    lastStride == 2              -> bail (deinterleave / strided DMA)
    lastStride power-of-two >= 4 -> bail (strided DMA)
    static non-power-of-two >= 3 -> ttasc.stride_load/store -> template

so a 2D tile only reaches the template when its inner axis is strided by a
non-power-of-two >= 3.  This script records, per named form and per lowering
mode, whether the template call appears and which marker instructions show up.

Compile only: no kernel is launched.
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
def check2d_load_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr,
                        OS: tl.constexpr, IS: tl.constexpr):
    offs = tl.arange(0, R)[:, None] * OS + tl.arange(0, C)[None, :] * IS
    v = tl.load(x_ptr + offs)
    dst = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    tl.store(out_ptr + dst, v)


@triton.jit
def check2d_store_kernel(out_ptr, R: tl.constexpr, C: tl.constexpr,
                         OS: tl.constexpr, IS: tl.constexpr):
    offs = tl.arange(0, R)[:, None] * OS + tl.arange(0, C)[None, :] * IS
    v = ((tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]) % 7)
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty))


MARKERS = ["triton_stride_load", "triton_stride_store", "memref.copy",
           "SIMT_LDG", "SIMT_STG", "MOV_SRC_TO_DST", "MTE2", "MTE3",
           "stride_load", "stride_store"]


def classify(rows, cols, outer_stride, inner_stride):
    """Axis-wise classification of one rank-2 tile.

    A rank-2 tile counts as 2D-strided only when BOTH axes are strided.  A
    rank-2 tile whose row dimension is 1 is a real lowering shape and must
    also be modelled, so it gets its own class.

        contiguous            IS == 1 and OS == C
        both_strided          IS >= 2 and OS > (C-1)*IS + 1
        inner_strided_only    IS >= 2 and OS == (C-1)*IS + 1
        outer_strided_only    IS == 1 and OS > C
        singleton_row_*       R == 1
    """
    rows, cols = int(rows), int(cols)
    outer_stride, inner_stride = int(outer_stride), int(inner_stride)
    if inner_stride == 1 and outer_stride == cols:
        return "contiguous"
    inner = inner_stride >= 2
    outer = outer_stride > (cols - 1) * inner_stride + 1
    if rows == 1:
        return "singleton_row_strided" if inner else "singleton_row_outer"
    if inner and outer:
        return "both_strided"
    if inner:
        return "inner_strided_only"
    return "outer_strided_only"


def forms(rows, cols, outer_pad=7):
    """(name, R, C, OS, IS) for one tile shape.

    both_sK_tight : OS == C*K  -> outer gap of K-1 elements, so BOTH axes
                   carry holes for K >= 2.  This is the primary 2D family.
    both_sK_gap   : OS == C*K + pad -> larger holes on both axes.
    contig / rowgap are boundary controls only; they never enter a strided fit.
    """
    out = [("contig", rows, cols, cols, 1)]
    if int(rows) > 1:
        out.append(("rowgap", rows, cols, cols + outer_pad, 1))
    for k in (2, 3, 4, 5):
        out.append((f"both_s{k}_tight", rows, cols, cols * k, k))
        out.append((f"both_s{k}_gap", rows, cols, cols * k + outer_pad, k))
    return out


def simd_opts(num_warps=1, grid=1):
    return {"num_warps": 1, "logical_program_count_hint": grid,
            "physical_vector_core_count_hint": 1, "superblock_factor": 1,
            "auto_simt_scope_mode": "off", "compile_mode": "simd",
            "compile_on_910_95": True, "enable_auto_blockify": False}


def simt_opts(num_warps=32, grid=1):
    return {"num_warps": int(num_warps), "logical_program_count_hint": grid,
            "physical_vector_core_count_hint": 1, "superblock_factor": 1,
            "auto_simt_scope_mode": "off", "compile_mode": "simt_only",
            "compile_on_910_95": True, "enable_auto_blockify": True}


def opts_for(mode, num_warps):
    if mode == "template":
        return template_opts(num_warps=num_warps, grid=1)
    if mode == "simd":
        return simd_opts()
    if mode == "simt_only":
        return simt_opts(num_warps)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["1x64", "8x16", "16x32"])
    ap.add_argument("--only-forms", nargs="+", default=None,
                    help="restrict to these named forms")
    ap.add_argument("--modes", nargs="+", default=["template", "simd", "simt_only"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--num-warps", type=int, default=32)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--asm-dir", type=Path, default=None)
    ap.add_argument("--asm-limit", type=int, default=0)
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "int8": torch.int8}
    torch_dtype = dtype_map[args.dtype]
    torch_npu.npu.set_device(args.device)

    total = 1 << 22
    x = torch.zeros(total, dtype=torch_dtype, device="npu")
    y = torch.zeros(total, dtype=torch_dtype, device="npu")

    results = []
    saved = 0
    for shape in args.shapes:
        rows, cols = (int(v) for v in shape.split("x"))
        for name, R, C, OS, IS in forms(rows, cols):
            if args.only_forms and name not in args.only_forms:
                continue
            max_addr = (R - 1) * OS + (C - 1) * IS
            if max_addr >= total:
                raise SystemExit(f"{name} {shape} needs {max_addr} elements")
            for mode in args.modes:
                for op, kernel, kargs in (("load", check2d_load_kernel, (x, y)),
                                          ("store", check2d_store_kernel, (y,))):
                    record = {"shape": shape, "form": name, "R": R, "C": C,
                              "OS": OS, "IS": IS,
                              "axis_class": classify(R, C, OS, IS),
                              "contiguous": bool(IS == 1 and OS == C),
                              "elements": R * C, "mode": mode, "op": op,
                              "max_addr": max_addr,
                              "inner_stride_non_pow2_ge3": bool(
                                  IS >= 3 and (IS & (IS - 1)) != 0)}
                    label = f"2d_{op}_{mode}_{name}_{shape}"
                    try:
                        compiled = kernel.warmup(
                            *kargs, R=R, C=C, OS=OS, IS=IS, grid=(1,),
                            **opts_for(mode, args.num_warps))
                        text = asm_text(compiled)
                        record["compiled"] = True
                        record["markers"] = {m: (m in text) for m in MARKERS}
                        record["contains_template_call"] = bool(
                            record["markers"].get(
                                "triton_stride_load" if op == "load"
                                else "triton_stride_store", False))
                        if args.asm_dir and args.asm_limit and saved < args.asm_limit:
                            record["asm_keys"] = save_asm(compiled, args.asm_dir, label)
                            saved += 1
                    except Exception as exc:
                        record["compiled"] = False
                        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
                    results.append(record)
                    mk = record.get("markers", {})
                    print(f"{shape:6s} {name:16s} IS={IS:2d} OS={OS:4d} "
                          f"{record['axis_class']:20s} {mode:9s} {op:5s} "
                          f"compiled={record['compiled']} "
                          f"template={record.get('contains_template_call')} "
                          f"memrefcopy={mk.get('memref.copy')} "
                          f"SIMT_LDG={mk.get('SIMT_LDG')} "
                          f"MOV={mk.get('MOV_SRC_TO_DST')}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "kind": "stride2d_path_check",
        "compile_only": True,
        "dtype": args.dtype,
        "shapes": args.shapes,
        "modes": args.modes,
        "contiguity_rule": "contiguous <=> IS == 1 and OS == C",
        "template_rule": "inner (last) stride static non-power-of-two >= 3",
        "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {args.out} cases={len(results)} "
          f"compile_failures={sum(1 for r in results if not r['compiled'])}")


if __name__ == "__main__":
    main()
