#!/usr/bin/env python3
"""Dump Triton compile artifacts for a strided test case/mode."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


def load_module(path):
    spec = importlib.util.spec_from_file_location("dump_case_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_asm(compiled, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in (getattr(compiled, "asm", None) or {}).items():
        if isinstance(value, bytes):
            (out_dir / f"{label}.{key}").write_bytes(value)
        elif isinstance(value, str):
            (out_dir / f"{label}.{key}").write_text(value, encoding="utf-8", errors="replace")
    metadata = getattr(compiled, "metadata", None)
    if metadata:
        (out_dir / f"{label}.metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["load", "store"])
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--module", type=Path, required=True)
    ap.add_argument("--num-warps", type=int, default=32)
    ap.add_argument("--logical", type=int, default=16)
    ap.add_argument("--physical", type=int, default=56)
    ap.add_argument("--enable-auto-blockify", action="store_true", default=True)
    ap.add_argument("--disable-auto-blockify", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--simd-simt-profile", default="")
    ap.add_argument("--auto-scope-mode", default="report")
    args = ap.parse_args()

    os.environ["TRITON_CACHE_DIR"] = str(args.out.parent / "cache_dump")
    module = load_module(args.module)
    if args.case == "load":
        src = torch.randn(4096, device="npu", dtype=torch.float32)
        dst = torch.zeros(1024, device="npu", dtype=torch.float32)
        args_kernel = (src, dst, 3, 1024)
        grid = (triton.cdiv(1024, 64),)
        kwargs = dict(BLOCK_SIZE=64)
    else:
        src = torch.randn(2048, device="npu", dtype=torch.float32)
        dst = torch.zeros(2048 * 3, device="npu", dtype=torch.float32)
        args_kernel = (src, dst, 2048)
        grid = (triton.cdiv(2048, 1024),)
        kwargs = dict(BLOCK_SIZE=1024, STRIDE=3)
    enable = not args.disable_auto_blockify
    opts = dict(num_warps=args.num_warps, compile_mode=args.mode,
                auto_simt_scope_mode="off" if args.mode != "simd_simt" else args.auto_scope_mode,
                enable_auto_blockify=enable,
                logical_program_count_hint=args.logical,
                physical_vector_core_count_hint=args.physical,
                debug=args.debug, auto_simt_model_profile=args.simd_simt_profile)
    kernel = module.strided_kernel if args.case == "load" else module.vcompute_stridestore_const_kernel
    compiled = kernel.warmup(*args_kernel, grid=grid, **kwargs, **opts)
    save_asm(compiled, args.out, f"{args.case}_{args.mode}")
    md = compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict") else dict(compiled.metadata)
    keys = ["auto_blockify_v1_enabled", "auto_blockify_v1_runtime_cap",
            "ta_auto_blockify_v1_materialized", "auto_simt_costmodel_analysis_ir",
            "auto_simt_effective_kind", "compile_mode", "parallel_mode",
            "route_transform_v1_materializable", "logical_program_count_hint",
            "ttir_layout_merge_applied", "ttir_layout_coalesce_factor",
            "ttir_layout_coalesce_axis", "ttir_layout_coalesce_grid_ceil_div",
            "auto_blockify_v1_requested", "auto_blockify_v1_selection_source",
            "auto_simt_analysis_ir_materialized"]
    info = {
        "case": args.case,
        "mode": args.mode,
        "opts": opts,
        "asm_keys": sorted((getattr(compiled, "asm", None) or {}).keys()),
    }
    for key in keys:
        info[key] = md.get(key)
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
