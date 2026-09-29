#!/usr/bin/env python3
"""Profile N launches of one strided route and report kernel Duration(us)."""
import argparse
import csv
import importlib.util
import os
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


def load_module(path):
    spec = importlib.util.spec_from_file_location("profile_case_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["load", "store", "empty"])
    ap.add_argument("--compile-mode", default="simd_simt")
    ap.add_argument("--auto-scope-mode", default="auto")
    ap.add_argument("--num-warps", type=int, default=32)
    ap.add_argument("--logical", type=int, default=16)
    ap.add_argument("--physical", type=int, default=56)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--profile", default=None, help="path to a forcing SIMD/SIMT profile")
    ap.add_argument("--simd-simt-profile", default="")
    ap.add_argument("--module", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.profile:
        os.environ["TRITON_ASCEND_SIMD_SIMT_PROFILE"] = args.profile
        args.simd_simt_profile = args.profile
    if args.out:
        os.environ["TRITON_CACHE_DIR"] = str(args.out.parent / f"cache_profile_{args.case}")

    if args.case == "load":
        module_path = args.module or Path("/home/c00946898/strided-next/test_cases/strided_kernels/test_stride_load.py.py")
        mod = load_module(module_path)
        n = 1024
        src = torch.randn(4096, device="npu", dtype=torch.float32)
        dst = torch.zeros(n, device="npu", dtype=torch.float32)
        launch = lambda opts: mod.strided_kernel[(triton.cdiv(n, 64),)](
            src, dst, 3, n, BLOCK_SIZE=64, **opts)
    elif args.case == "store":
        module_path = args.module or Path("/home/c00946898/strided-next/test_cases/strided_kernels/test_stridestore_const.py.py")
        mod = load_module(module_path)
        n = 2048
        src = torch.randn(n, device="npu", dtype=torch.float32)
        dst = torch.zeros(n * 3, device="npu", dtype=torch.float32)
        launch = lambda opts: mod.vcompute_stridestore_const_kernel[(triton.cdiv(n, 1024),)](
            src, dst, n, BLOCK_SIZE=1024, STRIDE=3, **opts)
    else:
        @triton.jit
        def empty_kernel(out_ptr):
            pid = tl.program_id(0)
            tl.store(out_ptr + pid, 1.0)
        out = torch.zeros(64, device="npu")
        launch = lambda opts: empty_kernel[(args.logical,)](out, **opts)

    opts = dict(num_warps=args.num_warps, compile_mode=args.compile_mode,
                auto_simt_scope_mode=args.auto_scope_mode,
                enable_auto_blockify=True,
                logical_program_count_hint=args.logical,
                physical_vector_core_count_hint=args.physical,
                auto_simt_model_profile=args.simd_simt_profile)
    for _ in range(10):
        launch(opts)
    torch.npu.synchronize()

    root = args.out.parent / f"profiler_{args.case}" if args.out else Path(f"/tmp/profiler_{args.case}")
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root)),
            record_shapes=False, profile_memory=False, with_stack=False,
            with_flops=False, with_modules=False) as profiler:
        for _ in range(args.reps):
            launch(opts)
    torch.npu.synchronize()
    durations = []
    files = list(root.rglob("kernel_details.csv"))
    for file in files:
        with file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("Duration(us)")
                if value:
                    durations.append(float(value))
    result = {
        "case": args.case,
        "compile_mode": args.compile_mode,
        "auto_scope_mode": args.auto_scope_mode,
        "profile": args.profile,
        "num_warps": args.num_warps,
        "logical": args.logical,
        "n": len(durations),
        "median_us": statistics.median(durations) if durations else None,
        "min_us": min(durations) if durations else None,
        "max_us": max(durations) if durations else None,
        "files": [str(f) for f in files],
    }
    print(result)
    if args.out:
        import json
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
