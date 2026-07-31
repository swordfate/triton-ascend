#!/usr/bin/env python3
"""Benchmark Grouped GEMM (M-dim): costmodel-predicted vs brute-force autotune.

Kernel source (copied as-is, imports only adjusted):
    kernels/group_gemm.py  ←  mojo_opset/mojo_opset/backends/ttx/kernels/npu/a2/group_gemm.py

Usage::

    TRITON_COSTMODEL_TOP_K=6 python bench/operators/bench_group_gemm.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
import triton  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "6"))


def main() -> None:
    # Inputs for grouped matmul: 2 groups of (512×2048) × (2048×4096)
    num_groups = 2
    M, K, N = 512, 2048, 4096
    size_per_group = torch.tensor([M, M], device="npu", dtype=torch.int32)
    A = torch.randn(num_groups * M, K, device="npu", dtype=torch.bfloat16)
    B = torch.randn(num_groups, K, N, device="npu", dtype=torch.bfloat16)
    C = torch.empty(num_groups * M, N, device="npu", dtype=torch.bfloat16)

    
    # Pipeline A
    common.setup_costmodel_env(enable=True, top_k=TOP_K)
    from kernels.mojo_opset import group_gemm as gg_a
    common.reload_kernel_module(gg_a)
    common.warmup_cache()

    print("=== Pipeline A: costmodel pre-filter ===")
    def _run_a():
        return gg_a.m_grouped_matmul_impl(
            A=A, B=B, C=C, size_per_group=size_per_group,
            num_groups=num_groups, M=M, N=N, K=K,
            strideBN=N, strideBK=K,
        )
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        gg_a,
        "_m_grouped_matmul_bKmajor_kernel",
        _run_a,
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # Pipeline B
    common.setup_costmodel_env(enable=False)
    from kernels.mojo_opset import group_gemm as gg_b
    common.reload_kernel_module(gg_b)
    common.warmup_cache()

    print("=== Pipeline B: brute-force ===")
    def _run_b():
        return gg_b.m_grouped_matmul_impl(
            A=A, B=B, C=C, size_per_group=size_per_group,
            num_groups=num_groups, M=M, N=N, K=K,
            strideBN=N, strideBK=K,
        )
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        gg_b,
        "_m_grouped_matmul_bKmajor_kernel",
        _run_b,
    )
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(
        a_elapsed, a_timings, a_preds, a_cm, a_hw,
        b_elapsed, b_timings, b_hw,
        TOP_K,
    )

    results = {
        "operator": "group_gemm",
        "shape": {"groups": num_groups, "M": M, "K": K, "N": N},
        "top_k": TOP_K,
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": b_elapsed / a_elapsed if a_elapsed > 0 else float("inf"),
        "predictions": {str(k): v for k, v in a_preds.items()},
        "a_timings_s": {str(k): v for k, v in a_timings.items()},
        "b_timings_s": {str(k): v for k, v in b_timings.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_group_gemm.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
