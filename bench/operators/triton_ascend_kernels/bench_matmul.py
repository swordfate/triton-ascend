#!/usr/bin/env python3
"""Benchmark Matmul: costmodel-predicted vs brute-force autotune.

Kernel source (copied as-is, imports only adjusted):
    kernels/matmul.py  ←  triton-ascend-kernels/src/triton_ascend_kernels/gemm/matmul.py

Usage::

    TRITON_COSTMODEL_TOP_K=5 python bench/operators/bench_matmul.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import triton  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as common

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    M, K, N = 1024, 4096, 8192
    mat_a = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
    mat_b = torch.randn(K, N, device="npu", dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    
    # Pipeline A: costmodel pre-filter
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=True, top_k=TOP_K)
    from kernels.triton_ascend_kernels import matmul as matmul_a
    common.reload_kernel_module(matmul_a)
    common.warmup_cache()

    print("=== Pipeline A: costmodel pre-filter ===")
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        matmul_a,
        "matmul_kernel",
        matmul_a.matmul,
        args=(mat_a, mat_b),
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ------------------------------------------------------------------
    # Pipeline B: brute-force (HW bench all configs)
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=False)
    from kernels.triton_ascend_kernels import matmul as matmul_b
    common.reload_kernel_module(matmul_b)
    common.warmup_cache()

    print("=== Pipeline B: brute-force ===")
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        matmul_b,
        "matmul_kernel",
        matmul_b.matmul,
        args=(mat_a, mat_b),
    )
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    common.print_comparison(
        a_elapsed, a_timings, a_preds, a_cm, a_hw,
        b_elapsed, b_timings, b_hw,
        TOP_K,
    )

    # Save results
    results = {
        "operator": "matmul",
        "shape": {"M": M, "K": K, "N": N},
        "top_k": TOP_K,
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": b_elapsed / a_elapsed if a_elapsed > 0 else float("inf"),
        "predictions": {str(k): v for k, v in a_preds.items()},
        "a_timings_s": {str(k): v for k, v in a_timings.items()},
        "b_timings_s": {str(k): v for k, v in b_timings.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_matmul.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
