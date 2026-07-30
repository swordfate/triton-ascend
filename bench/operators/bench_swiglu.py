#!/usr/bin/env python3
"""Benchmark SwiGLU: costmodel-predicted vs brute-force autotune.

Kernel source (copied as-is, imports only adjusted):
    kernels/activation/swiglu.py  ←  Q2TritonKernel/src/kernels/activation/swiglu.py

Usage::

    TRITON_COSTMODEL_TOP_K=5 python bench/operators/bench_swiglu.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import triton  # noqa: F401

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
    shape = (4, 1024, 4096)  # B, T, H
    a = torch.randn(shape, device="npu", dtype=torch.float16)
    b = torch.randn(shape, device="npu", dtype=torch.float16)

    # ------------------------------------------------------------------
    
    # Pipeline A: costmodel pre-filter
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=True, top_k=TOP_K)
    from kernels.q2tritonkernel.activation import swiglu as swiglu_a
    common.reload_kernel_module(swiglu_a)
    common.warmup_cache()

    print("=== Pipeline A: costmodel pre-filter ===")
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        swiglu_a,
        "_swiglu_fwd_kernel",
        swiglu_a.swiglu_fwd_impl,
        args=(a, b),
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ------------------------------------------------------------------
    # Pipeline B: brute-force (HW bench all configs)
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=False)
    from kernels.q2tritonkernel.activation import swiglu as swiglu_b
    common.reload_kernel_module(swiglu_b)
    common.warmup_cache()

    print("=== Pipeline B: brute-force ===")
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        swiglu_b,
        "_swiglu_fwd_kernel",
        swiglu_b.swiglu_fwd_impl,
        args=(a, b),
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
        "operator": "swiglu",
        "shape": list(shape),
        "top_k": TOP_K,
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": b_elapsed / a_elapsed if a_elapsed > 0 else float("inf"),
        "predictions": {str(k): v for k, v in a_preds.items()},
        "a_timings_s": {str(k): v for k, v in a_timings.items()},
        "b_timings_s": {str(k): v for k, v in b_timings.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_swiglu.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
