#!/usr/bin/env python3
"""Benchmark SDPA forward: costmodel-predicted vs brute-force autotune.

Kernel source (copied as-is, no import changes needed):
    kernels/sdpa.py  ←  Q2TritonKernel/src/kernels/sdpa.py

Uses ``kernel_sdpa_fwd`` (the training SDPA forward with active autotune).

Usage::

    TRITON_COSTMODEL_TOP_K=3 python bench/operators/bench_sdpa.py
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

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    B, H, S, D = 2, 8, 1024, 64  # batch, heads, seq_len, head_dim
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
    # Causal mask — upper triangle is False (masked out)
    mask = torch.tril(torch.ones(S, S, device="npu", dtype=torch.bool))

    # ------------------------------------------------------------------
    
    # Pipeline A: costmodel pre-filter
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=True, top_k=TOP_K)
    from kernels.q2tritonkernel import sdpa as sdpa_a
    common.reload_kernel_module(sdpa_a)
    common.warmup_cache()

    print("=== Pipeline A: costmodel pre-filter ===")
    # We use a thin wrapper because sdpa_fwd_impl has complex arg handling
    def _run_sdpa_a():
        return sdpa_a.sdpa_fwd_impl(q, k, v, mask=mask)

    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        sdpa_a,
        "kernel_sdpa_fwd",
        _run_sdpa_a,
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ------------------------------------------------------------------
    # Pipeline B: brute-force (HW bench all configs)
    # ------------------------------------------------------------------
    common.setup_costmodel_env(enable=False)
    from kernels.q2tritonkernel import sdpa as sdpa_b
    common.reload_kernel_module(sdpa_b)
    common.warmup_cache()

    print("=== Pipeline B: brute-force ===")
    def _run_sdpa_b():
        return sdpa_b.sdpa_fwd_impl(q, k, v, mask=mask)

    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        sdpa_b,
        "kernel_sdpa_fwd",
        _run_sdpa_b,
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
        "operator": "sdpa_fwd",
        "shape": {"B": B, "H": H, "S": S, "D": D},
        "top_k": TOP_K,
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": b_elapsed / a_elapsed if a_elapsed > 0 else float("inf"),
        "predictions": {str(k): v for k, v in a_preds.items()},
        "a_timings_s": {str(k): v for k, v in a_timings.items()},
        "b_timings_s": {str(k): v for k, v in b_timings.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_sdpa.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
