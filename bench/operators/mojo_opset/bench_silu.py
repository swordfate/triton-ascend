#!/usr/bin/env python3
"""Benchmark SiLU: costmodel-predicted vs brute-force autotune.

Kernel source (copied as-is, imports only adjusted):
    kernels/silu.py  ←  mojo_opset/mojo_opset/backends/ttx/kernels/npu/a2/silu.py

Usage::

    TRITON_COSTMODEL_TOP_K=5 python bench/operators/bench_silu.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
import triton  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "5"))


def main() -> None:
    shape = (4, 1024, 4096)
    x = torch.randn(shape, device="npu", dtype=torch.float16)

    
    # Pipeline A
    common.setup_costmodel_env(enable=True, top_k=TOP_K)
    from kernels.mojo_opset import silu as silu_a
    common.reload_kernel_module(silu_a)
    common.warmup_cache()

    # NOTE: silu_fwd_impl selects from 3 kernels based on shape/heuristics:
    #  _silu_fwd_kernel (9 confs) / _silu_fwd_nomask_kernel (5) /
    #  _silu_fwd_nomask_single_kernel (5).  tuner_fallbacks auto-detect.
    _silu_fallbacks = ["_silu_fwd_kernel", "_silu_fwd_nomask_kernel", "_silu_fwd_nomask_single_kernel"]

    print("=== Pipeline A: costmodel pre-filter ===")
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        silu_a, "_silu_fwd_kernel", silu_a.silu_fwd_impl, args=(x,),
        tuner_fallbacks=_silu_fallbacks)
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # Pipeline B
    common.setup_costmodel_env(enable=False)
    from kernels.mojo_opset import silu as silu_b
    common.reload_kernel_module(silu_b)
    common.warmup_cache()

    print("=== Pipeline B: brute-force ===")
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        silu_b, "_silu_fwd_kernel", silu_b.silu_fwd_impl, args=(x,),
        tuner_fallbacks=_silu_fallbacks)
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(
        a_elapsed, a_timings, a_preds, a_cm, a_hw,
        b_elapsed, b_timings, b_hw,
        TOP_K,
    )

    results = {
        "operator": "silu",
        "shape": list(shape),
        "top_k": TOP_K,
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": b_elapsed / a_elapsed if a_elapsed > 0 else float("inf"),
        "predictions": {str(k): v for k, v in a_preds.items()},
        "a_timings_s": {str(k): v for k, v in a_timings.items()},
        "b_timings_s": {str(k): v for k, v in b_timings.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "results_silu.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
