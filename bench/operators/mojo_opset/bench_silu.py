#!/usr/bin/env python3
"""Benchmark silu: costmodel vs brute-force autotune.

Source: mojo_opset/backends/ttx/kernels/npu/silu.py
"""

from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    # ---- inputs ----
    x = torch.randn(4, 1024, 4096, device='npu', dtype=torch.float16)

    # ---- Pipeline A ----
    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset.a5 import silu as mod_a
    common.reload_kernel_module(mod_a)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        mod_a, "_silu_fwd_kernel",
        lambda: mod_a.silu_fwd_impl(x),
        tuner_fallbacks=['_silu_fwd_nomask_kernel', '_silu_fwd_flatten_kernel'],
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ---- Pipeline B ----
    common.setup_costmodel_env(False)
    from kernels.mojo_opset.a5 import silu as mod_b
    common.reload_kernel_module(mod_b)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        mod_b, "_silu_fwd_kernel",
        lambda: mod_a.silu_fwd_impl(x),
        tuner_fallbacks=['_silu_fwd_nomask_kernel', '_silu_fwd_flatten_kernel'],
    )
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw,
                           b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_silu.json"), "w") as f:
        json.dump({"operator": "silu", "top_k": TOP_K, "speedup": b_elapsed / a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
