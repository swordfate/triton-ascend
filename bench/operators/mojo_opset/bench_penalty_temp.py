#!/usr/bin/env python3
"""Benchmark penalty_temp: costmodel vs brute-force autotune.

Source: mojo_opset/backends/ttx/kernels/npu/sample.py
"""

from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    # ---- inputs ----
    B, V = 2, 32000
    logits = torch.randn(B, V, device='npu', dtype=torch.float32)

    # ---- Pipeline A ----
    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset.a5 import sample as mod_a
    common.reload_kernel_module(mod_a)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        mod_a, "_fused_penalty_temp_kernel",
        lambda: mod_a.fused_penalties_temp_impl(logits, freqs=torch.zeros(1, device='npu')),
        tuner_fallbacks=None,
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ---- Pipeline B ----
    common.setup_costmodel_env(False)
    from kernels.mojo_opset.a5 import sample as mod_b
    common.reload_kernel_module(mod_b)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        mod_b, "_fused_penalty_temp_kernel",
        lambda: mod_b.fused_penalties_temp_impl(logits, freqs=torch.zeros(1, device='npu')),
        tuner_fallbacks=None,
    )
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw,
                           b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_penalty_temp.json"), "w") as f:
        json.dump({"operator": "penalty_temp", "top_k": TOP_K, "speedup": b_elapsed / a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
