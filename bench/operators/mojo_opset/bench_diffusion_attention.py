#!/usr/bin/env python3
"""Benchmark diffusion_attention: costmodel vs brute-force autotune.

Source: mojo_opset/backends/ttx/kernels/npu/diffution_attention.py
"""

from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    # ---- inputs ----
    B, H, S, D = 2, 8, 1024, 64
    q = torch.randn(B, H, S, D, device='npu', dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device='npu', dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device='npu', dtype=torch.bfloat16)
    mask = torch.tril(torch.ones(S, S, device='npu', dtype=torch.bool))

    # ---- Pipeline A ----
    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset.a5 import diffution_attention as mod_a
    common.reload_kernel_module(mod_a)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        mod_a, "kernel_sda_fwd_up",
        lambda: mod_a.diffusion_attention_fwd_impl(q, k, v, mask=mask),
        tuner_fallbacks=None,
    )
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    # ---- Pipeline B ----
    common.setup_costmodel_env(False)
    from kernels.mojo_opset.a5 import diffution_attention as mod_b
    common.reload_kernel_module(mod_b)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        mod_b, "kernel_sda_fwd_up",
        lambda: mod_a.diffusion_attention_fwd_impl(q, k, v, mask=mask),
        tuner_fallbacks=None,
    )
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw,
                           b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_diffusion_attention.json"), "w") as f:
        json.dump({"operator": "diffusion_attention", "top_k": TOP_K, "speedup": b_elapsed / a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
