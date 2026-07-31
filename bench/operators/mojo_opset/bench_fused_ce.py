#!/usr/bin/env python3
"""Benchmark Fused Linear Cross Entropy: costmodel vs brute-force autotune.

Source: mojo_opset/.../kernels/npu/a2/fused_linear_cross_entropy.py
"""

from __future__ import annotations
import json, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    B, S, H, V = 2, 1024, 4096, 32000
    hidden = torch.randn(B*S, H, device="npu", dtype=torch.float16)
    weight = torch.randn(V, H, device="npu", dtype=torch.float16)
    target = torch.randint(0, V, (B*S,), device="npu")

    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset import fused_linear_cross_entropy as cea
    common.reload_kernel_module(cea)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        cea, "_cross_entropy_kernel",
        lambda: cea.fused_linear_cross_entropy_fwd_impl(
            hidden, weight, target, None, None, -100, 0.0, 0.0, "mean", 0.0, False, torch.float16), args=())
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    common.setup_costmodel_env(False)
    from kernels.mojo_opset import fused_linear_cross_entropy as ceb
    common.reload_kernel_module(ceb)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        ceb, "_cross_entropy_kernel",
        lambda: ceb.fused_linear_cross_entropy_fwd_impl(
            hidden, weight, target, None, None, -100, 0.0, 0.0, "mean", 0.0, False, torch.float16), args=())
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw, b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_fused_ce.json"), "w") as f:
        json.dump({"operator": "fused_linear_cross_entropy", "top_k": TOP_K, "speedup": b_elapsed/a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
