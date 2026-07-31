#!/usr/bin/env python3
"""Benchmark Causal Conv1d: costmodel vs brute-force autotune.

Source: mojo_opset/.../kernels/npu/a2/convolution.py
"""

from __future__ import annotations
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    B, L, D = 2, 1024, 128
    x = torch.randn(B, L, D, device="npu", dtype=torch.float16)
    weight = torch.randn(D, 4, D, device="npu", dtype=torch.float16)
    cu_seqlens = torch.tensor([0, L, 2*L], device="npu", dtype=torch.int32)

    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset import convolution as ca
    common.reload_kernel_module(ca)
    common.warmup_cache()
    def _run_a():
        return ca.causal_conv1d_fwd_impl(x, weight, None, None, None, False, "silu", cu_seqlens)
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        ca, "causal_conv1d_fwd_kernel", _run_a, args=())
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    common.setup_costmodel_env(False)
    from kernels.mojo_opset import convolution as cb
    common.reload_kernel_module(cb)
    common.warmup_cache()
    def _run_b():
        return cb.causal_conv1d_fwd_impl(x, weight, None, None, None, False, "silu", cu_seqlens)
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        cb, "causal_conv1d_fwd_kernel", _run_b, args=())
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw, b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_convolution.json"), "w") as f:
        json.dump({"operator": "convolution", "top_k": TOP_K, "speedup": b_elapsed/a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
