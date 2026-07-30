#!/usr/bin/env python3
"""Benchmark Dynamic Quant: costmodel vs brute-force autotune.

Source: mojo_opset/.../kernels/npu/a2/quant.py
"""

from __future__ import annotations
import json, os, sys
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "4"))


def main():
    x = torch.randn(4, 1024, 4096, device="npu", dtype=torch.float16)
    dims = 4096

    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset import quant as qa
    common.reload_kernel_module(qa)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        qa, "scale_dynamic_quant_kernel", lambda: qa.dynamic_quant_impl(x), args=())
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    common.setup_costmodel_env(False)
    from kernels.mojo_opset import quant as qb
    common.reload_kernel_module(qb)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        qb, "scale_dynamic_quant_kernel", lambda: qb.dynamic_quant_impl(x), args=())
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw, b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_quant.json"), "w") as f:
        json.dump({"operator": "quant", "top_k": TOP_K, "speedup": b_elapsed/a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
