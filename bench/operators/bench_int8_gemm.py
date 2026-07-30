#!/usr/bin/env python3
"""Benchmark INT8 GEMM: costmodel vs brute-force autotune.

Source: mojo_opset/.../kernels/npu/a2/int8_gemm.py
"""

from __future__ import annotations
import json, os, sys
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "5"))


def main():
    M, K, N = 1024, 4096, 8192
    A = torch.randint(-128, 127, (M, K), device="npu", dtype=torch.int8)
    B = torch.randint(-128, 127, (K, N), device="npu", dtype=torch.int8)

    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset import int8_gemm as ia
    common.reload_kernel_module(ia)
    common.warmup_cache()
    def _run_a():
        b_t = ia.prepare_b_impl(B)
        return ia.int8_gemm_dequant_impl(A, b_t, torch.tensor(1.0, device="npu", dtype=torch.float32),
                                         torch.tensor(1.0, device="npu", dtype=torch.float32), None, M, N, torch.float16)
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        ia, "_int8_gemm_dequant_kernel", _run_a, args=())
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    common.setup_costmodel_env(False)
    from kernels.mojo_opset import int8_gemm as ib
    common.reload_kernel_module(ib)
    common.warmup_cache()
    def _run_b():
        b_t = ib.prepare_b_impl(B)
        return ib.int8_gemm_dequant_impl(A, b_t, torch.tensor(1.0, device="npu", dtype=torch.float32),
                                         torch.tensor(1.0, device="npu", dtype=torch.float32), None, M, N, torch.float16)
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        ib, "_int8_gemm_dequant_kernel", _run_b, args=())
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw, b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_int8_gemm.json"), "w") as f:
        json.dump({"operator": "int8_gemm", "top_k": TOP_K, "speedup": b_elapsed/a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
