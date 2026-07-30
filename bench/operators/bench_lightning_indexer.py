#!/usr/bin/env python3
"""Benchmark Lightning Indexer: costmodel vs brute-force autotune.

Source: mojo_opset/.../kernels/npu/a2/lightning_indexer.py
"""

from __future__ import annotations
import json, os, sys
import torch
import bench_common as common

TOP_K = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "2"))


def main():
    B, H, M, N, K = 1, 4, 64, 1024, 128
    q = torch.randn(B, H, M, K, device="npu", dtype=torch.float8_e5m2)
    q_s = torch.randn(B, H, M, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, N, K, device="npu", dtype=torch.float8_e5m2)
    k_s = torch.randn(B, H, N, device="npu", dtype=torch.float32)

    common.setup_costmodel_env(True, TOP_K)
    from kernels.mojo_opset import lightning_indexer as lia
    common.reload_kernel_module(lia)
    common.warmup_cache()
    a_elapsed, a_timings, a_preds, a_best, a_cm, a_hw = common.run_autotune_pass(
        lia, "lightning_indexer_kernel",
        lambda: lia.lightning_indexer_impl(q, q_s, k, k_s), args=())
    common.print_results("A", a_elapsed, a_timings, a_best, a_preds, a_cm, a_hw)

    common.setup_costmodel_env(False)
    from kernels.mojo_opset import lightning_indexer as lib
    common.reload_kernel_module(lib)
    common.warmup_cache()
    b_elapsed, b_timings, _, b_best, _, b_hw = common.run_autotune_pass(
        lib, "lightning_indexer_kernel",
        lambda: lib.lightning_indexer_impl(q, q_s, k, k_s), args=())
    common.print_results("B", b_elapsed, b_timings, b_best, hw_timing=b_hw)

    common.print_comparison(a_elapsed, a_timings, a_preds, a_cm, a_hw, b_elapsed, b_timings, b_hw, TOP_K)
    with open(os.path.join(os.path.dirname(__file__), "results_lightning_indexer.json"), "w") as f:
        json.dump({"operator": "lightning_indexer", "top_k": TOP_K, "speedup": b_elapsed/a_elapsed if a_elapsed else float("inf")}, f, indent=2)
    print("Results saved")


common.cleanup_cache_dirs()
if __name__ == "__main__":
    main()
