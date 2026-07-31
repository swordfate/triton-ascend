#!/usr/bin/env python3
"""Benchmark: costmodel-preselected autotune vs brute-force for matmul.

Pipeline A (costmodel):
    configs -> costmodel_bench() -> top-k prediction -> HW bench top-k -> best

Pipeline B (brute-force):
    configs -> HW bench all configs -> best

Usage::

    TRITON_COSTMODEL_TOP_K=3 python bench/bench_costmodel_matmul.py
"""

from __future__ import annotations

import json, os, shutil, time
from typing import Any, Dict, List

os.environ.setdefault("TRITON_COSTMODEL_SAVE_PREDICTIONS", "1")
os.environ.setdefault("TRITON_COSTMODEL_REUSE_TTIR", "0")

import torch
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # triggers _patch_autotune()


# ---------------------------------------------------------------------------
# Configs — simple tile-size sweep
# ---------------------------------------------------------------------------

def _build_configs() -> List:
    block_ms = [64, 128, 256]
    block_ns = [64, 128, 256]
    block_ks = [64, 128]
    configs = []
    for bm in block_ms:
        for bn in block_ns:
            for bk in block_ks:
                configs.append(triton.Config(
                    kwargs={"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_stages=2, num_warps=4,
                ))
    return configs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top_k_recall(pred_sorted, true_sorted, k=1):
    if not true_sorted or not pred_sorted:
        return 0.0
    true_top = set(true_sorted[:k])
    pred_top = set(pred_sorted[:k])
    return len(true_top & pred_top) / len(true_top)


def kendall_tau(pred_sorted, true_sorted):
    common = sorted(set(pred_sorted) & set(true_sorted), key=lambda x: pred_sorted.index(x))
    if len(common) < 2:
        return 0.0
    concordant = discordant = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = common[i], common[j]
            if (pred_sorted.index(a) < pred_sorted.index(b)) == (true_sorted.index(a) < true_sorted.index(b)):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = _build_configs()
    top_k = int(os.environ.get("TRITON_COSTMODEL_TOP_K", "5"))
    pipeline_sel = os.environ.get("TRITON_BENCH_PIPELINE", "AB")

    print(f"Number of candidate configs: {len(configs)}")
    print(f"Costmodel top_k: {top_k}")
    print()

    # Matmul shapes: (M, K) × (K, N)
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # ---- warmup ----
    _default_cache = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(_default_cache):
        shutil.rmtree(_default_cache, ignore_errors=True)
    print("=== Compiler warmup (not timed) ===")
    _warm_c = torch.empty(M, N, device="npu", dtype=torch.float16)
    @triton.autotune(configs=[configs[0]], key=["M", "N", "K"])
    @triton.jit
    def _warmup(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0); pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            ak = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            bk = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            acc += tl.dot(ak, bk)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c = acc.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    _warmup[grid](a, b, _warm_c, M, N, K,
                   a.stride(0), a.stride(1), b.stride(0), b.stride(1), _warm_c.stride(0), _warm_c.stride(1))
    print()

    # ---- pipeline functions ----
    def _run_pipeline_a():
        print("=== Pipeline A: costmodel pre-filter ===")
        t0 = time.time()
        @triton.autotune(configs=configs, key=["M", "N", "K"],
                         prune_configs_by={"costmodel": {"top_k": top_k}})
        @triton.jit
        def _a(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_m = tl.program_id(0); pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                ak = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
                bk = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
                acc += tl.dot(ak, bk)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        c = torch.empty(M, N, device="npu", dtype=torch.float16)
        _a[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        raw = _a.configs_timings
        elapsed = time.time() - t0
        timings = {c: v[0] if isinstance(v, (tuple, list)) else v for c, v in raw.items()}
        preds = getattr(_a, "_costmodel_predictions", {})
        cm = getattr(_a, "_costmodel_timing", {})
        hw = getattr(_a, "_hw_bench_timing", {})
        print(f"  Pipeline A total: {elapsed:.2f}s")
        if cm:
            print(f"    ├─ TTIR compile:    {cm.get('ttir_elapsed_s', 0):.2f}s")
            print(f"    ├─ costmodel eval:  {cm.get('eval_elapsed_s', 0):.2f}s")
            print(f"    ├─ HW compile:      {hw.get('compile_s', 0):.2f}s ({len(timings)} configs)")
            print(f"    └─ HW kernel:       {hw.get('bench_s', 0):.2f}s")
        print(f"  best: {_a.best_config}")
        for c, t in sorted(timings.items(), key=lambda x: x[1]):
            print(f"    {t:.4f} ms  |  {c}")
        return elapsed, timings, preds, cm, _a.best_config, _a

    def _run_pipeline_b():
        print("=== Pipeline B: brute-force (HW bench all configs) ===")
        t0 = time.time()
        @triton.autotune(configs=configs, key=["M", "N", "K"])
        @triton.jit
        def _b(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_m = tl.program_id(0); pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                ak = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
                bk = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
                acc += tl.dot(ak, bk)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c = acc.to(tl.float16)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        c = torch.empty(M, N, device="npu", dtype=torch.float16)
        _b[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        raw = _b.configs_timings
        elapsed = time.time() - t0
        timings = {c: v[0] if isinstance(v, (tuple, list)) else v for c, v in raw.items()}
        hw = getattr(_b, "_hw_bench_timing", {})
        print(f"  Pipeline B total: {elapsed:.2f}s")
        print(f"    ├─ HW compile:      {hw.get('compile_s', 0):.2f}s ({len(timings)} configs)")
        print(f"    └─ HW kernel:       {hw.get('bench_s', 0):.2f}s")
        print(f"  best: {_b.best_config}")
        for c, t in sorted(timings.items(), key=lambda x: x[1]):
            print(f"    {t:.4f} ms  |  {c}")
        return elapsed, timings, _b.best_config, _b

    # ---- run ----
    a_elapsed = a_timings = a_predictions = a_costmodel_timing = a_best = a_tuner = None
    b_elapsed = b_timings = b_best = b_tuner = None

    if pipeline_sel in ("A", "AB", "BA"):
        a_elapsed, a_timings, a_predictions, a_costmodel_timing, a_best, a_tuner = _run_pipeline_a()
        print()

    if pipeline_sel in ("B", "AB", "BA"):
        b_elapsed, b_timings, b_best, b_tuner = _run_pipeline_b()
        print()

    if pipeline_sel not in ("AB", "BA") or a_elapsed is None or b_elapsed is None:
        return

    # ---- comparison ----
    print("=== Comparison ===")
    speedup = b_elapsed / a_elapsed if a_elapsed > 0 else float("inf")

    cm = a_costmodel_timing if a_costmodel_timing else {}
    a_hw = getattr(a_tuner, "_hw_bench_timing", {})
    b_hw = getattr(b_tuner, "_hw_bench_timing", {})

    print(f"\n{'='*60}")
    print(f"Time breakdown")
    print(f"{'='*60}")
    print(f"  {'Phase':<30} {'Pipeline A':>12} {'Pipeline B':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'costmodel (TTIR+eval)':<30} {cm.get('elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'  └─ TTIR compile':<30} {cm.get('ttir_elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'  └─ costmodel eval':<30} {cm.get('eval_elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'HW bench (of which)':<30}")
    print(f"  {'  └─ compile':<30} {a_hw.get('compile_s', 0):>11.2f}s {b_hw.get('compile_s', 0):>11.2f}s")
    print(f"  {'  └─ kernel run':<30} {a_hw.get('bench_s', 0):>11.2f}s {b_hw.get('bench_s', 0):>11.2f}s")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Total':<30} {a_elapsed:>11.2f}s {b_elapsed:>11.2f}s")
    print(f"\n  speedup: {speedup:.2f}x {'(A faster)' if speedup >= 1.0 else '(A slower)'}")
    print(f"  configs: A→{len(a_timings)} HW, B→{len(b_timings)} HW")

    if a_predictions and b_timings:
        pred_sorted = sorted(a_predictions, key=lambda c: a_predictions[c])
        true_sorted = sorted(b_timings, key=lambda c: b_timings[c])

        print(f"\n  Prediction accuracy (costmodel vs real):")
        for k in (1, 3, 5):
            if k <= top_k:
                print(f"    top-{k} recall:  {top_k_recall(pred_sorted, true_sorted, k=k):.2f}")
        print(f"    Kendall tau:   {kendall_tau(pred_sorted, true_sorted):.3f}")

        print(f"\n  {'Pred(us)':>10}  {'Real(us)':>10}  Config")
        for cfg in pred_sorted[:20]:
            pred = a_predictions.get(cfg, float("inf"))
            real = b_timings.get(cfg, float("inf")) * 1000
            print(f"  {pred:10.1f}  {real:10.1f}  {cfg}")

    results = {
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": speedup,
        "top_k": top_k,
    }
    with open("bench_costmodel_matmul_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to bench_costmodel_matmul_results.json")


if __name__ == "__main__":
    main()
