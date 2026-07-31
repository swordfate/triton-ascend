#!/usr/bin/env python3
"""Benchmark: costmodel-preselected autotune vs brute-force hardware autotune.

Pipeline A (costmodel):
    configs -> costmodel_bench() -> top-k prediction -> HW bench top-k -> best

Pipeline B (brute-force):
    configs -> HW bench all configs -> best

Usage::

    python bench/bench_costmodel_vecadd.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

# Set before @triton.autotune decorators are evaluated (module-level)
os.environ.setdefault("TRITON_COSTMODEL_SAVE_PREDICTIONS", "1")
os.environ.setdefault("TRITON_COSTMODEL_REUSE_TTIR", "0")

import torch
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # triggers _patch_autotune()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def _build_configs() -> List:
    sizes = [128, 256, 512, 1024, 2048]
    configs = []
    for bs in sizes:
        for mb in (True, False):
            configs.append(triton.Config(
                kwargs={"BLOCK_SIZE": bs, "multibuffer": mb},
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
    order = os.environ.get("TRITON_BENCH_ORDER", "AB")
    print(f"Number of candidate configs: {len(configs)}")
    print(f"Costmodel top_k: {top_k}")
    print(f"Pipeline: {pipeline_sel}")
    print()

    size = 98432
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    # Clear stale cache so each benchmark run starts from scratch, then
    # warm one bishengir-compile invocation to load the toolchain.
    import shutil as _shutil
    _default_cache = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(_default_cache):
        _shutil.rmtree(_default_cache, ignore_errors=True)
    print("=== Compiler warmup (not timed) ===")
    _warm = torch.empty_like(x)
    @triton.autotune(configs=[configs[0]], key=["n_elements"])
    @triton.jit
    def _warmup(x_ptr, y_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0); bs = pid * BLOCK_SIZE
        off = bs + tl.arange(0, BLOCK_SIZE); m = off < n
        tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)
    _warmup[grid](x, y, _warm, size)
    print()

    # ── define pipeline functions ────────────────────────────────────
    def _run_pipeline_a():
        print("=== Pipeline A: costmodel pre-filter ===")
        t0 = time.time()
        @triton.autotune(configs=configs, key=["n_elements"],
                         prune_configs_by={"costmodel": {"top_k": top_k}})
        @triton.jit
        def _a(x_ptr, y_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0); bs = pid * BLOCK_SIZE
            off = bs + tl.arange(0, BLOCK_SIZE); m = off < n
            tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)
        o = torch.empty_like(x)
        _a[grid](x, y, o, size)
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
        @triton.autotune(configs=configs, key=["n_elements"])
        @triton.jit
        def _b(x_ptr, y_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0); bs = pid * BLOCK_SIZE
            off = bs + tl.arange(0, BLOCK_SIZE); m = off < n
            tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)
        o = torch.empty_like(x)
        _b[grid](x, y, o, size)
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

    # ── run selected pipeline(s) ──────────────────────────────────────
    a_elapsed = a_timings = a_predictions = a_costmodel_timing = a_best = a_tuner = None
    b_elapsed = b_timings = b_best = b_tuner = None

    if pipeline_sel in ("A", "B", "AB", "BA"):
        runners = [("A", _run_pipeline_a), ("B", _run_pipeline_b)]
        if order == "BA":
            runners = list(reversed(runners))

        for label, pipeline_fn in runners:
            if pipeline_sel not in ("AB", "BA") and label != pipeline_sel:
                print(f"  (skipping Pipeline {label})")
                continue
            print(f"  Running Pipeline {label}...")
            if label == "A":
                a_elapsed, a_timings, a_predictions, a_costmodel_timing, a_best, a_tuner = pipeline_fn()
            else:
                b_elapsed, b_timings, b_best, b_tuner = pipeline_fn()
            print()

    # ==================================================================
    # Comparison (skip if only one pipeline selected)
    # ==================================================================
    if pipeline_sel not in ("AB", "BA") or a_elapsed is None or b_elapsed is None:
        return
    print("=== Comparison ===")
    speedup = b_elapsed / a_elapsed if a_elapsed > 0 else float("inf")
    # ── parallelism info ──────────────────────────────────────────────
    from triton.backends.ascend.runtime.costmodel_runtime import get_costmodel_jobs
    import psutil
    cm_workers = get_costmodel_jobs(len(configs))
    hw_cores = psutil.cpu_count(logical=False)
    hw_a_workers = min(hw_cores * 3 // 4, top_k)
    hw_b_workers = min(hw_cores * 3 // 4, len(configs))

    print(f"\n{'='*60}")
    print(f"Time breakdown (cold start)")
    print(f"{'='*60}")
    cm = a_costmodel_timing if a_costmodel_timing else {}
    a_hw = getattr(a_tuner, "_hw_bench_timing", {})
    b_hw = getattr(b_tuner, "_hw_bench_timing", {})
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
    print(f"\n  Parallelism:")
    print(f"    Pipeline A costmodel: {cm_workers} workers (os.cpu_count, capped by configs)")
    print(f"    Pipeline A HW bench:  {hw_a_workers} workers (physical_cores×¾, capped by top_k)")
    print(f"    Pipeline B HW bench:  {hw_b_workers} workers (physical_cores×¾, capped by configs)")
    print(f"\n  speedup: {speedup:.2f}x {'(A faster)' if speedup >= 1.0 else '(A slower)'}")
    print(f"  configs: A→{len(a_timings)} HW, B→{len(b_timings)} HW")
    print(f"  TTIR reuse: {os.environ.get('TRITON_COSTMODEL_REUSE_TTIR', '1')}")

    # Costmodel prediction accuracy
    if a_predictions and b_timings:
        pred_sorted = sorted(a_predictions, key=lambda c: a_predictions[c])
        true_sorted = sorted(b_timings, key=lambda c: b_timings[c])

        print(f"  top-1 recall:  {top_k_recall(pred_sorted, true_sorted, k=1):.2f}")
        print(f"  top-3 recall:  {top_k_recall(pred_sorted, true_sorted, k=3):.2f}")
        print(f"  top-5 recall:  {top_k_recall(pred_sorted, true_sorted, k=5):.2f}")
        print(f"  Kendall tau:   {kendall_tau(pred_sorted, true_sorted):.3f}")

        print(f"\n  {'Pred(us)':>10}  {'Real(us)':>10}  Config")
        for cfg in pred_sorted:
            pred = a_predictions.get(cfg, float("inf"))
            real = b_timings.get(cfg, float("inf")) * 1000  # ms → us
            print(f"  {pred:10.3f}  {real:10.1f}  {cfg}")

    # Save results
    results = {
        "pipeline_a": {"elapsed_s": a_elapsed, "best": str(a_best)},
        "pipeline_b": {"elapsed_s": b_elapsed, "best": str(b_best)},
        "speedup": speedup,
    }
    with open("bench_costmodel_vecadd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to bench_costmodel_vecadd_results.json")


if __name__ == "__main__":
    main()
