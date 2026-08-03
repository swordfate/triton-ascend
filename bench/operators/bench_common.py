#!/usr/bin/env python3
"""Shared utilities for operator costmodel benchmarks."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.backends.ascend.runtime  # noqa: F401 — triggers _patch_autotune()

_cache_dirs: List[str] = []


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def setup_costmodel_env(enable: bool, top_k: float = 0.2) -> None:
    """Set (or clear) environment variables that control costmodel pruning.

    Each call creates a fresh ``TRITON_CACHE_DIR`` so that Pipeline A and
    Pipeline B compilation caches are isolated — without this, the same
    kernel function name would hit the disk cache from the previous run.

    Must be called *before* the kernel module is imported / reloaded.
    """
    if enable:
        os.environ["TRITON_ENABLE_COSTMODEL_PRUNE"] = "1"
        os.environ["TRITON_COSTMODEL_TOP_K"] = str(top_k)
        os.environ["TRITON_COSTMODEL_SAVE_PREDICTIONS"] = "1"
    else:
        for key in (
            "TRITON_ENABLE_COSTMODEL_PRUNE",
            "TRITON_COSTMODEL_TOP_K",
            "TRITON_COSTMODEL_SAVE_PREDICTIONS",
        ):
            os.environ.pop(key, None)

    # Fresh cache dir per pipeline — isolates compilation artifacts
    cache_dir = tempfile.mkdtemp(prefix="triton_bench_")
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    _cache_dirs.append(cache_dir)


def warmup_cache() -> None:
    """Compile a trivial kernel into the current TRITON_CACHE_DIR.

    The first compilation in a fresh cache dir pays a one-time
    toolchain (bishengir) startup cost and produces shared artefacts
    (npu_utils.so etc.) that subsequent compilations reuse.

    Must be called *after* ``setup_costmodel_env`` so the warmup
    artefacts land in the pipeline's private cache dir.
    """
    size = 64
    x = torch.randn(size, device="npu")
    y = torch.randn(size, device="npu")
    o = torch.empty_like(x)

    @triton.jit
    def _bench_warmup(x_ptr, y_ptr, o_ptr, n: tl.constexpr):
        off = tl.arange(0, n)
        tl.store(o_ptr + off, tl.load(x_ptr + off) + tl.load(y_ptr + off))

    _bench_warmup[(1,)](x, y, o, size)


def cleanup_cache_dirs() -> None:
    """Remove all temporary cache directories created during the benchmark."""
    for d in _cache_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    _cache_dirs.clear()


def reload_kernel_module(module: ModuleType) -> ModuleType:
    """Reload *module* so that ``@triton.autotune`` decorators are re-evaluated
    with the current environment variables.
    """
    return importlib.reload(module)


# ---------------------------------------------------------------------------
# Timings helpers
# ---------------------------------------------------------------------------

def extract_timings(tuner: Any) -> Dict[Any, float]:
    """Extract per-config timings (in seconds) from an AutoTilingTuner instance.

    Each value in ``configs_timings`` may be a scalar or a tuple (when the
    kernel was called with multiple argument sets).  We take the first element
    in the tuple case.
    """
    raw = getattr(tuner, "configs_timings", {})
    return {c: (v[0] if isinstance(v, (tuple, list)) else v) for c, v in raw.items()}


def extract_predictions(tuner: Any) -> Dict[Any, float]:
    """Extract costmodel predictions (latency in us) from a tuner instance."""
    return getattr(tuner, "_costmodel_predictions", {})


def extract_costmodel_timing(tuner: Any) -> Dict[str, float]:
    """Extract breakdown of costmodel time."""
    return getattr(tuner, "_costmodel_timing", {})


def extract_hw_timing(tuner: Any) -> Dict[str, float]:
    """Extract breakdown of hardware bench time."""
    return getattr(tuner, "_hw_bench_timing", {})


# ---------------------------------------------------------------------------
# Core runner (called from each bench script)
# ---------------------------------------------------------------------------

def find_active_tuner(module: ModuleType, candidates: List[str]) -> str:
    """Return the first candidate tuner whose ``run()`` was actually invoked.

    Some wrappers dispatch to different kernels based on heuristics
    (e.g. mojo's silu_fwd_impl selects from 3 variants, rmsnorm_infer_impl
    chooses between _infer_kernel and _infer_kernel_single).

    The Ascend autotuner sets ``self._was_called = True`` at the top of
    ``run()``, so even if all configs later fail HW compilation we can
    still tell which kernel was targeted.
    """
    for attr in candidates:
        tuner = getattr(module, attr, None)
        if tuner is not None and getattr(tuner, "_was_called", False):
            return attr
    # Fallback: try configs_timings (for kernels that use the standard
    # Triton autotuner which does not set _was_called).
    for attr in candidates:
        tuner = getattr(module, attr, None)
        if tuner is not None and getattr(tuner, "configs_timings", {}):
            return attr
    return candidates[0]


def run_autotune_pass(
    module: ModuleType,
    tuner_attr: str,
    wrapper_fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    tuner_fallbacks: Optional[List[str]] = None,
) -> Tuple[float, Dict[Any, float], Dict[Any, float], Any, Dict[str, float], Dict[str, float]]:
    """Run one autotune pass and return results.

    Parameters
    ----------
    module : ModuleType
        The kernel module (already imported with correct env vars).
    tuner_attr : str
        Primary attribute name of the tuner.  If the wrapper dispatches
        to a different kernel (e.g. mojo silu selects from 3 variants),
        pass ``tuner_fallbacks`` to auto-detect the active one.
    wrapper_fn : callable
        Function that triggers the autotune (e.g. ``module.gelu_fwd_impl``).
    args, kwargs :
        Positional and keyword arguments for *wrapper_fn*.

    Returns
    -------
    elapsed_s : float
    timings : dict  {config → seconds}
    predictions : dict  {config → microseconds} (empty if costmodel was off)
    best_config : triton.Config
    cm_timing : dict  costmodel breakdown (empty if off)
    hw_timing : dict  hardware bench breakdown
    """
    if kwargs is None:
        kwargs = {}

    t0 = time.time()
    wrapper_fn(*args, **kwargs)
    elapsed = time.time() - t0

    # Auto-detect which tuner was actually called.  Many wrappers dispatch
    # to a different kernel than tuner_attr based on input shapes.
    if tuner_fallbacks is None:
        # Discover all autotune-decorated kernels in the module automatically.
        import triton.runtime.autotuner as _autotuner_mod
        all_candidates = sorted(
            name for name, obj in vars(module).items()
            if isinstance(obj, _autotuner_mod.Autotuner)
        )
    else:
        all_candidates = [tuner_attr] + list(tuner_fallbacks)
    active = find_active_tuner(module, all_candidates)

    tuner = getattr(module, active)
    if active != tuner_attr and os.environ.get("TRITON_COSTMODEL_VERBOSE", "0") == "1":
        print(f"[bench] tuner auto-detected: {tuner_attr} → {active}")
    failed = getattr(tuner, "_compile_failed_configs", [])
    timings = extract_timings(tuner)
    predictions = extract_predictions(tuner)
    cm_timing = extract_costmodel_timing(tuner)
    hw_timing = extract_hw_timing(tuner)
    best = getattr(tuner, "best_config", None)

    if failed:
        print(f"  [!] {len(failed)} config(s) failed to compile on HW:")
        for cfg in failed:
            print(f"      - {cfg}")
        print()

    return elapsed, timings, predictions, best, cm_timing, hw_timing


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top_k_recall(pred_sorted: list, true_sorted: list, k: int = 1) -> float:
    """Fraction of the true top-k configs that appear in the predicted top-k."""
    if not true_sorted or not pred_sorted:
        return 0.0
    true_top = set(true_sorted[:k])
    pred_top = set(pred_sorted[:k])
    return len(true_top & pred_top) / len(true_top)


def kendall_tau(pred_sorted: list, true_sorted: list) -> float:
    """Kendall rank correlation between predicted and true orderings."""
    common = sorted(set(pred_sorted) & set(true_sorted), key=lambda x: pred_sorted.index(x))
    if len(common) < 2:
        return 0.0
    concordant = discordant = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = common[i], common[j]
            if (pred_sorted.index(a) < pred_sorted.index(b)) == (
                true_sorted.index(a) < true_sorted.index(b)
            ):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_results(
    label: str,
    elapsed: float,
    timings: Dict[Any, float],
    best: Any,
    predictions: Optional[Dict[Any, float]] = None,
    cm_timing: Optional[Dict[str, float]] = None,
    hw_timing: Optional[Dict[str, float]] = None,
) -> None:
    """Print a single pipeline's results."""
    print(f"=== Pipeline {label} ===")
    print(f"  Total elapsed: {elapsed:.2f}s")
    if cm_timing and cm_timing.get("elapsed_s"):
        print(f"    ├─ TTIR compile:   {cm_timing.get('ttir_elapsed_s', 0):.2f}s")
        print(f"    ├─ costmodel eval: {cm_timing.get('eval_elapsed_s', 0):.2f}s")
    if hw_timing:
        print(f"    ├─ HW compile:     {hw_timing.get('compile_s', 0):.2f}s ({len(timings)} configs)")
        print(f"    └─ HW kernel:      {hw_timing.get('bench_s', 0):.2f}s")
    print(f"  Best config: {best}")
    print(f"  Timings ({len(timings)} configs):")
    for c, t in sorted(timings.items(), key=lambda x: x[1])[:10]:
        extra = ""
        if predictions and c in predictions:
            extra = f"  [pred: {predictions[c]:.1f} us]"
        print(f"    {t*1000:8.3f} ms  |  {c}{extra}")
    if len(timings) > 10:
        print(f"    ... ({len(timings) - 10} more)")
    print()


def print_comparison(
    a_elapsed: float,
    a_timings: Dict[Any, float],
    a_predictions: Dict[Any, float],
    a_cm_timing: Dict[str, float],
    a_hw_timing: Dict[str, float],
    b_elapsed: float,
    b_timings: Dict[Any, float],
    b_hw_timing: Dict[str, float],
    top_k_val: int,
) -> None:
    """Print A-vs-B comparison table."""
    print("=" * 60)
    print("Comparison")
    print("=" * 60)

    speedup = b_elapsed / a_elapsed if a_elapsed > 0 else float("inf")
    print(f"  Pipeline A (costmodel): {a_elapsed:.2f}s  ({len(a_timings)} configs on HW)")
    print(f"  Pipeline B (brute):     {b_elapsed:.2f}s  ({len(b_timings)} configs on HW)")
    print(f"  Speedup: {speedup:.2f}x {'(A faster)' if speedup >= 1.0 else '(A slower)'}")
    print()

    print(f"  Time breakdown:")
    print(f"  {'Phase':<30} {'Pipeline A':>12} {'Pipeline B':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    cm = a_cm_timing
    print(f"  {'costmodel (TTIR+eval)':<30} {cm.get('elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'  └─ TTIR compile':<30} {cm.get('ttir_elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'  └─ costmodel eval':<30} {cm.get('eval_elapsed_s', 0):>11.2f}s {'—':>12}")
    print(f"  {'HW bench':<30}")
    print(f"  {'  └─ compile':<30} {a_hw_timing.get('compile_s', 0):>11.2f}s {b_hw_timing.get('compile_s', 0):>11.2f}s")
    print(f"  {'  └─ kernel run':<30} {a_hw_timing.get('bench_s', 0):>11.2f}s {b_hw_timing.get('bench_s', 0):>11.2f}s")

    if not a_predictions or not b_timings:
        print("\n  (no predictions to compare — costmodel may not have run)")
        return

    # Only compare configs present in both predictions and ground-truth.
    common_configs = [c for c in a_predictions if c in b_timings]
    if not common_configs:
        print("\n  (no overlapping configs between prediction and reality — "
              "likely HW compilation failures)")
        return

    pred_sorted = sorted(common_configs, key=lambda c: a_predictions[c])
    true_sorted = sorted(common_configs, key=lambda c: b_timings[c])

    print(f"\n  Prediction accuracy (costmodel vs real):")
    for k in (1, 3, 5):
        if k <= top_k_val:
            print(f"    top-{k} recall:  {top_k_recall(pred_sorted, true_sorted, k=k):.2f}")
    print(f"    Kendall tau:   {kendall_tau(pred_sorted, true_sorted):.3f}")

    print(f"\n  {'Pred(ms)':>12}  {'Real(ms)':>10}  Config")
    for cfg in pred_sorted:
        pred_ms = a_predictions[cfg] / 1000
        real_ms = b_timings[cfg] * 1000
        print(f"  {pred_ms:12.3f}  {real_ms:10.3f}  {cfg}")
