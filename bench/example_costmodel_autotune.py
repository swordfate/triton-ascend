#!/usr/bin/env python3
"""End-to-end example: costmodel-pruned autotune for vecadd.

This script demonstrates the costmodel pre-filter pipeline.  It defines a
simple vector-addition kernel, wraps it with ``@triton.autotune``, and
enables costmodel-based config pruning so that only the top-k predicted
configs are benchmarked on real hardware.

Usage::

    # Development – full logging to trace every step:
    TRITON_ENABLE_COSTMODEL_PRUNE=1 \\
    TRITON_COSTMODEL_TOP_K=5 \\
    TRITON_COSTMODEL_VERBOSE=1 \\
    TRITON_PRINT_AUTOTUNING=1 \\
        python bench/example_costmodel_autotune.py

    # Production – only timing output, no per-step logs:
    TRITON_ENABLE_COSTMODEL_PRUNE=1 \\
    TRITON_COSTMODEL_TOP_K=5 \\
        python bench/example_costmodel_autotune.py

Environment variables:
    TRITON_ENABLE_COSTMODEL_PRUNE:  set to ``1`` to enable costmodel pruning
    TRITON_COSTMODEL_TOP_K:        number of configs to keep after pruning (default 10)
    TRITON_COSTMODEL_VERBOSE:      set to ``1`` for per-step TTIR/costmodel logs
    TRITON_PRINT_AUTOTUNING:       set to ``1`` to see autotune progress
    TRITON_COSTMODEL_SAVE_PREDICTIONS: set to ``1`` to save predictions on the tuner

What to expect:
    1. The tuner generates candidate configs (BLOCK_SIZE + num_stages).
    2. Costmodel compiles TTIR for each config and predicts latency.
    3. Only the top-k configs are compiled to NPU binaries and benchmarked.
    4. The best config is selected and the kernel executes with it.
    5. If TRITON_PRINT_AUTOTUNING=1, the costmodel pruning statistics
       are printed alongside the final selection.
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# IMPORTANT: the ascend autotune patch must be active BEFORE the
# @triton.autotune decorator is evaluated at module-load time.  Importing
# the ascend runtime module triggers _patch_autotune() which replaces
# triton.autotune with the ascend version (AutoTilingTuner).
# ---------------------------------------------------------------------------
import triton.backends.ascend.runtime  # noqa: F401  — triggers _patch_autotune()


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128, "multibuffer": True}),
        triton.Config(kwargs={"BLOCK_SIZE": 256, "multibuffer": True}),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "multibuffer": True}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "multibuffer": True}),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "multibuffer": True}),
        triton.Config(kwargs={"BLOCK_SIZE": 128, "multibuffer": False}),
        triton.Config(kwargs={"BLOCK_SIZE": 256, "multibuffer": False}),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "multibuffer": False}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "multibuffer": False}),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "multibuffer": False}),
    ],
    key=["n_elements"],
    prune_configs_by={
        # Enable costmodel pruning with top_k.  This is equivalent to setting
        # TRITON_ENABLE_COSTMODEL_PRUNE=1 + TRITON_COSTMODEL_TOP_K=5.
        "costmodel": {"top_k": 5},
    },
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Print configuration
    enable = os.environ.get("TRITON_ENABLE_COSTMODEL_PRUNE", "0")
    top_k = os.environ.get("TRITON_COSTMODEL_TOP_K", "10")
    print(f"TRITON_ENABLE_COSTMODEL_PRUNE = {enable}")
    print(f"TRITON_COSTMODEL_TOP_K = {top_k}")
    print(f"Number of candidate configs = 15")
    print()

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    # Reference result via PyTorch
    output_torch = x + y

    # Triton kernel with autotune (costmodel-pruned if enabled)
    output_triton = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output_triton, size)

    # Verify correctness
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    print(f"\nMax difference (torch vs triton): {max_diff}")
    if max_diff < 1e-5:
        print("PASS: results match!")
    else:
        print("FAIL: results differ!")

    # If costmodel was enabled, show predictions
    tuner = add_kernel  # the tuner instance IS the decorated function
    if hasattr(tuner, "_costmodel_predictions") and tuner._costmodel_predictions:
        preds = tuner._costmodel_predictions
        print(f"\nCostmodel predictions ({len(preds)} configs):")
        for cfg, lat in sorted(preds.items(), key=lambda x: x[1]):
            print(f"  {lat:8.1f} us  |  {cfg}")


if __name__ == "__main__":
    main()
