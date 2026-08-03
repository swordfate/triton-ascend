#!/usr/bin/env python3
"""Dump SIMD vs SIMT TTIR for a gather kernel to compare IR structures.

Usage:
    python bench/debug_ttir_simd_vs_simt.py

This generates two TTIR files for the same gather kernel:
  - /tmp/ttir_simd.mlir   (compile_mode="simd")
  - /tmp/ttir_simt.mlir   (compile_mode="simt_only")

Diff them to see how the compiler treats structured vs unstructured
(indirect) memory access differently under each mode.
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

# Must import before @triton.jit to patch autotune path
import triton.backends.ascend.runtime  # noqa: F401


# ---------------------------------------------------------------------------
# A simple gather kernel — the key is the indirect load at the end
# ---------------------------------------------------------------------------

@triton.jit
def gather_kernel(
    data_ptr,        # [N]  source array
    indices_ptr,     # [M]  index array, values in [0, N)
    output_ptr,      # [M]  result
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # ── structured (continuous) load of indices ──
    idx = tl.load(indices_ptr + offs, mask=mask)

    # ── unstructured (indirect) load of data ──
    #    idx is NOT a range — it's dynamically loaded values.
    #    SIMD: compiler expands this into a scalar loop
    #    SIMT: compiler emits ascend.indirect_load
    val = tl.load(data_ptr + idx, mask=mask)

    tl.store(output_ptr + offs, val, mask=mask)


# ---------------------------------------------------------------------------
# Also a "structured-only" kernel for comparison —
# same shape, but all loads are continuous
# ---------------------------------------------------------------------------

@triton.jit
def vecadd_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)


# ---------------------------------------------------------------------------
# Helper: generate TTIR for a given compile_mode
# ---------------------------------------------------------------------------

def dump_ttir(jit_fn, device, config_kwargs, bound_args, label, out_path):
    """Call generate_ttir_for_costmodel and write TTIR to *out_path*."""
    from triton.backends.ascend.compiler import generate_ttir_for_costmodel

    ttir = generate_ttir_for_costmodel(jit_fn, config_kwargs, bound_args, device)
    if ttir is None:
        print(f"  FAILED to generate TTIR for {label}")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ttir)
    lines = ttir.count("\n")
    print(f"  {label}: {lines} lines → {out_path}")
    return ttir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = 0  # adjust if needed
    N = 1024 * 16
    M = 1024 * 4
    BLOCK = 128

    # ---- prepare gather inputs ----
    data = torch.randn(N, device="npu", dtype=torch.float32)
    indices = torch.randint(0, N, (M,), device="npu", dtype=torch.int64)
    output = torch.empty(M, device="npu", dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK"]),)

    # ---- prepare vecadd inputs ----
    x = torch.randn(N, device="npu", dtype=torch.float32)
    y = torch.randn(N, device="npu", dtype=torch.float32)
    o = torch.empty(N, device="npu", dtype=torch.float32)
    va_grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

    print("=" * 70)
    print("1. gather kernel — indirect load")
    print("=" * 70)

    # Warm up: call once to populate device_caches (needed by
    # generate_ttir_for_costmodel)
    gather_kernel[grid](data, indices, output, N, M, BLOCK)

    # Build bound_args (all function params) and config_kwargs (constexpr
    # values + compile mode option).
    gather_bound = {"data_ptr": data, "indices_ptr": indices,
                    "output_ptr": output, "N": N, "M": M, "BLOCK": BLOCK}
    gather_constexpr = {"N": N, "M": M, "BLOCK": BLOCK}

    # SIMD path
    simd_cfg = dict(gather_constexpr, compile_mode="simd")
    dump_ttir(gather_kernel, device, simd_cfg, gather_bound,
              "gather SIMD", "/tmp/ttir_gather_simd.mlir")

    # SIMT path
    simt_cfg = dict(gather_constexpr, compile_mode="simt_only")
    dump_ttir(gather_kernel, device, simt_cfg, gather_bound,
              "gather SIMT", "/tmp/ttir_gather_simt.mlir")

    # Mixed (default) path
    mix_cfg = dict(gather_constexpr, compile_mode="unstructured_in_simt")
    dump_ttir(gather_kernel, device, mix_cfg, gather_bound,
              "gather MIXED", "/tmp/ttir_gather_mixed.mlir")

    print()
    print("=" * 70)
    print("2. vecadd kernel — all structured (no indirect loads)")
    print("=" * 70)

    vecadd_kernel[va_grid](x, y, o, N, BLOCK)

    va_bound = {"x_ptr": x, "y_ptr": y, "o_ptr": o, "N": N, "BLOCK": BLOCK}
    va_constexpr = {"N": N, "BLOCK": BLOCK}

    dump_ttir(vecadd_kernel, device, dict(va_constexpr, compile_mode="simd"),
              va_bound, "vecadd SIMD", "/tmp/ttir_vecadd_simd.mlir")

    dump_ttir(vecadd_kernel, device, dict(va_constexpr, compile_mode="simt_only"),
              va_bound, "vecadd SIMT", "/tmp/ttir_vecadd_simt.mlir")

    print()
    print("=" * 70)
    print("Next steps:")
    print("  diff /tmp/ttir_gather_simd.mlir /tmp/ttir_gather_simt.mlir")
    print("  diff /tmp/ttir_vecadd_simd.mlir /tmp/ttir_vecadd_simt.mlir")
    print("=" * 70)


if __name__ == "__main__":
    main()
