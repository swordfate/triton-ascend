#!/usr/bin/env python3
"""Shared helpers for triton_stride_load/store template probes.

The probe scripts in this directory measure the *template* lowering path:

    tt.load/store -> ascend.stride_load/store -> triton_stride_load/store

This is not the same path as ``compile_mode=simt_only`` (pure SIMT LSU) or
``compile_mode=simd`` (MTE structured copy).  The template is forced with
``compile_mode=simd_simt_template``.  On this branch the Python backend also
needs ``parallel_mode=mix_simd_simt`` + ``compile_on_910_95=True`` for the
``triton-to-linalg`` rewrite to run; without them the generated TTAdapter falls
back to ``memref.copy``.  Each probe asserts the emitted TTAdapter contains the
``triton_stride_*`` call before it enters the timed section.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def busy_kernel(out_ptr, iters, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(iters):
        acc += 1.0
    tl.store(out_ptr + tl.arange(0, BLOCK), acc)


def template_opts(num_warps: int = 1, grid: int = 1):
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd_simt_template",
        "parallel_mode": "mix_simd_simt",
        "compile_on_910_95": True,
        "enable_auto_blockify": False,
    }


def busy_opts():
    return {
        "num_warps": 1,
        "logical_program_count_hint": 1,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd",
        "compile_on_910_95": True,
        "enable_auto_blockify": False,
    }


def read_aicore_freq_mhz(device: int):
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", str(device)],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    match = re.search(r"Aicore curFreq\(MHZ\)\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def measure_event_us(launch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end)  # ms


def slope_ns_per_iter(iters_list, event_ns_list):
    xs = [float(i) for i in iters_list]
    ys = [float(v) for v in event_ns_list]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def asm_text(compiled):
    asm = getattr(compiled, "asm", {}) or {}
    chunks = []
    for value in asm.values():
        if isinstance(value, bytes):
            chunks.append(value.decode("latin1", errors="replace"))
        elif isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks)


def assert_template_call(compiled, symbol: str):
    """Assert the TTAdapter / source contains the expected template call."""
    text = asm_text(compiled)
    if symbol not in text:
        raise RuntimeError(
            f"template call {symbol!r} not found in compiled asm; "
            "TTAdapter did not take the strided template path"
        )
    return True


def save_asm(compiled, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    asm = getattr(compiled, "asm", {}) or {}
    for key, value in asm.items():
        path = out_dir / f"{label}.{key}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8", errors="replace")
        else:
            continue
        keys.append({"key": key, "path": path.name, "bytes": len(value)})
    metadata = getattr(compiled, "metadata", None)
    if metadata is not None:
        try:
            (out_dir / f"{label}.metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        except TypeError:
            pass
    return keys


def freq_check(device, busy_out, compile_it=150_000_000):
    busy_kernel[(1,)](
        busy_out,
        compile_it,
        BLOCK=1024,
        **busy_opts(),
    )
    t0 = time.monotonic()
    freq = read_aicore_freq_mhz(device)
    query_seconds = time.monotonic() - t0
    torch.npu.synchronize()
    return freq, query_seconds


def make_ramp(busy_out, busy_iters, grid=1):
    def ramp():
        busy_kernel[(grid,)](
            busy_out,
            busy_iters,
            BLOCK=1024,
            **busy_opts(),
        )
        torch.npu.synchronize()

    return ramp
