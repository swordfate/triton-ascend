#!/usr/bin/env python3
"""Board Event probe: shaped strided load/store across element dtype x path.

One rotate-loop kernel performs a single shaped strided memory operation per
iteration; the target is the Event slope in ns/iteration.

Paths
-----
  simd       compile_mode=simd        -> MTE structured wide / per-element gather
  simt_only  compile_mode=simt_only   -> SIMT_LDG / SIMT_STG + 128B DCache
  template   compile_mode=simd_simt_template
                                      -> call @triton_stride_load/store

The probe ASSERTS the lowering actually taken before it times anything:
  * template   must contain "triton_stride_load" / "triton_stride_store"
  * simd / simt_only must NOT contain it

Protocol (identical to the v2 template calibration, see README 5.3):
  * iters in {2000, 8000}; per point take the MINIMUM Event over reps launches
  * target = ordinary least squares slope of (iters, min_event_ns)
  * ALU busy burst before every measured launch to hold boost
  * ALU witness after every attempt (15% tolerance vs the startup reference)
  * npu-smi Aicore frequency after every attempt (>= 1500 MHz)
  * spread = (max-min)/min of the per-iteration Event values, must be <= 0.40
  * an attempt is valid only if witness AND frequency AND spread all pass
  * a case is valid if at least one attempt is valid; the delivered target is
    the minimum target over the valid attempts

stride == 1 is never measured here: this probe covers strided accesses only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from template_stride_common import (
    asm_text,
    busy_kernel,
    busy_opts,
    freq_check,
    make_ramp,
    measure_event_us,
    read_aicore_freq_mhz,
    save_asm,
    slope_ns_per_iter,
    template_opts,
)


@triton.jit(do_not_specialize=["iters"])
def strided_load_loop_kernel(
    x_ptr,
    out_ptr,
    iters,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    STEP: tl.constexpr,
    MASK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK) * STRIDE
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(iters):
        acc += tl.load(x_ptr + (i & MASK) * STEP + offs).to(tl.float32)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(acc, axis=0))


@triton.jit(do_not_specialize=["iters"])
def strided_store_loop_kernel(
    out_ptr,
    iters,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    STEP: tl.constexpr,
    MASK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK) * STRIDE
    v = (tl.arange(0, BLOCK) % 7).to(out_ptr.dtype.element_ty)
    for i in range(iters):
        tl.store(out_ptr + (i & MASK) * STEP + offs, v)


TORCH_DTYPES = {
    "int8": torch.int8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}

ELEMENT_BYTES = {
    "int8": 1, "float16": 2, "bfloat16": 2,
    "float32": 4, "int32": 4, "int64": 8,
}


def simd_opts(num_warps=1, grid=1):
    return {
        "num_warps": 1,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simd",
        "compile_on_910_95": True,
        "enable_auto_blockify": False,
    }


def simt_opts(num_warps=4, grid=1):
    return {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
        "compile_mode": "simt_only",
        "compile_on_910_95": True,
        "enable_auto_blockify": True,
    }


def opts_for(path_name, num_warps, grid=1):
    if path_name == "template":
        return template_opts(num_warps=num_warps, grid=grid)
    if path_name == "simd":
        return simd_opts(num_warps=1, grid=grid)
    if path_name == "simt_only":
        return simt_opts(num_warps=num_warps, grid=grid)
    raise ValueError(path_name)


def assert_lowering(compiled, path_name, symbol):
    text = asm_text(compiled)
    present = symbol in text
    if path_name == "template":
        if not present:
            raise RuntimeError(
                f"{symbol!r} missing from compiled asm; the template path was "
                "not taken (TTAdapter fell back to structured copy)")
    else:
        if present:
            raise RuntimeError(
                f"{symbol!r} unexpectedly present for path {path_name!r}")
    return present


def correctness_load(x, block, stride, step, mask, sink, opts, grid):
    ref = x[0: block * stride: stride].to(torch.float32).sum()
    strided_load_loop_kernel[(grid,)](
        x, sink, 1, BLOCK=block, STRIDE=stride, STEP=step, MASK=mask, **opts)
    torch.npu.synchronize()
    got = sink[0].to(torch.float32)
    tol = max(1e-3, abs(float(ref)) * 1e-4)
    return bool(abs(float(got) - float(ref)) <= tol)


def correctness_store(out, block, stride, step, mask, opts, grid, torch_dtype):
    out.zero_()
    strided_store_loop_kernel[(grid,)](
        out, 1, BLOCK=block, STRIDE=stride, STEP=step, MASK=mask, **opts)
    torch.npu.synchronize()
    expected = (torch.arange(block, device="npu") % 7).to(torch_dtype)
    got = out[0: block * stride: stride]
    return bool(torch.equal(got, expected))


def run_case(op, path_name, dtype_name, torch_dtype, block, stride, num_warps,
             buffers, step, mask, iters_list, reps, device,
             reference_witness_ns, witness_tol, min_freq_mhz, max_attempts,
             busy_iters, max_spread, asm_dir, grid=1):
    x, out, sink, busy_out = buffers
    opts = opts_for(path_name, num_warps, grid=grid)
    symbol = "triton_stride_load" if op == "load" else "triton_stride_store"
    kernel = strided_load_loop_kernel if op == "load" else strided_store_loop_kernel
    label = f"{op}_{path_name}_{dtype_name}_b{block}_s{stride}_w{num_warps}"

    # tl.sum(acc) is float32, so the load sink must be float32: storing it
    # into an int8 buffer truncates and makes the correctness check fail.
    if op == "load":
        args = (x, sink)
    else:
        args = (out,)
    compiled = kernel.warmup(
        *args, 16, BLOCK=block, STRIDE=stride, STEP=step, MASK=mask,
        grid=(grid,), **opts)
    contains = assert_lowering(compiled, path_name, symbol)
    asm_keys = save_asm(compiled, asm_dir, label) if asm_dir else []

    # warm the exact runtime shape, then verify numerics once
    if op == "load":
        kernel[(grid,)](x, sink, 16, BLOCK=block, STRIDE=stride, STEP=step,
                        MASK=mask, **opts)
        torch.npu.synchronize()
        correct = correctness_load(x, block, stride, step, mask, sink, opts, grid)
    else:
        kernel[(grid,)](out, 16, BLOCK=block, STRIDE=stride, STEP=step,
                        MASK=mask, **opts)
        torch.npu.synchronize()
        correct = correctness_store(out, block, stride, step, mask, opts, grid,
                                    torch_dtype)

    ramp = make_ramp(busy_out, busy_iters, grid=grid)
    attempts = []
    for attempt in range(max_attempts):
        event_ns_list = []
        raw_rows = []
        for iters in iters_list:
            evs = []
            for _ in range(reps):
                ramp()
                if op == "load":
                    evs.append(measure_event_us(
                        lambda it=iters: kernel[(grid,)](
                            x, sink, it, BLOCK=block, STRIDE=stride, STEP=step,
                            MASK=mask, **opts)))
                else:
                    evs.append(measure_event_us(
                        lambda it=iters: kernel[(grid,)](
                            out, it, BLOCK=block, STRIDE=stride, STEP=step,
                            MASK=mask, **opts)))
            min_ms = min(evs)
            event_ns_list.append(min_ms * 1e6)
            raw_rows.append({"iters": int(iters), "event_ns_min": min_ms * 1e6,
                             "event_ns_median": statistics.median(evs) * 1e6,
                             "event_ns_all": [v * 1e6 for v in evs]})
        target = slope_ns_per_iter(iters_list, event_ns_list)
        per_iter = [e / max(1, it) for e, it in zip(event_ns_list, iters_list)]
        spread = ((max(per_iter) - min(per_iter)) / min(per_iter)
                  if min(per_iter) > 0 else float("inf"))
        ramp()
        post_witness_ns = measure_event_us(
            lambda: busy_kernel[(grid,)](busy_out, busy_iters, BLOCK=1024,
                                         **busy_opts())) * 1e6
        witness_ok = reference_witness_ns > 0 and abs(
            post_witness_ns / reference_witness_ns - 1.0) <= witness_tol
        post_freq_mhz = read_aicore_freq_mhz(device)
        frequency_ok = (post_freq_mhz is not None
                        and post_freq_mhz >= min_freq_mhz)
        attempts.append({
            "attempt": attempt,
            "rows": raw_rows,
            "target_ns": target,
            "post_witness_ns": post_witness_ns,
            "post_freq_mhz": post_freq_mhz,
            "witness_ok": bool(witness_ok),
            "frequency_ok": bool(frequency_ok),
            "spread": spread,
            "stable_ok": bool(spread <= max_spread),
        })
    accepted = [a for a in attempts
                if a["witness_ok"] and a["stable_ok"] and a["frequency_ok"]]
    best = min(accepted, key=lambda a: a["target_ns"]) if accepted else None
    elem_bytes = ELEMENT_BYTES[dtype_name]
    return {
        "path": path_name,
        "op": op,
        "dtype": dtype_name,
        "element_bytes": elem_bytes,
        "compile_mode": opts["compile_mode"],
        "block": int(block),
        "stride": int(stride),
        "stride_bytes": elem_bytes * int(stride),
        "num_warps": int(num_warps),
        "grid": int(grid),
        "target_ns": float(best["target_ns"]) if best is not None else float(
            statistics.median(a["target_ns"] for a in attempts)),
        "target_kind": "rotate_loop_min_event_slope_ns_per_iter",
        "valid": bool(best is not None),
        "correctness_ok": bool(correct),
        "contains_template_call": bool(contains),
        "iters_list": [int(i) for i in iters_list],
        "reps": int(reps),
        "options": opts,
        "asm_index": asm_keys,
        "attempts": attempts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["load", "store"], required=True)
    ap.add_argument("--path", dest="path_name",
                    choices=["simd", "simt_only", "template"], required=True)
    ap.add_argument("--dtype", dest="dtype_name", required=True,
                    choices=sorted(TORCH_DTYPES))
    ap.add_argument("--blocks", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--strides", nargs="+", type=int, default=[2, 3, 4, 5, 8])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[4])
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--iters", nargs="+", type=int, default=[2000, 8000])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rotate-step", type=int, default=0,
                    help="rotation window step in elements (0 = auto: 64 MB byte budget)")
    ap.add_argument("--rotate-mask", type=int, default=0,
                    help="rotation window mask (0 = auto). Historical probes used "
                         "step=8192 mask=16383 (512 MB for f32).")
    ap.add_argument("--busy-iters", type=int, default=2_000_000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-freq-mhz", type=int, default=1500)
    ap.add_argument("--witness-tol", type=float, default=0.15)
    ap.add_argument("--max-spread", type=float, default=0.40)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--asm-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch_dtype = TORCH_DTYPES[args.dtype_name]
    elem_bytes = ELEMENT_BYTES[args.dtype_name]
    for stride in args.strides:
        if stride < 2:
            raise SystemExit("stride == 1 must not be measured (contiguous)")

    torch_npu.npu.set_device(args.device)

    max_tile = max(args.blocks) * max(args.strides)
    if args.rotate_step > 0 and args.rotate_mask > 0:
        # Explicit window: use it to reproduce the historical probes'
        # 512 MB rotation window (step 8192 x mask 16383) so the two
        # datasets are directly comparable.
        step, mask = int(args.rotate_step), int(args.rotate_mask)
    else:
        step = 1 << max(10, (max_tile - 1).bit_length())
        # Default: a 64 MB byte budget for the rotation window.
        budget_elems = max(1 << 20, (1 << 25) // elem_bytes)
        mask = max(15, budget_elems // step)
        mask = (1 << mask.bit_length()) - 1
    total = (mask + 1) * step + max_tile + 4096
    window_bytes = total * elem_bytes

    base = (torch.arange(4096, dtype=torch.int64) % 7).to(torch_dtype)
    x = base.repeat(total // 4096 + 1)[:total].to("npu")
    out = torch.zeros(total, dtype=torch_dtype, device="npu")
    # float32 sink for the load kernel: tl.sum(acc) is float32 and an int8
    # destination would truncate it (this broke the int8 correctness check).
    sink = torch.zeros(16, dtype=torch.float32, device="npu")
    busy_out = torch.empty(4096, dtype=torch.float32, device="npu")

    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, **busy_opts())
        torch.npu.synchronize()
    freq_mhz, query_seconds = freq_check(args.device, busy_out)
    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, **busy_opts())
        torch.npu.synchronize()
    reference_witness_ns = measure_event_us(
        lambda: busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024,
                                  **busy_opts())) * 1e6

    cases = []
    t0 = time.monotonic()
    for block in args.blocks:
        for stride in args.strides:
            for num_warps in args.num_warps:
                case = run_case(
                    args.op, args.path_name, args.dtype_name, torch_dtype,
                    block, stride, num_warps, (x, out, sink, busy_out), step, mask,
                    args.iters, args.reps, args.device, reference_witness_ns,
                    args.witness_tol, args.min_freq_mhz, args.max_attempts,
                    args.busy_iters, args.max_spread, args.asm_dir,
                    grid=args.grid)
                cases.append(case)
                print(f'{args.op} {args.path_name} {args.dtype_name} b{block} '
                      f's{stride} gapB={elem_bytes*stride} W{num_warps}: '
                      f'{case["target_ns"]:.3f} ns/iter valid={case["valid"]} '
                      f'correct={case["correctness_ok"]}', flush=True)
    elapsed = time.monotonic() - t0
    payload = {
        "kind": "dtype_stride_measure",
        "op": args.op,
        "path": args.path_name,
        "dtype": args.dtype_name,
        "element_bytes": elem_bytes,
        "compile_mode": opts_for(args.path_name, args.num_warps[0],
                                 args.grid)["compile_mode"],
        "cases": cases,
        "buffer": {"step": int(step), "mask": int(mask), "total": int(total),
                    "window_bytes": int(window_bytes),
                   "bytes": int(total * elem_bytes)},
        "frequency_check": {
            "inflight_freq_mhz": freq_mhz,
            "query_seconds": query_seconds,
            "reference_witness_ns": reference_witness_ns,
            "witness_tolerance": args.witness_tol,
        },
        "elapsed_seconds": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    valid = sum(1 for c in cases if c["valid"])
    correct = sum(1 for c in cases if c["correctness_ok"])
    print(f"WROTE {args.out} valid={valid}/{len(cases)} correct={correct} "
          f"freq={freq_mhz}MHz elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
