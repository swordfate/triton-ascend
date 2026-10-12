#!/usr/bin/env python3
"""Board Event probe for rank-2 strided tiles.

Address set of one tile:

    addr(r, c) = r * OS + c * IS      r in [0,R), c in [0,C)

Named forms (see stride2d_path_check.py for the same list):

    contig             IS == 1, OS == C        -- fully contiguous, path check only
    rowgap             IS == 1, OS == C + pad  -- holes in the OUTER axis only
    inner_sK_tight     IS == K, OS == C * K    -- holes in the inner axis
    inner_sK_gap       IS == K, OS == C * K + pad -- holes in both axes

Protocol is identical to dtype_stride_measure_probe.py: rotate loop, min Event
per iteration point, ALU ramp, ALU witness (15%), npu-smi Aircore frequency
(>= 1500 MHz), spread <= 0.40, minimum target over valid attempts.

"contig" is measured only when --include-contig is passed, so that the
delivered dataset never mixes the contiguous anchor into a strided fit.
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

TORCH_DTYPES = {"int8": torch.int8, "float16": torch.float16,
                "bfloat16": torch.bfloat16, "float32": torch.float32,
                "int32": torch.int32, "int64": torch.int64}
ELEMENT_BYTES = {"int8": 1, "float16": 2, "bfloat16": 2,
                 "float32": 4, "int32": 4, "int64": 8}


@triton.jit(do_not_specialize=["iters"])
def stride2d_load_loop_kernel(x_ptr, out_ptr, iters, R: tl.constexpr,
                              C: tl.constexpr, OS: tl.constexpr,
                              IS: tl.constexpr, STEP: tl.constexpr,
                              MASK: tl.constexpr):
    offs = tl.arange(0, R)[:, None] * OS + tl.arange(0, C)[None, :] * IS
    acc = tl.zeros((R, C), dtype=tl.float32)
    for i in range(iters):
        acc += tl.load(x_ptr + (i & MASK) * STEP + offs).to(tl.float32)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(acc))


@triton.jit(do_not_specialize=["iters"])
def stride2d_store_loop_kernel(out_ptr, iters, R: tl.constexpr,
                               C: tl.constexpr, OS: tl.constexpr,
                               IS: tl.constexpr, STEP: tl.constexpr,
                               MASK: tl.constexpr):
    offs = tl.arange(0, R)[:, None] * OS + tl.arange(0, C)[None, :] * IS
    v = ((tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]) % 7)
    v = v.to(out_ptr.dtype.element_ty)
    for i in range(iters):
        tl.store(out_ptr + (i & MASK) * STEP + offs, v)


def opts_for(path_name, num_warps, grid=1):
    if path_name == "template":
        return template_opts(num_warps=num_warps, grid=grid)
    if path_name == "simd":
        return {"num_warps": 1, "logical_program_count_hint": grid,
                "physical_vector_core_count_hint": 1, "superblock_factor": 1,
                "auto_simt_scope_mode": "off", "compile_mode": "simd",
                "compile_on_910_95": True, "enable_auto_blockify": False}
    if path_name == "simt_only":
        return {"num_warps": int(num_warps), "logical_program_count_hint": grid,
                "physical_vector_core_count_hint": 1, "superblock_factor": 1,
                "auto_simt_scope_mode": "off", "compile_mode": "simt_only",
                "compile_on_910_95": True, "enable_auto_blockify": True}
    raise ValueError(path_name)


def assert_lowering(compiled, path_name, symbol):
    text = asm_text(compiled)
    present = symbol in text
    if path_name == "template" and not present:
        raise RuntimeError(f"{symbol!r} missing; template path not taken")
    if path_name != "template" and present:
        raise RuntimeError(f"{symbol!r} unexpectedly present for {path_name}")
    return present


def classify(rows, cols, outer_stride, inner_stride):
    rows, cols = int(rows), int(cols)
    outer_stride, inner_stride = int(outer_stride), int(inner_stride)
    if inner_stride == 1 and outer_stride == cols:
        return "contiguous"
    inner = inner_stride >= 2
    outer = outer_stride > (cols - 1) * inner_stride + 1
    if rows == 1:
        return "singleton_row_strided" if inner else "singleton_row_outer"
    if inner and outer:
        return "both_strided"
    if inner:
        return "inner_strided_only"
    return "outer_strided_only"


def build_forms(shapes, inner_strides, outer_pad, row_strides=None):
    """rank-2 forms.

    Two parameterisations:

    * legacy (row_strides is None): both_sK_tight has OS = cols*K and
      both_sK_gap has OS = cols*K + pad.
    * row-stride (row_strides given): the access is a subsample of a dense
      (rows*rs) x (cols*cs) matrix -- take rows 0, rs, 2rs, ... and columns
      0, cs, 2cs, ...  The linear address of (i, j) in that row-major matrix
      is i*(cols*cs) + j, so OS = rs * cols * cs.  This is the
      parameterisation the calibration now uses; rs IS the row stride, so
      there is no separate "gap" concept.
    """
    out = []
    for shape in shapes:
        rows, cols = (int(v) for v in shape.split("x"))
        if row_strides:
            for rs in row_strides:
                for k in inner_strides:
                    out.append((f"rs{rs}_cs{k}", rows, cols,
                                int(rs) * cols * int(k), k))
            continue
        out.append(("contig", rows, cols, cols, 1))
        if rows > 1:
            out.append(("rowgap", rows, cols, cols + outer_pad, 1))
        for k in inner_strides:
            out.append((f"both_s{k}_tight", rows, cols, cols * k, k))
            out.append((f"both_s{k}_gap", rows, cols, cols * k + outer_pad, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["load", "store"], required=True)
    ap.add_argument("--path", dest="path_name",
                    choices=["simd", "simt_only", "template"], required=True)
    ap.add_argument("--dtype", dest="dtype_name", default="float32",
                    choices=sorted(TORCH_DTYPES))
    ap.add_argument("--shapes", nargs="+", default=["1x64", "8x16", "16x32"])
    ap.add_argument("--inner-strides", nargs="+", type=int, default=[2, 3, 5])
    ap.add_argument("--outer-pad", type=int, default=7)
    ap.add_argument("--row-strides", nargs="+", type=int, default=None,
                    help="row stride rs; enables OS = rs*cols*cs and the "
                         "rs{rs}_cs{cs} form family (no tight/gap)")
    ap.add_argument("--forms", nargs="+", default=None,
                    help="restrict to these named forms")
    ap.add_argument("--include-contig", action="store_true",
                    help="also measure the fully contiguous anchor (never fitted)")
    ap.add_argument("--num-warps", nargs="+", type=int, default=[32])
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
    torch_npu.npu.set_device(args.device)

    all_forms = build_forms(args.shapes, args.inner_strides, args.outer_pad,
                            args.row_strides)
    if args.forms:
        all_forms = [f for f in all_forms if f[0] in set(args.forms)]
    if not args.include_contig:
        all_forms = [f for f in all_forms if f[0] != "contig"]
    if not all_forms:
        raise SystemExit("no forms selected")
    max_addr = max((R - 1) * OS + (C - 1) * IS for _, R, C, OS, IS in all_forms)

    if args.rotate_step > 0 and args.rotate_mask > 0:
        # Explicit window: reproduce the historical probes' 512 MB rotation
        # window (step 8192 x mask 16383) so the datasets are comparable.
        step, mask = int(args.rotate_step), int(args.rotate_mask)
    else:
        step = 1 << max(10, (max_addr + 1 - 1).bit_length())
        # Default: a 64 MB byte budget for the rotation window.
        budget_elems = max(1 << 20, (1 << 25) // elem_bytes)
        mask = max(15, budget_elems // step)
        mask = (1 << mask.bit_length()) - 1
    total = (mask + 1) * step + max_addr + 4096
    window_bytes = total * elem_bytes

    base = (torch.arange(4096, dtype=torch.int64) % 7).to(torch_dtype)
    x = base.repeat(total // 4096 + 1)[:total].to("npu")
    out = torch.zeros(total, dtype=torch_dtype, device="npu")
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
    for name, R, C, OS, IS in all_forms:
        for num_warps in args.num_warps:
            addr = ((torch.arange(R, device="npu")[:, None] * OS
                     + torch.arange(C, device="npu")[None, :] * IS).reshape(-1))
            contiguous = bool(IS == 1 and OS == C)
            axis_class = classify(R, C, OS, IS)
            opts = opts_for(args.path_name, num_warps, grid=args.grid)
            symbol = "triton_stride_load" if args.op == "load" else "triton_stride_store"
            label = f"2d_{args.op}_{args.path_name}_{args.dtype_name}_{name}_{R}x{C}_W{num_warps}"
            try:
                if args.op == "load":
                    compiled = stride2d_load_loop_kernel.warmup(
                        x, sink, 16, R=R, C=C, OS=OS, IS=IS, STEP=step,
                        MASK=mask, grid=(args.grid,), **opts)
                else:
                    compiled = stride2d_store_loop_kernel.warmup(
                        out, 16, R=R, C=C, OS=OS, IS=IS, STEP=step,
                        MASK=mask, grid=(args.grid,), **opts)
                contains = assert_lowering(compiled, args.path_name, symbol)
            except Exception as exc:
                print(f"COMPILE/ASSERT FAIL {label}: {type(exc).__name__}: {exc}",
                      flush=True)
                cases.append({"form": name, "R": R, "C": C, "OS": OS, "IS": IS,
                              "axis_class": classify(R, C, OS, IS),
                              "contiguous": contiguous, "num_warps": num_warps,
                              "path": args.path_name, "op": args.op,
                              "dtype": args.dtype_name,
                              "element_bytes": elem_bytes,
                              "valid": False, "correctness_ok": False,
                              "error": f"{type(exc).__name__}: {exc}"[:400]})
                continue
            if args.asm_dir:
                save_asm(compiled, args.asm_dir, label)

            if args.op == "load":
                # The kernel stores tl.sum(acc), which is FLOAT32.  Writing it
                # into 'out' (the ELEMENT dtype) overflows int8 as soon as the
                # tile holds ~43+ elements (values are 0..6, so the sum reaches
                # 127 around there) and every such case was flagged as a wrong
                # result.  Use the float32 'sink' exactly like the 1-D probe.
                stride2d_load_loop_kernel[(args.grid,)](
                    x, sink, 16, R=R, C=C, OS=OS, IS=IS, STEP=step, MASK=mask, **opts)
                torch.npu.synchronize()
                ref = x[addr].to(torch.float32).sum()
                stride2d_load_loop_kernel[(args.grid,)](
                    x, sink, 1, R=R, C=C, OS=OS, IS=IS, STEP=step, MASK=mask, **opts)
                torch.npu.synchronize()
                got = sink[0].to(torch.float32)
                correct = bool(abs(float(got) - float(ref))
                               <= max(1e-3, abs(float(ref)) * 1e-4))
            else:
                stride2d_store_loop_kernel[(args.grid,)](
                    out, 16, R=R, C=C, OS=OS, IS=IS, STEP=step, MASK=mask, **opts)
                torch.npu.synchronize()
                out.zero_()
                stride2d_store_loop_kernel[(args.grid,)](
                    out, 1, R=R, C=C, OS=OS, IS=IS, STEP=step, MASK=mask, **opts)
                torch.npu.synchronize()
                expected = (torch.arange(R * C, device="npu") % 7).to(torch_dtype)
                correct = bool(torch.equal(out[addr], expected))

            ramp = make_ramp(busy_out, args.busy_iters, grid=args.grid)
            attempts = []
            for attempt in range(args.max_attempts):
                event_ns_list = []
                raw_rows = []
                for iters in args.iters:
                    evs = []
                    for _ in range(args.reps):
                        ramp()
                        if args.op == "load":
                            evs.append(measure_event_us(
                                lambda it=iters: stride2d_load_loop_kernel[(args.grid,)](
                                    x, out, it, R=R, C=C, OS=OS, IS=IS, STEP=step,
                                    MASK=mask, **opts)))
                        else:
                            evs.append(measure_event_us(
                                lambda it=iters: stride2d_store_loop_kernel[(args.grid,)](
                                    out, it, R=R, C=C, OS=OS, IS=IS, STEP=step,
                                    MASK=mask, **opts)))
                    min_ms = min(evs)
                    event_ns_list.append(min_ms * 1e6)
                    raw_rows.append({"iters": int(iters),
                                     "event_ns_min": min_ms * 1e6,
                                     "event_ns_median": statistics.median(evs) * 1e6,
                                     "event_ns_all": [v * 1e6 for v in evs]})
                target = slope_ns_per_iter(args.iters, event_ns_list)
                per_iter = [e / max(1, it)
                            for e, it in zip(event_ns_list, args.iters)]
                spread = ((max(per_iter) - min(per_iter)) / min(per_iter)
                          if min(per_iter) > 0 else float("inf"))
                ramp()
                post_witness_ns = measure_event_us(
                    lambda: busy_kernel[(args.grid,)](busy_out, args.busy_iters,
                                                      BLOCK=1024, **busy_opts())) * 1e6
                witness_ok = reference_witness_ns > 0 and abs(
                    post_witness_ns / reference_witness_ns - 1.0) <= args.witness_tol
                post_freq_mhz = read_aicore_freq_mhz(args.device)
                frequency_ok = (post_freq_mhz is not None
                                and post_freq_mhz >= args.min_freq_mhz)
                attempts.append({
                    "attempt": attempt, "rows": raw_rows, "target_ns": target,
                    "post_witness_ns": post_witness_ns,
                    "post_freq_mhz": post_freq_mhz,
                    "witness_ok": bool(witness_ok),
                    "frequency_ok": bool(frequency_ok),
                    "spread": spread, "stable_ok": bool(spread <= args.max_spread)})
            accepted = [a for a in attempts
                        if a["witness_ok"] and a["stable_ok"] and a["frequency_ok"]]
            best = min(accepted, key=lambda a: a["target_ns"]) if accepted else None
            case = {
                "path": args.path_name, "op": args.op, "dtype": args.dtype_name,
                "element_bytes": elem_bytes, "form": name, "R": R, "C": C,
                "OS": OS, "IS": IS, "axis_class": axis_class, "contiguous": contiguous,
                "elements": R * C, "num_warps": int(num_warps),
                "grid": int(args.grid),
                "max_addr": int(max_addr),
                "target_ns": float(best["target_ns"]) if best is not None else float(
                    statistics.median(a["target_ns"] for a in attempts)),
                "target_kind": "rotate_loop_min_event_slope_ns_per_iter",
                "valid": bool(best is not None), "correctness_ok": bool(correct),
                "contains_template_call": bool(contains),
                "iters_list": [int(i) for i in args.iters], "reps": int(args.reps),
                "options": opts, "attempts": attempts,
            }
            cases.append(case)
            print(f'{args.op} {args.path_name} {args.dtype_name} {name} '
                  f'{R}x{C} OS={OS} IS={IS} W{num_warps}: '
                  f'{case["target_ns"]:.3f} ns/iter valid={case["valid"]} '
                  f'correct={case["correctness_ok"]}', flush=True)
    elapsed = time.monotonic() - t0
    payload = {
        "kind": "stride2d_measure", "op": args.op, "path": args.path_name,
        "dtype": args.dtype_name, "element_bytes": elem_bytes,
        "shapes": args.shapes, "inner_strides": args.inner_strides,
        "row_strides": args.row_strides,
        "outer_pad": args.outer_pad, "include_contig": bool(args.include_contig),
        "cases": cases,
        "buffer": {"step": int(step), "mask": int(mask), "total": int(total),
                    "window_bytes": int(window_bytes)},
        "frequency_check": {"inflight_freq_mhz": freq_mhz,
                            "query_seconds": query_seconds,
                            "reference_witness_ns": reference_witness_ns,
                            "witness_tolerance": args.witness_tol},
        "elapsed_seconds": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    valid = sum(1 for c in cases if c.get("valid"))
    correct = sum(1 for c in cases if c.get("correctness_ok"))
    print(f"WROTE {args.out} valid={valid}/{len(cases)} correct={correct} "
          f"elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
