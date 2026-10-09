#!/usr/bin/env python3
"""Board probe for the ``triton_stride_load`` template path (1D rank1).

Measures the Event slope of a rotate-loop kernel whose only memory operation is
one non-power-of-two static strided ``tl.load`` per iteration.  The kernel is
compiled with ``compile_mode=simd_simt_template`` and the probe asserts the
TTAdapter contains ``call @triton_stride_load`` before timing.
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
    assert_template_call,
    busy_kernel,
    busy_opts,
    freq_check,
    make_ramp,
    measure_event_us,
    save_asm,
    slope_ns_per_iter,
    template_opts,
)


@triton.jit(do_not_specialize=["iters"])
def template_stride_load_loop_kernel(
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
        acc += tl.load(x_ptr + (i & MASK) * STEP + offs)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(acc, axis=0))


def correctness_check(x, block, stride, step, out, opts):
    ref = x[0 : block * stride : stride].sum()
    template_stride_load_loop_kernel[(1,)](
        x, out, 1, BLOCK=block, STRIDE=stride, STEP=step, MASK=1, **opts
    )
    torch.npu.synchronize()
    got = out[0]
    return bool(torch.allclose(ref, got, rtol=1e-5, atol=1e-4))


def run_case(
    block,
    stride,
    num_warps,
    x,
    out,
    busy_out,
    step,
    mask,
    iters_list,
    reps,
    device,
    reference_witness_ns,
    witness_tol,
    max_attempts,
    busy_iters,
    max_spread,
    asm_dir,
):
    opts = template_opts(num_warps=num_warps, grid=1)
    grid = 1
    label = f"load_b{block}_s{stride}_w{num_warps}"
    compiled = template_stride_load_loop_kernel.warmup(
        x,
        out,
        16,
        BLOCK=block,
        STRIDE=stride,
        STEP=step,
        MASK=mask,
        grid=(grid,),
        **opts,
    )
    assert_template_call(compiled, "triton_stride_load")
    asm_keys = save_asm(compiled, asm_dir, label)

    # warm up exact runtime shape and verify numeric correctness
    template_stride_load_loop_kernel[(grid,)](
        x, out, 16, BLOCK=block, STRIDE=stride, STEP=step, MASK=mask, **opts
    )
    torch.npu.synchronize()
    correct = correctness_check(x, block, stride, step, out, opts)

    ramp = make_ramp(busy_out, busy_iters, grid=grid)
    attempts = []
    accepted = None
    for attempt in range(max_attempts):
        event_ns_list = []
        raw_rows = []
        for iters in iters_list:
            evs = []
            for _ in range(reps):
                ramp()
                evs.append(
                    measure_event_us(
                        lambda it=iters: template_stride_load_loop_kernel[(grid,)](
                            x,
                            out,
                            it,
                            BLOCK=block,
                            STRIDE=stride,
                            STEP=step,
                            MASK=mask,
                            **opts,
                        )
                    )
                )
            med_ms = statistics.median(evs)
            min_ms = min(evs)
            # The task protocol uses the minimum Event time per iteration
            # point; this rejects transient device contention/clock dips.
            event_ns_list.append(min_ms * 1e6)
            raw_rows.append(
                {
                    "iters": int(iters),
                    "event_ns_min": min_ms * 1e6,
                    "event_ns_median": med_ms * 1e6,
                    "event_ns_all": [v * 1e6 for v in evs],
                }
            )
        target = slope_ns_per_iter(iters_list, event_ns_list)
        per_iter = [
            event / max(1, it) for event, it in zip(event_ns_list, iters_list)
        ]
        spread = (
            (max(per_iter) - min(per_iter)) / min(per_iter)
            if min(per_iter) > 0
            else float("inf")
        )
        ramp()
        post_witness_ns = measure_event_us(
            lambda: busy_kernel[(grid,)](
                busy_out, busy_iters, BLOCK=1024, **busy_opts()
            )
        ) * 1e6
        witness_ok = reference_witness_ns > 0 and abs(
            post_witness_ns / reference_witness_ns - 1.0
        ) <= witness_tol
        stable_ok = spread <= max_spread
        row = {
            "attempt": attempt,
            "rows": raw_rows,
            "target_ns": target,
            "post_witness_ns": post_witness_ns,
            "witness_ok": bool(witness_ok),
            "spread": spread,
            "stable_ok": bool(stable_ok),
        }
        attempts.append(row)
    accepted_candidates = [a for a in attempts if a["witness_ok"] and a["stable_ok"]]
    # Use the minimum target among stable/boosted attempts: contention can
    # only make Event timing larger, so the minimum is the closest estimate.
    accepted = min(accepted_candidates, key=lambda a: a["target_ns"]) if accepted_candidates else None
    valid = accepted is not None
    return {
        "path": "triton_stride_load",
        "mode": "simd_simt_template",
        "block": int(block),
        "stride": int(stride),
        "num_warps": int(num_warps),
        "target_ns": float(accepted["target_ns"])
        if accepted is not None
        else float(statistics.median(r["target_ns"] for r in attempts)),
        "target_kind": "template_load_rotate_loop_min_event_slope_ns_per_iter",
        "valid": bool(valid),
        "correctness_ok": bool(correct),
        "step": int(step),
        "mask": int(mask),
        "iters_list": [int(i) for i in iters_list],
        "reps": int(reps),
        "options": opts,
        "asm_index": asm_keys,
        "attempts": attempts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--strides", nargs="+", type=int, default=list(range(3, 33)) + [40, 48, 63, 65, 80, 96, 127, 129, 160, 192, 255])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[1])
    ap.add_argument("--iters", nargs="+", type=int, default=[2000, 8000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--busy-iters", type=int, default=2_000_000)
    ap.add_argument("--step", type=int, default=8192)
    ap.add_argument("--mask", type=int, default=16383)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--witness-tol", type=float, default=0.15)
    ap.add_argument("--max-spread", type=float, default=0.40)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--asm-dir", type=Path, default=Path("asm_template_load"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    max_tile = max(args.blocks) * max(args.strides)
    total = args.mask * args.step + max_tile + 4096
    x = torch.arange(total, dtype=torch.float32, device="npu") % 7.0
    out = torch.zeros(1, dtype=torch.float32, device="npu")
    busy_out = torch.empty(4096, dtype=torch.float32, device="npu")

    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, **busy_opts())
        torch.npu.synchronize()
    freq_mhz, query_seconds = freq_check(args.device, busy_out)
    for _ in range(4):
        busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, **busy_opts())
        torch.npu.synchronize()
    reference_witness_ns = measure_event_us(
        lambda: busy_kernel[(1,)](busy_out, args.busy_iters, BLOCK=1024, **busy_opts())
    ) * 1e6

    cases = []
    t0 = time.monotonic()
    for block in args.blocks:
        for stride in args.strides:
            for num_warps in args.num_warps:
                case = run_case(
                    block,
                    stride,
                    num_warps,
                    x,
                    out,
                    busy_out,
                    args.step,
                    args.mask,
                    args.iters,
                    args.reps,
                    args.device,
                    reference_witness_ns,
                    args.witness_tol,
                    args.max_attempts,
                    args.busy_iters,
                    args.max_spread,
                    args.asm_dir,
                )
                cases.append(case)
                print(
                    f'template load b{block} s{stride} W{num_warps}: '
                    f'{case["target_ns"]:.3f} ns/iter valid={case["valid"]} '
                    f'correct={case["correctness_ok"]} '
                    f'witness={[round(a["post_witness_ns"] / 1e3, 2) for a in case["attempts"]]}us',
                    flush=True,
                )
    elapsed = time.monotonic() - t0
    payload = {
        "path": "triton_stride_load",
        "compile_mode": "simd_simt_template",
        "cases": cases,
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
    print(
        f"WROTE {args.out} valid={valid}/{len(cases)} freq={freq_mhz}MHz "
        f"query={query_seconds:.3f}s ref_witness={reference_witness_ns/1e3:.2f}us elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
