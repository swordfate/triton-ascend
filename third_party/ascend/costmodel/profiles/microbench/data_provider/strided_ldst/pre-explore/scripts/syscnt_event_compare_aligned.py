#!/usr/bin/env python3
"""Compare Triton in-kernel SYS_CNT vs torch.npu.Event for the strided loop probe.

Why:
  - SIMD mode can execute `MOV $0, SYS_CNT` via tl.inline_asm_elementwise.
  - SIMT mode rejects that inline asm during MLIR lowering, so SIMT must use a
    fallback timing method (torch.npu.Event slope over an in-kernel iteration
    loop).
  - Before trusting Event for SIMT we cross-check both methods on SIMD, where
    in-kernel SYS_CNT is directly available.

Both methods measure the same Triton kernel that loops `iters` times over a
strided tile.  We take the slope against `iters`, which cancels fixed
launch/prologue costs.

Cache modes:
  - warm: STEP=0 (all iterations touch the same tile).
  - rotate: each iteration touches a different tile inside a buffer larger
    than the AIV DCache (and typically larger than L2 for the bigger cases),
    so this is closer to a cold/steady-miss measurement.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

SYS_CNT_MHZ = 999.92  # board counter, cross-checked against Event slope (period ~1.00008 ns)
PAGE_BYTES = 4096


@triton.jit
def _get_sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def loop_syscnt(x_ptr, out_ptr, t0_ptr, t1_ptr, iters,
                BLOCK: tl.constexpr, STRIDE: tl.constexpr,
                STEP: tl.constexpr, MASK: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    t0 = _get_sys_cnt()
    for i in range(iters):
        acc += tl.load(x_ptr + (i & MASK) * STEP + offs)
    t1 = _get_sys_cnt()
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(acc, axis=0))
    tl.store(t0_ptr + tl.arange(0, 1), t0)
    tl.store(t1_ptr + tl.arange(0, 1), t1)


@triton.jit
def loop_noasm(x_ptr, out_ptr, iters,
               BLOCK: tl.constexpr, STRIDE: tl.constexpr,
               STEP: tl.constexpr, MASK: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    for i in range(iters):
        acc += tl.load(x_ptr + (i & MASK) * STEP + offs)
    tl.store(out_ptr + tl.arange(0, 1), tl.sum(acc, axis=0))


def opts(mode, grid=1, num_warps=1):
    common = {
        "num_warps": int(num_warps),
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "superblock_factor": 1,
        "auto_simt_scope_mode": "off",
    }
    if mode == "simd":
        return {**common, "compile_mode": "simd", "enable_auto_blockify": False}
    return {**common, "compile_mode": "simt_only", "enable_auto_blockify": True}


def measure_event(launch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.npu.synchronize()
    # torch Event.elapsed_time is milliseconds.
    return start.elapsed_time(end) * 1e3  # us


def make_buffer(block, stride, step, mask):
    total = mask * step + block * max(stride, 1) + 4096
    # Keep the tile base page-aligned.  Rotation advances by STEP elements
    # (8192 * 4B = 32KB = 8 pages), so with a page-aligned base every
    # iteration's tile starts at a 4KB boundary.  This removes the random
    # allocator page offsets that made the unaligned V2/V3 measurements noisy.
    raw = torch.ones(total + PAGE_BYTES // 4, dtype=torch.float32, device="npu")
    offset = (-int(raw.data_ptr())) % PAGE_BYTES // 4
    return raw[offset:offset + total]


def run_case(mode, block, stride, step, mask, iters_list, reps,
             num_warps=1, x=None):
    grid = 1
    if x is None:
        x = make_buffer(block, stride, step, mask)
    out = torch.zeros(1, dtype=torch.float32, device="npu")
    t0 = torch.zeros(1, dtype=torch.int64, device="npu")
    t1 = torch.zeros(1, dtype=torch.int64, device="npu")
    o = opts(mode, grid, num_warps=num_warps)
    rows = []
    if mode == "simd":
        loop_syscnt.warmup(x, out, t0, t1, 1, BLOCK=block, STRIDE=stride,
                           STEP=step, MASK=mask, grid=(grid,), **o)
    loop_noasm.warmup(x, out, 1, BLOCK=block, STRIDE=stride,
                      STEP=step, MASK=mask, grid=(grid,), **o)

    for iters in iters_list:
        # Warm up this exact runtime specialization so JIT compile time is not
        # accidentally included in the event window.
        loop_noasm[(grid,)](x, out, iters, BLOCK=block, STRIDE=stride,
                            STEP=step, MASK=mask, **o)
        if mode == "simd":
            loop_syscnt[(grid,)](x, out, t0, t1, iters, BLOCK=block,
                                 STRIDE=stride, STEP=step, MASK=mask, **o)
        torch.npu.synchronize()
        evs, sysc = [], []
        for _ in range(reps):
            def launch_event():
                loop_noasm[(grid,)](x, out, iters, BLOCK=block, STRIDE=stride,
                                    STEP=step, MASK=mask, **o)
            evs.append(measure_event(launch_event))
            if mode == "simd":
                loop_syscnt[(grid,)](x, out, t0, t1, iters, BLOCK=block,
                                     STRIDE=stride, STEP=step, MASK=mask, **o)
                torch.npu.synchronize()
                sysc.append(int(t1.cpu().item()) - int(t0.cpu().item()))
        row = {"iters": iters,
               "event_us_median": statistics.median(evs),
               "event_us_min": min(evs)}
        if sysc:
            row["syscnt_cycles_median"] = statistics.median(sysc)
            row["syscnt_cycles_min"] = min(sysc)
        rows.append(row)

    def slope(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

    its = [r["iters"] for r in rows]
    event_slope_us = slope(its, [r["event_us_min"] for r in rows])
    result = {
        "mode": mode, "block": block, "stride": stride,
        "num_warps": int(num_warps),
        "step": step, "mask": mask, "iters": iters_list, "reps": reps,
        "buffer_data_ptr": int(x.data_ptr()),
        "buffer_page_offset": int(x.data_ptr()) % PAGE_BYTES,
        "buffer_line_offset": int(x.data_ptr()) % 128,
        "buffer_bytes": int(x.numel()) * x.element_size(),
        "rows": rows,
        "event_slope_ns_per_iter": event_slope_us * 1e3,
        "event_slope_syscnt_cycles_per_iter": event_slope_us * SYS_CNT_MHZ,
    }
    if mode == "simd":
        result["syscnt_slope_cycles_per_iter"] = slope(
            its, [r["syscnt_cycles_min"] for r in rows])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--blocks", nargs="+", type=int, default=[32])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[1])
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--iters", nargs="+", type=int, default=[4000, 8000, 16000, 32000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rotate-step", type=int, default=8192)
    ap.add_argument("--rotate-mask", type=int, default=16383)
    ap.add_argument("--warm-mask", type=int, default=3)
    ap.add_argument("--warm-only", action="store_true")
    ap.add_argument("--rotate-only", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("syscnt_event_results"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    all_results = []
    for block in args.blocks:
        for num_warps in args.num_warps:
            for mode in args.modes:
                for stride in args.strides:
                    tile_elems = block * max(stride, 1)
                    cache_modes = [("warm", tile_elems, args.warm_mask)]
                    if args.warm_only:
                        cache_modes = cache_modes[:1]
                    if not args.warm_only:
                        cache_modes = [("rotate", args.rotate_step, args.rotate_mask)]
                    if not args.warm_only and not args.rotate_only:
                        cache_modes = [("warm", tile_elems, args.warm_mask),
                                       ("rotate", args.rotate_step, args.rotate_mask)]
                    for cache, step, mask in cache_modes:
                        result = run_case(mode, block, stride, step, mask, args.iters,
                                          args.reps, num_warps=num_warps)
                        result["cache"] = cache
                        all_results.append(result)
                        msg = (f'{mode} W={num_warps} b{block} s{stride} {cache}: '
                               f'event={result["event_slope_ns_per_iter"]:.2f} ns/iter')
                        if "syscnt_slope_cycles_per_iter" in result:
                            msg += f', syscnt={result["syscnt_slope_cycles_per_iter"]:.2f} cyc/iter'
                        print(msg, flush=True)
    (args.out / "syscnt_event_results.json").write_text(
        json.dumps({"cases": all_results}, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
