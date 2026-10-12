#!/usr/bin/env python3
"""Parallel warm-up of the store measurement kernel catalog.

Triton compiles one specialization per ``(mode, block, stride, num_warps)``.
A serial measurement sweep spends most of its wall time in that compile step.
This helper fills the shared ``TRITON_CACHE_DIR`` in parallel using CPU only
(the compiled kernels are not launched), after which the actual measurement
sweep hits the cache and runs at measurement speed.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store_measure_probe as smp  # noqa: E402


def worker_init(device: int, step: int, mask: int):
    torch_npu.npu.set_device(device)
    global _DUMMY, _STEP, _MASK
    _DUMMY = torch.empty(8, dtype=torch.float32, device="npu")
    _STEP = step
    _MASK = mask


def compile_one(case):
    mode, block, stride, num_warps = case
    opts = smp.simd_store_opts() if mode == "simd" else smp.simt_store_opts(num_warps)
    smp.store_loop_kernel.warmup(_DUMMY, 16, BLOCK=block, STRIDE=stride,
                                 STEP=_STEP, MASK=_MASK, grid=(1,), **opts)
    return f"{mode} b{block} s{stride} W{num_warps}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simd", "simt_only"], required=True)
    ap.add_argument("--blocks", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--strides", nargs="+", type=int,
                    default=list(range(1, 13)) +
                    [16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--num-warps", nargs="+", type=int, default=[1])
    ap.add_argument("--step", type=int, default=8192)
    ap.add_argument("--mask", type=int, default=16383)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cases = []
    for block in args.blocks:
        for stride in args.strides:
            warps = [1] if args.mode == "simd" else args.num_warps
            for num_warps in warps:
                cases.append((args.mode, block, stride, num_warps))
    cases = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard]
    print(f"precompile {args.mode}: {len(cases)} cases with {args.workers} workers",
          flush=True)
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=worker_init,
                  initargs=(args.device, args.step, args.mask)) as pool:
        for i, name in enumerate(pool.imap_unordered(compile_one, cases, chunksize=1)):
            if i % 50 == 0:
                print(f"  {i}/{len(cases)} {name} {time.monotonic()-t0:.1f}s", flush=True)
    print(f"precompile done {len(cases)} cases in {time.monotonic()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
