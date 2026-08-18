#!/usr/bin/env python3
"""Single-block cumsum microbenchmark for SIMD and SIMT routes.

Run twice, once per route:

    TRITON_ASCEND_COMPILE_MODE=simd TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_scan.py --out ascend_results/scan_simd.jsonl

    TRITON_ASCEND_COMPILE_MODE=simt_only TRITON_ASCEND_AUTO_SIMT_SCOPE=off \
    python bench/simt_autoscope/microbench_scan.py --out ascend_results/scan_simt.jsonl

The script reports the route from the environment variable for easier merging.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def single_block_cumsum_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, y, mask=mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ascend_results/scan_microbench.jsonl")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    device = "npu" if hasattr(torch, "npu") else "cuda"
    route = os.environ.get("TRITON_ASCEND_COMPILE_MODE", "unknown")

    def sync():
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    for n in [256, 1024, 4096, 8192]:
        BLOCK = n if n <= 8192 else 8192
        x = torch.randn(n, dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        grid = (1,)
        single_block_cumsum_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4, num_stages=1)
        sync()
        for _ in range(args.warmup):
            single_block_cumsum_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4, num_stages=1)
        sync()
        if hasattr(torch, "npu") and hasattr(torch.npu, "Event"):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.reps):
                single_block_cumsum_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4, num_stages=1)
            end.record()
            sync()
            latency_ms = start.elapsed_time(end) / args.reps
        else:
            t0 = time.perf_counter()
            for _ in range(args.reps):
                single_block_cumsum_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4, num_stages=1)
            sync()
            latency_ms = (time.perf_counter() - t0) * 1000.0 / args.reps

        row = {"route": route, "n": n, "BLOCK": BLOCK, "latency_ms": latency_ms}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
