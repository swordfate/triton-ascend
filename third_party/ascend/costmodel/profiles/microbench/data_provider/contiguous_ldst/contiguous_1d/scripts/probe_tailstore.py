#!/usr/bin/env python3
"""Store-at-dependency-chain-tail calibration (index_select structure).

The single-shot store formulas were fitted from loop probes where each
write-back drain overlaps the next iteration, so the slope misses the drain.
A store at the END of a dependent chain (indices -> gather -> store) is
different: the kernel cannot complete until its writes land, so the full
issue + wait + drain lands in the kernel duration.

A: idx = load(indices); v = load(src + idx*D + col); store(out + row*D, v)
B: same, but the 1KB store replaced by a 1-element sink store
cost = slope(A) - slope(B)  (per-iteration, two-trip slope)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["iters"])
def chain_store(src, indices, out, sink, iters, D: tl.constexpr, N: tl.constexpr):
    program = tl.program_id(0)
    row = tl.arange(0, N)
    col = tl.arange(0, D)
    acc = tl.zeros((1,), tl.float32)
    for step in range(iters):
        idx = tl.load(indices + (step & 7) * N + row)
        v = tl.load(src + idx[:, None] * D + col[None, :])
        tl.store(out + (step & 7) * N * D + row[:, None] * D + col[None, :], v)
        acc += tl.sum(v) * 0.0
    tl.store(sink + program + tl.arange(0, 1), acc)


@triton.jit(do_not_specialize=["iters"])
def chain_ctl(src, indices, out, sink, iters, D: tl.constexpr, N: tl.constexpr):
    program = tl.program_id(0)
    row = tl.arange(0, N)
    col = tl.arange(0, D)
    acc = tl.zeros((1,), tl.float32)
    for step in range(iters):
        idx = tl.load(indices + (step & 7) * N + row)
        v = tl.load(src + idx[:, None] * D + col[None, :])
        acc += tl.sum(v) * 1e-30
    tl.store(sink + program + tl.arange(0, 1), acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    D, N = 32, 8
    n = 8 * N * D + N * D
    src = (torch.rand(n) * 7).to(torch.float32).to("npu")
    indices = torch.randint(0, 8 * N, (8 * N,), dtype=torch.int32).to("npu")
    out = torch.zeros(8 * N * D, dtype=torch.float32, device="npu")
    sink = torch.zeros(4, dtype=torch.float32, device="npu")
    lo, hi, reps = 256, 1024, 4
    opts = dict(compile_mode="simt_only", num_warps=1, superblock_factor=1)
    fa = lambda t: chain_store[(1,)](src, indices, out, sink, t, D, N, **opts)
    fb = lambda t: chain_ctl[(1,)](src, indices, out, sink, t, D, N, **opts)
    for _ in range(3):
        fa(lo)
        fb(lo)
    torch.npu.synchronize()

    def timed(k, trips):
        td = Path("/tmp/tail_%d" % trips)
        if td.exists():
            shutil.rmtree(td)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=reps),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(td))) as prof:
            for _ in range(reps):
                k(trips)
                prof.step()
        torch.npu.synchronize()
        ds = []
        for p in sorted(td.rglob("kernel_details.csv")):
            for row in csv.DictReader(p.open(newline="")):
                if row.get("Duration(us)"):
                    ds.append(float(row["Duration(us)"]))
        return min(ds) if ds else None

    res = {}
    for name, k in (("A", fa), ("B", fb)):
        dlo, dhi = timed(k, lo), timed(k, hi)
        slope = (dhi - dlo) * 1000.0 / (hi - lo) if dlo and dhi else None
        res[name] = dict(lo=dlo, hi=dhi, slope=slope)
        print(name, "lo=%.1f hi=%.1f slope=%.1f" % (dlo or -1, dhi or -1,
                                                    slope or -1), flush=True)
    if res["A"]["slope"] is not None and res["B"]["slope"] is not None:
        per_store = res["A"]["slope"] - res["B"]["slope"]
        print("per-store-in-chain-tail = %.1f ns" % per_store, flush=True)
        res["per_store_ns"] = per_store
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
