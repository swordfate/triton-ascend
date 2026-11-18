#!/usr/bin/env python3
"""Faithful layer_norm masked-tile-load calibration.

Replicates the real kernel structure exactly: grid=2 physical programs,
W=64 warps, each warp independently processes one 128-col f16 row
(256B): 3 masked loads (x / weight / bias - weight and bias are the
same addresses for all warps = L2 broadcast) with intra-warp shuffle
reductions between them.

A: full kernel (3 loads + reductions + stores)
B: the 3 tile loads replaced by register constants (reductions/stores kept)
delta = the 3 loads' true in-context total; per-load = delta / 3
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

ROWS = 128  # logical rows; grid=2 programs x 64 rows each (F=64)
COLS = 128


@triton.jit(do_not_specialize=["iters"])
def ln_full(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, iters, N: tl.constexpr,
            C: tl.constexpr):
    warp_id = tl.program_id(0) * (N // 32) + tl.arange(0, N // 32)
    base = tl.program_id(0) * (N // 32)  # rows owned by this program
    row0 = tl.program_id(0) * 64
    r = tl.arange(0, 64)
    c = tl.arange(0, C)
    offs = r[:, None] * C + c[None, :]
    for step in range(iters):
        x = tl.load(x_ptr + base * C + offs,
                    mask=(r + base) < 10**6, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        tl.store(y_ptr + base * C + offs,
                 (centered * rstd[:, None] + w[None, :] + b[None, :]).to(tl.float16))
        tl.store(mean_ptr + base * 32 + r * 2, mean)
        acc = tl.sum(rstd) * 0.0
        tl.store(mean_ptr + base * 32 + r * 2 + 1, mean + acc)


@triton.jit(do_not_specialize=["iters"])
def ln_noload(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, iters, N: tl.constexpr,
              C: tl.constexpr):
    r = tl.arange(0, 64)
    c = tl.arange(0, C)
    offs = r[:, None] * C + c[None, :]
    for step in range(iters):
        x = (tl.full((64, C), 1.0, tl.float32) + step * 0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        w = tl.full((C,), 0.5, tl.float32)
        b = tl.full((C,), 0.25, tl.float32)
        tl.store(y_ptr + tl.program_id(0) * 64 * C + offs,
                 (centered * rstd[:, None] + w[None, :] + b[None, :]).to(tl.float16))
        tl.store(mean_ptr + tl.program_id(0) * 64 * 32 + r * 2, mean)
        tl.store(mean_ptr + tl.program_id(0) * 64 * 32 + r * 2 + 1, mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    x = (-2.3 + 0.5 * torch.randn(ROWS, COLS)).to(torch.float16).to("npu")
    w = torch.rand(COLS, dtype=torch.float16).to("npu")
    b = torch.rand(COLS, dtype=torch.float16).to("npu")
    y = torch.zeros_like(x)
    mean = torch.zeros(ROWS * 2, dtype=torch.float32, device="npu")
    lo, hi, reps = 64, 256, 8
    opts = dict(compile_mode="simt_only", num_warps=64, superblock_factor=1)
    fa = lambda t: ln_full[(2,)](x, w, b, y, mean, t, ROWS, COLS, **opts)
    fb = lambda t: ln_noload[(2,)](x, w, b, y, mean, t, ROWS, COLS, **opts)
    for _ in range(3):
        fa(lo)
        fb(lo)
    torch.npu.synchronize()

    def timed(k, trips):
        td = Path("/tmp/lnf_%d" % trips)
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
    for name, k in (("A_full", fa), ("B_noload", fb)):
        dlo, dhi = timed(k, lo), timed(k, hi)
        slope = (dhi - dlo) * 1000.0 / (hi - lo) if dlo and dhi else None
        res[name] = dict(lo=dlo, hi=dhi, slope=slope)
        print(name, "lo=%.1f hi=%.1f slope=%.1f" % (dlo or -1, dhi or -1,
                                                    slope or -1), flush=True)
    if res["A_full"]["slope"] is not None and res["B_noload"]["slope"] is not None:
        delta = res["A_full"]["slope"] - res["B_noload"]["slope"]
        per_load = delta / 3.0
        print("3-loads-in-context total = %.1f ns; per-load = %.1f ns" %
              (delta, per_load), flush=True)
        res["delta_ns"] = delta
        res["per_load_ns"] = per_load
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
