#!/usr/bin/env python3
"""Faithful layer_norm masked-tile-load calibration, incremental variants.

Replicates the real kernel structure exactly: grid=2 physical programs,
W=64 warps, each warp independently processes one 128-col f16 row
(256B): masked loads with intra-warp shuffle reductions between them.

Variants (identical shapes/masks/instruction structure, only the load
count differs):
  B:  no tile loads (register constants; reductions/stores kept)
  A1: x load only
  A2: x + weight loads
  A3: x + weight + bias loads  (= the real kernel)

Per-load in-context cost = (A2-A1), (A3-A2); sanity: (A1-B) + (A2-A1) +
(A3-A2) = A3-B = the 3-load total; A3 at trips=1 vs B at trips=1 gives
the single-shot (latency-exposed) total.
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

ROWS_PER_PROGRAM = 64
COLS = 128


@triton.jit(do_not_specialize=["iters"])
def ln_v(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, iters, NL: tl.constexpr,
         C: tl.constexpr):
    r = tl.arange(0, 64)
    c = tl.arange(0, C)
    base = tl.program_id(0) * 64
    offs = r[:, None] * C + c[None, :]
    for step in range(iters):
        x = tl.load(x_ptr + base * C + offs, mask=c < C,
                    other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / C
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        if NL >= 2:
            w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        else:
            w = tl.full((C,), 0.5, tl.float32)
        if NL >= 3:
            b = tl.load(b_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        else:
            b = tl.full((C,), 0.25, tl.float32)
        tl.store(y_ptr + base * C + offs,
                 (centered * rstd[:, None] + w[None, :] + b[None, :]).to(tl.float16))
        tl.store(mean_ptr + base * 2 * 64 + r * 2, mean)
        tl.store(mean_ptr + base * 2 * 64 + r * 2 + 1, mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    x = (-2.3 + 0.5 * torch.randn(128, COLS)).to(torch.float16).to("npu")
    w = torch.rand(COLS, dtype=torch.float16).to("npu")
    b = torch.rand(COLS, dtype=torch.float16).to("npu")
    y = torch.zeros_like(x)
    mean = torch.zeros(128 * 2, dtype=torch.float32, device="npu")
    lo, hi, reps = 64, 256, 8
    opts = dict(compile_mode="simt_only", num_warps=64, superblock_factor=1)
    res = {}
    for nl in (0, 1, 2, 3):
        fa = lambda t: ln_v[(2,)](x, w, b, y, mean, t, nl, COLS, **opts)
        for _ in range(3):
            fa(lo)
        torch.npu.synchronize()

        def timed(k, trips):
            td = Path("/tmp/lnv_%d_%d" % (nl, trips))
            if td.exists():
                shutil.rmtree(td)
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=1, warmup=1,
                                                         active=reps),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(td))) as prof:
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

        dlo, dhi = timed(fa, lo), timed(fa, hi)
        slope = (dhi - dlo) * 1000.0 / (hi - lo) if dlo and dhi else None
        one = timed(fa, 1)
        res[nl] = dict(lo=dlo, hi=dhi, slope=slope, one=one)
        print("NL=%d lo=%.1f hi=%.1f slope=%.1f one=%.1f" %
              (nl, dlo or -1, dhi or -1, slope or -1, one or -1), flush=True)
    d1 = res[1]["slope"] - res[0]["slope"]
    d2 = res[2]["slope"] - res[1]["slope"]
    d3 = res[3]["slope"] - res[2]["slope"]
    one_d = res[3]["one"] - res[0]["one"]
    print("per-load marginals: load1=%.1f load2=%.1f load3=%.1f | 3-load "
          "single-shot total=%.1f" % (d1, d2, d3, one_d), flush=True)
    res.update(load1=d1, load2=d2, load3=d3, single_shot_total=one_d)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
