#!/usr/bin/env python3
"""Masked tile-load context calibration (layer_norm structure).

The masked.load.l_gt_32_nospill segment (397ns for a 16KB f16 tile) is an
isolated-loop calibration.  layer_norm re-loads the tile in three DEPENDENT
passes (mean -> var -> normalize): pass k issues its loads only after pass
k-1 committed, so the loads do not pipeline across passes and each pays a
fuller latency.  This probe measures exactly that structure:

    pass k: acc_k = reduce(load(tile_k))   (scalar chaining via tl.load of a
            one-element flag buffer blocks hoisting)

Kernel A: 3 dependent passes over 3 distinct 16KB tiles (f16, W=64, te=4).
Kernel B (control): same shape, scalar work, no tile loads.
cost_per_load = (slope(A) - slope(B)) / 3
"""
import argparse, csv, json, shutil
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl

@triton.jit(do_not_specialize=["iters"])
def ln3(src, flag, out, iters, TILE: tl.constexpr, M: tl.constexpr):
    program = tl.program_id(0)
    rows = tl.arange(0, M)
    cols = tl.arange(0, TILE // M)
    idx = rows[:, None] * (TILE // M) + cols[None, :]
    acc = tl.zeros((1,), tl.float32)
    for step in range(iters):
        base = (step & 3) * TILE
        x = tl.load(src + base + idx, mask=idx >= 0, other=0.0).to(tl.float32)
        s0 = tl.sum(x) / TILE
        y = tl.load(src + base + TILE + idx, mask=idx >= 0, other=0.0).to(tl.float32)
        s1 = tl.sum((y - s0) * (y - s0)) / TILE
        z = tl.load(src + base + 2 * TILE + idx, mask=idx >= 0, other=0.0).to(tl.float32)
        s2 = tl.sum((z - s0) / (tl.sqrt(s1) + 1.0))
        acc += s2 * tl.load(flag + (step & 3))
    tl.store(out + program + tl.arange(0, 1), acc)

@triton.jit(do_not_specialize=["iters"])
def ln3ctl(flag, out, iters, TILE: tl.constexpr, M: tl.constexpr):
    program = tl.program_id(0)
    acc = tl.zeros((1,), tl.float32)
    for step in range(iters):
        acc += tl.load(flag + (step & 3)) * 3.0
    tl.store(out + program + tl.arange(0, 1), acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    TILE, M = 16384, 128
    src = (torch.rand(4 * TILE + TILE) * 7).to(torch.float16).to("npu")
    flag = torch.ones(8, dtype=torch.float32, device="npu")
    out = torch.zeros(4, dtype=torch.float32, device="npu")
    lo, hi, reps = 256, 1024, 4
    opts = dict(compile_mode="simt_only", num_warps=64, superblock_factor=1)
    fa = lambda t: ln3[(1,)](src, flag, out, t, TILE, M, **opts)
    fb = lambda t: ln3ctl[(1,)](flag, out, t, TILE, M, **opts)
    for _ in range(3):
        fa(lo); fb(lo)
    torch.npu.synchronize()
    def timed(k, trips):
        td = Path("/tmp/lnt_%d" % trips)
        if td.exists(): shutil.rmtree(td)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=reps),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(td))) as prof:
            for _ in range(reps):
                k(trips); prof.step()
        torch.npu.synchronize()
        ds = []
        for p in sorted(td.rglob("kernel_details.csv")):
            for row in csv.DictReader(p.open(newline="")):
                if row.get("Duration(us)"): ds.append(float(row["Duration(us)"]))
        return min(ds) if ds else None
    for name, k in (("A", fa), ("B", fb)):
        dlo, dhi = timed(k, lo), timed(k, hi)
        slope = (dhi - dlo) * 1000.0 / (hi - lo) if dlo and dhi else None
        print(name, "lo=%.1f hi=%.1f slope=%.1f" % (dlo or -1, dhi or -1, slope or -1), flush=True)
    # per-load = (slopeA - slopeB)/3
    out_rec = dict(v=1, tile=TILE, M=M, warps=64, lo=lo, hi=hi)
    json.dump(out_rec, open(args.out, "w"), indent=1)
    print("done ->", args.out, flush=True)

if __name__ == "__main__":
    main()
