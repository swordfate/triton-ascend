#!/usr/bin/env python3
"""E33: loop-domain trip-structure x grid x residency sweep (simd+simt).

Three trip structures (the audit's three classes, as general body families):
  L1    independent pipelined: a0 += load(src + p)          [existing curves]
  RED1  reduce trip: x = load; s = sum(x); store(dst+trip, s)   [rowred-like]
  DOT1  dot trip: x1 = load; x2 = load; d = sum(x1*x2); store    [gs_dot-like]
  M11   mixed: a0 += load; store(dst, a0+1)                   [continuity]

Sweep: engine {simd, simt_only} x e {128, 512, 2048} (512B/2KB/8KB)
       x g {1, 56} x res {hot, cold} x K {16, 64}
Caliber: profiler Duration differencing -> per-trip = (D64-D16)*1000/48 ns.

Usage: python3 e33_reduce_trip_sweep.py [--card 3]
"""
import argparse
import csv
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

HEADER = "import triton\nimport triton.language as tl\n"


def gen_module(path, sizes):
    parts = [HEADER]
    for e in sizes:
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e33_L1_%(e)d(src1, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    a0 = tl.zeros((%(e)d,), dtype=tl.float32)
    pid = tl.program_id(0)
    for step in range(iters):
        p = pid * iters * %(e)d + step * %(e)d + lane
        a0 += tl.load(src1 + p)
    tl.store(out + lane, a0 * 0 + 1.0)

@triton.jit(do_not_specialize=['iters'])
def e33_RED1_%(e)d(src1, dst, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    pid = tl.program_id(0)
    acc = 0.0
    for step in range(iters):
        p = pid * iters * %(e)d + step * %(e)d + lane
        x = tl.load(src1 + p)
        s = tl.sum(x)
        tl.store(dst + pid * iters + step, s)
        acc += s
    tl.store(out + lane, acc * 0 + 1.0)

@triton.jit(do_not_specialize=['iters'])
def e33_DOT1_%(e)d(src1, src2, dst, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    pid = tl.program_id(0)
    acc = 0.0
    for step in range(iters):
        p = pid * iters * %(e)d + step * %(e)d + lane
        x1 = tl.load(src1 + p)
        x2 = tl.load(src2 + p)
        d = tl.sum(x1 * x2)
        tl.store(dst + pid * iters + step, d)
        acc += d
    tl.store(out + lane, acc * 0 + 1.0)

@triton.jit(do_not_specialize=['iters'])
def e33_M11_%(e)d(src1, dst1, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    a0 = tl.zeros((%(e)d,), dtype=tl.float32)
    pid = tl.program_id(0)
    for step in range(iters):
        p = pid * iters * %(e)d + step * %(e)d + lane
        a0 += tl.load(src1 + p)
        tl.store(dst1 + p, a0 + 1.0)
    tl.store(out + lane, a0 * 0 + 1.0)
""" % {"e": e})
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g33", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=6):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e33_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=pairs),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        fn()
        torch.npu.synchronize()
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        for _ in range(pairs):
            fn()
            p.step()
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for r in csv.DictReader(h):
                dur = r.get("Duration(us)")
                name = r.get("Name") or ""
                if dur and name_filter in name:
                    durs.append(float(dur))
    shutil.rmtree(trace, ignore_errors=True)
    return round(statistics.median(durs), 3) if durs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=["simd", "simt_only"], nargs="+")
    ap.add_argument("--sizes", default=[128, 512, 2048], nargs="+", type=int)
    ap.add_argument("--grids", default=[1, 56], nargs="+", type=int)
    ap.add_argument("--out", default="/tmp/e33.jsonl")
    args = ap.parse_args()

    src = [torch.randn(56 * 64 * 2048, device="npu") for _ in range(2)]
    dst = torch.zeros(56 * 64 * 2048, device="npu")
    out = torch.zeros(2048, device="npu")
    dummy = torch.zeros(72 * 1024 * 1024, device="npu")

    bodies = ("L1", "RED1", "DOT1", "M11")
    rows = []
    for mode in args.engines:
        sizes = [s for s in args.sizes if s <= 4096] if mode != "simd" \
            else args.sizes
        gen_path = "/tmp/_gen_e33_%s.py" % mode
        gen_module(gen_path, sizes)
        m = load_module(gen_path)
        if mode == "simd":
            opts = {"compile_mode": "simd", "num_warps": 4}
        else:
            opts = {"compile_mode": "simt_only", "num_warps": 4,
                    "superblock_factor": 1}
        for body in bodies:
            for e in sizes:
                for res in ("hot", "cold"):
                    for g in args.grids:
                        d = {}
                        for K in (16, 64):
                            def fn(K=K, g=g, e=e, res=res, bd=body):
                                if res == "cold":
                                    dummy.fill_(1.0)
                                if bd == "L1":
                                    getattr(m, "e33_L1_%d" % e)[(g,)](
                                        src[0], out, K, e, **opts)
                                elif bd == "RED1":
                                    getattr(m, "e33_RED1_%d" % e)[(g,)](
                                        src[0], dst, out, K, e, **opts)
                                elif bd == "DOT1":
                                    getattr(m, "e33_DOT1_%d" % e)[(g,)](
                                        src[0], src[1], dst, out, K, e,
                                        **opts)
                                else:
                                    getattr(m, "e33_M11_%d" % e)[(g,)](
                                        src[0], dst, out, K, e, **opts)
                            d["D%d" % K] = prof_med_us(
                                fn, "e33_%s_%d" % (body, e), pairs=6)
                        slope = None
                        if d.get("D16") is not None and d.get("D64"):
                            slope = round((d["D64"] - d["D16"]) * 1000.0 / 48,
                                          1)
                        row = {"engine": mode, "body": body, "e": e,
                               "res": res, "g": g, **d, "slope_ns": slope}
                        rows.append(row)
                        print("ROW", json.dumps(row), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("E33 DONE", flush=True)


main()
