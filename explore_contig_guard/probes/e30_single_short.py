#!/usr/bin/env python3
"""E30v3: straight-line single-shot price vs bytes - PROFILER DIFFERENCING
caliber.  (v1/v2 in-kernel windows were abandoned: the s*0 tick dependency
constant-folds, tl.full baselines carry their own UB/fill memory cost, and
cards 1/2 had neighbor jobs at 100% util polluting everything.)

Whole-kernel profiler Duration at grid=1, warm; differences cancel launch/A:
  load_price(b)  = D_work(b) - D_storeonly(b)   (identical store; the store
                 depends on the load so the load cost is fully serialised in)
  store_price(b) = D_work(b) - D_loadsink(b)    (raw; includes the scalar
                 sum-sink offset, reported separately)
Kernels per (engine, e):  work  x=load(e); store(e, x+1)
                          storeonly  store(e, 1.0)
                          loadsink   x=load(e); store(out, sum(x))
                          min        store(32B)   [A(1) reference]
5 passes x (pairs=16 profiler launches, median) -> median across passes.

Usage: python3 e30_single_short.py [--engines simd simt_only]
"""
import argparse
import csv
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

SIZES_SIMD = (8, 128, 512, 2048, 4096, 8192, 16384)
SIZES_SIMT = (8, 128, 512, 2048, 4096)
PASSES = 5
PAIRS = 16
HEADER = "import triton\nimport triton.language as tl\n"

MIN_KERNEL = """
@triton.jit
def e30min(dst, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, 32)
    tl.store(dst + lane, 1.0)
"""


def gen_module(path, sizes):
    parts = [HEADER]
    for e in sizes:
        parts.append("""
@triton.jit
def e30w_%(e)d(src, dst, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    x = tl.load(src + lane)
    tl.store(dst + lane, x + 1.0)

@triton.jit
def e30s_%(e)d(dst, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    tl.store(dst + lane, 1.0)

@triton.jit
def e30l_%(e)d(src, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    x = tl.load(src + lane)
    tl.store(out, tl.sum(x))
""" % {"e": e})
    parts.append(MIN_KERNEL)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g30v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=PAIRS):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e30_"))
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


def shipped_load(b):
    return 105.31 + b / 167.56


def shipped_store(b):
    return 90.95 + b / 213.13


def v8_load(b):
    return (91.0407 + 3.5736 * math.log2(b) + 63.1191 * 4.0 / b
            + 0.7822 * (b / 256.0))


def v8_store(b):
    return (-37.7017 + 0.0083 * b + 6.3883 * math.log2(b)
            - 0.9234 * math.sqrt(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=["simd", "simt_only"], nargs="+")
    ap.add_argument("--out", default="/tmp/e30.jsonl")
    args = ap.parse_args()

    src = torch.randn(16384, device="npu")
    dst = torch.zeros(16384, device="npu")
    out = torch.zeros(1, device="npu")

    rows = []
    for mode in args.engines:
        sizes = SIZES_SIMD if mode == "simd" else SIZES_SIMT
        gen_path = "/tmp/_gen_e30v3_%s.py" % mode
        gen_module(gen_path, sizes)
        m = load_module(gen_path)
        opts = ({"compile_mode": "simd", "num_warps": 4} if mode == "simd"
                else {"compile_mode": "simt_only", "num_warps": 4,
                      "superblock_factor": 1})

        def fmin():
            m.e30min[(1,)](dst, out, 32, **opts)
        dmin = prof_med_us(fmin, "e30min")
        print("MIN", mode, dmin, flush=True)

        for pass_i in range(PASSES):
            for e in sizes:
                b = e * 4

                def fw(e=e):
                    getattr(m, "e30w_%d" % e)[(1,)](src, dst, out, e, **opts)

                def fs(e=e):
                    getattr(m, "e30s_%d" % e)[(1,)](dst, out, e, **opts)

                def fl(e=e):
                    getattr(m, "e30l_%d" % e)[(1,)](src, out, e, **opts)
                dw = prof_med_us(fw, "e30w_%d" % e)
                ds = prof_med_us(fs, "e30s_%d" % e)
                dl = prof_med_us(fl, "e30l_%d" % e)
                rec = {"mode": mode, "pass": pass_i, "e": e, "B": b,
                       "Dw": dw, "Ds": ds, "Dl": dl, "Dmin": dmin,
                       "load_price": round((dw - ds) * 1000, 1)
                       if dw and ds else None,
                       "store_raw": round((dw - dl) * 1000, 1)
                       if dw and dl else None}
                rows.append(rec)
                print("REC", json.dumps(rec), flush=True)

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("\n== medians (ns) ==")
    for mode in args.engines:
        sizes = SIZES_SIMD if mode == "simd" else SIZES_SIMT
        print("-- %s --" % mode)
        print("%7s %8s %8s %8s | %8s %8s %8s" %
              ("B", "load", "analytL", "v8load", "store_raw", "analytS",
               "v8store"))
        for e in sizes:
            b = e * 4
            sel = [r for r in rows if r["mode"] == mode and r["e"] == e]

            def med(k):
                vals = [r[k] for r in sel if r[k] is not None]
                return statistics.median(vals) if vals else float("nan")
            print("%7d %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f" %
                  (b, med("load_price"), shipped_load(b),
                   v8_load(b) if mode == "simd" else float("nan"),
                   med("store_raw"), shipped_store(b),
                   v8_store(b) if mode == "simd" else float("nan")))
    print("E30 DONE", flush=True)


main()
