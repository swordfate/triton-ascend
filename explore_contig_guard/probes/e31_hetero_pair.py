#!/usr/bin/env python3
"""E31v2: do heterogeneous-shape load pairs still get queue dilution?
PROFILER DIFFERENCING caliber (v1's tl.full baselines were polluted).

shipped model: groups are same-shape only -> a hetero pair (4KB+8KB) is
two n=1 groups -> analytic + analytic, i.e. NO dilution.  User proposal:
drop the shape group key, merge by direction with one lookup key.

kernels (simd, fp32, grid=1, straight-line):
  w{e}        x=load(e); store(e, x+1)          [single anchor]
  s{e}        store(e, 1.0)
  p{e1}_{e2}  x1=load(e1); x2=load(e2); store(e1,x1+1); store(e2,x2+1)
  b{e1}_{e2}  two const stores
  t3          3x load(1024) + 3 dep stores
  b3          3 const stores(1024)
Derived: single_load(e) = w-s ; pair_load(e1,e2) = p-b ; t3_load = t3-b3.
3 passes x pairs=12, median.  Also dumps shipped groupA cells (fp32 warm
load) for first/marginal comparison.

Usage: python3 e31_hetero_pair.py
"""
import csv
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

PROFILE = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
           "_C/ascend/costmodel_profiles/simd_simt/"
           "ascend_950pr_simd_simt_v1.json")
PASSES = 3
PAIRS = 12
SINGLES = (256, 1024, 2048, 4096, 8192, 16384)
PAIRPTS = ((1024, 1024), (1024, 2048), (2048, 2048), (256, 4096),
           (1024, 4096), (4096, 4096), (512, 8192))
HEADER = "import triton\nimport triton.language as tl\n"


def gen_module(path):
    parts = [HEADER]
    for e in SINGLES:
        parts.append("""
@triton.jit
def e31w_%(e)d(src, dst, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    x = tl.load(src + lane)
    tl.store(dst + lane, x + 1.0)

@triton.jit
def e31s_%(e)d(dst, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, %(e)d)
    tl.store(dst + lane, 1.0)
""" % {"e": e})
    for e1, e2 in PAIRPTS:
        parts.append("""
@triton.jit
def e31p_%(e1)d_%(e2)d(src1, src2, dst1, dst2, out, B1: tl.constexpr,
                       B2: tl.constexpr):
    lane1 = tl.arange(0, %(e1)d)
    lane2 = tl.arange(0, %(e2)d)
    x1 = tl.load(src1 + lane1)
    x2 = tl.load(src2 + lane2)
    tl.store(dst1 + lane1, x1 + 1.0)
    tl.store(dst2 + lane2, x2 + 1.0)

@triton.jit
def e31b_%(e1)d_%(e2)d(dst1, dst2, out, B1: tl.constexpr, B2: tl.constexpr):
    lane1 = tl.arange(0, %(e1)d)
    lane2 = tl.arange(0, %(e2)d)
    tl.store(dst1 + lane1, 1.0)
    tl.store(dst2 + lane2, 1.0)
""" % {"e1": e1, "e2": e2})
    parts.append("""
@triton.jit
def e31t3(src1, src2, src3, dst1, dst2, dst3, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, 1024)
    x1 = tl.load(src1 + lane)
    x2 = tl.load(src2 + lane)
    x3 = tl.load(src3 + lane)
    tl.store(dst1 + lane, x1 + 1.0)
    tl.store(dst2 + lane, x2 + 1.0)
    tl.store(dst3 + lane, x3 + 1.0)

@triton.jit
def e31b3(dst1, dst2, dst3, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, 1024)
    tl.store(dst1 + lane, 1.0)
    tl.store(dst2 + lane, 1.0)
    tl.store(dst3 + lane, 1.0)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g31v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=PAIRS):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e31_"))
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


def dump_groupA():
    try:
        with open(PROFILE) as h:
            prof = json.load(h)
        gA = (prof.get("contiguousV2", {}) or {}).get("groupA", {}) \
            if isinstance(prof.get("contiguousV2", {}), dict) else {}
        if not gA:
            # search any dict containing keys that look like group cells
            def find(o):
                if isinstance(o, dict):
                    ks = list(o.keys())
                    if ks and all(isinstance(k, str) for k in ks[:4]) \
                            and any("warm" in k or "cold" in k for k in ks):
                        return o
                    for v in o.values():
                        r = find(v)
                        if r:
                            return r
                return None
            gA = find(prof) or {}
        print("== groupA cells matching load+warm+f32 ==")
        for k, v in sorted(gA.items()):
            if isinstance(k, str) and "load" in k and "warm" in k \
                    and ("f32" in k or "fp32" in k):
                print("  %s -> %s" % (k, json.dumps(v)))
        print("== first 15 keys ==")
        for i, k in enumerate(sorted(gA.keys())):
            if i >= 15:
                break
            print("  %s" % k)
    except Exception as exc:  # noqa: BLE001
        print("groupA dump failed: %r" % exc)


def main():
    gen_path = "/tmp/_gen_e31v2.py"
    gen_module(gen_path)
    m = load_module(gen_path)
    opts = {"compile_mode": "simd", "num_warps": 4}
    src = torch.randn(16384, device="npu")
    src2 = torch.randn(16384, device="npu")
    src3 = torch.randn(16384, device="npu")
    dst = torch.zeros(16384, device="npu")
    dst2 = torch.zeros(16384, device="npu")
    dst3 = torch.zeros(16384, device="npu")
    out = torch.zeros(1, device="npu")

    dump_groupA()

    def meas(tag, fn, pairs=PAIRS):
        return prof_med_us(fn, tag, pairs=pairs)

    rows = []
    for pass_i in range(PASSES):
        row = {"pass": pass_i, "single": {}, "pair": {}}
        for e in SINGLES:
            dw = meas("e31w_%d" % e,
                      lambda e=e: getattr(m, "e31w_%d" % e)[(1,)](
                          src, dst, out, e, **opts))
            ds = meas("e31s_%d" % e,
                      lambda e=e: getattr(m, "e31s_%d" % e)[(1,)](
                          dst, out, e, **opts))
            row["single"][e] = {"w": dw, "s": ds,
                                "load": round((dw - ds) * 1000, 1)
                                if dw and ds else None}
        for e1, e2 in PAIRPTS:
            key = "%d_%d" % (e1, e2)
            dp = meas("e31p_%s" % key,
                      lambda e1=e1, e2=e2: getattr(
                          m, "e31p_%d_%d" % (e1, e2))[(1,)](
                          src, src2, dst, dst2, out, e1, e2, **opts))
            db = meas("e31b_%s" % key,
                      lambda e1=e1, e2=e2: getattr(
                          m, "e31b_%d_%d" % (e1, e2))[(1,)](
                          dst, dst2, out, e1, e2, **opts))
            row["pair"][key] = {"p": dp, "b": db,
                                "load2": round((dp - db) * 1000, 1)
                                if dp and db else None}
        dt3 = meas("e31t3", lambda: m.e31t3[(1,)](
            src, src2, src3, dst, dst2, dst3, out, 1024, **opts))
        db3 = meas("e31b3", lambda: m.e31b3[(1,)](
            dst, dst2, dst3, out, 1024, **opts))
        row["t3"] = {"t": dt3, "b": db3,
                     "load3": round((dt3 - db3) * 1000, 1)
                     if dt3 and db3 else None}
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    print("\n== medians across passes (ns) ==")
    for e in SINGLES:
        vals = [r["single"][e]["load"] for r in rows
                if r["single"][e]["load"] is not None]
        print("single 1x%-6d load=%s" % (e * 4, statistics.median(vals)))
    for e1, e2 in PAIRPTS:
        key = "%d_%d" % (e1, e2)
        vals = [r["pair"][key]["load2"] for r in rows
                if r["pair"][key]["load2"] is not None]
        print("pair   %-9s load2=%s" % (key, statistics.median(vals)))
    vals = [r["t3"]["load3"] for r in rows if r["t3"]["load3"] is not None]
    print("triple 3x4096    load3=%s" % statistics.median(vals))
    Path("/tmp/e31.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    print("E31 DONE", flush=True)


main()
