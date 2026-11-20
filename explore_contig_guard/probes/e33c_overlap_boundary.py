#!/usr/bin/env python3
"""E33c: mixed-body read/write overlap regime boundary.

Bodies (fp32, per-load 4KB, dependent on NLOADS constexpr-free generation):
  M11  1 load + 1 store   [overlap regime reference]
  M21  2 loads + 1 store
  M41  4 loads + 1 store  [memory_stream_l4 shape]
  L4   4 loads, no store [load-side anchor]
Sweep g {1, 56}, hot, K {16, 64} differencing.
Question: does the store stay overlapped (premium ~ kappa*S) or turn
additive (premium ~ S) as nL grows / grid hits the wave size?
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


def gen_module(path, e, nls):
    parts = [HEADER]
    # L4 anchor: 4 independent loads, no store
    decl4 = "\n".join(
        "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (j, e)
        for j in range(4))
    sig4 = ", ".join("src%d" % (j + 1) for j in range(4))
    body4 = "\n".join(
        "        a%d += tl.load(src%d + p)" % (j, j + 1) for j in range(4))
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e33c_L4_%d(src1, src2, src3, src4, out, iters, "
        "BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, %d)\n"
        "%s\n"
        "    pid = tl.program_id(0)\n"
        "    for step in range(iters):\n"
        "        p = pid * iters * %d + step * %d + lane\n"
        "%s\n"
        "    tl.store(out + lane, (a0 + a1 + a2 + a3) * 0 + 1.0)\n\n"
        % (e, e, decl4, e, e, body4))
    for nl in [n for n in nls if n > 0]:
        decl = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (j, e)
            for j in range(nl))
        body = "\n".join(
            "        a%d += tl.load(src%d + p)" % (j, j + 1)
            for j in range(nl))
        sig = ", ".join("src%d" % (j + 1) for j in range(nl))
        tail = "a0"
        if nl == 2:
            tail = "a0 + a1"
        elif nl == 4:
            tail = "a0 + a1 + a2 + a3"
        store = "        tl.store(dst1 + p, %s + 1.0)" % (
            "a0" if nl == 1 else tail)
        parts.append(
            "@triton.jit(do_not_specialize=['iters'])\n"
            "def e33c_M%d1_%d(%s, dst1, out, iters, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, %d)\n"
            "%s\n"
            "    pid = tl.program_id(0)\n"
            "    for step in range(iters):\n"
            "        p = pid * iters * %d + step * %d + lane\n"
            "%s\n"
            "%s\n"
            "    tl.store(out + lane, (%s) * 0 + 1.0)\n\n"
            % (nl, e, sig, e, decl, e, e, body, store, tail))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g33c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=6):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e33c_"))
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
    ap.add_argument("--e", type=int, default=1024)
    ap.add_argument("--grids", default=[1, 56], nargs="+", type=int)
    ap.add_argument("--out", default="/tmp/e33c.jsonl")
    args = ap.parse_args()
    e = args.e
    nls = (0, 1, 2, 4)
    gen_path = "/tmp/_gen_e33c.py"
    gen_module(gen_path, e, nls)
    m = load_module(gen_path)
    opts = {"compile_mode": "simd", "num_warps": 4}

    src = [torch.randn(56 * 64 * e, device="npu") for _ in range(4)]
    dst = torch.zeros(56 * 64 * e, device="npu")
    out = torch.zeros(e, device="npu")

    rows = []
    for nl in nls:
        body = "L4" if nl == 0 else "M%d1" % nl
        for g in args.grids:
            d = {}
            for K in (16, 64):
                def fn(K=K, g=g, nl=nl):
                    if nl == 0:
                        getattr(m, "e33c_L4_%d" % e)[(g,)](
                            *src[:4] + [out, K, e], **opts)
                    else:
                        getattr(m, "e33c_M%d1_%d" % (nl, e))[(g,)](
                            *([s for s in src[:nl]] + [dst, out, K, e]),
                            **opts)
                d["D%d" % K] = prof_med_us(fn, "e33c_%s_%d" % (body, e),
                                           pairs=6)
            slope = None
            if d.get("D16") is not None and d.get("D64"):
                slope = round((d["D64"] - d["D16"]) * 1000.0 / 48, 1)
            row = {"body": body, "nl": nl, "e": e, "g": g, **d,
                   "slope_ns": slope}
            rows.append(row)
            print("ROW", json.dumps(row), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("E33C DONE", flush=True)


main()
