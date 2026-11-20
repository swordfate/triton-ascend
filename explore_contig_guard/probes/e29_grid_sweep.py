#!/usr/bin/env python3
"""E29: loop-domain grid sweep (simd, fp32) - can the grid=56 gap
(188 vs 597 ns/iter, 3.2x) be fitted as a grid factor on the L/S curves?

Bodies (runtime scf.for; per-program region = K x e elements, program p
starts at p*K*e):
  L1 / L4 : 1 / 4 independent loads per iter, unpaired acc
  S1      : 1 constant store per iter
  M11     : 1 load -> dependent store per iter
Sweep: g in {1,2,4,7,14,28,56}; e in {256,1024,4096} (per-stream bytes
1/4/16KB per load); K in {32,128}.
Caliber: profiler Duration differencing -> per-iter = (D128-D32)*1000/96
ns (launch and A(P) cancel).  Residency: hot everywhere; cold = 288MB
dummy fill before the kernel (L2 evict), e=4096 only.

Usage: python3 e29_grid_sweep.py [--bodies L1 L4 S1 M11]
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

GRIDS = (1, 2, 4, 7, 14, 28, 56)
ES = (256, 1024, 4096)
KS = (32, 128)
DUMMY_ELEMS = 72 * 1024 * 1024  # 288 MB fp32 fill = L2 evict

HEADER = "import triton\nimport triton.language as tl\n"


def gen_module(path):
    parts = [HEADER]
    for e in ES:
        for nl in (1, 4):
            srcs = ", ".join("src%d" % (j + 1) for j in range(nl))
            decl = "\n".join(
                "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (j, e)
                for j in range(nl))
            body = "\n".join(
                "        a%d += tl.load(src%d + p)" % (j, j + 1)
                for j in range(nl))
            parts.append(
                "@triton.jit(do_not_specialize=['iters'])\n"
                "def e29_L%d_%d(%s, out, iters, BLOCK: tl.constexpr):\n"
                "    lane = tl.arange(0, %d)\n"
                "%s\n"
                "    pid = tl.program_id(0)\n"
                "    for step in range(iters):\n"
                "        p = pid * iters * %d + step * %d + lane\n"
                "%s\n"
                "    tl.store(out + lane, (a0 + a%d) * 0 + 1.0)\n\n"
                % (nl, e, srcs, e, decl, e, e, body, nl - 1))
        parts.append(
            "@triton.jit(do_not_specialize=['iters'])\n"
            "def e29_S1_%d(dst, out, iters, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, %d)\n"
            "    pid = tl.program_id(0)\n"
            "    for step in range(iters):\n"
            "        p = pid * iters * %d + step * %d + lane\n"
            "        tl.store(dst + p, 1.0)\n"
            "    tl.store(out + lane, 1.0)\n\n" % (e, e, e, e))
        parts.append(
            "@triton.jit(do_not_specialize=['iters'])\n"
            "def e29_M11_%d(src1, dst1, out, iters, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, %d)\n"
            "    a0 = tl.zeros((%d,), dtype=tl.float32)\n"
            "    pid = tl.program_id(0)\n"
            "    for step in range(iters):\n"
            "        p = pid * iters * %d + step * %d + lane\n"
            "        a0 += tl.load(src1 + p)\n"
            "        tl.store(dst1 + p, a0 + 1.0)\n"
            "    tl.store(out + lane, a0 * 0 + 1.0)\n\n"
            % (e, e, e, e, e))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g29", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=6):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e29_"))
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
    ap.add_argument("--bodies", default=["L1", "L4", "S1", "M11"], nargs="+")
    ap.add_argument("--out", default="/tmp/e29.jsonl")
    args = ap.parse_args()

    gen_path = "/tmp/_gen_e29.py"
    gen_module(gen_path)
    m = load_module(gen_path)

    NE = 56 * 128 * 4096  # max region per array (117MB)
    src = [torch.randn(NE, device="npu") for _ in range(4)]
    dst = [torch.zeros(NE, device="npu") for _ in range(2)]
    out = torch.zeros(4096, device="npu")
    dummy = torch.zeros(DUMMY_ELEMS, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    rows = []
    for body in args.bodies:
        nl = int(body[1]) if body[0] == "L" else 0
        for e in ES:
            kname = "e29_%s_%d" % (body, e)
            for res in ("hot", "cold"):
                if res == "cold" and e != 4096:
                    continue
                for g in GRIDS:
                    d = {}
                    for K in KS:
                        def fn(K=K, g=g, e=e, res=res, nl=nl, bd=body):
                            if res == "cold":
                                dummy.fill_(1.0)
                            if bd[0] == "L":
                                getattr(m, "e29_%s_%d" % (bd, e))[(g,)](
                                    *([s for s in src[:nl]] + [out, K, e]),
                                    **opts)
                            elif bd[0] == "S":
                                getattr(m, "e29_%s_%d" % (bd, e))[(g,)](
                                    dst[0], out, K, e, **opts)
                            else:
                                getattr(m, "e29_%s_%d" % (bd, e))[(g,)](
                                    src[0], dst[0], out, K, e, **opts)
                        d["D%d" % K] = prof_med_us(fn, kname, pairs=6)
                    slope = None
                    if d.get("D32") is not None and d.get("D128") is not None:
                        slope = round(
                            (d["D128"] - d["D32"]) * 1000.0 / (128 - 32), 1)
                    row = {"body": body, "e": e, "res": res, "g": g,
                           **d, "slope_ns": slope}
                    rows.append(row)
                    print("ROW", json.dumps(row), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("E29 DONE", flush=True)


main()
