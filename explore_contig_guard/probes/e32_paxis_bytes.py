#!/usr/bin/env python3
"""E32: is the straight-line multi-wave discount (pAxisFactor / waves^-0.32)
byte-dependent?  Shipped model: pAxisFactor(kind/engine/residency, P) is
byte-blind (fitted on one small geometry); single loads additionally decay
waves^-0.32 while single stores do not decay at all.  Both claims get
tested here by sweeping P at several per-program byte sizes.

simd, straight-line, profiler Duration.  Per-program price =
(D_kernel(P) - D_min(P)) / P with D_min = minimal 128B-store kernel at the
same P (A(P) proxy).  factor(P) = price(P) / price(1).

Bodies (per program, single running accumulator):
  L_1x1024  (4KB)   L_2x4096 (32KB)   L_8x4096 (128KB)   L_32x4096 (512KB)
  S_1x1024  (4KB)   S_1x8192 (32KB)
P in {1,4,14,28,56,112,224}; cold = 288MB dummy fill (L2 evict) at
b in {32KB,128KB,512KB} x P in {56,224}.

Usage: python3 e32_paxis_bytes.py
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

PS = (1, 4, 14, 28, 56, 112, 224)
DUMMY_ELEMS = 72 * 1024 * 1024  # 288MB fill
HEADER = "import triton\nimport triton.language as tl\n"


def gen_module(path):
    parts = [HEADER]
    for nl, e in ((1, 1024), (2, 4096), (4, 8192), (2, 16384)):
        # single running accumulator: a0 += load_j sequentially (keeps the
        # live register set at ~2 vectors instead of nl vectors)
        body = "\n".join(
            "    a0 += tl.load(src + base + %d * %d + lane)" % (j, e)
            for j in range(nl))
        parts.append(
            "@triton.jit\n"
            "def e32_L_%dx%d(src, out, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, %d)\n"
            "    base = tl.program_id(0) * %d * %d\n"
            "    a0 = tl.zeros((%d,), dtype=tl.float32)\n"
            "%s\n"
            "    tl.store(out + lane, a0 * 0 + 1.0)\n\n"
            % (nl, e, e, nl, e, e, body))
    for e in (1024, 8192):
        parts.append(
            "@triton.jit\n"
            "def e32_S_1x%d(dst, out, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, %d)\n"
            "    base = tl.program_id(0) * %d\n"
            "    tl.store(dst + base + lane, 1.0)\n"
            "    tl.store(out + lane, 1.0)\n\n" % (e, e, e))
    parts.append(
        "@triton.jit\n"
        "def e32_min(dst, out, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, 32)\n"
        "    base = tl.program_id(0) * 32\n"
        "    tl.store(dst + base + lane, 1.0)\n"
        "    tl.store(out + lane, 1.0)\n\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g32", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=6):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e32_"))
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
    ap.add_argument("--bodies", default=["L_1x1024", "L_2x4096",
                                         "L_8x4096", "L_32x4096",
                                         "S_1x1024", "S_1x8192"], nargs="+")
    ap.add_argument("--out", default="/tmp/e32.jsonl")
    args = ap.parse_args()

    gen_path = "/tmp/_gen_e32.py"
    gen_module(gen_path)
    m = load_module(gen_path)
    opts = {"compile_mode": "simd", "num_warps": 4}

    NE = 224 * 32 * 4096  # 117MB
    src = torch.randn(NE, device="npu")
    dst = torch.zeros(NE, device="npu")
    out = torch.zeros(4096, device="npu")
    dummy = torch.zeros(DUMMY_ELEMS, device="npu")

    rows = []
    for body in args.bodies:
        kname = "e32_%s" % body
        for res in ("hot", "cold"):
            if res == "cold" and body.startswith("L_1x"):
                continue
            for P in PS:
                def fn(P=P, res=res, body=body):
                    if res == "cold":
                        dummy.fill_(1.0)
                    if body.startswith("L_"):
                        getattr(m, "e32_%s" % body)[(P,)](src, out, 4096,
                                                          **opts)
                    else:
                        e_s = int(body.split("x")[1])
                        getattr(m, "e32_%s" % body)[(P,)](dst, out, e_s,
                                                          **opts)

                def fmin(P=P):
                    m.e32_min[(P,)](dst, out, 32, **opts)

                d = prof_med_us(fn, kname, pairs=6)
                dmin = prof_med_us(fmin, "e32_min", pairs=6)
                pp = None
                if d is not None and dmin is not None:
                    pp = round((d - dmin) * 1000.0 / P, 1)
                row = {"body": body, "res": res, "P": P, "D": d,
                       "Dmin": dmin, "perprog_ns": pp}
                rows.append(row)
                print("ROW", json.dumps(row), flush=True)
                with open(args.out + ".inc", "a") as h:
                    h.write(json.dumps(row) + "\n")

    print("\n== factors price(P)/price(1) ==")
    for body in args.bodies:
        for res in ("hot", "cold"):
            sel = [r for r in rows if r["body"] == body and r["res"] == res]
            if len(sel) < 2 or sel[0]["perprog_ns"] is None:
                continue
            base = sel[0]["perprog_ns"]
            fac = ["%.3f" % (r["perprog_ns"] / base) if r["perprog_ns"]
                   else "-" for r in sel]
            print("%-10s %-4s base=%7.1fns  factors %s" %
                  (body, res, base, " ".join(fac)))
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("E32 DONE", flush=True)


main()
