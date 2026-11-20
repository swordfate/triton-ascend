#!/usr/bin/env python3
"""E28: the mixed-body composition law (simd loop domain).

Pure anchors (loads-only / stores-only) are calibrated; this sweep maps
what x loads + y stores cost, to identify the law:
  H_max  : perIter = max(L(lB), S(sB))            (full overlap)
  H_add  : perIter = L(lB) + S(sB)                (no overlap)
  H_bw   : perIter = lat + (lB+sB)/BW             (shared L2/DRAM pipe)
  H_dep  : store-data dependency adds a serial term (DEP vs IND variants)

Families (fp32, simd, grid=1, K=64, SYS_CNT window/K):
  A_L  loads-only  (nl,le) nl{1,2,4} x le{1024,2048,4096} (reg cap)
  A_S  stores-only (ns,se) mirrored
  MD  mixed, stores DEPEND on loads (store a_j + c), le=se=e
  MI  mixed, stores INDEPENDENT constants,        le=se=e
  MD grid: nl{1,2,4} x ns{1,2,4} x e{1024,4096} (cap-feasible)
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

NS_TO_TICK = 1.00008
ITERS = 64
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""
HEADER = "import triton\nimport triton.language as tl\n\n" + COMMON
EB = 4  # fp32


def variants():
    pts = []

    def cap(nl, le):
        return nl * le * EB <= 32768

    for nl in (1, 2, 4):
        for le in (1024, 2048, 4096):
            if cap(nl, le):
                pts.append(("L_%d_%d" % (nl, le), "L", nl, le, 0, le, 0))
    for ns in (1, 2, 4):
        for se in (1024, 2048, 4096):
            if cap(ns, se):
                pts.append(("S_%d_%d" % (ns, se), "S", 0, 8, ns, se, 0))
    for nl in (1, 2, 4):
        for ns in (1, 2, 4):
            for e in (1024, 4096):
                if cap(nl, e) and cap(ns, e):
                    pts.append(("MD_%d_%d_%d" % (nl, ns, e), "MD",
                                nl, e, ns, e, 1))
                    pts.append(("MI_%d_%d_%d" % (nl, ns, e), "MI",
                                nl, e, ns, e, 0))
    return pts


def gen_module(path, variants):
    parts = [HEADER]
    for tag, kind, nl, le, ns, se, red in variants:
        ar = "lane = tl.arange(0, %d)" % le
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (s, le)
            for s in range(nl)) if nl else ""
        loads = "\n".join(
            "        a%d += tl.load(src%d + lane)" % (s, s + 1)
            for s in range(nl)) if nl else ""
        body = ""
        if kind == "S":
            v = "tl.full((%d,), 1, dtype=tl.float32)" % se
            ls = "tl.arange(0, %d)" % se
            body = "        tl.store(dst1 + %s, %s)\n" % (ls, v)
            if ns >= 2:
                body += "        tl.store(dst2 + %s, %s)\n" % (ls, v)
            if ns >= 4:
                body += ("        tl.store(dst1 + %d + %s, %s)\n"
                         "        tl.store(dst2 + %d + %s, %s)\n"
                         % (se, ls, v, se, ls, v))
        elif kind in ("MD", "MI"):
            for j in range(ns):
                dst = "dst1" if j % 2 == 0 else "dst2"
                src = "a%d" % min(j, max(nl - 1, 0))
                if kind == "MD":
                    val = "%s + %d" % (src, j + 1)
                else:
                    val = "tl.full((%d,), %d, dtype=tl.float32)" % (le, j + 1)
                body += "        tl.store(%s + lane, %s)\n" % (dst, val)
        sink = " + ".join("tl.sum(a%d)" % s for s in range(nl)) or "0.0"
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e28_%s(src1, src2, src3, src4, dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
    %s
%s
    s = 0.0
    t0 = _sys_cnt()
    for step in range(iters):
%s
%s
    t1 = _sys_cnt()
    tl.store(out + tl.arange(0, BLOCK), (%s) * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (tag, ar, accs, loads, body, sink))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e28", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def med_ns(fn, tick, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    xs = []
    for _ in range(reps):
        fn()
        torch.npu.synchronize()
        xs.append((int(tick[1].item()) - int(tick[0].item())) * NS_TO_TICK)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e28.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    vs = variants()
    gen_path = str(Path(args.out).parent / "_gen_e28.py")
    gen_module(gen_path, vs)
    mod = load_module(gen_path)

    srcs = [torch.zeros(4096, dtype=torch.float32, device="npu")
            for _ in range(4)]
    for t in srcs:
        t.fill_(1)
    dst1 = torch.zeros(16384, dtype=torch.float32, device="npu")
    dst2 = torch.zeros(16384, dtype=torch.float32, device="npu")
    out = torch.zeros(4096, dtype=torch.float32, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    rows = []
    for pass_i in range(args.passes):
        row = {}
        for tag, kind, nl, le, ns, se, red in vs:
            fn = lambda t=tag: getattr(mod, "e28_%s" % t)[(1,)](
                srcs[0], srcs[1], srcs[2], srcs[3], dst1, dst2, out,
                tick, ITERS, max(le, 8), **opts)
            try:
                row[tag] = round(med_ns(fn, tick, reps=args.reps), 1)
            except Exception as exc:
                print("SKIP", tag, type(exc).__name__, str(exc)[:80],
                      flush=True)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    data = []
    for tag, kind, nl, le, ns, se, red in vs:
        vals = [r[tag] for r in rows if tag in r]
        if not vals:
            continue
        data.append({"tag": tag, "kind": kind, "nl": nl, "le": le,
                     "ns": ns, "se": se, "lB": nl * le * EB,
                     "sB": ns * se * EB,
                     "meas": statistics.median(vals)})
    Path(args.out).write_text(
        "\n".join(json.dumps(d) for d in data) + "\n")
    for d in data:
        print("PT %-16s kind=%s lB=%6d sB=%6d meas=%8.1f" %
              (d["tag"], d["kind"], d["lB"], d["sB"], d["meas"]), flush=True)
    print("E28 DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
