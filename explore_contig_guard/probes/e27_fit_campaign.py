#!/usr/bin/env python3
"""E27: fit a parametric per-iter model for simd loop bodies with
features (nL, nS, load bytes, store bytes) - replacing the 4-bucket
clamped lookup - and compare against (M0) the shipped envelope+excess
and (M1) a plain linear model.

Sweep (bf16, simd, grid=1, K=64, SYS_CNT window / K, hot buffers):
  loads-only: nL in {1,2,4,8} x perLoad {8,64,256,1024,4096} elems
              (16B..8KB bf16), register guard nL*elems<=16384
  mixed:      3 load families x store configs (1 matched / 1 tiny /
              2 tiny / 4 tiny), plus 3 store-dominant points
Fits:
  M2  perIter = max(aL + cL*nL + lB/BL,  aS + cS*nS + sB/BS)
      (load-side params from loads-only LSQ; store-side from
       store-dominant LSQ)
  M1  linear LSQ [1, nL, nS, lB, sB]
  M0  shipped envelope (TL/TS 4-node clamp) + tiny excess
Report MAPE overall / short-axis (lB<=512B) / big subset.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

PROFILE = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
           "_C/ascend/costmodel_profiles/simd_simt/"
           "ascend_950pr_simd_simt_v1.json")
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

ELEMS = (8, 64, 256, 1024, 4096)


def sweep_points():
    pts = []
    for nl in (1, 2, 4, 8):
        for le in ELEMS:
            if nl * le <= 16384:
                pts.append((nl, le, 0, 0))
    # mixed: matched store, tiny stores
    for (nl, le) in ((2, 1024), (1, 4096), (4, 256)):
        pts.append((nl, le, 1, le))
        pts.append((nl, le, 1, 1))
        pts.append((nl, le, 2, 1))
        pts.append((nl, le, 4, 1))
    # store-dominant
    pts.append((1, 8, 1, 1024))
    pts.append((1, 8, 1, 4096))
    pts.append((2, 8, 1, 4096))
    # dedupe keep order, add tags
    seen, out = set(), []
    for (nl, le, ns, se) in pts:
        if (nl, le, ns, se) not in seen:
            seen.add((nl, le, ns, se))
            out.append(("L%dxE%d_S%dxE%d" % (nl, le, ns, se),
                        nl, le, ns, se))
    return out


def gen_module(path, variants):
    parts = [HEADER]
    for tag, nl, le, ns, se in variants:
        ar = "lane = tl.arange(0, %d)" % le
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.bfloat16)" % (s, le)
            for s in range(nl))
        loads = "\n".join(
            "        a%d += tl.load(src%d + lane)" % (s, s + 1)
            for s in range(nl))
        tstore = ("        tl.store(dst1 + tl.arange(0, 1), "
                  "tl.sum(a0) + 1.0)\n")
        stores = ""
        if ns >= 1:
            if se == le:
                stores += "        tl.store(dst1 + lane, a0 + 1.0)\n"
            else:
                stores += tstore
        if ns >= 2:
            stores += ("        tl.store(dst2 + tl.arange(0, 1), "
                       "tl.sum(a1 if %d > 1 else a0) + 2.0)\n" % nl)
        if ns >= 4:
            stores = stores + stores.replace("+ 2.0", "+ 2.0")  # dup tiny
        sink = " + ".join("tl.sum(a%d.to(tl.float32))" % s
                          for s in range(nl)) or "0.0"
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e27_%s(src1, src2, src3, src4, src5, src6, src7, src8, dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
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
""" % (tag, ar, accs, loads, stores, sink))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e27", path)
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


def lsq(X, y):
    import numpy as np
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e27.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    variants = sweep_points()
    gen_path = str(Path(args.out).parent / "_gen_e27.py")
    gen_module(gen_path, variants)
    mod = load_module(gen_path)

    srcs = [torch.zeros(e, dtype=torch.bfloat16, device="npu")
            for e in (4096, 4096, 4096, 4096)]
    for t in srcs:
        t.normal_()
    dst1 = torch.zeros(4096, dtype=torch.bfloat16, device="npu")
    dst2 = torch.zeros(4096, dtype=torch.bfloat16, device="npu")
    out = torch.zeros(4096, dtype=torch.float32, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    rows = []
    for pass_i in range(args.passes):
        row = {}
        for tag, nl, le, ns, se in variants:
            key = "L%dxE%d_S%dxE%d" % (nl, le, ns, se)
            fn = lambda t=tag: getattr(mod, "e27_%s" % t)[(1,)](
                srcs[0], srcs[1], srcs[2], srcs[3],
                srcs[0], srcs[1], srcs[2], srcs[3],
                dst1, dst2, out, tick, ITERS, max(le, 8), **opts)
            row[key] = round(med_ns(fn, tick, reps=args.reps) / ITERS, 1)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    # ---- assemble dataset ----
    data = []
    for tag, nl, le, ns, se in variants:
        key = "L%dxE%d_S%dxE%d" % (nl, le, ns, se)
        vals = [r[key] for r in rows]
        data.append({"nl": nl, "le": le, "ns": ns, "se": se,
                     "lB": nl * le * 2, "sB": ns * se * 2,
                     "meas": statistics.median(vals)})

    def m0(d):
        mixbytes = (4096.0, 8192.0, 16384.0, 32768.0)
        tl_ = (90.0, 113.7, 196.8, 392.0)
        ts_ = (106.8, 123.5, 214.8, 377.5)

        def cur(sizes, pts, b):
            b = min(max(b, sizes[0]), sizes[3])
            for i in (2, 1, 0):
                if b >= sizes[i]:
                    w = (b - sizes[i]) / (sizes[i + 1] - sizes[i])
                    return pts[i] + w * (pts[i + 1] - pts[i])
            return pts[0]
        e = max(cur(mixbytes, tl_, d["lB"]) if d["lB"] else 0.0,
                cur(mixbytes, ts_, d["sB"]) if d["sB"] else 0.0)
        if d["ns"] > 0 and d["se"] * 2 <= 128:
            c = 2 if (d["lB"] / max(1, d["nl"])) >= 16384 else 1
            e += max(0, d["ns"] - c) * 97.0
        return e

    # M2: load-side fit on loads-only subset
    L = [d for d in data if d["ns"] == 0]
    Sdom = [d for d in data if d["ns"] > 0 and
            (d["aS_pred"] if False else d["sB"]) >= d["lB"]]
    betaL = lsq([[1, d["nl"], d["lB"]] for d in L],
                [d["meas"] for d in L])
    aL, cL, invBL = betaL
    BL = 1.0 / max(invBL, 1e-9)

    def loadside(d):
        return aL + cL * d["nl"] + d["lB"] / BL

    # store-side: on mixed points, residual where meas > loadside
    M = [d for d in data if d["ns"] > 0]
    Xs, ys = [], []
    for d in M:
        resid = d["meas"] - loadside(d)
        if resid > 0:
            Xs.append([1, d["ns"], d["sB"]])
            ys.append(resid)
    if len(Xs) >= 3:
        bS = lsq(Xs, ys)
        aS, cS, invBS = bS
        BS = 1.0 / max(invBS, 1e-9)
    else:
        aS, cS, BS = 0.0, 0.0, 1.0

    def m2(d):
        return max(loadside(d), aS + cS * d["ns"] + d["sB"] / BS)

    # M1: linear
    beta1 = lsq([[1, d["nl"], d["ns"], d["lB"], d["sB"]] for d in data],
                [d["meas"] for d in data])

    def m1(d):
        return beta1[0] + beta1[1] * d["nl"] + beta1[2] * d["ns"] + \
            beta1[3] * d["lB"] + beta1[4] * d["sB"]

    def mape(pred, subset=None):
        ds = [d for d in data if not subset or subset(d)]
        if not ds:
            return float("nan")
        return 100.0 * sum(abs(pred(d) - d["meas"]) / d["meas"]
                           for d in ds) / len(ds)

    report = {
        "M2_params": {"aL": round(aL, 1), "cL": round(cL, 1),
                      "BL_Bns": round(BL, 1), "aS": round(aS, 1),
                      "cS": round(cS, 1), "BS_Bns": round(BS, 1)},
        "M1_params": [round(b, 3) for b in beta1],
        "MAPE_all": {"M0": round(mape(m0), 1), "M1": round(mape(m1), 1),
                     "M2": round(mape(m2), 1)},
        "MAPE_short": {"M0": round(mape(m0, lambda d: d["lB"] <= 512), 1),
                       "M1": round(mape(m1, lambda d: d["lB"] <= 512), 1),
                       "M2": round(mape(m2, lambda d: d["lB"] <= 512), 1)},
        "MAPE_big": {"M0": round(mape(m0, lambda d: d["lB"] >= 4096), 1),
                     "M1": round(mape(m1, lambda d: d["lB"] >= 4096), 1),
                     "M2": round(mape(m2, lambda d: d["lB"] >= 4096), 1)},
    }
    print("SUMMARY", json.dumps(report), flush=True)
    for d in data:
        print("PT nl=%d le=%d nS=%d se=%d lB=%d sB=%d meas=%.1f "
              "M0=%.1f M1=%.1f M2=%.1f" %
              (d["nl"], d["le"], d["ns"], d["se"], d["lB"], d["sB"],
               d["meas"], m0(d), m1(d), m2(d)), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"report": report,
                            "data": data}) + "\n")
    print("E27 DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
