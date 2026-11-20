#!/usr/bin/env python3
"""E27b: refined loop-domain fit campaign.  Per (mode, dtype):
  modes: simd | simt_only ; dtypes: int8 | fp16 (=bf16 alias) | fp32 | int64

Sweep families (grid=1, K=64, SYS_CNT window/K, hot buffers):
  L  loads-only     nl {1,2,4,8} x le {8,64,256,1024,4096} (byte guard)
  S  store-only     (ns,se) {(1,256),(1,1024),(1,4096),(2,1024),(4,256)}
  M  matched mix    (2,1024,1,1024) (1,4096,1,4096) (2,1024,2,1024)
  T0 tiny const     dom {(2,1024),(1,4096),(4,256)} x ns {1,2,4}
  T1 tiny red       same doms x ns {1,2,4}  (store = sum(acc))

Fit M3 (features: nL, nS, lB, sB, dom bytes):
  loadside  = stepL[nL] + max(0, lB - l0L)/BLS
  storeside = stepS[nS] + max(0, sB - l0S)/BSS
  perIter   = max(L,S) + kappa*min(L,S) + excess/chain from T0/T1 tables
M0 = shipped model (simd: envelope+excess+TL-chain; simt: line tables).
simt timing uses the E17g clock64 interception (SYS_CNT asm replaced by
ascend_dpx.clock64 in TTIR); simt loads-only windows get an in-window
dependent sink + barrier (E17g recipe).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

PROFILE = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
           "_C/ascend/costmodel_profiles/simd_simt/"
           "ascend_950pr_simd_simt_v1.json")
NS_TO_TICK = 1.00008
ITERS = 64

DTYPE_INFO = {  # name -> (triton dtype, torch dtype, bytes per element)
    "int8": ("tl.int8", torch.int8, 1),
    "fp16": ("tl.float16", torch.float16, 2),
    "fp32": ("tl.float32", torch.float32, 4),
    "int64": ("tl.int64", torch.int64, 8),
}


def _common(mode):
    cons = '"=l,~{memory}"' if mode != "simd" else '"=l"'
    return """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints=%s, args=[],
        dtype=tl.int64, is_pure=False, pack=1)
""" % cons


HEADER = "import triton\nimport triton.language as tl\n\n"
ONE = "tl.arange(0, 1)"

# --- clock64 hook (E17g recipe; inlined - e17g_simt_sweep.py runs its own
# main() at import time and must NOT be imported) ---
CLOCK_PATTERN = (
    r'tt\.elementwise_inline_asm "MOV \$0, SYS_CNT"'
    r' \{constraints = "=l,~\{memory\}", '
    r'packed_element = 1 : i32, pure = false\} -> i64'
)


def replace_markers(text):
    import re
    patched, count = re.subn(CLOCK_PATTERN, "ascend_dpx.clock64", text)
    if count < 1:
        raise ValueError("clock marker not matched (count=%d)" % count)
    return patched


def clock64_compilation():
    import subprocess
    from contextlib import contextmanager

    @contextmanager
    def hook():
        original_run = subprocess.run

        def run(command, *a, **kw):
            if (isinstance(command, (list, tuple)) and len(command) > 1
                    and Path(command[0]).name == "bishengir-compile"):
                source = Path(command[1])
                source.write_text(replace_markers(source.read_text()))
            return original_run(command, *a, **kw)

        subprocess.run = run
        try:
            yield
        finally:
            subprocess.run = original_run
    return hook()


def variants_for(mode, eb):
    # unpaired per-stream accumulators (the structure whose fit scored
    # 75/95 on the 100-case); acc bytes == load bytes, so the sweep
    # domain caps at byte_cap total load bytes per iteration.
    byte_cap = 32768 if mode == "simd" else 16384
    nacc = lambda nl: nl  # noqa: E731

    pts = []  # (tag, kind, nl, le, ns, se, red)
    for nl in (1, 2, 4, 8):
        for le in (8, 64, 256, 1024, 2048, 4096):
            if nacc(nl) * le * eb <= byte_cap:
                pts.append(("L_n%d_e%d" % (nl, le), "L", nl, le, 0, 0, 0))
    for ns, se in ((1, 256), (1, 1024), (1, 4096), (2, 1024), (4, 256),
                   (1, 8192), (2, 4096)):
        pts.append(("S_n%d_e%d" % (ns, se), "S", 0, 8, ns, se, 0))
    for nl in (1, 2, 4):
        for ns in (1, 2, 4):
            for e in (64, 256, 1024, 4096):
                if nl * e * eb <= byte_cap and ns * e * eb <= byte_cap:
                    pts.append(("M_n%d_e%d_s%d" % (nl, e, ns), "M",
                                nl, e, ns, e, 0))
    for (dl, de) in ((2, 1024), (1, 4096), (4, 256)):
        if nacc(dl) * de * eb > byte_cap:
            continue
        for ns in (1, 2, 4):
            pts.append(("T0_d%dx%d_s%d" % (dl, de, ns), "T", dl, de, ns,
                        1, 0))
            pts.append(("T1_d%dx%d_s%d" % (dl, de, ns), "T", dl, de, ns,
                        1, 1))
    return pts


def gen_module(path, variants, mode, td):
    parts = [HEADER + _common(mode)]
    for tag, kind, nl, le, ns, se, red in variants:
        ar = "lane = tl.arange(0, %d)" % max(le, 8)
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=%s)" % (s, le, td)
            for s in range(nl)) if nl else ""
        loads = "\n".join(
            "        a%d += tl.load(src%d + lane)" % (s, s + 1)
            for s in range(nl))
        body_store = ""
        if kind == "S":
            v = "tl.full((%d,), 1, dtype=%s)" % (se, td)
            ls = "tl.arange(0, %d)" % se
            body_store = "        tl.store(dst1 + %s, %s)\n" % (ls, v)
            if ns >= 2:
                body_store += ("        tl.store(dst2 + %s, %s)\n" % (ls, v))
            if ns >= 4:
                body_store += ("        tl.store(dst1 + %d + %s, %s)\n"
                               % (se, ls, v))
                body_store += ("        tl.store(dst2 + %d + %s, %s)\n"
                               % (se, ls, v))
        elif kind == "M":
            # distinct addresses per store stream (match the S-family
            # address pattern; same-buffer double-writes coalesce and
            # would corrupt the ns>=3 points)
            body_store = "        tl.store(dst1 + lane, a0 + 1)\n"
            if ns >= 2:
                src = "a1" if nl >= 2 else "a0"
                body_store += ("        tl.store(dst2 + lane, %s + 2)\n"
                               % src)
            if ns >= 3:
                src2 = "a%d" % min(2, max(nl - 1, 0))
                body_store += ("        tl.store(dst1 + %d + lane, %s + 3)\n"
                               % (le, src2))
            if ns >= 4:
                body_store += ("        tl.store(dst2 + %d + lane, a0 + 4)\n"
                               % le)
        elif kind == "T":
            for j in range(ns):
                dst = "dst1" if j % 2 == 0 else "dst2"
                if red:
                    src = "a%d" % min(j, max(nl - 1, 0))
                    val = "tl.sum(%s) + %d" % (src, j + 1)
                else:
                    val = "tl.full((1,), %d, dtype=%s)" % (j + 1, td)
                body_store += ("        tl.store(%s + %d + %s, %s)\n"
                               % (dst, j, ONE, val))
        sink = " + ".join("tl.sum(a%d.to(tl.float32))" % s
                          for s in range(nl)) or "0.0"
        # simt: loads AND stores are async.  Bodies whose stores do not
        # consume the loads need an in-window dependent sink + barrier to
        # force load completion (E17g recipe); bodies with stores need an
        # in-window readback of the LAST written address + barrier so the
        # window includes the store drain (otherwise only issue time is
        # measured).  Both are executed once, outside the K loop.
        close = ""
        if mode != "simd":
            loadsink = ("    tl.store(dst1 + %s, tl.sum(a0) * 0 + 1)\n"
                        "    tl.debug_barrier()\n" % ONE)
            if kind == "L":
                close = loadsink
            elif kind == "S":
                last = "dst2 + %d + %s" % (se, ONE) if ns >= 4 else \
                    ("dst2 + %s" % ONE if ns >= 2 else "dst1 + %s" % ONE)
                close = ("    r = tl.load(%s)\n"
                         "    tl.store(out + %s, r.to(tl.float32) * 0 + 1.0)\n"
                         "    tl.debug_barrier()\n" % (last, ONE))
            elif kind == "M":
                last = "dst2 + lane" if ns >= 2 else "dst1 + lane"
                close = ("    r = tl.load(%s)\n"
                         "    tl.store(out + %s, tl.sum(r) * 0 + 1.0)\n"
                         "    tl.debug_barrier()\n" % (last, ONE))
            elif kind == "T" and not red:
                # const stores: force load completion AND store drain
                last = "dst2 + %d + %s" % (ns - 1, ONE) if ns > 1 else \
                    "dst1 + %s" % ONE
                close = (loadsink +
                         "    r = tl.load(%s)\n"
                         "    tl.store(out + %s, r.to(tl.float32) * 0 + 1.0)\n"
                         "    tl.debug_barrier()\n" % (last, ONE))
            elif kind == "T":
                last = "dst2 + %d + %s" % (ns - 1, ONE) if ns > 1 else \
                    "dst1 + %s" % ONE
                close = ("    r = tl.load(%s)\n"
                         "    tl.store(out + %s, r.to(tl.float32) * 0 + 1.0)\n"
                         "    tl.debug_barrier()\n" % (last, ONE))
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e27b_%s(src1, src2, src3, src4, src5, src6, src7, src8, dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
    %s
%s
    s = 0.0
    t0 = _sys_cnt()
    for step in range(iters):
%s
%s
%s
    t1 = _sys_cnt()
    tl.store(out + tl.arange(0, BLOCK), (%s) * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (tag, ar, accs, loads, body_store, close, sink))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e27b", path)
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


def shipped_m0(mode, d):
    """Shipped model per-iter for one point dict (bytes-based, dtype-free)."""
    if mode == "simd":
        mb = (4096.0, 8192.0, 16384.0, 32768.0)
        tl_ = (90.0, 113.7, 196.8, 392.0)
        ts_ = (106.8, 123.5, 214.8, 377.5)

        def cur(sz, pt, b):
            b = min(max(b, sz[0]), sz[3])
            for i in (2, 1, 0):
                if b >= sz[i]:
                    w = (b - sz[i]) / (sz[i + 1] - sz[i])
                    return pt[i] + w * (pt[i + 1] - pt[i])
            return pt[0]
        e = max(cur(mb, tl_, d["lB"]) if d["lB"] else 0.0,
                cur(mb, ts_, d["sB"]) if d["sB"] else 0.0)
        if d["ns"] > 0 and d["seb"] <= 128:
            dom = d["lB"] / max(1, d["nl"])
            c = 2 if dom >= 16384 else 1
            e += max(0, d["ns"] - c) * 97.0
        if d["red"]:
            e += cur(mb, tl_, d["dom"]) + 97.0 * d["ns"]
        return e
    # simt line tables
    p = json.load(open(PROFILE))["simt"]["memory"]["contiguous_ldst_v2"]
    ln = p["loop_envelope_line_simt"]

    def cur(sz, pt, b):
        b = min(max(b, sz[0]), sz[3])
        for i in (2, 1, 0):
            if b >= sz[i]:
                w = (b - sz[i]) / (sz[i + 1] - sz[i])
                return pt[i] + w * (pt[i + 1] - pt[i])
        return pt[0]
    mb = ln["bytes"]
    e = 0.0
    if d["lB"]:
        e += (d["lB"] / 128.0) * cur(mb, ln["load_line32_l2"], d["lB"])
    if d["sB"]:
        e += (d["sB"] / 128.0) * cur(mb, ln["store_line_l2"], d["sB"])
    return e


def fit_mode(mode, variants, rows, eb):
    data = []
    for tag, kind, nl, le, ns, se, red in variants:
        vals = [r[tag] for r in rows if tag in r]
        if not vals:
            print("NOVAL", tag, flush=True)
            continue
        data.append({"tag": tag, "kind": kind, "nl": nl, "le": le,
                     "ns": ns, "se": se, "red": red,
                     "lB": nl * le * eb, "sB": ns * se * eb,
                     "seb": se * eb, "dom": le * eb,
                     "meas": statistics.median(vals)})

    L = [d for d in data if d["kind"] == "L"]
    S = [d for d in data if d["kind"] == "S"]
    M = [d for d in data if d["kind"] == "M"]
    T0 = [d for d in data if d["kind"] == "T" and not d["red"]]
    T1 = [d for d in data if d["kind"] == "T" and d["red"]]
    if not L:
        raise RuntimeError("no loads-only points measured")

    nls = sorted(set(d["nl"] for d in L))
    stepL = {}
    for nl in nls:
        small = [d["meas"] for d in L if d["nl"] == nl and d["lB"] <= 512]
        if small:
            stepL[nl] = statistics.median(small)
        else:
            stepL[nl] = min(d["meas"] for d in L if d["nl"] == nl)
    big = [d for d in L if d["lB"] >= 2048]
    best = None
    for l0 in (512, 1024, 2048):
        X = [[max(0.0, d["lB"] - l0)] for d in big]
        y = [d["meas"] - stepL[d["nl"]] for d in big]
        if len(X) < 2:
            continue
        b = lsq(X, y)
        err = sum((b[0] * x[0] - yy) ** 2 for x, yy in zip(X, y))
        if best is None or err < best[0]:
            best = (err, l0, b[0])
    if best is None:
        l0L, BLS = 1024, 160.0
    else:
        _, l0L, inv = best
        BLS = 1.0 / max(inv, 1e-9)

    nss = sorted(set(d["ns"] for d in S))
    stepS = {}
    for ns in nss:
        small = [d["meas"] for d in S if d["ns"] == ns and d["sB"] <= 1024]
        if small:
            stepS[ns] = statistics.median(small)
        else:
            stepS[ns] = min(d["meas"] for d in S if d["ns"] == ns)
    sbig = [d for d in S if d["sB"] >= 2048]
    if len(sbig) >= 2:
        best = None
        for l0 in (512, 1024, 2048):
            X = [[max(0.0, d["sB"] - l0)] for d in sbig]
            y = [d["meas"] - stepS[d["ns"]] for d in sbig]
            b = lsq(X, y)
            err = sum((b[0] * x[0] - yy) ** 2 for x, yy in zip(X, y))
            if best is None or err < best[0]:
                best = (err, l0, b[0])
        _, l0S, inv = best
        BSS = 1.0 / max(inv, 1e-9)
    elif sbig:
        d0 = sbig[0]
        l0S = 1024
        BSS = max(1.0, (d0["sB"] - l0S) /
                  max(1e-9, d0["meas"] - stepS[d0["ns"]]))
    else:
        l0S, BSS = 1024, 213.0

    def loadside(d):
        if d["nl"] == 0:
            return 0.0
        nlk = min(nls, key=lambda q: abs(q - d["nl"]))
        lo, hi = (nls[0], nls[-1]) if d["nl"] < nls[0] or d["nl"] > nls[-1] \
            else (None, None)
        if lo is not None:
            return stepL[nlk] + max(0.0, d["lB"] - l0L) / BLS
        import math
        lo2 = max([q for q in nls if q <= d["nl"]] or [nls[0]])
        hi2 = min([q for q in nls if q >= d["nl"]] or [nls[-1]])
        if lo2 == hi2:
            base = stepL[lo2]
        else:
            import math
            w = (math.log2(d["nl"]) - math.log2(lo2)) / \
                (math.log2(hi2) - math.log2(lo2))
            base = stepL[lo2] + w * (stepL[hi2] - stepL[lo2])
        return base + max(0.0, d["lB"] - l0L) / BLS

    def storeside(d):
        if d["ns"] == 0:
            return 0.0
        nsk = min(nss, key=lambda q: abs(q - d["ns"]))
        if d["ns"] < nss[0] or d["ns"] > nss[-1]:
            base = stepS[nsk]
        else:
            import math
            lo2 = max([q for q in nss if q <= d["ns"]] or [nss[0]])
            hi2 = min([q for q in nss if q >= d["ns"]] or [nss[-1]])
            if lo2 == hi2:
                base = stepS[lo2]
            else:
                w = (math.log2(d["ns"]) - math.log2(lo2)) / \
                    (math.log2(hi2) - math.log2(lo2))
                base = stepS[lo2] + w * (stepS[hi2] - stepS[lo2])
        return base + max(0.0, d["sB"] - l0S) / BSS

    kaps = []
    for d in M:
        Lv, Sv = loadside(d), storeside(d)
        if min(Lv, Sv) > 0:
            kaps.append((d["meas"] - max(Lv, Sv)) / min(Lv, Sv))
    kappa = min(1.0, max(0.0, statistics.median(kaps))) if kaps else 0.0

    def base(d):
        return max(loadside(d), storeside(d)) + \
            kappa * min(loadside(d), storeside(d))

    def deltas_of(fam):
        out = {}
        for domk in sorted(set(d["dom"] for d in fam)):
            pts = sorted([d for d in fam if d["dom"] == domk],
                         key=lambda d: d["ns"])
            dl = []
            for d in pts:
                dd = d.copy()
                dd["ns"] = 0
                dl.append((d["ns"], round(d["meas"] - base(dd), 1)))
            out[domk] = dl
        return out

    excess_fit = deltas_of(T0)
    chain_fit = deltas_of(T1)

    def interp(pairs, n):
        if not pairs:
            return 0.0
        if n <= pairs[0][0]:
            return pairs[0][1] * n / max(1, pairs[0][0])
        for (n0, v0), (n1, v1) in zip(pairs, pairs[1:]):
            if n <= n1:
                return v0 + (v1 - v0) * (n - n0) / (n1 - n0)
        (n0, v0), (n1, v1) = pairs[-2], pairs[-1]
        return v1 + (v1 - v0) / max(1, n1 - n0) * (n - n1)

    def domkey(dom):
        keys = sorted(excess_fit.keys() or chain_fit.keys())
        if not keys:
            return None
        return min(keys, key=lambda k: abs(math_log2(k) - math_log2(dom)))

    def math_log2(x):
        import math
        return math.log2(max(1.0, x))

    def m3(d):
        b = base(d)
        if d["kind"] == "T":
            tab = chain_fit.get(domkey(d["dom"])) if d["red"] else \
                excess_fit.get(domkey(d["dom"]))
            return b + interp(tab or [], d["ns"])
        if d["ns"] > 0 and d["seb"] <= 128 and not d["red"]:
            tab = excess_fit.get(domkey(d["dom"]))
            return b + interp(tab or [], d["ns"])
        if d["red"] and d["ns"] > 0:
            tab = chain_fit.get(domkey(d["dom"]))
            return b + interp(tab or [], d["ns"])
        return b

    def mape(pred, subset=None):
        ds = [d for d in data if not subset or subset(d)]
        if not ds:
            return float("nan")
        return 100.0 * sum(abs(pred(d) - d["meas"]) / d["meas"]
                           for d in ds) / len(ds)

    report = {
        "stepL": {str(k): round(v, 1) for k, v in stepL.items()},
        "byte_load": {"l0": l0L, "B_ns": round(BLS, 1)},
        "stepS": {str(k): round(v, 1) for k, v in stepS.items()},
        "byte_store": {"l0": l0S, "B_ns": round(BSS, 1)},
        "kappa": round(kappa, 3),
        "excess_T0_deltas": {"%d" % k: v for k, v in excess_fit.items()},
        "chain_T1_deltas": {"%d" % k: v for k, v in chain_fit.items()},
        "MAPE_all": {"M0": round(mape(lambda d: shipped_m0(mode, d)), 1),
                     "M3": round(mape(m3), 1)},
        "MAPE_loads": {"M0": round(mape(lambda d: shipped_m0(mode, d),
                                        lambda d: d["kind"] == "L"), 1),
                       "M3": round(mape(m3, lambda d: d["kind"] == "L"), 1)},
        "MAPE_tiny": {"M0": round(mape(lambda d: shipped_m0(mode, d),
                                       lambda d: d["kind"] == "T"), 1),
                      "M3": round(mape(m3, lambda d: d["kind"] == "T"), 1)},
    }
    return report, data, m3, lambda d: shipped_m0(mode, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="simd")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", default="/tmp/e27b.jsonl")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    td, tdt, eb = DTYPE_INFO[args.dtype]
    variants = variants_for(args.mode, eb)
    gen_path = str(Path(args.out).parent / "_gen_e27b.py")
    gen_module(gen_path, variants, args.mode, td)
    mod = load_module(gen_path)

    srcs = [torch.zeros(8192, dtype=tdt, device="npu") for _ in range(8)]
    for t in srcs:
        t.fill_(1)
    dst1 = torch.zeros(16384, dtype=tdt, device="npu")
    dst2 = torch.zeros(16384, dtype=tdt, device="npu")
    out = torch.zeros(4096, dtype=torch.float32, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": args.mode, "num_warps": 4}
    if args.mode == "simt_only":
        opts["superblock_factor"] = 1

    rows = []
    clock_hook = clock64_compilation if args.mode != "simd" else None
    for pass_i in range(args.passes):
        row = {}
        for tag, kind, nl, le, ns, se, red in variants:
            fn = lambda t=tag: getattr(mod, "e27b_%s" % t)[(1,)](
                *srcs, dst1, dst2, out, tick, ITERS, max(le, 8), **opts)
            try:
                if clock_hook is not None:
                    with clock_hook():
                        row[tag] = round(med_ns(fn, tick,
                                                reps=args.reps) / ITERS, 1)
                else:
                    row[tag] = round(med_ns(fn, tick,
                                            reps=args.reps) / ITERS, 1)
            except Exception as exc:
                print("SKIP", tag, type(exc).__name__,
                      str(exc)[:120], flush=True)
                continue
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    report, data, m3, m0 = fit_mode(args.mode, variants, rows, eb)
    print("SUMMARY", json.dumps(report), flush=True)
    for d in data:
        print("PT %-18s meas=%7.1f M0=%7.1f M3=%7.1f" %
              (d["tag"], d["meas"], m0(d), m3(d)), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"mode": args.mode, "dtype": args.dtype,
                            "report": report, "data": data}) + "\n")
    print("E27B DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
