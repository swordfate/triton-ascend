#!/usr/bin/env python3
"""E26: user hypothesis - maybe the LOOP body should be priced per
access with the straight-line group formula (first + (n-1)*marginal
+ overflow*kc*marginal), max(loadSide, storeSide), * K + A, instead of
the E17 total-bytes envelope; that would also remove the tiny-store /
reduction corrections.

Matrix (bf16, simd, grid=1, K=64, hot buffers):
  loads-only: (nL, perLoadElems) in
     (1|2|4, 8) 16B / (1|2|4, 256) 512B / (1|2|4, 1024) 2KB
     (1, 4096) 8KB / (2, 4096) 8KB
  mixed:     (nL=2, LE=1024) + stores nS in {1x 2B, 1x 2KB, 2x 2B}
Measured: SYS_CNT window around the for-loop / K (= steady per-iter ns).
Predictions printed alongside:
  envelope : max(TL(lB), TS(sB)) + kappa*min (kappa=0) + tiny excess
  group    : max(loadGroupPrice, storeGroupPrice) per straight-line
             table (bf16 aliases fp16 rows; miss -> per-access analytical)
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
BLOCK_MAX = 4096
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""
HEADER = "import triton\nimport triton.language as tl\n\n" + COMMON


def gen_module(path, variants):
    parts = [HEADER]
    for tag, nl, le, ns, se in variants:
        loads = "\n".join(
            "        a%d += tl.load(src%d + lane)" % (s, s + 1)
            for s in range(nl))
        stores = ""
        if ns >= 1:
            stores += "        tl.store(dst1 + lane, a0 + 1.0)\n"
        if ns >= 2:
            stores += "        tl.store(dst2 + lane, a1 + 2.0)\n"
        arange = ("lane = tl.arange(0, %d)" % le)
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.bfloat16)" % (s, le)
            for s in range(nl))
        sink = " + ".join("tl.sum(a%d.to(tl.float32))" % s
                          for s in range(nl)) or "0.0"
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e26_%s(src1, src2, src3, src4, dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
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
""" % (tag, arange, accs, loads, stores, sink))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e26", path)
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


def build_tables():
    prof = json.load(open(PROFILE))
    ga = prof["simd"]["memory"]["contiguous_ldst_v2"]["group_a"]
    mix = prof["simd"]["memory"]["contiguous_ldst_v2"]["loop_envelope_mix"]
    return ga, mix["bytes"], mix["tl_ns"], mix["ts_ns"], \
        prof["simd"]["memory"]["contiguous_ldst_v2"]["group_a_ct_kc"]


def curve(sizes, points, b):
    b = max(b, sizes[0])
    b = min(b, sizes[3])
    for i in (2, 1, 0):
        if b >= sizes[i]:
            w = ((b - sizes[i]) / (sizes[i + 1] - sizes[i]))
            return points[i] + w * (points[i + 1] - points[i])
    return points[0]


def bucket(b):
    best, bb = 1e9, 32
    for cand in (32, 64, 128, 256, 512, 1024, 4096, 16384):
        d = abs(b / cand if False else 0)  # placeholder
    import math
    for cand in (32, 64, 128, 256, 512, 1024, 4096, 16384):
        d = abs(math.log2(b) - math.log2(cand))
        if d < best:
            best, bb = d, cand
    return bb


def group_price(ga, direction, dtype, b_bytes, n):
    """first + (min(n,Q)-1)*marginal + max(0,n-Q)*kc*marginal; None=miss."""
    res = "warm"
    key = "simd/%s/%s/%d/%s" % (direction, dtype, bucket(b_bytes), res)
    if key not in ga:
        return None, key
    g = ga[key]
    if direction == "store":
        q = 8
    else:
        q = min(max(65536 // max(1, b_bytes), 4), 32)
    kc = {"load": 1.231, "store": 1.003}[
        direction] if res == "warm" else \
        {"load": 2.768, "store": 1.021}[direction]
    if n > q:
        return g["first"] + (q - 1) * g["marginal"] + \
            (n - q) * kc * g["marginal"], key
    return g["first"] + (n - 1) * g["marginal"], key


def predictions(ga, mixbytes, tl_, ts_, nL, le, nS, se):
    eb = le * 2                      # bf16
    sb = se * 2
    lB, sB = nL * eb, nS * sb
    # envelope (shipped)
    tl = curve(mixbytes, tl_, lB) if lB else 0.0
    ts = curve(mixbytes, ts_, sB) if sB else 0.0
    env = max(tl, ts)
    excess = 0.0
    if nS > 0 and se * 2 <= 128:
        c = 2 if eb >= 16384 else 1
        excess = max(0, nS - c) * 97.0
    env += excess
    # group model (user proposal)
    lp, lkey = group_price(ga, "load", "fp16", eb, nL) if nL else (0.0, "")
    if lp is None:
        lp = nL * (105.309 + eb / 167.558)
    sp, skey = group_price(ga, "store", "fp16", sb, nS) if nS else (0.0, "")
    if sp is None and nS:
        sp = nS * (90.954 + sb / 213.126)
    grp = max(lp, sp)
    return env, grp, (lkey or "-", skey or "-"), (tl, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e26.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    ga, mixbytes, tl_, ts_, _ = build_tables()
    variants = [
        ("L1x8B", 1, 8, 0, 0), ("L2x8B", 2, 8, 0, 0), ("L4x8B", 4, 8, 0, 0),
        ("L1x512B", 1, 256, 0, 0), ("L2x512B", 2, 256, 0, 0),
        ("L4x512B", 4, 256, 0, 0),
        ("L1x2KB", 1, 1024, 0, 0), ("L2x2KB", 2, 1024, 0, 0),
        ("L4x2KB", 4, 1024, 0, 0),
        ("L1x8KB", 1, 4096, 0, 0), ("L2x8KB", 2, 4096, 0, 0),
        ("M_2L2KB", 2, 1024, 0, 0),
        ("M_2L2KB_S1x2B", 2, 1024, 1, 1),
        ("M_2L2KB_S1x2KB", 2, 1024, 1, 1024),
        ("M_2L2KB_S2x2B", 2, 1024, 2, 1),
    ]
    gen_path = str(Path(args.out).parent / "_gen_e26.py")
    gen_module(gen_path, variants)
    mod = load_module(gen_path)

    bufs = [torch.zeros(BLOCK_MAX, dtype=torch.bfloat16, device="npu")
            for _ in range(4)]
    dst1 = torch.zeros(BLOCK_MAX, dtype=torch.bfloat16, device="npu")
    dst2 = torch.zeros(BLOCK_MAX, dtype=torch.bfloat16, device="npu")
    out = torch.zeros(BLOCK_MAX, dtype=torch.float32, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    rows = []
    for pass_i in range(args.passes):
        row = {"pass": pass_i}
        for tag, nl, le, ns, se in variants:
            fn = lambda t=tag: getattr(mod, "e26_%s" % t)[(1,)](
                bufs[0], bufs[1], bufs[2], bufs[3], dst1, dst2, out,
                tick, ITERS, le, **opts)
            med = med_ns(fn, tick, reps=args.reps)
            row[tag] = round(med / ITERS, 1)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    print("\n%-16s %10s %10s %10s %8s  keys" %
          ("variant", "meas/iter", "envelope", "group", "env-grp"))
    summary = {}
    for tag, nl, le, ns, se in variants:
        vals = [r[tag] for r in rows]
        m = statistics.median(vals)
        env, grp, keys, (tl, ts) = predictions(ga, mixbytes, tl_, ts_,
                                               nl, le, ns, se)
        summary[tag] = {"meas": m, "env": round(env, 1),
                        "grp": round(grp, 1), "keys": keys,
                        "tl": round(tl, 1), "ts": round(ts, 1)}
        print("%-16s %10.1f %10.1f %10.1f %8.1f  %s" %
              (tag, m, env, grp, env - grp, keys))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    print("E26 DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
