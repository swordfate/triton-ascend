#!/usr/bin/env python3
"""Fit the SIMT tail (element count is not a power of two) segments.

Geometry is the REAL element count n, not the padded block: a masked access is
priced by what it actually touches.  The feature set is byte-for-byte the one
assemble_spec.feats() computes, so the assembled spec needs no second
implementation.

Two flags carry the measured, dtype dependent threshold:

    slow  = [n % 32 != 0 and te >= 4]        the original, fp16-shaped threshold
    slowK = [n % 32 != 0 and te >= 8 / eb]   one SIMT_LDG moves at most 8 B per
                                             thread, so a thread owning more than
                                             8/eb elements needs a second one

The knee sits at te = 8/eb (int64 1, fp32 2, fp16 4, int8 8).  It is visible as a
jump between the n = 2^k - 1 and n = 2^k + 1 neighbours, and slowK lowers the tail
error from 7.24% to 6.89%.

Writes fitted/SIMT_TAIL_SEGMENTS.json.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
# 每线程寄存器数按线程数分档（AIV SIMT RegFile 128KB）：≤256→127, ≤512→64, ≤1024→32, ≤2048→16
REGB = {1: 508.0, 2: 508.0, 4: 508.0, 8: 508.0, 16: 256.0, 32: 128.0, 64: 64.0}
MAXN = {"load": 2048, "store": 8192}


def pad(n):
    return 1 << max(0, (n - 1).bit_length())


def kof(x, cap=2):
    return 2.0 ** min(cap, math.floor(math.log2(max(1.0, x))))


def feats(n, eb, w):
    by = n * eb
    lines = by / 128.0
    te = n / (32.0 * w)
    lpW = lines / w
    slow = 1.0 if (n % 32 and te >= 4.0) else 0.0
    slowK = 1.0 if (n % 32 and te >= 8.0 / eb) else 0.0
    part = 1.0 if (n * eb) % 128 else 0.0
    return dict(by=by, lines=lines, te=te, lpW=lpW, W=w, invW=1.0 / w, W2=w * w,
                rounds=math.ceil(te), K1=by * kof(lpW),
                K2=by * 2.0 ** min(2, math.floor(math.log2(max(1.0, te / 16)))),
                LN=by * math.log(1 + te), lin=1.0 if by >= 128 else 0.0,
                st=1.0 if te < 1 else 0.0, stW=(1.0 if te < 1 else 0.0) * w,
                lnW=math.log(1 + w), WlnW=w * math.log(1 + w),
                byLnW=by * math.log(1 + w), i8=1.0 if eb == 1 else 0.0,
                i8K=(by if eb == 1 else 0.0) * kof(lpW), g=max(0.0, te - 32),
                g2By=by * max(0.0, te - 64), waves_by=math.ceil(te) * by,
                lanes=min(n, 32.0 * w), s32=1.0 if n < 32 else 0.0,
                s32W=(1.0 if n < 32 else 0.0) * w, slow=slow, fast=1.0 - slow,
                slow_by=by * slow, fast_by=by * (1.0 - slow),
                slowK=slowK, fastK=1.0 - slowK,
                slowK_by=by * slowK, fastK_by=by * (1.0 - slowK),
                teK=math.ceil(te / (8.0 / eb)),
                part=part, part_by=by * part, half=1.0 if eb == 2 else 0.0,
                half_fast=((1.0 - slow) if eb == 2 else 0.0),
                half_fast_by=((by * (1.0 - slow)) if eb == 2 else 0.0),
                half_slow_by=((by * slow) if eb == 2 else 0.0),
                fill=(32.0 * eb) / 256.0, invWby=by / w, by2=by * by,
                linW=(1.0 if by >= 128 else 0.0) * w, lines_part=part,
                lines_full=1.0 - part, half_slow=(slow if eb == 2 else 0.0),
                fast_lnW=math.log(1 + w) * (1.0 - slow),
                slow_lnW=math.log(1 + w) * slow, byLnW2=by * math.log(1 + w),
                spill=max(0.0, (n * eb) / (32.0 * w) - REGB[w]) / REGB[w],
                spillL=by / 128.0 * (1.0 if (n * eb) / (32.0 * w) > REGB[w] else 0.0),
                wpl=max(1.0, 128.0 / (32.0 * eb)),
                linwpl=lines / max(1.0, 128.0 / (32.0 * eb)),
                rsqrtW=1.0 / math.sqrt(w),
                Wh=max(0.0, w - 16.0), Wh8=max(0.0, w - 8.0),
                Wh_by=by * max(0.0, w - 16.0), Wh8_by=by * max(0.0, w - 8.0),
                Wlin=(w if w <= 16 else 16.0))


def verdict(r):
    us = r.get("min_us") or {}
    ks = sorted(int(x) for x in us)
    if len(ks) != 2:
        return False
    a, b = us[str(ks[0])], us[str(ks[1])]
    return (r.get("status") == "ok" and r.get("cost_per_operation", 0) > 0
            and b >= 5.0 and abs((b / a) / (ks[1] / ks[0]) - 1) <= 0.25
            and r.get("spread", 1) <= 0.10)


def load():
    rows = []
    for name in ("raw_npow2_fp.jsonl", "raw_npow2_w14.jsonl"):
        path = os.path.join(ROOT, "raw", name)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not verdict(r):
                continue
            n = r["elements"]
            eb = EB[r["dtype"]]
            if pad(n) * eb < 128 or n > MAXN[r["direction"]]:
                continue
            rows.append(dict(cost=r["cost_per_operation"], _d=r["direction"],
                             _dt=r["dtype"], _n=n, _w=r["warps"], **feats(n, eb, r["warps"])))
    return rows


def key_of(r):
    if r["_d"] == "load":
        # load 走 fit_bottleneck.py 的瓶颈形式，不在这里分段
        return None
    if r["te"] < 1:
        return "tail|store|te<1"
    # W 只分两档（≤4 / >4）：地板随 W 从 30 涨到 132，但这一段靠地板项就能吸收，
    # 分成四档只把总误差从 9.33% 降到 7.92%，多 4 段 24 个系数，不值得。
    # 窄元素（int8/fp16）与宽元素（fp32/int64）归两族：两者的慢路径增长率差 2~4 倍，
    # 而族内几乎相同（见 PR_SUMMARY 5b.6b）；三分会让留出误差从 9.99% 升到 10.68%。
    base = "narrow" if r["i8"] + r["half"] > 0 else "wide"
    _w = r["_w"]
    wgd = "W1" if r["_w"]<=4 else "W2_4" if r["_w"]<=16 else "Wgt4"
    return "tail|store|te>=1,%s,%s" % (base, wgd)


BUDGET = {"tail|load|lines<=2": 5, "tail|load|2<lines<=32": 5,
          "tail|load|l>32": 5,
          "tail|store|te<1": 5}
for b in ("narrow", "wide"):
    for wgd in ["W1","W2_4","Wgt4"]:
        BUDGET["tail|store|te>=1,%s,%s" % (b, wgd)] = 5


def fit(rows, names):
    n = len(rows)
    if n < len(names) + 3:
        return None
    X = np.stack([np.ones(n)] + [np.array([r[f] for r in rows], float) for f in names], 1)
    y = np.array([r["cost"] for r in rows], float)
    w = 1.0 / y
    co = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return float((np.abs(X @ co - y) / y * 100).mean()), co


def greedy(rows, budget, gain=0.05):
    names = [k for k in rows[0] if k != "cost" and not k.startswith("_")]
    chosen = []
    best = fit(rows, [])
    if best is None:
        return None
    cur, co = best
    while len(chosen) < budget:
        pick = None
        for nm in names:
            if nm in chosen:
                continue
            got = fit(rows, chosen + [nm])
            if got and (pick is None or got[0] < pick[0]):
                pick = (got[0], nm, got[1])
        if pick is None or pick[0] > cur * (1 - gain):
            break
        chosen.append(pick[1])
        cur, co = pick[0], pick[2]
    return dict(n=len(rows), feats=chosen, coefs=[float(c) for c in co],
                mape=round(cur, 4))


def main():
    rows = load()
    spec = {}
    print("%-32s %5s %8s  %s" % ("segment", "n", "MAPE", "features"))
    for key, budget in BUDGET.items():
        rs = [r for r in rows if key_of(r) == key]
        if len(rs) < 5:
            continue
        g = greedy(rs, min(budget, max(1, len(rs) - 3)))
        spec[key] = g
        print("%-32s %5d %7.2f%%  %s" % (key, g["n"], g["mape"], ",".join(g["feats"])))
    tot = sum(g["mape"] * g["n"] for g in spec.values())
    cnt = sum(g["n"] for g in spec.values())
    print("\ntotal n=%d weighted MAPE %.2f%%  segments %d coefficients %d"
          % (cnt, tot / cnt, len(spec), sum(len(g["coefs"]) for g in spec.values())))
    out = os.path.join(ROOT, "fitted", "SIMT_TAIL_SEGMENTS.json")
    json.dump(spec, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
