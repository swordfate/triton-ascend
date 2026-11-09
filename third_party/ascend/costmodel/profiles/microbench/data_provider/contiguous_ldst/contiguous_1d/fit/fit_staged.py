"""Staged fit: physical quantities first, non-physical terms only if needed.

For every formula the feature pool is tried in two stages:
  stage 1 - only quantities that name a hardware process
  stage 2 - if the error is still above 10%, add the best remaining term
"""
import json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
REGB = {1: 508.0, 2: 508.0, 4: 508.0, 8: 508.0, 16: 256.0, 32: 128.0, 64: 64.0}
PHYS = ["by", "lines", "rounds", "W", "invW", "warpB", "fill", "part", "part_by",
        "slowK", "slowK_by", "s32", "lin", "spillL", "wpl_by", "lanes"]
EXTRA = ["LN", "by2", "lnW", "byLnW", "K1", "K2", "i8K", "Wh_by", "wh8_by", "W2", "waves_by"]


def kof(x, cap=2):
    return 2.0 ** min(cap, math.floor(math.log2(max(1.0, x))))


def feats(n, eb, w):
    by = n * eb
    lines = by / 128.0
    te = n / (32.0 * w)
    slowK = 1.0 if (n % 32 and te >= 8.0 / eb) else 0.0
    part = 1.0 if by % 128 else 0.0
    return dict(by=by, lines=lines, te=te, rounds=math.ceil(te), W=float(w), invW=1.0 / w,
                warpB=32.0 * eb, fill=(32.0 * eb) / 256.0, part=part, part_by=by * part,
                slowK=slowK, slowK_by=by * slowK, s32=1.0 if n < 32 else 0.0,
                lin=1.0 if by >= 128 else 0.0,
                spillL=lines * (1.0 if te * eb > REGB[w] else 0.0),
                wpl_by=by * max(1.0, 128.0 / (32.0 * eb)),
                lanes=min(n, 32.0 * w),
                LN=by * math.log(1 + te), by2=by * by, lnW=math.log(1 + w),
                byLnW=by * math.log(1 + w), K1=by * kof(lines / w),
                K2=by * 2.0 ** min(2, math.floor(math.log2(max(1.0, te / 16)))),
                i8K=(by if eb == 1 else 0.0) * kof(lines / w),
                Wh_by=by * max(0.0, w - 16.0), wh8_by=by * max(0.0, w - 8.0),
                W2=float(w) ** 2, waves_by=math.ceil(te) * by)


def aligned():
    rows = []
    for line in open(os.path.join(ROOT, "raw", "raw_pure_unified.jsonl")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["mode"] == "simt_only" and r["el"] > (2048 if r["dir"] == "load" else 8192):
            continue
        if r["mode"] == "simd" and r["el"] > (8192 if r["dir"] == "load" else 16384):
            continue
        rows.append(dict(c=r["cost"], mode=r["mode"], d=r["dir"], eb=r["eb"], w=r["W"],
                         **feats(r["el"], r["eb"], r["W"])))
    return rows


def tail():
    best = {}
    for name in ("raw_npow2_fp.jsonl", "raw_npow2_w14.jsonl"):
        for line in open(os.path.join(ROOT, "raw", name)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            us = r.get("min_us") or {}
            ks = sorted(int(x) for x in us)
            if len(ks) != 2:
                continue
            a, b = us[str(ks[0])], us[str(ks[1])]
            if not (r.get("status") == "ok" and r.get("cost_per_operation", 0) > 0 and b >= 5.0
                    and abs((b / a) / (ks[1] / ks[0]) - 1) <= 0.25 and r.get("spread", 1) <= 0.10):
                continue
            n, eb = r["elements"], EB[r["dtype"]]
            if (1 << max(0, (n - 1).bit_length())) * eb < 128:
                continue
            if n > (2048 if r["direction"] == "load" else 8192):
                continue
            k = (r["dtype"], r["direction"], n, r["warps"])
            if k not in best or r["spread"] < best[k][1]:
                best[k] = (r["cost_per_operation"], r["spread"])
    rows = []
    for (dt, d, n, w), (c, sp) in best.items():
        eb = EB[dt]
        rows.append(dict(c=c, d=d, eb=eb, w=w, n=n, **feats(n, eb, w)))
    return rows


def fit(rs, names):
    X = np.stack([np.ones(len(rs))] + [np.array([r[f] for r in rs], float) for f in names], 1)
    y = np.array([r["c"] for r in rs], float)
    wt = 1.0 / y
    co = np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]
    return co, float((np.abs(X @ co - y) / y * 100).mean())


def greedy(rs, pool, budget):
    chosen, co, m = [], *fit(rs, [])
    while len(chosen) < budget:
        pick = None
        for nm in pool:
            if nm in chosen:
                continue
            c2, m2 = fit(rs, chosen + [nm])
            if pick is None or m2 < pick[0]:
                pick = (m2, nm, c2)
        if pick is None or pick[0] > m * 0.98:
            break
        chosen.append(pick[1])
        m, co = pick[0], pick[2]
    return chosen, co, m


def stage(rs, budget=5):
    ch, co, m = greedy(rs, PHYS, budget)
    if m < 10.0:
        return ch, co, m, 1
    used = list(ch)
    while m >= 10.0 and len(used) < budget + 2:
        pick = None
        for nm in EXTRA:
            if nm in used:
                continue
            c2, m2 = fit(rs, used + [nm])
            if pick is None or m2 < pick[0]:
                pick = (m2, nm, c2)
        if pick is None or pick[0] >= m * 0.98:
            break
        used.append(pick[1])
        m, co = pick[0], pick[2]
    return used, co, m, 2


def show(name, rs, budget=5):
    ch, co, m, st = stage(rs, budget)
    print("%-26s n=%3d  MAPE %5.2f%%  阶段%d  %s" % (name, len(rs), m, st, ",".join(ch)))
    print("%-26s 系数 %s" % ("", " ".join("%s=%.4g" % (f, c) for f, c in zip(["1"] + ch, co))))
    return dict(n=len(rs), mape=round(m, 2), stage=st, feats=ch, coefs=[round(float(c), 6) for c in co])


def main():
    A = aligned()
    T = tail()
    out = {}
    out["simd.load"] = show("① simd load", [r for r in A if r["mode"] == "simd" and r["d"] == "load"])
    out["simd.store"] = show("② simd store", [r for r in A if r["mode"] == "simd" and r["d"] == "store"])
    for k, sel in (("lines<=2", lambda r: r["lines"] <= 2), ("2<lines<=32", lambda r: 2 < r["lines"] <= 32),
                   ("l>32,W<=4", lambda r: r["lines"] > 32 and r["w"] <= 4),
                   ("l>32,W>4", lambda r: r["lines"] > 32 and r["w"] > 4)):
        out["simt.load|" + k] = show("③ load " + k,
                                     [r for r in A if r["mode"] == "simt_only" and r["d"] == "load" and sel(r)])
    print("\n④ load 用瓶颈形式（fit_bottleneck.py）；⑤⑥ store 的物理量版 >10%，仍用分段")
    with open(os.path.join(ROOT, "fitted", "STAGED_FIT.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
