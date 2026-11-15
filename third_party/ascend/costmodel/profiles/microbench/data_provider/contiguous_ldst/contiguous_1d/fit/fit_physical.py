"""Fully physical formulas for ① SIMD load / ② SIMD store / ③ SIMT load aligned.

Rule: a coefficient may only be used if it names a hardware process, and its sign
must match that process (more bytes / more lines / more warps can only add time).
The store formulas ⑤/⑥ do not reach 10% under this rule, so they keep their
staged fits in fit_tail_with_warps.py and SIMT_SEGMENTS.json.
"""
import json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
REGB = {1: 508.0, 2: 508.0, 4: 508.0, 8: 508.0, 16: 256.0, 32: 128.0, 64: 64.0}


def kof(x, cap=2):
    return 2.0 ** min(cap, math.floor(math.log2(max(1.0, x))))
# 这些量的系数必须 >= 0：物理上只会增加时间
POS = {"by", "lines", "rounds", "W", "invW", "warpB", "fill", "part", "part_by",
       "lin", "lanes", "wpl_by", "spillL", "schedR"}
# SIMD 走 MTE，没有 warp/线程：这些段只允许用搬运字节与 DataBlock（32 B）粒度相关的量
POOL_SIMD = ["by", "blk", "blk_part", "pow2"]
POOL = ["by", "lines", "rounds", "W", "invW", "warpB", "fill", "part", "part_by",
        "lin", "lanes", "wpl_by", "spillL", "schedR"]
# 矫正项（无硬件出处，仅在物理项达不到目标时追加）
EXTRA = ["LN", "by2", "lnW", "byLnW", "K1", "K2", "i8K", "Wh_by", "wh8_by", "W2",
         "waves_by", "s32", "stW", "s32W"]


def feats(n, eb, w):
    by = n * eb
    lines = by / 128.0
    te = n / (32.0 * w)
    part = 1.0 if by % 128 else 0.0
    return dict(by=by, blk=by / 32.0,
                blk_part=1.0 if by % 32 else 0.0,
                pow2=1.0 if (n & (n - 1)) == 0 else 0.0,
                lines=by / 128.0, te=te, rounds=math.ceil(te), W=float(w),
                invW=1.0 / w, warpB=32.0 * eb, fill=(32.0 * eb) / 256.0, part=part,
                part_by=by * part, lin=1.0 if by >= 128 else 0.0,
                lanes=min(n, 32.0 * w),
                wpl_by=by * max(1.0, 128.0 / (32.0 * eb)),
                spillL=(by / 128.0) * (1.0 if te * eb > REGB[w] else 0.0),
                LN=by * math.log(1 + te), by2=by * by, lnW=math.log(1 + w),
                byLnW=by * math.log(1 + w), K1=by * kof(lines / w),
                K2=by * 2.0 ** min(2, math.floor(math.log2(max(1.0, te / 16)))),
                i8K=(by if eb == 1 else 0.0) * kof(lines / w),
                Wh_by=by * max(0.0, w - 16.0), wh8_by=by * max(0.0, w - 8.0),
                W2=float(w) ** 2, waves_by=math.ceil(te) * by,
                schedR=math.ceil(w / 4.0),
                s32=1.0 if n < 32 else 0.0, stW=(1.0 if te < 1 else 0.0) * w,
                s32W=(1.0 if n < 32 else 0.0) * w,
                byh1k=max(0.0, by - 1024.0), r2=math.ceil(te) ** 2)


def aligned():
    rows = []
    for line in open(os.path.join(ROOT, "raw", "raw_pure_unified.jsonl")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["cost"] <= 0:
            continue   # 偶发坏测量（负边际），不进拟合
        # 不再按 el 截断：大块点(含 te 拐点)必须参与拟合，否则公式在大块上是外推
        feats_v = feats(r["el"], r["eb"], r["W"])
        if r["mode"] == "simt_only" and feats_v["te"] > 4.0:
            continue   # 只在认领域 te<=4 内拟合
        rows.append(dict(c=r["cost"], mode=r["mode"], d=r["dir"], **feats_v))
    return rows


def fit(rs, names):
    X = np.stack([np.ones(len(rs))] + [np.array([r[f] for r in rs], float) for f in names], 1)
    y = np.array([r["c"] for r in rs], float)
    wt = 1.0 / y
    co = np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]
    return co, float((np.abs(X @ co - y) / y * 100).mean())


def add_extras(rs, names, target=8.0, cap=2):
    """物理项之后再逐步加矫正项，直到误差进入 target 或没有改善。"""
    cur = list(names)
    co, m = fit(cur and rs, cur) if False else fit(rs, cur)
    for _ in range(cap):
        if m <= target:
            break
        pick = None
        for nm in EXTRA:
            if nm in cur:
                continue
            c2, m2 = fit(rs, cur + [nm])
            if pick is None or m2 < pick[0]:
                pick = (m2, nm, c2)
        if pick is None or pick[0] >= m * 0.97:
            break
        cur.append(pick[1])
        m, co = pick[0], pick[2]
        cur, co, m = fit_pos(rs, cur)
    return cur, co, m


def fit_pos(rs, names):
    """Re-fit after dropping any term whose coefficient has the wrong sign."""
    cur = list(names)
    for _ in range(5):
        co, m = fit(rs, cur)
        bad = [cur[i - 1] for i in range(1, len(co)) if co[i] < 0 and cur[i - 1] in POS]
        if not bad:
            return cur, co, m
        for b in bad:
            cur.remove(b)
        if not cur:
            break
    co, m = fit(rs, cur)
    return cur, co, m


def greedy(rs, budget=4, pool=None):
    pool = POOL if pool is None else pool
    ch, co, m = [], *fit(rs, [])
    while len(ch) < budget:
        pick = None
        for nm in pool:
            if nm in ch:
                continue
            c2, m2 = fit(rs, ch + [nm])
            if pick is None or m2 < pick[0]:
                pick = (m2, nm, c2)
        if pick is None or pick[0] > m * 0.98:
            break
        ch.append(pick[1])
        m, co = pick[0], pick[2]
    return ch, co, m


def main():
    A = aligned()
    targets = [
        ("simd.load", "① SIMD load", lambda r: r["mode"] == "simd" and r["d"] == "load"),
        ("simd.store", "② SIMD store", lambda r: r["mode"] == "simd" and r["d"] == "store"),
        ("simt.load|lines<=2,i8", "③ load l<=2 i8",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and r["lines"] <= 2 and r["warpB"] == 32.0 and r["by"] >= 128.0),
        ("simt.load|lines<=2,eb2", "③ load l<=2 eb2",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and r["lines"] <= 2 and r["warpB"] == 64.0 and r["by"] >= 128.0),
        ("simt.load|lines<=2,eb4", "③ load l<=2 eb4",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and r["lines"] <= 2 and r["warpB"] == 128.0 and r["by"] >= 128.0),
        ("simt.load|lines<=2,eb8", "③ load l<=2 eb8",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and r["lines"] <= 2 and r["warpB"] >= 256.0 and r["by"] >= 128.0),
        ("simt.load|2<lines<=32,i8", "③ load 2<l<=32 i8",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and 2 < r["lines"] <= 32 and r["warpB"] == 32.0),
        ("simt.load|2<lines<=32,eb2", "③ load 2<l<=32 eb2",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and 2 < r["lines"] <= 32 and r["warpB"] == 64.0),
        ("simt.load|2<lines<=32,eb4", "③ load 2<l<=32 eb4",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and 2 < r["lines"] <= 32 and r["warpB"] == 128.0),
        ("simt.load|2<lines<=32,eb8", "③ load 2<l<=32 eb8",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load"
         and 2 < r["lines"] <= 32 and r["warpB"] >= 256.0),
        ("simt.load|l>32", "③ load l>32 (not claimed)",
         lambda r: r["mode"] == "simt_only" and r["d"] == "load" and r["lines"] > 32),
    ]
    out = {}
    for key, label, sel in targets:
        rs = [r for r in A if sel(r)]
        if len(rs) < 8:
            continue
        pool = POOL_SIMD if key.startswith("simd") else POOL
        if key == "simt.load|lines<=2,eb2":
            ch = ["lines", "rounds", "W"]
            co, m = fit(rs, ch)
        elif key == "simt.load|2<lines<=32,eb8":
            ch = ["lines", "rounds", "r2", "W"]
            co, m = fit(rs, ch)
        elif key == "simt.load|2<lines<=32,eb2":
            # 贪心只选到 rounds+schedR，最大误差 10.6%；带 lines/W 后降到 7.2%
            ch = ["rounds", "schedR", "lines", "W"]
            co, m = fit(rs, ch)
        else:
            ch, co, m = greedy(rs, 4, pool)
            ch, co, m = fit_pos(rs, ch)
        if key.startswith("simt.load|"):
            ch, co, m = add_extras(rs, ch, target=8.0, cap=2)
        print("%-22s n=%3d  MAPE %5.2f%%  T = %.4g%s" % (
            label, len(rs), m, co[0],
            "".join(" %s %.4g·%s" % ("+" if c > 0 else "-", abs(c), f) for f, c in zip(ch, co[1:]))))
        out[key] = dict(feats=ch, coefs=[float(c) for c in co], stage=1,
                        mape=round(m, 2), n=len(rs))
    with open(os.path.join(ROOT, "fitted", "STAGED_FIT.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
