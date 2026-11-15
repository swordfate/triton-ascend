"""Fit the physical-bottleneck form for SIMT load (formula 4).

    T = max( warp issue , loop unroll , data-path throughput , register spill )

Each branch is a linear expression in one hardware quantity, so every
coefficient has a physical meaning and there are no free-floating product
terms.  int64 needs its own coefficient set because it hits the per-thread
register budget (127 registers x 4 B) and because one warp instruction spans
two 128 B lines.
"""
import json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
# Per-thread register budget (AIV SIMT RegFile 128 KB, 4 B/register):
#   threads <= 256 (W<=8) -> 127 regs -> 508 B
#   <= 512 (W=16) -> 64 -> 256 B,  <= 1024 (W=32) -> 32 -> 128 B,  <= 2048 (W=64) -> 16 -> 64 B
REGB = {1: 508.0, 2: 508.0, 4: 508.0, 8: 508.0, 16: 256.0, 32: 128.0, 64: 64.0}
# 128 B DCache line / (32 lanes x eb bytes): how many warp instructions fill one line
WPL = {1: 4.0, 2: 2.0, 4: 1.0, 8: 1.0}


def ok(r):
    us = r.get("min_us") or {}
    ks = sorted(int(x) for x in us)
    if len(ks) != 2:
        return False
    a, b = us[str(ks[0])], us[str(ks[1])]
    return (r.get("status") == "ok" and r.get("cost_per_operation", 0) > 0 and b >= 5.0
            and abs((b / a) / (ks[1] / ks[0]) - 1) <= 0.25 and r.get("spread", 1) <= 0.10)


def pad(n):
    return 1 << max(0, (n - 1).bit_length())


def load_rows():
    best = {}
    for name in ("raw_npow2_fp.jsonl", "raw_npow2_w14.jsonl"):
        for line in open(os.path.join(ROOT, "raw", name)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not ok(r) or r["direction"] != "load":
                continue
            n, eb = r["elements"], EB[r["dtype"]]
            if pad(n) * eb < 128 or n > 2048:
                continue
            k = (r["dtype"], n, r["warps"])
            if k not in best or r["spread"] < best[k][1]:
                best[k] = (r["cost_per_operation"], r["spread"])
    out = []
    for (dt, n, w), (c, sp) in best.items():
        eb = EB[dt]
        te = n / (32.0 * w)
        out.append(dict(c=c, dt=dt, eb=eb, w=float(w), n=float(n), x=math.ceil(te),
                        lines=n * eb / 128.0, part=1.0 if n % 32 else 0.0,
                        sp1=1.0 if te * eb > REGB[w] / 2 else 0.0,
                        sp2=1.0 if te * eb > REGB[w] else 0.0))
    return out


def branch_values(r):
    """The four physical branches, each as (name, value) with the coefficient to fit."""
    return {
        "warp_issue": r["w"],
        "loop_unroll": 1.0,
        "loop_unroll_x": r["x"],
        "data_path": WPL[r["eb"]] * r["lines"] * r["part"],
        "spill": r["lines"] * r["spill"],
    }


def model(p, r):
    """四个并列瓶颈：warp 发射 / 循环展开 / 数据通路 / 寄存器溢出（两级）。"""
    a, b, c, d, e1, e2 = p
    return max(a * r["w"], b + c * r["x"], d * WPL[r["eb"]] * r["lines"] * r["part"],
               e1 * r["lines"] * r["sp1"], e2 * r["lines"] * r["sp2"])


def mape(p, rs):
    return float(np.mean([abs(model(p, r) - r["c"]) / r["c"] for r in rs]) * 100)


def fit(rs, init):
    p = list(init)
    cur = mape(p, rs)
    for step in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for i in range(len(p)):
                for f in (1 + step, 1 - step):
                    q = list(p)
                    q[i] *= f
                    m = mape(q, rs)
                    if m < cur - 1e-6:
                        p, cur = q, m
                        improved = True
    return p, cur


def main():
    rows = load_rows()
    out = {}
    # int8 / fp16 / fp32 share one formula: their slow path is per-8B-unit priced.
    narrow = [r for r in rows if r["eb"] <= 4]
    best, bp = 1e9, None
    for a0 in (1, 3, 10):
        for b0 in (100, 130, 160):
            for c0 in (5, 9, 15):
                for d0 in (1, 3, 8):
                    p, m = fit(narrow, (a0, b0, c0, d0, 20.0, 60.0))
                    if m < best:
                        best, bp = m, p
    out["tail|load|narrow"] = {"a": bp[0], "b": bp[1], "c": bp[2], "d": bp[3], "e1": bp[4], "e2": bp[5],
                               "mape": best, "n": len(narrow), "dtypes": "int8,fp16,fp32"}
    # int64 spans two lines per warp instruction and can overflow the register budget.
    wide = [r for r in rows if r["eb"] == 8]
    best, bp = 1e9, None
    for a0 in (1, 3, 10):
        for b0 in (100, 150, 200):
            for c0 in (5, 15, 40):
                for d0 in (1, 5, 20):
                    for e0 in (20, 60, 200):
                        p, m = fit(wide, (a0, b0, c0, d0, e0 / 4, e0))
                        if m < best:
                            best, bp = m, p
    out["tail|load|int64"] = {"a": bp[0], "b": bp[1], "c": bp[2], "d": bp[3], "e1": bp[4], "e2": bp[5],
                              "mape": best, "n": len(wide), "dtypes": "int64"}
    tot = sum(v["mape"] * v["n"] for v in out.values()) / sum(v["n"] for v in out.values())
    for k, v in out.items():
        print("%-18s n=%3d MAPE %5.2f%%  a=%.2f b=%.1f c=%.2f d=%.2f e1=%.1f e2=%.1f  (%s)"
              % (k, v["n"], v["mape"], v["a"], v["b"], v["c"], v["d"], v["e1"], v["e2"], v["dtypes"]))
    print("formula 4 (load) total n=%d  MAPE %.2f%%  coefficients %d"
          % (sum(v["n"] for v in out.values()), tot, 6 + 6))
    with open(os.path.join(ROOT, "fitted", "LOAD_BOTTLENECK.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
