#!/usr/bin/env python3
"""Compare two collect100.py TSVs: prediction deltas + route flips.

Usage: python3 compare100.py <BEFORE_TSV> <AFTER_TSV> <OUT_MD> [<LABEL>]
"""
import sys
from pathlib import Path


def load(p):
    rows = {}
    with open(p) as h:
        next(h)
        for line in h:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2]:
                rows[parts[0]] = (parts[1], float(parts[2]))
            else:
                rows[parts[0]] = (parts[1] if len(parts) > 1 else "?", None)
    return rows


def main():
    before = load(sys.argv[1])
    after = load(sys.argv[2])
    out = Path(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else "fix3"
    both = sorted(set(before) & set(after))
    flips, deltas, worse, better, evaluable = [], [], [], [], 0
    for name in both:
        rb, pb = before[name]
        ra, pa = after[name]
        if pa is None or pb is None:
            continue
        evaluable += 1
        d = (pa - pb) / pb * 100.0
        deltas.append((name, rb, pb, ra, pa, d))
        if d > 0.01:
            worse.append(d)
        elif d < -0.01:
            better.append(d)
        if rb != ra:
            flips.append((name, rb, pb, ra, pa, d))
    deltas.sort(key=lambda r: -abs(r[5]))
    with open(out, "w") as h:
        h.write("# fix3 对比：%s\n\n" % label)
        h.write("- 可评 case：%d（before=%d, after=%d）\n" %
                (evaluable, len(before), len(after)))
        h.write("- 预测值变化的 case：%d；route 翻转：%d\n\n" %
                (sum(1 for d in deltas if abs(d[5]) > 0.01), len(flips)))
        h.write("## route 翻转\n\n| case | route前 | 预测前(us) | route后 | 预测后(us) | Δ |\n|---|---|---:|---|---:|---:|\n")
        for name, rb, pb, ra, pa, d in flips:
            h.write("| %s | %s | %.3f | %s | %.3f | %+.1f%% |\n" %
                    (name, rb, pb, ra, pa, d))
        h.write("\n## 预测变化 top（|Δ| 降序）\n\n| case | route前→后 | 预测前(us) | 预测后(us) | Δ |\n|---|---|---:|---:|---:|\n")
        for name, rb, pb, ra, pa, d in deltas[:40]:
            if abs(d) <= 0.01:
                break
            h.write("| %s | %s→%s | %.3f | %.3f | %+.1f%% |\n" %
                    (name, rb, ra, pb, pa, d))
    print("evaluable=%d flips=%d changed=%d -> %s" %
          (evaluable, len(flips),
           sum(1 for d in deltas if abs(d[5]) > 0.01), out))


main()
