"""Turn E1/E3 JSONL into readable tables (run anywhere with python3).

E1: per-load marginal = diff_med_ns / K  ->  shows
  * single-shot vs K==1 loop (Q1),
  * amortization curve over K (Q2),
  * warm vs cold gap (Q4 L2 hypothesis).

E3: incremental per-load cost = (diff(U) - diff(U=1)) / (U - 1)  ->  shows
  whether extra same-shape loads overlap (Q3).

Usage: python3 analyze_e1_e3.py results/e1_fp32_c2.jsonl [results/e3_c3.jsonl ...]
"""
import json
import sys
from collections import defaultdict


def load(paths):
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    recs = list(load(sys.argv[1:]))
    e1 = [r for r in recs if r.get("exp") == "e1"]
    e3 = [r for r in recs if r.get("exp") == "e3"]

    if e1:
        print("=" * 100)
        print("E1  per-load marginal (ns) = diff_med_ns / K;  'abs' = kernel med_us")
        for mode in sorted({r["mode"] for r in e1}):
            print("\n-- mode=%s" % mode)
            print("%-14s %6s" % ("window\\K", " ".join("%7d" % k for k in
                                                      [1, 2, 4, 8, 16, 32, 64, 128])))
            bywin = defaultdict(dict)
            ss = {}
            for r in e1:
                if r["mode"] != mode or r.get("diff") is None:
                    continue
                k = r.get("K", 1)
                d = r["diff"]["diff_med_ns"]
                if r["tag"].startswith("ss_"):
                    ss[r["tag"]] = (d, r["probe"]["med_us"], r["ctl"]["med_us"])
                    continue
                per = d / k
                bywin[r["window"]][k] = per
            for win in ("same", "warm", "cold"):
                row = bywin.get(win, {})
                cells = " ".join("%7.1f" % row[k] if k in row else "      -"
                                 for k in [1, 2, 4, 8, 16, 32, 64, 128])
                print("%-14s %6s" % (win, cells))
            print("single-shot (loop-free):")
            for tag, (d, pa, pc) in sorted(ss.items()):
                print("  %-10s diff=%7.1f ns   probe_med=%7.2f us   ctl_med=%6.2f us"
                      % (tag, d, pa, pc))
            print("kind store rows are in the JSONL (same layout, 'store_' tags).")

    if e3:
        print("=" * 100)
        print("E3  incremental per-load cost: (diff(U)-diff(1))/(U-1)  [ns]; "
              "diff(U) itself in brackets")
        by = defaultdict(dict)
        for r in e3:
            if r.get("diff") is None:
                continue
            key = (r["mode"], r.get("kind"), r.get("chain"), r.get("grid"))
            by[key][r["U"]] = r["diff"]["diff_med_ns"]
        for (mode, kind, chain, grid), m in sorted(by.items(), key=str):
            label = "chain" if chain else "ind  "
            line = ["U=%d: " % u + "%7.1f" % m[u] if u in m else "      -"
                    for u in (1, 2, 3, 4, 6, 8)]
            inc = []
            us = sorted(m)
            for a, b in zip(us, us[1:]):
                inc.append((m[b] - m[a]) / (b - a))
            print("mode=%-9s kind=%-5s %s grid=%-2d | diff %s | incr %s" % (
                mode, kind, label, grid, " ".join(line),
                " ".join("%.0f" % x for x in inc)))


if __name__ == "__main__":
    main()
