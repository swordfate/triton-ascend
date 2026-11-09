"""把格点实测（min-of-N 已在上游完成）整理成 fitted/TABLES.json。"""
import json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "load.eb2": ["lat5.jsonl"],
    "load.eb4": ["lat5f32.jsonl"],
    "store.eb2": ["lat5s2.jsonl"],
    "store.eb4": ["lat5s4.jsonl"],
}
EB = {"load.eb2": 2, "load.eb4": 4, "store.eb2": 2, "store.eb4": 4}
SCRATCH = ROOT / "lattice_raw"

def main():
    out = {}
    for key, files in MANIFESTS.items():
        eb = EB[key]
        best = {}
        found = False
        for name in files:
            p = SCRATCH / name
            if not p.exists():
                continue
            found = True
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "ok" or not r.get("cost_raw"):
                    continue
                by = r["cols"] * eb
                te = r["te"]
                k = (te, by)
                if k not in best or r["cost_raw"] < best[k]:
                    best[k] = r["cost_raw"]
        if not found or not best:
            print("skip %s (no data)" % key)
            continue
        # 只保留至少有两个字节数的 te 行；单点行无法插值补齐，直接不设该 knot。
        counts = {}
        for (te_k, by_k) in best:
            counts.setdefault(te_k, set()).add(by_k)
        tes = sorted(te for te, s in counts.items() if len(s) >= 2)
        if len(tes) < 2:
            print("skip %s (fewer than two usable te rows)" % key)
            continue
        bys = sorted({k[1] for k in best})
        # 缺失格点在行内沿 by 做 log-log 线性插值补齐；整行缺失才放弃。
        grid = []
        complete = True
        for te in tes:
            row = []
            known = [(by, best[(te, by)]) for by in bys if (te, by) in best]
            if len(known) < 2:
                complete = False
                break
            for by in bys:
                if (te, by) in best:
                    row.append(round(best[(te, by)], 3))
                    continue
                lo = max((k for k in known if k[0] < by), key=lambda k: k[0], default=None)
                hi = min((k for k in known if k[0] > by), key=lambda k: k[0], default=None)
                if lo is None or hi is None:
                    # 落在已知区间之外：用最近两点在 log-log 上外推，
                    # 保证网格矩形（外推格点不会被查询命中，只用于补齐）。
                    if len(known) < 2:
                        complete = False
                        break
                    known_sorted = sorted(known)
                    lo, hi = (known_sorted[0], known_sorted[1]) if lo is None else (known_sorted[-2], known_sorted[-1])
                tx = (math.log(by) - math.log(lo[0])) / (math.log(hi[0]) - math.log(lo[0]))
                v = math.exp(math.log(lo[1]) * (1 - tx) + math.log(hi[1]) * tx)
                row.append(round(v, 3))
            if not complete:
                break
            grid.append(row)
        if not complete:
            print("skip %s (grid not fillable: %d te x %d by)" % (key, len(tes), len(bys)))
            continue
        out[key] = {"te_knots": tes, "by_knots": bys, "cost_ns": grid}
        print("%-10s %d te x %d by" % (key, len(tes), len(bys)))
    target = ROOT / "fitted" / "TABLES.json"
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote %s with %d tables" % (target, len(out)))

if __name__ == "__main__":
    main()
