#!/usr/bin/env python3
"""fit_direct2.py - build the calF2 direct table and inject into profiles.

Rows: per (engine, nL, nS, chunkBytes, K, grid) keep the best (min) whole
ns_prog across sessions/cards after the cv filter.  The C++ Formula B
interpolates T(K) = A + K*m across the K rows of one geometry.

Writes:
  results/direct_table2.json           (audit copy)
  repo profile      contiguous_ldst_v2.direct = {engine: {key: ns}}
  installed profile same injection
  contiguous_ldst_v2.direct_fixed_share = 0.3 (initial; tune on guards)
"""
import json
import re
import collections
from pathlib import Path

RES = Path("/root/work/explore_contig/results")
REPO = Path("/root/work/triton-ascend")
SRC_PROF = REPO / "third_party/ascend/costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json"
INST_PROF = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json")
FIXED_SHARE = 0.3


def main():
    import sys
    inputs = sys.argv[1:] or [str(RES / "calF2_card2.jsonl")]
    best = {}
    for fn in inputs:
        p = Path(fn)
        if not p.is_file():
            print("skip missing", fn)
            continue
        for line in p.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "cf2":
                continue
            if r.get("ns_prog", 0) <= 0:
                continue
            eng = "simd" if r["mode"] == "simd" else "simt"
            key = "n%d_s%d_b%d_k%d_g%d" % (r["nl"], r["nst"], r["bytes"],
                                           r["k"], r["grid"])
            cur = best.get((eng, key))
            if cur is None or r["ns_prog"] < cur:
                best[(eng, key)] = r["ns_prog"]

    table = collections.defaultdict(dict)
    for (eng, key), v in sorted(best.items()):
        table[eng][key] = round(v, 1)
    out = RES / "direct_table2.json"
    json.dump(table, open(out, "w"), indent=1)
    print("rows per engine:", {e: len(t) for e, t in table.items()})

    # coverage audit: which (nL,nS,b,g) have all three K
    audit = collections.Counter()
    for eng, t in table.items():
        for key in t:
            m = re.match(r"n(\d+)_s(\d+)_b(\d+)_k(\d+)_g(\d+)", key)
            audit[(eng, m.group(1), m.group(2), m.group(3), m.group(5))] += 1
    incomplete = {k: c for k, c in audit.items() if c < 3}
    print("geometries with <3 K rows:", incomplete if incomplete else "none")

    for prof_path in (SRC_PROF, INST_PROF):
        prof = json.load(open(prof_path))
        for eng in ("simd", "simt"):
            v2 = prof[eng].setdefault("memory", {}).setdefault(
                "contiguous_ldst_v2", {})
            v2["direct"] = table.get(eng, {})
            v2["direct_fixed_share"] = FIXED_SHARE
        json.dump(prof, open(prof_path, "w"), indent=1)
        print("injected", prof_path)


if __name__ == "__main__":
    main()
