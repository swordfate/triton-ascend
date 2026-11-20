#!/usr/bin/env python3
"""Build trip_reduce / trip_dot tables from E33+E33b and inject them into
the repo profile JSON (both engine scopes).

Table semantics: grid knot ("1"/"56") -> "hot"/"cold" -> per-trip-load
bytes -> per-iteration ns (RED1 = nL=1 reduce trip, DOT1 = nL=2 dot trip).
Bad points (slope <= 0 or None) are imputed as
hot(g) * cold(other-g) / hot(other-g).
"""
import json
import sys
from pathlib import Path

REPO = "/root/work/triton-ascend"
J = (REPO + "/third_party/ascend/costmodel/profiles/simd_simt/"
     "ascend_950pr_simd_simt_v1.json")
ENGINE_KEY = {"simd": "simd", "simt_only": "simt"}
BODIES = {"RED1": "trip_reduce", "DOT1": "trip_dot"}


def load(rows):
    out = {}
    for line in rows:
        r = json.loads(line)
        key = (r["engine"], BODIES.get(r["body"], r["body"]), r["e"],
               r["res"], r["g"])
        out[key] = r["slope_ns"]
    return out


def impute(v):
    return v is None or v <= 0


def main():
    data = load(open("/tmp/e33.jsonl").read().splitlines() +
                open("/tmp/e33b.jsonl").read().splitlines())
    profile = json.load(open(J))

    report = {}
    for eng in ("simd", "simt_only"):
        for body, tname in BODIES.items():
            tab = {}
            for g in (1, 56):
                gtab = {}
                for res in ("hot", "cold"):
                    m = {}
                    tname = BODIES[body]
                    # collect raw values at this (eng, body, g, res)
                    pts = {e * 4: data.get((eng, tname, e, res, g))
                           for e in (128, 512, 2048, 4096)}
                    # impute bad/missing points from the same (eng,body,res)
                    # at the other grid, scaled by that size's hot ratio
                    for b, v in pts.items():
                        if impute(v):
                            other = "cold" if res == "hot" else "hot"
                            og = 56 if g == 1 else 1
                            v_hot = data.get((eng, tname, int(b / 4), "hot",
                                              g))
                            v_other = data.get((eng, tname, int(b / 4), other,
                                                og))
                            v_hot_o = data.get((eng, tname, int(b / 4), "hot",
                                                og))
                            if not impute(v_hot) and not impute(v_other) \
                                    and not impute(v_hot_o) and v_hot_o:
                                v = v_hot * v_other / v_hot_o
                    pts = {b: round(v, 1) for b, v in pts.items()
                           if not impute(v)}
                    if pts:
                        gtab[res] = {str(b): v for b, v in sorted(pts.items())}
                if gtab:
                    tab[str(g)] = gtab
            profile[ENGINE_KEY[eng]]["memory"]["contiguous_ldst_v2"][
                "loop_envelope_fit"][tname] = tab
            report[(eng, tname)] = tab

    Path(J).write_text(json.dumps(profile, indent=1, ensure_ascii=False))
    for k, v in report.items():
        print(k, json.dumps(v))
    print("INJECTED ->", J)


main()
