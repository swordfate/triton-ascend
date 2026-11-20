#!/usr/bin/env python3
"""Remove the E33 trip-family experiment code (unconsumed: the reduction/
generic stage composition overrides it for the intended shapes).  Keeps
fix3, the clean loopfit path, and the env-gated E33PATH/E33SL prints."""
import sys

S = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
H = ("third_party/ascend/costmodel/include/AscendModel/RouteModel/"
     "ContiguousLdStV2CostModel.h")
PP = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
      "SimdSimtCostModel.cpp")
J = ("third_party/ascend/costmodel/profiles/simd_simt/"
     "ascend_950pr_simd_simt_v1.json")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"

    # 1. stage blocks: remove the whole trip-gate block (x2 engines)
    p = "%s/%s" % (root, S)
    s = open(p).read()
    start = "        // E33 reduce/dot trip families: bodies whose every trip ends in"
    end = "          }\n        }\n"
    removed = 0
    while True:
        i = s.find(start)
        if i < 0:
            break
        j = s.find(end, i)
        assert j > 0, "trip gate end not found"
        s = s[:i] + s[j + len(end):]
        removed += 1
    open(p, "w").write(s)
    print("stage trip-gate blocks removed: %d" % removed)
    s = open(p).read()
    assert "tripFitNs" not in s, "tripFitNs still referenced in stage"

    # 2. header: remove tripFitNs helper + tripReduce/tripDot members
    h = "%s/%s" % (root, H)
    s = open(h).read()
    a = s.find("/// E33 trip-table lookup: log2-lerp across grid knots")
    assert a > 0, "helper not found"
    bmarker = "inline double tripFitNs("
    b = s.find(bmarker, a)
    assert b > 0
    # find end of tripFitNs function: the closing "}\n" before the next
    # top-level declaration
    nxt = s.find("\n/// ", b)
    assert nxt > b, "next decl after helper not found"
    s = s[:a] + s[nxt + 1:]
    # members
    mstart = "  // E33 (2026-11-19) reduce/dot trip families: whole-trip per-iteration"
    m = s.find(mstart)
    assert m > 0, "members comment not found"
    mend_anchor = "      tripDot;"
    mend = s.find(mend_anchor, m)
    assert mend > 0
    s = s[:m] + s[mend + len(mend_anchor) + 1:]
    open(h, "w").write(s)
    print("header helper + members removed")

    # 3. parser: remove the trip_reduce/trip_dot parsing loop (x2 engines)
    pp = "%s/%s" % (root, PP)
    s = open(pp).read()
    pstart = '        for (const auto &tfp :'
    pend = '          }\n        }\n'
    removed = 0
    while True:
        i = s.find(pstart)
        if i < 0:
            break
        j = s.find(pend, i)
        assert j > 0, "parser loop end not found"
        s = s[:i] + s[j + len(pend):]
        removed += 1
    open(pp, "w").write(s)
    print("parser trip loops removed: %d" % removed)

    # 4. JSON: drop trip_reduce/trip_dot from both engines
    jf = "%s/%s" % (root, J)
    d = json.load(open(jf))
    for eng in ("simd", "simt"):
        lf = d[eng]["memory"]["contiguous_ldst_v2"]["loop_envelope_fit"]
        lf.pop("trip_reduce", None)
        lf.pop("trip_dot", None)
    open(jf, "w").write(json.dumps(d, indent=1, ensure_ascii=False))
    print("JSON trip keys removed")


import json
main()
