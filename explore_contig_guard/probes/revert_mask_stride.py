#!/usr/bin/env python3
"""Revert the E33d mask/stride structural terms (they over-apply:
13 regressions in the full 100-case).  Keeps the E33 trip families
(trip_reduce/trip_dot) and the debug prints."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
H = ("third_party/ascend/costmodel/include/AscendModel/RouteModel/"
     "ContiguousLdStV2CostModel.h")
PP = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
      "SimdSimtCostModel.cpp")
NL = chr(10)
BS = chr(92)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"

    # 1. stage blocks: drop the TERMS block, keep the charge with perIter
    p = "%s/%s" % (root, P)
    s = open(p).read()
    terms_start = "        // E33d structural terms: real loop bodies carry runtime masks"
    end_anchor = "        claimAll();" + NL + "        return;" + NL
    removed = 0
    while True:
        i = s.find(terms_start)
        if i < 0:
            break
        j = s.find(end_anchor, i)
        assert j > 0, "charge end not found"
        block = s[i:j]
        if "if (loadBytes == 0.0)" in block:
            charge = ("        if (loadBytes == 0.0)" + NL +
                      "          resources.store += perIter * kCyclesPerNano;" +
                      NL + "        else" + NL +
                      "          resources.load += perIter * "
                      "kCyclesPerNano;" + NL)
        elif "if (allStoresEnv)" in block:
            charge = ("        if (allStoresEnv)" + NL +
                      "          resources.store += perIter * kCyclesPerNano;" +
                      NL + "        else" + NL +
                      "          resources.load += perIter * "
                      "kCyclesPerNano;" + NL)
        else:
            raise AssertionError("no charge condition found in TERMS block")
        s = s[:i] + charge + s[j:]
        removed += 1
    open(p, "w").write(s)
    print("stage TERMS blocks reverted: %d" % removed)
    s = open(p).read()
    assert "perIterEff" not in s, "perIterEff still present"
    assert "maskTripNs" not in s, "maskTripNs still referenced"

    # 2. header: drop the three mask/stride members
    h = "%s/%s" % (root, H)
    s = open(h).read()
    old = """  // E33d (2026-11-19) structural terms for real (masked grid-stride)
  // loop bodies, measured on the memory_stream_l4 shape at grid 56:
  //   runtime-mask predication ~ +11 ns/trip, inter-trip address jump
  //   ~ +38 ns/trip once the launch spans >= stride_trip_min_grid.
  double maskTripNs = 0.0;
  double strideTripNs = 0.0;
  int64_t strideTripMinGrid = 56;
"""
    n = s.count(old)
    assert n == 1, "header members anchor %d" % n
    s = s.replace(old, "")
    open(h, "w").write(s)
    print("header members removed")

    # 3. parser: drop the three reads
    pp = "%s/%s" % (root, PP)
    s = open(pp).read()
    old = """        cv2.loopFit.maskTripNs =
            reader.optionalNumber(*lf, "mask_trip_ns", 0.0);
        cv2.loopFit.strideTripNs =
            reader.optionalNumber(*lf, "stride_trip_ns", 0.0);
        cv2.loopFit.strideTripMinGrid = static_cast<int64_t>(
            reader.optionalNumber(*lf, "stride_trip_min_grid", 56.0));"""
    n = s.count(old)
    assert n == 2, "parser anchor %d" % n
    s = s.replace(old, "")
    open(pp, "w").write(s)
    print("parser reads removed x2")


main()
