#!/usr/bin/env python3
"""Add E33d structural terms to the two loop-fit charge sites."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")

TERMS = """        // E33d structural terms: real loop bodies carry runtime masks
        // (+predication setup per trip) and, once the launch spans a
        // full wave, an inter-trip address jump (grid-stride windows
        // lose L2 locality).  Both measured on the memory_stream shape.
        bool bodyRuntimeMasked = false;
        for (const StageMemoryAccess *access : eligible)
          if (access->masked)
            bodyRuntimeMasked = true;
        double perIterEff = perIter;
        if (bodyRuntimeMasked)
          perIterEff += itf->second.maskTripNs;
        if (programCountHint >= itf->second.strideTripMinGrid &&
            stage.iterationCount >= 2)
          perIterEff += itf->second.strideTripNs;
"""

SITES = [
    ("simd", """        if (loadBytes == 0.0)
          resources.store += perIter * kCyclesPerNano;
        else
          resources.load += perIter * kCyclesPerNano;
"""),
    ("simt", """        if (allStoresEnv)
          resources.store += perIter * kCyclesPerNano;
        else
          resources.load += perIter * kCyclesPerNano;
"""),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    for tag, old in SITES:
        found = s.count(old)
        assert found == 1, "%s site anchor %d, want 1" % (tag, found)
        s = s.replace(old, TERMS + old)
    open(p, "w").write(s)
    print("TERMS_ADDED x2")


main()
