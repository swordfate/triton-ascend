#!/usr/bin/env python3
"""Fix the charge sites to use perIterEff (terms were computed but the
charge added the unmodified perIter)."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    fixes = [
        ("        if (allStoresEnv)" + NL +
         "          resources.store += perIter * kCyclesPerNano;" + NL +
         "        else" + NL +
         "          resources.load += perIter * kCyclesPerNano;",
         "        if (allStoresEnv)" + NL +
         "          resources.store += perIterEff * kCyclesPerNano;" + NL +
         "        else" + NL +
         "          resources.load += perIterEff * kCyclesPerNano;"),
        ("        if (loadBytes == 0.0)" + NL +
         "          resources.store += perIter * kCyclesPerNano;" + NL +
         "        else" + NL +
         "          resources.load += perIter * kCyclesPerNano;",
         "        if (loadBytes == 0.0)" + NL +
         "          resources.store += perIterEff * kCyclesPerNano;" + NL +
         "        else" + NL +
         "          resources.load += perIterEff * kCyclesPerNano;"),
    ]
    total = 0
    for old, new in fixes:
        n = s.count(old)
        assert n == 1, "anchor %d for %r" % (n, old[:50])
        s = s.replace(old, new)
        total += n
    open(p, "w").write(s)
    print("CHARGE_SITES_FIXED x%d" % total)


main()
