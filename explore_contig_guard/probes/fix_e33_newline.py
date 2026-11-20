#!/usr/bin/env python3
"""Fix the mangled fprintf newline in the E33 gate debug print."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
OLD = ('          fprintf(stderr, "E33GATE simd=%d tripRed=%d tripTiny=%d '
       'nLoads=%d nStores=%d redOps=%f tabR=%zu tabD=%zu\n",')
NEW = ('          fprintf(stderr, "E33GATE simd=%d tripRed=%d tripTiny=%d '
       'nLoads=%d nStores=%d redOps=%f tabR=%zu tabD=%zu\\n",')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    n = s.count(OLD)
    assert n == 2, "found %d mangled fprintf lines, want 2" % n
    open(p, "w").write(s.replace(OLD, NEW))
    print("FIXED_NEWLINES x%d" % n)


main()
