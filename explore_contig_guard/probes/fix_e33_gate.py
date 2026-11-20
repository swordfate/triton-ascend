#!/usr/bin/env python3
"""Relax the E33 trip gate (allow nStores==0) and log load/store bytes."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)
BS = chr(92)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()

    old_gate = "if (tripRed && tripTiny && (nLoads == 1 || nLoads == 2)) {"
    new_gate = "if (tripRed && (tripTiny || nStores == 0) && (nLoads == 1 || nLoads == 2)) {"
    assert s.count(old_gate) == 2, "gate anchor %d" % s.count(old_gate)
    s = s.replace(old_gate, new_gate)

    old_fmt = ('"E33GATE simd=%d tripRed=%d tripTiny=%d nLoads=%d '
               'nStores=%d redOps=%f tabR=%zu tabD=%zu' + BS + 'n",')
    new_fmt = ('"E33GATE simd=%d tripRed=%d tripTiny=%d nLoads=%d '
               'nStores=%d lB=%lld sB=%lld redOps=%f tabR=%zu tabD=%zu' +
               BS + 'n",')
    assert s.count(old_fmt) == 2, "fmt anchor %d" % s.count(old_fmt)
    s = s.replace(old_fmt, new_fmt)

    old_args = """                  (int)simd, (int)tripRed, (int)tripTiny, (int)nLoads,
                  (int)nStores, work.reductionOperations,"""
    new_args = """                  (int)simd, (int)tripRed, (int)tripTiny, (int)nLoads,
                  (int)nStores, (long long)loadBytes, (long long)storeBytes,
                  work.reductionOperations,"""
    assert s.count(old_args) == 2, "args anchor %d" % s.count(old_args)
    s = s.replace(old_args, new_args)

    open(p, "w").write(s)
    print("GATE_RELAXED_AND_LOGGED")


main()
