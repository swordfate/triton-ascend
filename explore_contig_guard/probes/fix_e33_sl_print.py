#!/usr/bin/env python3
"""Replace the E33SL debug print with a scope-safe version."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)
BS = chr(92)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    old = ('  if (getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr)' + NL +
           '    fprintf(stderr, "E33SL nL=%lld nS=%lld footprint=%lld' +
           BS + 'n",' + NL +
           '            (long long)nLoads, (long long)nStores,' + NL +
           '            (long long)[&]{ int64_t fp=0; for (const '
           'StageMemoryAccess *acc : eligible) fp += '
           'v2ElementCount(acc) * (acc->elementBits/8); return fp; }());')
    new = ('  int64_t dbgFp = 0;' + NL +
           '  for (const StageMemoryAccess *acc : eligible)' + NL +
           '    dbgFp += v2ElementCount(acc) * (acc->elementBits / 8);' +
           NL +
           '  if (getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr)' + NL +
           '    fprintf(stderr, "E33SL n=%lld fp=%lld' + BS + 'n",' + NL +
           '            (long long)eligible.size(), (long long)dbgFp);')
    n = s.count(old)
    assert n == 1, "anchor found %d times" % n
    open(p, "w").write(s.replace(old, new))
    print("E33SL_PRINT_REWRITTEN")


main()
