#!/usr/bin/env python3
"""Fix mangled newlines in the E33PATH/E33SL debug fprintf statements."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    fixes = [
        ('      fprintf(stderr, "E33PATH simd=%d iter=%lld lB=%lld sB=%lld '
         'nL=%lld nS=%lld' + NL + '",',
         '      fprintf(stderr, "E33PATH simd=%d iter=%lld lB=%lld sB=%lld '
         'nL=%lld nS=%lld' + chr(92) + 'n",'),
        ('    fprintf(stderr, "E33SL nL=%lld nS=%lld footprint=%lld' + NL +
         '",',
         '    fprintf(stderr, "E33SL nL=%lld nS=%lld footprint=%lld' +
         chr(92) + 'n",'),
    ]
    total = 0
    for old, new in fixes:
        n = s.count(old)
        assert n >= 1, "anchor not found: %r" % old[:60]
        s = s.replace(old, new)
        total += n
    open(p, "w").write(s)
    print("FIXED %d fprintf newlines" % total)


main()
