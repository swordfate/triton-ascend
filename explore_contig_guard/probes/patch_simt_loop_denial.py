#!/usr/bin/env python3
"""Close simt loop-domain claims: the simt loopfit curves are
drain-forced grid=1 calibrations whose only 100-case consumers
under-predict 26-54% per trip (strided_scatter -26%, rowred_r16 -54%);
no simt loop claim improves any case (the four improvements all come
from straight-line it=1 stages).  Keep simt loop claims closed (the
RUNV4..V9 admission default) until the simt loop domain gets its own
multi-program recalibration."""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)

ANCHOR = ("    // ---- Formula B: one evaluation for the whole loop body "
          "--------------" + NL)

GATE = """    // Simt loop bodies: the simt loopfit curves are drain-forced
    // grid=1 calibrations (E17g era).  Their only 100-case consumers
    // under-predict 26-54% per trip (strided_scatter -26%, rowred_r16
    // -54%) and no simt loop claim improves any case (the four v2
    // improvements all come from straight-line it=1 stages), so the
    // drain-forced per-trip price does not transfer to multi-program
    // simt launches.  Keep simt loop claims closed (the RUNV4..V9
    // admission default) until the simt loop domain gets its own
    // multi-program recalibration.
    if (!simd)
      return;
    // ---- Formula B: one evaluation for the whole loop body --------------
"""


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    n = s.count(ANCHOR)
    assert n == 1, "anchor count %d" % n
    s = s.replace(ANCHOR, GATE)
    open(p, "w").write(s)
    assert "drain-forced" in s
    print("simt loop denial inserted")


main()
