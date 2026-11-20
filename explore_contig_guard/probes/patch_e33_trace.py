#!/usr/bin/env python3
"""Add decisive charge-site and post-call trace prints (env-gated)."""
import sys

S = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)
BS = chr(92)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, S)
    s = open(p).read()

    # 1. charge-site print: right before the simd charge (after terms)
    old_c = """        if (loadBytes == 0.0)
          resources.store += perIterEff * kCyclesPerNano;
        else
          resources.load += perIterEff * kCyclesPerNano;"""
    new_c = """        if (getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr)
          fprintf(stderr, "E33CHARGE simd=%d perIterEff=%f loadBefore=%f\\n",
                  (int)simd, perIterEff, resources.load);
        if (loadBytes == 0.0)
          resources.store += perIterEff * kCyclesPerNano;
        else
          resources.load += perIterEff * kCyclesPerNano;
        if (getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr)
          fprintf(stderr, "E33CHARGE2 loadAfter=%f\\n", resources.load);"""
    assert s.count(old_c) == 1, "charge anchor %d" % s.count(old_c)
    s = s.replace(old_c, new_c)

    # 2. post-call print in mapWorkload: after mapContiguousV2Completion
    old_w = """    mapContiguousV2Completion(stage, work, profile, resources, simd, layout,
"""
    assert s.count(old_w) >= 1, "caller anchor %d" % s.count(old_w)
    # find the full call statement: it spans two lines ending with 'claimed);'
    idx = s.find(old_w)
    end_marker = "stridedAlreadyClaimed);"
    end_idx = s.find(end_marker, idx)
    assert end_idx > 0, "call end not found"
    insert_at = end_idx + len(end_marker)
    ins = (NL +
           '    if (getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr)' + NL +
           '      fprintf(stderr, "E33AFTER simd=%d resLoad=%f' + BS + 'n",' +
           NL + '              (int)simd, resources.load);')
    s = s[:insert_at] + ins + s[insert_at:]
    open(p, "w").write(s)
    print("TRACE_PRINTS_ADDED")


main()
