#!/usr/bin/env python3
"""Loop-domain admission gate for the contiguous-v2 loop branch.

The loopfit curves were calibrated on clean single-program bodies
(grid=1, hot L2, no mask, linear inter-trip walk, no reduction/scan
coupling - probe e27b_fit2).  Stage-level 4902-vs-v2 differencing on
the fresh clean 100-case run attributes five baseline-PASS -> now-FAIL
regressions to extrapolating those curves onto structures outside the
domain (per-iteration shortfall, fresh vs needed):
  muls_add_n1048576   +28.2 ns/trip (-20%)  grid-stride shell
  memory_stream_l4    +70.0 ns/trip (-31%)  grid-stride shell (E33d:
                        mask +11 + inter-trip jump +38 measured)
  pytest_fused_softmax +83.8 ns/trip (-23%) grid-stride rows + reduction
  pytest_strided_scatter -83.6 ns/trip (-26%) runtime-masked simt body
  pytest_permute_reshape  -204  ns/trip (-50%) strided-row members
                        (32B rows @ 384B stride priced as contiguous)
Gate: a loop body carrying a grid-stride shell, runtime partial masks,
reduction/scan coupling, or StridedWide members is outside the
calibrated domain - return unclaimed so the legacy measured group
curves / strided tiers price it exactly as the pre-v2 baseline did.
Unmasked non-grid-stride clean bodies stay admissible (E29: hot bodies
grid-flat; the elementwise and gs_dot families track within a few %).
Env override for A/B: ASCEND_V2_ALLOW_UNCALIBRATED_LOOP=1.
"""
import sys

P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")
NL = chr(10)

ANCHOR = ("  if (hasLoop) {" + NL +
          "    // ---- Formula B: one evaluation for the whole loop body "
          "--------------")

GATE = """  if (hasLoop) {
    // ---- Loop-domain admission (2026-11-20): the loopfit curves were
    // calibrated on clean single-program bodies (grid=1, hot L2, no
    // mask, linear inter-trip walk, no reduction/scan coupling -
    // probe e27b_fit2).  Stage-level baseline-vs-v2 differencing on
    // the 100-case suite attributes the five baseline-PASS -> FAIL
    // regressions to extrapolating those curves onto structures
    // outside that domain (per-iteration shortfall, fresh vs needed):
    //   muls_add_n1048576    +28.2 ns/trip (-20%)  grid-stride shell
    //   memory_stream_l4     +70.0 ns/trip (-31%)  grid-stride shell
    //                        (E33d: mask +11 + inter-trip jump +38)
    //   pytest_fused_softmax +83.8 ns/trip (-23%)  gs rows + reduction
    //   pytest_strided_scatter -83.6 ns/trip (-26%) runtime-masked simt
    //   pytest_permute_reshape -204  ns/trip (-50%) strided-row members
    //                        (32B rows @ 384B stride priced contiguous)
    // A body carrying a grid-stride shell, runtime partial masks,
    // reduction/scan coupling, or StridedWide members is outside the
    // calibrated domain: stay unclaimed so the legacy measured group
    // curves / strided tiers price it exactly as the pre-v2 baseline
    // did.  Clean unmasked non-grid-stride bodies remain admissible
    // (E29: hot bodies grid-flat; elementwise/gs_dot track a few %).
    {
      const char *allowUncalibrated =
          ::getenv("ASCEND_V2_ALLOW_UNCALIBRATED_LOOP");
      if (!(allowUncalibrated && *allowUncalibrated &&
            allowUncalibrated[0] != '0')) {
        bool outsideDomain = stage.features.isGridStrideLoopShell ||
                             stage.features.hasReduction ||
                             stage.features.hasPrefixScan ||
                             work.reductionOperations > 0.0 ||
                             work.shuffleLaneSteps > 0.0;
        for (const StageMemoryAccess *access : eligible) {
          if (access->masked && access->activeElementRatio > 0.0 &&
              access->activeElementRatio < 1.0)
            outsideDomain = true; // runtime-predicated body
          if (access->layoutClass() ==
              StageMemoryAccess::LayoutClass::StridedWide)
            outsideDomain = true; // strided-row rider priced as contiguous
        }
        if (outsideDomain)
          return;
      }
    }
    // ---- Formula B: one evaluation for the whole loop body --------------"""


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"
    p = "%s/%s" % (root, P)
    s = open(p).read()
    n = s.count(ANCHOR)
    assert n == 1, "anchor count %d" % n
    s = s.replace(ANCHOR, GATE)
    open(p, "w").write(s)
    print("gate inserted; hasLoop occurrences now:", s.count("if (hasLoop) {"))
    assert "ASCEND_V2_ALLOW_UNCALIBRATED_LOOP" in s
    assert "outsideDomain" in s


main()
