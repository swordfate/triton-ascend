#!/usr/bin/env python3
"""Patch E33d step 2: masked + grid-stride structural terms for the
loop-fit pipeline path.

E33d measurements (memory_stream_l4 shape, 4x4KB loads + 4KB store,
grid 56 hot): clean contiguous body 202.6 ns/trip vs real-shape replica
236.2 ns/trip.  Decomposition: runtime-mask predication ~ +11 ns/trip,
grid-stride inter-trip address jump ~ +38 ns/trip once the loop spans a
full wave (g1: ~0).  The E27e curves were calibrated on clean unmasked
contiguous bodies, so masked grid-stride real kernels underprice by the
sum (~+49 ns/trip at 16KB loads; memory_stream_l4 -22.4%).

Change:
  1. ContiguousV2LoopFit gains mask_trip_ns / stride_trip_ns /
     stride_trip_min_grid (profile-tunable constants).
  2. Parser (both engine sites) reads them.
  3. Both loop-fit stage blocks add the two terms to perIter when the
     body carries runtime masks / when the launch spans >= full wave
     with a multi-trip loop.
"""
import json
import sys

H = ("third_party/ascend/costmodel/include/AscendModel/RouteModel/"
     "ContiguousLdStV2CostModel.h")
P = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "SimdSimtCostModel.cpp")
S = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")

STRUCT_OLD = """struct ContiguousV2LoopFit {
  bool enabled = false;
  std::map<std::string, ContiguousV2LoopFitDtype> dtypes;
"""
STRUCT_NEW = """struct ContiguousV2LoopFit {
  bool enabled = false;
  std::map<std::string, ContiguousV2LoopFitDtype> dtypes;
  // E33d (2026-11-19) structural terms for real (masked grid-stride)
  // loop bodies, measured on the memory_stream_l4 shape at grid 56:
  //   runtime-mask predication ~ +11 ns/trip, inter-trip address jump
  //   ~ +38 ns/trip once the launch spans >= stride_trip_min_grid.
  double maskTripNs = 0.0;
  double strideTripNs = 0.0;
  int64_t strideTripMinGrid = 56;
"""


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"

    def patch(path, old, new, count, tag):
        p = "%s/%s" % (root, path)
        s = open(p).read()
        found = s.count(old)
        assert found == count, "%s: anchor %d, want %d" % (tag, found, count)
        open(p, "w").write(s.replace(old, new))
        print("patched %s (%s)" % (path, tag))

    patch(H, STRUCT_OLD, STRUCT_NEW, 1, "struct terms")

    PARSER_OLD = """        cv2.loopFit.enabled =
            reader.optionalNumber(*lf, "enabled", 0.0) != 0.0;"""
    PARSER_NEW = """        cv2.loopFit.enabled =
            reader.optionalNumber(*lf, "enabled", 0.0) != 0.0;
        cv2.loopFit.maskTripNs =
            reader.optionalNumber(*lf, "mask_trip_ns", 0.0);
        cv2.loopFit.strideTripNs =
            reader.optionalNumber(*lf, "stride_trip_ns", 0.0);
        cv2.loopFit.strideTripMinGrid = static_cast<int64_t>(
            reader.optionalNumber(*lf, "stride_trip_min_grid", 56.0));"""
    patch(P, PARSER_OLD, PARSER_NEW, 2, "parser x2")

    STAGE_OLD = """        if (loadBytes == 0.0)
          resources.store += perIter * kCyclesPerNano;
        else
          resources.load += perIter * kCyclesPerNano;
        claimAll();
        return;
"""
    STAGE_NEW = """        // E33d structural terms: real loop bodies carry runtime masks
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
        if (loadBytes == 0.0)
          resources.store += perIterEff * kCyclesPerNano;
        else
          resources.load += perIterEff * kCyclesPerNano;
        claimAll();
        return;
"""
    patch(S, STAGE_OLD, STAGE_NEW, 2, "stage x2")

    # profile constants: mask/stride structural terms (E33d measured)
    jpath = "%s/third_party/ascend/costmodel/profiles/simd_simt/" \
            "ascend_950pr_simd_simt_v1.json" % root
    prof = json.load(open(jpath))
    for eng in ("simd", "simt"):
        lf = prof[eng]["memory"]["contiguous_ldst_v2"]["loop_envelope_fit"]
        lf["mask_trip_ns"] = 11.0
        lf["stride_trip_ns"] = 38.0
        lf["stride_trip_min_grid"] = 56
    open(jpath, "w").write(json.dumps(prof, indent=1, ensure_ascii=False))
    print("JSON constants set (mask=11, stride=38, minGrid=56)")
    print("ALL_PATCHED")


main()
