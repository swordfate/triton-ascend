#!/usr/bin/env python3
"""Patch E33: reduce/dot trip families for the loop domain.

Evidence (E33, 2026-11-19, clean card 3, profiler K-differencing):
  bodies whose every trip ends in an in-trip reduction + small store
  (rowwise reduction / dot / reduction-epilogue shapes) do NOT pipeline
  across trips: per-trip cost = load curve + ~175-300 ns serial floor
  (simd) / ~230-300+ ns (simt), and the floor is grid- and
  residency-sensitive (cold g=56 enters the aggregate-bandwidth regime).
  The pipeline curves (E27e) miss this entirely -> rowred_r16 -54%,
  reduce_sum_epilogue -30%, grid_stride_dot_p56/p112 -21/-20%.

Change:
  1. ContiguousV2LoopFit gains trip_reduce / trip_dot tables
     (grid knot -> hot/cold -> per-trip-load bytes -> per-iteration ns),
     parsed from loop_envelope_fit at both engine sites.
  2. tripFitNs(): log2-lerp across grid knots, piecewise-linear over
     bytes.
  3. Stage loop blocks (simd+simt): when the body is reduction-fed
     (work.reductionOperations>0) with all stores <=128B and nLoads in
     {1,2}, the whole per-iteration price comes from the trip table
     (falls through to the pipeline curves when no table covers it).
"""
import sys

H = ("third_party/ascend/costmodel/include/AscendModel/RouteModel/"
     "ContiguousLdStV2CostModel.h")
P = "third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimdSimtCostModel.cpp"
S = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
     "StageCostModels.cpp")

STRUCT_OLD = """struct ContiguousV2LoopFit {
  bool enabled = false;
  std::map<std::string, ContiguousV2LoopFitDtype> dtypes;
};
"""
STRUCT_NEW = """struct ContiguousV2LoopFit {
  bool enabled = false;
  std::map<std::string, ContiguousV2LoopFitDtype> dtypes;
  // E33 (2026-11-19) reduce/dot trip families: whole-trip per-iteration
  // ns measured on bodies whose every trip ends in an in-trip reduction
  // + small store (rowwise reduction / dot / reduction-epilogue shapes).
  // Grid knot ("1"/"56") -> "hot"/"cold" -> per-trip load bytes -> ns.
  // nL=1 trips take tripReduce, nL=2 trips take tripDot; bytes keys are
  // per-trip-load bytes (dtype-independent).
  std::map<int64_t, std::map<std::string, std::map<int64_t, double>>>
      tripReduce;
  std::map<int64_t, std::map<std::string, std::map<int64_t, double>>>
      tripDot;
};

/// E33 trip-table lookup: log2-lerp across grid knots (hot is measured
/// grid-flat; cold interpolates the aggregate-bandwidth regime), then
/// piecewise-linear over per-trip load bytes with end clamping/scaling.
inline double tripFitNs(
    const std::map<int64_t,
                   std::map<std::string, std::map<int64_t, double>>> &tab,
    int64_t grid, bool cold, double bytes) {
  if (tab.empty() || bytes <= 0.0)
    return 0.0;
  auto atRes = [&](const std::map<std::string, std::map<int64_t, double>>
                       &g) -> const std::map<int64_t, double> * {
    auto r = g.find(cold ? "cold" : "hot");
    return r == g.end() ? nullptr : &r->second;
  };
  auto atBytes = [](const std::map<int64_t, double> &m,
                    double b) -> double {
    if (m.empty())
      return 0.0;
    auto hi = m.lower_bound((int64_t)b);
    if (hi == m.end()) {
      auto last = std::prev(m.end());
      return last->second * b / double(last->first);
    }
    if (hi == m.begin())
      return hi->second;
    auto lo = std::prev(hi);
    const double w = (b - double(lo->first)) /
                     double(hi->first - lo->first);
    return lo->second + w * (hi->second - lo->second);
  };
  auto exact = tab.find(grid);
  if (exact != tab.end()) {
    const auto *m = atRes(exact->second);
    return m ? atBytes(*m, bytes) : 0.0;
  }
  auto hi = tab.lower_bound(grid);
  if (hi == tab.end()) {
    const auto *m = atRes(std::prev(hi)->second);
    return m ? atBytes(*m, bytes) : 0.0;
  }
  if (hi == tab.begin()) {
    const auto *m = atRes(hi->second);
    return m ? atBytes(*m, bytes) : 0.0;
  }
  auto lo = std::prev(hi);
  const double lg0 = std::log2(double(lo->first));
  const double lg1 = std::log2(double(hi->first));
  const double w = (std::log2(double(grid)) - lg0) /
                   std::max(1e-9, lg1 - lg0);
  const auto *m0 = atRes(lo->second);
  const auto *m1 = atRes(hi->second);
  if (!m0 || !m1)
    return 0.0;
  const double v0 = atBytes(*m0, bytes);
  const double v1 = atBytes(*m1, bytes);
  if (v0 <= 0.0 || v1 <= 0.0)
    return std::max(v0, v1);
  return v0 + w * (v1 - v0);
}
"""

PARSER_OLD = """            cv2.loopFit.dtypes[dentry.first.str()] = fd;
          }
        }
"""
PARSER_NEW = """            cv2.loopFit.dtypes[dentry.first.str()] = fd;
          }
        }
        for (const auto &tfp :
             {std::pair<const char *,
                        std::map<int64_t,
                                 std::map<std::string,
                                          std::map<int64_t, double>>> *>(
                  "trip_reduce", &cv2.loopFit.tripReduce),
              std::pair<const char *,
                        std::map<int64_t,
                                 std::map<std::string,
                                          std::map<int64_t, double>>> *>(
                  "trip_dot", &cv2.loopFit.tripDot)}) {
          if (const auto *to = lf->getObject(tfp.first)) {
            for (const auto &gk : *to) {
              int64_t g = 0;
              if (StringRef(gk.first.str()).getAsInteger(10, g))
                continue;
              for (const auto &rk : *gk.second) {
                auto &dst = (*tfp.second)[g][rk.first.str()];
                if (const auto *bo = rk.second.getAsObject()) {
                  for (const auto &bk : *bo) {
                    int64_t b = 0;
                    if (StringRef(bk.first.str()).getAsInteger(10, b))
                      continue;
                    if (const auto n = bk.second.getAsNumber())
                      dst[b] = *n;
                  }
                }
              }
            }
          }
        }
"""

STAGE_OLD = """      if (itf != profile.contiguousV2.loopFit.dtypes.end()) {
"""
STAGE_NEW = """      if (itf != profile.contiguousV2.loopFit.dtypes.end()) {
        // E33 reduce/dot trip families: bodies whose every trip ends in
        // an in-trip reduction + small store (rowwise reduction / dot /
        // reduction-epilogue shapes) do not pipeline across trips; the
        // per-iteration price is the measured whole-trip table, grid-
        // and residency-aware.  Falls through to the pipeline curves
        // when no table covers the shape.
        const bool tripRed = work.reductionOperations > 0.0 ||
                             work.shuffleLaneSteps > 0.0;
        int64_t maxStoreChunk = 0;
        for (const StageMemoryAccess *access : eligible)
          if (access->store)
            maxStoreChunk = std::max<int64_t>(
                maxStoreChunk,
                v2ElementCount(access) * (access->elementBits / 8));
        const bool tripTiny = maxStoreChunk > 0 && maxStoreChunk <= 128;
        if (tripRed && tripTiny && (nLoads == 1 || nLoads == 2)) {
          const auto &ttab = (nLoads == 1) ? itf->second.tripReduce
                                           : itf->second.tripDot;
          int64_t loopFootprint = 0;
          for (const StageMemoryAccess *access : eligible)
            loopFootprint +=
                v2ElementCount(access) * (access->elementBits / 8);
          const double tripPer = contiguous_v2::tripFitNs(
              ttab, programCountHint,
              loopFootprint > profile.contiguousV2.l2CacheBytes,
              double(loadBytes) / double(nLoads));
          if (tripPer > 0.0) {
            if (loadBytes == 0.0)
              resources.store += tripPer * kCyclesPerNano;
            else
              resources.load += tripPer * kCyclesPerNano;
            claimAll();
            return;
          }
        }
"""


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/work/triton-ascend"

    def patch(path, old, new, count, tag):
        p = "%s/%s" % (root, path)
        s = open(p).read()
        found = s.count(old)
        assert found == count, "%s: anchor found %d times, want %d" % (
            tag, found, count)
        s = s.replace(old, new)
        open(p, "w").write(s)
        print("patched %s (%s)" % (path, tag))

    patch(H, STRUCT_OLD, STRUCT_NEW, 1, "struct+helper")
    patch(P, PARSER_OLD, PARSER_NEW, 2, "parser x2 engines")
    patch(S, STAGE_OLD, STAGE_NEW, 2, "stage x2 engines")
    print("ALL_PATCHED")


main()
