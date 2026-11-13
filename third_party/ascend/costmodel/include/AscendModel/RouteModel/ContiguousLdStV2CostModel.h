//===- ContiguousLdStV2CostModel.h - v2 contiguous pricing ----------------===//
//
// Epoch-calibrated contiguous (stride == 1) load/store pricing fitted from
// the 950PR exploration campaign (EXPERIMENTS.md E1-E12; see
// profiles/microbench/data_provider/contiguous_ldst_v2/explore/).
//
// Formula families (per engine/dtype/bytes/residency cell):
//   A (straight-line groups, N instances):
//       T(1) = first
//       T(N) = first + (N-1)*marginal            (N <= Q, queue-depth overlap)
//       T(N) = T(Q) + (N-Q)*c_transfer           (N >  Q)
//   B (runtime loops, K rounds):
//       T(K) = P + K*m                           (linear total)
//       store bodies may fit piecewise: K<=QK -> P+K*m ; K>QK -> P+QK*m
//                                            + (K-QK)*mHi
//   S penalty (simt only): dependent stores interleaved with loads serialize
//       the load stream: T += S*penalty.
//
// Residency: warm iff launch footprint (instances*bytes) <= l2_cache_bytes;
// simt hot (same-block DCache reuse) is NOT covered here yet - documented.
//
// All numbers are nanoseconds from same-session probe/control pair-diff
// medians (mode-immune protocol, EXPERIMENTS.md E8/E9/E12).
//===----------------------------------------------------------------------===//
#ifndef ASCENDMODEL_ROUTEMODEL_CONTIGUOUSLDSTV2COSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_CONTIGUOUSLDSTV2COSTMODEL_H

#include <cmath>
#include <cstdint>
#include <map>
#include <string>

namespace mlir {
namespace ascend {
namespace contiguous_v2 {

struct ContiguousV2Group {
  double first = 0.0;
  double marginal = 0.0;
  double cTransfer = 0.0;
  int64_t Q = 8;
};

struct ContiguousV2Loop {
  double P = 0.0;   // intercept: S_loop + per-launch startup
  double m = 0.0;   // steady per-round price
  bool piecewise = false;
  int64_t QK = 0;
  double mHi = 0.0;
};

struct ContiguousV2Params {
  bool enabled = false;
  int64_t l2CacheBytes = 0;
  // keys: "<direction>/<dtype>/<bytes>/<residency>"  (engine = this profile)
  std::map<std::string, ContiguousV2Group> groupA;
  // keys: "<body>/<dtype>/<bytes>/<residency>"
  std::map<std::string, ContiguousV2Loop> loopB;
  // keys: "<dtype>"
  std::map<std::string, double> sPenalty;
};

/// Static description of one contiguous access stream.
struct ContiguousV2Access {
  int64_t elements = 0;
  int64_t elementBytes = 1;
  bool store = false;
};

/// Price one contiguous access stream, ns.  Returns 0 when the v2 table has
/// no cell for the geometry (caller keeps its fallback path).
///
/// instances : number of same-shape accesses issued back to back (group).
/// loopRounds: loop trip count; 0 = not in a loop (straight-line context).
/// footprintBytes: total distinct bytes the launch touches (residency).
/// dependentStores: stores interleaved in the same stream (simt S).
inline double evaluateContiguousV2Ns(const ContiguousV2Params &params,
                                     const ContiguousV2Access &access,
                                     int64_t instances, int64_t loopRounds,
                                     int64_t footprintBytes,
                                     int64_t dependentStores) {
  if (!params.enabled || access.elements <= 0 || access.elementBytes <= 0)
    return 0.0;
  const int64_t bytes = access.elements * access.elementBytes;

  const char *dtype = nullptr;
  switch (access.elementBytes) {
  case 1: dtype = "int8"; break;
  case 2: dtype = "fp16"; break;
  case 4: dtype = "fp32"; break;
  case 8: dtype = "int64"; break;
  default: return 0.0;
  }
  // nearest calibrated bytes bucket (1024/4096/16384), clamped inside
  static const int64_t kBuckets[] = {1024, 4096, 16384};
  int64_t bucket = kBuckets[0];
  double bestDist = 1e30;
  for (int64_t b : kBuckets) {
    const double d = std::fabs(std::log2((double)bytes) -
                               std::log2((double)b));
    if (d < bestDist) {
      bestDist = d;
      bucket = b;
    }
  }
  const bool warm = footprintBytes <= params.l2CacheBytes;
  const char *res = warm ? "warm" : "cold";

  const double instancesF = (double)instances;
  const double roundsF = (double)loopRounds;

  if (loopRounds >= 2) {
    // ---- Formula B: loop body -------------------------------------------
    const char *body = access.store ? "store" : "indpt";
    const std::string key = std::string(body) + "/" + dtype + "/" +
                            std::to_string(bucket) + "/" + res;
    auto it = params.loopB.find(key);
    if (it == params.loopB.end())
      return 0.0;
    const ContiguousV2Loop &loop = it->second;
    double total;
    if (loop.piecewise && loopRounds > loop.QK)
      total = loop.P + loop.QK * loop.m +
              (roundsF - loop.QK) * loop.mHi;
    else
      total = loop.P + roundsF * loop.m;
    // simt: dependent stores in the same stream serialize loads (E3e)
    if (!access.store && dependentStores > 0) {
      auto pen = params.sPenalty.find(dtype);
      if (pen != params.sPenalty.end())
        total += (double)dependentStores * pen->second;
    }
    return total;
  }

  // ---- Formula A: straight-line group ---------------------------------
  const char *direction = access.store ? "store" : "load";
  const std::string key = std::string(direction) + "/" + dtype + "/" +
                          std::to_string(bucket) + "/" + res;
  auto it = params.groupA.find(key);
  if (it == params.groupA.end())
    return 0.0;
  const ContiguousV2Group &g = it->second;
  const int64_t n = instances;
  double total;
  if (g.Q > 0 && n > g.Q)
    total = g.first + (g.Q - 1) * g.marginal + (n - g.Q) * g.cTransfer;
  else
    total = g.first + (n - 1) * g.marginal;
  return total;
}

} // namespace contiguous_v2
} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_CONTIGUOUSLDSTV2COSTMODEL_H
