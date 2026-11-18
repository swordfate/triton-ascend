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
  // E17 (2026-11-17): mixed-body loop envelope replacement for Tier 2.
  // perIter = max(TL(loadBytes), TS(storeBytes))
  //           + kappa * min(TL, TS)          (non-aliased bodies)
  // TL/TS are 4-point measured curves (bytes log2-interpolated, clamped):
  // TL {4KB:76.3, 8KB:96.4, 16KB:167.9, 32KB:334.7},
  // TS {4KB:79.6, 8KB:93.9, 16KB:157.4, 32KB:276.1}, kappa = 0.18.
  // Fitted on the E17 two-direction sweep; replaces the byte-sum
  // envelope ONLY for mixed (load+store) bodies - single-direction
  // bodies keep the prior envelope (gate-validated on memory_stream).
  bool mixEnvelopeEnabled = false;
  double mixBytes[4] = {4096.0, 8192.0, 16384.0, 32768.0};
  double mixTL[4] = {76.3, 96.4, 167.9, 334.7};
  double mixTS[4] = {79.6, 93.9, 157.4, 276.1};
  double mixKappa = 0.18;
  // E17h2/E17i (2026-11-18): SIMT per-line envelope, measured on both
  // directions.  Loads are width-sensitive (fp16 64B instructions waste
  // half the 128B DCache line) with a 4KB issue penalty and a 32KB jump
  // (queue-depth hypothesis); stores are width-independent and nearly
  // size-flat at L2.  Residency: footprint (load+store bytes x K x
  // programs) vs L2 128MB.
  // E18b (2026-11-18): tiny dependent-store excess.  A loop body's
  // tiny (<=128B) load-dependent stores serialize once their count
  // exceeds the pipeline hidden capacity C (C=2 at >=16KB dominant
  // load, else 1); each excess store costs ~97 ns/iter (measured
  // 93-99 across 2/4/16KB loads - geometry-independent).
  bool tinyStoreExcessEnabled = false;
  double tinyStoreExcessNs = 97.0;
  double tinyStoreCapacity2Bytes = 16384.0;
  bool simtLineEnvelopeEnabled = false;
  double simtLineL2Bytes = 134217728.0;
  double simtLoadLine16L2[4] = {13.63, 12.77, 15.03, 28.11};
  double simtLoadLine16Dram[4] = {16.97, 22.12, 31.44, 50.85};
  double simtLoadLine32L2[4] = {11.26, 8.14, 5.97, 11.69};
  double simtLoadLine32Dram[4] = {15.19, 11.70, 10.59, 23.36};
  double simtLoadLine64L2[4] = {11.26, 8.14, 5.97, 11.69};
  double simtLoadLine64Dram[4] = {15.19, 11.70, 10.59, 23.36};
  double simtStoreLineL2[4] = {6.22, 6.46, 6.08, 6.08};
  double simtStoreLineDram[4] = {13.19, 18.41, 22.55, 17.48};
  // 2026-11-17 group_a compression: per-direction/residency cTransfer
  // ratio (cTransfer = kc * marginal; Q is structural, groupAQueues
  // below).  keys: "load/warm", "load/cold", "store/warm", "store/cold".
  std::map<std::string, double> groupACtKc = {
      {"load/warm", 1.231}, {"load/cold", 2.768},
      {"store/warm", 1.003}, {"store/cold", 1.021}};
  // keys: "<body>/<dtype>/<bytes>/<residency>"
  std::map<std::string, ContiguousV2Loop> loopB;
  // keys: "<dtype>"
  std::map<std::string, double> sPenalty;
  // P-axis overlap factors from E13: kind/engine/residency ->
  //   {P -> per-program price / per-program price at P=1}
  std::map<std::string, std::map<int64_t, double>> pAxis;
  // aggregate bandwidth floors (bytes/ns) from E4-C footprint sweeps
  double bwWarmBytesPerNs = 2400.0;
  double bwColdBytesPerNs = 900.0;
  // calF2 direct rows (per-engine subobject, keys are bare):
  //   "n<loads>_s<stores>_b<chunkBytes>_k<K>_g<grid>" ->
  //   whole-launch per-program-equivalent ns of the measured loop body.
  // The loop-body price interpolates T(K) = A + K*m across the swept K
  // rows of the same geometry; direct_fixed_share splits the intercept
  // against the route model's own boundary stages.
  std::map<std::string, double> directTable;
  // calF3 straight-line rows: "n<loads>_s<stores>_b<chunkBytes>_g<grid>"
  // -> whole-launch per-program-equivalent ns, guard-identical bodies.
  std::map<std::string, double> sdirectTable;
  // The route model prices its own boundary/fixed stages on top, so the
  // whole-launch row charges row - sdirect_fixed_offset (cycles).
  double sdirectFixedOffset = 2400.0;
  // calF2 tiny-store loop-body rows, same key format as `direct`:
  // "n<loads>_s<stores>_b<chunkBytes>_k<K>_g<grid>" -> whole-launch
  // per-program-equivalent ns.  Selected for loop bodies whose stores are
  // all <=128B beside a >=512B dominant chunk (reduction-epilogue shape).
  std::map<std::string, double> tinyDirectTable;
  // Share of the measured latency-chain excess (row value above the
  // linear T(K)=A+K*m trend) charged per iteration; the rest of the
  // excess overlaps the route boundary/launch.
  double tinyStoreChainShare = 0.636;
  // calH boundary-hiding coefficients: the launch boundary overlaps the
  // contiguous memory stream (first-load DRAM latency + cross-instance
  // head/tail overlap), so a regime-constant share of the additive
  // boundary is subtracted at candidate level.
  //   hide = boundary * c(wave regime, kernel has contiguous loads)
  double boundaryHideLoadWave1 = 0.47;
  double boundaryHideStoreWave1 = 0.28;
  double boundaryHideLoadWave23 = 0.23;
  double boundaryHideStoreWave23 = 0.20;
  double boundaryHideLoadWave4 = 0.20;
  double boundaryHideStoreWave4 = 0.0;
  double directFixedShare = 0.3;
  // resident program count for the current kernel (grid size)
  int64_t programCountHint = 1;
};

/// P-axis overlap factor: per-program price / price at P=1 (log2-P lerp,
/// clamped at the table ends).  kind = "loadsum" | "storeonly".
/// E14 decomposition (a pure scalar-store kernel reproduces the whole
/// T(P) curve while the per-program memory/reduce unit price stays flat
/// at 200-300 ns): inside one wave (P <= 56) the unit price does not
/// depend on P - the growth is per-block dispatch/prologue, which the
/// candidate-level boundary_residual(P) term owns.  The measured decay is
/// therefore applied only above one wave (multi-wave aggregate-bandwidth
/// amortization).
inline double pAxisFactor(const ContiguousV2Params &params,
                          const std::string &kind, const char *engineShort,
                          const char *res, int64_t programCount) {
  if (programCount > 0 && programCount <= 56)
    return 1.0;
  const std::string key = kind + "/" + engineShort + "/" + res;
  auto it = params.pAxis.find(key);
  if (it == params.pAxis.end() || it->second.empty())
    return 1.0;
  const auto &table = it->second;
  auto lo = table.upper_bound(programCount);
  if (lo != table.begin())
    --lo;
  auto hi = lo;
  if (hi != table.end())
    ++hi;
  if (hi == table.end())
    return lo->second;
  if (lo->first == programCount)
    return lo->second;
  const double w = (double)(programCount - lo->first) /
                   (double)(hi->first - lo->first);
  return lo->second + w * (hi->second - lo->second);
}

/// E17 envelope (unified set): per-iteration ns of a loop body moving
/// loadBytes of loads and storeBytes of stores (non-aliased); a
/// single-direction body (one side zero) takes the pure direction
/// curve.  Each
/// direction follows its measured concave curve (rate ramps with
/// transaction size); the two streams overlap, so the body pays the
/// larger stream plus a fitted share of the smaller (DRAM contention).
/// Shared log2-byte curve interpolation over a 4-point table
/// (endpoints clamped).
inline double envelopeCurve(const double *sizes, const double *points,
                            double bytes) {
  double b = bytes < sizes[0] ? sizes[0] : bytes;
  double v = points[3];
  for (int i = 3; i >= 0; --i) {
    if (b >= sizes[i]) {
      if (i == 3)
        return points[3];
      const double w = (std::log2(b) - std::log2(sizes[i])) /
                       (std::log2(sizes[i + 1]) - std::log2(sizes[i]));
      return points[i] + w * (points[i + 1] - points[i]);
    }
    v = points[i];
  }
  return v;
}

/// E17 mix envelope: per-iteration ns of a loop body moving loadBytes
/// of loads and storeBytes of stores (non-aliased); a single-direction
/// body takes the pure direction curve.
inline double mixEnvelopePerIter(const ContiguousV2Params &params,
                                 double loadBytes, double storeBytes) {
  auto curve = [&](const double *points, double bytes) {
    return envelopeCurve(params.mixBytes, points, bytes);
  };
  if (storeBytes <= 0.0)
    return curve(params.mixTL, loadBytes);
  if (loadBytes <= 0.0)
    return curve(params.mixTS, storeBytes);
  const double tl = curve(params.mixTL, loadBytes);
  const double ts = curve(params.mixTS, storeBytes);
  return std::max(tl, ts) + params.mixKappa * std::min(tl, ts);
}

/// E17h2/E17i SIMT line envelope: per-128B-line cost for one direction
/// (bits selects the load width table; stores share one width-free
/// table), interpolated over the per-iteration transaction size.
inline const double *simtLineTable(const ContiguousV2Params &params,
                                   int bits, bool dram, bool isStore) {
  if (isStore)
    return dram ? params.simtStoreLineDram : params.simtStoreLineL2;
  if (bits <= 16)
    return dram ? params.simtLoadLine16Dram : params.simtLoadLine16L2;
  if (bits <= 32)
    return dram ? params.simtLoadLine32Dram : params.simtLoadLine32L2;
  return dram ? params.simtLoadLine64Dram : params.simtLoadLine64L2;
}

inline double simtLineEnvelopePerIter(const ContiguousV2Params &params,
                                      double loadBytes, int loadBits,
                                      double storeBytes, bool dram) {
  double perIter = 0.0;
  if (loadBytes > 0.0)
    perIter += (loadBytes / 128.0) *
               envelopeCurve(params.mixBytes,
                             simtLineTable(params, loadBits, dram, false),
                             loadBytes);
  if (storeBytes > 0.0)
    perIter += (storeBytes / 128.0) *
               envelopeCurve(params.mixBytes,
                             simtLineTable(params, 32, dram, true),
                             storeBytes);
  return perIter;
}
/// Structural Q (2026-11-17 group_a compression): the queue-depth knee
/// of the segmented group price, from the byte count instead of a stored
/// per-cell value.  Store streams hold a flat 8-deep queue; load streams
/// scale it with in-flight bytes (warm 64KB / cold 32KB, clamped to the
/// calibrated range).  Verified against every stored warm load cell and
/// every store cell before the stored values were dropped.
inline int64_t groupAQueues(bool store, int64_t bytes, bool warm) {
  if (store)
    return 8;
  const int64_t kq = warm ? 65536 : 32768;
  const int64_t hi = warm ? 32 : 8;
  int64_t q = kq / (bytes < 1 ? 1 : bytes);
  q = q < 4 ? 4 : (q > hi ? hi : q);
  return q;
}

/// Structural cTransfer: the transfer-limited per-access price of the
/// n>Q regime as a fixed multiple of the queue-depth marginal (median
/// ratio of the dropped stored values, per direction/residency).
inline double groupACTransfer(const ContiguousV2Params &params,
                              bool store, bool warm, double marginal) {
  const auto it = params.groupACtKc.find(
      std::string(store ? "store" : "load") + "/" +
      (warm ? "warm" : "cold"));
  const double kc = it != params.groupACtKc.end() ? it->second : 1.0;
  return marginal * kc;
}

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
  const int64_t q = groupAQueues(access.store, bytes, warm);
  if (n > q) {
    const double cT =
        groupACTransfer(params, access.store, warm, g.marginal);
    return g.first + (q - 1) * g.marginal + (n - q) * cT;
  }
  return g.first + (n - 1) * g.marginal;
}

} // namespace contiguous_v2
} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_CONTIGUOUSLDSTV2COSTMODEL_H
