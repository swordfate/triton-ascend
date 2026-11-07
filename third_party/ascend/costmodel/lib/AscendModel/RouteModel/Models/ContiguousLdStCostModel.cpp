//===- ContiguousLdStCostModel.cpp - contiguous LD/ST calibration -*- C++ -*-===//
//
// Contiguous 1-D (stride == 1) load/store cost model.
//
// The coefficients and the feature list of every segment live in the stage
// profile under simd.memory.contiguous_ldst; they are produced from
// profiles/microbench/data_provider/contiguous_ldst/contiguous_1d/fitted/*.json
// by scripts/gen_contiguous_model.py.  Nothing numeric is compiled in here.
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/Models/ContiguousLdStCostModel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mlir {
namespace ascend {
namespace contiguous_ldst {

namespace {

enum Feature {
  F_K1 = 0,
  F_LN = 1,
  F_W = 2,
  F_W2 = 3,
  F_blk_part = 4,
  F_burst = 5,
  F_by = 6,
  F_byLnW = 7,
  F_half = 8,
  F_i8 = 9,
  F_invW = 10,
  F_ldg = 11,
  F_lines = 12,
  F_lnW = 13,
  F_lperw = 14,
  F_one = 15,
  F_rounds = 16,
  F_schedR = 17,
  F_te = 18,
  F_thr = 19,
  F_warpB = 20,
  F_byh1k = 21,
  F_d64 = 22,
  F_d64h8 = 23,
  F_mask = 24,
  F_log2eb = 25,
  F_warpRounds = 26,
  F_i8wRounds = 27,
  F_halfwRounds = 28,
  F_realBy = 29,
  F_invU,
  F_COUNT
};

/// Feature names, in exactly the enum order above; the profile JSON uses
/// these strings so the coefficients stay readable and reviewable.
constexpr const char *kFeatureNames[F_COUNT] = {
    "K1",
    "LN",
    "W",
    "W2",
    "blk_part",
    "burst",
    "by",
    "byLnW",
    "half",
    "i8",
    "invW",
    "ldg",
    "lines",
    "lnW",
    "lperw",
    "one",
    "rounds",
    "schedR",
    "te",
    "thr",
    "warpB",
    "byh1k",
    "d64",
    "d64h8",
    "mask",
    "log2eb",
    "warpRounds",
    "i8wRounds",
    "halfwRounds",
    "realBy",
    "invU",
};

/// Per-thread register budget in bytes.
///
/// SIMT register file: 128 KiB per AIV, so the per-thread budget follows the
/// launched thread count (32 threads per warp):
///   1..256 threads   -> 127 registers -> 508 B
///   257..512         ->  64           -> 256 B
///   513..1024        ->  32           -> 128 B
///   1025..2048       ->  16           ->  64 B
inline double regBytes(int warps) {
  const int threads = 32 * warps;
  const int registers = threads <= 256   ? 127
                        : threads <= 512 ? 64
                        : threads <= 1024 ? 32
                                         : 16;
  return 4.0 * static_cast<double>(registers);
}

/// 2^min(2, floor(log2(max(1, x))))
inline double kof(double x) {
  if (x < 1.0)
    x = 1.0;
  int e = static_cast<int>(std::floor(std::log2(x)));
  if (e > 2)
    e = 2;
  if (e < 0)
    e = 0;
  return static_cast<double>(1 << e);
}

inline double padPow2(int64_t n) {
  if (n <= 1)
    return 1.0;
  double p = 1.0;
  while (p < static_cast<double>(n))
    p *= 2.0;
  return p;
}

/// Fill the feature vector exactly as the calibration script does.
void fillFeatures(double *v, AccessForm form, int64_t elements, int elemBytes,
                  int warps, bool simd, int64_t concurrency) {
  for (int i = 0; i < F_COUNT; ++i)
    v[i] = 0.0;

  const double eb = static_cast<double>(elemBytes);
  const double W = static_cast<double>(warps);
  const double nReal = static_cast<double>(elements);
  // SIMD was calibrated on the real element count.  The SIMT latency curves
  // were measured on power-of-two blocks only, so Form A keeps the padded
  // width there; a non-power-of-two SIMT Form A is outside the fit and the
  // caller refuses it rather than widening the bytes behind the formula's back.
  const double e = (form == AccessForm::FormA && !simd) ? padPow2(elements)
                                                       : nReal;

  const double by = e * eb;
  const double lines = by / 128.0;
  const double te = e / (32.0 * W);
  const double thr = te * eb;
  const double ldg = std::ceil(thr / 8.0);
  const double warpB = 32.0 * eb;
  const double schedR = std::ceil(W / 4.0);
  const double isI8 = (elemBytes == 1) ? 1.0 : 0.0;
  const double isHalf = (elemBytes == 2) ? 1.0 : 0.0;

  v[F_one] = 1.0;
  v[F_by] = by;
  // Small-access shape: below one 128 B SIMT cache line the cost is set by
  // how many warps issue and by whether several warps share one line, not by
  // the byte count.
  v[F_d64] = (by >= 64.0) ? 1.0 : 0.0;
  v[F_d64h8] = v[F_d64] * std::max(0.0, W - 8.0);
  v[F_byh1k] = std::max(0.0, by - 1024.0);
  v[F_lines] = lines;
  v[F_te] = te;
  v[F_thr] = thr;
  v[F_ldg] = ldg;
  v[F_W] = W;
  v[F_W2] = W * W;
  // Throughput-regime features: the profile JSON names these directly.
  v[F_mask] = (form == AccessForm::FormB) ? 1.0 : 0.0;
  v[F_log2eb] = std::log2(eb);
  // small 专用：一个 warp 每轮要被调度 schedR 次，窄 dtype 各有一份
  v[F_warpRounds] = W * schedR;
  v[F_i8wRounds] = v[F_warpRounds] * isI8;
  v[F_halfwRounds] = v[F_warpRounds] * isHalf;
  // The throughput calibration used the real byte count, never the padded
  // power-of-two block that the latency curves are keyed on.
  v[F_realBy] = nReal * eb;
  v[F_invU] = 1.0 / static_cast<double>(concurrency > 0 ? concurrency : 1);
  v[F_invW] = 1.0 / W;
  v[F_warpB] = warpB;
  v[F_schedR] = schedR;
  v[F_blk_part] = (std::fmod(by, 32.0) != 0.0) ? 1.0 : 0.0;
  v[F_burst] = std::ceil(by / 16384.0) - 1.0;
  if (v[F_burst] < 0.0)
    v[F_burst] = 0.0;
  v[F_rounds] = std::ceil(te);
  v[F_K1] = by * kof(lines / W);
  v[F_LN] = by * std::log(1.0 + te);
  v[F_byLnW] = by * std::log(1.0 + W);
  v[F_lnW] = std::log(1.0 + W);
  v[F_i8] = isI8;
  v[F_half] = isHalf;
  v[F_lperw] = lines / schedR;
}

} // namespace

namespace {

/// Maps a feature name to its slot.  Built once so evaluation stays a lookup.
const llvm::StringMap<int> &featureIndex() {
  static const llvm::StringMap<int> index = [] {
    llvm::StringMap<int> map;
    for (int i = 0; i < F_COUNT; ++i)
      map[kFeatureNames[i]] = i;
    return map;
  }();
  return index;
}

} // namespace

namespace {

/// Index of the interval containing `value`, clamped so that the caller can
/// always interpolate between knots i and i+1.
size_t intervalOf(const std::vector<double> &knots, double value) {
  if (value <= knots.front())
    return 0;
  if (value >= knots.back())
    return knots.size() - 2;
  size_t i = 0;
  while (i + 2 < knots.size() && value > knots[i + 1])
    ++i;
  return i;
}

} // namespace

double ContiguousTable::interpolate(double te, double by) const {
  const double lte = std::log2(std::max(te, 1e-9));
  const double lby = std::log2(std::max(by, 1.0));
  const std::vector<double> teLog = [&] {
    std::vector<double> v;
    for (double k : teKnots)
      v.push_back(std::log2(std::max(k, 1e-9)));
    return v;
  }();
  const std::vector<double> byLog = [&] {
    std::vector<double> v;
    for (double k : byKnots)
      v.push_back(std::log2(std::max(k, 1.0)));
    return v;
  }();
  const size_t i = intervalOf(teLog, lte);
  const size_t j = intervalOf(byLog, lby);
  const double tx = (lte - teLog[i]) / (teLog[i + 1] - teLog[i]);
  const double ty = (lby - byLog[j]) / (byLog[j + 1] - byLog[j]);
  const double v00 = costsNs[i][j], v10 = costsNs[i + 1][j];
  const double v01 = costsNs[i][j + 1], v11 = costsNs[i + 1][j + 1];
  const double v0 = v00 * (1.0 - tx) + v10 * tx;
  const double v1 = v01 * (1.0 - tx) + v11 * tx;
  return v0 * (1.0 - ty) + v1 * ty;
}

namespace {

/// Linear interpolation on the by axis inside the measured interval; returns
/// 0 (refuse) outside it so the caller keeps the pre-FormS path.
double formsInterpBy(const std::map<int64_t, double> &table, int64_t by) {
  if (table.empty() || by < table.begin()->first || by > table.rbegin()->first)
    return 0.0;
  auto hi = table.lower_bound(by);
  if (hi->first == by)
    return hi->second;
  auto lo = std::prev(hi);
  const double w = static_cast<double>(by - lo->first) /
                   static_cast<double>(hi->first - lo->first);
  return lo->second + w * (hi->second - lo->second);
}

/// FormS single-shot pricing for one lone non-pipelined access.
double estimateFormSNs(const ContiguousAccessStats &astats,
                       const ContiguousFit &fit) {
  const int64_t by = astats.elements * astats.elementBytes;
  const double bpt = static_cast<double>(by) /
                     (32.0 * static_cast<double>(astats.warps));
  const double te = static_cast<double>(astats.elements) /
                    (32.0 * static_cast<double>(astats.warps));
  const char *form = astats.masked ? "B" : "A";
  if (astats.simd) {
    if (te < 8.0) {
      const std::string key =
          astats.store ? "store" : (std::string("load.") + form);
      auto it = fit.formSFormulas.find(key);
      if (it == fit.formSFormulas.end())
        return 0.0;
      double ns = 0.0;
      bool known = true;
      for (size_t i = 0; i < it->second.features.size(); ++i) {
        const double v = it->second.features[i] == "one"     ? 1.0
                         : it->second.features[i] == "bpt"   ? bpt
                         : it->second.features[i] == "log2eb"
                             ? std::log2(static_cast<double>(astats.elementBytes))
                             : -1.0;
        if (v < 0.0) {
          known = false;
          break;
        }
        ns += it->second.coefficients[i] * v;
      }
      if (!known || !(ns > 0.0))
        return 0.0;
      return std::max(ns, 15.0);
    }
    const std::string key = (astats.store ? "ldst.store." : "ldst.load.") +
                            std::string(form) + ".eb" +
                            std::to_string(astats.elementBytes);
    auto it = fit.formSTables.find(key);
    return it == fit.formSTables.end() ? 0.0 : formsInterpBy(it->second, by);
  }
  static const char *kDts[] = {"",      "int8", "fp16", "",     "fp32",
                               "",      "",     "",     "int64"};
  const std::string key =
      (astats.store ? "simt.store." : "simt.load.") + std::string(form) + "." +
      (astats.elementBytes >= 1 && astats.elementBytes <= 8
           ? kDts[astats.elementBytes]
           : "") +
      ".W" + std::to_string(astats.warps);
  auto it = fit.formSTables.find(key);
  return it == fit.formSTables.end() ? 0.0 : formsInterpBy(it->second, by);
}

} // namespace

double estimateContiguousCompletionNs(const ContiguousAccessStats &astats,
                                      const ContiguousFit &fit) {
  if (!astats.isValid() || !fit.isValid())
    return 0.0;
  if (astats.elements > fit.maxElements || astats.warps > fit.maxWarps)
    return 0.0;

  const bool isStore = astats.store;
  const AccessForm form = astats.masked ? AccessForm::FormB : AccessForm::FormA;
  const bool vectorized = astats.simd;
  const int elemBytes = static_cast<int>(astats.elementBytes);
  const int warps = static_cast<int>(astats.warps);

  // The SIMT latency curves were fitted on power-of-two block widths only.
  // Do not silently widen a non-power-of-two access into that fit.
  if (!vectorized && form == AccessForm::FormA &&
      padPow2(astats.elements) != static_cast<double>(astats.elements))
    return 0.0;

  double v[F_COUNT];
  fillFeatures(v, form, astats.elements, elemBytes, warps, vectorized,
               astats.concurrency);

  const double lines = v[F_lines];
  const double te = v[F_te];

  // Temporary: ASCEND_MODEL_CONTIG_DEBUG=1 traces every decision.
  static const bool dbg = [] {
    return ::getenv("ASCEND_MODEL_CONTIG_DEBUG") != nullptr;
  }();
  auto note = [&](const char *what, double value, const std::string &key) {
    if (!dbg)
      return;
    std::fprintf(stderr,
                 "[contig] elem=%lld eb=%d warps=%d by=%.0f te=%.1f vec=%d "
                 "%s key=%s -> %.1f\n",
                 static_cast<long long>(astats.elements), elemBytes, warps,
                 static_cast<double>(astats.elements) * elemBytes, te,
                 vectorized ? 1 : 0, what, key.c_str(), value);
  };

  // FormS: a lone single-shot access with NO repetition context (no loop,
  // no repeated iterations, no multi-access group) keeps the single-shot
  // calibration.  A loop-carried access (streams=true but pipelined=false)
  // must fall through to the serial formulas: the acc-chain amortizes the
  // startup, so its marginal cost is below the single-shot value.
  if (!astats.streams && astats.concurrency <= 1) {
    const double formNs = estimateFormSNs(astats, fit);
    if (formNs > 0.0) {
      note("forms", formNs, "forms");
      return formNs;
    }
  }
  // Below one 128 B SIMT cache line the transfer is latency bound: the cost is
  // set by how many warps each issue an access instruction and by whether
  // several of them contend for the same half line, not by the byte count.
  // Measured on 266 configs (min of six runs each).
  const double byBytes = static_cast<double>(astats.elements) *
                         static_cast<double>(elemBytes);
  note("enter", byBytes, std::string());
  // Below one 128 B SIMT cache line one formula per direction covers every
  // element width and both access forms (266 calibration points).
  if (!vectorized && byBytes < 128.0) {
    const std::string key = isStore ? "small.store" : "small.load";
    const auto small = fit.formulas.find(key);
    if (small != fit.formulas.end() && small->second.isValid()) {
      double sum = 0.0;
      for (size_t i = 0; i < small->second.features.size(); ++i) {
        const std::string &name = small->second.features[i];
        if (name == "one") {
          sum += small->second.coefficients[i];
          continue;
        }
        size_t index = F_COUNT;
        for (size_t f = 0; f < F_COUNT; ++f)
          if (name == kFeatureNames[f]) { index = f; break; }
        if (index < F_COUNT)
          sum += small->second.coefficients[i] * v[index];
      }
      if (sum > 0.0) {
        note("small", sum, key);
        return sum;
      }
    }
  }

  // Form B splits on 2*te*eb > register budget, matching the fitting script.
  const bool spillB = (2.0 * v[F_thr]) > regBytes(warps);

  // Formula selection: the conditions are structural, the coefficients are not.
  llvm::StringRef key;
  if (vectorized) {
    key = isStore ? "simd.store" : "simd.load";
  } else if (!isStore) {
    // One formula covers the whole unmasked+masked load region below 32 lines;
    // the mask term inside it absorbs the difference (measured on 243 points).
    if (lines <= 32.0)
      key = "normal.load";
    else if (!(form == AccessForm::FormB))
      key = llvm::StringRef(); // lines > 32 and unmasked: keep the fallback
    else if (!spillB)
      key = "masked.load.l_gt_32_nospill";
    else
      key = "masked.load.l_gt_32_spill";
  } else if (form == AccessForm::FormA) {
    if (te < 1.0)
      key = "unmasked.store.te_lt_1";
    else if (elemBytes == 1)
      key = "unmasked.store.te_ge_1_int8";
    else if (warps <= 4)
      key = "unmasked.store.te_ge_1_wide_w_le_4";
    else
      key = "unmasked.store.te_ge_1_wide_w_gt_4";
  } else {
    // A full-block mask still costs a real regime on the write side: fp16 is
    // predicated per element, so it cannot merge into the wide types.
    if (te < 1.0)
      key = "masked.store.te_lt_1";
    else if (elemBytes == 1)
      key = "masked.store.te_ge_1_int8";
    else if (elemBytes == 2)
      key = "masked.store.te_ge_1_fp16";
    else if (elemBytes == 4)
      key = "masked.store.te_ge_1_fp32";
    else
      key = "masked.store.te_ge_1_int64";
  }

  // Throughput regime: with no cross-iteration dependency the accesses
  // pipeline, so the calibrated throughput curves replace the latency ones.
  // Past the calibrated byte range the streaming term is an extrapolation, so
  // the latency curve keeps the access.  The two sides were measured over
  // different ranges: SIMD up to 16 KiB, SIMT only up to 8 KiB.
  const double pipelinedBytes =
      static_cast<double>(astats.elements) * astats.elementBytes;
  const double throughputLimit = vectorized ? 16384.0 : 8192.0;
  if (astats.pipelined && pipelinedBytes <= throughputLimit) {
    // regimes cover different domains (a >4 KiB unmasked load has no latency
    // slot but does have throughput curves).
    std::string bwKey;
    if (vectorized) {
      bwKey = isStore ? "bw.simd.store" : "bw.simd.load";
    } else {
      bwKey = std::string("bw.simt.") + (isStore ? "store" : "load") + ".W" +
              std::to_string(warps);
    }
    const auto bwIt = fit.formulas.find(bwKey);
    if (bwIt == fit.formulas.end() || !bwIt->second.isValid()) {
      // No throughput curve for this shape: fall back to the latency price and
      // leave a trace, otherwise the fallback is invisible in an A/B.
      note("bw-missing", 0.0, bwKey);
    } else {
      const llvm::StringMap<int> &bwIndex = featureIndex();
      double value = 0.0;
      bool known = true;
      for (size_t i = 0; i < bwIt->second.features.size(); ++i) {
        const auto slot = bwIndex.find(bwIt->second.features[i]);
        if (slot == bwIndex.end()) {
          known = false;
          break;
        }
        value += bwIt->second.coefficients[i] * v[slot->second];
      }
      if (known && value > 0.0) {
        // A second branch expresses the streaming floor of the same curve.
        if (!bwIt->second.maxFeatures.empty()) {
          double floorValue = 0.0;
          bool floorKnown = true;
          for (size_t i = 0; i < bwIt->second.maxFeatures.size(); ++i) {
            const auto slot = bwIndex.find(bwIt->second.maxFeatures[i]);
            if (slot == bwIndex.end()) {
              // Same policy as the latency branch: an unknown name is a profile
              // bug, so refuse the curve instead of silently dropping the floor.
              floorKnown = false;
              break;
            }
            floorValue += bwIt->second.maxCoefficients[i] * v[slot->second];
          }
          if (!floorKnown)
            return 0.0;
          value = std::max(value, floorValue);
        }
        note("bw", value, bwKey);
        return value;
      }
    }
  }
  // Measured-table regime: past eight elements per thread on a multi-line
  // transfer, cost follows the function's spill placement rather than a
  // curve.  Use the measured knots when the profile carries them.
  if (!vectorized && te >= 8.0) {
    const std::string key =
        (isStore ? std::string("store.eb") : std::string("load.eb")) +
        std::to_string(elemBytes);
    const auto table = fit.tables.find(key);
    if (table != fit.tables.end() && table->second.isValid()) {
      const double by = static_cast<double>(astats.elements) *
                        static_cast<double>(elemBytes);
      const double tableValue = table->second.interpolate(te, by);
      note("table", tableValue, key);
      return tableValue;
    }
  }
  if (key.empty())
    return 0.0;



  const auto it = fit.formulas.find(key);
  if (it == fit.formulas.end() || !it->second.isValid())
    return 0.0;
  const ContiguousFormula &formula = it->second;
  const llvm::StringMap<int> &index = featureIndex();
  double ns = 0.0;
  for (size_t i = 0; i < formula.features.size(); ++i) {
    const auto slot = index.find(formula.features[i]);
    // An unknown feature name is a profile bug: refuse rather than guess.
    if (slot == index.end())
      return 0.0;
    ns += formula.coefficients[i] * v[slot->second];
  }
  if (!(ns > 0.0))
    return 0.0;
  return ns;
}

} // namespace contiguous_ldst
} // namespace ascend
} // namespace mlir
