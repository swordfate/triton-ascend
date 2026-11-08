//===- ContiguousLdStCostModel.h - Contiguous 1-D load/store model ------===//
//
// Closed-form completion costs for ContiguousShort / ContiguousWide direct
// load/store accesses (stride == 1).
//
// Every coefficient lives in the stage profile under
// simd.memory.contiguous_ldst.formulas, so a recalibration is a JSON edit and
// never a rebuild.  This file only names the interfaces; the formula *shape*
// (which features a segment uses) is part of the same JSON block.
//
// An access is priced from its static geometry alone:
//   elements, elementBytes, warps, masked flag, direction, route.
// No operator or kernel name is ever consulted.
//
// Calibration and fitting:
//   profiles/microbench/data_provider/contiguous_ldst/contiguous_1d/
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_ASCEND_MODEL_ROUTEMODEL_CONTIGUOUSLDSTCOSTMODEL_H
#define MLIR_ASCEND_MODEL_ROUTEMODEL_CONTIGUOUSLDSTCOSTMODEL_H

#include "llvm/ADT/StringMap.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::ascend::contiguous_ldst {

/// Input form of a contiguous access, mirroring the TTIR mask.
enum class AccessForm { FormA, FormB };

/// Static geometry of one contiguous access.  These are the only inputs to
/// formula selection; nothing here depends on an operator name.
struct ContiguousAccessStats {
  int64_t elements = 0;     ///< real element count of the access
  int64_t elementBytes = 0; ///< 1 / 2 / 4 / 8
  int64_t warps = 0;        ///< warps per program (ignored by the SIMD formula)
  bool store = false;
  bool masked = false;
  bool simd = true;
  /// True when the enclosing loop carries no data dependency across iterations,
  /// so consecutive accesses pipeline and the throughput curves apply instead of
  /// the latency ones.
  bool pipelined = false;
  /// True when the access repeats (inside a loop, repeated iterations, or a
  /// same-shape group), regardless of whether it can pipeline.  A loop-carried
  /// access has streams=true but pipelined=false: its marginal cost belongs to
  /// the serial formulas, not the single-shot FormS tables.
  bool streams = false;
  /// Number of same-shape accesses issued back to back in one iteration.  The
  /// throughput fit carries a 1/unroll term, so this is the factor that term
  /// divides by; 1 means the access stands alone.
  int64_t concurrency = 1;

  bool isValid() const { return elements > 0 && elementBytes > 0 && warps > 0; }
};

/// One calibrated formula: feature names and their coefficients, both read
/// from the profile.
struct ContiguousFormula {
  std::vector<std::string> features;
  std::vector<double> coefficients;
  /// Optional second linear branch.  When present the formula evaluates to
  /// max(first, second), which is how the throughput curves express
  /// "latency per concurrent access, but never below streaming time".
  std::vector<std::string> maxFeatures;
  std::vector<double> maxCoefficients;

  bool isValid() const {
    if (features.empty() || features.size() != coefficients.size())
      return false;
    for (double coefficient : coefficients)
      if (!std::isfinite(coefficient))
        return false;
    // The optional second branch is given whole or not at all, and its
    // coefficients must be finite: a NaN would silently vanish inside max().
    if (maxFeatures.size() != maxCoefficients.size())
      return false;
    if (!maxFeatures.empty()) {
      // The two branches may use different feature sets; only the pairing of
      // names with coefficients has to be consistent.
      for (double coefficient : maxCoefficients)
        if (!std::isfinite(coefficient))
          return false;
    }
    return true;
  }
};

/// One measured cost table over (elements-per-thread, bytes).
///
/// Legacy format for the te>=8 SIMT stream regime.  The old calibration's
/// step-shaped values were an artifact of the probe kernel's own te-wide
/// accumulator spilling, not memory behavior; the regime is now priced by
/// the te8_formulas physics families (paired-control calibration).  The
/// table path remains only so old profiles keep loading.
struct ContiguousTable {
  std::vector<double> teKnots;              ///< elements per thread, ascending
  std::vector<double> byKnots;              ///< bytes, ascending
  std::vector<std::vector<double>> costsNs; ///< [te][by]

  bool isValid() const {
    if (teKnots.size() < 2 || byKnots.size() < 2)
      return false;
    if (costsNs.size() != teKnots.size())
      return false;
    for (const auto &row : costsNs) {
      if (row.size() != byKnots.size())
        return false;
      for (double cost : row)
        if (!std::isfinite(cost) || cost <= 0.0)
          return false;
    }
    return true;
  }

  /// Bilinear interpolation in log space; clamps to the knot range.
  double interpolate(double te, double by) const;
};

/// The simd.memory.contiguous_ldst block of the stage profile.
struct ContiguousFit {
  int64_t maxElements = 0;
  int64_t maxWarps = 0;
  /// Keyed by stable segment name, e.g. "simd.load", "a.load.lines_le_2".
  llvm::StringMap<ContiguousFormula> formulas;
  /// Keyed by "<dir>.eb<elementWidth>", e.g. "load.eb2", "store.eb4".
  llvm::StringMap<ContiguousTable> tables;
  /// FormS single-shot (u==1, non-pipelined) pricing: SIMD formulas for the
  /// te<8 domain ("load.A", "load.B", "store") plus by->ns measured tables
  /// ("ldst.load.A.eb4" = SIMD te>=8, "simt.load.A.fp32.W4" = SIMT).
  llvm::StringMap<ContiguousFormula> formSFormulas;
  llvm::StringMap<std::map<int64_t, double>> formSTables;
  /// SIMT te>=8 stream (loop-carried / pipelined) pricing: physics formulas
  /// ("load.eb4" specific, "load" merged across width) replacing the legacy
  /// bilinear knot tables; the tables remain as an old-profile fallback.
  llvm::StringMap<ContiguousFormula> te8Formulas;

  bool isValid() const { return maxElements > 0 && maxWarps > 0; }
};

/// Estimated completion cost of one contiguous access, in nanoseconds.
/// Returns 0 when the access is outside the calibrated domain or the profile
/// carries no formula for the segment the access falls into.
double estimateContiguousCompletionNs(const ContiguousAccessStats &stats,
                                      const ContiguousFit &fit);

} // namespace mlir::ascend::contiguous_ldst

#endif
