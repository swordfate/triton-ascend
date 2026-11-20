#!/usr/bin/env python3
"""E27e final: loop-body pricing = per-n curves + joint issue floor +
per-line mixed tax (fit v6 model), replacing the E17f envelope entirely.

- header: curve/fit structs + helpers + params member
- parser: loop_envelope_fit (curves/steps/taxes) at both engine sites
- stage: simd + simt loop blocks replaced (fit-only; Tier 3 fallback)
- JSON: loop_envelope_fit built from /tmp/loopfit6_* (fit v6 output)
"""
import json
from pathlib import Path

ROOT = Path("/root/work/triton-ascend")
H = ROOT / ("third_party/ascend/costmodel/include/AscendModel/RouteModel/"
            "ContiguousLdStV2CostModel.h")
PARSE = ROOT / ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
                "SimdSimtCostModel.cpp")
STAGE = ROOT / ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
                "StageCostModels.cpp")
JPATH = ROOT / ("third_party/ascend/costmodel/profiles/simd_simt/"
                "ascend_950pr_simd_simt_v1.json")
INST_J = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/"
              "_C/ascend/costmodel_profiles/simd_simt/"
              "ascend_950pr_simd_simt_v1.json")
DTYPES = ("int8", "fp16", "fp32", "int64")

H_STRUCTS = '''
/// E27e fitted loop-body model (per engine profile, per dtype).  Per-iter:
///   L = curve_nL(lB)   per-nL measured transfer curve (piecewise-linear
///                      through the sweep points of that stream count)
///   S = curve_nS(sB)   per-nS store curve (simd measured; simt = E17i
///                      drain-rate line data)
///   floor = max(stepL[nL], stepS[nS], stepT[nL+nS])
///           issue-throttle floors; stepT fitted on small-byte MIXED
///           bodies (the joint descriptor-issue wall)
///   base    = max(floor, L, S)              (simd)
///           = L + S                         (simt: E17g additive)
///   mixed line tax (simd, both directions, non-tiny stores):
///             base += cL*(lB/128) + cS*(sB/128)
///   tiny-store excess/chain delta tables on the load-only base
/// This REPLACES the retired E17f envelope + tiny-store excess + TL-chain
/// (simd) and the E17h2/E17i line tables (simt).  In-sample MAPE of the
/// fit: simd 6.9-12.4% (fp32 6.9%, mix 5.3%); probe e27b_fit2.py.
struct ContiguousV2Curve {
  std::vector<double> b;   // bytes knots
  std::vector<double> v;   // ns at knots
};

struct ContiguousV2LoopFitDtype {
  std::map<int64_t, ContiguousV2Curve> loadCurves;   // nL -> curve
  std::map<int64_t, ContiguousV2Curve> storeCurves;  // nS -> curve
  std::map<int64_t, double> stepL;                   // issue floors
  std::map<int64_t, double> stepS;
  std::map<int64_t, double> stepT;                   // joint (nL+nS) floor
  double cL = 0.0;                                   // per-load-line tax
  double cS = 0.0;                                   // per-store-line tax
  bool additive = false;                             // simt: L + S
  // dominant-load per-access bytes -> [(nS, delta ns), ...]
  std::map<int64_t, std::vector<std::pair<double, double>>> excess;
  std::map<int64_t, std::vector<std::pair<double, double>>> chain;
};

struct ContiguousV2LoopFit {
  bool enabled = false;
  std::map<std::string, ContiguousV2LoopFitDtype> dtypes;
};

struct ContiguousV2Params {'''

H_MEMBER = '''  // E27e fitted loop-body model (per-dtype); the only loop tier
  ContiguousV2LoopFit loopFit;
  // resident program count for the current kernel (grid size)
  int64_t programCountHint = 1;'''

H_HELPERS = '''
/// Piecewise-linear through a curve's knots; flat below the first, slope
/// -extended beyond the last (single-point curves are flat).
inline double loopFitCurveNs(const ContiguousV2Curve &c, double b) {
  const size_t n = std::min(c.b.size(), c.v.size());
  if (n == 0 || b <= 0.0)
    return 0.0;
  if (b <= c.b[0])
    return c.v[0];
  if (b >= c.b[n - 1]) {
    if (n < 2)
      return c.v[n - 1];
    const double slope = (c.v[n - 1] - c.v[n - 2]) /
                         std::max(1.0, c.b[n - 1] - c.b[n - 2]);
    return c.v[n - 1] + slope * (b - c.b[n - 1]);
  }
  for (size_t i = 0; i + 1 < n; ++i) {
    if (c.b[i] <= b && b <= c.b[i + 1]) {
      const double t = (b - c.b[i]) / std::max(1e-9, c.b[i + 1] - c.b[i]);
      return c.v[i] + t * (c.v[i + 1] - c.v[i]);
    }
  }
  return c.v[n - 1];
}

/// Value at stream count n: exact node, log2-lerp between bracketing
/// nodes, scaled beyond the ends.
inline double loopFitCurveNsLerp(
    const std::map<int64_t, ContiguousV2Curve> &m, int64_t n, double b) {
  if (m.empty() || n <= 0 || b <= 0.0)
    return 0.0;
  auto hit = m.find(n);
  if (hit != m.end())
    return loopFitCurveNs(hit->second, b);
  auto lo = m.lower_bound(n);
  if (lo == m.end()) {
    auto last = std::prev(m.end());
    return loopFitCurveNs(last->second, b) * n / last->first;
  }
  if (lo == m.begin())
    return loopFitCurveNs(lo->second, b) * n / lo->first;
  auto prev = std::prev(lo);
  const double w = (std::log2((double)n) - std::log2((double)prev->first)) /
                   (std::log2((double)lo->first) -
                    std::log2((double)prev->first));
  return (loopFitCurveNs(prev->second, b) +
          w * (loopFitCurveNs(lo->second, b) -
               loopFitCurveNs(prev->second, b)));
}

/// Piecewise-linear in nS with end-slope extrapolation (delta tables).
inline double loopFitSeriesNs(
    const std::vector<std::pair<double, double>> &pts, double n) {
  if (pts.empty())
    return 0.0;
  if (n <= pts.front().first)
    return pts.front().second * n / std::max(1.0, pts.front().first);
  for (size_t i = 1; i < pts.size(); ++i) {
    if (n <= pts[i].first) {
      const double w = (n - pts[i - 1].first) /
                       std::max(1e-9, pts[i].first - pts[i - 1].first);
      return pts[i - 1].second +
             w * (pts[i].second - pts[i - 1].second);
    }
  }
  const auto &a = pts[pts.size() - 2];
  const auto &b = pts.back();
  const double slope = (b.second - a.second) /
                       std::max(1.0, b.first - a.first);
  return b.second + slope * (n - b.first);
}

/// log2-lerp between bracketing dominant-load byte keys of the delta
/// tables, then piecewise-linear in nS.
inline double loopFitDomNs(
    const std::map<int64_t,
                   std::vector<std::pair<double, double>>> &tab,
    double dom, double n) {
  if (tab.empty())
    return 0.0;
  auto hi = tab.lower_bound((int64_t)std::max(1.0, dom));
  if (hi == tab.end())
    return loopFitSeriesNs(std::prev(tab.end())->second, n);
  if (hi == tab.begin())
    return loopFitSeriesNs(hi->second, n);
  auto lo = std::prev(hi);
  const double w =
      (std::log2(std::max(1.0, dom)) - std::log2((double)lo->first)) /
      (std::log2((double)hi->first) - std::log2((double)lo->first));
  return loopFitSeriesNs(lo->second, n) +
         w * (loopFitSeriesNs(hi->second, n) -
              loopFitSeriesNs(lo->second, n));
}

/// Per-iteration ns of one loop body from the fitted model (fit v6).
inline double loopFitPerIter(const ContiguousV2LoopFitDtype &f,
                             int64_t nL, double lB, int64_t nS,
                             double sB, double dom, bool tiny,
                             bool redFed) {
  const double L =
      nL > 0 ? loopFitCurveNsLerp(f.loadCurves, nL, lB) : 0.0;
  const double S =
      nS > 0 ? loopFitCurveNsLerp(f.storeCurves, nS, sB) : 0.0;
  if (f.additive) {
    // E17g: simt load and store streams are additive (no overlap).
    double per = L + S;
    if (nS > 0 && tiny) {
      double lbase = L;
      auto it = f.stepL.find(nL);
      if (it != f.stepL.end())
        lbase = std::max(lbase, it->second);
      auto itt = f.stepT.find(nL);
      if (itt != f.stepT.end())
        lbase = std::max(lbase, itt->second);
      per = lbase + loopFitDomNs(redFed ? f.chain : f.excess, dom,
                                 double(nS));
    }
    return per;
  }
  double floor = 0.0;
  auto itl = f.stepL.find(nL);
  if (itl != f.stepL.end())
    floor = std::max(floor, itl->second);
  auto its = f.stepS.find(nS);
  if (its != f.stepS.end())
    floor = std::max(floor, its->second);
  auto itt = f.stepT.find(nL + nS);
  if (itt != f.stepT.end())
    floor = std::max(floor, itt->second);
  double per = std::max(floor, std::max(L, S));
  if (nL > 0 && nS > 0)
    per += f.cL * lB / 128.0 + f.cS * sB / 128.0;
  if (nS > 0 && tiny) {
    // delta tables were fitted against the load-only base
    double lbase = std::max(floor, L);
    auto it = f.stepT.find(nL);
    if (it != f.stepT.end())
      lbase = std::max(lbase, it->second);
    per = lbase + loopFitDomNs(redFed ? f.chain : f.excess, dom,
                               double(nS));
  }
  return per;
}

/// P-axis overlap factor: per-program price / price at P=1 (log2-P lerp,
/// clamped at the table ends).'''

PARSE_SNIP = '''      if (const auto *lf = v2->getObject("loop_envelope_fit")) {
        cv2.loopFit.enabled =
            reader.optionalNumber(*lf, "enabled", 0.0) != 0.0;
        if (const auto *dts = lf->getObject("dtypes")) {
          for (const auto &dentry : *dts) {
            const auto *dobj = dentry.second.getAsObject();
            if (!dobj)
              continue;
            contiguous_v2::ContiguousV2LoopFitDtype fd;
            fd.cL = reader.optionalNumber(*dobj, "c_l", 0.0);
            fd.cS = reader.optionalNumber(*dobj, "c_s", 0.0);
            fd.additive = reader.optionalNumber(*dobj, "additive", 0.0) != 0.0;
            for (const auto &tab : {std::pair<const char *,
                  std::map<int64_t, double> *>("step_l", &fd.stepL),
                  std::pair<const char *,
                  std::map<int64_t, double> *>("step_s", &fd.stepS),
                  std::pair<const char *,
                  std::map<int64_t, double> *>("step_t", &fd.stepT)}) {
              if (const auto *so = dobj->getObject(tab.first)) {
                for (const auto &e : *so) {
                  int64_t k = 0;
                  if (StringRef(e.first.str()).getAsInteger(10, k))
                    continue;
                  if (const auto n = e.second.getAsNumber())
                    (*tab.second)[k] = *n;
                }
              }
            }
            for (const auto &cur : {std::pair<const char *,
                  std::map<int64_t, ContiguousV2Curve> *>(
                  "load_curves", &fd.loadCurves),
                  std::pair<const char *,
                  std::map<int64_t, ContiguousV2Curve> *>(
                  "store_curves", &fd.storeCurves)}) {
              if (const auto *co = dobj->getObject(cur.first)) {
                for (const auto &e : *co) {
                  int64_t k = 0;
                  if (StringRef(e.first.str()).getAsInteger(10, k))
                    continue;
                  const auto *o = e.second.getAsObject();
                  if (!o)
                    continue;
                  ContiguousV2Curve c;
                  if (const auto *ba = o->get("b"))
                    if (auto ja = ba->getAsArray())
                      for (const auto &x : *ja)
                        if (const auto v = x.getAsNumber())
                          c.b.push_back(*v);
                  if (const auto *va = o->get("ns"))
                    if (auto ja = va->getAsArray())
                      for (const auto &x : *ja)
                        if (const auto v = x.getAsNumber())
                          c.v.push_back(*v);
                  if (!c.b.empty() && c.b.size() == c.v.size())
                    (*cur.second)[k] = c;
                }
              }
            }
            for (const auto &tab : {std::pair<const char *,
                  std::map<int64_t,
                  std::vector<std::pair<double, double>>> *>(
                  "excess", &fd.excess),
                  std::pair<const char *,
                  std::map<int64_t,
                  std::vector<std::pair<double, double>>> *>(
                  "chain", &fd.chain)}) {
              if (const auto *xo = dobj->getObject(tab.first)) {
                for (const auto &e : *xo) {
                  int64_t k = 0;
                  if (StringRef(e.first.str()).getAsInteger(10, k))
                    continue;
                  auto arr = e.second.getAsArray();
                  if (!arr)
                    continue;
                  std::vector<std::pair<double, double>> pts;
                  for (const auto &p : *arr) {
                    auto pa = p.getAsArray();
                    if (!pa || pa->size() < 2)
                      continue;
                    const auto n0 = (*pa)[0].getAsNumber();
                    const auto n1 = (*pa)[1].getAsNumber();
                    if (n0 && n1)
                      pts.emplace_back(*n0, *n1);
                  }
                  if (!pts.empty())
                    (*tab.second)[k] = pts;
                }
              }
            }
            cv2.loopFit.dtypes[dentry.first.str()] = fd;
          }
        }
      }
      if (const auto *ck = v2->getObject("group_a_ct_kc")) {'''

FIT_LOOKUP = '''      int64_t domEB = 0;
      for (const StageMemoryAccess *access : eligible) {
        const int64_t b =
            v2ElementCount(access) * (access->elementBits / 8);
        const bool want = loadBytes > 0 ? !access->store : access->store;
        if (want && b > domEB)
          domEB = access->elementBits / 8;
      }
      const char *domDtype = ebName(domEB);
      auto itf = domDtype
                     ? profile.contiguousV2.loopFit.dtypes.find(domDtype)
                     : profile.contiguousV2.loopFit.dtypes.end();
      if (itf != profile.contiguousV2.loopFit.dtypes.end()) {
        int64_t maxChunk = 0;
        for (const StageMemoryAccess *access : eligible)
          if (access->store)
            maxChunk = std::max<int64_t>(
                maxChunk,
                v2ElementCount(access) * (access->elementBits / 8));
        const bool tiny = maxChunk > 0 && maxChunk <= 128;
        const bool redFed = work.reductionOperations > 0.0 ||
                            work.shuffleLaneSteps > 0.0;
'''

SIMD_NEW = '''    // E27e fitted loop-body model (the only simd loop-body tier):
    // per-nL/nS measured transfer curves + joint issue floor
    // (stepT[nL+nS], the mixed-descriptor wall) + per-128B-line mixed tax
    // + tiny-store excess/chain delta tables.  In-sample MAPE 6.9-12.4%
    // (fp32 6.9%, mix 5.3%).  The E17f envelope, tiny-store excess and
    // TL-chain formulas are retired; a dtype without a fit falls through
    // to the Tier 3 per-op tier below.
    if (simd && profile.contiguousV2.loopFit.enabled &&
        (loadBytes > 0.0 || storeBytes > 0.0) &&
        stage.iterationCount >= 1) {
''' + FIT_LOOKUP + '''        const double perIter = contiguous_v2::loopFitPerIter(
            itf->second, nLoads, double(loadBytes), nStores,
            double(storeBytes), double(domBytes), tiny, redFed) *
            v2WaveDecay;
        const bool allStoresEnv = std::all_of(
            eligible.begin(), eligible.end(),
            [](const StageMemoryAccess *a) { return a->store; });
        if (allStoresEnv)
          resources.store += perIter * kCyclesPerNano;
        else
          resources.load += perIter * kCyclesPerNano;
        claimAll();
        return;
      }
      // dtype without a fitted model: fall through to the Tier 3
      // per-op tier below (the envelope path is retired).
    }
'''

SIMT_NEW = '''    // E27e fitted loop-body model (simt; the only simt loop-body tier):
    // per-nL load curves (drain-forced probe) + E17i drain-rate store
    // curve, additive composition (E17g: no load/store overlap); tiny
    // excess/chain tables.  A dtype without a fit falls through to the
    // Tier 3 per-op tier below (the line-table path is retired).
    if (!simd && profile.contiguousV2.loopFit.enabled &&
        (loadBytes > 0.0 || storeBytes > 0.0) &&
        stage.iterationCount >= 1) {
''' + FIT_LOOKUP + '''        const double perIter = contiguous_v2::loopFitPerIter(
            itf->second, nLoads, double(loadBytes), nStores,
            double(storeBytes), double(domBytes), tiny, redFed);
        if (loadBytes == 0.0)
          resources.store += perIter * kCyclesPerNano;
        else
          resources.load += perIter * kCyclesPerNano;
        claimAll();
        return;
      }
      // dtype without a fitted model: fall through to the Tier 3
      // per-op tier below (the line-table path is retired).
    }
'''


def main():
    s = H.read_text()
    if "ContiguousV2LoopFitDtype" not in s:
        if "#include <vector>" not in s:
            s = s.replace("#include <string>",
                          "#include <string>\n#include <vector>", 1)
        s = s.replace("struct ContiguousV2Params {", H_STRUCTS, 1)
        s = s.replace(
            "  // resident program count for the current kernel (grid size)"
            "\n  int64_t programCountHint = 1;", H_MEMBER, 1)
        anchor = ("/// P-axis overlap factor: per-program price / price at "
                  "P=1 (log2-P lerp,\n/// clamped at the table ends).")
        assert anchor in s, "p_axis anchor missing"
        s = s.replace(anchor, H_HELPERS, 1)
        H.write_text(s)
        print("header patched")
    else:
        print("header already has loopfit")

    s = PARSE.read_text()
    if "loop_envelope_fit" not in s:
        n = s.count(
            '      if (const auto *ck = v2->getObject("group_a_ct_kc")) {')
        assert n == 2, "expected 2 ct_kc sites, got %d" % n
        s = s.replace(
            '      if (const auto *ck = v2->getObject("group_a_ct_kc")) {',
            PARSE_SNIP, 2)
        PARSE.write_text(s)
        print("parser patched at 2 sites")
    else:
        print("parser already patched")

    s = STAGE.read_text()
    if "loopFitPerIter" not in s:
        start = s.find(
            "    if (simd && profile.contiguousV2.mixEnvelopeEnabled &&")
        assert start >= 0, "simd loop block not found"
        endmark = "      claimAll();\n      return;\n    }\n"
        end = s.find(endmark, start)
        assert end > start
        end += len(endmark)
        s = s[:start] + SIMD_NEW + s[end:]
        start = s.find(
            "    if (!simd && profile.contiguousV2.simtLineEnvelopeEnabled &&")
        assert start >= 0, "simt loop block not found"
        end = s.find(endmark, start)
        assert end > start
        end += len(endmark)
        s = s[:start] + SIMT_NEW + s[end:]
        assert "mixEnvelopePerIter" not in s
        assert "simtLineEnvelopePerIter" not in s
        STAGE.write_text(s)
        print("stage patched (fit-only)")
    else:
        print("stage already patched")

    # ---- JSON from fit v6 output ----
    secs = {}
    for mode, eng in (("simd", "simd"), ("simt_only", "simt")):
        dt = {}
        for dtype in DTYPES:
            p = Path("/tmp/loopfit6_%s_%s.json" % (mode, dtype))
            if not p.exists():
                print("MISSING", p)
                continue
            sec = json.loads(p.read_text())
            dt[dtype] = {
                "load_curves": sec["load_curves"],
                "store_curves": sec["store_curves"],
                "step_l": sec["step_l"], "step_s": sec["step_s"],
                "step_t": sec["step_t"],
                "c_l": sec["c_l"], "c_s": sec["c_s"],
                "additive": sec["additive"],
                "excess": sec["excess"], "chain": sec["chain"],
                "mape_all_M3": sec["MAPE_all"]["M3"],
                "mape_mix_M3": sec["MAPE_mix"]["M3"],
            }
        if dt:
            secs[eng] = {
                "enabled": 1,
                "source": ("E27e fit v6: per-nL/nS measured transfer curves "
                           "+ joint issue floor stepT[nL+nS] + per-128B-line "
                           "mixed tax (simd) / additive (simt, E17g); "
                           "simt store curve from the E17i drain-rate line "
                           "data; probe e27b_fit2.py unpaired campaigns"),
                "dtypes": dt,
            }
    for path in (JPATH, INST_J):
        j = json.loads(path.read_text())
        for eng, sec in secs.items():
            j[eng]["memory"]["contiguous_ldst_v2"]["loop_envelope_fit"] = sec
        path.write_text(json.dumps(j, indent=1, ensure_ascii=False))
        json.loads(path.read_text())
        print("wrote", path)


main()
