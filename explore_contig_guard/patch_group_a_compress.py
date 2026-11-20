#!/usr/bin/env python3
"""Group_a compression code patch: Q/cT structural formulas."""
from pathlib import Path

BASE = Path('/root/work/triton-ascend/third_party/ascend/costmodel')
HDR = BASE / 'include/AscendModel/RouteModel/ContiguousLdStV2CostModel.h'
STG = BASE / 'lib/AscendModel/RouteModel/StageCostModels.cpp'
SSM = BASE / 'lib/AscendModel/RouteModel/SimdSimtCostModel.cpp'

# ---------- header: params field ----------
s = HDR.read_text()
old = '''  // keys: "<direction>/<dtype>/<bytes>/<residency>"  (engine = this profile)
  std::map<std::string, ContiguousV2Group> groupA;'''
new = '''  // keys: "<direction>/<dtype>/<bytes>/<residency>"  (engine = this profile)
  std::map<std::string, ContiguousV2Group> groupA;
  // 2026-11-17 group_a compression: per-direction/residency cTransfer
  // ratio (cTransfer = kc * marginal; Q is structural, groupAQueues
  // below).  keys: "load/warm", "load/cold", "store/warm", "store/cold".
  std::map<std::string, double> groupACtKc = {
      {"load/warm", 1.231}, {"load/cold", 2.768},
      {"store/warm", 1.003}, {"store/cold", 1.021}};'''
assert old in s, 'params field anchor'
s = s.replace(old, new, 1)

# ---------- header: helpers ----------
old = '''/// Static description of one contiguous access stream.
struct ContiguousV2Access {'''
new = '''/// Structural Q (2026-11-17 group_a compression): the queue-depth knee
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
struct ContiguousV2Access {'''
assert old in s, 'helpers anchor'
s = s.replace(old, new, 1)

# ---------- header: evaluateContiguousV2Ns ----------
old = '''  const int64_t n = instances;
  double total;
  if (g.Q > 0 && n > g.Q)
    total = g.first + (g.Q - 1) * g.marginal + (n - g.Q) * g.cTransfer;
  else
    total = g.first + (n - 1) * g.marginal;
  return total;'''
new = '''  const int64_t n = instances;
  const int64_t q = groupAQueues(access.store, bytes, warm);
  if (n > q) {
    const double cT =
        groupACTransfer(params, access.store, warm, g.marginal);
    return g.first + (q - 1) * g.marginal + (n - q) * cT;
  }
  return g.first + (n - 1) * g.marginal;'''
assert old in s, 'evaluate anchor'
s = s.replace(old, new, 1)
HDR.write_text(s)
print('header patched')

# ---------- StageCostModels.cpp: Formula A consumer ----------
s = STG.read_text()
old = '''    } else {
      if (g.Q > 0 && n > g.Q)
        total = g.first + (g.Q - 1) * g.marginal + (n - g.Q) * g.cTransfer;
      else
        total = g.first + (n - 1) * g.marginal;
    }'''
new = '''    } else {
      // Q and cTransfer are structural (2026-11-17 group_a compression):
      // Q from the byte count per direction/residency, cTransfer =
      // kc * marginal.  The stored cell values are no longer read.
      const int64_t q = contiguous_v2::groupAQueues(
          first->store, bytes, llvm::StringRef(res) == "warm");
      if (n > q) {
        const double cT = contiguous_v2::groupACTransfer(
            profile.contiguousV2, first->store,
            llvm::StringRef(res) == "warm", g.marginal);
        total = g.first + (q - 1) * g.marginal + (n - q) * cT;
      } else {
        total = g.first + (n - 1) * g.marginal;
      }
    }'''
assert old in s, 'consumer anchor'
s = s.replace(old, new, 1)
STG.write_text(s)
print('stage consumer patched')

# ---------- SimdSimtCostModel.cpp: parse group_a_ct_kc (both engines) ----------
s = SSM.read_text()
old = '''      if (const auto *lb = v2->getObject("loop_b")) {'''
new = '''      if (const auto *ck = v2->getObject("group_a_ct_kc")) {
        for (const auto &entry : *ck) {
          if (const auto v = entry.second.getAsNumber())
            cv2.groupACtKc[entry.first.str()] = *v;
        }
      }
      if (const auto *lb = v2->getObject("loop_b")) {'''
count = s.count(old)
assert count == 2, 'loop_b anchor count %d' % count
s = s.replace(old, new)
SSM.write_text(s)
print('parser patched x2')
