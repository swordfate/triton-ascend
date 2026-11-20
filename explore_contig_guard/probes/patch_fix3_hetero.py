#!/usr/bin/env python3
"""Patch fix3: drop the shape component of the straight-line group key.

E31 evidence (2026-11-19, clean card 3, profiler differencing):
  - hetero-shape load pairs dilute like homo pairs (4KB+8KB pair = 237 ns
    vs 382 ns summed singles; 3x4KB triple = 245 vs 233 from first+2*marg);
  - price = first(bucket of largest member) + sum(marginal(bucket of each
    other member)) tracks every in-table pair within +/-10%;
  - n=1 through the table first term tracks isolated single-shot better
    than the legacy analytic (4KB warm: first 160.1 vs analytic 129.75 vs
    measured 169-184).

Change (StageCostModels.cpp, mapContiguousV2Completion, straight-line
section): group by direction x dtype instead of direction x shape x mask.
  - partial-masked members keep the per-member active-byte analytical;
  - price = first(bucket(largest)) + sum(marginal(bucket(other))), kc-
    clamped at q(bucket(largest)) when n > q, x pAxisFactor as before;
  - n=1 also goes through the table first term;
  - any missing table cell falls the whole group back to the per-member
    analytical (identical to the previous miss path).
"""
import sys

PATH = ("third_party/ascend/costmodel/lib/AscendModel/RouteModel/"
        "StageCostModels.cpp")
START = ("  // ---- straight-line stages: Formula A per same-shape group"
         " ------------")
END = "\n}\n\nstatic void mapStridedShortMemoryCompletion("

NEW = r'''  // ---- straight-line stages: one group per direction x dtype ----------
  // E31 (2026-11-19): hetero-shape load pairs dilute like homo pairs
  // (4KB+8KB pair measured 237 ns vs 382 ns summed singles; 3x4KB triple
  // 245 vs 233 from first + 2*marginal), so the shape component of the
  // group key is dropped.  Price = first(bucket of the largest member) +
  // sum(marginal(bucket of each other member)), kc-clamped at the group
  // queue depth q(bucket of the largest member).  n=1 groups take the
  // table first term as well (E30/E31: table first tracks the isolated
  // single-shot better than the legacy analytic - 4KB warm load:
  // first 160.1 vs analytic 129.75 vs measured 169-184).  Any missing
  // table cell falls the whole group back to the per-member analytical.
  // Partial-masked members keep the per-member active-byte analytical:
  // the group_a cells were calibrated unmasked.
  std::map<std::pair<bool, int64_t>, std::vector<const StageMemoryAccess *>>
      dirGroups;
  std::vector<const StageMemoryAccess *> maskedMembers;
  for (const StageMemoryAccess *access : eligible) {
    const bool partialMasked = access->masked &&
                               access->activeElementRatio > 0.0 &&
                               access->activeElementRatio < 1.0;
    if (partialMasked)
      maskedMembers.push_back(access);
    else
      dirGroups[std::make_pair(access->store,
                               int64_t(access->elementBits / 8))]
          .push_back(access);
  }
  int64_t footprint = 0;
  for (const StageMemoryAccess *access : eligible)
    footprint += v2ElementCount(access) * (access->elementBits / 8);
  const char *res = footprint <= v2.l2CacheBytes ? "warm" : "cold";

  // Per-member legacy-consistent analytical (fallback + masked path).
  auto analyticMembers = [&](std::vector<const StageMemoryAccess *> &ms) {
    double ana = 0.0;
    for (const StageMemoryAccess *access : ms) {
      double b = double(v2ElementCount(access)) *
                 double(access->elementBits / 8);
      if (access->masked && access->activeElementRatio > 0.0 &&
          access->activeElementRatio < 1.0)
        b *= access->activeElementRatio;
      if (simd) {
        const bool isStore = access->store;
        const double decay = isStore ? 1.0 : v2WaveDecayLoad;
        ana += ((access->store ? profile.storeOperationSetupCycles
                               : profile.loadOperationSetupCycles) +
                b * (isStore ? 1.0 / profile.storeBytesPerCycle
                             : 1.0 / profile.loadBytesPerCycle)) *
               decay;
      } else {
        ana += double(llvm::divideCeil(v2ElementCount(access),
                                       int64_t(32))) /
               (access->store ? profile.storeWarpInstructionsPerCycle
                              : profile.loadWarpInstructionsPerCycle);
      }
    }
    if (!simd)
      ana *= v2WaveDecay;
    const double split = ana / double(ms.size());
    for (const StageMemoryAccess *access : ms) {
      claimed.push_back(access);
      const int64_t bmem =
          v2ElementCount(access) * (access->elementBits / 8);
      if (access->store) {
        resources.store += split;
        layout.storeBytes += bmem;
        layout.storeOperations += 1.0;
        layout.storeWarpInstructions +=
            llvm::divideCeil(v2ElementCount(access), int64_t(32));
        layout.storeShapes[std::make_pair(bmem, v2ElementCount(access))] +=
            1.0;
      } else {
        resources.load += split;
        layout.loadBytes += bmem;
        layout.loadOperations += 1.0;
        layout.loadWarpInstructions +=
            llvm::divideCeil(v2ElementCount(access), int64_t(32));
        layout.loadShapes[std::make_pair(bmem, v2ElementCount(access))] +=
            1.0;
      }
    }
  };
  for (const StageMemoryAccess *access : maskedMembers) {
    std::vector<const StageMemoryAccess *> one{access};
    analyticMembers(one);
  }

  for (auto &dirEntry : dirGroups) {
    auto &members = dirEntry.second;
    const bool isStore = dirEntry.first.first;
    const char *direction = isStore ? "store" : "load";
    const char *dtype = ebName(dirEntry.first.second);
    if (!dtype) {
      analyticMembers(members);
      continue;
    }
    const StageMemoryAccess *biggest = members.front();
    int64_t maxBytes = 0;
    for (const StageMemoryAccess *access : members) {
      const int64_t b = v2ElementCount(access) * (access->elementBits / 8);
      if (b > maxBytes) {
        maxBytes = b;
        biggest = access;
      }
    }
    auto cellOf = [&](int64_t bytes) {
      int64_t bucket = 1024;
      double bestDist = 1e30;
      for (int64_t b : {32, 64, 128, 256, 512, 1024, 4096, 16384}) {
        const double d = std::fabs(std::log2(std::max(1.0, double(bytes))) -
                                   std::log2((double)b));
        if (d < bestDist) {
          bestDist = d;
          bucket = b;
        }
      }
      const std::string k = std::string(direction) + "/" + dtype + "/" +
                            std::to_string(bucket) + "/" + res;
      auto itc = v2.groupA.find(k);
      if (itc == v2.groupA.end() && !simd)
        itc = v2.groupA.find(k + "/w" +
                             std::to_string(profile.activeWarpCount));
      return itc;
    };
    auto itFirst = cellOf(maxBytes);
    if (itFirst == v2.groupA.end()) {
      analyticMembers(members);
      continue;
    }
    const int64_t n = static_cast<int64_t>(members.size());
    const int64_t q = contiguous_v2::groupAQueues(
        isStore, maxBytes, llvm::StringRef(res) == "warm");
    double total;
    bool fitted = true;
    if (n <= q) {
      total = itFirst->second.first;
      for (const StageMemoryAccess *access : members) {
        if (access == biggest)
          continue;
        auto itc = cellOf(v2ElementCount(access) *
                          (access->elementBits / 8));
        if (itc == v2.groupA.end()) {
          fitted = false;
          break;
        }
        total += itc->second.marginal;
      }
      if (!fitted) {
        analyticMembers(members);
        continue;
      }
    } else {
      const double cT = contiguous_v2::groupACTransfer(
          profile.contiguousV2, isStore, llvm::StringRef(res) == "warm",
          itFirst->second.marginal);
      total = itFirst->second.first + (q - 1) * itFirst->second.marginal +
              (n - q) * cT;
    }
    const double pFac = contiguous_v2::pAxisFactor(
        profile.contiguousV2, isStore ? "storeonly" : "loadsum",
        simd ? "simd" : "simt", res, programCountHint);
    const double ns = total * pFac * kCyclesPerNano;
    const double split = ns / double(n);
    for (const StageMemoryAccess *access : members) {
      claimed.push_back(access);
      const int64_t bmem =
          v2ElementCount(access) * (access->elementBits / 8);
      if (access->store) {
        resources.store += split;
        layout.storeBytes += bmem;
        layout.storeOperations += 1.0;
        layout.storeWarpInstructions +=
            llvm::divideCeil(v2ElementCount(access), int64_t(32));
        layout.storeShapes[std::make_pair(bmem, v2ElementCount(access))] +=
            1.0;
      } else {
        resources.load += split;
        layout.loadBytes += bmem;
        layout.loadOperations += 1.0;
        layout.loadWarpInstructions +=
            llvm::divideCeil(v2ElementCount(access), int64_t(32));
        layout.loadShapes[std::make_pair(bmem, v2ElementCount(access))] +=
            1.0;
      }
    }
  }
'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PATH
    s = open(path).read()
    i = s.find(START)
    j = s.find(END)
    assert i >= 0, "start anchor not found"
    assert j > i, "end anchor not found"
    s2 = s[:i] + NEW + s[j:]
    open(path, "w").write(s2)
    print("patched %s: removed %d chars, inserted %d" %
          (path, j - i, len(NEW)))


main()
