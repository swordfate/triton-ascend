#!/usr/bin/env python3
"""Unified-set phase 5b: reduction-epilogue chain floor (E19).

When a loop body computes a REDUCTION and stores tiny values, the
store consumes the reduction output - the chain load-return ->
reduction -> store lands on the critical path and the body is
latency-bound.  E19 (exact reduce_sum body: 2KB load + 32B bias load
+ tl.sum + 32B store) measured steady-state 216-222 ns/iter = memory
chain 187 (TL(4KB-clamp) 90 + 97 x S) + reduction compute ~29 charged
by the compute path.  187 ns/iter x 8 = 1.496us = the row tier's
stage price (1.497) - the formula reproduces the measured row.

Discriminator: work.reductionOperations / shuffleLaneSteps > 0 (the
masked-element-0 store shape, E18b, has no reduction and pipelines -
no floor there).
"""
from pathlib import Path

c = Path('/root/work/triton-ascend/third_party/ascend/costmodel/lib/'
         'AscendModel/RouteModel/StageCostModels.cpp')
s = c.read_text()

OLD = """        if (maxStoreChunk > 0 && maxStoreChunk <= 128 && nStores >= 1) {
          const int64_t cap =
              domBytes >= int64_t(
                              profile.contiguousV2.tinyStoreCapacity2Bytes)
                  ? 2
                  : 1;
          perIterEff +=
              std::max(0.0, double(nStores - cap)) *
              profile.contiguousV2.tinyStoreExcessNs;
        }
      }"""
NEW = """        if (maxStoreChunk > 0 && maxStoreChunk <= 128 && nStores >= 1) {
          const int64_t cap =
              domBytes >= int64_t(
                              profile.contiguousV2.tinyStoreCapacity2Bytes)
                  ? 2
                  : 1;
          perIterEff +=
              std::max(0.0, double(nStores - cap)) *
              profile.contiguousV2.tinyStoreExcessNs;
        }
        // Phase 5b (E19): reduction-fed tiny stores - the store consumes
        // the reduction output, so load-return -> reduction -> store
        // serializes and the body is latency-bound.  Measured on the
        // exact reduce_sum body (2KB load + 32B bias + tl.sum + 32B
        // store): steady 216-222 ns/iter = memory chain TL(domLoad) +
        // 97 x S (187 for this shape = the row tier's per-iter price
        // 1.497us/8) + reduction compute charged by the compute path.
        // Bodies without a reduction (masked element-0 stores, E18b)
        // pipeline - no floor.
        if (maxStoreChunk > 0 && maxStoreChunk <= 128 &&
            (work.reductionOperations > 0.0 ||
             work.shuffleLaneSteps > 0.0)) {
          const double chain =
              contiguous_v2::envelopeCurve(
                  profile.contiguousV2.mixBytes,
                  profile.contiguousV2.mixTL, double(domBytes)) +
              profile.contiguousV2.tinyStoreExcessNs * double(nStores);
          perIterEff = std::max(perIterEff, chain);
        }
      }"""
assert OLD in s, 'phase-4 excess block not found'
s = s.replace(OLD, NEW)
c.write_text(s)
print('phase 5b chain floor wired')
