#!/usr/bin/env python3
"""Phase 5b correction: the tiny-store chain is ADDITIVE on the envelope
(not a max floor).  Evidence: the row tier's memory bucket for the
reduce_sum family = envelope(106.8) + chain(TL(4KB) 90 + 97 x S)
= 296.55, and the rowwise-reduction composition adds the reduction
service on top (77.6) -> 374 ns/iter = the measured-validated stage
price.  A max() floor silently under-charges (187 < 296)."""
from pathlib import Path

c = Path('/root/work/triton-ascend/third_party/ascend/costmodel/lib/'
         'AscendModel/RouteModel/StageCostModels.cpp')
s = c.read_text()
OLD = """          const double chain =
              contiguous_v2::envelopeCurve(
                  profile.contiguousV2.mixBytes,
                  profile.contiguousV2.mixTL, double(domBytes)) +
              profile.contiguousV2.tinyStoreExcessNs * double(nStores);
          perIterEff = std::max(perIterEff, chain);"""
NEW = """          // Additive, not a max floor: the tiny dependent store's
          // drain serializes AFTER the envelope's max'd services, so
          // the chain rides on top (row-tier bucket for the reduce_sum
          // family = envelope 106.8 + chain 189.75 = 296.55 ns/iter,
          // measured-validated through the composition).
          const double chain =
              contiguous_v2::envelopeCurve(
                  profile.contiguousV2.mixBytes,
                  profile.contiguousV2.mixTL, double(domBytes)) +
              profile.contiguousV2.tinyStoreExcessNs * double(nStores);
          perIterEff += chain;"""
assert OLD in s, 'chain floor not found'
s = s.replace(OLD, NEW)
c.write_text(s)
print('chain floor -> additive')
