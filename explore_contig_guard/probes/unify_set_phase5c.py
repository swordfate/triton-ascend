#!/usr/bin/env python3
"""Unified-set phase 5c: serial composition for reduction-fed store
chains (estimator layer).

When a pipelined-loop SIMD stage computes a reduction (r.reduction > 0),
the store consumes the reduction output - load -> reduction -> store is
a serial latency chain and the pipelined max composition is invalid.
E19 (exact reduce_sum body probe): steady 216-222 ns/iter = chain
TL(domLoad) + reduction + store-drain; E20: the in-kernel row-tier
stage price for this shape is 374 ns/iter while the max composition
gives 187.  Composition: perIter = max(loadService + r.reduction +
storeService, vectorService, issue).
"""
from pathlib import Path

c = Path('/root/work/triton-ascend/third_party/ascend/costmodel/lib/'
         'AscendModel/RouteModel/StageCostModels.cpp')
s = c.read_text()

OLD = """  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      const double vectorService =
          std::max(r.compute + r.dot + r.reduction + r.shuffle, r.issue);
      const double loadService =
          r.load + r.continuousLoadGroup + r.indirectLoadGroup + r.sharedLayout;
      const double storeService = r.store + r.continuousStoreGroup;
      // Model one source iteration as competing engine resources.
      return r.setup +
             count * std::max({loadService, storeService, vectorService,
                               vectorService + r.irregularTailMaterialization,
                               r.irregularMaskedTileFloor, r.atomic,
                               r.scalar + r.predicate});
    }
    return serial;"""
NEW = """  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      const double vectorService =
          std::max(r.compute + r.dot + r.reduction + r.shuffle, r.issue);
      const double loadService =
          r.load + r.continuousLoadGroup + r.indirectLoadGroup + r.sharedLayout;
      const double storeService = r.store + r.continuousStoreGroup;
      // Model one source iteration as competing engine resources.
      // Phase 5c (E19/E20): when the stage computes a reduction, its
      // output is what the body's stores consume - load -> reduction ->
      // store is a serial latency chain, so the reduction rides the
      // chain instead of the parallel vector service (reduce_sum
      // in-kernel 374 ns/iter vs max-composition 187).
      if (r.reduction > 0.0) {
        const double chain =
            loadService + storeService + r.reduction + r.compute +
            r.shuffle;
        return r.setup +
               count * std::max({chain, loadService, storeService,
                                 vectorService,
                                 vectorService + r.irregularTailMaterialization,
                                 r.irregularMaskedTileFloor, r.atomic,
                                 r.scalar + r.predicate});
      }
      return r.setup +
             count * std::max({loadService, storeService, vectorService,
                               vectorService + r.irregularTailMaterialization,
                               r.irregularMaskedTileFloor, r.atomic,
                               r.scalar + r.predicate});
    }
    return serial;"""
assert OLD in s, 'pipelined-loop composition not found'
s = s.replace(OLD, NEW)
c.write_text(s)
print('phase 5c serial composition wired')
