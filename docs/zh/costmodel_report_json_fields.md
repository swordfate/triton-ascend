# costmodel JSON Report 字段填充时序

> 对应 simt_costmodel 分支 `SimdSimtCostModel.cpp` + `SelectSimdSimtCostModel.cpp`

---

## 总流程

```
Python: _run_cpp_simd_simt_costmodel()
  → C++: SelectSimdSimtCostModelPass::runOnOperation()
    ├─ buildMixedSimtAnchorPlan()           ← Phase 0 (独立调用)
    ├─ analyzeSimdSimtCandidates(module, anchorPlan)  ← 传入同一个 plan
    │   ├─ analyzeSimdSimtFeatures(module, anchorPlan) ← Phase 1
    │   └─ estimateSimdSimtCandidates(features)       ← Phase 2-8
    ├─ 写入额外字段到 JSON                   ← Phase 9
    ├─ (如果 mixed) materializeSimtAnchorPlan(module, anchorPlan)  ← 同一个 plan!
    └─ appendJSONLine()                      ← 写文件
```

关键设计：`buildMixedSimtAnchorPlan` 在 `runOnOperation` 中调用，返回的 `SimtAnchorPlan` 同时传给 scoring 和 materialization，保证特征提取、评分、materialize 三个阶段看到的是**同一个 anchor 集合**。

下面按代码执行顺序，逐个解释每个字段。

---

## Phase 0: buildMixedSimtAnchorPlan — 构建 Anchor Plan

**文件**: `SimtAnchorAnalysis.cpp:545-589`, `SelectSimdSimtCostModel.cpp:113-114`

Anchor Plan 是 v10 的核心机制：在评分开始之前，先确定"哪些 op 是 SIMT 的候选"。这个 plan 是不可变的，后续所有阶段共享。

### 0.1 入口

```cpp
// SelectSimdSimtCostModel.cpp line 113-114
SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, options.compileOn91095);
auto reportOr = analyzeSimdSimtCandidates(module, anchorPlan, options);
//                  ↑ 传入同一个 anchorPlan
```

### 0.2 buildMixedSimtAnchorPlan 完整流程

```cpp
// SimtAnchorAnalysis.cpp line 545-589
SimtAnchorPlan buildMixedSimtAnchorPlan(ModuleOp module, bool compileOn91095) {
  SimtAnchorPlan plan;

  // 步骤 1: PreOrder walk，对每个 op 调用 analyzeAnchor
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto descriptor = analyzeAnchor(op, compileOn91095);  // ← 核心：识别 anchor
    if (!descriptor)
      return WalkResult::advance();  // 不识别 → 递归进入子 op
    plan.anchors.push_back(std::move(*descriptor));
    return WalkResult::skip();       // 识别到了 → 跳过子 op
  });

  // 步骤 2: 聚合 kernel 级别的 lowerability
  bool anyMixedNative = false, mixedBlocked = false;
  for (const auto &anchor : plan.anchors) {
    // allSimd: 取所有 anchor 中最差的状态
    plan.kernelLowerability.allSimd = combineWholeKernelStatus(
        plan.kernelLowerability.allSimd, anchor.lowerability.allSimd);
    // allSimtOnly: 同上
    plan.kernelLowerability.allSimtOnly = combineWholeKernelStatus(
        plan.kernelLowerability.allSimtOnly, anchor.lowerability.allSimtOnly);
    // 收集 reasons
    append_range(kernelLowerability.allSimdReasons, anchor.lowerability.allSimdReasons);
    append_range(kernelLowerability.allSimtOnlyReasons, anchor.lowerability.allSimtOnlyReasons);
    append_range(kernelLowerability.mixedReasons, anchor.lowerability.mixedReasons);

    if (anchor.lowerability.mixed == Native)
      anyMixedNative = true;
    else if (anchor.lowerability.allSimd != Native)
      mixedBlocked = true;
  }

  // 步骤 3: 判断 mixed 是否可用
  if (plan.anchors.empty()) {
    kernelLowerability.mixed = Unsupported;  // 没有 anchor → mixed 不可用
  } else if (anyMixedNative && !mixedBlocked) {
    kernelLowerability.mixed = Native;       // 至少一个 native 且没有 blocked
  } else {
    kernelLowerability.mixed = mixedBlocked ? Unsupported : BackendConditional;
  }
  return plan;
}
```

### 0.3 combineWholeKernelStatus — 取最差状态

```cpp
// SimtAnchorAnalysis.cpp line 459-476
static CandidateLoweringStatus combineWholeKernelStatus(
    CandidateLoweringStatus lhs, CandidateLoweringStatus rhs) {
  // 严重程度排序: Native(0) < BackendConditional(1) < AliasesMixed(2) < Unsupported(3)
  return rank(lhs) >= rank(rhs) ? lhs : rhs;  // 取更严重的
}
```

举例：kernel 有 3 个 anchor，其中 1 个 Histogram（allSimd=Unsupported, rank=3），另外 2 个都是 Native（rank=0），则 kernel 级别的 allSimd = Unsupported（取最差）。

### 0.4 analyzeAnchor — 6 种锚点识别

**文件**: `SimtAnchorAnalysis.cpp:312-456`

对每个 op 按 `name` 分发：

```
tt.gather → DirectGather
  lowerability: allSimd=Native(默认), allSimt=BackendConditional, mixed=Native(默认)
  materializable = compileOn91095 && mixed==Native → true (on A5)

tt.histogram → Histogram
  验证: input 是 static rank-1 tensor of i8/i16/i32/i64
        result 是 rank-1 i32, inputElems>0, numBins>0
  不满足 → mixed=Unsupported (reasons: "histogram_requires_static_rank1_...")
  lowerability: allSimd=Unsupported, allSimt=Unsupported, mixed=Native(if valid)
  materializable = compileOn91095 && mixed==Native

tt.scan → PlainOneDimensionalCumsum
  调用 analyzePlainOneDimensionalCumsum() (line 45-86):
    要求: 只有一条轴 extent>1, body 中只有一个真正的 combine op
         (arith.addf 或 arith.addi), terminator 是 tt.scan.return
    提取: axisExtent, elementType, reverse
  lowerability: allSimd=AliasesMixed, allSimt=BackendConditional, mixed=Native
  如果 axisExtent≤64 → mixedReasons: "template_uses_small_register_path..."
  如果 dtype 不支持 → mixed=Unsupported
  materializable = compileOn91095 && mixed==Native

tt.atomic_rmw / tt.atomic_cas → TensorAtomic
  提取: updateElements, addressRank, valueType, offsetType, operation,
        hasMask, staticMaskActiveFraction, resultUsed,
        addressIsLaneVarying, addressDependsOnLoadedIndex, contention
  验证: supported type/operation 组合 (isSupportedAtomicType)
        f16/bf16 + resultUsed → Unsupported
  lowerability: allSimd=Native(默认), allSimt=BackendConditional, mixed=Native(if valid)
  如果 Unsupported → mixed=Unsupported
  materializable = compileOn91095 && mixed==Native

scf.for → TriangularSolveLoop (需满足 isTriangularSolveLoop 条件)
  调用 isTriangularSolveLoop() (line 239-301):
    条件 1: body 中有 tt.load (rank-1, shape[0]=16) → 向量加载
    条件 2: body 中有 tt.reduce (axis=0)            → 轴0规约
    条件 3: body 中有 arith.select                   → 掩码更新
    条件 4: iter_args 有 16×16 triangular state，或
            ≥1 个 sibling 循环有同样 load/reduce/select 模式
  lowerability: allSimd=Native(默认), allSimt=BackendConditional, mixed=Native(默认)
  materializable = compileOn91095 && mixed==Native

tt.load / tt.store → LoadedIndexDependentMemory (需满足条件)
  条件: isLoadedIndexDependentMemoryOp() (line 530-535):
    hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op)
  pointerDependsOnLoadedIndex() (line 96-128):
    从 memoryOp 的 pointer 开始 BFS:
    - 遇到 BlockArgument → 追踪 scf.for 的 iter_args
    - 遇到 tt.load 或 tt.gather → return true (找到了!)
    - 其他 op → 继续追踪 operands
  lowerability: allSimd=Native(默认), allSimt=BackendConditional, mixed=Native(默认)
  materializable = compileOn91095 && mixed==Native

不匹配以上任何一种 → 不识别为 anchor，继续处理子 op
```

### 0.5 simpleMixedLowerability — 默认 lowerability 模板

大多数 anchor 使用这个模板（line 304-309）：

```cpp
static CandidateLowerability simpleMixedLowerability(StringRef allSimtReason) {
  CandidateLowerability result;
  // allSimd 保持默认 Native
  // mixed 保持默认 Native
  result.allSimtOnly = BackendConditional;   // 纯 SIMT 需要后端验证
  result.allSimtOnlyReasons.push_back(allSimtReason);
  return result;
}
```

只有 Histogram 和 PlainOneDimensionalCumsum 不使用这个模板，它们有自己的特殊 lowerability。

### 0.6 写入的 JSON 字段

`buildMixedSimtAnchorPlan` 的产物通过 `analyzeSimdSimtFeatures` 写入 `features.simtAnchors`：

```
simt_anchors.count                     = materializable roots 数量
simt_anchors.recognized_count          = 所有识别出的 anchors 数量
simt_anchors.mechanism_kinds           = ["direct_gather", "plain_1d_cumsum", ...]
simt_anchors.kernel_lowerability       = {all_simd: {...}, all_simt_only: {...}, mixed_simd_simt: {...}}
  → 这三个值直接决定了 allSimdCandidateLegal / allSimtOnlyCandidateLegal / mixedCandidateLegal
```

---

## Phase 1: analyzeSimdSimtFeatures — 全 kernel + anchor 双统计

**文件**: `SimdSimtCostModel.cpp:1780-2208`

接收 Phase 0 构建的 `anchorPlan`，计算每个 op 的 `inAnchor` 标记，然后统计全 kernel 和 anchor 内两份数据。

```
line 1787  SimdSimtFeatureSummary features;
line 1788  initializeWorkMaps(features);
line 1789  initializeWorkMaps(features.simtAnchors);
          → 所有 features.* 字段初始化为 0/false

line 1790  anchorRoots = anchorPlan.materializableRoots()
line 1793  anchorSet(anchorRoots)
          → features.simtAnchors.recognizedCount = plan.anchors.size()
          → features.simtAnchors.count = anchorRoots.size()

line 1863  module.walk([&](Operation *op) { ... })
          → 遍历每个 op，按 op 类型填充:

          基础计数:
            features.loadOps, storeOps, reduceOps, scanOps, gatherOps
            features.dotOps, atomicOps, histogramOps, broadcastOps
            features.expandDimsOps, splatOps, addPtrOps
            features.arithOps, mathOps
            features.addOps, subOps, mulOps, divOps, maxOps, absOps
            features.expOps, logOps, cmpOps, selectOps, castOps, clampOps
            features.scalarOps, scalarLoadOps, scalarStoreOps
            features.vectorPtrSplatOps, vectorReduceToScalarOps

          加权数据 (被 loopMultiplier 放大):
            features.weightedOps[op], features.opElements[op]
            features.loadBytes, storeBytes
            features.loadWarpInstructions, storeWarpInstructions
            features.dotFlops, dotOutputElements, dotMNK

          结构特征:
            features.maxTensorRank, maxTensorNumel, maxElementBits
            features.maskTensorOps, maskRankSum, maskBroadcastOps
            features.pointerTensorOps, pointerUnstructuredDims
            features.laneDependentPointerOps
            features.loadedIndexDependentMemoryOps
            features.rowLocalReduceOps
            features.maxReduceAxisExtent, weightedReduceAxisElements
            features.uniqueMaskValues, uniqueMaskRankSum, predicateElements

          循环信息:
            features.staticLoopCount, staticLoopTripCountSum, staticLoopTripCountMax
            features.hasUnknownTripCount   ← scf.for trip count 静态不可解析

          标志位:
            features.hasExplicitScope, hasControlFlow
            features.hasDynamicShape

          → 同步填充 features.simtAnchors.* (只用 inAnchor=true 的 op):
            loadOps, storeOps, reduceOps, scanOps, gatherOps, dotOps,
            atomicOps, histogramOps, weightedOps, opElements,
            loadBytes, storeBytes, loadWarpInstructions, storeWarpInstructions,
            dotFlops, staticLoopCount, staticLoopTripCountSum, maskRankSum,
            pointerTensorOps, laneDependentPointerOps,
            loadedIndexDependentMemoryOps, maxReduceAxisExtent,
            weightedReduceAxisElements, uniqueMaskValues, uniqueMaskRankSum,
            predicateElements, maxTensorNumel, maxElementBits, hasControlFlow,
            capturedTensorCount/Bytes, escapingTensorCount/Bytes,
            mechanismKinds, tensorAtomics, histograms, plainCumsums

line 2192  features.scalarOps = addOps + subOps + ... + clampOps
line 2197  features.hasDot = dotOps > 0
line 2198  features.hasGather = gatherOps > 0
line 2199  features.hasAtomic = atomicOps > 0
line 2200  features.hasHistogram = histogramOps > 0
line 2201  features.hasScan = scanOps > 0
line 2202  features.rank1IndirectVectorReduce = (条件组合)
```

---

## Phase 2: estimateSimdSimtCandidates 开场 — profile 元数据

**文件**: `SimdSimtCostModel.cpp:2210-2255`

```
line 2218  loadCandidateProfile(profilePath)
            → 解析 selection profile JSON
            → 如果引用了 microbenchmark_profile，加载微基准数据
            → 提取 op throughput、structural penalty、coverage bounds

┌─────────────────────────────────────────────────────────┐
│ ① profile 元数据 (line 2224-2238)                        │
├─────────────────────────────────────────────────────────┤
│ report.profileVersion              = profile.profileVersion
│ report.profileTarget               = profile.target
│ report.actualTarget                = options.actualTarget
│ report.profileContentSha256        = profile.contentSha256
│ report.selectionProfileContentSha256
│ report.microbenchmarkProfileVersion
│ report.microbenchmarkProfileTarget
│ report.microbenchmarkProfileContentSha256
│ report.scoreUnit                   = "system_cycle_selection_score"
│ report.scoreScope                  = "per_program_ranking_proxy" (默认)
│ report.minimumConfidenceForDecision = profile.minimumConfidence ("low")
│ report.targetCompatible            = targetMatches(profile, actualTarget)
└─────────────────────────────────────────────────────────┘
```

```
line 2241  report.applicability = evaluateSimtApplicability(features, compileOn91095)
            → applicability.mechanismDetected      = recognizedCount > 0 || !mechanisms.empty()
            → applicability.targetSupported        = compileOn91095
            → applicability.materializable          = targetSupported && materializableCount > 0
            → applicability.recognizedAnchorCount    = simtAnchors.recognizedCount
            → applicability.materializableAnchorCount = simtAnchors.count
            → applicability.mechanisms              = simtAnchors.mechanismKinds
            → applicability.reasons (空 = 可用, 非空 = 不可用)
```

```
┌─────────────────────────────────────────────────────────┐
│ ② 候选合法性 (line 2243-2253)                            │
├─────────────────────────────────────────────────────────┤
│ report.allSimdCandidateLegal     = kernelLowerability.allSimd == Native
│ report.allSimtOnlyCandidateLegal = compileOn91095 && !hasExplicitScope
│                                     && kernelLowerability.allSimtOnly == Native
│ report.mixedCandidateLegal       = !hasExplicitScope && applicability.materializable
│                                     && kernelLowerability.mixed == Native
└─────────────────────────────────────────────────────────┘
```

```
line 2254  report.includeFeaturesInJSON = options.includeFeaturesInJSON
line 2255  report.marginRatio           = options.marginRatio
```

---

## Phase 3: Coverage 检查

**文件**: `SimdSimtCostModel.cpp:2257-2314`

```
line 2265  report.breakdown.irregularDensity = laneDependentPtrOps / pointerTensorOps

line 2268  rankingCalibrationCoverage(features, weightedReductions, dotFlops,
                                     profile, irregularDensity)
            → 返回 {covered: bool, domain: string}

           优先级匹配:
           1. hasTriangularSolve && shape/anchor 在限内 → "triangular_solve_loop"
           2. dotFlops > 0, ≤16384, irreg≥0.25 → "tiny_irregular_dot"
           3. rank1IndirectVectorReduce → "rank1_indirect_vector_reduction"
           4. 有 mask + reduction + 循环 → "masked_rowwise_reduction"
           5. 都不匹配 → "out_of_calibration_domain"

           立即拒绝条件:
           - hasDynamicShape → "dynamic_shape"
           - hasUnknownTripCount (且非 triangular) → "unknown_loop_trip_count"
```

```
┌─────────────────────────────────────────────────────────┐
│ ③ coverage 结果 (line 2271-2273)                         │
├─────────────────────────────────────────────────────────┤
│ report.calibrationCovered    = covered
│ report.calibrationDomain     = domain
│ report.selectionScoreValid   = covered  ← 跟 coverage 绑定
└─────────────────────────────────────────────────────────┘
```

```
line 2276  if (calibrationCovered):
            → 查找 domain 的 Event calibration:
            ┌─────────────────────────────────────────────────┐
            │ report.eventRouteCalibrationApplied   = true    │
            │ report.eventAllSimtOnlyValidated      = domain值 │
            │ report.eventMixedSimdSimtValidated    = domain值 │
            │ report.eventRouteCalibrationSource    = domain值 │
            │ report.eventRouteCalibrationConfidence = domain值│
            │ report.eventRouteScoreMultipliers = {           │
            │   allSimd, allSimt, mixed }                     │
            └─────────────────────────────────────────────────┘

            line 2298  如果 Event 验证了纯 SIMT:
                         → report.allSimtOnlyCandidateLegal 可能变为 true
            line 2303  如果 Event 验证不通过 mixed:
                         → report.mixedCandidateLegal = false
```

**⚡ 关键分叉点 — coverage 不通过 + auto 模式时提前返回**:

```
line 2308  if (!covered && !scoreOutsideCalibrationCoverage):
line 2310    report.gateReasons += "target_incompatible"  (如果不兼容)
line 2312    report.gateReasons += "selection_score_invalid"
line 2313    report.gatePassed = false
line 2314    return report
            → candidateCostsEvaluated = false
            → 所有分数为 0，decision_kind = null
```

---

## Phase 4: 资源成本评分 — compute / memory / dot / shuffle / predicate

**文件**: `SimdSimtCostModel.cpp:2317-2520`

```
line 2324  vectorWidth = simdVectorWidthBits / elementBits

line 2358-2385  逐 op 类型计算:
  ┌─────────────────────────────────────────────────────┐
  │ ④ op_breakdown (每个 op 类型的 SIMD/SIMT 周期)       │
  ├─────────────────────────────────────────────────────┤
  │ report.breakdown.simdOpSystemCycles[op]              │
  │   = ceil(elements / vectorWidth) / throughput        │
  │ report.breakdown.simtOpSystemCycles[op]              │
  │   = elements / scalarThroughput                      │
  │ report.breakdown.simdComputeCycles  += simdCycles    │
  │ report.breakdown.simtComputeCycles  += simtCycles    │
  └─────────────────────────────────────────────────────┘

line 2387-2409  anchor 分账:
  ┌─────────────────────────────────────────────────────┐
  │ mixed.partition 中的 compute 分账                    │
  ├─────────────────────────────────────────────────────┤
  │ mixedSimdRegularComputeCycles = simdCompute - anchorSimdCost
  │ mixedSimtAnchorComputeCycles  = anchorSimtCost       │
  └─────────────────────────────────────────────────────┘
```

```
line 2411-2417  SIMD memory:
  ┌─────────────────────────────────────────────────────┐
  │ ⑤ memory (SIMD 侧)                                   │
  ├─────────────────────────────────────────────────────┤
  │ report.breakdown.simdLoadCycles  = loadBytes / mte2Rate
  │ report.breakdown.simdStoreCycles = storeBytes / mte3Rate
  │ report.breakdown.simdMemoryCycles = max(load, store)
  │ mixed.partition.simdRegularMemory = regularBytes / simdRate
  └─────────────────────────────────────────────────────┘

line 2444-2453  SIMT memory:
  ┌─────────────────────────────────────────────────────┐
  │ memory (SIMT 侧)                                     │
  ├─────────────────────────────────────────────────────┤
  │ report.breakdown.simtLoadCycles  = warpInsts / rate  │
  │ report.breakdown.simtStoreCycles = warpInsts / rate  │
  │ report.breakdown.simtMemoryCycles = load + store     │
  │ mixed.partition.simtAnchorMemory = anchorWarp / rate │
  └─────────────────────────────────────────────────────┘
```

```
line 2464-2485  SIMT shuffle:
  report.breakdown.simtShuffleInstructions = reductions * ceil(N/32) * log2(32)
  report.breakdown.simtShuffleCycles       = shuffleInsts / shuffleRate
  mixed.partition.simtAnchorShuffle        = anchorShuffle / shuffleRate

line 2489-2500  SIMT predicate:
  report.breakdown.simtPredicateInstructions = maskRankSum * ceil(N/32)
  report.breakdown.simtPredicateCycles       = predicateInsts / predicateRate
  mixed.partition.simtAnchorPredicate        = anchorPredicate / predicateRate
```

```
line 2502-2523  Dot (MatMul):
  report.breakdown.simdDotCycles = setup + dotFlops / simdDotRate
  report.breakdown.simtDotCycles = setup + dotFlops / simtDotRate
  mixed.partition.simdRegularDot = regularDot / simdDotRate
  mixed.partition.simtAnchorDot  = anchorDot / simtDotRate
```

---

## Phase 5: 合成 Analytical Cycles

**文件**: `SimdSimtCostModel.cpp:2525-2543`

```
line 2525  report.breakdown.simdSetupCycles = profile.simdSetupCycles
line 2526  report.breakdown.simtSetupCycles = profile.simtSetupCycles

line 2525-2543:
  ┌─────────────────────────────────────────────────────┐
  │ ⑥ analytical_candidate_costs                         │
  ├─────────────────────────────────────────────────────┤
  │ simdIssuePayload = max(compute+dot, memory)          │
  │ simtIssuePayload = max(compute+shuffle+dot, memory)  │
  │                     + predicate                      │
  │                                                     │
  │ A_SIMD = setup + simdPayload * issueScale(8.0)       │
  │ A_SIMT = setup + simtPayload * issueScale(8.0)       │
  └─────────────────────────────────────────────────────┘
```

---

## Phase 6: Structural Penalty (SIMD-only)

**文件**: `SimdSimtCostModel.cpp:2545-2593`

```
line 2547  report.breakdown.tinyDotUnderfill = 1 - dotFlops/tinyDotFlopsMax

line 2558-2582  逐项计算:
  ┌───────────────────────────────────────────────────────────┐
  │ ⑦ structure.components + structure.penalty_ratio          │
  ├───────────────────────────────────────────────────────────┤
  │ "irregular_addressing"            = min(cap, density*per) │
  │ "mask_materialization"            = min(cap, rank*per)    │
  │ "reduction_lowering"              = min(cap, red*per)     │
  │ "static_loop_control"             = min(cap, loop*per)    │
  │ "control_flow"                    = 0.03 (if has)         │
  │ "tiny_dot_startup"                = tiny*(1-flops/max)    │
  │ "rank1_indirect_vector_reduction" = 0.75 (if has)         │
  ├───────────────────────────────────────────────────────────┤
  │ structuralPenaltyRatio = sum(all)                         │
  │ simdStructuralPenalty  = A_SIMD * penaltyRatio            │
  └───────────────────────────────────────────────────────────┘

line 2591  candidate_costs.all_simd     = A_SIMD + simdStructuralPenalty
line 2594  candidate_costs.all_simt_only = A_SIMT
```

---

## Phase 7: Mixed 候选评分

**文件**: `SimdSimtCostModel.cpp:2597-2723`

```
line 2597  按 warp count 选最近的 setup fallback
line 2608  mixed.setup_fallback_num_warps  = nearest->numWarps
line 2610  mixed.standalone_serialized_setup = profile.simtSetupCycles (141)
line 2611  mixed.mixed_setup_fallback         = nearest->emptySimtSetupCycles
line 2613  mixed.setup_proxy_delta            = fallback - standalone
```

```
line 2622-2631:
  ┌─────────────────────────────────────────────────────┐
  │ ⑧ mixed.partition (SIMD/SIMT resources)             │
  ├─────────────────────────────────────────────────────┤
  │ simdRegularPayload = max(compute+dot, memory)       │
  │ simtAnchorPayload  = max(compute+dot+shuffle, memory)
  │                       + predicate                   │
  └─────────────────────────────────────────────────────┘

line 2633-2686  remainingStructuralPenalty (anchor 外的 residual)

line 2698  mixed.derived_simd_fraction
            = 1 - anchorPartitionWork / totalWork

line 2704-2723:
  if (anchorCount > 0):
    mixed = setupFallback + issueScale * (regular*(1+penalty) + anchor)
    mixed.cost_source = "materializable_anchor_resource_partition"
  else:
    mixed = max(allSimd, allSimt) + setupFallback
    mixed.cost_source = "inapplicable_without_materializable_anchor"

  candidate_costs.mixed_simd_simt = mixed
```

---

## Phase 8: Event Calibration + 排名 + Gate

**文件**: `SimdSimtCostModel.cpp:2725-2818`

```
line 2725  uncalibratedCandidateCosts = candidateCosts  ← 保存原始分数

line 2726-2738  Event calibration:
  if (eventRouteCalibration):
    candidateCosts.allSimd     *= domain.allSimdMultiplier
    candidateCosts.allSimtOnly *= domain.allSimtMultiplier
    candidateCosts.mixed       *= domain.mixedMultiplier
```

```
line 2740  candidateCostsEvaluated = true

line 2750-2761  排名:
  ┌─────────────────────────────────────────────────────┐
  │ ⑨ decision + scores                                 │
  ├─────────────────────────────────────────────────────┤
  │ decision_kind       = chooseBest(scores)             │
  │ runner_up_kind      = chooseRunnerUp(scores)         │
  │ best_score          = scores[decision]               │
  │ runner_up_score     = scores[runnerUp]               │
  │ decisionAdvantage   = allSimd - best                 │
  │ gain_score          = decisionAdvantage              │
  │ requiredGainScore   = max(64, baseline * margin)     │
  │ candidateRatiosToBest = scores / min(scores)         │
  └─────────────────────────────────────────────────────┘
```

```
line 2790  absoluteConfidence = min(all resource confidences)

line 2791-2798  rankingConfidence:
  if (unsupported 非空):
    → "none"
  else if (structuralPenalty > 0 && covered):
    → min(absoluteConfidence, profile.rankingConfidence)
  else:
    → absoluteConfidence
  if (!targetCompatible):
    → "none"
```

```
line 2802-2817  Gate 检查 (逐条):
  → "target_incompatible"                ← target 不匹配
  → "selection_score_invalid"             ← coverage 不通过
  → "unsupported_cost_terms"              ← unsupported 非空
  → "ranking_confidence_X_below_Y"        ← 置信度不够
  → "decision_advantage_not_above_required_gain"  ← 优势不够

line 2818  gate_passed = gateReasons.empty()
```

---

## Phase 9: 回到 runOnOperation — 额外字段

**文件**: `SelectSimdSimtCostModel.cpp:125-219`

```
line 125  recommended_decision_kind = candidateCostsEvaluated
              ? decision_kind : "backend_default"

line 129  effective_decision_kind = "backend_default"   (初始默认)

line 134  检查 explicitScope 冲突
line 154  onlyInsufficientGain = gateReasons 只有 gain 不够

line 158  if (autoMode && gatePassed && actionSupported):
              effective = recommended    ← 三条件同时满足才生效
line 162  else if (autoMode && onlyInsufficientGain):
              effective = "all_simd"     ← gain 不够，退回安全基线
line 172  else if (!autoMode):
              applicationReason = "report_mode"
line 174  else:
              applicationReason = gateReasons.front()

line 202  if (effective == "mixed_simd_simt"):
              materializeSimtAnchorPlan(module, anchorPlan)
              → scope.scope{vec_mode="simt"} 被写入 IR
```

```
line 208-218  在 report JSON 上叠加额外字段:
  ┌─────────────────────────────────────────────────────┐
  │ ⑩ 执行层字段 (C++ pass 添加)                        │
  ├─────────────────────────────────────────────────────┤
  │ mode                       = "auto" | "report"      │
  │ recommended_decision_kind  = recommended            │
  │ effective_decision_kind    = effective              │
  │ selection_source           = "cpp_cost_model" |     │
  │                              "cpp_cost_model_safe_baseline" |
  │                              "backend_default"      │
  │ application_reason         = 为什么不生效           │
  │ action_supported           = true/false             │
  │ materialized_simt_anchor_count = mixedAnchors.size()│
  └─────────────────────────────────────────────────────┘
```

---

## 字段分类索引

| 类别 | 字段前缀 | 填充阶段 |
|------|---------|---------|
| Anchor Plan | `features.simt_anchors.count`, `.mechanism_kinds`, `.kernel_lowerability` | Phase 0 |
| Profile 元数据 | `profile_*`, `score_unit`, `target_*` | Phase 2 |
| 特征 | `features.*` | Phase 1 |
| Anchor 统计 | `features.simt_anchors.*` (op counts) | Phase 1 |
| SIMT 适用性 | `applicability.*` | Phase 2 |
| 候选合法性 | `*CandidateLegal`, `candidate_roles`, `selectable_candidates` | Phase 2 (初值) + Phase 3 (Event 修正) |
| Coverage | `calibration_*`, `selection_score_valid` | Phase 3 |
| Op 周期明细 | `op_breakdown.*` | Phase 4 |
| 内存周期 | `memory.*` | Phase 4 |
| 计算周期 | `compute_only.*` | Phase 4 |
| Analytical | `analytical_candidate_costs` | Phase 5 |
| 结构惩罚 | `structure.*` | Phase 6 |
| Mixed 分账 | `mixed.*` | Phase 7 |
| 原始分数 | `event_route_calibration.raw_candidate_costs` | Phase 8 |
| 校准乘数 | `event_route_calibration.score_multipliers` | Phase 3 + Phase 8 |
| 最终分数 | `candidate_costs`, `candidate_ratios_to_best` | Phase 8 |
| 决策 | `decision_kind`, `runner_up_kind`, `best_score` 等 | Phase 8 |
| 置信度 | `*confidence` | Phase 8 |
| 门控 | `gate_passed`, `gate_reasons`, `unsupported` | Phase 8 |
| 执行 | `mode`, `recommended_*`, `effective_*`, `selection_source` | Phase 9 |
