# Rope Kernel Costmodel 诊断：预测与实测完全相反的根因分析

> 分支：`kx_simt_costmodel`（基于 `400f73560`）
> 涉及文件：
> - `test_cases/_fwd_grouped_kernel_stage1_rope.ttir`
> - `test_cases/_fwd_grouped_kernel_stage1_rope_without_costmodel_simd-simt.ttadapter`
> - `test_cases/_fwd_grouped_kernel_stage1_rope_costmodel_result.json`

## 实测结果

| 模式 | 延迟 | costmodel 预测 (candidateCosts) |
|------|------|-------------------------------|
| SIMD (compile_mode=simd) | 241 us | 4,460 (selection score) |
| SIMT (compile_mode=simd_simt, no costmodel) | **17 us** | 519,805 (selection score) |

costmodel 预测 SIMD 比 SIMT 好 117 倍，但实测 SIMT 比 SIMD 快 14 倍。**方向完全反了。**

---

## 零、背景知识：costmodel 评分用到哪些 profile 数据？

costmodel 有三个层次的数据来源，理解这一点对后续走读至关重要。

### 0.1 Profile 分层

```
ascend_davidv100_v1.json                david_v100_simd_simt_v1.json
(microbenchmark profile)                (selection profile)
─────────────────────────               ─────────────────────────
模型中性硬件微基准测量                    模型特定校准参数
  simd.f32.add.throughput = 3.30    ←── simd.ops.f32.add { throughput_measurement: "simd.f32.add.throughput" }
  simt.warp_size = 32               ←── simt.warp_size { warp_size_measurement: "simt.warp_size" }
  simt.shuffle.throughput = x       ←── simt.shuffle { throughput_measurement: "simt.shuffle.throughput" }
  ...                                       ...
                                       simt.camodel_effective:          ← profile 直接写死的数字
                                         warp_instructions_per_system_cycle:
                                           predicate: 0.038             ← 不是微基准，来自单一 FBGEMM workload
                                      selection_calibration:
                                        program_issue_scale: 8.0       ← profile 直接数字
                                        structural_penalty_ratio: {...}
                                        coverage: {...}
                                        event_route_score_multiplier:
                                          domains: {...}               ← A5 实测乘数
```

`resolveNumberOrMeasurement()` 函数（line 513）决定一个值从哪里取：
- 如果 profile JSON 字段有 `throughput_measurement`（字符串引用）→ 去 microbenchmark profile 查
- 否则 → 取 profile JSON 里的直接数字

**关键：`simtPredicateRate = 0.038` 是 profile 直接数字，不是微基准。** 它来自 `camodel_effective` 段，注释写明："These are effective issue rates for ONE FBGEMM workload, including its dependencies and stalls." —— 这是一个特定 kernel 的实测值，包含了那个 kernel 的 stall 特征，**不是通用 predicate 指令吞吐量**。

### 0.2 Event Route Calibration 乘数

三个已有 domain 及其乘数来源：

| Domain | all_simd | all_simt_only | 测自哪个 kernel |
|--------|----------|--------------|----------------|
| `masked_rowwise_reduction` | 67.628 | 0.808 | FBGEMM T128/D128/H8, num_warps=4 |
| `tiny_irregular_dot` | 1.321 | 1.0 | gather-dot-min M16/N16/K16, num_warps=4 |
| `triangular_solve_loop` | 290.881 | 1.714 | solve-tril BT16, num_warps=4 |

每个乘数 = 实测耗时 / 分析公式原始分。用 NPU Event 计数器在 A5 上取 100-200 次采样的中位数。

如果 kernel 的 `domain` 在 profile 的 `event_route_score_multiplier.domains` map 里查不到 → multiplier 全为 1.0 → 不校准。

---

## 一、Kernel 理解

### 1.1 整体结构

`_fwd_grouped_kernel_stage1_rope` 是一个 **Flash Attention with GQA（Grouped Query Attention）+ RoPE** 的 Triton kernel。TTIR 共 291 行，函数签名 19 个 block argument。

核心循环（TTIR line 176）：
```mlir
%120:3 = scf.for %arg19 = %50 to %52 step %c32_i64 ...
```
对应 Python 的 `for start_n in range(split_kv_start, split_kv_end, BLOCK_N)`。

循环上下界 `%50` 和 `%52` 都追溯到 `tt.load`（从 `kv_indptr` tensor 加载值），不是 `arith.constant`。

### 1.2 间接访存 = SIMT Anchor 候选

标记为 ★ 的 7 处 `tl.load` 的地址依赖另一个 `tl.load` 的结果：

| # | Python 变量 | TTIR 行号 | 依赖的 load 结果 | 在循环内？ |
|---|------------|----------|-----------------|----------|
| A1 | `cos` | ~96 | `pos` (从 `positions` load) | 否 |
| A2 | `sin` | ~98 | `pos` | 否 |
| A3 | `k_pe_last_token` | ~133 | `kv_indices[last]` | 否（scf.if 内） |
| A4 | `k_pe_rot_last_token` | ~135 | `kv_indices[last]` | 否（scf.if 内） |
| A5 | `k_pe` | ~192 | `kv_loc` (从 `kv_indices` load) | **是** |
| A6 | `kv` | ~210 | `kv_loc` | **是** |
| A7 | `v` | ~224 | `kv_loc` | **是** |

costmodel JSON 报告了 8 个 anchor，可能是某个 load 有变体被重复计数。

---

## 二、Costmodel 逐步走读

### 2.1 入口：`runOnOperation` (SelectSimdSimtCostModel.cpp:97)

```cpp
SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
auto report = analyzeSimdSimtCandidates(module, anchorPlan, options);
  → analyzeSimdSimtFeatures(module, anchorPlan)   // Phase 1: 特征提取
  → estimateSimdSimtCandidates(features, options)  // Phase 2: 评分
```

### 2.2 Phase 0：Anchor 识别 `buildMixedSimtAnchorPlan`

`module.walk(PreOrder)` 遍历所有 op，对每个 op 调 `analyzeAnchor(op)`：

```
tt.gather → tt.histogram → tt.scan → tt.atomic_* → TriangularSolve → isLoadedIndexDependentMemoryOp
```

对于 `tt.load`：检查 `pointerDependsOnLoadedIndex(op)` —— BFS 回溯地址 SSA chain，看是否到达另一个 `tt.load` 或 `tt.gather` 的结果。

**以 A5（循环内 k_pe load）为例**（TTIR line 192）：
```mlir
%164 = tt.load %163, %162, %cst_2
```
- `%163 = tt.addptr %107, %158`
- `%158` → `%156` → `%155` → `%154 = tt.expand_dims %153`
- `%153 = tt.load %152`（line 181，BFS 到达 `tt.load`！）
- → `pointerDependsOnLoadedIndex` 返回 true

识别为 `LoadedIndexDependentMemory` anchor。lowerability: all-SIMD=Native, all-SIMT=BackendConditional, mixed=Native。

### 2.3 Phase 1：特征提取 `analyzeSimdSimtFeatures`

`module.walk` 遍历所有 op，对每个 op 做四件事：

#### 2.3.1 分类加权

```cpp
features.weightedOps[weightedKind] += loopMultiplier;            // op 计数（loop 内=1）
features.opElements[weightedKind] += elements * loopMultiplier;  // 元素加权计数
```

**以 TTIR line 207 的 `tt.dot`（q_pe @ k_pe）为例：**
- lhs = 16×16, rhs = 16×32 → m=16, k=16, n=32
- `dotFlops += 2 * 16 * 32 * 16 * 1 = 16384`
- 三个 dot 累计 `dotFlops = 49152`

#### 2.3.2 Mask 计数（line 2097-2134）

```cpp
// 对每个 op，检查是否有 i1 类型的 tensor operand/result
if (isMaskTensorType(type))
    maskRankSum += type.getRank();   // 累加 rank

// 同时对 unique mask SSA value 做去重
if (uniqueMasks.insert(value).second) {
    uniqueMaskValues++;
    uniqueMaskRankSum += type.getRank();
    predicateElements += getStaticNumElements(type);  // 去重 mask 的总元素数
}
```

**以 TTIR line 192 的 `tt.load` 为例：**
```mlir
%164 = tt.load %163, %162, %cst_2   // %162: tensor<16x32xi1>, %cst_2: tensor<16x32xf16>
```
- `%162` 是 mask operand（i1 类型），rank = 2 → `maskRankSum += 2`
- `%cst_2` 是 other operand（f16 类型），不算 mask

**注意：`maskRankSum` 和 `uniqueMaskRankSum` 是不同的变量。**
- `maskRankSum`：每个包含 mask 的 op 的 mask rank 全量累加，同一逻辑 mask 在循环不同迭代中产生不同 SSA value 时各自计 rank
- `uniqueMaskRankSum`：对 SSA value 去重后再累加 rank

最终：`maskRankSum=153, maskTensorOps=53, uniqueMaskValues=37, uniqueMaskRankSum=66, predicateElements=8032`。

#### 2.3.3 循环识别（line 1968-1980）

```mlir
scf.for %arg19 = %50 to %52 step %c32_i64 ...
```

```cpp
getConstantInteger(%50) → arith.muli → nullopt ✗   // %50 追溯至 tt.load
getConstantInteger(%52) → arith.minsi → nullopt ✗   // %52 追溯至 tt.load
getConstantInteger(%c32_i64) → arith.constant 32 → 32 ✓
→ 有 operand 不是 constant → getKnownStaticLoopTripCount 返回 nullopt
→ hasUnknownTripCount = true
→ tripCount = nullopt.value_or(1) = 1
```

**此时 loopMultiplier 对循环内所有 op 都是 1**——所有加权计数都视为循环只执行一次。

#### 2.3.4 Feature Summary 关键结果

| 字段 | 值 | 含义 |
|------|-----|------|
| `loadOps` | 15 | 所有 `tt.load` 数量 |
| `opElements["load"]` | 2404 | load 覆盖的 tensor 元素总数 × loopMultiplier |
| `maskTensorOps` | 53 | 带 mask operand/result 的 op 总数 |
| `maskRankSum` | 153 | 所有 mask 的 rank 总和（未去重） |
| `uniqueMaskRankSum` | 66 | 去重后 mask 的 rank 总和 |
| `predicateElements` | 8032 | 去重 mask 的 SSA value 总元素数 |
| `dotFlops` | 49152 | 3 个 dot 的总 FLOPS |
| `hasUnknownTripCount` | **true** | 循环上下界不是 arith.constant |
| `staticLoopCount` | 1 | 只有 1 个 scf.for |
| `staticLoopTripCountSum` | **1** | 未知 → 默认值 |
| `loadedIndexDependentMemoryOps` | 8 | 间接访存 op 数 |
| `maxTensorNumel` | 512 | 最大 tensor = 16×32 |

### 2.4 Phase 2：`estimateSimdSimtCandidates` 完整走读

函数 ~600 行，按执行顺序 15 个逻辑块。每块说明产生的变量和含义。

---

#### Block 0: Setup（line 2217-2262）

```
变量                              含义                               来源
────────────────────────────────────────────────────────────────────────────
profile                    从 JSON 加载的完整 profile              loadCandidateProfile()
report.profileVersion      profile 版本标识                       profile.profileVersion
report.targetCompatible    硬件目标是否匹配 profile               targetMatches()
report.applicability       SIMT 机制是否被识别且可物化            evaluateSimtApplicability()
report.allSimdCandidateLegal     allSimd 路线是否合法             lowerability.allSimd == Native
report.allSimtOnlyCandidateLegal allSimtOnly 路线是否合法         compileOn91095 && lowerability == Native
report.mixedCandidateLegal       Mixed 路线是否合法               有可物化 anchor && lowerability == Native
```

---

#### Block 1: Coverage Check（line 2264-2322）

```
变量                              含义                               公式/来源
────────────────────────────────────────────────────────────────────────────────────
irregularDensity          不规则访存密度                     laneDependentPointerOps / pointerOps
[covered, domain]         是否在校准域内                    rankingCalibrationCoverage()
report.calibrationCovered  ← covered
report.calibrationDomain   ← domain 字符串
report.selectionScoreValid ← covered
eventRouteCalibration      指向 domain 对应的 Event 校准数据  profile.eventRouteCalibration.find(domain)
  └─ 找到 → eventRouteCalibrationApplied=true
  └─ 找不到 → nullptr, 三个 multiplier 保持 1.0
```

`rankingCalibrationCoverage()` 的匹配顺序：
1. `dynamic_shape` → 跳过
2. `triangular_solve_loop` → 需各项条件
3. **`hasUnknownTripCount` → 我们改为 `{true, "loop_trip_count_unknown"}`**
4. `tiny_irregular_dot` → 需 `staticLoopTrips == 0`（不匹配）
5. `rank1_indirect_vector_reduction` → 需 `dotFlops == 0`（不匹配）
6. `masked_rowwise_reduction` → 需 `dotFlops == 0`（不匹配）
7. `out_of_calibration_domain` → 拒绝

对我们的 kernel：`domain = "loop_trip_count_unknown"`，在 profile 的 `eventRouteCalibration` map 里查不到 → `eventRouteCalibration = nullptr`。

如果 `!covered && !scoreOutsideCalibrationCoverage` → **early return**（auto 模式下不评分）。

---

#### Block 2: Resource Constants（line 2324-2339）

```
变量           含义                                 值 (此 kernel)
─────────────────────────────────────────────────────────────────
numWarps       warp 数量                            32 (默认)
maxNumel       最大 tensor 元素数                    512
elementBits    最大元素位宽                         64 → clamp to 8-64
vectorWidth    SIMD 向量宽度                        2048/64 = 32 元素/指令
resourceConfidence  收集所有 profile 条目的 confidence    初始为空
```

---

#### Block 3: Op Classification Check（line 2341-2363）

检查 `gather`/`histogram`/`atomic` 是否有非零元素 → 标记为 `unsupported`。
检查是否有未分类的 `scalarOps` → 标记。

对你 kernel：gather/histogram/atomic 的 `opElements` 都是 0，没有这些特殊 op → `report.unsupported` 保持空。

---

#### Block 4: Compute Scoring — 全 kernel（line 2365-2392）

**第一次调用 `getProfileOpElements(features)`**：将 features 中的 op 计数映射为参与 profile 查表的 element 数。

```cpp
// 优先用 features.opElements 里的精确计数（已在特征提取中乘以 loopMultiplier）
// 没有精确计数的 → 用 rawCount * maxTensorNumel 估算
```

对每个 op 类型查 profile 表：

```cpp
simdCycles = ceil(elements / vectorWidth) / simdThroughput * simdFactor;
simtCycles = elements / simtThroughput * simtFactor;
simdComputeCycles += simdCycles;
simtComputeCycles += simtCycles;
```

结果（JSON）：
```json
"compute_only": { "simd": 162.3, "simd_dot": 140.0,
                  "simt": 115.2, "simt_dot": 412.6 }
```

**注意：纯计算部分 SIMT(115) < SIMD(162)，SIMT 算术反而更快。** dot 部分 SIMT(412) > SIMD(140)，但仍在同一量级。

---

#### Block 5: Compute Scoring — Mixed 分区（line 2394-2416）

**第二次调用 `getProfileOpElements(features.simtAnchors)`**：只看 anchor ops 的 element 计数。

两个重载的区别：
- 全 kernel 版：如果 `opElements` 没有 → 用 `rawCount * maxNumel` 估算
- anchor 版：如果 `opElements` 没有 → 直接返回 0（不估算）

```cpp
// 从 SIMD 部分减去 anchor 的 compute：
mixedSimdRegularComputeCycles -= simdCycles(anchors);
// 加到 SIMT 部分：
mixedSimtAnchorComputeCycles += simtCycles(anchors);
// 恢复全 kernel SIMD compute（因为初始是 0，需要加上全量）：
mixedSimdRegularComputeCycles += simdComputeCycles;
```

对你 kernel：anchor 只有 load（8 个间接 `tt.load`），没有 compute op → 第二次循环对 add/mul/div 等返回 0。只有 load 有非零元素（1632），但 load 不在 compute scoring 里处理（在 Block 6 memory scoring 处理）。

所以 `mixedSimdRegularComputeCycles = simdComputeCycles`, `mixedSimtAnchorComputeCycles = 0`。

---

#### Block 6: Memory Scoring（line 2418-2463）

```
变量                        公式                                         值
─────────────────────────────────────────────────────────────────────────────────
simdLoadCycles       = loadBytes / simdMte2BytesPerCycle               24.0
simdStoreCycles      = storeBytes / simdMte3BytesPerCycle               2.8
simdMemoryCycles     = max(load, store)   ← SIMD load/store 可并行     24.0

simtLoadCycles       = loadWarpInstructions / simtLoadWarpRate         459.7
simtStoreCycles      = storeWarpInstructions / simtStoreWarpRate        77.4
simtMemoryCycles     = load + store       ← SIMT load/store 串行       537.2

mixedSimdRegularMemoryCycles  非 anchor 的 SIMD memory 部分
mixedSimtAnchorMemoryCycles   anchor 的 SIMT memory 部分
```

---

#### Block 7: Shuffle Scoring（line 2465-2494）

SIMT reduction 需要 warp shuffle：
```cpp
shuffleLevels = log2(warpSize) = 5
simtShuffleInstructions = (reductions + scans) * ceil(maxNumel/warpSize) * shuffleLevels
simtShuffleCycles = instructions / simtShuffleRate
```

对你 kernel：`weightedReductions=2, weightedScans=0` → `simtShuffleInstructions = 2 * 16 * 5 = 160` → `simtShuffleCycles ≈ 195`。小量。

---

#### Block 8: Predicate Scoring（line 2496-2507）⚠️ 核心问题

```cpp
simtPredicateInstructions =
    features.maskRankSum * ceil(maxNumel / simtWarpSize);
//  = 153 * ceil(512 / 32) = 153 * 16 = 2448

simtPredicateCycles =
    simtPredicateInstructions / simtPredicateRate;
//  = 2448 / 0.038 = 64,421
```

**`maskRankSum = 153` 是怎么来的？**

在特征提取中（Block 2.3.2），遍历每个 op 时：
```cpp
for (Type type : op->getOperandTypes())
    if (isMaskTensorType(type))           // i1 类型的 tensor
        maskRanks.push_back(type.getRank());

for (Type type : op->getResultTypes())
    if (isMaskTensorType(type))
        maskRanks.push_back(type.getRank());

if (!maskRanks.empty()) {
    maskTensorOps++;                  // = 53 个 op 包含 mask
    for (int64_t rank : maskRanks)
        maskRankSum += rank;          // 累加每个 mask tensor 的 rank
}
```

具体例子 —— TTIR line 40-41 的两个 `arith.cmpi`：
```mlir
%10 = arith.cmpi slt, %6, %9 : tensor<16xi32> → tensor<16xi1>   // rank=1
%11 = arith.cmpi slt, %6, %cst_12 : tensor<16xi32> → tensor<16xi1>  // rank=1
```
这两个 op 的结果是 `tensor<16xi1>`，各贡献 rank=1 → `maskRankSum += 2`。

TTIR line 192 的 `tt.load`：
```mlir
%164 = tt.load %163, %162, %cst_2  // %162: tensor<16x32xi1> 是 mask operand
```
`%162` 的 rank = 2 → `maskRankSum += 2`。

53 个带 mask 的 op，每个 mask rank 1-4（取决于 `tensor<Nxi1>`, `tensor<NxMxi1>` 等）→ 累计 153。

**公式的直觉**：每个 mask 维度需要为每个 warp-block（32 lane/block）发射一条 SIMT predicate 指令来确定哪些 lane 参与执行。所以 `simtPredicateInstructions = 全部mask维度 × block数 = 153 × 16 = 2448`。

**`simtPredicateRate = 0.038` 从哪来？**

在 profile JSON 的 `simt.camodel_effective` 段（line 274-291）：
```json
"camodel_effective": {
    "warp_instructions_per_system_cycle": {
        "predicate": 0.0380,
        "memory": 2.1411,
        "shuffle": 0.3763
    },
    "source": "simt_camodel_calibration_scope_b8.json",
    "note": "These are effective issue rates for ONE FBGEMM workload,
             including its dependencies and stalls.
             They are fallback calibration seeds,
             NOT isolated instruction peak throughput."
}
```

这个 0.038 是 profile 直接数字（没有走 `throughput_measurement` 引用微基准），来自一个 FBGEMM kernel 的 CaModel profiling，包含那个 kernel 特有的 stall。

`0.038 warps/system_cycle` → 每条 warp predicate 指令约 26 system cycles。把这个极慢的速率乘以 2448 条 "指令" → 64,421 cycles。

---

#### Block 9: Dot Scoring（line 2509-2530）

```
simdDotCycles  = setup + dotFlops / simdDotFlopsPerCycle  = 64 + 49152/351 = 140
simtDotCycles  = setup + dotFlops / simtDotFlopsPerCycle  = 64 + 49152/141 = 412
```

---

#### Block 10: Assemble Analytical Cycles（line 2532-2550）

这是所有分量汇聚的位置：

```
simdIssuePayload = max(compute + dot, memory)
                 = max(162 + 140, 24) = 302

simtIssuePayload = max(compute + dot + shuffle, memory) + predicate
                 = max(115 + 412 + 0, 537) + 64421
                 = 537 + 64421 = 64958

simdAnalytical = setup + payload * programIssueScale
               = 21 + 302 * 8 = 2440

simtAnalytical = setup + payload * programIssueScale
               = 141 + 64958 * 8 = 519805
```

**predicate 把 SIMT payload 从 537 → 64958，爆炸 120 倍。** SIMD payload 302 完全正常。

---

#### Block 11: Structural Penalty（line 2552-2602）

7 个 SIMD-only 结构惩罚分量：

| 分量 | 公式 | 你的 kernel 的值 |
|------|------|-----------------|
| `irregular_addressing` | min(cap, density × perDensity) | 0.4 |
| `mask_materialization` | min(cap, maskRankSum × perMaskRank) | 0.35 |
| `reduction_lowering` | min(cap, reductions × perWeighted) | 0.04 |
| `static_loop_control` | min(cap, tripSum × perStaticLoop) | **0.008** (tripCount=1) |
| `control_flow` | hasControlFlow ? value : 0 | 0.03 |
| `tiny_dot_startup` | tinyDot ? tinyDot × underfill : 0 | 0 |
| `rank1_indirect` | rank1Indirect ? value : 0 | 0 |

`structuralPenaltyRatio = 0.828`

```
simdStructuralPenalty = simdAnalytical * structuralPenaltyRatio = 2440 * 0.828 = 2020

allSimd    = simdAnalytical + simdStructuralPenalty = 2440 + 2020 = 4460
allSimtOnly = simtAnalytical = 519805
```

---

#### Block 12: Mixed Candidate（line 2604-2730）

Mixed 候选 = 非 anchor 部分用 SIMD + anchor 部分用 SIMT，带 boundary setup 开销：

```
mixedSimdSimt = setupFallback + programIssueScale *
                (regularPayload*(1+residualPenalty) + anchorPayload) + boundaryCycles
             = 37463
```

---

#### Block 13: Event Route Calibration（line 2732-2745）

```
uncalibratedCandidateCosts = candidateCosts  // 保存原始分析分数

if (eventRouteCalibration != nullptr):
    allSimd    *= domain_multiplier       // 比如 67.628
    allSimtOnly *= domain_multiplier       // 比如 0.808
    mixedSimdSimt *= domain_multiplier     // 比如 3.537

→ candidateCosts = 校准后的分数
```

对你 kernel：`eventRouteCalibration = nullptr`（domain `"loop_trip_count_unknown"` 不在 profile 的 domains map 里）→ 三个 multiplier 保持 1.0 → `candidateCosts = uncalibratedCandidateCosts`。

---

#### Block 14: Ranking & Decision（line 2747-2796）

```
decision = chooseBest(costs)   → allSimd (4460，最低分 = 最优)
runnerUp = chooseRunnerUp      → mixed (37463)

decisionAdvantage = allSimd胜利的裕度
requiredGainScore = max(64, allSimd * 0.1) = 446

candidateRatiosToBest:
  allSimd: 1.0,  allSimtOnly: 117x,  mixed: 8.4x
```

---

#### Block 15: Confidence & Gates（line 2797-2826）

```
absoluteConfidence = 所有 profile 条目最低 confidence → "low"

rankingConfidence:
  unsupported 非空？ → "none"           (对你 kernel: false)
  structuralPenalty > 0 且 covered？ → min(absolute, profile.rankingConfidence)
                                     → min("low", "low") = "low"
  否则 → absoluteConfidence

Gate 检查:
  rankingConfidence("low", rank=1) < minimumConfidenceForDecision("low", rank=1)?
  → 1 < 1 = false → 通过！  (这个 profile 把门槛设成了 "low" 而非默认 "medium")

  其他 gate: unsupported 为空, targetCompatible=true, selectionScoreValid=true,
            decisionAdvantage(33003) > requiredGainScore(446) → 都通过

gatePassed = true
effectiveDecision = allSimd
```

**注意这个 profile 把 `minimumConfidenceForDecision` 设成了 `"low"`**（默认 `"medium"`）。这意味着即使 confidence 只有 low，gate 也会通过。这是一个显式的宽松配置。

---

## 三、根因总结

### 3.1 正确的地方

1. **Anchor 识别正确**：7-8 个 `tt.load` 的地址依赖 loaded index 被准确识别为 `LoadedIndexDependentMemory`。
2. **compute 评分大致正确**：SIMT compute(115) < SIMD compute(162)，方向对。
3. **memory 评分方向正确**：SIMT gather(537) > SIMD structured(25)，约 20x。
4. **coverage 准入生效**：我们修改的 `loop_trip_count_unknown` 让 kernel 能进入评分。

### 3.2 错误的地方

**唯一主因：SIMT Predicate 评分。** 64,421 cycles 占 allSimtOnly(519805) 的 99%。

```
simtPredicateCycles = 153 × 16 / 0.038 = 64,421
                        ↑    ↑      ↑
                   maskRankSum |  simtPredicateRate（单一 FBGEMM workload 的实测，
                            ceil(512/32)   不是通用吞吐，且包含了 workload 的 stall）
```

三个因子的叠加：
1. **`maskRankSum=153`**：所有 mask tensor 的 rank 总和，包括循环内重复的 mask。如果改用去重版 `uniqueMaskRankSum=66`，这个因子降到 66。
2. **`blockCount=16`**：每个 warp 32 lane，最大 tensor 512 元素，假定每个 mask rank 需要 16 条 warp predicate 指令。但在真实硬件上 mask 是指令内嵌的，不需要独立 predicate 指令。
3. **`simtPredicateRate=0.038`**：来自一个特定 FBGEMM kernel 的 CaModel profiling，包含该 kernel 的 stall 特征，不是通用微基准测量值。换算约 26 system cycles / predicate warp，极其慢。

### 3.3 修复优先级

| 优先级 | 修改点 | 预期效果 |
|--------|--------|---------|
| **P0** | Profile `simtPredicateRate` — 调高到接近 compute/memory 的吞吐水平 | `simtPredicateCycles` 从 64k 降到 < 2k |
| **P1** | 公式改用 `uniqueMaskRankSum` 替代 `maskRankSum`（去重后的 mask 维度累加）| `simtPredicateInstructions` 从 2448 降到 ~1000 |
| **P2** | 新增校准 domain + 测 multiplier：为"带 dot + 间接访存 + 动态循环"的 Attention 类 kernel 建立 domain，在 A5 上实测三路线耗时填入 profile | 分析公式剩余偏差（memory/dot overhead）得以校正 |
| **P3** | 实现精确 trip count（通过 JIT scalar capture） | `staticLoopTripCountSum` 反映真实循环次数，结构性惩罚更准 |

---

## 四、Mixed 分数中 Predicate 的贡献与 JSON 字段辨析

### 4.1 Mixed = 37463 的构成

mixed 候选分数的计算公式（`SimdSimtCostModel.cpp:2715-2719`）：

```
mixed = setup + programIssueScale × (regularPayloadWithResidual + anchorPayload)
```

代入 ROPE kernel 的实际数值（来自 costmodel JSON `.mixed.partition`）：

```
anchorPayload = max(compute + dot + shuffle, memory) + predicate
              = max(    0   +  0  +   0   , 300.81 ) + 3789.47
              = 300.81 + 3789.47
              = 4090.28              ← 其中 predicate 占 92.6%

regularPayload = max(compute + dot, memory)
               = max(162.33 + 140, 7.75)
               = 302.33

remainingStructuralPenalty = 0.868   ← anchor 之外剩余部分的 SIMD 结构惩罚

regularPayloadWithResidual = 302.33 × (1 + 0.868) = 564.75

mixed = 223 + 8.0 × (564.75 + 4090.28)
      = 223 + 37240.24
      = 37463  ✓
```

**Predicate 对 mixed 分数的贡献**：

```
predicatePerIteration = 3789.47
afterProgramIssueScale = 3789.47 × 8.0 = 30315.76

30315.76 / 37463 = 80.9%
```

**对比 pure SIMT：** predicate 占 98.9%（64421 × 8.0 / 521299）。

### 4.2 为什么有多个 mask_rank_sum / predicate 值

核心公式只有一个，但输入不同（`SimdSimtCostModel.cpp:2496-2507`）：

```
predicateInstructions = maskRankSum × ceil(maxNumel / warpSize)
predicateCycles       = predicateInstructions / simtPredicateRate(0.038)
```

**关键区分：全 kernel vs 仅 anchor。**

| 公式输入 | 来源字段 | 值 | 输出 | 用在哪里 |
|---------|---------|-----|------|---------|
| `features.maskRankSum` | 全 kernel 所有 mask tensor op 的 rank 累加（不去重） | **153** | `predicateCycles = 64,421` | **pure SIMT** 总分 |
| `features.simtAnchors.maskRankSum` | 仅 anchor 操作（8 个间接 load）的 mask rank | **9** | `anchorPredicateCycles = 3,789` | **mixed** 总分 |

计算验证：

```
全 kernel: 153 × ceil(512/32) / 0.038 = 153 × 16 / 0.038 = 2448 / 0.038 = 64,421 ✓
anchor  :   9 × ceil(512/32) / 0.038 =   9 × 16 / 0.038 =  144 / 0.038 =  3,789 ✓
```

**为什么 mixed 只有 9 而全 kernel 是 153？** 因为 anchor 是那 8 个间接 load（`K_Buffer[ kv_loc * stride + ... ]`），它们的 mask 就是 `offs_n < split_kv_end` 这几个 rank-1 mask。而全 kernel 包含所有 mask tensor（Q 的 mask、cos/sin 的 mask、store 的 mask、tl.where 的 mask 等），累计 153。混合路线把 anchor 按 SIMT 算（含 predicate），其余按 SIMD + structural penalty 算——所以 mixed 的 predicate 远小于 pure SIMT。

### 4.3 JSON 中所有 mask/predicate 相关字段速查

#### 全 kernel 特征（`.features.*`）

| 字段 | 值 | 含义 | 代码位置 |
|------|-----|------|---------|
| `mask_rank_sum` | **153** | 所有 mask tensor op 的 operand+result rank 累加，每次出现都算 | 遍历所有 op，`maskRanks.push_back(rank)` |
| `unique_mask_rank_sum` | **66** | 按 SSA value 去重后的 mask rank 累加 | 同一 mask value 被多处使用只算一次 |
| `mask_tensor_ops` | **53** | 涉及 mask tensor 的操作数量（每次出现都算） | `features.maskTensorOps++` |
| `unique_mask_values` | **37** | 去重后的 mask tensor SSA value 数量 | — |
| `predicate_elements` | **8,032** | 需要 predicate 的总元素数 | SIMD 侧使用的 predicate element 估计 |

#### Anchor 特征（`.features.simt_anchors.*`）

| 字段 | 值 | 含义 |
|------|-----|------|
| `mask_rank_sum` | **9** | 仅 anchor 操作的 mask rank 累加（不去重） |
| `unique_mask_rank_sum` | **8** | 仅 anchor 操作去重后的 mask rank 累加 |
| `unique_mask_values` | **5** | 仅 anchor 操作去重后的 mask SSA value 数量 |
| `predicate_elements` | **1,584** | 仅 anchor 的 predicate 元素数 |

#### SIMT 执行层（`.simt_execution.*`）

| 字段 | 值 | 公式 |
|------|-----|------|
| `predicate_warp_instructions` | **2,448** | `mask_rank_sum(153) × ceil(512/32) = 2448` |
| `predicate_system_cycles` | **64,421** | `2448 / 0.038` |

#### Mixed 分区（`.mixed.partition.*`）

| 字段 | 值 | 公式 |
|------|-----|------|
| `simt_anchor_predicate_system_cycles` | **3,789.47** | `anchorMaskRankSum(9) × ceil(512/32) / 0.038` |

### 4.4 完整分数构成一览

```
                      SIMD part          SIMT part         Predicate part
                      ─────────          ─────────         ──────────────
pure SIMD (4460):     all SIMD           —                 structural penalty only
pure SIMT (521299):   —                  compute+mem+dot   64421 × 8 = 515368 (99%)
mixed    (37463):     regular(564.75)×8  anchor mem×8      anchor pred 3789×8 = 30316 (81%)
```

Mixed 之所以比 pure SIMT 低一个数量级（37k vs 521k），是因为只对 8 个 anchor 操作按 SIMT 算 predicate（9 rank → 3789 cycles/iter），而不是全 kernel 所有 53 个 mask op（153 rank → 64421 cycles/iter）。

---

## 五、FAQ：公式细节答疑

### Q1: compute scoring 是向量计算，dot scoring 是 Cube 计算？为什么 Block 10 要 compute+dot？

**对。** `getProfileOpElements` 返回的 12 个 op（add/sub/mul/div/max/abs/exp/log/cmp/select/cast/clamp）是纯标量/向量算术，在 Ascend 上走 **Vector 单元（AIV）**。dot 在 Block 9 单独处理，走 **Cube 单元（AIC）**。

Vector 和 Cube 共享一条发射流水线（compute pipeline），它们和 Memory 流水线是并行的。所以 Block 10 的 roofline：

```cpp
simdIssuePayload = max(compute + dot, memory);
// 计算流水线 = Vector负载 + Cube负载
// 瓶颈 = max(计算流水线, 访存流水线)
```

---

### Q2: 谓词公式 `maskRankSum × ceil(maxNumel/warpSize)` 对吗？为什么不用精确值？

**这是一个启发式近似，不是精确统计。** 因为 costmodel 在 TTIR 层面运行，看不到 SIMT codegen 的结果——不知道 mask 最终编译成多少条 SIMT predicate 指令（mask 合并、predicate 寄存器分配、warp 调度都是后续 bisheng 编译器决定的）。

- 用 `maxNumel` 而非每个 op 的实际 tensor size：不知道编译器会不会把不同大小的 mask 合并
- 用 `maskRankSum` 而非 `uniqueMaskRankSum`：不知道哪些 mask 会被编译器去重优化
- 除以 `simtPredicateRate = 0.038`：这个速率来自单一 FBGEMM workload 的 CaModel profiling，**包含该 kernel 的 stall**，不是通用 predicate 吞吐

三个保守假设叠加 → 严重高估。

---

### Q3: `programIssueScale = 8.0` 是什么？

Profile JSON 直接写死的数字。**它把单次迭代的 payload 换算成 kernel 总发射周期，不是 trip count。**

trip count 已经通过 `loopMultiplier` 反映在 feature 的 `opElements` 里了（循环内的 op 元素数 × tripCount）。`programIssueScale` 反映的是固定放大因子——指令发射开销、流水线深度、warp 调度等与单次迭代数据量不成正比的开销。

```
simdAnalytical = simdSetup + simdIssuePayload × programIssueScale
               = 21        + 302              × 8.0
               = 2440
```

---

### Q4: structuralComponents 从哪来？需要像 domain 那样测 multiplier 吗？

全部从 profile JSON 的 `selection_calibration.simd_structural_penalty_ratio` 段加载（line 750-778），7 个参数 × 对应 kernel feature 值：

| 分量 | 公式 | profile 系数 |
|------|------|------------|
| `irregular_addressing` | `min(cap, irregularDensity × perDensity)` | `irregular_per_density`, `irregular_cap` |
| `mask_materialization` | `min(cap, maskRankSum × perMaskRank)` | `per_mask_rank`, `mask_cap` |
| `reduction_lowering` | `min(cap, reductions × perWeighted)` | `per_weighted_reduction`, `reduction_cap` |
| `static_loop_control` | `min(cap, tripSum × perStaticLoop)` | `per_static_loop_trip`, `loop_cap` |
| `control_flow` | `hasControlFlow ? value : 0` | `control_flow` |
| `tiny_dot_startup` | `tinyDot × underfill` | `tiny_dot`, `tiny_dot_flops_max` |
| `rank1_indirect` | `rank1Indirect ? value : 0` | `rank1_indirect_vector_reduction` |

**不需要像 domain multiplier 那样针对不同算子测。** 这些是**特征级别**的惩罚系数，每个 kernel 用自己的 feature 值代入统一公式。系数的校准可以通过一个代表性 kernel 完成，然后泛化到同模式的其他 kernel。

domain multiplier 则是**最终分数级别**的乘数，纠正整个分析公式的输出偏差，所以需要多个 domain 各自在 A5 上实测校准。

---

### Q5: mixedSetupFallbackCycles 是什么？

**SIMT 的空启动开销。** 从 profile JSON 的 `mixed_setup_fallback` 段加载（line 303-320），按 numWarps 分档（1/2/4/8/16/32 warps）。测量方式是用 **空 harness**——发射一个不做实际计算的 SIMT kernel，测量其 setup 周期。

```
mixedCost = fallbackSetup + programIssueScale × (simdPayload + simtPayload) + boundary
           ↑ 纯 SIMT 启动开销的代理值
```

之所以叫 "fallback（回落）"：Mixed 候选的 SIMD→SIMT 方向性过渡延迟没有被测量过，用纯 SIMT 空启动周期代替。profile JSON 中也标注了 `"directional_transition_measurement_status": "unmeasured"`。
