# Autoscope SIMT Costmodel 深度分析 (v10 更新版)

> **文档版本**: 对应 `simt_costmodel` 分支 commit `400f73560` (2026-08-04 squash)
> **旧版文档**: `autoscope_simit_costmodel_analysis.md` (对应旧版 v6 schema)
> **阅读建议**: 新旧文档对照阅读。本文档标注了所有与旧版的关键差异。

---

## 目录

1. [概述：v6 → v10 核心变化](#1-概述v6--v10-核心变化)
2. [代码目录结构重组](#2-代码目录结构重组)
3. [两模型 + 共享证据层的架构](#3-两模型--共享证据层的架构)
4. [SimtAnchorAnalysis：Anchor 模式匹配](#4-simtanchoranalysisanchor-模式匹配)
5. [SimtSelection.h 更新](#5-simtselectionh-更新)
6. [Scoring 公式详解 (v10)](#6-scoring-公式详解-v10)
7. [Mixed 候选评分：从 convex blend 到 resource partition](#7-mixed-候选评分从-convex-blend-到-resource-partition)
8. [Event Route Calibration：测量残差校正](#8-event-route-calibration测量残差校正)
9. [Profile JSON 与 Microbenchmark Profile](#9-profile-json-与-microbenchmark-profile)
10. [SelectSimdSimtCostModel Pass 更新](#10-selectsimdsimtcostmodel-pass-更新)
11. [MaterializeSimtScopes：两点增强](#11-materializesimtscopes两点增强)
12. [TriangularSolveLoop：新增锚点类型](#12-triangularsolveloop新增锚点类型)
13. [RowCoalescing：纯 SIMT 优化](#13-rowcoalescing纯-simt-优化)
14. [Python 集成层更新](#14-python-集成层更新)
15. [单元测试对比](#15-单元测试对比)
16. [UnstructureConversionPass 的桥接变化](#16-unstructureconversionpass-的桥接变化)
17. [总结：v10 的设计决策](#17-总结v10-的设计决策)
18. [与旧版文档的 Section 对照表](#18-与旧版文档的-section-对照表)

---

## 1. 概述：v6 → v10 核心变化

旧版（v6 schema, profile v5）的核心设计是：
- 每个 kernel 做三路评分（AllSIMD / AllSIMTOnly / MixedSIMDSIMT）
- Mixed cost = `(1-fraction)*simt_cost + fraction*simd_cost + transition_cost`（凸组合）
- Per-op 通过 `kSelectedForSimtAttr` 持久化标记
- 硬件参数直接嵌入 selection profile JSON

新版（v10 schema, profile v10）的核心变化：

| 维度 | 旧版 (v6) | 新版 (v10) |
|------|-----------|-----------|
| **Anchor 识别** | 简单的 `isMixedSimtAnchor()` 函数 | `SimtAnchorAnalysis` 完整模式匹配，6 种锚点类型 |
| **Mixed 评分** | convex blend `(1-f)*simt + f*simd + transition` | Resource partition: SIMT 锚点按 SIMT 费率，其余按 SIMD 费率，顺序执行 cost 相加 |
| **Per-op 标记** | `ascend.simt_costmodel.selected` bool attr 持久化 | 不可变 `SimtAnchorPlan`，无持久化 per-op 标记，`scope.scope{vec_mode="simt"}` 是唯一路由契约 |
| **Profile 分离** | 硬件参数嵌入 selection profile | `MicrobenchmarkProfile`（模型无关）+ selection profile（模型相关），通过 measurement key 引用 |
| **残差校正** | 无 | Event Route Calibration：分析公式 → 测量乘数校正 |
| **Setup cost** | 单一 transition cost | Per-warp-count fallback（来自 standalone empty-VF probe），非真正 directional transition |
| **候选合法性** | 简单 boolean | `CandidateLowerability`（Native/BackendConditional/AliasesMixed/Unsupported），Event 验证可提升 BackendConditional |
| **三角求解** | 不支持 | `TriangularSolveLoop` 锚点，专属 calibration domain，动态循环界限例外 |
| **纯 SIMT 优化** | 无 | `RowCoalescing` 行合并 |
| **Schema version** | 报告 schema v6, profile schema v2 | 报告 schema v10, profile schema v4 |

---

## 2. 代码目录结构重组

### 2.1 文件移动

旧版所有文件在 `Analysis/` 和 `include/Utils/` 下，新版重组为三个子模块：

```
costmodel/
├── README.md                          # 架构文档（新）
├── configs/                           # 硬件配置（不变）
├── include/AscendModel/
│   ├── Analysis/                      # 绝对 cost / HIVM 模型（不变）
│   ├── Profile/
│   │   └── MicrobenchmarkProfile.h    # 新：共享微基准测量
│   ├── RouteModel/                    # 新：SIMD/SIMT 路由模型
│   │   ├── SimdSimtCostModel.h        # ← 从 Analysis/ 移入
│   │   ├── SimtSelection.h            # ← 从 include/Utils/ 移入
│   │   └── SimtAnchorAnalysis.h       # 新：锚点分析
│   └── Transforms/
│       └── Passes.td                  # 更新：新 pass 定义
├── lib/AscendModel/
│   ├── Analysis/                      # 不变
│   ├── Profile/
│   │   ├── CMakeLists.txt             # 新
│   │   └── MicrobenchmarkProfile.cpp  # 新
│   ├── RouteModel/                    # 新
│   │   ├── CMakeLists.txt
│   │   ├── SimdSimtCostModel.cpp      # ← 从 Analysis/ 移入，2080→2841 行
│   │   ├── SimtAnchorAnalysis.cpp     # 新：602 行
│   │   └── Transforms/
│   │       ├── SelectSimdSimtCostModel.cpp  # 314→229 行（精简）
│   │       └── MaterializeSimtScopes.cpp    # 153→290 行（增强）
│   └── Transforms/                    # 不变
└── profiles/
    ├── README.md                      # 新
    ├── microbench/
    │   ├── ascend_davidv100_v1.json   # 更新：v1→v2，新测量项
    │   └── microbenchmark_profile_schema.json  # 新
    └── simd_simt/
        ├── david_v100_simd_simt_v1.json       # 大幅重写：311→332 行
        ├── simd_simt_profile_schema.json      # 大幅扩展：204→901 行
        └── david_v100_des_feedback_v1.json    # 新：DES 反馈占位
```

### 2.2 新增的 RowCoalescing

纯 SIMT 路径的预处理优化（独立于 RouteModel）：

```
include/TritonToLinalg/RowCoalescing.h   # 56 行
lib/TritonToLinalg/RowCoalescing.cpp      # 659 行
```

---

## 3. 两模型 + 共享证据层的架构

新版明确分离为两层模型 + 一个共享证据层：

```text
profiles/microbench/  (模型无关硬件测量)
        |
        v
AscendModelProfile    (MicrobenchmarkProfile loader)
   |             |
   v             v
AscendModelAnalysis    AscendModelRouteModel
(绝对/HIVM cost)       (SIMD/SIMT 路由选择)
          \               /
           v             v
      AscendModelTransforms / backend 集成
```

关键设计原则：
- **MicrobenchmarkProfile** 是模型无关的：包含物理测量值、单位、时钟域、测量范围、来源、置信度
- **Selection Profile** 是模型相关的：包含策略、校准、覆盖域、惩罚参数、通过 measurement key 引用 microbenchmark
- 两个模型共享测量数据，但不共享 scoring 公式

### 3.1 MicrobenchmarkProfile 数据结构

```cpp
struct MicrobenchmarkMeasurement {
  double value = 0.0;
  std::string unit;           // "MHz", "system_cycle", "byte/system_cycle", etc.
  std::string cycleDomain;    // "SYS_CNT", "wall_clock", "none"
  std::string scope;          // "single_aiv", "device", "single_warp", etc.
  std::string sourceKind;     // "isolated_microbenchmark", "architecture_fact", etc.
  std::string source;         // 具体来源文件
  std::string confidence;     // "none" | "low" | "medium" | "high"
};
```

每个测量值都有完整的 provenance。Consumer 通过 `requireValue(key, expectedUnit, expectedCycleDomain)` 读取，如果单位或时钟域不匹配则报错。

---

## 4. SimtAnchorAnalysis：Anchor 模式匹配

**这是 v10 最重要的架构变化。**

旧版中，"哪些 op 用 SIMT" 的判断散布在多个地方（`isMixedSimtAnchor` 函数 + UnstructureConversionPass 的静态规则）。新版将所有模式匹配集中到 **SimtAnchorAnalysis**，形成不可变的 **SimtAnchorPlan**，由 feature extraction、scoring、materialization 三个阶段共享。

### 4.1 六种 Anchor 类型

```cpp
enum class SimtAnchorKind {
  DirectGather,                // tt.gather
  LoadedIndexDependentMemory,  // tt.load/tt.store 的指针依赖 loaded/gathered index
  Histogram,                   // tt.histogram
  PlainOneDimensionalCumsum,   // tt.scan, 非约简轴 extent=1
  TensorAtomic,                // tt.atomic_rmw / tt.atomic_cas
  TriangularSolveLoop,         // scf.for + vector_load(16) + axis0_reduce + masked_update
};
```

### 4.2 每种 Anchor 的 Lowerability

每个 anchor 对三种路由各有独立的 `CandidateLoweringStatus`：

```cpp
enum class CandidateLoweringStatus {
  Unsupported,         // 此路由不支持该 anchor
  Native,              // 原生支持
  BackendConditional,  // 需要后端验证（可被 Event 验证提升）
  AliasesMixed,        // 此路由实质等同于 mixed（如 cumsum 在 all-SIMD 下）
};
```

具体每种 anchor 的 lowerability：

| Anchor | all-SIMD | all-SIMT | mixed |
|--------|----------|----------|-------|
| DirectGather | Native | BackendConditional | Native |
| LoadedIndexDependentMemory | Native | BackendConditional | Native |
| Histogram | **Unsupported** | **Unsupported** | Native (需 static rank-1) |
| PlainOneDimensionalCumsum | **AliasesMixed** | BackendConditional | Native (需 axis ≤64 或 supported dtype) |
| TensorAtomic | Native | BackendConditional | Native (需 supported dtype/operation) |
| TriangularSolveLoop | Native | BackendConditional | Native |

注意 Histogram 的特殊性：它**只能**在 mixed 模式下工作（all-SIMD 和 all-SIMT 均标记为 Unsupported）。

### 4.3 Anchor Plan 的构建和共享

```cpp
struct SimtAnchorPlan {
  llvm::SmallVector<SimtAnchorDescriptor, 0> anchors;
  CandidateLowerability kernelLowerability;  // 全 kernel 级别的组合状态

  llvm::SmallVector<Operation *> materializableRoots() const;
  int64_t materializableCount() const;
};
```

构建过程（`buildMixedSimtAnchorPlan`）：
1. **Pre-order walk** ModuleOp，对每个 op 调用 `analyzeAnchor(op)`
2. 如果识别为 anchor → 加入 plan，**skip children**（嵌套 op 不重复计数）
3. 遍历所有 anchor，通过 `combineWholeKernelStatus` 组合出 kernel 级别的 lowerability
4. 判断 mixed 是否 native：至少一个 anchor native 且没有被 blocked

**关键**: 这个 plan 是不可变的。Feature extraction 用它来 split 操作计数（哪些在 anchor 内、哪些在外），scoring 用它来计算 mixed cost，materialization 用它来创建 scope.scope 区域。三个阶段看到的永远是同一组操作。

### 4.4 关键辅助函数

**`isLoadedIndexDependentMemoryOp(op)`**：检查 load/store 的指针 SSA backward slice 是否到达 loaded/gathered index。这是**真正的数据依赖测试**，不是旧版的 rank-based proxy。

**`pointerDependsOnLoadedIndex(op)`**：从 memory op 的 pointer operand 开始 BFS，看是否能到达 `tt.load` 或 `tt.gather`。支持穿越 `scf.for` 的 iter_args。

**`getStaticMaskActiveFraction(mask)`**：静态分析 mask tensor 中 true 的比例。支持 `arith.constant`（dense i1 elements）、`IntegerAttr`、`tt.splat`。

**`isTriangularSolveLoop(op)`**：检测 solve_tril 的循环模式：
- 包含 `tt.load`（rank-1, shape[0]=16）
- 包含 `tt.reduce`（axis=0）
- 包含 `arith.select`（masked update）
- Iter_args 包含 16×16 的 triangular state，或至少一个 sibling 循环有相同模式

---

## 5. SimtSelection.h 更新

路径从 `include/Utils/SimtSelection.h` 移到 `include/AscendModel/RouteModel/SimtSelection.h`。

### 5.1 删除的 attribute

旧版有 `kSelectedForSimtAttr = "ascend.simt_costmodel.selected"` 和 `kScopeMaterializedAttr`——这些 per-op 持久化标记在新版中被移除。

### 5.2 关键函数保持不变

- `getEffectiveExecution(op)` — 向上查找最近的 `ascend.simt_costmodel.effective`
- `isModelControlled(op)` — 判断是否由 cost model 控制（非 backend_default）
- `isMixedModelDecision(op)` — 判断是否为 mixed 决策
- `hasEnclosingVectorMode(op, mode)` — 检查是否在 `scope.scope{vec_mode=...}` 内
- `shouldUseSimtTemplate(op, legacyForceSimt)` — **核心桥接函数**

### 5.3 shouldUseSimtTemplate 的行为变化

```cpp
inline bool shouldUseSimtTemplate(Operation *op, bool legacyForceSimt) {
  if (hasEnclosingVectorMode(op, "simd"))
    return false;                    // SIMD scope 永远赢

  const bool locallySelected = hasEnclosingVectorMode(op, "simt");
  if (isModelControlled(op))
    return locallySelected;          // 模型控制时只看 scope
  return legacyForceSimt || locallySelected;  // 传统模式保持兼容
}
```

**关键变化**：旧版有 `isSelectedForSimt(op)` 检查持久化的 per-op bool attr，新版**只有** `hasEnclosingVectorMode(op, "simt")` — 即只看是否在 `scope.scope{vec_mode="simt"}` 内。

这意味着：在模型控制模式下，要让一个 op 走 SIMT lowering，它**必须**在 materializer 创建的 SIMT scope 内。不再有 per-op attr 的后门。

---

## 6. Scoring 公式详解 (v10)

新版 scoring 公式保持了 fundamentals（SIMD/SIMT 计算/内存 roofline + structural penalty），但有重大调整。

### 6.1 SIMD Analytical Cost

```
simdCompute = Σ ceil(elements / vectorWidth) / opThroughput * factor
simdMemory  = max(loadBytes/simdMte2Rate, storeBytes/simdMte3Rate)
simdDot     = dotSetup + dotFlops / simdDotFlopsPerCycle

simdIssuePayload = max(simdCompute + simdDot, simdMemory)
A_SIMD = simdSetup + simdIssuePayload * programIssueScale
```

其中 `vectorWidth = simdVectorWidthBits / elementBits`，`programIssueScale = 8.0`（来自 profile）。

### 6.2 SIMT Analytical Cost

```
simtCompute  = Σ elements / scalarOpThroughput * factor
simtMemory   = loadWarpInsts/loadRate + storeWarpInsts/storeRate
simtShuffle  = weightedReductions * ceil(maxNumel/warpSize) * log2(warpSize) / shuffleRate
simtPredicate = maskRankSum * ceil(maxNumel/warpSize) / predicateRate
simtDot      = dotSetup + dotFlops / simtDotFlopsPerCycle

simtIssuePayload = max(simtCompute + simtShuffle + simtDot, simtMemory) + simtPredicate
A_SIMT = simtSetup + simtIssuePayload * programIssueScale
```

**新变化**：SIMT issue payload 中 `simtShuffle` 参与 compute roofline 的 max（旧版可能只在 mixed 路径才计算），`simtPredicate` 加在 max 之外。

### 6.3 Structural Penalty（SIMD-only）

```
Pstruct = min(irregularCap, irregularDensity * irregularPerDensity)
        + min(maskCap, maskRankSum * perMaskRank)
        + min(reductionCap, weightedReductions * perWeightedReduction)
        + min(loopCap, staticLoopTripCountSum * perStaticLoopTrip)
        + (hasControlFlow ? controlFlow : 0)
        + (tinyDot ? tinyDot * tinyDotUnderfill : 0)
        + (rank1IndirectVectorReduce ? rank1Penalty : 0)

allSimdCost = A_SIMD * (1 + Pstruct)
```

**新变化**：
- `irregularDensity = laneDependentPointerOps / pointerTensorOps`（基于 rank proxy，不是真正的地址 stride）
- tiny dot 有独立的 `tinyDotIrregularPerDensity` 和 `tinyDotIrregularCap`（比普通 irregular 更小）
- `tinyDotUnderfill = max(0, 1 - dotFlops/tinyDotFlopsMax)`

### 6.4 All-SIMT Cost

```
allSimtOnlyCost = A_SIMT
```

无 structural penalty（SIMT 天然处理 irregular addressing）。

---

## 7. Mixed 候选评分：从 convex blend 到 resource partition

**这是 scoring 公式最大的变化。**

### 7.1 旧版公式（回顾）

```
mixed_cost = (1 - simd_fraction) * simt_cost + simd_fraction * simd_cost + transition_cost
```

问题：
1. 这是一个**凸组合**——永远不会同时低于两个端点（除非 transition_cost 为负）
2. simd_fraction 是启发式 proxy（基于 static rules 的比例），不是精确的 anchor partition

### 7.2 新版公式

```
// 步骤 1：将操作按 anchor plan 分为两部分
//   - SIMT anchor 内的操作 → SIMT 费率
//   - 其余操作（regular）→ SIMD 费率

mixedSimdRegularCompute = simdCompute - anchorOps在SIMD下的cost
mixedSimtAnchorCompute  = anchorOps在SIMT下的cost

// 同样的 partition 应用于 dot、memory、shuffle、predicate

// 步骤 2：分别计算 roofline
mixedSimdRegularPayload = max(mixedSimdRegularCompute + mixedSimdRegularDot,
                               mixedSimdRegularMemory)
mixedSimtAnchorPayload  = max(mixedSimtAnchorCompute + mixedSimtAnchorDot + mixedSimtAnchorShuffle,
                               mixedSimtAnchorMemory) + mixedSimtAnchorPredicate

// 步骤 3：计算 remaining structural penalty（anchor 外的 residual）
remainingStructuralPenalty = 同样公式，但只用 non-anchor 特征值

// 步骤 4：顺序执行，相加
如果 anchor 数 > 0:
  mixedCost = setupFallback
            + programIssueScale * (
                mixedSimdRegularPayload * (1 + remainingStructuralPenalty)
                + mixedSimtAnchorPayload
              )
            + boundaryCycles
否则:
  mixedCost = max(allSimdCost, allSimtCost) + setupFallback
```

**关键洞察**：
- SIMD 和 SIMT 阶段是**顺序执行**的，不是并行的 → 它们的 roofline 相加（不是取 max）
- Setup fallback 来自 standalone empty-VF probe，不是真正的 directional transition
- 当没有 materializable anchor 时，mixed cost 退化为 `max(simd, simt) + setup`（必然大于两个端点，自动被淘汰）

### 7.3 Setup Fallback

从 profile 中按 warp count 选择最近的 fallback：

```
setupFallback = nearestWarpEntry.emptySimtSetupCycles
```

这些值来自 `simt.setup.transition_harness_net.warps_{1,2,4,8,16,32}`，是 standalone empty-VF probe（mode1 minus barrier-only mode6），**不包含实际的 SIMD→SIMT 方向转换延迟**。

对于 32 warps：223 cycles，对于 1-16 warps：182 cycles。

---

## 8. Event Route Calibration：测量残差校正

**这是 v10 新引入的概念。**

分析公式给出的是 feature-sensitive 的 base score，但与真实 NPU Event 测量之间存在 route-relative residual。Event calibration 用 domain-specific 乘数来校正：

```
calibrated_score = raw_analytical_structural_score * domain_multiplier
```

### 8.1 三个 Calibration Domain

每个 domain 有独立的三路乘数，来自 A5 card 0 上的真实 NPU Event 测量：

**1. `masked_rowwise_reduction`** (FBGEMM workload, 4 warps)
```
all_simd_multiplier:      67.628046
all_simt_only_multiplier:  0.807671
mixed_simd_simt_multiplier: 3.537276
all_simt_only_validated:  true (PASS)
mixed_simd_simt_validated: true (PASS)
```

**2. `tiny_irregular_dot`** (gather-dot-min M16/N16/K16, 4 warps)
```
all_simd_multiplier:       1.321185
all_simt_only_multiplier:  1.0       (未验证，因 correctness 不匹配)
mixed_simd_simt_multiplier: 0.666478
all_simt_only_validated:  false (256/256 elements mismatch)
mixed_simd_simt_validated: true (PASS, 13.18% faster than all-SIMD)
```

**3. `triangular_solve_loop`** (solve-tril BT16, 4 warps)
```
all_simd_multiplier:       290.881045
all_simt_only_multiplier:  1.713584
mixed_simd_simt_multiplier: 2.892885
all_simt_only_validated:  true (PASS)
mixed_simd_simt_validated: true (PASS)
// 但 all-SIMT 优势 < 10% margin，Post Check 保留 all-SIMD
```

### 8.2 乘数的含义

注意 `masked_rowwise_reduction` 的 all-SIMD 乘数高达 **67.6**。这意味着：
- 对于 FBGEMM 类 workload，raw analytical SIMD cost 严重低估了实际 SIMD 时间
- 通过乘数校正，score 从 "per-program ranking proxy" 变成了 Event-anchored scale
- 这就是为什么 report 的 `scoreUnit` 是 `"system_cycle_selection_score"` 而不是 literal cycles

### 8.3 乘数在 scoring 流程中的位置

```
feature extraction → coverage check → analytical scoring → structural penalty
    → allSimdCost = A_SIMD * (1+Pstruct)
    → allSimtCost = A_SIMT
    → mixedCost = setupFallback + programIssueScale * (...)

    → calibrated_allSimdCost = allSimdCost * domain_all_simd_multiplier
    → calibrated_allSimtCost = allSimtCost * domain_all_simt_multiplier
    → calibrated_mixedCost  = mixedCost  * domain_mixed_multiplier

    → ranking → gates → decision
```

Raw (uncalibrated) scores 和乘数都保留在 report 的 `event_route_calibration` 字段中。

---

## 9. Profile JSON 与 Microbenchmark Profile

### 9.1 Profile 分离

**旧版**: 硬件参数（如 `simdVectorWidthBits`、`simtWarpSize`）直接写在 selection profile JSON 中。

**新版**: 
- `profiles/microbench/ascend_davidv100_v1.json` — 模型无关测量（25 项 → 现在有更多项）
- `profiles/simd_simt/david_v100_simd_simt_v1.json` — 模型相关校准，通过 `throughput_measurement` / `empty_launch_measurement` 等 key 引用 microbenchmark

例如 SIMD add 操作：
```json
"f32.add": {
  "description": "Measured FP32 SIMD add base throughput...",
  "throughput_measurement": "simd.f32.add.throughput"
}
```

C++ loader 通过 `resolveNumberOrMeasurement` 从 microbenchmark profile 查找 `simd.f32.add.throughput`，找到：
```json
"simd.f32.add.throughput": {
  "value": 3.30,
  "unit": "vector_instruction/system_cycle",
  "cycle_domain": "SYS_CNT",
  "scope": "single_aiv_runtime_loop_ilp4_or_more_effective_source_vadd",
  "source_kind": "isolated_microbenchmark",
  "source": "triton_cases/SIMT_Test/tput.cce; concur2.cce",
  "confidence": "high"
}
```

### 9.2 新增的 Microbenchmark 测量项

相比旧版，microbenchmark 新增了多个测量项：

| 新增 key | 值 | 用途 |
|----------|-----|------|
| `simd.f32.add.dependent_latency` | 1.818 sys_cycle | 依赖链延迟 |
| `simt.setup.empty` | 115.0 sys_cycle | 无 barrier 的 async_invoke slope |
| `simt.setup.empty_with_barrier` | 141.0 sys_cycle | 有 barrier 的串行 empty-VF |
| `simt.shuffle.dependent_latency` | 27.28 sys_cycle | shuffle 依赖链延迟 |
| `simt.gm.load.bandwidth` | 22.55 byte/sys_cycle | 单 AIV GM load 带宽 |
| `simt.gm.store.bandwidth` | 16.53 byte/sys_cycle | 单 AIV GM store 带宽 |
| `simt.ub.load.throughput` | 0.507 warp_inst/sys_cycle | UB load 吞吐 |
| `simt.ub.store.throughput` | 0.530 warp_inst/sys_cycle | UB store 吞吐 |
| `simt.ub.load.bandwidth` | 64.94 byte/sys_cycle | UB load 带宽 |
| `simt.ub.store.bandwidth` | 67.86 byte/sys_cycle | UB store 带宽 |
| `simt.setup.transition_harness_net.warps_{1..32}` | 182-223 sys_cycle | Mixed setup fallback (per warp count) |

所有测量都带有 `unit`、`cycle_domain`、`scope`、`source`、`confidence` 元数据。

### 9.3 Profile Version 兼容性

C++ loader 接受 v3 到 v10 的 profile version：
```cpp
profile.profileVersion == "david-v100-simd-simt-20260727-v3" ||
... ||
profile.profileVersion == "david-v100-simd-simt-20260804-v10"
```

v5+ 必须引用 `microbenchmark_profile`，v7+ 使用 anchor partition，v9+ 需要 Event calibration domains。

---

## 10. SelectSimdSimtCostModel Pass 更新

### 10.1 流程精简

**旧版** (314 行)：
1. Extract features → 2. Score candidates → 3. Apply gates → 4. Mark per-op attrs → 5. Materialize scopes

**新版** (229 行)：
1. Clear previous selection → 2. Build anchor plan → 3. Score via `analyzeSimdSimtCandidates(module, anchorPlan)` → 4. Apply gates → 5. Set module attrs → 6. Materialize anchor plan (if mixed)

关键精简：
- 不再需要 per-op 标记循环
- Feature extraction 和 scoring 合并为一个调用
- Anchor plan 在 pass 内构建一次，传给 scoring 和 materialization

### 10.2 新增的决策逻辑

```cpp
// auto 模式 + gate 通过 + action 支持 → 应用推荐决策
if (autoMode && report.gatePassed && actionSupported) {
  effective = recommended;
  selectionSource = "cpp_cost_model";
}
// auto 模式 + 仅 gain margin 不足 + all-SIMD 合法 → 保留 all-SIMD baseline
else if (autoMode && onlyInsufficientGain && report.allSimdCandidateLegal) {
  effective = kAllSimd;
  selectionSource = "cpp_cost_model_safe_baseline";
}
// report 模式 → 不应用
else if (!autoMode) {
  // effective = backend_default（保持传统行为）
}
```

**关键变化**: 当 gain margin 不足时，新版明确设置为 `all_simd` 而不是 `backend_default`。这避免了 "模型拒绝了一个 marginal SIMT 推荐后意外走 legacy force path" 的问题。

### 10.3 显式 Scope 保护

如果 kernel 中已有用户显式写的 `scope.scope`（如 `al.scope(vector_mode="simt")`），模型不会覆盖：
```cpp
if (recommended == kMixedSimdSimt && hasExplicitScope) {
  actionSupported = false;
  applicationReason = "explicit_scope_present";
}
```

### 10.4 JSONL Report 输出

支持通过 `--report-file` 参数将 JSON report 追加到文件（一行一个 JSON）：
```cpp
if (failed(appendJSONLine(reportFile.getValue(), json)))
  module.emitWarning("failed to append...");
```

---

## 11. MaterializeSimtScopes：两点增强

### 11.1 支持范围包装（TriangularSolveLoop）

旧版 materializer 只能包装**单个 op**。新版支持包装**连续的操作范围**：

```cpp
// 单 op anchor（DirectGather, LoadedIndexDependentMemory, etc.）
wrapAnchorOperation(op);
  → 创建 scope.scope{vec_mode="simt"}，移入 op，thread SSA results

// 范围 anchor（TriangularSolveLoop）
collectTriangularSolveRange(anchor);
  → 找到从第一个 scf.for 到最后一个 arith.select 的连续范围
  → 只 thread escaping values（作用域外使用的 results）
wrapAnchorRange(range);
  → 创建 scope.scope{vec_mode="simt"}，移入整个范围
  → 只有 escaping values 通过 scope.return 传出
```

### 11.2 Compatibility Pass

新增 `MaterializeSimtScopesPass`（不同于 materializer 函数）作为兼容性验证：
```cpp
void runOnOperation() override {
  if (!isMixedModelDecision(module)) return;
  if (containsLocalSimtScope(module)) return;  // 已有 scope → OK
  module.emitError("mixed_simd_simt requires materialized scope.scope<simt>");
  signalPassFailure();
}
```

这个 pass 在 SelectSimdSimtCostModel pass **之后**运行，验证 mixed 决策确实带有 scope contract。

---

## 12. TriangularSolveLoop：新增锚点类型

这是 v10 的一个完整新功能——识别和 materialize solve_tril 的三角求解循环。

### 12.1 检测条件

```cpp
static bool isTriangularSolveLoop(Operation *op) {
  // 必须是 scf.for
  // Body 内必须包含：
  //   - tt.load (rank-1, shape[0]=16)     → 向量加载
  //   - tt.reduce (axis=0)                → 轴0规约
  //   - arith.select                      → 掩码更新
  // 且：
  //   - Iter_args 有 16×16 的 triangular state，或
  //   - 有 sibling 循环具有相同的 load/reduce/select 模式
}
```

### 12.2 动态循环界限的例外

三角求解的循环界限是 `min(T, 16)`, `min(T, 32)` 等——**故意使用动态值**。普通的 `hasUnknownTripCount` 检查会拒绝它。

新版在 coverage 检查中为 triangular solve 开了例外：
```cpp
if (hasTriangularSolve && !features.hasUnknownTripCount && ...)
  return {true, "triangular_solve_loop"};
if (hasTriangularSolve && maxNumel <= tinyDotMaxTensorNumel && ...)
  return {true, "triangular_solve_loop"};  // 也接受 unknown trip count
```

**但例外是有限的**：仍然要求 small tensor (≤256)、1-4 个 anchor、mask/reduction 在 profile 限制内。去掉 triangular mechanism evidence 后，会返回 `unknown_loop_trip_count` 拒绝（有单元测试覆盖此路径）。

### 12.3 Materialization 特殊处理

TriangularSolveLoop 的 materialize 与众不同——它包装的不是单个 op，而是一个**连续范围**：
- 从 setup（第一个 scf.for 之前最近的 tt.load 之后）开始
- 到 final update（最后一个 scf.for 之后的 arith.select chain）结束
- `scope.scope{vec_mode="simt"}` 包围整个三角求解逻辑

### 12.4 为何 all-SIMT 在 solve_tril 上不显著胜出

Event calibration 数据显示：
- all-SIMD: 22.1160 us (median)
- all-SIMT: 21.9315 us (median)
- mixed: 22.3130 us (median)

all-SIMT 只快 0.8%，低于 10% 的生产切换 margin，因此 auto mode 下会保留 all-SIMD baseline。

---

## 13. RowCoalescing：纯 SIMT 优化

### 13.1 目的

当 cost model 选择 `all_simt_only` 且 kernel 是 whole-body SIMT scope 时，在 TTIR 级别做行合并优化：

```python
# 优化前：H 个独立的 row-id 程序
row = pid(axis=0)
# 优化后：一个程序处理 H 行
rows = pid(axis=0) * H + arange(H)
```

这种变换减少了 grid launch 的维度，让 SIMT  warp 更充分地利用。

### 13.2 保守的 bail-out 条件

RowCoalescing 只对明确的 row-id 模式生效：
- 必须有 `tt.make_range` + `tt.expand_dims` + `tt.addptr` 的行索引模式
- 只折叠能被整除的维度
- 不处理复杂的分支或间接索引

### 13.3 Metadata 导出

```python
metadata["coalesce_factor"] = factor
metadata["coalesce_axis"] = axis
metadata["coalesce_grid_ceil_div"] = True/False
metadata["row_coalescing_applied"] = factor > 1
```

Driver 端使用 `coalesce_grid_ceil_div` 来决定 launcher 的 grid shrink 方式（因为 row mask 处理尾部）。

---

## 14. Python 集成层更新

### 14.1 新的 env vars 和 NPUOptions

```python
# NPUOptions 新字段
auto_simt_scope_mode: str     # "off" | "auto" | "report"
auto_simt_scope_dump: str     # 可选 JSONL report 输出路径
auto_simt_scope_margin: float # gain margin ratio (default 0.10)
auto_simt_model_profile: str  # 可选自定义 profile 路径
auto_simt_model_assets_hash: str  # profile 内容哈希，用于 cache 失效
```

对应环境变量：
- `TRITON_ASCEND_AUTO_SIMT_SCOPE` — 控制 auto/report/off
- `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP` — JSONL 输出路径
- `TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN` — gain margin
- `TRITON_ASCEND_AUTO_SIMT_PROFILE` — 自定义 profile
- `TRITON_ASCEND_COMPILE_MODE` — 设为 `simd_simt` 启用

### 14.2 `_run_cpp_simd_simt_costmodel` 函数

```python
def _run_cpp_simd_simt_costmodel(mod, metadata, opt) -> str:
    mode = opt.auto_simt_scope_mode
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":
        return "backend_default"

    # 1. 运行 select_simd_simt_costmodel pass
    pm = ir.pass_manager(mod.context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(pm, mode, profile, ...)
    # 2. 运行 materialize_simt_scopes pass（兼容性验证）
    ascend.passes.ttir.add_materialize_simt_scopes(pm)
    pm.run(mod)

    # 3. 读取 module attrs 获取决策
    effective = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.effective")
    return effective
```

### 14.3 纯 SIMT 路径的 TTIR 预处理

当选出 `all_simt_only` 或 whole-body SIMT scope 时：
```python
if cpp_all_simt or ascend.ir.is_whole_body_void_simt_scope(mod):
    # 1. RowCoalescing
    ascend.passes.ttir.add_row_coalescing(pm)
    pm.run(mod)
    _export_coalesce_metadata(mod, metadata)

    # 2. Inline void SIMT scopes
    ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)

    # 3. 清除 costmodel attrs（不进入 BiShengIR）
    ascend.ir.clear_simd_simt_costmodel_attrs(mod)

    # 4. 直接返回 TTIR string（跳过其他 pass）
    return str(mod)
```

### 14.4 Profile 路径解析

```python
def _costmodel_profiles_dir() -> Path:
    # 1. 原生 package 路径: .../_C/ascend/costmodel_profiles/
    # 2. 兼容旧 package 路径: .../backend/costmodel_profiles/
    # 3. 源码树路径: .../third_party/ascend/costmodel/profiles/
```

### 14.5 Asset Hash for Cache Invalidation

```python
def _auto_simt_asset_hash(path, default_name) -> str:
    # SHA256("selection-profile\0" + selection_bytes
    #        + "\0shared-microbenchmark\0" + shared_bytes)
```

组合了 selection profile 和它引用的 microbenchmark profile 的哈希，用于 JIT cache key。

### 14.6 BishengIR 标志

Mixed 模式需要特殊的 compiler 标志：
```python
"--enable-lib-call-no-inline=false"         # CANN 9.1 的 hivmc-a5 不能翻译 hacc.noinline
"--enable-hivm-delayed-cross-core-gss=false" # 避免 split-side anchor interval 反转
```

### 14.7 Python 前端 `scope` 增强

```python
# language/cann/extension/scope.py
al.scope(vector_mode="simd")   # 显式 SIMD scope
al.scope(vector_mode="simt")   # 显式 SIMT scope
# 等价别名: vec_mode="simd" / vec_mode="simt"
# 验证: vector_mode 不能与 core_mode="cube" 同时使用
```

---

## 15. 单元测试对比

### 15.1 SimdSimtCostModelTest

**旧版** (254 行，4 个测试)：
- `GatherDotGoldenScoresAndModelAdmission`
- `FbgemmGoldenScoresAndModelAdmission`
- `Rank1IndirectVectorReductionIsCovered`
- `OutOfCoverageAutoSkipsButDiagnosticsStillScore`

**新版** (424 行，8 个测试)：
- `GatherDotGoldenScoresRequireMaterializableMixedPlan` — **重命名**，增加了对 `simdStructuralPenaltyCycles == simdAnalyticalCycles * structuralPenaltyRatio` 的验证
- `FbgemmGoldenScoresRequireMaterializableMixedPlan` — **重命名**，验证了 Event calibration multipliers
- `Rank1IndirectVectorReductionIsCovered` — 不变
- **`SolveTrilBt16StaysOutsideMaskedRowwiseCalibration`** — 新：验证 BT16 不在 masked_rowwise 域
- **`TriangularUnknownLoopUsesBoundedCalibrationException`** — 新：验证三角求解动态循环例外
- **`UnknownLoopWithoutTriangularEvidenceRemainsRejected`** — 新：验证去掉 triangular evidence 后被拒绝
- **`TriangularUnknownLoopStillHonorsAnchorCountAndShapeBounds`** — 新：验证例外仍受限于 anchor count 和 shape 上限
- `OutOfCoverageAutoSkipsButDiagnosticsStillScore` — 不变

### 15.2 Golden Score 值变化

| 测试 | 旧版值 | 新版值 |
|------|--------|--------|
| GatherDot allSimd | 1606.31661024008 | `A_SIMD*(1+Pstruct)*all_simd_multiplier` (Event calibrated) |
| GatherDot allSimtOnly | 1396.79705238268 | 1408.4638112636317 |
| GatherDot mixed | 1417.74900816842 | `uncalibrated_mixed * 0.666478` (domain multiplier) |
| Fbgemm allSimtOnly | 13405.573578357647 | 13400.405822097307 |
| **Schema version** | **6** | **10** |
| **Profile version** | v5 | **v10** |
| Microbenchmark version | v1 | **v2** |

### 15.3 新增 PassesTest

```cpp
TEST(MaterializeSimtScopePreservesEscapingSSAResult)     // SSA result threading
TEST(NativeWholeBodySimtScopeDetectionAndInlining)       // Whole-body scope 检测
TEST(ModelControlledRoutingIgnoresLegacyGlobalForce)     // 模型控制 vs legacy
```

### 15.4 新增 Python Contract Test

```python
# test_compiler_costmodel_contract.py
# - costmodel forces use_bytecode=False
# - _costmodel_profiles_dir() 路径解析
```

---

## 16. UnstructureConversionPass 的桥接变化

旧版的 `UnstructureConversionPass.cpp` 中有静态规则（`isStructured`, `size<64`, `rank<=5`, `routeDiscreteMaskToSimt`）来决定哪些 op 走 SIMT。

新版中这些静态规则被 `shouldUseSimtTemplate()` 替代：
- 所有 backend lowering pass（UnstructureConversion, DiscreteMaskAccessConversion, StridedLoadStoreRewrite, TritonOpConverter, TritonToLinalg）都通过 `shouldUseSimtTemplate(op, legacyForceSimt)` 来决定
- 在 `isModelControlled(op)` 返回 true 时，只看 `hasEnclosingVectorMode(op, "simt")` —— 即 op 必须在 materializer 创建的 `scope.scope{vec_mode="simt"}` 内
- `scope.scope` 在 lowering 中存活（不被擦除），成为 BiShengIR 的 native region contract

---

## 17. 总结：v10 的设计决策

### 17.1 从 "per-op 标记" 到 "anchor plan + scope contract"

旧版的问题是 per-op bool attr 在不同 pass 间容易不一致（某 pass 写了 attr 但另一 pass 没读，或读到了过时的值）。新版用不可变的 `SimtAnchorPlan` + materialize 出的 `scope.scope` region 解决了这个问题——scope 是 IR 的一部分，不存在同步问题。

### 17.2 从 "convex blend" 到 "resource partition"

Convex blend 的问题在于它永远无法 beat 两个端点（除非 transition cost 为负）。新版按 exact anchor plan 做 resource partition——SIMT anchor 的操作按 SIMT 费率计，其余按 SIMD 费率计，两者顺序执行所以 cost 相加。这使得 mixed 模式在 "大部分是 SIMD 友好的 + 小部分是 SIMT 友好的" 场景下有理论上的优势。

### 17.3 从 "嵌入式硬件参数" 到 "分离的 Microbenchmark Profile"

分离的好处：
- 测量可以被多个模型共享
- 测量有明确的 unit/cycle_domain/scope/confidence
- Consumer 按 measurement key 引用，加载时验证单位和时钟域匹配
- 新增测量不会影响已有的 selection profile 格式

### 17.4 从 "纯分析公式" 到 "分析 + 测量残差校正"

纯分析公式的问题是它无法捕捉编译器后端的实际行为（指令调度、寄存器分配、memory coalescing 等）。Event calibration 通过在三个 bounded domain 上测量真实 NPU Event 时间来校正 route-relative residual。校正乘数保留在 report 中，使得 "预测" 和 "校正" 的关系对用户透明。

### 17.5 Confidence 和 Gate 系统

新版有更完整的 confidence 系统：
- 每个测量/operation 有 confidence（none/low/medium/high）
- 取所有参与计算的 confidence 的最低值作为 `absoluteConfidence`
- 当 structural penalty 非零时，与 `rankingConfidence`（来自 profile calibration）取最低
- Gate 检查：`confidenceRank(rankingConfidence) < confidenceRank(minimumConfidenceForDecision)` 则拒绝
- 当前 `minimumConfidenceForDecision = "low"`，所以实际上只要不是 "none" 就能通过

### 17.6 未解决的问题

1. **Directional transition cost 未测量**：当前 setup fallback 来自 standalone empty-VF probe，不是真正的 SIMD→SIMT 方向转换延迟。这导致 mixed cost 中的 setup 部分是 conservative overestimate。

2. **Coverage domain 有限**：只有三个 calibration domain（masked_rowwise_reduction, tiny_irregular_dot, triangular_solve_loop）。大多数 kernel 会落在 `out_of_calibration_domain`。

3. **SIMT memory rate 是 workload-effective**：测量来自 sequential rotating runtime-loop，包含了地址生成和循环开销，不是 intrinsic LSU peak。

4. **Mixed boundary cost 未测量**：`mixedBoundaryCycles` 目前为 0（report 中显示为 `null`）。

5. **DES feedback 未激活**：`david_v100_des_feedback_v1.json` 是占位文件，1672/1727 fallback ops 阻止了规则生效。

---

## 18. 与旧版文档的 Section 对照表

| 旧版 Section | 内容 | 新版对应 |
|-------------|------|---------|
| §1 Gather kernel demo | SIMT vs SIMD 示例 | 不变，无更新 |
| §2 A5 硬件能力 | 背景介绍 | 不变 |
| §3-4 静态决策规则 | `isStructured`, C++ 代码 | **被 §4 SimtAnchorAnalysis 替代** |
| §5 当前规则的问题 | 静态规则的局限 | 基本解决（anchor pattern matching） |
| §6 Costmodel 需要做什么 | 功能需求 | 更新为 §1 概述 |
| §7 源码索引 | 文件列表 | **更新为 §2 目录结构重组** |
| §8 C++ 函数详解 | analyzeFeatures + estimateCandidates | **更新为 §4 + §6 + §7** |
| §9 compile_mode 流 | Python → C++ 流程 | 更新为 §14 |
| §10-11 Select + Materialize | Pass 实现 | **重写为 §10 + §11** |
| §12 三 Pass 协作 | 流程图 | 概念不变，流程简化 |
| §13 关键设计决策 | 总结 | **更新为 §17** |
| §14 PipelineAnalysisPass 关系 | 绝对 cost model vs Route model | 更新为 §3 |
| §15 unstructured_in_simt 的 op 选择 | 静态规则 | **被 SimtAnchorAnalysis + scope.scope 契约替代** |
| §16 simd_simt → indirect_load | bridge 函数 | 更新为 §5 + §16 |
| §17 方法论反推 | microbenchmark → model → calibration | 更新为 §8 + §9 |
| §18 Microbenchmark 拆解 | 25 项测量分类 | **更新为 §9.2（新增 UB 测量 + transition harness）** |

**新增内容（旧版无对应）**：
- §3: 两模型 + 共享证据层架构
- §4.2: 每种 Anchor 的 Lowerability 表
- §7.2: Resource partition 公式详解
- §8: Event Route Calibration
- §12: TriangularSolveLoop
- §13: RowCoalescing
- §14.5: Asset Hash for Cache Invalidation
- §15: 单元测试对比
- §17.6: 未解决的问题

---

> **文档结束** — 总计覆盖 v10 schema 的所有关键架构变化。
> 对照旧版 `autoscope_simit_costmodel_analysis.md` (18 sections, 2252 lines) 阅读效果最佳。
