# CostModel 改动点设计记录

> 本文档用于记录后续对 SIMD/SIMT Route CostModel 的改动设计。每个改动点独立成章，
> 按“背景 / 目标 / 现状代码 / 目标设计 / 迁移步骤 / 风险”组织。
> 当前是第 1 个改动点。

---

## CHANGE-001：Stage 作为唯一评分方法，并去掉 Phase 模板切分

### 1. 背景

当前 StageCostModel V2 已经引入了 Stage 化的成本模型，但评分路径仍然不是“纯 Stage”
的：

- `StagePartitioner` 仍依赖 `PhaseBoundaryAnalysis` 用三个硬编码 domain 模板切分：
  - `TriangularRecurrence`
  - `LoadedIndexRowwiseReduction`
  - `IndirectUnderfilledDot`
- `StageBoundaryAnalysis` 仍通过 `partitionTriangular/partitionRowwise/partitionIndirectDot`
  从全局 `SimdSimtFeatureSummary` 中“取走” workload 来生成 Stage。
- 当 Stage 模型不适用时，仍会走一段旧的 aggregate feature 聚合评分 fallback。
- 全局 `analyzeSimdSimtFeatures` 在核心路径中承担了过多职责：phase 判定、模板切分、
  fallback 评分、合法性判断、报告输出都依赖它。

目标方向是：

1. **Stage 成为唯一的评分方法**，不再保留 aggregate feature fallback。
2. **去掉 Phase 模板切分**，改为直接基于 operation graph / SSA / 控制流 / 循环携带依赖
   做语义 Stage 划分。
3. **全局 `analyzeSimdSimtFeatures` 不再作为核心决策必需输入**，最多保留为诊断/报告用。

### 1.1 结论确认：Stage 划分与 Phase 划分没有逻辑依赖

经过对当前代码的梳理，可以确认一个关键结论：

> **Stage 的划分本质上不依赖 Phase 模板。Phase 只是当前实现中为了让三套 domain 模板
> 能复用而引入的一层人为分组。**

当前代码看起来有依赖，是因为 `StageBoundaryAnalysis` 的输入是 `PhaseBoundaryPlan`：

```text
PhaseBoundaryAnalysis
  -> 选 domain
  -> assignRootPhaseIds
  -> PhaseBoundaryPlan
  -> StageBoundaryAnalysis
```

但这只是**实现层面的耦合**，不是逻辑上的必须。Stage 真正需要的信息是：

- 哪些操作属于同一个算法串行区域；
- 哪些操作具有同一个 dominant semantics；
- 哪些操作属于同一个 local SIMT anchor；
- 操作之间的 SSA / 控制流 / 循环携带依赖。

这些信息都可以直接从：

```text
ModuleOp + SimtAnchorPlan + ProgramStructureAnalysis
```

中得到，并不需要“这个 kernel 是 Triangular / Rowwise / IndirectDot”这种 domain 判断。

因此：

- **直接去掉 Phase 是可行的。**
- 去掉后，`StageBoundaryAnalysis` 不再需要 `PhaseBoundaryPlan` 作为输入。
- 新的切分器可以直接基于 operation graph / SCC / 依赖拓扑生成 Stage。
- 这样就不需要局限于 phase 的三个模板，能显著增强泛化性。

### 2. 目标

- 任意 kernel 的 Stage 划分不再依赖“它长得像 solve_tril / rowwise / gather-dot”这种
  模板匹配。
- 所有候选路线（AllSIMD / AllSIMT / Mixed）都只由 Stage 成本 + 模式切换成本计算。
- Stage 的 workload、features、kind 全部从实际 operation ownership 推导。
- 删除 fallback aggregate 评分，避免同一份报告里存在两套不一致的计费逻辑。
- 新增 kernel 结构时，不需要新增 Phase domain 和状态机。

### 3. 现状代码位置

| 组件 | 位置 | 现状问题 |
|---|---|---|
| 全局 feature 分析 | `SimdSimtCostModel.cpp:1834` `analyzeSimdSimtFeatures` | 核心路径大量依赖它 |
| Phase domain 判定 | `StagePartitioner.cpp:1477` `PhaseBoundaryAnalysis::analyze` | 只有三个硬编码 domain |
| Phase 状态机 | `StagePartitioner.cpp:917` `assignRootPhaseIds` | 每个 domain 写死单调状态机 |
| 模板切分 | `StagePartitioner.cpp:562/662/714` | 从全局 workload 取走工作量 |
| fallback 聚合评分 | `SimdSimtCostModel.cpp:2481` 之后 | Stage 不适用时走第二套计费 |
| 全局 workload 基线 | `StagePartitioner.cpp:1384` `buildKernelStageWorkload` | 仅在模板/fallback/验证中使用 |

### 4. 目标设计

#### 4.1 去掉 Phase 模板

将 `PhaseBoundaryDomain` 和 `assignRootPhaseIds` 从核心路径移除，或降级为纯诊断视图。

新的 Stage 划分流程：

```text
ModuleOp + SimtAnchorPlan
        │
        ▼
ProgramStructureAnalysis
  收集 top-level semantic roots
  处理 AutoBlockify V1 调度壳与算法 body 分离
  处理复合 anchor 顺序归一化
        │
        ▼
SemanticStagePartitioner（新）
  基于 SSA 数据依赖边
  基于控制流边
  基于 scf.for/scf.while 的 loop-carried 依赖边
  用 SCC / 依赖闭包 / 拓扑分层找出：
      - 必须串行的递推/循环携带区域
      - 可并行的独立区域
      - 访存/计算/转换等单一语义区域
        │
        ▼
StagePartition
  stages[] + phases[]（phases 退化为依赖序分组，不参与模板匹配）
```

#### 4.2 Stage 作为唯一评分来源

- `StageCostEvaluator::evaluate` 是唯一产生候选成本的地方。
- `solveStageRoutes` 是唯一产生 AllSIMD / AllSIMT / Mixed 三路线成本的地方。
- 删除 `estimateSimdSimtCandidatesImpl` 中 `if (!stageModel)` 之后的 fallback 分支。
- `report.candidateCosts` 永远来自 `report.stageModel`。

#### 4.3 Stage 信息全部从 operation ownership 推导

- `StageWorkloadAnalysis` 继续从 owned operation tree 计算动态 workload。
- `StageFeatureAnalysis` 继续从 owned operation tree 计算结构特征。
- `StageKindClassifier` 继续从 `StageModelFeatures` 推导 Stage kind。
- StageBoundary 不再需要 `buildKernelStageWorkload(features)` 作为初始分配来源。

#### 4.4 合法性信息从 AnchorPlan / Module 直接获取

- `SimtAnchorPlan` 已经包含：
  - anchor 类型
  - materializable
  - lowerability（all-simd / all-simt / mixed）
  - scopeOperations / scopeInsertionPoint
  - captured / escaping 张量信息（可由 anchor 推导）
- `hasExplicitScope` 直接扫描 `scope.scope` 即可。
- `autoBlockifyV1Applied` 直接读取 module 上的 `ta.auto_blockify_v1*` 属性即可。

#### 4.5 全局 analyzeSimdSimtFeatures 的定位

- 核心路径不再依赖它。
- 可选保留一个轻量版本，仅用于：
  - JSON 报告输出
  - 测试校验
  - 调试诊断
- 如果报告也不需要，则可以直接删除。

### 5. 迁移步骤

1. 新增 `SemanticStagePartitioner`（或重构 `StagePartitioner`）：
   - 输入：`ModuleOp`、`SimtAnchorPlan`、`StagePartitionerOptions`
   - 不再输入 `SimdSimtFeatureSummary`
2. 实现基于 operation graph 的通用切分：
   - 收集 semantic roots
   - 构建 SSA / control / loop-carried 依赖图
   - 用 SCC 和依赖拓扑生成单一语义 Stage
   - 保持现有不变量：operation ownership 守恒、anchor ownership 不重叠、Stage 边界单调
3. 删除 `PhaseBoundaryAnalysis` 的模板 domain 状态机，或降级为诊断；
   `StageBoundaryAnalysis` 不再消费 `PhaseBoundaryPlan`，改为直接消费
   `ProgramStructure` + `SimtAnchorPlan`。
4. 删除 `partitionTriangular / partitionRowwise / partitionIndirectDot` 的模板 workload 分配。
5. 让 `StageWorkloadAnalysis / StageFeatureAnalysis / StageKindClassifier` 成为唯一
   per-stage 信息源。
6. 删除 `estimateSimdSimtCandidatesImpl` 的 fallback aggregate 分支。
7. 将 candidate legality 改为从 `SimtAnchorPlan` / module 属性直接计算。
8. 将 `evaluateStageModel` 的 `features.autoBlockifyV1Applied` 改为直接读 module 属性。
9. 更新单元测试：
   - `PassesTest.cpp` 从验证全局 feature 改为验证 semantic StagePartition。
   - `SimdSimtCostModelTest.cpp` 删除或改写基于 `SimdSimtFeatureSummary` 的 fallback 测试。
10. 保留 `report.features` 为可选诊断，但不得进入决策路径。

### 6. 验收标准

- 任意新的 kernel 结构不需要修改 `PhaseBoundaryDomain` 枚举或状态机。
- 所有三路线成本全部来自同一个 Stage 成本表 + route DP。
- 删除 fallback aggregate 后，已有用例（solve_tril / rowwise / gather-dot）结果仍合理。
- `SimtAnchorPlan` 仍然同时被评分和物化使用，保证“分数与物化同源”。
- 全局 `analyzeSimdSimtFeatures` 如果保留，仅影响报告/测试，不影响决策结果。

### 7. 风险与注意点

- **泛化切分的正确性**：SCC/依赖图切分需要避免把一个 Stage 切成多个同构碎片，
  也不要让一个 Stage 同时拥有多个不兼容的 dominant semantics。
- **现有三用例回归**：solve_tril 的复合 anchor、rowwise 的 loaded-index gather、
  gather-dot 的 tiny dot 都需要在通用切分下保持原来的 Stage 归属。
- **AutoBlockify V1 调度壳**：通用切分必须继续区分“调度循环”和“算法 body”，
  否则 workload 会 double-count。
- **复合 anchor 顺序归一化**：`ProgramStructureAnalysis` 现在做的 scope 重排必须保留，
  否则阶段边界与物化后的程序顺序不一致。
- **fallback 删除影响**：如果某些 kernel 目前只靠 fallback 评分，删除前必须先让
  semantic StagePartitioner 能覆盖它们。

---

## 后续改动点模板

后续新增改动点时，按以下结构追加：

```markdown
## CHANGE-XXX：标题

### 1. 背景
### 2. 目标
### 3. 现状代码位置
### 4. 目标设计
### 5. 迁移步骤
### 6. 验收标准
### 7. 风险与注意点
```