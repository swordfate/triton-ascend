# Stage / StageCostTable 数据结构梳理

本文整理当前 costmodel 中与 Stage 相关的重要数据结构、字段含义以及它们之间的逻辑关系。

---

## 1. Stage 相关：分析期 vs 成本期

这套代码里有两套 “Stage” 结构，容易混淆：

| 结构 | 阶段 | 位置 |
|---|---|---|
| `LogicalStage` | StagePartitioner 分析期 | `StageCostModels.h` |
| `LogicalStageCost` | StageCostEvaluator 成本期 | `StageRouteCostModel.h` |

---

## 2. `LogicalStage`：分析期的 Stage

它描述 “一个 Stage 拥有哪些操作、结构特征、工作量、合法模式”。

主要字段：

```cpp
struct LogicalStage {
  std::string id;
  StageCostModelKind costModelKind;
  StageScheduleKind scheduleKind;
  int64_t iterationCount;

  StageModelFeatures features;
  StageWorkload workload;

  std::vector<Operation *> operations;

  std::vector<Value> liveIns;
  std::vector<Value> liveOuts;
  int64_t liveInBytes;
  int64_t liveOutBytes;

  int64_t localSimtScopeCount;
  int64_t scopeInputTensorBytes;
  int64_t scopeOutputTensorBytes;
  std::vector<unsigned> simtAnchorIndices;

  bool simdLegal;
  bool simtLegal;
  bool localSimtMaterializable;
  std::vector<int64_t> legalSimtFactors;
  std::vector<int64_t> localSimtFactors;
};
```

它主要回答：

- 这个 Stage 是什么 kind？
- 它有多少次迭代？
- 它有哪些结构特征？
- 它有多少工作量？
- 它拥有哪些真实 TTIR 操作？
- 它能不能做 SIMD / SIMT / local SIMT？
- 如果做 local SIMT，scope 的输入输出是多少？

---

## 3. `StageWorkload`：模式无关的工作量

```cpp
struct StageWorkload {
  llvm::StringMap<double> operationElements;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  double predicateElements = 0.0;
  double shuffleLaneSteps = 0.0;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  double estimatedSpillTransactions = 0.0;
  bool paysKernelSetup = false;
};
```

它是 **和 SIMD/SIMT 无关** 的 “逻辑工作量”，单位是：

- 元素数
- 字节数
- warp 指令数
- FLOPs
- 事务数

注意：`StageWorkload` 里存的是 **per-iteration** 工作量，不是整个 Stage 的总量。

---

## 4. `StageResourceCycles`：模式相关资源周期

```cpp
struct StageResourceCycles {
  double setup = 0.0;
  double scalar = 0.0;
  double load = 0.0;
  double store = 0.0;
  double compute = 0.0;
  double predicate = 0.0;
  double shuffle = 0.0;
  double dot = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  double criticalPath = 0.0;
};
```

它表示：

> 把 `StageWorkload` 通过某个 mode 的 hardware profile 映射后，得到的 **每个 iteration 的资源周期数**。

关系：

```text
StageWorkload       → mapWorkload()       → StageResourceCycles
(逻辑工作量)              (按 SIMD/SIMT profile 映射)   (资源周期)
```

---

## 5. `StageImplementationCost`：一个具体实现

```cpp
struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  StageResourceCycles resources;
};
```

它表示：

> 这个 Stage 的某一个具体实现，例如 “SIMT F2 local scope” 的成本。

一个 Stage 可以有多个 `StageImplementationCost`：

```text
SIMD F1
SIMT F1 global
SIMT F2 global
SIMT F4 global
SIMT F1 local scope
SIMT F2 local scope
...
```

---

## 6. `LogicalStageCost`：成本期的 Stage

```cpp
struct LogicalStageCost {
  std::string id;
  std::string model;
  StageScheduleKind schedule;
  int64_t iterationCount;

  StageModelFeatures features;
  StageWorkload workload;

  int64_t ownedOperationCount;
  int64_t liveInCount;
  int64_t liveOutCount;
  int64_t liveInBytes;
  int64_t liveOutBytes;

  int64_t localSimtScopeCount;
  int64_t scopeInputTensorBytes;
  int64_t scopeOutputTensorBytes;
  std::vector<unsigned> simtAnchorIndices;
  bool localSimtMaterializable;

  std::vector<int64_t> legalSimtFactors;
  std::vector<int64_t> localSimtFactors;

  std::vector<StageImplementationCost> implementations;
};
```

它基本是 `LogicalStage` 的 “成本计算后快照”，额外多了：

```cpp
std::vector<StageImplementationCost> implementations;
```

---

## 7. `StageCostTable`：整个 kernel 的成本表

```cpp
struct StageCostTable {
  std::string domain;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;

  int64_t logicalProgramCountHint = 0;
  int64_t physicalCoreCountHint = 0;

  std::vector<LogicalPhaseCost> phases;
  std::vector<LogicalStageCost> stages;
};
```

它包含：

- domain：例如 `generic_dataflow`
- 是否完整拥有所有 operation
- 使用的 profile 版本
- runtime hint
- 按 Phase 组织的成本视图
- 拍平后的所有 Stage 成本

`StageCostEvaluator::evaluate()` 返回的就是它。

---

## 8. `StageCostModelSummary`：最终路由结果

```cpp
struct StageCostModelSummary {
  bool applied;
  std::string domain;
  ...
  StageTransitionCost transition;

  StageRoutePlan allSimd;
  StageRoutePlan allSimt;
  StageRoutePlan mixed;
};
```

它是在 `StageCostTable` 基础上，经过 `solveStageRoutes()` 后得到的：

- 三个路由各自的总成本；
- 每个路由每个 Stage 选了什么实现；
- 每个路由的 SuperBlock factor；
- wave 数量等。

---

## 9. 它们之间的逻辑关系

```text
StagePartitioner
   │
   ├── LogicalStage
   │     ├── operations
   │     ├── features
   │     ├── workload
   │     ├── liveIn/Out
   │     └── localSimtMaterializable / legalFactors
   │
   ▼
StageCostEvaluator::evaluate()
   │
   ├── 对每个 LogicalStage 生成多个 StageImplementationCost
   │     ├── StageImplementation          (mode / factor / localScope)
   │     ├── StageResourceCycles          (mapWorkload 结果)
   │     └── totalCycles                  (estimateStage + applySuperBlock)
   │
   ▼
StageCostTable
   ├── vector<LogicalPhaseCost>
   │     └── vector<LogicalStageCost>
   │           └── vector<StageImplementationCost>
   │
   ▼
solveStageRoutes()
   │
   ├── 为 all_simd / all_simt / mixed 各选一套实现
   ├── 得到 StageRoutePlan
   │     ├── implementations
   │     ├── logicalStageCycles
   │     ├── entryTransitionCycles
   │     └── totalCycles
   │
   ▼
StageCostModelSummary
```

---

## 10. 关键关系总结

### `StageWorkload` → `StageResourceCycles`

```text
StageWorkload
  + HardwareProfile
  + StageMode (SIMD/SIMT)
  → mapWorkload()
  → StageResourceCycles
```

### `StageResourceCycles` → `totalCycles`

```text
StageResourceCycles
  + StageCostModelKind
  + StageMode
  → estimateStage()
  → 基础 stageCycles

stageCycles
  + StageImplementation (factor/localScope)
  → applySuperBlock()
  → totalCycles
```

### `LogicalStage` → `LogicalStageCost`

```text
LogicalStage
  + HardwareProfile
  → StageCostEvaluator
  → LogicalStageCost
      └── implementations: [StageImplementationCost]
```

### `StageCostTable` → `StageRoutePlan`

```text
StageCostTable
  + StageTransitionCost
  → solveStageRoutes()
  → StageRoutePlan
      ├── 每个 Stage 选一个 implementation
      └── 累加 logicalStageCycles
```

---

## 11. 一句话记忆

> `StageWorkload` 是 “有多少活”，`StageResourceCycles` 是 “某种模式下这些活要多少周期”，`StageImplementationCost` 是 “某个具体 SIMD/SIMT/F 方案要多少周期”，`LogicalStageCost` 是 “一个 Stage 的所有可选方案”，`StageCostTable` 是 “整个 kernel 的所有 Stage 方案表”，`StageRoutePlan` 是 “最终挑出来的一条完整路由”。
