# CostModel 资源周期与 Stage 公式详解

本文整理 costmodel 中与资源周期估算、StageCostModelKind 公式以及语义族设计相关的详细解释。

---

## 1. 核心计算链路

真正决定一个 Stage 需要多少 cycle 的核心是两个函数：

```text
mapWorkload()
  StageWorkload + HardwareProfile + StageMode
  → StageResourceCycles

estimateStage()
  StageResourceCycles + StageCostModelKind
  → stageCycles
```

另外 `applySuperBlock()` 只影响 SIMT 且 `F > 1` 时的最终成本。

`mapWorkload()` 不是每个 Stage 一个，而是所有 Stage 共用的一个函数。  
不同 Stage 结果不同，是因为 workload / features / mode 不同。

---

## 2. 公共成本骨架

几乎所有 Stage 公式都基于：

```text
cost = setup + iterations * body
```

- `setup`：一次性固定开销
- `iterations`：该 Stage 的迭代次数
- `body`：每个 iteration 的核心成本

不是每个 Stage 都会真的付 `setup`：

```cpp
setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
if (work.dotFlops > 0)
  setup += profile.dotSetupCycles;
```

---

## 3. 每个资源周期为什么这么算

### 3.1 setup

含义：一次性固定开销。

来源：

- SIMD：profile 固定 `21.212121`
- SIMT：microbenchmark `simt.setup.empty_with_barrier = 141`

设计思想：kernel 启动、dot 单元启动等只付一次，不应随 iteration 重复放大。

---

### 3.2 scalar

```text
scalar = scalarOperations / scalarOperationsPerCycle
```

来源：

- SIMD：`scalar_operations_per_system_cycle = 1.0`
- SIMT：`scalar_operations_per_system_cycle = 4.0`

设计思想：用“标量操作数 / 平均标量吞吐”估算标量/地址/控制计算成本。

---

### 3.3 load / store

#### SIMD 连续访存

```text
load  = loadBytes  / loadBytesPerCycle
store = storeBytes / storeBytesPerCycle
```

来源：`vector_mte2_bytes_per_system_cycle = mte3_bytes_per_system_cycle = 202.25`，对应约 200GB/s 的 legacy seed。

#### SIMT 连续访存

```text
load  = loadWarpInstructions  / loadWarpInstructionsPerCycle
store = storeWarpInstructions / storeWarpInstructionsPerCycle
```

来源：

- `simt.gm.load.throughput = 0.1761917890625`
- `simt.gm.store.throughput = 0.12914221875`

`loadWarpInstructions` 由 workload 分析计算：

```cpp
loadWarpInstructions += ceil(elements / 32.0);
```

当前模型没有按每个 warp load 实际数据大小区分，这是一个已知简化。

#### indirect / gather

```text
loads  = max(loadWarpInstructions, loadBytes > 0 ? 1 : 0)
stores = max(storeWarpInstructions, storeBytes > 0 ? 1 : 0)

load  = loads  / indirectLoadTransactionsPerCycle
store = stores / indirectStoreTransactionsPerCycle

if (loads + stores > 0)
  load += indirectDependencyLatencyCycles
```

transaction 在这里表示“一次独立的 indirect/gather 访存请求”。

当前参数：

- SIMD：`0.125` transaction/cycle，依赖延迟 `200`
- SIMT：`0.5` transaction/cycle，依赖延迟 `100`

这些是 provisional seed，不是最终精确 microbenchmark。

---

### 3.4 compute

```text
SIMD:
  instructions = ceil(elements / vectorWidth)
  compute += instructions / throughput * factor

SIMT:
  instructions = elements
  compute += instructions / throughput * factor
```

- SIMD 一条指令处理 `vectorWidth` 个元素；
- SIMT 每个 lane 处理一个元素；
- `factor` 表示相对基础 op 的代价倍数，例如 div=12、exp=9、convert=1.5。

---

### 3.5 predicate

```text
SIMD:
  predicate = ceil(predicateElements / vectorWidth) / predicateOperationsPerCycle

SIMT:
  predicate = predicateElements / predicateOperationsPerCycle
```

`predicateElements` 是该 Stage 内所有 compare 类 predicate 元素总和，不是单个 op。

SIMD/SIMT 的 predicate rate 不同：

- SIMD：`3.3`
- SIMT：`141`

---

### 3.6 shuffle

```text
shuffle = shuffleLaneSteps / shuffleLanesPerCycle
```

`shuffleLaneSteps` 来自 reduction workload：

```text
shuffleLaneSteps += 输入总元素数 * ceil(log2(归约维度长度))
```

`shuffleLanesPerCycle`：

- SIMD：`vectorWidth = 64`
- SIMT：`32 * simt.shuffle.throughput ≈ 26.148`

---

### 3.7 dot

```text
if (dotFlops > 0) {
  setup += dotSetupCycles;
  dot = dotFlops / dotFlopsPerCycle;
}
```

dot 有自己的固定启动开销，和 kernel setup 是两类不同的一次性成本。

来源：

- SIMD：setup `128`，throughput `4096`
- SIMT：setup `64`，throughput `141`

---

### 3.8 issue

```text
issue = ceil(issueElements / issueWidth) / issueOperationsPerCycle
```

issue 表示前端发射上限。  
即使各执行单元算得过来，指令发射也可能成为瓶颈，所以很多公式用：

```text
max(execution, issue)
```

而不是相加。

---

### 3.9 spill

```text
spill = estimatedSpillTransactions / spillTransactionsPerCycle
```

当前 `spillTransactionsPerCycle` 固定为 1.0，而 `estimatedSpillTransactions` 目前基本恒为 0，属于预留字段，真正的 spill 估算尚未实现。

---

### 3.10 control flow

```text
loopControl     += loopBackedgeCount     * loopBackedgeCycles
branchControl   += conditionalBranchCount * conditionalBranchCycles
synchronization += synchronizationCount   * synchronizationCycles

SIMT:
divergence += divergentBranchCount * (1 - activeLaneRatio) * divergentBranchPenaltyCycles
```

含义：

- 循环 backedge、分支、同步都有固定控制开销；
- SIMT 分支发散用 `(1 - activeLaneRatio)` 惩罚不活跃 lane。

---

### 3.11 criticalPath

```text
if hasLoopCarriedDataDependency:
  criticalPath = scalar + compute + predicate + shuffle + dot

else if hasReduction:
  criticalPath = compute + predicate + shuffle
```

criticalPath 表示依赖链上不能并行的串行长度，供 recurrence 和 reduction 使用。

---

## 4. 语义族与 Stage 公式

### 4.1 AutoBlockify 族

包含：

```text
auto_blockify_dispatch
auto_blockify_loop
```

公式：

```text
cost = setup + dispatchCount * max(scalar + control, issue)
```

- dispatch：只付一次；
- loop：按 iterations 放大。

含义：调度外壳的标量/控制开销与前端 issue 取瓶颈。

---

### 4.2 连续访存族

包含：

```text
continuous_tile_memory
continuous_tile_store
continuous_short_load
cache_policy_store
```

SIMD overlap：

```text
cost = setup + iterations * (
  scalar + predicate + control + spill
  + max(load, store, issue)
)
```

否则串行：

```text
cost = setup + iterations * max(execution, issue)
```

含义：连续访存可与部分计算 overlap，但周边开销不能完全隐藏。

---

### 4.3 独立流水循环族

包含：

```text
independent_pipelined_loop
```

SIMD overlap：

```text
cost = setup + iterations * (
  max(load, store, compute+dot+shuffle, scalar+predicate+control, issue)
  + spill
)
```

含义：循环间无依赖时，多资源可以并行，取最大瓶颈。

---

### 4.4 recurrence 族

包含：

```text
loop_carried_recurrence
```

```text
critical = max(criticalPath + load + store + control + spill, issue)

SIMD:
cost = setup + iterations * critical

SIMT:
groups = min(parallelRecurrenceGroupCount, logicalWarpGroupCount)
cost = setup + max(ceil(iterations / groups) * critical, iterations * issue)
```

含义：recurrence 内部串行，不能像 independent loop 一样完全并行；SIMT 下多个独立 group 可以并行。

---

### 4.5 行归约族

包含：

```text
rowwise_reduction
```

```text
cost = setup + iterations * max(
  scalar + load + store + criticalPath + control + spill,
  issue
)
```

含义：reduction 的 load/compute/shuffle/store 更偏串行，不能简单 overlap。

---

### 4.6 Cube roofline 族

包含：

```text
cube_roofline
tiny_cube_roofline
```

SIMD overlap：

```text
cost = setup + iterations * (
  scalar + predicate + control + shuffle + spill
  + max(load, compute + dot, store, issue)
)
```

含义：Cube 计算、load、store 可并行，取主要瓶颈；周边工作额外加。

---

### 4.7 转换 / 打包族

包含：

```text
conversion_pack
elementwise_compute
```

SIMD overlap：

```text
cost = setup + iterations * (
  predicate + control + spill
  + max(scalar + compute, load, store, issue)
)
```

含义：cast/pack/elementwise 以计算为主，load/store 作为配合，取计算与访存/发射的瓶颈。

---

### 4.8 默认串行族

包含：

```text
scalar_issue
scalar_control
scalar_math
index_generation
predicate_mask
loop_predicate
indirect_scalar_memory
indirect_gather_memory
```

公式：

```text
cost = setup + iterations * max(execution, issue)
```

含义：这些 Stage 当前没有专门 overlap/roofline 模型，使用最保守的串行估计。  
indirect/gather 的特殊性主要在 `mapWorkload()` 中体现。

---

## 5. 当前限制与后续改进点

- 很多 profile 参数是固定/平均/legacy seed，不是精确 per-workload 值；
- SIMT 连续访存没有区分每个 warp load 的数据大小；
- indirect/gather 的 transaction 数统一使用 `loadWarpInstructions`，没有真正按 SIMD/SIMT 分开；
- `indirect_scalar_memory` 目前不会主动生成，统一落到 `indirect_gather_memory`；
- `estimatedSpillTransactions` 尚未实现，spill 成本恒为 0；
- SIMD structural penalty 等更细校准尚未在 C++ StageCostModels 中生效。
