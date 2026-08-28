# StageCostModelKind 与评估公式说明

本文整理当前 costmodel 中所有 `StageCostModelKind` 的含义和性能评估公式，方便后续做公式/参数校准。

---

## 1. 通用计算基础

### 1.1 每个 stage 先算出资源周期

在 `mapWorkload()` 中，根据 stage 的 workload 和当前 mode（SIMD/SIMT）计算：

```text
scalar   = scalarOperations / scalarOperationsPerCycle
load     = loadBytes / loadBytesPerCycle            (SIMD)
           loadWarpInstructions / loadWarpInstructionsPerCycle  (SIMT)
store    = 类似 load
compute  = 各 op elements / op throughput
predicate= predicateElements / predicateOperationsPerCycle
shuffle  = shuffleLaneSteps / shuffleLanesPerCycle
dot      = dotFlops / dotFlopsPerCycle
issue    = ceil(issueElements / issueWidth) / issueOperationsPerCycle
spill    = estimatedSpillTransactions / spillTransactionsPerCycle
```

然后再加上控制流：

```text
control = loopControl + branchControl + divergence + synchronization
```

其中 divergence 只在 SIMT 下按 active lane ratio 惩罚。

### 1.2 每个 stage 的 iteration

```text
iterations(stage) = max(1, stage.iterationCount)
```

### 1.3 serial body

```text
execution = scalar + load + store + compute + predicate + shuffle + dot + control + spill
serialBody = max(execution, issue)
```

也就是说：

```text
串行模型 = setup + iterations * max(execution, issue)
```

---

## 2. 动态循环边界怎么处理

- 如果 `scf.for` 的上下界和 step 都是常量，则能算出真实 trip count；
- 如果无法确定，则使用：

```text
fallbackLoopTripCount = max(1, stage.iterationCount / loopCount)
```

- 在 GenericDataflow 中，`iterations` 取：

```text
max(1, features.staticLoopTripCountMax)
```

所以：

```text
动态循环且无法静态估计时，iterationCount 通常退化为 1
```

也就是说动态边界循环不会被放大，所有循环体成本按一次迭代计算，后续如果需要可以用 camodel 提供 trip count hint 再修正。

---

## 3. 各 StageCostModelKind 的公式

### 3.1 `auto_blockify_dispatch`

含义：

- AutoBlockify V1 的物理 program 分发开销；
- 每个物理 program 只发生一次。

公式：

```text
cost = setup + 1 * max(scalar + control, issue)
```

设计原因：

- 它表示“启动一个物理 program 的一次性调度成本”；
- 不随逻辑 program 数量重复。

---

### 3.2 `auto_blockify_loop`

含义：

- AutoBlockify V1 的逻辑 program 外层循环开销；
- 每个逻辑 program 迭代都会发生。

公式：

```text
cost = setup + iterations * max(scalar + control, issue)
```

设计原因：

- 外层 `scf.for` 每处理一个 logical program 都要付出循环控制/分发成本；
- 所以按 `iterations` 放大。

---

### 3.3 `scalar_issue`

含义：

- 标量 / 地址 / index / 控制类计算为主的 stage。

公式（默认串行）：

```text
cost = setup + iterations * max(execution, issue)
```

### 3.4 `scalar_control`

含义：

- 标量控制流为主。

公式：

```text
同默认串行模型
```

当前 Generic 不主动产生，主要用于后续细分。

### 3.5 `scalar_math`

含义：

- 标量数学计算。

公式：

```text
同默认串行模型
```

### 3.6 `index_generation`

含义：

- 地址 / index 生成。

公式：

```text
同默认串行模型
```

### 3.7 `predicate_mask`

含义：

- mask / predicate 生成。

公式：

```text
同默认串行模型
```

### 3.8 `loop_predicate`

含义：

- 循环 predicate。

公式：

```text
同默认串行模型
```

---

### 3.9 `continuous_tile_memory`

含义：

- 连续 tile 访存（load 为主）。

SIMD overlap 时：

```text
cost = setup + iterations * (scalar + predicate + control + spill
       + max(load, store, issue))
```

否则：

```text
cost = setup + iterations * max(execution, issue)
```

### 3.10 `continuous_tile_store`

含义：

- 连续 tile store。

公式与 `continuous_tile_memory` 相同，只是资源上 store 占主导。

### 3.11 `continuous_short_load`

含义：

- 连续短 load。

公式同 `continuous_tile_memory`。

### 3.12 `cache_policy_store`

含义：

- 带 cache policy 的 store。

公式同 `continuous_tile_store`。

---

### 3.13 `indirect_scalar_memory`

含义：

- 标量 indirect memory。

公式：

```text
默认串行模型
```

### 3.14 `indirect_gather_memory`

含义：

- indirect / gather 访存。

默认串行模型：

```text
cost = setup + iterations * max(execution, issue)
```

其中 load 使用：

```text
indirectLoadTransactionsPerCycle
indirectDependencyLatencyCycles
```

---

### 3.15 `independent_pipelined_loop`

含义：

- 无循环间数据依赖，可流水。

SIMD overlap 时：

```text
cost = setup + iterations * (
         max(load, store, compute+dot+shuffle,
             scalar+predicate+control, issue)
         + spill
       )
```

否则：

```text
cost = setup + iterations * max(execution, issue)
```

### 3.16 `loop_carried_recurrence`

含义：

- 有循环间数据依赖，串行 recurrence。

SIMD：

```text
critical = max(criticalPath + load + store + control + spill, issue)
cost = setup + iterations * critical
```

SIMT：

```text
groups = min(parallelRecurrenceGroupCount, logicalWarpGroupCount)
cost = setup + max(ceil(iterations / groups) * critical,
                   iterations * issue)
```

设计原因：

- recurrence 内部串行，但不同 logical program / group 之间可以并行隐藏延迟；
- 所以 SIMT 用 group 数分摊 critical path。

---

### 3.17 `rowwise_reduction`

含义：

- 按行 / axis 做 reduction 或 scan。

公式：

```text
cost = setup + iterations * max(
         scalar + load + store + criticalPath + control + spill,
         issue
       )
```

其中：

```text
criticalPath = compute + predicate + shuffle
```

---

### 3.18 `cube_roofline`

含义：

- 大 dot / matmul，cube 单元 roofline。

SIMD overlap 时：

```text
cost = setup + iterations * (
         scalar + predicate + control + shuffle + spill
         + max(load, compute+dot, store, issue)
       )
```

否则：

```text
cost = setup + iterations * max(execution, issue)
```

### 3.19 `tiny_cube_roofline`

含义：

- 小 dot，低于 tinyDotFlopsMax。

公式与 `cube_roofline` 相同，只是判定为 tiny dot。

### 3.20 `conversion_pack`

含义：

- cast / convert / pack / unpack 类操作。

SIMD overlap 时：

```text
cost = setup + iterations * (
         predicate + control + spill
         + max(scalar+compute, load, store, issue)
       )
```

否则：

```text
cost = setup + iterations * max(execution, issue)
```

---

## 4. SuperBlock factor 如何影响 SIMT 成本

当 SIMT 使用 F2/F4 时：

```text
factor = implementation.superblockFactor
```

一般公式：

```text
issueFloor = setup + factor * iterations * issue
latencySensitive = iterations * (load + store + shuffle + divergence)
groupedBody = factor * max(0, body - latencySensitive)
              + factor * latencySensitive / effectiveFactor

cost = max(issueFloor, setup + groupedBody + pressure)
       + persistentStatePressure
```

对于 `loop_carried_recurrence`：

```text
cost = max(issueFloor, setup + recurrenceBody + pressure)
       + persistentStatePressure
```

设计原因：

- SuperBlock 用多个独立 logical program 隐藏 latency；
- 但不能增加 issue 带宽；
- 所以有 issueFloor 兜底；
- recurrence 的串行部分不能通过 F 降低，只能靠多个 group 并行。

---

## 5. 当前实际会产生的 StageCostModelKind

当前 GenericDataflow 和专用切分主要产生：

```text
auto_blockify_dispatch
auto_blockify_loop
scalar_issue
continuous_tile_memory
continuous_tile_store
indirect_gather_memory
independent_pipelined_loop
loop_carried_recurrence
rowwise_reduction
conversion_pack
cube_roofline / tiny_cube_roofline
```

以下 kind 当前基本不会主动产生，但公式已经存在：

```text
scalar_control
scalar_math
index_generation
predicate_mask
loop_predicate
continuous_short_load
cache_policy_store
indirect_scalar_memory
```

---

## 6. 调参建议

- 如果某个 stage 被估得太贵/太便宜，先看它是哪种 kind；
- 然后去 `david_v100_simd_simt_v1.json` 调整对应的 rate；
- 如果 rate 调不动，再去 `StageCostModels.cpp` 调整对应 kind 的公式。

重点参数：

```text
simd.stage_resources.scalar_operations_per_system_cycle
simd.stage_resources.issue_instructions_per_system_cycle
simt.stage_resources.scalar_operations_per_system_cycle
simt.stage_resources.issue_instructions_per_system_cycle
simt.memory / load_throughput
simt.shuffle
simt.camodel_effective
```
