# Stage-based SIMD/SIMT Route Cost Model

## 本轮重构的五个核心改造

| # | 改造 | 设计结论 |
|---:|---|---|
| 1 | 分段式 CostModel | `StagePartitioner` 将 Kernel 切成 Phase/Stage；每个 Stage 只计算 SIMD/SIMT 候选，Stage 求和得到 Phase，Phase 求和得到 Kernel mixed route |
| 2 | 去掉价值不大的门禁 | 删除 Coverage、Calibration Domain、置信度、Gain Margin 对 route 的截断；只保留编译正确性必需的合法性检查 |
| 3 | 考虑 AutoBlockify V1 | CostModel 位于 V1 之后，识别其新建的物理核调度循环，与算法循环分离计费 |
| 4 | 考虑 Layout 合并与优化 | CostModel 使用 ImplicitPermute、StridedAxisCoalescing、TileChunkCoalescing 之后的访存形态 |
| 5 | SIMT Scope SuperBlock | SuperBlock 仍是 Kernel 级调度；mixed kernel 内批处理 factor 个 logical program，SIMT Stage 由 factor 个 warp group 并行执行 |

## 1. 目标和架构

Route Model 不再把整个 kernel 压缩成一个 aggregate feature 向量后直接计算
`all-SIMD / all-SIMT / mixed` 三个总分，而是让一组职责清晰的组件依次产生
`StagePartition -> StageCostTable -> KernelRoutePlan -> RoutedTTIR`。

### 1.1 总体软件架构

在线编译链直接用主类/模块名表示，Profile/Calibration 作为离线旁路：

```text
Raw TTIR
   |
   v
+-----------------------------------+
| LayoutMergePipeline              |
| AutoBlockifyV1                   |
| MemoryAccessAnalysis             |
+-----------------------------------+
   | PreparedTTIR + MemoryAccessFacts
   v
+-----------------------------------+
| StagePartitioner                  |
| StageFeatureAnalysis             |
| StageWorkloadAnalysis            |
| StageModeLegalityAnalysis        |
+-----------------------------------+
   | StagePartition { Phase[] -> Stage[] }
   v
+-----------------------------------+       HardwareProfile
| StageCostEvaluator                | <---------------------------+
| StageCostModelRegistry            |                              |
| SIMDStageCostModel                |                              |
| SIMTStageCostModel                |                              |
+-----------------------------------+                              |
   | StageCostTable                                      |
   v                                                     |
+-----------------------------------+                              |
| KernelRouteSolver                 |                              |
| TransitionCostModel / DP          |                              |
+-----------------------------------+                              |
   | KernelRoutePlan                                    |
   v                                                     |
+-----------------------------------+      +--------------------------+
| SIMTScopeMaterializer            |      | ProfilePublisher         |
| ScopeSuperBlockPass              |      | MicrobenchmarkRunner     |
|                                  |      | CaModelAnalyzer          |
|                                  |      | EventValidator           |
+-----------------------------------+      +--------------------------+
   | RoutedTTIR
   v
Backend lowering -> NPU binary
```

#### 主组件与包含组件

| 主组件/模块 | 输入 | 包含或调用的组件 | 输出 | 不负责什么 |
|---|---|---|---|---|
| `LayoutMergePipeline` + `AutoBlockifyV1` | Raw TTIR、compile options | ImplicitPermute、StridedAxisCoalescing、TileChunkCoalescing、只读 `MemoryAccessAnalysis` | `PreparedTTIR`、`MemoryAccessFacts` | 不切 Stage，不选择 SIMD/SIMT |
| `StagePartitioner` | PreparedTTIR、MemoryAccessFacts | `StageFeatureAnalysis`、`StageWorkloadAnalysis`、`StageModeLegalityAnalysis` | `StagePartition` | 不计算 cycle，不决定 kernel route |
| `StageCostEvaluator` | StagePartition、HardwareProfile | `StageCostModelRegistry`、SIMD 模型树、SIMT 模型树、`ProfileProvider` | `StageCostTable` | 不改变 Stage 边界，不选择全局路线 |
| `KernelRouteSolver` | StagePartition、StageCostTable | `TransitionCostModel`、`RouteLegalityChecker`、动态规划求解、`RouteClassifier` | `KernelRoutePlan` | 不重新计算 Stage 成本，不改写 TTIR |
| `SIMTScopeMaterializer` | PreparedTTIR、StagePartition、KernelRoutePlan | `ScopePlanBuilder`、`ScopeSuperBlockPass`、Backend Pass Integration | `RoutedTTIR`，随后生成 binary | 不重新切 Stage，不重新评分 |
| `ProfilePublisher` | microbenchmark、CaModel、Event、HIVM/汇编分析 | `MicrobenchmarkRunner`、`CaModelAnalyzer`、`EventValidator` | versioned `HardwareProfile` | 不在线覆盖 KernelRoutePlan，不充当路线门禁 |

#### 核心概念与关键数据对象

下表统一定义后文使用的概念和组件间传递的数据对象；它们不是新的软件组件：

| 名称 | 类别 | 定义或内容 | 生产者 → 消费者 |
|---|---|---|---|
| `PreparedTTIR` | 数据对象 | 已完成 layout 合并与 AutoBlockify V1 的 TTIR | `LayoutMergePipeline/AutoBlockifyV1` → `StagePartitioner/SIMTScopeMaterializer` |
| `MemoryAccessFacts` | 数据对象 | structured/unstructured、stride、mask、连续字节和预计访存路线 | `MemoryAccessAnalysis` → `StagePartitioner` |
| `StagePartition` | 数据对象/层级 | 切分结果，包含有序 `Phase[]`、operation ownership、依赖和 live-in/out | `StagePartitioner` → `StageCostEvaluator/KernelRouteSolver/SIMTScopeMaterializer` |
| `Phase` | 执行层级 | 串行逻辑阶段；可以是算法阶段，也可以是 AutoBlockify V1 引入的 Physical Dispatch 阶段 | 包含于 `StagePartition` |
| `Stage` | 建模层级 | Phase 内使用一个主成本公式、且执行模式单一的最小单元，例如 predicate、连续 load、递推循环、Cube dot | 包含于 `Phase` |
| `StageMode` | 枚举概念 | Stage 的硬件执行模式，只有 `SIMD` 或 `SIMT`；不存在 mixed StageMode | `StageModeLegalityAnalysis` 产生合法集合，CostModel/RouteSolver 消费 |
| `StageCostModelKind` | 枚举概念 | Stage 的结构语义，回答“这段代码在做什么”，例如 `scalar_issue`、`continuous_tile_memory`、`loop_carried_recurrence`、`cube_roofline`；本身不指定 SIMD/SIMT | `StagePartitioner` → `StageCostModelRegistry` |
| `StageImplementation` | 候选对象 | 一个 Stage 的合法执行候选，由 `(StageMode, superblock_factor)` 标识；SIMD factor 固定为 1，SIMT 可有 F1/F2/F4 | `StageCostEvaluator` 枚举 |
| `StageImplementationCost` | 成本对象 | 单个候选的总 cycles、资源分解、合法性和来源 | 具体 `StageCostModel` → `StageCostTable` |
| `superblock_factor` | 参数概念 | SIMT implementation 的并发变体；F1/F2/F4 都属于 `StageMode::SIMT`，不是新的 Mode | `StageCostEvaluator` 枚举，SIMT 模型和 Materializer 消费 |
| `Transition` | 边概念/成本 | 相邻 Stage 选择不同 StageMode 时的一次方向性切换，例如 `SIMD→SIMT`；不属于 Stage 内部 | `TransitionCostModel` → `KernelRouteSolver` |
| `StageCostTable` | 数据对象 | 每个 Stage 的 SIMD、SIMT-F1/F2/F4 合法候选及 cycle 分解 | `StageCostEvaluator` → `KernelRouteSolver` |
| `KernelRoute` | 路线概念 | 为每个 Stage 选择一个 implementation 后形成的完整路线；全 SIMD、全 SIMT 或 mixed | `KernelRouteSolver` 求解 |
| `KernelRoutePlan` | 数据对象 | 最优 KernelRoute、模式切换边、连续 SIMT Stage 区间和总成本 | `KernelRouteSolver` → `SIMTScopeMaterializer` |
| `HardwareProfile` | 数据对象 | SIMD/SIMT Rate、带宽、延迟、控制流、spill、SuperBlock 与边界参数 | `ProfilePublisher` → `StageCostEvaluator/TransitionCostModel` |
| `RoutedTTIR` | 数据对象 | 已按 KernelRoutePlan 物化 SIMT scope 和 SuperBlock 调度的 TTIR | `SIMTScopeMaterializer` → Backend lowering |

`StageMode` 与 `StageCostModelKind` 是两个正交维度，共同决定具体成本模型：

```text
StageCostModelRegistry.lookup(StageMode, StageCostModelKind)

(SIMD, scalar_issue)              -> SIMDScalarStageCostModel
(SIMT, scalar_issue)              -> SIMTScalarStageCostModel
(SIMD, loop_carried_recurrence)   -> SIMDRecurrenceStageCostModel
(SIMT, loop_carried_recurrence)   -> SIMTRecurrenceStageCostModel
```

Phase 和 Stage 都不能为了凑模型分数任意切开；边界必须来自算法顺序、数据依赖、执行模式边界
或目标流水语义。候选路线的统一公式为：

```text
OperatorCost(route) = sum(PhaseCost[p])

PhaseCost[p] = sum(StageRouteCost[p][s])

StageRouteCost[p][s]
  = TransitionCost(previous_stage.mode,
                   current_implementation.mode)
  + StageInternalCost[p][s][current_implementation]
```

第一个 Stage 的 `EntryTransitionCost` 为零。这样 SIMD/SIMT 边界成本归入后一个 Stage，
不会成为无法归属的算子外附加项，同时满足“Stage 串行求和得到 Phase，Phase 串行求和得到算子”。

`all-SIMD`、`all-SIMT` 和 mixed 不再拥有三套独立公式。每个 Stage 只选择 SIMD 或 SIMT；
mixed 只表示整条 Kernel route 同时包含两种 StageMode。

### 1.2 CostModel 前后的 TTIR Pass 顺序

当前路由模型不再直接观察原始 TTIR，而是观察 layout merge 和
AutoBlockify V1 之后的 post-transform TTIR：

```text
原始 TTIR
  -> TTIR Layout Merge
       -> ImplicitPermute
       -> StridedAxisCoalescing
       -> TileChunkCoalescing
  -> AutoBlockify V1（变换合法性检查通过时，factor=1 基础循环）
  -> StagePartitioner + StageCostModel + RouteSolver
  -> Materialize scope.scope<vector_mode="simt">
  -> ScopeSuperBlockPass（设计中）
  -> TritonControlFlowOpt
  -> TritonToStructured（第一次）
  -> DiscreteMaskConversion
  -> TritonToAnnotation
  -> TritonToUnstructured
  -> TritonToHIVM / HFusion / LLVM
  -> BubbleUpOperation
  -> TritonToStructured（第二次）
  -> TritonToLinalg
```

因此 CostModel 已经能看到连续化、tile 聚合和 V1 调度循环的结果，但当前
还看不到 `TritonToStructured/Unstructured` 的最终分类。正确的补齐方式不是
把破坏性 lowering 整体前移，而是把其 pointer/mask 部分抽成 CostModel 前的
只读 Analysis，提供 structured/unstructured、stride、mask、连续字节和预计访存路线。

## 2. StagePartitioner

`StagePartitioner` 把经过 layout merge 和 AutoBlockify V1 的 TTIR 转换成 CostModel、
`KernelRouteSolver` 与 `SIMTScopeMaterializer` 共同使用的 `StagePartition`。它只回答四个问题：

1. Kernel 按什么顺序执行哪些 Phase/Stage；
2. 每个 Stage 包含哪些 TTIR operation；
3. 每个 Stage 有什么结构特征和静态工作量；
4. 每个 Stage 可以合法使用 SIMD、SIMT 以及哪些 SuperBlock factor。

它不读取 HardwareProfile，不计算 cycle，也不选择最终路线。

### 2.1 输入、输出与组件边界

| 项目 | 内容 |
|---|---|
| 输入 | `PreparedTTIR`、`MemoryAccessFacts`、target/backend capabilities、compile options |
| 输出 | `StagePartition { Phase[] -> Stage[] }` |
| 上游 | `LayoutMergePipeline`、`AutoBlockifyV1`、`MemoryAccessAnalysis` |
| 下游 | `StageCostEvaluator`、`KernelRouteSolver`、`SIMTScopeMaterializer` |
| 禁止依赖 | HardwareProfile、Event/CaModel 测量总量、workload/kernel 文件名、历史 route 结果 |

### 2.2 组件内流程

```text
PreparedTTIR + MemoryAccessFacts
  -> ProgramStructureAnalysis
       构建有序 semantic roots；分离 AutoBlockify 调度 shell 与算法 body；
       按 compound scope 的计划插入点生成“物化后逻辑顺序”
  -> PhaseBoundaryAnalysis
       为每个 semantic root 生成不可变 rootPhaseId，识别算法串行阶段
  -> StageBoundaryAnalysis
       按语义、依赖、控制区域和可物化边界产生候选 Stage
  -> StageWorkloadAnalysis
       统计该 Stage 精确拥有的静态工作量，不乘硬件 Rate
  -> StageFeatureAnalysis
       从 owned operations 提取循环、依赖、访存与控制流事实
  -> StageKindClassifier
       由真实 operation/features 分类唯一 StageCostModelKind，并验证强结构
  -> StageModeLegalityAnalysis
       计算 SIMD/SIMT 合法性与合法 SuperBlock factor
  -> StagePartitionVerifier
       检查 ownership、anchor、live-in/out、单一 Kind 和边界一致性
  -> StagePartition
```

前三项建立候选边界；后三项给候选 Stage 增加 feature、workload、合法模式和 Kind。如果一个
候选 Stage 需要两个主成本公式，例如“离散 gather + Cube dot”，Verifier 要求回到
`StageBoundaryAnalysis` 将其拆开。这个过程是结构收敛，不根据哪个模式分数更低反向修改边界。

#### 2.2.1 PhaseBoundaryAnalysis：识别算法级串行边界

`PhaseBoundaryAnalysis` 在 `PreparedTTIR` 上工作。它的输入不是算子名或历史性能，而是
`ProgramStructureAnalysis` 产生的 operation 顺序、region/control-flow、SSA def-use、memory effect
和 AutoBlockify V1 provenance。输出是有序的 `PhaseCandidate[]`。

边界识别分两步，不能把“结构识别”和“成本较低”混在一起：

1. `ProgramStructureAnalysis` 按 TTIR 执行顺序收集 semantic root。普通 region operation 作为一个 root
   传递拥有其内部 operation；AutoBlockify V1 是唯一特例：外层 `scf.for` 只拥有调度 shell，循环 body 的
   direct operations 重新暴露为算法 semantic roots，防止整个算法被调度 Stage 重复拥有；
2. compound scope 可能把无副作用的 tensor setup 从输入 load 前移动到 load 后。分析不修改 TTIR，
   但会按 `scopeInsertionPoint` 重排自己的 root 视图，形成 route 实际物化后的逻辑顺序。例如
   `setup → load → recurrence` 被规范化为 `load → scope(setup + recurrence)`；scope roots 必须在该
   逻辑顺序中连续，否则边界分析直接失败；
3. `PhaseBoundaryAnalysis` 根据 ordered roots、SIMT anchor facts 和依赖结构识别受支持的算法域，再按
   串行屏障产生 Phase。当前已实现的域是 triangular recurrence、loaded-index rowwise reduction 和
   indirect under-filled dot；识别依据是 TTIR 结构事实，不读取 kernel/workload 文件名；
4. 输出不是模糊的边界位置，而是与 `rootOperations` 等长的 `rootPhaseIds`。每个 root 恰好属于一个
   Phase，Phase ID 沿逻辑执行顺序单调，已经关闭的 Phase 不允许再次出现。`StageBoundaryAnalysis`
   只能消费这组不可变 Phase ownership，不能按成本重新切 Phase。

通用 CFG/SSA SCC 分析是未来扩展任意控制流 kernel 的方法，但不是当前三类域已经使用的实现；文档不再把
它写成既成事实。

Phase 边界的核心定义是：**一个 Phase 是不能与后继 Phase 在算法语义上重叠执行的最大有序区域**。

| 应切 Phase 的情况 | 原因 | Triton 示例 |
|---|---|---|
| AutoBlockify dispatch 与 logical-program loop | 物理 program 调度与逻辑计算的执行次数不同 | `physical_pid` 映射到多个 `logical_pid` 后进入循环 |
| 完整数据生成后才能进入下一算法步骤 | 存在算法级 producer/consumer 屏障 | solve_tril 的 diagonal load → diagonal inverse |
| 递推完成后结果才可被后续块消费 | 下游依赖完整 loop-carried state | diagonal inverse → merge |
| 有序副作用或控制区域改变算法阶段 | 不能跨边界重排或物化 | atomic/有序 store 前后的区域 |

以下情况本身不形成 Phase 边界：普通 SSA 链、`ptr += stride` 地址递推、同一 tile 内的
compare/select、仅为了让某条 route 分数更低而切分。

#### 2.2.2 StageBoundaryAnalysis：识别单一成本语义边界

`StageBoundaryAnalysis` 在每个 Phase 内继续切分。它输出 `LogicalStage[]`，目标是让每个 Stage
只需要一个 `StageCostModelKind`，并且可以作为一个完整 SIMD 或 SIMT scope 被物化。

当前实现按 ordered semantic roots 做确定性归属：

1. 先识别强结构：AutoBlockify schedule、recurrence/reduction、dot、loaded-index memory、连续 load/store、
   conversion/pack、predicate/index；
2. compound SIMT anchor 使用 `SimtAnchorPlan.scopeOperations` 给出的完整 operation 集合，全部根必须落在
   同一 Stage；anchor 只是边界证据，不参与 cost 比较；
3. 结构化循环默认由一个 Stage 传递拥有内部 operation。真实 loop-carried state 与循环内 predicate、
   reduction/update 同属 recurrence Stage；不能把循环 body 按普通 operation 次序错误拆开；
4. `StageKindClassifier` 用 owned operations 分类并验证 recurrence、reduction、dot、memory、conversion
   等强结构。边界模板给出的 Kind 与真实结构不匹配时，先按 operation graph 收敛到结构上唯一的 Kind；
   例如没有 FP8 conversion、只有 pointer-induction loop 的尾段会从 `conversion_pack` 收敛为
   `independent_pipelined_loop`。若同一 Stage 同时需要两个不兼容的主公式，则报 `requires_split`，不能
   用换 Kind 掩盖错误边界；
5. 对每个 Stage 的传递 operation 集计算 SSA live-in/live-out。定义在 Stage 外而在内部使用的 Value 是
   live-in；定义在内部且被 Stage 外 operation 使用的 Value 是 live-out；
6. `StagePartitionVerifier` 检查每个 semantic root 恰好归属一次、每个 SIMT anchor 恰好归属一个可物化
   Stage，并保证 route-selected Stage 与最终物化 anchor 一致。

| 应切 Stage 的情况 | 原因 | Triton 示例 |
|---|---|---|
| `StageCostModelKind` 改变 | 需要不同主成本公式 | indirect `tl.load` → `tl.dot` |
| 可重叠调度变为真实递推关键路径 | Roofline 不再适用 | 独立 tile load → loop-carried row update |
| 连续访存变为离散访存 | SIMD MTE 与 SIMT DCache 的映射公式不同 | block-pointer load → loaded-index gather |
| 合法 mode 或 scope 可物化边界改变 | 一个 scope 无法保持类型、layout 或副作用语义 | SIMT recurrence result 返回 SIMD consumer |
| 计算资源路径改变 | 资源吞吐率不同 | Cube `tl.dot` → MTE/store |

两个容易误切的例子：

- 随递推循环每次迭代变化的 mask/predicate 属于 recurrence Stage 的关键路径，不能仅因为它是 predicate
  就强行切成独立 Phase；
- solve_tril 的 merge/store 可以属于同一 Phase，但 dot 与 store 应是两个 Stage，因为前者按 Cube/compute
  计费，后者按 memory transaction 计费。

#### 2.2.3 边界判定顺序与冲突处理

边界分析按固定优先级执行，CostModel 分数不参与切分：

| 优先级 | 判定 | 形成的边界 | 代码动作 |
|---:|---|---|---|
| 1 | AutoBlockify V1 调度 shell 与算法 body | Phase + Stage | shell 只拥有 dispatch/loop control；body direct operations 重新成为 semantic roots |
| 2 | 算法级 producer/consumer 屏障、完整递推状态依赖、有序副作用 | Phase + Stage | `PhaseBoundaryAnalysis` 为每个 ordered root 写入不可变 `rootPhaseId` |
| 3 | region/loop 是一个传递 ownership 单元 | Stage 候选边界 | 普通 `scf.for/while` 连同 body 归一个 Stage；不会把循环体按文本 operation 拆散 |
| 4 | 主成本语义改变：dot、reduction/recurrence、indirect memory、continuous memory、conversion | Stage | `StageBoundaryAnalysis` 按 ordered roots 归属到不同 Stage |
| 5 | SIMT anchor 的 operation 集与 insertion point | 约束既有边界 | compound roots 必须在物化后的逻辑顺序连续，且完整归属一个可物化 Stage |
| 6 | 相邻候选同 Phase、同 Kind、同 schedule，且 SSA/layout/副作用允许 | 可合并 | 只在所有条件都满足时合并；不能为了减少 TransitionCost 合并 |

切分后再从实际 owned operation tree 提取 feature。`StageKindClassifier` 的作用不是重新发明边界，
而是做结构复核：模板 Kind 缺少证据时收敛为真实 Kind；如果同一 Stage 同时拥有两套互不兼容的
主公式，例如 `indirect gather + tt.dot` 或 `tt.dot + reduction`，立即返回
`requires_split`。这属于边界错误，不能由 Profile、惩罚项或优先选某个 Kind 掩盖。

`conversion` feature 本身不是强制切分条件：predicate-to-float、accumulator cast 等转换可以是
recurrence/reduction/dot 的辅助工作；只有 conversion/quantize/pack 成为该区域的主语义时，才分类为
`conversion_pack` Stage。边界判定不能把“出现一个转换 op”误当成“转换模型主导”。

当前三个目标结构域在 `StageBoundaryAnalysis` 内已经显式切开这些强结构；`requires_split` 是生产代码
中的守卫和单元测试。任意未知 CFG 的自动回切仍属于后续通用化工作，不能写成已完成。

最终必须同时满足：`rootPhaseIds` 单调连续、Stage ordinal 单调连续、每个 semantic root 精确归属一次、
每个 materializable anchor 精确归属一个 Stage、live-in/out 从同一 ownership 集导出。任一不变量失败就
终止 Stage CostModel，不允许回退到聚合分数继续选路。

#### 2.2.4 当前实现状态与缺口

| 项目 | 当前状态 | 仍需补齐 |
|---|---|---|
| `ProgramStructureAnalysis` | 已消费 post-layout/post-AutoBlockify TTIR；区分 V1 shell/body；按 compound scope 插入点规范化逻辑 root 顺序 | 任意嵌套 CFG 的统一 region graph |
| `PhaseBoundaryAnalysis` | 三类结构域已实现；输出等长的 ordered semantic roots / immutable `rootPhaseIds`，并验证 Phase 连续性 | 扩展到未知域的通用 SCC/side-effect 分区 |
| `StageBoundaryAnalysis` | 三类域均为 operation-graph ownership；无重叠、无遗漏 | 更通用的 nested-region 可切分边界 |
| `StageWorkloadAnalysis` | 从每个 Stage 的真实 owned operation 统计；循环 body 乘静态/结构 trip count，再归一成 `N_iter × per-iteration work`；不再按权重分摊总量 | 补齐 mask transaction、live bytes 与更精确的 conversion/pack 计数 |
| `StageFeatureAnalysis` | 从真实 operation tree 提取 loop-carried、pointer induction、memory、reduction、dot、control；pointer-only iter_arg 不再误判为数据递推 | 接入只读 `MemoryAccessFacts` 以区分更多 stride/cache 形态 |
| `StageKindClassifier` | 强结构匹配时保留语义 Kind；不匹配时按 operation graph 收敛唯一结构 Kind；不兼容主结构返回稳定的 `requires_split` 错误，并有单元测试 | 未知 CFG 的通用自动回切与重新分区机制 |
| live-in/live-out | 已从 SSA crossing 精确生成并输出数量 | 输出字节数、layout contract 与峰值 live set |
| Route → Materializer | mixed route 只物化被选为 SIMT 的 Stage 所拥有的 anchor | mixed scope 的 F2/F4 `ScopeSuperBlockPass` |
| feature-summary overload | 仅保留给无 ModuleOp 的单元测试/兼容 API，并显式标记 `feature_summary_fallback` | 不能用于生产校准结论 |

因此，三类目标用例的边界/ownership 报告现在可以用于下一步 CaModel 对照；但尚未实现的通用 CFG 分区、
MemoryAccessFacts 和 scope F2/F4 不能伪装成已完成。逐 Stage cycle 是否准确仍必须由真实 CaModel 数据校准。

### 2.3 StageFeatureAnalysis

`StageFeatureAnalysis` 描述“Stage 的结构是什么”，输出 `StageModelFeatures`：

| 特征组 | 主要字段/事实 | 用途 |
|---|---|---|
| 循环 | `has_loop`、trip count、backedge 数 | 选择直线、独立循环或递推模型 |
| 迭代依赖 | `has_loop_carried_data_dependency`、recurrence values | 禁止把真实递推错误套入 SIMD Roofline |
| 地址递推 | `has_pointer_induction` | 只计地址生成；不等同于真实数据递推 |
| 访存形态 | continuous/strided/indirect、mask、stride、预计连续字节 | 区分 MTE 连续路径与 DCache/离散 transaction 路径 |
| 计算结构 | scalar/vector、dot、reduction、conversion/pack | 选择 StageCostModelKind 和硬件资源映射 |
| 控制流 | branch、backedge、sync、divergent branch、active-lane ratio | 计算 SIMD predicate 或 SIMT divergence/reconvergence |
| 数据生命周期 | live-in/out、loop-carried state、峰值 live values | 估计 transition、寄存器压力与 spill 风险 |

分析必须基于 TTIR def-use、region/control-flow 和 `MemoryAccessFacts`。不能通过算子名识别
solve_tril、FBGEMM 或 gather-dot-min，也不能输出“SIMT 更合适”之类的路线结论。

### 2.4 StageWorkloadAnalysis

`StageWorkloadAnalysis` 描述“Stage 需要完成多少工作”，输出与执行模式无关的
`StageWorkload`：

| 工作量组 | 统计内容 |
|---|---|
| Memory | 每个 load/store 的 element bytes、元素数、地址类别、mask 覆盖率、连续跨度 |
| Scalar/Vector | integer、FP、compare、select、address-generation 等 operation count |
| Dot/Cube | dot 次数、M/N/K、dtype、accumulator type |
| Reduction/Shuffle | reduction 元素数、维度、理论 tree depth、跨迭代 state |
| Control | branch、loop backedge、barrier/sync 的静态动态次数表达式 |
| Conversion | FP16/FP32/FP8 convert、pack/unpack 的元素数 |
| Live data | live-in/out bytes、峰值 live tensor/scalar 数和估计 lifetime |

工作量必须满足：

- 每个 TTIR operation 只统计一次；
- `N_iter` 与“每次迭代工作量”分开保存，不能提前相乘后丢失结构；
- operation graph 中循环 body 的静态 operation 计数不是动态工作量。实现先使用常量上下界计算
  `trip_count`；无法直接得到时使用结构分析给出的循环次数。body 工作量乘动态次数后，再除以该
  Stage 的 `N_iter` 保存为 per-iteration workload。多个 sibling loop 先展平为统一迭代空间，不能把
  14 次逐行递推错误地只计一次；
- 不把逻辑 bytes 直接等同于 SIMD MTE transaction 或 SIMT DCache transaction；二者由第 4、5
  章的模式专属模型根据同一地址事实分别推导；
- 不使用 Profile Rate，不输出 cycles。

### 2.5 StageModeLegalityAnalysis

`StageModeLegalityAnalysis` 只判断“能否正确编译和执行”，不判断“是否更快”：

| 检查 | SIMD | SIMT |
|---|---|---|
| operation lowering | 所有 operation 有合法 SIMD/Structured lowering | 所有 operation 有合法 SIMT/Unstructured lowering |
| control region | region/terminator 能在 SIMD 路径保持语义 | 能完整放入 `scope<vector_mode="simt">` |
| live-in/out | layout/type 可由 SIMD consumer/producer 使用 | scope 参数与 `scope.return` 可精确物化 |
| memory effect | side effect 顺序不被改变 | scope 内外 side effect 与 alias 顺序安全 |
| Cube | 允许使用 Cube/MMAD 路径 | 只有存在合法 SIMT dot lowering 时才合法 |
| SuperBlock | factor 固定为 F1 | 分别检查 F1/F2/F4 的 warp、写地址、依赖、tail 和缓冲限制 |

输出至少包含：

```text
simd_legal
simt_legal
legal_simt_factors = {1, 2, 4} 的子集
illegal_reasons[mode/factor]
```

这里不检查 Coverage、置信度、gain margin 或 Event validated flag。相邻 Stage 的组合边界是否合法
由 `KernelRouteSolver` 中的 `RouteLegalityChecker` 检查，避免把单 Stage 合法性与路线合法性混在一起。

### 2.6 StageCostModelKind 分类

`StageKindClassifier` 根据 operation 与 `StageModelFeatures` 为每个 Stage 选择唯一 Kind：

```text
Stage.operations + Stage.features -> one StageCostModelKind
```

分类首先保留已有且有结构证据的专用 Kind（例如 recurrence、reduction、dot），然后对不匹配的
候选按真实结构收敛：dot → reduction → conversion/pack → loop（真实数据递推或独立流水）→
indirect memory → continuous memory → scalar。这个回退只修正“模板预期与真实 dtype/loop 形态不同”
的情况；若一个候选同时命中两个必须分别计费的强结构，仍必须返回 `requires_split` 交给
`StageBoundaryAnalysis`，不能挑一个“更像的”Kind覆盖全部工作。

### 2.7 StagePartition 数据结构

```cpp
struct StagePartition {
  SmallVector<LogicalPhase> phases;
  SmallVector<StageEdge> executionEdges;
};

struct LogicalPhase {
  std::string id;
  PhaseKind kind;
  SmallVector<LogicalStage> stages;
};

struct LogicalStage {
  std::string id;
  StageCostModelKind costModelKind;
  StageScheduleKind scheduleKind;
  SmallVector<Operation *> operations;
  int64_t iterationCount;
  StageModelFeatures features;
  StageWorkload workload;
  StageLiveValues liveIn;
  StageLiveValues liveOut;
  bool simdLegal;
  bool simtLegal;
  SmallVector<int64_t> legalSimtFactors;
};
```

`StagePartition` 不包含 `StageResourceCycles`、`StageImplementationCost` 或最终 mode。资源 cycle
属于 `StageCostEvaluator` 的 `StageCostTable`；最终 mode 属于 `KernelRouteSolver` 的
`KernelRoutePlan`。

必须满足以下不变量：

1. 每个参与建模的 TTIR operation 精确属于一个 Stage；
2. Phase/Stage 顺序保持原始控制依赖和数据依赖；
3. 每个 Stage 只有一个 `StageCostModelKind`，但可拥有 SIMD/SIMT 两种候选；
4. Stage 内不存在 mixed mode，Transition 只位于 Stage edge；
5. live-in/out 与 Materializer 使用同一组 SSA value 和 layout；
6. `StagePartitioner` 的输出与 Profile、实测性能和最终 route 无关。

### 2.8 Phase/Stage 边界规则

| 边界来源 | 是否形成 Phase | 是否形成 Stage | 例子 |
|---|---:|---:|---|
| 算法上的串行大步骤 | 是 | 是 | solve_tril 的 load diag、diagonal inverse、merge/store |
| AutoBlockify V1 物理调度 | 是 | 是 | dispatch prologue 与 logical-program loop |
| 控制区域/循环结构改变 | 视算法语义 | 是 | straight-line setup 与 recurrence loop |
| 主成本公式改变 | 否 | 是 | indirect gather 与 Cube dot |
| 真实 loop-carried 数据依赖开始/结束 | 视算法语义 | 是 | independent load 与 row recurrence |
| 仅 pointer induction | 否 | 否 | `ptr += stride` 不单独制造递推 Stage |
| 仅为了获得更低预测分数 | 否 | 否 | 禁止按 CostModel 结果反向切分 |

相邻候选 Stage 只有在同一 Phase、同一 Kind、相同 schedule、无强制控制/依赖边界且合并后仍可
精确物化时才能合并。

### 2.9 每个算子的 Head Phase

每个算子都显式包含一个 `head/setup` Phase，其中可以继续切成 scalar/index/predicate 等 Stage。
典型工作包括：

```text
program_id / block_id
shape 与 stride 标量计算
tile offset
base pointer
循环边界
初始 mask/predicate
```

这些工作不能继续隐藏在整 kernel 的固定 `S_SIMD` 或 `S_SIMT` 常数中。Head Phase 内每个 Stage
仍只有 `SIMD` 或 `SIMT` 两种模式。SIMD Stage 可以包含 main-scalar、MTE、SIMD 与 Cube 内部
流水；这不算 mixed。SIMT Stage 计算 SIMT scalar/warp、DCache、SIMT EXU 和谓词工作量。
AutoBlockify dispatch 等共享 setup 作为独立 Stage 等额参与候选计费，不引入第三种 StageMode。

## 3. StageCostEvaluator

`StageCostEvaluator` 是 Stage 成本计算的主组件。它消费第 2 章生成的 `StagePartition`，为每个
Stage 的所有合法 SIMD/SIMT implementation 计算成本，最终输出 `StageCostTable`。它负责组织
计算流程，不直接实现任何 SIMD/SIMT 成本公式。

### 3.1 组件内流程

```text
StagePartition
  -> ProfileProvider.getSnapshot(target, profileVersion)
       得到本次编译只读的 HardwareProfile
  -> for Phase in execution order
       -> for Stage in Phase
            -> enumerateImplementations(Stage)
                 SIMD-F1（simdLegal 时）
                 SIMT-F1/F2/F4（simtLegal 且 factor 合法时）
            -> for Implementation
                 StageCostModelRegistry.lookup(mode, costModelKind)
                 -> concrete StageCostModel.estimate(...)
                 -> validate StageImplementationCost
                 -> StageCostTable.add(stageId, implementation, cost)
  -> verifyEveryStageHasLegalCost()
  -> StageCostTable
```

流程约束：

1. 严格按 `StagePartition` 的 Phase/Stage 顺序遍历，但各候选成本彼此独立；
2. 只枚举 `StageModeLegalityAnalysis` 已声明合法的 mode/factor；
3. 不重新提取 feature、workload，不修改 Stage 边界；
4. 不比较跨 Stage 路线，不添加 TransitionCost；这些属于 `KernelRouteSolver`；
5. 一个候选只能由 Registry 返回的一个具体 StageCostModel 计费，防止多模型重复相加；
6. 所有成本必须有限、非负，并保留资源分解与 Profile 版本以便复核。

### 3.2 接口与输出数据结构

```cpp
struct StageCostModelContext {
  const LogicalStage &stage;       // kind/features/workload/live-in/out
  const HardwareProfile &profile;  // 本次编译的只读 snapshot
};

struct StageImplementation {
  StageMode mode;                  // SIMD / SIMT
  int64_t superblockFactor;        // SIMD=1；SIMT 可为 F1/F2/F4
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles;
  StageResourceCycles resources;
  std::string modelName;
  std::string profileVersion;
};

class StageCostEvaluator {
public:
  StageCostTable evaluate(const StagePartition &,
                          const TargetInfo &,
                          llvm::StringRef profileVersion) const;
};
```

`StageCostTable` 以 `stageId` 为主键保存候选列表：

```text
stage-0 -> { SIMD-F1: cost, SIMT-F1: cost, SIMT-F2: cost }
stage-1 -> { SIMD-F1: cost }
stage-2 -> { SIMD-F1: cost, SIMT-F1: cost }
```

这里没有 mixed implementation；mixed 只可能在后续 KernelRouteSolver 为不同 Stage 选择不同
StageMode 后形成。

### 3.3 StageCostModelRegistry

`StageCostModelRegistry` 是无状态模型目录，键为 `(StageMode, StageCostModelKind)`，值为一个
具体 `StageCostModel`：

```cpp
class StageCostModelRegistry {
public:
  const StageCostModel &lookup(StageMode,
                               StageCostModelKind) const;
  void registerModel(std::unique_ptr<StageCostModel>);
  LogicalResult verifyComplete() const;
};
```

示例映射：

| Registry key | 具体模型 |
|---|---|
| `(SIMD, scalar_issue)` | `SIMDScalarStageCostModel` |
| `(SIMT, scalar_issue)` | `SIMTScalarStageCostModel` |
| `(SIMD, continuous_tile_memory)` | `SIMDContinuousMemoryStageCostModel` |
| `(SIMT, continuous_tile_memory)` | `SIMTContinuousMemoryStageCostModel` |
| `(SIMD, loop_carried_recurrence)` | `SIMDRecurrenceStageCostModel` |
| `(SIMT, loop_carried_recurrence)` | `SIMTRecurrenceStageCostModel` |

Registry 不读取 Stage、Profile 或 workload，不计算 cost。注册重复 key 必须报错；Evaluator 请求的
合法 `(mode, kind)` 没有模型时必须显式报错，不能静默换用无关公式。

### 3.4 ProfileProvider

`ProfileProvider` 为一次编译提供不可变、版本化的 `HardwareProfile`：

```cpp
class ProfileProvider {
public:
  const HardwareProfile &getSnapshot(const TargetInfo &,
                                     llvm::StringRef version) const;
};
```

`HardwareProfile` 按消费方拆成明确的参数组：

| Profile 分组 | 内容 | 主要消费者 |
|---|---|---|
| `SIMDProfile` | MTE2/MTE3、EXU、ASU、Cube、main-scalar、issue、spill 参数 | SIMD StageCostModel 分支 |
| `SIMTProfile` | LSU/DCache、SIMT EXU、shuffle、branch/reconvergence、occupancy、spill 参数 | SIMT StageCostModel 分支 |
| `SuperBlockProfile` | F1/F2/F4 setup、同步、资源压力和 tail 参数 | SIMT SuperBlock wrapper |
| `TransitionProfile` | SIMD→SIMT、SIMT→SIMD 的 scope、同步、layout/materialization 参数 | `TransitionCostModel` |

ProfileProvider 负责 schema、单位、目标型号和版本校验；不包含算子名、shape 专用总耗时或历史
路线结论。一次编译开始后 snapshot 不得被 Event/CaModel 在线改写；测量结果只能经离线
`ProfilePublisher` 产生新版本。

HIVM DES 保留为独立的离线调度诊断工具，但不属于在线 `StageCostEvaluator`。当前没有可靠的
DES→Stage cycle 校准闭环，因此 Route report 中原先两个始终为空的 DES feedback 占位字段已删除；
下一轮主校准源是相同 TTIR/shape 下的 CaModel stage cycle，DES 只在需要解释底层调度时辅助使用。

### 3.5 StageCostModel 设计

#### 3.5.1 StageCostModel 两级模型树

`StageCostModel` 是具体成本模型的抽象基类，不是 Facade。它先按执行模式派生出 SIMD/SIMT
两大类，再在各自分支内按 Stage 语义派生具体模型。第 4、5 章就是这两棵分支的主体：

> StagePartitioner 决定“算哪一段”，KernelRouteSolver 只负责“把分数相加并选最小路线”；真正决定
> CostModel 成败的是第 4、5 章能否分别准确计算同一 Stage 的 SIMD 与 SIMT 成本。

```text
StageCostModel                            # 抽象基类：计算一个 mode 下的一种 Stage 语义
  |
  +-- SIMDStageCostModel                 # SIMD 抽象基类，第 4 章
  |     +-- SIMDDispatchStageCostModel
  |     +-- SIMDScalarStageCostModel     # 包含 scalar_issue 等 Scalar Kind
  |     +-- SIMDContinuousMemoryStageCostModel
  |     +-- SIMDIndirectMemoryStageCostModel
  |     +-- SIMDIndependentStageCostModel
  |     +-- SIMDRecurrenceStageCostModel
  |     +-- SIMDReductionStageCostModel
  |     +-- SIMDCubeStageCostModel
  |     `-- SIMDConversionPackStageCostModel
  |
  `-- SIMTStageCostModel                 # SIMT 抽象基类，第 5 章
        +-- SIMTDispatchStageCostModel
        +-- SIMTScalarStageCostModel     # 包含 scalar_issue 等 Scalar Kind
        +-- SIMTContinuousMemoryStageCostModel
        +-- SIMTIndirectMemoryStageCostModel
        +-- SIMTIndependentStageCostModel
        +-- SIMTRecurrenceStageCostModel
        +-- SIMTReductionStageCostModel
        +-- SIMTCubeStageCostModel
        `-- SIMTConversionPackStageCostModel
```

这里是“**模式优先、语义次之**”的两级分派：

```text
C_stage(SIMD, kind) = Registry.lookup(SIMD, kind).estimate(...)
C_stage(SIMT-Fk, kind) = Registry.lookup(SIMT, kind).estimate(..., factor=k)
```

两条分支共享 `StagePartition`、`StageModelFeatures`、`StageCostModelKind` 和静态 workload；
不共享硬件资源映射与最终公式。原因是同一个 `continuous_tile_memory` 在 SIMD 中映射到
MTE/UB 流水，在 SIMT 中映射到 LSU/DCache transaction；若仍由一个语义类内部用 `if(mode)`
切换，公式、资源和合法性会逐渐混在一起。

#### 3.5.2 StageCostModelKind 定义与分类

`StageCostModelKind` 描述一个 Stage 的**计算语义**，不描述它最终运行在 SIMD 还是 SIMT。
它由 `StagePartitioner` 根据 post-layout、post-AutoBlockify TTIR 赋给每个 `LogicalStage`，随后与
`StageMode` 一起构成 Registry 查询键：

```text
(StageMode, StageCostModelKind) -> concrete StageCostModel
```

目标版本包含 20 个 Kind，归入 8 个语义族：

| 语义族 | `StageCostModelKind` |
|---|---|
| Dispatch | `auto_blockify_dispatch`、`auto_blockify_loop` |
| Scalar | `scalar_issue`、`scalar_control`、`scalar_math`、`index_generation`、`predicate_mask`、`loop_predicate` |
| Continuous Memory | `continuous_tile_memory`、`continuous_tile_store`、`continuous_short_load`、`cache_policy_store` |
| Indirect Memory | `indirect_scalar_memory`、`indirect_gather_memory` |
| Independent Pipeline | `independent_pipelined_loop` |
| Recurrence / Reduction | `loop_carried_recurrence`、`rowwise_reduction` |
| Cube / Tiny Cube | `cube_roofline`、`tiny_cube_roofline` |
| Conversion / Pack | `conversion_pack` |

每个 Kind 的完整语义边界如下。这里定义的是“什么样的 Stage 使用什么模型”，不是对三个实验
用例的名称匹配：

| 语义族 | `StageCostModelKind` | 精确定义 | 分类所需的主要结构/数据 | Triton 示例（示意） |
|---|---|---|---|---|
| Dispatch | `auto_blockify_dispatch` | AutoBlockify V1 为每个物理 program 创建的一次性 PID、chunk 和边界 setup | V1 provenance、physical/logical program 数、chunk/tail | `pid = tl.program_id(0)`；物理 dispatch 由 V1 后续生成 |
| Dispatch | `auto_blockify_loop` | AutoBlockify V1 创建的 logical-program 聚合循环，不是原算法循环 | V1 生成循环标记、trip count、backedge/control event | 用户提交含大量 logical programs 的 grid；V1 将其包装为物理 program 内循环 |
| Scalar | `scalar_issue` | 不含显著分支、访存或向量计算的普通标量发射块 | scalar op 数、依赖链、issue class | `tile_id = pid * tiles_per_program + local_id` |
| Scalar | `scalar_control` | early return、条件选择或控制转移主导的标量块 | branch 数、taken probability、依赖链 | `if pid >= num_tiles: return` |
| Scalar | `scalar_math` | 除索引生成外的标量算术或特殊函数块 | arithmetic/SFU op 数、数据依赖 | `scale = 1.0 / tl.maximum(absmax, eps)` |
| Scalar | `index_generation` | tile offset、stride、div/rem、地址索引等地址生成工作 | index op 数、pointer induction、依赖链 | `offs = pid * BLOCK + tl.arange(0, BLOCK)` |
| Scalar | `predicate_mask` | 非循环专属的 compare、boundary mask 生成与 mask 应用 | predicate op 数、mask shape、active-lane ratio | `mask = offs < n; x = tl.load(ptr + offs, mask=mask)` |
| Scalar | `loop_predicate` | 随算法循环迭代变化的谓词、退出条件与 backedge 控制 | trip count、compare/branch/backedge 数 | `for i in tl.range(0, n): active = i < limit` |
| Continuous Memory | `continuous_tile_memory` | layout 合并后可证明连续的 tile load | GM/UB 或 GM/register bytes、transaction、alignment、tile count | `x = tl.load(ptr + tl.arange(0, BLOCK), mask=mask)` |
| Continuous Memory | `continuous_tile_store` | layout 合并后可证明连续的 tile store | store bytes、transaction、alignment、tile count | `tl.store(out + tl.arange(0, BLOCK), x, mask=mask)` |
| Continuous Memory | `continuous_short_load` | 连续但数据量较小、启动延迟不可忽略的 load | bytes、transaction、固定启动延迟 | `idx = tl.load(index_ptr + tl.arange(0, 16))` |
| Continuous Memory | `cache_policy_store` | 带 `.cg` 等明确 cache policy、不能等同普通连续 store 的写路径 | cache policy、bytes、transaction、write path | `tl.store(out + offs, x, cache_modifier='.cg')` |
| Indirect Memory | `indirect_scalar_memory` | 地址依赖运行时数据的标量 load/store | 地址依赖、transaction 数、未隐藏延迟 | `idx = tl.load(index_ptr + pid); x = tl.load(src + idx)` |
| Indirect Memory | `indirect_gather_memory` | 多 lane 地址离散的 gather/scatter，layout 合并后仍不连续 | 地址分布、有效 lane、DCache/GM transaction、驻留 Warp | `idx = tl.load(index_ptr + offs); x = tl.load(src + idx)` |
| Independent Pipeline | `independent_pipelined_loop` | 不存在真实 loop-carried 数据依赖，并且变换后调度结构证明多流水能够重叠的循环 | trip count、依赖图、流水资源占用、调度重叠证据 | `for k in tl.range(0, D, BLOCK): x = tl.load(row + k + offs); tl.store(out + k + offs, f(x))` |
| Recurrence / Reduction | `loop_carried_recurrence` | 第 `i` 轮消费第 `i-1` 轮产生的数据状态；pointer induction 不属于此类 | recurrence critical path、state size/lifetime、spill、trip count | `state = init; for i in tl.range(0, n): state = f(state, x_i)` |
| Recurrence / Reduction | `rowwise_reduction` | 沿指定维度进行 tree/serial/shuffle reduction，可另带跨 tile 累积状态 | reduction width/depth、shuffle/sync、跨 tile state | `row_sum = tl.sum(x, axis=0)` 或 `row_max = tl.max(x, axis=0)` |
| Cube / Tiny Cube | `cube_roofline` | 能映射 Cube 的规则 `tt.dot`，有效工作量足以覆盖主要 setup | M/N/K、dtype、load/MMAD/store、Cube utilization | `acc += tl.dot(a_block, b_block)`，例如 128×128×64 tile |
| Cube / Tiny Cube | `tiny_cube_roofline` | 仍可映射 Cube，但小 shape/不完整 tile 使 setup 与 underfill 不可忽略 | M/N/K、有效 MAC、固定 setup、underfill ratio | `acc = tl.dot(a_16x16, b_16x16)` |
| Conversion / Pack | `conversion_pack` | dtype convert、quantize、pack/unpack 主导的块 | 源/目标 dtype、元素数、pack ratio、寄存器与 spill 压力 | `q = (x * scale).to(tl.float8e4nv)` |

分类先保留有证据的专用 Kind；模板 Kind 与真实 operation graph 不一致时，按
`dot → reduction → conversion/pack → loop → indirect memory → continuous memory → scalar`
收敛。loop 再通过 iter_arg 的 def-use 区分真实数据递推与 pointer-only induction。若同一候选同时包含
两个需要不同主公式的强结构，`StageKindClassifier` 必须返回 `requires_split`，由
`StageBoundaryAnalysis` 拆成多个 Stage，不能选择一个近似 Kind 覆盖全部工作。

Kind 不包含 `SIMD`、`SIMT`、SuperBlock factor、算子名或 shape，也不存在 `mixed_boundary` Kind。
模式属于 `StageMode`，F1/F2/F4 属于 `StageImplementation`；mixed 只属于 KernelRoute。

Kind 与具体 C++ 类不是一一对应关系。一个具体模型可以支持同一语义族内公式相同的多个 Kind，
因此目标是两条模式分支各自拥有约 8 组语义模型，而不是机械创建 20 × 2 个空壳类。

#### 3.5.3 StageCostModel 接口与 StageCostModelRegistry 绑定

目标 C++ 接口为：

```cpp
class StageCostModel {
public:
  virtual StageMode getMode() const = 0;
  virtual bool supports(StageCostModelKind) const = 0;
  virtual StageImplementationCost
  estimate(const StageCostModelContext &,
           const StageImplementation &) const = 0;
};

class SIMDStageCostModel : public StageCostModel {
  StageMode getMode() const final { return StageMode::SIMD; }
};

class SIMTStageCostModel : public StageCostModel {
  StageMode getMode() const final { return StageMode::SIMT; }
};

class SIMDScalarStageCostModel final : public SIMDStageCostModel {
  // 只计算 Scalar Kind 的 SIMD 成本
};

class SIMTScalarStageCostModel final : public SIMTStageCostModel {
  // 只计算 Scalar Kind 的 SIMT 成本
};
```

因此不存在一个同时计算两种模式的 `ScalarIssueModel`。`scalar_issue` 是
`StageCostModelKind`；它分别由 `SIMDScalarStageCostModel` 和
`SIMTScalarStageCostModel` 计算。具体类只实现一个硬件模式的资源映射和成本公式，不能在
`estimate()` 内部通过 `if (mode)` 同时服务两条硬件路径。

#### 3.5.4 StageCostModel 公共成本外壳

对外统一入口是 `StageCostEvaluator`。它使用 `(StageMode, StageCostModelKind)` 从 Registry
获得具体模型，再计算该 Stage 的一个 implementation。

对一个 Stage，只分别计算合法的 `SIMD` 与 `SIMT-F1/F2/F4` implementation；不存在
`mixed Stage implementation`。mixed 由 KernelRouteSolver 组合不同模式的相邻 Stage 后产生。

SIMD 和 SIMT 真正共享的只有迭代记账外壳，不共享 `C_body`：

```text
C_stage(mode) = C_setup(mode)
              + N_iter × C_body(mode, StageCostModelKind, Features)
              + C_epilogue(mode)

C_body(SIMD)    = F_SIMD(StageCostModelKind, Features, R_SIMD)     # 第 4 章
C_body(SIMT-Fk) = F_SIMT(StageCostModelKind, Features, R_SIMT, k) # 第 5 章
```

#### 3.5.5 StageControlFlowRates 与串行保守回退

控制流事件的计数结构也可以共用，但硬件 Rate 必须按模式分别提供：

```text
C_control(mode) = C_unclassified(mode)
                + N_backedge × R_backedge(mode)
                + N_branch   × R_branch(mode)
                + N_sync     × R_sync(mode)

C_divergence(SIMT) = N_divergent
                   × (1 - active_lane_ratio)
                   × R_reconvergence(SIMT)
```

SIMD mask 通常进入 `C_predicate(SIMD)`，不能机械套用 SIMT 的 divergence/reconvergence。
下面的串行和只是两条分支在“无法证明重叠”时可使用的保守 fallback，不是二者共享的最终
成本公式；其中每个资源项也必须使用对应模式的 workload 映射和 Rate：

```text
C_serial(mode) = C_scalar(mode) + C_load(mode) + C_store(mode)
               + C_compute(mode) + C_predicate(mode) + C_shuffle(mode)
               + C_dot(mode) + C_control(mode) + C_spill(mode)
               + C_issue(mode)
```

Stage 内不计 `C_transition`。只有相邻 Stage 选择不同模式时，KernelRouteSolver 才在 Stage 边上
计一次 `TransitionCost`，不能乘 `N_iter`。

`C_predicate` 表示 compare、mask 生成和 mask 应用成本；masked-off lane 的效率损失应体现在
有效吞吐或单独资源项中，不能重复计费。`C_shuffle` 表示数据重排：SIMD 路径包括向量
permute、transpose、reduce/layout 重排，SIMT 路径包括 warp/lane shuffle；没有重排时为 0。

#### 3.5.6 StageCostModel 语义族与 StageMode 映射

| Stage 语义族 | `StageCostModelKind` | 匹配结构 | `SIMDStageCostModel` 分支 | `SIMTStageCostModel` 分支 |
|---|---|---|---|---|
| Dispatch | `auto_blockify_dispatch`、`auto_blockify_loop` | V1 的 PID/chunk setup 与物理核聚合循环 | `C_scalar + C_control + C_issue`；loop kind 乘迭代数 | 相同结构，使用对应物理实现的 Rate |
| Scalar | `scalar_issue`、`scalar_control`、`scalar_math`、`index_generation`、`predicate_mask`、`loop_predicate` | offset、索引、mask、标量控制 | `C_scalar+C_compute+C_predicate+C_control+C_issue+C_spill` | 相同结构，使用 SIMT scalar/predicate Rate |
| Continuous Memory | `continuous_tile_memory`、`continuous_tile_store`、`continuous_short_load`、`cache_policy_store` | layout 合并后的连续 load/store | 有重叠证据时：`C_scalar+C_predicate+C_control+C_spill+max(C_load,C_store,C_issue)` | transaction/latency 模型；没有 SIMT 调度重叠证据时使用 `C_serial(SIMT)` |
| Indirect Memory | `indirect_scalar_memory`、`indirect_gather_memory` | gather/scatter、数据相关地址 | `C_serial` | `C_serial`；SIMT resource 应体现 DCache transaction、驻留 Warp 和未隐藏延迟 |
| Independent Pipeline | `independent_pipelined_loop` | 无真实 loop-carried 数据依赖，且调度结果证明不同流水可以重叠 | `max(C_load,C_store,C_compute+C_dot+C_shuffle,C_scalar+C_predicate+C_control,C_issue)+C_spill` | occupancy/scheduler 证明可隐藏时使用 SIMT overlap；否则 `C_serial(SIMT)` |
| Recurrence / Reduction | `loop_carried_recurrence`、`rowwise_reduction` | 递推读取上一轮状态；reduction 具有独立的 tree/shuffle 结构 | 递推用 `C_critical_path`；reduction 用层数与并行宽度 | 递推用 SIMT 关键路径；reduction 用 shuffle depth、同步和 Rate |
| Cube / Tiny Cube | `cube_roofline`、`tiny_cube_roofline` | `tt.dot`、规则或 underfilled 小矩阵计算 | `C_scalar+C_predicate+C_control+C_shuffle+C_spill+max(C_load,C_compute+C_dot,C_store,C_issue)` | 没有 Cube fast path 时使用 `C_serial` |
| Conversion / Pack | `conversion_pack` | FP16/FP32/FP8 convert、pack/unpack | `C_predicate+C_control+C_spill+max(C_scalar+C_compute,C_load,C_store,C_issue)` | `C_serial` |
| **目标合计** | **20 个 Kind** | **8 个语义族 × 两个模式成本分支** |  |  |

#### 3.5.7 StagePartitioner 切分与 StageScheduleKind 约束

复合语义必须在 `StagePartitioner` 中拆开。例如“离散基址 + 连续 tile 访问”拆成
Indirect Memory Stage 与 Continuous Memory Stage；模式切换成本统一由 KernelRouteSolver 的 Stage
边计算，不创建 boundary Stage。

“没有真实 loop-carried 数据依赖”只说明跨迭代重叠是合法的，并不自动证明存在流水。
流水来自目标多流水资源和编译器调度；循环展开只是暴露更多 ILP 的方法之一。
`schedule == IndependentPipelined` 必须由 post-layout、post-AutoBlockify TTIR 的依赖和调度
结构直接产生，不能由 workload 名称或经验规则指定。

### 3.6 三个用例中的 Phase / Stage / StageCostModel 映射

#### solve_tril

| Phase | 当前实际 Stage | `StageCostModelKind` | ownership / 成本边界 | 主要候选 |
|---|---|---|---|---|
| P0 Physical Dispatch | `physical_program_dispatch` | `auto_blockify_dispatch` | V1 的 PID/chunk/setup roots | SIMD / pure-SIMT F1/F2/F4 |
| P0 Physical Dispatch | `logical_program_loop` | `auto_blockify_loop` | V1 schedule shell，与算法递推分开 | SIMD / pure-SIMT F1/F2/F4 |
| P1 Head / Mask | `head_index_mask` | `predicate_mask` | tile offset、pointer setup 和静态 triangular mask 合并计入同一主谓词 Stage | SIMD / SIMT |
| P2 Diagonal Load | `load_diagonal_tiles` | `continuous_tile_memory` | scope 外连续 block-pointer load；compound-scope 逻辑重排后仍独立 | SIMD MTE / SIMT LSU |
| P3 Diagonal Inverse | `diagonal_inverse_recurrence` | `loop_carried_recurrence` | 迭代 predicate、短 reduction、逐行更新和长寿命 state 作为一条递推关键路径；BT16 为 14 次动态迭代 | SIMD / local SIMT-F1；pure-SIMT F1/F2/F4 |
| P4 Merge / Store | `dense_dot_tail`（存在 dot 时） | `cube_roofline` | dense dot 保留 Cube，不能并入 recurrence scope | SIMD/Cube / SIMT dot |
| P4 Merge / Store | `store_inverse_tile` | `continuous_tile_store` | 最终连续 tile store；BT16 无 dense tail 时直接承接 recurrence | SIMD / SIMT |

#### FBGEMM rowwise quant

| Phase | 当前实际 Stage | `StageCostModelKind` | ownership / 成本边界 | 主要候选 |
|---|---|---|---|---|
| P1 Row Dispatch | `row_index_generation` | `index_generation` | valid guard、token/expert/score 索引和进入 gather 前的 setup | SIMD / SIMT |
| P2 Row Load | `indirect_row_gather` | `indirect_gather_memory` | loaded index 决定行地址；精确拥有 materializable gather anchors | SIMD / local SIMT-F1 |
| P3 Row Reduction | `rowwise_reduction` | `rowwise_reduction` | row tile load、tree reduction 与跨 tile max state 按 reduction 关键路径计费 | SIMD tree / SIMT shuffle |
| P4 Convert / Store | `conversion_pack_store` | 由真实结构分类：FP8 为 `conversion_pack`；无 convert 且只有 pointer induction loop 时为 `independent_pipelined_loop` | 同一源码在不同 dtype/`BLOCK_D` 下不能硬编码 Kind；pointer iter_arg 不算真实数据递推 | SIMD roofline / SIMT serial |

带 `.cg` cache policy 的 FBGEMM 当前会触发 AutoBlockify V1 的正确性黑名单，因此该 shape 没有伪造
P0 dispatch Stage；V1 真正启用时才由 operation provenance 创建对应 Stage。

#### gather-dot-min

| Phase | 当前实际 Stage | `StageCostModelKind` | ownership / 成本边界 | 主要候选 |
|---|---|---|---|---|
| P0 Physical Dispatch | `physical_program_dispatch` | `auto_blockify_dispatch` | V1 每物理核一次 setup | SIMD / pure-SIMT F1/F2/F4 |
| P0 Physical Dispatch | `logical_program_loop` | `auto_blockify_loop` | V1 logical-program schedule shell | SIMD / pure-SIMT F1/F2/F4 |
| P1 Index Setup | `index_generation` | `index_generation` | offsets、短 index load、mask 与地址 setup 的主语义为索引生成 | SIMD / SIMT |
| P2 Gather | `indirect_tile_gather` | `indirect_gather_memory` | A/B loaded-index gather；拥有两个精确 anchors | SIMD / local SIMT-F1 |
| P3 Dot | `tiny_cube_dot` | `tiny_cube_roofline` | 16×16×16 dot，单独计算 Cube setup/underfill | SIMD/Cube / SIMT dot |
| P4 Store | `store_dot_result` | `continuous_tile_store` | 规则输出 store | SIMD / SIMT |

`gather` 与 `dot` 是两个 Stage；若前者选 SIMT、后者选 SIMD/Cube，KernelRouteSolver 在二者之间
加入一次 directional transition，Materializer 据此形成 SIMT scope。边界本身不是 Stage。

### 3.7 StageCostEvaluator 目标实现清单

| 组件 | 必须实现的内容 | 当前实现状态 | 下一步校准 |
|---|---|---|---|
| `StageCostEvaluator` | 遍历 `StagePartition`，枚举每个 Stage 的合法 SIMD/SIMT-F1/F2/F4 implementation | 已实现；每个合法候选恰好由一个模型计算，28 个 Stage/Route 单元测试通过 | 对照 CaModel 检查每个 resource cycle |
| `StageCostModelRegistry` | 按 `(StageMode, StageCostModelKind)` 返回具体成本模型 | 已实现并验证 20 Kind × 2 Mode 覆盖唯一且完整 | 新 Kind 加入时保持 coverage test |
| `ProfileProvider` | 按 target/version 返回本次编译只读的 `HardwareProfile` | 已实现 target/version/正值校验；编译期间 snapshot 不变 | Profile schema 继续补充单位与来源审计 |
| SIMD 模型树 | MTE、compute/dot、predicate、shuffle、scalar、issue、spill、control 成本 | 资源映射和各语义族公式已实现 | 用 CaModel 分阶段 cycle 校正 overlap 与 Rate；ASU/main-scalar 仍需更细分离 |
| SIMT 模型树 | LSU/DCache、SIMT compute、shuffle、控制流、issue、spill 成本 | 资源映射和各语义族公式已实现 | 用 CaModel 校正 DCache transaction、occupancy 与 latency hiding |
| Recurrence / Reduction | 分开递推关键路径、tree reduction、shuffle depth 和跨 tile state | 已拆成 `SIMD/SIMTRecurrenceStageCostModel` 与 `SIMD/SIMTReductionStageCostModel` | 校正真实 critical path 与 spill transaction |
| SuperBlock | 建立 F1/F2/F4 的并发收益、同步、批处理和资源压力公式 | pure-SIMT route 已保证全 Stage 使用统一 factor；F2 后收益饱和且递推 live-out 产生压力；local mixed scope 当前只合法 F1 | 结合 ScopeSuperBlock 实现后开放 mixed F2/F4，并用 CaModel 校正收益 |
| Scope handoff | 统计 Mixed scope 的静态 tensor live-out，区分寄存器预算与 stack/register 交接 | 已实现；每个物化 scope 单独计费，大 live-out 不再被当成零成本 | 用 scope 边界 microbenchmark/CaModel 替换暂定带宽 |
| `StageImplementationCost` | 输出总 cycle、资源分解、模型名和 profile 版本 | 已实现有限/非负校验与 JSON 报告 | 增加 measured-vs-estimated 误差报告，不在线改分 |

规划代码位置：

- Evaluator、接口、两级模型树和 Registry：`third_party/ascend/costmodel/include/AscendModel/RouteModel/StageCostModels.h`
- SIMD/SIMT 具体公式：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`
- Profile 映射与只读校验：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`

## 4. SIMD Stage Cost

本章定义 `SIMDStageCostModel`，是 StageCostModel 的第一个核心分支。它不能直接接收一个
笼统的 `compute/load/store` 总量，而要把公共 workload 映射到 950PR SIMD 微架构资源，
再根据 Stage 语义和依赖关系选择公式：

```text
Stage + Features + Workload
  -> SIMD legality
  -> SIMD resource mapping
       MTE2 / MTE3 / EXU0,1 / ASU / Cube / main-scalar / issue / spill
  -> F_SIMD(StageCostModelKind, schedule, dependency)
  -> StageImplementationCost(mode=SIMD)
```

统一外壳为：

```text
C_stage_SIMD = C_setup
             + N_iter × F_SIMD(kind, features, R_SIMD)
             + C_epilogue
```

`F_SIMD` 的选择规则如下。第 4.2～4.4 节展开其中最关键的 Roofline 和依赖模型：

| Stage 语义族 | SIMD 子模型 | 核心成本关系 |
|---|---|---|
| Dispatch / Scalar / Predicate | SIMD scalar/control | main-scalar、predicate、issue 和控制流串行关键路径 |
| Continuous Memory | SIMD MTE pipeline | 有重叠证据时对 MTE2/MTE3/计算/issue 取 `max`，否则串行 |
| Indirect Memory | SIMD indirect memory | 地址生成、离散 transaction 和未隐藏延迟串行；不套连续 MTE Roofline |
| Independent Pipeline | SIMD extended roofline | 无真实 loop-carried 数据依赖且调度证明可重叠时取多流水 `max` |
| Recurrence / Reduction | SIMD critical path | 递推链延迟乘迭代数，独立旁路才能与 memory 重叠 |
| Cube / Tiny Cube | SIMD Cube roofline | Cube setup、load、MMAD、store、underfill 与 issue 下界 |
| Conversion / Pack | SIMD EXU/ASU pipeline | convert/pack 与访存仅在依赖允许时重叠 |

### 4.1 SIMD workload 到资源成本的映射

| 公共 workload/feature | `SIMDStageCostModel` 中的解释 |
|---|---|
| 连续 GM load/store bytes | 按 MTE2/MTE3 带宽、启动延迟和 tile 次数换算 |
| scalar/index ops | 按 main-scalar/ASU 指令 Rate 与依赖链换算 |
| vector compute ops | 按 EXU0/EXU1 吞吐和双发射条件换算 |
| dot shape/count | 按 Cube setup、有效 MAC 吞吐和小 shape underfill 换算 |
| predicate/mask | 按 compare/select 指令与无效 lane 比例换算；不能再重复计入 EXU |
| shuffle/layout work | 归属实际 EXU/ASU/模板流水，不能作为独立资源重复相加 |
| branch/backedge/sync | 使用 SIMD control Rate 计算 `C_control` |
| live value/register pressure | 使用 spill 指令数、stack transaction 和相关 stall 计算 `C_spill` |

### 4.2 可重叠的 SIMD Stage：扩展 Roofline

使用 Roofline 必须同时满足：没有真实 SSA/data memory loop-carried dependency，并且
变换后 IR/调度已经证明相关流水可以重叠。没有真实依赖只是合法性前提，不等于自动存在流水；
循环展开只是暴露 ILP 的一种方式。

```text
C_SIMD = C_setup
       + N_iter × (max(C_MTE2_load,
                       C_MTE3_store,
                       C_EXU,
                       C_ASU,
                       C_CUBE,
                       C_main_scalar,
                       C_issue)
                   + C_nonoverlapped_predicate
                   + C_control
                   + C_spill)
       + C_epilogue
```

`C_shuffle` 必须按最终 HIVM 指令落入实际执行流水（EXU、ASU 或专用模板成本），不能既作为
独立资源又重复计入 EXU/ASU。`StageResourceCycles` 必须按上述 950PR 微架构资源拆分，不能只
保留笼统的 `compute/shuffle/dot/scalar` 聚合项。

950PR 的 issue 下界为：

```text
C_issue = max(N_LD / 2,
              N_ST / 1,
              N_EXU / 2,
              N_ASU / 1,
              N_scalar / R_scalar)
```

这要求同一 VF 内相关指令没有依赖；“硬件有两个 EXU”不等于任意两条计算指令必然双发射。

判断边界必须明确：

```text
state_i = F(state_{i-1}, x_i)     -> 真数据依赖，禁止 roofline
ptr_i   = ptr_{i-1} + stride      -> pointer induction，不禁止 roofline
```

后者只增加地址生成成本；后续 pass 可把它改写成 `base + i × stride`，因此不能因为 pointer
SSA 形式看起来跨迭代就把独立循环错误串行化。所有 SIMD 具体 StageCostModel 必须共用同一个
`permitsSimdRoofline()` 判定，不允许各模型自行放宽。

### 4.3 循环间存在递推依赖：关键路径

```text
state[i] = F(state[i-1], input[i])
```

不能使用吞吐 Roofline 隐藏相邻迭代延迟：

```text
C_SIMD = C_setup
       + N_iter × (C_recurrence_critical_path
                 + C_non_overlapped_memory
                 + C_control
                 + C_spill)
       + C_epilogue
```

### 4.4 部分依赖

将阶段拆成串行 recurrence chain 与可并行旁路工作：

```text
C_SIMD = C_serial_chain + max(C_independent_compute, C_memory)
```

如果拆分后存在稳定且可物化的边界，应进一步拆为两个 Stage，而不是长期依赖该近似式。

## 5. SIMT Stage Cost

本章定义 `SIMTStageCostModel`，是 StageCostModel 的第二个核心分支。它复用同一 Stage 的
语义、依赖和 workload，但重新映射到 SIMT scalar/EXU、LSU/DCache、shuffle、控制流、
驻留 Warp 和 spill 资源：

```text
Stage + Features + Workload
  -> SIMT legality
  -> SIMT resource mapping
       scalar / EXU / LSU+DCache / shuffle / branch+reconvergence
       occupancy / issue / spill
  -> F_SIMT(StageCostModelKind, schedule, dependency)
  -> SuperBlock wrapper(F1/F2/F4)
  -> StageImplementationCost(mode=SIMT, factor=k)
```

统一外壳为：

```text
C_stage_SIMT_F1 = C_setup
                + N_iter × F_SIMT(kind, features, R_SIMT)
                + C_epilogue
```

`F_SIMT` 不能统一写成一个大串行和，也不能照搬 SIMD Roofline：

| Stage 语义族 | SIMT 子模型 | 核心成本关系 |
|---|---|---|
| Dispatch / Scalar / Predicate | SIMT scalar/control | thread/warp 指令数、predicate、branch 和 issue |
| Continuous Memory | SIMT coalesced memory | DCache transaction 数、有效 sector、吞吐和未隐藏 GM 延迟 |
| Indirect Memory | SIMT indirect memory | 离散 transaction、地址生成、cache 命中和驻留 Warp 隐藏能力 |
| Independent Loop | SIMT latency-hiding | 默认按依赖顺序；只有 occupancy/scheduler 模型证明可隐藏时才重叠 |
| Recurrence / Reduction | SIMT critical path | loop-carried 链、shuffle/reduction depth、同步和 spill |
| Cube / Tiny Cube | SIMT dot | 未走 Cube fast path时的 SIMT 指令吞吐、累加链和小 shape 利用率 |
| Conversion / Pack | SIMT conversion | scalar/vector convert、pack/unpack、寄存器压力和串行访存 |

F1 的保守基础形式为：

```text
F_SIMT_serial = C_scalar + C_compute + C_predicate + C_shuffle
              + C_load + C_store + C_control + C_spill + C_issue
```

只有从 IR、驻留 Warp 和目标调度证明可以重叠的项才能改为 `max()`；DCache transaction
吞吐和未隐藏 latency 也必须分开，不能把总 GM latency 对每条 load 重复相加。

### 5.1 SIMT workload 到资源成本的映射

| 公共 workload/feature | `SIMTStageCostModel` 中的解释 |
|---|---|
| load/store 地址集合 | 计算合并后的 DCache transaction/sector 数，而不是只看逻辑 bytes |
| 连续/离散事实 | 决定 transaction 利用率、cache 命中假设和可隐藏 latency |
| scalar/vector ops | 换算为每 thread、每 warp 的 SIMT EXU/issue 工作量 |
| predicate + active lane ratio | 计算谓词指令、无效 lane 和 branch divergence |
| reduction/shuffle | 按 shuffle 层数、每层 Rate、同步与递推深度换算 |
| register/stack pressure | 影响 occupancy，同时产生 STK/LDK 和额外 memory stall |
| `num_warps` / factor | 决定驻留 Warp、延迟隐藏上限和 SuperBlock 资源压力 |

### 5.2 控制流成本

控制流不再只使用一个 `has_control_flow` 布尔惩罚。每次 Stage 迭代计算：

```text
C_control = C_unclassified
          + N_backedge × R_backedge
          + N_branch × R_branch_issue
          + N_divergent × (1 - active_lane_ratio) × R_reconvergence
          + N_sync × R_sync
```

- `N_*` 来自变换后 TTIR 的 `StageModelFeatures`；
- `R_*` 来自目标 profile/CaModel microbenchmark；
- 普通 SIMD mask 主要进入 predicate 成本；SIMT 中 lane 走不同路径时才增加 divergence；
- `control` 字段暂时保留为尚未分类的兼容残量，不能与四个显式项重复计费；
- 有真实递推依赖时，`C_control` 加在每次 recurrence critical path 外侧，不能被 memory/compute
  roofline 隐藏。

### 5.3 Scope SuperBlock：SIMT Stage 的执行变体

局部 `scope.scope<vector_mode="simt">` 不能只通过增加一个属性就自然获得 SuperBlock。
现有 SuperBlock 的语义是：一个物理 program 承载 `factor` 个完整 logical program；因此它是
Kernel 级调度变换，而 CostModel 将 `factor=1/2/4` 视为每个 SIMT Stage 的不同实现候选。

统一映射公式为：

```text
base_pid    = physical_pid * superblock_factor
logical_pid = base_pid + task_id
task_id     = warp_id % superblock_factor
local_warp  = warp_id / superblock_factor
```

例如 `superblock_factor=2`：

| `physical_pid` | `base_pid` | 负责的 `logical_pid` |
|---:|---:|---|
| 0 | 0 | 0、1 |
| 1 | 2 | 2、3 |
| 2 | 4 | 4、5 |
| 3 | 6 | 6、7 |

对于 `SIMD Stage -> SIMT Stage -> SIMD Stage` 的 mixed kernel，目标调度是：

```text
base_pid = physical_pid * factor

SIMD producer Stage:
  为 base_pid ... base_pid+factor-1 准备 factor 份输入
  -> input[task_id]

SIMT Stage:
  由 task_id 选择 logical_pid 和对应输入
  factor 个 warp group 并行执行
  -> output[task_id]

SIMD consumer Stage:
  消费 factor 份 output，完成后续 SIMD/Cube 工作
```

这需要新的 `ScopeSuperBlockPass`，位置必须是：

```text
Route CostModel
  -> Materialize SIMT scope
  -> ScopeSuperBlockPass
  -> TritonToStructured / TritonToUnstructured
  -> lowering
```

Pass 必须同时物化 factor 维度的 scope live-in/live-out、Stage 边界同步、tail predicate 和
launch-grid 缩减，不能只改 SIMT VF 属性。必要合法性条件为：

```text
total_warps = num_warps * factor <= 64
logical_pid < logical_grid_size
不同 logical program 的写地址不重叠
无跨 logical-program 数据依赖
factor 倍的中间缓冲/shared-memory 需求可放置
Stage 边界 barrier 开销可接受
```

SuperBlock 成本不是简单除以 factor：

```text
C_latency_hiding(k) = C_latency_sensitive / min(k, useful_factor_limit)

C_SIMT_Fk = C_batch_setup(k)
           + C_latency_hiding(k)
           + C_nonoverlappable_work
           + C_sync(k)
           + C_resource_pressure(k)
           + C_tail(k)

C_resource_pressure(k)
  += max(0, k - useful_factor_limit)
     * loop_carried_live_out_bytes
     / persistent_state_bytes_per_cycle
```

收益来自更多独立 warp group 隐藏 latency；代价来自输入/输出批处理、同步、寄存器/stack/
中间缓冲压力和尾块浪费。`useful_factor_limit` 是 Profile 参数：当前 950 同 shape 实测表明
F2 已达到有效延迟隐藏上限，因此 F4 不再继续按 `1/4` 缩短 latency-sensitive 部分，并对递推
live-out 状态增加资源压力。这个规则作用于所有 StageModel，不绑定 solve_tril 名称。

### 5.4 `SIMT F1/F2/F4 / SIMD` 的准确含义

它表示**同一个 Stage 的不同候选实现**，不是四个串行 Stage：

| 标记 | 物理含义 | 逻辑 Warp 数 | 当前证据 |
|---|---|---:|---|
| `SIMD` | 普通 SIMD/VF 实现 | 不适用 | 已有 SIMD CaModel 数据 |
| `SIMT-F1` | 一个物理 program 承载一个 logical program | `num_warps` | 普通 SIMT 基线候选 |
| `SIMT-F2` | 一个物理 program 承载两个 logical program | `2 × num_warps` | solve_tril 已有同 shape CaModel 数据 |
| `SIMT-F4` | 一个物理 program 承载四个 logical program | `4 × num_warps` | 已有 solve_tril 同 shape profiler；慢于 F2 |

RouteSolver 将 F1/F2/F4 作为同一 SIMT Stage 的不同实现，并要求 pure-SIMT kernel 内所有
Stage 使用统一 factor。factor 是否出现由 AutoBlockify V1 的真实门禁决定；例如带不支持 cache
modifier 的 kernel 只产生 F1，不能伪造 F2/F4。实测只用于检查和修正公式误差，不作为路线门禁。

## 6. Transition Cost 与 scope live-out

SIMD/SIMT 边界不是固定常数。每条边计算：

```text
C_transition = C_scope_setup
             + C_sync
             + live_in_bytes  / boundary_input_bw
             + live_out_bytes / boundary_output_bw
             + C_layout_conversion
             + C_register_or_UB_materialization
```

当前已落地的可观测项是 scope live-out handoff：

```text
register_budget = local_scope_count * scope_result_register_budget_bytes
spill_bytes     = max(0, live_out_bytes - register_budget)
C_scope_result  = spill_bytes / scope_result_spill_bytes_per_cycle
```

小 scope 结果按寄存器驻留处理；超过预算的静态 tensor live-out 才支付串行 register/stack
交接成本。它修正了 FBGEMM 两个大 scope 被严重低估的问题。尚无同口径测量的 directional
固定启动值保持为 0，不再误用 standalone empty-SIMT VF startup。

方向必须区分：

```text
SIMD -> SIMT
SIMT -> SIMD
```

在真实 directional microbenchmark 完成前，可以使用保守 fallback，但报告必须明确标记
`source=fallback`，不能伪装成实测值；该来源标签只用于误差追踪，不是 route 门禁。

## 7. Mixed Kernel 与 scope 物化

每个 Stage 只选择 `SIMD` 或 `SIMT`。mixed 是 **kernel route 的分类结果**，不是第三种
StageMode：

```text
SIMD Stage -> SIMD Stage             = SIMD-only kernel
SIMT Stage -> SIMT Stage             = SIMT-only kernel
SIMD Stage -> SIMT Stage -> SIMD Stage = mixed kernel
```

Materializer 将 route 中连续的 SIMT Stage 合并为可物化的
`scope.scope<vector_mode="simt">`。相邻 Stage 同模式时不增加 transition；模式发生变化时，
只在边上增加一次第 6 章的 directional `TransitionCost`。如果边界无法精确物化，则该 route
不合法，不能通过引入 Stage 内 Mixed 绕过。

SIMD Stage 内部使用 main-scalar、MTE、SIMD EXU、ASU 或 Cube 都仍属于 SIMD；
`SIMT-F1/F2/F4` 是 SIMT Stage 的 SuperBlock 实现变体，详见 5.3 和 5.4。

## 8. Kernel Route Solver

串行 Stage 使用动态规划：

```text
dp[i][impl] = InternalCost(i, impl)
            + min_previous(
                dp[i-1][previous]
              + TransitionCost(previous.mode,
                               impl.mode))
```

Phase 只提供算法层级和报告边界；求解时可以将所有 Phase 中的 Stage 按执行顺序展平，但不得
跨越依赖边界重排。

约束包括：

- Stage route lowering 必须合法；
- scope 必须能在 TTIR 上精确物化；
- live-in/live-out type 与 layout 必须可跨边界；
- Cube tail 不得误放入 SIMT scope。

上述是保证 IR 可物化、可 lowering 和语义正确的必要条件，不是模型策略门禁。
当前阶段不再使用下列机制删除 CostModel 选中的合法候选：

```text
Coverage short-circuit
Calibration domain
ranking confidence
gain margin
Event validated flag
```

Event/CaModel 只用于校准 StageCost 参数和验证误差，不再把已选中的合法 route
改回 `backend_default`。

最终 kernel route 的分类规则是：

```text
所有 Stage 都选择 SIMD -> kernel SIMD-only
所有 Stage 都选择 SIMT -> kernel SIMT-only
其他任意组合           -> kernel mixed
```

## 9. 三个用例的最新实验结果（2026-08-21）

### 9.1 计量口径

- 设备：Ascend950PR_9579；通过 `/data/kaixin/set_env.sh` 使用容器内逻辑卡 0。
- 性能：充分 warmup 后读取 `torch_npu.profiler` 的设备 Kernel Duration；以下均为本轮重新编译、重新采集的数据。
- 正确性：每条进入表格的可执行路线均完成输出检查。FBGEMM 的 FP8 路线存在 6/524288 个编码值相差一个量化步的规约顺序漂移，scale tensor 通过容差检查。
- CostModel score：越小越好，是 per-program 的排序代理，不等于 profiler 时间，也不是 CaModel 的绝对硬件 cycle。
- `pure-SIMT F2/F4` 使用已经成熟的 NPUIR AutoBlockify V1；局部 Mixed scope 目前只能使用 F1。

### 9.2 solve_tril

固定 shape：`B1/T1024/H32/BT64`，512 个 logical program。

| 路线 | CostModel score | SuperBlock | 自动选择状态 | Profiler | 相对 SIMD | 正确性 |
|---|---:|---:|---|---:|---:|---|
| SIMD-only | 2,735.058 | — | 合法候选 | 354.284 us | 1.000× | PASS |
| Mixed | **1,247.622** | scope F1 | **recommended / effective** | 344.240 us | 1.029× | PASS |
| pure-SIMT | **2,427.871** | **F2** | 诊断候选 | **224.196 us** | **1.580×** | PASS |
| pure-SIMT | 高于 F2 | F4 | 诊断候选 | 276.558 us | 1.281× | PASS |

结论：当前 CostModel **误选 Mixed F1**；实测最优是 pure-SIMT F2。误差主要来自两处：

1. SuperBlock 对递推等待、tiny-dot 和 store 延迟隐藏的建模仍不完整；
2. 模型把 `dense_dot_tail + store` 的 SIMT/SIMD 相对成本估得过大，和同 shape CaModel 的比例不一致。

### 9.3 gather-dot-min

固定 shape：`M16/N16/K16/w32`。

| 路线 | CostModel score | 自动选择状态 | Profiler | 相对 SIMD | 正确性 |
|---|---:|---|---:|---:|---|
| SIMD-only | 494.715 | 合法候选 | 11.0125 us | 1.000× | PASS |
| pure-SIMT F1 | 470.477 | 诊断候选 | **3.162 us** | **3.483×** | PASS |
| Mixed F1 | **306.615** | **recommended / effective** | 7.542 us | 1.460× | PASS |

结论：Mixed 已正确物化并获得明显收益，但 CostModel 仍然**误选 Mixed**；实测最优是 pure-SIMT F1。当前 Mixed 分数低估了两次 scope 切换及大 tensor 经 UB 交接的真实成本。

### 9.4 FBGEMM gather-scale + FP8 rowwise quant

固定输入：`T256/D1024/E8/a512/BLOCK_D1024/grid512/w4`，启用 RowCoalescing factor 2。

| 路线 | CostModel score | AutoBlockify V1 | Profiler | 相对 SIMD | 正确性 |
|---|---:|---:|---:|---:|---|
| SIMD-only | 1,825.106 | — | 20.578 us | 1.000× | reference |
| pure-SIMT，无 V1 | — | 关闭 | 15.228 us | 1.351× | 容差 PASS |
| pure-SIMT F1 | — | F1 | 11.072 us | 1.859× | 容差 PASS |
| pure-SIMT F2 | — | F2 | 9.123 us | 2.255× | 容差 PASS |
| pure-SIMT F4 | **707.559** | **F4** | **8.904 us** | **2.311×** | 容差 PASS |
| Mixed F1 | 1,442.668 | scope F1 | 旧路线明显劣化，不再推荐 | — | — |
| CostModel auto | **选择 pure-SIMT F4** | F4 | 约 9.216 us | 2.233× | PASS |

结论：三个用例中，FBGEMM 已完成闭环。RowCoalescing 改善连续性，whole-kernel SuperBlock F4 进一步隐藏延迟；模型推荐与实测最优路线一致。auto 与 forced F4 的约 0.312 us 差异属于独立 profiler run 的波动，后续继续使用交错采样复核。

### 9.5 三用例总览

| 用例 | CostModel 推荐 | 实测最优 | 是否一致 | 当前判断 |
|---|---|---|---|---|
| solve_tril | Mixed F1 | pure-SIMT F2 | **否** | StageCost/SuperBlock 误差导致误选 |
| gather-dot-min | Mixed F1 | pure-SIMT F1 | **否** | Transition/UB handoff 成本偏低 |
| FBGEMM | pure-SIMT F4 | pure-SIMT F4 | **是** | RowCoalescing + whole-kernel SB 闭环 |

因此当前不是“三个 case 都校准完成”，而是 **1/3 路线选择完全一致，2/3 已跑通但仍存在排序误差**。

## 10. 当前 GAP 与下一步计划

### 10.1 当前 GAP

| GAP | 当前现象与根因 | 影响 |
|---|---|---|
| StageCostModel 绝对误差较大 | 解析公式已按 Stage 拆分，但 `setup/load/store/compute/predicate/shuffle/control/spill` 尚未逐项与同 TTIR、同 shape 的 CaModel cycle 对齐 | score 目前只能作排序代理；solve_tril 的 tiny-dot/store 比例明显失真 |
| SuperBlock 收益公式不完整 | 当前主要降低 memory/shuffle/divergence 等延迟，未完整表达多个 logical program 对递推等待、SIMT tiny-dot 以及 DCache store stall 的隐藏效果 | solve_tril F2 被低估，导致模型误选 Mixed |
| 局部 SIMT scope F2/F4 未打通 | whole-kernel pure-SIMT F2/F4 已由 NPUIR V1 支持；Mixed scope 仍只能 F1。直接套用 TA V1 会把 program-id 生成到 scope 外，并出现 BishengIR `gpu.thread_id` legalization 失败；仅放开 factor 还会因 launcher grid 未缩减造成 logical program 重复执行和错误输出 | 无法验证“第三阶段 SIMT-F2、其他 Stage SIMD”的理想 solve_tril 路线 |
| TransitionCost 仍缺定向实测 | 精确 live-in/live-out tensor bytes 和“producer store UB + consumer load UB”公式已经实现，但 SIMD→SIMT、SIMT→SIMD 固定启动延迟没有独立 microbenchmark；现有 182-cycle 数据只是 empty-VF harness proxy | gather 的 Mixed 成本被低估；不能把 `4 B/cycle/thread` 直接当成整个 scope 的聚合速率 |
| TA V1 与 NPUIR V1 语义尚未完全对齐 | NPUIR V1 的 whole-kernel block 聚合已成熟；TA 版本仍缺 launcher metadata/grid cap，以及 scope 内外 program-id、tensor/reduction 的一致映射 | pure-SIMT 应继续使用 NPUIR V1，不能用 TA 错误输出冒充性能收益 |
| Materializer 能力小于 RouteSolver 表达能力 | RouteSolver 可以表达任意连续 SIMT Stage，但当前 Materializer 只能稳定生成 F1 scope，且 Cube tail、跨 scope tensor 和控制流边界仍需逐类验证 | 模型可以“算出”暂时无法正确 lowering 的理想路线 |
| RowCoalescing 的代码归属与执行层级不一致 | `RowCoalescing.cpp` 位于 `TritonToLinalg` 目录，但当前实际由独立 `TTIRLayoutMergePass` 在 CostModel 之前复用和执行。运行顺序正确，源码位置却让人误以为 coalescing 发生在 CostModel 之后 | 架构难以从目录结构直接理解，后续维护容易重复执行、误改 pass 顺序，或错误判断 CostModel 看不到合并后的 TTIR |
| 实验与模型报告需统一 | profiler、CaModel、per-program score 的单位和聚合层级不同，历史章节曾混合展示 | 本文已删除重复历史表；后续每次只保留同一版本的三用例主表和原始数据索引 |
| DES 离线模块价值有限 | DES 不参与在线选择，也不能替代真实 profiler/CaModel 校准 | 保留为必要时解释 HIVM 调度的诊断工具，不扩展在线逻辑 |

### 10.2 下一步计划

| 优先级 | 工作 | 完成标准 |
|---:|---|---|
| P0 | 用同 TTIR、同 shape、同 factor 的 CaModel trace 校准 StageCostModel | 三个用例逐 Stage 输出解析 score、CaModel cycle、误差比例；先修 solve 的 recurrence/tiny-dot/store，再确认 FBGEMM 不回退 |
| P0 | 修正 SuperBlock 模型 | F1/F2/F4 分数能够解释 issue throughput floor、可隐藏 stall 和 register/stack pressure；solve 预测选择 F2，FBGEMM 仍选择 F4 |
| P1 | 建立 SIMD→SIMT、SIMT→SIMD 定向 microbenchmark | 分离固定启动成本与按 UB bytes 变化的斜率，替换 empty-VF fallback；gather 的 pure-SIMT/Mixed 排序与实测一致 |
| P1 | 实现 `ScopeSuperBlock` | 局部 scope 拥有正确的 logical PID、factor、grid 元数据和 live-in/live-out；solve Mixed-F2/F4 编译、正确性、profiler 全部 PASS |
| P2 | 整理 TTIR Layout Merge 的代码结构 | 将 RowCoalescing 从 `TritonToLinalg` 目录迁入独立 TTIR Layout Merge 组件，统一 pass 声明、头文件、CMake 和测试；保持 `RowCoalescing → CostModel → TritonToLinalg` 顺序，并删除后端中的重复入口/隐式复用，消除目录结构带来的误导 |
| P2 | 三用例统一回归 | 卡 0、统一 warmup/交错采样；输出三张“Stage score vs CaModel vs profiler”表，并执行 C++ UT、MLIR lit 与 Python correctness |
| P2 | 精简在线模块 | 继续保持 Coverage、Calibration domain、confidence、gain margin、Event gate 不参与在线 route；删除确认无消费者的旧实现，DES 仅离线诊断 |

### 10.3 今日结论

今天完成了 whole-kernel SuperBlock 候选与局部 scope 能力的拆分、精确 scope tensor bytes 统计、UB handoff 双向数据通路建模，以及 FBGEMM RowCoalescing + F4 的自动选择闭环。同时确认：

- solve_tril 和 gather 当前仍误选 Mixed；
- solve_tril 的正确最快路线是 pure-SIMT F2；
- Mixed scope F2/F4 不是简单放宽参数即可，需要真正的局部 ScopeSuperBlock 物化；
- 下一轮先用真实 CaModel Stage cycle 修 StageCostModel，再继续实现局部 scope F2/F4。
