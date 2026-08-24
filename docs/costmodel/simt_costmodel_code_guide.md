# StageCostModel V2 代码导读：调用链、关键实现与第一性原理

> 目标：让你完全掌握 PR 1708 head（`e2e2192d6`）中新增的 Stage-based Route Cost Model
> 实现思路。本文先给调用链，再逐组件讲关键函数实现，最后从第一性原理回答"为什么这样做"。
> 配套文档：《simt_costmodel_structured_design-v2.md》（设计文档）。
>
> 阅读范围（本分支新增/重构）：
> - `third_party/ascend/costmodel/lib/AscendModel/RouteModel/`（StagePartitioner.cpp 2109 行、
>   StageCostModels.cpp 915 行、StageRouteCostModel.cpp 585 行、SimtAnchorAnalysis.cpp 789 行、
>   SimdSimtCostModel.cpp 2905 行、Transforms/SelectSimdSimtCostModel.cpp 269 行、
>   Transforms/MaterializeSimtScopes.cpp 236 行）
> - `third_party/ascend/lib/AutoBlockifyV1/SIMTAutoBlockifyV1.cpp`（371 行）
> - `third_party/ascend/lib/TritonToLinalg/TTIRLayoutMergePass.cpp`（76 行）+ `RowCoalescing.cpp`（678 行）
> - `third_party/ascend/backend/compiler.py` 中 `simd_simt` 相关改动

---

## 目录

1. [全景：一次编译中 costmodel 的完整调用链](#1-全景一次编译中-costmodel-的完整调用链)
2. [前置变换：LayoutMerge 与 AutoBlockify V1（CostModel 的输入契约）](#2-前置变换layoutmerge-与-autoblockify-v1costmodel-的输入契约)
3. [SimtAnchorAnalysis：评分与物化共享的"锚点"](#3-simtanchoranalysis评分与物化共享的锚点)
4. [Feature 提取：从 post-transform TTIR 到 SimdSimtFeatureSummary](#4-feature-提取从-post-transform-ttir-到-simdsimtfeaturesummary)
5. [StagePartitioner：语义切分流水线](#5-stagepartitioner语义切分流水线)
6. [StageCostEvaluator 与 StageCostModels：stage 级成本公式](#6-stagecostevaluator-与-stagecostmodelsstage-级成本公式)
7. [solveStageRoutes：动态规划求解 kernel route](#7-solvestageroutes动态规划求解-kernel-route)
8. [SelectSimdSimtCostModel：决策落地与合法性收尾](#8-selectsimdsimtcostmodel决策落地与合法性收尾)
9. [MaterializeSimtScopes：把 route 变成 scope.scope](#9-materializesimtscopes把-route-变成-scopescope)
10. [第一性原理：为什么这套架构长这样](#10-第一性原理为什么这套架构长这样)

---

## 1. 全景：一次编译中 costmodel 的完整调用链

### 1.1 调用链总览

```
python 侧                          C++ 侧
─────────                         ──────
backend/compiler.py
  ttir_to_linalg(mod, metadata, opt)
    │  compile_mode == "simd_simt" && auto_simt_scope_mode != "off"
    ├─► _run_ttir_layout_merge(mod)          ──► add_ttir_layout_merge ──► TTIRLayoutMergePass
    ├─► _resolve_auto_blockify_v1_policy(...)     （TA V1 是否启用，黑名单判定）
    ├─► _run_cpp_simd_simt_costmodel(mod)    ──► add_select_simd_simt_costmodel ──► SelectSimdSimtCostModelPass
    │        （该 pass 内部完成 anchor 分析、feature 提取、stage 切分、
    │           stage 成本、route DP、决策、以及 mixed 的 scope 物化）
    │  ...并读取 module 上的属性：report_json / effective / superblock_factor
    ├─► _apply_cpp_simd_simt_decision(metadata, effective, factor, report)
    │        all_simd      → compile_mode=simd，关 V1
    │        mixed         → auto_simt_requested_kind=mixed
    │        all_simt_only → 可选开 whole-kernel V1
    │  effective==all_simt_only / 已是整函数 void simt scope
    │        → inline_void_simt_scopes_for_pure_simt, 直接返回 TTIR 走 pure-SIMT 编译器
    ├─► （否则继续走原有 ttadapter → linalg → bc → bishengir 链路，
    │     根据 metadata 里的 parallel_mode/auto_blockify_v1_* 拼接编译选项）
    │
    │  第 2 个 C++ 入口（校验用）：
    └─► add_materialize_simt_scopes ──► MaterializeSimtScopesPass（只校验契约，不改 IR）
```

**关键点**：Python 只是"调度 pass + 翻译决策"，一切分析/评分/选路都在 C++
`SelectSimdSimtCostModelPass::runOnOperation` 内完成。该 pass 内部调用链：

```
SelectSimdSimtCostModelPass::runOnOperation
├─ buildMixedSimtAnchorPlan(module, compileOn91095)      [SimtAnchorAnalysis.cpp:719]
│    └─ analyzeAnchor(op, compileOn91095)                [SimtAnchorAnalysis.cpp:499]
│         ├─ tt.gather             → DirectGather
│         ├─ tt.histogram          → Histogram
│         ├─ tt.scan（1D cumsum）  → PlainOneDimensionalCumsum
│         ├─ tt.atomic_*           → TensorAtomic
│         ├─ isTriangularSolveLoop → TriangularSolveLoop（并收集 scopeOperations）
│         └─ isLoadedIndexDependentMemoryOp → LoadedIndexDependentMemory
│    （每个 anchor 带 lowerability：Native/BackendConditional/AliasesMixed/Unsupported）
│
├─ analyzeSimdSimtCandidates(module, anchorPlan, options)   [SimdSimtCostModel.cpp:2897]
│    ├─ analyzeSimdSimtFeatures(module, anchorPlan)          [SimdSimtCostModel.cpp:1834]
│    │    对 module 做单遍 walk：op 计数、weightedOps/opElements、load/store bytes、
│    │    循环 trip count、loop-carried 依赖、anchor 内 captured/escaping 张量……
│    └─ estimateSimdSimtCandidatesImpl(features, options, module, &anchorPlan)
│         ├─ loadCandidateProfile(options.profilePath)       解析 JSON profile
│         ├─ evaluateSimtApplicability(...)
│         ├─ evaluateStageModel(features, profile, numWarps,
│         │                      wholeKernelSB, scopeSB, module, anchorPlan)
│         │    └─ StagePartitioner().partition(module, anchorPlan, features, opts)
│         │         ├─ PhaseBoundaryAnalysis().analyze(module, anchorPlan, features, opts)
│         │         │    ├─ analyze(features, opts)     → 判 domain（三个域之一）
│         │         │    ├─ ProgramStructureAnalysis().analyze(module, anchorPlan)
│         │         │    │    └─ collectTopLevelSemanticRoots + 复合 scope 顺序归一化
│         │         │    └─ assignRootPhaseIds(plan)    → 每个 root 一个单调 Phase id
│         │         ├─ StageBoundaryAnalysis().analyze(phasePlan, features, &anchorPlan)
│         │         │    ├─ partitionTriangular / partitionRowwise / partitionIndirectDot
│         │         │    │    （三套 Phase/Stage 模板，工作量从 kernel workload 里"精确取走"）
│         │         │    ├─ attachCompleteOperationOwnership(partition, phasePlan)
│         │         │    ├─ attachExactAnchorOwnership(partition, anchorPlan)
│         │         │    ├─ deriveStageLiveValues(partition)
│         │         │    └─ deriveLocalSimtScopeTraffic(partition, anchorPlan)
│         │         ├─ StageWorkloadAnalysis().analyze(partition)   ← 从 operation 树重算 workload
│         │         ├─ StageFeatureAnalysis().analyze(partition)    ← 从 operation 树重算 features
│         │         ├─ StageKindClassifier().analyze(partition, tinyDotFlopsMax)
│         │         ├─ StageModeLegalityAnalysis().analyze(partition, maxFactor, scopeSB)
│         │         └─ StagePartitionVerifier().verify(partition, kernelWorkload)
│         │              （守恒校验：Σ stage workload == kernel workload）
│         ├─ buildStageHardwareProfile(profile, numWarps)   → HardwareProfile
│         ├─ StageCostEvaluator().evaluate(partition, profile)
│         │    └─ 每个 Stage × 每个合法 implementation：
│         │         mapSIMDWorkload / mapSIMTWorkload → StageResourceCycles
│         │         registry.lookup(mode, kind) → 具体模型 estimate()
│         │         applySuperBlock(...)        → F2/F4 变体
│         └─ solveStageRoutes(costTable, transition)   → StageCostModelSummary
│              （3 类 route × factor 的 DP，见第 7 节）
│    └─ report.decision = chooseBest(...)       取三类候选里最小者
│
├─ buildSelectedMixedAnchorPlan(stageModel, anchorPlan)
│    （只保留 route 中 SIMT Stage 拥有的 anchor 子集 → 物化依据）
├─ module 属性写入：recommended / effective / scores / superblock_factor / report_json
└─ 若 effective==mixed：materializeSimtAnchorPlan(module, selectedPlan)
     └─ wrapAnchorOperation / wrapAnchorRange → 生成 scope.scope<simt>
```

### 1.2 为什么调用链长这样（一句话）

- **Python 只调度**：C++ 拥有全部决策逻辑，避免两处重复实现"什么能 SIMT"。
- **anchor plan 先行**：成本模型要计费的工作，必须与 materializer 实际搬进 scope 的工作
  是**同一组操作**——否则分数描述的路由和物化出来的路由不一致（见第 3 节）。
- **feature 提取在 anchor 之后**：`analyzeSimdSimtFeatures(module, anchorPlan)` 按
  anchor 的 `scopeOperations` 把工作分成"kernel 总工作量"与"anchor 内工作量"两份。

---

## 2. 前置变换：LayoutMerge 与 AutoBlockify V1（CostModel 的输入契约）

设计文档第 1.2 节要求 CostModel 看到的是 **post-layout、post-AutoBlockify** 的 TTIR。
两个新 pass 负责生产这份输入，并留下属性供 feature 提取消费。

### 2.1 TTIRLayoutMergePass（`TTIRLayoutMergePass.cpp`，76 行）

```cpp
ImplicitPermute 模式重写      // 识别隐含转置的 load/store/atomic，改写成显式转置
StridedAxisCoalescing::rewriteStridedAxisCoalesce(module)
TileChunkCoalescing::rewriteTileChunkCoalesce(module)
RowCoalescing::rewriteRowCoalesce(module)
CSE + Canonicalizer           // 清理
module->setAttr("ta.ttir_layout_merge.applied")
```

**为什么先跑 ImplicitPermute 再跑三个 coalescing？**
注释明说：coalescing 都识别原始 `tt.get_program_id` 图，而 AutoBlockify 会把它换成循环
归纳变量。所以 layout merge 必须发生在 AutoBlockify **之前**（compiler.py 中
`_run_ttir_layout_merge` 在 costmodel pass 之前调用；而 V1 在 pure-SIMT 路径中才跑）。

**为什么三个 coalescing 有顺序？** 它们共享 `hacc.coalesce_factor/axis` 属性，
先跑更具体的（StridedAxis / TileChunk），成功就写属性，RowCoalescing 看到属性直接 no-op。
这样三个变换不会互相覆盖对方已证明的访问模式。

### 2.2 RowCoalescing（`RowCoalescing.cpp`）——把"每行一个 program"变成"H 行一个 program"

这是给 rowwise 类 kernel（如 FBGEMM 的 rowwise quant，每 token 一行、D 维向量）设计的
**行提升**：原来 `pid` 是行号，现在 `pid` 对应 H 行，所有 1D tile 前插一维 H，
`tt.load/store` 的 mask 从 1D 行 mask 广播成 2D。关键函数：

- `matchRowSeed(moduleOp)`（L104）：找"唯一 pid + `cmpi pid >= n` + `cond_br` 到 return 块"
  的规范 rowwise 守卫。**为什么匹配这个模式**：证明 pid 是"被运行时行数守卫的行号"，
  才能安全地把一个 program 扩展成 H 行——若 pid 还被 `tl.num_programs` 消费过就放弃。
- `inferRowsPerProgram(maxBaseElements)`（L67）：按行 tile 大小反推 H。
  **为什么这样反推**：H 行共享一次 launch/setup，但每行寄存器和执行单元翻 H 倍；
  行越大 H 越小，`maxBaseElements > 1024 → H=1`（不值得 lift）。
- `rewriteMatchedRow`（L213）：核心重写。把 `pid` 替换成 `pid*H + arange(0,H)`，
  所有值经 `liftOperand`（expand_dims + broadcast 前插行维），store 增加 `rowMask`。
  `hacc.coalesce_factor = H` 属性被 compiler.py 导出为 launcher 网格收缩（grid / H）。

**为什么存在这个 pass**：Triton 源码里"每行一 program"是常见写法（grid = 行数），
物理核很多、行很小的时候 launch/setup 占比过高。合并后 CostModel 看到的 load 是
连续 2D tile（`continuous_tile_memory`），而不是 H 个 1D 离散行。

### 2.3 SIMTAutoBlockifyV1（`SIMTAutoBlockifyV1.cpp`）——物理核调度循环

NPUIR 同名 pass 的 TA 移植。**语义**：保持一个 logical program 的 tile 形状不变，
把物理 launch 限制到一波 vector core，把原始 logical programs 放进外层循环：

```
chunk = ceildiv(logicalGridSize, physicalVectorCoreCount)
for linear in [blockId*chunk, min((blockId+1)*chunk, logicalGridSize)):
    (pidX, pidY, pidZ) = unflatten(linear, gridX, gridY, gridZ)
```

关键实现点：

- `runOnOperation`（L96）：只处理 public、无结果的 entry kernel；收集原 `GetProgramIdOp`
  集合；split entry block；`scf.for` 步长默认 1（factor=1），tag `ta.auto_blockify_v1.loop`。
- **factor>1（SuperBlock）**（L180-209）：步长乘 factor，循环体内用
  `gpu.thread_id_x / 32 % factor` 得到 taskId，`logicalProgramId = iv + taskId`，
  越界 `if` 包住原 body。即一个物理 program 同时跑 factor 个 logical program。
- **关键属性**（L240-248）：新建的调度 op 全部打 `ta.auto_blockify_v1.schedule`，
  循环打 `ta.auto_blockify_v1.loop`。**为什么**：StagePartitioner 需要区分"调度壳"
  与"算法 body"——调度循环是 `auto_blockify_dispatch/loop` Stage，body 里的操作仍是
  普通语义 root，不能被子循环 double-own（见 5.2）。
- `TARefineSIMTAutoBlockifyV1SuperBlockPass`（L252）：纯-SIMT route 选 F2/F4 时，
  在已生成的 factor=1 调度循环上二次套 SuperBlock（step×factor + taskId 重算）。
  **为什么单独一个 pass**：编译链先以 factor=1 跑 V1 让 CostModel 看到调度循环，
  选完路由再 refine，避免"先按 F4 生成、路由又改回 F1"造成 IR 反复。

compiler.py 中 `_run_ta_simt_auto_blockify_v1` 传 `physical_vector_core_count`
（`NPUUtils().get_aivector_core_num()`），并把 `ta.auto_blockify_v1.materialized` 写进
metadata；NPUIR 的 V1（`--enable-auto-blockify-loop`）在 TA 已物化时被禁用，
防止同一调度跑两遍。

---

## 3. SimtAnchorAnalysis：评分与物化共享的"锚点"

### 3.1 为什么需要"锚点"概念

mixed route 的物理含义是"把某些操作搬进 `scope.scope<vector_mode="simt">`"。
如果成本模型和 materializer 各自独立判定"哪些操作该 SIMT"，二者必然漂移：
模型按 A 组操作计费，物化时搬走了 B 组，分数与实物不符。
**解法**：`SimtAnchorPlan` 是两者唯一的契约——feature 提取按它拆工作量，
selector 按它选路由，materializer 按它搬操作。注释原文：
"Keeping this contract in one place guarantees that mixed-candidate costs
describe the same operations that the selector can actually mark for SIMT."

### 3.2 关键函数

- `analyzeAnchor(op, compileOn91095)`（L499）：单 op 分类器，输出 `SimtAnchorDescriptor`
  （kind + facts + lowerability + materializable）。
  **分类逻辑**：
  - `tt.gather` → `DirectGather`；
  - `tt.histogram` → `Histogram`（检查 rank1 整数输入、i32 bins；**all-SIMD 直接标
    Unsupported**，因为 ascend950 的 histogram 符号就是 SIMT 模板）；
  - `tt.scan` → 只有 `analyzePlainOneDimensionalCumsum` 通过才算
    `PlainOneDimensionalCumsum`（1D、单 add 合并、支持 dtype）；
  - `tt.atomic_rmw/cas` → `TensorAtomic`（提取 updateElements、op 类型、lane 变化、
    mask 激活比例、地址是否依赖 loaded index；f16/bf16 的旧值语义被拒）；
  - `isTriangularSolveLoop` → `TriangularSolveLoop`；
  - 否则 `isLoadedIndexDependentMemoryOp`（load/store 指针 SSA 后向切片能到达
    `tt.load/tt.gather` 产生的索引）→ `LoadedIndexDependentMemory`。
- `isLoadedIndexDependentMemoryOp`（L704）：**真正的数据依赖测试**，替代旧版
  rank 代理（laneDependentPointerOps）。遍历指针反向切片，跨 `scf.for` iter_args
  时把 `scf.for` 的初始值与 `yield` 值都入栈继续追，直到命中 `tt.load`。
- `buildMixedSimtAnchorPlan`（L719）：**pre-order walk + skip**，保证 anchor 不重叠。
  对 TriangularSolveLoop 额外调用 `collectTriangularSolveScopeOperations`：
  - 找出从"最后一个输入 load 之后"到"最后一个递归循环 + 尾随 `uitofp/addf/select`"
    的**连续操作区间**；
  - 把区间内操作数反向切片里、位于首个 load 之前的**纯 tensor setup**
    （`make_range/expand_dims/broadcast/splat/arith 整数运算`）也拉进 scope，
    并且反复剔除"有 scope 外用户"的候选（固定点迭代，L465-482）。
  **为什么**：手写 solve_tril 的 scope 把 mask 构造放在 load 之后；自动物化要复现
  同一个边界，所以 `scopeInsertionPoint` 指向 load 之后。
- 最后聚合 kernel 级 `kernelLowerability`：全 anchor 的 all-simd/all-simt/mixed 状态
  取"最坏"，mixed 只有在"存在 native anchor 且无 anchor 阻止 all-simd"时为 Native。

---

## 4. Feature 提取：从 post-transform TTIR 到 SimdSimtFeatureSummary

`analyzeSimdSimtFeatures(module, anchorPlan)`（SimdSimtCostModel.cpp:1834）是**单遍
module walk**，产出模式无关的 `SimdSimtFeatureSummary`。关键逻辑：

1. **先读 pass 留的属性**（L1843-1860）：
   `ta.ttir_layout_merge.applied`、`hacc.coalesce_factor/axis`、
   `ta.auto_blockify_v1*`（schedule op 计数、loop 计数、动态 trip 标志）。
2. **建 anchor 集合**（L1861-1902）：materializable anchor 的 `scopeOperations` 全部入
   `anchorSet`；TriangularSolve 的 `scf.for` 固定 trip=14（16×16 块从第 2 行递推到第 16 行）。
3. **captured/escaping 张量**（L1911-1946）：anchor 内 op 的输入张量若在 scope 外定义
   → captured；anchor 内产生的结果若有 scope 外用户 → escaping。**为什么**：mixed
   的过渡成本（第 6/7 节 UB hand-off）按"真实跨 scope 的字节"计费，而不是按 stage
   live-in/out 粗算。
4. **主 walk**（L1966 起）：
   - V1 schedule op 直接跳过（不把调度循环当未知 trip 算法循环）；
   - op 计数（load/store/reduce/scan/gather/dot/atomic/histogram + arith/math 细分）；
   - `weightedOps/opElements` 乘 `loopMultiplier`（静态 trip 或结构性估计）；
   - `scf.for`：静态 trip 求值、`loopCarriedDataDependencyCount`（iter_args 中非地址
     类才计入；地址类计入 `pointerInductionDependencyCount`）；
   - load/store bytes 与 warp 指令数按 loopMultiplier 加权；
   - 分支：`scf.if/cf.cond_br` 计 conditional；条件 operand 是张量（lane-varying）→
     divergent。
5. **anchor 内同步累加**：同样的计数在 `features.simtAnchors` 里再记一份——
   **为什么**：mixed 成本 = 全 kernel 工作 − anchor 工作（SIMD 侧）+ anchor 工作
   （SIMT 侧），没有第二份计数就算不出来（见 SimdSimtCostModel.cpp:2550 的减法）。

**为什么 feature 是"模式无关"的**：`weightedOps`/`opElements` 只用通用名
（add/mul/div/exp/…），不含 kernel 名；下游 Stage 模型只消费这些计数。

---

## 5. StagePartitioner：语义切分流水线

这是本 PR 最大的新文件（2109 行）。核心职责：把 kernel 切成
**Phase（算法串行大步骤）→ Stage（单一 Kind 的最小建模单元）**，并给每个 Stage
配齐 workload、features、kind、合法性、live-in/out、anchor 归属。

### 5.1 三个受支持的 domain（PhaseBoundaryAnalysis）

`PhaseBoundaryAnalysis::analyze(features, opts)`（L1477）用**纯 feature 计数**判域：

| domain | 判别条件 |
|---|---|
| `TriangularRecurrence` | `simtAnchors.triangularSolves.size()==1 && count>0` |
| `LoadedIndexRowwiseReduction` | `dot==0 && reduce>0 && loadedIndexMemory>0 && load>0 && store>0` |
| `IndirectUnderfilledDot` | `dot>0 && reduce==0 && loadedIndexMemory>0 && dotFlops<=tinyDotFlopsMax && load>0 && store>0` |

**为什么只有三个域**：当前 materializer 只对三类机制有**经过验证的 SIMT 物化路径**
（solve_tril 递推、rowwise 归约、小 gather-dot）。domain 是"物化能力"的投影，
不是完备的 kernel 分类。设计文档明确：通用 CFG/SSA SCC 是未来方向，当前不假装实现。

### 5.2 ProgramStructureAnalysis：semantic roots 与调度壳分离

`collectTopLevelSemanticRoots`（L863）：遍历每个 `tt.func` 的 entry block，
顶层非 terminator op 都是 root；**特例**：带 `ta.auto_blockify_v1.loop` 的 `scf.for`
作为 root 收录后，其 body 的 direct ops 也**暴露为独立 root**（L874-880）。
**为什么**：V1 的循环只是调度壳，若把整个 `scf.for` 当一个 root，body 里全部算法
操作就被它"拥有"了；Stage 计费时调度循环和算法 body 必须分开（调度只付
`AutoBlockifyDispatch/Loop` 的标量+控制成本，算法操作按自己的 Kind 计费）。

`ProgramStructureAnalysis::analyze`（L1402）还在 anchor plan 上做**顺序归一化**：
复合 scope（solve_tril 的 `scopeOperations`）按 `scopeInsertionPoint` 重排 root 视图——
`setup → load → recurrence` 规范化为 `load → scope(setup+recurrence)`。
**为什么**：物化后的逻辑执行顺序和源码文本顺序不同，Phase 切分必须描述
"route 实际物化后的程序"，否则 `assignRootPhaseIds` 的单调性会被破坏。

### 5.3 assignRootPhaseIds：单调 Phase 状态机

（L917）三个枚举状态机（Triangular: Head→Load→Recurrence→MergeStore；Rowwise:
Index→Gather→Reduction→ConvertStore；IndirectDot: Index→Gather→Dot→OutputStore）。
对每个 root：
- 有 V1 属性 → `auto_blockify_dispatch`；
- anchor 区间内的 root → 强制置为 Recurrence/中段（**anchor 区域必须连续**，L940-948）；
- 遇到 `tt.dot/tt.store` → MergeStore（三角域）；
- 遇 `tt.load` → 从 Head 推进到 Load；遇 `tt.reduce` → Reduction；遇
  `isLoadedIndexDependentMemoryOp` → Gather。

**结束前校验**（L1046-1060）：Phase id 必须**单调**——已关闭的 Phase 不允许重新出现。
**为什么**：这是"边界来自算法顺序而不是成本"的硬约束；若按分数能来回切，就退化成
旧的"凑分数切分"。

### 5.4 StageBoundaryAnalysis：三套切分模板

`partitionTriangular`（L562）/`partitionRowwise`（L662）/`partitionIndirectDot`（L714）
结构一致，以 rowwise 为例：

```
P0 auto_blockify_dispatch + loop        （若 V1 已跑）
P1 row_dispatch      → IndexGeneration  （takeScalarAndPredicate + takeAllOperations）
P2 row_load          → IndirectGatherMemory（takeLoads；asLocalSIMT）
P3 row_reduction     → RowwiseReduction （shuffleLaneSteps + f32.max）
P4 convert_store     → ConversionPack   （takeStores + 剩余全部）
```

**workload 分配是"取走"（consume）而不是"复制"**：`buildKernelStageWorkload(features)`
先把 kernel 总量搬进 `remaining`，每个 Stage 用 `takeLoads/takeStores/takeDot/
takeScalarAndPredicate/takeAllOperations` 从 `remaining` 里**精确拿走**自己该付的工作，
最后 merge 剩余的进 store/head Stage。**为什么**：保证
`Σ stage workload == kernel workload`（守恒），Verifier 用 `near()` 复核
（L1864-1883）。若有工作没被任何 Stage 拿走，验证失败——这比"按权重分摊"诚实得多。

Triangular 域的细节（L604-625）：`recurrenceIterations = recurrenceLoopOps × (rows-2)`；
`parallelRecurrenceGroupCount = ceil(recurrenceIterations / iterationsPerGroup)`——
**为什么**：solve_tril 有多个兄弟递推循环（4 个 16×16 块），SIMT 可以用不同 warp
group 交错它们（SIMTRecurrenceStageCostModel 第 7 节会消费这个数）。

### 5.5 精确 operation ownership

`attachCompleteOperationOwnership`（L1098）：按 rootPhaseIds 顺序把每个 root 分配给
对应 Stage（`findStage` 查 id），维护 `lastStageOrdinal` 单调性，`owned` 集合保证
每个 root 恰好分配一次，最后 `owned.size() == rootOperations.size()`。

`deriveStageLiveValues`（L1249）：对每个 Stage，收集 owned 子树；operand 定义在
scope 外 → liveIn，result 有 scope 外用户 → liveOut；按静态 shape 算 bytes。
`deriveLocalSimtScopeTraffic`（L1280）：按 anchor 的 `scopeOperations`（不是 stage
live-in/out！）算真实 scope 边界字节——注释明说：stage 可以拥有 scope 外的一圈 SIMD
操作，用 stage live-out 计费会凭空发明 UB 流量。

### 5.6 StageFeatureAnalysis（L1565）与 StageKindClassifier（L1697）

- operation-graph 模式下 features **从 owned 操作树重新推导**（不是从模板抄）：
  `scf.for/while` → hasLoop；iter_args 非地址 → hasLoopCarriedDataDependency（地址 →
  hasPointerInduction，用 `isAddressOnlyLoopValue` 沿 addptr/arith 转发链判断）；
  `scf.if/cf.cond_br` → branch；load/store/gather/atomic → memory；reduce/scan →
  reduction；dot → dot；convert/pack 类 → conversion。
- **兄弟循环归一化**（L1639-1649）：多个兄弟递推循环压平进一个 Stage 的迭代空间，
  backedge/branch 计数要除以循环数再取整——否则每个循环都被收一次全量控制成本。
- `StageKindClassifier`：先查 `requires_split`——`hasDot && (hasReduction ||
  hasIndirectMemory || hasLoopCarried)` 直接报错（**不能挑一个更像的 Kind 覆盖**）；
  然后 `deriveKind()` 按强结构优先级 dot→reduction→conversion→loop→indirect→
  continuous→scalar 推导，再与 Stage 上已挂的 Kind 对表，不匹配即报 mismatch。
  转换 op 不单独构成 split 条件：predicate-to-float、累加器 cast 常是递推/dot 的附属。

### 5.7 StageModeLegalityAnalysis（L1977）

全 Stage `simdLegal=true, simtLegal=true, legalSimtFactors={1,2,4}`（按
maximumSuperblockFactor 截断）。local SIMT Stage 的 `localSimtFactors` 只有在
`scopeSuperblockMaterializable` 时才放开 F2/F4，否则锁 F1。
**为什么**：mixed F2/F4 需要 ScopeSuperBlockPass 把 SIMD producer + SIMT scope +
SIMD consumer 一起批处理；现在没实现，声称合法就是撒谎（compiler.py 注释同样强调）。

---

## 6. StageCostEvaluator 与 StageCostModels：stage 级成本公式

### 6.1 workload → 资源映射（mapSIMDWorkload / mapSIMTWorkload）

（StageCostModels.cpp:72 / 125）把 `StageWorkload`（模式无关）乘以 profile
（模式专属）变成 `StageResourceCycles`。两条路径的差异是**资源语义**：

| 资源 | SIMD（vectorWidth 32 元素/指令） | SIMT（warp 语义） |
|---|---|---|
| operationElements | `ceil(elements/vectorWidth)/throughput×factor` | `elements/throughput×factor` |
| load/store | 连续：`bytes/loadBytesPerCycle`；间接：`warpInstr/indirectLoadTransactionsPerCycle + 一次依赖延迟` | 连续：`warpInstr/loadWarpInstructionsPerCycle`；间接：同上 |
| issue | `ceil(issueElements/issueWidth)/issueOpsPerCycle` | 同形，warp issue |
| criticalPath | hasLoopCarried → scalar+compute+predicate+shuffle+dot；hasReduction → compute+predicate+shuffle | 同形 |

`serialBody`（L30）**故意把 issue 排除在 execution 之外**（`max(execution, issue)`）：
issue 是共享前端吞吐下界，不是独立指令流，加进去每指令重复计费。

### 6.2 Registry：模式 × Kind → 具体模型

18 个具体模型类（SIMD/SIMT × 9 类语义）。`supports()` 说明"一类多 Kind"：
SIMDScalar 吃全部 6 个 scalar Kind、SIMDContinuous 吃 4 个 continuous Kind——
与设计文档"约 8 组语义模型"一致。`lookup` 查重（多个模型支持同一 key → 报错）、
`verifyComplete` 枚举 20 Kind × 2 模式全部可查。

### 6.3 各模型的公式（关键差异）

| 模型 | 公式 | 为什么 |
|---|---|---|
| Dispatch（SIMD/SIMT 同形） | `setup + count×max(scalar+control, issue) + epilogue` | 调度只有标量+控制，无访存/计算负载；count 只对 loop kind 乘迭代 |
| Scalar | `setup + N×serialBody + epilogue` | 标量块天然串行 |
| ContinuousMemory (SIMD) | 可重叠时 `scalar+predicate+control+spill + max(load,store,issue)`；否则 serial | `permitsSimdOverlap` = schedule==IndependentPipelined 且无 loop-carried 依赖 |
| ContinuousMemory (SIMT) | 永远 `serialBody` | 当前 SIMT 路径没有可证明的重叠契约，不装 |
| IndirectMemory | `serialBody` | 离散地址 + 未隐藏延迟，不能套 MTE 连续 roofline |
| Independent (SIMD) | 可重叠时 `max(load,store,compute+dot+shuffle,scalar+predicate+control,issue)+spill` | 多流水 roofline |
| Recurrence (SIMD) | `setup + N×max(criticalPath+load+store+control+spill, issue) + epilogue` | 关键路径必须乘 N，不能被吞吐隐藏 |
| Recurrence (SIMT) | 关键路径乘 `ceil(N/interleavedGroups)`，issue 地板乘 N | **SIMT 能把多个独立递推组放不同 warp group 交错**：`parallelRecurrenceGroupCount` 与 `logicalWarpGroupCount` 取小者 |
| Reduction | `setup + N×max(scalar+load+store+criticalPath+control+spill, issue) + epilogue` | tree 深度是依赖链，issue 是全体指令流下界 |
| Cube (SIMD) | 可重叠时 `scalar+predicate+control+shuffle+spill + max(load, compute+dot, store, issue)` | Cube 与 MTE 并行 |
| ConversionPack (SIMD) | 可重叠时 `predicate+control+spill + max(scalar+compute, load, store, issue)` | convert/pack 与访存仅在依赖允许时重叠 |

### 6.4 applySuperBlock（L178）——F2/F4 的成本公式

```
latencySensitive = N×(load+store+shuffle+divergence)
pressure        = N×spill×max(0, factor-1)
persistentState = (递推? max(0, factor - pressureFreeFactor) × liveOutBytes / bytesPerCycle : 0)
cost = max(issueFloor, cost − latencySensitive + latencySensitive/effectiveFactor + pressure)
       + persistentState
```

**第一性原理**：SuperBlock 的本质是"把 factor 个独立 logical program 塞进一个物理
核"。它只能**隐藏延迟**（latencySensitive 项按有效因子缩小），不能**减少指令**
（issueFloor 是地板），还会**复制状态**（递推 Stage 的 liveOut 被 factor 份复制 →
持久态压力项）。三个 profile 参数
（`superblockUsefulFactorLimit`/`superblockPersistentStatePressureFreeFactor`/
`superblockPersistentStateBytesPerCycle`）分别表达"延迟隐藏收益上限"、
"复制状态免费的上限（F2 通常免费、F4 起收费）"与"每 cycle 能吞多少复制状态"。
所以 F4 不是 F2 的线性外推。

### 6.5 StageCostEvaluator::evaluate（L802）

校验 partition/profile → 逐 Phase 逐 Stage：
- 枚举合法 implementation：SIMD-F1 + SIMT×legalSimtFactors；
- `mapSIMDWorkload/mapSIMTWorkload` 算资源；
- `registry.lookup(mode, kind)` 取模型，`estimate()` 算 base cycles；
- `applySuperBlock` 包裹（SIMT factor>1 时）；
- 校验 `cost.isValid()`（有限非负、modelName/profileVersion 非空）。

输出 `StageCostTable`：stageId → 每个 implementation 的总周期 + 资源分解 + 模型名。

---

## 7. solveStageRoutes：动态规划求解 kernel route

（StageRouteCostModel.cpp:417）输入 `StageCostTable` + `StageTransitionCost`，
输出三类 route（AllSIMD/AllSIMT/Mixed）各一条最优路径。

### 7.1 状态与剪枝

```
State[exitMode∈{SIMD,SIMT}][routeClass∈{AllSIMD,AllSIMT,Mixed}] → map<factor, PartialRoute>
```

DP 按 stage 顺序推进；每个候选 (stage, implementation) 扩展所有前驱。剪枝规则：

1. **factor 一致性**（L482-486）：route 里已有 SIMT 后，后续 SIMT stage 的 factor
   必须与首次 SIMT 的 factor 相同。**为什么**：SuperBlock 是 kernel 级调度，
   一个 kernel 不可能一半 stage F2 一半 F4（launch 网格只有一份）。
2. **mixed 只接受"全 local SIMT"**（L496-504）：route 变 Mixed 时，
   `allSimtStagesLocal` 要求每个 SIMT stage 都 `localSimtMaterializable`。
   **为什么**：整 kernel SIMT 的 F2/F4 与局部 scope 的 F2/F4 是两种物化路径；
   纯-SIMT route 走 whole-kernel V1，mixed route 走 local scope。混合二者不可物化。

### 7.2 mixed 的计费技巧：mixedEquivalentCycles

`mixedEquivalentStageCost`（L38）：SIMT 且 localSimtMaterializable 的 stage，
在**进入 Mixed 类的那一刻**把成本换成：
```
selected.totalCycles
+ scopeCount×(simdToSimt + simtToSimd)          // 每 anchor 一对物理方向切换
+ inputBytes/simdUbStorePerCycle + inputBytes/simtUbLoadPerCycle   // 寄存器→UB→寄存器
+ outputBytes/simtUbStorePerCycle + outputBytes/simdUbLoadPerCycle
```
**为什么用"反事实等价成本"而不是在边上直接加**：route 在 stage 前还不知道自己是
mixed 还是 pure-SIMT；等到变 Mixed 时再一次性补记全部 scope 流量，而 AllSIMT 路径
永远不付这笔钱。`scopeCount` 很重要：materializer 每个 anchor 建一个 scope，
DP 若只看到一次模式切换，会把 N 个 scope 的边界成本记成 1 对。

### 7.3 输出

`toPlan` 把三类最优 PartialRoute 各自转成 `StageRoutePlan`
（implementations、entryTransitionCycles、logicalStageCycles、factor、totalCycles），
再按 Phase 聚合 `logicalPhaseCycles`。`StageCostModelSummary` 携带完整 stage 明细 +
transition + 三路线，直接进 `SimdSimtCostReport.stageModel`。

---

## 8. SelectSimdSimtCostModel：决策落地与合法性收尾

（SelectSimdSimtCostModel.cpp:120）`runOnOperation` 流程：

1. 清旧属性 → 建 anchor plan → `analyzeSimdSimtCandidates`。
2. `report.stageModel.applied` 时，`candidateCosts` 三个分数直接取 stage route 的
   totalCycles（SimdSimtCostModel.cpp:2454-2456）——**stage 模型是唯一打分者**；
   只有 stage 模型不适用（domain 未命中）才走旧的 aggregate feature 公式
   （L2480 之后的 fallback 分支，保留兼容）。
3. 决策 = `chooseBest`（合法候选中分数最小）。合法性：
   - all-simd：anchor 的 all-simd lowerability == Native；
   - all-simt：compileOn91095 && 无显式 scope && Native/BackendConditional；
   - mixed：无显式 scope && materializable && mixed lowerability == Native。
4. **应用前检查**（L172-212）：
   - mixed 且已有显式 scope → 不应用（不覆盖用户手写 scope）；
   - mixed 但选出的 anchor 为空 → 不应用；
   - mixed factor>1 且无 scopeSuperblock → 不应用（只能看不能跑）；
   - factor×numWarps > 64 → 不应用（warp 上限）；
   - all-simt factor>1 且既无 V1 又无 wholeKernelSuperblockMaterializable → 不应用。
   **为什么这些是"应用闸门"而不是"分数闸门"**：分数照常报告，
   不能物化的 route 只是不执行，不影响推荐与诊断。
5. 写属性：`recommended/effective/scores/superblock_factor/report_json`；
   `auto` 模式下 effective=recommended（actionSupported 时）。
6. effective==mixed → `materializeSimtAnchorPlan(module, selectedPlan)` 当场物化。
7. JSON 报告追加到 `reportFile`（每行一条）。

---

## 9. MaterializeSimtScopes：把 route 变成 scope.scope

### 9.1 materializeSimtAnchorPlan（MaterializeSimtScopes.cpp:151）

对 plan 中每个 materializable anchor（跳过已在 SIMT scope 内的、被 range 覆盖的）：

- **单 op anchor** → `wrapAnchorOperation`（L51）：在 op 前建 `scope.scope`（结果类型
  与 op 相同、`vector_mode="simt"`），把 op 移进 scope body，`scope.return` 返回原结果，
  scope 结果替换 scope 外的所有 use。**scope 不隔离（not IsolatedFromAbove）**，
  操作数天然是合法 capture——SIMD producer 留在外面。
- **复合 range anchor** → `wrapAnchorRange`（L84）：按 `scopeInsertionPoint` 插入 scope，
  只把**逃逸结果**（有 scope 外用户的值）经 `scope.return` 传出，scope 结果类型=
  逃逸值类型。这是 solve_tril 的路径：mask setup 被搬进 scope 且跨过 load。

### 9.2 MaterializeSimtScopesPass（L218）——契约校验

`isMixedModelDecision(module)` 且 module 里没有任何 `scope.scope<simt>` → 报错。
**为什么**：mixed 决策是 module 级属性，物化在同一个 pass invocation 内完成；
下游若看到 mixed 决策却没有 scope，说明有别的 pass 把 scope 吃了，必须显式失败
而不是静默降级。

### 9.3 simt_selection 工具（SimtSelection.h，纯 inline）

- `getEffectiveExecution`：沿 op 向上找最近的 `ascend.simt_costmodel.effective` 属性
  （函数级可覆盖 module 级）；
- `shouldUseSimtTemplate(op, legacyForceSimt)`：**model-controlled 编译里忽略全局
  force flag，只有显式 `vector_mode=simt` 才允许 SIMT lowering**；backend_default
  时保持旧全局行为。这是"route 权威"在 lowering 层的落地。
- `inlineVoidSimtScopesForPureSimt`：pure-SIMT 路径先把无结果 scope 内联掉——
  pure-SIMT 编译器不再需要 scope 结构（结果型 scope 留给 mixed lowering 路径）。

---

## 10. 第一性原理：为什么这套架构长这样

### 10.1 旧模型的两个结构性错误（本重构的动机）

**旧模型（aggregate feature）**：把整个 kernel 压成一个特征向量，算
`all-SIMD / all-SIMT / mixed` 三个总分。两个错误：

1. **信息坍缩**：递推循环的 critical path 和独立循环的 roofline 需要不同的数学
   公式；压成一个标量后只能用"结构惩罚系数"近似，系数靠经验，无法解释。
2. **分数不可归因**：mixed 分数是"anchor 工作 × SIMT 速率 + 其余 × SIMD 速率"的
   混合，边界成本挂在 kernel 级。没人能说清一个 154,136 到底贵在哪。

**第一性原理的推倒重来**：
> 一条 kernel route 的成本 = 串行 stage 成本之和 + 模式切换成本之和。
> 每个 stage 的成本公式由**它的关键路径结构**决定（递推 → critical path×N；
> 独立 → roofline；离散访存 → transaction）。stage 边界必须来自**算法顺序**，
> 而不是分数便宜就切。

### 10.2 由此导出的五个设计决策（与设计文档五条一一对应）

| 设计文档改造 | 代码落地 | 第一性原因 |
|---|---|---|
| 1. 分段式 CostModel | StagePartitioner→StageCostEvaluator→solveStageRoutes 三段，数据对象 StagePartition/StageCostTable/StageRoutePlan | 让"算哪一段"（结构）、"每段多少钱"（成本）、"怎么拼"（路由）各自可测试、可解释、可单独校准 |
| 2. 去掉价值不大的门禁 | coverage/confidence/gain-margin/Event-validated 全部不存在；只有 lowerability 与 materializable 参与合法性 | 门禁是"用历史性能否决结构结论"，本质是预测模型的自我怀疑；结构+微基准已经是答案，再加门禁只引入不可复现性 |
| 3. AutoBlockify V1 | `ta.auto_blockify_v1.*` 属性 + 调度壳/算法 body 分离 + `AutoBlockifyDispatch/Loop` stage | 调度循环的成本（每物理核一次 setup + 逻辑 program 迭代）与算法成本必须分开计，否则 V1 的收益/代价无法量化 |
| 4. Layout 合并 | TTIRLayoutMergePass（ImplicitPermute+三 coalescing）+ feature 读 `hacc.coalesce_factor` | CostModel 必须对"编译器实际会生成的访存形态"计费。按源码 1D 行计费会系统性高估行数、漏掉行提升后的连续 tile |
| 5. SIMT Scope SuperBlock | localSimtFactors 闸门 + applySuperBlock 公式 + compiler.py 的 scope_superblock 编译选项 | F2/F4 的收益来自多 warp group 隐藏延迟，代价是状态复制与同步；未实现的物化路径绝不声称合法 |

### 10.3 贯穿全程的三个不变量（代码里反复出现的校验）

1. **ownership 守恒**：每个 TTIR op 恰好属于一个 Stage（`attachCompleteOperationOwnership`
   的 `owned` 集合、Verifier 的 `ownedOperations.size()==modeledOperationCount`）；
   每个 workload 元素恰好被一个 Stage 取走（consume 系列 + `StageWorkloadAnalysis::verify`
   的 `near()` 复核）。
2. **分数与物化同源**：计费的 operation 集合 == materializer 搬进 scope 的集合
   （SimtAnchorPlan 单一契约 + `deriveLocalSimtScopeTraffic` 用 anchor 而非 stage 边界）。
3. **边界单调**：Phase id 只能前进不能回头（`assignRootPhaseIds` 的 closedPhases 检查），
   杜绝"按分数反向切分"。

### 10.4 为什么 SIMT 侧的公式普遍比 SIMD 保守

对照第 6.3 节：SIMD ContinuousMemory/Independent/Cube/Conversion 都有 roofline
（max）形态，SIMT 侧几乎全是 `serialBody`。**原因**（SimdSimtCostModel.cpp:2683 注释）：
当前 SIMT lowering 发射的是**依赖序 warp 指令流**（load→compute→shuffle→store），
没有实测重叠契约支撑 SIMT 的 roofline；`independent_pipelined_loop` 的 SIMT 版本因此
也是串行。这是诚实的建模：**没测过就不装重叠**。SIMT 的赢面不在吞吐重叠，而在
recurrence 的交错（parallelRecurrenceGroupCount）、DCache 对离散访问的容纳、
以及不必为小 dot 付 Cube setup。

### 10.5 三用例在代码里的投影（对照设计文档 3.6 的表）

| 用例 | domain | Phase/Stage（代码 id） | 关键参数来源 |
|---|---|---|---|
| solve_tril | TriangularRecurrence | head_index_mask / load_diagonal_tiles / diagonal_inverse_recurrence / dense_dot_tail + store_inverse_tile | `TriangularSolveFacts`：blockRows/Columns=16×16、recurrenceStartRow=2、recurrenceLoopCount=14×循环数、denseDotTailOps |
| FBGEMM rowwise quant | LoadedIndexRowwiseReduction | row_index_generation / indirect_row_gather / rowwise_reduction / conversion_pack_store | RowCoalescing 已把行合并为 2D tile（coalesceFactor 属性）；loaded-index gather 由 `isLoadedIndexDependentMemoryOp` 判定 |
| gather-dot-min | IndirectUnderfilledDot | index_generation / indirect_tile_gather / tiny_cube_dot / store_dot_result | `tinyDotFlopsMax=16384`（profile structural）决定 TinyCubeRoofline |

---

## 11. 常见疑问：StageCostModel 的九个本质问题

> 本节是针对代码阅读中容易产生的九个疑问的详细解答。所有结论都可以在上文对应
> 代码位置直接验证；回答尽量讲“代码实际做了什么”和“为什么这样做”，而不是只给
> 设计层口号。

---

### 11.1 evaluateSimtApplicability 是如何评估可行性的？具体法则是怎样的？

`evaluateSimtApplicability` 位于 `SimdSimtCostModel.cpp:1290`，它并不是一个完整的
“性能可行性”评估，而是进入 Stage 成本模型之前的一个**前置门槛**。它只回答一个问题：
“当前 kernel 是否存在值得继续做 SIMD/SIMT 路由分析的机制，并且目标后端能否物化”。

具体法则非常直接：

| 字段 | 计算方式 |
|---|---|
| `targetSupported` | 由调用方传入，当前是 `options.compileOn91095` |
| `recognizedAnchorCount` | `features.simtAnchors.recognizedCount`，即 `SimtAnchorPlan` 里识别出的 anchor 总数 |
| `materializableAnchorCount` | `features.simtAnchors.count`，即真正能物化（`materializable`）的 anchor 数 |
| `mechanisms` | 所有 anchor 的 `mechanismKinds` + `observedMixedKinds` 去重排序 |
| `mechanismDetected` | `recognizedAnchorCount > 0 || !mechanisms.empty()` |
| `materializable` | `targetSupported && materializableAnchorCount > 0` |

决策原因（`reasons`）按优先级只有三种：

1. 没有任何已识别机制：`no_recognized_simt_mechanism`
2. 有机制但目标不支持 SIMT 物化：`target_does_not_support_simt_materialization`
3. 有机制、目标支持，但没有可物化 anchor：`no_materializable_simt_anchor`

**本质**：这个函数不评估“SIMT 是否更快”，它只评估“有没有资格进入后续合法候选集合”。
是否更快由 `evaluateStageModel` → `StageCostEvaluator` → `solveStageRoutes` 决定。
因此不要把 `evaluateSimtApplicability` 看作可行性评分，它更像一个 `can_materialize` 门禁。

---

### 11.2 phase 似乎是根据模板来匹配的，这会导致泛化性非常差；讨论后说把 phase 去掉，直接按照语义去进行 stage 划分的话该怎么做最大化泛化性呢？

你的观察是正确的。当前 `PhaseBoundaryAnalysis::analyze(features, opts)`
（`StagePartitioner.cpp:1477`）只支持三个硬编码 domain：

- `TriangularRecurrence`
- `LoadedIndexRowwiseReduction`
- `IndirectUnderfilledDot`

并且 `assignRootPhaseIds`（`StagePartitioner.cpp:917`）为每个 domain 写死了
单调状态机：

```
Triangular: Head -> Load -> Recurrence -> MergeStore
Rowwise:    Index -> Gather -> Reduction -> ConvertStore
IndirectDot: Index -> Gather -> Dot -> OutputStore
```

这确实泛化性差：新增一种 kernel 结构就要新增一个 domain 和一套状态机。

如果“去掉 phase、直接按语义划分 stage”，最大化泛化性的做法应当是：

1. **用统一的语义图代替 domain 模板**
   把 top-level semantic roots 看成一张有向图：
   - SSA 数据依赖边（operand -> user）
   - 控制流边（region/block 顺序）
   - 循环携带依赖边（`scf.for`/`scf.while` 的 iter_args）
   然后在这个图上做通用的结构切分，而不是依赖“这是 solve_tril”这种 kernel 身份。

2. **用 SCC/依赖闭包找出“必须串行”的算法阶段**
   - 强连通分量（SCC）天然对应循环携带依赖/递推/迭代间依赖；
   - 递推 SCC 内部是一个不可拆散的串行 Stage；
   - 从 SCC 出发，再按“访存依赖”“控制依赖”扩展成连续区域。

3. **用 StageKindClassifier 的 deriveKind 作为 Stage 单一语义的判据**
   `StageKindClassifier::analyze`（`StagePartitioner.cpp:1697`）已经具备从 exact
   operation graph 推导 dominant kind 的能力：
   `dot -> reduction -> conversion -> loop -> indirect -> continuous -> scalar`。
   如果 StageBoundaryAnalysis 能先把图切成“每个子图只含一种 dominant semantics”
   的连通块，那么 phase 模板就不再需要。

4. **把“Phase”退化为 Stage 的序列分组**
   Phase 可以保留为“一组必须按程序顺序执行的 Stages”，但它不应由 domain 枚举产生，
   而应由图的拓扑序和依赖边界自动产生。例如：
   - 一个 SCC → 一个串行 Phase；
   - 一个 SCC 后面跟着的独立内存/计算块 → 另一个 Phase；
   - 多个互不依赖的同构子图 → 可以合并成同一类 Stage 或保留为并行 Stage。
   这样 Phase 就从“模板”变成“依赖图的可达性/拓扑分层”。

5. **保留 anchor 作为物化契约，而不是作为切分模板**
   `SimtAnchorPlan` 仍然应该决定哪些区域可以物化为 local SIMT，但它不应当决定
   “kernel 分几段”。通用切分器可以：
   - 先做语义图切分；
   - 再把 anchor 区域映射到对应 Stage；
   - 如果 anchor 跨了多个语义 Stage，则说明该 anchor 本身需要拆开或说明
     Stage 边界与物化边界不一致，应报错而不是强行套模板。

6. **最大化泛化性的关键原则**
   - 不再使用 kernel 名、算子组合模板、`dotOps == 0 && reduceOps > 0` 这类特征组合；
   - 全部边界都从 SSA/CFG/循环携带依赖中推导；
   - Stage 的 kind 由 `StageModelFeatures` 自动判定；
   - 新增硬件机制时，只需在 `SimtAnchorAnalysis` 和 `StageCostModels` 注册新机制/
     新 kind，不需要改 StagePartitioner 的 domain 状态机。

这样做的收益是：solve_tril、rowwise quant、gather-dot 只是“恰好落到同一种通用
切分结果”的特例，而不是“被模板枚举出来的三种情况”。

---

### 11.3 StageBoundaryAnalysis 是怎么划分的，划分的依据是什么，是 anchor 吗？

不是 anchor。`StageBoundaryAnalysis`（`StagePartitioner.cpp:1525`）的划分依据是
**PhaseBoundaryPlan 给出的算法 Phase 顺序**，anchor 只是后续的物化证据。

流程分两层：

1. **第一阶段：按 Phase 模板生成 Stage 骨架**
   `StageBoundaryAnalysis::analyze` 根据 `phasePlan.domain` 调用
   `partitionTriangular` / `partitionRowwise` / `partitionIndirectDot`
   （`StagePartitioner.cpp:562/662/714`）。
   这三套函数从 `buildKernelStageWorkload(features)` 里用
   `takeLoads/takeStores/takeDot/takeScalarAndPredicate/takeAllOperations`
   精确“取走” workload，生成一组 Stage。这个阶段本质上是“domain 模板切分”。

2. **第二阶段：精确 operation ownership**
   如果 `PhaseBoundaryPlan` 带有 operation graph，则调用
   `attachCompleteOperationOwnership`（`StagePartitioner.cpp:1098`）。
   它按照 `rootPhaseIds` 把每个 top-level semantic root 分配给对应 Stage，
   并维护：
   - Stage 序号单调（不能回退）；
   - 每个 root 只被分配一次；
   - 分配结果必须覆盖全部 root。

3. **anchor 的角色**
   anchor plan 只在 `attachExactAnchorOwnership`
   （`StagePartitioner.cpp:815`）和 `deriveLocalSimtScopeTraffic`
   （`StagePartitioner.cpp:1280`）中使用：
   - 判断某个 Stage 是否真的由可物化 local SIMT anchor 支撑；
   - 计算跨 scope 的真实 tensor 流量。
   它不决定 Stage 边界。

**一句话**：Stage 边界来自“算法 Phase 的单调顺序 + 每个 root 的语义归属”，
anchor 只是确认“这个 Stage 是否有对应的物化证据”。

---

### 11.4 StageWorkloadAnalysis 和 StageFeatureAnalysis 作用是什么，前面不是 analyzeSimdSimtFeatures 已经分析过了吗？和前面分析结果的逻辑关系是什么？

三者不是重复，而是不同层级的分析：

| 分析 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `analyzeSimdSimtFeatures` | 整个 module + anchor plan | `SimdSimtFeatureSummary` | kernel 级全局特征：op 计数、bytes、loop 信息、anchor 内 captured/escaping 等 |
| `StageWorkloadAnalysis` | 已划分好的 Stage（精确 operation ownership） | 每个 Stage 的 `workload` | 从每个 Stage 真正拥有的操作树重新计算动态工作量 |
| `StageFeatureAnalysis` | 已划分好的 Stage | 每个 Stage 的 `features` | 从每个 Stage 真正拥有的操作树重新计算结构特征 |

逻辑关系：

1. `analyzeSimdSimtFeatures` 是**全局粗粒度**分析。它先于 Stage 划分，主要用来：
   - 判断 domain；
   - 生成初始 `buildKernelStageWorkload(features)`；
   - 计算 anchor 内工作量，供 mixed 计费；
   - 提供 kernel 级 lowerability。
   它不知道“某个 op 最终属于哪个 Stage”。

2. Stage 划分完成后，`StageWorkloadAnalysis::analyze`
   （`StagePartitioner.cpp:1885`）用 `accumulateDynamicOperationTree` 对每个 Stage
   的 owned operations 重新累加 workload，并按 `iterationCount` 归一化。
   这是必要的，因为模板切分阶段用 `take*()` 得到的 workload 只是“按全局特征
   估出来的初始分配”；一旦拿到精确 operation ownership，就必须用真实操作树重算，
   否则 Stage 工作量可能与实际代码不一致。

3. `StageFeatureAnalysis::analyze`（`StagePartitioner.cpp:1565`）同理：
   - operation-graph 模式下，它从 owned 操作树重新推导 `hasLoop`、
     `hasLoopCarriedDataDependency`、`hasIndirectMemory`、`hasReduction`、
     `hasDot`、`hasConversionPack` 等；
   - 它还会做兄弟循环归一化，避免多个循环的 control 成本被重复计算。
   在 fallback（无 operation graph）模式下，它才退化为从 Stage kind 推断 features。

**关系总结**：
`analyzeSimdSimtFeatures` 负责“kernel 是什么”，
`StageWorkloadAnalysis` 负责“每个 Stage 有多少动态工作”，
`StageFeatureAnalysis` 负责“每个 Stage 是什么结构”。
前者是后两者的全局输入和守恒基准，后两者是 Stage 划分后的精确化/校验。

---

### 11.5 StageModeLegalityAnalysis 有必要存在吗？毕竟 StageKindClassifier 不是已经划分好 kind 类型了吗？

有必要。`StageKindClassifier` 和 `StageModeLegalityAnalysis` 回答的是两个不同问题：

- `StageKindClassifier`：这个 Stage 的**成本模型种类**是什么？
  （`LoopCarriedRecurrence` / `IndirectGatherMemory` / `CubeRoofline` ...）
- `StageModeLegalityAnalysis`：这个 Stage 可以有哪些**实现方式**？
  （SIMD？SIMT？SIMT 下可以用 F1/F2/F4？local mixed scope 能不能用 F2/F4？）

`StageModeLegalityAnalysis::analyze`（`StagePartitioner.cpp:1977`）设置：

- `simdLegal = true`
- `simtLegal = true`
- `legalSimtFactors = {1, 2, 4}`（由 `maximumSuperblockFactor` 截断）
- `localSimtMaterializable` 的 Stage 再设置 `localSimtFactors`

它把“后端当前能不能物化”作为独立约束：
- 纯-SIMT F2/F4 只有在 whole-kernel AutoBlockify V1 / SuperBlock 可用时才合法；
- local mixed F2/F4 只有在 `scopeSuperblockMaterializable` 时才合法；
- 否则 local mixed scope 只能 F1。

如果没有这个分析，`StageCostEvaluator` 会把所有 SIMT F2/F4 都当成可枚举候选，
`solveStageRoutes` 就可能选出一条当前后端无法执行的 route。所以它不是冗余，
而是“评分空间”之前的合法性裁剪。

---

### 11.6 buildStageHardwareProfile 要来干嘛的，前面不是已经有 loadCandidateProfile 去加载硬件信息了吗？

`loadCandidateProfile` 和 `buildStageHardwareProfile` 是两层：

1. `loadCandidateProfile`（`SimdSimtCostModel.cpp:696`）
   - 从 JSON 文件加载原始 `CandidateProfile`；
   - 解析 schema、microbenchmark 引用、simd/simt ops、memory、dot、
     structural 惩罚、transition 等；
   - 它是“文件格式层”的解析器。

2. `buildStageHardwareProfile`（`SimdSimtCostModel.cpp:1163`）
   - 把 `CandidateProfile` + `numWarps` 转换成 Stage 模型真正消费的
     `HardwareProfile`；
   - 做单位换算和派生计算，例如：
     - `simd.vectorWidth = simdVectorWidthBits / 32`
     - `simt.issueWidth = simtWarpSize`
     - `predicateOperationsPerCycle = throughput / factor`
     - `logicalWarpGroupCount = numWarps`
     - 把 superblock 参数、transition 参数复制到 `HardwareProfile`
   - 这是“领域模型层”的适配器。

**为什么不能直接用 CandidateProfile**：StageCostModels 不关心 JSON 结构，
它只关心 `HardwareProfile` 这种已经算好、校验过、缺省值也处理过的 C++ 对象。
`buildStageHardwareProfile` 让 JSON schema 和成本公式解耦：以后 profile 格式变化时，
只需要改这一层映射，不需要改 18 个 stage 模型。

---

### 11.7 applySuperBlock 是调整了之前的哪个评分吗？

`applySuperBlock`（`StageCostModels.cpp:178`）调整的是**单个 Stage 在某一个
SIMT SuperBlock implementation 下的周期数**，不是某个旧 aggregate 评分。

调用点在 `StageCostEvaluator::evaluate`（`StageCostModels.cpp:899`）：

```cpp
cost.totalCycles = applySuperBlock(
    stage, resources, implementation, profile,
    (*model)->estimate(context, implementation, resources));
```

也就是：

1. 先用对应 Stage 模型（例如 `SIMTRecurrenceStageCostModel`）算出基准
   `stageCycles`；
2. 再在 SIMT 且 `superblockFactor > 1` 时，用 `applySuperBlock` 修正这个
   Stage 的 `totalCycles`；
3. 修正后的 `StageImplementationCost` 进入 `StageCostTable`；
4. 最后 `solveStageRoutes` 基于这些修正后的 stage 成本做 DP。

它调整的是 **Stage 级 SIMT-F2/F4 候选成本**，影响的是最终 route DP 的输入，
而不是独立地在 route 之后再做一次全局加分。

公式本质：

```
latencySensitive = N×(load+store+shuffle+divergence)
cost' = max(issueFloor, base - latencySensitive + latencySensitive/effectiveFactor + pressure)
       + persistentStatePressure
```

因此它只把“可被多 logical program 隐藏的延迟”除以 `effectiveFactor`，
不能把算术/控制/同步也除以 factor。

---

### 11.8 solveStageRoutes 是在干什么，解决什么核心问题？核心法则是什么？

`solveStageRoutes`（`StageRouteCostModel.cpp:417`）解决的核心问题是：

> 给定每个 Stage 在 SIMD / SIMT-F1 / SIMT-F2 / SIMT-F4 下的成本和
> SIMD<->SIMT 边界成本，选择一个全局可物化的 kernel route：
> AllSIMD / AllSIMTOnly / Mixed，并让总周期最小。

它不是简单地对每个 Stage 取局部最优，因为它要同时满足：

1. **route class 一致性**
   模式序列一旦从 AllSIMD/AllSIMT 进入 Mixed，就不能再变回纯模式。

2. **SuperBlock factor 一致性**
   一个 kernel 中所有 SIMT Stage 必须使用同一个 `routeSuperblockFactor`。
   不允许前半段 F2、后半段 F4。

3. **Mixed 只允许 local materializable SIMT**
   如果 mixed route 中出现一个非 local SIMT 的 SIMT Stage，该 route 非法；
   因为 pure-SIMT 和 local mixed scope 是两种不同的物化路径。

4. **Mixed 要付真实 scope 边界成本**
   对 local SIMT Stage，在进入 Mixed 类时将其成本替换为
   `mixedEquivalentStageCost`：
   - 原 SIMT stage 成本；
   - 每个 anchor 一对 SIMD->SIMT + SIMT->SIMD 固定切换；
   - captured/escaping tensor 的 UB 双向搬运成本。

DP 状态：

```
State[exitMode][routeClass] -> map<superblockFactor, PartialRoute>
```

按 Stage 顺序扩展，保留每个 `(exitMode, routeClass, factor)` 的最优前缀，
最后分别从 AllSIMD / AllSIMT / Mixed 三类中取最优。

**核心法则**：route 成本 = 每个 Stage 在统一 route 约束下的实现成本之和 +
模式切换/数据搬运成本；约束必须保证“选出来的 route 一定能被物化”。

---

### 11.9 buildSelectedMixedAnchorPlan 这里怎么又在构建 anchor plan 了，和之前的区别是什么，有必要吗？

有必要。它构建的不是“完整 anchor plan”，而是“被选中的 mixed route 真正需要物化的
anchor 子集”。

区别：

| | 之前的完整 `SimtAnchorPlan` | `buildSelectedMixedAnchorPlan` 产物 |
|---|---|---|
| 来源 | `buildMixedSimtAnchorPlan(module, compileOn91095)` | `report.stageModel` 中 mixed route 的 SIMT Stage |
| 内容 | 所有识别出的、可物化的 anchor | 只保留 mixed route 中 implementation == SIMT 的 Stage 所拥有的 anchor |
| 用途 | feature 提取、全量评分、lowerability、materializer 的“全集” | 实际 `materializeSimtAnchorPlan` 只物化这个子集 |

`buildSelectedMixedAnchorPlan`（`SelectSimdSimtCostModel.cpp:81`）的流程：

1. 如果 `stageModel.mixed` 不合法或 implementation 数量对不上，返回空 plan；
2. 遍历每个 stage 的 mixed-route implementation；
3. 只收集 `implementation.mode == SIMT` 的 stage 所拥有的 `simtAnchorIndices`；
4. 从完整 plan 中取出这些 anchor，去重后作为 selected plan。

**必要性**：评分时我们允许所有 anchor 参与分析，但最终 mixed route 可能只选择
其中一部分 SIMT Stage。如果 materializer 把完整 anchor plan 全部搬进 scope，
就会物化出比评分更多的 SIMT 区域，导致“分数描述的路由”和“实际执行的路由”不一致。
这正是第 10.3 节强调的不变量：**分数与物化同源**。
`buildSelectedMixedAnchorPlan` 就是把 DP 选出的路转换回物化指令的唯一桥梁。

---

## 附：关键代码位置速查

| 想找什么 | 文件:行 |
|---|---|
| 决策 pass 入口 | costmodel/lib/AscendModel/RouteModel/Transforms/SelectSimdSimtCostModel.cpp:125 |
| anchor 识别/分类 | costmodel/lib/AscendModel/RouteModel/SimtAnchorAnalysis.cpp:499 |
| anchor plan 构建（不重叠 walk） | SimtAnchorAnalysis.cpp:719 |
| solve_tril scope 操作收集 | SimtAnchorAnalysis.cpp:387 |
| feature 提取主 walk | SimdSimtCostModel.cpp:1834（walk 起点 1966） |
| stage 模型入口（evaluateStageModel） | SimdSimtCostModel.cpp:1248 |
| HardwareProfile 组装 | SimdSimtCostModel.cpp:1163 |
| domain 判定 | StagePartitioner.cpp:1477 |
| root→Phase 状态机 | StagePartitioner.cpp:917 |
| 三套 Phase/Stage 模板 | StagePartitioner.cpp:562/662/714 |
| operation ownership 附着 | StagePartitioner.cpp:1098 |
| stage live 值/scope 流量 | StagePartitioner.cpp:1249/1280 |
| Kind 分类器 | StagePartitioner.cpp:1697 |
| 模式合法性 | StagePartitioner.cpp:1977 |
| 守恒验证 | StagePartitioner.cpp:1864/1913 |
| workload→资源映射 | StageCostModels.cpp:72/125 |
| 18 个具体模型 | StageCostModels.cpp:224-555 |
| SuperBlock 公式 | StageCostModels.cpp:178 |
| Stage 成本表计算 | StageCostModels.cpp:802 |
| route DP | StageRouteCostModel.cpp:417 |
| mixed 等价成本 | StageRouteCostModel.cpp:38 |
| scope 物化 | costmodel/lib/.../Transforms/MaterializeSimtScopes.cpp:151 |
| V1 调度循环生成 | lib/AutoBlockifyV1/SIMTAutoBlockifyV1.cpp:96 |
| V1 SuperBlock refine | SIMTAutoBlockifyV1.cpp:252 |
| layout merge | lib/TritonToLinalg/TTIRLayoutMergePass.cpp:31 |
| 行提升重写 | lib/TritonToLinalg/RowCoalescing.cpp:213 |
| compiler.py 决策翻译 | backend/compiler.py `_apply_cpp_simd_simt_decision` |