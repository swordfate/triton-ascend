# AutoScope 的 Cost Model（SIMD/SIMT 路由决策模型）—— 从第一性原理到实现

> 本文讲解 `triton-ascend` 中 **autoscope 的 cost model**（SIMD/SIMT Route Model），即编译期自动决定 kernel 用 SIMD、SIMT 还是混合方式执行的解析代价模型。
>
> 注意区分：它 **不是** autotuner 使用的那个 costmodel（后者位于 `third_party/ascend/backend/runtime/costmodel_runtime.py`，通过 `-ascend-perf-model` 管线做绝对延迟预测，供 autotuner 挑 config）。本文的模型做的是另一件事：**在同一个 kernel 的三种执行路线之间做相对排序**。

---

## 0. 阅读地图：这份代码是什么、在哪里、不是什么

### 0.1 核心文件清单

所有核心逻辑都在 `third_party/ascend/costmodel/lib/AscendModel/RouteModel/` 下：

| 文件 | 行数 | 职责 |
|---|---:|---|
| `SimdSimtCostModel.{h,cpp}` | 397 + 2905 | 特征提取、profile 加载、候选打分总入口 |
| `SimtAnchorAnalysis.{h,cpp}` | 169 + 789 | SIMT 锚点识别（哪些 op 值得搬进 SIMT） |
| `StagePartitioner.{h,cpp}` | 158 + 2109 | Phase/Stage 划分六步流水线 |
| `StageCostModels.{h,cpp}` | 238 + 915 | 每个 Stage 的解析代价模型树 + 注册表 |
| `StageRouteCostModel.{h,cpp}` | 237 + 585 | 路由动态规划求解器 |
| `Transforms/SelectSimdSimtCostModel.cpp` | 269 | 在线选择 pass（唯一决策入口） |
| `Transforms/MaterializeSimtScopes.cpp` | 236 | 把选中的锚点物化成 `scope.scope<simt>` 区域 |
| `../Profile/MicrobenchmarkProfile.{h,cpp}` | — | 共享微基准测量目录 |

配套数据：

| 文件 | 内容 |
|---|---|
| `costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json` | 选择模型 profile（v17，schema 10）：结构惩罚系数、SIMD/SIMT 吞吐、SuperBlock、scope 边界速率 |
| `costmodel/profiles/microbench/ascend_davidv100_v1.json` | 共享微基准目录：每条测量带 unit / cycle_domain / confidence |

接线侧：

| 文件 | 内容 |
|---|---|
| `backend/compiler.py` | `ttir_to_linalg` 里调度 C++ pass、读回决策属性、翻译成 metadata |
| `lib/TritonToLinalg/*` 等 | 下游 lowering 通过 `shouldUseSimtTemplate()` 读契约，真正把 op 降到 SIMT |

### 0.2 它不是什么（与 autotuner costmodel 的分界）

| | autotuner costmodel | autoscope Route Model（本文） |
|---|---|---|
| 入口 | `costmodel_runtime.py::run_costmodel` → C-API `run_costmodel_inproc` | MLIR pass `select-simd-simt-costmodel` |
| 管线选项 | `-ascend-perf-model arg-bindings=...` | pass 参数（profile 路径 / num_warps / 目标…） |
| 输出 | 绝对延迟估计 `Estimated Time: X us` | 三条路线的相对分数 + 决策 + JSON 报告 |
| 使用者 | autotuner 给候选 config 排序 | 编译器自己给执行范式排序 |
| 时机 | 编译期多次调用（每个 config 一次） | 编译期恰好一次（TTIR 上） |

两者共享同一份微基准测量数据（`MicrobenchmarkProfile`），但公式体系完全独立。

---

## 1. 问题从哪里来：为什么需要这个 cost model

### 1.1 硬件背景：一颗向量核上的两种执行范式

Ascend 向量核可以按两种范式执行同一段张量计算：

- **SIMD / Cube**：tensor 抽象。一条向量指令处理整个 tile（如 2048-bit 向量），矩阵乘走 Cube 单元。这是 Triton 原生 lowering 的目标——`tt.load/tt.dot` 直接映射到 MTE/Cube 指令。
- **SIMT**：标量多线程抽象。程序被组织成逻辑 warp（本目标 warp size = 32），每条指令由标量 ALU 在每个 lane 上各算一份，通过 shuffle 做跨 lane 归约。

两种范式各有擅长：

| 场景 | 为什么 SIMD 好 | 为什么 SIMT 好 |
|---|---|---|
| 规则连续访存 + 密集计算 | 一条 MTE/Cube 指令干一大片活 | 每线程逐元素太浪费 |
| 地址依赖其他数据的访存（gather、间接寻址） | 必须拆成逐元素事务，向量单元空转 | 天然就是每线程一次标量 load |
| 原子操作 / 直方图 | 后端没有好的向量化模板 | 标量原子语义直接对应 |
| 串行递推（如三角求解的行间依赖） | 向量化被依赖链卡死，SIMD 只能干等 | warp 内交错多个独立递推组可隐藏延迟 |
| 小规模 matmul（Cube 利用率极低） | Cube 启动开销吃掉全部收益 | 标量 FMA 流水反而干净 |

### 1.2 手写 scope 的困境 → autoscope

后端提供了显式标注手段：把一段 op 包进 `scope.scope {vector_mode="simt"}` 区域，区域内走标量管线。但要求用户手写有两个致命问题：

1. **需要专家知识**：哪种模式快取决于硬件微架构参数（向量宽度、warp 吞吐、MTE 带宽、Cube 启动周期……），换一代芯片结论就变；
2. **粒度难拿捏**：整 kernel 全 SIMT 往往亏（规则部分被拖慢），最优常常是"大部分 SIMD + 少数几个关键循环 SIMT"的混合体。

所以目标是 **autoscope**：编译器在编译期自动完成三选一——

```
all_simd        整个 kernel 按 Triton 原生 SIMD 路径降级
all_simt_only   整个 kernel 走纯标量管线（可叠加 SuperBlock F1/F2/F4）
mixed_simd_simt 大部分 SIMD，少数被识别为"SIMT 有利"的操作包进局部 simt scope
backend_default 不表态，沿用历史全局路由规则
```

### 1.3 为什么必须是编译期的解析模型，而不是实测

一个自然的反问：为什么不把三种路线各编一份、上硬件跑一遍再挑？原因有三层：

1. **编译期只有一次机会**：JIT 编译发生在用户进程里，编译产物要立刻用；不可能为每个 kernel 都跑三遍基准。
2. **编译机 ≠ 执行机**：交叉编译场景下目标板不在手边；即使在本机，autoscope 发生在 TTIR 层，此时连可执行的机器码都还没有。
3. **决策空间是连续谱**：mixed 还牵扯"哪些操作搬过去""SuperBlock 因子取几"，实测枚举爆炸。

于是只能走解析模型（analytical model）路线。但要清醒地给它定位——profile JSON 里写得非常诚实：

```json
"score_unit": "system_cycle_selection_score",
// 报告里同时声明：
"excludes": ["host_launch", "grid_wave_count"]
```

**它产出的是"同一 kernel 三种跑法的相对排序分数"，不是端到端运行时间预测。** 排序只需要单调性正确，不需要绝对值精确——这大幅放宽了对校准精度的要求，也让"没测过的项宁可保守高估"成为合理策略。

### 1.4 一个必须先建立的直觉：为什么这个选择是"值得建模"的困难问题

看两条关键吞吐（来自微基准实测）：

```
simd.f32.add.throughput = 3.3   vector_instruction/system_cycle   (confidence: high)
simt.f32.add.throughput = 141.0 scalar_op/system_cycle           (confidence: high)
```

SIMD 向量宽度 2048 bit ÷ 32 bit ≈ 64 lane，所以 SIMD 有效元素吞吐 ≈ 3.3 × 64 ≈ 211 elem/cycle，而 SIMT 是 141 elem/cycle——**两者在同一量级**。这意味着：

- 对纯计算 kernel，两种范式差距不大，选错的代价可控；
- 真正拉开差距的是**访存模式**（连续 vs 间接）、**结构**（依赖链、分支）、**启动/边界开销**；
- 所以模型的核心工作量在于精确刻画这些结构性因素，而不是把乘加吞吐调到小数点后两位。

---

---

## 2. 第一性原理推导：从"要选路由"一步步推出整个架构

这一节按"我们有了 X，还缺 Y，所以要 Z"的顺序把整个系统推导一遍。读完你会发现每个文件、每个函数都是这条推导链上必然出现的一环。

### 推导 0：要选路由，就必须比较 —— 分数从哪来？

要在三条路线间挑一条，必须给每条路线算出一个**可比较的标量分数**。而任何执行时间的解析估计都遵循同一个基本公式：

```
总周期 ≈ Σ_over_work( 该项工作的量 × 该项工作在该硬件模式下的单位耗时 )
              └────── 工作量账本 ──────┘   └──────── 价目表 ────────┘
```

所以这个系统**天然需要两个输入**：

| 输入 | 内容 | 对应代码 |
|---|---|---|
| 工作量账本 | kernel 做了多少次加法、多少字节访存、多少步 shuffle…… | `analyzeSimdSimtFeatures()` 产出的 `SimdSimtFeatureSummary` |
| 硬件价目表 | SIMD 向量宽度、每周期几条向量指令、MTE 带宽、warp 吞吐、Cube 启动开销…… | `loadCandidateProfile()` 解析的 `CandidateProfile` |

这两个输入一个来自 IR、一个来自校准数据，互相独立演化——这就是后面所有代码组织的两条主线。

### 推导 1：工作量账本从哪来？—— 在 TTIR 上做特征提取

第一个决策点：在编译器哪一层 IR 上统计工作量？

- 源码层（Triton python）：太远，shape/循环都还没实例化；
- LLVM IR：太低，已经过了大量变换，结构噪声大；
- **TTIR**：正合适。它是 Triton 的语义层——op 种类有限（load/store/dot/reduce/gather…）、tensor shape 和 mask 显式可见、循环是结构化的 `scf.for`。特征提取函数 `analyzeSimdSimtFeatures` 的注释也明确写了 *"Analyze generic TTIR without depending on Triton C++ op classes"*——它只靠 op 名字符串识别，不依赖具体 dialect C++ 类，因此对任意 TTIR 都能跑。

第二个决策点更微妙：**在哪份 TTIR 上提取？**

编译管线里，TTIR 在进入本模型之前会先经过两个变换：
1. `_run_ttir_layout_merge`：内存布局合并（改变访存形态）；
2. AutoBlockify V1：物理核调度循环包装（改变程序调度结构）。

如果用变换前的 TTIR 统计，算出来的就是"一个不会真正运行的程序"的工作量。所以模型坚持**先跑 layout merge，再计费**，并且通过读取 IR 标记来确认前置变换确实发生过：

```cpp
// SimdSimtCostModel.cpp:1843
features.ttirLayoutMergeApplied = module->hasAttr("ta.ttir_layout_merge.applied");
features.autoBlockifyV1Applied  = /* 扫描 ta.auto_blockify_v1* 属性 */;
```

这解释了 compiler.py 里调用顺序为什么是 `_run_ttir_layout_merge` 在前、`_run_cpp_simd_simt_costmodel` 在后。

### 推导 2：全 kernel 一个总分不够 —— SIMT 只在特定模式有优势

有了账本和价目表，最朴素的实现是：整本账按 SIMD 算一遍得分 A，按 SIMT 算一遍得分 S，取小者。这就是模型的 v1 形态。

但它立刻暴露一个问题：**SIMT 并不是对所有操作都慢**。规则密集计算上 SIMT 明显吃亏；但 gather / atomic / histogram / 一维 cumsum / 三角递推这类模式，SIMT 反而可能是唯一高效路径。所以正确的问题不是"全 kernel 用哪个"，而是：

> kernel 里有没有那么几个局部模式，值得单独搬进 SIMT？搬过去之后整体是否更快？

这就引出 **anchor（锚点）** 概念：在 TTIR 上做结构化模式匹配，找出"SIMT 有利且后端真的能落地"的操作集合。六类机制（`SimtAnchorKind`）：

| kind | 触发 op | 典型来源 |
|---|---|---|
| `DirectGather` | `tt.gather` | 直接索引重排 |
| `LoadedIndexDependentMemory` | 地址回溯可达 `tt.load/tt.gather` 的 `tt.load/tt.store` | embedding 查表等间接访存 |
| `Histogram` | `tt.histogram` | 直方图 |
| `PlainOneDimensionalCumsum` | 单一 addf combine 的一维 `tt.scan` | 前缀和 |
| `TensorAtomic` | `tt.atomic_rmw / tt.atomic_cas` | 张量原子更新 |
| `TriangularSolveLoop` | 特定形态的 `scf.for` | solve_tril 类三角递推 |

注意区分两个概念（这是代码里刻意分开的）：

- **recognized（识别到了机制）**：TTIR 里存在上述模式；
- **materializable（当前目标能落地）**：`compileOn91095 == true` 且该锚点的 mixed 合法性为 Native。

识别了但不能落地的锚点只进报告不参与选择——**合法性（legality）与代价（cost）从头到尾是两条独立的线**。

### 推导 3："打分的工作量" 必须等于 "落地的工作量" —— 共享一份不可变计划

这里有个很容易踩的坑：如果特征提取自己扫一遍找锚点，物化阶段又扫一遍，两次扫描的结果可能不一致——比如物化时因为某个 op 不满足条件少包了一个循环，那 mixed 分数就凭空便宜了。

解决方案是一个强约束（`SimtAnchorAnalysis.h` 注释原文）：

> Keeping Operation pointers here prevents those stages from independently rediscovering different anchor sets.

即 `buildMixedSimtAnchorPlan()` 只跑一次，产出**不可变的 `SimtAnchorPlan`**（含每个锚点的精确 op 集合），然后：

```
buildMixedSimtAnchorPlan(module)
        │ 同一份 plan
        ├──► analyzeSimdSimtFeatures(module, plan)      ← 按 plan 圈定锚点工作量
        ├──► estimateSimdSimtCandidates(..., &plan)     ← 打分
        └──► materializeSimtAnchorPlan(module, 子集)     ← 物化
```

三处消费者看到的锚点集合逐字节一致。"打分的 = 后端实际降级的"由构造保证，而不是靠约定。

### 推导 4：mixed 怎么定价？—— 把 kernel 切成串行的单模式 Stage

有了锚点，mixed 的朴素算法是：`mixed ≈ (kernel 总量 − 锚点量) 按 SIMD 计价 + 锚点量按 SIMT 计价 + 边界成本`。这正是代码里保留至今的**聚合公式 fallback 路径**（当 Stage 模型不适用时的兜底）。

但聚合公式有四个无法表达的效应：

1. **每段各自的合法性**：某段循环带递推依赖，SIMD 实现可能根本不可行，只能 SIMT；
2. **流水线重叠**：SIMD 的连续访存可以与计算 overlap，roofline 取 max；SIMT 当前降级是串行指令流，只能求和——这种差异必须按段建模；
3. **边界成本的精确计量**：SIMD/SIMT 寄存器堆不同，数据要经 UB 中转；搬运的字节数取决于**恰好哪些值跨过 scope 边界**，聚合公式算不准；
4. **SuperBlock 因子选择**：纯 SIMT 可以 F2/F4 批处理，因子影响延迟隐藏与寄存器压力，需要按 kernel 统一决策。

于是第二版架构登场：**把 kernel 表示成一串串行的逻辑 Stage，每个 Stage 整体属于一种模式，绝无混合 Stage**（`StageRouteCostModel.h` 开头注释）。这样 mixed route 就是"Stage 序列的模式指派"，上面四个问题全部有了着色板。

切分又分两层（`StagePartitioner.h`）：
- **Phase**：算法级串行边界（如 solve_tril 的 head → load → recurrence → merge&store），由单调状态机保证 Phase 连续；
- **Stage**：单一主导资源语义的实现段（标量索引生成 / 连续 tile 访存 / 循环携带递推 / Cube roofline…），Phase 内再细分。

### 推导 5：单个 Stage 的两种实现各多少钱？—— workload→resource 映射 + per-kind 模型树

现在问题缩小为：给定一个 Stage，它的 SIMD 实现多少周期、SIMT 实现多少周期？

设计上做了一个关键解耦（`StageWorkload` 注释：*"Values are logical elements/bytes, not mode-specific instructions or cycles"*）：

```
StageWorkload（逻辑工作量，与模式无关）
   │ mapSIMDWorkload()                    mapSIMTWorkload()
   ▼                                       ▼
StageResourceCycles（SIMD 资源周期）      StageResourceCycles（SIMT 资源周期）
   │                                       │
   └──────── per-kind StageCostModel::estimate() ────────► totalCycles
             （决定这些资源周期如何组合：串行 / 重叠 / 关键路径）
```

为什么要 per-kind 模型树而不是一个万能公式？因为不同结构的组合规律不同：
- 连续访存 Stage 的 load 可与其它工作并行 → body = 其它 + max(load, store, issue)；
- 循环携带递推 Stage 被依赖链卡死 → 走 criticalPath，但多个独立递推组可在 warp 组间交错；
- 归约 Stage 的树深是依赖链，issue 是吞吐下限 → max(criticalPath, issue floor)……

20 种 `StageCostModelKind` × 2 模式 = 注册表里的模型矩阵，`StageCostModelRegistry::verifyComplete()` 保证每种组合都有模型，漏注册直接报错。

### 推导 6：Stage 序列怎么组装成整 kernel 路由？—— 动态规划 + 边界成本

每个 Stage 有若干合法实现候选（SIMD-F1，以及合法时的 SIMT-F1/F2/F4），一条完整路由就是给每个 Stage 挑一个实现，外加约束：

- 整条 route 上所有 SIMT Stage 的 SuperBlock 因子必须一致（它是 kernel 级调度）；
- mixed route 要求所有 SIMT Stage 都 local-materializable（能被真实包成局部 scope）；
- mixed 的代价不是简单相加——每个物理 scope 要付 UB 双向搬运 + 定向切换。

这是一个带状态的最短路径问题，状态空间为 `(出口模式, 路线类别, SuperBlock 因子)`——`solveStageRoutes()` 的 DP。注释解释了为什么不能砍掉因子维度：

> Collapsing the factor dimension can discard a slightly slower F1 prefix that becomes globally optimal, or worse, combine F1 and F4 Stage costs into an unrealizable F4 kernel.

DP 结束后在三个类别（AllSIMD / AllSIMT / Mixed）里各取最优，得到三条候选路线的总周期。

### 推导 7：决策如何落地？—— 物化器与选择器共享同一份计划

分数只是建议，最终要让 IR 变成事实。落地分两种情况：

- **all_simt_only**：整 kernel 已有或将被放入一个 void SIMT scope，走纯标量降级管线；
- **mixed**：把选中 Stage 所拥有的锚点，按计划精确地包进 `scope.scope {vector_mode="simt"}` 区域——SSA 安全地搬运操作、收集"逃逸值"穿过 `scope.return`、替换外部使用。

关键工程约束：**选择 pass 在同一次调用内就地完成物化**（`SelectSimdSimtCostModel.cpp` 注释：*"Selector and Materializer consume the same immutable anchor plan in one pass invocation. No per-operation marker is persisted in TTIR."*）——不在 IR 上留中间标记，就没有二次失配的机会。物化之后还有一个兼容校验 pass（`materialize-simt-scopes`），若声明了 mixed 却找不到局部 scope 就让编译失败，宁可报错也不静默错路。

下游真正的降级（TritonToLinalg 等）则通过 `shouldUseSimtTemplate(op, legacyForce)` 这个统一查询口读取决策：显式 simd scope 一票否决；模型接管时只有局部 simt scope 能启用 SIMT 模板。

### 推导 8：怎么让人信服、可调试、可复现？—— 报告与版本化

解析模型最大的风险是"黑盒猜错"。所以整套系统把**可解释性当作一等公民**：

- 决策连同三路分数、逐项 breakdown（compute/memory/shuffle/predicate/dot/structural penalty）、Stage 明细、每条 route 的分段周期，全部序列化成 JSON，挂在 module attr `ascend.simt_costmodel.report_json` 上并可追加写入文件；
- profile 版本号在代码里钉死（`david-v100-simd-simt-20260820-v17`），schema 校验失败直接报错；profile 内容算 sha256（还专门实现了与 Python `json.dumps(sort_keys=True)` 逐字节兼容的序列化，保证历史 Python 模型与新 C++ 模型哈希一致）；
- 每个 profile 数值必须带 unit 与 cycle_domain（SYS_CNT），引用微基准时做域检查，防止把"每仿真周期"当成"每系统周期"。

---

## 3. 端到端调用链全景图

### 3.1 Python 编译管线入口

```
triton.compile(...)
  └─ backend/compiler.py :: ttir_to_linalg(mod, metadata, opt)          :346
       ├─ [仅当 metadata.compile_mode=="simd_simt" 且 opt.auto_simt_scope_mode!="off"]
       ├─ _run_ttir_layout_merge(mod, metadata)                         :246
       │    └─ pass: ttir-layout-merge  → 写入 ta.ttir_layout_merge.applied /
       │                                  hacc.coalesce_factor / coalesce_axis
       ├─ _resolve_auto_blockify_v1_policy(ttir, metadata, opt)         :262
       ├─ _run_cpp_simd_simt_costmodel(mod, metadata, opt)              :203
       │    ├─ profile = _costmodel_profiles_dir()/simd_simt/
       │    │            david_v100_simd_simt_v1.json                   :154/:209
       │    ├─ whole_kernel_superblock_materializable =
       │    │     opt.compile_on_910_95 and opt.enable_auto_blockify != False
       │    ├─ pm.add(select-simd-simt-costmodel){mode, profile, target,
       │    │        num_warps, compileOn91095, wholeKernelSB,
       │    │        scopeSB=False(未来 ScopeSuperBlock 才放开), reportFile}
       │    ├─ pm.add(materialize-simt-scopes)   ← 兼容校验 pass
       │    ├─ pm.run(...)                       ← 进入 C++（见 3.2）
       │    ├─ 读回 module attrs:
       │    │     ascend.simt_costmodel.report_json / effective /
       │    │     superblock_factor（非法值一律视为 1）
       │    └─ _apply_cpp_simd_simt_decision(metadata, effective, ...)  :176
       │          all_simd       → compile_mode="simd", parallel_mode="simd",
       │                          auto_blockify_v1_enabled=False
       │          mixed_simd_simt→ auto_simt_requested_kind="mixed_simd_simt"
       │          all_simt_only  → 若 whole-kernel SB 可物化则启用 AutoBlockify V1
       │
       ├─ [effective == all_simt_only 或已存在整体 void simt scope]      :353
       │    metadata.force_simt_only=True; parallel_mode="simt";
       │    shared_mem_dynamic_size=122880;
       │    ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)   ← C++ helper
       │    （可选）TA AutoBlockify V1 refine/run(selected_superblock_factor)
       │    return str(mod)   ← 纯 SIMT 路径直接返回 TTIR 给 bishengir，
       │                        不再经过 linalg 降级！
       │
       └─ [all_simd / mixed] 带着（可能已被物化的）scope 继续正常
            ttir → linalg 降级；下游 pass 通过 shouldUseSimtTemplate 读契约
```

### 3.2 C++ 选择 pass 主流程

`SelectSimdSimtCostModel.cpp :: runOnOperation()`（:125）：

```
① clearPreviousSelection(module)                      :69   清掉上次决策属性
② options{profilePath, actualTarget, numWarps,
         compileOn91095, wholeKernelSB, scopeSB}      :130
③ anchorPlan = buildMixedSimtAnchorPlan(module,
                                        compileOn91095)     [SimtAnchorAnalysis.cpp:719]
④ report = analyzeSimdSimtCandidates(module, anchorPlan,
                                     options)               [SimdSimtCostModel.cpp:2897]
⑤ recommended = stringify(report.decision)
⑥ action support 检查                                 :174–212
⑦ effective = (autoMode && supported) ? recommended
                                      : "backend_default"
⑧ 写 module attrs（recommended/effective/scores/
   superblock_factor）                                 :224
⑨ if effective == mixed:
      selectedPlan = buildSelectedMixedAnchorPlan(
                        report.stageModel, anchorPlan)   :80  ← 只取 DP 选中为 SIMT
                                                          的 Stage 所拥有的锚点
      materializeSimtAnchorPlan(module, selectedPlan)    :241 [MaterializeSimtScopes.cpp:151]
⑩ reportJSON 追加写文件 + 存 attr                      :247
```

### 3.3 打分内部组件协作图

`analyzeSimdSimtCandidates → estimateSimdSimtCandidatesImpl`（`SimdSimtCostModel.cpp:2394`）内部：

```
loadCandidateProfile(profilePath)                     :695
   └─ MicrobenchmarkProfile::loadFromFile(共享测量目录)

evaluateSimtApplicability(features, targetSupported)  :1289

evaluateStageModel(features, profile, numWarps,
                   ..., module, anchorPlan)           :1248
   ├─ StagePartitioner.partition(module, anchorPlan,
   │                             features, opts)       [StagePartitioner.cpp:2047]
   │    ① ProgramStructureAnalysis.analyze            :1402  语义根 + 复合锚点顺序归一
   │    ② PhaseBoundaryAnalysis.analyze               :1505
   │         特征域判定                                :1477（三域互斥）
   │         assignRootPhaseIds（单调状态机+连续性）     :917
   │    ③ StageBoundaryAnalysis.analyze               :1525
   │         partitionTriangular/Rowwise/Dot          :562/:662/:714（模板扣账）
   │         attachCompleteOperationOwnership          :1098（每 root 恰好一段）
   │         attachExactAnchorOwnership                :823（锚点↔Stage 绑定）
   │         deriveStageLiveValues                     :1249
   │         deriveLocalSimtScopeTraffic               :1280（scope 进出字节）
   │    ④ StageWorkloadAnalysis.analyze               :1885（动态工作量÷迭代数）
   │    ⑤ StageFeatureAnalysis.analyze                :1565（从 op 图重建特征）
   │    ⑥ StageKindClassifier.analyze                 :1697（requires_split / 派生 kind）
   │    ⑦ StageModeLegalityAnalysis.analyze           :1977（F1/F2/F4 合法因子）
   │    ⑧ StagePartitionVerifier.verify               :1913（所有权守恒）
   ├─ buildStageHardwareProfile(profile, numWarps)    :1162
   ├─ StageCostEvaluator.evaluate(partition, hw)       [StageCostModels.cpp:802]
   │    ├─ implementations = SIMD-F1 + SIMT-F{legal}  :871
   │    ├─ mapSIMDWorkload / mapSIMTWorkload           :72/:125
   │    ├─ registry.lookup(mode,kind)->estimate        :748（serialBody :30 等）
   │    └─ applySuperBlock                             :178
   └─ solveStageRoutes(costTable, transition)          [StageRouteCostModel.cpp:417]
        ├─ mixedEquivalentStageCost（UB 双向搬运）       :38
        └─ DP over [exitMode][routeClass][superblockFactor] :433

[stage 模型未命中 → 聚合公式 fallback]                 :2481–2878

chooseBest(legal candidates)                          :1313/:1331
```

一句话总结这张图：**anchor 计划定义"谁可以搬"；特征账本记录"干了多少活"；profile 提供单价；StagePartitioner 把 kernel 切成段；StageCostModels 给每段算三种候选价；DP 把段拼成三条总价；selector 挑最小并把选中部分就地物化。**

---

## 4. 组件精读：每一块为什么长这样、又是怎么实现的

以下逐个组件展开。每个组件都按同样的节奏讲：**先说它解决推导链上的哪个问题（why），再看代码怎么实现（how）**。

### 4.1 `SimtSelection.h` —— 决策的表达契约

**为什么需要它？**

推导 7 说决策要落地到 IR。但"落地"涉及三方：selector 写属性、materializer 写 scope、下游 lowering 读属性。如果三方的拼写或查找规则不一致，整个系统静默失效。所以需要一个只有头文件的小模块，把**契约的每一个字节**钉死：

```cpp
inline constexpr llvm::StringLiteral kEffectiveExecutionAttr =
    "ascend.simt_costmodel.effective";      // 决策挂在哪
inline constexpr llvm::StringLiteral kVectorModeAttr = "vector_mode";
inline constexpr llvm::StringLiteral kLegacyVectorModeAttr = "vector_type";
```

四个关键查询函数构成读取侧的完整语义：

```cpp
// 从 op 向外找最近的决策属性——函数级决策可以覆盖模块级默认值
StringAttr getEffectiveExecution(Operation *op);

// backend_default / 无决策 = 不接管，沿用历史全局路由；
// 只有具体模型决策才抑制 legacy 全局 force 标志
bool isModelControlled(Operation *op);

bool isMixedModelDecision(Operation *op);

// 单个 op 是否允许走 SIMT 模板：
//   显式 simd scope 一票否决 →
//   模型接管时只认局部 simt scope →
//   否则 legacy force || 局部 scope
bool shouldUseSimtTemplate(Operation *op, bool legacyForceSimt);
```

`shouldUseSimtTemplate` 的三段逻辑值得背下来，它就是"谁说了算"的优先级裁决：

```cpp
if (hasEnclosingVectorMode(op, "simd")) return false;        // 用户显式 simd 最高
const bool locallySelected = hasEnclosingVectorMode(op, "simt");
if (isModelControlled(op)) return locallySelected;           // 模型接管：只认局部 scope
return legacyForceSimt || locallySelected;                   // 未接管：保持历史行为
```

此外还有两个纯 SIMT 落地用的 helper：

- `findWholeBodyVoidSimtScope(root)`：判断一个函数体是否恰好是"常量 + 一个 void simt scope"的结构（替代了早期 Python 端对 TTIR 文本的正则猜测）；
- `inlineVoidSimtScopesForPureSimt(root)`：把 void scope 的内容搬出来并删除壳——因为纯 SIMT 编译管线不认识 scope 壳。

**实现要点**：全部是 inline 小函数、无状态；最近作用域优先（nested scope 时 nearest wins）。这种"契约即头文件"的写法保证了 selector/lowering/materializer 三方永远引用同一份语义。

### 4.2 `SimtAnchorAnalysis` —— 找出"值得用 SIMT 的那几刀"

**为什么单独一个组件？**

mixed 候选的前提是知道哪些 op 能搬进 SIMT。这个判断必须满足三个要求：(a) 结构化——只看 IR 形态，不看 workload 名字；(b) 可复用——特征提取与物化必须用同一份结果（推导 3）；(c) 自带合法性——每种机制要声明三条候选路线各自的可用性。`SimtAnchorDescriptor` 就是这三要求的产物：

```cpp
struct SimtAnchorDescriptor {
  Operation *operation;
  SmallVector<Operation*> scopeOperations;  // 将被移进同一个 scope 的精确 op 集合
  Operation *scopeInsertionPoint;           // scope 插入位置（可能与首 op 不同！）
  SimtAnchorKind kind;
  SimtAnchorFacts facts;                    // 各机制的结构化事实（variant）
  CandidateLowerability lowerability;       // 三条路线各自的可用性 + 理由
  bool materializable;
};
```

#### 4.2.1 合法性状态机

```cpp
enum class CandidateLoweringStatus {
  Unsupported,          // 这条路线对该模式根本不可行
  Native,               // 直接可落地
  BackendConditional,   // TTIR 层无障碍，取决于后端管线是否被选中
  AliasesMixed,         // 该符号在后端会与 mixed 模板撞名，不能作为独立路线
};
```

每条候选（allSimd / allSimtOnly / mixed）独立记录状态和理由列表。整 kernel 级别再合成一次：

```cpp
static CandidateLoweringStatus combineWholeKernelStatus(a, b) {
  // 取最差者：Native < BackendConditional < AliasesMixed < Unsupported
}
```

#### 4.2.2 六类锚点的识别逻辑

`analyzeAnchor(op)` 按 op 名分派，这里挑三个有代表性的讲透：

**(a) LoadedIndexDependentMemory —— 真·数据依赖测试**

老版本用"指针张量秩 >1"当代理指标（`laneDependentPointerOps` 至今保留只为报告兼容）。新实现做真正的 SSA 回溯：

```cpp
bool pointerDependsOnLoadedIndex(Operation *memoryOp) {
  worklist = {memoryOp->getOperand(0)};         // 指针操作数
  while (!worklist.empty()) {
    Value v = pop();
    if (BlockArgument arg in scf.for) {
      // 循环携带值：沿 init 操作数(number+2) 和 yield 操作数(number-1)继续回溯
      push(parent->getOperand(number + 2));
      push(yield->getOperand(number - 1));
      continue;
    }
    if (producer 是 tt.load / tt.gather) return true;   // 地址切片到达被加载的索引
    push(producer 的所有操作数);                          // 继续向上游
  }
  return false;
}
```

这是数据流意义上的判定："这个地址的计算切片里是否出现了从内存读出来的索引"。embedding 查表、按 token 索引取行等模式都能被抓住。

**(b) TriangularSolveLoop —— 复合 scope 的代表**

solve_tril 类内核在 TTIR 上呈现为带 16×16 迭代状态的 `scf.for`。识别条件刻意收得很紧（避免误伤普通循环）：体内要有 rank-1、shape=16 的向量 load、axis=0 的 `tt.reduce`、掩码更新用的 `arith.select`，且迭代参数里存在 16×16 张量状态（或存在同构兄弟循环）。

更有意思的是 **scope 操作集合的收集**（`collectTriangularSolveScopeOperations`）。手写的参考 scope 有一个精确的边界形态：

- 四个 16×16 输入 load 留在 scope 外（它们是 SIMD 友好的连续访存）；
- 纯张量的 mask 构造（make_range/expand_dims/broadcast/splat/比较/位运算……）**越过 load 移到后面**进 scope——这就是为什么描述符里有独立的 `scopeInsertionPoint`；
- 兄弟递推循环 + 尾部的 uitofp/addf/select 更新序列进 scope；
- 后续的稠密 `tt.dot` 留在外面（留给 Cube）。

mask 构造能否搬动还要做一个不动点检查：它的所有使用必须都落在选定集合内，否则留在原地当捕获值。这套逻辑保证自动物化出的 scope 与手写版**边界完全一致**——打分的边界就是降级的边界。

结构化事实也在这里抽取：

```cpp
facts.recurrenceStartRow = 2;                       // 首行由直接解出，递推从第 2 行开始
facts.recurrenceLoopCount = siblingLoops * (blockRows - 2);   // 16x16 → 每循环 14 次动态迭代
facts.denseDotTailOps     = /* 最后一个递推循环之后出现的 tt.dot 数 */;
facts.requiresCubeTailPartition = denseDotTailOps > 0;
```

注意 `recurrenceStartRow=2` 被显式写成字段而不是藏在魔法数里——注释原话："Keep the start row explicit so structural matching and reports do not hide that structural assumption in a magic trip count."

**(c) Histogram / Cumsum / Atomic —— 合法性各不相同**

- `tt.histogram`：allSimd 标记为 Unsupported（后端会把该符号别名成 mixed 模板）、allSimtOnly 也 Unsupported（纯 SIMT 管线不给它合法化）；mixed 只接受静态形状、一维整数输入、一维 i32 bin。
- 一维 cumsum（`analyzePlainOneDimensionalCumsum`）：要求恰好一个 combine op 且是加法；allSimd 标记 AliasesMixed（该符号本身就是 SIMT 模板的入口）；extent ≤ 64 时附注"模板走寄存器小路径，不需要 SIMT 异步调用"。
- `tt.atomic_*`：抽出一整套事实（更新元素数、地址秩、值/偏移类型、RMW 种子、静态 mask 活跃比例——通过常量 DenseElementsAttr 或 splat 传播分析、地址是否逐 lane 变化、是否依赖已加载索引）；f16/bf16 且结果被使用时判不支持（旧值语义待验证）。

#### 4.2.3 计划构建：一次遍历，互不重叠

```cpp
SimtAnchorPlan buildMixedSimtAnchorPlan(ModuleOp module, bool compileOn91095) {
  DenseSet<Operation *> covered;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (covered.contains(op)) return WalkResult::skip();   // 已被某个计划内 scope 覆盖
    auto d = analyzeAnchor(op, compileOn91095);
    ...
    for (auto *o : d->scopeOperations) covered.insert(o);  // 复合锚点整体登记
    plan.anchors.push_back(std::move(*d));
    return WalkResult::skip();                             // 锚点内部不再重复匹配
  });
  // 整 kernel 合法性合成：任一锚点 mixed-Native 且无 blocked → mixed Native
}
```

pre-order + skip 保证嵌套时外层锚点赢、集合互不重叠——物化器后续可以直接信任这份计划。

### 4.3 `analyzeSimdSimtFeatures` —— 把 TTIR 读成一本账

**为什么字段这么多？** 因为推导 4 里列的四类效应，每一条都需要对应的计量项支撑。这个函数本质上是在给 kernel 记一本流水账，而且**同时记两本**：kernel 总账（`SimdSimtFeatureSummary`）和锚点子账（`SimtAnchorFeatureSummary`）——后者正是 mixed 定价时要从总账里划出来的部分。

#### 4.3.1 开场：确认前置变换 + 圈定锚点

```cpp
features.ttirLayoutMergeApplied = module->hasAttr("ta.ttir_layout_merge.applied");
// ...扫描 ta.auto_blockify_v1* 属性记录调度信息...
for (anchor : plan.anchors) {
  if (!anchor.materializable) continue;
  anchorSet.insert(anchor.scopeOperations...);
  if (kind == TriangularSolveLoop)
    // 16x16 状态从第 2 行开始递推 → 满块 14 次迭代。
    // TTIR 上界是 min(runtime_remaining, block_end)，编译期非常量，
    // 但结构上已知满块估计——此 fallback 只给这类循环，不给普通循环。
    structuralTripEstimates[scf.for op] = 14;
}
```

随后一遍 walk 统计捕获/逃逸张量（锚点边界流量）：锚点内的 op 若使用了外部定义的张量 → `capturedTensorBytes`；锚点产生的结果若被外部使用 → `escapingTensorBytes`。这两个数字之后会变成 UB 搬运计费的依据。

#### 4.3.2 主遍历中的关键计量

主 walk 对每个非 V1-schedule 的 op 做（挑重点）：

| 计量 | 公式 | 为什么这么算 |
|---|---|---|
| 循环乘子 `loopMultiplier` | 沿父链累乘各层 `scf.for` 的建模 trip count；**跳过带 `ta.auto_blockify_v1.schedule` 的循环** | V1 的物理核调度循环不是算法循环，把它当算法循环会让所有工作量虚增一个调度维度 |
| 加权 op 计数 | `weightedOps[kind] += loopMult; opElements[kind] += elements*loopMult` | 区分"出现几次"和"干多少活" |
| load/store 流量 | `bytes = elements × loopMult × bits/8`；`warpInstrs = ceil(elements/32) × loopMult` | 字节喂 SIMD MTE roofline；warp 指令数喂 SIMT LSU 吞吐 |
| dot 工作量 | `flops = 2mnk × loopMult`，另存 MNK 列表 | 两种模式的 Cube/标量 FMA 价目不同 |
| 归约/扫描 shuffle | `shuffleLaneSteps = inputElements × ceil(log2(extent)) × loopMult` | 树形归约深度就是 log2——这是跨 lane 数据移动量的理论下界 |
| 谓词工作量 | `predicateLaneEvaluations = mask元素数 × loopMult`（按值去重另记 predicateElements） | SIMT 下谓词是逐 lane 评估的真实指令 |
| 分支分类 | `scf.if/cf.cond_br` 且条件是张量 → divergentBranchCount++ | SIMT 上发散分支要付 reconvergence 罚 |
| 循环依赖分类 | iter_args 除归纳变量外：若只流向地址（pointer/addptr/arith-int 链）→ pointerInduction；否则 → loopCarriedDataDependency | 指针归纳可以被下游地址规范化消除，不算真递推；真递推才会卡死 SIMD 的流水线模型 |
| 间接访存计数 | `isLoadedIndexDependentMemoryOp`（真 SSA 判定）；legacy 秩代理并行保留 | 新旧指标并存，报告兼容 |

还有一个组合模式标志值得注意：

```cpp
features.rank1IndirectVectorReduce =
    maxRank==1 && reduceOps>0 && vectorReduceToScalarOps>0 &&
    vectorPtrSplatOps>0 && scalarLoadOps>=2;
// “rank-1 向量间接归约”：gather 到标量归约的特定 lowering 形态，
// SIMD 上代价极高，聚合公式里给它一笔专门的结构惩罚。
```

#### 4.3.3 设计上最容易被忽视的一点

所有统计都是**静态**的（shape/trip count 来自常量或结构推断），遇到动态 shape 就保守处理（`getStaticNumElements` 动态维度返回 1 并置 `hasDynamicShape`）。模型从不假装知道运行时的值——宁可少算也不编造。

### 4.4 Profile 加载与校验 —— 把硬件读成价目表

**为什么要这么复杂的加载流程？** 因为这份 JSON 是模型的"物理常数表"，一旦被悄悄改错，所有决策跟着漂移且无人知晓。所以加载器实现了四道防线：

1. **fail-fast 解析门面**：`ProfileJSONReader` 每个字段都带上下文路径报错（`simd.memory.mte3_bytes_per_system_cycle must be a number`），第一个错误即停；
2. **单位与时间域校验**：数值可以是字面量，也可以是对共享微基准目录的引用：

```cpp
resolveNumberOrMeasurement(obj, "vector_mte2_bytes_per_system_cycle",
                           "throughput_measurement", "bit", ...)
// 引用时 MicrobenchmarkProfile::requireValue 会校验：
//   unit 匹配（如 "scalar_op/system_cycle"）
//   cycle_domain == SYS_CNT（系统周期域，防止拿仿真周期充数）
// 同时把测量的 confidence 传播回 profile
```

3. **相对定价**：未单独测过的算子用 `relative_to` 链表达——`f32.sub {relative_to: "f32.add", factor: 1.0}`，`div` factor 12、`exp` 9、`log` 12……这是"知道相对关系但没单独标定"的诚实编码方式；
4. **版本钉死 + 内容指纹**：

```cpp
if (profile.profileVersion != "david-v100-simd-simt-20260820-v17")
  return createStringError(...);              // 版本不符直接拒绝
// emitPythonCanonicalJSON：与 Python json.dumps(sort_keys=True) 逐字节兼容，
// 再 SHA256——历史上存在过 Python 参考实现，哈希必须跨实现对齐
```

v17 profile 的关键数字（读懂公式必备）：

| 组 | 关键项 | 值 |
|---|---|---|
| 校准 | `program_issue_scale` | 8.0（发射开销放大系数） |
| SIMD 结构惩罚 | irregular / tiny-dot 变体 | 0.8/cap0.5 与 0.24/cap0.12（tiny_dot_flops_max=16384） |
| | mask / reduction / loop / 控制流 | 每 rank 0.022(cap .35)、每次加权归约 0.02(cap .15)、每次静态 trip 0.008(cap .15)、0.03 |
| | rank-1 间接向量归约 / tiny-dot 启动 | 0.75 / 0.06 |
| SIMD 吞吐 | `f32.add` 实测 | 3.3 向量指令/系统周期（high） |
| | div/exp/log 相对系数 | 12 / 9 / 12 |
| SIMD 访存 | mte2 / mte3 | 202.25 B/周期（≈200 GB/s 种子） |
| SIMD Cube | 启动 / 吞吐 | 128 周期 / 4096 flop/周期 |
| SIMT | warp size | 32 |
| | 纯启动 | `simt.setup.empty_with_barrier` ≈ 141 周期（medium） |
| | `f32.add` 实测 | 141 scalar-op/系统周期（high） |
| | dot | 启动 64 周期 / 141 flop/周期 |
| | 谓词率 | 0.0380 warp 指令/周期（CaModel 有效值） |
| | shuffle / GM 读 / GM 写 | ≈0.82 / ≈0.176 / ≈0.129 warp 指令/周期 |
| Stage 资源 | scalar / issue | 1.0 与 6.0(SIMD)；4.0 与 4.0(SIMT) |
| | 间接访存事务 | 0.125 事务/周期 + 200 周期延迟(SIMD)；0.5 + 100(SIMT) |
| | 控制流 | 回边 1、分支 1、同步 1；发散罚 0(SIMD)/4(SIMT) |
| SuperBlock | useful_factor_limit / 压力免罚因子 / 状态带宽 | 4 / 2 / 8 B/周期 |
| Scope 交接 | 定向固定开销 / SIMD UB 读写 / SIMT UB 每线程读写 | 0 / 512、256 B/周期 / 4、4 B/线程/周期 |
| mixed 启动兜底 | warp ∈ {1..32} 各一条空 VF 测量 | 仅作低置信度兜底 |

### 4.5 `StagePartitioner` —— 把 kernel 切成串行阶段的流水线

**为什么是"六步流水线"而不是一个大函数？**

推导 4 要求把 kernel 切成 Phase/Stage。这件事如果写在一个函数里，会同时纠缠四种关注点：程序结构怎么读、边界怎么划、工作量怎么摊、合法性怎么定。代码把它拆成一条**单向数据流**，每一步只做一件事、产出不可变数据给下一步、出错立刻失败——任何一步的输出都可以单独打印调试：

```
① ProgramStructureAnalysis   语义根收集（"有哪些顶层操作"）
② PhaseBoundaryAnalysis      算法域识别 + Phase 指派（"按执行顺序分几个算法阶段"）
③ StageBoundaryAnalysis      模板划分 + 操作所有权精确化（"每个操作归哪个 Stage"）
④ StageWorkloadAnalysis      动态工作量累计（"每个 Stage 干多少活"）
⑤ StageFeatureAnalysis       结构特征重建（"每个 Stage 有哪些结构事实"）
⑥ StageKindClassifier        kind 校验/派生（"该用哪个代价模型"）
⑦ StageModeLegalityAnalysis  合法因子（"SIMT 能开 F2/F4 吗"）
⑧ StagePartitionVerifier     守恒校验（"账有没有记错"）
```

#### 第①步：语义根收集 —— 在哪一层切？

TTIR 函数体里的顶层 op 就是天然的程序骨架。但有一个特例：AutoBlockify V1 的调度循环是个"壳"，壳里的直接 body 操作才是算法操作。如果让壳拥有整个 body，工作量会被双重拥有。所以：

```cpp
// collectTopLevelSemanticRoots (:863)
for (顶层非 terminator op op : 函数体) {
  roots.push_back(op);
  if (op 是 ta.auto_blockify_v1.loop 且有 region)
    for (op 的直接 body 操作) roots.push_back(bodyOp);   // 拆壳：body 提升为独立语义根
}
```

另一个细节：solve_tril 的复合锚点会把 mask 构造移过 load，导致计划中的插入点在文本顺序上早于部分 scope ops 的当前位置。Phase 指派要求按执行序单调分配，所以这一步会在**分析视图里**把 scope 根重排到插入点位置（只调分析用的数组，不改 IR）。

#### 第②步：Phase 划分 —— 三种互斥的算法域

先看域识别（特征版重载 `:1477`，这也是判断"能不能走 Stage 模型"的总开关）：

```cpp
if (恰好 1 个 triangularSolve 事实 && 有锚点)            → TriangularRecurrence 域
else if (无 dot && reduce>0 && 间接访存>0 && load/store>0) → LoadedIndexRowwiseReduction 域
else if (dot>0 && 无 reduce && 间接访存>0 &&
         flops ≤ tinyDotFlopsMax(16384) && load/store>0) → IndirectUnderfilledDot 域
else return std::nullopt;   // ← 不属于三域之一：整个 Stage 模型不适用，
                            //    上层退回聚合公式 fallback（见 4.8）
```

注意这是**白名单式**的设计：只有被明确建模过的三类 kernel 形态才走精确 Stage 路线；其它一律走保守的聚合公式。宁可粗一点，也不对没把握的结构硬套模板。

域确定后，`assignRootPhaseIds(:917)` 用三个**单调状态机**把每个语义根标上 Phase id。以三角域为例，状态只能单向前进：

```
Head ──(遇到含 tt.load 的根)──► Load ──(进入锚点区间)──► Recurrence ──► MergeStore
```

转移条件是对 op 子树的"内容探测"（是否包含 tt.dot / tt.store / tt.reduce / 间接访存）。最后做一次连续性校验：**Phase 一旦关闭就不能重开**——这保证了 Phase 是执行序上的连续区间，而不是散落的标签。

#### 第③步：Stage 划分 —— 先模板扣账，再操作归属精确化

这是最核心的一步，分两层。

**第一层：模板划分（用聚合工作量扣账）。** 以三角域为例（`partitionTriangular :562`）：

```
remaining = 整 kernel 工作量账本（buildKernelStageWorkload :1383）
anchor    = consumeExact(remaining, 锚点子账)     ← 先把锚点的量从总账精确划走
[若 V1 已应用] prependAutoBlockifyStages           ← 调度壳单独立 phase 计费

head_index_mask            PredicateMask          直线型，paysKernelSetup，吃掉剩余标量+谓词+杂项
load_diagonal_tiles        ContinuousTileMemory   独立流水线，吃掉全部 load
diagonal_inverse_recurrence LoopCarriedRecurrence 串行递推 ★localSIMT★ 吃锚点账
                           iterations = facts.recurrenceLoopCount（兄弟循环数 × 14）
merge_store {
  dense_dot_tail           CubeRoofline           iterations = denseDotTailOps
  store_inverse_tile       ContinuousTileStore    吃掉 store + 余量
}
```

`consumeExact` 是"精确扣账"的关键：每个字段独立地 `min(剩余, 请求)` 并同步扣减，保证各 Stage 加起来恰好等于整 kernel——不多算也不漏算。

**第二层：操作归属精确化（有 op 图时覆盖第一层的工作量）。** `attachCompleteOperationOwnership(:1098)` 按 Phase id 把每个语义根映射到目标 Stage，并强制两条铁律：

- **序数单调**：root 在 Stage 序列中的位置必须不回退（`:1173` 报 "non-contiguous Stage ownership"）；
- **守恒**：`owned.size() == rootOperations.size()`——每个根恰好被拥有一次。

merge_store 这个 Phase 还有个精细处理：它内部其实有两个 Stage（dot 尾巴留给 Cube、store 留给向量单元），切分点是"第一个含 tt.store 的根"——在此之前都归 dense_dot_tail，之后归 store_inverse_tile。

接着是四个派生步骤，把"归属"变成可计费的信息：

| 步骤 | 做什么 | 为什么需要 |
|---|---|---|
| `attachExactAnchorOwnership :823` | 锚点属于某 Stage ⟺ 它的全部 scope 根都被该 Stage 拥有 | DP 选出 mixed 后，selector 要知道"这个 SIMT Stage 对应物化哪些锚点"；一个锚点跨两个 Stage 就无法物化 |
| `deriveStageLiveValues :1249` | live-ins/live-outs 及字节数 | SuperBlock 的持久状态压力要按 `liveOutBytes` 收费 |
| `deriveLocalSimtScopeTraffic :1280` | 每个 scope 的捕获/返回张量字节 | UB 双向搬运计费的直接依据；注意单 op scope 返回全部结果，range scope 只返回有外部使用者的值——与物化器的行为严格镜像 |
| `StageWorkloadAnalysis.analyze :1885` | 沿 op 树累计动态工作量（乘 trip count），再除以迭代数归一 | 循环体在 TTIR 里出现一次但执行 N 次；先乘 N 再除 N 听似多余，实则把"N 次×每 iteration 单价"的表达式交给代价模型去展开 |

#### 第④~⑥步：特征、kind、合法性

- **StageFeatureAnalysis(:1565)** 从拥有的 op 图重建结构事实：有无循环、是否携带真数据依赖（`isAddressOnlyLoopValue` 排除纯地址归纳）、间接访存（间接判定/gather/atomic）、归约、dot、转换包……多个兄弟递推循环合并成一个 Stage 时，控制事件计数要除以循环数归一。
- **StageKindClassifier(:1697)** 先查硬错误 `requires_split`：一个 Stage 同时拥有 dot 和（归约/间接访存/携带依赖）两种主导结构 → 直接报错拒绝建模——因为两个独立公式不能共用一段，静默选第一个会藏账或重复计费。然后按优先级链派生 kind：

```
dot ?（flops×iterations ≤ 16384 → TinyCubeRoofline，否则 CubeRoofline）
→ reduction → conversionPack（仅当没有更强结构时）
→ loop（携带依赖？LoopCarriedRecurrence : IndependentPipelinedLoop）
→ indirect(GatherMemory) → contiguous(store-only? Store : Memory) → ScalarIssue
```

派生结果与模板预设 kind 不符时**报错而不是静默改判**（附上完整事实清单），宁可编译失败也不错账。

- **StageModeLegalityAnalysis(:1977)** 定合法因子：`simdLegal` 恒真；`legalSimtFactors = {1} ∪ ({2} if max≥2) ∪ ({4} if max≥4)`，其中 max 由上层传入（=4 当且仅当 whole-kernel SuperBlock 可物化或 V1 已应用）；局部混合 scope 的因子默认压到 `{1}`，除非后端声明支持 Scope SuperBlock 批处理。**合法性在这里集中裁决，后面 DP 只在合法候选里挑**。

#### 最后：守恒校验

`StagePartitionVerifier.verify(:1913)`：id 唯一、materializable Stage 必须真的拥有操作和至少一个锚点、锚点所有权全局不重叠、被拥有操作数等于 modeledOperationCount。fallback 模式则退化为工作量守恒校验（各 Stage 总量 ≈ kernel 账本，相对误差 1e-6 内）。

一句话概括这条流水线的哲学：**每一层只信任上一层的产出，并在自己这一层加一条不变量；任何不变量被破坏都是硬错误，绝不带着错误的账往下走。**

### 4.6 `StageCostModels` —— 给每个 Stage 算 SIMD/SIMT 两种实现的周期

**为什么要拆"workload 映射"和"per-kind 组合模型"两层？**

因为它们的变化频率不同：资源映射规则（逻辑量÷吞吐率）跟着硬件 profile 走，而"这些资源周期如何组合成总时间"跟着程序结构走。拆开后换一代芯片只需换 profile，新增一种程序结构只需注册一对新模型。

`StageCostEvaluator::evaluate(:802)` 对每个 Stage 的主循环：

```cpp
implementations = [];                                   // 该 Stage 的合法候选集
if (stage.simdLegal)  implementations += {SIMD,  F1};
for (f : stage.legalSimtFactors)
  if (stage.simtLegal) implementations += {SIMT, f};    // F1/F2/F4

for (impl : implementations) {
  resources = impl.mode==SIMD ? mapSIMDWorkload(stage, hw.simd)
                              : mapSIMTWorkload(stage, hw.simt);
  model     = registry.lookup(impl.mode, stage.costModelKind);   // 查表，缺失即报错
  cost.totalCycles = applySuperBlock(stage, resources, impl,
                       model->estimate({stage, profile}, impl, resources));
}
```

#### mapSIMD/mapSIMTWorkload：逻辑量 → 资源周期

两张对照表就是这两个函数的全部内容：

| 资源项 | SIMD 映射 | SIMT 映射 | 备注 |
|---|---|---|---|
| compute | Σ ceil(elements/vectorWidth) ÷ 吞吐 × factor | Σ elements ÷ 吞吐 × factor | SIMD 除以向量宽再向上取整——不满一条向量也要占一拍 |
| load/store | bytes ÷ MTE 带宽；间接访存改用事务率且**每次迭代加一次依赖延迟** | warp 指令数 ÷ LSU 吞吐；同样有间接事务模式 | 间接访存的依赖延迟每次迭代都要付，这是 gather 类慢的根源 |
| predicate | ceil(elements/vw) ÷ 谓词率 | elements ÷ 谓词率 | |
| shuffle | laneSteps ÷ lanes-per-cycle | 同左 | |
| dot | setup（一次性）+ flops÷flops/cycle | 同左 | setup 只付一次，体现流水化收益 |
| issue | ceil(issueElements/issueWidth) ÷ 发射率 | 同左 | 共享前端的吞吐下限 |
| criticalPath | 携带依赖：scalar+compute+predicate+shuffle+dot；归约：compute+predicate+shuffle | 同左 | 给 per-kind 模型的关键路径提示 |

控制流统一由 `materializeControlFlow(:50)` 追加：回边数×回边周期、分支数×分支周期、同步数；**发散罚只在 SIMT 上收**，且乘 `(1 − activeLaneRatio)`——全活跃的分支没有浪费 lane，不该挨罚。

#### per-kind 模型树：资源周期的组合规律

所有模型共享一个基元：

```cpp
static double serialBody(r) {
  execution = scalar + load + store + compute + predicate + shuffle + dot
            + control + spill;
  return max(execution, r.issue);   // 注释：issue 是共享前端吞吐下界，不是额外指令流，
                                    // 加进求和会对每条指令双计费
}
```

18 个注册模型（20 kind 中 dispatch 两类共享、其余成对）的差异只在"怎么组合"。挑代表性的列出来：

| 家族 | SIMD 公式 | SIMT 公式 | 直觉解释 |
|---|---|---|---|
| Dispatch（V1 调度） | `setup + N·max(scalar+control, issue)` | 同左 | 调度壳就是一小段控制逻辑 |
| Scalar 族 | `setup + N·serialBody` | 同左 | 无重叠机会，全串行 |
| ContinuousMemory | 可重叠时 `setup+N·(scalar+predicate+control+spill+max(load,store,issue))` | `setup+N·serialBody` | **SIMD 的 MTE 与计算并行是硬件事实**；SIMT 当前降级是串行指令流，不许臆造重叠 |
| IndirectMemory | 串行（事务率+延迟已含在 load 里） | 串行 | 依赖链卡死，两边都没法 overlap |
| IndependentPipelined | `max(load, store, compute+dot+shuffle, scalar+predicate+control, issue)+spill` | serialBody | 五路资源的 roofline 取 max |
| Recurrence | `max(criticalPath+load+store+control+spill, issue)×N` | 关键迭代次数 = ⌈N ÷ min(独立组数, warp 组数=numWarps)⌉；总量 = max(criticalPath×关键迭代, 全局 issue 下限 N·issue) | **SIMT 的杀手锏**：多个独立递推组可在 warp 组间交错隐藏延迟，但发射带宽是全体迭代的硬下限，躲不掉 |
| Reduction | `(scalar+load+store+criticalPath树深+control+spill) vs issue 取 max` | 同左 | 树深是依赖链，发射率是下限 |
| Cube | 重叠时把 dot 并进 compute 路 | 串行 | |
| ConversionPack | 重叠时 `predicate+control+spill+max(scalar+compute, load, store, issue)` | 串行 | |

#### applySuperBlock：F2/F4 的收益与代价

SuperBlock（SuperBlock 因子 F）指一个物理核上跑 F 个独立逻辑程序组。它能隐藏延迟，但不是免费的：

```cpp
effectiveFactor          = min(F, superblockUsefulFactorLimit);      // 超过上限延迟隐藏不再增益
latencySensitive         = N · (load + store + shuffle + divergence);// 只有这部分能被别的组填满
pressure                 = N · spill · (F − 1);                      // 多组并存 → spill 变多
persistentStatePressure  = 携带依赖 && F > pressureFreeFactor
                             ? (F − pressureFree) · liveOutBytes ÷ 状态带宽 : 0;
                         // 递推状态要在 F 份寄存器里各留一份，超过免罚点开始收费
result = max(issueFloor,
             stageCycles − latencySensitive + latencySensitive/effectiveFactor + pressure)
             + persistentStatePressure;                                // 状态压力不能藏在 issue floor 后面
```

这段代码把"什么时候 F2/F4 值得开"变成了可计算的问题：访存/洗牌重的 Stage 受益，spill 多或递推状态大的 Stage 受罚——profile 里那三个超参（useful=4、pressure-free=2、8B/cycle）就是这条权衡曲线的三个旋钮。

### 4.7 `solveStageRoutes` —— 把 Stage 序列组装成三条路线的动态规划

**为什么是 DP 而不是贪心？** 因为一个 Stage 的最优选择未必是全局最优：前面省一拍可能导致路线类别翻转，触发不同的边界成本；SuperBlock 因子更是 kernel 级属性，中途换会拼出不可实现的组合。

**状态设计**（`StageRouteCostModel.cpp:417` 起）：

```cpp
struct PartialRoute {
  double totalCycles;              // 当前类别的真实累计
  double mixedEquivalentCycles;    // 反事实：若此后转成 mixed，前缀应按什么价累计
  StageMode exitMode;
  RouteClass routeClass;           // AllSIMD / AllSIMT / Mixed
  bool   allSimtStagesLocal;       // 所有 SIMT Stage 都可局部物化？
  vector<StageImplementation> implementations;
  int64_t routeSuperblockFactor;   // 本路线的统一 SuperBlock 因子
};
using FactorRoutes = std::map<int64_t /*factor*/, PartialRoute>;
using State = std::array<std::array<FactorRoutes, 3>, 2>;  // [exitMode][routeClass]→{factor: 最优前缀}
```

为什么保留 factor 维度？源码注释给了教科书级的理由：

> Collapsing the factor dimension can discard a slightly slower F1 prefix that becomes globally optimal, or worse, combine F1 and F4 Stage costs into an unrealizable F4 kernel.

**转移时的四条规则**（逐条对应一个物理/工程约束）：

1. **因子统一约束**：route 已含 SIMT 时，后续 SIMT Stage 必须沿用同一 factor——SuperBlock 是 kernel 级调度，不允许一半 F1 一半 F4；
2. **mixed 合法性**：`allSimtStagesLocal` 一票否决——mixed 里任何一个 SIMT Stage 不能物化，整条 mixed 作废；
3. **entryTransition = 0**：相邻 Stage 的模式标签切换不额外收费。注释说得极清楚："Local scopes pay both physical directions in mixedEquivalentStageCost. Adjacent Stage labels are a logical route description, not an additional hardware transition."——物理成本按**实际生成的 scope 个数**收，而不是按标签变化次数收；
4. **进入 Mixed 即换价目**：一旦 routeClass 变为 Mixed，累计值切换为 `mixedEquivalentCycles`——其中每个 SIMT Stage 的单价换成 `mixedEquivalentStageCost`。

`mixedEquivalentStageCost(:38)` 就是推导 4 里说的"边界成本精确计量"，公式完全对应物理现实：

```cpp
fixedTransitions = scopeCount · (simd→simt + simt→simd);        // 每个物理 scope 的定向切换
inputHandoff  = inputBytes/simdUbStoreRate
              + inputBytes/(simtUbLoadPerThread · activeThreads);  // SIMD 寄存器→UB→SIMT 寄存器
outputHandoff = outputBytes/(simtUbStorePerThread · activeThreads)
              + outputBytes/simdUbLoadRate;                        // SIMT 寄存器→UB→SIMD 寄存器
activeThreads = warpSize × activeLaneRatio;                        // 只付活跃线程的部分
```

DP 收尾后在 AllSIMD / AllSIMT / Mixed 三个类别里各取最小者（遍历 exitMode × factor），连同分段明细一起装进 `StageCostModelSummary`。至此三条候选路线各有了带完整出处的总周期。

### 4.8 `estimateSimdSimtCandidatesImpl` —— 决策入口与聚合公式 fallback

现在把镜头拉回打分的总入口（`SimdSimtCostModel.cpp:2394`）。它的结构是一个清晰的三段式：

```
① 合法性闸门（legality gates）——先定"谁有资格参赛"
② Stage 模型路径（若 4.5–4.7 的流水线命中三域之一）
③ 聚合公式路径（fallback：Stage 模型未命中时的兜底）
```

#### 第一段：合法性闸门

```cpp
allSimdLegal     = kernelLowerability.allSimd == Native;               // 锚点层面无障碍即可
allSimtLegal     = compileOn91095 && !hasExplicitScope
                   && (allSimtStatus == Native || BackendConditional); // 纯 SIMT 需要目标后端
mixedLegal       = !hasExplicitScope && applicability.materializable
                   && kernelLowerability.mixed == Native;              // mixed 还要求锚点真能物化
```

三条规则各自编码了一条工程事实：显式 scope 存在说明用户已经手工接管，模型不再越权；BackendConditional 允许 all-simt 进入比赛是因为"锚点本身没问题，缺的只是选中纯 SIMT 后端管线"这个条件可以在选择成立后满足。

#### 第二段：Stage 模型路径（首选）

```cpp
stageModel = evaluateStageModel(...);        // 4.5–4.7 的完整流水线
if (stageModel) {
  candidate.allSimd  = stageModel.allSimd.totalCycles;    // 分数直接来自 DP 总价
  candidate.mixed    = stageModel.mixed.totalCycles;
  legal &= stageModel.X.legal;                            // DP 层的合法性再 AND 进来
  decision = chooseBest(...);                             // 合法集合里取最小
  return report;
}
// 否则落到第三段
```

#### 第三段：聚合公式路径（fallback）

当 kernel 不属于三个白名单域（比如纯 histogram kernel、带 scan 的 kernel），Stage 流水线返回空，此时用保留至今的 v1 公式。它的骨架值得读懂，因为它就是推导 4 开头那个朴素想法的完整实现：

```cpp
// 每种算子的两份价签
simdCycles = ceil(elements/vectorWidth)/throughput * factor;
simtCycles = elements/throughput * factor;

// SIMD 访存走 roofline（读写并行取 max），SIMT 访存走串行和（依赖序指令流）
simdMemoryCycles = max(loadBytes/mte2, storeBytes/mte3);
simtMemoryCycles = loadWarpInstrs/rate + storeWarpInstrs/rate;

// payload：SIMD 允许计算与访存重叠 → max；SIMT 串行 → 求和
// （注释：当前 SIMT lowering 是依赖序 warp 指令流，没有任何测得的重叠契约
//  支持写 roofline，所以老实相加）
analytical = setup + payload × programIssueScale(8.0);

// 结构惩罚只加在 A_SIMD 上！注释原文：
// "changing SIMT throughput/setup must never change the all-SIMD score"
all_simd_score = simdAnalytical × (1 + structuralPenaltyRatio);
all_simt_score = simtAnalytical;

// mixed：regular 部分（扣除锚点份额后的 SIMD 价 + 残余结构惩罚）
//       + 锚点部分（SIMT 价）
//       + 启动兜底（按 numWarps 取最近的空 VF 测量值）
mixed = mixedSetupFallback(numWarps)
      + programIssueScale × (regularPayload×(1+残余惩罚) + anchorPayload);

// 若无可物化锚点：mixed = max(all_simd, all_simt) + setupFallback，
// 且 cost_source 标注 "inapplicable_without_materializable_anchor"
```

两个细节体现了这个团队的诚实文化：

- **候选独立性**：结构惩罚描述的是"SIMD roofline 没覆盖到的真实开销"（不规则寻址的额外拍数、mask 物化、循环开销……），它属于 SIMD 自己，绝不能因为 SIMT 参数变了而漂移；
- **启动兜底明码标价**：mixed 的启动开销目前没有测得的"方向性切换"数据，用的是"空 VF 探针"并标注 `confidence: low`、`unmeasured`——宁可承认粗糙，也不假装精确。

无论走哪条路径，最后都经过同一道出口：`chooseBest` 在合法候选里取分数最小者为 decision，并输出 `candidateRatiosToBest`（次优/最优比），让人一眼看出"赢了多少"。

### 4.9 `SelectSimdSimtCostModelPass` —— 决策与落地的总装车间

**为什么这个 pass 是"唯一决策入口"？** 文件头注释写明分工：

> This pass is the online owner of SIMD/SIMT candidate selection. Python only schedules the pass and reacts to its machine-readable execution intent.

即：**Python 只负责把 pass 排进管线、读回结果属性；所有智能都在 C++ 里。** 这保证了决策逻辑只有一份事实源。

`runOnOperation()` 的完整流程在 3.2 已列出，这里展开最关键的两块。

#### 决策 ≠ 生效：action support 二次审查

分数最小者只是 *recommended*；能否成为 *effective* 还要过一遍"落地能力"审查——每一项都对应一个明确的工程约束，失败时记录原因字符串进报告：

| 条件 | 原因码 | 背后的物理/工程事实 |
|---|---|---|
| mixed 但 IR 里已有显式 scope | `explicit_scope_present` | 用户手工接管过，模型不越权 |
| mixed 但 DP 的 SIMT Stage 一个锚点都没绑上 | `no_materializable_mixed_anchor` | 打分说能物化，物化器却无 op 可包——不一致即拒绝 |
| mixed 且 factor>1 但 Scope SuperBlock 未支持 | `scope_superblock_not_materializable` | 注释原文：factor>1 的 mixed 需要把周边 SIMD 生产者/消费者一起批处理，目前没有这个 pass |
| all_simt 且有显式 scope | `explicit_scope_present` | 保住用户手写的局部语义，不用纯 SIMT 整体覆盖 |
| factor × numWarps > 64 | `superblock_warp_limit_exceeded` | 硬件并发上限 |
| all_simt factor>1 但 AutoBlockify V1 不可用且未应用 | `superblock_requires_auto_blockify_v1` | F2/F4 是 kernel 级调度，必须有 V1 配合 |

只有 `mode=="auto"` **且** 通过审查，`effective` 才等于 `recommended`；否则回落 `backend_default` 并保留推荐值与原因。这个设计让"报告模式"（只出分析不动 IR）与"自动模式"共用同一套代码路径。

#### 就地物化：一次调用内闭环

```cpp
if (effective == kMixedSimdSimt &&
    failed(materializeSimtAnchorPlan(module, selectedMixedAnchorPlan))) {
  signalPassFailure();
}
```

`buildSelectedMixedAnchorPlan(:80)` 从 DP 结果里挑出被指派为 SIMT 的那些 Stage，收集它们绑定的锚点索引——**注意是子集**：DP 可能判断某个锚点留在 SIMD 更划算（比如它所在 Stage 的 SIMT 实现太贵），那就一个都不物化。这与推导 3 的"打分=落地"一脉相承：mixed 分数计费的锚点集合，正是此刻被包裹的集合。

最后把完整 JSON 报告存到 module attr 并按行追加到 `reportFile`，供离线分析与回归比对。

### 4.10 `MaterializeSimtScopes` —— SSA 安全的 scope 包裹

**为什么需要专门的物化器？** 把一串 op 移进 region 听似简单，但 MLIR 是 SSA 形式：移动后，区域内定义、区域外使用的值必须通过 region 返回值重新接线。两个函数分别处理两种形态。

**单 op 包裹 `wrapAnchorOperation`：**

```cpp
OperationState scopeState(loc, "scope.scope");
scopeState.addTypes(op->getResultTypes());          // scope 的结果类型 = 原 op 的结果类型
scopeState.addAttribute(kVectorModeAttr, "simt");
scope.scope = builder.create(scopeState);
op->moveBefore(scopeBody, end);                     // 只搬这一个 op
bodyBuilder.create("scope.return", originalResults);// 原结果作为返回值穿出去
original.replaceAllUsesExcept(替换值, returnOp);     // 外部使用改接 scope 结果
```

注释点出一个容易忽略的合法性依据："Scope regions are not isolated from above, so operands remain legal captures"——非隔离 region 可以自由引用外部值，所以区域外的生产者无需搬动。

**多 op 区间包裹 `wrapAnchorRange`：**

多了一个"逃逸值"判定——不是所有中间结果都要穿出去，只有被区间外使用的才需要：

```cpp
isInsideRange(user) { /* 沿父链查是否落在 planned 集合内 */ }
for (op : ops)
  for (result : op->getResults())
    if (∃ use 且 !isInsideRange(use.owner)) escaping.push_back(result);
// scope 类型 = 逃逸值类型集合；scope.return 操作数 = 逃逸值；
// 替换时跳过 return 自身和区间内部的使用
```

solve_tril 的递推状态就是典型逃逸值：循环内更新、dot 尾巴还要用，必须经 `scope.return` 交还给 SIMD 世界。

**编排层 `materializeSimtAnchorPlan`：**

遍历计划中的锚点，跳过不可物化、已被覆盖、已处于 simt scope 内的 op；多 op 锚点先包（range），单 op 锚点随后包；若最终一个都没物化则 emitError——因为走到这里的调用方已经声明了 mixed 生效，空手而归意味着上游出了 bug。

**兼容校验 pass `MaterializeSimtScopesPass`：**

selector 之后排队的独立小 pass，只做一件事：若模块声明 `effective == mixed` 却找不到任何局部 `scope.scope<simt>`，直接让编译失败。它是"声明与事实一致"的最后防线。

### 4.11 Python 侧接线与决策消费

**调用顺序为什么是 layout merge 在前？** `_run_ttir_layout_merge(:246)` 先跑，costmodel 后跑——因为模型读取 `ta.ttir_layout_merge.applied` / `hacc.coalesce_factor` 标记并按合并后的访存形态计费（推导 1）。反过来就变成给一个不会运行的程序打分。

**两个 SuperBlock 开关的一真一假：**

```python
whole_kernel_superblock_materializable = (
    opt.compile_on_910_95 and opt.enable_auto_blockify is not False)   # 真·可用性
# scope_sb 固定传 False，注释原文：
# "Whole-kernel AutoBlockify V1 cannot materialize a SuperBlock for a local
#  mixed scope. Claiming otherwise lets the Route Model select F2/F4 even
#  though the executable still runs the scope as F1."
```

这是典型的**诚实开关**：宁可让模型少选 F2/F4，也不能谎报能力导致实际执行的程序与被打分的程序不同。

**决策翻译表 `_apply_cpp_simd_simt_decision(:176)`：**

| effective | metadata 动作 |
|---|---|
| `all_simd` | `compile_mode="simd"`、`parallel_mode="simd"`、关闭 AutoBlockify V1（注释：all-SIMD 必须用原始逻辑网格启动） |
| `mixed_simd_simt` | 记录 `auto_simt_requested_kind`，后续阶段据此加混合运行时标志 |
| `all_simt_only` | 若 whole-kernel SB 可物化则自动启用 V1——选择权归路由模型而非环境变量 |

**纯 SIMT 快速通道（`:353`）：** 当 effective 为 all_simt_only 或整体已是 void scope 时：

```python
metadata["force_simt_only"] = True; metadata["parallel_mode"] = "simt"
metadata["shared_mem_dynamic_size"] = 122880
ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)   # 拆掉 scope 壳
# …TA AutoBlockify V1 用选出的 superblock_factor refine/run…
return str(mod)      # 直接返回 TTIR！纯 SIMT 管线绕过 linalg，
                     # 交给 bishengir-compile 的标量降级流水
```

**mixed / all_simd 则继续正常降级**，下游 `TritonToLinalgPass`、`UnstructureConversionPass` 等通过 `shouldUseSimtTemplate()` 逐 op 查询契约决定是否套用 SIMT 模板。

---

## 5. 实例走查：一个三角求解 kernel 的完整旅程

把前面所有零件串起来。设输入是一个 solve_tril 形态的 kernel：四个 16×16 输入 tile 连续 load；mask 构造 op 序列；若干兄弟 `scf.for` 循环（16×16 迭代参数携带三角状态，体内 rank-16 向量 load + axis-0 reduce + `arith.select` 掩码更新，动态执行约 14 行）；之后稠密 `tt.dot` 与结果 store。

**① 锚点计划。** `isTriangularSolveLoop` 命中 → kind=TriangularSolveLoop；事实抽取得到 `recurrenceStartRow=2`、`recurrenceLoopCount = 兄弟数×14`；`collectTriangularSolveScopeOperations` 找到插入点（最后一个输入 load 之后）、圈入可搬动的 mask 构造与全部兄弟循环及尾部 select 序列、确认 dot 尾巴留在外面。合法性：mixed=Native，allSimtOnly=BackendConditional（*"requires_cube_tail_partition"*）。

**② 特征账本。** 结构化 trip 估计 14 进入乘子；捕获张量（mask 输入常量等）与逃逸张量（递推结果喂 dot 尾巴）的字节数入账；kernel 总账与锚点子账并行累计。

**③ 合法性与域识别。** 无显式 scope、目标 910_95、锚点可物化 → 三条路线全部入场。特征命中 TriangularRecurrence 域（恰好一个三角锚点）。

**④ 切分与归属。** Phase 序列 head → diagonal_load → diagonal_inverse → merge_store（锚点区间连续性校验通过）；Stage 模板生成五个段，操作归属精确化后每个根恰归一段；锚点绑定到 `diagonal_inverse_recurrence` Stage，scope 进出字节算好。

**⑤ 逐段计价。** 关键对比发生在递推 Stage：SIMD 实现被依赖链卡死——criticalPath ≈ 14 次串行迭代 × (load+reduce+select)；SIMT 实现可以把兄弟递推组铺到多个 warp 组交错，关键迭代次数除以组数，只剩全局发射下限压着。其余四段都是 SIMD 强项（连续访存 roofline / Cube roofline）。SuperBlock 因子在 DP 中统一裁决。

**⑥ DP 组装。** mixed 路线 = 四个 SIMD Stage + 一个 SIMT Stage，加上 UB 双向搬运（捕获/逃逸字节）与 scope 定向切换；all_simd 路线全额吃下递推 criticalPath 再叠加结构惩罚；all_simt 路线整个 kernel 换成标量流。三者比较，通常 mixed 胜出——这正是这类 kernel 手写优化的方向，现在由模型自动得出。

**⑦ 决策与落地。** selector 判定 mixed 可支持（无显式 scope、锚点非空、factor=1）；就地物化 range scope（插入点在 load 之后，mask 构造与递推循环进 scope，逃逸值走 `scope.return`）；兼容 pass 验证通过；Python 读回 `effective=mixed` 继续正常降级，下游逐 op 查询 `shouldUseSimtTemplate` 完成真正的混合降级。

## 6. 设计原则清单

把散落在各处的注释和结构选择提炼成十条，它们比任何单个函数都更值得带走：

1. **打分的 = 落地的。** 一份不可变锚点计划贯穿特征提取、打分、物化三方；物化在同一次 pass 调用内完成，不留中间标记。
2. **每个操作恰好被拥有一次。** 所有权守恒是硬不变量，verifier 校验，破坏即编译失败。
3. **合法性与代价分离。** legality gates 先裁掉没资格的候选；代价只在合法集合里比较；BackendConditional 这类"条件可用"状态让两条线解耦演化。
4. **候选独立性。** 结构惩罚只加在 A_SIMD 上；改变 SIMT 吞吐绝不允许影响 all-SIMD 的分数。
5. **白名单式精确建模 + 保守兜底。** 只有三种被透彻理解的域走 Stage 流水线；其余退回聚合公式；没测过的重叠绝不建模（SIMT 一律串行求和），没测过的启动开销用兜底值并标注低置信度。
6. **宁可失败，不可静默错账。** `requires_split`、kind mismatch、所有权不守恒、profile 版本不符……全部硬错误。解析模型的信任来自每一次都算得对，而不是大多数时候算得对。
7. **校准数据版本化、可追溯。** 版本号钉死在代码里；数值必须带单位与周期域；内容 sha256 与历史 Python 实现字节级兼容。
8. **报告自解释。** 每个 major 项都能在 breakdown 里找到出处；`unsupported` 清单明示未建模项；`candidateRatiosToBest` 说明赢了多少；`application_reason` 解释为什么不生效。
9. **单一事实源。** 契约只有一个头文件、一个拼写、一个查询口；Python 只调度，C++ 决策。
10. **物理直觉显式编码为代码注释。** "相邻标签不是硬件切换""issue 是共享前端下限不能双计费""发散罚按活跃 lane 比例收"……这些判断全部写在公式旁边的注释里，让下一个维护者能区分"实现细节"与"硬件断言"。

---

## 附：快速定位速查表

| 你想看… | 去哪 |
|---|---|
| 决策入口 | `RouteModel/Transforms/SelectSimdSimtCostModel.cpp::runOnOperation` |
| 锚点怎么识别 | `RouteModel/SimtAnchorAnalysis.cpp::analyzeAnchor` |
| 特征字段含义 | `RouteModel/SimdSimtCostModel.h` 的两个 Summary 结构体 |
| 三域怎么判 | `RouteModel/StagePartitioner.cpp::PhaseBoundaryAnalysis::analyze` |
| 每段怎么计价 | `RouteModel/StageCostModels.cpp::mapSIMD/mapSIMTWorkload` + 各 `*StageCostModel::estimate` |
| F2/F4 怎么建模 | `StageCostModels.cpp::applySuperBlock` |
| 三条路线怎么拼 | `StageRouteCostModel.cpp::solveStageRoutes` + `mixedEquivalentStageCost` |
| fallback 公式 | `SimdSimtCostModel.cpp::estimateSimdSimtCandidatesImpl` 第三段 |
| scope 怎么包出来 | `Transforms/MaterializeSimtScopes.cpp::wrapAnchorOperation/wrapAnchorRange` |
| 下游怎么读决策 | `include/AscendModel/RouteModel/SimtSelection.h::shouldUseSimtTemplate` |
| 价目表在哪 | `costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json` + `profiles/microbench/ascend_davidv100_v1.json` |




