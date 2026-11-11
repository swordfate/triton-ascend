# RouteModel 连续访存计价机制速查（A–E）

> **基线 commit：`d4667dab9`**（revert(costmodel): drop the loop-overlap model）。
> 本文记录 autoscope RouteModel 中 `mapWorkload` 对 GM 访存按**执行顺序**排列的认领机制 A–E：
> 每个机制管哪种访问、具体估计公式、参数来自 profile 的哪个字段、基线时的实测数值。
> 以后对任一机制有改动，请在「§9 变更记录」加一行，并同步更新对应小节。
>
> 文中所有 `file:line` 均对应 `d4667dab9` 时点的文件（`third_party/ascend/costmodel/` 下）。
> 时间单位：system counter，**1000 MHz，1 cycle = 1 ns**。
> profile 数据来源：
> - 路由 profile：`profiles/simd_simt/ascend_950pr_simd_simt_v1.json`
> - 实测值（符号名解析）：`profiles/microbench/ascend_950pr_v1.json`

---

## 0. 公共前置

### 0.1 调用链

```
compiler.py:1147 _run_cpp_simd_simt_costmodel()        ← 开关 opt.auto_simt_scope_mode
  └─ SelectSimdSimtCostModelPass::runOnOperation()     (SelectSimdSimtCostModel.cpp:182)
       └─ analyzeSimdSimtCandidates()                  (SimdSimtCostModel.cpp:4399)
            └─ evaluateStageModel()                    (SimdSimtCostModel.cpp:3104)
                 ├─ StagePartitioner::partition()      ① 切 stage + 分类（Analysis/StagePartitioner.cpp）
                 ├─ StageCostEvaluator::evaluate()     (StageCostModels.cpp:3840)
                 │    └─ mapWorkload()                 ② 计价核心（StageCostModels.cpp:1226）＝本文 A–E
                 └─ solveStageRoutes()                 (StageRouteCostModel.cpp:3543)
            → chooseBest → all_simd / all_simt_only / mixed
```

### 0.2 访存描述与"体型"分类（所有机制的公共入口）

每条 `tt.load`/`tt.store` 由 `describeMemoryAccess`（StagePartitioner.cpp:1748）变成 `StageMemoryAccess`
（shape / strides[元素单位] / elementBits / masked / count[一次 Stage 迭代内动态执行次数] …），
strides 来自 `tt.make_tensor_ptr` 常量操作数或 `tt.addptr` 链仿射推断。

`layoutClass()`（StageRouteCostModel.cpp:524）按两个问题分体型：

| 体型 | 判定 | 含义 |
|---|---|---|
| **ContiguousWide** | 各维按 stride 排序后完美嵌套，且最内连续段 ≥ 32 B | **连续访存**（本文主题） |
| **ContiguousShort** | 完美嵌套，连续段 < 32 B | 连续访存 |
| StridedWide | 有洞（某维 stride > 前面各维长度之积），行 ≥ 32 B | 跨步 |
| StridedShort | 有洞，行 < 32 B | 跨步 |
| Unknown | stride 推断不出 | —— |

### 0.3 "整图连续"门票（仅 D 需要）

`blockPointerCanUseFlatMemoryCurve`（StagePartitioner.cpp:1683）：指针（剥掉 `tt.advance` 后）
**必须由 `tt.make_tensor_ptr` 定义**、所有有效维 stride 为正常量、完美嵌套 → 才计入
`work.continuousLoadCounts / continuousStoreCounts[(总字节, 总元素)] → 次数`。
普通指针加偏移（`tt.addptr` 链）**拿不到这张门票**。

### 0.4 循环次数（iterationCount）的四条渠道

`getLoopTripCount`（StagePartitioner.cpp:1977）依次尝试：

1. **常量边界**：scf.for 的 lb/ub/step 全常量 → 精确；
2. **运行时标量实参值**：后端经 capability JSON `runtime_scalar_values` 传入，挂为函数参数属性
   `ascend.costmodel.runtime_value`（SelectSimdSimtCostModel.cpp:61），`getConstantInteger`（:1301）认它 → 精确；
3. **可证上界**（如 `arith.minsi` 钳位链）→ 按上界估，并打 `upperBoundLoopCount` 标记；
4. **grid-stride 模式**（lower=program_id, step=num_programs）→ 用 `logicalProgramCountHint` 推。

四条全失败（真未知）→ 次数取 `max(1, iterationCount/循环个数)`（顶层即 1），打 `unknownTripCountLoopCount` 标记；
该标记使依赖精确次数的实测拟合拒绝认领（例：`simtStaticLoopDirectMemoryCompletion`，
SimdSimtCostModel.cpp:3407–3408 要求两标记均为 0），退回解析式——**宁可按 1 次保守估计，不用未校准次数外推**。

工作量口径：先按已知次数乘总账（`scaleWorkload`，`access.count *= scale`，StagePartitioner.cpp:1270），
再 `makePerIteration` 除以 `iterationCount` 归一到每次迭代（:3494），计价时 `setup + count × body` 乘回，不重复计数。

### 0.5 认领顺序与互斥（重要）

`mapWorkload` 非 indirect 分支（StageCostModels.cpp:1688 起）按固定顺序执行，**先到先得、后到只算剩余**：

```
A（strided-short，只认 StridedShort）
 → B（独立 load 组，SIMD）
 → SIMD 流水 strided store 组（StridedShort store，略）
 → C（布局组曲线，同 layoutKey 合并）
 → D（字节桶曲线，扣除 C 已覆盖的 count）
 → E（解析式兜底，只算全部未覆盖余量）
```

每步认领成功都把覆盖量记进 `LayoutCoverage layout`（bytes / operations / warpInstructions /
loadShapes），后续步骤按 `work.loadBytes − layout.loadBytes …` 拿剩余。**同一字节不会被计两次价。**

---

## 1. 机制 A：strided-short 闭环公式 —— 不收连续访存（划界用）

| 项 | 内容 |
|---|---|
| commit | `d4667dab9`（更早引入，本版未改） |
| 管哪种访问 | **仅 StridedShort**（跨步、行 < 32 B）。代码显式排除 stride-1：*"A unit inner stride is contiguous, not strided"*（StageCostModels.cpp:427）。列出仅为划清边界 |
| 准入条件 | `layoutClass()==StridedShort`；静态布局；地址不依赖 load 值 / 无非线性张量计算 / 无控制流选择；`effectiveRank() ≤ 2`；`innerStrideBytes > elementBytes`；stage 无同步 |
| 公式 | 闭环拟合（截距 + 系数向量），按 `selectStridedShortFormula` 选槽位后 `evaluateStridedShortCompletionFormula` 求值（Models/StridedShortCostModel.cpp）。形式：`cycles = intercept + Σ ci·xi`，槽位如 `simd_load_stride2_r1`（intercept 65.1033, coef [0.00847947, 0.00478897], 21 点 MAPE 1.22%） |
| 参数来源 | profile `strided_short_completion.*`（SIMD/SIMT 各一组槽位；标定自 rotate-loop 2000/8000 迭代斜率＝稳态每迭代边际成本） |
| 记账 | `resources.load/store += cycles × count`（另记 `stridedShortLoad/StoreCompletion`），并把该访问加入 `claimed`，使 masked-strided-copy floor 让位 |
| 代码 | StageCostModels.cpp:397–483 |

---

## 2. 机制 B：独立 load 组包络（SIMD 专用）

| 项 | 内容 |
|---|---|
| commit | `d4667dab9` |
| 管哪种访问 | **连续**（ContiguousShort/Wide，`isFlatContiguous()`）的 **FP32 load**，且是 store 之前的"平坦前缀"；一次 Stage 迭代内 1–4 条 |
| 准入条件 | ① SIMD 实现；② `permitsSimdOverlap`：调度 = IndependentPipelined 且无循环携带依赖（:162）；③ stage 无同步、无条件分支；④ profile `independent_loop_load_group.bytes_per_system_cycle > 0`；⑤ 前缀各 load 全 FP32 平铺连续；⑥ count 为正整数；⑦ 每组操作数 ∈ [minimum_operations, maximum_operations]，或单条时须 rank-1 |
| 组数 | `groupOccurrences = gcd(各 load 的 count)`——**重复单元数，不是 for 次数**（已知循环次数已先乘进 count）。每组内容 = 各 load × (count/gcd) 条 |
| 公式 | `setup += kernel_startup`（一次）<br>`continuousLoadGroup += 组数 × (iteration_startup + 组内字节 / bytes_per_cycle)` |
| 基线数值 | kernel_startup **546.076851852** cycle；iteration_startup **12.791269535** cycle；bytes_per_cycle **83.262254929** B/cycle（执行包络，含地址生成/发射/流水稳态，非裸 HBM 带宽）；操作数区间 [1, 4] |
| 参数来源 | profile `simd.memory.independent_loop_load_group.*`；证据：`bench_simd_memory_groups.py`，2/3/4 流循环 + 19/37 迭代单流对照 |
| 记账 | 覆盖量记入 `layout.loadBytes/loadOperations/loadWarpInstructions`（**不记 loadShapes**，故不影响 D 的 count 扣减，只扣 D/E 的字节与操作数余量） |
| 代码 | StageCostModels.cpp:1696–1762 |

---

## 3. 机制 C：布局组曲线（按完整指纹查表）

| 项 | 内容 |
|---|---|
| commit | `d4667dab9` |
| 管哪种访问 | **机制上不限体型**——key 是完整布局指纹，任何体型理论上都可查；实际能否认领取决于 profile 里有没有该 key。**基线数据只有 strided_wide 三条 key，故实践中连续访存从不被 C 认领** |
| 准入条件 | ① stage 属连续内存类（`isContinuousMemoryStage`：ContinuousTileMemory/Store/ShortLoad/CachePolicyStore，:215）；② **无循环、无同步、无条件分支**（:1846）；③ 相邻同 `store` 标志且同 `layoutKey` 的访问合并成一个组（count 累加）；④ key 在 profile 中存在；⑤ 合并后 count 为整数且 ∈ [minCount, maxCount] |
| 查表 key | `layoutKey() + ":warps=" + (SIMD?0:activeWarpCount) + ":factor=" + (SIMD?1:F)`；`layoutKey()` = `布局类:r=秩:row=行字节:stride=行距:rows=行数:bits=位宽:masked=:shape=:strides=`（StageRouteCostModel.cpp:596） |
| 公式 | `cost = first_system_cycles + (count − 1) × incremental_system_cycles`（首条冷启动 + 后续流水增量，`StageMemoryLayoutCost::evaluate`，:3294） |
| 基线数值 | 仅 3 条（16×16 fp32 masked strided_wide，row=64 B, stride=8192 B）：<br>load：warps4/F1 133.49，warps8/F2 145.77，warps16/F4 168.13（first=incremental）<br>store：warps4/F1 first 90.34 / inc 31.19；warps8/F2 first 120.90 / inc 55.91；warps16/F4（见 profile） |
| 参数来源 | profile `layout_memory.load / store`（key 即完整指纹）；证据 `950pr_simt_layout_memory_*` |
| 记账 | 命中计入 `continuousLoadGroup / continuousStoreGroup`，覆盖量记入 `layout.loadShapes/storeShapes[(bytes, elements)]` —— **D 按此扣减 count** |
| 代码 | StageCostModels.cpp:658–673（查表）、1835–1883（合并与记账）、3294（evaluate） |

---

## 4. 机制 D：连续内存字节桶曲线（按 (字节, 元素) 查表，两道门）

| 项 | 内容 |
|---|---|
| commit | `d4667dab9` |
| 管哪种访问 | **连续访存**，且拿到了 §0.3 门票（`tt.make_tensor_ptr` 完美嵌套）、FP32（`bytes == elements×4`） |
| 门 1（分区） | `blockPointerCanUseFlatMemoryCurve` 不通过 → 根本不进 `continuousLoadCounts/StoreCounts`，D 无表可查 |
| 门 2（计价） | 按当前超块因子查 `curves[factor][bytes]`，**只在实测字节区间内插值**；区间外 `interpolateBoundedOperationCost` 返回空（:864，注释 *"only evidence inside the measured byte interval"*）→ 放弃认领。注意：表的键是**字节**（1024 指 1024 B = 256 个 fp32，勿与 1024 个元素=4096 B 混淆） |
| 公式 | `resource += Σ availableCount × cost(bytes)`（字节轴区间内线性插值）<br>`startup = max(startupCurve(bytes))`（一次性，取 max 不累加）<br>`availableCount = count − C 已覆盖量`（layout.loadShapes） |
| 基线数值 | **只有 1024 B 单点**：SIMD load 16.834821429 / store 89.272321429 cycle（factor 1）；SIMT 另有 factor 1/2/4 的 1024 B 点。区间退化为 [1024,1024]，其余字节全部拒收 |
| 参数来源 | profile `continuous_memory.load/store_group_operation_system_cycle_measurements[factor][bytes]`（+ `*_startup_*`），符号名解析自 microbench（scope：`single_aiv_factor_1_224_logical_programs_4_waves_operation_count_4_to_32`） |
| 记账 | 计入 `continuousLoadGroup / continuousStoreGroup`；覆盖的 operations/bytes/warpInstructions 供 E 扣减 |
| 代码 | StageCostModels.cpp:1885–1932（mapContinuousCurves）、864（区间插值）；分区侧 StagePartitioner.cpp:1092/1136（建表）、1683（门票） |

---

## 5. 机制 E：解析式兜底（一切未覆盖余量）

| 项 | 内容 |
|---|---|
| commit | `d4667dab9` |
| 管哪种访问 | A–D 都没认领的 load/store（任意体型）；此外它是循环内连续访存的**默认价格**（C/D 数据覆盖极窄、B 条件苛刻） |
| SIMD 公式 | `load += 未覆盖字节数 / 167.557775299 + 未覆盖操作数 × 105.309240226`<br>`store += 未覆盖字节数 / 213.125675219 + 未覆盖操作数 × 90.954331652`<br>（字节按 MTE2/MTE3 通道折算带宽 + 每操作启动；另有 RowwiseReduction 追加 load 启动摊薄、循环搬运启动进 setup 两个特殊摊薄分支，:1962–1979） |
| SIMT 公式 | `load += 未覆盖 warp指令数 / load_warp_rate`，store 同理；warp 指令数 = `ceil(元素数/32)`。4-warp 基线速率：load **0.065788805**、store **0.263879445** warp_instruction/cycle |
| 参数来源 | SIMD：profile `simd.memory.vector_mte2/mte3_bytes_per_system_cycle` + `load/store_operation_startup_system_cycles`（解析自 `simd.memory.load/store.throughput`，4/32/128 KiB 连续流 startup+byte 联合拟合）；SIMT：`simt.memory.load/store_throughput_measurements[warps]`（`simt.gm.*.throughput.warps_N`） |
| 代码 | SIMD 侧 StageCostModels.cpp:1960–1991；SIMT 侧 :2106–2112（generated LDS/STS/LDK/STK 流亦在此之后按各自速率计） |

---

## 6. Stage 汇总公式（A–E 结果如何合成周期）

`estimateStageBase`（StageCostModels.cpp:2798）按 stage 类别合成，`count = iterationCount`：

**ContinuousTileMemory / Store（:2811）**，三分支：

1. 有实测组曲线（continuousLoadGroup/Store > 0）：
   `stageCycles = setup + count × max( 实测组 + load + store + sharedLayout , 其它执行资源 , issue )`
2. 无组曲线、SIMD 且可重叠：
   `stageCycles = setup + count × max( scalar + predicate + max(load, store, vectorExec) , issue )`
   —— load 与 store 之间也是 max（MTE2/MTE3 并行通道）
3. 其余（如 SIMT）：串行体 `setup + count × serialBody`（各项求和后与 issue 取 max）

**IndependentPipelinedLoop 的 SIMD 分支（:2838）**（整循环一个 stage 时的形态）：

```
stageCycles = setup + count × max( loadService, storeService, vectorService,
                                   vectorService+irregularTail, atomic, scalar+predicate )
loadService  = load + continuousLoadGroup + indirectLoadGroup + sharedLayout
storeService = store + continuousStoreGroup
```

SIMT 超块因子 F>1：实测包络"已覆盖 F 组"，各资源项乘 F 后复用上述公式（:2768–2790）。

**实例（pytest_relu_f32_2x4096x8，基线预测 4.628 µs vs 实测 4.361 µs，+6.1%）**：
整循环一个 IndependentPipelinedLoop stage，iterationCount=32（constexpr 32768/1024，精确）；
每迭代 2 条 4 KiB 连续 FP32 load + 1 条 4 KiB store。
- load → **B** 认领（gcd=1，1 组 2 条 ∈ [1,4]）：`12.79 + 8192/83.26 ≈ 111.2 cycle/迭代`；
- store → **E** 兜底：`90.95 + 4096/213.13 ≈ 110.2 cycle/迭代`；
- **C** 不认领（指纹无 key）、**D** 两道门都不过（非 make_tensor_ptr；4096 ∉ [1024,1024]）；
- 汇总 `32 × max(111.2, 110.2, 向量计算)` + setup/边界 ≈ 4.6 µs。
准确性来源：次数形状全静态精确；B 的校准几何与 kernel 同构（独立循环内 1–4 条平坦 FP32 流）；
load/store 两服务几乎相等使 max() 结果对"谁主导"不敏感；32 次迭代摊薄 546 cycle 启动。

---

## 7. 其它相关机制（简列，非 A–E 主线）

| 机制 | 位置 | 管什么 |
|---|---|---|
| SIMD 流水 strided store 组 | :1764–1814 | masked StridedShort FP32 store 整组包络（按行距查表，first + (n−1)×增量 + 尾行惩罚） |
| masked-strided-copy floor | :486–557 | IndependentPipelinedLoop 中 1 load+1 store 成对、一侧 StridedShort 一侧 ContiguousWide 的拷贝下限（对 load/store 资源取 max，A 已认领则让位） |
| SIMD 不规则 tile 补全 | :558–657 | masked / 尾部不对齐 / 小宽 tile 的 completion 下限（补字节计数看不见的流水收尾） |
| masked RMW / 计算地址标量化 / 控制流地址标量化 | :1992–2058 | 特殊地址形态的 SIMD 补全项 |
| dot 累加器共享布局交换（SIMT） | :1819–1835 | `dotStoreSharedCurve`：FMA→row-major 布局转换曲线 |
| store 足迹服务 floor | :701–733 | `store_footprint_service`：整 tile 写服务的下限 |
| 间接访存（indirect/segmented）整套 | :1456–1687 | loaded-index 依赖访存（本文不计） |

另注：枚举 `ContinuousShortLoad` / `CachePolicyStore` 在基线已**不再被赋值**（StagePartitioner 无赋值点），仅存于兼容性 switch。

---

## 8. 基线 profile 数据覆盖一览（950PR）

| 曲线 | 覆盖 | 缺口 |
|---|---|---|
| B `independent_loop_load_group` | 1–4 条平坦 FP32 load 流，83.26 B/cycle | 仅 SIMD；仅 load |
| C `layout_memory` | strided_wide 16×16 masked fp32 ×3 factor | **无任何连续指纹 key** |
| D `continuous_memory` | **1024 B 单点**（SIMD F1；SIMT F1/2/4） | 其余字节全靠 E |
| E MTE2/MTE3 | 167.56 / 213.13 B/cycle + 每操作启动（4/32/128 KiB 联合拟合） | —— |
| SIMT gm 速率 | warps 1–32 分档（4-warp：load 0.0658 / store 0.2639 warp_instr/cycle） | 顺序旋转循环口径，不含随机/聚并 |

**结构性事实：基线对"循环内连续访存"的实际定价者是 B（load）+ E（store/load 兜底）。**

---

## 9. 变更记录

| commit | 日期 | 机制 | 改动 | 影响 |
|---|---|---|---|---|
| `d4667dab9` | 2026-10-29 | A–E | 本文档基线（revert loop-overlap model） | — |
| `758de6357`–`bad6509a` | 2026-11 | B/E/SIMD 侧 | 连续访存标定接入、FormS 系列修正、SIMD 物理公式替换 te 分段、SIMD load/store 合并（mask 指示项）、命名重构（streams→repeated, pipelined→cross_iter_pipelined） | 见 `_contiguous_summary_v7.md` 与 git log；行号相对 d4667dab9 已漂移 |
| （本次） | 2026-11-09 | **新增 Swide 机制** | StridedWide 物理因子重拟合标定（SIMT 五段 rows×W + SIMD K=4/2），见 **§10**；profile 新增 `strided_wide_completion`，**先于机制 C 认领无 mask StridedWide**，C 对已认领访问让位 | SIMT 两槽位 14.79/12.40 FAIL → 9.64/9.52 PASS；SIMD 6.44/8.16 PASS（裁剪后） |
| （本次） | 2026-11-09 | Swide 集成 | `Models/StridedWideCostModel.{h,cpp}`（选择+求值，公式形状入 C++、数字入 profile，仿 StridedShort #2716 结构）；`mapStridedWideMemoryCompletion` 挂入 `mapWorkload`（`closedFormClaimed` 收据体系，机制 C/masked-copy/RMW 自动让位）；profile 950PR 两段系数；UT 黄金用例（`gen_integration.py` 生成，验证 switch 项序与 profile terms 一致） | 特征口径与拟合管线逐位 parity（2196 点全特征零偏差）；域外一律让位兜底 |
| | | | （以后改动在此追加） | |

---

## 10. StridedWide 物理因子重拟合（SIMT 五段 + SIMD 两段）

### 10.1 背景与目标

StridedWide（2D、恰一轴 stride==1、`rowBytes()>=32`、无 mask、inner/尾轴连续）此前只有 kaixin PR 里的
`layout_memory` 逐指纹曲线（覆盖极窄：仅 16×16 masked fp32 一族），开放式的
`simt_load/store_swide_r2_inner` 闭环槽位在早期标定中 SIMT MAPE 14.79%/12.40% 双 FAIL，未合入。

本轮在**同一原始数据、同一过滤口径**下重拟合，允许更细分段与更复杂因子，但**全部因子要求有明确物理含义**：
SIMT 用每线程/每 warp 工作量与内存系统量（线程是 SIMT 才有的概念）；SIMD 用纯几何/字节/line 因子（不含任何线程量）。

### 10.2 数据与口径（与早期标定完全一致）

| 项 | 值 |
|---|---|
| 原始数据 | `WIDE2_final.jsonl`（store，13667 行）/ `LOADV1.jsonl`（load，2124 行），本机 `D:\work_wj\mask_vs_nomask\`（**不入 git**，见 swide_phys/README） |
| 过滤 | `valid && var==0 && spread<=0.2`，去重 |
| 槽位点数 | SIMD load 551（rows≤4: 260 / >4: 291）、SIMD store 594（130/464）、SIMT load 463、SIMT store 588 |
| 验收口径 | 槽位标定域内 in-sample MAPE < 10%（与 StridedShort #2716 相同标准） |

### 10.3 分段（全部 TTIR 可计算：`rowCount()` + warp 数）

| 模式 | 段 |
|---|---|
| SIMD | rows≤4 / rows>4（load）；rows≤2 / rows>2（store）——沿用交付版 K |
| SIMT | **rows=2 / rows=8 / rows=64 且 W≤2 / rows=64 且 W∈[3,8] / rows=64 且 W>8**（rows3_W2 五段） |

### 10.4 因子族（SIMT 67 个 / SIMD 45 个，全带物理标注）

| 族 | 代表因子 | 含义 |
|---|---|---|
| 线程工作量（仅 SIMT） | `elem_per_lane`=n/(32W)、`iters`=ceil(n/(32W))、`h`=超出单轮元素数、`bytes_per_lane` | 每线程搬多少、跑几轮 |
| warp 结构（仅 SIMT） | `bytes_per_warp`、`line_per_warp`、`seg_per_warp`、`invW` | 每 warp 摊到的字节/line/段 |
| 内存系统 | `L128…L4096`（各粒度触达块数＝cache line/L2 slice 分散度）、`mx{64…4096}`（单块最多元素）、`S{32…4096}`（同块集中度 Σk²）、`sectors`、`mis32` | 地址离散度与块密度 |
| 段几何 | `rows`、`cseg`=run/eb、`cseg_mod32`（**每段元素数 mod 32＝lane 覆盖锯齿**，同字节下 f16 与 int8 差 4.2× 的元凶）、`duty`、`span`、`gap` | 描述符遍历与稀疏度 |
| W 拐点 hinge（仅 SIMT） | `minW/maxW{2,4,8,16}`、`r16invW`、`n2W` | 小 W 延迟区 / 大 W 吞吐区折点 |

### 10.5 结果（物理因子池，cap=18，CV-guarded ridge + Nelder-Mead）

| 槽位 | 交付版 | 全统计池 | **物理因子池** | 判定 |
|---|---:|---:|---:|---|
| load·SIMT | 14.79% FAIL | 8.74% | **9.64%** | PASS |
| store·SIMT | 12.40% FAIL | 8.12% | **9.49%** | PASS |
| load·SIMD | 5.28%（含线程味 `iters`） | — | **4.00%** | PASS |
| store·SIMD | 8.86% | — | **8.16%** | PASS |

SIMT 分段明细（物理池）：
- load：rows=2 3.49 / rows=8 13.12 / 64,W≤2 18.57 / 64,W3–8 13.21 / 64,W>8 7.30
- store：9.16 / 9.07 / 10.92 / 10.98 / 9.88

符号检查：全部系数中仅 3 处 mop 级负号（mx1024/-0.11、invW/-183、mx4096/-0.016）。

交付版按分组 cap 裁剪（SIMD load 6 / SIMD store 18 / SIMT load 14、store 8 /
masked load 10、store 8），对应模式级 MAPE：SIMD load 6.44%、SIMD store 8.16%、
SIMT load 9.64%、store 9.52%、masked load 9.36%、store 9.73%，全部 <10%。

### 10.6 如实声明的边界

1. **性质是标定曲线，不是物理律**：dtype 留一 OOS 约 21%/16%（SIMT），与原 K8 方案同一水平
   （17.3/20.1）——C-vs-lane 锯齿物理（README 5.8"本章最大未解问题"）线性不可迁移。
   因此 C++ 侧沿用 profile 曲线"**仅在实测域内消费**"的契约（实测域：rows∈{2,8,16,64}、run∈{32,128(,2048)}、
   gap∈{2,32,128,2048,8192}，rows=16 的扩展标定见第 6 条、W∈{1..64}、inner、run≥32）。
2. **顽固段**：`rows=64,W≤2`（台阶物理未解，load 段内 18.6%）与 `rows=8`（13.1%）单段仍 >10%，
   槽位 blend 达标。
3. 认领优先级：本机制**先于机制 C（layout_memory）认领 StridedWide**；C 对已认领访问让位。
   准入范围：**无 mask 的访问**，以及 **mask 动态全 true（`activeElementRatio==1.0`）**的访问。
   masked 数据：store 用 WIDE2 var=1（6872 点，SIMD/SIMT 网格与 var=0 对等）；load 为本轮新扫
   （主网格 1034 有效 + SIMT rows=16 补测 315×2）。
   - **SIMD**（load/store）与无 mask **共用槽位**：配对比值中位 1.000、p10–p90 0.999–1.004、
     99.2% 在 ±5% 内；联合拟合相对独立仅差 0.14–0.2pp。
   - **SIMT** 使用**独立的 masked 槽位**（16 个：store/load × rows {2,8,64} 的 W 分档 + rows=16 的
     W 分档）。原因：mask 因子方案（联合 + is_masked/交互项）在 SIMT 上不成立——store 联合后
     rows=64,W≤2 段 V0 17.1%、load 联合→V1 11.6%/最差段 17–22%；独立槽位 store 9.73%、load 9.36%。
   **不认领**：① 静态可证明部分覆盖的 mask（切在连续段内部，产生 sub-line 部分写/RMW，拟合没见过）；
   ② SIMT rows 不在 {2,8,16,64}、rows=16 且 run>128B、run/gap/span 出网格、eb∉{1,2,4} 的访问。
   这些继续走 kaixin 的 masked 指纹曲线 / 解析兜底。
   **kaixin 的 6 条 `strided_wide` 指纹键**（SIMT、masked、fp32、16×16、run 64B、stride 8192B、
   warps 4/8/16 × factor 1/2/4）已被 rows=16 masked 槽位覆盖（run 32–128B、gap 含 8128），
   由本机制先行认领；100-case 中 `solve_tril_*` 的 4 例即此几何（接管后 2 例变好、2 例小幅变差，
   均在 10% 内）。
4. **单流认领原则**：闭式标定的是"每次迭代一条不间断流"的稳态单价（rotate-loop，单操作）。
   同一方向有多条宽访存、或 count>1 时，硬件会聚合处理，按"N × 单流价"求和会偏高——实测反例
   `pytest_permute_reshape_fused_f32`（每迭代 2 条 64×8 流：2×151.4=302.8 cycles，而基线带宽
   兜底 111.1 更接近内核边际）。因此 `mapStridedWideMemoryCompletion` 只认领 **count==1 且该方向
   宽访存总数==1** 的访问，多流字节交回带宽兜底。
5. **加密 gap 验证（2026-11-11）**：为排除"离网 gap 插值"嫌疑，补测 SIMD 356 点（fp32 全 gap
   梯队 96–4096 + f16 三点，rows 2–256）。结论：原系数在加密点上泛化良好（load 2.99/7.08%、
   store 2.62/5.96%），且 `permute_reshape` 的几何（rows64/run32/gap352）实测 146.0/146.6 ns
   与模型 148.0/151.4 ns 仅差 1.3%/3.3%——偏差不是插值，而是多流聚合（见第 4 条）。
   同时 `storeLayoutService`（profile 的 `store_footprint_service`，kaixin #2528 引入）的准入
   「非 flat + unit inner stride」恰好等于 StridedWide 域，因此它**跳过已认领访问**——复刻其原有语义
   （`continuousStoreGroup==0` 才启用；kaixin 的 layout 曲线写 `continuousStoreGroup`，故从不叠加）。
6. **16KB 行距扩展（2026-11-11）**：`solve_tril_*_h64` 的 block ptr 行距 16384B（gap 16320B），
   超出原 gap 端点。补测 rows=16（run {32,64,128} × gap {16256,16320,16384}）：
   masked SIMT（W 1–64）重拟合后 store/load 槽位 9.73%/9.36%；SIMD 共享槽位直接外推，
   masked store/load 4.36/4.88%、unmasked 4.07/5.37%（27 点/方向，单点最大 10.5%），
   因此**不重拟合 SIMD 系数**。准入规则改为：`rows==16 && run<=128 && (simd || masked)`
   时 gap 闭区间 [2, 16384]B，其它家族仍为 [2, 8192]B。h64 的 4 个 solve_tril 在选中路线
   （stage 4/6/8 为 SIMD）下由本机制认领，替换原来的解析兜底。

### 10.7 复现

```bash
# 数据: D:\work_wj\mask_vs_nomask\{WIDE2_final,LOADV1}.jsonl（.gitignore 惯例，不入库）
python swide_phys/refit_simt.py eval-baseline   # 复核交付版 14.79/12.40（逐位一致）
python swide_phys/refit_simt.py fit-phys --cap 18   # 本表结果; 系数输出 phys_cap18.json
```

系数定稿：`swide_phys/phys_cap18.json`；运行日志 `swide_phys/fit_log_cap18.txt`。
