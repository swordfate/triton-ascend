# CostModel 分类体系复盘与扩展路线图

> 依据：4 份并行子代理对 31 个目标 kernel 的精读报告（VLLM×10、SGLang×5 + Liger×2、FlagGems×3、FBGEMM×6），加上对本仓库 Route Model 全部源码的走读。
> 仓库根目录的 `simt_costmodel_kind_coverage_and_taxonomy.md` 仅作交叉参考（其"公理 A/B/C"可取，7 族划分与判定树本文有不同结论）。
>
> 本文回答两件事：
> **第一部分**：现有 8 族 20 kind 是否合理？不合理在哪？怎么组织更有逻辑？
> **第二部分**：要让 costmodel 支持更多 kernel，下一步改哪些代码？给出一个可以立刻开工的完整示例。

---

# 第一部分：分类体系诊断与重组提案

## 1. 证据先行：31 个 kernel 暴露了什么

把四份报告的缺口按**机制**归并统计（同一机制在多个算子出现只记一次出现面，括号内是代表算子）：

| # | 缺失机制 | 出现面 | 代表证据 |
|---|---|---|---|
| G1 | **散射写 / 数据寻址 store** | 8+ | deepep/deepgemm src2dst、masked_select write_back、lora_expand 输出、fused_moe 输出、bias restore |
| G2 | **Scan / 前缀和（含块间 carry）** | 10+ | FlagGems cumsum 全家族、masked_select 两遍方案、FBGEMM fused_single_block/padding_cumsum 的 `tl.cumsum`+二分 |
| G3 | **搜索建索引**（标量二分、向量化二分、指针追逐） | 4 | SGLang seg_indptr（while 二分）、FBGEMM 固定 64/32 轮二分、deepgemm 两级 chase |
| G4 | **跨 program 协调**（atomic ticket、elected-CTA、debug_barrier、冗余副本换同步） | 4 | masked_select 大 N 路径、FBGEMM fused_single_block |
| G5 | **无 dot 的向量矩阵积**（manual qk/pv，M=1） | 2 | VLLM `_fwd_kernel_stage1` 用 `tl.sum(q*k)` 替代 tl.dot |
| G6 | **在线 softmax / 流式归一化**（递推+归约+SFU 融合） | 4 | decode attention ×3、instance_norm Welford |
| G7 | **Ragged 动态边界属性**（设备端 trip count、per-segment 偏移、环绕寻址） | 5+ | jagged_bmm、lora_expand、silu(num_tokens)、fused_moe padded EM |
| G8 | **位操作强度分级**（dtype cast ≪ bit unpack ≈ interleave/split/join ≈ rope 置换） | 4 | mx4 解包/量化、int2 unpack、rope 索引置换 |
| G9 | **SFU 强度分级**（exp/rsqrt/fp64-exp2/philox RNG） | 5 | rope 三角函数链、silu sigmoid、mx4 f64 exp2、随机舍入 |
| G10 | **小表 LUT gather**（16 项常驻表 vs 大空间离散 gather） | 2 | mx4 lookup_table |
| G11 | **arg 归约**（带索引追踪的归约） | 1 | sample_recovered_tokens 的 `tl.argmax` |
| G12 | **load 侧 cache policy** | 3 | conv1d `.ca`、instance_norm evict_first |

同时反向验证：现有 20 个 kind 里 `cache_policy_store` 在 31 个样本中仅 1 处命中（Liger `_mask_fwd_kernel` 的 `.cs`），`continuous_short_load` 与 `cache_policy_store` 共享同一公式——**冷类与热缺并存，说明枚举不是按"公式差异"生长的**。

## 2. 诊断：三个病灶

### 病灶一：三个互不正交的分类轴混在同一层枚举里

看现有族的隐含轴：

| 族 | 实际用的轴 |
|---|---|
| Dispatch ×2 | **来源轴**：编译器注入 vs 其它 |
| ScalarIssue/Math/IndexGen/PredicateMask/LoopPredicate、Continuous×4、Indirect×2、Cube×2、ConversionPack | **资源轴**：这段工作的主导资源是什么 |
| IndependentPipelinedLoop、LoopCarriedRecurrence、RowwiseReduction | **结构轴**：逐迭代成本怎么组合成总时间 |

于是一定出现无解问题："独立循环 + 连续 load" 归谁？"循环携带的标量更新"归谁？现在的答案是一条**经验优先级链**（dot > reduction > conversion > loop > indirect > contiguous > scalar），但这条顺序没有推导依据，换一组算子就要重排。

### 病灶二：kind 与成本公式脱钩（粒度错位）

Registry 的真实语义是：`(StageMode, Kind) → 组合策略模型类`。检查注册表可以发现：

- `ScalarIssue/ScalarControl/ScalarMath/IndexGeneration/PredicateMask/LoopPredicate` 六个 kind → **同一个** `SIMD/SIMTScalarStageCostModel`，同一个 serialBody 公式；
- `CubeRoofline/TinyCubeRoofline` 两个 kind → 同一个 Cube 模型类、同一个公式；
- `ContinuousTileMemory/Store/ShortLoad/CachePolicyStore` 四个 kind → 同一个访存公式。

即 **20 个 kind 里有一半不产生新的公式形状**。按"两个 kind 存在当且仅当主成本公式可区分"的标准，它们是伪区分；而真正需要新公式的机制（scan 的 log 步+carry 扇出、二分搜索的依赖 gather 链、散射写的写合并失败、协调同步的不可流水性）反而一个 kind 都没有。**粒度错位是"种类太少"与"逻辑混乱"的共同根源：该细的地方粗（缺机制），不该细的地方细（同公式多标签）。**

### 病灶三：整族缺席，而不是缺几个类

G1–G12 里，散射写、扫描、搜索、协调四类不是"某个 kind 没覆盖到"，而是**整个维度没有轴**。它们无法通过在旧体系里加 kind 解决，因为旧体系的任何位置放进去都会破坏优先级链。

## 3. 重组提案：两轴一格 + 属性旗标

### 3.1 三条正交维度

```
轴 A「组合结构」(structure)：逐迭代资源周期如何合成总时间
     —— 直接决定模型类与公式形状
轴 B「主导资源」(resource)：关键路径上哪种资源的速率说了算
     —— 决定 mapSIMD/mapSIMTWorkload 走哪个分支、用哪张价目表
轴 C「属性旗标」(flags)：一切不改变公式形状的差异
     —— 放进 StageModelFeatures，只调参数与合法性
```

**Kind := (A,B) 平面上真实存在且计价不同的格子。** 枚举由这张表机械导出，不再手拍优先级。

### 3.2 新枚举：7 族 14 类

```
族 0  Scheduling（调度壳，来源轴特例，保留现状）
  1. auto_blockify_dispatch
  2. auto_blockify_loop

族 1  Memory（直线/循环内的访存段）
  3. continuous_memory        ← 合并 tile_memory/tile_store/short_load/cache_policy_store
                                 flags: {direction, short_tile, cache_policy(load|store)}
  4. indirect_gather          ← 合并 indirect_scalar_memory/indirect_gather_memory（读侧）
                                 flags: {indirect_depth(指针追逐级数), lut_small_table}
  5. indirect_scatter         ★新增：数据寻址写（src2dst/masked_select/moe/lora 输出）
                                 flags: {permutation_unique(免原子), padded_offset}

族 2  Dataflow（跨 lane / 跨迭代的数据移动主导）
  6. reduction_rowwise        flags: {arg_reduce(G11), welford_streaming}
  7. scan_prefix              ★新增：tl.cumsum 家族 + 块间 carry 扇出
                                 flags: {segmented, carry_via_launch|carry_via_atomic}
  8. carried_recurrence       flags: {sliding_window_stencil(conv1d),
                                      online_softmax_fusion(G6), persistent_state_bytes}
  9. search_lookup            ★新增：二分/查表建索引（递推携带访存地址）
                                 flags: {fixed_iterations(32/64), vectorized}

族 3  Matrix（tensor core）
 10. cube_roofline            ← 合并 tiny_cube_roofline
                                 flags: {underfill_ratio, quantized_dot,
                                         grouped_routing(moe), jagged_segments}

族 4  VectorCompute（无 dot 的向量计算）
 11. vector_roofline          ★新增：吸收 conversion_pack + manual_dot_reduction(G5)
                                 flags: {sfu_intensity(G9), bitmanip_intensity(G8),
                                         rng_philox, interleave_rearrange}

族 5  Coordination（跨 program 协作）★新增族
 12. sync_coordination        flags: {atomic_ticket, elected_cta, replicated_scratch}

族 6  Scalar 兜底
 13. scalar_issue             ← 合并 issue/math/index_gen/predicate_mask
                                 flags: {divmod_heavy, fp64_precision}
 14. scalar_control           ← 合并 scalar_control/loop_predicate
                                 flags: {program_role_dispatch(FBGEMM pid==0/rope 三路指针)}
```

20 → 14，但机制覆盖从 16 种扩到 ~30 种（旗标计入）。**减少的是伪区分，增加的是真机制。**

### 3.3 判定树（替代经验优先级链）

按"对关键路径的约束强度"从强到弱提问，每步唯一出口：

```
Q0 编译器注入的调度壳（ta.auto_blockify_v1*）？ ──是→ 族0
Q1 有跨 program 协作原语（atomic/barrier/elected-CTA/私有副本广播）？
                                                ──是→ 族5 sync_coordination
Q2 关键路径被跨 lane/跨迭代数据移动支配？
     ├─ 塌缩成标量（tree/shuffle 归约）          ──→ 6 reduction_rowwise
     ├─ 前缀传播（cumsum/segmented scan）        ──→ 7 scan_prefix
     ├─ 本迭代消费上一迭代（含滑窗/在线softmax）──→ 8 carried_recurrence
     └─ 递推携带的是访存地址（二分/查表）        ──→ 9 search_lookup
Q3 tt.dot 在关键路径上？                        ──是→ 10 cube_roofline(+underfill)
Q4 地址数据相关（回溯切片到达 load/gather）？
     ├─ 读侧                                    ──→ 4 indirect_gather(+depth)
     └─ 写侧                                    ──→ 5 indirect_scatter
Q5 访存字节主导且地址可证连续？                 ──是→ 3 continuous_memory
Q6 向量计算主导（元素级数学/转换/位操作）？     ──是→ 11 vector_roofline
Q7 兜底：均匀分支/程序角色分发主导？            ──→ 14 scalar_control
    否则                                        ──→ 13 scalar_issue
```

`requires_split` 规则泛化为一句话：**一个 Stage 的 op 图让 Q2 与 Q3 同时答"是"、或 Q2 内部两类同时成立时，必须拆分**（与现状三角域里 dot+carried 报错的语义一致，只是判据从枚举特例变成树冲突）。

### 3.4 为什么工程上便宜：模型类几乎不用动

对照注册表：新体系的组合策略恰好映射到**现有 9 对模型类 + 2 对新类**：

| 新 kind | 复用/新增的模型类 | 改动量 |
|---|---|---|
| continuous_memory / cube_roofline / scalar_issue / scalar_control / dispatch×2 | 原 Continuous/Cube/Scalar/Dispatch 模型原样复用 | 只改 `supports()` 清单 |
| indirect_gather/scatter | 原 IndirectMemory 模型（scatter 侧重载 store 分支即可） | 小 |
| reduction_rowwise / carried_recurrence / vector_roofline | 原 Reduction/Recurrence 模型；vector_roofline 用 IndependentPipelined 的 overlap 公式 | 小 |
| scan_prefix / search_lookup / sync_coordination | **需新写 3 对模型**（这是真正的增量工作） | 中 |

也就是说，重组的主要工作量在**枚举、分类器（StageKindClassifier）、工作量归类（accumulateOneOperation）**三处，代价公式大部分继承——这正好是可以并行的切分线。

---

# 第二部分：下一步改哪些代码——四个可并行的工单 + 完整示例

## 4. 总路线：两条轨道

```
轨道 A（不动体系，先扩大覆盖）：
   大量目标 kernel 根本不走 Stage 白名单三域，直接落进聚合公式 fallback。
   把 fallback 里已知"未定价"项补上（scan、scatter、arg-reduce），
   覆盖面立刻从 3 个域扩大到大部分目标算子。改动集中在
   SimdSimtCostModel.cpp 一个文件。→ 工单 C，最快见效。

轨道 B（动体系，长期正确）：
   落地第 3 节的两轴一格重构 + 新域白名单。涉及枚举/分类器/模板/
   锚点/profile 五处，彼此接口清晰。→ 工单 A/B/D 可三人并行。
```

## 5. 工单拆分与文件触点

### 工单 A：枚举与注册表重构（改"骨架"）

| 文件 | 触点 | 内容 |
|---|---|---|
| `StageRouteCostModel.h` | `StageModelFeatures` (:48) | 新增 flags 字段：`cachePolicy`、`dynamicTripCount`、`raggedBounds`、`indirectDepth`、`underfillRatio`(double)、`sfuIntensity`、`bitmanipIntensity`、`argReduce`、`quantizedDot`、`groupedRouting` |
| `StageCostModels.h` | enum (:27) | 20 → 14（保留旧名作为 deprecated alias 一版过渡）；`stringify/parse` (:571/:617) 同步 |
| `StageCostModels.cpp` | 注册表 (:722) 与 `verifyComplete` (:770) | 合并 supports 清单；新增 ScanPrefix/SearchLookup/SyncCoordination 三对模型骨架（先实现公式，数值用保守种子） |
| `StagePartitioner.cpp` | `accumulateOneOperation` (:220)、`StageFeatureAnalysis` (:1565)、`StageKindClassifier` (:1697) | 按 3.3 判定树重写派生链；flags 从 op 图抽取（如 arg-reduce 看 reduce region 是否 select 索引、sfu 看 math.exp/log/rsqrt 计数密度） |

### 工单 B：新锚点 + 新白名单域（改"入口"）

以**散射压缩域（ScatterCompaction）**为例——它一次性覆盖 G1/G2/G3/G4 四大缺口，对应 masked_select、src2dst×2、fused_padding_cumsum 等 5+ 目标算子。完整步骤见第 6 节。

### 工单 C：聚合公式补价（改"兜底"，收益最快）

`SimdSimtCostModel.cpp::estimateSimdSimtCandidatesImpl` fallback 段（:2481–2878）：

1. 删掉 `"scan_template_ranking_uncalibrated"`（:2603）：scan 已有 `shuffleLaneSteps` 计量，先用 shuffle 价 + 串行 carry 修正定价；
2. 给 store 侧加间接分支：现在 `storeBytes` 一律除 MTE3 带宽（:2564），应识别 `loadedIndexDependentMemoryOp` 的 store 变体，改走事务率 + 依赖延迟；
3. arg-reduce 计量：`analyzeSimdSimtFeatures` 里 reduce 已计数（:2024），补一个"region 含 select/index cast"的探测，把 argmax 按 2×reduce 计费；
4. 锚点侧同理：`SimtAnchorFeatureSummary` 已有 atomic/histogram/cumsum facts 结构（`toAtomicFactsJSON` 等 :1411），把它们接进 mixed 锚点计价而不是只进 JSON 报告。

### 工单 D：profile v18 校准（改"价目表"）

`david_v100_simd_simt_v1.json` 需为新机制补种子并升版本号（加载器在 `SimdSimtCostModel.cpp:1051` 钉死版本串，必须同步）：

```json
"simt": { "stage_resources": {
    "scan":  { "carry_steps_per_system_cycle": ..., "note": "CaModel 待标" },
    "search":{ "dependent_gather_latency_system_cycles": ... } } }
```

校准来源现成：`profiles/microbench/data_provider/camodel/` 目录已有实验矩阵脚本。

## 6. 完整示例：新增 ScatterCompaction 域（工单 B 全流程）

目标：让 `masked_select`、`deepep/deepgemm_compute_src2dst`、`fused_padding_cumsum_and_segmented_arange` 这类"计数 → 前缀和 → 散射写"kernel 走精确 Stage 模型。共 7 步，每步给出文件与函数。

**Step 1 — 定义域枚举**（`StagePartitioner.h:24`）：

```cpp
enum class PhaseBoundaryDomain {
  TriangularRecurrence,
  LoadedIndexRowwiseReduction,
  IndirectUnderfilledDot,
  ScatterCompaction,          // ← 新增
};
```

**Step 2 — 特征识别条件**（`StagePartitioner.cpp:1477` 的特征版 `PhaseBoundaryAnalysis::analyze` 加一条，放在 rowwise/dot 判定之前）：

```cpp
// 散射压缩签名：有 scan（或 cumsum 型间接索引生成）+ 有 load + 有 store，
// 且 store 是 loaded-index 依赖（写地址来自数据）。
if (features.dotOps == 0 &&
    (features.scanOps > 0 ||
     features.loadedIndexDependentMemoryOps > 0) &&   // src2dst 无显式 scan，靠间接判定
    features.loadOps > 0 && features.storeOps > 0) {
  PhaseBoundaryPlan plan{PhaseBoundaryDomain::ScatterCompaction,
                         "scatter_compaction", std::nullopt};
  return std::optional<PhaseBoundaryPlan>{std::move(plan)};
}
```

注意与现有 rowwise 条件的先后：rowwise 要求 `reduceOps>0`，本域要求 `scanOps>0 或 scatter`，二者天然互斥；dot 域要求 `reduceOps==0 && dot>0`，也不冲突。

**Step 3 — Phase 状态机**（`assignRootPhaseIds :917` 加一个枚举与 case）：

```cpp
enum class ScatterPhase { Count, Prefix, ScatterWrite };
// 转移探测：含 tl.sum/向量累加根 → Count；
// 含 tt.scan/cumsum 或二分循环根 → Prefix；
// 之后全部 → ScatterWrite。单调前进 + 尾部连续性校验复用现有 closedPhases 逻辑(:1046)。
```

**Step 4 — Stage 模板**（`StageBoundaryAnalysis::analyze :1525` 的 switch 加分支，模板仿照 `partitionIndirectDot :714`）：

```cpp
static StagePartition partitionScatter(const SimdSimtFeatureSummary &f,
                                       const PhaseBoundaryPlan *g) {
  StagePartition p; p.domain = "scatter_compaction";
  StageWorkload remaining = buildKernelStageWorkload(f);
  prependAutoBlockifyStages(p, remaining, f, /*graphHasAb=*/false);
  // count_reduce：RowwiseReduction（吃 reduce/shuffle 账）
  // exclusive_prefix：scan_prefix ★新 kind（吃 scan 账 + carry 扇出）
  // gather_indices：indirect_gather（吃间接读账，depth=2 for deepgemm chase）
  // scatter_write ：indirect_scatter ★新 kind（吃间接写账）
  // 尾部 mask/标量收尾 → scalar_issue
}
```

随后 `attachCompleteOperationOwnership(:1098)` 的 phaseId→stage 映射表加三行——这个函数是纯查表，改动机械。

**Step 5 — 新 kind 进枚举与注册表**（与工单 A 汇合点）：

- `StageCostModels.h` enum 加 `ScanPrefix`、`IndirectScatter`（若走渐进路线，也可先复用 `LoopCarriedRecurrence`/`IndirectGatherMemory` 占位，把"新 kind"推迟到工单 A 落地后）；
- `StageCostModels.cpp` 注册 `SIMD/SIMTScanPrefixStageCostModel`、`SIMD/SIMTIndirectMemoryStageCostModel.supports()` 加 store 侧。**漏注册会被 `verifyComplete(:770)` 当场拦下**——这是安全网，放心增量开发。

**Step 6 — 锚点绑定**（`StagePartitioner.cpp:759 anchorMatchesStage`）：

```cpp
if (stage.costModelKind == StageCostModelKind::ScanPrefix)
  return anchor.kind == SimtAnchorKind::PlainOneDimensionalCumsum;
if (stage.costModelKind == StageCostModelKind::IndirectScatter)
  return anchor.kind == SimtAnchorKind::LoadedIndexDependentMemory;  // 写侧变体
```

`PlainOneDimensionalCumsum` 锚点已存在（`analyzePlainOneDimensionalCumsum :45`），无需新写；只需放宽其识别（当前要求"恰好一个 addf combine"，`tl.cumsum` 降级产物满足；FlagGems 的分块 cumsum 会产生 scf.for 包裹形态，需要在 `buildMixedSimtAnchorPlan` 里允许"scan 在循环体内"的嵌套匹配——这是本工单唯一的算法性新代码）。

**Step 7 — 测试与验收**（`third_party/ascend/unittest/costmodel_ut/`）：

1. 单元：构造最小 TTIR（load → cmp → cumsum → scatter store），断言 domain=scatter_compaction、五 Stage 齐全、`StagePartitionVerifier` 通过；
2. 端到端：用 bench 里的 masked_select TTIR 跑 `select-simd-simt-costmodel mode=report`，确认 report_json 里 `stage_model.applied=true` 且 mixed/all-simd 分数可解释；
3. 回归：跑既有 triangular/rowwise/indirect-dot 用例确认零行为变化（新判定排在旧条件之后或互斥）。

## 7. 并行开工建议

```
人 1（熟枚举）：工单 A —— 先出 14-kind 草案 + alias 映射，registry 骨架先行
人 2（熟管线）：工单 B Step 1–4（域识别/状态机/模板），kind 先用占位别名
人 3（熟公式）：工单 C —— 聚合公式补价，当天可见 coverage 提升
人 4（校准）：  工单 D —— CaModel 跑 scan/search/原子事务率种子
汇合点：A 的正式 kind 就绪后，B 的 Step 5–6 把占位符替换掉；
       D 的数字填进 v18 profile，全链路联调。
```

风险提示两条：一是 `verifyComplete` 要求 20×2 全注册，工单 A 合并 kind 时务必同步清理 `StageKindClassifier` 的 switch（`:1727`/`:1806` 两处逐 kind 校验），否则运行期 mismatch 报错；二是 profile 版本串钉死机制意味着 A/B/D 必须约定同一次合入升级 v18，不能各自单独 bump。
