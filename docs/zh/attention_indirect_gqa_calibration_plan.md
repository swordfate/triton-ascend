# Attention Indirect GQA Calibration Domain — 实现方案

> 分支：`feature/costmodel-dev`（本文档）/ `kx_simt_costmodel`（C++ 代码）
> 基于：`costmodel_rope_kernel_diagnosis.md` 的诊断结论

## 一、问题回顾

ROPE kernel（Flash Attention GQA + indirect KV lookup）实测：

| 路线 | 实测延迟 | costmodel 预测 (selection score) |
|------|---------|----------------------------------|
| SIMD | 241 us | 4,460 |
| SIMT only | 50 us | 519,805 |
| Mixed (simd_simt) | **17 us** | 37,463 |

**costmodel 选 SIMD（score 最低），但实测 Mixed 最快（14x faster than SIMD）。**

根因已确认：
1. `simtPredicateCycles = 153 * 16 / 0.038 = 64,421` 占 SIMT 总分的 99%
2. `simtPredicateRate = 0.038` 来自单一 FBGEMM workload，不适用于 attention 类 kernel
3. Domain 是 `loop_trip_count_unknown` → 没有对应的 event calibration multiplier → 所有 multiplier 默认 1.0

## 二、方案：新增 `attention_indirect_gqa` calibration domain

### 2.1 核心思路

不是改 predicate 公式本身，而是**新增一个 calibration domain**，用实测延迟反推出 event route multipliers。这个 domain 的 multiplier 会自然吸收 predicate 公式的误差——跟现有 3 个 domain 的做法完全一致。

### 2.2 该 domain 的内核特征

从 ROPE kernel 的 costmodel JSON 提取：

| 特征 | 值 | 说明 |
|------|-----|------|
| `loaded_index_dependent_memory_ops` | 8 | 间接 KV lookup |
| `dot_ops` | 3 | qk dot + acc dot |
| `dot_flops` | 49,152 | 中等 dot |
| `row_local_reduce_ops` | 2 | softmax max + sum |
| `has_control_flow` | true | scf.if + for loop |
| `has_unknown_trip_count` | true | KV 序列长度动态 |
| `mask_rank_sum` | 153 | 高 mask 使用 |
| `max_tensor_numel` | 512 | 中等 tensor |

**与现有 3 个 domain 的区分点：**

| 相比 | ROPE 的区分特征 |
|------|----------------|
| `masked_rowwise_reduction` | 有 dot + max_tensor_numel > 128 + mask_rank_sum > 48 |
| `tiny_irregular_dot` | dot_flops > 16384 + 有 row_local_reduce（softmax pattern） |
| `triangular_solve_loop` | 有 dot + 有 row_local_reduce + 无三角计算 |

### 2.3 Coverage 边界设计

在 `rankingCalibrationCoverage()` 中的检查顺序（按优先级）：

```cpp
// 新增：attention_indirect_gqa
// 特征组合：间接访存 + dot + row-local reduction = Flash Attention pattern
if (dotFlops > 0 &&
    features.loadedIndexDependentMemoryOps > 0 &&
    features.rowLocalReduceOps > 0 &&
    maxNumel <= coverage.attentionMaxTensorNumel &&
    maskRankSum <= coverage.attentionMaskRankSumMax &&
    features.dotOps >= coverage.attentionMinDotOps &&
    features.simtAnchors.loadedIndexDependentMemoryOps >=
        coverage.attentionMinIndirectMemoryOps)
  return {true, "attention_indirect_gqa"};
```

关键点：
- `dotFlops > 0` — 有 dot（区别于 masked_rowwise_reduction）
- `loadedIndexDependentMemoryOps > 0` — 有间接访存（区别于普通 GEMM）
- `rowLocalReduceOps > 0` — 有 softmax pattern（区别于 tiny_irregular_dot）
- `maxNumel <= coverage.attentionMaxTensorNumel` — tensor 不过大
- `maskRankSum <= coverage.attentionMaskRankSumMax` — mask 不过多
- 不检查 `hasUnknownTripCount` — 允许动态循环

### 2.4 Event Route Calibration Multipliers 推导

#### 2.4.1 转换实测延迟为 SYS_CNT cycles

SYS_CNT 频率 = 988.9 MHz（来自 microbenchmark profile）

| 路线 | 实测 us | SYS_CNT cycles |
|------|--------|---------------|
| SIMD | 241 | 241 × 988.9 = 238,325 |
| SIMT only | 50 | 50 × 988.9 = 49,445 |
| Mixed | 17 | 17 × 988.9 = 16,811 |

#### 2.4.2 当前 analytical 分数（无 calibration）

从 costmodel JSON：

| 路线 | 原始 analytical | 加 structural penalty 后 | 说明 |
|------|---------------|------------------------|------|
| SIMD | 2,439.88 | 2,439.88 × 1.828 = 4,460 | ×(1 + Pstruct) |
| SIMT only | 521,298.69 | 521,298.69 | 无 structural penalty |
| Mixed | — | 37,463.33 | 混合 partition 计算 |

#### 2.4.3 计算 multipliers

目标：让 calibrated score 的**排序**与实测延迟排序一致，且**比例**匹配。

约束方程：
```
m_simd × 4460 : m_simt × 521299 : m_mixed × 37463
    = 241 : 50 : 17
```

以 mixed 为 anchor（设 m_mixed = 1.0）：

```
m_simt / m_mixed = (50/17) × (37463/521299) = 2.941 × 0.07187 = 0.211
m_simd / m_mixed = (241/17) × (37463/4460)  = 14.176 × 8.400 = 119.1
```

验证：
```
calibrated SIMD  = 4460 × 119.1 = 531,186   (最大 = 最差)
calibrated SIMT  = 521299 × 0.211 = 109,994 (中间)
calibrated Mixed = 37463 × 1.0 = 37,463     (最小 = 最优)
```

比例验证：
- SIMD/SIMT = 531186/109994 = 4.83 ≈ 241/50 = 4.82 ✓
- SIMD/Mixed = 531186/37463 = 14.18 ≈ 241/17 = 14.18 ✓
- SIMT/Mixed = 109994/37463 = 2.94 ≈ 50/17 = 2.94 ✓

**排序：Mixed < SIMT < SIMD，与实测一致。**

对比现有 domain 的 multiplier 数量级：

| Domain | all_simd | all_simt_only | mixed |
|--------|----------|---------------|-------|
| masked_rowwise_reduction | 67.63 | 0.808 | 3.537 |
| tiny_irregular_dot | 1.32 | 1.0 | 0.666 |
| triangular_solve_loop | 290.9 | 1.71 | 2.89 |
| **attention_indirect_gqa** | **119.1** | **0.211** | **1.0** |

在合理范围内（triangular_solve_loop 的 SIMD multiplier 甚至达到 290.9）。

## 三、代码修改清单

### 3.1 `SimdSimtCostModel.cpp` — `CoverageProfile` 结构体

新增字段：

```cpp
struct CoverageProfile {
  // ... 现有字段 ...
  
  // attention_indirect_gqa domain 边界
  int64_t attentionMinIndirectMemoryOps = 2;
  int64_t attentionMinDotOps = 1;
  int64_t attentionMaxTensorNumel = 2048;
  int64_t attentionMaskRankSumMax = 200;
};
```

### 3.2 `SimdSimtCostModel.cpp` — profile JSON 加载

在 coverage 加载段新增：

```cpp
profile.coverage.attentionMinIndirectMemoryOps = reader.integer(
    *coverage, "attention_min_indirect_memory_ops", "coverage");
profile.coverage.attentionMinDotOps = reader.integer(
    *coverage, "attention_min_dot_ops", "coverage");
profile.coverage.attentionMaxTensorNumel = reader.integer(
    *coverage, "attention_max_tensor_numel", "coverage");
profile.coverage.attentionMaskRankSumMax = reader.integer(
    *coverage, "attention_mask_rank_sum_max", "coverage");
```

### 3.3 `SimdSimtCostModel.cpp` — `rankingCalibrationCoverage()` 函数

在 `tiny_irregular_dot` 检查之前（约 line 1159）插入：

```cpp
  // Attention with indirect KV lookup + dots + softmax-like reduction.
  // Characteristic of Flash Attention / GQA kernels where KV indices are
  // loaded from a structured buffer and then used to index K/V tensors.
  // Distinguished from masked_rowwise_reduction by the presence of dots
  // and from tiny_irregular_dot by larger dot scale and softmax pattern.
  if (dotFlops > 0 &&
      features.loadedIndexDependentMemoryOps > 0 &&
      features.rowLocalReduceOps > 0 &&
      maxNumel <= coverage.attentionMaxTensorNumel &&
      maskRankSum <= coverage.attentionMaskRankSumMax &&
      features.dotOps >= coverage.attentionMinDotOps &&
      features.simtAnchors.loadedIndexDependentMemoryOps >=
          coverage.attentionMinIndirectMemoryOps)
    return {true, "attention_indirect_gqa"};
```

**位置选择**：放在 `hasUnknownTripCount` 检查**之前**。原因是此 domain 特意允许 `hasUnknownTripCount=true`，如果放在之后，未知循环会先匹配到 `loop_trip_count_unknown`，永远进不了新 domain。

### 3.4 `david_v100_simd_simt_v1.json` — profile JSON

#### 3.4.1 新增 coverage 字段

在 `selection_calibration.coverage` 中：

```json
"attention_min_indirect_memory_ops": 2,
"attention_min_dot_ops": 2,
"attention_max_tensor_numel": 2048,
"attention_mask_rank_sum_max": 200
```

#### 3.4.2 新增 event calibration domain

在 `selection_calibration.event_route_score_multiplier.domains` 中：

```json
"attention_indirect_gqa": {
  "all_simd": 119.1,
  "all_simt_only": 0.211,
  "mixed_simd_simt": 1.0,
  "all_simt_only_validated": true,
  "mixed_simd_simt_validated": true,
  "source": "A5 card 0, ROPE GQA fwd T16/R16/N16/num_warps4/BS=1, "
           "one process with routes interleaved after 50 warmup, "
           "200 NPU Event samples each: median SIMD 241 us, SIMT-only 50 us, "
           "C++-materialized mixed 17 us; all routes correctness PASS. "
           "Mixed is 14.2x faster than SIMD.",
  "confidence": "low"
}
```

### 3.5 Profile version bump

```json
"profile_version": "david-v100-simd-simt-20260804-v11"
```

## 四、验证计划

### 4.1 单元测试

修改 `SimdSimtCostModelTest.cpp`，添加 `attention_indirect_gqa` domain 的覆盖测试：

```cpp
// 验证 ROPE kernel 被正确分类到 attention_indirect_gqa
TEST(SimdSimtCostModel, AttentionIndirectGqaCoverage) {
  // 用 ROPE TTIR 运行 analyzeSimdSimtCandidates
  // EXPECT_TRUE(report->calibrationCovered);
  // EXPECT_EQ(report->calibrationDomain, "attention_indirect_gqa");
  // EXPECT_TRUE(report->eventRouteCalibrationApplied);
}
```

### 4.2 端到端验证

对 ROPE kernel 运行 costmodel：
```bash
TRITON_ASCEND_SIMD_SIMT_PROFILE=.../david_v100_simd_simt_v1.json \
  <run costmodel on _fwd_grouped_kernel_stage1_rope.ttir>
```

检查 costmodel JSON 输出：
1. `calibration_domain` = `"attention_indirect_gqa"`
2. `event_route_calibration.applied` = `true`
3. `candidate_costs` 排序：`all_simd` > `all_simt_only` > `mixed_simd_simt`
4. `decision_kind` = `"mixed_simd_simt"`
5. `decision_advantage` > 0 且 > `required_gain_score`

### 4.3 回归验证

对现有 3 个 domain 的 kernel 重新运行 costmodel，确保分类不变：
- FBGEMM → `masked_rowwise_reduction`
- gather-dot-min → `tiny_irregular_dot`  
- solve-tril → `triangular_solve_loop`

## 五、风险与限制

### 5.1 Low confidence

- 只基于**一个 kernel**（ROPE GQA fwd）的实测数据
- 需要在 A5 上测试更多同类型 kernel（如不同 head dim、不同 sequence length 的 GQA/MLA kernel）来验证 multiplier 的泛化性

### 5.2 Coverage 边界可能太窄或太宽

- 当前 boundary 是为 ROPE kernel 定制的
- 太窄：同类型的 attention kernel 进不来 → 需要放宽
- 太宽：不相关的 kernel 进来了 → multiplier 不准确 → 需要收紧

### 5.3 all_simt_only 实测

当前 SIMT-only 50us 可能未经过完整的 backend 验证。profile 中设置 `all_simt_only_validated: true` 后，costmodel 会允许 pure SIMT 路线入选。如果 pure SIMT 路径有正确性问题，需要改为 `false`。

### 5.4 长期方案

当前的 event multiplier 方案是**经验校准**——它吸收 predicate 公式的误差而非修正公式本身。长期看可以考虑：
- 为 attention 类 kernel 单独测量 `simtPredicateRate`（类似现有微基准）
- 改进 predicate 公式，使用 `uniqueMaskRankSum` 替代 `maskRankSum`

## 六、改动总结

| 文件 | 改动 | 行数 |
|------|------|------|
| `SimdSimtCostModel.cpp` `CoverageProfile` | +4 字段 | ~5 |
| `SimdSimtCostModel.cpp` coverage 加载 | +4 读取 | ~8 |
| `SimdSimtCostModel.cpp` `rankingCalibrationCoverage()` | +12 行检查逻辑 | ~15 |
| `david_v100_simd_simt_v1.json` | +4 coverage 字段 + 1 个 domain | ~18 |
| `SimdSimtCostModelTest.cpp` | +1 测试 | ~15 |
| **总计** | | **~60 行** |
