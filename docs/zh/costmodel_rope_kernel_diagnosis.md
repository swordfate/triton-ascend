# Rope Kernel Costmodel 诊断：预测与实测完全相反的根因分析

> 分支：`kx_simt_costmodel`（基于 `400f73560`）
> 涉及文件：
> - `test_cases/_fwd_grouped_kernel_stage1_rope.ttir`
> - `test_cases/_fwd_grouped_kernel_stage1_rope.ttadapter`（非 costmodel 路径）
> - `test_cases/_fwd_grouped_kernel_stage1_rope_costmodel_result.json`（costmodel 输出）

## 实测结果

| 模式 | 延迟 | costmodel 预测 |
|------|------|--------------|
| SIMD (compile_mode=simd) | 241 us | 4,460 cycles |
| SIMT (compile_mode=simd_simt) | **17 us** | 521,299 cycles |

costmodel 预测 SIMD 比 SIMT 好 117 倍，但实测 SIMT 比 SIMD 快 14 倍。**方向完全反了。**

---

## 一、Kernel 理解

### 1.1 整体结构

`_fwd_grouped_kernel_stage1_rope` 是一个 **Flash Attention with GQA（Grouped Query Attention）+ RoPE** 的 Triton kernel。每个 program 处理一个 (batch, head_group, kv_split) 三元组。

```python
@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,                    # Q 矩阵 [Q_NOPE; Q_PE], shape: b × h × (d+r)
    K_Buffer,             # KV cache，shape: b*s × (c+r)
    V_buffer,             # V cache，shape: b*s × c
    cos_sin_cache,        # RoPE 正余弦表，shape: max_seq_len × (rotary_dim * 2)
    positions,            # 每个 batch 的位置
    sm_scale,             # softmax scale
    kv_indptr,            # KV cache 中各 batch 的起始偏移
    kv_indices,           # KV cache 中每个 token 的物理位置（间接索引）
    Att_Out,              # 输出：[batch, head, split, kv_lora_rank+1]
    k_pe_t_out,           # K 的 position embedding 输出
    # ... strides ...
    BLOCK_C: tl.constexpr,       # = 512 (kv_lora_rank)
    BLOCK_R: tl.constexpr,       # = 64  (qk_rope_head_dim)
    BLOCK_N: tl.constexpr,       # = 32  (KV block size)
    BLOCK_H: tl.constexpr,       # = 16  (num heads per group)
    NUM_KV_SPLITS: tl.constexpr, # = 2   (KV splits for pipelining)
):
```

每个 program 有 3 个 program_id 维度：
- `pid(0)` → 选择 batch
- `pid(1)` → 选择 head group
- `pid(2)` → 选择 KV split

### 1.2 数据流：从输入 tensor 到输出

```
Step 0: 读取 KV 元数据
  cur_batch_kv_start_idx = kv_indptr[batch]        ← structured load
  cur_batch_seq_len       = kv_indptr[batch+1] - start_idx  ← structured load
  kv_len_per_split = cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
  split_kv_start   = kv_len_per_split * split_kv_id
  split_kv_end     = min(split_kv_start + kv_len_per_split, cur_batch_seq_len)

Step 1: 加载 Q 和 Q_PE（结构化，不依赖其他 tensor load）
  q    = Q[batch, head, :kv_lora_rank]              ← structured
  q_pe = Q[batch, head, kv_lora_rank:kv_lora_rank+BLOCK_R]  ← structured

Step 2: 加载 RoPE cos/sin 值（间接，依赖 positions tensor）
  pos = positions[batch]                             ← structured scalar load
  cos = cos_sin_cache[pos * stride + offs_rotary]    ← ★ 间接：地址用 pos
  sin = cos_sin_cache[pos * stride + offs_rotary + rotary_dim//2]  ← ★ 间接

Step 3: 循环遍历 KV blocks
  for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    # 3a: 加载 KV 物理位置（结构化）
    kv_loc = kv_indices[cur_batch_kv_start_idx + start_n + arange(0, BLOCK_N)]

    # 3b: 用 kv_loc 做索引，从 K_Buffer 加载 key content（★ 间接）
    k_pe = K_Buffer[kv_loc * stride + offs_qk_r]     ← ★ 间接
    kv   = K_Buffer[kv_loc * stride + offs_c]         ← ★ 间接

    # 3c: 计算 attention scores
    qk = dot(q_pe, k_pe) + dot(q, kv)

    # 3d: 用 kv_loc 做索引，从 V_buffer 加载 value（★ 间接）
    v = V_buffer[kv_loc * stride + offs_c]            ← ★ 间接

    # 3e: Flash Attention softmax + accumulate
    qk *= sm_scale
    n_e_max = max(qk, axis=1)
    re_scale = exp(e_max - n_e_max)
    p = exp(qk - n_e_max)
    acc = acc * re_scale + dot(p, v)

Step 4: 写输出
  Att_Out[batch, head, split, :] = acc / e_sum
  Att_Out[batch, head, split, kv_lora_rank] = e_max + log(e_sum)
```

### 1.3 间接访存 = SIMT Anchor 候选

标记为 ★ 的 7 处 `tl.load` 的地址依赖另一个 `tl.load` 的结果：

| # | 变量 | TTIR 行号 | 依赖的 load 结果 | 在循环内？ |
|---|------|----------|-----------------|----------|
| A1 | `cos` (line 135) | ~96 | `pos` (line 90, 从 `positions` load) | 否 |
| A2 | `sin` (line 140) | ~98 | `pos` | 否 |
| A3 | `k_pe_last_token` (line 172) | ~133 | `kv_indices[last]` (line 123) | 否（scf.if 内） |
| A4 | `k_pe_rot_last_token` (line 177) | ~135 | `kv_indices[last]` | 否（scf.if 内） |
| A5 | `k_pe` (line 202) | ~192 | `kv_loc` (line 181, 从 `kv_indices` load) | **是** |
| A6 | `kv` (line 219) | ~210 | `kv_loc` | **是** |
| A7 | `v` (line 238) | ~224 | `kv_loc` | **是** |

costmodel 的 JSON 报告了 8 个 anchor，可能是某个 load 有变体被重复计数（排查时可在 `buildMixedSimtAnchorPlan` 中打印 operation name）。

这些间接访存正是 `simd_simt` 模式下 SIMT template 的目标——每个都变成 `call @triton_indirect_load`。

---

## 二、Costmodel 逐步走读（对着 TTIR）

### 2.1 入口：`runOnOperation` (SelectSimdSimtCostModel.cpp:97)

```cpp
SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
auto report = analyzeSimdSimtCandidates(module, anchorPlan, options);
```

### 2.2 Phase 0：`buildMixedSimtAnchorPlan` (SimtAnchorAnalysis.cpp:545)

`module.walk(PreOrder)` 遍历所有 op，对每个 op 调用 `analyzeAnchor(op)`。

#### `analyzeAnchor` (SimtAnchorAnalysis.cpp:312)

匹配顺序：`tt.gather` → `tt.histogram` → `tt.scan` → `tt.atomic_*` → TriangularSolve → **`isLoadedIndexDependentMemoryOp`**

对每个 `tt.load`：
1. `hasTensorPointerOperand(op)` → true（load 的第一个 operand 是 tensor pointer）
2. `pointerDependsOnLoadedIndex(op)` → BFS 回溯地址的 SSA define-chain

#### `pointerDependsOnLoadedIndex` (SimtAnchorAnalysis.cpp:96)

对 `tt.load` 的 address operand（第 0 个 operand），BFS 遍历 use-def chain，检查是否有祖先 op 是 `tt.load` 或 `tt.gather`。

**以 A5（循环内 k_pe load）为例：**

TTIR line 192:
```mlir
%164 = tt.load %163, %162, %cst_2
```
- `%163 = tt.addptr %107, %158`（tensor pointer）
- `%107 = tt.splat %arg1`（base = K_Buffer 的 memref base，不是 loaded）
- `%158 = arith.addi %156, %103`
  - `%156 = tt.broadcast %155`
    - `%155 = arith.muli %154, %97`
      - `%154 = tt.expand_dims %153` ← **%153 是 `tt.load` 的结果！**
      - `%153 = tt.load %152`（line 181，加载 `kv_indices[range]`）

BFS 到达 `%153`，其 defining op = `tt.load` → `pointerDependsOnLoadedIndex` 返回 **true**。

**结果：** A5 被识别为 `LoadedIndexDependentMemory` anchor。其他 6 处同理。

### 2.3 Phase 1：`analyzeSimdSimtFeatures` (SimdSimtCostModel.cpp:1890)

`module.walk` 遍历所有 op，对每个 op 做四件事：

#### 2.3.1 分类加权（line 1964-1966）

```cpp
features.weightedOps[weightedKind] += loopMultiplier;     // 计数 = 1
features.opElements[weightedKind] += elements * loopMultiplier;  // 元素数
```

**以 TTIR line 207 的 `tt.dot`（q_pe @ k_pe）为例：**

```mlir
%168 = tt.dot %88, %167, %cst_6 : tensor<16x16xf16> * tensor<16x32xf16> -> tensor<16x32xf32>
```

- `name = "tt.dot"` → `classifyWeightedOp` 返回空（dot 有单独处理）
- 进入 dot 处理（line 2021-2036）：
  ```cpp
  // lhs shape = [16, 16] → m=16, k=16
  // rhs shape = [16, 32] → k=16, n=32
  dotFlops += 2 * 16 * 32 * 16 * loopMultiplier = 2 * 8192 * 1 = 16384
  dotOutputElements += 16 * 32 * 1 = 512
  dotMNK.push_back({16, 32, 16})
  ```

同理其他两个 dot（line 211 和 line 241），累计 `dotFlops = 16384 + 16384 + 16384 = 49152`。

#### 2.3.2 Mask 计数（line 2097-2134）

对每个有 mask tensor operand/result 的 op：

```cpp
features.maskTensorOps++;      // 带 mask 的 op 数量 = 53
features.maskRankSum += rank;  // 累加每个 mask 的 tensor rank
```

**以 TTIR line 192 的 `tt.load` 为例：**

```mlir
%164 = tt.load %163, %162, %cst_2
```

- `%162: tensor<16x32xi1>` 是 mask operand，rank = 2
- `%cst_2: tensor<16x32xf16>` 是 other operand，这个不算 mask
- → `maskTensorOps++`, `maskRankSum += 2`

循环内的每个带 mask 的 load 都贡献 +2。加上 `arith.select`、`arith.cmpi` 等操作的 mask input/output → 最终 `maskRankSum = 153`。

同时 `recordUniqueMask` 对每个 mask 做去重：
```cpp
if (uniqueMasks.insert(value).second) {
    uniqueMaskValues++;
    uniqueMaskRankSum += type.getRank();   // = 66
    predicateElements += numel(type);      // = 8032
}
```

同一 mask value 被多个 op 共享时只计一次。最终 `uniqueMaskValues = 37, uniqueMaskRankSum = 66`。

#### 2.3.3 循环识别（line 1968-1980）

遇到 `scf.for`：

```mlir
// TTIR line 176
%120:3 = scf.for %arg19 = %50 to %52 step %c32_i64 ...
```

```cpp
auto knownTripCount = getKnownStaticLoopTripCount(op);
// getConstantInteger(%50) → arith.muli → nullopt ✗
// getConstantInteger(%52) → arith.minsi → nullopt ✗
// getConstantInteger(%c32_i64) → arith.constant 32 → 32 ✓
// → 有 operand 不是 constant → nullopt
// → hasUnknownTripCount = true
// → tripCount = nullopt.value_or(1) = 1
features.staticLoopCount += 1;       // = 1
features.staticLoopTripCountSum += 1; // = 1
```

**此时 loopMultiplier 对循环内所有 op 都是 1**——所有加权计数都视为循环只执行一次。

#### 2.3.4 `loadedIndexDependentMemoryOps` 计数（line 2138-2142）

```cpp
if (isLoadedIndexDependentMemoryOp(op)) {
    features.loadedIndexDependentMemoryOps++;  // = 8
    if (inAnchor) features.simtAnchors.loadedIndexDependentMemoryOps++; // = 8
}
```

#### 2.3.5 Feature Summary 结果

| 字段 | 值 | 来源 |
|------|-----|------|
| `loadOps` | 15 | 所有 `tt.load` |
| `weightedOps["load"]` | 15 | load ops × 1（loopMultiplier=1） |
| `opElements["load"]` | 2404 | ∑ loaded_tensor_numel × 1 |
| `maskTensorOps` | 53 | 带 mask operand 的 op 数 |
| `maskRankSum` | 153 | 所有 mask tensor 的 rank 总和 |
| `uniqueMaskValues` | 37 | 去重后的 mask SSA value 数 |
| `uniqueMaskRankSum` | 66 | 去重 mask 的 rank 总和 |
| `predicateElements` | 8032 | 去重 mask 的总元素数 |
| `dotFlops` | 49152 | 3 个 dot 的 FLOPS |
| `hasUnknownTripCount` | **true** | 循环上下界不是 arith.constant |
| `staticLoopCount` | 1 | |
| `staticLoopTripCountSum` | **1** | 未知 → 默认 1 |
| `loadedIndexDependentMemoryOps` | 8 | 间接访存 op 数 |
| `maxTensorNumel` | 512 | 最大 tensor = 16×32 |

### 2.4 Phase 2：`estimateSimdSimtCandidates` (SimdSimtCostModel.cpp:2337)

#### 2.4.1 Coverage 检查（line 2268）

```cpp
auto [covered, domain] = rankingCalibrationCoverage(features, ...);
// hasUnknownTripCount = true
// → return {true, "loop_trip_count_unknown"}  (我们的修改后)
// → calibrationCovered = true
// → 继续评分
```

（如果不带修改分支，这里会返回 `{false, "unknown_loop_trip_count"}`，评分被跳过，直接 `backend_default`。）

#### 2.4.2 SIMD/SIMT Compute 评分（line 2475）

`getProfileOpElements(features)` 返回每个 op 类型的加权元素数（已乘以 loopMultiplier=1），然后查 profile 表：

```cpp
// 对每个 opType：
simdCycles = ceil(elements / vectorWidth) / simdThroughput * simdFactor;
simtCycles = elements / simtThroughput * simtFactor;

// 例如 load：
//   elements = 2404, vectorWidth = 16
//   simdCycles = ceil(2404/16) / simdThroughput * factor ≈ 24.8
//   simtCycles = 2404 / simtThroughput * factor ≈ 459.7
```

| 组分 | SIMD cycles | SIMT cycles | 说明 |
|------|-----------|-----------|------|
| compute | 162.3 | 115.2 | 纯算术 op（add, mul, div, exp, log...）|
| dot | 140.0 | 412.6 | 3 个 dot 操作 |
| load | 24.8 | 459.7 | memory load bytes / bandwidth |
| store | 2.8 | 77.4 | memory store bytes / bandwidth |
| shuffle | 0 | 0 | 无 |
| predicate | **0** | **64,421** | ← ⚠️ |

SIMT 的 compute 部分（115+412=527）和 SIMD 的 compute 部分（162+140=302）在同一量级。SIMT memory（459+77=537）确实比 SIMD（24.8）高 20× —— 这是正确的，因为 gather 在 SIMT 中确实有 overhead。

**但 predicate 的 64,421 cycles 是问题所在。**

#### 2.4.3 ⚠️ SIMT Predicate 评分（line 2496-2501）

```cpp
report.breakdown.simtPredicateInstructions =
    static_cast<double>(features.maskRankSum) *
    std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize);
// = 153 * ceil(512 / 32) = 153 * 16 = 2448

report.breakdown.simtPredicateCycles =
    report.breakdown.simtPredicateInstructions / profile.simtPredicateRate;
// = 2448 / [profile_value] ≈ 2448 / 0.038 = 64,421
```

**为什么是 153 × 16？**

公式的直觉：SIMT 中每个 warp 需要一条 predicate 指令来确定哪些 lane 参与执行。每个 mask rank（比如一个 `tensor<16x32xi1>` 的 rank 2）需要为每个 block（blockCount = ceil(512/32) = 16 个 warp）生成 predicate。所以总共是 153 × 16 = 2448 条 "predicate 指令"。

**问题在于：**
1. `maskRankSum = 153` 包含所有 mask——循环内的每个带 mask 的 op 都贡献 rank，即使是同一个逻辑概念的 mask（如 `offs_n < split_kv_end`）在每次循环迭代中都产生不同的 SSA value，但 rank 被重复计数。
2. 在真实硬件（Ascend 950）上，mask 是**指令内嵌**的——一条 SIMT load 指令本身就带 mask 参数，不需要额外发射一条 predicate 指令。Predicate 是免费的。
3. `simtPredicateRate ≈ 0.038`（每 system cycle 只能处理 0.038 条 predicate 指令）这个值本身来自 microbenchmark 对**显式 predicate 生成**的测量，而非指令内嵌 mask。

#### 2.4.4 Payload 和 Analytical 评分（line 2644-2660）

```cpp
// Roofline: max(compute, memory)
simdIssuePayload = max(162.3 + 140.0, 24.8) = 302.3
simtIssuePayload = max(115.2 + 412.6 + 0, 537.2) + 64421
                 = max(527.8, 537.2) + 64421 = 537.2 + 64421 = 64958.2

// 加 program issue scale（profile 中 = 8）
simdAnalytical = simdSetup + simdPayload * scale = 21.2 + 302.3 * 8 = 2439.9
simtAnalytical = simtSetup + simtPayload * scale = 141   + 64958 * 8 = 519805.1
```

**到这里，差距已经是 519805 / 2440 ≈ 213x，全是 predicate 贡献的。**

#### 2.4.5 结构性惩罚（line 2662-2710）

```cpp
structuralComponents["static_loop_control"] =
    min(loopCap, 1 * perStaticLoopTrip);  // = 0.008（因为 tripCount=1）

// 如果 tripCount 已知（比如 =100），这里会是：
// min(loopCap, 100 * perStaticLoopTrip) = 更大
// 但相对于 predicate 的 64k，这点差异不重要

structuralPenaltyRatio = 0.4 + 0.35 + 0.04 + 0.03 + 0.008 = 0.828
simdStructuralPenalty = 2440 * 0.828 = 2020

allSimd    = 2440 + 2020 = 4460
allSimtOnly = 519805
```

#### 2.4.6 Confidence 和 Gate（line 2790-2812）

```cpp
report.absoluteConfidence = minimumConfidence(resourceConfidence);
// 各 op 的 profile confidence 最低值为 "low"

report.rankingConfidence = minimumConfidence(
    {report.absoluteConfidence, profile.rankingConfidence});
// = minimumConfidence({"low", "low"}) = "low"

report.minimumConfidenceForDecision = profile.minimumConfidence;
// = "low"（这个 profile 的特殊配置：把门槛设为了 "low"）

// Gate check:
confidenceRank("low") = 1, confidenceRank("low") = 1
// 1 < 1 → false → gate 通过！
```

**注意这个 profile 把 `minimumConfidenceForDecision` 设成了 `"low"`**（通常默认是 `"medium"`）。这意味着即使 confidence 只有 "low"，gate 也会通过。这是一个显式的宽松配置。

#### 2.4.7 决策（line 2770-2780）

```cpp
// 排序候选：
// allSimd = 4460 (best)
// mixed = 37463 (8.4x worse)
// allSimtOnly = 519805 (117x worse)

decision = allSimd
gainScore = 33003  // mixed - allSimd = 37463 - 4460，满足 margin
gatePassed = true
effectiveDecision = allSimd
```

---

## 三、根因总结

### 3.1 正确的地方

1. **Anchor 识别正确**：7-8 个 `tt.load` 的地址依赖 loaded index 被准确识别为 `LoadedIndexDependentMemory`。
2. **compute 评分大致正确**：SIMT compute(115) < SIMD compute(162)，符合直觉。
3. **memory 评分方向正确**：SIMT gather(537) > SIMD structured(25)，约 20x，方向对。

### 3.2 错误的地方

**唯一的主因：SIMT Predicate 评分。**

```
simtPredicateCycles = 153 × 16 / 0.038 = 64,421
                        ↑    ↑      ↑
                   maskRankSum |  simtPredicateRate（profile 中极低）
                            ceil(512/32)
```

64,421 是 SIMD 总分 4,460 的 14.4 倍。去掉这个项，SIMT 总分降到 $\sim 60,000$ 以下（主要来自 memory gather overhead），但仍然远高于 SIMD。

**但实测 SIMT 是 17us 而 SIMD 是 241us → SIMT 实际快了 14 倍。** 这说明不仅是 predicate 被高估，**SIMT 的 memory gather cost 和 dot cost 也被高估**——可能是因为 profile 中的 throughput 值不是针对这个芯片型号（`Ascend950PR_9579`）校准的，而是来自一个不同的 microbenchmark 目标。

### 3.3 三道防线：为什么 costmodel 仍然返回了 all_simd

1. **Coverage 放松后**：`loop_trip_count_unknown` → 准入（我们改的）
2. **Confidence 门槛降低**：`minimumConfidenceForDecision = "low"` 而不是默认的 `"medium"`
3. **Gate 通过**：gainScore 满足 margin

三道防线全过，决策 `all_simd` 被采纳。但 scored value 是错的。

### 3.4 修复优先级

| 优先级 | 修改点 | 预期效果 |
|--------|--------|---------|
| **P0** | Profile `simtPredicateRate` —— 调高到接近 memory/compute throughput 的水平（如果 predication 是免费的，应该 > 1.0） | `simtPredicateCycles` 从 64k 降到 < 1k |
| **P1** | 公式改用 `uniqueMaskRankSum` 替代 `maskRankSum`，并考虑 mask 复用 | `simtPredicateInstructions` 从 2448 降到 ~100 |
| **P2** | Profile `simtLoadRate` / `simtDotFlopsPerCycle` —— 针对目标芯片重新校准 | SIMT memory/dot cost 与实测对齐 |
| **P3** | 实现精确 trip count（通过 JIT scalar capture） | `staticLoopTripCountSum` 反映真实循环次数，结构性惩罚更准 |
