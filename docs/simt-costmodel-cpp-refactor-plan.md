# C++ Costmodel 重构计划（Step 6）

> 目标：把 A5 实测拟合出的 pattern-dependent rate 和 component-targeted
> penalty 落入 C++，同时修正 predicate / shuffle / dot / memory 的结构性问题。
> 原则：**惩罚要加到它真正对应的组件上；domain multiplier 应逐步退化为 1.0
> 并最终只保留 coverage/validated 闸门。**

## 1. 当前 C++ 的主要不合理点

| 问题 | 当前实现 | 修正方向 |
|---|---|---|
| structural penalty 整体乘在 SIMD 分数上 | `allSimd = simdAnalytical * (1 + Pstruct)` | 拆分到对应组件 |
| irregular_addressing 惩罚 | 乘整体 | 乘到 SIMD memory cycles |
| tiny_dot 惩罚 | 乘整体 | 乘到 SIMD dot cycles |
| mask/reduction/loop/control/rank1 | 乘整体 | 加到 SIMD compute cycles 的 lowering penalty |
| SIMD memory rate | 固定 `202.25 B/cycle` | 按 contiguous/strided/gather 查表/拟合函数 |
| SIMT GM rate | 固定 `0.176/0.129` | 按 contiguous/strided/gather 查表/拟合函数 |
| SIMT predicate | `maskRankSum * ceil(maxNumel/32) / 0.038` | 用 `mask_tensor_ops * ceil(maxNumel/32)` 和实测 bounds-mask rate |
| SIMT shuffle | 用整核 `maxTensorNumel` | 用 reduce/scan 自己的 `opElements` 或 reduce axis extent |
| SIMD dot | `setup + flops/4096` | 保留 4096，但加 `max(..., small_kernel_min_cycles)`；tiny_dot penalty 只加 dot |
| domain multiplier | 每个域三个常数 | 公式变准后逐步置 1，仅保留 `coverage` 和 `validated` 标志 |

## 2. 目标公式

### 2.1 SIMD

```text
irregularPenalty = min(irregular_cap, irregular_density * irregular_per_density)

simdMemoryCycles = max(loadBytes / loadRate(pattern),
                       storeBytes / storeRate(pattern))
                 * (1 + irregularPenalty)

simdDotCycles = max(dot_setup + dotFlops / dotFlopsPerCycle,
                    small_kernel_min_cycles)
if (tiny_dot):
    simdDotCycles *= (1 + tiny_dot * underfill)

loweringPenalty = maskPenalty + reductionPenalty + loopPenalty
                + controlFlowPenalty + rank1IndirectReductionPenalty
simdComputeCycles = computeOnlyCycles * (1 + loweringPenalty)

simdPayload = max(simdComputeCycles + simdDotCycles, simdMemoryCycles)
allSimdRaw = simdSetup + programIssueScale * simdPayload
```

### 2.2 SIMT

```text
simtMemoryCycles = loadWarpInstructions / loadRate(pattern)
                 + storeWarpInstructions / storeRate(pattern)

simtShuffleCycles = (reduceWarpInstructions + scanWarpInstructions)
                   / shuffleRate
// 其中 reduceWarpInstructions 使用 reduce op 自己的元素数，
// 不再使用整核 maxTensorNumel。

simtPredicateCycles = predicatedWarpInstructions / predicateRate
// 其中 predicatedWarpInstructions = maskTensorOps * ceil(maxTensorNumel/32)
// 或更准确的“被 mask 的 warp 指令数”。

simtPayload = max(simtCompute + simtShuffle + simtDot, simtMemory)
            + simtPredicate
allSimtRaw = simtSetup + programIssueScale * simtPayload
```

### 2.3 Mixed

继续使用 materializable anchor partition，但内存/计算/predicate/shuffle/dot
分别采用上面的 pattern-dependent 组件公式。

## 3. C++ 改动点

### 3.1 特征抽取 `analyzeSimdSimtFeatures`

- 新增/整理 pattern 特征：
  - `loadedIndexDependentMemoryOps`：gather/indirect 主标志；
  - `pointerUnstructuredDims` / `laneDependentPointerOps`：stride 代理；
  - 对 `tt.load/store` 增加 stride 估计（如 shape 维度和访问步长）；
  - 对 masked load，后续可选新增 `effectiveLoadBytes`（active_ratio 口径）。
- 记录每个 reduce/scan op 的 `opElements`，供 shuffle 指令数计算使用。

### 3.2 Profile 加载 `loadCandidateProfile`

新增可选字段（保持 schema_version=4，字段 optional）：

```json
"simd_memory": {
  "contiguous_bytes_per_cycle": 1250.76,
  "gather_bytes_per_cycle": 2.272,
  "strided_power_fit": { "A": 27.034, "exponent": -0.631 }
},
"simt_memory": {
  "load_contiguous_warp_instr_per_cycle": 0.4004,
  "store_contiguous_warp_instr_per_cycle": 0.4638,
  "load_gather_warp_instr_per_cycle": 0.0201,
  "store_gather_warp_instr_per_cycle": 0.0207,
  "strided_power_fit": { "load": {"A": 0.4471, "exponent": -0.8461}, ... }
},
"simt_predicate": {
  "bounds_mask_add_warp_instr_per_cycle": 0.2356,
  "predicated_select_warp_instr_per_cycle": 0.1851,
  "masked_gm_load_warp_instr_per_cycle": 0.1602
},
"simd_dot": {
  "small_kernel_min_cycles": 16233
}
```

### 3.3 评分 `estimateSimdSimtCandidates`

- 实现 `resolveSimdMemoryRate(features, profile)` 和
  `resolveSimtMemoryRates(features, profile)`；
- 实现 component-targeted penalty 公式；
- `simtShuffle` 改为 per-op 元素数；
- `simtPredicate` 改为 predicated warp 指令数 + 新 rate；
- `simdDot` 增加 small-kernel floor；
- domain multiplier 保留，但未来拟合后趋近 1.0。

## 4. 验证方案

1. 先跑 `prototype_new_formula.py`，观察 5 case 的 `predicted_ratio` 是否更接近
   `measured_ratio`；
2. C++ 改完后，在 A5 上重跑 `run_triton_benchmark.py --case all --route all`；
3. 用 `analyze_residuals.py` 看残差；
4. 若 5 case 残差可接受，再逐步把 domain multiplier 置 1 并回归。

## 5c. v5 实际-cycle 口径结果（program_issue_scale=1.0）

| case | measured ratio | v5 raw ratio |
|---|---|---|
| block_matmul | 1.106 | 0.993 |
| elementwise_silu_mul | 0.864 | 0.951 |
| rowwise_reduce_masked | 0.984 | 0.880 |
| single_block_cumsum | 4.904 | 4.811 |
| indirect_elementwise | 44.598 | 73.910 |

剩余问题：indirect_elementwise 仍高估约 1.66x。根因是模型把
`tt.load` 的所有 load bytes 都按 gather rate 计算，但实际一半是连续索引 load，
一半才是 gather data load。下一步需要拆分 contiguous/gather 字节。

## 5b. A5 第二轮微基准：dot 与 scan

### dot（SIMT 路由）

- `128x128x64` SIMT dot：`0.01414 ms`；SIMD dot：`0.01644 ms`。
- 结论：A5 上 `tl.dot` 在 SIMT 路由同样走 cube，SIMT dot 不应使用
  `141 scalar FMA/cycle`。profile 已改为 `simt.dot.flops_per_system_cycle=4096`、
  `startup_system_cycles=128`。
- `256x256x128` SIMT dot 在 A5 上触发 NPU 507035 错误，暂无法采集；
  不影响 cube 结论。

### scan（cumsum）

| n | SIMD latency ms | SIMT latency ms |
|---|---|---|
| 256 | 0.01152 | 0.01173 |
| 1024 | 0.01591 | 0.01323 |
| 4096 | 0.06067 | 0.01307 |
| 8192 | 0.12049 | 0.01299 |

- SIMD scan 近似 `cycles = 1000 + n / 0.0693`（O(n)）；
- SIMT scan 在 n>=1024 后几乎恒定，约 `13 μs`；
- 当前 costmodel 完全没有 scan 专用项，且把 scan 标为
  `scan_template_ranking_uncalibrated`。下一步需要新增 scan 组件项并移除该
  unsupported 标记。

## 5. Python 原型初步结果

| case | measured | old raw ratio | new raw ratio |
|---|---|---|---|
| block_matmul | 1.111 | 0.103 | 0.069 |
| elementwise_silu_mul | 0.713 | 0.036 | 0.333 |
| indirect_elementwise | 36.087 | 0.012 | 0.682 |
| rowwise_reduce_masked | 1.781 | 0.028 | 1.205 |
| single_block_cumsum | 4.577 | 0.008 | 0.060 |

说明：
- rowwise / elementwise 已明显改善；
- indirect 仍差很远，说明还需要更细的 SIMD indirect 建模（可能不仅是 memory rate，
  还包括 SIMD scatter/control 开销）；
- matmul / cumsum 还需要修正 dot floor 和 scan 的 SIMD/SIMT 建模。
