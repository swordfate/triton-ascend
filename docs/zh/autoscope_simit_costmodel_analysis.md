# simt_costmodel 分支实现分析

> 分支来源：`https://github.com/kaixin1976/triton-ascend.git` `simt_costmodel`（4 个 commit，7898 行新增）

---

## 一、整体架构

```
用户指定 compile_mode="simd_simt"
        │
        ▼
  compiler.py: make_ttir()
        │
        ├─ 1. TTIR 生成（与 SIMD 相同）
        │
        ▼
  compiler.py: ttir_to_linalg()
        │
        ├─ 2. _run_cpp_simd_simt_costmodel(mod, metadata, opt)
        │       │
        │       ├─ SelectSimdSimtCostModelPass  ← C++ pass
        │       │   ├─ analyzeSimdSimtFeatures(module)    → 特征提取
        │       │   ├─ estimateSimdSimtCandidates(features) → 评分
        │       │   ├─ chooseBest(scores)                  → 决策
        │       │   └─ 写入 module attr: report_json, effective
        │       │
        │       └─ MaterializeSimtScopesPass  ← C++ pass
        │           └─ 把被标记为 SIMT 的 op 包进 scope.scope {simt}
        │
        ├─ 3. Python 读 effective 决策:
        │       ├─ "all_simd"      → 不做任何改变，继续 SIMD 编译
        │       ├─ "all_simt_only" → force_simt_only=True, 走纯 SIMT 路径
        │       └─ "mixed_simd_simt" → parallel_mode="mix_simd_simt",
        │                               后续结合 scope.scope 标记编译
        │
        ▼
  继续下游编译（Linalg IR → ...）
```

新增了一个 `compile_mode` 值：`"simd_simt"`。与已有的区别：

| compile_mode | 含义 |
|---|---|
| `"simd"` | 纯 SIMD |
| `"unstructured_in_simt"` | 混合（原有默认），编译器静态规则逐 op 判断 |
| **`"simd_simt"`** | **新增：用 costmodel 自动决策 kernel 级别三选一** |
| `"simt_only"` | 纯 SIMT |

---

## 二、数据模型：Profile（硬件画像）

核心思想：把硬件特性抽象成一个 JSON profile 文件，costmodel 从 profile 读取参数，而非硬编码。

### 2.1 Profile 结构

文件：`costmodel_configs/david_v100_simd_simt_v1.json`（311 行）、`costmodel_configs/simd_simt_profile_schema.json`（915 行，schema 定义）

Profile 包含以下模块：

```json
{
  "profile_version": "david_v100_simd_simt_v1",
  "target": "Ascend950PR_9579",
  "score_unit": "system_cycles_per_program",
  "minimum_confidence": "medium",

  "program_issue_scale": 1.0,
  "simd_vector_width_bits": 2048,
  "simd_setup_cycles": 128,

  "simd_ops": {            // 每个 op 类型在 SIMD 上的 throughput（ops/cycle）
    "add":  {"throughput": 32.0, "factor": 1.0, "confidence": "medium"},
    "mul":  {"throughput": 32.0, "factor": 1.0, "confidence": "medium"},
    "load": {"throughput": 8.0,  "factor": 1.0, "confidence": "medium"},
    ...
  },
  "simd_mte2_bytes_per_cycle": 512.0,   // SIMD 加载带宽
  "simd_mte3_bytes_per_cycle": 256.0,   // SIMD 存储带宽
  "simd_dot_flops_per_cycle": 4096.0,   // SIMD MatMul 吞吐

  "simt_warp_size": 32,
  "simt_setup_cycles": 64,
  "simt_ops": {            // 每个 op 类型在 SIMT 上的 throughput
    "add":  {"throughput": 64.0, "factor": 1.0, "confidence": "medium"},
    "mul":  {"throughput": 64.0, "factor": 1.0, "confidence": "medium"},
    ...
  },
  "simt_load_warp_rate": 4.0,   // 每 cycle 可发射的 load warp 数
  "simt_store_warp_rate": 4.0,
  "simt_shuffle_rate": 32.0,
  "simt_predicate_rate": 16.0,
  "simt_dot_flops_per_cycle": 2048.0,

  "coverage": { ... },      // 校准覆盖域（超出此域 → 拒评）
  "structural": { ... },    // 结构惩罚系数（不规则访存、mask、reduction等）
  "mixed_blend": { ... },   // 混合模式的 SIMD/SIMT 配比系数
  "transitions": [ ... ]    // 混合模式下的核间切换代价
}
```

### 2.2 Microbenchmark Profile

文件：`costmodel_configs/microbench/ascend_davidv100_v1.json`（259 行）

独立于 candidate profile 的微基准数据库。包含更细粒度的 op-level 测量（如 `f32_add` 吞吐、`f16_matmul` 延迟等），供 future model 使用。当前 v2 模型尚未完全接入所有 microbenchmark 数据。

---

## 三、Phase 1：特征提取（Feature Extraction）

文件：`SimdSimtCostModel.cpp:analyzeSimdSimtFeatures()`

**只做静态分析，不涉及时序模拟。** 遍历 TTIR ModuleOp 中的每一个 operation，分类统计：

### 3.1 Op 分类与计数

```cpp
struct SimdSimtFeatureSummary {
  // 访存类
  int64_t loadOps, storeOps, gatherOps, atomicOps, histogramOps;
  double  loadBytes, storeBytes;

  // 计算类
  int64_t reduceOps, scanOps, dotOps, broadcastOps;
  int64_t arithOps, mathOps;
  int64_t addOps, subOps, mulOps, divOps, maxOps, absOps;
  int64_t expOps, logOps, cmpOps, selectOps, castOps, clampOps;

  // 结构特征
  int64_t maxTensorRank, maxTensorNumel, maxElementBits;
  int64_t maskTensorOps, maskRankSum, maskBroadcastOps;
  int64_t pointerTensorOps, pointerUnstructuredDims;
  int64_t laneDependentPointerOps;  // ← 间接访存密度（核心指标）
  int64_t scalarLoadOps, scalarStoreOps;  // ← 标量访存 ← SIMD 弱项

  // 循环信息
  int64_t staticLoopCount, staticLoopTripCountSum, staticLoopTripCountMax;

  // MatMul 信息
  int64_t dotFlops;
  std::vector<std::array<int64_t,3>> dotMNK;  // 每个 dot 的 (M,N,K)
};
```

### 3.2 加权计算

不是简单计数——每个 op 按其所在的循环嵌套深度加权：

```cpp
int64_t multiplier = 1;
for (Operation *parent = op->getParentOp(); parent;
     parent = parent->getParentOp()) {
  if (auto forOp = dyn_cast<scf::ForOp>(parent))
    multiplier *= estimatedTripCount(forOp);  // 静态可解析的 trip count
  // if/while 也计为乘数
}
features.weightedOps["add"] += multiplier;
features.opElements["add"]  += tensorNumel * multiplier;
```

这意味着：一个 `add` 在 `for i in range(1024)` 循环内的权重是裸 `add` 的 1024 倍。

### 3.3 关键结构信号

| 特征 | 含义 | 对决策的影响 |
|---|---|---|
| `laneDependentPointerOps` | 间接寻址的指针操作数 | 高 → SIMT 优势大（`indirect_load` 无需展开） |
| `pointerUnstructuredDims` | 非结构化指针维度数 | 高 → SIMD 需展开成标量循环 → SIMT 优势 |
| `scalarLoadOps` / `scalarStoreOps` | 标量访存数 | 高 → SIMD 弱项 → SIMT 优势 |
| `maskRankSum` | mask 的总 rank | 高 → SIMD 需材料化 mask → SIMT 优势 |
| `dotFlops` | MatMul 总算量 | 高 → SIMD Cube 单元优势 |
| `staticLoopTripCountSum` | 循环迭代总数 | 高 → SIMD pipelining 优势 |

---

## 四、Phase 2：评分模型（Analytical Scoring）

文件：`SimdSimtCostModel.cpp:estimateSimdSimtCandidates()`

### 4.1 资源成本（Resource Cost）

对 profile 中的每个 op 类型，分别算 SIMD 和 SIMT 的 cycles：

```cpp
// SIMD：操作被向量化（vector_width=2048bits / element_bits）
simd_cycles = ceil(elements / vector_width) / profile.simdOps[op].throughput;

// SIMT：操作被 warp 化
simt_cycles = elements / profile.simtOps[op].throughput;
```

**Memory**：
```cpp
// SIMD: MTE2(load) / MTE3(store) DMA 通道
simd_memory = max(load_bytes / mte2_rate, store_bytes / mte3_rate);

// SIMT: warp-coalesced global memory
simt_memory = load_warp_insns / load_warp_rate + store_warp_insns / store_warp_rate;
```

**Dot (MatMul)**：
```cpp
simd_dot = simd_dot_setup + dot_flops / simd_dot_flops_per_cycle;
simt_dot = simt_dot_setup + dot_flops / simt_dot_flops_per_cycle;
```

**SIMT 特有**：
```cpp
// Shuffle：warp 内 reduction 需要 shuffle
simt_shuffle = weighted_reductions * (max_nel / warp_size) * log2(warp_size) / shuffle_rate;

// Predicate：mask 材料化
simt_predicate = mask_rank_sum * (max_nel / warp_size) / predicate_rate;
```

**合成为 analytical cycles**：
```cpp
// SIMD: compute + dot 与 memory 取 max（可以并行）
simd_payload = max(simd_compute + simd_dot, simd_memory);
simd_analytical = simd_setup + simd_payload * program_issue_scale;

// SIMT: compute + dot + shuffle 与 memory 取 max，再加 predicate
simt_payload = max(simt_compute + simt_shuffle + simt_dot, simt_memory) + simt_predicate;
simt_analytical = simt_setup + simt_payload * program_issue_scale;
```

注意：SIMD 用 `max(compute, memory)`（DMA 与计算并行），SIMT 用 `compute + memory`（warp 调度不能同时做计算和访存）。这反映了两种硬件的根本差异。

### 4.2 结构惩罚（Structural Penalty）

某些 kernel 结构在 SIMT 上会有额外的开销。这些以**乘数**的形式加到 simt_analytical 上：

```cpp
structural_penalty = 0;

// 不规则访存（indirect load/store 的模式切换）
structural_penalty += min(cap, irregular_density * irregular_per_density);

// Mask 材料化（SIMT 需要显式 predicate）
structural_penalty += min(mask_cap, mask_rank_sum * per_mask_rank);

// Reduction lowering（warp shuffle + shared memory）
structural_penalty += min(reduction_cap, weighted_reductions * per_weighted_reduction);

// 静态循环控制（循环展开/流水线在 SIMD 上有优势）
structural_penalty += min(loop_cap, static_loop_trip_sum * per_static_loop_trip);

// Control flow（if/while 在 SIMT 上有 divergence 代价）
structural_penalty += has_control_flow ? control_flow_penalty : 0;
```

然后应用到 SIMD 候选：
```cpp
all_simd_cost = max(simd_analytical_cycles,
                    simt_analytical_cycles * (1 + structural_penalty));
//                     ↑ floor: SIMD 不能比 SIMT+penalty 更差
all_simt_only_cost = simt_analytical_cycles;
```

### 4.3 混合模式成本（Mixed Cost）

混合模式的核心：部分 SIMD + 部分 SIMT + 核间切换代价。

```cpp
// mixed_blend: SIMD 占的比例（0=全 SIMT, 1=全 SIMD）
mixed_blend = base
            + min(loop_cap, static_loop_trip_sum * per_loop)
            + min(mask_cap, mask_broadcast_ops * per_mask)
            + min(reduction_cap, reductions * per_reduction)
            + control_flow_penalty;

mixed_blend = min(mixed_cap, max(0, mixed_blend));

// 混合 paylaod: SIMT payload + blend_rate × (SIMD payload - SIMT payload)
mixed_payload = simt_payload + mixed_blend * (simd_payload - simt_payload);

// 切换代价（transition cost）：混合模式的 setup 比 standalone SIMT 高
mixed_transition = nearest_transition.empty_simt_setup - simt_setup;

mixed_cost = simt_setup + mixed_transition + max(0, mixed_payload);
```

### 4.4 Gating（决策门控）

即使算出了三个候选的分数，也不一定采纳——需要通过一系列 gate：

| Gate | 条件 | 不通过的行为 |
|---|---|---|
| target_compatible | profile.target 匹配当前芯片 | 拒评 |
| calibration_covered | 特征在 profile.coverage 域内 | 拒评（可配置跳过） |
| selection_score_valid | 特征能被 profile 覆盖 | 拒评 |
| unsupported_terms | 没有 profile 不认识的 op | 降级为 "none" confidence |
| ranking_confidence | 综合 confidence ≥ minimum | 拒评 |
| decision_advantage | best_score + 64 cycles + 10% margin < all_simd | 退回 all_simd |

```cpp
// 最终判断
report.gatePassed = report.gateReasons.empty();
if (!gatePassed) → effective = "all_simd"（默认回退，安全）
```

---

## 五、Materialize：混合模式的 IR 改造

文件：`MaterializeSimtScopes.cpp`

当选出 `"mixed_simd_simt"` 后，需要告诉下游编译器哪些 op 走 SIMT。方法：对每个标记了 `ascend.simt_selected` 的 op，包裹进 `scope.scope {simt}` region：

```mlir
// 原来是普通 op
%result = tt.load %ptr : tensor<8x32xf32>

// Materialize 之后
%result = "scope.scope"() <{vector_mode = "simt"}> ({
  %inner = tt.load %ptr : tensor<8x32xf32>
  scope.return %inner : tensor<8x32xf32>
}) : () -> tensor<8x32xf32>
```

下游 AscendNPU IR 遇到 `scope.scope {simt}` 时就知道这片代码要 SIMT 编译。

---

## 六、Python 侧集成

文件：`compiler.py:141-201`

```python
def _run_cpp_simd_simt_costmodel(mod, metadata, opt) -> str:
    """只在 compile_mode="simd_simt" 时触发"""

    # 1. 加载 profile
    profile = opt.auto_simt_model_profile or "david_v100_simd_simt_v1.json"

    # 2. 运行两个 C++ pass
    #    - SelectSimdSimtCostModel: 特征提取 + 评分 + 决策
    #    - MaterializeSimtScopes: 标记 SIMT op → scope.scope
    pm = ir.pass_manager(mod.context)
    add_select_simd_simt_costmodel(pm, mode, profile, ...)
    add_materialize_simt_scopes(pm)
    pm.run(mod)

    # 3. 读取决策
    report = get_attr(mod, "ascend.simt_costmodel.report_json")
    effective = get_attr(mod, "ascend.simt_costmodel.effective")

    # 4. 应对不同决策
    if effective == "all_simt_only":
        # 全 SIMT: 设置 force_simt_only, 跳过 Linalg 降级
        metadata["force_simt_only"] = True
        # 运行 RowCoalescing pass (SIMT 内存合并优化)
        # 然后 inline scope → 直接返回 TTIR
        return str(mod)
    elif effective == "mixed_simd_simt":
        # 混合模式: 记录 requested_kind → 后续编译使用 scope.scope 标记
        metadata["auto_simt_requested_kind"] = "mixed_simd_simt"
    # else all_simd: 默认路径，不做任何改变

    # 继续正常 SIMD 编译流程
    # (TritonToStructured → TritonToLinalg → ...)
```

`compile_mode="simd_simt"` 在 `NPUOptions.__post_init__` 中设置了 `parallel_mode="mix_simd_simt"` 和 `force_simt_template=True`，确保后续 pass 能识别 scope 标记。

---

## 七、测试覆盖

文件：`unittest/costmodel_ut/SimdSimtCostModelTest.cpp`（253 行）

三类测试：
1. **特征提取测试**：`gatherDotFeatures()` 构造特征 → `estimateSimdSimtCandidates()` → 验证 `decision != AllSIMD`（因为间接访存 + dot → SIMT 更优）
2. **纯 SIMD 测试**：无间接访存的 arith kernel → 验证 `decision == AllSIMD`
3. **Gate 测试**：超出 coverage → 验证 `gatePassed == false`

---

## 八、关键设计决策总结

| 设计选择 | 为什么 |
|---------|--------|
| **kernel 级三选一**（非逐 op 决策） | costmodel 预测的是整体 kernel 性能，逐 op 决策需要更细粒度的模型 |
| **Profile 驱动**（硬件参数在 JSON，不在代码） | 不同芯片/步进只需换 profile，不改代码 |
| **两阶段分离**（特征提取 vs 评分） | 特征提取可复用于调试/报告；评分公式可独立迭代 |
| **Gating 保守策略**（不满足条件退回 all_simd） | 不能因 costmodel 选错导致性能倒退 |
| **Materialize 用 MLIR scope** | 不修改 Triton dialect，下游编译器自然识别 |
| **不依赖 autotune/hardware bench** | 纯静态分析，零硬件开销 |

---

## 九、与你的 costmodel（PipelineAnalysisPass）的关系

| 维度 | 你的 PipelineAnalysisPass | 这个 simt_costmodel |
|------|--------------------------|-------------------|
| 目标 | 预测 kernel 在 SIMD 上的绝对 latency (us) | 比较 SIMD/SIMT/Mixed 三个候选的**相对分数** |
| 方法 | 自底向上：逐 op 调度 + roofline + wave 展开 | 自顶向下：统计特征 + profile 公式 |
| 硬件模型 | Cube/Vector 分离、loop multiplier | SIMD throughput / SIMT throughput 对查表 |
| 依赖 | C++ pass，在 costmodel 项目中 | C++ pass + JSON profile，在 costmodel 项目中 |
| 集成点 | `run_costmodel_inproc()` → autotuner | `ttir_to_linalg()` 中作为编译 pass 运行 |
| 输出 | "Estimated Time: X us" | effective 决策 + JSON report |

两者是互补关系：
- **PipelineAnalysisPass**：精确的 SIMD 延时预测 → 用于 autotuner 剪枝（已集成）
- **SimdSimtCostModel**：快速的 SIMD/SIMT 对比 → 用于 compile_mode 自动选择（本分支实现）
