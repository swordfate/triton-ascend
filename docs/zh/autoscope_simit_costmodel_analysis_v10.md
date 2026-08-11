# simt_costmodel 分支 v10 实现分析

> 分支来源：`https://github.com/kaixin1976/triton-ascend.git` `simt_costmodel`（squash 为 2 个 commit，~10000 行新增）
> 对应 commit：`400f73560`
> 旧版（v6）文档：`autoscope_simit_costmodel_analysis.md`（独立文档，非本文档的前一版本）
> v6-vs-v10 对比：`autoscope_simit_costmodel_v6_vs_v10.md`

---

## 一、整体架构

```
用户指定 compile_mode="simd_simt" + auto_simt_scope_mode="auto"
        │
        ▼
  compiler.py: make_ttir() → TTIR ModuleOp
        │
        ▼
  compiler.py: _run_cpp_simd_simt_costmodel(mod, metadata, opt)
        │
        ├─ 0. buildMixedSimtAnchorPlan(module)
        │      └─ 遍历 TTIR，识别 6 种 SIMT 锚点（anchor），构建不可变计划
        │
        ├─ 1. SelectSimdSimtCostModelPass
        │      ├─ analyzeSimdSimtFeatures(module, anchorPlan) → 双重特征提取
        │      │   ├─ 全 kernel 特征（统计所有 op，用循环乘数加权）
        │      │   └─ SIMT anchor 特征（只统计 anchor 内的 op）
        │      │
        │      ├─ estimateSimdSimtCandidates(features, options)
        │      │   ├─ 加载两层 profile（selection + microbenchmark）
        │      │   ├─ 校准覆盖域判断（coverage check）
        │      │   ├─ 分析公式评分（A_SIMD, A_SIMT, structural penalty）
        │      │   ├─ Resource partition 计算 mixed cost
        │      │   ├─ Event route calibration（测量乘数校正）
        │      │   └─ Gate 系统（target/compatibility/confidence/gain）
        │      │
        │      ├─ 决策 + 写入 module attr
        │      │   ├─ effective = "all_simd" | "all_simt_only" | "mixed_simd_simt"
        │      │   └─ report_json（完整 JSON 报告）
        │      │
        │      └─ materializeSimtAnchorPlan(module, anchorPlan)
        │          └─ 把 anchor op 包进 scope.scope{vec_mode="simt"}
        │
        ├─ 2. Python 读 effective 决策:
        │      ├─ "all_simd"      → 正常 SIMD 编译
        │      ├─ "all_simt_only" → force_simt_only, RowCoalescing, 纯 SIMT
        │      └─ "mixed_simd_simt" → parallel_mode="mix_simd_simt"
        │
        ▼
  继续下游编译
```

核心思路：**在编译期用一个基于硬件测量数据的分析模型，自动为每个 kernel 选择最优的 SIMD/SIMT 执行路线。**

模型不是凭空猜测——所有参数都来自真实的微基准测试（microbenchmark），运行在 A5 芯片（Ascend 950PR / dav-c310）上。评分公式是 first-principles 的（操作吞吐 × 操作数 + 内存带宽 / 数据量），然后用 Event 测量残差校正。

---

## 二、代码目录结构

新版将代码重组为三块：

```
third_party/ascend/costmodel/
├── include/AscendModel/
│   ├── Profile/
│   │   └── MicrobenchmarkProfile.h       ← 模型无关的硬件测量 loader
│   └── RouteModel/
│       ├── SimdSimtCostModel.h           ← 核心 API：特征 + 评分 + 报告
│       ├── SimtAnchorAnalysis.h          ← 锚点模式匹配（6 种 SIMT 锚点）
│       └── SimtSelection.h               ← 执行路由契约（scope.scope 桥接）
│
├── lib/AscendModel/
│   ├── Profile/
│   │   └── MicrobenchmarkProfile.cpp     ← 测量加载 + sha256 + 单位校验
│   └── RouteModel/
│       ├── SimdSimtCostModel.cpp         ← 主模型（~2841 行）
│       ├── SimtAnchorAnalysis.cpp        ← 锚点分析（~602 行）
│       └── Transforms/
│           ├── SelectSimdSimtCostModel.cpp  ← 选择 pass（~229 行）
│           └── MaterializeSimtScopes.cpp    ← 材料化 pass（~290 行）
│
└── profiles/
    ├── microbench/
    │   ├── ascend_davidv100_v1.json       ← 模型无关硬件测量
    │   └── microbenchmark_profile_schema.json
    └── simd_simt/
        ├── david_v100_simd_simt_v1.json   ← 路由模型策略 + 校准
        └── simd_simt_profile_schema.json  ← JSON Schema (~901 行)
```

**关键设计决策**：将硬件测量数据（microbenchmark profile）与模型参数（selection profile）分离。测量数据是模型无关的——物理值 + 单位 + 时钟域 + 来源 + 置信度。模型参数引用这些测量，但不重新定义它们。这样不同模型（绝对 cost model 和 route model）可以共享同一套测量。

---

## 三、数据模型：两层 Profile 分离

### 3.1 Microbenchmark Profile（模型无关）

文件：`profiles/microbench/ascend_davidv100_v1.json`（259 行）

设计理念：**每个测量值自带 provenance（来源追溯）**。

```json
{
  "profile_version": "david-v100-shared-microbench-20260730-v2",
  "target": "Ascend950PR/dav-c310",
  "measurements": {
    "simd.vector_width_bits": {
      "value": 2048,
      "unit": "bit",
      "cycle_domain": "none",
      "scope": "single_aiv",
      "source_kind": "architecture_fact",
      "source": "Ascend vector ISA width",
      "confidence": "high"
    },
    "simt.setup.empty_with_barrier": {
      "value": 141.0,
      "unit": "system_cycle",
      "cycle_domain": "SYS_CNT",
      "scope": "single_aiv_serialized_empty_vf_with_per_iteration_barrier",
      "source_kind": "isolated_microbenchmark",
      "source": "triton_cases/SIMT_Test/meas.cce",
      "confidence": "medium"
    }
  }
}
```

关键设计：`unit` 和 `cycle_domain` 不是给人看的注释——C++ loader 如果你引用了错误单位或时钟域的测量会直接报错：

```cpp
// MicrobenchmarkProfile.cpp
auto value = microbench->requireValue(
    "simd.f32.add.throughput",              // key
    "vector_instruction/system_cycle",       // expected unit
    "SYS_CNT"                                // expected cycle domain
);
// 如果实际测量不是 vector_instruction/system_cycle 或不在 SYS_CNT 域 → 直接报错
```

测量项分类（共计 25 项）：

**时钟和架构常量**（4 项）：
- `clock.sys_cnt.frequency_mhz` = 988.9 MHz（SYS_CNT 计数器频率）
- `clock.device_compute.frequency_mhz` = 1650 MHz（设备计算时钟）
- `simd.vector_width_bits` = 2048 bit（AIV 向量宽度）
- `simt.warp_size` = 32 lane

**SIMD 操作**（2 项）：
- `simd.f32.add.throughput` = 3.30 vector_inst/system_cycle
- `simd.f32.add.dependent_latency` = 1.818 system_cycle

**SIMT Setup**（2 项）：
- `simt.setup.empty` = 115.0 cycle（无 barrier slope）
- `simt.setup.empty_with_barrier` = 141.0 cycle（有 barrier，被模型使用）

**SIMT 操作**（1 项）：
- `simt.f32.add.throughput` = 141.0 scalar_op/system_cycle

**SIMT Shuffle**（2 项）：
- `simt.shuffle.throughput` = 0.817 warp_inst/system_cycle（32 warp, ILP4）
- `simt.shuffle.dependent_latency` = 27.28 cycle

**SIMT GM 内存**（4 项）：
- `simt.gm.load.throughput` = 0.176 warp_inst/system_cycle
- `simt.gm.store.throughput` = 0.129 warp_inst/system_cycle
- `simt.gm.load.bandwidth` = 22.55 byte/system_cycle
- `simt.gm.store.bandwidth` = 16.53 byte/system_cycle

**SIMT UB 内存**（4 项）：
- `simt.ub.load.throughput` = 0.507 warp_inst/system_cycle
- `simt.ub.store.throughput` = 0.530 warp_inst/system_cycle
- `simt.ub.load.bandwidth` = 64.94 byte/system_cycle
- `simt.ub.store.bandwidth` = 67.86 byte/system_cycle

**Transition Harness**（6 项）：
- `simt.setup.transition_harness_net.warps_{1,2,4,8,16,32}` = 182-223 cycle

### 3.2 Selection Profile（模型相关）

文件：`profiles/simd_simt/david_v100_simd_simt_v1.json`（332 行）

这个 profile 包含一切"模型特定"的参数——策略、校准、覆盖域、惩罚系数。它通过 `throughput_measurement` / `empty_launch_measurement` 等 key 引用 microbenchmark profile。

```json
{
  "schema_version": 4,
  "profile_version": "david-v100-simd-simt-20260804-v10",
  "microbenchmark_profile": "../microbench/ascend_davidv100_v1.json",

  "policy": {
    "minimum_confidence_for_decision": "low"
  },

  "selection_calibration": {
    "program_issue_scale": 8.0,
    "coverage": {
      "minimum_irregular_density": 0.25,
      "tiny_dot_flops_max": 16384,
      "rowwise_loop_trip_sum_max": 32,
      ...
    },
    "simd_structural_penalty_ratio": {
      "irregular_per_density": 0.8,
      "per_mask_rank": 0.022,
      "per_weighted_reduction": 0.02,
      ...
    },
    "event_route_score_multiplier": {
      "domains": {
        "masked_rowwise_reduction": { ... },
        "tiny_irregular_dot": { ... },
        "triangular_solve_loop": { ... }
      }
    }
  },

  "simd": {
    "ops": {
      "f32.add": { "throughput_measurement": "simd.f32.add.throughput" },
      "f32.mul": { "relative_to": "f32.add", "factor": 1.0 }
    },
    "memory": { "vector_mte2_bytes_per_system_cycle": 202.25 },
    "dot": { "flops_per_system_cycle": 4096.0 }
  },

  "simt": {
    "ops": {
      "f32.add": { "throughput_measurement": "simt.f32.add.throughput" },
      "f32.mul": { "relative_to": "f32.add", "factor": 1.0 }
    },
    "mixed_setup_fallback": {
      "1":  { "measurement": "simt.setup.transition_harness_net.warps_1" },
      "4":  { "measurement": "simt.setup.transition_harness_net.warps_4" },
      "32": { "measurement": "simt.setup.transition_harness_net.warps_32" }
    }
  }
}
```

**op 定义的 relative_to 机制**：`f32.mul` 不直接引用 microbenchmark，而是引用 `f32.add` 的 throughput × factor。这意味着只有少数"基准 op"有微基准测量，其他 op 通过相对系数推导。

**SIMD memory 不是从 microbenchmark 来的**：`vector_mte2_bytes_per_system_cycle = 202.25` 是模型特定的 fallback 值（对应 200 GB/s），profile 中明确写了"must not be presented as the davidV100 75/71 GB/s specification"。这是有意为之——在某些场景下需要 conservative estimate，而不是 bare-metal peak。

---

## 四、Phase 0：Anchor Plan 构建（新增）

**这是 v10 最重要的新增机制。**

旧版的问题是："哪些 op 用 SIMT" 的判断散布在 feature extraction、scoring、materialization 三个地方，如果三者看到不同的 op 集合，就会不一致。新版用一个**不可变的 `SimtAnchorPlan`** 统一三个阶段。

### 4.1 什么是 Anchor？

Anchor 是 TTIR 中能被识别为"适合 SIMT 执行"的操作模式。有 6 种类型：

```
SimtAnchorKind:
  DirectGather                 → tt.gather
  LoadedIndexDependentMemory   → tt.load/tt.store 指针依赖 loaded index
  Histogram                    → tt.histogram
  PlainOneDimensionalCumsum    → tt.scan (只有一条轴 extent > 1)
  TensorAtomic                 → tt.atomic_rmw / tt.atomic_cas
  TriangularSolveLoop          → scf.for 中 (vector_load+axis0_reduce+masked_update)
```

### 4.2 每种 Anchor 的 lowering 能力

不是所有 anchor 在所有路由上都可用。每个 anchor 有三种路由的 `CandidateLoweringStatus`：

```
enum CandidateLoweringStatus:
  Native              = 原生支持
  BackendConditional  = 需要后端验证（可能被 Event 验证提升）
  AliasesMixed        = 该路由实质等同于 mixed 模式
  Unsupported         = 不支持
```

具体每种 anchor 的状态表：

| Anchor | all-SIMD | all-SIMT | mixed | 备注 |
|--------|----------|----------|-------|------|
| DirectGather | Native | BackendConditional | Native | |
| LoadedIndexDependentMemory | Native | BackendConditional | Native | |
| Histogram | **Unsupported** | **Unsupported** | Native | 只能在 mixed 下工作！需要 static rank-1 int input + i32 bins |
| PlainOneDimensionalCumsum | **AliasesMixed** | BackendConditional | Native | 需要 supported dtype + axis extent |
| TensorAtomic | Native | BackendConditional | Native | 需要 supported dtype/operation/offset type |
| TriangularSolveLoop | Native | BackendConditional | Native | 需要 manual scope pattern 匹配 |

注意 Histogram 的特殊性：它**只能**在 mixed 模式下工作。如果 kernel 包含 histogram，all-SIMD 和 all-SIMT 候选都被标记为 Unsupported，只有 mixed 候选可选。

### 4.3 Anchor Plan 构建过程

`buildMixedSimtAnchorPlan(module, compileOn91095)`：

```cpp
SimtAnchorPlan plan;
// 1. Pre-order walk: 按源码顺序遍历
module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // 2. 尝试识别为 anchor
    auto descriptor = analyzeAnchor(op, compileOn91095);
    if (!descriptor)
        return WalkResult::advance();  // 不识别 → 继续遍历子节点

    // 3. 是 anchor → 加入 plan, 跳过子节点（嵌套 op 不重复计数）
    plan.anchors.push_back(std::move(*descriptor));
    return WalkResult::skip();
});

// 4. 组合出 kernel 级别的 lowerability
for (anchor : plan.anchors) {
    plan.kernelLowerability.allSimd     = combine(all anchors' allSimd status)
    plan.kernelLowerability.allSimtOnly = combine(all anchors' allSimtOnly status)
    // mixed: 至少一个 native 且没有被 blocked → Native
}
```

**为什么用 PreOrder + skip**：防止嵌套的 op 被重复识别。例如一个 `tt.gather` 在 `scf.for` 内——如果 for 本身被识别为 TriangularSolveLoop anchor，那么 gather 不应该再算一次。

### 4.4 analyzeAnchor 函数详解

这是锚点模式匹配的核心。每个 op 按类型进行深度分析：

**tt.gather**：最简单的识别。直接标记为 `DirectGather`。

**tt.histogram**：
```cpp
// 要求: input 是 static rank-1 tensor of i8/i16/i32/i64
//       result 是 static rank-1 i32 tensor
//       input_elements > 0 && num_bins > 0
// 不满足 → mixed = Unsupported
```

**tt.scan**（PlainOneDimensionalCumsum）：
```cpp
auto facts = analyzePlainOneDimensionalCumsum(op);
// 要求:
//   1. 只有一个 axis 的 extent > 1（其他 axis = 1）
//   2. region body 中只有一个真正的 combine op: arith.addf 或 arith.addi
//   3. terminator 是 tt.scan.return
//   （arith.extf/truncf/bitcast 不算 combine op）
//
// all-SIMD = AliasesMixed: cumsum 在 SIMD 上有符号，但实际执行会被编译器
//   转成 SIMT 模板 → 等效于 mixed
```

**tt.atomic_rmw / tt.atomic_cas**：
```cpp
// 提取: value_type, offset_type, operation, has_mask, mask_active_fraction
//       address_is_lane_varying, address_depends_on_loaded_index
//       result_used, contention
// 验证: supported type/operation 组合
// 特殊: f16/bf16 atomic 且 result_used → unsupported
//       (f16/bf16 atomic old_value 语义需要额外验证)
```

**TriangularSolveLoop 检测**（isTriangularSolveLoop）：
```cpp
// 条件 1: 是 scf.for
// 条件 2: body 中包含 tt.load (rank-1, shape[0]=16)   ← 向量加载
// 条件 3: body 中包含 tt.reduce (axis=0)              ← 轴0规约
// 条件 4: body 中包含 arith.select                     ← 掩码更新
// 条件 5: iter_args 有 16×16 triangular state，或者
//         有 ≥1 个 sibling 循环有同样的 load/reduce/select 模式

// 为什么条件 5 有两种路径:
//   - 单循环变体: 16×16 的 triangular state 在 iter_args 中显式存在
//   - 多循环变体: state shape 不在 loop arguments 中，但有多于一个
//     sibling 循环有相同的模式 → 保留这种形式
```

**LoadedIndexDependentMemory 检测**（isLoadedIndexDependentMemoryOp）：
```cpp
// 对 tt.load / tt.store:
//   1. operand 必须是有静态 shape 的 RankedTensorType (rank ≤ 5)
//   2. pointer operand 的 SSA backward slice 必须能到达 tt.load 或 tt.gather

// pointerDependsOnLoadedIndex(memoryOp):
//   从 memoryOp 的 pointer 开始 BFS:
//     - 遇到 BlockArgument → 追踪 scf.for 的 iter_args
//     - 遇到 tt.load 或 tt.gather → return true (找到了!)
//     - 其他 op → 继续追踪 operands
```

这个检查是**真正的数据依赖分析**，不是旧版简单的 rank-based proxy。它说：这个 load/store 的地址是不是从之前加载的数据推导出来的？如果是 → 这就是`LoadedIndexDependentMemory` 模式。

---

### 4.7 为什么 Anchor 识别必须放在特征提取之前

这是一个设计决策，不是实现细节。核心原因：**特征提取内部需要知道每个 op 是否在 anchor 内部**，这决定了统计维度的分叉方式，不可事后推算。

#### 直接依赖：`inAnchor` 标志

特征提取的 `module.walk` 中对每个 op 都调用 `isInAnchor` 判定（`SimdSimtCostModel.cpp` line 1813-1818）：

```cpp
auto isInAnchor = [&](Operation *op) {
    for (Operation *current = op; current; current = current->getParentOp())
        if (anchorSet.contains(current))
            return true;
    return false;
};
```

`isInAnchor` 在遍历中决定了下面所有计数走哪个分支：

```cpp
// 全局统计 + anchor 内统计，同时进行
features.weightedOps[weightedKind] += loopMultiplier;         // ← 全局
features.opElements[weightedKind] += elements * loopMultiplier;  // ← 全局
if (inAnchor) {
    features.simtAnchors.weightedOps[weightedKind] += loopMultiplier;  // ← anchor 内
    features.simtAnchors.opElements[weightedKind] += elements * loopMultiplier;  // ← anchor 内
}
```

**这不是"先全局统计，再从中过滤出 anchor 子集"——而是两类计数同时进行**，在同一个 `module.walk` 中完成。整个特征提取有 **20+ 个字段** 都按这个分叉模式统计：

| 统计项 | 全局（用于 all-SIMD / all-SIMT 评分） | Anchor 内（用于 mixed 评分） |
|--------|---------------------------------------|------------------------------|
| weightedOps | `features.weightedOps` | `features.simtAnchors.weightedOps` |
| opElements | `features.opElements` | `features.simtAnchors.opElements` |
| loadBytes | `features.loadBytes` | `features.simtAnchors.loadBytes` |
| maskRankSum | `features.maskRankSum` | `features.simtAnchors.maskRankSum` |
| predicateElements | `features.predicateElements` | `features.simtAnchors.predicateElements` |
| staticLoopCount | `features.staticLoopCount` | `features.simtAnchors.staticLoopCount` |
| loadedIndexDependentMemoryOps | `features.loadedIndexDependentMemoryOps` | `features.simtAnchors.loadedIndexDependentMemoryOps` |
| ... | ... | ... |

#### 如果顺序反过来会怎样

**选择 A：事后过滤**

`module.walk` 完成后 SSA 上下文已丢弃，feature extraction 不保存 op→anchor 的映射。事后无法推算哪些 op 在 anchor 内部——因为 anchor 的判定依赖 `isLoadedIndexDependentMemoryOp` 的 BFS 分析（需要 traverse SSA graph），而 feature summary 只保存聚合数字，不保存 SSA 关系。

**选择 B：分两次 walk**

先走全量 feature extraction（不区分 anchor），再 walk 一次只走 anchor 子树提取 anchor 特征。两个问题：

1. **两次 walk 的 `loopMultiplier` 可能不一致**——anchor 嵌套处理（`scf.for` 在 anchor 内 vs 外）会影响 `isInAnchor` 的传播，两次 walk 看到的拓扑不同时 anchor 内的计数就不可靠。
2. **效率**：虽然影响不大，但违反"数据只产生一次"的原则。更重要的是它**引入了一致性 bug 的可能性**——比如 anchor 识别逻辑的改动可能导致两次 walk 中同一个 op 的归属不同。

#### 更根本的依赖：物化（Materialization）

```cpp
// SelectSimdSimtCostModel.cpp line 202-206
if (effective == kMixedSimdSimt &&
    failed(materializeSimtAnchorPlan(module, anchorPlan))) {
    signalPassFailure();
    return;
}
```

`anchorPlan` 不光用于评分，评分决定了 `mixed` 之后，**同一个 `anchorPlan`** 被传给 `materializeSimtAnchorPlan`，用它 `materializableRoots()` 的 Operation 指针列表把具体 op 包进 `scope.scope{vec_mode="simt"}`。

#### 循环依赖：mixed 评分需要 anchor 信息

mixed 候选的计费方式（`SimdSimtCostModel.cpp` line 2736-2740）：

```cpp
// 把 anchor 内 ops 按 SIMT 费率计，其余按 SIMD 费率计
mixedSimdRegularComputeCycles = simdCompute(all ops) - simdCompute(anchor ops);
mixedSimtAnchorComputeCycles  = simtCompute(anchor ops);
mixedCost = mixedSimdRegular + mixedSimtAnchor + boundary;
```

如果没有 anchor 特征（`features.simtAnchors.*`），mixed 候选根本无法评分。如果评分在前、识别在后，就会形成循环依赖：

```
评分 → 决定 mixed → 需要知道哪些 op 是 anchor → 需要先识别
识别 → 为了什么？ → 为了评分用的分叉特征 → 需要在评分之前
```

#### 总结

整个 pipeline 的顺序是**硬依赖链**，不是可选优化：

```
识别 anchor → 提取特征（分叉）→ 评分（含 mixed）→ 物化（复用 anchor plan）
     │               │                   │                  │
     │               │                   │                  └── 需要 Operation* 列表
     │               │                   └── mixed 评分需要 anchor 特征
     │               └── inAnchor 标志决定 20+ 字段的分叉统计
     └── BFS 分析 SSA 依赖链，结果是不可变的 Operation* 集合
```

每个后续步骤都消费前一步的输出。颠倒任何一步都会导致上一步的输出无法产生。

---

## 五、Phase 1：特征提取（Feature Extraction）

文件：`SimdSimtCostModel.cpp:analyzeSimdSimtFeatures(module, anchorPlan)`，约 430 行

### 5.1 双重特征提取

v10 与旧版最大的区别：特征提取不再是"统计所有 op"那么简单。现在要同时统计**两个层面**：

1. **全 kernel 特征**（`SimdSimtFeatureSummary`）：kernel 中所有 op 的统计
2. **SIMT anchor 特征**（`SimtAnchorFeatureSummary`）：只有 anchor plan 中标记的 op 的统计

```cpp
auto anchorRoots = anchorPlan.materializableRoots();  // 获取 anchor op 列表
DenseSet<Operation*> anchorSet(anchorRoots);

// lambda: 判断 op 是否在 anchor 内
auto isInAnchor = [&](Operation *op) {
    for (Operation *current = op; current; current = current->getParentOp())
        if (anchorSet.contains(current))
            return true;
    return false;
};
```

这个 `isInAnchor` 不是简单的 `anchorSet.contains(op)`——它检查 op 的**祖先链**。因为 anchor 可能是一个 `scf.for`（TriangularSolveLoop），它内部的所有 op 都应该算在 anchor 内。

### 5.2 主遍历循环

```cpp
module.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    const int64_t elements = getOperationElements(op);
    const int64_t loopMultiplier = getLoopMultiplier(op);
    const bool inAnchor = isInAnchor(op);
    ...
```

`loopMultiplier` 的计算与旧版相同：从当前 op 出发沿 MLIR 树向上走，累乘所有 `scf.for` 的静态 trip count。

```cpp
static int64_t getLoopMultiplier(Operation *op) {
    int64_t multiplier = 1;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
        if (parent->getName().getStringRef() != "scf.for")
            continue;
        int64_t tripCount = getStaticLoopTripCount(parent);
        multiplier *= tripCount;  // 默认值 1（保守）
    }
    return multiplier;
}
```

### 5.3 Op 分类与 dual 统计

对每个关键 op，同时更新全 kernel 计数和 anchor 内计数：

```cpp
auto incrementRaw = [&](int64_t &counter, int64_t &anchorCounter) {
    ++counter;
    if (inAnchor)
        ++anchorCounter;
};

if (name == "tt.load")
    incrementRaw(features.loadOps, features.simtAnchors.loadOps);
else if (name == "tt.store")
    incrementRaw(features.storeOps, features.simtAnchors.storeOps);
else if (name == "tt.reduce")
    incrementRaw(features.reduceOps, features.simtAnchors.reduceOps);
// ... 等等
```

**加权 op 计数**（`weightedOps` 和 `opElements`）：

```cpp
llvm::StringRef weightedKind = classifyWeightedOp(name);
if (!weightedKind.empty()) {
    features.weightedOps[weightedKind] += loopMultiplier;
    features.opElements[weightedKind]  += elements * loopMultiplier;
    if (inAnchor) {
        features.simtAnchors.weightedOps[weightedKind] += loopMultiplier;
        features.simtAnchors.opElements[weightedKind]  += elements * loopMultiplier;
    }
}
```

`weightedOps` 是**操作次数**（被循环放大后），`opElements` 是**操作的元素数**（操作次数 × tensor 元素数 × 循环放大）。后者用于后续的 throughput 计算。

### 5.4 特殊统计

**load/store bytes 和 warp instructions**：
```cpp
if (name == "tt.load" || name == "tt.store") {
    bool load = name == "tt.load";
    auto [dataType, dataElements] = dataTypeAndElements(load);
    int64_t bitWidth = dataType ? getTypeBitWidth(dataType) : 32;
    double bytes = static_cast<double>(dataElements) * loopMultiplier * bitWidth / 8.0;
    int64_t warpInstructions = ceil(dataElements / 32.0) * loopMultiplier;

    if (load) {
        features.loadBytes += bytes;
        features.loadWarpInstructions += warpInstructions;
        // 同样更新 anchor 版本
    } else {
        features.storeBytes += bytes;
        features.storeWarpInstructions += warpInstructions;
    }
}
```

注意 `warpInstructions = ceil(dataElements / 32) * loopMultiplier`——假设每个 warp 32 个线程，每个线程处理一个元素，所以需要 `ceil(elements/32)` 条 warp 指令。这是一个简化假设（实际可能有 coalescing），但在缺少详细调度信息的情况下是可用的 first-order approximation。

**dot FLOPs**：
```cpp
if (name == "tt.dot" && op->getNumOperands() >= 2) {
    auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    int64_t m = lhs.getShape()[lhs.getRank() - 2];
    int64_t k = lhs.getShape()[lhs.getRank() - 1];
    int64_t n = rhs.getShape()[rhs.getRank() - 1];
    features.dotFlops += 2 * m * n * k * loopMultiplier;
    features.dotMNK.push_back({m, n, k});
    if (inAnchor)
        features.simtAnchors.dotFlops += 2 * m * n * k * loopMultiplier;
}
```

每个 dot 是 2×M×N×K FLOP（一次乘加 = 2 FLOP）。

**pointer statistics**（间接访存密度）：
```cpp
if (isPointerOperation) {  // tt.addptr, tt.load, tt.store
    // 按 unique shape (rank + dims) 分组
    set<string> uniqueShapes;
    for (type : op types) {
        string key = rank + "x" + dim0 + "x" + dim1 + ...;
        uniqueShapes.insert(key);
    }

    // 每个 unique pointer shape 计数一次
    features.pointerTensorOps += uniqueShapes.size();

    // 最大 rank 和 unstructured dims
    for (shape : uniqueShapes) {
        int rank = extractRank(shape);
        features.pointerUnstructuredDims += (rank > 1 ? rank : 0);
    }

    // lane-dependent 判断：只要有 rank > 1 的指针操作
    if (maxPointerRank > 1) {
        features.laneDependentPointerOps++;
    }
}
```

这个统计不是精确的"哪些访问是 indirect 的"——而是用 **rank-based proxy**：rank > 1 的指针操作被认为是"非常规的"，rank 越大越可能是非结构化的。profile 中注释明确写了："this is a rank-based proxy, not an inspection of actual address strides or lane dependence"。

**mask statistics**：
```cpp
if (!maskRanks.empty()) {
    features.maskTensorOps++;
    for (int64_t rank : maskRanks)
        features.maskRankSum += rank;

    // unique mask: 不同的 SSA value → 每个都是独立的 mask 表达式
    if (uniqueMasks.insert(value).second) {
        features.uniqueMaskValues++;
        features.uniqueMaskRankSum += rank;
        features.predicateElements += numElements;
    }
}
```

`maskRankSum` 被用于 SIMT predicate 周期的估算——每个 mask rank 对应一条 warp predicate 指令。

### 5.5 最终的后处理

遍历结束后，计算一些派生特征：

```cpp
features.scalarOps = addOps + subOps + mulOps + divOps + maxOps
                   + absOps + expOps + logOps + cmpOps + selectOps
                   + castOps + clampOps;

features.hasDot = dotOps > 0;
features.hasGather = gatherOps > 0;
features.hasAtomic = atomicOps > 0;
features.hasHistogram = histogramOps > 0;
features.hasScan = scanOps > 0;

// rank1_indirect_vector_reduce 的检测条件:
features.rank1IndirectVectorReduce =
    maxTensorRank == 1 &&          // 只有 rank-1 tensors
    reduceOps > 0 &&               // 有 reduction 操作
    vectorReduceToScalarOps > 0 && // reduce 返回标量（不是 tensor）
    vectorPtrSplatOps > 0 &&       // 有 vector pointer splat
    scalarLoadOps >= 2;            // 至少 2 个标量 load
```

`rank1IndirectVectorReduce` 是一个非常具体的模式：rank-1 tensor 上的 reduce，结果退化为标量，且使用 vector pointer + scalar load 的组合来间接访问。这是 SIMD 特别不擅长的模式，profile 给它分配了 0.75 的 penalty ratio。

### 5.6 Anchor 特征的额外统计

```cpp
// captured/escaping tensors
for (Operation *anchor : anchorRoots) {
    for (Value operand : anchor->getOperands()) {
        if (capturedTensors.insert(operand).second) {
            features.simtAnchors.capturedTensorCount++;
            features.simtAnchors.capturedTensorBytes += getStaticTensorBytes(type);
        }
    }
    for (Value result : anchor->getResults()) {
        if (!result.use_empty() && escapingTensors.insert(result).second) {
            features.simtAnchors.escapingTensorCount++;
            features.simtAnchors.escapingTensorBytes += getStaticTensorBytes(type);
        }
    }
}
```

`capturedTensors`：anchor 从外部"捕获"的输入 tensor（即 SIMD→SIMT 方向的数据传输）。
`escapingTensors`：anchor 的结果被外部使用的（即 SIMT→SIMD 方向的数据传输）。

这两个值报告在 JSON 中但目前**不参与评分**——是未来 directional transition cost 的占位符。

---

## 六、Phase 2：评分模型（Analytical Scoring）

文件：`SimdSimtCostModel.cpp:estimateSimdSimtCandidates()`，约 610 行

### 6.1 Profile 加载与 Coverage 检查

```cpp
auto profileOrError = loadCandidateProfile(options.profilePath);
CandidateProfile profile = std::move(*profileOrError);

// 1. 加载 selection profile JSON
// 2. 如果 profile 引用了 microbenchmark_profile key → 加载 microbenchmark
// 3. 验证 target 兼容性（wildcard match）
// 4. 解析 op profiles（throughput_measurement 或 relative_to + factor）
// 5. 解析 structural penalties, coverage bounds, event calibration domains
// 6. 计算 profile content SHA256（canonical JSON → hash）
```

**Coverage 检查**（在评分之前执行，可以廉价地排除不适用场景）：

```cpp
auto [covered, domain] = rankingCalibrationCoverage(features, ...);

// coverage 的判断顺序（第一个匹配的 rule 生效）：
// 1. 有 triangular_solve 且无 unknown trip count 且 shape 合适
//    → "triangular_solve_loop"
// 2. 有 triangular_solve 且有 unknown trip count 但仍在 shape/anchor 限制内
//    → "triangular_solve_loop" (例外)
// 3. hasUnknownTripCount → 直接拒绝 "unknown_loop_trip_count"
// 4. dotFlops > 0, ≤16384, 无静态循环, max numel ≤256, irregular_density ≥0.25
//    → "tiny_irregular_dot"
// 5. dotFlops == 0, rank1IndirectVectorReduce, 参数在限制内
//    → "rank1_indirect_vector_reduction"
// 6. dotFlops == 0, 有静态循环, mask/red/numel 在限制内
//    → "masked_rowwise_reduction"
// 7. 都不匹配 → "out_of_calibration_domain"
```

如果 coverage 检查不通过且 `scoreOutsideCalibrationCoverage == false`（auto 模式），直接返回 `selection_score_invalid` gate failure。

### 6.2 资源成本公式

对 profile 中每个 op 类型，分别算 SIMD 和 SIMT cycles：

```cpp
for (auto [opName, elements] : getProfileOpElements(features)) {
    const OpProfile &simd = profile.simdOps[opName];
    const OpProfile &simt = profile.simtOps[opName];

    // SIMD: 操作被向量化
    // vectorWidth = simdVectorWidthBits(2048) / elementBits(32) = 64 元素/指令
    double simdCycles = ceil(elements / vectorWidth) / simd.throughput * simd.factor;

    // SIMT: 每个元素一个标量操作
    double simtCycles = elements / simt.throughput * simt.factor;

    simdComputeCycles += simdCycles;
    simtComputeCycles += simtCycles;
}
```

注意关键差异：`ceil(elements / vectorWidth)` vs `elements`。SIMD 一次操作可以处理 vectorWidth 个元素（2048 bits / 32 = 64 FP32 元素），而 SIMT 是标量操作，每个线程每次处理 1 个元素。

**Memory**：

```cpp
// SIMD: DMA 通道，load 和 store 有独立带宽
simdLoadCycles  = loadBytes  / simdMte2BytesPerCycle;  // MTE2 = load DMA
simdStoreCycles = storeBytes / simdMte3BytesPerCycle;  // MTE3 = store DMA
simdMemoryCycles = max(simdLoadCycles, simdStoreCycles);  // DMA 可并行

// SIMT: warp-coalesced global memory
simtLoadCycles  = loadWarpInstructions  / simtLoadWarpRate;
simtStoreCycles = storeWarpInstructions / simtStoreWarpRate;
simtMemoryCycles = simtLoadCycles + simtStoreCycles;  // warp 不能同时 load 和 store
```

**Dot（MatMul）**：

```cpp
if (dotFlops) {
    simdDotCycles = simdDotSetup + dotFlops / simdDotFlopsPerCycle;
    simtDotCycles = simtDotSetup + dotFlops / simtDotFlopsPerCycle;
}
```

**SIMT 特有开销**：

```cpp
// Shuffle: warp 内 reduction 的 shuffle 指令数
// 每个 reduction 需要 ceil(maxNumel/warpSize) * log2(warpSize) 次 shuffle
int shuffleLevels = ceil(log2(warpSize));  // = 5 (warpSize=32)
simtShuffleInstructions = weightedReductions * ceil(maxNumel / warpSize) * shuffleLevels;
simtShuffleCycles = simtShuffleInstructions / simtShuffleRate;

// Predicate: mask 材料化
simtPredicateInstructions = maskRankSum * ceil(maxNumel / warpSize);
simtPredicateCycles = simtPredicateInstructions / simtPredicateRate;
```

### 6.3 合成为 Analytical Cycles

```cpp
// SIMD roofline: compute+dot 与 memory 取 max（DMA 与计算可并行）
simdIssuePayload = max(simdCompute + simdDot, simdMemory);

// SIMT roofline: compute+dot+shuffle 与 memory 取 max, 再加 predicate
simtIssuePayload = max(simtCompute + simtShuffle + simtDot, simtMemory)
                 + simtPredicate;

// program_issue_scale: profile 中的乘数 = 8.0
A_SIMD = simdSetup + simdIssuePayload * programIssueScale;
A_SIMT = simtSetup + simtIssuePayload * programIssueScale;
```

`program_issue_scale = 8.0` 是一个重要的校准参数——它把 issue payload 从"理论发放周期"放大到"实际执行周期"。这个值来自 profile 的 `selection_calibration` 而不是 microbenchmark，说明它是模型特定的拟合参数，不是硬件测量。

### 6.4 Structural Penalty（SIMD-only 惩罚）

这不是 SIMT 的 penalty——恰恰相反，它是 **SIMD 在不规则代码上的额外开销**：

```cpp
double structuralPenaltyRatio = 0;

// 不规则访存惩罚
irregularDensity = laneDependentPointerOps / pointerTensorOps;
structuralPenaltyRatio += min(irregularCap, irregularDensity * irregularPerDensity);

// Mask 材料化惩罚
structuralPenaltyRatio += min(maskCap, maskRankSum * perMaskRank);

// Reduction lowering 惩罚
structuralPenaltyRatio += min(reductionCap, weightedReductions * perWeightedReduction);

// 静态循环控制惩罚
structuralPenaltyRatio += min(loopCap, staticLoopTripCountSum * perStaticLoopTrip);

// Control flow 惩罚
if (hasControlFlow)
    structuralPenaltyRatio += controlFlowPenalty;

// 小 dot 惩罚（tiny dot underfill）
if (tinyDot) {
    tinyDotUnderfill = 1 - dotFlops/tinyDotFlopsMax;
    structuralPenaltyRatio += tinyDotPenalty * tinyDotUnderfill;
}

// Rank-1 indirect vector reduction 惩罚
if (rank1IndirectVectorReduce)
    structuralPenaltyRatio += rank1IndirectVectorReduction;
```

**设计理念**：这些开销是针对 SIMD 的——SIMD 在 irregular addressing、mask materialization、reduction lowering、loop control 等方面需要额外的指令或模式切换。SIMT 天然处理这些，所以 `A_SIMT` 不加 penalty。

```cpp
// 候选成本 = analytical + structural
simdStructuralPenaltyCycles = A_SIMD * structuralPenaltyRatio;
allSimdCost = A_SIMD + simdStructuralPenaltyCycles;  // = A_SIMD * (1 + Pstruct)
allSimtCost = A_SIMT;  // 无 penalty

// uncalibratedCandidateCosts = 以上计算结果
// calibrated costs 在后面被 Event calibration 乘数校正
```

### 6.5 Mixed 候选评分（Resource Partition）

**这是 v10 最核心的公式变化。**

旧版用 convex blend：`mixed = (1-f)*simt + f*simd + transition`。问题：凸组合永远不会同时低于两个端点（除非 transition 为负）。

新版用 **resource partition**：按 anchor plan 将操作精确分为两组——SIMT anchor 内的操作按 SIMT 费率，其余按 SIMD 费率。两个阶段**顺序执行**，所以 cost 相加（不是取 max）。

```cpp
// 步骤 1: 从全 kernel 成本中减去 anchor 的 SIMD 成本, 加上 anchor 的 SIMT 成本
mixedSimdRegularCompute = simdCompute - simdCostOfAnchorOps;
mixedSimtAnchorCompute  = simtCostOfAnchorOps;

mixedSimdRegularMemory  = max(regularLoadBytes/simdMte2Rate,
                              regularStoreBytes/simdMte3Rate);
mixedSimtAnchorMemory   = anchorLoadWarpInsts/simtLoadRate
                        + anchorStoreWarpInsts/simtStoreRate;

mixedSimdRegularDot     = simdDotSetup + regularDotFlops/simdDotRate;
mixedSimtAnchorDot      = simtDotSetup + anchorDotFlops/simtDotRate;

mixedSimtAnchorShuffle    = anchorShuffleInsts/simtShuffleRate;
mixedSimtAnchorPredicate  = anchorPredicateInsts/simtPredicateRate;

// 步骤 2: 分别计算 roofline
mixedSimdRegularPayload = max(mixedSimdRegularCompute + mixedSimdRegularDot,
                               mixedSimdRegularMemory);

mixedSimtAnchorPayload  = max(mixedSimtAnchorCompute + mixedSimtAnchorDot
                               + mixedSimtAnchorShuffle,
                               mixedSimtAnchorMemory)
                        + mixedSimtAnchorPredicate;

// 步骤 3: 剩余 structural penalty（只用 non-anchor 特征计算）
remainingStructuralPenalty = sameFormula(remainingPointer, remainingMask,
                                          remainingReductions, ...);

// 步骤 4: 最终 mixed cost
if (anchorCount > 0) {
    mixedCost = setupFallback
              + programIssueScale * (
                    mixedSimdRegularPayload * (1 + remainingStructuralPenalty)
                  + mixedSimtAnchorPayload
                )
              + boundaryCycles;  // 目前为 0
} else {
    // 没有 anchor 时: mixed 必然比两个端点都差 → 自动被淘汰
    mixedCost = max(allSimdCost, allSimtCost) + setupFallback;
}
```

**为什么 resource partition 能让 mixed 胜出**：

在 "大部分是 SIMD 友好的 + 小部分是 SIMT 友好的" 场景下：
- 如果用 all-SIMD：小部分的 SIMD cost 很高（irregular addressing 展开为标量循环）
- 如果用 all-SIMT：大部分的 SIMT cost 比 SIMD 高（向量化效率更高）
- 如果用 mixed：大部分走 SIMD，小部分走 SIMT，setup cost 小 → 总 cost 可能比两者都低

### 6.6 Setup Fallback

Mixed 模式的 setup cost 来自 profile 中按 warp count 选择的 fallback：

```cpp
// 选择最近的 warp count fallback
for (fallback : profile.mixedSetupFallbacks) {
    if (abs(fallback.numWarps - numWarps) < nearest distance)
        nearest = &fallback;
}

report.breakdown.standaloneSimtSetupCycles = profile.simtSetupCycles;  // 141.0
report.breakdown.mixedSetupFallbackCycles  = nearest->emptySimtSetupCycles;
report.breakdown.setupProxyDeltaCycles     = mixedFallback - standaloneSetup;
```

32 warps: 223 cycles。1-16 warps: 182 cycles。

profile 中明确标注：这些值来自 standalone empty-VF probe（mode1 minus barrier-only mode6），**不是真正的 directional SIMD→SIMT 转换延迟**。真正的 transition cost 仍未测量。

### 6.7 Event Route Calibration

**这是 v10 的第二个核心创新**。

分析公式给出的是 feature-sensitive 的 raw score，但与真实 NPU Event 测量之间存在 route-relative residual。Event calibration 用 domain-specific 乘数来校正：

```cpp
if (eventRouteCalibration) {
    calibratedAllSimdCost = allSimdCost * domain_all_simd_multiplier;
    calibratedAllSimtCost = allSimtCost * domain_all_simt_multiplier;
    calibratedMixedCost   = mixedCost  * domain_mixed_multiplier;
}
```

三个 calibration domain 的乘数来自 A5 card 0 上的真实测量：

**masked_rowwise_reduction**（FBGEMM, 4 warps）：
```
all_simd_multiplier:      67.628    ← SIMD 被严重低估
all_simt_only_multiplier:  0.808
mixed_simd_simt_multiplier: 3.537
```

**tiny_irregular_dot**（gather-dot-min, 4 warps）：
```
all_simd_multiplier:       1.321
all_simt_only_multiplier:  1.000  (未验证)
mixed_simd_simt_multiplier: 0.666  ← mixed 被高估
```

**triangular_solve_loop**（solve-tril BT16, 4 warps）：
```
all_simd_multiplier:       290.881  ← 巨大校正
all_simt_only_multiplier:  1.714
mixed_simd_simt_multiplier: 2.893
```

注意 `masked_rowwise_reduction` 的 all-SIMD 乘数是 **67.6**。这说明分析公式严重低估了 FBGEMM 类 workload 的 SIMD 时间（可能是因为 pipeline 重叠效果在纯分析公式中无法体现）。

**为什么需要 Event calibration**：分析公式只能捕捉 first-order effects（吞吐、带宽），无法捕捉编译器后端的 second-order effects（指令调度、寄存器分配、memory coalescing、pipeline 重叠）。Event calibration 通过在三个 bounded domain 上测量真实时间，来校正这些 residual。乘数被保留在 report 中，让校正关系对用户透明。

### 6.8 决策与 Gate 系统

```cpp
// 1. 排序 legal candidates
auto candidates = legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal);
stable_sort(candidates);  // 低分优先

report.decision = candidates[0];  // 最优
report.runnerUp  = candidates.size() > 1 ? candidates[1] : candidates[0];

// 2. 计算 advantage（相对于 all-SIMD baseline）
if (decision == AllSIMD)
    decisionAdvantage = runnerUpScore - bestScore;  // 自己赢自己
else
    decisionAdvantage = allSimdScore - bestScore;   // vs baseline

// 3. 计算 required gain（绝对 floor 64 cycles + margin%）
gainBaseline = allSimdLegal ? allSimdScore : runnerUpScore;
requiredGain = max(64.0, gainBaseline * options.marginRatio);  // default 10%

// 4. Gate 检查
if (!targetCompatible)        gateReasons += "target_incompatible";
if (!selectionScoreValid)      gateReasons += "selection_score_invalid";
if (!unsupported.empty())      gateReasons += "unsupported_cost_terms";
if (rankingConfidence < minimum) gateReasons += "ranking_confidence_...";
if (decision != AllSIMD && !(advantage > requiredGain))
    gateReasons += "decision_advantage_not_above_required_gain";

report.gatePassed = gateReasons.empty();
```

Gate 设计的关键洞见：当 gain margin 不足时，**不是退回 backend_default**（可能意外走 legacy force path），而是**明确设置为 all_simd**。代码中的 safe baseline 逻辑：

```cpp
// SelectSimdSimtCostModel.cpp
else if (autoMode && onlyInsufficientGain && report.allSimdCandidateLegal) {
    effective = kAllSimd;
    selectionSource = "cpp_cost_model_safe_baseline";
    applicationReason = "decision_gain_below_margin_keep_all_simd";
}
```

---

## 七、Materialize：混合模式的 IR 改造

文件：`MaterializeSimtScopes.cpp:materializeSimtAnchorPlan()`，约 290 行

当 effective decision 是 `mixed_simd_simt` 时，需要告诉下游编译器哪些 op 走 SIMT。方法：对 anchor plan 中的每个 anchor op，包裹进 `scope.scope{vec_mode="simt"}` region。

### 7.1 单 Op Anchor（wrapAnchorOperation）

用于 DirectGather、LoadedIndexDependentMemory、Histogram、PlainOneDimensionalCumsum、TensorAtomic：

```mlir
// 原始:
%0 = tt.gather %ptr, %indices : tensor<...>

// Materialize 后:
%0 = "scope.scope"() <{vec_mode = "simt"}> ({
    %inner = tt.gather %ptr, %indices : tensor<...>
    "scope.return"(%inner) : (tensor<...>) -> ()
}) : () -> tensor<...>
```

C++ 实现：

```cpp
static LogicalResult wrapAnchorOperation(Operation *op) {
    OpBuilder builder(op);

    // 1. 创建 scope.scope op，结果类型与原 op 相同
    OperationState scopeState(op->getLoc(), "scope.scope");
    scopeState.addTypes(op->getResultTypes());
    scopeState.addAttribute("vec_mode", builder.getStringAttr("simt"));
    scopeState.addRegion();
    Operation *scopeOp = builder.create(scopeState);

    // 2. 在 scope region 内创建 body block
    Region &scopeRegion = scopeOp->getRegion(0);
    auto *scopeBody = new Block();
    scopeRegion.push_back(scopeBody);

    // 3. 把原 op 移入 scope body
    SmallVector<Value> originalResults(op->getResults());
    op->moveBefore(scopeBody, scopeBody->end());

    // 4. 创建 scope.return，将原 op 的所有结果 thread 出去
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
    OperationState returnState(op->getLoc(), "scope.return");
    returnState.addOperands(originalResults);
    Operation *returnOp = bodyBuilder.create(returnState);

    // 5. 用 scope op 的结果替换原 op 的所有外部使用
    for (auto [original, replacement] : zip_equal(originalResults, scopeOp->getResults())) {
        original.replaceAllUsesExcept(replacement, returnOp);
    }
    return success();
}
```

关键点：scope region 不是 isolated from above——操作数可以从外部 SSA 值自由捕获。只有 **results** 需要通过 `scope.return` 传出。

### 7.2 范围 Anchor（wrapAnchorRange）

用于 TriangularSolveLoop——包装的不是单个 op，而是一段连续的 op：

```cpp
static SmallVector<Operation*> collectTriangularSolveRange(Operation *anchor) {
    Block *block = anchor->getBlock();

    // 1. 找到第一个和最后一个 TriangularSolveLoop 循环
    Operation *firstLoop = nullptr, *lastLoop = nullptr;
    for (Operation &nested : *block) {
        auto kind = classifyMixedSimtAnchor(&nested);
        if (kind == TriangularSolveLoop) {
            if (!firstLoop) firstLoop = &nested;
            lastLoop = &nested;
        }
    }

    // 2. 起点: 第一个循环之前的最后一个 tt.load 之后
    Operation *lastInputLoad = nullptr;
    Operation *cursor = firstLoop->getPrevNode();
    while (cursor && cursor->getName() != "tt.load")
        cursor = cursor->getPrevNode();
    Operation *start = lastInputLoad ? lastInputLoad->getNextNode() : firstLoop;

    // 3. 终点: 最后一个循环之后，直到 arith.uitofp/arith.addf/arith.select 链结束
    Operation *end = lastLoop;
    cursor = lastLoop->getNextNode();
    while (cursor) {
        llvm::StringRef name = cursor->getName().getStringRef();
        if (name != "arith.uitofp" && name != "arith.addf" && name != "arith.select")
            break;
        end = cursor;
        cursor = cursor->getNextNode();
    }

    // 4. 返回 [start, end] 范围内的所有 op
    SmallVector<Operation*> result;
    for (Operation *op = start; op; op = op->getNextNode()) {
        result.push_back(op);
        if (op == end) break;
    }
    return result;
}
```

`wrapAnchorRange` 与 `wrapAnchorOperation` 类似，但只 thread **escaping values**（在范围外有使用者的 results）：

```cpp
// 只有 escaping values 通过 scope.return 传出
DenseSet<Value> escaping;
for (Operation *op : ops)
    for (Value result : op->getResults())
        for (OpOperand &use : result.getUses())
            if (!isInsideRange(use.getOwner()))
                escaping.insert(result);

// scope op 的结果类型只包含 escaping types
SmallVector<Type> escapingTypes;
for (Value value : escaping)
    escapingTypes.push_back(value.getType());
```

### 7.3 MaterializeSimtScopesPass（兼容性验证）

SelectSimdSimtCostModel pass 之后运行，验证 mixed 决策确实带有 scope contract：

```cpp
void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!isMixedModelDecision(module))
        return;                          // 非 mixed → 跳过
    if (containsLocalSimtScope(module))
        return;                          // 已有 scope.scope{simt} → 通过
    module.emitError("mixed_simd_simt requires materialized scope.scope<simt>");
    signalPassFailure();
}
```

---

## 八、Python 侧集成

文件：`third_party/ascend/backend/compiler.py`

### 8.1 入口函数

```python
def _run_cpp_simd_simt_costmodel(mod, metadata, opt) -> str:
    """双重门控: compile_mode="simd_simt" 且 auto_simt_scope_mode != "off" 才触发"""

    mode = opt.auto_simt_scope_mode
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":
        return "backend_default"  # ← 直接返回，不跑 costmodel

    # 1. 加载 profile 路径
    profile = opt.auto_simt_model_profile or str(
        _costmodel_profiles_dir() / "simd_simt" / "david_v100_simd_simt_v1.json"
    )

    # 2. 运行两个 C++ pass
    pm = ir.pass_manager(mod.context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(
        pm, mode, profile, str(opt.arch),
        int(opt.num_warps), float(opt.auto_simt_scope_margin),
        bool(opt.compile_on_910_95), str(opt.auto_simt_scope_dump),
    )
    ascend.passes.ttir.add_materialize_simt_scopes(pm)
    pm.run(mod)

    # 3. 读取 module attrs
    report = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.report_json")
    effective = ascend.ir.get_string_attr(mod, "ascend.simt_costmodel.effective")
    return effective
```

### 8.2 环境变量控制

```
TRITON_ASCEND_AUTO_SIMT_SCOPE=auto|report|off
TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=/path/to/report.jsonl
TRITON_ASCEND_AUTO_SIMT_SCOPE_MARGIN=0.10
TRITON_ASCEND_AUTO_SIMT_PROFILE=/path/to/custom_profile.json
TRITON_ASCEND_COMPILE_MODE=simd_simt
```

- `auto`：gate 通过则自动应用推荐决策
- `report`：永远不应用决策，但输出报告（`effective = backend_default`）
- `off`：完全跳过 costmodel

### 8.3 纯 SIMT 路径

如果 effective 是 `all_simt_only`：

```python
if cpp_all_simt or ascend.ir.is_whole_body_void_simt_scope(mod):
    # 1. RowCoalescing: 合并行索引
    ascend.passes.ttir.add_row_coalescing(pm)
    pm.run(mod)
    _export_coalesce_metadata(mod, metadata)

    # 2. Inline void SIMT scopes
    ascend.ir.inline_void_simt_scopes_for_pure_simt(mod)

    # 3. 清除 costmodel attrs
    ascend.ir.clear_simd_simt_costmodel_attrs(mod)

    # 4. 直接返回 TTIR string（跳过后续 pass）
    metadata["parallel_mode"] = "simt"
    metadata["shared_mem_dynamic_size"] = 122880
    return str(mod)
```

### 8.4 Mixed 模式编译

```python
if metadata.get("auto_simt_requested_kind") == "mixed_simd_simt":
    # 验证: parallel_mode 必须是 mix_simd_simt
    applied = metadata["parallel_mode"] == "mix_simd_simt"
    if not applied and TRITON_ASCEND_AUTO_SIMT_STRICT_VERIFY:
        raise RuntimeError("C++ costmodel selected mixed but lowering didn't apply it")

# BishengIR 编译选项 (910_95)
if metadata.get("auto_simt_requested_kind") == "mixed_simd_simt":
    _compile_option_list += ["--enable-hivm-delayed-cross-core-gss=false"]
```

`--enable-hivm-delayed-cross-core-gss=false`：因为 split-mix-kernel 后多个 sibling SIMT scope 可能反转 split-side anchor interval，使用 cross-core GSS（而非 delayed 变体）直到 BiShengIR 修复这个 invariant。

### 8.5 Asset Hash for Cache Invalidation

```python
def _auto_simt_asset_hash(path, default_name) -> str:
    asset = Path(path) if path else _costmodel_profiles_dir() / default_name
    selection_bytes = asset.read_bytes()

    digest = hashlib.sha256()
    digest.update(b"selection-profile\0")
    digest.update(selection_bytes)

    # 如果 profile 引用了 shared microbenchmark → 也 hash 它
    profile = json.loads(selection_bytes)
    shared_ref = profile.get("microbenchmark_profile") if isinstance(profile, dict) else None
    if shared_ref:
        shared_asset = asset.parent / shared_ref
        digest.update(b"\0shared-microbenchmark\0")
        digest.update(shared_asset.read_bytes())

    return digest.hexdigest()
```

组合了 selection profile 和它引用的 microbenchmark profile 的哈希，用于 JIT cache key。如果任一 profile 变了，cache 自动失效。

### 8.6 Profile 路径解析

```python
def _costmodel_profiles_dir() -> Path:
    # 1. 原生 package: .../_C/ascend/costmodel_profiles/
    # 2. 兼容旧 package: .../backend/costmodel_profiles/
    # 3. 源码树: .../third_party/ascend/costmodel/profiles/
```

---

## 九、Python 前端 vec_mode scope

文件：`language/cann/extension/scope.py`

```python
class scope:
    def __init__(self, core_mode: str = None, vec_mode: str = None):
        """
        core_mode: "cube" | "vector"
        vec_mode:  "simd" | "simt"
        """
        # core_mode="cube" 时 vec_mode 不能设置
        # 至少需要一个参数

# 使用:
with al.scope(vec_mode="simt"):
    y = tl.load(base_ptr + indirect_indices)

# 生成 TTIR: scope.scope {vec_mode = "simt"} { ... }
```

与 simt_costmodel 分支的关系：在 `compile_mode="simd_simt"` + C++ cost model 自动决策 mixed 时，materializeSimtAnchorPlan 会**自动**创建这些 scope.scope——用户不需要手写 `al.scope(vec_mode="simt")`。但如果用户手写了，costmodel 会检测到 `hasExplicitScope` 并跳过 automatic materialization（避免冲突）。

---

## 十、测试覆盖

文件：`unittest/costmodel_ut/SimdSimtCostModelTest.cpp`（424 行，8 个测试）

### 10.1 测试设计

每个测试手动构造 `SimdSimtFeatureSummary`（模拟特征提取的输出），然后调 `estimateSimdSimtCandidates()` 验证 golden scores。

**GatherDotGoldenScoresRequireMaterializableMixedPlan**（gather + dot kernel）：

```cpp
// 构造特征: 3 loads, 1 store, 1 dot(16×16×16), 9 splats, irregular pointers
// 验证:
//   - simdStructuralPenaltyCycles == A_SIMD * structuralPenaltyRatio  (公式一致性)
//   - uncalibrated.allSimd == A_SIMD * (1+Pstruct)                    (公式正确性)
//   - uncalibrated.mixed == max(allSimd, allSimt) + 223.0             (无 anchor 退化)
//   - calibrated.mixed  == uncalibrated.mixed * 0.666478              (event calibration)
//   - calibrationDomain == "tiny_irregular_dot"
//   - mixedCandidateLegal == false  (因为 compileOn91095=true 但没有 materializable anchor)
```

**FbgemmGoldenScoresRequireMaterializableMixedPlan**（FBGEMM kernel）：

```cpp
// 验证 masked_rowwise_reduction domain:
//   - calibrated.allSimd == uncalibrated.allSimd * 67.628
//   - calibrated.allSimt  == uncalibrated.allSimt * 0.808
//   - calibrated.mixed    == uncalibrated.mixed * 3.537
```

**SolveTrilBt16StaysOutsideMaskedRowwiseCalibration**：

```cpp
// BT16 特征不满足任何 coverage domain → 被拒绝
// → gateReasons = ["selection_score_invalid"]
```

**TriangularUnknownLoopUsesBoundedCalibrationException**：

```cpp
// 模拟 triangular_solve 的动态循环（unknown trip count）
// 但有 triangular_solve_loop anchor evidence → 仍然通过 coverage
// → 验证 shuffle 和 predicate 周期 > 0
```

**UnknownLoopWithoutTriangularEvidenceRemainsRejected**：

```cpp
// 同样的 unknown trip count 特征，但去掉了 triangular evidence
// → 被拒绝 "unknown_loop_trip_count"
```

**TriangularUnknownLoopStillHonorsAnchorCountAndShapeBounds**：

```cpp
// 验证 triangular_solve 例外仍然受限于:
//   - anchor count: 0 → 拒绝
//   - max tensor numel: 257 → 拒绝
```

**OutOfCoverageAutoSkipsButDiagnosticsStillScore**：

```cpp
// dotFlops=16385 (刚好超出 tiny_dot_flops_max=16384) → coverage 拒绝
// auto 模式: scoreOutsideCalibrationCoverage=false → 0.0 scores + gate failed
// diagnostic 模式: scoreOutsideCalibrationCoverage=true → 有 scores 但仍 gate failed
```

### 10.2 PassesTest 新增测试

**MaterializeSimtScopePreservesEscapingSSAResult**：验证 scope.scope 的 SSA result remapping 正确——escaping values 被 thread 到 scope.return，内部 uses 指向原有的 op result。

**NativeWholeBodySimtScopeDetectionAndInlining**：验证 `findWholeBodyVoidSimtScope` 和 `inlineVoidSimtScopesForPureSimt` ——整个 kernel body 被一个 void SIMT scope 包裹时的检测和内联。

**ModelControlledRoutingIgnoresLegacyGlobalForce**：验证 `shouldUseSimtTemplate` 在 model-controlled 模式下忽略 legacy force flags。

---

## 十一、与下游 Lowering 的桥接

文件：`lib/TritonToUnstructure/UnstructureConversionPass.cpp` 等

### 11.1 shouldUseSimtTemplate

所有 backend lowering 通过 `shouldUseSimtTemplate` 决定单个 op 是否走 SIMT：

```cpp
inline bool shouldUseSimtTemplate(Operation *op, bool legacyForceSimt) {
    // 1. 如果在 SIMD scope 内 → 永远不 SIMT
    if (hasEnclosingVectorMode(op, "simd"))
        return false;

    // 2. 如果在 SIMT scope 内 → 是 SIMT
    const bool locallySelected = hasEnclosingVectorMode(op, "simt");

    // 3. 模型控制模式: 只看 scope
    if (isModelControlled(op))
        return locallySelected;

    // 4. 传统模式: legacy force flag 或 scope
    return legacyForceSimt || locallySelected;
}
```

### 11.2 scope.scope 的存活

在 Linalg lowering 中，`scope.scope{vec_mode="simt"}` 被保留（不擦除），成为 BiShengIR 的 native region contract。这是有意设计的——scope.scope 不是临时的标记，而是一个合法的 IR 构造。

---

## 十二、RowCoalescing：纯 SIMT 优化

当 costmodel 选择 `all_simt_only` 且 kernel 是 whole-body SIMT scope 时，在 TTIR 级别做行合并优化：

```
优化前:   row = pid(axis=0)           → 每个 program 处理 1 行
优化后:   rows = pid(axis=0)*H + arange(H) → 每个 program 处理 H 行
```

保守的 bail-out 条件：
- 只折叠能被整除的维度
- 必须有明确的 `tt.make_range → tt.expand_dims → tt.addptr` 行索引模式
- 不处理复杂的分支或间接索引

Driver 端使用 `coalesce_grid_ceil_div` metadata 来决定 launcher 的 grid shrink 方式。

---

## 十三、Microbenchmark 方法论与评分公式的对应

### 13.1 测量原则

所有 microbenchmark 基于 **SYS_CNT 硬件计数器**，在 A5 (Ascend950PR/dav-c310) 上实测。关键方法论：

1. **不是 bare-metal peak**：所有测量包含 runtime-loop 开销（循环控制、地址生成等）。测量值的 scope 字段明确标注（如 `single_aiv_runtime_loop_ilp4_or_more_effective_source_vadd`）。

2. **物理单位 + 时钟域**：每个测量带 `unit` 和 `cycle_domain`。Consumer 必须声明期望的单位和时钟域——不匹配则报错。

3. **来源可追溯**：每个测量标注源文件（如 `triton_cases/SIMT_Test/tput.cce`）和 measurement kind（`isolated_microbenchmark` vs `architecture_fact` vs `device_configuration`）。

### 13.2 测量项在评分公式中的使用

| 测量 key | 值 | 在评分中的使用 |
|----------|-----|---------------|
| `simd.vector_width_bits` | 2048 bit | `vectorWidth = 2048/elementBits` |
| `simt.warp_size` | 32 lane | `warpInstructions = ceil(elements/32)` |
| `simd.f32.add.throughput` | 3.30 vec_inst/cycle | SIMD add cycles |
| `simt.f32.add.throughput` | 141.0 scalar_op/cycle | SIMT add cycles，衍生 op 的 base |
| `simt.setup.empty_with_barrier` | 141.0 cycle | `simtSetupCycles` |
| `simt.shuffle.throughput` | 0.817 warp_inst/cycle | `simtShuffleCycles` |
| `simt.gm.load.throughput` | 0.176 warp_inst/cycle | `simtLoadCycles` |
| `simt.gm.store.throughput` | 0.129 warp_inst/cycle | `simtStoreCycles` |
| `simt.setup.transition_harness_net.warps_N` | 182-223 cycle | `mixedSetupFallbackCycles` |

### 13.3 未直接使用的测量

以下测量在 profile 中存在但**当前未被 Route Model 使用**（标记为"retained as context"或用于 future work）：

- `simt.ub.load/store.throughput` — UB 存取速率（可能用于 future 更精细的 SIMT memory model）
- `simt.gm.load/store.bandwidth` — 字节带宽（当前用 warp instruction rate 估算 memory）
- `simd.f32.add.dependent_latency` — 依赖链延迟（可能用于 future pipeline depth model）
- `simt.shuffle.dependent_latency` — shuffle 延迟（当前只用 throughput）

### 13.4 模型特定参数（不在 microbenchmark 中）

以下参数属于 selection profile 的 `simd`/`simt` 部分，不是共享 microbenchmark：

| 参数 | 值 | 来源 |
|------|-----|------|
| `simdMte2BytesPerCycle` | 202.25 | selection-model seed (legacy) |
| `simdMte3BytesPerCycle` | 202.25 | selection-model seed (legacy) |
| `simdDotFlopsPerCycle` | 4096.0 | AscendModel cube seed |
| `simtDotFlopsPerCycle` | 141.0 | scalar FMA seed |
| `simtPredicateRate` | 0.038 | CaModel workload-effective rate |
| `programIssueScale` | 8.0 | Bounded dev. activation fitting |

---

## 十四、作者方法论反推

从 profile 注释、代码结构和测试数据可以反推出作者的方法论：

### 步骤 1：建立微基准测试框架

在 A5 芯片上用 SYS_CNT 计数器做了一系列隔离测量。测量设计关注：
- **workload-effective**：不是单指令延迟，而是实际 runtime-loop 中的有效吞吐（包含循环开销）
- **独立**：每个测量针对单一操作类型，最小化 interaction
- **serialized**：sequential 执行以确保没有 pipeline 重叠

### 步骤 2：建立 First-Principles 分析模型

```
成本 = setup + max(compute+dot, memory) * program_issue_scale
       + structural_penalty * analytical_cost
```

这是经典的 roofline 模型变体。`max(compute, memory)` 反映了"计算和内存访问可以并行"的假设（SIMD DMA），而 SIMT 的 shuffle 和 predicate 有独立建模。

### 步骤 3：在三个 workload 上做 Event 校准

选择三个 representative workload（FBGEMM、gather-dot-min、solve-tril），在 A5 card 0 上跑真实验证：
- 30-50 warmup iterations → 100-200 NPU Event samples
- 三路编译（all-SIMD / all-SIMT / Auto mixed）
- 记录 median event time + correctness PASS/FAIL
- 计算 `multiplier = measured_time / analytical_score`

### 步骤 4：拟合 Structural Penalty 参数

从 Event 校准的 residual 反推出 penalty 参数——调整 `irregularPerDensity`, `perMaskRank`, `perWeightedReduction` 等直到分析公式的排名与测量一致。

### 步骤 5：定义 Coverage Domain

只在校准过的 workload 范围内做决策——任何超出 domain 的 kernel 自动退回 SIMD（safe baseline）。Coverage bounds 是模型 validity limits，不是硬件 limits。

### 步骤 6：迭代 Profile Version

从 v3 到 v10 的演变过程（在 loader 中可见版本兼容性代码）：
- v3-v4：基础分析模型
- v5：引入 shared microbenchmark profile
- v6：添加 structural penalty separation
- v7：引入 anchor partition（替代 convex blend）
- v8：添加 mixed setup fallback per warp count
- v9-v10：引入 Event route calibration

每次迭代不一定改变公式结构——可能只是调整 profile 参数或修复 bug。Loader 中保留了对 v3+ 的向后兼容。

---

## 十五、当前限制与未解决问题

### 15.1 Directional Transition Cost 未测量

Mixed setup fallback 来自 standalone empty-VF probe（无 SIMD 阶段），不是真正的 SIMD→SIMT 或 SIMT→SIMD 转换延迟。真正的 `directional_transition_system_cycles` 在 report 中显示为 `null`。

### 15.2 Coverage Domain 有限

只有三个 calibration domain。大多数 kernel 会落在 `out_of_calibration_domain` → 不采纳任何非 SIMD 决策。

### 15.3 SIMT Memory Rate 包含开销

GM load/store throughput 测量包含地址生成、值构造和循环控制开销——不是 intrinsic LSU issue peak。这意味着对于有独立地址计算硬件的架构，SIMT memory cost 可能被高估。

### 15.4 Mixed Boundary Cost 为零

`mixedBoundaryCycles = 0.0`。这假设 SIMD→SIMT 和 SIMT→SIMD 的"交接"开销可以忽略——目前没有测量数据支持或反驳这个假设。

### 15.5 DES Feedback 未激活

`david_v100_des_feedback_v1.json` 是占位文件，1672/1727 fallback ops 阻止了规则生效。DES（Design Space Exploration）feedback 机制被设计为"用实际编译结果反哺 profile 参数"，但目前未启用。

### 15.6 irregularDensity 是 Rank-Based Proxy

没有检查实际的地址 stride——只用 rank > 1 的指针操作数除以总指针操作数。这可能在 rank 高但实际是 regular strided access 的场景下产生误判。

---

> **文档结束**。配合 `autoscope_simit_costmodel_analysis.md`（旧版 v6 实现）和 `autoscope_simit_costmodel_v6_vs_v10.md`（版本对比）阅读可获得完整图景。
