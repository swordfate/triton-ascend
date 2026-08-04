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
    """双重门控：compile_mode="simd_simt" 且 auto_simt_scope_mode != "off" 才触发"""

    # 门控：两条件任一不满足 → 直接返回，不跑 costmodel
    mode = opt.auto_simt_scope_mode
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":
        return "backend_default"

    # auto_simt_scope_mode 可通过环境变量 TRITON_ASCEND_AUTO_SIMT_SCOPE=auto/off 控制
    # compile_mode="simd_simt" 是新增的第四个模式，专用于触发 costmodel
    # 已有的三个模式（simd/unstructured_in_simt/simt_only）走 legacy 路径，不经过 costmodel

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

## 八、关键函数逐行解读

### 8.1 `analyzeSimdSimtFeatures()` — 特征提取（~280 行）

这个函数的唯一任务：**遍历 TTIR ModuleOp 中的每一个 operation，统计出 `SimdSimtFeatureSummary`**。

#### 8.1.1 函数签名

```cpp
llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module);
```

- `ModuleOp module`：输入，整个 TTIR 的根节点。MLIR 中一切都是嵌套的 Operation/Region/Block，`module` 是顶层容器
- `llvm::Expected<T>`：返回值。要么是一个有效的 `T`，要么是一个错误信息（类似 Rust 的 `Result<T, E>`）。如果传入的 module 是空的（`!module` 为 true），直接返回错误字符串，不会崩
- `SimdSimtFeatureSummary`：输出，一个装满统计数据的结构体

#### 8.1.2 初始化阶段（lines 1418-1431）

```cpp
SimdSimtFeatureSummary features;        // 创建空的特征结构体，所有字段默认 = 0/false
initializeWorkMaps(features);           // 给 features.weightedOps 和 features.opElements
                                        //  两个 StringMap 预填充 16 个 key（"load","store",
                                        //  "reduce","scan","gather","histogram","atomic",
                                        //  "add","sub","mul","div","max","abs","cmp",
                                        //  "select","cast","clamp"），每个 value = 0
                                        // 这是为了后面用 features.weightedOps["add"] += ...
                                        // 时不需要先判断 key 是否存在
```

接着是一个 C++ lambda 函数 `updateTypeStats`（行 1422-1431）：

```cpp
auto updateTypeStats = [&](Type type) {
    // maxElementBits: 记录 kernel 中最大元素位宽（如 f32 → 32, f16 → 16）
    // 后面评分时用这个值来推算 SIMD 向量宽度能容纳多少个元素
    features.maxElementBits =
        std::max(features.maxElementBits, getTypeBitWidth(type));

    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        // maxTensorRank: 最大的张量维度数
        features.maxTensorRank =
            std::max<int64_t>(features.maxTensorRank, tensor.getRank());
        // maxTensorNumel: 最大的张量元素总数（所有维度乘起来）
        features.maxTensorNumel =
            std::max(features.maxTensorNumel, getStaticNumElements(type));
    }
};
```

`[&]` 表示这个 lambda **按引用**捕获外部作用域的所有变量（`features` 等），在里面修改会影响外面。`dyn_cast<RankedTensorType>` 是 MLIR 的类型转换——如果是张量类型就转换成功返回非空指针，否则返回空指针。`if (auto tensor = ...)` 利用了 C++17 的 if-with-initializer 语法：括号里声明变量，非空时进入 if 体。

#### 8.1.3 主遍历循环：`module.walk()` (lines 1433-1660)

```cpp
module.walk([&](Operation *op) {
```

- `module.walk(lambda)` 是 MLIR 提供的递归遍历方法。它会**递归地**访问 ModuleOp 内部的每一个 Operation（包括嵌套在 Region/Block/scf.for/scf.if 深处的），对每个 op 调用一次 lambda
- 这意味着 lambda 里面的代码对 kernel 中**每一个** `tt.load`、`arith.addf`、`scf.for`、`tt.dot` 等都会执行一次

lambda 内部第一步（lines 1434-1436）：获取基本信息

```cpp
llvm::StringRef name = op->getName().getStringRef();   // op 的名字，如 "tt.load"
const int64_t elements = getOperationElements(op);      // 这个 op 操作的元素数
const int64_t loopMultiplier = getLoopMultiplier(op);   // 循环乘数，见 8.1.4
```

**`getLoopMultiplier`**（lines 360-372）是本节最重要的辅助函数：

```cpp
static int64_t getLoopMultiplier(Operation *op) {
    int64_t multiplier = 1;
    // 从当前 op 出发，沿 MLIR 树向上走，寻找所有包裹它的 scf.for 循环
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
        if (parent->getName().getStringRef() != "scf.for")
            continue;  // 不是 for 循环，跳过
        int64_t tripCount = getStaticLoopTripCount(parent);
        // 如果能静态解析出 trip count（如 for i=0 to 1024 step 1），乘上去
        multiplier *= tripCount;
    }
    return multiplier;
}
```

举例：如果一个 `arith.addf` 在 `for i in range(1024)` 里，又在 `for j in range(256)` 里，`getLoopMultiplier` 返回 `1024 × 256 = 262144`。这意味着这个 add 操作实际上被执行了 26 万次。后面用它给 op 的贡献值加权。

**`getStaticLoopTripCount`**（lines 339-358）尝试从 MLIR 的 `scf.for %i = %lb to %ub step %step` 中静态解析出迭代次数。它去读取 lower bound、upper bound、step 三个操作数——如果都能追溯到 `arith.constant`，就能算出来；否则返回 1（保守假设）。

#### 8.1.4 遍历内部：类型统计（lines 1438-1445）

```cpp
for (Type type : op->getOperandTypes())  // 遍历所有输入的类型
    updateTypeStats(type);
for (Type type : op->getResultTypes())   // 遍历所有输出的类型
    updateTypeStats(type);
for (Region &region : op->getRegions())  // 遍历嵌套的 Region
    for (Block &block : region)          // 遍历 Region 里的 Block
        for (BlockArgument argument : block.getArguments())  // Block 的输入参数
            updateTypeStats(argument.getType());
```

这段代码确保 kernel 中出现的所有类型都被记录到 `features.maxElementBits/maxTensorRank/maxTensorNumel` 中，包括深层嵌套的 block argument（如 scf.for 的循环变量）。

#### 8.1.5 遍历内部：Op 分类与原始计数（lines 1447-1508）

接下来是一大段 if-else 链，对每个 op 进行分类：

```cpp
if (name.starts_with("arith."))  ++features.arithOps;     // arith 族 op
if (name.starts_with("math."))   ++features.mathOps;      // math 族 op
if (name.starts_with("scf.") || name.starts_with("cf."))
    features.hasControlFlow = true;                        // 检测到控制流

// Triton 原生 op
if      (name == "tt.load")        ++features.loadOps;
else if (name == "tt.store")       ++features.storeOps;
else if (name == "tt.reduce")      ++features.reduceOps;
else if (name == "tt.scan")        ++features.scanOps;
else if (name == "tt.gather")      ++features.gatherOps;
else if (name == "tt.dot")         ++features.dotOps;
else if (name.starts_with("tt.atomic"))  ++features.atomicOps;
else if (name == "tt.histogram")   ++features.histogramOps;
else if (name == "tt.broadcast")   ++features.broadcastOps;
// ... 等等

// 细分 arith/math
if      (name == "arith.addf" || name == "arith.addi")   ++features.addOps;
else if (name == "arith.subf" || name == "arith.subi")   ++features.subOps;
else if (name == "arith.mulf" || name == "arith.muli")   ++features.mulOps;
// ... 等等
```

注意：这些是**原始计数**（没乘 loopMultiplier）。带循环加权的计数在后面一步做。

#### 8.1.6 遍历内部：加权计数（lines 1510-1514）

```cpp
llvm::StringRef weightedKind = classifyWeightedOp(name);
if (!weightedKind.empty()) {
    features.weightedOps[weightedKind] += loopMultiplier;
    features.opElements[weightedKind]  += elements * loopMultiplier;
}
```

`classifyWeightedOp` 把细分的 op name 归并到粗粒度的类别：

| 原始 name | weightedKind |
|-----------|-------------|
| `arith.addf`, `arith.addi` | `"add"` |
| `tt.load` | `"load"` |
| `tt.store` | `"store"` |
| `tt.dot` | `"dot"` |
| `tt.gather` | `"gather"` |

然后：
- `features.weightedOps["add"] += loopMultiplier` — 加权 op 次数
- `features.opElements["add"] += elements * loopMultiplier` — 加权元素数

举例：`arith.addf` 在一个 `for i in range(1024)` 里，操作 `tensor<256xf32>`：
- `weightedOps["add"] += 1024`（这个 add 实际上被调用了 1024 次）
- `opElements["add"] += 256 * 1024 = 262144`（总共处理了 26 万个元素）

#### 8.1.7 遍历内部：循环统计（lines 1516-1522）

```cpp
if (name == "scf.for") {
    int64_t tripCount = getStaticLoopTripCount(op);
    ++features.staticLoopCount;          // 循环个数
    features.staticLoopTripCountSum += tripCount;  // 所有循环 trip count 之和
    features.staticLoopTripCountMax =
        std::max(features.staticLoopTripCountMax, tripCount);  // 最大的 trip count
}
```

#### 8.1.8 遍历内部：访存字节数和 warp 指令数（lines 1536-1552）

```cpp
if (name == "tt.load" || name == "tt.store") {
    bool load = name == "tt.load";  // true=load, false=store

    // 获取数据类型和数据元素数
    auto [dataType, dataElements] = dataTypeAndElements(load);
    int64_t bitWidth = dataType ? getTypeBitWidth(dataType) : 32;

    // 字节数 = 元素数 × 循环乘数 × (位宽/8)
    double bytes =
        static_cast<double>(dataElements) * loopMultiplier * bitWidth / 8.0;

    // 估算 warp 指令数
    // SIMT warp size = 32, 所以一条 warp 指令可以处理 32 个元素
    // warp_instructions = ceil(elements / 32) * loopMultiplier
    int64_t warpInstructions =
        static_cast<int64_t>(std::ceil(dataElements / 32.0)) * loopMultiplier;

    if (load) {
        features.loadBytes += bytes;
        features.loadWarpInstructions += warpInstructions;
    } else {
        features.storeBytes += bytes;
        features.storeWarpInstructions += warpInstructions;
    }
}
```

`dataTypeAndElements` lambda（行 1524-1535）的逻辑：
- Load：取 result(0) 的类型和元素数
- Store：取 operand(1) 的类型和元素数（store 的第二个参数是要存的值）
- Fallback：取 operand(0) 的类型

#### 8.1.9 遍历内部：MatMul FLOPs（lines 1554-1567）

```cpp
if (name == "tt.dot" && op->getNumOperands() >= 2) {
    auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (lhs && rhs && lhs.getRank() >= 2 && rhs.getRank() >= 2) {
        // 取倒数第二维和最后一维：lhs 是 [..., M, K], rhs 是 [..., K, N]
        int64_t m = lhs.getShape()[lhs.getRank() - 2];
        int64_t k = lhs.getShape()[lhs.getRank() - 1];
        int64_t n = rhs.getShape()[rhs.getRank() - 1];
        if (m > 0 && n > 0 && k > 0) {
            // FLOPs = 2 × M × N × K（每个元素一次乘加 = 2 FLOPs）
            features.dotFlops += 2 * m * n * k * loopMultiplier;
            features.dotOutputElements += m * n * loopMultiplier;
            features.dotMNK.push_back({m, n, k});  // 记录每个 dot 的 shape
        }
    }
}
```

#### 8.1.10 遍历内部：Reduction 分类（lines 1582-1592）

```cpp
if (name == "tt.reduce") {
    // rowLocalReduceOps：输入和输出的 rank 不同 → "沿着某一行做的 reduction"
    // 例如 reduce [M,N] → [M] 就是 row-local reduce
    if (rankedResultAndOperandRanks.size() > 1) {
        auto [minimum, maximum] =
            std::minmax_element(rankedResultAndOperandRanks.begin(),
                                rankedResultAndOperandRanks.end());
        if (*maximum > *minimum)
            ++features.rowLocalReduceOps;    // 最大 rank > 最小 rank → 降维了
    }
    // vectorReduceToScalarOps：有 ranked input 但没有 ranked output → 全场 reduce 成标量
    if (hasRankedInput && !hasRankedResult)
        ++features.vectorReduceToScalarOps;
}
```

#### 8.1.11 遍历内部：Mask 分析（lines 1594-1607）

```cpp
std::vector<int64_t> maskRanks;
for (Type type : op->getOperandTypes())
    if (isMaskTensorType(type))  // i1 类型（boolean）= mask
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());
for (Type type : op->getResultTypes())
    if (isMaskTensorType(type))
        maskRanks.push_back(cast<RankedTensorType>(type).getRank());

if (!maskRanks.empty()) {
    ++features.maskTensorOps;              // 出现了 mask 的 op 数
    for (int64_t rank : maskRanks)
        features.maskRankSum += rank;       // mask 的总 rank 之和（评分时用）
    if (name == "tt.broadcast" || name == "tt.expand_dims")
        ++features.maskBroadcastOps;        // 在 broadcast/expand_dims 上的 mask
}
```

#### 8.1.12 遍历内部：指针结构分析（lines 1609-1641）

```cpp
bool isPointerOperation =
    name == "tt.addptr" || name == "tt.load" || name == "tt.store";
if (isPointerOperation) {
    // 收集这个 op 中所有指针张量的唯一 shape（字符串去重）
    std::set<std::string> uniqueShapes;
    auto collectShape = [&](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor) return;
        // 把 shape 编码成字符串如 "2x32x64"（rank=2, dims=[32,64]）
        std::string key;
        llvm::raw_string_ostream os(key);
        os << tensor.getRank();
        for (int64_t dim : tensor.getShape())
            os << 'x' << dim;
        os.flush();
        uniqueShapes.insert(std::move(key));
    };
    for (Type type : op->getOperandTypes()) collectShape(type);
    for (Type type : op->getResultTypes()) collectShape(type);

    int64_t maxPointerRank = 0;
    for (const std::string &shape : uniqueShapes) {
        // 从 shape 字符串开头解析出 rank
        int64_t rank = 0;
        (void)shapeRef.take_front(shapeRef.find('x')).getAsInteger(10, rank);
        ++features.pointerTensorOps;
        maxPointerRank = std::max(maxPointerRank, rank);
        if (rank > 1)
            // pointerUnstructuredDims: 多维指针的维度总和
            // 例如 rank=2 的指针 count 为 2
            features.pointerUnstructuredDims += rank;
    }
    if (maxPointerRank > 1)
        // laneDependentPointerOps: 存在多维指针的 op 数
        // 这是"数据不规则程度"的核心度量
        ++features.laneDependentPointerOps;
}
```

`pointerUnstructuredDims` 和 `laneDependentPointerOps` 是整个模型中的核心信号：

| 值 | 含义 |
|---|---|
| `laneDependentPointerOps = 0` | 所有读写都是线性地址（vecadd 那种）→ SIMD 友好 |
| `laneDependentPointerOps` 大 | 很多间接/多维寻址 → SIMT 有优势 |
| `pointerUnstructuredDims` 大 | 指针维度高 → SIMD 需要大量标量展开 |

#### 8.1.13 遍历内部：标量访存和 Splat 检测（lines 1643-1659）

```cpp
// 检测标量 load/store——没有 ranked tensor 输入输出的 ptr load/store
// 这是 SIMD 的弱项（向量核对标量访存效率很低）
bool anyRankedType = llvm::any_of(op->getOperandTypes(),
    [](Type type) { return isa<RankedTensorType>(type); });
anyRankedType |= llvm::any_of(op->getResultTypes(),
    [](Type type) { return isa<RankedTensorType>(type); });

if (name == "tt.load" && !anyRankedType && op->getNumOperands() > 0 &&
    isPointerType(op->getOperand(0).getType()))
    ++features.scalarLoadOps;       // 标量 load

if (name == "tt.store" && !anyRankedType && op->getNumOperands() > 0 &&
    isPointerType(op->getOperand(0).getType()))
    ++features.scalarStoreOps;      // 标量 store

if (name == "tt.splat" && op->getNumOperands() > 0 &&
    op->getNumResults() > 0 &&
    isPointerType(op->getOperand(0).getType()) &&   // splat 的输入是指针 → 指针广播
    isa<RankedTensorType>(op->getResult(0).getType()))
    ++features.vectorPtrSplatOps;   // 把指针 splat 成向量 → 间接访存信号
```

#### 8.1.14 遍历结束后的汇总计算（lines 1662-1695）

```cpp
// 粗粒度的 scalarOps 总数
features.scalarOps =
    features.addOps + features.subOps + features.mulOps +
    features.divOps + features.maxOps + features.absOps +
    features.expOps + features.logOps + features.cmpOps +
    features.selectOps + features.castOps + features.clampOps;

// 布尔标记
features.hasDot = features.dotOps > 0;
features.hasGather = features.gatherOps > 0;
features.hasAtomic = features.atomicOps > 0;

// rank1IndirectVectorReduce: 一种特定的模式——
// rank=1 的 tensor + reduce + ptr splat + >=2 个标量 load
// → 这是"在 1D buffer 上做间接索引的 reduce"
features.rank1IndirectVectorReduce =
    features.maxTensorRank == 1 && features.reduceOps > 0 &&
    features.vectorReduceToScalarOps > 0 &&
    features.vectorPtrSplatOps > 0 && features.scalarLoadOps >= 2;

// observedMixedKinds: 给诊断/调试用的标签
if (features.gatherOps > 0)
    appendUnique(features.observedMixedKinds, "direct_gather");
// ...
bool indirectMixedCandidate =
    features.rank1IndirectVectorReduce ||
    (features.laneDependentPointerOps > 0 &&
     (features.maskBroadcastOps > 0 || features.staticLoopCount > 0 ||
      (features.dotOps > 0 && features.loadOps >= 3)));
if (indirectMixedCandidate)
    appendUnique(features.observedMixedKinds, "conditional_indirect_memory");

return features;  // 返回完整的特征结构体
```

---

### 8.2 `estimateSimdSimtCandidates()` — 评分与决策（~370 行）

输入：`SimdSimtFeatureSummary`（上面提取的特征）+ `SimdSimtCostModelOptions`（用户配置）

输出：`SimdSimtCostReport`（三个候选的分数 + 最终决定 + 是否通过门控）

#### 8.2.1 函数签名与前期校验（lines 1698-1710）

```cpp
llvm::Expected<SimdSimtCostReport>
mlir::ascend::estimateSimdSimtCandidates(
    const SimdSimtFeatureSummary &features,     // 特征（输入）
    const SimdSimtCostModelOptions &options) {  // 选项（输入）

    // 参数校验：marginRatio 必须在 [0, +∞)
    if (!std::isfinite(options.marginRatio) || options.marginRatio < 0.0)
        return llvm::createStringError(...);

    // 从 JSON 文件加载硬件 profile
    auto profileOrError = loadCandidateProfile(options.profilePath);
    if (!profileOrError)
        return profileOrError.takeError();
    CandidateProfile profile = std::move(*profileOrError);
```

#### 8.2.2 初始化 Report（lines 1711-1730）

```cpp
    SimdSimtCostReport report;
    // 把 profile 的元信息写入 report（版本号、目标芯片、SHA256 等）
    report.profileVersion = profile.profileVersion;
    report.profileTarget = profile.target;
    report.actualTarget = options.actualTarget;
    report.profileContentSha256 = profile.contentSha256;
    report.scoreUnit = profile.scoreUnit;
    report.minimumConfidenceForDecision = profile.minimumConfidence;

    // targetMatches: 检查 profile 的目标芯片是否匹配实际芯片
    report.targetCompatible = targetMatches(profile, options.actualTarget);

    report.features = features;        // 把输入特征存进 report（用于 JSON 输出）
    report.marginRatio = options.marginRatio;
```

#### 8.2.3 校准覆盖检查（lines 1732-1757）

这是个重要的门控：如果 kernel 特征超出 profile 的校准范围，直接拒评。

```cpp
    // 计算 irregularDensity（不规则密度）
    // = laneDependentPointerOps / pointerTensorOps
    //    （有多维指针的 op 数）/（有指针的 op 总数）
    // 范围是 [0, 1]。全连续 → 0；很多离散 → 接近 1
    const int64_t pointerOps =
        std::max<int64_t>(1, features.pointerTensorOps);
    report.breakdown.irregularDensity =
        std::min(1.0, static_cast<double>(features.laneDependentPointerOps) /
                       pointerOps);

    // rankingCalibrationCoverage: 检查特征是否在 profile.coverage 定义的域内
    // 如果 irregularDensity < minimumIrregularDensity、dotFlops > tinyDotFlopsMax
    // 等，就会判定为 out_of_calibration_domain
    auto [covered, domain] = rankingCalibrationCoverage(
        features, weightedReductions, dotFlops, profile,
        report.breakdown.irregularDensity);
    report.calibrationCovered = covered;
    report.calibrationDomain = std::move(domain);

    if (!report.calibrationCovered &&
        !options.scoreOutsideCalibrationCoverage) {
        report.gateReasons.push_back("selection_score_invalid");
        report.gatePassed = false;
        return report;  // 提前返回，不执行评分
    }
```

#### 8.2.4 资源成本：Op 级 Throughput 计算（lines 1759-1818）

这是整个评分模型的核心。对 profile 里列出的**每一个** op 类型，分别计算 SIMD 和 SIMT 下的 cycles：

```cpp
    const int64_t numWarps = std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
    const int64_t maxNumel = std::max<int64_t>(1, features.maxTensorNumel);
    const int64_t elementBits = std::max<int64_t>(8, features.maxElementBits);

    // vectorWidth: SIMD 向量宽度，单位是"元素数"不是 bit
    //   例如 vector_width_bits=2048, f32(32bit) → vectorWidth=64 个 f32 元素
    //   即 SIMD 一条指令可以并行处理 64 个 f32
    const int64_t vectorWidth =
        std::max<int64_t>(1, profile.simdVectorWidthBits / elementBits);
```

接着遍历 profile 中的 op 列表：

```cpp
    for (const auto &[opName, elements] : getProfileOpElements(features)) {
        // elements: 这个 op 的加权元素数（来自 features.opElements）
        // 例如 features.opElements["add"] = 262144 意味着总共处理了 26 万个 f32

        auto simdIterator = profile.simdOps.find(opName);
        auto simtIterator = profile.simtOps.find(opName);
        if (simdIterator == profile.simdOps.end() ||
            simtIterator == profile.simtOps.end()) {
            report.unsupported.push_back(opName.str());
            continue;  // profile 里没有这个 op → 标记为 unsupported
        }

        const OpProfile &simd = simdIterator->second;
        const OpProfile &simt = simtIterator->second;

        // SIMD cycles:
        //   需要多少条向量指令 = ceil(elements / vectorWidth)
        //   每条向量指令耗时 = 1 / throughput
        //   总 cycles = 指令数 / throughput * factor
        double simdCycles =
            std::ceil(static_cast<double>(elements) / vectorWidth) /
            simd.throughput * simd.factor;

        // SIMT cycles:
        //   SIMT 不需要显式考虑 vectorWidth——warp 会自动并行
        //   总 cycles = elements / throughput * factor
        double simtCycles =
            static_cast<double>(elements) / simt.throughput * simt.factor;

        report.breakdown.simdOpSystemCycles[opName] = simdCycles;
        report.breakdown.simtOpSystemCycles[opName] = simtCycles;
        report.breakdown.simdComputeCycles += simdCycles;
        report.breakdown.simtComputeCycles += simtCycles;
    }
```

**为什么 SIMD 有 `ceil(elements / vectorWidth)` 而 SIMT 没有？**
- SIMD 是向量指令：一条指令处理 `vectorWidth` 个元素（如 64 个 f32），如果元素数不能被整除，需要额外的部分向量指令 → 向上取整
- SIMT 是 warp 指令：32 个线程自动并行，程序员/编译器不需要手动打包 → 直接除 throughput

#### 8.2.5 资源成本：Memory（lines 1820-1851）

```cpp
    // SIMD memory
    // MTE2 = Memory Transfer Engine 2 (load), MTE3 = Memory Transfer Engine 3 (store)
    // SIMD 有独立的 DMA 引擎，load/store 可以和计算并行（所以后面用 max）
    report.breakdown.simdLoadCycles =
        features.loadBytes / profile.simdMte2BytesPerCycle;
    report.breakdown.simdStoreCycles =
        features.storeBytes / profile.simdMte3BytesPerCycle;
    report.breakdown.simdMemoryCycles =
        std::max(report.breakdown.simdLoadCycles,
                 report.breakdown.simdStoreCycles);
    // ↑ max: load 和 store 用不同 DMA 通道，可以并行，取瓶颈

    // SIMT memory
    // load/store 不能像 SIMD 那样和计算完全并行——warp scheduler 是串行的
    report.breakdown.simtLoadCycles =
        loadWarpInstructions / profile.simtLoadWarpRate;
    report.breakdown.simtStoreCycles =
        storeWarpInstructions / profile.simtStoreWarpRate;
    report.breakdown.simtMemoryCycles =
        report.breakdown.simtLoadCycles + report.breakdown.simtStoreCycles;
    // ↑ + sum: SIMT 的 load 和 store 使用同一总线，串行执行
```

#### 8.2.6 资源成本：Shuffle 和 Predicate（lines 1857-1873）

```cpp
    // Shuffle: warp 内 reduction（如 sum/max across lanes）需要通过 shuffle 指令
    //   shuffle 层级数 = log2(warp_size)（如 warp=32 → 5 级 shuffle）
    //   shuffle 总指令数 = reductions × ceil(maxNumel/warp_size) × shuffle_levels
    const int64_t shuffleLevels = static_cast<int64_t>(
        std::ceil(std::log2(static_cast<double>(profile.simtWarpSize))));
    report.breakdown.simtShuffleInstructions =
        static_cast<double>(weightedReductions + weightedScans) *
        std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize) *
        shuffleLevels;
    report.breakdown.simtShuffleCycles =
        report.breakdown.simtShuffleInstructions / profile.simtShuffleRate;

    // Predicate: mask 材料化
    //   SIMT 需要显式 predicate 来处理不规则 mask
    report.breakdown.simtPredicateInstructions =
        static_cast<double>(features.maskRankSum) *
        std::ceil(static_cast<double>(maxNumel) / profile.simtWarpSize);
    report.breakdown.simtPredicateCycles =
        report.breakdown.simtPredicateInstructions / profile.simtPredicateRate;
```

#### 8.2.7 合成 Analytical Cycles（lines 1886-1904）

```cpp
    // 固定开销（setup）
    report.breakdown.simdSetupCycles = profile.simdSetupCycles;
    report.breakdown.simtSetupCycles = profile.simtSetupCycles;

    // SIMD payload = max(compute + dot, memory)  ← compute 和 DMA 并行
    report.breakdown.simdIssuePayloadCycles =
        std::max(report.breakdown.simdComputeCycles +
                 report.breakdown.simdDotCycles,
                 report.breakdown.simdMemoryCycles);

    // SIMT payload = max(compute + shuffle + dot, memory) + predicate
    //                ← shuffle/dot 和 memory 互斥，predicate 额外串行
    report.breakdown.simtIssuePayloadCycles =
        std::max(report.breakdown.simtComputeCycles +
                 report.breakdown.simtShuffleCycles +
                 report.breakdown.simtDotCycles,
                 report.breakdown.simtMemoryCycles) +
        report.breakdown.simtPredicateCycles;

    // programIssueScale: 指令发射效率（默认 1.0，可通过 profile 调整）
    report.breakdown.programIssueScale = profile.programIssueScale;

    // 最终的 analytical cycles = setup + payload * issue_scale
    report.breakdown.simdAnalyticalCycles =
        profile.simdSetupCycles +
        report.breakdown.simdIssuePayloadCycles * profile.programIssueScale;
    report.breakdown.simtAnalyticalCycles =
        profile.simtSetupCycles +
        report.breakdown.simtIssuePayloadCycles * profile.programIssueScale;
```

核心公式：

```
SIMD:  setup + max(compute + dot, max(load, store)) × issue_scale
        └─ setup 固定 ─┘  └──── compute 和 DMA 并行 ────┘

SIMT:  setup + [max(compute + shuffle + dot, load + store) + predicate] × issue_scale
        └─ setup 固定 ─┘  └─ compute/访存串行 ─┘  └─ predicate 额外 ─┘
```

#### 8.2.8 结构惩罚（lines 1906-1955）

```cpp
    // 检测 tiny dot（小矩阵乘法）
    // dotFlops > 0 但 <= 阈值 → 不够填满 SIMD Cube 单元 → 结构上有劣势
    const bool tinyDot =
        dotFlops > 0 && dotFlops <= profile.structural.tinyDotFlopsMax;
    report.breakdown.tinyDotUnderfill =
        tinyDot ? std::max(0.0, 1.0 - dotFlops / profile.structural.tinyDotFlopsMax)
                : 0.0;

    // 不规则访存惩罚
    //   tiny dot 时用 tinyDotIrregularPerDensity 否则用 irregularPerDensity
    report.breakdown.structuralComponents["irregular_addressing"] =
        std::min(irregularCap,
                 irregularDensity * irregularPerDensity);

    // Mask 材料化惩罚
    report.breakdown.structuralComponents["mask_materialization"] =
        std::min(maskCap, maskRankSum * perMaskRank);

    // Reduction 展开惩罚
    report.breakdown.structuralComponents["reduction_lowering"] =
        std::min(reductionCap, weightedReductions * perWeightedReduction);

    // 静态循环控制惩罚
    report.breakdown.structuralComponents["static_loop_control"] =
        std::min(loopCap, staticLoopTripCountSum * perStaticLoopTrip);

    // 控制流惩罚（if/while → warp divergence）
    report.breakdown.structuralComponents["control_flow"] =
        hasControlFlow ? controlFlow_penalty : 0.0;

    // 把所有惩罚分量累加起来 → structuralPenaltyRatio
    for (const auto &component : report.breakdown.structuralComponents)
        report.breakdown.structuralPenaltyRatio += component.second;

    // structuralFloorCycles: SIMT_cycles × (1 + penalty)，作为 SIMD 的下限
    report.breakdown.structuralFloorCycles =
        penaltyRatio > 0.0 ? simtAnalytical * (1 + penaltyRatio) : 0.0;
```

#### 8.2.9 三个候选的分数（lines 1951-1955, 1974-2016）

```cpp
    // All SIMD：取 analytical 和 structural floor 的最大值
    //   如果 structural penalty 很高，SIMD 不能低于 floor
    report.candidateCosts.allSimd =
        std::max(report.breakdown.simdAnalyticalCycles,
                 report.breakdown.structuralFloorCycles);

    // All SIMT Only：直接用 analytical，不加 floor
    report.candidateCosts.allSimtOnly =
        report.breakdown.simtAnalyticalCycles;

    // ── 混合模式 ──

    // 找到最接近实际 numWarps 的 transition profile
    // transition 记录了不同 warp 数下 SIMD→SIMT 的切换代价
    const TransitionProfile *nearestTransition = nullptr;
    for (const TransitionProfile &transition : profile.transitions)
        if (!nearestTransition ||
            std::abs(transition.numWarps - numWarps) <
                std::abs(nearestTransition->numWarps - numWarps))
            nearestTransition = &transition;

    // mixedSetupCycles: 混合模式的额外 setup（比纯 SIMT 多一个"同时初始化"
    // SIMD 和 SIMT 上下文"的步骤）
    report.breakdown.mixedSetupCycles =
        nearestTransition->emptySimtSetupCycles;
    report.breakdown.transitionDeltaCycles =
        std::max(0.0, report.breakdown.mixedSetupCycles -
                       report.breakdown.standaloneSimtSetupCycles);

    // mixedBlend: 混合模式中 SIMD 占的比例（0=全 SIMT, 1=全 SIMD）
    //   由 base + 循环/掩码/规约/控制流 的分量组成
    double mixedBlend = profile.mixedBlend.base;
    mixedBlend += std::min(loopCap, staticLoopTripCountSum * perLoop);
    mixedBlend += std::min(maskCap, maskBroadcastOps * perMaskBroadcast);
    mixedBlend += std::min(reductionCap, weightedReductions * perReduction);
    if (hasControlFlow) mixedBlend += controlFlow_penalty;
    mixedBlend = std::min(mixedCap, std::max(0.0, mixedBlend));

    report.breakdown.mixedSimdFraction = mixedBlend;

    // 混合 payload = SIMT_payload + blend × (SIMD_payload - SIMT_payload)
    //   当 blend=0 → 纯 SIMT payload
    //   当 blend=1 → 纯 SIMD payload
    double simdPayload = max(0.0, allSimdCost - simdSetupCycles);
    double simtPayload = max(0.0, allSimtOnlyCost - simtSetupCycles);
    double mixedPayload =
        simtPayload + mixedBlend * (simdPayload - simtPayload);

    // 混合总成本 = 混合 setup + 混合 payload
    report.candidateCosts.mixedSimdSimt =
        report.breakdown.mixedSetupCycles + std::max(0.0, mixedPayload);

    // 特殊处理: tiny dot → 混合模式直接用 SIMT + 小矩阵残余修正
    if (tinyDot) {
        report.breakdown.tinyDotMixedResidualRatio =
            profile.tinyDotMixedPenaltyAtZero * report.breakdown.tinyDotUnderfill;
        report.candidateCosts.mixedSimdSimt =
            report.candidateCosts.allSimtOnly *
            (1.0 + report.breakdown.tinyDotMixedResidualRatio);
    }
```

#### 8.2.10 决策与 Gate（lines 2018-2070）

```cpp
    report.candidateCostsEvaluated = true;

    // 选最优
    report.decision = chooseBest(report.candidateCosts);
    report.runnerUp = chooseRunnerUp(report.candidateCosts, report.decision);

    // bestScore 和决策优势
    report.bestScore = report.candidateCosts.get(report.decision);
    report.decisionAdvantage =
        report.decision == SimdSimtCandidateKind::AllSIMD
            ? report.runnerUpScore - report.bestScore  // 如果选了 SIMD:
            //   advantage = 第二名 - 第一名（SIMD 需要足够大优势才能胜出）
            : report.candidateCosts.allSimd - report.bestScore;
            //   如果选了 SIMT/Mixed: advantage = all_simd - best
            //   （与 baseline SIMD 的差距）

    // gainScore = decisionAdvantage（别名）
    report.gainScore = report.decisionAdvantage;

    // 需要的 gain：max(64 cycles, all_simd * marginRatio)
    //   即至少需要 64 cycles 或 10%（默认）的改善
    report.requiredGainScore =
        std::max(64.0, report.candidateCosts.allSimd * options.marginRatio);

    // 各种 gate
    if (!report.targetCompatible)
        report.gateReasons.push_back("target_incompatible");
    if (!report.selectionScoreValid)
        report.gateReasons.push_back("selection_score_invalid");
    if (!report.unsupported.empty())
        report.gateReasons.push_back("unsupported_cost_terms");
    if (confidenceRank(report.rankingConfidence) <
        confidenceRank(report.minimumConfidenceForDecision))
        report.gateReasons.push_back("ranking_confidence_too_low");
    if (!(report.decisionAdvantage > report.requiredGainScore))
        report.gateReasons.push_back("decision_advantage_not_above_required_gain");

    // gatePassed = 没有 gateReasons
    report.gatePassed = report.gateReasons.empty();
    return report;
```

**`chooseBest`**（lines 1066-1095）：

```cpp
static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores) {
    // 优先级: 先看 all_simd vs all_simt_only，相等时选 mixed
    if (scores.allSimd <= scores.allSimtOnly && scores.allSimd <= scores.mixedSimdSimt)
        return SimdSimtCandidateKind::AllSIMD;
    if (scores.allSimtOnly <= scores.allSimd && scores.allSimtOnly <= scores.mixedSimdSimt)
        return SimdSimtCandidateKind::AllSIMTOnly;
    return SimdSimtCandidateKind::MixedSIMDSIMT;
}
```

**Gate 不通过的后果**（Python 侧 `compiler.py`）：

```python
# effective 默认是 "all_simd" 或 "backend_default"
# 如果 gatePassed == false, C++ 不会改变 effective
# Python 读到 "all_simd" → 什么都不做，继续正常 SIMD 编译
```

---

### 8.3 两个函数的协作总览

```
analyzeSimdSimtFeatures(module)
  │
  │ 遍历 TTIR 中的每一个 op
  │  ├─ 分类 op 类型（tt.load/tt.dot/arith.addf/...）
  │  ├─ 加权计数（乘以 loop multiplier）
  │  ├─ 统计结构特征（指针维度、mask rank、标量访存数...）
  │  ├─ 计算字节数、FLOPs、warp 指令数
  │  └─ 产出 SimdSimtFeatureSummary
  │
  ▼
estimateSimdSimtCandidates(features, options)
  │
  │ 1. 加载 JSON profile（硬件参数表）
  │ 2. 校准覆盖检查（特征是否在 profile 域内）
  │ 3. 对每个 op: 用 profile 里的 throughput 算 SIMD/SIMT cycles
  │    ├─ compute: ceil(elements/vectorWidth) / throughput
  │    ├─ memory:  bytes / BW、warp_insns / warp_rate
  │    ├─ dot:     setup + FLOPs / FLOPS_per_cycle
  │    ├─ shuffle: reductions * nel/32 * log2(32) / shuffle_rate
  │    └─ predicate: mask_rank_sum * nel/32 / predicate_rate
  │ 4. 合成 analytical cycles
  │    ├─ simd: setup + max(compute+dot, memory)
  │    └─ simt: setup + max(compute+shuffle+dot, memory) + predicate
  │ 5. 加结构惩罚（irregular/mask/reduction/loop/control_flow）
  │ 6. 算三个候选分数
  │    ├─ all_simd:   max(simd_analytical, simt*(1+penalty))
  │    ├─ all_simt_only: simt_analytical
  │    └─ mixed:  setup + simt_payload + blend*(simd_payload - simt_payload)
  │ 7. chooseBest → decision
  │ 8. Gate 检查 → gatePassed
  │
  ▼
SimdSimtCostReport
  ├─ decision: AllSIMD / AllSIMTOnly / MixedSIMDSIMT
  ├─ candidateCosts: 三个候选的分数
  ├─ gatePassed: 是否通过门控
  └─ breakdown: 详细的 cost breakdown（JSON 输出用）
```

## 九、`compile_mode` 是怎么传到 costmodel 的

问题：用户在 kernel 调用时写的 `compile_mode="simd_simt"`，最终是怎么到达 `_run_cpp_simd_simt_costmodel` 的？

### 9.1 完整链路

```
用户代码
  kernel[grid](args, compile_mode="simd_simt")
        │
        ▼
JITFunction.run()                                        ← jit.py:695
        │
        ├─ binder(*args, **kwargs)                       ← jit.py:710
        │    binder 是 create_function_from_signature() 动态生成的函数
        │    它有 **options 参数，会捕获所有非 kernel 形参的 kwarg
        │    包括 compile_mode="simd_simt"
        │    返回 (params, specialization, options_dict)
        │    options_dict = {"compile_mode": "simd_simt", ...}
        │
        ├─ _pack_args(backend, kwargs, ...)              ← jit.py:717
        │    │
        │    └─ backend.parse_options(kwargs)             ← jit.py:673
        │         │
        │         └─ backend/compiler.py:AscendBackend.parse_options()
        │               │
        │               └─ 从 kwargs 里提取 NPUOptions 字段
        │                  args = {k: kwargs[k] for k in NPUOptions 字段名 if k in kwargs}
        │                  → compile_mode="simd_simt" 被提取
        │                  → NPUOptions(compile_mode="simd_simt", ...)
        │
        │            NPUOptions.__post_init__()           ← compiler.py:1139
        │               compile_mode == "simd_simt"
        │               → force_simt_template = True
        │               → parallel_mode = "mix_simd_simt"
        │               → auto_simt_scope_mode = 标准化(环境变量或默认值)
        │
        ├─ _do_compile(key, signature, ..., options, ...)  ← jit.py:720
        │    │
        │    └─ triton.compile(src, target=..., options=options)
        │         │                                    ← compiler/compiler.py:228
        │         ├─ metadata = {
        │         │      "hash": hash,
        │         │      "target": target,
        │         │      **options.__dict__,  ← 这里！NPUOptions 所有字段展开进 metadata
        │         │      **env_vars,
        │         │    }
        │         │    → metadata["compile_mode"] = "simd_simt"
        │         │    → metadata["force_simt_template"] = True
        │         │    → metadata["parallel_mode"] = "mix_simd_simt"
        │         │
        │         └─ backend.add_stages(stages, options, ...)
        │               │
        │               └─ stages["ttadapter"] = lambda src, metadata:
        │                       ttir_to_linalg(src, metadata, options, ...)
        │                                            ↑         ↑
        │                                     metadata 里有    opt=NPUOptions
        │                                     compile_mode      对象也直接传入
        │
        └─ 每个 stage 按序执行，metadata 贯穿所有 stage
               │
               ▼
          ttir_to_linalg(mod, metadata, opt)             ← compiler.py:182
               │
               ├─ _run_cpp_simd_simt_costmodel(mod, metadata, opt)
               │    │
               │    ├─ opt.auto_simt_scope_mode  ← 从 NPUOptions 直接读
               │    └─ metadata.get("compile_mode") ← 从 metadata dict 读
               │       → 两个值都来自同一个 NPUOptions
               │
               └─ metadata["compile_mode"] == "simd_simt"
                    且 opt.auto_simt_scope_mode != "off"
                    → 运行 SelectSimdSimtCostModel + MaterializeSimtScopes
```

关键点在第 284 行：

```python
metadata = {
    "hash": hash,
    "target": target,
    **options.__dict__,    # ← NPUOptions 的所有字段一字排开
    **env_vars,
}
```

因为 `NPUOptions` 是一个 `@dataclass`，`options.__dict__` 包含所有字段：`compile_mode`、`force_simt_template`、`parallel_mode`、`num_warps`、`auto_simt_scope_mode` 等。它们是自动进入 metadata 的，不需要手动拷贝。

### 9.2 为什么在 C++ pass 里能同时读到 options 和 metadata

```python
def _run_cpp_simd_simt_costmodel(mod, metadata, opt) -> str:
    mode = opt.auto_simt_scope_mode                    # ← 从 NPUOptions 直接读
    if mode == "off" or metadata.get("compile_mode") != "simd_simt":  # ← 从 metadata 读
        return "backend_default"
```

两个来源其实是同一个 NPUOptions 对象的不同视图——`opt` 是原始 dataclass，`metadata` 是它的 dict 展开。代码里有时读 `opt.xxx` 有时读 `metadata["xxx"]`，本质上是一样的。

---

## 十、SelectSimdSimtCostModel Pass — 逐行解读（~250 行）

这是整个 autoscope 的"驾驶员"：它调用评分模型、解读结果、做出最终决策、标记 SIMT anchor op。文件：`costmodel/lib/AscendModel/Transforms/SelectSimdSimtCostModel.cpp`。

### 10.1 Pass 的输入参数

```cpp
// SelectSimdSimtCostModelPass 的成员变量（在 Passes.td 中定义）：
//   mode              ← "auto" 或 "report"
//   profilePath       ← JSON profile 文件路径
//   actualTarget      ← 当前芯片型号（如 "Ascend950PR_9579"）
//   numWarps          ← warp 数（默认 32）
//   marginRatio       ← 决策阈值（默认 0.10 = 10%）
//   compileOn91095    ← 是否在 A5 上编译
//   dumpPath          ← JSON 报告输出路径（可选）
```

### 10.2 `isMixedSimtAnchor()` — 判断一个 op 是否应标为 SIMT

```cpp
static bool isMixedSimtAnchor(Operation *op, bool compileOn91095) {
    // 只针对 A5 芯片；非 A5 没有 SIMT 硬件，直接返回 false
    if (!compileOn91095)
        return false;

    llvm::StringRef name = op->getName().getStringRef();

    // ── 总是走 SIMT 的 op 类型 ──
    if (name == "tt.gather" || name == "tt.histogram")
        return true;                          // gather/histogram 天生离散 → 总是 SIMT

    // ── 有条件走 SIMT 的 op ──
    if (name == "tt.scan")
        return isPlainOneDimensionalCumsum(op);  // 只有 1 维 cumsum（加和）适合 SIMT

    if (name.starts_with("tt.atomic"))
        return hasTensorPointerOperand(op);      // atomic 操作如果操作对象是张量 → SIMT

    if (name == "tt.load" || name == "tt.store")
        // 关键判断：这个 load/store 的指针是否依赖"从内存加载出来的索引"？
        // 即：是不是间接访存（indirect load/store）
        return hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op);

    return false;  // 其他所有 op（dot, add, mul, ...）→ 不标 SIMT，走默认 SIMD
}
```

**`pointerDependsOnLoadedIndex` 做了什么**（lines 97-128）：

从一个 memory op（load/store）的指针操作数出发，沿着 MLIR 的 SSA use-def 链反向追踪。如果能追溯到另一个 `tt.load`（即这个指针的索引来自之前的 load 结果），说明这是间接访存——数据依赖动态值。

```cpp
static bool pointerDependsOnLoadedIndex(Operation *memoryOp) {
    SmallVector<Value> worklist{memoryOp->getOperand(0)};  // 从指针操作数开始
    llvm::DenseSet<Value> visited;                          // 避免重复访问

    while (!worklist.empty()) {
        Value value = worklist.pop_back_val();              // BFS 遍历
        if (!visited.insert(value).second) continue;        // 已访问 → 跳过

        // 如果这个 value 是 BlockArgument（如 scf.for 的循环变量）
        // 需要跨越循环边界追踪
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            // ... 处理循环传递的值 ...
            continue;
        }

        // 如果这个 value 产生自某个 op
        Operation *producer = value.getDefiningOp();
        if (!producer) continue;

        // ← 关键：如果追溯到 tt.load 或 tt.gather
        //    说明指针依赖已加载的数据 → 间接访存
        llvm::StringRef name = producer->getName().getStringRef();
        if (name == "tt.load" || name == "tt.gather")
            return true;

        // 继续沿 SSA 链反向追踪
        llvm::append_range(worklist, producer->getOperands());
    }
    return false;
}
```

举个例子说明 `pointerDependsOnLoadedIndex` 判断什么：

```mlir
// gather kernel 的 TTIR 片段
%indices = tt.load %indices_ptr ...        // ← tt.load，加载索引数组
%data    = tt.load %data_ptr[%indices] ... // ← 指针依赖 %indices，而 %indices 来自 load
                                          //   → pointerDependsOnLoadedIndex = true
                                          //   → isMixedSimtAnchor = true
                                          //   → 这个 load 被标为 SIMT ← indirect_load
```

如果是普通的连续 load：

```mlir
// vecadd kernel 的 TTIR 片段
%x = tt.load %x_ptr[%linear_offset] ...   // ← 指针是线性偏移（program_id * BLOCK + range）
                                          //   不依赖其他 load 的结果
                                          //   → pointerDependsOnLoadedIndex = false
                                          //   → isMixedSimtAnchor = false
                                          //   → 不标 SIMT，走默认 SIMD
```

### 10.3 `collectMixedSimtAnchors()` — 收集所有应标 SIMT 的 op

```cpp
static SmallVector<Operation *>
collectMixedSimtAnchors(ModuleOp module, bool compileOn91095) {
    SmallVector<Operation *> anchors;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isMixedSimtAnchor(op, compileOn91095)) {
            anchors.push_back(op);           // 记录这个 op
            return WalkResult::skip();       // PreOrder + skip：不递归进入这个 op 内部
                                             //   （避免把 SIMT scope 内的子 op 重复标记）
        }
        return WalkResult::advance();        // 继续向深层遍历
    });
    return anchors;
}
```

`WalkOrder::PreOrder` 确保先处理外层 op，再进入嵌套（这与 `MaterializeSimtScopes` 的 `scope.scope` 包覆配合——如果一个 gather 已经被包进 scope，就不会重复处理里面的 load）。

### 10.4 `runOnOperation()` — Pass 的主入口（lines 202-280）

这是整个 pass 的执行体，分 6 个阶段：

#### 阶段 1：清理 + 初始化

```cpp
ModuleOp module = getOperation();          // 拿到当前 MLIR 模块
clearPreviousSelection(module);            // 清除上一次运行可能留下的 attrs
                                           // （ascend.simt_costmodel.effective / selected 等）

bool autoMode = mode.getValue() == "auto"; // mode 是 pass 参数："auto" → 会自动应用决策
                                           //                     "report" → 只出报告不动 IR

SimdSimtCostModelOptions options;          // 构建 costmodel 选项
options.profilePath   = profilePath.getValue();
options.actualTarget  = actualTarget.getValue();
options.numWarps      = numWarps.getValue();
options.marginRatio   = marginRatio.getValue();
// auto 模式下：out-of-coverage 的 kernel 不评分（安全第一）
// report 模式下：即使 out-of-coverage 也评分（用于离线分析）
options.scoreOutsideCalibrationCoverage = !autoMode;
```

#### 阶段 2：调用评分模型

```cpp
auto reportOr = analyzeSimdSimtCandidates(module, options);  // 第八章详解的评分函数
if (!reportOr) {
    // 如果 costmodel 本身出错（如 profile 解析失败），报错退出
    module.emitError("C++ SIMD/SIMT cost model failed: ")
        << llvm::toString(reportOr.takeError());
    signalPassFailure();
    return;
}
SimdSimtCostReport report = std::move(*reportOr);
// 此时 report.decision  = AllSIMD / AllSIMTOnly / MixedSIMDSIMT
//      report.gatePassed = 是否通过所有门控
```

#### 阶段 3：从 recommended 到 effective（决策生效判定）

```cpp
// recommended = 模型推荐的最优候选
std::string recommended =
    report.candidateCostsEvaluated
        ? stringifySimdSimtCandidate(report.decision).str()  // "all_simd" / "all_simt_only" / "mixed_simd_simt"
        : "backend_default";

// effective 初始化为 "backend_default"（= 不做任何改变）
std::string effective = "backend_default";
SmallVector<Operation *> mixedAnchors;
bool actionSupported = true;

// 对于 mixed 候选：检查是否可以落地
if (recommended == "mixed_simd_simt") {
    if (hasExplicitScope) {
        // 如果 kernel 里已经有手动 scope.scope → 不动（避免冲突）
        actionSupported = false;
    } else {
        // 收集需要标 SIMT 的 op
        mixedAnchors = collectMixedSimtAnchors(module, compileOn91095);
        if (mixedAnchors.empty()) {
            // 如果收集不到任何 anchor → 选 mix 没意义（没有 op 实际需要 SIMT）
            actionSupported = false;
        }
    }
} else if (recommended == "all_simt_only" && hasExplicitScope) {
    // 不要用手动 scope 的 kernel 覆盖为全 SIMT
    actionSupported = false;
}

// ── 只有 auto 模式 + gate 通过 + action 可落地 → 才真正采纳 ──
if (autoMode && report.gatePassed && actionSupported) {
    effective = recommended;                               // 采纳模型推荐

    if (effective == "mixed_simd_simt") {
        // 关键：逐 op 标记！
        for (Operation *anchor : mixedAnchors)
            anchor->setAttr("ascend.simt_costmodel.selected",
                            UnitAttr::get(module.getContext()));
        //    ↑ 给每个 mixed anchor op 打上标记
        //    后续 MaterializeSimtScopes 读取这个标记，包 scope.scope{simt}
    }
}
// 否则 effective 保持 "backend_default" → 走 legacy 路径
```

#### 阶段 4：写入 module attrs（供 Python 读取）

```cpp
Builder builder(module.getContext());
module->setAttr("ascend.simt_costmodel.recommended",
                builder.getStringAttr(recommended));
module->setAttr("ascend.simt_costmodel.effective",
                builder.getStringAttr(effective));
module->setAttr("ascend.simt_costmodel.report_json",
                builder.getStringAttr(reportJSON));
// 写入各个候选的分数（Python 可读，用于日志/调试）
module->setAttr("ascend.simt_costmodel.all_simd_score",
                builder.getF64FloatAttr(report.candidateCosts.allSimd));
module->setAttr("ascend.simt_costmodel.all_simt_score",
                builder.getF64FloatAttr(report.candidateCosts.allSimtOnly));
module->setAttr("ascend.simt_costmodel.mixed_score",
                builder.getF64FloatAttr(report.candidateCosts.mixedSimdSimt));
```

#### 阶段 5：检查 full-SIMT 情况

```cpp
// 检查：整个 kernel body 是否就是单个 void SIMT scope？
// 如果是 → 标记 scope_pure_simt_auto，Python 会直接走全 SIMT 路径
Operation *voidScope = findWholeBodyVoidSimtScope(module);
if (voidScope && effective == "all_simt_only") {
    module->setAttr("ascend.simt_costmodel.scope_pure_simt_auto",
                    builder.getBoolAttr(true));
}
```

#### 阶段 6：报告输出

```cpp
// 如果配置了 dumpPath，把 JSON 报告追加写入文件
if (!dumpPath.getValue().empty()) {
    appendJSONLine(dumpPath.getValue(), reportJSON);
}
```

---

## 十一、MaterializeSimtScopes Pass — 逐行解读（~150 行）

文件：`costmodel/lib/AscendModel/Transforms/MaterializeSimtScopes.cpp`。

它的唯一任务：把 `SelectSimdSimtCostModel` 标记的 op（带有 `ascend.simt_costmodel.selected` 属性）**包进 `scope.scope {simt}` region**，让下游 AscendNPU IR 编译器识别。

### 11.1 为什么需要这个 pass

`SelectSimdSimtCostModel` 只是在 op 上打了一个属性标记：

```mlir
%result = tt.load %ptr ...  {ascend.simt_costmodel.selected}
```

下游编译器不认识这个自定义属性。它认识的是 `scope.scope {vec_mode = "simt"}`。所以需要一个 pass 把属性标记转换为 MLIR region 结构。

### 11.2 `wrapSelectedOperation()` — 将单个 op 包进 scope

```cpp
static LogicalResult wrapSelectedOperation(Operation *op) {
    OpBuilder builder(op);  // builder 在 op 前面创建新指令

    // 创建 scope.scope op
    OperationState scopeState(op->getLoc(), "scope.scope");
    //   scope 的输出类型 = 原 op 的输出类型（保持 SSA 兼容）
    scopeState.addTypes(op->getResultTypes());
    //   设置 vec_mode = "simt"
    scopeState.addAttribute("vec_mode", builder.getStringAttr("simt"));
    //   分配一个空的 Region（后续填入 op）
    scopeState.addRegion();
    Operation *scopeOp = builder.create(scopeState);

    // 把 op 移入 scope region 内
    Block &body = scopeOp->getRegion(0).emplaceBlock();
    op->moveBefore(&body, body.end());  // 移动 op 到 scope 的 body 中

    // 创建 scope.return，把 op 的所有结果返回出去
    builder.setInsertionPointAfter(op);
    builder.create<scope::ReturnOp>(op->getLoc(), op->getResults());

    // 把 scope 的所有结果替换原来 op 的结果（保持 SSA）
    for (auto [i, result] : llvm::enumerate(scopeOp->getResults()))
        op->getResult(i).replaceAllUsesWith(result);

    return success();
}
```

### 11.3 转换过程示意

**输入**（`SelectSimdSimtCostModel` 标记后）：

```mlir
tt.func @kernel(...) {
    %data = tt.load %data_ptr[%idx] {ascend.simt_costmodel.selected} ...
    //                                ↑ 标记：这个 load 要走 SIMT
    %sum  = arith.addf %data, %bias ...
    tt.return
}
```

**输出**（`MaterializeSimtScopes` 执行后）：

```mlir
tt.func @kernel(...) {
    %data = "scope.scope"() <{vec_mode = "simt"}> ({
        %inner = tt.load %data_ptr[%idx] ...       // ← 原来的 load 被包进 scope
        scope.return %inner : tensor<8x32xf32>
    }) : () -> tensor<8x32xf32>
    //     ↑ scope.scope 的 vec_mode="simt" 告诉下游：这段走 SIMT

    %sum = arith.addf %data, %bias ...             // ← add 没被标记，留外面走 SIMD
    tt.return
}
```

下游 AscendNPU IR 编译器遇到 `scope.scope {vec_mode = "simt"}` 时，会把 region 内部的代码用 SIMT 编译器编译，外部的用 SIMD 编译器编译。

### 11.4 `runOnOperation()` — Pass 主入口

```cpp
void runOnOperation() override {
    ModuleOp module = getOperation();

    // 找到所有被标记 selected 的 op
    SmallVector<Operation *> selectedOps;
    module.walk([&](Operation *op) {
        if (isSelectedForSimt(op))  // 检查 ascend.simt_costmodel.selected 属性
            selectedOps.push_back(op);
    });

    // 对每个被标记的 op 执行包覆
    for (Operation *op : selectedOps) {
        // 跳过已经被包进 scope 的 op（避免重复包覆）
        if (hasSelectedAncestor(op))
            continue;
        // 跳过不能材质化的 op（如 terminator、isolated-from-above）
        if (!isMaterializable(op))
            continue;
        wrapSelectedOperation(op);
    }
}
```

`hasSelectedAncestor` 检查祖先链：如果 op 的某个外层 op 已经被包进 scope.scope（比如一个 gather 里嵌套的 load），就不再包覆这个内层 op——因为外层 scope 已经覆盖了它。

---

## 十二、三个 Pass 的协作流程

```
make_ttir() → 标准 TTIR 优化后的 IR
    │
    ▼
ttir_to_linalg()
    │
    ├─ SelectSimdSimtCostModel Pass
    │   ├─ analyzeSimdSimtCandidates(module)    → 评分三选一
    │   ├─ 判断 gatePassed + actionSupported
    │   ├─ 如果选 mixed: collectMixedSimtAnchors → 逐 op 标记 selected
    │   └─ 写入 module attrs（effective, report_json, scores）
    │
    ├─ MaterializeSimtScopes Pass
    │   ├─ 找到所有 selected op
    │   └─ 包进 scope.scope {simt}
    │
    ▼
Python 读取 effective:
    ├─ "all_simd"       → 继续正常 SIMD 编译
    ├─ "all_simt_only"  → force_simt_only=True, 走全 SIMT
    └─ "mixed_simd_simt"→ 结合 scope.scope 标记继续混合编译
    └─ "backend_default"→ gate 没通过/不满足条件 → legacy 路径
```

## 十三、关键设计决策总结

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

---

## 十五、`unstructured_in_simt` 是怎么逐 op 选 SIMT 的

`compile_mode="unstructured_in_simt"` 不走 costmodel。它靠 `UnstructureConversionPass` 对 **每个 load/store op** 独立做静态判断。

### 15.1 决策代码（我们分支的原始代码）

文件：`third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp:532-542`

```cpp
// 对 kernel 里的每一个 load/store op，独立判断：
bool indirectFastPathEnabled =
    compileOn91095Flag              // ① 必须是 A5 芯片
    && forceSimtTemplateFlag        // ② compile_mode 必须是 unstructured_in_simt
                                    //    （NPUOptions.__post_init__ 设 force_simt_template=True）
    && (
        (!ptrOffsetInfo.isStructured()  // ③ 指针是 非连续/离散 的
         && sizeInByte < 64)           //    且连续区域 < 64 字节
        ||
        routeDiscreteMaskToSimt         // ④ 或者上一个 pass 标记了"离散 mask，路由 SIMT"
    );

bool rankWithinLimit = resultShape.size() <= 5;  // 额外限制：维度 ≤ 5

if (indirectFastPathEnabled && rankWithinLimit) {
    tryRewriteIndirectFastPath(...);    // → tt.load → ascend.indirect_load (SIMT)
    return;
}
// 否则 → 回退标量循环 (SIMD)
```

### 15.2 四个条件逐个解释

**条件 ① `compileOn91095Flag`**：必须是在 A5（Ascend 950）上编译。A2/A3 没有 SIMT 硬件，直接跳过。

**条件 ② `forceSimtTemplateFlag`**：由 `NPUOptions.__post_init__` 根据 `compile_mode` 自动设置。`"unstructured_in_simt"` 时设为 True，`"simd"` 时设为 False。`"simt_only"` 时整个 pass 不运行（走另一条编译链路）。

**条件 ③ `!ptrOffsetInfo.isStructured() && sizeInByte < 64`**：这是核心判断。`ptrOffsetInfo.isStructured()` 分析指针表达式是否形成"规则的连续访存"：

- **结构化**（`isStructured() = true`）：指针是 `base + program_id * BLOCK + range(BLOCK)` 这种线性模式。SIMD DMA 可以直接处理，不需要 SIMT。
- **非结构化**（`isStructured() = false`）：指针依赖运行时值，比如 `base + index_array[i]`。SIMD 只能展开成标量循环，SIMT 的 `indirect_load` 更快。
- **`sizeInByte < 64`**：即使是非结构化，如果连续区域太小（< 64 字节），说明数据量不大，SIMT dispatch 开销可能超过收益。

**条件 ④ `routeDiscreteMaskToSimt`**：由 `DiscreteMaskAccessConversion` pass（在前一个 pass 运行）设置。它分析 `mask` 表达式是否非连续——例如 `mask = (indices < N)` 中 `indices` 是动态的。如果是非连续 mask 且满足 A5 + 维数 ≤ 5，就打上这个标记。

### 15.3 例子

```mlir
// Load A: 线性地址 — 结构化，连续
%a = tt.load %a_ptr[%linear_offset] ...
// → isStructured() = true → 条件③不满足 → SIMD（正常 DMA）

// Load B: 间接地址 — 非结构化，离散
%b = tt.load %b_ptr[%dynamic_indices] ...
// → isStructured() = false, sizeInByte = 32 < 64
// → 条件③满足 → rank = 2 ≤ 5
// → indirectFastPathEnabled = true
// → tryRewriteIndirectFastPath → tt.load 转 ascend.indirect_load (SIMT)
```

### 15.4 与 `simd_simt` 模式的本质区别

| | `unstructured_in_simt` | `simd_simt` |
|---|---|---|
| 决策粒度 | **每个 op** 独立 | **先 kernel 级三选一**，再逐 op 标记 |
| 谁决定 SIMT | 编译器 pass 静态规则 | costmodel 评分 → gate 通过 → 才启用 |
| 所有满足条件的 op 都走 SIMT？ | 是（无差别） | 否（只有 costmodel 批准的那些） |
| 需要 costmodel？ | 不需要 | 需要 |
| gather 怎么处理 | 满足条件 → indirect_load；不满足 → 标量循环 | `isMixedSimtAnchor` 直接判定 → SIMT |

---

## 十六、`simd_simt` 模式下怎么转成 `indirect_load`

costmodel 本身不做 IR 转换——它只打分和标记。真正的 `tt.load → ascend.indirect_load` 转换还是由 `UnstructureConversionPass` 完成。关键改动在 `simt_costmodel` 分支里通过 `shouldUseSimtTemplate()` 桥接了 costmodel 的决策和原有的转换 pass。

### 16.1 改了一行：从全局 flag 换成统一判断函数

**原来**（`unstructured_in_simt`）：

```cpp
bool indirectFastPathEnabled =
    compileOn91095Flag && forceSimtTemplateFlag &&  // ← 全局：所有 op 都允许
    ((!ptrOffsetInfo.isStructured() && sizeInByte < 64) || routeDiscreteMaskToSimt);
```

**simt_costmodel 分支改为**：

```cpp
bool simtFastPathRequested =
    shouldUseSimtTemplate(op, forceSimtTemplateFlag);  // ← 逐 op：只对批准的 op 允许
bool indirectFastPathEnabled =
    compileOn91095Flag && simtFastPathRequested &&
    ((!ptrOffsetInfo.isStructured() && sizeInByte < 64) || routeDiscreteMaskToSimt);
```

### 16.2 `shouldUseSimtTemplate()` — 统一两种模式的判断

文件：`third_party/ascend/include/Utils/SimtSelection.h:113-122`

```cpp
inline bool shouldUseSimtTemplate(Operation *op, bool legacyForceSimt) {
    // 如果被 scope.scope {simd} 包着 → 一定不用 SIMT
    if (hasEnclosingVectorMode(op, "simd"))
        return false;

    // 否则检查：op 是否被标记 selected 或被 scope.scope {simt} 包着？
    const bool locallySelected =
        isSelectedForSimt(op) || hasEnclosingVectorMode(op, "simt");

    // 关键分支：costmodel 是否在控制？
    if (isModelControlled(op))
        return locallySelected;        // ← 模型控制：只看局部标记
                                        //   只有 costmodel 批准的 op 才能走 SIMT

    return legacyForceSimt || locallySelected;  // ← legacy 模式：
                                                //   forceSimtTemplateFlag=True
                                                //   → 所有 op 都允许
}
```

### 16.3 两个模式下的行为对比

| | `unstructured_in_simt` | `simd_simt`（costmodel 选 mixed） |
|---|---|---|
| `isModelControlled(op)` | false | true |
| `shouldUseSimtTemplate` 返回 | `true \|\| locallySelected` = **true** | `locallySelected` |
| 哪些 op 能走 SIMT | **所有满足 structural 条件的 op** | **只有 scope.scope{simt} 里或标记了 selected 的 op** |
| 控制来源 | 编译器 pass 静态规则 | costmodel 决策 + 编译器 pass 静态规则 |

### 16.4 完整链路：从 costmodel 到 IR 转换

```
simd_simt 模式下，一个 indirect load op 的完整旅程：
                                                       │
① SelectSimdSimtCostModel                               │
  → isMixedSimtAnchor(op) = true                       │ ← gather / indirect load
  → op 打上 ascend.simt_costmodel.selected 属性        │
                                                       │
② MaterializeSimtScopes                                 │
  → 把 op 包进 scope.scope {simt}                      │
                                                       │
③ （编译器继续...）                                     │
                                                       │
④ UnstructureConversionPass                             │
  → shouldUseSimtTemplate(op, flag)                    │
    → hasEnclosingVectorMode(op, "simt") = true        │ ← 检测到 scope.scope {simt}
    → isModelControlled(op) = true                     │
    → return locallySelected = true                    │
  → indirectFastPathEnabled = true                     │
  → tryRewriteIndirectFastPath                         │
    → tt.load → ascend.indirect_load  ← 这里才真正转换  │
```

### 16.5 注意：scope.scope 和 indirect_load 是两个独立的概念

即使 op 被包进 `scope.scope {simt}`，要转成 `indirect_load` 仍需满足 structural 条件（非结构化 + size<64 + rank≤5）。不满足的 op 虽然在 scope 里（下游编译器会给它 SIMT 代码生成），但不会变成 `indirect_load` 指令。

反过来，在 `unstructured_in_simt` 模式下，即使 op 不在 scope 里，只要满足 structural 条件就能转成 `indirect_load`。

**总结**：`simd_simt` 和 `unstructured_in_simt` 用的是**同一个 pass** 做 `tt.load → indirect_load` 转换。区别只是 `simd_simt` 在前面插了 costmodel + Materialize 两步，把"允许走 SIMT 的 op 列表"从"所有满足条件的 op"收窄到"costmodel 批准的那些 op"。

---

## 十七、反推：作者是怎么写出这些公式和参数的

这些密密麻麻的条件、系数、门控不是凭空想出来的。从代码和 profile 文件的反推，作者的完整方法论是：

### 17.1 第一步：微基准测量——得到 op 级吞吐量

文件：`microbench/ascend_davidv100_v1.json`

作者写了一套微基准测试程序（文件里的 `source` 字段记录了来源：`triton_cases/SIMT_Test/tput.cce; concur2.cce`）。做法是：

1. 对 **每一个** op 类型（`f32.add`、`f32.mul`、`f32.div`、`load`、`store`……），写一个只包含该 op 的极简 kernel
2. 分别在 SIMD 和 SIMT 路径下编译
3. 在真实 A5 硬件上反复跑，读 `SYS_CNT` 硬件计数器（类似 GPU 的 clock cycles）
4. 记录饱和吞吐量（ops/cycle）

profile 里的 `confidence` 字段记录了数据来源的可信度：

```json
"source_kind": "isolated_microbenchmark",  // ← 实测
"source": "triton_cases/SIMT_Test/tput.cce",
"confidence": "high"

"source_kind": "architecture_fact",        // ← 硬件手册
"source": "Ascend vector ISA width",
"confidence": "high"
```

**每个 op 的 throughput 不是算出来的，是测出来的。** 你看到的那一长串 `simd_ops` / `simt_ops` 里的 `throughput` 值，都是作者在硬件上跑了微基准后填进去的。

### 17.2 第二步：第一原理建模——分析公式的骨架

有了 op 级吞吐量，作者从硬件架构知识出发，搭建了一个分析性成本模型：

1. **SIMD**：向量宽度 2048 bit → 一条指令处理 `2048/32 = 64` 个 f32。如果 op 操作了 500 个元素 → 需要 `ceil(500/64) = 8` 条向量指令。每条指令耗时 = `1/throughput` 周期。所以 SIMD cycles = `ceil(elements/64) / throughput`。

2. **SIMT**：warp size = 32。不需要向量打包，直接 `elements / throughput`。

3. **Memory**：SIMD 有独立 DMA 引擎（MTE2 load, MTE3 store），可以和计算并行 → `max(load, store)`。SIMT 的 load/store 共享总线 → `load + store`。

4. **Dot**：FLOPs = `2*M*N*K`，SIMD 有专用 Cube 单元（吞吐高），SIMT 没有（吞吐低）。

5. **Shuffle/Predicate**：reduction 在 SIMT 上需要 warp shuffle（log2(32)=5 级），mask 材料化需要 predicate 指令。

这些不是从数据里学的——是作者知道硬件怎么工作，直接写的物理模型。

### 17.3 第三步：硬件校准——修正分析模型的误差

分析模型是理想化的。真实硬件有 cache miss、bank conflict、指令调度气泡等效应。作者的解决方案：

1. 构造几个**有代表性的真实 kernel**（不是微基准那种单 op kernel，而是有 gather、dot、mask、循环的复杂 kernel）
2. 在真实硬件上用 `SYS_CNT` 硬件计数器测量实际耗时
3. 把 kernel 的 TTIR 喂给分析模型，算出**预测值**
4. 对比预测值与实测值的差距

profile 第 25 行的注释直接承认了：

```json
"source": "Bounded development activation fitted to repeated Ascend950PR
           Event runs for masked-rowwise and tiny-gather-dot structural
           domains; no workload-name matching; wider validation is still
           required before general rollout"
```

**"fitted to repeated Ascend950PR Event runs"** = 在 A5 上反复跑，采集 Event 计数器数据，拟合参数。

具体来说，他发现了以下规律：
- **不规则访存（irregular addressing）的 kernel**：预测值总是偏低（太乐观）→ 加一个与 `laneDependentPointerOps` 成正比的 penalty → `irregular_per_density = 0.8`（这个值是从数据里 fit 出来的）
- **有很多 mask 的 kernel**：SIMD 需要材料化 mask（额外的 DMA → 标量传输）→ 加 `per_mask_rank = 0.022` 的 penalty
- **有 reduction 的 kernel**：SIMT 上的 warp shuffle 开销比纯计算模型预估的大 → 加 `per_weighted_reduction = 0.02` 的 penalty
- **小矩阵乘（tiny dot）**：SIMD Cube 单元对于小矩阵利用率低（underfill）→ 单独建模 tiny dot case

每个 penalty 系数（0.8、0.022、0.02、0.06……）都是**从数据里 fit 出来的**，不是推理出来的。这就解释了为什么有那么多看似"魔法数字"的参数——它们是拟合结果。

### 17.4 第四步：校准域（Coverage）——诚实地划定模型的有效边界

作者没有声称模型在任何 kernel 上都准。profile 第 27 行的注释：

```json
"description": "Feature-domain limits within which Event-calibrated
                structural ranking is admitted; these are model validity
                bounds rather than hardware limits."
```

**"these are model validity bounds rather than hardware limits"** = 这些限制是模型的准确范围，不是硬件的能力限制。

具体来说：
- `tiny_dot_flops_max = 16384`：只在 dot FLOPs ≤ 16384 的 kernel 上验证过
- `rowwise_loop_trip_sum_max = 32`：只在循环总迭代数 ≤ 32 的 kernel 上验证过
- `minimum_irregular_density = 0.25`：只在 irregular density ≥ 0.25 的 kernel 上验证过

超出这些范围的 kernel → `calibrationCovered = false` → gate 不通过 → 退回 `all_simd`。这是非常诚实的工程实践：**在没测试过的域里不做决策。**

### 17.5 第五步：迭代——v1 到 v5

profile 版本号是 `v5`（`david-v100-simd-simt-20260728-v5`）。说明经过了 5 个版本的迭代。典型的迭代流程：

```
v1: 只有微基准数据 + 简单分析模型 → 在真实 kernel 上误差很大
v2: 加了 structural penalty → 对 masked-rowwise kernel 改善了
v3: 加了 tiny dot 分支 → 对小矩阵乘 kernel 改善了
v4: 加了 coverage + gate → 不在校准域内拒绝决策
v5: 调整了权重、更新了 profile 参数 → 当前版本
```

### 17.6 方法论总结

```
┌─ 阶段 1：微基准测量 ───────────────────────────────┐
│ 写 30+ 个单 op kernel → 分别 SIMD/SIMT 编译        │
│ → 在 A5 上跑 → 读 SYS_CNT → 得到每个 op 的吞吐    │
│ 产物：profile.simd_ops / simt_ops（186 个参数）    │
└────────────────────────────────────────────────────┘
                        │
┌─ 阶段 2：第一原理建模 ─────────────────────────────┐
│ 基于硬件架构知识写分析模型                          │
│ 产物：analyzeSimdSimtFeatures + estimateSimdSimtCandidates
│       （~600 行 C++）                              │
└────────────────────────────────────────────────────┘
                        │
┌─ 阶段 3：硬件校准 ─────────────────────────────────┐
│ 构造代表性 kernel → 实测 vs 预测 → 拟合 penalty    │
│ 产物：structural_penalty_ratio（16 个系数）         │
│       mixed_simd_fraction（8 个系数）              │
│ 这些系数不是推理的，是数据 fit 的                  │
└────────────────────────────────────────────────────┘
                        │
┌─ 阶段 4：校准域划定 ───────────────────────────────┐
│ 记录模型在哪些 kernel 模式上验证过                  │
│ 产物：coverage 域（11 个阈值）                     │
│ 超出域 → 拒绝决策                                   │
└────────────────────────────────────────────────────┘
                        │
┌─ 阶段 5：迭代 v1→v5 ──────────────────────────────┐
│ 每次迭代：扩展校准域 / 修正参数 / 添加新 pattern   │
│ 产物：profile_version = v5                        │
└────────────────────────────────────────────────────┘
```

所以回答你的问题："作者怎么知道这些条件会更好？" **他不知道，他测出来的。** op 级吞吐是微基准测的，penalty 系数是在真实 kernel 上对比预测值和实测值后拟合的，coverage 是他诚实地记录了"在这些范围之外我没测过所以不保证"。

这也是为什么 profile 是 JSON 而不是硬编码在 C++ 里——作者知道参数需要频繁迭代更新，放在 JSON 里可以不用重编译 C++ 就能调参。

---

## 十八、Microbenchmark 详细拆解：25 个指标测了什么、怎么用

文件：`backend/costmodel_profiles/microbench/ascend_davidv100_v1.json`

25 个指标，每个都有 `source` 字段记录了测量来源（如 `triton_cases/SIMT_Test/tput.cce`），`source_kind` 区分了实测（`isolated_microbenchmark`）还是硬件参数（`architecture_fact`/`device_configuration`）。

### 18.1 指标分类总览

```
时钟/架构常量 (4个)     → 把 cycle 数变成有物理意义的单位
SIMT 启动开销 (2个)     → 算 setup cost
混合模式过渡 (6个)      → 算 mixed 的切换代价
SIMT 访存 (8个)         → 算 load/store 的 cycle（GM + UB）
SIMT 计算+通信 (3个)    → 算 compute + shuffle 的 cycle
SIMD 计算基准 (2个)     → 算 SIMD compute 的基准
```

### 18.2 第一类：时钟与架构常量（4 个）

| 指标 | 值 | source_kind | 为什么需要 |
|------|-----|-------------|-----------|
| `clock.sys_cnt.frequency_mhz` | 988.9 MHz | isolated_microbenchmark | 所有 costmodel 输出以 SYS_CNT cycles 为单位，这个值用于把 cycles 换算成 wall-clock us |
| `clock.device_compute.frequency_mhz` | 1650 MHz | device_configuration | 将来做 device_compute ↔ SYS_CNT 互转（当前未用到） |
| `simd.vector_width_bits` | 2048 bit | architecture_fact | **SIMD 公式核心**：`vectorWidth = 2048/32 = 64` 元素/指令。决定 `ceil(elements/64) / throughput` 的分母 |
| `simt.warp_size` | 32 lane | architecture_fact | **SIMT 公式核心**：`warpInstructions = ceil(elements/32)`、`shuffleLevels = log2(32) = 5` |

测量方法：
- `clock.sys_cnt.frequency_mhz`：写一个 kernel 跑不同时长，读 SYS_CNT 计数器差值，线性回归斜率。来源：`triton_cases/SIMT_Test SYS_CNT slope calibration`
- `simd.vector_width_bits` 和 `simt.warp_size`：直接来自硬件 ISA 手册，不需要测量

### 18.3 第二类：SIMT 启动开销（2 个）

| 指标 | 值 | 怎么测 | 为什么需要 |
|------|-----|--------|-----------|
| `simt.setup.empty` | 115 cycles | 写一个**空 launch** 的 SIMT kernel（body 是 no-op），跑 SYS_CNT。来源 `meas.cce` | 得到 SIMT 纯启动开销。C++ 里作为 `simtSetupCycles`——评分公式中 `allSimtOnly` 的固定部分 |
| `simt.setup.empty_with_barrier` | 141 cycles | 同上，但 scope 外加了 barrier 指令 | 带上同步开销的版本，用于更保守的估计 |

在 C++ 评分公式中的使用（第 1902-1904 行）：

```cpp
simtAnalyticalCycles = profile.simtSetupCycles      // ← 115 cycles
                     + simtIssuePayloadCycles * profile.programIssueScale;
//                     ↑ 计算 + 访存 + shuffle + predicate 的加权和
```

### 18.4 第三类：混合模式过渡开销（6 个）

| 指标 | 值 | 怎么测 | 为什么需要 |
|------|-----|--------|-----------|
| `mixed.empty_simt_setup.warps_1` | 182 cycles | `simt_transition_microbench_tail16_barrier_20260713.txt` | **混合模式核心参数**：SIMD 核启动 SIMT region 需要上下文切换 |
| `mixed.empty_simt_setup.warps_2` | 182 cycles | 同上 | ↑ |
| `mixed.empty_simt_setup.warps_4` | 182 cycles | 同上 | ↑ |
| `mixed.empty_simt_setup.warps_8` | 182 cycles | 同上 | ↑ |
| `mixed.empty_simt_setup.warps_16` | 182 cycles | 同上 | ↑ |
| `mixed.empty_simt_setup.warps_32` | **223** cycles | 同上 | 32 warps 时跳到 223——可能触及了 warp scheduler 的扩展代价 |

这些值直接进入 `candidateCosts.mixedSimdSimt` 计算（C++ 第 1957-1972 行）：

```cpp
// 找到最接近当前 numWarps 的 transition profile
nearestTransition = argmin(|numWarps - transition.numWarps|);

// mixed 的 setup = 对应 warp 数的空 SIMT region setup
mixedSetupCycles = nearestTransition->emptySimtSetupCycles;  // 如 32 warps → 223

// transitionDelta = 混合比纯 SIMT 多出来的 setup 开销
transitionDelta = max(0, mixedSetupCycles - standaloneSimtSetupCycles);
//              = max(0, 223 - 115) = 108 cycles
```

### 18.5 第四类：SIMT 访存（8 个）—— Global Memory + Unified Buffer

| 子类 | 指标 | 值 | 怎么测 | 为什么需要 |
|------|------|-----|--------|-----------|
| **GM load** | `simt.gm.load.throughput` | 0.1731 warp_insn/cycle | `simt_gm_memory_david_v100_20260725.csv`：32 warps 交替执行，128 MiB 数据集，每个线程 8 个独立 load | **SIMT 内存公式**：`simtLoadCycles = loadWarpInstructions / 0.1731` |
| | `simt.gm.load.bandwidth` | 22.15 B/cycle | 同上 | 将来做 bytes→cycles 直转用（当前未用） |
| **GM store** | `simt.gm.store.throughput` | 0.1290 warp_insn/cycle | 同上 | `simtStoreCycles = storeWarpInstructions / 0.1290` |
| | `simt.gm.store.bandwidth` | 16.51 B/cycle | 同上 | 同上 |
| **UB load** | `simt.ub.load.throughput` | 0.5073 warp_insn/cycle | `simt_memory_david_v100_20260725.csv`：16 warps，128 KiB，4 组轮换 | UB 比 GM 快 3×（0.5073 vs 0.1731），留下做更精细的存访建模 |
| | `simt.ub.load.bandwidth` | 64.93 B/cycle | 同上 | 同上 |
| **UB store** | `simt.ub.store.throughput` | 0.5301 warp_insn/cycle | 同上 | 同上 |
| | `simt.ub.store.bandwidth` | 67.85 B/cycle | 同上 | 同上 |

**GM vs UB 的区别**：
- **GM**（Global Memory）= HBM，吞吐 0.17 warp_insn/cycle → 是瓶颈
- **UB**（Unified Buffer）= 片上 SRAM，吞吐 0.5 warp_insn/cycle → 快 3×

当前 C++ 代码只用 GM 的 throughput 值，因为特征提取阶段还不知道哪些数据会在 UB 里。UB 数据是给将来扩展留的。

GM load/store 在 C++ 中的使用（第 1844-1851 行）：

```cpp
// SIMT memory = load + store（共享总线，串行）
simtLoadCycles  = loadWarpInstructions  / 0.1731;  // GM load throughput
simtStoreCycles = storeWarpInstructions / 0.1290;  // GM store throughput
simtMemoryCycles = simtLoadCycles + simtStoreCycles;

// 对比 SIMD memory = max(load, store)（独立 DMA，并行）
simdMemoryCycles = max(simdLoadCycles, simdStoreCycles);
```

### 18.6 第五类：SIMT 计算与通信（3 个）+ SIMD 计算基准（2 个）

| 指标 | 值 | 怎么测 | 为什么需要 |
|------|-----|--------|-----------|
| `simt.f32.add.throughput` | **141.0** scalar_op/cycle | `tput.cce; concur2.cce`：写一个全是 f32 add 的 kernel，11 warps 饱和执行，读 SYS_CNT | **SIMT 计算基准**：其他所有 op（mul/div/exp/log...）的成本都相对于 add。如 `simtOps["mul"].factor = 1.2` → mul 成本是 add 的 1.2 倍 |
| `simt.shuffle.throughput` | **0.8172** warp_insn/cycle | `simt_shuffle_david_v100_20260725.csv`：32 warps，4 条独立 ILP 链，测 shfl-up 吞吐 | **reduction 在 SIMT 上的成本**：warp 内 reduce 需要通过 shuffle 指令交换数据 |
| `simt.shuffle.dependent_latency` | 27.27 cycles | 同上，但只用 1 warp + 依赖链 | shuffle 的单次延迟。当前未用，留作精细建模 |
| `simd.f32.add.throughput` | 3.30 vector_insn/cycle | 同源 | 每条 SIMD 向量指令耗时 `1/3.30 = 0.303` SYS_CNT cycles。类似地作为 SIMD 其他 op 的基准 |
| `simd.f32.add.dependent_latency` | 1.818 cycles | 同源 | SIMD add 依赖链延迟。当前未用 |

**为什么只测了 add，mul/div/exp 等不需要测？**

因为在 profile JSON 里，其他 op 是用**相对于 add 的 factor**表达的：

```json
"simt.f32.add": {"throughput": 141.0, "factor": 1.0},   // 基准
"simt.f32.mul": {"throughput": 141.0, "factor": 1.0},   // mul = 1× add
"simt.f32.div": {"throughput": 141.0, "factor": 1.7},   // div = 1.7× add（除法贵 70%）
"simt.f32.exp": {"throughput": 141.0, "factor": 8.0},   // exp = 8× add（指数非常贵）
```

在 C++ 评分公式中（第 1810-1811 行）：

```cpp
// SIMT 某个 op 的 cycles = elements / throughput * factor
simtCycles = elements / 141.0 * factor;  // 如 mul: factor=1.0, exp: factor=8.0
```

### 18.7 Shuffle 的使用细节

在 C++ 中，shuffle 只用于 SIMT 的 reduction 和 scan（第 1857-1866 行）：

```cpp
// shuffle 层级数 = log2(warp_size) = log2(32) = 5
const int64_t shuffleLevels = ceil(log2(32));

// shuffle 指令总数 = reductions × ceil(maxNumel/warp_size) × shuffleLevels
//   例如：32 reductions，maxNumel=256，warp=32 → 32 × 8 × 5 = 1280 条 shuffle 指令
simtShuffleInstructions =
    (weightedReductions + weightedScans) *
    ceil(maxNumel / 32) * 5;

// shuffle cycles = 1280 / 0.8172 ≈ 1566 cycles
simtShuffleCycles = simtShuffleInstructions / 0.8172;

// shuffle 被加到 SIMT payload 的计算中：
simtIssuePayloadCycles =
    max(simtCompute + simtShuffle + simtDot, simtMemory) + simtPredicate;
```

### 18.8 全部指标在评分公式中的流向图

```
                    ┌──────────────────────────────────────────────────┐
                    │              estimateSimdSimtCandidates          │
                    │                                                 │
simd.vector_width   │  simdCompute = Σ ceil(elems/64) /                │
  (2048 bit)        │               profile.simdOps[op].throughput     │
                    │               * profile.simdOps[op].factor       │
                    │                                                 │
simt.warp_size (32) │  simtCompute = Σ elems /                         │
                    │               profile.simtOps[op].throughput     │
                    │               * profile.simtOps[op].factor       │
                    │                                                 │
simt.gm.load        │  simtLoad   = loadWarpInsns / 0.1731            │
  (0.1731)          │  simtStore  = storeWarpInsns / 0.1290           │
simt.gm.store       │  simtMemory = simtLoad + simtStore               │
  (0.1290)          │                                                 │
                    │  simdMemory = max(simdLoad, simdStore)            │
                    │        (DMA 并行，按字节算，非 warp 指令)         │
                    │                                                 │
simt.shuffle        │  shuffleCycles =                                │
  (0.8172)          │    (reductions * maxNumel/32 * log2(32))        │
                    │    / 0.8172                                     │
                    │                                                 │
simt.setup.empty    │  simtAnalytical =                               │
  (115 cycles)      │    115 + max(compute+shuffle+dot, memory)       │
                    │          * programIssueScale                     │
                    │                                                 │
mixed.empty_simt    │  mixedSetup = nearestTransition                  │
  _setup.warps_*    │    (如 32 warps → 223 cycles)                    │
  (182-223 cycles)  │                                                 │
                    │  mixedCost = mixedSetup +                       │
                    │    simtPayload + blend *                         │
                    │    (simdPayload - simtPayload)                   │
                    │                                                 │
clock.sys_cnt       │  wall_time_us = selected_cycles / 988.9          │
  (988.9 MHz)       │    (costmodel 输出的是 selection score,          │
                    │     不是 wall time; 这个换算仅供参考)            │
                    └──────────────────────────────────────────────────┘
```

### 18.9 关键洞察：为什么 profile 是这样的结构

1. **微基准和候选模型分离**：`microbench/ascend_davidv100_v1.json` 存的是**模型无关的硬件实测数据**（op 吞吐、内存带宽、setup 开销），`david_v100_simd_simt_v1.json` 存的是**模型特定的参数**（penalty 系数、calibration 域、门控参数）。两者分开意味着同一个人 / 不同团队可以更新微基准数据（换芯片？重新测）而不影响模型公式，反之亦然。

2. **每个测量都有 provenance**：`source` 字段记录了"这个值是什么文件/命令/配置测出来的"。这是科学实验的 reproducibility 标准——别人可以复现，可以质疑。

3. **confidence 不是装饰**：`high`/`medium`/`low` 会实际影响 gate。如果一个指标是 `low` confidence，依赖它的候选 score 的 overall confidence 会被拉低，可能导致 gate 不通过——这是一种自动的"数据质量不够就别用"机制。

4. **当前只用了部分数据**：UB load/store throughput、shuffle latency、compute clock 等都还没用上。说明作者留了扩展空间——当模型需要更精细的内存层次建模时（比如区分 UB-resident vs GM-resident 的数据），这些数据已经就位。
