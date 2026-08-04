# Autoscope 方法二（时序模拟）架构设计

## 一、总体架构

```
Python 层
├─ compiler.py: _run_cpp_simd_simt_costmodel()     ← 不改，调度不变
└─ compiler.py: ttir_to_linalg()                    ← 不改，读 effective

C++ 层（5 个 Pass，1 个共享库）
├─ Pass 1: AnalyzeFeaturesPass           ←【复用】simt_costmodel 分支已有
├─ Pass 2: SimdTimingPass                ←【改造】PipelineAnalysisPass → A5
├─ Pass 3: SimtTimingPass                ←【新建】SIMT 时序模拟器
├─ Pass 4: MixedTimingPass               ←【新建】混合模式模拟器
└─ Pass 5: SelectExecutionPass           ←【改造】SelectSimdSimtCostModel 后半段

共享库
├─ SimtTimingModel.h / .cpp              ←【新建】SIMT 硬件模型（可被 Pass 3/4 共用）
├─ MixedAnchorSelector.h / .cpp          ←【改造】现有的 isMixedSimtAnchor 逻辑
└─ HardwareConfig.h / .cpp               ←【改造】加 A5 参数 + SIMT 参数
```

---

## 二、核心 Class 定义

### 2.1 硬件配置（HardwareConfig — 改造现有）

```cpp
// 文件: include/AscendModel/HardwareConfig.h

// ── 新增: SIMT 硬件参数 ──
struct SimtHardwareParams {
  int64_t warpSize = 32;             // 每 warp 的 lane 数
  int64_t maxWarpsPerSM = 32;        // 每 SM 最多同时 active warps
  int64_t sharedMemBytes = 65536;    // shared memory 大小
  int64_t numSMs = 20;               // SM 数量（对应 AIV 数？需确认 A5 架构）

  // 指令吞吐 (ops per cycle per SM)
  double computeThroughput = 141.0;  // f32 scalar op throughput（来自微基准）
  double gmLoadRate = 0.1731;        // warp_insn/cycle（来自微基准）
  double gmStoreRate = 0.1290;       // warp_insn/cycle
  double shuffleRate = 0.8172;       // warp_insn/cycle (来自微基准)
  double predicateRate = 16.0;       // 假设值, 需实测

  // 延迟 (cycles)
  double setupCycles = 115.0;        // 空 launch 开销（来自微基准）
  double gmLatency = 300.0;          // GM 访问延迟（需确认）
  double shuffleLatency = 27.3;      // shuffle 依赖链延迟（来自微基准）
};

// ── 已有: A5 SIMD 硬件参数 ──
// 只需要更新 numAIVCores, clockFrequency, MTE 带宽等数值
// 结构不变
```

### 2.2 SIMT 指令表示（新建）

```cpp
// 文件: include/AscendModel/Analysis/SimtTimingModel.h

enum class SimtOpKind {
  Compute,      // arith.addf, math.exp, etc.
  GlobalLoad,   // tt.load (assume GM resident)
  GlobalStore,  // tt.store
  Shuffle,      // warp-level shuffle (for reduction)
  Predicate,    // mask materialization
  Barrier,      // sync point
};

struct SimtInstruction {
  SimtOpKind kind;
  int64_t warpCount;      // 需要多少条 warp 指令
  double elements;        // 操作的元素数（用于带宽计算）
  double bytes;           // 传输字节数
  Operation *mlirOp;      // 来源 MLIR op
  int64_t loopMultiplier; // 循环乘数
};
```

### 2.3 SIMT 时序模拟器（新建）

```cpp
// 文件: include/AscendModel/Analysis/SimtTimingModel.h

class SimtTimingSimulator {
public:
  SimtTimingSimulator(const SimtHardwareParams &params, int64_t numWarps);

  /// 从 TTIR 的 Operation 构建指令列表
  void buildInstructions(ModuleOp module);

  /// 执行时序模拟，返回总 cycles
  /// 当前版本用简化模型：analytical（非 cycle-accurate）
  int64_t simulate();

  /// 类似 SIMD 的 wave 串行化
  int64_t getKernelCycles(int64_t numPrograms, int64_t numSMs,
                          int64_t numInnerIters = 0) const;

  // ── 结果访问 ──
  int64_t getComputeCycles() const;
  int64_t getMemoryCycles() const;
  int64_t getShuffleCycles() const;
  int64_t getPredicateCycles() const;

private:
  const SimtHardwareParams &params_;
  int64_t numWarps_;
  std::vector<SimtInstruction> instructions_;

  // 累计结果
  int64_t computeCycles_ = 0;
  int64_t memoryCycles_ = 0;
  int64_t shuffleCycles_ = 0;
  int64_t predicateCycles_ = 0;

  /// 分类一个 MLIR op 到 SimtOpKind（按名字字符串分类）
  static SimtOpKind classifyOp(llvm::StringRef opName);

  /// 计算单条指令的 cycle 数
  int64_t estimateInstructionCycles(const SimtInstruction &inst);
};
```

`simulate()` 的实现逻辑（首个版本，analytical model）：

```cpp
int64_t SimtTimingSimulator::simulate() {
  double totalWarpInsns = 0;
  double totalBytes = 0;

  for (auto &inst : instructions_) {
    int64_t weighted = inst.warpCount * inst.loopMultiplier;
    switch (inst.kind) {
    case SimtOpKind::Compute:
      totalWarpInsns += weighted;
      computeCycles_ = weighted / params_.computeThroughput;
      break;
    case SimtOpKind::GlobalLoad:
      totalWarpInsns += weighted;
      totalBytes += inst.bytes * inst.loopMultiplier;
      memoryCycles_ += weighted / params_.gmLoadRate;
      break;
    case SimtOpKind::GlobalStore:
      totalWarpInsns += weighted;
      memoryCycles_ += weighted / params_.gmStoreRate;
      break;
    case SimtOpKind::Shuffle:
      shuffleCycles_ += weighted / params_.shuffleRate;
      break;
    case SimtOpKind::Predicate:
      predicateCycles_ += weighted / params_.predicateRate;
      break;
    }
  }

  // SIMT: compute/shuffle 和 memory 可以部分 overlap
  // 但 warp scheduler 不能同时做 compute 和 memory access
  int64_t computeTotal = computeCycles_ + shuffleCycles_;
  int64_t memoryTotal = memoryCycles_;

  // 简化假设: 取 compute 和 memory 的最大值（它们可以在不同 warp 上 overlap）
  // 加 predicate（必须串行）
  return std::max(computeTotal, memoryTotal) + predicateCycles_;
}
```

### 2.4 混合 Anchor 选择器（改造现有）

```cpp
// 文件: include/Utils/SimtSelection.h（改造现有）

class MixedAnchorSelector {
public:
  /// 从 module 中收集所有应该标记 SIMT 的 op
  /// 规则同现有 isMixedSimtAnchor()
  std::vector<Operation *> selectAnchors(ModuleOp module,
                                          bool compileOn91095);

  /// 分裂 module 中的 op 为两个列表
  struct AnchorSplit {
    std::vector<Operation *> simtAnchors;   // 走 SIMT 的 ops
    std::vector<Operation *> simdOps;        // 走 SIMD 的 ops
  };
  AnchorSplit splitOps(ModuleOp module, bool compileOn91095);

private:
  /// 现有规则（不改）:
  ///   tt.gather / tt.histogram → SIMT
  ///   tt.scan (1D cumsum) → SIMT
  ///   tt.atomic_* (tensor ptr) → SIMT
  ///   tt.load / tt.store (pointer depends on loaded index) → SIMT
  bool isMixedSimtAnchor(Operation *op, bool compileOn91095);
};
```

### 2.5 混合模式模拟器（新建）

```cpp
// 文件: include/AscendModel/Analysis/MixedTimingModel.h

class MixedTimingSimulator {
public:
  MixedTimingSimulator(const HardwareConfig &simdConfig,
                       const SimtHardwareParams &simtParams,
                       int64_t numWarps);

  /// 初始化：输入 SIMD 和 SIMT 的 op 列表
  void initialize(const std::vector<Operation *> &simdOps,
                  const std::vector<Operation *> &simtOps,
                  ModuleOp module);

  /// 模拟混合执行
  /// 方法 5.2: SIMD ops 用 PipelineScheduler 算，SIMT ops 用 SimtTimingSimulator 算
  /// 总 cycle = max(simd_payload, simt_payload) + transition_cost
  int64_t simulate();

  int64_t getTransitionCost() const { return transitionCost_; }

private:
  // SIMD 侧
  std::unique_ptr<PipelineScheduler> simdScheduler_;    // 复用现有

  // SIMT 侧
  std::unique_ptr<SimtTimingSimulator> simtSimulator_;

  // 过渡开销（来自微基准 mixed.empty_simt_setup.warps_*）
  int64_t transitionCost_ = 0;

  /// 计算数据搬运开销：SIMT region 的输入来自 SIMD 侧，需要 DMA
  int64_t estimateDataTransferCycles();
};
```

---

## 三、Pass 定义

### 3.1 Pass 注册（Passes.td — 改造现有）

```tablegen
// 文件: include/AscendModel/Transforms/Passes.td

// ── 已有，不改 ──
def PipelineAnalysisPass : Pass<"pipeline-analysis", "ModuleOp"> { ... }

// ── 已有，需要加 SIMT 参数 ──
def SelectSimdSimtCostModelPass : Pass<"select-simd-simt-costmodel", "ModuleOp"> {
  // 不改结构，只是内部逻辑改为调三个模拟器
  let options = [
    Option<"mode", "mode", std::string, "auto", "auto|report">,
    Option<"profilePath", "profile-path", std::string, "", "hardware config JSON">,
    Option<"actualTarget", "target", std::string, "", "chip target">,
    Option<"numWarps", "num-warps", int64_t, 32, "SIMT warp count">,
    Option<"marginRatio", "margin", double, 0.10, "decision margin">,
    Option<"compileOn91095", "compile-91095", bool, false, "is A5?">,
    Option<"dumpPath", "dump-path", std::string, "", "report output">,
  ];
}

// ── 新建 ──
def AnalyzeFeaturesPass : Pass<"analyze-features", "ModuleOp"> {
  let summary = "Extract SimdSimtFeatureSummary from TTIR";
}
def SimdTimingPass : Pass<"simd-timing", "ModuleOp"> {
  let summary = "Run SIMD pipeline timing simulation";
}
def SimtTimingPass : Pass<"simt-timing", "ModuleOp"> {
  let summary = "Run SIMT warp timing simulation";
}
def MixedTimingPass : Pass<"mixed-timing", "ModuleOp"> {
  let summary = "Run mixed SIMD+SIMT timing simulation";
}
```

### 3.2 各 Pass 的输入输出

| Pass | 输入 | 输出（写 ModuleOp attr） |
|------|------|------------------------|
| `AnalyzeFeaturesPass` | ModuleOp | `ascend.feature_summary_json` |
| `SimdTimingPass` | ModuleOp | `ascend.simd_cycles` |
| `SimtTimingPass` | ModuleOp | `ascend.simt_cycles` |
| `MixedTimingPass` | ModuleOp | `ascend.mixed_cycles` |
| `SelectSimdSimtCostModelPass` | 上面四个 Pass 的 attrs | `ascend.simt_costmodel.effective`, `ascend.simt_costmodel.report_json`, `ascend.simt_costmodel.selected`（逐 op） |

### 3.3 SelectSimdSimtCostModelPass::runOnOperation() 的改造版

```cpp
void SelectSimdSimtCostModelPass::runOnOperation() override {
  ModuleOp module = getOperation();

  // ── Phase 1: 特征提取 ──
  auto features = analyzeSimdSimtFeatures(module);   // ← 复用现有函数
  // 存到 module attr（给后续 Pass 用）
  module->setAttr("ascend.feature_summary_json",
                  builder.getStringAttr(json(features)));

  // ── Phase 2: 三个模拟 ──
  HardwareConfig simdConfig = loadHardwareConfig("ascend_davidv100.json");
  SimtHardwareParams simtParams = loadSimtParams("microbench/ascend_davidv100_v1.json");

  // 2a: SIMD
  PipelineScheduler simdScheduler(&simdConfig);
  // ... 遍历 module.walk() 填充 simdScheduler ...
  simdScheduler.schedule();
  int64_t simdCycles = simdScheduler.getKernelCycles(numPrograms, numParallelUnits);

  // 2b: SIMT
  SimtTimingSimulator simtSim(simtParams, numWarps);
  simtSim.buildInstructions(module);
  int64_t simtCycles = simtSim.simulate();

  // 2c: Mixed
  MixedAnchorSelector selector;
  auto [simtOps, simdOps] = selector.splitOps(module, compileOn91095);
  MixedTimingSimulator mixedSim(simdConfig, simtParams, numWarps);
  mixedSim.initialize(simdOps, simtOps, module);
  int64_t mixedCycles = mixedSim.simulate();

  // ── Phase 3: 决策（逻辑同上，用模拟值替代 profile 公式值）──
  SimdSimtCandidateScores scores;
  scores.allSimd = simdCycles;
  scores.allSimtOnly = simtCycles;
  scores.mixedSimdSimt = mixedCycles;

  auto decision = chooseBest(scores);

  // ── Phase 4: Gate（简化为只有 margin gate）──
  double gain = scores.allSimd - scores.get(decision);
  double requiredGain = std::max(64.0, scores.allSimd * marginRatio);
  bool gatePassed = decision != SimdSimtCandidateKind::AllSIMD
                    ? gain > requiredGain
                    : true;  // 选 all_simd 一定通过

  // ── Phase 5: 写 effective + 标记 op ──
  std::string effective = gatePassed
      ? stringifySimdSimtCandidate(decision)
      : "all_simd";

  module->setAttr("ascend.simt_costmodel.effective", effective);
  module->setAttr("ascend.simt_costmodel.report_json", buildReportJSON(...));

  if (effective == "mixed_simd_simt") {
    for (Operation *anchor : simtOps)
      anchor->setAttr("ascend.simt_costmodel.selected",
                      UnitAttr::get(module.getContext()));
  }
}
```

---

## 四、完整调用链

### 4.1 Python → C++ → Python 链路

```
用户: kernel[grid](args, compile_mode="simd_simt")
  │
  ▼
JITFunction.run()                           ← jit.py:695
  │
  ├─ _pack_args() → NPUOptions               ← jit.py:717
  │    compile_mode="simd_simt"
  │    → __post_init__ → auto_simt_scope_mode=auto（默认）
  │
  ├─ _do_compile() → triton.compile()        ← jit.py:720
  │    → metadata = {**options.__dict__, ...} ← compiler.py:284
  │
  └─ add_stages()
       │
       ├─ make_ttir(mod, metadata, opt)      ← compiler.py:118
       │
       └─ ttir_to_linalg(mod, metadata, opt) ← compiler.py:182
            │
            ├─ _run_cpp_simd_simt_costmodel(mod, metadata, opt)
            │    │                              ← compiler.py:141
            │    ├─ mode != "off" && compile_mode == "simd_simt"?
            │    │   是 → 继续
            │    │   否 → return "backend_default"
            │    │
            │    ├─ pm.addPass(AnalyzeFeaturesPass)         ← Pass 1
            │    ├─ pm.addPass(SimdTimingPass)              ← Pass 2
            │    ├─ pm.addPass(SimtTimingPass)              ← Pass 3
            │    ├─ pm.addPass(MixedTimingPass)             ← Pass 4
            │    ├─ pm.addPass(SelectSimdSimtCostModelPass) ← Pass 5
            │    ├─ pm.addPass(MaterializeSimtScopes)       ← Pass 6（不改）
            │    └─ pm.run(mod)
            │         │
            │         ├─ Pass 1: 遍历 module → features → attr
            │         ├─ Pass 2: 读 features → PipelineScheduler → cycles
            │         ├─ Pass 3: 读 features → SimtTimingSimulator → cycles
            │         ├─ Pass 4: split ops → MixedTimingSimulator → cycles
            │         ├─ Pass 5: 读 2/3/4 的 cycles → chooseBest → gate
            │         │          → 写 effective + selected 标记
            │         └─ Pass 6: 读 selected → 包 scope.scope{simt}
            │
            ├─ 读 effective = get_attr(mod, "ascend.simt_costmodel.effective")
            │
            └─ 根据 effective 决定后续编译路径
                 ├─ "all_simd" → 正常 SIMD 链路
                 ├─ "all_simt_only" → force_simt_only → RowCoalescing → npubin
                 └─ "mixed_simd_simt" → 结合 scope.scope 走混合编译
```

### 4.2 Pass 间数据流（ModuleOp attr 通信）

```
AnalyzeFeaturesPass
  │
  │  输出: ascend.feature_summary_json
  │        ascend.feature_load_ops
  │        ascend.feature_dot_ops
  │        ascend.feature_gather_ops
  │        ...
  ▼
SimdTimingPass
  │  输入: ModuleOp（直接 walk）
  │  输出: ascend.simd_cycles
  │        ascend.simd_breakdown_json
  ▼
SimtTimingPass
  │  输入: ModuleOp（直接 walk）
  │  输出: ascend.simt_cycles
  │        ascend.simt_breakdown_json
  ▼
MixedTimingPass
  │  输入: ModuleOp（直接 walk + MixedAnchorSelector）
  │  输出: ascend.mixed_cycles
  │        ascend.mixed_breakdown_json
  │        ascend.mixed_simt_anchor_count
  ▼
SelectSimdSimtCostModelPass
  │  输入: ascend.simd_cycles
  │        ascend.simt_cycles
  │        ascend.mixed_cycles
  │  输出: ascend.simt_costmodel.effective
  │        ascend.simt_costmodel.report_json
  │        ascend.simt_costmodel.selected（逐 op attr）
  ▼
MaterializeSimtScopes
  │  输入: ascend.simt_costmodel.selected（逐 op attr）
  │  输出: scope.scope {simt}（MLIR IR 改写）
```

---

## 五、任务拆解

### 可并行分配的任务

```
任务 A: A5 SIMD 硬件参数 + PipelineAnalysisPass 移植
  ├─ 1. 从硬件文档获取 A5 的 AIV core 数、时钟频率、MTE 带宽
  ├─ 2. 更新 HardwareConfig A5 profile JSON
  ├─ 3. 验证：拿 mojo_opset gelu/silu 跑，对比真实 cycle vs 模拟
  └─ 产物: ascend_davidv100.json + SimdTimingPass

任务 B: SIMT 硬件参数收集 + SimtTimingSimulator
  ├─ 1. 从微基准/硬件文档获取 warp 参数、指令延迟
  ├─ 2. 实现 SimtHardwareParams 结构体
  ├─ 3. 实现 SimtTimingSimulator（先用 analytical model，非 cycle-accurate）
  ├─ 4. 验证：拿 gather kernel 跑，对比 SIMT 真实耗时 vs 模拟
  └─ 产物: SimtHardwareParams + SimtTimingSimulator

任务 C: MixedAnchorSelector + MixedTimingSimulator
  ├─ 1. 从 simt_costmodel 分支移植 isMixedSimtAnchor 规则
  ├─ 2. 实现 MixedTimingSimulator（先做方法 5.2）
  ├─ 3. 验证：拿 moe kernel（gather + dot）跑混合模拟
  └─ 产物: MixedAnchorSelector + MixedTimingSimulator

任务 D: 集成 + 评估框架
  ├─ 1. 改造 SelectSimdSimtCostModelPass::runOnOperation
  ├─ 2. Python 侧加环境变量 TRITON_ASCEND_AUTO_SIMT_SCOPE=auto
  ├─ 3. 写 A/B 对比脚本（autoscope 选 vs 默认）
  ├─ 4. 跑 mojo_opset 全部算子，出评估报告
  └─ 产物: 集成代码 + 评估脚本 + 报告
```

### 依赖关系

```
任务 A (SIMD) ───── 无依赖，先开始
任务 B (SIMT) ───── 无依赖，和 A 并行
任务 C (Mixed) ──── 依赖 A 和 B 的接口定义（不需要完成，只需接口稳定）
                   实际开始：等 A 的 PipelineScheduler 接口和 B 的
                   SimtTimingSimulator 接口定下来后即可
任务 D (集成) ──── 依赖 A、B、C 的代码完成
```

### 第一周里程碑

| 人 | Day 1-2 | Day 3-4 | Day 5 |
|----|---------|---------|-------|
| A (SIMD) | 收集 A5 硬件参数 | 更新 HardwareConfig JSON | 移植 PipelineAnalysisPass |
| B (SIMT) | 收集 SIMT 参数 | 实现 SimtHardwareParams | 实现 analytical SimtTimingSimulator |
| C (Mixed) | 移植 isMixedSimtAnchor | 定义 MixedTimingSimulator 接口 | 实现方法 5.2 初版 |
| D (集成) | 搭建评估框架 | 准备 mojo_opset 测试数据 | 跑 benchmark 采集基准数据 |

### 第二周里程碑

| 人 | Day 6-7 | Day 8-9 | Day 10 |
|----|---------|---------|--------|
| A | 验证 SIMD 精度 | 修正误差 | 稳定接口 |
| B | 验证 SIMT 精度 | 修正误差 | 稳定接口 |
| C | 端到端 mixed 测试 | 修正误差 | 稳定接口 |
| D | 集成 A+B+C → SelectExecutionPass | 跑全部算子 | 出评估报告 |

---

## 六、保守化策略

两周内不可能做到高精度。以下策略确保"宁可不推荐 SIMT，也不错误推荐"：

1. **SIMD 估计偏低（乐观）**：`simdCycles = PipelineScheduler 输出 × 0.8`（给 SIMD 20% 的 benefit of doubt）
2. **SIMT 估计偏高（悲观）**：`simtCycles = SimtTimingSimulator 输出 × 1.5`
3. **marginRatio 设高**：30%（不是 10%）
4. **Gate 收紧**：只在 gather/indirect-load 占比 > 30% + SIMT 估计仍快 > 30% 时才推荐
5. **默认退回**：任何不确定的情况 → `effective = "all_simd"`

这些策略不写在 C++ 里，而是作为可配置参数（JSON 或环境变量），方便调参。
