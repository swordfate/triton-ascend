# HIVMAnalysis 深度解读：从 NPUIR 到精确性能建模

## 0. 为什么需要 HIVMAnalysis？

在 costmodel-overview.md 的第 5 节，我们讲了 PipelineScheduler——它工作在 TTIR 转换后的 AscendModel dialect 上，对每个 Op 按 `HWUnit` 做流水线调度，然后用固定公式计算总 latency：

```cpp
// PipelineAnalysis.cpp:238-250
int64_t barrierCycles = numInnerIters * hwConfig->getPipeBarrierCyclesPerIter(); // 7500
double scalarFactor = hwConfig->getAIVScalarOverheadFactor();                   // 3.74
perProgramCycles = (totalCycles + barrierCycles) * (1.0 + scalarFactor);
numWaves = ceil(numPrograms / numParallelUnits);
return perProgramCycles * numWaves;
```

**这个公式的问题**非常明确：

1. **barrier = 7500 × numInnerIters**。不管 kernel 的同步结构多简单或多复杂，每个内层迭代都按 7500 cycles 算 barrier。但实际 NPUIR 里有 `set_flag`(1 cycle)、`wait_flag`(2 cycles)、`pipe_barrier`(4~64 cycles 不等)、`sync_block_set/wait`(跨核同步)、以及 `PIPE_ALL` barrier，它们的代价和依赖关系完全不同。

2. **scalarFactor = 3.74 是固定值**。这是从 FlashAttention `BM=48/64` 的 profiling 数据校准出来的——“纯 vector 时间只占 wall time 的 21%，所以乘以 3.74”。但对于不同的 kernel（纯 elementwise vs attention vs matmul），scalar/barrier/idle 的比例显然不同。

3. **没有 double-buffering 建模**。PipelineScheduler 完全不知道 `hivm.multi_buffer = 2`，因此无法正确建模相邻 wave 间 load/store/compute 的流水线重叠。

4. **没有 event 依赖建模**。PipelineScheduler 只跟踪每条 pipe 上 "上一个 op 什么时候结束"（`currentCycle`），然后 `start = max(unitFree, dependenciesReady)`。但 NPU 的同步不是简单的数据依赖——`set_flag[MTE3, MTE2, EVENT_ID0]` 和 `wait_flag[MTE3, MTE2, EVENT_ID0]` 之间是跨 pipe 的异步信号，需要精确的 event 时间跟踪。

**HIVMAnalysis 就是为了解决这些问题而写的。** 它直接解析 NPUIR（`hivm.hir.*` ops），从中提取完整的同步指令、buffer 分配、循环结构，然后在精确的 pipe 模型上做离散事件调度。

---

## 1. 整体架构：一条独立于 TTIR 的分析路径

```
NPUIR (.npuir.mlir)
      │
      ▼
sanitizeMlirBuffer()      ← 移除 hivm/hacc 自定义 dialect 属性，让 MLIR parser 能处理
      │
      ▼
analyzeModule()           ← 入口：遍历 IR tree
      │
      ├── analyzeParsedRegion() → analyzeParsedOperation()
      │       │
      │       ├── captureConstant()          ← 收集 arith.constant
      │       ├── captureDerivedScalarValue()← 折叠 arith 运算 (add/sub/mul/div/rem/min)
      │       ├── captureBufferMetadata()    ← 读取 hivm.multi_buffer annotation
      │       ├── scf::ForOp 处理            ← loop replay (DES) 或 multiplier (static)
      │       ├── populateHivmOp()           ← 解析 hivm.hir.* → ParsedOp
      │       │     ├── 确定 pipe (Vector/MTE2/MTE3/Cube/...)
      │       │     ├── 提取 set_flag/wait_flag event 信息
      │       │     └── 提取 pipe_barrier 信息
      │       ├── attachBufferAccessMetadata() ← 追踪 buffer root + multiBufferSlots
      │       ├── estimateDuration()         ← 计算每条指令的 cycle 数
      │       └── ingestParsedOp()           ← 建立依赖图 (数据依赖 + event 依赖 + pipe 顺序)
      │
      ▼
finalizeDiscreteEventReport() / finalizeScheduledReport()
      │
      ├── normalizeSyncBlockGenerations()    ← 跨 FuncOp 的 sync_block 匹配
      ├── wireCrossCoreSyncDependencies()    ← 跨核 set→wait 连线
      └── DES scheduling loop               ← 离散事件模拟
            ├── computeStartTime()            ← 考虑四种约束
            ├── startOp()                     ← 分配 buffer slot
            └── completeOp()                  ← 更新 event 可见时间
```

**关键设计决策**：HIVMAnalysis 被设计为**独立工具**而非 pipeline 中的 pass。它有两种使用模式：

- **MLIR Pass 模式**：`inproc-costmodel --analyze-hivm input.npuir.mlir`，走 `HIVMAnalysisPass::runOnOperation()`
- **直接文件模式**：`HIVMAnalyzer::analyzeFile(path)`，自己做 MLIR parsing

这两种模式最终都调用 `analyzeModule()`，走同一套分析逻辑。

---

## 2. 双路径解析：为什么需要 `TRITONSIM_HAS_BISHENGIR_HIVM` 宏

`HIVMAnalysis.cpp` 最显著的特征是大面积的 `#ifdef TRITONSIM_HAS_BISHENGIR_HIVM` 条件编译。这不是代码质量问题，而是一个精心设计的**双模式架构**。

### 2.1 背景：bishengir 是闭源的外部编译器

NPUIR 中的 `hivm.hir.*` ops 属于 bishengir 的 HIVM dialect。这个 dialect 的头文件（`bishengir/Dialect/HIVM/IR/HIVM.h`）和链接库只有在华为内部开发环境中才能获得。开源版本的 triton-ascend **不包含 bishengir**。

### 2.2 模式 A：有 bishengir（内部构建）—— `populateTypedHivmOp()`

```cpp
// HIVMAnalysis.cpp:617-781
static bool populateTypedHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  parsed.op.opName = getLeafOpName(op).str();

  // 通过 MLIR interface 直接读取 pipe 信息
  if (auto pipeIface = llvm::dyn_cast<mlir::hivm::OpPipeInterface>(op)) {
    parsed.op.pipe = convertTypedPipe(pipeIface.getPipe());
  }
  // 通过 MLIR interface 读取 core type
  if (auto coreIface = llvm::dyn_cast<mlir::hivm::CoreTypeInterface>(op))
    parsed.op.coreType = stringifyTypedCore(coreIface.getCoreType());
  // ...
}
```

当 bishengir HIVM dialect 可用时，`hivm::LoadOp`、`hivm::SetFlagOp` 等都有完整的 C++ 类型定义。可以通过 `llvm::dyn_cast` 做精确的类型匹配，通过 Op interface 读取 pipe、core type、event ID 等信息。**解析质量最高，不需要任何字符串解析。**

### 2.3 模式 B：无 bishengir（开源构建）—— `populateGenericHivmOp()`

```cpp
// HIVMAnalysis.cpp:844-907
static bool populateGenericHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  parsed.op.opName = getLeafOpName(op).str();
  parsed.op.coreType = inferGenericCoreType(op);   // 从 func.func 名字推断
  std::string opText = renderOperation(op);         // 打印 op 文本

  if (parsed.op.opName == "load") {
    // 从文本中解析地址空间，确定 pipe
    parsed.op.pipe = selectMTE2PipeForSpaces(spaces.first, spaces.second, ...);
  } else if (parsed.op.opName == "set_flag" || parsed.op.opName == "wait_flag") {
    // 从 op attributes 中解析 set_pipe/wait_pipe/static_event_id
    parsed.senderPipe = parsePipeToken(stringifyAttribute(op->getAttr("set_pipe")));
    parsed.receiverPipe = parsePipeToken(stringifyAttribute(op->getAttr("wait_pipe")));
    parsed.eventId = parseEventToken(stringifyAttribute(op->getAttr("static_event_id")));
  }
  // ...
}
```

在没有 bishengir 时，ops 被 MLIR parser 当作**未注册 dialect 的通用 operation** 处理——`op->getAttr("set_pipe")` 返回的是字符串化的 attribute（因为 MLIR parser 不知道这个 attribute 的类型，保留为 `StringAttr` 或 `BuiltinAttr`）。因此需要通过 `stringifyAttribute()` + `parsePipeToken()` 做**字符串级解析**。

**两种模式的主要差异**：

| 方面 | Typed (有 bishengir) | Generic (无 bishengir) |
|------|---------------------|----------------------|
| op 类型 | `hivm::LoadOp` 等精确 C++ 类型 | 通用 `Operation*` + 名字匹配 |
| pipe 获取 | `convertTypedPipe(pipeIface.getPipe())` | `parsePipeToken(stringifyAttribute(op->getAttr("pipe")))` |
| core type | `coreIface.getCoreType()` | `inferGenericCoreType(op)` + 字符串推断 |
| event ID | 直接读取 enum | `parseEventToken()` 字符串解析 |
| 依赖项 | 需要链接 `libHIVM` | 只需要标准 MLIR |
| PIPE_MTE2 歧义消除 | 通过 peer pipe 类型推断 | 通过 core type 或地址空间推断 |

### 2.4 sanitizeMlirBuffer()：处理原始 NPUIR 文件

当 HIVMAnalysis 通过 `analyzeFile()` 直接读取 `.npuir.mlir` 文件时，还有一层额外处理：

```cpp
// HIVMAnalysis.cpp:508-560
static std::string sanitizeMlirBuffer(llvm::StringRef buffer) {
  // 1. 过滤 warning 行
  // 2. 过滤 ld.lld / ERROR / WARNING 行
  // 3. 替换 #hivm.address_space<gm> → 0, <ub> → 1 等（整数 memory space）
  // 4. 删除 hacc.arg_type / hivm.func_core_type / hacc.function_kind 属性
}
```

**为什么需要 sanitize？** `.npuir.mlir` 文件是从 `bishengir-compile --bishengir-print-ir-after` 输出的，包含完整的 `#hivm.address_space<gm>` 等自定义 dialect 属性。在没有 bishengir 的构建中，MLIR parser 不认识这些属性格式，会解析失败。sanitize 把自定义属性替换为整数（MLIR 内置格式）或直接删除。

**为什么安全？** HIVMAnalysis 只关心 op 名称、pipe 标记、event ID、地址空间（gm/ub/l1）这些性能相关信息。`hacc.function_kind<DEVICE>` 这类 `hacc` 属性纯粹是后端编译器用的元数据，不影响性能建模。

### 2.5 sanitize 的 Fallback 策略

```cpp
// HIVMAnalysis.cpp:2728-2741
llvm::SmallVector<llvm::StringRef, 2> parseCandidates;
parseCandidates.push_back(rawBuffer);        // 先尝试原始内容
if (sanitized != rawBuffer)
  parseCandidates.push_back(sanitized);      // 失败了再尝试 sanitized
for (llvm::StringRef buffer : parseCandidates) {
  if (auto module = mlir::parseSourceString<mlir::ModuleOp>(buffer, &context)) {
    // 成功则直接分析
  }
}
```

先尝试 parse 原始内容（如果构建中有 bishengir，原始内容就能直接 parse），失败再用 sanitized 版本。这是一个**优雅的兼容性设计**。

---

## 3. 常量折叠引擎：`captureDerivedScalarValue()`

HIVMAnalysis 不是做 symbolic execution（符号执行）——它对 loop trip count、buffer size、offset 等需要**具体数值**。因此它内置了一个简化版的常量折叠引擎。

### 3.1 设计思路

NPUIR 中有大量的 arith 运算：

```mlir
%0 = arith.muli %arg7, %arg8 : i32          // num_x * num_y
%1 = arith.muli %0, %arg9 : i32             // * num_z = total_blocks
%2 = arith.ceildivsi %1, %c40_i32 : i32     // ceil(total/40) = num_waves
```

如果 HIVMAnalysis 不能算出 `total_blocks` 和 `num_waves` 的具体值，就无法确定 `scf.for` 的 trip count，也就无法正确设置 `loopMultiplier`。

### 3.2 实现

```cpp
// HIVMAnalysis.cpp:1229-1350
static bool captureDerivedScalarValue(mlir::Operation *op, AnalysisState &state) {
  // 支持的 op: IndexCast, IndexCastUI, TruncI, ExtSI, AddI, SubI, MulI, DivSI,
  //             RemSI, MinSI, CmpI, AndI, OrI, Select, CeilDivSI, FloorDivSI,
  //             get_block_idx, AffineApply
```

对每个支持的 Op：
1. 用 `resolveMLIRValue()` 递归解析 operands 到具体 int64 值
2. 执行运算得到结果
3. 将结果存入 `state.boundValues[result]`，供后续 op 引用

对于不支持或无法解析的 Op，直接跳过。**这是一种"尽力而为而不阻塞"的策略**——无法折叠的值不会导致分析失败，只是用默认值（tripCount=1, bytes=0）继续。

### 3.3 `hivm.hir.get_block_idx` 的特殊处理

```cpp
// HIVMAnalysis.cpp:1239-1246
if (op->getName().getStringRef() == "hivm.hir.get_block_idx") {
  auto it = state.argBindings.find("pid_x");
  if (it != state.argBindings.end())
    return recordValue(it->second);
  return false;
}
```

`get_block_idx` 是一个硬件指令——它的值取决于运行时当前物理核编号。HIVMAnalysis 用 `arg-bindings` 中的 `pid_x` 参数来替代。例如 `--analyze-hivm="arg-bindings=arg10=128,pid_x=0"` 表示分析 block_idx=0 的情况。

**为什么只支持 `pid_x`？** 因为大多数 kernel 只用 1D grid，`pid_x` 就足够了。对于 2D/3D grid，需要使用 `arg-bindings` 绑定更多的 `pid_y`、`pid_z`。这是一个已知的限制——如果 `arg-bindings` 没提供，`get_block_idx` 保持未知，依赖它的计算（如 `divsi %block_id, %grid_y` 反推 program_id）也会保持未知。

---

## 4. Loop Replay：DES 模式的核心

这是 HIVMAnalysis 区别于 PipelineScheduler 的一个关键创新。

### 4.1 问题

NPUIR 的 wave loop 结构是：

```mlir
scf.for %arg10 = %c0_i32 to %2 step %c1_i32 {
  // %13 = (loop_iter mod 2) ? EVENT_ID0 : EVENT_ID1  ← 交替选择
  hivm.hir.wait_flag[<MTE3>, <MTE2>, %13]
  hivm.hir.load ...  // x
  hivm.hir.load ...  // y
  hivm.hir.vadd ...
  hivm.hir.store ...
  hivm.hir.set_flag[<MTE3>, <MTE2>, %13]
}
```

如果只是把循环体里的每条 op 设置 `loopMultiplier = tripCount`（像 static 模式那样），你会丢失循环体内部的依赖关系——因为 wait_flag 使用的 `%13` 依赖 `loop_iter`，而不同迭代使用不同的 event ID。

### 4.2 Replay 策略

```cpp
// HIVMAnalysis.cpp:2069-2103
if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
  if (replayIterations && hasConcreteTripCount && tripCount > 1) {
    // DES 模式：展开循环，每次迭代独立分析
    for (int64_t iter = 0; iter < tripCount; ++iter) {
      loopState.boundValues[forOp.getInductionVar()] = lowerBound + iter * step;
      analyzeParsedRegion(op->getRegion(0), loopMultiplier, loopState, ...);
      if (iter + 1 < tripCount)
        advanceLoopCarriedState(forOp, loopState);   // 传递跨迭代状态
    }
  } else {
    // Static 模式：不展开，用 loopMultiplier 乘
    analyzeParsedRegion(op->getRegion(0), nestedMultiplier, loopState, ...);
  }
}
```

**DES (Discrete Event Simulation) 模式**下，`replayIterations = true`：循环被**物理展开**，每次迭代生成独立的一组 `HIVMOp`，它们的 `dependsOn` 关系精确反映了 event 的交替。

**Static 模式**下，`replayIterations = false`：循环体只分析一次，但每条 op 的 `loopMultiplier` 乘以 tripCount。这是一种近似——假设每次迭代的计算和同步模式完全一样。

### 4.3 `advanceLoopCarriedState()` 的妙用

```cpp
// HIVMAnalysis.cpp:1993-2036
static void advanceLoopCarriedState(mlir::scf::ForOp forOp,
                                    AnalysisState &loopState) {
  // 对于每个 loop-carried value (iter_arg):
  //   yield_op.operand[i] → for_op.region_arg[i+1] (下一次迭代的输入)
  // 将 yield 值关联的 state（producer、buffer root、constant、bound value）
  // 复制到对应的 region arg
}
```

这保证了迭代 N+1 的 ops 能正确依赖迭代 N 的 ops——如果迭代 N 的 `set_flag` 是 `event_id_0` 的 producer，迭代 N+1 的 `wait_flag` 可以正确找到这个依赖。

### 4.4 `seedLoopCarriedState()` 和 `propagateLoopResults()`

- **seed**：在进入循环前，将 `scf.for` 的 init args 关联的状态注入到循环体的 block argument
- **propagate**：循环分析完成后，将循环体内最新的状态传播回外部作用域

这两个函数配合 `advanceLoopCarriedState()` 完成了完整的**循环状态传播链**：

```
外部 state → seed → 循环体 iter0 → advance → iter1 → ... → iterN
                                                              ↓
外部 state ← propagate ←──────────────────────────────────────┘
```

---

## 5. `expandMacroOp()`：复合指令拆解

某些 `hivm.hir` op 实际上是**宏指令**——它们在硬件上横跨多个 pipe。如果不拆解，调度器无法正确建模流水线重叠。

### 5.1 `mmadL1` → `mmadL1.mte1` + `mmadL1.cube`

```cpp
// HIVMAnalysis.cpp:78-98
if (name == "mmadL1") {
  HIVMOp mte = parsed.op;
  mte.opName = "mmadL1.mte1";
  mte.pipe = HIVMPipe::MTE1;       // L1→L0 数据搬运
  mte.duration = estimateMmadL1MTE1Cycles(parsed, config);

  HIVMOp cube = parsed.op;
  cube.opName = "mmadL1.cube";
  cube.pipe = HIVMPipe::Cube;      // 实际计算
  cube.duration = parsed.op.duration;  // 注意：NOT 减去 MTE1

  // 两者 overlap，cube 不需要等 MTE1 完成
}
```

**为什么 cube duration 不减 MTE1？** 因为 MTE1 和 Cube 是**流水线重叠**的——MTE1 搬运 L1→L0A 的同时，Cube 可以开始计算已经就绪的数据。实际上 `mmadL1` 的总时间由两者中较长的决定，而 `startCycle/endCycle` 由调度器自动处理——MTE1 和 Cube 在不同 pipe 上，调度器会让它们并行启动，结束时间由各自 duration 决定。

### 5.2 `matmul` / `mix_matmul` → `matmul.mte2` + `matmul.cube` + `matmul.mte3`

```cpp
// HIVMAnalysis.cpp:101-132
if (name == "matmul" || name == "mix_matmul" || name == "mix_group_matmul") {
  int64_t preload = config.getMTE2StartupLatency();   // MTE2 预取
  int64_t drain   = config.getMTE3StartupLatency();   // MTE3 排空
  int64_t compute = parsed.op.duration - preload - drain;

  // 三条子 op，分别在不同 pipe 上
  HIVMOp mte2; mte2.pipe = CubeMTE2; mte2.duration = preload;
  HIVMOp cube; cube.pipe = Cube;     cube.duration = compute;
  HIVMOp mte3; mte3.pipe = MTE3;     mte3.duration = drain;
}
```

这是经典的三级流水线模型：MTE2 预取 → Cube 计算 → MTE3 写回。解耦后三条子 op 可以在不同 pipe 上重叠执行。

### 5.3 解耦后的依赖管理

```cpp
// HIVMAnalysis.cpp:1863-1876
for (HIVMOp &expanded : expandedOps) {
  if (previousExpandedId != max)
    expanded.dependsOn.push_back(previousExpandedId);  // 子 op 间顺序依赖
  if (expanded.pipe != mutableParsed.op.pipe)
    expanded.dependsOn.push_back(latestPipeProducer[expanded.pipe]);  // pipe 顺序
}
```

解耦后的子 op 之间保持顺序依赖（MTE2 必须先于 Cube 先于 MTE3），同时也继承原 op 在各自 pipe 上的顺序约束。

---

## 6. Cycle 估算：`estimateDuration()`

### 6.1 设计原则

HIVMAnalysis 的 cycle 估算遵循**简洁优先**原则——每条 op 只有 1~5 行代码。不像 PipelineScheduler 那样把 cycle 数写成 Op 类的成员函数，而是用**一个集中的 switch-like 函数**：

```cpp
// HIVMAnalysis.cpp:1496-1655 (简化)
static int64_t estimateDuration(const ParsedOp &parsed, const HardwareConfig &config) {
  if (opName == "set_mask_norm")    return 1;
  if (opName == "get_block_idx")   return 1;
  if (opName == "set_flag")        return 1;
  if (opName == "wait_flag")       return 2;
  if (opName == "pipe_barrier")    return pipe == Vector ? 4 : pipe == MTE2 ? 16 : 8;
  if (opName == "load")            return mte2Startup + bandwidth_cycles;
  if (opName == "store")           return mte3Startup + bandwidth_cycles;
  if (opName == "vadd")            return estimateVectorCycles(...);
  if (opName == "vbrc")            return estimateVectorCycles(...);
  if (opName == "matmul")          return estimateCubeCycles(...);
  // ... 总共约 20 种 op
}
```

**为什么不用 OOP？** 因为这里是 NPUIR——op 是 `hivm::LoadOp` 等外部 dialect 的实例，作者无法给它们加虚函数。所以用集中式的类型匹配来处理。

### 6.2 `set_flag` = 1, `wait_flag` = 2 的依据

```cpp
if (opName == "set_flag")  return 1;   // Scalar pipe 上的一条 store 指令
if (opName == "wait_flag") return 2;   // Scalar pipe 上的一条 load + check 指令
```

`set_flag` 和 `wait_flag` 在 PIPE_S（Scalar pipe）上执行——它们是简单的寄存器写入/轮询操作。
- `set_flag`：Scalar core 向目标 pipe 的事件寄存器写 1，1 个 cycle
- `wait_flag`：Scalar core 轮询目标 pipe 的事件寄存器直到读到 1，通常需要 2 个 cycle（读 + 条件分支）

### 6.3 `pipe_barrier` 的梯度差距

```cpp
// HIVMAnalysis.cpp:1612-1633
if (opName == "pipe_barrier") {
  case HIVMPipe::Vector:    return 4;    // Vector pipe 深度浅
  case HIVMPipe::VectorMTE2:
  case HIVMPipe::CubeMTE2:  return 16;   // DMA pipe 深度中等
  case HIVMPipe::MTE3:      return 16;   // DMA pipe 深度中等
  case HIVMPipe::All:       return 64;   // 跨所有 pipe，最长
}
```

`pipe_barrier` 不是简单的"等待"。它的语义是**排空目标 pipe 上的所有 in-flight 指令**。不同 pipe 的深度（pipeline depth）不同——Vector pipe 只有 4~5 级，DMA pipe (MTE2/MTE3) 有 16 级，PIPE_ALL 需要等所有 pipe 排空，约 64 cycle。

这个 4/16/64 的分档比 PipelineScheduler 的固定 7500 精确得多——因为 `pipe_barrier` 的代价取决于**目标 pipe 而不是 kernel 类型**。

### 6.4 `load` 和 `store` 的带宽模型

```cpp
// HIVMAnalysis.cpp:1635-1665
if (opName == "load") {
  int64_t bytes = parsed.op.bytes;
  auto spaces = parseLoadStoreSpaces(line);   // ("gm", "ub")
  double bandwidth = config.getMemoryBandwidthBytesPerCycle("hbm");
  int64_t transferCycles = ceil(bytes / bandwidth);
  return config.getMTE2StartupLatency() + latency + transferCycles;
}
```

对比 PipelineScheduler 的做法——PipelineScheduler 在 Op 类里实现 `estimateCycles()`：

```cpp
// AscendModelOps.cpp - MemoryOp::estimateCycles
static int64_t estimateMemoryCycles(int64_t bytes, int transferType, ...) {
  int64_t numBursts = (bytes + burstSize - 1) / burstSize;
  return numBursts * cyclesPerBurst + startupLatency;
}
```

两者本质相同（startup + 带宽），但 HIVMAnalysis 多了 `latency`（memory access latency）这一项，因为对不同地址空间组合（gm→ub, ub→gm, gm→l1, l1→l0a 等），latency 不同。

---

## 7. 依赖图构建：`ingestParsedOp()`

`ingestParsedOp` 是连接"解析"和"调度"的桥梁。它负责为每个 ParsedOp 建立三类依赖。

### 7.1 数据依赖（SSA 依赖）

```cpp
// HIVMAnalysis.cpp:2123-2127
for (mlir::Value operand : op->getOperands()) {
  auto it = state.valueProducers.find(operand);
  if (it != state.valueProducers.end())
    parsed.op.dependsOn.push_back(it->second);  // value → 它的 producer
}
```

和 PipelineScheduler 一样——如果一个 op 使用了另一个 op 的输出值（SSA use-def 链），就建立数据依赖。

### 7.2 Event 依赖（set_flag → wait_flag）

```cpp
// HIVMAnalysis.cpp:1848-1858
if (parsed.op.opName == "wait_flag" || parsed.op.opName == "sync_block_wait") {
  auto eventIt = state.eventProducers.find(opEventKey);
  if (eventIt != state.eventProducers.end())
    parsed.op.dependsOn.push_back(eventIt->second);  // wait → set
}
```

这是 PipelineScheduler **没有**的。`eventProducers` 是一个 map：`(sender, receiver, eventId) → opId`。当 `set_flag[MATE3, MATE2, EVENT_ID0]` 完成时，它注册为 `(MTE3, MTE2, EVENT_ID0)` 的 producer。后续的 `wait_flag[MTE3, MTE2, EVENT_ID0]` 就能找到这个依赖。

**但这里有一个微妙的问题**：如果同一个 event 被多次 set（如在 loop 中），`eventProducers` 被覆盖为最后一次 set。这在 static 模式下是近似处理——只依赖最后一次 set。在 DES 模式下有 `eventGeneration` 机制（见第 8 节）。

### 7.3 Pipe 顺序依赖

```cpp
// HIVMAnalysis.cpp:1902-1904
if (expanded.pipe != HIVMPipe::All && expanded.pipe != HIVMPipe::Unknown) {
  state.latestPipeProducer[expanded.pipe] = expanded.id;
}
```

每条 pipe 上，op 必须按程序顺序执行。这是通过 `latestPipeProducer[pipe]` 追踪的——每个 op 在处理时都会将 `latestPipeProducer` 中自己那条 pipe 的前一个 op 加入依赖列表。

### 7.4 PIPE_ALL barrier 的特殊处理

```cpp
// HIVMAnalysis.cpp:1896-1901
if (expanded.pipe == HIVMPipe::All && expanded.isBarrier) {
  for (HIVMPipe p : getCoreBarrierPipes(expanded.coreType))
    state.latestPipeProducer[p] = expanded.id;
}
```

`PIPE_ALL` barrier 会阻塞**该 core 上的所有 pipe**。所以它在每个 pipe 上都注册为最新的 producer——后续任何 pipe 上的 op 都必须等这个 barrier 完成。

---

## 8. DES 调度器：`finalizeDiscreteEventReport()`

这是 HIVMAnalysis 的**核心引擎**——大约 300 行代码，实现了完整的离散事件模拟器。

### 8.1 数据初始化

```cpp
// HIVMAnalysis.cpp:2372-2386
for (const HIVMOp &op : report.operations) {
  // 对每个 write buffer，根据 multiBufferSlots 创建对应数量的 slot
  for (const std::string &root : op.writeBuffers) {
    auto &state = bufferStates[root];
    if (state.slots.empty()) {
      int64_t count = std::max<int64_t>(1, op.multiBufferSlots);
      for (int64_t i = 0; i < count; ++i)
        state.slots.push_back(BufferSlotState{});
    }
    state.versionReadableAt.emplace(0, 0);  // version 0 始终可读
  }
}
```

每个 buffer 根据 `multiBufferSlots` 创建多个 slot。如果 `hivm.multi_buffer = 2`，则创建 2 个 slot——两个 slot 可以独立被读写，相邻 wave 交替使用。如果 multiBufferSlots = 1（默认），只有一个 slot，必须等写入完成后才能覆盖。

### 8.2 `computeStartTime()`：四种约束的交集

```cpp
// HIVMAnalysis.cpp:2477-2526
auto computeStartTime = [&](const HIVMOp &op) -> int64_t {
  int64_t start = readyAt[op.id];  // ① 数据依赖

  // ② Buffer 可读性依赖
  for (size_t idx = 0; idx < op.readBuffers.size(); ++idx) {
    int64_t version = readBufferVersions[idx];
    start = max(start, versionReadableAt[root][version]);
  }

  // ③ Buffer 可写性依赖
  for (const std::string &root : op.writeBuffers) {
    int64_t slotReady = min(slot.writableAt for slot in bufferStates[root]);
    start = max(start, slotReady);
  }

  // ④ Event 可见性依赖
  if (op.opName == "wait_flag" || op.opName == "sync_block_wait") {
    start = max(start, flagEventVisibleAt[{sender, receiver, eventId, generation}]);
  }

  // ⑤ Pipe 可用性
  return max(start, pipeAvailableAt[op.pipe]);
};
```

五种约束，按优先级排列：
1. **数据依赖**：SSA chain，和 PipelineScheduler 相同
2. **Buffer 可读**：需要确保读取的 buffer version 已经写完
3. **Buffer 可写**：需要至少一个 slot 空闲
4. **Event 可见**：`wait_flag` 需要等对应的 `set_flag` 完成
5. **Pipe 可用**：目标 pipe 上空闲

### 8.3 `startOp()`：Buffer Slot 分配

```cpp
// HIVMAnalysis.cpp:2552-2565
for (const std::string &root : op.writeBuffers) {
  // 找最早可写的 slot
  size_t bestSlot = 0;
  int64_t bestTime = state.slots[0].writableAt;
  for (size_t i = 1; i < state.slots.size(); ++i) {
    if (state.slots[i].writableAt < bestTime) {
      bestTime = state.slots[i].writableAt;
      bestSlot = i;
    }
  }
  state.slots[bestSlot].writableAt = endTime;
  // 记录分配: op_id → (root, slot_index)
}
```

**greedy slot 分配**：总是选最早可写的 slot。对于 multiBufferSlots=2，两个 slot 轮流使用——wave 0 写 slot 0，wave 1 写 slot 1，wave 2 时 slot 0 已经空闲，可以复用。

### 8.4 `completeOp()`：Event 时间更新

```cpp
// HIVMAnalysis.cpp:2423-2431
if (op.opName == "set_flag" && !op.eventId.empty()) {
  EventInstanceKey key{{sender, receiver, eventId}, eventGeneration};
  flagEventVisibleAt[key] = time;  // set_flag 完成时刻 = event 可见时刻
}
```

当一个 `set_flag` 在 `time` 时刻完成，对应的 event 从 `time` 时刻起对 `wait_flag` 可见。

**eventGeneration 的作用**：在 loop 中同一个 event（如 EVENT_ID0）可能被多次 set。DES 展开 loop 后，每次 set 有不同的 generation 号（1, 2, 3, ...），wait_flag 通过 generation 匹配正确的 set：

```
iter 0: set_flag[EVENT_ID0] gen=1 → flagEventVisibleAt[{..., gen=1}] = 100
iter 1: wait_flag[EVENT_ID0] gen=1 → start >= 100 ✓  (正确等 iter 0 的 set)
iter 1: set_flag[EVENT_ID0] gen=2 → flagEventVisibleAt[{..., gen=2}] = 250
iter 2: wait_flag[EVENT_ID0] gen=2 → start >= 250 ✓  (正确等 iter 1 的 set)
```

### 8.5 主调度循环

```cpp
// HIVMAnalysis.cpp:2587-2634
while (completedCount < numOps) {
  // Step 1: 在当前时刻，启动所有可以启动的 op
  for (opId : readyOps) {
    startTime = computeStartTime(op);
    if (startTime <= currentTime) {
      startOp(opId, currentTime);
    }
  }

  // Step 2: 完成所有在当前时刻应该完成的 op
  while (completions.top().time <= currentTime) {
    completeOp(opId, currentTime);
  }

  // Step 3: 如果没有任何进展，快进到下一个事件时刻
  if (!startedAny) {
    currentTime = min(nextCompletionTime, nextReadyStartTime);
  }
}
```

这是经典的**离散事件模拟循环**：
- `readyOps`：所有依赖已满足、等待启动的 op
- `completions`：min-heap，按完成时间排序的正在执行的 op
- `currentTime`：当前模拟时间，只在没有 op 可以启动时快进

### 8.6 跨核同步：`wireCrossCoreSyncDependencies()`

对于 mix mode kernel（CUBE + VECTOR 双核）：

```cpp
// HIVMAnalysis.cpp:2315-2341
static void wireCrossCoreSyncDependencies(HIVMAnalysisReport &report) {
  // sync_block_set (on AIC, event="block_0") → sync_block_wait (on AIV, event="block_0")
  for (HIVMOp &op : report.operations) {
    if (op.opName == "sync_block_set")   setOpById[{eventId, core, gen}] = op.id;
    if (op.opName == "sync_block_wait")  op.dependsOn.push_back(setOpById[{eventId, core, gen}]);
  }
}
```

跨核同步的 set 和 wait 在不同的 `func::FuncOp` 中（AIC 函数和 AIV 函数），它们之间的依赖无法通过 SSA use-def 链建立。因此需要在所有 op 解析完后，通过 event ID 和 core type 手动匹配。

---

## 9. Static 调度器：`finalizeScheduledReport()`

当 scheduler mode = `static` 时，使用简化版调度器：

```cpp
// HIVMAnalysis.cpp:2188-2259
for (HIVMOp &op : report.operations) {
  // 1. 从 dependsOn 计算 earliest start
  int64_t earliest = max(dep.endCycle for dep in dependsOn);

  // 2. 与 pipe 可用性比较
  if (op.pipe == HIVMPipe::All)   start = max(earliest, max(all_pipes_available));
  else if (op.pipe == Unknown)    start = earliest;
  else                            start = max(earliest, pipeAvailableAt[op.pipe]);

  // 3. 更新
  op.startCycle = start;
  op.endCycle = start + op.duration;
  pipeAvailableAt[op.pipe] = op.endCycle;
}

// 加权计算（不是简单 sum）
report.weightedCycles = max(pipe.weightedPipeCycles) + globalBarrierCycles;
```

**Static 模式不展开循环**——loop body 只出现一次，但每条 op 带着 `loopMultiplier`。最终的 `weightedCycles` 取各 pipe 的加权 busy cycles 的最大值（即瓶颈 pipe），加上全局 barrier 的加权 cycles。

**Static vs DES 的选择**：Static 模式更快（O(n) vs DES 的 O(n log n)），适合快速评估。DES 模式更精确（展开 loop + event generation + buffer slot），适合精确分析。

---

## 10. HIVMAnalysisPass：从 Pass 到报告

`HIVMAnalysisPass` 是一个非常薄的 wrapper（只有 95 行）：

```cpp
// HIVMAnalysisPass.cpp:33-92
struct HIVMAnalysisPass : public impl::HIVMAnalysisPassBase<HIVMAnalysisPass> {
  void runOnOperation() override {
    // 1. 加载 HardwareConfig (JSON)
    auto hardwareConfig = loadHardwareConfigForAnalysis(hardwareConfigPath, ...);

    // 2. 解析 scheduler mode: static / des
    auto schedulerOr = parseSchedulerMode(schedulerMode);

    // 3. 创建 analyzer 并运行
    HIVMAnalyzer analyzer(config, argBindingsStr, *schedulerOr);
    HIVMAnalysisReport report;
    analyzer.analyzeModule(module, report, error);

    // 4. 输出报告 (stderr 或文件)
    report.print(os, config);

    // 5. 可选：输出 Perfetto trace
    report.emitPerfettoTrace(os, config);
  }
};
```

它使用了 MLIR Pass 框架的自动 option 绑定（`Passes.td` 中的 `Option<"hardwareConfigPath", ...>` 自动生成命令行参数解析），但核心逻辑全在 `HIVMAnalyzer::analyzeModule()` 中。

---

## 11. Perfetto Trace 输出

HIVMAnalysis 还能生成 Perfetto trace 格式的输出，用于可视化流水线调度结果：

```cpp
// HIVMAnalysis.cpp:2840-2900
void HIVMAnalysisReport::emitPerfettoTrace(...) {
  // AIC pid=1: Cube, MTE1, CubeMTE2, FixPipe, Scalar(AIC)
  // AIV pid=2: Vector, VectorMTE2, MTE3, Scalar(AIV)
  // Shared pid=3: All, Unknown
```

每条 pipe 被映射到 Perfetto 的一个 track（tid），同一 core 的 pipes 在同一个 process（pid）内。这使得调度结果可以直接在 `ui.perfetto.dev` 中查看，像看硬件波形一样直观。

---

## 12. 为什么 HIVMAnalysis 没有被集成到 `run_costmodel_inproc`？

回到用户最初的问题。答案现在很清楚了：

1. **输入格式不兼容**：`run_costmodel_inproc` 输入 TTIR → `ConvertTritonToAscend` pass → AscendModel dialect。HIVMAnalysis 输入 NPUIR（`hivm.hir.*` ops）。要集成，需要在 `run_costmodel_inproc` 中调用 `bishengir-compile` 把 TTIR 变成 NPUIR——但这需要 bishengir 二进制在运行环境中存在。

2. **编译依赖**：`TRITONSIM_HAS_BISHENGIR_HIVM` 宏。在开源构建中 HIVMAnalysis 的核心解析逻辑不可用（回退到 generic 路径），而 generic 路径依赖于 MLIR parser 能否正确处理未注册 dialect——这本身就是不可靠的。

3. **设计定位**：HIVMAnalysis 被设计为**独立分析工具**（standalone tool），类似 Intel VTune 或 NVIDIA Nsight 的角色——开发者跑完 kernel 后，dump 出 NPUIR，用 `inproc-costmodel --analyze-hivm` 做 offline 分析。它不是面向 autotune 的在线评估工具。

4. **缺少 Python 桥接**：`triton_ascend.cc` 中没有 `run_hivm_analysis` 这样的 pybind11 函数。如果要做集成，需要在 `PYBIND11_MODULE` 中增加一个新的 binding 函数，调用 `HIVMAnalyzer::analyzeModule()`。

---

## 13. 总结：HIVMAnalysis 的设计哲学

对比 PipelineScheduler 和 HIVMAnalysis：

| 维度 | PipelineScheduler | HIVMAnalysis |
|------|------------------|-------------|
| 输入 IR | TTIR → AscendModel dialect | NPUIR (hivm.hir ops) |
| Op 数量 | ~25 种（AsecdModelOps.td） | ~30 种（含 set_flag/wait_flag 等 sync op） |
| 同步建模 | 固定公式 `7500 × numIters` | 逐条 event 依赖 + generation 匹配 |
| Buffer 建模 | 无 | multiBufferSlots + slot 状态追踪 |
| Op 分解 | 无（每个 Op 一条指令） | expandMacroOp (matmul→mte2+cube+mte3 等) |
| 调度器 | 简单 pipe 可用性 + 数据依赖 | 五种约束 (数据 + event + buffer读 + buffer写 + pipe) |
| Loop 处理 | multiplier 乘 | DES 展开 + state propagation |
| 跨核同步 | 无 | sync_block_set→wait 匹配 |
| 可视化 | 文本 timeline | Perfetto trace (HTML) |
| 编译依赖 | 无（纯 MLIR 标准 dialect） | 可选 bishengir（双路径） |
| 使用场景 | autotune (在线) | offline 分析 (工具) |

HIVMAnalysis 代表了**"精确但需要更多信息"**的性能建模路线——它需要 NPUIR 输入来获得同步指令、buffer 分配、地址空间等精确信息，但它能给出远比 PipelineScheduler 精确的调度结果。两者的关系不是替代，而是**互补**——PipelineScheduler 做快速 autotune 评估，HIVMAnalysis 做深度 offline 分析和验证。
