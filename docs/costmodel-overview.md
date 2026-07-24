# CostModel 全栈解析：从 TTIR 到延迟预估

本文档从零开始，逐步解释 Triton-Ascend 的 CostModel 如何工作。读完你能理解：

- 为什么用 MLIR Dialect + Pass 管线做性能评估
- AscendModel 方言的定义和注册流程
- 每条 NPU 指令的 cycle 数是怎么算的
- 调度器如何把单 Op 的 cycle 合并成完整的 kernel 耗时
- C++ Pass 管线如何串联，Python 侧如何调用
- 端到端示例：从 Triton kernel 到预估延迟

---

## 1. 为什么要用 MLIR Dialect + Pass 管线

CostModel 的输入是 **TTIR 文本**（`tt.func`、`tt.load`、`tt.dot` 等 Op）。它的任务是从这段文本里提取信息、按硬件规则算出预估的 cycle 数。

如果不借助 MLIR 框架，你需要**手写一个完整的 IR 处理系统**：

```
TTIR 文本 ──→ 手写 Parser ──→ 手写 IR 树 ──→ 手写遍历 ──→ 算 cycle

你需要自己实现:
  - MLIR 文本词法/语法解析（处理嵌套 module/func/block/region）
  - SSA 值的 use-def 链
  - Op 的类型系统（tensor<64x128xf16> 的 shape/dtype 推导）
  - 属性解析（{op_id = 3, estimated_cycles = 57}）
  - 循环嵌套识别和 trip count 推导
```

**MLIR Dialect + Pass 的方案**把上面所有脏活都交给了框架。整个流程分两段：

**第一段 — 框架负责（不需要我们写一行代码）**：

```
TTIR 文本 (字符串)
    │
    ▼  MLIR 框架的 Parser (parseSourceString)
    │  自动识别 "tt.func"、"scf.for"、"arith.addf" 等
    │  自动解析嵌套结构 (module → func → block → region → op)
    │  自动构建 SSA use-def 链
    │
    ▼
MLIR IR 树 (内存中的 C++ 对象)
  ModuleOp
    ├── tt.func @add_kernel
    │     ├── scf.for (循环)
    │     │     ├── tt.load  → SSA Value %19
    │     │     ├── tt.load  → SSA Value %21
    │     │     ├── arith.addf(%19, %21) → SSA Value %22
    │     │     └── tt.store(%22)
```

**第二段 — 我们写的 Pass 管线（在 IR 树上做转换和分析）**：

```
MLIR IR 树 (含 tt.* / arith.* / math.* 等 Op)
    │
    ▼  Pass ① ConvertTritonToAscend
    │  tt.load → ascend.vector_load, arith.addf → ascend.add, ...
    │
    ▼  Pass ② InsertDataTransfers
    │  Pass ③ AssignOpIDs
    │  Pass ④ EstimateCycles    ← 调每个 Op 的 estimateCycles()
    │  Pass ⑤ PipelineAnalysis  ← 调 PipelineScheduler
    │  Pass ⑥ PerfReport
    │
    ▼
输出: "Estimated Time: 3.25 us"
```

**核心收益**：

1. **不用写 parser/print 代码**：TableGen 根据 `.td` 定义自动生成每个 Op 的 `parse()`/`print()`/`verify()`/`build()`。你只需要定义"这个 Op 叫什么、有几个输入输出"，MLIR 框架就能解析和打印它。

2. **统一的遍历框架**：`module.walk([](Operation *op) { ... })` 一行代码遍历所有 Op，框架自动处理嵌套的 func、block、region、循环体。PassManager 保证 Pass 按顺序执行、状态正确传递。

3. **类型安全 + 自动生成 accessor**：TableGen 根据 `.td` 中定义的 `arguments = (ins AnyTensor:$lhs)` 自动在 `Ops.h.inc` 中生成 `getLhs()`、`getRhs()`、`getM()` 等有名字的 accessor，底层直接映射到 `getOperand(0)` / `getAttr("M")`。不需要手写魔法数字索引。

4. **硬件参数可配置**：换一块 NPU 芯片只需要替换 `ascend_910b.json`，不需要改 C++ 代码。

---

## 2. 定义 AscendModel 方言

要定义一个 MLIR 方言，你需要告诉 MLIR 框架三件事：
- **这个方言叫什么**（名字、命名空间）
- **有哪些 Op**（每个 Op 的输入、输出、属性）
- **Op 有什么共同行为**（Interface）

作者用 4 个 `.td` 文件表达了这些信息，它们的依赖关系决定了创建顺序：

```
① AscendModelBase.td          Dialect 声明 + 枚举定义 + Op 基类
        │
        ▼
② AscendModelInterfaces.td    Op Interface（estimateCycles/getHWUnit/...）
        │
        ▼
③ AscendModelOps.td          25 个 Op 定义（include ①②）
                                │
④ Passes.td                  7 个 Pass 声明（独立，不依赖 ①②③）
```

---

### 2.1 AscendModelBase.td —— Dialect 声明与硬件枚举

**文件**：`third_party/ascend/costmodel/include/AscendModel/IR/AscendModelBase.td`

这个文件做三件事：声明 Dialect 的存在、定义硬件相关枚举、提供所有 Op 的公共基类。

**第一步：声明 Dialect。**

```tablegen
def AscendModel_Dialect : Dialect {
  let name = "ascend";                    // MLIR 文本中 Op 的前缀: ascend.add, ascend.matmul
  let cppNamespace = "::mlir::ascend";   // C++ 代码的命名空间
}
```

所有后续 Op 定义都引用这个名字——后面定义的 `AscendModel_Op<"add">` 生成的 Op 在 MLIR 文本中就是 `ascend.add`。

**第二步：定义四类枚举。** CostModel 需要区分"一个 Op 在哪个硬件单元上执行""数据在哪一级存储空间""什么数据类型""什么向量/归约操作"。这些信息在 IR 中以字符串形式存在（比如 `{hw_unit = "cube"}`），但 C++ 代码需要类型安全的枚举值来做 switch-case 和调度。TableGen 用 `I32EnumAttrCase` 定义每个枚举值（C++ 名、数值、MLIR 字符串），`I32EnumAttr` 打包成完整枚举类型并自动生成 `stringify*()`/`symbolize*()` 转换函数：

| 枚举 | 值列表 | 用途 |
|------|-------|------|
| `HWUnit` | Cube, CubeMTE2, FixPipe, Vector, VecMTE2, MTE3, Scalar | 标识 Op 在哪个硬件流水线上执行 |
| `MemSpace` | HBM, L2, L1, UB | 标识数据在哪一级存储空间 |
| `DataType` | FP32, FP16, BF16, INT8, INT32 | 标识 Op 的数据类型 |
| `VecOpKind` / `ReduceKind` | Add, Sub, Mul, Div, Exp, Log, ... / Sum, Max, Min, Prod | 标识向量/归约操作的具体种类 |

以 `HWUnit` 为例，TableGen 定义：

```tablegen
def HWUnit_Cube     : I32EnumAttrCase<"Cube", 0, "cube">;
def HWUnit_Vector   : I32EnumAttrCase<"Vector", 3, "vector">;
// ... 共 7 个值

def HWUnitAttr : I32EnumAttr<"HWUnit", "Hardware execution unit", [
  HWUnit_Cube, HWUnit_CubeMTE2, HWUnit_FixPipe,
  HWUnit_Vector, HWUnit_VecMTE2, HWUnit_MTE3, HWUnit_Scalar
]> { let cppNamespace = "::mlir::ascend"; }
```

`HWUnit` 本身是一个**C++ 枚举类型**（`enum class HWUnit : uint32_t { Cube=0, CubeMTE2=1, ... }`），不是 IR 节点或 MLIR Type。它的值以**MLIR 属性**的形式附加在 Op 上——EstimateCyclesPass 调用 `op->setAttr("hw_unit", "cube")` 写入，后续通过 `op->getAttr("hw_unit")` 读取。

**`HWUnit` 在整个 costmodel 中的流动路径**：

```
AscendModelOps.cpp:  每个 Op::getHWUnit() 返回 HWUnit 枚举值
        │              例: AddOp::getHWUnit() → HWUnit::Vector
        │                   MatmulOp::getHWUnit() → HWUnit::Cube
        ▼
EstimateCycles.cpp:  EstimateCyclesPass 调 op.getHWUnit()，转成字符串
                     写入 IR: op->setAttr("hw_unit", "Vector")
                     并按 hwUnit 分类累计 FLOPs 和 bytes
        │
        ▼
PipelineScheduler:   初始化 7 个 HWUnitPipeline，每个对应一个 HWUnit 值
                     schedule() 中按 op.hwUnit 路由到对应的管道
        │
        ▼
PerfReportPass:      统计每个 HWUnit 的 busy cycles，找 bottleneck
```

**第三步：定义 Op 公共基类。** 避免每个 Op 定义重复写 `Op<AscendModel_Dialect, "name", [traits]>`：

```tablegen
class AscendModel_Op<string mnemonic, list<Trait> traits = []> :
    Op<AscendModel_Dialect, mnemonic, traits>;
```

---

### 2.2 AscendModelInterfaces.td —— "在写 Op 之前，先定义它们的能力合约"

**文件**：`third_party/ascend/costmodel/include/AscendModel/IR/AscendModelInterfaces.td`

为什么需要 Interface？因为后续的 `EstimateCyclesPass` 要遍历所有 Op，不管具体是 AddOp 还是 MatmulOp，统一调 `estimateCycles(config)`。MLIR 的 Op 用 CRTP 模板（不能做虚函数多态），Interface 是为这个场景设计的跨类型调用机制。

```tablegen
def EstimateCyclesOpInterface : OpInterface<"EstimateCyclesOpInterface"> {
  let cppNamespace = "::mlir::ascend";

  let methods = [
    // 纯虚 — 每个 Op 的公式不同
    InterfaceMethod<"estimateCycles", "int64_t", (ins "const HardwareConfig &":$config)>,
    InterfaceMethod<"getHWUnit", "HWUnit", (ins)>,

    // 有默认实现 — 只有少数 Op 需要覆写
    InterfaceMethod<"getTransferBytes", "int64_t", (ins), "return 0;">,
    InterfaceMethod<"getFlops",        "int64_t", (ins), "return 0;">,
    InterfaceMethod<"getCyclesPerVectorOp", "int", (ins), "return 1;">
  ];

  let extraClassDeclaration = [{
    int getElementBits();       // 从 operand 类型推断 bit 宽度
    int64_t getNumElements();   // 从 operand shape 推断元素个数
  }];
}
```

**纯虚 vs 默认实现的权衡**：25 个 Op 里只有 5 个是搬运 Op、只有计算 Op 需要 `getFlops()`。给另外 20 个各写一行 `return 0` 是噪音，所以给默认实现。`estimateCycles()` 每个 Op 都不一样——Add 用 Vector 公式、Matmul 用 Cube 公式、Broadcast 直接返回 1——没法给有意义的默认值，所以纯虚。

---

### 2.3 AscendModelOps.td —— 25 个 Op 定义

**文件**：`third_party/ascend/costmodel/include/AscendModel/IR/AscendModelOps.td`

开篇 include 前面两个文件：

```tablegen
include "AscendModelBase.td"
include "AscendModelInterfaces.td"
```

**先定义最核心的——矩阵乘**：

```tablegen
def Ascend_MatmulOp : AscendModel_Op<"matmul", [Pure,
    DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins
    AnyTensor:$lhs, AnyTensor:$rhs,
    I64Attr:$M, I64Attr:$N, I64Attr:$K,          // 矩阵维度 — 编译时已知
    OptionalAttr<I64Attr>:$estimated_cycles,     // Pass 写回的估算值
    OptionalAttr<I64Attr>:$op_id                 // AssignOpIDs Pass 写的序号
  );
  let results = (outs AnyTensor:$result);
}
```

注意：M/N/K 用 `I64Attr` 而不是从 tensor shape 推断。Shape 可能包含动态维度，但矩阵乘的 tile 划分必须在编译时确定。

**用模板继承批量定义结构相同的 Op**：5 个简单二元运算（Add/Sub/Mul/Max/Min）结构完全相同——都是两个 tensor 输入、一个 tensor 输出、都在 Vector 上跑。作者用 TableGen class 继承消除重复：

```tablegen
// 模板
class Ascend_VectorBinarySimple<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, OptionalAttr<I64Attr>:$estimated_cycles, OptionalAttr<I64Attr>:$op_id);
  let results = (outs AnyTensor:$result);
}

// 一行定义一个 Op
def Ascend_AddOp : Ascend_VectorBinarySimple<"add"> { ... }
def Ascend_SubOp : Ascend_VectorBinarySimple<"sub"> { ... }
def Ascend_MulOp : Ascend_VectorBinarySimple<"mul"> { ... }
```

Div 需要额外的模板参数——它不是 1 cycle/op：

```tablegen
class Ascend_VectorBinaryComplex<string mnemonic, int cycles_per_op>
    : AscendModel_Op<mnemonic, [Pure,
        DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getCyclesPerVectorOp", "getFlops"]>]> {
  int cyclesPerOp = cycles_per_op;
}
def Ascend_DivOp : Ascend_VectorBinaryComplex<"div", 4> { ... }
```

同样的模式用在一元（Neg/Abs/Relu/Cast/Exp/Log/Tanh/Sigmoid）、比较（CmpEq/CmpNe/CmpLt/CmpLe/CmpGt/CmpGe）、归约（ReduceSum/ReduceMax/ReduceMin/ReduceProd）Op。

**特殊 Op**：

```tablegen
// Store Op 没有输出 — 写到内存不产生新值
def Ascend_VectorStoreOp : AscendModel_Op<"vector_store", [...]> {
  let results = (outs);   // 空！
}

// Sync Op 不实现 Interface — 不是计算 Op
def Ascend_SyncOp : AscendModel_Op<"sync", []> {
  let arguments = (ins StrAttr:$sync_type);
  let results = (outs);
}
```

---

### 2.4 TableGen → .inc 文件生成

**文件**：`third_party/ascend/costmodel/include/AscendModel/CMakeLists.txt`

TableGen 不会直接编译 `.td`——它在 CMake 配置阶段运行 `llvm-tblgen`，生成 C++ 可 `#include` 的 `.inc` 文件。每次 `mlir_tablegen()` 调用产出一个文件：

```
从 AscendModelInterfaces.td:
  mlir_tablegen → AscendModel/IR/AscendModelInterfaces.h.inc  (接口声明)
  mlir_tablegen → AscendModel/IR/AscendModelInterfaces.cpp.inc (默认实现)

从 AscendModelOps.td:
  mlir_tablegen → AscendModel/IR/AscendModelOps.h.inc       (Op 类声明 + accessor)
  mlir_tablegen → AscendModel/IR/AscendModelOps.cpp.inc      (build/parse/print/verify 骨架)
  mlir_tablegen → AscendModel/IR/AscendModelDialect.h.inc    (Dialect 类声明)
  mlir_tablegen → AscendModel/IR/AscendModelDialect.cpp.inc  (Dialect 注册)
  mlir_tablegen → AscendModel/IR/AscendModelOpsEnums.h.inc   (枚举声明)
  mlir_tablegen → AscendModel/IR/AscendModelOpsEnums.cpp.inc (枚举↔字符串转换)
  ... (AttrDefs ×2, Types ×2)

从 Passes.td:
  mlir_tablegen → AscendModel/Transforms/Passes.h.inc        (Pass 声明)
```

共 **13 个 `.inc` 文件**。它们是代码片段，不是独立头文件——必须嵌入到真正的 `.h`/`.cpp` 中。

---

## 3. 硬件参数层：HardwareConfig

在写每个 Op 的 `estimateCycles()` 实现之前，必须先有 `HardwareConfig`——它是 Op 估算的数据源。每个 Op 的 `estimateCycles(config)` 都要调 `config.getVectorStartupLatency()`、`config.getHBMBandwidthGBs()` 等方法获取硬件参数。

**文件**：`third_party/ascend/costmodel/include/AscendModel/Analysis/HardwareConfig.h`

---

### 3.1 三种核心数据结构

`HardwareConfig` 用三个 struct 表达硬件的层次结构，一一对应 JSON 配置文件的顶层结构：

**MemorySpace** — 存储空间。从 HBM 到寄存器文件，每一级有容量、带宽、延迟：

```cpp
struct MemorySpace {
  std::string name;
  MemoryType type;             // OffChip(HBM) / OnChipShared(L2) / OnChipLocal(L1,UB) / RegisterFile(L0)
  size_t sizeBytes;
  double bandwidthBytesPerCycle;  // 注意单位！不是 GB/s
  int latencyCycles;              // 访问延迟（cycles）
};
// 910B: HBM 32GB/1.6TB/s, L2 192MB, L1 1MB, UB 256KB, L0A/L0B/L0C 各 64~256KB
```

**为什么 bandwidth 存 bytes/cycle 而不是 GB/s？** JSON 里填写时用 GB/s（人可读），但估算 cycle 的公式是 `ceil(bytes / bandwidthBytesPerCycle)`。如果存 GB/s，每次估算都要 `bytes * clockFreq / GBps` 做一次额外运算。在 JSON 解析时一次性转换，后续估算全是直接除法。

**ComputeUnit** — 计算单元。Cube（矩阵乘）和 Vector（SIMD）是两种不同的引擎：

```cpp
struct ComputeUnit {
  std::string name;
  ComputeUnitType type;          // MatrixEngine / SIMDEngine
  double tflopsFP16, tflopsFP32;
  int tileM, tileN, tileK;       // Cube 默认 tile 粒度 16×16×16
  StringMap<FractalSize> fractalSizes;  // 不同 dtype 的 fractal: fp16→16×16×16, fp32→16×8×16
  int widthElements;             // Vector 一次处理 128 个元素
};
```

**DataMover** — 数据搬运引擎。5 个独立的 DMA 通道，各有自己的源/目标存储空间和带宽：

```cpp
struct DataMover {
  std::string name, srcSpace;
  std::vector<std::string> dstSpaces;
  double bandwidthBytesPerCycle;
  int maxBurstBytes, alignmentBytes;
};
// cube_mte2: HBM→L1(200GB/s), mte1: L1→L0A/L0B(400GB/s), fixpipe: L0C→HBM(200GB/s)
// vector_mte2: HBM→UB(200GB/s), mte3: UB→HBM(200GB/s)
```

---

### 3.2 由调用需求倒推查询方法

作者不预先设计"硬件有哪些参数"，而是看**调用者需要什么**。只有两类调用者：

| 调用者 | 调用方法 |
|--------|---------|
| 每个 Op 的 `estimateCycles()`（`AscendModelOps.cpp`） | `getClockFrequencyGHz()`, `getHBMBandwidthGBs()`, `getCubeFractalSize()`, `getCubeStartupLatency()`, `getMTE2StartupLatency()`, `getFixPipeStartupLatency()`, `getMTE3StartupLatency()`, `getVectorStartupLatency()` |
| `PipelineScheduler` + `RooflineAnalyzer`（`PipelineAnalysis.cpp`） | `getName()`, `getAIVScalarOverheadFactor()`, `getPipeBarrierCyclesPerIter()`, `getNumAICCores()`, `getNumAIVCores()`, `getCubeTFlopsFP16()`, `getMemoryBandwidthTBps()`, `cyclesToMicroseconds()` |

每个方法的存在都对应一条实际调用。

---

### 3.3 校准参数：costmodel 的"不准确"修正

最体现 costmodel 工程属性的是这些校准参数——它们**不是** JSON 里的硬件规格，而是从 FlashAttention profiling 数据中拟合出来的修正系数。

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Analysis/HardwareConfig.cpp`

```cpp
int HardwareConfig::getVectorStartupLatency() const { return 35; }
// 从 10 → 35: 反映 dependent vector 指令之间 UB read-after-write 的 stall 惩罚

double HardwareConfig::getAIVScalarOverheadFactor() const { return 3.74; }
// 来源: FlashAttention profiling。AIV 墙钟时间中纯 vector 只占 21%
// factor = (1 - 0.211) / 0.211 = 3.74

int HardwareConfig::getPipeBarrierCyclesPerIter() const { return 7500; }
// 来源: BM=64, 1-wave, 3 次内部迭代
// AIV wall = 59187 cycles, idle = 39% = 23044
// 23044 / 3 ≈ 7500 cycles/barrier

int HardwareConfig::getNumAICCores() const { return 20; }
int HardwareConfig::getNumAIVCores() const { return 40; }
// 来源: profiling 配置 Block Dim=20, Mix Block Dim=40
```

这些值如果换了芯片型号（比如 910C 或 A5），就需要重新 calibrate。HardwareConfig 把它们和 JSON 里的硬件规格放在同一个类里，意味着"这是 910B 特有的行为参数"。

---

### 3.4 JSON 加载流程

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Analysis/HardwareConfig.cpp`

```
HardwareConfig::getDefault910B()
  → 先搜 ASCEND_CONFIG_PATH 环境变量
  → 搜不到就搜 configs/、../configs/ 等标准路径
  → 还搜不到就用 createHardcodedDefault910B() 硬编码 fallback

loadFromFile(path)
  → MemoryBuffer::getFile 读文件
  → json::parse 解析 JSON
  → loadFromJSON(json)
    → 遍历 memory_spaces → MemorySpace
    → 遍历 compute_units → ComputeUnit
    → 遍历 data_movers  → DataMover
    → 遍历 calibration → vectorOpCyclesPerInstruction (查表)
```

**工厂方法**保证调用者拿不到"半成品"——`auto config = HardwareConfig::loadFromFile("ascend_910b.json")` 返回的是一个已解析、已校验的完整对象。外部不需要知道内部的 `parseJSON`、`validate` 步骤。

---

## 4. 实现每个 Op 的 Cycle 估算

有了 Dialect 定义和 HardwareConfig 之后，接下来真正实现每个 Op 的 `estimateCycles()`。

---

### 4.1 胶水层：组织 .inc 文件

TableGen 生成的 13 个 `.inc` 不是独立头文件，必须嵌入到 `.h`/`.cpp` 中。

**`include/AscendModel/IR/AscendModelInterfaces.h`** 包装 Interface 的 .inc：

```cpp
namespace mlir::ascend {
  class HardwareConfig;            // forward declare — 只需要引用类型
  enum class HWUnit : uint32_t;
}
#include "AscendModel/IR/AscendModelInterfaces.h.inc"   // TableGen 生成的 Interface 声明（虚函数+默认实现）
```

**为什么用 forward declare 而不是 `#include "HardwareConfig.h"`？** Interface 的方法只用到 `const HardwareConfig &config`——C++ 的引用参数不需要知道类的完整定义。forward declare 避免 `Interfaces.h` 依赖 `HardwareConfig.h`，打破循环依赖。

**`include/AscendModel/IR/AscendModelDialect.h`** 按依赖顺序组织 6 个 .inc 的 include：

```
① #include "AscendModel/IR/AscendModelInterfaces.h"   (Interface — Op 类依赖它)
② #include "AscendModel/IR/AscendModelDialect.h.inc"   (Dialect 声明)
③ #include "AscendModel/IR/AscendModelOpsEnums.h.inc"  (枚举)
④ #define GET_ATTRDEF_CLASSES, include AttrDefs.h.inc  (属性)
⑤ #define GET_TYPEDEF_CLASSES, include Types.h.inc    (类型)
⑥ #define GET_OP_CLASSES, include Ops.h.inc            (Op 类 — 最后，依赖上面全部)
```

---

### 4.2 AscendModelOps.cpp — 核心估算实现

**文件**：`third_party/ascend/costmodel/lib/AscendModel/IR/AscendModelOps.cpp`

**三类公式**覆盖了所有 Op 的 cycle 估算：

**① Vector 运算** — 适用于 Add/Sub/Mul/Div/Exp/Log/Reduce/Broadcast 等：

```cpp
static int64_t estimateVectorCycles(int64_t numElements, int cyclesPerVectorOp,
                                    int elementBits, int startupLatency) {
  int64_t vectorWidth = 2048 / elementBits;         // FP16: 128, FP32: 64
  int64_t numVectorOps = (numElements + vectorWidth - 1) / vectorWidth;  // ceil
  return numVectorOps * cyclesPerVectorOp + startupLatency;
}
// 例: 256 个 FP32 元素的 add (1 cycle/op):
//   vectorWidth = 2048/32 = 64
//   numVectorOps = ceil(256/64) = 4
//   cycles = 4 * 1 + 35(startup) = 39
```

每个 Op 的 `estimateCycles()` 都调用这个通用公式，唯一的差异是 `cyclesPerVectorOp`：

| Op 类别 | cyclesPerVectorOp | 例 |
|---------|-----------------|-----|
| Add/Sub/Mul/Max/Min/Neg/Abs/Relu/Cast/Cmp | 1 | 简单指令，1 cycle/vector-op |
| Div | 12 | 除法延迟高 |
| Exp | 9 | transcendental，校准后上调 |
| Log | 12 | 迭代近似 |
| Tanh | 18 | 内部多次 exp |
| Sigmoid | 15 | 1/(1+exp(-x)) |
| Broadcast | 直接用 1+startup | 近乎零开销 |

**② 内存搬运** — 适用于 CubeLoadOp/VectorLoadOp/CubeStoreOp/VectorStoreOp：

```cpp
static int64_t estimateMemoryCycles(int64_t bytes, const HardwareConfig &config,
                                    int startupLatency) {
  double bandwidth_gbs = config.getHBMBandwidthGBs();  // 默认 1600 GB/s
  double time_seconds = bytes / (bandwidth_gbs * 1e9);
  double cycles = time_seconds * config.getClockFrequencyGHz() * 1e9;  // 1.85 GHz
  return static_cast<int64_t>(cycles) + startupLatency;
}

// 各搬运 Op 的区别只是 startupLatency 不同：
int64_t CubeLoadOp::estimateCycles(config)   → estimateMemoryCycles(..., getMTE2StartupLatency=50)
int64_t CubeStoreOp::estimateCycles(config)  → estimateMemoryCycles(..., getFixPipeStartupLatency=30)
int64_t VectorLoadOp::estimateCycles(config) → estimateMemoryCycles(..., getMTE2StartupLatency=50)
int64_t VectorStoreOp::estimateCycles(config) → estimateMemoryCycles(..., getMTE3StartupLatency=40)
```

**③ Cube 矩阵乘** — MatmulOp 专用：

```cpp
int64_t MatmulOp::estimateCycles(const HardwareConfig &config) {
  config.getCubeFractalSize(elemBits, fracM, fracN, fracK);  // 按 dtype 取 fractal
  int64_t totalFractals = ceil(m/fracM) * ceil(n/fracN) * ceil(k/fracK);
  return totalFractals + config.getCubeStartupLatency();  // startup=20
}
// 例: FP16 的 M=64,N=128,K=64 → fractal 16×16×16
//   fracM=ceil(64/16)=4, fracN=ceil(128/16)=8, fracK=ceil(64/16)=4
//   totalFractals = 4*8*4 = 128, cycles = 128 + 20 = 148
```

**Macro 批量展开**：Add/Sub/Mul 等结构相同的 Op，作者用宏避免重复：

```cpp
#define IMPL_SIMPLE_VECTOR_BINARY(OpClass)
  int64_t OpClass::estimateCycles(config) {
    return estimateVectorCycles(getNumElements(getLhs().getType()), 1,
                                getElementBitsFromType(getLhs().getType()),
                                config.getVectorStartupLatency());
  }
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }

IMPL_SIMPLE_VECTOR_BINARY(AddOp)
IMPL_SIMPLE_VECTOR_BINARY(SubOp)
IMPL_SIMPLE_VECTOR_BINARY(MulOp)
IMPL_SIMPLE_VECTOR_BINARY(MaxOp)
IMPL_SIMPLE_VECTOR_BINARY(MinOp)
#undef IMPL_SIMPLE_VECTOR_BINARY  // 用完清理
```

**Reduce 运算**的公式更复杂——分三步：

```
① numVectors = ceil(N / (2048/bit))            每向量的计算 (1 cycle/vector)
② vectorReduceCycles = log2(vectorWidth)        向量内的 tree-reduce
③ crossVectorCycles = log2(numVectors)          跨向量的 tree-reduce

total = numVectors + vectorReduceCycles + crossVectorCycles + startup
```

例：8192 个 FP32 元素 ReduceSum → numVectors=128, log2(64)=6, log2(128)=7 → 128+6+7+35 = 176 cycles。

---

## 5. 调度器：从单 Op Cycle 到 Kernel 总时间

第 4 节算出的是**每个 Op 孤立执行**的耗时。但真正的 kernel 里几十个 Op 交错执行——不同硬件单元可以并行，同一单元必须串行，还有数据依赖约束。

**文件**：`third_party/ascend/costmodel/include/AscendModel/Analysis/PipelineAnalysis.h` + `lib/AscendModel/Analysis/PipelineAnalysis.cpp`

---

### 5.1 为什么 sum(Op cycles) ≠ kernel time

假设 3 个 Op：Op0（Vector, 40 cycles）、Op1（Vector, 60 cycles, 依赖 Op0）、Op2（Cube, 100 cycles）：

```
       Vector:  ████████ Op0 ██████████████ Op1 ██████████████
                start=0   end=40  start=40  end=100
       Cube:    ████████████████████████████ Op2 ██████████████████████████████
                start=0                     end=100

kernel total = max(Vector end=100, Cube end=100) = 100
sum(cycles)  = 40 + 60 + 100 = 200 ← 严重高估！
```

**核心观察**：Ascend 910B 是完全流水的架构——所有硬件单元可以同时运行。唯一约束是：(a) 同一单元排队，(b) 数据依赖必须等待。

---

### 5.2 七步设计推演

**Step 1: `PipelineOp`** — 被调度的最小单元。每个 Op 的信息来自 EstimateCyclesPass 写的 IR attributes：

```cpp
struct PipelineOp {
  int64_t opId;                              // AssignOpIDs 给的序号
  HWUnit hwUnit;                             // 在 Cube / Vector / MTE2 / ... 哪个单元跑
  int64_t startCycle, duration, endCycle;    // 调度结果
  SmallVector<int64_t, 4> dependsOn;         // 依赖哪些前置 Op
};
```

**Step 2: `HWUnitPipeline`** — 跟踪一个硬件单元的"忙碌时间线"：

```cpp
class HWUnitPipeline {
  int64_t currentCycle;  // 该单元下次空闲的时间

  void scheduleOp(PipelineOp &op, int64_t earliestStart) {
    op.startCycle = max(currentCycle, earliestStart);
    op.endCycle = op.startCycle + op.duration;
    currentCycle = op.endCycle;  // 更新空闲时间
  }
};
```

**Step 3: `DependencyGraph`** — 邻接表 + 反向边 + Kahn 拓扑排序 + 循环检测：

```cpp
class DependencyGraph {
  DenseMap<int64_t, SmallVector<int64_t, 4>> edges;        // 正向 A→[B,C]
  DenseMap<int64_t, SmallVector<int64_t, 4>> reverseEdges; // 反向 B→[A]
  vector<int64_t> getTopologicalOrder();  // Kahn BFS
  bool hasCycle();                        // 防止死锁
};
```

**Step 4: `PipelineScheduler::schedule()`** — 核心 ASAP 算法：

```cpp
bool PipelineScheduler::schedule() {
  if (depGraph.hasCycle()) return false;                          // ① 拒绝循环依赖
  vector<int64_t> order = depGraph.getTopologicalOrder();         // ② 拓扑排序

  for (int64_t opId : order) {
    PipelineOp &op = operations[opId];
    int64_t earliest = getEarliestStartTime(op);                  // ③ 等所有前驱完成
    pipelines[op.hwUnit].scheduleOp(op, earliest);                // ④ 排到对应硬件管道
    totalCycles = max(totalCycles, op.endCycle);
  }
  return true;
}
```

**Step 5: `getKernelCycles()`** — 从单次执行推到完整 kernel：

```cpp
int64_t PipelineScheduler::getKernelCycles(int64_t numPrograms, int64_t numParallelUnits,
                                           int64_t numInnerIters) const {
  int64_t barrierCycles = numInnerIters * config.getPipeBarrierCyclesPerIter(); // ① barrier 同步
  double scalarFactor = config.getAIVScalarOverheadFactor();                     // ② scalar 开销
  int64_t perProgram = (totalCycles + barrierCycles) * (1.0 + scalarFactor);
  int64_t numWaves = ceil(numPrograms / numParallelUnits);                       // ③ wave 串行
  return perProgram * numWaves;
}
```

三种外推对应三种硬件现实开销：barrier（每次循环迭代间的 pipe_barrier 同步）、scalar overhead（循环控制、地址计算的标量指令）、wave serialisation（多个程序块排队到有限物理核上）。

**Step 6: `PerformanceReport`** — 结构化输出：

```cpp
struct PerformanceReport {
  int64_t totalCycles;        // 关键路径长度
  double totalTimeUs;         // 微秒数 = cycles / (GHz * 1000)
  int64_t kernelTotalCycles;  // 含 overhead
  map<HWUnit, double> unitUtilization;  // 各单元利用率
  HWUnit bottleneckUnit;      // 瓶颈单元
  double arithmeticIntensity; // FLOP/Byte
  bool isComputeBound;        // 计算受限 vs 带宽受限
  void print(raw_ostream &os);  // 人类可读
  string toJSON();              // 机器解析
};
```

**Step 7: `RooflineAnalyzer`** — 瓶颈分析（独立类，方便后续换分析方法）：

```cpp
bool RooflineAnalyzer::isComputeBound() const {
  double ai = totalFLOPs / totalBytes;          // 算术强度
  double ridgePoint = peakTFLOPS / peakBWTBps;  // Roofline 转折点
  return ai >= ridgePoint;                      // AI ≥ 转折点 → 计算受限
}
// 910B: ridge = 320 TFLOPS / 1.6 TB/s = 200 FLOP/Byte
// kernel AI = 50 → 内存受限; kernel AI = 500 → 计算受限
```

---

### 5.3 调度实例

3 个 Op：Op0（Vector, 40）、Op1（Vector, 60, 依赖 Op0）、Op2（Cube, 100）

```
Step 1: 拓扑序 = [0, 1, 2]（0 无依赖，1 依赖 0，2 无依赖）
Step 2: 调度 Op0: Vector 管道空闲, 依赖完成=0 → start=0, end=40
Step 3: 调度 Op1: Vector 管道 busy til 40, 依赖 Op0.end=40 → start=max(40,40)=40, end=100
Step 4: 调度 Op2: Cube 管道空闲, 依赖完成=0 → start=0, end=100
Step 5: totalCycles = max(40, 100, 100) = 100
```

---

## 6. Pass 管线：把估算嵌入编译流程

前面 2~5 节是"零件"——Dialect、硬件参数、Op 估算、调度器。这一节是把这些零件**组装成编译器 Pass 管线**，让它们能自动从 TTIR 一路跑到预估延迟。

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Transforms/`

---

### 6.1 6 个 Pass 的职责和数据流

`ascend-perf-model` 管线在 `PassRegistration.cpp` 中注册，依次运行：

```
① ConvertTritonToAscend:  TTIR Op → AscendModel Op
② InsertDataTransfers:    在两个 Compute 路径间插入显式搬运 Op
③ AssignOpIDs:            给每个 Op 写 op_id = 0, 1, 2, ...
④ EstimateCycles:         调每个 Op 的 estimateCycles(config) → 写 attributes
⑤ PipelineAnalysis:       读 attributes → PipelineScheduler → 写 module 属性
⑥ PerfReport:             汇总 → PerformanceReport → 打印
```

**Pass 之间不通过函数参数传递数据，而是通过 IR 上的 attributes**：

```
Step ③: ascend.add {op_id = 5}
Step ④: ascend.add {op_id = 5, estimated_cycles = 57, hw_unit = "Vector", flops = 1024}
Step ⑤: module attributes {total_cycles = 60320, roofline_cycles = 45200, ...}
Step ⑥: 读所有 attributes → PerformanceReport
```

---

### 6.2 管线组装

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Transforms/PassRegistration.cpp`

```cpp
void registerAscendModelPipeline() {
  PassPipelineRegistration<AscendPerfModelPipelineOptions>(
      "ascend-perf-model",
      "Run the full Ascend 910B performance modeling pipeline",
      [](OpPassManager &pm, const AscendPerfModelPipelineOptions &options) {
        pm.addPass(createConvertTritonToAscendPass());
        pm.addPass(createInsertDataTransfersPass());
        pm.addPass(createAssignOpIDsPass());
        pm.addPass(createEstimateCyclesPass(estimateOpts));
        pm.addPass(createPipelineAnalysisPass(pipelineOpts));
        pm.addPass(createPerfReportPass());
      });
}
```

选项（arg-bindings、hardware-config）在 Step ④ 和 Step ⑤ 两处被消费——EstimateCycles 用它解析循环 trip count，PipelineAnalysis 用它做调度分析。

---

### 6.3 ConvertTritonToAscend —— TTIR → AscendModel IR（850 行）

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Transforms/ConvertTritonToAscend.cpp`

这是 Pass 管线中第一个、也是最复杂的 Pass。它的任务是将 Triton 方言的 Op（`tt.load`、`tt.dot`、`tt.store` 等）转换为 AscendModel 方言的 Op（`ascend.vector_load`、`ascend.matmul`、`ascend.vector_store` 等），同时转换 arith/math 等通用 MLIR Op。

**Pass 框架结构**：每个 MLIR Pass 继承 TableGen 生成的 Base 类，覆写 `runOnOperation()`。通过 X-Macro 完成静态注册：

```cpp
// ① 声明这个 .cpp 文件要实现哪个 Pass
#define GEN_PASS_DEF_CONVERTTRITONTOASCENDPASS
#include "AscendModel/Transforms/Passes.h.inc"     // 展开 Base 类 + 注册代码

// ② 继承生成的 Base，覆写 runOnOperation
struct ConvertTritonToAscendPass
    : public impl::ConvertTritonToAscendPassBase<ConvertTritonToAscendPass> {
  using ConvertTritonToAscendPassBase::ConvertTritonToAscendPassBase;
  void runOnOperation() override { /* Pass 逻辑 */ }
};
```

`GEN_PASS_DEF_CONVERTTRITONTOASCENDPASS` 告诉 `Passes.h.inc`："对于 `ConvertTritonToAscendPass`，这次展开它的 Base 实现和注册代码"。

---

**Phase 1 — 分析阶段：标记哪些 load 给 Cube 用**

TTIR 中用 `tt.load` 做通用加载。但 MatrixMul（`tt.dot`）的输入需要走 `cube_load`（MTE2→L1 路径），普通 Vector 操作的输入走 `vector_load`（MTE2→UB 路径）。每个 `tt.load` 不知道自己被谁消费。

Phase 1 在全局层面建立这个映射：

```cpp
module.walk([&](Operation *op) {
  if (op名 != "tt.dot") return;
  for (Value operand : op->getOperands()) {
    Value current = operand;
    // 沿 use-def 链回溯，穿透 tt.trans / tt.bitcast / tt.reshape / tt.expand_dims
    while (Operation *def = current.getDefiningOp()) {
      if (def名 == "tt.load") {
        def->setAttr("ascend.used_by_dot", UnitAttr::get(ctx));  // ← 打标记
        break;
      }
      if (def名是 tt.trans/tt.bitcast/tt.reshape/tt.expand_dims)
        current = def->getOperand(0);  // 穿透形变 Op，继续回溯
      else break;  // 遇到非形变非 load 的 Op，停止回溯
    }
  }
});
```

这个标记在 Phase 2 的 `ConvertTritonLoad` 中被消费——有标记的 load 转为 `cube_load`，无标记的转为 `vector_load`。

---

**Phase 2 — 模式替换：12 种 RewritePattern，两级 benefit**

MLIR 的 `applyPatternsGreedily` 按 benefit 降序应用规则，直到 IR 不再变化。benefit=10 先执行（结构转换），benefit=1 后执行（tail cleanup）：

**benefit=10 — Triton Op → AscendModel 结构转换（5 个模式）**

| 模式 | 输入 | 输出 | 关键逻辑 |
|------|------|------|---------|
| `ConvertTritonDot` | `tt.dot(%a, %b, %c)` | `ascend.matmul` + `cube_store` + (可选) `vector_load` | 沿 use-def 链判断输入来自 Vector 还是 Cube 路径；非 Cube 输入自动插入 `vector_store` + `cube_load` 中转；输出侧检查消费者，如果被 Vector Op 使用则加 `cube_store` + `vector_load` |
| `ConvertTritonLoad` | `tt.load` | `ascend.cube_load` 或 `ascend.vector_load` | 读取 Phase 1 写的 `ascend.used_by_dot` 属性决定转换目标 |
| `ConvertTritonStore` | `tt.store` | `ascend.vector_store` | 丢弃指针 operand（TTIR 用 `!tt.ptr<f32>` 类型），只保留数据 operand |
| `ConvertTritonReduce` | `tt.reduce` | `ascend.reduce_{sum\|max\|min\|prod}` | 检查归约体，确定 kind：含 max→ReduceMax, 含 min→ReduceMin, 含 mul→ReduceProd, 默认→ReduceSum |
| `ConvertTritonTrans` | `tt.trans` | pass-through（替换为输入值） | Transpose 在 Ascend 硬件上零开销——地址计算被 DMA descriptor 吸收 |
| `ConvertTritonBroadcast` | `tt.broadcast` | `ascend.broadcast` | — |

**ConvertTritonDot 的数据流逻辑**（最复杂的单个模式，~120 行）：

对于 dot 的 lhs 和 rhs，调用 `ensureCubeInput()`：
```cpp
auto ensureCubeInput = [&](Value operand) -> Value {
  if (isFromLoad(operand)) return operand;       // 已被 ConvertTritonLoad 处理，直接用
  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  // 创建 VectorStore + CubeLoad 中转:
  rewriter.create<VectorStoreOp>(loc, operand, bytes, ...);  // UB → HBM (MTE3)
  auto cubeLoad = rewriter.create<CubeLoadOp>(loc, ...);     // HBM → L1  (MTE2)
  return cubeLoad.getResult();
};
```

产物示例：`tt.dot(%lhs_from_arith, %rhs_from_load)` 变成：

```
%lhs_vs = ascend.vector_store %lhs_from_arith ...      // 从 UB 搬出到 HBM
%lhs_cl = ascend.cube_load %lhs_vs ...                  // 从 HBM 搬入到 L1
%dot = ascend.matmul %lhs_cl, %rhs_from_load ...        // Cube 计算
ascend.cube_store %dot ...                              // L0C → HBM (FixPipe)
%vl = ascend.vector_load %dot ...                       // HBM → UB (MTE2) — 输出给 Vector 消费者
```

**benefit=1 — 清理 + arith/math 转换（7 个模式）**

| 模式 | 输入 | 输出 |
|------|------|------|
| `EraseDeadTritonAddrOps` | `tt.addptr`/`tt.splat`/`tt.make_range`/`tt.int_to_ptr`/`tt.ptr_to_int`/`tt.expand_dims`/`tt.bitcast`/`tt.trans`/`tt.reshape`（全部 results 无使用者） | 删除（硬件零开销，已被前面的 load/store 转换孤立） |
| `ConvertArithBinaryOp` | `arith.addf`/`subf`/`mulf`/`divf`/`maxf`/`minf` + `addi`/`subi`/`muli`/`divsi`/`divui`/`maxsi`/`maxui`/`minsi`/`minui` | `ascend.add`/`sub`/`mul`/`div`/`max`/`min` |
| `ConvertArithCmpOp` | `arith.cmpf`/`cmpi` | `ascend.cmp_eq`/`ne`/`lt`/`le`/`gt`/`ge`（映射 float 的 6 种 predicate + int 的 6 种 predicate） |
| `ConvertMathUnaryOp` | `math.exp`/`exp2`/`log`/`log2`/`sqrt`/`rsqrt`/`tanh`、`arith.negf`、`math.absf`/`absi` | `ascend.exp`/`log`/`sqrt`/`rsqrt`/`tanh`/`neg`/`abs` |
| `ConvertArithSelect` | `arith.select` | `ascend.select` |
| `ConvertArithCast` | `arith.extf`/`extsi`/`extui`/`truncf`/`trunci`/`sitofp`/`uitofp`/`fptosi`/`fptoui`/`bitcast`/`index_cast` | 全部 → `ascend.cast`（在 costmodel 视角下，所有类型转换都是 1 cycle/vector-op） |

**Fallback 机制**：`#ifdef TRITONSIM_HAS_TRITON` 编译选择。如果有 Triton 头文件就用类型安全的 `dyn_cast<triton::DotOp>`，否则用字符串匹配 `isOpNamed(op, "tt.dot")`。实现同样的逻辑但适配不同的编译环境。

**Greedy 重写的级联效应**：
```
Pass 1: ① tt.load → cube_load/vector_load  (benefit=10)
        ② tt.dot → matmul + cube_store       (benefit=10)
        ③ tt.store → vector_store            (benefit=10)
        ④ tt.addptr/tt.splat → 孤立，无使用者
Pass 2: ⑤ EraseDead 删除孤立 Op             (benefit=1)
        ⑥ arith.addf → ascend.add            (benefit=1)
        ⑦ math.exp → ascend.exp              (benefit=1)
```
benefit=10 的模式先打碎 Triton 的结构，产生孤立 Op；benefit=1 的 EraseDead 下一轮再清理。`applyPatternsGreedily` 自动迭代直到收敛。

---

---

### 6.4 EstimateCycles —— 核心估算 Pass

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Transforms/EstimateCycles.cpp`

这个 Pass 是 costmodel 的核心：遍历 IR 中所有 AscendModel Op，调用各自实现的 `estimateCycles()`，把结果写回 IR 的 attributes，同时收集 Roofline 统计数据。共三遍遍历。

**Pass 框架**：同 6.3，通过 `GEN_PASS_DEF_ESTIMATECYCLESPASS` → `#include "Passes.h.inc"` 注册。Pass 接受三个 option：

```cpp
// Passes.td 中定义，TableGen 生成 EstimateCyclesPassOptions struct
Option<"argBindingsStr", "arg-bindings", "std::string", "\"\"", "...">
Option<"loopTripCountsStr", "loop-trip-counts", "std::string", "\"\"", "...">
Option<"hardwareConfigPath", "hardware-config", "std::string", "\"\"", "...">
```

`runOnOperation()` 入口先加载 HardwareConfig、解析两种绑定参数：

```cpp
auto hardwareConfig = loadHardwareConfigForAnalysis(hardwareConfigPath, error);
// 解析 "arg-bindings=arg2=128,pid_x=0" → {2: 128, "x": 0}
parseBindings(argBindingsStr, argBindings, programIdBindings, ...);
// 解析 "loop-trip-counts=4,6588" → [4, 6588]
parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides, ...);
```

---

**第 1 遍 — 解析所有循环的 trip count**

收集所有 `scf::ForOp`，按出现顺序分配 trip count。两类来源：

```cpp
SmallVector<scf::ForOp> allLoops;
module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });

for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
  if (loopIdx < loopTripCountOverrides.size())
    tripCount = loopTripCountOverrides[loopIdx];               // 直接覆盖（优先）
    source = "override";
  else {
    // 用 arg-bindings 解析循环边界表达式: lower=0, upper=%arg4, step=%arg5
    auto result = getScfForTripCountWithBindings(forOp, argBindings, programIdBindings);
    if (result.isStatic) {
      tripCount = result.staticTripCount;                      // 成功算出
      source = "evaluated";
    } else {
      emitError("无法确定 trip count: " + result.errorMsg);    // 绑定不够
    }
  }
  // 把结果写回 forOp 的属性
  forOp->setAttr("ascend.trip_count", IntegerAttr::get(..., tripCount));
}
```

第一步优先使用显式覆盖（`loop-trip-counts`），第二步再用符号解析（`arg-bindings`），第三步如果都无法确定就报错。

---

**第 2 遍 — 估算每个 Op 的 cycle，同时收集 Roofline 统计**

遍历所有 Op，跳过 `scf::ForOp`/`scf::YieldOp`/`scf::IfOp`（它们不是计算 Op，不需要估算）。对每个 Compute Op：

```cpp
module.walk([&](Operation *op) {
  if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op)) return;

  if (auto cyclesOp = dyn_cast<EstimateCyclesOpInterface>(op)) {
    int64_t cycles = cyclesOp.estimateCycles(config);    // ① 单次执行 cycle
    HWUnit hwUnit = cyclesOp.getHWUnit();                // ② 在哪个硬件单元
    int64_t loopMultiplier = getLoopMultiplier(op);      // ③ 乘以循环嵌套倍数
    int64_t totalOpCycles = cycles * loopMultiplier;     // ④ 这个 Op 执行的总额外开销

    // 写入 Op 属性:
    op->setAttr("estimated_cycles", ...);                // 单次 cycle
    op->setAttr("hw_unit", stringifyHWUnit(hwUnit));     // "Cube"/"Vector"/...
    if (bytes > 0) op->setAttr("bytes", ...);            // 搬运量（搬运 Op）
    if (flops > 0) op->setAttr("flops", ...);            // 计算量（计算 Op）
    if (loopMultiplier > 1) op->setAttr("loop_multiplier", ...);

    // 累计 Roofline 统计（按 HWUnit 分类）:
    switch (hwUnit) {
      case HWUnit::Cube:     stats.cubeFlops += flops;     stats.cubeCycles     += totalOpCycles; break;
      case HWUnit::CubeMTE2: stats.cubeLoadBytes += bytes; stats.cubeLoadCycles  += totalOpCycles; break;
      case HWUnit::FixPipe:  stats.cubeStoreBytes += bytes; stats.cubeStoreCycles += totalOpCycles; break;
      case HWUnit::Vector:   stats.vectorFlops += flops;   stats.vectorCycles    += totalOpCycles; break;
      case HWUnit::VecMTE2:  stats.vectorLoadBytes += bytes; stats.vectorLoadCycles += totalOpCycles; break;
      case HWUnit::MTE3:     stats.vectorStoreBytes += bytes; stats.vectorStoreCycles += totalOpCycles; break;
    }
  }
});
```

**`getLoopMultiplier()` 处理循环嵌套**。如果 Op 在两层循环内——外层 trip=4、内层 trip=128——那 multiplier = 4 × 128 = 512。它沿 parent 链往上走，累积所有 `scf::ForOp` 上之前写的 `ascend.trip_count`：

```cpp
int64_t getLoopMultiplier(Operation *op) {
  int64_t multiplier = 1;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      if (auto attr = forOp->getAttr("ascend.trip_count"))
        multiplier *= attr.getInt();
    parent = parent->getParentOp();
  }
  return multiplier;
}
```

**Roofline 模型**：按 Cube 路径和 Vector 路径分别取 `max(compute, load, store)`，如果 Cube 和 Vector 可以并行就取 `max(cubePath, vectorPath)`：

```cpp
struct RooflineStats {
  int64_t calculateRooflineCycles(const HardwareConfig &config, bool overlap) const {
    int64_t cubePath = max({cubeCycles, cubeLoadCycles, cubeStoreCycles});
    int64_t vectorPath = max({vectorCycles, vectorLoadCycles, vectorStoreCycles});
    return overlap ? max(cubePath, vectorPath) : cubePath + vectorPath;
  }
};

int64_t rooflineCycles = stats.calculateRooflineCycles(config, true);
int64_t simpleSumCycles = cubeCycles + cubeLoadCycles + ...;  // 对比用（最悲观）
```

---

**第 3 遍 — 标注循环体**

最后给每个 `scf::ForOp` 附上 body 和总体的 cycle 统计：

```cpp
module.walk([&](scf::ForOp forOp) {
  int64_t tripCount = forOp->getAttr("ascend.trip_count").getInt();
  int64_t bodyCycles = 0;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op)) continue;
    if (auto cyclesAttr = op.getAttr("estimated_cycles"))
      bodyCycles += cyclesAttr.getInt();
  }
  forOp->setAttr("ascend.body_cycles", bodyCycles);
  forOp->setAttr("ascend.total_cycles", bodyCycles * tripCount);
});
```

最后将全局统计写入 Module 属性：

```cpp
module->setAttr("ascend.roofline_cycles", rooflineCycles);
module->setAttr("ascend.simple_sum_cycles", simpleSumCycles);
module->setAttr("ascend.total_ops", totalOps);
module->setAttr("ascend.hardware", config.getName());
```

这些 Module 属性被后续 `PipelineAnalysisPass` 和 `PerfReportPass` 读取。

---

### 6.5 PipelineAnalysisPass —— 调度分析

**文件**：`third_party/ascend/costmodel/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp`

这个 Pass 读 EstimateCycles 写在 Op 上的 attributes（`estimated_cycles`、`hw_unit`、`flops`、`bytes`），构建依赖图，运行 PipelineScheduler，把调度结果写回 Module 属性。

核心流程：

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();

  // ① 构建 PipelineOp 列表：从 IR attributes 读回所有已估算的 Op
  std::vector<PipelineOp> ops;
  module.walk([&](Operation *op) {
    if (!op->hasAttr("estimated_cycles")) return;
    PipelineOp pop;
    pop.opId = op->getAttrOfType<IntegerAttr>("op_id").getInt();
    pop.hwUnit = symbolizeHWUnit(op->getAttrOfType<StringAttr>("hw_unit").getValue());
    pop.duration = op->getAttrOfType<IntegerAttr>("estimated_cycles").getInt();
    pop.bytes = op->getAttrOfType<IntegerAttr>("bytes").getValueOr(0);
    pop.flops = op->getAttrOfType<IntegerAttr>("flops").getValueOr(0);
    pop.mlirOp = op;
    ops.push_back(pop);
  });

  // ② 建依赖图：从 MLIR use-def 链 + 循环嵌套推断依赖关系
  PipelineScheduler scheduler(&config);
  for (auto &op : ops) {
    scheduler.addOperation(op);
    // 同一 HWUnit 上的 Op 之间：前一个 → 后一个（硬件串行）
    // MLIR SSA use-def：消费者依赖生产者
    for (Value operand : op.mlirOp->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (auto idAttr = defOp->getAttrOfType<IntegerAttr>("op_id"))
          op.dependsOn.push_back(idAttr.getInt());
      }
    }
  }

  // ③ 执行调度
  scheduler.schedule();

  // ④ 写回 Module 属性
  module->setAttr("ascend.scheduled_cycles_one_iter", scheduler.getTotalCycles());
  module->setAttr("ascend.roofline_cycles", 从 EstimateCycles 传下来的 roofline 值);
  module->setAttr("ascend.simple_sum_cycles", 从 EstimateCycles 传下来的简单求和值);
}
```

**RooflineAnalyzer 的运行时机**：在 `schedule()` 之后，调度器中已有了所有 Op 的排序结果和总 cycle，RooflineAnalyzer 读取这些数据做瓶颈分析：

```cpp
RooflineAnalyzer analyzer(scheduler);
PerformanceReport report = analyzer.analyze();
// report 中填充: totalCycles, unitUtilization, bottleneckUnit,
// arithmeticIntensity, achievedTFLOPS, isComputeBound, ...
```

**依赖图的关键规则**：

1. **同一 HWUnit 上的 Op**：前一个结束才能开始后一个（`start >= prevOp.endCycle`）
2. **MLIR SSA use-def**：如果 Op B 的 operand 由 Op A 产生，B 依赖 A
3. **循环嵌套**：内层循环的 Op 不会和外层 Op 竞争同一个 HWUnit 的时间——已在 EstimateCyclesPass 中通过 `loopMultiplier` 处理

---

### 6.6 其他 Pass：InsertDataTransfers / AssignOpIDs / PerfReport

**InsertDataTransfersPass**（`third_party/ascend/costmodel/lib/AscendModel/Transforms/InsertDataTransfers.cpp`）

解决 Cube 和 Vector 路径之间的数据孤岛问题。910B 上 Cube 用 L1→L0A/L0B→L0C，Vector 用 UB。当 Vector 的计算结果流向 Cube 时：

```cpp
// 遍历所有 MatmulOp
for (MatmulOp matmul : matmulOps) {
  // 检查每个 operands：如果生产者是 Vector 路径 → 插入 vector_store + cube_load
  for (Value operand : {matmul.getLhs(), matmul.getRhs()}) {
    if (getComputePath(defOp) == Vector) {
      builder.create<VectorStoreOp>(loc, operand, bytes, ...);    // UB → HBM (MTE3)
      auto cubeLoad = builder.create<CubeLoadOp>(loc, ...);       // HBM → L1 (MTE2)
      matmul->setOperand(i, cubeLoad.getResult());
    }
  }
  // 检查 MatmulOp 的消费者：如果是 Vector 路径 → 插入 cube_store + vector_load
  for (OpOperand &use : matmulResult.getUsers()) {
    if (getComputePath(user) == Vector) {
      builder.create<CubeStoreOp>(loc, matmulResult, bytes, ...);   // L0C → HBM (FixPipe)
      auto vecLoad = builder.create<VectorLoadOp>(loc, ...);         // HBM → UB (MTE2)
      use->set(vecLoad.getResult());
    }
  }
}
```

**AssignOpIDsPass**（49 行，最简单）：给所有 `ascend` namespace 的 Op 从 0 开始编号，写 `{op_id = 0, 1, 2, ...}`。

**PerfReportPass**（~80 行）：读所有 attributes，组装 `PerformanceReport`，调 `report.print()` 输出人类可读报告、`report.toJSON()` 导 JSON。它不写任何 IR 属性——只读不写。

---

## 7. Python 桥接层

C++ Pass 管线写好了，但 Triton JIT 编译器是 Python 进程。需要一个通道让 Python 调用 C++ costmodel。

---

### 7.1 run_costmodel_inproc — C++ 侧入口

**文件**：`third_party/ascend/triton_ascend.cc`

这个函数是 `costmodel_runtime.py` 中 `run_costmodel()` 的 C++ 实现，通过 pybind11 暴露：

```cpp
// 手动构建 PassManager（不依赖 registerAscendModelPipeline）
mlir::PassManager pm(&context);
pm.addPass(mlir::ascend::createConvertTritonToAscendPass());
pm.addPass(mlir::ascend::createInsertDataTransfersPass());
pm.addPass(mlir::ascend::createAssignOpIDsPass());
pm.addPass(mlir::ascend::createEstimateCyclesPass(estimateOpts));
pm.addPass(mlir::ascend::createPipelineAnalysisPass(pipelineOpts));
// 注意: 没加 PerfReportPass — Python 侧不需要 JSON

if (mlir::failed(pm.run(*module))) throw ...;

// 从 Module 上读取 PipelineAnalysisPass 写入的属性
const double timeUs = extractEstimatedTimeUs(*module);
std::ostringstream os;
os << "Estimated Time: " << timeUs << " us";   // ← parse_latency() 的目标字符串
return os.str();
```

Python 侧的 `parse_latency()` 用正则从这行输出中提取浮点数：

```python
# costmodel_runtime.py
def parse_latency(output: str) -> float:
    match = re.search(r'Estimated Time:\s*([\d.]+)\s*us', output)
    return float(match.group(1)) if match else float('inf')
```

---

### 7.2 Python 运行时层

**文件**：`third_party/ascend/backend/runtime/costmodel_runtime.py`

这一层提供缓存、并行评估、主入口等胶水逻辑：

**核心函数调用链**：

```
costmodel_bench(items)              # 主入口 — 尚未接入 autotune，但函数已就绪
  ├─ _normalize_costmodel_items()   # 规范化输入 [{config, ttir}] → [(config, ttir, args)]
  ├─ _evaluate_pending_items()      # 线程池并行执行
  │    └─ _eval_one_costmodel_item() # 单 item 执行
  │         ├─ make_costmodel_cache_key()  # ttir + args → SHA256 hash
  │         ├─ load_costmodel_latency()    # 文件缓存命中就返回
  │         └─ run_costmodel()             # 缓存 miss → 真正调 C++
  │              └─ ascend_capi.run_costmodel_inproc(mlir_text, args)
  │                   └─ 第 7.1 节的 C++ Pass 管线
  └─ parse_latency(output) → float  # 正则解析 C++ 输出
```

**costmodel_bench() 的设计意图**：接受 `[{config, ttir, arg_bindings}]` 格式的输入，返回 `{config: latency_us}` 映射。autotune 在遍历候选 config 时调用它做预筛选——为每个 config 生成 TTIR、调 costmodel 估算延迟、按延迟排序后只编译 top-k。**目前函数已就绪但尚未接入 autotune 流程**。

---

## 8. 端到端示例

以下演示两个例子：纯 Vector（vecadd TTIR）和 Cube+Vector 混合，覆盖两种计算路径。

---

### 8.1 例子一：纯 Vector 路径 — vecadd.mlir

**输入**：`test/Triton/vecadd.mlir`，一个标准 Triton 向量加法 kernel 的 TTIR：

```
tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, ...) {
  %0 = tt.get_program_id x : i32                    ← SPMD: "我是第几个 program"
  %2 = tt.make_range {end=256, start=0} : tensor<256xi32>
  %3 = tt.splat %1 : i32 -> tensor<256xi32>
  %4 = arith.addi %3, %2 : tensor<256xi32>           ← 地址计算（零硬件开销）
  %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>
  scf.for %arg6 = 0 to %arg4 step 32 {               ← 循环, 每次迭代处理 32 元素
    %19 = tt.load ... : tensor<32xf32>               ← 从 HBM 加载
    %21 = tt.load ... : tensor<32xf32>
    %22 = arith.addf %19, %21 : tensor<32xf32>       ← 向量加法
    tt.store %22 ... : tensor<32xf32>                ← 存回 HBM
  }
}
```

**运行**：

```python
from triton._C.libtriton import ascend

# 读 TTIR 文本
with open("test/Triton/vecadd.mlir") as f:
    ttir = f.read()

# 调 C++ Pass 管线。arg-bindings 绑定动态循环边界
result = ascend.run_costmodel_inproc(ttir, [
    "-ascend-perf-model",
    "-ascend-perf-model=arg-bindings=arg4=1024,arg5=256",
    "-allow-unregistered-dialect",
])
print(result)
```

**输出解读**：Pass 管线内部执行流程：

```
ConvertTritonToAscend: tt.load → ascend.vector_load, arith.addf → ascend.add,
                       tt.store → ascend.vector_store, tt.addptr/tt.splat → 删除
InsertDataTransfers:  (纯 Vector 路径，无需跨路径搬运)
AssignOpIDs:          给每个 ascend.* Op 分配 op_id
EstimateCycles:       每个 Op 计算 cycle: vector_load + add + vector_store × trip_count
PipelineAnalysis:     Vector/MTE2/MTE3 并行调度 → total_cycles
→ "Estimated Time: X.XX us"
```

---

### 8.2 例子二：Cube + Vector 混合

这个例子演示跨 Computation 路径的数据搬运——Cube 算矩阵乘的结果，Vector 做 add。

**输入**：手写 AscendModel IR（跳过 TTIR 转换，聚焦跨路径行为）：

```python
mlir = """
module {
  func.func @cube_vector_mix(
      %arg0: tensor<64x64xf16>, %arg1: tensor<64x128xf16>,
      %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // Cube path: HBM -(cube_mte2)-> L1 -(mte1)-> L0A/L0B -> Cube -> L0C -(fixpipe)-> HBM
    %lhs = ascend.cube_load %arg0 {bytes = 8192 : i64} : tensor<64x64xf16> -> tensor<64x64xf16>
    %rhs = ascend.cube_load %arg1 {bytes = 16384 : i64} : tensor<64x128xf16> -> tensor<64x128xf16>
    %dot = ascend.matmul %lhs, %rhs {M = 64 : i64, N = 128 : i64, K = 64 : i64}
        : (tensor<64x64xf16>, tensor<64x128xf16>) -> tensor<64x128xf32>
    ascend.cube_store %dot {bytes = 32768 : i64} : tensor<64x128xf32>

    // Vector path: HBM -(vec_mte2)-> UB -> Vector -> UB -(mte3)-> HBM
    %vec_in = ascend.vector_load %arg2 {bytes = 32768 : i64} : tensor<64x128xf32> -> tensor<64x128xf32>
    %result = ascend.add %dot, %vec_in : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    ascend.vector_store %result {bytes = 32768 : i64} : tensor<64x128xf32>
    return %result : tensor<64x128xf32>
  }
}
"""

result = ascend.run_costmodel_inproc(mlir, ["-ascend-perf-model", "-allow-unregistered-dialect"])
print(result)
```

**关键观察：** `%dot` 在 Cube 路径上产生（L0C），但 `ascend.add` 在 Vector 路径上消费（需要 UB）。`InsertDataTransfersPass` 会在两步之间自动插入 `cube_store` + `vector_load`，把数据从 L0C → HBM → UB。最终 Cycle 数 = `max(CubePath, VectorPath)`（两者可以在不同硬件单元上并行）。

**逐步查看每个 Pass 的输出：** `triton-opt` 当前没有注册 AscendModel 的 Pass（`bin/RegisterTritonDialects.h` 缺少 `#include "AscendModel/Transforms/Passes.h"` 和 `registerAllAscendModelPasses()` 调用）。在补上这行之前，可以通过 Python 分步运行单个 Pass 来观察中间结果：

```python
from triton._C.libtriton import ascend

mlir = """<上面那段 IR 文本>"""

# ① ConvertTritonToAscend — 输入是 AscendModel IR，no-op
result = ascend.run_costmodel_inproc(mlir, [
    "-ascend-perf-model=hardware-config=configs/ascend_910b.json",
    "-allow-unregistered-dialect",
])
print(result)
```

`run_costmodel_inproc` 内部固定跑①-⑤全部 Pass，不支持逐步运行。如果需要逐步观察每个 Pass 的输出，需要补上 `RegisterTritonDialects.h` 中缺失的注册：

```cpp
// bin/RegisterTritonDialects.h 需要添加:
#include "AscendModel/Transforms/Passes.h"

// 在 registerTritonDialects 函数中添加:
mlir::ascend::registerAllAscendModelPasses();
```

补上后即可用 `triton-opt` 逐步运行：
```bash
triton-opt cube_mix.mlir --convert-triton-to-ascend --insert-data-transfers \
    --assign-op-ids --estimate-cycles --analyze-pipeline --perf-report
```

---

### 8.3 TTIR 是怎么产出的

```
Python kernel (@triton.jit)
  → ASTSource.make_ir()
    → ast_to_ttir()                       ← Python AST → raw TTIR
  → stages["ttir"] = make_ttir()          ← inline + CSE + LICM + loop unfold → 优化 TTIR
  → TTIR 文本                              ← 这就是 costmodel 的输入
```

在实际 Triton 编译流程中，TTIR 是 `make_ir()` → `ast_to_ttir()` 的产物。如果你设置了 `TRITON_DUMP_DIR` 环境变量，`make_ttir()` 会把优化后的 TTIR dump 到 `kernel.ttir.mlir` 文件。

---

## 9. 全栈数据流总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 层                                                      │
│  Triton kernel → ast_to_ttir() → TTIR 文本                      │
│  costmodel_bench([{config, ttir, arg_bindings}])               │
│       │                    ↑                                    │
│       │  run_costmodel()   │  parse_latency("Estimated: X us")  │
│       ▼                    │                                    │
├───────────────────────────┼────────────────────────────────────┤
│  pybind11: ascend_capi.run_costmodel_inproc(mlir_text, args)    │
├───────────────────────────┼────────────────────────────────────┤
│  C++ Pass 管线                                                  │
│  ① ConvertTritonToAscend   tt.load → ascend.vector_load        │
│  ② InsertDataTransfers     Vector↔Cube 跨路径搬运               │
│  ③ AssignOpIDs             op_id = 0, 1, 2, ...                │
│  ④ EstimateCycles          每个 Op::estimateCycles(config)       │
│  ⑤ PipelineAnalysis         PipelineScheduler::schedule()        │
│  ⑥ PerfReport               PerformanceReport::print()          │
├───────────────────────────┼────────────────────────────────────┤
│  数据源                                                         │
│  HardwareConfig ← ascend_910b.json + 校准参数                    │
│  AscendModelOps  ← 25 个 Op 的 cycle 公式                       │
└─────────────────────────────────────────────────────────────────┘
```

每一层的职责清晰：Dialect 定义"什么 Op 存在" → HardwareConfig 定义"硬件长什么样" → Ops.cpp 定义"每个 Op 多少 cycle" → Scheduler 定义"Op 之间怎么排" → Pass 把前面串联起来 → Python 桥接到编译流程。各层之间通过接口解耦（Op 只依赖 HardwareConfig 的查询方法、Pass 只依赖 Op 的 Interface），改任意一层不需要动其他层。

