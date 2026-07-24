# CostModel 代码完整走读

> 基于设计文档《Triton-Ascend 原生接入 Costmodel》+ 仓库源码，逐层深入。

---

## 总览：三层架构

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Python 运行时                          │
│   costmodel_runtime.py                          │
│   负责: autotuner 调用入口、缓存、多线程           │
├─────────────────────────────────────────────────┤
│ Layer 2: C++ 桥接层                             │
│   triton_ascend.cc (run_costmodel_inproc)        │
│   PassRegistration.cpp (ascend-perf-model 管线)  │
│   负责: 解析 MLIR → 跑 6 个 Pass → 返回 us       │
├─────────────────────────────────────────────────┤
│ Layer 3: AscendModel MLIR 分析层                 │
│   AscendModelOps.td (方言定义)                   │
│   EstimateCycles.cpp (周期估算)                   │
│   PipelineAnalysis.cpp (流水线分析)               │
│   负责: 把 IR 映射到硬件、算 cycle                │
└─────────────────────────────────────────────────┘
```

## 为什么做成 MLIR Pass 而不是独立程序？

CostModel 要做的所有事情——解析、遍历、匹配、替换、管线管理——MLIR 框架都已经提供了。做成 Pass 是站在巨人肩膀上，避免从零造轮子。

| | Pass 方案 | 独立程序方案 |
|---|---|---|
| **MLIR 解析** | `parseSourceString()` 一行 | 自己写 TTIR 语法解析器 |
| **IR 遍历** | `module.walk()` 一行 | 自己写递归遍历 |
| **模式匹配** | `OpRewritePattern` 框架 | 手写 if-else 链 |
| **单 Pass 测试** | `ctest -R CostModelPasses.EstimateCycles` | 逻辑混在一起，难拆开测 |
| **与编译器集成** | 同进程零拷贝 (传 Module 对象) | 序列化→写磁盘→读磁盘→反序列化 |
| **加新操作** | TableGen 3 行 + C++ 5 行 | 改解析器、IR 表示、估算逻辑三处 |
| **Cross-op 分析** | `module.walk()` 全局视角 | 需要自己构建全局数据结构 |
| **错误处理** | MLIR 自带 `emitError()` + 位置信息 | 自己实现 |

特别地，CostModel 的输入是 TTIR——这正是 Triton 编译器的输出。在同一个 MLIR context 里，costmodel 可以直接消费编译前端产生的内存对象（Module），零拷贝。独立程序需要把 TTIR 序列化成文本文件，再读回来解析，不仅慢，而且容易丢失元信息。

从下往上走——下层不依赖上层。

---

## Part 1：AscendModel 方言 —— 硬件建模的"词汇表"

### 为什么需要它？

TTIR 描述"算什么"（`tt.load`、`arith.addf`），但不描述"在 Ascend 硬件的哪个单元上算、开销多少"。所以需要一个新的 IR，专门做硬件建模。

```
TTIR (算什么)                    AscendModel IR (用什么硬件算)
─────────────────────────────────────────────────────────
tt.load tensor<256xf32>    →    ascend.vector_load {bytes=1024}  ← 用 VecMTE2 搬运
arith.addf                 →    ascend.add                       ← 在 Vector Core 算
tt.dot                     →    ascend.matmul {M,N,K}            ← 在 Cube Core 算
tt.store                   →    ascend.vector_store {bytes=1024} ← 用 MTE3 写出
```

### 代码在哪

```
third_party/ascend/costmodel/include/AscendModel/IR/
├── AscendModelBase.td      ← 方言定义 + 枚举 (硬件单元、内存空间)
├── AscendModelOps.td       ← 操作定义 (matmul, add, vector_load, ...)
├── AscendModelInterfaces.td ← 接口定义 (EstimateCyclesOpInterface)
└── AscendModelDialect.h    ← C++ 方言注册

third_party/ascend/costmodel/lib/AscendModel/IR/
└── AscendModelOps.cpp      ← 每个 Op 的 cycle 估算实现
```

### 硬件单元枚举

`AscendModelBase.td:42-55`，定义 7 个硬件单元：

```
HWUnit          真实硬件对应              做什么
─────────────────────────────────────────────────
Cube            AI Cube Core             矩阵乘 (MAC)
CubeMTE2        Cube 的 DMA 通道         数据从 HBM→L1
FixPipe         Cube 的输出通道           Cube 结果 L0C→HBM
Vector          AI Vector Core           逐元素运算 (add/exp/reduce)
VecMTE2         Vector 的 DMA 通道        数据从 HBM→UB
MTE3            Vector 的输出通道         Vector 结果 UB→HBM
Scalar          标量单元                  控制流、地址计算
```

核心假设：**全流水架构，所有硬件单元可并行**。总时间 = max(各单元耗时)。

### 关键接口：EstimateCyclesOpInterface

`AscendModelInterfaces.td:16-98`，每个 AscendModel Op 必须实现 5 个方法：

| 方法 | 作用 | MatmulOp 实现 | AddOp 实现 |
|------|------|-------------|-----------|
| `estimateCycles(config)` | 算 cycle | Cube fractal 公式 | Vector 指令表查表 |
| `getHWUnit()` | 跑在哪个单元 | `HWUnit::Cube` | `HWUnit::Vector` |
| `getTransferBytes()` | 搬运字节数 | 0（计算 Op） | 0（计算 Op） |
| `getFlops()` | 浮点运算量 | `2*M*N*K` | `N_elements` |
| `getCyclesPerVectorOp()` | 每向量指令几 cycle | 1（默认） | 1（简单）/ 3-15（复杂） |

### Cycle 估算公式

`AscendModelOps.cpp` 实现三组公式：

**搬运 Op**（第 70-79 行）：
```cpp
cycles = bytes / (bandwidth_gbs * 1e9) * clock_ghz * 1e9 + startup_latency
```

**Vector 计算 Op**（第 57-67 行）：
```cpp
vectorWidth = 2048 / elementBits          // Vector 单元宽度 2048 bits
numVectorOps = ceil(numElements / vectorWidth)
cycles = numVectorOps * cyclesPerOp + startup
```

**Cube Matmul**（第 116-147 行）：
```cpp
// 把大矩阵分解成 fractal (如 fp16: 16×16×16)，每个 fractal 1 cycle
totalFractals = ceil(M/fracM) * ceil(N/fracN) * ceil(K/fracK)
computeCycles = totalFractals + startup
```

复杂 Vector 算子的 cycle 数（经过 flash attention 实测校准）：
```cpp
IMPL_COMPLEX_VECTOR_UNARY(SqrtOp, 6)
IMPL_COMPLEX_VECTOR_UNARY(RsqrtOp, 6)
IMPL_COMPLEX_VECTOR_UNARY(ExpOp, 9)       // exp: 9 cycles（校准前是 3）
IMPL_COMPLEX_VECTOR_UNARY(LogOp, 12)      // log: 12 cycles
IMPL_COMPLEX_VECTOR_UNARY(TanhOp, 18)     // tanh: 18 cycles
IMPL_COMPLEX_VECTOR_UNARY(SigmoidOp, 15)  // sigmoid: 15 cycles
```

---

## Part 2：ConvertTritonToAscend —— TTIR → AscendModel

### 为什么需要它？

Part 1 定义了 AscendModel 的"词汇"，但手头有 TTIR。需要把 TTIR 翻译成 AscendModel。

### 代码位置

`third_party/ascend/costmodel/lib/AscendModel/Transforms/ConvertTritonToAscend.cpp`

### 核心映射表

| TTIR Op | AscendModel Op | 代码行 | 硬件单元 |
|---------|---------------|--------|---------|
| `tt.load` | `vector_load` 或 `cube_load`（取决于是否有 `ascend.used_by_dot` 标记） | 268-313 | VecMTE2/CubeMTE2 |
| `tt.dot` | `matmul(M,N,K)` + 自动插入 `cube_store` + `vector_load` | 143-266 | Cube |
| `tt.store` | `vector_store`（丢弃指针 operand，只保留数据） | 340-370 | MTE3 |
| `tt.trans` | 直接透传输入（无硬件开销） | 315-338 | — |
| `tt.reduce` | `reduce_sum/max/min/prod`（检查 body 确定 reduction 类型） | 415-474 | Vector |
| `tt.broadcast` | `broadcast` | 476-503 | Vector |
| `arith.addf/subf/mulf/divf` | `add/sub/mul/div` | 509-558 | Vector |
| `arith.maxnumf/minnumf` | `max/min` | 509-558 | Vector |
| `arith.cmpf/cmpi` | `cmp_eq/ne/lt/le/gt/ge` | 560-599 | Vector |
| `tt.addptr/splat/make_range` 等 | 直接删除（无硬件开销） | 372-407 | — |

### 关键细节：dot 的输入处理

ConvertTritonDot 的 `ensureCubeInput` lambda（第 190-203 行）：
```cpp
auto ensureCubeInput = [&](Value input) -> Value {
    if (isFromLoad(input) || isFromCubeOp(input))
        return input;                               // 已经是正确路径
    // 否则: 插入 vector_store (UB→HBM) + cube_load (HBM→L1)
    auto vecStore = rewriter.create<VectorStoreOp>(...);
    auto cubeLoad = rewriter.create<CubeLoadOp>(...);
    return cubeLoad.getResult();
};
```

### 关键细节：dot 的输出处理

ConvertTritonDot 第 236-262 行：
```cpp
// 总是创建 cube_store
rewriter.create<CubeStoreOp>(loc, matmul.getResult(), resultBytes, ...);
// 如果被 Vector 操作消费 → 再创建 vector_load，并重定向用户
if (usedByVectorOps) {
    auto vecLoad = rewriter.create<VectorLoadOp>(...);
    rewriter.replaceOp(op, vecLoad.getResult());  // 把原 dot 替换为 vecLoad
}
```

---

## Part 3：InsertDataTransfers —— 跨路径搬运

### 为什么需要它？

Part 2 的 ConvertTritonDot 已经内联了搬运逻辑。但可能有遗漏的场景（如多个 dot 之间的数据流、循环边界后的数据流向），InsertDataTransfers 作为防御性第二遍扫描。

### 代码位置

`third_party/ascend/costmodel/lib/AscendModel/Transforms/InsertDataTransfers.cpp`

### 硬件数据流模型

```
Vector 路径:  HBM ──MTE2──▶ UB ──▶ Vector ──▶ UB ──MTE3──▶ HBM
Cube 路径:    HBM ──MTE2──▶ L1 ──MTE1──▶ L0A/L0B ──▶ Cube ──▶ L0C ──FixPipe──▶ HBM

Vector 结果被 Cube 用 → 插入 vector_store(UB→HBM) + cube_load(HBM→L1)
Cube 结果被 Vector 用 → 插入 cube_store(L0C→HBM) + vector_load(HBM→UB)
```

### 去重机制

Guard 1（输入侧，第 123 行）：`if (isa<CubeLoadOp, VectorLoadOp>(defOp)) continue;`
Guard 2（输出侧，第 179-185 行）：`matmulResult.getUses()` 中 Vector 用户已被 ConvertTritonDot 的 `replaceOp` 重定向到 `VectorLoadOp`，所以 vectorUsers 为空 → 跳过。

**结论：Part 2 和 Part 3 不重复，是分层防御。**

---

## Part 4：EstimateCycles —— 每个 Op 算多少 cycle

### 为什么需要它？

Part 2+3 把 TTIR 变成了 AscendModel IR，但每个 Op 的 `estimated_cycles` 还是空的。

### 代码位置

`third_party/ascend/costmodel/lib/AscendModel/Transforms/EstimateCycles.cpp`

### 两步走

**第一步**（第 133-175 行）：解析所有循环，确定 trip count
```cpp
module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });
// 对每个循环：
//   如果有 arg-bindings → 代入参数求 trip count
//   如果是静态 bound → 直接取
//   否则 → 报错，要求用户传 arg-bindings
forOp->setAttr("ascend.trip_count", tripCount);
```

**trip count 是什么**：循环执行次数。`scf.for %i = 0 to 1024 step 64` → trip_count = 16。循环内所有 Op 的 cycle 都乘这个数。

**第二步**（第 185-230 行）：遍历所有非控制流 Op，调用 estimateCycles
```cpp
module.walk([&](Operation *op) {
    int64_t cycles = cyclesOp.estimateCycles(config);   // 调 Part 1 的实现
    int64_t loopMultiplier = getLoopMultiplier(op);      // 乘上外层循环次数
    int64_t totalOpCycles = cycles * loopMultiplier;

    // 按硬件单元分类累加
    switch (cyclesOp.getHWUnit()) {
        case HWUnit::Cube:     stats.cubeCycles     += totalOpCycles; break;
        case HWUnit::Vector:   stats.vectorCycles   += totalOpCycles; break;
        case HWUnit::CubeMTE2: stats.cubeLoadCycles  += totalOpCycles; break;
        // ...
    }
});
```

### Roofline 聚合公式

`RooflineStats::calculateRooflineCycles()`（第 69-85 行）：

```cpp
// Cube 路径: 计算、加载、写出取 max (可以 overlapped)
cubePathCycles   = max(cubeCycles,  cubeLoadCycles,  cubeStoreCycles);
// Vector 路径: max(compute, load, store)
vectorPathCycles = max(vectorCycles, vectorLoadCycles, vectorStoreCycles);
// Cube 和 Vector 并行 → 取 max
totalCycles      = max(cubePathCycles, vectorPathCycles);
```

### 为什么是 max 而不是加？

硬件单元各自有独立流水线，同时运行。Cube 在算的时候，Vector 也在算，MTE2 也在搬数据。所以总时间不是加起来，而是**取最慢的那个**。这就是 Roofline 模型的本质：瓶颈在哪个单元，总时间就等于那个单元的。

---

## Part 5：PipelineAnalysis —— 比 Roofline 更精细的调度

### 为什么需要它？

Roofline 假设同一硬件单元上的所有操作可以同时执行。但实际上，Vector Core 一次只能执行一条指令。3 个 add 操作必须排队：add1 → add2 → add3。

### 代码位置

- `costmodel/include/AscendModel/Analysis/PipelineAnalysis.h`（调度器声明）
- `costmodel/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp`（Pass 实现）

### 核心数据结构

`PipelineOp` 结构体（`PipelineAnalysis.h:28-49`）：
```cpp
struct PipelineOp {
    int64_t opId;
    HWUnit hwUnit;              // 硬件单元
    int64_t startCycle;         // 调度后：开始时间
    int64_t duration;           // 持续 cycle
    int64_t endCycle;           // 调度后：结束时间
    int64_t loopMultiplier;
    SmallVector<int64_t> dependsOn; // 数据依赖
};
```

### 调度规则

ASAP（As Soon As Possible）算法：
1. 数据依赖：必须等生产者的数据就绪
2. 硬件单元占用：同一单元上一个操作没结束，下一个不能开始
3. 不同单元：完全并行，互不阻塞

### 三个指标对比

PipelineAnalysis 输出三个值到 module 属性：

```cpp
module->setAttr("ascend.scheduled_cycles_one_iter", oneIterCycles); // 调度器
module->setAttr("ascend.roofline_cycles", rooflineTotalCycles);     // Roofline
module->setAttr("ascend.simple_sum_cycles", simpleSumCycles);       // 简单求和
```

理论上：`roofline ≤ scheduled ≤ simple_sum`

### 测试验证

`PipelineSchedulerTest.cpp` 中的 7 个测试精确描述了调度器行为：
1. 不同单元并行：Cube 100 + Vector 80 → total = max(100,80) = 100
2. 同一单元串行：Vector 40 + Vector 60 → total = 40+60 = 100
3. 数据依赖：Cube 100 → Vector 30（依赖 Cube 结果） → total = 100+30 = 130
4. 循环依赖检测
5. Kernel 级别：`(single + barrier) * (1 + overhead) * waves`

---

## Part 6：PerfReport + HIVMAnalysis

### PerfReportPass

读取前几个 Pass 写入的属性，生成格式化的性能报告。
- 代码：`costmodel/lib/AscendModel/Transforms/PerfReportPass.cpp`

### HIVMAnalysisPass（独立路径）

**不在 6-Pass 主管线中**。独立使用，需要已编译的 HIVM IR。

- 代码：`costmodel/lib/AscendModel/Transforms/HIVMAnalysisPass.cpp`
- 输入：BiSheng 编译后的 `.npuir.mlir`
- 用途：指令级精度分析 + Perfetto trace 可视化
- 模式：`static`（静态调度）或 `des`（离散事件仿真）

CostModel 精度层次：
```
粗: Roofline (EstimateCycles)        ← 纯公式，最快
中: Pipeline 调度 (PipelineAnalysis)  ← 考虑单元串行化
精: HIVM 调度 (HIVMAnalysis)          ← 用真实编译输出，指令级
真: 硬件实测                           ← 最准，最慢
```

---

## Part 7：Pass 管线组装 + Python 桥接

### 6-Pass 管线（PassRegistration.cpp:59-94）

```cpp
void registerAscendModelPipeline() {
    pm.addPass(createConvertTritonToAscendPass());   // 1. TTIR→AscendModel
    pm.addPass(createInsertDataTransfersPass());      // 2. 插入跨路径搬运
    pm.addPass(createAssignOpIDsPass());              // 3. 分配 Op ID
    pm.addPass(createEstimateCyclesPass(opts));       // 4. 估算 cycle
    pm.addPass(createPipelineAnalysisPass(opts));     // 5. 流水线分析
    pm.addPass(createPerfReportPass());               // 6. 生成报告
}
```

### Python 桥接（triton_ascend.cc:650-670）

```cpp
void init_triton_ascend(py::module &&m) {
    // m = triton._C.libtriton.ascend 子模块

    m.def("run_costmodel_inproc",
          [](const std::string &mlirText, const std::vector<std::string> &extraArgs) {
              return runAscendCostModelInProcess(mlirText, extraArgs);
          });

    auto passes = m.def_submodule("passes");
    auto passes_ttir = passes.def_submodule("ttir");
    // ... 注册所有 Ascend Pass 的 Python 接口
}
```

`m` 是 pybind11 的 `py::module` 对象，`m.def("函数名", lambda)` 往模块上挂 Python 函数。Python 里 `ascend.run_costmodel_inproc(...)` 就是调这个。

`m` 的创建过程（`main.cc:37,63`）：
```cpp
#define INIT_BACKEND(name) init_triton_##name(m.def_submodule(#name));
FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
// 展开为: init_triton_ascend(m.def_submodule("ascend"));
//        init_triton_nvidia(m.def_submodule("nvidia"));
//        init_triton_amd(m.def_submodule("amd"));
```

`TRITON_BACKENDS_TUPLE` 来自 `CMakeLists.txt:317`：
```cmake
set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})
```

---

### Python import 路径是怎么来的？

`triton._C.libtriton.ascend.run_costmodel_inproc()` 这个路径看起来很长，但它**不是文件系统上的目录**，而是 pybind11 在运行时动态构建的 Python 模块层级。

#### 完整的模块层级

```
Python 侧的 import 路径          C++ 侧怎么创建的
──────────────────────────────────────────────────
triton                           普通 Python 包 (磁盘上的 triton/ 目录)

triton._C                        普通 Python 包 (triton/_C/ 目录)

triton._C.libtriton              PYBIND11_MODULE(libtriton, m)  ← C++ 编译出的 .so
                                  实际文件: triton/_C/libtriton.cpython-xxx.so

triton._C.libtriton.ir           m.def_submodule("ir")           ← 全部在
triton._C.libtriton.passes       m.def_submodule("passes")       ← 同一个 .so
triton._C.libtriton.ascend       m.def_submodule("ascend")       ← 里动态创建
triton._C.libtriton.buffer_ir    m.def_submodule("buffer_ir")
triton._C.libtriton.interpreter  m.def_submodule("interpreter")
triton._C.libtriton.llvm         m.def_submodule("llvm")
```

`_C` 是 Python 社区的惯例（如 `torch._C`），表示"这是 C++ 扩展模块，不是纯 Python 代码"。

#### 创建过程（main.cc:51-63）

```cpp
// ① 编译出的 .so 文件就是 triton/_C/libtriton.cpython-xxx.so
PYBIND11_MODULE(libtriton, m) {
    // m = triton._C.libtriton

    // ② 创建子模块（都是纯内存操作，不创建文件）
    init_triton_ir(m.def_submodule("ir"));              // → .ir
    init_triton_passes(m.def_submodule("passes"));      // → .passes
    init_triton_ascend(m.def_submodule("ascend"));      // → .ascend
}
```

#### ascend 子模块内部（triton_ascend.cc:650-661）

```cpp
void init_triton_ascend(py::module &&m) {
    // m = triton._C.libtriton.ascend
    auto passes = m.def_submodule("passes");            // → .ascend.passes
    auto passes_ttir = passes.def_submodule("ttir");     // → .ascend.passes.ttir
    m.def("run_costmodel_inproc", ...);                  // → .ascend.run_costmodel_inproc()
}
```

#### Python 侧的对应关系

```python
from triton._C.libtriton import ascend       # ← m.def_submodule("ascend")
from triton._C.libtriton import ir           # ← m.def_submodule("ir")

ascend.run_costmodel_inproc(...)             # ← m.def("run_costmodel_inproc", ...)
ascend.passes.ttir.add_triton_to_llvm(pm)    # ← passes.def_submodule("ttir")
```

全部在**同一个 .so 文件**里，编译时把 C++ 代码（包括 costmodel）链接进去。Python `import` 时 pybind11 初始化函数被调用，动态构建出所有子模块。

---

## Part 8：Python 运行时层

### 代码位置

`third_party/ascend/backend/runtime/costmodel_runtime.py`

### 核心调用链路

```python
# 入口: autotuner 调用的函数
def costmodel_bench(config_ttir_items):
    # 输入: [{"config": cfg, "ttir": "IR文本", "arg_bindings": "arg3=100,pid_x=0"}, ...]
    # 输出: {cfg: latency_us, ...}

    for item in pending_items:
        # ① 生成缓存 key: SHA256(ttir + args)
        cache_key = make_costmodel_cache_key(ttir, extra_args)

        # ② 查缓存 (内存 → 磁盘)
        cached = load_costmodel_latency(cache_key)
        if cached: return cached

        # ③ 调 C++ 桥接
        output = run_costmodel(ttir, extra_args=["-ascend-perf-model", "arg-bindings=..."])

        # ④ 正则解析 "Estimated Time: X.XXX us" → float
        latency = parse_latency(output)

        # ⑤ 写缓存
        store_costmodel_latency(cache_key, latency)
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `TRITON_ASCEND_ENABLE_COSTMODEL` | 0 | 是否启用 costmodel |
| `TRITON_COSTMODEL_WORKER_NUM` | cpu_count | 并行工作线程数 |

### 设计文档中的约束

| 支持的配置参数 | 不支持的配置参数 |
|-------------|-------------|
| tiling 参数（BLOCK_SIZE, BLOCK_M/N/K 等） | multibuffer |
| runtime 整数参数（通过 arg-bindings） | enable_ubuf_saving |
| program_id / num_programs（通过 arg-bindings） | tile_mix_vector/cube_loop |
| | HIVM 相关参数 |
| | SIMT 后端编译选项 |

---

## 完整调用链路总结

```
autotuner.py
  │
  └─ costmodel_bench(items)              ← costmodel_runtime.py:200
       │                                   输入: config+TTIR+arg_bindings
       │
       ├─ make_costmodel_cache_key()       ← SHA256(ttir + args)
       ├─ load_costmodel_latency()         ← 查缓存
       │
       └─ run_costmodel(ttir, args)        ← costmodel_runtime.py:33
            │
            └─ ascend_capi.run_costmodel_inproc()  ← triton_ascend.cc:665
                 │                                   (pybind11)
                 │
                 └─ runAscendCostModelInProcess()    ← triton_ascend.cc:606
                      │
                      ├─ parse MLIR → Module
                      ├─ Pass 1: ConvertTritonToAscend     ← .cpp
                      ├─ Pass 2: InsertDataTransfers       ← .cpp
                      ├─ Pass 3: AssignOpIDs
                      ├─ Pass 4: EstimateCycles             ← .cpp + AscendModelOps.cpp
                      │    Roofline: max(compute, memory)
                      ├─ Pass 5: PipelineAnalysis
                      ├─ Pass 6: PerfReport
                      │    cycles → us = cycles / 1850
                      └─ return "Estimated Time: X.XXX us"
                            │
                            ▼
                 parse_latency() → 浮点数 → 缓存 → 返回 autotuner
```

---

## 源码与设计文档对应表

| 设计文档概念 | 对应代码 |
|------------|---------|
| 硬件单元 (7个) | `AscendModelBase.td:42-55` |
| Op 定义 (matmul/add/...) | `AscendModelOps.td` |
| Op 的 cycle 估算 | `AscendModelOps.cpp` |
| TTIR→AscendModel 转换 | `ConvertTritonToAscend.cpp` |
| 跨路径搬运插入 | `InsertDataTransfers.cpp` |
| per-op cycle 估算 + trip count | `EstimateCycles.cpp` |
| Roofline 聚合公式 | `EstimateCycles.cpp:69-85` |
| 流水线调度 (ASAP) | `PipelineAnalysisPass.cpp` + `PipelineAnalysis.h` |
| 6-Pass 管线组装 | `PassRegistration.cpp:59-94` |
| Python C++ 桥接 | `triton_ascend.cc` |
| Python 运行时入口 | `costmodel_runtime.py` |
| 硬件参数 JSON | `configs/ascend_910b.json`, `configs/ascend_davidv100.json` |
| C++ 硬件参数解析 | `HardwareConfig.h/cpp` |
| HIVM 独立分析路径 | `HIVMAnalysisPass.cpp` + `HIVMAnalysis.h` |

---

## TableGen 生成的 .inc 文件

`AscendModelOps.cpp` 中有两个关键 include：
```cpp
#include "AscendModel/IR/AscendModelInterfaces.cpp.inc"  // 接口默认实现
#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.cpp.inc"         // Op parser/printer/verifier
```

这些 `.inc` 文件由 `mlir-tblgen` 从 `.td` 文件自动生成，在 build 目录下。源文件中不存在。

```
源文件 (.td)                →  cmake build 生成 (.inc)
────────────────────────────────────────────────────
AscendModelOps.td           →  AscendModelOps.cpp.inc / .h.inc
AscendModelInterfaces.td    →  AscendModelInterfaces.cpp.inc / .h.inc
Passes.td                   →  Passes.h.inc
```
