# Triton-Ascend 完整教学课程

> 本课程遵循"一条主线串起全部"的设计思路：跟踪一段 Triton Python 代码从编写到在 Ascend NPU 上执行的完整旅程。

---

## 课程总览

### 一条主线

这个项目的本质是一句话：**把 Triton Python 代码，编译成能在 Ascend NPU 上跑的二进制。**

最自然的逻辑就是跟踪这段代码的完整旅程：

```
你写的 Triton Python 代码
        │
        ▼
   ① 这段代码长什么样？            → Triton 语言
        │
        ▼
   ② 它怎么变成 IR？IR 又是什么？   → MLIR 基础 + 代码生成
        │
        ▼
   ③ MLIR 这个工具怎么用？          → Dialect/Op/Pass
        │
        ▼
   ④ Triton 核心管线做什么？        → TTIR 优化 + Ascend Pass
        │
        ▼
   ⑤ 怎么跑起来？怎么调优？         → 运行时 + Autotuning
```

### 每一阶段的逻辑关系

| 阶段 | 要回答的问题 | 为什么必须在这学 |
|------|------------|--------------|
| ① Triton 语言 | "我们到底在编译什么东西？" | 你连输入都不认识，后面一切无从谈起 |
| ② MLIR 基础 + 代码生成 | "Python 怎么变成 IR？IR 长什么样？" | 看了 ① 的输出才自然引出——编译器怎么翻译它？ |
| ③ MLIR 工具链 | "Dialect/Op/Pass 怎么定义和使用的？" | ② 中你看到了 `tt.load`、`scf.for`，③ 告诉你怎么定义它们、怎么转换它们 |
| ④ Triton 核心管线 + Ascend Pass | "TTIR → Linalg 做了什么？为什么需要 Ascend 专属 Pass？" | ③ 学了 Pass 写法，④ 给你看真实 Pass 实例；GPU 和 Ascend 硬件差异驱动了这些 Pass |
| ⑤ 运行时 + 调优 | "编译完怎么跑？怎么跑得快？" | ④ 产生了二进制，⑤ 负责加载执行和寻找最优参数 |

### 学习原则

- **先看懂森林，再看清树木**：先建立全局地图，具体改哪段代码时再深入那一段的细节
- **每个阶段都遵循同一模式**：动机 → Demo → 源码导读 → 小结

---

## 阶段①：Triton 语言基础

### 1.1 动机：为什么先学 Triton 语言？

整个 Triton-Ascend 项目的输入就是 Triton Python 代码。你必须先知道：

- **用户写的是什么** —— 语法、语义
- **用户思考方式** —— block/tile 的编程模型
- **用户期待什么行为** —— 哪些是编译器保证的、哪些不是

否则后面看编译器代码时，你根本不知道 `tt.load` 为什么长成那样、`program_id` 在 IR 里为什么要变成 `tt.get_program_id`。

### 1.2 Triton 编程模型：SPMD

Triton 采用 **SPMD (Single Program, Multiple Data)** 模型：

```
同一个程序 → 多个实例 → 各自处理不同数据
```

**类比**：开运动会，100 个学生跑到操场上，每个人都拿到同一份指令："跑 100 米"。但每个人在自己的跑道（不同的数据区域）上执行，互不干扰。

在 Triton 里：
- **一个 Program** = 一份指令（你的 kernel 函数）
- **Grid** = 多少个 Program 实例同时跑
- 每个 Program 通过 `program_id` 知道自己负责哪一片数据

```
输入数据：[0, 1, 2, 3, 4, 5, 6, 7]  (8个元素)
BLOCK_SIZE = 4

Program 0 (处理 [0:4])   ← pid=0, offsets=[0,1,2,3]
Program 1 (处理 [4:8])   ← pid=1, offsets=[4,5,6,7]

两个 Program 同时执行同样代码，只是 pid 不同
```

### 1.3 Demo: 向量加法（逐行拆解）

```python
@triton.jit                          # ① 标记：这是个 Triton kernel
def add_kernel(
    x_ptr,                           # ② 指针：指向 input x 的首地址
    y_ptr,                           #    指针：指向 input y 的首地址
    output_ptr,                      #    指针：指向 output 的首地址
    n_elements,                      #    整数：一共有多少个元素
    BLOCK_SIZE: tl.constexpr,        # ③ 编译期常量：每个 program 处理多少元素
):
```

**③ 为什么需要 `tl.constexpr`？**

这是 Triton 的关键设计——它让编译器在**编译时就确定**每个 program 处理多少数据，从而可以做更激进的优化（比如循环展开）。如果 BLOCK_SIZE 是运行时变量，编译器就没法展开。

```python
    # ④ 获取身份标识
    pid = tl.program_id(axis=0)
```

SPMD 的核心：所有 program 执行同一份代码。`program_id` 是唯一区分它们的变量。

```
n_elements=8, BLOCK_SIZE=4  → 需要 ceil(8/4)=2 个 program

Program 0: pid=0, 处理元素 [0, 1, 2, 3]
Program 1: pid=1, 处理元素 [4, 5, 6, 7]
```

```python
    # ⑤ 计算自己的"工作范围"
    block_start = pid * BLOCK_SIZE           # pid=0 → 0,  pid=1 → 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # pid=0: offsets = 0 + [0,1,2,3] = [0,1,2,3]
    # pid=1: offsets = 4 + [0,1,2,3] = [4,5,6,7]
```

`tl.arange(0, BLOCK_SIZE)` 返回一个**向量** `[0, 1, 2, ..., BLOCK_SIZE-1]`。这里的 `block_start` 是标量，但标量+向量 = 向量（广播语义），所以 `offsets` 得到的是一个向量。

```python
    # ⑥ 边界保护
    mask = offsets < n_elements
```

为什么需要 mask？因为 `n_elements` 不一定是 `BLOCK_SIZE` 的整数倍。比如 `n_elements=10, BLOCK_SIZE=4`：

```
Program 2: pid=2, offsets=[8, 9, 10, 11]
mask = [8<10, 9<10, 10<10, 11<10] = [True, True, False, False]
```

mask 告诉 `load/store`：哪些位置是有效的、哪些应该填零或忽略。

```python
    # ⑦ 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
```

`x_ptr + offsets`：指针 + 向量偏移 = 一个指针向量，指向每个要加载的位置。

`mask=mask`：超出边界的元素用 0 填充。

```python
    # ⑧ 计算
    output = x + y    # 向量加法，SIMD
```

这是 triton 的优雅之处——你像写标量一样写向量操作。

```python
    # ⑨ 写回
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 1.4 Launch 机制

```python
def add(x, y):
    output = torch.empty_like(x)        # 预分配输出
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # grid = (ceil(n_elements / BLOCK_SIZE), )
    # 即：需要多少个 program

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    #        ↑
    #    这个方括号是 Triton 的特殊语法：传入 grid 函数
    #    BLOCK_SIZE=1024 是编译期常量，必须用 keyword 传入
    return output
```

`add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)` 的执行过程：

1. Triton 调用 `grid(meta)` 得到 `(num_programs,)`
2. 启动 `num_programs` 个 program 实例，每个都有不同的 `pid`
3. 所有实例并行执行 `add_kernel` 里的代码

### 1.5 Python API → IR 的桥梁

这项目最核心的问题：**Python 的 `tl.load()` 怎么变成 MLIR 的 `tt.load`？**

答案分两步：

**第一步**：`core.py` 里的 `tl.load()` 只是个入口，它调用 `_semantic.load()`。

```python
# python/triton/language/core.py:2111
def load(pointer, mask=None, other=None, ...):
    return _semantic.load(pointer, mask, ...)
```

**第二步**：`semantic.py` 里的 `Semantic.load()` 调用 `self.builder.create_*()` 来**构建 IR**。

```python
# python/triton/language/semantic.py:40
def program_id(self, axis):
    return self.tensor(self.builder.create_get_program_id(axis), tl.int32)
```

**关键理解**：你在 Python 里写 `tl.program_id(0)`，它不会马上执行——它是在**构建 IR**。类比：你写的 Python 代码不是指令，而是**一种脚本，告诉编译器"我要生成什么 IR"**。

```
Python:  tl.program_id(0)
            │ semantic.py: self.builder.create_get_program_id(0)
            ▼
TTIR:    %0 = tt.get_program_id x : i32
```

### 1.6 阶段①小结

| 概念 | 含义 | 代码中如何体现 |
|------|------|-------------|
| **Program** | 同一个 kernel 的并行实例 | `pid = tl.program_id(axis=0)` |
| **Grid** | 有多少个 program | `grid = lambda meta: (num_programs,)` |
| **Block** | 每个 program 处理的数据块 | `BLOCK_SIZE: tl.constexpr` |
| **Mask** | 边界保护 | `mask = offsets < n_elements` |
| **Load/Store** | 数据搬运 | `tl.load(...)`, `tl.store(...)` |
| **编译期常量** | 编译时已知的值 | `tl.constexpr` |
| **API→IR 桥梁** | Python 操作构建 IR | `core.py → semantic.py → builder.create_*()` |

---

## 阶段②：从 Python 到 MLIR —— 代码生成与 IR 基础

### 2.1 动机：builder 从哪来？Python 代码究竟生成了什么？

上一阶段我们知道：

```python
tl.program_id(0)          # Python
    ↓ semantic.py
self.builder.create_get_program_id(0)   # 创建 IR
    ↓
%0 = tt.get_program_id x : i32          # TTIR（MLIR）
```

### 2.2 完整的 Python → IR 链路

```
┌─────────────────────────────────────────────────────┐
│ Python 层                                            │
│                                                      │
│  tl.program_id(0)         ← 你写的代码                │
│       │                                              │
│       ▼                                              │
│  semantic.py              ← 语义层，做类型检查/验证     │
│  self.builder.create_get_program_id(0)               │
│       │                                              │
├─────────────────────────────────────────────────────┤
│ pybind11 层  (ir.cc)                                  │
│                                                      │
│  TritonOpBuilder        ← C++ 类，通过 pybind11       │
│  .def("create_get_program_id", ...)  → 暴露到Python   │
│       │                                              │
├─────────────────────────────────────────────────────┤
│ C++ / MLIR 层                                         │
│                                                      │
│  TritonOpBuilder        ← 包装了 MLIR OpBuilder        │
│  self.create<GetProgramIdOp>(axis)                    │
│       │                                              │
│       ▼                                              │
│  创建 MLIR Operation:  %0 = tt.get_program_id x       │
└─────────────────────────────────────────────────────┘
```

用源码验证这个链路：

**第一层 — semantic.py 调用 builder**：
```python
# python/triton/language/semantic.py:43
def program_id(self, axis):
    return self.tensor(self.builder.create_get_program_id(axis), tl.int32)
```

**第二层 — pybind11 桥接 (ir.cc:1680)**：
```cpp
// python/src/ir.cc:1680
.def("create_get_program_id",
     [](TritonOpBuilder &self, int axis) -> Value {
         return self.create<GetProgramIdOp>(axis);  // ← 这里创建 MLIR 操作
     })
```

**第三层 — C++ TritonOpBuilder 包装 OpBuilder**：
```cpp
// python/src/ir.h:16-23
class TritonOpBuilder {
public:
  TritonOpBuilder(mlir::MLIRContext *context, ...) {
    builder = std::make_unique<OpBuilder>(context);  // ← MLIR 原生的 OpBuilder
  }
};
```

**关键理解**：`TritonOpBuilder` 不是执行操作，而是**在内存中构建 IR 图**。每个 `create_*` 调用都往图中加一个节点。

### 2.3 code_generator 详细追踪

```
compile()                              # compiler.py:228 入口
  │
  ├─ src.make_ir()                     # compiler.py:308 → compiler.py:82
  │     │
  │     └─ ast_to_ttir(fn, src, ...)   # code_generator.py:1659
  │           │
  │           ├─ generator = CodeGenerator(...)
  │           │     │
  │           │     ├─ self.builder = ir.builder(context)     # 创建 Builder
  │           │     │   (TritonOpBuilder, C++ 对象)
  │           │     └─ self.module = self.builder.create_module()  # 创建 Module (空)
  │           │
  │           └─ generator.visit(fn.parse())          # 遍历 AST，构建 IR
  │                 │
  │                 └─ 每个 visit_* 方法处理一种 AST 节点
  │                    例如 visit_Assign → self.semantic.store() → builder.create_store()
  │                    每行 Python 代码都在往 module 里插入 IR 节点
  │
  └─ module = generator.module          # 拿到构建好的 Module
```

**关键类关系**：

```
CodeGenerator (AST 遍历器, code_generator.py:303)
  ├── .builder   → ir.builder (TritonOpBuilder, C++ 对象)
  │                 通过 create_xxx() 在内存中构建 IR 节点
  ├── .semantic  → TritonSemantic (语义层)
  │                 验证类型、广播等，然后调 builder
  └── .module    → ModuleOp (MLIR 顶层容器)
                   所有 IR 节点都挂在 module 下面
```

### 2.4 `mod` (Module) 对象的旅程

`mod` 就是 `CodeGenerator.module`——编译一开始创建的 MLIR 顶层容器。

```
创建:
  code_generator.py:343
  self.module = self.builder.create_module()

返回:
  code_generator.py:1691
  module = generator.module

传入第一站 (make_ttir):
  compiler.py:308
  module = src.make_ir(...)    ← 拿到刚生成的 module

传入第二站 (通过 stages 调度):
  compiler.py:1227
  stages["ttir"] = lambda src, metadata: make_ttir(src, ...)
                                           ↑
                                      module 被传给 make_ttir

传入第三站 (ttir_to_linalg):
  compiler.py:1231
  stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, ...)
```

`stages` 字典就是一个流水线调度表——上一个阶段的输出 `src` 就是下一个阶段的输入。`mod`/`src` 是同一个 `Module` 对象，只是贯穿整个流水线时叫法不同。

### 2.5 Operation 是怎么定义的？—— TableGen

MLIR 里定义操作不靠手写 C++，而是用 **TableGen**——一种 DSL（领域特定语言），写 `.td` 文件来自动生成 C++ 代码。

以 `GetProgramIdOp` 为例（`include/triton/Dialect/Triton/IR/TritonOps.td:616`）：

```tablegen
def TT_GetProgramIdOp : TT_Op<"get_program_id", [Pure]> {   // ① 定义一个 Op
    let arguments = (ins TT_ProgramDim:$axis);               // ② 输入: axis(x/y/z)
    let results = (outs I32:$result);                        // ③ 输出: i32
    let assemblyFormat = "$axis attr-dict `:` type($result)"; // ④ 文本格式
}
```

| 行 | 含义 |
|----|------|
| ① `TT_Op<"get_program_id", [Pure]>` | MLIR 名字叫 `tt.get_program_id`，Pure 表示无副作用 |
| ② `arguments` | 输入：一个 axis 参数（x/y/z） |
| ③ `results` | 输出：一个 i32 类型的值 |
| ④ `assemblyFormat` | 规定文本格式：`tt.get_program_id x : i32` |

对比更复杂的 LoadOp：

```tablegen
def TT_LoadOp : TT_Op<"load", [Pure, ...]> {
    let arguments = (
      ins
      AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,  // 指针（多种类型）
      Optional<TT_BoolLike>:$mask,                   // 可选的 mask
      Optional<TT_Type>:$other,                      // 可选的填充值
      DefaultValuedAttr<..., "false">:$isVolatile    // 属性：默认值
    );
    let results = (outs TT_Type:$result);
}
```

**LoadOp vs GetProgramIdOp**：
- `GetProgramIdOp`：1 个必需参数
- `LoadOp`：3 个操作数（其中 2 个可选）、5 个属性——正好对应 `tl.load(pointer, mask=mask, other=other, ...)`

### 2.6 完整数据流（Python → IR）

```
Python 代码:
  pid = tl.program_id(axis=0)

  ┌─ core.py ──────────────────────────────────────────┐
  │ def program_id(axis, _semantic=None):               │
  │     return _semantic.program_id(axis)               │
  └────────────────────┬────────────────────────────────┘
                       ▼
  ┌─ semantic.py ──────────────────────────────────────┐
  │ def program_id(self, axis):                         │
  │     return self.tensor(                              │
  │         self.builder.create_get_program_id(axis),   │
  │         tl.int32                                     │
  │     )                                                │
  └────────────────────┬────────────────────────────────┘
                       ▼
  ┌─ ir.cc (pybind11) ─────────────────────────────────┐
  │ .def("create_get_program_id",                       │
  │      [](TritonOpBuilder &self, int axis) -> Value { │
  │          return self.create<GetProgramIdOp>(axis);  │
  │      })                                              │
  └────────────────────┬────────────────────────────────┘
                       ▼
  ┌─ C++ / MLIR ───────────────────────────────────────┐
  │ TritonOpBuilder 内部的 OpBuilder                     │
  │ 调用 GetProgramIdOp::create()                        │
  │ 在内存中生成: %0 = tt.get_program_id x : i32        │
  └────────────────────────────────────────────────────┘
```

最终生成的 TTIR 文本（来自 `test/Triton/vecadd.mlir:5`）：
```mlir
%0 = tt.get_program_id x : i32
```

### 2.7 阶段②小结

| 概念 | 说明 | 在本项目中的位置 |
|------|------|----------------|
| **OpBuilder** | MLIR 的 IR 构造器 | `ir.h` 中的 `TritonOpBuilder` 包装了它 |
| **pybind11** | C++ ↔ Python 桥梁 | `ir.cc` 把所有 builder 方法暴露给 Python |
| **semantic.py** | Python 语义验证 + 调用 builder | `python/triton/language/semantic.py` |
| **core.py** | 用户接口，转发到 semantic | `python/triton/language/core.py` |
| **TableGen** | 定义 Op 的 DSL，自动生成 C++ | `include/triton/Dialect/Triton/IR/TritonOps.td` |
| **TTIR** | Python 代码生成的 MLIR 文本 | `test/Triton/*.mlir` 有大量例子 |

**核心要点**：你在 Triton Python 里写的每一行代码，本质是在**间接调用 C++ 的 OpBuilder，往 MLIR 图中插入节点**。它不是"执行"，而是"构建"。

---

## 阶段③：MLIR Pass —— IR 怎么被一步步转换

### 3.1 动机：为什么要"转换"IR？

TTIR 直接放到硬件上能跑吗？**不能。**
- Ascend NPU 不认识 `tt.load` 这个操作
- 昇腾编译器 BiSheng 只认识 Linalg 方言的操作
- 最终硬件只认识二进制指令

所以需要一层层**转换（lowering）**。

### 3.2 Pass 是什么？

**Pass = 一趟 IR → IR 的变换。**

类比：照片处理流水线。
```
原始照片 → [裁剪] → [调色] → [加滤镜] → [压缩] → 最终图片
```

每次操作输入是一张照片，输出也是一张照片，但内容被改变了。

MLIR Pass 同理：
```
输入: 一个 Module (包含很多 Func、Block、Op)
    │
    ▼
  [Pass 执行]
    │  遍历每个 Operation
    │  匹配特定的模式
    │  替换为新的 Operation
    ▼
输出: 同样的 Module，但里面的 Operation 变了
```

**关键**：Pass 不改变语义（计算结果相同），只改变**表达方式**（从高层抽象变为低层抽象）。

伪代码理解 Pass 的工作原理：
```python
def run_pass(module):
    for func in module.functions:
        for block in func.blocks:
            for op in block.operations:
                if op.name == "tt.load":           # 匹配
                    new_op = convert_to_linalg_load(op)  # 替换
                    replace(op, new_op)
```

真实的 Pass 用 C++ Pattern Rewrite 框架，但**思想完全一样**：遍历 → 匹配 → 替换。

### 3.3 `passes` 和 `pm` 的关系

```python
# compiler.py:36  — passes 是从 C++ 导入的 Python 模块
from triton._C.libtriton import ir, passes, ascend

# compiler.py:136 — pm 是一个 PassManager 对象
pm = ir.pass_manager(mod.context)

# compiler.py:138-145 — passes 往 pm 里注册 Pass
passes.common.add_inliner(pm)          # 注册内联 Pass
passes.ttir.add_combine(pm)            # 注册合并 Pass
passes.common.add_canonicalizer(pm)    # 注册规范化 Pass

# compiler.py:146 — pm 一次性执行所有注册好的 Pass
pm.run(mod, 'make_ttir')
```

类比：
```
passes = 工具箱      （里面有 inliner、canonicalizer、cse 等各种工具）
pm     = 传送带      （决定工具的执行顺序）

passes.common.add_inliner(pm)   = 把"内联器"放到传送带上
passes.common.add_cse(pm)       = 把"公共表达式消除器"放到传送带上
pm.run(mod)                     = 启动传送带，module 依次经过每个工具
```

`passes` 本身不执行任何变换，它只负责**往 `pm` 里注册**。真正执行变换是 `pm.run()` 的时候。

`pm` 是 pybind11 暴露的 C++ `PassManager` 类（`ir.cc:1889`），它内部管理着 Pass 的执行顺序、Pass 之间的依赖分析。

### 3.4 本项目的 Pass 流水线

#### 阶段 A：通用优化 (`make_ttir`) — compiler.py:138-146

```python
passes.common.add_inliner(pm)          # 函数内联
passes.ttir.add_combine(pm)            # 相邻操作合并
passes.common.add_canonicalizer(pm)    # 规范化（常量折叠等）
passes.ttir.add_reorder_broadcast(pm)  # 重排广播操作
passes.common.add_cse(pm)              # 公共子表达式消除
passes.common.add_licm(pm)             # 循环不变量外提
passes.common.add_symbol_dce(pm)       # 死代码消除
passes.ttir.add_loop_unroll(pm)        # 循环展开
pm.run(mod, 'make_ttir')               # 执行！
```

| Pass | 做什么 | 类比 |
|------|--------|------|
| `inliner` | 把函数调用展开（函数体嵌进去） | 把子程序的大括号内容复制到调用处 |
| `combine` | `a*2 + a*3` → `a*5` | 合并同类项 |
| `canonicalizer` | 统一 IR 格式，消除冗余 | 把 `x+0` 变成 `x` |
| `cse` | 相同计算只算一次 | 缓存重复结果 |
| `licm` | 循环里不变的东西移到循环外 | 把 `for i: a+1` 里不变的 `a+1` 提到循环前 |
| `loop_unroll` | 把循环体复制多份减少跳转 | 展开小循环 |

#### 阶段 B：Ascend 硬件适配 (`ttir_to_linalg`) — compiler.py:203-222

```python
ascend.passes.ttir.add_triton_to_structure(pm)           # 指针运算线性化
ascend.passes.ttir.add_discrete_mask_access_conversion(pm) # 非连续访存转换
ascend.passes.ttir.add_triton_to_unstructure(pm)          # 离散访存→标量循环
ascend.passes.ttir.add_triton_to_hivm(pm)                # 跨核同步指令
ascend.passes.ttir.add_triton_to_hfusion(pm)             # 算子融合
ascend.passes.ttir.add_triton_to_llvm(pm)                # 内联汇编→LLVM
ascend.passes.ttir.add_triton_to_linalg(pm)              # TTIR→Linalg（核心转换）
ascend.passes.ttir.add_dynamic_cv_pipeline(pm)           # 计算/向量流水优化
pm.run(mod, 'ttir_to_linalg')                            # 执行！
```

#### 整个流水线全景图

```
Triton Python 代码
      │ code_generator.py (构建 IR)
      ▼
┌──────── TTIR (Triton IR) ────────┐
│  tt.load, tt.store, arith.addf   │
│  scf.for, tt.make_range ...      │
└──────────────┬───────────────────┘
               │ 通用优化 Passes (inliner, combine, cse, licm...)
               ▼
┌──── 优化后的 TTIR ───────────────┐  ← 还是 TTIR，但更精简
│  常量折叠了、公共表达式消除了...    │
└──────────────┬───────────────────┘
               │ Ascend Passes (triton-to-structured, triton-to-linalg...)
               │ 这是 MLIR 世界的边界
               ▼
┌──── Linalg IR ──────────────────┐
│  linalg.generic, linalg.matmul  │  ← AscendNPU 编译器能理解的方言
└──────────────┬───────────────────┘
               │ ★ 这里不是 MLIR Pass 了，是外部编译器调用 ★
               │ BiSheng 编译器 (subprocess.run, 不是 MLIR Pass)
               ▼
┌──── kernel.o (二进制) ──────────┐
│  在 Ascend NPU 上执行             │
└─────────────────────────────────┘
```

**关键区分**：Linalg → 二进制不是一步 MLIR Pass，而是**三步外部命令行工具**逐步执行。所有 stage 在 `third_party/ascend/backend/compiler.py` 的 `add_stages()` 方法中注册：

```python
# third_party/ascend/backend/compiler.py:1225-1243
def add_stages(self, stages, options, language):
    if self.target.backend == "npu":
        stages["ttir"]      = lambda src, metadata: make_ttir(src, metadata, options)
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, ...)
        stages["mlirbc"]    = lambda src, metadata: linalg_to_bc_by_triton_mlir_opt(src, metadata, options)
        stages["bcmlir"]    = lambda src, metadata: bc_to_linalg_by_bishengir_opt(src, metadata, options)
        stages["npubin"]    = lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(src, metadata, options)
```

**Step 1: Linalg IR → MLIR Bytecode**。代码在 `third_party/ascend/backend/compiler.py:263-298`：

```python
# third_party/ascend/backend/compiler.py:263
def linalg_to_bc_by_triton_mlir_opt(linalg: str, metadata, opt):
    """Convert Linalg IR to MLIR Bytecode format using triton-mlir-opt."""
    triton_mlir_opt_path = _get_triton_mlir_opt_path()
    subprocess.run([
        triton_mlir_opt_path,          # 外部工具: triton-mlir-opt (项目编译产物)
        ttadapter_path,                # 输入: Linalg IR 文本文件
        "--emit-bytecode",             # 输出: MLIR 二进制格式
        "-o", bc_path,
    ], check=True, ...)
```

`triton-mlir-opt` 是项目编译时生成在 `bin/` 下的工具。这一步把 Linalg IR 文本转成 MLIR Bytecode 格式，方便跨进程传输。

**Step 2: MLIR Bytecode → Linalg IR 文本**。代码在 `third_party/ascend/backend/compiler.py:301-338`：

```python
# third_party/ascend/backend/compiler.py:301
def bc_to_linalg_by_bishengir_opt(bc_data: bytes, metadata, opt):
    """Convert MLIR Bytecode to MLIR text format using bishengir-opt."""
    bishengir_opt_path, env = _get_bishengir_opt_path()
    subprocess.run([
        bishengir_opt_path,            # 外部工具: bishengir-opt
        bc_path,                       # 输入: MLIR Bytecode
        "--mlir-print-debuginfo",
        "-o", mlir_path,               # 输出: Linalg IR 文本（还原回文本格式）
    ], env=env, ...)
```

注意函数名中的 "linalg"——`bishengir-opt` 只是把 Bytecode 还原为 Linalg IR 文本格式（格式转换），**不做方言转换**。Step 1+2 本质上是一个文本→二进制→文本的 round-trip，用于跨进程高效传输。

**Step 3: Linalg IR → (内部 AscendNPU IR) → kernel.o**。代码在 `third_party/ascend/backend/compiler.py:503-731`：

```python
# third_party/ascend/backend/compiler.py:503
def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    npu_compiler_path, env = _get_npucompiler_path()
    _compile_option_list = get_common_bishengir_compile_options(metadata)

    cmd_list = [npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file]

    # bishengir-compile 额外选项
    if npu_compiler_path.endswith("bishengir-compile"):
        _compile_option_list += [
            "--enable-hfusion-compile=true",
            "--enable-triton-kernel-compile=true",
        ]
    subprocess.run(cmd_list, env=env, ...)
    return Path(bin_path).read_bytes()          # 读回 .o 二进制
```

`bishengir-compile` 把 Linalg IR **直接**编译成 `kernel.o`。**AscendNPU IR（BiShengIR：HIVM/HACC/HFusion 方言）是 `bishengir-compile` 内部的中间表示**，不是独立的外部步骤。只有在 debug 时通过 `--bishengir-print-ir-after=hivm-graph-sync-solver` 才 dump 出来：

```python
# third_party/ascend/backend/compiler.py:688,706
_compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
# ...
_save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])
```

`_save_npuir_debug_output`（`compiler.py:457`）把 `bishengir-compile` 的 stdout 中的 IR dump 保存下来——NPUIR 是编译器 stdout 的输出内容，不是独立的文件产物。

**AscendNPU IR 的来源**：`third_party/ascend/AscendNPU-IR/` 是独立子模块，CMake 中引用（`third_party/ascend/CMakeLists.txt:8-21`）：

```cmake
# third_party/ascend/CMakeLists.txt:8-21
set(ASCENDNPU_IR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AscendNPU-IR" CACHE PATH
    "Path to AscendNPU-IR source root")
add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include)  # Tablegen'd files
```

它包含了 BiSheng 编译器的 HIVM Dialect、HACC Dialect、HFusion Dialect 等 IR 定义。编译时作为子模块一起编译，为 `bishengir-compile` 提供处理 NPUIR 所需的基础设施。

**完整链路总结**：

```
MLIR Pass (在 triton-ascend 进程内):
  TTIR → TritonToLinalg → Linalg IR

外部工具 (独立进程 subprocess.run):
  triton-mlir-opt:     Linalg IR 文本 → MLIR Bytecode (格式转换)
  bishengir-opt:       MLIR Bytecode → Linalg IR 文本 (格式转换，还原)
  bishengir-compile:   Linalg IR → (内部: AscendNPU IR) → kernel .o
                         ↑ NPUIR 是编译器的内部中间表示，仅 debug 时 dump
```

### 3.5 一个真实 Pass 的完整解剖：`TritonToLLVM`

每个 Pass 在项目里由**四个文件**构成，分布在三个位置：

```
third_party/ascend/
├── include/TritonToLLVM/
│   ├── Passes.td          ← ① TableGen 定义：Pass 的"身份证"
│   └── Passes.h           ← ② C++ 头文件：声明创建函数
├── lib/TritonToLLVM/
│   └── TritonToLLVM.cpp   ← ③ C++ 实现：Pass 的核心逻辑
└── triton_ascend.cc       ← ④ pybind11 注册：暴露给 Python
```

#### ① TableGen 定义 (`Passes.td`)

```tablegen
def TritonToLLVM : Pass<"triton-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Triton to LLVM dialect";
  let constructor = "triton::createTritonToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "tensor::TensorDialect", "arith::ArithDialect"];
}
```

这一小段 TableGen 注册一个 Pass：名字叫 `triton-to-llvm`，作用在 `ModuleOp` 上，依赖三个方言。

#### ② C++ 头文件 (`Passes.h`)

```cpp
namespace mlir::triton {
#define GEN_PASS_REGISTRATION
#include "ascend/include/TritonToLLVM/Passes.h.inc"  // 自动生成的基类

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLLVMPass();  // 工厂函数
}
```

#### ③ C++ 实现 (`TritonToLLVM.cpp`) — 核心骨架

```cpp
// 定义 Pass 结构体
struct TritonToLLVMPass
    : public mlir::triton::impl::TritonToLLVMBase<TritonToLLVMPass> {
  void runOnOperation() override;   // ← 每个 Pass 的入口函数
};

// 定义转换规则（Pattern）：匹配 tt.inline_asm → 替换为 llvm.inline_asm
struct ElementwiseInlineAsmOpConversion
    : OpRewritePattern<triton::ElementwiseInlineAsmOp> {

  LogicalResult matchAndRewrite(triton::ElementwiseInlineAsmOp op,
                                PatternRewriter &rewriter) const {
      // 用 rewriter.create<LLVM::InlineAsmOp>(...) 创建新操作
      // rewriter.replaceOp(op, outs);  ← 用新操作替换旧操作
  }
};

// runOnOperation — 整个 Pass 的入口
void TritonToLLVMPass::runOnOperation() {
  auto module = getOperation();

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect, tensor::TensorDialect, arith::ArithDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ElementwiseInlineAsmOpConversion>(...);

  applyPartialConversion(module, target, std::move(patterns));  // 执行！
}
```

**最核心的就是这三个东西**：

| 步骤 | 代码 | 做什么 |
|------|------|--------|
| **匹配** | `OpRewritePattern<Triton::ElementwiseInlineAsmOp>` | 找到所有 `tt.inline_asm` 操作 |
| **替换** | `rewriter.create<LLVM::InlineAsmOp>(...)` + `rewriter.replaceOp()` | 创建新的 `llvm.inline_asm`，替换旧的 |
| **执行** | `applyPartialConversion(module, target, patterns)` | 对 module 里所有匹配的操作执行替换 |

#### ④ pybind11 注册 (`triton_ascend.cc`)

```cpp
m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());
});
```

这一行让 Python 能写：`ascend.passes.ttir.add_triton_to_llvm(pm)`

### 3.6 四层文件的对应关系

```
Python 侧:                           C++ 侧:

compiler.py                          triton_ascend.cc (pybind11 桥接)
  │                                     │
  │ ascend.passes.ttir.add_triton_to_llvm(pm)
  │                                     │
  │                                     ▼
  │                                  m.def("add_triton_to_llvm", ...)
  │                                  → pm.addPass(createTritonToLLVMPass())
  │                                     │
  │                                     ▼
  │                                  Passes.h
  │                                  createTritonToLLVMPass()
  │                                     │
  │                                     ▼
  │                                  TritonToLLVM.cpp
  │                                  runOnOperation() {
  │                                    applyPartialConversion(
  │                                      module, target, patterns);
  │                                  }
  │                                     ↑
  │                                  Passes.td
  │                                  def TritonToLLVM : Pass<...>
```

| 文件 | 作用 | 类比 |
|------|------|------|
| `Passes.td` | TableGen 定义，声明 Pass 的名字、输入输出、依赖 | 身份证 |
| `Passes.h` | C++ 头文件，include 自动生成的代码 + 声明工厂函数 | 名片 |
| `*.cpp` | 核心实现：匹配什么操作、替换成什么 | 身体 |
| `triton_ascend.cc` | pybind11 注册，把 C++ 工厂函数暴露给 Python | 翻译官 |

### 3.7 所有 Pass 的目录一览

```
third_party/ascend/
├── include/                          # 头文件 + TableGen
│   ├── AutoBlockify/
│   ├── DiscreteMaskAccessConversion/
│   ├── DynamicCVPipeline/
│   ├── TritonControlFlowOpt/
│   ├── TritonToAnnotation/
│   ├── TritonToGraph/
│   ├── TritonToHFusion/
│   ├── TritonToHIVM/
│   ├── TritonToLinalg/               ← 最大最复杂的 Pass
│   ├── TritonToLLVM/                 ← 最简单的 Pass
│   ├── TritonToStructured/
│   ├── TritonToUnstructure/
│   └── Utils/
├── lib/                              # C++ 实现 (与 include 一一对应)
│   └── (同上 12 个目录)
└── triton_ascend.cc                  # pybind11 注册 (所有 Pass 都在这一个文件里)
```

---

## 阶段④：Ascend 后端 —— 为什么需要这些 Pass？

### 4.1 动机：Triton 是为 GPU 设计的，不是为 NPU

Triton 的原始设计面向 NVIDIA GPU。GPU 和 Ascend NPU 虽然都是 AI 加速器，但内部架构差异很大。

#### GPU vs Ascend NPU：内存层级对比

```
NVIDIA GPU 内存模型:              Ascend NPU 内存模型:

HBM (Global Memory)               GM (Global Memory / HBM)
  │                                  │
  ▼                                  ▼
L1 Cache + Shared Memory           L1 Buffer
  │                                  │
  ▼                                  ▼
Registers                          UB (Unified Buffer)
  │                                  │
  ▼                                  ▼
CUDA Core / Tensor Core            Cube Unit / Vector Unit
```

**关键差异**：
- GPU 的 Shared Memory 和 L1 Cache 是分离的；Ascend 的 L1 既是 cache 又是 buffer
- GPU 靠 warp scheduler 自动切换线程隐藏延迟；Ascend 需要编译器显式排流水
- GPU 有统一寻址模型；Ascend 的 UB/L1/GM 各有不同的访存指令

#### Triton 的假设 vs Ascend 的现实

| Triton 假设（GPU 思维） | Ascend 现实 |
|------------------------|------------|
| 指针运算直接映射为地址计算 | Ascend 不同内存空间有不同的指针类型 |
| block 间用 barrier 同步 | Ascend 有专门的跨核同步指令（HIVM） |
| load/store 是连续访存 | Ascend 对非连续访存需要特殊处理 |
| 计算和访存自动 overlap | Ascend 需要显式排计算和向量流水线 |

**每个 Ascend Pass 就是解决一个 GPU→NPU 的适配问题。**

### 4.2 逐个 Pass 解析

#### `TritonToStructured` — 指针线性化

**问题**：Triton 里你常写 2D 数据的行优先索引：
```python
offsets = row * WIDTH + col      # row * 1024 + col
row = offsets // 1024
col = offsets % 1024
```

GPU 上没问题。但 Ascend 不处理 `//` 和 `%` 在访存路径上的情况——Ascend 需要每个维度的偏移**分开表达**。

**TritonToStructured 做的事**：把 `//` 和 `%` 从指针/mask 表达式中消掉，拆成独立的维度。

```
原始:
  指针: ptr + x // 1024 * 4096 + x % 1024 * 4 + y
   mask: x // 1024 < 8 and x % 1024 < 1024

改写后:
  指针: ptr + x_offset + y_offset
   mask: x < 8192 and x_remainder < 1024
```

内部子步骤：

| 子步骤 | 做什么 |
|--------|--------|
| RewriteAddPtrOp | 分析指针表达式，提取每个维度的偏移信息，建模为 PtrState 对象 |
| CreateAddPtr | 用 PtrState 重建新的 AddPtrOp，消掉 `//` 和 `%` |
| RewriteLoadOp | 分析 mask 表达式，分解为各维度独立条件，建模为 MaskState |
| BuildMask | 重建新的 mask 表达式 |
| CreateLoad/CreateStore | 用新指针和新 mask 重建 load/store |

#### `DiscreteMaskAccessConversion` + `TritonToUnstructured` — 非连续访存

**问题**：GPU 上可以处理非连续 mask：
```python
x = tl.load(ptr, mask=[True, False, True, False, ...])
```

Ascend 的数据搬运单元处理不了非规则的 mask。

**DiscreteMaskAccessConversion**：把带非连续 mask 的 load/store 改写为"全量加载 + select"模式：
```
原始:
  %v = tt.load %ptr, %non_contiguous_mask

改写为:
  %all = tt.load %ptr          # 全量加载
  %v = select %non_contiguous_mask, %all, %other   # 用 mask 选择有效数据
```

**TritonToUnstructured**：把非连续轴上的 tensor 操作展开成标量循环：
```
原始 (tensor 操作):
  %v = tt.load %ptr[tensor<256x!tt.ptr<f32>>]

改写为 (标量循环):
  scf.for i = 0 to 256:
    scalar_load %ptr[i]
```

#### `TritonToLinalg` — 核心转换：TTIR → Linalg

这是整个 Ascend 后端的**核心转换**。它把 TTIR 的每个操作翻译为 Linalg（线性代数）方言的操作。

```
TTIR 操作              →    Linalg 操作
─────────────────────────────────────────
tt.load               →    memref.copy + ToTensorOp
tt.store              →    memref.copy
tt.make_range         →    linalg.generic
tt.splat              →    linalg.fill
arith.addf/mulf/...   →    linalg.generic (element-wise)
tt.dot (矩阵乘)       →    linalg.matmul
AtomicRMW             →    linalg.generic
```

**为什么是 Linalg？** 因为 BiSheng 编译器（Ascend 的下游编译器）的入口就是 Linalg 方言。它是连接 Triton 生态和 Ascend 生态的**桥梁方言**。

#### 其他 Pass 一句话

| Pass | 一句话 |
|------|--------|
| **TritonToHIVM** | Triton 的 block sync → Ascend 的跨核同步指令 |
| **TritonToHFusion** | 把多个操作融合（如 histogram）以减少数据搬运 |
| **TritonToLLVM** | Triton 内联汇编 → LLVM 内联汇编（给 CCE 用） |
| **BubbleUpOperation** | 把 extract/extract_slice 往上提，减少不必要循环 |
| **DynamicCVPipeline** | 计算和向量操作自动排流水，隐藏访存延迟 |
| **AutoBlockify** | 自动分解大 block，更好地利用 Ascend 的多核 |

### 4.3 阶段④小结

整个 Ascend 后端 Pass 的分类：

```
TTIR (通用 Triton IR)
  │
  ├─ 指针线性化 ───── TritonToStructured
  │   消掉 // 和 %，拆成独立维度
  │
  ├─ 非连续访存处理 ── DiscreteMaskAccessConversion + TritonToUnstructured
  │   把不规则访存变成标量循环或全量加载+select
  │
  ├─ 核心转换 ──────── TritonToLinalg
  │   TTIR → Linalg IR (与下游 BiSheng 编译器对接)
  │
  ├─ 算子融合 ──────── TritonToHFusion, BubbleUpOperation
  │   减少数据搬运
  │
  ├─ 同步翻译 ──────── TritonToHIVM
  │   block sync → Ascend 跨核指令
  │
  ├─ 流水调度 ──────── DynamicCVPipeline
  │   计算和向量流水线优化
  │
  └─ LLVM 适配 ─────── TritonToLLVM
     内联汇编转换
```

**核心理解**：每个 Pass 解决的都不是 Triton 语言的问题，而是 **Triton（GPU 思维）与 Ascend（NPU 硬件）之间的差距**。

---

## 阶段⑤：运行时与 Autotuning

### 5.1 动机：编译只是前半段

```
Python 代码 → TTIR → Pass 优化 → Linalg → BiSheng → 二进制 .o 文件
                                                          ↑
                                                    到这里，有文件了
                                                    但还没跑起来
```

剩下的问题：
1. 这个 `.o` 文件怎么加载到 NPU 上？
2. 怎么给它传参数（数据指针、BLOCK_SIZE 等）？
3. 编译时有那么多选项（BLOCK_SIZE 多大？要不要开 multibuffer？），怎么选最优的？

### 5.2 用户调用到 NPULauncher 的完整链路

```
① 用户代码
add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    │
    │  Python 语法: obj[key] 调用 __getitem__
    ▼
② JITFunction.__getitem__(grid)               ← jit.py:364
    return lambda *args, **kwargs: self.run(grid=grid, *args, **kwargs)
    │
    │  [grid] 只是"记住"了 grid，返回一个可调用对象
    ▼
③ JITFunction.run(*args, grid, **kwargs)      ← jit.py:695
    │
    ├─ 查缓存 (key = hash(参数 + 配置))
    │
    ├─ 缓存未命中 → 编译:
    │    _do_compile()                          ← jit.py:826
    │      └─ compiler.compile(src, target, options)   ← compiler.py:228
    │           ├─ src.make_ir()          → Module (TTIR)
    │           ├─ stages["ttir"]         → make_ttir (通用优化)
    │           ├─ stages["ttadapter"]    → ttir_to_linalg (Ascend Pass)
    │           ├─ stages["npubin"]       → 二进制文件 (缓存到磁盘)
    │           └─ return CompiledKernel(src, metadata_group, hash)
    │                                          ← compiler.py:386
    │
    ├─ 拿到 kernel (= CompiledKernel 对象)
    │
    ▼
④ kernel.run(grid_0, grid_1, grid_2, stream,   ← jit.py:744
             kernel.function, kernel.packed_metadata, ...)
    │
    │  kernel.run 是一个 property
    ▼
⑤ CompiledKernel.run (property)               ← compiler.py:501
    └─ _init_handles()                          ← compiler.py:464 (懒加载)
         │
         ├─ self._run = driver.active.launcher_cls(self.src, self.metadata)
         │            = NPULauncher(self.src, self.metadata)     ← driver.py:100
         │              │
         │              └─ 生成 C stub → 编译为 .so → import 拿到 launch 函数
         │
         └─ driver.active.utils.load_binary(    ← 加载 .o 二进制到设备
                self.kernel, ...)                ← self.kernel = 编译阶段缓存的二进制
    │
    ▼
⑥ NPULauncher.__call__(*args, **kwargs)        ← driver.py:131
    └─ self.launch(*args, **kwargs)             ← 通过 CANN 在 NPU 上执行
```

### 5.3 Module 如何变成 Driver 能用的二进制

**Module 不是直接传给 Driver，而是通过"编译→缓存→读取"的间接方式。**

```
compiler.compile() 内部:
─────────────────────
  module = src.make_ir()                         ① 生成初始 TTIR module

  for ext, compile_ir in stages.items():         ② 逐个 stage 执行
      next_module = compile_ir(module, metadata)
      │
      │  stages["ttir"]     → make_ttir      → 优化后的 TTIR module
      │  stages["ttadapter"] → ttir_to_linalg → Linalg IR module
      │  stages["npubin"]   → linalg→二进制   → 二进制文件 (subprocess.run BiSheng)
      │
      │  每个 stage 的输出都被缓存:
      metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
      module = next_module                          ③ 更新 module

  return CompiledKernel(src, metadata_group, hash)  ④ 返回时已不包含 module 对象
─────────────────────

CompiledKernel.__init__():
  self.asm = {                                      ⑤ 从磁盘读取所有 stage 产物
      'ttir':     (TTIR 文本),
      'ttadapter': (Linalg IR 文本),
      'mlirbc':   (MLIR Bytecode),                   ← triton-mlir-opt 产物
      'bcmlir':   (AscendNPU IR 文本),               ← bishengir-opt 产物
      'o':        (二进制字节),       ← 这个就是给 Driver 用的
  }
  self.kernel = self.asm['o']                       ⑥ 二进制就绪

CompiledKernel._init_handles():
  self.module, self.function, ... = driver.active.utils.load_binary(
      self.metadata.kernel_name,                    ⑦ 加载二进制到 NPU
      self.kernel,                                   ← 就是上面的 self.asm['o']
      self.metadata.shared, device, ...)
```

### 5.4 kernel.o、C Stub、.so 三者的关系

#### NPULauncher 初始化

```python
# driver.py:100
class NPULauncher:
    def __init__(self, src, metadata):
        # 第一步：生成 C stub 源码 + 编译为 .so
        self.so_launcher_path = self._make_launcher_stub_path()

        # 第二步：动态加载 .so，拿到 launch 函数
        mod = importlib.util.spec_from_file_location(
            "__triton_launcher", self.so_launcher_path)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")     # ← C stub 里的 launch 函数
```

#### `_make_launcher_stub_path` 做什么

```python
def _make_launcher_stub_path(self):
    # ① 生成 C 头文件 (include ACL/CANN 头文件)
    header_src = generate_npu_header_src()

    # ② 生成 C++ wrapper 代码 (参数解析、类型转换、kernel 调用)
    wrapper_src = make_launcher(constants, signature, self.metadata)

    # ③ header + wrapper → 一个 .cxx 文件 → 编译为 .so
    return make_npu_launcher_stub(header_src, wrapper_src, debug)
```

#### `make_npu_launcher_stub`

```python
def make_npu_launcher_stub(header_src, wrapper_src, debug):
    # 写入临时 .cxx 文件
    src_path = os.path.join(tmpdir, "launcher.cxx")
    with open(src_path, "w") as f:
        f.write(wrapper_src)

    # 用 CANN 编译器编译 .cxx → .so
    so_path = _build_npu_ext("launcher", src_path)
    return so_path
```

#### 三者关系图

```
┌──────────────────────────────────────────────────────────────┐
│                    编译阶段                                   │
│                                                              │
│  Triton 代码 → MLIR Passes → Linalg IR → Bytecode → AscendNPU IR → kernel.o │
│                                                              │
│  kernel.o = "算什么的代码" (设备端执行的计算逻辑)               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    运行阶段 (首次调用时)                       │
│                                                              │
│  ① 生成 C Stub (文本)                                        │
│     header_src: #include CANN 头文件                          │
│     wrapper_src:                                                │
│       - PyArg_ParseTuple 解析 Python 参数                      │
│       - 分配/管理 NPU 内存                                     │
│       - 加载 kernel.o 到设备                                   │
│       - 调用 CANN API 执行 kernel                              │
│                                                              │
│  ② 编译 C Stub → launcher.so                                 │
│     _build_npu_ext("launcher", "launcher.cxx")               │
│                                                              │
│  ③ 动态加载 .so → 拿到 Python 可调用的 launch 函数             │
│     mod = importlib.load("launcher.so")                       │
│     self.launch = mod.launch                                  │
│                                                              │
│  ④ 用户每次调用 → launch(*args) → CANN → NPU 执行             │
└──────────────────────────────────────────────────────────────┘
```

#### 类比

| 组件 | 类比 | 语言 | 跑在哪 |
|------|------|------|--------|
| `kernel.o` | 计算逻辑本身（"怎么算加法"） | 汇编/机器码 | **NPU 设备端** |
| C Stub → `.so` | 胶水层（"传参数、调 NPU"） | C++/CANN API | **CPU 主机端** |
| `NPULauncher` | Python 入口（"给用户调用的接口"） | Python | **CPU 主机端** |

`kernel.o` 和 `.so` 是**两件独立的事**：
- `kernel.o` = 编译 Triton 代码得到，NPU 上执行的算法
- `.so` = 自动生成的 C 代码编译得到，负责参数转换和 CANN 调用

`NPULauncher` 通过 Python 的 `importlib` 加载 `.so`，拿到 `launch()` 函数。用户每次调用 kernel 时，`launch()` 就把 `kernel.o` 连同参数一起送到 NPU 执行。

### 5.5 Autotuning：怎么找到最优配置？

#### 问题

同一个 kernel，不同参数性能天差地别：
```python
add_kernel(x, y, out, n, BLOCK_SIZE=128)    # 哪个最快？不知道
add_kernel(x, y, out, n, BLOCK_SIZE=256)
add_kernel(x, y, out, n, BLOCK_SIZE=512)
```

而且 Ascend 有更多选项（十几个编译选项）。

#### Autotuning 工作流程

```python
# autotuner.py:2023 — Autotuner.run()
def run(self, *args, **kwargs):
    key = self.generate_key_and_configs(*args, **kwargs)

    if key not in self.cache:          # ① 缓存没命中？需要重新调优
        pruned_configs = self.prune_configs(kwargs)       # ② 筛掉明显不行的配置

        timings = self._batch_bench(                      # ③ 每个配置都跑一遍测耗时
            *args, configs=pruned_configs, **kwargs
        )

        self.cache[key] = min(timings, key=timings.get)   # ④ 选最快的，缓存

    config = self.cache[key]
    return self.fn.run(*args, **config.all_kwargs(), **kwargs)  # ⑤ 用最优配置执行
```

#### 直观理解

```
用户代码:
  @triton.autotune(configs=[
      triton.Config(kwargs={'BLOCK_SIZE': 128}),
      triton.Config(kwargs={'BLOCK_SIZE': 256}),
      triton.Config(kwargs={'BLOCK_SIZE': 512}),
  ], key=['n_elements'])
  @triton.jit
  def add_kernel(...): ...

第一次调用 add_kernel(x, y, out, n=10000):
  ① key = hash(参数 + 配置列表) → 缓存未命中
  ② 筛掉不合适的配置 (如 BLOCK_SIZE=512 对 n=10000 太大)
  ③ batch_bench: 用 BLOCK_SIZE=128 跑一次, 用 256 跑一次
  ④ 选最快的 → 比如 256 最快 → 缓存 key → 256
  ⑤ 用 BLOCK_SIZE=256 执行，返回结果

第二次调用 add_kernel(x2, y2, out2, n=10000):
  ① key 相同 → 缓存命中 → 直接用 BLOCK_SIZE=256
```

#### Autotuning 工具集

| 模块 | 位置 | 做什么 |
|------|------|--------|
| `autotuner.py` | `runtime/autotuner.py` | 主调优引擎：生成配置、bench、选最优 |
| `autoparser.py` | `runtime/autoparser.py` | 解析 kernel 代码，找出可调优参数 |
| `tile_generator.py` | `runtime/tile_generator.py` | 自动生成 tile size 候选 |
| `ubtuner.py` | `runtime/ubtuner.py` | UB (Unified Buffer) 专用调优 |
| `costmodel_runtime.py` | `runtime/costmodel_runtime.py` | 调用 C++ Cost Model 预估性能（不用真跑） |
| `dsl_analysis/` | `runtime/dsl_analysis/` | 对 Triton 代码做静态分析，辅助调优 |

---

## 阶段⑥：全景回顾

### 端到端完整链路

```
add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
  │
  │  ┌─────────────────── 编译阶段 (首次调用) ───────────────────┐
  │  │                                                           │
  │  │  Python 代码                                               │
  │  │    → code_generator 遍历 AST, 通过 semantic.py 调 builder │
  │  │    → 构建 TTIR Module                                     │
  │  │    → make_ttir (通用优化: inline, combine, cse, licm...)  │
  │  │    → ttir_to_linalg (Ascend Pass x12)                     │
  │  │    → Linalg IR 写入磁盘                                    │
  │  │    → BiSheng 编译器 (subprocess.run) 生成 .o 二进制        │
  │  │    → 所有中间产物缓存到磁盘                                 │
  │  │    → 返回 CompiledKernel                                  │
  │  │                                                           │
  │  └───────────────────────────────────────────────────────────┘
  │
  │  ┌─────────────────── 运行阶段 (首次调用) ────────────────────┐
  │  │                                                           │
  │  │  CompiledKernel._init_handles()                            │
  │  │    → 从磁盘读取 .o 二进制 → load_binary 加载到 NPU         │
  │  │    → NPULauncher: 生成 C Stub 代码                         │
  │  │    → 编译 C Stub → launcher.so                            │
  │  │    → importlib 加载 .so → 拿到 launch 函数                 │
  │  │    → 缓存 .so 和 .o                                       │
  │  │                                                           │
  │  └───────────────────────────────────────────────────────────┘
  │
  │  ┌─────────────────── 每次调用 ──────────────────────────────┐
  │  │                                                           │
  │  │  NPULauncher.__call__(*args)                               │
  │  │    → launch(*args) → CANN Runtime → NPU 执行               │
  │  │                                                           │
  │  └───────────────────────────────────────────────────────────┘
  │
  ▼
  NPU 上实际执行计算，返回结果
```

### 关键人物表

| 角色 | 类/函数 | 文件 |
|------|---------|------|
| Triton 语言 API | `tl.load`, `tl.store`, `tl.program_id` 等 | `python/triton/language/core.py` |
| 语义层 | `TritonSemantic` | `python/triton/language/semantic.py` |
| IR 构建器 | `TritonOpBuilder` (C++, pybind11 暴露) | `python/src/ir.cc`, `ir.h` |
| AST 遍历 + Module 创建 | `CodeGenerator` | `python/triton/compiler/code_generator.py` |
| 编译入口 | `JITFunction.run()` | `python/triton/runtime/jit.py` |
| 编译调度 + Pass 流水线 | `compiler.compile()` | `python/triton/compiler/compiler.py` |
| Ascend Pass 实现 | 12 个 Pass 目录 | `third_party/ascend/lib/` |
| Ascend Pass 注册 | `add_triton_to_*` 函数 | `third_party/ascend/triton_ascend.cc` |
| 编译产物 | `CompiledKernel` | `python/triton/compiler/compiler.py` |
| 设备加载 + 启动 | `NPULauncher`, `NPUDriver` | `third_party/ascend/backend/driver.py` |
| Autotuning | `AutoTilingTuner` | `third_party/ascend/backend/runtime/autotuner.py` |
| 磁盘缓存 | `fn_cache_manager` | `python/triton/runtime/cache.py` |

### 核心数据流向

```
Triton Python 代码
  │  CodeGenerator (AST 遍历)
  ▼
TTIR (Module, MLIR 内存对象)
  │  make_ttir (通用优化 Pass)
  ▼
优化后的 TTIR
  │  ttir_to_linalg (Ascend Pass x12)
  ▼
Linalg IR
  │  triton-mlir-opt (外部调用, 文本→Bytecode)
  ▼
MLIR Bytecode
  │  bishengir-opt (外部调用, Bytecode→文本, 格式还原)
  ▼
Linalg IR (文本)
  │  bishengir-compile (外部调用, 内部包含 AscendNPU IR 阶段)
  ▼
kernel.o (二进制, 磁盘文件)
  │  CompiledKernel 从磁盘读取
  │  NPULauncher 加载到 NPU
  │  NPULauncher 生成 C Stub → 编译 .so → import
  ▼
CANN Runtime → Ascend NPU 执行
```

### 关键区分

| 容易混淆的东西 | 区分 |
|-------------|------|
| MLIR Pass vs 外部编译器 | MLIR Pass = IR→IR 变换（在 MLIR 框架内）；BiSheng = 外部命令行工具，Linalg→二进制 |
| `.o` vs `.so` | `.o` = NPU 设备端执行的计算逻辑（编译 Triton 得到）；`.so` = CPU 主机端的胶水代码（生成 C 代码编译得到） |
| `passes` vs `pm` | `passes` = 工具箱（注册 Pass 的模块）；`pm` = 传送带（PassManager，管理执行） |
| Module vs CompiledKernel | Module = MLIR 内存对象（编译过程中）；CompiledKernel = 编译完成后的包装对象（持有二进制引用） |

---

> 本课程基于 Triton-Ascend 3.6.0 源码编写。
