# Autoscope 教程：A5 上的 SIMT vs SIMD 决策，以及 Costmodel 要做什么

## 一、起点：从一个具体例子看 SIMT 和 SIMD 的区别

假设你要实现一个 **gather** 操作：有一个数据数组 `data` 和一个索引数组 `indices`，从 `data` 里按 `indices` 取出对应位置的元素。

```python
import triton
import triton.language as tl
import torch

@triton.jit
def gather_kernel(
    data_ptr,          # 数据数组 [N]
    indices_ptr,       # 索引数组 [M]，每个值在 [0, N) 之间
    output_ptr,        # 输出 [M]
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    # 当前 program 负责的 M 维度范围
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # 从 indices 加载：这是连续访存，对硬件友好
    idx = tl.load(indices_ptr + offs, mask=mask)

    # 用 idx 去 data 里取数据：这是离散/间接访存！
    # 因为 idx 的值是不可预测的（任意分布在 [0, N) 之间）
    val = tl.load(data_ptr + idx, mask=mask)

    tl.store(output_ptr + offs, val, mask=mask)
```

关键在 `tl.load(data_ptr + idx, mask=mask)` 这一行——**`idx` 不是一个等差数列，而是从 `indices` 数组里动态读出来的**。这叫 **离散访存（indirect/scattered access）**。

### 离散访存在 SIMD 和 SIMT 下的行为

- **SIMD（向量核）**：不支持"用运行时值做索引去访存"。编译器只能把这一行 **展开成标量循环**——逐元素加载 `indices[0]` → 取 `data[indices[0]]` → `indices[1]` → ... → **非常慢**
- **SIMT（类 GPU 核）**：原生支持 `indirect_load` 指令——硬件一条指令就能并行完成"用索引数组去取数据"

### 怎么让 kernel 走 SIMT？

调用时传 `compile_mode`：

```python
# 强制走 SIMT
gather_kernel[grid](data, indices, output, N, M, BLOCK=128,
                    compile_mode='simt_only')

# 混合模式（默认）：连续访存走 SIMD，离散访存走 SIMT
gather_kernel[grid](data, indices, output, N, M, BLOCK=128,
                    compile_mode='unstructured_in_simt')

# 强制走 SIMD（离散访存会被展开成标量循环）
gather_kernel[grid](data, indices, output, N, M, BLOCK=128,
                    compile_mode='simd')
```

看到 `compile_mode='simt_only'`。**autoscope 的目标就是：不用人手动写 `compile_mode`，而是自动判断这个 kernel 该走哪个模式。**

---

## 二、背景：为什么 A5 有两套编译路径

回到上面的 gather 例子，compiler 在编译 `tl.load(data_ptr + idx, ...)` 时，发现 `idx` 不是等差数列（`isStructured() == false`）——它判断这是**非结构化访存**。

### 昇腾 950 的两类计算核心

| 核心 | 处理离散访存的方式 | 编译路径 | TTIR → |
|------|-------------------|---------|--------|
| **SIMD**（向量核） | 展开成标量循环，逐元素加载 | Linalg IR → AscendNPU IR（传统路径） | 带上层优化 |
| **SIMT**（类 GPU 核） | `indirect_load` 硬件指令，并行取数 | 跳过 Linalg，Triton IR → AscendNPU IR 原生 | 轻量，解耦 |

对于 gather 的 `data_ptr[idx]`：
- SIMD：编译器看到"非结构化"，把 `tl.load(data_ptr + idx)` 拆成 for 循环，逐个 `idx[i]` 去 `data` 里取 → 几十上百倍慢
- SIMT：编译器识别出"间接寻址模式"，生成 `ascend.indirect_load` → 单条指令完成

### 问题：谁来选？怎么选？

SIMT 不是万能的——它有 dispatch 开销、warp 调度开销。对于**全连续访存**的 kernel（比如普通 matmul），SIMD 反而更快。

当前是**静态规则**在做选择，下一节看具体是怎么做的。

---

## 三、现状：静态规则决策

在 `compiler.py:1125-1158`，`compile_mode` 有三个可选值：

| `compile_mode` | `force_simt_template` | `force_simt_only` | 含义 |
|---|---|---|---|
| `"simd"` | False | False | 全部 SIMD |
| `"unstructured_in_simt"`（**默认**） | **True** | False | 混合：离散访存走 SIMT，其余 SIMD |
| `"simt_only"` | — | True | 全部 SIMT，跳过整个 Linalg 降级链路 |

### 3.1 `compile_mode` 的 Python 端处理

文件：`third_party/ascend/backend/compiler.py:1127-1163`

```python
# NPUOptions 中的定义
compile_mode: str = "unstructured_in_simt"  # 默认混合模式
force_simt_template: bool = False
force_simt_only: bool = False

def __post_init__(self):
    if self.compile_mode == "simd":
        object.__setattr__(self, "parallel_mode", "simd")
    elif self.compile_mode == "unstructured_in_simt":
        object.__setattr__(self, "force_simt_template", True)  # 启用混合
    elif self.compile_mode == "simt_only":
        object.__setattr__(self, "force_simt_only", True)       # 纯 SIMT
        object.__setattr__(self, "parallel_mode", "simt")
```

### 3.2 `compile_mode` 如何影响编译链路

文件：`third_party/ascend/backend/compiler.py:1315-1330`

```python
stages["ttir"] = _wrap("ttir", lambda src, metadata: make_ttir(src, metadata, options))
if options.force_simt_only:
    # SIMT 路径：TTIR 直接转 npubin，跳过 Linalg → AscendNPU IR 链路
    stages["npubin"] = _wrap("npubin",
        lambda src, metadata: ttir_to_npubin(src, metadata, options))
    return  # ← 提前返回，不走下面的 linalg/bc 阶段
# SIMD 路径：完整链路
stages["ttadapter"] = _wrap("ttadapter",
    lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True))
# ... bytecode/linalg_to_bin ...
```

### 3.3 `compile_mode` 如何传入 MLIR Pass

文件：`third_party/ascend/backend/compiler.py:240-267`

```python
force_simt_template = metadata["force_simt_template"]

# 传递给两个关键的 TTIR pass
add_discrete_mask_access_conversion(pm, compile_on_910_95, force_simt_template, ...)
add_triton_to_unstructure(pm, compile_on_910_95, force_simt_template)
```

### 3.4 Autotuner 中的 SIMT 感知

文件：`third_party/ascend/backend/runtime/autotuner.py:1996`

```python
def generate_key_and_configs(self, *args, **kwargs):
    # 从 kwargs 中读取 compile_mode，判断是否 SIMT 模式
    self.is_simt_mode = (
        kwargs.get("force_simt_only", False)
        or kwargs.get("compile_mode") == "simt_only"
    )
```

SIMT 模式下 autotuner 的行为变化（`autotuner.py:1898-1901`）：
- **SIMT**：展开 configs 时加不同 `num_warps` 值（如 16, 32）
- **SIMD**：展开 configs 时 toggle `multibuffer`（ping-pong pipeline）

---

## 四、混合模式：操作级的 SIMT/SIMD 决策

现在重点看默认模式 `"unstructured_in_simt"` 的决策逻辑。它在 **操作级别**（单个 load/store）决定走 SIMT 还是 SIMD。

### 4.1 两步 Pass 协作

1. **`discrete-mask-access-conversion`**：分析 load/store 的 mask 是否连续

   文件：`third_party/ascend/lib/TritonToUnstructure/DiscreteMaskAccessConversion.cpp`

   - 如果 mask 非连续 → 打上标记 `route_discrete_mask_to_simt`
   - 如果 mask 连续 → 拆成连续/离散边界，用 load + select / store 处理

2. **`triton-to-unstructure`**：对打了标记或判定为非结构化的访存，尝试转为 SIMT 快速通道

### 4.2 核心决策代码

文件：`third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp:532-542`

```cpp
// 是否启用 SIMT 间接访存快速通道
bool indirectFastPathEnabled =
    compileOn91095Flag                  // 1. 必须是 A5 芯片
    && forceSimtTemplateFlag            // 2. 必须是混合模式
    && ((!ptrOffsetInfo.isStructured()  // 3a. 访存是非结构化的
         && sizeInByte < 64)           //     且连续尺寸 < 64 字节
        || routeDiscreteMaskToSimt);    // 3b. 或离散 mask 已标记路由到 SIMT

// 额外限制：rank 不能超过 5
bool rankWithinLimit = resultShape.size() <= 5;

if (indirectFastPathEnabled && rankWithinLimit) {
    // 走 SIMT：tt.load → tt.indirect_load
    //          tt.store → tt.indirect_store
    //          tt.atomic_* → hivm.custom("__builtin_indirect_atomic")
} else {
    // 走 SIMD：回退为标量循环展开
}
```

### 4.3 SIMT 转换示例

以下 MLIR 测试展示了 `tt.load` 转换为 `ascend.indirect_load` 的过程：

文件：`third_party/ascend/unittest/Conversion/General/TritonToUnstructure/indirect_load.mlir`

```
// RUN: triton-opt --triton-to-structured
// RUN:   '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True'
// RUN:   '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True'

// 输入：一个间接索引的 load
%8 = tt.load %7 : tensor<32x!tt.ptr<i64>>  // 通过指针数组间接加载索引
%19 = tt.load %18 : tensor<8x32x!tt.ptr<f32>>  // 用加载的索引做二次间接寻址

// 输出（SIMT 路径）：转为 ascend.indirect_load
```

---

## 五、当前决策的问题

这套规则有两个缺陷：

1. **纯静态**：只看"是否非结构化"、"size < 64"，不看实际的运行时数据量、计算强度、内存带宽。一个 size=60 的非结构化访存可能数据量极小，SIMD 标量循环也很快，但被静态规则强制走了 SIMT（dispatch 开销反而更大）
2. **粒度太细**：在 **操作级别**（单个 load/store）做决策，而不是 **kernel 级别**。一个 kernel 里可能同时有结构化算子和离散访存—混合模式各自走各自的路径—但整体开销可能因为核心切换带来的同步成本而更高

---

## 六、Costmodel 要做什么

思路：用 costmodel 在 **kernel 级别** 预测 cycle 数，比较 SIMD 和 SIMT 两种编译路径的预估性能。

当前 costmodel 能预测 kernel 在 SIMD 路径下的 cycle：

```
Python kernel
  → generate_ttir_for_costmodel()  →  AST → TTIR → make_ttir()
  → run_costmodel_inproc(ttir_text) →  C++ PipelineAnalysisPass  →  cycle estimate (SIMD 硬件模型)
```

要让 costmodel 服务 autoscope，需要比较两个编译路径：

```
同一个 Python kernel
  ├─ Path A: compile_mode="simd" → costmodel → cycles_simd
  └─ Path B: compile_mode="simt_only" → costmodel → cycles_simt
                                        ↓
                               choose min(cycles_simd, cycles_simt)
                                        ↓
                     自动选择 compile_mode，传给下游编译
```

### 6.1 需要做的事：让 costmodel 支持 SIMT 模式的 TTIR 编译

文件：`third_party/ascend/backend/compiler.py:155` — `generate_ttir_for_costmodel`

当前固定走 SIMD 编译链路（`ast_to_ttir` → `make_ttir` → 得到 SIMD 优化后的 TTIR）。SIMT 路径（`"simt_only"`）是不同的：它跳过 `make_ttir` 的大部分优化、不经过 Linalg 降级。

需要增加分支：当指定 `compile_mode="simt_only"` 时，走 SIMT 的 TTIR 编译路径。

方法：在 `_pack_args` 时传入 `compile_mode="simt_only"`。参考 `compiler.py:1315-1317`：

```python
if options.force_simt_only:
    stages["npubin"] = ttir_to_npubin  # SIMT 路径：TTIR 直接转 npubin
    return  # 不经过 linalg
```

### 6.2 需要做的事：让 costmodel C++ pass 支持 SIMT 的 cycle 估计

文件：`third_party/ascend/costmodel/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp`

当前 `PipelineAnalysisPass` 基于 SIMD 硬件模型（Cube/Vector 单元、流水线调度、roofline model）。SIMT 的硬件模型不同——没有 Cube/Vector 分离，而是 warp 并行。

需要为 SIMT 路径写一个新的或修改现有的 pass，输入 `compile_mode` 参数，使用不同的硬件参数估计（SIMT warp 数、shared memory 带宽、SIMT 指令延迟等）。

### 6.3 需要做的事：在 autotuner 中集成比较逻辑

文件：`third_party/ascend/backend/runtime/autotuner.py:2277` — `AutoTilingTuner.run()`

对每个 kernel，编译 TTIR 并评估两种路径，自动选择：

```python
if self.enable_autoscope:  # 新开关，例如 TRITON_ENABLE_AUTOSCOPE
    # 编译 TTIR for SIMD
    ttir_simd = self._costmodel_compile_ttir(config, compile_mode="simd", ...)
    # 编译 TTIR for SIMT
    ttir_simt = self._costmodel_compile_ttir(config, compile_mode="simt_only", ...)

    # 分别评估
    cycles_simd = run_costmodel(ttir_simd, ...)   # SIMD 硬件模型
    cycles_simt = run_costmodel(ttir_simt, ...)   # SIMT 硬件模型（需新建）

    # 决策
    best_mode = "simt_only" if cycles_simt < cycles_simd else "simd"
    kwargs["compile_mode"] = best_mode
```

### 6.4 需要做的事：JIT cache key 加入 compile_mode

不同 `compile_mode` 下同一个 kernel 会生成不同的二进制 → 需要不同的 JIT cache key。当前 JIT cache key 不包含 `compile_mode`，需要加上。

文件：`python/triton/runtime/jit.py` — `compute_cache_key`

---

## 七、相关源码索引

| 文件 | 内容 |
|------|------|
| `docs/zh/architecture_design_and_core_features.md` | SIMD/SIMT 架构说明 (§3.2.2, §3.2.3) |
| `third_party/ascend/backend/compiler.py:1125-1163` | `NPUOptions.compile_mode` 定义与 `__post_init__` |
| `third_party/ascend/backend/compiler.py:1315-1330` | `compile_mode` 影响编译 stage 选择 |
| `third_party/ascend/backend/compiler.py:240-267` | `force_simt_template` 传入 MLIR Pass |
| `third_party/ascend/backend/runtime/autotuner.py:1996` | Autotuner 中的 `is_simt_mode` 判断 |
| `third_party/ascend/backend/runtime/autotuner.py:355-368` | SIMT config 展开（num_warps） |
| `third_party/ascend/backend/runtime/autotuner.py:397-432` | SIMD config 展开（multibuffer） |
| `third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp:532-542` | **核心决策**：操作级 SIMT/SIMD 选择 |
| `third_party/ascend/costmodel/lib/AscendModel/Transforms/PipelineAnalysisPass.cpp` | 当前 costmodel pass（仅 SIMD） |
| `third_party/ascend/backend/runtime/costmodel_runtime.py` | costmodel Python 接口 |
| `third_party/ascend/backend/compiler.py:155` | `generate_ttir_for_costmodel` 入口 |
| `third_party/ascend/unittest/autotune_ut/test_reduce_simt.py` | SIMT demo/test |
| `third_party/ascend/unittest/Conversion/General/TritonToUnstructure/indirect_load.mlir` | SIMT 转换 MLIR 测试 |

---

## 八、进度总结

| 已完成 | 待做 |
|--------|------|
| costmodel 可预测 SIMD 路径 cycle | costmodel 支持 SIMT 路径的 TTIR 编译 |
| costmodel 已接入 autotuner | costmodel 支持 SIMT 硬件模型的 cycle 估计 |
| `compile_mode` 三种模式已就绪 | `autoscope_decision` 集成到 `run()` |
| 混合模式操作级决策已实现 | JIT cache key 加入 `compile_mode` |
| `_build_costmodel_arg_bindings` 正确处理 constexpr | |
