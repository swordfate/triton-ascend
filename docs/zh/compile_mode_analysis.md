# compile_mode 详解：SIMD/SIMT 模板与混编

> **分支**: `kanuak/simd-simt-compile-mode`  
> **对应 commit**: `upstream/kanuak/simd-simt-compile-mode` (2026-08)  
> **代码仓库**: triton-ascend (主仓库)

---

## 目录

1. [概述：compile_mode 是什么](#1-概述compile_mode-是什么)
2. [四种 compile_mode 一句话总结](#2-四种-compile_mode-一句话总结)
3. [complete_mode 的完整参数流](#3-compile_mode-的完整参数流)
4. [NPUOptions：Python 侧入口](#4-npuoptionspython-侧入口)
5. [compile_mode 如何影响 C++ 编译 Pass](#5-compile_mode-如何影响-c-编译-pass)
6. [simd_simt vs simd_simt_template：关键分岔](#6-simd_simt-vs-simd_simt_template关键分岔)
7. [UnstructuredLoadOp/UnstructuredStoreOp：统一中间表示](#7-unstructuredloadopunstructuredstoreop统一中间表示)
8. [scope.vec_mode：代码块级别的 SIMD/SIMT 选择](#8-scopevec_mode代码块级别的-simdsimt-选择)
9. [BishengIR 侧：--enable-simd-simt-mix-compile](#9-bishengir-侧--enable-simd-simt-mix-compile)
10. [parallel_mode 和 runtime 行为](#10-parallel_mode-和-runtime-行为)
11. [废弃的旧选项和迁移路径](#11-废弃的旧选项和迁移路径)
12. [总结：四种模式的对比表](#12-总结四种模式的对比表)

---

## 1. 概述：compile_mode 是什么

`compile_mode` 是 NPUOptions 的一个字符串参数，控制 kernel 的**编译路线**——即 TTIR 中的算子如何被 lower 到最终的可执行代码。它统一了之前散落在多个 boolean flag（`force_simt_template`、`force_simt_only`）中的编译路线选择。

核心问题：A5 (Ascend 910_95 / 950) 芯片有 SIMD（向量核心）和 SIMT（类 GPU 标量核心）两种执行单元。对于**非结构化内存访问**（indirect/gather/scatter），有两种处理方式：

- **SIMD 方式**：硬件 gather_load/scatter_store 指令（hfusion 路径）
- **SIMT 方式**：软件模板（triton_indirect_load/triton_indirect_store 函数调用）

`compile_mode` 就是选择用哪种方式。

---

## 2. 四种 compile_mode 一句话总结

```
simd
  └─ 纯 SIMD 路径。所有算子走传统向量核心编译。

simd_simt
  └─ 混合编译，硬件路径。非结构化内存访问 → hfusion.gather_load/scatter_store。
     结构化访问 → SIMD 正常路径。
     需要 BishengIR 的 --enable-simd-simt-mix-compile。

simd_simt_template
  └─ 混合编译，软件模板路径。非结构化内存访问 → SIMT 模板函数调用
     (triton_indirect_load / triton_indirect_store)。
     结构化访问 → SIMD 正常路径。

simt_only
  └─ 纯 SIMT 路径。整个 kernel 走 SIMT 编译。
```

**注意**：`simt_template` 和 `unstructured_in_simt` 是 `simd_simt_template` 的废弃别名，仍然接受但会触发 DeprecationWarning。

---

## 3. compile_mode 的完整参数流

```
用户代码: @triton.autotune(..., compile_mode="simd_simt")
    │
    ▼
NPUOptions.compile_mode = "simd_simt"
    │
    ▼
NPUOptions.__post_init__():
    ├─ parallel_mode = "mix_simd_simt"
    ├─ shared_mem_dynamic_size = 221184
    └─ force_simt_template / force_simt_only (如果是旧选项触发)
    │
    ▼
metadata["compile_mode"] = "simd_simt"
metadata["parallel_mode"]  = "mix_simd_simt"
    │
    ▼
─── TTIR → Linalg 阶段 ─────────────────────────────────────
    compiler.py: _validate_compile_mode(metadata["compile_mode"])
    → 传给各 C++ pass:
      • DiscreteMaskAccessConversion(compileMode)
      • TritonToUnstructure(compileMode)
      • TritonToLinalg(compileMode)
    │
    C++ 侧: parseCompileMode("simd_simt") → CompileMode::SimdSimt
    │
    ▼
─── Linalg → Binary 阶段 ────────────────────────────────────
    compiler.py:
    if compile_mode == "simd_simt":
        bishengir_flags += ["--enable-simd-simt-mix-compile"]
        bishengir_flags += ["--num-warps=...", "--threads-per-warp=..."]
    │
    ▼
BiShengIR 编译器: hfusion.gather_load / hfusion.scatter_store
```

---

## 4. NPUOptions：Python 侧入口

### 4.1 compile_mode 字段定义

```python
# compiler.py NPUOptions

compile_mode: str = "simd"

# 合法值:
#   "simd"                 → 纯 SIMD (默认)
#   "simd_simt"            → 混编，硬件 gather/scatter
#   "simd_simt_template"   → 混编，SIMT 软件模板
#   "simt_only"            → 纯 SIMT
#   "simt_template"        → (废弃) = simd_simt_template
#   "unstructured_in_simt" → (废弃) = simd_simt_template
```

### 4.2 __post_init__ 自动推导

```python
def __post_init__(self):
    # 1. 废弃 flag 的向后兼容
    if self.force_simt_template:
        object.__setattr__(self, "compile_mode", "simd_simt_template")

    if self.force_simt_only:
        object.__setattr__(self, "compile_mode", "simt_only")

    _validate_compile_mode(self.compile_mode)

    # 2. 根据 compile_mode 设置关联字段
    if self.compile_mode == "simd":
        parallel_mode = "simd"
        shared_mem_dynamic_size = 221184

    elif self.compile_mode == "simd_simt":
        # 仅在 A5 (910_95) 上支持
        if not compile_on_910_95: raise ValueError(...)
        parallel_mode = "mix_simd_simt"
        shared_mem_dynamic_size = 221184

    elif self.compile_mode in ("simd_simt_template", "simt_template",
                                "unstructured_in_simt"):
        # 废弃别名 → 统一为 "simd_simt_template"
        if not compile_on_910_95: raise ValueError(...)
        parallel_mode = "simd"       # ← 注意！parallel_mode 是 "simd"
        shared_mem_dynamic_size = 221184

    elif self.compile_mode == "simt_only":
        force_simt_only = True
        parallel_mode = "simt"
        shared_mem_dynamic_size = 122880  # ← SIMT 专用 shared mem
```

### 4.3 关键差异：parallel_mode 的值

| compile_mode | parallel_mode | shared_mem_dynamic_size |
|-------------|---------------|------------------------|
| `simd` | `"simd"` | 221184 |
| `simd_simt` | `"mix_simd_simt"` | 221184 |
| `simd_simt_template` | `"simd"` | 221184 |
| `simt_only` | `"simt"` | 122880 |

**重要**：`simd_simt` 和 `simd_simt_template` 虽然都是"混合编译"，但 `parallel_mode` 不同：
- `simd_simt` → `parallel_mode = "mix_simd_simt"` → runtime 知道这是混合模式，需要启用 SIMT
- `simd_simt_template` → `parallel_mode = "simd"` → SIMT 模板被内联进 SIMD 代码，runtime 不感知 SIMT

这也是为什么 `simd_simt` 需要传 `--num-warps` 和 `--threads-per-warp` 给 BiShengIR——runtime 需要知道 warp 配置来启动 SIMT 部分。

---

## 5. compile_mode 如何影响 C++ 编译 Pass

compile_mode 通过 Python 侧传给三个关键 C++ pass：

```python
# compiler.py: ttir_to_linalg()

compile_mode = _validate_compile_mode(metadata.get("compile_mode", "simd"))
if force_simt_template:
    compile_mode = "simd_simt_template"  # 旧 flag 覆盖

# 传给各 pass
ascend.passes.ttir.add_discrete_mask_access_conversion(
    pm, ..., compile_mode)       # pass 1

ascend.passes.ttir.add_triton_to_unstructure(
    pm, ..., compile_mode)       # pass 2

ascend.passes.ttir.add_triton_to_linalg(
    pm, ..., compile_mode)       # pass 3
```

C++ 侧通过 `parseCompileMode` 将字符串转为 enum：

```cpp
// Utils.h
enum class CompileMode {
  Simd,             // "simd"
  SimdSimt,         // "simd_simt"
  SimdSimtTemplate, // "simd_simt_template"
  SimtOnly,         // "simt_only"
};

inline CompileMode parseCompileMode(llvm::StringRef mode) {
  return llvm::StringSwitch<CompileMode>(mode)
      .Case("simd", CompileMode::Simd)
      .Case("simd_simt", CompileMode::SimdSimt)
      .Case("simd_simt_template", CompileMode::SimdSimtTemplate)
      .Case("simt_template", CompileMode::SimdSimtTemplate)        // 废弃别名
      .Case("unstructured_in_simt", CompileMode::SimdSimtTemplate) // 废弃别名
      .Case("simt_only", CompileMode::SimtOnly)
      .Default(CompileMode::Simd);
}
```

另外还有一个 `resolveCompileMode` 用于同时处理旧 `forceSimtTemplate` flag：

```cpp
inline CompileMode resolveCompileMode(llvm::StringRef mode,
                                      bool forceSimtTemplate) {
  return forceSimtTemplate ? CompileMode::SimdSimtTemplate
                           : parseCompileMode(mode);
}
```

还有一个便利函数：
```cpp
inline bool isMixCompileMode(CompileMode mode) {
  return mode == CompileMode::SimdSimt || mode == CompileMode::SimdSimtTemplate;
}
```

---

## 6. simd_simt vs simd_simt_template：关键分岔

这是整个设计中最核心的分岔。两者的共同点是：都会把**非结构化内存访问**（unstructured load/store）转换为 `ascend.unstructured_load` / `ascend.unstructured_store` 中间 op。区别在于这个中间 op 之后**如何被 lower**。

### 6.1 TritonToUnstructure 阶段（两者共同路径）

```cpp
// UnstructureConversionPass.cpp

// 两个模式都走这个分支，产生同样的 UnstructuredLoadOp/UnstructuredStoreOp
bool useUnstructuredOp =
    compileOn91095Flag &&
    ((compileModeFlag == CompileMode::SimdSimt &&
      (ptrOffsetInfo.hasUnstructuredDim() || mixCompileDiscreteMask)) ||
     (compileModeFlag == CompileMode::SimdSimtTemplate &&
      simtTemplateLoadStoreFastPathEnabled &&
      rankWithinSimtTemplateLimit));

if (useUnstructuredOp) {
    tryRewriteUnstructuredLoadStoreFastPath(...);
    // → 创建 ascend.unstructured_load / ascend.unstructured_store
}
```

**差异点**：
- `SimdSimt`：只要访问有 unstructured 维度或带 `MixCompileDiscreteMask` 标记，就生成 unstructured op。**没有 rank ≤ 5 的限制，没有 size < 64 的限制。**
- `SimdSimtTemplate`：有 template-specific 的限制（rank ≤ 5、size < 64 或带 `MixCompileDiscreteMask` 标记）。

### 6.2 DiscreteMaskAccessConversion 阶段

```cpp
if (compileModeFlag == CompileMode::SimdSimt) {
    // simd_simt 模式：只打标记，不转换 mask
    // 离散 mask 的 load/store 保留原始 mask，留给后续 hfusion 路径处理
    op->setAttr("MixCompileDiscreteMask", rewriter.getUnitAttr());
    return success();  // ← 不展开 mask
}

if (compileModeFlag == CompileMode::SimdSimtTemplate) {
    // simd_simt_template 模式：打标记，也不展开
    // 因为 SIMT 模板需要原始 mask 做 lane predication
    op->setAttr("MixCompileDiscreteMask", rewriter.getUnitAttr());
    return failure();  // ← 让其他 pattern 继续处理
}
```

**差异点**：
- `SimdSimt`：标记后 `return success()`，阻止其他 pattern 修改此 op
- `SimdSimtTemplate`：标记后 `return failure()`，允许其他 pattern（如 discrete mask 展开）继续处理

### 6.3 TritonToLinalg 阶段（关键分岔）

```cpp
// TritonToLinalgPass.cpp

// 两个模式都注册了相同的 converter
patterns.add<TTOpConverters::UnstructuredLoadConverter>();
patterns.add<TTOpConverters::UnstructuredStoreConverter>();
```

`UnstructuredLoadConverter` 内部根据 `compileModeFlag` 做不同 lowering：

- **`CompileMode::SimdSimt`**：lower 到 `hfusion.gather_load` → 硬件 gather 指令。BishengIR 编译器在 `--enable-simd-simt-mix-compile` 下处理。
- **`CompileMode::SimdSimtTemplate`**：lower 到 SIMT 模板函数调用 `triton_indirect_load()` / `triton_indirect_store()` → 软件实现的间接访问模板。

### 6.4 StridedLoadStoreRewrite 阶段

```cpp
// 只在 SimdSimtTemplate 模式下运行
if (!(compileOn91095Flag &&
      compileModeFlag == CompileMode::SimdSimtTemplate)) {
    return success();  // SimdSimt 模式直接跳过
}

// 对非 permuted、有静态 last-axis stride > 1 的访问：
//   tt.load  → ascend.stride_load  → triton_stride_load 模板
//   tt.store → ascend.stride_store → triton_stride_store 模板
```

**差异点**：`SimdSimt` 不需要 strided 模板，因为硬件 gather 指令可以直接处理 strided 访问。

### 6.5 simt_template 模式的额外限制

在 UnstructureConversionPass 中，`simd_simt_template` 有额外的 fallback 条件：

```cpp
// SIMT 模板有 rank ≤ 5 和 size < 64 的限制
bool simtTemplateLoadStoreFastPathEnabled =
    compileOn91095Flag && forceSimtTemplateFlag &&
    ((!ptrOffsetInfo.isStructured() && sizeInByte < 64) ||
     mixCompileDiscreteMask);
```

如果超过限制，op 不会走 unstructured fast path，而是 fallback 到传统的 scalar loop 路径。

---

## 7. UnstructuredLoadOp/UnstructuredStoreOp：统一中间表示

这是该分支新引入的 TTIR op，取代了旧的 `IndirectLoadOp`/`IndirectStoreOp`。

### 7.1 定义

```
ascend.unstructured_load(base, indices, unstructured_dims, mask?, other?)
    → result_tensor

ascend.unstructured_store(base, indices, value, unstructured_dims, mask?)
```

**关键设计**：通过 `unstructured_dims` 属性标记哪些维度是非结构化的（即偏移量不是简单的 strided access），而 `indices` 是展开后的线性偏移量。TritonToLinalg 根据这个属性决定 lower 策略。

### 7.2 为什么统一

旧设计有 `IndirectLoadOp`（走模板）和硬件 gather（走 hfusion），两个不同的 op 导致代码分叉。新设计使用同一个 `UnstructuredLoadOp`，在不同 compile_mode 下 lower 到不同目标：

```
ascend.unstructured_load
    │
    ├─ CompileMode::SimdSimt
    │    → hfusion.gather_load（硬件指令，BishengIR 处理）
    │
    └─ CompileMode::SimdSimtTemplate
         → triton_indirect_load()（软件模板，Linalg 阶段 lower）
```

### 7.3 StrideLoadOp/StrideStoreOp：SIMT 模板专用

对于**有最后轴 stride > 1** 的 strided 访问，SIMT 模板路径有专门的优化：

```
tt.load (with last-axis stride > 1, non-permuted)
    → ascend.stride_load(base, offset, other, strides[], numels[])
        → triton_stride_load() 模板函数（per-dimension stride + numel）
```

`StrideLoadOp` 只在 `SimdSimtTemplate` 模式下产生。`SimdSimt` 模式走 unstructured 统一路径。

---

## 8. scope.vec_mode：代码块级别的 SIMD/SIMT 选择

### 8.1 Python 前端

```python
# scope.py
class scope:
    def __init__(self, core_mode: str = None, vec_mode: str = None):
        """
        core_mode: "cube" | "vector"  (可选)
        vec_mode:  "simd" | "simt"    (可选)
        """
        # 验证:
        # - vec_mode 不能与 core_mode="cube" 同时使用
        # - 至少需要一个参数
```

**使用示例**：
```python
# 对整个 kernel 用 SIMD
@triton.autotune(configs=..., compile_mode="simd_simt")
def my_kernel(...):
    # 结构化部分用 SIMD（默认）
    x = tl.load(ptr + offsets)

    # 非结构化部分用 SIMT（显式标注）
    with al.scope(vec_mode="simt"):
        y = tl.load(base_ptr + indirect_indices)  # gather
```

### 8.2 TTIR 层

`al.scope(vec_mode="simt")` 会生成 TTIR：

```mlir
%result = "scope.scope"() ({
    %val = "tt.load"(...) : ...
    "scope.return"(%val) : ...
}) {vec_mode = "simt"} : () -> tensor<...>
```

### 8.3 C++ 层检查和路由

```cpp
// Utils.h
inline bool hasScopeVecMode(Operation *op, llvm::StringRef mode) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto vecModeAttr = parent->getAttrOfType<StringAttr>("vec_mode")) {
      if (vecModeAttr.getValue() == mode)
        return true;
    }
  }
  return false;
}
```

这个函数被 TrionToLinalgPass 用来在 lowering 时判断某个 op 是否在 SIMT/SIMD scope 内，从而选择正确的 lower 策略。

**与 simt_costmodel 分支的对比**：simt_costmodel 分支的 `SimtSelection.h` 中也有 `hasEnclosingVectorMode`，功能完全相同。这说明 `vec_mode` scope 是横跨两个分支的通用机制——在 `simd-simt-compile-mode` 分支中由用户显式标注，在 `simt_costmodel` 分支中由 cost model 自动 materialize。

---

## 9. BishengIR 侧：--enable-simd-simt-mix-compile

当 `compile_mode == "simd_simt"` 时，Python 编译流程在 `linalg_to_bin` 阶段向 BiShengIR 编译器传递额外参数：

```python
# compiler.py: linalg_to_bin_enable_npu_compile_910_95()

if compile_mode == "simd_simt":
    _compile_option_list += ["--enable-simd-simt-mix-compile"]
    num_warps = metadata.get("num_warps", opt.num_warps)
    _compile_option_list += [f"--num-warps={num_warps}"]
    warp_size = metadata.get("warp_size", opt.warp_size)
    _compile_option_list += [f"--threads-per-warp={warp_size}"]
```

- `--enable-simd-simt-mix-compile`：告诉 BiShengIR 编译器这是混合 SIMD/SIMT 编译，`hfusion.gather_load`/`hfusion.scatter_store` 需要被正确处理
- `--num-warps`：SIMT warp 数量
- `--threads-per-warp`：每个 warp 的线程数（通常是 32）

对于 `simd_simt_template` 模式，**不需要**这些 flag——因为 SIMT 模板函数已经在 TTIR→Linalg 阶段被 lower 成普通函数调用，BiShengIR 不需要知道 SIMT 的存在。

---

## 10. parallel_mode 和 runtime 行为

`parallel_mode` 最终被写入 linalg IR 的元数据中：

```python
# compiler.py: _parse_linalg_metadata()
metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
```

这个值控制 runtime 的行为：

| parallel_mode | SIMT 启用 | shared mem | 说明 |
|--------------|----------|------------|------|
| `"simd"` | 否 | 221184 | 纯向量核心执行 |
| `"mix_simd_simt"` | 是 | 221184 | 混合执行，runtime 需要处理 SIMT 部分 |
| `"simt"` | 是 | 122880 | 纯 SIMT 执行，更小的 shared mem |

`mix_simd_simt` 模式与 `simd` 的 shared mem 大小相同（都是 221184），因为 SIMT 部分使用的是 VF（Vector Function）的 local memory，复用 SIMD UB 空间。而 `simt_only` 模式使用更小的 122880，因为纯 SIMT 对 UB 的使用方式不同。

---

## 11. 废弃的旧选项和迁移路径

| 旧选项 | 状态 | 迁移 |
|--------|------|------|
| `force_simt_template=True` | 废弃 | 改用 `compile_mode="simd_simt_template"` |
| `force_simt_only=True` | 废弃 | 改用 `compile_mode="simt_only"` |
| `compile_mode="unstructured_in_simt"` | 废弃别名 | 自动转为 `"simd_simt_template"` |
| `compile_mode="simt_template"` | 废弃别名 | 自动转为 `"simd_simt_template"` |
| `enable_dynamic_cv_flow_opt` | 移除 | 功能被移除 |
| `set_enable_dynamic_cv_flow_optimization` | 移除 | 相应的 pybind 函数被删除 |
| `set_buffer_count` 单例模式 | 改为 instance | 现在需要传 `mod` 参数 |

向后兼容处理在 `NPUOptions.__post_init__()` 中：
```python
if self.force_simt_template:
    warnings.warn("force_simt_template is deprecated, ...")
    object.__setattr__(self, "compile_mode", "simd_simt_template")
```

---

## 12. 总结：四种模式的对比表

| 维度 | simd | simd_simt | simd_simt_template | simt_only |
|------|------|-----------|-------------------|-----------|
| **用途** | 纯向量核心 | 混合，硬件 gather | 混合，软件模板 | 纯 SIMT |
| **平台** | 全部 | 仅 A5 (910_95) | 仅 A5 (910_95) | 全部 |
| **parallel_mode** | `"simd"` | `"mix_simd_simt"` | `"simd"` | `"simt"` |
| **shared_mem** | 221184 | 221184 | 221184 | 122880 |
| **非结构化访问** | scalar loop | `hfusion.gather_load` | `triton_indirect_load` 模板 | SIMT 原生 |
| **Strided 访问** | SIMD strided DMA | hardware gather | `triton_stride_load` 模板 | SIMT 原生 |
| **离散 mask** | 展开为 select | 保留原始 mask | 保留原始 mask（模板用） | SIMT 原生 |
| **BishengIR flag** | 无 | `--enable-simd-simt-mix-compile` | 无 | 无 |
| **模板 rank 限制** | N/A | 无限制 | rank ≤ 5 | N/A |
| **模板 size 限制** | N/A | 无限制 | size < 64 (或带 MixCompileDiscreteMask) | N/A |
| **UnstructuredLoadOp** | N/A | 是 → hfusion | 是 → 模板 ABI | N/A |
| **StrideLoadOp** | N/A | 否 | 是 (非 permuted, stride>1) | N/A |
| **vec_mode scope** | N/A | 支持 | 支持 | N/A |

### 选择建议

```
你的 kernel 有非结构化内存访问吗？
    │
    ├─ 没有 → compile_mode="simd"（默认）
    │
    └─ 有 →
        │
        ├─ 想用硬件 gather/scatter（更快，但依赖 BiShengIR 支持）
        │   → compile_mode="simd_simt"
        │
        └─ 想用软件 SIMT 模板（更灵活，rank≤5/size<64 限制）
            → compile_mode="simd_simt_template"

整个 kernel 都是 SIMT 风格（如从 CUDA 直接翻译过来的）？
    → compile_mode="simt_only"
```

### 与 simt_costmodel 分支的关系

在 `simt_costmodel` 分支中，`compile_mode` 增加了一个新值 `"simd_simt"`（注意与这边的 `"simd_simt"` 是不同的东西！）：

| 分支 | simd_simt 的含义 |
|------|-----------------|
| `simd-simt-compile-mode` (本分支) | 混合编译 + 硬件 gather/scatter |
| `simt_costmodel` | 由 C++ cost model 自动决策三种路由（AllSIMD/AllSIMTOnly/MixedSIMDSIMT） |

两边都使用 `scope.scope{vec_mode="simt"}` 作为 SIMT 操作的标记机制。`simd-simt-compile-mode` 分支中由用户显式写 `al.scope(vec_mode="simt")`，`simt_costmodel` 分支中由 SelectSimdSimtCostModel pass 自动 materialize。

---

> **文档结束**。配合 `autoscope_simit_costmodel_analysis.md` (v6) 和 `autoscope_simit_costmodel_analysis_v10.md` (v10) 阅读可获得 SIMD/SIMT 编译路线的完整图景。
