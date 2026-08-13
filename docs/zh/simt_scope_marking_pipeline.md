# SIMT 标记全链路：三条路线的 op 级详解

> 回答：`compile_mode` 不同取值下，哪些 op 会在哪一层（TTIR pass / TritonToLinalg / npuir 哪个 pass）被识别为 SIMT、怎么处理、结果长什么样。
> 三条路线**最终殊途同归**：npuir 里都是 `hivm.hir.indirect_load`（或 SIMT VF func）+ 父函数 `vf_mode=MIX`。

---

## 〇、路线总览：先分清是哪条

| 路线 | 触发条件 | 前端（triton-ascend）标记 | npuir 消费方式 |
|------|---------|--------------------------|---------------|
| **A. costmodel** | `compile_mode='simd_simt'` 且 `auto_simt_scope_mode != "off"` | `MaterializeSimtScopes` 包 `scope.scope {vec_mode="simt"}` | 读 scope 属性 → 补 attrs → outline 成 SIMT VF |
| **B. template**（kx 的 `simd_simt`） | `force_simt_template=True`（kx 中 `simd_simt` 自动置位） | `tt.load` → `ascend.indirect_load` → `call @triton_indirect_load` | 函数调用名匹配 → 逐 op 翻译 + 盖章 vf_mode=SIMT |
| **C. 非 template**（kanuak 的 `simd_simt`） | `compile_mode='simd_simt'`，force_simt_template=False | `tt.load` → `hfusion.gather_load`（**无任何 SIMT 标记**） | AutoScope 靠 op 类型模式匹配发现 → 自建 scope |

**怎么快速判断产物走的是哪条路线**：看 `.ttadapter` 文件——
- 有 `scope.scope {vec_mode="simt"}` → A
- 有 `func.call @triton_indirect_load_N` → B
- 有 `hfusion.gather_load` 且无 scope → C

---

## 一、路线 A：costmodel 路径

### 1.1 前端：anchor 识别（在 `ttir_to_linalg` 之前，TTIR 上）

**入口**：`compiler.py:168-207` `_run_cpp_simd_simt_costmodel` → C++ pass `SelectSimdSimtCostModel` → `SimtAnchorAnalysis.cpp::analyzeAnchor` (L312-457)

**6 种 anchor 的完整识别表**：

| # | TTIR op | Anchor kind | 检测逻辑（伪代码） | 文件:行 |
|---|---------|-------------|-------------------|---------|
| 1 | `tt.gather` | DirectGather | op 名匹配，无条件 | `SimtAnchorAnalysis.cpp:320` |
| 2 | `tt.histogram` | Histogram | 输入静态 shape + rank1 元素 i8/i16/i32/i64 + 结果 rank1 i32 bins | `:324, 349-355` |
| 3 | `tt.scan`（1D cumsum） | PlainOneDimensionalCumsum | body 恰好一个真实 op 且为 addf/addi + 非 axis 维全 1 + 元素类型 ∈ i8/i16/i32/f16/bf16/f32 | `:366`；`analyzePlainOneDimensionalCumsum` `:45-87` |
| 4 | `tt.atomic_rmw` / `tt.atomic_cas` | TensorAtomic | 结果静态 shape + 类型支持表 + offset 是 i32/i64 静态 tensor | `:386`；`getAtomicOperation` `:175-219` |
| 5 | `scf.for`（三角求解） | TriangularSolveLoop | body 含 rank1[16] load + axis=0 reduce + select + iter_arg 有 16x16 静态 tensor 状态 | `:439`；`isTriangularSolveLoop` `:239-302` |
| 6 | `tt.load` / `tt.store` | LoadedIndexDependentMemory | 首 operand 是 rank≤5 静态 shape tensor 指针 + **`pointerDependsOnLoadedIndex`**（BFS 反向 SSA 切片，遇到 tt.load/tt.gather 或 scf.for 迭代参数即真） | `:445, 530-536`；`pointerDependsOnLoadedIndex` `:96-129` |

**anchor 6 是重点**——它覆盖了 ROPE kernel 的 8 个间接 load 和 causal_conv1d 的 2 个间接 load：

```cpp
// pointerDependsOnLoadedIndex 伪代码
worklist = {load 的地址 operand}
while (worklist 非空) {
    value = worklist.pop()
    if (value 是 scf.for 的迭代参数) return true
    producer = value.getDefiningOp()
    if (producer 是 tt.load 或 tt.gather) return true   // ← 地址依赖另一个 load
    worklist.append(producer 的全部 operands)            // 继续回溯
}
return false
```

### 1.2 前端：包装（`add_materialize_simt_scopes` pass）

`MaterializeSimtScopes.cpp`：

- **单 op 包装** `wrapAnchorOperation` (:51-79)：anchor 1/2/3/4/6 用
  ```cpp
  OperationState scopeState(loc, "scope.scope");
  scopeState.addAttribute(kVectorModeAttr, "simt");  // ← vec_mode="simt"
  scopeState.addTypes(op 的全部 result 类型);
  // body 里 move 进 anchor op + scope.return 透出结果
  ```
- **多 op 区间包装** `wrapAnchorRange` (:139-203)：anchor 5（三角求解）用 `collectTriangularSolveRange` (:85-135) 收集连续区间（三角循环 + 尾部 arith.uitofp/addf/select）

**结果 IR 示例**（间接 load，路线 A）：
```mlir
// before (TTIR)
%164 = tt.load %163, %162, %cst_2 : tensor<16x32x!tt.ptr<f16>>

// after (TTIR + materialize_simt_scopes)
%164 = scope.scope {vec_mode = "simt"} {
  %inner = tt.load %163, %162, %cst_2 : tensor<16x32x!tt.ptr<f16>>
  scope.return %inner : tensor<16x32xf16>
}
```

### 1.3 中间层：TritonToLinalg 对 scope 的处理

`TritonToLinalgPass.cpp::isSIMTOp` (L170-174)：
```cpp
if (op->getName().getStringRef() == "scope.scope") {
    auto mode = op->getAttrOfType<StringAttr>(kVectorModeAttr);  // vec_mode
    if (mode && mode.getValue() == "simt") return true;           // ← 计为 SIMT op
}
```
scope 是 BiShengIR 原生构造，**原样透传**到 adapter 输出。父函数照常盖章 `parallel_mode="mix_simd_simt"`。

### 1.4 npuir：scope 的 custody chain（pass 顺序）

```
adapter 输入: scope.scope {vec_mode="simt"} + func parallel_mode="mix_simd_simt"
    │
    ▼ SplitMixKernel        SplitMixKernel.cpp:549-556   simt scope 设 tcore_type=VECTOR
    ▼ InlineScope           InlineScope.cpp:51-56,118-127 isSimtScope → setNoInline(true)，不内联
    ▼ AutoScope             AutoScope.cpp:246-258        补 outline + tcore_type=AIV + vf_mode=SIMT
    ▼ LegalizeBoolForSimtVF LegalizeBoolForSimtVF.cpp:169 scope 内 i1 布尔 i8 化
    ▼ InsertMemSemanticForSimtVF  :157                    边界插 acquire/release 语义
    ▼ OutlineScope          OutlineScope.cpp:271-284     scope → 独立 func，attrs 整体拷贝
    ▼ InsertAllocBasePlaceholder :49                     为 SIMT VF 内 alloc 插 base 占位
    ▼ InferSimtVFMemEffect      :107-149                 推导 VF 参数读写效应
    ▼ InferSimtVFMemScopeHint   :69-128                  逐实参记 mem-scope hint
    ▼ SplitSimtModule       SplitSimtModule.cpp:44-126   vf_mode=SIMT 的 func 拆进 SIMT 子模块
```

**核心逻辑**（AutoScope.cpp:246-258，对已存在的 simt scope）：
```cpp
if (scope 有 vector_type == "simt") {
    scope 补三个 attr:
      outline          (UnitAttr)         // ← 触发后续 OutlineScope
      tcore_type = AIV (TFuncCoreTypeAttr)
      vf_mode = SIMT   (VFModeAttr)       // ← 后续 isSIMTVF 识别依据
}
```

**OutlineScope**（OutlineScope.cpp:172-173）：scope 的全部 attrs（含 vf_mode=SIMT）原样拷到生成的 `@parent_scope` func 上——这就是 **"outlined SIMT VF 带 vf_mode=SIMT"** 的来源。

**SplitSimtModule**（SplitSimtModule.cpp:91-110）：
```cpp
for (funcOp : simtVFs) {            // isSIMTVF = func 带 vf_mode==SIMT
    newMod = 新建 ModuleOp;
    newMod.clone(funcOp);
    newMod.setAttr(hacc::SIMTModuleAttr, UnitAttr);   // ← 标为 SIMT 子模块
    // 主模块中的原 func：eraseBody + private + 尾部补 3 个 i32 grid size 参数
}
```

### 1.5 ⚠️ 已知断点：属性名不一致

**本地 checkout 前端发的是 `vec_mode`（MaterializeSimtScopes.cpp:55），但新版 AscendNPU-IR clone 全部消费者读的是 `vector_type`**（AutoScope.cpp:246、InlineScope.cpp:52、SplitMixKernel.cpp:549、NormalizeTypeConversion.cpp:1393、Utility.cpp:58）。clone 全库 grep `vec_mode` 零命中。这是前后端版本迁移中的不一致点，需要对齐（前端改发 `vector_type` 或 clone 兼容两者）。

---

## 二、路线 B：template 路径（kx 的 `simd_simt`）

### 2.1 前端：TritonToUnstructure 改写（`add_triton_to_unstructure` pass）

`UnstructureConversionPass.cpp` `matchAndRewrite` (L434-756)：

**SIMT 门**（L524-537，对 load/store）：
```cpp
simtTemplateLoadStoreFastPathEnabled =
    compileOn91095
    && shouldUseSimtTemplate(op, forceSimtTemplateFlag)
    && ((非结构化 && sizeInByte < 64) || route_discrete_mask_to_simt)
    && rank ≤ 5;
```

**op 映射表**：

| TTIR op | 触发 | 重写函数 | 结果 |
|---------|------|---------|------|
| `tt.load` | SIMT 门 + offset 已解析 + 非 scalarLike | `tryRewriteIndirectFastPath<LoadOp>` (:153-188) | `ascend.indirect_load` |
| `tt.store` | 同上 | 同上 (:189-205) | `ascend.indirect_store` |
| `tt.atomic_rmw` | 同上但**无 size<64 限制** + `canUseIndirectAtomicFastPath`（类型表） | `tryRewriteIndirectFastPath<AtomicRMWOp>` (:206-223) | `hivm.hir.custom "__builtin_indirect_atomic"` |
| `tt.atomic_cas` | 同上 | `tryConvertAtomicCasToIndirectCustom` (:224-242) | 同上 |
| 不满足门（size≥64 / rank>5 / 结构化） | — | 标量回退 (:556-755)：`scf.for` + `tensor.extract/extract_slice` + `tensor.insert/insert_slice` | scf.for + InsertSlice |

**离散 mask**（`DiscreteMaskAccessConversionPass.cpp`）：对 `tt.load`(:348-411)/`tt.store`(:275-346)/`tt.atomic_rmw`(:413-465)，若 `compileOn91095 && shouldUseSimtTemplate && rank≤5` → 只打 `route_discrete_mask_to_simt` 属性，把决策留给 unstructure 的 fast path；否则改写为 safe-load + select + `hivm.sync_block_lock/unlock`。

### 2.2 前端：TritonToLinalg outline（`add_triton_to_linalg` pass）

**Converter 全表**（`TritonOpConverter.h` 声明 / `.cpp` 实现）：

| 输入 op | Converter（h 行 / cpp 行） | 生成函数（private + call） |
|---------|--------------------------|--------------------------|
| `ascend.indirect_load` | IndirectLoadConverter h:682 / cpp:2703-2745 | `@triton_indirect_load` |
| `ascend.indirect_store` | IndirectStoreConverter h:716 / cpp:2837 | `@triton_indirect_store` |
| `tt.gather` | GatherConverter h:492 / cpp:1843-1877 | `@triton_gather` |
| `tt.histogram` | HistogramConverter h:736 / cpp:3094-3147 | 非 func：`hivm.custom "__builtin_histogram"` + vf_mode=SIMT + tcore_type=VECTOR |
| `tt.scan`（1D cumsum） | ScanConverter h:429 / cpp:1196-1240 | `@triton_cumsum`（cumprod/cummax/cummin 对应） |
| `ascend.stride_load` | StrideLoadConverter h:692 / cpp:2747 | `@triton_stride_load` |
| `ascend.stride_store` | StrideStoreConverter h:704 / cpp:2794 | `@triton_stride_store` |
| `ascend.index_put` | IndexPutConverter h:652 / cpp:2556 | `@triton_index_put` |
| `ascend.gather_out_to_ub` | GatherOutToUbConverter h:662 / cpp:2602 | `@triton__gather_out_to_ub` |
| `ascend.scatter_ub_to_out` | ScatterUbToOutConverter h:672 / cpp:2655 | `@triton_scatter_ub_to_out` |

**IndirectLoadConverter 伪代码**（cpp:2703-2745）：
```cpp
// 在 module 末尾创建无 body 的私有声明
func.func private @triton_indirect_load(
    memref<?xf16>,                  // base buffer
    tensor<16x32xi64>,              // indices
    tensor<16x32xi1>,               // mask（可选）
    tensor<16x32xf16>               // other（可选）
) -> tensor<16x32xf16> attributes {isVolatile = false}

// 原处替换
%r = func.call @triton_indirect_load(%base, %indices, %mask, %other)
```

**函数体从哪来**：由 BiShengIR 的 SIMT template 库在 npuir 阶段提供——adapter 里只是标记声明。

### 2.3 前端：函数盖章

`TritonToLinalgPass.cpp:490-505`：
```cpp
mix_mode      = 有 tt.dot ? "mix" : "aiv";
parallel_mode = existSIMTOp ? "mix_simd_simt" : "simd";
// existSIMTOp 遍历 isSIMTOp（L164-216 完整清单）：
//   scope.scope(vec_mode=simt)、hivm.custom(VECTOR+SIMT)、tt.gather、tt.histogram、
//   simt 1D cumsum、IndirectLoad/IndirectStore/IndexPut/GatherOutToUb/ScatterUbToOut/StrideLoad/StrideStore
```

### 2.4 npuir：消费链

```
adapter 输入: func.call @triton_indirect_load_N + func parallel_mode="mix_simd_simt"
    │
    ▼ AdaptTritonKernel      AdaptTritonKernel.cpp:1166-1226
    │   TritonIndirectLoadToHFusionIndirectLoadPattern (:365-477):
    │     call @triton_indirect_load → hfusion.indirect_load
    │     （mask i1→i8 extui、other 缺省 zero splat、记 DirectlyUsedGMArgIdxList）
    ▼ HFusionToHIVM          HFusionToHIVM.cpp:1576-1604
    │     hfusion.indirect_load → hivm::IndirectLoadOp（打印 hivm.hir.indirect_load）
    │     + setAttr(VFModeAttr, VFMode::SIMT)     ← 逐 op 盖章
    ▼ InferVFMode            InferVFMode.cpp:83-157, 201-213
    │     仅 hacc.entry：op 默认 SIMD、显式 SIMT、嵌套冲突→MIX
    │     → 父函数 vf_mode = MIX
    ▼ SplitSimtModule        同路线 A：vf_mode=SIMT 的 func 拆 SIMT 子模块
```

**AdaptTritonKernel 的全部 pattern**（`AdaptTritonKernel.cpp`，按 func 名前缀匹配）：

| Pattern | line | 目标 op |
|---------|------|---------|
| TritonGatherToHFusionGatherPattern | 195 | `hfusion.gather` |
| TritonCumToHFusionCumPattern | 268 | `hfusion.cumsum/cumprod/cummax/cummin` |
| **TritonIndirectLoadToHFusionIndirectLoadPattern** | **365-477** | `hfusion.indirect_load` |
| TritonStrideLoadToHFusionStrideLoadPattern | 482 | `hfusion.stride_load` |
| TritonStrideStoreToHFusionStrideStorePattern | 549 | `hfusion.stride_store` |
| TritonGatherTToHFusionGatherTPattern | 617 | `hfusion.gather_t` |
| TritonIndexPutToHFusionIndexPutPattern | 723 | `hfusion.index_put` |
| TritonScatterTOpToHFusionScatterTOpPattern | 882 | `hfusion.scatter_t` |
| TritonEmbeddingGatherToHFusionEmbeddingGatherPattern | 988 | `hfusion.embedding_gather` |
| TritonIndirectStoreToHFusionIndirectStorePattern | 1064 | `hfusion.indirect_store` |
| （还有 print/assert/flip/sort/bind_sub_block） | 87-958 | 对应 hfusion op |

**HFusionToHIVM 的 vf_mode=SIMT 盖章点**（HFusionToHIVM.cpp）：indirect_load:1597、stride_load:1620、stride_store:1642、indirect_store:1670、gather:872、gather_t:1705、index_put:1738、scatter_t:1771、embedding_gather:1563。

**结果 IR 示例**（causal_conv1d npuir 实测）：
```mlir
// npuir 中（内联在 MIX 父函数 scf.if 分支内）
hivm.hir.indirect_load ins(%arg2 : memref<?xf32>,        // base
                          %81 : memref<2x256xi64>,      // indices
                          %82 : memref<2x256xi8>,       // mask (i8)
                          %8 : memref<2x256xf32>)       // other
                      outs(%83 : memref<2x256xf32>)     // ← 该 op 带 vf_mode=SIMT
```

### 2.5 ⚠️ 已知断点

模板生成的私有 func 需要**函数级** `vf_mode=SIMT` 才能被 SplitSimtModule 拆出。本地旧 submodule 只在 `hivm.custom` op 上设了该 attr（TritonOpConverter.cpp:3136），未设函数级——因此本 checkout 的模板函数不会被拆分（SIMT 身份以逐 op 标记存在，如 causal_conv1d npuir 所示）。

---

## 三、路线 C：非 template（kanuak 的 `simd_simt`）

### 3.1 前端：TritonToUnstructure 改写

kanuak 的 `Utils.h:67-77`：`resolveCompileMode(mode, force)` = force ? `SimdSimtTemplate` : `parseCompileMode(mode)`（`simd_simt` → `SimdSimt`）。

| TTIR op | 触发 | 重写函数 | 结果 |
|---------|------|---------|------|
| `tt.load` | `useUnstructuredOp`（L575-592）= compileOn91095 && mode==SimdSimt && (有非结构化维 || mixCompileDiscreteMask || route_discrete_mask_to_simt) | `tryRewriteUnstructuredLoadStoreFastPath` (:169-233) | **`ascend.unstructured_load`**（带 `unstructured_dims` dense i64 数组） |
| `tt.store` | 同上 | 同上 | **`ascend.unstructured_store`** |
| `tt.atomic_rmw/cas` | **仅 template 模式**才走 custom（`simtTemplateAtomicFastPathEnabled` L566-568 要求 force） | — | 非 template 下回退 scf.for |

### 3.2 前端：TritonToLinalg 转换

kanuak `TritonOpConverter.cpp:3189 附近`：

```cpp
// UnstructuredLoadConverter::matchAndRewrite
if (compileMode == SimdSimt) {
    // 直接生成结构化 op，无 func ABI
    create<hfusion::GatherLoadOp>(base, indices, burstLength, mask, other, ...);
} else {  // SimdSimtTemplate
    // 同路线 B：@triton_indirect_load func + call
}
// UnstructuredStoreConverter（:3341 附近）：SimdSimt → hfusion::ScatterStoreOp
```

### 3.3 npuir：AutoScope 模式匹配发现（关键！）

```
adapter 输入: hfusion.gather_load（无任何 SIMT 标记）
    │
    ▼ HFusionToHIVM     HFusionToHIVM.cpp:1429-1483
    │     hfusion.gather_load → hivm.gather_load    （只搬 attrs，不设 vf_mode）
    ▼ AutoScope         AutoScope.cpp:64-66, 103-243
    │     isSimtSeedOp = isa<hivm::GatherLoadOp, hivm::ScatterStoreOp>
    │     ① 对 seed 的 indices/base/dst/mask/other 做保守 SSA 反向切片
    │        （memref 边界即停:112、动态 size insert_slice 跳过:126-132、遇其他 seed 停:117）
    │     ② 新建 scope.scope + outline + tcore_type=AIV + vf_mode=SIMT（createScope :162-187）
    │     ③ 克隆子图 + scope.return + replaceOp 原 seed
    ▼ LegalizeBoolForSimtVF → ... → SplitSimtModule    （同路线 A 全链）
```

**AutoScope 是这条路线"标记在 npuir"说法的来源**——SIMT 判定（哪些 op 是 SIMT）由 op 类型模式匹配在 npuir 内完成。

---

## 四、总对照表

### 4.1 op × 路线 → 标记

| TTIR op | A: costmodel | B: template（kx simd_simt） | C: 非 template（kanuak simd_simt） |
|---|---|---|---|
| `tt.load`（间接/非结构化） | anchor 6 → `scope.scope{vec_mode=simt}` | `ascend.indirect_load` → `@triton_indirect_load` | `ascend.unstructured_load` → `hfusion.gather_load`（无标记） |
| `tt.store`（同上） | anchor 6 → scope | `ascend.indirect_store` → `@triton_indirect_store` | `ascend.unstructured_store` → `hfusion.scatter_store`（无标记） |
| `tt.gather` | anchor 1 → scope | `@triton_gather` | `@triton_gather`（同 B） |
| `tt.histogram` | anchor 2 → scope | `__builtin_histogram` custom + vf_mode=SIMT | 同 B |
| `tt.scan`（1D cumsum） | anchor 3 → scope | `@triton_cumsum` | 同 B |
| `tt.atomic_rmw/cas` | anchor 4 → scope | `__builtin_indirect_atomic` custom | 回退 scf.for 标量循环 |
| `scf.for`（三角求解） | anchor 5 → 区间 scope | 依赖 scope 存续（无独立重写） | 同 B |
| strided `tt.load/store` | 非 anchor | `@triton_stride_load/store` | 同 B |
| 离散 mask load/store/atomic | 非 anchor | `route_discrete_mask_to_simt` attr → fast path | 同 B |
| 结构化 / size≥64 / rank>5 | — | scf.for + InsertSlice 回退 | 同 B |

### 4.2 处理层级 × 路线

| 处理层级 | A: costmodel | B: template | C: 非 template |
|---------|-------------|-------------|----------------|
| make_ttir（Python→TTIR） | 无 | 无 | 无 |
| select_simd_simt_costmodel（TTIR） | ✅ anchor 识别 + 打分 | ✗ | ✗ |
| materialize_simt_scopes（TTIR） | ✅ 包 scope | ✗ | ✗ |
| triton_to_unstructure（TTIR→Linalg） | 透传 | ✅ load/store/atomic 改写 | ✅ load/store 改写（unstructured 版） |
| triton_to_linalg（TTIR→Linalg） | 透传 scope + 盖章 | ✅ outline @triton_* + 盖章 | ✅ 直接 hfusion.gather_load + 盖章 |
| npuir AdaptTritonKernel | ✗ | ✅ call → hfusion op | ✗ |
| npuir HFusionToHIVM | ✗ | ✅ hfusion → hivm + vf_mode=SIMT | ✅ hfusion → hivm（无 vf_mode） |
| npuir AutoScope | ✅ 读 scope 补 attrs | ✗ | ✅ **种子发现 + 自建 scope** |
| npuir OutlineScope/SplitSimtModule | ✅ | ✅ | ✅ |

### 4.3 npuir 消费三种标记的本质

| Marker | 消费方式 | 类比 |
|--------|---------|------|
| A: `scope.scope {vec_mode=simt}` | 原样存活 + 逐级打标 → 1:1 outline | **打标后照抄** |
| B: `call @triton_indirect_load` | 函数名匹配 → 逐级翻译 + 显式盖章 vf_mode=SIMT | **1:1 翻译** |
| C: `hfusion.gather_load` | op 类型模式匹配发现 → 自建 scope 汇入 A 链 | **重新发现** |

---

## 附：关键文件索引

### 前端（triton-ascend）

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `backend/compiler.py` | :168-207, :1224-1226 | costmodel 入口 / simd_simt→force_simt_template 映射 |
| `costmodel/lib/.../SimtAnchorAnalysis.cpp` | :312-457, :96-129 | 6 种 anchor 识别 / pointerDependsOnLoadedIndex |
| `costmodel/lib/.../Transforms/MaterializeSimtScopes.cpp` | :51-79, :139-203 | 单 op / 区间 scope 包装 |
| `lib/TritonToUnstructure/UnstructureConversionPass.cpp` | :153-242, :524-548, :556-755 | indirect fast path / SIMT 门 / 标量回退 |
| `lib/DiscreteMaskAccessConversion/DiscreteMaskAccessConversionPass.cpp` | :275-465 | 离散 mask → route_discrete_mask_to_simt |
| `lib/TritonToLinalg/TritonOpConverter.cpp` | :2703-2745 等 | IndirectLoadConverter 等 10 个 converter |
| `lib/TritonToLinalg/TritonToLinalgPass.cpp` | :164-216, :490-505 | isSIMTOp 清单 / 函数盖章 |

### 后端（AscendNPU-IR）

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `bishengir/lib/Dialect/HFusion/Transforms/AdaptTritonKernel.cpp` | :365-477, :1166-1226 | triton_* call → hfusion op |
| `bishengir/lib/Dialect/HIVM/Conversion/HFusionToHIVM/HFusionToHIVM.cpp` | :1576-1778 | hfusion → hivm + vf_mode=SIMT 盖章点 |
| `bishengir/lib/Dialect/HIVM/Transforms/AutoScope.cpp` | :64-66, :103-243, :246-258 | seed 发现 / 反向切片建 scope / 已有 scope 打标 |
| `bishengir/lib/Dialect/HIVM/Transforms/OutlineScope.cpp` | :172-173, :271-284 | scope → func + attrs 拷贝 |
| `bishengir/lib/Dialect/HIVM/Transforms/InferVFMode.cpp` | :83-157, :201-213 | 父函数 MIX 判定 |
| `bishengir/lib/Dialect/HIVM/Transforms/SplitSimtModule.cpp` | :44-126 | SIMT 模块拆分 |
| `bishengir/lib/Dialect/HIVM/Utils/Utils.cpp` | :1496-1502 | isSIMTVF（func 级 vf_mode==SIMT） |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp` | :459-523 | regbase 主管线（含 mix 门控块 :508-521） |
| `bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td` | :1205-1208 | VFMode 枚举（SIMD=0/SIMT=1/MIX=2） |
