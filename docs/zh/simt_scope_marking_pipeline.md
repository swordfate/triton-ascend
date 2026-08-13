# SIMT 标记全链路：从 compile_mode 到 npuir

> 回答核心问题：`compile_mode='simd_simt'`（不用 costmodel）时，SIMT 的标记到底发生在哪一步？
> 结论一句话：**"哪些 op 是 SIMT" 的决策在 TTIR→adapter 阶段（triton-ascend 的 C++ pass）已经定型；npuir（BiShengIR）只消费这个契约做 lowering 与执行形态强化，不做独立的逐 op SIMT 判定。**

---

## 零、全景图

```
Python kernel + compile_mode='simd_simt'
    │  NPUOptions.__post_init__: force_simt_template=True, parallel_mode="mix_simd_simt"
    ▼
TTIR（tt.load/tt.dot/tt.reduce 原样）
    │  ① TritonToUnstructure: tt.load → ascend.indirect_load      ← SIMT 决策点
    │  ② TritonToLinalg:      ascend.indirect_load → call @triton_indirect_load_N
    │                         + func 盖章 parallel_mode="mix_simd_simt"
    ▼
Adapter IR (.ttadapter)          ← SIMT 契约在此完全定型
    │  ③ bishengir-compile 消费契约
    │     AdaptTritonKernel: call @triton_indirect_load → hfusion.indirect_load
    │     OutlineVectorFunction: 生成 _outlined_vf_N + hivm.vector_function
    │     HFusionToHIVM: hfusion.indirect_load → hivm.hir.indirect_load + vf_mode=SIMT
    │     InferVFMode: 父函数 → vf_mode=MIX
    │     SplitSimtModule: SIMT VF 拆独立 module
    ▼
NPU IR (.npuir)                  ← 执行形态：vf_mode 属性 + VF 拆分
```

---

## 一、Stage 0 — Python 入口

### 1.1 compile_mode 解析（kx/simit 语义）

`third_party/ascend/backend/compiler.py` `NPUOptions.__post_init__`（kx/simit 分支）：

```python
# compile_mode == "simd_simt" ⇒ 两条副作用：
force_simt_template = True          # 走 SimdSimtTemplate 语义
parallel_mode = "mix_simd_simt"     # kernel 级混合标记
```

**kanuak 与 kx/simit 的语义差异**（重要）：

| 分支 | `simd_simt` 的含义 | 产物 |
|------|-------------------|------|
| kanuak | SimdSimt 模式 | `hfusion.gather_load`（不 outline） |
| kx/simit | `force_simt_template=True` → SimdSimtTemplate 模式 | `triton_indirect_load` outline |

用户看到 `call @triton_indirect_load` 的产物匹配 **kx/simit** 语义。

### 1.2 pass pipeline（`ttir_to_linalg`，compiler.py）

`pm.run(mod, 'ttir_to_linalg')` 之前的 pass 顺序：

```
 1. add_triton_control_flow_opt
 2. add_triton_to_structure              # mask fallback + optimize_dynamic_offset
 3. add_discrete_mask_access_conversion  # (compile_on_910_95, force_simt_template, ...)
 4. add_triton_to_annotation
 5. add_triton_to_unstructure(...)       # ★ SIMT 决策点
 6. add_triton_to_hivm
 7. add_triton_to_hfusion
 8. add_triton_to_llvm
 9. add_bubble_up_operation
10. add_triton_to_structure（第二次）
11. add_triton_to_linalg(..., compile_mode)  # ★ outline + 盖章
```

costmodel 打开时（kx 分支），在此之前的额外两个 pass：
```
add_select_simd_simt_costmodel    # 打分 + 决策路线
add_materialize_simt_scopes       # 给 anchor 包 scope.scope {vec_mode="simt"}
```

costmodel 关闭（`auto_simt_scope_mode` 默认 "off"）时，这两个 pass 不运行，直接走 legacy 管线。

---

## 二、Stage 1 — TTIR 阶段：TritonToUnstructure（SIMT 决策点）

文件：`third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp`

### 2.1 模式解析

```cpp
// runOnOperation: 864-871 (kanuak)
unstructureCompileModeFlag = resolveCompileMode(compileMode, forceSimtTemplate);
forceSimtTemplateFlag = (mode == SimdSimtTemplate);
// resolveCompileMode: forceSimtTemplate ? SimdSimtTemplate : parseCompileMode(mode)
```

### 2.2 SIMT fast path 触发条件

```cpp
// 530-548 (kanuak)
simtTemplateLoadStoreFastPathEnabled =
    compileOn91095Flag && forceSimtTemplateFlag &&
    ((非结构化 && sizeInByte < 64) || mixCompileDiscreteMask) && rank ≤ 5;
```

命中 → `tryRewriteIndirectFastPath`（kx 工作树，:179）：

```cpp
rewriter.create<triton::ascend::IndirectLoadOp>(
    loc, resultType, newPtr, ptrOffset, mask, other, volatile);
// tt.load → ascend.indirect_load（仍是 MLIR op，未 outline、未建函数）
```

不命中 → 退化为 scf.for 标量循环 + `tensor::InsertSlice`（保留 tt.load）。

**"非结构化"判断**：load 的地址依赖运行时计算的 index（如 causal_conv1d 的 `task_id` 分解），编译器无法静态解析访问模式 → 判定为非结构化 → 走 SIMT 模板。

---

## 三、Stage 2 — TritonToLinalg：outline 为 `@triton_indirect_load`

文件：`third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp`

### 3.1 IndirectLoadConverter（:2703-2737, kx）

对每个 `ascend.indirect_load`：

1. 在 module 末尾创建**无 body 的私有函数声明**：
   ```mlir
   func.func private @triton_indirect_load[_N](
       memref<?xf16>,                  // base buffer
       tensor<...xi64>[, mask, other]  // indices / mask / other
   ) -> tensor<...> attributes {isVolatile = false}
   ```
2. 原处替换为 `func.call @triton_indirect_load_N`

**函数体从哪来**：由 BiShengIR 的 SIMT template 库/bitcode 在 npuir 阶段提供——adapter 里的只是"标记函数声明"。

### 3.2 函数级盖章（TritonToLinalgPass.cpp:475-490, 985-1000）

统计 kernel 内 SIMT op（`isSIMTOp`，:164-215），然后给 func 打属性：

```mlir
func.func @kernel(...) attributes {
    mix_mode = "mix",                    // 有 dot 则为 "mix"，无 dot 为 "aiv"
    parallel_mode = "mix_simd_simt"      // ★ kernel 级混合标记
}
```

注意：统计必须在 strided rewrite **之后**，否则会误标为 "simd"。

---

## 四、Stage 3 — .ttadapter：SIMT 契约完全定型

`.ttadapter` 就是 `ttir_to_linalg` 进程内 C++ pass 管线跑完后的 module dump（`dump_manager.put(str(mod), "kernel.ttadapter.mlir")`）。实测用户产物：

| 检查项 | 实测结果 |
|--------|---------|
| `scope.scope` | **0 处**（no-costmodel 路径不走 scope 机制） |
| `vec_mode` / `vector_mode` | 0 处 |
| `ascend.unstructured_load` | 0 处（已被 outline 消费） |
| `func.func private @triton_indirect_load` | **7 处声明 + 7 处 call** |
| func 属性 | `parallel_mode = "mix_simd_simt"` + `mix_mode = "mix"` |

**结论：到这个阶段，"哪些 load 走 SIMT" + "kernel 级 parallel_mode" 已经完全定型。** adapter 中的 `@triton_indirect_load` 调用就是 SIMT 标记本身。

---

## 五、Stage 4 — npuir（BiShengIR）：消费契约，不做独立判定

bishengir-compile 收到 adapter IR 后，按 regbase pipeline（A5）依次执行：

### 5.1 AdaptTritonKernelPass

文件：`AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/AdaptTritonKernel.cpp:365-477`

`TritonIndirectLoadToHFusionIndirectLoadPattern` 把标记函数调用重写为结构化 op：

```mlir
call @triton_indirect_load_3(...)  →  hfusion.indirect_load(...)
// mask i1→i8 转换、补 other、更新 DirectlyUsedGMArgIdxList
```

### 5.2 VF Outlining（HFusion 阶段）

`OutlineVectorFunction.cpp:150` 生成 `_outlined_vf_N` 命名，并打属性（:73-84）：

```mlir
func.call @kernel_fused_4_outlined_vf_0(...) {hivm.vector_function, no_inline}
```

### 5.3 HFusionToHIVM：SIMT 标记落到 op 上

文件：`AscendNPU-IR/bishengir/lib/Dialect/HIVM/Conversion/HFusionToHIVM/HFusionToHIVM.cpp:1576-1604`

```cpp
// HFusionToHIVMIndirectLoadOp:
// hfusion.indirect_load → hivm::IndirectLoadOp（打印为 hivm.hir.indirect_load）
// 并当场盖章：
op->setAttr(VFModeAttr::name, VFMode::SIMT);   // :1597-1598
```

**这是 npuir 里 `hivm.hir.indirect_load` + `vf_mode=SIMT` 逐 op 标记的来源。**

`VFModeAttr` 枚举（`HIVMAttrs.td:1205-1208`）：
```td
def HIVM_VF_SIMD : I32EnumAttrCase<"SIMD", 0>;
def HIVM_VF_SIMT : I32EnumAttrCase<"SIMT", 1>;
def HIVM_VF_MIX  : I32EnumAttrCase<"MIX",  2>;
```

### 5.4 InferVFMode：父函数判 MIX

文件：`AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/InferVFMode.cpp:83-157, 201-213`

规则：op 默认 SIMD、显式 SIMT 指定、嵌套冲突则 MIX；对 call 递归查被调函数。只作用于 `hacc.entry` 设备函数。

**这是父函数 `hivm.vf_mode = MIX` 的唯一来源**：kernel 里既有 SIMD VF 调用又有 SIMT op/函数 → 推断为 MIX。

### 5.5 Mix 专属阶段：AutoScope → OutlineScope → SplitSimtModule

regbase pipeline（`PassPipeline.cpp:508-521`，`EnableSimdSimtMixCompile` 时）：

1. **AutoScope**（`AutoScope.cpp`）：为 GatherLoad/ScatterStore 种子创建保守 subgraph scope；对已存在且 `vector_type="simt"` 的 scope 补 `outline` + `tcore_type=AIV` + `vf_mode=SIMT`（:246-258）
2. **LegalizeBoolForSimtVF / InsertMemSemanticForSimtVF / ...**：SIMT VF 合法性处理
3. **OutlineScope**（`OutlineScope.cpp:172-173`）：带 `outline` 的 SIMT scope → 独立 func，**拷贝全部属性**（含 `vf_mode=SIMT`）
4. **SplitSimtModule**（`SplitSimtModule.cpp:44-126`）：
   - 收集 `util::isSIMTVF`（带 `vf_mode=SIMT` 的 func，`Utils.cpp:1496-1502`）
   - 每个 SIMT VF clone 进独立 `ModuleOp` + `hacc.simt_module` 属性
   - 主模块中的 SIMT VF 保留 private 声明 + 尾部补 3 个 i32 grid size 参数
   - 拆分后 SIMT 模块走独立后端管线（TTIR→LLVM）

### 5.6 npuir 实例核对（causal_conv1d）

```
父函数:  hivm.vf_mode = #hivm.vf_mode<MIX>     ← InferVFMode
         hivm.func_core_type = <AIV>
         mix_mode = "aiv"                        ← 前端透传
         parallel_mode = "mix_simd_simt"         ← 前端透传

SIMD VF 调用:  {hivm.vector_function, no_inline}
合并过:        {hivm.vector_function, no_inline, ptc_simdvf}   ← MergeVecScope

SIMT 负载:     hivm.hir.indirect_load ins(...) outs(...)       ← 内联在父函数内
               （该 kernel 的 SIMT 负载未被 outline 成独立 VF，
                 SIMT 身份以逐 op 标记形式存在）
```

---

## 六、两条标记路线的对比

| | costmodel 路径 | no-costmodel 路径（legacy） |
|---|---|---|
| 决策者 | C++ costmodel（anchor 分析 + 打分） | TritonToUnstructure 的模式分析 |
| TTIR 标记 | `scope.scope {vec_mode="simt"}` 包住 anchor | `tt.load` → `ascend.indirect_load` |
| adapter 产物 | `scope.scope` 保留（BiShengIR 原生构造） | `call @triton_indirect_load_N` |
| backend 消费 | AutoScope/OutlineScope 识别 `vector_type="simt"` → outline + vf_mode=SIMT | AdaptTritonKernel → hfusion.indirect_load → hivm.indirect_load + vf_mode=SIMT |
| kernel 级属性 | costmodel 报告属性（ascend.simt_costmodel.*）+ parallel_mode | 只有 parallel_mode |

**两者殊途同归**：最终都是 BiShengIR 里的 `hivm.hir.indirect_load`（或 SIMT VF）+ 父函数 `vf_mode=MIX`。

---

## 七、回答最初的问题

> "同事说 simt 标记在 npuir；但 adapter 里已经有 call @triton_xx 了，那岂不是 adapter 已经标记好了？"

**你观察得对，同事的说法不准确。** 准确的分工是：

1. **adapter 阶段决定"是什么"**：哪些 load 是非结构化的、走 SIMT 模板（`ascend.indirect_load` → `@triton_indirect_load` 调用 + `parallel_mode="mix_simd_simt"`）——这是**语义决策**，在 triton-ascend 的 C++ pass 里完成。
2. **npuir 阶段决定"怎么执行"**：消费契约，做 VF outline、fusion、模块拆分、同步插入，并挂上执行形态属性（`hivm.vf_mode`、`hivm.vector_function`、`ptc_simdvf`）——这是**执行形态的物化**，不做独立判定。
3. 同事可能混淆的点：npuir 里的 `hivm.vf_mode=SIMT/MIX` 属性确实是在 BiShengIR 阶段才挂上的，但这些属性的依据（哪些 op 是 SIMT）来自 adapter 的契约，BiShengIR 只是"翻译"而非"决策"。

---

## 附：关键文件索引

| 阶段 | 文件 | 关键行 |
|------|------|--------|
| compile_mode 解析 | `third_party/ascend/backend/compiler.py` | NPUOptions.__post_init__:1221-1223 |
| pass pipeline | `third_party/ascend/backend/compiler.py` | ttir_to_linalg:155-230 |
| SIMT 决策 | `third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp` | :530-548, :864-871 |
| indirect_load 创建 | 同上 | tryRewriteIndirectFastPath:179 |
| outline 声明 | `third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp` | IndirectLoadConverter:2703-2737 |
| isSIMTOp | `third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp` | :164-215 |
| 函数盖章 | 同上 | :475-490, :985-1000 |
| costmodel scope | `third_party/ascend/costmodel/lib/AscendModel/RouteModel/Transforms/MaterializeSimtScopes.cpp` | :53-67, :171-180 |
| AdaptTritonKernel | `AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/AdaptTritonKernel.cpp` | :365-477 |
| VF outline | `AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/OutlineVectorFunction.cpp` | :73-84, :150 |
| HFusionToHIVM | `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Conversion/HFusionToHIVM/HFusionToHIVM.cpp` | :1576-1604 |
| InferVFMode | `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/InferVFMode.cpp` | :83-157, :201-213 |
| AutoScope | `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/AutoScope.cpp` | :246-258 |
| SplitSimtModule | `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/SplitSimtModule.cpp` | :44-126 |
| VFMode 枚举 | `AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td` | :1205-1208 |
| isSIMTVF | `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Utils/Utils.cpp` | :1496-1502 |
