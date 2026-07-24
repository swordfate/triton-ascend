# TTIR → TTAdapter (Linalg) 转换详解

## 0. 先修知识：这个转换的本质

TTIR 是 **Triton 的编程模型**——SPMD（单程序多数据）、指针类型、自动边界检查。TTAdapter（即 Linalg IR）是 **MLIR 标准方言**——memref（带地址空间的内存引用）、tensor（不可变多维数组）、显式数据搬运。

**核心变化方向**：把 Triton 的"智能"操作拆解为 MLIR 基本操作的组合——指针变成 memref，SPMD 变成函数参数，一条 `tt.load` 拆成 7 步显式操作。

转换由 `TritonToLinalgPass` 完成（`third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp`），pipeline 中有 11 个子步骤，核心是类型转换（`TritonTypeConverter`）+ 各 Op 的 `ConversionPattern`。

---

## 1. 函数签名：从 `tt.func` 到 `func.func`

### TTIR（kernel.ttir.mlir 第 7-11 行）

```mlir
tt.func public @add_kernel(
  %x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %n_elements: i32 {tt.divisibility = 16 : i32})
attributes {noinline = false}
```

### TTAdapter（kernel.ttadapter.mlir 第 7-19 行）

```mlir
func.func @add_kernel(
  %arg0: memref<?xi8>,                                   // 新增：同步锁
  %arg1: memref<?xi8>,                                   // 新增：workspace
  %x_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %y_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
  %output_ptr: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
  %n_elements: i32 {tt.divisibility = 16 : i32},
  %arg6: i32, %arg7: i32, %arg8: i32,                    // 新增：num_programs[0,1,2]
  %arg9: i32, %arg10: i32, %arg11: i32)                  // 新增：program_id[0,1,2]
attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
            global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"}
```

### 逐项解读

| # | 变化 | 代码依据 | 原因 |
|---|------|---------|------|
| 1.1 | `tt.func` → `func.func` | `TritonToLinalgPass.cpp:461` — `builder.create<func::FuncOp>()` | 脱离 Triton 方言，进入 MLIR 标准方言，后续 Linalg→NPUIR 的 pass 不认识 `tt.func` |
| 1.2 | `!tt.ptr<f32>` → `memref<?xf32>` | `TritonToLinalgPass.cpp:191-202` — `TritonTypeConverter` 中 `addConversion([](triton::PointerType ptrType) { return MemRefType::get({ShapedType::kDynamic}, elem); })` | Triton 指针 = 一个未知大小的内存地址。最自然的 MLIR 表示是动态大小的 1D memref（`?` = 动态维度）。后续 `memref.reinterpret_cast` 会加上具体大小和 offset |
| 1.3 | 多了 `%arg0`、`%arg1`（sync lock + workspace） | `TritonToLinalgPass.cpp:1269-1304` — 在转换后对所有带 `global_kernel` 属性的 `func::FuncOp` 在最前面 `insertArgument` | Ascend NPU 的每个 kernel 都需要同步锁（多 wave 间的同步）和 workspace（临时内存池）。TTIR 层面不需要关心这些硬件细节，但 Linalg 层开始接轨硬件 |
| 1.4 | 多了 `%arg6`~`%arg11`（6 个 i32） | `TritonToLinalgPass.cpp:223-255` — `addProgramInfo()` 追加 `TRITON_PROGRAM_INFO_ARG_COUNT = LAUNCH_GRID_RANK * 2 = 6` 个 i32 参数 | SPMD 模型中 `tl.program_id(axis)` 和 `tl.num_programs(axis)` 在 Triton 层面是"指令"，但底层实际是从硬件寄存器/函数参数读取。转换成 `func.func` 后，必须显式传入。`LAUNCH_GRID_RANK=3` 支持 x/y/z 三维 grid |
| 1.5 | 新增 `mix_mode = "aiv"` | `TritonToLinalgPass.cpp:468-477` — 检测是否有 `tt.dot`，没有则设 `"aiv"`（纯向量模式） | vecadd 只有 addf，无矩阵乘，所以是纯 AIV（Ascend Intelligent Vector）模式。后端需要这个标记决定用哪个编译器路径 |
| 1.6 | 新增 `parallel_mode = "simd"` | `TritonToLinalgPass.cpp:479-484` — 检测是否有 SIMT op，没有则设 `"simd"` | vecadd 无 SIMT 操作 |
| 1.7 | 新增 `global_kernel = "local"` | `addProgramInfo()` 中根据参数设置 | local kernel = Triton jit 编译的 kernel，区别于 global kernel（预编译的） |
| 1.8 | 新增 `tt.tensor_kind` | `MarkTensorKindPass`（pipeline 第 4 步）分析参数用途：`0` = INPUT，`1` = OUTPUT | x_ptr 和 y_ptr 是纯读（INPUT），output_ptr 是纯写（OUTPUT）。后端需要知道每个 memref 的读写属性来做数据流优化 |
| 1.9 | `noinline = false` 消失 | Triton 特有的属性，`func.func` 不认 | `noinline` 是 Triton JIT 编译的属性，到了 Linalg 层已不适用 |

### 参数映射推导：`%arg9` 为什么是 `program_id[0]`

```cpp
// FunctionConverter.cpp:29-41
auto func = op->getParentOfType<FunctionOpInterface>();
auto numArgs = func.getNumArguments();  // addProgramInfo 之后 = 10 (4 原始 + 6 program info)
auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);
// program_id(0) = arg index 10 - 3 + 0 = 7 (0-indexed)
```

`addProgramInfo` 追加 6 个参数，顺序是 `num_programs[0,1,2], program_id[0,1,2]`。所以 arg index 7 (0-indexed) = `program_id[0]`。

然后 workspace + sync lock 在转换完成后 `insertArgument` 在位置 0 和 1，所有原始参数后移两位：
- arg index 0 → sync lock
- arg index 1 → workspace
- arg index 7 → **arg index 9** = `%arg9` = `program_id[0]`

验证：`%block_start_0 = arith.muli %arg9, %block_start : i32` — `%arg9 * 1024`，正是 `pid * BLOCK_SIZE`。

---

## 2. 常量：从 Tensor 到 Scalar

### TTIR（第 12-18 行）

```mlir
%block_start = arith.constant 1024 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>    // 1024个0.0的tensor
%offsets = arith.constant dense<-2147483648> : tensor<1024xi64>  // overflow check
%offsets_0 = arith.constant dense<2147483647> : tensor<1024xi64> // overflow check
%c-2147483648_i64 = arith.constant -2147483648 : i64
%c2147483647_i64 = arith.constant 2147483647 : i64
%c1024_i32 = arith.constant 1024 : i32
```

### TTAdapter（第 20-23 行）

```mlir
%y = arith.constant 0.000000e+00 : f32        // 标量 0.0
%x = arith.constant 1024 : index               // 1024，index类型
%block_start = arith.constant 1024 : i32       // 1024，i32类型
%block_start_0 = arith.muli %arg9, %block_start : i32
%block_start_1 = arith.index_cast %block_start_0 : i32 to index
```

### 逐项解读

| # | 变化 | 代码依据 | 原因 |
|---|------|---------|------|
| 2.1 | `dense<0.0> : tensor<1024xf32>` → `0.0 : f32` 标量 | 在 TTIR 中 `%cst` 是 `tt.load` 的 `other` 参数（掩码外的填充值），必须是 tensor。转换后填充方式变了——用 `linalg.fill` 以**标量**填充，不需要 tensor 常量 | Load 转换代码（`LoadStoreConverter.cpp:220-222`）中 `linalg::FillOp` 的 `ins` 只需要一个标量值，broadcast 由 `linalg.fill` 自动完成 |
| 2.2 | Overflow 检查相关的 4 个常量全部消失 | `DeviceAssertConverter`（`TritonOpConverter.cpp:2082-2093`）检测到 message 包含 `"overflow detected"` 就擦除 | 这些 assert 是 Triton 编译器**自动插入**的整数溢出检查，不是用户写的。在 NPU 上，已知数据范围（BLOCK_SIZE=1024, offsets<2^31），不需要运行时检查 |
| 2.3 | `1024 : i64` → `1024 : i32` 和 `1024 : index` 两种 | 在 TTIR 中 `block_start` 用 i64 是因为指针运算需要 64 位地址。转换后 memref 的 offset/size/stride 用 `index` 类型（目标平台指针宽度），block_start 计算用 i32 因为 program_id 是 i32 | MLIR 中 `index` 类型的位宽 = 目标平台的指针宽度（aarch64 上 64-bit），这是 MLIR 的设计约定 |

---

## 3. SPMD → 显式参数：`program_id` 的消失

### TTIR（第 19 行）

```mlir
%pid = tt.get_program_id x : i32
```

一条 Triton 指令：从硬件读取当前 program 的 x 轴 ID。

### TTAdapter（无对应指令）

```mlir
%block_start_0 = arith.muli %arg9, %block_start : i32
```

`%arg9` 直接就是 program_id，不再有 `tt.get_program_id`。

**代码依据**：`FunctionConverter.cpp:29-41` — `GetProgramIDConverter::matchAndRewrite`

```cpp
LogicalResult GetProgramIDConverter::matchAndRewrite(
    triton::GetProgramIdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(numArgs - GetProgramIDConverter::LAUNCH_GRID_RANK + axis);
  rewriter.replaceOp(op, id);  // 直接把 tt.get_program_id 替换为对应的函数参数
  return success();
}
```

---

## 4. 指针运算：`splat` + `addptr` → `memref.reinterpret_cast`

这是整个转换中**最核心、最精妙**的部分。

### TTIR（第 27-37 行）

```mlir
%offsets_7 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
%offsets_8 = tt.splat %block_start_6 : i32 -> tensor<1024xi32>
%offsets_16 = arith.addi %offsets_8, %offsets_7 : tensor<1024xi32>
// ... overflow check ...
%x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
%x_18 = tt.addptr %x, %offsets_16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
%x_19 = tt.load %x_18, %mask_17, %cst : tensor<1024x!tt.ptr<f32>>
```

5 步操作才能描述"加载 x[pid*1024 : pid*1024+1024]"：
1. 生成 [0, 1023]
2. 广播 block_start
3. 相加得全局偏移
4. `splat` 基础指针
5. `addptr` 施加偏移

### TTAdapter（第 25 行）

```mlir
%x_2 = memref.reinterpret_cast %x_ptr to offset: [%block_start_1], sizes: [1024], strides: [1]
    : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
```

**一条指令完成**。`reinterpret_cast` 的含义是：从 `%x_ptr`（无限大小的 memref）中，"重新解释"出一块从 offset=`%block_start_1` 开始、大小=1024 元素、stride=1 的连续区域。

### 为什么可以这样折叠？

**代码依据**：`BlockDataParser::parseAddPtr`（`BlockPtrAnalysis.cpp:866-914`）

核心逻辑：
1. `parse(ptr)` → 提取 base pointer + base offset
2. `parse(offset)` → 提取 offset 的 sizes/offsets/strides
3. `Data.addBlock(ptrBlock, offsetBlock)` → offset 加到 ptr 的 offset 上

对于一个 `tt.addptr(tt.splat(base_ptr), pid*1024 + range(0,1024))`：
- base pointer = `x_ptr`
- offset 的 sizes = [1024], offsets = [pid*1024], strides = [1]

解析后：
- 总 offset = base_offset(0) + ptr_offset(pid*1024) = pid*1024
- sizes = [1024], strides = [1]

然后 `BlockDataParser::rewriteAddPtr:1372` 调用 `data.createCastOp()` 生成 `memref::ReinterpretCastOp`，将 offset 折叠进去。

### 为什么 `tt.make_range` 和 `tt.splat` 消失但 offset 还在？

因为它们不是被单独转换，而是被**消费/折叠**进了更大的结构中——`BlockDataParser` 递归解析指针链，提取 offset/size/stride 三元组，一次性生成 memref 视图。

---

## 5. `tt.load`：从 1 条指令变成 7 步操作

这是变化最大的部分，也是最体现 **"高级语义拆解为低级操作"** 的地方。

### TTIR（第 42 行）

```mlir
%x_19 = tt.load %x_18, %mask_17, %cst : tensor<1024x!tt.ptr<f32>>
```

一条指令完成了：指针解引用 + 边界检查 + mask 处理 + zero padding + 输出 tensor。

### TTAdapter（第 26-38 行）

```mlir
%x_3 = memref.alloc() : memref<1024xf32>              // ① 分配本地 buffer
%x_4 = arith.addi %block_start_1, %x : index          // ② block_start + 1024
%n_elements_5 = arith.index_cast %n_elements : i32 to index
%x_6 = arith.maxsi %block_start_1, %n_elements_5      // ③ max(block_start, n)
%x_7 = arith.minsi %x_4, %x_6 : index                 // ④ min(block_start+1024, ③)
%x_8 = arith.subi %x_7, %block_start_1 : index        // ⑤ 有效数据长度
%x_9 = arith.cmpi slt, %x_8, %x : index               // ⑥ 需要 mask padding?
scf.if %x_9 {
  linalg.fill ins(%y : f32) outs(%x_3 : memref<1024xf32>)  // ⑦a 先填充0.0
} {hivm.unlikely_condition}
%x_10 = memref.subview %x_2[0] [%x_8] [1]             // ⑦b 源 subview
%x_11 = memref.subview %x_3[0] [%x_8] [1]             // ⑦c 目标 subview
memref.copy %x_10, %x_11                               // ⑦d 数据搬运
%x_12 = bufferization.to_tensor %x_3 restrict writable // ⑧ memref→tensor
```

### 逐步解读

| 步骤 | 操作 | 代码依据 | 含义 |
|------|------|---------|------|
| ① | `memref.alloc()` | `LoadStoreConverter.cpp:360-361` — `allocOp = rewriter.create<memref::AllocOp>(loc, MemRefType::get(memRefShape, memRefElementType))` | 在本地内存（UB, Unified Buffer）中分配 1024 个 f32 的临时空间 |
| ②-④ | 边界计算 | `LoadStoreConverter.cpp:473-474` — `MaskState::parse(mask)` | 计算 `min(block_start+1024, n_elements)`，得到实际有效的结束位置。`maxsi` 处理 block_start 已经超出数组范围的情况 |
| ⑤ | 有效长度 | `subi` | `valid_end - block_start` = 实际有效数据量。例如 n=98432 时最后一个 block `pid=96, block_start=98304`，有效长度 = 98432-98304 = 128 |
| ⑥ | 是否需要 mask | `cmpi slt` | 如果有效长度 < 1024（BLOCK_SIZE），说明末尾那块不完整，需要 mask |
| ⑦a | `linalg.fill` | `LoadStoreConverter.cpp:185-225` — `fillTensorWithOtherForMaskScenario` | 先把整块 buffer 填 0（对应 TTIR 中 `other=%cst`）。多余的零会被后续 copy 覆盖 |
| ⑦b-⑦d | subview + copy | `LoadStoreConverter.cpp:515-518` — `mstate.getSubview` + `memref::CopyOp` | 用 subview 裁剪出有效数据范围（动态大小），然后 `memref.copy` 从 Global Memory (GM) 搬运到本地 (UB) |
| ⑧ | `bufferization.to_tensor` | `LoadStoreConverter.cpp:91-102` — `toTensorAndReplace` | 将 memref 包装成 tensor，因为后续 `arith.addf` 操作的是 tensor。**注意**：`restrict writable` 标记表示这个 tensor 独占底层 memref，不会有 aliasing 问题 |

### `{hivm.unlikely_condition}` 的含义

```cpp
// LoadStoreConverter.cpp:222-224
ifOp->setAttr(rewriter.getStringAttr("hivm.unlikely_condition"),
              UnitAttr::get(rewriter.getContext()));
```

告诉后端编译器（bisheng），这个 `scf.if` 的 then 分支是 **unlikely** 的——大多数情况下 tile 是完整的，不需要 mask padding。后端可以利用这个 hint 做分支预测优化，把 padding 路径放在指令 cache 的冷区。在 vecadd 中 `n_elements=98432, BLOCK_SIZE=1024`，97 个 program 中只有最后 1 个不完整，几率约 1%。

---

## 6. `arith.addf`：不变

### TTIR（第 46 行）

```mlir
%output = arith.addf %x_19, %y_21 : tensor<1024xf32>
```

### TTAdapter（第 49 行）

```mlir
%output = arith.addf %x_12, %y_17 : tensor<1024xf32>
```

唯一的区别是操作数名字变了（`%x_19`→`%x_12`, `%y_21`→`%y_17`），因为它们来自不同的转换路径。**运算本身完全不变**——`arith.addf` 是标准 MLIR 方言，TTIR 和 Linalg 都能用。

---

## 7. `tt.store`：从 1 条指令变成 4 步操作

### TTIR（第 47-49 行）

```mlir
%0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
%1 = tt.addptr %0, %offsets_16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
tt.store %1, %output, %mask_17 : tensor<1024x!tt.ptr<f32>>
```

### TTAdapter（第 50-53 行）

```mlir
%reinterpret_cast = memref.reinterpret_cast %output_ptr to offset: [%block_start_1], sizes: [1024], strides: [1]
%extracted_slice = tensor.extract_slice %output[0] [%x_8] [1]
%subview = memref.subview %reinterpret_cast[0] [%x_8] [1]
bufferization.materialize_in_destination %extracted_slice in writable %subview
```

| 步骤 | 操作 | 代码依据 | 含义 |
|------|------|---------|------|
| 7.1 | `reinterpret_cast` | 同 load，指定目标内存区域 | 确定写到 output_ptr 的哪个位置。`splat + addptr` 同样被 `BlockDataParser` 折叠 |
| 7.2 | `tensor.extract_slice` | `LoadStoreConverter.cpp:1143-1144` — `mlir::ConverterUtils::makeExtractSliceOp(val, srcOffsets, boundarySizes)` | 从 1024 元素的 tensor 中提取前 `%x_8` 个（有效数据部分），丢弃 mask 外的部分 |
| 7.3 | `memref.subview` | `LoadStoreConverter.cpp:1145-1146` — `makeSubViewOp(ptr, dstOffsets, boundarySizes)` | 在目标 memref 上裁剪出同样大小的窗口 |
| 7.4 | `materialize_in_destination` | `LoadStoreConverter.cpp:1147-1148` | 将 tensor slice 的内容写入 memref subview。这是 bufferization 框架的标准操作 |

`mask_17` 去哪了？`MaskState::parse()` 提取了连续 mask 的边界 → boundary size 变成动态 `%x_8` → store 不是无条件写整块，而是通过 `extract_slice` + `subview` 只写有效部分。

---

## 8. `tt.return` → `func.return`

### TTIR（第 50 行）

```mlir
tt.return
```

### TTAdapter（第 54 行）

```mlir
return loc(#loc)
```

**代码依据**：`TritonToLinalgPass.cpp:502-506`

```cpp
for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
}
```

把旧的 terminator (`tt.return`) 删除，创建新的 `func::ReturnOp`。`loc(#loc)` 保留了源代码位置信息用于调试。

---

## 完整转换映射汇总

```
TTIR                                TTAdapter (Linalg)                   转换器
────────────────────────────────────────────────────────────────────────────────
tt.func                           → func.func                           convertTTFunc
!tt.ptr<f32>                      → memref<?xf32>                       TritonTypeConverter
tt.get_program_id x               → %arg9 (函数参数)                      GetProgramIDConverter
arith.constant dense<0.0>         → arith.constant 0.0 (标量)             DenseConstant + canonicalize
tt.assert (overflow)              → 删除                                 DeviceAssertConverter
tt.make_range + tt.splat +        → memref.reinterpret_cast              parseAddPtr +
  tt.addptr                             (offset折叠)                     BlockDataParser
tt.load(mask, other)              → alloc + 边界计算 + linalg.fill        LoadConverter +
                                      + subview + memref.copy            MaskAnalysis
                                      + bufferization.to_tensor
arith.addf                        → arith.addf (不变)                    无需转换（标准方言）
tt.store(mask) + splat + addptr   → reinterpret_cast + extract_slice     StoreConverter +
                                      + subview +                        MaskAnalysis
                                      materialize_in_destination
tt.return                         → func.return                          convertTTFunc
(无)                              → sync lock + workspace 参数注入        insertArgument (post-conversion)
(无)                              → tt.tensor_kind 属性                  MarkTensorKindPass
(无)                              → mix_mode, parallel_mode 属性         convertTTFunc
```

## 总结：这个转换的"为什么"

TTIR 是 **面向程序员** 的表示——SPMD 模型、智能 load/store、自动边界检查。TTAdapter 是 **面向硬件后端** 的表示——显式内存管理、精确的数据搬运、memref 视图运算。

核心转换哲学：

1. **指针运算的代数化简**：`splat(base) + addptr(offsets)` → `reinterpret_cast(base, offset, sizes, strides)`。BlockDataParser 递归解析指针链，提取 offset/size/stride 三元组，一次性生成 memref 视图。

2. **隐式操作显式化**：`tt.load(mask, other)` 一条指令被拆成 7 步——分配 buffer、计算边界、填充默认值、裁剪视图、数据拷贝、转换为 tensor。每步都对应底层硬件的一个实际动作。

3. **SPMD 参数化**：`tt.get_program_id` 从"读取硬件寄存器的指令"变成"函数参数"。这让 kernel 变成了一个**纯函数**——给定所有参数，输出完全确定，不依赖任何全局状态。纯函数更容易做编译优化和测试。
