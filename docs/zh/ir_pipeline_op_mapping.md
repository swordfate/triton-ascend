# Triton IR 管线 — 经典 Op 变换速查

> 以 `_fwd_grouped_kernel_stage1_rope` (Flash Attention GQA + RoPE) 为例
> 三阶段 IR：`test_cases/` 下的 `.ttir` / `.ttadapter` 文件

## 一、管线概览

```
Python Triton kernel
    ↓ (triton.compile)
TTIR (Stage 1)                  ← 例: _fwd_grouped_kernel_stage1_rope.ttir
    ↓ (costmodel + SIMT materialization)
TTIR + SIMT scopes (Stage 2)    ← 例: ..._with_costmodel.ttir
    ↓ (triton-adapter lowering, ~100+ MLIR passes)
Adapter IR (Stage 3)            ← 例: ..._without_costmodel_simd-simt.ttadapter
    ↓ (BiShengIR / NPU compiler)
最终二进制
```

**关键点**：Stage 1→2 只在 costmodel 判定走 mixed_simd_simt 时才有实质性变化（把间接 load 的 `tt.load` 包进 SIMT scope）。Stage 2→3 才是**大变身**——所有高层 `tt.*` op 都下降到低层 dialect。

---

## 二、Op 分类速查表

### 2.1 计算类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.dot` (3个) | `func.call @triton_dot_*` (内联为向量乘加) | dot → NPU cube 指令序列，TTIR 阶段保留为 `tt.dot`，到 adapter 阶段才展开 |
| `tt.reduce` (4个: max, sum×2, add) | `arith.maxnumf`, `arith.addf`, `arith.mulf` 等 | reduction 完全展开为 arith 序列，不再有 `tt.reduce` |
| `arith.addf/subf/mulf/divf` | **同名保留** | 基础浮点运算直接透传 |
| `arith.addi/subi/muli/divsi/remsi` | **同名保留** | 整数地址运算直接透传 |
| `arith.cmpi` | **同名保留** | 比较运算透传（生成 mask） |
| `arith.select` | **同名保留** | 条件选择透传 |
| `math.exp` (2个) | **同名保留** | softmax 的 exp |
| `math.log` (1个) | **同名保留** | softmax 输出的 log |
| `arith.extsi` (TTIR 中不存在) | **新增** (12个) | 符号扩展 — adapter 阶段的类型转换 |
| `arith.index_cast` (TTIR 中不存在) | **新增** (16个) | index 类型转换 — adapter 阶段的地址计算 |
| `arith.truncf` (TTIR 中不存在) | **新增** (3个) | 浮点截断 — 精度转换 |

### 2.2 内存访问类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.load` — **结构化**（地址不依赖另一个 load） | `memref.load` | 连续/stride 访存 → memref 直接 load，15个 → 4个（其余变成 indirect_load） |
| `tt.load` — **间接**（地址依赖另一个 load 的结果） | `func.call @triton_indirect_load_N` | 每个间接 load 实例化一个独立函数（本例: 8个 → `@triton_indirect_load` ~ `@triton_indirect_load_5`），参数 `(memref<?xf16>, indices, mask, other)` |
| `tt.store` (3个) | `memref.copy` (4个) | store 变成 memref copy 操作 |
| 无 | `memref.alloc` (4个) | adapter 阶段显式分配临时 buffer |
| 无 | `memref.subview` (8个) | 从大 buffer 切子视图，对应 TTIR 的 pointer slicing |
| 无 | `memref.reinterpret_cast` (11个) | 重解释 memref 的形状/类型，对应 TTIR 的 reshape/expand_dims |

### 2.3 Pointer/Tensor 操作类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.splat` (28个) | **消失** → 融入 `arith.addi` + `arith.muli` | 基地址广播 → 直接参与地址计算 |
| `tt.broadcast` (28个) | **消失** → `memref.reinterpret_cast` / `memref.subview` | 广播语义 → memref shape 变换 |
| `tt.expand_dims` (23个) | **消失** → `memref.reinterpret_cast` | 维度扩展 → memref 重解释 |
| `tt.addptr` (22个) | **消失** → `arith.addi` + `arith.muli` | 指针偏移 → 地址算术展开：`ptr + offset * stride` = `addi(ptr, muli(offset, stride))` |
| `tt.make_range` (3个) | `tt.from_make_range` (2) + `tt.make_range_size` (2) + `tt.make_range_offset` (2) | range → 元数据保留（供后续 pass 使用），不再作为 op 语义 |
| `tt.ptr` (67个) | **类型注解，全部消失** | 纯类型标记（`!tt.ptr<tensor<...>>`），不产生 IR 指令 |

### 2.4 控制流类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `scf.for` (1个: KV block 循环) | **同名保留** (1个) | 循环结构保留，但 body 内容完全不同（TT: `tt.load`+`tt.dot` → adapter: `func.call`+`arith`+`memref`） |
| `scf.if` (4个→7个) | **同名保留** + 可能新增 | TTIR阶段4个（LAST_SPLIT判断、k_pe选择等），adapter 阶段增加到7个（其中部分来自 SIMT scope 的 `scf.if` 包装） |
| 无 (Stage 1) → `scf.if { hacc.simt_scope }` (Stage 2) | 被内联或保留为 `scf.if` + `func.call @triton_indirect_load_N` | SIMT scope 在 Stage 2 通过 `scf.if` + 属性标记实现；Stage 3 的 lowering 将其展开为 `func.call` 形式 |
| `scf.yield` | **同名保留** | 控制流的值返回 |

### 2.5 元数据/属性类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.get_program_id` (3个: batch, head, split) | **消失** → `func.func` 参数 | program_id 变成函数入口参数 |
| `tt.func` | `func.func` | 函数定义，去掉 tt 前缀 |
| `tt.return` | `func.return` | 返回语句，去掉 tt 前缀 |
| `tt.divisibility` (16个) | **同名保留** (16个) | 对齐提示，贯穿全程 |
| `tt.assert` | **消失** | debug assert 在 lowering 中去除 |
| 无 | `tt.tensor_kind` (9个) | adapter 阶段新增：标记 tensor 的内存位置（UB/L1/HBM 等） |
| 有 (Stage 2) | `ascend.simt_costmodel.*` **消失** | costmodel 报告属性只在 Stage 2 TTIR 上以 module attr 存在 |

---

## 三、Stage 1→2：Costmodel + SIMT Materialization

### 3.1 什么时候有变化

只有 costmodel 判定 `mixed_simd_simt` 且通过 margin gate 时，才会对 TTIR 做实质性修改。

### 3.2 具体变化

```
TTIR Stage 1:                          TTIR Stage 2 + materialization:
──────────────────                     ──────────────────────────────
%164 = tt.load %163, %162, %cst_2     scf.if %cond {
  : tensor<16x32xf16>                   %164 = tt.load %163, %162, %cst_2
                                          : tensor<16x32xf16>
                                        } {ascend.simt_scope}
```

间接 load 被包进 `scf.if`，带 `ascend.simt_scope` 属性标记，告诉后续 lowering pass "这里走 SIMT 路径"。

### 3.3 不变的部分

- 所有 `tt.dot`、`tt.reduce`、`arith.*`、`math.*` — **完全不动**
- 结构化 `tt.load` / `tt.store` — **不动**
- `scf.for` / `scf.if` 结构 — **保留**

---

## 四、Stage 2→3：Adapter Lowering（大变身）

### 4.1 `tt.dot` → `func.call @triton_dot_*`

3 个 dot 变成 3 个隐式的 dot 函数调用（在 adapter IR 中不一定以显式 `func.call` 出现，取决于 lowering 策略），最终对应 NPU 的 cube 指令。

本 ROPE kernel 的 3 个 dot：
```
(16×16) × (16×32) = 16×32   ← q_pe @ k_pe
(16×16) × (16×32) = 16×32   ← q @ kv
(16×32) × (32×16) = 16×16   ← p @ v (acc)
```

### 4.2 `tt.load` (间接) → `func.call @triton_indirect_load_N`

8 个间接 load → 8 个函数声明 + 调用点：

```
func.func private @triton_indirect_load_3(
    memref<?xf16>,       ← base buffer (K_Buffer / V_buffer)
    tensor<16x32xi64>,   ← linearized indices (kv_loc * stride + offset)
    tensor<16x32xi1>,    ← mask (offs_n < split_kv_end)
    tensor<16x32xf16>    ← other value (0.0)
) -> tensor<16x32xf16>
```

调用示例：
```
%154 = func.call @triton_indirect_load_3(
    %arg3,          ← K_Buffer memref
    %151,           ← 16×32 i64 indices
    %153,           ← bool mask
    %17             ← other = 0.0
) : ... -> tensor<16x32xf16>
```

### 4.3 `tt.load` (结构化) → `memref.load`

```
TTIR:  %42 = tt.load %41, %mask, %other : tensor<16x512xf16>
Adptr: %42 = memref.load %base[%offset] : memref<...xf16>
```

### 4.4 `tt.addptr` → `arith.addi` + `arith.muli`

```
TTIR:
  %158 = tt.addptr %107, %cst : tensor<16x32x!tt.ptr<f16>>
  ↓ 其中 %158 = base + offset * stride

Adptr:
  %addr = arith.muli %offset, %stride : i64
  %ptr  = arith.addi %base, %addr : i64
```

### 4.5 `tt.splat` / `tt.broadcast` / `tt.expand_dims` → `memref.reinterpret_cast`

这三个 TTIR op 都是 tensor shape 操作，不产生实际计算。到 adapter 阶段：
- `tt.splat`: 标量→tensor 广播 → 地址计算中直接用标量 + offset
- `tt.broadcast`: 维度广播 → `memref.reinterpret_cast` 改变 strides
- `tt.expand_dims`: 插入 size=1 的维度 → `memref.reinterpret_cast` 调整 shape

### 4.6 `tt.reduce` → `arith.*` 展开

```
TTIR:
  %200 = "tt.reduce"(%199) {axis=1: i32} ({
    ^bb0(%lhs: f32, %rhs: f32):
      %201 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %201 : f32
  }) : tensor<16x32xf32> → tensor<16xf32>

Adptr:
  (多个 arith.maxnumf 指令，对 axis=1 的元素逐对比大小)
```

### 4.7 `scf.for` body 完全改写

```
TTIR body:                           Adptr body:
──────────────────────               ─────────────────────
tt.load → kv_loc (结构化)            memref.load → kv_loc
tt.addptr → 地址计算                 arith.addi + arith.muli → 地址
tt.load → k_pe/v (间接)             func.call @triton_indirect_load_N → k_pe/v
tt.dot → qk                          (内联/调用的 dot 序列)
tt.reduce → softmax                  arith.maxnumf + math.exp + arith.addf + arith.divf
tt.store → 写输出                    memref.copy
```

---

## 五、快速定位：三大变化模式

### 模式 A：消失 → 隐入地址计算

`splat`, `broadcast`, `expand_dims`, `addptr`, `ptr` — 这些都是 **Triton 的指针抽象**，到 adapter 阶段全部消失。理解它们的关键是：它们只是描述"怎么算地址"，不产生实际数据移动。

### 模式 B：保留 op 名，保留语义

`arith.*`, `math.*`, `scf.if`, `scf.for`, `scf.yield` — 这些是MLIR 标准 dialect，adapter lowering 保留它们。

### 模式 C：替换为低层等价物

| 高层 | 低层 |
|------|------|
| `tt.dot` | `func.call @triton_dot_*` / cube 指令 |
| `tt.load` (间接) | `func.call @triton_indirect_load_N` |
| `tt.load` (结构化) | `memref.load` |
| `tt.store` | `memref.copy` |
| `tt.reduce` | `arith.*` 展开序列 |
| `tt.func` / `tt.return` | `func.func` / `func.return` |

---

## 六、REPO kernel 的三阶段速览

```
Stage 1: _fwd_grouped_kernel_stage1_rope.ttir (52KB, 291 lines)
  tt.load ×15  tt.dot ×3  tt.reduce ×4  tt.store ×3
  tt.splat ×28  tt.broadcast ×28  tt.expand_dims ×23  tt.addptr ×22
  scf.for ×1  scf.if ×4  arith.* ×122

Stage 2: ..._with_costmodel.ttir (66KB)
  (仅 mixed_simd_simt 场景有变化)
  + scf.if wrapper ×8 (每个间接 load 一个 SIMT scope)
  + module attrs: ascend.simt_costmodel.{effective,recommended,report_json,...}
  其余 op 不变

Stage 3: ..._without_costmodel_simd-simt.ttadapter (38KB)
  func.call @triton_indirect_load_N ×7 (8个声明)
  memref.load/store/alloc/copy/subview/reinterpret_cast ×31
  arith.* ×81 (addi/muli/cmpi/extsi/index_cast/...)
  scf.if ×7  scf.for ×1  math.exp ×2  math.log ×1
  tt.divisibility ×16  tt.tensor_kind ×9
  (tt.dot 内联不可见, tt.load/store/reduce/splat/broadcast/expand_dims/addptr 全部消失)
```
