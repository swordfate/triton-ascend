# Triton IR 管线 — 经典 Op 变换速查

> 以 `_fwd_grouped_kernel_stage1_rope` (Flash Attention GQA + RoPE) 为例
> 三阶段 IR：`test_cases/` 下的 `.ttir` / `.ttadapter` 文件

## 一、管线概览

```
Python Triton kernel (Stage 0)
    ↓ ast_to_ttir() — Python AST → MLIR
TTIR (Stage 1)                  ← 例: _fwd_grouped_kernel_stage1_rope.ttir
    ↓ costmodel + SIMT materialization
TTIR + SIMT scopes (Stage 2)    ← 例: ..._with_costmodel.ttir
    ↓ triton-adapter lowering, ~100+ MLIR passes
Adapter IR (Stage 3)            ← 例: ..._without_costmodel_simd-simt.ttadapter
    ↓ BiShengIR / NPU compiler
最终二进制
```

**关键点**：
- Stage 0→1：Python 语法糖全部显式化——广播、指针算术、隐式类型转换都变成 MLIR op
- Stage 1→2：只在 costmodel 判定 mixed_simd_simt 时变化（间接 load 包进 SIMT scope）
- Stage 2→3：**大变身**——所有高层 `tt.*` op 下降到低层 dialect

---

## 二、Stage 0→1：Python → TTIR

`triton.compile` 调用 `ast_to_ttir()`（`python/triton/compiler/code_generator.py`），将 Python AST 逐节点翻译为 TTIR。本节对照 ROPE kernel 源码给出具体实例。

### 2.1 函数签名 → `tt.func`

**Python (line 27-61):**
```python
@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q, K_Buffer, V_buffer, cos_sin_cache, positions,
    sm_scale, kv_indptr, kv_indices,
    Att_Out, k_pe_t_out,
    stride_qb, stride_qh, stride_buf_kbs, stride_buf_vbs,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_kpe_tokens_out_b,
    stride_cos_sin_cache_s, stride_positions_b,
    rotary_dim: tl.constexpr,      # ← tl.constexpr → i32 参数，有 tt.divisibility
    kv_lora_rank: tl.constexpr,
    ...
):
```

**TTIR (line 8):**
```mlir
  tt.func public @_fwd_grouped_kernel_stage1_rope(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},   // Q
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},   // K_Buffer
    %arg2: !tt.ptr<f16>,                                  // V_buffer
    %arg3: !tt.ptr<f16>,                                  // cos_sin_cache
    %arg4: !tt.ptr<i64>,                                  // positions
    %arg5: f32,                                           // sm_scale
    %arg6: !tt.ptr<i64>,                                  // kv_indptr
    %arg7: !tt.ptr<i64>,                                  // kv_indices
    %arg8: !tt.ptr<f16>,                                  // Att_Out
    %arg9: !tt.ptr<f16>,                                  // k_pe_t_out
    %arg10: i32 {tt.divisibility = 16 : i32},             // stride_qb ← tl.constexpr
    %arg11: i32, ...                                       // stride_qh
    ...
    %arg15: i32,                                          // rotary_dim ← tl.constexpr
    %arg16: i32,                                          // kv_lora_rank
    ...
  )
```

**规律：**
- **Tensor 参数**（`Q`, `K_Buffer` 等）→ `!tt.ptr<dtype>`，带 `tt.divisibility` 对齐提示
- **Scalar 参数**（`sm_scale`）→ `f32` 或 `i32`
- **`tl.constexpr`** → 普通 `i32` 参数，但带 `tt.divisibility`（如果值是 16 的倍数）
- **Strides** → 被保留为参数，因为 Triton 用它们做地址计算

### 2.2 `tl.program_id` → `tt.get_program_id`

**Python (line 63-65):**
```python
cur_batch = tl.program_id(0)
cur_head_id = tl.program_id(1)
split_kv_id = tl.program_id(2)
```

**TTIR (line 30-32):**
```mlir
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
```

`program_id(0/1/2)` → `tt.get_program_id x/y/z`，一一对应。

### 2.3 `tl.arange` → `tt.make_range`

**Python (line 75-76):**
```python
offs_c = tl.arange(0, BLOCK_C)          # [0, 1, ..., 511]
offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # [512, 513, ..., 575]
```

**TTIR (line 34, 43):**
```mlir
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %13 = tt.make_range {end = 32 : i32, start = 16 : i32} : tensor<16xi32>
```

`tl.arange(start, end)` → `tt.make_range {start, end}`。注意 BLOCK_C=512 但 TTIR 中是 16——这是因为 `tl.constexpr` 被 JIT 特化后，某些维度被 fold 成了常量，显示的可能是特化后的值。

### 2.4 广播（`[:, None]` / `[None, :]`） → `tt.expand_dims` + `tt.broadcast`

**Python (line 78-80):**
```python
off_q_pe = (
    cur_batch * stride_qb
    + cur_head[:, None] * stride_qh          # ← [16] → [16,1]
    + offs_qk_r[None, :]                      # ← [16] → [1,16]
)
```

**TTIR (line 45-52):**
```mlir
    // cur_head[:, None] → expand_dims(axis=1): [16] → [16,1]
    %15 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    // offs_qk_r[None, :] → expand_dims(axis=0): [16] → [1,16]
    %20 = tt.expand_dims %13 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    // 然后广播到相同 shape
    %21 = tt.broadcast %19 : tensor<16x1xi32> -> tensor<16x16xi32>
    %22 = tt.broadcast %20 : tensor<1x16xi32> -> tensor<16x16xi32>
```

**规律：** `tensor[None, :]` → `tt.expand_dims(axis=0)` + `tt.broadcast`；`tensor[:, None]` → `tt.expand_dims(axis=1)` + `tt.broadcast`。

### 2.5 `tl.load` → `tt.load`

**Python (line 86, 89-92) — 结构化 load:**
```python
cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
q = tl.load(Q + offs_q, mask=..., other=0.0)
```

**TTIR (line 59-62) — 标量 load:**
```mlir
    // kv_indptr + cur_batch → tt.addptr
    %29 = tt.addptr %arg6, %0 : !tt.ptr<i64>, i32         // kv_indptr + batch_idx
    %30 = tt.load %29 : !tt.ptr<i64>                       // 标量 load
    // kv_indptr + cur_batch + 1
    %31 = tt.addptr %29, %c1_i32 : !tt.ptr<i64>, i32
    %32 = tt.load %31 : !tt.ptr<i64>
```

**TTIR (line 70-71) — 向量 load:**
```mlir
    // Q 的 base ptr + 偏移
    %39 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %40 = tt.addptr %39, %26 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    // mask = %38, other = %cst_8 (0.0)
    %41 = tt.load %40, %38, %cst_8 : tensor<16x16x!tt.ptr<f16>>
```

**规律：**
- `ptr + scalar_offset` → `tt.addptr(ptr, offset)` — 指针算术显式化
- `mask=mask, other=other` → `tt.load %ptr, %mask, %other` — 第二个和第三个 operands
- 标量 load：没有 mask/other，直接 `tt.load %ptr : !tt.ptr<dtype>`

### 2.6 `tl.cdiv` → `arith.ceildivsi` + `arith.remsi`

**Python (line 94):**
```python
kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
```

`tl.cdiv(a, b)` → `arith.ceildivsi a, b`（向零取整除法的向上变体），有时会展开为 `(a + b - 1) // b` 的组合。

### 2.7 `RoPE: arange % (rotary_dim // 2)` → `arith.remsi`

**Python (line 111):**
```python
offs_rotary = tl.arange(0, BLOCK_R) % (rotary_dim // 2)
```

**TTIR (line 85):**
```mlir
    %55 = arith.remsi %54, %cst_12 : tensor<16xi32>
```

### 2.8 `tl.load` (间接，地址依赖另一个 load) → `tt.load` + BFS 标记

**Python (line 172-174):**
```python
kv_loc = tl.load(kv_indices + cur_batch_kv_start_idx + cur_batch_seq_len - 1)  # 先 load 索引
k_pe_last_token = tl.load(K_Buffer + kv_loc * stride_buf_kbs + offs_qk_r)     # 用索引用地址
```

**TTIR (line 190-192, 简化):**
```mlir
    // 第一个 load: 结构化 → kv_loc
    %xxx = tt.load %indices_ptr : ...
    // 第二个 load: 地址用了 %xxx（上一个 load 的结果）
    // → pointerDependsOnLoadedIndex BFS 会追溯到这个依赖关系
    %164 = tt.load %163, %162, %cst_2 : tensor<16x32x!tt.ptr<f16>>
```

**注意：** 在 TTIR 层面，间接 load 和普通 load **都是 `tt.load`**。区别在于 SSA use-def chain——间接 load 的地址 operand 能通过 BFS 追溯到另一个 `tt.load` 的结果。Costmodel 的 `pointerDependsOnLoadedIndex()` 专门做这个检测。

### 2.9 `tl.dot` → `tt.dot`

**Python (line 217):**
```python
qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))
```

**TTIR (line 207):**
```mlir
    %168 = tt.dot %88, %167, %cst_6 : tensor<16x16xf16> * tensor<16x32xf16> -> tensor<16x32xf32>
```

`tt.dot(lhs, rhs, acc)`，acc 是累加器（初始为 0）。shape `[M,K] * [K,N] -> [M,N]`。

### 2.10 `tl.maximum` / `tl.sum` → `tt.reduce`

**Python (line 245-246):**
```python
n_e_max = tl.maximum(tl.max(qk, 1), e_max)    # reduce max over axis=1
e_sum = e_sum * re_scale + tl.sum(p, 1)        # reduce sum over axis=1
```

**TTIR (简化):**
```mlir
    %200 = "tt.reduce"(%qk) {axis=1 : i32} ({
      ^bb0(%lhs: f32, %rhs: f32):
        %201 = arith.maxnumf %lhs, %rhs : f32
        tt.reduce.return %201 : f32
    }) : tensor<16x32xf32> -> tensor<16xf32>
```

`tl.max(x, axis=N)` → `tt.reduce(axis=N) { arith.maxnumf }`，`tl.sum(x, axis=N)` → `tt.reduce(axis=N) { arith.addf }`。

### 2.11 `for` → `scf.for`

**Python (line 191):**
```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    # body...
```

**TTIR (line 188):**
```mlir
    %120:3 = scf.for %arg19 = %50 to %52 step %c32_i64
        iter_args(%arg20 = ..., %arg21 = ..., %arg22 = ...)
        -> (tensor<16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
      // body...
      scf.yield %new_acc, %new_e_sum, %new_e_max : ...
    }
```

**规律：**
- `range(lower, upper, step)` → `scf.for %iv = %lower to %upper step %step`
- `iter_args` = 循环携带变量（对应 Python 中在循环前定义、循环内更新的变量：`acc`, `e_sum`, `e_max`）
- `scf.yield` = 每次迭代返回更新后的值

### 2.12 `if` → `scf.if`

**Python (line 102, 167-168):**
```python
if USE_ROPE:
    if LAST_SPLIT:
        kv_loc = tl.load(kv_indices + ...)
```

**TTIR (line 125, 158):**
```mlir
    %89 = scf.if %53 -> (tensor<16xf16>) {           // USE_ROPE ? ...
      // true branch: 计算 cos/sin/q_pe_rot
      scf.yield %q_pe_rotated : tensor<16xf16>
    } else {
      // false branch: 直接返回 q_pe
      scf.yield %q_pe : tensor<16xf16>
    }
    
    scf.if %90 {                                      // LAST_SPLIT ? ...
      // load k_pe_last_token
    }
```

**规律：** `tl.constexpr` 的 `if` 在 `@triton.jit` 特化时被消除（各特化版本生成不同 TTIR）。运行时 `if`（依赖 `tl.load` 的结果）→ `scf.if`。

### 2.13 `tl.zeros` / `tl.where` / `tl.exp` / `tl.log`

| Python | TTIR |
|--------|------|
| `tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")` | `arith.constant dense<0xFF800000> : tensor<16xf32>`（-inf 的 IEEE 754 表示） |
| `tl.where(mask, qk, float("-inf"))` | `arith.select %mask, %qk, %neg_inf` |
| `tl.exp(qk - n_e_max)` | `math.exp %diff` |
| `tl.log(e_sum)` | `math.log %e_sum` |

### 2.14 完整映射速查

| Python Triton | TTIR Op | 说明 |
|---------------|---------|------|
| `@triton.jit def fn(...)` | `tt.func @fn(...)` | 函数定义 |
| `tl.program_id(N)` | `tt.get_program_id x/y/z` | 0→x, 1→y, 2→z |
| `tl.arange(s, e)` | `tt.make_range {start=s, end=e}` | 生成整数序列 |
| `tl.load(ptr)` | `tt.load %ptr` | 标量 load |
| `tl.load(ptr, mask=m, other=o)` | `tt.load %ptr, %m, %o` | 向量 load + mask |
| `tl.store(ptr, val, mask=m)` | `tt.store %ptr, %val, %m` | store |
| `tl.dot(a, b)` | `tt.dot %a, %b, %acc` | 矩阵乘 |
| `tl.max(x, axis)` | `tt.reduce(axis) { arith.maxnumf }` | reduce max |
| `tl.sum(x, axis)` | `tt.reduce(axis) { arith.addf }` | reduce sum |
| `ptr + offset` | `tt.addptr %ptr, %offset` | 指针偏移 |
| `tl.arange(...) % N` | `arith.remsi %range, %N` | 取模（RoPE pattern） |
| `tl.cdiv(a, b)` | `arith.ceildivsi %a, %b` | 向上取整除法 |
| `a + b / a * b / a - b` | `arith.addf/mulf/subf` (float) / `arith.addi/muli/subi` (int) | 基础算术 |
| `a == b / a < b` | `arith.cmpi eq/slt` | 比较 |
| `tensor[None, :]` | `tt.expand_dims(axis=0)` + `tt.broadcast` | 广播 |
| `tensor[:, None]` | `tt.expand_dims(axis=1)` + `tt.broadcast` | 广播 |
| `tl.zeros(shape)` | `tt.splat` + `arith.constant` | 零张量 |
| `tl.where(cond, a, b)` | `arith.select %cond, %a, %b` | 条件选择 |
| `for ... in range(l, u, s)` | `scf.for %iv = %l to %u step %s` | 循环 |
| `if cond:` | `scf.if %cond` | 条件分支 |
| `tl.exp(x)` | `math.exp %x` | 指数 |
| `tl.log(x)` | `math.log %x` | 对数 |
| `tl.constexpr` 变量 | 常量折叠或 `i32` 参数 | 编译期常量 |

---

## 三、Op 分类速查表（TTIR → Adapter IR）

### 3.1 计算类

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

### 3.2 内存访问类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.load` — **结构化**（地址不依赖另一个 load） | `memref.load` | 连续/stride 访存 → memref 直接 load，15个 → 4个（其余变成 indirect_load） |
| `tt.load` — **间接**（地址依赖另一个 load 的结果） | `func.call @triton_indirect_load_N` | 每个间接 load 实例化一个独立函数（本例: 8个 → `@triton_indirect_load` ~ `@triton_indirect_load_5`），参数 `(memref<?xf16>, indices, mask, other)` |
| `tt.store` (3个) | `memref.copy` (4个) | store 变成 memref copy 操作 |
| 无 | `memref.alloc` (4个) | adapter 阶段显式分配临时 buffer |
| 无 | `memref.subview` (8个) | 从大 buffer 切子视图，对应 TTIR 的 pointer slicing |
| 无 | `memref.reinterpret_cast` (11个) | 重解释 memref 的形状/类型，对应 TTIR 的 reshape/expand_dims |

### 3.3 Pointer/Tensor 操作类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `tt.splat` (28个) | **消失** → 融入 `arith.addi` + `arith.muli` | 基地址广播 → 直接参与地址计算 |
| `tt.broadcast` (28个) | **消失** → `memref.reinterpret_cast` / `memref.subview` | 广播语义 → memref shape 变换 |
| `tt.expand_dims` (23个) | **消失** → `memref.reinterpret_cast` | 维度扩展 → memref 重解释 |
| `tt.addptr` (22个) | **消失** → `arith.addi` + `arith.muli` | 指针偏移 → 地址算术展开：`ptr + offset * stride` = `addi(ptr, muli(offset, stride))` |
| `tt.make_range` (3个) | `tt.from_make_range` (2) + `tt.make_range_size` (2) + `tt.make_range_offset` (2) | range → 元数据保留（供后续 pass 使用），不再作为 op 语义 |
| `tt.ptr` (67个) | **类型注解，全部消失** | 纯类型标记（`!tt.ptr<tensor<...>>`），不产生 IR 指令 |

### 3.4 控制流类

| TTIR (Stage 1/2) | Adapter IR (Stage 3) | 说明 |
|------------------|---------------------|------|
| `scf.for` (1个: KV block 循环) | **同名保留** (1个) | 循环结构保留，但 body 内容完全不同（TT: `tt.load`+`tt.dot` → adapter: `func.call`+`arith`+`memref`） |
| `scf.if` (4个→7个) | **同名保留** + 可能新增 | TTIR阶段4个（LAST_SPLIT判断、k_pe选择等），adapter 阶段增加到7个（其中部分来自 SIMT scope 的 `scf.if` 包装） |
| 无 (Stage 1) → `scf.if { hacc.simt_scope }` (Stage 2) | 被内联或保留为 `scf.if` + `func.call @triton_indirect_load_N` | SIMT scope 在 Stage 2 通过 `scf.if` + 属性标记实现；Stage 3 的 lowering 将其展开为 `func.call` 形式 |
| `scf.yield` | **同名保留** | 控制流的值返回 |

### 3.5 元数据/属性类

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

## 四、Stage 1→2：Costmodel + SIMT Materialization

### 4.1 什么时候有变化

只有 costmodel 判定 `mixed_simd_simt` 且通过 margin gate 时，才会对 TTIR 做实质性修改。

### 4.2 具体变化

```
TTIR Stage 1:                          TTIR Stage 2 + materialization:
──────────────────                     ──────────────────────────────
%164 = tt.load %163, %162, %cst_2     scf.if %cond {
  : tensor<16x32xf16>                   %164 = tt.load %163, %162, %cst_2
                                          : tensor<16x32xf16>
                                        } {ascend.simt_scope}
```

间接 load 被包进 `scf.if`，带 `ascend.simt_scope` 属性标记，告诉后续 lowering pass "这里走 SIMT 路径"。

### 4.3 不变的部分

- 所有 `tt.dot`、`tt.reduce`、`arith.*`、`math.*` — **完全不动**
- 结构化 `tt.load` / `tt.store` — **不动**
- `scf.for` / `scf.if` 结构 — **保留**

---

## 五、Stage 2→3：Adapter Lowering（大变身）

### 5.1 `tt.dot` → `func.call @triton_dot_*`

3 个 dot 变成 3 个隐式的 dot 函数调用（在 adapter IR 中不一定以显式 `func.call` 出现，取决于 lowering 策略），最终对应 NPU 的 cube 指令。

本 ROPE kernel 的 3 个 dot：
```
(16×16) × (16×32) = 16×32   ← q_pe @ k_pe
(16×16) × (16×32) = 16×32   ← q @ kv
(16×32) × (32×16) = 16×16   ← p @ v (acc)
```

### 5.2 `tt.load` (间接) → `func.call @triton_indirect_load_N`

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

### 5.3 `tt.load` (结构化) → `memref.load`

```
TTIR:  %42 = tt.load %41, %mask, %other : tensor<16x512xf16>
Adptr: %42 = memref.load %base[%offset] : memref<...xf16>
```

### 5.4 `tt.addptr` → `arith.addi` + `arith.muli`

```
TTIR:
  %158 = tt.addptr %107, %cst : tensor<16x32x!tt.ptr<f16>>
  ↓ 其中 %158 = base + offset * stride

Adptr:
  %addr = arith.muli %offset, %stride : i64
  %ptr  = arith.addi %base, %addr : i64
```

### 5.5 `tt.splat` / `tt.broadcast` / `tt.expand_dims` → `memref.reinterpret_cast`

这三个 TTIR op 都是 tensor shape 操作，不产生实际计算。到 adapter 阶段：
- `tt.splat`: 标量→tensor 广播 → 地址计算中直接用标量 + offset
- `tt.broadcast`: 维度广播 → `memref.reinterpret_cast` 改变 strides
- `tt.expand_dims`: 插入 size=1 的维度 → `memref.reinterpret_cast` 调整 shape

### 5.6 `tt.reduce` → `arith.*` 展开

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

### 5.7 `scf.for` body 完全改写

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

## 六、快速定位：三大变化模式

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

## 七、ROPE kernel 的三阶段速览

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
