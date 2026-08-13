# Computed-Index Gather 盲区：costmodel 与 template 路径的判定口径差异

> 算子：`silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`（SGLang ep_moe）
> 现象：costmodel 识别不到任何间接 load（anchor=0，mixed inapplicable），但同一个 kernel 不用 costmodel 跑 `simd_simt`（template 路径）的 ttadapter 里有 2 个 `@triton_indirect_load` 调用。
> 结论：两套判定对"间接访存"的定义不同——costmodel 要求地址依赖**从内存 load 出来的 index**，template 路径只要求地址是**非仿射表达式**（含 div/mod）。本 kernel 的 index 是算出来的（除/取模），只满足后者。

---

## 一、算子背景

### 1.1 Kernel 功能

MoE 激活量化：`y = silu(x) * gate_up * scale`，静态 per-tensor scale 量化，输出给 CUTLASS MoE GEMM 使用。

源码位置（SGLang）：`python/sglang/srt/layers/moe/ep_moe/kernels.py`（约 :433 起）。

### 1.2 实测延迟

| 路线 | 延迟 |
|------|------|
| SIMD（compile_mode=simd） | 16.3 us |
| SIMT-only（simt_only） | 3.9 us |
| 无 costmodel 的 simd_simt（template 路径） | 3.7 us |

### 1.3 Costmodel 输出（unknown loop 改成 true 后）

| 项 | 值 |
|----|-----|
| `simt_anchors.count` | **0** |
| `mechanisms` | `[]`（空） |
| `loaded_index_dependent_memory_ops` | **0** |
| `mixed_simd_simt` | **inapplicable**（reason: no_recognized_simt_anchor） |
| `all_simt_only` lowerability | native（整核纯 SIMT 可用） |
| candidate_costs | all_simd=507.04 / all_simt_only=8190.71 |
| decision | all_simd（**错误**，实测 SIMT 快 4.2x） |
| has_unknown_trip_count | true（动态 token 循环） |
| dot_ops / reduce_ops / gather_ops | 全 0 |
| mask_rank_sum | 4（很少） |
| max_tensor_numel | 256 |

完整 JSON：`triton-ascend-simit/test_cases/test_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.json`

---

## 二、关键 IR：index 是"算出来的"不是"load 出来的"

### 2.1 template 路径的 ttadapter（`silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.ttadapter`）

```mlir
// line 37: 动态 token 循环
scf.for %arg13 = %8 to %4 step %9 : i32 {
  %17 = linalg.fill ins(%arg13 : i32) ...            // loop_var 广播
  %18 = arith.addi %17, %11 : tensor<256xi32>        // = loop_var + arange(256)
  %19 = arith.divsi %18, %12 : tensor<256xi32>       // ← 除法：映射到 gate_up 行
  %20 = arith.cmpi slt, %18, %13 ...                 // mask
  %21 = arith.muli %19, %12 : tensor<256xi32>        // 乘回来
  %22 = arith.addi %18, %21 : tensor<256xi32>        // ← 取模模拟（%18 % %12）
  %23 = arith.extsi %22 : tensor<256xi32> → tensor<256xi64>
  %24 = func.call @triton_indirect_load(%arg2, %23, %20, %1)       // load 1（x）
  %25 = arith.addi %15, %23 : tensor<256xi64>        // gate_up 基址偏移 + 同一 index
  %26 = func.call @triton_indirect_load_0(%arg2, %25, %20, %1)     // load 2（gate_up）
  %27 = arith.subf %1, %24 ...                       // 1 - x
  %28 = math.exp %27 ...                             // sigmoid
  %30 = arith.divf %24, %29 ...                      // silu(x) = x * sigmoid(x)
  %31 = arith.mulf %30, %26 ...                      // * gate_up
  %32 = arith.mulf %31, %16 ...                      // * scale（量化）
  ... store
}
```

**地址链特征**：`loop_var + arange → divsi → muli → addi → extsi`。全程是算术运算，**没有任何 tt.load 参与地址计算**。但含 `divsi`（除法），是非仿射表达式。

### 2.2 为什么含除法

`gate_up` 是 2D tensor `[num_tokens, gate_up_size]`，kernel 用 1D 循环 + 256-wide tile 访问。把 1D 线性索引映射到 2D 需要 `div/mod`，Triton 前端就生成了 `divsi + muli + addi` 的取模模拟序列。这就是"非仿射"的来源。

---

## 三、两套判定口径的对比

### 3.1 Costmodel：`pointerDependsOnLoadedIndex`（SimtAnchorAnalysis.cpp:96-135）

```cpp
worklist = {load 的地址 operand(0)}
while (worklist 非空) {
    value = worklist.pop()
    if (value 是 scf.for 的 block arg) {
        if (argNumber > 0) {          // ← 只跟 iter_args
            把 init operand 和 yield operand 加入 worklist
        }
        continue;                     // ← 归纳变量（argNumber==0）不跟！
    }
    producer = value.getDefiningOp()
    if (producer 是 tt.load 或 tt.gather) return true;   // ← 命中条件
    worklist.append(producer 的全部 operands)
}
return false
```

**口径**：地址的 SSA 依赖链上**必须出现 `tt.load`/`tt.gather` 的产物**（即"从内存 load 出来的 index"），才算间接访存。

**对本 kernel**：地址链是 `loop_var(arg0) → addi → divsi → muli → addi → extsi`。loop_var 是归纳变量（argNumber==0）→ 被跳过不跟 → 链走到头也没遇到 tt.load → **返回 false** → 0 个 anchor。

**为什么刻意排除归纳变量**：如果跟了归纳变量，循环内所有地址含循环变量的 load 都会命中（几乎所有循环内 load 都含循环变量）→ anchor 爆炸性过度匹配 → 全 kernel 都变 SIMT。设计上是"宁缺毋滥"。

### 3.2 Template 路径：`isUnstructuredOrScalarlike`（UnstructureConversionPass.cpp:632）

```cpp
bool fullyUnstructured = ptrOffsetInfo.isUnstructuredOrScalarlike();
// SIMT 门（L524-537）：
simtTemplateLoadStoreFastPathEnabled =
    compileOn91095 && forceSimtTemplateFlag
    && ((非结构化 && sizeInByte < 64) || route_discrete_mask_to_simt)
    && rank <= 5;
```

**口径**：地址偏移**不是简单仿射表达式**（`base + stride × range` 形式）即视为非结构化，不管地址依赖的是 load 产物还是算术结果。

**对本 kernel**：偏移含 `divsi` → 非仿射 → unstructured → `ascend.indirect_load` → `@triton_indirect_load` 调用。2 个 load 全部命中。

### 3.3 对照表

| | Costmodel anchor | Template fast path |
|---|---|---|
| 代码位置 | `SimtAnchorAnalysis.cpp:96-135` | `UnstructureConversionPass.cpp:632, 524-537` |
| 判定依据 | 地址链含 `tt.load` 产物（**loaded index**） | 地址非仿射（含 div/mod 等，**computed index**） |
| 对 scf.for 归纳变量 | 不跟（防过度匹配） | 无此限制 |
| 本 kernel | false（0 anchor） | true（2 个 indirect_load） |
| 语义 | gather via memory-loaded index | 任意非规则地址 |

**两者交集之外的本 kernel**：index 由算术算出（computed index），无 loaded index → 只有 template 路径能抓到。

---

## 四、costmodel 表现的完整记录

### 4.1 Anchor 分析

```
recognized_anchor_count: 0
materializable_anchor_count: 0
mechanisms: []
applicability.reasons: ["no_recognized_simt_mechanism"]
mixed lowerability: unsupported（no_recognized_simt_anchor）
all_simt_only lowerability: native
```

### 4.2 打分（unknown loop 放行后）

| 候选 | raw | calibrated | 实测 |
|------|-----|-----------|------|
| all_simd | 507.04 | 507.04（multiplier=1） | 16.3 us |
| all_simt_only | 8190.71 | 8190.71（multiplier=1） | 3.9 us |
| mixed_simd_simt | 8413.71 | inapplicable | 3.7 us（legacy simd_simt 路径） |

- SIMD raw 只有 507：unknown trip count → loop multiplier=1 → 只算了一次迭代的 payload
- SIMT raw 8191 主要由 predicate 公式贡献（`simt_execution.predicate_system_cycles = 842.1`，占 10%；其余是 setup+payload）
- 决策错误：选了 SIMD（507 最低），实测 SIMD 最慢（16.3us）

### 4.3 已做的修复（`dynamic_loop_elementwise` domain）

kx_simt_costmodel 分支 commit `c4716ce30`：
- coverage：`dotFlops==0 && simtAnchors.count==0 && weightedReductions==0 && hasUnknownTripCount && maxNumel<=2048`
- multipliers（anchor 在 all_simt_only）：m_simd=67.52、m_simt=1.0、m_mixed=1.0
- 效果：校准后 SIMD=34,235 > SIMT=8,191 → 决策翻转为 all_simt_only → 走整核纯 SIMT（3.9us）

**但 mixed（3.7us）路线仍然不可达**——因为 costmodel 看不到任何 anchor，mixed 候选不存在。

---

## 五、待办：根本修复方向（未实施）

### 5.1 方案：扩展 anchor 识别支持 computed-index gather

在 `pointerDependsOnLoadedIndex` 之外增加有界检测：

```cpp
// 草案（未实现）：computed-index 检测
// 地址链从 scf.for 归纳变量出发，经过 divsi/remsi 等非仿射算术，
// 且没有 loaded index —— 也视为 SIMT anchor 候选
```

**必须谨慎设计的边界条件**：
1. **归纳变量跟随**：只跟"归纳变量 → divsi/remsi → 地址"这种链（含除法/取模的才是真 gather 模式），纯 `base + range × stride` 的仿射链不能算
2. **防过度匹配**：循环内普通连续 tile 访问（`offs = base + start_n + arange`）必须排除——它们地址链也含归纳变量但无 div/mod
3. **对其他 domain 的冲击**：anchor 计数变化会牵连 `attention_indirect_gqa`（`simtAnchors.loadedIndexDependentMemoryOps >= 2`）和 `indirect_elementwise`（`loadedIndexDependentMemoryOps > 0`）的 coverage 判定——ROPE/causal_conv1d 的分类可能受影响，需要回归验证
4. **scoring 影响**：新 anchor 进入 mixed partition 后，predicate/memory 成本计算会变

### 5.2 配套：mixed 路线的验证

本 kernel 若 mixed 复活（2 个 load 进 SIMT scope，其余 SIMD），需要在 A5 上验证 mixed 路线正确性 + 实测是否真是 3.7us。

---

## 附：相关文件索引

| 文件 | 位置 | 内容 |
|------|------|------|
| costmodel JSON | `triton-ascend-simit/test_cases/test_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.json` | features / scores / inapplicable mixed |
| ttadapter（template 路径） | `triton-ascend-simit/test_cases/silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.ttadapter` | 2 个 @triton_indirect_load + divsi 地址链 |
| kernel 源码 | SGLang `python/sglang/srt/layers/moe/ep_moe/kernels.py`（:433 起） | silu_mul + 静态 tensorwise quant |
| `pointerDependsOnLoadedIndex` | `third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimtAnchorAnalysis.cpp:96-135` | costmodel 口径（loaded index） |
| `isUnstructuredOrScalarlike` | `third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp:632` | template 口径（非仿射地址） |
| SIMT 门 | 同上 `:524-537` | 触发条件 |
| `dynamic_loop_elementwise` domain | kx_simt_costmodel `c4716ce30` + profile v10 | 已做的决策层修复 |
