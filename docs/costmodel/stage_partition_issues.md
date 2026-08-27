# Stage 划分不合理问题汇总

本文汇总当前 costmodel 在 FBGEMM 目标算子上暴露出的 stage 划分问题。

## 1. `_kernel_silu_quantize_mx4_unpack_npu`

### 现象

- 只切出 4 个 stage：
  - `auto_blockify_dispatch`
  - `generic_setup` / `scalar_issue`
  - `generic_store` / `continuous_tile_store`
- 日志显示：

```text
[92..92] generic_store (1 ops: scf.for)
```

### 问题

- 整个内层 `scf.for` 被当成 1 个 stage；
- 该循环内部实际包含：
  - load
  - silu / exp / mul
  - `tt.reduce`
  - MX4 打包
  - store
  - offset 更新
- 当前用 `continuous_tile_store` 一个模型拟合整段复杂计算，明显不合理。

---

## 2. `_rope_padded_kernel`

### 现象

- 有 10 个 load、6 个 store、12 个 anchor；
- 但最终只切出 `auto_blockify_dispatch` 两个 stage；
- 业务逻辑全被包在 `scf.execute_region` 等结构化 root 中，没有展开。

### 问题

- 业务 stage 完全缺失；
- load / store / anchor 都没有单独建模；
- 导致 all-SIMD 成本被错误地算成 0。

---

## 3. `fused_padding_cumsum_and_segmented_arange_kernel`

### 现象

- 切出 5 个 phase：
  - `generic_setup` / `scalar_issue`
  - `generic_anchor` / `indirect_gather_memory`
  - `generic_tail_setup` / `scalar_issue`
  - `generic_tail_store` / `continuous_tile_store`

### 合理点

- setup 独立；
- binary search 的 indirect 访存被识别；
- tail store 被分离。

### 不合理点

- `generic_anchor` 混合了：
  - `tl.cumsum`（scan/reduction）
  - 32 次 binary search（indirect gather）
  - `pid==0` 分支里的 store
- 最终统一标成 `indirect_gather_memory`，丢失 cumsum 和 store 语义；
- 缺少独立 load stage；
- Part 1 的 store 被藏在 anchor 里。

---

## 4. `fused_single_block_kernel`

### 现象

- 切出 4 个 phase：
  - `generic_setup` / `scalar_issue`
  - `generic_load` / `continuous_tile_memory`
  - `generic_anchor` / `indirect_gather_memory`

### 合理点

- load 被单独切出；
- cumsum + binary search 被识别为 anchor。

### 不合理点

- 完全没有 store stage；
- 源码中有 6 个 store，但都被算进 anchor / setup；
- anchor 内部混合：
  - `tl.cumsum`
  - 64 次 binary search
  - 多次 store
- 只用 `indirect_gather_memory` 无法表达这些混合语义；
- load 只切出 1 个，其余 3 个 load 都藏在 anchor 内。

---

## 5. 共性结论

当前最主要的问题不是公式参数，而是：

```text
结构化 op（scf.for / scf.execute_region）没有被展开，
导致一个 stage 里塞入了多种 StageCostModelKind 语义。
```

后续优化方向：

1. 对普通 `scf.for` 做 loop-shell flatten，把循环体展开成独立 root；
2. 对 `scf.execute_region` 等结构化 root 继续下钻；
3. 让 role machine 能把 load / reduce / convert / store 分开；
4. 再针对每个细分 stage 用 camodel 校准。
