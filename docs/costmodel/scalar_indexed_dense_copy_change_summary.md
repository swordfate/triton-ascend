# ScalarIndexedDenseCopy 改动总结

> 本文总结针对 `_padded_copy_gather` / `_padded_copy_scatter` /
> `_binned_copy_gather` / `_binned_copy_scatter` 四个算子的 costmodel 改动。
> 目标：让这四个 scalar-indexed dense copy 类算子进入 Stage 模型，并支持
> SIMT / mixed 路线。

---

## 1. 背景

这四个算子的共同结构是：

```text
scalar/index/bin 解析
  -> 分支 / 指针偏移
  -> 连续 vector load / convert / scale / store
```

实测在昇腾 A5 上：

| 算子 | SIMD (us) | SIMT (us) | 更优 |
|---|---|---|---|
| `_padded_copy_gather` | 79.084 | 70.288 | SIMT |
| `_padded_copy_scatter` | 107.739 | 68.73 | SIMT |
| `_binned_copy_gather` | 95.587 | 69.294 | SIMT |
| `_binned_copy_scatter` | 55.914 | 50.85 | SIMT |

改动前 report 显示：

```json
"stage_model": {
  "applied": false
}
```

也就是没有匹配任何 Phase，走了 fallback aggregate 评分，导致错误地选择 all_simd。

---

## 2. 改动内容

### 2.1 新增 Phase

```cpp
enum class PhaseBoundaryDomain {
  ...
  ScalarIndexedDenseCopy,
};
```

识别条件：

```cpp
features.dotOps == 0 &&
features.reduceOps == 0 &&
features.scanOps == 0 &&
features.loadOps > 0 &&
features.storeOps > 0 &&
features.loadedIndexDependentMemoryOps > 0 &&
features.scalarLoadOps > 0
```

### 2.2 新增 Stage 划分

```text
ScalarIndexedDenseCopy
├── scalar_index_setup     -> IndexGeneration
└── dense_tile_copy        -> ContinuousTileMemory
```

两个 Stage 都标记为 `asLocalSIMT`，允许参与 mixed 路线。

### 2.3 新增 ScalarIndexSetup anchor

新增：

```cpp
SimtAnchorKind::ScalarIndexSetup
```

`tryBuildScalarIndexSetupAnchor` 从顶层 `scf.for` 反推前面的 scalar setup 前缀，将其作为一个 anchor：

```text
scopeOperations = block 开头到循环之前的所有顶层 op
scopeInsertionPoint = block 开头
```

dense copy 仍由原有 `LoadedIndexDependentMemory` anchor 支持。

### 2.4 避免影响已有算子

`tryBuildScalarIndexSetupAnchor` 增加限制：

- 必须是 top-level `scf.for` / `scf.while`；
- 循环内必须有 tensor load/store；
- 循环内不能有 `tt.dot` / `tt.reduce` / `tt.scan`；
- 必须有 loaded-index dependent tensor memory；
- block 中循环前必须有 scalar load。

因此不会影响：

- TriangularRecurrence
- LoadedIndexRowwiseReduction
- IndirectUnderfilledDot
- PlainOneDimensionalCumsum

---

## 3. 当前效果

改动后这四个算子应进入：

```json
"stage_model": {
  "applied": true,
  "domain": "scalar_indexed_dense_copy"
}
```

并且 report 中应能看到：

```text
all_simt_only / mixed 分数优于 all_simd
```

与实测 SIMT 更快一致。

---

## 4. 后续可做的实验

为了确认“只把第一个 Stage 放 SIMT”或“只把第二个 Stage 放 SIMT”是否更快，
可以用 Python 手写 scope 做 A/B：

```python
import triton.language.extra.cann.extension as al

# 只把 scalar setup 放 SIMT
with al.scope(vector_mode="simt"):
    # scalar/index/bin 解析、分支、指针偏移
    ...

# 只把 dense copy 放 SIMT
with al.scope(vector_mode="simt"):
    # for _ in range(iterations):
    #     vector load / convert / scale / store
    ...
```

然后分别跑 profiler 对比：

- 全 SIMD baseline
- 全 SIMT baseline
- 仅 scalar_index_setup SIMT
- 仅 dense_tile_copy SIMT
- costmodel 推荐的 simt_only / mixed

---

## 5. 相关代码位置

| 内容 | 位置 |
|---|---|
| Phase 枚举 | `StagePartitioner.h` |
| Phase 识别 | `StagePartitioner.cpp` `PhaseBoundaryAnalysis::analyze` |
| Stage 切分 | `StagePartitioner.cpp` `partitionScalarIndexedDenseCopy` |
| anchor 识别 | `SimtAnchorAnalysis.cpp` `tryBuildScalarIndexSetupAnchor` |
| anchor 与 Stage 匹配 | `StagePartitioner.cpp` `anchorMatchesStage` |
