# Generic Stage Fallback 设计与近期问题整理

本文整理最近关于以下内容的讨论：

- Triton-Ascend PR 1764 与 AscendNPU-IR PR 2933 的本地合入/编译/验证
- CostModel 当前三个 Phase 的覆盖范围
- 如何让未知算子快速进入 `stage_model`
- Generic Stage Fallback 的实现、占位 Stage 的替换流程
- `StageCostModelKind` 覆盖范围与后续优化方向

---

## 1. 背景：两个 PR 分别做什么

### AscendNPU-IR PR 2933

- 提交信息：`auto scope is working on solve_tril, and offer 12% performance improvement over simt_only now`
- 主要是 NPU-IR 后端能力：
  - 新增 `MaterializeSIMTScopeSuperBlock`
  - 新增 `RestoreScopeSuperBlockFactor`
  - 修改 `SplitSimtModule`、`InferSimtVFMemScopeHint`、`InferSimtVFMemEffect`、`TileAndBindSubBlock` 等
- 作用：让 TA CostModel 给局部 SIMT scope 选择的 SuperBlock factor 真正在后端物化生效。

### Triton-Ascend PR 1764

- 主要是 TA CostModel / route selection 侧：
  - Stage Route Model
  - 三个已支持的 Phase：
    - `triangular_recurrence`
    - `loaded_index_rowwise_reduction`
    - `indirect_underfilled_dot`
  - 对应三个典型测试：
    - `solve_tril`
    - `fbgemm_rowwise_quant`
    - `gather_dot_min`

---

## 2. 本地合入与编译

### AscendNPU-IR

```bash
git clone https://gitcode.com/Ascend/AscendNPU-IR.git
cd AscendNPU-IR

git checkout master
git pull origin master

git fetch origin refs/merge-requests/2933/head:pr-2933
git merge pr-2933 --no-edit

git submodule update --init --recursive
./build-tools/build.sh -o ./build --build-type Release
```

### Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend

git fetch origin refs/pull/1764/head:pr-1764
git checkout pr-1764
git submodule update --init --recursive
```

由于 TA 内嵌 `third_party/ascend/AscendNPU-IR` 是 submodule，需要把 PR 2933 也合入该 submodule。

---

## 3. 验证三个算子

PR 1764 中新增了三个 CostModel 用例：

```text
third_party/ascend/unittest/pytest_ut/test_simd_simt_costmodel_cases.py
```

运行：

```bash
pip install pytest-xdist

python -m pytest -s -v \
  third_party/ascend/unittest/pytest_ut/test_simd_simt_costmodel_cases.py \
  -k 'gather_dot_min or solve_tril or fbgemm_rowwise_quant'
```

三个用例分别验证：

- 数值正确性
- route decision 是否符合预期
- 实际耗时是否在参考范围内

注意：

- `worker_id fixture not found` 是因为缺少 `pytest-xdist`
- 这些用例只支持 `910_95` / `Ascend950` / `910_958B`

---

## 4. 当前三个 Phase 的覆盖边界

当前 Stage Route Model 只识别三种 Phase：

| Phase | 典型算子 |
|---|---|
| `triangular_recurrence` | `solve_tril` |
| `loaded_index_rowwise_reduction` | `fbgemm_rowwise_quant` |
| `indirect_underfilled_dot` | `gather_dot_min` |

如果一个 kernel 不满足这三种 Phase 的 feature 组合：

- 原来会返回：

```json
{
  "stage_model": { "applied": false },
  "unsupported": ["stage_model_not_applicable"]
}
```

- 不会产生 `all_simd` / `all_simt_only` / `mixed_simd_simt` 分数
- 编译走 `backend_default`

---

## 5. Generic Stage Fallback 做了什么

为了让目标算子尽快进入 `stage_model`，我们在 PR 1764 基础上增加了 Generic 兜底 Phase。

改动文件：

```text
third_party/ascend/costmodel/include/AscendModel/Analysis/StagePartitioner.h
third_party/ascend/costmodel/lib/AscendModel/Analysis/StagePartitioner.cpp
```

核心改动：

1. 新增 `PhaseBoundaryDomain::Generic`
2. `identifyPhaseBoundary()` 不再返回空，而是返回 Generic
3. 新增 `partitionGeneric()`
4. `StageBoundaryAnalysis` 支持 Generic
5. `StageKindClassifier` 对 Generic domain 强制 `derive()` 真实 StageCostModelKind
6. 暂时放宽 `requires_split`，允许混合结构先进 stage_model

Generic 分组优先级：

```text
dot > reduction > conversion > loop > indirect memory > memory > scalar
```

连续同类 root 会合并成一个 phase/stage。

---

## 6. partitionGeneric 为什么先放一个“假的 stage”

现有专用 partitioner 例如：

- `partitionTriangular`
- `partitionRowwise`
- `partitionIndirectDot`

它们一开始就填真实 `StageCostModelKind`，例如：

```cpp
addPhase(partition, "row_load",
         asLocalSIMT(makeStage(
             "indirect_row_gather",
             StageCostModelKind::IndirectGatherMemory,
             StageScheduleKind::PartiallyDependent, 1, {})));
```

Generic 不知道最终 kind，所以先放一个 `ScalarIssue` 占位，结构如下：

```cpp
phase.stages.push_back(makeStage(
    currentPhaseId,
    StageCostModelKind::ScalarIssue,
    StageScheduleKind::StraightLine,
    1,
    {}));
```

这个占位 stage 的以下字段是空的/假的：

- `operations` 为空
- `workload` 为空
- `features` 为默认值
- `costModelKind` 是占位 `ScalarIssue`
- `scheduleKind` 是占位 `StraightLine`
- live-in / live-out 为空
- local SIMT scope 相关字段为 0

后续通过以下流程逐步填真：

```text
partitionGeneric()
        ↓
attachCompleteOperationOwnership()
        ↓
StageWorkloadAnalysis()
        ↓
StageFeatureAnalysis()
        ↓
StageKindClassifier::analyze()
        ↓
deriveStageLiveValues()
        ↓
deriveLocalSimtScopeTraffic()
```

---

## 7. 四个关键函数说明

### attachCompleteOperationOwnership

- 把 `plan.rootOperations` 中的真实 TTIR root 挂到对应 Stage
- 保证每个 root 只属于一个 Stage
- 校验没有漏挂、没有重挂
- 完成后 `stage.operations` 是真实的

### attachExactAnchorOwnership

- 把 `SimtAnchorPlan` 中可 materialize 的 anchor 精确关联到 Stage
- 决定 Stage 是否真的可以成为 local SIMT scope
- 如果 Stage 没有匹配 anchor，会关闭 local SIMT materialization

### deriveStageLiveValues

- 根据 Stage 内 operation ownership 计算 SSA live-in / live-out
- 计算 `liveInBytes` / `liveOutBytes`
- 用于评估 Stage 边界数据搬运和 SIMD/SIMT 切换成本

### deriveLocalSimtScopeTraffic

- 只计算真正被 `scope.scope` 包住的那部分 tensor 搬运
- 不把整个 Stage 的 live-out 都算成切换开销
- 计算：
  - `localSimtScopeCount`
  - `scopeInputTensorBytes`
  - `scopeOutputTensorBytes`
- 如果 scope 返回 pointer，会关闭该 Stage 的 local SIMT materialization

---

## 8. 占位 ScalarIssue 会不会导致所有 Generic stage 都变成 scalar_issue？

会，如果不做处理的话。

原因：

```cpp
auto compatible = [](StageCostModelKind kind,
                     const StageModelFeatures &facts) {
  switch (kind) {
    ...
    default:
      return true; // ScalarIssue 对任何 feature 都“兼容”
  }
};
```

如果只写：

```cpp
if (!compatible(stage.costModelKind, facts))
  stage.costModelKind = derive();
```

那么 Generic 占位的 `ScalarIssue` 永远不会被替换。

修复方式：

```cpp
if (partition.domain == "generic" ||
    !compatible(stage.costModelKind, facts))
  stage.costModelKind = derive();
```

也就是 Generic domain 强制用 features/workload 推导真实 kind。

---

## 9. StageCostModelKind 是否都能被 Generic 覆盖？

不能。

`derive()` 当前只会产生：

```text
tiny_cube_roofline
cube_roofline
rowwise_reduction
conversion_pack
independent_pipelined_loop
loop_carried_recurrence
indirect_gather_memory
continuous_tile_memory
continuous_tile_store
scalar_issue
```

有 AutoBlockify V1 时还会有：

```text
auto_blockify_dispatch
auto_blockify_loop
```

以下 kind 目前 Generic 不会主动产生：

```text
scalar_control
scalar_math
index_generation
predicate_mask
loop_predicate
continuous_short_load
cache_policy_store
indirect_scalar_memory
```

这不是 Generic 的 bug，而是原有 `derive()` 本身就是粗粒度实现。

这些 StageCostModelKind 是完整的评分词汇表：

- `StageCostEvaluator` 已支持它们的评分
- `stringifyStageCostModel` 已支持输出
- 后续更细的切分规则可以直接使用

---

## 10. 后续优化方向

1. 先跑目标算子，看 `logical_stages` 里的 `model`
2. 根据实际切分结果，在 Generic 中细化分组规则
3. 逐步支持更多 StageCostModelKind：
   - 纯地址计算 → `index_generation`
   - predicate 相关 → `predicate_mask`
   - 循环 predicate → `loop_predicate`
   - 短连续 load → `continuous_short_load`
   - cache policy store → `cache_policy_store`
   - 标量 indirect memory → `indirect_scalar_memory`
4. 当某个算子稳定后，可以把它从 Generic 提升为专用 partitioner

---

## 11. 当前提交状态

- 基础分支：`pr/1764`
- 当前分支：`generic-stage-partition`
- 当前提交：`feat(costmodel): add generic stage fallback for unknown kernels`
- 工作区：`/private/tmp/ta-pr1764-dev`
