# 1D gather/scatter：hfusion 替换 ablation（fallback / keep_tt_load）

## 目标

验证 cost model 选出 `mixed_simd_simt` 并 materialize `scope.scope {vector_mode="simt"}` 之后，
如果 scope 内不再生成 `hfusion.gather_load/scatter_store`，性能是否仍与开启 hfusion 时一致。
如果一致，说明 **costmodel + 原生 tt.load 路径可以替代 hfusion 转换**。

两个 kernel：

- `gather_1d`（f16，BLOCK=8192）：间接 load
- `scatter_1d`（i8，BLOCK=8192）：间接 store

## hfusion 路径回顾

开启 hfusion 时（当前基线 wheel）：

```
scope.scope {vector_mode="simt"}
  └─ ascend.unstructured_load/store
       └─ hfusion.gather_load / hfusion.scatter_store
            │ HFusionToHIVM
            ▼
          hivm.hir.gather_load / scatter_store
            │ mixed pipeline: convert-hivm-to-tritongpu
            ▼
          tt.load / tt.store      （在 hivm.vf_mode<SIMT> 的 tt.func 内）
            │ SIMT/DPX lowering
            ▼
          最终 SIMT 代码
```

所以 hfusion 是一个中间载体：NPUIR 先做 HFusion/HIVM 的 dimension analysis、flatten、
bufferization、SIMT VF memory scope 等分析，再在 mixed 路径把 gather/scatter 换回
`tt.load/store` 交给 SIMT lowering。TTIR 层不会出现 `scf.for` fallback。

## 两个 ablation 模式

两个 patch 都只影响 `hasEnclosingVectorMode(op, "simt")` 的 op；plain（无 scope）路径不受影响。

### 1. fallback

patch：`patches/fallback_in_simt_scope.patch`

```diff
 bool useUnstructuredOp =
+    !scopeForcesSimt &&
     compileOn91095Flag &&
     (...)
```

- auto：`scope.scope` 仍在，但 scope 内不走 unstructured/hfusion，而是进入
  `UnstructuredMemAccessConverter` 的 legacy scalar-loop fallback：
  `scf.for + tensor.extract + 标量 tt.load/store + tensor.insert`。
- plain：无 scope，`scopeForcesSimt=false`，legacy 全局 hfusion 不变。
- 用途：测“SIMT VF + 提前标量化”的性能下限。

运行（需要重建 wheel）：

```bash
git apply third_party/ascend/hfusion_1d_ablation/patches/fallback_in_simt_scope.patch
# rebuild triton-ascend wheel
TRITON_HFUSION_ABLATION=fallback python -m pytest -q -s \
  third_party/ascend/hfusion_1d_ablation/test_hfusion_1d_ablation.py
```

预期 IR（auto）：
- `hfusion.*`：0
- `scope.scope`：1
- scope 内：`scf.for`，标量 `tt.load/store`

### 2. keep_tt_load

patch：`patches/keep_tt_load_in_simt_scope.patch`

```cpp
if (scopeForcesSimt) {
  // keep the original tt.load/tt.store for TritonToLinalg
  return failure();
}
```

- auto：`scope.scope` 仍在，原始 tile `tt.load/store` 原样保留，不生成 unstructured，
  也不生成 scalar fallback；由 `TritonToLinalg` 和 NPUIR 通用 lowering 处理。
- plain：同 fallback，legacy 全局 hfusion 不变。
- 用途：测“costmodel scope + 原生 tt.load”的性能，这才是“用原生 tt.load 替代 hfusion”
  的直接对照。

运行（需要重建 wheel）：

```bash
git apply third_party/ascend/hfusion_1d_ablation/patches/keep_tt_load_in_simt_scope.patch
# rebuild triton-ascend wheel
TRITON_HFUSION_ABLATION=keep_tt_load python -m pytest -q -s \
  third_party/ascend/hfusion_1d_ablation/test_hfusion_1d_ablation.py
```

预期 IR（auto）：
- `hfusion.*`：0
- `scope.scope`：1
- scope 内：原始 `tt.load` / `tt.store`，无 `scf.for` fallback

## 基线（hfusion 开启，已实测）

| kernel | 模式 | hfusion | scope | kernel median |
|---|---|---|---:|---:|---:|
| gather_1d f16 | auto（mixed F2） | `hfusion.gather_load` ×1 | 有 | 26.005 us |
| gather_1d f16 | plain（legacy 全局） | `hfusion.gather_load` ×1 | 无 | 13.687 us |
| scatter_1d i8 | auto（mixed F2） | `hfusion.scatter_store` ×1 | 有 | 19.846 us |
| scatter_1d i8 | plain（legacy 全局） | `hfusion.scatter_store` ×1 | 无 | 19.976 us |

性能测量与 `test_simd_simt_costmodel_cases.py::_assert_performance` 相同：
`torch_npu.profiler`，skip_first=5/warmup=3/active=20，取 `kernel_details.csv` 中位数。

## 运行（默认=不修改源码）

```bash
cd ~/triton-ascend
python -m pytest -q -s \
  third_party/ascend/hfusion_1d_ablation/test_hfusion_1d_ablation.py
```

单独跑 kernel：

```bash
PROBE=third_party/ascend/hfusion_1d_ablation/gs_1d_mix_probe.py
python $PROBE gather warmup 8192 f16 auto
python $PROBE gather run    8192 f16 auto
python $PROBE scatter run   8192 i8  auto
# fallback / keep_tt_load 模式使用已打 patch 并重编的 wheel：
python $PROBE gather run 8192 f16 fallback
python $PROBE gather run 8192 f16 keep_tt_load
```

pytest 每个 case 会把 `ttadapter` IR 单独写到
`/tmp/triton_hfusion_1d_ablation_ir/` 并打印路径（可用
`TRITON_HFUSION_ABLATION_IR_DIR` 改目录）：

- `gather_auto.ttadapter.mlir`（基线）
- `gather_auto_fallback.ttadapter.mlir`
- `gather_auto_keep_tt_load.ttadapter.mlir`
- plain 不受两个 patch 影响，文件名始终是 `*_plain.ttadapter.mlir`

## 文件

| 文件 | 说明 |
|---|---|
| `test_hfusion_1d_ablation.py` | pytest：auto/plain 两种路由 + `TRITON_HFUSION_ABLATION=on/fallback/keep_tt_load` 期望断言 + 数值 + profiler 性能 |
| `gs_1d_mix_probe.py` | 独立探针：`gather|scatter`、`warmup|run`、`auto|plain|fallback|keep_tt_load`、dtype、num_warps、`GS_TAG` |
| `patches/fallback_in_simt_scope.patch` | scope 内强制 `useUnstructuredOp=false`，走 scalar fallback |
| `patches/keep_tt_load_in_simt_scope.patch` | scope 内直接 `return failure()`，保留原始 `tt.load/store` |

## 待完成

1. 分别在主构建树应用两个 patch，各重编一次 triton-ascend wheel（不需要重编 ascendnpu-ir）；
2. 用 `TRITON_HFUSION_ABLATION=fallback` / `keep_tt_load` 跑 pytest，确认 IR token 与数值；
3. 用 profiler 测 median us，和基线对比：
   - `keep_tt_load` 若与基线持平/更好 → hfusion 转换可被 costmodel + 原生 tt.load 替代；
   - `fallback` 明显更差 → 说明 scope 里提前标量化代价高，但这不是最终对照；
4. 记录结果并决定是否保留 hfusion 转换。
