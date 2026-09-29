# Strided load/store：route score 与整核实测对齐报告（2026-09-29）

## 0. 结论先行

1. **load 的“score 与实测差 3 倍”不是公式重复，而是测量 binary 形状不匹配。**
   - CostModel 的 analysis view 在 `ttir_layout_merge` 之后：AutoBlockify V1 F1 会把
     `grid=16 × BLOCK=64` coalesce 成 `tensor<16x64xf32>`，即一个 **1024-element**
     strided load。
   - forced `compile_mode="simd"` 的测量路径**不会跑** `_run_ttir_layout_merge`
     （layout merge 只在 `compile_mode="simd_simt"` 分支执行），所以 TTAdapter 里仍是
     `memref<64xf32, strided<[3]>>`，最终是 16 个 64-element program。
   - 走生产 `simd_simt` auto 路径后，最终 NPUIR 是
     `memref<16x64xf32, strided<[192,3]>> -> collapse_shape -> memref<1024xf32, strided<[3]>>`
     包在 `scf.for ... {autoblockify.subloop}` 里，和 analysis view 一致。
2. **whole-kernel score 与 profiler Duration 的差异主要是 kernel boundary，不是
   strided payload。**
   - 4 个 shape-matched route 点的 payload 误差本来是绝对量级正确的；
   - 加一个 A5 一次性的 `kernel_boundary_system_cycles = 830` 后，
     load SIMT / load SIMD / store SIMT / store SIMD 的整核 Event profiler 中位数误差分别约
     `+0.7% / -1.7% / +6.3% / -7.3%`，MAPE `4.01%`。
3. 代码已把 boundary / completion 放到 route envelope：
   - `StageTransitionCost` 增 `kernel_boundary_system_cycles` 与
     `all_simd / all_simt_f1 / all_simt_f2 / all_simt_f4 / mixed` completion 字段；
   - `solveStageRoutes` 在 wave scaling 和 all-SIMD schedule cleanup 之后，
     对每条 legal route **只加一次** boundary + completion；
   - report 新增 `payload_system_cycles` / `kernel_boundary_system_cycles` /
     `completion_system_cycles`，Stage 求和与 route total 不再混淆。
   - profile：`david-v100-simd-simt-20260929-v29-route-boundary`，schema 14。

## 1. 为什么 forced SIMD 形状会错

`ttir_to_linalg()` 中 layout merge 的入口是：

```python
if metadata.get("compile_mode") == "simd_simt" and opt.auto_simt_scope_mode != "off":
    _run_ttir_layout_merge(mod, metadata)
    ...
    analysis_ttir_code = _build_costmodel_analysis_ttir(mod, metadata, opt)
```

因此：

| 编译入口 | layout merge | AutoBlockify V1 | CostModel analysis view 的 load 形状 | 与实测 binary 一致？ |
|---|---|---|---|---|
| `compile_mode="simd"` (旧 forced SIMD 测法) | 否 | NPUIR V1 body 64 | —（没跑 CostModel） | × 只能测 64-element 版本 |
| `compile_mode="simd_simt", scope="auto"` | 是 | TA V1 + NPUIR V1 | `16x64` coalesced tile | ✓ 生产 route |
| `compile_mode="simd_simt"` + forcing profile 选 all_simd | 是 | 同生产 all-SIMD | `16x64` coalesced tile | ✓ shape-matched forced SIMD |

forced all_simd 的最终 NPUIR 关键行（`test_stride_load.py.py`，grid=16，W=32，stride=3）：

```text
scf.for %arg7 = %3 to %1 step %c56_i32 {
  %reinterpret_cast = memref.reinterpret_cast %arg2 ... sizes: [16, 64], strides: [192, 3]
  %collapse_shape = memref.collapse_shape [[0, 1]]
      : memref<16x64xf32, strided<[192, 3]>> into memref<1024xf32, strided<[3]>>
  hivm.hir.load ins(%collapse_shape : memref<1024xf32, strided<[3]>>) outs(%10 : memref<1024xf32, ub>)
}
```

所以结论是：**不要在整核对齐实验里直接 force `compile_mode="simd"`**。
要 force route，用 production `simd_simt` + 临时 forcing profile（只改变 route 选择，
不改被测 binary），或者后续增加一个 debug-only route force knob。

## 2. 测量协议和结果

测量使用 `torch_npu.profiler` 无内核插桩 `kernel_details.csv`；
每个配置重复 3 轮，每轮 30 次 launch，取每轮 `Duration(us)` 中位数，
下表 `实测` 用 3 个 run median 的中位数。

| case | 形状/W/grid | route | payload (SYS_CNT) | payload ns | +boundary (ns) | 实测 median (ns) | error |
|---|---|---:|---:|---:|---:|---:|---:|
| load | BLOCK=64, stride=3, W=32, grid=16 | selected all_simt | 897.489 | 907.6 | 1746.9 | 1734.0 | **+0.74%** |
| load | 同上 | forced all_simd | 2748.735 | 2779.6 | 3618.9 | 3681.0 | **−1.69%** |
| store | BLOCK=1024, stride=3, W=32, grid=2 | selected all_simt | 555.446 | 561.7 | 1401.0 | 1318.0 | **+6.30%** |
| store | 同上 | forced all_simd | 5514.551 | 5576.4 | 6415.8 | 6922.5 | **−7.32%** |

MAPE（4 个 route 点）= **4.01%**。

raw run medians：

```text
load  all_simt run medians (us): [1.775, 1.7285, 1.734]
load  all_simd run medians (us): [3.7045, 3.6745, 3.681]
store all_simt run medians (us): [1.318, 1.328, 1.3125]
store all_simd run medians (us): [6.9905, 6.9175, 6.9225]
```

boundary 来源：取 8 个 `(measured median - payload)` residual（4 route × 3 runs）的
中位数附近值，再按 4 条 route 的 MAPE 最小化，得到
`kernel_boundary_system_cycles = 830`（≈839.6 ns）。
它是 A5 + 当前 W32/grid 形状的 **低置信度** 常量，不是 solve_tril 的 `2500`。

## 3. 标定域内误差（沿用 calibration 数据，未重跑）

本轮没有重新做 calibration sweep；沿用已有 artifacts：

| 模型 | in-domain MAPE | block/strides |
|---|---:|---|
| load SIMD wide | 5.10% | block 8..512, stride 1/2 |
| load SIMD gather | 5.71% | block 8..512, stride 3..256 |
| load SIMT | 18.05% | block 32..256, strides 1..256 |
| store SIMD wide | 3.44% | block 8..512, stride 1 |
| store SIMD gather | 10.78% | block 32..256, stride 2..256 |
| store SIMT | 13.68% | block 32..256, strides 1..256 |

注意：本次 test kernel 的 store 是 BLOCK=1024，超过 store calibration 的 BLOCK≤512，
因此 store 的 `−7.32%` 是外推；load BLOCK=64 在域内。
后续 step 5 应在 BLOCK=256/512 做一次 store in-domain 整核验证。

## 4. 代码修改

已改（本地 branch `feature/strided-load-store-costmodel`）：

- `StageRouteCostModel.h/.cpp`
  - `StageTransitionCost`：boundary + route/factor completion 字段；
  - `StageRoutePlan`：`payloadCycles / boundaryCycles / completionCycles`；
  - `solveStageRoutes`：wave scaling 后只加一次 envelope；
  - report JSON 显式分开三者。
- `SimdSimtCostModel.cpp`
  - schema version 13→14；
  - 读取 root `route_boundary`；缺省为 0，兼容旧 profile。
- profile
  - `david-v100-simd-simt-20260929-v29-route-boundary`，schema 14；
  - `kernel_boundary_system_cycles = 830`；
  - `completion_system_cycles` 字段已建立，当前 strided test 都是 F1：
    F1 drain 已经含在边界里，因此值填 0；F2/F4 没有专门 probe，故仍为 0，
    不会制造假的 completion 费用。

## 5. 剩余工作

1. **F2/F4 completion/drain**：本轮 test 都是 F1。需要按 solve_tril 口径，
   对同 route、同 factor 做 in-kernel SYS_CNT/CLOCK64 phase 打点，标定
   `all_simt_f2_completion_system_cycles` / `all_simt_f4_completion_system_cycles`。
2. **in-domain 整核验证**：load BLOCK=256/512、store BLOCK=256/512，stride=3、W=32
   的 shape-matched selected/forced route，对照 calibration domain。
3. **debug-only route force knob**：建议后续加一个测试专用选项，
   避免为了 force route 而复制 profile；生产路径不变。
4. 继续保留 `strided*` 命名；runtime stride / block_ptr / mask tail / 非 4B
   dtype 另开扩展。

## 6. 反例：单常数 boundary 对小 tile 不成立（本轮新增）

按 step 5 又补了 `n=256` 的小 shape（同一 kernel、W=32、stride=3、
selected route 都是 all_simt），结果如下：

| case | block/grid | payload (SYS_CNT) | payload ns | +830 pred ns | 实测 median ns | error |
|---|---|---:|---:|---:|---:|---:|
| load | 32 / 8 | 1310.549 | 1325.3 | 2164.9 | 1090.0 | **+98.6%** |
| load | 64 / 4 | 1310.549 | 1325.3 | 2164.9 | 1109.5 | **+95.1%** |
| load | 128 / 2 | 1310.549 | 1325.3 | 2164.9 | 1017.0 | **+112.9%** |
| store | 64 / 4 | 1038.519 | 1050.2 | 1889.7 | 1190.5 | **+58.7%** |
| store | 128 / 2 | 1061.648 | 1073.5 | 1913.0 | 1158.5 | **+65.1%** |
| store | 256 / 1 | 1107.905 | 1120.3 | 1959.8 | 1142.5 | **+71.6%** |

解释：这些小 shape 的 `payload` 本身已经接近/高于整核实测，
再加 830 常数会明显过校正。原因不是 strided 公式，而是：

- 当前 route plan 是 Stage serial sum；小 grid 时 `auto_blockify` setup /
  dispatch / scalar prologue 在模型中是加法，但实际 binary 里和 GM 访存有
  重叠或没有完全 materialize；
- 830 这个值来自 n=1024/2048 的 4 个 target point，本质上吸收了这些点的
  `actual - payload`，不能外推到所有 grid。

因此结论必须写清楚：

1. `830` 只作为 target shape（load n=1024，store n=2048）的首版 whole-kernel
   校正，不能当全局常数；
2. 下一步必须按 solve_tril 口径做 in-kernel SYS_CNT / CLOCK64 phase 打点，
   把 kernel boundary 与 SIMT completion 分开，并至少建立 grid/wave 依赖；
3. 在完成 phase 标定前，profile 里的 boundary 只应用于目标形状或标记为
   low-confidence，不应据此宣称所有 strided shape 绝对 ns 已对齐。

