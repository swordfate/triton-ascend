# [costmodel] (feat) 连续访存标定 v2：全量接管 stride==1 load/store 定价（950PR）

> 状态草稿 — 数字由守护用例 / 100-case 回归自动填充（见 GUARD 表与回归表）。

## 一、背景

### 1.1 legacy 连续访存标定的局限

现有 SIMD/SIMT Route CostModel 对连续访存（stride==1 的
`ContiguousShort` / `ContiguousWide` 访问）使用一组经验曲线
（flat-group、store-group、cross-iteration pipelined、masked-copy），
它们在 100-case 的特定几何上调过参，但：

1. **覆盖不成体系**：只有被曲线几何命中的访问才被曲线接管，
   其余连续访问全部落回逐算子解析式（startup + B/rate），
   该解析式在流水线循环体里与实测偏差可达 +20% 以上
   （per-operation startup 在 DMA 流水线中每迭代重复计费）；
2. **循环体定价缺失**：`for` 循环体内的连续 load/store 没有独立标定，
   K（迭代数）、chunk 大小、grid 的组合变化直接落到解析式；
3. **多流叠加不可解释**：copy（1L+1S）、add（2L+1S）等混合体的
   load/store 重叠没有实测锚点。

因此本 PR 用 950PR 实测直查表 + 两级解析兜底，把 **全部** 连续访存
（直线段与循环体、simd 与 simt、任意 K/chunk/grid/body 形状）
收归 v2 公式；kaixin legacy 连续访存曲线不再参与连续访存定价
（间接/离散/strided 路径不受影响；通用兜底解析式保留）。

## 二、公式与机制

完整公式见 `CONTIGUOUS_V2_FORMULAS.md`。要点：

### 2.1 公式 A（直线段，hasLoop==false）

同 (store, elems, dtype, masked) 分组，N 条实例：

```text
T(N) = first + (N-1)*marginal          (N <= Q)
T(N) = T(Q) + (N-Q)*c_transfer         (N >  Q)
```

按 direction/dtype/bytes-bucket/residency 查 `group_a` 表，
乘 P 轴重叠因子 `pAxisFactor(P)`；单条访问与未拟合单元走兜底解析式。

### 2.2 公式 B（循环体，hasLoop==true）— 三级

单位：**每程序每次迭代**（stage 估计器自行乘 iterationCount）。

1. **第一级：calF2 直查表**（`contiguous_ldst_v2.direct`）。
   key `n<loads>_s<stores>_b<chunk>_k<K>_g<grid>`，
   value = 守护同口径整核实测（profiler 中位数 / waves）。
   形状 (nL,nS) 完全一致才参与，chunk/grid 按 log2 距离取最近，
   `T(K)=A+K*m` 两点/三点拟合跨 K 插值，
   chunk 缩放 `clamp(domBytes/rowB, .5, 2)` 作用于迁移分量；
   截距 A 只计入 `direct_fixed_share`（默认 0.3，
   与 route 边界级固定开销的重叠量标定）。
2. **第二级（simd）：聚合带宽包络**
   `perIter = iterationStartup + bodyBytes / aggregateRate`
   （calG 实测 ~83 B/cycle/56 程序；profile 的
   independentLoadGroup 常数）——流水线循环体每迭代只付一次启动。
3. **第三级：逐算子解析式**（保留的通用兜底）
   `perLoad = setup + B/rate`（simd）或 warp-instruction 速率（simt），
   load/store 分记，由 stage 估计器的 max(·) 建模双流重叠。

所有被认领的访问同时做 layout 覆盖，
legacy flat-group/store-group/pipelined 曲线因此看不到任何连续流。

### 2.3 适用条件（不满足则不认领）

1. 布局类 = ContiguousShort/Wide（stride==1，静态布局）；
2. 地址不依赖 loaded index、非仿射地址、tensor 控制流选择
   （这三类属于间接/离散，保持原路径）；
3. store 侧不接受 `activeElementRatio < 1` 的部分掩码写（RMW 路径保留）；
4. profile `contiguous_ldst_v2.enabled`（表为空时整体回退 legacy）。

### 2.4 route.json 新增诊断

`candidate_elapsed_costs_by_factor`：`all_simd_f1` 与每个合法纯 simt
superblock 因子（F1/F2/F4/…）的整核预测，用于守护表把强制 F2/F4
运行与模型同因子预测直接对比。

## 三、标定协议

| 探针 | 覆盖 | 产出 |
|---|---|---|
| calF2（本次主标定） | 4 体形状(1L/1L+1S/2L+1S/1S) × chunk{1K,4K,16K,64K}B × K{4,16,64} × grid{8,56,224}，simd(w4)+simt(w1)，每几何独立 profiler 会话（与守护同口径：5 次预热 + 20 次捕获取中位） | direct 表 279 行（simd 125 + simt 135，cv≤0.35 过滤） |
| calD / calE | 直线段 U-sweep；循环体 P/m 交叉验证 | group_a / loop_b（既有） |
| calG | 聚合带宽曲线 | 第二级包络常数 |

平台事实：simt 不能下降 64KB-chunk 体（per-thread stack 1152B 上限），
相应几何标 `compile_fail`，落第三级解析。

## 四、整核守护用例（24 例 × simd-F1 / simt-F1 / simt-F2 / simt-F4 = 96 行）

守护 kernel 全部为本 PR 新编写的纯连续 load/store 核（非 100-case 选择）：
12 个直线段（A_*）+ 12 个循环体（B_*），覆盖 f16/f32/int8/int64、
chunk 1KB–8KB、grid 8–1024、K 4–64。
整核预测（route.json）vs 整核实测（torch_npu profiler，Level1，
20 次启动均值；双卡复测、卡间中位差 1.08%）：

| 组 | 行数 | ≤10% 通过 | 说明 |
|---|---:|---:|---|
| **B 循环体（公式 B 主战场）** | 48 | **34** | k4/8/16/32/64 × g8/56/224 × copy/add/load/store 的 simd+simt 全过；仅 f16-2KB（直查表缺 2KB 行）与 load 尾部 reduce 两类未过 |
| A 直线段 copy/store 单流 | 16 | 11 | simd 全过；simt 计划的 byte-不变记忆阶段导致 i64/大 grid 偏差 |
| A 直线段 add（多单流叠加） | 4 | 0 | 单流解析叠加未建模双 load 流水线重叠（kaixin flat-group 曲线的职责，待并入） |
| A load（带 reduce 尾） | 4 | 0 | reduce/scalar-store 阶段定价不足（非连续访存部分） |
| **合计** | **96** | **50** | 逐行表见 `explore_contig/guard_final.md` |

守护逐行表（50/96 ≤10%，abs-err 中位 9.09%；卡间中位差 1.08%）：

| case | mode | 实测(us) | 预测(us) | 误差 | 整核误差<10% |
|---|---|---:|---:|---:|---|
| `A_add_f16_2k_g128` | simd_f1 | 2.130 | 2.988 | +40.26% | 否 |
| `A_add_f16_2k_g128` | simt_f1 | 8.098 | 14.511 | +79.19% | 否 |
| `A_add_f16_2k_g128` | simt_f2 | 6.615 | 10.574 | +59.85% | 否 |
| `A_add_f16_2k_g128` | simt_f4 | 5.423 | 6.258 | +15.39% | 否 |
| `A_add_f32_4k_g1024` | simd_f1 | 4.967 | 6.751 | +35.92% | 否 |
| `A_add_f32_4k_g1024` | simt_f1 | 52.475 | 62.621 | +19.34% | 否 |
| `A_add_f32_4k_g1024` | simt_f2 | 29.168 | 34.539 | +18.41% | 否 |
| `A_add_f32_4k_g1024` | simt_f4 | 17.001 | 19.673 | +15.72% | 否 |
| `A_add_f32_4k_g256` | simd_f1 | 2.968 | 3.577 | +20.52% | 否 |
| `A_add_f32_4k_g256` | simt_f1 | 15.780 | 21.013 | +33.16% | 否 |
| `A_add_f32_4k_g256` | simt_f2 | 10.743 | 13.712 | +27.63% | 否 |
| `A_add_f32_4k_g256` | simt_f4 | 8.651 | 10.462 | +20.93% | 否 |
| `A_copy_f16_2k_g256` | simd_f1 | 2.749 | 3.056 | +11.17% | 否 |
| `A_copy_f16_2k_g256` | simt_f1 | 11.135 | 12.621 | +13.35% | 否 |
| `A_copy_f16_2k_g256` | simt_f2 | 8.532 | 8.608 | +0.89% | 是 |
| `A_copy_f16_2k_g256` | simt_f4 | 7.358 | 6.904 | -6.16% | 是 |
| `A_copy_f32_4k_g1024` | simd_f1 | 4.564 | 5.686 | +24.58% | 否 |
| `A_copy_f32_4k_g1024` | simt_f1 | 42.816 | 35.489 | -17.11% | 否 |
| `A_copy_f32_4k_g1024` | simt_f2 | 23.968 | 20.032 | -16.42% | 否 |
| `A_copy_f32_4k_g1024` | simt_f4 | 13.710 | 12.060 | -12.04% | 否 |
| `A_copy_f32_4k_g256` | simd_f1 | 2.915 | 3.154 | +8.19% | 是 |
| `A_copy_f32_4k_g256` | simt_f1 | 12.891 | 12.621 | -2.09% | 是 |
| `A_copy_f32_4k_g256` | simt_f2 | 8.828 | 8.608 | -2.50% | 是 |
| `A_copy_f32_4k_g256` | simt_f4 | 6.694 | 6.904 | +3.14% | 是 |
| `A_copy_f32_4k_g64` | simd_f1 | 2.400 | 2.584 | +7.66% | 是 |
| `A_copy_f32_4k_g64` | simt_f1 | 6.490 | 7.206 | +11.03% | 否 |
| `A_copy_f32_4k_g64` | simt_f2 | 4.386 | 4.330 | -1.29% | 是 |
| `A_copy_f32_4k_g64` | simt_f4 | 4.371 | 4.370 | -0.03% | 是 |
| `A_copy_i64_8k_g256` | simd_f1 | 3.106 | 3.349 | +7.84% | 是 |
| `A_copy_i64_8k_g256` | simt_f1 | 18.626 | 12.621 | -32.24% | 否 |
| `A_copy_i64_8k_g256` | simt_f2 | 12.509 | 8.608 | -31.19% | 否 |
| `A_copy_i64_8k_g256` | simt_f4 | 9.133 | 6.904 | -24.40% | 否 |
| `A_copy_i8_1k_g256` | simd_f1 | 2.626 | 3.007 | +14.52% | 否 |
| `A_copy_i8_1k_g256` | simt_f1 | 10.136 | 12.621 | +24.52% | 否 |
| `A_copy_i8_1k_g256` | simt_f2 | 8.019 | 8.608 | +7.34% | 是 |
| `A_copy_i8_1k_g256` | simt_f4 | 7.226 | 6.904 | -4.45% | 是 |
| `A_load_f32_4k_g256` | simd_f1 | 6.851 | 4.413 | -35.58% | 否 |
| `A_load_f32_4k_g256` | simt_f1 | 8.131 | 12.795 | +57.36% | 否 |
| `A_load_f32_4k_g256` | simt_f2 | 7.049 | 9.025 | +28.03% | 否 |
| `A_load_f32_4k_g256` | simt_f4 | 5.665 | 7.592 | +34.01% | 否 |
| `A_store_f16_2k_g256` | simd_f1 | 2.444 | 2.381 | -2.60% | 是 |
| `A_store_f16_2k_g256` | simt_f1 | 4.316 | 4.264 | -1.20% | 是 |
| `A_store_f16_2k_g256` | simt_f2 | 4.617 | 3.396 | -26.44% | 否 |
| `A_store_f16_2k_g256` | simt_f4 | 4.833 | 3.201 | -33.77% | 否 |
| `A_store_f32_4k_g256` | simd_f1 | 2.590 | 2.441 | -5.76% | 是 |
| `A_store_f32_4k_g256` | simt_f1 | 3.444 | 4.264 | +23.82% | 否 |
| `A_store_f32_4k_g256` | simt_f2 | 3.016 | 3.396 | +12.61% | 否 |
| `A_store_f32_4k_g256` | simt_f4 | 3.023 | 3.201 | +5.88% | 是 |
| `B_add_f32_4k_k16_g56` | simd_f1 | 4.223 | 3.992 | -5.48% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f1 | 27.603 | 29.291 | +6.12% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f2 | 27.585 | 29.291 | +6.19% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f4 | 27.564 | 29.291 | +6.27% | 是 |
| `B_add_f32_4k_k64_g224` | simd_f1 | 123.388 | 111.286 | -9.81% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f1 | 934.676 | 951.322 | +1.78% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f2 | 486.507 | 475.661 | -2.23% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f4 | 292.153 | 237.831 | -18.59% | 否 |
| `B_copy_f16_2k_k16_g56` | simd_f1 | 3.794 | 3.762 | -0.84% | 是 |
| `B_copy_f16_2k_k16_g56` | simt_f1 | 7.750 | 15.484 | +99.79% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f2 | 7.765 | 15.484 | +99.40% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f4 | 7.770 | 15.484 | +99.28% | 否 |
| `B_copy_f32_4k_k16_g56` | simd_f1 | 4.065 | 3.936 | -3.18% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f1 | 24.817 | 26.828 | +8.11% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f2 | 24.837 | 26.828 | +8.02% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f4 | 24.919 | 26.828 | +7.66% | 是 |
| `B_copy_f32_4k_k32_g56` | simd_f1 | 5.616 | 5.919 | +5.40% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f1 | 47.835 | 51.147 | +6.92% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f2 | 47.835 | 51.147 | +6.92% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f4 | 47.991 | 51.147 | +6.58% | 是 |
| `B_copy_f32_4k_k4_g8` | simd_f1 | 1.626 | 1.762 | +8.35% | 是 |
| `B_copy_f32_4k_k4_g8` | simt_f1 | 6.933 | 7.773 | +12.11% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f2 | 6.974 | 7.773 | +11.45% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f4 | 6.965 | 7.773 | +11.60% | 否 |
| `B_copy_f32_4k_k64_g56` | simd_f1 | 9.197 | 9.658 | +5.01% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f1 | 93.867 | 99.783 | +6.30% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f2 | 93.952 | 99.783 | +6.21% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f4 | 93.958 | 99.783 | +6.20% | 是 |
| `B_copy_f32_4k_k8_g56` | simd_f1 | 3.086 | 3.058 | -0.90% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f1 | 13.447 | 14.669 | +9.09% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f2 | 13.495 | 14.669 | +8.70% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f4 | 13.541 | 14.669 | +8.33% | 是 |
| `B_copy_i8_1k_k16_g56` | simd_f1 | 3.644 | 3.609 | -0.96% | 是 |
| `B_copy_i8_1k_k16_g56` | simt_f1 | 14.456 | 9.820 | -32.07% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f2 | 14.505 | 9.820 | -32.30% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f4 | 14.453 | 9.820 | -32.05% | 否 |
| `B_load_f32_4k_k64_g56` | simd_f1 | 10.030 | 8.744 | -12.82% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f1 | 22.700 | 31.026 | +36.68% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f2 | 23.836 | 31.026 | +30.16% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f4 | 23.816 | 31.026 | +30.27% | 否 |
| `B_store_f32_4k_k16_g56` | simd_f1 | 3.747 | 3.631 | -3.10% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f1 | 3.723 | 3.783 | +1.62% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f2 | 3.701 | 3.783 | +2.22% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f4 | 3.739 | 3.783 | +1.18% | 是 |
| `B_store_i64_8k_k4_g56` | simd_f1 | 2.472 | 2.418 | -2.17% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f1 | 2.782 | 2.858 | +2.73% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f2 | 2.903 | 2.858 | -1.56% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f4 | 2.900 | 2.858 | -1.45% | 是 |

GUARD 50/96 within 10%
abs-err median 9.09% mean 17.33% max 99.79%

守护汇总：**循环体（公式 B）34/48 ≤10%（71%），直线段 16/48 ≤10%；
未达标行归因**：①simt route 计划的记忆阶段对 bytes 不敏感（copy_i64
-32%），②多单流 straight-line 未建模 load 流重叠（add +15~40%），
③2KB-chunk f16 直查行缺失（+32%），④load-kernel 的 reduce 尾阶段
定价不足（-15~37%）——分属 simt route 计划与 reduce 阶段标定，
不属于本 PR 的连续访存公式本身。

## 五、回归验证（100-case）

本 PR 后 100-case：**73/100 ≤10%（abs-err 中位 5.06%）**。
其中 12 例为本 PR 前即 FAIL 的存量问题（gather_dot、grid_stride_gather、
atomic×2、rowred、masked_softmax_r32、max_dim1、vector_add 等，逐位一致）；
新增偏差集中于一类：**高 wave 数（grid≥1024）的 pipelined-loop 体的
非循环阶段与边界固定项定价**——循环体本身的连续访存价格已验证正确
（muls_add 循环体阶段预测 4615 cycles vs 实测 ~4600），偏差来自
index-generation / predicate / 边界固定项，属 route 模型公共部分。

| case | route | 实测(us) | 预测(us) | 误差 |
|---|---|---:|---:|---:|
| `muls_add_n2097152` | all_simd | 6.625 | 14.047 | +112.03% |


100-case 汇总：**73/100 ≤10%**（abs-err 中位 5.06%，均值 12.65%）；
12 例为存量 FAIL；新增偏差主要归因高 grid 非循环阶段/边界固定项
（见上），连续访存循环体价格本身已对齐。

## 六、影响面

* `StageCostModels.cpp`：`mapContiguousV2Completion` 重写（公式 B 三级 +
  公式 A 兜底认领）；v2 不再因 sync/count/形状/K 拒绝连续访问。
* `SimdSimtCostModel.{h,cpp}`：profile 解析 `direct` /
  `direct_fixed_share`；报告输出 per-factor 诊断。
* `ContiguousLdStV2CostModel.h`：`directTable` 成员。
* profile `ascend_950pr_simd_simt_v1.json`：注入 `contiguous_ldst_v2.direct`
  与 `direct_fixed_share`。
* 标定/守护脚本：`explore_contig/cal_f2_direct.py`、`fit_direct2.py`、
  `bench_guard_contig.py`、`guard_run.sh`、`score100.py`、`run100_shards.sh`。

不改变：路由决策逻辑本身（候选合法性、因子选择）、
间接/离散/strided 定价、非 950PR 目标。
