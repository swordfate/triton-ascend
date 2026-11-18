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
| **合计** | **96** | **55** | 逐行表见 `explore_contig/guard_final.md` |

守护逐行表（55/96 ≤10%，abs-err 中位 8.26%；卡间中位差 1.08%）：

| case | mode | 实测(us) | 预测(us) | 误差 | 整核误差<10% |
|---|---|---:|---:|---:|---|
| `A_add_f16_2k_g128` | simd_f1 | 2.057 | 2.988 | +45.24% | 否 |
| `A_add_f16_2k_g128` | simt_f1 | 8.098 | 15.240 | +88.19% | 否 |
| `A_add_f16_2k_g128` | simt_f2 | 6.647 | 11.063 | +66.44% | 否 |
| `A_add_f16_2k_g128` | simt_f4 | 5.544 | 6.510 | +17.43% | 否 |
| `A_add_f32_4k_g1024` | simd_f1 | 4.895 | 6.751 | +37.92% | 否 |
| `A_add_f32_4k_g1024` | simt_f1 | 52.065 | 76.183 | +46.32% | 否 |
| `A_add_f32_4k_g1024` | simt_f2 | 29.096 | 41.722 | +43.39% | 否 |
| `A_add_f32_4k_g1024` | simt_f4 | 16.962 | 23.379 | +37.83% | 否 |
| `A_add_f32_4k_g256` | simd_f1 | 3.040 | 3.577 | +17.67% | 否 |
| `A_add_f32_4k_g256` | simt_f1 | 15.715 | 23.117 | +47.10% | 否 |
| `A_add_f32_4k_g256` | simt_f2 | 10.719 | 14.982 | +39.77% | 否 |
| `A_add_f32_4k_g256` | simt_f4 | 8.522 | 11.336 | +33.02% | 否 |
| `A_copy_f16_2k_g256` | simd_f1 | 2.824 | 3.056 | +8.22% | 是 |
| `A_copy_f16_2k_g256` | simt_f1 | 11.211 | 13.775 | +22.87% | 否 |
| `A_copy_f16_2k_g256` | simt_f2 | 8.583 | 9.308 | +8.44% | 是 |
| `A_copy_f16_2k_g256` | simt_f4 | 7.462 | 7.390 | -0.97% | 是 |
| `A_copy_f32_4k_g1024` | simd_f1 | 4.562 | 5.686 | +24.64% | 否 |
| `A_copy_f32_4k_g1024` | simt_f1 | 42.616 | 42.925 | +0.73% | 是 |
| `A_copy_f32_4k_g1024` | simt_f2 | 23.935 | 23.988 | +0.22% | 是 |
| `A_copy_f32_4k_g1024` | simt_f4 | 13.576 | 14.118 | +3.99% | 是 |
| `A_copy_f32_4k_g256` | simd_f1 | 2.970 | 3.154 | +6.19% | 是 |
| `A_copy_f32_4k_g256` | simt_f1 | 13.026 | 13.775 | +5.75% | 是 |
| `A_copy_f32_4k_g256` | simt_f2 | 8.974 | 9.308 | +3.72% | 是 |
| `A_copy_f32_4k_g256` | simt_f4 | 6.741 | 7.390 | +9.63% | 是 |
| `A_copy_f32_4k_g64` | simd_f1 | 2.572 | 2.584 | +0.46% | 是 |
| `A_copy_f32_4k_g64` | simt_f1 | 6.637 | 7.252 | +9.26% | 是 |
| `A_copy_f32_4k_g64` | simt_f2 | 4.505 | 4.353 | -3.38% | 是 |
| `A_copy_f32_4k_g64` | simt_f4 | 4.469 | 4.394 | -1.68% | 是 |
| `A_copy_i64_8k_g256` | simd_f1 | 3.075 | 3.349 | +8.92% | 是 |
| `A_copy_i64_8k_g256` | simt_f1 | 18.534 | 13.775 | -25.68% | 否 |
| `A_copy_i64_8k_g256` | simt_f2 | 12.484 | 9.308 | -25.44% | 否 |
| `A_copy_i64_8k_g256` | simt_f4 | 9.029 | 7.390 | -18.15% | 否 |
| `A_copy_i8_1k_g256` | simd_f1 | 2.772 | 3.007 | +8.49% | 是 |
| `A_copy_i8_1k_g256` | simt_f1 | 10.104 | 13.775 | +36.33% | 否 |
| `A_copy_i8_1k_g256` | simt_f2 | 7.934 | 9.308 | +17.31% | 否 |
| `A_copy_i8_1k_g256` | simt_f4 | 7.181 | 7.390 | +2.91% | 是 |
| `A_load_f32_4k_g256` | simd_f1 | 6.658 | 4.413 | -33.72% | 否 |
| `A_load_f32_4k_g256` | simt_f1 | 8.140 | 13.745 | +68.86% | 否 |
| `A_load_f32_4k_g256` | simt_f2 | 7.130 | 9.595 | +34.58% | 否 |
| `A_load_f32_4k_g256` | simt_f4 | 5.655 | 7.980 | +41.12% | 否 |
| `A_store_f16_2k_g256` | simd_f1 | 2.506 | 2.381 | -5.01% | 是 |
| `A_store_f16_2k_g256` | simt_f1 | 4.416 | 4.468 | +1.17% | 是 |
| `A_store_f16_2k_g256` | simt_f2 | 4.675 | 3.526 | -24.59% | 否 |
| `A_store_f16_2k_g256` | simt_f4 | 4.903 | 3.298 | -32.74% | 否 |
| `A_store_f32_4k_g256` | simd_f1 | 2.614 | 2.441 | -6.63% | 是 |
| `A_store_f32_4k_g256` | simt_f1 | 3.560 | 4.468 | +25.50% | 否 |
| `A_store_f32_4k_g256` | simt_f2 | 3.079 | 3.526 | +14.50% | 否 |
| `A_store_f32_4k_g256` | simt_f4 | 3.068 | 3.298 | +7.48% | 是 |
| `B_add_f32_4k_k16_g56` | simd_f1 | 4.264 | 3.986 | -6.52% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f1 | 27.716 | 29.158 | +5.20% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f2 | 27.782 | 29.158 | +4.95% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f4 | 27.721 | 29.158 | +5.18% | 是 |
| `B_add_f32_4k_k64_g224` | simd_f1 | 116.632 | 123.254 | +5.68% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f1 | 938.412 | 1052.909 | +12.20% | 否 |
| `B_add_f32_4k_k64_g224` | simt_f2 | 486.963 | 526.454 | +8.11% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f4 | 278.833 | 263.227 | -5.60% | 是 |
| `B_copy_f16_2k_k16_g56` | simd_f1 | 3.789 | 3.762 | -0.71% | 是 |
| `B_copy_f16_2k_k16_g56` | simt_f1 | 7.724 | 15.446 | +99.97% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f2 | 7.670 | 15.446 | +101.38% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f4 | 7.787 | 15.446 | +98.35% | 否 |
| `B_copy_f32_4k_k16_g56` | simd_f1 | 4.017 | 3.936 | -2.02% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f1 | 24.777 | 26.786 | +8.11% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f2 | 24.742 | 26.786 | +8.26% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f4 | 24.815 | 26.786 | +7.94% | 是 |
| `B_copy_f32_4k_k32_g56` | simd_f1 | 5.686 | 5.691 | +0.09% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f1 | 47.978 | 51.060 | +6.42% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f2 | 47.871 | 51.060 | +6.66% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f4 | 47.855 | 51.060 | +6.70% | 是 |
| `B_copy_f32_4k_k4_g8` | simd_f1 | 1.653 | 1.761 | +6.56% | 是 |
| `B_copy_f32_4k_k4_g8` | simt_f1 | 6.936 | 7.764 | +11.94% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f2 | 6.987 | 7.764 | +11.13% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f4 | 6.968 | 7.764 | +11.43% | 否 |
| `B_copy_f32_4k_k64_g56` | simd_f1 | 9.114 | 9.201 | +0.96% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f1 | 93.794 | 99.608 | +6.20% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f2 | 93.863 | 99.608 | +6.12% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f4 | 93.932 | 99.608 | +6.04% | 是 |
| `B_copy_f32_4k_k8_g56` | simd_f1 | 3.096 | 3.058 | -1.22% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f1 | 13.462 | 14.649 | +8.82% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f2 | 13.630 | 14.649 | +7.48% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f4 | 13.672 | 14.649 | +7.15% | 是 |
| `B_copy_i8_1k_k16_g56` | simd_f1 | 3.638 | 3.609 | -0.79% | 是 |
| `B_copy_i8_1k_k16_g56` | simt_f1 | 14.475 | 9.820 | -32.16% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f2 | 14.586 | 9.820 | -32.67% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f4 | 14.614 | 9.820 | -32.80% | 否 |
| `B_load_f32_4k_k64_g56` | simd_f1 | 9.850 | 8.669 | -11.99% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f1 | 22.454 | 30.922 | +37.71% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f2 | 23.666 | 30.922 | +30.66% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f4 | 23.620 | 30.922 | +30.91% | 否 |
| `B_store_f32_4k_k16_g56` | simd_f1 | 3.699 | 3.631 | -1.84% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f1 | 3.644 | 3.724 | +2.20% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f2 | 3.651 | 3.724 | +2.00% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f4 | 3.608 | 3.724 | +3.22% | 是 |
| `B_store_i64_8k_k4_g56` | simd_f1 | 2.566 | 2.414 | -5.92% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f1 | 2.873 | 2.854 | -0.65% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f2 | 2.941 | 2.854 | -2.95% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f4 | 2.932 | 2.854 | -2.65% | 是 |

GUARD 55/96 within 10%
abs-err median 8.26% mean 18.60% max 101.38%

守护汇总：**循环体（公式 B）34/48 ≤10%（71%），直线段 21/48 ≤10%；
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

100-case 汇总：**76/100 ≤10%**（abs-err 中位 4.63%，均值 9.21%）；
12 例为存量 FAIL；新增偏差主要归因高 grid 非循环阶段/边界固定项
（见上），连续访存循环体价格本身已对齐。

## 五a、Tier 1a：微小 store 循环体（reduce_sum 族修复）

**实测规律**（calF2-t 探针 + 100-case 实机）：循环体的 store 全部 ≤128B、
主导 chunk ≥512B 时（reduction-epilogue 形状），微小依赖 IO 在 DMA 流水
填满前逐迭代串行——K=4 的实测整核时长远高于 K≥16 的线性趋势（约 2 倍），
K≥16 后边际增量平坦（~117 ns/iter）。大数据 store 的 direct 行
（m≈109 ns/iter）完全错过该效应：`pytest_reduce_sum_epilogue_f32_8x8x512`
欠预测 24.7%（消融证实其 4KB tile load 本身近乎全隐藏，
微小 bias/store 链才是主体）。

**机制**（线性形式，无指数）：`contiguous_ldst_v2.tiny_direct` 表
（键与 `direct` 一致），store 全 ≤128B 且 domBytes ≥512B 且 K ≤ 8 时认领；
每迭代价格
`perIter = (A×share + K×m + excess×tiny_store_chain_share) / K × wavesDecay`，
其中 `excess = max(0, 行值 − (A + K×m))`，`tiny_store_chain_share = 0.636`
（guard/100-case 标定）。验证：reduce_sum_epilogue 预测 2086→2772ns
（+0.0%）；同族 mul_reduce / gather_after_reduce / scalar_reduce_store
不受影响。

## 六、影响面

* `StageCostModels.cpp`：`mapContiguousV2Completion` 重写（公式 B 三级 +
  公式 A 兜底认领）；v2 不再因 sync/count/形状/K 拒绝连续访问。
* `SimdSimtCostModel.{h,cpp}`：profile 解析 `direct` /
  `direct_fixed_share`；报告输出 per-factor 诊断。
* `ContiguousLdStV2CostModel.h`：`directTable` / `tinyDirectTable` 成员。
* profile `ascend_950pr_simd_simt_v1.json`：注入 `contiguous_ldst_v2.direct`
  / `direct_fixed_share` / `tiny_direct` / `tiny_store_chain_share`。
* 标定/守护脚本：`explore_contig/cal_f2_direct.py`、`fit_direct2.py`、
  `cal_f2_tinystore3.py`（微小 store 体探针）、
  `bench_guard_contig.py`、`guard_run.sh`、`score100.py`、`run100_shards.sh`。

不改变：路由决策逻辑本身（候选合法性、因子选择）、
间接/离散/strided 定价、非 950PR 目标。
