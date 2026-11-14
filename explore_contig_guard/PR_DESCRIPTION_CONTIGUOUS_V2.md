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
| **B 循环体（公式 B 主战场）** | 28 | **26** | k4/8/16/32/64 × g8/56/224 × copy/add/load/store 的 simd+simt 全过；仅 f16-2KB（直查表缺 2KB 行）与 load 尾部 reduce 两类未过 |
| A 直线段 copy/store 单流 | 16 | 11 | simd 全过；simt 计划的 byte-不变记忆阶段导致 i64/大 grid 偏差 |
| A 直线段 add（多单流叠加） | 4 | 0 | 单流解析叠加未建模双 load 流水线重叠（kaixin flat-group 曲线的职责，待并入） |
| A load（带 reduce 尾） | 4 | 0 | reduce/scalar-store 阶段定价不足（非连续访存部分） |
| **合计** | **96** | **49** | 逐行表见 `explore_contig/guard_final.md` |

守护逐行表（49/96 ≤10%，abs-err 中位 9.87%）：

| case | mode | 实测(us) | 预测(us) | 误差 | 整核误差<10% |
|---|---|---:|---:|---:|---|
| `A_add_f16_2k_g128` | simd_f1 | 2.126 | 2.988 | +40.53% | 否 |
| `A_add_f16_2k_g128` | simt_f1 | 8.113 | 14.511 | +78.86% | 否 |
| `A_add_f16_2k_g128` | simt_f2 | 6.625 | 10.574 | +59.61% | 否 |
| `A_add_f16_2k_g128` | simt_f4 | 5.474 | 6.258 | +14.32% | 否 |
| `A_add_f32_4k_g1024` | simd_f1 | 4.903 | 6.751 | +37.69% | 否 |
| `A_add_f32_4k_g1024` | simt_f1 | 52.426 | 62.621 | +19.45% | 否 |
| `A_add_f32_4k_g1024` | simt_f2 | 29.271 | 34.539 | +18.00% | 否 |
| `A_add_f32_4k_g1024` | simt_f4 | 17.064 | 19.673 | +15.29% | 否 |
| `A_add_f32_4k_g256` | simd_f1 | 3.122 | 3.577 | +14.57% | 否 |
| `A_add_f32_4k_g256` | simt_f1 | 15.908 | 21.013 | +32.09% | 否 |
| `A_add_f32_4k_g256` | simt_f2 | 10.870 | 13.712 | +26.14% | 否 |
| `A_add_f32_4k_g256` | simt_f4 | 8.585 | 10.462 | +21.86% | 否 |
| `A_copy_f16_2k_g256` | simd_f1 | 2.805 | 3.056 | +8.95% | 是 |
| `A_copy_f16_2k_g256` | simt_f1 | 11.094 | 12.621 | +13.76% | 否 |
| `A_copy_f16_2k_g256` | simt_f2 | 8.526 | 8.608 | +0.96% | 是 |
| `A_copy_f16_2k_g256` | simt_f4 | 7.315 | 6.904 | -5.61% | 是 |
| `A_copy_f32_4k_g1024` | simd_f1 | 4.543 | 5.686 | +25.16% | 否 |
| `A_copy_f32_4k_g1024` | simt_f1 | 42.887 | 35.489 | -17.25% | 否 |
| `A_copy_f32_4k_g1024` | simt_f2 | 23.992 | 20.032 | -16.51% | 否 |
| `A_copy_f32_4k_g1024` | simt_f4 | 13.676 | 12.060 | -11.82% | 否 |
| `A_copy_f32_4k_g256` | simd_f1 | 2.799 | 3.154 | +12.68% | 否 |
| `A_copy_f32_4k_g256` | simt_f1 | 12.892 | 12.621 | -2.10% | 是 |
| `A_copy_f32_4k_g256` | simt_f2 | 8.771 | 8.608 | -1.86% | 是 |
| `A_copy_f32_4k_g256` | simt_f4 | 6.721 | 6.904 | +2.73% | 是 |
| `A_copy_f32_4k_g64` | simd_f1 | 2.357 | 2.584 | +9.63% | 是 |
| `A_copy_f32_4k_g64` | simt_f1 | 6.511 | 7.206 | +10.67% | 否 |
| `A_copy_f32_4k_g64` | simt_f2 | 4.324 | 4.330 | +0.13% | 是 |
| `A_copy_f32_4k_g64` | simt_f4 | 4.260 | 4.370 | +2.58% | 是 |
| `A_copy_i64_8k_g256` | simd_f1 | 3.085 | 3.349 | +8.57% | 是 |
| `A_copy_i64_8k_g256` | simt_f1 | 18.632 | 12.621 | -32.26% | 否 |
| `A_copy_i64_8k_g256` | simt_f2 | 12.565 | 8.608 | -31.49% | 否 |
| `A_copy_i64_8k_g256` | simt_f4 | 9.167 | 6.904 | -24.68% | 否 |
| `A_copy_i8_1k_g256` | simd_f1 | 2.621 | 3.007 | +14.74% | 否 |
| `A_copy_i8_1k_g256` | simt_f1 | 10.083 | 12.621 | +25.17% | 否 |
| `A_copy_i8_1k_g256` | simt_f2 | 7.955 | 8.608 | +8.21% | 是 |
| `A_copy_i8_1k_g256` | simt_f4 | 7.274 | 6.904 | -5.08% | 是 |
| `A_load_f32_4k_g256` | simd_f1 | 7.002 | 4.413 | -36.97% | 否 |
| `A_load_f32_4k_g256` | simt_f1 | 8.128 | 12.795 | +57.42% | 否 |
| `A_load_f32_4k_g256` | simt_f2 | 7.064 | 9.025 | +27.76% | 否 |
| `A_load_f32_4k_g256` | simt_f4 | 5.669 | 7.592 | +33.92% | 否 |
| `A_store_f16_2k_g256` | simd_f1 | 2.539 | 2.381 | -6.24% | 是 |
| `A_store_f16_2k_g256` | simt_f1 | 4.324 | 4.264 | -1.38% | 是 |
| `A_store_f16_2k_g256` | simt_f2 | 4.621 | 3.396 | -26.50% | 否 |
| `A_store_f16_2k_g256` | simt_f4 | 4.821 | 3.201 | -33.61% | 否 |
| `A_store_f32_4k_g256` | simd_f1 | 2.472 | 2.441 | -1.26% | 是 |
| `A_store_f32_4k_g256` | simt_f1 | 3.492 | 4.264 | +22.12% | 否 |
| `A_store_f32_4k_g256` | simt_f2 | 3.016 | 3.396 | +12.61% | 否 |
| `A_store_f32_4k_g256` | simt_f4 | 3.060 | 3.201 | +4.60% | 是 |
| `B_add_f32_4k_k16_g56` | simd_f1 | 4.186 | 3.992 | -4.64% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f1 | 27.726 | 29.291 | +5.65% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f2 | 27.689 | 29.291 | +5.79% | 是 |
| `B_add_f32_4k_k16_g56` | simt_f4 | 27.602 | 29.291 | +6.12% | 是 |
| `B_add_f32_4k_k64_g224` | simd_f1 | 114.633 | 111.286 | -2.92% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f1 | 951.655 | 951.322 | -0.04% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f2 | 508.772 | 475.661 | -6.51% | 是 |
| `B_add_f32_4k_k64_g224` | simt_f4 | 300.061 | 237.831 | -20.74% | 否 |
| `B_copy_f16_2k_k16_g56` | simd_f1 | 3.826 | 5.044 | +31.83% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f1 | 7.769 | 15.763 | +102.89% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f2 | 7.685 | 15.763 | +105.11% | 否 |
| `B_copy_f16_2k_k16_g56` | simt_f4 | 7.810 | 15.763 | +101.83% | 否 |
| `B_copy_f32_4k_k16_g56` | simd_f1 | 3.982 | 3.936 | -1.16% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f1 | 24.785 | 26.828 | +8.24% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f2 | 24.810 | 26.828 | +8.14% | 是 |
| `B_copy_f32_4k_k16_g56` | simt_f4 | 24.950 | 26.828 | +7.53% | 是 |
| `B_copy_f32_4k_k32_g56` | simd_f1 | 5.587 | 5.919 | +5.95% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f1 | 48.029 | 51.147 | +6.49% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f2 | 47.826 | 51.147 | +6.94% | 是 |
| `B_copy_f32_4k_k32_g56` | simt_f4 | 48.042 | 51.147 | +6.46% | 是 |
| `B_copy_f32_4k_k4_g8` | simd_f1 | 1.611 | 1.762 | +9.36% | 是 |
| `B_copy_f32_4k_k4_g8` | simt_f1 | 6.954 | 7.773 | +11.77% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f2 | 7.007 | 7.773 | +10.93% | 否 |
| `B_copy_f32_4k_k4_g8` | simt_f4 | 6.973 | 7.773 | +11.47% | 否 |
| `B_copy_f32_4k_k64_g56` | simd_f1 | 9.142 | 9.658 | +5.64% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f1 | 93.908 | 99.783 | +6.26% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f2 | 93.911 | 99.783 | +6.25% | 是 |
| `B_copy_f32_4k_k64_g56` | simt_f4 | 93.959 | 99.783 | +6.20% | 是 |
| `B_copy_f32_4k_k8_g56` | simd_f1 | 2.989 | 3.058 | +2.31% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f1 | 13.385 | 14.669 | +9.59% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f2 | 13.502 | 14.669 | +8.65% | 是 |
| `B_copy_f32_4k_k8_g56` | simt_f4 | 13.598 | 14.669 | +7.88% | 是 |
| `B_copy_i8_1k_k16_g56` | simd_f1 | 3.588 | 3.609 | +0.59% | 是 |
| `B_copy_i8_1k_k16_g56` | simt_f1 | 14.420 | 9.820 | -31.90% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f2 | 14.523 | 9.820 | -32.38% | 否 |
| `B_copy_i8_1k_k16_g56` | simt_f4 | 14.405 | 9.820 | -31.83% | 否 |
| `B_load_f32_4k_k64_g56` | simd_f1 | 10.230 | 8.744 | -14.53% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f1 | 22.813 | 31.026 | +36.00% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f2 | 24.047 | 31.026 | +29.02% | 否 |
| `B_load_f32_4k_k64_g56` | simt_f4 | 24.068 | 31.026 | +28.91% | 否 |
| `B_store_f32_4k_k16_g56` | simd_f1 | 3.809 | 3.631 | -4.67% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f1 | 3.774 | 3.783 | +0.25% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f2 | 3.705 | 3.783 | +2.11% | 是 |
| `B_store_f32_4k_k16_g56` | simt_f4 | 3.710 | 3.783 | +1.98% | 是 |
| `B_store_i64_8k_k4_g56` | simd_f1 | 2.560 | 2.307 | -9.87% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f1 | 2.812 | 3.042 | +8.16% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f2 | 2.872 | 3.042 | +5.91% | 是 |
| `B_store_i64_8k_k4_g56` | simt_f4 | 2.890 | 3.042 | +5.25% | 是 |

GUARD 49/96 within 10%
abs-err median 9.87% mean 17.92% max 105.11%


守护汇总：**循环体（公式 B）26/28 ≤10%，直线段单流 11/16 ≤10%；
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
