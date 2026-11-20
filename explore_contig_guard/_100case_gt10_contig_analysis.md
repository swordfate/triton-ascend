# 100-case 误差>10% 算子 × 连续访存四分类归因（2026-11-19）

> 依据：`_100case_baseline_vs_commit_gridhint.md` 的 v140 实测列 +
> 三轮预测（基线 `git2e9efcac`（连续访存体系之前）/ 终版 `git9e6dd2a7`
> （E17 包络）/ 当前 `git467c2d83`（loopfit v6，= 本次 before 轮））。
> 归因方法：**跨版本误差稳定性**——三轮预测逐位不变的 case 不经过
> 连续访存公式；跟着连续访存变更移动的才是其消费者。实测口径与
> gridhint 文档一致（v140 历史实测，compile-only 不重测）。

## 一、当前（git467c2d83）仍 >10% 的 17 例总览

21 例并集（基线或终版任一 >10%）中，4 例已在当前版本 ≤10%：
`pytest_vector_add_n98432`（−0.0%）、`pytest_max_dim1`（+9.2%）、
`pytest_cumprod`（+2.4%）、`pytest_associative_scan_max`（+8.6%）。

## 二、逐例分类（17 例仍 >10%）

| # | case | 当前误差 | route | 有连续访存？ | 四分类 | 连续访存标定是主因？ |
|---|---|---:|---|---|---|---|
| 1 | `memory_stream_l4_n2097152` | −22.4% | simd | **是** | **循环域-simd** | **是（已确认）**。gridhint 文档已完整归因：loopfit/E17 曲线 grid=1 热 L2 标定，本 case 是 grid=56 + 每 trip 跳 896KB 的 DRAM 冷流。E29 干净复测同样归为 cold 聚合带宽 regime（问题1 修复对象）。注：9e6dd2a7 曾 −12.2%，E17f 常数重标后 −2.84%，loopfit v6 换曲线后回到 −22.4%——**回归由问题1 修复** |
| 2 | `rowred_r16_c128_k16` | **−54.4%** | simt | **是** | **循环域-simt** | **是（loopfit 回归）**。三轮 −44.2% → +1.45%（E17 simt 层）→ −54.4%（loopfit simt）。行归约的循环 load 体由 loopfit simt load 曲线计价，grid=1 drain-forced 标定在多程序 simt 上低估一半。**与 #1 同根：循环域缺 grid/驻留维，且 simt 也中招** |
| 3 | `pytest_reduce_sum_epilogue_f32_8x8x512` | **−29.7%** | simd | **是** | **循环域-simd** | **是（loopfit 回归）**。−14.1% → 0.0%（E17）→ −29.7%（loopfit）。归约主体的循环 load 被低估 |
| 4 | `grid_stride_dot_p56` | −20.9% | simt | **是（访存段）** | **循环域-simt** | **部分是（loopfit 回归）**。−19.4% → −11.2%（E17）→ −20.9%（loopfit）。dot 每 trip 的两条连续 load 流被低估；但 dot 的 FMA 计算与 trip 循环开销是否另有缺口需 ATTR 复核 |
| 5 | `grid_stride_dot_p112` | −20.2% | simt | **是（访存段）** | **循环域-simt** | 同 #4（−6.0% → −20.2% 回归） |
| 6 | `grid_stride_dot_p336` | −11.4% | simd | **是（访存段）** | **循环域-simd** | 部分是（基线即 −10.2%，连续访存变更影响小，缺口主体可能在其结构） |
| 7 | `pytest_sum_high_f32_64x8192` | −12.0% | simd | 有 load | —（未消费） | **否**。三轮误差逐位不变（−12.0/−12.0/−12.0）——该 kernel 的内存段不经过连续访存任何一层（走归约/masked 模型），缺口在其结构 |
| 8 | `cumsum_i32_w256` | −79.9% | simd | 有 load | —（未消费） | **否**。三轮逐位不变。持久化 scan 的跨块同步/多阶段结构缺失（−12µs 缺口），连续访存占比较小 |
| 9 | `cumsum_i32_w128_reverse` | −86.1% | simt | 有 load | —（未消费） | 同 #8 |
| 10 | `masked_softmax_r32_l150_v256_s150_k12` | −79.5% | simd | 有 load | —（几乎未消费） | **否**。−78.2→−79.5（仅 1.3pp 随 E17 动过），−79% 主体是持久化 softmax 结构缺失 |
| 11 | `pytest_cummin_f32_3x5x7_dim1` | −95.7% | simd | 有 load | —（未消费） | **否**。三轮逐位不变（实测 26.8µs vs 预测 1.16µs，缺口为 scan 结构） |
| 12 | `gather_dot_f16_k16` | −69.4% | simt | 否（gather 间接寻址） | —（indirect 域） | **否**。间接 load 不在连续访存域 |
| 13 | `gather_dot_f16_k16_p112` | −38.6% | simt | 否 | —（indirect 域） | 否 |
| 14 | `pytest_gather_dot_f16_k16_p56` | −19.4% | simt | 否 | —（indirect 域） | 否 |
| 15 | `grid_stride_gather_p224` | −21.7% | simt | 否 | —（indirect 域） | 否 |
| 16 | `atomic_continuous_i32_p56_b256` | +18.8% | simd | load 是连续 | —（atomic 域） | **否**。三轮不变；误差来自 p56 下 atomic 争用模型 |
| 17 | `atomic_discrete_i32_p56_b256` | +15.8% | simt | 否（离散） | —（atomic 域） | 否 |

（附：已在当前版本修复的 5 例中，`pytest_vector_add_n98432` 是历史上
唯一一例**直线域-单发**导致的 >10%——v8 §4.5 归因：单发公式被
dispatch 当通用段公式用，load 高估 4-7×、store 低估 2×；9e6dd2a7
起修复，当前 −0.0%。）

## 三、结论

1. **连续访存标定是主因的：5 例**（#1 memory_stream_l4、#2
   rowred_r16、#3 reduce_sum_epilogue、#4 grid_stride_dot_p56、
   #5 grid_stride_dot_p112）——**全部落在循环域**（simd 2 例、
   simt 3 例），且其中 4 例是 loopfit v6 相对 E17 时代的**回归**。
2. **共同根因指向问题1**：loopfit 曲线全部是 grid=1 热 L2 标定，
   真实 kernel 是多程序 + DRAM 冷流。E29 的 grid 扫描已经给出修法
   （hot 台阶 / cold 聚合带宽）。**重要新增结论：simt 循环体同样中招**
   （rowred −54%、grid_stride_dot_p56/p112），问题1 的 grid/驻留修正
   必须同时覆盖 simt 的 load 曲线——原计划只写了 simd。
3. **不连续访存的 11 例**（另有 1 例存疑：grid_stride_dot_p336
   走我们的 simd 循环计价但基线即 −10.2%，缺口主因可能是结构）
   分三堆：持久化 scan/softmax 结构缺失（cumsum×2、
   masked_softmax_r32、cummin，−79~−96% 是最大的一堆）、
   gather 间接寻址域（4 例）、atomic 争用模型（2 例）、归约结构
   （sum_high）。这些误差与连续访存标定无关，分四堆各自立项：
   scan 模型战役（最大）、indirect 域战役、atomic 争用标定、
   归约结构战役。
4. **直线域**（单发/组）当前没有 >10% 在逃案例：vector_add（单发）
   已修；本次 fix3（问题3 拆组键）影响的是直线域组/单发的计价精度，
   预期作用于 1-2µs 档 simd/simt 小 case 的误差分布（多为 PASS 区），
   不会直接消掉上表任何一例。

## 四、建议

- 问题1（循环域 grid/驻留）升级为**simd + simt 双侧**修正，验收集 =
  本表 #1-#5（memory_stream_l4、rowred_r16、reduce_sum_epilogue、
  grid_stride_dot_p56/p112），目标全部回到 ±10%。
- cumsum/masked_softmax/cummin 的结构缺口单独立项（scan 模型战役），
  不混入连续访存标定的验收口径。
