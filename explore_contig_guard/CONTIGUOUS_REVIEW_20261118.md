# 连续访存标定终版评审（CONTIGUOUS_REVIEW_20261118）

> 基线：`2e9efcacb8`（#3050）→ 新基线：`4902ae4a`（= 基线 + #3126
> StridedWide direct load/store costmodel；本地以 git-am 内容等价提交
> `0868cdaece` 落地）。终版：新基线 + 本 PR 的连续访存标定（分支
> `calib/unified-set-final`，HEAD 合并 `929cd70fa` 全部工作）。
> 所有 gate 均在 hint=grid harness、卡 1/2、8 分片、v140 同口径下测得。

---

## 一、背景：基线 2e9efcacb8 的连续访存路线

基线的连续访存计价有 **3 条路线**（不含 strided/indirect 等非连续域）：

| # | 路线 | 覆盖 | 公式形状 |
|---|---|---|---|
| 1 | 直线公式族（simd/simt × dtype × 方向 × 掩码 × W × lines 分桶） | 无循环的单发与流水直线组 | 串行 `T=a₀+a₁·by+a₂·blk_part+a₃·burst`；流水 `max(a₀+a₁/unroll, b·by)` |
| 2 | 独立 load 组聚合包络（independentLoopLoadGroup） | 流水多 load 组（组内共享 LSU） | 一次性启动 + 字节/83 B/cycle 聚合带宽 |
| 3 | 循环体：无独立模型 | 所有带 for 的循环体 | 沿用路线 1 的逐 op 公式按 K 线性外推（每 op 启动 + 字节/带宽） |

**是否特化：是。** 三个层面：

1. **桶级特化**：路线 1 的标定方法论是"逐桶曲线拟合"——每个
   (引擎, dtype, 掩码, W, lines) 桶一条独立手拟合公式（如 simt store
   按 (掩码 × dtype × W) 再分 6 桶）。泛化性只由桶内自变量（字节数、
   W）承载，桶与桶之间没有共享的参数结构——每遇到一个新桶就要一次
   新标定。实测的桶间行为差异确实存在（simt 流水 load 26.9% MAPE），
   但桶间没有可迁移的物理参数。
2. **循环体缺位**：路线 3 不是模型——循环体只是把直线逐 op 公式按
   K 线性外推。G3 实验证明循环体的真实行为是"流水线填满后进入
   稳态边际"（K=4 与 K=16 实测几乎一样长的翘头 + 之后线性区），
   逐 op 公式既给不出翘头也说不清稳态，误差随场景大幅波动。
3. **组效应特例**：路线 2 只为"同形多 load 组"一个场景打了补丁
   （聚合带宽 83 B/cycle），且只管 load 侧——它是特例补丁而非
   通用机制。

**特化的理由**：基线的标定方法论决定了它必然特化——所有非线性
效应（流水线饱和、启动摊销、读写重叠、延迟链）都没有参数化结构，
只能靠"新场景 → 新拟合"消化。这就是本 PR 建立参数化通用机制的
出发点。

---

## 二、本 PR 的终版连续访存计价机制

### 2.0 调用顺序（关卡图）

```
资格判定（进入集合的前提）：
  连续布局 ContiguousShort/Wide、静态布局、非间接寻址、
  非仿射地址、部分掩码 store 排除（走 RMW 路径）
输入修正：E15 掩码活跃字节折算 + claimAll 实例记账（static-for 展开）

第一刀：hasLoop？
├─ 否 → 直线域【机制一】：逐访问公式
│    ├─ 串行 schedule：各访问公式价逐条相加
│    └─ 流水 schedule：estimator 取 max(load, store, vector)
│        子线组摊销 / 链尾 store 排空为本域修正
└─ 是 → 循环域【E17 包络】，K = iterationCount ≥ 1
     ├─ (散射/compaction 组合行层：跨步域借道，非连续域)
     ├─ perIter = max(TL(loadBytes), TS(storeBytes))
     │            + tiny-store excess（S>C 时 (S−C)×97）
     │            + 归约链（有归约标记时 additive：TL(dom)+97×nStores）
     ├─ simt 循环体：逐 line 表（位宽 × 驻留 × 事务档）
     └─ 未被包络覆盖 → Tier 3 逐 op 兜底（simd 启动+字节/带宽、
        simt warp 指令速率）

第二刀（estimator 组合层，两条机制共用）：
  串行 schedule → 相加；流水 schedule → max(load, store, vector)
  例外（5c）：stage 有归约时（归约输出被 store 消费）→ 串行链

第三刀（候选层）：
  A(P) 边界残差（起步价）+ 波衰减（P>56 摊薄）
```

### 2.1 直线域：机制一（逐访问公式）

**(a) 动机实验**：基线直线公式本身即本机制的动机与起点——单发/
流水的逐访问行为稳定可拟合（v8 标定 8 族 simd/simt 串行 + 4 族
流水，MAPE 见 2.9 表 F1-F12），本 PR 保留其全部标定数据与拟合，
仅把"逐场景特化"的边界收敛到直线域（循环体迁出至包络）。

**(d) 公式与物理解释**：串行直线体内各访问背靠背发射、互不重叠，
故逐访问相加；流水直线组内 load/store/向量三类服务并行竞争，
故取瓶颈 max。串行公式 `a₀ + a₁·by + a₂·blk_part + a₃·burst`
的物理含义：a₀ 发射启动、a₁ 字节速率项、a₂ 块部分项、a₃ 突发项。

**(e) 参数分组**：按 (引擎 × dtype × 方向 × 掩码 × W × lines)
分桶，每桶 4 参数（串行）/ 2-3 参数（流水），全部继承基线标定。

### 2.2 循环域：E17 包络（simd）

**(a) 动机实验与结果**：
- **G3**：循环体逐 op 公式对 K 不敏感（K=4 实测≈K=16，线性式给
  4 倍差）→ 循环体需要独立于逐 op 的模型；
- **E16**：读写并行实测合成速率 75.6 B/ns，高于任一单流
  （load 49.1 / store 38.0），达理想双通道的 87% → load 与 store
  属两条服务通道、大比例并行；
- **E17 双向扫描**：load-only / store-only 曲线是凹形（速率随事务
  增大 53.7→97.9 B/ns）→ 需要按方向 × 字节档的曲线而非单常数；
- **E17f**：流数 × 每流事务 × 驻留三维扫描证明**流数无惩罚**
  （16KB 总量：1×16KB=200.7 / 2×8KB=197.5 / 4×4KB=196.8，±2%）
  → 模型只需"方向 × 总字节"两维；同时发现首轮 E17 探针偏快
  18-35%，据 E17f dyn/hot 口径重标（与 E16 复制探针、真实 kernel
  锚点三方互证）；
- **E17k**：调和实验证明 1:1 混合（106.6）≈ store-only（107.7）
  ——load 方向免费，**κ=0（纯 max）**同时拟合全部混合点。

**(d) 公式与物理解释**：

```
perIter = max( TL(loadBytes), TS(storeBytes) )
TL = {4KB:90.0, 8KB:113.7, 16KB:196.8, 32KB:392.0} ns（log2 插值、端点钳制）
TS = {4KB:106.8, 8KB:123.5, 16KB:214.8, 32KB:377.5} ns
```

物理：深流水稳态里每次迭代的 load 束与 store 束分属 MTE 的两条
服务通道并行工作，迭代时间 = 瓶颈方向的曲线价；曲线的凹形来自
事务越大单字节成本越低（4KB 45.5 B/ns → 32KB 83.6 B/ns，受依赖
链与事务粒度限制，E16b）。整核 `T(K) = A(P) + K × perIter`。

**(e) 参数**：TL 4 点 + TS 4 点（bytes 档共享 4 点）= 8 个标定数，
全部来自 E17f 实测节点（无拟合系数——曲线即数据）。

**(c) 补充实验（κ 的否决）**：E17k 调和实验：1:1 混合实测
106.6 ≈ store-only 107.7 → load 骑跨同一批 DCache 行免费；
κ 从 0.1 修正为 0（1:1 −0.2%、2:1 +1.7%）。

### 2.3 循环域：tiny-store excess（小 store 延迟链）

**(a) 动机实验与结果**：E18/E18b（load 链 + S 个 4B 依赖 store，
S∈{0..4} × loadBytes{2,4,16}KB × K∈{2..64}）：小 store 的完成
延迟把前几轮迭代串行卡住（K=4 实测≈K=16 的翘头），且存在流水
**隐藏容量 C**——16KB 大 load（深流水）能隐藏 2 个小 store、
≤4KB 只能隐藏 1 个；超出部分每个小 store 贡献 **97ns/迭代**
（依赖 store 的往返延迟成为瓶颈，几何无关：2k/4k/16k 三档实测
93.1-99.2 vs 模型 97）。

**(d) 公式与物理解释**：

```
excess_per_iter = max(0, nStores − C(domBytes)) × 97ns
C(domBytes) = 2 if domBytes ≥ 16KB else 1
```

物理：C 是 DMA 流水线能在 load 服务阴影里隐藏的依赖小 store 数；
超额的小 store 每个都必须等"load 返回 → 数值可用 → store 完成"
的往返。

**(e) 参数**：2 个（per_store_ns=97、capacity2 阈值 16384B）。
线性 regime MAPE = **2.2%**（8 点）。

### 2.4 循环域：归约链（reduction-fed store chain）

**(a) 动机实验与结果**：E19（精确复刻 reduce_sum 体：2KB load +
32B bias load + tl.sum + 32B 依赖 store）：稳态 **216-222 ns/迭代**
= 内存链 TL(dom) + 97×nStores（187）+ 归约计算（compute 通道
另收）——体是**延迟瓶颈**而非带宽瓶颈；E20 进一步证明 in-kernel
口径链代价 ×2（store 排空与下迭代 load 在 L2 相撞）。

**(d) 公式与物理解释**：当 stage 有归约标记时，包络价上**叠加**
链项：

```
perIterEff = perIter + ( TL(domBytes) + 97 × nStores )
```

组合层（5c）同时把该 stage 的组合从流水 max 改为串行链
（load + reduction + store）——归约输出是 store 的数据来源，
max 组合不成立。

**(e) 参数**：0 个新增（复用 TL 与 97）。

### 2.5 循环域：simt 逐 line 表

**(a) 动机实验与结果**：E17g（clock64 扫描）发现 simt store-only
干净线性 ~21 B/ns；E17h 判别实验（同字节数 fp32 vs fp16）证明
**simt load 代价 = f(warp 指令数 × 指令宽度 × 驻留档)**——fp32
每 warp 指令恰好 128B=1 条 SIMT DCache 行，fp16 的 64B 指令浪费
半条行（等效行代价 ×1.2-2.5）；E17h2/E17i 补全 (事务大小 × 位宽 ×
驻留) 全矩阵。

**(d) 公式与物理解释**：

```
perIter = (loadBytes/128) × loadLine[位宽][驻留][事务档]
        + (storeBytes/128) × storeLine[驻留][事务档]
驻留 = footprint(bytes×K×programs) > L2 128MB ? DRAM : L2
```

物理：SIMT 路径 GM → SIMT DCache(128B 行) → 线程寄存器；每条
128B 行的回填代价按来源（L2 6.0/DRAM 10.6 ns @16KB fp32）计费；
fp16 半行浪费与 4KB 发射惩罚（11.3）、32KB 反跳（11.7，队列深度
假设）都体现在表中。store 与宽度无关（回填是行粒度）。

**(e) 参数**：load 4 表（位宽 {16,32,64} × 驻留 × 4 事务档，
64-bit 复用 32-bit 行，标注推断）+ store 2 表 = 24 个数，
全部 E17h2/E17i 实测（拟合 = 数据本身）。

### 2.6 Tier 3：逐 op 兜底

基线兜底原样保留：simd 每访问 `启动(105.31/90.95 cycle) +
字节/带宽(167.56/213.13 B/cycle)` 分方向取 max；simt 按 warp
指令数/速率。兜底域 = simt 全部循环体、包络标定域外的直线形状。

### 2.7 estimator 组合层（两机制共用）

串行 schedule 相加；流水 schedule `max(load, store, vector)`；
例外（5c）：`r.reduction > 0` 的循环体按串行链组合（见 2.4）。

### 2.8 候选层

`A(P)` 边界残差表（起步价，1→408.3 … 56→1704.95 ns）+ 波衰减。
A 是 `T(K) = A(P) + K × perIter` 的 A——E20b 实测真实起步价
0.9-2.4µs（P∈{2..56}），现表大体正确（混合符号 ±350-530ns），
保留。

### 2.9 公式列表

| 编号 | 场景 | 公式 | 拟合点数 | MAPE |
|---|---|---|---|---|
| F1 | 直线串行 simd load | a₀+a₁·by+a₂·blk_part+a₃·burst | 133 | 6.7% |
| F2 | 直线串行 simt load（lines<1） | c₀..c₅ 六参数 | 133 | 6.9% |
| F3 | 直线串行 simt load（1≤lines≤32） | 同 F2 形状 | 154+89 | 7.5% / 9.4%（两段） |
| F4 | 直线串行 simt load（lines>32 掩码） | nospill/spill 两段 | 75 | 16.4% |
| F5 | 直线串行 simd store | 同 F1 形状 | 135 | 5.7% |
| F6 | 直线串行 simt store（lines<1） | 同 F2 形状 | 133 | 8.1% |
| F7 | 直线串行 simt store（te<1） | 同 F2 形状 | 181 | 6.5% |
| F8 | 直线串行 simt store（te≥1） | 6 桶 | 15（拟合集 167） | 14.0% |
| F9 | 直线流水 simd load | max(a₀+a₁/unroll, b·by) | 175 | 8.6% |
| F10 | 直线流水 simt load | 按 W 分桶加权 | 150 | 26.9% |
| F11 | 直线流水 simd store | 同 F9 形状 | 171 | 7.1% |
| F12 | 直线流水 simt store | 同 F10 形状 | 119 | 26.9% |
| F13 | 子线组摊销（同形多 load） | 52+(u−1)×26 ns | first 9-12/族、marginal 45-60 | first 2.3-4.9% / marginal 7.3-36.1% |
| E1 | 循环包络 simd load 曲线 TL | log2 字节插值 4 节点 {90, 113.7, 196.8, 392} | 4 节点（E17f 域内 10 格验证） | 域内 0-2.3% |
| E2 | 循环包络 simd store 曲线 TS | {106.8, 123.5, 214.8, 377.5} | 4 节点（域内 6 格验证） | 0-0.3% |
| E3 | 循环包络混合组合 | max(TL, TS)（κ=0） | E17k + E20 混合点 | 1:1 −0.2% / 2:1 +1.7% |
| E4 | tiny-store excess | max(0, nStores−C(dom))×97 | E18b 8 线性点 | 2.2% |
| E5 | 归约链（叠加项） | TL(dom) + 97×nStores（有归约标记时） | E19 形状锚点 | 实测 216-222 vs 模型 216（≈3%） |
| S1 | simt load 逐 line 表 | (bytes/128)×line[位宽][驻留][档] | E17h2 16 格 | 表=实测（样本内） |
| S2 | simt store 逐 line 表 | (bytes/128)×line[驻留][档] | E17i 16 格 | 表=实测（样本内） |
| T1 | Tier 3 兜底（simd） | 启动 + 字节/带宽，分方向 max | 基线继承 | 基线口径 |
| T2 | Tier 3 兜底（simt） | warp 指令数/速率 | 基线继承 | 基线口径 |

> 注：E1/E2/S1/S2 的"拟合"是节点直读（曲线=实测数据），因此样本内
> MAPE 为 0；泛化误差由 §三 的整核 gate 承载。包络全 34 格含多
> store 流（>1 流/方向）与域外 clamp 形状时 MAPE 8.4%，单流域内
> 22 格 6.8%——多 store 流为已知限制（每流串行效应未建模）。

---

## 三、误差结果

### 3.1 100-case（新基线 4902ae4a vs 终版）

口径：harness hint=grid（8 kernel bench）、compile-only 8 分片卡 1/2、v140 同口径实测列照抄。95 例可评（5 例无实测不可评）。

**汇总：基线 4902ae4a 75 PASS（中位 |err| 4.10%）→ 终版 75 PASS（中位 4.56%），0 例共同回归，集合换血 4 出 4 入。**

#### 基线 FAIL → 终版 PASS（4 例，新机制修复）

| case | 基线误差 | 终版误差 | 修复来源 |
|---|---:|---:|---|
| `pytest_vector_add_n98432` | +33.77% | -0.05% | 直线流水组效应（子线组摊销 + est max 组合） |
| `pytest_max_dim1_f32_m64_n8` | +12.12% | +9.15% | 低 grid 直行与包络组合 |
| `pytest_cumprod_f32_n1024` | +12.94% | +2.36% | 同上 |
| `pytest_associative_scan_max_f32_128x4x16_dim2` | +34.96% | +8.56% | 同上 |

#### 基线 PASS → 终版 FAIL（4 例，全部归因在案）

| case | 基线误差 | 终版误差 | 归因 | 责任域 |
|---|---:|---:|---|---|
| `muls_add_n1048576` | -0.25% | -10.75% | 包络窗高估（12MB 工作集 L2 驻留 vs 标定流式足迹）与非循环段低估相互抵消后剩 -510ns ≈ 三种测量口径差异量级 | 连续访存（已达测量精度极限，E20c） |
| `pytest_permute_reshape_fused_f32` | -0.62% | -38.34% | 跨步 32B rider 欠价 1.5µs + 双层壳结构 | 跨步域为主 + 壳结构 |
| `pytest_fused_softmax_f32_128x257` | -4.81% | -13.33% | exp/max/sum 计算欠价 | 计算/归约域 |
| `pytest_strided_scatter_f32_s16` | -7.51% | -13.37% | stride-16 散射 store 欠价 | 散射域（他组责任） |

#### 全表（100 case）

| # | case | 基线route | 实测(us)¹ | 基线预测 | 基线误差 | 基线 | 终版route | 终版预测 | 终版误差 | 终版 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
| 1 | `muls_add_n2097152` | simd+F1 | 6.625 | 6.735 | +1.66% | PASS | simd+F1 | 6.282 | -5.18% | PASS |
| 2 | `muls_add_n1048576` | simd+F1 | 4.746 | 4.734 | -0.25% | PASS | simd+F1 | 4.236 | -10.75% | **FAIL** |
| 3 | `memory_stream_l4_n2097152` | simd+F1 | 10.032 | 10.371 | +3.38% | PASS | simd+F1 | 9.352 | -6.78% | PASS |
| 4 | `memory_stream_l2_n1048576` | simd+F1 | 4.645 | 4.729 | +1.81% | PASS | simd+F1 | 4.231 | -8.91% | PASS |
| 5 | `cumsum_i32_w256` | simd+F1 | 14.856 | 2.986 | -79.90% | **FAIL** | simd+F1 | 2.986 | -79.90% | **FAIL** |
| 6 | `cumsum_i32_w128_reverse` | simt+F1 | 16.628 | 2.317 | -86.07% | **FAIL** | simt+F1 | 2.317 | -86.07% | **FAIL** |
| 7 | `gather_dot_f16_k16` | simt+F1 | 12.186 | 3.723 | -69.45% | **FAIL** | simt+F1 | 3.723 | -69.45% | **FAIL** |
| 8 | `gather_dot_f16_k16_p112` | simt+F1 | 6.061 | 3.723 | -38.57% | **FAIL** | simt+F1 | 3.723 | -38.57% | **FAIL** |
| 9 | `masked_copy_r16_l64_v64_s64_k8` | simd+F1 | 3.095 | 3.116 | +0.68% | PASS | simd+F1 | 2.932 | -5.27% | PASS |
| 10 | `masked_copy_r64_l128_v128_s256_k12` | simd+F1 | 7.48 | 7.669 | +2.53% | PASS | simd+F1 | 7.669 | +2.53% | PASS |
| 11 | `grid_stride_gather_p224` | simt+F1 | 3.836 | 3.002 | -21.74% | **FAIL** | simt+F1 | 3.002 | -21.74% | **FAIL** |
| 12 | `grid_stride_dot_p224` | simd+F1 | 5.027 | 4.984 | -0.86% | PASS | simd+F1 | 4.972 | -1.09% | PASS |
| 13 | `grid_stride_dot_p56` | simt+F1 | 3.976 | 3.205 | -19.39% | **FAIL** | simt+F1 | 3.158 | -20.57% | **FAIL** |
| 14 | `grid_stride_dot_p112` | simt+F1 | 4.72 | 3.89 | -17.58% | **FAIL** | simt+F1 | 3.795 | -19.60% | **FAIL** |
| 15 | `grid_stride_dot_p336` | simd+F1 | 6.035 | 5.418 | -10.22% | **FAIL** | simd+F1 | 5.399 | -10.54% | **FAIL** |
| 16 | `grid_stride_dot_p448` | simd+F1 | 6.0 | 5.851 | -2.48% | PASS | simd+F1 | 5.826 | -2.90% | PASS |
| 17 | `pytest_indirect_load_add_b8` | simt+F1 | 1.201 | 1.234 | +2.75% | PASS | simt+F1 | 1.234 | +2.75% | PASS |
| 18 | `pytest_indirect_index_div_b8` | simt+F1 | 1.244 | 1.245 | +0.08% | PASS | simt+F1 | 1.245 | +0.08% | PASS |
| 19 | `pytest_indirect_index_and_b8` | simt+F1 | 1.169 | 1.213 | +3.76% | PASS | simt+F1 | 1.213 | +3.76% | PASS |
| 20 | `pytest_indirect_index_maxmin_b8` | simt+F1 | 1.166 | 1.213 | +4.03% | PASS | simt+F1 | 1.213 | +4.03% | PASS |
| 21 | `pytest_indirect_load_2d_min_m4_n16_k8` | simt+F1 | 1.174 | 1.255 | +6.90% | PASS | simt+F1 | 1.255 | +6.90% | PASS |
| 22 | `pytest_indirect_load_2d_grid_m4_n16_k8` | simt+F1 | 1.72 | 1.573 | -8.55% | PASS | simt+F1 | 1.573 | -8.55% | PASS |
| 23 | `pytest_index_select_n64_d32_m8` | simd+F1 | 1.136 | 1.084 | -4.58% | PASS | simd+F1 | 1.084 | -4.58% | PASS |
| 24 | `pytest_index_put_n64_d32_m8` | simt+F1 | 1.166 | 1.264 | +8.40% | PASS | simt+F1 | 1.264 | +8.40% | PASS |
| 25 | `pytest_indirect_store_b8` | simt+F1 | 1.09 | 1.162 | +6.61% | PASS | simt+F1 | 1.162 | +6.61% | PASS |
| 26 | `pytest_gather_dot_f16_k16_p56` | simt+F1 | 4.619 | 3.722 | -19.42% | **FAIL** | simt+F1 | 3.722 | -19.42% | **FAIL** |
| 27 | `pytest_vector_add_n98432` | simd+F1 | 2.141 | 2.864 | +33.77% | **FAIL** | simd+F1 | 2.14 | -0.05% | PASS |
| 28 | `pytest_permute_x64_y32` | simd+F1 | 2.327 | 2.333 | +0.26% | PASS | simd+F1 | 2.333 | +0.26% | PASS |
| 29 | `pytest_sigmoid_f32_2x4096x8` | simd+F1 | 4.9 | 4.633 | -5.45% | PASS | simd+F1 | 4.713 | -3.82% | PASS |
| 30 | `pytest_rms_norm_f16_m128_n128` | simd+F1 | 6.152 | 6.611 | +7.46% | PASS | simd+F1 | 6.611 | +7.46% | PASS |
| 31 | `pytest_layer_norm_f16_m128_n128` | simt+F1 | 3.679 | 3.461 | -5.93% | PASS | simt+F1 | 3.429 | -6.80% | PASS |
| 32 | `pytest_insert_slice_n1024_s32` | simd+F1 | 1.117 | 1.078 | -3.49% | PASS | simd+F1 | 1.078 | -3.49% | PASS |
| 33 | `pytest_atomic_add_f32_32x32_c2` | simd+F1 | None | 1.849 | — | 不可评 | simd+F1 | 1.849 | — | 不可评 |
| 34 | `pytest_exp_f32_2x4096x8` | simd+F1 | 4.347 | 4.596 | +5.73% | PASS | simd+F1 | 4.488 | +3.24% | PASS |
| 35 | `pytest_log_f32_n781` | simd+F1 | 1.002 | 1.057 | +5.49% | PASS | simd+F1 | 1.047 | +4.49% | PASS |
| 36 | `pytest_relu_f32_2x4096x8` | simd+F1 | 4.377 | 4.628 | +5.73% | PASS | simd+F1 | 4.708 | +7.56% | PASS |
| 37 | `pytest_where_f32_2x1024x8` | simd+F1 | 1.947 | 1.959 | +0.62% | PASS | simd+F1 | 1.98 | +1.69% | PASS |
| 38 | `pytest_broadcast_bf16_2x4x8` | simd+F1 | 0.988 | 0.956 | -3.24% | PASS | simd+F1 | 0.956 | -3.24% | PASS |
| 39 | `pytest_max_dim1_f32_m64_n8` | simd+F1 | 1.246 | 1.397 | +12.12% | **FAIL** | simd+F1 | 1.36 | +9.15% | PASS |
| 40 | `pytest_mean_dim1_f32_m57_n3` | simd+F1 | 1.163 | 1.166 | +0.26% | PASS | simd+F1 | 1.146 | -1.46% | PASS |
| 41 | `pytest_mul_reduce_f16_16x1x1x1x1` | simd+F1 | 0.987 | 1.067 | +8.11% | PASS | simd+F1 | 1.067 | +8.11% | PASS |
| 42 | `pytest_cumprod_f32_n1024` | simt+F1 | 1.778 | 2.008 | +12.94% | **FAIL** | simt+F1 | 1.82 | +2.36% | PASS |
| 43 | `pytest_cummax_f32_16x32_dim1` | simt+F1 | 1.272 | 1.214 | -4.56% | PASS | simt+F1 | 1.214 | -4.56% | PASS |
| 44 | `pytest_atomic_max_i32_32x32_c2` | simd+F1 | 1.325 | 1.446 | +9.13% | PASS | simd+F1 | 1.446 | +9.13% | PASS |
| 45 | `atomic_continuous_i32_p1_b256` | simd+F1 | 1.049 | 1.009 | -3.81% | PASS | simd+F1 | 1.009 | -3.81% | PASS |
| 46 | `atomic_continuous_i32_p56_b256` | simd+F1 | 1.94 | 2.306 | +18.87% | **FAIL** | simd+F1 | 2.306 | +18.87% | **FAIL** |
| 47 | `atomic_discrete_i32_p1_b256` | simt+F1 | 1.572 | 1.527 | -2.86% | PASS | simt+F1 | 1.527 | -2.86% | PASS |
| 48 | `atomic_discrete_i32_p56_b256` | simt+F1 | 2.438 | 2.824 | +15.83% | **FAIL** | simt+F1 | 2.824 | +15.83% | **FAIL** |
| 49 | `rowred_r16_c128_k16` | simt+F1 | 28.477 | 15.88 | -44.24% | **FAIL** | simt+F1 | 19.93 | -30.01% | **FAIL** |
| 50 | `rowred_r64_c256_k22` | simd+F1 | 97.115 | 92.769 | -4.48% | PASS | simd+F1 | 92.769 | -4.48% | PASS |
| 51 | `softmax_power2` | simd+F1 | 66.101 | 65.2 | -1.36% | PASS | simd+F1 | 61.637 | -6.75% | PASS |
| 52 | `softmax_original` | simd+F1 | 32.494 | 31.261 | -3.79% | PASS | simd+F1 | 31.265 | -3.78% | PASS |
| 53 | `fze_default` | simt+F2 | 2825.22 | 2840.585 | +0.54% | PASS | simt+F2 | 2840.544 | +0.54% | PASS |
| 54 | `solve_tril_b1_t1024_h32_bt64` | mixed+F4 | 185.085 | 177.505 | -4.10% | PASS | mixed+F4 | 177.505 | -4.10% | PASS |
| 55 | `solve_tril_b1_t1024_h64_bt64` | mixed+F4 | 325.322 | 321.309 | -1.23% | PASS | mixed+F4 | 321.309 | -1.23% | PASS |
| 56 | `solve_tril_b8_t1024_h32_bt64` | mixed+F4 | 1506.969 | 1413.332 | -6.21% | PASS | mixed+F4 | 1413.332 | -6.21% | PASS |
| 57 | `solve_tril_b16_t1024_h64_bt64` | mixed+F4 | 5943.006 | 5840.174 | -1.73% | PASS | mixed+F4 | 5840.174 | -1.73% | PASS |
| 58 | `solve_tril_b1_t8192_h64_bt64` | mixed+F4 | 2980.632 | 2877.396 | -3.46% | PASS | mixed+F4 | 2877.396 | -3.46% | PASS |
| 59 | `solve_tril_b8_t8192_h32_bt64` | mixed+F4 | 11848.565 | 11754.981 | -0.79% | PASS | mixed+F4 | 11754.981 | -0.79% | PASS |
| 60 | `solve_tril_b16_t8192_h64_bt64` | mixed+F4 | 47645.837 | 47192.594 | -0.95% | PASS | mixed+F4 | 47192.594 | -0.95% | PASS |
| 61 | `solve_tril_b4_t131072_h32_bt64` | mixed+F4 | 95234.816 | 94488.175 | -0.78% | PASS | mixed+F4 | 94488.175 | -0.78% | PASS |
| 62 | `masked_reduction_r32_l150_v256_s150_k12` | simd+F1 | 47.341 | 43.913 | -7.24% | PASS | simd+F1 | 43.913 | -7.24% | PASS |
| 63 | `masked_reduction_r64_l150_v256_s150_k22` | simd+F1 | 98.008 | 92.947 | -5.16% | PASS | simd+F1 | 92.947 | -5.16% | PASS |
| 64 | `masked_softmax_r32_l150_v256_s150_k12` | simd+F1 | 78.009 | 16.995 | -78.21% | **FAIL** | simd+F1 | 16.023 | -79.46% | **FAIL** |
| 65 | `masked_softmax_r80_l150_v150_s150_k18` | simd+F1 | 32.261 | 31.26 | -3.10% | PASS | simd+F1 | 31.264 | -3.09% | PASS |
| 66 | `pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512` | simt+F4 | 8.659 | 8.259 | -4.62% | PASS | simt+F4 | 8.159 | -5.77% | PASS |
| 67 | `pytest_sum_low_f32_64x8192` | simd+F1 | 46.671 | 46.435 | -0.51% | PASS | simd+F1 | 46.435 | -0.51% | PASS |
| 68 | `pytest_sum_high_f32_64x8192` | simd+F1 | 4.5 | 3.961 | -11.98% | **FAIL** | simd+F1 | 3.961 | -11.98% | **FAIL** |
| 69 | `histogram_i32_r8192_b256_uniform` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 70 | `histogram_i32_r8192_b256_single` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 71 | `histogram_i32_r8192_b256_radix` | mixed+F1 | None | 73.133 | — | 不可评 | mixed+F1 | 60.076 | — | 不可评 |
| 72 | `pytest_discrete_mask_4way_f16_8x8` | simt+F1 | 1.0 | 1.029 | +2.90% | PASS | simt+F1 | 1.029 | +2.90% | PASS |
| 73 | `pytest_reduce_sum_epilogue_f32_8x8x512` | simd+F1 | 2.772 | 2.38 | -14.14% | **FAIL** | simd+F1 | 2.013 | -27.38% | **FAIL** |
| 74 | `pytest_max_dim0_f32_64x8` | simd+F1 | 1.251 | 1.371 | +9.59% | PASS | simd+F1 | 1.334 | +6.63% | PASS |
| 75 | `pytest_argmax_f32_37x125_dim1` | simt+F1 | 22.148 | 22.483 | +1.51% | PASS | simt+F1 | 22.483 | +1.51% | PASS |
| 76 | `pytest_gather_after_reduce_f32_16x1024` | simd+F1 | 4.608 | 4.745 | +2.97% | PASS | simd+F1 | 4.745 | +2.97% | PASS |
| 77 | `pytest_view_f32_2x256x16` | simd+F1 | 1.283 | 1.304 | +1.64% | PASS | simd+F1 | 1.304 | +1.64% | PASS |
| 78 | `pytest_permute_reshape_fused_f32` | simd+F1 | 6.471 | 6.431 | -0.62% | PASS | simd+F1 | 3.99 | -38.34% | **FAIL** |
| 79 | `pytest_nearest_resize_i32_1x4x64x64` | simt+F1 | 2.46 | 2.346 | -4.63% | PASS | simt+F1 | 2.346 | -4.63% | PASS |
| 80 | `pytest_sort_f32_256x16` | simd+F1 | 51.767 | 51.612 | -0.30% | PASS | simd+F1 | 51.612 | -0.30% | PASS |
| 81 | `pytest_topk_f32_16x64_k4` | simt+F1 | 1.605 | 1.588 | -1.06% | PASS | simt+F1 | 1.588 | -1.06% | PASS |
| 82 | `pytest_multi_input_scan_f32_2x5x10` | simt+F1 | 1.694 | 1.571 | -7.26% | PASS | simt+F1 | 1.571 | -7.26% | PASS |
| 83 | `pytest_cummin_f32_3x5x7_dim1` | simd+F1 | 26.797 | 1.16 | -95.67% | **FAIL** | simd+F1 | 1.16 | -95.67% | **FAIL** |
| 84 | `pytest_histogram_i32_2048_bins2` | mixed+F1 | None | 3.028 | — | 不可评 | mixed+F1 | 3.028 | — | 不可评 |
| 85 | `pytest_scalar_reduce_store_f32_n16` | simd+F1 | 1.498 | 1.559 | +4.07% | PASS | simd+F1 | 1.559 | +4.07% | PASS |
| 86 | `pytest_conditional_load_f32_n32` | simt+F1 | 1.106 | 1.117 | +0.99% | PASS | simt+F1 | 1.117 | +0.99% | PASS |
| 87 | `pytest_dot_bf16_16x32x16` | simd+F1 | 1.244 | 1.246 | +0.16% | PASS | simd+F1 | 1.246 | +0.16% | PASS |
| 88 | `pytest_fused_softmax_f32_128x257` | simd+F1 | 3.158 | 3.006 | -4.81% | PASS | simd+F1 | 2.737 | -13.33% | **FAIL** |
| 89 | `pytest_add_f32_2x4096x8` | simd+F1 | 4.349 | 4.628 | +6.42% | PASS | simd+F1 | 4.709 | +8.28% | PASS |
| 90 | `pytest_sqrt_f32_n781` | simd+F1 | 1.003 | 1.023 | +1.99% | PASS | simd+F1 | 1.013 | +1.00% | PASS |
| 91 | `pytest_strided_scatter_f32_s16` | simt+F1 | 4.302 | 3.979 | -7.51% | PASS | simt+F1 | 3.727 | -13.37% | **FAIL** |
| 92 | `pytest_block_ptr_strided_f32_4x8_s32x4` | simt+F1 | 1.112 | 1.091 | -1.89% | PASS | simt+F1 | 1.091 | -1.89% | PASS |
| 93 | `pytest_cos_f32_n781` | simd+F1 | 1.529 | 1.561 | +2.09% | PASS | simd+F1 | 1.551 | +1.44% | PASS |
| 94 | `pytest_mul_f32_2x4096x8` | simd+F1 | 4.355 | 4.628 | +6.27% | PASS | simd+F1 | 4.709 | +8.13% | PASS |
| 95 | `pytest_associative_scan_max_f32_128x4x16_dim2` | simt+F1 | 4.988 | 6.732 | +34.96% | **FAIL** | simt+F1 | 5.415 | +8.56% | PASS |
| 96 | `pytest_div_f32_2x4096x8` | simd+F1 | 4.365 | 4.628 | +6.03% | PASS | simd+F1 | 4.709 | +7.88% | PASS |
| 97 | `pytest_f2i32_f32_n3` | simd+F1 | 0.986 | 0.958 | -2.84% | PASS | simd+F1 | 0.957 | -2.94% | PASS |
| 98 | `pytest_chunk_cumsum_f32_b4_t2048_h8_bt128` | simt+F2 | 3.906 | 3.869 | -0.95% | PASS | simt+F2 | 3.869 | -0.95% | PASS |
| 99 | `pytest_maximum_f32_2x4096x8` | simd+F1 | 5.661 | 5.673 | +0.21% | PASS | simd+F1 | 5.754 | +1.64% | PASS |
| 100 | `pytest_strided_compaction_f32_s16` | simd+F1 | 2.956 | 2.916 | -1.35% | PASS | simd+F1 | 2.916 | -1.35% | PASS |


### 3.2 guard（24 case × 4 mode = 96 组合，终版 compile-only 预测 vs 历史 guard 实测）

**汇总：51/96 组合 PASS；四 mode 全部 <10% 的 case 7/24**（A_copy_f32_4k_g256、B_copy_f32_4k_k{8,16,32,64}_g56 全 k 扫描、B_store_f32_4k_k16_g56、B_store_i64_8k_k4_g56）。

guard 集合的本意是压测行表的覆盖边界；行表退役后，剩余缺口与 100-case 的簇归因一致（直线多流/掩码形状、散射组合、低 trip 循环），已在 §2 各机制的"已知限制"中声明，属于后续标定战役的范围而非机制形态问题。

| case | mode | 实测 | 本PR后预测 | 预测误差 | 整核误差是否<10% |
|---|---|---:|---:|---:|:---:|
