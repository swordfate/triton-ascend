# 连续访存统一集合场景（Unified Set）与实施计划

> 2026-11-17 立项。目标：把连续访存标定收敛为一个集合场景——直线域
> （机制一）+ 循环域（包络，L3 扩展后），其余机制要么是集合的输入修正、
> 要么按数据审计退役、要么作为"指名附加轴"挂接。本文档是实施依据。

---

## 0. 工作原则（最高优先级，全程适用）

**合并优先原则**：实施过程中若发现某个形态无法合并进集合场景，
**不得轻易放弃合并**。正确动作是一个反思循环：

1. 先怀疑**集合中对应的建模不准确**，而不是该形态特殊；
2. 用微探针对该形态直接实测——simd 用 SYS_CNT（`MOV $0, SYS_CNT`
   inline asm，is_pure=False，ns = ticks×1.00008）、simt 用
   `clock64` 标记替换法（见《Q4-costmodel偏差定位-操作指南-self.md》
   第 6 章；两者都要求结构相同的 no-load 参照变体做差）；
3. 把实测与集合模型对该形态的预测对比：不准就修常数/修形状/补轴，
   重试合并，直到准确；
4. 只有当探针复测证明集合模型在该形态上**已经准确**、而该形态仍
   系统性偏离，且能指认出集合缺失的**物理轴**时，才允许独立保留
   现有机制，且必须书面说明三件事：
   (a) 缺失的轴是什么、为什么集合张量积空间表达不了；
   (b) 探针证据（数据 + 脚本路径）；
   (c) 为什么不能做成集合上的附加轴（而必须是独立机制）。

> 历史教训（本项目已发生两次）：E17f 之前把 memory_stream −12% 归因
> 于"多流粒度"并计划加流数维度——扫描后发现流数轴是平的、真因是
> 探针代际偏差。**先扫轴、再下结论；轴平就不加，轴实才加。**

---

## 1. 集合场景定义

### 1.1 划分轴（集合的坐标系统）

```
第一刀：hasLoop（动态 for / constexpr for → 循环域；
                 static for 编译期展开 → 实例数进直线域；
                 无循环 → 直线域）
第二刀：引擎 simd / simt（两套独立参数）
第三刀：方向组合（单方向 load-only / store-only；混合 1:1~N:1）
第四刀：总字节数（log2 档位插值）
第五刀：K（循环域）→ T(K) = A + K·m 两段
第六刀：P（程序数）→ 波衰减 + 边界残差 A(P)
```

**完备性判据**：真实连续访存行为空间 ⊆ 上述六轴张量积。任何不能
落入的形态，先走 §0 的反思循环。

### 1.2 直线域（= 机制一，REVIEW §3.1-3.3）

- 串行 schedule：每条访问各付各的价（simd/simt 各一套按
  dtype×方向×驻留 的标定公式），load 时间 + store 时间**相加**；
- 流水 schedule（IndependentPipelined）：load 服务流与 store 服务流
  并行，estimator 取 **max({loadService, storeService, vector})**；
- 子线组摊销（group_a first/marginal）是本域的组效应修正；
- 链尾 store 排空已并入 store 公式截距。

### 1.3 循环域（= 扩展后的包络，REVIEW §3.5 机制四）

```
perIter(混合)   = max(TL(loadBytes), TS(storeBytes)) + κ×min(TL,TS)
perIter(仅load)  = TL(loadBytes)          ← 阶段 1 接入
perIter(仅store) = TS(storeBytes)         ← 阶段 1 接入
整核 T(K) = A(P) + K × perIter            ← 阶段 2 接入（K<16 同式）
simd: TL={90.0,113.7,196.8,392.0}, TS={106.8,123.5,214.8,377.5}, κ=0.1
simt: 待阶段 3 标定（形状 = max(load服务, store服务) + f(U,S) 串行项）
```

- `iterationCount ≥ 16` 门是 TL/TS 的**标定域**（E17f dyn/hot 扫描）；
  阶段 2 完成后 K<16 由 A+K·m 拼接接管，门退役；
- 方向感知入账：全 store 组记 store 流（与跨步组层一致），其余记
  load 流；
- K<16 的"每迭代量不收敛"（E17e）不阻碍拼接：包络只出 m（斜率），
  启动/排空全部由 A 承担。

### 1.4 输入修正（进入集合前的统一预处理，不是独立机制）

| 修正 | 作用 |
|---|---|
| E15 掩码活跃字节 | 掩码 lane 不发访存请求，按活跃比例折算字节 |
| 机制五 claimAll 实例记账 | static-for 展开/count>1 的访问按 instances 份入账，保证全覆盖不漏记 |
| 资格判定（间接寻址/非仿射地址/部分掩码 store 等排除） | 定义集合边界；被排除的走各自责任域 |

### 1.5 候选层

- 边界残差表（P 轴 1→408.3 … 56→1704.95）：整核 launch 边界/
  排空价，即循环域 A(P) 的来源；
- pAxisFactor / 波衰减：多波摊薄维度。

---

## 2. 现有机制与集合的关系（全景映射）

| 机制 | 与集合的关系 | 处置 |
|---|---|---|
| 机制一（直线 per-access） | **集合本体**（§1.2） | 保留 |
| 机制四（Tier2 包络 + Tier3 解析兜底） | **集合本体**（§1.3） | 保留；Tier3 继续做集合未覆盖处的最后兜底 |
| P 轴边界残差 + pAxisFactor/波衰减 | **集合本体**（A 与多波维度） | 保留 |
| 机制五（claimAll） | 集合的输入修正 | 保留 |
| E15 掩码修正 | 集合的输入修正 | 保留 |
| 子线组摊销 / 链尾排空 | 机制一同域修正 | 保留 |
| 机制二（direct 行表）/ 机制三（1a tiny 链） | **过渡态** | 阶段 0 合并、阶段 5 审计后按家族退役 |
| 基线 independent-load-group 直线包络 | 过渡态（直线域兜底） | 机制一全族覆盖后，审计退役 |
| 翘头 excess（1a 的特有项） | **集合缺失项** | 阶段 4 标定 excess(S) 后并入集合 |
| 别名 in-place RMW（2.3× 串行化） | 集合外（缺轴：地址别名） | 附加轴准入协议（§4） |
| 波内聚合带宽压力（−46~−57%） | 集合外（缺轴：波内数据量，N2） | 附加轴准入协议（§4），挂起 |
| 驻留 热/冷（2~6×） | 集合外（缺轴：驻留，**数据已有**） | 附加轴准入协议（§4），阶段 6 接线 |
| load×散射 store 共存 +84ns | 集合外（跨域干扰） | 维持现状（挂在集合输出上的附加项） |
| 部分掩码 store → RMW 路径 | 有意划出（连续布局但 RMW 语义） | 维持资格判定排除 |
| 跨步小行组层（calM） | 借道（跨步域缺自己的模型） | 跨步域自立后归还 |

---

## 3. 实施计划（每阶段独立过 gate，失败按 §0 回炉）

**gate 口径**（全程不变）：`run_generalization_matrix.py
--disable-profiler`，manifest v140_all_100，8 分片、卡 1/2 轮转；
实测列照抄 v140 表（5 例无实测不可评）；评分 = |误差| < 10%；
对照基线 = 上一轮同口径 tsv。任何阶段 gate 出现回归 → 该 case 用
探针定位（§0 循环），修集合建模而不是回退阶段。

| 阶段 | 内容 | 改动 | gate 判据 |
|---|---|---|---|
| **0** | 1a 并入 1：tiny_direct 表并入 directTable，excess 进 Tier 1 拟合；同机同构、行为等价 | StageCostModels.cpp 循环体瀑布 | 95 例 PASS 不降；reduce_sum_epilogue 保持 PASS |
| **1** | **单方向入包络**：✅ 已完成（ec17f16afa）——退化式 + 方向感知入账；gate 惰性（套件无 case 走该路径，曲线由 E17f 硬件锚点背书） | header `mixEnvelopePerIter` + cpp 分支条件 | 95 例 PASS 不降（82/95 持平 ✓） |
| **2** | K<16 两段拼接：✅ 已完成（bda9c5fef1）——包络从 K=1 生效（m=K16/32 稳态斜率，A 归候选层边界残差）；gate 惰性 | cpp 门 16→1 | 82/95 持平 ✓ |
| **3** | simt 包络战役：**部分合并**（见下方阶段 3 说明） | clock64 探针（e17g 系列）+ profile + cpp | simt store-only 包络接线过 gate；load 侧留 Tier 3 并记录理由 |
| **4** | simd 翘头 excess：**已接线**（d288f7d3cb）——E18b 矩阵（S{0..4}×loadBytes{2,4,16}KB）定公式：**流水隐藏容量 C（16KB 大 load → C=2，≤4KB → C=1），超出部分每个小 store ~97ns/迭代**（依赖 store 往返延迟，几何无关）；E18 v2 的"双 regime"读法是框架错误（16KB+s2 饱和是因为 C=2）。接线于 simd 包络混合全小 store 体：`excess = max(0, nStores − C(domBytes)) × 97ns`；1a 行表家族保持优先。gate 惰性（82/95 持平，套件小 store 体被 1a 行表接住） | profile `tiny_store_excess` + cpp | 82/95 持平 ✓ |
| **5** | 行表审计与退役：**审计已完成（f51878d9ec）**——`ASCEND_V2_SKIP_ROWS=1` 旁路 1a+1 行表跑全量 gate：**89/100 无行表仍 PASS 或预测不变；10 例有真实行价值（7~36pp），五簇**：①2x4096x8 元素级×4（+9.5pp 系统性）②小 store 链 rowred/reduce_sum（−31/−26pp——阶段 4 excess 的 C=1 未覆盖其幅度）③permute 双层壳实例结构（−36pp）④散射组合行（−20pp）⑤grid_stride_dot_p112/fused_softmax（−14/−7pp）；另 1 例负价值行（grid_stride_dot_p336）。**裁定：退役条件未满足，行表保留**。**逐簇归因（7092f4493）**：①2x4096x8 族 = κ 偏差（已修，κ=0 后无行表 +16→+8%）②rowred/reduce_sum = **归约-stage 域**（stage_6_rowwise_reduction 走 calibrated_tail_reduction_window 专用机器，非 contiguous 瀑布；E19 实测 链底 = TL(domLoad)+97×S = 行表价精确复现，已作为 floor 接线到包络备用，退役该簇行表需归约域模型配合——理由充分在案）③permute 双层壳 = stage_4 循环 2× 结构未编码（−2.35µs）④散射组合/p112/fused_softmax 待逐簇。阶段 5b gate：82/95 零回归。**阶段 5b 后续深挖（2026-11-18）**：链底在包络层被路由求值屏蔽——stage_6 的 all_simd 候选 work 快照天生无归约标记（geo=0，调试证实：同一 stage 两次求值，geo=1 只出现在另一候选），任何包络层判别都会被最终求值覆盖；且实测 in-kernel 链代价 = 374ns/迭代（行表口径）vs 独立探针 187（E19）——store 排空与下迭代 load 在 L2 相撞（反依赖），真实 kernel 里链代价 ×2。**正确的通用机制在 estimator 组合层**：当 stage 的归约输出是 tiny store 的数据依赖时（IR 数据流），禁止流水 max 组合、按串行求和（load+reduction+store 排空）。列为簇②的精确剩余项（需要 IR 数据流信息，是 estimator 层干预）。 | SKIP_ROWS 开关 + 对照 gate | 审计完成 |
| **6** | 附加轴接线：驻留（数据已有，按足迹 > L2 阈值启用冷档）→ 别名检测预研（静态同基指针可行性）→ 聚合压力战役（N2，挂起待真实 case） | 探针复用 + profile + cpp | 按轴分别过 gate |

**E20 四例归属裁定（2026-11-18，谁不准分析清楚）**：

| case | 循环窗实测（SYS_CNT/clock64） | 判定 |
|---|---|---|
| muls_add_n1048576 | 92.9ns/迭代（K=19）——**低于**包络 113.7 | **内存体不慢、包络没低估**；缺口 = 固定成本 A：真实 2.4-2.7µs vs 边界残差表 1.7µs（P=56，包络计价 case 系统性差 0.7-0.9µs——行表时代的 A 表与包络 stage 价不配套）。修法 = A(P) 表按包络时代重标（Δ(P) 校准战役 E20b），集合本体内 |
| fused_softmax | 1.395µs（348.7/row）vs stage 293/row | **exp/max/sum 计算欠价**（计算/归约域，非连续访存） |
| permute | 3.914µs vs stage 2.414µs | 跨步 32B load/store（rider）欠价 1.5µs + 壳结构 → **跨步域为主** |
| strided_scatter | all_simt + stride-16 散射 store | **散射/跨步域**（他组责任） |

**E20b/E20c 后续（2026-11-18，含负结果）**：Δ(P) 校准跑完——
**没有一致的正 Δ(P) 可接线**：K=1 体上模型边界在 P∈{2..56} 混合符号
（P=2：−282ns、P=8：−530、P=28：−345、P=56：+85），起步价大体正确。
muls_add_n1048576 的 −10.7% 经 E20c 决定性拆分（profiler 整核
4987.5ns = SYS_CNT 窗 1757.4 + 非循环 3230.1）：**两个误差相互抵消**——
(a) 包络窗高估 ~+403ns（12MB 工作集 L2 驻留，快于 E17f 20-50MB 流式
标定足迹的口径）；(b) 非循环低估 ~−1154ns（每迭代 grid-stride
标量/掩码/地址工作 ×19 + 启动，部分属标量域）。净剩 −510ns 与
三种测量口径间的差异（SYS_CNT 窗 / profiler Duration / v140
profiler-average 相差 5-20%）同量级——**该 case 已在测量精度极限，
继续调参等于校准噪声，记录为限制**。簇②（estimator 组合层）与
簇③④（壳/散射）仍是行表退役的实质前置。

**A(P) 重标的关键约束**：**A(P) 重标的关键约束**：A 表不能全局上移（71 个行价小核 case 会爆）；
正确形态 = **计价源感知的双表**——v2 包络认领的循环体（稳态价不含
启动）加 Δ(P) ≈ +0.8µs@P56，行价体维持现表（其 A 已含启动）。
Δ(P) 多点校准（P∈{8,14,28,56} × 真实 kernel 整核实测）为 E20b 战役。

**阶段 3 的部分合并裁定（2026-11-18，按 §0 原则执行）**：

clock64 探针战役（`probes/e17g_simt_sweep.py`、`probes/e17g6_ld_chain.py`、
`probes/e17h_simt_physics.py`，数据 `e17g_results.json`/
`e17g6_results.json`/`e17h_results.json`）的结论：

1. **store-only 侧干净、已并入**（a78dfa7772）：TS_simt(B) = B /
   21 B/ns，4/16/32KB 全尺寸线性一致（196/788/1557 ns），S=4 流并发
   微升至 25 B/ns——`loop_envelope_store_simt` 已接线；
2. **load 侧：物理因子已找到（E17h），建模方案已定，待接线**。
   按 950 架构手册（SIMT 通路 = GM → SIMT DCache(128B 行，每 AIV
   32~128KB) → 线程寄存器；fp32 每 warp 指令 = 128B = 1 行）设计
   判别实验：**cost = f(warp 指令数 × 指令宽度 × 驻留档)**——

   | 格子（16KB load，K 斜率） | /warp 指令 | /128B 行 |
   |---|---|---|
   | fp32 DCache-hot（定块重读） | 0.60 | 0.60 |
   | fp32 L2 流 | 6.00 | 6.00 |
   | fp32 DRAM 流 | 10.40 | 10.40 |
   | fp16 L2 流（64B 指令） | 7.51 | **15.02** |
   | fp16 DRAM 流 | 15.85 | **31.70** |

   fp32 两因子简并（1 指令=1 行）；fp16 64B 指令只填半条 128B 行 →
   等效行代价 ×2.5；DCache 命中比 L2 便宜 10×、比 DRAM 便宜 17×。
   **接线方案**：`perIter = (bytes/128B) × perLineCost(elementBits,
   residency)`，residency 按足迹 = bytes_per_iter×K×56 对比 L2 128MB
   分档（stream 两档；重访热档需 reuse 检测，记录为限制）；
   int64（256B/指令）一档按 fp32 行代价复用（推断，标注）；
3. **探针方法论沉淀**：块累加链在 U≥4×16KB 寄存器溢出（SIMT 寄存器
   128KB/AIV、128 线程 → 127 寄存器/线程，U4×BLOCK4096 = 128 累加
   寄存器/线程）——按寄存器预算裁剪格子（U×BLOCK ≤ 8192）；s=0 体
   必须加窗口内消费者；掩码 store 的指针必须是块指针（标量指针 +
   块值编译报错）。

---

## 4. 附加轴准入协议（集合外项的唯一合法存在形式）

任何集合外机制必须满足三条（§0 原则的书面化）：

| 附加轴 | 缺失的物理轴 | 现有证据 | 准入条件 |
|---|---|---|---|
| 驻留 | 足迹 vs L2 容量 | E17f 冷档全套数据（`probes/e17f_results.json`）+ E16b 减半证据 | 阶段 6 接线：footprint > 阈值 → 冷档曲线 |
| 别名 | 地址别名（loop-carried RMW） | E17：275.3 vs 119.6 ns/迭代（2.3×） | 先做静态同基指针检测预研；检测可行才标定乘子 |
| 聚合压力 | 波内总数据量 | G-1：−46~−57%（b2048/b4096） | 专项压力曲线战役；不在套件点上冒险改公式（N2） |
| 跨域共存 | 邻居是散射 store | E 系 84ns/轮 | 维持现状：集合输出上的挂接项 |

## 5. 基建

- 100-case 对账：`extract_run.py`（实测列严格按"实测us"列取、
  无实测=不可评）+ `gen_compare_doc.py`；
- 探针：SYS_CNT 模板见 `probes/e17f_stream_sweep.py`；
  clock64 模板见 Q4 指南 §6.2（`ascend_dpx.clock64` 标记替换法）；
- 环境：卡 1/2/3 可用、4/5/6 禁用；compile-only gate 无需 profiler；
  bishengir PATH 前置（手册 §10.2 陷阱）；
- 历史对账文档：`_100case_baseline_vs_commit_gridhint.md`
  （基线 75 / 9e6dd2a7 81 / E17f 82，可评 95 例）。
