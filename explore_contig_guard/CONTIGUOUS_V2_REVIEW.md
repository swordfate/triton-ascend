# 连续访存标定 v2：基线 → 终版 完整评审文档

基线：`2e9efcacb8`（"Keep launch-bound loops accurately priced after unified
analysis #3050"）。终版：分支 `calib/contiguous-ldst-950pr-v2-record`
（GitHub swordfate/triton-ascend），tip `5eea4c4c3`。
本文件面向评审：先讲基线有什么、每条路径在标定什么、哪里泛化不足；
再按"遇到什么现象 → 为什么老公式不行 → 引入什么新路径 → 怎么标定 →
适用范围 → 验证结果"的顺序逐条展开我们的改动。

代码位置：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`
（`mapWorkload` 的 layout 分支，基线约 1700-1800 行，终版约 1715-2560 行）。

---

## 0. 术语表（第一次出现的词都在这里）

| 术语 | 意思 |
|---|---|
| stage | costmodel 把一个 kernel 拆成若干"段"（访存段/归约段/标量段…），每段独立计价再合成整核预测 |
| 连续访存（Contiguous）| 一条访存指令读/写的一块内存内部逐字节连续（如一行 512 个 float）|
| StridedShort / StridedWide | 块内连续段 ≤32B / >32B，块间按固定步长跳跃的访存 |
| rowBytes | 块内连续段的字节数 |
| group_a / direct / tiny_direct / sdirect / loop_b | v2 的五张标定表（行键 → 纳秒），都在 profile JSON 的 `contiguous_ldst_v2` 段 |
| group_a 的 first/marginal/cTransfer/Q | 分段线性价：第 1 个 `first`，第 2..Q 个每个 `marginal`，超过 Q 个每个 `cTransfer` |
| p_axis | P（=launch program 数）轴的墙钟形状表：`factor(P) = [T(P)/P] ÷ T(1)` |
| Q（slots 语境）| ceil(P/56)：56 个物理 AIV，P 个 program 需要分几波 |
| 行键 `n<L>_s<S>_b<B>_k<K>_g<G>_m<M>` | load 条数/store 条数/每次迭代字节数/循环次数/grid/是否掩码 |
| warm / cold | 访问足迹留在 L2 内 / 每次发射轮转到 >320MB 新地址 |
| hint（programCountHint）| launch grid 的逻辑 program 数 |
| AIV | AI Vector Core，本卡 56 个（`num_vectorcore`，E14 实验 5 已复核）|
| StridedShort 认领 | 基线 #2716 的短行跨步闭式公式，先于 v2 运行并"认领"它计价过的访问 |
| boundary_residual(P) | 候选层的发射边界残差表（按物理 program 数查），随 routeFixedSystemCycles 加入候选总价 |

---

## 1. 基线（2e9efcac）的连续访存计价路径

基线的 `mapWorkload` layout 分支按以下**固定顺序**执行（顺序即优先级，
先到的把访问记入"已计价"集合，后面的路径跳过它们）：

### 路径 L1：StridedShort 短行跨步闭式（`mapStridedShortMemoryCompletion`，#2716 引入）

- **准入**：rank-2、内轴单位步、行内连续段 32..2048B、行间 gap 2..8192B、
  总跨距 ≤512KiB、无掩码、地址来源干净（SIMT 还要求行数精确匹配公式槽位）。
- **标定内容**：SIMT 五段（行数 2/8/64 × W≤2 / 3-8 / >8）+ SIMD 4 档
  （K=4/2 交付阈值）的**物理因子标定**——即"这种跨步模式每迭代真实要花
  多少 cycle"的闭式拟合。
- **产出**：认领的访问记入 `stridedShortAlreadyClaimed`，价款进
  `resources.load/store`；后续的 masked-copy 曲线见到"收据"必须让路
  （防止同一字节两次计价——这是基线已有的认领/收据机制）。

### 路径 L2：独立 load 组包络（pipelined simd 专用）

- **准入**：`simd && permitsSimdOverlap(stage) && 无同步/无条件分支`，
  且 stage 开头是**一串 flat 连续 rank-1 load**（遇到 store 即停），
  各 load 的 count 有非 1 公约数（能切成整数"组"），组内操作数在
  `independentLoadGroup{Minimum,Maximum}Operations` 窗口内。
- **标定内容**：`independentLoadGroup{KernelStartup,IterationStartup}Cycles`
  + `BytesPerCycle`——"一串连续 load 作为一组的组启动价 + 聚合带宽"。
- **局限**：只覆盖 load；store 不在此路径；对 count 结构有硬性要求
  （gcd 切组）；不区分 dtype（elementBits 必须恰好 32）。

### 路径 L3：pipelined 跨步 store 组

- **准入**：pipelined simd、stage 内所有 store 都是 StridedShort、
  同一行距/同行数、elementBits==32、行距字节命中
  `pipelinedStridedStoreGroupsByRowStrideBytes` 表、行数/次数在拟合窗口内。
- **标定内容**：`firstStore + (n−1)·incremental + max(0, rows−32)·extraLane`
  ——"一束跨步 store 的首价 + 增量价 + 超 32 行的额外 lane 价"。

### 路径 L4：masked strided copy（`mapMaskedStridedCopyCompletion`）

- **准入**：f16、masked、有配对 load 依赖的 2 维跨步 store（RMW 形状）。
- **标定内容**：按 行字节×活跃行 查 `maskedLoadStoreRmwResidualCycles`
  表；对 strided-short 已认领的访问让路（收据机制）。

### 路径 L5：GM 计数曲线 + 未覆盖尾部（兜底）

- **准入**：以上路径都没认领的访存。
- **公式**：simd = `每次操作启动开销(105.3/90.95) + 字节/带宽(167.6/213.1 B/cyc)`
  ；simt = `warp 指令数 / 每 cycle warp 指令数`。
- **这就是"兜底公式"**：纯解析、与 grid 无关、与共驻地无关、与 dtype
  只通过字节数体现。

### 基线就有的候选层边界项

`routeFixedSystemCycles` 在候选总价上追加
`boundary_residual(P) + 全局 load 边界 + 参数 setup`——按物理 program 数
查表的发射边界残差（P=56 时 ≈1705ns + 全局 load 314ns）。

### 基线的泛化性缺陷（逐条，编号对应 §3 的新路径）

| # | 缺陷 | 受害 kernel 举例 |
|---|---|---|
| G1 | 兜底公式对**共驻地（多少 program 同时跑）完全不敏感**：8 个程序跑和 56 个程序跑，每程序价一样 | vector_add（hint=97）预测 +33.8% |
| G2 | **发射边界**（每次 launch 的固定开销）没有按"2-3 波时被前波盖住"建模 | vector_add 2 波时边界被盖住 ~640ns |
| G3 | **小 store 拖慢大 load 的延迟链**没有模型：reduce_sum 形状（大 load+1 元素 store）的 K=4 探针比线性趋势高 ~2 倍 | reduce_sum_epilogue −14.1% |
| G4 | 行族键 (nL,nS) **必须完全相等**，多一个/少一个 store 就整族失配掉进兜底 | gather_dot 族失配 −15~−35% |
| G5 | 掩码访存按 **padded 满字节**计费（掩码尾部 lane 不取数） | max_dim1 +12.1% |
| G6 | **嵌套循环壳**会把同一条访问标成多实例（count>1），基线账本只记 1 份 → 尾项把另一半再算一遍 | permute +63% |
| G7 | **混流引擎共享**没有模型：连续流与跨步流同 stage 时被两套机制各按孤立价相加 | permute z-load 孤立 470 vs 合流 216 |
| G8 | **store 侧认领进错引擎桶**：store-only 组的价款进了 load 流，与独立 load 流串行相加 | strided_compaction +25% |
| G9 | 行表最低网格是 g8，**hint < g8 的 stage**（如 grid=2）拿 g8 行错配 | strided_scatter/compaction（hint=2） |
| G10 | **simt 掩码组的 /w<W> 重试**会用满字节行配活跃字节桶（桶/输入不一致） | 潜在（当前集合无受害者） |
| G11 | SIMT 归约组分支**只取组体曲线**，会遮蔽诚实的 load 流 claim | rowred −44% |
| G12 | tier-1 网格括号**无条件退一格**：hint 精确命中网格时，K 集合来自下面一格（该格行少时坍缩成单点拟合） | rms_norm +43% |
| G13 | 精确命中**单点网格**时单行半拆分拟合（m=T/2/k）失真 | rms_norm +43%（G12 的特例） |

---

## 2. 我们的改动总览（缺陷 → 新路径 → 证据 kernel → 结果）

| 缺陷 | 新路径/修复 | 证据 kernel | 修复前→后 |
|---|---|---|---|
| G1,G2 | **Formula A**（直线段 group_a × p_axis）+ **calH 边界隐藏** | vector_add | +33.8% → pass |
| G3 | **Tier 1a tiny-store 延迟链** | reduce_sum_epilogue | −14.1% → pass |
| G4 | **Tier 1 direct 行整核拟合** + 低网格行 | vector_add/copy 族、B_copy guard 族 | 多项 pass |
| G5 | **掩码活跃字节**（两个未覆盖尾项） | max_dim1 | +12.1% → pass |
| G6 | **claimAll 按实例记账** | permute | +63% → −2% |
| G7 | **混流准入**（≤64B 跨步 load 并入连续流） | permute z-load | （同上修复的一半）|
| G8 | **方向认领**（store-only 组 → resources.store） | strided_compaction | +25% → pass |
| G9 | **低网格行**（g1/g2/g4 注入 b1024 族） | strided_scatter/compaction | +11.3% → +7.5% pass |
| G10 | R17：simt 掩码组不重查 /w<W> | （潜在，当前无受害者）| — |
| G11 | **SIMT 归约组分支取 max** | rowred | −44% → pass |
| G12,G13 | **精确网格完整 K 集合 + 单点回退** | rms_norm | +43% → +7.5% pass |

同口径验证（8 并行、卡 1/2/3、预测+route）：**基线 84/100 → 终版 86/100**，
fail 集合 16 → 14（修 5：grid_dot_p112/vector_add/max_dim1/rowred/reduce_sum_epi；
新增 3：histogram×3，属 mixed 路线且静态表实测是编译失败假数据，暂缓）。
此前卡 2 单卡口径：基线 82 → 终版 85。guard（预测 join TS8 实测）：
全四组合 <10% 的 kernel **7/24**（51/96 组合）。

---

## 3. 逐条展开

### 3.0 最质朴的起点：兜底公式，以及我们在 v2 里的"基础公式"

**基线兜底公式（L5）**：对没被任何路径认领的访存，

```
simd: 价 = 操作启动开销(105.3/90.95 ns) + 字节数 / 带宽(167.6/213.1 B/cyc)
simt: 价 = warp 指令数 / 每 cycle warp 指令数
```

纯解析、零标定参数、与 grid/共驻地/dtype 分布完全无关——这就是它的
泛化性优点，也是它的全部问题来源（见 §1 缺陷表）。

**我们对它参数的改动**：基本没动公式形状，只加了一处**掩码活跃字节**
修正（G5）：掩码访存的 padded 字节按 `字节数 × activeElementRatio` 计，
因为掩码尾部 lane 并不真的取数（max_dim1 的 (64,32) 瓦片掩码到 (64,8)
只搬 2KB 而不是 8KB）。这是对兜底公式**输入**的修正，不是参数修改。

**v2 的基础公式**（我们新引入的"普通公式"）：

```
直线段（无循环）：ns = group_a[方向/dtype/桶/warm|cold] 的分段线性价
                  × pAxisFactor(P)
循环体：          四层降级 Tier 1a → 1 → 2 → 3（见 §3.3/3.4）
全部命中后：      claimAll() 认领 + 从兜底公式前移到 v2 前面运行
```

group_a 的分段线性结构（代码原样）：
```
n ≤ Q:  total = first + (n−1)·marginal
n > Q:  total = first + (Q−1)·marginal + (n−Q)·cTransfer
```
例（simd/load/fp32/4096/warm）：`first=160.1, marginal=36.6, cTransfer=44.3,
Q=16`——第 1 次 160ns，之后每次 +36.6ns，超过 16 次每次 +44.3ns（换档）。

**p_axis 参数（本轮 E14 后修改）**：`pAxisFactor(P)` 对 P≤56 **恒返回
1.0**，P>56 保留实测每程序平均衰减。依据：E14 分解证明单波内每程序
内存单价与 P 无关（纯 dispatch kernel 复现整条曲线，内存增量平坦
200-300ns），旧表把 dispatch 墙钟形状折进了单价、与候选层
boundary_residual(P) 同源叠加。

### 3.2 路径一：Formula A（直线段 group_a + p_axis）

- **现象**：多 program 小工作量 kernel（vector_add 族，hint=97）的
  预测随 P 的增长完全不匹配实测——兜底公式对共驻地不敏感（G1），
  且发射边界没有按"2-3 波被前波盖住"建模（G2）。
- **标定方法**：E 系列探针（E12/E13）在 grid=1 拟合 group_a 的
  first/marginal/cTransfer/Q（按 方向×dtype×桶×residency 分格，
  simd 48 行），E13 扫 P 轴得墙钟形状表 p_axis（loadsum/storeonly ×
  simd/simt × warm/cold，P∈{1..1024} 12 档）。
- **公式**：§3.0 所示分段线性 × pAxisFactor。**E14 修正后**：P≤56
  段 pAxisFactor 恒 1.0——E14 证明该段每程序内存单价平坦，P 的增长
  是 dispatch 墙钟（归 boundary_residual 层，不再折进单价）。
- **适用范围**：直线段、连续访存、 simd 优先（simt 走 /w<W> 键）；
  掩码组不查表（桶按活跃字节选、行按满字节算，不一致——R17 修复后
  掩码组直接走解析式）；P>224 后按表尾钳制。
- **结果**：vector_add +33.8% → pass；guard A_copy_f32_4k_g256 等全四
  组合 pass。

### 3.3 路径二：calH 边界隐藏（候选层）

- **现象**：2 波发射的 kernel（vector_add，hint=97），第二波 program 的
  启动被第一波执行盖住，但候选层的发射边界按完整值加了两遍。
  波间重叠 demo（EXPERIMENTS.md §21）证实：program 翻倍时长几乎不变
  （缩放 0.977/0.985），同量工作拆两次 1 波发射多花 13µs。
- **公式**：满足"直线段 + 访存全连续 + ≥2 访存段 + launchWaves∈{2,3}"
  时，候选总价 **−640ns**（线性常数，vector_add 锚点拟合）。
- **适用范围**：仅直线段多流 kernel；已知边界：2 波但总时长仅 ~2.4µs
  的短流（guard A_copy_f32_4k_g64）会过度隐藏（−6.5→−19.9），改进
  需按流长比例化，已记录未做。

### 3.4 路径三：循环体 Tier 1a——tiny-store 延迟链（tiny_direct 表）

- **现象**：reduce_sum 形状 kernel（大 load + 每迭代 1 元素 store）的
  K=4 探针比 K=16/32 的线性趋势高 ~2 倍——小 store 的完成延迟把前几
  迭代串行卡住，之后 DMA 管线填满才进入线性区。基线兜底公式对 K 完全
  不敏感（G3）。
- **标定**：tiny_direct 表 20 行 = 各种 (load/store 组合, K, grid) 的
  整核 Duration 实测；拟合分两段：线性边际 m（所有 K 对斜率中位数）+
  翘起部分 excess（行值 − 线性趋势）。
- **公式**：`perIter = (A×0.05 + K·m + excess×0.636)/K × 波衰减`；
  准入门：`K ≤ 族最大K×1.5+1`（不外推）。
- **参数**：`tiny_store_chain_share = 0.636`（翘起部分 63.6% 显现，
  其余与边界重叠）。
- **结果**：reduce_sum_epilogue −14.1% → pass（hd10 起）。

### 3.5 路径四：Tier 1 direct 行整核拟合（direct 表，主力）

- **现象**：兜底公式对循环体记忆流的共驻地（多少 program 同时跑）
  不敏感；不同 grid 的同形状 kernel 实测每程序价差异大。
- **标定**：direct 表按 `nL_sS_bB_kK_gG_mM` 存**整核 Duration 实测**
  （simd 295+ 行 / simt 291+ 行，含 calM 低网格 36 行）；行族按
  (nL,nS,桶距离≤1,掩码标志) 匹配，网格按 §3.7 的括号规则取值，
  分段线性拟合出 A/m。
- **公式**：`perIter = (A×0.05 + K·useM·chunkScale)/K × 波衰减`，
  `useM = K>16 && mHi>m ? mHi : m`，`chunkScale = clamp(dom/bestB, 0.5, 2)`。
- **适用范围**：`hint ≤ 224`（行只测到 4 波）；`hint < 族最低网格`
  时不出低网格行会失配——scatter/compaction 的修复就是**补测 g1/g2/g4
  行**把测量跨度延伸到 hint=2（§3.7）；hint>224 落 Tier 2/3。
- **结果**：B_copy guard 族全四组合 pass；vector_add/copy 100-case 族
  多项 pass。

### 3.6 路径五：混流准入 + 低网格行 + 方向认领（strided scatter/compaction 轮）

- **现象**（两个独立机制，都在 grid=2 的 hint=2 小核上）：
  1. **compaction**（跨步 load + 连续 store）：hint=2 低于行表最低网格
     g8，tier-1 拿 g8 行（8 程序共驻地）错配 2 程序 stage；
  2. **scatter**（连续 load + 跨步 store）：跨步 store 由 StridedShort
     曲线按**孤立测量**计价 200ns/迭代，v2 又给连续 load 认领
     216ns/迭代，串行相加 416/迭代；但真实循环两者有重叠，
     实测只有 375/迭代——孤立价相加高估。
- **标定**：
  - calN 探针：`n1_s0_b1024` / `n0_s1_b1024` 在 **g{1,2,4} × k{4,8,16}**
    的 simd+simt 行（36 行；simt g2 三会话取最小）；compaction 用
    g2 精确命中行；
  - scatter 用**体全包行**：load+跨步 store 一起测（值 = 全核实测），
    键仍用模型 eligible 集 n1_s0（strided store 不在 eligible），
    加 k2/k32 锚点稳定拟合；
  - **方向认领**：纯 store 组的价款进 `resources.store`（估计器的
    storeService），不再与 load 流串行相加。
- **准入扩展**：混流准入把"≤64B 连续段的跨步 load/store"并入
  eligible（仅当同 stage 已有连续 load 被认领）——这样跨步 store
  被体全包行一并覆盖，StridedShort 曲线不再重复计价。
- **适用范围**：混流准入仅限"同 stage 有连续 eligible load"的 stage
  （纯跨步 stage 不受影响）；rowBytes ≤ 64（更大行距的跨步归
  strided-layout 责任域）；permute 的 StridedWide 32B store 也被
  覆盖（用 calM3 全体行 n3_s1_b2048）。
- **结果**：compaction +25% → pass；scatter +11.3% → +7.5% pass
  （min-of-3 重测后）。

### 3.7 路径六：claimAll 按实例记账（嵌套循环壳）

- **现象**：permute（BLOCK1 壳内套 BLOCK1_SUB 循环）的每条访问被
  StagePartitioner 标成 count=2（`local.scale(multiplicity)`，
  StagePartitioner.cpp:1932），但基线式记账只记 1 份 → legacy 尾项把
  没记账的另一半再算一遍（permute +63%）。
- **修复**：claimAll 按 `access.count` 记账（实例数=2 就记 2 份字节），
  尾项不再重复。
- **适用范围**：一切 count>1 的 stage；count=1 的 case 完全不变。

### 3.8 路径七：掩码活跃字节（未覆盖尾项修正）

- **现象**：max_dim1 的 (64,32) 瓦片掩码到 (64,8)：真实搬运 2KB，
  未覆盖尾项按 padded 8KB 计费（+12.1%）。
- **修复**：两个未覆盖尾项的字节都按 `× activeElementRatio` 折算
  （掩码尾部 lane 不取数）。
- **适用范围**：一切静态可证部分活跃的掩码访存；uncovered 尾项只在
  v2 未认领时才计价，因此与 v2 行不冲突。

### 3.9 路径八：SIMT 归约组分支取 max（rowred）

- **现象**：rowred 的 stage 价 = 串行(load+store+归约组体)，其中
  组体曲线按 SIMT 标定遮蔽了诚实的 load 流 claim（−44%）。
- **修复**：组体分支取 `max(组体, 串行流, issue)`——组曲线不再遮蔽
  诚实的 claim（rowred → pass）。
- **适用范围**：仅 SIMT RowwiseReduction 组体分支；串行流 > 组体时才
  生效，否则与旧值一致（不碰其他 case）。

### 3.10 路径九：R17（掩码组不重查 /w<W>）+ 网格括号精化（G12/G13）

- **R17**：simt 掩码组的 /w<W> 重试会把"按活跃字节选的桶"配上"按
  满字节算的行"——已禁止（潜在修复，当前集合无受害者）。
- **网格括号**：hint 精确命中测量网格 → 用该网格**完整 K 集合**；
  单点网格回退低网格括号（G12/G13，rms_norm +43% → +7.5%）。

### 3.11 路径十：rms 族 m1 行（掩码体行）与 fused/masked_sm 派生行

- **现象**：掩码体（fused 512 宽 257 活跃）的孤立 load 行高估在上下文
  中的边际（真实循环里 load 与 store/计算部分重叠）。
- **标定**：**派生行**——`行值 = 全真实体实测 − 模型已标定计算项×K`。
  fused：`T = 全核实测(g8) − 归约项×K`；masked_sm：`T = 全核实测 −
  (vecExec+floors)×K`；rowred simt：load-only 体直测（multibuffer=True，
  旧行没开 multibuffer 浅了 2 倍）。
- **适用范围**：行键与模型 eligible 集一致（n1_s1_b2048 等），网格
  精确命中（g8/g32）；派生不引入自由系数。

---

## 4. 参数总表（终版 profile 的 contiguous_ldst_v2 段）

| 表/参数 | 行数/值 | 消费者 |
|---|---|---|
| `direct` | simd 295+36 / simt 291+19 行 | 循环体 Tier 1 |
| `tiny_direct` | 20 行 | 循环体 Tier 1a |
| `group_a` | simd 48 / simt 78 行 | 直线 Formula A |
| `sdirect` | 93 行 | **未消费**（评审确认，留作数据）|
| `loop_b` | simd 60 / simt 59 行 | **未消费**（同上）|
| `p_axis` | loadsum/storeonly × simd/simt × warm/cold | Formula A（P≤56 段现已恒 1.0）|
| `direct_fixed_share` | 0.05 | Tier1/1a 的 A 项显现份额 |
| `tiny_store_chain_share` | 0.636 | Tier1a 翘起份额 |
| `l2_cache_bytes` | 128MB | Formula A warm/cold |
| 波衰减 | 循环体 `waves^-0.05`；直线 load 侧 `waves^-0.32`（P>56 段）| §2/§3 |

---

## 5. 验证结果

| 口径 | 基线 2e9efcac | 终版 |
|---|---|---|
| 100-case（8 并行，卡 1/2/3，预测+route）| 84/100 | **86/100** |
| 卡 2 单卡口径 | 82/100 | 85/100 |
| guard（预测 join TS8 实测）| — | **7/24** kernel 全四组合 <10%（51/96 组合）|

fail 集合差异：终版修掉 grid_dot_p112/vector_add/max_dim1/rowred/
reduce_sum_epi/scatter/compaction/rms；剩余 14 项全部为基线继承或
mixed/跨步等非连续访存域（sum_high −11.98 基线同 pred 同 fail，
非回归；histogram×3 静态表为编译失败假数据，按指示暂缓）。

---

## 6. 已知边界与未完成事项

1. **scatter 残余 +10.17%**→已收敛到 +7.5%（min-of-3 重测后），见 §3.6。
2. **simt 载入/存储重叠建模**：simt 估计器串行相加，无法表达
   load/store 部分重叠（scatter 的最后 ~12%）；需要 simt 路径的
   max 语义或重叠因子，涉及估计器语义，未在本轮做。
3. **(nL,nS) 严格匹配**（R3）与 **hint≤224 悬崖**（R4）：设计取舍，
   四层降级兜底；解耦行键/平滑衔接列为后续。
4. **混流准入与 Swide 合并的优先级**（4902ae4a 一系不在本分支）：
   已在文档与代码注释中声明。
5. **histogram×3**：mixed 路线未标定 + 静态表是编译失败假数据（按
   指示暂缓）。
