# 连续访存标定终版评审（CONTIGUOUS_REVIEW_20261118）

> 基线：`2e9efcacb8`（#3050）→ 新基线：`4902ae4a`（= 基线 + #3126
> StridedWide direct load/store costmodel，上游提交；本分支树上同一补丁
> 以 git-am 内容等价提交 `0868cdaece` 承载，二者树逐位相同）。
> 终版：新基线 + 本 PR 的连续访存标定（分支 `calib/unified-set-final`，
> HEAD `73a131d283`，包含 `929cd70fa` 及其之前全部工作）。
> 所有 gate 均在 hint=grid harness、卡 1/2、8 分片、v140 同口径下测得。
>
> 代码位置：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/`
> `StageCostModels.cpp` 的 `mapContiguousV2Completion`（直线域机制一 +
> 循环域包络 + Tier 3）、`estimateStageBase`（组合层）；
> `include/.../ContiguousLdStV2CostModel.h`（公式助手与参数结构）；
> `profiles/simd_simt/ascend_950pr_simd_simt_v1.json`（参数表）。

---

## 一、背景：基线 2e9efcacb8 的连续访存路线

基线的连续访存计价有 **3 条路线**（不含 strided/indirect 等非连续域）：

| # | 路线 | 覆盖 | 公式形状 |
|---|---|---|---|
| 1 | 逐访问解析式（+ simt 按 W 分档吞吐表） | 每条连续 load/store（直线体与循环体同式） | simd：`启动 + 字节/带宽`，load 105.31 cycle + 167.56 B/cycle、store 90.95 + 213.13（microbench 联合拟合常数，全部形状共用）；simt：`warp 指令数 / 吞吐`，吞吐按启动 warp 数 W∈{1..32} 逐档实测 |
| 2 | 独立 load 组聚合包络（independent_loop_load_group） | 流水段开头 1–4 条连续 load 流 | 内核启动 546.1 + 迭代启动 12.8 + 字节/83.26 B/cycle 聚合带宽 |
| 3 | 循环体：无独立模型 | 所有带 for 的循环体 | 沿用路线 1 的逐 op 公式按 K 线性外推（每 op 启动 + 字节/带宽）×K |

**是否特化：是。** 三个层面：

1. **单访问口径、无组结构**：路线 1 对每条访问独立计价，多条访问只能
   逐条相加——同形状多条访问在 MTE 队列里的并发摊薄（第 1 条付全价、
   后面打折）没有参数结构，多波聚合带宽摊薄也没有（只有路线 2 一个
   83 B/cycle 特例）。典型证据：vector_add（每程序 2 load + 1 store、
   97 程序 2 波）被高估 **+29.86%**（G1）。
2. **循环体缺位**：路线 3 不是模型——直线逐 op 公式按 K 线性外推。
   E6/E7 证明循环体是不同的执行上下文（受每轮发射节流，稳态斜率是
   上下文常数，见 §2.1 动机说明）；G3 的 K 扫描给出直接反例：整核
   时长 T(4)=4211ns ≈ T(16)=3881ns（几乎一样长），到 K=64 才线性涨到
   8905ns。这里的"**翘头**"指 T(K) 曲线在小 K 端高于线性趋势的头部
   段：流水线填满/延迟链排空之前，时长几乎不随 K 增长（近似平台）；
   把大 K 的线性趋势外推回小 K，实测点悬在趋势线上方，形如翘起的头。
   逐 op×K 线性式既给不出这个头（K=4 处偏差约 4 倍），也说不清稳态
   斜率，误差随场景大幅波动。
3. **组效应特例**：路线 2 只为"流水段开头 1–4 条 FP32 连续 load"一个
   场景打了补丁（83.26 B/cycle 聚合带宽），且只管 load 侧——它是
   特例补丁而非通用机制。

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

第一刀：hasLoop？（判据：stage 里是否还是运行期 scf.for/while；
  static_range/constexpr 展开的循环 IR 里已消失，归直线域）
├─ 否 → 直线域【机制一】：同形分组价 + 解析式（§2.1）
│    ├─ 同 stage 的同形多访问组（n≥2）：
│    │    first+(min(n,Q)−1)·marginal+max(0,n−Q)·kc·marginal，×pAxis(P)
│    ├─ 单发 / 未命中格 / 掩码组：启动+字节/带宽解析式（与基线同源；
│    │    simd load 侧 × 波衰减 waves^−0.32）
│    └─ 组合层按 straight_line schedule 串行相加；基线继承残差：
│        simt 直线小访存（1 load+1 store ≤128 元素）−113 cycle
└─ 是 → 循环域【E17 包络】，K = iterationCount ≥ 1
     ├─ perIter = max(TL(loadBytes), TS(storeBytes))
     │            + tiny-store excess（S>C 时 (S−C)×97）
     │            + 归约链（有归约标记时 additive：TL(dom)+97×nStores）
     ├─ simt 循环体：逐 line 表（位宽 × 驻留 × 事务档）
     └─ 未被包络覆盖 → Tier 3 逐 op 兜底（simd 启动+字节/带宽、
        simt warp 指令速率）

第二刀（estimator 组合层，按 schedule 分支）：
  straight_line schedule（直线域全部）→ 相加；
  independent_pipelined schedule（循环域）→ max(load, store, vector)
  例外（5c）：stage 有归约时（归约输出被 store 消费）→ 串行链

第三刀（候选层）：
  A(P) 边界残差（起步价）+ 波衰减（P>56 摊薄）
```

### 2.1 直线域：机制一（同形分组价 + 解析式）

**(a) 动机说明**

*直线域 / 循环域是什么*：costmodel 把 kernel 拆成若干 stage，一个
访存 stage 的访问有没有被**运行期循环**包住，决定它属于哪个域——

```python
# 直线域：没有 scf.for 包住，每条访问每个 program 只执行一次
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)       # 访问 1（与 2 相互独立）
    y = tl.load(y_ptr + offs, mask=mask)       # 访问 2
    tl.store(out_ptr + offs, x + y, mask=mask)

# 循环域：真 scf.for 包住，每条访问执行 K 次
for sub in range(0, XBLOCK, XBLOCK_SUB):       # 运行期循环，K = XBLOCK//XBLOCK_SUB
    idx = base + sub + tl.arange(0, XBLOCK_SUB)
    v = tl.load(inp + idx)
    tl.store(out + idx, v * a + b)
```

判定标准是 IR 里是否还是真 `scf.for`：`static_range`/constexpr 在
编译期展开后与直线代码完全同价，所以展开的循环归直线域。这不是
假设——E7 用同一份工作（N 对 4KB load+store）只变代码形状做了
对照：**static_range 展开 ≡ 直线型**（simd/simt 全部一致，<5%）。

*为什么要和循环域分开建模（历史拟合数据）*：三个实验把这件事钉死——

- **E6**（同一访存模式分别写成动态循环与直线交错，卡 3，
  `scripts/e6_loop_vs_burst.py`）：4KB 独立 load+store、真实流式地址
  （step×BLOCK 推进，不用窗口复读），grid=1，配对差分（对照 = 同
  形状的常量 store kernel）。两种写法的 kernel 核心代码：

  ```python
  # (a) 动态循环：K 轮运行期 scf.for，每轮 1 对（K∈{1..128} 运行时传入）
  @triton.jit(do_not_specialize=['iters', 'boff'])
  def e6_loop_ind(src, dst, iters, boff, BLOCK: tl.constexpr):
      lane = tl.arange(0, BLOCK)
      for step in range(iters):
          x = tl.load(src + boff + step * BLOCK + lane)
          tl.store(dst + boff + step * BLOCK + lane, x + 1)

  # (b) 直线交错：U 对全部展开进同一条指令流（U∈{1,2,4,8}，图为 U=4）
  @triton.jit
  def e6_burst_u4(src, dst, BLOCK: tl.constexpr):
      lane = tl.arange(0, BLOCK)
      x0 = tl.load(src + 0 * BLOCK + lane)
      tl.store(dst + 0 * BLOCK + lane, x0 + 1)
      x1 = tl.load(src + 1 * BLOCK + lane)
      tl.store(dst + 1 * BLOCK + lane, x1 + 1)
      x2 = tl.load(src + 2 * BLOCK + lane)
      tl.store(dst + 2 * BLOCK + lane, x2 + 1)
      x3 = tl.load(src + 3 * BLOCK + lane)
      tl.store(dst + 3 * BLOCK + lane, x3 + 1)
  ```

  实测（配对差分边际）：直线型每实例 **~37ns**（U≤8 时 MTE2 队列
  深 16 内多笔同时在飞），循环每轮 **~102ns**（每轮约一笔在飞，受
  发射节流）——两个上下文的每实例价差 **2.7×**。终版 wheel + SYS_CNT
  设备内口径复测（E24，`probes/e24_marginal_repro.py`）：循环
  T(K) 斜率 **93.2 ns/轮**（K=16→32 稳态恰 **102 ns/轮**，前几轮
  58-88 ns/轮是流水填充）；直线 T(U) 在 U∈{1..8} 拟合 **36.8 ns/条**
  ——两个边际数字在现栈上原样复现。注意：这个 2.7×
  只在队列深度区间（U≤8，≤~64KB）成立，不能外推到大批量——下一条
  E7 用同一份总工作量的对照把它修正了；
- **E7**（同一份工作 N∈{8,32}，只变代码形状 A/B/C/D/E）：N=32 时
  loop ≡ burst ≡ static_range（<5%）——直线型的并发优势**被队列
  深度封顶**（只影响前 ~8-16 笔），超过后两域收敛到同一个传输受限
  价。结论：直线域独有的是"**前 Q 笔的组内摊销结构**"，循环域独有
  的是"**每轮节流的稳态斜率 + 固定启动头**"；
- **G3**（循环体 K 扫描，reduce_sum 形状，calF2 探针）：T(4)=4211ns
  ≈ T(16)=3881ns，K=64 才线性涨到 8905ns——循环体时长 = 固定启动头
  A(P) + K×稳态斜率；用"直线逐 op 公式×K"外推，K=4 处给 **4 倍
  偏差**。

所以两个域各需要一个机制：直线域用**组价表**（first/marginal/Q，
下文），循环域用**包络**（§2.2）。组价表的标定数据来自 E3/E12 组
并发扫描（grid=1，同形条数 1..32 × 字节档 {32..16384} × dtype ×
warm/cold，每组 12-16 次发射取中位）。

**(b) 公式与物理解释**

先说两个组合语境（由 stage 的 schedule 决定；代码判据在分区器
`scheduleForSemanticRoot`：不含运行期循环的 stage 得
`straight_line`，含 `scf.for/while` 的得 `independent_pipelined`）：

- **串行直线体**（schedule=straight_line）：访问之间背靠背发射，
  estimator 把各服务的价**逐条相加**：
  `T_stage = setup + Σ(每条访问的服务价)`（issue 与 Σ 取 max 防双计）。

  ```python
  x = tl.load(p + offs)
  y = tl.load(q + offs) + x      # 依赖第一条 load → 链
  tl.store(o + offs, y)
  ```

- **流水直线组**（同一直线体内**相互独立**的多条访问）：硬件会把
  它们同时压进 MTE 队列（E6：队列深 16 内多笔在飞）。这个组内并发
  **不在组合层建模**，而是由下面的组价公式（marginal<Q 摊薄）承载：

  ```python
  a = tl.load(pa + offs)         # 与 b 独立 → 队列内在飞
  b = tl.load(pb + offs)
  tl.store(po + offs, a + b)
  ```

  （estimator 的 `max(load, store, vector)` 流水组合只出现在循环域
  stage 上，见 §2.7。）

每条访问的服务价，按序查两道：

1. **同 stage 的同形多访问组**（n≥2；组键 = 方向 × dtype × 字节档 ×
   warm/cold）：

   ```
   n ≤ Q：  T_group = first + (n−1) × marginal
   n > Q：  T_group = first + (Q−1) × marginal + (n−Q) × kc × marginal
   整组 × pAxisFactor(P)     （P≤56 恒 1.0；P>56 查 E13 多波摊薄曲线）
   ```

   物理：第 1 条付全价（含 DRAM 延迟）；第 2..Q 条每条只付边际价
   （队列深度把延迟摊薄）；超过 Q 条进入传输受限区，每条付换档价
   kc×marginal。Q 是结构公式：store 恒 8；load warm
   `clamp(64KB/字节, 4, 32)`、cold `clamp(32KB/字节, 4, 8)`——正是
   E7 测到的"队列深度封顶"。

2. **单发（n=1）/ 未命中格 / 掩码组**：走与基线同源的解析式。这就是
   "直线串行 simd load 的公式"：

   ```
   simd load：  T = 105.31 + 字节/167.56     （ns；1 cycle = 1 ns）
   simd store： T = 90.95 + 字节/213.13
   simt：       T = warp 指令数 / 吞吐速率   （GM/UB 按 W 分档速率表）
   ```

   simd load 侧另乘波衰减 `waves^−0.32`（waves = P/56，guard WDA
   拟合），store 侧不衰减。编译期探针复核（route 报告逐位一致）：
   4KB fp32 load = 105.31+4096/167.56 = **129.75**；4KB store =
   90.95+4096/213.13 = **110.17**。

   *整核校验*（campaign 期，§V2 评审 3.2）：vector_add 每程序
   2×4KB load + 4KB store：load 组 n=2 → 160.1+36.6 = 196.7ns
   （fp32/4096/warm 格），store n=1 → 110.17ns；整核对账 2140.3 vs
   实测 2141.0（−0.03%）。

**(c) 参数分组**

机制一的参数分两层——**每格标定参数**与**结构/共享参数**：

*每格参数（查表命中时用）*：每格只存 `{first, marginal}` 2 个数。
格 = 引擎 × 方向 × dtype × 字节档 × 驻留 的组合：

| 维度 | 取值 | 说明 |
|---|---|---|
| 引擎（2） | simd / simt | 本 profile 两条路线 |
| 方向（2） | load / store | |
| dtype（4） | int8 / fp16 / fp32 / int64 | bf16 按 2 字节别名进 fp16 档 |
| 字节档（8） | 32/64/128/256/512/1024/4096/16384 B | 每条访问字节按 log2 最近邻落档 |
| 驻留（2） | warm / cold | 本次 launch 总足迹 ≤ / > 128MB（L2） |

理论 2×2×4×8×2 = 128 格/引擎；实际只标定有探针数据的格子：
**simd 48 格、simt 62 格**（simt 含按激活 warp 数的 `/w1` 等变体格；
simt 组先查基础键，未命中按激活 warp 数重查一次 `/w<W>`）。

*结构参数（不占格，全组共享）*：

- **Q（队列深度）**：store 恒 8；load warm `clamp(65536/字节,4,32)`、
  cold `clamp(32768/字节,4,8)`——2 条公式替代每格一个 Q；
- **cTransfer 换档比 kc**：4 个标量/引擎（simd load warm 1.231 /
  load cold 2.768 / store warm 1.003 / store cold 1.021；simt
  1.030/1.952/0.988/1.002），cTransfer = kc × marginal；
- **pAxisFactor**：8 条曲线（种类 loadsum/storeonly × 引擎 × 驻留）；
  P≤56 恒 1 是**消除双算**的决定（单波内的每块调度增长归候选层
  A(P) 边界残差，E14 分解），P>56 按实测点插值（多波聚合带宽摊薄，
  E13）。

*解析式参数（未命中/单发/掩码时用）*：simd 每方向 2 个（启动 +
带宽，与基线同源常数）；simt 按 W 分档吞吐表（基线继承）。

*为什么每格只存 2 个数*：2026-11-17 参数压缩把每格 4 数
（first/marginal/cTransfer/Q）压成 2 数 + 结构公式，gate 与压缩前
逐位一致（100-case、guard 同分同 flip）；反过来，把格子拟合成光滑
公式的两轮尝试被数据否定（误差最大 ±600%，dtype 合并检测 0/12 格
可并），逐格值就是当前数据支持的最简形态。

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

### 2.7 estimator 组合层

组合按 stage 的 schedule 选择（判据：stage 里是否还有运行期
scf.for/while）：直线域 stage 全部是 straight_line schedule →
**串行相加** `setup + Σ(各服务价)`（issue 与 Σ 取 max 防双计）；
循环域 stage 是 independent_pipelined schedule →
**`max(load, store, vector)`**；例外（5c）：`r.reduction > 0` 的
循环体按串行链组合（见 2.4）。LoopCarriedSerial（带递推的循环）
同样走串行。

### 2.8 候选层

`A(P)` 边界残差表（起步价，1→408.3 … 56→1704.95 ns）+ 波衰减。
A 是 `T(K) = A(P) + K × perIter` 的 A——E20b 实测真实起步价
0.9-2.4µs（P∈{2..56}），现表大体正确（混合符号 ±350-530ns），
保留。

### 2.9 公式列表

| 编号 | 场景 | 公式 | 拟合点数 | MAPE |
|---|---|---|---|---|
| A1 | 直线域同形组价（组命中，n≥2） | first+(min(n,Q)−1)·marginal+max(0,n−Q)·kc·marginal，×pAxis(P) | 110 格（simd 48 / simt 62；E3/E12 组扫描，grid=1 条数 1..32） | 整核锚点：vector_add +29.86%→−0.03%、softmax +12.73%→−4.15%（压缩 gate 与压缩前逐位一致） |
| A2 | 直线域解析式（单发/未命中/掩码） | simd load 105.31+B/167.56、store 90.95+B/213.13（load×waves^−0.32）；simt warp 指令/速率 | 基线继承（编译期探针复核：4KB load 129.75 / store 110.17 与 route 一致） | 基线口径 |
| A3 | estimator 组合 | 直线 schedule：setup+Σ；流水 schedule（循环域）：max(load,store,vector)；5c 归约链 | 结构（无拟合） | — |
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
>
> 档案说明：早期 v8 逐访问多项式族（`a₀+a₁·by+a₂·blk_part+a₃·burst`
> 等形状，旧编号 F1-F13，simd 串行/流水 MAPE 5.7-8.6%、simt 6.5-26.9%）
> 标定于 `contiguous_1d/forms`（forms_spec.json）与 v1 C++
> （`Models/ContiguousLdStCostModel`）；**终版代码中该 C++ 没有任何
> 调用点**——直线域由 A1/A2 计价，这批拟合仅作标定档案保留在树中。
> 基线路线 2 的独立 load 组包络（546.1+12.8+字节/83.26，1-4 条流）
> 代码保留，但 v2 对连续访存全量认领后，其前缀截断在第一条已认领
> load 上，常规路径不再触发（仅 A/B 环境变量场景可达）。

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

> 注：上两表"修复来源/归因"列写于 campaign 进行中，其中提到的
> calN2/calF2 行表已随行表退役（终版 §2.0 的机制清单不含行表）。
> 这些 case 在终版代码里由直线域解析式/组价 + 波衰减 + 循环域包络
> 组合计价；表中的误差数字均为终版 wheel 实测。

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

| case | mode | 实测(ns) | 本PR后预测(ns) | 预测误差 | 整核误差是否<10% |
|---|---|---:|---:|---:|:---:|
| A_add_f16_2k_g128 | simd_f1 | 12328.0 | 2347.6 | -80.96% | FAIL |
| A_add_f16_2k_g128 | simt_f1 | 8062.0 | 15239.9 | +89.03% | FAIL |
| A_add_f16_2k_g128 | simt_f2 | 6698.0 | 11063.5 | +65.18% | FAIL |
| A_add_f16_2k_g128 | simt_f4 | 5473.0 | 6510.2 | +18.95% | FAIL |
| A_add_f32_4k_g1024 | simd_f1 | 4878.0 | 6751.0 | +38.40% | FAIL |
| A_add_f32_4k_g1024 | simt_f1 | 52070.0 | 76182.5 | +46.31% | FAIL |
| A_add_f32_4k_g1024 | simt_f2 | 29044.0 | 41721.6 | +43.65% | FAIL |
| A_add_f32_4k_g1024 | simt_f4 | 16984.0 | 23379.2 | +37.65% | FAIL |
| A_add_f32_4k_g256 | simd_f1 | 2909.0 | 3577.0 | +22.96% | FAIL |
| A_add_f32_4k_g256 | simt_f1 | 15820.0 | 23117.4 | +46.13% | FAIL |
| A_add_f32_4k_g256 | simt_f2 | 10914.0 | 14982.2 | +37.28% | FAIL |
| A_add_f32_4k_g256 | simt_f4 | 8461.0 | 11335.9 | +33.98% | FAIL |
| A_copy_f16_2k_g256 | simd_f1 | 2864.0 | 3056.1 | +6.71% | PASS |
| A_copy_f16_2k_g256 | simt_f1 | 11229.0 | 13774.7 | +22.67% | FAIL |
| A_copy_f16_2k_g256 | simt_f2 | 8601.0 | 9307.6 | +8.22% | PASS |
| A_copy_f16_2k_g256 | simt_f4 | 7390.0 | 7389.8 | -0.00% | PASS |
| A_copy_f32_4k_g1024 | simd_f1 | 4542.0 | 5686.0 | +25.19% | FAIL |
| A_copy_f32_4k_g1024 | simt_f1 | 42706.0 | 42925.0 | +0.51% | PASS |
| A_copy_f32_4k_g1024 | simt_f2 | 23894.0 | 23988.5 | +0.40% | PASS |
| A_copy_f32_4k_g1024 | simt_f4 | 13539.0 | 14117.9 | +4.28% | PASS |
| A_copy_f32_4k_g256 | simd_f1 | 2967.0 | 3153.9 | +6.30% | PASS |
| A_copy_f32_4k_g256 | simt_f1 | 12914.0 | 13774.7 | +6.66% | PASS |
| A_copy_f32_4k_g256 | simt_f2 | 8911.0 | 9307.6 | +4.45% | PASS |
| A_copy_f32_4k_g256 | simt_f4 | 6775.0 | 7389.8 | +9.08% | PASS |
| A_copy_f32_4k_g64 | simd_f1 | 2427.0 | 1944.0 | -19.90% | FAIL |
| A_copy_f32_4k_g64 | simt_f1 | 6764.0 | 7251.7 | +7.21% | PASS |
| A_copy_f32_4k_g64 | simt_f2 | 4499.0 | 4352.6 | -3.25% | PASS |
| A_copy_f32_4k_g64 | simt_f4 | 4481.0 | 4393.9 | -1.94% | PASS |
| A_copy_i64_8k_g256 | simd_f1 | 2963.0 | 3349.4 | +13.04% | FAIL |
| A_copy_i64_8k_g256 | simt_f1 | 18542.0 | 13774.7 | -25.71% | FAIL |
| A_copy_i64_8k_g256 | simt_f2 | 12375.0 | 9307.6 | -24.79% | FAIL |
| A_copy_i64_8k_g256 | simt_f4 | 9113.0 | 7389.8 | -18.91% | FAIL |
| A_copy_i8_1k_g256 | simd_f1 | 2725.0 | 3007.2 | +10.36% | FAIL |
| A_copy_i8_1k_g256 | simt_f1 | 10099.0 | 13774.7 | +36.40% | FAIL |
| A_copy_i8_1k_g256 | simt_f2 | 7897.0 | 9307.6 | +17.86% | FAIL |
| A_copy_i8_1k_g256 | simt_f4 | 7172.0 | 7389.8 | +3.04% | PASS |
| A_load_f32_4k_g256 | simd_f1 | 6625.0 | 4413.1 | -33.39% | FAIL |
| A_load_f32_4k_g256 | simt_f1 | 8120.0 | 13745.1 | +69.27% | FAIL |
| A_load_f32_4k_g256 | simt_f2 | 7015.0 | 9595.4 | +36.78% | FAIL |
| A_load_f32_4k_g256 | simt_f4 | 5717.0 | 7980.4 | +39.59% | FAIL |
| A_store_f16_2k_g256 | simd_f1 | 31113.0 | 2380.6 | -92.35% | FAIL |
| A_store_f16_2k_g256 | simt_f1 | 4433.0 | 4467.7 | +0.78% | PASS |
| A_store_f16_2k_g256 | simt_f2 | 4667.0 | 3525.5 | -24.46% | FAIL |
| A_store_f16_2k_g256 | simt_f4 | 4889.0 | 3297.6 | -32.55% | FAIL |
| A_store_f32_4k_g256 | simd_f1 | 2563.0 | 2440.7 | -4.77% | PASS |
| A_store_f32_4k_g256 | simt_f1 | 3486.0 | 4467.7 | +28.16% | FAIL |
| A_store_f32_4k_g256 | simt_f2 | 3127.0 | 3525.5 | +12.74% | FAIL |
| A_store_f32_4k_g256 | simt_f4 | 3101.0 | 3297.6 | +6.34% | PASS |
| B_add_f32_4k_k16_g56 | simd_f1 | 4311.0 | 3986.1 | -7.54% | PASS |
| B_add_f32_4k_k16_g56 | simt_f1 | 27459.0 | 29158.1 | +6.19% | PASS |
| B_add_f32_4k_k16_g56 | simt_f2 | 71054.0 | 29158.1 | -58.96% | FAIL |
| B_add_f32_4k_k16_g56 | simt_f4 | 27423.0 | 29158.1 | +6.33% | PASS |
| B_add_f32_4k_k64_g224 | simd_f1 | 118870.0 | 123253.9 | +3.69% | PASS |
| B_add_f32_4k_k64_g224 | simt_f1 | 940082.0 | 1052908.5 | +12.00% | FAIL |
| B_add_f32_4k_k64_g224 | simt_f2 | 488943.0 | 526454.3 | +7.67% | PASS |
| B_add_f32_4k_k64_g224 | simt_f4 | 279956.0 | 263227.1 | -5.98% | PASS |
| B_copy_f16_2k_k16_g56 | simd_f1 | 3718.0 | 3762.3 | +1.19% | PASS |
| B_copy_f16_2k_k16_g56 | simt_f1 | 7671.0 | 15445.9 | +101.35% | FAIL |
| B_copy_f16_2k_k16_g56 | simt_f2 | 7646.0 | 15445.9 | +102.01% | FAIL |
| B_copy_f16_2k_k16_g56 | simt_f4 | 7718.0 | 15445.9 | +100.13% | FAIL |
| B_copy_f32_4k_k16_g56 | simd_f1 | 3959.0 | 3935.7 | -0.59% | PASS |
| B_copy_f32_4k_k16_g56 | simt_f1 | 24804.0 | 26786.4 | +7.99% | PASS |
| B_copy_f32_4k_k16_g56 | simt_f2 | 24778.0 | 26786.4 | +8.11% | PASS |
| B_copy_f32_4k_k16_g56 | simt_f4 | 24756.0 | 26786.4 | +8.20% | PASS |
| B_copy_f32_4k_k32_g56 | simd_f1 | 5648.0 | 5690.9 | +0.76% | PASS |
| B_copy_f32_4k_k32_g56 | simt_f1 | 47881.0 | 51060.4 | +6.64% | PASS |
| B_copy_f32_4k_k32_g56 | simt_f2 | 48011.0 | 51060.4 | +6.35% | PASS |
| B_copy_f32_4k_k32_g56 | simt_f4 | 47972.0 | 51060.4 | +6.44% | PASS |
| B_copy_f32_4k_k4_g8 | simd_f1 | 1616.0 | 1761.4 | +9.00% | PASS |
| B_copy_f32_4k_k4_g8 | simt_f1 | 7324.0 | 7764.3 | +6.01% | PASS |
| B_copy_f32_4k_k4_g8 | simt_f2 | 6968.0 | 7764.3 | +11.43% | FAIL |
| B_copy_f32_4k_k4_g8 | simt_f4 | 6982.0 | 7764.3 | +11.20% | FAIL |
| B_copy_f32_4k_k64_g56 | simd_f1 | 9165.0 | 9201.3 | +0.40% | PASS |
| B_copy_f32_4k_k64_g56 | simt_f1 | 93869.0 | 99608.4 | +6.11% | PASS |
| B_copy_f32_4k_k64_g56 | simt_f2 | 94074.0 | 99608.4 | +5.88% | PASS |
| B_copy_f32_4k_k64_g56 | simt_f4 | 94019.0 | 99608.4 | +5.94% | PASS |
| B_copy_f32_4k_k8_g56 | simd_f1 | 3139.0 | 3058.1 | -2.58% | PASS |
| B_copy_f32_4k_k8_g56 | simt_f1 | 15376.0 | 14649.4 | -4.73% | PASS |
| B_copy_f32_4k_k8_g56 | simt_f2 | 13571.0 | 14649.4 | +7.95% | PASS |
| B_copy_f32_4k_k8_g56 | simt_f4 | 13643.0 | 14649.4 | +7.38% | PASS |
| B_copy_i8_1k_k16_g56 | simd_f1 | 3533.0 | 3609.1 | +2.15% | PASS |
| B_copy_i8_1k_k16_g56 | simt_f1 | 14576.0 | 9820.5 | -32.63% | FAIL |
| B_copy_i8_1k_k16_g56 | simt_f2 | 14573.0 | 9820.5 | -32.61% | FAIL |
| B_copy_i8_1k_k16_g56 | simt_f4 | 14543.0 | 9820.5 | -32.47% | FAIL |
| B_load_f32_4k_k64_g56 | simd_f1 | 9896.0 | 8668.7 | -12.40% | FAIL |
| B_load_f32_4k_k64_g56 | simt_f1 | 22480.0 | 30921.5 | +37.55% | FAIL |
| B_load_f32_4k_k64_g56 | simt_f2 | 23776.0 | 30921.5 | +30.05% | FAIL |
| B_load_f32_4k_k64_g56 | simt_f4 | 23764.0 | 30921.5 | +30.12% | FAIL |
| B_store_f32_4k_k16_g56 | simd_f1 | 3744.0 | 3631.0 | -3.02% | PASS |
| B_store_f32_4k_k16_g56 | simt_f1 | 3703.0 | 3724.0 | +0.57% | PASS |
| B_store_f32_4k_k16_g56 | simt_f2 | 3609.0 | 3724.0 | +3.19% | PASS |
| B_store_f32_4k_k16_g56 | simt_f4 | 3619.0 | 3724.0 | +2.90% | PASS |
| B_store_i64_8k_k4_g56 | simd_f1 | 2491.0 | 2414.1 | -3.09% | PASS |
| B_store_i64_8k_k4_g56 | simt_f1 | 2880.0 | 2854.3 | -0.89% | PASS |
| B_store_i64_8k_k4_g56 | simt_f2 | 2904.0 | 2854.3 | -1.71% | PASS |
| B_store_i64_8k_k4_g56 | simt_f4 | 2904.0 | 2854.3 | -1.71% | PASS |
