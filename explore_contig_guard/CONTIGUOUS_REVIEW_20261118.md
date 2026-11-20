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
└─ 是 → 循环域【E27e 拟合模型】，K = iterationCount ≥ 1
     ├─ simd：perIter = max(stepT[nL+nS], 曲线_nL(lB), 曲线_nS(sB))
     │        + cL×(lB/128) + cS×(sB/128)（混合体行税）
     │        小 store（全 ≤128B）：excess/chain delta 表（按 dom 分键）
     ├─ simt：perIter = 曲线_nL(lB) + E17i排空曲线(sB)（可加，E17g）
     └─ dtype 无拟合 → Tier 3 逐 op 兜底（同直线域解析式）

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

- **E6/E24**（同一访存模式分别写成动态循环与直线交错）：E6 原始探针
  （`scripts/e6_loop_vs_burst.py`，卡 3，配对差分口径）给出直线每
  实例 ~37ns / 循环每轮 ~102ns；下表为**终版 wheel 上的 SYS_CNT
  设备内口径复测**（`probes/e24_marginal_repro.py`，卡 1，4KB fp32
  对、流式地址、grid=1，每点 30 次发射取中位 × 3 遍）。kernel 核心
  代码：

  ```python
  # (a) 动态循环：K 轮运行期 scf.for，每轮 1 对（K 运行时传入，{1..32}）
  @triton.jit(do_not_specialize=['iters'])
  def e24_loop_ind(src, dst, out, tick_ptr, iters, BLOCK: tl.constexpr):
      lane = tl.arange(0, BLOCK)
      t0 = _sys_cnt()                    # SYS_CNT：1 tick = 1.00008 ns
      for step in range(iters):
          x = tl.load(src + step * BLOCK + lane)
          tl.store(dst + step * BLOCK + lane, x + 1)
      t1 = _sys_cnt()                    # 窗口时长 = t1 - t0（设备内）
      tl.store(tick_ptr + 0, t0)
      tl.store(tick_ptr + 1, t1)

  # (b) 直线交错：U 对全部展开进同一条指令流（U∈{1,2,4,8,16}，图为 U=4）
  @triton.jit
  def e24_burst_u4(src, dst, out, tick_ptr, BLOCK: tl.constexpr):
      lane = tl.arange(0, BLOCK)
      t0 = _sys_cnt()
      x0 = tl.load(src + 0 * BLOCK + lane)
      tl.store(dst + 0 * BLOCK + lane, x0 + 1)
      x1 = tl.load(src + 1 * BLOCK + lane)
      tl.store(dst + 1 * BLOCK + lane, x1 + 1)
      x2 = tl.load(src + 2 * BLOCK + lane)
      tl.store(dst + 2 * BLOCK + lane, x2 + 1)
      x3 = tl.load(src + 3 * BLOCK + lane)
      tl.store(dst + 3 * BLOCK + lane, x3 + 1)
      t1 = _sys_cnt()
      tl.store(tick_ptr + 0, t0)
      tl.store(tick_ptr + 1, t1)
  ```

  窗口中位时长与边际（ns；1 对档的窗口被 launch 首访回程延迟主导
  ——E24b/E24c 定位见下——故不列出）：

  | 对数 | 2 | 4 | 8 | 16 | 32 |
  |---|---:|---:|---:|---:|---:|
  | 循环形 T(K) ns | 124 | 244 | 477 | 1156 | 2765 |
  | 循环每轮边际 ns/轮 | — | 60 | 58 | 85 | **101** |
  | 直线形 T(U) ns | 377 | 435 | 544 | 804 | 2428 |
  | 直线每对边际 ns/对 | — | 29 | 27 | 32 | **102** |

  边际结论：循环的每轮边际随流水填满爬升（60→85→101），稳态
  **~102 ns/轮**（K=16→32 边际 101；两次独立复测 100.6-101.9；
  K∈{4..32} 拟合 92）；直线形在队列深度内（U≤16）每对边际平稳于
  **~30±8 ns**（两次复测拟合 28/36）——MTE2 队列深 16 内多笔同时
  在飞，循环每轮约一笔在飞、受发射节流，稳态比值约 **3×**。单对
  （1 对/轮）窗口本身不可用：直线形常量地址的窗口含本 launch 首访
  回程延迟 ~300ns（E24b：t0 前加一条预热 load 后降到 **17ns**；
  E24c：地址改为运行期解析后降到 **40ns**）——是发射形态产物，
  与形状语义无关，边际（斜率）不受影响。

  整核对照（E24d，profiler 口径、无 SYS_CNT，单 kernel 单会话，
  终版 wheel）：同对数下两形状整核差仅 **4-7%**（循环形略快），
  1-2 对时相同（~0.78µs）；K=4 为 0.98 vs 1.07µs、K=32 为 3.9 vs
  4.0µs——两条曲线在大对数处斜率收敛到同一 ~105 ns/对，与 E7 的
  整核结论一致（直线形的固定脚手架成本抵掉边际优势）。

  边界提示：~3× 是**每条边际**之比，只在队列深度区间（U≤16）
  成立——U=16→32 直线每对边际升到 **~102 ns/对**
  （(2428−804)/16），与循环稳态收敛到同一传输受限价（E7 历史
  U=32 外推 ~125 同方向）。
- **E7**（同一份工作，只换代码写法）：固定总访存量 = N 对 4KB
  load+store（N∈{8,32}），把**同样的 N 对**写成四种形状——
  A：运行期 `scf.for` 每轮 1 对、共 N 轮；C：不写循环、N 对直接
  平铺；D：`tl.static_range(N)`（前端展开，应与 C 同 IR）；E：
  运行期循环、每轮 static 展开 4 对——各配一个只 store 的对照
  kernel。结果：**D 与 C 整核中位完全同价**（引擎 × N 全部一致）；
  N=32 时 A=C=D=E（差 <5%）。含义：① static_range 展开后就是直线
  代码（分类判据应该用 IR）；② 直线形的并发优势**被队列深度封顶**
  （只影响前 ~8-16 笔），超过后两域收敛到同一传输受限价。结论：
  直线域独有的是"**前 Q 笔的组内摊销结构**"，循环域独有的是
  "**每轮节流的稳态斜率 + 固定启动头**"；
- **G3**（同一个循环体 kernel，只把循环次数 K 从 4 加到 64）：
  kernel 是 guard 用例 3 的形状（reduce_sum 尾处理体：2KB 大块
  load + 32B bias load + `tl.sum` 行归约 + 32B 小 store），用整核
  探针（calF2，`cal_f2_direct.py`）扫 K∈{4,16,64}：T(4)=4211ns ≈
  T(16)=3881ns（循环次数 ×4、总时长几乎不变），到 K=64 才线性涨到
  8905ns——循环体时长 = 固定启动头 A(P) + K×稳态斜率；用"直线逐
  op 公式×K"外推，K=4 处给 **4 倍偏差**。

所以两个域各需要一个机制：直线域用**组价表**（first/marginal/Q，
下文），循环域用**包络**（§2.2）。组价表的标定数据来自 E3/E12 组
并发扫描（grid=1，同形条数 1..32 × 字节档 {32..16384} × dtype ×
warm/cold，每组 12-16 次发射取中位）。

**(b) 公式与物理解释**

**组合语境：直线域只有一种**。不含运行期循环的 stage 拿到
`straight_line` schedule（代码判据在分区器 `scheduleForSemanticRoot`：
只有含 `scf.for/while` 的 stage 才得 `independent_pipelined`），
estimator 对它**一律串行相加**：`T_stage = setup + Σ(每条访问的
服务价)`（issue 与 Σ 取 max 防双计）。`max(load, store, vector)`
的流水组合只属于循环域 stage（§2.7）。

评审时曾被追问"两条 load 反正都要被加起来，依赖结构不都一样吗"——
是对的，下面两个 kernel 在分区器眼里就是同一类直线访存体，组合
方式完全相同：

```python
x = tl.load(p + offs)
y = tl.load(q + offs) + x      # 依赖链
tl.store(o + offs, y)

a = tl.load(pa + offs)         # a、b 之后同样要相加——依赖结构一样
b = tl.load(pb + offs)
tl.store(po + offs, a + b)
```

直线域的硬件并发（E6/E24：同形多条访问在 MTE 队列深 16 内一齐
在飞，边际 37ns/条）**不由组合层表达**，而由下面的组价公式承载：
同形状访问组内第 1 条付全价、第 2..Q 条只付边际价（队列摊薄）。
实证（route 报告逐位核对）：直线 2 load + 1 store 的 stage 时长 =
setup 36.38 + load 解析价 117.53 = **153.91**（串行相加，精确
吻合）。

每条访问的服务价，按序查两道。先定义"组"与"n"：直线 stage 的全部
连续访问按**同形状**归组（组键 = 方向 × 元素个数 × 元素宽度 × 是否
掩码），**n = 组内访问条数**（不是 load+store 对数）——1 条 4KB
load + 1 条 4KB store 是**两组**、各自 n=1；2 条一样的 4KB load +
1 条 store 是 load 组 n=2 + store 组 n=1。每组再算查表键 =
`方向/dtype/字节档/warm或cold`（字节档 = 本组字节按 log2 最近邻
落入 32..16384 八档之一；warm/cold = 整 stage 足迹 ≤/> 128MB L2）。
**未命中格** = 该键在组价表里没有格（110 个标定格只覆盖有探针数据
的几何）；**掩码组** = 编译期可证部分 lane 活跃的组（表按满字节
标定、组按活跃字节选档，会错配，故不查表）。三种情况（n=1 / 未
命中 / 掩码）都走解析式；**n≥2 且键命中**才走组价公式。注意 n=1
即使键命中也不查 first：first 是在"组内有兄弟访问排队"的上下文里
标定的，孤发一条用它会双算发射开销（代码注释明示；route 报告逐位
验证：4KB load = 129.75）。分组键必要性的判别（E25b，
`probes/e25b_bigpair.py`）：po2 形状里能构造的"同档异形"对是
16KB+32KB（32KB 由钳制落 16384 档）——若按查表键归组，这对并成
16384 行的 n=2（396ns）；现行的精确分组使各自走字节成正比的解析式
（504ns）。profiler 整核实测（2×16KB 锚 1.28µs vs 该对 1.31-1.45µs，
+~100-150ns）**支持现行**——按桶合并会低估 ~100ns，故保留精确形状
分组。E25c 的 SYS_CNT 窗口复核（5 遍同方向）：w(16KB+32KB) 比
w(2×16KB) 稳定贵 **~230-330ns**（现行预测 +108、按桶合并预测 +0）
——方向与幅度都进一步否定按桶合并；现行模型的 +108 反而偏保守。附带发现一个真实缺口：**>16KB 的钳制欠价**——2×32KB 同形组
按 16384 行计价 396ns，实测比 2×16KB 对贵 ~270ns 而 模型给 0
（E12 扫描止步 16KB，大访问档缺标定），列为复核项。

1. **同 stage 的同形多访问组**（n≥2 且键命中）：

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

   适用面澄清：解析式按**每条访问**计价，不限"1 load+1 store"——
   多条访问但查不到组的（dtype/字节档没标定的格子）也逐条用它再
   相加。组价 first 与解析式的分工（数字对比，fp32/warm：4096B 组
   first 160.1 vs 解析 129.75；16384B 组 264.1 vs 203.1）：组上下文
   的第 1 条实测比孤立单发贵 30-60ns——这是 E12 组扫描的直读值，
   vector_add 整核对账（−0.03%）依赖它；"为什么组上下文第 1 条
   更贵"没有独立机制实验，列为复核项。

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
实际覆盖（simd 段逐键清点）：load/warm = fp16+fp32 全 8 档；load/
cold = 4 dtype × {1024,4096,16384}；store warm/cold = 4 dtype ×
{1024,4096,16384}——小档（≤512B）只有 warm load 的 fp16/fp32 有，
int8/int64 的 warm load 整档缺席。落档是**硬分箱**（log2 最近邻 +
端点钳制：\<32B 记 32、\>16KB 记 16384），**不做档间插值**——这点
与循环域包络（TL/TS 曲线做 log2 插值）不同；落到的档若没标定，
即"未命中"，整组走解析式。

*结构参数（不占格，全组共享）*：

- **Q（队列深度）**：store 恒 8；load warm `clamp(65536/字节,4,32)`、
  cold `clamp(32768/字节,4,8)`——2 条公式替代每格一个 Q；
- **cTransfer 换档比 kc**：4 个标量/引擎（simd load warm 1.231 /
  load cold 2.768 / store warm 1.003 / store cold 1.021；simt
  1.030/1.952/0.988/1.002），cTransfer = kc × marginal；
- **pAxisFactor**：曲线族按（种类 loadsum/storeonly × 引擎 × 驻留）
  查表；P≤56 恒 1 是**消除双算**的决定（单波内的每块调度增长归候选
  层 A(P) 边界残差，E14 分解），P>56 按实测点插值（多波聚合带宽摊薄；
  E13：loadsum/simd/warm 在 P=56/112/224 的因子 0.211/0.131/0.084，
  每翻倍 ×0.62，同期实测聚合带宽 350→620→1100 GB/s）。两条已知
  限制：① 因子**与字节量无关**（不按每程序搬运量分档）——"每程序
  搬运量已大、带宽已饱和时衰减应减弱"未被建模；循环侧的对应物
  （波衰减 waves^−0.05，大足迹体上几乎为 1）支持这一直觉；② simt
  直线组实际查不到表（simt 段无 loadsum/storeonly/simt 键）→ 恒
  1.0，simt 多波摊薄未建模。

*解析式参数（未命中/单发/掩码时用）*：simd 每方向 2 个（启动 +
带宽，与基线同源常数）；simt 按 W 分档吞吐表（基线继承）。

*为什么每格只存 2 个数*：2026-11-17 参数压缩把每格 4 数
（first/marginal/cTransfer/Q）压成 2 数 + 结构公式，gate 与压缩前
逐位一致（100-case、guard 同分同 flip）；反过来，把格子拟合成光滑
公式的两轮尝试被数据否定（误差最大 ±600%，dtype 合并检测 0/12 格
可并），逐格值就是当前数据支持的最简形态。

*出处说明（基线 vs 本 PR）*：分组（含掩码位）、查表键、组价表与
Q/kc/pAxis 结构都是**本 PR 新增**——基线代码里不存在任何同形分组
（`loadShapes` 等只是 layout 记账，不参与计价）。"分组"概念的雏形
是基线路线 2 的前缀组包络（开头 1-4 条 FP32 load、单一聚合带宽
83 B/cycle），形状限定与公式都完全不同。直线域真正从基线继承的
只有：解析式常数、组合层，以及休眠的路线 2 常数。
### 2.2 循环域（simd）：E27e 拟合模型

**(a) 动机说明与实验依据**：评审质疑"循环体能否像直线域一样按条数×字节拟合，而不是 4 档钳制近似"，由此触发 E26/E27b/E27c/E28 四轮实验（探针 `e26_loop_group_vs_envelope.py`、`e27b_fit2.py`、`e28_mixed_law.py`，bf16/int8/fp32/int64，grid=1，K=64，SYS_CNT 窗口/K=稳态每-迭代价）：

- **E26 15 点矩阵**：把直线组价公式逐条搬进循环上下文，**系统性高估 +40~90%（MAPE≈65%）**——first/marginal 是直线（burst）排队上下文的标定值，循环体每轮受发射节流，两者不可互换；同时暴露包络自身的缺口：≤512B 循环被 4KB 钳制价高估 16-51%、4 流时低估 ~17%（极小事务端每流 ~11ns 边际）、tiny-store excess 在 2×2KB dom 几何上过罚 ~3.5×。
- **E28 混合律挖掘**（26 个 x load + y store 混合点，MD 依赖 / MI 独立双变体）：① **MD≡MI**——store 数据依赖不改变价格（差 1-8%），"依赖链"假说否定；② 混合 adder 与两方向总字节成正比：`max(L,S) + c_s·(sB/128) + c_l·(lB/128)` 在 fp32 密网格 **MAPE 3.0%**（max 8.5%、可加 11.9%、共享带宽 14.8%）；③ MD/MI 无差 + 逐字节比例 ⇒ 这不是"行结构"效应，物理归属是 **MTE2/MTE3 双队列并行时共享下游（L2 端口/GM 控制器）的逐字节竞争串行化**——store 方向不可隐藏比例（0.37ns/128B）是 load 方向（0.11）的 3.4 倍；κ 常数法（0/0.15/0.275 三档）无法表达这种随字节线性增长的形状，退役。
- **E28b simt 混合律**（drain 修正后）：simt 混合体**超加性**（混合 ≈ 1.7-2.0× 可加），流数不影响、只有总字节有意义——与 E6/E1 的"store 干扰 load"记录一致；simt 组合律确定为可加（E17g）叠加干扰系数，精确形式待锁态（双峰控制）后标定。

**(b) 结构与公式**（代码 `loopFitPerIter`，ContiguousLdStV2CostModel.h）：

```
L = 曲线_nL(lB)     每-nL 实测传输曲线：该条数下逐字节实测点连成的
                    分段线性折线（节点=实测价），nL 介于两档之间按
                    log2 插值，域外沿尾斜率延伸（含 64KB 实测锚点）
S = 曲线_nS(sB)     每-nS store 曲线（同构）
floor = max(stepL[nL], stepS[nS], stepT[nL+nS])
        发射节流地板：stepL/stepS = 单方向小字节体的每轮发射墙；
        stepT = 混合体联合发射墙（nL+nS 个描述符一起算，小字节
        混合点直读：{3:97, 4:97, 5:102, 6:120, 8:146}）
base（混合体）= max(floor, L, S) + cL×(lB/128) + cS×(sB/128)
base（单方向）= max(floor, 曲线)
tiny store（全 store ≤128B）：
    非归约 fed：base = max(stepT[nL], L) + excess表[dom](nS)
    归约 fed：  base = max(stepT[nL], L) + chain表[dom](nS)
    excess/chain = 按 dom 字节分键的 nS delta 表（小字节混合点直读，
    分段线性），替代原 97ns 常数 × nStores
```

**三个成分逐个解释**：

- **L = 曲线_nL(lB)（每-nL 实测传输曲线）**：对每个 load 流数
  nL，单独测一族"只含 load 的循环体"，扫每流字节档得到实测点列，
  连成分段线性折线。fp32 实例（nL=1）：32B→54.4、256B→55.0、
  1KB→65.7、4KB→78.7、8KB→92.4、16KB→140.0 ns/迭代。它回答：
  "一个循环体每迭代用 nL 条 load 搬 lB 字节，load 侧稳态要多久"。
  覆盖：grid=1、热 L2、每方向 ≤32KB（寄存器上限）、nL 1-8；域外
  沿尾斜率延伸（fp32 已补 64KB 实测锚点 512.2ns）。
- **S = 曲线_nS(sB)（每-nS store 曲线）**：同构，对"只含 store 的
  循环体"按 nS∈{1,2,4} 各测一条（simd 实测；simt 侧用 E17i 排空
  率曲线——我们的窗口协议测不到 simt store 排空）。
- **floor = max(stepL[nL], stepS[nS], stepT[nL+nS])（发射节流地板）**：
  即使字节→0，循环每轮也要把 nL+nS 条访存描述符从 Main Scalar
  发射出去（架构手册：Scalar 把 MTE 指令发射进各指令队列），这个
  发射动作有每轮成本，三个已测下界取最紧的一个：
  - stepL[nL]：只含 nL 条 load 的小字节体实测地板（fp32:
    {1:55, 2:59, 4:93, 8:191}——nL=8 时 191ns 就是 8 条指令的
    纯发射墙）；
  - stepS[nS]：只含 store 的同理；
  - stepT[nL+nS]：混合体联合发射墙——nL+nS 条描述符共享同一个
    标量发射前端，一起发射比单方向更贵（小字节混合点直读：
    {3:97, 4:97, 5:102, 6:120, 8:146}）。
  取 max = 取已知最紧下界，防止 L/S 曲线在字节很小时插值出低于
  发射墙的价。

**公式走向（确认评审的理解）**：所有公式得到的都是**每迭代时间**；
整核 T = A(P) + K × perIter（候选层）。分支规则：
- 全 load 或全 store → 单方向公式（perIter = 那个方向的曲线；
  曲线的小字节端已含各自的发射地板）；
- 混合体（又有 load 又有 store，store 非 tiny）→ base(混合体)
  公式（max + 行税）；
- 混合体且全 store ≤128B → 在 base 上叠 tiny-store 修正——
  "大 load 传输期间流水线能藏掉前 C 条小 store（C 随主导 load
  尺寸增大：16KB 藏 2 条、1KB 只藏 1 条），超出部分每条每迭代
  付实测往返差价"（fp32 实例：dom=16KB 时第 1/2 条免费、第 4 条
  +55ns；dom=1KB 时第 1 条就 +24.2ns）。
- 精确化：tiny-store 修正**只在混合体触发**——纯 store 且全 tiny
  的循环不在标定域内（隐藏现象需要大 load 存在），走 S 曲线
  外推/Tier 3。

**物理对应**：max(floor, L, S) = 每轮发射节流墙 vs 传输/回程时间取瓶颈（E6：每轮约一笔在飞）；cL/cS 项 = 双队列并行时共享下游（L2 端口/GM 控制器）的逐字节竞争串行化（E28：MD≡MI 证明与数据依赖无关）；excess/chain 表 = 小 store 完成往返超出流水隐藏容量的几何敏感部分（E18b/E19 现象保留、常数改几何表）。

**(c) 参数量与拟合质量**：每 dtype 每引擎 ≈40-60 个数（曲线节点=实测直读，无光滑拟合系数——组价表两轮光滑拟合被否 ±600% 的教训沿用）。样内 MAPE（fp32 simd）：全域 **6.9%**、loads 6.6%、mix 5.3%、tiny 12.5%（四 dtype 全域 6.9-12.4%）；对照退役包络在同矩阵 29.8-42.9%。

**(d) 已知限制（如实记录）**：① 拟合条件为 grid=1 热 L2 缓冲；真实 kernel 是 grid=56 + DRAM 流式——同体实测 188 vs **597** ns/迭代（E28c，3.2×），memory_stream_l4 因此 -22.4%（唯一 flip，见 §3.1）；② 混合 adder 的 (sB, nS, P) 曲面只标了 P=1 切片，P 维标定列为下一场战役；③ simt store 排空率沿用 E17i（我们的窗口测不到 simt store 排空），simt 混合超加性的定量形式待锁态（E8 双峰控制）后标定。

### 2.3 循环域（simt）：load 曲线 + E17i 排空曲线（可加组合）

**(b) 结构与公式**（simt 分支，`loopFitPerIter` 的 additive 路径）：

```
L = 曲线_nL(lB)       simt load 传输曲线：每 nL∈{1,2,4} 一条分段线性
                      实测折线（drain-forced 探针：窗口内放 dependent
                      sink + barrier 强制 load 完成——simt 的 load
                      异步发射，不等完成窗口就关了，E17g 配方）
S = E17i排空曲线(sB) = (sB/128) × line_ns(sB)
                      simt store 走 GM → SIMT DCache(128B 行) → 线程
                      寄存器，每行回填按 E17i 实测排空率计（L2 档：
                      4KB 6.22 / 8KB 6.46 / 16KB 6.08 / 32KB 6.08
                      ns/行）

perIter = L + S                     可加组合（E17g，grid=56 口径）；
                                    ⚠ 见下方 E28b：grid=1 实测为
                                    超可加（混合 ≈1.7-2.0× 可加），
                                    组合律未定（标定复核项）

tiny store（全 store ≤128B）：
    perIter = max(stepL[nL], stepT[nL], L) + excess/chain表[dom](nS)
    （load-only base 换成 tiny delta 表；表与 simd 同构——按主导
     load 字节分键、nS 分段线性）
```

**E28b 判别实验：混合体的组合律（kernel 代码 + 实测表）**。评审
质疑"perIter = L+S 是否成立"，专门测了 simt 混合体（fp32、
grid=1、K=64、clock64 hook、回读排空，3 遍 ±0.1ns；探针
`e28b_simt_mix.py`）。kernel 核心代码：

```python
# 混合体（1L+1S 例）：每迭代 1 条 load + 1 条 store，流式地址
@triton.jit
def e28b_M1_1_1024(src1, src2, dst1, dst2, out, tick_ptr,
                   iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, 1024)
    t0 = _sys_cnt()
    for step in range(iters):
        p = step * 1024 + lane
        a0 += tl.load(src1 + p)          # 4KB fp32 load
        a1 += tl.load(src2 + p)          # 4KB fp32 load
        tl.store(dst1 + p, a0 + 1.0)     # 4KB store（依赖 load）
        tl.store(dst2 + p, a1 + 2.0)     # 4KB store（依赖 a1）
    t1 = _sys_cnt()
```

实测表（ns/迭代，3 遍中位）：

| 体（每迭代） | 总字节 | 实测 |
|---|---:|---:|
| 1L（4KB load） | 4KB | 529.7 |
| 1S（4KB store） | 4KB | 203.9 |
| 1L+1S | 8KB | **1268.4** |
| 2L+1S | 12KB | 1269.0 |
| 1L+2S | 12KB | 1268.0 |
| 2L+2S | 16KB | 1269.1 |
| 1L+1S（16KB 每条） | 32KB | 5164.7 |
| 2L+2S（16KB 每条） | 64KB | 5165.7 |

**三个实测事实**：① 流数完全不影响——同字节数下 (1L,1S)/(2L,1S)/
(2L,2S)/(1L,2S) 四种组合差 <0.2%；② 混合体对总字节**严格线性、
过原点**：8KB→1268.7、32KB→5165.2，即 `perIter = (lB+sB)/6.3 B/ns`
（grid=1 共享带宽律）；③ 对可加组合的比值 = 1.73×(4KB)/2.01×
(16KB)——**超可加，且随字节增大更超**。物理：simt 单程序（grid=1
低占空）load 与 store 都各自被 GM 延迟/排空支配，混流时 store 逐
出 SIMT DCache 里 load 要用的行（E6/E1 记录的超加性），总速率由
共享带宽封顶。

**对公式的影响（如实）**：shipped 可加组合（E17g grid=56 口径）
在本探针条件下低估混合体 42-50%；但组合律的正确形式（共享带宽
律 vs 可加 vs 其他）**依赖发射条件（grid/驻留/双峰状态），尚未
定式**——列为标定复核项；可加组合是保守且与 E17g 同源的现行
选择。

数据通路对照（架构手册）：simd 是 GM → UB（MTE2/MTE3 直传，两通道
可并行）；simt 是 GM → SIMT DCache（128B 行回填）→ 线程寄存器，
**load 与 store 串行**——所以 simt 用可加而非 max。

**(c) 拟合质量与参数**：load 曲线四 dtype MAPE **1.0-16.5%**（逐
nL 实测节点直读）；store 侧 = E17i 数据本身（我们的窗口协议测不到
simt store 排空，故沿用其 grid=56 标定）。参数量每 dtype ≈3 条
load 曲线（各 2-5 节点）+ store 排空表 + excess/chain dom 表。

**(d) 已知限制**：① store 排空率沿用 E17i 的 grid=56 标定（本
campaign 的窗口协议在 simt 下只测到发射、测不到排空）；② simt
混合体的**超加性**——本 campaign grid=1 实测混合 ≈ 1.7-2.0×
可加（E28b：1L+1S@4KB 实测 1270 vs 可加 736ns）——尚未进入公式，
现行可加组合在混合体上偏乐观，定量形式待锁态（E8 双峰控制）后
标定；③ **SYS_CNT 窗口在 simt 混合体上饱和（计数器伪影，E28c
profiler 交叉验证）**：8/12/12/16KB 四种混合体的 SYS_CNT 读数
完全相同（1268-1269ns/迭代），而 profiler 整核明确随字节线性
增长（25.9/38.5/29.5/50.2/101.4/200.9µs，~3.2ns/KB），且同字节
下 load 方向比 store 方向贵 ~30%（4KB load+8KB store 29.5 vs
8KB load+4KB store 38.5）——simt 混合体的 SYS_CNT 读数不可用于
字节标度判断，组合律未定式（profiler 相对值有效、绝对值膨胀
~20×）；④ simt 会话级双峰（E7/E8，同 kernel 相邻会话 2.8×）
未归因，影响测量口径不影响公式形态。原 E17h2/E17i 逐 line 查表
退役，其排空率数据以曲线形式保留在 store 侧。

### 2.4 循环域公式小结

| 编号 | 公式 | 拟合点数 | MAPE |
|---|---|---|---|
| LF1 | 每-nL 传输曲线（分段线性） | 每 dtype 12-20 点 | 6.6-10.9% |
| LF2 | 每-nS store 曲线（simd） | 每 nS 3 点 | ≤0.7% |
| LF3 | 联合发射地板 stepT[nL+nS] | 小字节混合点直读 | floor 项 |
| LF4 | 混合行税 cL·lB/128 + cS·sB/128 | LSQ（13-18 混合点） | fp32 mix 5.3% |
| LF5 | tiny excess/chain dom 表 | 每 dom 3 点 × 3 dom | tiny 9.6-21.4% |
| LF6 | simt：load 曲线 + E17i store 曲线可加 | 同 LF1/LF2 数据 | loads 1.0-16.5% |

### 2.5 循环域公式：与直线域的结构对照

直线域 = 1 个解析式（单发/未命中/掩码）+ 2 个组价分支（n≤Q / n>Q）；循环域 = 逐条数传输曲线 + 联合发射地板 + 行税 + tiny delta 表。直线域的复杂度在"分组与查表"，循环域的复杂度在"曲线与修正"；两者共享候选层 A(P)、组合层与 Tier 3 兜底。

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
| L1 | 循环域 simd：每-nL 传输曲线 | 逐 nL 分段线性实测节点（16B..64KB，含 64KB 锚点） | 每 dtype 12-20 点（E27b/E27c 扫描） | 载入域 6.6-10.9% |
| L2 | 循环域 simd：每-nS store 曲线 | 逐 nS 分段线性实测节点 | 每 nS 3 点 | ≤0.7% |
| L3 | 循环域 simd：联合发射地板 | max(stepL[nL], stepS[nS], stepT[nL+nS])，stepT 小字节混合点直读 | stepT 5 点 | floor 项 |
| L4 | 循环域 simd：混合行税 | +cL·(lB/128)+cS·(sB/128)（LSQ，fp32 cL=-0.08/cS=0.44） | 13-18 混合点 | fp32 mix 5.3% |
| L5 | 循环域 simd：tiny excess/chain | dom 分键 nS delta 表（分段线性） | 每 dom 3 点 × 3 dom | tiny 9.6-21.4% |
| L6 | 循环域 simt：load 曲线 + E17i store 曲线（可加） | 曲线_nL(lB) + 排空曲线(sB) | 同 L1 数据 + E17i | loads 1.0-16.5% |
| T1 | Tier 3 兜底（simd） | 启动 + 字节/带宽，分方向 max | 基线继承 | 基线口径 |
| T2 | Tier 3 兜底（simt） | warp 指令数/速率 | 基线继承 | 基线口径 |

> 注：LF1/LF2/LF6 的"拟合"是节点直读（曲线=实测数据点连成分段线性），
> 因此样本内 MAPE 为 0 量级；LF3/LF4/LF5 为直读/LSQ 结构项。泛化
> 误差由 §三 的整核 gate 承载。循环域拟合的已知限制见 §2.2(d)
> （grid 条件差 3.2×、混合 adder 的 P 维未标、simt store 排空沿用
> E17i）。
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
