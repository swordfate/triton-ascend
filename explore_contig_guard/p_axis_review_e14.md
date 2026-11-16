# E14：P 轴曲线分解——T(P) 的增长到底来自哪里（回应 P 轴质疑）

日期：2026-11-16；卡 2；simd / warm / BLOCK=1024 f32（4KB/程序）/ 每点 12-16 次
发射取中位数（设备侧 Duration）。脚本：`explore_contig/e14_p_decompose.py`。

## 1. 五个变体（逐个剥离）

| 变体 | 内容 | 剥离了什么 |
|---|---|---|
| ctl1 | 标量 store，grid=1 | 参考地板 |
| ctlP | 标量 store，grid=P | **纯 dispatch/prologue**（无 load、无 4KB、无 sum） |
| sumonly | x=full(BLOCK)，sum，写 1 元素 | 归约链（无 load、无 4KB burst） |
| ldmax | load 4KB，写 max(x) | load + 便宜树消费（无 sum-add 链） |
| loadsum | load 4KB，sum，写 1 元素 | = E13 探针原型 |

## 2. 实测（ns，设备侧）

```
P      ctlP   sumonly   ldmax   loadsum | loadsum−ctlP  ldmax−ctlP  sumonly−ctlP
1      593*     752      876      795   |     202           —            —
2      829      971     1290     1112   |     283          461          142
4     1034     1492     1313     1286   |     252          279           —
8     1359     1554     1716     1666   |     307          357          195
16    2178     2319     2531     2442   |     264          353          141
32    3929     3928     4054     4064   |     135          126            0
56    3873     4002     4134     4100   |     226          261          129
112   4873     5081     5183     5124   |     251          310          208
224   6031     6185     6452     6334   |     303          421          154
* ctl1（grid=1）= 585，与 ctlP(1) 同域
```

AIV count 实测 = 56（`driver.active.utils` 与 route 的
`physical_vector_core_count_hint` 一致，评审实验 5 ✓）。

## 3. 结论（回答评审 4 问）

### Q1：T(P) 增长来自什么？

**设备侧每 block 的 dispatch/prologue/调度墙钟**。证据链：

1. `ctlP`（纯标量 store：无 load、无 4KB burst、无 sum）**单独复现了整条
   P 曲线**：593→829→1034→1359→2178→3928→3873→4873→6031，与 loadsum 曲线
   形状逐点吻合（P=32 时 ctlP=3929 vs loadsum=4064，占 97%）；
2. 剥掉 ctlP 后，**内存/归约的增量是平的**：loadsum−ctlP ≈ 200-300ns
   （P=1..224 全程），ldmax−ctlP ≈ 130-420，sumonly−ctlP ≈ 0-208——
   都不随 P 系统性增长；
3. warm≡cold（E13）进一步排除带宽竞争；
4. `sumonly−ctlP`（归约链）与 `ldmax−ctlP`（load 链）量级相当（100-350），
   且互相独立——归约链不是主驱动。

因此评审 4.2 的推断**成立**：P 曲线的主导成分是每 block 的
dispatch/prologue（实测折算 ≈ 60-100ns/block 的调度墙钟，P=16→32 处有
超线性台阶 2178→3929，疑似调度器占用台阶）。

### Q2：P≤56 时 p_axis 还有"每程序内存单价"的意义吗？

**没有。** p_axis(P) = [T(P)/P]÷T(1) 里，T(P) 的分子被 dispatch 墙钟主导，
除以 P 之后得到的是"平摊到每个 program 的调度+内存混合墙钟"，不是内存
单价。真正的每程序内存/归约单价是**平的**：

```
load + sum 消耗：≈ 200-300 ns（P 无关）
load + max 消耗：≈ 130-420 ns（P 无关）
归约链 alone：  ≈ 0-200 ns（P 无关，噪声级）
```

若需要"每波墙钟"，正确量是 **T(P) 本身**（即 ctlP 形状的曲线），
而不是 T(P)/P。

### Q3：与 boundary_residual 是否重复计费？

**结构上存在同源叠加**。候选层：

```
预测 = Σ_stage( group_a(grid=1) × pAxisFactor(P) ) + routeFixedSystemCycles
```

其中 `routeFixedSystemCycles` 含 `boundary_residual(P)`（P=56 ≈ 1705ns）
+ 全局 load 边界 + 参数 setup。而 `pAxisFactor(P)` 的分子 T(P) 里同样含
dispatch 墙钟（E14 ctlP 证明其主导性）——**同一物理现象（按 block 数
增长的调度墙钟）在两层各建模了一次**。量级上 boundary_residual 只覆盖
了其中 ~1705/3280（E14 ctlP 的 P56−P1 差），剩余部分经 p_axis 折进
stage 价。

**修正建议（归因到层）**：
- dispatch/prologue 是"每个候选一次"的量 → 归 **boundary_residual(P)**
  一层，量级按 E14 ctlP 曲线重标（当前 1705@56 低估了 ~3280 的实差）；
- p_axis 表从"墙钟占用比"改为**剔除 dispatch 后的纯内存/归约占用比**
  ——按 E14 数据，剔除后该比值在 P≤56 基本平坦（≈1），P>56 段才由
  聚合带宽接管 → 即 **P≤56 时 pAxisFactor 应恒为 ~1**（stage 价不再乘
  占用衰减），高波段的衰减保留；
- 或者等价地：保留现表，但把 `boundary_residual(P)` 从候选层去掉、
  并把 group_a 改为按 T(P) 整核拟合（两者只留一个）。前者改动小、
  与 E14 数据一致，推荐前者。

### Q4：量纲检查与组合修正

`group_a(grid=1)[ns] × pAxisFactor[无量纲] × Q[个]`：
- group_a(grid=1) 的语义 = "1 个 program 独跑时的整核价"（含 1 份
  dispatch + 1 份内存链）；
- × pAxisFactor(P)（墙钟平摊比）后 = **T(P)/P**——"平摊到每 program 的
  墙钟"，已经不是内存单价；
- × Q：P≤56 时 Q=1 → stage = T(P)/P，而 stage 语义应是
  **max(T(P), Σper-program)** ≈ T(P)——差 P 倍的语义错位；
  P>56 时 Q=ceil(P/56)，stage = T(P)/P × Q ≈ T(P)/56·Q ——又回到
  平摊墙钟的波数倍。
- **修正**：Stage 价在 Q=1 段直接取"整核行价"（即 group_a 行按整核
  测量、不再除/乘 P），P 轴只保留 >56 段的波间重叠衰减；dispatch 统一
  归 boundary_residual。这也正是 scatter/compaction 修复中"按整核行
  注入低网格行"能够对齐实测的原因。

## 4. 与既有证据的一致性

- CAModel 逐核时间线（e358fa9ac5）：grid=112 时**一条 lane 背靠背执行
  2 个 program**——即块级串行调度，与"dispatch 墙钟随 P 线性增长"一致；
- warm≡cold：dispatch 是纯控制流，不读数据 → 与内存状态无关 ✓；
- simd/simt 拐点都在 56：两种引擎共享同一套 block 调度器 ✓；
- scatter/compaction（hint=2）的低网格修复：hint 远低于 g8，行表中
  g8 行的"8 程序调度墙钟"被错配到 2 程序的 stage → 低网格行
  （g{1,2,4}）吸收 dispatch 后回归实测 ✓。
