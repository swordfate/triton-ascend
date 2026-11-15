# 连续访存标定 v2 — 公式与标定手册（Ascend 950PR）

> 适用范围：Route CostModel 中 **stride == 1 的连续 load/store**（ContiguousShort /
> ContiguousWide）的全部整段定价。间接访存（indirect）、离散（gather/scatter）、
> strided 访存不在本文件范围内，仍走原有曲线/解析式。
>
> 实现位置：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`
> 的 `mapContiguousV2Completion()`；参数结构与解析在
> `ContiguousLdStV2CostModel.h` 与 `SimdSimtCostModel.cpp` 的
> `contiguous_ldst_v2` 读取块。标定数据与探针脚本在 `explore_contig/`。

---

## 0. 总原则

1. **全覆盖**：所有连续访存（直线段与循环体、simd 与 simt、任意 K 与体形状）
   都由 v2 认领（claim + layout 覆盖），legacy 的 flat-group / store-group /
   pipelined 曲线不再对连续访存定价。
2. **两级结构**：实测直查表（calF2 direct rows，几何命中时使用）→
   通用逐算子解析式（兜底公式，任何几何都成立）。除兜底解析式外不使用
   kaixin 的 legacy 连续访存标定。
3. **单位语义**：钩子加入 `resources.load/store` 的价格是 **每程序每次迭代** 的
   周期数；stage 估计器 (`estimateStageBase`) 自己乘 `stage.iterationCount`
   与 wave 算术。直线段（iterationCount == 1）即为整段每程序价格。
4. **公式从简**：新增规律一律采用线性组合、常数分档或 `max()` 组合，
   不引入新的指数或幂次；现有两个波衰减指数（-0.05 / -0.32）保留，
   但不再新增同类项。
4. **测量口径**：全部常数来自 `torch_npu.profiler` 的 kernel Duration
   （profiler 绝对时长/中位数，ABAB 对齐），与守护用例同口径。

---

## 1. 公式 A — 直线段（straight-line stage，`hasLoop == false`）

同一 (store?, 元素数, dtype, masked) 的连续访问组成一组，组内 N 条背靠背实例：

```text
T(N)   = first + (N-1) * marginal              (N <= Q, 队列深度重叠)
T(N)   = first + (Q-1)*marginal + (N-Q) * c_transfer   (N >  Q)
```

* 按 direction/dtype/bytes/residency 单元查表
  （`contiguous_ldst_v2.group_a`，key = `<load|store>/<dtype>/<bucket>/<warm|cold>`）。
* bytes bucket：{32, 64, 128, 256, 512, 1024, 4096, 16384} 中取 log2 最近。
* residency：整段 launch footprint ≤ `l2_cache_bytes` 记 warm，否则 cold。
* P 轴（核内重叠）：拟合值乘 `pAxisFactor(kind, engine, residency, P)`，
  其中 P = `sourceLogicalProgramCountHint`；表来自 E13 P-sweep 并以
  守护整核数据外推至 P=256/1024（`contiguous_ldst_v2.p_axis`，
  log2-P 线性插值，端点截断）。
* **单条访问（N==1）** 与 **未拟合单元**：使用兜底解析式（§4），不查表，
  避免 first 项把发射开销重复计一次。解析式的 **load 侧**乘
  `waves^-0.32`（守护 A 组拟合：copy/add 直线段的 load 流随 wave 数
  衰减；store 侧随 wave 数微增，不衰减）。

## 2. 公式 B — 循环体（`hasLoop == true`）

循环体中的全部连续访问按**体形状 (nL, nS)** 与主导访问 chunk 字节
`domBytes` 定价，一次性认领整段：

### 2.1 第一级：calF2 实测直查表（`contiguous_ldst_v2.direct`）

key = `n<loads>_s<stores>_b<chunkBytes>_k<K>_g<grid>`，
value = **整核 per-program 等效纳秒**（实测 kernel Duration 中位数 ÷ waves）。

查表规则：

1. 形状 (nL, nS) **完全一致** 的行参与；
2. 打分按 **同波共存**（co-residency）：`score = 3 * |log2(domBytes/rowB)|
   + 0.5 * |log2(clamp(P,1,56) / clamp(rowG,1,56))|
   + 0.05 * |log2(P / rowG)|`，取最近行；`score > 3.5`（chunk 偏差 > 2×）
   不认领第一级。每程序价格由同波共存程序数决定，与总 wave 数无关；
3. 对选中几何，取其全部 K 行拟合 `T(K) = A + K*m`：
   * `m` = 所有 K 点对的斜率中位数；
   * `A` = 中心 K 点残差（`T(Kc) − Kc*m`，截断 ≥ 0）；
   * K > 16 且高 K 斜率 `mHi` 比 `m` 高 5% 以上时按两段（写压力拐点）处理；
4. 单 K 行时以 50/50 分割近似；
5. chunk 缩放：实际体是行几何的 `chunkScale = clamp(domBytes/rowB, 0.5, 2)` 倍时，
   迁移分量（m）乘 `chunkScale`，启动分量（A）不变（算子条数相同）；
6. **截距分摊**：A 里含与 route model 边界级（dispatch/boundary stage）
   重复的开销，故只计入 `A * direct_fixed_share`
   （profile key `contiguous_ldst_v2.direct_fixed_share`，守护整核误差
   标定为 **0.05** —— 截距几乎全部与 route 边界级重叠）；

每迭代价格：

```text
perIter = (A * share + K * m(chunkScale)) / K * waves^-0.125
                                                          [周期, 1 GHz]
resources.load += perIter
```

`waves = max(1, P/56)`；循环体的 wave 衰减指数 -0.125 由守护
B_add_g224 标定（多 wave 循环体的每程序价格近似平坦，指数取小值）。

### 2.1a 第一级 a：calF2 微小 store 循环体（`tiny_direct`，Tier 1a）

规律（calF2_tinystore + 100-case 实测）：循环体的 **store 全部 ≤128B、
主导 chunk ≥512B** 时，微小依赖 IO 在 DMA 流水填满前逐迭代串行——
K=4 的实测整核时长远高于 K≥16 的线性趋势（约 2 倍），而 K≥16 后
边际增量平坦（~117 ns/iter）。大数据 store 的 direct 行（m≈109 ns/iter）
完全错过该效应，reduce_sum_epilogue 族因此欠预测 24.7%。

修法（线性形式，无指数）：

* 表：`contiguous_ldst_v2.tiny_direct`，键与 `direct` 完全一致
  （`n<loads>_s<stores>_b<chunk>_k<K>_g<grid>`），值 = 整核 per-program ns。
* 选择条件：循环体 store 均 ≤128B 且 domBytes ≥512B 且 K ≤ 8。
* 每迭代价格（线性组合）：

```text
excess  = max(0, 实测行值 − (A + K×m))        [流水填充的真实逐程序时间]
perIter = (A×share + K×m + excess×chainShare) / K × wavesDecay
```

  A、m 来自插值行的线性拟合（斜率中位数 + 中心 K 残差，同 §2.1）；
  `chainShare` = profile 键 `tiny_store_chain_share`（标定 **0.636**）——
  excess 的其余部分与 route 边界级重叠。

标定数据：n2_s1_b4096 × K{4,16,64} × g{8,56,224}（g224 缺 K64：该形状
探针触发编译器/设备故障 507035）。K4/g8 取 100-case 实机值 2772ns
（与守护同协议，最可信）。

验证：pytest_reduce_sum_epilogue_f32_8x8x512 预测 2086→2772ns
（−24.75% → +0.0%）；同族 mul_reduce / gather_after_reduce /
scalar_reduce_store 不受影响（store/dominant 条件不匹配）。

### 2.2 第二级（兜底）：逐算子解析式

任何几何（形状不在表内、chunk 太偏、表为空）都落到：

```text
simd:  perLoad  = loadOperationSetupCycles  + bytes / loadBytesPerCycle
       perStore = storeOperationSetupCycles + bytes / storeBytesPerCycle
simt:  perLoad  = ceil(elems/32) / loadWarpInstructionsPerCycle
       perStore = ceil(elems/32) / storeWarpInstructionsPerCycle
resources.load  += Σ perLoad * instances      （每次迭代）
resources.store += Σ perStore * instances
```

load 与 store 分别记入各自资源；流水线 stage 估计器取
`max(loadService, storeService, …)` 建模双流重叠，串行 stage 取求和。

### 2.3 覆盖规则（全量认领）

* 不再按 `count>1`、体形状、K 大小、`synchronizationCount` 拒绝——
  连续访问一律由 v2 认领并做 layout 覆盖
  （loadBytes/storeBytes/Operations/WarpInstructions/Shapes），
  上层曲线因此看不到任何连续流。
* mask 部分写（`activeElementRatio < 1` 的 store）除外——它属于 RMW
  完成路径，不是纯连续写。
* `hasStaticLayout` 不成立 / 地址依赖 loaded index / 非仿射地址的访问
  是间接/离散范畴，不在此认领。

## 3. simt 依赖写惩罚（S penalty）

simt 下 load 循环体中交错 dependent store 会使 load 流串行化。
profile 保留 `contiguous_ldst_v2.s_penalty`（key = dtype）作为文档化字段；
calF2 直查行的 store 体本身已含该效应，公式 B 不再额外乘惩罚项。

## 4. 兜底解析式（泛化的，保留）

直线段与循环体共用（§1 未命中 / §2.2）。常数来自 legacy profile 的
物理速率字段（`loadOperationSetupCycles`、`loadBytesPerCycle`、
`storeWarpInstructionsPerCycle` 等）——这是保留的通用兜底公式；
除此之外的 kaixin legacy 连续访存曲线（flat-group、store-group、
cross-iteration pipelined、masked-copy）不再参与连续访存定价。

## 5. route.json 诊断输出

* `candidate_elapsed_costs`：{all_simd, all_simt_only, mixed_simd_simt} 整核预测（ns）。
* `candidate_elapsed_costs_by_factor`（本 PR 新增）：
  `all_simd_f1` + 每个合法纯 simt superblock 因子的
  `all_simt_only_f{1,2,4,...}` 整核预测——守护用例强制 F2/F4 运行时
  直接与模型同因子预测对比，无需重推固定开销。

## 6. 标定协议（explore_contig/）

| 探针 | 内容 | 产出的表 |
|---|---|---|
| calD (`cal_d_grid.py`) | 直线段 U-sweep × bytes × grid × W | `group_a` 的 first/marginal/c_transfer/Q |
| calE (`cal_e_loopgrid.py`) | 循环体 indpt/indpt2/chain/store × K × grid | `loop_b` 的 P/m（交叉验证） |
| calF2 (`cal_f2_direct.py`) | 整核绝对时长：4 体形状（与守护 kernel 同体：lcopy/ladd/lload/lstore） × chunk{1K,4K,16K,64K}B × K{4,16,64} × grid{8,56,224}，simd(w4)+simt(w1)；**每几何独立 profiler 会话**（harness 同款 Level1 配置，op_statistic 20 次启动均值，会话外 5 次预热） | `direct` 直查行（本次最终采用） |
| calG (`cal_g_bandwidth.py`) | 聚合带宽曲线（per-program rate vs P） | 物理上限校验（~83 B/ns/56 程序） |
| calF2-t (`cal_f2_tinystore3.py`) | 微小 store 循环体：n2_s1（4KB tile + 8B bias + 8B store）× K{4,16,64} × g{8,56,224}，每几何独立会话 ×2 | `tiny_direct` 行 + `tiny_store_chain_share` |

拟合脚本：`fit_v2_grid.py`（group_a/loop_b）、`fit_direct2.py`（direct 行 +
`direct_fixed_share` 注入 profile）。测量口径教训（三连坑）：
① 批内轮转多 kernel 会互相踢 L2 → 必须**每几何独立会话**；
② kernel 体形状微差（尾部 store、`x+0` vs `x+1`）带来 ~10% 摆动 →
标定 kernel 必须与守护 kernel **同体**；
③ aic Level1 采集自身有 ~5-7% 开销 → 标定与守护必须**同配置**。
平台事实：simt 无法下降 64KB-chunk 体（per-thread stack 1152B 上限）。

## 7. 已知边界

* chain（loop-carried）体在表内以 (1,0) 形状锚定；`LoopCarriedRecurrence`
  stage 的依赖因子语义与独立循环不同，守护表覆盖 K∈{4,16,64} 验证。
* footprint 超过 L2 的 warm/cold 差异由行值本身承载（探针按真实 footprint
  测量）；未细分 residency 维度。
* `tiny_direct` 当前只覆盖 n2_s1_b4096 形状；n2_s0（masked store 被排除
   后）与其它 chunk 的微小 store 体未标定，落在 §2.1 的 direct 行。
* `direct_fixed_share`（0.05）与两条 wave 衰减指数（循环体 -0.125、
  直线段 load 侧 -0.32）是本标定的三个整核校准常数，随守护用例
  整核数据更新。
* chunk 维度当前覆盖 {256B..64KB}；f16 2KB-chunk 行缺失（守护
  B_copy_f16 +32% 即源于最近行 + chunkScale 的近似），待补标定。
* route.json 新增 `candidate_elapsed_costs_by_factor`：含合法纯 simt
  各 superblock 因子的整核预测；含 runtime-loop stage 的 kernel 按
  wave 比 `ceil(P/f/56)/ceil(P/56)` 缩放（实测 grid-224 F2=0.53×F1），
  单波内因子为 no-op。

* **跨 stage DMA 重叠物理规律**（CAModel 仿真确认，`msprof op simulator`）：
  - 独立 load/store（无循环依赖）：跨 stage 完全流水。vector_add 仿真
    显示 2 VLD/cycle 并行发射，store 与 load 尾部重叠。
  - loop-carried dependency：逐 iteration 串行。masked_softmax 仿真显示
    每轮 VLD 等前一轮 VST 完成才发射。
  - v2 calF2 直查行按独立 load 模式标定，对独立场景正确。
  - reduction-stage case（masked_softmax/softmax）的偏差来自
    kaixin rowwise_reduction 曲线对 compute 时间低估，
    非连续访存标定问题。
* **sdirect（直线段整核直查）实验记录**：calF3 已按守护同体测得
  93×2 行直线段整核数据（profile 键 `contiguous_ldst_v2.sdirect`，
  已入库未启用）。stage 级充值实验失败并回滚：whole-launch 行与
  估计器的 wave 算术不可组合（每程序价格在 latency-bound 单波与
  BW-bound 多波 regime 间相差约 10×，grid-1024 充值实测 10× 高估）。
  正确集成需要在候选级（factor 已知处）按 wave 比
  `ceil(P/f/56)` 重缩放，并解决 copy 类 kernel 的 load/store 双
  stage 拆分归账问题——待后续 PR。
